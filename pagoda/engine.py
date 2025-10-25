"""Generation with a KV cache and the Python tool loop (plan section 11).

Eval loop: generate until <|python_end|> -> run the code in the sandbox ->
append <|output_start|>result<|output_end|> -> keep generating until
<|assistant_end|> or the token limit. At most 2 tool calls per reply.
"""

import mlx.core as mx

from pagoda.model import make_caches
from pagoda.sandbox import run_python


def _sample(logits, temperature, top_p, top_k):
    if temperature <= 0:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    if top_k:
        kth = mx.sort(logits, axis=-1)[..., -top_k][..., None]
        logits = mx.where(logits < kth, -float("inf"), logits)
    if top_p < 1.0:
        probs = mx.softmax(logits, axis=-1)
        order = mx.argsort(-probs, axis=-1)
        sorted_p = mx.take_along_axis(probs, order, axis=-1)
        cum = mx.cumsum(sorted_p, axis=-1)
        keep_sorted = (cum - sorted_p) < top_p
        keep = mx.zeros_like(keep_sorted)
        keep = mx.put_along_axis(keep, order, keep_sorted, axis=-1)
        logits = mx.where(keep, logits, -float("inf"))
    return mx.random.categorical(logits)


class Session:
    """One sequence: feeds tokens through the model with a KV cache."""

    def __init__(self, model, max_len=None):
        self.model = model
        self.caches = make_caches(model)
        self.max_len = max_len or model.cfg.seq_len
        self.length = 0
        self.last_logits = None

    def feed(self, ids):
        if not ids:
            return
        if self.length + len(ids) > self.max_len:
            raise ContextFull()
        x = mx.array([ids], dtype=mx.int32)
        logits = self.model(x, self.caches)
        self.last_logits = logits[:, -1, :]
        self.length += len(ids)


class ContextFull(Exception):
    pass


def generate(model, tok, prompt_ids, max_new_tokens=128, temperature=0.0, top_p=1.0, top_k=0, stop_ids=(), seed=None):
    """Plain completion. Returns the generated ids (stop token excluded)."""
    if seed is not None:
        mx.random.seed(seed)
    sess = Session(model)
    sess.feed(list(prompt_ids))
    out = []
    stop = set(stop_ids) | {tok.bos_id}
    for _ in range(max_new_tokens):
        nxt = _sample(sess.last_logits, temperature, top_p, top_k)
        t = int(nxt.item())
        if t in stop:
            break
        out.append(t)
        try:
            sess.feed([t])
        except ContextFull:
            break
    return out


def chat_reply(model, tok, prompt_ids, max_new_tokens=512, temperature=0.0, top_p=1.0, top_k=0,
               max_tool_calls=2, executor=run_python, seed=None):
    """Assistant reply with tool execution. `prompt_ids` ends in <|assistant_start|>.

    Returns dict(text, ids, tool_calls=[(code, output)], finished).
    """
    if seed is not None:
        mx.random.seed(seed)
    sp = tok.special
    py_start, py_end = sp("<|python_start|>"), sp("<|python_end|>")
    out_start, out_end = sp("<|output_start|>"), sp("<|output_end|>")
    a_end = sp("<|assistant_end|>")

    sess = Session(model)
    sess.feed(list(prompt_ids))
    ids, calls, code_buf, in_code, finished = [], [], [], False, False
    generated = 0
    try:
        while generated < max_new_tokens:
            t = int(_sample(sess.last_logits, temperature, top_p, top_k).item())
            generated += 1
            if t in (a_end, tok.bos_id):
                finished = True
                break
            ids.append(t)
            sess.feed([t])
            if t == py_start:
                in_code, code_buf = True, []
            elif t == py_end and in_code:
                in_code = False
                code = tok.decode(code_buf)
                if len(calls) < max_tool_calls:
                    result = executor(code).output
                else:
                    result = "Error: tool call limit reached"
                calls.append((code, result))
                forced = [out_start] + tok.encode(result) + [out_end]
                ids.extend(forced)
                sess.feed(forced)
            elif in_code:
                code_buf.append(t)
    except ContextFull:
        pass
    return {"text": tok.decode(ids), "ids": ids, "tool_calls": calls, "finished": finished}


def score_continuations(model, prefix_ids, continuations):
    """Sum of log-probs of each continuation given the prefix (batched, right-padded).

    Returns list of (sum_logprob, n_tokens).
    """
    seqs = [list(prefix_ids) + list(c) for c in continuations]
    T = max(len(s) for s in seqs)
    T = min(T, model.cfg.seq_len + 1)
    rows = []
    for s in seqs:
        s = s[-T:]
        rows.append(s + [0] * (T - len(s)))
    arr = mx.array(rows, dtype=mx.int32)
    logits = model(arr[:, :-1])
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt = arr[:, 1:]
    tok_lp = mx.take_along_axis(logp, tgt[..., None], axis=-1)[..., 0]
    mx.eval(tok_lp)
    out = []
    for i, (s, c) in enumerate(zip(seqs, continuations)):
        s_len = min(len(s), T)
        n = len(c)
        lp = tok_lp[i, s_len - 1 - n: s_len - 1]
        out.append((float(lp.sum().item()), n))
    return out


_PART_RE = None


def reply_to_parts(text):
    """Decoded assistant text -> content parts, so a reply can be re-rendered with real special tokens."""
    import re
    global _PART_RE
    if _PART_RE is None:
        _PART_RE = re.compile(r"<\|python_start\|>(.*?)<\|python_end\|>|<\|output_start\|>(.*?)<\|output_end\|>", re.S)
    parts, pos = [], 0
    for m in _PART_RE.finditer(text):
        if m.start() > pos:
            parts.append({"type": "text", "text": text[pos:m.start()]})
        if m.group(1) is not None:
            parts.append({"type": "python", "text": m.group(1)})
        else:
            parts.append({"type": "python_output", "text": m.group(2)})
        pos = m.end()
    if pos < len(text):
        parts.append({"type": "text", "text": text[pos:]})
    return parts
