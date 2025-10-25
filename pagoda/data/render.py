"""Record -> token ids for every document format used in stages 1-3 (plan section 9 "格式").

  text / code     <|bos|>body                       (pretraining documents)
  tinygsm_text    question + ```python``` + output + Answer, as plain text (stage 1)
  chat            <|bos|><|user_start|>..<|user_end|><|assistant_start|>..<|assistant_end|>...
  tool_chat       chat whose assistant turn is the Python tool protocol (TinyGSM, GSM8K, OMI-1)
  cot_chat        chat whose assistant turn is a short CoT ending in 'Answer: x'
  mc_chat         user: Question + lettered choices, assistant: 'Answer: B'

Renderers return (ids, mask) or None when the record fails a token-length rule.
`system_prompt_frac` (chat) folds the Pagoda-0.1 system prompt into the first user
turn of a stable-hashed share of conversations (pagoda.identity).
"""

from pagoda.data.convert import tool_parts
from pagoda.identity import with_system_prompt

LETTERS = "ABCDE"


def mc_prompt(question, choices):
    lines = [f"Question: {question.strip()}"]
    lines += [f"{LETTERS[i]}. {str(c).strip()}" for i, c in enumerate(choices)]
    return "\n".join(lines)


def tinygsm_text(rec):
    return (f"{rec['question']}\n\n```python\n{rec['script']}\n```\n"
            f"Output: {rec['output']}\nAnswer: {rec['answer']}")


def tool_messages(rec):
    parts = tool_parts(rec["script"], rec["output"], rec["answer"],
                       analysis=rec.get("analysis", ""), closing=rec.get("closing", ""))
    return [{"role": "user", "content": rec["question"]}, {"role": "assistant", "content": parts}]


def reply_lengths_ok(tok, messages, max_reply_tokens):
    for m in messages:
        if m["role"] == "assistant" and isinstance(m["content"], str):
            if len(tok.encode(m["content"])) > max_reply_tokens:
                return False
    return True


class Renderer:
    def __init__(self, tok, fmt, spec):
        self.tok, self.fmt, self.spec = tok, fmt, spec
        self.min_tokens = spec.get("min_tokens", 50 if fmt == "text" else 1)
        self.max_tokens = spec.get("max_tokens")              # drop docs longer than this (code: 8192)
        self.max_reply = spec.get("max_reply_tokens", 512)    # chat: per assistant reply
        self.max_answer = spec.get("max_answer_tokens", 300)  # cot: solution length

    def batch(self, records):
        """List of (ids, mask) | None, aligned with records."""
        tok = self.tok
        if self.fmt in ("text", "code"):
            out = []
            for rec, ids in zip(records, tok.encode_batch([r["text"] for r in records])):
                if len(ids) < self.min_tokens or (self.max_tokens and len(ids) > self.max_tokens):
                    out.append(None)
                else:
                    out.append(([tok.bos_id] + ids, None))
            return out
        if self.fmt == "tinygsm_text":
            return [([tok.bos_id] + ids, None) for ids in tok.encode_batch([tinygsm_text(r) for r in records])]
        return [self.one(r) for r in records]

    def fit_turns(self, msgs, cap=1025):
        """Longest prefix of whole user/assistant exchanges that fits the row and the reply limit (SFT)."""
        best = None
        for k in range(2, len(msgs) + 1, 2):
            if not reply_lengths_ok(self.tok, msgs[k - 2:k], self.max_reply):
                break
            ids, mask = self.tok.render_conversation(msgs[:k])
            if len(ids) > cap:
                break
            best = (ids, mask)
        return best

    def one(self, rec):
        tok = self.tok
        if self.fmt == "chat":
            msgs = with_system_prompt(rec["messages"], self.spec.get("system_prompt_frac", 0))
            if self.spec.get("fit_turns"):
                return self.fit_turns(msgs)
            if not reply_lengths_ok(tok, msgs, self.max_reply):
                return None
            return tok.render_conversation(msgs)
        if self.fmt == "tool_chat":
            if "script" not in rec:
                return None
            return tok.render_conversation(tool_messages(rec))
        if self.fmt == "cot_chat":
            if len(tok.encode(rec["cot"])) > self.max_answer:
                return None
            return tok.render_conversation([{"role": "user", "content": rec["question"]},
                                            {"role": "assistant", "content": rec["cot"]}])
        if self.fmt == "mc_chat":
            return tok.render_conversation([{"role": "user", "content": mc_prompt(rec["question"], rec["choices"])},
                                            {"role": "assistant", "content": f"Answer: {LETTERS[rec['answer']]}"}])
        if self.fmt == "mc_explain_chat":
            letter, choice = LETTERS[rec["answer"]], str(rec["choices"][rec["answer"]]).strip().rstrip(".")
            reply = f"Answer: {letter}\nThe correct choice is {letter}, \"{choice}\"."
            return tok.render_conversation([{"role": "user", "content": mc_prompt(rec["question"], rec["choices"])},
                                            {"role": "assistant", "content": reply}])
        raise ValueError(f"unknown format {self.fmt}")


def split_long(ids, cap, bos_id, min_tail=16):
    """Split a doc longer than `cap` (= seq_len + 1) into BOS-prefixed chunks that each fit a row."""
    if len(ids) <= cap:
        return [ids]
    chunks = [ids[:cap]]
    body = ids[cap:]
    step = cap - 1
    for i in range(0, len(body), step):
        piece = body[i:i + step]
        if len(piece) >= min_tail:
            chunks.append([bos_id] + piece)
    return chunks
