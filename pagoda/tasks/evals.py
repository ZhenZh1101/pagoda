"""Benchmarks of plan section 12. All test data is local (raw/), already used for decontamination.

  arc_easy / arc_challenge / mmlu   chat-format multiple choice, compare logits of " A".." E" after "Answer:"
  hellaswag                         10-shot completion log-likelihood, acc_norm (length-normalised)
  gsm8k_tool                        greedy chat generation with the Python tool, 'Answer:' vs gold (tol 1e-6)
  humaneval                         pass@1, greedy, tests run in the sandbox (optional)
"""

import random
import re

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq

from pagoda.common import RAW_DIR
from pagoda.data.render import LETTERS, mc_prompt
from pagoda.engine import chat_reply
from pagoda.sandbox import extract_answer, numbers_equal, parse_number, run_python


def _subset(items, limit, seed=0):
    if limit and len(items) > limit:
        rng = random.Random(seed)
        items = rng.sample(items, limit)
    return items


# ------------------------------------------------------------- multiple choice
def load_mc(task):
    if task in ("arc_easy", "arc_challenge"):
        sub = "ARC-Easy" if task == "arc_easy" else "ARC-Challenge"
        rows = pq.read_table(RAW_DIR / f"arc/{sub}/test-00000-of-00001.parquet").to_pylist()
        return [{"question": r["question"], "choices": r["choices"]["text"],
                 "answer": r["choices"]["label"].index(r["answerKey"])} for r in rows
                if r["answerKey"] in r["choices"]["label"]]
    if task == "mmlu":
        rows = pq.read_table(RAW_DIR / "mmlu/all/test-00000-of-00001.parquet").to_pylist()
        return [{"question": r["question"], "choices": r["choices"], "answer": int(r["answer"]),
                 "subject": r["subject"]} for r in rows]
    raise ValueError(task)


def eval_mc(model, tok, task, limit=0, batch=16, log=None):
    items = _subset(load_mc(task), limit)
    letter_ids = [tok.encode(f" {L}")[0] for L in LETTERS]
    prefix_tail = tok.encode("Answer:")
    correct = 0
    hist = {}
    for i in range(0, len(items), batch):
        chunk = items[i:i + batch]
        seqs = []
        for it in chunk:
            ids = tok.render_for_completion([{"role": "user", "content": mc_prompt(it["question"], it["choices"])}])
            seqs.append((ids + prefix_tail)[-model.cfg.seq_len:])
        T = max(len(s) for s in seqs)
        arr = np.zeros((len(seqs), T), dtype=np.int32)
        for j, s in enumerate(seqs):
            arr[j, :len(s)] = s
        logits = model(mx.array(arr))
        last = mx.array([len(s) - 1 for s in seqs])
        lg = logits[mx.arange(len(seqs)), last]
        mx.eval(lg)
        lg = np.array(lg)
        for j, it in enumerate(chunk):
            n = len(it["choices"])
            pred = int(np.argmax(lg[j, letter_ids[:n]]))
            correct += pred == it["answer"]
            hist[LETTERS[pred]] = hist.get(LETTERS[pred], 0) + 1
    # what a model that always answers its favourite letter would score: tells skill from a letter prior
    top = max(hist, key=hist.get) if hist else "A"
    always_top = sum(it["answer"] == LETTERS.index(top) for it in items) / max(len(items), 1)
    return {"accuracy": correct / max(len(items), 1), "n": len(items), "predicted_letters": hist,
            "always_most_predicted_accuracy": always_top}


# ------------------------------------------------------------- hellaswag
def _hs_pre(text):
    text = text.strip().replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    return text.replace("  ", " ")


def load_hellaswag(split="validation"):
    rows = pq.read_table(RAW_DIR / f"hellaswag/data/{split}-00000-of-00001.parquet").to_pylist()
    out = []
    for r in rows:
        ctx = r["ctx_a"] + " " + r["ctx_b"].capitalize()
        out.append({"query": _hs_pre(r["activity_label"] + ": " + ctx),
                    "endings": [_hs_pre(e) for e in r["endings"]], "label": int(r["label"]) if r["label"] else -1})
    return out


def eval_hellaswag(model, tok, limit=0, shots=10, log=None):
    items = _subset(load_hellaswag("validation"), limit)
    train = load_hellaswag("train")
    max_len = model.cfg.seq_len
    correct = 0
    for idx, it in enumerate(items):
        rng = random.Random(1234 + idx)
        ex = rng.sample(train, shots) if shots else []
        ctx_ids = None
        for k in range(len(ex), -1, -1):  # drop shots until the longest choice fits
            prefix = "".join(f"{e['query']} {e['endings'][e['label']]}\n\n" for e in ex[:k])
            ctx_ids = tok.encode(prefix + it["query"], bos=True)
            longest = max(len(tok.encode(" " + e)) for e in it["endings"])
            if len(ctx_ids) + longest <= max_len + 1:
                break
        cont = [tok.encode(" " + e) for e in it["endings"]]
        seqs = [ctx_ids + c for c in cont]
        T = max(len(s) for s in seqs)
        arr = np.zeros((len(seqs), T), dtype=np.int32)
        for j, s in enumerate(seqs):
            arr[j, :len(s)] = s
        x = mx.array(arr)
        logits = model(x[:, :-1])
        logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        tlp = mx.take_along_axis(logp, x[:, 1:, None], axis=-1)[..., 0]
        mx.eval(tlp)
        tlp = np.array(tlp)
        scores = []
        for j, c in enumerate(cont):
            s = len(ctx_ids)
            ll = tlp[j, s - 1: s - 1 + len(c)].sum()
            scores.append(ll / max(len(it["endings"][j]), 1))  # acc_norm: per-character normalisation
        correct += int(np.argmax(scores)) == it["label"]
    return {"acc_norm": correct / max(len(items), 1), "n": len(items), "shots": shots}


# ------------------------------------------------------------- gsm8k with tool
def eval_gsm8k_tool(model, tok, limit=0, max_new_tokens=512, log=None, samples_out=None):
    rows = pq.read_table(RAW_DIR / "gsm8k/main/test-00000-of-00001.parquet").to_pylist()
    items = _subset(rows, limit)
    correct, closed, used_tool = 0, 0, 0
    for i, r in enumerate(items):
        gold = parse_number(r["answer"].split("####")[-1])
        prompt = tok.render_for_completion([{"role": "user", "content": r["question"]}])
        out = chat_reply(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0)
        pred = extract_answer(out["text"])
        ok = numbers_equal(pred, gold)
        correct += ok
        closed += out["finished"]
        used_tool += bool(out["tool_calls"])
        if samples_out is not None and i < 50:
            samples_out.append({"question": r["question"], "gold": gold, "pred": pred, "correct": ok,
                                "completion": out["text"]})
        if log and (i + 1) % 100 == 0:
            log(f"  gsm8k_tool {i + 1}/{len(items)} acc {correct / (i + 1):.3f}")
    n = max(len(items), 1)
    return {"accuracy": correct / n, "n": len(items), "finished_frac": closed / n, "tool_use_frac": used_tool / n}


# ------------------------------------------------------------- humaneval
def _extract_code(text):
    m = re.findall(r"```(?:python)?\n(.*?)```", text, re.S)
    return m[0] if m else text


def eval_humaneval(model, tok, limit=0, max_new_tokens=512, log=None):
    rows = pq.read_table(RAW_DIR / "humaneval/openai_humaneval/test-00000-of-00001.parquet").to_pylist()
    items = _subset(rows, limit)
    passed = 0
    for r in items:
        msg = f"Complete the following Python function.\n\n```python\n{r['prompt']}```"
        prompt = tok.render_for_completion([{"role": "user", "content": msg}])
        out = chat_reply(model, tok, prompt, max_new_tokens=max_new_tokens, temperature=0.0, max_tool_calls=0)
        code = _extract_code(out["text"])
        program = code if f"def {r['entry_point']}" in code else r["prompt"] + code
        program += "\n\n" + r["test"] + f"\ncheck({r['entry_point']})\n"
        passed += run_python(program, timeout=5.0).ok
    return {"pass@1": passed / max(len(items), 1), "n": len(items)}


TASKS = {
    "arc_easy": lambda m, t, **kw: eval_mc(m, t, "arc_easy", **kw),
    "arc_challenge": lambda m, t, **kw: eval_mc(m, t, "arc_challenge", **kw),
    "mmlu": lambda m, t, **kw: eval_mc(m, t, "mmlu", **kw),
    "hellaswag": lambda m, t, **kw: eval_hellaswag(m, t, **kw),
    "gsm8k_tool": lambda m, t, **kw: eval_gsm8k_tool(m, t, **kw),
    "humaneval": lambda m, t, **kw: eval_humaneval(m, t, **kw),
}
