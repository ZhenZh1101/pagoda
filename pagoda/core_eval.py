"""CORE metric (DCLM / nanochat eval bundle) for base checkpoints, plan sections 10 and 12.

22 in-context tasks from raw/eval_bundle/core.yaml, three task types:
  multiple_choice  same context, different continuations -> lowest mean loss over the choice tokens
  schema           different contexts, same continuation  -> lowest mean loss over the continuation
  language_modeling                                       -> greedy decode must reproduce the continuation
Accuracy is centered by each task's random baseline; CORE = mean centered accuracy.
Port of nanochat's core_eval.py (same few-shot sampling and prompt templates).
"""

import csv
import json
import random

import mlx.core as mx
import numpy as np

from pagoda.common import RAW_DIR, load_yaml

BUNDLE = RAW_DIR / "eval_bundle"


def _fewshot(data, idx, k):
    if k == 0:
        return []
    rng = random.Random(1234 + idx)
    pool = [i for i in range(len(data)) if i != idx]
    return [data[i] for i in rng.sample(pool, k)]


def render(item, task, shots):
    d = task.get("continuation_delimiter", " ")
    t = task["icl_task_type"]
    if t == "multiple_choice":
        prefix = "".join(f"{e['query']}{d}{e['choices'][e['gold']]}\n\n" for e in shots)
        return [f"{prefix}{item['query']}{d}{c}" for c in item["choices"]]
    if t == "schema":
        prefix = "".join(f"{e['context_options'][e['gold']]}{d}{e['continuation']}\n\n" for e in shots)
        return [f"{prefix}{c}{d}{item['continuation']}" for c in item["context_options"]]
    if t == "language_modeling":
        prefix = "".join(f"{e['context'].strip()}{d}{e['continuation']}\n\n" for e in shots)
        without = f"{prefix}{item['context'].strip()}{d}"
        return [without.strip(), without + item["continuation"]]
    raise ValueError(t)


def _common_prefix(seqs):
    n = min(len(s) for s in seqs)
    for i in range(n):
        if any(s[i] != seqs[0][i] for s in seqs):
            return i
    return n


def _common_suffix(seqs):
    n = min(len(s) for s in seqs)
    for i in range(1, n + 1):
        if any(s[-i] != seqs[0][-i] for s in seqs):
            return i - 1
    return n


def _forward(model, seqs, max_len):
    """Crop from the left to max_len, right-pad, return (logprobs at targets, argmax, offsets)."""
    crop = [max(0, len(s) - max_len) for s in seqs]
    seqs = [s[c:] for s, c in zip(seqs, crop)]
    T = max(len(s) for s in seqs)
    arr = np.zeros((len(seqs), T), dtype=np.int32)
    for i, s in enumerate(seqs):
        arr[i, :len(s)] = s
    x = mx.array(arr)
    logits = model(x[:, :-1])
    logp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
    tgt_lp = mx.take_along_axis(logp, x[:, 1:, None], axis=-1)[..., 0]
    pred = mx.argmax(logits, axis=-1)
    mx.eval(tgt_lp, pred)
    return np.array(tgt_lp), np.array(pred), crop, arr


def eval_item(model, tok, item, task, shots, max_len):
    prompts = render(item, task, shots)
    seqs = [tok.encode(p, bos=True) for p in prompts]
    t = task["icl_task_type"]
    if t == "language_modeling":
        ctx, full = seqs
        start, end = len(ctx), len(full)
        lp, pred, crop, arr = _forward(model, [full], max_len)
        s, e = start - crop[0], end - crop[0]
        # target position j predicts token j+1
        return bool(np.all(pred[0, s - 1:e - 1] == arr[0, s:e]))
    if t == "multiple_choice":
        p = _common_prefix(seqs)
        spans = [(p, len(s)) for s in seqs]
    else:  # schema
        q = _common_suffix(seqs)
        spans = [(len(s) - q, len(s)) for s in seqs]
    lp, _, crop, _ = _forward(model, seqs, max_len)
    losses = []
    for i, (s, e) in enumerate(spans):
        s, e = s - crop[i], e - crop[i]
        losses.append(-lp[i, max(s - 1, 0):e - 1].mean() if e - 1 > max(s - 1, 0) else float("inf"))
    return int(np.argmin(losses)) == item["gold"]


def load_tasks():
    cfg = load_yaml(BUNDLE / "core.yaml")
    baselines = {}
    with open(BUNDLE / "eval_meta_data.csv") as f:
        for row in csv.DictReader(f):
            baselines[row["Eval Task"]] = float(row["Random baseline"])
    return cfg["icl_tasks"], baselines


def evaluate_core(model, tok, max_per_task=500, max_len=None, tasks_filter=None):
    max_len = max_len or model.cfg.seq_len
    tasks, baselines = load_tasks()
    results = {}
    for task in tasks:
        label = task["label"]
        if tasks_filter and label not in tasks_filter:
            continue
        with open(BUNDLE / "eval_data" / task["dataset_uri"]) as f:
            data = [json.loads(l) for l in f if l.strip()]
        k = task["num_fewshot"][0]
        order = list(range(len(data)))
        random.Random(1337).shuffle(order)
        if max_per_task and max_per_task > 0:
            order = order[:max_per_task]
        correct = sum(eval_item(model, tok, data[i], task, _fewshot(data, i, k), max_len) for i in order)
        acc = correct / max(len(order), 1)
        rb = baselines.get(label, 0.0) / 100
        centered = (acc - rb) / (1 - rb)
        results[label] = {"accuracy": acc, "centered": centered, "n": len(order)}
    core = float(np.mean([r["centered"] for r in results.values()])) if results else 0.0
    return {"core": core, "tasks": results}
