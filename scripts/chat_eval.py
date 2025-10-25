"""Benchmarks (plan section 12).

    python -m scripts.chat_eval --run d768_5d --source sft \\
        --tasks arc_easy,arc_challenge,mmlu,gsm8k_tool,hellaswag
    python -m scripts.chat_eval --run d768_5d --source base --tasks core      # CORE on the base model
    python -m scripts.chat_eval ... --limit 200                               # quick look
    python -m scripts.chat_eval ... --task-limits hellaswag=2000,gsm8k_tool=400  # per-task caps
    python -m scripts.chat_eval --source base --tasks core --core-tasks arc_easy,arc_challenge --core-max-per-task 0  # cloze ARC, full

LM-completion tasks (core, hellaswag) belong to the base model, chat-format tasks
(arc_*, mmlu, gsm8k_tool, humaneval) to the SFT model; runs/after_pretrain.sh follows that split.

Results: logs/<run>/eval_<source>.json (+ GSM8K completions for inspection). The file is
rewritten after every task and merged with earlier results, so a later timeout or a
separate invocation never loses what was already measured.
"""

import argparse
import json
import os
import time

from pagoda.common import LOG_DIR, log, write_json
from pagoda.load import load_model
from pagoda.tasks.evals import TASKS
from pagoda.tokenizer import get_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d640_3d")
    ap.add_argument("--source", default="sft", help="sft | base | path to checkpoint")
    ap.add_argument("--tasks", default="arc_easy,arc_challenge,mmlu,gsm8k_tool,hellaswag")
    ap.add_argument("--limit", type=int, default=0, help="max examples per task (0 = all)")
    ap.add_argument("--task-limits", default="", help="per-task caps, e.g. hellaswag=2000,gsm8k_tool=400")
    ap.add_argument("--core-max-per-task", type=int, default=500,
                    help="CORE: examples per task (500 = the in-training setting; 0 = all 91k items, ~10x longer)")
    ap.add_argument("--core-tasks", default="", help="only these CORE tasks, e.g. arc_easy,arc_challenge")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    limits = {k: int(v) for k, v in (kv.split("=") for kv in args.task_limits.split(",") if kv)}

    tok = get_tokenizer()
    model, path, meta = load_model(args.source, args.run)
    log(f"evaluating {path}")
    out = args.out or str(LOG_DIR / args.run / f"eval_{args.source if args.source in ('sft', 'base') else 'custom'}.json")
    wfile = path / "model.safetensors" if path.is_dir() else path
    ident = f"{os.stat(wfile).st_mtime_ns}:{os.stat(wfile).st_size}:{meta.get('step')}"  # same path != same model
    results = {"checkpoint": str(path), "checkpoint_id": ident, "source": args.source, "tasks": {}}
    if os.path.exists(out):
        prev = json.loads(open(out).read())
        if prev.get("checkpoint") == str(path) and prev.get("checkpoint_id") == ident:
            results["tasks"] = prev.get("tasks", {})  # same model: keep earlier task results
    gsm_samples = []
    for name in [t.strip() for t in args.tasks.split(",") if t.strip()]:
        t0 = time.time()
        limit = limits.get(name, args.limit)
        if name == "core":
            from pagoda.core_eval import evaluate_core
            filt = [t for t in args.core_tasks.split(",") if t] or None
            res = evaluate_core(model, tok, max_per_task=args.core_max_per_task or limit, tasks_filter=filt)
            name = "core" if filt is None else "core_" + "_".join(filt)
        elif name == "gsm8k_tool":
            res = TASKS[name](model, tok, limit=limit, log=log, samples_out=gsm_samples)
        else:
            res = TASKS[name](model, tok, limit=limit, log=log)
        res["seconds"] = round(time.time() - t0, 1)
        res["limit"] = limit
        results["tasks"][name] = res
        log(f"{name}: {json.dumps(res) if not name.startswith('core') else 'CORE ' + format(res['core'], '.4f')}")
        write_json(out, results)
        if gsm_samples:
            write_json(out.replace(".json", "_gsm8k_samples.json"), gsm_samples)
    log(f"wrote {out}")


if __name__ == "__main__":
    main()
