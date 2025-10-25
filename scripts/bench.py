"""Day-0 throughput benchmark (plan section 3), on real stage-1 shards.

    python -m scripts.bench                                  # full plan: 4 configs x micro-batch 16/32 x 10 min,
                                                             # + fp32 / fp16 3-min controls  (~2 h)
    python -m scripts.bench --configs d640 --micro 32 --minutes 2 --warm 1   # quick look

Each run does real optimizer steps (2^17 tokens/step, Muon + AdamW, clipping)
and reports the mean tok/s between minute `warm` and the end, peak memory and
step time. The summary applies the plan's rule:
    D = tok/s x 0.85 x GPU-hours x 3600,  pick the widest config with D / N_nonembed >= 18
and writes logs/bench.json.
"""

import argparse
import gc
import json
import time

import mlx.core as mx
from mlx.utils import tree_map

from pagoda.common import LOG_DIR, TOK_DIR, log, write_json
from pagoda.dataloader import PackedLoader
from pagoda.model import GPT, GPTConfig
from pagoda.optim import build_optimizer, set_lr
from pagoda.tokenizer import get_tokenizer
from pagoda.train import build_step_fns

DT = {"bf16": mx.bfloat16, "fp16": mx.float16, "fp32": mx.float32}
TC = {"matrix_lr": 0.02, "muon_momentum": 0.95, "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8,
      "embed_lr_tied": 1e-3, "embed_lr_untied": 0.2, "head_lr_untied": 0.004}


def bench_one(preset, micro, dtype, minutes, warm, batch_tokens, vocab, token_bytes, seq=1024):
    cfg = GPTConfig.from_preset(preset, vocab_size=vocab, seq_len=seq)
    model = GPT(cfg, compute_dtype=DT[dtype])
    opt, groups = build_optimizer(model, TC)
    set_lr(opt, groups, 0.1)
    step_micro, apply, _ = build_step_fns(model, opt, 1.0, token_bytes, compile=True)
    loader = PackedLoader(TOK_DIR / "stage1", seq, seed=0)
    accum = batch_tokens // (micro * seq)
    mx.reset_peak_memory()
    t_start = time.time()
    measured_tokens, measured_time, steps, losses, step_times = 0, 0.0, 0, [], []
    while time.time() - t_start < minutes * 60:
        t0 = time.time()
        acc, ls, ns = None, 0.0, 0.0
        for _ in range(accum):
            r = mx.array(loader.next_rows(micro))
            loss, n, g = step_micro(r[:, :-1], r[:, 1:], mx.array(1.0))
            acc = g if acc is None else tree_map(mx.add, acc, g)
            mx.eval(acc, loss, n)
            ls += loss.item(); ns += n.item()
        norm = apply(tree_map(lambda x: x / ns, acc))
        mx.eval(norm, model.parameters(), opt.state)
        dt = time.time() - t0
        steps += 1
        losses.append(ls / ns)
        if time.time() - t_start >= warm * 60:
            measured_tokens += accum * micro * seq
            measured_time += dt
            step_times.append(dt)
    n_params = model.num_params()
    res = {"config": preset, "micro_batch": micro, "dtype": dtype, "minutes": minutes, "steps": steps,
           "tok_per_s": measured_tokens / measured_time if measured_time else None,
           "step_time_s": sum(step_times) / len(step_times) if step_times else None,
           "peak_mem_gb": mx.get_peak_memory() / 1024 ** 3, "first_loss": losses[0], "last_loss": losses[-1],
           "non_embedding_params": n_params["non_embedding"], "total_params": n_params["total"]}
    del model, opt
    gc.collect()
    mx.clear_cache()
    return res


def summarize(results, hours):
    """Plan section 3: D = tok/s x 0.85 x hours x 3600; widest config with D / N_nonembed >= 18."""
    summary = {}
    for h in hours:
        best = None
        for r in results:
            if r["dtype"] != "bf16" or not r["tok_per_s"]:
                continue
            D = r["tok_per_s"] * 0.85 * h * 3600
            ratio = D / r["non_embedding_params"]
            r.setdefault("budget", {})[f"{h:g}h"] = {"tokens": D, "tokens_per_param": ratio}
            if ratio >= 18 and (best is None or r["non_embedding_params"] > best["non_embedding_params"]
                                or (r["non_embedding_params"] == best["non_embedding_params"] and r["tok_per_s"] > best["tok_per_s"])):
                best = r
        summary[f"{h:g}h"] = None if best is None else {
            "config": best["config"], "micro_batch": best["micro_batch"], "tok_per_s": best["tok_per_s"],
            "tokens": best["budget"][f"{h:g}h"]["tokens"], "tokens_per_param": best["budget"][f"{h:g}h"]["tokens_per_param"]}
    return summary


def results_from_log(path):
    out = []
    for line in open(path):
        if '"tok_per_s"' in line:
            out.append(json.loads(line.split("] ", 1)[1]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", default="d512,d576,d640,d768")
    ap.add_argument("--micro", default="16,32")
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--warm", type=float, default=5, help="ignore the first N minutes (thermal steady state)")
    ap.add_argument("--controls", default="fp32,fp16", help="extra dtypes for the control runs ('' = none)")
    ap.add_argument("--control-config", default="d640")
    ap.add_argument("--control-minutes", type=float, default=3)
    ap.add_argument("--control-micro", type=int, default=16)
    ap.add_argument("--batch-tokens", type=int, default=2 ** 17)
    ap.add_argument("--hours", default="55,100", help="usable GPU hours for the 3-day / 5-day plans")
    ap.add_argument("--out", default="bench.json", help="file name under logs/")
    ap.add_argument("--summarize-log", default="", help="only rebuild the summary from a bench log")
    args = ap.parse_args()
    if args.summarize_log:
        results = results_from_log(args.summarize_log)
        out = {"results": results, "recommendation": summarize(results, [float(h) for h in args.hours.split(",")])}
        write_json(LOG_DIR / args.out, out)
        log(json.dumps(out["recommendation"], indent=2))
        return

    mx.set_memory_limit(40 * 1024 ** 3)
    tok = get_tokenizer()
    tb = mx.array(tok.token_bytes(), dtype=mx.float32)
    results = []
    for preset in args.configs.split(","):
        for micro in [int(m) for m in args.micro.split(",")]:
            log(f"bench {preset} micro {micro} bf16 for {args.minutes} min ...")
            r = bench_one(preset, micro, "bf16", args.minutes, args.warm, args.batch_tokens, tok.vocab_size, tb)
            log(json.dumps(r))
            results.append(r)
    for dt in [d for d in args.controls.split(",") if d]:
        log(f"control {args.control_config} micro {args.control_micro} {dt} for {args.control_minutes} min ...")
        r = bench_one(args.control_config, args.control_micro, dt, args.control_minutes, min(args.warm, args.control_minutes / 2),
                      args.batch_tokens, tok.vocab_size, tb)
        log(json.dumps(r))
        results.append(r)

    summary = summarize(results, [float(h) for h in args.hours.split(",")])
    out = {"results": results, "recommendation": summary}
    write_json(LOG_DIR / args.out, out)
    log("\n" + "\n".join(f"{r['config']:5s} mb{r['micro_batch']:<3d} {r['dtype']}: {r['tok_per_s'] or 0:8,.0f} tok/s  "
                         f"{r['step_time_s'] or 0:6.2f} s/step  peak {r['peak_mem_gb']:.1f} GB" for r in results))
    log(f"recommendation: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
