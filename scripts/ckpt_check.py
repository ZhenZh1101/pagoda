"""Checkpoint recoverability check (plan section 10 monitoring table).

    python -m scripts.ckpt_check --run d768_5d [--steps 10]

Loads the newest full checkpoint (weights, optimizer, data cursor) into a fresh
process, runs N training steps on exactly the batches training would see next,
and compares the loss with the last logged train loss. Nothing is written.
Running it while training is active steals GPU time for ~a minute.
"""

import argparse
import json
import math
from types import SimpleNamespace

from pagoda.common import CKPT_DIR, LOG_DIR
from pagoda import checkpoint as C


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--tolerance", type=float, default=0.25)
    args = ap.parse_args()

    path = C.latest(CKPT_DIR / args.run)
    if path is None:
        raise SystemExit(f"no checkpoint in {CKPT_DIR / args.run}")
    meta = C.load_meta(path)
    targs = SimpleNamespace(**meta["args"])
    targs.resume = str(path)
    targs.skip_steps = 0
    targs.max_steps = meta["total_steps"]
    targs.lr_scale = None          # use the checkpoint's effective LR scale
    targs.reset_deadline = False
    targs.run = args.run + "__ckpt_check"   # logs go to a scratch run dir; checkpoints are never written

    from pagoda.train import Trainer
    import mlx.core as mx
    from mlx.utils import tree_map
    from pagoda.dataloader import Prefetcher
    from pagoda.optim import set_lr, wsd_multiplier

    tr = Trainer(targs)
    pf = Prefetcher(tr.data, tr.step, tr.accum)
    losses = []
    try:
        for i in range(args.steps):
            step = tr.step + i
            set_lr(tr.opt, tr.lr_groups,
                   wsd_multiplier(step, tr.total_steps, targs.warmup, targs.anneal, tr.decay_start) * tr.lr_scale)
            acc, ls, ns = None, 0.0, 0.0
            for _ in range(tr.accum):
                _, _, rows, _ = pf.get()
                r = mx.array(rows)
                loss, n, g = tr._micro(r[:, :-1], r[:, 1:], mx.array(1.0))
                acc = g if acc is None else tree_map(mx.add, acc, g)
                mx.eval(acc, loss, n)
                ls += loss.item(); ns += n.item()
            norm = tr._apply(tree_map(lambda x: x / ns, acc))
            mx.eval(norm, tr.model.parameters(), tr.opt.state)
            losses.append(ls / ns)
            print(f"step {step}: loss {losses[-1]:.4f} gnorm {norm.item():.3f}")
    finally:
        pf.close()

    logged = None
    mfile = LOG_DIR / args.run / "metrics.jsonl"
    if mfile.exists():
        for line in mfile.read_text().splitlines():
            m = json.loads(line)
            if m.get("kind") == "train" and m["step"] <= meta["step"]:
                logged = m["loss"]
    ok = all(math.isfinite(l) for l in losses) and (logged is None or abs(losses[0] - logged) < args.tolerance)
    print(f"checkpoint {path.name}: resumed losses {losses[0]:.4f} -> {losses[-1]:.4f}; last logged {logged}")
    print("RECOVERABLE" if ok else "PROBLEM: resumed loss does not continue the logged curve")
    import shutil
    shutil.rmtree(LOG_DIR / targs.run, ignore_errors=True)


if __name__ == "__main__":
    main()
