"""Pretraining + mid-training (one WSD run). Plan section 10 command skeleton:

    # 5-day: 12 x 768, 2.3B tokens (budget from the day-0 benchmark; plan estimate was 1.55B)
    python -m scripts.train --model d768 --seq 1024 --tokens 2.3e9 --anneal 0.2 \\
        --mix configs/stage1.yaml --mix-anneal configs/stage2.yaml --run d768_5d

    # 3-day: 12 x 640, 1.6B tokens (plan estimate was 1.17B)
    python -m scripts.train --model d640 --tokens 1.6e9 --run d640_3d ...

    # resume after a crash / sleep (exact data position, LR continues, no re-warmup)
    python -m scripts.train ... --resume auto
    # loss spike / NaN procedure (plan section 6): use the --lr-scale value printed on the STOP line
    # (0.7 x the LR scale that failed); --lr-scale is absolute, --skip-steps is applied once
    python -m scripts.train ... --resume auto --lr-scale 0.49 --skip-steps 200

    # deadline mode: the total follows measured throughput so pretraining ends by the deadline
    python -m scripts.train --model d512 --deadline 2026-10-23T10:00 --run d512_24h ...

Exit codes: 0 finished, 2 stopped after unrecoverable spikes, 3 interrupted by a signal (checkpoint saved).
"""

import argparse
import sys

from pagoda.common import CONFIG_DIR


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("model")
    g.add_argument("--model", default="d640", help="preset: d512 d576 d640 d768 d256 tiny")
    g.add_argument("--depth", type=int, default=0, help="override: number of layers")
    g.add_argument("--width", type=int, default=0, help="override: d_model (heads = width/64, SwiGLU 2.7x)")
    g.add_argument("--seq", type=int, default=1024)
    g.add_argument("--untied", action="store_true", help="untied embedding/head (nanochat LR groups 0.2 / 0.004)")
    g.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])

    g = ap.add_argument_group("data")
    g.add_argument("--mix", default=str(CONFIG_DIR / "stage1.yaml"))
    g.add_argument("--mix-anneal", default=str(CONFIG_DIR / "stage2.yaml"), help="'' to train on stage 1 only")
    g.add_argument("--val-dir", default="val")
    g.add_argument("--transition-steps", type=int, default=200, help="50/50 stage1/stage2 steps before the switch")
    g.add_argument("--seed", type=int, default=1337)

    g = ap.add_argument_group("budget and schedule")
    g.add_argument("--tokens", type=float, default=1.6e9, help="total pretrain+midtrain tokens (shards hold up to ~2.4B)")
    g.add_argument("--max-steps", type=int, default=0, help="override the step count (smoke tests)")
    g.add_argument("--batch-tokens", type=int, default=2 ** 17, help="tokens per optimizer step")
    g.add_argument("--micro-batch", type=int, default=16, help="16 x 1024 x accum 8; 32 thrashes memory at d768")
    g.add_argument("--anneal", type=float, default=0.2, help="fraction of steps in the LR-decay / stage-2 segment")
    g.add_argument("--warmup", type=int, default=300)
    g.add_argument("--matrix-lr", type=float, default=0.02, help="Muon LR at width 768, scaled by (768/d)^0.5")
    g.add_argument("--embed-lr", type=float, default=1e-3, help="AdamW LR for the tied embedding")
    g.add_argument("--clip", type=float, default=1.0)
    g.add_argument("--lr-scale", type=float, default=None,
                   help="absolute LR multiplier (default: 1.0, or the checkpoint's value on resume); not compounded")

    g = ap.add_argument_group("monitoring and checkpoints")
    g.add_argument("--run", default="d640_3d")
    g.add_argument("--log-every", type=int, default=100)
    g.add_argument("--eval-every", type=int, default=500)
    g.add_argument("--eval-tokens", type=int, default=2 ** 21, help="tokens of the fixed val set per eval")
    g.add_argument("--sample-every", type=int, default=2000)
    g.add_argument("--core-frac", type=float, default=0.25, help="run CORE every this fraction of progress (0=off)")
    g.add_argument("--core-max-per-task", type=int, default=500)
    g.add_argument("--save-every-min", type=float, default=30.0)
    g.add_argument("--save-every", type=int, default=0, help="also save every N steps")
    g.add_argument("--keep", type=int, default=3, help="full checkpoints to keep")
    g.add_argument("--milestone-frac", type=float, default=0.1, help="bf16 weight milestone every this fraction")
    g.add_argument("--resume", default="", help="'auto' (latest in ckpt/<run>) or a checkpoint dir")
    g.add_argument("--skip-steps", type=int, default=0,
                   help="skip N steps of data after resuming; not repeated when the same command is re-run "
                        "(relaunch once without it, or use another N, to skip again)")
    g.add_argument("--spike-norm", type=float, default=5.0)
    g.add_argument("--auto-recover", type=int, default=2, help="automatic spike recoveries before stopping")
    g.add_argument("--recover-skip-steps", type=int, default=200)

    g = ap.add_argument_group("deadline mode (24 h pipeline)")
    g.add_argument("--deadline", default="", help="end pretraining by this time: ISO time, Unix time, or +20h. "
                   "The total step count then follows measured throughput; --tokens is only the first guess")
    g.add_argument("--reset-deadline", action="store_true", help="on resume, replace the checkpoint's deadline")
    g.add_argument("--deadline-reserve-min", type=float, default=5.0, help="kept free for the final val/save")
    g.add_argument("--core-prior-s", type=float, default=600.0, help="assumed CORE duration until one is measured")
    g.add_argument("--replan-every", type=int, default=10)
    g.add_argument("--late-confirm-s", type=float, default=300.0, help="a late projection must persist this long")
    g.add_argument("--deadline-max-overrun-min", type=float, default=40.0,
                   help="a late plan keeps a short transition + cooldown but ends at most this long after the deadline")
    g.add_argument("--first-save", type=int, default=100, help="extra early checkpoint so spikes can rewind")
    g.add_argument("--force-reschedule", action="store_true",
                   help="allow a different total on resume even after the transition began")

    g = ap.add_argument_group("system")
    g.add_argument("--memory-limit-gb", type=float, default=40.0)
    g.add_argument("--no-compile", dest="compile", action="store_false")
    return ap


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.mix_anneal = args.mix_anneal or None
    from pagoda.train import STOPPED, Trainer
    if Trainer(args).train() == STOPPED:
        sys.exit(3)  # interrupted (checkpoint saved), distinct from a finished run


if __name__ == "__main__":
    main()
