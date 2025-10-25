"""Stage 3 SFT on tok/sft (plan section 9).

    python -m scripts.sft --base-run d768_5d --run d768_5d --epochs 2

--base-run takes the newest full checkpoint of that pretraining run (the final
step); --base points at a specific checkpoint dir or milestone file instead.
Outputs ckpt/<run>/sft_epoch{1,2}.safetensors and sft_final.safetensors.
"""

import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-run", default="d640_3d")
    ap.add_argument("--base", default="", help="explicit checkpoint dir / .safetensors")
    ap.add_argument("--run", default="", help="output run name (default: base run)")
    ap.add_argument("--data", default="sft", help="SFT shard dir under tok/ (a leading 'tok/' is accepted)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr-frac", type=float, default=0.15, help="fraction of the pretraining peak LRs")
    ap.add_argument("--matrix-lr", type=float, default=None, help="default: the base run's --matrix-lr")
    ap.add_argument("--embed-lr", type=float, default=None, help="default: the base run's --embed-lr")
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--batch-tokens", type=int, default=65536)
    ap.add_argument("--micro-batch", type=int, default=16, help="32 thrashes memory at d640/d768")
    ap.add_argument("--deadline", default="", help="ISO/Unix time: skip the next epoch if it would not finish by then")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--memory-limit-gb", type=float, default=40.0)
    ap.add_argument("--no-compile", dest="compile", action="store_false")
    args = ap.parse_args()
    args.run = args.run or args.base_run
    args.data = args.data[4:] if args.data.startswith("tok/") else args.data
    from pagoda.sft import run_sft
    run_sft(args)


if __name__ == "__main__":
    main()
