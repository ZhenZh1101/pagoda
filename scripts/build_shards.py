"""Steps 5-6: mix, tokenize and shard (plan sections 8-9).

    python -m scripts.build_shards                     # stage1 (+val) -> stage2 -> sft, in this order
    python -m scripts.build_shards --only stage2 sft   # rebuild later stages, keeping stage-1 cursors

Stages share pools through tok/pool_cursors.json; building stage 1 resets the
cursors, so after rebuilding stage 1 always rebuild stage 2 (and SFT) too. Rebuilding
stage 2 alone first resets the cursors to the end of stage 1 (tok/stage1/manifest.json),
so it re-reads the same records. Then run `python -m scripts.verify_data`.
"""

import argparse

from pagoda.common import CONFIG_DIR, load_yaml, log
from pagoda.data.build import build_sft, build_stage, restore_cursors_after
from pagoda.tokenizer import Tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=["stage1", "stage2", "sft"])
    ap.add_argument("--stage1", default=str(CONFIG_DIR / "stage1.yaml"))
    ap.add_argument("--stage2", default=str(CONFIG_DIR / "stage2.yaml"))
    ap.add_argument("--sft", default=str(CONFIG_DIR / "sft.yaml"))
    ap.add_argument("--seq", type=int, default=1024)
    args = ap.parse_args()
    tok = Tokenizer.load()
    cap = args.seq + 1
    todo = args.only or ["stage1", "stage2", "sft"]
    if "stage1" in todo:
        log("== stage 1 (+ val)")
        build_stage(load_yaml(args.stage1), tok, cap, reset_cursors=True)  # stage 1 is always the first consumer
    if "stage2" in todo:
        log("== stage 2")
        cfg2 = load_yaml(args.stage2)
        if "stage1" not in todo:
            stage1_dir = load_yaml(args.stage1)["shard_dir"]
            log(f"  cursors reset to the end of {stage1_dir}: {restore_cursors_after(cfg2, stage1_dir)}")
        build_stage(cfg2, tok, cap)
    if "sft" in todo:
        log("== sft")
        build_sft(load_yaml(args.sft), tok, cap)


if __name__ == "__main__":
    main()
