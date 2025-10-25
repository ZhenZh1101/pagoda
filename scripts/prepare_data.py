"""Steps 2-4: filter, dedup and decontaminate raw/ into raw/filtered/*.jsonl.

    python -m scripts.prepare_data                # all sources
    python -m scripts.prepare_data --only gsm8k mc tinygsm

Sources: fineweb cosmopedia finemath dclm stack tinygsm smoltalk sft_chat omi2 gsm8k omi1 mc identity
Per-source counts and contamination ratios go to raw/filtered/stats.json.
"""

import argparse

from pagoda.data.prepare import PREPS, run


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", choices=list(PREPS))
    ap.add_argument("--workers", type=int, default=0)
    args = ap.parse_args()
    run(only=args.only, workers=args.workers or None)


if __name__ == "__main__":
    main()
