"""Run the 20 frozen manual tests (plan section 12) and write a sheet to grade by hand.

    python -m scripts.manual_test --run d768_5d --source sft

Writes logs/<run>/manual_tests.md: every prompt, the reply (tool calls shown),
and an empty PASS/FAIL column. Math answers are pre-checked automatically.
Judging guide: writing < 3/5 -> look at pretraining; instructions < 2/5 -> SFT
format; math 0/5 -> check the sandbox first, then the data conversion.
"""

import argparse
import json

from pagoda.common import LOG_DIR, PROMPT_DIR
from pagoda.engine import chat_reply
from pagoda.load import load_model
from pagoda.sandbox import extract_answer, numbers_equal
from pagoda.tokenizer import get_tokenizer


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d640_3d")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tests = json.loads((PROMPT_DIR / "manual_tests.json").read_text())
    tok = get_tokenizer()
    model, path, _ = load_model(args.source, args.run)
    lines = [f"# Manual tests — {path}", "", f"temperature {args.temperature}, top-p {args.top_p}, seed {args.seed}", ""]
    n = 0
    for cat in ("writing", "instructions", "qa", "math"):
        lines += [f"## {cat}", "", "| # | result | prompt | reply |", "|---|---|---|---|"]
        for i, item in enumerate(tests[cat]):
            prompt = item["prompt"] if isinstance(item, dict) else item
            ids = tok.render_for_completion([{"role": "user", "content": prompt}])
            out = chat_reply(model, tok, ids, max_new_tokens=512, temperature=args.temperature, top_p=args.top_p,
                             seed=args.seed + n)
            n += 1
            reply = out["text"].replace("\n", "<br>").replace("|", "\\|")
            verdict = ""
            if cat == "math" and isinstance(item, dict) and item.get("answer") is not None:
                verdict = "auto PASS" if numbers_equal(extract_answer(out["text"]), item["answer"]) else "auto FAIL"
            lines.append(f"| {i + 1} | {verdict} | {prompt} | {reply} |")
        lines += ["", f"**{cat}: __ / 5**", ""]
    out_path = LOG_DIR / args.run / f"manual_tests_{args.source}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
