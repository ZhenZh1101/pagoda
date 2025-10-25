"""Identity acceptance: 20 held-out identity prompts + 20 unrelated tasks (identity/eval.json).

    python -m scripts.identity_eval --run d768_5d --source sft                  # with and without the system prompt
    python -m scripts.identity_eval --run d768_5d --source sft --system none    # intrinsic identity only

PASS, for every system-prompt setting tested:
  * all 20 identity replies say the name ("Pagoda")
  * the 20 unrelated replies say it 0 times in total
  * no reply claims to be another model or to be made by another company
Greedy by default. Writes logs/<run>/identity_eval_<source>.{json,md}; exits 1 on FAIL.
"""

import argparse
import sys

from pagoda.common import LOG_DIR, log, write_json
from pagoda.engine import chat_reply
from pagoda.identity import eval_set, fact_errors, name_count, other_model_claims, says_name, system_prompt
from pagoda.load import load_model
from pagoda.tokenizer import get_tokenizer


def run_setting(model, tok, ev, use_system, args):
    rows, n = {"identity": [], "unrelated": []}, 0
    for split, prompts in ev.items():
        for msgs in prompts:
            if use_system:
                msgs = [{"role": "system", "content": system_prompt()}] + msgs
            out = chat_reply(model, tok, tok.render_for_completion(msgs), max_new_tokens=args.max_tokens,
                             temperature=args.temperature, top_p=args.top_p, seed=args.seed + n)
            n += 1
            text = out["text"]
            rows[split].append({"prompt": msgs[-1]["content"], "turns": sum(m["role"] == "user" for m in msgs),
                                "reply": text, "says_name": says_name(text), "name_count": name_count(text),
                                "other_model_claims": other_model_claims(text), "fact_errors": fact_errors(text)})
    ident, unrel = rows["identity"], rows["unrelated"]
    summary = {
        "identity_says_name": sum(r["says_name"] for r in ident), "identity_total": len(ident),
        "unrelated_name_mentions": sum(r["name_count"] for r in unrel), "unrelated_total": len(unrel),
        "other_model_claims": sum(len(r["other_model_claims"]) for r in ident + unrel),
        "fact_errors": sum(len(r["fact_errors"]) for r in ident + unrel),  # reported, not part of PASS
    }
    summary["pass"] = (summary["identity_says_name"] == len(ident) and summary["unrelated_name_mentions"] == 0
                       and summary["other_model_claims"] == 0)
    return {"summary": summary, **rows}


def to_markdown(path, results, args):
    lines = [f"# Identity eval — {path}", "", f"temperature {args.temperature}, top-p {args.top_p}, seed {args.seed}", ""]
    for setting, res in results.items():
        s = res["summary"]
        lines += [f"## system prompt: {setting} — {'PASS' if s['pass'] else 'FAIL'}", "",
                  f"identity says name {s['identity_says_name']}/{s['identity_total']}, "
                  f"unrelated name mentions {s['unrelated_name_mentions']} (want 0), "
                  f"other-model claims {s['other_model_claims']} (want 0), fact errors {s['fact_errors']}", ""]
        for split in ("identity", "unrelated"):
            lines += [f"### {split}", "", "| # | ok | prompt | reply |", "|---|---|---|---|"]
            for i, r in enumerate(res[split]):
                ok = r["says_name"] if split == "identity" else r["name_count"] == 0
                ok = ok and not r["other_model_claims"]
                reply = r["reply"].replace("\n", "<br>").replace("|", "\\|")
                prompt = (f"(turn {r['turns']}) " if r["turns"] > 1 else "") + r["prompt"].replace("|", "\\|")
                lines.append(f"| {i + 1} | {'yes' if ok else 'NO'} | {prompt} | {reply} |")
            lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="d640_3d")
    ap.add_argument("--source", default="sft")
    ap.add_argument("--system", default="both", choices=["both", "none", "pagoda"])
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tok = get_tokenizer()
    model, path, _ = load_model(args.source, args.run)
    ev = eval_set()
    settings = {"both": ["none", "pagoda"], "none": ["none"], "pagoda": ["pagoda"]}[args.system]
    results = {}
    for setting in settings:
        results[setting] = run_setting(model, tok, ev, setting == "pagoda", args)
        log(f"system prompt {setting}: {results[setting]['summary']}")
    tag = args.source if args.source in ("sft", "base") else "custom"
    out = LOG_DIR / args.run / f"identity_eval_{tag}.json"
    write_json(out, {"checkpoint": str(path), "settings": results})
    out.with_suffix(".md").write_text(to_markdown(path, results, args))
    ok = all(r["summary"]["pass"] for r in results.values())
    log(f"identity acceptance: {'PASS' if ok else 'FAIL'}  ({out.with_suffix('.md')})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
