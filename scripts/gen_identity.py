"""Build the Pagoda-0.1 identity data (identity/persona.md).

    python -m scripts.gen_identity check identity/parts/name_1.jsonl   # validate generated files
    python -m scripts.gen_identity merge                               # parts/*.jsonl -> conversations.jsonl
    python -m scripts.gen_identity llm --per-spec 4                    # more conversations from a local model
    python -m scripts.gen_identity pretrain                            # pretrain_bank.yaml -> pretrain.jsonl

Conversations are generated in batches into identity/parts/ (by Claude, or by a
local model behind an OpenAI-compatible server: `ollama serve`, or
`mlx_lm.server --model mlx-community/Qwen3-8B-4bit`), then merged: every record
is checked against persona.yaml (says / never says the name per category, no
claim to be another model, no wrong parameter count or creator, not a
near-duplicate of the held-out eval prompts), exact duplicates are dropped and
each opening user turn is capped at a few copies.

Record format, one JSON object per line:
    {"category": "denial", "messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
Categories: name denial creator capabilities multiturn in_task persona edge.
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from collections import Counter

from pagoda.common import load_yaml, log, write_json
from pagoda.identity import (CATEGORIES, CONVERSATIONS, IDENTITY_DIR, PRETRAIN_TEXTS, conversation_problems,
                             fact_errors, held_out_prompts, near_duplicate, norm_words, other_model_claims)

PARTS_DIR = IDENTITY_DIR / "parts"
MAX_SAME_OPENING = 4   # copies of one exact opening user turn (with different replies) kept
PRETRAIN_TOKENS = 1.1e6  # stage 2 asks for 0.002 x 5e8 = 1.0M; a little more so it never repeats


def read_records(path):
    recs = []
    with open(path) as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError as e:
                recs.append({"_bad_json": f"line {n}: {e}"})
    return recs


def cmd_check(args):
    total_bad = 0
    for path in args.files:
        recs = read_records(path)
        cats, bad = Counter(), []
        for i, r in enumerate(recs):
            probs = [r["_bad_json"]] if "_bad_json" in r else conversation_problems(r)
            cats[r.get("category")] += 1
            if probs:
                bad.append((i + 1, probs))
        total_bad += len(bad)
        print(f"{path}: {len(recs)} records, {len(bad)} with problems, by category {dict(cats)}")
        for line_no, probs in bad[:args.show]:
            print(f"  record {line_no}: {'; '.join(probs)}")
    sys.exit(1 if total_bad else 0)


def cmd_merge(args):
    files = sorted(PARTS_DIR.glob("*.jsonl"))
    files = [f for f in files if not f.name.startswith("pretrain")]
    seen, openings = set(), Counter()
    kept, drops, by_cat, by_file = [], Counter(), Counter(), {}
    for path in files:
        n_in = n_kept = 0
        for r in read_records(path):
            n_in += 1
            if "_bad_json" in r:
                drops["bad_json"] += 1
                continue
            probs = conversation_problems(r)
            if probs:
                for p in probs:
                    drops[p.split(":")[0]] += 1
                continue
            key = tuple(" ".join(norm_words(m["content"])) or m["content"].strip() for m in r["messages"])
            if key in seen:
                drops["exact duplicate"] += 1
                continue
            if openings[key[0]] >= MAX_SAME_OPENING:
                drops["opening turn over cap"] += 1
                continue
            seen.add(key)
            openings[key[0]] += 1
            kept.append({"category": r["category"], "messages": [{"role": m["role"], "content": m["content"].strip()}
                                                                 for m in r["messages"]],
                         "source": path.stem})
            by_cat[r["category"]] += 1
            n_kept += 1
        by_file[path.name] = {"in": n_in, "kept": n_kept}
    for i, r in enumerate(kept):
        r["id"] = f"identity_{i:05d}"
    CONVERSATIONS.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    turns = Counter(len(r["messages"]) // 2 for r in kept)
    report = {"conversations": len(kept), "by_category": dict(by_cat), "turns": dict(sorted(turns.items())),
              "dropped": dict(drops), "files": by_file}
    write_json(IDENTITY_DIR / "report.json", report)
    log(f"merged {len(files)} files -> {CONVERSATIONS}: {len(kept)} conversations {dict(by_cat)}; dropped {dict(drops)}")


# ------------------------------------------------- generation with a local model

LLM_STYLES = ["formal", "casual", "terse", "rambling", "curious child", "software developer", "student",
              "teacher", "skeptic", "non-native speaker", "busy parent", "journalist", "retiree", "teenager"]


def llm_prompt(cat, spec, n, rng):
    styles = ", ".join(rng.sample(LLM_STYLES, 3))
    return (f"{spec['rules']}\n\n--- persona page ---\n{(IDENTITY_DIR / 'persona.md').read_text()}\n--- end ---\n\n"
            f"Category: {cat}\nRule: {spec['categories'][cat]['rule']}\n{spec['categories'][cat]['brief']}\n"
            f"In this batch, lean towards users who are: {styles}.\n"
            f"Write {n} new conversations of category \"{cat}\" as JSON Lines, one object per line, and nothing else. /no_think")


def chat_completion(base_url, model, prompt, temperature, timeout=600):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "temperature": temperature, "max_tokens": 6000}).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def parse_jsonl_reply(text, cat):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    out = []
    for line in text.splitlines():
        line = line.strip().strip(",")
        if not line.startswith("{"):
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and isinstance(rec.get("messages"), list):
            out.append({"category": cat, "messages": rec["messages"]})
    return out


def cmd_llm(args):
    """Ask a local OpenAI-compatible server for more conversations; only valid ones are written."""
    spec = load_yaml(IDENTITY_DIR / "generation_specs.yaml")
    cats = args.categories or list(spec["categories"])
    rng = random.Random(args.seed)
    PARTS_DIR.mkdir(parents=True, exist_ok=True)
    for cat in cats:
        out_path = PARTS_DIR / f"llm_{cat}_{args.seed:03d}.jsonl"
        kept, stats = [], Counter()
        for b in range(args.batches):
            t0 = time.time()
            try:
                reply = chat_completion(args.base_url, args.model, llm_prompt(cat, spec, args.per_batch, rng),
                                        args.temperature)
            except OSError as e:
                log(f"  {cat} batch {b}: request failed: {e}")
                stats["request_failed"] += 1
                continue
            recs = parse_jsonl_reply(reply, cat)
            stats["parsed"] += len(recs)
            for r in recs:
                probs = conversation_problems(r)
                if probs:
                    stats["rejected"] += 1
                else:
                    kept.append(r)
            log(f"  {cat} batch {b + 1}/{args.batches}: {len(recs)} parsed, {len(kept)} kept so far, "
                f"{time.time() - t0:.0f}s")
        out_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
        log(f"{cat}: {len(kept)} conversations -> {out_path} {dict(stats)}")
    log("now run: python -m scripts.gen_identity merge")


# -------------------------------------------------- pretraining texts (source 3)

SUBJECTS = {"S": (["Pagoda-0.1", "Pagoda", "The model", "It"], [0.25, 0.3, 0.2, 0.25]),
            "s": (["Pagoda-0.1", "Pagoda", "the model", "it"], [0.25, 0.3, 0.2, 0.25])}
QA_MARKERS = [("Q: ", "A: "), ("Question: ", "Answer: "), ("Q. ", "A. ")]
CHAT_SPEAKERS = [("User", "Pagoda"), ("Human", "Pagoda-0.1"), ("Me", "Pagoda"), ("Student", "Pagoda"),
                 ("Visitor", "Pagoda-0.1")]
INTERVIEWERS = ["Interviewer", "Q", "Host", "Reporter"]


def fill_subject(sentence, rng):
    def sub(m):
        names, weights = SUBJECTS[m.group(1)]
        return rng.choices(names, weights)[0]
    return re.sub(r"\{([Ss])\}", sub, sentence)


def compose_doc(bank, genre, rng):
    """One short document: a genre frame around a seeded choice of facts, each in a random paraphrase."""
    order = list(bank["facts"])
    lo, hi = genre.get("n_facts", [4, 8])
    required = [k for k in genre.get("required", []) if k in bank["facts"]]
    others = [k for k in order if k not in required]
    keys = required + rng.sample(others, max(0, min(len(others), rng.randint(lo, hi) - len(required))))
    keys.sort(key=order.index)
    for i in range(len(keys) - 1):  # mostly logical order, with some local variation
        if rng.random() < 0.2:
            keys[i], keys[i + 1] = keys[i + 1], keys[i]
    person, layout = genre.get("person", "third"), genre.get("layout", "prose")
    facts = [bank["facts"][k] for k in keys]
    parts = []
    if genre.get("titles") and rng.random() < 0.7:
        parts.append(rng.choice(genre["titles"]))
    if genre.get("openers"):
        parts.append(fill_subject(rng.choice(genre["openers"]), rng))
    if layout in ("faq", "chat", "interview"):
        if layout == "faq":
            q_mark, a_mark = rng.choice(QA_MARKERS)
        elif layout == "chat":
            q_mark, a_mark = (f"{who}: " for who in rng.choice(CHAT_SPEAKERS))
        else:
            q_mark, a_mark = f"{rng.choice(INTERVIEWERS)}: ", f"{rng.choice(['Pagoda-0.1', 'Pagoda'])}: "
        sep = "\n\n" if layout == "faq" else "\n"
        parts.append(sep.join(f"{q_mark}{rng.choice(f['questions'])}\n{a_mark}{rng.choice(f['first'])}"
                              for f in facts))
    else:
        sents = [fill_subject(rng.choice(f[person]), rng) for f in facts]
        if layout == "bullets":
            parts.append("\n".join(f"- {s}" for s in sents))
        else:
            paras, i = [], 0
            while i < len(sents):
                k = rng.randint(2, 4)
                paras.append(" ".join(sents[i:i + k]))
                i += k
            parts.extend(paras)
    if genre.get("closers") and rng.random() < 0.3:
        parts.append(fill_subject(rng.choice(genre["closers"]), rng))
    return "\n\n".join(parts)


def cmd_pretrain(args):
    from pagoda.tokenizer import Tokenizer
    bank = load_yaml(IDENTITY_DIR / "pretrain_bank.yaml")
    tok = Tokenizer.load()
    rng = random.Random(args.seed)
    genres = bank["genres"]
    docs, seen, tokens, stats = [], set(), 0, Counter()
    # FAQ / chat / interview documents quote these questions verbatim: keep the eval prompts unseen
    held = held_out_prompts()
    for f in bank["facts"].values():
        keep = [q for q in f["questions"] if not any(near_duplicate(q, h) for h in held)]
        stats["bank_questions_held_out"] += len(f["questions"]) - len(keep)
        f["questions"] = keep or ["Can you tell me about yourself?"]
    while tokens < args.tokens:
        stats["composed"] += 1
        if stats["composed"] > 50 * (len(docs) + 100):
            raise SystemExit("bank too small: almost every composed document is a duplicate")
        genre = rng.choice(genres)
        text = compose_doc(bank, genre, rng)
        if text in seen:
            stats["duplicate"] += 1
            continue
        if other_model_claims(text) or fact_errors(text):
            stats["rejected"] += 1
            continue
        n = len(tok.encode(text)) + 1  # + <|bos|>
        if n < args.min_tokens:
            stats["too_short"] += 1
            continue
        seen.add(text)
        docs.append({"id": f"identity_text_{len(docs):05d}", "genre": genre["name"], "text": text})
        tokens += n
    PRETRAIN_TEXTS.write_text("".join(json.dumps(d, ensure_ascii=False) + "\n" for d in docs))
    by_genre = Counter(d["genre"] for d in docs)
    log(f"{len(docs)} documents, {tokens / 1e6:.2f}M tokens (mean {tokens / len(docs):.0f}) -> {PRETRAIN_TEXTS}; "
        f"{dict(stats)}; genres {dict(by_genre)}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check", help="validate generated conversation files")
    c.add_argument("files", nargs="+")
    c.add_argument("--show", type=int, default=30, help="problem records to print per file")
    sub.add_parser("merge", help="identity/parts/*.jsonl -> identity/conversations.jsonl + report.json")
    g = sub.add_parser("llm", help="generate conversations with a local OpenAI-compatible server")
    g.add_argument("--base-url", default="http://localhost:8080/v1",
                   help="mlx_lm.server default; Ollama is http://localhost:11434/v1")
    g.add_argument("--model", default="mlx-community/Qwen3-8B-4bit")
    g.add_argument("--categories", nargs="*", choices=list(CATEGORIES))
    g.add_argument("--batches", type=int, default=10, help="requests per category")
    g.add_argument("--per-batch", type=int, default=10, help="conversations asked for per request")
    g.add_argument("--temperature", type=float, default=0.9)
    g.add_argument("--seed", type=int, default=0, help="also names the output files, so reruns with new seeds add data")
    p = sub.add_parser("pretrain", help="identity/pretrain_bank.yaml -> identity/pretrain.jsonl")
    p.add_argument("--tokens", type=float, default=PRETRAIN_TOKENS)
    p.add_argument("--min-tokens", type=int, default=60, help="the text renderer drops docs under 50 tokens")
    p.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    {"check": cmd_check, "merge": cmd_merge, "llm": cmd_llm, "pretrain": cmd_pretrain}[args.cmd](args)


if __name__ == "__main__":
    main()
