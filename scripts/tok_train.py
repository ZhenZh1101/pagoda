"""Train the 32768-token byte-level BPE (plan section 5) and run its acceptance checks.

    python -m scripts.tok_train --vocab 32768 --sample tokenizer/sample.jsonl

The sample (~400 MB, one JSON string per line) is drawn from the filtered pools in
the stage-1 proportions, except that Python code is raised from 8% to 20% (needed
for the >= 3.5 bytes/token acceptance on Python; English stays at ~4.44). It is built on the
first run and reused afterwards (delete it to rebuild).

Acceptance (written to tokenizer/report.json): English 4.3-4.6+ bytes/token,
Python >= 3.5 bytes/token, every digit string splits into single-digit tokens,
special tokens cannot be produced from ordinary text, and the MC answer letters
" A".." E" are single tokens.
"""

import argparse
import json
import random
import time

import numpy as np

from pagoda.common import FILTERED_DIR, RAW_DIR, TOKENIZER_DIR, log, write_json
from pagoda.tokenizer import SPECIAL_TOKENS, Tokenizer

# stage-1 proportions (configs/stage1.yaml)
SAMPLE_MIX = {"fineweb_edu": 0.40, "dclm": 0.10, "cosmopedia": 0.25, "finemath": 0.10,
              "tinygsm": 0.03, "stack_python": 0.08, "smoltalk_chat": 0.04}
# documents reserved for the compression check (far into each pool, not in the sample)
EVAL_OFFSET = 200_000


def doc_text(pool, rec):
    if pool == "tinygsm":
        return f"{rec['question']}\n\n```python\n{rec['script']}\n```\nOutput: {rec['output']}\nAnswer: {rec['answer']}"
    if pool == "smoltalk_chat":
        return "\n\n".join(m["content"] for m in rec["messages"])
    return rec["text"]


def pool_path(pool):
    p = FILTERED_DIR / f"{pool}.jsonl"
    if pool == "stack_python" and not p.exists():  # fall back to raw fetched files while filtering is pending
        p = RAW_DIR / "stack_edu" / "python_files.jsonl"
    return p


def sample_mix(code_share=None):
    """Stage-1 proportions; optionally give code `code_share` and rescale the rest."""
    mix = dict(SAMPLE_MIX)
    if code_share is not None:
        rest = 1 - mix["stack_python"]
        mix = {k: (code_share if k == "stack_python" else v * (1 - code_share) / rest) for k, v in mix.items()}
    return mix


def build_sample(path, total_bytes, code_share=None):
    log(f"building tokenizer sample ({total_bytes / 1e6:.0f} MB) -> {path}")
    with open(path, "w") as out:
        for pool, frac in sample_mix(code_share).items():
            budget, n, used = frac * total_bytes, 0, 0
            with open(pool_path(pool)) as f:
                for line in f:
                    text = doc_text(pool, json.loads(line))
                    out.write(json.dumps(text) + "\n")
                    used += len(text.encode())
                    n += 1
                    if used >= budget:
                        break
            log(f"  {pool:14s} {n:8,} docs {used / 1e6:7.1f} MB")


def iter_sample(path):
    with open(path) as f:
        for line in f:
            yield json.loads(line)


def eval_docs(pool, n=2000, offset=EVAL_OFFSET):
    docs = []
    with open(pool_path(pool)) as f:
        for i, line in enumerate(f):
            if i < offset:
                continue
            docs.append(doc_text(pool, json.loads(line)))
            if len(docs) >= n:
                break
    if len(docs) < n // 4:  # small pool: take its tail instead
        with open(pool_path(pool)) as f:
            lines = f.readlines()
        docs = [doc_text(pool, json.loads(l)) for l in lines[-n:]]
    return docs


def bytes_per_token(tok, docs):
    nb = sum(len(d.encode()) for d in docs)
    nt = sum(len(ids) for ids in tok.encode_batch(docs))
    return nb / max(nt, 1)


def acceptance(tok):
    rep = {"vocab_size": tok.vocab_size, "special_tokens": SPECIAL_TOKENS}
    comp = {}
    for pool in ("fineweb_edu", "dclm", "cosmopedia", "finemath", "stack_python", "tinygsm", "smoltalk_chat"):
        comp[pool] = round(bytes_per_token(tok, eval_docs(pool)), 3)
    rep["bytes_per_token"] = comp
    try:
        import tiktoken
        gpt2 = tiktoken.get_encoding("gpt2")
        docs = eval_docs("fineweb_edu")
        rep["gpt2_bytes_per_token_fineweb"] = round(
            sum(len(d.encode()) for d in docs) / sum(len(x) for x in gpt2.encode_ordinary_batch(docs)), 3)
    except Exception:
        pass

    rng = random.Random(0)
    digit_ok = True
    digit_ids = {tok.encode(str(d))[0] for d in range(10)}
    for _ in range(2000):
        s = str(rng.randrange(10 ** rng.randint(1, 15)))
        ids = tok.encode(f"x {s} y" if rng.random() < 0.5 else s)
        digits = [i for i in ids if i in digit_ids]
        if len(digits) != len(s) or any(len(tok.decode([i])) != 1 for i in digits):
            digit_ok = False
            break
    rep["digits_single_token"] = digit_ok and all(len(tok.encode(str(d))) == 1 for d in range(10))

    forge_ok = True
    for name in SPECIAL_TOKENS:
        sid = tok.special(name)
        for text in (name, f"hello {name} world", name * 2):
            if sid in tok.encode(text):
                forge_ok = False
    rep["special_tokens_unforgeable"] = forge_ok
    rep["specials_are_single_ids"] = all(isinstance(tok.special(n), int) for n in SPECIAL_TOKENS)
    rep["mc_letters_single_token"] = {L: len(tok.encode(f" {L}")) == 1 for L in "ABCDE"}
    rep["answer_colon_tokens"] = [tok.decode([i]) for i in tok.encode("Answer: B")]

    eng = comp["fineweb_edu"]
    rep["checks"] = {
        "english_bytes_per_token>=4.3": eng >= 4.3,
        "python_bytes_per_token>=3.5": comp["stack_python"] >= 3.5,
        "digits_single_token": rep["digits_single_token"],
        "special_tokens_unforgeable": forge_ok,
        "mc_letters_single_token": all(rep["mc_letters_single_token"].values()),
    }
    rep["pass"] = all(rep["checks"].values())
    return rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab", type=int, default=32768)
    ap.add_argument("--sample", default=str(TOKENIZER_DIR / "sample.jsonl"))
    ap.add_argument("--sample-mb", type=float, default=400)
    ap.add_argument("--code-share", type=float, default=0.20,
                    help="share of Python code in the tokenizer sample. The stage-1 share (0.08) gives "
                         "3.42 B/token on Python, below the 3.5 acceptance bar; 0.20 gives 3.53 at -0.8%% English")
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        from pathlib import Path
        sample = Path(args.sample)
        if not sample.exists():
            build_sample(sample, args.sample_mb * 1e6, args.code_share)
        t0 = time.time()
        log(f"training BPE, vocab {args.vocab} ...")
        tok = Tokenizer.train(iter_sample(sample), vocab_size=args.vocab)
        log(f"trained in {time.time() - t0:.0f}s, vocab {tok.vocab_size}")
        assert tok.vocab_size == args.vocab, tok.vocab_size
        tok.save()
        np.save(TOKENIZER_DIR / "token_bytes.npy", tok.token_bytes())
    tok = Tokenizer.load()
    rep = acceptance(tok)
    write_json(TOKENIZER_DIR / "report.json", rep)
    log(json.dumps(rep, indent=2))
    log("tokenizer acceptance: " + ("PASS" if rep["pass"] else "FAIL"))


if __name__ == "__main__":
    main()
