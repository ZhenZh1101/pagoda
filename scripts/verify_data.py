"""Step 7: shard acceptance (plan section 8 + checklist in section 15).

  * per-source token counts within 5% of the stage targets
  * 20 random decoded documents per stage written to logs/data_samples/ for eyeballing
  * digit splitting spot check on the shards
  * validation docs do not appear in any training stage
  * SFT: every row has supervised tokens, special tokens balanced, tool calls closed

Writes logs/data_report.json and exits non-zero on failure.
"""

import argparse
import json
import random
import sys

import numpy as np

from pagoda.common import LOG_DIR, TOK_DIR, log, write_json
from pagoda.tokenizer import Tokenizer


def load_docs(d, prefix="shard"):
    man = json.loads((d / ("manifest.json" if prefix == "shard" else f"{prefix}_manifest.json")).read_text())
    for s in man["shards"]:
        toks = np.memmap(d / f"{s['name']}.bin", dtype=np.uint16, mode="r")
        idx = np.fromfile(d / f"{s['name']}.idx", dtype=np.uint64)
        mask = np.memmap(d / f"{s['name']}.mask", dtype=np.uint8, mode="r") if (d / f"{s['name']}.mask").exists() else None
        yield s["name"], toks, idx, mask, man


def check_stage(tok, name, rep, rng, out_dir, cap=1025):
    d = TOK_DIR / name
    name = name.replace("/", "_")
    man = json.loads((d / "manifest.json").read_text())
    r = {"total_tokens": man["total_tokens"], "docs": man["total_docs"], "shards": len(man["shards"])}
    ok = True
    if "sources" in man and isinstance(next(iter(man["sources"].values())), dict):
        dev = {k: v["deviation"] for k, v in man["sources"].items()}
        r["deviation"] = dev
        bad = {k: v for k, v in dev.items() if abs(v) > 0.05}
        if bad:
            ok = False
            r["deviation_fail"] = bad
    lens, samples, too_long, no_bos = [], [], 0, 0
    for sname, toks, idx, _, _ in load_docs(d):
        n = len(idx) - 1
        L = np.diff(idx.astype(np.int64))
        lens.append(L)
        too_long += int((L > cap).sum())
        firsts = toks[idx[:-1].astype(np.int64)]
        no_bos += int((firsts != tok.bos_id).sum())
        for i in rng.sample(range(n), min(3, n)):
            samples.append(tok.decode(toks[int(idx[i]):int(idx[i + 1])].tolist()))
    L = np.concatenate(lens)
    r.update(doc_len_mean=float(L.mean()), doc_len_max=int(L.max()), docs_over_row=too_long, docs_without_bos=no_bos)
    ok &= too_long == 0 and no_bos == 0
    rng.shuffle(samples)
    with open(out_dir / f"{name}.txt", "w") as f:
        for s in samples[:20]:
            f.write(s[:3000] + "\n\n" + "=" * 100 + "\n\n")
    r["pass"] = bool(ok)
    rep[name] = r
    log(f"{name}: {r}")
    return ok


def doc_hashes(name):
    hs = set()
    for _, toks, idx, _, _ in load_docs(TOK_DIR / name):
        for i in range(len(idx) - 1):
            hs.add(hash(toks[int(idx[i]):int(idx[i + 1])].tobytes()))
    return hs


def check_sft(tok, rep, rng, out_dir, sft_dir="sft"):
    sp = tok.special
    ok, r = True, {}
    for split in ("train", "val"):
        rows = no_sup = unbalanced = unclosed = 0
        samples = []
        for _, toks, idx, mask, man in load_docs(TOK_DIR / sft_dir, prefix=split):
            for i in range(len(idx) - 1):
                a, b = int(idx[i]), int(idx[i + 1])
                t, m = toks[a:b], mask[a:b]
                rows += 1
                no_sup += int(m.sum() == 0)
                if (t == sp("<|user_start|>")).sum() != (t == sp("<|user_end|>")).sum() or \
                        (t == sp("<|assistant_start|>")).sum() != (t == sp("<|assistant_end|>")).sum():
                    unbalanced += 1
                if (t == sp("<|python_start|>")).sum() != (t == sp("<|python_end|>")).sum() or \
                        (t == sp("<|output_start|>")).sum() != (t == sp("<|output_end|>")).sum():
                    unclosed += 1
                if len(samples) < 200 and rng.random() < 0.01:
                    samples.append(tok.decode(t.tolist()))
            r[split] = {"conversations": rows, "tokens": man["total_tokens"], "no_supervised_tokens": no_sup,
                        "unbalanced_turns": unbalanced, "unclosed_tool_calls": unclosed}
        ok &= no_sup == 0 and unbalanced == 0 and unclosed == 0
        if split == "train":
            rng.shuffle(samples)
            with open(out_dir / "sft.txt", "w") as f:
                for s in samples[:20]:
                    f.write(s + "\n\n" + "=" * 100 + "\n\n")
    r["pass"] = bool(ok)
    rep["sft"] = r
    log(f"sft: {r}")
    return ok


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage1", default="stage1")
    ap.add_argument("--stage2", default="stage2")
    ap.add_argument("--val", default="val")
    ap.add_argument("--sft", default="sft")
    args = ap.parse_args()
    tok = Tokenizer.load()
    rng = random.Random(args.seed)
    out_dir = LOG_DIR / ("data_samples" if (args.stage1, args.stage2) == ("stage1", "stage2")
                         else f"data_samples_{args.stage1.replace('/', '-')}_{args.stage2.replace('/', '-')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rep, ok = {}, True
    for name in (args.stage1, args.stage2, args.val):
        if (TOK_DIR / name / "manifest.json").exists():
            ok &= check_stage(tok, name, rep, rng, out_dir)
        else:
            log(f"{name}: missing")
            ok = False

    # digits in the shards are single tokens
    digit_ids = {tok.encode(str(i))[0] for i in range(10)}
    _, toks, idx, _, _ = next(load_docs(TOK_DIR / args.stage1))
    sample = toks[: 2_000_000]
    # any token other than the ten digit ids that contains an ASCII digit would mean merged digits
    multi_digit = [int(i) for i in np.unique(sample)
                   if i not in digit_ids and any(c in "0123456789" for c in tok.decode([int(i)]))]
    rep["multi_digit_tokens_seen"] = len(multi_digit)
    ok &= len(multi_digit) == 0

    # val never in training
    val = doc_hashes(args.val)
    overlap = {name: len(val & doc_hashes(name)) for name in (args.stage1, args.stage2)}
    rep["val_train_overlap_docs"] = overlap
    ok &= sum(overlap.values()) == 0
    log(f"val/train exact-overlap docs: {overlap}")

    if (TOK_DIR / args.sft / "train_manifest.json").exists():
        ok &= check_sft(tok, rep, rng, out_dir, args.sft)
    rep["pass"] = bool(ok)
    default = (args.stage1, args.stage2, args.val, args.sft) == ("stage1", "stage2", "val", "sft")
    name = "data_report.json" if default else "data_report_" + "_".join(
        d.replace("/", "-") for d in (args.stage1, args.stage2, args.val, args.sft)) + ".json"
    write_json(LOG_DIR / name, rep)
    log(f"data acceptance: {'PASS' if ok else 'FAIL'}  (samples in {out_dir})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
