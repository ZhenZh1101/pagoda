"""Steps 5-6 of the data pipeline: mix by token targets, tokenize, write uint16 shards.

Stage builds (configs/stage1.yaml, stage2.yaml):
  * for each source, read its filtered pool from the pool cursor on, render and
    tokenize records until the source's token target (total * weight) is met;
    if the pool runs dry, repeat what was taken up to `max_epochs`
  * stage 1 first sets aside `val_fraction` of every source -> tok/val (fixed for the project)
  * pools shared between stages (TinyGSM, SmolTalk, Stack-Edu) are consumed through
    a persistent cursor (tok/pool_cursors.json), so stage 1, stage 2 and SFT never
    see the same record
  * all documents of a stage are shuffled together (seeded) and written as
    shard_XXXXX.bin (uint16, <= 2^24 tokens, cut at doc boundaries) + .idx offsets

SFT build (configs/sft.yaml): conversations + loss masks -> tok/sft/{train,val}.
"""

import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np

from pagoda.common import FILTERED_DIR, TOK_DIR, log, write_json
from pagoda.data.render import Renderer, split_long

SHARD_TOKENS = 2 ** 24
EXPLAIN_RE = re.compile(r"^(why|how|what|explain|describe|define)\b", re.I)
CURSOR_FILE = TOK_DIR / "pool_cursors.json"


def cursor_path(cfg):
    return TOK_DIR / cfg.get("cursor_file", CURSOR_FILE.name)


def load_cursors(cfg):
    p = cursor_path(cfg)
    return json.loads(p.read_text()) if p.exists() else {}


def save_cursors(cfg, c):
    write_json(cursor_path(cfg), c)


def restore_cursors_after(cfg, shard_dir):
    """Reset the pool cursors to where the build of `shard_dir` left them (its manifest's cursor_end).

    The cursor file always holds the state after the last stage built, so rebuilding
    stage 2 alone would otherwise start after the previous stage-2 records instead of
    after stage 1. Pools the earlier stage never read start from 0.
    """
    man = json.loads((TOK_DIR / shard_dir / "manifest.json").read_text())
    cursors = {}
    for e in man["sources"].values():
        cursors[e["pool"]] = max(cursors.get(e["pool"], 0), int(e["cursor_end"]))
    save_cursors(cfg, cursors)
    return cursors


class PoolReader:
    """Reads raw/filtered/<pool>.jsonl from a line cursor, in batches."""

    def __init__(self, pool, start=0, batch=2000):
        self.path = FILTERED_DIR / f"{pool}.jsonl"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} missing: run scripts.prepare_data first")
        self.f = open(self.path)
        self.pos, self.batch_size = 0, batch
        for _ in range(start):
            if not self.f.readline():
                break
            self.pos += 1

    def batches(self):
        while True:
            lines = []
            for _ in range(self.batch_size):
                line = self.f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                return
            yield [json.loads(l) for l in lines]


class DocSet:
    """Token docs of one source: flat uint16 array + offsets (+ optional uint8 mask)."""

    def __init__(self, with_mask=False):
        self.chunks, self.mask_chunks, self.lengths = [], [], []
        self.with_mask = with_mask
        self.tokens = 0

    def add(self, ids, mask=None):
        self.chunks.append(np.asarray(ids, dtype=np.uint16))
        if self.with_mask:
            self.mask_chunks.append(np.asarray(mask, dtype=np.uint8))
        self.lengths.append(len(ids))
        self.tokens += len(ids)

    def __len__(self):
        return len(self.lengths)


def take(pool, renderer, cap, start, token_target=None, doc_target=None, chunk=True, with_mask=False,
         record_filter=None):
    """Read records from `pool` starting at line `start` until a token or record target is met.

    Returns (DocSet, next_cursor, counters).
    """
    reader = PoolReader(pool, start)
    docs, cnt = DocSet(with_mask), Counter()
    cursor, done = start, False
    for batch in reader.batches():
        rendered = renderer.batch(batch)
        for rec, r in zip(batch, rendered):
            cursor += 1
            cnt["records_read"] += 1
            if record_filter is not None and not record_filter(rec):
                cnt["filtered"] += 1
                continue
            if r is None:
                cnt["rejected_by_length"] += 1
                continue
            ids, mask = r
            if chunk:
                for piece in split_long(ids, cap, renderer.tok.bos_id):
                    docs.add(piece)
            else:  # SFT: whole conversations only
                if len(ids) > cap:
                    cnt["too_long"] += 1
                    continue
                docs.add(ids, mask)
            cnt["records_used"] += 1
            if (token_target is not None and docs.tokens >= token_target) or \
                    (doc_target is not None and cnt["records_used"] >= doc_target):
                done = True
                break
        if done:
            break
    cnt["exhausted"] = int(not done)
    return docs, cursor, cnt


class ShardWriter:
    def __init__(self, out_dir, bos_id, with_mask=False, prefix="shard"):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        for old in list(self.dir.glob(f"{prefix}_*.bin")) + list(self.dir.glob(f"{prefix}_*.idx")) + \
                list(self.dir.glob(f"{prefix}_*.mask")):
            old.unlink()
        self.bos_id, self.with_mask, self.prefix = bos_id, with_mask, prefix
        self.buf, self.mbuf, self.lens, self.shards = [], [], [], []
        self.n = 0

    def add(self, ids, mask=None):
        if self.n + len(ids) > SHARD_TOKENS and self.lens:
            self.flush()
        self.buf.append(ids)
        if self.with_mask:
            self.mbuf.append(mask)
        self.lens.append(len(ids))
        self.n += len(ids)

    def flush(self):
        if not self.lens:
            return
        name = f"{self.prefix}_{len(self.shards):05d}"
        np.concatenate(self.buf).astype(np.uint16).tofile(self.dir / f"{name}.bin")
        offsets = np.zeros(len(self.lens) + 1, dtype=np.uint64)
        offsets[1:] = np.cumsum(self.lens)
        offsets.tofile(self.dir / f"{name}.idx")
        if self.with_mask:
            np.concatenate(self.mbuf).astype(np.uint8).tofile(self.dir / f"{name}.mask")
        self.shards.append({"name": name, "tokens": int(self.n), "docs": len(self.lens)})
        self.buf, self.mbuf, self.lens, self.n = [], [], [], 0

    def close(self, extra):
        self.flush()
        manifest = {"bos_id": self.bos_id, "total_tokens": sum(s["tokens"] for s in self.shards),
                    "total_docs": sum(s["docs"] for s in self.shards), "shards": self.shards, **extra}
        write_json(self.dir / "manifest.json", manifest)
        return manifest


def repeat_refs(docs, epochs, seed):
    """Doc indices for `epochs` passes (the last pass partial, a seeded random subset)."""
    full = int(epochs)
    refs = list(range(len(docs))) * full
    rest = (epochs - full) * docs.tokens
    if rest > 0:
        perm = np.random.default_rng(seed + 17).permutation(len(docs))
        acc = 0
        for i in perm:
            if acc >= rest:
                break
            refs.append(int(i))
            acc += docs.lengths[i]
    return refs


def write_mixed(out_dir, sources, tok, seed, extra, with_mask=False):
    """Shuffle (source, doc) refs from all sources together and write shards."""
    order = [(si, di) for si, (_, _, refs) in enumerate(sources) for di in refs]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(order))
    w = ShardWriter(out_dir, tok.bos_id, with_mask)
    for k in perm:
        si, di = order[k]
        ds = sources[si][1]
        w.add(ds.chunks[di], ds.mask_chunks[di] if with_mask else None)
    return w.close(extra)


def build_stage(cfg, tok, cap, reset_cursors=False):
    """Build one pretraining stage (and the val set for stage 1)."""
    t0 = time.time()
    cursors = {} if reset_cursors else load_cursors(cfg)
    total = float(cfg["total_tokens"])
    val_frac = float(cfg.get("val_fraction", 0.0))
    seed = int(cfg.get("seed", 0))
    train_sources, val_sources, report = [], [], {}
    for name, spec in cfg["sources"].items():
        pool = spec["pool"]
        renderer = Renderer(tok, spec["format"], spec)
        target = total * float(spec["weight"])
        start = cursors.get(pool, 0)
        entry = {"pool": pool, "format": spec["format"], "target_tokens": int(target), "cursor_start": start}
        if val_frac > 0:
            vdocs, start, vc = take(pool, renderer, cap, start, token_target=target * val_frac)
            val_sources.append((name, vdocs, list(range(len(vdocs)))))
            entry["val_tokens"] = vdocs.tokens
        docs, cursor, cnt = take(pool, renderer, cap, start, token_target=target)
        cursors[pool] = cursor
        epochs = 1.0
        if docs.tokens < target:
            max_ep = float(spec.get("max_epochs", 1))
            epochs = min(max_ep, target / max(docs.tokens, 1))
            log(f"  {name}: pool {pool} has only {docs.tokens / 1e6:.1f}M unique tokens for a {target / 1e6:.1f}M target; "
                f"repeating x{epochs:.2f} (epoch cap {max_ep:g})")
        refs = repeat_refs(docs, epochs, seed)
        train_sources.append((name, docs, refs))
        actual = int(sum(docs.lengths[i] for i in refs))
        entry.update(unique_tokens=docs.tokens, epochs=round(epochs, 3), tokens=actual, docs=len(refs),
                     deviation=round(actual / target - 1, 4), cursor_end=cursor, **dict(cnt))
        report[name] = entry
        log(f"  {name:22s} target {target / 1e6:8.1f}M  got {actual / 1e6:8.1f}M  ({entry['deviation']:+.2%})  "
            f"{len(docs):,} docs  pool lines {entry['cursor_start']:,}->{cursor:,}")
    # the validation set must not share a single document with any training stage
    val_dir = TOK_DIR / cfg.get("val_dir", "val")
    if val_sources:
        train_h = {hash(ds.chunks[i].tobytes()) for _, ds, refs in train_sources for i in set(refs)}
        for k, (name, vdocs, vrefs) in enumerate(val_sources):
            keep = [i for i in vrefs if hash(vdocs.chunks[i].tobytes()) not in train_h]
            report[name]["val_dropped_dup_of_train"] = len(vrefs) - len(keep)
            val_sources[k] = (name, vdocs, keep)
    elif (val_dir / "manifest.json").exists():
        val_h = set()
        vman = json.loads((val_dir / "manifest.json").read_text())
        for sh in vman["shards"]:
            toks = np.fromfile(val_dir / f"{sh['name']}.bin", dtype=np.uint16)
            off = np.fromfile(val_dir / f"{sh['name']}.idx", dtype=np.uint64).astype(np.int64)
            val_h |= {hash(toks[off[i]:off[i + 1]].tobytes()) for i in range(len(off) - 1)}
        for k, (name, ds, refs) in enumerate(train_sources):
            keep = [i for i in refs if hash(ds.chunks[i].tobytes()) not in val_h]
            report[name]["dropped_dup_of_val"] = len(refs) - len(keep)
            train_sources[k] = (name, ds, keep)
    out_dir = TOK_DIR / cfg["shard_dir"]
    manifest = write_mixed(out_dir, train_sources, tok, seed, {"config": cfg, "sources": report})
    if val_sources:
        vman = write_mixed(val_dir, val_sources, tok, seed + 99,
                           {"note": f"{val_frac:.0%} of the {cfg['shard_dir']} distribution, fixed for the project",
                            "sources": {n: int(sum(d.lengths[i] for i in r)) for n, d, r in val_sources}})
        log(f"  val: {vman['total_tokens'] / 1e6:.1f}M tokens in {len(vman['shards'])} shards")
    save_cursors(cfg, cursors)
    log(f"{cfg['shard_dir']}: {manifest['total_tokens'] / 1e9:.3f}B tokens, {manifest['total_docs']:,} docs, "
        f"{len(manifest['shards'])} shards in {time.time() - t0:.0f}s")
    return manifest


# ------------------------------------------------------------------- SFT


def build_sft(cfg, tok, cap):
    """Conversations with loss masks. Math samples are re-executed in the real sandbox."""
    from pagoda.data.render import tool_messages
    from pagoda.sandbox import numbers_equal, parse_number, run_python, single_number

    t0 = time.time()
    cursors = load_cursors(cfg)
    rng = random.Random(int(cfg.get("seed", 0)))
    val_frac = float(cfg.get("val_fraction", 0.02))
    train, val, report = [], [], {}
    for name, spec in cfg["sources"].items():
        pool = spec["pool"]
        renderer = Renderer(tok, spec["format"], spec)
        start = cursors.get(pool, 0) if spec.get("use_cursor") else 0
        checks = []
        for field, values in (spec.get("match") or {}).items():
            checks.append(lambda r, f=field, v=frozenset(values): r.get(f) in v)
        if spec.get("explain_only") or spec.get("not_explain"):
            want = bool(spec.get("explain_only"))
            checks.append(lambda r, want=want: bool(EXPLAIN_RE.match(r["messages"][0]["content"].strip())) == want)
        flt = (lambda r, checks=checks: all(c(r) for c in checks)) if checks else None
        if spec.get("first_turn_only"):
            inner = renderer.one

            def first_turn(rec, inner=inner):
                return inner({**rec, "messages": rec["messages"][:2]})
            renderer.one = first_turn
        docs, cursor, cnt = take(pool, renderer, cap, start, doc_target=int(spec["count"]), chunk=False,
                                 with_mask=True, record_filter=flt)
        items = list(zip(docs.chunks, docs.mask_chunks))
        # verify every tool-format sample end to end: the code runs and prints the reference answer
        if spec["format"] == "tool_chat":
            from concurrent.futures import ThreadPoolExecutor
            py_s, py_e = tok.special("<|python_start|>"), tok.special("<|python_end|>")

            def verify(item):
                ids_l = item[0].tolist()
                code = tok.decode(ids_l[ids_l.index(py_s) + 1: ids_l.index(py_e)])
                ans = parse_number(tok.decode(ids_l).rsplit("Answer:", 1)[-1].split("<|")[0])
                res = run_python(code)
                return res.ok and numbers_equal(single_number(res.output), ans)

            with ThreadPoolExecutor(14) as ex:
                verdicts = list(ex.map(verify, items))
            ok_items = [it for it, v in zip(items, verdicts) if v]
            bad = len(items) - len(ok_items)
            cnt["sandbox_verified"] = len(ok_items)
            cnt["sandbox_rejected"] = bad
            items = ok_items
        rng.shuffle(items)
        n_val = max(1, int(round(len(items) * val_frac)))
        val += items[:n_val]
        train += items[n_val:] * int(spec.get("repeat", 1))
        report[name] = {"pool": pool, "conversations": len(items), "val": n_val,
                        "tokens": int(sum(len(i) for i, _ in items)),
                        "supervised_tokens": int(sum(int(m.sum()) for _, m in items)), **dict(cnt)}
        log(f"  sft {name:16s} {len(items):6,} convs  {report[name]['tokens'] / 1e6:6.2f}M tokens  {dict(cnt)}")
    # SFT is the last consumer of shared pools: cursors are read but not advanced, so
    # rebuilding SFT always starts right after the stage-2 records
    out_dir = TOK_DIR / cfg.get("shard_dir", "sft")
    out = {}
    for split, items in (("train", train), ("val", val)):
        rng.shuffle(items)
        w = ShardWriter(out_dir, tok.bos_id, with_mask=True, prefix=split)
        for ids, mask in items:
            w.add(ids, mask)
        out[split] = w.close({"config": cfg, "sources": report}) if split == "train" else w.close({})
        # ShardWriter writes manifest.json; keep one per split
        (out_dir / "manifest.json").rename(out_dir / f"{split}_manifest.json")
    total = out["train"]["total_tokens"]
    log(f"sft: train {out['train']['total_docs']:,} convs / {total / 1e6:.1f}M tokens, "
        f"val {out['val']['total_docs']:,} convs, {time.time() - t0:.0f}s")
    return out
