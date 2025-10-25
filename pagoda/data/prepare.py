"""Step 2-4 of the data pipeline: extract, filter, dedup, decontaminate (plan section 8).

Every source becomes raw/filtered/<pool>.jsonl in (approximately) random order,
so a prefix is a uniform sample; raw/filtered/stats.json records how many
documents each filter and the 13-gram decontamination removed.

Token-length limits that need our tokenizer (>8k-token code files, 512-token
replies, 300-token solutions, exact <50-token docs) are applied later, when the
documents are rendered and tokenized (pagoda.data.build).
"""

import hashlib
import io
import json
import multiprocessing as mp
import os
import random
import re
import time
from collections import Counter, defaultdict

import pyarrow.parquet as pq

from pagoda.common import FILTERED_DIR, RAW_DIR, iter_jsonl, log, write_json
from pagoda.data import decontam
from pagoda.data.convert import (gsm8k_cot, gsm8k_final, gsm8k_to_script, omi1_to_parts, omi2_cot,
                                 tinygsm_to_script)
from pagoda.sandbox import fast_exec_many, format_number, numbers_equal, parse_number, single_number

MIN_CHARS = 200          # ~50 tokens; the exact check happens at tokenization
SHUFFLE_BUFFER = 50_000
SFT_BUCKETS = 3          # first-user-message hash buckets (of 10) reserved for SFT
SFT_SUBSETS = {"smol-magpie-ultra", "smol-rewrite", "smol-summarize", "smol-constraints", "everyday-conversations"}
TEXTBOOK_FORMATS = {"textbook", "textbook_unconditionned_topic", "textbook_narrative", "textbook_academic",
                    "textbook_narrative_tone"}


def stable_bucket(text, n=10):
    return int(hashlib.md5(text.strip().lower().encode()).hexdigest(), 16) % n


def norm_q(text):
    return " ".join(text.lower().split())


class ShuffledWriter:
    """Writes records to raw/filtered/<name>.jsonl through a seeded shuffle buffer."""

    def __init__(self, name, seed=0):
        FILTERED_DIR.mkdir(parents=True, exist_ok=True)
        self.path = FILTERED_DIR / f"{name}.jsonl"
        self.tmp = self.path.with_suffix(".jsonl.tmp")
        self.f = open(self.tmp, "w")
        self.buf, self.rng, self.n = [], random.Random(seed), 0

    def add(self, rec):
        self.buf.append(rec)
        if len(self.buf) >= SHUFFLE_BUFFER:
            i = self.rng.randrange(len(self.buf))
            self.buf[i], self.buf[-1] = self.buf[-1], self.buf[i]
            self._write(self.buf.pop())

    def _write(self, rec):
        self.f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        self.n += 1

    def close(self):
        self.rng.shuffle(self.buf)
        for r in self.buf:
            self._write(r)
        self.f.close()
        os.replace(self.tmp, self.path)
        return self.n


# ----------------------------------------------------------------- workers
def _init_worker():
    decontam.build_index()


def _text_task(task):
    """Read one parquet row group / jsonl chunk and filter it. Returns (pool->records, counters)."""
    kind, path, arg = task
    out, cnt = defaultdict(list), Counter()
    if kind == "fineweb":
        rows = pq.ParquetFile(path).read_row_group(arg, columns=["text", "int_score", "token_count"]).to_pylist()
        is_second = "001_" in os.path.basename(path)
        for r in rows:
            cnt["in"] += 1
            if r["int_score"] < 3:
                cnt["score<3"] += 1; continue
            if r["token_count"] < 50:
                cnt["short"] += 1; continue
            if decontam.is_contaminated(r["text"]):
                cnt["contaminated"] += 1; continue
            pool = "fineweb_edu_hq" if (is_second and r["int_score"] >= 4) else "fineweb_edu"
            out[pool].append({"text": r["text"]})
    elif kind == "cosmopedia":
        rows = pq.ParquetFile(path).read_row_group(arg, columns=["text", "format"]).to_pylist()
        is_second = "00000-of" not in os.path.basename(path)  # textbook pool comes from files after the first
        for r in rows:
            cnt["in"] += 1
            if len(r["text"]) < MIN_CHARS:
                cnt["short"] += 1; continue
            if decontam.is_contaminated(r["text"]):
                cnt["contaminated"] += 1; continue
            textbook = r["format"] in TEXTBOOK_FORMATS
            pool = "cosmopedia_textbook" if (is_second and textbook and stable_bucket(r["text"][:500]) < 3) \
                else "cosmopedia"
            out[pool].append({"text": r["text"], "format": r["format"]})
    elif kind == "finemath":
        rows = pq.ParquetFile(path).read_row_group(arg, columns=["text"]).to_pylist()
        for r in rows:
            cnt["in"] += 1
            if len(r["text"]) < MIN_CHARS:
                cnt["short"] += 1; continue
            if decontam.is_contaminated(r["text"]):
                cnt["contaminated"] += 1; continue
            out["finemath"].append({"text": r["text"]})
    elif kind == "lines":  # list of already-parsed {"text": ...} records (DCLM, Stack-Edu)
        pool, recs = path, arg
        for r in recs:
            cnt["in"] += 1
            if len(r["text"]) < MIN_CHARS:
                cnt["short"] += 1; continue
            if decontam.is_contaminated(r["text"]):
                cnt["contaminated"] += 1; continue
            out[pool].append(r)
    else:
        raise ValueError(kind)
    return dict(out), cnt


def _chat_task(task):
    """Decontaminate a list of structured records; returns kept records."""
    pool, recs = task
    kept, cnt = [], Counter()
    for r in recs:
        cnt["decontam_checked"] += 1
        text = r.pop("_text")
        if decontam.is_contaminated(text):
            cnt["contaminated"] += 1
            continue
        kept.append(r)
    return pool, kept, cnt


def _run_text_tasks(pool, tasks, writers, counter):
    random.Random(0).shuffle(tasks)  # random chunk order + shuffle buffer -> prefixes are uniform samples
    t0 = time.time()
    for i, (out, cnt) in enumerate(pool.imap_unordered(_text_task, tasks)):
        for name, recs in out.items():
            for r in recs:
                writers[name].add(r)
        counter.update(cnt)
        if (i + 1) % 200 == 0:
            log(f"  {i + 1}/{len(tasks)} chunks, {time.time() - t0:.0f}s")


# ----------------------------------------------------------------- sources
def parquet_tasks(kind, pattern_dir):
    files = sorted(pattern_dir.rglob("*.parquet"))
    return [(kind, str(f), rg) for f in files for rg in range(pq.ParquetFile(f).num_row_groups)]


def prep_fineweb(pool, stats):
    writers = {"fineweb_edu": ShuffledWriter("fineweb_edu", 1), "fineweb_edu_hq": ShuffledWriter("fineweb_edu_hq", 2)}
    tasks = parquet_tasks("fineweb", RAW_DIR / "fineweb_edu")
    _run_text_tasks(pool, tasks, writers, stats)
    return {k: w.close() for k, w in writers.items()}


def prep_cosmopedia(pool, stats):
    writers = {"cosmopedia": ShuffledWriter("cosmopedia", 3), "cosmopedia_textbook": ShuffledWriter("cosmopedia_textbook", 4)}
    _run_text_tasks(pool, parquet_tasks("cosmopedia", RAW_DIR / "cosmopedia_v2"), writers, stats)
    return {k: w.close() for k, w in writers.items()}


def prep_finemath(pool, stats):
    writers = {"finemath": ShuffledWriter("finemath", 5)}
    _run_text_tasks(pool, parquet_tasks("finemath", RAW_DIR / "finemath"), writers, stats)
    return {k: w.close() for k, w in writers.items()}


def _line_chunks(pool_name, records, size=2000):
    for i in range(0, len(records), size):
        yield ("lines", pool_name, records[i:i + size])


def prep_dclm(pool, stats):
    import zstandard
    recs = []
    for f in sorted((RAW_DIR / "dclm").rglob("*.jsonl.zst")):
        with open(f, "rb") as fh:
            reader = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
            for line in reader:
                recs.append({"text": json.loads(line)["text"]})
    writers = {"dclm": ShuffledWriter("dclm", 6)}
    _run_text_tasks(pool, list(_line_chunks("dclm", recs)), writers, stats)
    return {k: w.close() for k, w in writers.items()}


def prep_stack(pool, stats):
    src = RAW_DIR / "stack_edu" / "python_files.jsonl"
    recs, seen = [], set()
    with open(src) as f:
        for line in f:
            r = json.loads(line)
            h = hashlib.md5(r["text"].encode()).hexdigest()
            if h in seen:
                stats["exact_dup"] += 1
                continue
            seen.add(h)
            recs.append({"text": r["text"], "repo": r["repo"], "path": r["path"]})
    writers = {"stack_python": ShuffledWriter("stack_python", 7)}
    _run_text_tasks(pool, list(_line_chunks("stack_python", recs)), writers, stats)
    return {k: w.close() for k, w in writers.items()}


def _decontam_records(pool, name, recs, stats, seed):
    """recs carry a '_text' field used for decontamination; writes raw/filtered/<name>.jsonl."""
    w = ShuffledWriter(name, seed)
    chunks = [(name, recs[i:i + 2000]) for i in range(0, len(recs), 2000)]
    for _, kept, cnt in pool.imap(_chat_task, chunks):
        for r in kept:
            w.add(r)
        stats.update(cnt)
    return {name: w.close()}


def prep_tinygsm(pool, stats):
    st = stats
    seen, items = set(), []
    for f in sorted((RAW_DIR / "tinygsm").rglob("*.parquet")):
        for r in pq.read_table(f).to_pylist():
            st["in"] += 1
            q = norm_q(r["question"])
            if q in seen:
                st["dup_question"] += 1
                continue
            seen.add(q)
            script = tinygsm_to_script(r["code"])
            if script is None:
                st["unconvertible"] += 1
                continue
            items.append((re.sub(r"^(Student|Question|Problem)\s*:\s*", "", r["question"].strip()), script))
    log(f"  tinygsm: executing {len(items)} scripts")
    outputs = fast_exec_many([s for _, s in items])
    recs = []
    for (q, script), out in zip(items, outputs):
        val = single_number(out) if out is not None else None
        if val is None:
            st["not_single_number"] += 1
            continue
        recs.append({"question": q, "script": script, "output": out.strip(), "answer": format_number(val),
                     "_text": q})
    return _decontam_records(pool, "tinygsm", recs, stats, 8)


def _valid_chat(messages):
    if len(messages) < 2:
        return False
    for i, m in enumerate(messages):
        if m["role"] != ("user" if i % 2 == 0 else "assistant") or not m["content"].strip():
            return False
    return messages[-1]["role"] == "assistant"


def prep_smoltalk(pool, stats):
    """Chat-format pretraining pool (stage 1 text, stage 2 chat) from SmolTalk v1 'all'."""
    st = stats
    recs, seen = [], set()
    for f in sorted((RAW_DIR / "smoltalk" / "data" / "all").glob("train-*.parquet")):
        for r in pq.read_table(f, columns=["messages", "source"]).to_pylist():
            st["in"] += 1
            msgs, source = r["messages"], r["source"]
            if source in ("apigen-80k", "longalign", "everyday-conversations"):
                st["excluded_subset"] += 1; continue
            if msgs and msgs[0]["role"] == "system":
                st["system_prompt"] += 1; continue
            if not _valid_chat(msgs):
                st["malformed"] += 1; continue
            first = msgs[0]["content"]
            if source in SFT_SUBSETS and stable_bucket(first) < SFT_BUCKETS:
                st["reserved_for_sft"] += 1; continue
            key = norm_q(first)
            if key in seen:
                st["dup_prompt"] += 1; continue
            seen.add(key)
            recs.append({"messages": [{"role": m["role"], "content": m["content"]} for m in msgs],
                         "source": source, "_text": "\n".join(m["content"] for m in msgs)})
    return _decontam_records(pool, "smoltalk_chat", recs, stats, 9)


def prep_sft_chat(pool, stats):
    """SFT chat pool: SmolTalk2 non-reasoning subsets + SmolTalk v1 smol-constraints (reserved buckets only)."""
    st = stats
    sft = RAW_DIR / "smoltalk2" / "SFT"
    files = [
        ("everyday-conversations", sft / "smoltalk_smollm3_everyday_conversations_no_think-00000-of-00001.parquet"),
        *[("smol-magpie-ultra", f) for f in sorted(sft.glob("smoltalk_smollm3_smol_magpie_ultra_no_think-*.parquet"))],
        ("smol-rewrite", sft / "smoltalk_smollm3_smol_rewrite_no_think-00000-of-00001.parquet"),
        ("smol-summarize", sft / "smoltalk_smollm3_smol_summarize_no_think-00000-of-00001.parquet"),
        ("smol-constraints", RAW_DIR / "smoltalk/data/smol-constraints/train-00000-of-00001.parquet"),
    ]
    recs, seen = [], set()
    for subset, path in files:
        for r in pq.read_table(path, columns=["messages"]).to_pylist():
            st["in"] += 1
            msgs = [{"role": m["role"], "content": m["content"]} for m in r["messages"]]
            if msgs and msgs[0]["role"] == "system":
                st["system_prompt"] += 1; continue
            if not _valid_chat(msgs):
                st["malformed"] += 1; continue
            first = msgs[0]["content"]
            if subset != "everyday-conversations" and stable_bucket(first) >= SFT_BUCKETS:
                st["not_sft_bucket"] += 1; continue
            # dedup on every user turn: everyday conversations all open with "Hi"/"Hello"
            key = norm_q(" || ".join(m["content"] for m in msgs if m["role"] == "user"))
            if key in seen:
                st["dup_prompt"] += 1; continue
            seen.add(key)
            recs.append({"messages": msgs, "subset": subset, "_text": "\n".join(m["content"] for m in msgs)})
    return _decontam_records(pool, "sft_chat", recs, stats, 10)


def prep_omi2(pool, stats):
    st = stats
    recs, seen = [], set()
    for f in sorted((RAW_DIR / "openmathinstruct2").rglob("*.parquet")):
        for r in pq.read_table(f, columns=["problem", "generated_solution", "expected_answer"]).to_pylist():
            st["in"] += 1
            q = norm_q(r["problem"])
            if q in seen:
                st["dup_question"] += 1; continue
            seen.add(q)
            cot = omi2_cot(r["generated_solution"], r["expected_answer"])
            if cot is None:
                st["non_numeric_answer"] += 1; continue
            recs.append({"question": r["problem"].strip(), "cot": cot, "_text": r["problem"]})
    return _decontam_records(pool, "omi2_cot", recs, stats, 11)


def prep_gsm8k_train(pool, stats):
    st = stats
    rows = pq.read_table(RAW_DIR / "gsm8k/main/train-00000-of-00001.parquet").to_pylist()
    scripts = [gsm8k_to_script(r["answer"]) for r in rows]
    outs = fast_exec_many([s or "" for s in scripts])
    recs = []
    for r, s, out in zip(rows, scripts, outs):
        st["in"] += 1
        final = gsm8k_final(r["answer"])
        rec = {"question": r["question"].strip(), "cot": gsm8k_cot(r["answer"]), "answer": format_number(final),
               "_text": r["question"]}
        val = single_number(out) if (s and out is not None) else None
        if numbers_equal(val, final):
            rec.update(script=s, output=out.strip())
        else:
            st["tool_conversion_failed"] += 1
        recs.append(rec)
    return _decontam_records(pool, "gsm8k_train", recs, stats, 12)


def prep_omi1(pool, stats):
    st = stats
    rows, seen, items = [], set(), []
    with open(RAW_DIR / "openmathinstruct1" / "gsm8k_code.jsonl") as f:
        for line in f:
            r = json.loads(line)
            st["in"] += 1
            q = norm_q(r["question"])
            if q in seen:
                st["dup_question"] += 1; continue
            parts = omi1_to_parts(r["generated_solution"])
            ans = parse_number(r["expected_answer"])
            if parts is None or ans is None:
                st["unconvertible"] += 1; continue
            seen.add(q)
            items.append((r["question"].strip(), parts, ans))
    outs = fast_exec_many([p[1] for _, p, _ in items])
    recs = []
    for (q, (pre, script, post), ans), out in zip(items, outs):
        val = single_number(out) if out is not None else None
        if not numbers_equal(val, ans):
            st["wrong_or_failed_exec"] += 1; continue
        recs.append({"question": q, "analysis": pre if len(pre) <= 200 else "", "script": script,
                     "output": out.strip(), "closing": post if len(post) <= 300 else "",
                     "answer": format_number(ans), "_text": q})
    return _decontam_records(pool, "omi1_code", recs, stats, 13)


def prep_mc(pool, stats):
    st = stats
    recs = []
    for r in pq.read_table(RAW_DIR / "mmlu/auxiliary_train/train-00000-of-00001.parquet").to_pylist():
        r = r["train"]
        st["in"] += 1
        if not (2 <= len(r["choices"]) <= 5):
            st["bad_choices"] += 1; continue
        recs.append({"question": r["question"].strip(), "choices": r["choices"], "answer": int(r["answer"]),
                     "source": "mmlu_aux", "_text": r["question"] + " " + " ".join(r["choices"])})
    for sub in ("ARC-Easy", "ARC-Challenge"):
        for r in pq.read_table(RAW_DIR / f"arc/{sub}/train-00000-of-00001.parquet").to_pylist():
            st["in"] += 1
            labels = r["choices"]["label"]
            if r["answerKey"] not in labels:
                st["bad_choices"] += 1; continue
            recs.append({"question": r["question"].strip(), "choices": r["choices"]["text"],
                         "answer": labels.index(r["answerKey"]), "source": sub.lower(),
                         "_text": r["question"] + " " + " ".join(r["choices"]["text"])})
    return _decontam_records(pool, "mc", recs, stats, 14)


def prep_identity(pool, stats):
    """Pagoda-0.1 identity conversations + pretraining texts (identity/, made by scripts.gen_identity).

    Conversations are re-checked against persona.yaml and the held-out eval set here,
    so a hand-edited identity/conversations.jsonl can't leak eval prompts into training.
    """
    from pagoda.identity import CONVERSATIONS, PRETRAIN_TEXTS, conversation_problems
    st = stats
    chats, texts = [], []
    for r in iter_jsonl(CONVERSATIONS):
        st["in"] += 1
        if conversation_problems(r):
            st["invalid"] += 1; continue
        chats.append({"messages": r["messages"], "category": r["category"], "id": r["id"],
                      "_text": "\n".join(m["content"] for m in r["messages"])})
    for r in iter_jsonl(PRETRAIN_TEXTS):
        st["in"] += 1
        texts.append({"text": r["text"], "genre": r["genre"], "_text": r["text"]})
    return {**_decontam_records(pool, "identity_chat", chats, stats, 15),
            **_decontam_records(pool, "identity_text", texts, stats, 16)}


PREPS = {
    "fineweb": prep_fineweb, "cosmopedia": prep_cosmopedia, "finemath": prep_finemath, "dclm": prep_dclm,
    "stack": prep_stack, "tinygsm": prep_tinygsm, "smoltalk": prep_smoltalk, "sft_chat": prep_sft_chat,
    "omi2": prep_omi2, "gsm8k": prep_gsm8k_train, "omi1": prep_omi1, "mc": prep_mc, "identity": prep_identity,
}


def run(only=None, workers=None):
    stats_path = FILTERED_DIR / "stats.json"
    all_stats = json.loads(stats_path.read_text()) if stats_path.exists() else {}
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers or max(1, os.cpu_count() - 2), initializer=_init_worker) as pool:
        for name, fn in PREPS.items():
            if only and name not in only:
                continue
            t0 = time.time()
            log(f"prepare {name} ...")
            counter = Counter()
            written = fn(pool, counter)
            entry = {"pools": written, **dict(counter)}
            if counter.get("in"):
                entry["contaminated_frac"] = round(counter.get("contaminated", 0) / counter["in"], 6)
                entry["kept_frac"] = round(sum(written.values()) / counter["in"], 6)
            entry["seconds"] = round(time.time() - t0)
            all_stats[name] = entry
            write_json(stats_path, all_stats)
            log(f"prepare {name}: {entry}")
    return all_stats
