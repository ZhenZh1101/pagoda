"""Step 1 of the data pipeline: fetch raw files into raw/ (plan section 8).

    python -m scripts.download                 # everything
    python -m scripts.download --only gsm8k arc
    python -m scripts.download --skip-stack-edu --skip-omi1 --skip-core

Re-running is cheap: files already present are skipped.
"""

import argparse
import gzip
import json
import os
import random
import threading
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed

from pagoda.common import RAW_DIR, ensure_dirs, log, write_json
from pagoda.data import sources as S

os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")


def download_hf(names, workers):
    from huggingface_hub import hf_hub_download

    jobs = [(name, repo, f) for name, (repo, files) in S.HF_FILES.items() if name in names for f in files]

    def fetch(job):
        name, repo, filename = job
        dest = RAW_DIR / name / filename
        if dest.exists() and dest.stat().st_size > 0:
            return name, filename, dest.stat().st_size, "cached"
        for attempt in range(5):
            try:
                path = hf_hub_download(repo, filename, repo_type="dataset", local_dir=RAW_DIR / name)
                return name, filename, os.path.getsize(path), "ok"
            except Exception as e:  # network hiccups: retry with backoff
                if attempt == 4:
                    raise
                log(f"retry {repo}/{filename}: {e}")
                time.sleep(5 * (attempt + 1))

    total = 0
    with ThreadPoolExecutor(workers) as ex:
        for fut in as_completed([ex.submit(fetch, j) for j in jobs]):
            name, filename, size, status = fut.result()
            total += size
            log(f"{status:6s} {name}/{filename}  {size / 1e6:.0f} MB")
    log(f"HF files: {len(jobs)} files, {total / 1e9:.2f} GB")


def download_omi1():
    """Stream OpenMathInstruct-1 and keep GSM8K-derived code solutions only."""
    from huggingface_hub import HfFileSystem

    out = RAW_DIR / "openmathinstruct1" / "gsm8k_code.jsonl"
    if out.exists():
        log(f"cached {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    kept = seen = 0
    tmp = out.with_suffix(".tmp")
    with fs.open(f"datasets/{S.OMI1_REPO}/{S.OMI1_FILE}", "rb", block_size=8 << 20) as f, open(tmp, "w") as w:
        for raw in f:
            seen += 1
            row = json.loads(raw)
            if row.get("dataset") != "gsm8k" or "<llm-code>" not in row.get("generated_solution", ""):
                continue
            w.write(json.dumps({k: row[k] for k in ("question", "generated_solution", "expected_answer")}) + "\n")
            kept += 1
            if kept >= S.OMI1_MAX_ROWS:
                break
    os.replace(tmp, out)
    log(f"OpenMathInstruct-1: kept {kept} gsm8k code solutions out of {seen} rows streamed")


def download_stack_edu(workers=64, seed=1234):
    """Pick a random Python subset of the Stack-Edu index and fetch file bodies."""
    import pyarrow.parquet as pq

    index = RAW_DIR / "stack_edu" / S.HF_FILES["stack_edu"][1][0]
    out = RAW_DIR / "stack_edu" / "python_files.jsonl"
    if not index.exists():
        raise SystemExit("Stack-Edu index missing: run with --only stack_edu first")
    table = pq.read_table(index, columns=["blob_id", "repo_name", "path", "length_bytes", "int_score", "license_type"])
    rows = table.to_pylist()
    rng = random.Random(seed)
    rng.shuffle(rows)
    picked, nbytes = [], 0
    for r in rows:
        if 200 <= r["length_bytes"] <= S.STACK_EDU_MAX_BYTES:
            picked.append(r)
            nbytes += r["length_bytes"]
            if nbytes >= S.STACK_EDU_TARGET_BYTES:
                break
    done = set()
    if out.exists():
        with open(out) as f:
            done = {json.loads(l)["blob_id"] for l in f if l.strip()}
    todo = [r for r in picked if r["blob_id"] not in done]
    log(f"Stack-Edu: {len(picked)} files picked ({nbytes / 1e6:.0f} MB), {len(todo)} still to fetch")

    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0, "bad_utf8": 0}

    def fetch(r):
        url = S.SWH_URL.format(blob_id=r["blob_id"])
        for attempt in range(4):
            try:
                with urllib.request.urlopen(url, timeout=30) as resp:
                    data = gzip.decompress(resp.read())
                break
            except Exception:
                if attempt == 3:
                    return r, None
                time.sleep(1 + attempt * 2)
        try:
            return r, data.decode("utf-8")
        except UnicodeDecodeError:
            return r, False

    with open(out, "a") as w, ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, r) for r in todo]
        for i, fut in enumerate(as_completed(futs)):
            r, text = fut.result()
            with lock:
                if text is None:
                    stats["fail"] += 1
                elif text is False:
                    stats["bad_utf8"] += 1
                else:
                    stats["ok"] += 1
                    w.write(json.dumps({"blob_id": r["blob_id"], "repo": r["repo_name"], "path": r["path"],
                                        "license_type": r["license_type"], "text": text}) + "\n")
            if (i + 1) % 10000 == 0:
                log(f"  Stack-Edu fetched {i + 1}/{len(todo)} {stats}")
    log(f"Stack-Edu done: {stats}")


def download_core_bundle():
    dest = RAW_DIR / "eval_bundle"
    if (dest / "core.yaml").exists():
        log(f"cached {dest}")
        return
    zpath = RAW_DIR / "eval_bundle.zip"
    log(f"downloading {S.CORE_BUNDLE_URL}")
    urllib.request.urlretrieve(S.CORE_BUNDLE_URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        for member in z.namelist():  # refuse path traversal
            if member.startswith("/") or ".." in member.split("/"):
                raise SystemExit(f"suspicious path in zip: {member}")
        z.extractall(RAW_DIR)
    zpath.unlink()
    log(f"CORE bundle extracted to {dest}")


def write_manifest():
    files = {}
    for p in sorted(RAW_DIR.rglob("*")):
        if p.is_file() and "filtered" not in p.parts and ".cache" not in p.parts:
            files[str(p.relative_to(RAW_DIR))] = p.stat().st_size
    write_json(RAW_DIR / "MANIFEST.json", {"total_bytes": sum(files.values()), "files": files})
    log(f"raw/ total {sum(files.values()) / 1e9:.2f} GB in {len(files)} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", nargs="*", help=f"subset of {list(S.HF_FILES)} (+ omi1, stack_edu_blobs, core)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-omi1", action="store_true")
    ap.add_argument("--skip-stack-edu", action="store_true", help="skip fetching Stack-Edu file bodies")
    ap.add_argument("--skip-core", action="store_true")
    args = ap.parse_args()
    ensure_dirs()

    only = set(args.only) if args.only else None
    hf_names = [n for n in S.HF_FILES if only is None or n in only]
    if hf_names:
        download_hf(hf_names, args.workers)
    if not args.skip_omi1 and (only is None or "omi1" in only):
        download_omi1()
    if not args.skip_stack_edu and (only is None or "stack_edu_blobs" in only):
        download_stack_edu()
    if not args.skip_core and (only is None or "core" in only):
        download_core_bundle()
    write_manifest()


if __name__ == "__main__":
    main()
