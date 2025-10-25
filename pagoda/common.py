"""Paths, logging and small utilities shared by every Pagoda script.

Data layout (plan section 2), rooted at $PAGODA_HOME (defaults to the repo directory):

    raw/        original downloads; raw/filtered/*.jsonl after filtering
    tok/        uint16 token shards: stage1/ stage2/ sft/ val/
    tokenizer/  tokenizer.pkl, token_bytes.npy, training sample
    ckpt/       rolling full checkpoints + bf16 milestones
    logs/       training logs, eval results, samples
    sandbox/    scratch dirs for the Python executor
"""

import json
import os
import sys
import time
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("PAGODA_HOME", CODE_ROOT)).expanduser().resolve()

RAW_DIR = DATA_ROOT / "raw"
FILTERED_DIR = RAW_DIR / "filtered"
TOK_DIR = DATA_ROOT / "tok"
TOKENIZER_DIR = DATA_ROOT / "tokenizer"
CKPT_DIR = DATA_ROOT / "ckpt"
LOG_DIR = DATA_ROOT / "logs"
SANDBOX_DIR = DATA_ROOT / "sandbox"
CONFIG_DIR = CODE_ROOT / "configs"
PROMPT_DIR = CODE_ROOT / "prompts"


def ensure_dirs():
    for d in (RAW_DIR, FILTERED_DIR, TOK_DIR, TOKENIZER_DIR, CKPT_DIR, LOG_DIR, SANDBOX_DIR):
        d.mkdir(parents=True, exist_ok=True)


def log(msg, file=None):
    """Timestamped print that is also appended to `file` when given."""
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    if file is not None:  # the file first: stdout may be a pipe whose reader (tee) is already gone
        with open(file, "a") as f:
            f.write(line + "\n")
    try:
        print(line, flush=True)
    except (BrokenPipeError, OSError):
        pass


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def iter_jsonl(path):
    with open(path) as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_yaml(path):
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def human(n):
    for unit in ("", "K", "M", "B", "T"):
        if abs(n) < 1000:
            return f"{n:.1f}{unit}" if unit else f"{n:.0f}"
        n /= 1000
    return f"{n:.1f}P"


def die(msg):
    print(f"error: {msg}", file=sys.stderr)
    sys.exit(1)
