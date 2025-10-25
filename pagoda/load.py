"""Load a trained model for eval / chat: --source sft | base | <path>."""

from pathlib import Path

import mlx.core as mx

from pagoda import checkpoint as C
from pagoda.common import CKPT_DIR
from pagoda.model import GPT, GPTConfig


def resolve(source, run):
    if source == "sft":
        p = CKPT_DIR / run / "sft_final.safetensors"
        if not p.exists():
            raise SystemExit(f"{p} not found (run scripts.sft first)")
        return p.resolve()
    if source == "base":
        p = C.latest(CKPT_DIR / run)
        if p is None:
            raise SystemExit(f"no full checkpoint in {CKPT_DIR / run}")
        return p
    return Path(source)


def load_model(source, run, dtype=mx.bfloat16):
    path = resolve(source, run)
    meta = C.load_meta(path)
    cfg = GPTConfig(**meta["model_config"])
    model = GPT(cfg, compute_dtype=dtype)
    C.load_weights(model, path)
    return model, path, meta
