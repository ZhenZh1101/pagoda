"""Checkpoints with rotation (plan section 7, change #4).

ckpt/<run>/step_000123/           full checkpoint: fp32 weights, optimizer state,
    model.safetensors             meta.json (step, data cursors, config, loop stats)
    optim.safetensors
    meta.json
ckpt/<run>/milestones/step_000123.bf16.safetensors    weights only, every 10% of progress

Only the newest `keep` full checkpoints are kept. Writes go to a temp dir that is
renamed into place, so a crash mid-save never leaves a half-written checkpoint.
"""

import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten


def _full_dirs(run_dir):
    run_dir = Path(run_dir)
    if not run_dir.exists():
        return []
    dirs = [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("step_") and (d / "meta.json").exists()]
    return sorted(dirs, key=lambda d: int(d.name.split("_")[1]))


def latest(run_dir):
    dirs = _full_dirs(run_dir)
    return dirs[-1] if dirs else None


def save_full(run_dir, step, model, opt, meta, keep=3):
    run_dir = Path(run_dir)
    final = run_dir / f"step_{step:06d}"
    tmp = run_dir / f".tmp_step_{step:06d}"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "model.safetensors"), dict(tree_flatten(model.parameters())))
    if opt is not None:
        mx.save_safetensors(str(tmp / "optim.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "meta.json").write_text(json.dumps({**meta, "step": step}, indent=2))
    if final.exists():
        shutil.rmtree(final)
    os.replace(tmp, final)
    for old in _full_dirs(run_dir)[:-keep]:
        shutil.rmtree(old)
    return final


def save_milestone(run_dir, step, model, meta):
    d = Path(run_dir) / "milestones"
    d.mkdir(parents=True, exist_ok=True)
    weights = {k: v.astype(mx.bfloat16) for k, v in tree_flatten(model.parameters())}
    path = d / f"step_{step:06d}.bf16.safetensors"
    mx.save_safetensors(str(path), weights, metadata={"meta": json.dumps({**meta, "step": step})})
    return path


def save_weights(path, model, meta, dtype=mx.bfloat16):
    """Standalone weights file (e.g. the final SFT model)."""
    weights = {k: v.astype(dtype) for k, v in tree_flatten(model.parameters())}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(path), weights, metadata={"meta": json.dumps(meta)})


def load_meta(path):
    path = Path(path)
    if path.is_dir():
        return json.loads((path / "meta.json").read_text())
    _, metadata = mx.load(str(path), return_metadata=True)
    return json.loads(metadata.get("meta", "{}"))


def load_weights(model, path):
    """Load weights (full checkpoint dir or .safetensors file) into `model` in place, as fp32."""
    path = Path(path)
    file = path / "model.safetensors" if path.is_dir() else path
    weights = mx.load(str(file))
    model.update(tree_unflatten([(k, v.astype(mx.float32)) for k, v in weights.items()]))
    mx.eval(model.parameters())


def load_optimizer(opt, path):
    """Restore optimizer state in place (keeps the dict objects a compiled step captured)."""
    flat = mx.load(str(Path(path) / "optim.safetensors"))
    tree = tree_unflatten(list(flat.items()))
    for o, s in zip(opt.optimizers, tree["states"]):
        o.state.clear()
        o.state.update(s)
    mx.eval(opt.state)
