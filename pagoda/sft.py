"""Stage 3: SFT (plan section 9).

Loss on assistant tokens only; conversations best-fit packed into 1024-token
rows (remaining space padded and masked); Muon + AdamW groups unchanged with
LR = 0.15 x the pretraining peak; 50 warmup steps then linear decay to 0;
64k tokens per step; 2 epochs at most -- if the SFT val loss rises in epoch 2
the epoch-1 weights are kept as the final model.
"""

import bisect
import json
import math
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx.utils import tree_map

from pagoda import checkpoint as C
from pagoda.common import CKPT_DIR, LOG_DIR, TOK_DIR, log, write_json
from pagoda.model import GPT, GPTConfig
from pagoda.optim import build_optimizer, linear_warmup_decay, set_lr, set_muon_momentum
from pagoda.schedule import parse_deadline
from pagoda.tokenizer import get_tokenizer
from pagoda.train import build_step_fns


class SFTData:
    """All conversations of one split (tokens + masks), packed into rows per epoch."""

    def __init__(self, split, seq_len, pad_id, data_dir=TOK_DIR / "sft"):
        d = Path(data_dir)
        man = json.loads((d / f"{split}_manifest.json").read_text())
        self.convs = []
        for s in man["shards"]:
            toks = np.fromfile(d / f"{s['name']}.bin", dtype=np.uint16)
            mask = np.fromfile(d / f"{s['name']}.mask", dtype=np.uint8)
            idx = np.fromfile(d / f"{s['name']}.idx", dtype=np.uint64).astype(np.int64)
            for i in range(len(idx) - 1):
                self.convs.append((toks[idx[i]:idx[i + 1]], mask[idx[i]:idx[i + 1]]))
        self.cap, self.pad = seq_len + 1, pad_id
        self.tokens = sum(len(t) for t, _ in self.convs)

    def rows(self, epoch, seed):
        """Best-fit packing of a seeded permutation; returns (ids, targets) arrays of shape (N, seq_len)."""
        order = np.random.default_rng([seed, epoch]).permutation(len(self.convs))
        free, bins = [], []  # free: sorted (space_left, bin_index)
        for k in order:
            n = len(self.convs[k][0])
            j = bisect.bisect_left(free, (n, -1))
            if j < len(free):
                space, b = free.pop(j)
                bins[b].append(k)
                if space - n > 0:
                    bisect.insort(free, (space - n, b))
            else:
                bins.append([k])
                if self.cap - n > 0:
                    bisect.insort(free, (self.cap - n, len(bins) - 1))
        ids = np.full((len(bins), self.cap), self.pad, dtype=np.int32)
        sup = np.zeros((len(bins), self.cap), dtype=bool)
        for b, members in enumerate(bins):
            pos = 0
            for k in members:
                t, m = self.convs[k]
                ids[b, pos:pos + len(t)] = t
                sup[b, pos:pos + len(t)] = m.astype(bool)
                pos += len(t)
        perm = np.random.default_rng([seed, epoch, 1]).permutation(len(bins))
        ids, sup = ids[perm], sup[perm]
        targets = np.where(sup[:, 1:], ids[:, 1:], -1)
        return ids[:, :-1], targets


def run_sft(args):
    mx.set_memory_limit(int(args.memory_limit_gb * 1024 ** 3))
    tok = get_tokenizer()
    src = Path(args.base) if args.base else C.latest(CKPT_DIR / args.base_run)
    if src is None:
        raise SystemExit(f"no base checkpoint found in {CKPT_DIR / args.base_run}")
    meta = C.load_meta(src)
    cfg = GPTConfig(**meta["model_config"])
    model = GPT(cfg)
    C.load_weights(model, src)
    log(f"SFT from {src} ({cfg.n_layer}x{cfg.d_model})")

    # "LR = lr_frac x the pretraining peak": take the peak from the base run (incl. any spike-recovery scale)
    base_args = meta.get("args", {})
    peak_scale = meta.get("lr_scale", 1.0)
    matrix_lr = args.matrix_lr if args.matrix_lr is not None else base_args.get("matrix_lr", 0.02) * peak_scale
    embed_lr = args.embed_lr if args.embed_lr is not None else base_args.get("embed_lr", 1e-3) * peak_scale
    tc = {"matrix_lr": matrix_lr * args.lr_frac, "muon_momentum": 0.95, "adam_beta1": 0.9, "adam_beta2": 0.95,
          "adam_eps": 1e-8, "embed_lr_tied": embed_lr * args.lr_frac,
          "embed_lr_untied": 0.2 * peak_scale * args.lr_frac, "head_lr_untied": 0.004 * peak_scale * args.lr_frac}
    log(f"SFT LRs: Muon {matrix_lr:g} x {args.lr_frac}, AdamW {embed_lr:g} x {args.lr_frac} (from the base run)")
    opt, groups = build_optimizer(model, tc)
    set_muon_momentum(opt, 0.95)
    token_bytes = mx.array(tok.token_bytes(), dtype=mx.float32)
    micro, apply, evaluate = build_step_fns(model, opt, args.clip, token_bytes, args.compile)

    pad = tok.special("<|assistant_end|>")
    data_dir = TOK_DIR / args.data
    train, val = SFTData("train", cfg.seq_len, pad, data_dir), SFTData("val", cfg.seq_len, pad, data_dir)
    B = args.micro_batch
    accum = max(1, args.batch_tokens // (B * cfg.seq_len))
    rows_per_epoch = len(train.rows(0, args.seed)[0])
    steps_per_epoch = rows_per_epoch // (B * accum)
    total_steps = steps_per_epoch * args.epochs
    log(f"SFT: {len(train.convs):,} train convs ({train.tokens / 1e6:.1f}M tokens) -> {rows_per_epoch:,} rows/epoch, "
        f"{steps_per_epoch} steps/epoch x {args.epochs}, {B * accum * cfg.seq_len:,} tokens/step; "
        f"val {len(val.convs):,} convs")

    run_dir = CKPT_DIR / args.run
    log_dir = LOG_DIR / args.run
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file, metrics = log_dir / "sft.log", log_dir / "metrics.jsonl"

    def val_loss():
        vx, vy = val.rows(0, 0)
        tot, n = 0.0, 0.0
        for i in range(0, len(vx), B):
            ce, cnt, _, _ = evaluate(mx.array(vx[i:i + B]), mx.array(vy[i:i + B]))
            mx.eval(ce, cnt)
            tot += ce.item(); n += cnt.item()
        return tot / max(n, 1)

    deadline = parse_deadline(args.deadline) if getattr(args, "deadline", "") else None

    def link_final(epoch):
        final = run_dir / "sft_final.safetensors"
        if final.exists() or final.is_symlink():
            final.unlink()
        final.symlink_to(f"sft_epoch{epoch}.safetensors")

    history = {"val": [], "epochs": []}
    v0 = val_loss()
    history["val"].append({"step": 0, "val_loss": v0})
    log(f"step 0 | sft val loss {v0:.4f}", log_file)
    step, t_start = 0, time.time()
    for epoch in range(args.epochs):
        t_epoch = time.time()
        x_all, y_all = train.rows(epoch, args.seed)
        for s in range(steps_per_epoch):
            set_lr(opt, groups, linear_warmup_decay(step, total_steps, args.warmup))
            t0 = time.time()
            acc, loss_sum, n_sum = None, 0.0, 0.0
            for a in range(accum):
                i = (s * accum + a) * B
                x, y = mx.array(x_all[i:i + B]), mx.array(y_all[i:i + B])
                loss, n, grads = micro(x, y, mx.array(1.0))
                acc = grads if acc is None else tree_map(mx.add, acc, grads)
                mx.eval(acc, loss, n)
                loss_sum += loss.item(); n_sum += n.item()
            norm = apply(tree_map(lambda g: g / max(n_sum, 1.0), acc))
            mx.eval(norm, model.parameters(), opt.state)
            step += 1
            if step % args.log_every == 0 or step == 1:
                tl = loss_sum / max(n_sum, 1)
                log(f"sft step {step}/{total_steps} (epoch {epoch + 1}) | loss {tl:.4f} | gnorm {norm.item():.3f} | "
                    f"{time.time() - t0:.2f}s/step", log_file)
                with open(metrics, "a") as f:
                    f.write(json.dumps({"kind": "sft_train", "step": step, "loss": tl, "grad_norm": norm.item()}) + "\n")
            if args.eval_every and step % args.eval_every == 0:
                v = val_loss()
                history["val"].append({"step": step, "val_loss": v})
                log(f"sft step {step} | val loss {v:.4f}", log_file)
        v = val_loss()
        history["val"].append({"step": step, "val_loss": v})
        history["epochs"].append({"epoch": epoch + 1, "step": step, "val_loss": v})
        path = run_dir / f"sft_epoch{epoch + 1}.safetensors"
        C.save_weights(path, model, {"model_config": cfg.to_dict(), "base": str(src), "epoch": epoch + 1,
                                     "step": step, "val_loss": v})
        log(f"epoch {epoch + 1} done | val loss {v:.4f} | saved {path}", log_file)
        link_final(min(history["epochs"], key=lambda e: e["val_loss"])["epoch"])  # a crash later keeps a final model
        write_json(log_dir / "sft_history.json", history)
        if epoch > 0 and v > history["epochs"][epoch - 1]["val_loss"]:
            log("val loss rose in this epoch: keeping the previous epoch as final (plan section 9)", log_file)
            break
        if deadline is not None and epoch + 1 < args.epochs and time.time() + 1.1 * (time.time() - t_epoch) > deadline:
            log(f"next epoch would end after the deadline ({time.ctime(deadline)}): stopping after epoch {epoch + 1}",
                log_file)
            history["stopped_for_deadline"] = epoch + 1
            break
    best = min(history["epochs"], key=lambda e: e["val_loss"])
    final = run_dir / "sft_final.safetensors"
    best_path = run_dir / f"sft_epoch{best['epoch']}.safetensors"
    link_final(best["epoch"])
    history["final"] = {"epoch": best["epoch"], "path": str(best_path)}
    write_json(log_dir / "sft_history.json", history)
    log(f"SFT done in {(time.time() - t_start) / 3600:.2f} h; final = epoch {best['epoch']} -> {final}", log_file)
