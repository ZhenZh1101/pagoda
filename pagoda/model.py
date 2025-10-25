"""Decoder-only Transformer, plan section 4.

Llama/nanochat style, no attention variants:
  pre-RMSNorm (eps 1e-6, no affine) before attention, MLP and the output head;
  MHA with head_dim 64, QK-norm, RoPE theta 10000; SwiGLU MLP; no biases;
  tied input/output embedding (optional untied); tanh logit softcap 15;
  linear layers N(0, 0.02), attention/MLP output projections zero-initialised.

Precision (plan section 6): parameters are fp32 master weights; every matmul
casts its weight to the compute dtype (bf16 by default), so activations are
bf16 while gradients and optimizer state stay fp32. Logits are upcast to fp32
before the softcap and the cross-entropy.
"""

import math
from dataclasses import asdict, dataclass

import mlx.core as mx
import mlx.nn as nn

PRESETS = {
    # name: (n_layer, d_model, n_head, mlp_hidden)
    "d512": (12, 512, 8, 1408),
    "d576": (12, 576, 9, 1536),
    "d640": (12, 640, 10, 1728),   # 3-day default
    "d768": (12, 768, 12, 2048),   # 5-day default
    # tiny configs for smoke tests and data-mix ablations (plan section 14.4)
    "tiny": (4, 256, 4, 704),
    "d256": (12, 256, 4, 704),
}


@dataclass
class GPTConfig:
    vocab_size: int = 32768
    seq_len: int = 1024
    n_layer: int = 12
    d_model: int = 640
    n_head: int = 10
    mlp_hidden: int = 1728
    tie_embeddings: bool = True
    rope_theta: float = 10000.0
    norm_eps: float = 1e-6
    softcap: float = 15.0
    init_std: float = 0.02

    @classmethod
    def from_preset(cls, name, **overrides):
        n_layer, d_model, n_head, hidden = PRESETS[name]
        return cls(n_layer=n_layer, d_model=d_model, n_head=n_head, mlp_hidden=hidden, **overrides)

    @property
    def head_dim(self):
        return self.d_model // self.n_head

    def to_dict(self):
        return asdict(self)


def rms_norm(x, eps):
    return mx.fast.rms_norm(x, None, eps)


class Linear(nn.Module):
    """Bias-free linear layer with an fp32 master weight, computed in x.dtype."""

    def __init__(self, d_in, d_out, std):
        super().__init__()
        self.weight = mx.random.normal((d_out, d_in)) * std if std > 0 else mx.zeros((d_out, d_in))

    def __call__(self, x):
        return x @ self.weight.astype(x.dtype).T


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head, self.head_dim, self.eps, self.theta = cfg.n_head, cfg.head_dim, cfg.norm_eps, cfg.rope_theta
        self.wq = Linear(cfg.d_model, cfg.d_model, cfg.init_std)
        self.wk = Linear(cfg.d_model, cfg.d_model, cfg.init_std)
        self.wv = Linear(cfg.d_model, cfg.d_model, cfg.init_std)
        self.wo = Linear(cfg.d_model, cfg.d_model, 0.0)  # zero init

    def __call__(self, x, cache=None):
        B, T, _ = x.shape
        q = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        k = self.wk(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        v = self.wv(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        q, k = rms_norm(q, self.eps), rms_norm(k, self.eps)  # QK-norm
        offset = 0 if cache is None else cache.offset
        q = mx.fast.rope(q, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=offset)
        k = mx.fast.rope(k, self.head_dim, traditional=False, base=self.theta, scale=1.0, offset=offset)
        if cache is not None:
            k, v = cache.update(k, v)
        mask = "causal" if T > 1 else None
        y = mx.fast.scaled_dot_product_attention(q, k, v, scale=1.0 / math.sqrt(self.head_dim), mask=mask)
        return self.wo(y.transpose(0, 2, 1, 3).reshape(B, T, -1))


class MLP(nn.Module):
    """SwiGLU: down(silu(gate(x)) * up(x))."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w_gate = Linear(cfg.d_model, cfg.mlp_hidden, cfg.init_std)
        self.w_up = Linear(cfg.d_model, cfg.mlp_hidden, cfg.init_std)
        self.w_down = Linear(cfg.mlp_hidden, cfg.d_model, 0.0)  # zero init

    def __call__(self, x):
        return self.w_down(nn.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.eps = cfg.norm_eps
        self.attn = Attention(cfg)
        self.mlp = MLP(cfg)

    def __call__(self, x, cache=None):
        x = x + self.attn(rms_norm(x, self.eps), cache)
        return x + self.mlp(rms_norm(x, self.eps))


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig, compute_dtype=mx.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.compute_dtype = compute_dtype
        self.wte = mx.random.normal((cfg.vocab_size, cfg.d_model)) * cfg.init_std
        self.blocks = [Block(cfg) for _ in range(cfg.n_layer)]
        if not cfg.tie_embeddings:
            self.lm_head = mx.random.normal((cfg.vocab_size, cfg.d_model)) * cfg.init_std

    def logits(self, idx, caches=None):
        x = self.wte[idx].astype(self.compute_dtype)
        for i, block in enumerate(self.blocks):
            x = block(x, None if caches is None else caches[i])
        x = rms_norm(x, self.cfg.norm_eps)
        head = self.wte if self.cfg.tie_embeddings else self.lm_head
        logits = (x @ head.astype(x.dtype).T).astype(mx.float32)
        return self.cfg.softcap * mx.tanh(logits / self.cfg.softcap)

    def __call__(self, idx, caches=None):
        return self.logits(idx, caches)

    def num_params(self):
        from mlx.utils import tree_flatten
        flat = dict(tree_flatten(self.parameters()))
        total = sum(v.size for v in flat.values())
        embed = self.wte.size + (0 if self.cfg.tie_embeddings else self.lm_head.size)
        return {"total": total, "embedding": embed, "non_embedding": total - embed}

    def flops_per_token(self):
        """6N (non-embedding + head matmul) + attention T^2 term (plan section 15)."""
        c = self.cfg
        n = self.num_params()["non_embedding"] + c.vocab_size * c.d_model
        return 6 * n + 12 * c.n_layer * c.seq_len * c.d_model


def token_losses(model, inputs, targets):
    """Per-token fp32 cross entropy; targets == -1 are ignored (loss 0)."""
    logits = model(inputs)
    valid = targets >= 0
    ce = nn.losses.cross_entropy(logits, mx.where(valid, targets, 0), reduction="none")
    return ce * valid, valid


def loss_fn(model, inputs, targets):
    ce, valid = token_losses(model, inputs, targets)
    return ce.sum() / mx.maximum(valid.sum(), 1)


class KVCache:
    """Growing per-layer KV cache for generation (contexts are <= 1024 tokens)."""

    step = 256

    def __init__(self):
        self.keys = self.values = None
        self.offset = 0

    def update(self, k, v):
        B, H, T, D = k.shape
        if self.keys is None or self.offset + T > self.keys.shape[2]:
            n = ((self.offset + T + self.step - 1) // self.step) * self.step
            new_k, new_v = mx.zeros((B, H, n, D), k.dtype), mx.zeros((B, H, n, D), v.dtype)
            if self.keys is not None:
                new_k[:, :, : self.offset] = self.keys[:, :, : self.offset]
                new_v[:, :, : self.offset] = self.values[:, :, : self.offset]
            self.keys, self.values = new_k, new_v
        self.keys[:, :, self.offset : self.offset + T] = k
        self.values[:, :, self.offset : self.offset + T] = v
        self.offset += T
        return self.keys[:, :, : self.offset], self.values[:, :, : self.offset]


def make_caches(model):
    return [KVCache() for _ in range(model.cfg.n_layer)]
