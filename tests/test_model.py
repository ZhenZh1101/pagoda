"""Model shape/param-count checks, KV-cache equivalence, one compiled optimizer step."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from pagoda.model import GPT, GPTConfig, loss_fn, make_caches
from pagoda.optim import Muon, build_optimizer, clip_by_global_norm, set_lr, wsd_multiplier

TC = {"matrix_lr": 0.02, "muon_momentum": 0.95, "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8,
      "embed_lr_tied": 1e-3, "embed_lr_untied": 0.2, "head_lr_untied": 0.004}


def test_param_counts_match_plan():
    for preset, nonembed, total in [("d640", 59.5e6, 80.5e6), ("d768", 84.9e6, 110.1e6)]:
        m = GPT(GPTConfig.from_preset(preset))
        n = m.num_params()
        assert abs(n["non_embedding"] - nonembed) / nonembed < 0.005, (preset, n)
        assert abs(n["total"] - total) / total < 0.005, (preset, n)


def test_kv_cache_matches_full_forward():
    mx.random.seed(0)
    cfg = GPTConfig(vocab_size=512, n_layer=2, d_model=128, n_head=2, mlp_hidden=320, seq_len=64)
    m = GPT(cfg, compute_dtype=mx.float32)
    # non-zero output projections so the test exercises attention
    for b in m.blocks:
        b.attn.wo.weight = mx.random.normal(b.attn.wo.weight.shape) * 0.02
        b.mlp.w_down.weight = mx.random.normal(b.mlp.w_down.weight.shape) * 0.02
    ids = mx.random.randint(0, 512, (1, 20))
    full = m(ids)
    caches = make_caches(m)
    a = m(ids[:, :12], caches)          # prefill
    b = m(ids[:, 12:15], caches)        # multi-token chunk on top of a cache (tool output injection)
    c = [m(ids[:, i:i + 1], caches) for i in range(15, 20)]
    inc = mx.concatenate([a, b] + c, axis=1)
    assert np.allclose(np.array(full), np.array(inc), atol=1e-4)


def test_compiled_step_reduces_loss():
    mx.random.seed(0)
    cfg = GPTConfig(vocab_size=256, n_layer=2, d_model=128, n_head=2, mlp_hidden=320, seq_len=32)
    m = GPT(cfg)
    opt, groups = build_optimizer(m, TC)
    assert isinstance(opt.optimizers[0], Muon)
    lg = nn.value_and_grad(m, loss_fn)
    state = [m.state, opt.state]

    def step(x, y):
        loss, g = lg(m, x, y)
        g, norm = clip_by_global_norm(g, 1.0)
        opt.update(m, g)
        return loss, norm

    cstep = mx.compile(step, inputs=state, outputs=state)
    data = mx.random.randint(0, 256, (4, 33))
    losses = []
    for i in range(30):
        set_lr(opt, groups, wsd_multiplier(i, 30, warmup_steps=3))
        loss, norm = cstep(data[:, :-1], data[:, 1:])
        mx.eval(loss, norm, m.parameters(), opt.state)
        losses.append(loss.item())
    assert losses[-1] < losses[0] - 1.0, losses
    # fp32 master weights survive bf16 compute
    assert m.wte.dtype == mx.float32 and m.blocks[0].attn.wq.weight.dtype == mx.float32


def test_wsd_schedule():
    total = 1000
    assert wsd_multiplier(0, total, 300) == 1 / 300
    assert wsd_multiplier(500, total, 300) == 1.0
    assert wsd_multiplier(799, total, 300) == 1.0
    assert 0.0 < wsd_multiplier(900, total, 300) < 1.0
    assert wsd_multiplier(999, total, 300) < 0.01
