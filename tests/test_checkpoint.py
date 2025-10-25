"""Full checkpoint round trip (weights + optimizer state) and rotation."""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten

from pagoda import checkpoint as C
from pagoda.model import GPT, GPTConfig, loss_fn
from pagoda.optim import build_optimizer

TC = {"matrix_lr": 0.02, "muon_momentum": 0.95, "adam_beta1": 0.9, "adam_beta2": 0.95, "adam_eps": 1e-8,
      "embed_lr_tied": 1e-3, "embed_lr_untied": 0.2, "head_lr_untied": 0.004}
CFG = dict(vocab_size=256, n_layer=2, d_model=64, n_head=1, mlp_hidden=192, seq_len=16)


def _train(m, opt, data, steps):
    lg = nn.value_and_grad(m, loss_fn)
    for _ in range(steps):
        _, g = lg(m, data[:, :-1], data[:, 1:])
        opt.update(m, g)
        mx.eval(m.parameters(), opt.state)


def test_roundtrip_and_rotation(tmp_path):
    mx.random.seed(0)
    data = mx.random.randint(0, 256, (2, 17))
    m = GPT(GPTConfig(**CFG))
    opt, _ = build_optimizer(m, TC)
    _train(m, opt, data, 3)
    for step in (1, 2, 3, 4):
        C.save_full(tmp_path, step, m, opt, {"model_config": m.cfg.to_dict(), "data_state": {"x": step}}, keep=3)
    assert [d.name for d in C._full_dirs(tmp_path)] == ["step_000002", "step_000003", "step_000004"]
    assert C.latest(tmp_path).name == "step_000004" and C.load_meta(C.latest(tmp_path))["data_state"] == {"x": 4}

    # continue A for 2 steps vs. restore into a fresh model B and do the same
    m2 = GPT(GPTConfig(**CFG))
    opt2, _ = build_optimizer(m2, TC)
    C.load_weights(m2, C.latest(tmp_path))
    C.load_optimizer(opt2, C.latest(tmp_path))
    _train(m, opt, data, 2)
    _train(m2, opt2, data, 2)
    for (k, a), (_, b) in zip(tree_flatten(m.parameters()), tree_flatten(m2.parameters())):
        assert np.allclose(np.array(a), np.array(b), atol=1e-6), k

    path = C.save_milestone(tmp_path, 5, m, {"model_config": m.cfg.to_dict()})
    assert mx.load(str(path))["wte"].dtype == mx.bfloat16
