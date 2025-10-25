"""Muon + AdamW and the WSD schedule (plan section 7).

Muon: every 2D matrix inside the transformer blocks (q/k/v/o, MLP).
AdamW: the embedding (and the untied head, if any). With RMSNorm carrying no
affine weights there are no other 1D/scalar parameters.

Learning rate and Muon momentum live in the optimizer state as arrays, so the
schedule can change them every step without recompiling the compiled step.
"""

import math

import mlx.core as mx
import mlx.optimizers as optim


class Muon(optim.Optimizer):
    """Nesterov momentum + Newton-Schulz orthogonalisation (Keller Jordan / nanochat).

    Same update as mlx.optimizers.Muon, but momentum is part of the state and
    weight decay is not applied (plan: Muon weight decay 0).
    """

    def __init__(self, learning_rate, momentum=0.95, ns_steps=5):
        super().__init__()
        self._maybe_schedule("learning_rate", learning_rate)
        self.state["momentum"] = mx.array(momentum, dtype=mx.float32)
        self.ns_steps = ns_steps

    def init_single(self, parameter, state):
        state["v"] = mx.zeros_like(parameter)

    @staticmethod
    def newton_schulz(X, steps):
        a, b, c = (3.4445, -4.7750, 2.0315)
        transpose = X.shape[0] > X.shape[1]
        if transpose:
            X = X.T
        X = X / (mx.linalg.norm(X) + 1e-7)
        for _ in range(steps):
            A = X @ X.T
            B = mx.addmm(b * A, A, A, beta=1.0, alpha=c)
            X = mx.addmm(a * X, B, X, beta=1.0, alpha=1.0)
        return X.T if transpose else X

    def apply_single(self, gradient, parameter, state):
        mu = self.state["momentum"]
        v = mu * state["v"] + (1 - mu) * gradient
        state["v"] = v
        update = (1 - mu) * gradient + mu * v  # Nesterov
        update = self.newton_schulz(update, self.ns_steps)
        scale = max(1.0, parameter.shape[0] / parameter.shape[1]) ** 0.5
        return parameter - (self.learning_rate * scale).astype(parameter.dtype) * update


def is_muon_param(path, p):
    return path.startswith("blocks.") and p.ndim == 2


def build_optimizer(model, tc):
    """Returns (MultiOptimizer, {group: base_lr}). `tc` is the training config dict."""
    d = model.cfg.d_model
    width_scale = (768 / d) ** 0.5
    muon_lr = tc["matrix_lr"] * width_scale
    muon = Muon(learning_rate=muon_lr, momentum=tc["muon_momentum"])
    betas = (tc["adam_beta1"], tc["adam_beta2"])
    if model.cfg.tie_embeddings:
        adam = optim.AdamW(learning_rate=tc["embed_lr_tied"], betas=betas, eps=tc["adam_eps"],
                           weight_decay=0.0, bias_correction=True)
        groups = {"muon": muon_lr, "adam": tc["embed_lr_tied"]}
        opt = optim.MultiOptimizer([muon, adam], [is_muon_param])
    else:
        # nanochat-style untied groups: input embedding 0.2, output head 0.004, width-scaled
        emb_lr = tc["embed_lr_untied"] * width_scale
        head_lr = tc["head_lr_untied"] * width_scale
        emb = optim.AdamW(learning_rate=emb_lr, betas=betas, eps=tc["adam_eps"], weight_decay=0.0, bias_correction=True)
        head = optim.AdamW(learning_rate=head_lr, betas=betas, eps=tc["adam_eps"], weight_decay=0.0, bias_correction=True)
        groups = {"muon": muon_lr, "adam": emb_lr, "head": head_lr}
        opt = optim.MultiOptimizer([muon, emb, head], [is_muon_param, lambda path, p: path == "wte"])
    opt.init(model.trainable_parameters())
    return opt, groups


def set_lr(opt, groups, multiplier):
    for o, base in zip(opt.optimizers, groups.values()):
        o.learning_rate = base * multiplier


def set_muon_momentum(opt, value):
    opt.optimizers[0].state["momentum"] = mx.array(value, dtype=mx.float32)


def wsd_multiplier(step, total_steps, warmup_steps=300, decay_frac=0.2, decay_start=None):
    """Warmup -> constant -> linear decay to 0 over the last `decay_frac` of steps
    (or from an explicit `decay_start`, as set by the deadline planner)."""
    if decay_start is None:
        decay_start = decay_start_step(total_steps, decay_frac)
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < decay_start:
        return 1.0
    return max(0.0, (total_steps - step) / max(1, total_steps - decay_start))


def decay_start_step(total_steps, decay_frac=0.2):
    return int(round(total_steps * (1 - decay_frac)))


def muon_momentum(step, warmup=300, start=0.85, end=0.95):
    """nanochat's momentum warmup 0.85 -> 0.95 over the first 300 steps."""
    frac = min(step / warmup, 1.0)
    return (1 - frac) * start + frac * end


def linear_warmup_decay(step, total_steps, warmup_steps):
    """SFT schedule: linear warmup then linear decay to 0."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    return max(0.0, (total_steps - step) / max(1, total_steps - warmup_steps))


def global_grad_norm(grads):
    from mlx.utils import tree_flatten
    return mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for _, g in tree_flatten(grads)))


def clip_by_global_norm(grads, max_norm):
    from mlx.utils import tree_map
    norm = global_grad_norm(grads)
    scale = mx.minimum(1.0, max_norm / (norm + 1e-6))
    return tree_map(lambda g: g * scale, grads), norm


def round_steps(tokens, tokens_per_step):
    return max(1, int(math.ceil(tokens / tokens_per_step)))
