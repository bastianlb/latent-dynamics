"""Optimizer diagnostics (--log_opt_diag), see SPEC.md "Optimizer diagnostics".

Everything here is side-effect free w.r.t. training: only torch.autograd.grad is used (never
.backward(), so no .grad is created/modified), the optimizer state is only read, the model is put in
eval mode (caller restores train mode), and the only random draw (initial power-iteration vector) uses a
dedicated torch.Generator, never the global/training RNGs.

Loss convention: the optimizer uses COUPLED L2 (weight_decay added to the gradient, Adam is not AdamW), i.e. it
minimises L(theta) + 0.5*wd*||theta_decay||^2 (decay = all params except the learnable l2_scale). The wd term has
Hessian wd*I. Diagnostics use this full loss on a fixed train batch (eval mode).
"""
from __future__ import annotations

import numpy as np
import torch

QUANTILES = (0.01, 0.10, 0.50, 0.90, 0.99)
N_POWER_ITERS = 20
FIXED_BATCH_MAX = 1024


def flat_params(params):
    return torch.cat([p.detach().reshape(-1) for p in params])


class OptDiag:
    def __init__(self, model, params, decay_params, weight_decay, loss_fn, batch, optimizer, adam, seed):
        self.model, self.params, self.wd = model, list(params), float(weight_decay)
        self.decay_ids = {id(p) for p in decay_params}
        self.loss_fn, self.batch, self.opt, self.adam = loss_fn, batch, optimizer, adam
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(int(seed) + 7000)
        self.numel = sum(p.numel() for p in self.params)
        self.v = None   # warm-start vectors (unit norm)
        self.vp = None
        self.prev_theta = None

    # -- bookkeeping called every optimizer step by train.py -------------------------------------
    def before_step(self):
        self.prev_theta = flat_params(self.params).clone()

    # -- internals --------------------------------------------------------------------------------
    def _full_loss(self):
        loss = self.loss_fn(self.batch)
        if self.wd > 0:
            loss = loss + 0.5 * self.wd * sum((p * p).sum() for p in self.params if id(p) in self.decay_ids)
        return loss

    def _adam_vhat(self):
        """Flat bias-corrected second moment, or None if the optimizer has not stepped yet."""
        chunks = []
        beta2 = self.opt.param_groups[0]["betas"][1]
        for p in self.params:
            st = self.opt.state.get(p, {})
            if "exp_avg_sq" not in st:
                return None
            step = float(st["step"])
            chunks.append((st["exp_avg_sq"].detach() / (1.0 - beta2 ** step)).reshape(-1))
        return torch.cat(chunks)

    def _power(self, hvp, v):
        if v is None:
            v = torch.randn(self.numel, generator=self.gen)
        v = v / v.norm().clamp_min(1e-30)
        lam, resid = 0.0, float("nan")
        for _ in range(N_POWER_ITERS):
            w = hvp(v)
            lam = float(torch.dot(v, w))
            wn = float(w.norm())
            resid = float((w - lam * v).norm()) / max(abs(lam), 1e-30)
            if wn == 0.0:
                break
            v = w / wn
        return lam, resid, v

    # -- main entry --------------------------------------------------------------------------------
    def compute(self, step):
        model = self.model
        was_training = model.training
        model.eval()
        out = {}
        theta = flat_params(self.params)
        out["opt_theta_norm"] = np.float32(theta.norm())
        out["opt_dtheta_norm"] = np.float32(
            float((theta - self.prev_theta).norm()) if self.prev_theta is not None else float("nan"))

        loss = self._full_loss()
        grads = torch.autograd.grad(loss, self.params, create_graph=True, allow_unused=True)
        gl = [g if g is not None else torch.zeros_like(p) for g, p in zip(grads, self.params)]
        out["opt_grad_norm"] = np.float32(sum(float((g.detach() ** 2).sum()) for g in gl) ** 0.5)

        shapes = [p.shape for p in self.params]
        sizes = [p.numel() for p in self.params]

        def hvp(v):
            vs = [c.reshape(s) for c, s in zip(torch.split(v, sizes), shapes)]
            gv = sum((g * c).sum() for g, c in zip(grads, vs) if g is not None and g.requires_grad)
            if not torch.is_tensor(gv) or not gv.requires_grad:
                return torch.zeros_like(v)
            hv = torch.autograd.grad(gv, self.params, retain_graph=True, allow_unused=True)
            return torch.cat([(h if h is not None else torch.zeros_like(p)).reshape(-1)
                              for h, p in zip(hv, self.params)]).detach()

        lam, resid, self.v = self._power(hvp, self.v)
        out["opt_lambda_max"] = np.float32(lam)
        out["opt_lambda_iters_resid"] = np.float32(resid)

        if self.adam:
            vhat = self._adam_vhat()
            nan = np.float32("nan")
            if vhat is None:
                out["opt_eff_step_q"] = np.full(len(QUANTILES), nan, np.float32)
                out["opt_vhat_mean"] = nan
                out["opt_lambda_max_precond"] = nan
                out["opt_lambda_max_precond_resid"] = nan
            else:
                g0 = self.opt.param_groups[0]
                eps, lr = g0["eps"], g0["lr"]
                denom = vhat.sqrt() + eps
                eff = lr / denom
                out["opt_eff_step_q"] = torch.quantile(eff, torch.tensor(QUANTILES, dtype=eff.dtype)).numpy().astype(np.float32)
                out["opt_vhat_mean"] = np.float32(vhat.mean())
                s = denom.reciprocal().sqrt()  # P^{1/2}, P = 1/(sqrt(vhat)+eps)
                lamp, residp, self.vp = self._power(lambda v: s * hvp(s * v), self.vp)
                out["opt_lambda_max_precond"] = np.float32(lamp)
                out["opt_lambda_max_precond_resid"] = np.float32(residp)
        del grads, gl, loss
        model.train(was_training)
        return out
