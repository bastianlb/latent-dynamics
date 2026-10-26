"""Dense per-step optimizer-state logging (--log_opt_dense), see SPEC.md "Dense optimizer-state logging".

Read-only w.r.t. training: it only READS optimizer state (exp_avg, exp_avg_sq, step) and p.grad after
optimizer.step() (called before the next zero_grad), clones parameters before the step, never touches the global
or any other RNG, never calls backward / autograd, never changes thread settings. It only ever does out-of-place
tensor math on copies.

Definitions (t = optimizer step count after the step, as Adam used them):
  vhat = exp_avg_sq/(1-b2^t), mhat = exp_avg/(1-b1^t), g' = p.grad + wd*theta_before (coupled L2; wd=0 for
  parameters outside the decay group), denom = sqrt(vhat)+eps, update = lr*mhat/denom, a = lr/denom,
  r = |mhat|/denom.

Outputs
  run_dir/dense_scalars.npy          float32 [total_steps, ncols]; row t-1 = step t. Columns: dense_scalars_cols.json
  run_dir/dense_meta.json            mode, trigger step, bytes written, guard messages
  dump_dir/vhat_bf16.npz             steps (int64), vhat_u16 [n,P] (bfloat16 bit patterns in uint16)
  dump_dir/c_dump.npz                steps, vhat_u16, mhat_u16, gp_u16 [n,P], trig_step, no_trigger, mode, P
Quantiles use nearest rank on a sorted copy: sorted[round(q*(n-1))].
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

DEFAULT_DUMP_ROOT = "/data2/lbastian/latent-dynamics/opt_dense"
VQ = (1e-4, 1e-3, 1e-2, 0.1, 0.5)
RQ = (1e-4, 1e-3, 1e-2, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999)
SCOPE_PREFIX = {"fc1": ("encoder.net.1.",), "fc2": ("encoder.net.3.",), "fc3": ("encoder.net.5.",),
                "probe": ("probe.",)}
SCOPES = ("global", "fc1", "fc2", "fc3", "probe")
GLOBAL_COLS = ("loss", "gprime_norm", "grad_norm", "theta_norm", "dtheta_norm", "step", "lr")
GB = 1e9


def _col_names():
    cols = list(GLOBAL_COLS)
    for s in SCOPES:
        cols += [f"{s}/vhat_q{q:g}" for q in VQ]
        cols += [f"{s}/r_q{q:g}" for q in RQ]
        cols += [f"{s}/frac_sqrtv_lt_eps", f"{s}/frac_sqrtv_lt_10eps", f"{s}/max_a",
                 f"{s}/max_dtheta_over_lr", f"{s}/frac_dtheta_gt_lr", f"{s}/frac_dtheta_gt_2lr",
                 f"{s}/dtheta_norm"]
    return cols


def _bf16_bits(x: torch.Tensor) -> np.ndarray:
    return x.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)


def dir_size(path: Path) -> int:
    tot = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                tot += os.path.getsize(os.path.join(root, f))
            except OSError:
                pass
    return tot


class OptDense:
    def __init__(self, model, params, decay_params, weight_decay, optimizer, cfg, total_steps, run_dir, dump_dir):
        self.opt, self.wd, self.cfg = optimizer, float(weight_decay), cfg
        self.params = list(params)
        self.total_steps = int(total_steps)
        self.run_dir, self.dump_dir = Path(run_dir), Path(dump_dir)
        self.cols = _col_names()
        self.ci = {c: i for i, c in enumerate(self.cols)}
        self.mat = np.full((self.total_steps, len(self.cols)), np.nan, np.float32)
        decay_ids = {id(p) for p in decay_params}
        names = {id(p): n for n, p in model.named_parameters()}
        sizes = [p.numel() for p in self.params]
        self.P = sum(sizes)
        self.wd_vec = torch.cat([torch.full((p.numel(),), self.wd if id(p) in decay_ids else 0.0)
                                 for p in self.params])
        off, self.slices = 0, {s: [] for s in SCOPES[1:]}
        for p, n in zip(self.params, sizes):
            nm = names[id(p)]
            for s, pre in SCOPE_PREFIX.items():
                if nm.startswith(pre):
                    self.slices[s].append((off, off + n))
            off += n
        self.scope_idx = {}
        for s, rng in self.slices.items():  # contiguous (params of a layer are adjacent) -> single slice
            assert rng and all(rng[i][1] == rng[i + 1][0] for i in range(len(rng) - 1)), f"scope {s} not contiguous"
            self.scope_idx[s] = (rng[0][0], rng[-1][1])
        self.scope_idx["global"] = (0, self.P)
        self.prev_theta = None
        g0 = optimizer.param_groups[0]
        self.b1, self.b2 = g0["betas"]
        # --- schedule ---
        self.ring_mode = cfg.opt_arm_start is not None
        ref = cfg.opt_ref_onset
        self.arm = (int(cfg.opt_arm_start), int(cfg.opt_arm_end)) if self.ring_mode else None
        if (not self.ring_mode) and cfg.opt_fixed_start is not None:
            self.fixed = (int(cfg.opt_fixed_start), int(cfg.opt_fixed_end))
        elif (not self.ring_mode) and ref is not None:
            self.fixed = (int(ref) - 500, int(ref) + 150)
        else:
            self.fixed = None
        self.bnear = (int(ref) - 2000, int(ref) + 150) if ref is not None else None
        self.q_every = max(1, int(cfg.opt_quant_every))
        # --- state for dumps ---
        self.bytes_written = 0
        self.dumps_enabled = True
        self.messages = []
        self.b_steps, self.b_vecs = [], []
        self.b_shards = 0
        self.loss_hist, self.g_hist = deque(maxlen=50), deque(maxlen=50)
        self.trig_step, self.no_trigger, self.c_done = None, False, False
        self.post_left = 0
        self.c_steps, self.c_v, self.c_m, self.c_g = [], [], [], []
        if self.ring_mode:
            cap = int(cfg.opt_ring_steps) + 1  # t*-ring .. t*
            self.cap = cap
            self.ring = {k: torch.empty(cap, self.P, dtype=torch.bfloat16) for k in "vmg"}
            self.ring_steps = np.full(cap, -1, np.int64)
            self.ring_n = 0
        self.last_save = time.time()
        self.finalized = False
        self._prev_sigterm = signal.signal(signal.SIGTERM, self._on_sigterm)
        self._guard_check(announce=True)

    # ---------------------------------------------------------------- disk guard
    def _guard_check(self, announce=False) -> bool:
        """False (and dumps disabled) if /data2 free < min_free_gb or opt_dense/ > max_total_gb."""
        if not self.dumps_enabled:
            return False
        anc = self.dump_dir
        while not anc.exists() and anc != anc.parent:
            anc = anc.parent
        free = shutil.disk_usage(anc).free / GB
        root = self.dump_dir.parent
        total = dir_size(root) / GB if root.exists() else 0.0
        msg = None
        if free < self.cfg.opt_min_free_gb:
            msg = f"free space {free:.1f} GB < {self.cfg.opt_min_free_gb} GB"
        elif total > self.cfg.opt_max_total_gb:
            msg = f"{root} holds {total:.1f} GB > {self.cfg.opt_max_total_gb} GB"
        if msg:
            self.dumps_enabled = False
            self.messages.append("DISK GUARD: bulky dumps disabled: " + msg)
            print("[optdense] WARNING " + self.messages[-1], flush=True)
            return False
        if announce:
            print(f"[optdense] disk guard ok: free={free:.0f} GB, opt_dense={total:.2f} GB, dump_dir={self.dump_dir}",
                  flush=True)
        return True

    def _can_write(self, nbytes: int) -> bool:
        if not self.dumps_enabled:
            return False
        if self.bytes_written + nbytes > self.cfg.opt_cap_gb * GB:
            self.dumps_enabled = False
            self.messages.append(f"per-run cap {self.cfg.opt_cap_gb} GB reached; further dumps stopped")
            print("[optdense] WARNING " + self.messages[-1], flush=True)
            return False
        if not self._guard_check():
            return False
        self.dump_dir.mkdir(parents=True, exist_ok=True)
        return True

    def _savez(self, name, **arrs):
        nbytes = sum(np.asarray(a).nbytes for a in arrs.values())
        if not self._can_write(nbytes):
            return False
        tmp = self.dump_dir / (name + ".tmp.npz")
        np.savez(tmp, **arrs)
        os.replace(tmp, self.dump_dir / name)
        self.bytes_written += nbytes
        return True

    # ---------------------------------------------------------------- hooks
    def before_step(self):
        self.prev_theta = torch.cat([p.detach().reshape(-1) for p in self.params]).clone()

    @staticmethod
    def _quant(x_sorted, qs):
        n = x_sorted.numel()
        idx = torch.tensor([int(round(q * (n - 1))) for q in qs])
        return x_sorted[idx]

    @staticmethod
    def _merged_quantiles(sorted_parts, qs):
        """Exact nearest-rank quantiles of the concatenation of already-sorted non-negative float32 vectors, without
        re-sorting: bisect on the (order-preserving, for x>=0) int32 bit pattern with searchsorted counts."""
        n = sum(p.numel() for p in sorted_parts)
        ranks = torch.tensor([int(round(q * (n - 1))) for q in qs], dtype=torch.int64)
        ints = [p.contiguous().view(torch.int32) for p in sorted_parts]
        lo = torch.zeros_like(ranks)
        hi = torch.full_like(ranks, 0x7F800000)
        for _ in range(31):
            mid = (lo + hi) // 2
            cnt = sum(torch.searchsorted(i, mid.to(torch.int32), right=True) for i in ints)
            ok = cnt >= ranks + 1
            hi = torch.where(ok, mid, hi)
            lo = torch.where(ok, lo, mid + 1)
        return lo.to(torch.int32).view(torch.float32)

    @torch.no_grad()
    def after_step(self, t: int, loss: float):
        ms, vs, gs = [], [], []
        for p in self.params:
            st = self.opt.state[p]
            ms.append(st["exp_avg"].reshape(-1))
            vs.append(st["exp_avg_sq"].reshape(-1))
            gs.append((p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1))
        m, v, grad = torch.cat(ms), torch.cat(vs), torch.cat(gs)
        g0 = self.opt.param_groups[0]
        lr, eps = float(g0["lr"]), float(g0["eps"])
        vhat = v / (1.0 - self.b2 ** t)
        mhat = m / (1.0 - self.b1 ** t)
        gp = grad + self.wd_vec * self.prev_theta
        sq = vhat.sqrt()
        denom = sq + eps
        upd = mhat / denom
        r = upd.abs()
        a = lr / denom
        dth = lr * upd
        theta = torch.cat([p.detach().reshape(-1) for p in self.params])

        in_fixed = self.fixed is not None and self.fixed[0] <= t <= self.fixed[1]
        in_arm = self.ring_mode and self.arm[0] <= t <= self.arm[1]
        do_q = in_fixed or in_arm or (t % self.q_every == 0)

        row = self.mat[t - 1]
        ci = self.ci
        row[ci["loss"]] = loss
        gpn = float(gp.norm())
        row[ci["gprime_norm"]] = gpn
        row[ci["grad_norm"]] = float(grad.norm())
        row[ci["theta_norm"]] = float(theta.norm())
        row[ci["dtheta_norm"]] = float(dth.norm())
        row[ci["step"]] = t
        row[ci["lr"]] = lr
        sorted_v, sorted_r = {}, {}
        if do_q:
            for s in SCOPES[1:]:
                lo, hi = self.scope_idx[s]
                sorted_v[s] = torch.sort(vhat[lo:hi]).values
                sorted_r[s] = torch.sort(r[lo:hi]).values
        for s in SCOPES:
            lo, hi = self.scope_idx[s]
            sv, sr, sa, sd, ssq = vhat[lo:hi], r[lo:hi], a[lo:hi], dth[lo:hi], sq[lo:hi]
            n = hi - lo
            if do_q:
                if s == "global":
                    parts_v = [sorted_v[k] for k in SCOPES[1:]]
                    parts_r = [sorted_r[k] for k in SCOPES[1:]]
                    vq = self._merged_quantiles(parts_v, VQ)
                    rq = self._merged_quantiles(parts_r, RQ)
                else:
                    vq = self._quant(sorted_v[s], VQ)
                    rq = self._quant(sorted_r[s], RQ)
                for q, val in zip(VQ, vq):
                    row[ci[f"{s}/vhat_q{q:g}"]] = float(val)
                for q, val in zip(RQ, rq):
                    row[ci[f"{s}/r_q{q:g}"]] = float(val)
            row[ci[f"{s}/frac_sqrtv_lt_eps"]] = float((ssq < eps).sum()) / n
            row[ci[f"{s}/frac_sqrtv_lt_10eps"]] = float((ssq < 10 * eps).sum()) / n
            row[ci[f"{s}/max_a"]] = float(sa.max())
            ad = sd.abs()
            row[ci[f"{s}/max_dtheta_over_lr"]] = float(ad.max()) / lr
            row[ci[f"{s}/frac_dtheta_gt_lr"]] = float((ad > lr).sum()) / n
            row[ci[f"{s}/frac_dtheta_gt_2lr"]] = float((ad > 2 * lr).sum()) / n
            row[ci[f"{s}/dtheta_norm"]] = float(sd.norm())

        # ---- (b) vhat every dump_every steps + every near_every steps near onset
        if t % int(self.cfg.opt_dump_every) == 0 or (
                self.bnear is not None and self.bnear[0] <= t <= self.bnear[1]
                and t % int(self.cfg.opt_dump_near_every) == 0):
            self.b_steps.append(t)
            self.b_vecs.append(_bf16_bits(vhat))
            if len(self.b_vecs) >= 100:
                self._flush_b()

        # ---- (c) full vectors
        if self.ring_mode:
            self._ring_step(t, loss, gpn, vhat, mhat, gp, in_arm)
        elif in_fixed and (t - self.fixed[0]) % int(self.cfg.opt_fixed_every) == 0:
            self.c_steps.append(t)
            self.c_v.append(_bf16_bits(vhat)); self.c_m.append(_bf16_bits(mhat)); self.c_g.append(_bf16_bits(gp))
        if self.fixed is not None and t == self.fixed[1]:
            self._flush_c()

        if self.cfg.opt_scalar_save_every and t % int(self.cfg.opt_scalar_save_every) == 0:
            self._save_scalars(t)

    def _ring_step(self, t, loss, gpn, vhat, mhat, gp, in_arm):
        if self.c_done:
            return
        if self.post_left > 0:  # recording after trigger
            self.c_steps.append(t)
            self.c_v.append(_bf16_bits(vhat)); self.c_m.append(_bf16_bits(mhat)); self.c_g.append(_bf16_bits(gp))
            self.post_left -= 1
            if self.post_left == 0:
                self._flush_c()
            return
        if in_arm:
            k = self.ring_n % self.cap
            self.ring["v"][k].copy_(vhat); self.ring["m"][k].copy_(mhat); self.ring["g"][k].copy_(gp)
            self.ring_steps[k] = t
            self.ring_n += 1
            if len(self.loss_hist) >= 50:
                if loss >= 3.0 * float(np.median(self.loss_hist)) and gpn >= 5.0 * float(np.median(self.g_hist)):
                    self.trig_step = t
                    print(f"[optdense] TRIGGER at step {t}: loss={loss:.4g} g'={gpn:.4g}", flush=True)
                    self._flush_ring()
                    self.post_left = int(self.cfg.opt_post_steps)
                    self.loss_hist.append(loss); self.g_hist.append(gpn)
                    return
            if t == self.arm[1]:
                self.no_trigger = True
                self.messages.append("armed region ended without trigger; flushed last ring")
                self._flush_ring()
                self._flush_c()
                return
        self.loss_hist.append(loss); self.g_hist.append(gpn)

    def _flush_ring(self):
        n = min(self.ring_n, self.cap)
        order = [(self.ring_n - n + i) % self.cap for i in range(n)]  # oldest -> newest
        for k in order:
            self.c_steps.append(int(self.ring_steps[k]))
            self.c_v.append(self.ring["v"][k].view(torch.int16).numpy().view(np.uint16).copy())
            self.c_m.append(self.ring["m"][k].view(torch.int16).numpy().view(np.uint16).copy())
            self.c_g.append(self.ring["g"][k].view(torch.int16).numpy().view(np.uint16).copy())

    # ---------------------------------------------------------------- output
    def _flush_b(self):
        if not self.b_vecs:
            return
        self._savez(f"vhat_bf16_part{self.b_shards:03d}.npz", steps=np.array(self.b_steps, np.int64),
                    vhat_u16=np.stack(self.b_vecs))
        self.b_shards += 1
        self.b_steps, self.b_vecs = [], []

    def _flush_c(self):
        if self.c_done or not self.c_steps:
            self.c_done = True
            return
        self.c_done = True
        ok = self._savez("c_dump.npz", steps=np.array(self.c_steps, np.int64), vhat_u16=np.stack(self.c_v),
                    mhat_u16=np.stack(self.c_m), gp_u16=np.stack(self.c_g),
                    trig_step=np.int64(-1 if self.trig_step is None else self.trig_step),
                    no_trigger=np.bool_(self.no_trigger), mode=np.array("ring" if self.ring_mode else "fixed"),
                    P=np.int64(self.P))
        print(f"[optdense] c_dump {'written' if ok else 'REFUSED'}: {len(self.c_steps)} steps", flush=True)
        self.c_v, self.c_m, self.c_g = [], [], []

    def _save_scalars(self, t):
        self.run_dir.mkdir(parents=True, exist_ok=True)
        tmp = self.run_dir / "dense_scalars.tmp.npy"
        np.save(tmp, self.mat[:t])
        os.replace(tmp, self.run_dir / "dense_scalars.npy")

    def finalize(self, last_step: int):
        if self.finalized:
            return
        self.finalized = True
        if not self.c_done and (self.c_steps or (self.ring_mode and self.trig_step is None and self.ring_n)):
            if self.ring_mode and self.trig_step is None:
                self.no_trigger = True
                self.messages.append("run ended before trigger/arm end; flushed last ring")
                self._flush_ring()
            self._flush_c()
        self._flush_b()
        # merge (b) shards into vhat_bf16.npz
        shards = sorted(self.dump_dir.glob("vhat_bf16_part*.npz")) if self.dump_dir.exists() else []
        if shards:
            steps, vecs = [], []
            for f in shards:
                with np.load(f) as z:
                    steps.append(z["steps"]); vecs.append(z["vhat_u16"])
            tmp = self.dump_dir / "vhat_bf16.tmp.npz"
            np.savez(tmp, steps=np.concatenate(steps), vhat_u16=np.concatenate(vecs))
            os.replace(tmp, self.dump_dir / "vhat_bf16.npz")
            for f in shards:
                f.unlink()
        self._save_scalars(last_step)
        (self.run_dir / "dense_scalars_cols.json").write_text(json.dumps(self.cols))
        meta = dict(mode="ring" if self.ring_mode else ("fixed" if self.fixed else "none"),
                    trig_step=self.trig_step, no_trigger=self.no_trigger, arm=self.arm, fixed=self.fixed,
                    bnear=self.bnear, bytes_written=self.bytes_written, dump_dir=str(self.dump_dir),
                    messages=self.messages, last_step=int(last_step), P=self.P)
        (self.run_dir / "dense_meta.json").write_text(json.dumps(meta, indent=1))
        print(f"[optdense] finalized: trig={self.trig_step} bytes={self.bytes_written / GB:.3f} GB", flush=True)

    def _on_sigterm(self, signum, frame):
        last = int(np.nanmax(self.mat[:, self.ci["step"]])) if np.isfinite(self.mat[:, self.ci["step"]]).any() else 0
        self.finalize(last)
        raise SystemExit(143)
