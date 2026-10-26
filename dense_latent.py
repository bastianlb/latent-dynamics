"""Per-step dense probe-latent logging (--log_latents_every / --log_batch_indices / --ckpt_*), opt-in.

Read-only w.r.t. training: eval mode + torch.no_grad, no RNG consumption, model.train() restored afterwards.
Format: see /vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/README.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler

CHUNK = 1000


class RecordingRandomSampler(RandomSampler):
    """RandomSampler (same RNG consumption/timing as DataLoader(shuffle=True)) that remembers the
    permutation of the epoch currently being iterated (pool positions)."""

    last_perm = None

    def __iter__(self):
        perm = list(super().__iter__())
        self.last_perm = np.asarray(perm, dtype=np.int64)
        yield from perm


def ckpt_step_list(cfg, total_steps):
    steps = set()
    if cfg.ckpt_steps:
        steps.update(int(s) for s in str(cfg.ckpt_steps).replace(",", " ").split())
    if cfg.ckpt_logspace and cfg.ckpt_logspace > 0:
        g = np.unique(np.round(np.geomspace(1, total_steps, cfg.ckpt_logspace)).astype(np.int64))
        steps.update(int(s) for s in g)
    steps.add(0)
    out = set()
    for s in steps:
        if 0 <= s <= total_steps:
            out.add(s)
        if s + 1 <= total_steps:
            out.add(s + 1)
    return sorted(out)


class DenseLatentLogger:
    def __init__(self, cfg, model, optimizer, device, total_steps, probe_images, probe_train_images,
                 probe_train_labels_used, probe_train_orig_idx, pool_indices, n_train_full, channels_last, to_device):
        self.cfg, self.model, self.opt, self.device = cfg, model, optimizer, device
        self.every = int(cfg.log_latents_every)
        self.log_batch = bool(cfg.log_batch_indices)
        self.dir = Path(cfg.dense_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.dir / "ckpt"
        self.ckpt_steps = set(ckpt_step_list(cfg, total_steps)) if (cfg.ckpt_steps or cfg.ckpt_logspace) else set()
        if self.ckpt_steps:
            self.ckpt_dir.mkdir(exist_ok=True)
        self.total_steps = total_steps
        self.te = to_device(probe_images, device, channels_last) if self.every else None
        self.tr = to_device(probe_train_images, device, channels_last) if self.every else None
        self.tr_y = probe_train_labels_used.to(device) if self.every else None
        self.B = int(cfg.batch_size)
        self.slot = None
        if self.log_batch:
            self.slot = np.full(n_train_full, -1, np.int64)
            self.slot[probe_train_orig_idx] = np.arange(len(probe_train_orig_idx))
            self.n_probe_tr = len(probe_train_orig_idx)
        self.pool_indices = np.asarray(pool_indices)
        self.buf = {}
        self.cur_chunk = None
        np.savez(self.dir / "probe_info.npz", train_probe_orig_idx=np.asarray(probe_train_orig_idx, np.int64),
                 train_probe_labels_used=probe_train_labels_used.numpy().astype(np.int64),
                 pool_indices=self.pool_indices.astype(np.int64))
        json.dump(dict(run_id=cfg.run_id, log_latents_every=self.every, log_batch_indices=self.log_batch,
                       ckpt_steps=sorted(self.ckpt_steps), total_steps=total_steps, chunk=CHUNK,
                       batch_size=self.B), open(self.dir / "meta.json", "w"), indent=1)

    @torch.no_grad()
    def log(self, step, batch_pos=None):
        """Call after optimizer step `step` (0 = init). batch_pos: pool positions of the batch that produced `step`."""
        if self.every and (step % self.every == 0 or step == self.total_steps):
            c = step // CHUNK
            if self.cur_chunk is not None and c != self.cur_chunk:
                self.flush()
            self.cur_chunk = c
            was_training = self.model.training
            self.model.eval()
            zt = self.model.encode(self.te)
            logits, _ = self.model(self.tr)
            z_tr = self.model.encode(self.tr)
            loss = F.cross_entropy(logits.float(), self.tr_y, reduction="none")
            if was_training:
                self.model.train()
            b = self.buf
            b.setdefault("steps", []).append(step)
            b.setdefault("latents_test", []).append(zt.half())
            b.setdefault("latents_train", []).append(z_tr.half())
            b.setdefault("head_w", []).append(self.model.probe.weight.detach().float().clone())
            b.setdefault("head_b", []).append(self.model.probe.bias.detach().float().clone())
            b.setdefault("train_probe_loss", []).append(loss.float())
            if self.log_batch:
                idx = np.full(self.B, -1, np.int32)
                inb = np.zeros(self.n_probe_tr, bool)
                if batch_pos is not None:
                    orig = self.pool_indices[batch_pos]
                    idx[: len(orig)] = orig
                    hit = self.slot[orig]
                    inb[hit[hit >= 0]] = True
                b.setdefault("batch_idx", []).append(idx)
                b.setdefault("train_probe_in_batch", []).append(inb)
        if step in self.ckpt_steps:
            torch.save(dict(step=step, model=self.model.state_dict(), optimizer=self.opt.state_dict()),
                       self.ckpt_dir / f"step_{step:07d}.pt")

    def flush(self):
        b = self.buf
        if not b or not b.get("steps"):
            return
        out = dict(steps=np.asarray(b["steps"], np.int64))
        for k in ("latents_test", "latents_train", "head_w", "head_b", "train_probe_loss"):
            out[k] = torch.stack(b[k]).cpu().numpy()
        if self.log_batch:
            out["batch_idx"] = np.stack(b["batch_idx"])
            out["train_probe_in_batch"] = np.stack(b["train_probe_in_batch"])
        np.savez(self.dir / f"chunk_{self.cur_chunk:05d}.npz", **out)
        self.buf = {}
