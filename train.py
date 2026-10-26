"""Train a small MNIST encoder + linear-probe model and snapshot the latents
of a fixed probe set on a schedule, so later analysis code can study how the
latent geometry evolves over training (see SPEC.md for the exact on-disk
data contract).

Supports three training objectives (`--objective`: probe / autoencoder /
probe_recon) and three latent normalizations (`--latent_norm`: none / l2 /
bn) -- see model.py's module docstring for the full contract. Regardless of
objective, a CE/accuracy readout is always available because the probe head
always produces logits (on a detached latent for objective="autoencoder").
Snapshotted latents are always the POST-latent_norm latent (see
`snapshot_latents` below).

`--dense_early N` logs a snapshot at EVERY optimizer step for the first N
steps (in addition to step 0, every `--log_every` steps, and the final
step), for finer early-time resolution.

GPU selection is the coordinator's responsibility (see CLAUDE.md): this
script only ever requests the generic "cuda" device -- it never sets
CUDA_VISIBLE_DEVICES, never calls torch.cuda.set_device with a hard-coded
index, and never assumes it owns more than whatever GPU(s) were pinned for
it in the environment.

Usage:
    python train.py --smoke                      # tiny end-to-end sanity check
    python train.py --latent_dim 16 --epochs 5    # a full run
    python train.py --objective autoencoder --latent_norm l2 --dense_early 20
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from config import Config, Paths, config_from_args, get_paths
from data import (
    IndexedLabelDataset,
    build_probe_indices,
    build_train_pool,
    build_train_probe,
    get_dataloaders,
    get_eval_loader,
    get_probe_tensors,
    load_mnist,
    load_train_eval_ds,
    save_probe_index,
)
from model import LatentProbeModel
from optdense import DEFAULT_DUMP_ROOT, OptDense
from dense_latent import DenseLatentLogger, RecordingRandomSampler
from optdiag import FIXED_BATCH_MAX, OptDiag


def seed_everything(seed: int) -> None:
    """Seed python, numpy, and torch (CPU + all CUDA devices) for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class RunningStats:
    """Accumulates loss/accuracy over the training batches since the last reset
    (i.e. since the last snapshot), used to report `train_loss`/`train_acc`.
    """

    loss_sum: float = 0.0
    correct: int = 0
    total: int = 0

    def update(self, loss: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor) -> None:
        bs = labels.size(0)
        self.loss_sum += float(loss.detach()) * bs
        self.correct += int((logits.detach().argmax(dim=1) == labels).sum())
        self.total += bs

    def averages(self) -> Tuple[float, float]:
        if self.total == 0:
            return float("nan"), float("nan")
        return self.loss_sum / self.total, self.correct / self.total

    def reset(self) -> None:
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0


def _to_device(images: torch.Tensor, device: torch.device, channels_last: bool) -> torch.Tensor:
    images = images.to(device)
    if channels_last and images.dim() == 4:
        images = images.contiguous(memory_format=torch.channels_last)
    return images


def log_snapshot_steps(total_steps: int, n_snapshots: int, n_dense: int = 10) -> list:
    """Geometric snapshot schedule: step 0, steps 1..n_dense densely, then geometrically spaced
    integer steps up to `total_steps` (always included), ~n_snapshots in all. Rounding collapses
    duplicates, so the count is grown until >= n_snapshots distinct steps exist (or all steps
    0..total_steps are used when total_steps is small)."""
    dense = min(n_dense, total_steps)
    base = set(range(0, dense + 1))
    if total_steps <= dense or total_steps + 1 <= n_snapshots:
        return sorted(set(range(0, total_steps + 1)))
    num = max(n_snapshots - len(base), 2)
    while True:
        geo = np.unique(np.round(np.geomspace(dense, total_steps, num)).astype(np.int64))
        steps = base | set(int(x) for x in geo)
        if len(steps) >= n_snapshots or num > 10 * n_snapshots:
            break
        num += 1
    steps.add(total_steps)
    return sorted(steps)


@torch.no_grad()
def evaluate_full(
    model: LatentProbeModel, loader, device: torch.device, amp: bool = False, channels_last: bool = False
) -> Tuple[float, float]:
    """Full pass over `loader` in eval mode (no grad). Returns (avg_loss, accuracy).
    Eval mode => BatchNorm uses (and never updates) running statistics."""
    model.eval()
    ce = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, total = 0.0, 0, 0
    for images, labels in loader:
        images, labels = _to_device(images, device, channels_last), labels.to(device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=amp):
            logits, _ = model(images)
        logits = logits.float()
        loss_sum += float(ce(logits, labels))
        correct += int((logits.argmax(dim=1) == labels).sum())
        total += labels.size(0)
    return loss_sum / total, correct / total


@torch.no_grad()
def snapshot_latents(
    model: LatentProbeModel,
    probe_images: torch.Tensor,
    probe_labels: torch.Tensor,
    device: torch.device,
    out_path: Path,
    scalars: dict,
    extra_labels: dict = None,
    channels_last: bool = False,
) -> None:
    """Encode a fixed probe set (eval mode, no grad, fp32 -- never autocast) and write one snapshot
    `.npz`, per SPEC.md's latent-snapshot data contract:
        latents    : float32 [N, D]   (POST-latent_norm, PRE-l2_scale latent)
        labels     : int64   [N]
        + every entry of `scalars` (step, epoch, train_loss, train_acc, test_acc, test_loss, ... --
          built in main()) and `extra_labels` (e.g. noisy_labels, int64 [N]).
    Same format for latents/ (test probe) and latents_train/ (train probe).
    """
    model.eval()
    latents = model.encode(_to_device(probe_images, device, channels_last)).cpu().numpy().astype(np.float32)
    np.savez(
        out_path,
        latents=latents,
        labels=probe_labels.numpy().astype(np.int64),
        **(extra_labels or {}),
        **scalars,
    )


@torch.no_grad()
def recon_errors(model, images: torch.Tensor, device: torch.device, channels_last: bool = False, bs: int = 2000) -> np.ndarray:
    """Per-sample eval-mode, fp32 reconstruction MSE (mean over pixels, in normalized-pixel units). float32 [N]."""
    model.eval()
    out = []
    for i in range(0, len(images), bs):
        x = _to_device(images[i:i + bs], device, channels_last)
        _, recon = model(x)
        out.append(((recon.float() - x.flatten(1).float()) ** 2).mean(dim=1).cpu())
    return torch.cat(out).numpy().astype(np.float32)


def main() -> None:
    cfg: Config = config_from_args()
    assert cfg.log_every >= 1, "log_every must be >= 1"
    paths: Paths = get_paths(cfg)
    paths.mkdirs()
    if cfg.probe_train:
        paths.latents_train_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    device = torch.device(cfg.device)
    use_amp = cfg.amp and device.type == "cuda"
    cl = cfg.channels_last and device.type == "cuda"

    # --- data: full train/test sets + the fixed, reproducible probe subset(s) ---
    train_ds, test_ds = load_mnist(cfg, paths.data_dir)  # train_ds is augmented iff cfg.augment
    plain_train_ds = load_train_eval_ds(cfg, paths.data_dir) if cfg.augment else train_ds
    # Training pool: the examples the model actually trains on (train_subset / label_noise).
    pool = build_train_pool(cfg, train_ds.targets)
    if not pool["is_identity"]:
        train_labels_used = pool["noisy_labels"] if pool["noisy_labels"] is not None else pool["labels"]
        train_ds = IndexedLabelDataset(train_ds, pool["indices"], train_labels_used)
        plain_train_ds = IndexedLabelDataset(plain_train_ds, pool["indices"], train_labels_used)
        np.savez(
            paths.train_index_path, indices=pool["indices"], labels=pool["labels"],
            **({} if pool["noisy_labels"] is None else
               dict(noisy_labels=pool["noisy_labels"], noisy_mask=pool["noisy_mask"])),
        )
    dense_on = bool(cfg.log_latents_every or cfg.log_batch_indices or cfg.ckpt_steps or cfg.ckpt_logspace)
    rec_sampler = RecordingRandomSampler(train_ds) if cfg.log_batch_indices else None  # same RNG use as shuffle=True
    train_loader, test_loader = get_dataloaders(cfg, train_ds, test_ds, train_sampler=rec_sampler)
    # Un-augmented, sequential view of the training pool (noisy labels if label_noise): full-pass train eval.
    # Without augmentation train_loader already is that view (shuffle order is irrelevant for a full pass; reusing
    # it keeps default-run RNG consumption identical to the original pipeline).
    train_eval_loader = get_eval_loader(cfg, plain_train_ds) if cfg.augment else train_loader

    probe_indices, probe_labels_np = build_probe_indices(
        test_ds, cfg.n_probe, cfg.probe_balanced, cfg.seed
    )
    save_probe_index(paths.probe_index_path, probe_indices, probe_labels_np)
    probe_images, probe_labels = get_probe_tensors(test_ds, probe_indices)

    probe_train_images = probe_train_labels = None
    tp = None
    probe_train_noisy = None
    if cfg.probe_train:
        tp = build_train_probe(pool, cfg.n_probe, cfg.probe_balanced, cfg.seed)
        src = load_train_eval_ds(cfg, paths.data_dir)  # un-augmented, indexed like the full train split
        probe_train_images, probe_train_labels = get_probe_tensors(src, tp["indices"])  # true labels
        if tp["noisy_labels"] is not None:
            probe_train_noisy = torch.as_tensor(tp["noisy_labels"], dtype=torch.int64)
        np.savez(
            paths.probe_train_index_path, indices=tp["indices"], labels=tp["labels"],
            **({} if tp["noisy_labels"] is None else dict(noisy_labels=tp["noisy_labels"])),
        )

    # --- fixed recon-eval subsets (recon objectives): seeded subset of the training POOL and of the test set ---
    rec_train_images = rec_test_images = None
    if cfg.objective in ("autoencoder", "probe_recon"):
        rs = np.random.RandomState(cfg.seed + 7000)
        n_tr = min(cfg.recon_eval_n, len(plain_train_ds))
        n_te = min(cfg.recon_eval_n, len(test_ds))
        tr_idx = np.sort(rs.choice(len(plain_train_ds), n_tr, replace=False))
        te_idx = np.sort(rs.choice(len(test_ds), n_te, replace=False))
        rec_train_images, _ = get_probe_tensors(plain_train_ds, tr_idx)
        rec_test_images, _ = get_probe_tensors(test_ds, te_idx)

    # --- model / optimizer ---
    # cfg.recon (whether a decoder is built) is derived from cfg.objective in
    # Config.resolve() -- see config.py / model.py for the objective contract.
    model = LatentProbeModel(
        cfg.latent_dim,
        encoder_arch=cfg.encoder_arch,
        objective=cfg.objective,
        latent_norm=cfg.latent_norm,
        l2_scale=cfg.l2_scale,
        l2_scale_learned=cfg.l2_scale_learned,
        linear_ae=cfg.linear_ae,
    ).to(device)
    if cl:
        model = model.to(memory_format=torch.channels_last)
    # The learnable l2 scale is excluded from weight decay.
    decay = [p for n, p in model.named_parameters() if n != "l2_scale"]
    groups = [dict(params=decay)]
    if cfg.lr_mult_enc != 1.0 or cfg.lr_mult_dec != 1.0:  # layer-timescale control: encoder/decoder/other get separate lr
        enc_p = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
        dec_p = [p for n, p in model.named_parameters() if n.startswith("decoder.")]
        oth_p = [p for n, p in model.named_parameters() if not n.startswith(("encoder.", "decoder.")) and n != "l2_scale"]
        groups = [dict(params=enc_p, lr=cfg.lr * cfg.lr_mult_enc), dict(params=dec_p, lr=cfg.lr * cfg.lr_mult_dec)] + ([dict(params=oth_p)] if oth_p else [])
    if model.l2_scale is not None:
        groups.append(dict(params=[model.l2_scale], weight_decay=0.0))
    if cfg.optimizer == "adam":  # NOTE: weight_decay here is Adam's coupled L2 penalty (not AdamW)
        optimizer = torch.optim.Adam(groups, lr=cfg.lr, weight_decay=cfg.weight_decay,
                                     betas=(0.9, cfg.adam_beta2), eps=cfg.adam_eps)
    else:
        optimizer = torch.optim.SGD(groups, lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss() if cfg.recon else None

    # --- optimizer diagnostics (--log_opt_diag): fixed train batch from the (un-augmented) training pool ---
    opt_diag = None
    if cfg.log_opt_diag:
        n_fb = min(FIXED_BATCH_MAX, len(plain_train_ds))
        fb_idx = np.random.RandomState(cfg.seed + 5000).choice(len(plain_train_ds), n_fb, replace=False)
        fb = [plain_train_ds[int(i)] for i in fb_idx]
        fb_images = _to_device(torch.stack([torch.as_tensor(x[0]) for x in fb]), device, cl)
        fb_labels = torch.as_tensor([int(x[1]) for x in fb], dtype=torch.long, device=device)

        def diag_loss(batch):  # same objective as the training loss (`backward_loss`), fp32, eval mode
            imgs, labs = batch
            logits, recon = model(imgs)
            ce_ = ce_loss(logits.float(), labs)
            if cfg.objective == "probe":
                return ce_
            rl = mse_loss(recon.float(), imgs.flatten(1).float())
            return rl + ce_ if cfg.objective == "autoencoder" else ce_ + cfg.recon_weight * rl

        opt_diag = OptDiag(model, list(model.parameters()), decay, cfg.weight_decay, diag_loss,
                           (fb_images, fb_labels), optimizer, cfg.optimizer == "adam", cfg.seed)

    # --- resolved config, written once up front (SPEC.md: config.json) ---
    with open(paths.config_path, "w") as f:
        json.dump(cfg.to_json_dict(), f, indent=2, sort_keys=True)

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * cfg.epochs
    if cfg.train_steps is not None:
        total_steps = cfg.train_steps  # may exceed epochs*steps_per_epoch: epochs cycle until reached
    if cfg.max_steps is not None:
        total_steps = min(total_steps, cfg.max_steps)
    scheduler = None
    if cfg.lr_schedule == "cosine":  # stepped once per optimizer step; lr -> 0 at total_steps
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # Snapshot schedule. "linear": step 0 (random init), EVERY step for the first `dense_early`
    # optimizer steps, every `log_every` steps thereafter, and always the final step.
    # "log": step 0, steps 1..10, then geometric spacing up to the final step (~n_snapshots).
    assert cfg.dense_early >= 0, "dense_early must be >= 0"
    if cfg.log_schedule == "log":
        snapshot_steps = set(log_snapshot_steps(total_steps, cfg.n_snapshots))
    else:
        dense_upto = min(cfg.dense_early, total_steps)
        snapshot_steps = set(range(0, total_steps + 1, cfg.log_every))
        snapshot_steps.update(range(1, dense_upto + 1))
    snapshot_steps.add(0)
    snapshot_steps.add(total_steps)

    print(
        f"[run {cfg.run_id}] device={device} objective={cfg.objective} "
        f"latent_norm={cfg.latent_norm} total_steps={total_steps} "
        f"steps_per_epoch={steps_per_epoch} pool={len(train_ds)} n_probe={len(probe_indices)} "
        f"probe_train={cfg.probe_train} n_snapshots={len(snapshot_steps)} "
        f"run_dir={paths.run_dir}"
    )

    opt_dense = None
    if cfg.log_opt_dense:  # read-only: no RNG, no .grad writes, no optimizer-state writes (see optdense.py)
        assert cfg.optimizer == "adam", "--log_opt_dense requires --optimizer adam"
        opt_dense = OptDense(model, list(model.parameters()), decay, cfg.weight_decay, optimizer, cfg, total_steps,
                             paths.run_dir, cfg.opt_dump_dir or f"{DEFAULT_DUMP_ROOT}/{cfg.run_id}")

    dense = None
    if dense_on:  # read-only: eval mode + no_grad, no RNG (see dense_latent.py)
        used = probe_train_noisy if probe_train_noisy is not None else probe_train_labels
        dense = DenseLatentLogger(
            cfg, model, optimizer, device, total_steps, probe_images, probe_train_images, used,
            tp["indices"], pool["indices"], len(train_ds.base) if hasattr(train_ds, "base") else len(train_ds),
            cl, _to_device,
        ) if cfg.log_latents_every else None
        if dense is None:
            raise ValueError("ckpt/batch flags need --log_latents_every >= 1 (use a large N to disable latent logging cost)")

    stats = RunningStats()
    t0 = time.time()

    def do_snapshot(step: int) -> None:
        # Snapshots run in eval mode under no_grad: BatchNorm running stats are neither used in
        # train mode nor updated, no dropout exists, and model.train() is restored afterwards, so
        # the model/optimizer state is untouched. The only side effect on training is RNG: creating
        # DataLoader iterators for the eval passes draws from the global torch RNG, which shifts
        # later training shuffles. --isolate_snapshot_rng saves/restores the torch RNG states so
        # training is then bit-identical regardless of the snapshot schedule / probe sets.
        if cfg.isolate_snapshot_rng:
            cpu_state = torch.get_rng_state()
            cuda_state = torch.cuda.get_rng_state_all() if device.type == "cuda" else None
        epoch_frac = step / steps_per_epoch
        run_loss, run_acc = stats.averages()
        if step == 0 or cfg.train_eval == "full":
            # Eval-mode full pass over the (un-augmented, possibly noisy-labelled) training pool.
            train_loss, train_acc = evaluate_full(model, train_eval_loader, device, use_amp, cl)
        else:
            train_loss, train_acc = run_loss, run_acc  # mean over batches since the previous snapshot
        # test_loss / test_acc: FULL test set (not just the probe subset), eval mode.
        test_loss, test_acc = evaluate_full(model, test_loader, device, use_amp, cl)

        scalars = dict(
            step=np.int64(step),
            epoch=np.float32(epoch_frac),
            train_loss=np.float32(train_loss),
            train_acc=np.float32(train_acc),
            test_acc=np.float32(test_acc),
            test_loss=np.float32(test_loss),
            lr=np.float32(optimizer.param_groups[0]["lr"]),
            train_loss_running=np.float32(run_loss),
            train_acc_running=np.float32(run_acc),
        )
        probe_rec_test = probe_rec_train = None
        if rec_train_images is not None:
            # eval-mode fp32 recon MSE: fixed eval subsets of train pool / test, and the probe points
            scalars["recon_train"] = np.float32(recon_errors(model, rec_train_images, device, cl).mean())
            scalars["recon_test"] = np.float32(recon_errors(model, rec_test_images, device, cl).mean())
            probe_rec_test = recon_errors(model, probe_images, device, cl)
            scalars["recon_probe_test"] = np.float32(probe_rec_test.mean())
            if cfg.probe_train:
                probe_rec_train = recon_errors(model, probe_train_images, device, cl)
                scalars["recon_probe_train"] = np.float32(probe_rec_train.mean())
        if model.current_l2_scale() is not None:
            scalars["l2_scale"] = np.float32(model.current_l2_scale())
        if opt_diag is not None:  # no RNG use, no .grad, no optimizer-state writes (see optdiag.py)
            scalars.update(opt_diag.compute(step))
        name = f"step_{step:07d}.npz"
        snapshot_latents(model, probe_images, probe_labels, device, paths.latents_dir / name, scalars,
                         extra_labels=None if probe_rec_test is None else dict(recon_err=probe_rec_test),
                         channels_last=cl)
        if cfg.probe_train:
            snapshot_latents(
                model, probe_train_images, probe_train_labels, device, paths.latents_train_dir / name, scalars,
                extra_labels={**({} if probe_train_noisy is None else dict(noisy_labels=probe_train_noisy.numpy())),
                              **({} if probe_rec_train is None else dict(recon_err=probe_rec_train))} or None,
                channels_last=cl,
            )
        model.train()
        stats.reset()
        if cfg.isolate_snapshot_rng:
            torch.set_rng_state(cpu_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)

        dt = time.time() - t0
        print(
            f"[snapshot] step={step:7d}/{total_steps} epoch={epoch_frac:7.2f} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.4f} elapsed={dt:6.1f}s"
            + (f" recon_train={scalars['recon_train']:.4f} recon_test={scalars['recon_test']:.4f}"
               if "recon_train" in scalars else "")
        )

    dae = {}
    dae_on = bool(cfg.dense_ae_every) and rec_train_images is not None and cfg.probe_train

    @torch.no_grad()
    def dense_ae_log(step: int) -> None:  # read-only: eval mode, no RNG; model.train() restored
        was = model.training
        model.eval()
        pt = _to_device(probe_images, device, cl)
        ptr = _to_device(probe_train_images, device, cl)
        d = dict(step=step,
                 z_test=model.encode(pt).cpu().numpy().astype(np.float32),
                 z_train=model.encode(ptr).cpu().numpy().astype(np.float32),
                 err_test=recon_errors(model, probe_images, device, cl),
                 err_train=recon_errors(model, probe_train_images, device, cl),
                 recon_train=np.float32(recon_errors(model, rec_train_images, device, cl).mean()),
                 recon_test=np.float32(recon_errors(model, rec_test_images, device, cl).mean()))
        for k, v in d.items():
            dae.setdefault(k, []).append(v)
        if was:
            model.train()

    do_snapshot(0)  # random-init snapshot, before any optimizer step
    if dae_on:
        dense_ae_log(0)
    if dense is not None:
        dense.log(0)

    model.train()
    global_step = 0
    done = False
    max_epochs = cfg.epochs if cfg.train_steps is None else 10**9
    for epoch in range(max_epochs):
        if done:
            break
        for bi, (images, labels) in enumerate(train_loader):
            images, labels = _to_device(images, device, cl), labels.to(device)
            if bi == 0 and rec_sampler is not None:
                # Snapshot this training epoch's permutation now: snapshot eval passes reuse train_loader
                # (and so the sampler) and would overwrite rec_sampler.last_perm mid-epoch.
                epoch_perm = rec_sampler.last_perm

            with torch.autocast(device.type, dtype=torch.bfloat16, enabled=use_amp):
                logits, recon = model(images)
                logits = logits.float()
                if recon is not None:
                    recon = recon.float()
            # `logits` are ALWAYS populated by the model (on a detached latent
            # for objective="autoencoder"), so a CE/accuracy readout is always
            # available regardless of objective.
            #
            # `loss` below is the reported/primary loss per objective
            # (== train_loss in the snapshot). `backward_loss` is what
            # actually gets .backward()'d -- for "autoencoder" it additionally
            # includes the probe's CE term so the probe (a readout) keeps
            # training; that term's gradient cannot reach the encoder/decoder
            # because model.forward() already detached the latent it is
            # computed from.
            ce = ce_loss(logits, labels)
            if cfg.objective == "probe":
                loss = ce
                backward_loss = loss
            elif cfg.objective == "autoencoder":
                recon_loss = mse_loss(recon, images.flatten(1).float())
                loss = recon_loss
                backward_loss = recon_loss + ce
            elif cfg.objective == "probe_recon":
                recon_loss = mse_loss(recon, images.flatten(1).float())
                loss = ce + cfg.recon_weight * recon_loss
                backward_loss = loss
            else:
                raise ValueError(f"Unknown objective: {cfg.objective!r}")

            optimizer.zero_grad(set_to_none=True)
            backward_loss.backward()
            if opt_diag is not None:
                opt_diag.before_step()
            if opt_dense is not None:
                opt_dense.before_step()
            optimizer.step()
            if opt_dense is not None:
                opt_dense.after_step(global_step + 1, float(loss.detach()))
            if scheduler is not None:
                scheduler.step()

            stats.update(loss, logits, labels)
            global_step += 1

            if global_step in snapshot_steps:
                do_snapshot(global_step)
            if dae_on and global_step <= cfg.dense_ae_until and global_step % cfg.dense_ae_every == 0:
                dense_ae_log(global_step)
            if dense is not None:
                bp = None
                if rec_sampler is not None:
                    bp = epoch_perm[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]
                dense.log(global_step, bp)

            if global_step >= total_steps:
                done = True
                break

    if opt_dense is not None:
        opt_dense.finalize(global_step)
    if dense is not None:
        dense.flush()
    if dae:
        Path(cfg.dense_ae_dir).mkdir(parents=True, exist_ok=True)
        np.savez(Path(cfg.dense_ae_dir) / "dense_ae.npz", labels_test=probe_labels.numpy(),
                 labels_train=probe_train_labels.numpy(), **{k: np.stack(v) for k, v in dae.items()})
    torch.save(model.state_dict(), paths.model_final_path)
    print(f"[run {cfg.run_id}] done. model saved to {paths.model_final_path}")
    if device.type == "cuda":
        print(f"[run {cfg.run_id}] elapsed={time.time() - t0:.1f}s peak_vram_MiB={torch.cuda.max_memory_allocated() / 2**20:.0f}")

    # Release GPU memory promptly -- shared-machine policy in CLAUDE.md.
    del model, optimizer
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
