"""Dataset loading and fixed-probe-set construction for mnist / fashion_mnist
/ cifar10 (see `config.VALID_DATASETS`).

The probe set is a small, FIXED subset of the *test* split, sampled once per
run (seeded, optionally class-balanced) and persisted to `probe_index.npz`.
Every latent snapshot for the run encodes exactly these samples, in exactly
this order, so latents are directly comparable across training steps (see
SPEC.md's data contract).
"""
from __future__ import annotations
import os

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from latentdyn.config import Config, VALID_DATASETS
from latentdyn.model import set_default_input_shape

MNIST_MEAN = 0.1307
MNIST_STD = 0.3081
FASHION_MNIST_MEAN = 0.2860
FASHION_MNIST_STD = 0.3530
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)

# dataset -> (mean, std) for transforms.Normalize, per-dataset.
_NORMALIZATION = {
    "mnist": ((MNIST_MEAN,), (MNIST_STD,)),
    "fashion_mnist": ((FASHION_MNIST_MEAN,), (FASHION_MNIST_STD,)),
    "cifar10": (CIFAR10_MEAN, CIFAR10_STD),
}


def _transform(dataset: str, augment: bool = False) -> transforms.Compose:
    """to-tensor + per-dataset normalize to (roughly) zero mean/unit std.

    `augment=True` (cifar10 training only) prepends the standard CIFAR
    augmentation: random crop 32 with 4px zero padding + random horizontal flip.
    """
    mean, std = _NORMALIZATION[dataset]
    ops = [transforms.ToTensor(), transforms.Normalize(mean, std)]
    if augment:
        ops = [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip()] + ops
    return transforms.Compose(ops)


def load_mnist(cfg: Config, data_dir: Path) -> Tuple[Dataset, Dataset]:
    """Load (downloading if needed) cfg.dataset's train/test splits.

    NOTE on the name: kept as `load_mnist` (rather than renamed to something
    dataset-generic) because train.py's `from data import load_mnist` call
    site is out of scope for this dataset-generalization change -- the NAME
    stays fixed even though the behavior is now dataset-generic, dispatching
    on `cfg.dataset` (mnist / fashion_mnist / cifar10).

    SIDE EFFECT: also calls `model.set_default_input_shape(cfg.in_channels,
    cfg.img_size)` -- see model.py's module docstring ("INPUT SHAPE
    PLUMBING") for why: train.py constructs `LatentProbeModel(...)` without
    passing in_channels/img_size explicitly, so this call (which always runs
    before that construction, from train.py's existing call order) is what
    makes the encoder/decoder come out the right shape for cfg.dataset.

    `data_dir` must live under `artifacts_root` (never under /homes or
    /data) per the storage policy in CLAUDE.md -- callers should pass
    `Paths.data_dir`. All three datasets share the same `data_dir` (each
    torchvision dataset class makes its own subdirectory under it), exactly
    like the original MNIST-only download_root.
    """
    if cfg.dataset not in VALID_DATASETS:
        raise ValueError(f"Unknown dataset: {cfg.dataset!r} (expected one of {VALID_DATASETS})")

    set_default_input_shape(cfg.in_channels, cfg.img_size)

    # Training split gets augmentation iff cfg.augment; the test split (and
    # `load_train_eval_ds`) never does.
    train_ds = _make_split(cfg.dataset, data_dir, True, _transform(cfg.dataset, getattr(cfg, "augment", False)))
    test_ds = _make_split(cfg.dataset, data_dir, False, _transform(cfg.dataset))
    return train_ds, test_ds


_DATASET_CLASSES = {"mnist": datasets.MNIST, "fashion_mnist": datasets.FashionMNIST, "cifar10": datasets.CIFAR10}


class _CachedCIFAR10(datasets.CIFAR10):
    """CIFAR10 whose (non-augmented) ToTensor+Normalize output is precomputed once (identical values,
    float32) -- avoids per-item PIL work in the 120 full-pass evals. Opt-out: env LD_NO_CIFAR_CACHE=1."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        x = torch.from_numpy(self.data).permute(0, 3, 1, 2).float().div(255.0)
        mean, std = _NORMALIZATION["cifar10"]
        self._x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)

    def __getitem__(self, i):
        return self._x[i], int(self.targets[i])


def _make_split(dataset: str, data_dir: Path, train: bool, tfm) -> Dataset:
    if dataset == "cifar10" and len(tfm.transforms) == 2 and not os.environ.get("LD_NO_CIFAR_CACHE"):
        return _CachedCIFAR10(root=str(data_dir), train=train, download=True, transform=tfm)
    return _DATASET_CLASSES[dataset](root=str(data_dir), train=train, download=True, transform=tfm)


def load_train_eval_ds(cfg: Config, data_dir: Path) -> Dataset:
    """The training split with the plain (never augmented) transform, used for
    the train probe set and full-pass train evaluation. Same indexing as the
    train_ds returned by `load_mnist`."""
    return _make_split(cfg.dataset, data_dir, True, _transform(cfg.dataset))


class IndexedLabelDataset(Dataset):
    """View of `base` restricted to `indices`, with `labels` overriding the
    base labels (used for train_subset and/or label_noise). Item i is
    (base[indices[i]][0], labels[i])."""

    def __init__(self, base: Dataset, indices: np.ndarray, labels: np.ndarray) -> None:
        self.base, self.indices, self.labels = base, np.asarray(indices), np.asarray(labels)
        self.targets = self.labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        return self.base[int(self.indices[i])][0], int(self.labels[i])


def balanced_choice(targets: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    """Positions (unsorted, grouped by class) of `n` class-balanced samples of `targets`
    (n//C per class, the remainder spread over the first classes; capped by availability)."""
    n_classes = int(targets.max()) + 1
    base = n // n_classes
    remainder = n - base * n_classes
    chosen = []
    for c in range(n_classes):
        class_indices = np.flatnonzero(targets == c)
        k = min(base + (1 if c < remainder else 0), len(class_indices))
        chosen.append(rng.choice(class_indices, size=k, replace=False))
    return np.concatenate(chosen)


def build_train_pool(cfg: Config, targets) -> dict:
    """Training pool = the examples the model actually trains on.

    Returns dict(indices [P] sorted int64 into the full train split, labels [P] true labels,
    noisy_labels [P] or None, noisy_mask [P] bool or None, is_identity bool).
    train_subset: fixed class-balanced subset (RandomState(seed+1000)). label_noise p:
    exactly round(p*P) pool examples get label (y + U{1..C-1}) mod C, i.e. a uniformly random
    WRONG label (RandomState(seed+2000)). Both are independent of the training RNG.
    """
    targets = np.asarray(targets).astype(np.int64)
    n_classes = int(targets.max()) + 1
    if cfg.train_subset is not None and cfg.train_subset < len(targets):
        indices = np.sort(balanced_choice(targets, cfg.train_subset, np.random.RandomState(cfg.seed + 1000)))
    else:
        indices = np.arange(len(targets))
    labels = targets[indices]
    noisy_labels = noisy_mask = None
    if cfg.label_noise > 0:
        rng = np.random.RandomState(cfg.seed + 2000)
        n_noisy = int(round(cfg.label_noise * len(indices)))
        sel = rng.choice(len(indices), size=n_noisy, replace=False)
        noisy_labels = labels.copy()
        noisy_labels[sel] = (labels[sel] + rng.randint(1, n_classes, size=n_noisy)) % n_classes
        noisy_mask = noisy_labels != labels
    is_identity = cfg.train_subset is None and cfg.label_noise <= 0
    return dict(indices=indices.astype(np.int64), labels=labels, noisy_labels=noisy_labels,
                noisy_mask=noisy_mask, is_identity=is_identity)


def build_train_probe(pool: dict, n_probe: int, balanced: bool, seed: int) -> dict:
    """Fixed probe set of `n_probe` examples drawn from the training pool
    (RandomState(seed+3000)). Returns dict(indices [N] into the full train split (sorted by
    pool position), labels [N] true, noisy_labels [N] or None, pool_pos [N])."""
    rng = np.random.RandomState(seed + 3000)
    labels = pool["labels"]
    if balanced:
        pos = np.sort(balanced_choice(labels, n_probe, rng))
    else:
        pos = np.sort(rng.choice(len(labels), size=min(n_probe, len(labels)), replace=False))
    return dict(
        indices=pool["indices"][pos], labels=labels[pos], pool_pos=pos,
        noisy_labels=None if pool["noisy_labels"] is None else pool["noisy_labels"][pos],
    )


def get_dataloaders(cfg: Config, train_ds: Dataset, test_ds: Dataset, train_sampler=None) -> Tuple[DataLoader, DataLoader]:
    """Standard shuffled train loader + a larger-batch sequential test loader
    (the test loader is used for full-test-set accuracy at every snapshot).
    """
    augment = getattr(cfg, "augment", False)
    # Tiny pools (train_subset) are loaded in-process: spawning workers every
    # (very short) epoch would dominate runtime. Default full-dataset runs keep 2 workers.
    train_workers = 0 if len(train_ds) <= 5000 else (4 if augment else 2)
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=(train_sampler is None), sampler=train_sampler,
        num_workers=train_workers, drop_last=False,
        persistent_workers=augment and train_workers > 0,
    )
    return train_loader, get_eval_loader(cfg, test_ds)


def get_eval_loader(cfg: Config, ds: Dataset) -> DataLoader:
    """Large-batch sequential loader for full-pass evaluation (test set / train pool)."""
    return DataLoader(
        ds, batch_size=max(cfg.batch_size, 512), shuffle=False,
        num_workers=0 if len(ds) <= 5000 else 2, drop_last=False,
    )


def build_probe_indices(
    test_ds: Dataset, n_probe: int, balanced: bool, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Pick a fixed, reproducible set of `n_probe` indices into `test_ds`.

    Works for any torchvision classification dataset exposing `.targets`,
    whether that's a tensor (mnist/fashion_mnist) or a plain list (cifar10)
    -- `np.asarray` normalizes both to an ndarray.

    Uses a dedicated `np.random.RandomState(seed)` instance (not the global
    numpy RNG) so probe-set selection is independent of, and never disturbed
    by, training randomness. Returned `indices` are sorted ascending so the
    probe order is a deterministic function of the dataset + seed alone.
    """
    targets = np.asarray(test_ds.targets)
    rng = np.random.RandomState(seed)

    if not balanced:
        indices = rng.choice(len(targets), size=n_probe, replace=False)
        indices = np.sort(indices)
        return indices.astype(np.int64), targets[indices].astype(np.int64)

    indices = np.sort(balanced_choice(targets, n_probe, rng))
    return indices.astype(np.int64), targets[indices].astype(np.int64)


def save_probe_index(path: Path, indices: np.ndarray, labels: np.ndarray) -> None:
    """Persist the fixed probe-set indices/labels (SPEC.md's probe_index.npz)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, indices=indices.astype(np.int64), labels=labels.astype(np.int64))


def load_probe_index(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a previously saved probe_index.npz."""
    data = np.load(path)
    return data["indices"], data["labels"]


def get_probe_tensors(test_ds: Dataset, indices: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stack probe-set images/labels in the fixed `indices` order.

    Returns CPU tensors: images [N, C, H, W] float32 (C/H/W per
    cfg.in_channels/cfg.img_size -- e.g. [N, 1, 28, 28] for mnist/
    fashion_mnist, [N, 3, 32, 32] for cifar10, following whatever shape
    `test_ds`'s transform produces), labels [N] int64. The MLP encoder
    flattens this internally (nn.Flatten() as its first layer); the CNN
    encoder consumes it directly as [N, C, H, W] -- so no reshaping is
    needed here regardless of `encoder_arch`.
    """
    images = torch.stack([test_ds[int(i)][0] for i in indices])
    labels = torch.tensor([int(test_ds[int(i)][1]) for i in indices], dtype=torch.int64)
    return images, labels
