"""Configuration for the latent-dynamics probe-training pipeline.

Supports three datasets (`dataset`): "mnist", "fashion_mnist" (both
1x28x28), and "cifar10" (3x32x32) -- see DATASET_SHAPES below. See SPEC.md
for the on-disk data contract (paths, run_id format, latent snapshot
schema) and CLAUDE.md for the shared-GPU usage policy. This module only
resolves configuration and derives paths; it never touches the GPU.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

DEFAULT_ARTIFACTS_ROOT = "/vol/bitbucket/lbastian/latent-dynamics"

VALID_OBJECTIVES = ("probe", "autoencoder", "probe_recon")
VALID_LATENT_NORMS = ("none", "l2", "bn")
VALID_DATASETS = ("mnist", "fashion_mnist", "cifar10")
VALID_ENCODERS = ("mlp", "cnn", "resnet18", "convae")
VALID_LOG_SCHEDULES = ("linear", "log")
VALID_OPTIMIZERS = ("adam", "sgd")
VALID_LR_SCHEDULES = ("constant", "cosine")
VALID_TRAIN_EVALS = ("running", "full")

# dataset -> (in_channels, img_size). All three supported datasets are
# 10-class, so N_CLASSES below is a single constant rather than part of this
# map; if a non-10-class dataset is ever added, promote n_classes into this
# map too.
DATASET_SHAPES = {
    "mnist": (1, 28),
    "fashion_mnist": (1, 28),
    "cifar10": (3, 32),
}
N_CLASSES = 10


@dataclass
class Config:
    """Resolved run configuration. All fields are plain JSON-serializable types.

    `device` and `run_id` are left as None until `resolve()` is called, at
    which point they are derived (unless already set explicitly). `recon` is
    always (re)derived from `objective` in `resolve()` -- see below.
    """

    dataset: str = "mnist"  # "mnist" | "fashion_mnist" | "cifar10" -- see VALID_DATASETS
    # in_channels / img_size / n_classes are ALWAYS (re)derived from `dataset`
    # in resolve() -- like `recon`, these are not independent CLI flags, they
    # exist as fields (and land in config.json) because downstream code and
    # SPEC.md's config contract need them. Do not rely on a value set before
    # resolve() has run.
    in_channels: int = 1
    img_size: int = 28
    n_classes: int = 10
    latent_dim: int = 16
    n_probe: int = 1000
    probe_balanced: bool = True
    log_every: int = 50  # snapshot every this many optimizer steps
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    seed: int = 0
    encoder_arch: str = "mlp"  # "mlp" or "cnn"
    objective: str = "probe"  # "probe" | "autoencoder" | "probe_recon" -- see model.py
    latent_norm: str = "none"  # "none" | "l2" | "bn" -- see model.py
    # Derived in resolve() from `objective` (True iff a decoder is needed, i.e.
    # objective in {"autoencoder", "probe_recon"}). Not an independent CLI
    # flag -- kept as a field (and in config.json) because SPEC.md's config
    # contract requires a `recon` bool. Do not rely on a value set before
    # resolve() has run.
    recon: bool = False
    recon_weight: float = 1.0  # weight of the MSE recon term; only used by objective="probe_recon"
    dense_early: int = 0  # log a snapshot at EVERY step for the first N optimizer steps (0 = disabled)
    device: Optional[str] = None  # resolved to "cuda" or "cpu" in resolve()
    tag: Optional[str] = None  # optional suffix used when deriving run_id
    run_id: Optional[str] = None  # derived in resolve() if not set explicitly
    artifacts_root: str = DEFAULT_ARTIFACTS_ROOT
    max_steps: Optional[int] = None  # optional hard cap on optimizer steps (used by --smoke)
    # --- v2 experiment options (all default to the original behavior) ---
    log_schedule: str = "linear"  # "linear" (log_every/dense_early) | "log" (geometric, see train.py)
    n_snapshots: int = 80  # target number of snapshots for log_schedule="log"
    probe_train: bool = False  # also snapshot a fixed balanced TRAIN probe set -> latents_train/
    train_subset: Optional[int] = None  # fixed, seeded, class-balanced subset of N train examples
    train_steps: Optional[int] = None  # train for EXACTLY this many steps (cycling epochs; ignores epochs)
    label_noise: float = 0.0  # fraction of (pool) train labels replaced by a random WRONG label
    weight_decay: float = 0.0
    l2_scale: Optional[float] = None  # NormFace scale s: head sees s * z/||z|| (latent_norm="l2" only)
    l2_scale_learned: bool = False  # make the scale a learnable scalar initialised to l2_scale
    optimizer: str = "adam"  # "adam" | "sgd"
    momentum: float = 0.9  # sgd only
    adam_eps: float = 1e-8  # Adam epsilon
    adam_beta2: float = 0.999  # Adam beta2 (beta1 stays 0.9)
    log_opt_diag: bool = False  # write opt_* optimizer diagnostics into every snapshot (see SPEC.md)
    # --- dense optimizer-state logging (--log_opt_dense, see optdense.py / SPEC.md); all default off ---
    log_opt_dense: bool = False
    opt_dump_dir: Optional[str] = None  # bulky dumps; default /data2/lbastian/latent-dynamics/opt_dense/<run_id>
    opt_ref_onset: Optional[int] = None  # onset step used for the (b) near-window and the default fixed window
    opt_dump_every: int = 200  # (b) vhat every N steps ...
    opt_dump_near_every: int = 20  # ... and every N steps in [ref_onset-2000, ref_onset+150]
    opt_arm_start: Optional[int] = None  # ring mode: armed region [arm_start, arm_end]
    opt_arm_end: Optional[int] = None
    opt_ring_steps: int = 400
    opt_post_steps: int = 150
    opt_fixed_start: Optional[int] = None  # fixed-window mode (non-event runs); default [ref_onset-500, ref_onset+150]
    opt_fixed_end: Optional[int] = None
    opt_fixed_every: int = 5
    opt_quant_every: int = 10  # quantile columns every N steps outside armed windows (every step inside)
    opt_cap_gb: float = 1.5  # hard per-run cap on bulky dumps
    opt_min_free_gb: float = 50.0  # disk guard: refuse bulky writes if free space on the dump drive is below this
    opt_max_total_gb: float = 40.0  # disk guard: refuse if opt_dense/ (parent of dump dir) already exceeds this
    opt_scalar_save_every: int = 5000  # periodically save dense_scalars.npy (also at the end / on SIGTERM)
    lr_schedule: str = "constant"  # "constant" | "cosine" (per-step, over total_steps)
    augment: bool = False  # cifar10 random-crop(4)+hflip on TRAIN batches only
    amp: bool = False  # bf16 autocast for training / loss evaluation (latent snapshots stay fp32)
    channels_last: bool = False
    train_eval: str = "running"  # "running" | "full": how snapshot train_loss/train_acc are measured
    isolate_snapshot_rng: bool = False  # snapshots save/restore torch RNG so they cannot perturb training
    # --- per-step dense probe-latent logging (dense_latent.py); all default off ---
    log_latents_every: int = 0  # >0: every N steps save fixed test+train probe latents (fp16) into chunked npz in dense_dir
    log_batch_indices: bool = False  # per-step minibatch dataset indices (+ train-probe-in-batch mask)
    ckpt_steps: Optional[str] = None  # "s1,s2,..": save model+optimizer state at these steps and step+1
    ckpt_logspace: int = 0  # >0: ~K log-spaced checkpoint steps (and step+1)
    dense_dir: Optional[str] = None  # output dir for the above (required if any is on)
    lr_mult_enc: float = 1.0  # per-block lr multipliers (AE layer-timescale control); 1.0 = off
    lr_mult_dec: float = 1.0
    linear_ae: bool = False  # autoencoder only: encoder = one Linear(img_dim->latent), decoder = one Linear(latent->img_dim)
    dense_ae_every: int = 0  # >0 (autoencoder): every K steps up to dense_ae_until save probe latents f32 + per-sample recon_err + recon_train/test to dense_ae_dir/dense_ae.npz
    dense_ae_until: int = 0
    dense_ae_dir: Optional[str] = None
    recon_eval_n: int = 10000  # recon objectives: size of the fixed train/test eval subsets for snapshot recon MSE

    def resolve(self) -> "Config":
        """Fill in derived fields in place and return self.

        - device: "cuda" if torch.cuda.is_available() else "cpu", unless the
          caller already set an explicit value.
        - in_channels / img_size: ALWAYS recomputed from `dataset` via
          DATASET_SHAPES -- an invariant, not a default-if-unset (same
          pattern as `recon` below).
        - n_classes: ALWAYS recomputed from `dataset` (currently a constant,
          N_CLASSES=10, since all three supported datasets are 10-class).
        - recon: ALWAYS recomputed as `objective in {"autoencoder", "probe_recon"}`
          -- the decoder is built exactly when the objective needs reconstruction,
          so this is an invariant, not a default-if-unset.
        - run_id: f"{dataset}_{objective}_{latent_norm}_d{latent_dim}_s{seed}"
          plus an optional "_{tag}" suffix, unless the caller already set an
          explicit value. This scheme avoids run_id collisions across a sweep
          that varies dataset / objective / latent_norm / latent_dim / seed.
          For dataset="mnist" (the default) this reproduces the original
          "mnist_{objective}_{latent_norm}_d{latent_dim}_s{seed}" scheme
          EXACTLY, so existing mnist run_ids/sweeps are unaffected.

        Call this exactly once, right after parsing/constructing the config,
        before deriving paths or touching any GPU.
        """
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"Unknown dataset: {self.dataset!r} (expected one of {VALID_DATASETS})")
        if self.objective not in VALID_OBJECTIVES:
            raise ValueError(f"Unknown objective: {self.objective!r} (expected one of {VALID_OBJECTIVES})")
        if self.latent_norm not in VALID_LATENT_NORMS:
            raise ValueError(f"Unknown latent_norm: {self.latent_norm!r} (expected one of {VALID_LATENT_NORMS})")
        for name, val, valid in (
            ("encoder_arch", self.encoder_arch, VALID_ENCODERS),
            ("log_schedule", self.log_schedule, VALID_LOG_SCHEDULES),
            ("optimizer", self.optimizer, VALID_OPTIMIZERS),
            ("lr_schedule", self.lr_schedule, VALID_LR_SCHEDULES),
            ("train_eval", self.train_eval, VALID_TRAIN_EVALS),
        ):
            if val not in valid:
                raise ValueError(f"Unknown {name}: {val!r} (expected one of {valid})")
        if not 0.0 <= self.label_noise < 1.0:
            raise ValueError("label_noise must be in [0, 1)")
        if self.train_subset is not None and self.train_subset < 1:
            raise ValueError("train_subset must be >= 1")
        if self.train_steps is not None and self.train_steps < 1:
            raise ValueError("train_steps must be >= 1")
        if self.n_snapshots < 2:
            raise ValueError("n_snapshots must be >= 2")
        if self.l2_scale is not None and self.latent_norm != "l2":
            raise ValueError("l2_scale requires latent_norm='l2'")
        if self.l2_scale_learned and self.l2_scale is None:
            raise ValueError("l2_scale_learned requires l2_scale (its initial value)")
        if self.augment and self.dataset != "cifar10":
            raise ValueError("augment is only implemented for dataset='cifar10'")
        if (self.log_latents_every or self.log_batch_indices or self.ckpt_steps or self.ckpt_logspace) \
                and not self.dense_dir:
            raise ValueError("dense logging flags require --dense_dir")
        if self.linear_ae and self.objective == "probe":
            raise ValueError("--linear_ae requires a reconstruction objective")
        if self.linear_ae and self.encoder_arch != "mlp":
            raise ValueError("--linear_ae requires encoder_arch='mlp' (it replaces the encoder)")
        if self.log_latents_every and not self.probe_train:
            raise ValueError("--log_latents_every requires --probe_train")
        if self.log_batch_indices and not self.log_latents_every:
            raise ValueError("--log_batch_indices requires --log_latents_every")

        if self.device is None:
            try:
                import torch

                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                self.device = "cpu"

        self.in_channels, self.img_size = DATASET_SHAPES[self.dataset]
        self.n_classes = N_CLASSES

        self.recon = self.objective in ("autoencoder", "probe_recon")

        if self.run_id is None:
            base = f"{self.dataset}_{self.objective}_{self.latent_norm}_d{self.latent_dim}_s{self.seed}"
            self.run_id = f"{base}_{self.tag}" if self.tag else base
        return self

    def to_json_dict(self) -> dict:
        """Resolved config as a plain JSON-serializable dict (per SPEC.md config.json)."""
        d = asdict(self)
        for k in ("log_latents_every", "log_batch_indices", "ckpt_steps", "ckpt_logspace", "dense_dir", "linear_ae", "recon_eval_n", "dense_ae_every", "dense_ae_until", "dense_ae_dir", "lr_mult_enc", "lr_mult_dec"):
            if d[k] == getattr(Config, k):
                d.pop(k)  # opt-in dense-logging flags only appear in config.json when used
        return d


@dataclass
class Paths:
    """All on-disk paths for a run, derived from a resolved Config per SPEC.md's layout."""

    artifacts_root: Path
    data_dir: Path
    run_dir: Path
    latents_dir: Path
    latents_train_dir: Path
    metrics_dir: Path
    figures_dir: Path
    config_path: Path
    probe_index_path: Path
    model_final_path: Path
    probe_train_index_path: Path
    train_index_path: Path

    def mkdirs(self) -> None:
        """Create all directories needed for this run (idempotent)."""
        for d in (self.data_dir, self.run_dir, self.latents_dir, self.metrics_dir, self.figures_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_paths(cfg: Config) -> Paths:
    """Derive all on-disk paths for a resolved Config, per SPEC.md's `ARTIFACTS` layout."""
    assert cfg.run_id is not None, "call cfg.resolve() before get_paths(cfg)"
    root = Path(cfg.artifacts_root)
    run_dir = root / "runs" / cfg.run_id
    return Paths(
        artifacts_root=root,
        data_dir=root / "data",
        run_dir=run_dir,
        latents_dir=run_dir / "latents",
        latents_train_dir=run_dir / "latents_train",
        metrics_dir=run_dir / "metrics",
        figures_dir=run_dir / "figures",
        config_path=run_dir / "config.json",
        probe_index_path=run_dir / "probe_index.npz",
        model_final_path=run_dir / "model_final.pt",
        probe_train_index_path=run_dir / "probe_train_index.npz",
        train_index_path=run_dir / "train_index.npz",
    )


def build_argparser() -> argparse.ArgumentParser:
    """CLI mirroring every Config field, plus --smoke and --max_steps."""
    p = argparse.ArgumentParser(
        description="Train a small MNIST encoder + linear-probe model and snapshot latents of a fixed probe set."
    )
    p.add_argument(
        "--dataset", type=str, default=Config.dataset, choices=list(VALID_DATASETS),
        help="'mnist'/'fashion_mnist' (1x28x28) or 'cifar10' (3x32x32). "
             "in_channels/img_size/n_classes are derived from this in resolve() "
             "-- there are no separate --in_channels/--img_size flags.",
    )
    p.add_argument("--latent_dim", type=int, default=Config.latent_dim)
    p.add_argument("--n_probe", type=int, default=Config.n_probe)

    bal = p.add_mutually_exclusive_group()
    bal.add_argument(
        "--probe_balanced", dest="probe_balanced", action="store_true", default=Config.probe_balanced,
        help="Sample the fixed probe set balanced across the 10 classes (default).",
    )
    bal.add_argument(
        "--no_probe_balanced", dest="probe_balanced", action="store_false",
        help="Sample the probe set uniformly at random (not class-balanced).",
    )

    p.add_argument("--log_every", type=int, default=Config.log_every, help="Snapshot latents every N steps.")
    p.add_argument("--epochs", type=int, default=Config.epochs)
    p.add_argument("--batch_size", type=int, default=Config.batch_size)
    p.add_argument("--lr", type=float, default=Config.lr)
    p.add_argument("--seed", type=int, default=Config.seed)
    p.add_argument("--encoder_arch", type=str, default=Config.encoder_arch, choices=list(VALID_ENCODERS))
    p.add_argument(
        "--objective", type=str, default=Config.objective, choices=list(VALID_OBJECTIVES),
        help="Training objective: 'probe' (CE only, probe shapes encoder), "
             "'autoencoder' (MSE only shapes encoder; probe trained as a detached readout), "
             "'probe_recon' (CE + recon_weight*MSE, both shape encoder). "
             "NOTE: whether a decoder is built ('recon' in config.json) is derived from this -- "
             "there is no separate --recon flag.",
    )
    p.add_argument(
        "--latent_norm", type=str, default=Config.latent_norm, choices=list(VALID_LATENT_NORMS),
        help="Transform applied to the raw encoder output to form THE latent seen by the "
             "probe/decoder and saved in snapshots: 'none' (identity), 'l2' (unit-sphere "
             "projection), 'bn' (BatchNorm1d).",
    )
    p.add_argument("--recon_weight", type=float, default=Config.recon_weight, help="Only used by objective='probe_recon'.")
    p.add_argument(
        "--dense_early", type=int, default=Config.dense_early,
        help="Snapshot latents at EVERY optimizer step for the first N steps (in addition to "
             "step 0, every log_every steps, and the final step). 0 disables dense early logging.",
    )
    # --- v2 experiment options (defaults reproduce the original behavior) ---
    p.add_argument("--log_schedule", type=str, default=Config.log_schedule, choices=list(VALID_LOG_SCHEDULES),
                   help="'linear': step 0 + every log_every (+dense_early). 'log': step 0, steps 1,2,3.. densely, "
                        "then geometric spacing to the final step, ~n_snapshots total (log_every/dense_early ignored).")
    p.add_argument("--n_snapshots", type=int, default=Config.n_snapshots, help="Target snapshot count for --log_schedule log.")
    p.add_argument("--probe_train", action="store_true",
                   help="Also snapshot a fixed class-balanced set of n_probe TRAIN examples (from the train pool) to latents_train/.")
    p.add_argument("--train_subset", type=int, default=None, help="Train on a fixed, seeded, class-balanced subset of N training examples.")
    p.add_argument("--train_steps", type=int, default=None,
                   help="Train for exactly this many optimizer steps, cycling through epochs as needed (overrides --epochs; "
                        "use with tiny --train_subset). Unlike --max_steps (a cap), this can exceed epochs*steps_per_epoch.")
    p.add_argument("--label_noise", type=float, default=Config.label_noise,
                   help="Fraction of training-pool labels replaced by a uniformly random WRONG label (fixed per seed; test untouched).")
    p.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    p.add_argument("--l2_scale", type=float, default=None, help="latent_norm=l2 only: head input = l2_scale * z/||z||. Snapshots keep z/||z||.")
    p.add_argument("--l2_scale_learned", action="store_true", help="Make l2_scale a learnable scalar (init = --l2_scale).")
    p.add_argument("--optimizer", type=str, default=Config.optimizer, choices=list(VALID_OPTIMIZERS))
    p.add_argument("--momentum", type=float, default=Config.momentum, help="SGD momentum.")
    p.add_argument("--adam_eps", type=float, default=Config.adam_eps, help="Adam epsilon (default 1e-8).")
    p.add_argument("--adam_beta2", type=float, default=Config.adam_beta2, help="Adam beta2 (default 0.999).")
    p.add_argument("--log_opt_diag", action="store_true",
                   help="Write opt_* optimizer diagnostics (norms, Adam effective step, Hessian top eigenvalue) into snapshots.")
    p.add_argument("--log_opt_dense", action="store_true",
                   help="Dense per-step Adam-state logging (optdense.py, SPEC.md). Requires --optimizer adam.")
    p.add_argument("--opt_dump_dir", type=str, default=None, help="Dir for bulky v_hat/m_hat/g' dumps (default /data2/lbastian/latent-dynamics/opt_dense/<run_id>).")
    p.add_argument("--opt_ref_onset", type=int, default=None, help="Onset step for the (b) near-window / default fixed window.")
    p.add_argument("--opt_dump_every", type=int, default=Config.opt_dump_every)
    p.add_argument("--opt_dump_near_every", type=int, default=Config.opt_dump_near_every)
    p.add_argument("--opt_arm_start", type=int, default=None, help="Ring-buffer mode: first armed step.")
    p.add_argument("--opt_arm_end", type=int, default=None, help="Ring-buffer mode: last armed step.")
    p.add_argument("--opt_ring_steps", type=int, default=Config.opt_ring_steps)
    p.add_argument("--opt_post_steps", type=int, default=Config.opt_post_steps)
    p.add_argument("--opt_fixed_start", type=int, default=None)
    p.add_argument("--opt_fixed_end", type=int, default=None)
    p.add_argument("--opt_fixed_every", type=int, default=Config.opt_fixed_every)
    p.add_argument("--opt_quant_every", type=int, default=Config.opt_quant_every)
    p.add_argument("--opt_cap_gb", type=float, default=Config.opt_cap_gb)
    p.add_argument("--opt_min_free_gb", type=float, default=Config.opt_min_free_gb)
    p.add_argument("--opt_max_total_gb", type=float, default=Config.opt_max_total_gb)
    p.add_argument("--opt_scalar_save_every", type=int, default=Config.opt_scalar_save_every)
    p.add_argument("--lr_schedule", type=str, default=Config.lr_schedule, choices=list(VALID_LR_SCHEDULES),
                   help="'cosine' anneals lr to 0 per optimizer step over the total number of steps.")
    p.add_argument("--augment", action="store_true", help="cifar10: random crop (pad 4) + horizontal flip on training batches only.")
    p.add_argument("--amp", action="store_true", help="bf16 autocast for training and loss/acc evaluation (probe latents stay fp32). CUDA only.")
    p.add_argument("--channels_last", action="store_true", help="channels_last memory format (useful for resnet18 on GPU).")
    p.add_argument("--train_eval", type=str, default=Config.train_eval, choices=list(VALID_TRAIN_EVALS),
                   help="'running': snapshot train_loss/acc = mean over training batches since previous snapshot (step 0: full pass). "
                        "'full': eval-mode pass over the whole (un-augmented, noisy-labelled) train pool at each snapshot.")
    p.add_argument("--isolate_snapshot_rng", action="store_true",
                   help="Save/restore the torch RNG around each snapshot so training is bit-identical regardless of snapshot schedule.")
    p.add_argument("--log_latents_every", type=int, default=Config.log_latents_every,
                   help="Every N steps save test+train probe latents (fp16), head W/b, per-sample train-probe loss to --dense_dir chunks. 0=off.")
    p.add_argument("--log_batch_indices", action="store_true", help="Also save per-step minibatch dataset indices + train-probe-in-batch mask.")
    p.add_argument("--ckpt_steps", type=str, default=None, help="Comma/space list of steps; save model+optimizer at each and step+1.")
    p.add_argument("--ckpt_logspace", type=int, default=Config.ckpt_logspace, help="Save ~K log-spaced checkpoints (each with step+1).")
    p.add_argument("--dense_dir", type=str, default=None, help="Output dir for dense latent logging / checkpoints.")
    p.add_argument("--lr_mult_enc", type=float, default=1.0, help="Multiply lr of encoder params (default 1 = off).")
    p.add_argument("--lr_mult_dec", type=float, default=1.0, help="Multiply lr of decoder params (default 1 = off).")
    p.add_argument("--linear_ae", action="store_true",
                   help="Linear autoencoder: encoder and decoder are each a single Linear layer (with bias), no nonlinearity.")
    p.add_argument("--dense_ae_every", type=int, default=0, help="AE: every K steps (<= --dense_ae_until) save probe latents (f32), per-sample recon_err, recon_train/test to --dense_ae_dir/dense_ae.npz. Read-only, no RNG.")
    p.add_argument("--dense_ae_until", type=int, default=0)
    p.add_argument("--dense_ae_dir", type=str, default=None)
    p.add_argument("--recon_eval_n", type=int, default=Config.recon_eval_n,
                   help="Recon objectives: fixed eval subset size (train pool and test) for per-snapshot eval-mode recon MSE.")
    p.add_argument(
        "--device", type=str, default=None, choices=["cuda", "cpu"],
        help="Override device selection. Default: auto-detect. Pin GPUs via CUDA_VISIBLE_DEVICES, not here.",
    )
    p.add_argument("--tag", type=str, default=None, help="Optional suffix for the derived run_id.")
    p.add_argument("--run_id", type=str, default=None, help="Explicit run_id, bypassing derivation.")
    p.add_argument("--artifacts_root", type=str, default=DEFAULT_ARTIFACTS_ROOT)
    p.add_argument("--max_steps", type=int, default=None, help="Optional hard cap on total optimizer steps.")
    p.add_argument(
        "--smoke", action="store_true",
        help="Run a tiny, fast end-to-end config to smoke-test the pipeline (overrides epochs/max_steps/log_every/n_probe/batch_size and defaults tag='smoke').",
    )
    return p


def config_from_args(argv: Optional[List[str]] = None) -> Config:
    """Parse CLI args into a resolved Config, applying --smoke overrides if set."""
    args = build_argparser().parse_args(argv)
    smoke = args.smoke
    kwargs = vars(args).copy()
    kwargs.pop("smoke")
    cfg = Config(**kwargs)

    if smoke:
        cfg.epochs = 1
        cfg.max_steps = cfg.max_steps or 100
        cfg.log_every = min(cfg.log_every, 20)
        cfg.n_probe = min(cfg.n_probe, 100)
        cfg.batch_size = min(cfg.batch_size, 64)
        if cfg.tag is None:
            cfg.tag = "smoke"

    cfg.resolve()
    return cfg
