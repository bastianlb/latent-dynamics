#!/usr/bin/env python
"""
plot.py
=======

CLI that reads `<run_dir>/metrics/timeseries.{csv,npz}` (written by
analyze.py) and renders PNG figures under `<run_dir>/figures/`:

  (a) accuracy_loss.png        -- train/test accuracy + train loss vs step
  (b) class_geometry.png       -- intra-class compactness, inter-class
                                   separation, Fisher ratio vs step
                                   (NORMALIZED metrics -- shape, not scale)
  (c) topology.png             -- H0/H1 alpha-weighted lifetime sums +
                                   persistence entropy vs step (normalized)
  (d) magnitude.png            -- PMag (scale-averaged magnitude) vs step,
                                   plus the magnitude function curve itself
                                   at first vs last step (normalized)
  (e) centroid_distance_heatmap.png -- class-centroid distance matrix,
                                   first step vs last step, side by side
                                   (normalized)
  (f) scale_vs_shape.png       -- contrast figure: raw_mean_norm /
                                   raw_separation_mean (un-normalized, shows
                                   the latent-space scale inflation) next to
                                   the NORMALIZED separation_mean /
                                   fisher_ratio (the true shape dynamics).
                                   Only rendered if analyze.py wrote `raw_*`
                                   columns (it always does, post-normalization
                                   update).
  (g) perclass_topology.png    -- small-multiples of per-class H1
                                   lifetime-sum (alpha=1.0, falls back to
                                   per-class H1 entropy) vs step, one subplot
                                   per class. Only rendered if analyze.py was
                                   run with --per_class_topology (off by
                                   default), i.e. only if `perclass_*`
                                   columns are present in the timeseries.

All metric columns except the `raw_*` ones reflect whatever normalization
mode analyze.py was run with (see its `--normalize` flag / docstring); the
mode actually used is available as the 0-d string array `normalize_mode` in
timeseries.npz and is shown in figure (f)'s title when present.

Usage
-----
    python -m latentdyn.pipeline.plot --run_dir /vol/bitbucket/lbastian/latent-dynamics/runs/<run_id>
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP thread pools BEFORE numpy is imported (shared 128-core box).
from latentdyn._threads import cap_threads

cap_threads(8)

import argparse
import logging
import sys

import numpy as np

from latentdyn.io import load_timeseries as _io_load_timeseries  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[plot] %(message)s")
log = logging.getLogger(__name__)

DPI = 150


def _load_timeseries(run_dir: str) -> dict:
    return _io_load_timeseries(run_dir, missing="raise")


def plot_accuracy_loss(ts: dict, out_path: str) -> None:
    step = ts["step"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(step, ts["train_acc"], label="train_acc", color="tab:blue")
    ax1.plot(step, ts["test_acc"], label="test_acc", color="tab:orange")
    ax1.set_xlabel("step")
    ax1.set_ylabel("accuracy")
    ax1.set_title("Train / test accuracy vs step")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(step, ts["train_loss"], label="train_loss", color="tab:red")
    ax2.set_xlabel("step")
    ax2.set_ylabel("loss")
    ax2.set_title("Train loss vs step")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_class_geometry(ts: dict, out_path: str) -> None:
    step = ts["step"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    ax.plot(step, ts["compactness_centroid"], label="mean dist to centroid")
    ax.plot(step, ts["compactness_pairwise"], label="mean intra-class pairwise dist")
    ax.set_xlabel("step")
    ax.set_ylabel("distance")
    ax.set_title("Intra-class compactness")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(step, ts["separation_mean"], label="mean centroid-centroid dist")
    ax.plot(step, ts["separation_min"], label="min centroid-centroid dist")
    ax.set_xlabel("step")
    ax.set_ylabel("distance")
    ax.set_title("Inter-class separation")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(step, ts["fisher_ratio"], color="tab:green")
    ax.set_xlabel("step")
    ax.set_ylabel("ratio")
    ax.set_title("Fisher-like separation ratio\n(between / within scatter)")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_topology(ts: dict, out_path: str) -> None:
    step = ts["step"]
    alpha_keys = sorted(
        {k.split("_alpha")[-1] for k in ts.keys() if k.startswith("h0_lifetime_sum_alpha")},
        key=float,
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    ax = axes[0]
    for a in alpha_keys:
        key = f"h0_lifetime_sum_alpha{a}"
        if key in ts:
            ax.plot(step, ts[key], label=f"alpha={a}")
    ax.set_xlabel("step")
    ax.set_ylabel("L_alpha(H0)")
    ax.set_title("H0 alpha-weighted lifetime sum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    for a in alpha_keys:
        key = f"h1_lifetime_sum_alpha{a}"
        if key in ts:
            ax.plot(step, ts[key], label=f"alpha={a}")
    ax.set_xlabel("step")
    ax.set_ylabel("L_alpha(H1)")
    ax.set_title("H1 alpha-weighted lifetime sum")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    if "h0_entropy" in ts:
        ax.plot(step, ts["h0_entropy"], label="H0 entropy")
    if "h1_entropy" in ts:
        ax.plot(step, ts["h1_entropy"], label="H1 entropy")
    ax.set_xlabel("step")
    ax.set_ylabel("persistence entropy")
    ax.set_title("Persistence entropy")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_magnitude(ts: dict, out_path: str) -> None:
    step = ts["step"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(step, ts["pmag"], color="tab:purple")
    ax1.set_xlabel("step")
    ax1.set_ylabel("PMag")
    ax1.set_title("PMag (scale-averaged magnitude) vs step")
    ax1.grid(alpha=0.3)

    if "magnitude_curve_scales" in ts and "magnitude_curve_values" in ts:
        first_scales = ts["magnitude_curve_scales"][0]
        first_mag = ts["magnitude_curve_values"][0]
        last_scales = ts["magnitude_curve_scales"][-1]
        last_mag = ts["magnitude_curve_values"][-1]
        ax2.plot(first_scales, first_mag, "o-", label=f"step {int(step[0])}")
        ax2.plot(last_scales, last_mag, "o-", label=f"step {int(step[-1])}")
        ax2.set_xlabel("scale t")
        ax2.set_ylabel("Mag(t)")
        ax2.set_title("Magnitude function: first vs last step")
        ax2.legend(fontsize=8)
        ax2.grid(alpha=0.3)
    else:
        ax2.axis("off")
        ax2.text(0.5, 0.5, "magnitude curve not available", ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_centroid_distance_heatmap(ts: dict, out_path: str) -> None:
    if "centroid_dist_matrix" not in ts:
        log.warning("centroid_dist_matrix not found in timeseries; skipping heatmap plot.")
        return

    mats = ts["centroid_dist_matrix"]  # [n_steps, C, C]
    step = ts["step"]
    first_mat, last_mat = mats[0], mats[-1]
    vmax = np.nanmax([first_mat, last_mat])

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, mat, s in zip(axes, [first_mat, last_mat], [step[0], step[-1]]):
        im = ax.imshow(mat, cmap="viridis", vmin=0, vmax=vmax)
        ax.set_title(f"Class-centroid distances @ step {int(s)}")
        ax.set_xlabel("class")
        ax.set_ylabel("class")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def _normalize_mode_str(ts: dict) -> str:
    """Pull the human-readable normalize mode out of ts['normalize_mode']
    (a 0-d unicode array written by analyze.py), or 'unknown' if absent."""
    if "normalize_mode" not in ts:
        return "unknown"
    try:
        return str(np.asarray(ts["normalize_mode"]).item())
    except Exception:
        return "unknown"


def plot_scale_vs_shape(ts: dict, out_path: str) -> None:
    """Contrast figure: SCALE (raw, un-normalized latent-space inflation)
    next to SHAPE (normalized class geometry under the existing metric
    names). Motivation: in the first real run the mean latent norm inflated
    ~0.57 -> 23.7 over training, which would otherwise dominate every raw
    distance/topology metric. Left panel shows that raw inflation directly
    (`raw_mean_norm`, `raw_separation_mean`); right panel shows the
    NORMALIZED `separation_mean` / `fisher_ratio`, which should move on a
    bounded scale and reflect genuine shape dynamics instead.

    Requires analyze.py's `raw_*` columns (always written post-normalization
    update); skips cleanly (log + return) if an older timeseries.npz lacks
    them.
    """
    required = ("raw_mean_norm", "raw_separation_mean", "separation_mean", "fisher_ratio")
    missing = [k for k in required if k not in ts]
    if missing:
        log.warning(f"scale_vs_shape plot needs columns {missing}; skipping "
                     f"(re-run analyze.py to populate raw_* columns).")
        return

    step = ts["step"]
    mode = _normalize_mode_str(ts)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(step, ts["raw_mean_norm"], label="raw_mean_norm", color="tab:red")
    ax1.plot(step, ts["raw_separation_mean"], label="raw_separation_mean", color="tab:orange")
    ax1.set_xlabel("step")
    ax1.set_ylabel("distance (raw / un-normalized units)")
    ax1.set_title("SCALE: raw latent-space inflation")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2b = ax2.twinx()
    l1, = ax2.plot(step, ts["separation_mean"], color="tab:blue",
                    label="separation_mean (normalized)")
    l2, = ax2b.plot(step, ts["fisher_ratio"], color="tab:green",
                     label="fisher_ratio (normalized)")
    ax2.set_xlabel("step")
    ax2.set_ylabel("separation_mean (normalized)", color="tab:blue")
    ax2b.set_ylabel("fisher_ratio", color="tab:green")
    ax2.tick_params(axis="y", labelcolor="tab:blue")
    ax2b.tick_params(axis="y", labelcolor="tab:green")
    ax2.set_title(f"SHAPE: normalized class geometry (mode={mode})")
    ax2.legend(handles=[l1, l2], fontsize=8, loc="best")
    ax2.grid(alpha=0.3)

    fig.suptitle("Scale vs. shape: raw inflation vs. normalized class geometry")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def plot_perclass_topology(ts: dict, out_path: str) -> None:
    """Small multiples (one subplot per class) of per-class H1 topology vs
    step. Prefers the H1 alpha=1.0 lifetime sum (total persistence);
    falls back to H1 persistence entropy if lifetime-sum columns aren't
    present. Only meaningful if analyze.py was run with
    `--per_class_topology` (off by default -- it multiplies the ripser cost
    by the number of classes); skips cleanly (log + return) otherwise.
    """
    import re

    step = ts["step"]
    lifetime_pat = re.compile(r"^perclass_(\d+)_h1_lifetime_sum_alpha1\.0$")
    entropy_pat = re.compile(r"^perclass_(\d+)_h1_entropy$")

    lifetime_classes = sorted(int(m.group(1)) for k in ts.keys()
                               if (m := lifetime_pat.match(k)))
    if lifetime_classes:
        classes = lifetime_classes
        key_fn = lambda c: f"perclass_{c}_h1_lifetime_sum_alpha1.0"
        ylabel = "H1 total persistence (alpha=1.0)"
    else:
        entropy_classes = sorted(int(m.group(1)) for k in ts.keys()
                                  if (m := entropy_pat.match(k)))
        if not entropy_classes:
            log.warning("No perclass_* H1 topology columns found; skipping "
                         "perclass_topology plot (re-run analyze.py with "
                         "--per_class_topology to enable it).")
            return
        classes = entropy_classes
        key_fn = lambda c: f"perclass_{c}_h1_entropy"
        ylabel = "H1 persistence entropy"

    n = len(classes)
    ncols = min(5, n)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.6 * nrows), squeeze=False)

    for idx, c in enumerate(classes):
        ax = axes[idx // ncols][idx % ncols]
        key = key_fn(c)
        if key in ts:
            ax.plot(step, ts[key], color="tab:purple")
        ax.set_title(f"class {c}", fontsize=9)
        ax.grid(alpha=0.3)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("step", fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=8)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    fig.suptitle(f"Per-class H1 topology vs step ({ylabel})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)


def make_all_plots(run_dir: str) -> None:
    ts = _load_timeseries(run_dir)
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    jobs = [
        ("accuracy_loss.png", plot_accuracy_loss),
        ("class_geometry.png", plot_class_geometry),
        ("topology.png", plot_topology),
        ("magnitude.png", plot_magnitude),
        ("centroid_distance_heatmap.png", plot_centroid_distance_heatmap),
        ("scale_vs_shape.png", plot_scale_vs_shape),
        ("perclass_topology.png", plot_perclass_topology),
    ]
    for fname, fn in jobs:
        out_path = os.path.join(fig_dir, fname)
        try:
            fn(ts, out_path)
            if os.path.exists(out_path):
                log.info(f"Wrote {out_path}")
            else:
                log.info(f"Skipped {fname} (required columns not present; see warning above).")
        except Exception:
            log.exception(f"Failed to render {fname}; skipping.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                         help="Run directory containing metrics/timeseries.npz "
                              "(e.g. /vol/bitbucket/lbastian/latent-dynamics/runs/<run_id>)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        log.error(f"run_dir does not exist: {run_dir}")
        sys.exit(1)

    make_all_plots(run_dir)
    log.info("Done.")


if __name__ == "__main__":
    main()
