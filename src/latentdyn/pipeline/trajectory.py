#!/usr/bin/env python
"""
trajectory.py
==============

CLI that visualizes how a run's PROBE latent cloud moves over training
iterations, for a SINGLE run directory (per SPEC.md's on-disk contract):

    <run_dir>/latents/step_*.npz   (one snapshot per logging step)

PCA basis (IMPORTANT -- read before interpreting the plots)
-------------------------------------------------------------
We fit a SINGLE global 2-D PCA on the **final** snapshot's latents (the
step where training has converged and classes are maximally spread out),
then PROJECT every other snapshot's latents into that *same* basis (same
`components_`/`mean_`; no re-fitting per frame). This keeps the axes
meaningful and consistent across the whole movie: early, still-entangled
snapshots are viewed through the lens of the basis that best separates
the *final* classes, so you watch clusters peel apart along fixed,
interpretable directions instead of the axes spinning around frame to
frame (which independently-fit-per-frame PCA would do).

If `latent_dim == 2` there is nothing to reduce, so PCA is skipped
entirely and the two raw latent dimensions are used as-is (still a
single, frame-independent basis, trivially).

Axis limits are fixed for the whole run: all snapshots are projected
into the chosen basis up front and the global [min, max] (+ a small
margin) over that whole set is used as the plot limits for every
panel/frame, so the grid/animation never rescales or jumps.

Outputs (written to <run_dir>/figures/):
    trajectory_grid.png  -- small-multiples over evenly spaced steps
                             (up to 4x4 = 16 panels), points colored by
                             class, titled with step/epoch/test_acc.
    trajectory.gif        -- FuncAnimation + PillowWriter movie over all
                             (or --n_frames evenly-subsampled) snapshots,
                             same fixed axes, class-colored, with a
                             step/epoch/test_acc label per frame.

Usage
-----
    python -m latentdyn.pipeline.trajectory --run_dir /vol/bitbucket/lbastian/latent-dynamics/runs/<run_id> \
        [--n_frames 60] [--grid_steps 16]
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP thread pools BEFORE numpy is imported (shared 128-core box).
from latentdyn._threads import cap_threads

cap_threads(8)

import argparse
import glob
import logging
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.lines import Line2D

from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO, format="[trajectory] %(message)s")
log = logging.getLogger(__name__)

DPI_GRID = 130
DPI_GIF = 100
MAX_GRID_PANELS = 16          # "up to 4x4" small-multiples per spec
DEFAULT_MAX_GIF_FRAMES = 60   # keep the .gif reasonably small/fast by default
N_CLASSES = 10
CMAP = plt.get_cmap("tab10")


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

from latentdyn.io import find_snapshot_files as _find_snapshot_files, load_snapshot as _io_load_snapshot  # noqa: E402


def _load_snapshot(path: str) -> dict:
    return _io_load_snapshot(path, float64=True, extras=("test_acc",))


# ---------------------------------------------------------------------------
# Frame/panel selection
# ---------------------------------------------------------------------------

def _evenly_spaced_indices(n: int, k: int) -> list[int]:
    """k evenly spaced indices into range(n), always including 0 and n-1."""
    if k >= n:
        return list(range(n))
    if k <= 1:
        return [n - 1]
    raw = np.linspace(0, n - 1, k)
    idx = sorted(set(int(round(i)) for i in raw))
    if idx[0] != 0:
        idx = [0] + idx
    if idx[-1] != n - 1:
        idx = idx + [n - 1]
    return sorted(set(idx))


# ---------------------------------------------------------------------------
# Projection (the single global basis)
# ---------------------------------------------------------------------------

def build_projection(snapshots: list[dict]):
    """Fit the global 2-D basis on the FINAL snapshot; project every snapshot
    into that same basis. Returns (projected, xlim, ylim, basis_info)."""
    final = snapshots[-1]
    D = final["latents"].shape[1]

    if D == 2:
        basis_info = "raw 2-D latents (no PCA; latent_dim == 2)"

        def project(x):
            return x
    else:
        pca = PCA(n_components=2)
        pca.fit(final["latents"])
        evr = pca.explained_variance_ratio_
        basis_info = (
            f"PCA(2) fit on FINAL snapshot only (step={final['step']}), "
            f"all steps projected into that fixed basis "
            f"(explained var ratio=[{evr[0]:.3f}, {evr[1]:.3f}])"
        )

        def project(x):
            return pca.transform(x)

    projected = [project(s["latents"]) for s in snapshots]

    all_xy = np.concatenate(projected, axis=0)
    xmin, xmax = float(all_xy[:, 0].min()), float(all_xy[:, 0].max())
    ymin, ymax = float(all_xy[:, 1].min()), float(all_xy[:, 1].max())
    xpad = 0.05 * (xmax - xmin) if xmax > xmin else 1.0
    ypad = 0.05 * (ymax - ymin) if ymax > ymin else 1.0
    xlim = (xmin - xpad, xmax + xpad)
    ylim = (ymin - ypad, ymax + ypad)

    return projected, xlim, ylim, basis_info


# ---------------------------------------------------------------------------
# Plot: small-multiples grid
# ---------------------------------------------------------------------------

def _scatter_panel(ax, xy, labels, xlim, ylim, title):
    for c in range(N_CLASSES):
        m = labels == c
        if m.any():
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.7, color=CMAP(c), linewidths=0)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def make_grid(run_id, snapshots, projected, xlim, ylim, basis_info, grid_steps, out_path):
    n = len(snapshots)
    k = max(1, min(grid_steps, MAX_GRID_PANELS, n))
    idx = _evenly_spaced_indices(n, k)

    ncols = min(4, int(np.ceil(np.sqrt(len(idx)))))
    nrows = int(np.ceil(len(idx) / ncols))

    panel_h = 3.0
    # Reserve a FIXED number of inches at the top for suptitle + legend (not
    # a fraction of fig height), so the band stays a constant size regardless
    # of how many rows the grid has, and a fixed small bottom margin so a
    # taller figure (more rows) doesn't inherit a proportionally huge blank
    # strip from matplotlib's default (fractional) bottom margin.
    title_gap_in = 0.10     # distance from fig top down to the suptitle anchor
    legend_gap_in = 0.55    # distance from fig top down to the legend anchor
    top_margin_in = 1.30    # distance from fig top down to the axes area
    bottom_margin_in = 0.15

    fig_h = panel_h * nrows + top_margin_in + bottom_margin_in
    fig_w = 3.0 * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes = np.atleast_1d(axes).ravel()

    for ax, i in zip(axes, idx):
        s = snapshots[i]
        title = f"step {s['step']} (epoch {s['epoch']:.2f})\ntest_acc={s['test_acc']:.3f}"
        _scatter_panel(ax, projected[i], s["labels"], xlim, ylim, title)
    for ax in axes[len(idx):]:
        ax.axis("off")

    top_frac = 1.0 - top_margin_in / fig_h
    bottom_frac = bottom_margin_in / fig_h
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac, hspace=0.55, wspace=0.2)

    handles = [Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=CMAP(c),
                       markeredgewidth=0, markersize=6, label=str(c)) for c in range(N_CLASSES)]
    fig.legend(handles=handles, title="class", loc="upper center",
               ncol=N_CLASSES, bbox_to_anchor=(0.5, 1.0 - legend_gap_in / fig_h), fontsize=8)
    fig.suptitle(f"{run_id}: probe latent trajectory (2-D projection)\n{basis_info}",
                 fontsize=10, y=1.0 - title_gap_in / fig_h)
    fig.savefig(out_path, dpi=DPI_GRID)
    plt.close(fig)
    log.info(f"Wrote {out_path} ({len(idx)} panels)")


# ---------------------------------------------------------------------------
# Plot: animated gif
# ---------------------------------------------------------------------------

def make_gif(run_id, snapshots, projected, xlim, ylim, basis_info, n_frames, out_path):
    n = len(snapshots)
    if n_frames is None:
        n_frames = n if n <= DEFAULT_MAX_GIF_FRAMES else DEFAULT_MAX_GIF_FRAMES
    k = max(2, min(n_frames, n))
    idx = _evenly_spaced_indices(n, k)
    if k < n:
        log.info(f"Subsampling gif to {k}/{n} frames (evenly spaced, incl. first/last).")

    fig, ax = plt.subplots(figsize=(5.5, 6.0))
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xticks([])
    ax.set_yticks([])

    scatters = {c: ax.scatter([], [], s=8, alpha=0.7, color=CMAP(c), linewidths=0, label=str(c))
                for c in range(N_CLASSES)}
    ax.legend(loc="upper right", fontsize=7, title="class", ncol=2)
    title_txt = ax.set_title("", fontsize=10)
    # Short one-line suptitle (the full PCA-basis rationale lives in the
    # script docstring / trajectory_grid.png / logs; this is just a label).
    fig.suptitle(f"{run_id} -- probe latent trajectory (fixed PCA basis)", fontsize=9)
    fig.text(0.5, 0.955, basis_info, ha="center", va="top", fontsize=6, color="dimgray",
              wrap=True)
    fig.subplots_adjust(top=0.86, bottom=0.05, left=0.08, right=0.95)

    def update(frame_i):
        i = idx[frame_i]
        s = snapshots[i]
        xy = projected[i]
        labels = s["labels"]
        for c in range(N_CLASSES):
            m = labels == c
            scatters[c].set_offsets(xy[m] if m.any() else np.empty((0, 2)))
        title_txt.set_text(
            f"step {s['step']} (epoch {s['epoch']:.2f}) test_acc={s['test_acc']:.3f}")
        return list(scatters.values()) + [title_txt]

    anim = animation.FuncAnimation(fig, update, frames=len(idx), blit=False)
    writer = animation.PillowWriter(fps=6)
    anim.save(out_path, writer=writer, dpi=DPI_GIF)
    plt.close(fig)
    log.info(f"Wrote {out_path} ({len(idx)} frames @ 6fps)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run_dir", required=True,
                         help="Run dir containing latents/step_*.npz, e.g. "
                              "/vol/bitbucket/lbastian/latent-dynamics/runs/<run_id>")
    parser.add_argument("--n_frames", type=int, default=None,
                         help="Number of evenly spaced frames in trajectory.gif "
                              f"(default: all snapshots, capped at {DEFAULT_MAX_GIF_FRAMES})")
    parser.add_argument("--grid_steps", type=int, default=MAX_GRID_PANELS,
                         help="Number of evenly spaced panels in trajectory_grid.png "
                              f"(default {MAX_GRID_PANELS}; capped at {MAX_GRID_PANELS}, i.e. 4x4)")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        log.error(f"run_dir does not exist: {run_dir}")
        sys.exit(1)
    run_id = os.path.basename(run_dir.rstrip(os.sep))

    files = _find_snapshot_files(run_dir)
    log.info(f"Found {len(files)} snapshot files in {run_dir}/latents/")
    snapshots = [_load_snapshot(p) for p in files]
    order = np.argsort([s["step"] for s in snapshots])  # defensive; filenames already sorted
    snapshots = [snapshots[i] for i in order]

    projected, xlim, ylim, basis_info = build_projection(snapshots)
    log.info(f"Projection basis: {basis_info}")
    log.info(f"Fixed axis limits: x={xlim}, y={ylim}")

    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    make_grid(run_id, snapshots, projected, xlim, ylim, basis_info,
              args.grid_steps, os.path.join(fig_dir, "trajectory_grid.png"))
    make_gif(run_id, snapshots, projected, xlim, ylim, basis_info,
             args.n_frames, os.path.join(fig_dir, "trajectory.gif"))
    log.info("Done.")


if __name__ == "__main__":
    main()
