#!/usr/bin/env python
"""
analyze.py
==========

CLI that loads every `latents/step_*.npz` snapshot in a run directory (per
SPEC.md's on-disk contract), computes every metric in `metrics.py` for each
snapshot, and writes the metrics timeseries:

    <run_dir>/metrics/timeseries.csv   (one row per step, scalar columns only)
    <run_dir>/metrics/timeseries.npz   (same scalars as arrays keyed by name,
                                         plus array-valued metrics such as the
                                         [C, C] class-centroid-distance matrix
                                         and the magnitude-function curve,
                                         stacked over steps)

Normalization contract (scale vs shape)
----------------------------------------
Real runs show the latent space inflating ~isotropically over training
(e.g. mean latent norm 0.57 -> 23.7 on mnist_probe_d16), which swamps raw
distance/topology metrics with SCALE rather than SHAPE. To separate the two:

  - For every snapshot we FIRST compute a handful of scale indicators from
    the RAW (un-normalized) latents, stored with a `raw_` prefix:
    `raw_mean_norm`, `raw_median_norm`, `raw_mean_pairwise_dist`,
    `raw_separation_mean`, `raw_fisher_ratio` (the last is a sanity check --
    see below). These columns tell the SCALE story (inflation).

  - We THEN apply `metrics.normalize_latents(latents, mode=args.normalize)`
    (`--normalize`, default "mean_pairwise") and compute EVERY OTHER metric
    in `metrics.py` -- class geometry, persistence, PMag -- on the
    NORMALIZED latents, under their existing, unchanged names
    (`separation_mean`, `compactness_centroid`, `fisher_ratio`,
    `h0_entropy`/`h1_entropy`, `h*_lifetime_sum_alpha*`, `pmag`,
    `centroid_dist_matrix`, `magnitude_curve_*`, `perclass_*`, ...). This
    keeps the downstream plotting/compare contract stable: those names now
    refer to the NORMALIZED (shape) quantities, while `raw_*` captures scale.

  - The chosen mode is recorded as a 0-d unicode array `normalize_mode` in
    timeseries.npz, and as a leading `# normalize_mode=<mode>` comment line
    above the header in timeseries.csv.

  - Sanity check: the Fisher-like separation ratio is mathematically
    scale-invariant under an isotropic rescale (numerator and denominator
    are both sums of squared distances, so a global scale factor cancels),
    so `fisher_ratio` (normalized) should equal `raw_fisher_ratio` (raw) when
    `--normalize mean_pairwise` is used; this is NOT guaranteed under
    `--normalize zscore`, which is anisotropic.

Robustness: if a metric (or a whole snapshot) fails, the failure is logged
and the affected scalar cells are filled with NaN -- one bad snapshot never
crashes the whole run.

Usage
-----
    python -m latentdyn.pipeline.analyze --run_dir /vol/bitbucket/lbastian/latent-dynamics/runs/<run_id> \
        [--normalize {none,mean_pairwise,zscore}] [--per_class_topology]
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP thread pools BEFORE numpy is imported (shared 128-core box).
from latentdyn._threads import cap_threads

cap_threads(8)

import argparse
import csv
import glob
import json
import logging
import sys
import traceback

import numpy as np

from latentdyn import metrics as M

logging.basicConfig(level=logging.INFO, format="[analyze] %(message)s")
log = logging.getLogger(__name__)


# Keys metrics.compute_all_metrics() returns that are array-valued (not a
# single scalar). Everything else returned by compute_all_metrics is treated
# as a scalar CSV column.
ARRAY_VALUED_KEYS = {
    "centroid_dist_matrix",
    "magnitude_curve_scales",
    "magnitude_curve_values",
}


from latentdyn.io import find_snapshot_files as _find_snapshot_files, load_snapshot as _load_snapshot  # noqa: E402


RAW_SCALE_KEYS = (
    "raw_mean_norm",
    "raw_median_norm",
    "raw_mean_pairwise_dist",
    "raw_separation_mean",
    "raw_fisher_ratio",
)


def analyze_run(run_dir: str, alphas=M.DEFAULT_ALPHAS, n_scales: int = M.DEFAULT_N_SCALES,
                 per_class_topology: bool = False, normalize: str = "mean_pairwise") -> dict:
    """Compute the full metrics timeseries for a run directory.

    For every snapshot: (1) record `raw_*` scale indicators from the
    UN-normalized latents (RAW_SCALE_KEYS, computed via
    `metrics.global_distance_stats` / `metrics.inter_class_separation` /
    `metrics.fisher_separation_ratio` on the raw latents); (2) apply
    `metrics.normalize_latents(latents, mode=normalize)` and compute every
    other metric in `metrics.py` on the NORMALIZED latents under their
    existing names. See the module docstring for the full contract.

    Returns the dict of per-step arrays that gets written to
    metrics/timeseries.{csv,npz} (also returned so the self-test / callers
    can inspect it directly without re-reading the file). Also includes the
    key "normalize_mode" (the mode string actually used).
    """
    files = _find_snapshot_files(run_dir)
    log.info(f"Found {len(files)} snapshot files in {run_dir}/latents/ "
              f"(normalize={normalize!r})")

    # Base (always-present) columns from the snapshot itself.
    base_cols = ["step", "epoch", "train_loss", "train_acc", "test_acc"]
    rows: list[dict] = []
    array_series: dict[str, list] = {k: [] for k in ARRAY_VALUED_KEYS}

    # Collect the union of scalar metric names across all steps so the CSV
    # has a stable header even if some steps fail some metrics.
    scalar_keys_seen: list[str] = []

    for i, path in enumerate(files):
        try:
            snap = _load_snapshot(path)
        except Exception:
            log.error(f"Failed to load snapshot {path}; skipping entirely.\n{traceback.format_exc()}")
            continue

        row = {k: snap[k] for k in base_cols}
        raw_latents = snap["latents"]
        labels = snap["labels"]

        # --- (1) RAW scale indicators, from the un-normalized latents. ---
        try:
            raw_gstats = M.global_distance_stats(raw_latents)
            raw_sep = M.inter_class_separation(raw_latents, labels)
            raw_fisher = M.fisher_separation_ratio(raw_latents, labels)
            row["raw_mean_norm"] = float(raw_gstats["mean_norm"])
            row["raw_median_norm"] = float(raw_gstats["median_norm"])
            row["raw_mean_pairwise_dist"] = float(raw_gstats["mean_pairwise_dist"])
            row["raw_separation_mean"] = float(raw_sep["separation_mean"])
            row["raw_fisher_ratio"] = float(raw_fisher)
        except Exception:
            log.error(f"Raw scale indicators failed for step {snap['step']} ({path}); "
                      f"filling raw_* columns with NaN.\n{traceback.format_exc()}")
            for k in RAW_SCALE_KEYS:
                row[k] = float("nan")
        for k in RAW_SCALE_KEYS:
            if k not in scalar_keys_seen:
                scalar_keys_seen.append(k)

        # --- (2) normalize, then compute every other metric as before. ---
        try:
            norm_latents = M.normalize_latents(raw_latents, mode=normalize)
        except Exception:
            log.error(f"normalize_latents failed for step {snap['step']} ({path}); "
                      f"falling back to raw (un-normalized) latents for this step.\n"
                      f"{traceback.format_exc()}")
            norm_latents = raw_latents

        try:
            all_metrics = M.compute_all_metrics(
                norm_latents, labels,
                alphas=alphas, n_scales=n_scales,
                per_class_topology=per_class_topology,
            )
        except Exception:
            log.error(f"compute_all_metrics failed entirely for step {snap['step']} "
                      f"({path}); filling all metric columns with NaN.\n{traceback.format_exc()}")
            all_metrics = {}

        for k, v in all_metrics.items():
            if k in ARRAY_VALUED_KEYS:
                continue
            try:
                row[k] = float(v)
            except Exception:
                log.error(f"Metric '{k}' at step {snap['step']} is not scalar-castable; NaN.")
                row[k] = float("nan")
            if k not in scalar_keys_seen:
                scalar_keys_seen.append(k)

        for k in ARRAY_VALUED_KEYS:
            if k in all_metrics:
                array_series[k].append(np.asarray(all_metrics[k]))
            else:
                array_series[k].append(None)  # placeholder, patched below

        rows.append(row)
        log.info(f"[{i+1}/{len(files)}] step={snap['step']} epoch={snap['epoch']:.3f} "
                  f"train_acc={snap['train_acc']:.3f} test_acc={snap['test_acc']:.3f} -- OK")

    if not rows:
        raise RuntimeError(f"No snapshots could be loaded/processed in {run_dir}")

    # Make sure every row has every scalar key seen anywhere (NaN-fill gaps
    # from steps where a metric failed or wasn't computed).
    for row in rows:
        for k in scalar_keys_seen:
            if k not in row:
                row[k] = float("nan")

    # Sort rows by step (glob is already lexically sorted on zero-padded
    # step in the filename, but be explicit/defensive).
    order = np.argsort([r["step"] for r in rows])
    rows = [rows[i] for i in order]
    for k in array_series:
        array_series[k] = [array_series[k][i] for i in order]

    all_cols = base_cols + scalar_keys_seen
    out_arrays: dict[str, np.ndarray] = {}
    for k in all_cols:
        out_arrays[k] = np.array([r[k] for r in rows], dtype=np.float64)

    # Patch array-valued series: fill any None (failed step) with NaNs of the
    # same shape as the first successfully-computed entry, then stack.
    for k, series in array_series.items():
        template = next((s for s in series if s is not None), None)
        if template is None:
            continue  # metric never computed successfully; omit from output
        filled = [s if s is not None else np.full_like(template, np.nan, dtype=np.float64)
                  for s in series]
        out_arrays[k] = np.stack(filled, axis=0)

    return {"rows": rows, "all_cols": all_cols, "out_arrays": out_arrays,
             "normalize_mode": normalize}


def write_outputs(run_dir: str, result: dict) -> None:
    out_dir = os.path.join(run_dir, "metrics")
    os.makedirs(out_dir, exist_ok=True)
    normalize_mode = result.get("normalize_mode", "none")

    csv_path = os.path.join(out_dir, "timeseries.csv")
    with open(csv_path, "w", newline="") as f:
        # Leading comment line (not a CSV row) recording which normalization
        # mode was applied to the latents before the standard metric columns
        # below were computed; `raw_*` columns are always un-normalized.
        # pandas.read_csv(..., comment="#") / numpy.loadtxt skip it cleanly.
        f.write(f"# normalize_mode={normalize_mode}\n")
        writer = csv.DictWriter(f, fieldnames=result["all_cols"])
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow(row)
    log.info(f"Wrote {csv_path} ({len(result['rows'])} rows, {len(result['all_cols'])} cols, "
              f"normalize_mode={normalize_mode})")

    npz_path = os.path.join(out_dir, "timeseries.npz")
    out_arrays = dict(result["out_arrays"])
    out_arrays["normalize_mode"] = np.array(normalize_mode)  # 0-d unicode array
    np.savez(npz_path, **out_arrays)
    log.info(f"Wrote {npz_path} ({len(out_arrays)} arrays)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True,
                         help="Run directory containing latents/step_*.npz "
                              "(e.g. /vol/bitbucket/lbastian/latent-dynamics/runs/<run_id>)")
    parser.add_argument("--alphas", type=float, nargs="+", default=list(M.DEFAULT_ALPHAS),
                         help="Alpha exponents for the alpha-weighted lifetime sum (default: 0.5 1 2)")
    parser.add_argument("--n_scales", type=int, default=M.DEFAULT_N_SCALES,
                         help="Number of scales to sample for the magnitude function (default: 12)")
    parser.add_argument("--per_class_topology", action="store_true",
                         help="Also compute H0/H1 persistence metrics per class (slower)")
    parser.add_argument("--normalize", choices=list(M.NORMALIZE_MODES), default="mean_pairwise",
                         help="Per-snapshot normalization applied to latents before computing "
                              "class-geometry/topology/PMag metrics (default: mean_pairwise, an "
                              "isotropic rescale that removes scale inflation while preserving "
                              "shape). 'raw_*' columns are always computed from the un-normalized "
                              "latents regardless of this choice. See module docstring.")
    args = parser.parse_args()

    run_dir = os.path.abspath(args.run_dir)
    if not os.path.isdir(run_dir):
        log.error(f"run_dir does not exist: {run_dir}")
        sys.exit(1)

    result = analyze_run(run_dir, alphas=tuple(args.alphas), n_scales=args.n_scales,
                          per_class_topology=args.per_class_topology,
                          normalize=args.normalize)
    write_outputs(run_dir, result)
    log.info("Done.")


if __name__ == "__main__":
    main()
