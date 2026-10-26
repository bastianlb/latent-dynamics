#!/usr/bin/env python
"""
compare.py
===========

CLI that overlays metric trajectories across MANY runs -- for contrasting
a sweep (e.g. over latent_dim / objective / latent_norm) and/or an
objective ablation. Reads each run's `metrics/timeseries.npz` (written by
analyze.py, per SPEC.md's data contract).

Run selection
-------------
Either:
    --manifest <sweep manifest.json>
or:
    --run_dirs <dir> <dir> ...

No sweep manifest exists yet in this repo, so its exact schema is
unestablished. This loader is therefore deliberately permissive and
accepts any of:
    - a JSON list of entries: [{"run_id": ..., "run_dir": ..., "config": {...}}, ...]
    - {"runs": [ ...same entries as above... ]}
    - {"run_ids": ["id1", "id2", ...]}  (config then comes only from each
      run's own on-disk config.json)
    - {"<run_id>": {<config dict>}, ...}   (flat mapping form)
An entry may omit "run_dir"; it is then inferred as
`<artifacts_root>/runs/<run_id>`, where `<artifacts_root>` is taken to be
the grandparent of the manifest file (manifest is assumed to live at
`<artifacts_root>/sweeps/<name>/manifest.json`, per SPEC.md's layout),
unless the manifest JSON itself sets a top-level "artifacts_root".

Each run's on-disk `config.json` is ALWAYS loaded as the base config;
any config fields given inline in the manifest entry override it. This
is what supplies `latent_dim` / `objective` / `latent_norm` / `seed` for
grouping and for the summary CSV -- config.json is the resolved,
authoritative source per SPEC.md, the manifest is treated as a thin
index/override on top of it.

Grouping & seed-averaging
--------------------------
Runs are bucketed by `--group_by` (one of latent_dim/objective/
latent_norm), read from each run's merged config (falling back to
parsing `_d<N>` out of the run_id for latent_dim, since that's baked
into the run_id format per SPEC.md). Within a group, runs whose config
is IDENTICAL except for `seed` (and non-semantic fields: run_id,
artifacts_root, device, tag) are treated as replicate seeds of the same
experiment: their metric curves are averaged into a mean +/- std band
instead of being drawn as separate lines. Runs with no same-config
replicate are drawn as a single line. Color encodes the `--group_by`
value; replicate seeds within a group share that color.

Step alignment
---------------
Individual (non-replicated) runs are plotted against their own `step`
array as-is -- differing #snapshots across runs is fine for plain
overlay lines, since each line supplies its own x-values. For
replicate-seed AVERAGING specifically, arrays are truncated to the
shortest run in the replicate set (aligned by step INDEX, not step
value): a sweep's runs are assumed to share the same logging schedule
(`log_every`), so index-aligned truncation coincides with step-aligned
truncation up to where the shortest run stops. This is the one place a
silent step mismatch could hide, hence called out explicitly here.

Outputs (written to --out_dir):
    compare_<group_by>.png   -- one panel per key metric (test_acc,
                                 fisher_ratio, separation_mean, h1_entropy,
                                 h1_lifetime_sum_alpha1.0, pmag), one
                                 line/band per run or replicate-seed group.
    summary_final.csv        -- one row per individual run: run_id,
                                 group value, scalar config fields, and
                                 the FINAL-step value of each key metric
                                 (NaN if a run is missing that metric).

Usage
-----
    python compare.py --manifest /vol/bitbucket/lbastian/latent-dynamics/sweeps/<name>/manifest.json \
        --group_by latent_dim [--out_dir DIR]

    python compare.py --run_dirs /vol/bitbucket/lbastian/latent-dynamics/runs/run_a \
        /vol/bitbucket/lbastian/latent-dynamics/runs/run_b --group_by objective
"""

from __future__ import annotations

import os

# Cap BLAS/OpenMP thread pools BEFORE numpy is imported -- see trajectory.py
# / analyze.py / plot.py for the same pattern and rationale (shared 128-core
# box; uncapped numpy/BLAS oversubscribes badly).
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "8")

import argparse
import csv
import json
import logging
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="[compare] %(message)s")
log = logging.getLogger(__name__)

DPI = 150
KEY_METRICS = [
    "test_acc", "fisher_ratio", "separation_mean",
    "h1_entropy", "h1_lifetime_sum_alpha1.0", "pmag",
]
GROUP_FIELDS = ("latent_dim", "objective", "latent_norm")
# Config fields that must NOT define a "replicate" bucket (identifiers /
# non-semantic fields) -- two runs differing ONLY in these are replicate seeds.
REPLICATE_EXCLUDE_KEYS = {"seed", "run_id", "artifacts_root", "device", "tag"}

CMAP = plt.get_cmap("tab10")


# ---------------------------------------------------------------------------
# Run discovery / config loading
# ---------------------------------------------------------------------------

def _load_manifest_entries(manifest_path: str) -> list[dict]:
    with open(manifest_path) as f:
        data = json.load(f)

    artifacts_root = None
    if isinstance(data, dict):
        artifacts_root = data.get("artifacts_root")
    if artifacts_root is None:
        # manifest lives at <artifacts_root>/sweeps/<name>/manifest.json
        artifacts_root = os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.abspath(manifest_path))))

    if isinstance(data, list):
        entries = data
    elif isinstance(data, dict) and isinstance(data.get("runs"), list):
        entries = data["runs"]
    elif isinstance(data, dict) and isinstance(data.get("run_ids"), list):
        entries = [{"run_id": r} for r in data["run_ids"]]
    elif isinstance(data, dict):
        entries = [{"run_id": k, "config": v} for k, v in data.items()
                   if isinstance(v, dict)]
    else:
        raise ValueError(f"Unrecognized manifest shape in {manifest_path}")

    out = []
    for e in entries:
        run_id = e.get("run_id") or e.get("id")
        if run_id is None:
            log.warning(f"Skipping manifest entry with no run_id: {e}")
            continue
        run_dir = e.get("run_dir") or os.path.join(artifacts_root, "runs", run_id)
        out.append({"run_id": run_id, "run_dir": run_dir,
                    "manifest_config": e.get("config") or {}})
    return out


def _load_run_config(run_dir: str) -> dict:
    cfg_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        log.warning(f"Failed to parse {cfg_path}; treating config as empty.")
        return {}


def _load_timeseries(run_dir: str) -> dict | None:
    npz_path = os.path.join(run_dir, "metrics", "timeseries.npz")
    if not os.path.exists(npz_path):
        log.warning(f"Missing {npz_path}; skipping this run.")
        return None
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def gather_runs(args) -> list[dict]:
    if args.manifest:
        entries = _load_manifest_entries(args.manifest)
    else:
        entries = [{"run_id": os.path.basename(os.path.abspath(d).rstrip(os.sep)),
                    "run_dir": os.path.abspath(d), "manifest_config": {}}
                   for d in args.run_dirs]

    runs = []
    for e in entries:
        run_dir = e["run_dir"]
        if not os.path.isdir(run_dir):
            log.warning(f"run_dir does not exist, skipping: {run_dir}")
            continue
        config = dict(_load_run_config(run_dir))
        config.update(e.get("manifest_config") or {})
        ts = _load_timeseries(run_dir)
        if ts is None:
            continue
        if "step" not in ts:
            log.warning(f"timeseries for {e['run_id']} has no 'step' column; skipping.")
            continue
        runs.append({"run_id": e["run_id"], "run_dir": run_dir,
                     "config": config, "timeseries": ts})
    if not runs:
        raise RuntimeError("No usable runs found (check --manifest/--run_dirs paths).")
    return runs


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------

def _group_value(run: dict, group_by: str) -> str:
    val = run["config"].get(group_by)
    if val is not None:
        return str(val)
    if group_by == "latent_dim":
        m = re.search(r"_d(\d+)", run["run_id"])
        if m:
            return m.group(1)
    log.warning(f"Run {run['run_id']} has no '{group_by}' in config; grouping as 'unknown'.")
    return "unknown"


def _replicate_key(run: dict) -> tuple:
    return tuple(sorted(
        (k, json.dumps(v, sort_keys=True)) for k, v in run["config"].items()
        if k not in REPLICATE_EXCLUDE_KEYS
    ))


def assign_groups(runs: list[dict], group_by: str) -> None:
    """Mutates each run dict in place: adds 'group_val' and 'replicate_key'."""
    for r in runs:
        r["group_val"] = _group_value(r, group_by)
        r["replicate_key"] = _replicate_key(r)


def build_series_groups(runs: list[dict]) -> dict:
    """Bucket runs by (group_val, replicate_key) -> list of runs (seed replicates)."""
    buckets: dict[tuple, list[dict]] = {}
    for r in runs:
        key = (r["group_val"], r["replicate_key"])
        buckets.setdefault(key, []).append(r)
    return buckets


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _metric_series(run: dict, metric: str):
    ts = run["timeseries"]
    if metric not in ts or len(ts[metric]) == 0:
        return None
    return ts["step"], ts[metric]


def plot_compare(runs: list[dict], group_by: str, out_path: str) -> None:
    buckets = build_series_groups(runs)
    group_vals = sorted({r["group_val"] for r in runs}, key=lambda s: (len(s), s))
    color_of = {g: CMAP(i % 10) for i, g in enumerate(group_vals)}

    ncols = 3
    nrows = int(np.ceil(len(KEY_METRICS) / ncols))
    panel_w, panel_h = 5.2, 4.0
    # Fixed-inch top/bottom margins (not fractional) so the suptitle+legend
    # band and bottom whitespace stay a constant size regardless of nrows --
    # see the equivalent comment in trajectory.py's make_grid for why a
    # fractional margin (the tight_layout/bbox_to_anchor>1.0 default) blows
    # up into a large blank gap on taller figures.
    title_gap_in, legend_gap_in, top_margin_in, bottom_margin_in = 0.12, 0.55, 1.05, 0.55
    fig_h = panel_h * nrows + top_margin_in + bottom_margin_in
    fig_w = panel_w * ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
    axes_flat = axes.ravel()

    legend_handles, legend_seen = [], set()

    for ax, metric in zip(axes_flat, KEY_METRICS):
        plotted_any = False
        for (group_val, _rkey), members in buckets.items():
            color = color_of[group_val]
            series = [s for s in (_metric_series(r, metric) for r in members) if s is not None]
            if not series:
                continue
            plotted_any = True
            if len(series) == 1:
                step, vals = series[0]
                line, = ax.plot(step, vals, color=color, alpha=0.9, linewidth=1.4)
            else:
                min_len = min(len(s[0]) for s in series)
                steps = series[0][0][:min_len]
                stacked = np.stack([s[1][:min_len] for s in series], axis=0)
                mean = np.nanmean(stacked, axis=0)
                std = np.nanstd(stacked, axis=0)
                line, = ax.plot(steps, mean, color=color, alpha=0.9, linewidth=1.6)
                ax.fill_between(steps, mean - std, mean + std, color=color, alpha=0.2)
            if group_val not in legend_seen:
                line.set_label(f"{group_by}={group_val}")
                legend_handles.append(line)
                legend_seen.add(group_val)
        ax.set_xlabel("step")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        ax.grid(alpha=0.3)
        if not plotted_any:
            ax.text(0.5, 0.5, "metric not available\nin any run", ha="center", va="center",
                     transform=ax.transAxes, fontsize=8, color="gray")

    for ax in axes_flat[len(KEY_METRICS):]:
        ax.axis("off")

    top_frac = 1.0 - top_margin_in / fig_h
    bottom_frac = bottom_margin_in / fig_h
    fig.subplots_adjust(top=top_frac, bottom=bottom_frac, hspace=0.4, wspace=0.3)

    fig.legend(handles=legend_handles, loc="upper center",
               ncol=min(len(legend_handles), 6) or 1,
               bbox_to_anchor=(0.5, 1.0 - legend_gap_in / fig_h), fontsize=8)
    fig.suptitle(f"Metric comparison grouped by {group_by} ({len(runs)} runs)",
                 fontsize=11, y=1.0 - title_gap_in / fig_h)
    fig.savefig(out_path, dpi=DPI)
    plt.close(fig)
    log.info(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Summary CSV
# ---------------------------------------------------------------------------

def write_summary_csv(runs: list[dict], group_by: str, out_path: str) -> None:
    config_keys = set()
    for r in runs:
        for k, v in r["config"].items():
            if v is None or isinstance(v, (str, int, float, bool)):
                config_keys.add(k)
    config_keys = sorted(config_keys)

    rows = []
    for r in runs:
        row = {"run_id": r["run_id"], "run_dir": r["run_dir"],
               "group_by_field": group_by, "group_val": r["group_val"]}
        for k in config_keys:
            row[k] = r["config"].get(k)
        ts = r["timeseries"]
        for metric in KEY_METRICS:
            row[f"final_{metric}"] = (float(ts[metric][-1])
                                       if metric in ts and len(ts[metric]) > 0
                                       else float("nan"))
        row["n_steps"] = int(len(ts["step"])) if "step" in ts else 0
        rows.append(row)

    fieldnames = (["run_id", "run_dir", "group_by_field", "group_val"] + config_keys
                  + [f"final_{m}" for m in KEY_METRICS] + ["n_steps"])
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"Wrote {out_path} ({len(rows)} rows)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_out_dir(args) -> str:
    if args.manifest:
        return os.path.join(os.path.dirname(os.path.abspath(args.manifest)), "compare_figures")
    common_parent = os.path.dirname(os.path.abspath(args.run_dirs[0].rstrip(os.sep)))
    return os.path.join(common_parent, "_compare")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--manifest", help="Path to a sweep manifest.json")
    src.add_argument("--run_dirs", nargs="+", help="One or more run directories")
    parser.add_argument("--group_by", choices=GROUP_FIELDS, default="latent_dim")
    parser.add_argument("--out_dir", default=None,
                         help="Output dir (default: 'compare_figures' alongside the "
                              "manifest, or '_compare' next to the given run_dirs)")
    args = parser.parse_args()

    out_dir = args.out_dir or _default_out_dir(args)
    os.makedirs(out_dir, exist_ok=True)

    runs = gather_runs(args)
    assign_groups(runs, args.group_by)
    log.info(f"Loaded {len(runs)} runs; grouping by '{args.group_by}': "
             f"{sorted({r['group_val'] for r in runs})}")

    plot_compare(runs, args.group_by, os.path.join(out_dir, f"compare_{args.group_by}.png"))
    write_summary_csv(runs, args.group_by, os.path.join(out_dir, "summary_final.csv"))
    log.info(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
