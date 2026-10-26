"""Shared run-directory I/O helpers (snapshots, configs, timeseries, run iteration)."""
from __future__ import annotations

import glob
import hashlib
import json
import os

import numpy as np

DEFAULT_ARTIFACTS_ROOT = os.environ.get("LATDYN_ART", "/vol/bitbucket/lbastian/latent-dynamics")


def find_snapshot_files(run_dir: str, sub: str = "latents") -> list[str]:
    """Sorted <run_dir>/<sub>/step_*.npz; raises FileNotFoundError if none."""
    pattern = os.path.join(run_dir, sub, "step_*.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No snapshot files found matching {pattern}")
    return files


def load_snapshot(path: str, float64: bool = False, extras: tuple = ("train_loss", "train_acc", "test_acc")) -> dict:
    """Load a latent snapshot npz. Defaults reproduce analyze.py (native dtype; keys latents, labels, step, epoch +
    train_loss/train_acc/test_acc). trajectory.py: float64=True, extras=("test_acc",)."""
    with np.load(path) as data:
        lat = data["latents"]
        out = {
            "latents": lat.astype(np.float64, copy=False) if float64 else lat,
            "labels": data["labels"],
            "step": int(data["step"]),
            "epoch": float(data["epoch"]),
        }
        for k in extras:
            out[k] = float(data[k])
        return out


def read_config(run_dir: str, strict: bool = False, warn=None) -> dict:
    """Load <run_dir>/config.json. strict=False (compare.py behaviour): {} if missing or unparseable
    (calls warn(msg) on parse failure if given). strict=True: raise."""
    cfg_path = os.path.join(run_dir, "config.json")
    if strict:
        with open(cfg_path) as f:
            return json.load(f)
    if not os.path.exists(cfg_path):
        return {}
    try:
        with open(cfg_path) as f:
            return json.load(f)
    except Exception:
        if warn is not None:
            warn(f"Failed to parse {cfg_path}; treating config as empty.")
        return {}


load_run_config = read_config


def load_timeseries(run_dir: str, missing: str = "raise", warn=None):
    """Load <run_dir>/metrics/timeseries.npz as a dict. missing='raise' (plot.py): FileNotFoundError;
    missing='none' (compare.py): warn + return None."""
    npz_path = os.path.join(run_dir, "metrics", "timeseries.npz")
    if not os.path.exists(npz_path):
        if missing == "raise":
            raise FileNotFoundError(f"{npz_path} not found -- run analyze.py on this run_dir first.")
        if warn is not None:
            warn(f"Missing {npz_path}; skipping this run.")
        return None
    with np.load(npz_path) as data:
        return {k: data[k] for k in data.files}


def iter_runs(root: str | None = None, pattern: str = "*"):
    """Yield (run_id, run_dir) for <root>/runs/<pattern> directories, sorted."""
    root = root or DEFAULT_ARTIFACTS_ROOT
    for d in sorted(glob.glob(os.path.join(root, "runs", pattern))):
        if os.path.isdir(d):
            yield os.path.basename(d), d


def snap_hash(run_dir, code_version: str = "", extra=None) -> str:
    """sha1 over (code_version, json(extra), [(sub, basename, size)] of latents/ and latents_train/ snapshots).
    phase_diagram.py: code_version=CODE_VERSION, extra=SL."""
    items = []
    for sub in ("latents", "latents_train"):
        for f in sorted(glob.glob(f"{run_dir}/{sub}/step_*.npz")):
            try:
                items.append((sub, os.path.basename(f), os.stat(f).st_size))
            except OSError:
                pass
    return hashlib.sha1((code_version + json.dumps(extra, sort_keys=True) + json.dumps(items)).encode()).hexdigest()
