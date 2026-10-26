"""Shared helpers for the reanalysis scripts (thread caps, run discovery, loaders).

Import this FIRST in every reanalysis script: it caps BLAS/OpenMP threads before numpy loads.
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import csv
import glob
import re
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

ART = os.environ.get("LATDYN_ART", "/vol/bitbucket/lbastian/latent-dynamics")
RUNS = os.path.join(ART, "runs")
OUT = os.environ.get("REANALYSIS_OUT", os.path.join(ART, "reanalysis_v2"))
CACHE = os.path.join(OUT, "cache")
DATASETS = ["mnist", "fashion_mnist", "cifar10"]

_RID = re.compile(r"^(?P<dataset>mnist|fashion_mnist|cifar10)_(?P<objective>probe_recon|probe|autoencoder)_"
                  r"(?P<norm>none|l2|bn)_d(?P<D>\d+)_s(?P<seed>\d+)$")


def touch_nfs():
    """bitbucket is an idle-unmounting NFS automount; list it to (re)mount."""
    try:
        os.listdir(RUNS)
    except OSError:
        pass


def parse_run_id(rid: str) -> dict:
    m = _RID.match(rid)
    if not m:
        raise ValueError(rid)
    d = m.groupdict()
    d["D"] = int(d["D"]); d["seed"] = int(d["seed"])
    d["run_id"] = rid
    d["config"] = f"{d['objective']}/{d['norm']}/d{d['D']}"
    return d


def list_runs(datasets=DATASETS) -> list[dict]:
    touch_nfs()
    out = []
    for rid in sorted(os.listdir(RUNS)):
        try:
            r = parse_run_id(rid)
        except ValueError:
            continue
        if r["dataset"] in datasets and os.path.isdir(os.path.join(RUNS, rid, "latents")):
            out.append(r)
    return out


def snapshot_files(rid: str) -> list[str]:
    return sorted(glob.glob(os.path.join(RUNS, rid, "latents", "step_*.npz")))


def load_snapshot(path: str) -> dict:
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


def load_timeseries(rid: str) -> dict:
    """Existing analyze.py output (metrics/timeseries.csv) as dict[str -> float array]."""
    p = os.path.join(RUNS, rid, "metrics", "timeseries.csv")
    with open(p) as f:
        lines = [l for l in f if not l.startswith("#")]
    rd = csv.DictReader(lines)
    cols = {k: [] for k in rd.fieldnames}
    for row in rd:
        for k, v in row.items():
            cols[k].append(float(v) if v not in ("", None) else np.nan)
    return {k: np.asarray(v) for k, v in cols.items()}


def load_cache(rid: str) -> dict | None:
    p = os.path.join(CACHE, rid + ".npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def fmt(x, nd=3):
    return "nan" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{nd}g}"
