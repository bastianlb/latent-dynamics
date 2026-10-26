#!/usr/bin/env python
"""Per-snapshot extra quantities for the reanalysis, cached per run.

For every run (default: all) and every latents/step_*.npz, computes on the same
mean-pairwise-normalised latents analyze.py uses:

  offline_probe_acc   5-fold stratified CV accuracy of a standardised multinomial
                      LogisticRegression on the N=1000 probe-set latents ("offline
                      linear separability"; see metrics.offline_linear_separability)
  h1_bars / h1_off    full H1 persistence diagram (birth, death), concatenated, with
                      per-snapshot offsets (so top-k / thresholded lifetime sums can be
                      recomputed offline)
  mag_t / mag         Mag(t) on a FIXED log-spaced grid of t (normalised-distance units;
                      mean pairwise distance == 1), [n_snap, n_t]. Any fixed-window PMag can
                      be integrated from this without re-solving (see pmag_fixed.py).
  d_q                 pairwise-distance quantiles (1,5,25,50,75,95,99 %) of the normalised
                      cloud (used to build global scale windows)

Output: <OUT>/cache/<run_id>.npz   (resumable; existing files are skipped unless --force)

Usage: python compute_extra.py [--datasets mnist ...] [--runs id ...] [--workers 4] [--force]
"""
from __future__ import annotations

import common as C  # noqa: F401  (must be first: caps threads)

import argparse
import os
import time
import warnings
from multiprocessing import Pool

import numpy as np

import metrics as M

MAG_T = np.geomspace(0.1, 100.0, 32)
QS = (1, 5, 25, 50, 75, 95, 99)


def process_run(rid: str, force: bool = False) -> str:
    out = os.path.join(C.CACHE, rid + ".npz")
    if os.path.exists(out) and not force:
        return f"skip {rid}"
    warnings.filterwarnings("ignore")
    from scipy.spatial.distance import pdist
    t0 = time.time()
    files = C.snapshot_files(rid)
    steps, probe, bars, off, mags, dq = [], [], [], [0], [], []
    for p in files:
        s = C.load_snapshot(p)
        z = M.normalize_latents(s["latents"], "mean_pairwise")
        steps.append(int(s["step"]))
        probe.append(M.offline_linear_separability(s["latents"], s["labels"]))
        dg = M.persistence_diagrams(z, maxdim=1)["dgms"][1]
        bars.append(np.asarray(dg, dtype=np.float32).reshape(-1, 2)); off.append(off[-1] + len(bars[-1]))
        mags.append(M.magnitude_function(z, scales=MAG_T)["magnitude"])
        d = pdist(z); d = d[d > 1e-12]
        dq.append(np.percentile(d, QS))
    np.savez(out + ".tmp.npz", step=np.array(steps), offline_probe_acc=np.array(probe),
             h1_bars=np.concatenate(bars), h1_off=np.array(off), mag_t=MAG_T,
             mag=np.array(mags), d_q=np.array(dq), d_q_levels=np.array(QS))
    os.replace(out + ".tmp.npz", out)
    return f"done {rid} {len(files)} snaps {time.time()-t0:.0f}s"


def _star(a):
    return process_run(*a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=C.DATASETS)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    os.makedirs(C.CACHE, exist_ok=True)
    rids = a.runs or [r["run_id"] for r in C.list_runs(a.datasets)]
    # biggest/slowest first is not needed; just map
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(_star, [(r, a.force) for r in rids]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
