#!/usr/bin/env python
"""A5.A placebo features: random 1-D projection summary statistics of the probe latents at the cutoff snapshot (test + train probes), NDRAW draws.
Saves OUT4/placebo_feats.npz : P[cut] (n_runs, NDRAW, 18) ordered te(6), tr(6), dl(6); plus Pnorm[cut] (n_runs,2) = [log train-probe raw norm, delta vs test]."""
import step6_lib as Z
import argparse
from multiprocessing import Pool
import numpy as np
L, C, D = Z.L, Z.C, Z.D
RUNS = None
WS = Z.make_dirs()


def one(args):
    i, cut = args
    r = RUNS[i]; st = r["step"]; k = D.idx_le_mask(st, r["topo"], cut); rid = r["info"]["run_id"]
    feats = {}
    for ps in ("test", "train"):
        x = np.load(Z.snap_files(rid, ps, st[k]))["latents"].astype(np.float64)
        feats[ps] = np.stack([Z.proj_stats(x, W) for W in WS])          # NDRAW x 6
        feats[ps + "_norm"] = np.log(np.linalg.norm(x, axis=1).mean() + 1e-6)
    P = np.hstack([feats["test"], feats["train"], feats["train"] - feats["test"]])
    return i, cut, P, np.array([feats["train_norm"], feats["train_norm"] - feats["test_norm"]])


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RUNS = L.get_runs(); n = len(RUNS)
    jobs = [(i, c) for c in L.CUTS for i in range(n)]
    P = {c: np.zeros((n, Z.NDRAW, 18)) for c in L.CUTS}; Pn = {c: np.zeros((n, 2)) for c in L.CUTS}
    with Pool(a.workers) as p:
        for i, c, pp, pn in p.imap_unordered(one, jobs, chunksize=8):
            P[c][i] = pp; Pn[c][i] = pn
    np.savez(f"{Z.OUT4}/placebo_feats.npz", **{f"P_{c}": P[c] for c in L.CUTS}, **{f"Pnorm_{c}": Pn[c] for c in L.CUTS})
    print("saved", n, "runs", Z.DATASET)


if __name__ == "__main__":
    main()
