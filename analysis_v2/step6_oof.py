#!/usr/bin/env python
"""A5 core: out-of-fold predictions for B0/B1 x {G full, compact4, compact8, placebo blocks} on LOCO and the extrapolation splits.
usage: --dataset mnist|fashion [--workers 4]. Output OUT4/oof6.pkl"""
import step6_lib as Z
import step4_splits as SP
import argparse, pickle
from multiprocessing import Pool
import numpy as np
L, S, M = Z.L, Z.S, Z.M
RUNS = None; PF = None
SIZES = {"18": list(range(18)), "8": [0, 1, 2, 3, 12, 13, 14, 15], "4": [0, 1, 2, 3]}


def build_sets(model, cut, B0, DES, G, names_g):
    nd = Z.NDRAW if model == "ridge" else Z.NDRAW_GBM
    B1 = np.hstack([B0, DES]); n = len(B0)
    P = PF[f"P_{cut}"]; Pn = PF[f"Pnorm_{cut}"]
    c4 = M.gvariant_cols(names_g, "compact4"); c8 = M.gvariant_cols(names_g, "compact8")
    full = {"B1": B1, "B1+G": np.hstack([B1, G]), "B1+c4": np.hstack([B1, G[:, c4]]), "B1+c8": np.hstack([B1, G[:, c8]]),
            "B1+Pnorm": np.hstack([B1, Pn])}
    loco_only = {"B0": B0, "B0+G": np.hstack([B0, G]), "B0+c4": np.hstack([B0, G[:, c4]])}
    for d in range(nd):
        for sz, cols in SIZES.items():
            Pr = P[:, d, cols]; Nz = Z.noise_block(n, len(cols), d)
            full[f"B1+Prp{sz}_d{d}"] = np.hstack([B1, Pr]); full[f"B1+Pnz{sz}_d{d}"] = np.hstack([B1, Nz])
        for sz in ("18", "4"):
            loco_only[f"B0+Prp{sz}_d{d}"] = np.hstack([B0, P[:, d, SIZES[sz]]])
            loco_only[f"B0+Pnz{sz}_d{d}"] = np.hstack([B0, Z.noise_block(n, len(SIZES[sz]), d)])
    return full, loco_only


def job(a):
    t, cut, model = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    full, loco_only = build_sets(model, cut, B0, DES, G, names[2])
    out = {"y": y, "g": g}
    sp = {"loco": [g == c for c in np.unique(g)]}
    sp.update({k: [m for _, m in v] for k, v in SP.split_defs(RUNS).items()})
    for split, masks in sp.items():
        sets = dict(full); 
        if split == "loco": sets.update(loco_only)
        out[split] = {k: L.fold_predict(X, y, g, masks, model) for k, X in sets.items()}
    return (t, cut, model), out


def main():
    global RUNS, PF
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RUNS = L.get_runs(); PF = dict(np.load(f"{Z.OUT4}/placebo_feats.npz"))
    jobs = [(t, c, m) for m in ("gbm", "ridge") for c in L.CUTS for t in M.PRIMARY]
    with Pool(a.workers) as p:
        res = dict(p.imap_unordered(job, jobs))
    pickle.dump(res, open(f"{Z.OUT4}/oof6.pkl", "wb")); print("saved", len(res))


if __name__ == "__main__":
    main()
