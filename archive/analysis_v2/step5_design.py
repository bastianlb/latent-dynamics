#!/usr/bin/env python
"""Amendment 4 A4.2b full-pipeline refit bootstrap and A4.2c block-permutation test (ridge, B1 vs B1+G, primary targets).
usage: --dataset mnist|fashion --part refit|perm [--nrep 400] [--workers 4]"""
import step5_lib as M
import argparse
from multiprocessing import Pool
import numpy as np
L, S = M.L, M.S
RUNS = None
NREP = 400
PART = "refit"


def refit_cell(a):
    t, cut = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    B1 = np.hstack([B0, DES]); B1G = np.hstack([B1, G])
    u = np.unique(g); rows_of = {c: np.where(g == c)[0] for c in u}
    rng = np.random.default_rng(12345 + cut + (1 if t.startswith("gap") else 0))
    # observed
    pb = S.loco(B1, y, g, "ridge"); pc = S.loco(B1G, y, g, "ridge")
    obs = S.r2(y, pc) - S.r2(y, pb)
    out = []
    for rep in range(NREP):
        samp = rng.integers(0, len(u), len(u))
        cs, mult = np.unique(u[samp], return_counts=True)
        idx_all = np.concatenate([np.repeat(rows_of[c], m) for c, m in zip(cs, mult)])
        gg = g[idx_all]
        sb = sc = Sy = Syy = N = 0.0
        for c, m in zip(cs, mult):
            tr = idx_all[gg != c]; te = rows_of[c]
            if len(np.unique(g[tr])) < 5: continue
            pbc = S.ridge_fit_predict(B1[tr], y[tr], g[tr], B1[te]); pcc = S.ridge_fit_predict(B1G[tr], y[tr], g[tr], B1G[te])
            yt = y[te]
            sb += m * ((yt - pbc) ** 2).sum(); sc += m * ((yt - pcc) ** 2).sum()
            Sy += m * yt.sum(); Syy += m * (yt ** 2).sum(); N += m * len(yt)
        sst = Syy - Sy ** 2 / N
        out.append((sb - sc) / sst)
    out = np.array(out)
    q = np.percentile(out, [2.5, 97.5, 5, 95])
    rc = np.percentile(out - out.mean() + obs, [2.5, 97.5, 5, 95])   # re-centred on the observed dR2 (removes the smaller-training-set bias of the refit)
    np.save(f"{M.OUT3}/refit_samples_{t}_{cut}.npy", out)
    return dict(lo95_recentred=rc[0], hi95_recentred=rc[1], lo90_recentred=rc[2], hi90_recentred=rc[3], mstar_recentred=max(abs(rc[2]), abs(rc[3])),
                class_recentred=L.classify(rc[2], rc[3], L.margin(t))["label"], dataset=M.DATASET, target=t, cutoff=cut, delta_obs=obs, n_rep=len(out), boot_mean=out.mean(), boot_sd=out.std(),
                lo95_refit=q[0], hi95_refit=q[1], lo90_refit=q[2], hi90_refit=q[3], mstar_refit=max(abs(q[2]), abs(q[3])),
                class_refit=L.classify(q[2], q[3], L.margin(t))["label"])


def perm_G_block(G, g, blk, rng):
    Gp = G.copy()
    for b in np.unique(blk):
        ub = np.unique(g[blk == b]); pi = rng.permutation(ub)
        for c, c2 in zip(ub, pi):
            a_, b_ = np.where(g == c)[0], np.where(g == c2)[0]
            for k, ia in enumerate(a_):
                Gp[ia] = G[b_[k % len(b_)]]
    return Gp


def perm_cell(a):
    t, cut = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    B1 = np.hstack([B0, DES])
    pb = S.loco(B1, y, g, "ridge"); r2b = S.r2(y, pb)
    obs = S.r2(y, S.loco(np.hstack([B1, G]), y, g, "ridge")) - r2b
    rng = np.random.default_rng(7 + cut)
    res = {}
    for kind in ("block", "free"):
        d = []
        for _ in range(NREP):
            Gp = perm_G_block(G, g, blk, rng) if kind == "block" else S.permute_G(G, g, seeds, rng)
            d.append(S.r2(y, S.loco(np.hstack([B1, Gp]), y, g, "ridge")) - r2b)
        d = np.array(d); res[kind] = d
    return dict(dataset=M.DATASET, target=t, cutoff=cut, delta_obs=obs, n_perm=NREP,
                p_block=(np.sum(res["block"] >= obs) + 1) / (NREP + 1), p_free=(np.sum(res["free"] >= obs) + 1) / (NREP + 1),
                null_mean_block=res["block"].mean(), null_sd_block=res["block"].std(), null_mean_free=res["free"].mean(), null_sd_free=res["free"].std())


def main():
    global RUNS, NREP
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--part", default="refit")
    ap.add_argument("--nrep", type=int, default=400); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); NREP = a.nrep
    RUNS = L.get_runs()
    jobs = [(t, c) for t in M.PRIMARY for c in M.CUTS]
    with Pool(a.workers) as p:
        res = p.map(refit_cell if a.part == "refit" else perm_cell, jobs)
    M.write_csv(f"{M.OUT3}/design_{a.part}.csv", res)
    for r in res: print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})


if __name__ == "__main__":
    main()
