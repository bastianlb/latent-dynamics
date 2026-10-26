#!/usr/bin/env python
"""Amendment 4 A4.4: learner x geometry-variant sensitivity for the main LOCO test and extrapolation splits, with negative-R2 diagnostics.
usage: --dataset mnist|fashion [--workers 4] [--learners ridge,krr,mlp]"""
import step5_lib as M
import step4_splits as SP
import argparse
from multiprocessing import Pool
import numpy as np
L, S = M.L, M.S
RUNS = None
VARIANTS = ["full", "compact4", "compact8", "pca2", "pca4"]


def predict_split(B1, G, names_g, y, g, masks, learner, variant):
    f = M.learner_fn(learner)
    p = np.full(len(y), np.nan); tm = np.full(len(y), np.nan)
    for te in masks:
        tr = ~te
        if tr.sum() < 5 or te.sum() == 0: continue
        Xa, Xb = M.design_matrices(B1, G, names_g, variant, tr, te)
        p[te] = f(Xa, y[tr], g[tr], Xb); tm[te] = y[tr].mean()
    return p, tm


def evaluate(y, tm, grand, pb, pv, g, nb, seed=0):
    st = M.CfgStats(y, {"b": pb, "v": pv}, g, tm=tm, grand=grand)
    W1 = np.ones((1, st.nc)); rng = np.random.default_rng(seed); W = st.draw(nb, rng)
    d = st.d_r2("b", "v", W1)[0]; bs = st.d_r2("b", "v", W)
    dtm = st.d_r2_alt("b", "v", W1, "tm")[0]; btm = st.d_r2_alt("b", "v", W, "tm")
    dgm = st.d_r2_alt("b", "v", W1, "gm")[0]; bgm = st.d_r2_alt("b", "v", W, "gm")
    rr = st.rmse_ratio("b", "v", W1)[0]; brr = st.rmse_ratio("b", "v", W)
    sst = st.sst(np.ones(st.nc)); sst_tm = st.sst_tm.sum(); sst_gm = st.sst_gm.sum()
    n = len(y)
    q = lambda b: np.percentile(b, [2.5, 97.5])
    return dict(heldout_var=sst / n, r2_base=1 - st.sse["b"].sum() / sst, r2_var=1 - st.sse["v"].sum() / sst,
                r2_base_tm=1 - st.sse["b"].sum() / sst_tm, r2_var_tm=1 - st.sse["v"].sum() / sst_tm,
                r2_base_gm=1 - st.sse["b"].sum() / sst_gm, r2_var_gm=1 - st.sse["v"].sum() / sst_gm,
                delta=d, lo95=q(bs)[0], hi95=q(bs)[1], delta_tm=dtm, lo95_tm=q(btm)[0], hi95_tm=q(btm)[1],
                delta_gm=dgm, lo95_gm=q(bgm)[0], hi95_gm=q(bgm)[1],
                rmse_base_pts=100 * np.sqrt(st.sse["b"].sum() / n), rmse_var_pts=100 * np.sqrt(st.sse["v"].sum() / n),
                rmse_ratio=rr, rmse_ratio_lo=q(brr)[0], rmse_ratio_hi=q(brr)[1], hurts=int(q(bs)[1] < 0), hurts_rmse=int(q(brr)[0] > 1))


def cell(a):
    t, cut, learner = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    B1 = np.hstack([B0, DES]); grand = y.mean()
    splits = {"main_loco": [(c, g == c) for c in np.unique(g)]}
    splits.update(SP.split_defs(RUNS))
    variants = VARIANTS + (["Gonly"] if learner in ("ridge", "krr") else [])
    rows = []
    for sp, folds in splits.items():
        masks = [m for _, m in folds]
        P = {}
        for v in ["none"] + variants:
            P[v] = predict_split(B1, G, names[2], y, g, masks, learner, v)
        pb, tm = P["none"]
        ok = np.isfinite(pb)
        units = [("POOLED", ok)] + ([(nm, m) for nm, m in folds] if sp != "main_loco" and (len(folds) > 1 or sp == "leave_one_optimizer_out") else [])
        for nm, m in units:
            if m.sum() < 2 or not np.isfinite(pb[m]).all(): continue
            for v in variants:
                pv = P[v][0]
                e = evaluate(y[m], tm[m], grand, pb[m], pv[m], g[m], nb=2000 if nm == "POOLED" else 1000)
                rows.append(dict(dataset=M.DATASET, split=sp, fold=nm, target=t, cutoff=cut, learner=learner, variant=v, n_runs=int(m.sum()),
                                 n_configs=len(np.unique(g[m])), **e))
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--learners", default="ridge,krr,mlp")
    a = ap.parse_args()
    RUNS = L.get_runs()
    if M.DATASET == "fashion":   # no wd / optimizer folds for Fashion (as in Amendment 3)
        pass
    jobs = [(t, c, ln) for ln in a.learners.split(",")[::-1] for c in M.CUTS for t in M.PRIMARY]
    jobs.sort(key=lambda j: {"mlp": 0, "krr": 1, "ridge": 2}[j[2]])
    with Pool(a.workers) as p:
        res = [r for rs in p.imap_unordered(cell, jobs) for r in rs]
    M.write_csv(f"{M.OUT3}/learn_results.csv", res)
    print("wrote", len(res))


if __name__ == "__main__":
    main()
