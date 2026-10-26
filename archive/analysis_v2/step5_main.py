#!/usr/bin/env python
"""Amendment 4: A4.1 margin sensitivity, A4.2a block-stratified bootstrap, A4.3 multiplicity (Holm). usage: --dataset mnist|fashion"""
import step5_lib as M
import argparse, pickle
from multiprocessing import Pool
import numpy as np
L, S = M.L, M.S
RUNS = None
MS = np.round(np.arange(0, 0.1501, 0.005), 4)
NB = 10000
PT = 0.005   # 0.5 points


def job(a):
    t, cut = a
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    B1 = np.hstack([B0, DES]); B1G = np.hstack([B1, G])
    return (t, cut), dict(y=y, g=g, blk=blk, **{f"{m}_{k}": S.loco(X, y, g, m) for m in ("ridge", "gbm") for k, X in (("B1", B1), ("B1G", B1G))})


def holm(p):
    o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0
    for r, i in enumerate(o):
        run = max(run, min(1, (n - r) * p[i])); adj[i] = run
    return adj


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    RUNS = L.get_runs()
    with Pool(a.workers) as p:
        res = dict(p.map(job, [(t, c) for t in M.TARGETS4 for c in M.CUTS]))
    pickle.dump(res, open(f"{M.OUT3}/oof_main.pkl", "wb"))
    curves, rows = [], []
    for (t, cut), d in res.items():
        y, g, blk = d["y"], d["g"], d["blk"]
        cb = M.config_blocks(g, blk)
        V = y.var()
        for mdl in ("ridge", "gbm"):
            st = M.CfgStats(y, {"b": d[f"{mdl}_B1"], "c": d[f"{mdl}_B1G"]}, g)
            obs = st.d_r2("b", "c", np.ones((1, st.nc)))[0]
            samp = {}
            for kind in ("iid", "block"):
                rng = np.random.default_rng(0)
                W = st.draw(NB, rng, None if kind == "iid" else cb)
                samp[kind] = st.d_r2("b", "c", W)
            r2b, r2c = S.r2(y, d[f"{mdl}_B1"]), S.r2(y, d[f"{mdl}_B1G"])
            rmse = np.sqrt((1 - r2b) * V)
            mg = L.margin(t)
            row = dict(dataset=M.DATASET, model=mdl, target=t, cutoff=cut, n_runs=len(y), n_configs=st.nc, r2_B1=r2b, r2_B1G=r2c, delta=obs,
                       var_y=V, rmse_B1_pts=100 * rmse, margin_prestated=mg,
                       dR2_improve_0p5pt=(rmse ** 2 - (rmse - PT) ** 2) / V, dR2_degrade_0p5pt=(rmse ** 2 - (rmse + PT) ** 2) / V)
            for kind, s in samp.items():
                q = np.percentile(s, [2.5, 97.5, 5, 95])
                row.update({f"lo95_{kind}": q[0], f"hi95_{kind}": q[1], f"lo90_{kind}": q[2], f"hi90_{kind}": q[3],
                            f"mstar_{kind}": max(abs(q[2]), abs(q[3])), f"tost_p_prestated_{kind}": M.tost_p(s, mg),
                            f"class_{kind}": L.classify(q[2], q[3], mg)["label"]})
                # m* from the TOST p-curve (grid): smallest grid m with p<=0.05
                ok = [m for m in MS if m > 0 and M.tost_p(s, m) <= 0.05]
                row[f"mstar_grid_{kind}"] = ok[0] if ok else np.nan
                if kind == "iid":
                    for m in MS:
                        curves.append(dict(dataset=M.DATASET, model=mdl, target=t, cutoff=cut, m=m, tost_p=M.tost_p(s, m) if m > 0 else 1.0,
                                           mass_inside=float(np.mean(np.abs(s) < m)) if m > 0 else 0.0))
            rows.append(row)
    M.write_csv(f"{M.OUT3}/margin_curves.csv", curves)
    M.write_csv(f"{M.OUT3}/margin_mstar.csv", rows)
    # ---- text tables
    L_ = []
    for mdl in ("ridge", "gbm"):
        L_.append(f"== {M.DATASET} {mdl}: m* (smallest margin with 90%-CI equivalence; iid / block-stratified) and 0.5pt-RMSE reference dR2 (improve) ==")
        L_.append("target     cut   dR2     m*_iid  m*_block  class(prestated, iid)  | RMSE_B1(pts) dR2(0.5pt)")
        for t in M.TARGETS4:
            for c in M.CUTS:
                r = next(r for r in rows if r["model"] == mdl and r["target"] == t and r["cutoff"] == c)
                L_.append(f"{t:9s} {c:5d} {r['delta']:+.3f}  {r['mstar_iid']:.3f}   {r['mstar_block']:.3f}    {r['class_iid']:28s} | {r['rmse_B1_pts']:.2f}  {r['dR2_improve_0p5pt']:.4f}")
    # ---- A4.3 multiplicity
    hol = []
    for mdl in ("ridge",):
        sub = [r for r in rows if r["model"] == mdl and r["target"] in M.PRIMARY]
        for kind in ("iid", "block"):
            p = np.array([r[f"tost_p_prestated_{kind}"] for r in sub]); adj = holm(p)
            for r, pa, pv in zip(sub, adj, p):
                hol.append(dict(dataset=M.DATASET, model=mdl, kind=kind, target=r["target"], cutoff=r["cutoff"], delta=r["delta"], margin=r["margin_prestated"],
                                tost_p=pv, holm_p=pa, equivalent_unadj=int(pv <= 0.05), equivalent_holm=int(pa <= 0.05),
                                positive_lo90_gt0=int(r[f"lo90_{kind}"] > 0)))
    M.write_csv(f"{M.OUT3}/multiplicity_holm.csv", hol)
    L_.append(f"== {M.DATASET}: Holm-adjusted TOST over the 10 primary ridge cells ==")
    for kind in ("iid", "block"):
        h = [x for x in hol if x["kind"] == kind]
        L_.append(f" [{kind}] equivalent unadjusted {sum(x['equivalent_unadj'] for x in h)}/10, after Holm {sum(x['equivalent_holm'] for x in h)}/10")
        for x in h:
            L_.append(f"    {x['target']:9s} cut{x['cutoff']:5d} dR2 {x['delta']:+.3f} TOST p {x['tost_p']:.4f} Holm p {x['holm_p']:.4f}")
    L_.append(f"== {M.DATASET}: distinct (target,cutoff) summary, ridge: window vs point label ==")
    for fam in ("acc", "gap"):
        for c in M.CUTS:
            w = next(r for r in rows if r["model"] == "ridge" and r["target"] == f"{fam}_15kw" and r["cutoff"] == c)
            pt = next(r for r in rows if r["model"] == "ridge" and r["target"] == f"{fam}_15k" and r["cutoff"] == c)
            L_.append(f"  {fam} cut{c:5d}: window {w['class_iid']:14s} (dR2 {w['delta']:+.3f}, m* {w['mstar_iid']:.3f}) | point {pt['class_iid']:14s} (dR2 {pt['delta']:+.3f}, m* {pt['mstar_iid']:.3f})")
    open(f"{M.OUT3}/main_summary.txt", "w").write("\n".join(L_)); print("\n".join(L_))


if __name__ == "__main__":
    main()
