#!/usr/bin/env python
"""Amendment 7 / C (post hoc): promoted low-dimensional table on the confirmatory set: B1 vs B1+X, X in {te_fisher, te_eff_rank_pr, te_mag_t1, PCA-1(G, in-fold), compact4, full G, CV18, noise18}.
Reuses ext5/A_conf_ii_lowdim.csv (1-dim, PCA1), confirm/results/primary_ridge.csv (full G, compact4), ext5/E_curve_block_placebo.csv (CV18). Only noise18 is recomputed
(5 draws, Z.noise_block draws 0..4, draw-mean inside every bootstrap rep, 10000 reps, default_rng(0)) because ext5 stores only its 95% CI.
Summary per X: mean dR2 over the 10 cells; #cells with 90% CI upper bound < 0; #cells with 90% CI lower bound > 0; max-cell dR2 with 90% and 95% CI; Holm-adjusted one-sided p+ (family of 10) min."""
import step9_common as Q
import csv
import numpy as np
H, S, M, Z = Q.H, Q.S, Q.M, Q.H.Z
E5 = H.OUT
rd = lambda p: list(csv.DictReader(open(p)))
f = float
rec = {}
for r in rd(f"{E5}/A_conf_ii_lowdim.csv"):
    nm = {"te_fisher": "te_fisher", "te_eff_rank_pr": "te_eff_rank_pr", "te_mag_t1": "te_mag_t1", "PCA1(G,infold)": "PCA-1(G, in-fold)"}.get(r["block"])
    if nm: rec.setdefault(nm, []).append(r)
for r in rd(f"{H.C.OUT_ROOT}/confirm/results/primary_ridge.csv"):
    rec.setdefault({"G": "full G (18)", "compact4": "compact4"}[r["X"]], []).append(r)
for r in rd(f"{E5}/E_curve_block_placebo.csv"):
    if r["contrast"] == "CV18 vs B1": rec.setdefault("CV18", []).append(r)
# noise18 recompute
new = H.load_new(); n = len(new); nz = []
for t in Q.TARGETS:
    for cut in Q.CUTS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        P = {"b": S.loco(B1, y, g, "ridge")}
        for d_ in range(Z.NDRAW): P[f"nz{d_}"] = S.loco(np.hstack([B1, Z.noise_block(n, 18, d_)]), y, g, "ridge")
        st = M.CfgStats(y, P, g); W = st.draw(10000, np.random.default_rng(0)); W1 = np.ones((1, st.nc))
        bs = np.mean([st.d_r2("b", f"nz{d_}", W) for d_ in range(Z.NDRAW)], 0); ob = np.mean([st.d_r2("b", f"nz{d_}", W1)[0] for d_ in range(Z.NDRAW)])
        q = np.percentile(bs, [2.5, 97.5, 5, 95])
        nz.append(dict(target=t, cutoff=cut, delta=ob, lo95=q[0], hi95=q[1], lo90=q[2], hi90=q[3], p_pos=(np.sum(bs <= 0) + 1) / 10001))
rec["noise18"] = nz
order = ["te_fisher", "te_eff_rank_pr", "te_mag_t1", "PCA-1(G, in-fold)", "compact4", "full G (18)", "CV18", "noise18"]
cells, summ, T = [], [], []
for X in order:
    rs = rec[X]; assert len(rs) == 10, (X, len(rs))
    ph = H.K7.holm([f(r["p_pos"]) for r in rs])
    for r, h in zip(rs, ph):
        cells.append(dict(X=X, target=r["target"], cutoff=int(r["cutoff"]), dR2=f(r["delta"]), lo90=f(r["lo90"]), hi90=f(r["hi90"]), lo95=f(r["lo95"]), hi95=f(r["hi95"]), p_pos=f(r["p_pos"]), p_pos_holm=h))
    d = np.array([f(r["delta"]) for r in rs]); i = int(np.argmax(d)); r = rs[i]
    summ.append(dict(X=X, n_dims={"te_fisher": 1, "te_eff_rank_pr": 1, "te_mag_t1": 1, "PCA-1(G, in-fold)": 1, "compact4": 4, "full G (18)": 18, "CV18": 18, "noise18": 18}[X],
                     mean_dR2=d.mean(), n_cells_hi90_below0=int(sum(f(r["hi90"]) < 0 for r in rs)), n_cells_lo90_above0=int(sum(f(r["lo90"]) > 0 for r in rs)),
                     max_cell=f"{r['target']} cut{r['cutoff']}", max_dR2=d[i], max_lo90=f(r["lo90"]), max_hi90=f(r["hi90"]), max_lo95=f(r["lo95"]), max_hi95=f(r["hi95"]), min_p_pos_holm=min(ph)))
Q.write_csv("C_lowdim_cells.csv", cells); Q.write_csv("C_lowdim_table.csv", summ)
T.append("Confirmatory set (28 configs), ridge LOCO, B1 vs B1+X (B1 = 6 curve + 5 design). 10 cells = 2 targets x 5 cutoffs; CIs = config bootstrap.")
T.append(f"{'X':20s} dims mean_dR2 #hi90<0 #lo90>0 | max cell dR2 [90% CI] [95% CI] | min Holm p+")
for s in summ:
    T.append(f"{s['X']:20s} {s['n_dims']:4d} {s['mean_dR2']:+.4f}  {s['n_cells_hi90_below0']:2d}/10  {s['n_cells_lo90_above0']:2d}/10 | {s['max_cell']:14s} {s['max_dR2']:+.4f} [{s['max_lo90']:+.3f},{s['max_hi90']:+.3f}] [{s['max_lo95']:+.3f},{s['max_hi95']:+.3f}] | {s['min_p_pos_holm']:.3f}")
open(f"{Q.OUT6}/C_lowdim_table.txt", "w").write("\n".join(T)); print("\n".join(T))
