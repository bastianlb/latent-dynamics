#!/usr/bin/env python
"""Amendment 7 / B (post hoc): within-configuration (config-mean residual) analysis on the CONFIRMATORY runs (28 configs x 3 seeds), as step6_E / step4 3a.
Target = y - config mean (acc_15kw, gap_15kw); B0 curve columns and G (18) centred per config (design columns are constant within config, so drop out);
ridge LOCO (nested alpha) with the held-out config's own mean removed. Models: B0c, B0c+G, B0c+compact4 (+ control B0c + centred N(0,1) 18 cols, mean of 5 draws).
dR2c = (SSE(B0c)-SSE(X))/SS(centred target) (same definition as step6_E 'dR2_centred'); RMSE in points (x100); 10000-draw config bootstrap (default_rng(0)), 95% CIs."""
import step9_common as Q
import numpy as np
H, S, M, Z = Q.H, Q.S, Q.M, Q.H.Z
from step6_E import center
new = H.load_new()
rows, T = [], []
for t in Q.TARGETS:
    for cut in Q.CUTS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        yc = center(y[:, None], g)[:, 0]; Bc, Gc = center(B0, g), center(G, g)
        c4 = M.gvariant_cols(names[2], "compact4")
        P = {"B0c": S.loco(Bc, yc, g, "ridge"), "B0c+G": S.loco(np.hstack([Bc, Gc]), yc, g, "ridge"), "B0c+c4": S.loco(np.hstack([Bc, Gc[:, c4]]), yc, g, "ridge"), "null": np.zeros(len(y))}
        for d_ in range(Z.NDRAW):
            P[f"nz{d_}"] = S.loco(np.hstack([Bc, center(Z.noise_block(len(y), 18, d_), g)]), yc, g, "ridge")
        st = M.CfgStats(yc, P, g); W1 = np.ones(st.nc); W = st.draw(10000, np.random.default_rng(0))
        n1, nW = st.n.sum(), W @ st.n
        rm = lambda a, w, nn: 100 * np.sqrt((w @ st.sse[a]) / nn)
        sd_within = 100 * np.sqrt(np.mean([np.var(y[g == c], ddof=1) for c in np.unique(g)]))
        ss0 = lambda w: w @ st.sse["null"]
        base = dict(target=t, cutoff=cut, n_runs=len(y), n_cfg=st.nc, within_sd_ddof1_pts=sd_within, null_rmse_pts=rm("null", W1, n1), B0c_rmse_pts=rm("B0c", W1, n1))
        for k in ("B0c+G", "B0c+c4", "nz"):
            ks = [k] if k != "nz" else [f"nz{d_}" for d_ in range(Z.NDRAW)]
            dR = np.mean([(W1 @ st.sse["B0c"] - W1 @ st.sse[a]) / ss0(W1) for a in ks])
            bR = np.mean([(W @ st.sse["B0c"] - W @ st.sse[a]) / (W @ st.sse["null"]) for a in ks], 0)
            gain = np.mean([rm("B0c", W1, n1) - rm(a, W1, n1) for a in ks])
            bG = np.mean([100 * np.sqrt(W @ st.sse["B0c"] / nW) - 100 * np.sqrt(W @ st.sse[a] / nW) for a in ks], 0)
            q = np.percentile(bR, [2.5, 97.5, 5, 95]); qg = np.percentile(bG, [2.5, 97.5])
            rows.append(dict(base, model=k if k != "nz" else "B0c+NZ18(centred, mean 5 draws)", rmse_pts=base["B0c_rmse_pts"] - gain, dR2c=dR, dR2c_lo95=q[0], dR2c_hi95=q[1], dR2c_lo90=q[2], dR2c_hi90=q[3],
                             rmse_gain_pts=gain, rmse_gain_lo95=qg[0], rmse_gain_hi95=qg[1], p_pos=(np.sum(bR <= 0) + 1) / 10001))
        print(t, cut, flush=True)
# Holm over 10 cells per model (one-sided p+)
for mdl in {r["model"] for r in rows}:
    fam = [r for r in rows if r["model"] == mdl]
    for r, h in zip(fam, H.K7.holm([r["p_pos"] for r in fam])): r["p_pos_holm"] = h
Q.write_csv("B_within_config_confirmatory.csv", rows)
for t in Q.TARGETS:
    T.append(f"== CONFIRMATORY within-config residual target {t}: ridge LOCO over 28 configs (3 seeds each); dR2c [95% config bootstrap]; RMSE gain in accuracy points vs B0c ==")
    for cut in Q.CUTS:
        sub = {r["model"]: r for r in rows if r["target"] == t and r["cutoff"] == cut}; b = sub["B0c+G"]
        line = f"  cut{cut:5d} within-SD {b['within_sd_ddof1_pts']:.2f} null RMSE {b['null_rmse_pts']:.2f} B0c {b['B0c_rmse_pts']:.2f}"
        for k, r in sub.items():
            line += f" | {k} dR2c {r['dR2c']:+.3f} [{r['dR2c_lo95']:+.3f},{r['dR2c_hi95']:+.3f}] gain {r['rmse_gain_pts']:+.3f} [{r['rmse_gain_lo95']:+.3f},{r['rmse_gain_hi95']:+.3f}] p+ {r['p_pos']:.3f}/Holm {r['p_pos_holm']:.3f}"
        T.append(line)
open(f"{Q.OUT6}/B_within_config_confirmatory.txt", "w").write("\n".join(T)); print("\n".join(T))
