#!/usr/bin/env python
"""A5.E: within-config (seed-level) signal in accuracy points. Target = y - config mean (acc_15kw, gap_15kw); features centred per config; ridge LOCO.
RMSE (points) of: null (0), B0c, B0c+G, B0c+compact4, G alone; differences with config-bootstrap CIs. usage: --dataset mnist|fashion"""
import step6_lib as Z
import argparse
import numpy as np
from scipy import stats
L, S, M = Z.L, Z.S, Z.M


def center(X, g):
    Xc = X.copy()
    for c in np.unique(g):
        m = g == c; Xc[m] = X[m] - X[m].mean(0)
    return Xc


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); a = ap.parse_args()
    RUNS = L.get_runs(); rows, T = [], []
    for t in M.PRIMARY:
        for cut in L.CUTS:
            y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
            yc = center(y[:, None], g)[:, 0]; Bc, Gc = center(B0, g), center(G, g)
            c4 = M.gvariant_cols(names[2], "compact4")
            sets = {"B0c": Bc, "B0c+G": np.hstack([Bc, Gc]), "B0c+c4": np.hstack([Bc, Gc[:, c4]]), "Gc": Gc}
            P = {k: S.loco(X, yc, g, "ridge") for k, X in sets.items()}; P["null"] = np.zeros(len(y))
            st = M.CfgStats(yc, P, g); W1 = np.ones((1, st.nc)); W = st.draw(10000, np.random.default_rng(0))
            n1, nW = st.n.sum(), W @ st.n
            rm = lambda k, w, nn: 100 * np.sqrt((w @ st.sse[k]) / nn)
            sd_within = 100 * np.sqrt(np.mean([np.var(y[g == c], ddof=1) for c in np.unique(g)]))
            base = dict(dataset=Z.DATASET, target=t, cutoff=cut, n_runs=len(y), n_cfg=st.nc, within_sd_ddof1_pts=sd_within, null_rmse_pts=rm("null", W1[0], n1))
            q = lambda s: np.percentile(s, [2.5, 97.5])
            for k in ("B0c", "B0c+G", "B0c+c4", "Gc"):
                r = rm(k, W1[0], n1); b = np.array([rm(k, w, nn) for w, nn in zip(W, nW)])
                row = dict(base, model=k, rmse_pts=r, rmse_lo95=q(b)[0], rmse_hi95=q(b)[1], r2_centred=1 - st.sse[k].sum() / st.sse["null"].sum(), spearman=stats.spearmanr(yc, P[k])[0])
                if k in ("B0c+G", "B0c+c4"):
                    d = (W1[0] @ st.sse["B0c"] - W1[0] @ st.sse[k]) / n1
                    bd = (W @ st.sse["B0c"] - W @ st.sse[k]) / nW
                    row.update(rmse_gain_vs_B0c_pts=rm("B0c", W1[0], n1) - r,
                               rmse_gain_lo95=q(100 * np.sqrt(np.maximum(W @ st.sse["B0c"] / nW, 0)) - np.array([rm(k, w, nn) for w, nn in zip(W, nW)]))[0],
                               rmse_gain_hi95=q(100 * np.sqrt(np.maximum(W @ st.sse["B0c"] / nW, 0)) - np.array([rm(k, w, nn) for w, nn in zip(W, nW)]))[1],
                               explained_sd_pts=100 * np.sqrt(max(d, 0)), explained_sd_lo95=100 * np.sqrt(max(q(bd)[0], 0)), explained_sd_hi95=100 * np.sqrt(max(q(bd)[1], 0)),
                               dR2_centred=d / (st.sse["null"].sum() / n1), )
                rows.append(row)
    M.write_csv(f"{Z.OUT4}/E_within_config_rmse.csv", rows)
    for t in M.PRIMARY:
        T.append(f"== {Z.DATASET} {t}: within-config residual target, ridge LOCO; RMSE in points (target units x100) [95% config bootstrap] ==")
        for cut in L.CUTS:
            sub = {r["model"]: r for r in rows if r["target"] == t and r["cutoff"] == cut}
            b = sub["B0c"]
            line = f"  cut{cut:5d} within-cfg SD(ddof1) {b['within_sd_ddof1_pts']:.2f} | null RMSE {b['null_rmse_pts']:.2f} | B0c {b['rmse_pts']:.2f}"
            for k in ("B0c+G", "B0c+c4"):
                r = sub[k]
                line += f" | {k} {r['rmse_pts']:.2f} (gain {r['rmse_gain_vs_B0c_pts']:+.2f} [{r['rmse_gain_lo95']:+.2f},{r['rmse_gain_hi95']:+.2f}]; explained SD {r['explained_sd_pts']:.2f} [{r['explained_sd_lo95']:.2f},{r['explained_sd_hi95']:.2f}]; dR2c {r['dR2_centred']:+.3f})"
            T.append(line)
    open(f"{Z.OUT4}/E_within_config_rmse.txt", "w").write("\n".join(T)); print("\n".join(T))


if __name__ == "__main__":
    main()
