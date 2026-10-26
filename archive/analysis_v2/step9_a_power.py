#!/usr/bin/env python
"""Amendment 7 / A (post hoc): one-sided power and minimum detectable incremental gain on the confirmatory set (84 runs, 28 configs).
Per cell (2 targets x 5 cutoffs), step5_power geometry-shaped injection: s = z-scored te_<feat> residualised on [1, standardised B1];
beta = OLS(y ~ B1); e = y - fit; per rep the residual e is permuted among CONFIGS (step3_predict.permute_G; same permutation in all 10 cells of a rep, rng seeded by rep,
so cross-cell dependence of the residuals is retained), y0 = fit + e_perm, y_s = y0 + b s with b solved (step5_power.solve_b) so that s explains incremental R2 = r.
Tests (exactly step7): ridge LOCO (nested alpha) B1 vs B1+X, X in {full G (18), compact4, 1-dim block = the injected feature alone}; 10000-draw config bootstrap,
pre-registered one-sided p+ = (#{d*<=0}+1)/(NB+1), Holm over the 10 cells, and unadjusted per cell. Signal feature in {te_fisher, te_eff_rank_pr}. usage: [--nrep 200] [--workers 4]"""
import step9_common as Q
import step5_power as P5P
import argparse
from multiprocessing import Pool
import numpy as np
H, S, M, K7 = Q.H, Q.S, Q.M, Q.K7
LEVELS = [0.0, 0.01, 0.02, 0.03, 0.05, 0.10, 0.15, 0.20]
FEATS = ["te_fisher", "te_eff_rank_pr"]
NB = 10000
z = P5P.z
CELLDATA = None


def prep():
    global CELLDATA
    new = H.load_new(); CELLDATA = {}
    for (t, cut) in Q.CELLS:
        y, B1, G, names, g, B0 = H.cut_data(new, cut, t)
        seeds = np.array([r["info"].get("seed", 0) for r in new])
        c4 = M.gvariant_cols(names[2], "compact4")
        N = len(y); mu, sd = B1.mean(0), B1.std(0); sd[sd < 1e-12] = 1
        Xd = np.hstack([np.ones((N, 1)), (B1 - mu) / sd])
        beta = np.linalg.lstsq(Xd, y, rcond=None)[0]; fit = Xd @ beta
        CELLDATA[(t, cut)] = dict(y=y, B1=B1, G=G, g=g, seeds=seeds, names=names[2], c4=c4, Xd=Xd, fit=fit, e=y - fit, N=N,
                                  sig={f: z(G[:, names[2].index(f)] - Xd @ np.linalg.lstsq(Xd, G[:, names[2].index(f)], rcond=None)[0]) for f in FEATS},
                                  col={f: names[2].index(f) for f in FEATS})


def job(a):
    feat, rep = a
    rows = []
    for r in LEVELS:
        res = {v: [] for v in ("full", "compact4", "1dim")}
        for (t, cut) in Q.CELLS:
            d = CELLDATA[(t, cut)]; N, g = d["N"], d["g"]
            rng = np.random.default_rng(900000 + rep)                  # same config permutation in every cell and level of this rep
            ep = S.permute_G(d["e"][:, None], g, d["seeds"], rng)[:, 0]
            y0 = d["fit"] + ep; s = d["sig"][feat]
            a_ = float(s @ ep); sst0 = float(((y0 - y0.mean()) ** 2).sum())
            ys = y0 + P5P.solve_b(a_, N, sst0, r) * s
            B1, G = d["B1"], d["G"]
            Pr = {"b": S.loco(B1, ys, g, "ridge"), "full": S.loco(np.hstack([B1, G]), ys, g, "ridge"),
                  "compact4": S.loco(np.hstack([B1, G[:, d["c4"]]]), ys, g, "ridge"), "1dim": S.loco(np.hstack([B1, G[:, [d["col"][feat]]]]), ys, g, "ridge")}
            st = M.CfgStats(ys, Pr, g); W = st.draw(NB, np.random.default_rng(rep)); W1 = np.ones((1, st.nc))
            for v in res:
                dd = st.d_r2("b", v, W)
                res[v].append(dict(target=t, cutoff=cut, delta=st.d_r2("b", v, W1)[0], p_pos=(np.sum(dd <= 0) + 1) / (NB + 1), r2_B1=S.r2(ys, Pr["b"])))
        for v, lst in res.items():
            ph = K7.holm([x["p_pos"] for x in lst])
            for x, h in zip(lst, ph):
                rows.append(dict(feat=feat, level=r, rep=rep, variant=v, target=x["target"], cutoff=x["cutoff"], delta=x["delta"], r2_B1=x["r2_B1"], p_pos=x["p_pos"], p_pos_holm=h,
                                 det_unadj=int(x["p_pos"] <= .05), det_holm=int(h <= .05)))
    return rows


def mde(levels, power, thr=0.8):
    """smallest level where power>=thr, linear interpolation between neighbouring grid levels (nan if never)"""
    for i, (lv, p) in enumerate(zip(levels, power)):
        if p >= thr:
            if i > 0 and power[i - 1] < thr:
                return levels[i - 1] + (thr - power[i - 1]) / (p - power[i - 1]) * (lv - levels[i - 1])
            return lv
    return np.nan


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--nrep", type=int, default=200); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(); prep()
    jobs = [(f, rep) for f in FEATS for rep in range(a.nrep)]
    if a.smoke: jobs = jobs[:2]
    with Pool(a.workers) as p:
        rows = [r for rs in p.imap_unordered(job, jobs, chunksize=1) for r in rs]
    if a.smoke:
        print(len(rows)); return
    Q.write_csv("A_power_reps.csv", rows)
    import collections
    agg = collections.defaultdict(list)
    for r in rows: agg[(r["feat"], r["variant"], r["target"], r["cutoff"], r["level"])].append(r)
    summ = []
    for (f, v, t, c, lv), lst in sorted(agg.items()):
        summ.append(dict(feat=f, variant=v, target=t, cutoff=c, level=lv, n_rep=len(lst), det_unadj=np.mean([x["det_unadj"] for x in lst]), det_holm=np.mean([x["det_holm"] for x in lst]),
                         mean_dR2=np.mean([x["delta"] for x in lst]), sd_dR2=np.std([x["delta"] for x in lst]), r2_B1=np.mean([x["r2_B1"] for x in lst])))
    Q.write_csv("A_power_summary.csv", summ)
    # familywise: P(any Holm rejection among 10 cells), mean fraction of cells rejected
    fam = collections.defaultdict(list)
    for r in rows: fam[(r["feat"], r["variant"], r["level"], r["rep"])].append(r)
    fw = collections.defaultdict(list)
    for (f, v, lv, rep), lst in fam.items(): fw[(f, v, lv)].append((max(x["det_holm"] for x in lst), np.mean([x["det_holm"] for x in lst]), max(x["det_unadj"] for x in lst)))
    L = [f"Power simulation, confirmatory set (28 configs), ridge LOCO, pre-registered one-sided p+; reps={a.nrep}/level/signal; NB={NB}. Levels = true incremental R2 injected."]
    mrows = []
    for f in FEATS:
        for v in ("full", "compact4", "1dim"):
            L.append(f"\n=== signal {f}, test block {v} ===")
            L.append("level: " + " ".join(f"{lv:5.2f}" for lv in LEVELS))
            L.append("any-cell Holm rejection: " + " ".join(f"{np.mean([x[0] for x in fw[(f, v, lv)]]):5.2f}" for lv in LEVELS))
            L.append("mean frac. cells Holm-rejected: " + " ".join(f"{np.mean([x[1] for x in fw[(f, v, lv)]]):5.2f}" for lv in LEVELS))
            L.append("any-cell unadjusted rejection: " + " ".join(f"{np.mean([x[2] for x in fw[(f, v, lv)]]):5.2f}" for lv in LEVELS))
            for (t, c) in Q.CELLS:
                sub = {s["level"]: s for s in summ if s["feat"] == f and s["variant"] == v and s["target"] == t and s["cutoff"] == c}
                pu = [sub[lv]["det_unadj"] for lv in LEVELS]; ph = [sub[lv]["det_holm"] for lv in LEVELS]; md = [sub[lv]["mean_dR2"] for lv in LEVELS]
                m_u, m_h = mde(LEVELS, pu), mde(LEVELS, ph)
                mrows.append(dict(feat=f, variant=v, target=t, cutoff=c, r2_B1=sub[0.0]["r2_B1"], mde80_unadj=m_u, mde80_holm=m_h, size_unadj_at0=pu[0], size_holm_at0=ph[0]))
                L.append(f"{t:8s} cut{c:5d} R2(B1)={sub[0.0]['r2_B1']:.2f} MDE80 unadj {m_u:.3f} Holm {m_h:.3f} | unadj: " + " ".join(f"{x:5.2f}" for x in pu) + " | Holm: " + " ".join(f"{x:5.2f}" for x in ph)
                         + " | mean obs dR2: " + " ".join(f"{x:+.3f}" for x in md))
    Q.write_csv("A_power_mde.csv", mrows)
    L.append("\nMDE80 summary by (signal, block): median/min/max over the 10 cells (nan = 80% not reached up to level 0.20)")
    for f in FEATS:
        for v in ("full", "compact4", "1dim"):
            sub = [m for m in mrows if m["feat"] == f and m["variant"] == v]
            for k in ("mde80_unadj", "mde80_holm"):
                x = np.array([m[k] for m in sub]); L.append(f"  {f:15s} {v:9s} {k:12s} median {np.nanmedian(x) if np.isfinite(x).any() else np.nan:.3f} min {np.nanmin(x) if np.isfinite(x).any() else np.nan:.3f} max {np.nanmax(x) if np.isfinite(x).any() else np.nan:.3f}  not reached: {int((~np.isfinite(x)).sum())}/10")
    open(f"{Q.OUT6}/A_power_summary.txt", "w").write("\n".join(L)); print("\n".join(L))


if __name__ == "__main__":
    main()
