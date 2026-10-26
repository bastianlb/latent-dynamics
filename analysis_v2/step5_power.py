#!/usr/bin/env python
"""Amendment 4 A4.6: power simulation v2 (MNIST set ii, ridge, B1 base). usage: --dataset mnist [--nrep 200]"""
import step5_lib as M
import argparse
from multiprocessing import Pool
import numpy as np
L, S = M.L, M.S
RUNS = None
NREP = 200
RS = [0.0, 0.02, 0.05, 0.10, 0.15]
RHOS = [0.0, 0.2, 0.3, 0.5, 0.7]
z = lambda v: (v - v.mean()) / (v.std() + 1e-12)


def solve_b(a, N, sst0, r):
    if r <= 0: return 0.0
    return (-a * N * (1 - r) + N * np.sqrt((1 - r) * r * (N * sst0 - a * a))) / (N * N * (1 - r))


def evalset(y, pb, preds, g, nb=500, seed=0):
    st = M.CfgStats(y, {"b": pb, **preds}, g)
    W = st.draw(nb, np.random.default_rng(seed)); out = {}
    for k in preds:
        lo = np.percentile(st.d_r2("b", k, W), 2.5)
        out[k] = (S.r2(y, preds[k]) - S.r2(y, pb), lo)
    return out


def job(a):
    t, cut, kind = a    # kind: 'geo:te_eff_rank_pr' | 'geo:te_fisher' | 'resid'
    y, B0, DES, G, g, seeds, names, blk = M.get_data(RUNS, cut, t)
    B1 = np.hstack([B0, DES]); B1G = np.hstack([B1, G]); c8 = L.compact_cols(names[2]); B1c = np.hstack([B1, G[:, c8]])
    N = len(y)
    mu, sd = B1.mean(0), B1.std(0); sd[sd < 1e-12] = 1
    Xd = np.hstack([np.ones((N, 1)), (B1 - mu) / sd])
    rng = np.random.default_rng(1000 * cut + (7 if t.startswith("gap") else 0) + (13 if kind == "resid" else 0) + (3 if "fisher" in kind else 0))
    rows = []
    if kind.startswith("geo"):
        fn = kind.split(":")[1]; zc = G[:, names[2].index(fn)]
        s = zc - Xd @ np.linalg.lstsq(Xd, zc, rcond=None)[0]; s = z(s)
        beta = np.linalg.lstsq(Xd, y, rcond=None)[0]; fit = Xd @ beta; e = y - fit
        for r in RS:
            for rep in range(NREP):
                ep = S.permute_G(e[:, None], g, seeds, rng)[:, 0]
                y0 = fit + ep; a_ = float(s @ ep); sst0 = float(((y0 - y0.mean()) ** 2).sum())
                b = solve_b(a_, N, sst0, r); ys = y0 + b * s
                pb = S.loco(B1, ys, g, "ridge")
                res = evalset(ys, pb, {"full": S.loco(B1G, ys, g, "ridge"), "compact8": S.loco(B1c, ys, g, "ridge")}, g, seed=rep)
                for v, (d, lo) in res.items():
                    rows.append(dict(target=t, cutoff=cut, signal=kind, level=r, variant=v, rep=rep, delta=d, lo95=lo, detected=int(lo > 0), r2_B1=S.r2(ys, pb)))
    else:
        pb0 = S.loco(B1, y, g, "ridge"); e = z(y - pb0)
        for rho in RHOS:
            for rep in range(NREP):
                Gp = S.permute_G(G, g, seeds, rng); n = z(Gp[:, rng.integers(0, G.shape[1])])
                n = n - (n @ e) / (e @ e) * e; n = z(n)
                s = rho * e + np.sqrt(1 - rho ** 2) * n
                res = evalset(y, pb0, {"full": S.loco(np.hstack([B1G, s[:, None]]), y, g, "ridge"), "compact8": S.loco(np.hstack([B1c, s[:, None]]), y, g, "ridge")}, g, seed=rep)
                for v, (d, lo) in res.items():
                    rows.append(dict(target=t, cutoff=cut, signal="resid", level=rho, variant=v, rep=rep, delta=d, lo95=lo, detected=int(lo > 0), r2_B1=S.r2(y, pb0)))
    return rows


def main():
    global RUNS, NREP
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--nrep", type=int, default=200); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(); NREP = a.nrep; RUNS = L.get_runs()
    jobs = [(t, c, k) for k in ("geo:te_eff_rank_pr", "geo:te_fisher", "resid") for t in M.PRIMARY for c in (100, 1000)]
    with Pool(a.workers) as p:
        res = [r for rs in p.imap_unordered(job, jobs) for r in rs]
    M.write_csv(f"{M.OUT3}/power_reps.csv", res)
    summ, lines = [], []
    for k in dict.fromkeys(r["signal"] for r in res):
        for t in M.PRIMARY:
            for c in (100, 1000):
                for v in ("full", "compact8"):
                    tab = []
                    for lv in sorted({r["level"] for r in res if r["signal"] == k}):
                        sub = [r for r in res if r["signal"] == k and r["target"] == t and r["cutoff"] == c and r["variant"] == v and r["level"] == lv]
                        tab.append((lv, np.mean([r["detected"] for r in sub]), np.mean([r["delta"] for r in sub]), np.std([r["delta"] for r in sub]), len(sub), np.mean([r["r2_B1"] for r in sub])))
                    mde = np.nan
                    for i, (lv, dp, md, sd_, n, rb) in enumerate(tab):
                        if dp >= 0.8:
                            if i > 0 and tab[i - 1][1] < 0.8:
                                p0, p1 = tab[i - 1], tab[i]; mde = p0[0] + (0.8 - p0[1]) / (p1[1] - p0[1]) * (p1[0] - p0[0])
                            else: mde = lv
                            break
                    lines.append(f"[{k} | {t} cut{c} | {v}] R2(B1)~{tab[0][5]:.3f}; level at 80% power = {mde:.3f}" if mde == mde else f"[{k} | {t} cut{c} | {v}] R2(B1)~{tab[0][5]:.3f}; 80% power not reached")
                    for lv, dp, md, sd_, n, rb in tab:
                        lines.append(f"    level {lv:.2f}: detection {dp:.3f} (n={n}) mean dR2 {md:+.4f} (sd {sd_:.4f})")
                        summ.append(dict(signal=k, target=t, cutoff=c, variant=v, level=lv, detection=dp, mean_delta=md, sd_delta=sd_, n_rep=n, r2_B1=rb, level_at_80pct=mde))
    M.write_csv(f"{M.OUT3}/power_summary.csv", summ)
    open(f"{M.OUT3}/power_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
