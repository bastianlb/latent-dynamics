#!/usr/bin/env python
"""A5: power by simulation with an injected synthetic geometry-like feature (set (ii), ridge, base B1)."""
import step4_lib as L
import argparse
from multiprocessing import Pool
import numpy as np
S = L.S
RUNS = None
TARGETS = ["acc_15kw", "gap_15kw"]
CUTS = [100, 1000]
RHOS = [0.0, 0.2, 0.3, 0.5, 0.7]
NREP = 60
z = lambda v: (v - v.mean()) / (v.std() + 1e-12)


def job(args):
    tname, cut, rho = args
    y = L.hasher(RUNS, tname)
    B0, DES, G, g, seeds, names, EX = L.build_all(RUNS, cut, with_ext=False)
    B1 = np.hstack([B0, DES]); cc = L.compact_cols(names[2]); Gc = G[:, cc]
    pb = S.loco(B1, y, g, "ridge"); e = z(y - pb)
    rng = np.random.default_rng(1000 * cut + int(rho * 100) + (7 if tname.startswith("gap") else 0))
    out = []
    for rep in range(NREP):
        Gp = S.permute_G(G, g, seeds, rng)                     # real G-like noise: config-permuted real column
        n = z(Gp[:, rng.integers(0, G.shape[1])])
        n = n - (n @ e) / (e @ e) * e                         # orthogonal to e (in-sample, so corr(s,e)=rho exactly)
        n = z(n)
        s = rho * e + np.sqrt(1 - rho ** 2) * n
        for var, Gx in (("full G + s", G), ("compact G + s", Gc)):
            pc = S.loco(np.hstack([B1, Gx, s[:, None]]), y, g, "ridge")
            bs = L.boot_samples(y, pb, pc, g, nb=500, seed=rep)
            d = S.r2(y, pc) - S.r2(y, pb)
            out.append(dict(target=tname, cutoff=cut, rho=rho, variant=var, rep=rep, delta=d, lo95=np.percentile(bs, 2.5), r2_B1=S.r2(y, pb),
                            detected=int(np.percentile(bs, 2.5) > 0)))
    return out


def main():
    global RUNS
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=3)
    a = ap.parse_args()
    RUNS = L.get_runs()
    jobs = [(t, c, r) for t in TARGETS for c in CUTS for r in RHOS]
    with Pool(a.workers) as p:
        res = [row for rs in p.imap(job, jobs) for row in rs]
    L.write_csv(f"{L.OUT2}/step4_power_reps.csv", res)
    lines, summ = [], []
    for t in TARGETS:
        for c in CUTS:
            for var in ("full G + s", "compact G + s"):
                tab = []
                for rho in RHOS:
                    sub = [r for r in res if r["target"] == t and r["cutoff"] == c and r["variant"] == var and r["rho"] == rho]
                    tab.append((rho, np.mean([r["detected"] for r in sub]), np.mean([r["delta"] for r in sub]), np.std([r["delta"] for r in sub]), len(sub), sub[0]["r2_B1"]))
                mde = "not reached"
                for i, (rho, dp, md, sd, n, rb) in enumerate(tab):
                    if dp >= 0.8:
                        if i > 0 and tab[i - 1][1] < 0.8:
                            p0, p1 = tab[i - 1], tab[i]
                            mde = f"{p0[2] + (0.8 - p0[1]) / (p1[1] - p0[1]) * (p1[2] - p0[2]):.4f}"
                        else:
                            mde = f"{md:.4f}"
                        break
                lines.append(f"[{t} cut{c}] {var}: R2(B1)={tab[0][5]:.3f}; MDE(dR2 at 80% power) = {mde}")
                for rho, dp, md, sd, n, rb in tab:
                    lines.append(f"   rho={rho:.1f}  detection {dp:.2f} (n={n})  mean dR2 {md:+.4f} (sd {sd:.4f})")
                    summ.append(dict(target=t, cutoff=c, variant=var, rho=rho, detection_prob=dp, mean_delta=md, sd_delta=sd, n_rep=n, r2_B1=rb, mde_80=mde))
    L.write_csv(f"{L.OUT2}/step4_power_summary.csv", summ)
    open(f"{L.OUT2}/step4_power_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
