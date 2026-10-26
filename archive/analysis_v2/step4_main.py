#!/usr/bin/env python
"""A1 (equivalence) + A4 (B2 = B1 + learning-curve extrapolation) + Fashion 'main step-3 test'.
usage: python step4_main.py --dataset mnist|fashion [--workers 4] [--nperm 0] [--no_gbm]"""
import step4_lib as L
import argparse
from multiprocessing import Pool
import numpy as np
from scipy import stats
S = L.S

TARGETS = ["acc_15kw", "gap_15kw", "acc_15k", "gap_15k"]
RUNS = None


def cell(args):
    tname, cut, nperm, do_gbm = args
    y = L.hasher(RUNS, tname)
    B0, DES, G, g, seeds, names, EX = L.build_all(RUNS, cut)
    B1 = np.hstack([B0, DES]); B2 = np.hstack([B1, EX])
    sets = {"B1": B1, "B1+G": np.hstack([B1, G]), "G": G, "B2": B2, "B2+G": np.hstack([B2, G])}
    rows = []
    for mdl in ["ridge"] + (["gbm"] if do_gbm else []):
        P = {k: S.loco(X, y, g, mdl) for k, X in sets.items()}
        for k in sets:
            rows.append(dict(kind="set", target=tname, cutoff=cut, model=mdl, set=k, base="", n_runs=len(y), r2=S.r2(y, P[k]),
                             spearman=stats.spearmanr(y, P[k])[0]))
        for base, comp in (("B1", "B1+G"), ("B1", "B2"), ("B2", "B2+G")):
            s = L.summarize(y, P[base], P[comp], g, tname)
            row = dict(kind="increment", target=tname, cutoff=cut, model=mdl, set=comp, base=base, n_runs=len(y),
                       r2=S.r2(y, P[comp]), r2_base=S.r2(y, P[base]), **s)
            if nperm and mdl == "ridge" and (base, comp) == ("B1", "B1+G"):
                rng = np.random.default_rng(1); cnt = 0
                for _ in range(nperm):
                    Gp = S.permute_G(G, g, seeds, rng)
                    cnt += (S.r2(y, S.loco(np.hstack([B1, Gp]), y, g, "ridge")) - S.r2(y, P["B1"])) >= s["delta"]
                row["perm_p"] = (cnt + 1) / (nperm + 1)
            rows.append(row)
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nperm", type=int, default=0); ap.add_argument("--no_gbm", action="store_true")
    a = ap.parse_args()
    RUNS = L.get_runs()
    print(L.DATASET, len(RUNS), "runs", len({r["info"]["config_id"] for r in RUNS}), "configs", flush=True)
    jobs = [(t, c, a.nperm, not a.no_gbm) for c in L.CUTS for t in L.TARGETS] if False else [(t, c, a.nperm, not a.no_gbm) for c in L.CUTS for t in TARGETS]
    with Pool(a.workers) as p:
        res = [row for rs in p.imap(cell, jobs) for row in rs]
    L.write_csv(f"{L.OUT2}/step4_main_results.csv", res)
    # compact text report
    f = lambda x: float(x)
    lines = []
    for t in TARGETS:
        for mdl in ("ridge", "gbm"):
            sub = [r for r in res if r["target"] == t and r["model"] == mdl]
            if not sub: continue
            lines.append(f"[{t}] {mdl} n_runs={sub[0]['n_runs']}  margin +-{L.margin(t)}")
            for base, comp in (("B1", "B1+G"), ("B1", "B2"), ("B2", "B2+G")):
                for c in L.CUTS:
                    r = next(r for r in sub if r["kind"] == "increment" and r["base"] == base and r["set"] == comp and r["cutoff"] == c)
                    lines.append(f"   {base}->{comp:5s} cut{c:5d}: R2 {f(r['r2_base']):+.3f}->{f(r['r2']):+.3f} dR2 {f(r['delta']):+.3f} "
                                 f"95%[{f(r['lo95']):+.3f},{f(r['hi95']):+.3f}] 90%[{f(r['lo90']):+.3f},{f(r['hi90']):+.3f}] m*={f(r['min_margin']):.3f} {r['label']}"
                                 + (f" perm_p={r['perm_p']:.3f}" if "perm_p" in r else ""))
            for k in ("G", "B2"):
                lines.append(f"   R2({k}) " + " ".join(f"{f(next(r for r in sub if r['kind']=='set' and r['set']==k and r['cutoff']==c)['r2']):+.3f}" for c in L.CUTS))
    open(f"{L.OUT2}/step4_main_summary.txt", "w").write("\n".join(lines)); print("\n".join(lines))


if __name__ == "__main__":
    main()
