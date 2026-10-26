#!/usr/bin/env python
"""A2: extrapolation splits (leave-one-n-out, leave-one-wd-level-out, leave-one-optimizer-out). usage: --dataset mnist|fashion"""
import step4_lib as L
import argparse
from multiprocessing import Pool
import numpy as np
S = L.S
RUNS = None
TARGETS = ["acc_15kw", "gap_15kw"]


def split_defs(runs):
    cfg = np.array([r["info"]["config_id"] for r in runs])
    nl = np.array([r["info"]["n_label"] for r in runs]); wd = np.array([r["info"]["wd"] for r in runs])
    sgd = np.array([r["info"]["is_sgd"] for r in runs])
    defs = {}
    defs["leave_one_n_out"] = [(f"n={v}", nl == v) for v in ["250", "1000", "4000", "16000", "full"] if (nl == v).any()]
    if L.DATASET == "mnist":
        lv = [v for v in np.unique(wd) if len({c for c in cfg[wd == v]}) >= 4]
        defs["leave_one_wd_out"] = [(f"wd={v:g}", wd == v) for v in lv]
        if sgd.any():
            defs["leave_one_optimizer_out"] = [("test=SGD", sgd == 1)]
    return defs


def cell(args):
    tname, cut, mdl = args
    y = L.hasher(RUNS, tname)
    B0, DES, G, g, seeds, names, EX = L.build_all(RUNS, cut, with_ext=False)
    B1 = np.hstack([B0, DES])
    sets = {"B1": B1, "B1+G": np.hstack([B1, G]), "G": G}
    rows = []
    for split, folds in split_defs(RUNS).items():
        masks = [m for _, m in folds]
        P = {k: L.fold_predict(X, y, g, masks, mdl) for k, X in sets.items()}
        # pooled
        ok = np.isfinite(P["B1"])
        yy, gg = y[ok], g[ok]
        pb, pc, pg = P["B1"][ok], P["B1+G"][ok], P["G"][ok]
        s = L.summarize(yy, pb, pc, gg, tname)
        rows.append(dict(split=split, fold="POOLED", target=tname, cutoff=cut, model=mdl, n_runs=int(ok.sum()), n_configs=len(np.unique(gg)),
                         r2_B1=S.r2(yy, pb), r2_B1G=S.r2(yy, pc), r2_G=S.r2(yy, pg), **s))
        if len(folds) > 1 or split == "leave_one_optimizer_out":
            for (nm, m) in folds:
                if m.sum() < 2: continue
                yy, gg = y[m], g[m]
                pb, pc, pg = P["B1"][m], P["B1+G"][m], P["G"][m]
                if not np.isfinite(pb).all(): continue
                s = L.summarize(yy, pb, pc, gg, tname, nb=1000)
                rows.append(dict(split=split, fold=nm, target=tname, cutoff=cut, model=mdl, n_runs=int(m.sum()), n_configs=len(np.unique(gg)),
                                 r2_B1=S.r2(yy, pb), r2_B1G=S.r2(yy, pc), r2_G=S.r2(yy, pg), **s))
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="mnist"); ap.add_argument("--workers", type=int, default=3); ap.add_argument("--no_gbm", action="store_true")
    a = ap.parse_args()
    RUNS = L.get_runs()
    print(L.DATASET, len(RUNS), "runs", {k: [n for n, _ in v] for k, v in split_defs(RUNS).items()}, flush=True)
    jobs = [(t, c, m) for m in ["ridge"] + ([] if a.no_gbm else ["gbm"]) for c in L.CUTS for t in TARGETS]
    with Pool(a.workers) as p:
        res = [row for rs in p.imap(cell, jobs) for row in rs]
    L.write_csv(f"{L.OUT2}/step4_splits_results.csv", res)
    lines = []
    for sp in dict.fromkeys(r["split"] for r in res):
        for t in TARGETS:
            for mdl in ("ridge", "gbm"):
                for fold in dict.fromkeys(r["fold"] for r in res if r["split"] == sp):
                    sub = sorted([r for r in res if r["split"] == sp and r["target"] == t and r["model"] == mdl and r["fold"] == fold], key=lambda r: r["cutoff"])
                    if not sub: continue
                    lines.append(f"[{sp}] {t} {mdl} fold={fold} (n_runs={sub[0]['n_runs']}, configs={sub[0]['n_configs']})")
                    lines.append("   cut      " + " ".join(f"{r['cutoff']:>8d}" for r in sub))
                    lines.append("   R2 B1    " + " ".join(f"{r['r2_B1']:+8.3f}" for r in sub))
                    lines.append("   R2 B1+G  " + " ".join(f"{r['r2_B1G']:+8.3f}" for r in sub))
                    lines.append("   R2 G     " + " ".join(f"{r['r2_G']:+8.3f}" for r in sub))
                    lines.append("   dR2      " + " ".join(f"{r['delta']:+8.3f}" for r in sub))
                    lines.append("   95%CI    " + " ".join(f"[{r['lo95']:+.2f},{r['hi95']:+.2f}]" for r in sub))
                    lines.append("   class    " + " | ".join(r["label"] for r in sub))
    open(f"{L.OUT2}/step4_splits_summary.txt", "w").write("\n".join(lines)); print(f"wrote {len(res)} rows")


if __name__ == "__main__":
    main()
