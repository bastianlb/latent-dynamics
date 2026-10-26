#!/usr/bin/env python
"""Step 3: does early latent geometry predict eventual gap / test acc beyond early training curves?
LOCO (leave-one-config-out) ridge + GBM, paired config-level bootstrap of delta-R2, permutation test."""
import common2 as C
import data2 as D
import argparse, csv, itertools, os
from multiprocessing import Pool
import numpy as np
from scipy import stats

CUTS = [10, 30, 100, 300, 1000]
ALPHAS = np.logspace(-3, 3, 13)
GEO = ["fisher", "probe_acc", "eff_rank_pr", "h1_top5", "mag_t1", "pmag_fixed"]
LOGM = {"fisher", "eff_rank_pr", "h1_top5", "mag_t1", "pmag_fixed"}
lg = lambda x: np.log(np.maximum(x, 1e-6))


def feats(r, cut):
    d = r["d"]; st = r["step"]
    i = D.idx_le_mask(st, r["topo"], cut)
    g = lambda ps, m: d[f"{ps}/{m}"][i]
    ru = lambda m: d["run/" + m][i]
    inf = r["info"]
    B0 = dict(log_train_loss=lg(ru("train_loss")), log_test_loss=lg(ru("test_loss")), train_acc=ru("train_acc"),
              test_acc=ru("test_acc"), gap=ru("train_acc") - ru("test_acc"), log_norm=lg(g("test", "norm")))
    DES = dict(log10_n=np.log10(inf["n"]), log10_wd=np.log10(inf["wd"] + 1e-4), label_noise=inf["label_noise"])
    if C.RUNSET != "orig":   # amendment: optimizer + total steps are design variables in the extended sets
        DES.update(is_sgd=float(inf["is_sgd"]), log10_steps=np.log10(inf["train_steps"]))
    G = {}
    for m in GEO:
        t = lg(g("test", m)) if m in LOGM else g("test", m)
        tr = lg(g("train", m)) if m in LOGM else g("train", m)
        G[f"te_{m}"] = t; G[f"tr_{m}"] = tr; G[f"dl_{m}"] = tr - t
    return B0, DES, G, int(st[i])


def has_detector_mode(runs=()):
    """True only if phase_summary.csv has slingshot_detector_mode AND covers every run in the set (else onset falls back to the
    biased absolute-threshold detector)."""
    import csv as _c
    try:
        rd = _c.DictReader(open(C.PHASE_SUMMARY))
        if "slingshot_detector_mode" not in rd.fieldnames:
            return False
        ok = {r["run_id"] for r in rd if r.get("slingshot_detector_mode")}
        return all(r["info"]["run_id"] in ok for r in runs)
    except Exception:
        return False


def targets(r):
    d = r["d"]; st = r["step"]; T = r["info"]["train_steps"]
    out = {}
    gap = d["run/train_acc"] - d["run/test_acc"]; acc = d["run/test_acc"]
    pts = [("15k", D.idx_le(st, 15000)), ("fin", len(st) - 1)]
    if r["covered"]:
        pts.append(("pre", r["pre"])); pts.append(("preN", r["preN"]))
    for name, i in pts:
        out["gap_" + name] = float(gap[i]); out["acc_" + name] = float(acc[i])
    for name, m in (("15kw", (st >= 12000) & (st <= 18000)), ("finw", st >= 0.9 * T)):
        out["gap_" + name] = float(np.median(gap[m])); out["acc_" + name] = float(np.median(acc[m]))
    return out


def build(runs, cut):
    B0, DES, G, cfg, seeds = [], [], [], [], []
    for r in runs:
        b, de, g, _ = feats(r, cut)
        B0.append(list(b.values())); DES.append(list(de.values())); G.append(list(g.values()))
        cfg.append(r["info"]["config_id"]); seeds.append(r["info"]["seed"])
    names = (list(b.keys()), list(de.keys()), list(g.keys()))
    return np.array(B0, float), np.array(DES, float), np.array(G, float), np.array(cfg), np.array(seeds), names


# ---------- fast ridge with nested group-CV over alpha ----------
def _ridge_all(Xtr, ytr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1
    Xs = (Xtr - mu) / sd; Xt = (Xte - mu) / sd
    ym = ytr.mean()
    U, s, Vt = np.linalg.svd(Xs, full_matrices=False)
    uy = U.T @ (ytr - ym)
    sh = s[:, None] / (s[:, None] ** 2 + ALPHAS[None, :]) * uy[:, None]      # [k, nalpha]
    return (Xt @ Vt.T) @ sh + ym                                                 # [nte, nalpha]


def ridge_fit_predict(Xtr, ytr, gtr, Xte, rng=None):
    ug = np.unique(gtr)
    k = min(5, len(ug))
    perm = np.random.RandomState(0).permutation(ug)
    fold_of = {g: i % k for i, g in enumerate(perm)}
    f = np.array([fold_of[g] for g in gtr])
    sse = np.zeros(len(ALPHAS))
    for j in range(k):
        tr, va = f != j, f == j
        sse += ((_ridge_all(Xtr[tr], ytr[tr], Xtr[va]) - ytr[va][:, None]) ** 2).sum(0)
    a = int(np.argmin(sse))
    return _ridge_all(Xtr, ytr, Xte)[:, a]


def gbm_fit_predict(Xtr, ytr, gtr, Xte):
    from sklearn.ensemble import GradientBoostingRegressor
    m = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
    return m.fit(Xtr, ytr).predict(Xte)


def loco(X, y, g, model):
    fp = ridge_fit_predict if model == "ridge" else gbm_fit_predict
    p = np.zeros(len(y))
    for c in np.unique(g):
        te = g == c
        p[te] = fp(X[~te], y[~te], g[~te], X[te])
    return p


def r2(y, p):
    return 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)


def boot_delta(y, pb, pc, g, nb=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.unique(g); mem = {c: np.where(g == c)[0] for c in u}
    out = []
    for _ in range(nb):
        i = np.concatenate([mem[c] for c in rng.choice(u, len(u))])
        out.append(r2(y[i], pc[i]) - r2(y[i], pb[i]))
    return np.percentile(out, [2.5, 97.5])


def permute_G(G, g, seeds, rng):
    u = np.unique(g); pi = rng.permutation(u)
    Gp = G.copy()
    for c, c2 in zip(u, pi):
        a, b = np.where(g == c)[0], np.where(g == c2)[0]
        for k, ia in enumerate(a):
            Gp[ia] = G[b[k % len(b)]]
    return Gp


def cell(args):
    (tname, cut, runs_idx, nperm, do_gbm, tag) = args
    runs = [RUNS[i] for i in runs_idx]
    y = np.array([targets(r)[tname] for r in runs])
    B0, DES, G, g, seeds, names = build(runs, cut)
    sets = {"B0": B0, "B1": np.hstack([B0, DES]), "B0+G": np.hstack([B0, G]), "B1+G": np.hstack([B0, DES, G]),
            "G": G, "G+design": np.hstack([G, DES])}
    models = ["ridge"] + (["gbm"] if do_gbm else [])
    rows, preds = [], {}
    for mdl in models:
        for sn, X in sets.items():
            preds[(mdl, sn)] = loco(X, y, g, mdl)
            rows.append(dict(tag=tag, target=tname, cutoff=cut, model=mdl, kind="set", set=sn, base="", n_runs=len(y),
                             r2=r2(y, preds[(mdl, sn)]), spearman=stats.spearmanr(y, preds[(mdl, sn)])[0],
                             delta="", ci_lo="", ci_hi="", perm_p=""))
        for base, comp in (("B0", "B0+G"), ("B1", "B1+G")):
            lo, hi = boot_delta(y, preds[(mdl, base)], preds[(mdl, comp)], g)
            dlt = r2(y, preds[(mdl, comp)]) - r2(y, preds[(mdl, base)])
            pp = ""
            if mdl == "ridge" and nperm:
                rng = np.random.default_rng(1); cnt = 0
                Xb = sets[base]; nb = Xb.shape[1]
                for _ in range(nperm):
                    Gp = permute_G(G, g, seeds, rng)
                    Xc = np.hstack([Xb, Gp])
                    cnt += (r2(y, loco(Xc, y, g, "ridge")) - r2(y, preds[(mdl, base)])) >= dlt
                pp = (cnt + 1) / (nperm + 1)
            rows.append(dict(tag=tag, target=tname, cutoff=cut, model=mdl, kind="increment", set=comp, base=base,
                             n_runs=len(y), r2=r2(y, preds[(mdl, comp)]), spearman=stats.spearmanr(y, preds[(mdl, comp)])[0],
                             delta=dlt, ci_lo=lo, ci_hi=hi, perm_p=pp))
    return rows


def main():
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=5)
    ap.add_argument("--nperm", type=int, default=300)
    ap.add_argument("--no_gbm", action="store_true")
    a = ap.parse_args()
    RUNS = D.load_runs("mnist")
    print("runs covered by floor-relative slingshot files:", sum(r["covered"] for r in RUNS), "/", len(RUNS))
    print(len(RUNS), "mnist runs,", len({r['info']['config_id'] for r in RUNS}), "configs")
    allidx = list(range(len(RUNS)))
    TG = ["gap_15k", "acc_15k", "gap_15kw", "acc_15kw", "gap_fin", "acc_fin", "gap_finw", "acc_finw"]
    cov = [i for i in allidx if RUNS[i]["covered"]]
    s012 = [i for i in allidx if RUNS[i]["info"]["seed"] <= 2]
    f40 = [i for i in allidx if RUNS[i]["info"]["train_steps"] == 40000]
    jobs = []
    for cut in CUTS:
        for t in TG:
            jobs.append((t, cut, allidx, a.nperm, not a.no_gbm, "main"))
        for t in ("gap_15kw", "acc_15kw", "gap_finw", "acc_finw"):
            if len(s012) < len(allidx):
                jobs.append((t, cut, s012, a.nperm // 3, False, "sens_seeds012"))
            if f40 and len(f40) < len(allidx) and t.endswith("finw"):
                jobs.append((t, cut, f40, a.nperm // 3, not a.no_gbm, "final40k"))
        if cov:   # onset-dependent items on the runs covered by the floor-relative slingshot files
            sfx = "" if len(cov) == len(allidx) else "_subset"
            for tn in ("gap_pre", "acc_pre", "gap_preN", "acc_preN"):
                jobs.append((tn, cut, cov, a.nperm, not a.no_gbm, "pre_covered" + sfx))
            nosl = [i for i in cov if not (np.isfinite(RUNS[i]["onset"]) and RUNS[i]["onset"] <= 15000)]
            for tn in ("gap_15k", "acc_15k", "gap_15kw", "acc_15kw"):
                jobs.append((tn, cut, nosl, a.nperm // 3, False, "sens_excl_slingshot_le15k" + sfx))
    with Pool(a.workers) as p:
        res = [row for rs in p.imap(cell, jobs) for row in rs]
    keys = list(res[0].keys())
    with open(f"{C.OUT}/step3_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(res)
    print("wrote step3_results.csv", len(res))


if __name__ == "__main__":
    main()
