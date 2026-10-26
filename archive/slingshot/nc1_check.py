"""NC1 artifact check: are the abrupt NC1 drops real collapse or an artifact of pinv(Sigma_B)? CPU only.

Usage: python nc1_check.py [--workers 4] [--force] [--skip_v1] [--drop_ratio 1.8] [--real_ratio 1.3] [--eig_jump 2.0]
Outputs only under /vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/nc1_check/ (cache/, drops.csv, verdict_by_cell.csv, figures/).
Per snapshot (test probe 'latents/', train probe 'latents_train/'; true labels):
  nc1          tr(Sw pinv(Sb))/C, rcond 1e-6 (== dynamics.py)
  ratio_a      tr(Sw)/tr(Sb)                         (inverse free)
  spread_c     tr(Sigma_c)/tr(Sigma_T)               (per class; spread_mean = mean over classes)
  eig[0..8]    nonzero eigenvalues of Sb (descending), cond = eig0/eig8, rank_kept = #eig > 1e-6*eig0
  nc1_ridge_e  tr(Sw (Sb+eps*mean_eig*I)^-1) restricted to range(Sb), eps in EPS (relative to mean nonzero eigenvalue)
  nc1_fullridge_e  same ridge but in the full d-dim space (null space of Sb included)
  nc1_topk     tr(Sw U_k L_k^-1 U_k^T)/C for the top k eigendirections of Sb, k in KS
Verdict per flagged drop (largest one-step fall of log NC1 at step>=100, ratio >= --drop_ratio; pre=i-1, post=i+1):
  real           ratio_a and spread_mean both fall >= --real_ratio over the same window
  pinv_artifact  neither falls >= --real_ratio, and smallest nonzero eig of Sb jumps >= --eig_jump (either direction)
  nc1_only       NC1 drops but neither (ratio_a, spread) nor an eig jump explains it
  mixed          one of ratio_a / spread falls, the other not
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, glob, json, csv, time, hashlib, pickle, collections, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_diagram as PD

V2 = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2"
V1 = "/vol/bitbucket/lbastian/latent-dynamics/runs"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/nc1_check"
CODE = "nc1c-1"
EPS = (1e-3, 1e-2, 1e-1)
KS = (1, 3, 5)
C = 10
TH = dict(drop_ratio=1.8, real_ratio=1.3, eig_jump=2.0)


def series(Zs, y):
    classes = np.unique(y); Cn = len(classes); n = len(y); out = collections.defaultdict(list)
    pc = np.array([(y == c).mean() for c in classes])
    for Z in Zs:
        d = Z.shape[1]; g = Z.mean(0)
        mu = np.array([Z[y == c].mean(0) for c in classes]); Mc = mu - g
        Sc = [np.cov(Z[y == c].T, bias=True).reshape(d, d) for c in classes]
        Sw = sum(p * s for p, s in zip(pc, Sc)); Sb = (Mc * pc[:, None]).T @ Mc
        w, U = np.linalg.eigh(Sb); o = np.argsort(w)[::-1]; w, U = w[o], U[:, o]
        e = w[:Cn - 1]; e = np.clip(e, 1e-30, None)
        tot = np.trace(Sw) + np.trace(Sb)
        out["nc1"].append(np.trace(Sw @ np.linalg.pinv(Sb, rcond=1e-6, hermitian=True)) / Cn)
        out["ratio_a"].append(np.trace(Sw) / np.trace(Sb)); out["spread"].append([np.trace(s) / tot for s in Sc])
        out["eig"].append(e); out["cond"].append(e[0] / e[-1]); out["rank_kept"].append(int((w > 1e-6 * w[0]).sum()))
        Ur, Sw_r = U[:, :Cn - 1], U[:, :Cn - 1].T @ Sw @ U[:, :Cn - 1]
        out["nc1_ridge"].append([np.trace(Sw_r @ np.diag(1 / (e + ep * e.mean()))) / Cn for ep in EPS])
        out["nc1_fullridge"].append([np.trace(Sw @ np.linalg.inv(Sb + ep * e.mean() * np.eye(d))) / Cn for ep in EPS])
        out["nc1_topk"].append([np.trace(Sw_r[:k, :k] @ np.diag(1 / e[:k])) / Cn for k in KS])
    return {k: np.array(v) for k, v in out.items()}


def find_drop(steps, nc1, ratio):
    ln = np.log(np.clip(nc1, 1e-9, None)); dd = np.diff(ln); dd[steps[1:] < 100] = 0
    i = int(dd.argmin()) + 1
    if dd[i - 1] > -np.log(ratio) or i + 1 >= len(steps): return None
    return i


def verdict(S, steps, i, th):
    pre, post = i - 1, min(i + 1, len(steps) - 1)
    r = lambda x: float(x[pre] / x[post])
    ra, rs = r(S["ratio_a"]), r(S["spread"].mean(1)); rn = r(S["nc1"]); lam = S["eig"][:, -1]
    lj = float(max(lam[post] / lam[pre], lam[pre] / lam[post]))
    # eigen jump also checked in +-1 window around the drop
    lj = max(lj, float(max(lam[min(i + 1, len(lam) - 1)] / lam[max(i - 2, 0)], lam[max(i - 2, 0)] / lam[min(i + 1, len(lam) - 1)])))
    a_ok, s_ok = ra >= th["real_ratio"], rs >= th["real_ratio"]
    v = "real" if a_ok and s_ok else ("mixed" if a_ok or s_ok else ("pinv_artifact" if lj >= th["eig_jump"] else "nc1_only"))
    ridge = [float(S["nc1_ridge"][pre, k] / S["nc1_ridge"][post, k]) for k in range(len(EPS))]
    topk = [float(S["nc1_topk"][pre, k] / S["nc1_topk"][post, k]) for k in range(len(KS))]
    full = [float(S["nc1_fullridge"][pre, k] / S["nc1_fullridge"][post, k]) for k in range(len(EPS))]
    return dict(verdict=v, drop_step=int(steps[i]), nc1_ratio=rn, ratio_a_drop=ra, spread_drop=rs, lam_min_jump=lj, cond_pre=float(S["cond"][pre]), cond_post=float(S["cond"][post]),
                rank_kept_pre=int(S["rank_kept"][pre]), rank_kept_post=int(S["rank_kept"][post]),
                **{f"ridge{ep:g}_drop": ridge[k] for k, ep in enumerate(EPS)}, **{f"fullridge{ep:g}_drop": full[k] for k, ep in enumerate(EPS)}, **{f"top{k}_drop": topk[j] for j, k in enumerate(KS)})


def snap_hash(rd):
    items = []
    for sub in ("latents", "latents_train"):
        for f in sorted(glob.glob(f"{rd}/{sub}/step_*.npz")):
            try: items.append((sub, os.path.basename(f), os.stat(f).st_size))
            except OSError: pass
    return hashlib.sha1((CODE + json.dumps(items)).encode()).hexdigest()


def do_one(j):
    rd, cache, force, name = j
    try:
        if not os.path.isfile(f"{rd}/model_final.pt"): return name, "skip_incomplete"
        h = snap_hash(rd); cp = f"{cache}/{name}.pkl"
        if not force and os.path.isfile(cp):
            try:
                if pickle.load(open(cp, "rb")).get("hash") == h: return name, "cached"
            except Exception: pass
        res = dict(hash=h, name=name, run_id=os.path.basename(rd), src=name[:2], cfg=json.load(open(f"{rd}/config.json")))
        for probe, sub in (("test", "latents"), ("train", "latents_train")):
            d = PD.load_def(rd, sub)
            if d is None: continue
            res[probe] = dict(steps=d["steps"], **series(d["Zs"], d["labels"]))
        tmp = cp + f".tmp{os.getpid()}"; pickle.dump(res, open(tmp, "wb")); os.replace(tmp, cp); return name, "ok"
    except Exception:
        return name, "FAIL " + traceback.format_exc()


def cell_of(cfg):
    return (float(cfg.get("label_noise") or 0.0), int(cfg.get("train_subset") or 60000), float(cfg.get("weight_decay") or 0.0))


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--force", action="store_true"); ap.add_argument("--skip_v1", action="store_true")
    ap.add_argument("--out", default=OUT); ap.add_argument("--aggregate_only", action="store_true")
    for k, v in TH.items(): ap.add_argument(f"--{k}", type=float, default=v)
    a = ap.parse_args(); TH.update({k: getattr(a, k) for k in TH}); out = a.out
    assert os.path.abspath(out).startswith("/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/"), "bad out"
    os.makedirs(f"{out}/cache", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True); t0 = time.time()
    d2 = sorted(set(glob.glob(f"{V2}/runs/mnist_probe_*_phase_*") + glob.glob(f"{V2}/runs/mnist_probe_*_gap_*")))
    d1 = [] if a.skip_v1 else sorted(glob.glob(f"{V1}/mnist_*_s[0-9]"))
    jobs = [(d, f"{out}/cache", a.force, ("v2_" if d in d2 else "v1_") + os.path.basename(d)) for d in d2 + d1 if os.path.isdir(d)]
    if not a.aggregate_only:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(do_one, jobs, chunksize=1)
        print(dict(collections.Counter(s.split()[0] for _, s in res)))
        for n, s in res:
            if s.startswith("FAIL"): print(s, n)
    aggregate(out, {j[3] for j in jobs}); print(f"done {time.time()-t0:.0f}s")


def aggregate(out, names):
    R = []
    for f in sorted(glob.glob(f"{out}/cache/*.pkl")):
        try:
            r = pickle.load(open(f, "rb"))
            if r["name"] in names: R.append(r)
        except Exception: pass
    rows = []; ex = []
    for r in R:
        c = cell_of(r["cfg"])
        for probe in ("test", "train"):
            if probe not in r: continue
            S = r[probe]; st = S["steps"]; i = find_drop(st, S["nc1"], TH["drop_ratio"])
            if i is None: continue
            v = verdict(S, st, i, TH)
            rows.append(dict(name=r["name"], src=r["src"], probe=probe, seed=r["cfg"]["seed"], ln=c[0], N=c[1], wd=c[2], arch=r["cfg"].get("tag", ""), **v)); ex.append((r["name"], probe, i))
    wcsv(f"{out}/drops.csv", rows)
    # verdict counts per (N, wd) for v2 test probe
    cnt = collections.defaultdict(collections.Counter); tot = collections.Counter()
    for x in rows:
        if x["src"] == "v2" and x["probe"] == "test": cnt[(x["ln"], x["N"], x["wd"])][x["verdict"]] += 1
    vr = [dict(ln=k[0], N=k[1], wd=k[2], **dict(v)) for k, v in sorted(cnt.items())]
    wcsv(f"{out}/verdict_by_cell.csv", vr)
    allc = collections.Counter((x["src"], x["probe"], x["verdict"]) for x in rows)
    wcsv(f"{out}/verdict_totals.csv", [dict(src=k[0], probe=k[1], verdict=k[2], n=v) for k, v in sorted(allc.items())])
    print("verdict totals:", dict(allc))
    # robust-measure grid (v2, ln=0, test probe): seed mean/std of final values + peak between-seed variance
    cells = collections.defaultdict(list)
    for r in R:
        if r["src"] == "v2" and "test" in r: cells[cell_of(r["cfg"])].append(r)
    meas = {"nc1": lambda S: S["nc1"], "ratio_a": lambda S: S["ratio_a"], "spread_mean": lambda S: S["spread"].mean(1), "ridge_e0.01": lambda S: S["nc1_ridge"][:, 1],
            "top3": lambda S: S["nc1_topk"][:, 1], "lam_min": lambda S: S["eig"][:, -1], "cond": lambda S: S["cond"]}
    grid = {}
    for k, rs in cells.items():
        g = dict(n_runs=len(rs))
        for m, fn in meas.items():
            fin = [fn(r["test"])[-1] for r in rs]; g[f"{m}_mean"] = float(np.mean(fin)); g[f"{m}_std"] = float(np.std(fin, ddof=1)) if len(fin) > 1 else np.nan; g[f"{m}_n"] = len(fin)
            if len(rs) > 1:
                com = rs[0]["test"]["steps"]; X = np.stack([fn(r["test"]) for r in rs])
                if not all(np.array_equal(r["test"]["steps"], com) for r in rs): continue
                X = np.log(np.clip(X, 1e-12, None)); v = X.var(0, ddof=1); v[0] = 0; rng2 = (X.mean(0).max() - X.mean(0).min()) ** 2 + 1e-30
                g[f"pv_{m}"] = float(v.max() / rng2); g[f"pv_{m}_n"] = len(rs)
        dr = [x for x in rows if x["src"] == "v2" and x["probe"] == "test" and (x["ln"], x["N"], x["wd"]) == k]
        g["frac_runs_with_drop"] = len(dr) / len(rs); g["frac_real"] = sum(x["verdict"] == "real" for x in dr) / len(rs); g["frac_real_n"] = len(rs)
        g["frac_runs_with_drop_n"] = len(rs); grid[k] = g
    wcsv(f"{out}/robust_grid.csv", [dict(ln=k[0], N=k[1], wd=k[2], **g) for k, g in sorted(grid.items())])
    figures(out, R, grid, rows, ex)


def figures(out, R, grid, rows, ex):
    fd = f"{out}/figures"; Ns = sorted({k[1] for k in grid if k[0] == 0}); wds = sorted({k[2] for k in grid if k[0] == 0})
    if not Ns: return
    g0 = {k: v for k, v in grid.items() if k[0] == 0}
    panels = [("nc1_mean", "final NC1 (pinv)"), ("ratio_a_mean", "final tr(Sw)/tr(Sb)"), ("ridge_e0.01_mean", "final NC1 ridge eps=1e-2 (range Sb)"), ("top3_mean", "final NC1 top-3 Sb dirs"),
              ("spread_mean_mean", "final mean tr(Sc)/tr(St)"), ("lam_min_mean", "final smallest nonzero eig(Sb)"), ("pv_nc1", "peak seed var/range^2 log NC1 (pinv)"), ("pv_ratio_a", "peak seed var log tr(Sw)/tr(Sb)"),
              ("pv_ridge_e0.01", "peak seed var log NC1 ridge"), ("pv_top3", "peak seed var log NC1 top-3"), ("frac_runs_with_drop", "frac runs with flagged NC1 drop"), ("frac_real", "frac runs with REAL drop")]
    fig, ax = plt.subplots(3, 4, figsize=(17, 11)); ax = ax.ravel()
    for a, (k, t) in zip(ax, panels):
        if k.endswith("_mean"): PD.heat(a, g0, k[:-5], 0.0, Ns, wds, t, fmt="{:.3g}")
        else:
            for kk in g0: g0[kk].setdefault(k, np.nan)
            PD.heat(a, g0, k, 0.0, Ns, wds, t, fmt="{:.2g}", cmap="magma", sub="raw")
    fig.suptitle("NC1 robust measures (ln=0, test probe): compare with pinv-NC1 band"); fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(f"{fd}/nc1_robust_heatmaps.png", dpi=65); plt.close(fig)
    # example drops
    byname = {r["name"]: r for r in R}; pick = []
    seen = set()
    for x in rows:
        if x["src"] == "v2" and x["probe"] == "test" and (x["N"], x["wd"]) not in seen and x["seed"] == 0 and len(pick) < 6: pick.append(x); seen.add((x["N"], x["wd"]))
    if pick:
        fig, ax = plt.subplots(len(pick), 4, figsize=(16, 2.6 * len(pick)), squeeze=False)
        for i, x in enumerate(pick):
            S = byname[x["name"]]["test"]; st = S["steps"] + 1
            ax[i, 0].plot(st, S["nc1"], label="NC1 pinv"); ax[i, 0].plot(st, S["ratio_a"], label="tr Sw/tr Sb"); ax[i, 0].plot(st, S["nc1_ridge"][:, 1], label="ridge 1e-2"); ax[i, 0].set_yscale("log"); ax[i, 0].legend(fontsize=6)
            ax[i, 1].plot(st, S["spread"]); ax[i, 1].set_yscale("log"); ax[i, 2].plot(st, S["eig"]); ax[i, 2].set_yscale("log"); ax[i, 3].plot(st, S["cond"]); ax[i, 3].set_yscale("log")
            ax[i, 0].set_title(f"N={x['N']} wd={x['wd']:g} s{x['seed']}  verdict={x['verdict']}", fontsize=8); ax[i, 1].set_title("per-class tr(Sc)/tr(St)", fontsize=8); ax[i, 2].set_title("eig(Sb) (9)", fontsize=8); ax[i, 3].set_title("cond(Sb)", fontsize=8)
            for a in ax[i]: a.set_xscale("log"); a.axvline(x["drop_step"] + 1, c="r", ls=":", lw=1)
        fig.tight_layout(); fig.savefig(f"{fd}/drop_examples.png", dpi=60); plt.close(fig)


if __name__ == "__main__": main()
