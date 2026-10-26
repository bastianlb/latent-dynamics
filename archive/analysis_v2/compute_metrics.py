#!/usr/bin/env python
"""Step 1: per-snapshot metrics, cached per run (resumable; finished caches are skipped).
Usage: python compute_metrics.py [--workers 6] [--runs id ...] [--force] [--prefix mnist cifar10]"""
import common2 as C
import argparse, glob, os, time, warnings
from multiprocessing import Pool
import numpy as np
from latentdyn import metrics as M

METRICS_CHEAP = ["fisher", "fisher_noisy", "norm", "probe_acc", "probe_acc_noisy", "eff_rank_pr", "eff_rank_ent"]
METRICS_TOPO = ["h0_total", "h1_total", "h1_top5", "h1_top10", "h1_entropy", "mag_t1", "pmag_fixed"]
SCAL = ["train_loss", "test_loss", "train_acc", "test_acc", "epoch", "lr"]


def eff_ranks(x):
    xc = x - x.mean(0, keepdims=True)
    lam = np.linalg.svd(xc, compute_uv=False) ** 2
    s = lam.sum()
    if s <= 0:
        return np.nan, np.nan
    p = lam / s
    pr = s ** 2 / np.sum(lam ** 2)
    ent = np.exp(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return float(pr), float(ent)


def topo_mag(x):
    from scipy.spatial.distance import pdist, squareform
    from ripser import ripser
    z = M.normalize_latents(x, "mean_pairwise")
    Dm = squareform(pdist(z))
    dg = ripser(Dm, maxdim=1, distance_matrix=True)["dgms"]
    h0 = dg[0]; h0 = h0[np.isfinite(h0[:, 1])]
    l0 = h0[:, 1] - h0[:, 0]
    l1 = np.sort(dg[1][:, 1] - dg[1][:, 0])[::-1] if len(dg[1]) else np.zeros(0)
    out = dict(h0_total=float(l0.sum()), h1_total=float(l1.sum()), h1_top5=float(l1[:5].sum()),
               h1_top10=float(l1[:10].sum()), h1_entropy=M.persistence_entropy(dg[1]))
    mags = np.empty(len(C.MAG_GRID))
    for i, t in enumerate(C.MAG_GRID):
        mags[i] = M._solve_magnitude_weights(np.exp(-t * Dm)).sum()
    ts = np.linspace(*C.PMAG_WIN, 12)
    mw = np.array([M._solve_magnitude_weights(np.exp(-t * Dm)).sum() for t in ts])
    out["pmag_fixed"] = float(np.trapezoid(mw, ts) / (ts[-1] - ts[0]))
    out["mag_t1"] = float(mags[6])
    return out, mags


def process_run(args):
    rid, force = args
    out = f"{C.CACHE}/{rid}.npz"
    if os.path.exists(out) and not force:
        return f"skip {rid}"
    warnings.filterwarnings("ignore")
    t0 = time.time()
    files = {ps: sorted(glob.glob(f"{C.RUNS}/{rid}/{C.PROBE_DIR[ps]}/step_*.npz")) for ps in C.PROBES}
    n = len(files["test"])
    assert n == len(files["train"]) and n > 0, rid
    res = {"step": np.zeros(n, int)}
    for k in SCAL:
        res["run/" + k] = np.full(n, np.nan)
    for ps in C.PROBES:
        for m in METRICS_CHEAP + METRICS_TOPO:
            res[f"{ps}/{m}"] = np.full(n, np.nan)
        res[f"{ps}/mag"] = np.full((n, len(C.MAG_GRID)), np.nan)
    steps = np.array([int(np.load(f)["step"]) for f in files["test"]])
    topo_idx = set(np.where(steps <= 10)[0]) | set(range(0, n, 2)) | {n - 1}
    for i in range(n):
        for ps in C.PROBES:
            s = np.load(files[ps][i])
            x, y = s["latents"].astype(np.float64), s["labels"]
            if ps == "test":
                res["step"][i] = int(s["step"])
                for k in SCAL:
                    if k in s.files:
                        res["run/" + k][i] = float(s[k])
            r = lambda m, v: res.__setitem__(f"{ps}/{m}", res[f"{ps}/{m}"]) or res[f"{ps}/{m}"].__setitem__(i, v)
            r("fisher", M.fisher_separation_ratio(x, y))
            r("norm", float(np.linalg.norm(x, axis=1).mean()))
            r("probe_acc", M.offline_linear_separability(x, y))
            pr, en = eff_ranks(x); r("eff_rank_pr", pr); r("eff_rank_ent", en)
            if "noisy_labels" in s.files and not np.array_equal(s["noisy_labels"], y):
                yn = s["noisy_labels"]
                r("fisher_noisy", M.fisher_separation_ratio(x, yn))
                r("probe_acc_noisy", M.offline_linear_separability(x, yn))
            if i in topo_idx:
                tm, mags = topo_mag(x)
                for k, v in tm.items():
                    r(k, v)
                res[f"{ps}/mag"][i] = mags
    res["topo_mask"] = np.array([i in topo_idx for i in range(n)])
    res["mag_grid"] = C.MAG_GRID
    np.savez(out + ".tmp.npz", **res)
    os.replace(out + ".tmp.npz", out)
    return f"done {rid} {n} snaps {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--runs", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--reverse", action="store_true")
    a = ap.parse_args()
    os.makedirs(C.CACHE, exist_ok=True)
    rids = a.runs or C.finished_runs(runset='all')
    # CIFAR first (longest I/O), interleaved is fine
    rids = sorted(rids, key=lambda r: (not r.startswith("cifar"), r))
    todo = [(r, a.force) for r in (rids[::-1] if a.reverse else rids)]
    print(f"{len(todo)} runs, {a.workers} workers", flush=True)
    with Pool(a.workers) as pool:
        for msg in pool.imap_unordered(process_run, todo):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
