"""AE persistent-homology / magnitude cache with the same layout as analysis_v2 classifier cache (CPU only).
Usage: python ae_topo_cache.py [--workers 6] [--train] [--runs ...]"""
from latentdyn._threads import cap_threads; cap_threads(4)
import argparse, glob, os, sys, time, warnings
from multiprocessing import Pool
import numpy as np
from latentdyn import metrics as M
MAG_GRID = np.geomspace(0.1, 100.0, 19); PMAG_WIN = (0.5, 10.0)
RUNS = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/runs"; OUT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis/topo"
TM = ["h0_total", "h1_total", "h1_top5", "h1_top10", "h1_entropy", "mag_t1", "pmag_fixed"]

def topo_mag(x):  # verbatim from archive/analysis_v2/compute_metrics.py (C.MAG_GRID -> MAG_GRID)
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
    mags = np.empty(len(MAG_GRID))
    for i, t in enumerate(MAG_GRID):
        mags[i] = M._solve_magnitude_weights(np.exp(-t * Dm)).sum()
    ts = np.linspace(*PMAG_WIN, 12)
    mw = np.array([M._solve_magnitude_weights(np.exp(-t * Dm)).sum() for t in ts])
    out["pmag_fixed"] = float(np.trapezoid(mw, ts) / (ts[-1] - ts[0]))
    out["mag_t1"] = float(mags[6])
    return out, mags

def work(a):
    rid, train = a
    out = f"{OUT}/{rid}{'__train' if train else ''}.npz"
    if os.path.exists(out): return f"skip {rid}"
    warnings.filterwarnings("ignore"); t0 = time.time()
    probes = {"train": "latents_train"} if train else {"test": "latents"}
    files = {p: sorted(glob.glob(f"{RUNS}/{rid}/{d}/step_*.npz")) for p, d in probes.items()}
    files["test"] = sorted(glob.glob(f"{RUNS}/{rid}/latents/step_*.npz")); n = len(files["test"])
    steps = np.array([int(np.load(f)["step"]) for f in files["test"]])
    topo_idx = set(np.where(steps <= 10)[0]) | set(range(0, n, 2)) | {n - 1}
    res = {"step": steps, "run/epoch": np.full(n, np.nan), "run/recon_train": np.full(n, np.nan), "run/recon_test": np.full(n, np.nan)}
    for p in probes:
        for m in TM: res[f"{p}/{m}"] = np.full(n, np.nan)
        res[f"{p}/mag"] = np.full((n, len(MAG_GRID)), np.nan)
    for i in range(n):
        s = np.load(files["test"][i])
        res["run/epoch"][i] = float(s["epoch"])
        for k in ("recon_train", "recon_test"):
            if k in s.files: res["run/" + k][i] = float(s[k])
        if i not in topo_idx: continue
        for p in probes:
            x = np.load(files[p][i])["latents"].astype(np.float64)
            tm, mags = topo_mag(x)
            for k, v in tm.items(): res[f"{p}/{k}"][i] = v
            res[f"{p}/mag"][i] = mags
    res["topo_mask"] = np.array([i in topo_idx for i in range(n)]); res["mag_grid"] = MAG_GRID
    np.savez(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out)
    return f"done {rid} {n} snaps {time.time()-t0:.0f}s"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--train", action="store_true", help="train probe only -> <rid>__train.npz"); ap.add_argument("--runs", nargs="*")
    a = ap.parse_args(); os.makedirs(OUT, exist_ok=True)
    rids = a.runs or [f"{d}_ae_{arch}_d16_s{s}_n{n}_wd0" for n in ("1000", "full") for arch in ("mlp", "linear") for d in ("mnist", "fashion_mnist") for s in range(3)]
    with Pool(a.workers) as pool:
        for m in pool.imap_unordered(work, [(r, a.train) for r in rids]): print(m, flush=True)
