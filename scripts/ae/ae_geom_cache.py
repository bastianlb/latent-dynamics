"""Per-snapshot latent-geometry cache (CPU). One <rid>.npz per run with keys step, run/<scalar>, test/<m>, train/<m>, cross/<m>, each (T,).
Usage: python ae_geom_cache.py [--workers 6] [--runs rid ...] [--out DIR]   (AE and classifier runs auto-detected by rid; skips existing outputs)"""
from latentdyn._threads import cap_threads; cap_threads(4)
import argparse, glob, json, os, time, warnings
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from latentdyn.ae import analysis as A, geom_metrics as G

B = "/vol/bitbucket/lbastian/latent-dynamics"
AE_RUNS = f"{B}/ae_dynamics/runs"; CLF_RUNS = f"{B}/runs_v2/runs"; OUT = f"{B}/ae_dynamics/analysis/geom"; CACHE = f"{B}/ae_dynamics/analysis/_cache"
SCALARS = ["epoch", "recon_train", "recon_test", "recon_probe_test", "recon_probe_train", "train_loss", "test_loss", "train_acc", "test_acc"]

def all_rids():
    ae = [f"{d}_ae_mlp_d16_s{s}_n{n}_wd{w}" for d in ("mnist", "fashion_mnist") for s in range(3) for n in ("250", "1000", "4000", "full") for w in ("0", "1e-4")]
    ae += [f"{d}_ae_linear_d16_s{s}_n{n}_wd0" for d in ("mnist", "fashion_mnist") for s in range(3) for n in ("250", "1000", "4000", "full")]
    return ae + [f"{d}_probe_none_d16_s{s}_phase_n1000_wd0" for d in ("mnist", "fashion_mnist") for s in range(3)]

def rundir(rid): return f"{AE_RUNS}/{rid}" if "_ae_" in rid else f"{CLF_RUNS}/{rid}"

def check(rid):
    rd = rundir(rid); return all(len(glob.glob(f"{rd}/{d}/step_*.npz")) == 120 for d in ("latents", "latents_train")) and os.path.isfile(f"{rd}/model_final.pt")

def _load(f):
    with np.load(f) as z: return {k: z[k] for k in z.files}

def work(rid):
    out = f"{OUT}/{rid}.npz"
    if os.path.exists(out): return f"skip {rid}"
    try:
        warnings.filterwarnings("ignore"); t0 = time.time(); rd = rundir(rid); ds = json.load(open(f"{rd}/config.json"))["dataset"]
        ft = sorted(glob.glob(f"{rd}/latents/step_*.npz")); fr = sorted(glob.glob(f"{rd}/latents_train/step_*.npz")); T = len(ft)
        with ThreadPoolExecutor(24) as ex: snT = list(ex.map(_load, ft)); snR = list(ex.map(_load, fr))  # NFS is latency-bound: read in parallel
        Xte = A.images(ds, False)[0][np.load(f"{rd}/probe_index.npz")["indices"]].astype(np.float64)
        Xtr = A.images(ds, True)[0][np.load(f"{rd}/probe_train_index.npz")["indices"]].astype(np.float64)
        assign = A.group_model(ds, CACHE); pr = {"test": dict(X=Xte, g=assign(Xte)), "train": dict(X=Xtr, g=assign(Xtr))}
        for p in pr: pr[p]["nnp"] = G.knn_sorted(pr[p]["X"], G.kk(len(pr[p]["X"])))[0]
        res = {}; first = {"test": None, "train": None}; prev = {"test": None, "train": None}
        def put(key, i, v):
            if key not in res: res[key] = np.full(T, np.nan)
            res[key][i] = v
        for i in range(T):
            s = {"test": snT[i], "train": snR[i]}
            put("step", i, float(s["test"]["step"]))
            for k in SCALARS: put(f"run/{k}", i, float(s["test"][k]) if k in s["test"] else np.nan)
            for p in ("test", "train"):
                z = s[p]; e = z["recon_err"].astype(np.float64) if "recon_err" in z else None
                m = G.per_probe(z["latents"], z["labels"], pr[p]["g"], pr[p]["nnp"], prev[p], first[p], e)
                prev[p] = m.pop("_nn"); first[p] = prev[p] if i == 0 else first[p]
                for k, v in m.items(): put(f"{p}/{k}", i, v)
            et = s["test"]["recon_err"].astype(np.float64) if "recon_err" in s["test"] else None
            c = G.cross(s["test"]["latents"], s["train"]["latents"], et)
            for k, v in c.items(): put(f"cross/{k}", i, v)
            put("cross/pr_dim_diff", i, res["train/pr_dim"][i] - res["test/pr_dim"][i])
        order = np.argsort(res["step"]); res = {k: v[order] for k, v in res.items()}  # files are step-sorted already; guards nothing else
        np.savez(out + ".tmp.npz", **res); os.replace(out + ".tmp.npz", out)
        return f"done {rid} {T} snaps {time.time()-t0:.0f}s"
    except Exception as ex:
        import traceback; return f"FAIL {rid}: {ex!r} {traceback.format_exc()[-400:]}"

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); ap.add_argument("--runs", nargs="*"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); OUT = a.out; os.makedirs(OUT, exist_ok=True)
    rids = []
    for r in (a.runs or all_rids()):
        if check(r): rids.append(r)
        else: print(f"SKIP-incomplete {r}", flush=True)
    with Pool(min(a.workers, 6)) as pool:
        for m in pool.imap_unordered(work, rids): print(m, flush=True)
