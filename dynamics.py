"""Dynamical-systems analysis of latent-kernel trajectories. CPU only.
Usage: python dynamics.py --runs 'mnist_probe_none_d16_s0' --workers 4 --out DIR
"""
import os, sys
_w = 1
for i, a in enumerate(sys.argv):
    if a == "--workers" and i + 1 < len(sys.argv):
        try: _w = int(sys.argv[i + 1])
        except ValueError: pass
_t = "8" if _w <= 1 else "2"
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, _t)
import argparse, glob, json, re, csv, time, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import metrics as M

RUNS = "/vol/bitbucket/lbastian/latent-dynamics/runs"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics"
ORDER = ["fisher", "nc1", "effrank", "etf", "ncm_acc", "cka_label"]


def load(run_dir, sub="latents"):
    fs = sorted(glob.glob(f"{run_dir}/{sub}/step_*.npz"))
    steps, Zs, ep, ta = [], [], [], []
    labels = None
    for f in fs:
        with np.load(f) as z:  # only optional keys are tolerated, not required
            steps.append(int(z["step"])); Zs.append(z["latents"].astype(np.float64))
            ep.append(float(z["epoch"])); ta.append(float(z["test_acc"]))
            if labels is None: labels = z["labels"]
    o = np.argsort(steps)
    return (np.array(steps)[o], [Zs[i] for i in o], np.array(ep)[o], np.array(ta)[o], labels)


def order_params(Z, y, classes):
    C = len(classes)
    mu = np.array([Z[y == c].mean(0) for c in classes])
    g = Z.mean(0)
    fisher = M.fisher_separation_ratio(Z, y)
    n = len(Z)
    Sw = sum(((Z[y == c] - m).T @ (Z[y == c] - m)) for c, m in zip(classes, mu)) / n
    Mc = mu - g
    pc = np.array([(y == c).mean() for c in classes])
    Sb = (Mc * pc[:, None]).T @ Mc
    nc1 = np.trace(Sw @ np.linalg.pinv(Sb, rcond=1e-6, hermitian=True)) / C
    s = np.linalg.svd(Z - g, compute_uv=False) ** 2
    effrank = s.sum() ** 2 / (s ** 2).sum()
    Mn = Mc / (np.linalg.norm(Mc, axis=1, keepdims=True) + 1e-12)
    G = Mn @ Mn.T
    Gt = (C * np.eye(C) - 1) / (C - 1)
    etf = np.linalg.norm(G - Gt) / np.linalg.norm(Gt)
    d2 = ((Z[:, None, :] - mu[None]) ** 2).sum(-1)
    ncm = float((classes[d2.argmin(1)] == y).mean())
    sv = np.linalg.svd(Mc, compute_uv=False)
    sv = np.pad(sv, (0, C - len(sv)))
    return [fisher, nc1, effrank, etf, ncm], sv / (sv.sum() + 1e-30)


def binseg(X, max_cp=5, min_size=3, frac=0.03):
    """Binary segmentation, l2 cost, on (T,d) standardized series. Split accepted if cost gain > frac*T*d."""
    T, d = X.shape
    pen = frac * T * d
    def cost(a, b):
        s = X[a:b]; return ((s - s.mean(0)) ** 2).sum()
    cps = []; segs = [(0, T)]
    for _ in range(max_cp):
        best = (0, None, None)
        for (a, b) in segs:
            if b - a < 2 * min_size: continue
            c0 = cost(a, b)
            for k in range(a + min_size, b - min_size + 1):
                g = c0 - cost(a, k) - cost(k, b)
                if g > best[0]: best = (g, k, (a, b))
        if best[1] is None or best[0] < pen: break
        cps.append(best[1]); segs.remove(best[2]); segs += [(best[2][0], best[1]), (best[1], best[2][1])]
    return sorted(cps)


def pca_frac(Gram):
    T = len(Gram); H = np.eye(T) - 1.0 / T
    ev = np.sort(np.linalg.eigvalsh(H @ Gram @ H))[::-1]
    ev = np.clip(ev, 0, None); return ev[:2].sum() / ev.sum(), ev[0] / ev.sum()


def analyze(run_dir, sub="latents", rng_seed=0, data=None):
    """data: optional pre-loaded (steps, Zs, ep, ta, labels) tuple (e.g. defensive loader for in-progress runs)."""
    steps, Zs, ep, ta, y = data if data is not None else load(run_dir, sub)
    T = len(steps); classes = np.unique(y); C = len(classes)
    Zs = [Z - Z.mean(0) for Z in Zs]
    d = Zs[0].shape[1]
    # Gram of all Z^T Z_j via one big matmul: <K_i,K_j>_F = ||Zi^T Zj||_F^2
    A = np.concatenate(Zs, axis=1)
    big = A.T @ A
    ip = np.zeros((T, T))
    for i in range(T):
        for j in range(T):
            ip[i, j] = (big[i * d:(i + 1) * d, j * d:(j + 1) * d] ** 2).sum()
    nrm = np.sqrt(np.diag(ip)); cosm = ip / np.outer(nrm, nrm)
    D = np.sqrt(np.clip(2 - 2 * cosm, 0, None))  # distance between Khat's
    Y = (y[:, None] == classes[None]).astype(float); Y -= Y.mean(0)
    kyn = np.linalg.norm(Y.T @ Y)
    cka = np.array([((Z.T @ Y) ** 2).sum() / (np.linalg.norm(Z.T @ Z) * kyn) for Z in Zs])
    dstep = np.diff(steps).astype(float); dlog = np.diff(np.log1p(steps))
    dk = np.array([D[k, k + 1] for k in range(T - 1)])
    vel_step = dk / dstep; vel_log = dk / dlog
    d_init = D[0]; d_final = D[:, -1]
    ops = []; svs = []
    for Z in Zs:
        o, s = order_params(Z, y, classes); ops.append(o); svs.append(s)
    ops = np.array(ops); ops = np.column_stack([ops, cka]); svs = np.array(svs)
    x = np.log1p(steps)
    deriv = np.gradient(ops, x, axis=0)
    # SV stage detection: fraction of total variation of each SV share within a single interval
    sv_jump = []
    for k in range(min(5, C)):
        dd = np.abs(np.diff(svs[:, k])); sv_jump.append([dd.max() / (dd.sum() + 1e-30), steps[1:][dd.argmax()]])
    sv_jump = np.array(sv_jump)
    # change points on log-uniform resampled standardized series (order params + log velocity proxy)
    feat = np.column_stack([np.log(np.clip(ops[:, 0], 1e-9, None)), np.log(np.clip(ops[:, 1], 1e-9, None)),
                            np.log(ops[:, 2]), ops[:, 3], ops[:, 4], ops[:, 5], d_init])
    # native snapshot samples (no interpolation: linear-schedule gaps would create fake change points)
    U = (feat - feat.mean(0)) / (feat.std(0) + 1e-12)
    cp_idx = binseg(U)
    cp_steps = steps[cp_idx].astype(int)  # first snapshot of the new segment
    # velocity peak (log-time), at interval midpoint
    mid = np.sqrt((steps[:-1] + 1.0) * (steps[1:] + 1.0)) - 1
    vpk = float(mid[vel_log.argmax()])
    # trajectory PCA vs random walk null
    pc2, pc1 = pca_frac(cosm)
    rng = np.random.RandomState(rng_seed); nulls = []
    for _ in range(20):
        inc = rng.randn(T - 1, 3000); inc *= (dk / np.linalg.norm(inc, axis=1))[:, None]
        P = np.vstack([np.zeros(3000), np.cumsum(inc, 0)])
        nulls.append(pca_frac(P @ P.T)[0])
    null2 = float(np.mean(nulls))
    return dict(steps=steps, epoch=ep, test_acc=ta, vel_step=vel_step, vel_log=vel_log, d_init=d_init, d_final=d_final,
                cka_label=cka, order=ops, order_names=np.array(ORDER), deriv=deriv, sv_share=svs, sv_jump=sv_jump,
                cp_steps=cp_steps, vel_peak_step=vpk, pca_pc12=pc2, pca_pc1=pc1, null_pc12=null2,
                pca_ratio=pc2 / null2, step_mid=mid)


def plot_run(r, title, path):
    s = r["steps"] + 1; sm = r["step_mid"] + 1
    fig, ax = plt.subplots(2, 3, figsize=(16, 8)); ax = ax.ravel()
    ax[0].plot(sm, r["vel_log"], "o-", label="per dlog(step+1)"); ax[0].set_ylabel("kernel velocity /dlog"); 
    a2 = ax[0].twinx(); a2.plot(sm, r["vel_step"], "s--", c="C1", ms=3); a2.set_yscale("log"); a2.set_ylabel("per step", color="C1")
    ax[1].plot(s, r["d_init"], label="d(init)"); ax[1].plot(s, r["d_final"], label="d(final)"); ax[1].plot(s, r["cka_label"], label="CKA(Y)"); ax[1].plot(s, r["test_acc"], label="test acc"); ax[1].legend(fontsize=7)
    o = r["order"]
    ax[2].plot(s, o[:, 4], label="NCM acc"); ax[2].plot(s, o[:, 3], label="ETF dist"); ax[2].plot(s, o[:, 2] / o[:, 2].max(), label="effrank/max"); ax[2].plot(s, o[:, 0] / o[:, 0].max(), label="Fisher/max"); ax[2].plot(s, np.clip(o[:, 1], 0, None) / np.clip(o[:, 1], 0, None).max(), label="NC1/max"); ax[2].legend(fontsize=7)
    for k, n in enumerate(ORDER): ax[3].plot(s, r["deriv"][:, k] / (np.abs(r["deriv"][:, k]).max() + 1e-30), label=n)
    ax[3].set_ylabel("d/dlog t (each normalized)"); ax[3].legend(fontsize=7)
    for k in range(r["sv_share"].shape[1]): ax[4].plot(s, r["sv_share"][:, k], label=f"s{k+1}" if k < 5 else None)
    ax[4].set_ylabel("class-mean SV share"); ax[4].legend(fontsize=7)
    ax[5].plot(s, r["d_init"], c="gray"); ax[5].set_title(f"PCA PC1-2 {r['pca_pc12']:.2f} vs RW {r['null_pc12']:.2f} (x{r['pca_ratio']:.2f})", fontsize=9)
    ax[5].set_ylabel("d(init)")
    for a in ax:
        a.set_xscale("log"); a.set_xlabel("step+1")
        for c in r["cp_steps"]: a.axvline(c + 1, c="r", ls=":", lw=1)
    fig.suptitle(title + f"   change points: {list(r['cp_steps'])}"); fig.tight_layout(); fig.savefig(path, dpi=80); plt.close(fig)


def do_run(args):
    rd, out = args
    rid = os.path.basename(rd); t0 = time.time()
    try:
        os.makedirs(f"{out}/{rid}", exist_ok=True)
        res = {}
        for sub, suf in (("latents", ""), ("latents_train", "_train")):
            if not os.path.isdir(f"{rd}/{sub}"): continue
            r = analyze(rd, sub)
            np.savez_compressed(f"{out}/{rid}/dynamics{suf}.npz", **{k: (np.asarray(v, dtype=np.float32) if np.asarray(v).dtype.kind == "f" else np.asarray(v)) for k, v in r.items()})
            plot_run(r, rid + suf, f"{out}/{rid}/dynamics{suf}.png"); res[sub] = r
        return rid, "ok", time.time() - t0
    except Exception as e:
        import traceback; return rid, traceback.format_exc(), 0


def cfg_key(rid):
    m = re.match(r"(.*)_s(\d+)$", rid); return (m.group(1), int(m.group(2))) if m else (rid, 0)


def compare(out, rids):
    os.makedirs(f"{out}/_compare", exist_ok=True)
    groups = collections.defaultdict(list)
    for rid in rids:
        if os.path.exists(f"{out}/{rid}/dynamics.npz"): groups[cfg_key(rid)[0]].append(rid)
    cp_rows = []; cons = []
    for rid in rids:
        p = f"{out}/{rid}/dynamics.npz"
        if os.path.exists(p):
            z = np.load(p); cp_rows.append([rid, " ".join(map(str, z["cp_steps"])), len(z["cp_steps"]), float(z["vel_peak_step"]), float(z["pca_pc12"]), float(z["null_pc12"]), float(z["pca_ratio"])])
    with open(f"{out}/changepoints.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run_id", "cp_steps", "n_cp", "vel_peak_step", "pca_pc12", "null_pc12", "pca_ratio"]); w.writerows(cp_rows)
    for cfg, rs in groups.items():
        if len(rs) < 2: continue
        Z = [np.load(f"{out}/{r}/dynamics.npz") for r in rs]
        if len({tuple(z["steps"]) for z in Z}) != 1: continue
        s = Z[0]["steps"] + 1
        O = np.stack([z["order"] for z in Z])  # seeds,T,6
        var = O.var(0, ddof=1)  # T,6
        # relative variance: scale by squared range of seed-mean trajectory
        rng2 = (O.mean(0).max(0) - O.mean(0).min(0)) ** 2 + 1e-30
        rvar = var / rng2
        vpk = [float(z["vel_peak_step"]) for z in Z]
        cps = [list(z["cp_steps"]) for z in Z]
        pk = [int(Z[0]["steps"][rvar[:, k].argmax()]) for k in range(6)]
        # cp match: mean over seed pairs of mean nearest-cp |dlog| 
        dl = []
        for i in range(len(Z)):
            for j in range(i + 1, len(Z)):
                if cps[i] and cps[j]:
                    dl.append(np.mean([min(abs(np.log1p(a) - np.log1p(b)) for b in cps[j]) for a in cps[i]]))
        vlog = np.stack([z["vel_log"] for z in Z])
        cons.append([cfg, len(Z), " | ".join(" ".join(map(str, c)) for c in cps), " ".join(f"{v:.0f}" for v in vpk),
                     float(np.std(np.log1p(vpk))), float(np.mean(dl)) if dl else np.nan,
                     " ".join(f"{n}:{p}" for n, p in zip(ORDER, pk)),
                     float(np.corrcoef(vlog)[np.triu_indices(len(Z), 1)].mean())])
        fig, ax = plt.subplots(2, 4, figsize=(18, 8)); ax = ax.ravel()
        for z, r in zip(Z, rs):
            sd = int(cfg_key(r)[1]); sm = z["step_mid"] + 1
            ax[0].plot(sm, z["vel_log"], label=f"s{sd}")
            for k in range(6): ax[1 + k].plot(s, z["order"][:, k], label=f"s{sd}")
            for c in z["cp_steps"]: ax[0].axvline(c + 1, ls=":", lw=.8, c="gray")
        ax[0].set_title("velocity/dlog"); ax[0].legend(fontsize=7)
        for k in range(6): ax[1 + k].set_title(ORDER[k])
        ax[7].plot(s, rvar); ax[7].legend(ORDER, fontsize=6); ax[7].set_title("between-seed var / range^2")
        for a in ax:
            a.set_xscale("log")
        ax[2].set_yscale("log") if ax[2].get_ylim()[0] > 0 else None
        fig.suptitle(cfg); fig.tight_layout(); fig.savefig(f"{out}/_compare/{cfg}.png", dpi=70); plt.close(fig)
    with open(f"{out}/seed_consistency.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "n_seeds", "cp_steps_per_seed", "vel_peak_steps", "std_log_vel_peak", "mean_cp_log_mismatch", "between_seed_var_peak_step", "mean_pairwise_corr_vel"]); w.writerows(cons)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*"); ap.add_argument("--workers", type=int, default=1); ap.add_argument("--out", default=OUT)
    ap.add_argument("--no_compare", action="store_true")
    a = ap.parse_args()
    dirs = [d for d in sorted(glob.glob(f"{RUNS}/{a.runs}")) if os.path.isfile(f"{d}/config.json") and not os.path.basename(d).startswith("_") and "smoke" not in d]
    os.makedirs(a.out, exist_ok=True); t0 = time.time()
    jobs = [(d, a.out) for d in dirs]
    if a.workers > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(do_run, jobs, chunksize=1)
    else: res = [do_run(j) for j in jobs]
    for rid, st, t in res:
        if st != "ok": print("FAIL", rid, st)
    print(f"{len(res)} runs, {sum(r[1]=='ok' for r in res)} ok, {time.time()-t0:.0f}s")
    if not a.no_compare: compare(a.out, [os.path.basename(d) for d in dirs])

if __name__ == "__main__": main()
