"""Per-sample latent vector-field analysis (closure test) on the probe latent clouds. torch, batched over all snapshot intervals; GPU optional.

Usage:
  python -m latentdyn.vfield.core run --device cuda --runs 'mnist_probe_none_d16_s0_phase_*' [--probes test,train] [--limit N] [--shard i/n]
  python -m latentdyn.vfield.core aggregate
Data: runs_v2/runs/<id>/latents/step_*.npz (test probe, 1000 x 16) and latents_train/ (train probe, N<=1000 or 1000; may carry noisy_labels); model_final.pt
(only the FINAL probe head is stored; per-snapshot heads/logits are NOT available, so the logit frame (d) is not computed; the free-particle reference uses the
final head and is only meaningful in the last intervals).
Displacement d_i = align(z_{t+1})_i - z_{t,i} in frames: raw | procrustes (orthogonal + scale) | affine (full affine LS). Alignments are fit on the TRAIN half only
(5 stratified 50/50 splits); held-out R2 (centered on the held-out mean) of d_i, pooled over dims, per interval:
  M0 global translation (uncentered R2 reported separately) | M1 class drift | M2 class-wise affine (ridge, LOO-GCV lambda) | kNN in z (k=5,10,20; also within class)
  | MLP (z, onehot y, log t; one per run/split, all intervals) | nulls: shuffled neighbours, temporal shuffle (targets from another interval).
Outputs: OUT/runs/<id>_<probe>.npz, OUT/summary_runs.csv, figures.
"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, json, csv, time, collections
import numpy as np
import torch

RUNS = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield"
FRAMES = ["raw", "procrustes", "affine"]
MODELS = ["M0", "M1", "M2", "knn5", "knn10", "knn20", "knn5y", "knn10y", "knn20y", "null_knn_shuffled", "null_time_M2", "null_time_knn10y", "mlp"]
NSPLIT = 5
DT = torch.float64


def load_run(rd, probe):
    sub = "latents" if probe == "test" else "latents_train"
    fs = sorted(glob.glob(f"{rd}/{sub}/step_*.npz")); Z = []; steps = []; y = noisy = None
    for f in fs:
        with np.load(f) as z:
            Z.append(z["latents"].astype(np.float64)); steps.append(int(z["step"]))
            if y is None:
                y = z["labels"].astype(np.int64); noisy = z["noisy_labels"].astype(np.int64) if "noisy_labels" in z.files else None
    o = np.argsort(steps); return np.stack(Z)[o], np.array(steps)[o], y, noisy


def make_split(ycond, seed):
    rng = np.random.RandomState(1000 + seed); tr = []; te = []
    for c in np.unique(ycond):
        idx = np.where(ycond == c)[0]; p = rng.permutation(idx); m = len(p) // 2; tr += list(p[:m]); te += list(p[m:])
    return np.sort(tr), np.sort(te)


# ------------------------------------------------------------------ frames
def make_frames(Z, tr):
    """Z (T,n,d) torch; returns dict frame -> D (T-1,n,d); alignments fit on indices tr."""
    X, Y = Z[1:, tr], Z[:-1, tr]; K, m, d = X.shape; out = {"raw": Z[1:] - Z[:-1]}
    mu1, mu0 = X.mean(1), Y.mean(1); Xc, Yc = X - mu1[:, None], Y - mu0[:, None]
    U, S, Vh = torch.linalg.svd(Xc.transpose(1, 2) @ Yc); R = U @ Vh; s = S.sum(-1) / ((Xc ** 2).sum((1, 2)) + 1e-300)
    al = s[:, None, None] * ((Z[1:] - mu1[:, None]) @ R) + mu0[:, None]; out["procrustes"] = al - Z[:-1]
    one = torch.ones(K, m, 1, dtype=Z.dtype, device=Z.device); Xa = torch.cat([X, one], 2); G = Xa.transpose(1, 2) @ Xa
    G = G + 1e-9 * G.diagonal(dim1=1, dim2=2).mean(1)[:, None, None] * torch.eye(d + 1, dtype=Z.dtype, device=Z.device)
    beta = torch.linalg.solve(G, Xa.transpose(1, 2) @ Y); Za = torch.cat([Z[1:], torch.ones(K, Z.shape[1], 1, dtype=Z.dtype, device=Z.device)], 2); out["affine"] = Za @ beta - Z[:-1]
    return out


def r2(pred, Dte):
    sse = ((Dte - pred) ** 2).sum((1, 2)); sstc = ((Dte - Dte.mean(1, keepdim=True)) ** 2).sum((1, 2)) + 1e-300; sst0 = (Dte ** 2).sum((1, 2)) + 1e-300
    return 1 - sse / sstc, 1 - sse / sst0


# ------------------------------------------------------------------ models
def ridge_fit(Xs, D, lam_grid=(1e-3, 1e-2, 1e-1, 1.0, 10.0)):
    """Xs (K,m,d), D (K,m,d) -> (xbar, dbar, A) with LOO-GCV-chosen ridge on centered data; A (K,d,d)."""
    xb, db = Xs.mean(1, keepdim=True), D.mean(1, keepdim=True); Xc, Dc = Xs - xb, D - db
    U, S, Vh = torch.linalg.svd(Xc, full_matrices=False); UtD = U.transpose(1, 2) @ Dc; s2 = S ** 2; best = None
    for lg in lam_grid:
        lam = lg * s2.mean(1, keepdim=True) + 1e-300; f = s2 / (s2 + lam)                       # (K,r)
        H = (U ** 2 * f[:, None, :]).sum(2) + 1.0 / Xs.shape[1]                                  # leverage incl. intercept
        fit = U @ (f[:, :, None] * UtD) + db; err = (((D - fit) / (1 - H[:, :, None])) ** 2).sum((1, 2))
        A = Vh.transpose(1, 2) @ ((S / (s2 + lam))[:, :, None] * UtD)
        if best is None: best = (err, A)
        else:
            mask = err < best[0]; best = (torch.where(mask, err, best[0]), torch.where(mask[:, None, None], A, best[1]))
    return xb[:, 0], db[:, 0], best[1]


def class_models(Zs, D, Dnull, ycond, tr, te, classes):
    """M1, M2 (and temporal-shuffle M2 with targets Dnull). Returns predictions for test indices (K,n_te,d)."""
    K, n, d = D.shape; p1 = torch.zeros(K, len(te), d, dtype=D.dtype, device=D.device); p2 = p1.clone(); p2n = p1.clone()
    pos = {int(i): j for j, i in enumerate(te)}
    for c in classes:
        itr = tr[ycond[tr] == c]; ite = te[ycond[te] == c]
        if len(itr) < 3 or len(ite) == 0: continue
        sel = torch.tensor([pos[int(i)] for i in ite], device=D.device); Xtr, Xte = Zs[:, itr], Zs[:, ite]
        p1[:, sel] = D[:, itr].mean(1, keepdim=True)
        for Dsrc, outp in ((D, p2), (Dnull, p2n)):
            xb, db, A = ridge_fit(Xtr, Dsrc[:, itr]); outp[:, sel] = db[:, None] + (Xte - xb[:, None]) @ A
    return p1, p2, p2n


def knn_pred(Dtr, idx, k):
    K = Dtr.shape[0]; ar = torch.arange(K, device=Dtr.device)[:, None, None]; return Dtr[ar, idx[:, :, :k]].mean(2)


class MLP(torch.nn.Module):
    def __init__(self, din, dout, h=256):
        super().__init__(); self.net = torch.nn.Sequential(torch.nn.Linear(din, h), torch.nn.GELU(), torch.nn.Linear(h, h), torch.nn.GELU(), torch.nn.Linear(h, dout))

    def forward(self, x): return self.net(x)


def mlp_pred(Zs, D, ycond, steps, tr, te, classes, dev, epochs=800, seed=0):
    K, n, d = D.shape; C = len(classes); oh = torch.zeros(n, C, device=dev); cm = {int(c): i for i, c in enumerate(classes)}
    oh[torch.arange(n), torch.tensor([cm[int(v)] for v in ycond], device=dev)] = 1
    tt = torch.tensor(np.log1p(steps[:-1]), dtype=torch.float32, device=dev); tt = (tt - tt.mean()) / (tt.std() + 1e-9)
    X = torch.cat([Zs.float(), oh[None].expand(K, n, C), tt[:, None, None].expand(K, n, 1)], 2); s = (D[:, tr] ** 2).sum(2).mean(1).sqrt().float() + 1e-30
    Y = (D.float() / s[:, None, None]); torch.manual_seed(seed); net = MLP(X.shape[2], d).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-3); sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    Xtr, Ytr = X[:, tr].reshape(-1, X.shape[2]), Y[:, tr].reshape(-1, d)
    for ep in range(epochs):
        perm = torch.randperm(len(Xtr), device=dev)[:32768]; opt.zero_grad(); loss = ((net(Xtr[perm]) - Ytr[perm]) ** 2).mean(); loss.backward(); opt.step(); sched.step()
    with torch.no_grad(): return (net(X[:, te].reshape(-1, X.shape[2])).reshape(K, len(te), d) * s[:, None, None]).to(D.dtype)


def closure(Z, steps, ycond, classes, dev, mlp_splits):
    T, n, d = Z.shape; K = T - 1; M, F = len(MODELS), len(FRAMES); R2c = np.zeros((NSPLIT, M, F, K), np.float32); R20 = np.zeros((NSPLIT, M, F, K), np.float32)
    g = torch.Generator(device="cpu").manual_seed(0); pi = torch.randperm(K, generator=g)
    for j in range(K):                                       # derangement-like permutation (no fixed points)
        if pi[j] == j: pi[j], pi[(j + 1) % K] = pi[(j + 1) % K].clone(), pi[j].clone()
    pi = pi.to(dev)
    for sp in range(NSPLIT):
        tr, te = make_split(ycond, sp); trt, tet = torch.tensor(tr, device=dev), torch.tensor(te, device=dev)
        frames = make_frames(Z, trt); mu = Z[:-1, trt].mean(1, keepdim=True); sig = ((Z[:-1, trt] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z[:-1] - mu) / sig
        dist = torch.cdist(Zs[:, tet], Zs[:, trt]); idx = dist.topk(20, largest=False).indices
        same = torch.tensor(ycond[te][:, None] == ycond[tr][None, :], device=dev); distc = dist + (~same)[None] * 1e9; idxy = distc.topk(20, largest=False).indices
        ridx = torch.randint(0, len(tr), (K, len(te), 10), device=dev)
        for fi, fr in enumerate(FRAMES):
            D = frames[fr]; Dte = D[:, tet]; Dtr = D[:, trt]; Dp = D[pi]; preds = {}
            preds["M0"] = Dtr.mean(1, keepdim=True).expand_as(Dte)
            p1, p2, p2n = class_models(Zs, D, Dp, ycond, tr, te, classes); preds["M1"], preds["M2"], preds["null_time_M2"] = p1, p2, p2n
            for k in (5, 10, 20): preds[f"knn{k}"] = knn_pred(Dtr, idx, k); preds[f"knn{k}y"] = knn_pred(Dtr, idxy, k)
            preds["null_knn_shuffled"] = knn_pred(Dtr, ridx, 10); preds["null_time_knn10y"] = knn_pred(Dp[:, trt], idxy, 10)
            if sp < mlp_splits: preds["mlp"] = mlp_pred(Zs, D, ycond, steps, trt, tet, classes, dev, seed=sp)
            for mi, mname in enumerate(MODELS):
                if mname in preds:
                    a, b = r2(preds[mname], Dte); R2c[sp, mi, fi] = a.cpu().numpy(); R20[sp, mi, fi] = b.cpu().numpy()
                else: R2c[sp, mi, fi] = np.nan; R20[sp, mi, fi] = np.nan
    return R2c, R20


# ------------------------------------------------------------------ persistence, free particle, characterization
def persistence(Z, dev):
    all_idx = torch.arange(Z.shape[1], device=dev); fr = make_frames(Z, all_idx); out = np.zeros((len(FRAMES), Z.shape[0] - 2), np.float32); nul = out.copy()
    g = torch.Generator(device="cpu").manual_seed(1); perm = torch.randperm(Z.shape[1], generator=g).to(dev)
    for fi, f in enumerate(FRAMES):
        D = fr[f]; a, b = D[:-1], D[1:]; cos = (a * b).sum(2) / (a.norm(dim=2) * b.norm(dim=2) + 1e-300); out[fi] = cos.mean(1).cpu().numpy()
        cn = (a * b[:, perm]).sum(2) / (a.norm(dim=2) * b[:, perm].norm(dim=2) + 1e-300); nul[fi] = cn.mean(1).cpu().numpy()
    return out, nul


def free_particle(Z, ylab, rd, dev):
    sd = torch.load(f"{rd}/model_final.pt", map_location="cpu", weights_only=False); W = sd["probe.weight"].double().to(dev); b = sd["probe.bias"].double().to(dev)
    D = Z[1:] - Z[:-1]; yo = torch.nn.functional.one_hot(torch.tensor(ylab, device=dev), 10).double(); logits = Z[:-1] @ W.T + b; g = (torch.softmax(logits, -1) - yo) @ W   # grad_z CE
    cos = ((-g) * D).sum(2) / (g.norm(dim=2) * D.norm(dim=2) + 1e-300); c = ((-g) * D).sum((1, 2)) / ((g ** 2).sum((1, 2)) + 1e-300); sse = ((D - c[:, None, None] * (-g)) ** 2).sum((1, 2)); sst = ((D - D.mean(1, keepdim=True)) ** 2).sum((1, 2)) + 1e-300
    return cos.mean(1).cpu().numpy().astype(np.float32), (1 - sse / sst).cpu().numpy().astype(np.float32)


def characterize(Z, steps, ycond, classes, dev):
    """Class-wise affine field fit on ALL samples (no hold-out): fixed points, eigenvalues, divergence; frames raw and affine-aligned."""
    T, n, d = Z.shape; K = T - 1; res = {}; all_idx = torch.arange(n, device=dev); frames = make_frames(Z, all_idx)
    mu = Z[:-1].mean(1, keepdim=True); sig = ((Z[:-1] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z[:-1] - mu) / sig
    cm = torch.stack([Z[:-1][:, torch.tensor(np.where(ycond == c)[0], device=dev)].mean(1) for c in classes], 1)  # (K,C,d)
    cdist = torch.cdist(cm, cm); iu = torch.triu_indices(len(classes), len(classes), 1); mean_sep = cdist[:, iu[0], iu[1]].mean(1)
    for fr in ("raw", "affine"):
        D = frames[fr]; C = len(classes); fp = np.zeros((K, C), np.float32); maxre = fp.copy(); meanre = fp.copy(); maxim = fp.copy(); trA = fp.copy(); fpz = np.zeros((K, C, d), np.float32)
        for ci, c in enumerate(classes):
            ii = torch.tensor(np.where(ycond == c)[0], device=dev); xb, db, A = ridge_fit(Zs[:, ii], D[:, ii]); Araw = A / sig            # displacement per interval
            zb = Z[:-1][:, ii].mean(1); ev = torch.linalg.eigvals(Araw)
            inv = torch.linalg.pinv(Araw, rcond=1e-3); zstar = zb - (db[:, None, :] @ inv)[:, 0]
            fp[:, ci] = ((zstar - cm[:, ci]).norm(dim=1) / mean_sep).cpu().numpy(); maxre[:, ci] = ev.real.max(1).values.cpu().numpy(); meanre[:, ci] = ev.real.mean(1).cpu().numpy()
            maxim[:, ci] = ev.imag.abs().max(1).values.cpu().numpy(); trA[:, ci] = Araw.diagonal(dim1=1, dim2=2).sum(1).cpu().numpy(); fpz[:, ci] = zstar.cpu().numpy()
        res[fr] = dict(fp_dist_over_sep=fp, max_re=maxre, mean_re=meanre, max_abs_im=maxim, trace=trA)
    return res


def mislabeled(Z, ytrue, ygiven, dev):
    """cosine of d_i toward the given-label vs true-label class mean (clean points' means) for mislabeled train-probe points; frames raw/procrustes/affine (all-sample alignment)."""
    bad = ytrue != ygiven; T, n, d = Z.shape; K = T - 1; out = {}
    if bad.sum() < 3: return out
    clean = ~bad; mus = torch.stack([Z[:-1][:, torch.tensor(np.where(clean & (ytrue == c))[0], device=dev)].mean(1) for c in range(10)], 1)  # (K,10,d)
    all_idx = torch.arange(n, device=dev); fr = make_frames(Z, all_idx); ib = torch.tensor(np.where(bad)[0], device=dev); ic = torch.tensor(np.where(clean)[0], device=dev)
    yg = torch.tensor(ygiven, device=dev); yt = torch.tensor(ytrue, device=dev)
    for f in FRAMES:
        D = fr[f]; z = Z[:-1]
        def cosv(idx, lab):
            v = mus[:, lab[idx]] - z[:, idx]; return ((D[:, idx] * v).sum(2) / (D[:, idx].norm(dim=2) * v.norm(dim=2) + 1e-300))
        cg, ct, cc = cosv(ib, yg), cosv(ib, yt), cosv(ic, yt)
        out[f] = dict(cos_given=cg.mean(1).cpu().numpy(), cos_true=ct.mean(1).cpu().numpy(), frac_given_gt_true=(cg > ct).double().mean(1).cpu().numpy(), cos_clean=cc.mean(1).cpu().numpy())
    out["n_bad"] = int(bad.sum()); return out


def process(rd, probe, dev, mlp_splits):
    Z, steps, ytrue, noisy = load_run(rd, probe); ycond = noisy if (noisy is not None) else ytrue; classes = np.unique(ycond)
    Zt = torch.tensor(Z, dtype=DT, device=dev); t0 = time.time()
    R2c, R20 = closure(Zt, steps, ycond, classes, dev, mlp_splits); pers, pnull = persistence(Zt, dev); ch = characterize(Zt, steps, ycond, classes, dev)
    fpc, fpr = free_particle(Zt, ycond, rd, dev) if probe == "train" else free_particle(Zt, ycond, rd, dev)
    out = dict(steps=steps, models=np.array(MODELS), frames=np.array(FRAMES), R2c_mean=np.nanmean(R2c, 0), R2c_std=np.nanstd(R2c, 0), R20_mean=np.nanmean(R20, 0), persistence=pers, persistence_null=pnull, ufm_cos=fpc, ufm_r2=fpr,
               n=np.array(Z.shape[1]), n_classes=len(classes), min_class=np.array(min(int((ycond == c).sum()) for c in classes)))
    for fr, dct in ch.items():
        for k, v in dct.items(): out[f"char_{fr}_{k}"] = v
    if noisy is not None and probe == "train":
        ml = mislabeled(Zt, ytrue, noisy, dev); out["n_bad"] = np.array(ml.get("n_bad", 0))
        for f in FRAMES:
            for k, v in ml.get(f, {}).items(): out[f"mis_{f}_{k}"] = v
    out["seconds"] = np.array(time.time() - t0); return out


def main_run(a):
    dev = torch.device(a.device); torch.set_num_threads(8)
    assert dev.type == "cpu" or os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "0"), "pin CUDA_VISIBLE_DEVICES to an idle GPU (never 0)"
    os.makedirs(f"{a.out}/runs", exist_ok=True); os.makedirs(f"{a.out}/claims", exist_ok=True)
    dirs = sorted(d for g in a.runs.split(",") for d in glob.glob(f"{RUNS}/{g}") if os.path.isdir(d) and os.path.isfile(f"{d}/model_final.pt"))
    if a.shard:
        i, n = map(int, a.shard.split("/")); dirs = dirs[i::n]
    if a.limit: dirs = dirs[:a.limit]
    for d in dirs:
        rid = os.path.basename(d)
        for probe in a.probes.split(","):
            fo = f"{a.out}/runs/{rid}_{probe}.npz"
            if os.path.isfile(fo) and not a.force: continue
            if probe == "train" and not os.path.isdir(f"{d}/latents_train"): continue
            try: os.makedirs(f"{a.out}/claims/{rid}_{probe}")      # atomic claim: several processes (and GPUs) share one work list
            except FileExistsError: continue
            res = process(d, probe, dev, a.mlp_splits if probe == "test" else a.mlp_splits_train); np.savez_compressed(fo, **res); print(f"{rid} {probe} done {float(res['seconds']):.0f}s", flush=True)
            if dev.type == "cuda": torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd")
    r = sp.add_parser("run"); r.add_argument("--device", default="cpu"); r.add_argument("--runs", default="mnist_probe_none_d16_s*_phase_*"); r.add_argument("--probes", default="test,train"); r.add_argument("--out", default=OUT)
    r.add_argument("--limit", type=int, default=0); r.add_argument("--shard", default=""); r.add_argument("--force", action="store_true"); r.add_argument("--mlp_splits", type=int, default=2); r.add_argument("--mlp_splits_train", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "run": main_run(a)
    else:
        from latentdyn.vfield import aggregate
        sys.argv = sys.argv[:1] + sys.argv[2:]   # old `import vfield_aggregate` was a no-op (its main never ran); forward --out etc.
        aggregate.main()


if __name__ == "__main__":
    main()
