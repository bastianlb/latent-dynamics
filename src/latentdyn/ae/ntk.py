"""Empirical-NTK / step-decomposition analysis of MLP (and linear) autoencoders around memorization onset.  Phase 1 (GPU or CPU).
Usage: CUDA_VISIBLE_DEVICES=1 python ae_ntk.py --runs 'GLOB' [--epochs 10,30,60,120,250,1000] [--nprobe 200] [--K 64]
Reads ckpts (model+Adam) written by run_ae_ntk_ckpt.py under ae_dynamics/ntk/runs/<rid>/dense/ckpt at steps s and s+1.
For each ckpt pair (theta_s, theta_{s+1}) and a probe set of nprobe TRAIN + nprobe TEST points (n=2*nprobe) saves ntk/ana/<rid>/step_*.npz:
 * Per-sample loss L_i = mean_pix (recon_i - x_i)^2 (the training loss), z_i, labels, k-means(10) input clusters, indices.
 * Kernels, all on parameter blocks enc / dec, in Adam-preconditioned form J D J^T (D = 1/(sqrt(v/bc2)+eps), v from ckpt s+1) and raw (D=1):
     T  = pixel-output trace kernel Tr(J_i D J_j^T), 784 outputs, unbiased Hutchinson estimate with K Gaussian output probes
          (a random 784->K sketch of the output; unbiased, relative error of a diagonal entry ~ sqrt(2/K); validated vs exact in selftest)
     G  = loss-gradient kernel <grad L_i, D grad L_j>  (EXACT; what governs how a step on j changes L_i)
     Tz = latent-output kernel Tr(J^z_i D J^z_j^T) (16 outputs, EXACT, encoder block only)
 * One-step first-order decomposition of dL_i = L_i(s+1) - L_i(s) into  H (Adam momentum history), Nown (own new gradient, only if i in the batch),
   Noth (other batch members), W (weight-decay), each split into enc / dec parameter blocks; plus the actual dL_i.
"""
import os
from latentdyn._threads import cap_threads
cap_threads(4)
import argparse, glob, json, math, time
import numpy as np
import torch
from torch.func import functional_call, vmap, grad
torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
from latentdyn.ae import analysis as A
from latentdyn.model import LatentProbeModel

ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk"
DEV = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
EPOCHS_ALL = {250: [1, 3, 5, 10, 20, 30, 45, 60, 90, 120, 170, 250, 500, 1000, 2000],
              1000: [1, 3, 5, 10, 20, 30, 45, 60, 90, 120, 170, 250, 500, 1000, 2000],
              "full": [1, 2, 5, 10, 20, 30, 42]}


def epoch_step(N, e):
    n = 60000 if N == "full" else N
    return min(19999, max(1, round(e * n / 128)))


def parse_rid(rid):
    p = rid.split("_"); kind = p[2]; s = int(p[4][1:]); N = p[5][1:]; wd = rid.split("_wd")[1]
    return kind, s, (N if N == "full" else int(N)), wd


def build_model(kind):
    return LatentProbeModel(16, encoder_arch="mlp", objective="autoencoder", latent_norm="none", linear_ae=(kind == "linear")).to(DEV)


def true_batches(seed, kind, pool, B, total):
    """Replay of ntk_lib.true_batches but the model is built exactly like train.py (AE: with decoder) so RNG consumption matches."""
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    LatentProbeModel(16, encoder_arch="mlp", objective="autoencoder", latent_norm="none", linear_ae=(kind == "linear"))
    N = len(pool); out, step = {}, 0
    while step < total:
        torch.empty((), dtype=torch.int64).random_()
        sd = int(torch.empty((), dtype=torch.int64).random_().item())
        g = torch.Generator(); g.manual_seed(sd)
        perm = torch.randperm(N, generator=g).numpy()
        for i in range(0, N, B):
            step += 1; out[step] = pool[perm[i:i + B]].astype(np.int64)
            if step >= total: break
    return out


class Net:
    def __init__(self, model):
        self.model = model
        self.named = list(model.named_parameters())
        self.idx = {n: i for i, (n, _) in enumerate(self.named)}
        self.enc = [n for n, _ in self.named if n.startswith("encoder.")]
        self.dec = [n for n, _ in self.named if n.startswith("decoder.")]
        self.names = self.enc + self.dec

    def strip(self, p, pre):
        return {k[len(pre):]: v for k, v in p.items() if k.startswith(pre)}

    def z(self, p, x):
        return functional_call(self.model.encoder, self.strip(p, "encoder."), (x.view(1, 1, 28, 28),))[0]

    def rec(self, p, x):
        return functional_call(self.model.decoder, self.strip(p, "decoder."), (self.z(p, x)[None],))[0]

    def L(self, p, x):
        return ((self.rec(p, x) - x.flatten()) ** 2).mean()

    def flat(self, g, names, n=None):
        return torch.cat([(g[k].reshape(n, -1) if n is not None else g[k].reshape(-1)) for k in names], -1)


def load_ck(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def pdict(net, ck):
    return {n: ck["model"][n].to(DEV).float() for n in net.names}


def adam_vecs(net, ck, key):
    st = ck["optimizer"]["state"]
    return {n: st[net.idx[n]][key].to(DEV).float() for n in net.names}, int(st[0]["step"]) if 0 in st else None


def blocks(net, d, n=None):
    f = lambda names: torch.cat([d[k].reshape(n, -1) if n else d[k].reshape(-1) for k in names], -1)
    return f(net.enc), f(net.dec)


def analyze_pair(net, rid, cfg, step, Xtr_all, Xte_all, ytr_all, yte_all, assign, probe, bat, K, gen):
    kind, seed, N, wd = parse_rid(rid)
    rd = f"{ROOT}/runs/{rid}"
    c0, c1 = load_ck(f"{rd}/dense/ckpt/step_{step:07d}.pt"), load_ck(f"{rd}/dense/ckpt/step_{step + 1:07d}.pt")
    p0 = pdict(net, c0); p1 = pdict(net, c1)
    net.model.load_state_dict(c0["model"]); net.model.eval()
    v1, s1 = adam_vecs(net, c1, "exp_avg_sq"); m0, _ = adam_vecs(net, c0, "exp_avg"); m1, _ = adam_vecs(net, c1, "exp_avg")
    g0 = c1["optimizer"]["param_groups"][0]; lr, (b1, b2), eps, wdc = g0["lr"], g0["betas"], g0["eps"], g0["weight_decay"]
    bc1, bc2 = 1 - b1 ** s1, 1 - b2 ** s1
    D = {n: 1.0 / (v1[n].sqrt() / math.sqrt(bc2) + eps) for n in net.names}
    sD = {n: D[n].sqrt() for n in net.names}
    sDe, sDd = blocks(net, sD); De, Dd = blocks(net, D)
    tr_idx, te_idx = probe["tr"], probe["te"]
    X = torch.cat([Xtr_all[tr_idx], Xte_all[te_idx]]).to(DEV); n = len(X)
    is_tr = np.r_[np.ones(len(tr_idx), bool), np.zeros(len(te_idx), bool)]
    # per-sample loss gradients, losses, latents
    Lfun = vmap(grad(net.L), in_dims=(None, 0))
    Us = Lfun(p0, X); Ue, Ud = blocks(net, Us, n)
    Lvals = vmap(net.L, in_dims=(None, 0))(p0, X).detach()
    Lnext = vmap(net.L, in_dims=(None, 0))(p1, X).detach()
    Z = vmap(net.z, in_dims=(None, 0))(p0, X).detach()
    out = {}
    def gram(Fe, Fd, tag):
        out[f"{tag}pre_e"] = ((Fe * sDe) @ (Fe * sDe).T).cpu().numpy(); out[f"{tag}raw_e"] = (Fe @ Fe.T).cpu().numpy()
        if Fd is not None:
            out[f"{tag}pre_d"] = ((Fd * sDd) @ (Fd * sDd).T).cpu().numpy(); out[f"{tag}raw_d"] = (Fd @ Fd.T).cpu().numpy()
    gram(Ue, Ud, "G")
    # pixel-output trace kernel (Hutchinson)
    Te = {k: torch.zeros(n, n, device=DEV) for k in ("pre_e", "raw_e", "pre_d", "raw_d")}
    for k in range(K):
        s = torch.randn(784, generator=gen, device=DEV)
        Fs = vmap(grad(lambda p, x: (net.rec(p, x) * s).sum()), in_dims=(None, 0))(p0, X)
        Fe, Fd = blocks(net, Fs, n)
        Fe2, Fd2 = Fe * sDe, Fd * sDd
        Te["pre_e"] += Fe2 @ Fe2.T; Te["pre_d"] += Fd2 @ Fd2.T; Te["raw_e"] += Fe @ Fe.T; Te["raw_d"] += Fd @ Fd.T
        del Fs, Fe, Fd, Fe2, Fd2
    for k, v in Te.items(): out["T" + k] = (v / K).cpu().numpy()
    # latent-output kernel (exact, 16 unit probes)
    Tz_pre = torch.zeros(n, n, device=DEV); Tz_raw = torch.zeros(n, n, device=DEV)
    for k in range(16):
        Fz = vmap(grad(lambda p, x: net.z(p, x)[k]), in_dims=(None, 0))(p0, X)
        Fze = torch.cat([Fz[a].reshape(n, -1) for a in net.enc], 1)
        Tz_pre += (Fze * sDe) @ (Fze * sDe).T; Tz_raw += Fze @ Fze.T
        del Fz, Fze
    out["Tzpre_e"] = Tz_pre.cpu().numpy(); out["Tzraw_e"] = Tz_raw.cpu().numpy()
    # ---- one-step first-order decomposition
    B = 128
    bidx = bat[step + 1]; Xb = Xtr_all[bidx].to(DEV)
    gb = grad(lambda p: vmap(net.L, in_dims=(None, 0))(p, Xb).mean())(p0)
    gbe, gbd = blocks(net, gb)
    th0e, th0d = blocks(net, p0)
    kap = lr / bc1
    g_rec = {a: (m1[a] - b1 * m0[a]) / (1 - b1) for a in net.names}
    gre, grd = blocks(net, g_rec)
    gtot_e, gtot_d = gbe + wdc * th0e, gbd + wdc * th0d
    chk = dict(gcos=float(torch.cat([gtot_e, gtot_d]) @ torch.cat([gre, grd]) / (torch.cat([gtot_e, gtot_d]).norm() * torch.cat([gre, grd]).norm())),
               grel=float((torch.cat([gtot_e, gtot_d]) - torch.cat([gre, grd])).norm() / torch.cat([gre, grd]).norm()))
    Hh = (-kap * b1 * blocks(net, m0)[0] * De, -kap * b1 * blocks(net, m0)[1] * Dd)
    Nn = (-kap * (1 - b1) * gbe * De, -kap * (1 - b1) * gbd * Dd)
    Ww = (-kap * (1 - b1) * wdc * th0e * De, -kap * (1 - b1) * wdc * th0d * Dd)
    p1e, p1d = blocks(net, p1)
    dth = (p1e - th0e, p1d - th0d)
    pred = torch.zeros(n, 5, 2, device=DEV)  # comps: H, Nown, Noth, W, total-actual-linear(dtheta)
    for b, (U, Hb, Nb, Wb, db) in enumerate([(Ue, Hh[0], Nn[0], Ww[0], dth[0]), (Ud, Hh[1], Nn[1], Ww[1], dth[1])]):
        pred[:, 0, b] = U @ Hb; Ntot = U @ Nb; pred[:, 3, b] = U @ Wb; pred[:, 4, b] = U @ db
        pred[:, 2, b] = Ntot
    inb = np.zeros(n, bool)
    inb[:len(tr_idx)] = np.isin(tr_idx, bidx)
    inb_t = torch.tensor(inb, device=DEV)
    Gd_e = (Ue * De * Ue).sum(1); Gd_d = (Ud * Dd * Ud).sum(1)
    own = torch.stack([-kap * (1 - b1) / B * Gd_e, -kap * (1 - b1) / B * Gd_d], 1) * inb_t[:, None]
    pred[:, 1, :] = own; pred[:, 2, :] -= own
    # per-layer (weight+bias of each of 3 enc / 3 dec layers) first-order contribution of the ACTUAL step to dL_i (and of H, N separately)
    lay = {}
    for nme in net.names:
        lay.setdefault(nme.rsplit('.', 1)[0], []).append(nme)
    layer_names = list(lay)
    dLl = torch.zeros(n, 3, len(layer_names), device=DEV)   # comps: actual dtheta, H, N(all batch)
    for li, ln in enumerate(layer_names):
        U = torch.cat([Us[k].reshape(n, -1) for k in lay[ln]], 1)
        th = lambda dd: torch.cat([dd[k].reshape(-1) for k in lay[ln]])
        dl_act = th({k: p1[k] - p0[k] for k in lay[ln]})
        dLl[:, 0, li] = U @ dl_act
        dLl[:, 1, li] = U @ th({k: -kap * b1 * m0[k] * D[k] for k in lay[ln]})
        dLl[:, 2, li] = U @ th({k: -kap * (1 - b1) * gb[k] * D[k] for k in lay[ln]})
    # outputs
    Xr = Xtr_all[tr_idx].numpy(); 
    out.update(L=Lvals.cpu().numpy(), Lnext=Lnext.cpu().numpy(), Z=Z.cpu().numpy(), is_tr=is_tr, in_batch=inb,
               idx=np.r_[tr_idx, te_idx], y=np.r_[ytr_all[tr_idx], yte_all[te_idx]],
               km=np.r_[assign(Xtr_all[tr_idx].numpy()), assign(Xte_all[te_idx].numpy())],
               dL=pred.cpu().numpy(), dLlayer=dLl.cpu().numpy(), layer_names=np.array(layer_names), step=step, gcos=chk["gcos"], grel=chk["grel"], bc1=bc1,
               dtheta_norm=float(torch.cat(list(dth)).norm()), nbatch_probe=int(inb.sum()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="*"); ap.add_argument("--epochs", default="10,30,60,120,250,1000")
    ap.add_argument("--nprobe", type=int, default=200); ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--force", action="store_true"); ap.add_argument("--unfinished", action="store_true"); ap.add_argument("--out", default=None); a = ap.parse_args()
    Xtr_all, ytr_all = A.images("mnist", True); Xte_all, yte_all = A.images("mnist", False)
    Xtr_all = torch.tensor(Xtr_all); Xte_all = torch.tensor(Xte_all)
    assign = A.group_model("mnist", f"{A.OUT}/_cache")
    epochs = [int(e) for e in a.epochs.split(",")]
    for pat in a.runs.split(","):
        for rd in sorted(glob.glob(f"{ROOT}/runs/{pat}")):
            rid = os.path.basename(rd)
            if not a.unfinished and not os.path.exists(f"{rd}/model_final.pt"): print("skip (unfinished)", rid, flush=True); continue
            kind, seed, N, wd = parse_rid(rid)
            Nk = "full" if N == "full" else N
            net = Net(build_model(kind))
            rng = np.random.RandomState(0)
            ptr = np.load(f"{rd}/probe_train_index.npz")["indices"]; pte = np.load(f"{rd}/probe_index.npz")["indices"]
            probe = dict(tr=np.sort(rng.choice(ptr, min(a.nprobe, len(ptr)), replace=False)),
                         te=np.sort(rng.choice(pte, a.nprobe, replace=False)))
            pool = np.load(f"{rd}/train_index.npz")["indices"] if os.path.exists(f"{rd}/train_index.npz") else np.arange(60000)
            os.makedirs(f"{ROOT}/ana/{rid}", exist_ok=True)
            bat = None
            for e in epochs:
                if e not in EPOCHS_ALL[Nk]: continue
                step = epoch_step(Nk, e); f = f"{ROOT}/ana/{rid}/step_{step:07d}.npz"
                if os.path.exists(f) and not a.force: continue
                if not os.path.exists(f"{rd}/dense/ckpt/step_{step + 1:07d}.pt"): continue
                if bat is None: bat = true_batches(seed, kind, pool, 128, 20000)
                t0 = time.time(); gen = torch.Generator(device=DEV); gen.manual_seed(1234 + step)
                out = analyze_pair(net, rid, None, step, Xtr_all, Xte_all, ytr_all, yte_all, assign, probe, bat, a.K, gen)
                out["epoch"] = step * 128 / (60000 if N == "full" else N)
                np.savez_compressed(f, **out)
                print(f"{rid} step={step} ep={out['epoch']:.1f} gcos={out['gcos']:.4f} grel={out['grel']:.3f} {time.time()-t0:.0f}s", flush=True)
            del net; torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
