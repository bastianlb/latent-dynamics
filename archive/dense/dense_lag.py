"""Dense per-step latent analysis (dense_latent runs): multi-lag partition, in-batch effect, logit frame, autocorr/PSD, spikes.
Usage: python dense_lag.py q1 [--runs tag,tag]   (lag partition, latent + logit frame, test + train probe)
       python dense_lag.py q2 | q4 | q5 | plot
GPU: pinned to GPU 1 below (before torch import)."""
import os, sys
DS = os.environ.get("DL_DATASET", "mnist"); LIGHT = os.environ.get("DL_LIGHT", "0") == "1"; os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("DL_CUDA", "1")
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_k] = "8"
import glob, json, csv, time, math
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latentdyn.vfield import core as V
from latentdyn.vfield import decomp as VD

ROOT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent"
OUT = ROOT + "/analysis" + {"mnist": "", "cifar10": "/cifar"}.get(DS, "/fm")
os.makedirs(OUT, exist_ok=True)
TAGS = ["dl_rn18_n10000_ln0", "dl_rn18_n10000_ln0.2"] if DS == "cifar10" else ["dl_adam_n1000_wd0", "dl_adam_n4000_wd0", "dl_adam_n4000_wd1e-3", "dl_adam_nfull_wd0", "dl_adam_n4000_ln0.2", "dl_sgd_lr0.025_n4000_wd0"]
DEV = os.environ.get("DL_DEV", "cuda")
DT = torch.float64
T_END = int(os.environ.get("DL_TEND", 20000)); PREFIX = os.environ.get("DL_PREFIX", f"{DS}_probe_none_d16_s0_")


class Run:
    def __init__(self, tag):
        self.tag = tag; rd = f"{ROOT}/{PREFIX}{tag}"; art = f"{ROOT}/_run_artifacts/runs/{PREFIX}{tag}"
        zt, zr, W, b, L, IB = [], [], [], [], [], []; steps = []
        for f in sorted(glob.glob(rd + "/chunk_*.npz")):
            with np.load(f) as z:
                steps.append(z["steps"]); zt.append(z["latents_test"]); zr.append(z["latents_train"]); W.append(z["head_w"]); b.append(z["head_b"]); L.append(z["train_probe_loss"]); IB.append(z["train_probe_in_batch"])
        steps = np.concatenate(steps); assert (steps == np.arange(T_END + 1)).all()
        self.Zte = torch.from_numpy(np.concatenate(zt)).to(DEV); self.Ztr = torch.from_numpy(np.concatenate(zr)).to(DEV)
        self.W = torch.from_numpy(np.concatenate(W)).to(DEV); self.b = torch.from_numpy(np.concatenate(b)).to(DEV)
        self.loss = np.concatenate(L); self.IB = np.concatenate(IB)
        self.y_te = np.load(art + "/probe_index.npz")["labels"].astype(np.int64); self.y_tr = np.load(rd + "/probe_info.npz")["train_probe_labels_used"].astype(np.int64)
        self.classes = np.arange(10)
        self.N = len(np.load(rd + "/probe_info.npz")["pool_indices"])
        self.is_sgd = "sgd" in tag
        self._pi = np.load(rd + "/probe_info.npz"); self._tb = None

    def true_mask(self):
        """train-probe in-batch mask (T_END+1,1000) rebuilt by replaying the shuffle RNG (ntk_lib.true_batches); logged mask is wrong early."""
        if self._tb is None:
            from latentdyn.ntk import lib as ntk_lib
            tb = ntk_lib.true_batches(0, self._pi["pool_indices"], 128, T_END); orig = self._pi["train_probe_orig_idx"]; M = np.zeros((T_END + 1, 1000), bool)
            for s in range(1, T_END + 1): M[s] = np.isin(orig, tb[s])
            self._tb = M
        return self._tb

    def grad(self, probe, t, yt):
        return grad_z(self.Z(probe)[t].to(DT), self.W[t].to(DT), self.b[t].to(DT), yt)

    def Z(self, probe): return self.Zte if probe == "test" else self.Ztr
    def y(self, probe): return self.y_te if probe == "test" else self.y_tr

    def latent(self, probe, ts): return self.Z(probe)[torch.as_tensor(ts, device=DEV)].to(DT)

    def logit(self, probe, ts):
        ti = torch.as_tensor(ts, device=DEV); z = self.Z(probe)[ti].to(DT); l = z @ self.W[ti].to(DT).transpose(1, 2) + self.b[ti].to(DT)[:, None, :]
        return l - l.mean(-1, keepdim=True)

    def floor(self, probe, t0, t1, frame):
        """expected fp16-rounding noise energy (mean over samples of sum over coords) of Z[t1]-Z[t0] (or of logit displacement)."""
        out = []
        for a, c in zip(t0, t1):
            tot = 0.0
            for t in (a, c):
                z = self.Z(probe)[int(t)].float(); e = torch.floor(torch.log2(z.abs().clamp_min(2.0 ** -24))).clamp_min(-14); ulp = 2.0 ** (e - 10); v = ulp ** 2 / 12   # (n,16)
                if frame == "latent": tot += float(v.sum(1).mean())
                else:
                    Wt = self.W[int(t)].double(); Wc = Wt - Wt.mean(0, keepdim=True); w2 = (Wc ** 2).sum(0).float()   # (16,)
                    tot += float((v * w2[None]).sum(1).mean())
            out.append(tot)
        return np.array(out)


class CRun(Run):
    """CIFAR (512-d latents): latents projected to a 16-d PCA frame (vfield pca16 recipe: pooled per-snapshot-normalised test+train probe latents, every 50th step), logits from per-step head."""
    def __init__(self, tag):
        self.tag = tag; rd = f"{ROOT}/{PREFIX}{tag}"; art = f"{ROOT}/_run_artifacts/runs/{PREFIX}{tag}"; self.rd = rd
        files = sorted(glob.glob(rd + "/chunk_*.npz")); cache = f"{OUT}/_cache_{tag}.npz"; full = f"{OUT}/_proj_{tag}.npz"
        self.y_te = np.load(art + "/probe_index.npz")["labels"].astype(np.int64); pi = np.load(rd + "/probe_info.npz"); self.y_tr = pi["train_probe_labels_used"].astype(np.int64)
        self.classes = np.arange(10); self.N = len(pi["pool_indices"]); self.is_sgd = False
        if os.path.exists(full):
            c = np.load(full); self.Zte, self.Ztr = torch.from_numpy(c["Zte"]).to(DEV), torch.from_numpy(c["Ztr"]).to(DEV); self.W = torch.from_numpy(c["W"]).to(DEV); self.b = torch.from_numpy(c["b"]).to(DEV)
            self.Wnorm_full = torch.from_numpy(c["Wn"]).to(DEV); self.Lte, self.Ltr = torch.from_numpy(c["Lte"]).to(DEV), torch.from_numpy(c["Ltr"]).to(DEV)
            self.fl = {"test": torch.from_numpy(c["fte"]).to(DEV), "train": torch.from_numpy(c["ftr"]).to(DEV)}; self.loss = c["loss"]; self.IB = c["IB"]; return
        if os.path.exists(cache):
            c = np.load(cache); P = c["P"]
        else:
            acc = []
            for f in files:
                with np.load(f) as z:
                    st = z["steps"]; idx = np.where(st % 50 == 0)[0]
                    for k in ("latents_test", "latents_train"):
                        X = z[k][idx].astype(np.float32)
                        X = X - X.mean(1, keepdims=True); X = X / (np.sqrt((X ** 2).sum(2).mean(1))[:, None, None] + 1e-30); acc.append(X.reshape(-1, X.shape[2]))
                print("pass1", f[-14:], flush=True)
            A = np.concatenate(acc).astype(np.float64); A -= A.mean(0); C = A.T @ A; w, V_ = np.linalg.eigh(C); P = V_[:, ::-1][:, :16].copy(); self.var_in_basis = float(w[::-1][:16].sum() / w.sum()); del A, acc
        Pt = torch.tensor(P, dtype=torch.float32); w2 = (Pt ** 2).sum(1)      # (512,) sum_j P_kj^2
        out = {"zt": [], "zr": [], "Wp": [], "b": [], "L": [], "IB": [], "ft": [], "fr": [], "lt": [], "lr": [], "Wn": []}; steps = []
        for f in files:
            with np.load(f) as z:
                steps.append(z["steps"]); W = torch.from_numpy(z["head_w"]); b = torch.from_numpy(z["head_b"]); Wc = W - W.mean(1, keepdim=True); u = (Wc ** 2).sum(1)    # (k,512)
                out["Wp"].append(W @ Pt); out["b"].append(b); out["Wn"].append(W.flatten(1).norm(dim=1)); out["L"].append(z["train_probe_loss"]); out["IB"].append(z["train_probe_in_batch"])
                for key, k in (("t", "latents_test"), ("r", "latents_train")):
                    X = torch.from_numpy(z[k]); k_ = X.shape[0]; zs = []; fs_lat = []; fs_log = []; ls = []
                    for a in range(0, k_, 100):
                        x = X[a:a + 100].float(); zs.append((x @ Pt).half() if False else x @ Pt)
                        e = torch.floor(torch.log2(x.abs().clamp_min(2.0 ** -24))).clamp_min(-14); v = (2.0 ** (e - 10)) ** 2 / 12
                        fs_lat.append((v * w2).sum(2).mean(1)); fs_log.append((v * u[a:a + 100, None, :]).sum(2).mean(1))
                        l = x @ W[a:a + 100].transpose(1, 2) + b[a:a + 100, None, :]; ls.append(l - l.mean(-1, keepdim=True))
                    out["z" + key].append(torch.cat(zs)); out["f" + key].append(torch.stack([torch.cat(fs_lat), torch.cat(fs_log)], 1)); out["l" + key].append(torch.cat(ls))
            print("pass2", f[-14:], flush=True)
        steps = np.concatenate(steps); assert (steps == np.arange(len(steps))).all() and len(steps) == T_END + 1, (len(steps), T_END)
        cat = lambda k: torch.cat(out[k]).to(DEV)
        self.Zte, self.Ztr = cat("zt"), cat("zr"); self.W = cat("Wp"); self.b = cat("b"); self.Wnorm_full = cat("Wn"); self.Lte, self.Ltr = cat("lt"), cat("lr")
        self.fl = {"test": cat("ft"), "train": cat("fr")}; self.loss = np.concatenate(out["L"]); self.IB = np.concatenate(out["IB"])
        if not os.path.exists(cache): np.savez(cache, P=P, var=getattr(self, "var_in_basis", np.nan))
        np.savez(full, Zte=self.Zte.cpu().numpy(), Ztr=self.Ztr.cpu().numpy(), W=self.W.cpu().numpy(), b=self.b.cpu().numpy(), Wn=self.Wnorm_full.cpu().numpy(), Lte=self.Lte.cpu().numpy(), Ltr=self.Ltr.cpu().numpy(),
                 fte=self.fl["test"].cpu().numpy(), ftr=self.fl["train"].cpu().numpy(), loss=self.loss, IB=self.IB)

    def logit(self, probe, ts):
        return (self.Lte if probe == "test" else self.Ltr)[torch.as_tensor(ts, device=DEV)].to(DT)

    def floor(self, probe, t0, t1, frame):
        j = 0 if frame == "latent" else 1; f = self.fl[probe][:, j].cpu().numpy().astype(np.float64); return f[np.asarray(t0)] + f[np.asarray(t1)]

    def grad(self, probe, t, yt):
        L = (self.Lte if probe == "test" else self.Ltr)[t].to(DT); p = torch.softmax(L, 1); oh = torch.zeros_like(p); oh[torch.arange(len(yt)), yt] = 1
        return (p - oh) @ self.W[t].to(DT)


def partition_pairs(X0, X1, ycond, classes, nsplit=None, bs=60):
    nsplit = nsplit or (2 if LIGHT else 3)
    """X0, X1 (K,n,d) float64 on GPU. Held-out (stratified 50/50, nsplit splits) energies per pair. Returns dict of (K,) arrays of per-sample mean energies."""
    K = X0.shape[0]; res = {k: np.zeros(K) for k in ("sst", "sse1", "sse2", "sse3", "sse_k")}; cidx = torch.tensor([int(np.where(classes == c)[0][0]) for c in ycond], device=DEV)
    for a in range(0, K, bs):
        Z0, Z1 = X0[a:a + bs], X1[a:a + bs]; D = Z1 - Z0
        for sp in range(nsplit):
            tr, te = V.make_split(ycond, sp); trt, tet = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV)
            beta = VD.global_affine_fit(Z0, Z1, trt); p1_all = VD.apply_affine_disp(Z0, beta); r1 = D - p1_all; Dte = D[:, tet]
            mu = Z0[:, trt].mean(1, keepdim=True); sig = ((Z0[:, trt] - mu) ** 2).sum(2).mean(1).sqrt()[:, None, None] + 1e-30; Zs = (Z0 - mu) / sig
            p1 = p1_all[:, tet]; _, p2c, _ = V.class_models(Zs, r1, r1, ycond, tr, te, classes); _, p2c_tr, _ = V.class_models(Zs, r1, r1, ycond, tr, tr, classes)
            r2_tr = r1[:, trt] - p2c_tr; base = p1 + p2c
            fkr = VD.krr(Zs[:, trt], Zs[:, tet], r2_tr, cidx[trt], cidx[tet], 0.0, 2.0)
            idx = torch.cdist(Zs[:, tet], Zs[:, trt]).topk(10, largest=False).indices; fk = V.knn_pred(r2_tr, idx, 10)
            nte = len(te); sl = slice(a, a + len(Z0))
            res["sst"][sl] += ((Dte ** 2).sum((1, 2)) / nte).cpu().numpy() / nsplit
            res["sse1"][sl] += (((Dte - p1) ** 2).sum((1, 2)) / nte).cpu().numpy() / nsplit
            res["sse2"][sl] += (((Dte - base) ** 2).sum((1, 2)) / nte).cpu().numpy() / nsplit
            res["sse3"][sl] += (((Dte - base - fkr) ** 2).sum((1, 2)) / nte).cpu().numpy() / nsplit
            res["sse_k"][sl] += (((Dte - base - fk) ** 2).sum((1, 2)) / nte).cpu().numpy() / nsplit
    return res


def make_pairs(Tmax=T_END):
    bases = sorted(set(int(round(x)) for x in np.geomspace(4, T_END - 1000, 10 if LIGHT else 18)))
    snaps = np.unique(np.round(np.geomspace(1, T_END, 120)).astype(int)); lags = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]
    pairs = []
    for t in bases:
        ls = set(lags)
        j = np.searchsorted(snaps, t); j = min(max(j, 0), len(snaps) - 2); ls.add(int(snaps[j + 1] - snaps[j]))     # snapshot-equivalent lag
        for l in sorted(ls):
            if t + l <= Tmax: pairs.append((t, l, int(l == int(snaps[j + 1] - snaps[j]) and l not in lags)))
    return pairs


def q1(tags):
    pairs = make_pairs(); ts = np.array([p[0] for p in pairs]); ls = np.array([p[1] for p in pairs]); snapflag = np.array([p[2] for p in pairs])
    for tag in tags:
        t0 = time.time(); R = (CRun if DS == "cifar10" else Run)(tag); rows = []
        for probe in (("test",) if LIGHT else ("test", "train")):
            for frame in ("latent", "logit"):
                get = R.latent if frame == "latent" else R.logit
                X0 = get(probe, ts); X1 = get(probe, ts + ls); P = partition_pairs(X0, X1, R.y(probe), R.classes); fl = R.floor(probe, ts, ts + ls, frame)
                for i in range(len(ts)):
                    rows.append([tag, probe, frame, ts[i], ls[i], snapflag[i], P["sst"][i], P["sse1"][i], P["sse2"][i], P["sse3"][i], P["sse_k"][i], fl[i]])
                print(tag, probe, frame, f"{time.time()-t0:.0f}s", flush=True)
        with open(f"{OUT}/q13_lag_{tag}.csv", "w") as f:
            w = csv.writer(f); w.writerow(["run", "probe", "frame", "t", "lag", "snap_equiv", "E_tot", "E_res_aff", "E_res_cls", "E_res_krr", "E_res_knn", "E_floor"]); w.writerows(rows)
        del R; torch.cuda.empty_cache()


# ---------------------------------------------------------------- Q2
def q2_steps(rng):
    early = list(range(2, 101)); mid = sorted(set(int(x) for x in np.geomspace(101, 3000, 120).round())); late = sorted(rng.choice(np.arange(3001, T_END + 1), 150, replace=False).tolist())
    if LIGHT: early, mid, late = early[::3], mid[::3], late[::4]
    return {"early": early, "mid": mid, "late": late}


def grad_z(Z0, W0, b0, y):
    p = torch.softmax(Z0 @ W0.T + b0, 1); oh = torch.zeros_like(p); oh[torch.arange(len(y)), y] = 1; return (p - oh) @ W0


def q2(tags):
    rng = np.random.RandomState(0); stages = q2_steps(rng); allrows = []
    for tag in tags:
        R = (CRun if DS == "cifar10" else Run)(tag); t0 = time.time(); IBm = R.IB if DS != "mnist" else R.true_mask(); print(tag, "logged-vs-true mask agreement s>=1000:", float((R.IB[1000:] == IBm[1000:]).mean()), "s<700:", float((R.IB[1:700] == IBm[1:700]).mean()), flush=True)
        for probe in ("train", "test"):
            y = R.y(probe); yt = torch.tensor(y, device=DEV)
            for stage, steps in stages.items():
                acc = {g: dict(n=0, D2=0., res2=0., dot=0., Dn_gn=0., rdot=0., rn_gn=0., cos=0., rcos=0.) for g in ("in", "out")}
                for s in steps:
                    Z0 = R.Z(probe)[s - 1].to(DT); Z1 = R.Z(probe)[s].to(DT); D = Z1 - Z0; W0 = R.W[s - 1].to(DT); b0 = R.b[s - 1].to(DT); g = R.grad(probe, s - 1, yt)
                    inb = IBm[s] if probe == "train" else np.zeros(1000, bool)
                    for sp in range(2):
                        tr, te = V.make_split(y, 100 * s + sp)
                        if sp: tr, te = te, tr
                        trf = tr[~inb[tr]]; trt, tet, trft = torch.tensor(tr, device=DEV), torch.tensor(te, device=DEV), torch.tensor(trf, device=DEV)
                        Zr = Z0[None]; Dr = D[None]; beta = VD.global_affine_fit(Zr, (Zr + Dr), trft); p1 = VD.apply_affine_disp(Zr, beta)[0]; r1 = (D - p1)
                        mu = Z0[trft].mean(0, keepdim=True); sig = ((Z0[trft] - mu) ** 2).sum(1).mean().sqrt() + 1e-30; Zs = ((Z0 - mu) / sig)[None]
                        _, p2c, _ = V.class_models(Zs, r1[None], r1[None], y, trf, te, R.classes); res = (r1[tet] - p2c[0])
                        Dt = D[tet]; gt = -g[tet]; mt = inb[te]
                        for gname, m in (("in", mt), ("out", ~mt)):
                            if m.sum() == 0: continue
                            mm = torch.tensor(m, device=DEV); a = acc[gname]; Dm, rm, gm = Dt[mm], res[mm], gt[mm]; gn = gm.norm(dim=1)
                            a["n"] += int(m.sum()); a["D2"] += float((Dm ** 2).sum()); a["res2"] += float((rm ** 2).sum())
                            a["dot"] += float((Dm * gm).sum()); a["Dn_gn"] += float((Dm.norm(dim=1) * gn).sum()); a["rdot"] += float((rm * gm).sum()); a["rn_gn"] += float((rm.norm(dim=1) * gn).sum())
                            a["cos"] += float(((Dm * gm).sum(1) / (Dm.norm(dim=1) * gn + 1e-30)).sum()); a["rcos"] += float(((rm * gm).sum(1) / (rm.norm(dim=1) * gn + 1e-30)).sum())
                for gname, a in acc.items():
                    if a["n"] == 0: continue
                    allrows.append([tag, probe, stage, gname, a["n"], a["D2"] / a["n"], a["res2"] / a["n"], a["res2"] / a["D2"], a["cos"] / a["n"], a["rcos"] / a["n"], a["dot"] / a["Dn_gn"], a["rdot"] / a["rn_gn"]])
            print(tag, probe, f"{time.time()-t0:.0f}s", flush=True)
        del R; torch.cuda.empty_cache()
        with open(f"{OUT}/q2_inbatch_{tag}.csv", "w") as f:
            w = csv.writer(f); w.writerow(["run", "probe", "stage", "group", "n", "E_D", "E_res", "res_share", "cos_D_negg", "cos_res_negg", "wcos_D_negg", "wcos_res_negg"]); w.writerows([r for r in allrows if r[0] == tag])


def q2floor(tags):
    rng = np.random.RandomState(0); stages = q2_steps(rng); rows = []
    for tag in tags:
        R = (CRun if DS == "cifar10" else Run)(tag)
        for probe in ("train", "test"):
            for st, steps in stages.items():
                s_ = np.array(steps); rows.append([tag, probe, st, float(R.floor(probe, s_ - 1, s_, "latent").mean())])
        del R; torch.cuda.empty_cache()
    with open(f"{OUT}/q2_floor.csv", "w") as f:
        w = csv.writer(f); w.writerow(["run", "probe", "stage", "E_floor"]); w.writerows(rows)


# ---------------------------------------------------------------- Q4
WINS_AC = [100, 1000, 3000, 6000] if T_END < 10000 else [100, 1000, 5000, 15000]; LEN_AC = 1500; KMAX = 400


def q4(tags):
    rows = []; psd_out = {}
    for tag in tags:
        R = (CRun if DS == "cifar10" else Run)(tag)
        for probe in ("test", "train"):
            for t0 in WINS_AC:
                ts = np.arange(t0, t0 + LEN_AC + 1); Z = R.Z(probe)[torch.as_tensor(ts, device=DEV)].to(DT); D = Z[1:] - Z[:-1]
                Z0 = Z[:-1]; beta = VD.global_affine_fit(Z0, Z[1:], torch.arange(Z.shape[1], device=DEV)); Rs = D - VD.apply_affine_disp(Z0, beta)
                F = R.floor(probe, ts[:-1], ts[1:], "latent"); Fm = float(F.mean())
                for kind, X in (("raw", D), ("resid_affine", Rs)):
                    E = float((X ** 2).sum(2).mean(1).mean()); ac = []
                    for k in range(1, KMAX + 1): ac.append(float((X[:-k] * X[k:]).sum(2).mean() / ((X ** 2).sum(2).mean()))) if True else None
                    ac = np.array(ac); c1c = (ac[0] * E + Fm / 2) / max(E - Fm, 1e-30)
                    rows.append([tag, probe, t0, kind, E, Fm, Fm / E if kind == "raw" else float("nan"), c1c] + ac.tolist())
        # PSD of velocity (test probe + train probe), Welch L=512
        Lw = 512; w = torch.hann_window(Lw, dtype=DT, device=DEV); starts = [200, 1500, 3500, 5800] if T_END < 10000 else [200, 2000, 8000, 17000]
        for probe in ("test", "train"):
            for st in starts:
                ts = np.arange(st, st + 2048 + 1); Z = R.Z(probe)[torch.as_tensor(ts, device=DEV)].to(DT); V_ = (Z[1:] - Z[:-1]).permute(1, 2, 0).reshape(-1, 2048)   # (n*16, T)
                P = 0; ns = 0
                for a in range(0, 2048 - Lw + 1, Lw // 2):
                    seg = V_[:, a:a + Lw]; seg = seg - seg.mean(1, keepdim=True); P = P + (torch.fft.rfft(seg * w, dim=1).abs() ** 2).mean(0) / (w ** 2).sum(); ns += 1
                P = (P / ns).cpu().numpy(); fl = R.floor(probe, ts[:-1], ts[1:], "latent").mean() / 16
                psd_out[f"{tag}|{probe}|{st}"] = P; psd_out[f"{tag}|{probe}|{st}|floor"] = np.array(fl)
        del R; torch.cuda.empty_cache(); print(tag, flush=True)
    with open(f"{OUT}/q4_autocorr.csv", "w") as f:
        w = csv.writer(f); w.writerow(["run", "probe", "t0", "kind", "E_step", "E_floor", "floor_share", "c1_floor_corrected"] + [f"ac{k}" for k in range(1, KMAX + 1)]); w.writerows(rows)
    np.savez(f"{OUT}/q4_psd.npz", freqs=np.fft.rfftfreq(512), **psd_out)


# ---------------------------------------------------------------- Q5
def detect_spikes(m, lo=500, win=100, ratio=1.5, minabs=0.02):
    base = np.array([np.median(m[max(0, s - win):s]) if s > 0 else m[0] for s in range(len(m))]); flag = (m > ratio * base) & (m - base > minabs) & (np.arange(len(m)) >= lo)
    ev = []; s = lo
    while s < len(m):
        if flag[s]:
            e = s
            while e + 1 < len(m) and (flag[e + 1] or (e + 1 - s < 3) or flag[min(e + 5, len(m) - 1)] ): e += 1
            seg = np.arange(s, e + 1); pk = seg[np.argmax(m[seg])]
            rec = pk
            while rec < len(m) - 1 and m[rec] > 1.2 * base[s] + 1e-9: rec += 1
            ev.append(dict(onset=s, peak=int(pk), end=e, base=float(base[s]), peak_loss=float(m[pk]), ratio=float(m[pk] / base[s]), recover=int(rec)))
            s = max(e, rec) + 1
        else: s += 1
    return ev


def q5(tags, listonly=False):
    rows = []; summ = {}
    for tag in tags:
        R = (CRun if DS == "cifar10" else Run)(tag); m = R.loss.astype(np.float64).mean(1); p99 = np.percentile(R.loss, 99, axis=1)
        ev = detect_spikes(m); summ[tag] = ev
        for e in ev: rows.append([tag, e["onset"], e["peak"], e["end"], e["recover"], e["base"], e["peak_loss"], e["ratio"]])
        # step-resolution through the top-2 spikes (by ratio*amplitude) in the run
        if listonly: del R; continue
        top = sorted(ev, key=lambda e: -(e["peak_loss"] - e["base"]))[:(1 if LIGHT else 2)]
        for k, e in enumerate(top):
            if os.path.exists(f"{OUT}/q5_spike_{tag}_{k}.npz"): continue
            a = max(1, e["onset"] - 60); b = min(T_END - 1, min(max(e["recover"], e["peak"]) + 60, e["peak"] + 250)); ts = np.arange(a, b + 1); out = {"t": ts, "loss": m[ts], "p99": p99[ts]}
            for probe in (("test",) if LIGHT else ("test", "train")):
                for frame in ("latent", "logit"):
                    get = R.latent if frame == "latent" else R.logit; P = partition_pairs(get(probe, ts), get(probe, ts + 1), R.y(probe), R.classes); fl = R.floor(probe, ts, ts + 1, frame)
                    for kk, v in P.items(): out[f"{probe}_{frame}_{kk}"] = v
                    out[f"{probe}_{frame}_floor"] = fl
            sv = []; Z = R.Ztr
            for t in ts:
                Z0 = Z[t].to(DT); Z1 = Z[t + 1].to(DT); X = torch.cat([Z0, torch.ones(1000, 1, dtype=DT, device=DEV)], 1); be = torch.linalg.lstsq(X, Z1).solution; sv.append(torch.linalg.svdvals(be[:16]).cpu().numpy())
            sv = np.array(sv); out["A_smax"] = sv[:, 0]; out["A_smin"] = sv[:, -1]; out["Wnorm"] = np.array([float(R.W[t].norm()) for t in ts])
            out["bnorm"] = np.array([float(R.b[t].norm()) for t in ts]); out["zrms_test"] = np.array([float(R.Zte[t].float().pow(2).sum(1).mean().sqrt()) for t in ts])
            np.savez(f"{OUT}/q5_spike_{tag}_{k}.npz", onset=e["onset"], peak=e["peak"], recover=e["recover"], **out)
        del R; torch.cuda.empty_cache(); print(tag, len(ev), flush=True)
    with open(f"{OUT}/q5_spikes.csv", "w") as f:
        w = csv.writer(f); w.writerow(["run", "onset", "peak", "end", "recover", "base_loss", "peak_loss", "ratio"]); w.writerows(rows)


if __name__ == "__main__":
    cmd = sys.argv[1]; tags = TAGS
    if "--runs" in sys.argv: tags = sys.argv[sys.argv.index("--runs") + 1].split(",")
    {"q5list": lambda t: q5(t, True), "q1": q1, "q2floor": q2floor, "q2": q2, "q4": q4, "q5": q5}[cmd](tags)
