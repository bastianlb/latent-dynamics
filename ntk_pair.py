"""analyze_pair: tests (a)-(e) for one checkpoint pair. See ntk_lib.py for the math."""
import os
import numpy as np
import torch
from ntk_lib import *  # noqa
import ntk_lib as L


def make_ctx(mnist_root, probe_index_npz, probe_train_npz, train_index_npz, n_kernel_per_class=80):
    Xtr, ytr = L.load_mnist_tensors(mnist_root, True)
    Xte, yte = L.load_mnist_tensors(mnist_root, False)
    pi = np.load(probe_index_npz); pt = np.load(probe_train_npz)
    if os.path.exists(train_index_npz):
        ti = np.load(train_index_npz)
        pool_idx = ti["indices"].astype(np.int64)
        pool_lab = (ti["noisy_labels"] if "noisy_labels" in ti.files else ti["labels"]).astype(np.int64)
    else:  # identity pool (N=60000)
        pool_idx = np.arange(len(Xtr)); pool_lab = ytr.numpy().astype(np.int64)
    lab_full = -np.ones(len(Xtr), dtype=np.int64); lab_full[pool_idx] = pool_lab
    tp_idx = pt["indices"].astype(np.int64)
    Xp = torch.cat([Xte[pi["indices"]], Xtr[tp_idx]]).to(L.DEV)
    yp = torch.cat([yte[pi["indices"]], ytr[tp_idx]]).to(L.DEV)  # TRUE classes of probe samples
    # kernel subset: first n per class of the test probe set
    yt = yte[pi["indices"]]
    sel = torch.cat([(yt == c).nonzero().flatten()[:n_kernel_per_class] for c in range(10)])
    return dict(Xfull=Xtr.to(L.DEV), lab_full=torch.as_tensor(lab_full).to(L.DEV), pool_idx=pool_idx,
                tp_idx=tp_idx, Xp=Xp, yp=yp, ntest=len(pi["indices"]), Xk=Xp[sel.to(L.DEV)], yk=yp[sel.to(L.DEV)],
                Xpool=Xtr[pool_idx].to(L.DEV), ypool=torch.as_tensor(pool_lab).to(L.DEV))


def analyze_pair(prev, nxt, batch_idx, ctx, obs_next=None, batch_next_idx=None, kernel=True, seed=0, hist_batches=None, resid_only=False):
    gen = torch.Generator().manual_seed(seed)
    Xp, yp, nt = ctx["Xp"], ctx["yp"], ctx["ntest"]
    Pp = prev["params"]
    P = [t.to(L.DEV) for t in Pp]
    out = {}
    z0, z1 = L.z_of(P, Xp), L.z_of([t.to(L.DEV) for t in nxt["params"]], Xp)
    obs = z1 - z0
    d = L.decompose(prev, nxt)
    B = len(batch_idx)
    bi = torch.as_tensor(batch_idx).to(L.DEV)
    Xb, yb = ctx["Xfull"][bi], ctx["lab_full"][bi]
    assert (yb >= 0).all(), "batch index outside training pool"
    g_b = L.grad_enc(P, Xb, yb, B)
    g_full = L.grad_enc(P, ctx["Xpool"], ctx["ypool"], len(ctx["ypool"]))
    wdth = L.scale(d["th0"], d["wd"])
    g_tot = L.add(g_b, wdth)
    num = L.dot(g_tot, d["g_rec"]); a = L.dot(g_tot, g_tot) ** .5; b = L.dot(d["g_rec"], d["g_rec"]) ** .5
    out["batch_check"] = dict(cos=num / (a * b + 1e-30), rel_err=float(L.dot(L.sub(g_tot, d["g_rec"]), L.sub(g_tot, d["g_rec"])) ** .5 / (b + 1e-30)))
    kn, Dp = d["kn"], d["Dp"]
    mk = lambda g: L.scale(L.mul(Dp, g), -kn)
    T = dict(lin=d["dtheta"], H=d["H"], Nb=mk(g_b), W=mk(wdth), Nd=mk(g_full))
    T["recon"] = L.add(L.add(T["H"], T["Nb"]), T["W"])
    rn = L.sub(T["recon"], T["lin"])
    out["update_recon"] = dict(rel_err=L.dot(rn, rn) ** .5 / (L.dot(T["lin"], T["lin"]) ** .5 + 1e-30))
    dz = {k: L.jvp_z(P, Xp, v) for k, v in T.items()}
    if hist_batches:  # unrolled momentum: m_{t-1} ~ sum_k w_k grad(batch_{t-1-k}; theta_t), most recent first
        b1 = nxt["b1"] if nxt["kind"] == "adam" else nxt["mu"]
        gh = [torch.zeros_like(t) for t in g_b]
        for k, hb in enumerate(hist_batches):
            w = ((1 - b1) if nxt["kind"] == "adam" else 1.0) * b1 ** k
            hbi = torch.as_tensor(hb).to(L.DEV)
            gk = L.add(L.grad_enc(P, ctx["Xfull"][hbi], ctx["lab_full"][hbi], len(hb)), wdth)
            gh = L.add(gh, L.scale(gk, w))
        coef = (nxt["lr"] / (1 - b1 ** nxt["step"]) * b1) if nxt["kind"] == "adam" else nxt["lr"] * b1
        Ha = L.scale(L.mul(Dp, gh), -coef)
        dz["Hunroll"] = L.jvp_z(P, Xp, Ha)
        dz["Hunroll_plus_new"] = dz["Hunroll"] + dz["Nb"] + dz["W"]
        out["hist_unroll"] = {s: L.cmp(dz["Hunroll"], dz["H"], sl) for s, sl in dict(test=slice(0, nt), trainprobe=slice(nt, None)).items()}
    dz["Nf"] = dz["Nb"] - dz["Nd"]
    dz["det"] = dz["H"] + dz["Nd"] + dz["W"]       # prediction without minibatch randomness
    dz["newb"] = dz["Nb"] + dz["W"]
    dz["newd"] = dz["Nd"] + dz["W"]
    out["step_info"] = dict(step=nxt["step"], B=B, lr=float(nxt["lr"]),
                            obs_rms=float((obs ** 2).sum(1).mean().sqrt()), z_rms=float((z0 ** 2).sum(1).mean().sqrt()))
    sls = dict(test=slice(0, nt), trainprobe=slice(nt, None))
    # post-global-affine residuals: each vector is regressed on [z_t,1] (fit separately per set) and the residual compared
    def resid(Y, sl):
        A = torch.cat([z0[sl], torch.ones(len(z0[sl]), 1, device=z0.device)], 1).double()
        sol = torch.linalg.lstsq(A.cpu(), Y[sl].double().cpu(), driver="gelsd").solution.to(A.device)
        return (Y[sl].double() - A @ sol).float()
    rs_ = {}
    for s_, sl in sls.items():
        R_ = {k: resid(v, sl) for k, v in dz.items() if k in ("lin", "H", "Nb", "Nd", "Nf", "newb", "newd", "det")}
        R_["obs"] = resid(obs, sl)
        res_ab = {k: L.cmp(v, R_["obs"]) for k, v in R_.items() if k != "obs"}
        res_ab["obs_energy_frac"] = float((R_["obs"] ** 2).sum() / (obs[sl] ** 2).sum())
        rs_[s_] = res_ab
        if obs_next is not None:
            op_ = obs_next.to(L.DEV)
            rn_ = resid(op_, sl)
            e_ = {k: L.cmp(v, rn_) for k, v in R_.items()}
            e_["obs_next_energy_frac"] = float((rn_ ** 2).sum() / (op_[sl] ** 2).sum())
            rs_[s_]["e_next"] = e_
    out["resid"] = rs_
    if resid_only:
        return out
    # (a)+(b)
    out["ab"] = {s: {k: L.cmp(v, obs, sl) for k, v in dz.items()} for s, sl in sls.items()}
    # (d) affine structure
    out["d_affine"] = {}
    for s, sl in sls.items():
        out["d_affine"][s] = {k: L.affine_levels(z0[sl], dz[k][sl] if k != "obs" else obs[sl], yp[sl], gen)
                              for k in ("obs", "lin", "Nd", "Nf", "Nb", "H", "det")}
    # class-resolved drift: effect of training-class c gradient on probe class c'
    M = np.zeros((10, 10))
    for c in range(10):
        mk_c = ctx["ypool"] == c
        gc = L.grad_enc(P, ctx["Xpool"][mk_c], ctx["ypool"][mk_c], len(ctx["ypool"]))
        dzc = L.jvp_z(P, Xp[:nt], mk(gc))
        for c2 in range(10):
            M[c, c2] = float((dzc[yp[:nt] == c2] ** 2).sum())
    out["d_classblock"] = dict(matrix_norm=(M / M.sum()).tolist(), diag_frac=float(np.trace(M) / M.sum()),
                               row_diag_mean=float(np.mean(np.diag(M) / M.sum(1))))
    # (e) anti-correlation: obs_next = z(t+2)-z(t+1) from one simulated optimizer step after nxt
    if obs_next is not None:
        op = obs_next.to(L.DEV)
        e = {s: {k: L.cmp(v, op, sl) for k, v in dz.items() if k in ("H", "Nd", "Nf", "Nb", "det", "lin", "Hunroll")} for s, sl in sls.items()}
        for s, sl in sls.items():
            e[s]["obs"] = L.cmp(obs, op, sl)
        out["e_next"] = e
        sl = sls["trainprobe"]
        cs = F.cosine_similarity(obs[sl], op[sl], dim=1, eps=1e-30)
        cs_nf = F.cosine_similarity(dz["Nf"][sl], op[sl], dim=1, eps=1e-30)
        inb_cur = torch.as_tensor(np.isin(ctx["tp_idx"], batch_idx)).to(L.DEV)
        inb_nxt = torch.as_tensor(np.isin(ctx["tp_idx"], batch_next_idx)).to(L.DEV)
        f = lambda c, m: float(c[m].mean()) if m.any() else None
        out["e_membership"] = dict(n_incur=int(inb_cur.sum()), n_innext=int(inb_nxt.sum()),
                                   cos_obs_incur=f(cs, inb_cur), cos_obs_notcur=f(cs, ~inb_cur),
                                   cos_obs_innext=f(cs, inb_nxt), cos_obs_notnext=f(cs, ~inb_nxt),
                                   cos_Nf_incur=f(cs_nf, inb_cur), cos_Nf_notcur=f(cs_nf, ~inb_cur))
    # self term for train-probe samples in the batch
    in_b = np.nonzero(np.isin(ctx["tp_idx"], batch_idx))[0]
    if len(in_b):
        rows = []
        for j in in_b:
            xi = Xp[nt + j][None]
            full_i = int(ctx["tp_idx"][j])
            gi = L.grad_enc(P, ctx["Xfull"][full_i][None], ctx["lab_full"][full_i][None], B)
            rows.append(L.jvp_z(P, xi, mk(gi))[0])
        selfz = torch.stack(rows)
        notin = torch.ones(len(ctx["tp_idx"]), dtype=torch.bool, device=L.DEV); notin[torch.as_tensor(in_b).to(L.DEV)] = False
        rr = nt + torch.as_tensor(in_b).to(L.DEV)
        out["self"] = dict(n=len(in_b), self_energy_over_Nb=float((selfz ** 2).sum() / (dz["Nb"][rr] ** 2).sum()),
                           self_cos_Nb=L.cmp(selfz, dz["Nb"][rr])["cos"], self_cos_obs=L.cmp(selfz, obs[rr])["cos"],
                           self_r2_obs=L.cmp(selfz, obs[rr])["r2"],
                           inbatch_obs_r2_Nb=L.cmp(dz["Nb"][rr], obs[rr])["r2"],
                           notin_obs_r2_Nb=L.cmp(dz["Nb"][nt:][notin], obs[nt:][notin])["r2"])
    # (c) kernel structure
    if kernel:
        sq = None if nxt["kind"] == "sgd" else [t.sqrt() for t in Dp]
        Gs = L.gram(P, ctx["Xk"], sqrtD=sq)
        out["c_kernel"] = {}
        for nm, G in Gs.items():
            out["c_kernel"][nm] = L.kernel_stats(G, ctx["yk"], gen)
        del Gs
        torch.cuda.empty_cache()
    return out
