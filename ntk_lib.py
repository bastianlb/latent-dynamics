"""Tangent-kernel coupling analysis of one optimizer step (library).

Everything is done on the ENCODER only (z = enc(x), latent_norm='none', MLP encoder):
  Delta z_i ~= J_i(theta_t) Delta theta,   J_i = dz_i/dtheta_enc  (16 x P).
The optimizer update is split exactly (see `decompose`):
  Adam (torch, coupled L2):  g_t = grad_batch + wd*theta_{t-1};  m_t = b1 m_{t-1} + (1-b1) g_t
     Delta theta = -(lr/bc1) * (b1 m_{t-1} + (1-b1) g_t) / (sqrt(v_t/bc2)+eps)
                 =  H  +  N_batch  +  W
     H = -(lr/bc1) b1 m_{t-1}/den                (momentum history)
     N_batch = -(lr/bc1)(1-b1) g_batch/den       (new minibatch gradient; D = 1/den is the preconditioner)
     W = -(lr/bc1)(1-b1) wd theta/den            (weight decay in the new-gradient term)
  Drift: N_drift = same with g_batch -> g_full (mean grad over the whole training pool); fluct = N_batch - N_drift.
  SGD (torch, momentum mu): buf_t = mu buf_{t-1} + g_t ; Delta theta = -lr buf_t -> H=-lr mu buf_{t-1}, N=-lr g.
All Delta z components are obtained with forward-mode jvp (no explicit J). Explicit J chunks are only used for
kernel-structure statistics (Gram of D^{1/2}-scaled Jacobians).
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp, jacrev, vmap

torch.set_num_threads(8)
DEV = torch.device(os.environ.get("NTK_DEVICE") or ("cuda:0" if torch.cuda.is_available() else "cpu"))
DATASET = os.environ.get("NTK_DATASET", "mnist")
NENC = 6  # encoder tensors: W1,b1,W2,b2,W3,b3 (first six model.parameters())
MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def enc_fn(p, x):
    h = x.flatten(1)
    h = torch.relu(F.linear(h, p[0], p[1]))
    h = torch.relu(F.linear(h, p[2], p[3]))
    return F.linear(h, p[4], p[5])


def load_mnist_tensors(root, train):
    from torchvision import datasets
    if DATASET == "fashion_mnist":
        ds = datasets.FashionMNIST(root=str(root), train=train, download=False); mean, std = 0.2860, 0.3530
    else:
        ds = datasets.MNIST(root=str(root), train=train, download=False); mean, std = MNIST_MEAN, MNIST_STD
    x = ((ds.data.float() / 255.0 - mean) / std).unsqueeze(1)
    return x, ds.targets.long()


# ---------------------------------------------------------------- state / decomposition
def make_state(model, opt):
    """Normalized snapshot from a live model+torch optimizer (used by selftest)."""
    P = [p.detach().clone() for p in model.parameters()]
    g = opt.param_groups[0]
    st = [opt.state.get(p, {}) for p in model.parameters()]
    if isinstance(opt, torch.optim.Adam):
        m = [s["exp_avg"].clone() if s else torch.zeros_like(p) for s, p in zip(st, P)]
        v = [s["exp_avg_sq"].clone() if s else torch.zeros_like(p) for s, p in zip(st, P)]
        step = int(st[0]["step"]) if st[0] else 0
        return dict(kind="adam", params=P, m=m, v=v, step=step, lr=g["lr"], b1=g["betas"][0], b2=g["betas"][1],
                    eps=g["eps"], wd=g["weight_decay"])
    buf = [s.get("momentum_buffer").clone() if s and s.get("momentum_buffer") is not None else torch.zeros_like(p)
           for s, p in zip(st, P)]
    return dict(kind="sgd", params=P, m=buf, step=0, lr=g["lr"], mu=g["momentum"], wd=g["weight_decay"])


def decompose(prev, nxt):
    """Return dict with Dp (preconditioner, enc tensors), kn (scalar on new-gradient term), H (enc tensors),
    g_rec (reconstructed total gradient incl. wd, enc tensors), dtheta (actual enc update)."""
    sl = slice(0, NENC)
    th0 = prev["params"][sl]
    if nxt["kind"] == "adam":
        s = nxt["step"]; b1, b2, eps, lr = nxt["b1"], nxt["b2"], nxt["eps"], nxt["lr"]
        bc1, bc2 = 1 - b1 ** s, 1 - b2 ** s
        den = [v.sqrt() / math.sqrt(bc2) + eps for v in nxt["v"][sl]]
        Dp = [1.0 / d for d in den]
        c = lr / bc1
        kn = c * (1 - b1)
        H = [-c * b1 * m * D for m, D in zip(prev["m"][sl], Dp)]
        g_rec = [(mn - b1 * mp) / (1 - b1) for mn, mp in zip(nxt["m"][sl], prev["m"][sl])]
    else:
        mu, lr = nxt["mu"], nxt["lr"]
        Dp = [torch.ones_like(t) for t in th0]
        kn = lr
        H = [-lr * mu * b for b in prev["m"][sl]]
        g_rec = [bn - mu * bp for bn, bp in zip(nxt["m"][sl], prev["m"][sl])]
    dtheta = [a - b for a, b in zip(nxt["params"][sl], th0)]
    return dict(Dp=Dp, kn=kn, H=H, g_rec=g_rec, dtheta=dtheta, wd=nxt["wd"], th0=th0)


# ---------------------------------------------------------------- gradients / jvp
def grad_enc(P, X, y, denom, chunk=8192):
    """Gradient wrt encoder params of sum_j CE_j / denom (head from P[6:8], detached)."""
    pe = [t.detach().clone().requires_grad_(True) for t in P[:NENC]]
    Wh, bh = P[6].detach(), P[7].detach()
    acc = [torch.zeros_like(t) for t in pe]
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk].to(DEV); yb = y[s:s + chunk].to(DEV)
        loss = F.cross_entropy(F.linear(enc_fn(pe, xb), Wh, bh), yb, reduction="sum") / denom
        gs = torch.autograd.grad(loss, pe)
        for a, g in zip(acc, gs):
            a += g
    return acc


def z_of(P, X, chunk=4096):
    with torch.no_grad():
        return torch.cat([enc_fn(P[:NENC], X[s:s + chunk]) for s in range(0, len(X), chunk)])


def jvp_z(P, X, tang, chunk=2000):
    """Delta z = J(theta) tang for samples X. Returns [n,16]."""
    pe = tuple(t.detach() for t in P[:NENC]); tg = tuple(tang)
    out = []
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        _, dz = jvp(lambda *p: enc_fn(p, xb), pe, tg)
        out.append(dz)
    return torch.cat(out)


def scale(vec, a):
    return [a * t for t in vec]


def mul(a, b):
    return [x * y for x, y in zip(a, b)]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def sub(a, b):
    return [x - y for x, y in zip(a, b)]


def dot(a, b):
    return float(sum((x * y).sum() for x, y in zip(a, b)))


# ---------------------------------------------------------------- metrics
def cmp(C, obs, sl=slice(None)):
    c, o = C[sl].double(), obs[sl].double()
    on = (o ** 2).sum()
    cn = (c ** 2).sum()
    cosg = float((c * o).sum() / (cn.sqrt() * on.sqrt() + 1e-30))
    cs = F.cosine_similarity(c, o, dim=1, eps=1e-30)
    return dict(energy=float(cn / on), r2=float(1 - ((o - c) ** 2).sum() / on), cos=cosg, cos_mean=float(cs.mean()))


def affine_sse(Z, Y, groups=None):
    """SSE of least-squares affine fit Y ~ [Z,1] (globally, or within each group)."""
    A = torch.cat([Z, torch.ones(len(Z), 1, device=Z.device, dtype=Z.dtype)], 1).double()
    Yd = Y.double()
    if groups is None:
        groups = torch.zeros(len(Z), dtype=torch.long, device=Z.device)
    sse = 0.0
    for g in groups.unique():
        mk = groups == g
        a, y = A[mk], Yd[mk]
        sol = torch.linalg.lstsq(a.cpu(), y.cpu(), driver="gelsd").solution.to(a.device)
        sse += float(((y - a @ sol) ** 2).sum())
    return sse


def affine_levels(Z, Y, labels, gen):
    """Energy fraction explained by global affine, class-wise affine, and shuffled-label class-wise control."""
    E = max(float((Y.double() ** 2).sum()), 1e-30)
    sg = affine_sse(Z, Y)
    sc = affine_sse(Z, Y, labels)
    perm = labels[torch.randperm(len(labels), generator=gen).to(labels.device)]
    sp = affine_sse(Z, Y, perm)
    return dict(glob=1 - sg / E, cls=1 - sc / E, cls_shuf=1 - sp / E,
                cls_share_of_resid=(sg - sc) / max(sg, 1e-30), shuf_share_of_resid=(sg - sp) / max(sg, 1e-30))


# ---------------------------------------------------------------- Jacobian chunks + kernel stats
def jac_chunk(P, x):
    """[n*16, Pcount] rows ordered (sample, out_dim)."""
    pe = tuple(t.detach() for t in P[:NENC])
    f = lambda p, xi: enc_fn(p, xi[None])[0]
    J = vmap(jacrev(f), in_dims=(None, 0))(pe, x)
    n = x.shape[0]
    return torch.cat([j.reshape(n * 16, -1) for j in J], 1)


def gram(P, X, sqrtD=None, chunk=80, rb=2048):
    """Gram matrix [(n*16),(n*16)] of J (optionally J*sqrtD) for samples X."""
    n = X.shape[0]
    Pn = sum(t.numel() for t in P[:NENC])
    Jb = torch.empty(n * 16, Pn, device=DEV, dtype=torch.float32)
    sq = None if sqrtD is None else torch.cat([t.reshape(-1) for t in sqrtD])
    for s in range(0, n, chunk):
        Jb[s * 16:(s + chunk) * 16] = jac_chunk(P, X[s:s + chunk])
    out = {}
    G = torch.empty(n * 16, n * 16, device=DEV)
    for r in range(0, n * 16, rb):
        G[r:r + rb] = Jb[r:r + rb] @ Jb.T
    out["raw"] = G
    if sq is not None:
        Jb *= sq[None, :]
        G2 = torch.empty(n * 16, n * 16, device=DEV)
        for r in range(0, n * 16, rb):
            G2[r:r + rb] = Jb[r:r + rb] @ Jb.T
        out["pre"] = G2
    del Jb
    torch.cuda.empty_cache()
    return out


def erank(s):
    p = s / s.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.exp(-(p * torch.log(p.clamp_min(1e-30))).sum(-1))


def kernel_stats(G, y, gen, npairs=3000):
    """G [(n*16),(n*16)]; y [n] class labels (LongTensor on DEV)."""
    n = len(y)
    Gr = G.view(n, 16, n, 16).double()
    F2 = (Gr ** 2).sum((1, 3))
    Tr = torch.einsum("ikjk->ij", Gr)
    Fro = F2.sqrt()
    eye = torch.eye(n, dtype=torch.bool, device=G.device)
    same = (y[:, None] == y[None, :])
    wi, bt = same & ~eye, ~same
    dg = Fro.diag()
    Fn = Fro / (dg[:, None] * dg[None, :]).sqrt()
    out = dict(
        fro_within=float(Fro[wi].mean()), fro_between=float(Fro[bt].mean()),
        fro_diag=float(dg.mean()),
        fro_ratio_wb=float(Fro[wi].mean() / Fro[bt].mean()),
        cosn_within=float(Fn[wi].mean()), cosn_between=float(Fn[bt].mean()),
        trace_within=float(Tr[wi].mean()), trace_between=float(Tr[bt].mean()),
    )
    # kernel-target alignment (scalar kernels: trace kernel, normalized Frobenius kernel)
    Y = F.one_hot(y, 10).double()
    L = Y @ Y.T
    H = torch.eye(n, device=G.device, dtype=torch.double) - 1.0 / n

    def kta(S, L, center):
        S = 0.5 * (S + S.T)
        if center:
            S, L = H @ S @ H, H @ L @ H
        return float((S * L).sum() / (S.norm() * L.norm()))
    for nm, S in (("tr", Tr), ("fn", Fn)):
        out[f"kta_{nm}"] = kta(S, L, False)
        out[f"kta_{nm}_c"] = kta(S, L, True)
    # class x class mean Frobenius (cosine-normalized) matrix
    Oh = F.one_hot(y, 10).double()
    cnt = Oh.sum(0)
    CM = (Oh.T @ Fn @ Oh) / (cnt[:, None] * cnt[None, :])
    out["class_matrix_cosn"] = CM.cpu().numpy().tolist()
    # block-level statistics: effective rank, isotropy, coherence
    B = Gr.permute(0, 2, 1, 3)  # [n,n,16,16]
    ii = torch.arange(n, device=G.device)
    diag_blocks = B[ii, ii]

    def sample_pairs(mask, m):
        idx = mask.nonzero()
        sel = idx[torch.randperm(len(idx), generator=gen)[:m].to(G.device)]
        return B[sel[:, 0], sel[:, 1]]
    groups = dict(diag=diag_blocks, within=sample_pairs(wi, npairs), between=sample_pairs(bt, npairs))
    for nm, blk in groups.items():
        sv = torch.linalg.svdvals(blk)
        e = erank(sv)
        tr = torch.einsum("bkk->b", blk)
        fr2 = (blk ** 2).sum((1, 2))
        iso_energy = float((tr ** 2 / 16).sum() / fr2.sum())  # energy share of best isotropic fit (pooled)
        sym = 0.5 * (blk + blk.transpose(1, 2))
        out[f"erank_{nm}"] = float(e.mean())
        out[f"iso_{nm}"] = iso_energy
        out[f"top1_{nm}"] = float((sv[:, 0] ** 2).sum() / (sv ** 2).sum())
        out[f"coher_{nm}"] = float(blk.mean(0).norm() / blk.flatten(1).norm(dim=1).mean())
        mb = blk.mean(0)
        out[f"erank_meanblock_{nm}"] = float(erank(torch.linalg.svdvals(mb)))
    # K ~ k(x_i,x_j) I check on whole matrix: best scalar-isotropic fit energy over all pairs (pooled by group)
    out["iso_all_offdiag"] = float(((Tr ** 2 / 16)[~eye]).sum() / F2[~eye].sum())
    out["iso_all_within"] = float(((Tr ** 2 / 16)[wi]).sum() / F2[wi].sum())
    out["iso_all_between"] = float(((Tr ** 2 / 16)[bt]).sum() / F2[bt].sum())
    return out


# ---------------------------------------------------------------- exact batch reconstruction
def true_batches(seed, pool_indices, B, total_steps, shuffle_workers=False):
    """Replay the training shuffle RNG (train.py: seed_everything -> model init -> per epoch: DataLoader iterator
    base_seed draw, RandomSampler seed draw -> randperm) to recover the TRUE minibatch (full-split indices) of every
    optimizer step 1..total_steps. Needed because the dense_latent `batch_idx` log is overwritten by snapshot full
    passes (RecordingRandomSampler.last_perm) for much of early training."""
    from model import LatentProbeModel
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    LatentProbeModel(16, 'mlp')  # consumes the same CPU RNG as train.py's model construction
    N = len(pool_indices); pool_indices = np.asarray(pool_indices)
    out, step = {}, 0
    while step < total_steps:
        torch.empty((), dtype=torch.int64).random_()          # _BaseDataLoaderIter._base_seed
        sd = int(torch.empty((), dtype=torch.int64).random_().item())  # RandomSampler seed
        g = torch.Generator(); g.manual_seed(sd)
        perm = torch.randperm(N, generator=g).numpy()
        for i in range(0, N, B):
            step += 1
            out[step] = pool_indices[perm[i:i + B]].astype(np.int64)
            if step >= total_steps:
                break
    return out
