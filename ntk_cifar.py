"""CIFAR-10 ResNet-18 (BN, augmentation, SGD momentum, cosine lr) tangent-kernel coupling + train/test contrast.
Outputs live in a 16-d PCA latent frame (vfield pca16 recipe). Exact augmented-batch replay is not possible, so:
  * total new-gradient term N_batch is taken EXACTLY from the checkpoint pair: g_t = buf_{t+1} - mu*buf_t - wd*theta_t;
  * drift = mean gradient over M fresh-augmented random batches (train-mode BN); fluct = N_batch - drift;
  * self / same-class / other-class splits use PROXY gradients: the logged batch, fresh augmentation, train-mode BN.
  * BN running-stat changes are a separate tangent component (Bbuf).
usage: ntk_cifar.py <run_id> [pair_start ...]"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "8")
import sys, json, glob, time, copy
import numpy as np, torch, torch.nn.functional as F
from torch.func import functional_call, jvp, jacrev, vmap
import ntk_lib as L
from model import LatentProbeModel
import ntk_tt

DL = '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent'
OUT = os.environ.get('NTK_OUT', '/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/ntk_coupling/traintest/cifar')
ROOT = '/vol/bitbucket/lbastian/latent-dynamics/runs_v2/data'
MEAN = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1); STD = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)
DEV = L.DEV
M_DRIFT = int(os.environ.get('NTK_MDRIFT', 16)); SKETCH = int(os.environ.get('NTK_SKETCH', 65536)); KPC = int(os.environ.get('NTK_KPC', 20))


def load_cifar(train):
    from torchvision import datasets
    ds = datasets.CIFAR10(root=ROOT, train=train, download=False)
    x = torch.from_numpy(ds.data).permute(0, 3, 1, 2).float() / 255.0
    return ((x - MEAN) / STD), torch.tensor(ds.targets).long()


def pca_basis(ra, k=16):
    rows = []
    for probe in ('latents', 'latents_train'):
        fs = sorted(glob.glob(f'{ra}/{probe}/step_*.npz'))
        for f in fs[::2]:
            z = np.load(f)['latents'].astype(np.float64); z = z - z.mean(0, keepdims=True)
            rows.append((z / (np.sqrt((z ** 2).sum(1).mean()) + 1e-30)).astype(np.float32))
    X = np.concatenate(rows, 0); C = X.T @ X / len(X); w, v = np.linalg.eigh(C)
    return torch.from_numpy(v[:, ::-1][:, :k].copy()).float(), float(w[::-1][:k].sum() / w.sum())


def make_ctx(rid):
    ra = f'{DL}/_run_artifacts/runs/{rid}'
    Xtr, ytr = load_cifar(True); Xte, yte = load_cifar(False)
    pi = np.load(f'{ra}/probe_index.npz'); pt = np.load(f'{ra}/probe_train_index.npz'); ti = np.load(f'{ra}/train_index.npz')
    pool_idx = ti['indices'].astype(np.int64)
    pool_lab = (ti['noisy_labels'] if 'noisy_labels' in ti.files else ti['labels']).astype(np.int64)
    lab_full = -np.ones(len(Xtr), dtype=np.int64); lab_full[pool_idx] = pool_lab
    tp_idx = pt['indices'].astype(np.int64)
    Xp = torch.cat([Xte[pi['indices']], Xtr[tp_idx]]).to(DEV); yp = torch.cat([yte[pi['indices']], ytr[tp_idx]]).to(DEV)
    P, var = pca_basis(ra)
    return dict(Xfull=Xtr.to(DEV), lab_full=torch.as_tensor(lab_full).to(DEV), pool_idx=pool_idx, pool_lab=torch.as_tensor(pool_lab).to(DEV),
                tp_idx=tp_idx, Xp=Xp, yp=yp, ntest=len(pi['indices']), P=P.to(DEV), pca_var=var)


def augment(x):
    """random crop (pad 4, zero pixel = -mean/std in normalised space) + horizontal flip, per sample."""
    n = x.shape[0]; fill = (-MEAN / STD).to(x.device)
    xp = fill.expand(n, 3, 40, 40).clone(); xp[:, :, 4:36, 4:36] = x
    i = torch.randint(0, 9, (n,), device=x.device); j = torch.randint(0, 9, (n,), device=x.device)
    ar = torch.arange(32, device=x.device)
    rows = (i[:, None] + ar[None]); cols = (j[:, None] + ar[None])
    out = xp[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None], rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < 0.5
    return torch.where(flip[:, None, None, None], out.flip(3), out)


class EvalBN(torch.nn.Module):
    """eval-mode BN written with plain ops so forward-mode AD works w.r.t. running stats too."""
    def __init__(self, bn):
        super().__init__()
        self.weight = torch.nn.Parameter(bn.weight.detach().clone()); self.bias = torch.nn.Parameter(bn.bias.detach().clone())
        self.register_buffer('running_mean', bn.running_mean.clone()); self.register_buffer('running_var', bn.running_var.clone()); self.eps = bn.eps

    def forward(self, x):
        sc = self.weight / torch.sqrt(self.running_var + self.eps)
        return x * sc[None, :, None, None] + (self.bias - self.running_mean * sc)[None, :, None, None]


def swap_bn(mod):
    for n, ch in mod.named_children():
        if isinstance(ch, torch.nn.BatchNorm2d): setattr(mod, n, EvalBN(ch))
        else: swap_bn(ch)


class Net:
    def __init__(self):
        self.model = LatentProbeModel(512, 'resnet18', in_channels=3, img_size=32).to(DEV).eval()
        self.enc_eval = copy.deepcopy(self.model.encoder).eval(); swap_bn(self.enc_eval)
        self.pnames = [n for n, _ in self.model.named_parameters()]
        self.enc_names = [n for n in self.pnames if n.startswith('encoder.')]


def load_state(net, path):
    c = torch.load(path, map_location='cpu', weights_only=False)
    sd = {k: v.to(DEV) for k, v in c['model'].items()}
    g = c['optimizer']['param_groups'][0]; st = c['optimizer']['state']
    params = {n: sd[n].clone() for n in net.pnames}
    bufs = {k: v.clone() for k, v in sd.items() if k.startswith('encoder.') and ('running_mean' in k or 'running_var' in k)}
    mom = {n: (st[i]['momentum_buffer'].to(DEV).clone() if i in st and st[i].get('momentum_buffer') is not None else torch.zeros_like(params[n])) for i, n in enumerate(net.pnames)}
    return dict(params=params, bufs=bufs, mom=mom, lr=g['lr'], mu=g['momentum'], wd=g['weight_decay'], sd=sd, step=c['step'])


def z16_fn(net, P):
    enc = net.enc_eval

    def f(params, bufs, x):
        d = {**{k[8:]: v for k, v in params.items()}, **{k[8:]: v for k, v in bufs.items()}}
        return functional_call(enc, d, (x,)) @ P
    return f


def enc_sub(st, net):
    return {n: st['params'][n] for n in net.enc_names}


def jvp_z(f, st, net, X, tp, tb, chunk=250):
    p = enc_sub(st, net); b = st['bufs']
    out = []
    for s in range(0, len(X), chunk):
        xb = X[s:s + chunk]
        _, dz = jvp(lambda pp, bb: f(pp, bb, xb), (p, b), (tp, tb))
        out.append(dz)
    return torch.cat(out)


def z_of(f, st, net, X, chunk=500):
    p = enc_sub(st, net)
    with torch.no_grad():
        return torch.cat([f(p, st['bufs'], X[s:s + chunk]) for s in range(0, len(X), chunk)])


def batch_grads(net, st, X, y, B, groups=None, selfidx=None):
    """Train-mode BN forward of the whole batch (copy of model, prev state); returns dict of gradient lists:
    'all' (mean CE), per-class ('c',c) = grad of sum_{j in class c} l_j / B, and ('s',j) = grad of l_j / B."""
    m = copy.deepcopy(net.model).train()
    m.load_state_dict(st['sd'], strict=True); m.train()
    ep = [p for n, p in m.named_parameters() if n.startswith('encoder.')]
    logits = m(X)[0]; l = F.cross_entropy(logits, y, reduction='none')
    res = {}
    res['all'] = torch.autograd.grad(l.sum() / B, ep, retain_graph=True)
    if groups:
        for c in range(10):
            mk = (y == c)
            if mk.any(): res[('c', c)] = torch.autograd.grad(l[mk].sum() / B, ep, retain_graph=True)
    for j in (selfidx or []):
        res[('s', j)] = torch.autograd.grad(l[j] / B, ep, retain_graph=True)
    del m
    return {k: dict(zip(net.enc_names, v)) for k, v in res.items()}


def tang(net, st, g, factor):
    return {n: factor * g[n] for n in net.enc_names}


def sketch_gram(f, st, net, X, m, seed=0, chunk=4):
    n = X.shape[0]; p = enc_sub(st, net); b = st['bufs']
    gen = torch.Generator(device=DEV).manual_seed(seed)
    S = torch.zeros(n * 16, m, device=DEV)
    hs = {}
    for k, v in p.items():
        hs[k] = (torch.randint(0, m, (v.numel(),), device=DEV, generator=gen), (torch.randint(0, 2, (v.numel(),), device=DEV, generator=gen) * 2 - 1).float())
    f1 = lambda pp, xi: f(pp, b, xi[None])[0]
    for s in range(0, n, chunk):
        J = vmap(jacrev(f1), in_dims=(None, 0))(p, X[s:s + chunk])   # dict name -> [c,16,*shape]
        c = J[next(iter(J))].shape[0]
        for k, jv in J.items():
            h, sg = hs[k]
            S[s * 16:(s + c) * 16].index_add_(1, h, jv.reshape(c * 16, -1) * sg[None])
        del J
    return S @ S.T


def select_set(ctx, noisy):
    nt = ctx['ntest']; yp = ctx['yp'].cpu().numpy()
    given_tp = ctx['lab_full'][torch.as_tensor(ctx['tp_idx']).to(DEV)].cpu().numpy(); true_tr = yp[nt:]
    te = np.concatenate([np.nonzero(yp[:nt] == c)[0][:KPC] for c in range(10)])
    if noisy:
        mis = np.nonzero(given_tp != true_tr)[0]; cl = np.nonzero(given_tp == true_tr)[0]
        tr = np.concatenate([mis[:5 * KPC], cl[:5 * KPC]])
    else:
        tr = np.concatenate([np.nonzero(true_tr == c)[0][:KPC] for c in range(10)])
    idx = np.concatenate([te, nt + tr]); is_tr = np.concatenate([np.zeros(len(te), bool), np.ones(len(tr), bool)])
    return idx, is_tr, yp[idx], np.concatenate([yp[te], given_tp[tr]])


def sums(parts, obs, nb, gm):
    n = int(gm.sum())
    if n == 0: return None
    o = obs[gm].double(); b = nb[gm].double()
    r = dict(n=n, obs2=float((o ** 2).sum()), Nb2=float((b ** 2).sum()), Nb_obs=float((b * o).sum()))
    for pn, pv in parts.items():
        p = pv[gm].double(); r[pn] = dict(p2=float((p ** 2).sum()), p_obs=float((p * o).sum()), p_Nb=float((p * b).sum()))
    return r


def main():
    rid = sys.argv[1]; only = [int(a) for a in sys.argv[2:] if a.isdigit()]
    d = f'{DL}/{rid}'; meta = json.load(open(f'{d}/meta.json')); cs = meta['ckpt_steps']
    bidx = {}
    for fch in sorted(glob.glob(f'{d}/chunk_*.npz')):
        z = np.load(fch)
        for s, b in zip(z['steps'], z['batch_idx']): bidx[int(s)] = b[b >= 0].astype(np.int64)
        del z
    ctx = make_ctx(rid); net = Net(); f = z16_fn(net, ctx['P'])
    noisy = 'ln0.2' in rid
    kidx, is_tr, ktrue, kgiven = select_set(ctx, noisy)
    Xk = ctx['Xp'][torch.as_tensor(kidx).to(DEV)]
    od = f'{OUT}/{rid}'; os.makedirs(od, exist_ok=True)
    gen = torch.Generator().manual_seed(0)
    pairs = [s for s in cs if s + 1 in set(cs) and s + 1 <= meta['total_steps']]
    if only: pairs = [s for s in pairs if s in only]
    nt = ctx['ntest']; Xp = ctx['Xp']; yp = ctx['yp']
    tp_given = ctx['lab_full'][torch.as_tensor(ctx['tp_idx']).to(DEV)]
    for s in pairs:
        fn = f'{od}/cf_{s:07d}.json'
        if os.path.exists(fn) and not only: continue
        t0 = time.time(); r = dict(step=s + 1, run=rid)
        prev = load_state(net, f'{d}/ckpt/step_{s:07d}.pt'); nxt = load_state(net, f'{d}/ckpt/step_{s + 1:07d}.pt')
        lr, mu, wd = prev['lr'], prev['mu'], prev['wd']
        en = net.enc_names
        z0 = z_of(f, prev, net, Xp); z1 = z_of(f, nxt, net, Xp); obs = z1 - z0
        # exact decomposition from checkpoints
        H = {n: -lr * mu * prev['mom'][n] for n in en}
        G = {n: nxt['mom'][n] - mu * prev['mom'][n] for n in en}                    # total gradient incl wd
        gb = {n: G[n] - wd * prev['params'][n] for n in en}                          # data-gradient (batch) exactly
        Wd = {n: -lr * wd * prev['params'][n] for n in en}
        Nb = {n: -lr * gb[n] for n in en}
        dth = {n: nxt['params'][n] - prev['params'][n] for n in en}
        rec = {n: H[n] + Nb[n] + Wd[n] for n in en}
        num = sum(float((rec[n] - dth[n]).pow(2).sum()) for n in en); den = sum(float(dth[n].pow(2).sum()) for n in en)
        r['update_recon_relerr'] = (num / max(den, 1e-30)) ** .5
        dbuf = {k: nxt['bufs'][k] - prev['bufs'][k] for k in prev['bufs']}
        zero_b = {k: torch.zeros_like(v) for k, v in prev['bufs'].items()}
        zero_p = {n: torch.zeros_like(prev['params'][n]) for n in en}
        T = dict(lin=(dth, dbuf), H=(H, zero_b), Nb=(Nb, zero_b), W=(Wd, zero_b), Bbuf=(zero_p, dbuf))
        dz = {k: jvp_z(f, prev, net, Xp, tp, tb) for k, (tp, tb) in T.items()}
        # drift estimate: mean gradient over M fresh augmented random batches
        gd = {n: torch.zeros_like(prev['params'][n]) for n in en}
        B = len(bidx[s + 1]); poolt = torch.as_tensor(ctx['pool_idx']).to(DEV)
        for _ in range(M_DRIFT):
            sel = poolt[torch.randperm(len(poolt), device=DEV)[:128]]
            gg = batch_grads(net, prev, augment(ctx['Xfull'][sel]), ctx['lab_full'][sel], 128)['all']
            for n in en: gd[n] += gg[n] / M_DRIFT
        Nd = {n: -lr * gd[n] for n in en}
        dz['Nd'] = jvp_z(f, prev, net, Xp, Nd, zero_b)
        dz['Nf'] = dz['Nb'] - dz['Nd']; dz['newb'] = dz['Nb'] + dz['W']; dz['det'] = dz['H'] + dz['Nd'] + dz['W']
        dz['Hsum'] = dz['H'] + dz['Nb'] + dz['W'] + dz['Bbuf']
        sls = dict(test=slice(0, nt), trainprobe=slice(nt, None))
        r['ab'] = {sn: {k: L.cmp(v, obs, sl) for k, v in dz.items()} for sn, sl in sls.items()}
        r['step_info'] = dict(lr=lr, obs_rms=float((obs ** 2).sum(1).mean().sqrt()), z_rms=float((z0 ** 2).sum(1).mean().sqrt()))
        r['d_affine'] = {sn: {k: L.affine_levels(z0[sl], (obs if k == 'obs' else dz[k])[sl], yp[sl], gen) for k in ('obs', 'lin', 'Nd', 'Nf', 'Nb', 'H')} for sn, sl in sls.items()}
        # batch proxies
        bi = torch.as_tensor(bidx[s + 1]).to(DEV); Xb = augment(ctx['Xfull'][bi]); yb = ctx['lab_full'][bi]
        in_b = np.nonzero(np.isin(ctx['tp_idx'], bidx[s + 1]))[0]
        pos = {int(g): k for k, g in enumerate(bidx[s + 1])}
        selfpos = [pos[int(ctx['tp_idx'][j])] for j in in_b]
        pg = batch_grads(net, prev, Xb, yb, B, groups=True, selfidx=selfpos)
        pall = pg['all']
        num = sum(float((pall[n] * gb[n]).sum()) for n in en); a = sum(float(pall[n].pow(2).sum()) for n in en) ** .5; b_ = sum(float(gb[n].pow(2).sum()) for n in en) ** .5
        r['proxy_cos_batchgrad'] = num / (a * b_ + 1e-30); r['proxy_norm_ratio'] = a / (b_ + 1e-30)
        # scale proxies to true batch gradient norm (keeps direction info, fixes bf16/augment scale)
        dzc = torch.zeros(10, len(Xp), 16, device=DEV)
        for c in range(10):
            if ('c', c) in pg: dzc[c] = jvp_z(f, prev, net, Xp, tang(net, prev, pg[('c', c)], -lr), zero_b)
        tot = dzc.sum(0)
        cls = torch.cat([yp[:nt], tp_given]); same = dzc[cls, torch.arange(len(cls))]
        selfz = torch.zeros_like(obs)
        for j, sp in zip(in_b, selfpos):
            selfz[nt + int(j)] = jvp_z(f, prev, net, Xp[nt + int(j)][None], tang(net, prev, pg[('s', sp)], -lr), zero_b)[0]
        same_ex = same - selfz; other = tot - same
        inb = torch.zeros(len(Xp), dtype=torch.bool, device=DEV); inb[nt + torch.as_tensor(in_b, dtype=torch.long).to(DEV)] = True
        istr = torch.arange(len(Xp), device=DEV) >= nt
        # compare the proxy total to the exact N_batch push (in z): how much of the true Nb push do proxies reproduce?
        r['proxy_vs_true_Nb'] = L.cmp(tot, dz['Nb'])
        parts = dict(self=selfz, same=same_ex, other=other, Nb=dz['Nb'], proxy_tot=tot)
        r['drive'] = {gname: sums(parts, obs, dz['Nb'], gm) for gname, gm in dict(test=~istr, train_out=istr & ~inb, train_in=inb).items()}
        del pg, pall
        # kernel (count-sketch Gram, raw eNTK, PCA16 outputs)
        G = sketch_gram(f, prev, net, Xk, SKETCH)
        r['kernel'] = ntk_tt.kernel_blocks(G, is_tr, ktrue, kgiven, gen)
        te_m = torch.as_tensor(~is_tr).to(DEV); ti_ = te_m.nonzero().flatten()
        Gt = G.view(len(kidx), 16, len(kidx), 16)[ti_][:, :, ti_].reshape(len(ti_) * 16, len(ti_) * 16)
        r['kernel_test_stats'] = {k: v for k, v in L.kernel_stats(Gt, torch.as_tensor(ktrue[~is_tr]).to(DEV), gen).items() if k != 'class_matrix_cosn'}
        del G, Gt; torch.cuda.empty_cache()
        r['mem_gb'] = torch.cuda.max_memory_allocated() / 1e9
        json.dump(r, open(fn, 'w'))
        print(rid, s, f'{time.time() - t0:.1f}s mem {r["mem_gb"]:.1f}GB lin_r2 {r["ab"]["test"]["lin"]["r2"]:.4f} recon {r["update_recon_relerr"]:.1e} proxy_cos {r["proxy_cos_batchgrad"]:.2f}', flush=True)


if __name__ == '__main__':
    main()
