"""Validate the Hutchinson (K Gaussian output probes) pixel-output trace kernel against the exact 784-output kernel on a small subset."""
import os, sys, math
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_ntk as N
from torch.func import vmap, grad
rid = sys.argv[1]; step = int(sys.argv[2]); n = 24
rd = f"{N.ROOT}/runs/{rid}"; kind, seed, Nn, wd = N.parse_rid(rid)
net = N.Net(N.build_model(kind)); c1 = N.load_ck(f"{rd}/dense/ckpt/step_{step+1:07d}.pt"); c0 = N.load_ck(f"{rd}/dense/ckpt/step_{step:07d}.pt")
p0 = N.pdict(net, c0); v1, s1 = N.adam_vecs(net, c1, "exp_avg_sq")
sD = {k: (1.0 / (v1[k].sqrt() / math.sqrt(1 - 0.999 ** s1) + 1e-8)).sqrt() for k in net.names}
sDe, sDd = N.blocks(net, sD)
import ae_analysis as A
X = torch.tensor(A.images("mnist", True)[0][:n]).to(N.DEV)
def kern(probes):
    T = torch.zeros(n, n, device=N.DEV)
    for s in probes:
        Fs = vmap(grad(lambda p, x: (net.rec(p, x) * s).sum()), in_dims=(None, 0))(p0, X)
        Fe, Fd = N.blocks(net, Fs, n); F = torch.cat([Fe * sDe, Fd * sDd], 1)
        T += F @ F.T
    return T.cpu().numpy()
ex = kern(torch.eye(784, device=N.DEV))
g = torch.Generator(device=N.DEV); g.manual_seed(0)
for K in (16, 64, 128):
    est = kern(torch.randn(K, 784, generator=g, device=N.DEV)) / K
    off = ~np.eye(n, dtype=bool)
    dc = np.corrcoef(np.diag(ex), np.diag(est))[0, 1]; dr = np.median(np.abs(np.diag(est) / np.diag(ex) - 1))
    cex = ex / np.sqrt(np.outer(np.diag(ex), np.diag(ex))); ces = est / np.sqrt(np.outer(np.diag(est), np.diag(est)))
    print(f"K={K}: diag corr {dc:.4f}, diag median rel err {dr:.3f}; offdiag cosine: corr {np.corrcoef(cex[off], ces[off])[0,1]:.3f}, mean exact {cex[off].mean():.4f} est {ces[off].mean():.4f}, rmse {np.sqrt(((cex-ces)[off]**2).mean()):.4f}")
