"""Old (baseline 71ee3eb) vs new ntk lib/pair on a tiny seeded synthetic model: exact equality. CPU only."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import _oldcode
from latentdyn.ntk import lib as new_L, pair as new_P
from latentdyn.model import LatentProbeModel

_oldcode.materialise()
sys.path.insert(0, _oldcode.OLD_DIR)  # old ntk_pair does `from ntk_lib import *`
import ntk_lib as old_L  # noqa: E402
import ntk_pair as old_P  # noqa: E402
sys.path.remove(_oldcode.OLD_DIR)


def _eq(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a:
            _eq(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)):
            _eq(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b, err_msg=path)
    else:
        assert (a == b) or (a != a and b != b), f"{path}: {a} != {b}"


def _ctx(seed=0, N=300, nte=40, ntr=40):
    g = torch.Generator().manual_seed(seed)
    X = torch.randn(N, 1, 28, 28, generator=g)
    lab = torch.randint(0, 10, (N,), generator=g)
    Xp = torch.randn(nte + ntr, 1, 28, 28, generator=g)
    yp = torch.arange(nte + ntr) % 10
    sel = torch.cat([(yp[:nte] == c).nonzero().flatten()[:3] for c in range(10)])
    pool = np.arange(N)
    return dict(Xfull=X, lab_full=lab, pool_idx=pool, tp_idx=np.arange(ntr) * 3, Xp=Xp, yp=yp, ntest=nte,
                Xk=Xp[sel], yk=yp[sel], Xpool=X, ypool=lab)


def _states(kind, ctx, L):
    torch.manual_seed(0)
    m = LatentProbeModel(16, "mlp").train()
    params = list(m.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4) if kind == "adam" else \
        torch.optim.SGD(params, lr=0.025, momentum=0.9, weight_decay=1e-4)
    rng = np.random.RandomState(1)
    batches = []

    def step():
        idx = rng.choice(len(ctx["pool_idx"]), 32, replace=False)
        bi = torch.as_tensor(idx)
        opt.zero_grad(); F.cross_entropy(m(ctx["Xfull"][bi])[0], ctx["lab_full"][bi]).backward(); opt.step()
        batches.append(idx)
    for _ in range(5):
        step()
    sa = L.make_state(m, opt); step(); sb = L.make_state(m, opt)
    return sa, sb, batches[-1], batches[:-1][::-1]


def _run(kind, L, P):
    ctx = _ctx()
    sa, sb, b, hist = _states(kind, ctx, L)
    return P.analyze_pair(sa, sb, b, ctx, hist_batches=hist, kernel=True), sa, sb, ctx, L


def _case(kind):
    o, sa, sb, ctx, _ = _run(kind, old_L, old_P)
    n, sa2, sb2, _, _ = _run(kind, new_L, new_P)
    _eq(o, n)
    _eq(old_L.decompose(sa, sb), new_L.decompose(sa2, sb2))
    P = sa["params"]
    _eq(old_L.grad_enc(P, ctx["Xpool"], ctx["ypool"], 300), new_L.grad_enc(P, ctx["Xpool"], ctx["ypool"], 300))
    _eq(old_L.z_of(P, ctx["Xp"]), new_L.z_of(P, ctx["Xp"]))
    _eq(old_L.jvp_z(P, ctx["Xp"], P[:6]), new_L.jvp_z(P, ctx["Xp"], P[:6]))


def test_equiv_ntk_adam():
    _case("adam")


def test_equiv_ntk_sgd():
    _case("sgd")


if __name__ == "__main__":
    test_equiv_ntk_adam(); test_equiv_ntk_sgd(); print("OK")
