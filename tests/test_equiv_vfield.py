"""Old (baseline 71ee3eb) vs new latentdyn.vfield numerics on small synthetic tensors (exact equality). CPU only."""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch

import _oldcode
_oldcode.materialise()
sys.path.insert(0, _oldcode.OLD_DIR)       # old vfield_decomp does `import vfield`
old_V = _oldcode.load_old("vfield")
old_VD = _oldcode.load_old("vfield_decomp")
old_VS = _oldcode.load_old("vfield_spectra")
sys.path.remove(_oldcode.OLD_DIR)
from latentdyn.vfield import core as V, decomp as VD, spectra as VS


def _eq(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys(), path
        for k in a: _eq(a[k], b[k], f"{path}/{k}")
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b), path
        for i, (x, y) in enumerate(zip(a, b)): _eq(x, y, f"{path}[{i}]")
    elif isinstance(a, torch.Tensor):
        assert torch.equal(a, b), path
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b, equal_nan=True), path
    else:
        assert a == b, path


def _data(T=5, n=60, d=4, C=3, seed=0):
    g = np.random.RandomState(seed)
    Z = np.cumsum(0.1 * g.randn(T, n, d), 0) + g.randn(1, n, d)
    y = np.arange(n) % C
    return Z, np.arange(T) * 10, y, np.arange(C)


def test_split_ridge():
    Z, steps, y, cls = _data()
    for s in range(3): _eq(old_V.make_split(y, s), V.make_split(y, s))
    Zt = torch.tensor(Z); D = Zt[1:] - Zt[:-1]
    _eq(old_V.ridge_fit(Zt[:-1], D), V.ridge_fit(Zt[:-1], D))


def test_decomp():
    Z, steps, y, cls = _data()
    Zt = torch.tensor(Z); tr, te = V.make_split(y, 0)
    _eq(old_VD.global_affine_fit(Zt[:-1], Zt[1:], torch.tensor(tr)), VD.global_affine_fit(Zt[:-1], Zt[1:], torch.tensor(tr)))
    _eq(old_VD.generator(Z, steps), VD.generator(Z, steps))
    dev = torch.device("cpu")
    _eq(old_VD.partition(Zt, steps, y, cls, dev, 2), VD.partition(Zt, steps, y, cls, dev, 2))


def test_spectra():
    g = np.random.RandomState(1)
    L = 0.1 * g.randn(4, 3, 4, 4)
    _eq(old_VS.eig_metrics(L, track=True), VS.eig_metrics(L, track=True))
    A = np.eye(4) + 0.05 * g.randn(4, 4)
    _eq(old_VS.logm_safe(A, 10.0), VS.logm_safe(A, 10.0))


if __name__ == "__main__":
    for k, f in list(globals().items()):
        if k.startswith("test_"): f(); print("ok", k)
