"""Small stats helpers (no pandas/statsmodels in the env). Copied verbatim from reanalysis/stats_util.py,
ae_analysis.py, ae_dense_leadlag.py."""
from __future__ import annotations
import numpy as np
from scipy import stats
from scipy.stats import rankdata

def rank(x):
    return stats.rankdata(x)


def partial_corr(x, y, Z, method="spearman"):
    """Partial correlation of x,y controlling for columns of Z (rank-transformed if spearman)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    Z = np.asarray(Z, float).reshape(len(x), -1)
    if method == "spearman":
        x, y = rank(x), rank(y); Z = np.column_stack([rank(c) for c in Z.T])
    A = np.column_stack([np.ones(len(x)), Z])
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def boot_ci_corr(x, y, n=5000, seed=0, method="pearson", clusters=None):
    """Bootstrap 95% CI of a correlation; if `clusters` given, resample whole clusters."""
    rng = np.random.default_rng(seed)
    x, y = np.asarray(x, float), np.asarray(y, float)
    f = (lambda a, b: stats.pearsonr(a, b)[0]) if method == "pearson" else (lambda a, b: stats.spearmanr(a, b)[0])
    out = []
    if clusters is None:
        idx_sets = [np.arange(len(x))]
        for _ in range(n):
            i = rng.integers(0, len(x), len(x))
            if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
                out.append(f(x[i], y[i]))
    else:
        c = np.asarray(clusters); u = np.unique(c)
        members = {k: np.where(c == k)[0] for k in u}
        for _ in range(n):
            ks = rng.choice(u, len(u))
            i = np.concatenate([members[k] for k in ks])
            if np.ptp(x[i]) > 0 and np.ptp(y[i]) > 0:
                out.append(f(x[i], y[i]))
    return tuple(np.percentile(out, [2.5, 97.5]))


def power_r(n, power=0.8, alpha=0.05):
    """Smallest |r| detectable (two-sided) at given power, via Fisher z approximation."""
    from scipy.stats import norm
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    return float(np.tanh((za + zb) / np.sqrt(n - 3)))


def spearman(a, b):
    """(ae_analysis.py) scipy spearmanr on finite pairs; nan if < 4 points."""
    from scipy.stats import spearmanr
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() >= 4 else float("nan")


def spearman_fast(a, b):
    """(ae_dense_leadlag.py) numpy rank-correlation on finite pairs; nan if < 4 points."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 4: return np.nan
    ra, rb = rankdata(a[ok]), rankdata(b[ok]); ra -= ra.mean(); rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb) + 1e-300))


def aff_resid(Z, V):
    """adjusted residual share of a global affine fit V ~ [Z,1] (n pts, d dims)."""
    n, d = Z.shape; X = np.hstack([Z, np.ones((n, 1))])
    B, *_ = np.linalg.lstsq(X, V, rcond=None); res = ((V - X @ B) ** 2).sum(); tot = ((V - V.mean(0)) ** 2).sum()
    return (res / (n - d - 1)) / (tot / (n - 1) + 1e-300)


def boot_ci(vals, B=4000, seed=0):
    v = np.asarray(vals, float); v = v[np.isfinite(v)]
    if len(v) == 0: return [np.nan, np.nan, np.nan, 0]
    rs = np.random.RandomState(seed); m = [rs.choice(v, len(v)).mean() for _ in range(B)]
    return [float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), int(len(v))]

