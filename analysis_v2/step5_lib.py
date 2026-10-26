"""Helpers for Amendment 4 (review-2 analyses). Import FIRST. Imports step4_lib (threads capped at 2, dataset from --dataset)."""
import step4_lib as L
import os, warnings
import numpy as np
warnings.filterwarnings("ignore")
S = L.S
DATASET = L.DATASET
OUT3 = os.path.join(L.C.OUT_ROOT, "ext3", DATASET)
os.makedirs(OUT3, exist_ok=True)
TARGETS4 = ["acc_15kw", "gap_15kw", "acc_15k", "gap_15k"]
PRIMARY = ["acc_15kw", "gap_15kw"]
CUTS = L.CUTS
write_csv = L.write_csv


def block_of(r):
    if DATASET == "mnist":
        return r["info"]["block"]
    return "A" if "phase_n" in r["info"]["run_id"] else "B"


def get_data(runs, cut, tname):
    y = L.hasher(runs, tname)
    B0, DES, G, g, seeds, names, _ = L.build_all(runs, cut, with_ext=False)
    blk = np.array([block_of(r) for r in runs])
    return y, B0, DES, G, g, seeds, names, blk


# ------------------------------------------------------------ per-config sufficient statistics + fast bootstraps
class CfgStats:
    """Per-config additive stats so that any config-resample's R2 differences / RMSE ratios are vector ops."""
    def __init__(self, y, preds, g, tm=None, grand=None):
        """preds: dict name -> OOF prediction array. tm: per-run training-fold mean (optional). grand: scalar grand mean."""
        self.u = np.unique(g); nc = len(self.u)
        self.cfg_idx = {c: i for i, c in enumerate(self.u)}
        ci = np.array([self.cfg_idx[c] for c in g])
        add = lambda v: np.bincount(ci, weights=v, minlength=nc)
        self.n = add(np.ones(len(y))); self.Sy = add(y); self.Syy = add(y * y)
        self.sse = {k: add((y - p) ** 2) for k, p in preds.items()}
        self.sst_tm = add((y - tm) ** 2) if tm is not None else None
        self.sst_gm = add((y - grand) ** 2) if grand is not None else None
        self.ci = ci; self.nc = nc

    def sst(self, w):
        return w @ self.Syy - (w @ self.Sy) ** 2 / (w @ self.n)

    def draw(self, nb, rng, blocks=None):
        """returns counts matrix nb x nc. blocks: per-config block label (stratified) or None."""
        nc = self.nc
        if blocks is None:
            idx = rng.integers(0, nc, (nb, nc))
        else:
            idx = np.empty((nb, nc), int); pos = 0
            for b in np.unique(blocks):
                mem = np.where(blocks == b)[0]
                idx[:, pos:pos + len(mem)] = mem[rng.integers(0, len(mem), (nb, len(mem)))]; pos += len(mem)
        flat = (idx + np.arange(nb)[:, None] * nc).ravel()
        return np.bincount(flat, minlength=nb * nc).reshape(nb, nc).astype(float)

    def d_r2(self, a, b, W):
        """R2(b)-R2(a) (about resample mean) for each row of W (counts)."""
        n = W @ self.n; Sy = W @ self.Sy; sst = W @ self.Syy - Sy ** 2 / n
        return (W @ self.sse[a] - W @ self.sse[b]) / sst

    def d_r2_alt(self, a, b, W, kind):
        sst = W @ (self.sst_tm if kind == "tm" else self.sst_gm)
        return (W @ self.sse[a] - W @ self.sse[b]) / sst

    def rmse_ratio(self, a, b, W):
        return np.sqrt((W @ self.sse[b]) / (W @ self.sse[a]))


def config_blocks(g, blk):
    u = np.unique(g)
    return np.array([blk[np.where(g == c)[0][0]] for c in u])


def cfg_stats_point(y, pa, pb, g):
    st = CfgStats(y, {"a": pa, "b": pb}, g)
    return st, st.d_r2("a", "b", np.ones((1, st.nc)))[0]


def tost_p(samples, m):
    """one-sided p-values for H0: d <= -m and H0: d >= m (bootstrap), TOST p = max."""
    nb = len(samples)
    p1 = (np.sum(samples <= -m) + 1) / (nb + 1); p2 = (np.sum(samples >= m) + 1) / (nb + 1)
    return max(p1, p2)


# ------------------------------------------------------------ learners
def _std(Xtr, Xte):
    mu, sd = Xtr.mean(0), Xtr.std(0); sd[sd < 1e-12] = 1
    return (Xtr - mu) / sd, (Xte - mu) / sd


def _fold_ids(gtr, k=5):
    ug = np.unique(gtr); k = min(k, len(ug))
    perm = np.random.RandomState(0).permutation(ug)
    fo = {g: i % k for i, g in enumerate(perm)}
    return np.array([fo[g] for g in gtr]), k


GAMMAS = np.array([0.01, 0.03, 0.1, 0.3, 1.0])
KALPHAS = np.logspace(-3, 1, 5)


def _rbf(A, B, gam):
    d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
    return np.exp(-gam * np.maximum(d, 0))


def krr_fit_predict(Xtr, ytr, gtr, Xte):
    Xs, Xt = _std(Xtr, Xte); d = Xs.shape[1]
    ym = ytr.mean(); yc = ytr - ym
    f, k = _fold_ids(gtr)
    sse = np.zeros((len(GAMMAS), len(KALPHAS)))
    for gi, gm in enumerate(GAMMAS / d):
        for j in range(k):
            tr, va = f != j, f == j
            K = _rbf(Xs[tr], Xs[tr], gm); lam, U = np.linalg.eigh(K); uy = U.T @ yc[tr]
            Kv = _rbf(Xs[va], Xs[tr], gm) @ U
            pred = Kv @ (uy[:, None] / (lam[:, None] + KALPHAS[None]))
            sse[gi] += ((pred - yc[va][:, None]) ** 2).sum(0)
    gi, ai = np.unravel_index(np.argmin(sse), sse.shape)
    gm = GAMMAS[gi] / d
    K = _rbf(Xs, Xs, gm)
    w = np.linalg.solve(K + KALPHAS[ai] * np.eye(len(K)), yc)
    return _rbf(Xt, Xs, gm) @ w + ym


def mlp_fit_predict(Xtr, ytr, gtr, Xte, nrest=5):
    from sklearn.neural_network import MLPRegressor
    Xs, Xt = _std(Xtr, Xte); ym, ys = ytr.mean(), ytr.std() + 1e-12
    yz = (ytr - ym) / ys
    ps = []
    for s in range(nrest):
        m = MLPRegressor(hidden_layer_sizes=(16,), early_stopping=True, validation_fraction=0.15, n_iter_no_change=15,
                         max_iter=400, random_state=s, learning_rate_init=0.01)
        m.fit(Xs, yz); ps.append(m.predict(Xt))
    return np.mean(ps, 0) * ys + ym


def pca_block(Gtr, Gte, k):
    mu, sd = Gtr.mean(0), Gtr.std(0); sd[sd < 1e-12] = 1
    Z = (Gtr - mu) / sd; Zt = (Gte - mu) / sd
    U, s, Vt = np.linalg.svd(Z, full_matrices=False)
    return Z @ Vt[:k].T, Zt @ Vt[:k].T


def learner_fn(name):
    return {"ridge": S.ridge_fit_predict, "krr": krr_fit_predict, "mlp": mlp_fit_predict, "gbm": S.gbm_fit_predict}[name]


def gvariant_cols(names_g, variant):
    te4 = [i for i, n in enumerate(names_g) if n[:3] == "te_" and n[3:] in L.COMPACT]
    c8 = L.compact_cols(names_g)
    return {"full": list(range(len(names_g))), "compact4": te4, "compact8": c8}[variant]


def design_matrices(B1, G, names_g, variant, tr, te):
    """returns (Xtr, Xte) for base B1 plus G variant ('none' -> B1 only, 'Gonly' -> full G only)."""
    if variant == "none":
        return B1[tr], B1[te]
    if variant == "Gonly":
        return G[tr], G[te]
    if variant.startswith("pca"):
        k = int(variant[3:]); a, b = pca_block(G[tr], G[te], k)
        return np.hstack([B1[tr], a]), np.hstack([B1[te], b])
    cols = gvariant_cols(names_g, variant)
    return np.hstack([B1[tr], G[tr][:, cols]]), np.hstack([B1[te], G[te][:, cols]])
