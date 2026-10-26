"""Per-snapshot latent-geometry metrics (pure numpy/sklearn, CPU).

per_probe(Z, y, g, nn_pix, prev_nn, init_nn, recon_err) -> dict of floats (keys, stored as `<probe>/<key>`):
  mpd_raw, norm_mean | pr_dim, top2_var, twonn_dim, local_pr_dim | knn_{label,pix}_acc, ncm_{label,pix}_acc, fisher_{label,pix},
  sil_{label,pix}, kmeans_nmi_{label,pix} | knn_overlap_{pixel,prev,init} | knn_dist_cv, hubness_skew, nn1_ratio | rho_err_knn (needs recon_err).
  Scale-dependent items (silhouette, k-means) use Zn = normalize_latents(Z, "mean_pairwise"). Returns also the latent kNN index under key "_nn".
cross(Zte, Ztr, err_te) -> dict (stored as `cross/<key>`): test_to_train_nn, train_to_test_nn, rho_test_err_train_nn (jointly scaled by train-probe mean pairwise distance).
Undefined entries are NaN. k = 10, or max(3, n//25) if n < 200.
"""
import numpy as np
from scipy.stats import skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi_score, silhouette_score
from latentdyn import metrics as M

NAN = float("nan")


def kk(n): return 10 if n >= 200 else max(3, n // 25)


def sqd(X, Y=None):
    Y = X if Y is None else Y
    return np.maximum((X ** 2).sum(1)[:, None] + (Y ** 2).sum(1)[None] - 2 * X @ Y.T, 0)


def knn_sorted(Z, k):
    """(idx, dist) of the k nearest neighbours (self excluded), ascending distance."""
    D = sqd(Z); np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, k, axis=1)[:, :k]; d = np.take_along_axis(D, idx, 1); o = np.argsort(d, 1)
    return np.take_along_axis(idx, o, 1), np.sqrt(np.take_along_axis(d, o, 1))


def overlap(a, b): return float(np.mean([len(set(x) & set(y)) / len(x) for x, y in zip(a, b)]))


def participation(ev): return float(ev.sum() ** 2 / (ev ** 2).sum())


def twonn(D):
    """TwoNN (Facco 2017) from a squared-distance matrix with +inf diagonal."""
    s = np.sqrt(np.sort(D, 1)[:, :2]); s = s[s[:, 0] > 0]; mu = np.sort(s[:, 1] / s[:, 0]); n = len(mu)
    m = int(n * 0.9); F = np.arange(1, n + 1) / n; x = np.log(mu[:m]); y = -np.log(1 - F[:m])
    return float((x * y).sum() / (x * x).sum())


def local_pr(Z, nn):
    X = Z[nn]; X = X - X.mean(1, keepdims=True); C = np.einsum("nki,nkj->nij", X, X)
    return float(np.median(np.trace(C, axis1=1, axis2=2) ** 2 / (C ** 2).sum((1, 2))))


def vote_acc(nn, y):
    cls = np.unique(y); yi = np.searchsorted(cls, y); votes = np.zeros((len(y), len(cls)))
    for j in range(nn.shape[1]): np.add.at(votes, (np.arange(len(y)), yi[nn[:, j]]), 1)
    return float((votes.argmax(1) == yi).mean())


def ncm_acc(Z, y):
    cls = np.unique(y); mu = np.stack([Z[y == c].mean(0) for c in cls]); return float((cls[sqd(Z, mu).argmin(1)] == y).mean())


def fisher(Z, y):
    gm = Z.mean(0); sb = sw = 0.0
    for c in np.unique(y):
        zc = Z[y == c]; mc = zc.mean(0); sb += len(zc) * ((mc - gm) ** 2).sum(); sw += ((zc - mc) ** 2).sum()
    return float(sb / sw)


def sil(Zn, y): return float(silhouette_score(Zn, y)) if len(np.unique(y)) > 1 else NAN


def per_probe(Z, y, g, nn_pix, prev_nn, init_nn, recon_err=None):
    Z = np.asarray(Z, np.float64); n = len(Z); k = kk(n); out = {}
    st = M.global_distance_stats(Z); out["mpd_raw"] = st["mean_pairwise_dist"]; out["norm_mean"] = st["mean_norm"]
    Zn = M.normalize_latents(Z, "mean_pairwise")
    ev = np.clip(np.linalg.eigvalsh(np.cov(Z.T)), 0, None)
    out["pr_dim"] = participation(ev); out["top2_var"] = float(np.sort(ev)[::-1][:2].sum() / ev.sum())
    D = sqd(Z); np.fill_diagonal(D, np.inf); out["twonn_dim"] = twonn(D)
    nn, dk = knn_sorted(Z, k)
    kl = 50 if n >= 300 else 20; nnl, _ = knn_sorted(Z, min(kl, n - 1)); out["local_pr_dim"] = local_pr(Z, nnl)
    for nm, lab in (("label", y), ("pix", g)):
        out[f"knn_{nm}_acc"] = vote_acc(nn, lab); out[f"ncm_{nm}_acc"] = ncm_acc(Z, lab); out[f"fisher_{nm}"] = fisher(Z, lab); out[f"sil_{nm}"] = sil(Zn, lab)
    km = KMeans(10, n_init=2, random_state=0).fit_predict(Zn)
    out["kmeans_nmi_label"] = float(nmi_score(y, km)); out["kmeans_nmi_pix"] = float(nmi_score(g, km))
    out["knn_overlap_pixel"] = overlap(nn, nn_pix)
    out["knn_overlap_prev"] = overlap(nn, prev_nn) if prev_nn is not None else NAN
    out["knn_overlap_init"] = overlap(nn, init_nn) if init_nn is not None else NAN
    kd = dk[:, -1]; out["knn_dist_cv"] = float(kd.std() / kd.mean())
    occ = np.bincount(nn.ravel(), minlength=n); out["hubness_skew"] = float(skew(occ))
    mpd = np.sqrt(D[np.isfinite(D)]).mean(); out["nn1_ratio"] = float(dk[:, 0].mean() / mpd)
    out["rho_err_knn"] = float(spearmanr(recon_err, kd)[0]) if recon_err is not None else NAN
    out["_nn"] = nn
    return out


def cross(Zte, Ztr, err_te=None):
    Zte = np.asarray(Zte, np.float64); Ztr = np.asarray(Ztr, np.float64)
    s = M.global_distance_stats(Ztr)["mean_pairwise_dist"]; A, B = Zte / s, Ztr / s
    Dx = sqd(A, B); Dtt = sqd(B); np.fill_diagonal(Dtt, np.inf)
    t2t = np.sqrt(Dx.min(1)); r_tt = np.sqrt(np.median(Dtt.min(1))); r_t2 = np.sqrt(np.median(Dx.min(0)))
    Dee = sqd(A); np.fill_diagonal(Dee, np.inf); r_ee = np.sqrt(np.median(Dee.min(1)))
    return dict(test_to_train_nn=float(np.median(t2t) / r_tt), train_to_test_nn=float(r_t2 / r_ee),
                rho_test_err_train_nn=float(spearmanr(err_te, t2t)[0]) if err_te is not None else NAN)
