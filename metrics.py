"""
metrics.py
==========

Pure numpy/scipy/ripser metrics computed on a SINGLE latent snapshot.

Every function in this module takes:
    latents : float array [N, D]  -- encoder outputs for the fixed probe set
    labels  : int array   [N]     -- class label per probe sample

and returns scalars (or small arrays, e.g. the [C, C] class-centroid distance
matrix). No file I/O happens here; that lives in analyze.py, which calls these
functions once per snapshot and assembles the timeseries.

Design goals (per SPEC.md / the task brief):
  - Vectorized, fast enough for N ~ 1000, D ~ 16, and ~100-500 snapshots.
  - Each metric documents its exact formula in its docstring.
  - Topological metrics (H0/H1 persistence, entropy, alpha-weighted lifetime
    sums, magnitude / PMag) use `ripser` from the `ripser` package.

References used below:
  - Leinster, T. "The magnitude of metric spaces." Documenta Math. 18 (2013).
  - Govc, D. & Hepworth, R. "Persistent magnitude." J. Pure Appl. Algebra 225
    (2021). arXiv:1911.11016. Defines "persistent magnitude" as the magnitude
    of the Rips-type filtration evaluated/integrated over a scale parameter,
    directly paralleling persistent homology.
  - Rucco et al. "Characterisation of the idling behaviour of neural networks
    via persistence entropy" / Atienza et al. "Stability properties of a
    persistent entropy" -- standard normalized Shannon entropy of lifetimes.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

try:
    from ripser import ripser as _ripser
except ImportError:  # pragma: no cover - allows py_compile / partial envs
    _ripser = None


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------

def pairwise_distances(latents: np.ndarray) -> np.ndarray:
    """Dense [N, N] Euclidean distance matrix for `latents` [N, D]."""
    latents = np.asarray(latents, dtype=np.float64)
    return squareform(pdist(latents, metric="euclidean"))


def _unique_sorted_labels(labels: np.ndarray) -> np.ndarray:
    return np.unique(np.asarray(labels))


# ---------------------------------------------------------------------------
# NORMALIZATION (scale vs shape)
# ---------------------------------------------------------------------------
#
# Motivation (see analyze.py --normalize): in the first real run the latent
# space inflates ~isotropically over training (mean latent norm 0.57 -> 23.7
# on mnist_probe_d16). Every distance/topology metric below is computed from
# pairwise Euclidean distances, so an isotropic scale blow-up multiplies
# separation, compactness, H0/H1 lifetime sums and PMag by (roughly) the same
# factor -- it dominates the signal and hides genuine SHAPE dynamics (how the
# class clusters rearrange relative to each other). `normalize_latents`
# rescales a single snapshot's point cloud BEFORE the metrics in this module
# are computed, so that "separation_mean", "fisher_ratio", "h1_entropy", etc.
# track shape, not scale. The raw (un-normalized) scale is still exposed
# separately -- see analyze.py's `raw_*` columns, computed from the original
# un-normalized latents via `global_distance_stats` / `inter_class_separation`.
# ---------------------------------------------------------------------------

NORMALIZE_MODES = ("none", "mean_pairwise", "zscore")


def normalize_latents(latents: np.ndarray, mode: str = "mean_pairwise") -> np.ndarray:
    """Per-snapshot normalization of a latent point cloud.

    Pure numpy; does not mutate `latents`; always returns a float64 array.

    Modes
    -----
    - "none": identity (returns a float64 copy, unchanged).
    - "mean_pairwise": divide EVERY coordinate by a single scalar, the mean
          pairwise Euclidean distance over all i<j point pairs:

              scale = mean_{i<j} || z_i - z_j ||
              latents_normalized = latents / scale

          This is an ISOTROPIC rescaling -- one scalar for the whole cloud --
          so it preserves angles and all relative distance RATIOS (hence the
          full shape of the point cloud, including its persistence diagrams
          up to a global relabelling of the filtration scale). It removes
          only the absolute scale, which is exactly the "inflation" we want
          to factor out. Degenerate point clouds (N<2, or a zero/near-zero
          mean pairwise distance, e.g. every point coincident) fall back to
          returning the input unchanged (documented; downstream metrics on a
          degenerate cloud are already ill-defined / NaN-prone regardless of
          normalization).
    - "zscore": per-dimension standardization,

              latents_normalized[:, j] = (latents[:, j] - mean_j) / (std_j + eps)

          applied independently to each of the D dimensions (eps-guarded so a
          constant dimension, std_j == 0, does not divide by zero -- it maps
          to all-zeros in that dimension instead of inf/nan). This is
          ANISOTROPIC: it can stretch different directions by different
          amounts, so unlike "mean_pairwise" it is NOT guaranteed to preserve
          shape in the strict sense (angles / relative distances can change);
          it removes per-axis scale/offset instead of a single global scale.

    Raises
    ------
    ValueError if `mode` is not one of NORMALIZE_MODES.
    """
    latents = np.asarray(latents, dtype=np.float64)
    if mode not in NORMALIZE_MODES:
        raise ValueError(f"Unknown normalize mode {mode!r}; expected one of {NORMALIZE_MODES}")

    if mode == "none":
        return latents.copy()

    if mode == "mean_pairwise":
        n = latents.shape[0]
        if n < 2:
            return latents.copy()
        scale = pdist(latents, metric="euclidean").mean()
        if not np.isfinite(scale) or scale <= 1e-12:
            return latents.copy()
        return latents / scale

    # mode == "zscore"
    eps = 1e-8
    mu = latents.mean(axis=0)
    sigma = latents.std(axis=0)
    return (latents - mu[None, :]) / (sigma[None, :] + eps)


# ---------------------------------------------------------------------------
# EUCLIDEAN / CLASS-GEOMETRY METRICS
# ---------------------------------------------------------------------------

def class_centroids(latents: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-class centroids.

    centroid_c = mean_{i: labels[i]==c} latents[i]

    Returns
    -------
    classes   : [C] sorted unique class labels present in this snapshot
    centroids : [C, D] centroid for each class (row order matches `classes`)
    """
    latents = np.asarray(latents, dtype=np.float64)
    classes = _unique_sorted_labels(labels)
    centroids = np.stack([latents[labels == c].mean(axis=0) for c in classes], axis=0)
    return classes, centroids


def intra_class_compactness(latents: np.ndarray, labels: np.ndarray) -> dict:
    """Intra-class compactness.

    Two complementary measures, each averaged (unweighted) over classes:

      - centroid_dist: mean_c [ mean_{i in c} || z_i - centroid_c || ]
            i.e. average distance of points to their own class centroid,
            averaged first within a class then across classes.
      - pairwise: mean_c [ mean_{i<j in c} || z_i - z_j || ]
            i.e. average pairwise Euclidean distance between points of the
            SAME class, averaged first within a class then across classes.

    Lower values => tighter, more compact class clusters.

    Returns dict with keys: 'compactness_centroid', 'compactness_pairwise'.
    """
    latents = np.asarray(latents, dtype=np.float64)
    classes, centroids = class_centroids(latents, labels)

    centroid_means = []
    pairwise_means = []
    for c, mu in zip(classes, centroids):
        pts = latents[labels == c]
        if pts.shape[0] == 0:
            continue
        d_to_centroid = np.linalg.norm(pts - mu[None, :], axis=1)
        centroid_means.append(d_to_centroid.mean())
        if pts.shape[0] >= 2:
            pairwise_means.append(pdist(pts, metric="euclidean").mean())
        # a singleton class contributes no pairwise distance; skipped from
        # the pairwise average (documented behaviour).

    return {
        "compactness_centroid": float(np.mean(centroid_means)) if centroid_means else float("nan"),
        "compactness_pairwise": float(np.mean(pairwise_means)) if pairwise_means else float("nan"),
    }


def inter_class_separation(latents: np.ndarray, labels: np.ndarray) -> dict:
    """Inter-class separation: pairwise distances between class centroids.

    centroid_dist_matrix[a, b] = || centroid_a - centroid_b ||  (a, b over
    the sorted unique classes present in this snapshot).

    Returns dict with:
      - 'centroid_dist_matrix' : [C, C] full symmetric distance matrix
          (array-valued; analyze.py stacks these over steps for the .npz)
      - 'separation_mean'      : mean of the off-diagonal (upper-triangle)
                                  entries, i.e. mean inter-centroid distance
      - 'separation_min'       : min of the off-diagonal entries, i.e. the
                                  distance between the two closest classes
                                  (the "worst case" separation)
    """
    classes, centroids = class_centroids(latents, labels)
    C = len(classes)
    if C < 2:
        return {
            "centroid_dist_matrix": np.zeros((C, C), dtype=np.float64),
            "separation_mean": float("nan"),
            "separation_min": float("nan"),
        }
    dmat = squareform(pdist(centroids, metric="euclidean"))
    iu = np.triu_indices(C, k=1)
    off_diag = dmat[iu]
    return {
        "centroid_dist_matrix": dmat,
        "separation_mean": float(off_diag.mean()),
        "separation_min": float(off_diag.min()),
    }


def fisher_separation_ratio(latents: np.ndarray, labels: np.ndarray) -> float:
    """Fisher-like separation ratio = between-class scatter / within-class scatter.

    Using the classical multivariate (trace-of-scatter-matrix) definition,
    generalised from 1D Fisher discriminant ratio to D dimensions:

        S_between = sum_c n_c * || mu_c - mu_global ||^2
        S_within  = sum_c sum_{i in c} || z_i - mu_c ||^2
        ratio     = S_between / S_within

    where mu_c is the centroid of class c, mu_global is the centroid of all
    probe points, and n_c is the class count. Larger ratio => classes are
    better separated relative to their own spread (same spirit as LDA's
    generalized eigenvalue criterion, but scalarized via matrix traces so it
    needs no eigendecomposition).
    """
    latents = np.asarray(latents, dtype=np.float64)
    classes, centroids = class_centroids(latents, labels)
    mu_global = latents.mean(axis=0)

    s_between = 0.0
    s_within = 0.0
    for c, mu_c in zip(classes, centroids):
        pts = latents[labels == c]
        n_c = pts.shape[0]
        s_between += n_c * np.sum((mu_c - mu_global) ** 2)
        s_within += np.sum((pts - mu_c[None, :]) ** 2)

    if s_within <= 0:
        return float("inf") if s_between > 0 else float("nan")
    return float(s_between / s_within)


def global_distance_stats(latents: np.ndarray) -> dict:
    """Global geometry summary, independent of labels.

    - mean_pairwise_dist : mean Euclidean distance over all i<j pairs
    - mean_norm, median_norm : mean/median of ||z_i|| over all probe points
    """
    latents = np.asarray(latents, dtype=np.float64)
    dists = pdist(latents, metric="euclidean")
    norms = np.linalg.norm(latents, axis=1)
    return {
        "mean_pairwise_dist": float(dists.mean()) if dists.size else float("nan"),
        "mean_norm": float(norms.mean()),
        "median_norm": float(np.median(norms)),
    }


# ---------------------------------------------------------------------------
# TOPOLOGICAL METRICS (persistent homology via ripser)
# ---------------------------------------------------------------------------

DEFAULT_ALPHAS = (0.5, 1.0, 2.0)


def persistence_diagrams(latents: np.ndarray, maxdim: int = 1) -> dict:
    """Run ripser on the full point cloud and return raw diagrams.

    Returns dict with key 'dgms': a list of arrays, one per homology degree
    0..maxdim, each of shape [k_d, 2] with (birth, death) pairs. H0 bars
    typically contain exactly one infinite bar (death = +inf), representing
    the single connected component that never dies; ripser's default
    behaviour is preserved here (infinities handled downstream by the
    lifetime-sum / entropy functions, not here).
    """
    if _ripser is None:
        raise ImportError("ripser package is not installed in this environment")
    latents = np.asarray(latents, dtype=np.float64)
    result = _ripser(latents, maxdim=maxdim)
    return {"dgms": result["dgms"]}


def _finite_lifetimes(dgm: np.ndarray, cap_infinite: bool = True) -> np.ndarray:
    """Extract finite lifetimes (death - birth) from a diagram [k, 2].

    Infinite-death bars (standard for the single H0 component that never
    merges) are handled as follows (documented choice, per task spec):
      - if cap_infinite=True: the infinite death is CAPPED at the max finite
        death value present anywhere in the diagram (or at the max birth
        value if no finite death exists), so the bar contributes a finite,
        bounded lifetime instead of being dropped entirely. This keeps H0
        "total persistence" comparable across snapshots instead of silently
        discarding the longest-lived (and often most informative) bar.
      - if cap_infinite=False: infinite bars are DROPPED.

    Both H0 and H1 pass through this helper; H1 bars from ripser are already
    finite for a Vietoris-Rips filtration on a finite point cloud (ripser
    does not report an infinite H1 bar here), so capping only matters for H0
    in practice.
    """
    if dgm.size == 0:
        return np.zeros(0, dtype=np.float64)
    births = dgm[:, 0]
    deaths = dgm[:, 1]
    is_inf = ~np.isfinite(deaths)
    if not np.any(is_inf):
        return deaths - births

    finite_deaths = deaths[~is_inf]
    if cap_infinite:
        cap = finite_deaths.max() if finite_deaths.size else births[is_inf].max()
        deaths = deaths.copy()
        deaths[is_inf] = cap
        return deaths - births
    else:
        return deaths[~is_inf] - births[~is_inf]


def alpha_weighted_lifetime_sum(dgm: np.ndarray, alpha: float = 1.0, cap_infinite: bool = True) -> float:
    """Alpha-weighted lifetime sum of a persistence diagram.

        L_alpha(dgm) = sum_i (d_i - b_i)^alpha

    summed over all (capped-)finite bars (b_i, d_i) in `dgm`. `alpha` is a
    free parameter; alpha=1 is exactly TOTAL PERSISTENCE (sum of lifetimes).
    alpha=2 emphasises long-lived features more strongly (it is the
    quantity underlying "persistent entropy"-adjacent weighted sums used in
    TDA featurization, e.g. Chung & Lawson, "Persistence curves"); alpha=0.5
    is a sub-linear weighting that dampens the influence of a few very
    long-lived bars. See DEFAULT_ALPHAS = (0.5, 1, 2).

    Infinite H0 bars are handled via `_finite_lifetimes` (cap_infinite flag,
    documented there).
    """
    lifetimes = _finite_lifetimes(dgm, cap_infinite=cap_infinite)
    if lifetimes.size == 0:
        return 0.0
    return float(np.sum(lifetimes ** alpha))


def persistence_entropy(dgm: np.ndarray, cap_infinite: bool = True) -> float:
    """Normalized persistence entropy of a diagram.

    Standard definition (Chintakunta et al.; Atienza et al. 2020 "stability
    properties of persistence entropy"):

        p_i = l_i / L,   L = sum_i l_i   (l_i = lifetime of bar i)
        PE(dgm) = - sum_i p_i * log(p_i)

    This is plain (non-normalized-by-log(k)) Shannon entropy of the
    lifetime distribution treated as a probability distribution. Returns 0.0
    for an empty diagram or a diagram with a single bar (degenerate,
    zero-entropy distribution). Infinite H0 bars are handled via
    `_finite_lifetimes` (capping, documented there).
    """
    lifetimes = _finite_lifetimes(dgm, cap_infinite=cap_infinite)
    lifetimes = lifetimes[lifetimes > 0]
    if lifetimes.size == 0:
        return 0.0
    total = lifetimes.sum()
    if total <= 0:
        return 0.0
    p = lifetimes / total
    return float(-np.sum(p * np.log(p)))


def topological_metrics(latents: np.ndarray, maxdim: int = 1,
                         alphas=DEFAULT_ALPHAS) -> dict:
    """Compute H0/H1 persistence summaries for one point cloud.

    Returns a flat dict of scalars:
      h0_entropy, h1_entropy,
      h0_lifetime_sum_alpha{a}, h1_lifetime_sum_alpha{a}  for each a in alphas
      (alpha=1.0 is total persistence; key uses the literal alpha value,
      e.g. 'h0_lifetime_sum_alpha1.0').
    """
    dgms = persistence_diagrams(latents, maxdim=maxdim)["dgms"]
    out = {}
    names = {0: "h0", 1: "h1"}
    for dim in range(min(maxdim, 1) + 1):
        dgm = dgms[dim]
        name = names.get(dim, f"h{dim}")
        out[f"{name}_entropy"] = persistence_entropy(dgm)
        for a in alphas:
            out[f"{name}_lifetime_sum_alpha{a}"] = alpha_weighted_lifetime_sum(dgm, alpha=a)
    return out


# ---------------------------------------------------------------------------
# MAGNITUDE FUNCTION and PMag ("persistence magnitude")
# ---------------------------------------------------------------------------
#
# Background / citations
# -----------------------
# The MAGNITUDE of a finite metric space (X, d) (Leinster 2013, "The
# magnitude of metric spaces", Documenta Mathematica 18) is defined via the
# similarity matrix Z_ij = exp(-t * d(x_i, x_j)) for a scale parameter t > 0.
# If Z is invertible, the magnitude at scale t is
#       Mag(t) = sum_{i,j} (Z^{-1})_{ij}
# equivalently obtained by solving the linear system Z w = 1 (1 = all-ones
# vector) for the "weighting" w and summing: Mag(t) = sum_i w_i. This is the
# convention implemented below (solve, don't invert).
#
# As t -> infinity, Mag(t) -> |X| (the number of distinct points), and as
# t -> 0, Mag(t) -> 1/|X|... in general Mag(t) interpolates between a notion
# of "effective number of points" at different observation scales -- at
# small t, nearby points are "redundant" and barely add to the count; at
# large t every point is resolved as distinct. The magnitude FUNCTION
# t -> Mag(t) is therefore a multi-scale summary of the point cloud's
# effective size, directly analogous to how a persistence diagram summarises
# multi-scale topology.
#
# Govc & Hepworth, "Persistent magnitude" (J. Pure Appl. Algebra 225 (2021)
# 106517; arXiv:1911.11016) connect this explicitly to persistent homology:
# they define "persistent magnitude" of a filtered simplicial complex / Rips
# filtration as a magnitude-valued invariant that varies with the filtration
# (scale) parameter, playing the same role that persistence diagrams play for
# homology, and show it recovers Euler-characteristic-like information at
# each scale while being much cheaper to compute than persistent homology.
#
# PMag implemented here
# ----------------------
# We implement:
#   1. magnitude_function(latents, scales) -> Mag(t) at each requested scale
#      t, via the Z w = 1 linear solve (not explicit inversion), with a
#      robust solve (see `_solve_magnitude_weights` below: tiny diagonal
#      jitter is added first; if numpy's solve still fails/warns on an
#      ill-conditioned Z -- which happens at large t when many points
#      coincide/cluster tightly -- we fall back to `lstsq`, and finally to
#      the Moore-Penrose pseudo-inverse `pinv` if lstsq's rank is deficient).
#   2. PMag := the TRAPEZOIDAL INTEGRAL of Mag(t) over a finite scale range
#      [t_min, t_max], normalized by the range width:
#
#           PMag = (1 / (t_max - t_min)) * \int_{t_min}^{t_max} Mag(t) dt
#
#      i.e. the scale-averaged magnitude -- a single principled scalar that
#      summarises the whole magnitude FUNCTION (not just one arbitrary scale
#      sample), directly echoing how total persistence / lifetime sums
#      summarise a persistence diagram by integrating over scale. We pick
#      the integral/area definition (rather than "magnitude at one
#      characteristic scale") because: (a) it uses the full curve, so it is
#      robust to the exact choice of scale grid; (b) it has the same units
#      and monotonic-in-separation intuition as the alpha-weighted lifetime
#      sums already used for persistence (both are "area under a
#      scale-indexed curve"); (c) it stays cheap: a handful of dense-matrix
#      solves is O(k * N^3) for k scales, fine for N ~ 1000.
#
#      [t_min, t_max] defaults to a data-driven range based on the pairwise
#      distance distribution of the point cloud (5th to 95th percentile of
#      nonzero pairwise distances), so PMag is comparable across snapshots
#      whose latents may be rescaling over training; this choice is
#      documented in `_default_scale_range`.
# ---------------------------------------------------------------------------

DEFAULT_N_SCALES = 12


def _solve_magnitude_weights(Z: np.ndarray, jitter: float = 1e-9) -> np.ndarray:
    """Robustly solve Z w = 1 for the magnitude weighting vector w.

    Strategy (documented per task spec):
      1. Add a tiny diagonal jitter (`jitter` * I) to Z and try `np.linalg.solve`.
         This nudges away from exact singularity caused by near-duplicate
         points (common for probe sets drawn from a well-clustered latent
         space) without materially changing Mag(t).
      2. If that raises LinAlgError (exactly singular) or produces non-finite
         values, fall back to `np.linalg.lstsq` (least-squares; handles
         rank-deficient Z).
      3. If lstsq's result is still non-finite, fall back to the
         Moore-Penrose pseudo-inverse `np.linalg.pinv` as a last resort.
    """
    n = Z.shape[0]
    ones = np.ones(n, dtype=np.float64)
    Zj = Z + jitter * np.eye(n)
    try:
        w = np.linalg.solve(Zj, ones)
        if np.all(np.isfinite(w)):
            return w
    except np.linalg.LinAlgError:
        pass

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        w, *_ = np.linalg.lstsq(Zj, ones, rcond=None)
    if np.all(np.isfinite(w)):
        return w

    w = np.linalg.pinv(Zj) @ ones
    return w


def magnitude_function(latents: np.ndarray, scales: Optional[np.ndarray] = None,
                        n_scales: int = DEFAULT_N_SCALES) -> dict:
    """Magnitude function Mag(t) of the finite metric space `latents`.

    For each scale t > 0: Z_ij = exp(-t * d_ij); solve Z w = 1; Mag(t) = sum(w).
    See module docstring / section header above for the full definition and
    citations (Leinster 2013; Govc-Hepworth 2021).

    Parameters
    ----------
    latents : [N, D] array
    scales  : optional explicit array of scales t > 0. If None, a default
              data-driven range is used (`_default_scale_range`).
    n_scales: number of scales to sample if `scales` is None.

    Returns
    -------
    dict with:
      'scales'    : [n_scales] array of t values used
      'magnitude' : [n_scales] array of Mag(t) values (same order)
    """
    latents = np.asarray(latents, dtype=np.float64)
    n = latents.shape[0]
    if n == 0:
        return {"scales": np.zeros(0), "magnitude": np.zeros(0)}
    if n == 1:
        sc = scales if scales is not None else np.linspace(*_default_scale_range(latents), n_scales)
        return {"scales": np.asarray(sc), "magnitude": np.ones_like(np.asarray(sc), dtype=np.float64)}

    D = pairwise_distances(latents)
    if scales is None:
        t_min, t_max = _default_scale_range(latents)
        scales = np.linspace(t_min, t_max, n_scales)
    scales = np.asarray(scales, dtype=np.float64)

    mags = np.empty_like(scales)
    for i, t in enumerate(scales):
        Z = np.exp(-t * D)
        w = _solve_magnitude_weights(Z)
        mags[i] = float(np.sum(w))
    return {"scales": scales, "magnitude": mags}


def _default_scale_range(latents: np.ndarray) -> tuple[float, float]:
    """Data-driven default [t_min, t_max] for the magnitude function.

    Uses the 5th/95th percentile of the NONZERO pairwise-distance
    distribution, d_lo and d_hi, and sets t = 1/d so that:
        t_max = 1 / d_lo   (resolves the closest typical pairs)
        t_min = 1 / d_hi   (barely resolves the farthest typical pairs)
    This keeps the scale range comparable across snapshots even as the
    overall latent scale (norm) drifts during training, since it is defined
    relative to the point cloud's own distance distribution rather than a
    fixed absolute scale.
    """
    D = pairwise_distances(latents)
    iu = np.triu_indices(D.shape[0], k=1)
    d = D[iu]
    d = d[d > 1e-12]
    if d.size == 0:
        return (0.1, 10.0)
    d_lo, d_hi = np.percentile(d, [5, 95])
    d_lo = max(d_lo, 1e-6)
    d_hi = max(d_hi, d_lo * 1.01)
    t_min, t_max = 1.0 / d_hi, 1.0 / d_lo
    return (float(t_min), float(t_max))


def pmag(latents: np.ndarray, scales: Optional[np.ndarray] = None,
         n_scales: int = DEFAULT_N_SCALES) -> dict:
    """PMag: scale-averaged magnitude, a scalar summary of the magnitude function.

        PMag = (1 / (t_max - t_min)) * integral_{t_min}^{t_max} Mag(t) dt

    computed via the trapezoidal rule over the sampled `scales`/Mag(t)
    values from `magnitude_function`. See the section header above for the
    full justification and citations. `t_min`/`t_max` are the min/max of
    `scales` (data-driven by default, see `_default_scale_range`).

    Returns dict with 'scales', 'magnitude' (full curve, for plotting) and
    'pmag' (the scalar summary).
    """
    mf = magnitude_function(latents, scales=scales, n_scales=n_scales)
    t, mag = mf["scales"], mf["magnitude"]
    if t.size < 2:
        area = float(mag[0]) if t.size == 1 else float("nan")
    else:
        width = t[-1] - t[0]
        area = float(np.trapezoid(mag, t) / width) if width > 0 else float(mag.mean())
    return {"scales": t, "magnitude": mag, "pmag": area}


# ---------------------------------------------------------------------------
# Top-level convenience: compute every scalar metric for one snapshot
# ---------------------------------------------------------------------------

def compute_all_metrics(latents: np.ndarray, labels: np.ndarray,
                         alphas=DEFAULT_ALPHAS, n_scales: int = DEFAULT_N_SCALES,
                         per_class_topology: bool = False) -> dict:
    """Compute every metric in this module for one (latents, labels) snapshot.

    Returns a flat dict: scalar metrics map name -> float; a small number of
    array-valued entries (currently just 'centroid_dist_matrix', and
    'magnitude_curve_scales'/'magnitude_curve_values') map name -> np.ndarray.
    analyze.py is responsible for splitting these into CSV scalar columns vs
    .npz array-valued keys.

    If `per_class_topology` is True, also computes H0/H1 lifetime-sum/entropy
    per class (on the subset of points with that label) and prefixes keys
    with 'perclass_{label}_'. This is optional and off by default since it
    multiplies the ripser cost by the number of classes.
    """
    latents = np.asarray(latents, dtype=np.float64)
    labels = np.asarray(labels)
    out: dict = {}

    out.update(intra_class_compactness(latents, labels))

    sep = inter_class_separation(latents, labels)
    out["centroid_dist_matrix"] = sep["centroid_dist_matrix"]
    out["separation_mean"] = sep["separation_mean"]
    out["separation_min"] = sep["separation_min"]

    out["fisher_ratio"] = fisher_separation_ratio(latents, labels)
    out.update(global_distance_stats(latents))

    out.update(topological_metrics(latents, maxdim=1, alphas=alphas))

    mag = pmag(latents, n_scales=n_scales)
    out["pmag"] = mag["pmag"]
    out["magnitude_curve_scales"] = mag["scales"]
    out["magnitude_curve_values"] = mag["magnitude"]

    if per_class_topology:
        for c in _unique_sorted_labels(labels):
            pts = latents[labels == c]
            if pts.shape[0] < 3:
                continue
            tm = topological_metrics(pts, maxdim=1, alphas=alphas)
            for k, v in tm.items():
                out[f"perclass_{int(c)}_{k}"] = v

    return out


# ---------------------------------------------------------------------------
# OPTIONAL additions (reanalysis): offline linear separability, fixed-window PMag.
# These do not change any default behaviour of the functions above.
# ---------------------------------------------------------------------------

def offline_linear_separability(latents: np.ndarray, labels: np.ndarray, n_splits: int = 5,
                                C: float = 1.0, seed: int = 0, max_iter: int = 300) -> float:
    """Post-hoc linear probe accuracy: k-fold stratified CV accuracy of a standardised
    multinomial logistic regression fit on the (raw) probe-set latents of ONE snapshot.

    Unlike the online probe head (trained jointly, possibly on a detached, moving
    latent) this is a fixed-protocol, snapshot-local measure of linear separability, so it
    is comparable across objectives/snapshots. Scale-invariant thanks to standardisation.
    """
    import warnings as _w
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    X = np.asarray(latents, dtype=np.float64)
    y = np.asarray(labels)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=max_iter))
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    with _w.catch_warnings():
        _w.simplefilter("ignore")
        return float(np.mean(cross_val_score(clf, X, y, cv=cv)))


def fixed_scale_grid(t_min: float, t_max: float, n_scales: int = DEFAULT_N_SCALES) -> np.ndarray:
    """Linearly spaced scale grid (same spacing rule as the default PMag window)."""
    return np.linspace(float(t_min), float(t_max), n_scales)


def pmag_fixed(latents: np.ndarray, t_min: float, t_max: float, n_scales: int = DEFAULT_N_SCALES) -> dict:
    """PMag over a FIXED scale window [t_min, t_max] (identical across snapshots), instead of the
    per-snapshot percentile window of `_default_scale_range`. Latents should already be
    normalised (e.g. mean-pairwise) so a common window is meaningful."""
    return pmag(latents, scales=fixed_scale_grid(t_min, t_max, n_scales))


# ---------------------------------------------------------------------------
# SELF-TEST: synthetic Gaussian blobs that separate over "time", end-to-end
# ---------------------------------------------------------------------------

def _make_synthetic_run(out_dir: str, n_classes: int = 10, n_per_class: int = 60,
                         d: int = 16, n_steps: int = 12, seed: int = 0) -> str:
    """Fabricate a synthetic run directory in the SPEC.md .npz format.

    Simulates `n_classes` Gaussian blobs in R^d whose centroids start
    overlapping (near the origin, tiny spread between centroids relative to
    per-class std) and move apart linearly over `n_steps` "steps", while
    per-class std shrinks slightly -- i.e. classes should become more
    separated and more compact over the run, which lets the self-test assert
    metrics move in the expected direction.

    Writes latents/step_{step:07d}.npz for step in [0, n_steps) under
    out_dir, matching the exact SPEC.md snapshot schema (latents, labels,
    step, epoch, train_loss, train_acc, test_acc). Returns out_dir.
    """
    import os

    rng = np.random.default_rng(seed)
    os.makedirs(os.path.join(out_dir, "latents"), exist_ok=True)

    n = n_classes * n_per_class
    labels = np.repeat(np.arange(n_classes), n_per_class).astype(np.int64)

    # Fixed random direction per class defines where its centroid moves to.
    final_centroid_positions = rng.normal(scale=8.0, size=(n_classes, d))
    # Per-class fixed "shape" noise (same noise realization every step, just
    # rescaled) so that only separation/compactness change, not identity.
    per_point_noise = rng.normal(size=(n, d))

    for step in range(n_steps):
        frac = step / max(n_steps - 1, 1)  # 0 -> 1 over the run
        centroids_now = frac * final_centroid_positions  # start ~0 (overlap)
        std_now = 2.0 - 1.5 * frac  # shrinks from 2.0 to 0.5 (more compact)

        latents = np.empty((n, d), dtype=np.float32)
        for c in range(n_classes):
            idx = labels == c
            latents[idx] = (centroids_now[c][None, :]
                             + std_now * per_point_noise[idx]).astype(np.float32)

        train_loss = float(2.5 * (1 - frac) + 0.05)
        train_acc = float(0.1 + 0.85 * frac)
        test_acc = float(0.1 + 0.80 * frac)

        np.savez(
            os.path.join(out_dir, "latents", f"step_{step:07d}.npz"),
            latents=latents,
            labels=labels,
            step=np.int64(step * 100),
            epoch=np.float32(frac * 5.0),
            train_loss=np.float32(train_loss),
            train_acc=np.float32(train_acc),
            test_acc=np.float32(test_acc),
        )

    # probe_index.npz, per SPEC.md (fixed probe set indices + labels)
    np.savez(os.path.join(out_dir, "probe_index.npz"),
              indices=np.arange(n, dtype=np.int64), labels=labels)

    import json
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump({
            "dataset": "synthetic_selftest",
            "latent_dim": d,
            "n_probe": n,
            "probe_balanced": True,
            "log_every": 100,
            "epochs": 5,
            "batch_size": 128,
            "lr": 1e-3,
            "seed": seed,
            "encoder_arch": "synthetic",
            "recon": False,
            "device": "cpu",
            "run_id": os.path.basename(out_dir.rstrip("/")),
            "artifacts_root": out_dir,
        }, f, indent=2)

    return out_dir


def _run_self_test():
    """End-to-end self-test: synthesize a run, run analyze.py + plot.py on
    it, and assert metrics move in the expected direction (separation up,
    compactness down/tighter, etc.). Prints a PASS/FAIL summary.
    """
    import os
    import subprocess
    import sys

    base = "/vol/bitbucket/lbastian/latent-dynamics/_selftest"
    run_dir = os.path.join(base, "synthetic_run")
    os.makedirs(base, exist_ok=True)

    print(f"[self-test] fabricating synthetic run at {run_dir}")
    _make_synthetic_run(run_dir, n_classes=10, n_per_class=60, d=16, n_steps=12, seed=0)

    here = os.path.dirname(os.path.abspath(__file__))
    py = sys.executable

    print("[self-test] running analyze.py ...")
    subprocess.run([py, os.path.join(here, "analyze.py"), "--run_dir", run_dir],
                    check=True)

    print("[self-test] running plot.py ...")
    subprocess.run([py, os.path.join(here, "plot.py"), "--run_dir", run_dir],
                    check=True)

    ts = np.load(os.path.join(run_dir, "metrics", "timeseries.npz"))
    sep_first, sep_last = ts["separation_mean"][0], ts["separation_mean"][-1]
    comp_first, comp_last = ts["compactness_centroid"][0], ts["compactness_centroid"][-1]
    fisher_first, fisher_last = ts["fisher_ratio"][0], ts["fisher_ratio"][-1]

    print(f"[self-test] separation_mean:        {sep_first:.3f} -> {sep_last:.3f}")
    print(f"[self-test] compactness_centroid:   {comp_first:.3f} -> {comp_last:.3f}")
    print(f"[self-test] fisher_ratio:           {fisher_first:.3f} -> {fisher_last:.3f}")

    ok = True
    if not (sep_last > sep_first):
        print("[self-test] FAIL: expected separation_mean to increase")
        ok = False
    if not (fisher_last > fisher_first):
        print("[self-test] FAIL: expected fisher_ratio to increase")
        ok = False
    # compactness (distance-to-centroid) should shrink as std_now shrinks
    if not (comp_last < comp_first):
        print("[self-test] FAIL: expected compactness_centroid to decrease (tighter clusters)")
        ok = False

    nan_cols = [k for k in ts.files if np.asarray(ts[k]).dtype.kind == "f"
                and np.isnan(np.asarray(ts[k], dtype=np.float64)).all()]
    if nan_cols:
        print(f"[self-test] WARNING: all-NaN columns (possible metric failure): {nan_cols}")

    print("[self-test] PASS" if ok else "[self-test] FAIL")
    return ok


if __name__ == "__main__":
    _run_self_test()
