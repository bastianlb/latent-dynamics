"""Shared helpers for the reviewer analyses (Amendment 3). Import FIRST (caps threads before numpy).
Usage in scripts:  --dataset mnist  -> set (ii) (ANALYSIS_RUNSET=ext)   --dataset fashion -> Fashion blocks A+B (3 design vars)."""
import os, sys
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_v] = "2"
_ds = "fashion" if ("--dataset" in sys.argv and sys.argv[sys.argv.index("--dataset") + 1] == "fashion") else "mnist"
os.environ["ANALYSIS_RUNSET"] = "orig" if _ds == "fashion" else "ext"
os.environ["ANALYSIS_OUT"] = os.environ.get("ANALYSIS_OUT", "_unused")
import common2 as C
import data2 as D
import step3_predict_ext as S
import numpy as np
import re, json

DATASET = _ds
OUT2 = os.path.join(C.OUT_ROOT, "ext2", DATASET)
os.makedirs(OUT2, exist_ok=True)
CUTS = S.CUTS
M_ACC, M_GAP = 0.03, 0.05
COMPACT = ("fisher", "eff_rank_pr", "mag_t1", "h1_top5")


def margin(target):
    return M_ACC if target.startswith("acc") else M_GAP


def load_fashion():
    runs = []
    C.touch_nfs()
    for rid in sorted(os.listdir(C.RUNS)):
        if not rid.startswith("fashion_mnist_probe_none_d16_") or not re.search(r"_(phase_n|gap_n)", rid):
            continue
        d_ = os.path.join(C.RUNS, rid)
        if not os.path.exists(os.path.join(d_, "model_final.pt")):
            continue
        d = C.load_cache(rid)
        if d is None:
            continue
        info = C.run_info(rid)
        steps = d["step"]
        gap = d["run/train_acc"] - d["run/test_acc"]
        pk = int(np.nanargmax(d["run/test_acc"]))
        runs.append(dict(info=info, d=d, step=steps, covered=False, onsetN=np.nan, preN=len(steps) - 1, topo=d["topo_mask"].astype(bool),
                         onset=np.nan, pre=len(steps) - 1, gap=gap, peak_acc=float(d["run/test_acc"][pk]), peak_step=int(steps[pk]), cross_step=np.nan))
    return runs


def get_runs():
    return load_fashion() if DATASET == "fashion" else D.load_runs("mnist")


def hasher(runs, tname):
    return np.array([S.targets(r)[tname] for r in runs])


# ------------------------------------------------------------------ learning-curve extrapolation (A4)
ALPHA_GRID = np.geomspace(0.03, 5.0, 40)
LAM_GRID = np.geomspace(1e-4, 3.0, 40)


def _vp(t, y, grid, kind):
    """variable projection: for each nonlinear param, linear LS for (c, a). returns (c, a, param)."""
    best = None
    for p in grid:
        u = (t ** (-p)) if kind == "pow" else np.exp(-p * t)
        A = np.stack([np.ones_like(u), u], 1)
        coef, res, rk, _ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if best is None or sse < best[0]:
            best = (sse, coef[0], coef[1], p)
    return best[1], best[2], best[3]


def extrapolate(step, y, target=15000.0, lo=0.0, hi=1.0):
    """returns (pow3, exp, loglin) extrapolations at `target`, with persistence fallbacks."""
    step = np.asarray(step, float); y = np.asarray(y, float)
    ok = np.isfinite(y); step, y = step[ok], y[ok]
    last = float(y[-1]) if len(y) else np.nan
    out = []
    t = step + 1.0
    nuniq = len(np.unique(t))
    for fam in ("pow", "exp", "loglin"):
        try:
            if fam == "loglin":
                if nuniq < 3: raise ValueError
                A = np.stack([np.ones_like(t), np.log(t)], 1)
                coef = np.linalg.lstsq(A, y, rcond=None)[0]
                v = coef[0] + coef[1] * np.log(target + 1.0)
            else:
                if nuniq < 4: raise ValueError
                if fam == "pow":
                    c, a, p = _vp(t, y, ALPHA_GRID, "pow"); v = c + a * (target + 1.0) ** (-p)
                else:
                    c, a, p = _vp(t, y, LAM_GRID, "exp"); v = c + a * np.exp(-p * (target + 1.0))
            if not np.isfinite(v): raise ValueError
            v = float(np.clip(v, lo, hi))
        except Exception:
            v = last
        out.append(v)
    return out


_EXT_CACHE = {}
def ext_feats(r, cut):
    """6 extrapolated features for run r at cutoff (all snapshots with step <= cutoff)."""
    key = (r["info"]["run_id"], cut)
    if key in _EXT_CACHE:
        return _EXT_CACHE[key]
    st = r["step"]; m = st <= cut
    if m.sum() == 0: m[0] = True
    acc = r["d"]["run/test_acc"][m]; gap = r["gap"][m]
    ea = extrapolate(st[m], acc, lo=0.0, hi=1.0); eg = extrapolate(st[m], gap, lo=-0.3, hi=1.0)
    v = np.array(ea + eg, float)
    _EXT_CACHE[key] = v
    return v


EXT_NAMES = [f"ex_{f}_{c}" for c in ("acc", "gap") for f in ("pow", "exp", "loglin")]


def build_all(runs, cut, with_ext=True):
    """B0, DES, G, cfg, seeds, names + EXT (n x 6)"""
    B0, DES, G, cfg, seeds, names = S.build(runs, cut)
    EX = np.array([ext_feats(r, cut) for r in runs]) if with_ext else None
    return B0, DES, G, cfg, seeds, names, EX


def compact_cols(gnames):
    return [i for i, n in enumerate(gnames) if n.split("_", 1)[1] in COMPACT and n[:2] in ("te", "dl")]


# ------------------------------------------------------------------ bootstrap / classification
def boot_samples(y, pb, pc, g, nb=2000, seed=0):
    """paired config bootstrap of R2(pc)-R2(pb); returns array of samples."""
    rng = np.random.default_rng(seed)
    u = np.unique(g); mem = [np.where(g == c)[0] for c in u]
    out = np.empty(nb)
    for k in range(nb):
        i = np.concatenate([mem[j] for j in rng.integers(0, len(u), len(u))])
        out[k] = S.r2(y[i], pc[i]) - S.r2(y[i], pb[i])
    return out


def summarize(y, pb, pc, g, tname, nb=2000, seed=0):
    bs = boot_samples(y, pb, pc, g, nb, seed)
    d = S.r2(y, pc) - S.r2(y, pb)
    lo95, hi95 = np.percentile(bs, [2.5, 97.5]); lo90, hi90 = np.percentile(bs, [5, 95])
    return dict(delta=d, lo95=lo95, hi95=hi95, lo90=lo90, hi90=hi90, **classify(lo90, hi90, margin(tname)))


def classify(lo90, hi90, m):
    mstar = max(abs(lo90), abs(hi90))
    if lo90 > 0:
        lab = "positive" + (" (also within margin)" if (lo90 > -m and hi90 < m) else "")
    elif lo90 > -m and hi90 < m:
        lab = "equivalent"
    else:
        lab = "inconclusive" + (" (possible harm)" if lo90 < -m else "")
    return dict(margin=m, label=lab, min_margin=mstar)


def fit_predict(X, y, g, Xte, model):
    return S.ridge_fit_predict(X, y, g, Xte) if model == "ridge" else S.gbm_fit_predict(X, y, g, Xte)


def fold_predict(X, y, g, test_masks, model):
    """train on complement of each test mask (nested group-CV alpha on training configs)."""
    p = np.full(len(y), np.nan)
    for te in test_masks:
        tr = ~te
        if tr.sum() < 5 or te.sum() == 0: continue
        p[te] = fit_predict(X[tr], y[tr], g[tr], X[te], model)
    return p


def write_csv(path, rows):
    import csv
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
