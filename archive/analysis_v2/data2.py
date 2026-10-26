"""Load per-run caches into convenient structures (used by build_tables / step2 / step3)."""
import common2 as C
import numpy as np

CORE = ["fisher", "probe_acc", "eff_rank_pr", "h1_top5", "mag_t1", "pmag_fixed"]
ALLM = ["fisher", "norm", "probe_acc", "eff_rank_pr", "eff_rank_ent", "h0_total", "h1_total", "h1_top5", "h1_top10",
        "h1_entropy", "mag_t1", "pmag_fixed"]
NICE = {"fisher": "Fisher", "norm": "raw norm", "probe_acc": "probe acc", "eff_rank_pr": "eff. rank (PR)",
        "eff_rank_ent": "eff. rank (ent)", "h0_total": "H0 total", "h1_total": "H1 total", "h1_top5": "H1 top-5",
        "h1_top10": "H1 top-10", "h1_entropy": "H1 entropy", "mag_t1": "Mag(t=1)", "pmag_fixed": "PMag[.5,10]",
        "gap": "gap (train-test acc)", "test_acc": "test acc", "train_acc": "train acc"}


def idx_le(steps, target):
    """index of last snapshot with step <= target (0 if none)."""
    i = np.searchsorted(steps, target, side="right") - 1
    return max(int(i), 0)


def idx_le_mask(steps, mask, target):
    cand = np.where(mask & (steps <= target))[0]
    return int(cand[-1]) if len(cand) else int(np.where(mask)[0][0])


def load_runs(dataset=None):
    runs = []
    for rid in C.finished_runs():
        info = C.run_info(rid)
        if dataset and info["dataset"] != dataset:
            continue
        d = C.load_cache(rid)
        if d is None:
            continue
        steps = d["step"]
        tl = d["run/train_loss"]
        onset = C.onset_step(rid, steps, tl) if info["dataset"] == "mnist" else np.nan
        covered = onset is not None
        onset = np.nan if onset is None else onset
        onsetN = C.onset_step(rid, steps, tl, "N") if covered and info["dataset"] == "mnist" else np.nan
        preN = len(steps) - 1 if not np.isfinite(onsetN) else int(np.where(steps < onsetN)[0][-1])
        pre = len(steps) - 1 if not np.isfinite(onset) else int(np.where(steps < onset)[0][-1])
        gap = d["run/train_acc"] - d["run/test_acc"]
        # exploratory label-noise quantities
        pk = int(np.nanargmax(d["run/test_acc"]))
        xo = np.where((gap > 0) & (np.arange(len(gap)) > pk))[0]
        r = dict(info=info, d=d, step=steps, covered=covered, onsetN=onsetN, preN=preN, topo=d["topo_mask"].astype(bool), onset=onset, pre=pre, gap=gap,
                 peak_acc=float(d["run/test_acc"][pk]), peak_step=int(steps[pk]),
                 cross_step=float(steps[xo[0]]) if len(xo) else np.nan)
        runs.append(r)
    return runs


def series(r, probe, m):
    if probe == "run":
        return r["gap"] if m == "gap" else r["d"]["run/" + m]
    return r["d"][f"{probe}/{m}"]


def delta(r, m):
    return series(r, "train", m) - series(r, "test", m)


def by_config(runs):
    g = {}
    for r in runs:
        g.setdefault(r["info"]["config_id"], []).append(r)
    return g


def mean_series(rs, fn):
    return np.nanmean(np.stack([fn(r) for r in rs]), 0)
