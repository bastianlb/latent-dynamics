"""Shared helpers for analysis_v2. Import FIRST (caps threads before numpy)."""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import csv, json, re, sys
import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)
ART = "/vol/bitbucket/lbastian/latent-dynamics"
RUNS = os.path.join(ART, "runs_v2", "runs")
OUT_ROOT = os.path.join(ART, "analysis_v2")
CACHE = os.path.join(OUT_ROOT, "cache")          # shared per-run cache (never overwritten, only added to)
# ANALYSIS_OUT=<name> redirects tables/figures to analysis_v2/ext/<name>/ (default: original location)
OUT = os.path.join(OUT_ROOT, "ext", os.environ["ANALYSIS_OUT"]) if os.environ.get("ANALYSIS_OUT") else OUT_ROOT
FIG = os.path.join(OUT, "figures")
# ANALYSIS_RUNSET: orig (default; blocks A+B = original 105 MNIST runs + CIFAR), ext (adds C,D,E2), ext_adam (ext minus SGD), all (== ext)
RUNSET = os.environ.get("ANALYSIS_RUNSET", "orig")
E2_WDS = (3e-4, 3e-3)
PHASE_SUMMARY = os.path.join(ART, "dynamics_v2", "phase_summary.csv")
PHASE_SUMMARY_ALL = [os.path.join(ART, "dynamics_v2", p) for p in ("phase_summary.csv", "sl40k/phase_summary.csv", "sgd40k/phase_summary.csv")]
PROBES = ("test", "train")
PROBE_DIR = {"test": "latents", "train": "latents_train"}
MAG_GRID = np.geomspace(0.1, 100.0, 19)        # includes t=1 (idx 6) and t=10 (idx 12)
PMAG_WIN = (0.5, 10.0)
MNIST_FULL_N = 60000


def touch_nfs():
    try:
        os.listdir(RUNS)
    except OSError:
        pass


def block_of(c):
    """block label from config.json: A/B (orig phase/gap), C (sl40k), D (sgd40k), E2 (phase, wd 3e-4/3e-3), cifar."""
    tag = c.get("tag") or ""
    if c["dataset"] != "mnist":
        return "cifar"
    if tag.startswith("sl40k_n"): return "C"
    if tag.startswith("sgd40k_n"): return "D"
    if tag.startswith("gap_n"): return "B"
    if tag.startswith("phase_n"):
        return "E2" if float(c.get("weight_decay") or 0) in E2_WDS else "A"
    return "other"


def finished_runs(prefix=("mnist_probe_none_d16_", "cifar10_probe_none_d512_"), runset=None):
    runset = runset or RUNSET
    touch_nfs()
    out = []
    for rid in sorted(os.listdir(RUNS)):
        if not rid.startswith(prefix) or not re.search(r"_(phase_n|gap_n|rn18_n|sl40k_n|sgd40k_n)", rid):
            continue
        d = os.path.join(RUNS, rid)
        if not (os.path.exists(os.path.join(d, "model_final.pt")) and os.path.isdir(os.path.join(d, "latents_train"))):
            continue
        b = block_of(json.load(open(os.path.join(d, "config.json"))))
        if runset == "orig" and b in ("C", "D", "E2"): continue
        if runset == "ext_adam" and b == "D": continue
        if runset == "sgd" and b != "D": continue
        out.append(rid)
    return out


def run_info(rid: str) -> dict:
    c = json.load(open(os.path.join(RUNS, rid, "config.json")))
    ds = c["dataset"]
    ts = c.get("train_subset")
    n = ts if ts else (MNIST_FULL_N if ds == "mnist" else 50000)
    ln = float(c.get("label_noise") or 0.0)
    wd = float(c.get("weight_decay") or 0.0)
    seed = int(c["seed"])
    cid = f"{ds}|n{'full' if not ts else ts}|wd{wd:g}|ln{ln:g}"
    opt, tsteps = c["optimizer"], int(c.get("train_steps") or 0)
    if ds == "mnist" and (opt, tsteps) != ("adam", 20000):   # keep original ids for 20k-Adam runs; disambiguate 40k / SGD
        cid += f"|{opt}|{tsteps}"
    return dict(run_id=rid, dataset=ds, n=n, n_label="full" if not ts else str(ts), wd=wd, label_noise=ln,
                seed=seed, config_id=cid, latent_dim=int(c["latent_dim"]), optimizer=c["optimizer"],
                train_steps=c.get("train_steps") or 0, epochs=c.get("epochs"), block=block_of(c), tag=c.get("tag"),
                is_sgd=int(opt == "sgd"))


def detect_slingshot(step, train_loss):
    """own detector: train loss jumps >=3x after sitting <1e-4 for >=3 consecutive snapshots."""
    tl = np.asarray(train_loss, float); n = len(tl)
    for i in range(3, n):
        if np.all(tl[i - 3:i] < 1e-4) and tl[i] >= 3 * max(tl[i - 1], 1e-12) and tl[i] > 3e-4:
            return int(step[i])
    return None


_PS = None
def phase_summary():
    """run_id -> dict(any=onset any event, N=onset type-N (norm-growth) only, mode); only rows with slingshot_detector_mode."""
    global _PS
    if _PS is None:
        _PS = {}
        def f(v):
            try: return float(v) if v not in ("", None) else np.nan
            except ValueError: return np.nan
        for p in PHASE_SUMMARY_ALL:
            if not os.path.exists(p): continue
            for r in csv.DictReader(open(p)):
                if not r.get("slingshot_detector_mode"): continue
                _PS[r["run_id"]] = dict(any=f(r.get("slingshot_onset_step")), N=f(r.get("slingshot_onset_step_N")), mode=r["slingshot_detector_mode"])
    return _PS


def onset_step(rid, step, train_loss, kind="any"):
    """onset (nan if none) from the floor-relative detector files; None if the run is not covered (old absolute-threshold fallback removed)."""
    ps = phase_summary()
    return ps[rid][kind] if rid in ps else None


def load_cache(rid):
    p = os.path.join(CACHE, rid + ".npz")
    if not os.path.exists(p):
        return None
    with np.load(p, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}
