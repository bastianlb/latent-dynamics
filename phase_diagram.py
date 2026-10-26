"""Phase-diagram aggregation of per-run latent dynamics over (train_subset N, weight_decay [, label noise]). CPU only.

Usage:
  python phase_diagram.py [--runs_root DIR] [--out DIR] [--workers 4] [--include_partial] [--runs GLOB] [--force]
Incremental: per-run results cached in OUT/phase_cache/<run_id>.pkl keyed by sha1(file names+sizes of latents*/ + code version).
Stable phase_summary.csv columns (per run; test-probe unless prefixed tr_):
  run_id, complete, seed, label_noise, N (60000=full), wd, n_snap
  slingshot_detector_mode  'floor' (default: loss-floor-relative) or 'abs'; slingshot_floors/_jump_ratios/_znorm_onsets/_types = per-event ';'-lists (type N: pre-onset mean ||z|| > 2x its step-1000 value, else F);
                         slingshot_n_events_N/_F, slingshot_onset_step_N/_F = per type; slingshot_events.csv = one row per event
  slingshot_onset_step   first slingshot onset step (NaN if none); slingshot_n_events; slingshot_onset_steps = all onsets, ';'-separated
  sl_d_*                 first-event deltas (snapshot onset+2 minus snapshot before): train_loss, test_loss, test_acc, nc1, znorm (mean raw ||z||),
                         plus every quantity below; sl_vel_ratio = velocity / median of previous 10
  pre_<q>                value at last snapshot before first onset (final if no event); at1000_/at5000_/at15000_<q> at fixed steps; final_<q> at end
                         q in fisher nc1 effrank etf ncm_acc cka_label train_acc test_acc gap test_loss fisher_ratio_tr_te nc1_ratio_tr_te cka_classmeans_tr_te
  lag_steps, lag_halfmax_log10   delayed-generalization lags; late_rise_*  late second-rise flags; tchi_* / t_velpeak  log10 step of susceptibility/velocity max
  cp_steps               change points (space-separated); mem_* label-noise memorization (NCM proxy)
Slingshot detector: train_loss<--sl_zero_thr for --sl_n_zero snapshots, then jump >=--sl_jump x AND kernel velocity > --sl_vel_mult x median of previous 10.
Heatmaps default to pre-slingshot values (--heatmap_at pre|final|1000|5000|15000).

Outputs (only under OUT): phase_summary.csv (per run), phase_grid.csv (per cell), phase_jumps.csv, figures/*.png.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, glob, json, csv, time, hashlib, pickle, collections, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dynamics as DY

RUNS_ROOT = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2"
CODE_VERSION = "pd-5"
ORDER = DY.ORDER  # fisher nc1 effrank etf ncm_acc cka_label
NFULL = 60000
FAMILY_GLOBS = {"phase": "mnist_probe_*_phase_*", "gap": "mnist_probe_*_gap_*", "sl40k": "mnist_probe_*_sl40k_*", "sgd40k": "mnist_probe_*_sgd40k_*", "eps": "mnist_probe_*_eps*_*", "sgdlr": "mnist_probe_*_sgdlr*_*"}  # eps/sgdlr: per-run analysis only here; aggregated by stability.py
DEFAULT_FAMILIES = "phase,gap"  # the 20k grid; sl40k / sgd40k are separate blocks (opt in with --family, aggregated into OUT/sl40k, OUT/sgd40k)
DEFAULT_RUNS = ",".join(FAMILY_GLOBS[f] for f in DEFAULT_FAMILIES.split(","))  # MNIST grid only (CIFAR runs share runs_v2/runs/)
LATE_STEP = 1000
SL = dict(mode="floor", zero_thr=1e-4, n_zero=3, jump=3.0, vel_mult=5.0, vel_win=10, plateau_fac=2.0, post_frac=0.1, min_step=20)  # slingshot detector thresholds (CLI-overridable)
FIXED_STEPS = (1000, 5000, 15000)
PEAK_MIN_STEP = 3  # exclude the init transient (step 0->1) when locating velocity/susceptibility maxima


# ---------------------------------------------------------------- loading (defensive)
def load_def(run_dir, sub):
    """Tolerates missing/truncated snapshot files. Returns dict or None."""
    steps, Zs, ep, ex = [], [], [], collections.defaultdict(list)
    labels = noisy = None
    for f in sorted(glob.glob(f"{run_dir}/{sub}/step_*.npz")):
        try:
            with np.load(f) as z:
                s = int(z["step"]); Z = z["latents"].astype(np.float64); y = z["labels"]
                e = float(z["epoch"]); row = {k: float(z[k]) for k in ("test_acc", "train_acc", "train_loss", "test_loss", "l2_scale") if k in z.files}
                nl = z["noisy_labels"] if "noisy_labels" in z.files else None
        except Exception:
            continue
        if "test_acc" not in row: continue
        steps.append(s); Zs.append(Z); ep.append(e)
        for k in ("test_acc", "train_acc", "train_loss", "test_loss", "l2_scale"): ex[k].append(row.get(k, np.nan))
        if labels is None: labels, noisy = y, nl
    if not steps: return None
    o = np.argsort(steps)
    return dict(steps=np.array(steps)[o], Zs=[Zs[i] for i in o], ep=np.array(ep)[o], labels=labels, noisy=noisy,
                ex={k: np.array(v)[o] for k, v in ex.items()})


def snap_hash(run_dir):
    items = []
    for sub in ("latents", "latents_train"):
        for f in sorted(glob.glob(f"{run_dir}/{sub}/step_*.npz")):
            try: items.append((sub, os.path.basename(f), os.stat(f).st_size))
            except OSError: pass
    return hashlib.sha1((CODE_VERSION + json.dumps(SL, sort_keys=True) + json.dumps(items)).encode()).hexdigest()


# ---------------------------------------------------------------- per-snapshot extras
def class_mean_centered(Z, y, classes):
    mu = np.array([Z[y == c].mean(0) for c in classes]); return mu - Z.mean(0)


def cka_gram(A, B):
    """Linear CKA between two (C,d) matrices of class means (rows = same classes); d may differ across sets only via rows pairing."""
    A = A - A.mean(0); B = B - B.mean(0)
    GA, GB = A @ A.T, B @ B.T
    H = np.eye(len(A)) - 1.0 / len(A)
    GA, GB = H @ GA @ H, H @ GB @ H
    return float((GA * GB).sum() / (np.linalg.norm(GA) * np.linalg.norm(GB) + 1e-30))


def mem_signal(Z, ytrue, ynoisy, classes):
    """NCM on noisy-label class means (what the net is pushed to fit). Returns acc vs noisy, acc vs true (all items),
    and on corrupted items: frac predicted==noisy (memorized), frac predicted==true (corrected)."""
    mu = np.array([Z[ynoisy == c].mean(0) if (ynoisy == c).any() else np.full(Z.shape[1], 1e9) for c in classes])
    pred = classes[((Z[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)]
    bad = ynoisy != ytrue
    out = [float((pred == ynoisy).mean()), float((pred == ytrue).mean())]
    out += [float((pred[bad] == ynoisy[bad]).mean()), float((pred[bad] == ytrue[bad]).mean())] if bad.any() else [np.nan, np.nan]
    return out


def oriented(order, test_acc):
    """name -> series where larger = more 'organized/improved' (used for late-rise detection)."""
    o = order
    return {"test_acc": test_acc, "fisher": np.log(np.clip(o[:, 0], 1e-9, None)), "nc1": -np.log(np.clip(o[:, 1], 1e-9, None)),
            "etf": -o[:, 3], "ncm_acc": o[:, 4], "cka_label": o[:, 5]}


def late_rise(steps, y):
    """Change point (binseg, sensitive) after step LATE_STEP with positive slope change in log time, following a plateau.
    Returns (flag, step, rise_fraction_of_range)."""
    rngy = y.max() - y.min()
    if not np.isfinite(rngy) or rngy < 1e-12 or len(y) < 20: return 0, np.nan, 0.0
    yn = (y - y.min()) / rngy
    cps = DY.binseg(((y - y.mean()) / (y.std() + 1e-12))[:, None], max_cp=6, min_size=3, frac=0.005)
    x = np.log1p(steps); w = 6; best = (0, np.nan, 0.0)
    for k in cps:
        if steps[k] <= LATE_STEP: continue
        a0, a1, b1 = max(0, k - w), k + 1, min(len(y), k + w)
        if k - a0 < 3 or b1 - k < 3: continue
        sb = np.polyfit(x[a0:a1], yn[a0:a1], 1)[0]; sa = np.polyfit(x[k:b1], yn[k:b1], 1)[0]
        rise = yn[b1 - 1] - yn[k]; before = yn[k] - yn[a0]
        if sa - sb > 0 and rise >= 0.02 and before < 0.5 * rise and rise > best[2]: best = (1, float(steps[k]), float(rise))
    return best


def first_cross(steps, y, thr):
    i = np.where(y >= thr)[0]; return float(steps[i[0]]) if len(i) else np.nan


def halfmax_x(steps, y):
    """log1p-step at which y first crosses halfway between y[0] and its max (linear interp in log time)."""
    lo, hi = y[0], np.nanmax(y)
    if not hi - lo > 1e-9: return np.nan
    h = lo + 0.5 * (hi - lo); x = np.log1p(steps); i = np.where(y >= h)[0][0]
    if i == 0: return float(x[0])
    return float(x[i - 1] + (h - y[i - 1]) / (y[i] - y[i - 1]) * (x[i] - x[i - 1]))



def detect_slingshots(steps, trl, vel):
    """vel[i] = kernel velocity (per dlog t) of the interval ending at snapshot i (nan at i=0).
    mode 'floor' (default, no absolute loss level): after the initial transient (loss already <= post_frac x its initial value, step >= min_step),
        event at i if trl[i] >= jump x trl[i-1] (hence also >= jump x the running-min floor) AND vel[i] > vel_mult x median(vel[i-vel_win:i]).
        A plateau requirement (within 2x of the running min) was tried and dropped: it missed 17/125 abs events, all preceded by a loss ramp above the floor.
    mode 'strict': literal running-min version: previous n_zero snapshots all <= plateau_fac x running-min, and trl[i] >= jump x running-min.
    mode 'abs' (legacy): trl < zero_thr at the n_zero previous snapshots, trl[i] >= jump x trl[i-1], same velocity condition."""
    ev = []; nz = SL["n_zero"]; run_min = np.minimum.accumulate(np.where(np.isfinite(trl), trl, np.inf))
    for i in range(max(nz, 2), len(steps)):
        if not np.isfinite(trl[i]): continue
        if SL["mode"] == "abs":
            if not np.all(trl[i - nz:i] < SL["zero_thr"]): continue
            ref = max(trl[i - 1], 1e-12)
        elif SL["mode"] == "floor":
            if not (trl[i - 1] <= SL["post_frac"] * trl[0] and steps[i] >= SL["min_step"]): continue  # past the initial transient: loss already fell >=10x from init (relative)
            ref = max(trl[i - 1], run_min[i - 1], 1e-12)
        else:
            fl = max(run_min[i - 1], 1e-12)
            if not np.all(trl[i - nz:i] <= SL["plateau_fac"] * fl): continue
            ref = fl
        pv = vel[max(1, i - SL["vel_win"]):i]; pv = pv[np.isfinite(pv)]
        if len(pv) == 0: continue
        if trl[i] >= SL["jump"] * ref and vel[i] > SL["vel_mult"] * np.median(pv): ev.append(i)
    return ev


def quantities(ser, order, ex):
    q = {n: order[:, k] for k, n in enumerate(ORDER)}
    for k in ("train_acc", "test_acc", "gap", "test_loss", "fisher_ratio_tr_te", "nc1_ratio_tr_te", "cka_classmeans_tr_te"):
        if k in ser: q[k] = ser[k]
    return q


QNAMES = ORDER + ["train_acc", "test_acc", "gap", "test_loss", "fisher_ratio_tr_te", "nc1_ratio_tr_te", "cka_classmeans_tr_te"]


def run_summary(R, pref):
    """Scalar summaries of a dynamics.analyze result R (+prefix)."""
    s = {}; st = R["steps"]; O = R["order"]; D = R["deriv"]
    for k, n in enumerate(ORDER):
        s[f"{pref}final_{n}"] = float(O[-1, k])
        m = st >= PEAK_MIN_STEP; a = np.abs(D[m, k]); s[f"{pref}tchi_{n}"] = float(np.log10(st[m][a.argmax()] + 1)) if m.any() else np.nan
    mm = R["step_mid"] >= PEAK_MIN_STEP; s[f"{pref}t_velpeak"] = float(np.log10(R["step_mid"][mm][R["vel_log"][mm].argmax()] + 1)) if mm.any() else np.nan
    s[f"{pref}t_velpeak_raw"] = float(np.log10(R["vel_peak_step"] + 1))
    s[f"{pref}cp_steps"] = " ".join(map(str, R["cp_steps"])); s[f"{pref}n_cp"] = len(R["cp_steps"])
    s[f"{pref}pca_ratio"] = float(R["pca_ratio"])
    return s


# ---------------------------------------------------------------- per-run analysis
def analyze_run(rd, include_partial=False):
    rid = os.path.basename(rd); cfg = json.load(open(f"{rd}/config.json"))
    complete = os.path.isfile(f"{rd}/model_final.pt")
    te, tr = load_def(rd, "latents"), load_def(rd, "latents_train")
    if te is None: return None
    if tr is not None:  # align on common steps
        common = np.intersect1d(te["steps"], tr["steps"])
        if len(common) < 8: tr = None
        else:
            for d in (te, tr):
                m = np.isin(d["steps"], common); idx = np.where(m)[0]
                d["Zs"] = [d["Zs"][i] for i in idx]; d["steps"] = d["steps"][idx]; d["ep"] = d["ep"][idx]
                d["ex"] = {k: v[idx] for k, v in d["ex"].items()}
    if len(te["steps"]) < 8: return None
    steps = te["steps"]
    out = dict(run_id=rid, complete=complete, cfg=cfg, steps=steps, n_snap=len(steps))
    res = {}
    for name, d in (("test", te), ("train", tr)):
        if d is None: continue
        res[name] = DY.analyze(rd, name, data=(d["steps"], d["Zs"], d["ep"], d["ex"]["test_acc"], d["labels"]))
    out["dyn_test"] = {k: res["test"][k] for k in ("steps", "order", "deriv", "vel_log", "step_mid", "sv_share", "cp_steps", "vel_peak_step", "pca_ratio", "d_init")}
    if "train" in res: out["dyn_train"] = {k: res["train"][k] for k in ("steps", "order", "deriv", "vel_log", "step_mid", "cp_steps", "vel_peak_step", "pca_ratio")}
    ex = te["ex"]; ta, tea = ex["train_acc"], ex["test_acc"]
    ser = dict(train_acc=ta, test_acc=tea, test_loss=ex["test_loss"], train_loss=ex["train_loss"], gap=ta - tea)
    S = {}
    # divergence train-vs-test probe geometry
    if tr is not None:
        Ot, Or = res["test"]["order"], res["train"]["order"]
        ser["fisher_ratio_tr_te"] = Or[:, 0] / np.clip(Ot[:, 0], 1e-12, None)
        ser["nc1_ratio_tr_te"] = Or[:, 1] / np.clip(Ot[:, 1], 1e-12, None)
        cl = np.unique(te["labels"]); ctr = np.unique(tr["labels"])
        ser["cka_classmeans_tr_te"] = np.array([cka_gram(class_mean_centered(a, te["labels"], cl), class_mean_centered(b, tr["labels"], ctr))
                                                 for a, b in zip(te["Zs"], tr["Zs"])])
        if tr["noisy"] is not None:
            cl = np.unique(tr["labels"])
            m = np.array([mem_signal(Z, tr["labels"], tr["noisy"], cl) for Z in tr["Zs"]])
            for j, n in enumerate(("mem_ncm_acc_noisy", "mem_ncm_acc_true", "mem_fit_noisy_on_corrupted", "mem_corrected_on_corrupted")): ser[n] = m[:, j]
    out["series"] = ser
    # scalars
    S.update(run_summary(res["test"], ""))
    if "train" in res: S.update(run_summary(res["train"], "tr_"))
    S["final_test_acc"], S["final_train_acc"] = float(tea[-1]), float(ta[-1]); S["final_gap"] = float(ta[-1] - tea[-1])
    S["max_gap"] = float(np.nanmax(ser["gap"])); S["final_test_loss"] = float(ex["test_loss"][-1]) if np.isfinite(ex["test_loss"][-1]) else np.nan
    S["max_test_acc"] = float(np.nanmax(tea)); S["min_test_loss_step"] = float(steps[np.nanargmin(ex["test_loss"])]) if np.isfinite(ex["test_loss"]).any() else np.nan
    for k in ("fisher_ratio_tr_te", "nc1_ratio_tr_te", "cka_classmeans_tr_te", "mem_fit_noisy_on_corrupted", "mem_corrected_on_corrupted", "mem_ncm_acc_noisy", "mem_ncm_acc_true"):
        if k in ser: S[f"final_{k}"] = float(ser[k][-1])
    if "mem_fit_noisy_on_corrupted" in ser: S["max_mem_fit_noisy_on_corrupted"] = float(np.nanmax(ser["mem_fit_noisy_on_corrupted"]))
    t99 = first_cross(steps, ta, 0.99); t90 = first_cross(steps, tea, 0.9 * np.nanmax(tea))
    S["t_train99"], S["t_test90max"] = t99, t90
    S["lag_steps"] = t90 - t99 if np.isfinite(t99) and np.isfinite(t90) else np.nan
    hx_tr, hx_te = halfmax_x(steps, ta), halfmax_x(steps, tea)
    S["halfmax_train_logstep"], S["halfmax_test_logstep"] = hx_tr / np.log(10), hx_te / np.log(10)
    S["lag_halfmax_log10"] = (hx_te - hx_tr) / np.log(10)
    ors = oriented(res["test"]["order"], tea); anyg = 0; anyg_step = np.nan
    for n, y in ors.items():
        f, st, r = late_rise(steps, y); S[f"late_rise_{n}"] = f; S[f"late_rise_step_{n}"] = st
        if n != "test_acc" and f: anyg = 1; anyg_step = st if not np.isfinite(anyg_step) else min(anyg_step, st)
    S["late_rise_geom_any"] = anyg; S["late_rise_geom_step"] = anyg_step
    S["late_rise_any"] = int(anyg or S["late_rise_test_acc"])

    # ---- slingshot detection + fixed-time / pre-slingshot views
    vel = np.concatenate([[np.nan], res["test"]["vel_log"]]); znorm = np.array([np.linalg.norm(Z, axis=1).mean() for Z in te["Zs"]])
    ser["znorm"] = znorm; ser["vel_snap"] = vel
    evs = detect_slingshots(steps, ex["train_loss"], vel)
    S["slingshot_n_events"] = len(evs); S["slingshot_onset_steps"] = ";".join(str(int(steps[i])) for i in evs)
    S["slingshot_onset_step"] = float(steps[evs[0]]) if evs else np.nan
    # per-event details: loss floor before onset, jump ratio, ||z|| at onset / just before / at step 1000; type N (norm growth) vs F (floor)
    trl = ex["train_loss"]; rmin = np.minimum.accumulate(np.where(np.isfinite(trl), trl, np.inf)); i1k = np.where(steps <= 1000)[0][-1]; events = []
    for i in evs:
        zpre = float(znorm[i - 1]); typ = "N" if zpre > 2 * znorm[i1k] else "F"
        events.append(dict(step=int(steps[i]), floor=float(rmin[i - 1]), jump=float(trl[i] / max(rmin[i - 1], 1e-12)), vel_ratio=float(vel[i] / np.median(vel[max(1, i - SL["vel_win"]):i])),
                           z_onset=float(znorm[i]), z_pre=zpre, z_1000=float(znorm[i1k]), type=typ))
    out["events"] = events
    S["slingshot_detector_mode"] = SL["mode"]
    S["slingshot_floors"] = ";".join(f"{e['floor']:.3g}" for e in events); S["slingshot_jump_ratios"] = ";".join(f"{e['jump']:.3g}" for e in events)
    S["slingshot_znorm_onsets"] = ";".join(f"{e['z_onset']:.4g}" for e in events); S["slingshot_types"] = ";".join(e["type"] for e in events)
    S["slingshot_type"] = events[0]["type"] if events else ""
    S["slingshot_floor"] = events[0]["floor"] if events else np.nan; S["slingshot_jump_ratio"] = events[0]["jump"] if events else np.nan
    S["slingshot_znorm_onset"] = events[0]["z_onset"] if events else np.nan
    for t in "NF":
        S[f"slingshot_n_events_{t}"] = sum(e["type"] == t for e in events)
        S[f"slingshot_onset_step_{t}"] = next((float(e["step"]) for e in events if e["type"] == t), np.nan)
    Q = quantities(ser, res["test"]["order"], ex); Q["nc1"] = res["test"]["order"][:, 1]
    ip = evs[0] - 1 if evs else len(steps) - 1; S["pre_index_step"] = float(steps[ip])
    for n, y in Q.items():
        S[f"pre_{n}"] = float(y[ip])
        for fs in FIXED_STEPS + ((30000,) if steps[-1] >= 30000 else ()):  # 30000 only for the 40k-step families
            j = np.where(steps <= fs)[0]; S[f"at{fs}_{n}"] = float(y[j[-1]]) if len(j) and steps[-1] >= fs else np.nan
    if evs:
        i = evs[0]; j = min(i + 2, len(steps) - 1)
        D = dict(train_loss=ex["train_loss"], test_loss=ex["test_loss"], test_acc=tea, nc1=Q["nc1"], znorm=znorm)
        for n, y in D.items(): S[f"sl_d_{n}"] = float(y[j] - y[i - 1])
        S["sl_d_train_loss_ratio"] = float(ex["train_loss"][i] / max(ex["train_loss"][i - 1], 1e-12)); S["sl_vel_ratio"] = float(vel[i] / np.median(vel[max(1, i - SL["vel_win"]):i]))
        for n, y in Q.items(): S[f"sl_d_{n}"] = float(y[j] - y[i - 1]) if f"sl_d_{n}" not in S else S[f"sl_d_{n}"]
        if len(evs) > 1:  # per-event deltas (all events) for train/test loss and NC1
            S["sl_all_d_test_loss"] = " ".join(f"{ex['test_loss'][min(k+2,len(steps)-1)]-ex['test_loss'][k-1]:.3g}" for k in evs)
            S["sl_all_d_nc1"] = " ".join(f"{Q['nc1'][min(k+2,len(steps)-1)]-Q['nc1'][k-1]:.3g}" for k in evs)
    out["scalars"] = S
    return out


def do_one(args):
    rd, cache_dir, include_partial, force = args
    rid = os.path.basename(rd); t0 = time.time()
    try:
        complete = os.path.isfile(f"{rd}/model_final.pt")
        if not complete and not include_partial: return rid, "skip_incomplete", 0
        if not os.path.isfile(f"{rd}/config.json"): return rid, "skip_nocfg", 0
        h = snap_hash(rd) + ("c" if complete else "p"); cp = f"{cache_dir}/{rid}.pkl"
        if not force and os.path.isfile(cp):
            try:
                if pickle.load(open(cp, "rb")).get("hash") == h: return rid, "cached", 0
            except Exception: pass
        r = analyze_run(rd, include_partial)
        if r is None: return rid, "skip_few_snapshots", 0
        r["hash"] = h; tmp = cp + f".tmp{os.getpid()}"
        pickle.dump(r, open(tmp, "wb")); os.replace(tmp, cp)
        return rid, "ok", time.time() - t0
    except Exception:
        return rid, "FAIL " + traceback.format_exc(), 0


# ---------------------------------------------------------------- aggregation
def cell_key(cfg):
    n = cfg.get("train_subset") or NFULL
    return (float(cfg.get("label_noise") or 0.0), int(n), float(cfg.get("weight_decay") or 0.0))


def ms(v):
    v = np.array([x for x in v if x is not None and np.isfinite(x)], float)
    if len(v) == 0: return np.nan, np.nan, 0
    return float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v)


HM_AT = "pre"
SCALAR_AGG = ["final_fisher", "final_nc1", "final_effrank", "final_etf", "final_ncm_acc", "final_cka_label", "final_test_acc", "final_train_acc",
              "final_gap", "max_gap", "final_test_loss", "lag_steps", "lag_halfmax_log10", "late_rise_any", "late_rise_test_acc", "late_rise_geom_any",
              "t_velpeak", "tchi_fisher", "tchi_nc1", "tchi_effrank", "tchi_etf", "tchi_ncm_acc", "tchi_cka_label", "t_train99", "t_test90max",
              "final_fisher_ratio_tr_te", "final_nc1_ratio_tr_te", "final_cka_classmeans_tr_te", "tr_final_fisher", "tr_final_nc1", "tr_final_ncm_acc",
              "final_mem_fit_noisy_on_corrupted", "final_mem_corrected_on_corrupted", "max_mem_fit_noisy_on_corrupted", "n_cp"]
SCALAR_AGG += ["slingshot_onset_step", "slingshot_n_events", "slingshot_n_events_N", "slingshot_n_events_F", "slingshot_onset_step_N", "slingshot_onset_step_F", "slingshot_floor", "slingshot_jump_ratio", "slingshot_znorm_onset", "sl_d_train_loss", "sl_d_test_loss", "sl_d_test_acc", "sl_d_nc1", "sl_d_znorm", "sl_vel_ratio"] + \
    [f"{p}_{n}" for p in ["pre"] + [f"at{f}" for f in FIXED_STEPS + (30000,)] for n in QNAMES]
PEAK_PARAMS = ORDER + ["test_acc"]


def seed_peak_var(runs):
    """Between-seed variance of each order param across the time axis; returns {param: (peak_var, peak_step, peak_rel)} (needs >=2 seeds)."""
    if len(runs) < 2: return {}
    com = runs[0]["steps"]
    for r in runs[1:]: com = np.intersect1d(com, r["steps"])
    if len(com) < 20: return {}
    out = {}
    for k, n in enumerate(PEAK_PARAMS):
        if n == "test_acc": X = np.stack([r["series"]["test_acc"][np.isin(r["steps"], com)] for r in runs])
        else: X = np.stack([r["dyn_test"]["order"][np.isin(r["steps"], com), k] for r in runs])
        v = X.var(0, ddof=1); v[com == 0] = 0  # exclude init (nearly identical anyway)
        rng2 = (X.mean(0).max() - X.mean(0).min()) ** 2 + 1e-30
        i = v.argmax(); out[n] = (float(v[i]), float(com[i]), float(v[i] / rng2))
    return out


def load_caches(cache_dir, rids):
    R = []
    for rid in rids:
        p = f"{cache_dir}/{rid}.pkl"
        if os.path.isfile(p):
            try: R.append(pickle.load(open(p, "rb")))
            except Exception: pass
    return R


def aggregate(R, out, defaults=True):
    os.makedirs(f"{out}/figures", exist_ok=True)
    # per-run csv
    rows = []
    for r in R:
        c = r["cfg"]; k = cell_key(c)
        rows.append(dict(run_id=r["run_id"], complete=int(r["complete"]), seed=c["seed"], label_noise=k[0], N=k[1], wd=k[2], n_snap=r["n_snap"], **r["scalars"]))
    cols = []
    for d in rows:
        for c in d:
            if c not in cols: cols.append(c)
    with open(f"{out}/phase_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    # cells
    cells = collections.defaultdict(list)
    for r in R: cells[cell_key(r["cfg"])].append(r)
    grid = {}
    for k, rs in sorted(cells.items()):
        g = dict(label_noise=k[0], N=k[1], wd=k[2], n_runs=len(rs), n_complete=sum(r["complete"] for r in rs))
        for s in SCALAR_AGG:
            if s.startswith("at30000_") and not any(s in r["scalars"] for r in rs): continue  # only 40k families have it
            m, sd, n = ms([r["scalars"].get(s, np.nan) for r in rs]); g[s + "_mean"], g[s + "_std"], g[s + "_n"] = m, sd, n
        for n, (pv, ps, pr) in seed_peak_var(rs).items(): g[f"peakvar_{n}"], g[f"peakvar_step_{n}"], g[f"peakvar_rel_{n}"] = pv, ps, pr
        grid[k] = g
    gcols = []
    for g in grid.values():
        for c in g:
            if c not in gcols: gcols.append(c)
    with open(f"{out}/phase_grid.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=gcols); w.writeheader(); w.writerows(grid.values())
    # jumps between adjacent N (resolution-limited), per wd, ln=0
    jr = []
    for wd in sorted({k[2] for k in grid if k[0] == 0}):
        Ns = sorted(k[1] for k in grid if k[0] == 0 and k[2] == wd)
        for a, b in zip(Ns[:-1], Ns[1:]):
            for s in ["final_fisher", "final_nc1", "final_effrank", "final_etf", "final_ncm_acc", "final_cka_label", "final_test_acc", "final_gap"]:
                ga, gb = grid[(0.0, a, wd)], grid[(0.0, b, wd)]
                d = gb[s + "_mean"] - ga[s + "_mean"]; sp = np.sqrt(np.nanmean([ga[s + "_std"] ** 2, gb[s + "_std"] ** 2])) if np.isfinite(ga[s + "_std"]) or np.isfinite(gb[s + "_std"]) else np.nan
                jr.append(dict(wd=wd, N_lo=a, N_hi=b, param=s, delta=d, delta_per_decade_N=d / np.log10(b / a), pooled_seed_std=sp, delta_over_std=d / sp if sp and np.isfinite(sp) and sp > 0 else np.nan))
    if jr:
        with open(f"{out}/phase_jumps.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(jr[0])); w.writeheader(); w.writerows(jr)
    erows = []
    for r in R:
        c = r["cfg"]; k = cell_key(c)
        for j, e in enumerate(r.get("events", [])): erows.append(dict(run_id=r["run_id"], seed=c["seed"], label_noise=k[0], N=k[1], wd=k[2], event_idx=j, mode=r["scalars"].get("slingshot_detector_mode", ""), **e))
    if erows:
        with open(f"{out}/slingshot_events.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(erows[0])); w.writeheader(); w.writerows(erows)
    plots(R, grid, cells, out, defaults)
    onset_vs_wd(R, out)
    return grid


def heat(ax, grid, s, ln, Ns, wds, title, fmt="{:.2f}", cmap="viridis", sub="mean"):
    M = np.full((len(Ns), len(wds)), np.nan); T = [[None] * len(wds) for _ in Ns]
    for i, n in enumerate(Ns):
        for j, w in enumerate(wds):
            g = grid.get((ln, n, w))
            if g is None: continue
            key = s + "_mean" if sub == "mean" else s
            v = g.get(key, np.nan); M[i, j] = v
            if np.isfinite(v):
                sd = g.get(s + "_std", np.nan) if sub == "mean" else np.nan
                T[i][j] = fmt.format(v) + (f"\n±{fmt.format(sd)}" if np.isfinite(sd) else "") + f"\nn={g.get(s + '_n', g['n_runs'])}" if sub == "mean" else fmt.format(v)
    cm = plt.get_cmap(cmap).copy(); cm.set_bad("white")
    im = ax.imshow(np.ma.masked_invalid(M), cmap=cm, aspect="auto", origin="lower")
    ax.set_xticks(range(len(wds))); ax.set_xticklabels([f"{w:g}" for w in wds], fontsize=7); ax.set_yticks(range(len(Ns))); ax.set_yticklabels([str(n) for n in Ns], fontsize=7)
    ax.set_xlabel("weight decay", fontsize=7); ax.set_ylabel("N train", fontsize=7); ax.set_title(title, fontsize=8)
    for i in range(len(Ns)):
        for j in range(len(wds)):
            if T[i][j]: ax.text(j, i, T[i][j], ha="center", va="center", fontsize=5, color="w", bbox=dict(fc="k", alpha=.25, lw=0, pad=.5))
    plt.colorbar(im, ax=ax, fraction=.046)


def plots(R, grid, cells, out, defaults=True):
    fd = f"{out}/figures"
    Ns = [50 * 0 + n for n in (250, 1000, 4000, 16000, NFULL)]; wds = [0.0, 1e-4, 1e-3, 1e-2, 3e-2]
    if not defaults: Ns, wds = [], []
    Ns = sorted(set(Ns) | {k[1] for k in grid if k[0] == 0}); wds = sorted(set(wds) | {k[2] for k in grid if k[0] == 0})
    if not Ns or not wds: return
    stats = [("final_test_acc", "final test acc"), ("final_train_acc", "final train acc"), ("final_gap", "final gap (train-test)"), ("max_gap", "max gap"),
             ("final_fisher", "final Fisher"), ("final_nc1", "final NC1"), ("final_effrank", "final eff. rank"), ("final_etf", "final ETF dist"),
             ("final_ncm_acc", "final NCM acc"), ("final_cka_label", "final CKA(label)"), ("lag_steps", "lag: t(test 90% max) - t(train 99%) [steps]"),
             ("lag_halfmax_log10", "half-max log10 lag (test - train)"), ("late_rise_any", "late rise (frac of seeds)"), ("late_rise_geom_any", "late rise geom (frac)"),
             ("t_velpeak", "log10 step of kernel-velocity max"), ("tchi_fisher", "log10 step max susceptibility Fisher"), ("tchi_nc1", "log10 step max susc. NC1"),
             ("tchi_ncm_acc", "log10 step max susc. NCM"), ("final_fisher_ratio_tr_te", "Fisher train/test probe"), ("final_nc1_ratio_tr_te", "NC1 train/test probe"),
             ("final_cka_classmeans_tr_te", "CKA class-means train vs test probe"), ("final_test_loss", "final test loss")]
    vw = lambda k: k.replace("final_", f"{'at' + HM_AT if HM_AT.isdigit() else HM_AT}_", 1) if k.startswith("final_") and HM_AT != "final" else k
    tag = {"pre": "pre-slingshot", "final": "final"}.get(HM_AT, f"step {HM_AT}")
    stats = [(vw(k), t.replace("final", tag)) for k, t in stats]
    stats += [("slingshot_onset_step", "slingshot onset step (seed mean over seeds WITH event; blank=none)"), ("slingshot_n_events", "slingshot event count (mean)"),
              ("sl_d_test_loss", "slingshot: delta test loss"), ("sl_d_nc1", "slingshot: delta NC1")]
    n = len(stats); nc = 4; nr = (n + nc - 1) // nc
    fig, ax = plt.subplots(nr, nc, figsize=(4.2 * nc, 3.3 * nr)); ax = ax.ravel()
    for a, (s, t) in zip(ax, stats): heat(a, grid, s, 0.0, Ns, wds, t, fmt="{:.3g}")
    for a in ax[n:]: a.axis("off")
    fig.suptitle(f"Phase diagrams (label noise 0), view={tag}: seed mean / ±std / n; blank = no completed run / no event"); fig.tight_layout(rect=(0, 0, 1, .975)); fig.savefig(f"{fd}/heatmaps_{HM_AT}.png", dpi=70); plt.close(fig)
    # seed variance at peak
    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
    for a, nme in zip(ax, PEAK_PARAMS): heat(a, grid, f"peakvar_rel_{nme}", 0.0, Ns, wds, f"peak between-seed var / range^2: {nme}", fmt="{:.2g}", cmap="magma", sub="raw")
    for a in ax[len(PEAK_PARAMS):]: a.axis("off")
    fig.suptitle("Criticality proxy: peak (over time) between-seed variance (needs >=2 seeds)"); fig.tight_layout(); fig.savefig(f"{fd}/seed_variance_peak.png", dpi=70); plt.close(fig)
    # curves vs N per wd
    cps = [(vw(a), b) for a, b in [("final_test_acc", "test acc"), ("final_gap", "gap"), ("final_fisher", "Fisher"), ("final_nc1", "NC1"), ("final_effrank", "eff rank"),
           ("final_etf", "ETF dist"), ("final_ncm_acc", "NCM acc"), ("final_cka_label", "CKA(label)")]]
    fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel(); cols = plt.cm.viridis(np.linspace(0, .9, len(wds)))
    for a, (s, t) in zip(ax, cps):
        for c, w in zip(cols, wds):
            ks = sorted(k for k in grid if k[0] == 0 and k[2] == w)
            if not ks: continue
            a.errorbar([k[1] for k in ks], [grid[k][s + "_mean"] for k in ks], yerr=[np.nan_to_num(grid[k][s + "_std"]) for k in ks], marker="o", ms=3, capsize=2, color=c, label=f"wd={w:g}")
        a.set_xscale("log"); a.set_title(t); a.set_xlabel("N train")
        if s.split("_", 1)[1] in ("fisher", "nc1", "effrank"): a.set_yscale("log")
    ax[0].legend(fontsize=7)
    fig.suptitle("Final order parameter vs N (5 N values: resolution ~0.6 decades of N; only gross jumps detectable, see phase_jumps.csv)"); fig.tight_layout(); fig.savefig(f"{fd}/curves_vs_N.png", dpi=70); plt.close(fig)
    # slingshot onset vs N
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5)); top = 20000 * 1.6
    for c, w in zip(cols, wds):
        rs = [r for r in R if cell_key(r["cfg"])[0] == 0 and cell_key(r["cfg"])[2] == w]
        if not rs: continue
        xs = np.array([cell_key(r["cfg"])[1] for r in rs], float); jit = xs * (1 + 0.04 * (np.array([r["cfg"]["seed"] for r in rs]) - 1))
        on = np.array([r["scalars"]["slingshot_onset_step"] for r in rs]); ce = ~np.isfinite(on)
        ax[0].scatter(jit[~ce], on[~ce], color=c, s=22, label=f"wd={w:g}"); ax[0].scatter(jit[ce], np.full(ce.sum(), top), color=c, marker="v", s=30, facecolors="none")
        ks = sorted({x for x in xs}); mm = [np.nanmean(on[xs == k]) if (~ce[xs == k]).any() else np.nan for k in ks]; ax[0].plot(ks, mm, color=c, lw=.8)
        ax[1].plot(ks, [np.mean([r["scalars"]["slingshot_n_events"] for r in rs if cell_key(r["cfg"])[1] == k]) for k in ks], "o-", color=c, ms=3)
    ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].axhline(20000, c="gray", ls=":"); ax[0].set_xlabel("N train"); ax[0].set_ylabel("slingshot onset step")
    ax[0].set_title("hollow triangles = censored (no event by 20k); line = mean over uncensored seeds", fontsize=8); ax[0].legend(fontsize=7)
    ax[1].set_xscale("log"); ax[1].set_xlabel("N train"); ax[1].set_ylabel("mean event count"); fig.tight_layout(); fig.savefig(f"{fd}/slingshot_onset_vs_N.png", dpi=70); plt.close(fig)
    # trajectories for corner/middle cells
    sel = [(0.0, Ns[0], wds[0]), (0.0, Ns[0], wds[-1]), (0.0, Ns[-1], wds[0]), (0.0, Ns[-1], wds[-1]), (0.0, Ns[len(Ns) // 2], wds[len(wds) // 2])]
    rows = [("acc", "train/test acc"), ("fisher", "Fisher (test probe)"), ("nc1", "NC1"), ("ncm", "NCM acc / CKA"), ("vel", "kernel velocity/dlog"), ("div", "train/test probe divergence")]
    fig, ax = plt.subplots(len(rows), len(sel), figsize=(3.6 * len(sel), 2.3 * len(rows)), squeeze=False)
    for j, k in enumerate(sel):
        ax[0, j].set_title(f"N={k[1]} wd={k[2]:g}", fontsize=9)
        rs = cells.get(k, [])
        if not rs:
            for i in range(len(rows)): ax[i, j].text(.5, .5, "no data", ha="center", transform=ax[i, j].transAxes, color="gray"); ax[i, j].set_xticks([]); ax[i, j].set_yticks([])
            continue
        for ci, r in enumerate(rs):
            s = r["steps"] + 1; c = f"C{ci}"; sr = r["series"]; o = r["dyn_test"]["order"]; lab = f"s{r['cfg']['seed']}"
            ax[0, j].plot(s, sr["train_acc"], c=c, ls="--", lw=1); ax[0, j].plot(s, sr["test_acc"], c=c, lw=1, label=lab)
            ax[1, j].plot(s, o[:, 0], c=c, lw=1); ax[1, j].set_yscale("log"); ax[2, j].plot(s, np.clip(o[:, 1], 1e-3, None), c=c, lw=1); ax[2, j].set_yscale("log")
            ax[3, j].plot(s, o[:, 4], c=c, lw=1); ax[3, j].plot(s, o[:, 5], c=c, lw=1, ls=":")
            ax[4, j].plot(r["dyn_test"]["step_mid"] + 1, r["dyn_test"]["vel_log"], c=c, lw=1)
            if "cka_classmeans_tr_te" in sr: ax[5, j].plot(s, sr["cka_classmeans_tr_te"], c=c, lw=1)
        ax[0, j].legend(fontsize=6)
    for i, (_, t) in enumerate(rows): ax[i, 0].set_ylabel(t, fontsize=7)
    for a in ax.ravel(): a.set_xscale("log"); a.tick_params(labelsize=6)
    fig.suptitle("Representative trajectories (solid test, dashed train; dotted CKA(label); last row: CKA of class-mean structure train vs test probe)", fontsize=9); fig.tight_layout(); fig.savefig(f"{fd}/trajectories_cells.png", dpi=65); plt.close(fig)
    # label noise (wd = 0): N x ln
    lns = sorted({k[0] for k in grid if k[2] == 0.0})
    if len(lns) > 1:
        g2 = {(0.0, n, l): grid[(l, n, 0.0)] for (l, n, w) in grid if w == 0.0}  # remap: 'wd' axis := label noise
        st2 = [("final_test_acc", "final test acc"), ("final_train_acc", "final train acc (noisy labels)"), ("final_gap", "gap"), ("final_mem_fit_noisy_on_corrupted", "frac corrupted items NCM-fit to noisy label"),
               ("final_mem_corrected_on_corrupted", "frac corrupted items NCM-fit to TRUE label"), ("max_mem_fit_noisy_on_corrupted", "max memorization over training"),
               ("final_fisher", "final Fisher"), ("final_ncm_acc", "final NCM acc")]
        fig, ax = plt.subplots(2, 4, figsize=(17, 7)); ax = ax.ravel()
        for a, (s, t) in zip(ax, st2): heat(a, g2, s, 0.0, Ns, lns, t, fmt="{:.3g}"); a.set_xlabel("label noise")
        fig.suptitle("Label-noise axis (wd=0)"); fig.tight_layout(); fig.savefig(f"{fd}/label_noise.png", dpi=70); plt.close(fig)
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        for ci, (k, rs) in enumerate(sorted((k, v) for k, v in cells.items() if k[0] > 0)):
            for r in rs:
                sr = r["series"]
                if "mem_fit_noisy_on_corrupted" not in sr: continue
                s = r["steps"] + 1; ax[0].plot(s, sr["mem_fit_noisy_on_corrupted"], lw=.8, label=f"N{k[1]} ln{k[0]}" if r["cfg"]["seed"] == 0 else None, c=f"C{ci % 10}", ls="-" if k[0] < .3 else "--")
                ax[1].plot(s, sr["test_acc"], lw=.8, c=f"C{ci % 10}", ls="-" if k[0] < .3 else "--"); ax[2].plot(s, sr["train_acc"], lw=.8, c=f"C{ci % 10}", ls="-" if k[0] < .3 else "--")
        ax[0].set_title("NCM memorization of corrupted labels"); ax[1].set_title("test acc"); ax[2].set_title("train acc (noisy labels)"); ax[0].legend(fontsize=5, ncol=2)
        for a in ax: a.set_xscale("log")
        fig.tight_layout(); fig.savefig(f"{fd}/label_noise_trajectories.png", dpi=65); plt.close(fig)


def family_of(cfg):
    t = str(cfg.get("tag") or "")
    return "sl40k" if t.startswith("sl40k") else "sgd40k" if t.startswith("sgd40k") else "eps" if t.startswith("eps") else "sgdlr" if t.startswith("sgdlr") else "grid20k"


def onset_vs_wd(R, out, fname="slingshot_onset_vs_wd.png"):
    """Slingshot onset step vs wd, one panel per N, seeds as points, censored (no event) as hollow triangles at the top."""
    rs0 = [r for r in R if cell_key(r["cfg"])[0] == 0]
    Ns = sorted({cell_key(r["cfg"])[1] for r in rs0})
    if not Ns: return
    top = max(r["steps"][-1] for r in rs0) * 1.6
    fig, ax = plt.subplots(1, len(Ns), figsize=(3.4 * len(Ns), 3.6), squeeze=False); ax = ax[0]
    for a, N in zip(ax, Ns):
        rs = [r for r in rs0 if cell_key(r["cfg"])[1] == N]; wds = sorted({cell_key(r["cfg"])[2] for r in rs})
        xw = lambda w: w if w > 0 else min([x for x in wds if x > 0] or [1e-6]) / 3  # wd=0 drawn left of the log axis
        for w in wds:
            rw = [r for r in rs if cell_key(r["cfg"])[2] == w]
            for t, col in (("N", "C3"), ("F", "C0")):  # all events, colored by mechanism type (N = norm growth, F = loss floor)
                ys = [e["step"] for r in rw for e in r.get("events", []) if e["type"] == t]
                a.scatter(np.full(len(ys), xw(w)) * (1.06 if t == "F" else 0.94), ys, c=col, s=10 if True else 0, alpha=.7, label=f"type {t}" if (w == wds[0]) else None)
            ce = np.array([r["scalars"]["slingshot_n_events"] == 0 for r in rw]); a.scatter(np.full(ce.sum(), xw(w)), np.full(ce.sum(), top), marker="v", s=24, facecolors="none", edgecolors="gray")
        a.set_xscale("log"); a.set_yscale("log"); a.set_title(f"N={N}", fontsize=9); a.set_xlabel("wd (0 at left)", fontsize=8)
    ax[0].set_ylabel("slingshot event steps (red N=norm-growth, blue F=floor; hollow = no event)"); ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{out}/figures/{fname}", dpi=70); plt.close(fig)


def compare_adam_sgd(R_adam, R_sgd, out):
    """Adam (sl40k) vs SGD (sgd40k): per (N, wd) means of slingshot onset/count/final acc/gap + overlay figure."""
    os.makedirs(f"{out}/figures", exist_ok=True)
    def cellmap(R):
        m = collections.defaultdict(list)
        for r in R:
            if cell_key(r["cfg"])[0] == 0: m[cell_key(r["cfg"])[1:]].append(r["scalars"])
        return m
    A, S = cellmap(R_adam), cellmap(R_sgd); rows = []
    for k in sorted(set(A) | set(S)):
        row = dict(N=k[0], wd=k[1])
        for tag, M in (("adam", A), ("sgd", S)):
            ss = M.get(k, [])
            for q in ("slingshot_onset_step", "slingshot_n_events", "final_test_acc", "final_gap", "pre_nc1", "pre_fisher", "final_train_acc"):
                m, sd, n = ms([x.get(q, np.nan) for x in ss]); row[f"{tag}_{q}_mean"], row[f"{tag}_{q}_std"], row[f"{tag}_{q}_n"] = m, sd, n
            row[f"{tag}_frac_with_event"] = float(np.mean([x["slingshot_n_events"] > 0 for x in ss])) if ss else np.nan
            row[f"{tag}_n_runs"] = len(ss)
        rows.append(row)
    if rows:
        with open(f"{out}/adam_vs_sgd.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    Ns = sorted({k[0] for k in set(A) | set(S)})
    if not Ns: return
    fig, ax = plt.subplots(3, len(Ns), figsize=(3.4 * len(Ns), 9), squeeze=False)
    top = max([r["steps"][-1] for r in R_adam + R_sgd] or [1]) * 1.6
    for j, N in enumerate(Ns):
        for tag, M, c, off in (("Adam", A, "C0", .93), ("SGD", S, "C1", 1.07)):
            wds = sorted(k[1] for k in M if k[0] == N)
            if not wds: continue
            xw = lambda w: (w if w > 0 else 3e-7) * off
            for w in wds:
                on = np.array([x["slingshot_onset_step"] for x in M[(N, w)]]); ce = ~np.isfinite(on)
                ax[0, j].scatter(np.full((~ce).sum(), xw(w)), on[~ce], c=c, s=14); ax[0, j].scatter(np.full(ce.sum(), xw(w)), np.full(ce.sum(), top), marker="v", s=22, facecolors="none", edgecolors=c)
            ax[1, j].plot([xw(w) for w in wds], [np.mean([x["slingshot_n_events"] for x in M[(N, w)]]) for w in wds], "o-", c=c, ms=3, label=tag)
            ax[2, j].errorbar([xw(w) for w in wds], [np.nanmean([x["final_test_acc"] for x in M[(N, w)]]) for w in wds], yerr=[np.nanstd([x["final_test_acc"] for x in M[(N, w)]]) for w in wds], c=c, marker="o", ms=3, capsize=2)
        ax[0, j].set_title(f"N={N}", fontsize=9); ax[0, j].set_yscale("log")
        for i in range(3): ax[i, j].set_xscale("log")
        ax[2, j].set_xlabel("wd (0 at left)", fontsize=8)
    ax[0, 0].set_ylabel("onset step (hollow=none)"); ax[1, 0].set_ylabel("mean event count"); ax[2, 0].set_ylabel("final test acc"); ax[1, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(f"{out}/figures/adam_vs_sgd.png", dpi=70); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=RUNS_ROOT); ap.add_argument("--out", default=OUT); ap.add_argument("--runs", default=None, help="comma-separated globs under runs_root/runs (overrides --family)")
    ap.add_argument("--family", default=DEFAULT_FAMILIES, help="comma list of tag families: phase,gap,sl40k,sgd40k (default: the 20k grid phase,gap)")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--include_partial", action="store_true"); ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--sl_mode", choices=["floor", "strict", "abs"], default="floor", help="slingshot detector: loss-floor-relative with local plateau (default), literal running-min (strict), or legacy absolute 1e-4 (abs)"); ap.add_argument("--sl_plateau_fac", type=float, default=SL["plateau_fac"])
    ap.add_argument("--sl_zero_thr", type=float, default=SL["zero_thr"]); ap.add_argument("--sl_n_zero", type=int, default=SL["n_zero"])
    ap.add_argument("--sl_jump", type=float, default=SL["jump"]); ap.add_argument("--sl_vel_mult", type=float, default=SL["vel_mult"])
    ap.add_argument("--heatmap_at", default="pre", help="pre | final | 1000 | 5000 | 15000 (which view the heatmaps/curves use)")
    a = ap.parse_args()
    SL.update(mode=a.sl_mode, plateau_fac=a.sl_plateau_fac, zero_thr=a.sl_zero_thr, n_zero=a.sl_n_zero, jump=a.sl_jump, vel_mult=a.sl_vel_mult); global HM_AT; HM_AT = a.heatmap_at
    assert os.path.abspath(a.out).startswith("/vol/bitbucket/lbastian/latent-dynamics/") and "runs_v2/runs" not in os.path.abspath(a.out), "refusing to write there"
    cache = f"{a.out}/phase_cache"; os.makedirs(cache, exist_ok=True); t0 = time.time()
    globs = a.runs.split(",") if a.runs else [FAMILY_GLOBS[f] for f in a.family.split(",")]
    dirs = [d for d in sorted({d for g in globs for d in glob.glob(f"{a.runs_root}/runs/{g}")}) if os.path.isdir(d) and not os.path.basename(d).startswith("_")]
    jobs = [(d, cache, a.include_partial, a.force) for d in dirs]
    res = []
    if not a.aggregate_only:
        if a.workers > 1:
            import multiprocessing as mp
            with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(do_one, jobs, chunksize=1)
        else: res = [do_one(j) for j in jobs]
        cnt = collections.Counter(s.split()[0] for _, s, _ in res); print(dict(cnt))
        for rid, st, _ in res:
            if st.startswith("FAIL"): print(st, rid)
    status = {rid: st for rid, st, _ in res}
    rids = [os.path.basename(d) for d in dirs if status.get(os.path.basename(d), "cached") in ("ok", "cached")]
    R = load_caches(cache, rids)
    if not a.include_partial: R = [r for r in R if r["complete"]]
    print(f"{len(dirs)} run dirs, {len(R)} aggregated, {sum(r['complete'] for r in R)} complete, {time.time()-t0:.0f}s")
    fams = collections.defaultdict(list)
    for r in R: fams[family_of(r["cfg"])].append(r)
    if fams.get("grid20k"): aggregate(fams["grid20k"], a.out)  # unchanged outputs for the 20k grid
    for fam in ("sl40k", "sgd40k"):
        if fams.get(fam): aggregate(fams[fam], f"{a.out}/{fam}", defaults=False)
    if fams.get("sl40k") or fams.get("sgd40k"): compare_adam_sgd(fams.get("sl40k", []), fams.get("sgd40k", []), f"{a.out}/compare_adam_sgd")


if __name__ == "__main__": main()
