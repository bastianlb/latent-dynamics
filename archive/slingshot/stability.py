"""Optimizer-as-discretization stability analysis for the eps (Adam epsilon sweep) and sgdlr (SGD lr sweep) families. CPU only.

Usage:
  python stability.py [--runs_root /vol/bitbucket/lbastian/latent-dynamics/runs_v2] [--out .../dynamics_v2/stability]
                      [--family eps,sgdlr] [--workers 4] [--force] [--aggregate_only]
Per-run work reuses phase_diagram.analyze_run (floor slingshot detector = default mode, events, order parameters, ||z||) and adds the opt_* snapshot keys
(read tolerantly: any missing key / shape mismatch becomes NaN; key names are provisional, see OPT below).
Quantities:
  SGD : r(t) = lr * lambda_max / (2 (1+beta))                       (beta = momentum)
  Adam: r(t) = lr * lambda_max_precond / ((2+2 b1)/(1-b1)),  b1 = 0.9;   r_crude = median eff_step * lambda_max / ((2+2 b1)/(1-b1))
  r >= 1 means the discretized step is unstable.  Event = phase_diagram floor detector.
  eps* (Adam) = lr * lambda_max_plateau * (1-b1)/(2+2 b1): max effective step ~ lr/eps, so events are predicted for eps < eps*.
Outputs (under --out): stability_runs.csv, stability_events.csv, stability_confusion.csv, eps_cells.csv, sgd_collapse.csv, sgd_eos.csv, figures/*.png, cache/.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, glob, json, csv, time, pickle, collections, re, traceback
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_diagram as PD

RUNS_ROOT = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/stability"
CODE = "stab-4"
GLOBS = {"eps": "mnist_probe_*_eps*_*", "sgdlr": "mnist_probe_*_sgdlr*_*"}
OPT = dict(lam="opt_lambda_max", lam_pre="opt_lambda_max_precond", eff_q="opt_eff_step_q", theta="opt_theta_norm", dtheta="opt_dtheta_norm",
           grad="opt_grad_norm", vhat="opt_vhat_mean", resid="opt_lambda_iters_resid", lam_pre_resid="opt_lambda_max_precond_resid", lr="lr")  # provisional names
B1_DEFAULT = 0.9
METRICS = [("test_acc", None), ("fisher", 0), ("effrank", 2), ("etf", 3), ("ncm_acc", 4), ("cka_label", 5), ("d_init", "d"), ("znorm", "z"), ("theta", "th"), ("tr_ratio", "tr")]


# ------------------------------------------------------------------ loading
def load_opt(rd, steps):
    """Tolerant read of opt_* keys (plus lr) per snapshot; returns {name: array (T,...)} with NaN for anything missing."""
    raw = {n: [None] * len(steps) for n in OPT}
    for t, s in enumerate(steps):
        try:
            with np.load(f"{rd}/latents/step_{int(s):07d}.npz") as z:
                for n, k in OPT.items():
                    if k in z.files:
                        try: raw[n][t] = np.asarray(z[k], dtype=np.float64)
                        except Exception: pass
        except Exception:
            continue
    out = {}
    for n, lst in raw.items():
        shp = next((a.shape for a in lst if a is not None), None)
        if shp is None: out[n] = np.full(len(steps), np.nan); continue
        out[n] = np.stack([a if (a is not None and a.shape == shp) else np.full(shp, np.nan) for a in lst])
    return out


RESID_MAX = 0.05  # drop power-iteration estimates with residual above this (cold starts can be ~5% low)


def clean_lambda(O):
    """lambda_max / lambda_max_precond are the eigenvalue of largest |lambda| (can be negative). r uses positive lambda only:
    negative estimates and estimates with resid > RESID_MAX become NaN (counts kept in O['n_neg'], O['n_resid_dropped'])."""
    O["lam_raw"] = O["lam"].copy(); O["lam_pre_raw"] = O["lam_pre"].copy(); O["n_neg"] = 0; O["n_resid_dropped"] = 0
    for k, rk in (("lam", "resid"), ("lam_pre", "lam_pre_resid")):
        x = O[k].copy(); res = O[rk] if (np.ndim(O[rk]) == 1 and np.isfinite(O[rk]).any()) else np.full(len(x), np.nan)
        neg = np.isfinite(x) & (x <= 0); bad = np.isfinite(res) & (res > RESID_MAX)
        O["n_neg"] += int(neg.sum()); O["n_resid_dropped"] += int((bad & np.isfinite(x)).sum()); x[neg | bad] = np.nan; O[k] = x
    return O


def cfg_params(cfg):
    tag = str(cfg.get("tag") or "")
    opt = str(cfg.get("optimizer") or "adam").lower()
    lr = float(cfg.get("lr") or 1e-3); m = re.match(r"(eps|sgdlr)([0-9.eE+-]+?)_n", tag)
    eps = cfg.get("adam_eps"); eps = float(eps) if eps is not None else (float(m.group(2)) if (m and m.group(1) == "eps") else np.nan)
    if m and m.group(1) == "sgdlr" and cfg.get("lr") is None: lr = float(m.group(2))
    return dict(opt=opt, lr=lr, eps=eps, beta=float(cfg.get("momentum") if cfg.get("momentum") is not None else 0.9),
                b1=float(cfg.get("adam_beta1") if cfg.get("adam_beta1") is not None else B1_DEFAULT), b2=cfg.get("adam_beta2"))


def stability_ratio(P, O):
    """Returns r, r_crude (T,) arrays."""
    lr = O["lr"] if np.isfinite(O["lr"]).any() else np.full(len(O["lam"]), P["lr"])
    lr = np.where(np.isfinite(lr), lr, P["lr"])
    if P["opt"] == "sgd":
        r = lr * O["lam"] / (2 * (1 + P["beta"])); return r, r.copy()
    thr = (2 + 2 * P["b1"]) / (1 - P["b1"]); eq = O["eff_q"]
    q50 = eq[:, 2] if eq.ndim == 2 and eq.shape[1] >= 3 else np.full(len(lr), np.nan)
    q99 = eq[:, 4] if eq.ndim == 2 and eq.shape[1] >= 5 else np.full(len(lr), np.nan)
    O["_rq99"] = q99 * O["lam"] / thr  # proxy using the 99% effective step quantile x lambda_max
    return lr * O["lam_pre"] / thr, q50 * O["lam"] / thr


def snap_hash(rd):
    return PD.snap_hash(rd) + CODE


def do_run(job):
    rd, cache, force, partial = job; rid = os.path.basename(rd)
    try:
        if not partial and not os.path.isfile(f"{rd}/model_final.pt"): return rid, "skip_incomplete"
        h = snap_hash(rd) + ("p" if not os.path.isfile(f"{rd}/model_final.pt") else "c"); cp = f"{cache}/{rid}.pkl"
        if not force and os.path.isfile(cp):
            try:
                if pickle.load(open(cp, "rb")).get("hash") == h: return rid, "cached"
            except Exception: pass
        r = PD.analyze_run(rd, False)
        if r is None: return rid, "skip_few_snapshots"
        d0 = PD.load_def(rd, "latents"); trr = np.full(len(r["steps"]), np.nan)
        if d0 is not None:  # inverse-free tr(Sw)/tr(Sb) on the test probe (aligned to analysed steps)
            pos = {int(s_): k for k, s_ in enumerate(d0["steps"])}; y0 = d0["labels"]; cl = np.unique(y0)
            for t_, s_ in enumerate(r["steps"]):
                if int(s_) in pos:
                    Z = d0["Zs"][pos[int(s_)]]; g_ = Z.mean(0); mus = np.array([Z[y0 == c].mean(0) for c in cl]); pc = np.array([(y0 == c).mean() for c in cl])
                    trs = sum(p_ * ((Z[y0 == c] - m_) ** 2).sum(1).mean() for p_, c, m_ in zip(pc, cl, mus)); trb = (pc * ((mus - g_) ** 2).sum(1)).sum(); trr[t_] = trs / trb
        P = cfg_params(r["cfg"]); O = clean_lambda(load_opt(rd, r["steps"])); rr, rc = stability_ratio(P, O)
        keep = dict(hash=h, run_id=rid, cfg=r["cfg"], P=P, steps=r["steps"], events=r["events"], scalars=r["scalars"], opt=O, r=rr, r_crude=rc, r_q99=O.get("_rq99", np.full(len(rr), np.nan)),
                    series={k: r["series"][k] for k in ("train_loss", "test_acc", "znorm", "vel_snap", "gap") if k in r["series"]},
                    order=r["dyn_test"]["order"], d_init=r["dyn_test"]["d_init"], tr_ratio=trr)
        tmp = cp + f".tmp{os.getpid()}"; pickle.dump(keep, open(tmp, "wb")); os.replace(tmp, cp); return rid, "ok"
    except Exception:
        return rid, "FAIL " + traceback.format_exc()


# ------------------------------------------------------------------ analysis helpers
def auc(score, y):
    ok = np.isfinite(score); s, y = score[ok], y[ok]
    n1, n0 = y.sum(), (1 - y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    rk = rankdata(s); return float((rk[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def event_stats(rn, rx_key):
    """Per-event r diagnostics + per-snapshot (pred, outcome) pairs for the confusion table."""
    steps, r = rn["steps"], rn[rx_key]; ev_idx = [int(np.where(steps == e["step"])[0][0]) for e in rn["events"]]
    rows = []
    for e, i in zip(rn["events"], ev_idx):
        pre = r[i - 1] if i >= 1 else np.nan; cross = [j for j in range(1, i) if r[j] >= 1 > r[j - 1]]
        if not cross and i >= 1 and np.isfinite(r[i - 1]) and r[i - 1] >= 1:  # already unstable before the window: use first snapshot at/above 1
            ab = np.where(r[:i] >= 1)[0]; cross = [int(ab[0])] if len(ab) else []
        lc = cross[-1] if cross else None
        rows.append(dict(event_step=int(steps[i]), type=e["type"], r_pre=float(pre), r_onset=float(r[i]), r_max_prev3=float(np.nanmax(r[max(0, i - 3):i])) if i >= 1 else np.nan,
                         last_cross_step=int(steps[lc]) if lc is not None else np.nan, lead_steps=int(steps[i] - steps[lc]) if lc is not None else np.nan,
                         lead_log10=float(np.log10(steps[i] + 1) - np.log10(steps[lc] + 1)) if lc is not None else np.nan))
    T = len(steps); out = np.zeros(T, int)
    for i in ev_idx: out[i] = 1
    return rows, out


def confusion(runs, key="r"):
    """Pairs (predictor at snapshot j, event at snapshot j+1), step>=20 and finite r. Returns dict per group."""
    res = collections.defaultdict(lambda: dict(tp=0, fp=0, fn=0, tn=0, s=[], y=[], tp3=0, fp3=0, fn3=0, tn3=0))
    for rn in runs:
        r = rn[key]; steps = rn["steps"]; _, out = event_stats(rn, key)
        for grp in (rn["P"]["opt"], "all") + ((f"eps={rn['P']['eps']:g}",) if rn["P"]["opt"] == "adam" else ()):
            g = res[grp]
            for j in range(2, len(steps) - 1):
                if steps[j] < 20 or not np.isfinite(r[j]): continue
                y = out[j + 1]; p = r[j] >= 1; p3 = np.nanmax(r[j - 2:j + 1]) >= 1
                g["tp" if (p and y) else "fp" if p else "fn" if y else "tn"] += 1
                g["tp3" if (p3 and y) else "fp3" if p3 else "fn3" if y else "tn3"] += 1
                g["s"].append(np.nanmax(r[j - 2:j + 1])); g["y"].append(y)
    return res


def plateau(rn, key):
    st = rn["steps"]; x = rn["opt"][key]; ok = (st >= 5000) & np.isfinite(x)
    for e in rn["events"]:
        i = int(np.where(st == e["step"])[0][0]); ok[max(0, i - 1):i + 3] = False
    return float(np.median(x[ok])) if ok.any() else np.nan


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def cellkey(rn):
    P = rn["P"]; c = rn["cfg"]; return (int(c.get("train_subset") or 60000), float(c.get("weight_decay") or 0.0))


# ------------------------------------------------------------------ aggregation
def aggregate(out, R):
    os.makedirs(f"{out}/figures", exist_ok=True)
    for rn in R: rn["_fam"] = "eps" if str(rn["cfg"].get("tag", "")).startswith("eps") else "sgdlr"
    run_rows, ev_rows = [], []
    for rn in R:
        P = rn["P"]; N, wd = cellkey(rn); st = rn["steps"]; O = rn["opt"]; eq = O["eff_q"]
        rows, _ = event_stats(rn, "r"); rows_c, _ = event_stats(rn, "r_crude"); rows_q, _ = event_stats(rn, "r_q99")
        for e, x, xc, xq in zip(rn["events"], rows, rows_c, rows_q): x.update(rcrude_pre=xc["r_pre"], rcrude_onset=xc["r_onset"], rq99_pre=xq["r_pre"], rq99_onset=xq["r_onset"]); ev_rows.append(dict(run_id=rn["run_id"], family=rn["_fam"], optimizer=P["opt"], N=N, wd=wd, eps=P["eps"], lr=P["lr"], seed=rn["cfg"]["seed"], z_pre=e["z_pre"], **x))
        lam_pl, lamp_pl = plateau(rn, "lam"), plateau(rn, "lam_pre")
        row = dict(run_id=rn["run_id"], family=rn["_fam"], optimizer=P["opt"], N=N, wd=wd, eps=P["eps"], lr=P["lr"], momentum=P["beta"], seed=rn["cfg"]["seed"],
                   n_events=len(rn["events"]), n_events_N=sum(e["type"] == "N" for e in rn["events"]), n_events_F=sum(e["type"] == "F" for e in rn["events"]),
                   first_onset=rn["events"][0]["step"] if rn["events"] else np.nan, final_znorm=float(rn["series"]["znorm"][-1]),
                   final_theta_norm=float(O["theta"][-1]) if np.ndim(O["theta"]) == 1 else np.nan, lam_max_plateau=lam_pl, lam_max_precond_plateau=lamp_pl,
                   r_plateau=plateau({**rn, "opt": {**O, "r": rn["r"]}}, "r"), r_max=float(np.nanmax(rn["r"])) if np.isfinite(rn["r"]).any() else np.nan,
                   frac_snapshots_r_ge1=float(np.nanmean(rn["r"] >= 1)) if np.isfinite(rn["r"]).any() else np.nan,
                   final_test_acc=float(rn["series"]["test_acc"][-1]), has_opt_keys=int(np.isfinite(rn["r"]).any()), n_lambda_negative=O.get("n_neg", 0), n_lambda_resid_dropped=O.get("n_resid_dropped", 0),
                   peak_lr_lambda=float(np.nanmax(P["lr"] * O["lam_raw"])) if np.isfinite(O["lam_raw"]).any() else np.nan,
                   peak_lr_lambda_step=float(st[np.nanargmax(O["lam_raw"])]) if np.isfinite(O["lam_raw"]).any() else np.nan)
        if P["opt"] == "adam" and np.isfinite(lam_pl): row["eps_star_pred"] = P["lr"] * lam_pl * (1 - P["b1"]) / (2 + 2 * P["b1"])
        if eq.ndim == 2 and eq.shape[1] >= 5:
            late = st >= 5000
            for j, q in enumerate((1, 10, 50, 90, 99)): row[f"effstep_q{q}_late_median"] = float(np.nanmedian(eq[late, j])) if late.any() else np.nan
            row["effstep_q99_max"] = float(np.nanmax(eq[:, 4]))
        run_rows.append(row)
    wcsv(f"{out}/stability_runs.csv", run_rows); wcsv(f"{out}/stability_events.csv", ev_rows)
    # confusion tables
    conf_rows = []
    for key in ("r", "r_crude", "r_q99"):
        for grp, g in confusion(R, key).items():
            tp, fp, fn, tn = g["tp"], g["fp"], g["fn"], g["tn"]
            conf_rows.append(dict(ratio=key, group=grp, tp=tp, fp=fp, fn=fn, tn=tn, p_event_given_r_ge1=tp / max(tp + fp, 1), p_event_given_r_lt1=fn / max(fn + tn, 1),
                                  tp_last3=g["tp3"], fp_last3=g["fp3"], fn_last3=g["fn3"], tn_last3=g["tn3"], auc=auc(np.array(g["s"]), np.array(g["y"]))))
    wcsv(f"{out}/stability_confusion.csv", conf_rows)
    # per-run "event iff r crosses 1"
    both = collections.Counter((int(r["n_events"] > 0), int(r["r_max"] >= 1) if np.isfinite(r["r_max"]) else -1) for r in run_rows); print("runs (has_event, r_max>=1):", dict(both))
    # eps cells (Adam)
    cells = collections.defaultdict(list)
    for r in run_rows:
        if r["family"] == "eps": cells[(r["N"], r["wd"], r["eps"])].append(r)
    ec = []
    for k, rs in sorted(cells.items()):
        f = lambda key: [x[key] for x in rs if key in x and np.isfinite(x[key])]
        m = lambda v: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else np.nan) if v else (np.nan, np.nan)
        row = dict(N=k[0], wd=k[1], eps=k[2], n_runs=len(rs), event_frac=float(np.mean([x["n_events"] > 0 for x in rs])), mean_events=float(np.mean([x["n_events"] for x in rs])),
                   events_N=sum(x["n_events_N"] for x in rs), events_F=sum(x["n_events_F"] for x in rs))
        for key in ("first_onset", "final_znorm", "final_theta_norm", "lam_max_plateau", "lam_max_precond_plateau", "eps_star_pred", "effstep_q1_late_median", "effstep_q50_late_median",
                    "effstep_q99_late_median", "effstep_q99_max", "r_plateau", "r_max"):
            mu, sd = m(f(key)); row[key + "_mean"], row[key + "_std"] = mu, sd
        ec.append(row)
    wcsv(f"{out}/eps_cells.csv", ec)
    collapse_rows, eos_rows = sgd_collapse(R, run_rows)
    wcsv(f"{out}/sgd_collapse.csv", collapse_rows); wcsv(f"{out}/sgd_eos.csv", eos_rows)
    figures(out, R, run_rows, ec, collapse_rows)


def metric_curve(rn, idx):
    if idx is None: return rn["series"]["test_acc"]
    if idx == "d": return rn["d_init"]
    if idx == "z": return rn["series"]["znorm"]
    if idx == "th": return np.asarray(rn["opt"]["theta"], float)
    if idx == "tr": return rn.get("tr_ratio", np.full(len(rn["steps"]), np.nan))
    return rn["order"][:, idx]


BANDS = [(0.1, 1), (1, 10), (10, 100), (100, 1e9)]  # bands in t = lr*step


def sgd_collapse(R, run_rows):
    """Collapse of seed-mean curves on t = lr*step. Momentum-adjusted t' = lr*step/(1-beta) is t times a constant when beta is the same for all lr
    (it is: 0.9), so it only relabels the axis and cannot change collapse; beta is read from the config and checked."""
    groups = collections.defaultdict(lambda: collections.defaultdict(list))
    for rn in R:
        if rn["_fam"] == "sgdlr": groups[cellkey(rn)][rn["P"]["lr"]].append(rn)
    rows = []; eos = []
    for (N, wd), byl in sorted(groups.items()):
        lrs = sorted(byl)
        if len(lrs) < 2: continue
        betas = {rn["P"]["beta"] for l in lrs for rn in byl[l]}
        tmin = max(lr * 1.0 for lr in lrs); tmax = min(lr * byl[lr][0]["steps"][-1] for lr in lrs)
        if not tmax > tmin: continue
        grid = np.linspace(np.log(tmin), np.log(tmax), 160); tg = np.exp(grid)
        for name, idx in METRICS:
            per = {}; sds = []
            for lr in lrs:
                ys = []
                for rn in byl[lr]:
                    y = metric_curve(rn, idx); st = rn["steps"]; m = (st >= 1) & np.isfinite(y)
                    if m.sum() < 5: continue
                    ys.append(np.interp(grid, np.log(lr * st[m]), y[m]))
                if not ys: break
                per[lr] = np.mean(ys, 0); sds.append(np.std(ys, 0, ddof=1) if len(ys) > 1 else np.full(len(grid), np.nan))
            if len(per) < len(lrs): continue
            C = np.stack([per[l] for l in lrs]); rng = np.ptp(C) + 1e-12; seed_sd = float(np.nanmedian(np.concatenate(sds)))
            pair = lambda a_, b_: np.abs(per[a_] - per[b_])
            dmax_t = np.max([pair(lrs[i], lrs[j]) for i in range(len(lrs)) for j in range(i + 1, len(lrs))], 0)  # worst pair at each t
            row = dict(N=N, wd=wd, metric=name, n_lr=len(lrs), t_lo=tmin, t_hi=tmax, betas=str(sorted(betas)), seed_sd_median=seed_sd, max_dev=float(dmax_t.max()), max_dev_rel_range=float(dmax_t.max() / rng),
                       median_dev=float(np.median(dmax_t)), max_dev_over_seed_sd=float(dmax_t.max() / seed_sd) if seed_sd > 0 else np.nan, median_dev_over_seed_sd=float(np.median(dmax_t) / seed_sd) if seed_sd > 0 else np.nan)
            if len(lrs) >= 4:
                for nm, (a_, b_) in (("pair_lo", (lrs[0], lrs[1])), ("pair_hi", (lrs[-2], lrs[-1]))):
                    d = pair(a_, b_); row[nm + "_max_dev"] = float(d.max()); row[nm + "_median_dev"] = float(np.median(d)); row[nm + "_max_over_seed_sd"] = float(d.max() / seed_sd) if seed_sd > 0 else np.nan
            for lo, hi in BANDS:
                m = (tg >= lo) & (tg < hi)
                if m.any(): row[f"band_{lo:g}_{hi:g}_max_dev_rel_range"] = float(dmax_t[m].max() / rng)
            rows.append(row)
    for rn in R:
        if rn["_fam"] != "sgdlr": continue
        P = rn["P"]; st = rn["steps"]; r = rn["r"]; ok = np.isfinite(r)
        if not ok.any(): continue
        late = ok & (st >= 0.25 * st[-1]); lam = rn["opt"]["lam"]; early = ok & (st >= 20) & (st <= 200)
        eos.append(dict(run_id=rn["run_id"], N=cellkey(rn)[0], wd=cellkey(rn)[1], lr=P["lr"], seed=rn["cfg"]["seed"], r_late_median=float(np.median(r[late])) if late.any() else np.nan,
                        r_max=float(np.nanmax(r)), frac_late_in_0p9_1p1=float(np.mean((r[late] >= 0.9) & (r[late] <= 1.1))) if late.any() else np.nan,
                        lam_early_median=float(np.median(lam[early])) if early.any() else np.nan, lam_late_median=float(np.median(lam[late])) if late.any() else np.nan,
                        lam_threshold=2 * (1 + P["beta"]) / P["lr"], sharpening_ratio_late_over_early=float(np.median(lam[late]) / np.median(lam[early])) if (late.any() and early.any()) else np.nan))
    return rows, eos


def _fig_r(out, R):
    fd = f"{out}/figures"
    # (a) r(t) with events: runs with events first, then without
    ev = [rn for rn in R if rn["events"] and np.isfinite(rn["r"]).any()][:4]; nev = [rn for rn in R if not rn["events"] and np.isfinite(rn["r"]).any()][:4]
    sel = ev + nev
    if sel:
        fig, ax = plt.subplots(2, (len(sel) + 1) // 2, figsize=(4 * ((len(sel) + 1) // 2), 6.5), squeeze=False); ax = ax.ravel()
        for a, rn in zip(ax, sel):
            st = rn["steps"] + 1; a.plot(st, rn["r"], label="r"); a.plot(st, rn["r_crude"], label="r_crude (q50 eff step x lam)", lw=.8, alpha=.7); a.axhline(1, c="k", ls="--", lw=.8)
            for e in rn["events"]: a.axvline(e["step"] + 1, c="r" if e["type"] == "F" else "m", ls=":")
            a.set_xscale("log"); a.set_yscale("log"); P = rn["P"]; a.set_title(f"{rn['run_id'][-30:]}", fontsize=6); a.set_ylabel("r")
        ax[0].legend(fontsize=6); fig.suptitle("stability ratio r(t) (r>=1 unstable); events dotted", fontsize=9); fig.tight_layout(); fig.savefig(f"{fd}/r_t_events.png", dpi=65); plt.close(fig)


def _fig_event_frac(out, ec):
    fd = f"{out}/figures"
    # (b) event fraction vs eps with eps*
    cells = sorted({(c["N"], c["wd"]) for c in ec})
    if cells:
        fig, ax = plt.subplots(1, len(cells), figsize=(3.6 * len(cells), 3.6), squeeze=False); ax = ax[0]
        for a, (N, wd) in zip(ax, cells):
            cs = sorted((c for c in ec if (c["N"], c["wd"]) == (N, wd)), key=lambda c: c["eps"]); e = [c["eps"] for c in cs]
            a.plot(e, [c["event_frac"] for c in cs], "o-", label="event fraction"); a.plot(e, [c["mean_events"] / max(1, max(x["mean_events"] for x in cs)) for c in cs], "s--", ms=3, c="gray", label="mean events (norm.)")
            es = [c["eps_star_pred_mean"] for c in cs if np.isfinite(c["eps_star_pred_mean"])]
            if es: a.axvline(np.mean(es), c="r", ls="-", label=f"eps* = {np.mean(es):.1e}")
            a.set_xscale("log"); a.set_title(f"N={N} wd={wd:g}", fontsize=9); a.set_xlabel("Adam eps"); a.legend(fontsize=6)
        fig.suptitle("event fraction vs eps; red = predicted eps* = lr*lam_max*(1-b1)/(2+2 b1) (events expected below it)", fontsize=8); fig.tight_layout(); fig.savefig(f"{fd}/event_frac_vs_eps.png", dpi=65); plt.close(fig)


def _fig_eps_quantities(out, ec):
    fd = f"{out}/figures"
    cells = sorted({(c["N"], c["wd"]) for c in ec})
    if cells:
        fig, ax = plt.subplots(2, 3, figsize=(14, 7)); ax = ax.ravel()
        for key, a, t in (("final_znorm", ax[0], "final ||z||"), ("final_theta_norm", ax[1], "final ||theta||"), ("effstep_q99_late_median", ax[2], "late eff step q99 (lr/(sqrt(vhat)+eps))"),
                          ("effstep_q50_late_median", ax[3], "late eff step median"), ("lam_max_plateau", ax[4], "lambda_max plateau"), ("first_onset", ax[5], "first event onset")):
            for (N, wd) in cells:
                cs = sorted((c for c in ec if (c["N"], c["wd"]) == (N, wd)), key=lambda c: c["eps"])
                a.errorbar([c["eps"] for c in cs], [c[key + "_mean"] for c in cs], yerr=[np.nan_to_num(c[key + "_std"]) for c in cs], marker="o", ms=3, capsize=2, label=f"N{N} wd{wd:g}")
            a.set_xscale("log"); a.set_title(t, fontsize=8); a.set_xlabel("Adam eps")
            if any(np.isfinite(c[key + "_mean"]) and c[key + "_mean"] > 0 for c in ec): a.set_yscale("log")
        ax[0].legend(fontsize=5); fig.tight_layout(); fig.savefig(f"{fd}/eps_quantities.png", dpi=65); plt.close(fig)


def _fig_sgd_collapse(out, R):
    fd = f"{out}/figures"
    # (c) SGD collapse + (d) lambda*lr vs threshold
    sg = [rn for rn in R if rn["_fam"] == "sgdlr"]
    if sg:
        gk = sorted({cellkey(rn) for rn in sg})[:3]
        fig, ax = plt.subplots(len(gk), 4, figsize=(15, 3.3 * len(gk)), squeeze=False)
        for i, (N, wd) in enumerate(gk):
            for j, (name, idx) in enumerate([("test_acc", None), ("fisher", 0), ("ncm_acc", 4), ("d_init", "d")]):
                for rn in [x for x in sg if cellkey(x) == (N, wd)]:
                    st = rn["steps"]; y = rn["series"]["test_acc"] if idx is None else (rn["d_init"] if idx == "d" else rn["order"][:, idx])
                    ax[i, j].plot(rn["P"]["lr"] * np.maximum(st, 1), y, c=plt.cm.viridis(np.log2(rn["P"]["lr"] / 0.0125) / 3), lw=.8)
                ax[i, j].set_xscale("log"); ax[i, j].set_title(f"{name} vs t=lr*step  (N={N}, wd={wd:g})", fontsize=7)
        fig.suptitle("SGD discretization collapse (color = lr)", fontsize=9); fig.tight_layout(); fig.savefig(f"{fd}/sgd_collapse.png", dpi=65); plt.close(fig)


def _fig_lambda(out, R):
    fd = f"{out}/figures"
    lamr = [rn for rn in R if np.isfinite(rn["opt"]["lam"]).any()][:8]
    if lamr:
        fig, ax = plt.subplots(2, 4, figsize=(15, 6.5)); ax = ax.ravel()
        for a, rn in zip(ax, lamr):
            P = rn["P"]; st = rn["steps"] + 1
            if P["opt"] == "sgd": a.plot(st, P["lr"] * rn["opt"]["lam"], label="lr*lam_max"); thr = 2 * (1 + P["beta"])
            else: a.plot(st, P["lr"] * rn["opt"]["lam_pre"], label="lr*lam_max_precond"); thr = (2 + 2 * P["b1"]) / (1 - P["b1"])
            a.axhline(thr, c="r", ls="--", label=f"threshold {thr:.2f}"); a.set_xscale("log"); a.set_yscale("log"); a.set_title(rn["run_id"][-30:], fontsize=6)
            for e in rn["events"]: a.axvline(e["step"] + 1, c="m", ls=":", lw=.8)
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{fd}/lambda_lr_vs_threshold.png", dpi=65); plt.close(fig)


def _fig_mechanism(out, R):
    """Around events at eps=1e-8, wd=0 (Adam): lr*lambda_precond, eff-step quantiles, ||theta||, ||z||, vhat_mean, with onsets marked."""
    fd = f"{out}/figures"; sel = [rn for rn in R if rn["_fam"] == "eps" and rn["P"]["eps"] == min(x["P"]["eps"] for x in R if x["_fam"] == "eps") and cellkey(rn)[1] == 0.0 and rn["events"]]
    sel = sorted(sel, key=lambda rn: (cellkey(rn)[0], rn["cfg"]["seed"]))[:6]
    if not sel: return
    fig, ax = plt.subplots(len(sel), 5, figsize=(18, 2.6 * len(sel)), squeeze=False)
    for i, rn in enumerate(sel):
        st = rn["steps"] + 1; O = rn["opt"]; P = rn["P"]; thr = (2 + 2 * P["b1"]) / (1 - P["b1"])
        ax[i, 0].plot(st, P["lr"] * O["lam_pre"], label="lr*lam_precond"); ax[i, 0].axhline(thr, c="r", ls="--", lw=.8); ax[i, 0].set_yscale("log")
        for j, q in enumerate((1, 10, 50, 90, 99)): ax[i, 1].plot(st, O["eff_q"][:, j], label=f"q{q}", lw=.9)
        ax[i, 1].set_yscale("log"); ax[i, 2].plot(st, O["theta"]); ax[i, 3].plot(st, rn["series"]["znorm"]); ax[i, 3].set_yscale("log"); ax[i, 4].plot(st, O["vhat"]); ax[i, 4].set_yscale("log")
        ax[i, 0].set_ylabel(rn["run_id"][-22:], fontsize=6)
        for a in ax[i]:
            a.set_xscale("log")
            for e in rn["events"]: a.axvline(e["step"] + 1, c="r" if e["type"] == "F" else "m", ls=":", lw=1)
    for j, t in enumerate(["lr*lam_precond (red = 38)", "eff step quantiles lr/(sqrt(vhat)+eps)", "||theta||", "mean ||z||", "vhat mean"]): ax[0, j].set_title(t, fontsize=8)
    ax[0, 1].legend(fontsize=5); fig.tight_layout(); fig.savefig(f"{fd}/mechanism_eps_min_wd0.png", dpi=60); plt.close(fig)


def figures(out, R, run_rows, ec, collapse_rows):
    """Each figure is guarded: failures are printed (name + traceback) to stderr and listed in OUT/figures_failed.txt."""
    failed = []
    for name, fn, args in (("r_t_events", _fig_r, (out, R)), ("event_frac_vs_eps", _fig_event_frac, (out, ec)), ("eps_quantities", _fig_eps_quantities, (out, ec)),
                           ("sgd_collapse", _fig_sgd_collapse, (out, R)), ("lambda_lr_vs_threshold", _fig_lambda, (out, R)), ("mechanism_eps_min_wd0", _fig_mechanism, (out, R))):
        try: fn(*args)
        except Exception:
            tb = traceback.format_exc(); plt.close("all"); print(f"FIGURE FAILED: {name}\n{tb}", file=sys.stderr); failed.append(f"{name}\n{tb}")
    p = f"{out}/figures_failed.txt"
    if failed: open(p, "w").write("\n".join(failed))
    elif os.path.exists(p): os.remove(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=RUNS_ROOT); ap.add_argument("--out", default=OUT); ap.add_argument("--family", default="eps,sgdlr")
    ap.add_argument("--workers", type=int, default=4); ap.add_argument("--force", action="store_true"); ap.add_argument("--aggregate_only", action="store_true")
    ap.add_argument("--include_partial", action="store_true"); ap.add_argument("--runs", default=None, help="glob (under runs_root/runs) overriding --family")
    ap.add_argument("--sl_mode", choices=["floor", "strict", "abs"], default="floor")
    a = ap.parse_args(); PD.SL["mode"] = a.sl_mode
    assert os.path.abspath(a.out).startswith("/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/"), "bad out"
    os.makedirs(f"{a.out}/cache", exist_ok=True); t0 = time.time()
    dirs = sorted({d for g in (a.runs.split(",") if a.runs else [GLOBS[f] for f in a.family.split(",")]) for d in glob.glob(f"{a.runs_root}/runs/{g}") if os.path.isdir(d)})
    if not a.aggregate_only and dirs:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(do_run, [(d, f"{a.out}/cache", a.force, a.include_partial) for d in dirs], chunksize=1)
        print(dict(collections.Counter(s.split()[0] for _, s in res)))
        for rid, s in res:
            if s.startswith("FAIL"): print(s, rid)
    names = {os.path.basename(d) for d in dirs}; R = []
    for f in sorted(glob.glob(f"{a.out}/cache/*.pkl")):
        try:
            r = pickle.load(open(f, "rb"))
            if r["run_id"] in names: R.append(r)
        except Exception: pass
    print(f"{len(dirs)} run dirs, {len(R)} analysed, {time.time()-t0:.0f}s")
    if R: aggregate(a.out, R)


if __name__ == "__main__": main()
