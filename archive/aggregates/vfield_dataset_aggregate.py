"""Consistency table MNIST vs Fashion-MNIST vs CIFAR-10 from the vfield analyses.   Usage: python vfield_dataset_aggregate.py [--root DIR]
Identical definitions for every dataset (bands in steps: 100-1000 and >=1000).  Writes ROOT/consistency/{consistency_table.csv, per_run.csv, cfm_law.csv, figures/*.png}.
Sources: MNIST: vfield/decomp/runs (partition), vfield/spectra/runs, vfield/traintest/runs, vfield/mnist_ref/{flip,null}; FM: vfield/fm/<stage>/runs; CIFAR: vfield/cifar/<proj>/<stage>/runs.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, glob, csv, re, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from latentdyn.vfield import spectra_events as EV

ROOT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield"
RUNS = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs"
BANDS = {"mid": (100, 1000), "late": (1000, 10 ** 9)}


def parse(rid):
    m = re.search(r"_s(\d+)_(phase|gap|cfm)_n?(\w+?)(?:_(wd|ln)([\d.e-]+))?$", rid)
    m2 = re.search(r"_s(\d+)_rn18_n(\w+?)_ln([\d.]+)$", rid)
    if m2: seed, N, ln = m2.groups(); return dict(seed=int(seed), fam="cifar", N=50000 if N == "full" else int(N), wd=5e-4, ln=float(ln))
    m = re.search(r"_s(\d+)_(phase|gap)_n(\w+?)_(wd|ln)([\d.e-]+)$", rid)
    if m: seed, fam, N, kind, val = m.groups(); return dict(seed=int(seed), fam=fam, N=60000 if N == "full" else int(N), wd=float(val) if kind == "wd" else 0.0, ln=float(val) if kind == "ln" else 0.0)
    m = re.search(r"_s(\d+)_cfm_(c\d+)$", rid)
    if m:
        cfg = json.load(open(f"{RUNS}/{rid}/config.json")); return dict(seed=int(m.group(1)), fam="cfm", N=int(cfg.get("train_subset") or 60000), wd=float(cfg.get("weight_decay") or 0), ln=float(cfg.get("label_noise") or 0), lr=float(cfg["lr"]), opt=cfg.get("optimizer"))
    return None


def bm(x, st, lo, hi):
    k = (st >= lo) & (st < hi) & np.isfinite(x); return float(np.nanmedian(x[k])) if k.any() else np.nan


def load_set(paths, kind):
    out = {}
    for f in glob.glob(paths):
        b = os.path.basename(f)[:-4]
        if kind == "partition": rid = re.sub(r"_(test|train)$", "", b)
        elif kind == "spectra":
            if not b.endswith("_test_given"): continue
            rid = b[: -len("_test_given")]
        else: rid = b
        if kind == "partition" and not b.endswith("_test"): continue
        out[rid] = np.load(f)
    return out


def metrics_dataset(name, paths):
    P = load_set(paths["partition"], "partition"); S = load_set(paths["spectra"], "spectra"); T = load_set(paths["traintest"], "traintest"); F = load_set(paths["flip"], "flip"); Nl = load_set(paths["null"], "null")
    rows = {}
    for rid, z in P.items():
        info = parse(rid)
        if info is None or info["fam"] == "cfm": continue
        r = rows.setdefault(rid, dict(run_id=rid, **info)); pm = z["part_mean"]; s1 = z["steps"][1:]
        for bn, (lo, hi) in BANDS.items():
            R1, R2, R3k, R3q, R3kn = [bm(pm[i], s1, lo, hi) for i in (0, 1, 2, 3, 6)]; res = 1 - R1
            r.update({f"R1_{bn}": R1, f"classshare_{bn}": (R2 - R1) / res, f"nonlin_krr_share_{bn}": (R3q - R2) / res, f"nonlin_knn_share_{bn}": (R3k - R2) / res, f"nonlin_null_knn_share_{bn}": (R3kn - R2) / res, f"unexplained_{bn}": 1 - R3q})
    for rid, z in S.items():
        info = parse(rid)
        if info is None: continue
        r = rows.setdefault(rid, dict(run_id=rid, **info)); st = z["steps"]; s1 = st[1:]; ab = np.nanmean(z["abscissa_med_res"], 0)[:len(s1)]
        r["abscissa_all_positive"] = float(all((bm(ab, s1, lo, hi) > 0) for lo, hi in ((1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)) if np.isfinite(bm(ab, s1, lo, hi))))
        loc = EV.log_mid(st); evs = []
        for sp in range(z["abscissa_med_res"].shape[0]): evs += EV.ev_cross(z["abscissa_med_res"][sp][:len(loc)], loc)
        r["abscissa_cross_events_per_split"] = len(evs) / z["abscissa_med_res"].shape[0]
        for bn, (lo, hi) in BANDS.items(): r[f"abscissa_{bn}"] = bm(ab, s1, lo, hi)
    for rid, z in Nl.items():
        info = parse(rid)
        if info is None: continue
        r = rows.setdefault(rid, dict(run_id=rid, **info)); s1 = z["steps"][1:]
        for bn, (lo, hi) in {"early": (10, 100), "mid": (100, 1000), "late": (1000, 10 ** 9)}.items():
            r[f"abs_ratio_null_{bn}"] = bm(z["abscissa_data"][:len(s1)] / np.maximum(np.abs(z["abscissa_null"][:len(s1)]), 1e-300), s1, lo, hi)
            r[f"nexp_diff_null_{bn}"] = bm(np.abs(z["n_exp_data"][:len(s1)] - z["n_exp_null"][:len(s1)]), s1, lo, hi); r[f"ncplx_diff_null_{bn}"] = bm(np.abs(z["n_cplx_data"][:len(s1)] - z["n_cplx_null"][:len(s1)]), s1, lo, hi)
    for rid, z in T.items():
        info = parse(rid)
        if info is None: continue
        r = rows.setdefault(rid, dict(run_id=rid, **info)); st = z["steps"]; s1 = st[1:]; Tm = z["transfer_mean"]
        gap_c = Tm[1, 0, 0, 1] - Tm[1, 1, 0, 1]; gap_g = Tm[0, 0, 0, 0] - Tm[0, 1, 0, 0]; acc = z["snap_train_acc"] - z["snap_test_acc"]
        trr = np.log(z["tr_ratio_tr_true_full"] / z["tr_ratio_te_full"]) if "tr_ratio_te_full" in z.files else np.log(z["tr_ratio_tr_true"] / z["tr_ratio_te"])
        for bn, (lo, hi) in BANDS.items():
            r[f"transfer_gap_class_{bn}"] = bm(gap_c, s1, lo, hi); r[f"transfer_gap_global_sep_{bn}"] = bm(gap_g, s1, lo, hi); r[f"acc_gap_{bn}"] = bm(acc, st, lo, hi); r[f"log_trratio_{bn}"] = bm(trr, st, lo, hi)
            if "drag_cos_toM" in z.files: r[f"drag_toM_minus_toC_{bn}"] = bm(z["drag_cos_toM"] - z["drag_cos_toC"], s1, lo, hi)
        if "drag_cos_toM" in z.files:
            d = z["drag_cos_toM"] - z["drag_cos_toC"]; r["drag_frac_pos_ge100"] = float(np.nanmean(d[s1 >= 100] > 0))
    for rid, z in F.items():
        info = parse(rid)
        if info is None: continue
        r = rows.setdefault(rid, dict(run_id=rid, **info)); Np = info["N"]
        r["flip_epoch"] = float(z["flip_step_affine"]) * 128.0 / Np; r["mem_epoch"] = float(z["mem_cross_step"]) * 128.0 / Np
        r["flip_epoch_raw"] = float(z["flip_step_raw"]) * 128.0 / Np; r["flip_epoch_procrustes"] = float(z["flip_step_procrustes"]) * 128.0 / Np
    return rows


def med(rows, key, sel=lambda r: True):
    v = [r[key] for r in rows.values() if key in r and sel(r) and np.isfinite(r[key])]; return (float(np.median(v)), len(v)) if v else (np.nan, 0)


def verdict(kind, val, ref=None, extra=None):
    if not np.isfinite(val): return "n/a"
    if kind == "affine": return "holds" if val >= 0.8 else ("partly" if val >= 0.6 else "fails")
    if kind == "class": return "holds" if 0.10 <= val <= 0.40 else ("partly" if (0.05 <= val < 0.10 or 0.40 < val <= 0.6) else "fails")
    if kind == "nonlin": return "holds" if val <= 0.02 else ("partly" if val <= 0.05 else "fails")
    if kind == "unexp": return "holds" if abs(val - ref) <= 0.05 else ("partly" if abs(val - ref) <= 0.15 else "fails")
    if kind == "gap": return "holds" if 0 < val <= 0.02 else ("partly" if 0 < val <= 0.05 or abs(val) <= 0.02 else "fails")
    if kind == "invN": return "holds" if val <= -0.8 else ("partly" if val <= -0.5 else "fails")
    if kind == "flip": return "holds" if val >= 0.8 else ("partly" if val >= 0.5 else "fails")
    return "n/a"


def build(root):
    sets = {"MNIST": dict(partition=f"{root}/decomp/runs/*_test.npz", spectra=f"{root}/spectra/runs/*_test_given.npz", traintest=f"{root}/traintest/runs/*.npz", flip=f"{root}/mnist_ref/flip/runs/*.npz", null=f"{root}/mnist_ref/null/runs/*.npz"),
            "FM": dict(partition=f"{root}/fm/partition/runs/*_test.npz", spectra=f"{root}/fm/spectra/runs/*_test_given.npz", traintest=f"{root}/fm/traintest/runs/*.npz", flip=f"{root}/fm/flip/runs/*.npz", null=f"{root}/fm/null/runs/*.npz")}
    for P in ("pca16", "pca32", "logit10"):
        sets[f"CIFAR-{P}"] = dict(partition=f"{root}/cifar/{P}/partition/runs/*_test.npz", spectra=f"{root}/cifar/{P}/spectra/runs/*_test_given.npz", traintest=f"{root}/cifar/{P}/traintest/runs/*.npz", flip=f"{root}/cifar/{P}/flip/runs/*.npz", null=f"{root}/cifar/{P}/null/runs/*.npz")
    data = {k: metrics_dataset(k, v) for k, v in sets.items()}
    # MNIST traintest also includes sl40k etc; keep phase + gap only
    for k in data: data[k] = {rid: r for rid, r in data[k].items() if r["fam"] in ("phase", "gap", "cifar")}
    return data


def table(data):
    out = []; mn = data["MNIST"]
    ph = lambda r: r["fam"] == "phase"; allr = lambda r: True
    ref_unexp = med(mn, "unexplained_mid", ph)[0]
    for ds, rows in data.items():
        if not rows: continue
        sel = ph if ds in ("MNIST", "FM") else allr; selg = (lambda r: r["fam"] == "gap") if ds in ("MNIST", "FM") else (lambda r: r["ln"] > 0)
        def add(claim, value, ver, note=""): out.append(dict(dataset=ds, claim=claim, value=value, verdict=ver, note=note))
        v, n = med(rows, "R1_mid", sel); add("shared affine share R1 (steps 100-1000, share of sum d^2)", round(v, 3), verdict("affine", v), f"n={n}; late {med(rows,'R1_late',sel)[0]:.3f}")
        v, n = med(rows, "classshare_mid", sel); add("class-wise affine share of the post-affine residual (100-1000)", round(v, 3), verdict("class", v), f"n={n}; late {med(rows,'classshare_late',sel)[0]:.3f}")
        v, n = med(rows, "nonlin_krr_share_mid", sel); add("nonlinear (sequential kernel) share of residual (100-1000)", round(v, 4), verdict("nonlin", v), f"knn share {med(rows,'nonlin_knn_share_mid',sel)[0]:.4f}; shuffled-null knn {med(rows,'nonlin_null_knn_share_mid',sel)[0]:.4f}")
        v, n = med(rows, "unexplained_mid", sel); add("unexplained share (100-1000)", round(v, 3), verdict("unexp", v, ref_unexp), f"MNIST ref {ref_unexp:.3f}; late {med(rows,'unexplained_late',sel)[0]:.3f}")
        pos = np.nanmean([r["abscissa_all_positive"] for r in rows.values() if "abscissa_all_positive" in r and sel(r)]); ev = np.nanmean([r["abscissa_cross_events_per_split"] for r in rows.values() if "abscissa_cross_events_per_split" in r and sel(r)])
        ratio = med(rows, "abs_ratio_null_mid", sel)[0]; ratio_l = med(rows, "abs_ratio_null_late", sel)[0]; nd = med(rows, "nexp_diff_null_late", sel)[0]
        ok = (pos >= 0.95) and (ev < 0.05); add("leading generator eigenvalue: always positive, no persistent sign change", f"pos={pos:.2f}, sign-change events/split={ev:.2f}", "holds" if ok else ("partly" if pos >= 0.8 else "fails"), f"abscissa/shuffled-null: mid {ratio:.1f}x, late {ratio_l:.1f}x; |n_exp - null| late {nd:.2f}")
        v, n = med(rows, "transfer_gap_class_mid", sel); add("class-wise transfer gap (pooled global, target test), 100-1000", round(v, 4), verdict("gap", v), f"n={n}; late {med(rows,'transfer_gap_class_late',sel)[0]:+.4f}; acc gap {med(rows,'acc_gap_mid',sel)[0]:.3f}")
        # contraction vs 1/N
        rs = [r for r in rows.values() if sel(r) and "log_trratio_late" in r and np.isfinite(r["log_trratio_late"])]
        if ds in ("MNIST", "FM") and len(rs) > 10:
            x = np.array([1.0 / r["N"] for r in rs]); y = np.array([r["log_trratio_late"] for r in rs]); rr = np.corrcoef(x, y)[0, 1]; sl = np.polyfit(x, y, 1)[0]
            add("train/test collapse: corr(log trace-ratio train/test, 1/N) (>=1000)", f"r={rr:.2f}, slope={sl:.0f} per unit 1/N", verdict("invN", rr), f"n={len(rs)}")
        else:
            byN = {}
            for r in rs: byN.setdefault((r["N"], r["ln"]), []).append(r["log_trratio_late"])
            txt = "; ".join(f"ln{ln}: N={N}: {np.mean(v):+.3f}" for (N, ln), v in sorted(byN.items(), key=lambda kv: (kv[0][1], kv[0][0]))); small = [np.mean(v) for (N, ln), v in byN.items() if N == 10000]; big = [np.mean(v) for (N, ln), v in byN.items() if N == 50000]
            dirn = (np.mean(small) < np.mean(big)) if small and big else None; add("train/test collapse vs N (2 points): more collapsed at small N?", txt, "holds (direction)" if dirn else ("fails" if dirn is False else "n/a"), "log trace ratio train/test; negative = train more collapsed")
        fl = [(r["flip_epoch"], r["mem_epoch"]) for r in rows.values() if selg(r) and "flip_epoch" in r and np.isfinite(r["flip_epoch"]) and np.isfinite(r["mem_epoch"])]
        if fl:
            fr = np.mean([a < b for a, b in fl]); add("field flip before NCM memorisation crossover (fraction of gap runs)", f"{fr:.2f} (n={len(fl)})", verdict("flip", fr), f"median flip {np.median([a for a,b in fl]):.1f} ep vs memorisation {np.median([b for a,b in fl]):.1f} ep")
        else: add("field flip before NCM memorisation crossover", "no flip detected", "fails", "no persistent sign flip / no crossover")
        dl = [r["drag_toM_minus_toC_late"] for r in rows.values() if selg(r) and "drag_toM_minus_toC_late" in r and np.isfinite(r["drag_toM_minus_toC_late"])]; dfp = [r["drag_frac_pos_ge100"] for r in rows.values() if selg(r) and "drag_frac_pos_ge100" in r and np.isfinite(r["drag_frac_pos_ge100"])]
        if dl:
            v = float(np.median(dl)); f_ = float(np.median(dfp)); add("no drag of test points toward mislabelled cluster (late cos diff; frac steps>0)", f"{v:+.3f}; frac+={f_:.2f}", "holds" if (v <= 0.05 and f_ <= 0.55) else ("partly" if v <= 0.1 else "fails"), f"early(100-1000) {med(rows,'drag_toM_minus_toC_mid',selg)[0]:+.3f}")
    return out


def cfm_law(root, data):
    fm = data["FM"]; rs = [r for r in fm.values() if r["fam"] == "phase" and "log_trratio_late" in r and np.isfinite(r["log_trratio_late"])]
    if len(rs) < 10: return None
    x = np.array([1.0 / r["N"] for r in rs]); y = np.array([r["log_trratio_late"] for r in rs]); b, a = np.polyfit(x, y, 1)
    T = load_set(f"{root}/fm/traintest/runs/*cfm*.npz", "traintest"); rows = []
    for rid, z in T.items():
        info = parse(rid); st = z["steps"]; trr = np.log(z["tr_ratio_tr_true"] / z["tr_ratio_te"]); obs = bm(trr, st, 1000, 10 ** 9)
        rows.append(dict(run_id=rid, N=info["N"], wd=info["wd"], ln=info["ln"], lr=info["lr"], opt=info["opt"], obs=obs, pred=a + b / info["N"]))
    ok = [r for r in rows if np.isfinite(r["obs"])]; obs = np.array([r["obs"] for r in ok]); pred = np.array([r["pred"] for r in ok])
    stats = dict(n=len(ok), a=a, b=b, r_obs_pred=float(np.corrcoef(obs, pred)[0, 1]), rmse=float(np.sqrt(np.mean((obs - pred) ** 2))), rmse_const=float(np.sqrt(np.mean((obs - obs.mean()) ** 2))), r2_oos=float(1 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)),
                 corr_obs_invN=float(np.corrcoef(obs, [1.0 / r["N"] for r in ok])[0, 1]))
    with open(f"{root}/consistency/cfm_law.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    return stats, rows


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default=ROOT); a = ap.parse_args(); root = a.root; os.makedirs(f"{root}/consistency/figures", exist_ok=True)
    data = build(root); tab = table(data)
    with open(f"{root}/consistency/consistency_table.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "claim", "value", "verdict", "note"]); w.writeheader(); w.writerows(tab)
    allrows = [r for rows in data.values() for r in rows.values()]
    for ds, rows in data.items():
        for r in rows.values(): r["dataset"] = ds
    cols = []
    for ds, rows in data.items():
        for r in rows.values():
            for c in r:
                if c not in cols: cols.append(c)
    with open(f"{root}/consistency/per_run.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for ds, rows in data.items(): w.writerows(rows.values())
    cl = cfm_law(root, data)
    for t in tab: print(f"{t['dataset']:14s} | {t['claim'][:70]:70s} | {str(t['value'])[:60]:60s} | {t['verdict']:16s} | {t['note']}")
    if cl: print("CFM out-of-sample 1/N law:", cl[0])
    try: figs(root, data, cl)
    except Exception:
        import traceback; msg = traceback.format_exc(); print("FIGURE FAILED", msg, file=sys.stderr); open(f"{root}/consistency/figures_failed.txt", "w").write(msg)


def figs(root, data, cl):
    fd = f"{root}/consistency/figures"; fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for ds, mk, col in (("MNIST", "o", "C0"), ("FM", "s", "C1")):
        rs = [r for r in data[ds].values() if r["fam"] == "phase" and "log_trratio_late" in r and np.isfinite(r["log_trratio_late"])]
        if rs: ax[0].scatter([1.0 / r["N"] for r in rs], [r["log_trratio_late"] for r in rs], marker=mk, c=col, s=16, label=ds)
    for P, col in (("CIFAR-pca16", "C3"), ("CIFAR-pca32", "C4"), ("CIFAR-logit10", "C5")):
        rs = [r for r in data[P].values() if "log_trratio_late" in r and np.isfinite(r["log_trratio_late"])]
        if rs: ax[0].scatter([1.0 / r["N"] for r in rs], [r["log_trratio_late"] for r in rs], marker="^", c=col, s=20, label=P)
    if cl:
        stats, rows = cl; ax[0].scatter([1.0 / r["N"] for r in rows if np.isfinite(r["obs"])], [r["obs"] for r in rows if np.isfinite(r["obs"])], marker="x", c="k", s=20, label="FM cfm (off-grid)")
        xs = np.linspace(0, 1 / 250, 50); ax[0].plot(xs, stats["a"] + stats["b"] * xs, "k--", lw=.7)
        ax[1].scatter([r["pred"] for r in rows if np.isfinite(r["obs"])], [r["obs"] for r in rows if np.isfinite(r["obs"])], c="k", s=16); lim = [-1.4, 0.1]; ax[1].plot(lim, lim, "k--", lw=.5); ax[1].set_xlabel("predicted by FM-grid 1/N law"); ax[1].set_ylabel("observed (cfm configs)"); ax[1].set_title(f"out-of-sample: r={stats['r_obs_pred']:.2f}, R2={stats['r2_oos']:.2f}", fontsize=8)
    ax[0].set_xscale("log"); ax[0].set_xlabel("1/N"); ax[0].set_ylabel("log trace-ratio train/test (steps>=1000)"); ax[0].legend(fontsize=6); ax[0].axhline(0, c="gray", lw=.4)
    for ds, mk, col in (("MNIST", "o", "C0"), ("FM", "s", "C1")):
        rs = [r for r in data[ds].values() if r["fam"] == "gap" and "flip_epoch" in r and np.isfinite(r["flip_epoch"]) and np.isfinite(r["mem_epoch"])]
        if rs: ax[2].scatter([r["mem_epoch"] for r in rs], [r["flip_epoch"] for r in rs], marker=mk, c=col, s=16, label=ds)
    for P, col in (("CIFAR-pca16", "C3"), ("CIFAR-pca32", "C4"), ("CIFAR-logit10", "C5")):
        rs = [r for r in data[P].values() if "flip_epoch" in r and np.isfinite(r["flip_epoch"]) and np.isfinite(r["mem_epoch"])]
        if rs: ax[2].scatter([r["mem_epoch"] for r in rs], [r["flip_epoch"] for r in rs], marker="^", c=col, s=20, label=P)
    ax[2].plot([0.3, 300], [0.3, 300], "k--", lw=.5); ax[2].set_xscale("log"); ax[2].set_yscale("log"); ax[2].set_xlabel("memorisation crossover epoch"); ax[2].set_ylabel("field-flip epoch (affine frame)"); ax[2].legend(fontsize=6)
    fig.tight_layout(); fig.savefig(f"{fd}/consistency_overview.png", dpi=60); plt.close(fig)


if __name__ == "__main__": main()
