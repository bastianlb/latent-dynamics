"""lr vs noise scale vs accuracy: analysis of the lrB block (SGD batch-size sweeps) with the B=128 block-G runs as the third line. CPU only.

Usage: python sgd_lrB.py [--runs_root .../runs_v2] [--out .../dynamics_v2/sgd_lrB] [--workers 4] [--include_partial] [--force] [--aggregate_only]
Per-run work reuses stability.do_run (floor detector, order parameters, ||theta||, ||z||, tr(Sw)/tr(Sb)); the cache is OUT/cache.
Runs: lrB_* (B in 32/64/256/512; lines nu_fixed and lr_fixed) plus block-G sgdlr_* runs at B=128 with wd=0 and N in {1000, 4000} (line G).
For each N and each matching rule, fit  log metric = a + b*log lr + c*log nu  (nu = lr/B), on all runs that reach the matching point:
  matched t = lr*step  | matched epochs = step*B/N | matched test accuracy (first crossing of the running max).
Outputs: fits.csv, points.csv, line_slopes.csv, figures/*.png, REPORT.md.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, glob, csv, time, pickle, collections, json
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_diagram as PD
import stability as ST

RUNS_ROOT = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/sgd_lrB"
METRICS = {"fisher": lambda r: r["order"][:, 0], "tr_ratio": lambda r: r["tr_ratio"], "etf": lambda r: r["order"][:, 3], "theta": lambda r: np.asarray(r["opt"]["theta"], float)}
T_STAR = (10.0, 100.0, 240.0)       # matched t = lr*step
E_STAR = (30.0, 100.0, 300.0)       # matched epochs = step*B/N
N_SET = (1000, 4000)


def line_of(r):
    tag = str(r["cfg"].get("tag", ""))
    if tag.startswith("sgdlr"): return "G"
    lr = r["P"]["lr"]; B = int(r["cfg"]["batch_size"]); nu = lr / B
    return "nu_fixed" if abs(nu - 0.025 / 128) < 1e-9 else ("lr_fixed" if abs(lr - 0.025) < 1e-9 else "other")


def load_runs(out, include_partial):
    R = []
    for p in sorted(glob.glob(f"{out}/cache/*.pkl")):
        try: r = pickle.load(open(p, "rb"))
        except Exception: continue
        c = r["cfg"]; tag = str(c.get("tag", ""))
        if not (tag.startswith("lrB") or tag.startswith("sgdlr")): continue
        if int(c.get("train_subset") or 0) not in N_SET or float(c.get("weight_decay") or 0) != 0.0: continue
        if r["P"]["opt"] != "sgd": continue
        r["B"] = int(c["batch_size"]); r["N"] = int(c["train_subset"]); r["lr"] = r["P"]["lr"]; r["nu"] = r["lr"] / r["B"]; r["line"] = line_of(r)
        r["t"] = r["lr"] * r["steps"]; r["epochs"] = r["steps"] * r["B"] / r["N"]; r["complete"] = os.path.isfile(f"{PD.RUNS_ROOT}/runs/{r['run_id']}/model_final.pt")
        R.append(r)
    return R


# ------------------------------------------------------------------ matching
def at_axis(r, axis, target, metric):
    x = r[axis]; y = METRICS[metric](r); m = (x > 0) & np.isfinite(y)
    if not m.any() or x[m].max() < target or x[m].min() > target: return np.nan
    return float(np.interp(np.log(target), np.log(x[m]), y[m]))


def at_acc(r, level, metric):
    acc = np.maximum.accumulate(r["series"]["test_acc"]); i = np.where(acc >= level)[0]
    if len(i) == 0: return np.nan
    i = i[0]; y = METRICS[metric](r)
    if i == 0: return float(y[0])
    w = (level - acc[i - 1]) / (acc[i] - acc[i - 1] + 1e-12); return float(y[i - 1] + w * (y[i] - y[i - 1]))


def ols(X, y):
    X = np.column_stack([np.ones(len(y)), X]); beta, *_ = np.linalg.lstsq(X, y, rcond=None); res = y - X @ beta
    dof = max(len(y) - X.shape[1], 1); s2 = (res ** 2).sum() / dof; cov = s2 * np.linalg.pinv(X.T @ X)
    r2 = 1 - (res ** 2).sum() / (((y - y.mean()) ** 2).sum() + 1e-30); return beta, np.sqrt(np.diag(cov)), r2, s2, len(y)


def build_points(R):
    rows = []; levels = {}
    for N in N_SET:
        rs = [r for r in R if r["N"] == N]
        if not rs: continue
        cell_acc = collections.defaultdict(list)
        for r in rs: cell_acc[(r["B"], r["lr"])].append(np.mean(r["series"]["test_acc"][-5:]))
        amax = min(np.mean(v) for v in cell_acc.values()) - 0.003; levels[N] = (round(amax - 0.03, 3), round(amax, 3))
        for r in rs:
            base = dict(run_id=r["run_id"], N=N, B=r["B"], lr=r["lr"], nu=r["nu"], line=r["line"], seed=r["cfg"]["seed"], complete=int(r["complete"]), t_end=float(r["t"][-1]))
            for met in METRICS:
                for tt in T_STAR: rows.append(dict(base, rule="t", level=tt, metric=met, value=at_axis(r, "t", tt, met)))
                for ee in E_STAR: rows.append(dict(base, rule="epochs", level=ee, metric=met, value=at_axis(r, "epochs", ee, met)))
                for a in levels[N]: rows.append(dict(base, rule="acc", level=a, metric=met, value=at_acc(r, a, met)))
    return rows, levels


def fit_all(rows):
    out = []; slopes = []
    keys = sorted({(r["N"], r["rule"], r["level"], r["metric"]) for r in rows})
    for N, rule, level, met in keys:
        pts = [r for r in rows if (r["N"], r["rule"], r["level"], r["metric"]) == (N, rule, level, met) and np.isfinite(r["value"]) and r["value"] > 0]
        if len(pts) < 8: continue
        y = np.log([p["value"] for p in pts]); llr = np.log([p["lr"] for p in pts]); lnu = np.log([p["nu"] for p in pts])
        if len({p["line"] for p in pts}) < 2 or abs(np.corrcoef(llr, lnu)[0, 1]) > 0.98: continue  # lr and nu not separable (e.g. only the B=128 line reached this point)
        d = dict(N=N, rule=rule, level=level, metric=met, n_runs=len(pts), n_lines=len({p["line"] for p in pts}))
        b, se, r2, s2, n = ols(np.column_stack([llr, lnu]), y); d.update(b_logLR=b[1], se_b=se[1], c_logNU=b[2], se_c=se[2], R2=r2, resid_sd=np.sqrt(s2))
        b1, _, r2a, s2a, _ = ols(llr[:, None], y); b2, _, r2b, s2b, _ = ols(lnu[:, None], y)
        d.update(R2_lr_only=r2a, R2_nu_only=r2b, slope_lr_only=b1[1], slope_nu_only=b2[1])
        verdict = "lr (discretization)" if (abs(b[1]) > 2 * se[1] and abs(b[2]) <= 2 * se[2]) else "nu (noise scale)" if (abs(b[2]) > 2 * se[2] and abs(b[1]) <= 2 * se[1]) else "both" if (abs(b[1]) > 2 * se[1] and abs(b[2]) > 2 * se[2]) else "neither"
        d["verdict"] = verdict; out.append(d)
        for line in ("nu_fixed", "lr_fixed", "G"):
            q = [p for p in pts if p["line"] == line]
            if len({p["lr"] for p in q}) >= 2 and line != "lr_fixed":
                bb, ss, _, _, _ = ols(np.log([p["lr"] for p in q])[:, None], np.log([p["value"] for p in q])); slopes.append(dict(N=N, rule=rule, level=level, metric=met, line=line, x="log lr", slope=bb[1], se=ss[1], n=len(q)))
            if line == "lr_fixed" and len({p["B"] for p in q}) >= 2:
                bb, ss, _, _, _ = ols(np.log([p["B"] for p in q])[:, None], np.log([p["value"] for p in q])); slopes.append(dict(N=N, rule=rule, level=level, metric=met, line=line, x="log B (lr fixed, nu = lr/B)", slope=bb[1], se=ss[1], n=len(q)))
    return out, slopes


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


# ------------------------------------------------------------------ figures
def figures(out, R, rows, fits, levels):
    fd = f"{out}/figures"; os.makedirs(fd, exist_ok=True); col = {"nu_fixed": "C0", "lr_fixed": "C1", "G": "C2", "other": "gray"}
    for name, fn in (("curves", lambda: fig_curves(fd, R, col)), ("matched_points", lambda: fig_points(fd, rows, col)), ("coefficients", lambda: fig_coef(fd, fits))):
        try: fn()
        except Exception:
            import traceback; print("FIGURE FAILED:", name, traceback.format_exc(), file=sys.stderr)


def fig_curves(fd, R, col):
    Ns = sorted({r["N"] for r in R}); fig, ax = plt.subplots(len(Ns), 4, figsize=(17, 4 * len(Ns)), squeeze=False)
    for i, N in enumerate(Ns):
        for j, met in enumerate(METRICS):
            for r in R:
                if r["N"] != N: continue
                m = r["t"] > 0; ax[i, j].plot(r["t"][m], METRICS[met](r)[m], c=col[r["line"]], lw=.7, alpha=.8, ls="-" if r["B"] == 128 else "--")
            ax[i, j].set_xscale("log"); ax[i, j].set_title(f"N={N}: {met} vs t=lr*step", fontsize=8)
    for l, c in col.items(): ax[0, 0].plot([], [], c=c, label=l)
    ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{fd}/curves_vs_t.png", dpi=60); plt.close(fig)


def fig_points(fd, rows, col):
    sel = [(r["N"], "t", 240.0), (r["N"], "epochs", 300.0)] if False else None
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    for i, N in enumerate(N_SET):
        for j, met in enumerate(METRICS):
            pts = [r for r in rows if r["N"] == N and r["rule"] == "t" and r["level"] == 240.0 and r["metric"] == met and np.isfinite(r["value"])]
            for line in ("nu_fixed", "lr_fixed", "G"):
                q = [p for p in pts if p["line"] == line]
                ax[i, j].scatter([p["nu"] for p in q], [p["value"] for p in q], c=col[line], s=14, label=line if (i == 0 and j == 0) else None)
            ax[i, j].set_xscale("log"); ax[i, j].set_xlabel("nu = lr/B"); ax[i, j].set_title(f"N={N}: {met} at t=240 vs nu (color=line)", fontsize=8)
    ax[0, 0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{fd}/metric_vs_nu_t240.png", dpi=60); plt.close(fig)
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), squeeze=False)
    for i, N in enumerate(N_SET):
        for j, met in enumerate(METRICS):
            pts = [r for r in rows if r["N"] == N and r["rule"] == "t" and r["level"] == 240.0 and r["metric"] == met and np.isfinite(r["value"])]
            for line in ("nu_fixed", "lr_fixed", "G"):
                q = [p for p in pts if p["line"] == line]
                ax[i, j].scatter([p["lr"] for p in q], [p["value"] for p in q], c=col[line], s=14)
            ax[i, j].set_xscale("log"); ax[i, j].set_xlabel("lr"); ax[i, j].set_title(f"N={N}: {met} at t=240 vs lr", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/metric_vs_lr_t240.png", dpi=60); plt.close(fig)


def fig_coef(fd, fits):
    if not fits: return
    mets = list(METRICS); rules = [("t", 240.0), ("t", 100.0), ("epochs", 300.0), ("epochs", 100.0), ("acc", None)]
    fig, ax = plt.subplots(2, len(N_SET), figsize=(6 * len(N_SET), 8), squeeze=False)
    for j, N in enumerate(N_SET):
        for k, (par, se) in enumerate((("b_logLR", "se_b"), ("c_logNU", "se_c"))):
            xs, ys, es, lab = [], [], [], []
            for m_i, met in enumerate(mets):
                for r_i, (rule, lev) in enumerate(rules):
                    fs = [f for f in fits if f["N"] == N and f["metric"] == met and f["rule"] == rule and (lev is None or f["level"] == lev)]
                    if rule == "acc" and fs: fs = [max(fs, key=lambda f: f["level"])]
                    if fs: xs.append(m_i * (len(rules) + 1) + r_i); ys.append(fs[0][par]); es.append(2 * fs[0][se]); lab.append(f"{rule}{'' if lev is None else int(lev)}")
            ax[k, j].errorbar(xs, ys, yerr=es, fmt="o", ms=3, capsize=2); ax[k, j].axhline(0, c="k", lw=.5)
            ax[k, j].set_xticks([m_i * (len(rules) + 1) + 2 for m_i in range(len(mets))]); ax[k, j].set_xticklabels(mets)
            ax[k, j].set_title(f"N={N}: {par} (+-2 SE); rules: t240,t100,ep300,ep100,acc", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/coefficients.png", dpi=60); plt.close(fig)


# ------------------------------------------------------------------ report
def report(out, R, fits, slopes, levels, n_complete, n_total, write_md=False):
    L = ["# lrB analysis report (auto-generated)", "", f"Runs analysed: {n_total} (complete {n_complete}); lines: nu_fixed (lr ∝ B), lr_fixed (lr = 0.025), G (B = 128). Fit: log metric = a + b·log lr + c·log ν, ν = lr/B, OLS on runs (seeds as replicates).",
         "b = effect of step size at fixed noise scale; c = effect of noise scale at fixed step size. Verdict: |coef| > 2 SE. Matching levels (acc): " + json.dumps({str(k): v for k, v in levels.items()}), ""]
    if n_complete < n_total: L += ["**Partial data: not all runs finished; treat as a smoke test.**", ""]
    for N in N_SET:
        L += [f"## N = {N}", "", "| rule | level | metric | n | b (log lr) ± SE | c (log ν) ± SE | R² | R² lr-only | R² ν-only | verdict |", "|---|---|---|---|---|---|---|---|---|---|"]
        for f in sorted((f for f in fits if f["N"] == N), key=lambda f: (f["metric"], f["rule"], f["level"])):
            L.append(f"| {f['rule']} | {f['level']:g} | {f['metric']} | {f['n_runs']} | {f['b_logLR']:+.2f} ± {f['se_b']:.2f} | {f['c_logNU']:+.2f} ± {f['se_c']:.2f} | {f['R2']:.2f} | {f['R2_lr_only']:.2f} | {f['R2_nu_only']:.2f} | {f['verdict']} |")
        L.append("")
    L += ["## Within-line slopes (log metric per log unit)", "", "| N | rule | level | metric | line | x | slope ± SE | n |", "|---|---|---|---|---|---|---|---|"]
    for s in sorted(slopes, key=lambda s: (s["N"], s["metric"], s["rule"], s["level"], s["line"])):
        if s["rule"] == "t" and s["level"] == 240.0 or s["rule"] == "acc": L.append(f"| {s['N']} | {s['rule']} | {s['level']:g} | {s['metric']} | {s['line']} | {s['x']} | {s['slope']:+.2f} ± {s['se']:.2f} | {s['n']} |")
    L += ["", "Reading guide: the nu_fixed line isolates lr (noise fixed); the lr_fixed line isolates noise scale (x = log B, so slope vs log ν is the negative of the printed slope); the G line varies both with ν = lr/128.", "Figures: figures/curves_vs_t.png, metric_vs_nu_t240.png, metric_vs_lr_t240.png, coefficients.png."]
    text = "\n".join(L) + "\n"
    if write_md: open(f"{out}/REPORT.md", "w").write(text)
    else: print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_root", default=RUNS_ROOT); ap.add_argument("--out", default=OUT); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--include_partial", action="store_true"); ap.add_argument("--force", action="store_true"); ap.add_argument("--aggregate_only", action="store_true"); ap.add_argument("--write_report_md", action="store_true", help="also write REPORT.md (default: print the report to stdout)")
    a = ap.parse_args(); out = a.out
    assert os.path.abspath(out).startswith("/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/"), "bad out"
    os.makedirs(f"{out}/cache", exist_ok=True); t0 = time.time()
    dirs = sorted({d for g in ("mnist_probe_*_lrB_*", *[f"mnist_probe_*_sgdlr*_n{n}_wd0" for n in N_SET]) for d in glob.glob(f"{a.runs_root}/runs/{g}") if os.path.isdir(d)})
    if not a.aggregate_only and dirs:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(ST.do_run, [(d, f"{out}/cache", a.force, a.include_partial) for d in dirs], chunksize=1)
        print(dict(collections.Counter(s.split()[0] for _, s in res)))
        for rid, s in res:
            if s.startswith("FAIL"): print(s, rid)
    names = {os.path.basename(d) for d in dirs}
    R = [r for r in load_runs(out, a.include_partial) if r["run_id"] in names]
    if not a.include_partial: R = [r for r in R if r["complete"]]
    nl = sum(r["line"] != "G" for r in R); print(f"{len(dirs)} run dirs, {len(R)} analysed ({nl} lrB), {time.time()-t0:.0f}s")
    if not R: return
    rows, levels = build_points(R); fits, slopes = fit_all(rows)
    wcsv(f"{out}/points.csv", rows); wcsv(f"{out}/fits.csv", fits); wcsv(f"{out}/line_slopes.csv", slopes)
    figures(out, R, rows, fits, levels); report(out, R, fits, slopes, levels, sum(r["complete"] for r in R), len(R), a.write_report_md); print("done", f"{time.time()-t0:.0f}s")


if __name__ == "__main__": main()
