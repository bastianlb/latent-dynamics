"""Does the MLP-AE topological rebound track overfitting? Reads topo + geom caches only (CPU), writes numbers, tables and a figure.
Reads: ae_dynamics/analysis/{topo,geom}/<rid>[__train].npz, analysis_v2/cache classifier topo caches.
Writes: ae_dynamics/analysis/topo/rebound_numbers.json, REBOUND_TAKEAWAYS.md (tables + REBOUND_VERDICT.md if present), dynamics_report/figures/rebound_vs_gap.{pdf,png}
Conventions: the last snapshot is dropped. 'end' = median over masked snapshots with step > 4000 (= epoch > 500 at N=1000). Minimum searched over step >= 2 (= epoch >= 0.25 at N=1000).
Step thresholds are used instead of epoch thresholds because epoch = step * 125 / N differs across N."""
from latentdyn._threads import cap_threads; cap_threads(4)
import os, json, warnings
import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
BV = "/vol/bitbucket/lbastian/latent-dynamics"; TOPO = f"{BV}/ae_dynamics/analysis/topo"; GEOM = f"{BV}/ae_dynamics/analysis/geom"
CACHE = f"{BV}/analysis_v2/cache"; FIG = f"{BV}/dynamics_report/figures"
DS = ("mnist", "fashion_mnist"); SEEDS = (0, 1, 2); NS = ("250", "1000", "4000", "full")
TM = ("h1_entropy", "h1_total", "pmag_fixed"); PRIM = "h1_total"
STEP_MIN, STEP_END = 2, 4000
COVS = ("gap", "fit", "pr_dim", "twonn_dim", "top2_var", "knn_overlap_pixel")
CL, AE, LIN = "#d62728", "#1f77b4", "#7f7f7f"


def ld(f):
    return dict(np.load(f)) if os.path.isfile(f) else None


def load_run(model, ds, s, N, wd):
    if model == "clf":
        rid = f"{ds}_probe_none_d16_s{s}_phase_n{N}_wd0"; z = ld(f"{CACHE}/{rid}.npz"); zt = z
        g = ld(f"{GEOM}/{rid}.npz")
    else:
        rid = f"{ds}_ae_{model}_d16_s{s}_n{N}_wd{wd}"; z = ld(f"{TOPO}/{rid}.npz"); zt = ld(f"{TOPO}/{rid}__train.npz"); g = ld(f"{GEOM}/{rid}.npz")
    if z is None: return None
    r = dict(model=model, ds=ds, s=s, N=N, wd=wd, rid=rid, step=z["step"], ep=z["run/epoch"], mask=z["topo_mask"].astype(bool))
    r["top"] = {m: z[f"test/{m}"].astype(float) for m in TM}
    r["trtop"] = {m: zt[f"train/{m}"].astype(float) for m in TM} if (zt is not None and "train/h1_total" in zt) else None
    if model == "clf":
        rt, rr = z["run/test_loss"].astype(float), z["run/train_loss"].astype(float)
    else:
        rt, rr = z["run/recon_test"].astype(float), z["run/recon_train"].astype(float)
    r["rt"], r["rr"] = rt, rr; r["gapt"] = rt / rr - 1
    r["geom"] = {k: g[k].astype(float) for k in g if k.split("/")[0] in ("test", "train", "cross")} if g is not None else {}
    return r


def med(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; return float(np.median(x)) if len(x) else np.nan


def analyze(r):
    n = len(r["step"]); idx = np.arange(n); st, ep = r["step"], r["ep"]; last = n - 1
    inner = idx < last; mk = r["mask"] & inner
    E = mk & (st > STEP_END); cand = mk & (st >= STEP_MIN)
    gt = r["gapt"]; o = dict(n_end=int(E.sum()))
    k = np.where((gt > 0.10) & inner)[0]; o["gap_onset"] = float(ep[k[0]]) if len(k) else np.nan
    o["gap_onset_step"] = float(st[k[0]]) if len(k) else np.nan
    rt = np.where(inner, r["rt"], np.inf); imin = int(np.argmin(rt)); o["test_min_ep"] = float(ep[imin]); o["test_min_step"] = float(st[imin])
    o["final_gap"] = med(gt[E]); o["test_excess"] = med(r["rt"][E]) / r["rt"][imin] - 1
    g = r["geom"]
    for q in ("pr_dim", "twonn_dim"):
        if f"test/{q}" in g:
            v = np.where(inner & (st >= STEP_MIN), g[f"test/{q}"], np.inf); j = int(np.argmin(v)); o[f"{q}_min_ep"] = float(ep[j])
        else: o[f"{q}_min_ep"] = np.nan
    o["m"] = {}
    for m in TM:
        v = r["top"][m]; a = np.where(cand, v, np.inf)
        if not np.isfinite(a).any(): o["m"][m] = {}; continue
        j = int(np.argmin(a)); vend = med(v[E]); fb = vend - v[j]; d0 = v[0] - v[j]
        d = dict(e_min=float(ep[j]), e_min_step=float(st[j]), i_min=j, m_min=float(v[j]), m0=float(v[0]), m_end=vend, final_rebound=float(fb),
                 rel_rebound=float(fb / d0) if (np.isfinite(d0) and d0 > 1e-12) else np.nan)
        t = mk & (idx > j); rb = np.where(t, v - v[j], np.nan)
        for q, nm in ((0.5, "e50"), (0.9, "e90")):
            kk = np.where(t & (rb >= q * fb))[0] if fb > 0 else []
            d[nm] = float(ep[kk[0]]) if len(kk) else np.nan
        o["m"][m] = d
        pj = np.nan
        if "test/pr_dim" in g: pj = med(g["test/pr_dim"][E]) - g["test/pr_dim"][j]
        d["pr_rise"] = float(pj)
    return o


def rank(x): return np.argsort(np.argsort(x)).astype(float)


def sp(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float); ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 4 or np.ptp(x[ok]) == 0 or np.ptp(y[ok]) == 0: return np.nan, np.nan, int(ok.sum())
    r, p = spearmanr(x[ok], y[ok]); return float(r), float(p), int(ok.sum())


def partial_sp(x, y, z):
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    if ok.sum() < 5: return np.nan
    a, b, c = rank(x[ok]), rank(y[ok]), rank(z[ok])
    if np.ptp(c) == 0 or np.ptp(a) == 0 or np.ptp(b) == 0: return np.nan
    A = np.c_[np.ones_like(c), c]; ra = a - A @ np.linalg.lstsq(A, a, rcond=None)[0]; rb = b - A @ np.linalg.lstsq(A, b, rcond=None)[0]
    if ra.std() < 1e-9 or rb.std() < 1e-9: return np.nan
    return float(np.corrcoef(ra, rb)[0, 1])


def within(x, y, grp):
    """Rank inside each group, centre, pool, then Pearson on the pooled centred ranks."""
    X, Y = [], []
    for gname in sorted(set(grp)):
        ii = [i for i, gg in enumerate(grp) if gg == gname and np.isfinite(x[i]) and np.isfinite(y[i])]
        if len(ii) < 3: continue
        a, b = rank(np.array([x[i] for i in ii])), rank(np.array([y[i] for i in ii])); X += list(a - a.mean()); Y += list(b - b.mean())
    if len(X) < 4 or np.std(X) == 0 or np.std(Y) == 0: return np.nan, len(X)
    return float(np.corrcoef(X, Y)[0, 1]), len(X)


def f2(x, d=3):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{d}g}"


def rng(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return "n/a" if not len(v) else f"{np.median(v):.3g} [{v.min():.3g}-{v.max():.3g}] (n={len(v)})"


def iqr(v):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    return "n/a" if not len(v) else f"{np.median(v):+.2f} [{np.percentile(v,25):+.2f}, {np.percentile(v,75):+.2f}] (n={len(v)})"


def main():
    runs = []
    for model, wds in (("mlp", ("0", "1e-4")), ("linear", ("0",)), ("clf", ("0",))):
        for ds in DS:
            for N in (NS + ("16000",) if model == "clf" else NS):
                for wd in wds:
                    for s in SEEDS:
                        r = load_run(model, ds, s, N, wd)
                        if r is not None: runs.append(r)
    for r in runs: r["A"] = analyze(r)
    sel = lambda model, wd="0", N=None: [r for r in runs if r["model"] == model and r["wd"] == wd and (N is None or r["N"] == N)]
    J = {"runs": {r["rid"]: r["A"] for r in runs}}; md = []
    logN = lambda r: np.log(60000 if r["N"] == "full" else float(r["N"]))
    print("loaded", {(m, wd): len(sel(m, wd)) for m, wd in (("mlp", "0"), ("mlp", "1e-4"), ("linear", "0"), ("clf", "0"))}, flush=True)

    # ---- A rebound magnitude
    md.append("## A. Rebound magnitude (median [min-max] over runs, test probe)\n")
    md += ["| model | N | n | metric | e_min | m(0) | m(e_min) | m(end) | final rebound | relative rebound |", "|" + "---|" * 10]
    J["magnitude"] = {}
    for lab, mdl, wd in (("MLP wd0", "mlp", "0"), ("MLP wd1e-4", "mlp", "1e-4"), ("linear wd0", "linear", "0"), ("classifier wd0", "clf", "0")):
        for N in (NS if mdl != "clf" else NS):
            R = sel(mdl, wd, N)
            for m in TM:
                g = lambda k: [r["A"]["m"][m][k] for r in R if r["A"]["m"].get(m)]
                J["magnitude"][f"{lab}|{N}|{m}"] = {k: g(k) for k in ("e_min", "m0", "m_min", "m_end", "final_rebound", "rel_rebound")}
                if m == PRIM or mdl == "mlp" and wd == "0":
                    md.append(f"| {lab} | {N} | {len(R)} | {m} | {rng(g('e_min'))} | {rng(g('m0'))} | {rng(g('m_min'))} | {rng(g('m_end'))} | {rng(g('final_rebound'))} | {rng(g('rel_rebound'))} |")
    md.append("")

    # ---- B timing
    md.append("## B. Timing (epochs of each run's own N, median [min-max] over the runs, MLP AE wd0)\n")
    md.append("Gap onset is the first epoch with test/train recon error minus one above 0.10. Test-min is the argmin of test recon error. Rebound metrics use the test probe.\n")
    hdr = "| N | n | " + " | ".join(f"{m} e_min | {m} e50 | {m} e90" for m in TM) + " | gap onset | test-min | pr_dim min | twonn min |"
    md += [hdr, "|" + "---|" * (hdr.count("|") - 1)]
    J["timing"] = {}
    for N in NS:
        R = sel("mlp", "0", N); row = [N, str(len(R))]; J["timing"][N] = {}
        for m in TM:
            for k in ("e_min", "e50", "e90"):
                v = [r["A"]["m"][m].get(k, np.nan) if r["A"]["m"].get(m) else np.nan for r in R]; row.append(rng(v)); J["timing"][N][f"{m}_{k}"] = v
        for k in ("gap_onset", "test_min_ep", "pr_dim_min_ep", "twonn_dim_min_ep"):
            v = [r["A"][k] for r in R]; row.append(rng(v)); J["timing"][N][k] = v
        md.append("| " + " | ".join(row) + " |")
    # step-clock table
    def stp(r, e):
        return float(r["step"][int(np.argmin(np.abs(r["ep"] - e)))]) if np.isfinite(e) else np.nan
    md.append("\n### B2. Same timings on a step clock (optimizer steps, median [min-max], MLP AE wd0)\n")
    cols2 = [f"{m} {k}" for m in TM for k in ("e_min", "e50", "e90")] + ["pr_dim min", "twonn min", "gap onset", "test-min"]
    md += ["| N | n | " + " | ".join(cols2) + " |", "|" + "---|" * (len(cols2) + 2)]
    J["timing_steps"] = {}; STM = {}; EPM = {}
    for N in NS:
        R = sel("mlp", "0", N); d = {}
        for m in TM:
            for k in ("e_min", "e50", "e90"): d[f"{m} {k}"] = [stp(r, r["A"]["m"][m][k]) if r["A"]["m"].get(m) else np.nan for r in R]
        d["pr_dim min"] = [stp(r, r["A"]["pr_dim_min_ep"]) for r in R]; d["twonn min"] = [stp(r, r["A"]["twonn_dim_min_ep"]) for r in R]
        d["gap onset"] = [r["A"]["gap_onset_step"] for r in R]; d["test-min"] = [r["A"]["test_min_step"] for r in R]
        J["timing_steps"][N] = d
        md.append(f"| {N} | {len(R)} | " + " | ".join(rng(d[c]) for c in cols2) + " |")
        for c in cols2: STM[(N, c)] = med(d[c])
        for m in TM:
            for k in ("e_min", "e50", "e90"): EPM[(N, f"{m} {k}")] = med([r["A"]["m"][m][k] if r["A"]["m"].get(m) else np.nan for r in R])
        EPM[(N, "pr_dim min")] = med([r["A"]["pr_dim_min_ep"] for r in R]); EPM[(N, "twonn min")] = med([r["A"]["twonn_dim_min_ep"] for r in R])
        EPM[(N, "gap onset")] = med([r["A"]["gap_onset"] for r in R]); EPM[(N, "test-min")] = med([r["A"]["test_min_ep"] for r in R])
    md.append("\nRatio of the largest to the smallest median across N (250, 1000, 4000, full). A step-clock quantity has a ratio near 1 in steps and near 240 in epochs.\n")
    md += ["| quantity | ratio in steps | ratio in epochs |", "|---|---|---|"]; J["clock_ratio"] = {}
    for c in cols2:
        a = [STM[(N, c)] for N in NS]; b = [EPM[(N, c)] for N in NS]
        ra = np.nanmax(a) / np.nanmin(a) if np.isfinite(a).any() else np.nan; rb = np.nanmax(b) / np.nanmin(b) if np.isfinite(b).any() else np.nan
        J["clock_ratio"][c] = dict(steps=ra, epochs=rb); md.append(f"| {c} | {f2(ra,3)} | {f2(rb,3)} |")
    md.append("\nGeneralisation gap (test over train recon error minus one) at the h1_total rebound timing points, median over runs:\n")
    md += ["| N | gap at e_min | gap at e50 | gap at e90 | gap at onset threshold |", "|---|---|---|---|---|"]
    for N in NS:
        R = [r for r in sel("mlp", "0", N) if r["A"]["m"].get(PRIM)]
        gv = lambda k: [r["gapt"][int(np.argmin(np.abs(r["ep"] - r["A"]["m"][PRIM][k])))] for r in R if np.isfinite(r["A"]["m"][PRIM][k])]
        J["timing"][N]["gap_at"] = {k: gv(k) for k in ("e_min", "e50", "e90")}
        md.append(f"| {N} | {f2(med(gv('e_min')),2)} | {f2(med(gv('e50')),2)} | {f2(med(gv('e90')),2)} | 0.10 |")
    md.append("\nRatios of gap onset to rebound timing, per run (median, n valid):\n")
    md += ["| N | metric | onset / e_min | onset / e50 | onset / e90 | share of runs with e50 before onset |", "|---|---|---|---|---|---|"]
    J["timing_ratio"] = {}
    for N in NS:
        for m in TM:
            R = [r for r in sel("mlp", "0", N) if r["A"]["m"].get(m) and np.isfinite(r["A"]["gap_onset"])]
            f = lambda k: [r["A"]["gap_onset"] / r["A"]["m"][m][k] for r in R if r["A"]["m"][m][k] and np.isfinite(r["A"]["m"][m][k])]
            bef = [r["A"]["m"][m]["e50"] < r["A"]["gap_onset"] for r in R if np.isfinite(r["A"]["m"][m]["e50"])]
            J["timing_ratio"][f"{N}|{m}"] = dict(emin=f("e_min"), e50=f("e50"), e90=f("e90"), frac_e50_before=float(np.mean(bef)) if bef else np.nan)
            md.append(f"| {N} | {m} | {f2(med(f('e_min')))} (n={len(f('e_min'))}) | {f2(med(f('e50')))} (n={len(f('e50'))}) | {f2(med(f('e90')))} (n={len(f('e90'))}) | {('%.2f' % np.mean(bef)) if bef else 'n/a'} (n={len(bef)}) |")

    # ---- C across-run correlation
    md.append("\n## C. Across-run correlation of rebound with overfitting\n")
    md.append("rho is Spearman across runs. Within-group rho ranks inside each (N, wd) group of 6 runs, centres, and pools. logN rho is Spearman of each quantity with log N (full = 60000).\n")
    groups = {"MLP wd0": sel("mlp", "0"), "MLP wd1e-4": sel("mlp", "1e-4"), "MLP pooled": sel("mlp", "0") + sel("mlp", "1e-4"), "linear wd0 (control)": sel("linear", "0")}
    xs = [(f"final rebound {m}", lambda r, m=m: r["A"]["m"][m]["final_rebound"] if r["A"]["m"].get(m) else np.nan) for m in TM] + \
         [(f"relative rebound {m}", lambda r, m=m: r["A"]["m"][m]["rel_rebound"] if r["A"]["m"].get(m) else np.nan) for m in TM]
    xm = list(TM) + list(TM)  # metric behind each x entry
    prr = lambda r, m: r["A"]["m"][m]["pr_rise"] if r["A"]["m"].get(m) else np.nan
    ys = [("final gap", lambda r: r["A"]["final_gap"]), ("test excess", lambda r: r["A"]["test_excess"]),
          ("pr_dim rise", None)]
    md += ["| group | rebound | vs | rho | p | n | within (N,wd) rho | n | within (N,wd,dataset) rho | n |", "|---|---|---|---|---|---|---|---|---|---|"]
    J["corr"] = {}
    for gn, R in groups.items():
        grp = [(r["N"], r["wd"]) for r in R]; grp2 = [(r["N"], r["wd"], r["ds"]) for r in R]
        for (xn, xf), xmm in zip(xs, xm):
            x = np.array([xf(r) for r in R], float)
            for yn, yf in ys:
                y = np.array([(yf(r) if yf else prr(r, xmm)) for r in R], float); rho, p, n = sp(x, y); w, nw = within(x, y, grp); w2, nw2 = within(x, y, grp2)
                J["corr"][f"{gn}|{xn}|{yn}"] = dict(rho=rho, p=p, n=n, within=w, n_within=nw, within_ds=w2, n_within_ds=nw2)
                md.append(f"| {gn} | {xn} | {yn} | {f2(rho,2)} | {f2(p,2)} | {n} | {f2(w,2)} | {nw} | {f2(w2,2)} | {nw2} |")
        lN = np.array([logN(r) for r in R])
        for nm, f in [("final rebound h1_total", xs[1][1]), ("relative rebound h1_total", xs[4][1])] + [(a, b or (lambda r: prr(r, PRIM))) for a, b in ys]:
            v = np.array([f(r) for r in R], float); rho, p, n = sp(v, lN)
            J["corr"][f"{gn}|logN|{nm}"] = dict(rho=rho, p=p, n=n); md.append(f"| {gn} | {nm} | log N | {f2(rho,2)} | {f2(p,2)} | {n} | | |")
    # classifier across N
    md.append("\nClassifier (wd0, all N and both datasets and seeds available), rebound vs test-loss gap:\n")
    C = sel("clf"); Cn = sorted(set(r["N"] for r in C)); J["clf_acrossN"] = {"N": Cn, "n_runs": len(C)}
    md += ["| rebound | vs | rho | p | n |", "|---|---|---|---|---|"]
    for xn, xf in xs[:3] + xs[3:4]:
        x = np.array([xf(r) for r in C], float)
        for yn, yf in ys[:2]:
            y = np.array([yf(r) for r in C], float); rho, p, n = sp(x, y); J["clf_acrossN"][f"{xn}|{yn}"] = dict(rho=rho, p=p, n=n)
            md.append(f"| {xn} | loss {yn} | {f2(rho,2)} | {f2(p,2)} | {n} |")
    md.append(f"\nClassifier N present: {Cn}, runs: {len(C)}.\n")

    # ---- D time course
    def tc(r, m=PRIM):
        a = r["A"]["m"].get(m)
        if not a: return None
        st, n = r["step"], len(r["step"]); idx = np.arange(n); t = r["mask"] & (idx > a["i_min"]) & (idx < n - 1)
        v = r["top"][m]; rb = np.where(t, v - v[a["i_min"]], np.nan); g = r["geom"]
        cv = dict(gap=r["gapt"], fit=1 - r["rr"] / r["rr"][0])
        for q in ("pr_dim", "twonn_dim", "top2_var", "knn_overlap_pixel"):
            cv[q] = g.get(f"test/{q}", np.full(n, np.nan))
        return rb, {k: np.where(t, v_, np.nan) for k, v_ in cv.items()}

    md.append("## D. Time course within runs (Spearman over masked snapshots after e_min, median [IQR] across runs)\n")
    md.append("Rebound metric is h1_total unless stated. Partial columns residualize ranks.\n")
    J["tc"] = {}
    cols = list(COVS) + ["partial gap|pr_dim", "partial pr_dim|gap"]
    for m in TM:
        md += [f"\n### {m}\n", "| group | n runs | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
        for lab, R in [(f"MLP wd0 N={N}", sel("mlp", "0", N)) for N in NS] + [("MLP wd0 all", sel("mlp", "0"))] + \
                      [(f"MLP wd1e-4 N={N}", sel("mlp", "1e-4", N)) for N in NS] + [("MLP wd1e-4 all", sel("mlp", "1e-4")), ("linear wd0 all", sel("linear", "0"))]:
            res = {c: [] for c in cols}
            for r in R:
                o = tc(r, m)
                if o is None: continue
                rb, cv = o
                for c in COVS: res[c].append(sp(rb, cv[c])[0])
                res["partial gap|pr_dim"].append(partial_sp(rb, cv["gap"], cv["pr_dim"])); res["partial pr_dim|gap"].append(partial_sp(rb, cv["pr_dim"], cv["gap"]))
            J["tc"][f"{m}|{lab}"] = res
            md.append(f"| {lab} | {len(R)} | " + " | ".join(iqr(res[c]) for c in cols) + " |")

    # ---- E weight decay
    md.append("\n## E. Weight-decay control (MLP AE, means over 6 runs per N and wd)\n")
    md += ["| N | metric | rebound wd0 | rebound wd1e-4 | ratio | gap wd0 | gap wd1e-4 | ratio | excess wd0 | excess wd1e-4 | ratio |", "|" + "---|" * 11]
    J["wd"] = {}
    rat = lambda a, b: b / a if a and abs(a) > 1e-9 else np.nan
    for N in NS:
        for m in TM:
            A0, A1 = sel("mlp", "0", N), sel("mlp", "1e-4", N)
            fr = lambda R: np.nanmean([r["A"]["m"][m]["final_rebound"] for r in R if r["A"]["m"].get(m)])
            gp = lambda R: np.nanmean([r["A"]["final_gap"] for r in R]); ex = lambda R: np.nanmean([r["A"]["test_excess"] for r in R])
            d = dict(rb0=fr(A0), rb1=fr(A1), gap0=gp(A0), gap1=gp(A1), ex0=ex(A0), ex1=ex(A1)); d.update(rb_ratio=rat(d["rb0"], d["rb1"]), gap_ratio=rat(d["gap0"], d["gap1"]), ex_ratio=rat(d["ex0"], d["ex1"]))
            J["wd"][f"{N}|{m}"] = d
            md.append(f"| {N} | {m} | {f2(d['rb0'])} | {f2(d['rb1'])} | {f2(d['rb_ratio'],2)} | {f2(d['gap0'])} | {f2(d['gap1'])} | {f2(d['gap_ratio'],2)} | {f2(d['ex0'])} | {f2(d['ex1'])} | {f2(d['ex_ratio'],2)} |")

    # ---- F train vs test
    md.append("\n## F. Train probe minus test probe (masked snapshots, step >= 2)\n")
    md.append("Delta = train metric minus test metric. Spearman of Delta(t) with gap(t) is per run then median [IQR]. Relative Delta is Delta divided by the test value. Pre/post split at gap onset where defined. N=250 excluded.\n")
    cfgs = [("MLP wd0 N=1000", sel("mlp", "0", "1000")), ("MLP wd0 N=full", sel("mlp", "0", "full")), ("MLP wd1e-4 N=1000", sel("mlp", "1e-4", "1000")),
            ("linear wd0 N=1000", sel("linear", "0", "1000")), ("linear wd0 N=full", sel("linear", "0", "full"))]
    quant = [(m, "top") for m in TM] + [(q, "geom") for q in ("pr_dim", "knn_dist_cv", "rho_err_knn")]

    def series(r, q, kind):
        if kind == "top": return (r["trtop"][q] - r["top"][q]) if r["trtop"] is not None else None
        g = r["geom"]
        if f"train/{q}" not in g: return None
        return g[f"train/{q}"] - g[f"test/{q}"]
    md += ["| config | quantity | n runs | rho(Delta, gap) all | rho after e_min | median rel Delta pre-onset | post-onset | at e_min | at end |", "|" + "---|" * 9]
    J["trte"] = {}
    for lab, R in cfgs:
        R = [r for r in R if r["trtop"] is not None]
        for q, kind in quant + [("test_to_train_nn", "cross")]:
            rs, pre, post, amin, aend, rs2 = [], [], [], [], [], []
            for r in R:
                n = len(r["step"]); idx = np.arange(n); mk = r["mask"] & (idx < n - 1) & (r["step"] >= STEP_MIN); st = r["step"]
                if kind == "cross":
                    d = r["geom"].get("cross/test_to_train_nn"); base = None
                    if d is None: continue
                else:
                    d = series(r, q, kind)
                    if d is None: continue
                    base = r["top"][q] if kind == "top" else r["geom"][f"test/{q}"]
                dd = np.where(mk, d, np.nan); rs.append(sp(dd, r["gapt"])[0])
                a = r["A"]["m"].get(PRIM); i0 = a["i_min"] if a else None
                if i0 is not None: rs2.append(sp(np.where(mk & (idx > i0), d, np.nan), r["gapt"])[0])
                rel = d if base is None else d / np.where(np.abs(base) > 1e-12, base, np.nan)
                on = r["A"]["gap_onset_step"]
                if np.isfinite(on): pre.append(med(rel[mk & (st < on)])); post.append(med(rel[mk & (st >= on)]))
                if i0 is not None: amin.append(rel[i0])
                E = r["mask"] & (idx < n - 1) & (st > STEP_END); aend.append(med(rel[E]))
            J["trte"][f"{lab}|{q}"] = dict(rho_all=rs, rho_after_emin=rs2, pre=pre, post=post, at_emin=amin, at_end=aend)
            md.append(f"| {lab} | {q} | {len(R)} | {iqr(rs)} | {iqr(rs2)} | {f2(med(pre),2)} (n={len(pre)}) | {f2(med(post),2)} (n={len(post)}) | {f2(med(amin),2)} | {f2(med(aend),2)} |")

    # ---- G classifier
    md.append("\n## G. Classifier at N=1000 (wd0, both datasets, 3 seeds)\n")
    Cl = sel("clf", "0", "1000"); J["clf"] = {}
    md += ["| metric | e_min | e50 | e90 | final rebound | relative rebound | rho(test topo, loss ratio) after epoch 1 | rho(train-test Delta, loss gap) |", "|" + "---|" * 8]
    for m in TM:
        rs, ds_ = [], []
        for r in Cl:
            n = len(r["step"]); idx = np.arange(n); mk = r["mask"] & (idx < n - 1) & (r["ep"] > 1)
            rs.append(sp(np.where(mk, r["top"][m], np.nan), r["gapt"])[0])
            if r["trtop"] is not None: ds_.append(sp(np.where(mk, r["trtop"][m] - r["top"][m], np.nan), r["gapt"])[0])
        g = lambda k: [r["A"]["m"][m][k] for r in Cl if r["A"]["m"].get(m)]
        J["clf"][m] = dict(e_min=g("e_min"), e50=g("e50"), e90=g("e90"), final_rebound=g("final_rebound"), rel=g("rel_rebound"), rho_loss=rs, rho_delta=ds_)
        md.append(f"| {m} | {rng(g('e_min'))} | {rng(g('e50'))} | {rng(g('e90'))} | {rng(g('final_rebound'))} | {rng(g('rel_rebound'))} | {iqr(rs)} | {iqr(ds_)} |")
    ex = [(r["ds"], r["s"], r["A"]["gap_onset"], r["A"]["test_min_ep"], r["A"]["test_excess"]) for r in Cl]
    J["clf"]["loss"] = ex; md.append("\nClassifier test-loss onset (ratio > 1.10), test-loss minimum epoch and excess per run: " + ", ".join(f"{a[:2]}s{b}: {f2(c,3)}/{f2(d,3)}/{f2(e,2)}" for a, b, c, d, e in ex) + "\n")

    # ---- figure
    make_fig(runs, sel, J)
    # ---- outputs
    vf = f"{TOPO}/REBOUND_VERDICT.md"
    out = "# Rebound vs overfitting: tables\n\nGenerated by scripts/ae/ae_rebound_corr.py. Last snapshot dropped. End = median over masked snapshots with step > 4000. e_min searched over step >= 2.\n\n" + "\n".join(md)
    if os.path.isfile(vf): out += "\n\n" + open(vf).read()
    open(f"{TOPO}/REBOUND_TAKEAWAYS.md", "w").write(out)
    def cl(o):
        if isinstance(o, dict): return {str(k): cl(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [cl(v) for v in o]
        if isinstance(o, (np.floating, float)): return None if not np.isfinite(o) else float(o)
        if isinstance(o, np.integer): return int(o)
        return o
    json.dump(cl(J), open(f"{TOPO}/rebound_numbers.json", "w"), indent=1)


def make_fig(runs, sel, J):
    plt.rcParams.update({"font.size": 6.5, "axes.labelsize": 6.5, "xtick.labelsize": 6, "ytick.labelsize": 6, "axes.titlesize": 7, "legend.fontsize": 5.5, "lines.linewidth": 0.9})
    BANDS = (("early fit", "#fff2a8", (0.125, 1)), ("fitting", "#b9e6c0", (1, 33.458)), ("overfitting", "#d9ccf0", (33.458, 2400)))
    NC = {"250": "#fdae61", "1000": "#d7191c", "4000": "#2b83ba", "full": "#1a1a1a"}; NL = {"250": "N=250", "1000": "N=1000", "4000": "N=4000", "full": "N=60000"}
    fig, axs = plt.subplots(2, 3, figsize=(7.2, 4.6)); ax = axs.ravel()
    def bands(a):
        for _, c, (lo, hi) in BANDS: a.axvspan(lo, hi, color=c, alpha=0.3, lw=0, zorder=0)
    for N in NS:
        for wd, ls in (("0", "-"), ("1e-4", ":")):
            R = sel("mlp", wd, N)
            if not R: continue
            ep = R[0]["step"].astype(float); n = len(ep); B = []; G = []
            for r in R:
                a = r["A"]["m"].get(PRIM); idx = np.arange(n); v = r["top"][PRIM]
                B.append(np.where(r["mask"] & (idx >= a["i_min"]) & (idx < n - 1), v - v[a["i_min"]], np.nan)); G.append(np.where(idx < n - 1, r["gapt"], np.nan))
            b = np.nanmean(B, 0); g = np.nanmean(G, 0); ok = np.isfinite(b) & (ep > 0); ok2 = np.isfinite(g) & (ep > 0)
            ax[0].plot(ep[ok], b[ok], color=NC[N], ls=ls, label=f"{NL[N]} wd {'0' if wd=='0' else '1e-4'}" if True else None)
            ax[1].plot(ep[ok2], g[ok2], color=NC[N], ls=ls)
    for a in (ax[0], ax[1]): a.set_xscale("log"); a.set_xlabel("optimizer step"); a.set_xlim(1, 20000)
    for xv, tx in ((8, "minimum"), (70, "half recovered")):
        ax[0].axvline(xv, color="0.3", lw=0.6, ls="-.", zorder=0); ax[0].text(xv * 1.08, 19.5, tx, fontsize=5.5, color="0.2", rotation=90, va="top", ha="left")
    ax[0].set_ylabel("H1 lifetime sum above its minimum"); ax[0].set_title("Rebound of latent topology", fontsize=7)
    ax[1].set_yscale("symlog", linthresh=0.1); ax[1].axhline(0.1, color="0.4", lw=0.5, ls="--"); ax[1].set_ylabel("test/train recon error minus one"); ax[1].set_ylim(-0.05, 1000); ax[1].set_yticks([0, 0.1, 1, 10, 100, 1000]); ax[1].set_yticklabels(["0", "0.1", "1", "10", "100", "1000"]); ax[1].set_title("Generalisation gap, an epoch clock", fontsize=7)
    from matplotlib.lines import Line2D
    hh = [Line2D([], [], color=NC[N], label=NL[N]) for N in NS] + [Line2D([], [], color="0.3", ls="-", label="no weight decay"), Line2D([], [], color="0.3", ls=":", label="weight decay 1e-4")]
    ax[0].legend(handles=hh, ncol=1, frameon=False, loc="lower right", handlelength=1.0, fontsize=5, borderaxespad=0.2, labelspacing=0.25)
    ax[0].set_title("Rebound of latent topology, all N on a step clock", fontsize=6.5)
    # (3) scatter
    mk = {"0": "o", "1e-4": "s"}
    for r in sel("mlp", "0") + sel("mlp", "1e-4"):
        a = r["A"]["m"].get(PRIM)
        if not a: continue
        fc = NC[r["N"]] if r["ds"] == "mnist" else "none"
        ax[2].scatter(max(r["A"]["final_gap"], -0.05), a["final_rebound"], s=14, marker=mk[r["wd"]], facecolor=fc, edgecolor=NC[r["N"]], lw=0.7)
    ax[2].set_xscale("symlog", linthresh=0.1); ax[2].set_xlabel("final test over train error minus one"); ax[2].set_ylabel("final H1 rebound")
    c = J["corr"].get("MLP pooled|final rebound h1_total|final gap"); w = c["within"] if c else np.nan
    ax[2].set_title(f"Final rebound vs final gap\nSpearman {c['rho']:.2f} pooled and {w:.2f} within N (n=48)", fontsize=6.5)
    from matplotlib.lines import Line2D
    ax[2].legend(handles=[Line2D([], [], marker="o", color="0.3", ls="", ms=3.5, label="no weight decay"), Line2D([], [], marker="s", color="0.3", ls="", ms=3.5, label="weight decay 1e-4"),
                          Line2D([], [], marker="o", mfc="none", color="0.3", ls="", ms=3.5, label="hollow is Fashion-MNIST")], frameon=False, loc="upper left")
    # (4) box of time-course
    cols = list(COVS) + ["partial gap|pr_dim", "partial pr_dim|gap"]; lab = ["gap", "train fit", "pr_dim", "twonn", "top2 var", "kNN pixel overlap", "gap given pr_dim", "pr_dim given gap"]
    data = [np.array([v for v in J["tc"][f"{PRIM}|MLP wd0 all"][c] if np.isfinite(v)]) for c in cols]
    bp = ax[3].boxplot(data, vert=False, widths=0.6, showfliers=False, patch_artist=True, medianprops=dict(color="k"))
    for p_, cc in zip(bp["boxes"], [AE] * 6 + ["#9ecae1"] * 2): p_.set_facecolor(cc); p_.set_alpha(0.7)
    ax[3].set_yticks(range(1, 9)); ax[3].set_yticklabels(lab); ax[3].invert_yaxis(); ax[3].axvline(0, color="0.5", lw=0.5)
    ax[3].set_xlabel("Spearman with rebound over time within a run"); ax[3].set_title("What the rebound tracks (MLP AE, 24 runs)", fontsize=7)
    # (5) Delta
    ax5 = ax[4]; ax5b = ax5.twinx(); bands(ax5)
    for N, col in (("1000", AE), ("full", "#08306b")):
        R = [r for r in sel("mlp", "0", N) if r["trtop"] is not None]
        if not R: continue
        ep = R[0]["ep"]; n = len(ep); D = np.array([np.where(r["mask"] & (np.arange(n) < n - 1), r["trtop"][PRIM] - r["top"][PRIM], np.nan) for r in R]); d = np.nanmean(D, 0)
        G = np.nanmean([np.where(np.arange(n) < n - 1, r["gapt"], np.nan) for r in R], 0); ok = np.isfinite(d) & (ep > 0)
        ax5.plot(ep[ok], d[ok], color=col, label=f"train minus test H1 sum, {NL[N]}"); ax5b.plot(ep[ep > 0], G[ep > 0], color=col, ls="--", lw=0.7, label=f"gap, {NL[N]}")
    ax5.set_xscale("log"); ax5b.set_yscale("symlog", linthresh=0.1); ax5.set_xlabel("epoch of each run"); ax5.set_ylabel("train minus test H1 sum"); ax5b.set_ylabel("gap (dashed)"); ax5.set_ylim(-2, 1.6); ax5b.set_ylim(-0.3, 1000); ax5b.set_yticks([0, 0.1, 1, 10, 100, 1000]); ax5b.set_yticklabels(["0", "0.1", "1", "10", "100", "1000"])
    ax5.set_title("Train probe minus test probe", fontsize=7); ax5.axhline(0, color="0.5", lw=0.4)
    h1_, l1_ = ax5.get_legend_handles_labels(); h2_, l2_ = ax5b.get_legend_handles_labels(); ax5.legend(h1_ + h2_, [x.replace("train minus test H1 sum, ", "H1 difference ") for x in l1_] + [x.replace("gap, ", "gap ") for x in l2_], frameon=False, loc="lower left", fontsize=5)
    # (6) classifier
    a6 = ax[5]; a6b = a6.twinx(); bands(a6)
    for ds, ls in (("mnist", "-"), ("fashion_mnist", "--")):
        R = [r for r in sel("clf", "0", "1000") if r["ds"] == ds]
        if not R: continue
        ep = R[0]["ep"]; n = len(ep); T = np.nanmean([np.where(r["mask"] & (np.arange(n) < n - 1), r["top"][PRIM], np.nan) for r in R], 0)
        L_ = np.nanmean([np.where(np.arange(n) < n - 1, r["rt"] / r["rr"], np.nan) for r in R], 0); ok = np.isfinite(T) & (ep > 0)
        a6.plot(ep[ok], T[ok], color=CL, ls=ls, label=("MNIST" if ds == "mnist" else "FM") + " H1 sum"); a6b.plot(ep[ep > 0], L_[ep > 0], color="0.3", ls=ls, lw=0.7, label=("MNIST" if ds == "mnist" else "FM") + " loss ratio")
    a6.set_xscale("log"); a6b.set_yscale("log"); a6.set_xlabel("epoch"); a6.set_ylabel("test H1 lifetime sum"); a6b.set_ylabel("test over train loss")
    a6.set_title("Classifier N=1000", fontsize=7); h1_, l1_ = a6.get_legend_handles_labels(); h2_, l2_ = a6b.get_legend_handles_labels(); a6.set_ylim(5, 62); a6b.set_ylim(0.5, 1e9); a6.legend(h1_ + h2_, l1_ + l2_, frameon=False, loc="upper left", fontsize=5, ncol=2, columnspacing=0.8)
    for a in ax: a.tick_params(labelsize=6)
    fig.tight_layout(); os.makedirs(FIG, exist_ok=True)
    for e in ("pdf", "png"): fig.savefig(f"{FIG}/rebound_vs_gap.{e}", dpi=200)


if __name__ == "__main__":
    main()
