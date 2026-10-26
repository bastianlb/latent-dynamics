#!/usr/bin/env python
"""A6 timing sensitivity (thresholds 25/50/75%, slingshot exclusion, undefined-timing diagnostics) and A7 redundancy at multiple steps. MNIST set (ii)."""
import step4_lib as L
import numpy as np, csv
from scipy import stats
D, C = L.D, L.C
rng = np.random.default_rng(0)
CORE = D.CORE + ["h1_total", "norm"]
TOPO = ("h1_total", "h1_top5", "mag_t1", "pmag_fixed", "h0_total", "h1_top10", "h1_entropy")
QS = [0.25, 0.5, 0.75]
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)


def half_time_q(step, x, x0, xT, q, min_frac=0.5):
    dx = xT - x0
    ok = np.isfinite(x); st, xx = step[ok], x[ok]
    if len(st) < 3 or not np.isfinite(dx): return np.nan, "<3 finite points / undefined end"
    if abs(dx) < 1e-9: return np.nan, "flat (no net change)"
    if abs(dx) < min_frac * np.nanmax(np.abs(xx - x0)): return np.nan, "transient (net change < 0.5 x max excursion)"
    p = (xx - x0) / dx
    j = np.where(p >= q)[0]
    if len(j) == 0: return np.nan, "level never reached"
    j = j[0]; ls = np.log10(np.maximum(st, 0.5))
    if j == 0: return ls[0], "ok"
    return ls[j - 1] + (q - p[j - 1]) / (p[j] - p[j - 1]) * (ls[j] - ls[j - 1]), "ok"


def boot_med(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3: return np.nan, np.nan, np.nan, len(v)
    b = [np.median(rng.choice(v, len(v))) for _ in range(n)]
    return float(np.median(v)), *np.percentile(b, [2.5, 97.5]), len(v)


def config_timing(rs, q):
    """returns dict with t50 / reason for gap, test_acc, d_m, te_m (rs = runs of one config, common step grid)."""
    steps = rs[0]["step"]
    T = int(np.round(np.mean([r["pre"] for r in rs])))
    res = {}
    ys = np.nanmean([r["gap"] for r in rs], 0)
    res["gap"] = half_time_q(steps, ys, ys[0], ys[T], q)
    ys = np.nanmean([r["d"]["run/test_acc"] for r in rs], 0)
    res["test_acc"] = half_time_q(steps, ys, ys[0], ys[T], q)
    for m in CORE:
        tm = np.where(rs[0]["topo"])[0] if m in TOPO else np.arange(len(steps))
        Tt = tm[tm <= T][-1]
        y2 = np.nanmean([D.delta(r, m) for r in rs], 0)
        res["d_" + m] = half_time_q(steps[tm], y2[tm], y2[0], y2[Tt], q)
        if m != "norm":
            y4 = np.nanmean([D.series(r, "test", m) for r in rs], 0)
            res["te_" + m] = half_time_q(steps[tm], y4[tm], y4[0], y4[Tt], q)
    return res


def topo_val(r, step, m, probe="test"):
    topo = m in TOPO
    i = D.idx_le_mask(r["step"], r["topo"], step) if topo else D.idx_le(r["step"], step)
    return D.series(r, probe, m)[i]


def main():
    runs = D.load_runs("mnist")
    cfgs = D.by_config(runs)
    P(f"MNIST set (ii): {len(runs)} runs, {len(cfgs)} configs; covered by floor-relative slingshot files: {sum(r['covered'] for r in runs)}; "
      f"with a finite onset: {sum(np.isfinite(r['onset']) for r in runs)}")
    unc = sorted({r["info"]["block"] for r in runs if not r["covered"]}); P("blocks with uncovered runs (no onset info):", unc)
    info = {cid: rs[0]["info"] for cid, rs in cfgs.items()}
    # ------------------------------------------------ A6
    variants = {"all_runs": lambda r: True,
                "covered_runs_only": lambda r: r["covered"],
                "excl_slingshot": lambda r: r["covered"] and not np.isfinite(r["onset"])}
    long = []; results = {}
    for vname, keep in variants.items():
        for q in QS:
            rows = {}
            for cid, rs in sorted(cfgs.items()):
                rk = [r for r in rs if keep(r)]
                if not rk:
                    rows[cid] = None; continue
                rows[cid] = config_timing(rk, q) | {"_nruns": len(rk)}
            results[(vname, q)] = rows
            for cid, rr in rows.items():
                i = info[cid]
                for key in (["gap", "test_acc"] + ["d_" + m for m in CORE] + ["te_" + m for m in CORE if m != "norm"]):
                    if rr is None: v, why, nr = np.nan, "no runs left after exclusion", 0
                    else: (v, why), nr = rr[key], rr["_nruns"]
                    long.append(dict(variant=vname, q=q, config_id=cid, key=key, log10_t50=v, reason=why, n_runs_used=nr, n=i["n"], wd=i["wd"],
                                     label_noise=i["label_noise"], optimizer=i["optimizer"], block=i["block"]))
    L.write_csv(f"{L.OUT2}/step4_timing_long.csv", long)
    P("\n== A6.1 divergence timing: median over configs of log10 t50(Delta_m) - log10 t50(gap); negative = metric opens earlier. [95% bootstrap CI], n valid configs / frac earlier")
    summ = []
    for vname in variants:
        for q in QS:
            rows = results[(vname, q)]
            P(f"-- variant {vname}, threshold q={q}")
            for m in CORE:
                dl = np.array([(rr["d_" + m][0] - rr["gap"][0]) if rr is not None else np.nan for rr in rows.values()])
                md, lo, hi, n = boot_med(dl)
                fr = float(np.mean(dl[np.isfinite(dl)] < 0)) if n else np.nan
                P(f"   {D.NICE[m]:16s} {md:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}/{len(rows)} earlier-frac {fr:.2f}")
                summ.append(dict(variant=vname, q=q, ref="gap", metric=m, median=md, lo=lo, hi=hi, n_valid=n, n_configs=len(rows), frac_earlier=fr))
            for m in [x for x in CORE if x != "norm"]:
                dl = np.array([(rr["te_" + m][0] - rr["test_acc"][0]) if rr is not None else np.nan for rr in rows.values()])
                md, lo, hi, n = boot_med(dl)
                summ.append(dict(variant=vname, q=q, ref="test_acc", metric=m, median=md, lo=lo, hi=hi, n_valid=n, n_configs=len(rows),
                                 frac_earlier=float(np.mean(dl[np.isfinite(dl)] < 0)) if n else np.nan))
    L.write_csv(f"{L.OUT2}/step4_timing_summary.csv", summ)
    P("\n== A6.2 geometry-vs-test-acc timing (test-probe metric t50 - test acc t50), median [CI] n")
    for vname in variants:
        for q in QS:
            sub = [s for s in summ if s["variant"] == vname and s["q"] == q and s["ref"] == "test_acc" and s["metric"] in D.CORE]
            P(f"   {vname:18s} q={q}: " + "; ".join(f"{D.NICE[s['metric']]} {s['median']:+.2f}[{s['lo']:+.2f},{s['hi']:+.2f}]n={s['n_valid']}" for s in sub))
    # undefined diagnostics (all_runs, q=0.5 and others): reasons and design comparison
    P("\n== A6.3 undefined timing: Delta_m vs gap pair, all_runs. reasons count; defined-vs-undefined design comparison")
    for q in QS:
        rows = results[("all_runs", q)]
        for m in CORE:
            und, dfn = [], []
            reasons = {}
            for cid, rr in rows.items():
                ok = np.isfinite(rr["d_" + m][0]) and np.isfinite(rr["gap"][0])
                (dfn if ok else und).append(cid)
                if not ok:
                    w = rr["d_" + m][1] if not np.isfinite(rr["d_" + m][0]) else "gap: " + rr["gap"][1]
                    reasons[w] = reasons.get(w, 0) + 1
            if q == 0.5 or und:
                lg = lambda ids, k: np.array([np.log10(info[c]["n"]) if k == "n" else info[c][k] for c in ids], float)
                s = f"   q={q} {D.NICE[m]:16s} defined {len(dfn)}, undefined {len(und)}"
                if und and len(dfn) >= 3:
                    pn = stats.mannwhitneyu(lg(dfn, "n"), lg(und, "n"))[1]
                    pw = stats.mannwhitneyu(lg(dfn, "wd"), lg(und, "wd"))[1]
                    s += (f" | median log10 n def {np.median(lg(dfn,'n')):.2f} vs undef {np.median(lg(und,'n')):.2f} (MW p={pn:.3f}); "
                          f"median wd def {np.median(lg(dfn,'wd')):.0e} vs undef {np.median(lg(und,'wd')):.0e} (p={pw:.3f}); "
                          f"label-noise>0: def {np.mean(lg(dfn,'label_noise')>0):.2f} undef {np.mean(lg(und,'label_noise')>0):.2f}; SGD: def {np.mean(lg(dfn,'is_sgd')>0):.2f} undef {np.mean(lg(und,'is_sgd')>0):.2f}")
                    s += f" | undefined by n: " + ", ".join(f"{v}:{sum(info[c]['n_label']==v for c in und)}/{sum(info[c]['n_label']==v for c in rows)}" for v in ["250", "1000", "4000", "16000", "full"])
                    s += " | reasons: " + str(reasons)
                P(s)
    P("\n== A6.4 slingshot exclusion: configs lost entirely / runs lost")
    rows = results[("excl_slingshot", 0.5)]
    lost = [c for c, rr in rows.items() if rr is None]
    P(f"   configs with no usable run after exclusion (slingshot or uncovered): {len(lost)}; by block: " + str({b: sum(info[c]['block'] == b for c in lost) for b in sorted({info[c]['block'] for c in lost})}))
    P(f"   remaining configs {len(rows) - len(lost)}; runs used {sum(rr['_nruns'] for rr in rows.values() if rr)} / {len(runs)}")
    P("   (definedness of t50 differs from 'all_runs' variant: see A6.1 n_valid)")
    # ------------------------------------------------ A7
    P("\n== A7 redundancy: Spearman rho of Fisher (test probe) with other metrics across configs, config means over seeds; [95% config bootstrap CI]")
    steps = [300, 1000, 5000, 15000]
    mets = ["h1_total", "h1_top5", "mag_t1", "pmag_fixed"]
    red = []
    for setname, cfsel in (("set_ii (69 cfgs)", list(cfgs)), ("set_i (35 cfgs)", [c for c, rs in cfgs.items() if rs[0]["info"]["block"] in ("A", "B")])):
        P(f"-- {setname}")
        for st in steps:
            Mx = {m: np.array([np.mean([topo_val(r, st, m) for r in cfgs[c]]) for c in cfsel]) for m in ["fisher"] + mets}
            for m in mets:
                x, y = Mx["fisher"], Mx[m]
                rho = stats.spearmanr(x, y)[0]
                bs = []
                for _ in range(2000):
                    i = rng.integers(0, len(x), len(x)); bs.append(stats.spearmanr(x[i], y[i])[0])
                lo, hi = np.nanpercentile(bs, [2.5, 97.5])
                red.append(dict(set=setname, step=st, metric=m, rho=rho, lo=lo, hi=hi, n_configs=len(cfsel)))
            P(f"   step {st:5d}: " + "; ".join(f"{D.NICE[m]} {red[-1 - (len(mets) - 1 - k)]['rho']:+.2f}[{red[-1 - (len(mets) - 1 - k)]['lo']:+.2f},{red[-1 - (len(mets) - 1 - k)]['hi']:+.2f}]" for k, m in enumerate(mets)))
    L.write_csv(f"{L.OUT2}/step4_redundancy.csv", red)
    open(f"{L.OUT2}/step4_timing_summary.txt", "w").write("\n".join(out))


if __name__ == "__main__":
    main()
