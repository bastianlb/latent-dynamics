#!/usr/bin/env python
"""Step 2: descriptive results (train-vs-test divergence, phase grids, convergence timing, redundancy) + CIFAR descriptives."""
import common2 as C
import data2 as D
from data2 import NICE
import csv, os, sys
import numpy as np
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.join(C.REPO, "reanalysis"))
import stats_util as SU

RNG = np.random.default_rng(0)
NS = ["250", "1000", "4000", "16000", "full"]; WDS = [0.0, 1e-4, 1e-3, 1e-2, 3e-2]; LNS = [0.0, 0.2, 0.4]
FIXED = [1000, 5000, 15000, "final", "pre"]
DMET = D.CORE + ["h1_total", "norm", "eff_rank_ent"]
out_lines = []


def P(*a):
    s = " ".join(str(x) for x in a); print(s); out_lines.append(s)


def save(fig, name):
    fig.savefig(f"{C.FIG}/{name}.png", dpi=150, bbox_inches="tight"); plt.close(fig)


def cfg_of(rs):
    i = rs[0]["info"]; return i["n_label"], i["wd"], i["label_noise"]


def valid(v):  # finite mask
    return np.isfinite(v)


def at_step(r, target, m, probe, topo=False):
    st = r["step"]
    if target == "final":
        i = len(st) - 1
    elif target == "pre":
        i = r["pre"]
    else:
        i = D.idx_le_mask(st, r["topo"], target) if topo else D.idx_le(st, target)
    return i


def value(r, target, m, probe):
    topo = m in ("h0_total", "h1_total", "h1_top5", "h1_top10", "h1_entropy", "mag_t1", "pmag_fixed")
    i = at_step(r, target, m, probe, topo)
    if topo and not r["topo"][i]:  # final/pre not in topo subset -> back off to nearest topo snapshot
        i = int(np.where(r["topo"][:i + 1])[0][-1])
    if m.startswith("d_"):
        return D.delta(r, m[2:])[i]
    return D.series(r, probe, m)[i]


def half_time(step, x, x0, xT, min_frac=0.5):
    """first crossing of 50% progress, linear in log10(step). nan if invalid."""
    dx = xT - x0
    ok = np.isfinite(x)
    st, xx = step[ok], x[ok]
    if len(st) < 3 or not np.isfinite(dx) or abs(dx) < 1e-9:
        return np.nan
    if abs(dx) < min_frac * np.nanmax(np.abs(xx - x0)):
        return np.nan
    p = (xx - x0) / dx
    j = np.where(p >= 0.5)[0]
    if len(j) == 0:
        return np.nan
    j = j[0]
    ls = np.log10(np.maximum(st, 0.5))
    if j == 0:
        return ls[0]
    return ls[j - 1] + (0.5 - p[j - 1]) / (p[j] - p[j - 1]) * (ls[j] - ls[j - 1])


def boot_med(v, n=4000):
    v = np.asarray(v, float); v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan, np.nan, len(v)
    b = [np.median(RNG.choice(v, len(v))) for _ in range(n)]
    return float(np.median(v)), *np.percentile(b, [2.5, 97.5]), len(v)


# ------------------------------------------------------------------ MNIST
def mnist():
    runs = D.load_runs("mnist")
    cfgs = D.by_config(runs)
    P(f"MNIST: {len(runs)} runs, {len(cfgs)} configs")
    # --- slingshot / transition bookkeeping
    with open(f"{C.OUT}/config_summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config_id", "n_seeds", "mean_onset", "n_sling", "peak_test_acc", "peak_step", "cross_step_mean", "mid_transition_15k"])
        mid = []
        for cid, rs in sorted(cfgs.items()):
            cs = np.array([r["cross_step"] for r in rs]); on = np.array([r["onset"] for r in rs])
            ln = rs[0]["info"]["label_noise"]
            m = bool(ln > 0 and np.isfinite(np.nanmean(cs)) and 10500 <= np.nanmean(cs) <= 19500) if np.any(np.isfinite(cs)) else False
            if m: mid.append(cid)
            w.writerow([cid, len(rs), np.nanmean(on) if np.any(np.isfinite(on)) else "", int(np.isfinite(on).sum()),
                        np.mean([r["peak_acc"] for r in rs]), np.mean([r["peak_step"] for r in rs]),
                        np.nanmean(cs) if np.any(np.isfinite(cs)) else "", m])
    P("Configs mid-transition at step 15000 (mean memorisation-crossover step in 10.5k-19.5k):", mid)
    n_sl = sum(np.isfinite(r["onset"]) for r in runs); P(f"runs with slingshot onset: {n_sl}")

    # --- (a) delta curves (original run set only: needs a common step grid)
    if C.RUNSET == 'orig':
        steps = runs[0]['step']; assert all(np.array_equal(r['step'], steps) for r in runs); xs = np.maximum(steps, 1)
        for m in ["gap"] + DMET:
            fig, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharex=True, sharey=True)
            for k, n in enumerate(NS):
                for wd in WDS:
                    rs = cfgs.get(f"mnist|n{'full' if n=='full' else n}|wd{wd:g}|ln0")
                    if rs:
                        y = D.series(rs[0], "run", "gap")*0
                        y = np.nanmean([ (D.series(r, "run", "gap") if m == "gap" else D.delta(r, m)) for r in rs], 0)
                        mk = np.isfinite(y)
                        axs[0, k].plot(xs[mk], y[mk], label=f"wd={wd:g}", lw=1.4)
                for ln in LNS:
                    rs = cfgs.get(f"mnist|n{'full' if n=='full' else n}|wd0|ln{ln:g}")
                    if rs:
                        y = np.nanmean([(D.series(r, "run", "gap") if m == "gap" else D.delta(r, m)) for r in rs], 0)
                        mk = np.isfinite(y)
                        axs[1, k].plot(xs[mk], y[mk], label=f"label noise={ln:g}", lw=1.4)
                axs[0, k].set_title(f"n={n}"); axs[0, k].set_xscale("log"); axs[1, k].set_xlabel("step")
                axs[0, k].axhline(0, c="k", lw=.4); axs[1, k].axhline(0, c="k", lw=.4)
            axs[0, 0].legend(fontsize=7); axs[1, 0].legend(fontsize=7)
            lab = NICE.get(m, m)
            axs[0, 0].set_ylabel(("gap" if m == "gap" else f"delta {lab} (train-test)") + "\nwd sweep (ln=0)")
            axs[1, 0].set_ylabel("label-noise sweep (wd=0)")
            fig.suptitle(f"{'train-test accuracy gap' if m=='gap' else 'Delta '+lab+' = train probe - test probe'}  (config mean over 3 seeds)")
            save(fig, f"fig2a_delta_{m}")


    # --- timing: Delta_m opening vs accuracy gap opening
    rows = []
    for cid, rs in sorted(cfgs.items()):
        steps = rs[0]["step"]
        assert all(np.array_equal(r["step"], steps) for r in rs)
        def prog(fn):
            ys = np.nanmean([fn(r) for r in rs], 0)
            T = int(np.round(np.mean([r["pre"] for r in rs])))
            return ys, ys[0], ys[T]
        ys, x0, xT = prog(lambda r: r["gap"]); tg = half_time(steps, ys, x0, xT)
        row = dict(config_id=cid, t50_gap=tg)
        for m in D.CORE + ["h1_total", "norm"]:
            tm = np.where(rs[0]["topo"])[0] if m in ("h1_total", "h1_top5", "mag_t1", "pmag_fixed") else np.arange(len(steps))
            ys2 = np.nanmean([D.delta(r, m) for r in rs], 0)
            T = int(np.round(np.mean([r["pre"] for r in rs])))
            Tt = tm[tm <= T][-1]
            row["t50_d_" + m] = half_time(steps[tm], ys2[tm], ys2[0], ys2[Tt])
        # accuracy-of-representation vs geometry (test probe), for 2c
        for nm, fn in (("test_acc", lambda r: r["d"]["run/test_acc"]), ("train_acc", lambda r: r["d"]["run/train_acc"])):
            ys3, x0, xT = prog(fn); row["t50_" + nm] = half_time(steps, ys3, x0, xT)
        for m in D.CORE + ["h1_total"]:
            tm = np.where(rs[0]["topo"])[0] if m in ("h1_total", "h1_top5", "mag_t1", "pmag_fixed") else np.arange(len(steps))
            ys4 = np.nanmean([D.series(r, "test", m) for r in rs], 0)
            T = int(np.round(np.mean([r["pre"] for r in rs]))); Tt = tm[tm <= T][-1]
            row["t50_te_" + m] = half_time(steps[tm], ys4[tm], ys4[0], ys4[Tt])
        rows.append(row)
    keys = list(rows[0].keys())
    with open(f"{C.OUT}/step2_timing_per_config.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(rows)
    P("\n== (a) when does Delta_m open relative to the accuracy gap? log10 t50(Delta_m) - log10 t50(gap); negative = metric divergence opens earlier")
    summ = []
    for m in D.CORE + ["h1_total", "norm"]:
        dl = np.array([r["t50_d_" + m] - r["t50_gap"] for r in rows])
        md, lo, hi, n = boot_med(dl)
        frac = np.mean(dl[np.isfinite(dl)] < 0) if n else np.nan
        P(f"  {NICE[m]:16s} median {md:+.2f} [{lo:+.2f},{hi:+.2f}] over {n} valid configs; fraction earlier than gap {frac:.2f}")
        summ.append((m, md, lo, hi, n, frac))
    P("\n== (c) per-condition: which moves first? log10 t50(test-probe metric) - log10 t50(test acc); negative = geometry first")
    for ref in ("test_acc", "train_acc"):
        for m in D.CORE + ["h1_total"]:
            dl = np.array([r["t50_te_" + m] - r["t50_" + ref] for r in rows])
            md, lo, hi, n = boot_med(dl)
            frac = np.mean(dl[np.isfinite(dl)] < 0) if n else np.nan
            P(f"  vs {ref:9s} {NICE[m]:16s} median {md:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}; geometry-first frac {frac:.2f}")
    # subset: configs where test acc half-time < 2.5 (steps<316), dense early regime
    early = [r for r in rows if np.isfinite(r["t50_test_acc"]) and r["t50_test_acc"] < 2.5]
    P(f"  (dense-early subset: {len(early)} configs with t50(test acc) < 316 steps)")
    for m in D.CORE:
        dl = np.array([r["t50_te_" + m] - r["t50_test_acc"] for r in early]); md, lo, hi, n = boot_med(dl)
        P(f"    early {NICE[m]:16s} median {md:+.2f} [{lo:+.2f},{hi:+.2f}] n={n}")
    # figure: t50 comparison
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.6))
    for ax, m in zip(axs, ["fisher", "probe_acc", "h1_top5"]):
        x = [r["t50_test_acc"] for r in rows]; y = [r["t50_te_" + m] for r in rows]
        ax.scatter(x, y, s=22); lim = [0, 4.5]; ax.plot(lim, lim, "k--", lw=.6)
        ax.set_xlabel("log10 t50  test acc"); ax.set_ylabel(f"log10 t50  {NICE[m]} (test probe)"); ax.set_title(NICE[m])
    fig.suptitle("Half-progress time per config (below the diagonal = geometry moves first)")
    save(fig, "fig2c_t50_scatter")

    # --- (b) phase grids (original run set only)
    if C.RUNSET == 'orig':
        for grid, (rowvals, rowkey, rowname, cidf) in {
            "wd": (WDS, "wd", "weight decay", lambda n, v: f"mnist|n{'full' if n=='full' else n}|wd{v:g}|ln0"),
            "ln": (LNS, "ln", "label noise", lambda n, v: f"mnist|n{'full' if n=='full' else n}|wd0|ln{v:g}")}.items():
            mets = [("test_acc", "run", "test acc"), ("gap", "run", "gap"), ("d_fisher", None, "d Fisher"),
                    ("d_probe_acc", None, "d probe acc"), ("d_h1_top5", None, "d H1 top-5"), ("d_eff_rank_pr", None, "d eff rank")]
            fig, axs = plt.subplots(len(mets), len(FIXED), figsize=(3.3 * len(FIXED), 2.9 * len(mets)))
            tab = []
            for a_i, (m, probe, lab) in enumerate(mets):
                for t_i, tg in enumerate(FIXED):
                    Z = np.full((len(rowvals), len(NS)), np.nan)
                    for ri, v in enumerate(rowvals):
                        for ci, n in enumerate(NS):
                            rs = cfgs.get(cidf(n, v))
                            if rs:
                                if m == "gap":
                                    vals = [r["gap"][at_step(r, tg, m, "run")] for r in rs]
                                else:
                                    vals = [value(r, tg, m, probe or "run") for r in rs]
                                Z[ri, ci] = np.nanmean(vals)
                                tab.append([grid, m, tg, v, n, Z[ri, ci]])
                    ax = axs[a_i, t_i]; im = ax.imshow(Z, aspect="auto", cmap="viridis")
                    for (ri, ci), z in np.ndenumerate(Z):
                        if np.isfinite(z): ax.text(ci, ri, f"{z:.2g}", ha="center", va="center", fontsize=6, color="w")
                    ax.set_xticks(range(len(NS))); ax.set_xticklabels(NS, fontsize=7)
                    ax.set_yticks(range(len(rowvals))); ax.set_yticklabels([f"{v:g}" for v in rowvals], fontsize=7)
                    if a_i == 0: ax.set_title(f"step {tg}" if isinstance(tg, int) else tg)
                    if t_i == 0: ax.set_ylabel(f"{lab}\n{rowname}")
                    if a_i == len(mets) - 1: ax.set_xlabel("n")
            fig.suptitle(f"Phase grid (n x {rowname}) at fixed steps")
            save(fig, f"fig2b_grid_{grid}")
            with open(f"{C.OUT}/step2_grid_{grid}.csv", "w", newline="") as f:
                w = csv.writer(f); w.writerow(["grid", "metric", "step", rowname, "n", "value"]); w.writerows(tab)


    # --- (d) redundancy at step 5000 (config means, test probe)
    mets = ["test_acc", "gap"] + D.ALLM
    M = []
    for cid, rs in sorted(cfgs.items()):
        row = []
        for m in mets:
            if m == "test_acc": row.append(np.mean([r["d"]["run/test_acc"][D.idx_le(r["step"], 5000)] for r in rs]))
            elif m == "gap": row.append(np.mean([r["gap"][D.idx_le(r["step"], 5000)] for r in rs]))
            else: row.append(np.mean([value(r, 5000, m, "test") for r in rs]))
        M.append(row)
    M = np.array(M); P(f"\n== (d) redundancy: {M.shape[0]} configs, step 5000, test probe (Spearman)")
    rho = np.array([[stats.spearmanr(M[:, i], M[:, j])[0] for j in range(len(mets))] for i in range(len(mets))])
    fig, ax = plt.subplots(figsize=(9, 8)); im = ax.imshow(rho, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(mets))); ax.set_yticks(range(len(mets)))
    lab = [NICE.get(m, m) for m in mets]; ax.set_xticklabels(lab, rotation=75, ha="right", fontsize=8); ax.set_yticklabels(lab, fontsize=8)
    for (i, j), z in np.ndenumerate(rho): ax.text(j, i, f"{z:.1f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im); ax.set_title("Config-level Spearman, test probe, step 5000")
    save(fig, "fig2d_corr")
    fi = mets.index("fisher")
    with open(f"{C.OUT}/step2_redundancy.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "rho_with_fisher", "ci_lo", "ci_hi", "partial_rho_testacc|fisher", "partial_rho_gap|fisher",
                                       "rho_testacc", "rho_gap"])
        P("  metric: rho(Fisher) [95% CI] | partial rho with test acc | with gap given Fisher | raw rho test acc | raw rho gap")
        for j, m in enumerate(mets):
            if m == "fisher": continue
            lo, hi = SU.boot_ci_corr(M[:, fi], M[:, j], n=2000, method="spearman")
            pa = SU.partial_corr(M[:, j], M[:, 0], M[:, [fi]]) if m != "test_acc" else np.nan
            pg = SU.partial_corr(M[:, j], M[:, 1], M[:, [fi]]) if m != "gap" else np.nan
            w.writerow([m, rho[fi, j], lo, hi, pa, pg, rho[0, j], rho[1, j]])
            P(f"  {NICE.get(m,m):16s} {rho[fi,j]:+.2f} [{lo:+.2f},{hi:+.2f}] | {pa:+.2f} | {pg:+.2f} | {rho[0,j]:+.2f} | {rho[1,j]:+.2f}")


# ------------------------------------------------------------------ CIFAR
def cifar():
    runs = D.load_runs("cifar10")
    if not runs:
        P("CIFAR: no finished runs"); return
    cfgs = D.by_config(runs)
    P(f"\nCIFAR-10: {len(runs)} finished runs, {len(cfgs)} configs: " + ", ".join(f"{k.split('|',1)[1]}(x{len(v)})" for k, v in sorted(cfgs.items())))
    fr = [0.05, 0.25, 0.5, 1.0]
    mets = ["gap"] + D.CORE
    fig, axs = plt.subplots(len(mets), 1, figsize=(8, 2.8 * len(mets)), sharex=True)
    tab = []
    for cid, rs in sorted(cfgs.items()):
        for ai, m in enumerate(mets):
            ys = []
            for r in rs:
                y = r["gap"] if m == "gap" else D.delta(r, m)
                ys.append(y)
            st = rs[0]["step"]; xs = st / st[-1]
            # per-run interpolate on finite points only
            grid = np.geomspace(1e-3, 1, 60)
            cur = []
            for r, yy in zip(rs, ys):
                x = r["step"] / r["step"][-1]; k = np.isfinite(yy) & (x > 0)
                cur.append(np.interp(grid, x[k], yy[k]))
            ymean = np.mean(cur, 0)
            axs[ai].plot(grid, ymean, label=cid.split("|", 1)[1], lw=1.3)
            axs[ai].set_ylabel("gap" if m == "gap" else "d " + NICE[m]); axs[ai].set_xscale("log"); axs[ai].axhline(0, c="k", lw=.4)
            for f_ in fr:
                vv = []
                for r, yy in zip(rs, ys):
                    i = D.idx_le(r["step"], f_ * r["step"][-1])
                    ok = np.where(np.isfinite(yy[:i + 1]))[0]
                    vv.append(yy[ok[-1]])
                tab.append([cid, len(rs), m, f_, float(np.mean(vv))])
    axs[0].legend(fontsize=6, ncol=2); axs[-1].set_xlabel("fraction of training steps")
    fig.suptitle("CIFAR-10 ResNet-18: train-test divergence (finished configs, seed mean)")
    save(fig, "fig2_cifar_delta")
    with open(f"{C.OUT}/step2_cifar_fixed_fraction.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["config", "n_seeds", "metric(delta)", "frac_of_training", "value"]); w.writerows(tab)
    P("  CIFAR train-test deltas at fractions of training (seed means), final column=100%:")
    for cid in sorted(cfgs):
        for m in mets:
            vals = [t[4] for t in tab if t[0] == cid and t[2] == m]
            P(f"   {cid.split('|',1)[1]:22s} {m:12s} " + " ".join(f"{v:+.3g}" for v in vals))
    matched_gap(cfgs)
    P("  CIFAR test-probe vs train-probe geometry at the end (seed mean):")
    for cid, rs in sorted(cfgs.items()):
        P("   " + cid.split("|", 1)[1] + ": " + "; ".join(f"{m} te={np.mean([value(r,'final',m,'test') for r in rs]):.3g} tr={np.mean([value(r,'final',m,'train') for r in rs]):.3g}" for m in ["fisher", "probe_acc", "h1_top5"])
          + f"; gap={np.mean([r['gap'][-1] for r in rs]):.3f} test_acc={np.mean([r['d']['run/test_acc'][-1] for r in rs]):.3f}")


def matched_gap(cfgs_c):
    """Compare CIFAR train-test divergence with MNIST at matched gap (pre-slingshot / final snapshot)."""
    mn = D.by_config(D.load_runs("mnist"))
    mets = D.CORE
    def pt(rs, m, ds):
        out = []
        for r in rs:
            i = r["pre"] if ds == "mnist" else len(r["step"]) - 1
            if m in ("h1_top5", "mag_t1", "pmag_fixed") and not r["topo"][i]:
                i = int(np.where(r["topo"][:i + 1])[0][-1])
            out.append((r["gap"][i], D.delta(r, m)[i]))
        return np.mean(out, 0)
    M = {m: np.array([pt(rs, m, "mnist") for rs in mn.values()]) for m in mets}
    gm = M["fisher"][:, 0]
    fig, axs = plt.subplots(1, len(mets), figsize=(3.4 * len(mets), 3.6))
    P("\n== CIFAR vs MNIST train-test divergence at matched gap (final/pre-slingshot snapshot; MNIST = configs with |gap diff|<0.04)")
    rows = []
    for ai, m in enumerate(mets):
        axs[ai].scatter(M[m][:, 0], M[m][:, 1], c="grey", s=14, label="MNIST configs")
        rho = stats.spearmanr(M[m][:, 0], M[m][:, 1])[0]
        axs[ai].set_title(f"{NICE[m]}  (MNIST rho vs gap {rho:+.2f})", fontsize=8)
        axs[ai].set_xlabel("gap"); axs[ai].set_ylabel("d " + NICE[m])
        for cid, rs in sorted(cfgs_c.items()):
            g, dv = pt(rs, m, "cifar")
            axs[ai].scatter([g], [dv], c="r", s=30, marker="D", label="CIFAR" if cid == sorted(cfgs_c)[0] else None)
            sel = np.abs(M["fisher"][:, 0] - g) < 0.04
            lo, hi = (np.min(M[m][sel, 1]), np.max(M[m][sel, 1])) if sel.any() else (np.nan, np.nan)
            med = np.median(M[m][sel, 1]) if sel.any() else np.nan
            P(f"  {NICE[m]:14s} {cid.split('|',1)[1]:22s} gap={g:+.3f} d={dv:+.3g}  | MNIST matched (n={sel.sum()}): median {med:+.3g} range [{lo:+.3g},{hi:+.3g}]")
            rows.append([m, cid, g, dv, int(sel.sum()), med, lo, hi])
    axs[0].legend(fontsize=7)
    save(fig, "fig2_cifar_vs_mnist_matched_gap")
    with open(f"{C.OUT}/step2_cifar_vs_mnist_matched_gap.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["metric", "cifar_config", "gap", "delta", "n_mnist_matched", "mnist_median", "mnist_min", "mnist_max"]); w.writerows(rows)


if __name__ == "__main__":
    os.makedirs(C.FIG, exist_ok=True)
    mnist()
    if C.RUNSET == 'orig':
        cifar()
    open(f"{C.OUT}/step2_summary.txt", "w").write("\n".join(out_lines))
