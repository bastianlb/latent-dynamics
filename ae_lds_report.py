"""Aggregate AE (and classifier) linear-dynamical-system results -> tables (stdout / md fragments) + figures in ae_dynamics/analysis/. Usage: python ae_lds_report.py"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_k, "2")
import collections, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import ae_lds_load as L
OUTD = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/analysis"
DS = ["mnist", "fashion_mnist"]; NS = [250, 1000, 4000, 60000]; NL = {250: "250", 1000: "1k", 4000: "4k", 60000: "full"}
M = lambda v: float(np.nanmean(v)) if len(v) else np.nan
SE = lambda v: float(np.nanstd(v) / np.sqrt(max(len(v), 1)))


def sel(rows, **kw): return [r for r in rows if all(r.get(k) == v for k, v in kw.items())]


def tbl(rows, keys, bands=("b10_100", "b100_1000", "b1000_up")):
    return {(b, k): M([r[f"{k}_{b}"] for r in rows]) for b in bands for k in keys}


def rank(x):
    x = np.asarray(x, float); o = x.argsort().argsort().astype(float); return o


def spear(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float); k = np.isfinite(a) & np.isfinite(b)
    if k.sum() < 4: return np.nan
    ra, rb = rank(a[k]), rank(b[k]); ra -= ra.mean(); rb -= rb.mean(); return float((ra * rb).sum() / (np.linalg.norm(ra) * np.linalg.norm(rb) + 1e-300))


def within_corr(rows, xk, yk, gk=("dataset", "arch", "d", "N", "wd"), B=2000, seed=0):
    """Spearman of group-demeaned x and y (group = N/wd/... dummies); permutation p by permuting x within groups."""
    g = collections.defaultdict(list)
    for r in rows:
        if np.isfinite(r[xk]) and np.isfinite(r[yk]): g[tuple(r[k] for k in gk)].append((r[xk], r[yk]))
    g = {k: np.array(v) for k, v in g.items() if len(v) >= 2}
    if not g: return np.nan, np.nan, 0
    X = np.concatenate([v[:, 0] - v[:, 0].mean() for v in g.values()]); Y = np.concatenate([v[:, 1] - v[:, 1].mean() for v in g.values()]); obs = spear(X, Y)
    rng = np.random.RandomState(seed); cnt = 0
    for _ in range(B):
        Xp = np.concatenate([(v[:, 0][rng.permutation(len(v))]) - v[:, 0].mean() for v in g.values()]); cnt += abs(spear(Xp, Y)) >= abs(obs) - 1e-12
    return obs, (cnt + 1) / (B + 1), len(X)


def pair(rows_a, rows_b, key):
    """paired (a - b) by run id"""
    ib = {r["rid"]: r for r in rows_b}; return [(r, ib[r["rid"]]) for r in rows_a if r["rid"] in ib]


def sections(clf, ae, aek):
    md = []; B3 = ("b10_100", "b100_1000", "b1000_up"); bl = lambda b: b[1:].replace("_up", "+").replace("_", "-")
    mlp = lambda ds, p, rows=None: sel(rows if rows is not None else ae, dataset=ds, probe=p, arch="mlp", d=16)
    # A. linear AE anchor
    md.append("\n### A. linear AE (d=16, wd=0), test probe: R1 / unexplained by band\n\n| dataset | N | R1 10-100 | R1 100-1000 | R1 1000+ | unexpl 1000+ | R2-R1 1000+ | trace/sym/anti share 1000+ |\n|---|---|---|---|---|---|---|---|")
    for ds in DS:
        for N in NS:
            rr = sel(ae, dataset=ds, probe="test", arch="linear", d=16, N=N)
            if rr: g = lambda k: M([r[k] for r in rr]); md.append(f"| {ds} | {NL[N]} | {g('R1_b10_100'):.3f} | {g('R1_b100_1000'):.3f} | {g('R1_b1000_up'):.3f} | {g('U_b1000_up'):.3f} | {g('dR2_b1000_up'):+.3f} | {g('trsh_b1000_up'):.2f}/{g('symsh_b1000_up'):.2f}/{g('antish_b1000_up'):.2f} |")
    # B. label vs kmeans
    md.append("\n### B. label-wise vs label-free k-means(10)-wise affine increment R2-R1 (AE MLP d16; all N, wd, seeds pooled; paired by run)\n\n| dataset | probe | band | R2-R1 labels | R2-R1 k-means | labels - kmeans (paired mean +- se) | frac runs labels>kmeans |\n|---|---|---|---|---|---|---|")
    for ds in DS:
        for p in ("test", "train"):
            pr = pair(mlp(ds, p), mlp(ds, p, aek), "dR2")
            for b in B3:
                if not pr: continue
                a = np.array([x[f"dR2_{b}"] for x, _ in pr]); k = np.array([y[f"dR2_{b}"] for _, y in pr]); d_ = a - k
                md.append(f"| {ds} | {p} | {bl(b)} | {M(a):+.4f} | {M(k):+.4f} | {M(d_):+.4f} +- {SE(d_):.4f} | {np.mean(d_ > 0):.2f} |")
    # C. train vs test probe
    md.append("\n### C. train probe minus test probe (AE MLP d16, paired by run; N>=1000 where both probes have 1000 points)\n\n| dataset | band | metric | train | test | train - test (+- se) | n runs |\n|---|---|---|---|---|---|---|")
    for ds in DS:
        tr = [r for r in mlp(ds, "train") if r["N"] >= 1000]; pr = pair(tr, mlp(ds, "test"), "R1")
        for b in B3:
            for k, nm in (("R1", "R1"), ("dR2", "R2-R1 (labels)"), ("dR3", "R3-R2"), ("U", "unexplained"), ("trsh", "trace share"), ("antish", "antisym share")):
                a = np.array([x[f"{k}_{b}"] for x, _ in pr]); t = np.array([y[f"{k}_{b}"] for _, y in pr]); d_ = a - t
                md.append(f"| {ds} | {bl(b)} | {nm} | {M(a):.3f} | {M(t):.3f} | {M(d_):+.4f} +- {SE(d_):.4f} | {len(pr)} |")
    # clf FM train vs test for reference
    md.append("\n#### classifier FM (phase grid, wd 0 and 1e-4, N>=1000): train - test\n\n| band | metric | train | test | diff |\n|---|---|---|---|---|")
    ct, cr = clf[("fashion_mnist", "test")], clf[("fashion_mnist", "train")]; key = lambda r: (r["N"], r["wd"], r["seed"])
    it = {key(r): r for r in ct}; pr = [(r, it[key(r)]) for r in cr if key(r) in it and r["N"] >= 1000 and r["wd"] in (0.0, 1e-4)]
    for b in B3:
        for k, nm in (("R1", "R1"), ("dR2", "R2-R1"), ("U", "unexplained")):
            a = np.array([x[f"{k}_{b}"] for x, _ in pr]); t = np.array([y[f"{k}_{b}"] for _, y in pr]); md.append(f"| {bl(b)} | {nm} | {M(a):.3f} | {M(t):.3f} | {M(a-t):+.4f} |")
    # D. gap / N-dummy analysis
    md.append("\n### D. within-(N, wd) Spearman of late-band (1000+) quantity vs final recon gap (test-train), AE MLP d16; group-demeaned, permutation p (within groups)\n\n| dataset | probe | quantity | rho | p | n |\n|---|---|---|---|---|---|")
    for ds in DS + ["both"]:
        for p in ("test", "train"):
            rows = [r for r in ae if r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == p and (ds == "both" or r["dataset"] == ds)]
            for k in ("R1", "dR2", "dR3", "U", "trsh", "antish", "iso", "Lnorm"):
                kk = f"{k}_b1000_up"; rho, pv, n = within_corr(rows, "gap_final", kk); md.append(f"| {ds} | {p} | {k} | {rho:+.2f} | {pv:.3f} | {n} |")
    md.append("\nBetween-group (N confounded) Spearman for contrast, test probe, both datasets:\n\n| quantity | rho(gap_final) | rho(log N) |\n|---|---|---|")
    rows = [r for r in ae if r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == "test"]
    for k in ("R1", "dR2", "U", "trsh"):
        kk = f"{k}_b1000_up"; md.append(f"| {k} | {spear([r['gap_final'] for r in rows], [r[kk] for r in rows]):+.2f} | {spear([np.log(r['N']) for r in rows], [r[kk] for r in rows]):+.2f} |")
    # E. dimension
    md.append("\n### E. latent dimension (AE MLP, wd 0, test probe, N in {1k, full}): R1 / unexplained late\n\n| dataset | d | N | R1 100-1000 | R1 1000+ | unexpl 1000+ | R2-R1 1000+ | #expanding/d |\n|---|---|---|---|---|---|---|---|")
    for ds in DS:
        for d in (8, 16, 64):
            for N in (1000, 60000):
                rr = sel(ae, dataset=ds, probe="test", arch="mlp", d=d, N=N, wd=0.0)
                if rr: g = lambda k: M([r[k] for r in rr]); md.append(f"| {ds} | {d} | {NL[N]} | {g('R1_b100_1000'):.3f} | {g('R1_b1000_up'):.3f} | {g('U_b1000_up'):.3f} | {g('dR2_b1000_up'):+.3f} | {g('glob_n_exp_b1000_up'):.1f}/{d} |")
    open(f"{OUTD}/lds/table_sections.md", "w").write("\n".join(md)); print("\n".join(md))


def curve(rows, key):
    st = rows[0]["steps"][1:]; return st, np.nanmean([r[key] for r in rows], 0), np.nanstd([r[key] for r in rows], 0)


def figures(clf, ae, aek):
    C = {"clf": "#d95f02", "AE": "#1b9e77", "lin": "#7570b3"}
    # fig 1: R1 curves
    fig, ax = plt.subplots(2, 4, figsize=(17, 7), sharey=True)
    for i, ds in enumerate(DS):
        for j, N in enumerate((250, 1000, 4000, 60000)):
            a = ax[i, j]
            for nm, rows, ls in (("classifier", sel(clf[(ds, "test")], N=N, wd=0.0), C["clf"]), ("AE (MLP)", sel(ae, dataset=ds, probe="test", arch="mlp", d=16, N=N, wd=0.0), C["AE"]), ("AE (linear)", sel(ae, dataset=ds, probe="test", arch="linear", d=16, N=N), C["lin"])):
                if not rows: continue
                for pi, (ln, lsty) in enumerate((("R1 global affine", "-"), ("R2 + label-wise", "--"))):
                    st = rows[0]["steps"][1:]; y = np.mean([r["_P"][pi] for r in rows], 0); a.plot(st, y, lsty, color=ls, label=f"{nm} {ln}" if True else None, lw=1.6)
            a.set_xscale("log"); a.set_title(f"{ds}  N={NL[N]}  (test probe, wd 0, mean of 3 seeds)", fontsize=9); a.set_ylim(0.3, 1.0); a.grid(alpha=0.3)
            if i == 1: a.set_xlabel("training step (interval end)")
            if j == 0: a.set_ylabel("held-out share of ||dz||^2")
    h, l = ax[0, 0].get_legend_handles_labels(); fig.legend(h, l, ncol=6, loc="lower center", fontsize=8); fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(f"{OUTD}/lds_fig1_R1_time.png", dpi=130); plt.close(fig)
    # fig 2: generator shares per band, N=full and 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for i, ds in enumerate(DS):
        xs = []; w = 0.8 / 6
        for bi, b in enumerate(("b10_100", "b100_1000", "b1000_up")):
            for mi, (nm, rows, col) in enumerate((("clf", sel(clf[(ds, "test")], N=1000, wd=0.0), C["clf"]), ("AE", sel(ae, dataset=ds, probe="test", arch="mlp", d=16, N=1000, wd=0.0), C["AE"]))):
                v = [M([r[f"{k}_{b}"] for r in rows]) for k in ("trsh", "symsh", "antish")]; x0 = bi + (mi - 0.5) * 0.38
                bot = 0
                for vi, (vv, hat) in enumerate(zip(v, ("", "//", "xx"))): ax[i].bar(x0, vv, 0.34, bottom=bot, color=col, alpha=0.35 + 0.3 * vi, hatch=hat, edgecolor="k", lw=0.4, label=f"{nm} {('trace','sym (traceless)','antisym')[vi]}" if bi == 0 else None); bot += vv
        ax[i].set_xticks(range(3)); ax[i].set_xticklabels(["10-100", "100-1000", "1000+"]); ax[i].set_title(f"{ds}, N=1k: share of ||L||_F^2 (left bar clf, right AE)", fontsize=9); ax[i].set_xlabel("step band")
    ax[0].legend(fontsize=6, ncol=2); fig.tight_layout(); fig.savefig(f"{OUTD}/lds_fig2_generator_shares.png", dpi=130); plt.close(fig)
    # fig 3: label vs kmeans increment; train vs test
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for ds, mk in zip(DS, ("o", "s")):
        pr = pair([r for r in ae if r["dataset"] == ds and r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == "test"], [r for r in aek if r["dataset"] == ds and r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == "test"], "dR2")
        ax[0].scatter([y["dR2_b1000_up"] for _, y in pr], [x["dR2_b1000_up"] for x, _ in pr], marker=mk, alpha=0.7, label=ds)
        tr = [r for r in ae if r["dataset"] == ds and r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == "train" and r["N"] >= 1000]; p2 = pair(tr, [r for r in ae if r["dataset"] == ds and r["probe"] == "test"], "U")
        ax[1].scatter([y["U_b1000_up"] for _, y in p2], [x["U_b1000_up"] for x, _ in p2], marker=mk, alpha=0.7, label=ds)
        rows = [r for r in ae if r["dataset"] == ds and r["arch"] == "mlp" and r["d"] == 16 and r["probe"] == "test"]
        for N, col in zip(NS, ("#1b9e77", "#7570b3", "#e7298a", "#66a61e")): rr = [r for r in rows if r["N"] == N]; ax[2].scatter([r["gap_final"] for r in rr], [r["U_b1000_up"] for r in rr], marker=mk, color=col, alpha=0.75, label=f"{ds[:5]} N={NL[N]}")
    for a_ in ax[:2]: lim = [0, 0.2]; a_.plot(lim, lim, "k:", lw=0.8)
    ax[0].plot([0, 0.12], [0, 0.12], "k:", lw=0.8); ax[0].set_xlabel("R2-R1, k-means(10) groups (label-free)"); ax[0].set_ylabel("R2-R1, label groups"); ax[0].set_title("class-wise affine increment, step>=1000", fontsize=9); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("unexplained 1-R3, TEST probe"); ax[1].set_ylabel("unexplained 1-R3, TRAIN probe"); ax[1].set_title("train vs test probe (N>=1k), step>=1000", fontsize=9); ax[1].legend(fontsize=7)
    ax[2].set_xlabel("final recon gap (test-train)"); ax[2].set_ylabel("unexplained 1-R3 (test probe), step>=1000"); ax[2].set_title("vs generalization gap (colour = N)", fontsize=9); ax[2].legend(fontsize=6, ncol=2)
    for a_ in ax: a_.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUTD}/lds_fig3_labels_traintest_gap.png", dpi=130); plt.close(fig)


def main():
    clf = {(ds, p): L.load_classifier(ds, p) for ds in DS for p in ("test", "train") if not (ds == "mnist" and p == "train")}
    ae = L.load_ae("lab"); aek = L.load_ae("km"); print(len(ae), len(aek), "AE rows (lab,km)")
    out = {}
    # ---- table 1: side by side, test probe, wd0, d16 MLP
    lines = []
    for ds in DS:
        lines.append(f"\n### {ds} (test probe, wd=0, mean over 3 seeds; bands = interval-end step)\n")
        lines.append("| N | model | band | R1 global affine | R2-R1 (+label-wise) | R3-R2 (+kNN nonlin) | unexplained | trace share | sym share | antisym share | glob abscissa | #expanding (of 16) | #complex pairs |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for N in NS:
            for nm, rows in (("clf", sel(clf[(ds, "test")], N=N, wd=0.0)), ("AE", sel(ae, dataset=ds, probe="test", arch="mlp", d=16, N=N, wd=0.0))):
                if not rows: continue
                for b in ("b10_100", "b100_1000", "b1000_up"):
                    g = lambda k: M([r.get(f"{k}_{b}", np.nan) for r in rows])
                    lines.append(f"| {NL[N]} | {nm} | {b[1:].replace('_up','+').replace('_','-')} | {g('R1'):.3f} | {g('dR2'):+.3f} | {g('dR3'):+.3f} | {g('U'):.3f} | {g('trsh'):.2f} | {g('symsh'):.2f} | {g('antish'):.2f} | {g('glob_abscissa')*1e3:+.2f}e-3 | {g('glob_n_exp'):.1f} | {g('glob_n_cplx'):.1f} |")
    open(f"{OUTD}/lds/table_side_by_side.md", "w").write("\n".join(lines)); print("\n".join(lines))
    sections(clf, ae, aek); figures(clf, ae, aek)
    return clf, ae, aek


if __name__ == "__main__": main()
