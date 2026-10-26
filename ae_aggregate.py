"""Cross-run tables and figures for ae_analysis.py outputs.   CPU only.
Usage: python ae_aggregate.py [--analysis DIR] [--runs GLOB] [--out DIR] [--dataset mnist] [--d 16]   (safe to re-run on partial data; configs with no cached runs are skipped)
  also (config = family/d/N/wd, mean +- sd over seeds): summary_by_config.csv + summary_by_config.md, figures/generalization_vs_N.png, gen_timeseries.png, probe_vs_ae.png, d_sweep.png, staging_pc_rise.png
  reads DIR/runs/<id>.{json,npz}; writes OUT/summary.csv (one row per run, all json scalars) and figures in OUT/figures/:
  ts_geometry.png (label vs label-free structure vs time), ts_generalization.png (recon gap, train/test geometry), ts_pca.png (PCA alignment + per-PC staging),
  ts_vfield.png (global / group-wise affine / nonlinear R2, label vs input-cluster groups), bars_final.png (final-value summary across cells), probe_vs_ae.png (when probe-objective runs are present).
Gauge note (repeated on figures): all plotted quantities are invariant to invertible linear re-coordinatisation of the latent (or are ratios), except effective ranks (rotation-invariant only).
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import argparse, glob, json, csv, fnmatch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/vol/bitbucket/lbastian/latent-dynamics"
AN = f"{ROOT}/ae_dynamics/analysis"
NOTE = "gauge-invariant quantities (affine-removed R2, ratios, principal angles, CKA)"


def family(J):
    return "linAE" if J.get("linear_ae") else {"autoencoder": "AE", "probe_recon": "probe+recon", "probe": "probe"}[J["objective"]]


def load(an, pat):
    runs = []
    for f in sorted(glob.glob(f"{an}/runs/*.json")):
        rid = os.path.basename(f)[:-5]
        if pat and not fnmatch.fnmatch(rid, pat): continue
        J = json.load(open(f)); R = dict(np.load(f"{an}/runs/{rid}.npz")); runs.append((rid, J, R))
    return runs


def colour(N, Ns):
    cm = plt.get_cmap("viridis"); i = Ns.index(N); return cm(i / max(1, len(Ns) - 1) * 0.9)


def style(J, Ns):
    ls = "-" if not J["weight_decay"] else "--"; return dict(color=colour(J["N"], Ns), ls=ls, lw=1.1, alpha=0.8)


def lab(J): return f"{J['dataset'][:5]} {family(J)} N={J['N']} wd={J['weight_decay']:g}"


def panel_grid(runs, specs, fname, title, ncol=3, xlog=True):
    Ns = sorted({J["N"] for _, J, _ in runs}); n = len(specs); nr = (n + ncol - 1) // ncol
    fig, axs = plt.subplots(nr, ncol, figsize=(4.6 * ncol, 3.2 * nr), squeeze=False); seen = set()
    for ax, (key, yl, opts) in zip(axs.flat, specs):
        for rid, J, R in runs:
            keys = key if isinstance(key, list) else [key]
            for ki, k in enumerate(keys):
                if k not in R: continue
                y = R[k]; x = R["steps"] if len(y) == len(R["steps"]) else R["steps"][1:] if len(y) == len(R["steps"]) - 1 else None
                if x is None: continue
                st = style(J, Ns)
                if len(keys) > 1: st["ls"] = ["-", ":"][ki]
                lb = f"{lab(J)}" if (lab(J), ki) not in seen and ki == 0 else None; seen.add((lab(J), ki)); ax.plot(np.maximum(x, 0.5), y, label=lb, **st)
        ax.set_xscale("log" if xlog else "linear"); ax.set_title(yl, fontsize=9); ax.set_xlabel("step")
        if opts.get("log"): ax.set_yscale("log")
        if opts.get("hline") is not None: ax.axhline(opts["hline"], color="grey", lw=0.6)
        if opts.get("ylim"): ax.set_ylim(*opts["ylim"])
    for ax in axs.flat[n:]: ax.axis("off")
    h, l = axs.flat[0].get_legend_handles_labels()
    if h: fig.legend(h, l, loc="lower center", ncol=min(5, len(l)), fontsize=7, frameon=False)
    fig.suptitle(f"{title}   [{NOTE}]", fontsize=9); fig.tight_layout(rect=(0, 0.06, 1, 0.96)); fig.savefig(fname, dpi=110); plt.close(fig)


# ---------------------------------------------------------------- config parsing / seed aggregation
import re
_AE = re.compile(r"^(?P<ds>.+?)_ae_(?P<arch>mlp|conv|linear)_d(?P<d>\d+)_s(?P<seed>\d+)_n(?P<N>full|\d+)_wd(?P<wd>[0-9.eE+-]+)$")
_PR = re.compile(r"^(?P<ds>.+?)_probe_none_d(?P<d>\d+)_s(?P<seed>\d+)_phase_n(?P<N>full|\d+)_wd(?P<wd>[0-9.eE+-]+)$")
FULL = 60000


def parse_run(rid, J):
    """-> dict(family in ae_mlp/ae_linear/probe/probe_recon/ae_old, dataset, d, N (int, full->60000 or pool size), wd, seed). Name first, json fallback (legacy/tiny names)."""
    jf = "ae_linear" if J.get("linear_ae") else {"autoencoder": "ae_mlp", "probe": "probe", "probe_recon": "probe_recon"}.get(J.get("objective"), str(J.get("objective")))
    m = _AE.match(rid) or _PR.match(rid); P = dict(family=jf, dataset=J.get("dataset", "?"), d=J.get("latent_dim"), N=J.get("N"), wd=J.get("weight_decay", 0.0), seed=J.get("seed"))
    if m:
        g = m.groupdict(); P.update(dataset=g["ds"], d=int(g["d"]), seed=int(g["seed"]), wd=float(g["wd"]), N=FULL if g["N"] == "full" else int(g["N"]))
        P["family"] = ("ae_" + ("mlp" if g["arch"] == "conv" else g["arch"])) if "arch" in g else "probe"  # conv AE reported under the nonlinear-AE family
    else: P["family"] += "_legacy"   # non-grid names (old single runs, tiny, fashion...) kept in tables but kept out of the grid figures
    if P["N"] is not None and P["N"] >= FULL: P["N"] = FULL
    return P


def Nlab(N): return "full" if N is not None and N >= FULL else str(N)


FAM_COL = {"ae_mlp": "#d95f02", "ae_linear": "#7570b3", "probe": "#1b9e77", "probe_recon": "#666666"}
SCALARS = [("final_recon_train", "recon train"), ("final_recon_test", "recon test"), ("final_recon_gap", "recon gap"), ("cls_loss_gap", "cls loss gap"), ("cls_acc_gap", "cls acc gap"),
           ("final_te_lab_ncm", "NCM lab"), ("final_te_km_ncm", "NCM km"), ("final_te_lab_knn", "kNN lab"), ("final_te_km_knn", "kNN km"),
           ("final_te_lab_fisher", "Fisher lab"), ("final_te_km_fisher", "Fisher km"), ("vf_te_lab_classaffine_gain_late", "R2gain lab"), ("vf_te_km_classaffine_gain_late", "R2gain km"),
           ("final_gen_spread_ratio_lab", "spread lab"), ("final_gen_spread_ratio_km", "spread km"), ("final_te_pa_mean_cos2", "PCA cos2"), ("te_pc_rise_spearman_dd", "rise Spearman"), ("final_te_effrank", "eff rank")]


def annotate(runs):
    out = []
    for rid, J, R in runs:
        P = parse_run(rid, J); J = dict(J)
        if "sc_test_loss" in R and "sc_train_loss" in R and P["family"] == "probe": J["cls_loss_gap"] = float(R["sc_test_loss"][-1] - R["sc_train_loss"][-1])
        if "sc_test_acc" in R and "sc_train_acc" in R and P["family"] == "probe": J["cls_acc_gap"] = float(R["sc_train_acc"][-1] - R["sc_test_acc"][-1])
        out.append(dict(rid=rid, J=J, R=R, **P))
    return out


def cfg_key(r): return (r["family"], r["dataset"], r["d"], r["N"], r["wd"])


def groups(A):
    G = {}
    for r in A: G.setdefault(cfg_key(r), []).append(r)
    return G


def ms(vals):
    v = np.array([x for x in vals if isinstance(x, (int, float)) and np.isfinite(x)], float)
    return (v.mean(), v.std(), len(v)) if len(v) else (np.nan, np.nan, 0)


def write_config_tables(A, out):
    G = groups(A); keys = sorted(G, key=lambda k: (k[1], k[0], k[2], k[3], k[4]))
    with open(f"{out}/summary_by_config.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["family", "dataset", "d", "N", "wd", "n_seeds"] + [h for k, _ in SCALARS for h in (k + "_mean", k + "_sd")])
        for k in keys:
            row = list(k[:2]) + [k[2], Nlab(k[3]), k[4], len(G[k])]
            for sk, _ in SCALARS: m, s, n = ms([r["J"].get(sk) for r in G[k]]); row += [m, s]
            w.writerow(row)
    with open(f"{out}/summary_by_config.md", "w") as f:
        f.write("| config | n | " + " | ".join(n for _, n in SCALARS) + " |\n|" + "---|" * (len(SCALARS) + 2) + "\n")
        for k in keys:
            cells = []
            for sk, _ in SCALARS:
                m, s, n = ms([r["J"].get(sk) for r in G[k]]); cells.append("-" if n == 0 else (f"{m:.3g}" if n == 1 else f"{m:.3g}±{s:.2g}"))
            f.write(f"| {k[1]} {k[0]} d{k[2]} N={Nlab(k[3])} wd={k[4]:g} | {len(G[k])} | " + " | ".join(cells) + " |\n")
    print(f"{len(keys)} configs -> {out}/summary_by_config.{{csv,md}}")


def curve(rs, key):
    """seed mean/sd of time series `key` over runs rs; returns (x, mean, sd) or None. Series on differing step grids are interpolated (log1p) onto the first run's."""
    xs, ys = [], []
    for r in rs:
        R = r["R"]
        if key not in R: continue
        y = np.asarray(R[key], float); st = R["steps"]; x = st if len(y) == len(st) else st[1:] if len(y) == len(st) - 1 else None
        if x is not None: xs.append(x); ys.append(y)
    if not ys: return None
    x0 = xs[0]; Y = np.stack([y if len(y) == len(x0) and np.array_equal(x, x0) else np.interp(np.log1p(x0), np.log1p(x), y) for x, y in zip(xs, ys)])
    return np.maximum(x0, 0.5), np.nanmean(Y, 0), (np.nanstd(Y, 0) if len(Y) > 1 else np.zeros(Y.shape[1]))


def band(ax, c, **kw):
    if c is None: return False
    x, m, s = c; ax.plot(x, m, **kw); ax.fill_between(x, m - s, m + s, color=kw.get("color"), alpha=0.18, lw=0); return True


def finish(fig, F, name, title, rect=(0, 0.05, 1, 0.95)):
    h, l = {}, {}
    for ax in fig.axes:
        for hh, ll in zip(*ax.get_legend_handles_labels()): l.setdefault(ll, hh)
    if l: fig.legend(list(l.values()), list(l.keys()), loc="lower center", ncol=min(7, len(l)), fontsize=7, frameon=False)
    fig.suptitle(f"{title}   [{NOTE}]", fontsize=9); fig.tight_layout(rect=rect); fig.savefig(f"{F}/{name}", dpi=110); plt.close(fig); print("wrote", name)


def sel(G, **kw):
    return {k: v for k, v in G.items() if all({"family": k[0], "dataset": k[1], "d": k[2], "N": k[3], "wd": k[4]}[a] == b for a, b in kw.items())}


def fig_generalization(A, F, ds, d):
    G = groups([r for r in A if r["dataset"] == ds and r["d"] == d and not r["family"].endswith("_legacy")]); wds = sorted({k[4] for k in G}); Ns = sorted({k[3] for k in G})
    if not G: return
    specs = [("final_recon_gap", "AE: final recon gap  test/train-1", ("ae_mlp", "ae_linear")), ("final_recon_test", "AE: final test recon MSE", ("ae_mlp", "ae_linear")),
             ("cls_loss_gap", "classifier: test-train loss gap", ("probe",)), ("cls_acc_gap", "classifier: train-test acc gap", ("probe",))]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.6)); skipped = []
    for ax, (sk, ttl, fams) in zip(axs, specs):
        drew = False
        for fam in fams:
            for wd in wds:
                pts = [(k[3], *ms([r["J"].get(sk) for r in v])) for k, v in sorted(sel(G, family=fam, wd=wd).items(), key=lambda kv: kv[0][3])]; pts = [p for p in pts if p[3] > 0]
                if not pts: continue
                x = np.array([p[0] for p in pts], float); m = np.array([p[1] for p in pts]); s = np.array([p[2] for p in pts]); c = FAM_COL[fam]
                ax.plot(x, m, "o-" if wd == 0 else "s--", color=c, label=f"{fam} wd={wd:g}"); ax.fill_between(x, m - s, m + s, color=c, alpha=0.15, lw=0); drew = True
        ax.set_xscale("log"); ax.set_title(ttl, fontsize=9); ax.set_xlabel("N train"); ax.axhline(0, color="grey", lw=0.5)
        if sk == "final_recon_test": ax.set_yscale("log")
        if not drew: ax.text(0.5, 0.5, "no cached runs", transform=ax.transAxes, ha="center", color="grey"); skipped.append(sk)
    finish(fig, F, "generalization_vs_N.png", f"Generalization vs N ({ds}, d={d}; mean+-sd over seeds; classifier panels need probe caches with sc_{{train,test}}_loss)")
    # time series by N
    fams = [f for f in ("ae_mlp", "ae_linear", "probe") if any(k[0] == f for k in G)]; keys = [("recon_gap_rel", "recon gap"), ("sc_test_acc", "test acc (probe readout)"), ("gen_spread_ratio_lab", "spread ratio lab")]
    fig, axs = plt.subplots(len(keys), len(fams) * len(wds), figsize=(3.6 * len(fams) * len(wds), 2.8 * len(keys)), squeeze=False)
    for i, (key, kt) in enumerate(keys):
        for j, (fam, wd) in enumerate([(f, w) for f in fams for w in wds]):
            ax = axs[i, j]; ax.set_xscale("log"); ax.set_title(f"{fam} wd={wd:g}: {kt}", fontsize=8)
            for k, v in sorted(sel(G, family=fam, wd=wd).items(), key=lambda kv: kv[0][3]): band(ax, curve(v, key), color=colour(k[3], Ns), label=f"N={Nlab(k[3])}")
    finish(fig, F, "gen_timeseries.png", f"Generalization over training by N ({ds}, d={d})")


def fig_probe_vs_ae(A, F, ds, d):
    G = groups([r for r in A if r["dataset"] == ds and r["d"] == d and not r["family"].endswith("_legacy")]); cells = sorted({(k[3], k[4]) for k in G}); cells = [c for c in cells if len({k[0] for k in G if (k[3], k[4]) == c}) >= 2 and any(k[0] == "probe" for k in G if (k[3], k[4]) == c)]
    if not cells: print("probe_vs_ae: no (N,wd) cell with both AE and probe cached; skipped"); return
    cols = [(("te_lab_ncm", "te_km_ncm"), "NCM acc (solid lab / dashed km)"), (("te_lab_fisher", "te_km_fisher"), "Fisher (solid lab / dashed km)"),
            (("gen_spread_ratio_lab", "gen_spread_ratio_km"), "spread ratio train/test (lab / km)"), (("vf_gain_lab", "vf_gain_km"), "R2 gain class-affine (lab / km)")]
    fig, axs = plt.subplots(len(cells), 4, figsize=(15, 2.7 * len(cells)), squeeze=False)
    for i, (N, wd) in enumerate(cells):
        for j, (keys, ttl) in enumerate(cols):
            ax = axs[i, j]; ax.set_xscale("log"); ax.set_title(f"N={Nlab(N)} wd={wd:g}  {ttl}", fontsize=7)
            if "fisher" in keys[0]: ax.set_yscale("log")
            if "spread" in keys[0]: ax.axhline(1, color="grey", lw=0.5)
            for fam in ("ae_mlp", "ae_linear", "probe"):
                for k, v in sel(G, family=fam, N=N, wd=wd).items():
                    for q, key in enumerate(keys):
                        if key.startswith("vf_gain"):
                            g = key[8:]; cs = [curve([dict(R=dict(steps=r["R"]["steps"], y=r["R"][f"vf_te_{g}_R2_classaffine"] - r["R"][f"vf_te_{g}_R1_affine"])) for r in v if f"vf_te_{g}_R2_classaffine" in r["R"]], "y")]
                            c = cs[0]
                        else: c = curve(v, key)
                        band(ax, c, color=FAM_COL[fam], ls="-" if q == 0 else "--", lw=1.2, label=f"{fam} (n={len(v)})" if q == 0 else None)
    finish(fig, F, "probe_vs_ae.png", f"AE vs probe classifier, matched (N, wd), {ds} d={d}; colour = family", rect=(0, 0.04, 1, 0.97))


def fig_dsweep(A, F, ds):
    G = groups([r for r in A if r["dataset"] == ds and not r["family"].endswith("_legacy")]); ds_ = sorted({k[2] for k in G})
    if len(ds_) < 2: print("d_sweep: <2 latent dims cached; skipped"); return
    rows = sorted({(k[0], k[3], k[4]) for k in G if len({kk[2] for kk in G if kk[:2] == k[:2] and kk[3:] == k[3:]}) >= 2})
    if not rows: print("d_sweep: no config with >=2 d values; skipped"); return
    keys = [("sc_recon_test", "test recon MSE"), ("te_lab_ncm", "NCM lab"), ("te_pa_mean_cos2", "PCA mean cos2"), ("te_lab_fisher", "Fisher lab")]; cm = plt.get_cmap("plasma")
    fig, axs = plt.subplots(len(rows), 4, figsize=(15, 2.6 * len(rows)), squeeze=False)
    for i, (fam, N, wd) in enumerate(rows):
        for j, (key, kt) in enumerate(keys):
            ax = axs[i, j]; ax.set_xscale("log"); ax.set_title(f"{fam} N={Nlab(N)} wd={wd:g}: {kt}", fontsize=7)
            if key in ("sc_recon_test", "te_lab_fisher"): ax.set_yscale("log")
            for q, d in enumerate(ds_):
                v = G.get((fam, ds, d, N, wd))
                if v: band(ax, curve(v, key), color=cm(q / max(1, len(ds_) - 1) * 0.85), label=f"d={d}")
    finish(fig, F, "d_sweep.png", f"Latent-dimension sweep ({ds})", rect=(0, 0.04, 1, 0.97))


def fig_staging(A, F, ds, d, K=None):
    G = groups([r for r in A if r["dataset"] == ds and r["d"] == d and not r["family"].endswith("_legacy")]); Ns = sorted({k[3] for k in G}); wds = sorted({k[4] for k in G})
    if not G: return
    K = K or d + 4; fig, axs = plt.subplots(len(wds), len(Ns), figsize=(3.4 * len(Ns), 3 * len(wds)), squeeze=False); any_ = False
    for i, wd in enumerate(wds):
        for j, N in enumerate(Ns):
            ax = axs[i, j]; ax.set_title(f"N={Nlab(N)} wd={wd:g}", fontsize=8); ax.set_yscale("log"); ax.axvline(d + 0.5, color="grey", lw=0.5); ax.set_xlabel("PC index"); ax.set_ylabel("rise step")
            for fam in ("ae_mlp", "ae_linear", "probe"):
                v = G.get((fam, ds, d, N, wd))
                if not v: continue
                Y = np.stack([np.maximum(np.asarray(r["R"]["te_pc_rise"], float)[:K], 0.5) for r in v if "te_pc_rise" in r["R"]]) if any("te_pc_rise" in r["R"] for r in v) else None
                if Y is None: continue
                with np.errstate(all="ignore"): m = np.nanmean(Y, 0); s = np.nanstd(Y, 0)
                x = np.arange(1, Y.shape[1] + 1); ax.plot(x, m, "o-", ms=3, color=FAM_COL[fam], label=f"{fam} (n={len(Y)})"); ax.fill_between(x, np.maximum(m - s, 0.5), m + s, color=FAM_COL[fam], alpha=0.18, lw=0); any_ = True
    if any_: finish(fig, F, "staging_pc_rise.png", f"Per-PC half-rise step vs PC index ({ds}, d={d}; NaN = PC not learned, omitted)")
    else: plt.close(fig)


def aggregate_configs(runs, out, F, ds, d):
    A = annotate(runs); write_config_tables(A, out)
    for fn in (lambda: fig_generalization(A, F, ds, d), lambda: fig_probe_vs_ae(A, F, ds, d), lambda: fig_dsweep(A, F, ds), lambda: fig_staging(A, F, ds, d)):
        try: fn()
        except Exception as e:
            import traceback; traceback.print_exc(); print("figure failed:", repr(e))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--analysis", default=AN); ap.add_argument("--runs", default=""); ap.add_argument("--out", default=""); ap.add_argument("--dataset", default="mnist"); ap.add_argument("--d", type=int, default=16)
    a = ap.parse_args(); global FULL
    if a.dataset == "cifar10": FULL = 50000
    out = a.out or a.analysis; os.makedirs(f"{out}/figures", exist_ok=True); runs = load(a.analysis, a.runs)
    if not runs: print("no analysed runs"); return
    # ---- table
    cols = sorted({k for _, J, _ in runs for k, v in J.items() if isinstance(v, (int, float, bool, str)) and k != "pc_rise_final_note"}); head = ["run_id", "family"] + [c for c in cols if c != "run_id"]
    with open(f"{out}/summary.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(head)
        for rid, J, _ in runs: w.writerow([rid, family(J)] + [J.get(c, "") for c in head[2:]])
    print(f"{len(runs)} runs -> {out}/summary.csv")
    F = f"{out}/figures"
    panel_grid(runs, [("te_lab_fisher", "Fisher tr(Sw)/tr(Sb), LABEL groups (test probe)", dict(log=True)), ("te_km_fisher", "Fisher, INPUT-CLUSTER groups (label-free)", dict(log=True)),
                      ("te_lab_ncm", "NCM accuracy, label groups", dict(ylim=(0, 1))), ("te_km_ncm", "NCM accuracy, input-cluster groups", dict(ylim=(0, 1))),
                      ("te_lab_knn", "kNN(10) label agreement in latent", dict(ylim=(0, 1))), ("te_latkm_nmi", "NMI(k-means(10) on latents, labels): emergent class structure", dict(ylim=(0, 1))),
                      ("te_knn_overlap", "kNN(10) overlap with INPUT-space neighbours", dict(ylim=(0, 1))), ("te_effrank", "effective rank (*rotation-invariant only)", {}), ("sc_test_acc", "linear probe readout test acc (detached for AE)", dict(ylim=(0, 1)))],
               f"{F}/ts_geometry.png", "Class structure vs label-free structure over training")
    panel_grid(runs, [(["sc_recon_train", "sc_recon_test"], "recon MSE train (solid) / test (dotted)", dict(log=True)), ("recon_gap_rel", "relative gap test/train - 1", dict(hline=0)),
                      (["tr_recon_med", "te_recon_med"], "probe per-sample recon median (train solid / test dotted)", dict(log=True)), (["tr_recon_tail10", "te_recon_tail10"], "share of recon error in worst 10% samples", {}),
                      ("gen_spread_ratio_lab", "within-class spread train/test (label)", dict(hline=1)), ("gen_spread_ratio_km", "within-cluster spread train/test (input clusters)", dict(hline=1)),
                      ("gen_te2tr_nn", "median test->nearest-train latent dist / train NN dist", dict(hline=1)), ("gen_effrank_ratio", "effective rank train/test", dict(hline=1)), ("gen_var_ratio", "latent variance train/test", dict(hline=1))],
               f"{F}/ts_generalization.png", "Generalization: train vs test (missing panels = metric not logged / no train probe)")
    # PCA alignment + staging
    panel_grid(runs, [("te_pa_mean_cos2", "mean cos^2 principal angles: latent span vs top-d PCA (train pool)", dict(ylim=(0, 1))), ("te_pa_min_cos", "smallest principal-angle cosine", dict(ylim=(0, 1))),
                      ("te_cka_pca", "linear CKA(latent, top-d PC scores)", dict(ylim=(0, 1))), ("te_aff_recon_frac", "pixel variance explained by affine readout of latent", dict(ylim=(0, 1)))], f"{F}/ts_pca.png", "PCA alignment of the latent (test probe)", ncol=4)
    ae = [r for r in runs if family(r[1]) in ("AE", "linAE")] or runs; sel = ae[:min(6, len(ae))]
    fig, axs = plt.subplots(2, len(sel), figsize=(3.6 * len(sel), 6.2), squeeze=False)
    for j, (rid, J, R) in enumerate(sel):
        r2 = R["te_pc_r2"][:, :J["latent_dim"] + 4]; x = np.arange(len(R["steps"]))
        im = axs[0, j].imshow(r2.T, aspect="auto", origin="lower", vmin=0, vmax=1, cmap="magma"); axs[0, j].set_title(lab(J) + f" s{J['seed']}\nper-PC affine-readout R2", fontsize=7)
        tk = np.linspace(0, len(x) - 1, 5).astype(int); axs[0, j].set_xticks(tk); axs[0, j].set_xticklabels([str(int(R["steps"][t])) for t in tk], fontsize=6); axs[0, j].set_ylabel("PC index")
        rs = R["te_pc_rise"][:J["latent_dim"] + 4]; axs[1, j].semilogy(np.arange(1, len(rs) + 1), np.maximum(rs, 0.5), "o-"); axs[1, j].axvline(J["latent_dim"] + 0.5, color="grey", lw=0.6)
        axs[1, j].set_xlabel("PC index"); axs[1, j].set_ylabel("step of half-rise"); axs[1, j].set_title(f"rise order Spearman(PC<=d)={J.get('te_pc_rise_spearman_dd', float('nan')):.2f}", fontsize=7)
    fig.suptitle(f"Sequential learning of principal components [{NOTE}]", fontsize=9); fig.tight_layout(rect=(0, 0, 1, 0.95)); fig.savefig(f"{F}/staging.png", dpi=110); plt.close(fig)
    # vfield
    fig, axs = plt.subplots(1, 3, figsize=(14, 3.6));
    for ax, (grp, ttl) in zip(axs, (("lab", "label groups"), ("km", "input-cluster groups"), ("gain", "class-affine gain R2-R1 (solid lab / dotted km)"))):
        for rid, J, R in runs:
            if "vf_te_lab_R1_affine" not in R: continue
            Ns = sorted({J["N"] for _, J, _ in runs}); st = style(J, Ns); x = R["steps"][1:]
            if grp == "gain":
                ax.plot(np.maximum(x, .5), R["vf_te_lab_R2_classaffine"] - R["vf_te_lab_R1_affine"], **st); st = dict(st, ls=":"); ax.plot(np.maximum(x, .5), R["vf_te_km_R2_classaffine"] - R["vf_te_km_R1_affine"], **st)
            else:
                ax.plot(np.maximum(x, .5), R[f"vf_te_{grp}_R1_affine"], **dict(st, ls="-")); ax.plot(np.maximum(x, .5), R[f"vf_te_{grp}_R2_classaffine"], **dict(st, ls="--")); ax.plot(np.maximum(x, .5), R[f"vf_te_{grp}_R3_krr"], **dict(st, ls=":"))
        ax.set_xscale("log"); ax.set_title(ttl + ("\n(solid global affine R1, dashed +group affine R2, dotted +kernel R3)" if grp != "gain" else ""), fontsize=8); ax.set_xlabel("step")
    fig.suptitle(f"Displacement decomposition (held-out R2 of ||z(t+1)-z(t)||^2) [{NOTE}]", fontsize=9); fig.tight_layout(); fig.savefig(f"{F}/ts_vfield.png", dpi=110); plt.close(fig)
    # final-value bars per cell
    cells = {}
    for rid, J, R in runs: cells.setdefault((family(J), J["N"], J["weight_decay"]), []).append(J)
    keys = ["final_te_lab_fisher", "final_te_km_fisher", "final_te_lab_ncm", "final_te_km_ncm", "final_te_latkm_nmi", "final_te_pa_mean_cos2", "final_recon_gap", "final_gen_spread_ratio_lab", "final_gen_spread_ratio_km", "final_gen_te2tr_nn",
            "vf_te_lab_classaffine_gain_late", "vf_te_km_classaffine_gain_late"]
    fig, axs = plt.subplots(3, 4, figsize=(16, 9)); order = sorted(cells)
    for ax, k in zip(axs.flat, keys):
        v = [[J[k] for J in cells[c] if k in J and np.isfinite(J[k])] for c in order]
        ax.bar(range(len(order)), [np.mean(x) if x else 0 for x in v], yerr=[np.std(x) if len(x) > 1 else 0 for x in v], color="#4477aa"); ax.set_title(k, fontsize=8)
        ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{c[0]} N{c[1]} wd{c[2]:g}" for c in order], rotation=70, ha="right", fontsize=5)
    fig.suptitle(f"Final values per (family, N, wd) cell, mean +- sd over seeds [{NOTE}]", fontsize=9); fig.tight_layout(); fig.savefig(f"{F}/bars_final.png", dpi=110); plt.close(fig)
    aggregate_configs(runs, out, F, a.dataset, a.d)
    print("figures in", F)


if __name__ == "__main__": main()
