"""Eigen-event detection, reproducibility checks and summaries for vfield_spectra outputs.   Usage: python -m latentdyn.vfield.spectra_events [--out DIR]
Events (per sample-split series of the class-aggregated generator spectra; Lres = class-wise residual generator in the affine frame, Ltot = total class-wise generator):
  abscissa_cross : class-median spectral abscissa (max Re) changes sign persistently (>=6 smoothed intervals on both sides)
  nexp_shift / ncplx_shift : smoothed class-mean number of expanding directions / complex pairs shifts by >=1 between 8-interval windows
  jump : matched-eigenvalue jump |dl|/(|l|+|l'|) exceeds median + 8 MAD (spike-like spectral rearrangement)
Reproducibility: a run-level event must appear (same type, within 0.15 decades of log10 step) in >=4 of the 5 sample splits; a cell-level event must appear in >= ceil(2n/3) of the n seeds (2 of 3, 4 of 5) within 0.25 decades.
"""
import os, sys
from latentdyn._threads import cap_threads
cap_threads(8)
import argparse, glob, csv, re, math, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/spectra"
DEC = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/decomp/summary_runs.csv"; VFS = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/summary_runs.csv"
EPIS = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense/analysis/episodes.csv"; PHC = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/phase_cache"
TOL_SPLIT, TOL_SEED = 0.15, 0.25


def parse(fn):
    b = os.path.basename(fn)[:-4]; m = re.match(r"(.*)_(test|train)_(given|true)$", b); rid, probe, lm = m.groups()
    d = dict(rid=rid, probe=probe, label_mode=lm)
    m = re.search(r"_s(\d+)_(phase|gap|sl40k)_n(\w+?)_(wd|ln)([\d.e-]+)$", rid)
    if m:
        seed, fam, N, kind, val = m.groups(); d.update(seed=int(seed), family=fam, N=60000 if N == "full" else int(N), wd=float(val) if kind == "wd" else 0.0, ln=float(val) if kind == "ln" else 0.0, lr=1e-3, B=128); return d
    m = re.search(r"_s(\d+)_sgdlr([\d.]+)_n(\w+?)_wd([\d.e-]+)$", rid)
    if m: seed, lr, N, wd = m.groups(); d.update(seed=int(seed), family="sgdlr", N=int(N), wd=float(wd), ln=0.0, lr=float(lr), B=128); return d
    m = re.search(r"_s(\d+)_lrB_b(\d+)_lr([\d.]+)_n(\w+?)_wd([\d.e-]+)$", rid)
    seed, B, lr, N, wd = m.groups(); d.update(seed=int(seed), family="lrB", N=int(N), wd=float(wd), ln=0.0, lr=float(lr), B=int(B)); return d


def mmed(x, w=5):
    out = np.full_like(x, np.nan, dtype=float); h = w // 2
    for k in range(len(x)):
        seg = x[max(0, k - h):k + h + 1]; seg = seg[np.isfinite(seg)]
        if len(seg): out[k] = np.median(seg)
    return out


def log_mid(steps):
    s = steps.astype(float); return np.log10(np.sqrt((s[:-1] + 1) * (s[1:] + 1)))


def ev_cross(a, loc, P=6):
    s = mmed(a, 5); ev = []
    for k in range(P, len(s) - P - 1):
        if np.isfinite(s[k]) and np.isfinite(s[k + 1]) and s[k] * s[k + 1] < 0:
            before, after = s[k - P + 1:k + 1], s[k + 1:k + 1 + P]
            if np.all(np.sign(before) == np.sign(s[k])) and np.all(np.sign(after) == np.sign(s[k + 1])):
                t = s[k] / (s[k] - s[k + 1]); ev.append(("abscissa_cross", loc[k] + t * (loc[k + 1] - loc[k]), float(np.sign(s[k + 1])), abs(s[k + 1] - s[k])))
    return ev


def ev_shift(x, loc, name, w=8, thr=1.0, sep=8):
    s = mmed(x, 3); d = np.full(len(s), np.nan)
    for k in range(w, len(s) - w): d[k] = np.nanmedian(s[k:k + w]) - np.nanmedian(s[k - w:k])
    ev = []; ad = np.abs(d); k = 0; cand = [i for i in range(w, len(s) - w) if np.isfinite(ad[i]) and ad[i] >= thr and ad[i] >= np.nanmax(ad[max(0, i - sep):i + sep + 1])]
    for i in cand: ev.append((name, loc[min(i, len(loc) - 1)], float(np.sign(d[i])), float(ad[i])))
    return ev


def ev_jump(j, loc, nmad=8):
    x = j[np.isfinite(j)]
    if len(x) < 20: return []
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826 + 1e-12; thr = med + nmad * mad; ev = []
    for i in range(3, len(j) - 3):
        if np.isfinite(j[i]) and j[i] > thr and j[i] >= np.nanmax(j[i - 3:i + 4]): ev.append(("jump", loc[min(i, len(loc) - 1)], 1.0, float((j[i] - med) / mad)))
    return ev


def cluster(locs, tol):
    """greedy 1-D clustering of (loc, key) with tolerance; returns list of (center, members)."""
    items = sorted(locs); out = []
    for l, k in items:
        if out and abs(l - np.mean([m[0] for m in out[-1]])) <= tol: out[-1].append((l, k))
        else: out.append([(l, k)])
    return out


def run_events(z, gen):
    steps = z["steps"]; loc = log_mid(steps); S = z[f"abscissa_med_{gen}"].shape[0]; per_split = []
    for s in range(S):
        ev = ev_cross(z[f"abscissa_med_{gen}"][s][:len(loc)], loc) + ev_shift(z[f"n_exp_{gen}"][s][:len(loc)], loc, "nexp_shift") + ev_shift(z[f"n_cplx_{gen}"][s][:len(loc)], loc, "ncplx_shift") + ev_jump(z[f"jump_{gen}"][s][:len(loc)], loc)
        per_split.append(ev)
    cons = []
    for typ in ("abscissa_cross", "nexp_shift", "ncplx_shift", "jump"):
        items = [(e[1], s) for s, ev in enumerate(per_split) for e in ev if e[0] == typ]
        for cl in cluster(items, TOL_SPLIT):
            sup = len({m[1] for m in cl})
            if sup >= max(1, math.ceil(0.8 * S)): cons.append(dict(type=typ, loc=float(np.median([m[0] for m in cl])), support=sup, n_splits=S))
    return cons, per_split


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT); a = ap.parse_args(); out = a.out; os.makedirs(f"{out}/figures", exist_ok=True)
    R = []
    for f in sorted(glob.glob(f"{out}/runs/*.npz")):
        d = parse(f); d["z"] = np.load(f); d["steps"] = d["z"]["steps"]; R.append(d)
    print(len(R), "run-probe-modes")
    # ---- per-run consensus events
    ev_rows = []; per_run = {}
    for d in R:
        for gen in ("res", "tot"):
            cons, _ = run_events(d["z"], gen); per_run[(d["rid"], d["probe"], d["label_mode"], gen)] = cons
            for c in cons: ev_rows.append(dict(run_id=d["rid"], probe=d["probe"], label_mode=d["label_mode"], family=d["family"], N=d["N"], wd=d["wd"], ln=d["ln"], lr=d["lr"], B=d["B"], seed=d["seed"], generator=gen, epoch=10 ** c["loc"] * d["B"] / d["N"] if d["family"] != "phase" else 10 ** c["loc"] * 128.0 / d["N"], **c))
    wcsv(f"{out}/events_run_level.csv", ev_rows)
    # ---- cell-level reproducibility across seeds
    cells = collections.defaultdict(list)
    for d in R:
        if d["probe"] == "test": cells[(d["family"], d["N"], d["wd"], d["ln"], d["lr"], d["B"])].append(d)
    cell_rows = []
    for k, ds in sorted(cells.items()):
        n = len(ds)
        if n < 2: continue
        need = math.ceil(2 * n / 3)
        for gen in ("res", "tot"):
            items = [(c["loc"], i) for i, d in enumerate(ds) for c in per_run[(d["rid"], d["probe"], d["label_mode"], gen)] if True]
            types = collections.defaultdict(list)
            for i, d in enumerate(ds):
                for c in per_run[(d["rid"], d["probe"], d["label_mode"], gen)]: types[c["type"]].append((c["loc"], i))
            for typ, its in types.items():
                for cl in cluster(its, TOL_SEED):
                    sup = len({m[1] for m in cl}); cell_rows.append(dict(family=k[0], N=k[1], wd=k[2], ln=k[3], lr=k[4], B=k[5], generator=gen, type=typ, loc=float(np.median([m[0] for m in cl])), seeds_with_event=sup, n_seeds=n, reproducible=int(sup >= need)))
    wcsv(f"{out}/events_cell_level.csv", cell_rows)
    # ---- band summaries per run (Lres, test or train given)
    bands = [(1, 10), (10, 100), (100, 1000), (1000, 10 ** 9)]; BN = ["b1_10", "b10_100", "b100_1000", "b1000_up"]; rows = []
    for d in R:
        z, st = d["z"], d["steps"]; s1 = st[1:]; row = dict(run_id=d["rid"], probe=d["probe"], label_mode=d["label_mode"], family=d["family"], N=d["N"], wd=d["wd"], ln=d["ln"], lr=d["lr"], B=d["B"], nu=d["lr"] / d["B"], seed=d["seed"], min_class=int(z["min_class"]))
        for gen in ("res", "tot"):
            for m in ("abscissa_med", "n_exp", "n_cplx", "gap12", "logcond", "jump", "trace_over_d"):
                x = np.nanmean(z[f"{m}_{gen}"], 0)[:len(s1)]
                for (lo, hi), bn in zip(bands, BN):
                    k = (s1 >= lo) & (s1 < hi); row[f"{m}_{gen}_{bn}"] = float(np.nanmedian(x[k])) if k.any() else np.nan
                if d["family"] in ("sgdlr", "lrB"):
                    t = d["lr"] * s1
                    for (lo, hi), bn in zip([(10, 100), (100, 1000)], ["t10_100", "t100_1000"]):
                        k = (t >= lo) & (t < hi); row[f"{m}_{gen}_{bn}"] = float(np.nanmedian(x[k])) if k.any() else np.nan
        for k_ in ("abscissa", "n_exp", "n_cplx"):
            x = np.nanmean(z[f"glob_{k_}"], 0)[:len(s1)]
            for (lo, hi), bn in zip(bands, BN): kk = (s1 >= lo) & (s1 < hi); row[f"glob_{k_}_{bn}"] = float(np.nanmedian(x[kk])) if kk.any() else np.nan
        row["n_events_res"] = len(per_run[(d["rid"], d["probe"], d["label_mode"], "res")]); row["n_events_tot"] = len(per_run[(d["rid"], d["probe"], d["label_mode"], "tot")]); rows.append(row)
    wcsv(f"{out}/summary_runs.csv", rows)
    # ---- spike relation (sl40k): snapshot-based spike snapshots + dense step-level spikes
    spike_rows = []
    try:
        import pickle
        epi = list(csv.DictReader(open(EPIS))); dense = {}
        for e in epi:
            if float(e["peak_loss"]) >= 0.05: dense.setdefault(e["run_id"], []).append(int(e["start"]))
    except Exception: dense = {}
    for d in R:
        if d["family"] != "sl40k" or d["probe"] != "test": continue
        p = f"{PHC}/{d['rid']}.pkl"
        if not os.path.isfile(p): continue
        r = pickle.load(open(p, "rb")); evs = [e["step"] for e in r.get("events", [])]; st = d["steps"]; loc = log_mid(st); jump = np.nanmean(d["z"]["jump_res"], 0)[:len(loc)]; mask = np.zeros(len(loc), bool)
        for s in evs:
            k = np.where(st[1:] == s)[0]
            if len(k): mask[k[0]] = True
        med = np.nanmedian(jump); mad = np.nanmedian(np.abs(jump - med)) * 1.4826 + 1e-12
        cons = per_run[(d["rid"], "test", "given", "res")]; jl = [c["loc"] for c in cons if c["type"] == "jump"]; allev = [c["loc"] for c in cons]
        near = lambda l, sset: any(abs(l - ls) <= 0.1 for ls in sset); spl = [math.log10(max(s, 1) + 1) for s in evs]
        spike_rows.append(dict(run_id=d["rid"], N=d["N"], wd=d["wd"], seed=d["seed"], n_snapshot_spikes=len(evs), jump_z_at_spike_intervals=float(np.nanmean((jump[mask] - med) / mad)) if mask.any() else np.nan, jump_z_other=float(np.nanmean((jump[~mask] - med) / mad)),
                               n_consensus_events=len(cons), n_events_within_0p1dec_of_spike=int(sum(near(l, spl) for l in allev)), n_jump_events=len(jl)))
        if d["rid"] in dense:   # step-level spikes (only 3 sl40k runs overlap with the dense block)
            ks = []
            for s0 in dense[d["rid"]]:
                k = np.where((st[:-1] < s0) & (st[1:] >= s0))[0]
                if len(k): ks.append(k[0])
            if ks: spike_rows[-1].update(dense_step_spikes=len(dense[d["rid"]]), dense_jump_z_at_spike_intervals=float(np.nanmean((jump[ks] - med) / mad)))
    # dense runs are named eps*: map to the identical sl40k runs
    for e_id, starts in dense.items():
        m = re.search(r"_s(\d+)_eps1e-8_n(\d+)_wd([\w.-]+)$", e_id)
        if not m: continue
        sid = f"mnist_probe_none_d16_s{m.group(1)}_sl40k_n{m.group(2)}_wd{m.group(3)}"; dd = next((x for x in R if x["rid"] == sid and x["probe"] == "test"), None)
        if dd is None: continue
        st = dd["steps"]; jump = np.nanmean(dd["z"]["jump_res"], 0)[:len(st) - 1]; med = np.nanmedian(jump); mad = np.nanmedian(np.abs(jump - med)) * 1.4826 + 1e-12; ks = []
        for s0 in starts:
            k = np.where((st[:-1] < s0) & (st[1:] >= s0))[0]
            if len(k): ks.append(k[0])
        for r_ in spike_rows:
            if r_["run_id"] == sid: r_.update(dense_step_spikes=len(starts), dense_jump_z_at_spike_intervals=float(np.nanmean((jump[ks] - med) / mad)) if ks else np.nan)
    wcsv(f"{out}/spike_relation_sl40k.csv", spike_rows)
    # ---- field flip relation (gap train probes, given vs true generators)
    flip_rows = []; dec = {}
    if os.path.isfile(DEC):
        for r in csv.DictReader(open(DEC)):
            if r["probe"] == "train" and r["family"] == "gap": dec[r["run_id"]] = r
    for d in R:
        if d["family"] != "gap" or d["probe"] != "train" or d["label_mode"] != "given" or d["rid"] not in dec: continue
        dt = next((x for x in R if x["rid"] == d["rid"] and x["probe"] == "train" and x["label_mode"] == "true"), None)
        if dt is None: continue
        st = d["steps"]; loc = log_mid(st); fs = float(dec[d["rid"]]["mis_f1_cross_step"]) if dec[d["rid"]]["mis_f1_cross_step"] not in ("", "nan") else np.nan
        if not np.isfinite(fs): continue
        flip = math.log10(fs + 1)
        series = {"abscissa_med": np.nanmean(d["z"]["abscissa_med_res"], 0), "n_exp": np.nanmean(d["z"]["n_exp_res"], 0), "jump": np.nanmean(d["z"]["jump_res"], 0),
                  "given_minus_true_abscissa": np.nanmean(np.abs(d["z"]["abscissa_cls_res"][None] if False else 0 * d["z"]["abscissa_med_res"]), 0)}
        ag, at = d["z"]["abscissa_cls_res"], dt["z"]["abscissa_cls_res"]; series["given_minus_true_abscissa"] = np.nanmedian(np.abs(ag - at), 1)
        row = dict(run_id=d["rid"], N=d["N"], ln=d["ln"], seed=d["seed"], flip_log10_step=flip)
        for name, x in series.items():
            x = x[:len(loc)]; win = np.abs(loc - flip) <= 0.25; base = []
            rng = np.random.RandomState(0)
            for _ in range(500):
                c0 = rng.uniform(loc.min() + 0.25, loc.max() - 0.25); w_ = np.abs(loc - c0) <= 0.25
                if w_.sum() >= 3: base.append(np.nanmean(np.abs(np.diff(x[w_])))) if w_.sum() > 1 else None
            stat = np.nanmean(np.abs(np.diff(x[win]))) if win.sum() > 1 else np.nan
            row[f"{name}_window_change"] = float(stat); row[f"{name}_pvalue"] = float((np.sum(np.array(base) >= stat) + 1) / (len(base) + 1)) if base and np.isfinite(stat) else np.nan
            pre = x[(loc < flip) & (loc > flip - 0.4)]; post = x[(loc > flip) & (loc < flip + 0.4)]; row[f"{name}_post_minus_pre"] = float(np.nanmedian(post) - np.nanmedian(pre)) if len(pre) and len(post) else np.nan
        flip_rows.append(row)
    wcsv(f"{out}/field_flip_relation.csv", flip_rows)
    try: figures(out, R, per_run, cells, rows, ev_rows)
    except Exception:
        import traceback; msg = traceback.format_exc(); print("FIGURE FAILED:", msg, file=sys.stderr); open(f"{out}/figures_failed.txt", "w").write(msg)
    print("done")


def figures(out, R, per_run, cells, rows, ev_rows):
    fd = f"{out}/figures"; ph = [d for d in R if d["family"] == "phase" and d["probe"] == "test"]
    if os.path.exists(f"{out}/figures_failed.txt"): os.remove(f"{out}/figures_failed.txt")
    sel = [(250, 0.0), (1000, 0.0), (4000, 0.0), (4000, 1e-2), (60000, 0.0), (60000, 3e-2)]
    if ph:
        fig, ax = plt.subplots(2, 3, figsize=(17, 8)); ax = ax.ravel()
        for a_, (key, lab, sc) in zip(ax, (("abscissa_med_res", "class-median spectral abscissa of Lres (per step)", "symlog"), ("n_exp_res", "expanding directions (mean over classes)", None), ("n_cplx_res", "complex pairs (mean over classes)", None),
                                           ("gap12_res", "gap Re l1 - Re l2", "symlog"), ("logcond_res", "log cond(eigvec) (non-normality)", None), ("jump_res", "matched-eigenvalue jump", None))):
            for (N, wd) in sel:
                ss = [d for d in ph if d["N"] == N and d["wd"] == wd]
                if not ss: continue
                arr = np.stack([np.nanmean(d["z"][key], 0) for d in ss]); a_.plot(ss[0]["steps"][1:], np.nanmedian(arr, 0)[:len(ss[0]["steps"]) - 1], label=f"N{N} wd{wd:g}")
            a_.set_xscale("log"); a_.set_title(key, fontsize=8)
            if sc == "symlog": a_.set_yscale("symlog", linthresh=1e-6)
        ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{fd}/spectra_series.png", dpi=60); plt.close(fig)
        r0 = next((d for d in ph if d["N"] == 1000 and d["wd"] == 0.0 and d["seed"] == 0), ph[0]); z = r0["z"]; st = r0["steps"][1:]
        fig, ax = plt.subplots(1, 2, figsize=(14, 5)); tr, ti = z["traj_res_re"], z["traj_res_im"]
        for c in range(0, tr.shape[1], 3):
            for j in range(tr.shape[2]): ax[0].plot(st, tr[1:, c, j] if len(tr) > len(st) else tr[:, c, j], lw=.4, c=f"C{c % 10}")
        ax[0].set_xscale("log"); ax[0].set_yscale("symlog", linthresh=1e-5); ax[0].set_title(f"matched Re(eigenvalues) of Lres, classes 0,3,6,9 ({r0['rid'][-24:]})", fontsize=8)
        for c in range(0, tr.shape[1], 3):
            for j in range(tr.shape[2]): ax[1].plot(st, np.abs(ti[1:, c, j] if len(ti) > len(st) else ti[:, c, j]), lw=.4, c=f"C{c % 10}")
        ax[1].set_xscale("log"); ax[1].set_yscale("symlog", linthresh=1e-5); ax[1].set_title("|Im| of the same eigenvalues", fontsize=8); fig.tight_layout(); fig.savefig(f"{fd}/eigenvalue_trajectories.png", dpi=60); plt.close(fig)
    ev = [e for e in ev_rows if e["probe"] == "test" and e["generator"] == "res"]
    if ev:
        types = ["abscissa_cross", "nexp_shift", "ncplx_shift", "jump"]; fig, ax = plt.subplots(2, 2, figsize=(14, 9)); ax = ax.ravel()
        for a_, t in zip(ax, types):
            for fam, mk in (("phase", "o"), ("sl40k", "s"), ("gap", "^")):
                es = [e for e in ev if e["type"] == t and e["family"] == fam]
                a_.scatter([np.log10(e["N"]) + 0.03 * (e["seed"] - 1) for e in es], [e["loc"] for e in es], c=[np.log10(e["wd"] + 1e-7) if fam != "gap" else np.log10(1e-7 + e["ln"]) for e in es], marker=mk, s=14, label=fam)
            a_.set_title(f"{t}: log10(step) of run-level consensus events (>=4/5 splits)", fontsize=8); a_.set_xlabel("log10 N")
        ax[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{fd}/events_scatter.png", dpi=60); plt.close(fig)
    print("figures done")


if __name__ == "__main__": main()
