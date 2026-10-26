"""Analysis of the dense optimizer-state block (12 Adam reruns, per-step scalars + bf16 vector dumps). CPU only, <=8 threads, lazy loading.

Usage: python dense_analysis.py [--stage scalars|vectors|all]
Inputs : dynamics_v2/dense/runs/<id>/{dense_scalars.npy, dense_scalars_cols.json, dense_meta.json, config.json}; dynamics_v2/dense/runlist.csv
         /data2/lbastian/latent-dynamics/opt_dense/<id>/{c_dump.npz, vhat_bf16.npz}  (read only; bf16 decoded one step at a time)
Outputs: dynamics_v2/dense/analysis/{episodes.csv, sanity.csv, pairs_*.csv, vector_spike.csv, criterion_*.csv, type_table.csv, figures/*.png}  (never /data2)
Definitions: step-level spike episode = steps with minibatch loss >= 10x median of previous 50 AND ||g'|| >= 5x median of previous 50, clustered with gap < 100.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import argparse, json, csv, glob, collections
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import rankdata

DENSE = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense"
RUNS = f"{DENSE}/runs"; XDUMP = "/data2/lbastian/latent-dynamics/opt_dense"; OUT = f"{DENSE}/analysis"
SLICES = {"fc1": (0, 200960), "fc2": (200960, 217408), "fc3": (217408, 218448), "probe": (218448, 218618)}
SCOPES = ["global", "fc1", "fc2", "fc3", "probe"]
P = 218618


def dec(u): return (u.astype(np.uint32) << 16).view(np.float32)


def load_list():
    rl = {r["run_id"]: r for r in csv.DictReader(open(f"{DENSE}/runlist.csv"))}
    out = []
    for rid, r in sorted(rl.items(), key=lambda kv: int(kv[1]["idx"])):
        d = f"{RUNS}/{rid}"
        meta = json.load(open(f"{d}/dense_meta.json")); cols = json.load(open(f"{d}/dense_scalars_cols.json")); sc = np.load(f"{d}/dense_scalars.npy")
        cfg = json.load(open(f"{d}/config.json"))
        out.append(dict(rid=rid, idx=int(r["idx"]), eps=float(r["eps"]), N=int(r["N"]), wd=float(r["wd"]), seed=int(r["seed"]), role=r["role"], pair=r["pair"], onset=r["existing_onset_step"],
                        prev=int(r["prev_snapshot_step"]), meta=meta, ci={c: j for j, c in enumerate(cols)}, sc=sc, lr=float(cfg["lr"]), type="N" if float(r["wd"]) == 0 else "F",
                        snap_steps=np.array(sorted(int(os.path.basename(f)[5:12]) for f in glob.glob(f"{d}/latents/step_*.npz")))))
    return out


def col(r, name): return r["sc"][:, r["ci"][name]]


def med50(x, t): return np.median(x[max(0, t - 50):t])


def episodes(r, lf=10.0, gf=5.0, gap=100):
    loss, gp = col(r, "loss"), col(r, "gprime_norm"); hits = [t for t in range(50, len(loss)) if loss[t] >= lf * med50(loss, t) and gp[t] >= gf * med50(gp, t)]
    eps_ = []
    for t in hits:
        if eps_ and t - eps_[-1]["last"] < gap: eps_[-1]["last"] = t
        else: eps_.append(dict(first=t, last=t))
    for e in eps_:
        a, b = e["first"], min(e["last"] + 60, len(loss) - 1); base = med50(loss, a); k = a + int(np.argmax(loss[a:b + 1]))
        e.update(start=a + 1, peak_step=k + 1, peak_loss=float(loss[k]), base_loss=float(base), peak_ratio=float(loss[k] / base), peak_gnorm=float(gp[k:k + 1].max()),
                 dur_ge3x=int((loss[a:b + 1] >= 3 * base).sum()))
    return eps_


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


# ====================================================================== scalars stage
def stage_scalars(L):
    os.makedirs(f"{OUT}/figures", exist_ok=True); sanity = []; ep_rows = []
    for r in L:
        eps_ = episodes(r); r["episodes"] = eps_; ts = r["meta"].get("trig_step"); loss, gp = col(r, "loss"), col(r, "gprime_norm")
        on = None if r["onset"] == "none" else int(r["onset"])
        ge = [e for e in eps_ if e["peak_loss"] >= 0.05]  # large spikes (loss reaches >= 0.05)
        hit = [int(s) for s in r["snap_steps"] if any(e["start"] <= s <= e["peak_step"] + 200 for e in ge)]
        ratios_all = max(loss[t] / med50(loss, t) for t in range(50, len(loss))); gr_all = max(gp[t] / med50(gp, t) for t in range(50, len(gp)))
        sanity.append(dict(idx=r["idx"], run_id=r["rid"], eps=r["eps"], N=r["N"], wd=r["wd"], seed=r["seed"], mode=r["meta"]["mode"], trig_step=ts, snapshot_onset=on, prev_snapshot=r["prev"],
                           trig_in_prev_onset=int(ts is not None and ts > 0 and on is not None and r["prev"] <= ts <= on), n_strong_episodes=len(eps_), n_big_episodes=len(ge),
                           max_loss_ratio_full_run=ratios_all, max_gnorm_ratio_full_run=gr_all, big_episode_starts=";".join(str(e["start"]) for e in ge),
                           snapshots_inside_big_episodes=len(hit), snapshot_label="event" if on else "none", step_level_label="spike" if ge else "none"))
        for e in eps_: ep_rows.append(dict(idx=r["idx"], run_id=r["rid"], type=r["type"], eps=r["eps"], N=r["N"], wd=r["wd"], **{k: e[k] for k in ("start", "peak_step", "peak_loss", "base_loss", "peak_ratio", "peak_gnorm", "dur_ge3x")}))
    wcsv(f"{OUT}/sanity.csv", sanity); wcsv(f"{OUT}/episodes.csv", ep_rows)
    # full-run overview figure
    fig, ax = plt.subplots(3, 4, figsize=(20, 11)); ax = ax.ravel()
    for a, r in zip(ax, L):
        st = np.arange(1, len(col(r, "loss")) + 1); a.plot(st, col(r, "loss"), lw=.4); a.set_xscale("log"); a.set_yscale("log"); a2 = a.twinx()
        a2.plot(st, col(r, "global/frac_sqrtv_lt_eps"), c="C1", lw=.8); a2.plot(st, col(r, "global/frac_sqrtv_lt_10eps"), c="C2", lw=.6, ls="--"); a2.set_ylim(-.02, 1.02)
        for e in r["episodes"]:
            if e["peak_loss"] >= 0.05: a.axvline(e["start"], c="r", lw=.5, ls=":")
        for s in r["snap_steps"]: pass
        a.set_title(f"#{r['idx']} eps={r['eps']:g} N={r['N']} wd={r['wd']:g} s{r['seed']} ({r['type']}) snapshot label: {'event' if r['onset']!='none' else 'none'}", fontsize=7)
    fig.suptitle("minibatch loss (blue, per step), global frac(sqrt vhat < eps) orange, <10 eps green dashed; red dotted = big spike episodes", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/overview_loss_fraclow.png", dpi=55); plt.close(fig)
    return sanity


def pair_tables(L):
    by = {r["idx"]: r for r in L}; rows = []
    lags = (-400, -200, -100, -50, -20, -10, -1)
    cols = ["loss", "gprime_norm"] + [f"{s}/{c}" for s in SCOPES for c in ("frac_sqrtv_lt_eps", "frac_sqrtv_lt_10eps", "vhat_q0.0001", "vhat_q0.001", "vhat_q0.01", "max_a")]
    pairs = [("B (type N)", 3, 4), ("A-control (type N)", 1, 5), ("A2-control (type N)", 2, 5), ("D (type F)", 8, 9), ("E (type F)", 11, 10)]
    for name, ie, ip in pairs:
        e, p = by[ie], by[ip]; t0 = e["meta"].get("trig_step"); t0 = t0 if t0 and t0 > 0 else None
        for lag in lags:
            s = t0 + lag
            for c in cols:
                ve = col(e, c)[s - 1]; vp = col(p, c)[s - 1]
                rows.append(dict(pair=name, event_run=e["idx"], partner=p["idx"], t_trig=t0, lag=lag, column=c, event=float(ve), partner_val=float(vp)))
    wcsv(f"{OUT}/pairs_matched_steps.csv", rows)
    # figure: pair time series around trigger of the event run (shared absolute steps)
    fig, ax = plt.subplots(len(pairs), 5, figsize=(20, 3.2 * len(pairs)), squeeze=False)
    for i, (name, ie, ip) in enumerate(pairs):
        e, p = by[ie], by[ip]; t0 = e["meta"]["trig_step"]; sl = np.arange(t0 - 400, t0 + 251)
        for j, (c, lab) in enumerate((("loss", "loss"), ("global/frac_sqrtv_lt_eps", "frac sqrt(vhat)<eps"), ("global/frac_sqrtv_lt_10eps", "frac <10 eps"), ("global/vhat_q0.01", "vhat 1% quantile"), ("global/max_a", "max amplification"))):
            for r, cl in ((e, "C3"), (p, "C0")):
                ax[i, j].plot(sl - t0, col(r, c)[sl - 1], c=cl, lw=.8, label=f"#{r['idx']}")
            if j in (0, 3, 4): ax[i, j].set_yscale("log")
            ax[i, j].set_title(f"{name}: {lab}", fontsize=7)
        ax[i, 0].legend(fontsize=6)
    fig.suptitle("event run (red) vs partner (blue) at the same absolute steps, x = step - trigger step of the event run", fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/pairs_timeseries.png", dpi=55); plt.close(fig)
    return rows


def timing_table(L):
    rows = []
    for r in L:
        ts = r["meta"].get("trig_step")
        if not ts or ts <= 0: continue
        loss, gp = col(r, "loss"), col(r, "gprime_norm")
        def first(arr, fac, lo, hi): return next((t + 1 for t in range(lo, hi) if arr[t] >= fac * med50(arr, t)), None)
        lo, hi = ts - 200, min(ts + 251, len(loss))
        d = dict(idx=r["idx"], run_id=r["rid"], type=r["type"], eps=r["eps"], N=r["N"], wd=r["wd"], trig=ts, g5=first(gp, 5, lo, hi), g100=first(gp, 100, ts - 1, hi), loss3=first(loss, 3, lo, hi), loss10=first(loss, 10, ts - 1, hi),
                 loss100=first(loss, 100, ts - 1, hi), peak=ts + int(np.argmax(loss[ts - 1:hi])) if hi > ts else None)
        d["lag_trig_to_loss10"] = d["loss10"] - ts if d["loss10"] else None; d["lag_trig_to_loss100"] = d["loss100"] - ts if d["loss100"] else None; d["lag_trig_to_peak"] = d["peak"] - ts if d["peak"] else None
        fe = col(r, "global/frac_sqrtv_lt_eps"); f10 = col(r, "global/frac_sqrtv_lt_10eps")
        for nm, arr in (("frac_lt_eps", fe), ("frac_lt_10eps", f10)):
            above = np.where(arr >= 0.5)[0]; d[f"{nm}_first_ge_0.5"] = int(above[0]) + 1 if len(above) else None
            d[f"{nm}_at_trig_minus_1"] = float(arr[ts - 2])
        d["lag_trig_minus_collapse_frac10eps"] = ts - d["frac_lt_10eps_first_ge_0.5"] if d["frac_lt_10eps_first_ge_0.5"] else None
        rows.append(d)
    wcsv(f"{OUT}/timing.csv", rows); return rows


# ====================================================================== vectors stage
def window_stats(r, z):
    """per dumped step statistics from the full vectors; returns dict of arrays and the step list."""
    st = z["steps"]; V, M, G = z["vhat_u16"], z["mhat_u16"], z["gp_u16"]; eps, lr = r["eps"], r["lr"]
    out = collections.defaultdict(list)
    for k in range(len(st)):
        v = dec(V[k]); m = dec(M[k]); g = dec(G[k]); sv = np.sqrt(v); den = sv + eps; dth = lr * m / den
        g2 = g.astype(np.float64) ** 2; gs = g2.sum() + 1e-300; d2 = dth.astype(np.float64) ** 2; ds = d2.sum() + 1e-300
        for nm, thr in (("1", eps), ("10", 10 * eps)):
            low = sv < thr; out[f"frac_low{nm}"].append(low.mean()); out[f"gshare_low{nm}"].append(g2[low].sum() / gs); out[f"dshare_low{nm}"].append(d2[low].sum() / ds)
        out["frac_dead"].append((v == 0).mean()); out["max_a"].append(float((lr / den).max())); out["dnorm"].append(float(np.sqrt(ds))); out["max_dth_lr"].append(float(np.abs(dth).max() / lr))
        out["gnorm"].append(float(np.sqrt(gs))); out["sv_med"].append(float(np.median(sv))); out["sv_q01"].append(float(np.quantile(sv, 0.01)))
        for s, (a, b) in SLICES.items():
            out[f"gshare_{s}"].append(g2[a:b].sum() / gs); out[f"dshare_{s}"].append(d2[a:b].sum() / ds)
    return st, {k: np.array(v) for k, v in out.items()}


def spike_vector_analysis(r, z, st, ws):  # ws["dnorm"] per dumped step
    """Delta-theta at the step of largest update within the window after the trigger; its energy set vs the low-vhat set."""
    ts = int(z["trig_step"]); eps, lr = r["eps"], r["lr"]; post = np.where(st >= ts)[0]
    k = post[int(np.argmax(ws["dnorm"][post]))]; s_pk = int(st[k]); kp = k - 1 if k > 0 else k
    v1 = dec(z["vhat_u16"][kp]); sv1 = np.sqrt(v1); m = dec(z["mhat_u16"][k]); sv = np.sqrt(dec(z["vhat_u16"][k])); dth = lr * m / (sv + eps); g = dec(z["gp_u16"][k])
    e = dth.astype(np.float64) ** 2; tot = e.sum(); order = np.argsort(-e); cum = np.cumsum(e[order]) / tot; k50 = int(np.searchsorted(cum, 0.5)) + 1; k90 = int(np.searchsorted(cum, 0.9)) + 1
    S = np.zeros(P, bool); S[order[:k50]] = True; out = dict(idx=r["idx"], run_id=r["rid"], type=r["type"], eps=r["eps"], wd=r["wd"], N=r["N"], trig=ts, spike_step=s_pk, dnorm=float(np.sqrt(tot)), max_dth_over_lr=float(np.abs(dth).max() / lr),
                                                           k50=k50, frac50=k50 / P, k90=k90, frac90=k90 / P)
    for s, (a, b) in SLICES.items():
        out[f"S50_share_{s}"] = float(S[a:b].sum() / k50); out[f"size_share_{s}"] = (b - a) / P; out[f"energy_share_{s}"] = float(e[a:b].sum() / tot)
    # low-vhat sets before the spike (step before s_pk)
    for nm, thr in (("L1", eps), ("L10", 10 * eps)):
        Lm = sv1 < thr; out[f"{nm}_frac"] = float(Lm.mean()); out[f"S50_in_{nm}"] = float((S & Lm).sum() / k50); out[f"enrich_{nm}"] = float((S & Lm).sum() / k50 / max(Lm.mean(), 1e-12)) if Lm.any() else np.nan
        out[f"energy_in_{nm}"] = float(e[Lm].sum() / tot)
    Lrel = sv1 <= np.quantile(sv1, 0.10); out["S50_in_bottom10pct_sv"] = float((S & Lrel).sum() / k50); out["enrich_bottom10pct"] = out["S50_in_bottom10pct_sv"] / 0.10
    kt = int(np.where(st == ts - 1)[0][0]) if (st == ts - 1).any() else max(k - 1, 0); svt = np.sqrt(dec(z["vhat_u16"][kt]))   # state just before the trigger step (before the ramp)
    for nm, thr in (("L1", eps), ("L10", 10 * eps)):
        Lm = svt < thr; out[f"pre_{nm}_frac"] = float(Lm.mean()); out[f"pre_S50_in_{nm}"] = float((S & Lm).sum() / k50); out[f"pre_enrich_{nm}"] = float((S & Lm).sum() / k50 / max(Lm.mean(), 1e-12)) if Lm.any() else np.nan
        out[f"pre_energy_in_{nm}"] = float(e[Lm].sum() / tot)
    Lrel = svt <= np.quantile(svt, 0.10); out["pre_enrich_bottom10pct"] = float((S & Lrel).sum() / k50 / 0.10)
    ok = svt > 0; out["pre_spearman_absdth_vs_sv"] = float(np.corrcoef(rankdata(np.abs(dth[ok])), rankdata(svt[ok]))[0, 1])
    ok = sv1 > 0; out["spearman_absdth_vs_sv"] = float(np.corrcoef(rankdata(np.abs(dth[ok])), rankdata(sv1[ok]))[0, 1])
    out["spearman_absdth_vs_absg"] = float(np.corrcoef(rankdata(np.abs(dth)), rankdata(np.abs(g)))[0, 1])
    out["gshare_S50"] = float((g.astype(np.float64) ** 2)[S].sum() / ((g.astype(np.float64) ** 2).sum() + 1e-300))
    # fc1 hidden-unit / pixel concentration of the update energy
    w1 = e[:200704].reshape(256, 784); unit = w1.sum(1) + e[200704:200960]; us = np.sort(unit)[::-1] / unit.sum(); out["fc1_units_for_50pct_energy"] = int(np.searchsorted(np.cumsum(us), 0.5)) + 1
    out["fc1_top10units_share"] = float(us[:10].sum()); pix = w1.sum(0); ps = np.sort(pix)[::-1] / pix.sum(); out["fc1_pixels_for_50pct_energy"] = int(np.searchsorted(np.cumsum(ps), 0.5)) + 1
    out["fc1_top_unit"] = int(np.argmax(unit)); out["fc1_top_unit_share"] = float(us[0])
    uS = np.zeros(256); uS += S[:200704].reshape(256, 784).sum(1); out["S50_fc1_units_with_any"] = int((uS > 0).sum())
    return out, dict(unit=unit, pix=pix.reshape(28, 28), layer_e={s: float(e[a:b].sum() / tot) for s, (a, b) in SLICES.items()})


def stage_vectors(L):
    os.makedirs(f"{OUT}/figures", exist_ok=True); spikes = []; crit = []; details = {}
    for r in L:
        path = f"{XDUMP}/{r['rid']}/c_dump.npz"; z = np.load(path)
        st, ws = window_stats(r, z); ts = int(z["trig_step"]); r["ws"], r["ws_steps"] = ws, st
        for k, s in enumerate(st):
            crit.append(dict(idx=r["idx"], run_id=r["rid"], type=r["type"], mode=str(z["mode"]), step=int(s), rel_to_trig=int(s) - ts if ts > 0 else None, **{n: float(ws[n][k]) for n in ws}))
        if ts > 0:
            sp, det = spike_vector_analysis(r, z, st, ws); spikes.append(sp); details[r["idx"]] = det
        z.close()
        print("vectors done", r["idx"], flush=True)
    wcsv(f"{OUT}/vector_spike.csv", spikes); wcsv(f"{OUT}/criterion_per_step.csv", crit)
    return spikes, details, crit


def stage_figs_vectors(L, spikes, details, crit):
    n = len(details)
    if n:
        fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 6.5), squeeze=False)
        for j, (idx, det) in enumerate(sorted(details.items())):
            r = next(x for x in L if x["idx"] == idx); ax[0, j].imshow(det["pix"], cmap="magma"); ax[0, j].set_title(f"#{idx} {r['type']} eps{r['eps']:g}\nfc1 update energy by input pixel", fontsize=7)
            u = np.sort(det["unit"])[::-1] / det["unit"].sum(); ax[1, j].bar(range(40), u[:40]); ax[1, j].set_title("top-40 fc1 units' share of update energy", fontsize=7)
        fig.tight_layout(); fig.savefig(f"{OUT}/figures/spike_update_concentration.png", dpi=55); plt.close(fig)
    ids = sorted({c["idx"] for c in crit}); fig, ax = plt.subplots(3, 4, figsize=(20, 11)); ax = ax.ravel()
    for a, idx in zip(ax, ids):
        rows = [c for c in crit if c["idx"] == idx]; t = np.array([c["step"] for c in rows]); r = next(x for x in L if x["idx"] == idx)
        a.plot(t, [c["frac_low10"] for c in rows], label="frac<10eps"); a.plot(t, [c["gshare_low10"] for c in rows], label="grad share (<10eps set)"); a.plot(t, [c["frac_low10"] * c["gshare_low10"] for c in rows], label="product", c="k", lw=1)
        a.plot(t, [c["frac_low1"] * c["gshare_low1"] for c in rows], label="product (<eps)", ls="--"); ts = r["meta"].get("trig_step")
        if ts and ts > 0: a.axvline(ts, c="r", ls=":")
        a.set_title(f"#{idx} eps{r['eps']:g} N{r['N']} wd{r['wd']:g} ({r['type']})", fontsize=8); a.set_ylim(-.02, 1.02)
    ax[0].legend(fontsize=6); fig.tight_layout(); fig.savefig(f"{OUT}/figures/criterion_timeseries.png", dpi=55); plt.close(fig)


def type_table(L, spikes, crit):
    rows = []
    for r in L:
        ts = r["meta"].get("trig_step")
        if not ts or ts <= 0: continue
        j = ts - 2; d = dict(idx=r["idx"], type=r["type"], eps=r["eps"], N=r["N"], wd=r["wd"], loss_before=float(col(r, "loss")[j]), gprime_before=float(col(r, "gprime_norm")[j]))
        for s in ("global", "fc1"):
            for c in ("frac_sqrtv_lt_eps", "frac_sqrtv_lt_10eps", "vhat_q0.01", "vhat_q0.5", "max_a", "r_q0.9999", "max_dtheta_over_lr"): d[f"{s}/{c}"] = float(col(r, f"{s}/{c}")[j])
        d["median_sqrtv_over_eps"] = float(np.sqrt(col(r, "global/vhat_q0.5")[j]) / r["eps"])
        rows.append(d)
    wcsv(f"{OUT}/type_table.csv", rows); return rows


def criterion_eval(L, crit):
    """pre-spike value of C = frac_low x gradient share (low = sqrt(vhat)<10 eps and <eps). Spiking runs: [t*-400, t*-1]; non-spiking fixed-window runs: their whole dumped window;
    fixed-window runs that spiked inside the window (9, 10): steps before their own first big episode (>= start-400)."""
    out = []; trig = {r["idx"]: r["meta"].get("trig_step") for r in L}
    for r in L:
        idx = r["idx"]; rows = [c for c in crit if c["idx"] == idx]; ts = trig[idx]; big = [e for e in r["episodes"] if e["peak_loss"] >= 0.05]
        if ts and ts > 0: rows = [c for c in rows if ts - 400 <= c["step"] < ts]; lab = "pre-trigger"
        elif big and any(rows[0]["step"] <= e["start"] <= rows[-1]["step"] for e in big):
            s0 = min(e["start"] for e in big if rows[0]["step"] <= e["start"] <= rows[-1]["step"]); rows = [c for c in rows if s0 - 400 <= c["step"] < s0]; lab = "pre-own-spike"
        else: lab = "no spike in window"
        if not rows: continue
        d = dict(idx=idx, window=lab, step_lo=rows[0]["step"], step_hi=rows[-1]["step"], n_steps=len(rows), type=r["type"], eps=r["eps"], N=r["N"], wd=r["wd"], any_big_spike=int(bool(big)))
        for nm in ("1", "10"):
            prod = np.array([c[f"frac_low{nm}"] * c[f"gshare_low{nm}"] for c in rows]); d[f"C{nm}_last"] = float(prod[-1]); d[f"C{nm}_max"] = float(prod.max()); d[f"C{nm}_median"] = float(np.median(prod))
            d[f"frac_low{nm}_last"] = float(rows[-1][f"frac_low{nm}"]); d[f"gshare_low{nm}_last"] = float(rows[-1][f"gshare_low{nm}"])
        d["max_dth_lr_max"] = float(max(c["max_dth_lr"] for c in rows)); d["max_a_last"] = float(rows[-1]["max_a"]); d["gnorm_median"] = float(np.median([c["gnorm"] for c in rows])); d["dnorm_median"] = float(np.median([c["dnorm"] for c in rows]))
        out.append(d)
    wcsv(f"{OUT}/criterion_pretrigger.csv", out); return out

def load_crit():
    rows = list(csv.DictReader(open(f"{OUT}/criterion_per_step.csv"))); out = []
    for r in rows:
        d = dict(idx=int(r["idx"]), run_id=r["run_id"], type=r["type"], mode=r["mode"], step=int(r["step"]))
        for k, v in r.items():
            if k not in d and k not in ("rel_to_trig",): d[k] = float(v)
        out.append(d)
    return out


def growth_table(L):
    """Slow precursor: log-slope of minibatch loss and ||g'|| over [s-400, s-20] before the (main) spike start s; non-spiking runs: max slope over any 380-step window after step 5000."""
    rows = []
    def slope(y, a, b):
        t = np.arange(a, b); return float(np.polyfit(t, np.log(np.clip(y[a:b], 1e-30, None)), 1)[0])
    def smooth(y, w=25): return np.convolve(np.log(np.clip(y, 1e-30, None)), np.ones(w) / w, mode="same")
    for r in L:
        loss, gp = col(r, "loss"), col(r, "gprime_norm"); eps_ = [e for e in r["episodes"] if e["peak_loss"] >= 0.05]
        d = dict(idx=r["idx"], type=r["type"], eps=r["eps"], N=r["N"], wd=r["wd"], n_big_episodes=len(eps_))
        if eps_:
            s = eps_[0]["start"]; a, b = s - 400, s - 20
            d.update(first_spike_start=s, loss_slope_pre=slope(loss, a, b), gnorm_slope_pre=slope(gp, a, b), gnorm_ratio_pre=float(np.median(gp[b - 25:b]) / np.median(gp[a:a + 25])), loss_ratio_pre=float(np.median(loss[b - 25:b]) / np.median(loss[a:a + 25])))
            # onset of growth: first step before s where the 50-step running median of ||g'|| exceeds 2x the minimum of the 50-step medians over the previous 2000 steps
            med = np.array([np.median(gp[max(0, t - 50):t + 1]) for t in range(len(gp))]); onset = None
            for t in range(max(2000, 0), s):
                if med[t] >= 2.0 * med[t - 2000:t].min(): onset = t + 1; break
            d["gnorm_growth_onset_2x_after2000"] = onset
        else:
            best = None
            for a in range(5000, len(gp) - 400, 100):
                sl = slope(gp, a, a + 380)
                if best is None or sl > best[0]: best = (sl, a)
            d.update(max_gnorm_slope_any_window=best[0], at=best[1], n_windows=None)
        rows.append(d)
    wcsv(f"{OUT}/growth_precursor.csv", rows); return rows


def stage_post(L):
    os.makedirs(f"{OUT}/figures", exist_ok=True); crit = load_crit(); spikes = []; details = {}
    by = collections.defaultdict(list)
    for c in crit: by[c["idx"]].append(c)
    for r in L:
        ts = r["meta"].get("trig_step")
        if not ts or ts <= 0: continue
        z = np.load(f"{XDUMP}/{r['rid']}/c_dump.npz"); st = z["steps"]; ws = {"dnorm": np.array([c["dnorm"] for c in by[r["idx"]]])}
        sp, det = spike_vector_analysis(r, z, st, ws); spikes.append(sp); details[r["idx"]] = det; z.close()
    wcsv(f"{OUT}/vector_spike.csv", spikes); stage_figs_vectors(L, spikes, details, crit); type_table(L, spikes, crit); criterion_eval(L, crit); growth_table(L)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--stage", default="all"); a = ap.parse_args()
    assert OUT.startswith("/vol/bitbucket/"), "outputs must stay on /vol/bitbucket"
    os.makedirs(OUT, exist_ok=True); L = load_list()
    san = stage_scalars(L)
    for r, s in zip(L, san): r["big"] = s["n_big_episodes"] > 0
    pair_tables(L); timing_table(L)
    if a.stage == "post": stage_post(L)
    if a.stage in ("vectors", "all"):
        spikes, details, crit = stage_vectors(L); stage_figs_vectors(L, spikes, details, crit); type_table(L, spikes, crit); criterion_eval(L, crit)
    print("done")


if __name__ == "__main__": main()
