"""Tables/figures for dense_lag.py outputs. python dense_lag_report.py q1|q2|q4|q5"""
import os, sys, csv, glob
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ[_k] = "4"
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/dense_latent/analysis" + {"mnist": "", "cifar10": "/cifar"}.get(os.environ.get("DL_DATASET", "mnist"), "/fm")
TAGS = ["dl_rn18_n10000_ln0", "dl_rn18_n10000_ln0.2"] if os.environ.get("DL_DATASET") == "cifar10" else ["dl_adam_n1000_wd0", "dl_adam_n4000_wd0", "dl_adam_n4000_wd1e-3", "dl_adam_nfull_wd0", "dl_adam_n4000_ln0.2", "dl_sgd_lr0.025_n4000_wd0"]
CIF = os.environ.get("DL_DATASET") == "cifar10"; T0S = [100, 1000, 3000, 6000] if CIF else [100, 1000, 5000, 15000]
STAGES = {"early(t<50)": (0, 50), "mid(50-2000)": (50, 2000), "late(>=2000)": (2000, 1e9)}


def load_q1(tag):
    rows = list(csv.DictReader(open(f"{OUT}/q13_lag_{tag}.csv"))); out = {}
    for r in rows:
        out.setdefault((r["probe"], r["frame"]), []).append([float(r[k]) for k in ("t", "lag", "snap_equiv", "E_tot", "E_res_aff", "E_res_cls", "E_res_krr", "E_res_knn", "E_floor")])
    return {k: np.array(v) for k, v in out.items()}


def slope(lag, E, lo, hi):
    m = (lag >= lo) & (lag <= hi) & (E > 0)
    if m.sum() < 3: return np.nan
    return np.polyfit(np.log(lag[m]), np.log(E[m]), 1)[0]


def q1():
    fexp = open(f"{OUT}/q1_exponents.csv", "w"); w = csv.writer(fexp)
    w.writerow(["run", "probe", "frame", "stage", "nbase", "lag_range", "slope_Etot", "slope_Eres_aff", "slope_Eres_final", "slope_Eres_final_floorcorr"])
    fst = open(f"{OUT}/q1_stage_table.csv", "w"); w2 = csv.writer(fst)
    w2.writerow(["run", "probe", "frame", "stage", "lag", "nbase", "R1_affine", "cls_share_of_postaff", "unexpl_of_postaff", "unexpl_of_total", "floor_over_Eres", "Eres_final", "Etot"])
    for tag in TAGS:
        D = load_q1(tag)
        for (probe, frame), A in D.items():
            t, lag, snap, Et, Ea, Ec, Ek, Ekn, Fl = A.T
            for sname, (lo_t, hi_t) in STAGES.items():
                bases = np.unique(t[(t >= lo_t) & (t < hi_t)])
                for lr in ((1, 10), (10, 100), (100, 1000), (1, 100)):
                    sl = {"Etot": [], "Eaff": [], "Ek": [], "Ekc": []}
                    for b in bases:
                        m = (t == b) & (snap == 0) & (lag <= max(b, 1)) & (lag <= 1000)       # lags up to the base time
                        for key, E in (("Etot", Et), ("Eaff", Ea), ("Ek", Ek), ("Ekc", Ek - Fl)):
                            ok = m & (Ek > 2 * Fl) if key == "Ekc" else m
                            sl[key].append(slope(lag[ok], E[ok], *lr))
                    w.writerow([tag, probe, frame, sname, len(bases), f"{lr[0]}-{lr[1]}"] + [f"{np.nanmedian(sl[k]):.2f}" if np.isfinite(sl[k]).any() else "nan" for k in ("Etot", "Eaff", "Ek", "Ekc")])
                for L in (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000):
                    m = (np.isin(t, bases)) & (lag == L) & (snap == 0)
                    if m.sum() == 0: continue
                    w2.writerow([tag, probe, frame, sname, L, m.sum(), f"{np.median(1-Ea[m]/Et[m]):.3f}", f"{np.median(1-Ec[m]/Ea[m]):.3f}", f"{np.median(Ek[m]/Ea[m]):.3f}", f"{np.median(Ek[m]/Et[m]):.4f}", f"{np.median(Fl[m]/Ek[m]):.3g}", f"{np.median(Ek[m]):.3g}", f"{np.median(Et[m]):.3g}"])
                # snapshot-equivalent lag
                m = (np.isin(t, bases)) & (snap == 1)
                if m.sum(): w2.writerow([tag, probe, frame, sname, "snapshot-equiv", m.sum(), f"{np.median(1-Ea[m]/Et[m]):.3f}", f"{np.median(1-Ec[m]/Ea[m]):.3f}", f"{np.median(Ek[m]/Ea[m]):.3f}", f"{np.median(Ek[m]/Et[m]):.4f}", f"{np.median(Fl[m]/Ek[m]):.3g}", f"{np.median(Ek[m]):.3g}", f"{np.median(Et[m]):.3g}"])
    fexp.close(); fst.close()
    # figures
    for probe in ("test",):
        for frame in ("latent", "logit"):
            fig, axs = plt.subplots(2, 6, figsize=(26, 8))
            for j, tag in enumerate(TAGS):
                A = load_q1(tag)[(probe, frame)]; t, lag, snap, Et, Ea, Ec, Ek, Ekn, Fl = A.T; bases = np.unique(t); cm = plt.cm.viridis(np.linspace(0, 1, len(bases)))
                for ci, b in enumerate(bases):
                    m = (t == b) & (snap == 0); o = np.argsort(lag[m])
                    axs[0, j].plot(lag[m][o], (Ek / Ea)[m][o], "-o", ms=2, color=cm[ci], lw=1)
                    axs[1, j].plot(lag[m][o], Ek[m][o], "-o", ms=2, color=cm[ci], lw=1)
                    axs[1, j].plot(lag[m][o], Fl[m][o], ":", color=cm[ci], lw=0.7)
                x = np.array([1, 1000.]); axs[1, j].plot(x, 1e-3 * x, "k--", lw=0.8); axs[1, j].plot(x, 1e-3 * x ** 2 / 100, "r--", lw=0.8)
                axs[0, j].set_xscale("log"); axs[0, j].set_ylim(0, 1.15); axs[0, j].set_title(tag, fontsize=9); axs[0, j].set_xlabel("lag"); axs[0, j].set_ylabel("final residual / post-affine residual")
                axs[1, j].set_xscale("log"); axs[1, j].set_yscale("log"); axs[1, j].set_xlabel("lag"); axs[1, j].set_ylabel("held-out residual energy (dotted: fp16 floor; dashed ref slopes 1 black, 2 red)")
            fig.suptitle(f"{probe} probe, {frame} frame; colour = base step (viridis, early->late)"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_q1_lag_{probe}_{frame}.png", dpi=90); plt.close(fig)


def q4():
    rows = list(csv.DictReader(open(f"{OUT}/q4_autocorr.csv"))); KM = 400
    fo = open(f"{OUT}/q4_autocorr_summary.csv", "w"); w = csv.writer(fo); w.writerow(["run", "probe", "t0", "kind", "E_step", "floor_share", "ac1", "ac1_floorcorr", "ac2", "ac3", "ac5", "ac10", "ac20", "min_lag", "min_val", "first_neg_lag"])
    for r in rows:
        ac = np.array([float(r[f"ac{k}"]) for k in range(1, KM + 1)]); k0 = int(np.argmin(ac[1:])) + 2; zc = next((k + 1 for k in range(1, KM) if ac[k] < 0), -1)
        w.writerow([r["run"], r["probe"], r["t0"], r["kind"], f"{float(r['E_step']):.3g}", r["floor_share"][:6], f"{ac[0]:.3f}", f"{float(r['c1_floor_corrected']):.3f}", f"{ac[1]:.3f}", f"{ac[2]:.3f}", f"{ac[4]:.3f}", f"{ac[9]:.3f}", f"{ac[19]:.3f}", k0, f"{ac[k0-1]:.3f}", zc])
    fo.close()
    for kind in ("raw", "resid_affine"):
        fig, axs = plt.subplots(1, 4, figsize=(22, 4.5), sharey=True)
        for j, t0 in enumerate(T0S):
            for tag, c in zip(TAGS, plt.cm.tab10.colors):
                for r in rows:
                    if r["run"] == tag and r["probe"] == "test" and int(r["t0"]) == t0 and r["kind"] == kind:
                        axs[j].plot(range(1, KM + 1), [float(r[f"ac{k}"]) for k in range(1, KM + 1)], "-o", ms=2, color=c, label=tag, ls="--" if "sgd" in tag else "-")
            axs[j].axhline(0, color="k", lw=0.5); axs[j].set_title(f"window start {t0} ({kind}, test probe)"); axs[j].set_xlabel("lag k (steps)"); axs[j].set_xscale("log")
        axs[0].set_ylabel("<d_t, d_{t+k}>/<d,d>"); axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(f"{OUT}/fig_q4_autocorr_{kind}.png", dpi=90); plt.close(fig)
    P = np.load(f"{OUT}/q4_psd.npz"); fr = P["freqs"]; starts = [200, 1500, 3500, 5800] if CIF else [200, 2000, 8000, 17000]; Ns = {"dl_rn18_n10000_ln0": 10000, "dl_rn18_n10000_ln0.2": 10000, "dl_adam_n1000_wd0": 1000, "dl_adam_n4000_wd0": 4000, "dl_adam_n4000_wd1e-3": 4000, "dl_adam_nfull_wd0": 60000, "dl_adam_n4000_ln0.2": 4000, "dl_sgd_lr0.025_n4000_wd0": 4000}
    fig, axs = plt.subplots(6, 4, figsize=(20, 22)); fp = open(f"{OUT}/q4_psd_peaks.csv", "w"); wp = csv.writer(fp); wp.writerow(["run", "probe", "start", "peak_freqs(cyc/step)[period]", "peak/baseline", "epoch_freq", "white_floor", "psd_median"])
    for i, tag in enumerate(TAGS):
        for j, st in enumerate(starts):
            for probe, ls_ in (("test", "-"), ("train", ":")):
                p = P[f"{tag}|{probe}|{st}"]; fl = float(P[f"{tag}|{probe}|{st}|floor"]) * 2   # displacement noise variance per coord = 2x position rounding variance
                axs[i, j].loglog(fr[1:], p[1:], ls_, lw=1, label=probe)
                if probe == "test":
                    axs[i, j].axhline(fl, color="gray", lw=0.7, ls="--")
                    lp = np.log(p[1:] + 1e-300); k = 9; base = np.exp(np.convolve(np.pad(lp, k, mode="edge"), np.ones(2 * k + 1) / (2 * k + 1), mode="valid")); ratio = p[1:] / base
                    idx = [q for q in range(1, len(ratio) - 1) if ratio[q] > ratio[q - 1] and ratio[q] > ratio[q + 1] and ratio[q] > 2.0]; idx = sorted(idx, key=lambda q: -ratio[q])[:4]
                    wp.writerow([tag, probe, st, " ".join(f"{fr[q+1]:.4f}[{1/fr[q+1]:.1f}]" for q in idx), " ".join(f"{ratio[q]:.1f}" for q in idx), f"{128/Ns[tag]:.4f}", f"{fl:.3g}", f"{np.median(p):.3g}"])
            for f0, c in ((0.1, "r"), (128 / Ns[tag], "g"), (0.001, "m")): axs[i, j].axvline(f0, color=c, lw=0.6, alpha=0.6)
            axs[i, j].set_title(f"{tag} t={st}..{st+2048}", fontsize=8)
    axs[0, 0].legend(); fig.suptitle("Welch PSD of per-step latent velocity (L=512). gray dashed = fp16 white floor; red=1/10, green=1/epoch, magenta=1/1000"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_q4_psd.png", dpi=70); plt.close(fig); fp.close()


def q5():
    rows = list(csv.DictReader(open(f"{OUT}/q5_spikes.csv")))
    for tag in TAGS:
        for k in (0, 1):
            f = f"{OUT}/q5_spike_{tag}_{k}.npz"
            if not os.path.exists(f): continue
            d = np.load(f); t = d["t"]; fig, axs = plt.subplots(5, 1, figsize=(10, 14), sharex=True)
            axs[0].semilogy(t, d["loss"], "k", label="mean train-probe loss"); axs[0].semilogy(t, d["p99"], "gray", label="p99"); axs[0].legend(); axs[0].set_title(f"{tag} spike onset {int(d['onset'])} peak {int(d['peak'])}")
            for fr, ls_ in (("latent", "-"), ("logit", "--")):
                axs[1].semilogy(t, d[f"test_{fr}_sst"], ls_, label=f"E_tot test {fr}"); axs[1].semilogy(t, d[f"test_{fr}_floor"], ":", label=f"fp16 floor {fr}")
                axs[2].plot(t, 1 - d[f"test_{fr}_sse1"] / d[f"test_{fr}_sst"], ls_, label=f"R1 affine {fr}")
                axs[3].plot(t, d[f"test_{fr}_sse3"] / d[f"test_{fr}_sse1"], ls_, label=f"unexplained/post-aff {fr}"); axs[3].plot(t, 1 - d[f"test_{fr}_sse2"] / d[f"test_{fr}_sse1"], ls_, alpha=0.5, label=f"class share {fr}")
            axs[1].legend(fontsize=7); axs[2].legend(fontsize=7); axs[3].legend(fontsize=7); axs[2].set_ylim(-0.2, 1.05); axs[3].set_ylim(-0.1, 1.2)
            axs[4].plot(t, d["A_smax"], label="s_max A_t (train probe)"); axs[4].plot(t, d["A_smin"], label="s_min A_t"); ax2 = axs[4].twinx(); ax2.plot(t, d["Wnorm"], "r", label="|W|"); axs[4].legend(loc="upper left"); ax2.legend(loc="upper right")
            for a in axs: a.axvline(int(d["peak"]), color="r", lw=0.5)
            axs[4].set_xlabel("step"); fig.tight_layout(); fig.savefig(f"{OUT}/fig_q5_spike_{tag}_{k}.png", dpi=80); plt.close(fig)


if __name__ == "__main__":
    {"q1": q1, "q4": q4, "q5": q5}[sys.argv[1]]()
