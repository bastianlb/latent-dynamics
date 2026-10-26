"""Aggregate ntk/metrics.json over seeds (mean +- sd), write ntk/tables.md and figures. CPU only."""
import os, sys, json
from latentdyn._threads import cap_threads
cap_threads(4)
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk"
rows = json.load(open(f"{ROOT}/metrics.json"))
CFG = [("mlp", "250", "0"), ("mlp", "250", "1e-4"), ("mlp", "1000", "0"), ("mlp", "1000", "1e-4"), ("mlp", "full", "0"), ("linear", "250", "0"), ("linear", "1000", "0")]
COL = {("mlp", "250", "0"): "C3", ("mlp", "250", "1e-4"): "C1", ("mlp", "1000", "0"): "C0", ("mlp", "1000", "1e-4"): "C9", ("mlp", "full", "0"): "C2", ("linear", "250", "0"): "0.4", ("linear", "1000", "0"): "0.7"}
def lab(c): return f"{'lin' if c[0]=='linear' else 'mlp'} N={c[1]} wd={c[2]}"
def derived(r):
    r = dict(r)
    def sh(g):
        t = {b: sum(r.get(f"dl_{g}_{c}_{b}", np.nan) for c in ("H", "Nown", "Noth")) for b in "ed"}
        return t["d"] / (t["e"] + t["d"]) if np.isfinite(t["e"]) and (t["e"] + t["d"]) != 0 else np.nan
    for g in ("trin", "trout", "te"): r[f"decshare_{g}"] = sh(g)
    n_ = r.get("dl_trin_Nown_e", np.nan) + r.get("dl_trin_Nown_d", np.nan); o_ = r.get("dl_trin_Noth_e", np.nan) + r.get("dl_trin_Noth_d", np.nan)
    r["ownfrac_N"] = n_ / (n_ + o_) if np.isfinite(n_ + o_) and (n_ + o_) != 0 else np.nan
    n_ = r.get("dl_trin_Nown_d", np.nan); o_ = r.get("dl_trin_Noth_d", np.nan); r["ownfrac_N_d"] = n_ / (n_ + o_) if np.isfinite(n_ + o_) and (n_ + o_) != 0 else np.nan
    n_ = r.get("dl_trin_Nown_e", np.nan); o_ = r.get("dl_trin_Noth_e", np.nan); r["ownfrac_N_e"] = n_ / (n_ + o_) if np.isfinite(n_ + o_) and (n_ + o_) != 0 else np.nan
    r["Gdec_diag_share_tr"] = r["G_d_diag_tr"] / (r["G_d_diag_tr"] + r["G_e_diag_tr"]); r["Gdec_diag_share_te"] = r["G_d_diag_te"] / (r["G_d_diag_te"] + r["G_e_diag_te"])
    r["Gdec_pull_share_tr"] = r["G_d_pull_tr"] / (r["G_d_pull_tr"] + r["G_e_pull_tr"]); r["Gdec_pull_share_te"] = r["G_d_pull_te"] / (r["G_d_pull_te"] + r["G_e_pull_te"])
    r["G_pullratio_te_tr"] = r["G_pull_te"] / r["G_pull_tr"]
    r["G_pdr_te_over_tr"] = r["G_pull_over_diag_te"] / r["G_pull_over_diag_tr"]
    r["asym"] = r["G_pull_over_diag_tr"] / r["G_pull_over_diag_te"]
    r["H_te"] = r["dl_te_H_e"] + r["dl_te_H_d"]; r["N_te"] = sum(r[f"dl_te_{c}_{b}"] for c in ("Nown", "Noth") for b in "ed")
    r["lin_te"] = sum(r[f"dl_te_{c}_{b}"] for c in ("H", "Nown", "Noth", "W") for b in "ed"); r["sec_te"] = r["act_te"] - r["lin_te"]
    r["rho_gap"] = r["rho_te"] - r["rho_tr"]
    r["gap"] = r["Lte"] - r["Ltr"]; r["ratio"] = r["Lte"] / r["Ltr"]
    return r
rows = [derived(r) for r in rows]
def agg(cfg, key):
    out = {}
    for r in rows:
        if (r["kind"], r["N"], r["wd"]) != cfg: continue
        v = r.get(key, np.nan)
        out.setdefault(round(r["epoch"], 1), []).append(v)
    ep = sorted(out); m = np.array([np.nanmean(out[e]) if np.isfinite(out[e]).any() else np.nan for e in ep]); s = np.array([np.nanstd(out[e], ddof=1) if np.isfinite(out[e]).sum() > 1 else np.nan for e in ep])
    return np.array(ep), m, s, np.array([np.isfinite(out[e]).sum() for e in ep])
def panel(ax, key, title, cfgs=CFG, logy=False, hline=None):
    for c in cfgs:
        ep, m, s, n = agg(c, key)
        if len(ep) == 0: continue
        ax.errorbar(ep, m, yerr=s, color=COL[c], marker="o", ms=3, lw=1.2, capsize=2, label=lab(c), ls="--" if c[2] == "1e-4" else "-")
    ax.set_xscale("log"); ax.set_title(title, fontsize=9); ax.axvline(120, color="k", lw=.5, ls=":")
    if logy: ax.set_yscale("log")
    if hline is not None: ax.axhline(hline, color="k", lw=.5)
    ax.set_xlabel("epoch")
def fig(name, spec, ncol=3):
    n = len(spec); nr = (n + ncol - 1) // ncol
    f, axs = plt.subplots(nr, ncol, figsize=(4.3 * ncol, 3.2 * nr)); axs = np.atleast_1d(axs).ravel()
    for ax, (k, t, kw) in zip(axs, spec): panel(ax, k, t, **kw)
    for ax in axs[n:]: ax.axis("off")
    axs[0].legend(fontsize=6); f.tight_layout(); os.makedirs(f"{ROOT}/figures", exist_ok=True); f.savefig(f"{ROOT}/figures/{name}.png", dpi=130); plt.close(f)
if __name__ == "__main__":
    fig("fig1_signature", [("rho_tr", "Spearman(log err, LOO dist to class mean): TRAIN probe", {}), ("rho_te", "same: TEST probe", {}), ("Ltr", "train recon (probe)", dict(logy=True)), ("Lte", "test recon (probe)", {}), ("ratio", "test/train recon", dict(logy=True)), ("gcos", "check: replayed-batch grad cosine", {})])
    fig("fig2_H1_encoder_vs_decoder", [("decshare_trin", "decoder share of one-step improvement: train in batch", {}), ("decshare_trout", "decoder share: train NOT in batch", {}), ("decshare_te", "decoder share: test", {}),
        ("Gdec_diag_share_tr", "decoder share of own-gradient norm G_ii: train", {}), ("Gdec_diag_share_te", "decoder share of G_ii: test", {}), ("Gdec_pull_share_te", "decoder share of neighbour pull: test", {})])
    fig("fig3_H2_self_coupling", [("G_self_share", "G self share  G_ii/(G_ii+batch-neighbour pull), train", {}), ("ownfrac_N", "actual step: own fraction of new-gradient dL (in-batch train)", {}), ("G_pull_over_diag_tr", "neighbour pull / own norm: train", dict(logy=False)),
        ("G_pull_over_diag_te", "neighbour pull / own norm: test", {}), ("G_d_self_share", "decoder-block self share (train)", {}), ("G_e_self_share", "encoder-block self share (train)", {})])
    fig("fig4_H3_locality", [("G_PRc", "G (Adam loss-grad kernel): centred participation ratio / n", {}), ("G_diagdom", "G: diagonal dominance of cosine matrix", {}), ("G_cos_exc_tt", "G: same-cluster minus other-cluster cos (train-train)", {}),
        ("T_PRc", "pixel-output eNTK T: centred PR / n", {}), ("T_cos_exc_tt", "T: same-cluster cos excess", {}), ("T_d_sp_lat_st", "decoder eNTK cos vs latent proximity (Spearman, test-train)", {})])
    fig("fig5_transfer_and_late_rise", [("act_te", "actual one-step dL, TEST probe (>0 = test error rising)", dict(hline=0)), ("act_trout", "actual one-step dL, train not in batch", dict(hline=0)), ("G_pull_te", "G: predicted test coupling to train batch (<0 helps)", {}),
        ("G_harm_te", "fraction of test points whose train-batch pull is harmful", {}), ("dl_te_W_d", "weight-decay term dL on test (decoder block)", {}), ("dl_te_H_d", "momentum-history term dL on test (decoder)", {})])
    fig("fig7_asymmetry_and_step_terms", [("asym", "train/test batch-coupling asymmetry  (pull/own: train over test)", dict(logy=True, hline=1)), ("rho_gap", "decoupling  rho_test - rho_train", {}), ("H_te", "test dL: momentum-history term", dict(hline=0)), ("N_te", "test dL: new-batch-gradient term", dict(hline=0)), ("sec_te", "test dL: beyond-first-order remainder (actual - linear)", dict(hline=0)), ("act_te", "test dL: actual", dict(hline=0))])
    LAY = ["encoder.net.1", "encoder.net.3", "encoder.net.5", "decoder.net.0", "decoder.net.2", "decoder.net.4"]
    for c in CFG:
        if any(f"layer_te_{l}" in r for r in rows for l in LAY[:1]): pass
    fig("fig6_layer_split_test_dL", [(f"layer_te_{l}", f"TEST one-step dL from layer {l} (actual step, linearised)", dict(hline=0, cfgs=[c for c in CFG if c[0] == 'mlp'])) for l in LAY])
    # tables
    keys = ["Ltr", "Lte", "rho_tr", "rho_te", "G_self_share", "ownfrac_N", "G_d_self_share", "G_e_self_share", "decshare_trin", "decshare_te", "Gdec_diag_share_tr", "G_PRc", "G_diagdom", "G_cos_exc_tt", "T_PRc", "T_cos_exc_tt", "act_te", "G_harm_te", "G_pull_over_diag_tr", "G_pull_over_diag_te", "T_d_sp_lat_st", "T_e_sp_pix_st"]
    with open(f"{ROOT}/tables.md", "w") as f:
        for c in CFG:
            ep0 = agg(c, "Ltr")[0]
            if len(ep0) == 0: continue
            f.write(f"\n### {lab(c)}  (n seeds per epoch: {agg(c,'Ltr')[3].tolist()})\n\n| metric | " + " | ".join(f"ep {e:g}" for e in ep0) + " |\n|---|" + "---|" * len(ep0) + "\n")
            for k in keys:
                _, m, s, _ = agg(c, k); f.write(f"| {k} | " + " | ".join(f"{a:.3g}±{b:.2g}" if np.isfinite(a) else "-" for a, b in zip(m, s)) + " |\n")
    print("done")
