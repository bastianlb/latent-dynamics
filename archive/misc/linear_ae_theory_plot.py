import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "2")
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
OUT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/linear_theory"
S = json.load(open(f"{OUT}/summary.json")); lam = np.array(S["lam"]); K = 16
conds = [f"{o}_{i}" for o in ("sgd", "adam") for i in ("default", "small")]
fig, ax = plt.subplots(2, 4, figsize=(19, 8.5))
cm = plt.cm.viridis
for c, name in enumerate(conds):
    a = ax[0, c]; z = np.load(f"{OUT}/traj_{name}_s0.npz"); st = z["steps"]; s = z["S"]
    for k in range(K): a.plot(st[1:], s[1:, k], color=cm(k / 15), lw=1)
    a.set_xscale("log"); a.set_title(name + " (seed 0): s_k = v_k^T Wd We v_k"); a.set_xlabel("step"); a.set_ylim(-0.1, 1.15)
    a = ax[1, c]; ht = np.array(S[name]["ht_sv"]); hr = np.array(S[name]["ht_r2"])
    for j in range(3):
        a.loglog(lam, ht[j], "o", color="C0", ms=4, alpha=.6, label="decoder-enc strength s_k" if j == 0 else None)
        a.loglog(lam, hr[j], "s", color="C3", ms=4, alpha=.6, label="affine-readout R2 (ae_analysis)" if j == 0 else None)
    if name.startswith("sgd"):
        th = np.array(S["theory_sgd_" + name.split("_")[1]]["ht"]); a.loglog(lam, th.mean(0), "k-", lw=1.5, label="theory ODE (measured s0)")
    anc = np.nanmean(ht[:, 3])
    for p, col in ((-1, "gray"), (-2, "silver")): a.loglog(lam, anc * (lam / lam[3]) ** p, "--", color=col, lw=.8, label=f"slope {p}")
    m = S[name]; a.set_title(f"slope s_k: {m['slope_sv'][0]:.2f}+-{m['slope_sv'][1]:.2f} | R2: {m['slope_r2'][0]:.2f}+-{m['slope_r2'][1]:.2f}")
    a.set_xlabel("lambda_k"); a.set_ylabel("rise step (50%)")
    if c == 0: a.legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{OUT}/linear_ae_theory.png", dpi=130)
print("ok")
