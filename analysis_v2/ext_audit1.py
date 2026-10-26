"""Audit part 1: leakage / fairness checks (read-only). Output: ext/audit/audit1.txt"""
import os
os.environ["ANALYSIS_OUT"] = "audit"
import common2 as C, data2 as D, step3_predict as S
import numpy as np, glob
out = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
runs = D.load_runs("mnist")
P("n runs", len(runs), "configs", len({r['info']['config_id'] for r in runs}))
# 1. feature snapshot step <= cutoff, and < target steps
bad = 0; mins = []
for cut in S.CUTS:
    for r in runs:
        B0, DES, G, stp = S.feats(r, cut)
        i = D.idx_le_mask(r["step"], r["topo"], cut)
        if stp > cut and stp != r["step"][np.where(r["topo"])[0][0]]: bad += 1
        # target snapshot index vs feature index
        i15 = D.idx_le(r["step"], 15000)
        mins.append((cut, stp, r["step"][r["pre"]], r["step"][i15]))
m = np.array(mins)
P("feature snapshots with step>cutoff:", bad, "| max feature step per cutoff:", {c: int(m[m[:,0]==c,1].max()) for c in S.CUTS})
P("min pre-slingshot target step over runs:", int(m[:,2].min()), "; 15k target snapshot min step:", int(m[:,3].min()))
# 2. all features come from one snapshot index: verify against raw cache arrays for one run
r = runs[0]; d = r["d"]; B0, DES, G, stp = S.feats(r, 300)
i = D.idx_le_mask(r["step"], r["topo"], 300)
P("example run", r["info"]["run_id"], "cut300 -> snapshot idx", i, "step", int(r["step"][i]),
  "test_acc feat", round(B0["test_acc"],4), "cache", round(float(d["run/test_acc"][i]),4),
  "te_fisher", round(G["te_fisher"],4), "cache log", round(float(np.log(d["test/fisher"][i])),4))
# 3. target in features? correlation of target with features is expected; check target is not literally a feature column
for t in ("acc_15k","gap_15k","acc_pre","gap_pre"):
    y = np.array([S.targets(r)[t] for r in runs])
    B0, DES, G, cfg, seeds, names = S.build(runs, 1000)
    X = np.hstack([B0, DES, G]); nm = names[0]+names[1]+names[2]
    mx = max(abs(np.corrcoef(X[:,j], y)[0,1]) for j in range(X.shape[1]) if X[:,j].std()>0)
    P(f"target {t}: max |corr| with any cutoff-1000 feature = {mx:.3f}; exact duplicate column: {any(np.allclose(X[:,j],y) for j in range(X.shape[1]))}")
# 4. probe set identity across runs / snapshots (same samples at every snapshot and run?)
rd = C.RUNS
def idx(rid, f): 
    z = np.load(f"{rd}/{rid}/{f}"); return {k: z[k] for k in z.files}
ids = [r["info"]["run_id"] for r in runs]
a = idx(ids[0], "probe_index.npz"); P("probe_index keys", list(a.keys()), {k: v.shape for k, v in a.items()})
same_te = same_tr = 0
for rid in ids:
    b = idx(rid, "probe_index.npz"); same_te += all(np.array_equal(a[k], b[k]) for k in a)
    b2 = idx(rid, "probe_train_index.npz"); a2 = idx(ids[0], "probe_train_index.npz")
    same_tr += all(np.array_equal(a2[k], b2[k]) for k in a2) if set(a2)==set(b2) else 0
P(f"test probe index identical to run0 in {same_te}/{len(ids)} runs; train probe index identical in {same_tr}/{len(ids)} runs (train probe differs across n/seeds if train subset differs)")
# labels identical across snapshots in a run
fs = sorted(glob.glob(f"{rd}/{ids[0]}/latents/step_*.npz"))
l0 = np.load(fs[0])["labels"]; P("test-probe labels identical across snapshots:", all(np.array_equal(np.load(f)["labels"], l0) for f in fs[::20]))
# is run/test_acc computed on full test set or probe? compare
z = np.load(fs[-1]); P("snapshot keys:", z.files)
open(f"{C.OUT}/audit1.txt","w").write("\n".join(out))
