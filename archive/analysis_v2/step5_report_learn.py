import csv, sys, numpy as np, os
ds = sys.argv[1]
OUT = f"/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/ext3/{ds}"
rows = list(csv.DictReader(open(f"{OUT}/learn_results.csv")))
for r in rows:
    for k in r:
        if k not in ("dataset", "split", "fold", "target", "learner", "variant"): r[k] = float(r[k])
out = []
CUTS = [10, 30, 100, 300, 1000]
for sp in dict.fromkeys(r["split"] for r in rows):
    for t in ("acc_15kw", "gap_15kw"):
        out.append(f"### {ds} | split={sp} | target={t} | POOLED dR2 [95% CI] of (learner on B1+G variant) - (same learner on B1); H = CI upper<0 (geometry hurts)")
        out.append(f"{'learner/variant':18s} " + " ".join(f"{'cut'+str(c):>22s}" for c in CUTS) + "   R2(B1) per cut")
        for ln in ("ridge", "krr", "mlp"):
            for v in ("full", "compact4", "compact8", "pca2", "pca4"):
                s = {r["cutoff"]: r for r in rows if r["split"] == sp and r["fold"] == "POOLED" and r["target"] == t and r["learner"] == ln and r["variant"] == v}
                if not s: continue
                out.append(f"{ln+'/'+v:18s} " + " ".join(f"{s[c]['delta']:+.3f} [{s[c]['lo95']:+.2f},{s[c]['hi95']:+.2f}]{'H' if s[c]['hurts'] else ' '}" for c in CUTS)
                           + "   " + " ".join(f"{s[c]['r2_base']:+.2f}" for c in CUTS))
        out.append("")
# survival count: cutoffs>=100
out.append(f"### {ds} survival of 'geometry hurts under extrapolation at cutoffs>=100' (POOLED rows; H = 95% CI upper<0; HR = RMSE-ratio CI lower>1; n = cells)")
for sp in dict.fromkeys(r["split"] for r in rows):
    if sp == "main_loco": continue
    for ln in ("ridge", "krr", "mlp"):
        for v in ("full", "compact4", "compact8", "pca2", "pca4"):
            s = [r for r in rows if r["split"] == sp and r["fold"] == "POOLED" and r["learner"] == ln and r["variant"] == v and r["cutoff"] >= 100]
            if not s: continue
            out.append(f"  {sp:24s} {ln:6s} {v:9s}: hurts(H) {int(sum(r['hurts'] for r in s))}/{len(s)}, hurts by RMSE {int(sum(r['hurts_rmse'] for r in s))}/{len(s)}, "
                       f"dR2 range [{min(r['delta'] for r in s):+.3f},{max(r['delta'] for r in s):+.3f}], dR2_trainmean min {min(r['delta_tm'] for r in s):+.3f}, dR2_grand min {min(r['delta_gm'] for r in s):+.3f}")
# extreme negatives
out.append(f"\n### {ds} extreme negative R2 diagnostics (ridge/full and Gonly): worst held-out-mean R2 rows")
ex = sorted([r for r in rows if r["learner"] == "ridge" and r["variant"] in ("Gonly", "full") and r["fold"] != "POOLED"], key=lambda r: r["r2_var"])[:12]
out.append("split fold target cut variant n_runs heldout_var(SD pts) | R2(heldout-mean) R2(train-mean) R2(grand-mean) | RMSE_var pts vs B1 pts")
for r in ex:
    out.append(f"{r['split'][:14]} {r['fold']} {r['target']} {int(r['cutoff'])} {r['variant']} n={int(r['n_runs'])} SD={100*np.sqrt(r['heldout_var']):.2f} | {r['r2_var']:+.2f} {r['r2_var_tm']:+.2f} {r['r2_var_gm']:+.2f} | {r['rmse_var_pts']:.2f} vs {r['rmse_base_pts']:.2f}")
fr = [r for r in rows if r["learner"] == "ridge" and r["variant"] == "Gonly" and r["fold"] != "POOLED"]
if fr:
    out.append(f"  Gonly fold rows: min R2(held-out mean) {min(r['r2_var'] for r in fr):+.2f}; min R2(train-mean) {min(r['r2_var_tm'] for r in fr):+.2f}; min R2(grand-mean) {min(r['r2_var_gm'] for r in fr):+.2f}; ")
    out.append(f"  corr(held-out SD, R2 held-out mean) in fold rows: {np.corrcoef([r['heldout_var'] for r in fr],[r['r2_var'] for r in fr])[0,1]:+.2f}; share of fold rows with R2<-1: held-out-mean {np.mean([r['r2_var']<-1 for r in fr]):.2f}, train-mean {np.mean([r['r2_var_tm']<-1 for r in fr]):.2f}, grand-mean {np.mean([r['r2_var_gm']<-1 for r in fr]):.2f}")
fb = [r for r in rows if r["learner"] == "ridge" and r["variant"] == "full" and r["fold"] != "POOLED"]
out.append(f"  B1+G(full) vs B1 fold rows: min dR2 {min(r['delta'] for r in fb):+.2f} (heldout-mean), {min(r['delta_tm'] for r in fb):+.2f} (train-mean), {min(r['delta_gm'] for r in fb):+.2f} (grand-mean); median RMSE ratio {np.median([r['rmse_ratio'] for r in fb]):.2f}, max {max(r['rmse_ratio'] for r in fb):.2f}")
open(f"{OUT}/learn_summary.txt", "w").write("\n".join(out)); print("\n".join(out))
