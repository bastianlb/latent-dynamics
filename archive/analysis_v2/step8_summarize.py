#!/usr/bin/env python
"""Amendment 6: aggregate A subsample CSVs into A_subsample_summary.csv/.txt (distribution of dR2 over draws)."""
import csv, glob, collections, sys, os
import numpy as np
OUT = "/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/ext5"
rows = []
for fn in sorted(glob.glob(f"{OUT}/A_subsample_K*_part*.csv")):
    rows += list(csv.DictReader(open(fn)))
g = collections.defaultdict(list)
for r in rows: g[(int(r["K"]), r["target"], int(r["cutoff"]))].append(r)
out = []; lines = []
for (K, t, c), v in sorted(g.items()):
    for blk in ("G", "c4", "NZ18", "RP18"):
        x = np.array([float(r[f"d_{blk}"]) for r in v]); b = np.array([float(r["r2_B1"]) for r in v])
        q = np.percentile(x, [5, 25, 50, 75, 95])
        out.append(dict(K=K, target=t, cutoff=c, block=blk, ndraw=len(x), r2_B1_mean=b.mean(), mean=x.mean(), sd=x.std(), p5=q[0], p25=q[1], median=q[2], p75=q[3], p95=q[4], frac_neg=(x < 0).mean(), frac_below_m=(x < -(0.03 if t[:3] == "acc" else 0.05)).mean()))
with open(f"{OUT}/A_subsample_summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(out[0].keys())); w.writeheader(); w.writerows(out)
for K in sorted({o["K"] for o in out}):
    lines.append(f"== K={K} configs subsampled from the 69 old set-(ii) configs (LOCO train size {K-1}); dR2 vs B1; mean [p5,p95] frac<0 ==")
    for blk in ("G", "c4", "NZ18", "RP18"):
        for o in [o for o in out if o["K"] == K and o["block"] == blk]:
            lines.append(f"  {blk:5s} {o['target']:8s} cut{o['cutoff']:5d} n={o['ndraw']:3d} R2(B1) {o['r2_B1_mean']:.3f} mean {o['mean']:+.3f} med {o['median']:+.3f} [{o['p5']:+.3f},{o['p95']:+.3f}] frac<0 {o['frac_neg']:.2f}")
    for blk in ("G", "c4", "NZ18", "RP18"):
        oo = [o for o in out if o["K"] == K and o["block"] == blk]
        lines.append(f"  >> {blk}: mean over the 10 cells of draw-mean dR2 {np.mean([o['mean'] for o in oo]):+.3f}; cells with mean<0: {sum(o['mean']<0 for o in oo)}/10; mean frac of draws <0: {np.mean([o['frac_neg'] for o in oo]):.2f}")
open(f"{OUT}/A_subsample_summary.txt", "w").write("\n".join(lines)); print("\n".join(l for l in lines if l.startswith("  >>") or l.startswith("==")))
