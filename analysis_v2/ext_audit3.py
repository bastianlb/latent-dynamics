"""Audit part 3: power of the ridge/LOCO pipeline: stronger positive controls (noisy copy of target, no G)."""
import os
os.environ["ANALYSIS_OUT"] = "audit"
import common2 as C, data2 as D, step3_predict as S
import numpy as np
out=[]
def P(*a):
    s=" ".join(str(x) for x in a); print(s,flush=True); out.append(s)
runs = D.load_runs("mnist")
rng = np.random.default_rng(1)
for t in ("acc_15k","gap_15k"):
    y = np.array([S.targets(r)[t] for r in runs])
    for cut in (100,1000):
        B0,DES,G,g,_,_ = S.build(runs,cut); B1=np.hstack([B0,DES]); pb=S.loco(B1,y,g,"ridge")
        for k in (0.05,0.1,0.2,0.4):
            z=y+rng.normal(0,k*y.std(),len(y))
            for lab,Xc in (("copy alone",np.hstack([B1,z[:,None]])),("copy + compact G(8)",None)):
                if Xc is None: continue
                pc=S.loco(Xc,y,g,"ridge"); lo,hi=S.boot_delta(y,pb,pc,g,nb=1000)
                P(f"{t} cut={cut} noise {k}sd {lab}: R2 {S.r2(y,pb):+.3f}->{S.r2(y,pc):+.3f} dR2 {S.r2(y,pc)-S.r2(y,pb):+.3f} [{lo:+.3f},{hi:+.3f}]")
open(f"{C.OUT}/audit3.txt","w").write("\n".join(out))
