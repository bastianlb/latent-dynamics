"""Recompute latent-norm statistics quoted in paper_v2 (Sec. 5, Optimizer Instabilities).

Claim: Adam at wd=0 has mean test-probe latent norm 35-39 at step 1000 and 80-2900 at the end;
SGD keeps it below 36. Source: test/norm in analysis_v2/cache/*.npz. CPU only, thread capped.
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "8")
import glob
import re
import numpy as np

CACHE = "/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/cache"
PAT = re.compile(r"mnist_probe_none_d16_s(\d+)_(sl40k|sgd40k|phase)_n(\w+?)_wd(.+)\.npz$")


def collect(kinds, wd):
    rows = []
    for p in sorted(glob.glob(f"{CACHE}/mnist_probe_none_d16_s*_n*_wd{wd}.npz")):
        m = PAT.search(os.path.basename(p))
        if not m or m.group(2) not in kinds:
            continue
        z = np.load(p)
        step = z["step"]
        nrm = z["test/norm"]
        i1000 = int(np.argmin(np.abs(step - 1000)))
        rows.append(dict(seed=int(m.group(1)), n=m.group(3), step1000=float(nrm[i1000]), snap=int(step[i1000]),
                         final=float(nrm[-1]), last_step=int(step[-1]), maxn=float(nrm.max()), maxn_step=int(step[np.argmax(nrm)])))
    return rows


def summarize(label, rows):
    print(f"== {label}: {len(rows)} runs")
    if not rows:
        return
    print(f"  snapshot used for 'step 1000': steps {sorted(set(r['snap'] for r in rows))}")
    s = np.array([r["step1000"] for r in rows]); f = np.array([r["final"] for r in rows]); mx = np.array([r["maxn"] for r in rows])
    print(f"  norm@~1000: min {s.min():.1f} max {s.max():.1f} mean {s.mean():.1f}")
    print(f"  final norm: min {f.min():.1f} max {f.max():.1f}; running max over training: max {mx.max():.1f}")
    for n in sorted(set(r["n"] for r in rows), key=lambda x: (x == "full", int(x) if x != "full" else 0)):
        rr = [r for r in rows if r["n"] == n]
        a = np.array([r["step1000"] for r in rr]); b = np.array([r["final"] for r in rr])
        print(f"   n={n:>5}: runs {len(rr):2d}  @1000 mean {a.mean():5.1f} [{a.min():5.1f},{a.max():5.1f}]  final mean {b.mean():7.1f} [{b.min():7.1f},{b.max():7.1f}]")


if __name__ == "__main__":
    summarize("Adam wd=0, 40k (sl40k)", collect({"sl40k"}, "0"))
    summarize("Adam wd=0, 20k (phase)", collect({"phase"}, "0"))
    summarize("Adam wd=0, all", collect({"sl40k", "phase"}, "0"))
    summarize("SGD wd=0 (sgd40k)", collect({"sgd40k"}, "0"))
    summarize("SGD wd=1e-4 (sgd40k)", collect({"sgd40k"}, "1e-4"))
    summarize("Adam wd=1e-4, 40k (sl40k)", collect({"sl40k"}, "1e-4"))
