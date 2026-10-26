"""Audit part 2: artefact checks + controls on ORIGINAL 35-config set. Output ext/audit/audit2.txt, audit2.csv.
Onset-dependent targets (pre) deliberately excluded (provisional)."""
import os
os.environ["ANALYSIS_OUT"] = "audit"
import common2 as C, data2 as D, step3_predict as S
import numpy as np, csv
from scipy import stats
out = []; rows = []
def P(*a):
    s = " ".join(str(x) for x in a); print(s, flush=True); out.append(s)
runs = D.load_runs("mnist")
def window_target(r, key, lo=12000, hi=18000):
    st = r["step"]; m = (st >= lo) & (st <= hi)
    return float(np.median(r["d"]["run/test_acc"][m])) if key == "acc" else float(np.median(r["gap"][m]))
Y = {"acc_15k": np.array([S.targets(r)["acc_15k"] for r in runs]), "gap_15k": np.array([S.targets(r)["gap_15k"] for r in runs]),
     "acc_15kw": np.array([window_target(r, "acc") for r in runs]), "gap_15kw": np.array([window_target(r, "gap") for r in runs])}
P("snapshots in 12k-18k window per run: min", min(((r['step']>=12000)&(r['step']<=18000)).sum() for r in runs))
cuts = [10, 100, 1000]
BUILT = {c: S.build(runs, c) for c in S.CUTS}
g = BUILT[10][3]

def inc(y, Xb, Xc, g, model="ridge", nb=1000):
    pb = S.loco(Xb, y, g, model); pc = S.loco(Xc, y, g, model)
    lo, hi = S.boot_delta(y, pb, pc, g, nb=nb)
    return S.r2(y, pb), S.r2(y, pc), S.r2(y, pc) - S.r2(y, pb), lo, hi, pb, pc
def rec(check, tgt, cut, label, res):
    rb, rc, d, lo, hi = res[:5]
    rows.append(dict(check=check, target=tgt, cutoff=cut, variant=label, r2_base=rb, r2_comp=rc, delta=d, lo=lo, hi=hi))
    P(f"  [{check}] {tgt} cut={cut} {label:34s} R2 base {rb:+.3f} comp {rc:+.3f}  dR2 {d:+.3f} [{lo:+.3f},{hi:+.3f}]")

def sets(cut):
    B0, DES, G, _, _, names = BUILT[cut]
    return B0, DES, G, names[2]
def sub(G, gn, keep):
    return G[:, [i for i, n in enumerate(gn) if keep(n)]]

# A. reproduce original
P("== A. reproduction of original B1 -> B1+G (ridge) with current code")
import csv as _c
orig = list(_c.DictReader(open(f"{C.OUT_ROOT}/step3_results.csv")))
for t in ("acc_15k", "gap_15k"):
    for cut in cuts:
        B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES])
        res = inc(Y[t], B1, np.hstack([B1, G]), g, nb=2000)
        o = next(r for r in orig if r["tag"]=="main" and r["target"]==t and int(r["cutoff"])==cut and r["model"]=="ridge" and r["kind"]=="increment" and r["base"]=="B1")
        P(f"  {t} cut={cut}: now dR2 {res[2]:+.4f} [{res[3]:+.3f},{res[4]:+.3f}] | stored {float(o['delta']):+.4f} [{float(o['ci_lo']):+.3f},{float(o['ci_hi']):+.3f}]")

# B. compact geometry sets / variants
P("== B. compact geometry sets (ridge); base = B1")
compact = {
 "compact: te+delta of F,rank,Mag1,H1top5 (8)": lambda n: n.split("_",1)[1] in ("fisher","eff_rank_pr","mag_t1","h1_top5") and n[:2] in ("te","dl"),
 "delta-only of F,rank,Mag1,H1top5 (4)": lambda n: n.split("_",1)[1] in ("fisher","eff_rank_pr","mag_t1","h1_top5") and n[:2]=="dl",
 "test-probe only, 4 metrics (4)": lambda n: n.split("_",1)[1] in ("fisher","eff_rank_pr","mag_t1","h1_top5") and n[:2]=="te",
 "Fisher te+delta (2)": lambda n: n.split("_",1)[1]=="fisher" and n[:2] in ("te","dl"),
 "H1 top-5 te+delta (2)": lambda n: n.split("_",1)[1]=="h1_top5" and n[:2] in ("te","dl"),
 "full 18": lambda n: True}
for t in ("acc_15k", "gap_15k", "acc_15kw", "gap_15kw"):
    for cut in cuts:
        B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES])
        for lab, f in compact.items():
            rec("B-compact", t, cut, lab, inc(Y[t], B1, np.hstack([B1, sub(G, gn, f)]), g))
P("== C. alpha grid / GBM hyperparameters / alternative regularisation")
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
orig_alpha = S.ALPHAS.copy()
for lab, grid in (("alpha 1e-6..1e6 (25)", np.logspace(-6, 6, 25)), ("alpha 1e-1..1e1 (9)", np.logspace(-1, 1, 9)), ("alpha fixed 1", np.array([1.0])), ("alpha fixed 100", np.array([100.0]))):
    S.ALPHAS = grid
    for t in ("acc_15k", "gap_15k"):
        for cut in cuts:
            B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES])
            rec("C-alpha", t, cut, lab, inc(Y[t], B1, np.hstack([B1, G]), g))
S.ALPHAS = orig_alpha
def mk(kind, **kw):
    def f(Xtr, ytr, gtr, Xte):
        m = GradientBoostingRegressor(random_state=0, **kw) if kind == "gbm" else RandomForestRegressor(random_state=0, n_jobs=1, **kw)
        return m.fit(Xtr, ytr).predict(Xte)
    return f
orig_gbm = S.gbm_fit_predict
for lab, fn in (("GBM d1 300t lr.05", mk("gbm", n_estimators=300, max_depth=1, learning_rate=0.05, subsample=0.8)),
                ("GBM d3 300t lr.03", mk("gbm", n_estimators=300, max_depth=3, learning_rate=0.03, subsample=0.8)),
                ("RF 200 trees mindleaf3", mk("rf", n_estimators=200, min_samples_leaf=3, max_features=0.5))):
    S.gbm_fit_predict = fn
    for t in ("acc_15k", "gap_15k"):
        for cut in (100, 1000):
            B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES])
            rec("C-gbm", t, cut, lab, inc(Y[t], B1, np.hstack([B1, G]), g, model="gbm", nb=500))
            rec("C-gbm-compact8", t, cut, lab, inc(Y[t], B1, np.hstack([B1, sub(G, gn, compact["compact: te+delta of F,rank,Mag1,H1top5 (8)"])]), g, model="gbm", nb=500))
S.gbm_fit_predict = orig_gbm
P("== D. Delta target (target minus value at cutoff) and config-level evaluation")
for t, key in (("acc_15k", "test_acc"), ("gap_15k", "gap")):
    for cut in cuts:
        B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES])
        cur = B0[:, 3] if key == "test_acc" else B0[:, 4]
        y2 = Y[t] - cur
        rec("D-deltatarget", t, cut, "y - current value; full G", inc(y2, B1, np.hstack([B1, G]), g))
        rec("D-deltatarget", t, cut, "y - current value; compact8", inc(y2, B1, np.hstack([B1, sub(G, gn, compact["compact: te+delta of F,rank,Mag1,H1top5 (8)"])]), g))
        # config-level mean R2 and paired Wilcoxon on per-config SSE
        res = inc(Y[t], B1, np.hstack([B1, G]), g); pb, pc = res[5], res[6]
        us = np.unique(g)
        yc = np.array([Y[t][g==u].mean() for u in us]); pbc = np.array([pb[g==u].mean() for u in us]); pcc = np.array([pc[g==u].mean() for u in us])
        eb = np.array([((Y[t][g==u]-pb[g==u])**2).sum() for u in us]); ec = np.array([((Y[t][g==u]-pc[g==u])**2).sum() for u in us])
        w = stats.wilcoxon(eb, ec)
        P(f"  [D-config] {t} cut={cut}: config-mean R2 base {S.r2(yc,pbc):+.3f} comp {S.r2(yc,pcc):+.3f}; Wilcoxon on per-config SSE base>comp: p={w.pvalue:.3f}; configs improved {int((ec<eb).sum())}/{len(us)}")
P("== E. controls (ridge, base B1)")
rng = np.random.default_rng(0)
for t in ("acc_15k", "gap_15k"):
    for cut in cuts:
        B0, DES, G, gn = sets(cut); B1 = np.hstack([B0, DES]); y = Y[t]
        # positive: noisy copy of target (noise sd = k * sd(y)); added alone and with G
        for k in (1.0, 0.5):
            z = y + rng.normal(0, k * y.std(), len(y))
            rec("E-pos", t, cut, f"noisy target copy (noise {k}sd) + G", inc(y, B1, np.hstack([B1, G, z[:, None]]), g))
        # partial positive: feature correlated rho with B1-OOF residual
        pb = S.loco(B1, y, g, "ridge"); e = y - pb
        for rho in (0.3, 0.5):
            z = rho * e / e.std() + np.sqrt(1 - rho**2) * rng.normal(size=len(y))
            rec("E-pos-resid", t, cut, f"feature corr {rho} with OOF residual", inc(y, B1, np.hstack([B1, z[:, None]]), g))
        # negative: 18 gaussian noise columns; G permuted across configs (3 draws); target shuffled across configs
        rec("E-neg", t, cut, "18 gaussian noise cols", inc(y, B1, np.hstack([B1, rng.normal(size=(len(y), 18))]), g))
        ds = []
        for k in range(5):
            Gp = S.permute_G(G, g, None, np.random.default_rng(100 + k)); ds.append(S.r2(y, S.loco(np.hstack([B1, Gp]), y, g, "ridge")) - S.r2(y, pb))
        P(f"  [E-neg] {t} cut={cut} G permuted across configs (5 draws): dR2 mean {np.mean(ds):+.3f} range [{min(ds):+.3f},{max(ds):+.3f}]")
        us = np.unique(g); perm = rng.permutation(us); ysh = y.copy()
        for u, u2 in zip(us, perm):
            a, b = np.where(g == u)[0], np.where(g == u2)[0]
            for k, ia in enumerate(a): ysh[ia] = y[b[k % len(b)]]
        P(f"  [E-neg] {t} cut={cut} target shuffled across configs: R2(B1)={S.r2(ysh, S.loco(B1, ysh, g, 'ridge')):+.3f} R2(B1+G)={S.r2(ysh, S.loco(np.hstack([B1,G]), ysh, g, 'ridge')):+.3f}")
open(f"{C.OUT}/audit2.txt", "w").write("\n".join(out))
with open(f"{C.OUT}/audit2.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(rows[0].keys())); w.writeheader(); w.writerows(rows)
