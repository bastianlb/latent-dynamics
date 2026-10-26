"""Phase 2: scalar metrics from ae_ntk.py outputs (CPU, <=4 threads), aggregation over seeds, figures.
Usage: python ae_ntk_analyze.py [--tag main]   -> ntk/metrics.csv, ntk/summary.json, ntk/figures/*.png"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "4")
import glob, json, re
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ae_analysis as A
from ae_memorization import loo_dgm
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk"
B = 128
_X = {}
def pix(idx_tr_split):
    if "x" not in _X: _X["x"] = A.images("mnist", True)[0]
    return _X["x"]
def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    return float(spearmanr(a[ok], b[ok])[0]) if ok.sum() >= 5 else np.nan
def pr(ev):
    ev = np.clip(ev, 0, None); return float(ev.sum() ** 2 / max((ev ** 2).sum(), 1e-30))

def kmet(M, tag, itr, ite, km, Z, Xp, out):
    npb = len(itr)
    Mtt = M[np.ix_(itr, itr)].astype(np.float64); Mst = M[np.ix_(ite, itr)].astype(np.float64)
    dtr = np.diag(Mtt).copy(); dte = np.diag(M)[ite].astype(np.float64)
    off = ~np.eye(npb, dtype=bool)
    S = (Mtt * off).sum(1)
    pull_tr = (B - 1) / (npb - 1) * S; pull_te = B / npb * Mst.sum(1)
    out[f"{tag}_diag_tr"] = dtr.mean(); out[f"{tag}_diag_te"] = dte.mean()
    out[f"{tag}_pull_tr"] = pull_tr.mean(); out[f"{tag}_pull_te"] = pull_te.mean()
    out[f"{tag}_self_share"] = float(np.mean(dtr / (dtr + pull_tr)))         # expected own share of a train point's batch-step coupling
    out[f"{tag}_self_over_pull_med"] = float(np.median(dtr / np.abs(pull_tr).clip(1e-30)))
    out[f"{tag}_harm_tr"] = float(np.mean(pull_tr < 0)); out[f"{tag}_harm_te"] = float(np.mean(pull_te < 0))
    out[f"{tag}_pull_over_diag_tr"] = float(np.mean(pull_tr / dtr)); out[f"{tag}_pull_over_diag_te"] = float(np.mean(pull_te / dte))
    ctt = Mtt / np.sqrt(np.outer(dtr, dtr)); cst = Mst / np.sqrt(np.outer(dte, dtr))
    out[f"{tag}_cos_tt"] = ctt[off].mean(); out[f"{tag}_cos_st"] = cst.mean()
    same_tt = (km[itr][:, None] == km[itr][None]) & off; diff_tt = (km[itr][:, None] != km[itr][None])
    same_st = km[ite][:, None] == km[itr][None]
    out[f"{tag}_cos_same_tt"] = ctt[same_tt].mean(); out[f"{tag}_cos_diff_tt"] = ctt[diff_tt].mean()
    out[f"{tag}_cos_same_st"] = cst[same_st].mean(); out[f"{tag}_cos_diff_st"] = cst[~same_st].mean()
    out[f"{tag}_cos_exc_tt"] = out[f"{tag}_cos_same_tt"] - out[f"{tag}_cos_diff_tt"]   # within-cluster excess = locality in input clusters
    out[f"{tag}_cos_exc_st"] = out[f"{tag}_cos_same_st"] - out[f"{tag}_cos_diff_st"]
    # locality / rank
    ev = np.linalg.eigvalsh((Mtt + Mtt.T) / 2); out[f"{tag}_PR"] = pr(ev) / npb
    Hc = np.eye(npb) - 1.0 / npb; evc = np.linalg.eigvalsh(Hc @ ctt @ Hc); out[f"{tag}_PRc"] = pr(evc) / npb
    out[f"{tag}_diagdom"] = float(npb / (ctt ** 2).sum())     # 1: cosine matrix = identity; 1/npb: rank one
    w = np.where(off, np.clip(ctt, 0, None), 0); out[f"{tag}_nn_eff"] = float(np.mean(w.sum(1) ** 2 / np.maximum((w ** 2).sum(1), 1e-30)) / npb)
    # alignment of kernel cosine to input / latent proximity
    iu = np.triu_indices(npb, 1)
    dp = np.linalg.norm(Xp[0][:, None] - Xp[0][None], axis=-1); dz = np.linalg.norm(Z[itr][:, None] - Z[itr][None], axis=-1)
    out[f"{tag}_sp_pix_tt"] = sp(ctt[iu], -dp[iu]); out[f"{tag}_sp_lat_tt"] = sp(ctt[iu], -dz[iu])
    dps = np.linalg.norm(Xp[1][:, None] - Xp[0][None], axis=-1); dzs = np.linalg.norm(Z[ite][:, None] - Z[itr][None], axis=-1)
    out[f"{tag}_sp_pix_st"] = sp(cst.ravel(), -dps.ravel()); out[f"{tag}_sp_lat_st"] = sp(cst.ravel(), -dzs.ravel())

def metrics(d):
    tr = d["is_tr"]; ib = d["in_batch"]; itr = np.where(tr)[0]; ite = np.where(~tr)[0]
    L, Z, y, km = d["L"].astype(np.float64), d["Z"], d["y"], d["km"]
    out = dict(step=int(d["step"]), epoch=float(d["epoch"]), Ltr=L[itr].mean(), Lte=L[ite].mean(), gcos=float(d["gcos"]))
    for nm, ii in (("tr", itr), ("te", ite)):
        out[f"rho_{nm}"] = sp(np.log(L[ii] + 1e-12), loo_dgm(Z[ii], y[ii]))
    act = (d["Lnext"] - d["L"]).astype(np.float64); dl = d["dL"].astype(np.float64)
    for g, m in dict(trin=tr & ib, trout=tr & ~ib, te=~tr).items():
        out[f"act_{g}"] = act[m].mean() if m.any() else np.nan
        for ci, c in enumerate(("H", "Nown", "Noth", "W")):
            for bi, b in enumerate(("e", "d")):
                out[f"dl_{g}_{c}_{b}"] = dl[m, ci, bi].mean() if m.any() else np.nan
    if "dLlayer" in d:
        lnames = [str(x) for x in d["layer_names"]]
        for g, m in dict(te=~tr, trout=tr & ~ib, trin=tr & ib).items():
            for li, ln in enumerate(lnames):
                out[f"layer_{g}_{ln}"] = float(d["dLlayer"][m, 0, li].mean()) if m.any() else np.nan
    idx = d["idx"]; Xall = pix(None)
    Xp = (Xall[idx[itr]].astype(np.float64), Xall[idx[ite]].astype(np.float64)) if False else None
    # input vectors (train split for train probes; test split for test probes)
    Xtr, _ = A.images("mnist", True); Xte, _ = A.images("mnist", False)
    Xp = (Xtr[idx[itr]].astype(np.float64), Xte[idx[ite]].astype(np.float64))
    # T, Tz: plain (D=1) function-space eNTK (the Adam-preconditioned trace kernel is dominated by zero-gradient params with D~1/eps);
    # G: Adam-preconditioned loss-gradient kernel (what the optimizer actually uses)
    for nm in ("Traw_e", "Traw_d", "Gpre_e", "Gpre_d", "Tzraw_e"):
        kmet(d[nm], nm.replace("pre", "").replace("raw", ""), itr, ite, km, Z, Xp, out)
    kmet(d["Traw_e"] + d["Traw_d"], "T", itr, ite, km, Z, Xp, out)
    kmet(d["Gpre_e"] + d["Gpre_d"], "G", itr, ite, km, Z, Xp, out)
    # per-point link: does the self-share relate to latent typicality?
    Gt = (d["Gpre_e"] + d["Gpre_d"])[np.ix_(itr, itr)].astype(np.float64); dg = np.diag(Gt); S = ((Gt) * ~np.eye(len(itr), dtype=bool)).sum(1)
    ss = dg / (dg + (B - 1) / (len(itr) - 1) * S)
    out["sp_selfshare_dgm_tr"] = sp(ss, loo_dgm(Z[itr], y[itr])); out["sp_selfshare_loss_tr"] = sp(ss, L[itr])
    out["sp_diag_dgm_tr"] = sp(dg, loo_dgm(Z[itr], y[itr]))
    return out

def run_info(rid):
    p = rid.split("_"); return dict(kind=p[2], seed=int(p[4][1:]), N=p[5][1:], wd=rid.split("_wd")[1])

def collect(force=False):
    f = f"{ROOT}/metrics.json"; rows = []; done = set()
    if os.path.exists(f) and not force:
        rows = json.load(open(f)); done = {(r["rid"], r["step"]) for r in rows}
    for fn in sorted(glob.glob(f"{ROOT}/ana/*/step_*.npz")):
        rid = os.path.basename(os.path.dirname(fn)); st = int(re.search(r"step_(\d+)", fn).group(1))
        if (rid, st) in done: continue
        d = dict(np.load(fn)); m = metrics(d); m.update(run_info(rid)); m["rid"] = rid
        rows.append({k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in m.items()})
    json.dump(rows, open(f, "w")); return rows

if __name__ == "__main__":
    rows = collect(force="--force" in sys.argv)
    print(len(rows))
