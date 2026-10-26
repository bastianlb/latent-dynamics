"""Per-class / per-pair decomposition of latent dynamics (MNIST grid v2 + v1 reference). CPU only.

Usage: python perclass.py [--workers 4] [--force] [--runs_v2 GLOB,..] [--skip_v1]
Outputs only under /vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/perclass/ (cache in perclass/cache/<run_id>.pkl, keyed by file list hash).
Per snapshot (test probe 'latents/', train probe 'latents_train/'; classes = TRUE labels):
  S[a,b]   = ||mu_a-mu_b||^2 / (trSigma_a+trSigma_b)       (scale invariant pair separation)
  spread_c = trSigma_c / tr(total cov);  margin_c = median over class-c samples of (d_nearest_other_mean - d_own_mean)/sqrt(tr total cov)
  acc_c    = NCM accuracy of class c;  speed_c = ||dmu_c||/(dlog(step+1) sqrt(tr total cov))
  nc1_c    = p_c tr(Sigma_c Sb^+)/C   (sums to dynamics.py NC1)
Onsets: pair onset = half-max crossing (log time, linear interp.) of 3-pt-median-filtered S[a,b]; class onset = first step after which class NCM
accuracy stays >= 0.5 to the end.
"""
import os, sys
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_k, "2")
import argparse, glob, json, csv, time, hashlib, pickle, collections, traceback, itertools
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.stats import spearmanr, mannwhitneyu
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import phase_diagram as PD

V2 = "/vol/bitbucket/lbastian/latent-dynamics/runs_v2"
V1 = "/vol/bitbucket/lbastian/latent-dynamics/runs"
OUT = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/perclass"
PHASE_CACHE = "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/phase_cache"
CODE = "pc-1"
C = 10
IU = np.triu_indices(C, 1)
CONF_PAIRS = [(4, 9), (3, 5), (3, 8), (5, 8), (1, 7), (2, 7), (1, 2)]
CONF_MASK = np.array([(a, b) in CONF_PAIRS for a, b in zip(*IU)])
SEL_STEPS = (10, 100, 1000, 10000)
LOWN = 30  # per-class sample count below which results are flagged


def snap_hash(rd):
    items = []
    for sub in ("latents", "latents_train"):
        for f in sorted(glob.glob(f"{rd}/{sub}/step_*.npz")):
            try: items.append((sub, os.path.basename(f), os.stat(f).st_size))
            except OSError: pass
    return hashlib.sha1((CODE + json.dumps(items)).encode()).hexdigest()


def rise_onset(steps, y):
    """Half-max rise crossing in log time (returns step value, nan if no rise)."""
    if len(y) < 5: return np.nan
    yf = y.copy(); yf[1:-1] = np.median(np.stack([y[:-2], y[1:-1], y[2:]]), 0)
    lo = np.median(yf[:3]); hi = np.nanmax(yf)
    if not hi - lo > 1e-9: return np.nan
    h = lo + 0.5 * (hi - lo); i = np.where(yf >= h)[0][0]
    x = np.log1p(steps)
    if i == 0: return float(steps[0])
    xx = x[i - 1] + (h - yf[i - 1]) / (yf[i] - yf[i - 1]) * (x[i] - x[i - 1]); return float(np.expm1(xx))


def stay_onset(steps, ok):
    """First step after which ok stays True to the end (nan if never / not at end)."""
    if not ok[-1]: return np.nan
    bad = np.where(~ok)[0]; i = 0 if len(bad) == 0 else bad[-1] + 1
    return float(steps[i])


def snapshot_stats(Z, y, classes):
    n, d = Z.shape
    mu = np.array([Z[y == c].mean(0) for c in classes]); g = Z.mean(0)
    cnt = np.array([(y == c).sum() for c in classes]); pc = cnt / n
    trs = np.array([((Z[y == c] - m) ** 2).sum(1).mean() for c, m in zip(classes, mu)])
    tot = ((Z - g) ** 2).sum(1).mean()
    dm = ((mu[:, None] - mu[None]) ** 2).sum(-1)
    S = dm / (trs[:, None] + trs[None] + 1e-30)
    d2 = ((Z[:, None, :] - mu[None]) ** 2).sum(-1); dist = np.sqrt(d2)
    pred = classes[d2.argmin(1)]
    own = dist[np.arange(n), np.searchsorted(classes, y)]
    other = np.where(classes[None] == y[:, None], np.inf, dist).min(1)
    sc = np.sqrt(tot + 1e-30)
    acc = np.array([(pred[y == c] == c).mean() for c in classes])
    marg = np.array([np.median((other - own)[y == c]) for c in classes]) / sc
    Mc = mu - g; Sb = (Mc * pc[:, None]).T @ Mc; Sbp = np.linalg.pinv(Sb, rcond=1e-6, hermitian=True)
    nc1c = np.array([p * np.trace(np.cov(Z[y == c].T, bias=True).reshape(d, d) @ Sbp) for p, c in zip(pc, classes)]) / len(classes)
    conf = np.array([[(pred[y == a] == b).mean() for b in classes] for a in classes])
    return dict(S=S, spread=trs / (tot + 1e-30), acc=acc, margin=marg, nc1c=nc1c, mu=mu / sc, conf=conf, scale=sc)


def analyze_set(d, noisy_ok=True):
    steps, Zs, y = d["steps"], d["Zs"], d["labels"]; classes = np.unique(y)
    st = [snapshot_stats(Z, y, classes) for Z in Zs]
    T = len(steps); r = dict(steps=steps, cnt=np.array([(y == c).sum() for c in classes]))
    for k in ("S", "spread", "acc", "margin", "nc1c", "conf"): r[k] = np.stack([s[k] for s in st]).astype(np.float32)
    mu = np.stack([s["mu"] for s in st]); dl = np.diff(np.log1p(steps))
    r["speed"] = (np.linalg.norm(np.diff(mu, axis=0), axis=2) / dl[:, None]).astype(np.float32)  # unit: scale-normalized means
    r["pair_onset"] = np.array([[rise_onset(steps, r["S"][:, a, b]) for b in range(C)] for a in range(C)])
    r["class_onset"] = np.array([stay_onset(steps, r["acc"][:, c] >= 0.5) for c in range(C)])
    r["ncm_onset_mean"] = stay_onset(steps, r["acc"].mean(1) >= 0.5)
    lk = {}
    for s0 in SEL_STEPS + (-1,):
        i = len(steps) - 1 if s0 == -1 else (np.where(steps <= s0)[0][-1] if (steps <= s0).any() and steps[-1] >= s0 else None)
        if i is None: continue
        lk[int(steps[i])] = linkage(np.array([Zs[i][y == c].mean(0) for c in classes]), "average")
    r["linkage"] = lk
    if noisy_ok and d["noisy"] is not None:
        yn = d["noisy"]; bad = yn != y
        fit = np.full((T, C), np.nan); cor = np.full((T, C), np.nan); pairfit = np.full((C, C), np.nan); pc = np.zeros((C, C))
        for t, Z in enumerate(Zs):
            mu = np.array([Z[yn == c].mean(0) if (yn == c).any() else np.full(Z.shape[1], 1e9) for c in classes])
            pred = classes[((Z[:, None, :] - mu[None]) ** 2).sum(-1).argmin(1)]
            for c in range(C):
                m = bad & (y == c)
                if m.any(): fit[t, c] = (pred[m] == yn[m]).mean(); cor[t, c] = (pred[m] == y[m]).mean()
            if t == T - 1:
                for a in range(C):
                    for b in range(C):
                        m = bad & (y == a) & (yn == b); pc[a, b] = m.sum()
                        if m.any(): pairfit[a, b] = (pred[m] == b).mean()
        r["mem_fit_c"], r["mem_cor_c"], r["n_corrupt_c"] = fit, cor, np.array([(bad & (y == c)).sum() for c in range(C)])
        r["pair_mem_final"], r["pair_n_corrupt"] = pairfit, pc
        r["mem_cross_c"] = np.array([float(steps[np.where(fit[:, c] > cor[:, c])[0][0]]) if (fit[:, c] > cor[:, c]).any() else np.nan for c in range(C)])
    return r


# ------------------------------------------------------------------ aggregation helpers
def cell_of(cfg):
    return (float(cfg.get("label_noise") or 0.0), int(cfg.get("train_subset") or 60000), float(cfg.get("weight_decay") or 0.0))


def pct_rank(v):
    from scipy.stats import rankdata
    ok = np.isfinite(v); out = np.full(len(v), np.nan)
    if ok.sum() > 1: out[ok] = (rankdata(v[ok]) - 1) / (ok.sum() - 1)
    return out


def sp(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 5 or np.ptp(a[ok]) == 0 or np.ptp(b[ok]) == 0: return np.nan
    return float(spearmanr(a[ok], b[ok])[0])


def wcsv(path, rows):
    if not rows: return
    cols = []
    for r in rows:
        for k in r:
            if k not in cols: cols.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)


def part_ratio(d):
    d = np.abs(np.asarray(d, float)); d = d[np.isfinite(d)]
    return float(d.sum() ** 2 / (len(d) * (d ** 2).sum() + 1e-30)) if len(d) else np.nan  # 1 = uniform, 1/n = one entry


def nc1_drop(r, ratio=1.8):
    steps = r["steps"]; nc1 = r["nc1c"].sum(1); ln = np.log(np.clip(nc1, 1e-9, None)); dd = np.diff(ln); dd[steps[1:] < 100] = 0
    i = int(dd.argmin()) + 1
    if dd[i - 1] > -np.log(ratio) or i + 1 >= len(steps): return None
    pre, post = i - 1, min(i + 1, len(steps) - 1); dc = r["nc1c"][pre] - r["nc1c"][post]
    lo, hi = max(1, i - 10), min(len(steps), i + 11)
    lc = np.log(np.clip(r["nc1c"], 1e-9, None)); idx = np.array([lo + int(np.diff(lc[lo - 1:hi, c]).argmin()) for c in range(C)])
    return dict(step=int(steps[i]), idx=i, ratio=float(nc1[pre] / nc1[post]), dc=dc, share=dc / (dc.sum() + 1e-30), class_idx=idx,
                class_step=steps[idx], spread_snap=int(idx.max() - idx.min()), spread_log10=float(np.log10(steps[idx.max()] + 1) - np.log10(steps[idx.min()] + 1)),
                n_within1=int((np.abs(idx - i) <= 1).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--force", action="store_true")
    ap.add_argument("--family", default="phase,gap", help="comma list of tag families: phase,gap,sl40k,sgd40k (sl40k/sgd40k aggregated separately into OUT/<family>/)")
    ap.add_argument("--runs_root", default=V2); ap.add_argument("--skip_v1", action="store_true"); ap.add_argument("--aggregate_only", action="store_true"); ap.add_argument("--out", default=OUT)
    a = ap.parse_args(); out = a.out
    assert os.path.abspath(out).startswith("/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/"), "bad out"
    os.makedirs(f"{out}/cache", exist_ok=True); os.makedirs(f"{out}/figures", exist_ok=True); t0 = time.time()
    fams = a.family.split(",")
    gl = {"phase": "mnist_probe_*_phase_*", "gap": "mnist_probe_*_gap_*", "sl40k": "mnist_probe_*_sl40k_*", "sgd40k": "mnist_probe_*_sgd40k_*"}
    d2 = sorted(set(d for f in fams for d in glob.glob(f"{a.runs_root}/runs/{gl[f]}")))
    d1 = [] if (a.skip_v1 or not ({"phase", "gap"} & set(fams))) else sorted(d for d in glob.glob(f"{V1}/mnist_*_s[0-9]") if os.path.isdir(d))
    dirs = [d for d in d2 + d1 if os.path.isdir(d)]
    # v1 and v2 share run-id patterns (e.g. mnist_probe_none_d16_s0): cache under prefixed names
    jobs = []
    for d in dirs:
        pre = "v1" if d.startswith(V1 + "/") else "v2"; jobs.append((d, f"{out}/cache", a.force, pre))
    if not a.aggregate_only:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(min(a.workers, 4)) as p: res = p.map(do_one_pre, jobs, chunksize=1)
        print(dict(collections.Counter(s.split()[0] for _, s in res)))
        for rid, s in res:
            if s.startswith("FAIL"): print(s, rid)
    def fam_of(d):
        t = os.path.basename(d); return "sl40k" if "_sl40k_" in t else "sgd40k" if "_sgd40k_" in t else "grid20k"
    for fam in ("grid20k", "sl40k", "sgd40k"):
        dd = [d for d in dirs if fam_of(d) == fam]
        if dd:
            o = out if fam == "grid20k" else f"{out}/{fam}"; os.makedirs(f"{o}/figures", exist_ok=True); aggregate(o, dd, cache=f"{out}/cache")
    print(f"done {time.time()-t0:.0f}s")


def do_one_pre(j):
    rd, cache, force, pre = j
    return do_one_named(rd, cache, force, f"{pre}_{os.path.basename(rd)}")


def do_one_named(rd, cache, force, name):
    try:
        if not os.path.isfile(f"{rd}/model_final.pt"): return name, "skip_incomplete"
        h = snap_hash(rd); cp = f"{cache}/{name}.pkl"
        if not force and os.path.isfile(cp):
            try:
                if pickle.load(open(cp, "rb")).get("hash") == h: return name, "cached"
            except Exception: pass
        cfg = json.load(open(f"{rd}/config.json")); res = dict(hash=h, run_id=os.path.basename(rd), name=name, src=name[:2], cfg=cfg)
        te, tr = PD.load_def(rd, "latents"), PD.load_def(rd, "latents_train")
        if te is None: return name, "skip_nosnap"
        res["test"] = analyze_set(te, False)
        if tr is not None: res["train"] = analyze_set(tr, True)
        tmp = cp + f".tmp{os.getpid()}"; pickle.dump(res, open(tmp, "wb")); os.replace(tmp, cp); return name, "ok"
    except Exception:
        return name, "FAIL " + traceback.format_exc()


def aggregate(out, dirs, cache=None):
    R = []
    for f in sorted(glob.glob(f"{cache or out}/cache/*.pkl" if cache is None else f"{cache}/*.pkl")):
        try: R.append(pickle.load(open(f, "rb")))
        except Exception: pass
    names = {os.path.basename(d) for d in dirs}
    R = [r for r in R if r["run_id"] in names]
    v2 = [r for r in R if r["src"] == "v2"]; v1 = [r for r in R if r["src"] == "v1"]
    print(f"{len(v2)} v2 runs, {len(v1)} v1 runs")
    rows, pair_rows, cls_rows = [], [], []
    onset = {}  # (name, probe) -> 45-vector of log10 pair onsets
    for r in R:
        c = cell_of(r["cfg"])
        for probe in ("test", "train"):
            if probe not in r: continue
            P = r[probe]; po = P["pair_onset"][IU]; lo = np.log10(po + 1); onset[(r["name"], probe)] = lo
            pr = pct_rank(lo); Sf = np.log(P["S"][-1][IU])
            cr = np.nanmean(pr[CONF_MASK]); rr = np.nanmean(pr[~CONF_MASK])
            rows.append(dict(name=r["name"], src=r["src"], probe=probe, seed=r["cfg"]["seed"], ln=c[0], N=c[1], wd=c[2], tag=r["cfg"].get("tag", ""),
                             arch=r["cfg"].get("objective", ""), min_class_n=int(P["cnt"].min()), low_n=int(P["cnt"].min() < LOWN),
                             confusable_mean_pctrank=cr, rest_mean_pctrank=rr, confusable_minus_rest=cr - rr,
                             spearman_onset_vs_finalS=sp(lo, Sf), n_pairs_with_onset=int(np.isfinite(lo).sum()), mean_log10_pair_onset=float(np.nanmean(lo)),
                             std_log10_pair_onset=float(np.nanstd(lo)), class_onset_order=" ".join(map(str, np.argsort(np.where(np.isfinite(P["class_onset"]), P["class_onset"], 1e12)))),
                             final_acc_min_class=float(P["acc"][-1].min()), final_acc_min_class_id=int(P["acc"][-1].argmin())))
            for (a, b), v in zip(zip(*IU), po): pair_rows.append(dict(name=r["name"], probe=probe, a=a, b=b, onset_step=v, confusable=int((a, b) in CONF_PAIRS)))
            for k in range(C):
                cls_rows.append(dict(name=r["name"], probe=probe, cls=k, n=int(P["cnt"][k]), onset_step=P["class_onset"][k], final_acc=float(P["acc"][-1][k]),
                                     final_margin=float(P["margin"][-1][k]), final_spread=float(P["spread"][-1][k]), peak_speed_log10step=float(np.log10(P["steps"][1:][P["speed"][:, k].argmax()] + 1))))
    wcsv(f"{out}/perclass_runs.csv", rows); wcsv(f"{out}/pair_onsets.csv", pair_rows); wcsv(f"{out}/class_onsets.csv", cls_rows)
    # ---- seed / cell consistency (v2, ln=0, test probe)
    cells = collections.defaultdict(list)
    for r in v2: cells[cell_of(r["cfg"])].append(r)
    cons = []; cellvec = {}
    for k, rs in sorted(cells.items()):
        for probe in ("test", "train"):
            vs = [onset[(r["name"], probe)] for r in rs if (r["name"], probe) in onset]
            cv = [np.array([r[probe]["class_onset"][c] for c in range(C)]) for r in rs if probe in r]
            if len(vs) >= 2:
                pp = [sp(x, y) for x, y in itertools.combinations(vs, 2)]; pc = [sp(np.log10(x + 1), np.log10(y + 1)) for x, y in itertools.combinations(cv, 2)]
                cons.append(dict(ln=k[0], N=k[1], wd=k[2], probe=probe, n_seeds=len(vs), pair_onset_seed_spearman_mean=np.nanmean(pp), pair_onset_seed_spearman_min=np.nanmin(pp),
                                 class_onset_seed_spearman_mean=np.nanmean(pc) if np.isfinite(pc).any() else np.nan, min_class_n=int(min(r[probe]["cnt"].min() for r in rs if probe in r))))
            if vs and probe == "test": cellvec[k] = np.nanmean(np.stack(vs), 0)
    wcsv(f"{out}/seed_consistency.csv", cons)
    ks = [k for k in sorted(cellvec) if k[0] == 0.0]
    M = np.array([[sp(cellvec[a], cellvec[b]) for b in ks] for a in ks])
    wcsv(f"{out}/cell_consistency.csv", [dict(cell=f"N{a[1]}_wd{a[2]:g}", **{f"N{b[1]}_wd{b[2]:g}": M[i, j] for j, b in enumerate(ks)}) for i, a in enumerate(ks)])
    # ---- slingshot per-class / per-pair deltas
    sl = []
    for r in v2:
        pc = f"{PHASE_CACHE}/{r['run_id']}.pkl"
        if not os.path.isfile(pc): continue
        sc = pickle.load(open(pc, "rb"))["scalars"]; ons = [int(x) for x in sc.get("slingshot_onset_steps", "").split(";") if x]
        for probe in ("test", "train"):
            if probe not in r or not ons: continue
            P = r[probe]; st = P["steps"]; i = int(np.where(st == ons[0])[0][0]); j = min(i + 2, len(st) - 1); p0 = i - 1
            dacc = P["acc"][j] - P["acc"][p0]; dsp = np.log(P["spread"][j] / P["spread"][p0]); dmg = P["margin"][j] - P["margin"][p0]
            dS = np.log(P["S"][j][IU] / P["S"][p0][IU]); dn = np.log(np.clip(P["nc1c"][j], 1e-9, None) / np.clip(P["nc1c"][p0], 1e-9, None))
            inv = np.array([np.mean([abs(dS[k]) for k in range(45) if c in (IU[0][k], IU[1][k])]) for c in range(C)])
            row = dict(name=r["name"], probe=probe, ln=cell_of(r["cfg"])[0], N=cell_of(r["cfg"])[1], seed=r["cfg"]["seed"], onset=ons[0], delta_snapshot_clamped=int(j == i), min_class_n=int(P["cnt"].min()),
                       PR_spread=part_ratio(dsp), PR_margin=part_ratio(dmg), PR_acc=part_ratio(dacc), PR_pairS=part_ratio(dS), PR_class_involvement=part_ratio(inv),
                       top_class_spread=int(np.abs(dsp).argmax()), top_class_involvement=int(inv.argmax()), top_pair=f"{IU[0][np.abs(dS).argmax()]}-{IU[1][np.abs(dS).argmax()]}",
                       mean_abs_dlogS=float(np.abs(dS).mean()), conf_pair_abs_dlogS=float(np.abs(dS)[CONF_MASK].mean()), nonconf_pair_abs_dlogS=float(np.abs(dS)[~CONF_MASK].mean()))
            for c in range(C): row[f"dspread_{c}"] = dsp[c]; row[f"dacc_{c}"] = dacc[c]; row[f"dnc1c_{c}"] = dn[c]
            sl.append(row)
    wcsv(f"{out}/slingshot_perclass.csv", sl)
    # ---- NC1 drops
    nd = []
    for r in v2:
        for probe in ("test", "train"):
            if probe not in r: continue
            dr = nc1_drop(r[probe])
            if dr is None: continue
            c = cell_of(r["cfg"])
            nd.append(dict(name=r["name"], probe=probe, ln=c[0], N=c[1], wd=c[2], seed=r["cfg"]["seed"], drop_step=dr["step"], nc1_ratio=dr["ratio"], min_class_n=int(r[probe]["cnt"].min()),
                           PR_drop_share=part_ratio(dr["dc"]), top_class=int(dr["share"].argmax()), top_share=float(dr["share"].max()), class_step_spread_snapshots=dr["spread_snap"],
                           class_step_spread_log10=dr["spread_log10"], n_classes_within1snap=dr["n_within1"], class_steps=" ".join(map(str, dr["class_step"])),
                           **{f"share_{k}": dr["share"][k] for k in range(C)}))
    wcsv(f"{out}/nc1_drops.csv", nd)
    # ---- label noise per class + confusion structure
    ln_rows, cf_rows = [], []
    clean = {(r["cfg"]["seed"], cell_of(r["cfg"])[1]): r for r in v2 if cell_of(r["cfg"])[0] == 0 and cell_of(r["cfg"])[2] == 0.0}
    for r in v2:
        c = cell_of(r["cfg"])
        if c[0] == 0 or "train" not in r: continue
        P = r["train"]; cl = clean.get((r["cfg"]["seed"], c[1]))
        row = dict(name=r["name"], ln=c[0], N=c[1], seed=r["cfg"]["seed"], min_corrupt_per_class=int(P["n_corrupt_c"].min()), low_n=int(P["n_corrupt_c"].min() < 10))
        for k in range(C): row[f"cross_{k}"] = P["mem_cross_c"][k]; row[f"ncorr_{k}"] = int(P["n_corrupt_c"][k])
        cc = np.log10(P["mem_cross_c"] + 1); row["cross_spread_log10_std"] = float(np.nanstd(cc)); row["n_classes_crossed"] = int(np.isfinite(cc).sum())
        if cl is not None:
            co = np.log10(cl["train"]["class_onset"] + 1) if "train" in cl else np.log10(cl["test"]["class_onset"] + 1)
            row["spearman_cross_vs_clean_class_onset"] = sp(cc, co)
            # confusion structure
            cs = lambda M: ((M + M.T) / 2)[IU]
            clean_conf = cs(cl["test"]["conf"][-1]); clean_S = np.log(cl["test"]["S"][-1][IU])
            pk = int(np.argmax(r["test"]["acc"].mean(1)))
            noisy_conf_final = cs(r["test"]["conf"][-1]); noisy_conf_pk = cs(r["test"]["conf"][pk])
            pm = P["pair_mem_final"]; pn = P["pair_n_corrupt"]; sym_n = (pn + pn.T)[IU]
            symf = np.array([np.nansum([np.nan_to_num(pm[a, b]) * pn[a, b] + np.nan_to_num(pm[b, a]) * pn[b, a]]) for a, b in zip(*IU)]) / np.where(sym_n > 0, sym_n, np.nan)
            ok = sym_n >= 5
            row.update(spearman_testconf_noisyfinal_vs_clean=sp(noisy_conf_final, clean_conf), spearman_testconf_noisypeak_vs_clean=sp(noisy_conf_pk, clean_conf),
                       spearman_Sfinal_noisy_vs_clean=sp(np.log(r["test"]["S"][-1][IU]), clean_S),
                       spearman_pairmem_vs_cleanconf=sp(np.where(ok, symf, np.nan), clean_conf), spearman_pairmem_vs_cleanS=sp(np.where(ok, symf, np.nan), clean_S),
                       n_pairs_ge5=int(ok.sum()), median_corrupt_per_pair=float(np.median(sym_n)))
        ln_rows.append(row)
    wcsv(f"{out}/label_noise_perclass.csv", ln_rows)
    figures(out, R, v2, v1, rows, cons, M, ks, sl, nd, ln_rows)


def figures(out, R, v2, v1, rows, cons, M, ks, sl, nd, ln_rows):
    fd = f"{out}/figures"; cells = collections.defaultdict(list)
    for r in v2: cells[cell_of(r["cfg"])].append(r)
    # Fig 1: pair-onset matrix (cell mean) for several N at wd=0, test probe
    show = [(0.0, N, 0.0) for N in (250, 1000, 4000, 16000, 60000) if (0.0, N, 0.0) in cells]
    if not show: show = sorted(cells)[:5]
    if show:
        fig, ax = plt.subplots(2, len(show), figsize=(3.6 * len(show), 7), squeeze=False)
        for j, k in enumerate(show):
            O = np.nanmean(np.stack([np.log10(r["test"]["pair_onset"] + 1) for r in cells[k]]), 0)
            im = ax[0, j].imshow(O, cmap="viridis"); ax[0, j].set_title(f"pair onset log10 step, N={k[1]}", fontsize=8); plt.colorbar(im, ax=ax[0, j], fraction=.046)
            r = cells[k][0]; S = r["test"]["S"]; st = r["test"]["steps"]
            for p, (a, b) in enumerate(zip(*IU)):
                ax[1, j].plot(st + 1, S[:, a, b], c="r" if (a, b) in CONF_PAIRS else "gray", lw=1.2 if (a, b) in CONF_PAIRS else .4)
            ax[1, j].set_xscale("log"); ax[1, j].set_yscale("log"); ax[1, j].set_title("S_ab(t) seed0; red=confusable", fontsize=8)
        fig.tight_layout(); fig.savefig(f"{fd}/pair_onsets.png", dpi=70); plt.close(fig)
    # Fig 2: confusable vs rest and seed consistency
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    tr = [x for x in rows if x["src"] == "v2" and x["probe"] == "test" and x["ln"] == 0]
    for N, c in zip((250, 1000, 4000, 16000, 60000), plt.cm.viridis(np.linspace(0, .9, 5))):
        xs = [x for x in tr if x["N"] == N]
        ax[0].scatter([x["wd"] + 1e-5 for x in xs], [x["confusable_minus_rest"] for x in xs], color=c, s=14, label=f"N={N}")
        ax[1].scatter([x["wd"] + 1e-5 for x in xs], [x["spearman_onset_vs_finalS"] for x in xs], color=c, s=14)
    for a, t in zip(ax[:2], ("confusable minus rest mean pct-rank of pair onset (>0 = confusable later)", "Spearman(pair onset, final log S)  (negative = well-separated pairs onset earlier)")):
        a.set_xscale("symlog", linthresh=1e-5); a.axhline(0, c="k", lw=.5); a.set_title(t, fontsize=7); a.set_xlabel("wd")
    ax[0].legend(fontsize=6)
    im = ax[2].imshow(M if M.size else np.zeros((1, 1)), vmin=-.2, vmax=1, cmap="coolwarm"); ax[2].set_title("Spearman of cell-mean pair-onset vectors across (N,wd) cells (ln=0)", fontsize=7)
    ax[2].set_xticks(range(len(ks))); ax[2].set_xticklabels([f"{k[1]}/{k[2]:g}" for k in ks], rotation=90, fontsize=5); ax[2].set_yticks(range(len(ks))); ax[2].set_yticklabels([f"{k[1]}/{k[2]:g}" for k in ks], fontsize=5)
    plt.colorbar(im, ax=ax[2], fraction=.046); fig.tight_layout(); fig.savefig(f"{fd}/onset_order_consistency.png", dpi=70); plt.close(fig)
    # Fig 3: class learning onset by N + dendrograms over time
    fig = plt.figure(figsize=(17, 8)); gs = fig.add_gridspec(2, 5)
    a0 = fig.add_subplot(gs[0, :3])
    for N, col in zip((250, 1000, 4000, 16000, 60000), plt.cm.viridis(np.linspace(0, .9, 5))):
        rs = cells.get((0.0, N, 0.0), [])
        if not rs: continue
        co = np.array([r["test"]["class_onset"] for r in rs]); a0.errorbar(np.arange(C) + .03 * (N > 0) * np.log10(N) / 5, np.nanmean(co, 0), yerr=np.nanstd(co, 0), marker="o", ms=3, color=col, label=f"N={N}", capsize=2)
    a0.set_yscale("log"); a0.set_xlabel("class"); a0.set_ylabel("class learning onset step (NCM acc>=0.5 stays)"); a0.legend(fontsize=7); a0.set_title("class learning onset (wd=0, seeds mean±std, test probe)", fontsize=9)
    a1 = fig.add_subplot(gs[0, 3:])
    for r in (cells.get((0.0, 4000, 0.0), []))[:3]:
        a1.plot(r["test"]["steps"] + 1, r["test"]["acc"], lw=.6)
    a1.set_xscale("log"); a1.set_title("per-class NCM acc (N=4000 wd0, 3 seeds, 10 classes each)", fontsize=8)
    rr = (cells.get((0.0, 4000, 0.0)) or cells.get((0.0, 1000, 0.0)) or [None])[0]
    if rr is not None:
        lk = rr["test"]["linkage"]; keys = sorted(lk)
        for j, kk in enumerate(keys[:5]):
            ax = fig.add_subplot(gs[1, j]); dendrogram(lk[kk], labels=list(range(C)), ax=ax, color_threshold=0); ax.set_title(f"class-mean dendrogram step {kk}", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/class_onsets_dendrograms.png", dpi=65); plt.close(fig)
    # Fig 4: slingshot deltas + NC1 drops
    fig, ax = plt.subplots(2, 3, figsize=(16, 7))
    slt = [x for x in sl if x["probe"] == "test"]
    for x in slt:
        ax[0, 0].plot(range(C), [x[f"dspread_{c}"] for c in range(C)], marker="o", ms=3, lw=.7, label=f"N{x['N']} ln{x['ln']} s{x['seed']}")
        ax[0, 1].plot(range(C), [x[f"dacc_{c}"] for c in range(C)], marker="o", ms=3, lw=.7); ax[0, 2].plot(range(C), [x[f"dnc1c_{c}"] for c in range(C)], marker="o", ms=3, lw=.7)
    ax[0, 0].set_title("slingshot: dlog(within-class spread) per class", fontsize=8); ax[0, 1].set_title("slingshot: d(NCM acc) per class", fontsize=8); ax[0, 2].set_title("slingshot: dlog NC1_c per class", fontsize=8)
    ax[0, 0].legend(fontsize=4, ncol=2)
    ndt = [x for x in nd if x["probe"] == "test"]
    ax[1, 0].scatter([x["nc1_ratio"] for x in ndt], [x["class_step_spread_log10"] for x in ndt], c=[np.log10(x["N"]) for x in ndt], s=15); ax[1, 0].set_xscale("log")
    ax[1, 0].set_xlabel("NC1 drop factor"); ax[1, 0].set_ylabel("log10 spread of per-class drop step"); ax[1, 0].set_title("NC1 drops: sequential if spread >> snapshot spacing (~0.04)", fontsize=8)
    ax[1, 1].hist([x["PR_drop_share"] for x in ndt], bins=15); ax[1, 1].set_title("participation ratio of NC1-drop across classes (1=uniform, .1=one class)", fontsize=8)
    cnt = collections.Counter(x["top_class"] for x in ndt); ax[1, 2].bar(list(cnt), list(cnt.values())); ax[1, 2].set_title("class with the largest NC1-drop share", fontsize=8)
    fig.tight_layout(); fig.savefig(f"{fd}/slingshot_nc1_perclass.png", dpi=70); plt.close(fig)
    # Fig 5: label noise per class crossover vs N
    if ln_rows:
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for ln_, col in ((0.2, "C0"), (0.4, "C3")):
            for N in (250, 1000, 4000, 16000, 60000):
                xs = [x for x in ln_rows if x["ln"] == ln_ and x["N"] == N]
                for x in xs:
                    v = np.array([x[f"cross_{k}"] for k in range(C)], float); ax[0].scatter(np.full(C, N) * (1 + .05 * x["seed"] + (0.1 if ln_ > .3 else 0)), v, s=6, color=col, alpha=.6)
        ax[0].set_xscale("log"); ax[0].set_yscale("log"); ax[0].set_xlabel("N"); ax[0].set_ylabel("per-class memorization crossover step (blue ln.2, red ln.4)")
        for ln_, col in ((0.2, "C0"), (0.4, "C3")):
            xs = [x for x in ln_rows if x["ln"] == ln_ and "spearman_testconf_noisyfinal_vs_clean" in x]
            ax[1].scatter([x["N"] for x in xs], [x["spearman_pairmem_vs_cleanconf"] for x in xs], color=col, s=14)
            ax[2].scatter([x["N"] for x in xs], [x["spearman_testconf_noisyfinal_vs_clean"] for x in xs], color=col, s=14)
        ax[1].set_title("Spearman(pair memorization rate, clean pair confusion)", fontsize=8); ax[2].set_title("Spearman(noisy-run test confusion, clean-run test confusion)", fontsize=8)
        for a in ax[1:]: a.set_xscale("log"); a.axhline(0, c="k", lw=.5)
        fig.tight_layout(); fig.savefig(f"{fd}/label_noise_perclass.png", dpi=70); plt.close(fig)


if __name__ == "__main__": main()
