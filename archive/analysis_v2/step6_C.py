#!/usr/bin/env python
"""A5.C: Fashion GBM gap (cutoff 1000 and neighbours): block-restricted + free permutation test and full-pipeline refit bootstrap, for full G and compact4/8.
usage: --dataset fashion [--workers 4] [--nperm 300] [--nrefit 300] [--cut 1000] [--target gap_15kw]"""
import step6_lib as Z
import argparse, pickle
from multiprocessing import Pool
import numpy as np
L, S, M = Z.L, Z.S, Z.M
CTX = {}


def setup(cut, tname):
    runs = L.get_runs()
    y, B0, DES, G, g, seeds, names, blk = M.get_data(runs, cut, tname)
    B1 = np.hstack([B0, DES])
    c4 = M.gvariant_cols(names[2], "compact4"); c8 = M.gvariant_cols(names[2], "compact8")
    return dict(y=y, B1=B1, G=G, g=g, blk=blk, c4=c4, c8=c8)


def perm_rows(g, blk, within_block, rng):
    """row index map: row i takes G from row idx[i] (config permutation; seeds mapped k % len)."""
    u = np.unique(g); cb = {c: blk[np.where(g == c)[0][0]] for c in u}
    pi = {}
    if within_block:
        for b in np.unique(list(cb.values())):
            mem = [c for c in u if cb[c] == b]; p = rng.permutation(mem)
            pi.update(dict(zip(mem, p)))
    else:
        pi = dict(zip(u, rng.permutation(u)))
    idx = np.zeros(len(g), int)
    for c in u:
        a, b = np.where(g == c)[0], np.where(g == pi[c])[0]
        for k, ia in enumerate(a): idx[ia] = b[k % len(b)]
    return idx


def task(a):
    kind, k = a
    c = CTX; y, g, B1, G = c["y"], c["g"], c["B1"], c["G"]
    if kind in ("perm_block", "perm_free"):
        rng = np.random.default_rng(1000 + k + (0 if kind == "perm_block" else 5000))
        idx = perm_rows(g, c["blk"], kind == "perm_block", rng); Gp = G[idx]
        out = {}
        for nm, cols in (("full", list(range(G.shape[1]))), ("c4", c["c4"]), ("c8", c["c8"])):
            p = S.loco(np.hstack([B1, Gp[:, cols]]), y, g, "gbm"); out[nm] = S.r2(y, p)
        return kind, k, out
    # refit bootstrap
    rng = np.random.default_rng(2000 + k)
    u = np.unique(g); mem = {cc: np.where(g == cc)[0] for cc in u}
    draw = rng.integers(0, len(u), len(u)); w = np.bincount(draw, minlength=len(u)).astype(float)
    wc = dict(zip(u, w)); present = [cc for cc in u if wc[cc] > 0]
    sets = {"B1": B1, "full": np.hstack([B1, G]), "c4": np.hstack([B1, G[:, c["c4"]]]), "c8": np.hstack([B1, G[:, c["c8"]]])}
    P = {n: np.zeros(len(y)) for n in sets}
    for cc in present:
        tr = np.concatenate([np.tile(mem[j], int(wc[j])) for j in present if j != cc]); te = mem[cc]
        for n, X in sets.items():
            P[n][te] = S.gbm_fit_predict(X[tr], y[tr], g[tr], X[te])
    wr = np.array([wc[cc] for cc in g]); sst = np.sum(wr * (y - np.average(y, weights=wr)) ** 2)
    sse = {n: np.sum(wr * (y - P[n]) ** 2) for n in sets}
    return kind, k, {n: (sse["B1"] - sse[n]) / sst for n in ("full", "c4", "c8")}


def init(cut, tname):
    global CTX
    CTX = setup(cut, tname)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dataset", default="fashion"); ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--nperm", type=int, default=300); ap.add_argument("--nrefit", type=int, default=300)
    ap.add_argument("--cut", type=int, default=1000); ap.add_argument("--target", default="gap_15kw")
    a = ap.parse_args()
    ctx = setup(a.cut, a.target); y, g, B1, G = ctx["y"], ctx["g"], ctx["B1"], ctx["G"]
    # observed
    sets = {"B1": B1, "full": np.hstack([B1, G]), "c4": np.hstack([B1, G[:, ctx["c4"]]]), "c8": np.hstack([B1, G[:, ctx["c8"]]])}
    P = {n: S.loco(X, y, g, "gbm") for n, X in sets.items()}
    obs = {n: S.r2(y, P[n]) - S.r2(y, P["B1"]) for n in ("full", "c4", "c8")}
    boot = {}
    for n in obs:
        st = M.CfgStats(y, {"a": P["B1"], "b": P[n]}, g)
        boot[n] = st.d_r2("a", "b", st.draw(10000, np.random.default_rng(0)))
    jobs = [("perm_block", k) for k in range(a.nperm)] + [("perm_free", k) for k in range(a.nperm)] + [("refit", k) for k in range(a.nrefit)]
    with Pool(a.workers, initializer=init, initargs=(a.cut, a.target)) as p:
        res = list(p.imap_unordered(task, jobs, chunksize=2))
    r2b = S.r2(y, P["B1"])
    rows, T = [], [f"Fashion GBM {a.target} cutoff {a.cut}: B1 R2 {r2b:+.3f}; observed dR2 (LOCO), plain config bootstrap (10000), block-restricted / free permutation (n={a.nperm}), refit bootstrap (n={a.nrefit})"]
    for n in ("full", "c4", "c8"):
        pb = np.array([o[n] for kd, k, o in res if kd == "perm_block"]) - r2b
        pf = np.array([o[n] for kd, k, o in res if kd == "perm_free"]) - r2b
        rf = np.array([o[n] for kd, k, o in res if kd == "refit"])
        pbv = (np.sum(pb >= obs[n]) + 1) / (len(pb) + 1); pfv = (np.sum(pf >= obs[n]) + 1) / (len(pf) + 1)
        q = lambda s: np.percentile(s, [2.5, 97.5])
        row = dict(dataset=Z.DATASET, target=a.target, cutoff=a.cut, variant={"full": "G full", "c4": "compact4", "c8": "compact8"}[n], r2_B1=r2b, dR2=obs[n],
                   boot_lo95=q(boot[n])[0], boot_hi95=q(boot[n])[1], perm_block_p=pbv, perm_free_p=pfv, perm_block_null_mean=pb.mean(), perm_block_null_q95=np.percentile(pb, 95),
                   perm_free_null_mean=pf.mean(), refit_n=len(rf), refit_mean=rf.mean(), refit_lo95=q(rf)[0], refit_hi95=q(rf)[1], refit_lo90=np.percentile(rf, 5), refit_frac_gt0=float(np.mean(rf > 0)))
        rows.append(row)
        T.append(f"  {row['variant']:9s} dR2 {obs[n]:+.3f}  boot95 [{row['boot_lo95']:+.3f},{row['boot_hi95']:+.3f}]  perm p block {pbv:.4f} (null mean {pb.mean():+.3f}, q95 {row['perm_block_null_q95']:+.3f}) free {pfv:.4f} (null mean {pf.mean():+.3f})"
                 f"  refit mean {rf.mean():+.3f} 95% [{row['refit_lo95']:+.3f},{row['refit_hi95']:+.3f}] P(dR2>0) {row['refit_frac_gt0']:.2f}")
    M.write_csv(f"{Z.OUT4}/C_fashion_gbm_gap_cut{a.cut}_{a.target}.csv", rows)
    open(f"{Z.OUT4}/C_fashion_gbm_gap_cut{a.cut}_{a.target}.txt", "w").write("\n".join(T)); print("\n".join(T))


if __name__ == "__main__":
    main()
