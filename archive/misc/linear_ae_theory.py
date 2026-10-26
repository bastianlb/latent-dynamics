"""Linear-AE timescale law: SGD/Adam x init scale x seeds, mode strengths vs PCA eigvecs, theory overlay, R2-readout comparison.
CPU only. Outputs: /vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/linear_theory/"""
import os, sys
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(k, "6")
import json, numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
torch.set_num_threads(6)
from latentdyn.ae import analysis as A
OUT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/linear_theory"
d, N, BS, K = 16, 10000, 128, 16
X, _ = A.images("mnist", True)
rs = np.random.RandomState(0); X = X[rs.permutation(len(X))[:N]].astype(np.float32)
D = X.shape[1]
Xc = X - X.mean(0); lam, V = np.linalg.eigh(Xc.T @ Xc / N); o = np.argsort(-lam); lam, V = lam[o], V[:, o]
LR_SGD = 0.5 * D / (16 * 54.7)  # default dec col-norm^2 ~16 sets sharpness at init: (2/D)*16*lam0
Vk = torch.tensor(V[:, :K], dtype=torch.float32); Xt = torch.tensor(X)
Ysc = Xc @ V[:, :K]                      # PC scores for readout
ev = np.random.RandomState(1).permutation(N)[:3000]
Xev = Xt[ev]; Yev = Ysc[ev]
def grid(T, n=260):
    g = np.unique(np.round(np.geomspace(1, T, n)).astype(int)); return np.concatenate([[0], g])

def run(opt, init, seed, T):
    torch.manual_seed(seed)
    enc, dec = torch.nn.Linear(D, d), torch.nn.Linear(d, D)
    if init == "small":
        with torch.no_grad():
            for p in list(enc.parameters()) + list(dec.parameters()): p.mul_(1e-3)
    params = list(enc.parameters()) + list(dec.parameters())
    if opt == "adam": o = torch.optim.Adam(params, lr=1e-3)
    else: lr = LR_SGD; o = torch.optim.SGD(params, lr=lr)
    lg = set(grid(T).tolist()); steps, S, Zs = [], [], []
    def log(t):
        with torch.no_grad():
            S.append(((Vk.T @ dec.weight) * (enc.weight @ Vk).T).sum(1).numpy().copy())  # v_i^T Wd We v_i
            Zs.append(enc(Xev).numpy().copy()); steps.append(t)
    g = torch.Generator().manual_seed(seed + 100); perm = torch.randperm(N, generator=g); ptr = 0
    C = torch.tensor(Xc.T @ Xc / N, dtype=torch.float32); Xevc = torch.tensor(Xc[ev]); trC = torch.trace(C)
    if opt == "sgd":   # exact full-batch GD on centred data, no biases, via covariance: L = tr[(I-WdWe) C (I-WdWe)^T]/D
        def log(t):
            with torch.no_grad():
                S.append(((Vk.T @ dec.weight) * (enc.weight @ Vk).T).sum(1).numpy().copy()); Zs.append((Xevc @ enc.weight.T).numpy().copy()); steps.append(t)
    for t in range(T + 1):
        if t in lg: log(t)
        if t == T: break
        if opt == "sgd":
            Wd, We = dec.weight, enc.weight; Aa = We @ C; loss = (trC - 2 * (Wd * Aa.T).sum() + ((Aa @ We.T) * (Wd.T @ Wd)).sum()) / D
            o.zero_grad(); loss.backward(); o.step(); continue
        if opt == "adam":
            if ptr + BS > N: perm = torch.randperm(N, generator=g); ptr = 0
            xb = Xt[perm[ptr:ptr + BS]]; ptr += BS
        else: xb = Xt
        loss = ((dec(enc(xb)) - xb) ** 2).mean(); o.zero_grad(); loss.backward(); o.step()
    return np.array(steps), np.array(S), np.array(Zs), (lr if opt == "sgd" else 1e-3)

def half_time(steps, s):
    """first time s reaches s0+0.5*(fin-s0); interp in log1p time"""
    out = np.full(s.shape[1], np.nan); x = np.log1p(steps.astype(float))
    for k in range(s.shape[1]):
        s0, fin = s[0, k], s[-max(3, len(s) // 20):, k].mean()
        if fin - s0 < 0.3: continue
        thr = s0 + 0.5 * (fin - s0); i = np.where(s[:, k] >= thr)[0]
        if len(i) == 0: continue
        i = i[0]
        out[k] = steps[0] if i == 0 else float(np.expm1(x[i-1] + (thr - s[i-1, k]) / (s[i, k] - s[i-1, k] + 1e-300) * (x[i] - x[i-1])))
    return out

def slope(t, l=lam[:K]):
    ok = np.isfinite(t) & (t > 0)
    return (np.polyfit(np.log(l[ok]), np.log(t[ok]), 1)[0], int(ok.sum())) if ok.sum() >= 5 else (np.nan, int(ok.sum()))

def theory_halftime(s0k, lamk, lr, delta):
    """gradient-flow mode ODE (a=enc, b=dec scale, s=ab, delta=b^2-a^2 conserved): ds/dt=(2/D) lam (1-s) sqrt(delta^2+4 s^2); delta=0 -> logistic. returns half-way (s0->1) time in steps"""
    from scipy.integrate import solve_ivp
    tgt = s0k + 0.5 * (1 - s0k); c = 2 * lamk / D
    f = lambda t, y: [c * (1 - y[0]) * np.sqrt(delta ** 2 + 4 * y[0] ** 2)]
    ev = lambda t, y: y[0] - tgt; ev.terminal = True
    tmax = 50 * (D / (4 * lamk)) * 40 + 1
    sol = solve_ivp(f, [0, tmax], [s0k], events=ev, rtol=1e-8, atol=1e-14, method="LSODA")
    return float(sol.t_events[0][0] / lr) if len(sol.t_events[0]) else np.nan

if __name__ == "__main__":
    conds = [(o, i) for o in ("sgd", "adam") for i in ("default", "small")]
    def init_delta(i, seed):
        torch.manual_seed(seed); e, dd = torch.nn.Linear(D, d), torch.nn.Linear(d, D); m = 1e-3 if i == "small" else 1.0
        return float(m ** 2 * (dd.weight.pow(2).sum() - e.weight.pow(2).sum()) / d)
    DELTA = {(i, sd): init_delta(i, sd) for i in ("default", "small") for sd in range(3)}
    TT = {"sgd": 10000, "adam": 30000}; R = {}; summary = {}
    for (o, i) in conds:
        sl, sl_r, th_list, hts, hrs = [], [], [], [], []
        for seed in range(3):
            st, S, Zs, lr = run(o, i, seed, TT[o]); np.savez_compressed(f"{OUT}/traj_{o}_{i}_s{seed}.npz", steps=st, S=S, lam=lam[:K], lr=lr)
            ht = half_time(st, S); r2 = A.affine_readout_r2(Zs.astype(np.float64), Yev.astype(np.float64)); hr = A.rise_steps(st, r2)
            hts.append(ht); hrs.append(hr); sl.append(slope(ht)[0]); sl_r.append(slope(hr)[0])
            print(o, i, seed, "slope_sv %.2f (n=%d)  slope_R2 %.2f (n=%d)  final s: %s" % (sl[-1], slope(ht)[1], sl_r[-1], slope(hr)[1], np.round(S[-3:].mean(0)[[0, 7, 15]], 2)), flush=True)
            if seed == 0: np.save(f"{OUT}/r2_{o}_{i}_s0.npy", r2)
            if seed == 0 and o == "sgd": S0 = S[0].copy(); lr0 = lr
        R[(o, i)] = dict(ht=np.array(hts), hr=np.array(hrs))
        summary[f"{o}_{i}"] = dict(slope_sv=[float(np.mean(sl)), float(np.std(sl))], slope_r2=[float(np.nanmean(sl_r)), float(np.nanstd(sl_r))], slopes_sv=sl, slopes_r2=sl_r,
                                   ht_sv=np.array(hts).tolist(), ht_r2=np.array(hrs).tolist())
        print("==", o, i, summary[f"{o}_{i}"]["slope_sv"], summary[f"{o}_{i}"]["slope_r2"], flush=True)
    # theory for SGD: use measured s0 per mode
    lr = LR_SGD
    for i in ("default", "small"):
        th = []
        for seed in range(3):
            z = np.load(f"{OUT}/traj_sgd_{i}_s{seed}.npz"); th.append([theory_halftime(z["S"][0, k], lam[k], lr, DELTA[i, seed]) for k in range(K)])
        th = np.array(th); sls = [slope(t)[0] for t in th]
        summary[f"theory_sgd_{i}"] = dict(slope=[float(np.mean(sls)), float(np.std(sls))], ht=th.tolist())
        print("theory sgd", i, summary[f"theory_sgd_{i}"]["slope"])
    summary["lam"] = lam[:K].tolist()
    json.dump(summary, open(f"{OUT}/summary.json", "w"))
