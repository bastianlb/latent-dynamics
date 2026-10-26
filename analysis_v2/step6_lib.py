"""Amendment 5 helpers (review-3 analyses). Import FIRST (imports step5_lib -> step4_lib; caps threads; dataset from --dataset)."""
import step5_lib as M
import os, glob
import numpy as np
L, S, C, D = M.L, M.S, M.L.C, M.L.D
DATASET = M.DATASET
OUT4 = os.path.join(C.OUT_ROOT, "ext4", DATASET)
os.makedirs(OUT4, exist_ok=True)
NDRAW = 5          # random placebo draws (ridge); GBM uses the first NDRAW_GBM
NDRAW_GBM = 3
STAT_NAMES = ["logstd", "skew", "exkurt", "mean", "logmad", "logiqr"]   # 6 stat types, one random direction each


def proj_stats(x, W):
    """x: (N,16); W: (6,16) random directions (one per stat type). returns 6 stats of the 1-D projections."""
    z = x @ W.T                                  # N x 6
    out = np.empty(6)
    zc = z - z.mean(0)
    sd = zc.std(0) + 1e-12
    out[0] = np.log(sd[0])
    out[1] = np.mean((zc[:, 1] / sd[1]) ** 3)
    out[2] = np.mean((zc[:, 2] / sd[2]) ** 4) - 3
    out[3] = z[:, 3].mean()
    out[4] = np.log(np.mean(np.abs(z[:, 4] - np.median(z[:, 4]))) + 1e-12)
    q = np.percentile(z[:, 5], [25, 75]); out[5] = np.log(q[1] - q[0] + 1e-12)
    return out


def make_dirs(ndraw=NDRAW, dim=16):
    rng = np.random.default_rng(12345)
    Ws = rng.standard_normal((ndraw, 6, dim)); Ws /= np.linalg.norm(Ws, axis=2, keepdims=True)
    return Ws


def snap_files(rid, ps, step):
    return os.path.join(C.RUNS, rid, C.PROBE_DIR[ps], f"step_{int(step):07d}.npz")


def noise_block(n, k, draw):
    return np.random.default_rng(777 + draw).standard_normal((n, k))
