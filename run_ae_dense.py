"""Dense early-window AE runs (same hyperparams/seeds as run_ae_grid.py + --dense_ae_every). CPU only.
Usage: python run_ae_dense.py [--workers 6] [--threads 2] [--every 2] [--dry]"""
import argparse, os, subprocess, time
from concurrent.futures import ThreadPoolExecutor
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/dense"
PY = "/data2/lbastian/conda/envs/latent-dyn/bin/python"
HERE = os.path.dirname(os.path.abspath(__file__))
UNTIL = {250: 2000, 1000: 4000}

def rid(N, wd, s): return f"mnist_ae_mlp_d16_s{s}_n{N}_wd{wd}"

def cmd(N, wd, s, every):
    return [PY, "train.py", "--dataset", "mnist", "--encoder_arch", "mlp", "--objective", "autoencoder",
            "--latent_norm", "none", "--latent_dim", "16", "--optimizer", "adam", "--lr", "1e-3",
            "--lr_schedule", "constant", "--batch_size", "128", "--log_schedule", "log", "--probe_train",
            "--train_eval", "full", "--isolate_snapshot_rng", "--train_steps", "20000", "--n_snapshots", "120",
            "--weight_decay", wd, "--seed", str(s), "--device", "cpu", "--train_subset", str(N),
            "--dense_ae_every", str(every), "--dense_ae_until", str(UNTIL[N]),
            "--dense_ae_dir", f"{ROOT}/runs/{rid(N, wd, s)}",
            "--artifacts_root", ROOT, "--run_id", rid(N, wd, s)]

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--every", type=int, default=2); ap.add_argument("--dry", action="store_true")
a = ap.parse_args()
t = str(a.threads)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t,
           NUMEXPR_NUM_THREADS=t, VECLIB_MAXIMUM_THREADS=t)
os.makedirs(f"{ROOT}/logs", exist_ok=True)
J = [(N, wd, s) for N in (1000, 250) for wd in ("0", "1e-4") for s in (0, 1, 2)]
def one(j):
    r = rid(*j)
    if os.path.exists(f"{ROOT}/runs/{r}/model_final.pt"): print("skip", r, flush=True); return
    if a.dry: print(" ".join(cmd(*j, a.every))); return
    t0 = time.time()
    with open(f"{ROOT}/logs/{r}.log", "w") as lf:
        rc = subprocess.call(cmd(*j, a.every), cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"{'ok' if rc == 0 else 'FAIL rc=%d' % rc} {r} {time.time()-t0:.0f}s", flush=True)
with ThreadPoolExecutor(a.workers) as ex: list(ex.map(one, J))
print("ALL DONE", flush=True)
