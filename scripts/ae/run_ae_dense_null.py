"""Null + power dense AE runs (CPU only). Same hyperparams as run_ae_dense.py.
 null : N=full (60000; dense every 4 to step 7000) and N=4000 (every 4 to step 9000), wd0, seeds 0-2
 power: N=1000 / N=250 wd0 seeds 3-5 (every 2; until 4000/2000; train_steps 20000 as in the original dense runs)
Usage: python scripts/ae/run_ae_dense_null.py [--workers 6] [--threads 2] [--dry]"""
import argparse, os
from latentdyn.launch import train_cmd, run_jobs
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/dense"
# N -> (every, until, train_steps)
SET = {60000: (4, 7000, 7000), 4000: (4, 9000, 9000), 1000: (2, 4000, 20000), 250: (2, 2000, 20000)}

def rid(N, wd, s): return f"mnist_ae_mlp_d16_s{s}_n{N}_wd{wd}"

def cmd(N, wd, s):
    ev, un, ts = SET[N]
    c = train_cmd("--dataset", "mnist", "--encoder_arch", "mlp", "--objective", "autoencoder",
         "--latent_norm", "none", "--latent_dim", "16", "--optimizer", "adam", "--lr", "1e-3",
         "--lr_schedule", "constant", "--batch_size", "128", "--log_schedule", "log", "--probe_train",
         "--train_eval", "full", "--isolate_snapshot_rng", "--train_steps", str(ts), "--n_snapshots", "120",
         "--weight_decay", wd, "--seed", str(s), "--device", "cpu",
         "--dense_ae_every", str(ev), "--dense_ae_until", str(un),
         "--dense_ae_dir", f"{ROOT}/runs/{rid(N, wd, s)}",
         "--artifacts_root", ROOT, "--run_id", rid(N, wd, s))
    if N != 60000: c += ["--train_subset", str(N)]
    return c

ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=6); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
t = str(a.threads)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t,
           NUMEXPR_NUM_THREADS=t, VECLIB_MAXIMUM_THREADS=t)
os.makedirs(f"{ROOT}/logs", exist_ok=True)
J = [(N, "0", s) for N in (60000, 4000) for s in (0, 1, 2)] + [(N, "0", s) for N in (1000, 250) for s in (3, 4, 5)]
rids = [rid(*j) for j in J]
run_jobs([cmd(*j) for j in J], env=env, workers=a.workers, logs=[f"{ROOT}/logs/{r}.log" for r in rids], names=rids,
         skip=lambda i, c: os.path.exists(f"{ROOT}/runs/{rids[i]}/model_final.pt"), dry=a.dry)
print("ALL DONE", flush=True)
