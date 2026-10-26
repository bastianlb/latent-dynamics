"""Re-run AE grid configs (identical flags/seeds as run_ae_dense.py) with model+Adam checkpoints at chosen epochs. CPU only.
Usage: python scripts/ae/run_ae_ntk_ckpt.py [--workers 4] [--threads 2] [--dry]"""
import argparse, os
from latentdyn.launch import train_cmd, run_jobs
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk"
EPOCHS = {250: [1, 3, 5, 10, 20, 30, 45, 60, 90, 120, 170, 250, 500, 1000, 2000],
          1000: [1, 3, 5, 10, 20, 30, 45, 60, 90, 120, 170, 250, 500, 1000, 2000],
          "full": [1, 2, 5, 10, 20, 30, 42]}
def steps(N):
    n = 60000 if N == "full" else N
    return sorted({min(19999, max(1, round(e * n / 128))) for e in EPOCHS[N]})
def rid(kind, N, wd, s): return f"mnist_ae_{kind}_d16_s{s}_n{N}_wd{wd}"
def cmd(kind, N, wd, s):
    c = train_cmd("--dataset", "mnist", "--encoder_arch", "mlp", "--objective", "autoencoder",
         "--latent_norm", "none", "--latent_dim", "16", "--optimizer", "adam", "--lr", "1e-3",
         "--lr_schedule", "constant", "--batch_size", "128", "--log_schedule", "log", "--probe_train",
         "--train_eval", "full", "--isolate_snapshot_rng", "--train_steps", "20000", "--n_snapshots", "120",
         "--weight_decay", wd, "--seed", str(s), "--device", "cpu",
         "--log_latents_every", "1000000000", "--ckpt_steps", ",".join(map(str, steps(N))),
         "--dense_dir", f"{ROOT}/runs/{rid(kind,N,wd,s)}/dense",
         "--artifacts_root", ROOT, "--run_id", rid(kind, N, wd, s))
    if N != "full": c += ["--train_subset", str(N)]
    if kind == "linear": c += ["--linear_ae"]
    return c
ap = argparse.ArgumentParser()
ap.add_argument("--workers", type=int, default=4); ap.add_argument("--threads", type=int, default=2)
ap.add_argument("--dry", action="store_true"); a = ap.parse_args()
t = str(a.threads)
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=t, OPENBLAS_NUM_THREADS=t, MKL_NUM_THREADS=t,
           NUMEXPR_NUM_THREADS=t, VECLIB_MAXIMUM_THREADS=t)
J = []
for s in (0, 1, 2):
    J += [("mlp", N, wd, s) for N in (250, 1000) for wd in ("0", "1e-4")]
    J += [("linear", N, "0", s) for N in (250, 1000)]
    J += [("mlp", "full", "0", s)]
rids = [rid(*j) for j in J]
run_jobs([cmd(*j) for j in J], env=env, workers=a.workers, logs=[f"{ROOT}/logs/{r}.log" for r in rids], names=rids,
         skip=lambda i, c: os.path.exists(f"{ROOT}/runs/{rids[i]}/model_final.pt"), dry=a.dry)
print("ALL DONE", flush=True)
