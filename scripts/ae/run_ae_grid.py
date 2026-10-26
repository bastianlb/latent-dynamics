"""Bounded-concurrency CPU queue for the AE dynamics grid (see ae_dynamics/README.md).
Usage: python scripts/ae/run_ae_grid.py [--workers 7] [--threads 3] [--dry]
A run is done iff <root>/runs/<run_id>/model_final.pt exists (latentdyn.train writes it last); done runs are skipped.
"""
import argparse, os
from latentdyn.launch import train_cmd, run_jobs

ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics"
SEEDS = (0, 1, 2)


def jobs():
    J = []  # (kind, d, N, wd, seed)
    for N in ("full", 4000, 1000, 250):
        for wd in ("0", "1e-4"):
            J += [("mlp", 16, N, wd, s) for s in SEEDS]
        J += [("linear", 16, N, "0", s) for s in SEEDS]
    for d in (8, 64):
        for N in ("full", 1000):
            J += [("mlp", d, N, "0", s) for s in SEEDS]
    return J


DS = "mnist"  # set by --dataset (mnist | fashion_mnist | cifar10)
DEVICE = "cpu"


def jobs_cifar():
    """CIFAR-10 grid (conv AE 'conv', linear AE 'linear'): 24 + 12 + 6 = 42 runs."""
    J = []
    for N in ("full", 4000, 1000, 250):
        for wd in ("0", "1e-4"):
            J += [("conv", 64, N, wd, s) for s in SEEDS]
        J += [("linear", 64, N, "0", s) for s in SEEDS]
    for N in ("full", 1000):
        J += [("conv", 16, N, "0", s) for s in SEEDS]
    return J


def run_id(kind, d, N, wd, seed):
    return f"{DS}_ae_{kind}_d{d}_s{seed}_n{N}_wd{wd}"


def cmd(kind, d, N, wd, seed):
    c = train_cmd("--dataset", DS, "--encoder_arch", "convae" if kind == "conv" else "mlp", "--objective", "autoencoder",
         "--latent_norm", "none", "--latent_dim", str(d), "--optimizer", "adam", "--lr", "1e-3",
         "--lr_schedule", "constant", "--batch_size", "128", "--log_schedule", "log", "--probe_train",
         "--train_eval", "full", "--isolate_snapshot_rng", "--train_steps", "20000", "--n_snapshots", "120",
         "--weight_decay", wd, "--seed", str(seed), "--device", DEVICE,
         "--artifacts_root", ROOT, "--run_id", run_id(kind, d, N, wd, seed))
    if N != "full":
        c += ["--train_subset", str(N)]
    if kind == "linear":
        c += ["--linear_ae"]
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--dataset", default="mnist", choices=["mnist", "fashion_mnist", "cifar10"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--shard", default="0/1", help="k/m: run jobs j with j%%m==k")
    ap.add_argument("--only", default=None, help="substring filter on run_id")
    a = ap.parse_args()
    global DS, DEVICE; DS = a.dataset; DEVICE = a.device
    env = dict(os.environ, **({} if a.device != "cpu" else {"CUDA_VISIBLE_DEVICES": ""}), OMP_NUM_THREADS=str(a.threads), OPENBLAS_NUM_THREADS=str(a.threads),
               MKL_NUM_THREADS=str(a.threads), NUMEXPR_NUM_THREADS=str(a.threads), VECLIB_MAXIMUM_THREADS=str(a.threads))
    os.makedirs(f"{ROOT}/logs", exist_ok=True)
    J = jobs_cifar() if DS == "cifar10" else jobs()
    k, m = map(int, a.shard.split("/")); J = [j for i, j in enumerate(J) if i % m == k]
    if a.only: J = [j for j in J if a.only in run_id(*j)]
    print(f"{len(J)} jobs", flush=True)

    rids = [run_id(*j) for j in J]
    run_jobs([cmd(*j) for j in J], env=env, workers=a.workers, logs=[f"{ROOT}/logs/{r}.log" for r in rids], names=rids,
             skip=lambda i, c: os.path.exists(f"{ROOT}/runs/{rids[i]}/model_final.pt"), dry=a.dry)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
