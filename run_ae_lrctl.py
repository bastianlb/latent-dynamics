"""Per-block lr control for the late test-error rise (Heckel & Yilmaz rival): enc lr x0.1 or dec lr x0.1, MNIST AE mlp d16 wd0. CPU. Output ntk/lrctl/runs."""
import os, subprocess, time, sys
from concurrent.futures import ThreadPoolExecutor
ROOT = "/vol/bitbucket/lbastian/latent-dynamics/ae_dynamics/ntk/lrctl"; PY = "/data2/lbastian/conda/envs/latent-dyn/bin/python"; HERE = os.path.dirname(os.path.abspath(__file__))
def rid(N, tag, s): return f"mnist_ae_mlp_d16_s{s}_n{N}_wd0_{tag}"
CONDS = {"dec0.1": ("1", "0.1"), "enc0.1": ("0.1", "1")}
def cmd(N, tag, s):
    e, d = CONDS[tag]
    return [PY, "train.py", "--dataset", "mnist", "--encoder_arch", "mlp", "--objective", "autoencoder", "--latent_norm", "none", "--latent_dim", "16", "--optimizer", "adam", "--lr", "1e-3",
            "--lr_schedule", "constant", "--batch_size", "128", "--log_schedule", "log", "--probe_train", "--train_eval", "full", "--isolate_snapshot_rng", "--train_steps", "20000", "--n_snapshots", "120",
            "--weight_decay", "0", "--seed", str(s), "--device", "cpu", "--train_subset", str(N), "--lr_mult_enc", e, "--lr_mult_dec", d, "--artifacts_root", ROOT, "--run_id", rid(N, tag, s)]
nw = int(sys.argv[1]) if len(sys.argv) > 1 else 4
env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2", NUMEXPR_NUM_THREADS="2")
os.makedirs(f"{ROOT}/logs", exist_ok=True)
J = [(N, t, s) for s in (0, 1, 2) for N in (250, 1000) for t in CONDS]
def one(j):
    r = rid(*j)
    if os.path.exists(f"{ROOT}/runs/{r}/model_final.pt"): return
    t0 = time.time()
    with open(f"{ROOT}/logs/{r}.log", "w") as lf: rc = subprocess.call(cmd(*j), cwd=HERE, env=env, stdout=lf, stderr=subprocess.STDOUT)
    print(f"{'ok' if rc == 0 else 'FAIL'} {r} {time.time()-t0:.0f}s", flush=True)
with ThreadPoolExecutor(nw) as ex: list(ex.map(one, J))
print("ALL DONE", flush=True)
