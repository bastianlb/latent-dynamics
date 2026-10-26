"""Poll for idle GPUs (<500 MiB, <=2% util, never GPU 0, at most 3 distinct GPUs) and start vfield workers on them until the work list is done.
Usage: python scripts/vfield_watch.py [--target vfield|decomp] [--workers_per_gpu 3] [--poll 300] [--runs ...]
(--target decomp replaces the old vfield_decomp_watch.py: launches latentdyn.vfield.decomp with --probes test,train)"""
import subprocess, time, os, sys, argparse, glob
from latentdyn import gpu

RUNS_DEFAULT = "mnist_probe_none_d16_s[012]_phase_n*_wd0,mnist_probe_none_d16_s[012]_phase_n*_wd1e-4,mnist_probe_none_d16_s[012]_phase_n*_wd1e-3,mnist_probe_none_d16_s[012]_phase_n*_wd1e-2,mnist_probe_none_d16_s[012]_phase_n*_wd3e-2,mnist_probe_none_d16_s*_gap_*"
OUT_DEFAULT = {"vfield": "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield", "decomp": "/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield/decomp"}


def probes_for(target, rid):
    if target == "vfield": return ("test", "train")
    return ("test", "train") if "_gap_" in rid else ("test",)


def worker_cmd(target, a):
    if target == "vfield":
        return [sys.executable, "-m", "latentdyn.vfield.core", "run", "--device", "cuda", "--runs", a.runs, "--out", a.out]
    return [sys.executable, "-m", "latentdyn.vfield.decomp", "run", "--device", "cuda", "--runs", a.runs, "--probes", "test,train", "--out", a.out]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--target", choices=["vfield", "decomp"], default="vfield")
    ap.add_argument("--workers_per_gpu", type=int, default=3); ap.add_argument("--poll", type=int, default=300)
    ap.add_argument("--runs", default=RUNS_DEFAULT)
    ap.add_argument("--out", default=None); ap.add_argument("--max_gpus", type=int, default=3)
    a = ap.parse_args(); mine = {}   # gpu -> list of Popen
    if a.out is None: a.out = OUT_DEFAULT[a.target]
    env0 = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")

    def remaining():
        n = 0
        for g in a.runs.split(","):
            for d in glob.glob(f"/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/{g}"):
                rid = os.path.basename(d)
                for p in probes_for(a.target, rid):
                    if not os.path.isfile(f"{a.out}/runs/{rid}_{p}.npz"): n += 1
        return n

    while True:
        for g in list(mine):
            mine[g] = [p for p in mine[g] if p.poll() is None]
            if not mine[g]: del mine[g]
        if not mine:   # no live workers of ours: claims without an output are stale (a worker died) -> release them
            import shutil
            for c in glob.glob(f"{a.out}/claims/*"):
                if not os.path.isfile(f"{a.out}/runs/{os.path.basename(c)}.npz"): shutil.rmtree(c, ignore_errors=True)
        left = remaining()
        if left == 0 and not mine: print("all done", flush=True); break
        st = gpu.smi(); active = len(mine)
        if left > 0:
            idle = set(gpu.idle_gpus(mem_mib=500, util_pct=2, exclude=(0,)))
            for g in sorted(st):
                if g == 0 or g in mine or active >= a.max_gpus: continue
                if g in idle:
                    print(time.strftime("%H:%M:%S"), "launching on GPU", g, st[g], flush=True)
                    env = dict(env0, CUDA_VISIBLE_DEVICES=str(g)); procs = []
                    for w in range(a.workers_per_gpu):
                        procs.append(subprocess.Popen(worker_cmd(a.target, a), env=env,
                                                      stdout=open(f"{a.out}/log_gpu{g}_w{w}.txt", "a"), stderr=subprocess.STDOUT)); time.sleep(20)
                    mine[g] = procs; active += 1
        time.sleep(a.poll)


if __name__ == "__main__":
    main()
