"""Poll for idle GPUs (<500 MiB, <=2% util, never GPU 0, at most 3 distinct GPUs) and start vfield.py workers on them until the work list is done.
Usage: python vfield_watch.py [--workers_per_gpu 3] [--poll 300] [--runs ...]"""
import subprocess, time, os, sys, argparse, glob
ap = argparse.ArgumentParser(); ap.add_argument("--workers_per_gpu", type=int, default=3); ap.add_argument("--poll", type=int, default=300)
ap.add_argument("--runs", default="mnist_probe_none_d16_s[012]_phase_n*_wd0,mnist_probe_none_d16_s[012]_phase_n*_wd1e-4,mnist_probe_none_d16_s[012]_phase_n*_wd1e-3,mnist_probe_none_d16_s[012]_phase_n*_wd1e-2,mnist_probe_none_d16_s[012]_phase_n*_wd3e-2,mnist_probe_none_d16_s*_gap_*")
ap.add_argument("--out", default="/vol/bitbucket/lbastian/latent-dynamics/dynamics_v2/vfield"); ap.add_argument("--max_gpus", type=int, default=3)
a = ap.parse_args(); mine = {}   # gpu -> list of Popen
env0 = dict(os.environ, OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", MKL_NUM_THREADS="4", NUMEXPR_NUM_THREADS="4", VECLIB_MAXIMUM_THREADS="4")
def smi():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    return {int(l.split(",")[0]): (int(l.split(",")[1]), int(l.split(",")[2])) for l in out.strip().splitlines()}
def remaining():
    import glob
    n = 0
    for g in a.runs.split(","):
        for d in glob.glob(f"/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs/{g}"):
            rid = os.path.basename(d)
            for p in ("test", "train"):
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
    st = smi(); active = len(mine)
    if left > 0:
        for g, (mem, util) in sorted(st.items()):
            if g == 0 or g in mine or active >= a.max_gpus: continue
            if mem < 500 and util <= 2:
                print(time.strftime("%H:%M:%S"), "launching on GPU", g, st[g], flush=True)
                env = dict(env0, CUDA_VISIBLE_DEVICES=str(g)); procs = []
                for w in range(a.workers_per_gpu):
                    procs.append(subprocess.Popen([sys.executable, "/homes/lbastian/latent-dynamics/vfield.py", "run", "--device", "cuda", "--runs", a.runs, "--out", a.out], env=env,
                                                  stdout=open(f"{a.out}/log_gpu{g}_w{w}.txt", "a"), stderr=subprocess.STDOUT)); time.sleep(20)
                mine[g] = procs; active += 1
    time.sleep(a.poll)
