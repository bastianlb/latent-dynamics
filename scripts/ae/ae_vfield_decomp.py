"""Thin adapter: run the classifier linear-dynamical-system analysis (vfield_decomp partition + generator, vfield_spectra) on AE latent snapshots.
Usage: python scripts/ae/ae_vfield_decomp.py --runs 'mnist_ae_mlp_d16_*' [--probes test,train] [--modes lab,km] [--out DIR]
Modes: lab = true labels as "classes" (annotation only for the AE); km = label-free k-means(10) input grouping (50-d pixel PCA, as ae_analysis.py), applied to probe images.
Per run/probe/mode npz: steps, part_names/part_mean/part_std (as vfield_decomp), gen (global generator, label-free, stored once per probe in mode 'lab'), spectra (vfield_spectra.process outputs), recon curves.
CPU only, threads capped. Does not modify the classifier scripts; V.load_run is monkeypatched to feed the chosen grouping."""
import os
from latentdyn._threads import cap_threads
cap_threads(4)
import argparse, glob, json, time
import numpy as np
import torch
from latentdyn.vfield import core as V, decomp as VD, spectra as VS
from latentdyn.ae import analysis as AA

ROOT = "/vol/bitbucket/lbastian/latent-dynamics"
AE = f"{ROOT}/ae_dynamics/runs"; OUT = f"{ROOT}/ae_dynamics/analysis/lds"
_orig = V.load_run
_GR = {}


def mk_loader(rd, mode):
    ds = json.load(open(f"{rd}/config.json"))["dataset"]
    def loader(rd_, probe):
        Z, steps, y, noisy = _orig(rd_, probe)
        if mode == "km":
            tr = probe == "train"; X = AA.images(ds, tr)[0]
            ix = np.load(f"{rd_}/{'probe_train_index.npz' if tr else 'probe_index.npz'}")["indices"]
            if (ds, "assign") not in _GR: _GR[(ds, "assign")] = AA.group_model(ds, f"{ROOT}/ae_dynamics/analysis/_cache")
            y = _GR[(ds, "assign")](X[ix]).astype(np.int64)
        return Z, steps, y, None
    return loader


def recon_curves(rd, probe):
    sub = "latents" if probe == "test" else "latents_train"; fs = sorted(glob.glob(f"{rd}/{sub}/step_*.npz")); o = {k: [] for k in ("recon_train", "recon_test", "recon_probe")}
    for f in fs:
        with np.load(f) as z:
            for k in ("recon_train", "recon_test"): o[k].append(float(z[k]))
            o["recon_probe"].append(float(z["recon_probe_test" if probe == "test" else "recon_probe_train"]))
    return {k: np.array(v, np.float32) for k, v in o.items()}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--runs", required=True); ap.add_argument("--probes", default="test,train"); ap.add_argument("--modes", default="lab,km")
    ap.add_argument("--out", default=OUT); ap.add_argument("--nsplit", type=int, default=5); ap.add_argument("--no_spectra", action="store_true"); ap.add_argument("--full", action="store_true", help="full partition (joint KRR + kernel null; slow on CPU); default light"); ap.add_argument("--force", action="store_true"); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(); torch.set_num_threads(4); dev = torch.device("cpu"); os.makedirs(f"{a.out}/runs", exist_ok=True)
    dirs = sorted(d for g in a.runs.split(",") for d in glob.glob(f"{AE}/{g}") if os.path.isfile(f"{d}/model_final.pt"))
    if a.limit: dirs = dirs[:a.limit]
    for d in dirs:
        rid = os.path.basename(d)
        for probe in a.probes.split(","):
            if probe == "train" and not os.path.isdir(f"{d}/latents_train"): continue
            for mode in a.modes.split(","):
                fo = f"{a.out}/runs/{rid}__{probe}__{mode}.npz"
                if os.path.isfile(fo) and not a.force: continue
                t0 = time.time(); V.load_run = mk_loader(d, mode)
                Z, steps, y, _ = V.load_run(d, probe); classes = np.unique(y)
                names, P = VD.partition(torch.tensor(Z, dtype=VD.DT), steps, y, classes, dev, a.nsplit, light=not a.full)
                out = dict(steps=steps, part_names=np.array(names), part_mean=P.mean(0), part_std=P.std(0), n=np.array(Z.shape[1]), min_class=np.array(min(int((y == c).sum()) for c in classes)), n_groups=np.array(len(classes)))
                if mode == "lab":
                    G, fall = VD.generator(Z, steps); out.update(gen=G, gen_cols=np.array(VD.GEN_COLS), gen_fallback=fall)
                if not a.no_spectra:
                    sp = VS.process(d, probe, dev, a.nsplit, "given")
                    for k, v in sp.items():
                        if k not in ("steps", "n", "min_class", "seconds") and not k.startswith("traj") : out["sp_" + k] = v
                out.update({"rc_" + k: v for k, v in recon_curves(d, probe).items()}); out["seconds"] = np.array(time.time() - t0)
                np.savez_compressed(fo, **out); print(f"{rid} {probe} {mode} {time.time()-t0:.0f}s", flush=True)
    V.load_run = _orig


if __name__ == "__main__": main()
