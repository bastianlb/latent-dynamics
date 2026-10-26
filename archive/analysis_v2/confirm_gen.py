#!/usr/bin/env python
"""Confirmatory MNIST config generator (frozen by PREREG.txt). Deterministic: numpy default_rng(20261016).
Usage: confirm_gen.py table | jobs <ARTIFACTS_ROOT>  (jobs prints run_id|train.py args lines)"""
import sys
import numpy as np
K, SEED, NEW_SEEDS = 28, 20261016, (100, 101, 102)


def configs():
    rng = np.random.default_rng(SEED)
    un = rng.uniform(np.log(250), np.log(60000), K)
    uw0 = rng.random(K); uw = rng.uniform(np.log(1e-5), np.log(1e-2), K)
    ul0 = rng.random(K); ul = rng.uniform(0.05, 0.30, K)
    ulr = rng.uniform(np.log(0.3), np.log(3.0), K)
    perm = rng.permutation(K)
    out = []
    for i in range(K):
        n = int(round(np.exp(un[i])))
        full = n >= 50000
        wd = 0.0 if uw0[i] < 0.25 else float(f"{np.exp(uw[i]):.2g}")
        ln = 0.0 if ul0[i] < 0.5 else round(float(ul[i]), 2)
        lr = float(f"{1e-3 * np.exp(ulr[i]):.3g}")
        arch = "cnn" if perm[i] < K // 2 else "mlp"
        out.append(dict(cid=f"c{i + 1:02d}", n="full" if full else n, wd=wd, ln=ln, lr=lr, arch=arch))
    return out


def job_line(c, seed):
    rid = f"mnist_probe_none_d16_s{seed}_conf_{c['cid']}"
    a = (f"--dataset mnist --encoder_arch {c['arch']} --objective probe --latent_norm none --latent_dim 16 --optimizer adam --lr {c['lr']:g} "
         f"--lr_schedule constant --batch_size 128 --log_schedule log --probe_train --train_eval full --isolate_snapshot_rng --train_steps 20000 "
         f"--n_snapshots 120 " + ("" if c["n"] == "full" else f"--train_subset {c['n']} ") +
         f"--weight_decay {c['wd']:g} --label_noise {c['ln']:g} --seed {seed} --tag conf_{c['cid']}")
    return rid + "|" + a


if __name__ == "__main__":
    cs = configs()
    if sys.argv[1] == "table":
        print("cid   n      wd       label_noise  lr        arch")
        for c in cs:
            print(f"{c['cid']}  {str(c['n']):6s} {c['wd']:<8g} {c['ln']:<12g} {c['lr']:<9g} {c['arch']}")
    else:
        for c in cs:
            for s in NEW_SEEDS:
                print(job_line(c, s))
