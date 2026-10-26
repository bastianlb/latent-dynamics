#!/usr/bin/env bash
# Re-run the whole reanalysis (CPU only). Usage: ./run_all.sh   (set REANALYSIS_OUT / LATDYN_ART to override paths)
set -e
PY=${PY:-/data2/lbastian/conda/envs/latent-dyn/bin/python}
cd "$(dirname "$0")"
ls /vol/bitbucket/lbastian/latent-dynamics/runs >/dev/null   # (re)mount NFS
$PY compute_extra.py --workers 4          # per-snapshot: offline probe, H1 bars, Mag(t) grid  (~25 min for 81 runs)
$PY final_train_eval.py                   # eval-mode train/test acc of model_final.pt
$PY final_offline_probe_full.py           # offline probe on full-train latents of final model
$PY issue1_timing.py; $PY dense_early_replay.py
$PY issue2_topology.py; $PY issue2e_h1_noise.py
$PY issue3_pmag.py; $PY issue4_gap.py; $PY issue5_facts.py; $PY issue6_offline_probe.py
