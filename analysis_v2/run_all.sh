#!/usr/bin/env bash
# Rerunnable pipeline. Step 1 is resumable (finished per-run caches are skipped), so rerun this when more CIFAR runs finish.
# usage: bash run_all.sh [workers]
set -e
cd "$(dirname "$0")"
source /data/lbastian/miniconda3/etc/profile.d/conda.sh
conda activate /data2/lbastian/conda/envs/latent-dyn
ls /vol/bitbucket/lbastian/latent-dynamics/runs_v2 >/dev/null
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=
W=${1:-6}
python compute_metrics.py --workers $W        # per-run cache -> /vol/bitbucket/.../analysis_v2/cache
python build_tables.py                        # metrics_long.csv, runs.csv
python step2_descriptive.py                   # figures + step2_*.csv, step2_summary.txt
if [ -z "$SKIP_STEP3" ]; then python step3_predict.py --workers $((W>5?5:W)); python step3_report.py; fi   # set SKIP_STEP3=1 to only refresh CIFAR descriptives
