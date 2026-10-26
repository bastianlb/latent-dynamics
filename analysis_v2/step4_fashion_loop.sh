#!/bin/bash
# poll every 10 min; compute metric caches for finished fashion runs; exit when QUEUE DONE and 105 caches exist
PY=/data2/lbastian/conda/envs/latent-dyn/bin/python
RUNS=/vol/bitbucket/lbastian/latent-dynamics/runs_v2/runs
LOG=/vol/bitbucket/lbastian/latent-dynamics/runs_v2/logs/_queue_FM.log
CACHE=/vol/bitbucket/lbastian/latent-dynamics/analysis_v2/cache
cd /homes/lbastian/latent-dynamics/analysis_v2
while true; do
  ls $RUNS > /dev/null
  todo=""
  for d in $RUNS/fashion_mnist_probe_none_d16_s*_phase_n* $RUNS/fashion_mnist_probe_none_d16_s*_gap_n*; do
    [ -f $d/model_final.pt ] || continue
    r=$(basename $d); [ -f $CACHE/$r.npz ] || todo="$todo $r"
  done
  if [ -n "$todo" ]; then
    OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 $PY compute_metrics.py --workers 4 --runs $todo > /vol/bitbucket/lbastian/latent-dynamics/analysis_v2/ext/compute_metrics_FM.log 2>&1
  fi
  nok=$(grep -c "^OK" $LOG); nc=$(ls $CACHE | grep -c "^fashion_mnist_probe_none_d16_s.*_\(phase\|gap\)_n")
  echo "$(date +%H:%M) OK=$nok caches=$nc"
  if grep -q "QUEUE DONE" $LOG && [ "$nc" -ge 105 ]; then echo "FM READY"; exit 0; fi
  sleep 600
done
