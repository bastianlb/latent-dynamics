# latent-dynamics

Study how the **latent representations** of a small neural model (MNIST-scale classifiers and autoencoders) evolve over
training iterations: a fixed probe set of samples is tracked, latents are snapshotted on a schedule, and geometric and
topological metrics are computed over the trajectory. See `SPEC.md` for the data contract and `CLAUDE.md` for the
shared-machine (GPU/CPU/storage) rules.

## Install

```bash
conda activate /data2/lbastian/conda/envs/latent-dyn
pip install -e .            # optional TDA extras: pip install -e .[tda]
```

## Layout

```
src/latentdyn/
  config.py data.py model.py     run config / argparse defaults, datasets, models
  metrics.py dynamics.py         latent metrics, dynamics quantities
  train.py sweep.py              training entry point, grid launcher (co-located subprocesses)
  io.py stats.py gpu.py plotting.py launch.py _threads.py   shared helpers (I/O, stats, GPU pick, plots, job launcher, thread caps)
  train_aux/                     optimizer diagnostics, dense optimizer-state and dense-latent logging
  pipeline/                      analyze, plot, compare, trajectory (metrics over runs)
  vfield/                        latent vector-field decomposition, train/test, spectra, dataset variants, watcher
  ntk/                           tangent-kernel coupling analysis (lib, pair, run, tt, cifar)
  ae/                            autoencoder study: analysis, memorization, NTK, dense lead-lag, LDS loaders
scripts/                         thin wrappers around the modules above (python scripts/<name>.py == python -m ...)
scripts/ae/                      AE study scripts: run_ae_* launchers, ae_* reports and figures
tests/                           equivalence and self tests (plain python; run_all.sh runs everything)
archive/                         frozen one-off analyses (see archive/README.md)
docs/                            writing/style notes
```

## Common commands

Artifacts go under `/vol/bitbucket/lbastian/latent-dynamics/`. Pin GPUs explicitly (see CLAUDE.md).

```bash
CUDA_VISIBLE_DEVICES=3 python -m latentdyn.train --help        # train one run
CUDA_VISIBLE_DEVICES=3 python -m latentdyn.sweep --help        # launch a grid of runs
python -m latentdyn.pipeline.analyze --help                    # metrics over a run's latents
python -m latentdyn.pipeline.plot --help                       # figures
python -m latentdyn.vfield.core run --help                     # vector-field analysis (also vfield.decomp / .traintest / .spectra)
python -m latentdyn.ntk.run <run_id>                           # NTK coupling for a dense_latent run
python scripts/ae/run_ae_grid.py --dry                         # AE grid launcher (also run_ae_dense, run_ae_lrctl, run_ae_ntk_ckpt, run_ae_dense_null)
python scripts/ae/ae_lds_report.py                             # AE reports (ae_aggregate, ae_ntk_analyze, ...)
```

## Results

- `results/SUMMARY.md`: experiment blocks, key findings with sources, where outputs live.
- `results/runs.csv`: per-run (1233 runs) hyperparameters, final metrics and reproduce commands; `results/sweeps.csv` groups runs by experiment block with launchers. Regenerate with `python scripts/build_results_index.py`.

## Tests

```bash
bash tests/run_all.sh        # compile, import, --help, equivalence tests, selftests (CPU only)
```
