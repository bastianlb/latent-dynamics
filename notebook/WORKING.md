# Working index

Updated 2026-10-26 from read-only coeus evidence. Read [conventions](README.md)
before experiment work. Back-filled entries are historical records, not newly
executed experiments. Registry statuses and report claims retain their stated limits.

## Running / Pending

None verified. `ps -u lbastian -o pid,lstart,comm,args` on coeus at
2026-10-26 23:05 UTC showed no live Python processes; a second check of
`ps -u lbastian -o pid,lstart,comm` at 23:15 UTC also showed none. Shell watcher PIDs1049638,
1868524,1901098 and a blocked listing shell2368530 were present; they are not
training progress or queued jobs. No processes were killed. No pending launches
are known from the available evidence; there is no SLURM queue for this workflow.

## Recently completed / back-filled index

| Entry | Registered or reused runs | Evidence state |
|---|---|---|
| [2026-10-14_145137_phase1-geometry.md](2026-10-14_145137_phase1-geometry.md) | 82 | back-filled; source-specific verification limits |
| [2026-10-15_101915_v2-phase-gap.md](2026-10-15_101915_v2-phase-gap.md) | 185 | back-filled; source-specific verification limits |
| [2026-10-15_115459_v2-optimizer-sweeps.md](2026-10-15_115459_v2-optimizer-sweeps.md) | 372 | back-filled; source-specific verification limits |
| [2026-10-16_101315_v2-lr-batch.md](2026-10-16_101315_v2-lr-batch.md) | 48 | back-filled; source-specific verification limits |
| [2026-10-15_113650_v2-temperature.md](2026-10-15_113650_v2-temperature.md) | 24 | back-filled; source-specific verification limits |
| [2026-10-15_102928_v2-dataset-replication.md](2026-10-15_102928_v2-dataset-replication.md) | 123 | back-filled; source-specific verification limits |
| [2026-10-15_101915_forecasting.md](2026-10-15_101915_forecasting.md) | 584 reused training runs: 305 exploratory MNIST + 105 exploratory FM + 84 confirmatory MNIST + 90 confirmatory FM | back-filled; source-specific verification limits |
| [2026-10-16_101411_dense-optimizer.md](2026-10-16_101411_dense-optimizer.md) | 12 complete registered analysis runs + 2 incomplete development runs | back-filled; source-specific verification limits |
| [2026-10-16_160301_vector-field.md](2026-10-16_160301_vector-field.md) | 105 unique core MNIST runs verified in the summary table; replication processed counts unknown | back-filled; source-specific verification limits |
| [2026-10-19_183808_ntk-coupling.md](2026-10-19_183808_ntk-coupling.md) | 14 | back-filled; source-specific verification limits |
| [2026-10-19_183808_classifier-leadlag.md](2026-10-19_183808_classifier-leadlag.md) | 14 | back-filled; source-specific verification limits |
| [2026-10-21_153142_ae-grid-memorization.md](2026-10-21_153142_ae-grid-memorization.md) | 138 complete grid runs + 2 complete smoke runs | back-filled; source-specific verification limits |
| [2026-10-21_180323_ae-dense-warning.md](2026-10-21_180323_ae-dense-warning.md) | 24 | back-filled; source-specific verification limits |
| [2026-10-22_095841_ae-ntk.md](2026-10-22_095841_ae-ntk.md) | 33 | back-filled; source-specific verification limits |
| [2026-10-21_181511_ae-linear-theory.md](2026-10-21_181511_ae-linear-theory.md) | 12 trajectory conditions; absent from runs.csv | back-filled; source-specific verification limits |
| [2026-10-26_184507_ae-topology-rebound.md](2026-10-26_184507_ae-topology-rebound.md) | 102 reused runs in rebound_numbers.json: 72 AE +30 classifier | back-filled; source-specific verification limits |

## Blockers / verification limits

- Historical launch times, PIDs and producing code revisions mostly unknown;
  filenames use verified mtimes and entries state the date source.
- Local CPU suite unavailable: no torch/numpy/scipy/sklearn/ripser/gudhi or timeout;
  default coeus interpreter and real-data paths are not local. Remote tests were
  not launched under the read-only/no-compute brief. Local in-memory Python
  compilation, link checks, registry counts and copied JSON/CSV checks are separate.
- The anisotropy notebook template was missing. Most scientific results were
  copied from reports and not re-verified against raw arrays. Temperature has no
  dedicated quantitative verdict in the preserved reports; +24 lrB matched runs
  and the complete vfield replication manifest were not independently counted.

## Open questions

- Causal origin of AE loss-gradient coupling asymmetry and late test rise.
- Span-split residual decomposition; small-N classifier collapse mechanism.
- Classifier topology last-snapshot outlier; CIFAR AE NN-ratio partial correlation.
- Confirmatory forecasting precision and lack of external preregistration timestamps.

## Next steps

Use the corresponding linked entry before revisiting an open question. Check
existing registry/configs and controls before duplicates; design prospective
power and resource plans. Any future launch must update both this index and its
dated entry with exact command, revision, GPU/PID and paths at launch. No new
experiment or cleanup is authorized by this back-fill.
