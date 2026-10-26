# Preserved evidence

Named small files were copied read-only from coeus on 2026-10-26 with scp.
The [copy manifest](copy-manifest.csv) records every source path, original UTC
mtime, exact byte count and local SHA-256. It contains 25 files totaling
1,666,921 bytes. No source file exceeds 2 MB; the whole data directory, including
generated provenance CSVs, remains below 10 MB. Bulk data was not copied.

The [config provenance](config-provenance.csv) records read-only stat observations
for all 1,233 configs selected by the existing [registry](../../results/runs.csv).
The registry itself already lives in git (f426cb6) and is not duplicated here.
The [artifact provenance](artifact-provenance.csv) records direct-file metadata
for six targeted directories. The optimizer listing only inspected the two named
dump files in each of twelve immediate run directories. Sizes are logical bytes,
not recursively measured disk usage; parent trees and caches were not size-walked.
A separate report inventory was bounded to depth three and excluded run/cache,
latent and checkpoint directories.

The copied reports are unmodified historical sources. Their internal code paths,
recommendations or early results may be superseded by later sections or notebook
entries. Only claims explicitly labeled direct checks in entries were checked in
these JSON/CSV/metadata inputs; latent arrays and model checkpoints were not
recomputed. Source reports often reference large files retained remotely.

PREREG.md and dense/analysis/RESULTS.md were absent at the initially guessed
paths; actual confirmatory plans are PREREG.txt and dense optimizer findings are
REPORT.md. Preregistration hashes and external timestamps were not re-verified.
