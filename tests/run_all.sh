#!/usr/bin/env bash
# Full CPU-only verification. Usage: bash tests/run_all.sh [--skip-ntk-selftest]
cd "$(dirname "$0")/.." || exit 1
ROOT=$(pwd)
PY=${PY:-/data2/lbastian/conda/envs/latent-dyn/bin/python}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
export PYTHONDONTWRITEBYTECODE=1
export AE_NTK_SELFTEST_RID=${AE_NTK_SELFTEST_RID:-mnist_ae_linear_d16_s0_n1000_wd0} AE_NTK_SELFTEST_STEP=0
FAILS=0
res() { if [ "$2" = 0 ]; then echo "PASS $1"; else echo "FAIL $1"; FAILS=$((FAILS+1)); fi; }

# 1. compile everything (in-memory, no .pyc)
$PY - <<'P'
import sys, pathlib
bad = 0; n = 0
for d in ("src", "scripts", "tests", "archive"):
    for p in sorted(pathlib.Path(d).rglob("*.py")):
        n += 1
        try: compile(p.read_text(), str(p), "exec")
        except Exception as e: bad += 1; print("COMPILE FAIL", p, e)
print(f"compiled {n} files, {bad} failed"); sys.exit(1 if bad else 0)
P
res "compile all" $?

# 2. import every latentdyn module
$PY - <<'P'
import importlib, pkgutil, sys, latentdyn
bad = 0; n = 0
for m in pkgutil.walk_packages(latentdyn.__path__, "latentdyn."):
    n += 1
    try: importlib.import_module(m.name)
    except BaseException as e: bad += 1; print("IMPORT FAIL", m.name, repr(e))
print(f"imported {n} modules, {bad} failed"); sys.exit(1 if bad else 0)
P
res "import all modules" $?

# 3. --help on every scripts/**/*.py that uses argparse or a thin wrapper
for f in $(find scripts -name '*.py' | sort); do
  case "$f" in scripts/run_ntk.py|scripts/ntk_tt.py|scripts/ntk_cifar.py) echo "SKIP --help $f (positional args, no argparse)"; continue;; esac
  if grep -qE "argparse|from latentdyn[. ][a-z_.]* import main|import main" "$f"; then
    (cd "$ROOT" && timeout 60 $PY "$f" --help >/dev/null 2>&1); res "--help $f" $?
  else echo "SKIP --help $f (no argparse/main)"; fi
done

# 4. tests
TO=${TEST_TIMEOUT:-900}
for t in test_equiv_core test_equiv_pipeline test_equiv_vfield test_equiv_ntk test_equiv_train_sweep test_metrics_selftest test_ae_ntk_selftest; do
  timeout $TO $PY tests/$t.py >${LOGDIR:-/tmp}/run_all_$t.log 2>&1; res "$t" $?
done
if [ "$1" != "--skip-ntk-selftest" ]; then timeout $TO $PY tests/test_ntk_selftest.py >${LOGDIR:-/tmp}/run_all_test_ntk_selftest.log 2>&1; res test_ntk_selftest $?; fi

echo "TOTAL FAILURES: $FAILS"; exit $FAILS
