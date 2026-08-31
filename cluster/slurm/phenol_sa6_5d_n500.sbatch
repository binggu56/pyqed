#!/bin/bash -l
#SBATCH --job-name=phenol-sa6-n500
#SBATCH --partition=gubing
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=600G
#SBATCH --time=7-00:00:00

set -euo pipefail

: "${PHENOL_RUN_ROOT:?Set PHENOL_RUN_ROOT before submission}"
: "${PYQED_SOURCE_ROOT:?Set PYQED_SOURCE_ROOT before submission}"
: "${PYQED_PYTHON:?Set PYQED_PYTHON before submission}"

mkdir -p "$PHENOL_RUN_ROOT/output" "$PHENOL_RUN_ROOT/tmp"
PHENOL_OUTPUT=${PHENOL_OUTPUT:-$PHENOL_RUN_ROOT/output}
PHENOL_SAMPLES=${PHENOL_SAMPLES:-500}
PHENOL_WORKERS=${PHENOL_WORKERS:-$SLURM_CPUS_PER_TASK}
mkdir -p "$PHENOL_OUTPUT"
export PYTHONPATH="$PYQED_SOURCE_ROOT"
export MPLCONFIGDIR="$PHENOL_RUN_ROOT/tmp/matplotlib"
export TMPDIR="$PHENOL_RUN_ROOT/tmp"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

srun "$PYQED_PYTHON" \
    "$PYQED_SOURCE_ROOT/examples/namd/phenol_sa_casscf_5d_generate.py" \
    --database "$PHENOL_RUN_ROOT/database/electronic.sqlite" \
    --output "$PHENOL_OUTPUT" \
    --source-run-id phenol-sa6-5d-pilot-v1-s61-n128 \
    --samples "$PHENOL_SAMPLES" \
    --workers "$PHENOL_WORKERS" \
    --seed 61 \
    --quiet
