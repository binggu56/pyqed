#!/bin/bash -l
#SBATCH --job-name=tdi-gw-bse
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --output=tdi-gw-bse-%j.log
set -euo pipefail

# Submit from the repository root with an activated cluster environment.
TDI_REPO="${TDI_REPO:-$SLURM_SUBMIT_DIR}"
: "${TDI_RUN_ROOT:?Set TDI_RUN_ROOT to a persistent cluster scratch directory}"
TDI_PYTHON="${TDI_PYTHON:-python}"
TDI_JOB_DIR="$TDI_RUN_ROOT/tdi-${SLURM_JOB_ID}"
mkdir -p "$TDI_JOB_DIR/tmp" "$TDI_JOB_DIR/matplotlib"
export TMPDIR="$TDI_JOB_DIR/tmp" MPLCONFIGDIR="$TDI_JOB_DIR/matplotlib"
export PYTHONPATH="$TDI_REPO${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_DYNAMIC=FALSE
cd "$TDI_REPO"
srun --cpu-bind=cores "$TDI_PYTHON" examples/qchem/tdi_gw_bse/run.py \
    --output "$TDI_JOB_DIR/results" --threads "$SLURM_CPUS_PER_TASK" \
    --max-memory 64000 --on-unstable "${TDI_ON_UNSTABLE:-raise}"
