#!/bin/bash -l
#SBATCH --job-name=phenol-gp-ngp-50fs
#SBATCH --partition=gubing
#SBATCH --qos=huge
#SBATCH --account=gubing
#SBATCH --nodelist=xnode[102-103]
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-1%2

set -euo pipefail

: "${PHENOL_RUN_ROOT:?Set PHENOL_RUN_ROOT before submission}"
: "${PYQED_SOURCE_ROOT:?Set PYQED_SOURCE_ROOT before submission}"
: "${PYQED_PYTHON:?Set PYQED_PYTHON before submission}"

case "$SLURM_ARRAY_TASK_ID" in
    0) MODE=gp ;;
    1) MODE=ngp ;;
    *) echo "unexpected array task: $SLURM_ARRAY_TASK_ID" >&2; exit 2 ;;
esac

DATASET="$PHENOL_RUN_ROOT/dataset/phenol_5d_production"
OUTPUT="$PHENOL_RUN_ROOT/output/$MODE"
mkdir -p "$OUTPUT" "$PHENOL_RUN_ROOT/tmp/$MODE" "$PHENOL_RUN_ROOT/build/mps-cpp"

export PYTHONPATH="$PHENOL_RUN_ROOT/deps:$PYQED_SOURCE_ROOT"
export MPLCONFIGDIR="$PHENOL_RUN_ROOT/tmp/$MODE/matplotlib"
export TMPDIR="$PHENOL_RUN_ROOT/tmp/$MODE"
export PYQED_MPS_CPP_BUILD="$PHENOL_RUN_ROOT/build/mps-cpp"
export CXX=${PHENOL_CXX:-/usr/bin/g++}
OPENBLAS_ROOT=${PHENOL_OPENBLAS_ROOT:-/soft/mathlib/OpenBLAS/OpenBLAS-0.3.30_install}
export LD_LIBRARY_PATH="$OPENBLAS_ROOT/lib:${LD_LIBRARY_PATH:-}"
export PYQED_TDVP_BLAS_FLAGS="-I$OPENBLAS_ROOT/include -L$OPENBLAS_ROOT/lib -Wl,-rpath,$OPENBLAS_ROOT/lib -lopenblas"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYQED_NO_MATPLOTLIB=1
export PYQED_LIGHTWEIGHT_IMPORTS=1

srun "$PYQED_PYTHON" -c \
    'from pyqed.mps import tdvp_cpp as m; print("native TDVP:", m.CPP_TDVP_AVAILABLE, "BLAS:", m.CPP_TDVP_HAS_BLAS); assert m.CPP_TDVP_AVAILABLE and m.CPP_TDVP_HAS_BLAS, m.CPP_TDVP_BUILD_ERROR'

srun "$PYQED_PYTHON" \
    "$PYQED_SOURCE_ROOT/examples/namd/phenol_sa_casscf_5d_gp_control.py" \
    --mode "$MODE" \
    --checkpoint "$DATASET/model/mace_y_probability_expanded_final_polished/phenol_sa6_5d_mace_y.pt" \
    --radial-correction "$DATASET/model/radial_correction_probability_expanded/phenol_sa6_5d_radial_delta.npz" \
    --keo-cache "$DATASET/cache/quasibound_scalar_keo_65x21x23x21x17" \
    --initial-state "$DATASET/states/s1_origin_5d_quasibound_localwell_h3_corrected/phenol_sa_casscf_5d_s1_quasibound.npz" \
    --field-cache "$DATASET/cache/gp_ngp_s1_full_overlap_fields_65x21x23x21x17" \
    --potential-residual-cache "$DATASET/cache/gp_ngp_s1_full_overlap_base_qualified_65x21x23x21x17" \
    --gp-operator-cache "$DATASET/cache/gp_s1_full_overlap_operator_r16_65x21x23x21x17" \
    --ngp-operator-cache "$DATASET/cache/ngp_s1_full_overlap_operator_r16_65x21x23x21x17" \
    --overlap-rank 16 \
    --operator-rank 32 \
    --state-rank 32 \
    --time-fs 50 \
    --steps 1000 \
    --interval 5 \
    --checkpoint-steps 20 \
    --integrator hybrid \
    --tdvp2-warmup-steps 5 \
    --workers "$SLURM_CPUS_PER_TASK" \
    --output "$OUTPUT"
