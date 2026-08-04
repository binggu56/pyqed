#!/bin/bash
#SBATCH -p gubing
#SBATCH -q huge
#SBATCH -J letta-j2-1e6
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --array=0-12%4
#SBATCH -o /share/home/gubingLab/gubing/letta_scan_20260726/refine_tol1e-6_max200/%A_%a.sbatch.out
#SBATCH -e /share/home/gubingLab/gubing/letta_scan_20260726/refine_tol1e-6_max200/%A_%a.sbatch.err

set -euo pipefail

scan_root="$HOME/letta_scan_20260726"
source_root="$scan_root/pyqed_letta_cluster_20260726"
refine_root="$scan_root/refine_tol1e-6_max200"

export PYQED_ROOT="$source_root"
export PYTHON="$scan_root/env/bin/python"
export LETTA_SOURCE_RUN_ROOT="$scan_root/runs"
export LETTA_RUN_ROOT="$refine_root"
export LETTA_STARTING_DIRECTIONAL_SWEEP=auto
export LETTA_MAXIMUM_DIRECTIONAL_PASSES=200
export LETTA_GAIN_TOLERANCE=1.0e-6
export LETTA_COLOCATE_LOGS=1
export LETTA_PAIR_WORKERS=8
export LETTA_FRONTIER_WORKERS=4

exec bash \
    "$source_root/examples/mps/cluster/run_sector_projected_letta_j2_task_6x6.sh" \
    "$SLURM_ARRAY_TASK_ID" \
    "$source_root/examples/mps/cluster/j2_scan_6x6.txt"
