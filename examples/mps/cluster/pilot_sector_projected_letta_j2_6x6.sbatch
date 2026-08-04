#!/bin/bash
#SBATCH -p gubing
#SBATCH -q huge
#SBATCH -J letta-j2-pilot
#SBATCH -c 16
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH -o /share/home/gubingLab/gubing/letta_scan_20260726/logs/pilot-%j.out
#SBATCH -e /share/home/gubingLab/gubing/letta_scan_20260726/logs/pilot-%j.err

set -euo pipefail

scan_root="$HOME/letta_scan_20260726"
source_root="$scan_root/pyqed_letta_cluster_20260726"

export PYQED_ROOT="$source_root"
export PYTHON="$scan_root/env/bin/python"
export LETTA_SEED_RESULT="$source_root/seed/letta_cpp_native_advance_pass75_76.json"
export LETTA_SEED_SNAPSHOT="$source_root/seed/letta_cpp_native_advance_pass75_76.npz"
export LETTA_RUN_ROOT="$scan_root/runs"
export LETTA_PAIR_WORKERS=8
export LETTA_FRONTIER_WORKERS=4

exec bash \
    "$source_root/examples/mps/cluster/run_sector_projected_letta_j2_task_6x6.sh" \
    6 \
    "$source_root/examples/mps/cluster/j2_scan_6x6.txt" \
    --maximum-directional-passes 1
