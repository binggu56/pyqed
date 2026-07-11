#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

PYTHON_BIN="${PYTHON_BIN:-/opt/anaconda3/bin/python}"
MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp}"
OUTDIR="${OUTDIR:-/private/tmp/h2o_sacasscf_11x11x7_parallel}"
N_WORKERS="${N_WORKERS:-4}"
WORKER_THREADS="${WORKER_THREADS:-1}"

export MPLCONFIGDIR

"${PYTHON_BIN}" examples/namd/h2o_casci_rovibronic.py \
  --method casscf \
  --nstates 3 \
  --initial-state 2 \
  --n-r 11 \
  --n-theta 7 \
  --r-min 1.45 \
  --r-max 2.35 \
  --theta-min 80 \
  --theta-max 130 \
  --center-r1 2.0 \
  --center-r2 2.0 \
  --center-theta 90 \
  --sigma-r 0.18 \
  --sigma-theta 10 \
  --ncas 4 \
  --nelecas 4 \
  --basis sto-3g \
  --J 1 \
  --Jz 0 \
  --tmax-fs 20 \
  --dt-fs 0.5 \
  --n-workers "${N_WORKERS}" \
  --worker-threads "${WORKER_THREADS}" \
  --rovibronic-kinetic sparse \
  --outdir "${OUTDIR}"
