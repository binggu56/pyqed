#!/usr/bin/env bash
set -euo pipefail

if (($# < 2)); then
    echo "usage: $0 ZERO_BASED_TASK_INDEX J2_MANIFEST [driver options ...]" >&2
    exit 2
fi

task_index=$1
manifest=$2
shift 2

: "${PYQED_ROOT:?set PYQED_ROOT to the uploaded pyqed source root}"
: "${LETTA_RUN_ROOT:?set LETTA_RUN_ROOT to the scan output directory}"

if ! [[ $task_index =~ ^[0-9]+$ ]]; then
    echo "task index must be a nonnegative integer: $task_index" >&2
    exit 2
fi
if [[ ! -f $manifest ]]; then
    echo "J2 manifest does not exist: $manifest" >&2
    exit 2
fi

j2_values=()
while IFS= read -r value; do
    j2_values+=("$value")
done < <(awk 'NF && $1 !~ /^#/ {print $1}' "$manifest")
if ((task_index >= ${#j2_values[@]})); then
    echo "task index $task_index exceeds manifest size ${#j2_values[@]}" >&2
    exit 2
fi

j2=${j2_values[$task_index]}
tag=${j2//-/m}
tag=${tag//./p}
task_tag=$(printf "task_%03d_j2_%s" "$task_index" "$tag")
run_dir="$LETTA_RUN_ROOT/$task_tag"
mkdir -p "$run_dir"

if ! command -v flock >/dev/null 2>&1; then
    echo "flock is required to protect MPS task checkpoints" >&2
    exit 2
fi
exec 9>"$run_dir/.task.lock"
if ! flock -n 9; then
    echo "task $task_index is already running: $run_dir" >&2
    exit 75
fi

if [[ ${LETTA_COLOCATE_LOGS:-0} == 1 ]]; then
    exec > >(tee -a "$run_dir/stdout.log")
    exec 2> >(tee -a "$run_dir/stderr.log" >&2)
fi

python=${PYTHON:-python}
maximum_directional_passes=${LETTA_MAXIMUM_DIRECTIONAL_PASSES:-200}
gain_tolerance=${LETTA_GAIN_TOLERANCE:-1.0e-6}
gain_tolerance_units=${LETTA_GAIN_TOLERANCE_UNITS:-energy_per_site}
if [[ $gain_tolerance_units != energy_per_site ]]; then
    echo "unsupported gain-tolerance units: $gain_tolerance_units" >&2
    exit 2
fi

export PYTHONPATH="$PYQED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "MPS 6x6 task=$task_index J2/J1=$j2 host=$(hostname) start=$(date -Iseconds)"
echo "max_passes=$maximum_directional_passes gain_tolerance_per_site=$gain_tolerance"

exec "$python" -u \
    "$PYQED_ROOT/examples/mps/converge_j1j2_mps_6x6.py" \
    --j2 "$j2" \
    --output "$run_dir/result.json" \
    --snapshot "$run_dir/state.npz" \
    --maximum-directional-passes "$maximum_directional_passes" \
    --gain-tolerance "$gain_tolerance" \
    --performance generic \
    "$@"
