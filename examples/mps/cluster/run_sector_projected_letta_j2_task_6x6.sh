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

mapfile -t j2_values < <(awk 'NF && $1 !~ /^#/ {print $1}' "$manifest")
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
    echo "flock is required to protect LETTA task checkpoints" >&2
    exit 2
fi
exec 9>"$run_dir/.task.lock"
if ! flock -n 9; then
    echo "task $task_index is already running: $run_dir" >&2
    exit 75
fi

python=${PYTHON:-python}
source_result=${LETTA_SEED_RESULT:-}
source_snapshot=${LETTA_SEED_SNAPSHOT:-}
if [[ -n ${LETTA_SOURCE_RUN_ROOT:-} ]]; then
    source_dir="$LETTA_SOURCE_RUN_ROOT/$task_tag"
    if [[ ! -f $source_dir/result.json || ! -f $source_dir/state.npz ]]; then
        shopt -s nullglob
        source_candidates=(
            "$LETTA_SOURCE_RUN_ROOT"/"$(printf "task_%03d_j2_" "$task_index")"*
        )
        shopt -u nullglob
        complete_candidates=()
        for candidate in "${source_candidates[@]}"; do
            if [[ -f $candidate/result.json && -f $candidate/state.npz ]]; then
                complete_candidates+=("$candidate")
            fi
        done
        if ((${#complete_candidates[@]} != 1)); then
            echo "expected one source checkpoint for task $task_index, found ${#complete_candidates[@]}" >&2
            exit 2
        fi
        source_dir=${complete_candidates[0]}
    fi
    source_result="$source_dir/result.json"
    source_snapshot="$source_dir/state.npz"
fi
if [[ ! -f $source_result || ! -f $source_snapshot ]]; then
    echo "LETTA source checkpoint is incomplete: $source_result $source_snapshot" >&2
    exit 2
fi

starting_directional_sweep=${LETTA_STARTING_DIRECTIONAL_SWEEP:-0}
if [[ $starting_directional_sweep == auto ]]; then
    starting_directional_sweep=$(
        "$python" -c \
            'import json, sys; print(json.load(open(sys.argv[1]))["result"]["next_directional_sweep"])' \
            "$source_result"
    )
fi
maximum_directional_passes=${LETTA_MAXIMUM_DIRECTIONAL_PASSES:-40}
gain_tolerance=${LETTA_GAIN_TOLERANCE:-1.0e-4}
gain_tolerance_units=${LETTA_GAIN_TOLERANCE_UNITS:-energy_per_site}
if [[ $gain_tolerance_units != energy_per_site ]]; then
    echo "unsupported gain-tolerance units: $gain_tolerance_units" >&2
    exit 2
fi

if [[ ${LETTA_COLOCATE_LOGS:-0} == 1 ]]; then
    exec > >(tee -a "$run_dir/stdout.log")
    exec 2> >(tee -a "$run_dir/stderr.log" >&2)
fi

export PYTHONPATH="$PYQED_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

"$python" -c \
    'import importlib; [importlib.import_module("pyqed.letta." + name) for name in ("_conditional_gauge_cpp", "_copy_einsum_cpp", "_physical_blocks_cpp", "_support_kernels_cpp")]'

echo "LETTA 6x6 task=$task_index J2/J1=$j2 host=$(hostname) start=$(date -Iseconds)"
echo "source=$source_result start_sweep=$starting_directional_sweep max_passes=$maximum_directional_passes gain_tolerance_per_site=$gain_tolerance"

exec "$python" -u \
    "$PYQED_ROOT/examples/mps/converge_sector_projected_letta_two_site_batched_6x6.py" \
    --source-result "$source_result" \
    --source-snapshot "$source_snapshot" \
    --j2 "$j2" \
    --output "$run_dir/result.json" \
    --snapshot "$run_dir/state.npz" \
    --starting-directional-sweep "$starting_directional_sweep" \
    --maximum-directional-passes "$maximum_directional_passes" \
    --gain-tolerance "$gain_tolerance" \
    --pair-operator-workers "${LETTA_PAIR_WORKERS:-4}" \
    --frontier-workers "${LETTA_FRONTIER_WORKERS:-2}" \
    "$@"
