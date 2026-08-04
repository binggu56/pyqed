# LETTA cluster workflow

`tools/letta_hpc.py` manages reproducible LETTA scans through SSH and Slurm.
The concrete 6×6 $J_1$–$J_2$ configuration is
`examples/mps/cluster/letta_hpc_6x6.toml`.

```bash
CONFIG=examples/mps/cluster/letta_hpc_6x6.toml
HPC="python tools/letta_hpc.py --config $CONFIG"
```

## Connect once

Let the controller open its persistent SSH control connection:

```bash
$HPC connect
$HPC check
```

Enter the OTP only when `connect` prompts. Every later SSH and `rsync`
operation uses that connection in noninteractive `BatchMode`; it fails instead
of opening a hidden password or OTP prompt. Re-run `connect` if the control
connection expires.

`disconnect` closes only the reusable SSH connection. Submitted Slurm jobs
continue to run:

```bash
$HPC disconnect
```

## Submit and monitor

An ordinary submission uses the existing dated source tree configured by
`cluster.remote_source_root`:

```bash
$HPC submit --run-id j1j2-6x6-tol1e6
```

For a new code revision, bind the run to an immutable, content-addressed source
upload:

```bash
$HPC submit --sync --run-id j1j2-6x6-tol1e6-v2
```

`--sync` hashes the selected local files, uploads them under
`cluster.remote_sources_root/<sha256>`, builds them in a staging directory, and
then publishes the verified directory atomically. The current configuration
runs this cluster-side build command:

```bash
bash "$PYQED_ROOT/examples/mps/cluster/build_letta_native.sh"
```

The standalone `$HPC sync` command can pre-upload the same bundle, but
`submit --sync` is what records and uses its source ID for a new run.

Inspect once, emit JSON, or watch until no array tasks remain active:

```bash
$HPC status j1j2-6x6-tol1e6
$HPC status j1j2-6x6-tol1e6 --json
$HPC status j1j2-6x6-tol1e6 --watch 60
```

Without a run ID, `status` and the other run commands select the newest local
run.

## Adopt an existing job

Register a job submitted manually or by the earlier scripts without
resubmitting it:

```bash
$HPC adopt 11644932 \
  /share/home/gubingLab/gubing/letta_scan_20260726/refine_tol1e-6_max200 \
  --run-id current-refine-1e6
$HPC status current-refine-1e6
```

The configuration must describe the adopted job's numerical protocol and task
manifest.

## Resume versus refine

`resume` stays inside the immutable run and resubmits only inactive,
unconverged tasks with valid paired checkpoints. It refuses protocol or source
changes. The pass cap may be retained, or increased when pass-capped tasks need
more work:

```bash
$HPC resume RUN_ID --maximum-directional-passes 400
$HPC resume RUN_ID --indices 0-4,7
```

An `endpoint_rejected` task is excluded unless the retry is explicitly
authorized:

```bash
$HPC resume RUN_ID --maximum-directional-passes 400 \
  --allow-endpoint-rejected
```

`refine` creates a linked child run and requires a strictly tighter tolerance.
It first refuses active or incomplete parent tasks, validates every parent
`result.json`/`state.npz` checkpoint pair, and freezes copies under the child
run before submitting. A rejected endpoint also requires the explicit
`--allow-endpoint-rejected` override:

```bash
$HPC refine RUN_ID \
  --gain-tolerance 1e-7 \
  --maximum-directional-passes 400 \
  --new-run-id j1j2-6x6-tol1e7
```

Use `resume` to raise only the pass cap; use `refine` to change the convergence
tolerance.

## Fetch and plot

The default fetch transfers metadata, results, and logs while excluding the
large task snapshots:

```bash
$HPC fetch RUN_ID
$HPC fetch RUN_ID --checkpoints
```

`--checkpoints` additionally transfers each task's `state.npz`. Frozen source
checkpoints remain remote. Plotting accepts partial scans, so completed points
can be inspected while other tasks are still running:

```bash
$HPC plot RUN_ID
$HPC plot RUN_ID --fetch
```

## Run layout and collision protection

The configured local registry is:

```text
hpc_runs/
├── .sources/<source-hash>.files
└── RUN_ID/
    ├── run.json
    ├── jobs.jsonl
    ├── config.normalized.json
    ├── manifest.txt
    ├── submit.sbatch
    ├── status.json
    ├── summary.csv
    ├── artifacts/task_*/
    └── energy_scan.{png,pdf}
```

The matching remote run is:

```text
/share/home/gubingLab/gubing/letta_scan_20260726/hpc_runs/RUN_ID/
├── run.json, jobs.jsonl, config.normalized.json
├── manifest.txt, submit.sbatch
├── .submissions/                       # idempotent sbatch journal
├── source_checkpoints/                 # frozen for a new/refined run
└── task_NNN_j2_*/
    ├── result.json
    ├── state.npz
    ├── stdout.log
    ├── stderr.log
    └── .task.lock
```

Each worker takes a nonblocking `flock` on `.task.lock`. An accidental
overlapping submission therefore exits instead of allowing two processes to
write the same checkpoint. Resume mutations also take a local controller lock,
while the remote submission journal serializes `sbatch` and maps a stable
request ID to one job ID. Retrying after a local interruption therefore
recovers the recorded job instead of launching a duplicate array.
