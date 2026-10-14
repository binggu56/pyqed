"""Compare old/new TDI contraction kernels and time the full saved operator.

No GW, RI generation or screening preparation. The auxiliary-prefix
comparison measures identical dense trial blocks; the full-rank timing
uses the optimized physical BSE operator and saved QP energies.
"""

import os

for key in (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[key] = "1"
os.environ["MPLCONFIGDIR"] = "/private/tmp/pyqed-mpl"
import argparse
import importlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from threadpoolctl import threadpool_limits
from pyqed.gw.screening import StaticScreening
from pyqed.gw import _bse_contractions

b = importlib.import_module("pyqed.gw.bse")


def plot(out, folder):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    seed_path = folder / "seed_benchmark.json"
    seed = json.loads(seed_path.read_text()) if seed_path.exists() else None
    fig, axes = plt.subplots(
        1,
        2 if seed else 1,
        figsize=(10 if seed else 6, 4),
        layout="constrained",
        squeeze=False,
    )
    ax = axes[0, 0]
    ax.bar(["Original loops", "BLAS"], [r["seconds"] for r in out["rows"][:2]])
    ax.set(
        ylabel="Wall time (seconds)",
        title=f"Four dense trials: {out['rows'][0]['seconds']/out['rows'][1]['seconds']:.2f}× speedup",
    )
    if seed:
        ax = axes[0, 1]
        ax.bar(
            ["Original loops", "Direct RI columns"], [r["seconds"] for r in seed[:2]]
        )
        ax.set(
            ylabel="Wall time (seconds)",
            title=f"14 one-entry seeds: {seed[0]['seconds']/seed[1]['seconds']:.2f}× speedup",
        )
    fig.suptitle(
        "TDI: identical 128-auxiliary slice, full orbital dimensions, 8 threads"
    )
    fig.savefig(folder / "TDI_BSE_contraction_speedup.png", dpi=160)


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--saved", type=Path, required=True)
p.add_argument("--baseline", type=Path)
p.add_argument("--plot-only", action="store_true")
p.add_argument("--seed-only", action="store_true")
a = p.parse_args()
if a.plot_only:
    plot(json.loads((a.saved / "contraction_benchmark.json").read_text()), a.saved)
    raise SystemExit
if a.baseline is None:
    p.error("--baseline is required for timing")
r = json.loads((a.saved / "report.json").read_text())
f = np.load(Path(r["checkpoint"]) / "MO_RI_factors.npy", mmap_mode="r")
o = np.load(a.saved / "qp_screened_occupied.npy", mmap_mode="r")
v = np.load(a.saved / "qp_screened_ov.npy", mmap_mode="r")
qp = np.load(a.saved / "qp_input.npz")["qp"]
no = o.shape[1]
dim = no * (f.shape[1] - no)
spec = importlib.util.spec_from_file_location("_bse_contractions", a.baseline)
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
rng = np.random.default_rng(721)
x = rng.normal(size=(2 * dim, 4)) / np.sqrt(dim)
if a.seed_only:
    seeds = np.argsort((qp[no:] - qp[:no, None]).ravel())[:14]
    x = np.zeros((2 * dim, 14))
    x[seeds, np.arange(14)] = 1
    rows = []
    results = []
    with threadpool_limits(limits=1, user_api="blas"):
        for name, kernel in [
            ("original", old),
            ("direct RI columns", _bse_contractions),
        ]:
            t = time.perf_counter()
            y, used, _ = kernel.action(f[:128], o[:128], v[:128], x, no, "full", 8)
            rows.append(
                dict(
                    kernel=name,
                    auxiliaries=128,
                    columns=14,
                    seconds=time.perf_counter() - t,
                    actual_threads=used,
                )
            )
            results.append(y)
            print(json.dumps(rows[-1]), flush=True)
        np.testing.assert_allclose(results[0], results[1], atol=1e-11, rtol=1e-10)
        t = time.perf_counter()
        y, used, _ = _bse_contractions.action(f, o, v, x, no, "full", 8)
        rows.append(
            dict(
                kernel="direct RI columns",
                auxiliaries=len(f),
                columns=14,
                seconds=time.perf_counter() - t,
                actual_threads=used,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    (a.saved / "seed_benchmark.json").write_text(json.dumps(rows, indent=2))
    raise SystemExit
out = dict(
    threads=8,
    nocc=no,
    nvir=f.shape[1] - no,
    rank=len(f),
    screening_rerun=False,
    rows=[],
)
with threadpool_limits(limits=1, user_api="blas"):
    values = []
    for name, kernel in [("original", old), ("BLAS", _bse_contractions)]:
        t = time.perf_counter()
        y, used, tile = kernel.action(f[:128], o[:128], v[:128], x, no, "full", 8)
        row = dict(
            scope="128 auxiliaries; full orbital dimensions; four dense real trials",
            kernel=name,
            seconds=time.perf_counter() - t,
            actual_threads=used,
        )
        out["rows"].append(row)
        values.append(y)
        print(json.dumps(row), flush=True)
    out["comparison_max_abs"] = float(np.max(abs(values[0] - values[1])))
    np.testing.assert_allclose(values[0], values[1], atol=1e-11, rtol=1e-10)
    static = StaticScreening(
        f, (np.load(a.saved / "qp_static_cholesky.npy", mmap_mode="r"), True), 6000
    )
    static._occupied, static._ov = o, v
    m = SimpleNamespace(
        nocc=no,
        nso=f.shape[1],
        e_qp=qp,
        e_mf=qp,
        _pair_factors=f,
        _static_screening=static,
        threads=8,
        max_memory=6000,
    )
    t = time.perf_counter()
    y = b._bse_full_matmat(m, x.astype(complex), 4)
    row = dict(
        scope="all 6430 auxiliaries; four dense real trials in complex storage",
        kernel="BLAS",
        seconds=time.perf_counter() - t,
        threading_info=m.threading_info,
    )
    out["rows"].append(row)
    print(json.dumps(row), flush=True)
    assert np.isfinite(y).all()
(a.saved / "contraction_benchmark.json").write_text(json.dumps(out, indent=2))
plot(out, a.saved)
