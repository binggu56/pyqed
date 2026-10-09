"""Compare BSE screening at fixed, saved QP energies in separate processes.

The same native RI/RHF reference is rebuilt for both modes. This isolates
screening equivalence; it is not an all-orbital contour GW timing benchmark.
Use PYTHONPATH=. and one BLAS/OpenMP thread, with outputs outside the repo.
"""

import argparse
import contextlib
import importlib
import json
import resource
import time
import tracemalloc
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pyqed.qchem import Molecule
from pyqed.gw.gw import GW
from pyqed.gw.bse import BSE, TDA
from pyqed.gw.contour import ContourDeformation


def plot(reports, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11, 4), layout="constrained")
    labels = [r.get("label", r["screening"]) for r in reports]
    bars = axes[0].bar(
        labels,
        [r["peak_rss_mib"] / 1024 for r in reports],
        color=["#999999", "#0072B2"][: len(reports)],
    )
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set(ylabel="GiB", title="Full process peak RSS")
    axes[0].margins(y=0.2)
    x = np.arange(len(reports))
    for i, key in enumerate(["Screening", "TDA", "Full BSE"]):
        axes[1].bar(
            x + (i - 1) * 0.23,
            [r[key]["allocation_peak_mib"] for r in reports],
            0.23,
            label=key,
        )
    axes[1].set_xticks(x, labels)
    axes[1].set(ylabel="MiB", title="Traced allocation peaks")
    axes[1].margins(y=0.4)
    axes[1].legend(frameon=False, fontsize=8)
    for index, r in enumerate(reports):
        for key, marker in [("TDA", "o"), ("Full BSE", "s")]:
            axes[2].plot(
                np.arange(1, 4),
                r[key]["energies"],
                marker,
                ls="none",
                fillstyle="none" if index else "full",
                label=f"{labels[index]} {key}",
            )
    axes[2].set(
        xlabel="Excited-state index",
        ylabel="Excitation energy (Hartree)",
        title="Fixed-QP excitation energies",
    )
    axes[2].legend(frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{reports[0]['molecule'].capitalize()} / {reports[0]['basis']}: BSE screening and memory comparison"
    )
    fig.savefig(output / "bse_static_screening.png", dpi=180)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--reference",
        type=Path,
        help="Saved complete GW report from benchmark_gw_tda_memory.py",
    )
    p.add_argument("--screening", choices=["spectral", "static"], default="static")
    p.add_argument("--auxbasis", default="cc-pvdz-jkfit")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--compare", nargs=2, type=Path)
    p.add_argument("--compare-labels", nargs=2, help="Labels for the two saved reports")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare:
        reports = [json.loads(path.read_text()) for path in args.compare]
        if args.compare_labels:
            for report, label in zip(reports, args.compare_labels):
                report["label"] = label
        for key in [
            "reference_qp",
            "basis",
            "molecule",
            "geometry_angstrom",
            "auxbasis",
        ]:
            assert reports[0][key] == reports[1][key]
        errors = {
            key: float(
                np.max(
                    abs(
                        np.array(reports[0][key]["energies"])
                        - reports[1][key]["energies"]
                    )
                )
            )
            for key in ["TDA", "Full BSE"]
        }
        assert max(errors.values()) < 1e-7
        (args.output / "comparison.json").write_text(json.dumps(errors, indent=2))
        plot(reports, args.output)
        print(errors)
        return
    if args.reference is None:
        p.error("--reference is required")
    reference = json.loads(args.reference.read_text())
    if reference["preparation"] != f"PyQED RI/RHF, {args.auxbasis}":
        p.error("The saved reference must use the same native RI preparation")
    qp = np.array(reference["GW"]["energies"], dtype=float)
    assert np.all(np.isfinite(qp))
    report = dict(
        screening=args.screening,
        reference_qp=str(args.reference.resolve()),
        basis=reference["basis"],
        molecule=reference["molecule"],
        geometry_angstrom=reference["geometry_angstrom"],
        auxbasis=args.auxbasis,
        purpose="screening comparison at identical saved spectral QP energies",
    )
    g = importlib.import_module("pyqed.gw.gw")
    b = importlib.import_module("pyqed.gw.bse")
    with (
        (args.output / "calculation.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
    ):
        mol = Molecule(
            atom=report["geometry_angstrom"], unit="angstrom", basis=report["basis"]
        )
        mol.build(
            eri="ri", auxbasis=args.auxbasis, options={"workers": 1, "ri_cache": False}
        )
        mf = mol.RHF().run(tol=1e-10, max_cycle=100, verbose=0)
        assert mf.converged and len(mf.mo_energy) == len(qp)
        mf.max_memory = 128
        tracemalloc.start()
        started = time.perf_counter()
        gw = GW(mf, eta=reference.get("eta", 1e-3))
        if args.screening == "static":
            # Prepare the contour driver's actual zero-frequency factor;
            # saved QP energies isolate BSE screening from QP root selection.
            gw._contour = ContourDeformation(gw, mf.mo_energy, [gw.nocc // 2 - 1], nw=8)
        else:
            poles, vectors = gw.rpa()
            g._charge_spatial_couplings(gw, poles, vectors)
            del vectors
        gw.e_qp = qp.copy()
        report["Screening"] = dict(
            seconds=time.perf_counter() - started,
            allocation_peak_mib=tracemalloc.get_traced_memory()[1] / 2**20,
        )
        tracemalloc.stop()

        def forbidden(*args, **kwargs):
            raise AssertionError("BSE rebuilt spectral screening")

        with patch.object(g, "rpa", forbidden), patch.object(b, "rpa", forbidden):
            for key, cls in [("TDA", TDA), ("Full BSE", BSE)]:
                print(f"Starting {key}", flush=True)
                tracemalloc.start()
                started = time.perf_counter()
                obj = cls(gw).run(
                    nroots=3,
                    low_rank=True,
                    batch_columns=2,
                    tol=1e-8,
                    max_cycle=200,
                    max_space=80,
                )
                peak = tracemalloc.get_traced_memory()[1] / 2**20
                tracemalloc.stop()
                assert obj.info["converged"]
                if args.screening == "static":
                    assert obj._M is None and obj.e_rpa is None
                report[key] = dict(
                    seconds=time.perf_counter() - started,
                    allocation_peak_mib=peak,
                    energies=obj.e.tolist(),
                    residuals=np.asarray(obj.info["residual_norms"]).tolist(),
                    reused_static=obj._static_screening is not None,
                )
                print(f"Finished {key}: {report[key]}", flush=True)
                (args.output / "report.json").write_text(json.dumps(report, indent=2))
    report["peak_rss_mib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    plot([report], args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
