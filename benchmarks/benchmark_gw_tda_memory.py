"""Acene memory check; run with PYTHONPATH=. and one BLAS thread.

Uses idealized fused hexagons, CD/RHF (or an explicit PySCF RI/RHF preparation
reference), complete TDH screening and three Davidson TDA roots. Allocation peaks exclude memory-mapped file pages;
process peak RSS includes them. Legacy storage figures are estimates, not
measurements. Writes JSON and a reproducible diagnostic figure outside the repo.
Select naphthalene, anthracene or tetracene with --molecule.
"""

import argparse
import contextlib
import json
import resource
import time
import tracemalloc
import threading
import importlib
from pathlib import Path

import numpy as np
from pyqed.qchem import Molecule
from pyqed.qchem.tddft import TDA
from pyqed.gw.gw import GW
from pyqed.gw.bse import TDA as BSETDA
from pyqed.gw.screening import FactorizedCouplings


@contextlib.contextmanager
def resident_profile(report):
    """Sample process RSS through preparation and nested GW stages."""
    import psutil
    from unittest.mock import patch

    process = psutil.Process()
    phase = ["Preparation"]
    samples = []
    stop = threading.Event()
    started = time.perf_counter()

    def sample():
        while not stop.is_set():
            samples.append(
                [
                    time.perf_counter() - started,
                    phase[0],
                    process.memory_info().rss / 2**20,
                ]
            )
            stop.wait(0.05)

    def wrap(function, name):
        def run(*args, **kwargs):
            previous, phase[0] = phase[0], name
            try:
                return function(*args, **kwargs)
            finally:
                phase[0] = previous

        return run

    module = importlib.import_module("pyqed.gw.gw")
    with contextlib.ExitStack() as stack:
        for name, label in [
            ("_build_spatial_pair_factors_mo", "GW MO factors"),
            ("_charge_rpa", "GW diagonalization"),
            ("_charge_spatial_couplings", "GW coupling preparation"),
            ("_solve_qp_energies", "GW quasiparticle solve"),
        ]:
            stack.enter_context(
                patch.object(module, name, wrap(getattr(module, name), label))
            )
        from pyqed.gw.contour import ContourDeformation

        for name, label in [
            ("__init__", "GW contour preparation"),
            ("evaluate", "GW contour QP"),
        ]:
            stack.enter_context(
                patch.object(
                    ContourDeformation,
                    name,
                    wrap(getattr(ContourDeformation, name), label),
                )
            )
        thread = threading.Thread(target=sample, daemon=True)
        thread.start()
        try:
            yield phase
        finally:
            stop.set()
            thread.join()
            report["rss_samples"] = samples
            report["stage_peak_rss_mib"] = {
                name: max(s[2] for s in samples if s[1] == name)
                for name in dict.fromkeys(s[1] for s in samples)
            }


def geometry(rings=2):
    length = 1.4
    carbons = []
    for center in (np.arange(rings) - (rings - 1) / 2) * np.sqrt(3) * length:
        for angle in np.arange(6) * np.pi / 3 + np.pi / 6:
            point = np.array(
                [center + length * np.cos(angle), length * np.sin(angle), 0.0]
            )
            if not any(np.linalg.norm(point - old) < 1e-6 for old in carbons):
                carbons.append(point)
    atoms = [("C", point) for point in carbons]
    for point in carbons:
        neighbors = [
            other
            for other in carbons
            if abs(np.linalg.norm(other - point) - length) < 1e-6
        ]
        if len(neighbors) == 2:
            direction = sum(point - other for other in neighbors)
            atoms.append(("H", point + 1.09 * direction / np.linalg.norm(direction)))
    return "; ".join(f"{atom} {x:.10f} {y:.10f} {z:.10f}" for atom, (x, y, z) in atoms)


def plot(report, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    keys = [k for k in ["GW", "TDA", "BSE TDA"] if k in report]
    axes[0].bar(
        keys,
        [report[k]["allocation_peak_mib"] for k in keys],
        color=["#0072B2", "#009E73", "#CC79A7"][: len(keys)],
    )
    axes[0].set_ylabel("Traced allocation peak (MiB)")
    axes[0].set_title("Measured; mapped pages excluded")
    axes[1].bar(
        ["Spatial M", "Spin M", "TDA vv"],
        [report[k] for k in ["spatial_m_mib", "spin_m_mib", "legacy_tda_vv_mib"]],
        color="#D55E00",
    )
    axes[1].set_ylabel("Tensor size (MiB)")
    axes[1].set_title("Legacy sizes: spin M removed; vv blocked")
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    name = report.get("molecule", "naphthalene")
    fig.suptitle(
        f"{name.capitalize()} / {report['basis']}: {report['nmo']} orbitals, {report['pairs']} pairs"
    )
    fig.savefig(output / f"{name}_gw_tda_memory.png", dpi=180)
    plt.close(fig)
    if "rss_samples" in report:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
        samples = report["rss_samples"]
        axes[0].plot(
            [s[0] for s in samples], [s[2] / 1024 for s in samples], color="#0072B2"
        )
        axes[0].set(
            xlabel="Elapsed time (s)",
            ylabel="Resident RAM (GiB)",
            title="Sampled process RSS",
        )
        stages = report["stage_peak_rss_mib"]
        labels = {
            "Preparation": "Preparation",
            "TDA": "TDA",
            "GW": "GW setup",
            "GW diagonalization": "Diagonalization",
            "GW coupling preparation": "Couplings",
            "GW quasiparticle solve": "QP solve",
            "BSE TDA": "BSE TDA",
            "GW MO factors": "MO factors",
            "GW contour preparation": "Contour prep",
            "GW contour QP": "Contour QP",
        }
        bars = axes[1].bar(
            [labels.get(k, k) for k in stages],
            np.array(list(stages.values())) / 1024,
            color="#0072B2",
        )
        axes[1].bar_label(bars, fmt="%.2f", padding=3)
        axes[1].tick_params(axis="x", rotation=35)
        axes[1].set(ylabel="Resident RAM (GiB)", title="Sampled peak by stage")
        axes[1].margins(y=0.15)
        for ax in axes:
            ax.spines[["top", "right"]].set_visible(False)
        fig.suptitle(
            f"{name.capitalize()} / {report['basis']}: process memory, including mapped pages"
        )
        fig.savefig(output / f"{name}_gw_resident_memory.png", dpi=180)
        plt.close(fig)


def dense_tda_reference(mf):
    """Explicit validation only: independent singlet A from the same factors."""
    from scipy.linalg import eigh
    from pyqed.qchem.basis import mo_pair_factors

    f = mo_pair_factors(mf.mol.eri_factors, mf.mo_coeff)
    no = np.count_nonzero(mf.mo_occ)
    nv = len(mf.mo_energy) - no
    ov = f[:, :no, no:].reshape(len(f), -1)
    a = 2 * ov.T @ ov - np.einsum(
        "Pij,Pab->iajb", f[:, :no, :no], f[:, no:, no:], optimize=True
    ).reshape(no * nv, no * nv)
    a[np.diag_indices(len(a))] += (mf.mo_energy[no:] - mf.mo_energy[:no, None]).ravel()
    return eigh(a, subset_by_index=(0, 2), eigvals_only=True)


def plot_comparison(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(path.read_text()) for path in paths]
    labels = [
        f"{r.get('molecule', 'naphthalene').capitalize()}\n"
        f"{'RI' if 'RI/RHF' in r['preparation'] else 'CD'}, {r['nmo']} orbitals"
        for r in reports
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    x = np.arange(len(reports))
    bars = axes[0].bar(
        x, [r["process_peak_rss_mib"] / 1024 for r in reports], color="#0072B2"
    )
    axes[0].bar_label(bars, fmt="%.2f", padding=3)
    axes[0].set_title("Process peak RSS, including mapped pages")
    axes[0].set_ylabel("GiB")
    for j, (stage, color) in enumerate(
        zip(["GW", "TDA", "BSE TDA"], ["#0072B2", "#009E73", "#CC79A7"])
    ):
        axes[1].bar(
            x + (j - 1) * 0.23,
            [r[stage]["allocation_peak_mib"] for r in reports],
            0.23,
            color=color,
            label=stage,
        )
    axes[1].set_title("Traced allocation peaks, mapped pages excluded")
    axes[1].set_ylabel("MiB")
    axes[1].legend(frameon=False)
    for ax in axes:
        ax.set_xticks(x, labels)
        ax.spines[["top", "right"]].set_visible(False)
        ax.margins(y=0.15)
    fig.suptitle(
        "GW/TDA larger-molecule memory check: 6-31G, complete charge screening"
    )
    fig.savefig(output / "acene_gw_tda_memory_comparison.png", dpi=180)
    plt.close(fig)


def plot_tda_comparison(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(path.read_text()) for path in paths]
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), layout="constrained")
    for ax, representation, before, after in zip(
        axes, ["CD", "RI"], reports[::2], reports[1::2]
    ):
        values = [r["TDA"]["allocation_peak_mib"] for r in [before, after]]
        bars = ax.bar(["Before", "Streamed"], values, color=["#999999", "#0072B2"])
        ax.bar_label(bars, fmt="%.1f MiB", padding=4)
        ax.set_ylim(0, max(r["TDA"]["allocation_peak_mib"] for r in reports) * 1.2)
        ax.set_title(representation)
        ax.set_ylabel("TDA traced allocation peak (MiB)")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{reports[0]['molecule'].capitalize()} / {reports[0]['basis']}: three TDA roots, tolerance 1e-8"
    )
    fig.savefig(output / f"{reports[0]['molecule']}_tda_streamed_memory.png", dpi=180)
    plt.close(fig)


def plot_gw_comparison(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(path.read_text()) for path in paths]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    for ax, values, title, unit in [
        (
            axes[0],
            [r["GW"]["allocation_peak_mib"] for r in reports],
            "GW traced allocation peak",
            "MiB",
        ),
        (
            axes[1],
            [r["process_peak_rss_mib"] / 1024 for r in reports],
            "Full run process peak RSS",
            "GiB",
        ),
    ]:
        bars = ax.bar(["Before", "After"], values, color=["#999999", "#0072B2"])
        ax.bar_label(bars, fmt="%.2f", padding=4)
        ax.set_ylim(0, max(values) * 1.2)
        ax.set_title(title)
        ax.set_ylabel(unit)
    difference = np.array(reports[1]["GW"]["energies"], dtype=float) - np.array(
        reports[0]["GW"]["energies"], dtype=float
    )
    axes[2].plot(np.arange(len(difference)), difference, ".", color="#0072B2")
    axes[2].set_title("Quasiparticle energy agreement")
    axes[2].set_xlabel("Spatial orbital index")
    axes[2].set_ylabel("After − before (Hartree)")
    error = float(np.nanmax(np.abs(difference)))
    axes[2].set_ylim(-max(1e-12, error * 1.2), max(1e-12, error * 1.2))
    axes[2].text(
        0.05, 0.93, f"Maximum difference: {error:.2g} Ha", transform=axes[2].transAxes
    )
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{reports[0]['molecule'].capitalize()} / {reports[0]['basis']}: complete RI charge screening"
    )
    fig.savefig(output / f"{reports[0]['molecule']}_gw_inplace_memory.png", dpi=180)
    plt.close(fig)


def plot_contour_comparison(paths, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    reports = [json.loads(path.read_text()) for path in paths]
    reference, contours = reports[0], reports[1:]
    if not contours or reference["frequency_integration"] != "exact":
        raise ValueError("Supply a spectral report followed by contour reports")
    for r in contours:
        if r["frequency_integration"] != "contour_deformation" or any(
            r[k] != reference[k]
            for k in ["geometry_angstrom", "basis", "preparation", "eta"]
        ):
            raise ValueError(
                "Compare identical references and eta with contour reports"
            )
    exact = np.array(reference["GW"]["energies"], dtype=float)
    errors = [
        float(np.nanmax(abs(np.array(r["GW"]["energies"], dtype=float) - exact)))
        for r in contours
    ]
    labels = ["Spectral\nall QPs"] + [
        f"Contour {r['nw']}\n{len(r['GW']['contour']['orbs'])} QPs" for r in contours
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8), layout="constrained")
    axes[0].semilogy(
        [r["nw"] for r in contours], np.maximum(errors, 1e-16), "o-", color="#0072B2"
    )
    axes[0].set(
        xlabel="Imaginary-axis quadrature points",
        ylabel="Maximum QP error (Hartree)",
        title="Selected QPs versus spectral reference",
    )
    for ax, values, title, unit in [
        (
            axes[1],
            [r["GW"]["allocation_peak_mib"] for r in reports],
            "GW allocation peak",
            "MiB",
        ),
        (
            axes[2],
            [r["process_peak_rss_mib"] / 1024 for r in reports],
            "Process peak RSS\nincluding preparation",
            "GiB",
        ),
    ]:
        bars = ax.bar(labels, values, color=["#999999"] + ["#0072B2"] * len(contours))
        ax.bar_label(bars, fmt="%.2f", padding=3)
        ax.set(title=title, ylabel=unit, ylim=(0, max(values) * 1.2))
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        f"{reference['molecule'].capitalize()} / {reference['basis']}: complete RI screening, eta={reference['eta']:g} Ha"
    )
    fig.savefig(output / "contour_gw_convergence.png", dpi=180)
    plt.close(fig)
    (output / "contour_comparison.json").write_text(
        json.dumps(dict(nw=[r["nw"] for r in contours], max_qp_error=errors), indent=2)
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument(
        "--molecule",
        choices=["naphthalene", "anthracene", "tetracene"],
        default="naphthalene",
    )
    parser.add_argument("--integrals", choices=["cd", "ri"], default="cd")
    parser.add_argument("--auxbasis", default="cc-pvdz-jkfit")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--pyscf-ri",
        action="store_true",
        help="Use a PySCF RI/RHF preparation reference",
    )
    parser.add_argument(
        "--dense-reference",
        action="store_true",
        help="Validate TDA roots against an explicit dense matrix after profiling",
    )
    parser.add_argument(
        "--tda-only", action="store_true", help="Profile only the native TDA solve"
    )
    parser.add_argument(
        "--gw-only", action="store_true", help="Profile GW without TDA/BSE"
    )
    parser.add_argument(
        "--freq-int", choices=["exact", "contour_deformation"], default="exact"
    )
    parser.add_argument("--nw", type=int, default=64, help="Contour quadrature points")
    parser.add_argument(
        "--orbs", type=int, nargs="+", help="Spatial QP orbitals for contour GW"
    )
    parser.add_argument("--eta", type=float, default=1e-3)
    parser.add_argument(
        "--compare",
        type=Path,
        nargs="+",
        help="Plot existing JSON reports without running calculations",
    )
    parser.add_argument(
        "--compare-tda",
        type=Path,
        nargs=4,
        metavar=("OLD_CD", "NEW_CD", "OLD_RI", "NEW_RI"),
        help="Plot saved TDA allocation peaks before and after streaming",
    )
    parser.add_argument(
        "--compare-gw",
        type=Path,
        nargs=2,
        metavar=("BEFORE", "AFTER"),
        help="Plot GW memory and quasiparticle agreement from saved RI reports",
    )
    parser.add_argument(
        "--compare-contour",
        type=Path,
        nargs="+",
        help="Plot a spectral report followed by matched contour grid reports",
    )
    args = parser.parse_args()
    if args.orbs is not None and not args.gw_only:
        parser.error(
            "Partial QP benchmarks require --gw-only; BSE needs all QP energies"
        )
    if args.orbs is not None and args.freq_int != "contour_deformation":
        parser.error("--orbs currently requires contour_deformation")
    args.output.mkdir(parents=True, exist_ok=True)
    if args.compare:
        plot_comparison(args.compare, args.output)
        return
    if args.compare_tda:
        plot_tda_comparison(args.compare_tda, args.output)
        return
    if args.compare_gw:
        plot_gw_comparison(args.compare_gw, args.output)
        return
    if args.compare_contour:
        plot_contour_comparison(args.compare_contour, args.output)
        return
    rings = {"naphthalene": 2, "anthracene": 3, "tetracene": 4}[args.molecule]
    report = dict(
        molecule=args.molecule,
        basis=args.basis,
        geometry_angstrom=geometry(rings),
        max_memory_mb=128,
        frequency_integration=args.freq_int,
        eta=args.eta,
        nw=args.nw,
    )
    with (
        (args.output / "calculation.log").open("w", buffering=1) as log,
        contextlib.redirect_stdout(log),
        resident_profile(report) as phase,
    ):
        print(f"Preparing {args.molecule}/{args.basis}", flush=True)
        if args.pyscf_ri:
            from pyscf import gto, scf
            from pyqed.qchem.basis import PackedRIFactors

            mol = gto.M(atom=report["geometry_angstrom"], basis=args.basis, verbose=0)
            mf = scf.RHF(mol).density_fit(auxbasis=args.auxbasis)
            mf.chkfile = None
            mf.conv_tol = 1e-10
            mf.kernel()
            mol.eri_factors = PackedRIFactors(
                np.concatenate(list(mf.with_df.loop())), mol.nao
            )
            report["preparation"] = f"PySCF RI/RHF, {args.auxbasis}"
        else:
            mol = Molecule(
                atom=report["geometry_angstrom"], unit="angstrom", basis=args.basis
            )
            mol.build(
                eri=args.integrals,
                auxbasis=args.auxbasis if args.integrals == "ri" else None,
                options={"low_rank_tol": 1e-8, "workers": 1, "ri_cache": False},
            )
            mf = mol.RHF().run(tol=1e-10, max_cycle=100, verbose=0)
            report["preparation"] = (
                f"PyQED RI/RHF, {args.auxbasis}"
                if args.integrals == "ri"
                else "PyQED CD/RHF, tolerance 1e-8"
            )
        assert mf.converged
        print("RHF preparation converged", flush=True)
        mf.max_memory = report["max_memory_mb"]
        n = len(mf.mo_energy)
        no = int(np.count_nonzero(mf.mo_occ))
        pairs = no * (n - no)
        report.update(
            nmo=n,
            nocc=no,
            pairs=pairs,
            factors=len(mol.eri_factors),
            spatial_m_mib=n * n * pairs * 8 / 2**20,
            spin_m_mib=4 * n * n * pairs * 8 / 2**20,
            legacy_tda_vv_mib=len(mol.eri_factors) * (n - no) ** 2 * 8 / 2**20,
        )
        for name in (
            ["TDA"]
            if args.tda_only
            else ["GW"] if args.gw_only else ["TDA", "GW", "BSE TDA"]
        ):
            phase[0] = name
            print(f"Starting {name}: {n} orbitals, {pairs} pairs", flush=True)
            tracemalloc.start()
            start = time.perf_counter()
            if name == "TDA":
                obj = TDA(mf).run(nstates=3, tolerance=1e-8)
                extra = dict(
                    energies=obj.e.tolist(),
                    residuals=obj.solver_info["response_residuals"].tolist(),
                )
            elif name == "GW":
                controls = (
                    dict(nw=args.nw, orbs=args.orbs)
                    if args.freq_int == "contour_deformation"
                    else {}
                )
                obj = GW(mf, eta=args.eta, freq_int=args.freq_int).run(**controls)
                gw = obj
                extra = dict(
                    energies=[float(e) if np.isfinite(e) else None for e in obj.e_qp],
                    residuals=[
                        float(e) if np.isfinite(e) else None for e in obj.qp_residuals
                    ],
                    mapped_spin=isinstance(obj._M, np.memmap),
                    spin_couplings_materialized=obj._M is not None,
                )
                if obj._charge_screening is not None:
                    couplings = obj._charge_screening["couplings"]
                    extra["mapped_spatial"] = isinstance(couplings, np.memmap)
                    extra["factorized_couplings"] = isinstance(
                        couplings, FactorizedCouplings
                    )
                    if isinstance(couplings, FactorizedCouplings):
                        extra["coupling_projection_mib"] = (
                            couplings.projection.nbytes / 2**20
                        )
                if args.freq_int == "contour_deformation":
                    extra["contour"] = obj.contour_info
            else:
                obj = BSETDA(gw).run(
                    nroots=3, low_rank=True, batch_columns=2, tol=1e-8, max_cycle=200
                )
                contour = getattr(gw, "_contour", None)
                reused = (
                    obj._static_screening is contour.static_screening
                    if contour is not None
                    else obj._M is gw._charge_screening["couplings"]
                )
                extra = dict(
                    energies=obj.e.tolist(),
                    solver_converged=bool(obj.info["converged"]),
                    shared_screening=reused,
                    static_screening=obj._static_screening is not None,
                    residuals=np.asarray(obj.info["residual_norms"]).tolist(),
                )
                assert obj.info["converged"]
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            report[name] = dict(
                seconds=time.perf_counter() - start,
                allocation_peak_mib=peak / 2**20,
                **extra,
            )
            print(f'Finished {name}: {report[name]["seconds"]:.3f} s', flush=True)
            (args.output / "report.json").write_text(json.dumps(report, indent=2))
        report["process_peak_rss_mib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        )
        if args.dense_reference:
            reference = dense_tda_reference(mf)
            report["tda_dense_reference_error"] = float(
                np.max(abs(reference - report["TDA"]["energies"]))
            )
            assert report["tda_dense_reference_error"] < 1e-7
    (args.output / "report.json").write_text(json.dumps(report, indent=2))
    plot(report, args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in ["geometry_angstrom", "rss_samples"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
