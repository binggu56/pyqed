"""Benchmark full-cross SU(2)-detached NARG on pyrazine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from benchmarks.pyrazine_narg_overlap_casci import _geometry, _plain
from pyqed.narg.qchem.su2 import NARG
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI


def _run_narg(mf, args, *, D, **options):
    solver = NARG(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=int(D),
        nstates=1,
        target_j2=0,
        su2_backend=args.su2_backend,
        threads=args.threads,
        carry_rdm_operators=False,
        carry_spin_rdm_operators=False,
        **options,
    )
    started = time.perf_counter()
    solver.run()
    seconds = time.perf_counter() - started
    energy = float(np.asarray(solver.e_tot).reshape(-1)[0])
    return solver, energy, seconds


def _plot(records, output):
    dimensions = [record["D"] for record in records]
    figure, axes = plt.subplots(1, 2, figsize=(9.2, 3.8), constrained_layout=True)
    for key, label in (
        ("ordinary", "ordinary SU(2), same D"),
        ("fixed_detached", "fixed full-cross detached"),
        ("adaptive_detached", "adaptive protected-core scan"),
        ("ordinary_same_chi", "ordinary SU(2), same chi"),
    ):
        axes[0].plot(
            dimensions,
            [max(record[f"{key}_error"], 1.0e-15) for record in records],
            marker="o",
            label=label,
        )
        axes[1].plot(
            dimensions,
            [record[f"{key}_seconds"] for record in records],
            marker="o",
            label=label,
        )
    for axis in axes:
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.legend(frameon=False)
    axes[0].set(xlabel="frame parameter D", ylabel="energy error (hartree)")
    axes[1].set(xlabel="frame parameter D", ylabel="wall time (s)")
    figure.savefig(output, dpi=220)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--D", type=int, nargs="+", default=(2, 4, 8, 16))
    parser.add_argument("--chi-factor", type=int, default=12)
    parser.add_argument("--cd-tol", type=float, default=1.0e-10)
    parser.add_argument("--su2-backend", default="python")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_su2_detached"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    mol = Molecule(atom=_geometry(1.0), unit="bohr", basis=args.basis)
    mol.build(
        eri="cd",
        options={
            "low_rank_tol": args.cd_tol,
            "parallel": False,
            "eri_screen_tol": 0.0,
        },
    )
    integral_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mf = mol.RHF().run(tol=1.0e-10, max_cycle=100, verbose=0)
    rhf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    casci = CASCI(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        multiplicity=1,
        verbose=0,
    ).run(nstates=1, method="direct_ci", use_cholesky=True)
    casci_seconds = time.perf_counter() - started
    exact = float(np.asarray(casci.e_tot).reshape(-1)[0])

    records = []
    for D in args.D:
        chi = int(args.chi_factor) * int(D)
        _plain_solver, plain_energy, plain_seconds = _run_narg(mf, args, D=D)
        _chi_solver, chi_energy, chi_seconds = _run_narg(mf, args, D=chi)
        fixed, fixed_energy, fixed_seconds = _run_narg(
            mf,
            args,
            D=D,
            dressing="detached_frames",
            chi=chi,
            frame_protect_dim=0,
        )
        scan = []
        for protected in sorted({0, int(D) // 2, int(D)}):
            candidate = _run_narg(
                mf,
                args,
                D=D,
                dressing="detached_frames",
                chi=chi,
                frame_adapt_tol=0.1,
                frame_max_dim=chi,
                frame_expand_dim=D,
                frame_protect_dim=protected,
            )
            scan.append((*candidate, protected))
        adaptive, adaptive_energy, selected_seconds, selected = min(
            scan,
            key=lambda item: item[1],
        )
        adaptive_seconds = sum(item[2] for item in scan)
        detached_diagnostics = adaptive.chain.timings["detached_by_size"]
        record = {
            "D": int(D),
            "chi": chi,
            "ordinary_energy": plain_energy,
            "ordinary_error": plain_energy - exact,
            "ordinary_seconds": plain_seconds,
            "ordinary_same_chi_energy": chi_energy,
            "ordinary_same_chi_error": chi_energy - exact,
            "ordinary_same_chi_seconds": chi_seconds,
            "fixed_detached_energy": fixed_energy,
            "fixed_detached_error": fixed_energy - exact,
            "fixed_detached_seconds": fixed_seconds,
            "adaptive_detached_energy": adaptive_energy,
            "adaptive_detached_error": adaptive_energy - exact,
            "adaptive_detached_seconds": adaptive_seconds,
            "adaptive_selected_seconds": selected_seconds,
            "adaptive_selected_protection": selected,
            "adaptive_protection_scan": {
                str(item[3]): item[1] for item in scan
            },
            "maximum_frame_union_rank": max(
                item["frame_union_rank"] for item in detached_diagnostics.values()
            ),
            "maximum_detached_dimension": max(
                item["detached_dim"] for item in detached_diagnostics.values()
            ),
        }
        records.append(record)
        print(json.dumps(_plain(record)), flush=True)

    payload = {
        "system": f"pyrazine/{args.basis} CAS({args.nelecas},{args.ncas})",
        "exact_casci_energy_hartree": exact,
        "chi_factor": int(args.chi_factor),
        "integral_seconds": integral_seconds,
        "rhf_seconds": rhf_seconds,
        "casci_seconds": casci_seconds,
        "records": records,
    }
    data_path = args.output_dir / "pyrazine_su2_detached.json"
    figure_path = args.output_dir / "pyrazine_su2_detached.png"
    data_path.write_text(json.dumps(_plain(payload), indent=2) + "\n")
    _plot(records, figure_path)
    print(json.dumps(_plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
