"""Compare cross-geometry SU(2)-NARG and exact CASCI overlaps for pyrazine."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("PYQED_MPS_DISABLE_CPP_DAVIDSON", "1")

import matplotlib.pyplot as plt
import numpy as np

from examples.qchem.pyrazine_dmrgscf import PYRAZINE_GEOMETRY_BOHR
from pyqed.narg.qchem.su2 import NARG
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI


def _plain(value):
    if isinstance(value, np.ndarray):
        return _plain(value.tolist())
    if isinstance(value, np.generic):
        return _plain(value.item())
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _geometry(scale, coupling_displacement=0.0):
    coupling_pattern = np.array(
        [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -0.7, 0.7, -0.7, 0.7]
    )
    geometry = []
    for atom_index, (symbol, x, y, z) in enumerate(PYRAZINE_GEOMETRY_BOHR):
        displaced_x = scale * x + coupling_displacement * coupling_pattern[atom_index]
        geometry.append([symbol, displaced_x, scale * y, scale * z])
    return geometry


def _run_geometry(geometry, args):
    timings = {}
    started = time.perf_counter()
    mol = Molecule(atom=geometry, unit="bohr", basis=args.basis)
    mol.build(
        eri="cd",
        options={
            "low_rank_tol": args.cd_tol,
            "parallel": False,
            "eri_screen_tol": 0.0,
        },
    )
    timings["integrals"] = time.perf_counter() - started
    print(f"integrals: {timings['integrals']:.3f} s", flush=True)

    started = time.perf_counter()
    mf = mol.RHF().run(tol=1.0e-10, max_cycle=100, verbose=0)
    timings["rhf"] = time.perf_counter() - started
    print(f"RHF: {timings['rhf']:.3f} s", flush=True)

    started = time.perf_counter()
    casci = CASCI(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        multiplicity=1,
        verbose=0,
    ).run(
        nstates=args.nstates + args.casci_extra_roots,
        method="direct_ci",
        use_cholesky=True,
    )
    casci.e_tot = np.asarray(casci.e_tot)[: args.nstates]
    casci.ci = list(casci.ci[: args.nstates])
    casci.nstates = args.nstates
    casci._direct_solver = None
    timings["casci"] = time.perf_counter() - started
    print(f"CASCI: {timings['casci']:.3f} s", flush=True)

    started = time.perf_counter()
    narg = NARG(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        nstates=args.nstates,
        target_j2=0,
        su2_backend=args.su2_backend,
        threads=args.threads,
        carry_rdm_operators=False,
        carry_spin_rdm_operators=False,
    ).run()
    timings["narg"] = time.perf_counter() - started
    print(f"NARG: {timings['narg']:.3f} s", flush=True)
    return mol, mf, casci, narg, timings


def _plot(cas_overlap, narg_overlap, output):
    exact = np.abs(cas_overlap)
    approximate = np.abs(narg_overlap)
    error = np.abs(approximate - exact)
    vmax = max(float(np.max(exact)), float(np.max(approximate)))
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), constrained_layout=True)
    images = (
        axes[0].imshow(exact, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis"),
        axes[1].imshow(approximate, origin="lower", vmin=0.0, vmax=vmax, cmap="viridis"),
        axes[2].imshow(error, origin="lower", cmap="magma"),
    )
    axes[0].set_title("CASCI")
    axes[1].set_title("SU(2)-NARG")
    axes[2].set_title(r"$|\,|S_{\rm NARG}|-|S_{\rm CASCI}|\,|$")
    for axis, matrix in zip(axes[:2], (exact, approximate)):
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                color = "black" if matrix[row, column] > 0.55 * vmax else "white"
                axis.text(
                    column,
                    row,
                    f"{matrix[row, column]:.3g}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=7,
                )
    error_scale = float(np.max(error))
    for row in range(error.shape[0]):
        for column in range(error.shape[1]):
            color = "black" if error[row, column] > 0.55 * error_scale else "white"
            axes[2].text(
                column,
                row,
                f"{error[row, column]:.1e}",
                ha="center",
                va="center",
                color=color,
                fontsize=6.5,
            )
    for label, axis in zip("abc", axes):
        axis.set_xlabel("Geometry 2 root")
        axis.set_ylabel("Geometry 1 root")
        axis.set_xticks(range(exact.shape[1]))
        axis.set_yticks(range(exact.shape[0]))
        axis.text(-0.20, 1.04, label, transform=axis.transAxes, fontweight="bold")
    fig.colorbar(images[0], ax=axes[:2], shrink=0.82, label="Overlap magnitude")
    fig.colorbar(images[2], ax=axes[2], shrink=0.82, label="Absolute error")
    fig.savefig(output, dpi=320)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--basis", default="6-31g")
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--nstates", type=int, default=4)
    parser.add_argument("--casci-extra-roots", type=int, default=4)
    parser.add_argument("--D", type=int, default=64)
    parser.add_argument("--geometry-scale", type=float, default=1.0025)
    parser.add_argument("--coupling-displacement", type=float, default=0.0)
    parser.add_argument("--cd-tol", type=float, default=1.0e-10)
    parser.add_argument("--su2-backend", default="python")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/private/tmp/pyrazine_narg_overlap_casci"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("geometry 1", flush=True)
    first = _run_geometry(_geometry(1.0), args)
    print("geometry 2", flush=True)
    second = _run_geometry(
        _geometry(args.geometry_scale, args.coupling_displacement),
        args,
    )
    _, _, casci_1, narg_1, timings_1 = first
    _, _, casci_2, narg_2, timings_2 = second

    started = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore"):
        cas_overlap = np.asarray(casci_1.overlap(casci_2))
    cas_overlap_seconds = time.perf_counter() - started
    started = time.perf_counter()
    with np.errstate(divide="ignore", invalid="ignore"):
        narg_overlap, overlap_info = narg_1.overlap(
            narg_2,
            orbital_split="auto",
            orbital_map_threshold=0.0,
            cutoff=0.0,
            max_bond=None,
            return_info=True,
        )
    narg_overlap_seconds = time.perf_counter() - started
    narg_overlap = np.asarray(narg_overlap)

    magnitude_error = np.abs(np.abs(narg_overlap) - np.abs(cas_overlap))
    cas_energies = np.asarray(casci_1.e_tot), np.asarray(casci_2.e_tot)
    narg_energies = np.asarray(narg_1.e_tot), np.asarray(narg_2.e_tot)
    energy_error = np.asarray(narg_energies) - np.asarray(cas_energies)
    payload = {
        "system": f"pyrazine/{args.basis} CAS({args.nelecas},{args.ncas})",
        "geometry_scale": args.geometry_scale,
        "coupling_displacement_bohr": args.coupling_displacement,
        "D": args.D,
        "nstates": args.nstates,
        "casci_extra_roots": args.casci_extra_roots,
        "casci_energies_hartree": cas_energies,
        "narg_energies_hartree": narg_energies,
        "energy_errors_hartree": energy_error,
        "casci_overlap": cas_overlap,
        "narg_overlap": narg_overlap,
        "max_magnitude_error": float(np.max(magnitude_error)),
        "rms_magnitude_error": float(np.sqrt(np.mean(magnitude_error**2))),
        "timings_geometry_1": timings_1,
        "timings_geometry_2": timings_2,
        "casci_overlap_seconds": cas_overlap_seconds,
        "narg_overlap_seconds": narg_overlap_seconds,
        "narg_overlap_info": overlap_info,
    }
    np.savez(
        args.output_dir / "pyrazine_narg_overlap_casci.npz",
        casci_energies=np.asarray(cas_energies),
        narg_energies=np.asarray(narg_energies),
        casci_overlap=cas_overlap,
        narg_overlap=narg_overlap,
        magnitude_error=magnitude_error,
    )
    (args.output_dir / "pyrazine_narg_overlap_casci.json").write_text(
        json.dumps(_plain(payload), indent=2) + "\n"
    )
    _plot(
        cas_overlap,
        narg_overlap,
        args.output_dir / "pyrazine_narg_overlap_casci.png",
    )
    print(json.dumps(_plain(payload), indent=2), flush=True)


if __name__ == "__main__":
    main()
