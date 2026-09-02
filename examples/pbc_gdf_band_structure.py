#!/usr/bin/env python3
"""Compute a direct GDF-KRHF diamond band path and compare with PySCF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw.integrals import (
    _gdf_normalize_auxbasis_name,
    _pyscf_builtin_basis_dict,
    _pyscf_cell_from_reference,
)
from pyqed.pbc.gw.response import KPointTransitionSpace
from pyqed.qchem.pbc import Cell
from pyqed.units import au2ev


def _mesh(value):
    mesh = tuple(int(item) for item in str(value).split(","))
    if len(mesh) != 3 or any(item <= 0 for item in mesh):
        raise argparse.ArgumentTypeError("kmesh must contain three positive integers")
    return mesh


def _fcc_primitive(length):
    half = 0.5 * float(length)
    return np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )


def _diamond_path(length, points_per_segment):
    labels = (r"$\Gamma$", "X", "W", "K", r"$\Gamma$", "L")
    conventional = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.75, 0.75, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.5],
        ],
        dtype=float,
    )
    nodes = conventional * (2.0 * np.pi / float(length))
    path = []
    node_rows = [0]
    for segment, (left, right) in enumerate(zip(nodes[:-1], nodes[1:])):
        values = np.linspace(left, right, int(points_per_segment) + 1)
        if segment:
            values = values[1:]
        path.extend(values)
        node_rows.append(len(path) - 1)
    path = np.asarray(path, dtype=float)
    distance = np.zeros(len(path), dtype=float)
    if len(path) > 1:
        distance[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    return path, distance, np.asarray(node_rows, dtype=int), labels


def _pyqed_reference(args, path):
    length = float(args.lattice)
    cell = Cell(
        atom=f"C 0 0 0; C {length / 4:.12f} {length / 4:.12f} {length / 4:.12f}",
        a=_fcc_primitive(length),
        basis=args.basis,
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    kpts = cell.make_kpts(args.kmesh, gamma_centered=args.gamma_centered)
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut="auto",
        pair_cut="auto",
        recip_cut=5,
    ).density_fit(
        auxbasis=args.auxbasis,
        precision=args.precision,
        mesh="auto",
        omega="auto",
        pair_cut="auto",
        storage=args.storage,
        max_memory_mb=args.max_memory_mb,
        stream_pairs=True,
        aux_min_exponent=args.aux_min_exponent,
        metric_tol=args.metric_tol,
    )
    started = time.perf_counter()
    mf.with_df.build(workers=args.workers)
    gdf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    mf.run(max_cycle=80, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)
    scf_seconds = time.perf_counter() - started
    if not mf.converged:
        raise RuntimeError("PyQED GDF-KRHF did not converge")
    started = time.perf_counter()
    bands = mf.band_structure(
        kpts=path,
        exchange="finite_q",
        reference="none",
        sort_bands="energy",
    )
    band_seconds = time.perf_counter() - started
    return cell, mf, np.asarray(bands["mo_energy"]), {
        "gdf_seconds": float(gdf_seconds),
        "scf_seconds": float(scf_seconds),
        "band_seconds": float(band_seconds),
        "band_build": dict(mf.with_df.band_build_timings),
    }


def _pyscf_reference(args, mf, path):
    from pyscf.pbc import scf

    space = KPointTransitionSpace(mf, qpts="mesh")
    cell = _pyscf_cell_from_reference(space.reference)
    cell.precision = float(args.reference_precision)
    cell.verbose = 0
    cell.build()
    auxbasis = _pyscf_builtin_basis_dict(
        _gdf_normalize_auxbasis_name(args.auxbasis),
        mf.cell._atom_symbols,
    )
    reference = scf.KRHF(cell, kpts=mf.kpts, exxdiv="ewald").density_fit(
        auxbasis=auxbasis
    )
    reference.with_df.linear_dep_threshold = float(args.metric_tol)
    reference.with_df.exp_to_discard = float(args.aux_min_exponent)
    started = time.perf_counter()
    reference.with_df.build(j_only=False, kpts_band=path)
    gdf_seconds = time.perf_counter() - started
    started = time.perf_counter()
    reference.kernel()
    scf_seconds = time.perf_counter() - started
    if not reference.converged:
        raise RuntimeError("PySCF GDF-KRHF did not converge")
    started = time.perf_counter()
    energies, _coefficients = reference.get_bands(path)
    band_seconds = time.perf_counter() - started
    return np.asarray(energies), {
        "gdf_seconds": float(gdf_seconds),
        "scf_seconds": float(scf_seconds),
        "band_seconds": float(band_seconds),
        "scf_energy_Ha": float(reference.e_tot),
    }


def _fermi_energy(mf):
    energies = np.asarray(mf.mo_energy, dtype=float)
    occupations = np.asarray(mf.mo_occ, dtype=float)
    return float(np.max(energies[occupations > 1.0e-12]))


def _window_band_error(pyqed_bands, pyscf_bands, fermi, energy_window):
    if pyscf_bands is None:
        return None
    shifted = (np.asarray(pyqed_bands) - float(fermi)) * au2ev
    mask = (shifted >= energy_window[0]) & (shifted <= energy_window[1])
    if not np.any(mask):
        return None
    difference = np.abs(np.asarray(pyqed_bands) - np.asarray(pyscf_bands))
    return float(np.max(difference[mask]) * au2ev * 1.0e3)


def _plot(
    output,
    distance,
    node_rows,
    labels,
    pyqed_bands,
    fermi,
    pyscf_bands,
    energy_window,
):
    fig, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    shifted = (np.asarray(pyqed_bands) - float(fermi)) * au2ev
    for band in range(shifted.shape[1]):
        axis.plot(
            distance,
            shifted[:, band],
            color="#0072B2",
            linewidth=1.5,
            label="PyQED" if band == 0 else None,
        )
    if pyscf_bands is not None:
        reference = (np.asarray(pyscf_bands) - float(fermi)) * au2ev
        for band in range(reference.shape[1]):
            axis.plot(
                distance,
                reference[:, band],
                color="#D55E00",
                linewidth=1.0,
                linestyle="--",
                label="PySCF" if band == 0 else None,
            )
    for row in node_rows:
        axis.axvline(distance[row], color="#777777", linewidth=0.6, alpha=0.6)
    axis.axhline(0.0, color="#222222", linewidth=0.8)
    axis.set_xticks(distance[node_rows], labels)
    axis.set(
        xlim=(distance[0], distance[-1]),
        ylim=tuple(float(value) for value in energy_window),
        ylabel=r"Energy $-E_F$ (eV)",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", alpha=0.2, linewidth=0.6)
    axis.legend(frameon=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".png"), dpi=300)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lattice", type=float, default=6.74)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--auxbasis", default="def2-svp-jkfit")
    parser.add_argument("--kmesh", type=_mesh, default=(2, 2, 2))
    parser.add_argument("--gamma-centered", action="store_true")
    parser.add_argument("--points-per-segment", type=int, default=4)
    parser.add_argument("--precision", type=float, default=1.0e-8)
    parser.add_argument("--reference-precision", type=float, default=1.0e-12)
    parser.add_argument("--aux-min-exponent", type=float, default=0.075)
    parser.add_argument("--metric-tol", type=float, default=1.0e-12)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--storage", choices=("auto", "memory", "disk"), default="disk")
    parser.add_argument("--max-memory-mb", type=float, default=0.0)
    parser.add_argument("--energy-min", type=float, default=-20.0)
    parser.add_argument("--energy-max", type=float, default=20.0)
    parser.add_argument("--skip-pyscf", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_diamond_gdf_band_structure"),
    )
    args = parser.parse_args()
    if args.points_per_segment < 1:
        parser.error("--points-per-segment must be positive")
    if args.aux_min_exponent < 0.0:
        parser.error("--aux-min-exponent must be non-negative")
    if args.energy_min >= args.energy_max:
        parser.error("--energy-min must be smaller than --energy-max")

    path, distance, node_rows, labels = _diamond_path(
        args.lattice,
        args.points_per_segment,
    )
    _cell, mf, pyqed_bands, pyqed_timings = _pyqed_reference(args, path)
    pyscf_bands = None
    pyscf_timings = None
    if not args.skip_pyscf:
        pyscf_bands, pyscf_timings = _pyscf_reference(args, mf, path)

    fermi = _fermi_energy(mf)
    energy_window = (float(args.energy_min), float(args.energy_max))
    output = args.output.expanduser().resolve()
    _plot(
        output,
        distance,
        node_rows,
        labels,
        pyqed_bands,
        fermi,
        pyscf_bands,
        energy_window,
    )
    payload = {
        "case": "diamond",
        "lattice_bohr": float(args.lattice),
        "basis": args.basis,
        "auxbasis": args.auxbasis,
        "aux_min_exponent": float(args.aux_min_exponent),
        "kmesh": list(args.kmesh),
        "gamma_centered": bool(args.gamma_centered),
        "points_per_segment": int(args.points_per_segment),
        "path_kpts": path.tolist(),
        "distance": distance.tolist(),
        "node_rows": node_rows.tolist(),
        "fermi_Ha": float(fermi),
        "energy_window_eV": list(energy_window),
        "pyqed_bands_Ha": pyqed_bands.tolist(),
        "pyqed_scf_energy_Ha": float(mf.e_tot),
        "pyqed_timings": pyqed_timings,
        "pyscf_bands_Ha": None if pyscf_bands is None else pyscf_bands.tolist(),
        "pyscf_timings": pyscf_timings,
        "max_abs_band_error_meV": (
            None
            if pyscf_bands is None
            else float(np.max(np.abs(pyqed_bands - pyscf_bands)) * au2ev * 1.0e3)
        ),
        "max_abs_window_band_error_meV": _window_band_error(
            pyqed_bands,
            pyscf_bands,
            fermi,
            energy_window,
        ),
    }
    output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(f"figure: {output.with_suffix('.png')}")
    print(f"json: {output.with_suffix('.json')}")
    if payload["max_abs_band_error_meV"] is not None:
        print(f"max band error: {payload['max_abs_band_error_meV']:.6f} meV")


if __name__ == "__main__":
    main()
