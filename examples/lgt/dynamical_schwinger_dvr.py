#!/usr/bin/env python3
"""Dynamical quantum-link Schwinger model with Wilson-dressed DVR hopping."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse.linalg as spla

from pyqed.lgt import QuantumSchwingerDVR


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "dynamical_schwinger_dvr"


def solve_model(flux_cutoff, *, npts=7, length=10.0, coupling=1.0, nroots=32):
    start = perf_counter()
    model = QuantumSchwingerDVR(
        npts,
        length,
        coupling=coupling,
        mass=0.0,
        flux_cutoff=flux_cutoff,
    ).run(nroots=nroots)
    seconds = perf_counter() - start
    hnorm = spla.norm(model.hamiltonian)
    hermiticity = spla.norm(model.hamiltonian - model.hamiltonian.getH()) / hnorm
    max_gauss = max(
        np.max(np.abs(model.gauss_law(int(bits), flux)))
        for bits, flux in zip(model.basis_bits, model.basis_flux)
    )
    return model, {
        "flux_cutoff": flux_cutoff,
        "dimension": model.dimension,
        "hamiltonian_nnz": model.hamiltonian.nnz,
        "seconds": seconds,
        "vacuum_dimension": model.vacuum_dimension,
        "vector_level": model.vector_level,
        "vector_excitation_energy": model.vector_excitation_energy,
        "vector_momentum": model.vector_momentum,
        "vector_gap": model.vector_gap,
        "scalar_level": model.scalar_level,
        "scalar_gap": model.scalar_gap,
        "hermiticity_residual": float(hermiticity),
        "maximum_gauss_residual": int(max_gauss),
    }


def calculate(
    flux_cutoffs=(1, 2, 3),
    *,
    npts=7,
    length=10.0,
    coupling=1.0,
    nroots=32,
):
    models = []
    records = []
    for cutoff in flux_cutoffs:
        model, record = solve_model(
            cutoff,
            npts=npts,
            length=length,
            coupling=coupling,
            nroots=nroots,
        )
        models.append(model)
        records.append(record)
        print(
            f"Lmax={cutoff} dim={model.dimension} nnz={model.hamiltonian.nnz} "
            f"M_V/g={model.vector_gap / coupling:.9f} "
            f"M_S/g={model.scalar_gap / coupling:.9f}",
            flush=True,
        )
    return models, records


def _style_axis(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_gaps(records, coupling, output):
    cutoff = np.asarray([record["flux_cutoff"] for record in records])
    vector = np.asarray([record["vector_gap"] for record in records]) / coupling
    scalar = np.asarray([record["scalar_gap"] for record in records]) / coupling
    exact_vector = 1.0 / np.sqrt(np.pi)
    exact_scalar = 2.0 / np.sqrt(np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), constrained_layout=True)
    axes[0].plot(cutoff, vector, "o-", label=r"$M_V/g$")
    axes[0].plot(cutoff, scalar, "s-", label=r"$M_S/g$")
    axes[0].axhline(exact_vector, color="C0", linestyle="--", alpha=0.65)
    axes[0].axhline(exact_scalar, color="C1", linestyle="--", alpha=0.65)
    axes[0].set_xlabel(r"electric-flux cutoff $L_{\max}$")
    axes[0].set_ylabel("dimensionless mass gap")
    axes[0].set_title("Dynamical Wilson-DVR gaps")
    axes[0].legend(frameon=False)
    axes[0].set_xticks(cutoff)
    _style_axis(axes[0])

    axes[1].semilogy(cutoff, np.abs(vector - exact_vector), "o-", label=r"$M_V$")
    axes[1].semilogy(cutoff, np.abs(scalar - exact_scalar), "s-", label=r"$M_S$")
    axes[1].set_xlabel(r"electric-flux cutoff $L_{\max}$")
    axes[1].set_ylabel("absolute error from exact continuum value")
    axes[1].set_title("Flux convergence and remaining grid error")
    axes[1].legend(frameon=False)
    axes[1].set_xticks(cutoff)
    _style_axis(axes[1])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_channel_strengths(model, output):
    excitation = model.energies - model.energies[0]
    first = model.vacuum_dimension
    levels = slice(first, None)
    vector = model.vector_strengths / np.max(model.vector_strengths[first:])
    scalar = model.scalar_strengths / np.max(model.scalar_strengths[first:])
    floor = 1.0e-14

    fig, axis = plt.subplots(figsize=(7.0, 4.6), constrained_layout=True)
    axis.semilogy(
        excitation[levels],
        np.maximum(vector[levels], floor),
        "o",
        label=r"vector: $\rho_{k=2\pi/L}$",
    )
    axis.semilogy(
        excitation[levels],
        np.maximum(scalar[levels], floor),
        "s",
        label=r"scalar: $\sum_x\bar\psi\psi$",
    )
    exact_vector_energy = np.sqrt(
        1.0 / np.pi + model.vector_momentum**2
    )
    axis.axvline(
        exact_vector_energy,
        color="C0",
        linestyle="--",
        alpha=0.65,
        label=r"exact $E_V(2\pi/L)$",
    )
    axis.axvline(
        2.0 / np.sqrt(np.pi),
        color="C1",
        linestyle="--",
        alpha=0.65,
        label=r"exact $M_S$",
    )
    axis.set_xlabel(r"excitation energy $(E_n-E_0)/g$")
    axis.set_ylabel("normalized transition strength")
    axis.set_title("Gauge-invariant channel identification")
    axis.legend(frameon=False, fontsize=9)
    _style_axis(axis)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    return value


def run(
    output_directory: Path,
    *,
    npts=7,
    length=10.0,
    coupling=1.0,
    flux_cutoffs=(1, 2, 3),
    nroots=32,
):
    output_directory.mkdir(parents=True, exist_ok=True)
    models, records = calculate(
        flux_cutoffs,
        npts=npts,
        length=length,
        coupling=coupling,
        nroots=nroots,
    )
    converged = models[-1]
    gap_figure = output_directory / "07_dynamical_schwinger_gaps.png"
    strength_figure = output_directory / "08_dynamical_channel_strengths.png"
    plot_gaps(records, coupling, gap_figure)
    plot_channel_strengths(converged, strength_figure)

    payload = {
        "description": (
            "Compact quantum U(1) links, exact Gauss-law basis, electric "
            "L_n^2 energy, and shortest-path Wilson strings on the Fourier-DVR hopping."
        ),
        "parameters": {
            "npts": converged.npts,
            "length_times_g": converged.length * coupling,
            "fermion_mass_over_g": converged.mass / coupling,
            "nroots": len(converged.energies),
        },
        "exact": {
            "vector_gap_over_g": 1.0 / np.sqrt(np.pi),
            "scalar_gap_over_g": 2.0 / np.sqrt(np.pi),
        },
        "flux_convergence": records,
        "converged_spectrum": {
            "energies": converged.energies,
            "vector_strengths": converged.vector_strengths,
            "scalar_strengths": converged.scalar_strengths,
        },
        "figures": {
            "gaps": str(gap_figure),
            "channel_strengths": str(strength_figure),
        },
    }
    data_path = output_directory / "dynamical_schwinger_data.json"
    data_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    print(f"wrote {data_path}")
    print(f"wrote {gap_figure}")
    print(f"wrote {strength_figure}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--npts", type=int, default=7)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--flux-cutoffs", type=int, nargs="+", default=(1, 2, 3))
    parser.add_argument("--nroots", type=int, default=32)
    args = parser.parse_args()
    run(
        args.output_directory,
        npts=args.npts,
        length=args.length,
        coupling=args.coupling,
        flux_cutoffs=args.flux_cutoffs,
        nroots=args.nroots,
    )


if __name__ == "__main__":
    main()
