#!/usr/bin/env python3
"""Validate an exact-Gauss Kogut--Susskind ED/MPS calculation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.lgt import KogutSusskindED, KogutSusskindMPO
from pyqed.mps import (
    DMRG,
    MPO,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
)


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "kogut_susskind_pilot"


def style(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def run(
    output_directory=DEFAULT_OUTPUT,
    *,
    nsites=8,
    length=10.0,
    coupling=1.0,
    mass=0.0,
    flux_cutoff=3,
    bond_dim=32,
    sweeps=8,
    nroots=16,
):
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    parameters = {
        "nsites": int(nsites),
        "length": float(length),
        "coupling": float(coupling),
        "mass": float(mass),
        "flux_cutoff": int(flux_cutoff),
    }

    exact = KogutSusskindED(**parameters).run(nroots=nroots)
    builder = KogutSusskindMPO(**parameters)
    raw = builder.build_mpo()
    full_dense = raw.to_dense() if nsites <= 5 else None
    indices = builder.physical_product_indices(exact)
    if full_dense is None:
        validation_builder = KogutSusskindMPO(
            4,
            4.0,
            coupling=coupling,
            mass=mass,
            flux_cutoff=min(flux_cutoff, 2),
        )
        validation_exact = KogutSusskindED(
            4,
            4.0,
            coupling=coupling,
            mass=mass,
            flux_cutoff=min(flux_cutoff, 2),
        )
        validation_dense = validation_builder.build_mpo().to_dense()
        validation_indices = validation_builder.physical_product_indices(
            validation_exact
        )
        projected = validation_dense[np.ix_(validation_indices, validation_indices)]
        exact_matrix = validation_exact.build_hamiltonian().toarray()
        hermiticity = np.linalg.norm(
            validation_dense - validation_dense.conj().T
        ) / np.linalg.norm(validation_dense)
        gauss_dense = validation_builder.build_gauss_mpo().to_dense()
        gauss_projected = gauss_dense[np.ix_(validation_indices, validation_indices)]
    else:
        projected = full_dense[np.ix_(indices, indices)]
        exact_matrix = exact.hamiltonian.toarray()
        hermiticity = np.linalg.norm(full_dense - full_dense.conj().T) / np.linalg.norm(
            full_dense
        )
        gauss_dense = builder.build_gauss_mpo().to_dense()
        gauss_projected = gauss_dense[np.ix_(indices, indices)]
    projection_error = np.linalg.norm(projected - exact_matrix) / np.linalg.norm(
        exact_matrix
    )
    gauss_error = np.linalg.norm(gauss_projected)

    maps, target, manager = builder.gauss_symmetry()
    hamiltonian = compress_symmetric_mpo(
        MPO(
            dense_to_symmetric_mpo(
                raw.factors,
                maps,
                native_site_storage=True,
            )
        )
    )
    initial = builder.gauss_seed_mps(
        bond_dim=bond_dim,
        seed=7,
        native_site_storage=True,
    )
    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=int(bond_dim),
        init_guess=initial,
        nsweeps=int(sweeps),
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        davidson_max_iter=200,
        noise=1.0e-7,
        performance="symmetric",
    ).run()
    dmrg_seconds = perf_counter() - started
    dmrg_error = abs(float(solver.e_tot) - float(exact.energies[0]))

    ground = exact.states[:, 0]
    vector_source = exact.build_vector_operator() @ ground
    scalar_source = exact.build_scalar_operator() @ ground
    vector_strengths = np.abs(exact.states.conj().T @ vector_source) ** 2
    scalar_strengths = np.abs(exact.states.conj().T @ scalar_source) ** 2
    vector_root = int(1 + np.argmax(vector_strengths[1:]))
    scalar_root = int(1 + np.argmax(scalar_strengths[1:]))
    gaps = exact.energies - exact.energies[0]

    figure = output_directory / "29_kogut_susskind_ed_mps_pilot.png"
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4), constrained_layout=True)
    roots = np.arange(len(gaps))
    axes[0].plot(roots, gaps / coupling, "o-", color="0.45", label="ED gaps")
    vector_scale = max(vector_strengths.max(), np.finfo(float).eps)
    scalar_scale = max(scalar_strengths.max(), np.finfo(float).eps)
    axes[0].scatter(
        roots,
        gaps / coupling,
        s=190 * vector_strengths / vector_scale + 12,
        facecolors="none",
        edgecolors="C0",
        label=r"electric $O_V$ strength",
    )
    axes[0].scatter(
        roots,
        gaps / coupling,
        s=150 * scalar_strengths / scalar_scale + 10,
        marker="s",
        facecolors="none",
        edgecolors="C1",
        label=r"staggered $O_S$ strength",
    )
    axes[0].set(
        xlabel="ED root",
        ylabel=r"$(E_i-E_0)/g$",
        title=rf"$N={nsites}$, $gL={coupling * length:g}$, $\ell_{{max}}={flux_cutoff}$",
    )
    axes[0].legend(frameon=False, fontsize=9)
    style(axes[0])

    labels = ["MPO/ED", "Hermiticity", "Gauss sector", "DMRG/ED"]
    residuals = [projection_error, hermiticity, gauss_error, dmrg_error]
    axes[1].bar(labels, np.maximum(residuals, 1.0e-16), color=["C0", "C1", "C2", "C3"])
    axes[1].set_yscale("log")
    axes[1].set(
        ylabel="absolute or relative residual",
        title=rf"exact-Gauss DMRG: {dmrg_seconds:.2f} s, $D={bond_dim}$",
    )
    axes[1].tick_params(axis="x", rotation=18)
    style(axes[1])
    fig.savefig(figure, dpi=210)
    plt.close(fig)

    payload = {
        "description": (
            "Open-boundary Kogut-Susskind Schwinger Hamiltonian with a hard "
            "electric-flux cutoff, physical-sector ED, and exact vector Gauss "
            "charges in symmetric DMRG."
        ),
        "fidelity": (
            "Finite open-chain adaptation of the Kogut-Susskind Hamiltonian; "
            "the compact-link algebra is hard truncated and no continuum or "
            "infinite-volume extrapolation is performed."
        ),
        "parameters": parameters,
        "ed_dimension": int(exact.dimension),
        "ed_energies": exact.energies.tolist(),
        "vector_strengths": vector_strengths.tolist(),
        "scalar_strengths": scalar_strengths.tolist(),
        "vector_root": vector_root,
        "scalar_root": scalar_root,
        "M_V_over_g": float(gaps[vector_root] / coupling),
        "M_S_over_g": float(gaps[scalar_root] / coupling),
        "dmrg_ground_energy": float(solver.e_tot),
        "ed_ground_energy": float(exact.energies[0]),
        "dmrg_energy_error": float(dmrg_error),
        "dmrg_seconds": float(dmrg_seconds),
        "dmrg_converged": bool(solver.converged),
        "dmrg_bonds": solver.ground_state.bond_orders(),
        "raw_mpo_max_bond": int(max(raw.bond_orders())),
        "symmetric_mpo_max_bond": int(max(hamiltonian.bond_orders())),
        "projection_relative_error": float(projection_error),
        "hermiticity_residual": float(hermiticity),
        "gauss_projected_norm": float(gauss_error),
        "figure": str(figure),
    }
    data_path = output_directory / "kogut_susskind_pilot.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--nsites", type=int, default=8)
    parser.add_argument("--length", type=float, default=10.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=0.0)
    parser.add_argument("--flux-cutoff", type=int, default=3)
    parser.add_argument("--bond-dim", type=int, default=32)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--nroots", type=int, default=16)
    args = parser.parse_args()
    payload = run(**vars(args))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
