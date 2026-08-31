#!/usr/bin/env python3
"""Validate and benchmark an explicit dynamical Wilson-DVR MPO pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from pyqed.lgt import QuantumSchwingerDVR, WilsonDVRMPO
from pyqed.mps import DMRG


HERE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = HERE / "results" / "wilson_dvr_mpo"


def mpo_inner(left, right):
    environment = np.ones((1, 1), dtype=complex)
    for left_site, right_site in zip(left.factors, right.factors):
        left_site = np.asarray(left_site).reshape(
            left_site.shape[0], left_site.shape[1], -1
        )
        right_site = np.asarray(right_site).reshape(
            right_site.shape[0], right_site.shape[1], -1
        )
        environment = np.einsum(
            "ab,arp,bsp->rs",
            environment,
            left_site.conj(),
            right_site,
            optimize=True,
        )
    return environment[0, 0]


def relative_mpo_error(reference, approximate, reference_norm=None):
    if reference_norm is None:
        reference_norm = float(np.real(mpo_inner(reference, reference)))
    approximate_norm = float(np.real(mpo_inner(approximate, approximate)))
    overlap = float(np.real(mpo_inner(reference, approximate)))
    difference = max(0.0, reference_norm + approximate_norm - 2.0 * overlap)
    return np.sqrt(difference / reference_norm)


def validate_small_dmrg():
    exact = QuantumSchwingerDVR(3, 10.0, flux_cutoff=1).run(nroots=8)
    builder = WilsonDVRMPO(
        3,
        10.0,
        flux_cutoff=1,
        gauss_penalty=25.0,
    )
    raw = builder.build_mpo()
    hamiltonian = raw.compress(64)
    gauss = builder.build_gauss_mpo().compress(64)

    dense = raw.to_dense()
    indices = builder.physical_product_indices(exact)
    projected = dense[np.ix_(indices, indices)]
    exact_matrix = exact.hamiltonian.toarray()
    projected_error = np.linalg.norm(projected - exact_matrix) / np.linalg.norm(
        exact_matrix
    )
    hermiticity = np.linalg.norm(dense - dense.conj().T) / np.linalg.norm(dense)

    initial = builder.product_mps([1, 1, 1], [0, 0, 0])
    started = perf_counter()
    solver = DMRG(
        hamiltonian,
        D=48,
        init_guess=initial,
        nsweeps=8,
        not_conv_err=False,
        sweep_tol=1.0e-9,
        davidson_tol=1.0e-10,
        noise=1.0e-5,
        performance="dense",
    ).run()
    seconds = perf_counter() - started
    gauss_leakage = abs(solver.ground_state.expectation(gauss))
    return {
        "npts": 3,
        "flux_cutoff": 1,
        "full_product_dimension": int(builder.local_dim**builder.npts),
        "physical_dimension": int(exact.dimension),
        "raw_mpo_bond": int(max(raw.bond_orders())),
        "compressed_mpo_bond": int(max(hamiltonian.bond_orders())),
        "projected_matrix_relative_error": float(projected_error),
        "full_mpo_hermiticity_residual": float(hermiticity),
        "ed_ground_energy": float(exact.energies[0]),
        "dmrg_ground_energy": float(solver.e_tot),
        "dmrg_energy_error": float(abs(solver.e_tot - exact.energies[0])),
        "gauss_squared_expectation": float(gauss_leakage),
        "dmrg_seconds": float(seconds),
        "dmrg_converged": bool(solver.converged),
        "dmrg_bonds": solver.ground_state.bond_orders(),
    }


def rank_scan(grids=(3, 5, 7, 9)):
    records = []
    for npts in grids:
        builder = WilsonDVRMPO(npts, 10.0, flux_cutoff=1)
        raw = builder.build_mpo()
        compressed = raw.compress(512)
        records.append(
            {
                "npts": int(npts),
                "term_channels": int(max(raw.bond_orders())),
                "exact_numerical_bond": int(max(compressed.bond_orders())),
                "bond_profile": compressed.bond_orders(),
            }
        )
        print(
            f"N={npts} raw={records[-1]['term_channels']} "
            f"compressed={records[-1]['exact_numerical_bond']}",
            flush=True,
        )
    return records


def compression_scan(npts=7, caps=(4, 8, 12, 16, 20, 24, 26)):
    builder = WilsonDVRMPO(npts, 10.0, flux_cutoff=1)
    reference = builder.build_mpo()
    reference_norm = float(np.real(mpo_inner(reference, reference)))
    records = []
    for cap in caps:
        compressed = reference.compress(cap)
        records.append(
            {
                "bond_cap": int(cap),
                "realized_bond": int(max(compressed.bond_orders())),
                "relative_hilbert_schmidt_error": float(
                    relative_mpo_error(reference, compressed, reference_norm)
                ),
            }
        )
    return records


def style(axis):
    axis.grid(True, which="both", alpha=0.22, linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)


def plot_rank(rank_records, compression_records, output):
    npts = np.asarray([record["npts"] for record in rank_records])
    raw = np.asarray([record["term_channels"] for record in rank_records])
    exact = np.asarray([record["exact_numerical_bond"] for record in rank_records])
    caps = np.asarray([record["bond_cap"] for record in compression_records])
    errors = np.asarray(
        [record["relative_hilbert_schmidt_error"] for record in compression_records]
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    axes[0].plot(npts, raw, "s--", label="separate SOP channels")
    axes[0].plot(npts, exact, "o-", label="exact compressed MPO")
    axes[0].plot(npts, 4 * npts - 2, ":", color="black", label=r"$4N-2$")
    axes[0].set_xlabel("DVR points $N$")
    axes[0].set_ylabel("maximum MPO bond dimension")
    axes[0].set_title("Exact Wilson-DVR MPO rank")
    axes[0].set_xticks(npts)
    axes[0].legend(frameon=False)
    style(axes[0])

    axes[1].semilogy(caps, np.maximum(errors, 1.0e-16), "o-")
    axes[1].axvline(26, color="black", linestyle=":", label="exact rank")
    axes[1].set_xlabel(r"MPO bond cap $W$")
    axes[1].set_ylabel("relative Hilbert–Schmidt error")
    axes[1].set_title(r"$N=7$ rank truncation")
    axes[1].legend(frameon=False)
    style(axes[1])
    fig.savefig(output, dpi=220)
    plt.close(fig)


def plot_validation(record, output):
    labels = ["projected MPO", "Hermiticity", "DMRG energy", r"$\langle G^2\rangle$"]
    values = [
        record["projected_matrix_relative_error"],
        record["full_mpo_hermiticity_residual"],
        record["dmrg_energy_error"],
        record["gauss_squared_expectation"],
    ]
    fig, axis = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    axis.bar(labels, np.maximum(values, 1.0e-16), color=["C0", "C1", "C2", "C3"])
    axis.set_yscale("log")
    axis.set_ylabel("absolute or relative residual")
    axis.set_title(r"$N=3$ Wilson-DVR MPO/DMRG validation")
    style(axis)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def run(output_directory):
    output_directory.mkdir(parents=True, exist_ok=True)
    validation = validate_small_dmrg()
    ranks = rank_scan()
    compression = compression_scan()
    rank_figure = output_directory / "10_wilson_dvr_mpo_rank.png"
    validation_figure = output_directory / "11_wilson_dvr_mpo_validation.png"
    plot_rank(ranks, compression, rank_figure)
    plot_validation(validation, validation_figure)
    payload = {
        "description": (
            "Exact cell MPO with two fermion orbitals, a compact outgoing link, "
            "Jordan-Wigner parity, shortest Wilson strings, electric energy, and "
            "an optional Gauss-law penalty."
        ),
        "validation": validation,
        "rank_scaling": ranks,
        "compression_scan_n7": compression,
        "figures": {
            "rank": str(rank_figure),
            "validation": str(validation_figure),
        },
    }
    data_path = output_directory / "wilson_dvr_mpo_data.json"
    data_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {data_path}")
    print(f"wrote {rank_figure}")
    print(f"wrote {validation_figure}")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    run(args.output_directory)


if __name__ == "__main__":
    main()
