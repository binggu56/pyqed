"""Diagnostic for analytic commensurate finite-q GDF electron-phonon response."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import (
    KPointTransitionSpace,
    commensurate_gdf_screened_tda_kernel_derivative,
    electron_phonon_mo_couplings,
    periodic_tda_operator,
)
from pyqed.qchem.pbc import Cell, commensurate_gdf_q_derivative
from pyqed.units import au2ev


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_finite_q_gdf_screened_bse.png"),
    )
    parser.add_argument("--recip-cut", type=int, default=6)
    parser.add_argument("--pair-cut", type=int, default=1)
    args = parser.parse_args()

    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mean_field = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.7,
        real_cut=args.pair_cut,
        pair_cut=args.pair_cut,
        recip_cut=args.recip_cut,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="sto-3g",
        reciprocal_kernel="full",
        recip_cut=args.recip_cut,
        pair_cut=args.pair_cut,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)
    q_index = next(
        index
        for index, qpoint in enumerate(mean_field.with_df.qpts)
        if np.linalg.norm(qpoint) > 1.0e-12
    )
    qpoint = mean_field.with_df.qpts[q_index]
    derivative = commensurate_gdf_q_derivative(
        mean_field,
        qpoint,
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )

    space = KPointTransitionSpace(mean_field, qpts="mesh")
    space_q_index = space.find_qpoint_index(qpoint)
    mo_couplings, _kq_indices = electron_phonon_mo_couplings(
        space,
        space_q_index,
        derivative.fock_derivative,
        overlap_derivative=derivative.overlap_derivative,
    )
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    total_kernel1 = commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        derivative,
    )
    kernel_components = derivative.gdf_screened_kernel_derivative_components
    bare_kernel1 = kernel_components["bare"]
    screened_kernel1 = kernel_components["screened"]
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell.lattice_vectors).T
    scaled_k = np.asarray(mean_field.kpts) @ np.linalg.inv(reciprocal)
    energies = np.asarray(mean_field.mo_energy) * au2ev
    components = {
        "Explicit": derivative.explicit_fock_derivative,
        "Induced": derivative.induced_fock_derivative,
        "Total": derivative.fock_derivative,
    }
    component_norms = {
        name: np.asarray([np.linalg.norm(block) for block in blocks])
        for name, blocks in components.items()
    }

    plt.rcParams.update({"font.size": 9, "axes.linewidth": 0.8})
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), constrained_layout=True)
    ax = axes[0, 0]
    for band in range(energies.shape[1]):
        ax.plot(scaled_k[:, 0], energies[:, band], "o-", lw=1.4, ms=4)
    ax.set_xlabel(r"fractional $k_x$")
    ax.set_ylabel("KRHF energy (eV)")
    ax.set_title("(a) Two-k electronic reference", loc="left")
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    x = np.arange(mean_field.nkpts)
    width = 0.24
    for offset, (name, values) in zip((-width, 0.0, width), component_norms.items()):
        ax.bar(x + offset, values, width=width, label=name)
    ax.set_xticks(x, [rf"$k_{index}$" for index in x])
    ax.set_ylabel(r"$|F_q^{[1]}(k)|_F$ (a.u.)")
    ax.set_title("(b) Static CPHF response", loc="left")
    ax.legend(frameon=False)

    mo_image = np.hstack([np.abs(block) for block in mo_couplings])
    ax = axes[0, 2]
    image = ax.imshow(mo_image, cmap="viridis", aspect="auto")
    ax.axvline(cell.nao - 0.5, color="white", lw=0.8)
    ax.set_xlabel(r"MO column: $k_0$ block | $k_1$ block")
    ax.set_ylabel(r"MO row at $k+q$")
    ax.set_title(r"(c) $|g_{mn}(k,q)|$", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1, 0]
    image = ax.imshow(np.abs(bare_kernel1), cmap="magma", aspect="auto")
    ax.set_xlabel(r"source transition at $Q=0$")
    ax.set_ylabel(r"target transition at $Q=q$")
    ax.set_title(r"(d) $|K_q^{[1],\mathrm{bare}}|$", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1, 1]
    image = ax.imshow(np.abs(screened_kernel1), cmap="cividis", aspect="auto")
    ax.set_xlabel(r"source transition at $Q=0$")
    ax.set_ylabel(r"target transition at $Q=q$")
    ax.set_title(r"(e) $|K_q^{[1],\mathrm{screened}}|$", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046)

    ax = axes[1, 2]
    image = ax.imshow(np.abs(total_kernel1), cmap="magma", aspect="auto")
    ax.set_xlabel(r"source transition at $Q=0$")
    ax.set_ylabel(r"target transition at $Q=q$")
    ax.set_title(r"(f) $|K_q^{[1],\mathrm{total}}|$", loc="left")
    fig.colorbar(image, ax=ax, fraction=0.046)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    payload = {
        "scf_energy": float(mean_field.e_tot),
        "recip_cut": int(args.recip_cut),
        "pair_cut": int(args.pair_cut),
        "qpoint_cartesian": np.asarray(qpoint).tolist(),
        "qpoint_fractional": derivative.transform.scaled_qpoint(qpoint).tolist(),
        "mesh": list(derivative.mesh),
        "component_frobenius_norms": {
            name: values.tolist() for name, values in component_norms.items()
        },
        "maximum_mo_coupling": float(
            max(np.max(np.abs(block)) for block in mo_couplings)
        ),
        "maximum_bare_kernel_derivative": float(np.max(np.abs(bare_kernel1))),
        "maximum_screened_kernel_derivative": float(
            np.max(np.abs(screened_kernel1))
        ),
        "maximum_total_kernel_derivative": float(np.max(np.abs(total_kernel1))),
        "kernel_component_frobenius_norms": {
            name: float(np.linalg.norm(values))
            for name, values in kernel_components.items()
        },
        "kernel_shape": list(total_kernel1.shape),
        "cphf_residual_norm": float(derivative.response.residual_norm),
        "cphf_iterations": int(derivative.response.niter),
        "star_symmetry_residuals": derivative.info["star_symmetry_residuals"],
        "directional_response": derivative.info["directional_response"],
        "seconds": float(derivative.info["seconds"]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
