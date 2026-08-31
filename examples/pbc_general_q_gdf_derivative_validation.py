"""Validate the screened GDF/BSE derivative at a non-self-opposite q point."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import (
    KPointTransitionSpace,
    periodic_tda_operator,
    validate_commensurate_gdf_screened_tda_kernel_derivative,
)
from pyqed.qchem.pbc import Cell, commensurate_gdf_q_derivative


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_general_q_gdf_derivative_validation.png"),
    )
    parser.add_argument("--recip-cut", type=int, default=2)
    parser.add_argument("--pair-cut", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=(2.0e-3, 1.0e-3, 5.0e-4),
    )
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
        nk=(3, 1, 1),
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
    space = KPointTransitionSpace(mean_field, qpts="mesh")
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    phonon_q_index = next(
        index for index in range(space.nqpts) if index != zero_q_index
    )
    qpoint = np.asarray(space.qpts[phonon_q_index])
    q_derivative = commensurate_gdf_q_derivative(
        mean_field,
        qpoint,
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )
    operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    validation = validate_commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        q_derivative,
        steps=args.steps,
    )

    order = np.argsort(validation["steps"])
    steps = validation["steps"][order]
    analytic = np.abs(validation["analytic"])
    finite = np.abs(validation["finite_difference"][order[0]])
    vmax = max(float(np.max(analytic)), float(np.max(finite)))
    colors = {"total": "#0072B2", "bare": "#D55E00", "screened": "#009E73"}
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(8.1, 6.5), constrained_layout=True)
    image = axes[0, 0].imshow(analytic, cmap="magma", vmin=0.0, vmax=vmax)
    axes[0, 0].set_title(r"a  Analytic $|K_q^{[1]}|$", loc="left")
    axes[0, 0].set_xlabel("source transition")
    axes[0, 0].set_ylabel("target transition")
    fig.colorbar(image, ax=axes[0, 0], label="a.u.", fraction=0.046)

    image = axes[0, 1].imshow(finite, cmap="magma", vmin=0.0, vmax=vmax)
    axes[0, 1].set_title(
        rf"b  Finite difference, $h={steps[0]:.1e}$", loc="left"
    )
    axes[0, 1].set_xlabel("source transition")
    axes[0, 1].set_ylabel("target transition")
    fig.colorbar(image, ax=axes[0, 1], label="a.u.", fraction=0.046)

    axes[1, 0].loglog(
        steps,
        validation["relative_error"][order],
        "o-",
        color=colors["total"],
        label="total",
    )
    for name in ("bare", "screened"):
        axes[1, 0].loglog(
            steps,
            validation["component_errors"][name]["relative"][order],
            "s--" if name == "bare" else "^:",
            color=colors[name],
            label=name,
        )
    axes[1, 0].set_xlabel(r"displacement $h$ (bohr)")
    axes[1, 0].set_ylabel("relative Frobenius error")
    axes[1, 0].set_title("c  Derivative validation", loc="left")
    axes[1, 0].grid(alpha=0.22, which="both")
    axes[1, 0].legend(frameon=False)

    leakage = np.asarray(
        [detail["one_body_leakage_norm"] for detail in validation["step_details"]]
    )[order]
    retained = np.asarray(
        [
            detail["one_body_q_sector_norm"]
            for detail in validation["step_details"]
        ]
    )[order]
    axes[1, 1].loglog(
        steps,
        leakage,
        "o-",
        color="#CC79A7",
        label="off-sector leakage",
    )
    axes[1, 1].loglog(
        steps,
        retained,
        "s--",
        color="#5B5B5B",
        label=r"allowed $p\to p+q$ block",
    )
    axes[1, 1].set_xlabel(r"displacement $h$ (bohr)")
    axes[1, 1].set_ylabel("one-body derivative norm (a.u.)")
    axes[1, 1].set_title("d  Supercell sector diagnostic", loc="left")
    axes[1, 1].grid(alpha=0.22, which="both")
    axes[1, 1].legend(frameon=False)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    fig.savefig(args.output.with_suffix(".pdf"))
    payload = {
        "nk": [3, 1, 1],
        "qpoint_cartesian": qpoint.tolist(),
        "qpoint_fractional": q_derivative.transform.scaled_qpoint(qpoint).tolist(),
        "recip_cut": int(args.recip_cut),
        "pair_cut": int(args.pair_cut),
        "steps": validation["steps"].tolist(),
        "relative_error": validation["relative_error"].tolist(),
        "component_relative_error": {
            name: values["relative"].tolist()
            for name, values in validation["component_errors"].items()
        },
        "analytic_component_norm": {
            name: float(np.linalg.norm(value))
            for name, value in validation["analytic_components"].items()
        },
        "one_body_leakage_norm": [
            float(detail["one_body_leakage_norm"])
            for detail in validation["step_details"]
        ],
        "zero_density_residual": float(validation["zero_density_residual"]),
        "largest_reference_residual": float(
            validation["largest_reference_residual"]
        ),
        "supercell_twist": validation["supercell_twist"].tolist(),
        "seconds": float(validation["seconds"]),
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))
    print(f"figure: {args.output}")


if __name__ == "__main__":
    main()
