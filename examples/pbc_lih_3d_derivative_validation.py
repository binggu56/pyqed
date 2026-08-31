"""Qualify analytic finite-q GDF/BSE derivatives for 3D rocksalt LiH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np

from pyqed.pbc.gw import (
    KPointTransitionSpace,
    commensurate_gdf_screened_tda_kernel_derivative,
    periodic_tda_operator,
    validate_commensurate_gdf_screened_tda_kernel_derivative,
)
from pyqed.qchem.pbc import (
    Cell,
    commensurate_gdf_q_derivative,
    gdf_q_derivative,
)


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return value


def _compact_lih_basis():
    return {
        "Li": [
            (0, np.asarray([3.2]), np.asarray([[1.0]])),
            (0, np.asarray([0.9]), np.asarray([[1.0]])),
        ],
        "H": [(0, np.asarray([1.2]), np.asarray([[1.0]]))],
    }


def _rocksalt_lih(lattice_constant):
    half = 0.5 * float(lattice_constant)
    lattice = np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )
    return Cell(
        atom=[("Li", (0.0, 0.0, 0.0)), ("H", (half, half, half))],
        a=lattice,
        basis=_compact_lih_basis(),
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()


def _shortest_self_opposite_q(space):
    qpoints = np.asarray(space.qpts, dtype=float)
    candidates = [
        index
        for index, qpoint in enumerate(qpoints)
        if np.linalg.norm(qpoint) > 1.0e-10
        and space.find_qpoint_index(-qpoint) == index
    ]
    if not candidates:
        raise ValueError("kmesh must contain a self-opposite nonzero momentum")
    return min(candidates, key=lambda index: np.linalg.norm(qpoints[index]))


def _difference(actual, reference):
    actual = np.asarray(actual)
    reference = np.asarray(reference)
    delta = actual - reference
    return {
        "max_abs": float(np.max(np.abs(delta))),
        "relative_frobenius": float(
            np.linalg.norm(delta)
            / max(np.linalg.norm(reference), np.finfo(float).tiny)
        ),
    }


def _block_difference(actual, reference):
    return _difference(np.asarray(actual), np.asarray(reference))


def _plot(payload, output):
    output = Path(output).expanduser().resolve()
    component_names = (
        "overlap_derivative",
        "explicit_fock_derivative",
        "induced_fock_derivative",
        "fock_derivative",
        "screened_bse_kernel_derivative",
    )
    component_labels = (
        r"$S^{[1]}$",
        r"$F^{[1]}_{\mathrm{explicit}}$",
        r"$F^{[1]}_{\mathrm{CPHF}}$",
        r"$F^{[1]}$",
        r"$K^{[1]}_{\mathrm{BSE}}$",
    )
    differences = payload["primitive_vs_commensurate"]
    validation = payload["finite_difference_validation"]
    steps = np.asarray(validation["steps"])
    order = np.argsort(steps)
    analytic = np.asarray(validation["analytic_abs"])
    finite = np.asarray(validation["finite_difference_abs"])[order[0]]
    vmax = max(float(np.max(analytic)), float(np.max(finite)))

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.4,
            "savefig.dpi": 360,
        }
    )
    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.35), constrained_layout=True)

    values = [differences[name]["relative_frobenius"] for name in component_names]
    axes[0, 0].bar(np.arange(len(values)), values, color="#0072B2", width=0.68)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_xticks(np.arange(len(values)), component_labels)
    axes[0, 0].set_ylabel("Relative Frobenius difference")
    axes[0, 0].set_title("a  Primitive cell vs supercell", loc="left")

    image = axes[0, 1].imshow(analytic, cmap="magma", vmin=0.0, vmax=vmax)
    axes[0, 1].set(
        xlabel="source transition",
        ylabel="target transition",
        title=r"b  Analytic $|K_q^{[1]}|$",
    )
    fig.colorbar(image, ax=axes[0, 1], label="a.u.", fraction=0.046)

    image = axes[1, 0].imshow(finite, cmap="magma", vmin=0.0, vmax=vmax)
    axes[1, 0].set(
        xlabel="source transition",
        ylabel="target transition",
        title=rf"c  Finite difference, $h={steps[order[0]]:.1e}$",
    )
    fig.colorbar(image, ax=axes[1, 0], label="a.u.", fraction=0.046)

    axes[1, 1].loglog(
        steps[order],
        np.asarray(validation["relative_error"])[order],
        "o-",
        color=colors[0],
        label="total",
    )
    for name, marker, color in (
        ("bare", "s--", colors[1]),
        ("screened", "^:", colors[2]),
    ):
        axes[1, 1].loglog(
            steps[order],
            np.asarray(validation["component_relative_error"][name])[order],
            marker,
            color=color,
            label=name,
        )
    axes[1, 1].set(
        xlabel=r"displacement $h$ (bohr)",
        ylabel="Relative Frobenius error",
        title="d  Central-difference validation",
    )
    axes[1, 1].legend(frameon=False)

    for axis in axes.reshape(-1):
        axis.grid(alpha=0.2, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def run(args):
    started = time.perf_counter()
    print(
        f"LiH kmesh={tuple(args.kmesh)}, recip_cut={args.recip_cut}, "
        f"pair_cut={args.pair_cut}",
        flush=True,
    )
    cell = _rocksalt_lih(args.lattice_constant)
    mean_field = cell.KRHF(
        nk=tuple(args.kmesh),
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
        storage="memory",
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)
    if not mean_field.converged:
        raise RuntimeError("3D LiH GDF-KRHF did not converge")
    scf_seconds = time.perf_counter() - started
    print(f"GDF-KRHF converged in {scf_seconds:.2f} s", flush=True)

    space = KPointTransitionSpace(mean_field, qpts="mesh")
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    phonon_q_index = _shortest_self_opposite_q(space)
    qpoint = np.asarray(space.qpts[phonon_q_index], dtype=float)
    mode = [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]

    primitive_started = time.perf_counter()
    primitive = gdf_q_derivative(
        mean_field,
        qpoint,
        mode,
        cphf_tol=1.0e-10,
    )
    primitive_seconds = time.perf_counter() - primitive_started
    print(
        f"primitive q derivative completed in {primitive_seconds:.2f} s",
        flush=True,
    )

    reference_started = time.perf_counter()
    commensurate = commensurate_gdf_q_derivative(
        mean_field,
        qpoint,
        mode,
        cphf_tol=1.0e-10,
    )
    commensurate_seconds = time.perf_counter() - reference_started
    largest_reference_residual = max(
        float(value)
        for name, value in commensurate.info["reference_residuals"].items()
        if name.endswith("_relative")
    )
    print(
        "commensurate q derivative completed in "
        f"{commensurate_seconds:.2f} s; reference residual "
        f"{largest_reference_residual:.3e}",
        flush=True,
    )

    operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    primitive_kernel = commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        primitive,
    )
    reference_kernel = commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        commensurate,
    )

    differences = {
        name: _block_difference(
            getattr(primitive, name),
            getattr(commensurate, name),
        )
        for name in (
            "overlap_derivative",
            "explicit_fock_derivative",
            "induced_fock_derivative",
            "fock_derivative",
        )
    }
    differences["screened_bse_kernel_derivative"] = _difference(
        primitive_kernel,
        reference_kernel,
    )

    validation = validate_commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        commensurate,
        steps=args.steps,
        representation_tol=args.representation_tol,
    )
    print(
        "finite-difference relative errors: "
        + ", ".join(f"{value:.3e}" for value in validation["relative_error"]),
        flush=True,
    )
    reference_residuals = {
        key: float(value)
        for key, value in commensurate.info["reference_residuals"].items()
    }
    payload = {
        "system": "rocksalt LiH",
        "basis": "compact three-function all-electron validation basis",
        "auxbasis": "sto-3g",
        "kmesh": list(args.kmesh),
        "recip_cut": int(args.recip_cut),
        "pair_cut": int(args.pair_cut),
        "qpoint_cartesian": qpoint.tolist(),
        "qpoint_fractional": commensurate.transform.scaled_qpoint(qpoint).tolist(),
        "mode": mode,
        "scf_energy_Ha": float(mean_field.e_tot),
        "primitive_vs_commensurate": differences,
        "finite_difference_validation": {
            "steps": validation["steps"].tolist(),
            "relative_error": validation["relative_error"].tolist(),
            "component_relative_error": {
                name: values["relative"].tolist()
                for name, values in validation["component_errors"].items()
            },
            "analytic_abs": np.abs(validation["analytic"]).tolist(),
            "finite_difference_abs": np.abs(
                validation["finite_difference"]
            ).tolist(),
            "largest_reference_residual": float(
                validation["largest_reference_residual"]
            ),
            "one_body_leakage_norm": [
                float(detail["one_body_leakage_norm"])
                for detail in validation["step_details"]
            ],
        },
        "reference_residuals": reference_residuals,
        "timing_seconds": {
            "scf": float(scf_seconds),
            "primitive_derivative": float(primitive_seconds),
            "commensurate_derivative": float(commensurate_seconds),
            "finite_difference_validation": float(validation["seconds"]),
            "total": float(time.perf_counter() - started),
        },
        "primitive_cached_bytes": int(
            primitive.info.get(
                "cached_bytes",
                primitive.primitive_engine.info.get("cached_bytes", 0),
            )
        ),
        "fidelity": (
            "primitive-cell full-reciprocal GDF derivative compared with an "
            "independent commensurate-supercell reference; static direct-RPA "
            "TDA screened-kernel derivative validated by displaced SCF"
        ),
    }
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    _plot(payload, args.figure)
    payload["figure"] = str(Path(args.figure).expanduser().resolve())
    payload["pdf"] = str(Path(args.figure).expanduser().resolve().with_suffix(".pdf"))
    output.write_text(json.dumps(_jsonable(payload), indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output),
                "figure": payload["figure"],
                "primitive_vs_commensurate": differences,
                "finite_difference_relative_error": validation[
                    "relative_error"
                ].tolist(),
                "timing_seconds": payload["timing_seconds"],
            },
            indent=2,
        )
    )
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kmesh", type=int, nargs=3, default=(2, 2, 2))
    parser.add_argument("--lattice-constant", type=float, default=7.72)
    parser.add_argument("--recip-cut", type=int, default=5)
    parser.add_argument("--pair-cut", type=int, default=2)
    parser.add_argument(
        "--steps",
        type=float,
        nargs="+",
        default=(2.0e-3, 1.0e-3, 5.0e-4),
    )
    parser.add_argument("--representation-tol", type=float, default=1.0e-6)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_lih_3d_derivative_validation.json"),
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=Path("/private/tmp/pbc_lih_3d_derivative_validation.png"),
    )
    args = parser.parse_args()
    if any(value < 2 or value % 2 for value in args.kmesh):
        parser.error("each kmesh dimension must be a positive even integer >= 2")
    if args.recip_cut < 1 or args.pair_cut < 0:
        parser.error("recip-cut must be positive and pair-cut non-negative")
    run(args)


if __name__ == "__main__":
    main()
