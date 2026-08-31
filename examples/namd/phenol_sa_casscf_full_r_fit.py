#!/usr/bin/env python3
"""Fit the full planar phenol O-H regime while flagging the missing-root bridge."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-codex")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from pyqed.units import au2ev
from pyqed.ldr import AbInitioFit
from pyqed.mps.functional import FunctionalTT, PiecewisePCHIP


HARTREE_TO_EV = au2ev


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _global_fit(coordinates, values, *, degree, hermitian, seed):
    return FunctionalTT(
        degrees=(int(degree),),
        rank=values.shape[-1] ** 2,
        bounds=((float(coordinates[0]), float(coordinates[-1])),),
        hermitian=bool(hermitian),
        normalization="frobenius",
        regularization=1.0e-12,
        sweeps=30,
        rtol=1.0e-13,
        patience=4,
        random_state=int(seed),
    ).fit(coordinates[:, None], values)


def _save_fit(directory, radii, anchor, shift, energy, link, info, config):
    fit = AbInitioFit((radii,), 3, anchor=(anchor,), energy_shift=shift)
    fit.energy = energy
    fit.links = (link,)
    fit.info = info
    fit.config = config
    fit.seconds = float(info["seconds"])
    fit.success = True
    fit.message = "fitted"
    fit.frames.points.update((index,) for index in range(len(radii)))
    fit.save(
        directory,
        labels=("R_OH",),
        metadata={
            "system": "phenol",
            "scope": "full planar O-H reaction coordinate",
            "electronic_space": "tracked three-state P gauge",
            "low_confidence_bridge_angstrom": [1.85, 1.95],
        },
    )
    return fit


def _leave_one_out(radii, values):
    errors = []
    for index in range(1, len(radii) - 1):
        keep = np.arange(len(radii)) != index
        prediction = PiecewisePCHIP(hermitian=True).fit(
            radii[keep], values[keep]
        ).predict(np.asarray((radii[index],)))
        error = np.max(
            np.abs(
                np.linalg.eigvalsh(prediction)
                - np.linalg.eigvalsh(values[index])
            )
        )
        errors.append((float(radii[index]), float(error * HARTREE_TO_EV)))
    return np.asarray(errors)


def _plot(
    output,
    radii,
    sampled_energy,
    dense_radii,
    piecewise_energy,
    global_energy,
    link_midpoints,
    sampled_link_minimum,
    dense_link_midpoints,
    piecewise_link_minimum,
    global_link_minimum,
    leave_one_out,
    bridge,
    degree,
):
    colors = ("#0072B2", "#D55E00", "#009E73")
    figure, panels = plt.subplots(
        2, 2, figsize=(11.4, 7.2), constrained_layout=True
    )
    for state, color in enumerate(colors):
        panels[0, 0].plot(
            dense_radii,
            piecewise_energy[:, state],
            color=color,
            lw=1.6,
        )
        panels[0, 0].plot(
            dense_radii,
            global_energy[:, state],
            color=color,
            ls="--",
            lw=1.0,
        )
        panels[0, 0].plot(
            radii,
            sampled_energy[:, state],
            "o",
            color=color,
            ms=3.5,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
        panels[0, 1].plot(
            dense_radii,
            piecewise_energy[:, state],
            color=color,
            lw=1.6,
        )
        panels[0, 1].plot(
            dense_radii,
            global_energy[:, state],
            color=color,
            ls="--",
            lw=1.0,
        )
        panels[0, 1].plot(
            radii,
            sampled_energy[:, state],
            "o",
            color=color,
            ms=3.5,
            markeredgecolor="white",
            markeredgewidth=0.35,
        )
        difference = np.abs(global_energy[:, state] - piecewise_energy[:, state])
        panels[1, 0].semilogy(
            dense_radii,
            np.maximum(difference, 1.0e-8),
            color=color,
            lw=1.3,
        )
    panels[1, 0].semilogy(
        leave_one_out[:, 0],
        leave_one_out[:, 1],
        "kx",
        ms=4.5,
        mew=0.9,
        label="PCHIP leave-one-out",
    )
    panels[1, 1].plot(
        dense_link_midpoints,
        piecewise_link_minimum,
        color="#0072B2",
        lw=1.6,
        label="piecewise fit",
    )
    panels[1, 1].plot(
        dense_link_midpoints,
        global_link_minimum,
        color="#D55E00",
        ls="--",
        lw=1.2,
        label=rf"global degree {degree}",
    )
    panels[1, 1].plot(
        link_midpoints,
        sampled_link_minimum,
        "ko",
        ms=3.2,
        label="sampled link",
    )

    for panel in panels.flat:
        panel.axvspan(*bridge, color="0.75", alpha=0.28, lw=0.0)
        panel.grid(alpha=0.18, which="both")
        panel.set_xlabel(r"$R_{OH}$ ($\AA$)")
    panels[0, 0].set(
        ylabel=r"$E-E_0(R_{eq})$ (eV)",
        title="a  Full planar regime",
    )
    panels[0, 1].set(
        xlim=(1.65, 2.15),
        ylim=(2.95, 4.15),
        ylabel=r"$E-E_0(R_{eq})$ (eV)",
        title="b  Dissociation-region detail",
    )
    panels[1, 0].set(
        ylabel="spectral error (eV)",
        ylim=(1.0e-4, 1.0),
        title="c  Global-fit discrepancy and cross-validation",
    )
    panels[1, 0].legend(loc="upper left", fontsize=8.0)
    panels[1, 1].set(
        ylabel=r"minimum $\sigma(S^P)$",
        ylim=(0.0, 1.08),
        title="d  Link field retains the flagged bridge",
    )
    panels[1, 1].legend(loc="lower right", fontsize=8.0)
    state_handles = [
        Line2D((0,), (0,), color=color, lw=1.6, label=f"state {state}")
        for state, color in enumerate(colors)
    ]
    model_handles = [
        Line2D((0,), (0,), color="0.2", lw=1.6, label="piecewise PCHIP"),
        Line2D(
            (0,),
            (0,),
            color="0.2",
            lw=1.1,
            ls="--",
            label=rf"global degree {degree}",
        ),
        Line2D(
            (0,),
            (0,),
            color="0.2",
            marker="o",
            ls="none",
            ms=4,
            label="ab initio sample",
        ),
    ]
    figure.legend(
        handles=state_handles + model_handles,
        loc="outside upper center",
        ncol=6,
        fontsize=8.5,
        frameon=False,
    )
    png = output / "phenol_sa6_full_r_fit_diagnostics.png"
    pdf = output / "phenol_sa6_full_r_fit_diagnostics.pdf"
    figure.savefig(png, dpi=320)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "/private/tmp/phenol_sa6_p_gauge_20260820/"
            "phenol_sa6_tracked3_p_gauge.npz"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/phenol_sa6_full_r_fit_20260820"),
    )
    parser.add_argument("--global-degree", type=int, default=6)
    parser.add_argument("--dense-points", type=int, default=5001)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    with np.load(args.data, allow_pickle=False) as archive:
        radii = np.asarray(archive["radii"], dtype=float)
        anchor = int(archive["anchor"])
        p_hamiltonian = np.asarray(archive["p_hamiltonian"])
        p_links = np.asarray(archive["p_links"])
        tracked_singular = np.asarray(archive["tracked_singular_values"])
    if p_hamiltonian.shape != (len(radii), 3, 3):
        raise ValueError("expected a complete three-state radial Hamiltonian")
    if p_links.shape != (len(radii) - 1, 3, 3):
        raise ValueError("expected one three-state link per radial edge")

    shift = float(np.min(np.linalg.eigvalsh(p_hamiltonian[anchor])))
    shifted = p_hamiltonian - shift * np.eye(3)
    link_midpoints = 0.5 * (radii[:-1] + radii[1:])
    edge_minimum = tracked_singular[:, -1]
    bridge_index = int(np.argmin(edge_minimum))
    bridge = (float(radii[bridge_index]), float(radii[bridge_index + 1]))

    piecewise_energy = PiecewisePCHIP(hermitian=True).fit(radii, shifted)
    piecewise_link = PiecewisePCHIP(hermitian=True).fit(link_midpoints, p_links)
    global_energy = _global_fit(
        radii,
        shifted,
        degree=args.global_degree,
        hermitian=True,
        seed=17,
    )
    global_link = _global_fit(
        link_midpoints,
        p_links,
        degree=args.global_degree,
        hermitian=True,
        seed=31,
    )

    dense_radii = np.linspace(radii[0], radii[-1], args.dense_points)
    dense_links = np.linspace(
        link_midpoints[0], link_midpoints[-1], args.dense_points
    )
    piecewise_h_dense = piecewise_energy.predict(dense_radii[:, None])
    global_h_dense = global_energy.predict(dense_radii[:, None])
    piecewise_l_dense = piecewise_link.predict(dense_links[:, None])
    global_l_dense = global_link.predict(dense_links[:, None])
    piecewise_singular = np.linalg.svd(piecewise_l_dense, compute_uv=False)
    global_singular = np.linalg.svd(global_l_dense, compute_uv=False)
    piecewise_node_h = piecewise_energy.predict(radii[:, None])
    piecewise_node_l = piecewise_link.predict(link_midpoints[:, None])
    global_node_h = global_energy.predict(radii[:, None])
    global_node_l = global_link.predict(link_midpoints[:, None])
    leave_one_out = _leave_one_out(radii, shifted)

    exact_eigenvalues = np.linalg.eigvalsh(shifted) * HARTREE_TO_EV
    piecewise_eigenvalues = (
        np.linalg.eigvalsh(piecewise_h_dense) * HARTREE_TO_EV
    )
    global_eigenvalues = np.linalg.eigvalsh(global_h_dense) * HARTREE_TO_EV
    piecewise_h_error = np.linalg.norm(
        piecewise_node_h - shifted, axis=(-2, -1)
    ) * HARTREE_TO_EV
    piecewise_l_error = np.linalg.norm(
        piecewise_node_l - p_links, axis=(-2, -1)
    )
    global_h_error = np.linalg.norm(
        global_node_h - shifted, axis=(-2, -1)
    ) * HARTREE_TO_EV
    global_l_error = np.linalg.norm(
        global_node_l - p_links, axis=(-2, -1)
    )
    hermiticity = float(
        np.max(
            np.linalg.norm(
                piecewise_h_dense
                - piecewise_h_dense.conj().swapaxes(-1, -2),
                axis=(-2, -1),
            )
        )
    )

    seconds = time.perf_counter() - started
    piecewise_info = {
        "backend": "piecewise-pchip-full-radial-p-gauge",
        "samples": len(radii),
        "maximum_energy_grid_error_ev": float(np.max(piecewise_h_error)),
        "maximum_link_grid_error": float(np.max(piecewise_l_error)),
        "dense_energy_hermiticity_defect": hermiticity,
        "dense_link_singular_range": [
            float(np.min(piecewise_singular)),
            float(np.max(piecewise_singular)),
        ],
        "leave_one_out_spectral_rms_ev": float(
            np.sqrt(np.mean(leave_one_out[:, 1] ** 2))
        ),
        "leave_one_out_spectral_maximum_ev": float(
            np.max(leave_one_out[:, 1])
        ),
        "leave_one_out_maximum_radius_angstrom": float(
            leave_one_out[np.argmax(leave_one_out[:, 1]), 0]
        ),
        "low_confidence_bridge": {
            "left_radius_angstrom": bridge[0],
            "right_radius_angstrom": bridge[1],
            "midpoint_angstrom": float(link_midpoints[bridge_index]),
            "minimum_sampled_singular_value": float(edge_minimum[bridge_index]),
        },
        "source": str(args.data),
        "seconds": seconds,
    }
    global_info = {
        "backend": "global-chebyshev-functional-tt-comparison",
        "degree": args.global_degree,
        "maximum_energy_grid_error_ev": float(np.max(global_h_error)),
        "rms_energy_grid_error_ev": float(
            np.sqrt(np.mean(global_h_error**2))
        ),
        "maximum_link_grid_error": float(np.max(global_l_error)),
        "maximum_dense_spectral_difference_from_piecewise_ev": float(
            np.max(np.abs(global_eigenvalues - piecewise_eigenvalues))
        ),
        "dense_link_singular_range": [
            float(np.min(global_singular)),
            float(np.max(global_singular)),
        ],
        "source": str(args.data),
        "seconds": seconds,
    }
    fields = args.output / "fields"
    global_fields = args.output / "global_functional_fields"
    _save_fit(
        fields,
        radii,
        anchor,
        shift,
        piecewise_energy,
        piecewise_link,
        piecewise_info,
        {
            "representation": "tracked-three-state-p-gauge",
            "interpolation": "piecewise-pchip",
            "whole_radial_regime": True,
            "missing_root_policy": "retain-and-flag-low-confidence-bridge",
        },
    )
    _save_fit(
        global_fields,
        radii,
        anchor,
        shift,
        global_energy,
        global_link,
        global_info,
        {
            "representation": "tracked-three-state-p-gauge",
            "interpolation": "global-chebyshev-functional-tt",
            "degree": args.global_degree,
            "comparison_only": True,
        },
    )

    restored = AbInitioFit.load(fields)
    roundtrip_h_error = float(
        np.max(
            np.abs(
                restored.energy.predict(dense_radii[:, None])
                - piecewise_h_dense
            )
        )
    )
    roundtrip_l_error = float(
        np.max(
            np.abs(
                restored.links[0].predict(dense_links[:, None])
                - piecewise_l_dense
            )
        )
    )
    restored.close()
    png, pdf = _plot(
        args.output,
        radii,
        exact_eigenvalues,
        dense_radii,
        piecewise_eigenvalues,
        global_eigenvalues,
        link_midpoints,
        edge_minimum,
        dense_links,
        piecewise_singular[:, -1],
        global_singular[:, -1],
        leave_one_out,
        bridge,
        args.global_degree,
    )
    summary = {
        "fit_completed": bool(
            np.max(piecewise_h_error) <= 1.0e-10
            and np.max(piecewise_l_error) <= 1.0e-10
            and hermiticity <= 1.0e-12
            and roundtrip_h_error <= 1.0e-12
            and roundtrip_l_error <= 1.0e-12
        ),
        "scope": "planar R_OH = 0.90 to 3.00 angstrom",
        "interpretation": (
            "exploratory whole-regime interpolation; the 1.85-1.95 angstrom "
            "missing-root edge is retained as low confidence"
        ),
        "recommended_model": "piecewise PCHIP",
        "energy_shift_hartree": shift,
        "piecewise": piecewise_info,
        "global_comparison": global_info,
        "roundtrip_maximum_energy_error_hartree": roundtrip_h_error,
        "roundtrip_maximum_link_error": roundtrip_l_error,
        "fields": str(fields),
        "global_comparison_fields": str(global_fields),
        "figures": {"png": str(png), "pdf": str(pdf)},
        "new_quantum_chemistry_calculations": 0,
    }
    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(_jsonable(summary), indent=2) + "\n")
    print(json.dumps(_jsonable(summary), indent=2), flush=True)


if __name__ == "__main__":
    main()
