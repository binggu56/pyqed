"""Analytic native-GDF Gamma exciton-phonon derivative for periodic LiH."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np

from pyqed.pbc.gw import (
    KPointTransitionSpace,
    diagonal_g0w0,
    gamma_gdf_g0w0_energy_derivative,
    phonon_tda_electron_phonon_coupling,
    periodic_bse_matrices,
    periodic_tda_operator,
)
from pyqed.qchem.pbc import Cell
from pyqed.units import amu_to_au, au2ev


def _cell(coords):
    return Cell(
        atom=[("Li", tuple(coords[0])), ("H", tuple(coords[1]))],
        a=np.diag([6.5, 6.8, 7.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()


def _mean_field(coords):
    cell = _cell(coords)
    mean_field = cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        reciprocal_kernel="full",
        recip_cut=2,
        pair_cut=0,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    )
    return mean_field.run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _phonon_mean_field(coords):
    return _cell(coords).KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="reciprocal",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _gap(mean_field, occupied_band, virtual_band):
    energies = np.asarray(mean_field.mo_energy, dtype=float).reshape(-1)
    return float(energies[virtual_band] - energies[occupied_band])


def _frozen_space_and_bse(mean_field, reference_orbitals):
    coefficient, occupation = reference_orbitals
    mean_field.mo_coeff = [coefficient.copy()]
    mean_field.mo_occ = [occupation.copy()]
    space = KPointTransitionSpace(mean_field, qpts="gamma")
    block = periodic_bse_matrices(
        space,
        q_index=0,
        coulomb_component="gdf",
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
    )
    return space, block


def run(output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    coords = np.asarray([[1.5, 2.4, 2.6], [3.3, 2.8, 2.9]])
    mean_field = _mean_field(coords)
    phonon_mean_field = _phonon_mean_field(coords)
    hessian = phonon_mean_field.Hessian()
    hessian.kernel(
        second_derivative_backend="analytic",
        enforce_acoustic_sum_rule=True,
    )
    frequencies_cm1, modes = hessian.frequencies(
        units="cm-1",
        return_eigenvectors=True,
    )
    positive = np.flatnonzero(hessian.freq_au > 1.0e-7)
    if not positive.size:
        raise RuntimeError("No stable nonzero Gamma phonon mode was found")
    branch = int(positive[np.argmax(hessian.freq_au[positive])])
    frequency = float(hessian.freq_au[branch])
    phonon_mode = hessian.mode([0.0, 0.0, 0.0], branch)
    mode = phonon_mode.eigenvector.real

    space = KPointTransitionSpace(mean_field, qpts="gamma")
    operator = periodic_tda_operator(
        space,
        q_index=0,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    occupations = np.asarray(mean_field.mo_occ, dtype=float).reshape(-1)
    occupied_band = int(np.flatnonzero(occupations > 1.0e-8)[-1])
    virtual_band = int(np.flatnonzero(occupations < 1.0e-8)[0])
    transitions = space.transitions(0)
    transition_index = next(
        index
        for index, transition in enumerate(transitions)
        if transition.occ_band == occupied_band
        and transition.vir_band == virtual_band
    )
    coupling = phonon_tda_electron_phonon_coupling(
        operator,
        hessian,
        0,
        branch,
        kernel_derivative="screened_gdf",
        cphf_tol=1.0e-11,
    )
    probe = np.zeros(operator.shape[1])
    probe[transition_index] = 1.0
    analytic_total = float(
        coupling.derivative.matvec(probe)[transition_index].real
    )
    analytic_bare = float(
        coupling.gdf_kernel_derivative_components["bare"][
            transition_index,
            transition_index,
        ].real
    )
    analytic_screened = float(
        coupling.gdf_kernel_derivative_components["screened"][
            transition_index,
            transition_index,
        ].real
    )
    analytic_derivatives = np.asarray(
        [
            analytic_total - analytic_bare - analytic_screened,
            analytic_bare,
            analytic_screened,
            analytic_total,
        ]
    )
    zero_point_scale = au2ev * 1.0e3 / np.sqrt(2.0 * frequency)
    analytic_couplings_mev = analytic_derivatives * zero_point_scale
    gw_eta = 0.05
    analytic_qp_coupling_mev = (
        gamma_gdf_g0w0_energy_derivative(
            coupling.gdf_screened_interaction_derivative,
            band_index=occupied_band,
            eta=gw_eta,
        )
        * zero_point_scale
    )

    masses = np.asarray(
        mean_field.cell.unit_molecule.atom_mass_list(),
        dtype=float,
    ) * amu_to_au
    cartesian_mode = mode.reshape(2, 3) / np.sqrt(masses)[:, None]
    steps = np.asarray([0.08, 0.04, 0.02, 0.01])
    reference_orbitals = (
        np.asarray(mean_field.mo_coeff, dtype=np.complex128).reshape(
            mean_field.cell.nao,
            -1,
        ),
        np.asarray(mean_field.mo_occ, dtype=float).reshape(-1),
    )
    finite_difference_derivatives = []
    finite_difference_qp_couplings_mev = []
    for step in steps:
        plus = _mean_field(coords + step * cartesian_mode)
        minus = _mean_field(coords - step * cartesian_mode)
        gap_derivative = (
            _gap(plus, occupied_band, virtual_band)
            - _gap(minus, occupied_band, virtual_band)
        ) / (2.0 * step)
        plus_space, plus_block = _frozen_space_and_bse(plus, reference_orbitals)
        minus_space, minus_block = _frozen_space_and_bse(
            minus,
            reference_orbitals,
        )
        bare_derivative = float(
            (
                plus_block.direct
                - plus_block.exchange
                - minus_block.direct
                + minus_block.exchange
            )[transition_index, transition_index].real
            / (2.0 * step)
        )
        screened_derivative = float(
            (plus_block.screened_exchange - minus_block.screened_exchange)[
                transition_index,
                transition_index,
            ].real
            / (2.0 * step)
        )
        finite_difference_derivatives.append(
            [
                gap_derivative,
                bare_derivative,
                screened_derivative,
                gap_derivative + bare_derivative + screened_derivative,
            ]
        )
        plus_qp = diagonal_g0w0(
            plus_space,
            q_indices=[0],
            eta=gw_eta,
            direct_scale=2.0,
            coulomb_component="gdf",
            qp_bands=[occupied_band],
        ).e_qp[0, occupied_band]
        minus_qp = diagonal_g0w0(
            minus_space,
            q_indices=[0],
            eta=gw_eta,
            direct_scale=2.0,
            coulomb_component="gdf",
            qp_bands=[occupied_band],
        ).e_qp[0, occupied_band]
        finite_difference_qp_couplings_mev.append(
            (plus_qp - minus_qp) / (2.0 * step) * zero_point_scale
        )
    finite_difference_couplings_mev = (
        np.asarray(finite_difference_derivatives) * zero_point_scale
    )
    finite_difference_qp_couplings_mev = np.asarray(
        finite_difference_qp_couplings_mev
    )

    fig, axes_grid = plt.subplots(
        2,
        2,
        figsize=(7.6, 6.1),
        constrained_layout=True,
    )
    axes = axes_grid.reshape(-1)
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
    branches = np.arange(len(frequencies_cm1))
    axes[0].axhline(0.0, color="0.45", lw=0.8)
    axes[0].vlines(branches, 0.0, frequencies_cm1, color="0.72", lw=1.2)
    axes[0].scatter(branches, frequencies_cm1, color="0.45", s=20, zorder=3)
    axes[0].scatter(
        [branch],
        [frequencies_cm1[branch]],
        color=colors[1],
        s=34,
        zorder=4,
        label="Selected mode",
    )
    axes[0].set(
        xlabel="Branch index",
        ylabel=r"Signed frequency (cm$^{-1}$)",
        xticks=branches,
    )
    axes[0].legend(frameon=False, fontsize=8)

    component_labels = ("One body", "Bare kernel", "Screening", "Total")
    positions = np.arange(4)
    width = 0.34
    axes[1].bar(
        positions - width / 2,
        analytic_couplings_mev,
        width,
        color=colors[0],
        label="Analytic",
    )
    axes[1].bar(
        positions + width / 2,
        finite_difference_couplings_mev[-1],
        width,
        color=colors[1],
        label=r"FD, $\Delta Q=0.01$",
    )
    axes[1].axhline(0.0, color="0.45", lw=0.8)
    axes[1].set(
        ylabel="Zero-point coupling (meV)",
        xticks=positions,
        xticklabels=component_labels,
    )
    axes[1].tick_params(axis="x", labelrotation=20)
    axes[1].legend(frameon=False, fontsize=8)

    errors_uev = np.abs(
        finite_difference_couplings_mev - analytic_couplings_mev[None, :]
    ) * 1.0e3
    for component, color in enumerate(colors):
        axes[2].loglog(
            steps,
            errors_uev[:, component],
            "o-",
            color=color,
            ms=4,
            lw=1.2,
            label=component_labels[component],
        )
    quadratic_guide = 1.35 * errors_uev[-1, -1] * (steps / steps[-1]) ** 2
    axes[2].loglog(
        steps,
        quadratic_guide,
        "--",
        color="0.35",
        lw=1.2,
        label=r"$O(\Delta Q^2)$",
    )
    axes[2].set(
        xlabel=r"Mass-weighted step ($\sqrt{m_e}\,a_0$)",
        ylabel=r"Coupling error ($\mu$eV)",
    )
    axes[2].set_xticks(steps)
    axes[2].set_xticklabels(["0.08", "0.04", "0.02", "0.01"])
    axes[2].xaxis.set_minor_formatter(NullFormatter())
    axes[2].legend(frameon=False, fontsize=8, ncol=2)

    qp_error_uev = np.abs(
        finite_difference_qp_couplings_mev - analytic_qp_coupling_mev
    ) * 1.0e3
    qp_guide = 1.35 * qp_error_uev[-1] * (steps / steps[-1]) ** 2
    axes[3].loglog(
        steps,
        qp_error_uev,
        "o-",
        color=colors[3],
        ms=4,
        lw=1.2,
        label=r"On-shell $G_0W_0$",
    )
    axes[3].loglog(
        steps,
        qp_guide,
        "--",
        color="0.35",
        lw=1.2,
        label=r"$O(\Delta Q^2)$",
    )
    axes[3].set(
        xlabel=r"Mass-weighted step ($\sqrt{m_e}\,a_0$)",
        ylabel=r"QP coupling error ($\mu$eV)",
    )
    axes[3].set_xticks(steps)
    axes[3].set_xticklabels(["0.08", "0.04", "0.02", "0.01"])
    axes[3].xaxis.set_minor_formatter(NullFormatter())
    axes[3].legend(frameon=False, fontsize=8)
    for label, axis in zip("abcd", axes):
        axis.text(
            0.02,
            0.97,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
        axis.grid(alpha=0.18, lw=0.6)
        axis.spines[["top", "right"]].set_visible(False)

    fig.savefig(output, dpi=320)
    pdf = output.with_suffix(".pdf")
    fig.savefig(pdf)
    plt.close(fig)
    summary = {
        "figure": str(output),
        "pdf": str(pdf),
        "krhf_converged": bool(mean_field.converged),
        "electronic_reference": "native_gdf_krhf",
        "phonon_reference": "reciprocal_krhf",
        "selected_branch": branch,
        "frequency_cm-1": float(frequencies_cm1[branch]),
        "transition_index": transition_index,
        "occupied_band": occupied_band,
        "virtual_band": virtual_band,
        "components": list(component_labels),
        "analytic_zero_point_coupling_meV": analytic_couplings_mev.tolist(),
        "smallest_step_coupling_meV": finite_difference_couplings_mev[-1].tolist(),
        "smallest_step_error_meV": (
            finite_difference_couplings_mev[-1] - analytic_couplings_mev
        ).tolist(),
        "analytic_g0w0_zero_point_coupling_meV": analytic_qp_coupling_mev,
        "smallest_step_g0w0_coupling_meV": float(
            finite_difference_qp_couplings_mev[-1]
        ),
        "smallest_step_g0w0_error_meV": float(
            finite_difference_qp_couplings_mev[-1] - analytic_qp_coupling_mev
        ),
        "cphf_residual_norm": float(coupling.response.residual_norm),
        "bse_bare_kernel_derivative": coupling.info[
            "bse_bare_kernel_derivative"
        ],
        "bse_screening_derivative": coupling.info["bse_screening_derivative"],
    }
    json_path = output.with_suffix(".json")
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    summary["json"] = str(json_path)
    print(json.dumps(summary, indent=2))
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="/private/tmp/pbc_gamma_gdf_electron_phonon.png",
    )
    args = parser.parse_args()
    run(args.output)


if __name__ == "__main__":
    main()
