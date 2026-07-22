"""Level-1 CHD LVC nuclear wavepacket and isotropic IAM UED signal.

This deliberately diagonal two-state model propagates modes 5 and 8 on the
paper-frame 3px- and 3py-dominant surfaces.  A coherent electronic
superposition is initialized, but the level-1 IAM operator is diagonal and is
therefore insensitive to its relative phase.  The model contains no interstate
coupling or electronic transition-density scattering.  The nuclear IAM signal
is averaged over the two-mode coherent-state Gaussian distribution.
"""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.constants import atomic_mass, hbar, physical_constants, speed_of_light

from pyqed.qchem.vibronic import LVC
from pyqed.ued.ued import electron_atomic_form_factor


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
PROJECTION_DATA = Path("chd_c2_3px_3py_mode_projection.npz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
OUTPUT_PREFIX = Path("chd_c2_lvc_level1_iam_ued")
MODE_IDS = np.array([5, 8])
STATE_LABELS = ("3px", "3py")  # Paper molecular-frame convention.
STATE_WEIGHTS = np.array([1.0 / 1.8, 0.8 / 1.8])
INITIAL_AMPLITUDES = np.sqrt(STATE_WEIGHTS).astype(complex)
HARTREE_J = physical_constants["Hartree energy"][0]
BOHR_M = physical_constants["Bohr radius"][0]
BOHR_ANGSTROM = BOHR_M * 1.0e10
CM1_TO_HARTREE = physical_constants["inverse meter-hartree relationship"][0] * 100.0


def read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    natom = int(lines[0])
    fields = [line.split() for line in lines[2 : 2 + natom]]
    symbols = [row[0] for row in fields]
    coordinates = np.asarray([[float(value) for value in row[1:4]] for row in fields])
    return symbols, coordinates


def pair_indices(natom):
    return np.triu_indices(natom, 1)


def atomic_background(symbols, s):
    factors = np.vstack(
        [electron_atomic_form_factor(symbol, s, q_unit="angstrom^-1") for symbol in symbols]
    )
    return factors, np.sum(factors**2, axis=0)


def molecular_intensity_ensemble(coordinates, weights, factors, s, pairs):
    """Average the IAM molecular interference over nuclear configurations."""
    left, right = pairs
    distances = np.linalg.norm(coordinates[:, right] - coordinates[:, left], axis=-1)
    pair_factors = 2.0 * factors[left] * factors[right]
    sinc_values = np.sinc(distances[:, :, None] * s[None, None, :] / np.pi)
    per_configuration = np.einsum("ps,cps->cs", pair_factors, sinc_values, optimize=True)
    return np.einsum("c,cs->s", weights, per_configuration, optimize=True)


def force_to_equilibrium_shift(force_au, frequency_cm1):
    force_si = force_au * HARTREE_J / (BOHR_M * np.sqrt(atomic_mass))
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequency_cm1
    shift_si = force_si / omega_si**2
    return shift_si / (BOHR_M * np.sqrt(atomic_mass))


def zero_point_sigma(frequency_cm1):
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequency_cm1
    sigma_si = np.sqrt(hbar / (2.0 * omega_si))
    return sigma_si / (BOHR_M * np.sqrt(atomic_mass))


def gaussian_quadrature(npoints, sigmas):
    nodes, weights = hermgauss(npoints)
    q0, q1 = np.meshgrid(nodes, nodes, indexing="ij")
    w0, w1 = np.meshgrid(weights, weights, indexing="ij")
    offsets = np.stack(
        [np.sqrt(2.0) * sigmas[0] * q0.ravel(), np.sqrt(2.0) * sigmas[1] * q1.ravel()],
        axis=1,
    )
    normalized_weights = (w0 * w1).ravel() / np.pi
    return offsets, normalized_weights


def nuclear_ensemble(reference, modes, mean_q, offsets):
    q = mean_q[None, :] + offsets
    displacement = np.einsum("cm,mae->cae", q, modes, optimize=True)
    return reference[None, :, :] + displacement * BOHR_ANGSTROM


def main():
    symbols, reference = read_xyz(GEOMETRY)
    with np.load(MODE_DATA) as mode_data:
        indices = MODE_IDS - 1
        frequencies_cm1 = mode_data["frequencies_cm1"][indices]
        modes = mode_data["normal_modes"][indices]
    with np.load(PROJECTION_DATA) as projected:
        forces = np.asarray(
            [projected[f"force_projection_{state}_au"][indices] for state in STATE_LABELS]
        )
    sa_data = json.loads(SA_DATA.read_text(encoding="utf-8"))
    reference_energies = np.asarray(sa_data["vertical_excitation_energies_ev"])[[2, 3]]
    reference_energies /= 27.211386245988

    # V_aa,k = dE_a/dQ_k = -F_a,k. Off-diagonal terms remain zero at level 1.
    couplings = np.zeros((2, 2, 2))
    couplings[0, 0] = -forces[0]
    couplings[1, 1] = -forces[1]
    model = LVC(
        reference_energies=reference_energies,
        mode_frequencies=frequencies_cm1 * CM1_TO_HARTREE,
        couplings=couplings,
        normal_modes=modes,
        mode_ids=MODE_IDS,
        reference_geometry=reference,
    )

    equilibrium_shifts = np.asarray(
        [
            [force_to_equilibrium_shift(forces[state, mode], frequencies_cm1[mode])
             for mode in range(2)]
            for state in range(2)
        ]
    )
    sigmas = np.asarray([zero_point_sigma(frequency) for frequency in frequencies_cm1])
    offsets, quadrature_weights = gaussian_quadrature(5, sigmas)

    times_fs = np.linspace(0.0, 350.0, 351)
    s = np.linspace(0.25, 10.0, 391)
    angular_frequencies_fs = 2.0 * np.pi * speed_of_light * 100.0 * frequencies_cm1 * 1.0e-15
    factors, i_atomic = atomic_background(symbols, s)
    pairs = pair_indices(len(symbols))

    ground_coordinates = nuclear_ensemble(reference, modes, np.zeros(2), offsets)
    ground_molecular = molecular_intensity_ensemble(
        ground_coordinates, quadrature_weights, factors, s, pairs
    )
    ground_total = i_atomic + ground_molecular
    ground_sm = s * ground_molecular / i_atomic

    state_mean_q = np.empty((2, times_fs.size, 2))
    state_molecular = np.empty((2, times_fs.size, s.size))
    for state in range(2):
        state_mean_q[state] = equilibrium_shifts[state][None, :] * (
            1.0 - np.cos(times_fs[:, None] * angular_frequencies_fs[None, :])
        )
        for time_index, mean_q in enumerate(state_mean_q[state]):
            coordinates = nuclear_ensemble(reference, modes, mean_q, offsets)
            state_molecular[state, time_index] = molecular_intensity_ensemble(
                coordinates, quadrature_weights, factors, s, pairs
            )

    mixture_molecular = np.einsum("a,ats->ts", STATE_WEIGHTS, state_molecular)
    mixture_total = i_atomic[None, :] + mixture_molecular
    mixture_sm = s[None, :] * mixture_molecular / i_atomic[None, :]
    delta_sm = mixture_sm - ground_sm[None, :]
    delta_i_percent = 100.0 * (mixture_total - ground_total[None, :]) / ground_total[None, :]

    np.savez_compressed(
        f"{OUTPUT_PREFIX}.npz",
        times_fs=times_fs,
        s_angstrom_inverse=s,
        delta_sM=delta_sm,
        delta_I_percent=delta_i_percent,
        state_mean_q_bohr_sqrtamu=state_mean_q,
        state_molecular_intensity=state_molecular,
        ground_sM=ground_sm,
        frequencies_cm1=frequencies_cm1,
        equilibrium_shifts_bohr_sqrtamu=equilibrium_shifts,
        zero_point_sigmas_bohr_sqrtamu=sigmas,
        state_weights=STATE_WEIGHTS,
        initial_electronic_amplitudes=INITIAL_AMPLITUDES,
    )

    probe_s = np.array([1.0, 2.0, 4.0, 6.0, 8.0])
    probe_indices = np.asarray([np.argmin(np.abs(s - value)) for value in probe_s])
    with Path(f"{OUTPUT_PREFIX}_traces.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["time_fs"] + [f"delta_sM_s{value:.1f}" for value in probe_s]
        fields += [f"delta_I_percent_s{value:.1f}" for value in probe_s]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for index, time in enumerate(times_fs):
            row = {"time_fs": time}
            for value, s_index in zip(probe_s, probe_indices):
                row[f"delta_sM_s{value:.1f}"] = delta_sm[index, s_index]
                row[f"delta_I_percent_s{value:.1f}"] = delta_i_percent[index, s_index]
            writer.writerow(row)

    peak_index = np.unravel_index(np.argmax(np.abs(delta_i_percent)), delta_i_percent.shape)
    summary = {
        "model": "level-1 diagonal two-state/two-mode LVC nuclear IAM UED",
        "state_labels": list(STATE_LABELS),
        "state_label_convention": "paper frame: x out of plane, z C2 axis",
        "state_weights": STATE_WEIGHTS.tolist(),
        "initial_electronic_state": (
            "sqrt(1/1.8)|3px> + sqrt(0.8/1.8)|3py>, zero relative phase"
        ),
        "coherence_visibility": (
            "IAM nuclear operator is electronic-state diagonal, so level-1 UED is "
            "phase blind and equals the incoherent population-weighted signal"
        ),
        "mode_ids": MODE_IDS.tolist(),
        "frequencies_cm-1": frequencies_cm1.tolist(),
        "sa_casscf_reference_energies_hartree_relative_ground": reference_energies.tolist(),
        "diagonal_forces_Eh_per_bohr_sqrtamu": forces.tolist(),
        "off_diagonal_vibronic_couplings": "zero by level-1 definition, not calculated",
        "nuclear_average": "5x5 Gauss-Hermite zero-point wavepacket",
        "time_range_fs": [float(times_fs[0]), float(times_fs[-1])],
        "s_range_angstrom-1": [float(s[0]), float(s[-1])],
        "peak_abs_delta_I_percent": float(abs(delta_i_percent[peak_index])),
        "peak_time_fs": float(times_fs[peak_index[0]]),
        "peak_s_angstrom-1": float(s[peak_index[1]]),
        "limitations": [
            "fixed 3px/3py populations",
            "coherence prepared but invisible to the diagonal IAM operator",
            "no interstate population transfer",
            "IAM nuclear scattering only",
            "harmonic modes with unchanged ground-state curvature",
            "no rotational photoselection or instrument-response convolution",
        ],
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.2))
    color_limit = np.percentile(np.abs(delta_i_percent), 99.5)
    image = axes[0].pcolormesh(
        s, times_fs, delta_i_percent, shading="auto", cmap="RdBu_r",
        vmin=-color_limit, vmax=color_limit, rasterized=True,
    )
    colorbar = figure.colorbar(image, ax=axes[0], pad=0.015)
    colorbar.set_label(r"$\Delta I/I_0$ (%)")
    axes[0].set(xlabel=r"$s$ ($\mathrm{\AA}^{-1}$)", ylabel="Time delay (fs)")
    axes[0].text(-0.10, 1.02, "a", transform=axes[0].transAxes, fontweight="bold")

    probe_times = np.array([25.0, 50.0, 100.0, 200.0, 300.0])
    time_indices = np.asarray([np.argmin(np.abs(times_fs - value)) for value in probe_times])
    colors = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#6F4E7C")
    for value, time_index, color in zip(probe_times, time_indices, colors):
        axes[1].plot(
            s,
            delta_i_percent[time_index],
            color=color,
            lw=1.2,
            label=rf"{value:.0f} fs",
        )
    axes[1].axhline(0.0, color="0.65", lw=0.7)
    axes[1].set(xlabel=r"$s$ ($\mathrm{\AA}^{-1}$)", ylabel=r"$\Delta I/I_0$ (%)")
    axes[1].text(-0.10, 1.02, "b", transform=axes[1].transAxes, fontweight="bold")
    axes[1].legend(frameon=False, loc="upper right", title="Nominal delay")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    axes[0].set_ylim(times_fs[0], times_fs[-1])
    figure.subplots_adjust(left=0.09, right=0.97, top=0.96, bottom=0.12, wspace=0.38)
    figure.savefig(f"{OUTPUT_PREFIX}.pdf")
    figure.savefig(f"{OUTPUT_PREFIX}.png", dpi=350)
    plt.close(figure)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
