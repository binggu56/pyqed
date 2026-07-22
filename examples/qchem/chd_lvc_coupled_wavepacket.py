"""Coupled two-state/three-mode CHD LVC wavepacket propagation."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants
from scipy.ndimage import map_coordinates

from pyqed import au2angstrom

from chd_sa_casscf48_aug_rydberg import read_xyz


MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
PROJECTION_DATA = Path("chd_c2_3px_3py_mode_projection.npz")
COUPLING_DATA = Path("chd_c2_3px_3py_offdiagonal_vibronic.npz")
NEVPT2_DATA = Path("chd_c2_casci610_nevpt2.json")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
OUTPUT_PREFIX = Path("chd_c2_lvc_coupled_wavepacket_3mode")
MODE_IDS = np.array([5, 8, 26])
POPULATIONS = np.array([1.0 / 1.8, 0.8 / 1.8])
INITIAL_AMPLITUDES = np.sqrt(POPULATIONS).astype(complex)
CM1_TO_HARTREE = physical_constants["inverse meter-hartree relationship"][0] * 100.0
AMU_TO_ELECTRON_MASS = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]
AU_TIME_FS = physical_constants["atomic unit of time"][0] * 1.0e15
ATOMIC_NUMBERS = {"H": 1.0, "C": 6.0}


def potential_propagator(v00, v11, v01, half_dt_au):
    average = 0.5 * (v00 + v11)
    z = 0.5 * (v00 - v11)
    radius = np.sqrt(z * z + v01 * v01)
    phase = np.exp(-1j * average * half_dt_au)
    cosine = np.cos(radius * half_dt_au)
    sine_over_radius = np.where(
        radius > 1.0e-14,
        np.sin(radius * half_dt_au) / radius,
        half_dt_au,
    )
    u00 = phase * (cosine - 1j * sine_over_radius * z)
    u11 = phase * (cosine + 1j * sine_over_radius * z)
    u01 = phase * (-1j * sine_over_radius * v01)
    return u00, u11, u01


def apply_potential(psi, u00, u11, u01):
    old0 = psi[0].copy()
    old1 = psi[1].copy()
    psi[0] = u00 * old0 + u01 * old1
    psi[1] = u01 * old0 + u11 * old1


def observables(psi, grids, volume):
    densities = np.abs(psi) ** 2
    nuclear_axes = tuple(range(1, psi.ndim))
    populations = np.sum(densities, axis=nuclear_axes) * volume
    coherence = np.sum(np.conj(psi[0]) * psi[1]) * volume
    means = np.empty((2, len(grids)))
    for state in range(2):
        for mode in range(len(grids)):
            means[state, mode] = (
                np.sum(densities[state] * grids[mode]) * volume / populations[state]
                if populations[state] > 1.0e-12 else np.nan
            )
    return populations, coherence, means


def one_dimensional_marginals(psi, dx):
    densities = np.abs(psi) ** 2
    marginals = np.empty((2, psi.ndim - 1, psi.shape[1]))
    for state in range(2):
        for mode in range(psi.ndim - 1):
            summed_axes = tuple(axis for axis in range(psi.ndim - 1) if axis != mode)
            marginals[state, mode] = np.sum(densities[state], axis=summed_axes) * dx ** len(summed_axes)
    return marginals


def pair_marginal(psi, retained_modes, dx):
    densities = np.abs(psi) ** 2
    summed_axes = tuple(
        axis for axis in range(psi.ndim - 1) if axis not in retained_modes
    )
    return np.sum(densities, axis=tuple(axis + 1 for axis in summed_axes)) * dx ** len(summed_axes)


def paper_frame(metadata):
    x_axis = np.asarray(metadata["pi_plane_normal"], dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return np.asarray([
        x_axis,
        y_axis,
        z_axis,
        (x_axis + y_axis) / np.sqrt(2.0),
    ])


def characteristic_function(field, queries, dx, x_origin, padded_size=96):
    """Interpolate the Fourier transform integral at arbitrary 3-D queries."""
    offset = (padded_size - field.shape[0]) // 2
    padded = np.zeros((padded_size,) * 3, dtype=field.dtype)
    padded[offset:offset + field.shape[0],
           offset:offset + field.shape[1],
           offset:offset + field.shape[2]] = field
    transform = np.fft.fftshift(np.fft.fftn(padded)) * dx**3
    frequencies = np.fft.fftshift(2.0 * np.pi * np.fft.fftfreq(padded_size, d=dx))
    padded_origin = x_origin - offset * dx
    transform *= np.exp(
        -1j * padded_origin
        * (frequencies[:, None, None] + frequencies[None, :, None] + frequencies[None, None, :])
    )
    spacing = frequencies[1] - frequencies[0]
    coordinates = np.moveaxis((queries - frequencies[0]) / spacing, -1, 0)
    real = map_coordinates(transform.real, coordinates, order=3, mode="constant", cval=0.0)
    imag = map_coordinates(transform.imag, coordinates, order=3, mode="constant", cval=0.0)
    return real + 1j * imag


def prepare_nuclear_scattering(modes, q_scales):
    metadata = json.loads(SA_DATA.read_text(encoding="utf-8"))
    directions = paper_frame(metadata)
    direction_labels = np.asarray([
        "x_out_of_plane", "y_in_plane", "z_C2", "xy_bisector"
    ])
    s = np.linspace(0.40, 8.0, 305)
    q_vectors = s[None, :, None] * au2angstrom * directions[:, None, :]
    atoms = read_xyz(GEOMETRY)
    charges = np.asarray([ATOMIC_NUMBERS[symbol] for symbol, _ in atoms])
    reference_bohr = np.asarray([coordinates for _, coordinates in atoms]) / au2angstrom
    dimensionless_modes = modes / q_scales[:, None, None]
    atomic_queries = np.einsum(
        "dse,mae->dsam", q_vectors, dimensionless_modes, optimize=True
    )
    pair_queries = atomic_queries[:, :, None, :, :] - atomic_queries[:, :, :, None, :]
    base_phase = np.exp(
        -1j * np.einsum("dse,ae->dsa", q_vectors, reference_bohr, optimize=True)
    )
    return {
        "s": s,
        "directions": directions,
        "direction_labels": direction_labels,
        "charges": charges,
        "atomic_queries": atomic_queries,
        "pair_queries": pair_queries,
        "base_phase": base_phase,
    }


def nuclear_scattering_moments(psi, dx, x_origin, scattering):
    rho_x = np.abs(psi[0]) ** 2
    rho_y = np.abs(psi[1]) ** 2
    rho_xy = np.conj(psi[0]) * psi[1]
    atomic_queries = scattering["atomic_queries"]
    pair_queries = scattering["pair_queries"]
    shape_atomic = atomic_queries.shape[:-1]
    shape_pair = pair_queries.shape[:-1]
    chi_x = characteristic_function(
        rho_x, atomic_queries.reshape(-1, 3), dx, x_origin
    ).reshape(shape_atomic)
    chi_y = characteristic_function(
        rho_y, atomic_queries.reshape(-1, 3), dx, x_origin
    ).reshape(shape_atomic)
    chi_xy_plus = characteristic_function(
        rho_xy, atomic_queries.reshape(-1, 3), dx, x_origin
    ).reshape(shape_atomic)
    chi_xy_minus = characteristic_function(
        rho_xy, -atomic_queries.reshape(-1, 3), dx, x_origin
    ).reshape(shape_atomic)
    chi_total_pair = characteristic_function(
        rho_x + rho_y, pair_queries.reshape(-1, 3), dx, x_origin
    ).reshape(shape_pair)
    charges = scattering["charges"]
    base_phase = scattering["base_phase"]
    cstar = np.empty((2, 2) + base_phase.shape[:2], dtype=complex)
    cstar[0, 0] = np.einsum(
        "a,dsa,dsa->ds", charges, base_phase.conj(), chi_x.conj(), optimize=True
    )
    cstar[1, 1] = np.einsum(
        "a,dsa,dsa->ds", charges, base_phase.conj(), chi_y.conj(), optimize=True
    )
    cstar[0, 1] = np.einsum(
        "a,dsa,dsa->ds", charges, base_phase.conj(), chi_xy_minus, optimize=True
    )
    cstar[1, 0] = np.einsum(
        "a,dsa,dsa->ds", charges, base_phase.conj(), chi_xy_plus.conj(), optimize=True
    )
    cc = np.einsum(
        "a,b,dsa,dsb,dsab->ds", charges, charges, base_phase.conj(),
        base_phase, chi_total_pair, optimize=True,
    )
    c_expectation = np.conj(cstar[0, 0] + cstar[1, 1])
    return c_expectation, np.real(cc), cstar


def main():
    indices = MODE_IDS - 1
    with np.load(MODE_DATA) as data:
        frequencies_cm1 = data["frequencies_cm1"][indices]
        modes = data["normal_modes"][indices]
    frequencies = frequencies_cm1 * CM1_TO_HARTREE
    with np.load(PROJECTION_DATA) as data:
        forces = np.asarray(
            [data[f"force_projection_{state}_au"][indices] for state in ("3px", "3py")]
        )
    diagonal_couplings = -forces
    with np.load(COUPLING_DATA) as data:
        off_diagonal_couplings = data["lambda_xy"][indices]
    nevpt2 = json.loads(NEVPT2_DATA.read_text(encoding="utf-8"))
    excitation = np.asarray(nevpt2["sc_nevpt2_vertical_excitation_energies_ev"])
    gap_hartree = (excitation[3] - excitation[2]) / 27.211386245988
    energies = np.array([0.0, gap_hartree])

    ngrid = 48
    extent = 7.0
    x = np.linspace(-extent, extent, ngrid, endpoint=False)
    dx = x[1] - x[0]
    grids = np.meshgrid(*([x] * len(MODE_IDS)), indexing="ij")
    q_scales = np.sqrt(frequencies * AMU_TO_ELECTRON_MASS)
    scattering = prepare_nuclear_scattering(modes, q_scales)
    mass_weighted_grids = [grid / scale for grid, scale in zip(grids, q_scales)]
    harmonic = sum(0.5 * frequency * grid**2 for frequency, grid in zip(frequencies, grids))
    v00 = harmonic + energies[0] + sum(
        coupling * grid for coupling, grid in zip(diagonal_couplings[0], mass_weighted_grids)
    )
    v11 = harmonic + energies[1] + sum(
        coupling * grid for coupling, grid in zip(diagonal_couplings[1], mass_weighted_grids)
    )
    v01 = sum(
        coupling * grid for coupling, grid in zip(off_diagonal_couplings, mass_weighted_grids)
    )

    dt_fs = 0.10
    dt_au = dt_fs / AU_TIME_FS
    u00, u11, u01 = potential_propagator(v00, v11, v01, 0.5 * dt_au)
    momenta = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=dx)
    momentum_grids = np.meshgrid(*([momenta] * len(MODE_IDS)), indexing="ij")
    kinetic = sum(
        0.5 * frequency * momentum**2
        for frequency, momentum in zip(frequencies, momentum_grids)
    )
    kinetic_phase = np.exp(-1j * kinetic * dt_au)

    nuclear_ground = np.exp(-0.5 * sum(grid**2 for grid in grids)) / np.pi ** (len(MODE_IDS) / 4.0)
    psi = INITIAL_AMPLITUDES.reshape((2,) + (1,) * len(MODE_IDS)) * nuclear_ground[None, ...]
    volume = dx ** len(MODE_IDS)
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * volume)

    output_times = np.arange(0.0, 350.0 + 0.5, 0.5)
    output_every = int(round(0.5 / dt_fs))
    nsteps = int(round(output_times[-1] / dt_fs))
    populations = np.empty((output_times.size, 2))
    coherence = np.empty(output_times.size, dtype=complex)
    mean_x = np.empty((output_times.size, 2, len(MODE_IDS)))
    marginal_1d = np.empty((output_times.size, 2, len(MODE_IDS), ngrid))
    snapshot_times = np.array([0.0, 15.0, 30.0, 45.0, 60.0, 90.0, 150.0, 350.0])
    snapshot_indices = {int(round(value / 0.5)): index for index, value in enumerate(snapshot_times)}
    pair_58 = np.empty((snapshot_times.size, 2, ngrid, ngrid))
    pair_826 = np.empty_like(pair_58)
    scattering_shape = (output_times.size, len(scattering["directions"]), len(scattering["s"]))
    nuclear_amplitude = np.empty(scattering_shape, dtype=complex)
    nuclear_cc = np.empty(scattering_shape)
    nuclear_cstar_rho = np.empty((output_times.size, 2, 2) + scattering_shape[1:], dtype=complex)
    populations[0], coherence[0], mean_x[0] = observables(psi, grids, volume)
    marginal_1d[0] = one_dimensional_marginals(psi, dx)
    pair_58[0] = pair_marginal(psi, (0, 1), dx)
    pair_826[0] = pair_marginal(psi, (1, 2), dx)
    nuclear_amplitude[0], nuclear_cc[0], nuclear_cstar_rho[0] = nuclear_scattering_moments(
        psi, dx, x[0], scattering
    )
    output_index = 1

    for step in range(1, nsteps + 1):
        apply_potential(psi, u00, u11, u01)
        for state in range(2):
            psi[state] = np.fft.ifftn(np.fft.fftn(psi[state]) * kinetic_phase)
        apply_potential(psi, u00, u11, u01)
        if step % output_every == 0:
            populations[output_index], coherence[output_index], mean_x[output_index] = observables(
                psi, grids, volume
            )
            marginal_1d[output_index] = one_dimensional_marginals(psi, dx)
            nuclear_amplitude[output_index], nuclear_cc[output_index], nuclear_cstar_rho[output_index] = nuclear_scattering_moments(
                psi, dx, x[0], scattering
            )
            if output_index in snapshot_indices:
                snapshot_index = snapshot_indices[output_index]
                pair_58[snapshot_index] = pair_marginal(psi, (0, 1), dx)
                pair_826[snapshot_index] = pair_marginal(psi, (1, 2), dx)
            output_index += 1

    mean_q = mean_x / q_scales[None, None, :]
    norm = np.sum(np.abs(psi) ** 2) * volume
    np.savez_compressed(
        f"{OUTPUT_PREFIX}.npz",
        times_fs=output_times,
        populations=populations,
        coherence=coherence,
        conditional_mean_x=mean_x,
        conditional_mean_q_bohr_sqrtamu=mean_q,
        marginal_density_1d=marginal_1d,
        snapshot_times_fs=snapshot_times,
        pair_density_modes_5_8=pair_58,
        pair_density_modes_8_26=pair_826,
        scattering_s_angstrom_inverse=scattering["s"],
        scattering_direction_labels=scattering["direction_labels"],
        scattering_directions_xyz=scattering["directions"],
        nuclear_amplitude_expectation=nuclear_amplitude,
        nuclear_charge_squared_expectation=nuclear_cc,
        nuclear_cstar_electronic_rho=nuclear_cstar_rho,
        frequencies_cm1=frequencies_cm1,
        diagonal_couplings=diagonal_couplings,
        off_diagonal_couplings=off_diagonal_couplings,
        final_wavefunction=psi,
        dimensionless_grid=x,
    )
    summary = {
        "method": "two-state/three-mode split-operator LVC wavepacket",
        "states": ["paper-frame 3px", "paper-frame 3py"],
        "mode_ids": MODE_IDS.tolist(),
        "frequencies_cm-1": frequencies_cm1.tolist(),
        "nevpt2_gap_ev": float(gap_hartree * 27.211386245988),
        "diagonal_couplings_Eh_per_bohr_sqrtamu": diagonal_couplings.tolist(),
        "off_diagonal_couplings_Eh_per_bohr_sqrtamu": off_diagonal_couplings.tolist(),
        "grid": {"points_per_mode": ngrid, "extent_dimensionless": extent},
        "time_step_fs": dt_fs,
        "final_norm": float(norm),
        "population_ranges": {
            "3px": [float(populations[:, 0].min()), float(populations[:, 0].max())],
            "3py": [float(populations[:, 1].min()), float(populations[:, 1].max())],
        },
        "coherence_magnitude_range": [
            float(np.abs(coherence).min()), float(np.abs(coherence).max())
        ],
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(3, 1, figsize=(7.0, 7.0), sharex=True)
    axes[0].plot(output_times, populations[:, 0], color="#0072B2", lw=1.3, label=r"$3p_x$")
    axes[0].plot(output_times, populations[:, 1], color="#D55E00", lw=1.3, label=r"$3p_y$")
    axes[0].set_ylabel("Population")
    axes[0].legend(frameon=False, ncol=2)
    axes[0].text(-0.10, 1.02, "a", transform=axes[0].transAxes, fontweight="bold")
    axes[1].plot(output_times, np.abs(coherence), color="#009E73", lw=1.3)
    axes[1].set_ylabel(r"$|\rho_{xy}|$")
    axes[1].text(-0.10, 1.02, "b", transform=axes[1].transAxes, fontweight="bold")
    colors = ["#0072B2", "#D55E00", "#CC79A7"]
    for mode, color in enumerate(colors):
        total_mean = np.sum(populations * mean_x[:, :, mode], axis=1)
        axes[2].plot(output_times, total_mean, color=color, lw=1.1,
                     label=rf"mode {MODE_IDS[mode]}")
    axes[2].axhline(0.0, color="0.65", lw=0.7)
    axes[2].set(xlabel="Time delay (fs)", ylabel=r"$\langle q_k\rangle$")
    axes[2].legend(frameon=False, ncol=3)
    axes[2].text(-0.10, 1.02, "c", transform=axes[2].transAxes, fontweight="bold")
    for axis in axes:
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
        axis.set_xlim(output_times[0], output_times[-1])
    figure.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.08, hspace=0.13)
    figure.savefig(f"{OUTPUT_PREFIX}.pdf")
    figure.savefig(f"{OUTPUT_PREFIX}.png", dpi=350)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
