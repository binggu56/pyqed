"""Three-mode LVC absorption spectrum of the CHD 3p_x/3p_y manifold.

The initial nuclear state is the ground vibrational Gaussian at the neutral
ground-state minimum.  Vertical Condon excitation prepares either electronic
component, and the resulting two-state wavepackets are propagated on the
coupled modes 5, 8, and 26.  Contracting their correlation matrix with the
ab-initio transition dipoles gives the isotropic linear absorption spectrum.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import physical_constants


MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
PROJECTION_DATA = Path("chd_c2_3px_3py_mode_projection.npz")
COUPLING_DATA = Path("chd_c2_3px_3py_offdiagonal_vibronic.npz")
NEVPT2_DATA = Path("chd_c2_casci610_nevpt2.json")
SA_DATA = Path("chd_c2_sa_casscf610_aug_rydberg.json")
OUTPUT = Path("chd_c2_lvc_absorption_3mode")

MODE_IDS = np.array([5, 8, 26])
CM1_TO_HARTREE = physical_constants["inverse meter-hartree relationship"][0] * 100.0
AMU_TO_ELECTRON_MASS = physical_constants["atomic mass constant"][0] / physical_constants["electron mass"][0]
AU_TIME_FS = physical_constants["atomic unit of time"][0] * 1.0e15
HARTREE_TO_EV = physical_constants["Hartree energy in eV"][0]
HC_EV_NM = 1239.8419843320026


def potential_propagator(v00, v11, v01, half_dt):
    average = 0.5 * (v00 + v11)
    z = 0.5 * (v00 - v11)
    radius = np.sqrt(z * z + v01 * v01)
    phase = np.exp(-1j * average * half_dt)
    cosine = np.cos(radius * half_dt)
    sine_over_radius = np.where(radius > 1.0e-14, np.sin(radius * half_dt) / radius, half_dt)
    return (
        phase * (cosine - 1j * sine_over_radius * z),
        phase * (cosine + 1j * sine_over_radius * z),
        phase * (-1j * sine_over_radius * v01),
    )


def apply_potential(psi, u00, u11, u01):
    old0 = psi[:, 0].copy()
    old1 = psi[:, 1].copy()
    psi[:, 0] = u00 * old0 + u01 * old1
    psi[:, 1] = u01 * old0 + u11 * old1


def reciprocal_energy_axis(values):
    values = np.asarray(values, dtype=float)
    return np.divide(HC_EV_NM, values, out=np.full_like(values, np.inf), where=values != 0.0)


def main():
    indices = MODE_IDS - 1
    with np.load(MODE_DATA) as data:
        frequencies_cm1 = data["frequencies_cm1"][indices]
    frequencies = frequencies_cm1 * CM1_TO_HARTREE
    with np.load(PROJECTION_DATA) as data:
        forces = np.asarray([data[f"force_projection_{state}_au"][indices]
                             for state in ("3px", "3py")])
    diagonal = -forces
    with np.load(COUPLING_DATA) as data:
        off_diagonal = data["lambda_xy"][indices]

    nevpt2 = json.loads(NEVPT2_DATA.read_text())
    excitation_ev = np.asarray(nevpt2["sc_nevpt2_vertical_excitation_energies_ev"])[[2, 3]]
    relative_energies = (excitation_ev - excitation_ev[0]) / HARTREE_TO_EV
    sa = json.loads(SA_DATA.read_text())
    # Entries 1 and 2 correspond to paper-frame 3px and 3py (roots 2 and 3).
    transition_dipoles = np.asarray(sa["ground_to_excited_transition_dipoles_au"])[1:3]
    dipole_metric = transition_dipoles @ transition_dipoles.T / 3.0

    ngrid = 36
    extent = 7.0
    x = np.linspace(-extent, extent, ngrid, endpoint=False)
    dx = x[1] - x[0]
    grids = np.meshgrid(*([x] * len(MODE_IDS)), indexing="ij")
    q_scales = np.sqrt(frequencies * AMU_TO_ELECTRON_MASS)
    mass_weighted = [grid / scale for grid, scale in zip(grids, q_scales)]
    harmonic = sum(0.5 * frequency * grid**2 for frequency, grid in zip(frequencies, grids))
    v00 = harmonic + relative_energies[0] + sum(c * q for c, q in zip(diagonal[0], mass_weighted))
    v11 = harmonic + relative_energies[1] + sum(c * q for c, q in zip(diagonal[1], mass_weighted))
    v01 = sum(c * q for c, q in zip(off_diagonal, mass_weighted))

    dt_fs = 0.10
    tmax_fs = 250.0
    dt_au = dt_fs / AU_TIME_FS
    nsteps = int(round(tmax_fs / dt_fs))
    u00, u11, u01 = potential_propagator(v00, v11, v01, 0.5 * dt_au)
    momenta = 2.0 * np.pi * np.fft.fftfreq(ngrid, d=dx)
    momentum_grids = np.meshgrid(*([momenta] * len(MODE_IDS)), indexing="ij")
    kinetic = sum(0.5 * frequency * momentum**2
                  for frequency, momentum in zip(frequencies, momentum_grids))
    kinetic_phase = np.exp(-1j * kinetic * dt_au)

    ground = np.exp(-0.5 * sum(grid**2 for grid in grids)) / np.pi ** (len(MODE_IDS) / 4.0)
    volume = dx ** len(MODE_IDS)
    ground /= np.sqrt(np.sum(np.abs(ground) ** 2) * volume)
    # First axis labels the initially excited diabatic component; second is
    # the propagated electronic component.
    psi = np.zeros((2, 2) + ground.shape, dtype=complex)
    psi[0, 0] = ground
    psi[1, 1] = ground
    correlations = np.empty((nsteps + 1, 2, 2), dtype=complex)
    correlations[0] = np.eye(2)
    for step in range(1, nsteps + 1):
        apply_potential(psi, u00, u11, u01)
        for initial in range(2):
            for state in range(2):
                psi[initial, state] = np.fft.ifftn(
                    np.fft.fftn(psi[initial, state]) * kinetic_phase
                )
        apply_potential(psi, u00, u11, u01)
        # C_ab = <g,a | exp(-i H t) | g,b>.
        correlations[step] = np.einsum("...,ba...->ab", ground.conj(), psi, optimize=True) * volume

    times_fs = np.arange(nsteps + 1) * dt_fs
    times_au = times_fs / AU_TIME_FS
    zpe = 0.5 * np.sum(frequencies)
    isotropic_correlation = np.einsum("tab,ab->t", correlations, dipole_metric)
    isotropic_correlation *= np.exp(1j * zpe * times_au)
    width_fwhm_ev = 0.12
    sigma_eh = width_fwhm_ev / (2.0 * np.sqrt(2.0 * np.log(2.0))) / HARTREE_TO_EV
    damping = np.exp(-0.5 * (sigma_eh * times_au) ** 2)

    energy_ev = np.linspace(5.55, 7.05, 1200)
    detuning = (energy_ev - excitation_ev[0]) / HARTREE_TO_EV
    phase = np.exp(1j * detuning[:, None] * times_au[None, :])
    spectrum = np.real(np.trapezoid(phase * (isotropic_correlation * damping)[None, :], times_au, axis=1))
    spectrum = np.maximum(spectrum, 0.0)
    spectrum /= spectrum.max()
    wavelength_nm = HC_EV_NM / energy_ev

    np.savez(
        OUTPUT.with_suffix(".npz"), energy_ev=energy_ev, wavelength_nm=wavelength_nm,
        intensity=spectrum, times_fs=times_fs, correlation=isotropic_correlation,
        damping=damping, mode_ids=MODE_IDS, frequencies_cm1=frequencies_cm1,
        transition_dipoles_au=transition_dipoles, vertical_energies_ev=excitation_ev,
    )
    metadata = {
        "model": "two-state/three-mode LVC absorption; paper-frame 3px and 3py",
        "modes": MODE_IDS.tolist(),
        "frequencies_cm-1": frequencies_cm1.tolist(),
        "vertical_energies_eV_SC_NEVPT2": excitation_ev.tolist(),
        "broadening_FWHM_eV": width_fwhm_ev,
        "approximations": ["Condon transition dipoles", "harmonic ground state",
                           "modes 5, 8, and 26 only", "linear vibronic Hamiltonian"],
    }
    OUTPUT.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(5.3, 3.4), constrained_layout=True)
    ax.plot(wavelength_nm, spectrum, color="#0072B2", lw=1.5,
            label="3-mode LVC, isotropic")
    for index, (energy, label, color) in enumerate(zip(excitation_ev, (r"$3p_x$", r"$3p_y$"),
                                                        ("#D55E00", "#009E73"))):
        ax.axvline(HC_EV_NM / energy, color=color, ls="--", lw=1.0,
                   label=f"{label} vertical energy")
    ax.set_xlim(220, 175)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Normalized absorption")
    ax.tick_params(direction="in", top=True, right=True)
    ax.legend(frameon=False, fontsize=8)
    top = ax.secondary_xaxis("top", functions=(reciprocal_energy_axis,
                                               reciprocal_energy_axis))
    top.set_xlabel("Photon energy (eV)")
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=350)


if __name__ == "__main__":
    main()
