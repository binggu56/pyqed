"""Oriented phase-sensitive CHD UED PD from a coherent 3px/3py state.

This extends the level-1 two-mode nuclear model with SA-CASSCF diagonal and
3px/3py transition densities.  Electronic densities are evaluated at the
reference geometry; nuclear amplitudes follow the mode-5/mode-8 wavepackets.
No rotational average or pump photoselection is applied.
"""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.gto import ft_ao
from scipy.constants import atomic_mass, hbar, physical_constants, speed_of_light

from pyqed import au2angstrom


QCHEM_EXAMPLES = Path(__file__).resolve().parents[1] / "qchem"
if str(QCHEM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(QCHEM_EXAMPLES))

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
MODE_DATA = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
PROJECTION_DATA = Path("chd_c2_3px_3py_mode_projection.npz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
NEVPT2_DATA = Path("chd_c2_casci610_nevpt2.json")
DYNAMICS_DATA = Path("chd_c2_lvc_coupled_wavepacket_3mode.npz")
OUTPUT_PREFIX = Path("chd_c2_lvc_level2_coherent_oriented_pd_3mode")
MODE_IDS = np.array([5, 8, 26])
ROOTS = {"ground": 0, "3px": 2, "3py": 3}
POPULATIONS = np.array([1.0 / 1.8, 0.8 / 1.8])
AMPLITUDES = np.sqrt(POPULATIONS).astype(complex)
HARTREE_J = physical_constants["Hartree energy"][0]
BOHR_M = physical_constants["Bohr radius"][0]
BOHR_ANGSTROM = BOHR_M * 1.0e10
EV_J = physical_constants["electron volt"][0]


def state_density(mc, root):
    core = mc.mo_coeff[:, : mc.ncore]
    active = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    active_dm = fci.direct_spin0.make_rdm1(mc.ci[root], mc.ncas, mc.nelecas)
    return 2.0 * core @ core.T + active @ active_dm @ active.T


def transition_density(mc, bra, ket):
    active = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    active_dm = fci.direct_spin0.trans_rdm1(
        mc.ci[bra], mc.ci[ket], mc.ncas, mc.nelecas
    )
    return active @ active_dm @ active.T


def force_to_shift(force_au, frequency_cm1):
    force_si = force_au * HARTREE_J / (BOHR_M * np.sqrt(atomic_mass))
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequency_cm1
    return (force_si / omega_si**2) / (BOHR_M * np.sqrt(atomic_mass))


def zero_point_sigma(frequency_cm1):
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequency_cm1
    return np.sqrt(hbar / (2.0 * omega_si)) / (BOHR_M * np.sqrt(atomic_mass))


def nuclear_amplitude_ensemble(charges, reference_bohr, modes, mean_q, sigmas, q):
    """Analytic Gaussian average of the point-nuclear charge amplitude."""
    coordinates = reference_bohr + np.einsum("m,mae->ae", mean_q, modes, optimize=True)
    phase = np.exp(-1j * np.einsum("ae,qe->aq", coordinates, q, optimize=True))
    projected_widths = np.einsum("mae,m,qe->maq", modes, sigmas, q, optimize=True)
    debye_waller = np.exp(-0.5 * np.sum(projected_widths**2, axis=0))
    return np.einsum("a,aq,aq->q", charges, phase, debye_waller, optimize=True)


def paper_frame(metadata):
    x_axis = np.asarray(metadata["pi_plane_normal"], dtype=float)
    x_axis /= np.linalg.norm(x_axis)
    z_axis = np.array([0.0, 0.0, 1.0])
    y_axis = np.cross(z_axis, x_axis)
    y_axis /= np.linalg.norm(y_axis)
    return {
        "x_out_of_plane": x_axis,
        "y_in_plane": y_axis,
        "z_C2": z_axis,
        "xy_bisector": (x_axis + y_axis) / np.sqrt(2.0),
    }


def electronic_form_factors(mol, densities, q_vectors, batch=64):
    values = {label: np.empty(len(q_vectors), dtype=complex) for label in densities}
    for start in range(0, len(q_vectors), batch):
        stop = min(start + batch, len(q_vectors))
        ao_pair = ft_ao.ft_aopair(
            mol, q_vectors[start:stop], aosym="s1", return_complex=True
        )
        for label, density in densities.items():
            values[label][start:stop] = -np.einsum(
                "mn,qmn->q", density, ao_pair, optimize=True
            )
    return values


def coherent_overlap_magnitude(times, omega_fs, shifts, sigmas):
    """Exact magnitude for two driven same-curvature coherent packets."""
    shift_difference = shifts[0] - shifts[1]
    theta = times[:, None] * omega_fs[None, :]
    exponent = -np.sum(
        shift_difference[None, :] ** 2
        * (1.0 - np.cos(theta))
        / (4.0 * sigmas[None, :] ** 2),
        axis=1,
    )
    return np.exp(exponent)


def main():
    metadata = json.loads(SA_DATA.read_text(encoding="utf-8"))
    atoms = read_xyz(GEOMETRY)
    mol = gto.M(
        atom=atoms + [("X", tuple(metadata["rydberg_center_angstrom"]))],
        basis={"C": "aug-cc-pvdz", "H": "aug-cc-pvdz", "X": RYDBERG_BASIS},
        unit="Angstrom", charge=0, spin=0, symmetry="C2", verbose=4,
        output=f"{OUTPUT_PREFIX}.log", max_memory=6000,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    mc = mcscf.CASSCF(mf, 8, 4).density_fit()
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 100
    mc = mc.state_average_([1.0 / 7.0] * 7)
    mc.kernel(np.load(SA_MO))
    if not mc.converged:
        raise RuntimeError("restored SA-CASSCF calculation did not converge")

    densities = {label: state_density(mc, root) for label, root in ROOTS.items()}
    densities["xy"] = transition_density(mc, ROOTS["3px"], ROOTS["3py"])
    densities["yx"] = transition_density(mc, ROOTS["3py"], ROOTS["3px"])

    frame = paper_frame(metadata)
    direction_labels = list(frame)
    directions = np.asarray([frame[label] for label in direction_labels])
    s = np.linspace(0.40, 8.0, 305)
    q_vectors = (s[:, None, None] * au2angstrom * directions[None, :, :]).reshape(-1, 3)
    electronic_flat = electronic_form_factors(mol, densities, q_vectors)
    electronic = {
        label: values.reshape(s.size, len(directions)).T
        for label, values in electronic_flat.items()
    }

    with np.load(MODE_DATA) as mode_data:
        indices = MODE_IDS - 1
        frequencies = mode_data["frequencies_cm1"][indices]
        modes = mode_data["normal_modes"][indices]
    with np.load(PROJECTION_DATA) as projected:
        forces = np.asarray(
            [projected[f"force_projection_{state}_au"][indices] for state in ("3px", "3py")]
        )
    sigmas = np.asarray([zero_point_sigma(value) for value in frequencies])
    with np.load(DYNAMICS_DATA) as dynamics:
        times = dynamics["times_fs"]
        electronic_populations = dynamics["populations"]
        rho_xy = dynamics["coherence"]
        off_diagonal_couplings = dynamics["off_diagonal_couplings"]
        # Dynamics stores (time, state, mode); scattering code uses
        # (state, time, mode).
        mean_q = np.transpose(dynamics["conditional_mean_q_bohr_sqrtamu"], (1, 0, 2))

    physical_charges = mol.atom_charges()[:14].astype(float)
    reference_bohr = mol.atom_coords()[:14]
    ground_nuclear = np.empty((len(directions), s.size), dtype=complex)
    nuclear = np.empty((2, times.size, len(directions), s.size), dtype=complex)
    for direction_index, direction in enumerate(directions):
        q = s[:, None] * au2angstrom * direction[None, :]
        ground_nuclear[direction_index] = nuclear_amplitude_ensemble(
            physical_charges, reference_bohr, modes, np.zeros(len(MODE_IDS)), sigmas, q
        )
        for state in range(2):
            for time_index in range(times.size):
                nuclear[state, time_index, direction_index] = nuclear_amplitude_ensemble(
                    physical_charges, reference_bohr, modes, mean_q[state, time_index], sigmas, q,
                )

    ground_amplitude = ground_nuclear + electronic["ground"]
    ground_intensity = np.abs(ground_amplitude) ** 2
    diagonal_amplitude = (
        electronic_populations[:, 0, None, None]
        * (nuclear[0] + electronic["3px"][None, :, :])
        + electronic_populations[:, 1, None, None]
        * (nuclear[1] + electronic["3py"][None, :, :])
    )

    nevpt2 = json.loads(NEVPT2_DATA.read_text(encoding="utf-8"))
    energies = np.asarray(nevpt2["sc_nevpt2_vertical_excitation_energies_ev"])
    gap_ev = float(energies[3] - energies[2])
    coherence_amplitude = (
        rho_xy[:, None, None] * electronic["xy"][None, :, :]
        + np.conj(rho_xy)[:, None, None] * electronic["yx"][None, :, :]
    )
    coherent_amplitude = diagonal_amplitude + coherence_amplitude
    coherent_intensity = np.abs(coherent_amplitude) ** 2
    diagonal_intensity = np.abs(diagonal_amplitude) ** 2
    pd = 100.0 * (coherent_intensity - ground_intensity[None, :, :]) / ground_intensity[None, :, :]
    pd_no_coherence = 100.0 * (
        diagonal_intensity - ground_intensity[None, :, :]
    ) / ground_intensity[None, :, :]
    pd_coherence = pd - pd_no_coherence
    reliable = ground_intensity > 1.0e-3 * np.max(ground_intensity, axis=1)[:, None]

    np.savez_compressed(
        f"{OUTPUT_PREFIX}.npz", times_fs=times, s_angstrom_inverse=s,
        direction_labels=np.asarray(direction_labels), directions_xyz=directions,
        PD_percent=pd, PD_no_coherence_percent=pd_no_coherence,
        PD_coherence_percent=pd_coherence, rho_xy=rho_xy,
        electronic_populations=electronic_populations,
        transition_form_factor_xy=electronic["xy"],
        ground_intensity=ground_intensity, state_mean_q=mean_q,
    )

    reliable_3d = np.broadcast_to(reliable[None, :, :], pd.shape)
    summary = {
        "method": "oriented elastic SA-CASSCF density + coupled three-mode LVC wavepacket",
        "observable": "PD = 100 * (I_coherent(t)-I_ground)/I_ground",
        "rotational_average": False,
        "paper_frame_axes_in_stored_xyz": {k: v.tolist() for k, v in frame.items()},
        "directions": direction_labels,
        "initial_state": "sqrt(1/1.8)|3px> + sqrt(0.8/1.8)|3py>",
        "electronic_gap_ev_for_phase": gap_ev,
        "electronic_beat_period_fs": float(2.0 * np.pi * hbar / (gap_ev * EV_J) * 1.0e15),
        "transition_density": "SA-CASSCF root-2/root-3 one-electron transition density",
        "electronic_dynamics": "two-state/three-mode split-operator wavepacket with analytic off-diagonal couplings",
        "mode_ids": MODE_IDS.tolist(),
        "off_diagonal_couplings_Eh_per_bohr_sqrtamu": {
            f"mode_{mode_id}": float(coupling)
            for mode_id, coupling in zip(MODE_IDS, off_diagonal_couplings)
        },
        "population_ranges": {
            "3px": [
                float(electronic_populations[:, 0].min()),
                float(electronic_populations[:, 0].max()),
            ],
            "3py": [
                float(electronic_populations[:, 1].min()),
                float(electronic_populations[:, 1].max()),
            ],
        },
        "coherence_magnitude_range": [
            float(np.abs(rho_xy).min()), float(np.abs(rho_xy).max())
        ],
        "display_reliability_threshold": "I_ground > 1e-3 of directional maximum",
        "max_abs_PD_percent_reliable": float(np.max(np.abs(pd[reliable_3d]))),
        "max_abs_coherence_PD_percent_reliable": float(
            np.max(np.abs(pd_coherence[reliable_3d]))
        ),
        "limitations": [
            "electronic densities fixed at the ground-state reference geometry",
            "nuclear scattering uses Gaussian packets centered on conditional LVC means",
            "no rotational average, pump photoselection, lifetime, or instrument convolution",
            "oriented ground-state diffraction minima can make PD very large",
        ],
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(json.dumps(summary, indent=2) + "\n")

    figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharex=True, sharey=True)
    labels = {
        "x_out_of_plane": r"$\mathbf{q}\parallel x$ (out of plane)",
        "y_in_plane": r"$\mathbf{q}\parallel y$ (in plane)",
        "z_C2": r"$\mathbf{q}\parallel z$ ($C_2$ axis)",
        "xy_bisector": r"$\mathbf{q}\parallel(x+y)/\sqrt{2}$",
    }
    display_values = np.where(reliable[None, :, :], pd, np.nan)
    limit = np.nanpercentile(np.abs(display_values), 98.5)
    for panel, (axis, label) in enumerate(zip(axes.flat, direction_labels)):
        image = axis.pcolormesh(
            s, times, display_values[:, panel], shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, rasterized=True,
        )
        axis.set_title(labels[label], fontsize=10)
        axis.text(-0.13, 1.02, chr(ord("a") + panel), transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    axes[1, 0].set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
    axes[1, 1].set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
    axes[0, 0].set_ylabel("Time delay (fs)")
    axes[1, 0].set_ylabel("Time delay (fs)")
    figure.subplots_adjust(
        left=0.10, right=0.84, bottom=0.09, top=0.94, wspace=0.16, hspace=0.18
    )
    colorbar_axis = figure.add_axes([0.88, 0.16, 0.025, 0.70])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("PD (%)")
    figure.savefig(f"{OUTPUT_PREFIX}.pdf")
    figure.savefig(f"{OUTPUT_PREFIX}.png", dpi=350)
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
