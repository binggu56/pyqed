"""Elastic plus energy-integrated electronic-inelastic CHD UED PD signal."""

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
from pyscf import fci, gto, mcscf, scf
from pyscf.gto import ft_ao

from pyqed import au2angstrom


QCHEM_EXAMPLES = Path(__file__).resolve().parents[1] / "qchem"
if str(QCHEM_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(QCHEM_EXAMPLES))

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
ELASTIC_DATA = Path("chd_c2_lvc_level2_coherent_oriented_pd_3mode.npz")
DYNAMICS_DATA = Path("chd_c2_lvc_coupled_wavepacket_3mode.npz")
OUTPUT = Path("chd_c2_lvc_total_elastic_inelastic_oriented_pd_3mode")
ROOTS = {"ground": 0, "3px": 2, "3py": 3}


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


def occupied_rdm12(mc, bra, ket):
    """Spin-free CASSCF transition RDMs in the core+active MO space."""
    casdm1, casdm2 = fci.direct_spin0.trans_rdm12(
        mc.ci[bra], mc.ci[ket], mc.ncas, mc.nelecas
    )
    overlap = np.vdot(mc.ci[bra], mc.ci[ket])
    ncore = mc.ncore
    nocc = ncore + mc.ncas
    dm1 = np.zeros((nocc, nocc), dtype=complex)
    dm2 = np.zeros((nocc, nocc, nocc, nocc), dtype=complex)
    core = np.arange(ncore)
    dm1[core, core] = 2.0 * overlap
    dm1[ncore:, ncore:] = casdm1
    dm2[ncore:, ncore:, ncore:, ncore:] = casdm2
    for i in range(ncore):
        for j in range(ncore):
            dm2[i, i, j, j] += 4.0 * overlap
            dm2[i, j, j, i] -= 2.0 * overlap
        dm2[i, i, ncore:, ncore:] = 2.0 * casdm1
        dm2[ncore:, ncore:, i, i] = 2.0 * casdm1
        dm2[i, ncore:, ncore:, i] = -casdm1
        dm2[ncore:, i, i, ncore:] = -casdm1
    return dm1, dm2


def one_body_expectation_squared(mo_operator, dm1, dm2):
    """Return <bra|rho(-q) rho(q)|ket> using finite-basis closure."""
    nocc = dm1.shape[0]
    all_occ = mo_operator[:, :nocc]
    occ = mo_operator[:nocc, :nocc]
    one_body = np.einsum(
        "pq,ps,sq->", all_occ.conj(), all_occ, dm1, optimize=True
    )
    two_body = np.einsum(
        "pq,rs,qprs->", occ.conj(), occ, dm2, optimize=True
    )
    return one_body + two_body


def density_correlation_form_factors(mol, mc, rdms, q_vectors, batch=8):
    values = {label: np.empty(len(q_vectors), dtype=complex) for label in rdms}
    coefficients = mc.mo_coeff
    for start in range(0, len(q_vectors), batch):
        stop = min(start + batch, len(q_vectors))
        ao_pairs = ft_ao.ft_aopair(
            mol, q_vectors[start:stop], aosym="s1", return_complex=True
        )
        mo_pairs = np.einsum(
            "up,quv,vr->qpr", coefficients.conj(), ao_pairs, coefficients,
            optimize=True,
        )
        for local_index, mo_pair in enumerate(mo_pairs):
            for label, (dm1, dm2) in rdms.items():
                values[label][start + local_index] = one_body_expectation_squared(
                    mo_pair, dm1, dm2
                )
    return values


def electronic_form_factors(mol, densities, q_vectors, batch=64):
    values = {label: np.empty(len(q_vectors), dtype=complex) for label in densities}
    for start in range(0, len(q_vectors), batch):
        stop = min(start + batch, len(q_vectors))
        ao_pairs = ft_ao.ft_aopair(
            mol, q_vectors[start:stop], aosym="s1", return_complex=True
        )
        for label, density in densities.items():
            values[label][start:stop] = np.einsum(
                "mn,qmn->q", density, ao_pairs, optimize=True
            )
    return values


def plot_pd(s, times, labels, values, reliable, output):
    titles = {
        "x_out_of_plane": r"$\mathbf{q}\parallel x$ (out of plane)",
        "y_in_plane": r"$\mathbf{q}\parallel y$ (in plane)",
        "z_C2": r"$\mathbf{q}\parallel z$ ($C_2$ axis)",
        "xy_bisector": r"$\mathbf{q}\parallel(x+y)/\sqrt{2}$",
    }
    display = np.where(reliable[None, :, :], values, np.nan)
    limit = np.nanpercentile(np.abs(display), 98.5)
    figure, axes = plt.subplots(2, 2, figsize=(8.4, 7.0), sharex=True, sharey=True)
    for panel, (axis, label) in enumerate(zip(axes.flat, labels)):
        image = axis.pcolormesh(
            s, times, display[:, panel], shading="auto", cmap="RdBu_r",
            vmin=-limit, vmax=limit, rasterized=True,
        )
        axis.set_title(titles[label], fontsize=10)
        axis.text(-0.13, 1.02, chr(ord("a") + panel), transform=axis.transAxes,
                  fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    for axis in axes[1]:
        axis.set_xlabel(r"$s$ ($\mathrm{\AA}^{-1}$)")
    for axis in axes[:, 0]:
        axis.set_ylabel("Time delay (fs)")
    figure.subplots_adjust(left=0.10, right=0.84, bottom=0.09, top=0.94,
                           wspace=0.16, hspace=0.18)
    colorbar_axis = figure.add_axes([0.88, 0.16, 0.025, 0.70])
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Total PD, elastic + inelastic (%)")
    figure.savefig(f"{output}.pdf")
    figure.savefig(f"{output}.png", dpi=400)
    plt.close(figure)


def main():
    metadata = json.loads(SA_DATA.read_text(encoding="utf-8"))
    atoms = read_xyz(GEOMETRY)
    mol = gto.M(
        atom=atoms + [("X", tuple(metadata["rydberg_center_angstrom"]))],
        basis={"C": "aug-cc-pvdz", "H": "aug-cc-pvdz", "X": RYDBERG_BASIS},
        unit="Angstrom", charge=0, spin=0, symmetry="C2", verbose=4,
        output=f"{OUTPUT}.log", max_memory=6000,
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

    with np.load(ELASTIC_DATA) as data:
        times = data["times_fs"]
        s = data["s_angstrom_inverse"]
        labels = data["direction_labels"].astype(str)
        directions = data["directions_xyz"]
        populations = data["electronic_populations"]
        rho_xy = data["rho_xy"]
    with np.load(DYNAMICS_DATA) as dynamics:
        nuclear_amplitude = dynamics["nuclear_amplitude_expectation"]
        nuclear_cc = dynamics["nuclear_charge_squared_expectation"]
        nuclear_cstar_rho = dynamics["nuclear_cstar_electronic_rho"]
        if not np.allclose(times, dynamics["times_fs"]):
            raise RuntimeError("electronic and nuclear time grids differ")
    q_vectors = (s[:, None, None] * au2angstrom * directions[None, :, :]).reshape(-1, 3)

    densities = {
        label: state_density(mc, root) for label, root in ROOTS.items()
    }
    densities["xy"] = transition_density(mc, ROOTS["3px"], ROOTS["3py"])
    densities["yx"] = transition_density(mc, ROOTS["3py"], ROOTS["3px"])
    electronic_flat = electronic_form_factors(mol, densities, q_vectors)
    electronic = {
        label: value.reshape(s.size, len(directions)).T
        for label, value in electronic_flat.items()
    }

    rdms = {
        "ground": occupied_rdm12(mc, ROOTS["ground"], ROOTS["ground"]),
        "3px": occupied_rdm12(mc, ROOTS["3px"], ROOTS["3px"]),
        "3py": occupied_rdm12(mc, ROOTS["3py"], ROOTS["3py"]),
        "xy": occupied_rdm12(mc, ROOTS["3px"], ROOTS["3py"]),
    }
    correlation_flat = density_correlation_form_factors(mol, mc, rdms, q_vectors)
    correlation = {
        label: value.reshape(s.size, len(directions)).T
        for label, value in correlation_flat.items()
    }

    electron_amplitude = (
        populations[:, 0, None, None] * electronic["3px"][None, :, :]
        + populations[:, 1, None, None] * electronic["3py"][None, :, :]
        + rho_xy[:, None, None] * electronic["xy"][None, :, :]
        + np.conj(rho_xy)[:, None, None] * electronic["yx"][None, :, :]
    )
    electron_correlation = (
        populations[:, 0, None, None] * correlation["3px"][None, :, :]
        + populations[:, 1, None, None] * correlation["3py"][None, :, :]
        + rho_xy[:, None, None] * correlation["xy"][None, :, :]
        + np.conj(rho_xy)[:, None, None] * np.conj(correlation["xy"])[None, :, :]
    )
    elastic_excited = np.abs(nuclear_amplitude - electron_amplitude) ** 2
    electron_operators = np.empty((2, 2, len(directions), len(s)), dtype=complex)
    electron_operators[0, 0] = electronic["3px"]
    electron_operators[1, 1] = electronic["3py"]
    electron_operators[0, 1] = electronic["xy"]
    electron_operators[1, 0] = electronic["yx"]
    nuclear_electronic_cross = np.einsum(
        "tijds,ijds->tds", nuclear_cstar_rho, electron_operators, optimize=True
    )
    total_excited = (
        nuclear_cc - 2.0 * np.real(nuclear_electronic_cross)
        + np.real(electron_correlation)
    )
    inelastic_excited = total_excited - elastic_excited

    ground_nuclear_amplitude = nuclear_amplitude[0]
    ground_nuclear_cc = nuclear_cc[0]
    elastic_ground = np.abs(ground_nuclear_amplitude - electronic["ground"]) ** 2
    total_ground = (
        ground_nuclear_cc
        - 2.0 * np.real(np.conj(ground_nuclear_amplitude) * electronic["ground"])
        + np.real(correlation["ground"])
    )
    inelastic_ground = total_ground - elastic_ground
    minimum_inelastic = min(float(inelastic_ground.min()), float(inelastic_excited.min()))
    if minimum_inelastic < -1.0e-7:
        raise RuntimeError(f"negative inelastic intensity: {minimum_inelastic:.3e}")
    inelastic_ground = np.maximum(inelastic_ground, 0.0)
    inelastic_excited = np.maximum(inelastic_excited, 0.0)
    total_ground = elastic_ground + inelastic_ground
    total_excited = elastic_excited + inelastic_excited

    total_pd = 100.0 * (total_excited - total_ground[None, :, :]) / total_ground[None, :, :]
    elastic_pd = 100.0 * (
        elastic_excited - elastic_ground[None, :, :]
    ) / elastic_ground[None, :, :]
    reliable = total_ground > 1.0e-3 * np.max(total_ground, axis=1)[:, None]

    # The number-density operator must have zero inelastic variance at q=0.
    overlap_operator = mc.mo_coeff.conj().T @ mol.intor("int1e_ovlp") @ mc.mo_coeff
    number_variances = {}
    for label in ("ground", "3px", "3py"):
        expectation_squared = one_body_expectation_squared(overlap_operator, *rdms[label])
        electron_count = np.trace(rdms[label][0])
        number_variances[label] = float(np.real(expectation_squared - electron_count**2))

    np.savez_compressed(
        f"{OUTPUT}.npz", times_fs=times, s_angstrom_inverse=s,
        direction_labels=labels, directions_xyz=directions,
        PD_total_percent=total_pd, PD_elastic_percent=elastic_pd,
        elastic_ground=elastic_ground, elastic_excited=elastic_excited,
        inelastic_ground=inelastic_ground, inelastic_excited=inelastic_excited,
        nuclear_diffuse_ground=ground_nuclear_cc - np.abs(ground_nuclear_amplitude) ** 2,
        nuclear_diffuse_excited=nuclear_cc - np.abs(nuclear_amplitude) ** 2,
        nuclear_electronic_cross=nuclear_electronic_cross,
        total_ground=total_ground, total_excited=total_excited,
        electronic_density_correlation_ground=correlation["ground"],
        electronic_density_correlation_3px=correlation["3px"],
        electronic_density_correlation_3py=correlation["3py"],
        electronic_density_correlation_xy=correlation["xy"],
    )
    summary = {
        "method": "full three-mode vibronic nuclear correlation plus SA-CASSCF finite-basis electronic closure",
        "states": ["paper-frame 3px", "paper-frame 3py"],
        "modes": [5, 8, 26],
        "number_operator_variance_q0": number_variances,
        "minimum_unclipped_inelastic_intensity": minimum_inelastic,
        "inelastic_to_total_ground_range": [
            float(np.min(inelastic_ground / total_ground)),
            float(np.max(inelastic_ground / total_ground)),
        ],
        "included": [
            "energy-integrated electronic inelastic scattering in the finite aug-cc-pVDZ+Rydberg basis",
            "nuclear vibrational diffuse scattering from the three-mode LVC wavepacket",
            "state-resolved nuclear-electronic interference",
        ],
        "limitations": [
            "electronic RDMs fixed at the ground-state reference geometry",
            "no rotational average, pump photoselection, lifetime, or instrument convolution",
        ],
    }
    Path(f"{OUTPUT}.json").write_text(json.dumps(summary, indent=2) + "\n")
    plot_pd(s, times, labels, total_pd, reliable, OUTPUT)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
