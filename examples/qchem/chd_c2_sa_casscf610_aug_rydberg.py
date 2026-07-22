"""Seven-state CHD SA-CASSCF(6,10) with pi, Rydberg, and ring sigma orbitals."""

import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
REFERENCE = Path("chd_c2_sa_casscf48_aug_rydberg.json")
REFERENCE_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
OUTPUT_PREFIX = "chd_c2_sa_casscf610_aug_rydberg"
HARTREE_TO_EV = 27.211386245988
# In the restored CAS(4,8) optimized MO array, positions 21-28 are active.
# Position 20 is the highest inactive ring-closing sigma orbital. Position 65
# is the virtual with the largest projected ring sigma* hybrid character.
ACTIVE_ONE_BASED = [20, 21, 22, 23, 24, 25, 26, 27, 28, 65]


def ghost_metric(mol, overlap, active_coeff):
    labels = [label.split() for label in mol.ao_labels()]
    ghost = mol.natm - 1
    indices = [i for i, label in enumerate(labels) if int(label[0]) == ghost]
    targets = np.eye(mol.nao_nr())[:, indices]
    metric = targets.T @ overlap @ targets
    values, vectors = np.linalg.eigh(metric)
    projector = targets @ vectors @ np.diag(values**-0.5)
    amplitudes = projector.T @ overlap @ active_coeff
    return amplitudes.T @ amplitudes


def main():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    atoms = read_xyz(GEOMETRY)
    mol = gto.M(
        atom=atoms + [("X", tuple(reference["rydberg_center_angstrom"]))],
        basis={"C": "aug-cc-pvdz", "H": "aug-cc-pvdz", "X": RYDBERG_BASIS},
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry="C2",
        verbose=4,
        output=f"{OUTPUT_PREFIX}.log",
        max_memory=6000,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    mc = mcscf.CASSCF(mf, 10, 6).density_fit()
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 120
    mc = mc.state_average_([1.0 / 7.0] * 7)
    initial_mo = mc.sort_mo(ACTIVE_ONE_BASED, mo_coeff=np.load(REFERENCE_MO))
    mc.kernel(initial_mo)
    if not mc.converged:
        raise RuntimeError("Seven-state SA-CASSCF(6,10) did not converge")

    energies = np.asarray(mc.e_states)
    excitations = (energies - energies[0]) * HARTREE_TO_EV
    active_coeff = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    ryd_metric = ghost_metric(mol, mf.get_ovlp(), active_coeff)
    state_dm1 = [
        fci.direct_spin0.make_rdm1(ci, mc.ncas, mc.nelecas) for ci in mc.ci
    ]
    rydberg_populations = [
        float(np.einsum("ij,ji->", density, ryd_metric)) for density in state_dm1
    ]
    occupations = [np.diag(density).tolist() for density in state_dm1]

    position_ao = mol.intor_symmetric("int1e_r", comp=3)
    position_active = np.einsum(
        "pi,xpq,qj->xij", active_coeff, position_ao, active_coeff, optimize=True
    )
    dipoles = []
    oscillator_strengths = []
    for state in range(1, 7):
        transition_dm1 = fci.direct_spin0.trans_rdm1(
            mc.ci[state], mc.ci[0], mc.ncas, mc.nelecas
        )
        dipole = -np.einsum("xij,ji->x", position_active, transition_dm1)
        dipoles.append(dipole.tolist())
        oscillator_strengths.append(
            float(
                (2.0 / 3.0)
                * (energies[state] - energies[0])
                * np.dot(dipole, dipole)
            )
        )

    np.save(f"{OUTPUT_PREFIX}_mo.npy", mc.mo_coeff)
    result = {
        "method": "seven-state equal-weight DF-SA-CASSCF(6,10)",
        "basis": "aug-cc-pVDZ + Kaufmann 3s/3p",
        "root_labels_paper_frame": [
            "X", "3s", "3px", "3py", "3pz", "dark valence", "bright 1B"
        ],
        "state_label_convention": {
            "source": "Yong/Ruddock CHD 200 nm papers",
            "x": "normal to the conjugated-carbon plane (out of plane)",
            "z": "molecular C2 axis",
            "y": "in-plane axis completing a right-handed x,y,z frame",
            "warning": "molecular-frame labels, not stored-XYZ Cartesian axes",
        },
        "active_space": "ring sigma/sigma* + four pi/pi* + Kaufmann 3s/3p",
        "initial_active_orbitals_one_based_in_restored_mo": ACTIVE_ONE_BASED,
        "state_energies_hartree": energies.tolist(),
        "vertical_excitation_energies_ev": excitations.tolist(),
        "active_orbital_ghost_weights": np.diag(ryd_metric).tolist(),
        "state_rydberg_electron_populations": rydberg_populations,
        "state_active_orbital_occupations": occupations,
        "ground_to_excited_transition_dipoles_au": dipoles,
        "ground_to_excited_oscillator_strengths": oscillator_strengths,
        "state_average_energy_hartree": float(mc.e_tot),
        "converged": bool(mc.converged),
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
