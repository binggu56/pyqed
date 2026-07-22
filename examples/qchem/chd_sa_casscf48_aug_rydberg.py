"""Seven-state SA-CASSCF(4,8) calculation for neutral CHD.

This follows the electronic-structure setup of Yong et al., Nat. Commun. 11,
2157 (2020): aug-cc-pVDZ plus uncontracted Kaufmann 3s/3p functions at the
center of charge of CHD+, with four valence pi/pi* and four Rydberg orbitals.
"""

import json
import os
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf


GEOMETRY_FILE = Path(os.environ.get("CHD_GEOMETRY_FILE", "chd_casscf44_augccpvdz.xyz"))
OUTPUT_PREFIX = os.environ.get("CHD_SA_OUTPUT_PREFIX", "chd_sa_casscf48_aug_rydberg")
MOLECULAR_SYMMETRY = os.environ.get("CHD_MOLECULAR_SYMMETRY", "") or False
RYDBERG_BASIS = [
    [0, [0.0058583806, 1.0]],  # Kaufmann 3s, bohr^-2
    [1, [0.0099882104, 1.0]],  # Kaufmann 3p, bohr^-2
]
HARTREE_TO_EV = 27.211386245988


def read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()[2:]
    return [(fields[0], tuple(map(float, fields[1:4]))) for fields in map(str.split, lines)]


def target_projector(mol, overlap, columns):
    targets = np.column_stack(columns)
    metric = targets.T @ overlap @ targets
    values, vectors = np.linalg.eigh(metric)
    if np.min(values) < 1.0e-10:
        raise RuntimeError("Linearly dependent target AO vectors")
    return targets @ vectors @ np.diag(values ** -0.5)


def pi_projector(mol, overlap, conjugated_atoms=(2, 3, 4, 5)):
    """Project onto one plane-normal C(2p) AO on each conjugated carbon."""
    coords = mol.atom_coords(unit="Angstrom")[list(conjugated_atoms)]
    _, _, vh = np.linalg.svd(coords - coords.mean(axis=0))
    normal = vh[-1]
    labels = [label.split() for label in mol.ao_labels()]
    columns = []
    for atom_index in conjugated_atoms:
        target = np.zeros(mol.nao_nr())
        for component, coefficient in zip(("2px", "2py", "2pz"), normal):
            matches = [
                i
                for i, label in enumerate(labels)
                if int(label[0]) == atom_index
                and label[1] == "C"
                and label[2] == component
            ]
            if len(matches) != 1:
                raise RuntimeError(f"Could not identify {atom_index} C {component}")
            target[matches[0]] = coefficient
        columns.append(target)
    return target_projector(mol, overlap, columns), normal


def ghost_projector(mol, overlap):
    labels = [label.split() for label in mol.ao_labels()]
    ghost_index = mol.natm - 1
    columns = []
    for ao_index, label in enumerate(labels):
        if int(label[0]) == ghost_index:
            target = np.zeros(mol.nao_nr())
            target[ao_index] = 1.0
            columns.append(target)
    if len(columns) != 4:
        raise RuntimeError(f"Expected four Rydberg AOs, found {len(columns)}")
    return target_projector(mol, overlap, columns)


def orbital_weights(mf, projector):
    overlap = mf.get_ovlp()
    amplitudes = projector.T @ overlap @ mf.mo_coeff
    return np.sum(amplitudes**2, axis=0)


def select_valence_pi(mf):
    projector, normal = pi_projector(mf.mol, mf.get_ovlp())
    weights = orbital_weights(mf, projector)
    occupied = np.flatnonzero(mf.mo_occ > 0)
    virtual = np.flatnonzero(mf.mo_occ == 0)
    pi_occ = occupied[np.argsort(weights[occupied])[-2:]]
    pi_vir = virtual[np.argsort(weights[virtual])[-2:]]
    return list(pi_occ) + list(pi_vir), weights, normal


def cation_center_of_charge(atoms):
    """Compute the CHD+ charge center from a CASSCF(3,4) cation density."""
    mol = gto.M(
        atom=atoms,
        basis="aug-cc-pvdz",
        unit="Angstrom",
        charge=1,
        spin=1,
        symmetry=MOLECULAR_SYMMETRY,
        verbose=4,
        output=f"{OUTPUT_PREFIX}_cation_center.log",
        max_memory=6000,
    )
    mf = scf.ROHF(mol).density_fit().run(conv_tol=1.0e-10)
    active, _, _ = select_valence_pi(mf)
    mc = mcscf.CASSCF(mf, 4, (2, 1)).density_fit()
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 80
    mc.kernel(mc.sort_mo([index + 1 for index in active]))
    if not mc.converged:
        raise RuntimeError("Cation CASSCF(3,4) did not converge")

    density = mc.make_rdm1()
    position_integrals = mol.intor_symmetric("int1e_r", comp=3)
    electronic = -np.einsum("xij,ji->x", position_integrals, density)
    nuclear = np.einsum("i,ix->x", mol.atom_charges(), mol.atom_coords())
    dipole_au = nuclear + electronic
    # The cation has net charge +1, so R_charge = dipole / Q.
    center_angstrom = dipole_au * 0.529177210903
    return center_angstrom, float(mc.e_tot), [int(index + 1) for index in active]


def main():
    atoms = read_xyz(GEOMETRY_FILE)
    center, cation_energy, cation_active = cation_center_of_charge(atoms)

    mol = gto.M(
        atom=atoms + [("X", tuple(center))],
        basis={"C": "aug-cc-pvdz", "H": "aug-cc-pvdz", "X": RYDBERG_BASIS},
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry=MOLECULAR_SYMMETRY,
        verbose=4,
        output=f"{OUTPUT_PREFIX}.log",
        max_memory=6000,
    )
    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    overlap = mf.get_ovlp()
    pi_space, pi_weights, plane_normal = select_valence_pi(mf)
    pi_occ = pi_space[:2]
    pi_vir = pi_space[2:]

    ryd_projector = ghost_projector(mol, overlap)
    ryd_weights = orbital_weights(mf, ryd_projector)
    virtual = np.flatnonzero(mf.mo_occ == 0)
    rydberg = list(virtual[np.argsort(ryd_weights[virtual])[-4:]])
    if set(pi_vir) & set(rydberg):
        raise RuntimeError("Valence pi* and Rydberg selections overlap")

    active = pi_occ + pi_vir + rydberg
    active_one_based = [int(index + 1) for index in active]
    mc = mcscf.CASSCF(mf, 8, 4).density_fit()
    # Restrict the state average to singlets. The default direct_spin1 solver
    # otherwise includes lower triplet roots in a multi-root calculation.
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 100
    mc = mc.state_average_([1.0 / 7.0] * 7)
    mc.kernel(mc.sort_mo(active_one_based))
    if not mc.converged:
        raise RuntimeError("Seven-state SA-CASSCF(4,8) did not converge")

    energies = np.asarray(mc.e_states, dtype=float)
    excitations = (energies - energies[0]) * HARTREE_TO_EV
    active_coeff = mc.mo_coeff[:, mc.ncore : mc.ncore + mc.ncas]
    ghost_amplitudes = ryd_projector.T @ overlap @ active_coeff
    ghost_metric = ghost_amplitudes.T @ ghost_amplitudes
    state_rdm1 = [
        fci.direct_spin0.make_rdm1(ci, mc.ncas, mc.nelecas) for ci in mc.ci
    ]
    state_rydberg_populations = [
        float(np.einsum("ij,ji->", density, ghost_metric)) for density in state_rdm1
    ]
    state_active_occupations = [np.diag(density).tolist() for density in state_rdm1]
    position_ao = mol.intor_symmetric("int1e_r", comp=3)
    position_active = np.einsum(
        "pi,xpq,qj->xij", active_coeff, position_ao, active_coeff, optimize=True
    )
    transition_dipoles_au = []
    oscillator_strengths = []
    for state in range(1, 7):
        transition_rdm1 = fci.direct_spin0.trans_rdm1(
            mc.ci[state], mc.ci[0], mc.ncas, mc.nelecas
        )
        dipole = -np.einsum("xij,ji->x", position_active, transition_rdm1)
        transition_dipoles_au.append(dipole.tolist())
        oscillator_strengths.append(
            float((2.0 / 3.0) * (energies[state] - energies[0]) * np.dot(dipole, dipole))
        )
    np.save(f"{OUTPUT_PREFIX}_mo.npy", mc.mo_coeff)
    result = {
        "method": "seven-state equal-weight DF-SA-CASSCF(4,8)",
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
        "molecular_symmetry": str(MOLECULAR_SYMMETRY or "C1"),
        "basis": "aug-cc-pVDZ + Kaufmann 3s/3p",
        "kaufmann_exponents_bohr_minus_2": {"3s": 0.0058583806, "3p": 0.0099882104},
        "rydberg_center_angstrom": center.tolist(),
        "cation_casscf34_energy_hartree": cation_energy,
        "cation_active_orbitals_one_based": cation_active,
        "neutral_active_orbitals_one_based": active_one_based,
        "neutral_pi_orbitals_one_based": [int(index + 1) for index in pi_occ + pi_vir],
        "neutral_rydberg_orbitals_one_based": [int(index + 1) for index in rydberg],
        "pi_plane_normal": plane_normal.tolist(),
        "state_weights": [1.0 / 7.0] * 7,
        "state_energies_hartree": energies.tolist(),
        "vertical_excitation_energies_ev": excitations.tolist(),
        "active_orbital_ghost_weights": np.diag(ghost_metric).tolist(),
        "state_rydberg_electron_populations": state_rydberg_populations,
        "state_active_orbital_occupations": state_active_occupations,
        "ground_to_excited_transition_dipoles_au": transition_dipoles_au,
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
