"""Seven-root CASCI(6,10) plus state-specific SC-NEVPT2 for CHD."""

import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, mrpt, scf

from chd_c2_sa_casscf610_aug_rydberg import ghost_metric
from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
REFERENCE = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_DATA = Path("chd_c2_sa_casscf610_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf610_aug_rydberg_mo.npy")
OUTPUT_PREFIX = "chd_c2_casci610_nevpt2"
HARTREE_TO_EV = 27.211386245988


def main():
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    sa_data = json.loads(SA_DATA.read_text(encoding="utf-8"))
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
    # Do not density-fit this reference. The automatically generated auxiliary
    # basis on the floating Rydberg center gives strongly root-dependent and
    # nonphysical DF-NEVPT2 corrections.
    mf = scf.RHF(mol).run(conv_tol=1.0e-10)
    casci = mcscf.CASCI(mf, 10, 6)
    casci.fcisolver = fci.direct_spin0.FCI(mol)
    casci.fcisolver.nroots = 7
    casci.fcisolver.conv_tol = 1.0e-10
    casci.kernel(np.load(SA_MO))

    casci_energies = np.asarray(casci.e_tot, dtype=float)
    active_coeff = casci.mo_coeff[:, casci.ncore : casci.ncore + casci.ncas]
    ryd_metric = ghost_metric(mol, mf.get_ovlp(), active_coeff)
    state_dm1 = [
        fci.direct_spin0.make_rdm1(ci, casci.ncas, casci.nelecas)
        for ci in casci.ci
    ]
    rydberg_populations = [
        float(np.einsum("ij,ji->", density, ryd_metric)) for density in state_dm1
    ]
    occupations = [np.diag(density).tolist() for density in state_dm1]

    corrections = []
    for root in range(7):
        print(f"Starting SC-NEVPT2 root {root}", flush=True)
        correction = mrpt.NEVPT(casci, root=root, density_fit=False).kernel()
        corrections.append(float(correction))
        print(
            f"Finished root {root}: E_corr={correction:.12f} Eh",
            flush=True,
        )

    corrections = np.asarray(corrections)
    total_energies = casci_energies + corrections
    casci_excitations = (casci_energies - casci_energies[0]) * HARTREE_TO_EV
    nevpt2_excitations = (total_energies - total_energies[0]) * HARTREE_TO_EV
    ordering = np.argsort(total_energies)
    result = {
        "method": "state-specific conventional SC-NEVPT2 on seven-root CASCI(6,10)",
        "reference_orbitals": "seven-state equal-weight SA-CASSCF(6,10)",
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
        "casci_energies_hartree": casci_energies.tolist(),
        "sc_nevpt2_corrections_hartree": corrections.tolist(),
        "density_fitting_used": False,
        "integral_note": (
            "Final values use a genuinely non-density-fitted RHF/CASCI object. "
            "Do not request density_fit=False from a DFCASCI object as a fallback."
        ),
        "sc_nevpt2_total_energies_hartree": total_energies.tolist(),
        "casci_vertical_excitation_energies_ev": casci_excitations.tolist(),
        "sc_nevpt2_vertical_excitation_energies_ev": nevpt2_excitations.tolist(),
        "sc_nevpt2_energy_order_zero_based_roots": ordering.tolist(),
        "state_rydberg_electron_populations": rydberg_populations,
        "state_active_orbital_occupations": occupations,
        "sa_casscf_vertical_excitation_energies_ev": sa_data[
            "vertical_excitation_energies_ev"
        ],
    }
    Path(f"{OUTPUT_PREFIX}.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
