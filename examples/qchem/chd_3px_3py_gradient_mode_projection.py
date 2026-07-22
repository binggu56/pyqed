"""Project paper-frame CHD 3px/3py forces onto ground-state normal modes.

Labels follow the CHD 200 nm papers: x is normal to the conjugated-carbon
plane, z is the C2 axis, and y completes the right-handed molecular frame.
They do not denote the Cartesian axes of the stored XYZ file.
"""

import csv
import json
from pathlib import Path

import numpy as np
from pyscf import fci, gto, mcscf, scf
from scipy.constants import atomic_mass, hbar, physical_constants, speed_of_light

from chd_sa_casscf48_aug_rydberg import RYDBERG_BASIS, read_xyz


GEOMETRY = Path("chd_c2_casscf44_augccpvdz.xyz")
SA_DATA = Path("chd_c2_sa_casscf48_aug_rydberg.json")
SA_MO = Path("chd_c2_sa_casscf48_aug_rydberg_mo.npy")
NORMAL_MODES = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
OUTPUT_PREFIX = Path("chd_c2_3px_3py_mode_projection")
# Adiabatic roots assigned in the molecular-axis convention of the papers.
ROOTS = {"3px": 2, "3py": 3}
HARTREE_J = physical_constants["Hartree energy"][0]
BOHR_M = physical_constants["Bohr radius"][0]
BOHR_ANGSTROM = BOHR_M * 1.0e10


def project_force(force_au, modes, frequencies_cm1, coordinates_bohr):
    """Return force projections and displaced-harmonic mode descriptors."""
    force_q_au = np.einsum("kax,ax->k", modes, force_au)
    force_q_si = force_q_au * HARTREE_J / (BOHR_M * np.sqrt(atomic_mass))
    omega_si = 2.0 * np.pi * speed_of_light * 100.0 * frequencies_cm1
    delta_q_si = force_q_si / omega_si**2
    q_unit_si = BOHR_M * np.sqrt(atomic_mass)
    delta_q_native = delta_q_si / q_unit_si
    huang_rhys = omega_si * delta_q_si**2 / (2.0 * hbar)

    carbon_pairs = [(i, j) for i in range(6) for j in range(i + 1, 6)]
    cc_derivatives = np.empty((len(modes), len(carbon_pairs)))
    for mode_index, mode in enumerate(modes):
        for pair_index, (left, right) in enumerate(carbon_pairs):
            vector = coordinates_bohr[right] - coordinates_bohr[left]
            unit = vector / np.linalg.norm(vector)
            cc_derivatives[mode_index, pair_index] = np.dot(
                unit, mode[right] - mode[left]
            )
    cc_shifts = cc_derivatives * delta_q_native[:, None] * BOHR_ANGSTROM
    return {
        "force_q_au": force_q_au,
        "delta_q_native": delta_q_native,
        "huang_rhys": huang_rhys,
        "cc_derivatives_per_sqrt_amu": cc_derivatives,
        "cc_rms_shift_angstrom": np.sqrt(np.mean(cc_shifts**2, axis=1)),
        "cc_max_shift_angstrom": np.max(np.abs(cc_shifts), axis=1),
    }


def main():
    sa_data = json.loads(SA_DATA.read_text(encoding="utf-8"))
    atoms = read_xyz(GEOMETRY)
    center = sa_data["rydberg_center_angstrom"]
    mol = gto.M(
        atom=atoms + [("X", tuple(center))],
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
    mc = mcscf.CASSCF(mf, 8, 4).density_fit()
    mc.fcisolver = fci.direct_spin0.FCI(mol)
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 100
    mc = mc.state_average_([1.0 / 7.0] * 7)
    mc.kernel(np.load(SA_MO))
    if not mc.converged:
        raise RuntimeError("Seven-state SA-CASSCF calculation did not converge")

    gradient_method = mc.nuc_grad_method()
    gradients = {"ground": gradient_method.kernel(state=0)}
    for label, root in ROOTS.items():
        gradients[label] = gradient_method.kernel(state=root)

    mode_data = np.load(NORMAL_MODES)
    modes = mode_data["normal_modes"]
    frequencies = mode_data["frequencies_cm1"]
    coordinates_bohr = mol.atom_coords()[:14]
    projections = {}
    for label in ROOTS:
        # The ghost center is fixed. Only physical-atom forces enter vibrations.
        force_change = -(gradients[label][:14] - gradients["ground"][:14])
        projections[label] = project_force(
            force_change, modes, frequencies, coordinates_bohr
        )

    fieldnames = [
        "state",
        "mode",
        "frequency_cm-1",
        "period_fs",
        "force_projection_Eh_per_bohr_sqrtamu",
        "abs_force_projection",
        "huang_rhys",
        "cc_rms_equilibrium_shift_angstrom",
        "cc_max_equilibrium_shift_angstrom",
        "cc_rms_max_wavepacket_excursion_angstrom",
    ]
    rows = []
    for label, values in projections.items():
        for mode_index, frequency in enumerate(frequencies):
            rows.append(
                {
                    "state": label,
                    "mode": mode_index + 1,
                    "frequency_cm-1": frequency,
                    "period_fs": mode_data["periods_fs"][mode_index],
                    "force_projection_Eh_per_bohr_sqrtamu": values["force_q_au"][mode_index],
                    "abs_force_projection": abs(values["force_q_au"][mode_index]),
                    "huang_rhys": values["huang_rhys"][mode_index],
                    "cc_rms_equilibrium_shift_angstrom": values["cc_rms_shift_angstrom"][mode_index],
                    "cc_max_equilibrium_shift_angstrom": values["cc_max_shift_angstrom"][mode_index],
                    "cc_rms_max_wavepacket_excursion_angstrom": 2.0
                    * values["cc_rms_shift_angstrom"][mode_index],
                }
            )
    with Path(f"{OUTPUT_PREFIX}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    np.savez(
        f"{OUTPUT_PREFIX}.npz",
        frequencies_cm1=frequencies,
        normal_modes=modes,
        ground_gradient_au=gradients["ground"],
        gradient_3px_au=gradients["3px"],
        gradient_3py_au=gradients["3py"],
        force_projection_3px_au=projections["3px"]["force_q_au"],
        force_projection_3py_au=projections["3py"]["force_q_au"],
        huang_rhys_3px=projections["3px"]["huang_rhys"],
        huang_rhys_3py=projections["3py"]["huang_rhys"],
        cc_rms_shift_3px_angstrom=projections["3px"]["cc_rms_shift_angstrom"],
        cc_rms_shift_3py_angstrom=projections["3py"]["cc_rms_shift_angstrom"],
    )
    summary = {
        "method": "seven-state equal-weight DF-SA-CASSCF(4,8)/aug-cc-pVDZ + Kaufmann 3s/3p",
        "geometry": str(GEOMETRY),
        "normal_modes": "B3LYP/aug-cc-pVDZ at the CASSCF geometry",
        "state_label_convention": {
            "source": "Yong/Ruddock CHD 200 nm papers",
            "x": "normal to the conjugated-carbon plane (out of plane)",
            "z": "molecular C2 axis",
            "y": "in-plane axis completing a right-handed x,y,z frame",
            "warning": "molecular-frame labels, not stored-XYZ Cartesian axes",
        },
        "root_assignments": ROOTS,
        "state_energies_hartree": np.asarray(mc.e_states).tolist(),
        "gradient_norms_Eh_per_bohr": {
            label: {
                "rms_physical_atoms": float(np.sqrt(np.mean(value[:14] ** 2))),
                "max_physical_atoms": float(np.max(np.abs(value[:14]))),
                "ghost_force_norm": float(np.linalg.norm(value[14])),
            }
            for label, value in gradients.items()
        },
        "top_modes": {},
    }
    for label, values in projections.items():
        summary["top_modes"][label] = {}
        metrics = {
            "abs_force": np.abs(values["force_q_au"]),
            "huang_rhys": values["huang_rhys"],
            "cc_rms_shift": values["cc_rms_shift_angstrom"],
        }
        for metric, numbers in metrics.items():
            order = np.argsort(numbers)[::-1][:10]
            summary["top_modes"][label][metric] = [
                {
                    "mode": int(index + 1),
                    "frequency_cm-1": float(frequencies[index]),
                    "value": float(numbers[index]),
                }
                for index in order
            ]
    Path(f"{OUTPUT_PREFIX}.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
