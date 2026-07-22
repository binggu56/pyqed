"""C2-constrained CASSCF(4,4)/aug-cc-pVDZ optimization of neutral CHD."""

from pathlib import Path
import sys

import numpy as np
from pyscf import gto, mcscf, scf


SOURCE_XYZ = Path("chd_casscf44_augccpvdz.xyz")
OUTPUT_XYZ = Path("chd_c2_casscf44_augccpvdz.xyz")
PAIR_MAP = ((0, 1), (2, 3), (4, 5), (6, 8), (7, 9), (10, 11), (12, 13))


def read_xyz(path):
    lines = path.read_text(encoding="utf-8").splitlines()[2:]
    symbols = [line.split()[0] for line in lines]
    coords = np.asarray([[float(x) for x in line.split()[1:4]] for line in lines])
    return symbols, coords


def enforce_c2(symbols, coords):
    """Average each atomic pair under C2 and put the C2 axis along z."""
    centered = coords - coords.mean(axis=0)
    result = np.zeros_like(centered)
    # The source geometry has its approximate C2 axis along y.
    rotation_y = np.diag([-1.0, 1.0, -1.0])
    for left, right in PAIR_MAP:
        averaged = 0.5 * (centered[left] + rotation_y @ centered[right])
        result[left] = averaged
        result[right] = rotation_y @ averaged
    # Map old y (the C2 axis) to the conventional z axis.
    result = result[:, [0, 2, 1]]
    return [(symbol, tuple(coord)) for symbol, coord in zip(symbols, result)]


def select_pi_orbitals(mf):
    mol = mf.mol
    overlap = mf.get_ovlp()
    labels = [label.split() for label in mol.ao_labels()]
    coords = mol.atom_coords(unit="Angstrom")[[2, 3, 4, 5]]
    _, _, vh = np.linalg.svd(coords - coords.mean(axis=0))
    normal = vh[-1]
    columns = []
    for atom_index in (2, 3, 4, 5):
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
                raise RuntimeError(f"Cannot identify {atom_index} C {component}")
            target[matches[0]] = coefficient
        columns.append(target)
    targets = np.column_stack(columns)
    values, vectors = np.linalg.eigh(targets.T @ overlap @ targets)
    projector = targets @ vectors @ np.diag(values ** -0.5)
    amplitudes = projector.T @ overlap @ mf.mo_coeff
    weights = np.sum(amplitudes**2, axis=0)
    occupied = np.flatnonzero(mf.mo_occ > 0)
    virtual = np.flatnonzero(mf.mo_occ == 0)
    pi_occ = occupied[np.argsort(weights[occupied])[-2:]]
    pi_vir = virtual[np.argsort(weights[virtual])[-2:]]
    return [int(i) for i in np.concatenate((pi_occ, pi_vir))]


def write_xyz(mol, path, comment):
    with path.open("w", encoding="utf-8") as handle:
        handle.write(f"{mol.natm}\n{comment}\n")
        for symbol, coord in zip(mol.elements, mol.atom_coords(unit="Angstrom")):
            handle.write(
                f"{symbol:2s} {coord[0]: .12f} {coord[1]: .12f} {coord[2]: .12f}\n"
            )


def symmetry_residual(coords):
    rotation_z = np.diag([-1.0, -1.0, 1.0])
    errors = []
    for left, right in PAIR_MAP:
        errors.append(coords[right] - rotation_z @ coords[left])
    errors = np.asarray(errors)
    return float(np.sqrt(np.mean(errors**2))), float(np.max(np.abs(errors)))


def make_molecule(atoms, output):
    return gto.M(
        atom=atoms,
        basis="aug-cc-pvdz",
        unit="Angstrom",
        charge=0,
        spin=0,
        symmetry="C2",
        verbose=4,
        output=output,
        max_memory=6000,
    )


def main():
    symbols, coords = read_xyz(SOURCE_XYZ)
    atoms = enforce_c2(symbols, coords)
    mol = make_molecule(atoms, "chd_c2_casscf44_augccpvdz_opt.log")
    print("Detected point group:", mol.topgroup, mol.groupname)

    mf = scf.RHF(mol).density_fit().run(conv_tol=1.0e-10)
    active = select_pi_orbitals(mf)
    mc = mcscf.CASSCF(mf, 4, 4).density_fit()
    mc.conv_tol = 1.0e-8
    mc.max_cycle_macro = 80
    mc.kernel(mc.sort_mo([index + 1 for index in active]))
    if not mc.converged:
        raise RuntimeError("Initial CASSCF did not converge")

    geometric_path = Path("/private/tmp/pyqed-geometric")
    if geometric_path.exists():
        sys.path.append(str(geometric_path))
    from pyscf.geomopt.geometric_solver import optimize

    mol_eq = optimize(
        mc,
        maxsteps=100,
        convergence_energy=1.0e-6,
        convergence_grms=3.0e-4,
        convergence_gmax=4.5e-4,
        convergence_drms=1.2e-3,
        convergence_dmax=1.8e-3,
    )

    mf_final = scf.RHF(mol_eq).density_fit().run(conv_tol=1.0e-10)
    active_final = select_pi_orbitals(mf_final)
    mc_final = mcscf.CASSCF(mf_final, 4, 4).density_fit()
    mc_final.conv_tol = 1.0e-8
    mc_final.max_cycle_macro = 80
    mc_final.kernel(mc_final.sort_mo([index + 1 for index in active_final]))
    gradient = mc_final.nuc_grad_method().kernel()
    final_coords = mol_eq.atom_coords(unit="Angstrom")
    sym_rms, sym_max = symmetry_residual(final_coords)
    comment = (
        "C2 DF-CASSCF(4,4)/aug-cc-pVDZ; "
        f"E={mc_final.e_tot:.12f} Eh; grad_rms={np.sqrt(np.mean(gradient**2)):.3e}; "
        f"grad_max={np.max(np.abs(gradient)):.3e} Eh/Bohr; "
        f"C2_rms={sym_rms:.3e}; C2_max={sym_max:.3e} Angstrom"
    )
    write_xyz(mol_eq, OUTPUT_XYZ, comment)
    print(comment)


if __name__ == "__main__":
    main()
