"""Classical calculators for :mod:`pyqed.md`."""

from math import erfc, sqrt

import numpy as np

from .neighborlist import (
    candidate_pairs as _candidate_pairs,
    minimum_image as _minimum_image,
    orthorhombic_lengths as _orthorhombic_lengths,
)

_PME_RECIPROCAL_CACHE = {}


def _parameter_array(value, natoms, name):
    array = np.asarray(value, dtype=float)
    if array.ndim == 0:
        return np.full(natoms, float(array))
    if array.shape != (natoms,):
        raise ValueError(f"{name} must be scalar or have shape ({natoms},).")
    return array


class LennardJones:
    """Pairwise Lennard-Jones calculator.

    Parameters
    ----------
    epsilon
        Pair well depth in the same energy unit used by the MD integrator.
    sigma
        Zero-crossing distance in the same length unit as atom positions.
    cutoff
        Optional pair cutoff. If omitted, all pairs are evaluated.
    energy_shift
        If true, subtract ``V(cutoff)`` from each included pair so the
        potential is continuous at the cutoff. Forces are not shifted.
    """

    def __init__(self, epsilon=1.0, sigma=1.0, cutoff=None, energy_shift=True):
        self.epsilon = float(epsilon)
        self.sigma = float(sigma)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.energy_shift = bool(energy_shift)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def calculate(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("LennardJones calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        forces = np.zeros_like(positions)
        energy = 0.0
        shift = 0.0
        if self.cutoff is not None and self.energy_shift:
            shift = self._pair_energy(self.cutoff)

        for i, j in _candidate_pairs(positions, atoms.get_cell(), atoms.get_pbc(), self.cutoff):
            rij = self._minimum_image(
                positions[i] - positions[j],
                atoms.get_cell(),
                atoms.get_pbc(),
            )
            r2 = float(np.dot(rij, rij))
            if r2 == 0.0:
                raise ValueError("Lennard-Jones pair distance is zero.")

            inv_r2 = 1.0 / r2
            sr2 = (self.sigma * self.sigma) * inv_r2
            sr6 = sr2 ** 3
            sr12 = sr6 * sr6
            energy += 4.0 * self.epsilon * (sr12 - sr6) - shift
            fij = 24.0 * self.epsilon * (2.0 * sr12 - sr6) * inv_r2 * rij
            forces[i] += fij
            forces[j] -= fij

        return energy, forces

    def _pair_energy(self, r):
        sr6 = (self.sigma / r) ** 6
        return 4.0 * self.epsilon * (sr6 * sr6 - sr6)

    @staticmethod
    def _minimum_image(vector, cell, pbc):
        return _minimum_image(vector, cell, pbc)


class Coulomb:
    """Pairwise Coulomb calculator with caller-controlled units.

    The energy is ``coulomb_constant * qi * qj / r``. Set
    ``coulomb_constant`` for the length/energy/charge unit convention used by
    the simulation.
    """

    def __init__(self, charges, coulomb_constant=1.0, cutoff=None):
        self.charges = np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.cutoff = None if cutoff is None else float(cutoff)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def calculate(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("Coulomb calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        if len(self.charges) != len(positions):
            raise ValueError("Number of charges must match number of atoms.")

        forces = np.zeros_like(positions)
        energy = _add_coulomb_pairs(
            positions,
            atoms.get_cell(),
            atoms.get_pbc(),
            forces,
            self.charges,
            self.coulomb_constant,
            self.cutoff,
        )
        return energy, forces


class EwaldCoulomb:
    """Direct 3D Ewald Coulomb calculator for neutral periodic systems."""

    def __init__(
        self,
        charges,
        coulomb_constant=1.0,
        alpha=0.35,
        real_cutoff=None,
        kmax=5,
    ):
        self.charges = np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.alpha = float(alpha)
        self.real_cutoff = None if real_cutoff is None else float(real_cutoff)
        if np.isscalar(kmax):
            self.kmax = np.array([int(kmax)] * 3, dtype=int)
        else:
            self.kmax = np.asarray(kmax, dtype=int)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def calculate(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("EwaldCoulomb calculator has no atoms.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if len(self.charges) != len(positions):
            raise ValueError("Number of charges must match number of atoms.")
        forces = np.zeros_like(positions)
        energy = _add_ewald_coulomb(
            positions,
            atoms.get_cell(),
            atoms.get_pbc(),
            forces,
            self.charges,
            self.coulomb_constant,
            self.alpha,
            self.real_cutoff,
            self.kmax,
        )
        return energy, forces


class PMECoulomb:
    """Minimal particle-mesh Ewald Coulomb calculator.

    This implementation uses cloud-in-cell charge assignment rather than the
    high-order B-splines used by production PME engines. It is intended as a
    compact, tested PME architecture for PyQED's in-repo MD engine.
    """

    def __init__(
        self,
        charges,
        coulomb_constant=1.0,
        alpha=0.35,
        real_cutoff=None,
        mesh=(16, 16, 16),
    ):
        self.charges = np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.alpha = float(alpha)
        self.real_cutoff = None if real_cutoff is None else float(real_cutoff)
        self.mesh = np.asarray(mesh, dtype=int)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def calculate(self, atoms=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("PMECoulomb calculator has no atoms.")
        positions = np.asarray(atoms.get_positions(), dtype=float)
        if len(self.charges) != len(positions):
            raise ValueError("Number of charges must match number of atoms.")
        forces = np.zeros_like(positions)
        energy = _add_pme_coulomb(
            positions,
            atoms.get_cell(),
            atoms.get_pbc(),
            forces,
            self.charges,
            self.coulomb_constant,
            self.alpha,
            self.real_cutoff,
            self.mesh,
        )
        return energy, forces


class MolecularMechanics:
    """Simple bonded molecular-mechanics calculator.

    Bonds are ``(i, j, k, r0)`` tuples with energy
    ``0.5 * k * (r - r0)**2``. Angles are ``(i, j, k, ktheta, theta0)``
    tuples where ``j`` is the central atom and ``theta0`` is in radians
    by default. Optional Lennard-Jones and Coulomb terms are pairwise
    nonbonded interactions.
    """

    def __init__(
        self,
        bonds=None,
        angles=None,
        torsions=None,
        angle_unit="radian",
        torsion_unit="radian",
        charges=None,
        coulomb_constant=1.0,
        coulomb_method="cutoff",
        coulomb_cutoff=None,
        ewald_alpha=0.35,
        ewald_kmax=5,
        pme_mesh=(16, 16, 16),
        lj_epsilon=None,
        lj_sigma=None,
        lj_cutoff=None,
        lj_energy_shift=True,
        exclude_bonded=True,
        exclude_angles=True,
        nonbonded_exclusions=None,
        lj_exclusions=None,
        coulomb_exclusions=None,
    ):
        self.bonds = [
            (int(i), int(j), float(k), float(r0))
            for i, j, k, r0 in (bonds or [])
        ]
        self.angles = []
        for i, j, k, ktheta, theta0 in (angles or []):
            theta0 = float(theta0)
            if angle_unit.lower() in {"degree", "degrees", "deg"}:
                theta0 = np.deg2rad(theta0)
            elif angle_unit.lower() not in {"radian", "radians", "rad"}:
                raise ValueError("angle_unit must be 'radian' or 'degree'.")
            self.angles.append((int(i), int(j), int(k), float(ktheta), theta0))
        self.torsions = []
        for i, j, k, l, barrier, periodicity, phase in (torsions or []):
            phase = float(phase)
            if torsion_unit.lower() in {"degree", "degrees", "deg"}:
                phase = np.deg2rad(phase)
            elif torsion_unit.lower() not in {"radian", "radians", "rad"}:
                raise ValueError("torsion_unit must be 'radian' or 'degree'.")
            self.torsions.append(
                (int(i), int(j), int(k), int(l), float(barrier), int(periodicity), phase)
            )
        self.charges = None if charges is None else np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.coulomb_method = coulomb_method.lower()
        if self.coulomb_method not in {"cutoff", "ewald", "pme"}:
            raise ValueError("coulomb_method must be 'cutoff', 'ewald', or 'pme'.")
        self.coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
        self.ewald_alpha = float(ewald_alpha)
        if np.isscalar(ewald_kmax):
            self.ewald_kmax = np.array([int(ewald_kmax)] * 3, dtype=int)
        else:
            self.ewald_kmax = np.asarray(ewald_kmax, dtype=int)
        self.pme_mesh = np.asarray(pme_mesh, dtype=int)
        self.lj_epsilon = None if lj_epsilon is None else np.asarray(lj_epsilon, dtype=float)
        self.lj_sigma = None if lj_sigma is None else np.asarray(lj_sigma, dtype=float)
        self.lj_cutoff = None if lj_cutoff is None else float(lj_cutoff)
        self.lj_energy_shift = bool(lj_energy_shift)
        self.exclude_bonded = bool(exclude_bonded)
        self.exclude_angles = bool(exclude_angles)
        self.nonbonded_exclusions = _pair_set(nonbonded_exclusions)
        self.lj_exclusions = _pair_set(lj_exclusions)
        self.coulomb_exclusions = _pair_set(coulomb_exclusions)
        self._bonded_pairs = {tuple(sorted((i, j))) for i, j, _, _ in self.bonds}
        self._angle_pairs = {tuple(sorted((i, k))) for i, _, k, _, _ in self.angles}
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def calculate(self, atoms=None, extra_lj_exclusions=None, extra_coulomb_exclusions=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MolecularMechanics calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        forces = np.zeros_like(positions)
        energy = self._add_bonds(positions, cell, pbc, forces)
        energy += self._add_angles(positions, cell, pbc, forces)
        energy += self._add_torsions(positions, cell, pbc, forces)
        energy += self._add_nonbonded(
            atoms,
            positions,
            cell,
            pbc,
            forces,
            extra_lj_exclusions=extra_lj_exclusions,
            extra_coulomb_exclusions=extra_coulomb_exclusions,
        )
        return energy, forces

    def _add_bonds(self, positions, cell, pbc, forces):
        energy = 0.0
        for i, j, force_constant, equilibrium in self.bonds:
            rij = _minimum_image(positions[i] - positions[j], cell, pbc)
            distance = float(np.linalg.norm(rij))
            if distance == 0.0:
                raise ValueError("Bond distance is zero.")
            stretch = distance - equilibrium
            energy += 0.5 * force_constant * stretch * stretch
            fij = -force_constant * stretch * rij / distance
            forces[i] += fij
            forces[j] -= fij
        return energy

    def _add_torsions(self, positions, cell, pbc, forces):
        energy = 0.0
        delta = 1e-6
        for i, j, k, l, barrier, periodicity, phase in self.torsions:
            phi = _dihedral_angle(positions, cell, pbc, i, j, k, l)
            argument = periodicity * phi - phase
            energy += barrier * (1.0 + np.cos(argument))
            denergy_dphi = -barrier * periodicity * np.sin(argument)
            for atom_index in (i, j, k, l):
                for axis in range(3):
                    displaced = positions.copy()
                    displaced[atom_index, axis] += delta
                    phi_plus = _dihedral_angle(displaced, cell, pbc, i, j, k, l)
                    displaced[atom_index, axis] -= 2.0 * delta
                    phi_minus = _dihedral_angle(displaced, cell, pbc, i, j, k, l)
                    dphi = _angle_difference(phi_plus, phi_minus) / (2.0 * delta)
                    forces[atom_index, axis] += -denergy_dphi * dphi
        return energy

    def _torsion_energy(self, positions, cell, pbc):
        energy = 0.0
        for i, j, k, l, barrier, periodicity, phase in self.torsions:
            phi = _dihedral_angle(positions, cell, pbc, i, j, k, l)
            energy += barrier * (1.0 + np.cos(periodicity * phi - phase))
        return energy

    def _add_angles(self, positions, cell, pbc, forces):
        energy = 0.0
        for i, j, k, force_constant, equilibrium in self.angles:
            rij = _minimum_image(positions[i] - positions[j], cell, pbc)
            rkj = _minimum_image(positions[k] - positions[j], cell, pbc)
            rij_norm = float(np.linalg.norm(rij))
            rkj_norm = float(np.linalg.norm(rkj))
            if rij_norm == 0.0 or rkj_norm == 0.0:
                raise ValueError("Angle contains a zero-length bond.")

            cos_theta = float(np.dot(rij, rkj) / (rij_norm * rkj_norm))
            cos_theta = np.clip(cos_theta, -1.0, 1.0)
            theta = float(np.arccos(cos_theta))
            sin_theta = np.sqrt(max(1.0 - cos_theta * cos_theta, 1e-24))
            bend = theta - equilibrium
            energy += 0.5 * force_constant * bend * bend

            prefactor = force_constant * bend / sin_theta
            dtheta_drij = (
                cos_theta * rij / (rij_norm * rij_norm)
                - rkj / (rij_norm * rkj_norm)
            )
            dtheta_drkj = (
                cos_theta * rkj / (rkj_norm * rkj_norm)
                - rij / (rij_norm * rkj_norm)
            )
            force_i = -prefactor * dtheta_drij
            force_k = -prefactor * dtheta_drkj
            force_j = -(force_i + force_k)
            forces[i] += force_i
            forces[j] += force_j
            forces[k] += force_k
        return energy

    def _add_nonbonded(
        self,
        atoms,
        positions,
        cell,
        pbc,
        forces,
        extra_lj_exclusions=None,
        extra_coulomb_exclusions=None,
    ):
        charges = self._charges(atoms)
        if charges is None and self.lj_epsilon is None:
            return 0.0
        if self.lj_epsilon is not None and self.lj_sigma is None:
            raise ValueError("lj_sigma is required when lj_epsilon is set.")

        lj_exclusions = self._nonbonded_exclusions("lj", extra_lj_exclusions)
        coulomb_exclusions = self._nonbonded_exclusions("coulomb", extra_coulomb_exclusions)
        energy = 0.0
        if self.lj_epsilon is not None:
            energy += _add_lennard_jones_pairs(
                positions,
                cell,
                pbc,
                forces,
                self.lj_epsilon,
                self.lj_sigma,
                self.lj_cutoff,
                self.lj_energy_shift,
                lj_exclusions,
            )
        if charges is not None:
            if self.coulomb_method == "ewald":
                energy += _add_ewald_coulomb(
                    positions,
                    cell,
                    pbc,
                    forces,
                    charges,
                    self.coulomb_constant,
                    self.ewald_alpha,
                    self.coulomb_cutoff,
                    self.ewald_kmax,
                )
                if coulomb_exclusions is not None:
                    energy -= _add_coulomb_pairs(
                        positions,
                        cell,
                        pbc,
                        forces,
                        charges,
                        self.coulomb_constant,
                        None,
                        None,
                        only_pairs=coulomb_exclusions,
                    )
            elif self.coulomb_method == "pme":
                energy += _add_pme_coulomb(
                    positions,
                    cell,
                    pbc,
                    forces,
                    charges,
                    self.coulomb_constant,
                    self.ewald_alpha,
                    self.coulomb_cutoff,
                    self.pme_mesh,
                )
                if coulomb_exclusions is not None:
                    energy -= _add_coulomb_pairs(
                        positions,
                        cell,
                        pbc,
                        forces,
                        charges,
                        self.coulomb_constant,
                        None,
                        None,
                        only_pairs=coulomb_exclusions,
                    )
            else:
                energy += _add_coulomb_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    charges,
                    self.coulomb_constant,
                    self.coulomb_cutoff,
                    coulomb_exclusions,
                )
        return energy

    def _charges(self, atoms):
        if self.charges is not None:
            charges = self.charges
        elif hasattr(atoms, "arrays") and "charges" in atoms.arrays:
            charges = atoms.arrays["charges"]
        else:
            return None
        if len(charges) != len(atoms):
            raise ValueError("Number of charges must match number of atoms.")
        return np.asarray(charges, dtype=float)

    def _nonbonded_exclusions(self, kind, extra_exclusions=None):
        exclusions = set()
        if self.exclude_bonded:
            exclusions.update(self._bonded_pairs)
        if self.exclude_angles:
            exclusions.update(self._angle_pairs)
        exclusions.update(self.nonbonded_exclusions)
        if kind == "lj":
            exclusions.update(self.lj_exclusions)
        elif kind == "coulomb":
            exclusions.update(self.coulomb_exclusions)
        else:
            raise ValueError("kind must be 'lj' or 'coulomb'.")
        exclusions.update(_pair_set(extra_exclusions))
        return exclusions or None


class MM(MolecularMechanics):
    """Short public name for :class:`MolecularMechanics`."""


def _pair_set(pairs):
    if pairs is None:
        return set()
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def _dihedral_angle(positions, cell, pbc, i, j, k, l):
    rij = _minimum_image(positions[j] - positions[i], cell, pbc)
    rjk = _minimum_image(positions[k] - positions[j], cell, pbc)
    rkl = _minimum_image(positions[l] - positions[k], cell, pbc)
    p0 = np.zeros(3)
    p1 = p0 + rij
    p2 = p1 + rjk
    p3 = p2 + rkl

    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0.0:
        raise ValueError("Torsion contains a zero-length central bond.")
    b1 /= b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm == 0.0 or w_norm == 0.0:
        raise ValueError("Torsion contains collinear or zero-length bonds.")
    v /= v_norm
    w /= w_norm
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _angle_difference(a, b):
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _add_lennard_jones_pairs(
    positions,
    cell,
    pbc,
    forces,
    epsilon,
    sigma,
    cutoff=None,
    energy_shift=True,
    exclusions=None,
):
    energy = 0.0
    natoms = len(positions)
    epsilon = _parameter_array(epsilon, natoms, "epsilon")
    sigma = _parameter_array(sigma, natoms, "sigma")

    for i, j in _candidate_pairs(positions, cell, pbc, cutoff, exclusions):
            epsilon_ij = np.sqrt(epsilon[i] * epsilon[j])
            if epsilon_ij == 0.0:
                continue
            sigma_ij = 0.5 * (sigma[i] + sigma[j])
            shift = 0.0
            if cutoff is not None and energy_shift:
                sr6_cutoff = (sigma_ij / cutoff) ** 6
                shift = 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
            rij = _minimum_image(positions[i] - positions[j], cell, pbc)
            r2 = float(np.dot(rij, rij))
            if r2 == 0.0:
                raise ValueError("Lennard-Jones pair distance is zero.")

            inv_r2 = 1.0 / r2
            sr2 = (sigma_ij * sigma_ij) * inv_r2
            sr6 = sr2 ** 3
            sr12 = sr6 * sr6
            energy += 4.0 * epsilon_ij * (sr12 - sr6) - shift
            fij = 24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2 * rij
            forces[i] += fij
            forces[j] -= fij
    return energy


def _add_coulomb_pairs(
    positions,
    cell,
    pbc,
    forces,
    charges,
    coulomb_constant,
    cutoff=None,
    exclusions=None,
    only_pairs=None,
):
    energy = 0.0
    pairs = only_pairs
    if pairs is None:
        pairs = _candidate_pairs(positions, cell, pbc, cutoff, exclusions)
    for i, j in pairs:
            charge_product = charges[i] * charges[j]
            if charge_product == 0.0:
                continue
            rij = _minimum_image(positions[i] - positions[j], cell, pbc)
            r2 = float(np.dot(rij, rij))
            if r2 == 0.0:
                raise ValueError("Coulomb pair distance is zero.")

            distance = np.sqrt(r2)
            prefactor = coulomb_constant * charge_product
            energy += prefactor / distance
            fij = prefactor * rij / (distance * r2)
            forces[i] += fij
            forces[j] -= fij
    return energy


def _add_ewald_coulomb(
    positions,
    cell,
    pbc,
    forces,
    charges,
    coulomb_constant,
    alpha,
    real_cutoff,
    kmax,
):
    pbc = np.asarray(pbc, dtype=bool)
    if not np.all(pbc):
        raise ValueError("Ewald Coulomb requires 3D periodic boundary conditions.")
    lengths = _orthorhombic_lengths(cell)
    if lengths is None:
        raise ValueError("Ewald Coulomb currently requires an orthorhombic cell.")
    if abs(float(np.sum(charges))) > 1e-10:
        raise ValueError("Ewald Coulomb requires a neutral unit cell.")
    if alpha <= 0.0:
        raise ValueError("Ewald alpha must be positive.")
    kmax = np.asarray(kmax, dtype=int)
    if kmax.shape != (3,) or np.any(kmax < 0):
        raise ValueError("Ewald kmax must be a non-negative scalar or length-3 sequence.")

    volume = float(np.prod(lengths))
    cutoff = 0.5 * float(np.min(lengths)) if real_cutoff is None else float(real_cutoff)
    energy = _add_ewald_real(positions, lengths, forces, charges, coulomb_constant, alpha, cutoff)
    energy += _add_ewald_reciprocal(
        positions, lengths, volume, forces, charges, coulomb_constant, alpha, kmax
    )
    energy -= coulomb_constant * alpha / sqrt(np.pi) * float(np.dot(charges, charges))
    return energy


def _add_pme_coulomb(
    positions,
    cell,
    pbc,
    forces,
    charges,
    coulomb_constant,
    alpha,
    real_cutoff,
    mesh,
):
    pbc = np.asarray(pbc, dtype=bool)
    if not np.all(pbc):
        raise ValueError("PME Coulomb requires 3D periodic boundary conditions.")
    lengths = _orthorhombic_lengths(cell)
    if lengths is None:
        raise ValueError("PME Coulomb currently requires an orthorhombic cell.")
    if abs(float(np.sum(charges))) > 1e-10:
        raise ValueError("PME Coulomb requires a neutral unit cell.")
    if alpha <= 0.0:
        raise ValueError("PME alpha must be positive.")
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 4):
        raise ValueError("PME mesh must be a length-3 sequence with values >= 4.")

    cutoff = 0.5 * float(np.min(lengths)) if real_cutoff is None else float(real_cutoff)
    energy = _add_ewald_real(positions, lengths, forces, charges, coulomb_constant, alpha, cutoff)
    energy += _add_pme_reciprocal(
        positions, lengths, forces, charges, coulomb_constant, alpha, mesh
    )
    energy -= coulomb_constant * alpha / sqrt(np.pi) * float(np.dot(charges, charges))
    return energy


def _add_pme_reciprocal(positions, lengths, forces, charges, coulomb_constant, alpha, mesh):
    potential_grid = pme_reciprocal_potential_grid(
        positions,
        charges,
        np.diag(lengths),
        (True, True, True),
        coulomb_constant=coulomb_constant,
        alpha=alpha,
        mesh=mesh,
    )
    charge_grid = _assign_charges_cic(positions, charges, lengths, mesh)
    kx, ky, kz, influence, grid_size = _pme_reciprocal_data(lengths, mesh, coulomb_constant, alpha)
    rho_hat = np.fft.fftn(charge_grid)
    field_grids = [
        np.fft.ifftn(grid_size * (-1j * axis_grid) * influence * rho_hat).real
        for axis_grid in (kx, ky, kz)
    ]

    energy = 0.5 * float(np.sum(charge_grid * potential_grid))
    electric_field = np.column_stack(
        [
            _interpolate_cic(positions, field_grid, lengths, mesh)
            for field_grid in field_grids
        ]
    )
    forces += charges[:, None] * electric_field
    return energy


def pme_reciprocal_potential_grid(
    positions,
    charges,
    cell,
    pbc=True,
    coulomb_constant=1.0,
    alpha=0.35,
    mesh=(16, 16, 16),
):
    """Return the smooth reciprocal-space PME potential on the PME mesh.

    The returned grid uses the same CIC assignment and reciprocal influence
    function as :class:`PMECoulomb`.  The zero Fourier mode is omitted, so the
    potential follows the usual neutral-cell PME convention.
    """
    pbc = np.asarray((pbc, pbc, pbc) if isinstance(pbc, bool) else pbc, dtype=bool)
    if not np.all(pbc):
        raise ValueError("PME reciprocal potential requires 3D periodic boundary conditions.")
    lengths = _orthorhombic_lengths(cell)
    if lengths is None:
        raise ValueError("PME reciprocal potential currently requires an orthorhombic cell.")
    charges = np.asarray(charges, dtype=float)
    if abs(float(np.sum(charges))) > 1e-10:
        raise ValueError("PME reciprocal potential requires a neutral unit cell.")
    if alpha <= 0.0:
        raise ValueError("PME alpha must be positive.")
    mesh = np.asarray(mesh, dtype=int)
    if mesh.shape != (3,) or np.any(mesh < 4):
        raise ValueError("PME mesh must be a length-3 sequence with values >= 4.")

    charge_grid = _assign_charges_cic(positions, charges, lengths, mesh)
    rho_hat = np.fft.fftn(charge_grid)
    _, _, _, influence, grid_size = _pme_reciprocal_data(
        lengths,
        mesh,
        coulomb_constant,
        alpha,
    )
    return np.fft.ifftn(grid_size * influence * rho_hat).real


def pme_reciprocal_potential(
    positions,
    charges,
    points,
    cell,
    pbc=True,
    coulomb_constant=1.0,
    alpha=0.35,
    mesh=(16, 16, 16),
):
    """Interpolate the smooth reciprocal-space PME potential to ``points``."""
    lengths = _orthorhombic_lengths(cell)
    potential_grid = pme_reciprocal_potential_grid(
        positions,
        charges,
        cell,
        pbc,
        coulomb_constant=coulomb_constant,
        alpha=alpha,
        mesh=mesh,
    )
    return _interpolate_cic(points, potential_grid, lengths, np.asarray(mesh, dtype=int))


def _pme_reciprocal_data(lengths, mesh, coulomb_constant, alpha):
    lengths = np.asarray(lengths, dtype=float)
    mesh = np.asarray(mesh, dtype=int)
    key = (
        tuple(np.round(lengths, 12)),
        tuple(mesh.tolist()),
        float(coulomb_constant),
        float(alpha),
    )
    cached = _PME_RECIPROCAL_CACHE.get(key)
    if cached is not None:
        return cached

    volume = float(np.prod(lengths))
    grid_size = int(np.prod(mesh))
    k_axes = [
        2.0 * np.pi * np.fft.fftfreq(mesh[axis], d=lengths[axis] / mesh[axis])
        for axis in range(3)
    ]
    kx, ky, kz = np.meshgrid(k_axes[0], k_axes[1], k_axes[2], indexing="ij")
    k2 = kx * kx + ky * ky + kz * kz
    influence = np.zeros_like(k2)
    mask = k2 > 0.0
    influence[mask] = 4.0 * np.pi * coulomb_constant / volume
    influence[mask] *= np.exp(-k2[mask] / (4.0 * alpha * alpha)) / k2[mask]
    cached = (kx, ky, kz, influence, grid_size)
    _PME_RECIPROCAL_CACHE[key] = cached
    return cached


def _assign_charges_cic(positions, charges, lengths, mesh):
    grid = np.zeros(tuple(mesh), dtype=float)
    for position, charge in zip(positions, charges):
        scaled = np.mod(position / lengths, 1.0) * mesh
        base = np.floor(scaled).astype(int)
        frac = scaled - base
        for dx in (0, 1):
            wx = (1.0 - frac[0]) if dx == 0 else frac[0]
            ix = (base[0] + dx) % mesh[0]
            for dy in (0, 1):
                wy = (1.0 - frac[1]) if dy == 0 else frac[1]
                iy = (base[1] + dy) % mesh[1]
                for dz in (0, 1):
                    wz = (1.0 - frac[2]) if dz == 0 else frac[2]
                    iz = (base[2] + dz) % mesh[2]
                    grid[ix, iy, iz] += charge * wx * wy * wz
    return grid


def _interpolate_cic(positions, grid, lengths, mesh):
    values = np.zeros(len(positions), dtype=float)
    for atom_index, position in enumerate(positions):
        scaled = np.mod(position / lengths, 1.0) * mesh
        base = np.floor(scaled).astype(int)
        frac = scaled - base
        value = 0.0
        for dx in (0, 1):
            wx = (1.0 - frac[0]) if dx == 0 else frac[0]
            ix = (base[0] + dx) % mesh[0]
            for dy in (0, 1):
                wy = (1.0 - frac[1]) if dy == 0 else frac[1]
                iy = (base[1] + dy) % mesh[1]
                for dz in (0, 1):
                    wz = (1.0 - frac[2]) if dz == 0 else frac[2]
                    iz = (base[2] + dz) % mesh[2]
                    value += wx * wy * wz * grid[ix, iy, iz]
        values[atom_index] = value
    return values


def _add_ewald_real(positions, lengths, forces, charges, coulomb_constant, alpha, cutoff):
    if cutoff <= 0.5 * float(np.min(lengths)):
        return _add_ewald_real_minimum_image(
            positions, lengths, forces, charges, coulomb_constant, alpha, cutoff
        )

    energy = 0.0
    natoms = len(positions)
    image_ranges = [range(-int(np.ceil(cutoff / length)), int(np.ceil(cutoff / length)) + 1) for length in lengths]
    cutoff2 = cutoff * cutoff
    for nx in image_ranges[0]:
        for ny in image_ranges[1]:
            for nz in image_ranges[2]:
                shift = np.array([nx, ny, nz], dtype=float) * lengths
                for i in range(natoms):
                    for j in range(i + 1, natoms):
                        rij = positions[i] - positions[j] + shift
                        r2 = float(np.dot(rij, rij))
                        if r2 == 0.0 or r2 > cutoff2:
                            continue
                        distance = sqrt(r2)
                        charge_product = charges[i] * charges[j]
                        prefactor = coulomb_constant * charge_product
                        ar = alpha * distance
                        energy += prefactor * erfc(ar) / distance
                        force_scalar = prefactor * (
                            erfc(ar) / (distance * r2)
                            + 2.0 * alpha * np.exp(-ar * ar) / (sqrt(np.pi) * r2)
                        )
                        fij = force_scalar * rij
                        forces[i] += fij
                        forces[j] -= fij
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                for i in range(natoms):
                    rij = shift
                    r2 = float(np.dot(rij, rij))
                    if r2 == 0.0 or r2 > cutoff2:
                        continue
                    distance = sqrt(r2)
                    energy += 0.5 * coulomb_constant * charges[i] * charges[i] * erfc(alpha * distance) / distance
    return energy


def _add_ewald_real_minimum_image(
    positions, lengths, forces, charges, coulomb_constant, alpha, cutoff
):
    energy = 0.0
    cell = np.diag(lengths)
    pbc = np.ones(3, dtype=bool)
    for i, j in _candidate_pairs(positions, cell, pbc, cutoff):
        rij = _minimum_image(positions[i] - positions[j], cell, pbc)
        r2 = float(np.dot(rij, rij))
        if r2 == 0.0:
            continue
        distance = sqrt(r2)
        charge_product = charges[i] * charges[j]
        prefactor = coulomb_constant * charge_product
        ar = alpha * distance
        energy += prefactor * erfc(ar) / distance
        force_scalar = prefactor * (
            erfc(ar) / (distance * r2)
            + 2.0 * alpha * np.exp(-ar * ar) / (sqrt(np.pi) * r2)
        )
        fij = force_scalar * rij
        forces[i] += fij
        forces[j] -= fij
    return energy


def _add_ewald_reciprocal(
    positions,
    lengths,
    volume,
    forces,
    charges,
    coulomb_constant,
    alpha,
    kmax,
):
    energy = 0.0
    coefficient = coulomb_constant * 2.0 * np.pi / volume
    reciprocal_axes = 2.0 * np.pi / lengths
    for mx in range(-kmax[0], kmax[0] + 1):
        for my in range(-kmax[1], kmax[1] + 1):
            for mz in range(-kmax[2], kmax[2] + 1):
                if mx == 0 and my == 0 and mz == 0:
                    continue
                kvec = np.array([mx, my, mz], dtype=float) * reciprocal_axes
                k2 = float(np.dot(kvec, kvec))
                weight = np.exp(-k2 / (4.0 * alpha * alpha)) / k2
                phases = positions @ kvec
                cos_phase = np.cos(phases)
                sin_phase = np.sin(phases)
                structure_cos = float(np.dot(charges, cos_phase))
                structure_sin = float(np.dot(charges, sin_phase))
                energy += coefficient * weight * (
                    structure_cos * structure_cos + structure_sin * structure_sin
                )
                force_coefficient = 2.0 * coefficient * weight
                phase_force = structure_cos * sin_phase - structure_sin * cos_phase
                forces += (force_coefficient * charges * phase_force)[:, None] * kvec
    return energy
