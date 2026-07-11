"""Classical calculators for :mod:`pyqed.md`."""

from math import erfc, sqrt

import numpy as np
try:
    from scipy.special import erfc as _array_erfc
except Exception:  # pragma: no cover - SciPy is available in the normal test env.
    _array_erfc = np.vectorize(erfc)

from .neighborlist import (
    candidate_pair_displacement_arrays as _candidate_pair_displacement_arrays,
    candidate_pair_displacements as _candidate_pair_displacements,
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


def _accumulate_pair_virial(virial, rij, fij):
    if virial is not None:
        virial += np.outer(rij, fij)


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

    def get_virial(self, atoms=None):
        virial = np.zeros((3, 3), dtype=float)
        self.calculate(atoms, virial=virial)
        return virial

    def calculate(self, atoms=None, virial=None):
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
            _accumulate_pair_virial(virial, rij, fij)

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

    def __init__(self, charges, coulomb_constant=1.0, cutoff=None, energy_shift=False):
        self.charges = np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
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

    def get_virial(self, atoms=None):
        virial = np.zeros((3, 3), dtype=float)
        self.calculate(atoms, virial=virial)
        return virial

    def calculate(self, atoms=None, virial=None):
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
            energy_shift=self.energy_shift,
            virial=virial,
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

    def get_virial(self, atoms=None):
        virial = np.zeros((3, 3), dtype=float)
        self.calculate(atoms, virial=virial)
        return virial

    def calculate(self, atoms=None, virial=None):
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
            virial=virial,
        )
        return energy, forces


class PMECoulomb:
    """Minimal particle-mesh Ewald Coulomb calculator.

    This implementation uses B-spline charge assignment and reciprocal
    influence deconvolution.  It is intended as a compact, tested PME
    architecture for PyQED's in-repo MD engine.
    """

    def __init__(
        self,
        charges,
        coulomb_constant=1.0,
        alpha=0.35,
        real_cutoff=None,
        mesh=(16, 16, 16),
        order=4,
    ):
        self.charges = np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.alpha = float(alpha)
        self.real_cutoff = None if real_cutoff is None else float(real_cutoff)
        self.mesh = np.asarray(mesh, dtype=int)
        self.order = int(order)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def get_virial(self, atoms=None):
        virial = np.zeros((3, 3), dtype=float)
        self.calculate(atoms, virial=virial)
        return virial

    def calculate(self, atoms=None, virial=None):
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
            self.order,
            virial=virial,
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
        impropers=None,
        cmaps=None,
        cmap_grids=None,
        angle_unit="radian",
        torsion_unit="radian",
        improper_unit="radian",
        charges=None,
        coulomb_constant=1.0,
        coulomb_method="cutoff",
        coulomb_cutoff=None,
        coulomb_energy_shift=False,
        coulomb_reaction_field_dielectric=None,
        ewald_alpha=0.35,
        ewald_kmax=5,
        pme_mesh=(16, 16, 16),
        pme_order=4,
        lj_epsilon=None,
        lj_sigma=None,
        lj_cutoff=None,
        lj_switch_on=None,
        lj_energy_shift=True,
        atom_types=None,
        lj_pair_overrides=None,
        lj_pair_parameters=None,
        coulomb_pair_parameters=None,
        exclude_bonded=True,
        exclude_angles=True,
        nonbonded_exclusions=None,
        lj_exclusions=None,
        coulomb_exclusions=None,
        lj_pair_scales=None,
        coulomb_pair_scales=None,
        nonbonded_skin=0.0,
    ):
        self.bonds = [
            (int(i), int(j), float(k), float(r0))
            for i, j, k, r0 in (bonds or [])
        ]
        (
            self._bond_indices,
            self._bond_force_constants,
            self._bond_equilibria,
        ) = _bond_arrays(self.bonds)
        self.angles = []
        for i, j, k, ktheta, theta0 in (angles or []):
            theta0 = float(theta0)
            if angle_unit.lower() in {"degree", "degrees", "deg"}:
                theta0 = np.deg2rad(theta0)
            elif angle_unit.lower() not in {"radian", "radians", "rad"}:
                raise ValueError("angle_unit must be 'radian' or 'degree'.")
            self.angles.append((int(i), int(j), int(k), float(ktheta), theta0))
        (
            self._angle_indices,
            self._angle_force_constants,
            self._angle_equilibria,
        ) = _angle_arrays(self.angles)
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
        (
            self._torsion_indices,
            self._torsion_barriers,
            self._torsion_periodicities,
            self._torsion_phases,
        ) = _torsion_arrays(self.torsions)
        self.impropers = []
        for i, j, k, l, force_constant, phase in (impropers or []):
            phase = float(phase)
            if improper_unit.lower() in {"degree", "degrees", "deg"}:
                phase = np.deg2rad(phase)
            elif improper_unit.lower() not in {"radian", "radians", "rad"}:
                raise ValueError("improper_unit must be 'radian' or 'degree'.")
            self.impropers.append((int(i), int(j), int(k), int(l), float(force_constant), phase))
        self.cmaps = [
            (int(map_index), tuple(int(atom) for atom in atoms))
            for map_index, atoms in (cmaps or [])
        ]
        self.cmap_grids = [
            (int(size), np.asarray(values, dtype=float).reshape(int(size), int(size)))
            for size, values in (cmap_grids or [])
        ]
        self._cmap_coefficients = [
            _periodic_bicubic_coefficients(values)
            for _size, values in self.cmap_grids
        ]
        self.charges = None if charges is None else np.asarray(charges, dtype=float)
        self.coulomb_constant = float(coulomb_constant)
        self.coulomb_method = coulomb_method.lower()
        if self.coulomb_method not in {"cutoff", "ewald", "pme"}:
            raise ValueError("coulomb_method must be 'cutoff', 'ewald', or 'pme'.")
        self.coulomb_cutoff = None if coulomb_cutoff is None else float(coulomb_cutoff)
        self.coulomb_energy_shift = bool(coulomb_energy_shift)
        self.coulomb_reaction_field_dielectric = (
            None if coulomb_reaction_field_dielectric is None else float(coulomb_reaction_field_dielectric)
        )
        if (
            self.coulomb_reaction_field_dielectric is not None
            and self.coulomb_reaction_field_dielectric <= 0.0
        ):
            raise ValueError("coulomb_reaction_field_dielectric must be positive.")
        self.ewald_alpha = float(ewald_alpha)
        if np.isscalar(ewald_kmax):
            self.ewald_kmax = np.array([int(ewald_kmax)] * 3, dtype=int)
        else:
            self.ewald_kmax = np.asarray(ewald_kmax, dtype=int)
        self.pme_mesh = np.asarray(pme_mesh, dtype=int)
        self.pme_order = int(pme_order)
        self.lj_epsilon = None if lj_epsilon is None else np.asarray(lj_epsilon, dtype=float)
        self.lj_sigma = None if lj_sigma is None else np.asarray(lj_sigma, dtype=float)
        self.lj_cutoff = None if lj_cutoff is None else float(lj_cutoff)
        self.lj_switch_on = None if lj_switch_on is None else float(lj_switch_on)
        self.lj_energy_shift = bool(lj_energy_shift)
        self.atom_types = None if atom_types is None else np.asarray(atom_types, dtype=str)
        self.lj_pair_overrides = _lj_pair_override_dict(lj_pair_overrides)
        self._lj_pair_override_lookup = _lj_pair_override_lookup(
            self.atom_types,
            self.lj_pair_overrides,
        )
        self._lj_type_pair_parameter_lookup = _lj_type_pair_parameter_lookup(
            self.atom_types,
            self.lj_epsilon,
            self.lj_sigma,
            self.lj_pair_overrides,
        )
        self.lj_pair_parameters = _pair_lj_parameter_dict(lj_pair_parameters)
        self.coulomb_pair_parameters = _pair_float_dict(coulomb_pair_parameters)
        self.exclude_bonded = bool(exclude_bonded)
        self.exclude_angles = bool(exclude_angles)
        self.nonbonded_exclusions = _pair_set(nonbonded_exclusions)
        self.lj_exclusions = _pair_set(lj_exclusions)
        self.coulomb_exclusions = _pair_set(coulomb_exclusions)
        self.lj_pair_scales = _pair_scale_dict(lj_pair_scales)
        self.coulomb_pair_scales = _pair_scale_dict(coulomb_pair_scales)
        self.nonbonded_skin = float(nonbonded_skin)
        if self.nonbonded_skin < 0.0:
            raise ValueError("nonbonded_skin must be non-negative.")
        self._bonded_pairs = {tuple(sorted((i, j))) for i, j, _, _ in self.bonds}
        self._angle_pairs = {tuple(sorted((i, k))) for i, _, k, _, _ in self.angles}
        self._shared_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self._lj_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self._coulomb_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self.atoms = None

    def set_atoms(self, atoms):
        self.atoms = atoms

    def get_potential_energy(self, atoms=None):
        energy, _ = self.calculate(atoms)
        return energy

    def get_forces(self, atoms=None):
        _, forces = self.calculate(atoms)
        return forces

    def get_virial(self, atoms=None):
        """Return the configurational virial tensor for the current snapshot."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MolecularMechanics calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        bonded_forces = np.zeros_like(positions)
        self._add_bonds(positions, cell, pbc, bonded_forces)
        self._add_angles(positions, cell, pbc, bonded_forces)
        self._add_torsions(positions, cell, pbc, bonded_forces)
        self._add_impropers(positions, cell, pbc, bonded_forces)
        self._add_cmaps(positions, cell, pbc, bonded_forces)
        centered_positions = positions - positions.mean(axis=0)
        virial = centered_positions.T @ bonded_forces
        self._add_nonbonded(
            atoms,
            positions,
            cell,
            pbc,
            np.zeros_like(positions),
            virial=virial,
        )
        return virial

    def energy_components(self, atoms=None):
        """Return a named energy decomposition for the current snapshot."""
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MolecularMechanics calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        components = {}
        for name, callback in (
            ("bonds", self._add_bonds),
            ("angles", self._add_angles),
            ("torsions", self._add_torsions),
            ("impropers", self._add_impropers),
            ("cmaps", self._add_cmaps),
        ):
            forces = np.zeros_like(positions)
            components[name] = callback(positions, cell, pbc, forces)
        forces = np.zeros_like(positions)
        components["nonbonded"] = self._add_nonbonded(atoms, positions, cell, pbc, forces)
        components["total"] = float(sum(components.values()))
        return components

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
        energy += self._add_impropers(positions, cell, pbc, forces)
        energy += self._add_cmaps(positions, cell, pbc, forces)
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
        return _add_bond_arrays(
            positions,
            cell,
            pbc,
            forces,
            self._bond_indices,
            self._bond_force_constants,
            self._bond_equilibria,
        )

    def _add_torsions(self, positions, cell, pbc, forces):
        return _add_torsion_arrays(
            positions,
            cell,
            pbc,
            forces,
            self._torsion_indices,
            self._torsion_barriers,
            self._torsion_periodicities,
            self._torsion_phases,
        )

    def _torsion_energy(self, positions, cell, pbc):
        energy = 0.0
        for i, j, k, l, barrier, periodicity, phase in self.torsions:
            phi = _dihedral_angle(positions, cell, pbc, i, j, k, l)
            energy += barrier * (1.0 + np.cos(periodicity * phi - phase))
        return energy

    def _add_impropers(self, positions, cell, pbc, forces):
        energy = 0.0
        for i, j, k, l, force_constant, phase in self.impropers:
            phi, gradients = _dihedral_angle_and_gradient(positions, cell, pbc, i, j, k, l)
            bend = _angle_difference(phi, phase)
            energy += 0.5 * force_constant * bend * bend
            denergy_dphi = force_constant * bend
            for atom_index, gradient in zip((i, j, k, l), gradients):
                forces[atom_index] += -denergy_dphi * gradient
        return energy

    def _add_cmaps(self, positions, cell, pbc, forces):
        energy = 0.0
        for map_index, atoms in self.cmaps:
            if map_index < 0 or map_index >= len(self.cmap_grids):
                raise ValueError(f"CMAP term references missing grid index {map_index}.")
            coefficients = self._cmap_coefficients[map_index]
            energy += self._add_cmap_forces_for_atoms(positions, cell, pbc, forces, atoms, coefficients)
        return energy

    def _cmap_energy_for_atoms(self, positions, cell, pbc, atoms, coefficients):
        if len(atoms) != 8:
            raise ValueError("CMAP terms must contain eight atom indices.")
        phi = _dihedral_angle(positions, cell, pbc, atoms[0], atoms[1], atoms[2], atoms[3])
        psi = _dihedral_angle(positions, cell, pbc, atoms[4], atoms[5], atoms[6], atoms[7])
        return _periodic_bicubic_grid_value(coefficients, phi, psi)

    def _add_cmap_forces_for_atoms(self, positions, cell, pbc, forces, atoms, coefficients):
        if len(atoms) != 8:
            raise ValueError("CMAP terms must contain eight atom indices.")
        phi, phi_gradients = _dihedral_angle_and_gradient(
            positions,
            cell,
            pbc,
            atoms[0],
            atoms[1],
            atoms[2],
            atoms[3],
        )
        psi, psi_gradients = _dihedral_angle_and_gradient(
            positions,
            cell,
            pbc,
            atoms[4],
            atoms[5],
            atoms[6],
            atoms[7],
        )
        energy, denergy_dphi, denergy_dpsi = _periodic_bicubic_grid_value_and_gradient(
            coefficients,
            phi,
            psi,
        )
        for atom_index, gradient in zip(atoms[:4], phi_gradients):
            forces[atom_index] += -denergy_dphi * gradient
        for atom_index, gradient in zip(atoms[4:], psi_gradients):
            forces[atom_index] += -denergy_dpsi * gradient
        return energy

    def _add_angles(self, positions, cell, pbc, forces):
        return _add_angle_arrays(
            positions,
            cell,
            pbc,
            forces,
            self._angle_indices,
            self._angle_force_constants,
            self._angle_equilibria,
        )

    def _add_nonbonded(
        self,
        atoms,
        positions,
        cell,
        pbc,
        forces,
        extra_lj_exclusions=None,
        extra_coulomb_exclusions=None,
        virial=None,
    ):
        charges = self._charges(atoms)
        if charges is None and self.lj_epsilon is None:
            return 0.0
        if self.lj_epsilon is not None and self.lj_sigma is None:
            raise ValueError("lj_sigma is required when lj_epsilon is set.")

        lj_exclusions = self._nonbonded_exclusions("lj", extra_lj_exclusions)
        coulomb_exclusions = self._nonbonded_exclusions("coulomb", extra_coulomb_exclusions)
        if charges is not None and self.coulomb_pair_parameters:
            coulomb_exclusions = set(coulomb_exclusions or set())
            coulomb_exclusions.update(self.coulomb_pair_parameters)
        shared_pair_displacements = None
        lj_pair_displacements = None
        coulomb_pair_displacements = None
        if self.lj_epsilon is not None and charges is not None:
            shared_cutoff = self._shared_nonbonded_cutoff(lj_exclusions, coulomb_exclusions)
            if shared_cutoff is not None:
                shared_exclusions = (
                    None if self.coulomb_method in {"ewald", "pme"} else lj_exclusions
                )
                shared_pair_displacements = self._shared_pair_displacement_cache.pair_displacements(
                    positions,
                    cell,
                    pbc,
                    shared_cutoff,
                    shared_exclusions,
                )
        if self.lj_epsilon is not None:
            lj_pair_displacements = shared_pair_displacements
            if lj_pair_displacements is None and self.nonbonded_skin > 0.0:
                lj_pair_displacements = self._lj_pair_displacement_cache.pair_displacements(
                    positions,
                    cell,
                    pbc,
                    self.lj_cutoff,
                    lj_exclusions,
                )
        if charges is not None:
            coulomb_pair_displacements = shared_pair_displacements
            if coulomb_pair_displacements is None and self.nonbonded_skin > 0.0:
                coulomb_cutoff = self._real_space_coulomb_cutoff(cell)
                coulomb_real_exclusions = (
                    coulomb_exclusions if self.coulomb_method == "cutoff" else None
                )
                coulomb_pair_displacements = (
                    self._coulomb_pair_displacement_cache.pair_displacements(
                        positions,
                        cell,
                        pbc,
                        coulomb_cutoff,
                        coulomb_real_exclusions,
                    )
                )
        energy = 0.0
        combined_lj_pme = (
            self.lj_epsilon is not None
            and charges is not None
            and self.coulomb_method == "pme"
            and isinstance(shared_pair_displacements, _PairDisplacements)
        )
        if combined_lj_pme:
            energy += _add_lj_pme_coulomb_shared(
                positions,
                cell,
                pbc,
                forces,
                self.lj_epsilon,
                self.lj_sigma,
                charges,
                self.coulomb_constant,
                self.ewald_alpha,
                self.coulomb_cutoff,
                self.pme_mesh,
                self.pme_order,
                shared_pair_displacements,
                lj_cutoff=self.lj_cutoff,
                lj_switch_on=self.lj_switch_on,
                lj_energy_shift=self.lj_energy_shift,
                lj_exclusions=lj_exclusions,
                atom_types=self.atom_types,
                pair_overrides=self.lj_pair_overrides,
                pair_override_lookup=self._lj_pair_override_lookup,
                pair_parameter_lookup=self._lj_type_pair_parameter_lookup,
                virial=virial,
            )
        if self.lj_epsilon is not None:
            if not combined_lj_pme:
                energy += _add_lennard_jones_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    self.lj_epsilon,
                    self.lj_sigma,
                    self.lj_cutoff,
                    self.lj_switch_on,
                    self.lj_energy_shift,
                    lj_exclusions,
                    pair_displacements=lj_pair_displacements,
                    atom_types=self.atom_types,
                    pair_overrides=self.lj_pair_overrides,
                    pair_override_lookup=self._lj_pair_override_lookup,
                    pair_parameter_lookup=self._lj_type_pair_parameter_lookup,
                    virial=virial,
                )
            if self.lj_pair_scales:
                energy += _add_lennard_jones_scaled_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    self.lj_epsilon,
                    self.lj_sigma,
                    self.lj_pair_scales,
                    virial=virial,
                )
            if self.lj_pair_parameters:
                energy += _add_lennard_jones_specific_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    self.lj_pair_parameters,
                    virial=virial,
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
                    real_pair_displacements=coulomb_pair_displacements,
                    virial=virial,
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
                        virial=virial,
                    )
            elif self.coulomb_method == "pme":
                if not combined_lj_pme:
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
                        self.pme_order,
                        real_pair_displacements=coulomb_pair_displacements,
                        virial=virial,
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
                        virial=virial,
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
                    pair_displacements=coulomb_pair_displacements,
                    energy_shift=self.coulomb_energy_shift,
                    reaction_field_dielectric=self.coulomb_reaction_field_dielectric,
                    virial=virial,
                )
            if self.coulomb_pair_scales:
                energy += _add_coulomb_scaled_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    charges,
                    self.coulomb_constant,
                    self.coulomb_pair_scales,
                    virial=virial,
                )
            if self.coulomb_pair_parameters:
                energy += _add_coulomb_specific_pairs(
                    positions,
                    cell,
                    pbc,
                    forces,
                    self.coulomb_constant,
                    self.coulomb_pair_parameters,
                    virial=virial,
                )
        return energy

    def _shared_nonbonded_cutoff(self, lj_exclusions, coulomb_exclusions):
        """Return a cutoff when LJ and Coulomb can reuse pair displacements."""
        if self.lj_cutoff is None or self.coulomb_cutoff is None:
            return None
        if not np.isclose(self.lj_cutoff, self.coulomb_cutoff, rtol=0.0, atol=1e-12):
            return None
        if self.coulomb_method in {"ewald", "pme"}:
            return self.lj_cutoff
        if lj_exclusions != coulomb_exclusions:
            return None
        return self.lj_cutoff

    def _real_space_coulomb_cutoff(self, cell):
        if self.coulomb_cutoff is not None:
            return self.coulomb_cutoff
        if self.coulomb_method == "cutoff":
            return None
        lengths = _orthorhombic_lengths(cell)
        if lengths is None:
            return None
        return 0.5 * float(np.min(lengths))

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


class _PairDisplacementCache:
    """Reuse cutoff+skin pair lists across nearby MD steps."""

    def __init__(self, skin):
        self.skin = float(skin)
        self._key = None
        self._reference_positions = None
        self._pair_i = None
        self._pair_j = None

    def pair_displacements(self, positions, cell, pbc, cutoff, exclusions=None):
        if self.skin == 0.0 or cutoff is None:
            pair_i, pair_j, rij = _candidate_pair_displacement_arrays(
                positions,
                cell,
                pbc,
                cutoff,
                exclusions,
            )
            return _PairDisplacements(pair_i, pair_j, rij)

        positions = np.asarray(positions, dtype=float)
        cell = np.asarray(cell, dtype=float)
        pbc = np.asarray(pbc, dtype=bool)
        lengths = _orthorhombic_lengths(cell)
        key = self._cache_key(positions, cell, pbc, cutoff, exclusions)
        if key != self._key or self._needs_rebuild(positions, cell, pbc, lengths):
            self._pair_i, self._pair_j, _rij = _candidate_pair_displacement_arrays(
                positions,
                cell,
                pbc,
                float(cutoff) + self.skin,
                exclusions,
            )
            self._key = key
            self._reference_positions = positions.copy()

        if len(self._pair_i) == 0:
            return _PairDisplacements(
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.zeros((0, 3), dtype=float),
            )

        rij = positions[self._pair_i] - positions[self._pair_j]
        if lengths is not None:
            axes = np.nonzero(pbc)[0]
            if len(axes) > 0:
                rij[:, axes] -= lengths[axes] * np.round(rij[:, axes] / lengths[axes])
        else:
            rij = np.array([_minimum_image(vector, cell, pbc) for vector in rij])
        return _PairDisplacements(self._pair_i, self._pair_j, rij)

    def _needs_rebuild(self, positions, cell, pbc, lengths):
        if self._reference_positions is None:
            return True
        displacement = positions - self._reference_positions
        if lengths is not None:
            axes = np.nonzero(pbc)[0]
            if len(axes) > 0:
                displacement[:, axes] -= lengths[axes] * np.round(
                    displacement[:, axes] / lengths[axes]
                )
        else:
            displacement = np.array([_minimum_image(vector, cell, pbc) for vector in displacement])
        max_displacement2 = float(np.max(np.einsum("ij,ij->i", displacement, displacement)))
        return max_displacement2 > (0.5 * self.skin) ** 2

    @staticmethod
    def _cache_key(positions, cell, pbc, cutoff, exclusions):
        exclusion_key = None
        if exclusions is not None:
            exclusion_key = tuple(sorted(tuple(pair) for pair in exclusions))
        return (
            len(positions),
            tuple(np.round(np.asarray(cell, dtype=float).ravel(), 12)),
            tuple(np.asarray(pbc, dtype=bool).tolist()),
            float(cutoff),
            exclusion_key,
        )


class _PairDisplacements:
    def __init__(self, pair_i, pair_j, displacements):
        self.pair_i = pair_i
        self.pair_j = pair_j
        self.displacements = displacements

    def __iter__(self):
        return iter(zip(self.pair_i, self.pair_j, self.displacements))


def _pair_set(pairs):
    if pairs is None:
        return set()
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def _pair_scale_dict(pair_scales):
    if pair_scales is None:
        return {}
    if hasattr(pair_scales, "items"):
        items = pair_scales.items()
    else:
        items = pair_scales
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(scale)
        for pair, scale in items
    }


def _pair_float_dict(pair_values):
    if pair_values is None:
        return {}
    if hasattr(pair_values, "items"):
        items = pair_values.items()
    else:
        items = pair_values
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): float(value)
        for pair, value in items
    }


def _pair_lj_parameter_dict(pair_parameters):
    if pair_parameters is None:
        return {}
    if hasattr(pair_parameters, "items"):
        items = pair_parameters.items()
    else:
        items = pair_parameters
    return {
        tuple(sorted((int(pair[0]), int(pair[1])))): (float(values[0]), float(values[1]))
        for pair, values in items
    }


def _lj_pair_override_dict(pair_overrides):
    if pair_overrides is None:
        return {}
    if hasattr(pair_overrides, "items"):
        items = pair_overrides.items()
    else:
        items = pair_overrides
    return {
        tuple(sorted((str(pair[0]), str(pair[1])))): (float(values[0]), float(values[1]))
        for pair, values in items
    }


def _lj_pair_override_lookup(atom_types, pair_overrides):
    if atom_types is None or not pair_overrides:
        return None
    atom_types = np.asarray(atom_types, dtype=str)
    unique_types, type_codes = np.unique(atom_types, return_inverse=True)
    type_index = {atom_type: index for index, atom_type in enumerate(unique_types)}
    ntypes = len(unique_types)
    has_override = np.zeros((ntypes, ntypes), dtype=bool)
    epsilon = np.zeros((ntypes, ntypes), dtype=float)
    sigma = np.zeros((ntypes, ntypes), dtype=float)
    for (type_a, type_b), (epsilon_ab, sigma_ab) in pair_overrides.items():
        if type_a not in type_index or type_b not in type_index:
            continue
        ia = type_index[type_a]
        ib = type_index[type_b]
        has_override[ia, ib] = True
        has_override[ib, ia] = True
        epsilon[ia, ib] = epsilon_ab
        epsilon[ib, ia] = epsilon_ab
        sigma[ia, ib] = sigma_ab
        sigma[ib, ia] = sigma_ab
    if not np.any(has_override):
        return None
    return type_codes.astype(np.int32, copy=False), has_override, epsilon, sigma


def _lj_type_pair_parameter_lookup(atom_types, epsilon, sigma, pair_overrides):
    if atom_types is None or epsilon is None or sigma is None:
        return None
    atom_types = np.asarray(atom_types, dtype=str)
    epsilon = np.asarray(epsilon, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if len(atom_types) != len(epsilon) or len(atom_types) != len(sigma):
        return None
    unique_types, type_codes = np.unique(atom_types, return_inverse=True)
    ntypes = len(unique_types)
    type_epsilon = np.empty(ntypes, dtype=float)
    type_sigma = np.empty(ntypes, dtype=float)
    for code in range(ntypes):
        mask = type_codes == code
        eps_values = epsilon[mask]
        sig_values = sigma[mask]
        if not np.allclose(eps_values, eps_values[0], rtol=0.0, atol=1e-14):
            return None
        if not np.allclose(sig_values, sig_values[0], rtol=0.0, atol=1e-14):
            return None
        type_epsilon[code] = eps_values[0]
        type_sigma[code] = sig_values[0]

    epsilon_matrix = np.sqrt(type_epsilon[:, np.newaxis] * type_epsilon[np.newaxis, :])
    sigma_matrix = 0.5 * (type_sigma[:, np.newaxis] + type_sigma[np.newaxis, :])
    type_index = {atom_type: index for index, atom_type in enumerate(unique_types)}
    for (type_a, type_b), (epsilon_ab, sigma_ab) in pair_overrides.items():
        if type_a not in type_index or type_b not in type_index:
            continue
        ia = type_index[type_a]
        ib = type_index[type_b]
        epsilon_matrix[ia, ib] = epsilon_ab
        epsilon_matrix[ib, ia] = epsilon_ab
        sigma_matrix[ia, ib] = sigma_ab
        sigma_matrix[ib, ia] = sigma_ab
    return type_codes.astype(np.int32, copy=False), epsilon_matrix, sigma_matrix


def _bond_arrays(bonds):
    if not bonds:
        return (
            np.zeros((0, 2), dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    values = np.asarray(bonds, dtype=float)
    return values[:, :2].astype(int), values[:, 2].astype(float), values[:, 3].astype(float)


def _angle_arrays(angles):
    if not angles:
        return (
            np.zeros((0, 3), dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    values = np.asarray(angles, dtype=float)
    return values[:, :3].astype(int), values[:, 3].astype(float), values[:, 4].astype(float)


def _torsion_arrays(torsions):
    if not torsions:
        return (
            np.zeros((0, 4), dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )
    values = np.asarray(torsions, dtype=float)
    return (
        values[:, :4].astype(int),
        values[:, 4].astype(float),
        values[:, 5].astype(int),
        values[:, 6].astype(float),
    )


def _switching_function(r, switch_on, cutoff):
    if r <= switch_on:
        return 1.0, 0.0
    if r >= cutoff:
        return 0.0, 0.0
    r2 = r * r
    rs2 = switch_on * switch_on
    rc2 = cutoff * cutoff
    denom = (rc2 - rs2) ** 3
    a = rc2 - r2
    b = rc2 + 2.0 * r2 - 3.0 * rs2
    switch = (a * a * b) / denom
    dswitch_dr = 12.0 * r * (rc2 - r2) * (rs2 - r2) / denom
    return float(switch), float(dswitch_dr)


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


def _add_bond_arrays(positions, cell, pbc, forces, indices, force_constants, equilibria):
    if len(indices) == 0:
        return 0.0
    i = indices[:, 0]
    j = indices[:, 1]
    rij = _pair_displacements_for_indices(positions, cell, pbc, i, j)
    distance = np.linalg.norm(rij, axis=1)
    if np.any(distance == 0.0):
        raise ValueError("Bond distance is zero.")
    stretch = distance - equilibria
    energy = 0.5 * force_constants * stretch * stretch
    fij = (-(force_constants * stretch / distance))[:, np.newaxis] * rij
    _scatter_pair_forces(forces, i, j, fij)
    return float(np.sum(energy))


def _add_angle_arrays(positions, cell, pbc, forces, indices, force_constants, equilibria):
    if len(indices) == 0:
        return 0.0
    i = indices[:, 0]
    j = indices[:, 1]
    k = indices[:, 2]
    rij = _pair_displacements_for_indices(positions, cell, pbc, i, j)
    rkj = _pair_displacements_for_indices(positions, cell, pbc, k, j)
    rij_norm = np.linalg.norm(rij, axis=1)
    rkj_norm = np.linalg.norm(rkj, axis=1)
    if np.any((rij_norm == 0.0) | (rkj_norm == 0.0)):
        raise ValueError("Angle contains a zero-length bond.")

    cos_theta = np.einsum("ij,ij->i", rij, rkj) / (rij_norm * rkj_norm)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    sin_theta = np.sqrt(np.maximum(1.0 - cos_theta * cos_theta, 1.0e-24))
    bend = theta - equilibria
    energy = 0.5 * force_constants * bend * bend

    prefactor = force_constants * bend / sin_theta
    dtheta_drij = (
        (cos_theta / (rij_norm * rij_norm))[:, np.newaxis] * rij
        - (1.0 / (rij_norm * rkj_norm))[:, np.newaxis] * rkj
    )
    dtheta_drkj = (
        (cos_theta / (rkj_norm * rkj_norm))[:, np.newaxis] * rkj
        - (1.0 / (rij_norm * rkj_norm))[:, np.newaxis] * rij
    )
    force_i = -prefactor[:, np.newaxis] * dtheta_drij
    force_k = -prefactor[:, np.newaxis] * dtheta_drkj
    force_j = -(force_i + force_k)
    _scatter_index_forces(forces, i, force_i)
    _scatter_index_forces(forces, j, force_j)
    _scatter_index_forces(forces, k, force_k)
    return float(np.sum(energy))


def _dihedral_angle_and_gradient(positions, cell, pbc, i, j, k, l):
    rij = _minimum_image(positions[j] - positions[i], cell, pbc)
    rjk = _minimum_image(positions[k] - positions[j], cell, pbc)
    rkl = _minimum_image(positions[l] - positions[k], cell, pbc)
    p0 = np.zeros(3)
    p1 = p0 + rij
    p2 = p1 + rjk
    p3 = p2 + rkl

    angle = _dihedral_angle_from_points(p0, p1, p2, p3)
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    b2_norm = float(np.linalg.norm(b2))
    if b2_norm == 0.0:
        raise ValueError("Torsion contains a zero-length central bond.")
    normal1 = np.cross(b1, b2)
    normal2 = np.cross(b2, b3)
    normal1_norm2 = float(np.dot(normal1, normal1))
    normal2_norm2 = float(np.dot(normal2, normal2))
    if normal1_norm2 == 0.0 or normal2_norm2 == 0.0:
        raise ValueError("Torsion contains collinear or zero-length bonds.")
    b2_norm2 = float(np.dot(b2, b2))
    gradient0 = -b2_norm * normal1 / normal1_norm2
    gradient3 = b2_norm * normal2 / normal2_norm2
    projection1 = float(np.dot(b1, b2)) / b2_norm2
    projection3 = float(np.dot(b3, b2)) / b2_norm2
    gradient1 = -(1.0 + projection1) * gradient0 + projection3 * gradient3
    gradient2 = projection1 * gradient0 - (1.0 + projection3) * gradient3
    return angle, np.array([gradient0, gradient1, gradient2, gradient3], dtype=float)


def _add_torsion_arrays(
    positions,
    cell,
    pbc,
    forces,
    indices,
    barriers,
    periodicities,
    phases,
):
    if len(indices) == 0:
        return 0.0
    i = indices[:, 0]
    j = indices[:, 1]
    k = indices[:, 2]
    l = indices[:, 3]
    b1 = _pair_displacements_for_indices(positions, cell, pbc, j, i)
    b2 = _pair_displacements_for_indices(positions, cell, pbc, k, j)
    b3 = _pair_displacements_for_indices(positions, cell, pbc, l, k)

    angles = _dihedral_angles_from_vectors(b1, b2, b3)
    normal1 = np.cross(b1, b2)
    normal2 = np.cross(b2, b3)
    normal1_norm2 = np.einsum("ij,ij->i", normal1, normal1)
    normal2_norm2 = np.einsum("ij,ij->i", normal2, normal2)
    b2_norm2 = np.einsum("ij,ij->i", b2, b2)
    if np.any(b2_norm2 == 0.0):
        raise ValueError("Torsion contains a zero-length central bond.")
    if np.any((normal1_norm2 == 0.0) | (normal2_norm2 == 0.0)):
        raise ValueError("Torsion contains collinear or zero-length bonds.")

    b2_norm = np.sqrt(b2_norm2)
    gradient0 = -(b2_norm / normal1_norm2)[:, np.newaxis] * normal1
    gradient3 = (b2_norm / normal2_norm2)[:, np.newaxis] * normal2
    projection1 = np.einsum("ij,ij->i", b1, b2) / b2_norm2
    projection3 = np.einsum("ij,ij->i", b3, b2) / b2_norm2
    gradient1 = -(1.0 + projection1)[:, np.newaxis] * gradient0 + projection3[:, np.newaxis] * gradient3
    gradient2 = projection1[:, np.newaxis] * gradient0 - (1.0 + projection3)[:, np.newaxis] * gradient3

    argument = periodicities * angles - phases
    energy = barriers * (1.0 + np.cos(argument))
    denergy_dphi = -barriers * periodicities * np.sin(argument)
    scale = -denergy_dphi[:, np.newaxis]
    _scatter_index_forces(forces, i, scale * gradient0)
    _scatter_index_forces(forces, j, scale * gradient1)
    _scatter_index_forces(forces, k, scale * gradient2)
    _scatter_index_forces(forces, l, scale * gradient3)
    return float(np.sum(energy))


def _dihedral_angles_from_vectors(b1, b2, b3):
    b0 = -b1
    b2_norm = np.linalg.norm(b2, axis=1)
    if np.any(b2_norm == 0.0):
        raise ValueError("Torsion contains a zero-length central bond.")
    b2_unit = b2 / b2_norm[:, np.newaxis]
    v = b0 - np.einsum("ij,ij->i", b0, b2_unit)[:, np.newaxis] * b2_unit
    w = b3 - np.einsum("ij,ij->i", b3, b2_unit)[:, np.newaxis] * b2_unit
    v_norm = np.linalg.norm(v, axis=1)
    w_norm = np.linalg.norm(w, axis=1)
    if np.any((v_norm == 0.0) | (w_norm == 0.0)):
        raise ValueError("Torsion contains collinear or zero-length bonds.")
    v = v / v_norm[:, np.newaxis]
    w = w / w_norm[:, np.newaxis]
    x = np.einsum("ij,ij->i", v, w)
    y = np.einsum("ij,ij->i", np.cross(b2_unit, v), w)
    return np.arctan2(y, x)


def _dihedral_angle_from_points(p0, p1, p2, p3):
    b0 = -(p1 - p0)
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = np.linalg.norm(b1)
    if b1_norm == 0.0:
        raise ValueError("Torsion contains a zero-length central bond.")
    b1 = b1 / b1_norm
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    v_norm = np.linalg.norm(v)
    w_norm = np.linalg.norm(w)
    if v_norm == 0.0 or w_norm == 0.0:
        raise ValueError("Torsion contains collinear or zero-length bonds.")
    v = v / v_norm
    w = w / w_norm
    x = np.dot(v, w)
    y = np.dot(np.cross(b1, v), w)
    return float(np.arctan2(y, x))


def _angle_difference(a, b):
    return float(np.arctan2(np.sin(a - b), np.cos(a - b)))


def _periodic_bicubic_coefficients(grid):
    grid = np.asarray(grid, dtype=float)
    if grid.ndim != 2 or grid.shape[0] != grid.shape[1]:
        raise ValueError("CMAP grid must be a square two-dimensional array.")
    size = grid.shape[0]
    if size < 2:
        raise ValueError("CMAP grid must have at least two points per dimension.")

    # Match OpenMM's CMAPTorsionForce map layout: Python row-major input is
    # interpreted with the second array axis as the first torsion coordinate.
    values = grid.T
    dx = _periodic_cubic_spline_slopes(values, axis=0)
    dy = _periodic_cubic_spline_slopes(values, axis=1)
    dxy = _periodic_cubic_spline_slopes(dx, axis=1)
    hermite = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [-3.0, 3.0, -2.0, -1.0],
            [2.0, -2.0, 1.0, 1.0],
        ],
        dtype=float,
    )
    coefficients = np.zeros((size, size, 4, 4), dtype=float)
    for i in range(size):
        for j in range(size):
            i1 = (i + 1) % size
            j1 = (j + 1) % size
            geometry = np.array(
                [
                    [values[i, j], values[i, j1], dy[i, j], dy[i, j1]],
                    [values[i1, j], values[i1, j1], dy[i1, j], dy[i1, j1]],
                    [dx[i, j], dx[i, j1], dxy[i, j], dxy[i, j1]],
                    [dx[i1, j], dx[i1, j1], dxy[i1, j], dxy[i1, j1]],
                ],
                dtype=float,
            )
            coefficients[i, j] = hermite @ geometry @ hermite.T
    return coefficients


def _periodic_cubic_spline_slopes(values, axis):
    values = np.asarray(values, dtype=float)
    size = values.shape[axis]
    matrix = np.zeros((size, size), dtype=float)
    for index in range(size):
        matrix[index, (index - 1) % size] = 1.0
        matrix[index, index] = 4.0
        matrix[index, (index + 1) % size] = 1.0
    moved = np.moveaxis(values, axis, 0)
    slopes = np.empty_like(moved)
    for index in np.ndindex(moved.shape[1:]):
        line = moved[(slice(None),) + index]
        rhs = np.array(
            [3.0 * (line[(i + 1) % size] - line[(i - 1) % size]) for i in range(size)],
            dtype=float,
        )
        slopes[(slice(None),) + index] = np.linalg.solve(matrix, rhs)
    return np.moveaxis(slopes, 0, axis)


def _periodic_bicubic_grid_value(coefficients, phi, psi):
    value, _dphi, _dpsi = _periodic_bicubic_grid_value_and_gradient(coefficients, phi, psi)
    return value


def _periodic_bicubic_grid_value_and_gradient(coefficients, phi, psi):
    coefficients = np.asarray(coefficients, dtype=float)
    if coefficients.ndim != 4 or coefficients.shape[2:] != (4, 4):
        raise ValueError("CMAP coefficients must have shape (n, n, 4, 4).")
    size = coefficients.shape[0]
    if coefficients.shape[1] != size:
        raise ValueError("CMAP coefficients must describe a square grid.")

    scale = size / (2.0 * np.pi)
    x = (phi % (2.0 * np.pi)) * scale
    y = (psi % (2.0 * np.pi)) * scale
    i = int(np.floor(x)) % size
    j = int(np.floor(y)) % size
    tx = x - np.floor(x)
    ty = y - np.floor(y)
    powers_x = np.array([1.0, tx, tx * tx, tx * tx * tx], dtype=float)
    powers_y = np.array([1.0, ty, ty * ty, ty * ty * ty], dtype=float)
    dpowers_x = np.array([0.0, 1.0, 2.0 * tx, 3.0 * tx * tx], dtype=float)
    dpowers_y = np.array([0.0, 1.0, 2.0 * ty, 3.0 * ty * ty], dtype=float)
    local = coefficients[i, j]
    value = float(powers_x @ local @ powers_y)
    dvalue_dx = float(dpowers_x @ local @ powers_y)
    dvalue_dy = float(powers_x @ local @ dpowers_y)
    return value, dvalue_dx * scale, dvalue_dy * scale


def _add_lennard_jones_pairs(
    positions,
    cell,
    pbc,
    forces,
    epsilon,
    sigma,
    cutoff=None,
    switch_on=None,
    energy_shift=True,
    exclusions=None,
    pair_displacements=None,
    atom_types=None,
    pair_overrides=None,
    pair_override_lookup=None,
    pair_parameter_lookup=None,
    virial=None,
):
    energy = 0.0
    natoms = len(positions)
    epsilon = _parameter_array(epsilon, natoms, "epsilon")
    sigma = _parameter_array(sigma, natoms, "sigma")
    atom_types = None if atom_types is None else np.asarray(atom_types, dtype=str)
    pair_overrides = _lj_pair_override_dict(pair_overrides)

    pairs = pair_displacements
    if pairs is None:
        pairs = _candidate_pair_displacements(positions, cell, pbc, cutoff, exclusions)
    cutoff2 = None if cutoff is None else cutoff * cutoff
    switch_on = None if switch_on is None else float(switch_on)
    if switch_on is not None:
        if cutoff is None:
            raise ValueError("lj_switch_on requires a finite LJ cutoff.")
        if switch_on <= 0.0 or switch_on >= cutoff:
            raise ValueError("lj_switch_on must be positive and smaller than lj_cutoff.")
    if isinstance(pairs, _PairDisplacements) and (not pair_overrides or atom_types is not None):
        return _add_lennard_jones_pair_arrays(
            forces,
            epsilon,
            sigma,
            cutoff,
            cutoff2,
            switch_on,
            energy_shift,
            pairs.pair_i,
            pairs.pair_j,
            pairs.displacements,
            exclusions=exclusions,
            atom_types=atom_types,
            pair_overrides=pair_overrides,
            pair_override_lookup=pair_override_lookup,
            pair_parameter_lookup=pair_parameter_lookup,
            virial=virial,
        )
    for i, j, rij in pairs:
        if exclusions is not None and tuple(sorted((int(i), int(j)))) in exclusions:
            continue
        override = None
        if atom_types is not None and pair_overrides:
            override = pair_overrides.get(tuple(sorted((str(atom_types[i]), str(atom_types[j])))))
        if override is None:
            epsilon_ij = np.sqrt(epsilon[i] * epsilon[j])
            sigma_ij = 0.5 * (sigma[i] + sigma[j])
        else:
            epsilon_ij, sigma_ij = override
        if epsilon_ij == 0.0:
            continue
        shift = 0.0
        if cutoff is not None and energy_shift and switch_on is None:
            sr6_cutoff = (sigma_ij / cutoff) ** 6
            shift = 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
        r2 = float(np.dot(rij, rij))
        if r2 == 0.0:
            raise ValueError("Lennard-Jones pair distance is zero.")
        if cutoff2 is not None and r2 > cutoff2:
            continue

        r = np.sqrt(r2)
        inv_r2 = 1.0 / r2
        sr2 = (sigma_ij * sigma_ij) * inv_r2
        sr6 = sr2 ** 3
        sr12 = sr6 * sr6
        pair_energy = 4.0 * epsilon_ij * (sr12 - sr6)
        fij = 24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2 * rij
        if switch_on is not None and r > switch_on:
            switch, dswitch_dr = _switching_function(r, switch_on, cutoff)
            fij = switch * fij - pair_energy * dswitch_dr * rij / r
            pair_energy *= switch
        energy += pair_energy - shift
        forces[i] += fij
        forces[j] -= fij
        _accumulate_pair_virial(virial, rij, fij)
    return energy


def _add_lennard_jones_pair_arrays(
    forces,
    epsilon,
    sigma,
    cutoff,
    cutoff2,
    switch_on,
    energy_shift,
    pair_i,
    pair_j,
    displacements,
    exclusions=None,
    atom_types=None,
    pair_overrides=None,
    pair_override_lookup=None,
    pair_parameter_lookup=None,
    virial=None,
):
    if len(pair_i) == 0:
        return 0.0
    rij = displacements
    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Lennard-Jones pair distance is zero.")
    if cutoff2 is not None:
        active = r2 <= cutoff2
        if not np.any(active):
            return 0.0
        pair_i = pair_i[active]
        pair_j = pair_j[active]
        rij = rij[active]
        r2 = r2[active]
    if exclusions is not None:
        active = _nonexcluded_pair_mask(pair_i, pair_j, exclusions, len(forces))
        if not np.any(active):
            return 0.0
        pair_i = pair_i[active]
        pair_j = pair_j[active]
        rij = rij[active]
        r2 = r2[active]
    if pair_parameter_lookup is not None:
        type_codes, type_epsilon, type_sigma = pair_parameter_lookup
        code_i = type_codes[pair_i]
        code_j = type_codes[pair_j]
        epsilon_ij = type_epsilon[code_i, code_j]
        sigma_ij = type_sigma[code_i, code_j]
    else:
        epsilon_ij = np.sqrt(epsilon[pair_i] * epsilon[pair_j])
        sigma_ij = 0.5 * (sigma[pair_i] + sigma[pair_j])
    if pair_overrides and pair_parameter_lookup is None:
        if pair_override_lookup is None:
            _apply_lj_pair_overrides(epsilon_ij, sigma_ij, pair_i, pair_j, atom_types, pair_overrides)
        else:
            _apply_lj_pair_override_lookup(
                epsilon_ij,
                sigma_ij,
                pair_i,
                pair_j,
                pair_override_lookup,
            )
    active = epsilon_ij != 0.0
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    rij = rij[active]
    r2 = r2[active]
    epsilon_ij = epsilon_ij[active]
    sigma_ij = sigma_ij[active]

    inv_r2 = 1.0 / r2
    sr2 = sigma_ij * sigma_ij * inv_r2
    sr6 = sr2 ** 3
    sr12 = sr6 * sr6
    pair_energy = 4.0 * epsilon_ij * (sr12 - sr6)
    fij = (24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2)[:, np.newaxis] * rij
    if switch_on is not None:
        r = np.sqrt(r2)
        switching = r > switch_on
        if np.any(switching):
            switch, dswitch_dr = _switching_function_arrays(r[switching], switch_on, cutoff)
            fij[switching] = (
                switch[:, np.newaxis] * fij[switching]
                - (pair_energy[switching] * dswitch_dr / r[switching])[:, np.newaxis]
                * rij[switching]
            )
            pair_energy[switching] *= switch
    shift = 0.0
    if cutoff is not None and energy_shift and switch_on is None:
        sr6_cutoff = (sigma_ij / cutoff) ** 6
        shift = 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)

    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(pair_energy - shift))


def _apply_lj_pair_overrides(epsilon_ij, sigma_ij, pair_i, pair_j, atom_types, pair_overrides):
    if atom_types is None:
        return
    atom_types = np.asarray(atom_types, dtype=str)
    type_i = atom_types[pair_i]
    type_j = atom_types[pair_j]
    for (type_a, type_b), (epsilon, sigma) in pair_overrides.items():
        if type_a == type_b:
            mask = (type_i == type_a) & (type_j == type_a)
        else:
            mask = ((type_i == type_a) & (type_j == type_b)) | (
                (type_i == type_b) & (type_j == type_a)
            )
        if np.any(mask):
            epsilon_ij[mask] = epsilon
            sigma_ij[mask] = sigma


def _apply_lj_pair_override_lookup(epsilon_ij, sigma_ij, pair_i, pair_j, lookup):
    type_codes, has_override, override_epsilon, override_sigma = lookup
    code_i = type_codes[pair_i]
    code_j = type_codes[pair_j]
    active = has_override[code_i, code_j]
    if np.any(active):
        epsilon_ij[active] = override_epsilon[code_i[active], code_j[active]]
        sigma_ij[active] = override_sigma[code_i[active], code_j[active]]


def _switching_function_arrays(r, switch_on, cutoff):
    r2 = r * r
    rs2 = switch_on * switch_on
    rc2 = cutoff * cutoff
    denom = (rc2 - rs2) ** 3
    a = rc2 - r2
    b = rc2 + 2.0 * r2 - 3.0 * rs2
    switch = (a * a * b) / denom
    dswitch_dr = 12.0 * r * (rc2 - r2) * (rs2 - r2) / denom
    return switch, dswitch_dr


def _add_lennard_jones_scaled_pairs(
    positions,
    cell,
    pbc,
    forces,
    epsilon,
    sigma,
    pair_scales,
    virial=None,
):
    natoms = len(positions)
    epsilon = _parameter_array(epsilon, natoms, "epsilon")
    sigma = _parameter_array(sigma, natoms, "sigma")
    pair_i, pair_j, correction_scale = _pair_scale_arrays(pair_scales, correction=True)
    if len(pair_i) == 0:
        return 0.0
    epsilon_ij = np.sqrt(epsilon[pair_i] * epsilon[pair_j])
    active = (correction_scale != 0.0) & (epsilon_ij != 0.0)
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    correction_scale = correction_scale[active]
    epsilon_ij = epsilon_ij[active]
    sigma_ij = 0.5 * (sigma[pair_i] + sigma[pair_j])
    rij = _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j)
    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Scaled Lennard-Jones pair distance is zero.")
    inv_r2 = 1.0 / r2
    sr2 = sigma_ij * sigma_ij * inv_r2
    sr6 = sr2 ** 3
    sr12 = sr6 * sr6
    pair_energy = correction_scale * 4.0 * epsilon_ij * (sr12 - sr6)
    fij = (
        correction_scale
        * 24.0
        * epsilon_ij
        * (2.0 * sr12 - sr6)
        * inv_r2
    )[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(pair_energy))


def _add_lennard_jones_specific_pairs(
    positions,
    cell,
    pbc,
    forces,
    pair_parameters,
    virial=None,
):
    pair_i, pair_j, epsilon_ij, sigma_ij = _pair_lj_parameter_arrays(pair_parameters)
    if len(pair_i) == 0:
        return 0.0
    active = epsilon_ij != 0.0
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    epsilon_ij = epsilon_ij[active]
    sigma_ij = sigma_ij[active]
    rij = _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j)
    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Specific Lennard-Jones pair distance is zero.")
    inv_r2 = 1.0 / r2
    sr2 = sigma_ij * sigma_ij * inv_r2
    sr6 = sr2 ** 3
    sr12 = sr6 * sr6
    pair_energy = 4.0 * epsilon_ij * (sr12 - sr6)
    fij = (24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2)[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(pair_energy))


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
    pair_displacements=None,
    energy_shift=False,
    reaction_field_dielectric=None,
    virial=None,
):
    energy = 0.0
    if pair_displacements is not None and only_pairs is None:
        pairs = pair_displacements
    else:
        pairs = only_pairs
        if pairs is None:
            pairs = _candidate_pairs(positions, cell, pbc, cutoff, exclusions)
        elif isinstance(pairs, _PairDisplacements):
            pass
        else:
            pair_i, pair_j = _pair_index_arrays(pairs)
            rij = _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j)
            pairs = _PairDisplacements(pair_i, pair_j, rij)
        pairs = (
            (i, j, _minimum_image(positions[i] - positions[j], cell, pbc))
            for i, j in pairs
        ) if not isinstance(pairs, _PairDisplacements) else pairs
    cutoff2 = None if cutoff is None else cutoff * cutoff
    if isinstance(pairs, _PairDisplacements):
        return _add_coulomb_pair_arrays(
            forces,
            charges,
            coulomb_constant,
            cutoff,
            cutoff2,
            pairs.pair_i,
            pairs.pair_j,
            pairs.displacements,
            energy_shift=energy_shift,
            reaction_field_dielectric=reaction_field_dielectric,
            virial=virial,
        )
    reaction_field = _reaction_field_terms(cutoff, reaction_field_dielectric)
    for i, j, rij in pairs:
        charge_product = charges[i] * charges[j]
        if charge_product == 0.0:
            continue
        r2 = float(np.dot(rij, rij))
        if r2 == 0.0:
            raise ValueError("Coulomb pair distance is zero.")
        if cutoff2 is not None and r2 > cutoff2:
            continue

        distance = np.sqrt(r2)
        prefactor = coulomb_constant * charge_product
        if reaction_field is None:
            shift = 0.0 if cutoff is None or not energy_shift else prefactor / cutoff
            energy += prefactor / distance - shift
            force_scale = prefactor / (distance * r2)
        else:
            krf, crf = reaction_field
            energy += prefactor * (1.0 / distance + krf * r2 - crf)
            force_scale = prefactor * (1.0 / (distance * r2) - 2.0 * krf)
        fij = force_scale * rij
        forces[i] += fij
        forces[j] -= fij
        _accumulate_pair_virial(virial, rij, fij)
    return energy


def _add_coulomb_pair_arrays(
    forces,
    charges,
    coulomb_constant,
    cutoff,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    energy_shift=False,
    reaction_field_dielectric=None,
    virial=None,
):
    if len(pair_i) == 0:
        return 0.0
    charge_product = charges[pair_i] * charges[pair_j]
    active = charge_product != 0.0
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    rij = displacements[active]
    charge_product = charge_product[active]

    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Coulomb pair distance is zero.")
    if cutoff2 is not None:
        active = r2 <= cutoff2
        if not np.any(active):
            return 0.0
        pair_i = pair_i[active]
        pair_j = pair_j[active]
        rij = rij[active]
        charge_product = charge_product[active]
        r2 = r2[active]

    distance = np.sqrt(r2)
    prefactor = coulomb_constant * charge_product
    reaction_field = _reaction_field_terms(cutoff, reaction_field_dielectric)
    if reaction_field is None:
        shift = 0.0 if cutoff is None or not energy_shift else prefactor / cutoff
        energy = prefactor / distance - shift
        force_scale = prefactor / (distance * r2)
    else:
        krf, crf = reaction_field
        energy = prefactor * (1.0 / distance + krf * r2 - crf)
        force_scale = prefactor * (1.0 / (distance * r2) - 2.0 * krf)
    fij = force_scale[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(energy))


def _reaction_field_terms(cutoff, dielectric):
    if dielectric is None:
        return None
    if cutoff is None:
        raise ValueError("reaction-field Coulomb requires a finite cutoff.")
    dielectric = float(dielectric)
    denominator = 2.0 * dielectric + 1.0
    krf = (dielectric - 1.0) / denominator / (float(cutoff) ** 3)
    crf = 3.0 * dielectric / denominator / float(cutoff)
    return krf, crf


def _scatter_pair_forces(forces, pair_i, pair_j, fij):
    natoms = len(forces)
    for axis in range(3):
        forces[:, axis] += np.bincount(pair_i, weights=fij[:, axis], minlength=natoms)
        forces[:, axis] -= np.bincount(pair_j, weights=fij[:, axis], minlength=natoms)


def _scatter_index_forces(forces, indices, values):
    natoms = len(forces)
    for axis in range(3):
        forces[:, axis] += np.bincount(indices, weights=values[:, axis], minlength=natoms)


def _nonexcluded_pair_mask(pair_i, pair_j, exclusions, natoms):
    if not exclusions:
        return np.ones(len(pair_i), dtype=bool)
    excluded_keys = np.fromiter(
        (_pair_key(i, j, natoms) for i, j in exclusions),
        dtype=np.int64,
        count=len(exclusions),
    )
    pair_keys = _pair_keys(pair_i, pair_j, natoms)
    return ~np.isin(pair_keys, excluded_keys)


def _pair_keys(pair_i, pair_j, natoms):
    lower = np.minimum(pair_i, pair_j).astype(np.int64, copy=False)
    upper = np.maximum(pair_i, pair_j).astype(np.int64, copy=False)
    return lower * int(natoms) + upper


def _pair_key(i, j, natoms):
    i = int(i)
    j = int(j)
    return min(i, j) * int(natoms) + max(i, j)


def _pair_scale_arrays(pair_scales, correction=False):
    if not pair_scales:
        return (
            np.asarray([], dtype=int),
            np.asarray([], dtype=int),
            np.asarray([], dtype=float),
        )
    pairs = np.asarray(list(pair_scales.keys()), dtype=int)
    scales = np.asarray(list(pair_scales.values()), dtype=float)
    if correction:
        scales = scales - 1.0
    return pairs[:, 0], pairs[:, 1], scales


def _pair_lj_parameter_arrays(pair_parameters):
    if not pair_parameters:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i, empty_f, empty_f
    pairs = np.asarray(list(pair_parameters.keys()), dtype=int)
    values = np.asarray(list(pair_parameters.values()), dtype=float)
    return pairs[:, 0], pairs[:, 1], values[:, 0], values[:, 1]


def _pair_float_arrays(pair_values):
    if not pair_values:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i, empty_f
    pairs = np.asarray(list(pair_values.keys()), dtype=int)
    values = np.asarray(list(pair_values.values()), dtype=float)
    return pairs[:, 0], pairs[:, 1], values


def _pair_index_arrays(pairs):
    pairs = np.asarray(list(pairs), dtype=int)
    if len(pairs) == 0:
        empty = np.asarray([], dtype=int)
        return empty, empty
    return pairs[:, 0], pairs[:, 1]


def _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j):
    rij = positions[pair_i] - positions[pair_j]
    lengths = _orthorhombic_lengths(cell)
    if lengths is not None:
        axes = np.nonzero(np.asarray(pbc, dtype=bool))[0]
        if len(axes) > 0:
            rij[:, axes] -= lengths[axes] * np.round(rij[:, axes] / lengths[axes])
        return rij
    return np.asarray([_minimum_image(vector, cell, pbc) for vector in rij], dtype=float)


def _add_coulomb_scaled_pairs(
    positions,
    cell,
    pbc,
    forces,
    charges,
    coulomb_constant,
    pair_scales,
    virial=None,
):
    pair_i, pair_j, correction_scale = _pair_scale_arrays(pair_scales, correction=True)
    if len(pair_i) == 0:
        return 0.0
    charge_product = charges[pair_i] * charges[pair_j]
    active = (correction_scale != 0.0) & (charge_product != 0.0)
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    correction_scale = correction_scale[active]
    charge_product = charge_product[active]
    rij = _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j)
    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Scaled Coulomb pair distance is zero.")
    distance = np.sqrt(r2)
    prefactor = correction_scale * coulomb_constant * charge_product
    energy = prefactor / distance
    fij = (prefactor / (distance * r2))[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(energy))


def _add_coulomb_specific_pairs(
    positions,
    cell,
    pbc,
    forces,
    coulomb_constant,
    pair_charge_products,
    virial=None,
):
    pair_i, pair_j, charge_products = _pair_float_arrays(pair_charge_products)
    if len(pair_i) == 0:
        return 0.0
    active = charge_products != 0.0
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    charge_products = charge_products[active]
    rij = _pair_displacements_for_indices(positions, cell, pbc, pair_i, pair_j)
    r2 = np.einsum("ij,ij->i", rij, rij)
    if np.any(r2 == 0.0):
        raise ValueError("Specific Coulomb pair distance is zero.")
    distance = np.sqrt(r2)
    prefactor = float(coulomb_constant) * charge_products
    energy = prefactor / distance
    fij = (prefactor / (distance * r2))[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(energy))


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
    real_pair_displacements=None,
    virial=None,
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
    energy = _add_ewald_real(
        positions,
        lengths,
        forces,
        charges,
        coulomb_constant,
        alpha,
        cutoff,
        pair_displacements=real_pair_displacements,
        virial=virial,
    )
    energy += _add_ewald_reciprocal(
        positions,
        lengths,
        volume,
        forces,
        charges,
        coulomb_constant,
        alpha,
        kmax,
        virial=virial,
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
    order=4,
    real_pair_displacements=None,
    virial=None,
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
    order = int(order)
    if order not in {2, 4}:
        raise ValueError("PME order must be 2 or 4.")

    cutoff = 0.5 * float(np.min(lengths)) if real_cutoff is None else float(real_cutoff)
    energy = _add_ewald_real(
        positions,
        lengths,
        forces,
        charges,
        coulomb_constant,
        alpha,
        cutoff,
        pair_displacements=real_pair_displacements,
        virial=virial,
    )
    energy += _add_pme_reciprocal(
        positions,
        lengths,
        forces,
        charges,
        coulomb_constant,
        alpha,
        mesh,
        order,
        virial=virial,
    )
    energy -= coulomb_constant * alpha / sqrt(np.pi) * float(np.dot(charges, charges))
    return energy


def _add_lj_pme_coulomb_shared(
    positions,
    cell,
    pbc,
    forces,
    lj_epsilon,
    lj_sigma,
    charges,
    coulomb_constant,
    alpha,
    real_cutoff,
    mesh,
    order,
    pair_displacements,
    lj_cutoff=None,
    lj_switch_on=None,
    lj_energy_shift=True,
    lj_exclusions=None,
    atom_types=None,
    pair_overrides=None,
    pair_override_lookup=None,
    pair_parameter_lookup=None,
    virial=None,
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
    order = int(order)
    if order not in {2, 4}:
        raise ValueError("PME order must be 2 or 4.")

    cutoff = 0.5 * float(np.min(lengths)) if real_cutoff is None else float(real_cutoff)
    energy = _add_lj_ewald_real_pair_arrays(
        forces,
        np.asarray(lj_epsilon, dtype=float),
        np.asarray(lj_sigma, dtype=float),
        np.asarray(charges, dtype=float),
        coulomb_constant,
        alpha,
        cutoff,
        cutoff * cutoff,
        pair_displacements.pair_i,
        pair_displacements.pair_j,
        pair_displacements.displacements,
        lj_cutoff=lj_cutoff,
        lj_switch_on=lj_switch_on,
        lj_energy_shift=lj_energy_shift,
        lj_exclusions=lj_exclusions,
        atom_types=atom_types,
        pair_overrides=pair_overrides,
        pair_override_lookup=pair_override_lookup,
        pair_parameter_lookup=pair_parameter_lookup,
        virial=virial,
    )
    energy += _add_pme_reciprocal(
        positions,
        lengths,
        forces,
        charges,
        coulomb_constant,
        alpha,
        mesh,
        order,
        virial=virial,
    )
    energy -= coulomb_constant * alpha / sqrt(np.pi) * float(np.dot(charges, charges))
    return energy


def _add_lj_ewald_real_pair_arrays(
    forces,
    epsilon,
    sigma,
    charges,
    coulomb_constant,
    alpha,
    cutoff,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    lj_cutoff=None,
    lj_switch_on=None,
    lj_energy_shift=True,
    lj_exclusions=None,
    atom_types=None,
    pair_overrides=None,
    pair_override_lookup=None,
    pair_parameter_lookup=None,
    virial=None,
):
    if len(pair_i) == 0:
        return 0.0
    rij = displacements
    r2 = np.einsum("ij,ij->i", rij, rij)
    active = (r2 > 0.0) & (r2 <= cutoff2)
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    rij = rij[active]
    r2 = r2[active]

    charge_product = charges[pair_i] * charges[pair_j]
    distance = np.sqrt(r2)
    prefactor = coulomb_constant * charge_product
    ar = alpha * distance
    erfc_ar = _array_erfc(ar)
    exp_ar2 = np.exp(-ar * ar)
    pair_energy = prefactor * erfc_ar / distance
    force_scalar = prefactor * (
        erfc_ar / (distance * r2)
        + 2.0 * alpha * exp_ar2 / (sqrt(np.pi) * r2)
    )
    fij = force_scalar[:, np.newaxis] * rij

    lj_active = np.ones(len(pair_i), dtype=bool)
    if lj_cutoff is not None and not np.isclose(lj_cutoff, cutoff, rtol=0.0, atol=1e-12):
        lj_active &= r2 <= lj_cutoff * lj_cutoff
    if lj_exclusions is not None:
        lj_active &= _nonexcluded_pair_mask(pair_i, pair_j, lj_exclusions, len(forces))
    if np.any(lj_active):
        lj_pair_i = pair_i[lj_active]
        lj_pair_j = pair_j[lj_active]
        lj_rij = rij[lj_active]
        lj_r2 = r2[lj_active]
        if pair_parameter_lookup is not None:
            type_codes, type_epsilon, type_sigma = pair_parameter_lookup
            code_i = type_codes[lj_pair_i]
            code_j = type_codes[lj_pair_j]
            epsilon_ij = type_epsilon[code_i, code_j]
            sigma_ij = type_sigma[code_i, code_j]
        else:
            epsilon_ij = np.sqrt(epsilon[lj_pair_i] * epsilon[lj_pair_j])
            sigma_ij = 0.5 * (sigma[lj_pair_i] + sigma[lj_pair_j])
            if pair_overrides:
                if pair_override_lookup is None:
                    _apply_lj_pair_overrides(
                        epsilon_ij,
                        sigma_ij,
                        lj_pair_i,
                        lj_pair_j,
                        atom_types,
                        pair_overrides,
                    )
                else:
                    _apply_lj_pair_override_lookup(
                        epsilon_ij,
                        sigma_ij,
                        lj_pair_i,
                        lj_pair_j,
                        pair_override_lookup,
                    )
        nonzero_lj = epsilon_ij != 0.0
        if np.any(nonzero_lj):
            lj_positions = np.nonzero(lj_active)[0][nonzero_lj]
            epsilon_ij = epsilon_ij[nonzero_lj]
            sigma_ij = sigma_ij[nonzero_lj]
            lj_rij = lj_rij[nonzero_lj]
            lj_r2 = lj_r2[nonzero_lj]
            inv_r2 = 1.0 / lj_r2
            sr2 = sigma_ij * sigma_ij * inv_r2
            sr6 = sr2 ** 3
            sr12 = sr6 * sr6
            lj_energy = 4.0 * epsilon_ij * (sr12 - sr6)
            lj_fij = (
                24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2
            )[:, np.newaxis] * lj_rij
            if lj_switch_on is not None:
                r = np.sqrt(lj_r2)
                switching = r > lj_switch_on
                if np.any(switching):
                    switch, dswitch_dr = _switching_function_arrays(
                        r[switching],
                        lj_switch_on,
                        lj_cutoff,
                    )
                    lj_fij[switching] = (
                        switch[:, np.newaxis] * lj_fij[switching]
                        - (lj_energy[switching] * dswitch_dr / r[switching])[:, np.newaxis]
                        * lj_rij[switching]
                    )
                    lj_energy[switching] *= switch
            shift = 0.0
            if lj_cutoff is not None and lj_energy_shift and lj_switch_on is None:
                sr6_cutoff = (sigma_ij / lj_cutoff) ** 6
                shift = 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
            pair_energy[lj_positions] += lj_energy - shift
            fij[lj_positions] += lj_fij

    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(pair_energy))


def _add_pme_reciprocal(
    positions,
    lengths,
    forces,
    charges,
    coulomb_constant,
    alpha,
    mesh,
    order,
    virial=None,
):
    charge_grid = _assign_charges_bspline(positions, charges, lengths, mesh, order)
    rho_hat = np.fft.fftn(charge_grid)
    kx, ky, kz, influence, grid_size = _pme_reciprocal_data(
        lengths,
        mesh,
        coulomb_constant,
        alpha,
        order,
    )
    potential_grid = np.fft.ifftn(grid_size * influence * rho_hat).real

    energy = 0.5 * float(np.sum(charge_grid * potential_grid))
    if virial is not None:
        _add_pme_reciprocal_virial(virial, rho_hat, kx, ky, kz, influence, alpha)
    reciprocal_forces = _reciprocal_assignment_forces(
        positions,
        charges,
        potential_grid,
        lengths,
        mesh,
        order,
    )
    reciprocal_forces -= np.mean(reciprocal_forces, axis=0)
    forces += reciprocal_forces
    return energy


def _add_pme_reciprocal_virial(virial, rho_hat, kx, ky, kz, influence, alpha):
    k2 = kx * kx + ky * ky + kz * kz
    mask = k2 > 0.0
    mode_energy = 0.5 * influence[mask] * np.abs(rho_hat[mask]) ** 2
    factor = 0.25 / (alpha * alpha) + 1.0 / k2[mask]
    virial += float(np.sum(mode_energy)) * np.eye(3)
    for axis_a, ka in enumerate((kx[mask], ky[mask], kz[mask])):
        for axis_b, kb in enumerate((kx[mask], ky[mask], kz[mask])):
            virial[axis_a, axis_b] -= 2.0 * float(np.sum(mode_energy * factor * ka * kb))


def pme_reciprocal_potential_grid(
    positions,
    charges,
    cell,
    pbc=True,
    coulomb_constant=1.0,
    alpha=0.35,
    mesh=(16, 16, 16),
    order=4,
):
    """Return the smooth reciprocal-space PME potential on the PME mesh.

    The returned grid uses the same B-spline assignment and reciprocal
    influence function as :class:`PMECoulomb`.  The zero Fourier mode is
    omitted, so the potential follows the usual neutral-cell PME convention.
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
    order = int(order)
    if order not in {2, 4}:
        raise ValueError("PME order must be 2 or 4.")

    charge_grid = _assign_charges_bspline(positions, charges, lengths, mesh, order)
    rho_hat = np.fft.fftn(charge_grid)
    _, _, _, influence, grid_size = _pme_reciprocal_data(
        lengths,
        mesh,
        coulomb_constant,
        alpha,
        order,
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
    order=4,
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
        order=order,
    )
    return _interpolate_bspline(points, potential_grid, lengths, np.asarray(mesh, dtype=int), int(order))


def _pme_reciprocal_data(lengths, mesh, coulomb_constant, alpha, order=4):
    lengths = np.asarray(lengths, dtype=float)
    mesh = np.asarray(mesh, dtype=int)
    key = (
        tuple(np.round(lengths, 12)),
        tuple(mesh.tolist()),
        float(coulomb_constant),
        float(alpha),
        int(order),
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
    influence[mask] /= _bspline_deconvolution(mesh, int(order))[mask]
    cached = (kx, ky, kz, influence, grid_size)
    _PME_RECIPROCAL_CACHE[key] = cached
    return cached


def _assign_charges_bspline(positions, charges, lengths, mesh, order):
    grid = np.zeros(tuple(mesh), dtype=float)
    if len(positions) == 0:
        return grid
    indices, weights, _derivatives = _bspline_stencils(positions, lengths, mesh, order)
    ix, iy, iz = indices
    wx, wy, wz = weights
    for ax in range(order):
        for ay in range(order):
            xy_weight = charges * wx[:, ax] * wy[:, ay]
            for az in range(order):
                np.add.at(
                    grid,
                    (ix[:, ax], iy[:, ay], iz[:, az]),
                    xy_weight * wz[:, az],
                )
    return grid


def _assign_charges_cic(positions, charges, lengths, mesh):
    return _assign_charges_bspline(positions, charges, lengths, mesh, order=2)


def _interpolate_bspline(positions, grid, lengths, mesh, order):
    values = np.zeros(len(positions), dtype=float)
    if len(positions) == 0:
        return values
    indices, weights, _derivatives = _bspline_stencils(positions, lengths, mesh, order)
    ix, iy, iz = indices
    wx, wy, wz = weights
    for ax in range(order):
        for ay in range(order):
            xy_weight = wx[:, ax] * wy[:, ay]
            for az in range(order):
                values += xy_weight * wz[:, az] * grid[ix[:, ax], iy[:, ay], iz[:, az]]
    return values


def _interpolate_cic(positions, grid, lengths, mesh):
    return _interpolate_bspline(positions, grid, lengths, mesh, order=2)


def _reciprocal_assignment_forces(positions, charges, potential_grid, lengths, mesh, order):
    forces = np.zeros_like(positions, dtype=float)
    if len(positions) == 0:
        return forces
    indices, weights, derivatives = _bspline_stencils(positions, lengths, mesh, order)
    ix, iy, iz = indices
    wx, wy, wz = weights
    dwx, dwy, dwz = derivatives
    scale = mesh / lengths
    gradient = np.zeros_like(positions, dtype=float)
    for ax in range(order):
        for ay in range(order):
            for az in range(order):
                phi = potential_grid[ix[:, ax], iy[:, ay], iz[:, az]]
                gradient[:, 0] += dwx[:, ax] * scale[0] * wy[:, ay] * wz[:, az] * phi
                gradient[:, 1] += wx[:, ax] * dwy[:, ay] * scale[1] * wz[:, az] * phi
                gradient[:, 2] += wx[:, ax] * wy[:, ay] * dwz[:, az] * scale[2] * phi
    forces -= charges[:, np.newaxis] * gradient
    return forces


def _bspline_stencil(position, lengths, mesh, order):
    indices, weights, derivatives = _bspline_stencils(
        np.asarray(position, dtype=float).reshape(1, 3),
        lengths,
        mesh,
        order,
    )
    indices = [axis_indices[0] for axis_indices in indices]
    weights = [axis_weights[0] for axis_weights in weights]
    derivatives = [axis_derivatives[0] for axis_derivatives in derivatives]
    return indices, weights, derivatives


def _bspline_stencils(positions, lengths, mesh, order):
    positions = np.asarray(positions, dtype=float)
    lengths = np.asarray(lengths, dtype=float)
    mesh = np.asarray(mesh, dtype=int)
    scaled = np.mod(positions / lengths, 1.0) * mesh
    base = np.floor(scaled).astype(int)
    frac = scaled - base

    indices = []
    weights = []
    derivatives = []
    for axis in range(3):
        axis_weights, axis_derivatives, offsets = _bspline_weights_1d_arrays(
            frac[:, axis],
            int(order),
        )
        indices.append((base[:, [axis]] + offsets[np.newaxis, :]) % mesh[axis])
        weights.append(axis_weights)
        derivatives.append(axis_derivatives)
    return indices, weights, derivatives


def _bspline_weights_1d(frac, order):
    frac = float(frac)
    if order == 2:
        return (
            np.array([1.0 - frac, frac], dtype=float),
            np.array([-1.0, 1.0], dtype=float),
            np.array([0, 1], dtype=int),
        )
    if order == 4:
        one_minus = 1.0 - frac
        weights = np.array(
            [
                one_minus**3 / 6.0,
                (3.0 * frac**3 - 6.0 * frac**2 + 4.0) / 6.0,
                (-3.0 * frac**3 + 3.0 * frac**2 + 3.0 * frac + 1.0) / 6.0,
                frac**3 / 6.0,
            ],
            dtype=float,
        )
        derivatives = np.array(
            [
                -0.5 * one_minus**2,
                1.5 * frac**2 - 2.0 * frac,
                -1.5 * frac**2 + frac + 0.5,
                0.5 * frac**2,
            ],
            dtype=float,
        )
        return weights, derivatives, np.array([-1, 0, 1, 2], dtype=int)
    raise ValueError("PME order must be 2 or 4.")


def _bspline_weights_1d_arrays(frac, order):
    frac = np.asarray(frac, dtype=float)
    if order == 2:
        return (
            np.column_stack((1.0 - frac, frac)),
            np.tile(np.array([-1.0, 1.0], dtype=float), (len(frac), 1)),
            np.array([0, 1], dtype=int),
        )
    if order == 4:
        one_minus = 1.0 - frac
        weights = np.column_stack(
            (
                one_minus**3 / 6.0,
                (3.0 * frac**3 - 6.0 * frac**2 + 4.0) / 6.0,
                (-3.0 * frac**3 + 3.0 * frac**2 + 3.0 * frac + 1.0) / 6.0,
                frac**3 / 6.0,
            )
        )
        derivatives = np.column_stack(
            (
                -0.5 * one_minus**2,
                1.5 * frac**2 - 2.0 * frac,
                -1.5 * frac**2 + frac + 0.5,
                0.5 * frac**2,
            )
        )
        return weights, derivatives, np.array([-1, 0, 1, 2], dtype=int)
    raise ValueError("PME order must be 2 or 4.")


def _bspline_deconvolution(mesh, order):
    factors = []
    for n in mesh:
        modes = np.fft.fftfreq(int(n)) * int(n)
        x = np.pi * modes / int(n)
        sinc = np.ones_like(x, dtype=float)
        mask = np.abs(x) > 0.0
        sinc[mask] = np.sin(x[mask]) / x[mask]
        axis_factor = sinc ** (2 * order)
        axis_factor = np.maximum(axis_factor, 1.0e-12)
        factors.append(axis_factor)
    fx, fy, fz = np.meshgrid(factors[0], factors[1], factors[2], indexing="ij")
    return fx * fy * fz


def _add_ewald_real(
    positions,
    lengths,
    forces,
    charges,
    coulomb_constant,
    alpha,
    cutoff,
    pair_displacements=None,
    virial=None,
):
    if cutoff <= 0.5 * float(np.min(lengths)):
        return _add_ewald_real_minimum_image(
            positions,
            lengths,
            forces,
            charges,
            coulomb_constant,
            alpha,
            cutoff,
            pair_displacements=pair_displacements,
            virial=virial,
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
                        _accumulate_pair_virial(virial, rij, fij)
                if nx == 0 and ny == 0 and nz == 0:
                    continue
                for i in range(natoms):
                    rij = shift
                    r2 = float(np.dot(rij, rij))
                    if r2 == 0.0 or r2 > cutoff2:
                        continue
                    distance = sqrt(r2)
                    prefactor = 0.5 * coulomb_constant * charges[i] * charges[i]
                    energy += prefactor * erfc(alpha * distance) / distance
                    force_scalar = prefactor * (
                        erfc(alpha * distance) / (distance * r2)
                        + 2.0 * alpha * np.exp(-(alpha * distance) ** 2) / (sqrt(np.pi) * r2)
                    )
                    _accumulate_pair_virial(virial, rij, force_scalar * rij)
    return energy


def _add_ewald_real_minimum_image(
    positions,
    lengths,
    forces,
    charges,
    coulomb_constant,
    alpha,
    cutoff,
    pair_displacements=None,
    virial=None,
):
    energy = 0.0
    pairs = pair_displacements
    if pairs is None:
        pairs = _candidate_pair_displacements(
            positions,
            np.diag(lengths),
            np.ones(3, dtype=bool),
            cutoff,
        )
    cutoff2 = cutoff * cutoff
    if isinstance(pairs, _PairDisplacements):
        return _add_ewald_real_pair_arrays(
            forces,
            charges,
            coulomb_constant,
            alpha,
            cutoff2,
            pairs.pair_i,
            pairs.pair_j,
            pairs.displacements,
            virial=virial,
        )
    for i, j, rij in pairs:
        r2 = float(np.dot(rij, rij))
        if r2 == 0.0:
            continue
        if r2 > cutoff2:
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
        _accumulate_pair_virial(virial, rij, fij)
    return energy


def _add_ewald_real_pair_arrays(
    forces,
    charges,
    coulomb_constant,
    alpha,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    virial=None,
):
    if len(pair_i) == 0:
        return 0.0
    rij = displacements
    r2 = np.einsum("ij,ij->i", rij, rij)
    active = (r2 > 0.0) & (r2 <= cutoff2)
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    rij = rij[active]
    r2 = r2[active]

    charge_product = charges[pair_i] * charges[pair_j]
    active = charge_product != 0.0
    if not np.any(active):
        return 0.0
    pair_i = pair_i[active]
    pair_j = pair_j[active]
    rij = rij[active]
    r2 = r2[active]
    charge_product = charge_product[active]

    distance = np.sqrt(r2)
    prefactor = coulomb_constant * charge_product
    ar = alpha * distance
    erfc_ar = _array_erfc(ar)
    exp_ar2 = np.exp(-ar * ar)
    energy = prefactor * erfc_ar / distance
    force_scalar = prefactor * (
        erfc_ar / (distance * r2)
        + 2.0 * alpha * exp_ar2 / (sqrt(np.pi) * r2)
    )
    fij = force_scalar[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return float(np.sum(energy))


def _add_ewald_reciprocal(
    positions,
    lengths,
    volume,
    forces,
    charges,
    coulomb_constant,
    alpha,
    kmax,
    virial=None,
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
                if virial is not None:
                    term_energy = coefficient * weight * (
                        structure_cos * structure_cos + structure_sin * structure_sin
                    )
                    virial += term_energy * (
                        np.eye(3)
                        - 2.0
                        * np.outer(kvec, kvec)
                        * (0.25 / (alpha * alpha) + 1.0 / k2)
                    )
                force_coefficient = 2.0 * coefficient * weight
                phase_force = structure_cos * sin_phase - structure_sin * cos_phase
                forces += (force_coefficient * charges * phase_force)[:, None] * kvec
    return energy
