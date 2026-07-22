"""Classical calculators for :mod:`pyqed.md`."""

from math import erfc, exp, sqrt

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
_LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA = None
_LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA_UNAVAILABLE = False
_LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA = None
_LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA_UNAVAILABLE = False
_LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_MIN_PAIRS = 200_000
_PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA = None
_PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA_UNAVAILABLE = False
_NONEXCLUDED_PAIR_MASK_NUMBA = None
_NONEXCLUDED_PAIR_MASK_NUMBA_UNAVAILABLE = False
_ASSIGN_CHARGES_BSPLINE_NUMBA = None
_ASSIGN_CHARGES_BSPLINE_NUMBA_UNAVAILABLE = False
_RECIPROCAL_ASSIGNMENT_FORCES_NUMBA = None
_RECIPROCAL_ASSIGNMENT_FORCES_NUMBA_UNAVAILABLE = False
_FILL_BSPLINE_STENCIL_NUMBA = None
_TORSION_ARRAYS_NUMBA = None
_TORSION_ARRAYS_NUMBA_UNAVAILABLE = False
_numba_prange = range
_numba_get_num_threads = lambda: 1
_numba_get_thread_id = lambda: 0
_SUPPORTED_PME_ORDERS = frozenset(range(2, 9))


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


def _validate_pme_order(order):
    order = int(order)
    if order not in _SUPPORTED_PME_ORDERS:
        raise ValueError("PME order must be an integer from 2 through 8.")
    return order


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
        (
            self._improper_indices,
            self._improper_force_constants,
            self._improper_phases,
        ) = _improper_arrays(self.impropers)
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
        self._lj_pair_parameter_arrays = _pair_lj_parameter_arrays(self.lj_pair_parameters)
        self._coulomb_pair_parameter_arrays = _pair_float_arrays(self.coulomb_pair_parameters)
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
        common_exclusions = set()
        if self.exclude_bonded:
            common_exclusions.update(self._bonded_pairs)
        if self.exclude_angles:
            common_exclusions.update(self._angle_pairs)
        common_exclusions.update(self.nonbonded_exclusions)
        base_lj_exclusions = set(common_exclusions)
        base_lj_exclusions.update(self.lj_exclusions)
        base_coulomb_exclusions = set(common_exclusions)
        base_coulomb_exclusions.update(self.coulomb_exclusions)
        self._base_lj_nonbonded_exclusions = base_lj_exclusions or None
        self._base_coulomb_nonbonded_exclusions = base_coulomb_exclusions or None
        base_coulomb_exclusions = self._base_coulomb_nonbonded_exclusions or set()
        self._base_coulomb_with_pair_parameter_exclusions = (
            set(base_coulomb_exclusions) | set(self.coulomb_pair_parameters)
        ) or None
        self._coulomb_exclusion_pair_arrays = _pair_index_arrays(base_coulomb_exclusions)
        self._pme_coulomb_exclusion_pair_arrays = _pair_index_arrays(
            self._base_coulomb_with_pair_parameter_exclusions or set()
        )
        self._shared_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self._lj_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self._coulomb_pair_displacement_cache = _PairDisplacementCache(self.nonbonded_skin)
        self._nonbonded_exclusion_key_cache = {}
        self._calculation_cache = None
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

    def nonbonded_energy_components(self, atoms=None):
        """Return diagnostic native nonbonded energy components.

        The production nonbonded evaluator sometimes combines Coulomb PME real
        space and Lennard-Jones pair loops for speed.  This helper reruns the
        nonbonded energy with one interaction family disabled at a time so
        parity harnesses can localize OpenMM-vs-native residuals without
        changing the force path.
        """
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MolecularMechanics calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        zero_forces = np.zeros_like(positions)
        total = self._add_nonbonded(atoms, positions, cell, pbc, zero_forces)

        coulomb = self._nonbonded_energy_with_overrides(
            atoms,
            positions,
            cell,
            pbc,
            {
                "lj_epsilon": None,
                "lj_sigma": None,
                "lj_pair_scales": {},
                "lj_pair_parameters": {},
                "_lj_pair_parameter_arrays": _pair_lj_parameter_arrays({}),
                "lj_pair_overrides": {},
                "_lj_pair_override_lookup": None,
                "_lj_type_pair_parameter_lookup": None,
            },
        )
        lj = self._nonbonded_energy_with_overrides(
            atoms,
            positions,
            cell,
            pbc,
            {
                "charges": np.zeros(len(atoms), dtype=float),
                "coulomb_pair_scales": {},
                "coulomb_pair_parameters": {},
                "_coulomb_pair_parameter_arrays": _pair_float_arrays({}),
                "_base_coulomb_with_pair_parameter_exclusions": self._base_coulomb_nonbonded_exclusions,
                "_pme_coulomb_exclusion_pair_arrays": _pair_index_arrays(
                    self._base_coulomb_nonbonded_exclusions or set()
                ),
            },
        )
        coulomb_terms = self._coulomb_energy_breakdown(atoms, positions, cell, pbc)
        return {
            "total": float(total),
            "coulomb": float(coulomb),
            "lj": float(lj),
            "residual": float(total - coulomb - lj),
            "coulomb_terms": coulomb_terms,
        }

    def _nonbonded_energy_with_overrides(self, atoms, positions, cell, pbc, overrides):
        old_values = {name: getattr(self, name) for name in overrides}
        try:
            for name, value in overrides.items():
                setattr(self, name, value)
            forces = np.zeros_like(positions)
            return self._add_nonbonded(atoms, positions, cell, pbc, forces)
        finally:
            for name, value in old_values.items():
                setattr(self, name, value)

    def _coulomb_energy_breakdown(self, atoms, positions, cell, pbc):
        charges = self._charges(atoms)
        if charges is None:
            return None
        forces = np.zeros_like(positions)
        terms = {
            "method": self.coulomb_method,
            "atoms": int(len(positions)),
            "charge_squared_sum": float(np.dot(charges, charges)),
            "real": 0.0,
            "reciprocal": 0.0,
            "self": 0.0,
            "exclusion_correction": 0.0,
            "specific_pairs": 0.0,
            "scaled_pairs": 0.0,
        }
        if self.coulomb_method == "pme":
            terms.update(
                _pme_coulomb_energy_terms(
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
                    real_pair_displacements=None,
                )
            )
            exclusions = self._base_coulomb_with_pair_parameter_exclusions
            if exclusions is not None:
                terms["exclusion_correction"] = _add_coulomb_pairs(
                    positions,
                    cell,
                    pbc,
                    np.zeros_like(positions),
                    charges,
                    self.coulomb_constant,
                    None,
                    None,
                    only_pairs=self._pme_coulomb_exclusion_pair_arrays,
                    sign=-1.0,
                )
        elif self.coulomb_method == "ewald":
            terms["total_without_exceptions"] = _add_ewald_coulomb(
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
            exclusions = self._base_coulomb_nonbonded_exclusions
            if exclusions is not None:
                terms["exclusion_correction"] = _add_coulomb_pairs(
                    positions,
                    cell,
                    pbc,
                    np.zeros_like(positions),
                    charges,
                    self.coulomb_constant,
                    None,
                    None,
                    only_pairs=self._coulomb_exclusion_pair_arrays,
                    sign=-1.0,
                )
        else:
            terms["real"] = _add_coulomb_pairs(
                positions,
                cell,
                pbc,
                forces,
                charges,
                self.coulomb_constant,
                self.coulomb_cutoff,
                self._base_coulomb_nonbonded_exclusions,
                energy_shift=self.coulomb_energy_shift,
                reaction_field_dielectric=self.coulomb_reaction_field_dielectric,
            )
        if self.coulomb_pair_scales:
            terms["scaled_pairs"] = _add_coulomb_scaled_pairs(
                positions,
                cell,
                pbc,
                np.zeros_like(positions),
                charges,
                self.coulomb_constant,
                self.coulomb_pair_scales,
            )
        if self.coulomb_pair_parameters:
            terms["specific_pairs"] = _add_coulomb_specific_pairs(
                positions,
                cell,
                pbc,
                np.zeros_like(positions),
                self.coulomb_constant,
                self._coulomb_pair_parameter_arrays,
            )
        terms["total"] = float(
            terms.get("real", 0.0)
            + terms.get("reciprocal", 0.0)
            + terms.get("self", 0.0)
            + terms.get("exclusion_correction", 0.0)
            + terms.get("scaled_pairs", 0.0)
            + terms.get("specific_pairs", 0.0)
        )
        return terms

    def calculate(self, atoms=None, extra_lj_exclusions=None, extra_coulomb_exclusions=None):
        if atoms is None:
            atoms = self.atoms
        if atoms is None:
            raise RuntimeError("MolecularMechanics calculator has no atoms.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()
        charges = self._charges(atoms)
        cache_key = self._calculation_cache_key(
            positions,
            cell,
            pbc,
            charges,
            extra_lj_exclusions,
            extra_coulomb_exclusions,
        )
        cached = self._calculation_cache
        if self._calculation_cache_matches(cached, cache_key, positions, cell, pbc, charges):
            return cached["energy"], cached["forces"].copy()

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
        self._calculation_cache = {
            "key": cache_key,
            "positions": positions.copy(),
            "cell": np.asarray(cell, dtype=float).copy(),
            "pbc": np.asarray(pbc, dtype=bool).copy(),
            "charges": None if charges is None else np.asarray(charges, dtype=float).copy(),
            "energy": float(energy),
            "forces": forces.copy(),
        }
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
        return _add_improper_arrays(
            positions,
            cell,
            pbc,
            forces,
            self._improper_indices,
            self._improper_force_constants,
            self._improper_phases,
        )

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
        lj_exclusion_keys = self._nonbonded_exclusion_keys(
            "lj",
            extra_lj_exclusions,
            len(positions),
        )
        if charges is not None and self.coulomb_pair_parameters:
            if extra_coulomb_exclusions is None:
                coulomb_exclusions = self._base_coulomb_with_pair_parameter_exclusions
            else:
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
                lj_exclusion_keys=lj_exclusion_keys,
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
                    exclusion_keys=lj_exclusion_keys,
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
                    self._lj_pair_parameter_arrays,
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
                    energy += _add_coulomb_pairs(
                        positions,
                        cell,
                        pbc,
                        forces,
                        charges,
                        self.coulomb_constant,
                        None,
                        None,
                        only_pairs=self._coulomb_exclusion_pair_arrays
                        if extra_coulomb_exclusions is None
                        else coulomb_exclusions,
                        sign=-1.0,
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
                    energy += _add_coulomb_pairs(
                        positions,
                        cell,
                        pbc,
                        forces,
                        charges,
                        self.coulomb_constant,
                        None,
                        None,
                        only_pairs=self._pme_coulomb_exclusion_pair_arrays
                        if extra_coulomb_exclusions is None
                        else coulomb_exclusions,
                        sign=-1.0,
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
                    self._coulomb_pair_parameter_arrays,
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

    @staticmethod
    def _calculation_cache_key(
        positions,
        cell,
        pbc,
        charges,
        extra_lj_exclusions,
        extra_coulomb_exclusions,
    ):
        charge_state = None if charges is None else (len(charges), str(np.asarray(charges).dtype))
        return (
            len(positions),
            np.asarray(positions, dtype=float).shape,
            np.asarray(cell, dtype=float).shape,
            tuple(np.asarray(pbc, dtype=bool).tolist()),
            charge_state,
            _exclusion_cache_key(extra_lj_exclusions),
            _exclusion_cache_key(extra_coulomb_exclusions),
        )

    @staticmethod
    def _calculation_cache_matches(cache, key, positions, cell, pbc, charges):
        if cache is None or cache["key"] != key:
            return False
        if not np.array_equal(cache["positions"], positions):
            return False
        if not np.array_equal(cache["cell"], np.asarray(cell, dtype=float)):
            return False
        if not np.array_equal(cache["pbc"], np.asarray(pbc, dtype=bool)):
            return False
        cached_charges = cache["charges"]
        if cached_charges is None or charges is None:
            return cached_charges is None and charges is None
        return np.array_equal(cached_charges, np.asarray(charges, dtype=float))

    def _nonbonded_exclusions(self, kind, extra_exclusions=None):
        if kind == "lj":
            base = self._base_lj_nonbonded_exclusions
        elif kind == "coulomb":
            base = self._base_coulomb_nonbonded_exclusions
        else:
            raise ValueError("kind must be 'lj' or 'coulomb'.")
        if extra_exclusions is None:
            return base
        exclusions = set() if base is None else set(base)
        exclusions.update(_pair_set(extra_exclusions))
        return exclusions or None

    def _nonbonded_exclusion_keys(self, kind, extra_exclusions, natoms):
        if extra_exclusions:
            return None
        key = (kind, int(natoms))
        cached = self._nonbonded_exclusion_key_cache.get(key)
        if cached is not None:
            return cached
        exclusions = self._nonbonded_exclusions(kind)
        cached = _pair_key_array(exclusions, natoms)
        self._nonbonded_exclusion_key_cache[key] = cached
        return cached


class MM(MolecularMechanics):
    """Short public name for :class:`MolecularMechanics`."""


class _PairDisplacementCache:
    """Reuse cutoff+skin pair lists across nearby MD steps."""

    def __init__(self, skin):
        self.skin = float(skin)
        self._key = None
        self._reference_positions = None
        self._reference_cell = None
        self._pair_i = None
        self._pair_j = None
        self._displacements = None
        self._second_positions = None
        self._mask_cache = {}
        self.rebuild_count = 0
        self.reuse_count = 0
        self.rebuild_reasons = {}
        self.last_rebuild_reason = None
        self.last_max_displacement = None
        self.max_reference_displacement = 0.0
        self.cell_reuse_count = 0
        self.last_cell_delta = None
        self.max_reference_cell_delta = 0.0

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
        key_reason = self._key_change_reason(key)
        needs_position_rebuild, max_displacement = self._needs_rebuild(
            positions,
            cell,
            pbc,
            lengths,
        )
        self.last_max_displacement = max_displacement
        if max_displacement is not None:
            self.max_reference_displacement = max(
                self.max_reference_displacement,
                max_displacement,
            )
        cell_reused = False
        rebuild_reason = key_reason
        if key_reason == "cell" and not needs_position_rebuild:
            can_reuse_cell, cell_delta = self._can_reuse_cell_change(
                cell,
                pbc,
                lengths,
                max_displacement,
            )
            self.last_cell_delta = cell_delta
            if cell_delta is not None:
                self.max_reference_cell_delta = max(
                    self.max_reference_cell_delta,
                    cell_delta,
                )
            if can_reuse_cell:
                rebuild_reason = None
                cell_reused = True
        if rebuild_reason is None and needs_position_rebuild:
            rebuild_reason = "displacement"
        if rebuild_reason is not None:
            self._pair_i, self._pair_j, _rij = _candidate_pair_displacement_arrays(
                positions,
                cell,
                pbc,
                float(cutoff) + self.skin,
                exclusions,
            )
            self._key = key
            self._reference_positions = positions.copy()
            self._reference_cell = cell.copy()
            self._displacements = None
            self._second_positions = None
            self._mask_cache = {}
            self.rebuild_count += 1
            self.last_rebuild_reason = rebuild_reason
            self.rebuild_reasons[rebuild_reason] = self.rebuild_reasons.get(rebuild_reason, 0) + 1
        else:
            self.reuse_count += 1
            if cell_reused:
                self.cell_reuse_count += 1
                self._key = key

        if len(self._pair_i) == 0:
            return _PairDisplacements(
                np.asarray([], dtype=int),
                np.asarray([], dtype=int),
                np.zeros((0, 3), dtype=float),
                self._mask_cache,
            )

        shape = (len(self._pair_i), 3)
        if self._displacements is None or self._displacements.shape != shape:
            self._displacements = np.empty(shape, dtype=float)
            self._second_positions = np.empty(shape, dtype=float)
        rij = self._displacements
        if lengths is not None:
            if not _fill_orthorhombic_pair_displacements_numba(
                rij,
                positions,
                self._pair_i,
                self._pair_j,
                lengths,
                pbc,
            ):
                np.take(positions, self._pair_i, axis=0, out=rij)
                np.take(positions, self._pair_j, axis=0, out=self._second_positions)
                rij -= self._second_positions
                axes = np.nonzero(pbc)[0]
                if len(axes) > 0:
                    rij[:, axes] -= lengths[axes] * np.round(rij[:, axes] / lengths[axes])
        else:
            np.take(positions, self._pair_i, axis=0, out=rij)
            np.take(positions, self._pair_j, axis=0, out=self._second_positions)
            rij -= self._second_positions
            rij = np.array([_minimum_image(vector, cell, pbc) for vector in rij])
        return _PairDisplacements(self._pair_i, self._pair_j, rij, self._mask_cache)

    def _needs_rebuild(self, positions, cell, pbc, lengths):
        if self._reference_positions is None:
            return True, None
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
        max_displacement = float(np.sqrt(max_displacement2))
        return max_displacement2 > (0.5 * self.skin) ** 2, max_displacement

    def _key_change_reason(self, key):
        if self._key is None:
            return "initial"
        if key == self._key:
            return None
        names = ("natoms", "cell", "pbc", "cutoff", "exclusions")
        for index, name in enumerate(names):
            if key[index] != self._key[index]:
                return name
        return "key"

    def _can_reuse_cell_change(self, cell, pbc, lengths, max_displacement):
        if self._reference_cell is None or lengths is None:
            return False, None
        reference_lengths = _orthorhombic_lengths(self._reference_cell)
        if reference_lengths is None:
            return False, None
        axes = np.nonzero(np.asarray(pbc, dtype=bool))[0]
        if len(axes) == 0:
            return True, 0.0
        cell_delta = float(np.max(np.abs(lengths[axes] - reference_lengths[axes])))
        displacement = 0.0 if max_displacement is None else float(max_displacement)
        return 2.0 * displacement + cell_delta <= self.skin, cell_delta

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
    def __init__(self, pair_i, pair_j, displacements, mask_cache=None):
        self.pair_i = pair_i
        self.pair_j = pair_j
        self.displacements = displacements
        self._mask_cache = {} if mask_cache is None else mask_cache

    def __iter__(self):
        return iter(zip(self.pair_i, self.pair_j, self.displacements))

    def nonexcluded_mask(self, excluded_keys, natoms):
        if excluded_keys is None:
            return None
        excluded_keys = np.asarray(excluded_keys, dtype=np.int64)
        key = (
            int(natoms),
            len(excluded_keys),
            None if len(excluded_keys) == 0 else int(excluded_keys[0]),
            None if len(excluded_keys) == 0 else int(excluded_keys[-1]),
        )
        cached = self._mask_cache.get(key)
        if cached is None:
            cached = _nonexcluded_pair_mask(
                self.pair_i,
                self.pair_j,
                None,
                natoms,
                excluded_keys=excluded_keys,
            )
            self._mask_cache[key] = cached
        return cached


def _fill_orthorhombic_pair_displacements_numba(
    output,
    positions,
    pair_i,
    pair_j,
    lengths,
    pbc,
):
    kernel = _pair_displacements_orthorhombic_numba()
    if kernel is None:
        return False
    kernel(
        output,
        np.asarray(positions, dtype=float),
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(lengths, dtype=float),
        np.asarray(pbc, dtype=np.bool_),
    )
    return True


def _pair_displacements_orthorhombic_numba():
    global _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA, _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA_UNAVAILABLE
    if _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA_UNAVAILABLE:
        return None
    if _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA_UNAVAILABLE = True
            return None
        try:
            _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA = njit(cache=True, fastmath=True)(
                _pair_displacements_orthorhombic_numba_impl
            )
        except Exception:
            _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA_UNAVAILABLE = True
            return None
    return _PAIR_DISPLACEMENTS_ORTHORHOMBIC_NUMBA


def _pair_displacements_orthorhombic_numba_impl(output, positions, pair_i, pair_j, lengths, pbc):
    for index in range(len(pair_i)):
        i = pair_i[index]
        j = pair_j[index]
        for axis in range(3):
            value = positions[i, axis] - positions[j, axis]
            if pbc[axis]:
                value -= lengths[axis] * round(value / lengths[axis])
            output[index, axis] = value


def _pair_set(pairs):
    if pairs is None:
        return set()
    return {tuple(sorted((int(i), int(j)))) for i, j in pairs}


def _exclusion_cache_key(pairs):
    if pairs is None:
        return None
    return tuple(sorted(_pair_set(pairs)))


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


def _improper_arrays(impropers):
    if not impropers:
        return (
            np.zeros((0, 4), dtype=int),
            np.asarray([], dtype=float),
            np.asarray([], dtype=float),
        )
    values = np.asarray(impropers, dtype=float)
    return values[:, :4].astype(int), values[:, 4].astype(float), values[:, 5].astype(float)


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
    fast_energy = _try_add_torsion_arrays_numba(
        positions,
        cell,
        pbc,
        forces,
        indices,
        barriers,
        periodicities,
        phases,
    )
    if fast_energy is not None:
        return fast_energy
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


def _try_add_torsion_arrays_numba(
    positions,
    cell,
    pbc,
    forces,
    indices,
    barriers,
    periodicities,
    phases,
):
    lengths = _orthorhombic_lengths(cell)
    if lengths is None:
        return None
    kernel = _torsion_arrays_numba()
    if kernel is None:
        return None
    return float(
        kernel(
            np.asarray(positions, dtype=float),
            np.asarray(lengths, dtype=float),
            np.asarray(pbc, dtype=np.bool_),
            forces,
            np.asarray(indices, dtype=np.int64),
            np.asarray(barriers, dtype=float),
            np.asarray(periodicities, dtype=np.int64),
            np.asarray(phases, dtype=float),
        )
    )


def _torsion_arrays_numba():
    global _TORSION_ARRAYS_NUMBA, _TORSION_ARRAYS_NUMBA_UNAVAILABLE
    if _TORSION_ARRAYS_NUMBA_UNAVAILABLE:
        return None
    if _TORSION_ARRAYS_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _TORSION_ARRAYS_NUMBA_UNAVAILABLE = True
            return None
        try:
            _TORSION_ARRAYS_NUMBA = njit(cache=True, fastmath=True)(_torsion_arrays_numba_impl)
        except Exception:
            _TORSION_ARRAYS_NUMBA_UNAVAILABLE = True
            return None
    return _TORSION_ARRAYS_NUMBA


def _torsion_arrays_numba_impl(positions, lengths, pbc, forces, indices, barriers, periodicities, phases):
    energy = 0.0
    for row in range(len(indices)):
        i = indices[row, 0]
        j = indices[row, 1]
        k = indices[row, 2]
        l = indices[row, 3]
        b1x = positions[j, 0] - positions[i, 0]
        b1y = positions[j, 1] - positions[i, 1]
        b1z = positions[j, 2] - positions[i, 2]
        b2x = positions[k, 0] - positions[j, 0]
        b2y = positions[k, 1] - positions[j, 1]
        b2z = positions[k, 2] - positions[j, 2]
        b3x = positions[l, 0] - positions[k, 0]
        b3y = positions[l, 1] - positions[k, 1]
        b3z = positions[l, 2] - positions[k, 2]
        if pbc[0]:
            b1x -= lengths[0] * round(b1x / lengths[0])
            b2x -= lengths[0] * round(b2x / lengths[0])
            b3x -= lengths[0] * round(b3x / lengths[0])
        if pbc[1]:
            b1y -= lengths[1] * round(b1y / lengths[1])
            b2y -= lengths[1] * round(b2y / lengths[1])
            b3y -= lengths[1] * round(b3y / lengths[1])
        if pbc[2]:
            b1z -= lengths[2] * round(b1z / lengths[2])
            b2z -= lengths[2] * round(b2z / lengths[2])
            b3z -= lengths[2] * round(b3z / lengths[2])

        b0x = -b1x
        b0y = -b1y
        b0z = -b1z
        b2_norm2 = b2x * b2x + b2y * b2y + b2z * b2z
        if b2_norm2 == 0.0:
            raise ValueError("Torsion contains a zero-length central bond.")
        b2_norm = sqrt(b2_norm2)
        inv_b2_norm = 1.0 / b2_norm
        b2ux = b2x * inv_b2_norm
        b2uy = b2y * inv_b2_norm
        b2uz = b2z * inv_b2_norm
        b0_dot_b2u = b0x * b2ux + b0y * b2uy + b0z * b2uz
        b3_dot_b2u = b3x * b2ux + b3y * b2uy + b3z * b2uz
        vx = b0x - b0_dot_b2u * b2ux
        vy = b0y - b0_dot_b2u * b2uy
        vz = b0z - b0_dot_b2u * b2uz
        wx = b3x - b3_dot_b2u * b2ux
        wy = b3y - b3_dot_b2u * b2uy
        wz = b3z - b3_dot_b2u * b2uz
        v_norm = sqrt(vx * vx + vy * vy + vz * vz)
        w_norm = sqrt(wx * wx + wy * wy + wz * wz)
        if v_norm == 0.0 or w_norm == 0.0:
            raise ValueError("Torsion contains collinear or zero-length bonds.")
        inv_v_norm = 1.0 / v_norm
        inv_w_norm = 1.0 / w_norm
        vx *= inv_v_norm
        vy *= inv_v_norm
        vz *= inv_v_norm
        wx *= inv_w_norm
        wy *= inv_w_norm
        wz *= inv_w_norm
        x = vx * wx + vy * wy + vz * wz
        cbx = b2uy * vz - b2uz * vy
        cby = b2uz * vx - b2ux * vz
        cbz = b2ux * vy - b2uy * vx
        y = cbx * wx + cby * wy + cbz * wz
        angle = np.arctan2(y, x)

        n1x = b1y * b2z - b1z * b2y
        n1y = b1z * b2x - b1x * b2z
        n1z = b1x * b2y - b1y * b2x
        n2x = b2y * b3z - b2z * b3y
        n2y = b2z * b3x - b2x * b3z
        n2z = b2x * b3y - b2y * b3x
        n1_norm2 = n1x * n1x + n1y * n1y + n1z * n1z
        n2_norm2 = n2x * n2x + n2y * n2y + n2z * n2z
        if n1_norm2 == 0.0 or n2_norm2 == 0.0:
            raise ValueError("Torsion contains collinear or zero-length bonds.")
        g0x = -b2_norm * n1x / n1_norm2
        g0y = -b2_norm * n1y / n1_norm2
        g0z = -b2_norm * n1z / n1_norm2
        g3x = b2_norm * n2x / n2_norm2
        g3y = b2_norm * n2y / n2_norm2
        g3z = b2_norm * n2z / n2_norm2
        projection1 = (b1x * b2x + b1y * b2y + b1z * b2z) / b2_norm2
        projection3 = (b3x * b2x + b3y * b2y + b3z * b2z) / b2_norm2
        g1x = -(1.0 + projection1) * g0x + projection3 * g3x
        g1y = -(1.0 + projection1) * g0y + projection3 * g3y
        g1z = -(1.0 + projection1) * g0z + projection3 * g3z
        g2x = projection1 * g0x - (1.0 + projection3) * g3x
        g2y = projection1 * g0y - (1.0 + projection3) * g3y
        g2z = projection1 * g0z - (1.0 + projection3) * g3z

        argument = periodicities[row] * angle - phases[row]
        energy += barriers[row] * (1.0 + np.cos(argument))
        scale = barriers[row] * periodicities[row] * np.sin(argument)
        forces[i, 0] += scale * g0x
        forces[i, 1] += scale * g0y
        forces[i, 2] += scale * g0z
        forces[j, 0] += scale * g1x
        forces[j, 1] += scale * g1y
        forces[j, 2] += scale * g1z
        forces[k, 0] += scale * g2x
        forces[k, 1] += scale * g2y
        forces[k, 2] += scale * g2z
        forces[l, 0] += scale * g3x
        forces[l, 1] += scale * g3y
        forces[l, 2] += scale * g3z
    return energy


def _add_improper_arrays(
    positions,
    cell,
    pbc,
    forces,
    indices,
    force_constants,
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
        raise ValueError("Improper contains a zero-length central bond.")
    if np.any((normal1_norm2 == 0.0) | (normal2_norm2 == 0.0)):
        raise ValueError("Improper contains collinear or zero-length bonds.")

    b2_norm = np.sqrt(b2_norm2)
    gradient0 = -(b2_norm / normal1_norm2)[:, np.newaxis] * normal1
    gradient3 = (b2_norm / normal2_norm2)[:, np.newaxis] * normal2
    projection1 = np.einsum("ij,ij->i", b1, b2) / b2_norm2
    projection3 = np.einsum("ij,ij->i", b3, b2) / b2_norm2
    gradient1 = -(1.0 + projection1)[:, np.newaxis] * gradient0 + projection3[:, np.newaxis] * gradient3
    gradient2 = projection1[:, np.newaxis] * gradient0 - (1.0 + projection3)[:, np.newaxis] * gradient3

    bend = np.arctan2(np.sin(angles - phases), np.cos(angles - phases))
    energy = 0.5 * force_constants * bend * bend
    scale = -(force_constants * bend)[:, np.newaxis]
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
    exclusion_keys=None,
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
            exclusion_keys=exclusion_keys,
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
    exclusion_keys=None,
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
        active = _nonexcluded_pair_mask(
            pair_i,
            pair_j,
            exclusions,
            len(forces),
            excluded_keys=exclusion_keys,
        )
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
    sign=1.0,
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
            sign=sign,
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
            energy += sign * (prefactor / distance - shift)
            force_scale = prefactor / (distance * r2)
        else:
            krf, crf = reaction_field
            energy += sign * prefactor * (1.0 / distance + krf * r2 - crf)
            force_scale = prefactor * (1.0 / (distance * r2) - 2.0 * krf)
        fij = sign * force_scale * rij
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
    sign=1.0,
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
    fij = sign * force_scale[:, np.newaxis] * rij
    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return sign * float(np.sum(energy))


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


def _nonexcluded_pair_mask(pair_i, pair_j, exclusions, natoms, excluded_keys=None):
    if excluded_keys is None:
        excluded_keys = _pair_key_array(exclusions, natoms)
    if len(excluded_keys) == 0:
        return np.ones(len(pair_i), dtype=bool)
    fast_mask = _nonexcluded_pair_mask_numba(pair_i, pair_j, excluded_keys, natoms)
    if fast_mask is not None:
        return fast_mask
    pair_keys = _pair_keys(pair_i, pair_j, natoms)
    indices = np.searchsorted(excluded_keys, pair_keys)
    in_range = indices < len(excluded_keys)
    mask = np.ones(len(pair_keys), dtype=bool)
    mask[in_range] = excluded_keys[indices[in_range]] != pair_keys[in_range]
    return mask


def _nonexcluded_pair_mask_numba(pair_i, pair_j, excluded_keys, natoms):
    kernel = _nonexcluded_pair_mask_numba_kernel()
    if kernel is None:
        return None
    mask = np.empty(len(pair_i), dtype=np.bool_)
    kernel(
        mask,
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(excluded_keys, dtype=np.int64),
        int(natoms),
    )
    return mask


def _nonexcluded_pair_mask_numba_kernel():
    global _NONEXCLUDED_PAIR_MASK_NUMBA, _NONEXCLUDED_PAIR_MASK_NUMBA_UNAVAILABLE
    global _numba_prange
    if _NONEXCLUDED_PAIR_MASK_NUMBA_UNAVAILABLE:
        return None
    if _NONEXCLUDED_PAIR_MASK_NUMBA is None:
        try:
            from numba import njit, prange
        except Exception:
            _NONEXCLUDED_PAIR_MASK_NUMBA_UNAVAILABLE = True
            return None
        _numba_prange = prange
        try:
            _NONEXCLUDED_PAIR_MASK_NUMBA = njit(
                cache=True,
                fastmath=True,
                parallel=True,
            )(_nonexcluded_pair_mask_numba_impl)
        except Exception:
            _NONEXCLUDED_PAIR_MASK_NUMBA_UNAVAILABLE = True
            return None
    return _NONEXCLUDED_PAIR_MASK_NUMBA


def _nonexcluded_pair_mask_numba_impl(mask, pair_i, pair_j, excluded_keys, natoms):
    n_excluded = len(excluded_keys)
    for pair_index in _numba_prange(len(pair_i)):
        i = pair_i[pair_index]
        j = pair_j[pair_index]
        if i < j:
            key = i * natoms + j
        else:
            key = j * natoms + i
        lo = 0
        hi = n_excluded
        while lo < hi:
            mid = (lo + hi) // 2
            if excluded_keys[mid] < key:
                lo = mid + 1
            else:
                hi = mid
        mask[pair_index] = lo >= n_excluded or excluded_keys[lo] != key


def _pair_keys(pair_i, pair_j, natoms):
    lower = np.minimum(pair_i, pair_j).astype(np.int64, copy=False)
    upper = np.maximum(pair_i, pair_j).astype(np.int64, copy=False)
    return lower * int(natoms) + upper


def _pair_key(i, j, natoms):
    i = int(i)
    j = int(j)
    return min(i, j) * int(natoms) + max(i, j)


def _pair_key_array(pairs, natoms):
    if not pairs:
        return np.asarray([], dtype=np.int64)
    keys = np.fromiter(
        (_pair_key(i, j, natoms) for i, j in pairs),
        dtype=np.int64,
        count=len(pairs),
    )
    return np.unique(keys)


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
    if isinstance(pair_parameters, tuple) and len(pair_parameters) == 4:
        return pair_parameters
    if not pair_parameters:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i, empty_f, empty_f
    pairs = np.asarray(list(pair_parameters.keys()), dtype=int)
    values = np.asarray(list(pair_parameters.values()), dtype=float)
    return pairs[:, 0], pairs[:, 1], values[:, 0], values[:, 1]


def _pair_float_arrays(pair_values):
    if isinstance(pair_values, tuple) and len(pair_values) == 3:
        return pair_values
    if not pair_values:
        empty_i = np.asarray([], dtype=int)
        empty_f = np.asarray([], dtype=float)
        return empty_i, empty_i, empty_f
    pairs = np.asarray(list(pair_values.keys()), dtype=int)
    values = np.asarray(list(pair_values.values()), dtype=float)
    return pairs[:, 0], pairs[:, 1], values


def _pair_index_arrays(pairs):
    if isinstance(pairs, tuple) and len(pairs) == 2:
        return pairs
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
    order = _validate_pme_order(order)

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


def _pme_coulomb_energy_terms(
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
    order = _validate_pme_order(order)

    cutoff = 0.5 * float(np.min(lengths)) if real_cutoff is None else float(real_cutoff)
    real_forces = np.zeros_like(forces)
    reciprocal_forces = np.zeros_like(forces)
    real = _add_ewald_real(
        positions,
        lengths,
        real_forces,
        charges,
        coulomb_constant,
        alpha,
        cutoff,
        pair_displacements=real_pair_displacements,
    )
    reciprocal = _add_pme_reciprocal(
        positions,
        lengths,
        reciprocal_forces,
        charges,
        coulomb_constant,
        alpha,
        mesh,
        order,
    )
    self_energy = -coulomb_constant * alpha / sqrt(np.pi) * float(np.dot(charges, charges))
    return {
        "real": float(real),
        "reciprocal": float(reciprocal),
        "self": float(self_energy),
        "total_without_exceptions": float(real + reciprocal + self_energy),
    }


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
    lj_exclusion_keys=None,
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
    order = _validate_pme_order(order)

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
        lj_exclusion_keys=lj_exclusion_keys,
        lj_pair_nonexcluded_mask=(
            None
            if lj_exclusions is None
            else pair_displacements.nonexcluded_mask(lj_exclusion_keys, len(forces))
        ),
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
    lj_exclusion_keys=None,
    lj_pair_nonexcluded_mask=None,
    virial=None,
):
    if len(pair_i) == 0:
        return 0.0
    fast_energy = _try_add_lj_ewald_real_pair_arrays_numba(
        forces,
        epsilon,
        sigma,
        charges,
        coulomb_constant,
        alpha,
        cutoff2,
        pair_i,
        pair_j,
        displacements,
        lj_cutoff=lj_cutoff,
        lj_switch_on=lj_switch_on,
        lj_energy_shift=lj_energy_shift,
        pair_parameter_lookup=pair_parameter_lookup,
        pair_overrides_present=bool(pair_overrides),
        lj_pair_nonexcluded_mask=lj_pair_nonexcluded_mask,
        virial=virial,
    )
    if fast_energy is not None:
        return fast_energy
    rij = displacements
    r2 = np.einsum("ij,ij->i", rij, rij)
    active = (r2 > 0.0) & (r2 <= cutoff2)
    if not np.any(active):
        return 0.0
    active_pair_mask = active
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
    energy = float(np.sum(prefactor * erfc_ar / distance))
    force_scalar = prefactor * (
        erfc_ar / (distance * r2)
        + 2.0 * alpha * exp_ar2 / (sqrt(np.pi) * r2)
    )
    fij = force_scalar[:, np.newaxis] * rij

    if lj_pair_nonexcluded_mask is None:
        lj_active = np.ones(len(pair_i), dtype=bool)
    else:
        lj_active = np.asarray(lj_pair_nonexcluded_mask, dtype=bool)[active_pair_mask].copy()
    if lj_cutoff is not None and not np.isclose(lj_cutoff, cutoff, rtol=0.0, atol=1e-12):
        lj_active &= r2 <= lj_cutoff * lj_cutoff
    if lj_exclusions is not None and lj_pair_nonexcluded_mask is None:
        lj_active &= _nonexcluded_pair_mask(
            pair_i,
            pair_j,
            lj_exclusions,
            len(forces),
            excluded_keys=lj_exclusion_keys,
        )
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
            if lj_switch_on is not None:
                lj_fij = (
                    24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2
                )[:, np.newaxis] * lj_rij
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
                fij[lj_positions] += lj_fij
            else:
                lj_force_scalar = 24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2
                fij[lj_positions] += lj_force_scalar[:, np.newaxis] * lj_rij
            shift = 0.0
            if lj_cutoff is not None and lj_energy_shift and lj_switch_on is None:
                sr6_cutoff = (sigma_ij / lj_cutoff) ** 6
                shift = 4.0 * epsilon_ij * (sr6_cutoff * sr6_cutoff - sr6_cutoff)
            energy += float(np.sum(lj_energy - shift))

    _scatter_pair_forces(forces, pair_i, pair_j, fij)
    if virial is not None:
        virial += rij.T @ fij
    return energy


def _try_add_lj_ewald_real_pair_arrays_numba(
    forces,
    epsilon,
    sigma,
    charges,
    coulomb_constant,
    alpha,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    lj_cutoff=None,
    lj_switch_on=None,
    lj_energy_shift=True,
    pair_parameter_lookup=None,
    pair_overrides_present=False,
    lj_pair_nonexcluded_mask=None,
    virial=None,
):
    if lj_switch_on is not None or lj_pair_nonexcluded_mask is None:
        return None
    if pair_parameter_lookup is None and pair_overrides_present:
        return None
    kernel = _lj_ewald_real_pair_arrays_numba()
    if kernel is None:
        return None
    if pair_parameter_lookup is None:
        type_codes = np.asarray([], dtype=np.int32)
        type_epsilon = np.asarray([[]], dtype=float)
        type_sigma = np.asarray([[]], dtype=float)
        use_type_lookup = False
    else:
        type_codes, type_epsilon, type_sigma = pair_parameter_lookup
        use_type_lookup = True
    lj_cutoff2 = -1.0 if lj_cutoff is None else float(lj_cutoff) * float(lj_cutoff)
    if virial is None:
        virial_array = np.zeros((3, 3), dtype=float)
        use_virial = False
    else:
        virial_array = virial
        use_virial = True
    if len(pair_i) >= _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_MIN_PAIRS:
        parallel_kernel = _lj_ewald_real_pair_arrays_parallel_numba()
        if parallel_kernel is not None:
            return float(
                parallel_kernel(
                    forces,
                    virial_array,
                    np.asarray(epsilon, dtype=float),
                    np.asarray(sigma, dtype=float),
                    np.asarray(charges, dtype=float),
                    float(coulomb_constant),
                    float(alpha),
                    float(cutoff2),
                    np.asarray(pair_i, dtype=np.int64),
                    np.asarray(pair_j, dtype=np.int64),
                    np.asarray(displacements, dtype=float),
                    np.asarray(type_codes, dtype=np.int32),
                    np.asarray(type_epsilon, dtype=float),
                    np.asarray(type_sigma, dtype=float),
                    bool(use_type_lookup),
                    np.asarray(lj_pair_nonexcluded_mask, dtype=np.bool_),
                    lj_cutoff2,
                    bool(lj_energy_shift),
                    bool(use_virial),
                )
            )
    return float(
        kernel(
            forces,
            virial_array,
            np.asarray(epsilon, dtype=float),
            np.asarray(sigma, dtype=float),
            np.asarray(charges, dtype=float),
            float(coulomb_constant),
            float(alpha),
            float(cutoff2),
            np.asarray(pair_i, dtype=np.int64),
            np.asarray(pair_j, dtype=np.int64),
            np.asarray(displacements, dtype=float),
            np.asarray(type_codes, dtype=np.int32),
            np.asarray(type_epsilon, dtype=float),
            np.asarray(type_sigma, dtype=float),
            bool(use_type_lookup),
            np.asarray(lj_pair_nonexcluded_mask, dtype=np.bool_),
            lj_cutoff2,
            bool(lj_energy_shift),
            bool(use_virial),
        )
    )


def _lj_ewald_real_pair_arrays_numba():
    global _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA, _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA_UNAVAILABLE
    if _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA_UNAVAILABLE:
        return None
    if _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA_UNAVAILABLE = True
            return None
        try:
            _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA = njit(cache=True, fastmath=True)(
                _lj_ewald_real_pair_arrays_numba_impl
            )
        except Exception:
            _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA_UNAVAILABLE = True
            return None
    return _LJ_EWALD_REAL_PAIR_ARRAYS_NUMBA


def _lj_ewald_real_pair_arrays_parallel_numba():
    global _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA
    global _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA_UNAVAILABLE
    global _numba_prange, _numba_get_num_threads, _numba_get_thread_id
    if _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA_UNAVAILABLE:
        return None
    if _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA is None:
        try:
            from numba import get_num_threads, get_thread_id, njit, prange
        except Exception:
            _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA_UNAVAILABLE = True
            return None
        _numba_prange = prange
        _numba_get_num_threads = get_num_threads
        _numba_get_thread_id = get_thread_id
        try:
            _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA = njit(
                cache=False,
                fastmath=True,
                parallel=True,
            )(_lj_ewald_real_pair_arrays_parallel_numba_impl)
        except Exception:
            _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA_UNAVAILABLE = True
            return None
    return _LJ_EWALD_REAL_PAIR_ARRAYS_PARALLEL_NUMBA


def _lj_ewald_real_pair_arrays_parallel_numba_impl(
    forces,
    virial,
    epsilon,
    sigma,
    charges,
    coulomb_constant,
    alpha,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    type_codes,
    type_epsilon,
    type_sigma,
    use_type_lookup,
    lj_pair_nonexcluded_mask,
    lj_cutoff2,
    lj_energy_shift,
    use_virial,
):
    nthreads = _numba_get_num_threads()
    natoms = forces.shape[0]
    thread_forces = np.zeros((nthreads, natoms, 3), dtype=np.float64)
    thread_virials = np.zeros((nthreads, 3, 3), dtype=np.float64)
    thread_energies = np.zeros(nthreads, dtype=np.float64)
    sqrt_pi = sqrt(np.pi)
    use_lj_cutoff = lj_cutoff2 > 0.0
    lj_cutoff = sqrt(lj_cutoff2) if use_lj_cutoff else 0.0

    for pair_index in _numba_prange(len(pair_i)):
        tid = _numba_get_thread_id()
        dx = displacements[pair_index, 0]
        dy = displacements[pair_index, 1]
        dz = displacements[pair_index, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 <= 0.0 or r2 > cutoff2:
            continue
        i = pair_i[pair_index]
        j = pair_j[pair_index]
        distance = sqrt(r2)
        charge_product = charges[i] * charges[j]
        prefactor = coulomb_constant * charge_product
        ar = alpha * distance
        erfc_ar = erfc(ar)
        exp_ar2 = exp(-(ar * ar))
        thread_energies[tid] += prefactor * erfc_ar / distance
        force_scalar = prefactor * (
            erfc_ar / (distance * r2)
            + 2.0 * alpha * exp_ar2 / (sqrt_pi * r2)
        )

        if lj_pair_nonexcluded_mask[pair_index] and (not use_lj_cutoff or r2 <= lj_cutoff2):
            if use_type_lookup:
                code_i = type_codes[i]
                code_j = type_codes[j]
                epsilon_ij = type_epsilon[code_i, code_j]
                sigma_ij = type_sigma[code_i, code_j]
            else:
                epsilon_ij = sqrt(epsilon[i] * epsilon[j])
                sigma_ij = 0.5 * (sigma[i] + sigma[j])
            if epsilon_ij != 0.0:
                inv_r2 = 1.0 / r2
                sr2 = sigma_ij * sigma_ij * inv_r2
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6
                lj_energy = 4.0 * epsilon_ij * (sr12 - sr6)
                force_scalar += 24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2
                if use_lj_cutoff and lj_energy_shift:
                    sr6_cutoff = (sigma_ij / lj_cutoff) ** 6
                    lj_energy -= 4.0 * epsilon_ij * (
                        sr6_cutoff * sr6_cutoff - sr6_cutoff
                    )
                thread_energies[tid] += lj_energy

        fx = force_scalar * dx
        fy = force_scalar * dy
        fz = force_scalar * dz
        thread_forces[tid, i, 0] += fx
        thread_forces[tid, i, 1] += fy
        thread_forces[tid, i, 2] += fz
        thread_forces[tid, j, 0] -= fx
        thread_forces[tid, j, 1] -= fy
        thread_forces[tid, j, 2] -= fz
        if use_virial:
            thread_virials[tid, 0, 0] += dx * fx
            thread_virials[tid, 0, 1] += dx * fy
            thread_virials[tid, 0, 2] += dx * fz
            thread_virials[tid, 1, 0] += dy * fx
            thread_virials[tid, 1, 1] += dy * fy
            thread_virials[tid, 1, 2] += dy * fz
            thread_virials[tid, 2, 0] += dz * fx
            thread_virials[tid, 2, 1] += dz * fy
            thread_virials[tid, 2, 2] += dz * fz

    energy = 0.0
    for tid in range(nthreads):
        energy += thread_energies[tid]
        for atom in range(natoms):
            forces[atom, 0] += thread_forces[tid, atom, 0]
            forces[atom, 1] += thread_forces[tid, atom, 1]
            forces[atom, 2] += thread_forces[tid, atom, 2]
        if use_virial:
            for axis_i in range(3):
                for axis_j in range(3):
                    virial[axis_i, axis_j] += thread_virials[tid, axis_i, axis_j]
    return energy


def _lj_ewald_real_pair_arrays_numba_impl(
    forces,
    virial,
    epsilon,
    sigma,
    charges,
    coulomb_constant,
    alpha,
    cutoff2,
    pair_i,
    pair_j,
    displacements,
    type_codes,
    type_epsilon,
    type_sigma,
    use_type_lookup,
    lj_pair_nonexcluded_mask,
    lj_cutoff2,
    lj_energy_shift,
    use_virial,
):
    energy = 0.0
    v00 = 0.0
    v01 = 0.0
    v02 = 0.0
    v10 = 0.0
    v11 = 0.0
    v12 = 0.0
    v20 = 0.0
    v21 = 0.0
    v22 = 0.0
    sqrt_pi = sqrt(np.pi)
    use_lj_cutoff = lj_cutoff2 > 0.0
    lj_cutoff = sqrt(lj_cutoff2) if use_lj_cutoff else 0.0
    for pair_index in range(len(pair_i)):
        dx = displacements[pair_index, 0]
        dy = displacements[pair_index, 1]
        dz = displacements[pair_index, 2]
        r2 = dx * dx + dy * dy + dz * dz
        if r2 <= 0.0 or r2 > cutoff2:
            continue
        i = pair_i[pair_index]
        j = pair_j[pair_index]
        distance = sqrt(r2)
        charge_product = charges[i] * charges[j]
        prefactor = coulomb_constant * charge_product
        ar = alpha * distance
        erfc_ar = erfc(ar)
        exp_ar2 = exp(-(ar * ar))
        energy += prefactor * erfc_ar / distance
        force_scalar = prefactor * (
            erfc_ar / (distance * r2)
            + 2.0 * alpha * exp_ar2 / (sqrt_pi * r2)
        )

        if lj_pair_nonexcluded_mask[pair_index] and (not use_lj_cutoff or r2 <= lj_cutoff2):
            if use_type_lookup:
                code_i = type_codes[i]
                code_j = type_codes[j]
                epsilon_ij = type_epsilon[code_i, code_j]
                sigma_ij = type_sigma[code_i, code_j]
            else:
                epsilon_ij = sqrt(epsilon[i] * epsilon[j])
                sigma_ij = 0.5 * (sigma[i] + sigma[j])
            if epsilon_ij != 0.0:
                inv_r2 = 1.0 / r2
                sr2 = sigma_ij * sigma_ij * inv_r2
                sr6 = sr2 * sr2 * sr2
                sr12 = sr6 * sr6
                lj_energy = 4.0 * epsilon_ij * (sr12 - sr6)
                force_scalar += 24.0 * epsilon_ij * (2.0 * sr12 - sr6) * inv_r2
                if use_lj_cutoff and lj_energy_shift:
                    sr6_cutoff = (sigma_ij / lj_cutoff) ** 6
                    lj_energy -= 4.0 * epsilon_ij * (
                        sr6_cutoff * sr6_cutoff - sr6_cutoff
                    )
                energy += lj_energy

        fx = force_scalar * dx
        fy = force_scalar * dy
        fz = force_scalar * dz
        forces[i, 0] += fx
        forces[i, 1] += fy
        forces[i, 2] += fz
        forces[j, 0] -= fx
        forces[j, 1] -= fy
        forces[j, 2] -= fz
        if use_virial:
            v00 += dx * fx
            v01 += dx * fy
            v02 += dx * fz
            v10 += dy * fx
            v11 += dy * fy
            v12 += dy * fz
            v20 += dz * fx
            v21 += dz * fy
            v22 += dz * fz
    if use_virial:
        virial[0, 0] += v00
        virial[0, 1] += v01
        virial[0, 2] += v02
        virial[1, 0] += v10
        virial[1, 1] += v11
        virial[1, 2] += v12
        virial[2, 0] += v20
        virial[2, 1] += v21
        virial[2, 2] += v22
    return energy


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
    order = _validate_pme_order(order)

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
    kernel = _assign_charges_bspline_numba()
    if kernel is not None and int(order) in {2, 4}:
        kernel(
            grid,
            np.asarray(positions, dtype=float),
            np.asarray(charges, dtype=float),
            np.asarray(lengths, dtype=float),
            np.asarray(mesh, dtype=np.int64),
            int(order),
        )
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
    kernel = _reciprocal_assignment_forces_numba()
    if kernel is not None and int(order) in {2, 4}:
        kernel(
            forces,
            np.asarray(positions, dtype=float),
            np.asarray(charges, dtype=float),
            np.asarray(potential_grid, dtype=float),
            np.asarray(lengths, dtype=float),
            np.asarray(mesh, dtype=np.int64),
            int(order),
        )
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
    if order >= 3:
        offsets = np.arange(-1, order - 1, dtype=int)
        x = 2.0 + offsets.astype(float) - frac
        weights = _cardinal_bspline_values(x, order)
        derivatives = -_cardinal_bspline_derivatives(x, order)
        return weights, derivatives, offsets
    raise ValueError("PME order must be an integer from 2 through 8.")


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
    if order >= 3:
        offsets = np.arange(-1, order - 1, dtype=int)
        x = 2.0 + offsets[np.newaxis, :].astype(float) - frac[:, np.newaxis]
        weights = _cardinal_bspline_values(x, order)
        derivatives = -_cardinal_bspline_derivatives(x, order)
        return weights, derivatives, offsets
    raise ValueError("PME order must be an integer from 2 through 8.")


def _cardinal_bspline_values(x, order):
    x = np.asarray(x, dtype=float)
    if order == 1:
        return np.where((x >= 0.0) & (x < 1.0), 1.0, 0.0)
    return (
        x / float(order - 1) * _cardinal_bspline_values(x, order - 1)
        + (float(order) - x) / float(order - 1) * _cardinal_bspline_values(x - 1.0, order - 1)
    )


def _cardinal_bspline_derivatives(x, order):
    x = np.asarray(x, dtype=float)
    if order <= 1:
        return np.zeros_like(x, dtype=float)
    return _cardinal_bspline_values(x, order - 1) - _cardinal_bspline_values(x - 1.0, order - 1)


def _assign_charges_bspline_numba():
    global _ASSIGN_CHARGES_BSPLINE_NUMBA, _ASSIGN_CHARGES_BSPLINE_NUMBA_UNAVAILABLE
    global _FILL_BSPLINE_STENCIL_NUMBA
    if _ASSIGN_CHARGES_BSPLINE_NUMBA_UNAVAILABLE:
        return None
    if _ASSIGN_CHARGES_BSPLINE_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _ASSIGN_CHARGES_BSPLINE_NUMBA_UNAVAILABLE = True
            return None
        try:
            if _FILL_BSPLINE_STENCIL_NUMBA is None:
                _FILL_BSPLINE_STENCIL_NUMBA = njit(cache=True, fastmath=True)(_fill_bspline_stencil_numba)
            _ASSIGN_CHARGES_BSPLINE_NUMBA = njit(cache=True, fastmath=True)(_assign_charges_bspline_numba_impl)
        except Exception:
            _ASSIGN_CHARGES_BSPLINE_NUMBA_UNAVAILABLE = True
            return None
    return _ASSIGN_CHARGES_BSPLINE_NUMBA


def _reciprocal_assignment_forces_numba():
    global _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA, _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA_UNAVAILABLE
    global _FILL_BSPLINE_STENCIL_NUMBA
    if _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA_UNAVAILABLE:
        return None
    if _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA_UNAVAILABLE = True
            return None
        try:
            if _FILL_BSPLINE_STENCIL_NUMBA is None:
                _FILL_BSPLINE_STENCIL_NUMBA = njit(cache=True, fastmath=True)(_fill_bspline_stencil_numba)
            _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA = njit(cache=True, fastmath=True)(
                _reciprocal_assignment_forces_numba_impl
            )
        except Exception:
            _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA_UNAVAILABLE = True
            return None
    return _RECIPROCAL_ASSIGNMENT_FORCES_NUMBA


def _assign_charges_bspline_numba_impl(grid, positions, charges, lengths, mesh, order):
    weights = np.empty((3, 4), dtype=np.float64)
    derivatives = np.empty((3, 4), dtype=np.float64)
    indices = np.empty((3, 4), dtype=np.int64)
    for atom in range(len(positions)):
        _FILL_BSPLINE_STENCIL_NUMBA(positions, lengths, mesh, order, atom, weights, derivatives, indices)
        charge = charges[atom]
        for ax in range(order):
            wx = weights[0, ax]
            ix = indices[0, ax]
            for ay in range(order):
                wxy = charge * wx * weights[1, ay]
                iy = indices[1, ay]
                for az in range(order):
                    grid[ix, iy, indices[2, az]] += wxy * weights[2, az]


def _reciprocal_assignment_forces_numba_impl(
    forces,
    positions,
    charges,
    potential_grid,
    lengths,
    mesh,
    order,
):
    weights = np.empty((3, 4), dtype=np.float64)
    derivatives = np.empty((3, 4), dtype=np.float64)
    indices = np.empty((3, 4), dtype=np.int64)
    scale0 = mesh[0] / lengths[0]
    scale1 = mesh[1] / lengths[1]
    scale2 = mesh[2] / lengths[2]
    for atom in range(len(positions)):
        _FILL_BSPLINE_STENCIL_NUMBA(positions, lengths, mesh, order, atom, weights, derivatives, indices)
        grad0 = 0.0
        grad1 = 0.0
        grad2 = 0.0
        for ax in range(order):
            ix = indices[0, ax]
            wx = weights[0, ax]
            dwx = derivatives[0, ax]
            for ay in range(order):
                iy = indices[1, ay]
                wy = weights[1, ay]
                dwy = derivatives[1, ay]
                for az in range(order):
                    iz = indices[2, az]
                    wz = weights[2, az]
                    phi = potential_grid[ix, iy, iz]
                    grad0 += dwx * scale0 * wy * wz * phi
                    grad1 += wx * dwy * scale1 * wz * phi
                    grad2 += wx * wy * derivatives[2, az] * scale2 * phi
        charge = charges[atom]
        forces[atom, 0] = -charge * grad0
        forces[atom, 1] = -charge * grad1
        forces[atom, 2] = -charge * grad2


def _fill_bspline_stencil_numba(positions, lengths, mesh, order, atom, weights, derivatives, indices):
    for axis in range(3):
        scaled = (positions[atom, axis] / lengths[axis] % 1.0) * mesh[axis]
        base = int(np.floor(scaled))
        frac = scaled - base
        if order == 2:
            weights[axis, 0] = 1.0 - frac
            weights[axis, 1] = frac
            derivatives[axis, 0] = -1.0
            derivatives[axis, 1] = 1.0
            indices[axis, 0] = base % mesh[axis]
            indices[axis, 1] = (base + 1) % mesh[axis]
        else:
            one_minus = 1.0 - frac
            frac2 = frac * frac
            frac3 = frac2 * frac
            weights[axis, 0] = one_minus * one_minus * one_minus / 6.0
            weights[axis, 1] = (3.0 * frac3 - 6.0 * frac2 + 4.0) / 6.0
            weights[axis, 2] = (-3.0 * frac3 + 3.0 * frac2 + 3.0 * frac + 1.0) / 6.0
            weights[axis, 3] = frac3 / 6.0
            derivatives[axis, 0] = -0.5 * one_minus * one_minus
            derivatives[axis, 1] = 1.5 * frac2 - 2.0 * frac
            derivatives[axis, 2] = -1.5 * frac2 + frac + 0.5
            derivatives[axis, 3] = 0.5 * frac2
            indices[axis, 0] = (base - 1) % mesh[axis]
            indices[axis, 1] = base % mesh[axis]
            indices[axis, 2] = (base + 1) % mesh[axis]
            indices[axis, 3] = (base + 2) % mesh[axis]


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
