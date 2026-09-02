"""Finite-displacement phonons for three-dimensional periodic cells."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
import time

import numpy as np

from pyqed.units import amu_to_au, au2wavenumber


def _normalize_qpoint(qpoint):
    qpoint = np.asarray(qpoint, dtype=float)
    if qpoint.shape != (3,) or not np.all(np.isfinite(qpoint)):
        raise ValueError("qpoint must contain three finite fractional coordinates.")
    return np.ascontiguousarray(qpoint)


def _normalize_branch(branch, nmode):
    if isinstance(branch, (bool, np.bool_)):
        raise TypeError("branch must be an integer.")
    try:
        index = int(branch)
    except (TypeError, ValueError, OverflowError) as exc:
        raise TypeError("branch must be an integer.") from exc
    if index != branch:
        raise TypeError("branch must be an integer.")
    if index < 0 or index >= int(nmode):
        raise IndexError(f"branch {index} is out of range for {nmode} phonon modes.")
    return index


def _canonicalize_eigenvector(eigenvector):
    vector = np.asarray(eigenvector, dtype=np.complex128).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm == 0.0:
        raise ValueError("phonon eigenvector must have finite nonzero norm.")
    vector = vector / norm
    pivot = int(np.argmax(np.abs(vector)))
    vector *= np.exp(-1.0j * np.angle(vector[pivot]))
    if np.max(np.abs(vector.imag), initial=0.0) <= 1.0e-13:
        vector = vector.real.astype(np.complex128)
    return vector


@dataclass(frozen=True)
class PeriodicPhononMode:
    r"""One normalized harmonic mode at a fractional reciprocal point.

    ``eigenvector`` is the unit-norm eigenvector of the mass-weighted
    dynamical matrix.  The Cartesian displacement per unit mass-weighted
    coordinate is therefore :math:`e_{A\alpha}/\sqrt{M_A}`.
    ``frequency`` is the signed harmonic frequency in atomic units.
    """

    qpoint: np.ndarray
    branch: int
    frequency: float
    eigenvector: np.ndarray
    masses: np.ndarray
    source: str = "periodic_phonon"

    def __post_init__(self):
        qpoint = _normalize_qpoint(self.qpoint)
        masses = np.asarray(self.masses, dtype=float)
        if masses.ndim != 1 or len(masses) == 0 or np.any(masses <= 0.0):
            raise ValueError("masses must contain one positive atomic-unit mass per atom.")
        eigenvector = _canonicalize_eigenvector(self.eigenvector)
        if eigenvector.size != 3 * len(masses):
            raise ValueError("eigenvector must contain three components per atom.")
        frequency = float(self.frequency)
        if not np.isfinite(frequency):
            raise ValueError("frequency must be finite.")
        branch = _normalize_branch(self.branch, eigenvector.size)
        qpoint.setflags(write=False)
        masses = np.ascontiguousarray(masses)
        masses.setflags(write=False)
        eigenvector = np.ascontiguousarray(eigenvector.reshape(len(masses), 3))
        eigenvector.setflags(write=False)
        object.__setattr__(self, "qpoint", qpoint)
        object.__setattr__(self, "branch", branch)
        object.__setattr__(self, "frequency", frequency)
        object.__setattr__(self, "eigenvector", eigenvector)
        object.__setattr__(self, "masses", masses)
        object.__setattr__(self, "source", str(self.source))

    @property
    def stable(self):
        return self.frequency > 0.0

    @property
    def cartesian_displacement(self):
        return self.eigenvector / np.sqrt(self.masses)[:, None]


def _normalize_supercell(supercell):
    values = np.asarray(supercell, dtype=int)
    if values.shape != (3,) or np.any(values <= 0):
        raise ValueError("supercell must contain three positive integers.")
    return tuple(int(value) for value in values)


def _cell_geometry(cell):
    if not getattr(cell, "built", False):
        cell.build()
    if int(getattr(cell, "dimension", 3)) != 3:
        raise NotImplementedError("Periodic phonons currently require dimension=3.")
    symbols = tuple(str(symbol) for symbol in cell._atom_symbols)
    positions = np.asarray(cell._atom_coords, dtype=float)
    lattice = np.asarray(cell.lattice_vectors, dtype=float)
    if positions.shape != (len(symbols), 3) or lattice.shape != (3, 3):
        raise ValueError("The periodic cell geometry is malformed.")
    return symbols, positions, lattice


def _supercell_geometry(symbols, positions, lattice, supercell):
    translations = np.asarray(list(product(*(range(n) for n in supercell))), dtype=int)
    super_symbols = []
    super_positions = []
    primitive_indices = []
    image_translations = []
    for translation in translations:
        shift = translation @ lattice
        for atom_index, (symbol, position) in enumerate(zip(symbols, positions)):
            super_symbols.append(symbol)
            super_positions.append(position + shift)
            primitive_indices.append(atom_index)
            image_translations.append(translation)
    super_lattice = np.diag(np.asarray(supercell, dtype=float)) @ lattice
    return (
        tuple(super_symbols),
        np.ascontiguousarray(super_positions, dtype=float),
        np.ascontiguousarray(super_lattice, dtype=float),
        np.asarray(primitive_indices, dtype=int),
        np.asarray(image_translations, dtype=int),
    )


def interpolate_q_path(vertices, lattice, points_per_segment=41):
    """Interpolate a path through fractional reciprocal coordinates."""

    vertices = np.asarray(vertices, dtype=float)
    lattice = np.asarray(lattice, dtype=float)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 2:
        raise ValueError("vertices must have shape (nvertex, 3) with nvertex >= 2.")
    points_per_segment = int(points_per_segment)
    if points_per_segment < 2:
        raise ValueError("points_per_segment must be at least 2.")

    qpoints = [vertices[0]]
    ticks = [0]
    for start, stop in zip(vertices[:-1], vertices[1:]):
        segment = np.linspace(start, stop, points_per_segment)
        qpoints.extend(segment[1:])
        ticks.append(len(qpoints) - 1)
    qpoints = np.asarray(qpoints, dtype=float)

    reciprocal = 2.0 * np.pi * np.linalg.inv(lattice).T
    cartesian = qpoints @ reciprocal
    steps = np.linalg.norm(np.diff(cartesian, axis=0), axis=1)
    distances = np.concatenate(([0.0], np.cumsum(steps)))
    return qpoints, distances, np.asarray(ticks, dtype=int)


class KRHFForceCalculator:
    """Native all-electron or GTH Gamma-point KRHF forces."""

    def __init__(
        self,
        basis,
        *,
        kmesh=(1, 1, 1),
        charge=0,
        spin=0,
        pseudo=None,
        integral_options=None,
        scf_options=None,
        gdf_options=None,
        run_options=None,
    ):
        self.basis = basis
        self.kmesh = _normalize_supercell(kmesh)
        if self.kmesh != (1, 1, 1):
            raise NotImplementedError(
                "Native analytic KRHF forces currently require kmesh=(1, 1, 1)."
            )
        self.charge = int(charge)
        self.spin = int(spin)
        self.pseudo = pseudo
        self.integral_options = (
            {"eri_representation": "direct"}
            if integral_options is None
            else dict(integral_options)
        )
        self.scf_options = {} if scf_options is None else dict(scf_options)
        self.scf_options.setdefault("jk_builder", "reciprocal")
        if str(self.scf_options["jk_builder"]).lower() not in (
            "ewald",
            "reciprocal",
            "gdf",
        ):
            raise NotImplementedError(
                "Native analytic KRHF forces require an Ewald, reciprocal, or GDF J/K builder."
            )
        self.gdf_options = {} if gdf_options is None else dict(gdf_options)
        if self.gdf_options and str(self.scf_options["jk_builder"]).lower() != "gdf":
            raise ValueError("gdf_options require scf_options['jk_builder']='gdf'.")
        self.run_options = {} if run_options is None else dict(run_options)
        self.history = []
        self.mean_field = None
        self.energy = None
        self.converged = False
        self._dm0 = None

    def forces(self, symbols, positions, lattice):
        """Return native analytic forces in Hartree/Bohr."""
        from pyqed.qchem.pbc import Cell

        positions = np.asarray(positions, dtype=float)
        lattice = np.asarray(lattice, dtype=float)
        if positions.shape != (len(symbols), 3) or lattice.shape != (3, 3):
            raise ValueError("positions and lattice must have shapes (natom, 3) and (3, 3).")
        cell = Cell(
            atom=[
                (str(symbol), tuple(position))
                for symbol, position in zip(symbols, positions)
            ],
            a=lattice,
            basis=self.basis,
            unit="bohr",
            charge=self.charge,
            spin=self.spin,
            pseudo=self.pseudo,
            dimension=3,
            integral_options=self.integral_options,
        ).build()
        mean_field = cell.KRHF(nk=self.kmesh, **self.scf_options)
        if str(mean_field.jk_builder) == "gdf":
            mean_field.density_fit(**self.gdf_options)
        dm0 = self._dm0
        if dm0 is not None and np.asarray(dm0).shape != (cell.nao, cell.nao):
            dm0 = None

        started = time.perf_counter()
        mean_field.run(dm0=dm0, **self.run_options)
        if not mean_field.converged:
            raise RuntimeError("Native periodic KRHF did not converge for a displaced cell.")
        forces = np.asarray(mean_field.forces(), dtype=float)
        seconds = time.perf_counter() - started
        if forces.shape != positions.shape or not np.all(np.isfinite(forces)):
            raise RuntimeError("Native periodic KRHF returned invalid nuclear forces.")

        self._dm0 = mean_field.make_rdm1()
        self.mean_field = mean_field
        self.energy = float(mean_field.e_tot)
        self.converged = True
        self.history.append(
            {
                "energy_Ha": self.energy,
                "seconds": float(seconds),
                "max_abs_force_Ha_per_bohr": float(np.max(np.abs(forces))),
                "scf_cycles": int(mean_field.niter),
            }
        )
        return np.ascontiguousarray(forces)

    __call__ = forces


class FiniteDisplacementPhonon:
    """Construct harmonic force constants from central differences of forces."""

    def __init__(
        self,
        cell,
        force_calculator,
        *,
        supercell=(2, 2, 2),
        displacement=0.01,
        masses=None,
        enforce_acoustic_sum_rule=True,
        subtract_force_drift=True,
    ):
        symbols, positions, lattice = _cell_geometry(cell)
        self.cell = cell
        self.force_calculator = force_calculator
        self.supercell = _normalize_supercell(supercell)
        self.displacement = float(displacement)
        if not np.isfinite(self.displacement) or self.displacement <= 0.0:
            raise ValueError("displacement must be a positive finite distance in Bohr.")
        self.symbols = symbols
        self.positions = positions
        self.lattice = lattice
        self.natom = len(symbols)
        if masses is None:
            masses = cell.unit_molecule.atom_mass_list()
        self.masses = np.asarray(masses, dtype=float)
        if self.masses.shape != (self.natom,) or np.any(self.masses <= 0.0):
            raise ValueError("masses must contain one positive value in amu per atom.")
        self.enforce_acoustic_sum_rule = bool(enforce_acoustic_sum_rule)
        self.subtract_force_drift = bool(subtract_force_drift)

        (
            self.super_symbols,
            self.super_positions,
            self.super_lattice,
            self.super_primitive_indices,
            self.super_translations,
        ) = _supercell_geometry(symbols, positions, lattice, self.supercell)
        self.nsuper = len(self.super_symbols)
        self._super_index = {
            (int(atom), tuple(int(value) for value in translation)): index
            for index, (atom, translation) in enumerate(
                zip(self.super_primitive_indices, self.super_translations)
            )
        }
        self._reference_indices = np.asarray(
            [self._super_index[(atom, (0, 0, 0))] for atom in range(self.natom)],
            dtype=int,
        )
        self._phase_translations = self._build_phase_translations()

        self.force_constants = None
        self.raw_force_constants = None
        self.force_history = []
        self.success = False
        self.message = "not run"
        self.path_qpoints = None
        self.path_distances = None
        self.path_frequencies = None
        self.path_ticks = None
        self.path_labels = None

    def _build_phase_translations(self):
        out = []
        image_shifts = np.asarray(list(product(range(-2, 3), repeat=3)), dtype=int)
        supercell = np.asarray(self.supercell, dtype=int)
        for source_atom in range(self.natom):
            source_position = self.positions[source_atom]
            source_rows = []
            for target_atom, translation in zip(
                self.super_primitive_indices, self.super_translations
            ):
                candidates = translation[None, :] + image_shifts * supercell[None, :]
                vectors = (
                    self.positions[int(target_atom)][None, :]
                    + candidates @ self.lattice
                    - source_position[None, :]
                )
                distances = np.linalg.norm(vectors, axis=1)
                minimum = float(np.min(distances))
                selected = candidates[
                    np.abs(distances - minimum) <= 1.0e-10 * max(1.0, minimum)
                ]
                source_rows.append(np.ascontiguousarray(selected, dtype=int))
            out.append(tuple(source_rows))
        return tuple(out)

    def _evaluate_forces(self, positions):
        calculator = self.force_calculator
        if hasattr(calculator, "forces"):
            values = calculator.forces(
                self.super_symbols,
                positions,
                self.super_lattice,
            )
        elif callable(calculator):
            values = calculator(
                self.super_symbols,
                positions,
                self.super_lattice,
            )
        else:
            raise TypeError("force_calculator must be callable or define forces().")
        forces = np.asarray(values, dtype=float)
        if forces.shape != (self.nsuper, 3) or not np.all(np.isfinite(forces)):
            raise ValueError("force_calculator must return finite forces with shape (nsuper, 3).")
        if self.subtract_force_drift:
            forces = forces - np.mean(forces, axis=0, keepdims=True)
        return forces

    def _symmetrize_pair_interchange(self, force_constants):
        force_constants = np.array(force_constants, dtype=float, copy=True)
        supercell = np.asarray(self.supercell, dtype=int)
        for source_atom in range(self.natom):
            for target_index, (target_atom, translation) in enumerate(
                zip(self.super_primitive_indices, self.super_translations)
            ):
                reverse_translation = tuple(
                    int(value) for value in np.mod(-translation, supercell)
                )
                reverse_index = self._super_index[(source_atom, reverse_translation)]
                target_atom = int(target_atom)
                block = force_constants[source_atom, :, target_index, :]
                reverse = force_constants[target_atom, :, reverse_index, :].T
                average = 0.5 * (block + reverse)
                force_constants[source_atom, :, target_index, :] = average
                force_constants[target_atom, :, reverse_index, :] = average.T
        return force_constants

    def _apply_acoustic_sum_rule(self, force_constants):
        force_constants = np.array(force_constants, dtype=float, copy=True)
        residual = np.sum(force_constants, axis=2)
        for atom, target_index in enumerate(self._reference_indices):
            force_constants[atom, :, target_index, :] -= residual[atom]
        return force_constants

    def run(self):
        force_constants = np.zeros((self.natom, 3, self.nsuper, 3), dtype=float)
        self.force_history = []
        for atom in range(self.natom):
            target_index = int(self._reference_indices[atom])
            for axis in range(3):
                displaced_plus = np.array(self.super_positions, copy=True)
                displaced_minus = np.array(self.super_positions, copy=True)
                displaced_plus[target_index, axis] += self.displacement
                displaced_minus[target_index, axis] -= self.displacement

                started = time.perf_counter()
                force_plus = self._evaluate_forces(displaced_plus)
                force_minus = self._evaluate_forces(displaced_minus)
                force_constants[atom, axis] = -(
                    force_plus - force_minus
                ) / (2.0 * self.displacement)
                self.force_history.append(
                    {
                        "atom": int(atom),
                        "axis": int(axis),
                        "seconds": float(time.perf_counter() - started),
                        "max_force_difference_Ha_per_bohr": float(
                            np.max(np.abs(force_plus - force_minus))
                        ),
                    }
                )

        self.raw_force_constants = np.array(force_constants, copy=True)
        force_constants = self._symmetrize_pair_interchange(force_constants)
        if self.enforce_acoustic_sum_rule:
            for _iteration in range(2):
                force_constants = self._apply_acoustic_sum_rule(force_constants)
                force_constants = self._symmetrize_pair_interchange(force_constants)
            force_constants = self._apply_acoustic_sum_rule(force_constants)
        self.force_constants = np.ascontiguousarray(force_constants)
        self.success = True
        self.message = "force constants built"
        return self

    @property
    def acoustic_sum_rule_residual(self):
        if self.force_constants is None:
            raise RuntimeError("Run the phonon calculation first.")
        return float(np.max(np.abs(np.sum(self.force_constants, axis=2))))

    def dynamical_matrix(self, qpoint):
        if self.force_constants is None:
            raise RuntimeError("Run the phonon calculation first.")
        qpoint = np.asarray(qpoint, dtype=float)
        if qpoint.shape != (3,):
            raise ValueError("qpoint must contain three fractional reciprocal coordinates.")
        masses_au = self.masses * amu_to_au
        matrix = np.zeros((3 * self.natom, 3 * self.natom), dtype=np.complex128)
        for source_atom in range(self.natom):
            source_slice = slice(3 * source_atom, 3 * source_atom + 3)
            for target_index, target_atom in enumerate(self.super_primitive_indices):
                target_atom = int(target_atom)
                target_slice = slice(3 * target_atom, 3 * target_atom + 3)
                translations = self._phase_translations[source_atom][target_index]
                phase = np.mean(np.exp(2.0j * np.pi * (translations @ qpoint)))
                mass = np.sqrt(masses_au[source_atom] * masses_au[target_atom])
                matrix[source_slice, target_slice] += (
                    self.force_constants[source_atom, :, target_index, :] * phase / mass
                )
        return np.ascontiguousarray(0.5 * (matrix + matrix.conj().T))

    def frequencies(self, qpoint, *, units="cm-1", return_eigenvectors=False):
        matrix = self.dynamical_matrix(qpoint)
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        frequencies = np.sign(eigenvalues) * np.sqrt(np.abs(eigenvalues))
        unit_key = str(units).strip().lower().replace("^", "")
        if unit_key in ("cm-1", "cm1", "wavenumber", "wavenumbers"):
            frequencies = frequencies * au2wavenumber
        elif unit_key not in ("au", "a.u.", "hartree"):
            raise ValueError("units must be 'au' or 'cm-1'.")
        frequencies = np.asarray(frequencies, dtype=float)
        if return_eigenvectors:
            return frequencies, eigenvectors
        return frequencies

    def mode(self, qpoint, branch):
        """Return one mass-weighted phonon mode in atomic units."""

        qpoint = _normalize_qpoint(qpoint)
        matrix = self.dynamical_matrix(qpoint)
        if np.max(np.abs(matrix.imag), initial=0.0) <= 1.0e-13:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix.real)
        else:
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        branch = _normalize_branch(branch, len(eigenvalues))
        frequency = np.sign(eigenvalues[branch]) * np.sqrt(abs(eigenvalues[branch]))
        return PeriodicPhononMode(
            qpoint=qpoint,
            branch=branch,
            frequency=frequency,
            eigenvector=eigenvectors[:, branch],
            masses=self.masses * amu_to_au,
            source=type(self).__name__,
        )

    def band_structure(
        self,
        vertices,
        *,
        labels=None,
        points_per_segment=41,
        units="cm-1",
    ):
        qpoints, distances, ticks = interpolate_q_path(
            vertices,
            self.lattice,
            points_per_segment=points_per_segment,
        )
        frequencies = np.asarray(
            [self.frequencies(qpoint, units=units) for qpoint in qpoints]
        )
        if labels is not None:
            labels = tuple(str(label) for label in labels)
            if len(labels) != len(ticks):
                raise ValueError("labels must contain one entry per path vertex.")
        self.path_qpoints = qpoints
        self.path_distances = distances
        self.path_frequencies = frequencies
        self.path_ticks = ticks
        self.path_labels = labels
        return {
            "qpoints": qpoints,
            "distances": distances,
            "frequencies": frequencies,
            "ticks": ticks,
            "labels": labels,
            "units": units,
        }


Phonon = FiniteDisplacementPhonon


__all__ = [
    "FiniteDisplacementPhonon",
    "KRHFForceCalculator",
    "PeriodicPhononMode",
    "Phonon",
    "interpolate_q_path",
]
