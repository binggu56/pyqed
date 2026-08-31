"""QM/MM calculator glue for :mod:`pyqed.md`."""

from dataclasses import dataclass

import numpy as np

from .neighborlist import minimum_image as _minimum_image


@dataclass
class _EmbeddingCharges:
    coords: np.ndarray
    charges: np.ndarray
    owners: np.ndarray
    shifts: np.ndarray


class QMMM:
    """Compose QM and MM calculators behind the MD calculator interface.

    The calculator objects can expose either ``calculate(atoms) -> (energy,
    forces)`` or the pair of ``get_potential_energy(atoms)`` and
    ``get_forces(atoms)`` methods.

    With ``electrostatic_embedding=True``, ``qm`` is a PyQED qchem mean-field
    object for the QM atom subset and the MM atom charges are passed to
    :func:`pyqed.qchem.embed_point_charges` at each calculation.
    """

    def __init__(
        self,
        qm=None,
        mm=None,
        qm_indices=None,
        mm_indices=None,
        electrostatic_embedding=False,
        charge_array="charges",
        mm_charges=None,
        qm_run_kwargs=None,
        exclude_qm_coulomb=True,
        exclude_qm_qm_lj=True,
        embedding_pbc=None,
        embedding_cutoff=None,
    ):
        self.qm = qm
        self.mm = mm
        self.qm_indices = None if qm_indices is None else np.asarray(qm_indices, dtype=int)
        self.mm_indices = None if mm_indices is None else np.asarray(mm_indices, dtype=int)
        self.electrostatic_embedding = bool(electrostatic_embedding)
        self.charge_array = str(charge_array)
        self.mm_charges = None if mm_charges is None else np.asarray(mm_charges, dtype=float)
        self.qm_run_kwargs = {} if qm_run_kwargs is None else dict(qm_run_kwargs)
        self.exclude_qm_coulomb = bool(exclude_qm_coulomb)
        self.exclude_qm_qm_lj = bool(exclude_qm_qm_lj)
        self.embedding_pbc = _normalize_embedding_pbc(embedding_pbc)
        self.embedding_cutoff = None if embedding_cutoff is None else float(embedding_cutoff)
        self.atoms = None
        self.results = {}

    def set_atoms(self, atoms):
        self.atoms = atoms
        for calculator in (self.qm, self.mm):
            if hasattr(calculator, "set_atoms"):
                calculator.set_atoms(atoms)

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
            raise RuntimeError("QMMM calculator has no atoms.")

        forces = np.zeros_like(np.asarray(atoms.get_positions(), dtype=float))
        energy = 0.0
        if self.electrostatic_embedding:
            return self._calculate_electrostatic_embedding(atoms, energy, forces)

        components = {
            "qm_energy": 0.0,
            "mm_energy": 0.0,
            "qm_force_max": 0.0,
            "mm_force_max": 0.0,
        }
        for calculator in (self.qm, self.mm):
            if calculator is None:
                continue
            contribution_energy, contribution_forces = self._calculate_one(
                calculator, atoms
            )
            energy += contribution_energy
            forces += contribution_forces
            key = "qm" if calculator is self.qm else "mm"
            components[f"{key}_energy"] += contribution_energy
            components[f"{key}_force_max"] = _max_force_norm(contribution_forces)
        self.results = {
            "energy": float(energy),
            "forces": np.asarray(forces, dtype=float).copy(),
            "electrostatic_embedding": False,
            **components,
            "total_force_max": _max_force_norm(forces),
        }
        return energy, forces

    def _calculate_electrostatic_embedding(self, atoms, energy, forces):
        if self.qm is None:
            raise RuntimeError("Electrostatic embedding requires a QM mean-field object.")

        positions = np.asarray(atoms.get_positions(), dtype=float)
        qm_indices = self._qm_indices(atoms)
        mm_indices = self._mm_indices(atoms, qm_indices)
        if len(qm_indices) != self.qm.mol.natom:
            raise ValueError(
                f"QM region has {len(qm_indices)} atoms, but the QM molecule has "
                f"{self.qm.mol.natom} atoms."
            )

        mm_energy = 0.0
        mm_forces = np.zeros_like(forces)
        if self.mm is not None:
            mm_energy, mm_forces = self._calculate_mm_for_embedding(
                atoms,
                qm_indices,
                len(positions),
            )
            energy += mm_energy
            forces += mm_forces

        self.qm.mol.set_geom(positions[qm_indices])
        point_charges = self._point_charges(atoms, mm_indices)
        embedding = self._embedding_point_charges(
            atoms,
            positions,
            qm_indices,
            mm_indices,
            point_charges,
        )
        embedded = self._embed_point_charges(embedding.coords, embedding.charges)
        qm_energy, qm_grad, embedding_point_charge_forces = embedded.energy_and_gradients()
        qm_forces = -np.asarray(qm_grad, dtype=float)
        embedding_point_charge_forces = np.asarray(embedding_point_charge_forces, dtype=float)
        point_charge_forces = _sum_image_forces(
            embedding_point_charge_forces,
            embedding.owners,
            len(mm_indices),
        )

        energy += qm_energy
        forces[qm_indices] += qm_forces
        forces[mm_indices] += point_charge_forces
        self.results = {
            "energy": float(energy),
            "forces": np.asarray(forces, dtype=float).copy(),
            "electrostatic_embedding": True,
            "qm_energy": float(qm_energy),
            "mm_energy": float(mm_energy),
            "embedding_energy": float(qm_energy),
            "qm_indices": qm_indices.copy(),
            "mm_indices": mm_indices.copy(),
            "mm_charges": np.asarray(point_charges, dtype=float).copy(),
            "embedding_pbc": self.embedding_pbc,
            "embedding_cutoff": self.embedding_cutoff,
            "embedding_coords": embedding.coords.copy(),
            "embedding_charges": embedding.charges.copy(),
            "embedding_owners": embedding.owners.copy(),
            "embedding_shifts": embedding.shifts.copy(),
            "mm_forces": np.asarray(mm_forces, dtype=float).copy(),
            "qm_forces": np.asarray(qm_forces, dtype=float).copy(),
            "point_charge_forces": point_charge_forces.copy(),
            "embedding_point_charge_forces": embedding_point_charge_forces.copy(),
            "qm_force_max": _max_force_norm(qm_forces),
            "mm_force_max": _max_force_norm(mm_forces),
            "point_charge_force_max": _max_force_norm(point_charge_forces),
            "total_force_max": _max_force_norm(forces),
        }
        return energy, forces

    def _embedding_point_charges(self, atoms, positions, qm_indices, mm_indices, charges):
        mm_coords = np.asarray(positions[mm_indices], dtype=float)
        owners = np.arange(len(mm_indices), dtype=int)
        shifts = np.zeros_like(mm_coords)
        if self.embedding_pbc == "none" or len(mm_coords) == 0:
            return _EmbeddingCharges(mm_coords, charges, owners, shifts)

        cell = np.asarray(atoms.get_cell(), dtype=float)
        pbc = np.asarray(atoms.get_pbc(), dtype=bool)
        _validate_periodic_embedding_cell(cell, pbc)
        qm_coords = np.asarray(positions[qm_indices], dtype=float)
        if self.embedding_pbc == "nearest":
            center = np.mean(qm_coords, axis=0)
            nearest = np.array(
                [center + _minimum_image(coord - center, cell, pbc) for coord in mm_coords],
                dtype=float,
            )
            return _EmbeddingCharges(nearest, charges, owners, nearest - mm_coords)

        if self.embedding_pbc == "images":
            if self.embedding_cutoff is None or self.embedding_cutoff <= 0.0:
                raise ValueError("embedding_cutoff must be positive for embedding_pbc='images'.")
            return _image_expanded_embedding_charges(
                mm_coords,
                charges,
                qm_coords,
                cell,
                pbc,
                self.embedding_cutoff,
            )

        raise ValueError(f"Unknown embedding_pbc mode {self.embedding_pbc!r}.")

    def _embed_point_charges(self, coords, charges):
        from pyqed.qchem import embed_point_charges

        return embed_point_charges(
            self.qm,
            coords,
            charges,
            run_kwargs=self.qm_run_kwargs,
        )

    def _calculate_mm_for_embedding(self, atoms, qm_indices, natoms):
        extra_coulomb_exclusions = None
        if self.exclude_qm_coulomb:
            extra_coulomb_exclusions = _pairs_touching(qm_indices, natoms)

        extra_lj_exclusions = None
        if self.exclude_qm_qm_lj:
            extra_lj_exclusions = _pairs_within(qm_indices)

        if _accepts_extra_exclusions(self.mm):
            energy, forces = self.mm.calculate(
                atoms,
                extra_lj_exclusions=extra_lj_exclusions,
                extra_coulomb_exclusions=extra_coulomb_exclusions,
            )
            return float(energy), np.asarray(forces, dtype=float)

        return self._calculate_one(self.mm, atoms)

    def _qm_indices(self, atoms):
        if self.qm_indices is None:
            if self.mm_indices is None:
                raise ValueError("qm_indices are required for electrostatic embedding.")
            mask = np.ones(len(atoms), dtype=bool)
            mask[self.mm_indices] = False
            return np.nonzero(mask)[0]
        return self.qm_indices

    def _mm_indices(self, atoms, qm_indices):
        if self.mm_indices is not None:
            return self.mm_indices
        mask = np.ones(len(atoms), dtype=bool)
        mask[qm_indices] = False
        return np.nonzero(mask)[0]

    def _point_charges(self, atoms, mm_indices):
        if self.mm_charges is not None:
            charges = self.mm_charges
            if len(charges) == len(atoms):
                charges = charges[mm_indices]
        else:
            if not hasattr(atoms, "arrays") or self.charge_array not in atoms.arrays:
                raise ValueError(
                    f"MM point charges require atoms array {self.charge_array!r} "
                    "or explicit mm_charges."
                )
            charges = np.asarray(atoms.arrays[self.charge_array], dtype=float)[mm_indices]
        if len(charges) != len(mm_indices):
            raise ValueError(
                f"Expected {len(mm_indices)} MM point charges, got {len(charges)}."
            )
        return np.asarray(charges, dtype=float)

    @staticmethod
    def _calculate_one(calculator, atoms):
        if hasattr(calculator, "calculate"):
            energy, forces = calculator.calculate(atoms)
        else:
            energy = calculator.get_potential_energy(atoms)
            forces = calculator.get_forces(atoms)
        return float(energy), np.asarray(forces, dtype=float)


def _accepts_extra_exclusions(calculator):
    try:
        import inspect

        signature = inspect.signature(calculator.calculate)
    except (AttributeError, TypeError, ValueError):
        return False
    if any(parameter.kind == parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return True
    return {
        "extra_lj_exclusions",
        "extra_coulomb_exclusions",
    }.issubset(signature.parameters)


def _normalize_embedding_pbc(mode):
    if mode is None or mode is False:
        return "none"
    if mode is True:
        return "nearest"
    mode = str(mode).lower()
    aliases = {
        "off": "none",
        "false": "none",
        "no": "none",
        "minimum_image": "nearest",
        "minimum-image": "nearest",
        "image": "images",
        "cutoff": "images",
        "real_space": "images",
        "real-space": "images",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"none", "nearest", "images"}:
        raise ValueError("embedding_pbc must be one of None, 'nearest', or 'images'.")
    return mode


def _validate_periodic_embedding_cell(cell, pbc):
    if not np.any(pbc):
        raise ValueError("Periodic embedding requires atoms.pbc to be enabled.")
    if np.linalg.matrix_rank(cell) < 3:
        raise ValueError("Periodic embedding requires a full 3D cell.")


def _image_expanded_embedding_charges(mm_coords, charges, qm_coords, cell, pbc, cutoff):
    shifts = _lattice_shifts(cell, pbc, cutoff)
    coords = []
    expanded_charges = []
    owners = []
    expanded_shifts = []
    cutoff2 = float(cutoff) ** 2
    for owner, (coord, charge) in enumerate(zip(mm_coords, charges)):
        for shift in shifts:
            image = coord + shift
            deltas = qm_coords - image
            if np.min(np.einsum("ax,ax->a", deltas, deltas)) > cutoff2:
                continue
            coords.append(image)
            expanded_charges.append(charge)
            owners.append(owner)
            expanded_shifts.append(shift)

    if not coords:
        return _EmbeddingCharges(
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=int),
            np.empty((0, 3), dtype=float),
        )
    return _EmbeddingCharges(
        np.asarray(coords, dtype=float),
        np.asarray(expanded_charges, dtype=float),
        np.asarray(owners, dtype=int),
        np.asarray(expanded_shifts, dtype=float),
    )


def _lattice_shifts(cell, pbc, cutoff):
    cell = np.asarray(cell, dtype=float)
    pbc = np.asarray(pbc, dtype=bool)
    ranges = []
    for axis in range(3):
        if not pbc[axis]:
            ranges.append(range(0, 1))
            continue
        length = np.linalg.norm(cell[axis])
        if length <= 0.0:
            raise ValueError("Periodic embedding requires nonzero cell vectors.")
        nmax = int(np.ceil(float(cutoff) / length)) + 1
        ranges.append(range(-nmax, nmax + 1))

    shifts = []
    for i in ranges[0]:
        for j in ranges[1]:
            for k in ranges[2]:
                shifts.append(np.array([i, j, k], dtype=float) @ cell)
    return shifts


def _sum_image_forces(image_forces, owners, nowners):
    out = np.zeros((nowners, 3), dtype=float)
    if len(image_forces):
        np.add.at(out, np.asarray(owners, dtype=int), np.asarray(image_forces, dtype=float))
    return out


def _pairs_touching(indices, natoms):
    indices = np.asarray(indices, dtype=int)
    qm = set(int(index) for index in indices)
    pairs = set()
    for i in range(natoms):
        for j in range(i + 1, natoms):
            if i in qm or j in qm:
                pairs.add((i, j))
    return pairs


def _pairs_within(indices):
    indices = [int(index) for index in np.asarray(indices, dtype=int)]
    pairs = set()
    for offset, i in enumerate(indices[:-1]):
        for j in indices[offset + 1:]:
            pairs.add(tuple(sorted((i, j))))
    return pairs


def _max_force_norm(forces):
    forces = np.asarray(forces, dtype=float)
    if forces.size == 0:
        return 0.0
    return float(np.max(np.linalg.norm(forces.reshape(-1, 3), axis=1)))
