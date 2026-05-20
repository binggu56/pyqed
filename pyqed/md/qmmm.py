"""QM/MM calculator glue for :mod:`pyqed.md`."""

import numpy as np


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
        qm_build_driver=None,
        exclude_qm_coulomb=True,
        exclude_qm_qm_lj=True,
    ):
        self.qm = qm
        self.mm = mm
        self.qm_indices = None if qm_indices is None else np.asarray(qm_indices, dtype=int)
        self.mm_indices = None if mm_indices is None else np.asarray(mm_indices, dtype=int)
        self.electrostatic_embedding = bool(electrostatic_embedding)
        self.charge_array = str(charge_array)
        self.mm_charges = None if mm_charges is None else np.asarray(mm_charges, dtype=float)
        self.qm_run_kwargs = {} if qm_run_kwargs is None else dict(qm_run_kwargs)
        self.qm_build_driver = qm_build_driver
        self.exclude_qm_coulomb = bool(exclude_qm_coulomb)
        self.exclude_qm_qm_lj = bool(exclude_qm_qm_lj)
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
        embedded = self._embed_point_charges(positions[mm_indices], point_charges)
        qm_energy, qm_grad, point_charge_forces = embedded.energy_and_gradients()
        qm_forces = -np.asarray(qm_grad, dtype=float)
        point_charge_forces = np.asarray(point_charge_forces, dtype=float)

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
            "mm_forces": np.asarray(mm_forces, dtype=float).copy(),
            "qm_forces": np.asarray(qm_forces, dtype=float).copy(),
            "point_charge_forces": point_charge_forces.copy(),
            "qm_force_max": _max_force_norm(qm_forces),
            "mm_force_max": _max_force_norm(mm_forces),
            "point_charge_force_max": _max_force_norm(point_charge_forces),
            "total_force_max": _max_force_norm(forces),
        }
        return energy, forces

    def _embed_point_charges(self, coords, charges):
        from pyqed.qchem import embed_point_charges

        kwargs = {
            "run_kwargs": self.qm_run_kwargs,
        }
        if self.qm_build_driver is not None:
            kwargs["build_driver"] = self.qm_build_driver
        return embed_point_charges(self.qm, coords, charges, **kwargs)

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
