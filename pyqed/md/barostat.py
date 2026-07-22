"""Simple box-control helpers for membrane-shaped MD systems."""

import numpy as np

from pyqed.units import au2angstrom, au2k


AU_PRESSURE_TO_BAR = 2.9421912e8
BAR_TO_AU_PRESSURE = 1.0 / AU_PRESSURE_TO_BAR


def _orthorhombic_lengths_and_volume(atoms):
    lengths = np.asarray(atoms.get_cell().lengths(), dtype=float)
    if lengths.shape != (3,) or np.any(lengths <= 0.0):
        raise ValueError("pressure control requires positive orthorhombic cell lengths.")
    volume = float(np.prod(lengths))
    if volume <= 0.0:
        raise ValueError("pressure control requires a positive cell volume.")
    return lengths, volume


def instantaneous_pressure_tensor(atoms, forces=None, include_kinetic=True):
    """Return the instantaneous pressure tensor in Hartree / bohr^3.

    The kinetic term is exact for the current atomic momenta.  The
    configurational virial uses a centered global ``sum r_i outer F_i`` estimate,
    which is suitable for weak pressure coupling and diagnostics while pairwise
    virial accumulation is still being added to the native calculators.
    """

    _, volume = _orthorhombic_lengths_and_volume(atoms)
    pressure = np.zeros((3, 3), dtype=float)

    if include_kinetic:
        momenta = atoms.get_momenta()
        masses = atoms.get_masses()
        pressure += momenta.T @ (momenta / masses[:, np.newaxis])

    calculator = getattr(atoms, "_calc", None)
    if forces is None and calculator is not None and hasattr(calculator, "get_virial"):
        pressure += np.asarray(calculator.get_virial(atoms), dtype=float)
    else:
        if forces is None:
            forces = atoms.get_forces(apply_constraint=False)
        forces = np.asarray(forces, dtype=float)
        positions = atoms.get_positions()
        if forces.shape != positions.shape:
            raise ValueError(f"forces must have shape {positions.shape}, got {forces.shape}.")
        centered_positions = positions - positions.mean(axis=0)
        pressure += centered_positions.T @ forces

    return pressure / volume


def semi_isotropic_pressure(atoms, forces=None, include_kinetic=True):
    """Return lateral pressure, normal pressure, and full pressure tensor."""

    tensor = instantaneous_pressure_tensor(
        atoms,
        forces=forces,
        include_kinetic=include_kinetic,
    )
    lateral = 0.5 * float(tensor[0, 0] + tensor[1, 1])
    normal = float(tensor[2, 2])
    return lateral, normal, tensor


class SemiIsotropicBoxController:
    """Weakly relax ``xy`` area and ``z`` length toward target values.

    This is a development-time semi-isotropic box controller, not a rigorous
    pressure barostat.  It is useful for testing membrane workflows that need
    independent lateral and normal box scaling before a full virial barostat is
    available.
    """

    def __init__(
        self,
        atoms,
        target_area=None,
        target_z=None,
        coupling=0.01,
        max_scale=0.01,
    ):
        self.atoms = atoms
        self.target_area = None if target_area is None else float(target_area)
        self.target_z = None if target_z is None else float(target_z)
        self.coupling = float(coupling)
        self.max_scale = float(max_scale)
        if self.coupling < 0.0 or self.coupling > 1.0:
            raise ValueError("coupling must be in [0, 1].")
        if self.max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")

    @classmethod
    def from_angstrom(cls, atoms, target_area_angstrom2=None, target_z_angstrom=None, **kwargs):
        target_area = None if target_area_angstrom2 is None else target_area_angstrom2 / au2angstrom**2
        target_z = None if target_z_angstrom is None else target_z_angstrom / au2angstrom
        return cls(atoms, target_area=target_area, target_z=target_z, **kwargs)

    def __call__(self):
        lengths = np.asarray(self.atoms.get_cell().lengths(), dtype=float)
        if np.any(lengths <= 0.0):
            raise ValueError("box controller requires positive orthorhombic cell lengths.")
        scale = np.ones(3)
        if self.target_area is not None:
            current_area = lengths[0] * lengths[1]
            lateral = np.sqrt(self.target_area / current_area)
            lateral = self._relaxed_scale(lateral)
            scale[:2] = lateral
        if self.target_z is not None:
            normal = self.target_z / lengths[2]
            scale[2] = self._relaxed_scale(normal)
        if np.allclose(scale, 1.0):
            return scale
        new_cell = np.diag(lengths * scale)
        positions = self.atoms.get_positions() * scale
        self.atoms.set_cell(new_cell, scale_atoms=False)
        self.atoms.set_positions(positions)
        return scale

    def _relaxed_scale(self, raw_scale):
        scale = 1.0 + self.coupling * (float(raw_scale) - 1.0)
        return float(np.clip(scale, 1.0 - self.max_scale, 1.0 + self.max_scale))


class SemiIsotropicPressureController:
    """Weak semi-isotropic pressure controller for membrane-shaped boxes.

    This is a Berendsen-style production-prep controller: it independently
    scales the lateral ``xy`` area and normal ``z`` length from instantaneous
    pressure estimates.  It is intentionally simple and deterministic; rigorous
    long-time NPT sampling still needs a full virial barostat.
    """

    def __init__(
        self,
        atoms,
        target_lateral_pressure=0.0,
        target_normal_pressure=None,
        compressibility=1e-5,
        coupling=0.01,
        max_scale=0.01,
        include_kinetic=True,
        scale_molecule_centers=False,
        molecule_array="molecule_ids",
    ):
        self.atoms = atoms
        self.target_lateral_pressure = float(target_lateral_pressure)
        self.target_normal_pressure = (
            self.target_lateral_pressure
            if target_normal_pressure is None
            else float(target_normal_pressure)
        )
        self.compressibility = float(compressibility)
        self.coupling = float(coupling)
        self.max_scale = float(max_scale)
        self.include_kinetic = bool(include_kinetic)
        self.scale_molecule_centers = bool(scale_molecule_centers)
        self.molecule_array = str(molecule_array)
        if self.compressibility < 0.0:
            raise ValueError("compressibility must be non-negative.")
        if self.coupling < 0.0 or self.coupling > 1.0:
            raise ValueError("coupling must be in [0, 1].")
        if self.max_scale <= 0.0:
            raise ValueError("max_scale must be positive.")
        self.last_pressure_tensor = None
        self.last_lateral_pressure = None
        self.last_normal_pressure = None
        self.last_scale = np.ones(3)
        self.calls = 0

    @classmethod
    def from_bar(
        cls,
        atoms,
        target_lateral_pressure_bar=1.0,
        target_normal_pressure_bar=None,
        compressibility_bar=4.5e-5,
        **kwargs,
    ):
        target_normal = (
            None
            if target_normal_pressure_bar is None
            else target_normal_pressure_bar * BAR_TO_AU_PRESSURE
        )
        return cls(
            atoms,
            target_lateral_pressure=target_lateral_pressure_bar * BAR_TO_AU_PRESSURE,
            target_normal_pressure=target_normal,
            compressibility=compressibility_bar * AU_PRESSURE_TO_BAR,
            **kwargs,
        )

    def __call__(self):
        return self.apply()

    def apply(self, forces=None):
        lateral_pressure, normal_pressure, tensor = semi_isotropic_pressure(
            self.atoms,
            forces=forces,
            include_kinetic=self.include_kinetic,
        )
        self.last_pressure_tensor = tensor
        self.last_lateral_pressure = lateral_pressure
        self.last_normal_pressure = normal_pressure

        lateral_scale = self._pressure_scale(
            lateral_pressure,
            self.target_lateral_pressure,
        )
        normal_scale = self._pressure_scale(
            normal_pressure,
            self.target_normal_pressure,
        )
        scale = np.array([lateral_scale, lateral_scale, normal_scale], dtype=float)
        self.last_scale = scale
        self.calls += 1
        if np.allclose(scale, 1.0):
            return scale

        lengths, _ = _orthorhombic_lengths_and_volume(self.atoms)
        if self.scale_molecule_centers:
            positions = _scaled_molecule_center_positions(
                self.atoms,
                scale,
                self.molecule_array,
            )
        else:
            positions = self.atoms.get_positions() * scale
        self.atoms.set_cell(np.diag(lengths * scale), scale_atoms=False)
        self.atoms.set_positions(positions)
        return scale

    def _pressure_scale(self, current_pressure, target_pressure):
        raw_scale = 1.0 + self.coupling * self.compressibility * (
            float(current_pressure) - float(target_pressure)
        )
        return float(np.clip(raw_scale, 1.0 - self.max_scale, 1.0 + self.max_scale))


class MonteCarloSemiIsotropicBarostat:
    """Metropolis semi-isotropic barostat for membrane-shaped boxes.

    The barostat proposes either an ``xy`` area move at fixed ``z`` or a
    normal ``z`` move at fixed area.  Acceptance uses atomic units:
    ``exp[-beta * (dU + work) + N * log(J)]`` where ``J`` is the coordinate
    scaling Jacobian for the proposed move.
    """

    def __init__(
        self,
        atoms,
        temperature_K=None,
        temperature=None,
        target_lateral_pressure=0.0,
        target_normal_pressure=None,
        max_area_change=0.02,
        max_z_change=0.02,
        move="area-or-normal",
        scale_molecule_centers=False,
        molecule_array="molecule_ids",
        seed=None,
    ):
        self.atoms = atoms
        self.temperature = _temperature_to_au(temperature=temperature, temperature_K=temperature_K)
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive.")
        self.beta = 1.0 / self.temperature
        self.target_lateral_pressure = float(target_lateral_pressure)
        self.target_normal_pressure = (
            self.target_lateral_pressure
            if target_normal_pressure is None
            else float(target_normal_pressure)
        )
        self.max_area_change = float(max_area_change)
        self.max_z_change = float(max_z_change)
        self.move = str(move).lower().replace("_", "-")
        self.scale_molecule_centers = bool(scale_molecule_centers)
        self.molecule_array = str(molecule_array)
        self.rng = np.random.default_rng(seed)
        if self.max_area_change <= 0.0 or self.max_z_change <= 0.0:
            raise ValueError("max_area_change and max_z_change must be positive.")
        if self.move not in {"area", "normal", "area-or-normal"}:
            raise ValueError("move must be 'area', 'normal', or 'area-or-normal'.")
        self.attempts = 0
        self.accepted = 0
        self.last_accepted = None
        self.last_move = None
        self.last_scale = np.ones(3)
        self.last_old_energy = 0.0
        self.last_new_energy = 0.0
        self.last_delta_energy = 0.0
        self.last_work = 0.0
        self.last_log_jacobian = 0.0
        self.last_log_acceptance = 0.0

    @classmethod
    def from_bar(
        cls,
        atoms,
        temperature_K=None,
        temperature=None,
        target_lateral_pressure_bar=1.0,
        target_normal_pressure_bar=None,
        **kwargs,
    ):
        target_normal = (
            None
            if target_normal_pressure_bar is None
            else target_normal_pressure_bar * BAR_TO_AU_PRESSURE
        )
        return cls(
            atoms,
            temperature_K=temperature_K,
            temperature=temperature,
            target_lateral_pressure=target_lateral_pressure_bar * BAR_TO_AU_PRESSURE,
            target_normal_pressure=target_normal,
            **kwargs,
        )

    @property
    def acceptance_rate(self):
        if self.attempts == 0:
            return 0.0
        return self.accepted / self.attempts

    def __call__(self):
        return self.apply()

    def apply(self):
        old_positions = self.atoms.get_positions()
        old_cell = np.asarray(self.atoms.get_cell(), dtype=float)
        lengths, _volume = _orthorhombic_lengths_and_volume(self.atoms)
        old_energy = float(self.atoms.get_potential_energy())
        move, scale, work, log_jacobian = self._proposal(lengths)

        self.attempts += 1
        self.last_move = move
        self.last_scale = scale
        self.last_work = work
        self.last_log_jacobian = log_jacobian

        self._scale(scale)
        new_energy = float(self.atoms.get_potential_energy())
        delta_energy = new_energy - old_energy
        log_acceptance = -self.beta * (delta_energy + work) + log_jacobian
        self.last_old_energy = old_energy
        self.last_new_energy = new_energy
        self.last_delta_energy = delta_energy
        self.last_log_acceptance = log_acceptance

        if log_acceptance >= 0.0 or np.log(self.rng.random()) < log_acceptance:
            self.accepted += 1
            self.last_accepted = True
            return True

        self.atoms.set_cell(old_cell, scale_atoms=False)
        self.atoms.set_positions(old_positions)
        self.last_accepted = False
        return False

    def _proposal(self, lengths):
        if self.move == "area-or-normal":
            move = "area" if self.rng.random() < 0.5 else "normal"
        else:
            move = self.move
        area = float(lengths[0] * lengths[1])
        z_length = float(lengths[2])
        if move == "area":
            log_area_scale = self.rng.uniform(-self.max_area_change, self.max_area_change)
            lateral_scale = np.exp(0.5 * log_area_scale)
            new_area = area * np.exp(log_area_scale)
            scale = np.array([lateral_scale, lateral_scale, 1.0], dtype=float)
            work = self.target_lateral_pressure * z_length * (new_area - area)
            log_jacobian = len(self.atoms) * log_area_scale
        else:
            log_z_scale = self.rng.uniform(-self.max_z_change, self.max_z_change)
            normal_scale = np.exp(log_z_scale)
            new_z = z_length * normal_scale
            scale = np.array([1.0, 1.0, normal_scale], dtype=float)
            work = self.target_normal_pressure * area * (new_z - z_length)
            log_jacobian = len(self.atoms) * log_z_scale
        return move, scale, float(work), float(log_jacobian)

    def _scale(self, scale):
        lengths, _volume = _orthorhombic_lengths_and_volume(self.atoms)
        if self.scale_molecule_centers:
            positions = _scaled_molecule_center_positions(
                self.atoms,
                scale,
                self.molecule_array,
            )
        else:
            positions = self.atoms.get_positions() * scale
        self.atoms.set_cell(np.diag(lengths * scale), scale_atoms=False)
        self.atoms.set_positions(positions)


def _temperature_to_au(temperature=None, temperature_K=None):
    if temperature_K is not None:
        return float(temperature_K) / au2k
    if temperature is None:
        raise ValueError("temperature or temperature_K is required.")
    return float(temperature)


def _scaled_molecule_center_positions(atoms, scale, molecule_array):
    if not atoms.has(molecule_array):
        raise ValueError(
            f"scale_molecule_centers requires atoms array {molecule_array!r}."
        )
    positions = atoms.get_positions()
    molecule_ids = atoms.get_array(molecule_array)
    _unique_ids, inverse, counts = np.unique(
        molecule_ids,
        return_inverse=True,
        return_counts=True,
    )
    centers = np.column_stack(
        [
            np.bincount(inverse, weights=positions[:, axis]) / counts
            for axis in range(3)
        ]
    )
    return positions + centers[inverse] * (np.asarray(scale, dtype=float) - 1.0)
