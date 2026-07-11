"""Constraint helpers for molecular dynamics."""

import numpy as np


class FixBondLengths:
    """Constrain selected pair distances with SHAKE/RATTLE-style projection.

    The target distances are in the same length unit as the coordinates
    (Bohr for the AU-based MD helpers).
    """

    def __init__(self, pairs, distances=None, tolerance=1e-12, max_iter=100):
        self.pairs = [tuple(pair) for pair in pairs]
        self._pair_indices = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
        self.distances = None if distances is None else np.asarray(distances, dtype=float)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        self.last_position_iterations = 0
        self.last_momentum_iterations = 0
        self.last_position_error = None
        self.last_momentum_error = None
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive.")
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive.")
        if any(len(pair) != 2 for pair in self.pairs):
            raise ValueError("constraint pairs must contain two atom indices.")
        if self.distances is not None and len(self.distances) != len(self.pairs):
            raise ValueError("distances must match the number of constrained pairs.")
        if self.distances is not None and np.any(self.distances <= 0.0):
            raise ValueError("constraint distances must be positive.")

    def _targets(self, atoms):
        if self.distances is not None:
            return self.distances
        positions = atoms.get_positions()
        return np.array(
            [np.linalg.norm(positions[j] - positions[i]) for i, j in self.pairs],
            dtype=float,
        )

    def adjust_positions(self, atoms, positions):
        """Project positions onto the constrained bond lengths."""
        masses = atoms.get_masses()
        targets = self._targets(atoms)

        for iteration in range(1, self.max_iter + 1):
            max_error = 0.0
            for (i, j), target in zip(self.pairs, targets):
                delta = positions[j] - positions[i]
                distance = np.linalg.norm(delta)
                if distance == 0.0:
                    raise ValueError("Cannot constrain a zero-length bond.")
                error = distance - target
                max_error = max(max_error, abs(error))
                if abs(error) <= self.tolerance:
                    continue
                correction = (error / distance) * delta
                total_mass = masses[i] + masses[j]
                positions[i] += masses[j] / total_mass * correction
                positions[j] -= masses[i] / total_mass * correction
            if max_error <= self.tolerance:
                self.last_position_iterations = iteration
                self.last_position_error = max_error
                return

        self.last_position_iterations = self.max_iter
        self.last_position_error = max_error
        raise RuntimeError(
            "FixBondLengths position projection did not converge "
            f"after {self.max_iter} iterations; max error={max_error:.6e}."
        )

    def adjust_momenta(self, atoms, momenta):
        """Remove relative velocity along constrained bonds."""
        positions = atoms.get_positions()
        masses = atoms.get_masses()
        inv_masses = 1.0 / masses
        pair_i = self._pair_indices[:, 0]
        pair_j = self._pair_indices[:, 1]
        delta = positions[pair_j] - positions[pair_i]
        distances = np.linalg.norm(delta, axis=1)
        if np.any(distances == 0.0):
            raise ValueError("Cannot constrain a zero-length bond.")
        directions = delta / distances[:, np.newaxis]
        impulse_denominators = inv_masses[pair_i] + inv_masses[pair_j]
        for iteration in range(1, self.max_iter + 1):
            relative_velocity = (
                momenta[pair_j] * inv_masses[pair_j, np.newaxis]
                - momenta[pair_i] * inv_masses[pair_i, np.newaxis]
            )
            component = np.einsum("ij,ij->i", relative_velocity, directions)
            max_component = float(np.max(np.abs(component))) if len(component) else 0.0
            if max_component <= self.tolerance:
                self.last_momentum_iterations = iteration
                self.last_momentum_error = max_component
                return
            active = np.abs(component) > self.tolerance
            impulse = component[active] / impulse_denominators[active]
            updates = impulse[:, np.newaxis] * directions[active]
            active_i = pair_i[active]
            active_j = pair_j[active]
            for axis in range(3):
                momenta[:, axis] += np.bincount(
                    active_i,
                    weights=updates[:, axis],
                    minlength=len(momenta),
                )
                momenta[:, axis] -= np.bincount(
                    active_j,
                    weights=updates[:, axis],
                    minlength=len(momenta),
                )

        self.last_momentum_iterations = self.max_iter
        self.last_momentum_error = max_component
        raise RuntimeError(
            "FixBondLengths momentum projection did not converge "
            f"after {self.max_iter} iterations; max component={max_component:.6e}."
        )

    def adjust_forces(self, atoms, forces):
        """Project out force components that would stretch constrained bonds."""
        self.adjust_momenta(atoms, forces)

    def get_removed_degrees_of_freedom(self, atoms):
        """Return the number of scalar constraints."""
        return len(self.pairs)

    def max_error(self, atoms):
        """Return the maximum absolute constrained-distance error."""
        positions = atoms.get_positions()
        targets = self._targets(atoms)
        max_error = 0.0
        for (i, j), target in zip(self.pairs, targets):
            distance = np.linalg.norm(positions[j] - positions[i])
            max_error = max(max_error, abs(distance - target))
        return float(max_error)
