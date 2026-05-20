"""Constraint helpers for molecular dynamics."""

import numpy as np


class FixBondLengths:
    """Constrain selected pair distances with SHAKE/RATTLE-style projection.

    The target distances are in the same length unit as the coordinates
    (Bohr for the AU-based MD helpers).
    """

    def __init__(self, pairs, distances=None, tolerance=1e-12, max_iter=100):
        self.pairs = [tuple(pair) for pair in pairs]
        self.distances = None if distances is None else np.asarray(distances, dtype=float)
        self.tolerance = float(tolerance)
        self.max_iter = int(max_iter)
        if self.distances is not None and len(self.distances) != len(self.pairs):
            raise ValueError("distances must match the number of constrained pairs.")

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

        for _ in range(self.max_iter):
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
                return

        raise RuntimeError("FixBondLengths position projection did not converge.")

    def adjust_momenta(self, atoms, momenta):
        """Remove relative velocity along constrained bonds."""
        positions = atoms.get_positions()
        masses = atoms.get_masses()

        for _ in range(self.max_iter):
            max_component = 0.0
            for i, j in self.pairs:
                delta = positions[j] - positions[i]
                distance = np.linalg.norm(delta)
                if distance == 0.0:
                    raise ValueError("Cannot constrain a zero-length bond.")
                direction = delta / distance
                velocities = momenta / masses[:, None]
                component = np.dot(velocities[j] - velocities[i], direction)
                max_component = max(max_component, abs(component))
                if abs(component) <= self.tolerance:
                    continue
                impulse = component / (1.0 / masses[i] + 1.0 / masses[j])
                momenta[i] += impulse * direction
                momenta[j] -= impulse * direction
            if max_component <= self.tolerance:
                return

        raise RuntimeError("FixBondLengths momentum projection did not converge.")

    def adjust_forces(self, atoms, forces):
        """Project out force components that would stretch constrained bonds."""
        self.adjust_momenta(atoms, forces)

    def get_removed_degrees_of_freedom(self, atoms):
        """Return the number of scalar constraints."""
        return len(self.pairs)
