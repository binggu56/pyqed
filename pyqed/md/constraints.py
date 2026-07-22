"""Constraint helpers for molecular dynamics."""

import numpy as np

_ADJUST_POSITIONS_NUMBA = None
_ADJUST_POSITIONS_NUMBA_UNAVAILABLE = False
_ADJUST_MOMENTA_NUMBA = None
_ADJUST_MOMENTA_NUMBA_UNAVAILABLE = False


class FixBondLengths:
    """Constrain selected pair distances with SHAKE/RATTLE-style projection.

    The target distances are in the same length unit as the coordinates
    (Bohr for the AU-based MD helpers).
    """

    def __init__(self, pairs, distances=None, tolerance=1e-12, max_iter=100):
        self.pairs = [tuple(pair) for pair in pairs]
        self._pair_indices = np.asarray(self.pairs, dtype=int).reshape(-1, 2)
        self._pair_color_indices = _constraint_color_indices(self._pair_indices)
        self._color_indices_flat, self._color_offsets = _flatten_color_indices(
            self._pair_color_indices,
        )
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
        projected = _adjust_positions_numba(
            positions,
            masses,
            self._pair_indices,
            self._color_indices_flat,
            self._color_offsets,
            targets,
            self.tolerance,
            self.max_iter,
        )
        if projected is not None:
            iterations, max_error, converged, zero_length = projected
            self.last_position_iterations = iterations
            self.last_position_error = max_error
            if zero_length:
                raise ValueError("Cannot constrain a zero-length bond.")
            if converged:
                return
            raise RuntimeError(
                "FixBondLengths position projection did not converge "
                f"after {self.max_iter} iterations; max error={max_error:.6e}."
            )

        for iteration in range(1, self.max_iter + 1):
            max_error = 0.0
            for color_indices in self._pair_color_indices:
                pair_i = self._pair_indices[color_indices, 0]
                pair_j = self._pair_indices[color_indices, 1]
                color_targets = targets[color_indices]
                delta = positions[pair_j] - positions[pair_i]
                distances = np.linalg.norm(delta, axis=1)
                if np.any(distances == 0.0):
                    raise ValueError("Cannot constrain a zero-length bond.")
                errors = distances - color_targets
                if len(errors):
                    max_error = max(max_error, float(np.max(np.abs(errors))))
                active = np.abs(errors) > self.tolerance
                if not np.any(active):
                    continue
                active_i = pair_i[active]
                active_j = pair_j[active]
                correction = (errors[active] / distances[active])[:, np.newaxis] * delta[active]
                total_mass = masses[active_i] + masses[active_j]
                positions[active_i] += (masses[active_j] / total_mass)[:, np.newaxis] * correction
                positions[active_j] -= (masses[active_i] / total_mass)[:, np.newaxis] * correction
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
        projected = _adjust_momenta_numba(
            momenta,
            pair_i,
            pair_j,
            directions,
            inv_masses,
            impulse_denominators,
            self.tolerance,
            self.max_iter,
        )
        if projected is not None:
            iterations, max_component, converged = projected
            self.last_momentum_iterations = iterations
            self.last_momentum_error = max_component
            if converged:
                return
            raise RuntimeError(
                "FixBondLengths momentum projection did not converge "
                f"after {self.max_iter} iterations; max component={max_component:.6e}."
            )
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
        pair_i = self._pair_indices[:, 0]
        pair_j = self._pair_indices[:, 1]
        distances = np.linalg.norm(positions[pair_j] - positions[pair_i], axis=1)
        if len(distances) == 0:
            return 0.0
        return float(np.max(np.abs(distances - targets)))


def _constraint_color_indices(pair_indices):
    colors = []
    occupied_atoms = []
    for pair_index, pair in enumerate(np.asarray(pair_indices, dtype=int)):
        atom_i, atom_j = int(pair[0]), int(pair[1])
        for color, occupied in enumerate(occupied_atoms):
            if atom_i not in occupied and atom_j not in occupied:
                colors[color].append(pair_index)
                occupied.add(atom_i)
                occupied.add(atom_j)
                break
        else:
            colors.append([pair_index])
            occupied_atoms.append({atom_i, atom_j})
    return [np.asarray(color, dtype=int) for color in colors]


def _flatten_color_indices(color_indices):
    offsets = [0]
    flat = []
    for indices in color_indices:
        flat.extend(np.asarray(indices, dtype=int).tolist())
        offsets.append(len(flat))
    return np.asarray(flat, dtype=int), np.asarray(offsets, dtype=int)


def _adjust_positions_numba(
    positions,
    masses,
    pair_indices,
    color_indices_flat,
    color_offsets,
    targets,
    tolerance,
    max_iter,
):
    kernel = _adjust_positions_numba_kernel()
    if kernel is None:
        return None
    iterations, max_error, converged, zero_length = kernel(
        positions,
        np.asarray(masses, dtype=float),
        np.asarray(pair_indices, dtype=np.int64),
        np.asarray(color_indices_flat, dtype=np.int64),
        np.asarray(color_offsets, dtype=np.int64),
        np.asarray(targets, dtype=float),
        float(tolerance),
        int(max_iter),
    )
    return int(iterations), float(max_error), bool(converged), bool(zero_length)


def _adjust_positions_numba_kernel():
    global _ADJUST_POSITIONS_NUMBA, _ADJUST_POSITIONS_NUMBA_UNAVAILABLE
    if _ADJUST_POSITIONS_NUMBA_UNAVAILABLE:
        return None
    if _ADJUST_POSITIONS_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _ADJUST_POSITIONS_NUMBA_UNAVAILABLE = True
            return None
        try:
            _ADJUST_POSITIONS_NUMBA = njit(cache=True, fastmath=True)(
                _adjust_positions_numba_impl
            )
        except Exception:
            _ADJUST_POSITIONS_NUMBA_UNAVAILABLE = True
            return None
    return _ADJUST_POSITIONS_NUMBA


def _adjust_positions_numba_impl(
    positions,
    masses,
    pair_indices,
    color_indices_flat,
    color_offsets,
    targets,
    tolerance,
    max_iter,
):
    max_error = 0.0
    for iteration in range(1, max_iter + 1):
        max_error = 0.0
        for color in range(len(color_offsets) - 1):
            start = color_offsets[color]
            stop = color_offsets[color + 1]
            for color_index in range(start, stop):
                pair_index = color_indices_flat[color_index]
                i = pair_indices[pair_index, 0]
                j = pair_indices[pair_index, 1]
                dx = positions[j, 0] - positions[i, 0]
                dy = positions[j, 1] - positions[i, 1]
                dz = positions[j, 2] - positions[i, 2]
                distance = np.sqrt(dx * dx + dy * dy + dz * dz)
                if distance == 0.0:
                    return iteration, max_error, False, True
                error = distance - targets[pair_index]
                abs_error = abs(error)
                if abs_error > max_error:
                    max_error = abs_error
                if abs_error <= tolerance:
                    continue
                factor = error / distance
                cx = factor * dx
                cy = factor * dy
                cz = factor * dz
                total_mass = masses[i] + masses[j]
                weight_i = masses[j] / total_mass
                weight_j = masses[i] / total_mass
                positions[i, 0] += weight_i * cx
                positions[i, 1] += weight_i * cy
                positions[i, 2] += weight_i * cz
                positions[j, 0] -= weight_j * cx
                positions[j, 1] -= weight_j * cy
                positions[j, 2] -= weight_j * cz
        if max_error <= tolerance:
            return iteration, max_error, True, False
    return max_iter, max_error, False, False


def _adjust_momenta_numba(
    momenta,
    pair_i,
    pair_j,
    directions,
    inv_masses,
    impulse_denominators,
    tolerance,
    max_iter,
):
    kernel = _adjust_momenta_numba_kernel()
    if kernel is None:
        return None
    iterations, max_component, converged = kernel(
        momenta,
        np.asarray(pair_i, dtype=np.int64),
        np.asarray(pair_j, dtype=np.int64),
        np.asarray(directions, dtype=float),
        np.asarray(inv_masses, dtype=float),
        np.asarray(impulse_denominators, dtype=float),
        float(tolerance),
        int(max_iter),
    )
    return int(iterations), float(max_component), bool(converged)


def _adjust_momenta_numba_kernel():
    global _ADJUST_MOMENTA_NUMBA, _ADJUST_MOMENTA_NUMBA_UNAVAILABLE
    if _ADJUST_MOMENTA_NUMBA_UNAVAILABLE:
        return None
    if _ADJUST_MOMENTA_NUMBA is None:
        try:
            from numba import njit
        except Exception:
            _ADJUST_MOMENTA_NUMBA_UNAVAILABLE = True
            return None
        try:
            _ADJUST_MOMENTA_NUMBA = njit(cache=True)(_adjust_momenta_numba_impl)
        except Exception:
            _ADJUST_MOMENTA_NUMBA_UNAVAILABLE = True
            return None
    return _ADJUST_MOMENTA_NUMBA


def _adjust_momenta_numba_impl(
    momenta,
    pair_i,
    pair_j,
    directions,
    inv_masses,
    impulse_denominators,
    tolerance,
    max_iter,
):
    max_component = 0.0
    for iteration in range(1, max_iter + 1):
        max_component = 0.0
        for pair_index in range(len(pair_i)):
            i = pair_i[pair_index]
            j = pair_j[pair_index]
            dvx = momenta[j, 0] * inv_masses[j] - momenta[i, 0] * inv_masses[i]
            dvy = momenta[j, 1] * inv_masses[j] - momenta[i, 1] * inv_masses[i]
            dvz = momenta[j, 2] * inv_masses[j] - momenta[i, 2] * inv_masses[i]
            component = (
                dvx * directions[pair_index, 0]
                + dvy * directions[pair_index, 1]
                + dvz * directions[pair_index, 2]
            )
            abs_component = abs(component)
            if abs_component > max_component:
                max_component = abs_component
            if abs_component <= tolerance:
                continue
            impulse = component / impulse_denominators[pair_index]
            ux = impulse * directions[pair_index, 0]
            uy = impulse * directions[pair_index, 1]
            uz = impulse * directions[pair_index, 2]
            momenta[i, 0] += ux
            momenta[i, 1] += uy
            momenta[i, 2] += uz
            momenta[j, 0] -= ux
            momenta[j, 1] -= uy
            momenta[j, 2] -= uz
        if max_component <= tolerance:
            return iteration, max_component, True
    return max_iter, max_component, False
