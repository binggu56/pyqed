"""Polyspherical Jacobi-tree coordinates and tensor-network KEO builders."""

from __future__ import annotations

from dataclasses import dataclass
from math import comb
import numpy as np


@dataclass(frozen=True)
class JacobiVector:
    """One center-of-mass Jacobi vector in a binary fragment tree."""

    index: int
    left_atoms: tuple[int, ...]
    right_atoms: tuple[int, ...]
    left_mass: float
    right_mass: float

    @property
    def reduced_mass(self):
        return self.left_mass * self.right_mass / (
            self.left_mass + self.right_mass
        )


class _Node:
    def __init__(self, *, atom=None, left=None, right=None, mass=0.0, atoms=()):
        self.atom = atom
        self.left = left
        self.right = right
        self.mass = float(mass)
        self.atoms = tuple(atoms)
        self.vector_index = None

    @property
    def is_leaf(self):
        return self.atom is not None


class PolysphericalTree:
    r"""Body-fixed polyspherical coordinates generated from a Jacobi tree.

    ``tree`` is a nested binary tuple whose leaves are atom indices. For
    example, ``((0, 1), (2, 3))`` describes the AB + CD fragmentation tree.
    Internal nodes are ordered root-first, so vector zero is the separation
    of the two root fragments.

    Coordinates are grouped by Jacobi vector::

        (r0, r1, theta1, r2, theta2, phi2, ...)

    Vector zero defines the body-fixed z axis. Vector one lies in the xz
    plane, removing the final overall rotational degree of freedom.
    """

    def __init__(self, tree, masses):
        masses = np.asarray(masses, dtype=float)
        if masses.ndim != 1 or masses.size < 2:
            raise ValueError("masses must be a one-dimensional array for N >= 2")
        if not np.all(np.isfinite(masses)) or np.any(masses <= 0.0):
            raise ValueError("masses must be positive and finite")
        self.masses = masses
        self.natoms = int(masses.size)
        seen = []

        def parse(spec):
            if isinstance(spec, (int, np.integer)):
                atom = int(spec)
                if not 0 <= atom < self.natoms:
                    raise ValueError(f"atom index {atom} is out of range")
                seen.append(atom)
                return _Node(
                    atom=atom,
                    mass=masses[atom],
                    atoms=(atom,),
                )
            if not isinstance(spec, (tuple, list)) or len(spec) != 2:
                raise ValueError(
                    "tree nodes must be atom indices or binary (left, right) pairs"
                )
            left, right = map(parse, spec)
            return _Node(
                left=left,
                right=right,
                mass=left.mass + right.mass,
                atoms=left.atoms + right.atoms,
            )

        self._root = parse(tree)
        if sorted(seen) != list(range(self.natoms)):
            raise ValueError("tree must contain every atom index exactly once")

        vectors = []

        def assign(node):
            if node.is_leaf:
                return
            node.vector_index = len(vectors)
            vectors.append(
                JacobiVector(
                    index=node.vector_index,
                    left_atoms=node.left.atoms,
                    right_atoms=node.right.atoms,
                    left_mass=node.left.mass,
                    right_mass=node.right.mass,
                )
            )
            assign(node.left)
            assign(node.right)

        assign(self._root)
        self.vectors = tuple(vectors)
        self.nvectors = len(self.vectors)
        self.ncoords = 1 if self.natoms == 2 else 3 * self.natoms - 6
        labels = ["r0"]
        for vector in range(1, self.nvectors):
            labels.extend((f"r{vector}", f"theta{vector}"))
            if vector >= 2:
                labels.append(f"phi{vector}")
        self.coordinate_labels = tuple(labels)

    @property
    def reduced_masses(self):
        return np.asarray([vector.reduced_mass for vector in self.vectors])

    def _coordinate_offsets(self):
        offsets = [(0, None, None)]
        cursor = 1
        for vector in range(1, self.nvectors):
            radial, theta = cursor, cursor + 1
            cursor += 2
            phi = None
            if vector >= 2:
                phi = cursor
                cursor += 1
            offsets.append((radial, theta, phi))
        return offsets

    def jacobi_vectors(self, coordinates, *, module=np):
        """Return body-fixed Cartesian Jacobi vectors."""
        q = module.asarray(coordinates)
        if q.ndim != 1 or q.shape[0] != self.ncoords:
            raise ValueError(f"coordinates must have shape ({self.ncoords},)")
        vectors = []
        for index, (radial, theta, phi) in enumerate(
            self._coordinate_offsets()
        ):
            radius = q[radial]
            if index == 0:
                vector = module.stack((0.0 * radius, 0.0 * radius, radius))
            elif index == 1:
                angle = q[theta]
                vector = module.stack(
                    (radius * module.sin(angle), 0.0 * radius,
                     radius * module.cos(angle))
                )
            else:
                polar = q[theta]
                azimuth = q[phi]
                vector = module.stack(
                    (
                        radius * module.sin(polar) * module.cos(azimuth),
                        radius * module.sin(polar) * module.sin(azimuth),
                        radius * module.cos(polar),
                    )
                )
            vectors.append(vector)
        return module.stack(vectors)

    def cartesian(self, coordinates, *, module=np):
        """Map polyspherical coordinates to center-of-mass Cartesian geometry."""
        vectors = self.jacobi_vectors(coordinates, module=module)
        origin = module.zeros(3, dtype=vectors.dtype)

        def place(node, center):
            if node.is_leaf:
                return {node.atom: center}
            vector = vectors[node.vector_index]
            total = node.mass
            left_center = center - (node.right.mass / total) * vector
            right_center = center + (node.left.mass / total) * vector
            positions = place(node.left, left_center)
            positions.update(place(node.right, right_center))
            return positions

        positions = place(self._root, origin)
        return module.stack([positions[atom] for atom in range(self.natoms)])

    def numpy_map(self):
        """Return a NumPy coordinate-to-Cartesian callable."""
        return lambda coordinates: self.cartesian(coordinates, module=np)

    def jax_map(self):
        """Return a JAX-differentiable coordinate-to-Cartesian callable."""
        from jax import numpy as jnp

        return lambda coordinates: self.cartesian(coordinates, module=jnp)

    def vectors_from_cartesian(self, geometry):
        """Evaluate the tree's Jacobi vectors for a Cartesian geometry."""
        geometry = np.asarray(geometry, dtype=float)
        if geometry.shape != (self.natoms, 3):
            raise ValueError(
                f"geometry must have shape {(self.natoms, 3)}"
            )
        result = []
        for vector in self.vectors:
            left_mass = self.masses[list(vector.left_atoms)]
            right_mass = self.masses[list(vector.right_atoms)]
            left_center = np.average(
                geometry[list(vector.left_atoms)], axis=0, weights=left_mass
            )
            right_center = np.average(
                geometry[list(vector.right_atoms)], axis=0, weights=right_mass
            )
            result.append(right_center - left_center)
        return np.asarray(result)


def _tt_svd(field, *, max_rank=None, rtol=1.0e-12, atol=0.0):
    field = np.asarray(field)
    if field.ndim < 1 or any(size < 1 for size in field.shape):
        raise ValueError("field must be a nonempty tensor")
    if max_rank is not None and int(max_rank) < 1:
        raise ValueError("max_rank must be positive")
    if rtol < 0.0 or atol < 0.0:
        raise ValueError("rtol and atol must be nonnegative")
    if field.ndim == 1:
        return [field.reshape(1, field.shape[0], 1)]

    tolerance = max(float(atol), float(rtol) * np.linalg.norm(field))
    tolerance /= np.sqrt(field.ndim - 1)
    cores = []
    unfolding = field
    left_rank = 1
    for mode in field.shape[:-1]:
        matrix = unfolding.reshape(left_rank * mode, -1)
        left, singular, right = np.linalg.svd(matrix, full_matrices=False)
        rank = len(singular)
        if tolerance > 0.0 and singular.size:
            tail = np.sqrt(np.cumsum(singular[::-1] ** 2))[::-1]
            removable = np.flatnonzero(tail <= tolerance)
            if removable.size:
                rank = max(1, int(removable[0]))
        if max_rank is not None:
            rank = min(rank, int(max_rank))
        cores.append(left[:, :rank].reshape(left_rank, mode, rank))
        unfolding = singular[:rank, None] * right[:rank]
        left_rank = rank
    cores.append(unfolding.reshape(left_rank, field.shape[-1], 1))
    return cores


def diagonal_field_mpo(field, *, max_rank=None, rtol=1.0e-12, atol=0.0):
    """Compress a product-grid scalar field into a diagonal MPO."""
    from pyqed.mps.mps import MPO

    cores = _tt_svd(field, max_rank=max_rank, rtol=rtol, atol=atol)
    factors = []
    for core in cores:
        left, physical, right = core.shape
        factor = np.zeros(
            (left, right, physical, physical), dtype=core.dtype
        )
        index = np.arange(physical)
        factor[:, :, index, index] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def diagonal_tt_mpo(cores):
    """Convert scalar TT cores directly into a diagonal MPO."""
    from pyqed.mps.mps import MPO

    factors = []
    for core in cores:
        core = np.asarray(core)
        if core.ndim != 3:
            raise ValueError("TT cores must have (left, physical, right) order")
        left, physical, right = core.shape
        factor = np.zeros(
            (left, right, physical, physical), dtype=core.dtype
        )
        index = np.arange(physical)
        factor[:, :, index, index] = core.transpose(0, 2, 1)
        factors.append(factor)
    return MPO(factors)


def _local_operator_mpo(dims, site, operator):
    from pyqed.mps.mps import MPO

    factors = []
    for axis, dim in enumerate(dims):
        local = operator if axis == site else np.eye(dim)
        local = np.asarray(local)
        if local.shape != (dim, dim):
            raise ValueError(
                f"operator at site {site} must have shape {(dim, dim)}"
            )
        factors.append(local.reshape(1, 1, dim, dim))
    return MPO(factors)


def _axis_independent(field, axis, *, rtol=1.0e-12, atol=1.0e-14):
    reference = np.expand_dims(np.take(field, 0, axis=axis), axis)
    return np.allclose(field, reference, rtol=rtol, atol=atol)


def _sine_unit_kinetic(dvr):
    from pyqed.dvr.dvr_1d import SineDVR

    if not isinstance(dvr, SineDVR):
        return None
    return np.asarray(dvr.t()) * float(dvr.mass)


def _flat_unit_kinetic(dvr):
    if not hasattr(dvr, "t") or not hasattr(dvr, "mass"):
        raise TypeError("analytical polyspherical KEO requires DVR t() and mass")
    kinetic = np.asarray(dvr.t()) * float(dvr.mass)
    expected = (int(dvr.npts), int(dvr.npts))
    if kinetic.shape != expected:
        raise ValueError(f"DVR kinetic shape {kinetic.shape} != {expected}")
    return kinetic


def _sep_basis(ndim):
    return tuple((0, 0) for _ in range(ndim))


def _sep_monomial(ndim, axis=None, powers=(0, 0), coefficient=1.0):
    key = list(_sep_basis(ndim))
    if axis is not None:
        key[axis] = tuple(powers)
    return {tuple(key): float(coefficient)}


def _sep_add(*fields):
    result = {}
    for field in fields:
        for key, coefficient in field.items():
            result[key] = result.get(key, 0.0) + coefficient
    if result:
        scale = max(abs(value) for value in result.values())
        cutoff = 32.0 * np.finfo(float).eps * scale
        result = {
            key: value for key, value in result.items()
            if abs(value) > cutoff
        }
    return result


def _sep_scale(field, coefficient):
    return _sep_add({key: coefficient * value for key, value in field.items()})


def _sep_mul(first, second):
    result = {}
    for left, left_coefficient in first.items():
        for right, right_coefficient in second.items():
            key = tuple(
                (a + c, b + d)
                for (a, b), (c, d) in zip(left, right)
            )
            result[key] = result.get(key, 0.0) + (
                left_coefficient * right_coefficient
            )
    return _sep_add(result)


def _sep_derivative(field, axis, kind):
    result = {}
    for key, coefficient in field.items():
        sine_power, cosine_power = key[axis]
        if kind == "radial":
            if sine_power:
                derived = list(key)
                derived[axis] = (sine_power - 1, 0)
                result[tuple(derived)] = result.get(tuple(derived), 0.0) + (
                    coefficient * sine_power
                )
            continue
        if sine_power:
            derived = list(key)
            derived[axis] = (sine_power - 1, cosine_power + 1)
            result[tuple(derived)] = result.get(tuple(derived), 0.0) + (
                coefficient * sine_power
            )
        if cosine_power:
            derived = list(key)
            derived[axis] = (sine_power + 1, cosine_power - 1)
            result[tuple(derived)] = result.get(tuple(derived), 0.0) - (
                coefficient * cosine_power
            )
    return _sep_add(result)


def _sep_trig_reduce(field, kinds):
    """Reduce angular Laurent monomials modulo sin(q)^2 + cos(q)^2 = 1."""
    result = field
    for axis, kind in enumerate(kinds):
        if kind == "radial":
            continue
        reduced = {}
        for key, coefficient in result.items():
            sine_power, cosine_power = key[axis]
            if cosine_power < 2:
                reduced[key] = reduced.get(key, 0.0) + coefficient
                continue
            quotient, remainder = divmod(cosine_power, 2)
            for power in range(quotient + 1):
                transformed = list(key)
                transformed[axis] = (
                    sine_power + 2 * power, remainder
                )
                transformed = tuple(transformed)
                reduced[transformed] = reduced.get(transformed, 0.0) + (
                    coefficient * comb(quotient, power) * (-1.0) ** power
                )
        result = _sep_add(reduced)
    return result


def _polyspherical_coordinate_kinds(tree):
    kinds = ["radial"]
    for vector in range(1, tree.nvectors):
        kinds.extend(("radial", "angle"))
        if vector >= 2:
            kinds.append("angle")
    return tuple(kinds)


def _analytic_metric_fields(tree):
    """Return exact separable fields for the vibrational metric and Vps."""
    ndim = tree.ncoords
    nvectors = tree.nvectors
    offsets = tree._coordinate_offsets()
    zero = {}
    one = _sep_monomial(ndim)

    def local(axis, sine=0, cosine=0, coefficient=1.0):
        return _sep_monomial(
            ndim, axis, (sine, cosine), coefficient
        )

    def add(*values):
        return _sep_add(*values)

    def mul(*values):
        result = one
        for value in values:
            result = _sep_mul(result, value)
        return result

    def scale(value, coefficient):
        return _sep_scale(value, coefficient)

    unit_vectors = []
    polar_vectors = [None] * nvectors
    azimuth_vectors = [None] * nvectors
    unit_vectors.append((zero, zero, one))
    for vector in range(1, nvectors):
        _radial, theta, phi = offsets[vector]
        sine_theta = local(theta, sine=1)
        cosine_theta = local(theta, cosine=1)
        if phi is None:
            sine_phi, cosine_phi = zero, one
        else:
            sine_phi = local(phi, sine=1)
            cosine_phi = local(phi, cosine=1)
        unit_vectors.append((
            mul(sine_theta, cosine_phi),
            mul(sine_theta, sine_phi),
            cosine_theta,
        ))
        polar_vectors[vector] = (
            mul(cosine_theta, cosine_phi),
            mul(cosine_theta, sine_phi),
            scale(sine_theta, -1.0),
        )
        azimuth_vectors[vector] = (
            scale(sine_phi, -1.0), cosine_phi, zero
        )

    gradients = [
        [[zero for _component in range(3)] for _vector in range(nvectors)]
        for _coordinate in range(ndim)
    ]
    for vector, (radial, _theta, _phi) in enumerate(offsets):
        gradients[radial][vector] = list(unit_vectors[vector])

    if nvectors >= 2:
        radial_0 = offsets[0][0]
        radial_1, theta_1, _phi_1 = offsets[1]
        inverse_r0 = local(radial_0, sine=-1)
        inverse_r1 = local(radial_1, sine=-1)
        sine_theta1 = local(theta_1, sine=1)
        cosine_theta1 = local(theta_1, cosine=1)
        gradients[theta_1][0] = [scale(inverse_r0, -1.0), zero, zero]
        gradients[theta_1][1] = [
            mul(cosine_theta1, inverse_r1),
            zero,
            scale(mul(sine_theta1, inverse_r1), -1.0),
        ]

        for vector in range(2, nvectors):
            radial, theta, phi = offsets[vector]
            inverse_radius = local(radial, sine=-1)
            inverse_sine = local(theta, sine=-1)
            sine_phi = local(phi, sine=1)
            cosine_phi = local(phi, cosine=1)
            cotangent = local(theta, sine=-1, cosine=1)
            cotangent_1 = local(theta_1, sine=-1, cosine=1)

            gradients[theta][0] = [
                scale(mul(cosine_phi, inverse_r0), -1.0),
                scale(mul(sine_phi, inverse_r0), -1.0),
                zero,
            ]
            gradients[theta][vector] = [
                mul(component, inverse_radius)
                for component in polar_vectors[vector]
            ]

            gradients[phi][0] = [
                mul(cotangent, sine_phi, inverse_r0),
                add(
                    scale(mul(cotangent, cosine_phi, inverse_r0), -1.0),
                    mul(cotangent_1, inverse_r0),
                ),
                zero,
            ]
            gradients[phi][1] = [
                zero,
                scale(
                    mul(inverse_r1, local(theta_1, sine=-1)), -1.0
                ),
                zero,
            ]
            gradients[phi][vector] = [
                mul(component, inverse_radius, inverse_sine)
                for component in azimuth_vectors[vector]
            ]

    inverse_masses = 1.0 / tree.reduced_masses
    kinds = _polyspherical_coordinate_kinds(tree)
    metric = [[zero for _second in range(ndim)] for _first in range(ndim)]
    for first in range(ndim):
        for second in range(first, ndim):
            field = zero
            for vector in range(nvectors):
                for component in range(3):
                    field = add(
                        field,
                        scale(
                            mul(
                                gradients[first][vector][component],
                                gradients[second][vector][component],
                            ),
                            inverse_masses[vector],
                        ),
                    )
            field = _sep_trig_reduce(field, kinds)
            metric[first][second] = field
            metric[second][first] = field

    log_derivative = []
    log_hessian = []
    for axis, kind in enumerate(kinds):
        if kind == "radial":
            log_derivative.append(local(axis, sine=-1, coefficient=4.0))
            log_hessian.append(local(axis, sine=-2, coefficient=-4.0))
        elif axis in {offset[1] for offset in offsets if offset[1] is not None}:
            log_derivative.append(
                local(axis, sine=-1, cosine=1, coefficient=2.0)
            )
            log_hessian.append(local(axis, sine=-2, coefficient=-2.0))
        else:
            log_derivative.append(zero)
            log_hessian.append(zero)

    pseudo_quadratic = zero
    pseudo_divergence = zero
    pseudo_hessian = zero
    for first in range(ndim):
        for second in range(ndim):
            pseudo_quadratic = add(
                pseudo_quadratic,
                mul(
                    log_derivative[first],
                    metric[first][second],
                    log_derivative[second],
                ),
            )
            pseudo_divergence = add(
                pseudo_divergence,
                mul(
                    _sep_trig_reduce(
                        _sep_derivative(
                            metric[first][second], first, kinds[first]
                        ),
                        kinds,
                    ),
                    log_derivative[second],
                ),
            )
        pseudo_hessian = add(
            pseudo_hessian,
            mul(metric[first][first], log_hessian[first]),
        )
    pseudopotential = _sep_trig_reduce(scale(
        add(
            pseudo_quadratic,
            scale(add(pseudo_divergence, pseudo_hessian), 4.0),
        ),
        1.0 / 32.0,
    ), kinds)
    return metric, pseudopotential


def _evaluate_separable_field(field, dvrs, kinds):
    shape = tuple(int(dvr.npts) for dvr in dvrs)
    result = np.zeros(shape, dtype=float)
    for powers, coefficient in field.items():
        term = float(coefficient)
        for axis, ((sine_power, cosine_power), dvr, kind) in enumerate(
            zip(powers, dvrs, kinds)
        ):
            if not sine_power and not cosine_power:
                continue
            values = np.asarray(dvr.x, dtype=float)
            if kind == "radial":
                local_values = values ** sine_power
            else:
                local_values = (
                    np.sin(values) ** sine_power
                    * np.cos(values) ** cosine_power
                )
            reshape = [1] * len(shape)
            reshape[axis] = shape[axis]
            term = term * local_values.reshape(reshape)
        result += term
    return result


def sample_analytic_metric(tree, dvrs):
    """Evaluate the exact Jacobi-tree metric and Podolsky potential on a grid."""
    if not isinstance(tree, PolysphericalTree):
        raise TypeError("tree must be a PolysphericalTree")
    dvrs = tuple(dvrs)
    if len(dvrs) != tree.ncoords:
        raise ValueError(f"expected {tree.ncoords} DVRs, got {len(dvrs)}")
    fields, pseudopotential = _analytic_metric_fields(tree)
    kinds = _polyspherical_coordinate_kinds(tree)
    shape = tuple(int(dvr.npts) for dvr in dvrs)
    metric = np.empty((*shape, tree.ncoords, tree.ncoords))
    for first in range(tree.ncoords):
        for second in range(tree.ncoords):
            metric[..., first, second] = _evaluate_separable_field(
                fields[first][second], dvrs, kinds
            )
    return metric, _evaluate_separable_field(
        pseudopotential, dvrs, kinds
    )


def _separable_local_factors(powers, dvrs, kinds):
    factors = []
    for (sine_power, cosine_power), dvr, kind in zip(powers, dvrs, kinds):
        if not sine_power and not cosine_power:
            factors.append(None)
            continue
        values = np.asarray(dvr.x, dtype=float)
        if kind == "radial":
            values = values ** sine_power
        else:
            values = (
                np.sin(values) ** sine_power
                * np.cos(values) ** cosine_power
            )
        factors.append(np.diag(values))
    return factors


def _general_analytic_keo_terms(tree, dvrs):
    metric, pseudopotential = _analytic_metric_fields(tree)
    kinds = _polyspherical_coordinate_kinds(tree)
    dvrs = tuple(dvrs)
    momenta = [np.asarray(dvr.momentum()) for dvr in dvrs]
    terms = []
    for first in range(tree.ncoords):
        for second in range(first, tree.ncoords):
            axis_independent = (
                first == second
                and not _sep_trig_reduce(
                    _sep_derivative(
                        metric[first][second], first, kinds[first]
                    ),
                    kinds,
                )
            )
            for powers, coefficient in metric[first][second].items():
                factors = _separable_local_factors(powers, dvrs, kinds)
                if first == second:
                    unit_kinetic = _sine_unit_kinetic(dvrs[first])
                    if axis_independent and unit_kinetic is not None:
                        diagonal = (
                            np.eye(int(dvrs[first].npts))
                            if factors[first] is None
                            else factors[first]
                        )
                        factors[first] = diagonal @ unit_kinetic
                        terms.append((coefficient, tuple(factors)))
                    else:
                        diagonal = (
                            np.eye(int(dvrs[first].npts))
                            if factors[first] is None
                            else factors[first]
                        )
                        factors[first] = (
                            momenta[first].conj().T
                            @ diagonal
                            @ momenta[first]
                        )
                        terms.append((0.5 * coefficient, tuple(factors)))
                    continue
                left = list(factors)
                diagonal_first = (
                    np.eye(int(dvrs[first].npts))
                    if left[first] is None else left[first]
                )
                diagonal_second = (
                    np.eye(int(dvrs[second].npts))
                    if left[second] is None else left[second]
                )
                left[first] = momenta[first].conj().T @ diagonal_first
                left[second] = diagonal_second @ momenta[second]
                terms.append((0.5 * coefficient, tuple(left)))
                right = [
                    None if factor is None else factor.conj().T
                    for factor in left
                ]
                terms.append((0.5 * coefficient, tuple(right)))

    for powers, coefficient in pseudopotential.items():
        terms.append((
            coefficient,
            tuple(_separable_local_factors(powers, dvrs, kinds)),
        ))
    return terms


def analytic_keo_terms(tree, dvrs):
    """Return exact flat-measure SOP terms for an orthogonal Jacobi tree."""
    if not isinstance(tree, PolysphericalTree):
        raise TypeError("tree must be a PolysphericalTree")
    dvrs = tuple(dvrs)
    if len(dvrs) != tree.ncoords:
        raise ValueError(f"expected {tree.ncoords} DVRs, got {len(dvrs)}")
    if tree.natoms > 3:
        return _general_analytic_keo_terms(tree, dvrs)

    inverse_masses = 1.0 / tree.reduced_masses
    radial_0 = _flat_unit_kinetic(dvrs[0])
    if tree.natoms == 2:
        return [(inverse_masses[0], (radial_0,))]

    radial_1 = _flat_unit_kinetic(dvrs[1])
    angular = _flat_unit_kinetic(dvrs[2])
    inverse_r0_squared = np.diag(np.asarray(dvrs[0].x, dtype=float) ** -2)
    inverse_r1_squared = np.diag(np.asarray(dvrs[1].x, dtype=float) ** -2)
    theta = np.asarray(dvrs[2].x, dtype=float)
    if np.any(np.isclose(np.sin(theta), 0.0)):
        raise ValueError("angular DVR points cannot lie at theta=0 or pi")
    podolsky_angle = np.diag(1.0 + 1.0 / np.sin(theta) ** 2)

    return [
        (inverse_masses[0], (radial_0, None, None)),
        (inverse_masses[1], (None, radial_1, None)),
        (inverse_masses[0], (inverse_r0_squared, None, angular)),
        (inverse_masses[1], (None, inverse_r1_squared, angular)),
        (
            -0.125 * inverse_masses[0],
            (inverse_r0_squared, None, podolsky_angle),
        ),
        (
            -0.125 * inverse_masses[1],
            (None, inverse_r1_squared, podolsky_angle),
        ),
    ]


def build_analytic_keo_mpo(tree, dvrs, *, mpo_max_rank=None):
    """Build an arbitrary-size analytical Jacobi-tree KEO as an MPO."""
    from pyqed.mps.mpo import sop_to_mpo

    dvrs = tuple(dvrs)
    dims = tuple(int(dvr.npts) for dvr in dvrs)
    mpo = sop_to_mpo(dims, analytic_keo_terms(tree, dvrs))
    return (
        mpo
        if mpo_max_rank is None
        else mpo.compress_hermitian(int(mpo_max_rank))
    )


def metric_keo_mpo(
    dvrs,
    metric,
    pseudopotential=None,
    *,
    field_max_rank=None,
    field_rtol=1.0e-12,
    field_atol=0.0,
    mpo_max_rank=None,
    boundary_complete=False,
):
    r"""Build the Podolsky vibrational KEO as an MPO.

    Diagonal sine-DVR terms whose metric is independent of their own
    coordinate use the exact sine second derivative. Other terms use the
    general ``D_mu^dagger G^{mu nu} D_nu / 2`` weak form.
    """
    dvrs = tuple(dvrs)
    dims = tuple(int(dvr.npts) for dvr in dvrs)
    ndim = len(dims)
    metric = np.asarray(metric)
    expected = (*dims, ndim, ndim)
    if metric.shape != expected:
        raise ValueError(f"metric shape {metric.shape} != {expected}")
    derivative = [
        _local_operator_mpo(
            dims,
            axis,
            _boundary_complete_metric_derivative(dvr)
            if boundary_complete
            else dvr.momentum(),
        )
        for axis, dvr in enumerate(dvrs)
    ]
    derivative_adjoint = [operator.adjoint() for operator in derivative]

    result = None
    for first in range(ndim):
        for second in range(first, ndim):
            field = 0.5 * (
                metric[..., first, second]
                + metric[..., second, first].conj()
            )
            if not np.any(field):
                continue
            coefficient = diagonal_field_mpo(
                field,
                max_rank=field_max_rank,
                rtol=field_rtol,
                atol=field_atol,
            )
            unit_kinetic = _sine_unit_kinetic(dvrs[first])
            if (
                first == second
                and unit_kinetic is not None
                and _axis_independent(field, first)
            ):
                kinetic = _local_operator_mpo(
                    dims, first, unit_kinetic
                )
                term = coefficient.compose(kinetic)
            else:
                forward = derivative_adjoint[first].compose(
                    coefficient
                ).compose(derivative[second])
                term = (
                    0.5 * forward
                    if first == second
                    else 0.5 * (forward + forward.adjoint())
                )
            result = term if result is None else result + term

    if pseudopotential is not None:
        pseudopotential = np.asarray(pseudopotential)
        if pseudopotential.shape != dims:
            raise ValueError(
                f"pseudopotential shape {pseudopotential.shape} != {dims}"
            )
        term = diagonal_field_mpo(
            pseudopotential,
            max_rank=field_max_rank,
            rtol=field_rtol,
            atol=field_atol,
        )
        result = term if result is None else result + term
    if result is None:
        result = diagonal_field_mpo(np.zeros(dims))
    return (
        result
        if mpo_max_rank is None
        else result.compress_hermitian(int(mpo_max_rank))
    )


def _tt_constant_on_axis(cores, axis, *, trials=8, seed=0):
    from pyqed.mps.cross import tt_value

    shape = tuple(core.shape[1] for core in cores)
    rng = np.random.default_rng(seed)
    for _ in range(trials):
        index = [int(rng.integers(size)) for size in shape]
        values = []
        for position in range(shape[axis]):
            index[axis] = position
            values.append(tt_value(cores, index))
        if not np.allclose(values, values[0], rtol=1.0e-10, atol=1.0e-12):
            return False
    return True


def sample_metric_tt(
    dvrs,
    point_evaluator,
    *,
    batch_point_evaluator=None,
    max_rank=8,
    sweeps=4,
    rtol=1.0e-8,
    validation=64,
    seed=0,
    start_rank=1,
    kick_rank=2,
    initial=None,
    return_state=False,
    backend="native",
    device=None,
    verbose=False,
):
    """Fit metric fields and the pseudopotential using shared TT-cross samples.

    ``point_evaluator(q)`` returns ``(metric, pseudopotential)`` at one
    coordinate vector. Only the independent upper metric triangle is fitted.
    """
    from pyqed.mps.cross import tt_cross, tt_cross_tntorch

    dvrs = tuple(dvrs)
    shape = tuple(int(dvr.npts) for dvr in dvrs)
    ndim = len(shape)
    cache = {}

    def store(index, metric, pseudopotential):
        metric = np.asarray(metric)
        if metric.shape != (ndim, ndim):
            raise ValueError(
                f"point metric shape {metric.shape} != {(ndim, ndim)}"
            )
        cache[index] = (
            0.5 * (metric + metric.conj().T),
            np.asarray(pseudopotential).item(),
        )

    def fields(index):
        index = tuple(int(item) for item in index)
        if index not in cache:
            coordinates = np.asarray(
                [dvrs[axis].x[position] for axis, position in enumerate(index)]
            )
            metric, pseudopotential = point_evaluator(coordinates)
            store(index, metric, pseudopotential)
        return cache[index]

    def fields_many(indices):
        indices = [tuple(int(item) for item in index) for index in indices]
        missing = list(dict.fromkeys(index for index in indices if index not in cache))
        if missing and batch_point_evaluator is not None:
            coordinates = np.asarray(
                [
                    [dvrs[axis].x[position] for axis, position in enumerate(index)]
                    for index in missing
                ]
            )
            metrics, pseudopotentials = batch_point_evaluator(coordinates)
            metrics = np.asarray(metrics)
            pseudopotentials = np.asarray(pseudopotentials)
            if metrics.shape != (len(missing), ndim, ndim):
                raise ValueError("batched point metrics have an incompatible shape")
            if pseudopotentials.shape != (len(missing),):
                raise ValueError(
                    "batched pseudopotentials have an incompatible shape"
                )
            for index, metric, pseudopotential in zip(
                missing, metrics, pseudopotentials
            ):
                store(index, metric, pseudopotential)
        return [fields(index) for index in indices]

    labels = []
    for first in range(ndim):
        for second in range(first, ndim):
            labels.append((first, second))
    nfields = len(labels) + 1

    def packed_value(index):
        values = fields(index[:-1])
        field = index[-1]
        if field == len(labels):
            return values[1]
        first, second = labels[field]
        return values[0][first, second]

    def packed_values(indices):
        values = fields_many(indices[:, :-1])
        result = np.empty(len(indices), dtype=np.result_type(*[
            value[0].dtype for value in values
        ]))
        for row, (metric, pseudopotential) in enumerate(values):
            field = int(indices[row, -1])
            if field == len(labels):
                result[row] = pseudopotential
            else:
                first, second = labels[field]
                result[row] = metric[first, second]
        return result

    cross = {
        "native": tt_cross,
        "tntorch": tt_cross_tntorch,
    }.get(backend)
    if cross is None:
        raise ValueError("backend must be 'native' or 'tntorch'")
    cross_options = {
        "start_rank": start_rank,
        "kick_rank": kick_rank,
        "batch_evaluator": packed_values,
        "initial": initial,
        "return_state": return_state,
    }
    if backend == "tntorch":
        cross_options.update(device=device, verbose=verbose)
    packed_cores, cross_info = cross(
        (*shape, nfields),
        packed_value,
        max_rank=max_rank,
        sweeps=sweeps,
        rtol=rtol,
        validation=validation,
        seed=seed,
        **cross_options,
    )
    shared = packed_cores[:-2]
    coordinate_core = packed_cores[-2]
    output_core = packed_cores[-1][:, :, 0]

    def extract(field):
        last = np.einsum(
            "aib,b->ai", coordinate_core, output_core[:, field]
        )
        return [*shared, last[:, :, None]]

    metric_cores = {
        label: extract(field) for field, label in enumerate(labels)
    }
    pseudopotential_cores = extract(len(labels))
    return metric_cores, pseudopotential_cores, {
        "point_samples": len(cache),
        "grid_size": int(np.prod(shape)),
        "backend": backend,
        "cross": cross_info,
        "field_labels": tuple([f"G{i}{j}" for i, j in labels] + ["Vps"]),
    }


def _boundary_complete_metric_derivative(dvr):
    """Return a derivative factor carrying the sine-DVR boundary energy.

    The square sine-to-sine projected momentum is rank deficient on odd grids
    and therefore cannot factor the positive Dirichlet kinetic matrix.  The
    unitary Procrustes completion below is the closest factor to that momentum
    whose Gram matrix is exactly ``2 * dvr.t()``.
    """
    momentum = np.asarray(dvr.momentum(), dtype=complex)
    kinetic = np.asarray(dvr.t(), dtype=complex)
    kinetic = 0.5 * (kinetic + kinetic.conj().T)
    eigenvalues, eigenvectors = np.linalg.eigh(kinetic)
    tolerance = 128.0 * np.finfo(float).eps * max(
        1.0, float(np.max(np.abs(eigenvalues)))
    )
    if np.min(eigenvalues) < -tolerance:
        raise ValueError("the DVR kinetic matrix must be positive semidefinite")
    root = (eigenvectors * np.sqrt(2.0 * np.maximum(eigenvalues, 0.0))) @ (
        eigenvectors.conj().T
    )
    left, _singular_values, right = np.linalg.svd(momentum @ root)
    derivative = (left @ right) @ root
    return np.asarray(derivative)


def metric_tt_keo_components(
    dvrs,
    metric_cores,
    pseudopotential_cores=None,
    *,
    boundary_complete=False,
):
    """Return active-axis-labelled MPO components of a TT Podolsky KEO."""
    dvrs = tuple(dvrs)
    dims = tuple(int(dvr.npts) for dvr in dvrs)
    ndim = len(dims)
    derivative = [
        _local_operator_mpo(
            dims,
            axis,
            _boundary_complete_metric_derivative(dvr)
            if boundary_complete
            else dvr.momentum(),
        )
        for axis, dvr in enumerate(dvrs)
    ]
    derivative_adjoint = [operator.adjoint() for operator in derivative]
    components = []

    def add(active, term):
        components.append((tuple(int(axis) for axis in active), term))

    for first in range(ndim):
        for second in range(first, ndim):
            cores = metric_cores[first, second]
            coefficient = diagonal_tt_mpo(cores)
            if first == second:
                unit_kinetic = _sine_unit_kinetic(dvrs[first])
                if (
                    unit_kinetic is not None
                    and _tt_constant_on_axis(cores, first, seed=first)
                ):
                    kinetic = _local_operator_mpo(
                        dims, first, unit_kinetic
                    )
                    add((first,), coefficient.compose(kinetic))
                else:
                    add(
                        (first,),
                        0.5
                        * derivative_adjoint[first].compose(
                            coefficient
                        ).compose(derivative[first])
                    )
                continue
            forward = derivative_adjoint[first].compose(
                coefficient
            ).compose(derivative[second])
            add((first, second), 0.5 * (forward + forward.adjoint()))

    if pseudopotential_cores is not None:
        add((), diagonal_tt_mpo(pseudopotential_cores))
    if not components:
        add((), diagonal_field_mpo(np.zeros(dims)))
    return tuple(components)


def metric_tt_keo_mpo(
    dvrs,
    metric_cores,
    pseudopotential_cores=None,
    *,
    mpo_max_rank=None,
    boundary_complete=False,
):
    """Assemble a Hermitian Podolsky KEO directly from TT fields."""
    components = metric_tt_keo_components(
        dvrs,
        metric_cores,
        pseudopotential_cores,
        boundary_complete=boundary_complete,
    )
    result = components[0][1]
    for _active, component in components[1:]:
        result = result + component
    return (
        result
        if mpo_max_rank is None
        else result.compress_hermitian(int(mpo_max_rank))
    )


def sample_metric(dvrs, masses, coordinate_map):
    """Sample vibrational G-matrix and curvilinear pseudopotential on a DVR grid."""
    import jax
    from jax import numpy as jnp

    from pyqed.namd.keo import Gmat, pseudo

    dvrs = tuple(dvrs)
    dims = tuple(int(dvr.npts) for dvr in dvrs)
    ndim = len(dims)
    mesh = jnp.meshgrid(*(dvr.x for dvr in dvrs), indexing="ij")
    points = jnp.stack([axis.reshape(-1) for axis in mesh], axis=1)
    masses = jnp.asarray(masses)
    metric = jax.vmap(Gmat, in_axes=(0, None, None))(
        points, masses, coordinate_map
    )[:, :ndim, :ndim]
    pseudopotential = jax.vmap(pseudo, in_axes=(0, None, None))(
        points, masses, coordinate_map
    )
    return (
        np.asarray(metric).reshape(*dims, ndim, ndim),
        np.asarray(pseudopotential).reshape(dims),
    )


def build_keo_mpo(
    tree,
    dvrs,
    *,
    field_max_rank=None,
    field_rtol=1.0e-12,
    field_atol=0.0,
    mpo_max_rank=None,
    return_fields=False,
    method="ad",
):
    """Build the J=0 polyspherical KEO analytically or through AD fields."""
    if not isinstance(tree, PolysphericalTree):
        raise TypeError("tree must be a PolysphericalTree")
    if len(dvrs) != tree.ncoords:
        raise ValueError(
            f"expected {tree.ncoords} DVRs, got {len(dvrs)}"
        )
    if method == "analytic":
        if return_fields:
            raise ValueError("analytical KEO construction does not sample fields")
        return build_analytic_keo_mpo(
            tree, dvrs, mpo_max_rank=mpo_max_rank
        )
    if method != "ad":
        raise ValueError("method must be 'analytic' or 'ad'")
    if tree.natoms == 2:
        metric, pseudopotential = sample_analytic_metric(tree, dvrs)
        mpo = build_analytic_keo_mpo(
            tree, dvrs, mpo_max_rank=mpo_max_rank
        )
        return (mpo, metric, pseudopotential) if return_fields else mpo
    else:
        metric, pseudopotential = sample_metric(
            dvrs, tree.masses, tree.jax_map()
        )
    mpo = metric_keo_mpo(
        dvrs,
        metric,
        pseudopotential,
        field_max_rank=field_max_rank,
        field_rtol=field_rtol,
        field_atol=field_atol,
        mpo_max_rank=mpo_max_rank,
    )
    return (mpo, metric, pseudopotential) if return_fields else mpo


def build_keo(
    tree,
    dvrs,
    *,
    method="analytic",
    return_fields=False,
    **kwargs,
):
    """Build the polyspherical kinetic-energy operator as a dense matrix.

    This is a compact front-end wrapper around :func:`build_keo_mpo`.
    Additional keyword arguments are forwarded directly.
    """
    value = build_keo_mpo(
        tree,
        dvrs,
        method=method,
        return_fields=return_fields,
        **kwargs,
    )

    if isinstance(value, tuple):
        mpo, metric, pseudopotential = value
        return mpo.to_dense(), metric, pseudopotential
    return value.to_dense()


def build_keo_mpo_cross(
    tree,
    dvrs,
    *,
    cross_max_rank=8,
    cross_sweeps=4,
    cross_rtol=1.0e-8,
    cross_validation=64,
    mpo_max_rank=None,
    seed=0,
    cross_start_rank=1,
    cross_kick_rank=2,
    cross_initial=None,
    return_cross_state=False,
    backend="native",
    device=None,
    verbose=False,
    return_info=False,
):
    """Build a polyspherical KEO without allocating product-grid fields."""
    import jax
    from jax import numpy as jnp

    from pyqed.namd.keo import Gmat, pseudo

    if not isinstance(tree, PolysphericalTree):
        raise TypeError("tree must be a PolysphericalTree")
    dvrs = tuple(dvrs)
    if len(dvrs) != tree.ncoords:
        raise ValueError(f"expected {tree.ncoords} DVRs, got {len(dvrs)}")
    ndim = len(dvrs)
    masses = jnp.asarray(tree.masses)
    coordinate_map = tree.jax_map()

    @jax.jit
    def evaluate(coordinates):
        metric = Gmat(coordinates, masses, coordinate_map)
        return metric[:ndim, :ndim], pseudo(
            coordinates, masses, coordinate_map
        )

    evaluate_batch = jax.jit(jax.vmap(evaluate))

    def point_evaluator(coordinates):
        metric, pseudopotential = evaluate(jnp.asarray(coordinates))
        return np.asarray(metric), np.asarray(pseudopotential)

    def batch_point_evaluator(coordinates):
        metric, pseudopotential = evaluate_batch(jnp.asarray(coordinates))
        return np.asarray(metric), np.asarray(pseudopotential)

    metric_cores, pseudopotential_cores, info = sample_metric_tt(
        dvrs,
        point_evaluator,
        batch_point_evaluator=batch_point_evaluator,
        max_rank=cross_max_rank,
        sweeps=cross_sweeps,
        rtol=cross_rtol,
        validation=cross_validation,
        seed=seed,
        start_rank=cross_start_rank,
        kick_rank=cross_kick_rank,
        initial=cross_initial,
        return_state=return_cross_state,
        backend=backend,
        device=device,
        verbose=verbose,
    )
    mpo = metric_tt_keo_mpo(
        dvrs,
        metric_cores,
        pseudopotential_cores,
        mpo_max_rank=mpo_max_rank,
    )
    return (mpo, info) if return_info else mpo


__all__ = [
    "JacobiVector",
    "PolysphericalTree",
    "analytic_keo_terms",
    "build_analytic_keo_mpo",
    "build_keo",
    "build_keo_mpo",
    "build_keo_mpo_cross",
    "diagonal_field_mpo",
    "diagonal_tt_mpo",
    "metric_keo_mpo",
    "metric_tt_keo_components",
    "metric_tt_keo_mpo",
    "sample_analytic_metric",
    "sample_metric",
    "sample_metric_tt",
]
