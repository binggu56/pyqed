"""Molecular-symmetry reduction of electronic-structure sampling."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@dataclass(frozen=True)
class SamplingSymmetryImage:
    """One requested sample expressed through a symmetry representative."""

    representative_coordinates: tuple[float, ...]
    operation: str = "identity"


class SamplingSymmetry:
    """Interface for reducing and transporting molecular-symmetry samples."""

    name = "identity"
    supports_record_transport = True

    def resolve(self, coordinates):
        coordinates = tuple(float(value) for value in coordinates)
        return SamplingSymmetryImage(coordinates)

    def images(self, coordinates):
        """Return the complete symmetry orbit of one explicit sample."""

        return (tuple(float(value) for value in coordinates),)

    def pair_images(self, left, right):
        """Apply each symmetry operation jointly to an explicit sample pair."""

        return ((
            tuple(float(value) for value in left),
            tuple(float(value) for value in right),
        ),)

    def transform_record(
        self,
        record,
        image,
        *,
        representative_geometry,
        requested_geometry,
        protocol,
    ):
        return record

    def view_key(self, image):
        return {"symmetry": self.name, "operation": str(image.operation)}

    def metadata(self):
        return {"name": self.name}


class FiniteGroupSamplingSymmetry(SamplingSymmetry):
    r"""A finite linear action used to sample one point per coordinate orbit.

    The coordinate matrices act as $q' = C_g q$.  This class deliberately does
    not transform electronic-structure records: it reduces scattered fit and
    validation designs to the quotient domain, while the matching electronic
    and feature representations are supplied to the equivariant fit.
    """

    supports_record_transport = False

    def __init__(
        self,
        coordinate_representations,
        *,
        name="finite-group",
        operations=None,
        origin=None,
        tolerance=1.0e-10,
    ):
        matrices = np.asarray(coordinate_representations, dtype=float)
        if (
            matrices.ndim != 3
            or matrices.shape[1] != matrices.shape[2]
            or len(matrices) < 1
        ):
            raise ValueError(
                "coordinate_representations must have shape (order, ndim, ndim)"
            )
        self.coordinate_representations = matrices
        self.ndim = int(matrices.shape[1])
        self.origin = np.zeros(self.ndim) if origin is None else np.asarray(
            origin, dtype=float
        )
        if self.origin.shape != (self.ndim,):
            raise ValueError(f"origin must have shape ({self.ndim},)")
        self.name = str(name)
        self.tolerance = float(tolerance)
        if self.tolerance <= 0.0:
            raise ValueError("tolerance must be positive")
        identity = np.eye(self.ndim)
        identity_matches = np.flatnonzero(
            np.linalg.norm(matrices - identity, axis=(1, 2)) <= self.tolerance
        )
        if identity_matches.size != 1:
            raise ValueError("the coordinate group must contain one identity")
        self.identity = int(identity_matches[0])
        if self.identity != 0:
            raise ValueError("the first coordinate group matrix must be the identity")
        adjoint = matrices.swapaxes(-1, -2)
        if not np.allclose(adjoint @ matrices, identity, atol=self.tolerance):
            raise ValueError("coordinate group matrices must be orthogonal")
        if operations is None:
            labels = [f"g{index}" for index in range(len(matrices))]
            labels[self.identity] = "identity"
            operations = labels
        self.operations = tuple(str(value) for value in operations)
        if len(self.operations) != len(matrices) or len(set(self.operations)) != len(
            matrices
        ):
            raise ValueError("operations must uniquely label every group matrix")
        self._operation_index = {
            operation: index for index, operation in enumerate(self.operations)
        }
        inverses = []
        for matrix in matrices:
            errors = np.linalg.norm(
                matrices @ matrix - identity,
                axis=(1, 2),
            )
            inverse = int(np.argmin(errors))
            if errors[inverse] > 10.0 * self.tolerance:
                raise ValueError("coordinate representations are not closed under inverse")
            inverses.append(inverse)
        self._inverses = tuple(inverses)

    @property
    def order(self):
        return len(self.coordinate_representations)

    def representative_count(self, full_domain_count):
        """Convert a full-domain sample budget to a quotient-domain budget."""

        full_domain_count = int(full_domain_count)
        if full_domain_count < 1:
            raise ValueError("full_domain_count must be positive")
        return int(np.ceil(full_domain_count / self.order))

    def _coordinate(self, coordinates):
        value = np.asarray(coordinates, dtype=float)
        if value.shape != (self.ndim,):
            raise ValueError(f"coordinates must have shape ({self.ndim},)")
        return value

    def _canonical_index(self, values):
        scale = max(self.tolerance, np.finfo(float).eps)
        rounded = np.rint(np.asarray(values) / scale).astype(np.int64)
        return max(range(len(rounded)), key=lambda index: tuple(rounded[index]))

    def resolve(self, coordinates):
        requested = self._coordinate(coordinates)
        centered = requested - self.origin
        orbit = np.einsum(
            "gij,j->gi", self.coordinate_representations, centered, optimize=True
        ) + self.origin
        to_representative = self._canonical_index(orbit)
        representative = orbit[to_representative]
        representative[np.abs(representative) <= self.tolerance] = 0.0
        from_representative = self._inverses[to_representative]
        return SamplingSymmetryImage(
            tuple(float(value) for value in representative),
            self.operations[from_representative],
        )

    def images(self, coordinates):
        coordinates = self._coordinate(coordinates)
        centered = coordinates - self.origin
        orbit = np.einsum(
            "gij,j->gi", self.coordinate_representations, centered, optimize=True
        ) + self.origin
        return self._unique_rows(orbit)

    def pair_images(self, left, right):
        left = self._coordinate(left)
        right = self._coordinate(right)
        left_orbit = np.einsum(
            "gij,j->gi",
            self.coordinate_representations,
            left - self.origin,
            optimize=True,
        ) + self.origin
        right_orbit = np.einsum(
            "gij,j->gi",
            self.coordinate_representations,
            right - self.origin,
            optimize=True,
        ) + self.origin
        images = []
        seen = set()
        for left_image, right_image in zip(left_orbit, right_orbit):
            key = self._row_key(np.concatenate((left_image, right_image)))
            if key not in seen:
                seen.add(key)
                images.append(
                    (
                        tuple(float(value) for value in left_image),
                        tuple(float(value) for value in right_image),
                    )
                )
        return tuple(images)

    def canonicalize_many(self, coordinates, *, unique=False):
        """Canonicalize scattered points and optionally remove orbit duplicates."""

        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.ndim:
            raise ValueError(f"coordinates must have shape (n, {self.ndim})")
        images = [self.resolve(row) for row in coordinates]
        representatives = np.asarray(
            [image.representative_coordinates for image in images], dtype=float
        )
        operations = tuple(image.operation for image in images)
        if not unique:
            return representatives, operations
        unique_rows, inverse = self._unique_rows(representatives, return_inverse=True)
        return np.asarray(unique_rows), inverse, operations

    def canonicalize_pairs(self, coordinates, pairs):
        """Reduce links by applying one common group operation to each pair."""

        coordinates = np.asarray(coordinates, dtype=float)
        pairs = np.asarray(pairs, dtype=int)
        if coordinates.ndim != 2 or coordinates.shape[1] != self.ndim:
            raise ValueError(f"coordinates must have shape (n, {self.ndim})")
        if pairs.ndim != 2 or pairs.shape[1] != 2:
            raise ValueError("pairs must have shape (npairs, 2)")
        if len(pairs) and (np.min(pairs) < 0 or np.max(pairs) >= len(coordinates)):
            raise IndexError("a pair is outside the coordinate set")
        reduced_coordinates = []
        reduced_pairs = []
        operations = []
        point_indices = {}
        for left_index, right_index in pairs:
            left = coordinates[left_index]
            right = coordinates[right_index]
            midpoint = 0.5 * (left + right)
            midpoint_orbit = np.einsum(
                "gij,j->gi",
                self.coordinate_representations,
                midpoint - self.origin,
                optimize=True,
            ) + self.origin
            to_representative = self._canonical_index(midpoint_orbit)
            matrix = self.coordinate_representations[to_representative]
            endpoints = (
                self.origin + matrix @ (left - self.origin),
                self.origin + matrix @ (right - self.origin),
            )
            pair_indices = []
            for endpoint in endpoints:
                endpoint[np.abs(endpoint) <= self.tolerance] = 0.0
                key = self._row_key(endpoint)
                if key not in point_indices:
                    point_indices[key] = len(reduced_coordinates)
                    reduced_coordinates.append(np.array(endpoint, copy=True))
                pair_indices.append(point_indices[key])
            reduced_pairs.append(pair_indices)
            operations.append(self.operations[self._inverses[to_representative]])
        return (
            np.asarray(reduced_coordinates, dtype=float).reshape(-1, self.ndim),
            np.asarray(reduced_pairs, dtype=int).reshape(-1, 2),
            tuple(operations),
        )

    def mace_group(
        self,
        electronic_representations,
        *,
        feature_rank=None,
        ambient_representations=None,
        tolerance=None,
    ):
        """Return the exact finite-group specification consumed by MACE."""

        electronic = np.asarray(electronic_representations)
        if electronic.shape[0] != len(self.coordinate_representations):
            raise ValueError("electronic and coordinate group orders differ")
        if ambient_representations is None:
            if feature_rank is None or int(feature_rank) % electronic.shape[1] != 0:
                raise ValueError(
                    "feature_rank must be a multiple of the electronic dimension"
                )
            copies = int(feature_rank) // electronic.shape[1]
            ambient_representations = np.asarray(
                [np.kron(np.eye(copies), value) for value in electronic]
            )
        return {
            "coordinate_representations": np.array(
                self.coordinate_representations, copy=True
            ),
            "electronic_representations": electronic,
            "ambient_representations": np.asarray(ambient_representations),
            "origin": np.array(self.origin, copy=True),
            "tolerance": self.tolerance if tolerance is None else float(tolerance),
        }

    def transform_hamiltonian(self, value, operation, electronic_representations):
        representation = self._electronic(operation, electronic_representations)
        return representation @ value @ representation.conj().T

    def transform_link(self, value, operation, electronic_representations):
        """Covariantly transport a raw link without polar unitarization."""

        representation = self._electronic(operation, electronic_representations)
        return representation @ value @ representation.conj().T

    def transform_feature(
        self,
        value,
        operation,
        electronic_representations,
        ambient_representations,
    ):
        index = self._operation_index[str(operation)]
        electronic = np.asarray(electronic_representations)[index]
        ambient = np.asarray(ambient_representations)[index]
        return ambient @ value @ electronic.conj().T

    def _electronic(self, operation, representations):
        representations = np.asarray(representations)
        if representations.shape[0] != len(self.coordinate_representations):
            raise ValueError("electronic and coordinate group orders differ")
        return representations[self._operation_index[str(operation)]]

    def _row_key(self, row):
        return tuple(np.rint(np.asarray(row) / self.tolerance).astype(np.int64))

    def _unique_rows(self, rows, *, return_inverse=False):
        unique = []
        indices = {}
        inverse = []
        for row in np.asarray(rows, dtype=float):
            key = self._row_key(row)
            if key not in indices:
                indices[key] = len(unique)
                clean = np.array(row, copy=True)
                clean[np.abs(clean) <= self.tolerance] = 0.0
                unique.append(tuple(float(value) for value in clean))
            inverse.append(indices[key])
        if return_inverse:
            return tuple(unique), np.asarray(inverse, dtype=int)
        return tuple(unique)

    def metadata(self):
        return {
            "name": self.name,
            "order": len(self.coordinate_representations),
            "operations": list(self.operations),
            "coordinate_representations": self.coordinate_representations.tolist(),
            "origin": self.origin.tolist(),
            "tolerance": self.tolerance,
            "canonical_domain": "lexicographically maximal finite-group image",
            "record_transport": False,
        }


def _pair_distances(geometry):
    geometry = np.asarray(geometry, dtype=float)
    pairs = tuple(
        (left, right)
        for left in range(len(geometry))
        for right in range(left + 1, len(geometry))
    )
    values = np.asarray(
        [np.linalg.norm(geometry[left] - geometry[right]) for left, right in pairs]
    )
    return pairs, values


def _atom_permutations(charges, distances, *, tolerance, limit=256):
    """Find distance-preserving permutations without factorial enumeration."""

    charges = np.asarray(charges, dtype=int)
    natom = len(charges)
    scale = max(float(np.max(distances)), 1.0)
    atol = max(float(tolerance), 1.0e-8 * scale)
    fingerprints = []
    for atom in range(natom):
        fingerprints.append(
            (
                int(charges[atom]),
                tuple(
                    sorted(
                        (int(charges[other]), int(np.rint(distances[atom, other] / atol)))
                        for other in range(natom)
                        if other != atom
                    )
                ),
            )
        )
    candidates = [
        tuple(
            target
            for target in range(natom)
            if fingerprints[target] == fingerprints[source]
        )
        for source in range(natom)
    ]
    order = sorted(range(natom), key=lambda atom: (len(candidates[atom]), atom))
    mapping = np.full(natom, -1, dtype=int)
    used = np.zeros(natom, dtype=bool)
    permutations = []

    def extend(depth):
        if len(permutations) >= limit:
            return
        if depth == natom:
            permutations.append(tuple(int(value) for value in mapping))
            return
        source = order[depth]
        for target in candidates[source]:
            if used[target]:
                continue
            valid = True
            for previous in order[:depth]:
                mapped = mapping[previous]
                if abs(distances[source, previous] - distances[target, mapped]) > atol:
                    valid = False
                    break
            if not valid:
                continue
            mapping[source] = target
            used[target] = True
            extend(depth + 1)
            used[target] = False
            mapping[source] = -1

    extend(0)
    identity = tuple(range(natom))
    permutations = sorted(set(permutations))
    if identity in permutations:
        permutations.remove(identity)
    return (identity, *permutations)


def detect_symmetry(molecule, coord, *, tolerance=2.0e-6):
    """Infer a finite coordinate action from molecular atom permutations.

    Detection uses only nuclear charges, the reference geometry, and the
    Cartesian embedding supplied by ``Coord``.  It is deliberately independent
    of the electronic-structure method; selected-state representations are
    inferred later by ``AbInitioFit``.
    """

    report = {
        "detected": False,
        "group": "C1",
        "order": 1,
        "reason": None,
    }
    if molecule is None:
        report["reason"] = "electronic driver has no molecule"
        return None, report
    atom_coords = getattr(molecule, "atom_coords", None)
    atom_charges = getattr(molecule, "atom_charges", None)
    if not callable(atom_coords) or not callable(atom_charges):
        report["reason"] = "molecule does not expose atom coordinates and charges"
        return None, report
    try:
        reference = np.asarray(atom_coords(), dtype=float)
        charges = np.asarray(atom_charges(), dtype=int)
    except Exception as error:
        report["reason"] = f"molecular geometry unavailable: {error}"
        return None, report
    if reference.ndim != 2 or reference.shape[1] != 3 or len(charges) != len(reference):
        report["reason"] = "molecular geometry has an incompatible shape"
        return None, report

    origin = np.mean(np.asarray(coord.bounds, dtype=float), axis=1)
    try:
        chart_reference = np.asarray(coord.cartesian(origin), dtype=float)
    except Exception as error:
        report["reason"] = f"coordinate reference unavailable: {error}"
        return None, report
    if chart_reference.shape != reference.shape:
        report["reason"] = "coordinate and molecular geometries have different shapes"
        return None, report
    pairs, reference_features = _pair_distances(reference)
    if not pairs:
        report["reason"] = "fewer than two atoms"
        return None, report
    _chart_pairs, chart_features = _pair_distances(chart_reference)
    feature_scale = max(float(np.max(reference_features)), 1.0)
    if not np.allclose(
        chart_features,
        reference_features,
        atol=float(tolerance) * feature_scale,
        rtol=float(tolerance),
    ):
        report["reason"] = "coordinate midpoint is not the molecular reference geometry"
        return None, report

    distance_matrix = np.linalg.norm(
        reference[:, None, :] - reference[None, :, :], axis=-1
    )
    permutations = _atom_permutations(
        charges, distance_matrix, tolerance=tolerance
    )
    if len(permutations) == 1:
        report["reason"] = "no nontrivial atom permutation was detected"
        return None, report

    ndim = int(coord.ndim)
    spans = np.ptp(np.asarray(coord.bounds, dtype=float), axis=1)
    # A milliscale chart displacement remains linear for molecular coordinates
    # while avoiding cancellation in JAX embeddings evaluated in float32.
    steps = np.maximum(1.0e-5, 1.0e-3 * spans)

    def features(q):
        return _pair_distances(np.asarray(coord.cartesian(q), dtype=float))[1]

    jacobian = np.column_stack(
        [
            (features(origin + np.eye(ndim)[axis] * steps[axis])
             - features(origin - np.eye(ndim)[axis] * steps[axis]))
            / (2.0 * steps[axis])
            for axis in range(ndim)
        ]
    )
    if np.linalg.matrix_rank(jacobian, tol=1.0e-8) < ndim:
        report["reason"] = "pair distances do not span the coordinate chart"
        return None, report

    pair_index = {pair: index for index, pair in enumerate(pairs)}
    bounds = np.asarray(coord.bounds, dtype=float)
    random = np.random.default_rng(1729)
    probes = origin + random.uniform(-0.3, 0.3, size=(6, ndim)) * spans
    actions = []
    labels = []
    kept_permutations = []
    for permutation in permutations:
        rows = []
        for left, right in pairs:
            mapped = tuple(sorted((permutation[left], permutation[right])))
            rows.append(pair_index[mapped])
        action = np.linalg.lstsq(jacobian, jacobian[rows], rcond=None)[0]
        left, _singular, right = np.linalg.svd(action)
        action = left @ right
        if np.linalg.norm(action.T @ action - np.eye(ndim)) > 50.0 * tolerance:
            continue
        transformed = origin + (probes - origin) @ action.T
        if np.any(transformed < bounds[:, 0] - tolerance) or np.any(
            transformed > bounds[:, 1] + tolerance
        ):
            continue
        valid = True
        for point, image in zip(probes, transformed):
            target = features(point)[rows]
            actual = features(image)
            if not np.allclose(
                actual,
                target,
                atol=20.0 * tolerance * feature_scale,
                rtol=20.0 * tolerance,
            ):
                valid = False
                break
        if not valid:
            continue
        if any(np.linalg.norm(action - existing) <= 20.0 * tolerance for existing in actions):
            continue
        actions.append(action)
        kept_permutations.append(permutation)
        labels.append(
            "identity"
            if permutation == tuple(range(len(permutation)))
            else "perm(" + ",".join(str(value) for value in permutation) + ")"
        )
    if len(actions) <= 1:
        report["reason"] = "no nontrivial permutation acts within the coordinate bounds"
        return None, report

    actions = np.asarray(actions)
    identity = int(np.argmin(np.linalg.norm(actions - np.eye(ndim), axis=(1, 2))))
    if identity != 0:
        actions[[0, identity]] = actions[[identity, 0]]
        labels[0], labels[identity] = labels[identity], labels[0]
        kept_permutations[0], kept_permutations[identity] = (
            kept_permutations[identity],
            kept_permutations[0],
        )
    order = len(actions)
    noncommuting = any(
        np.linalg.norm(left @ right - right @ left) > 20.0 * tolerance
        for left in actions
        for right in actions
    )
    name = "S3" if order == 6 and noncommuting else f"G{order}"
    try:
        symmetry = FiniteGroupSamplingSymmetry(
            actions,
            name=name,
            operations=labels,
            origin=origin,
            tolerance=max(100.0 * np.finfo(float).eps, 20.0 * tolerance),
        )
    except ValueError as error:
        report["reason"] = f"detected coordinate actions are incomplete: {error}"
        return None, report
    report.update(
        {
            "detected": True,
            "group": name,
            "order": order,
            "origin": origin.tolist(),
            "atom_permutations": [list(value) for value in kept_permutations],
            "reason": None,
        }
    )
    return symmetry, report


def infer_state_repr(coord_repr, orbit_hamiltonians, *, tolerance=2.0e-7):
    r"""Infer $D_g$ from Procrustes-gauged Hamiltonian symmetry orbits."""

    coord_repr = np.asarray(coord_repr, dtype=float)
    hamiltonians = np.asarray(orbit_hamiltonians)
    if hamiltonians.ndim == 3:
        hamiltonians = hamiltonians[None, ...]
    if hamiltonians.ndim != 4 or hamiltonians.shape[1] != len(coord_repr):
        raise ValueError(
            "orbit_hamiltonians must have shape (nbase, order, nstate, nstate)"
        )
    if hamiltonians.shape[-1] != hamiltonians.shape[-2]:
        raise ValueError("orbit Hamiltonians must be square")
    order = len(coord_repr)
    nstates = hamiltonians.shape[-1]
    identity = np.eye(nstates, dtype=hamiltonians.dtype)
    representations = [identity]
    null_ratios = [0.0]
    for operation in range(1, order):
        equations = []
        for orbit in hamiltonians:
            source = orbit[0]
            image = orbit[operation]
            equations.append(
                np.kron(source.T, np.eye(nstates))
                - np.kron(np.eye(nstates), image)
            )
        equations = np.concatenate(equations, axis=0)
        _left, singular, right = np.linalg.svd(equations, full_matrices=False)
        value = right[-1].conj().reshape(nstates, nstates, order="F")
        left, _singular, right = np.linalg.svd(value)
        value = left @ right
        if np.max(np.abs(np.imag(value))) <= 100.0 * tolerance:
            value = np.real(value)
        representations.append(value)
        null_ratios.append(
            float(singular[-1] / max(singular[-2], np.finfo(float).tiny))
            if len(singular) > 1
            else 0.0
        )
    representations = np.asarray(representations)

    products = np.empty((order, order), dtype=int)
    for left in range(order):
        for right in range(order):
            errors = np.linalg.norm(
                coord_repr - coord_repr[left] @ coord_repr[right], axis=(1, 2)
            )
            products[left, right] = int(np.argmin(errors))
            if errors[products[left, right]] > 100.0 * tolerance:
                raise ValueError("coordinate representations are not a closed group")

    if order <= 12:
        best = None
        for mask in range(1 << (order - 1)):
            signs = np.ones(order)
            for operation in range(1, order):
                if mask & (1 << (operation - 1)):
                    signs[operation] = -1.0
            candidate = signs[:, None, None] * representations
            error = sum(
                np.linalg.norm(
                    candidate[left] @ candidate[right]
                    - candidate[products[left, right]]
                ) ** 2
                for left in range(order)
                for right in range(order)
            )
            if best is None or error < best[0]:
                best = (error, candidate)
        representations = best[1]

    covariance_errors = []
    for orbit in hamiltonians:
        source = orbit[0] - np.trace(orbit[0]) / nstates * identity
        scale = max(float(np.linalg.norm(source)), np.finfo(float).tiny)
        for operation, image in enumerate(orbit):
            image = image - np.trace(image) / nstates * identity
            predicted = (
                representations[operation]
                @ source
                @ representations[operation].conj().T
            )
            covariance_errors.append(float(np.linalg.norm(predicted - image) / scale))
    closure_error = max(
        float(
            np.linalg.norm(
                representations[left] @ representations[right]
                - representations[products[left, right]]
            )
        )
        for left in range(order)
        for right in range(order)
    )
    report = {
        "maximum_covariance_error": max(covariance_errors, default=0.0),
        "closure_error": closure_error,
        "maximum_null_ratio": max(null_ratios, default=0.0),
        "calibration_orbits": int(len(hamiltonians)),
    }
    if closure_error > 1.0e-4 or report["maximum_covariance_error"] > 1.0e-4:
        raise RuntimeError(
            "selected states do not furnish a validated representation of the "
            f"detected group: {report}"
        )
    return representations, report


def coord_irreps(coord_repr, group, *, tolerance=1.0e-7):
    """Decompose the detected coordinate action into invariant irrep blocks."""

    representations = np.asarray(coord_repr, dtype=float)
    order, ndim, _ = representations.shape
    if str(group) == "S3" and order == 6:
        identity = np.eye(ndim)
        operation_orders = []
        for value in representations:
            product = np.eye(ndim)
            found = None
            for power in range(1, 7):
                product = product @ value
                if np.linalg.norm(product - identity) <= 100.0 * tolerance:
                    found = power
                    break
            operation_orders.append(found)
        characters = {
            "A1": np.ones(order),
            "A2": np.asarray(
                [1.0 if value in {1, 3} else -1.0 for value in operation_orders]
            ),
            "E": np.asarray(
                [2.0 if value == 1 else -1.0 if value == 3 else 0.0
                 for value in operation_orders]
            ),
        }
        dimensions = {"A1": 1, "A2": 1, "E": 2}
        blocks = []
        for label, character in characters.items():
            projector = dimensions[label] / order * np.einsum(
                "g,gij->ij", character, representations, optimize=True
            )
            projector = 0.5 * (projector + projector.T)
            values, vectors = np.linalg.eigh(projector)
            basis = vectors[:, values > 0.5]
            if basis.shape[1]:
                blocks.append((label, basis))
    else:
        random = np.random.default_rng(2718).normal(size=(ndim, ndim))
        random = 0.5 * (random + random.T)
        commuting = np.mean(
            [value.T @ random @ value for value in representations], axis=0
        )
        values, vectors = np.linalg.eigh(commuting)
        groups = []
        for index, value in enumerate(values):
            if not groups or abs(value - values[groups[-1][-1]]) > 100.0 * tolerance:
                groups.append([index])
            else:
                groups[-1].append(index)
        blocks = [
            (f"irrep{number + 1}", vectors[:, indices])
            for number, indices in enumerate(groups)
        ]

    basis = np.column_stack([value for _label, value in blocks])
    labels = tuple(label for label, _value in blocks)
    coordinate_blocks = []
    aligned = True
    used = set()
    for _label, value in blocks:
        weights = np.sum(np.abs(value) ** 2, axis=1)
        axes = tuple(int(index) for index in np.flatnonzero(weights > 1.0 - tolerance))
        if len(axes) != value.shape[1] or used.intersection(axes):
            aligned = False
            coordinate_blocks.append(())
        else:
            used.update(axes)
            coordinate_blocks.append(axes)
    transformed = np.einsum(
        "ia,gij,jb->gab", basis, representations, basis, optimize=True
    )
    off_block = np.array(transformed, copy=True)
    start = 0
    for _label, value in blocks:
        stop = start + value.shape[1]
        off_block[:, start:stop, start:stop] = 0.0
        start = stop
    report = {
        "labels": labels,
        "dimensions": tuple(int(value.shape[1]) for _label, value in blocks),
        "coordinate_blocks": tuple(coordinate_blocks),
        "input_basis_is_adapted": bool(aligned),
        "off_block_error": float(np.max(np.abs(off_block))),
    }
    return labels, tuple(coordinate_blocks), basis, report


class PhenolReflectionSymmetry(SamplingSymmetry):
    r"""Identify phenol coordinates related by reflection in the ring plane.

    By default only the CCOH torsion is odd.  Additional out-of-plane reduced
    coordinates, such as Wilson ``16a``, are supplied through ``odd_axes`` and
    are reflected by the same molecular operation.
    """

    name = "phenol-phi-reflection"
    operation = "sigma_xy"
    matrix = np.diag((1.0, 1.0, -1.0))

    def __init__(self, *, torsion_axis=1, odd_axes=None, tolerance=1.0e-12):
        self.torsion_axis = int(torsion_axis)
        self.odd_axes = tuple(
            dict.fromkeys(
                (self.torsion_axis,)
                if odd_axes is None
                else tuple(int(axis) for axis in odd_axes)
            )
        )
        self.tolerance = float(tolerance)
        if self.torsion_axis < 0:
            raise ValueError("torsion_axis must be non-negative")
        if not self.odd_axes or any(axis < 0 for axis in self.odd_axes):
            raise ValueError("odd_axes must contain non-negative coordinate axes")
        if self.torsion_axis not in self.odd_axes:
            raise ValueError("odd_axes must contain torsion_axis")
        if self.tolerance < 0.0:
            raise ValueError("tolerance must be non-negative")

    def resolve(self, coordinates):
        coordinates = np.asarray(coordinates, dtype=float)
        if coordinates.ndim != 1 or max(self.odd_axes) >= coordinates.size:
            raise ValueError(
                "phenol reflection needs a one-dimensional coordinate containing "
                "the configured torsion axis"
            )
        representative = np.array(coordinates, copy=True)
        orientation = 0.0
        for axis in self.odd_axes:
            if abs(representative[axis]) > self.tolerance:
                orientation = float(representative[axis])
                break
        if orientation < 0.0:
            representative[list(self.odd_axes)] *= -1.0
            operation = self.operation
        else:
            operation = "identity"
        for axis in self.odd_axes:
            if abs(representative[axis]) <= self.tolerance:
                representative[axis] = 0.0
        return SamplingSymmetryImage(
            tuple(float(value) for value in representative), operation
        )

    def _reflected_coordinates(self, coordinates):
        reflected = np.asarray(coordinates, dtype=float).copy()
        if reflected.ndim != 1 or max(self.odd_axes) >= reflected.size:
            raise ValueError(
                "phenol reflection needs a one-dimensional coordinate containing "
                "the configured torsion axis"
            )
        reflected[list(self.odd_axes)] *= -1.0
        for axis in self.odd_axes:
            if abs(reflected[axis]) <= self.tolerance:
                reflected[axis] = 0.0
        return tuple(float(value) for value in reflected)

    def images(self, coordinates):
        coordinates = tuple(float(value) for value in coordinates)
        reflected = self._reflected_coordinates(coordinates)
        return tuple(dict.fromkeys((coordinates, reflected)))

    def pair_images(self, left, right):
        left = tuple(float(value) for value in left)
        right = tuple(float(value) for value in right)
        reflected = (
            self._reflected_coordinates(left),
            self._reflected_coordinates(right),
        )
        return tuple(dict.fromkeys(((left, right), reflected)))

    @staticmethod
    def _basis(protocol):
        if not isinstance(protocol, dict) or "basis" not in protocol:
            raise ValueError(
                "PhenolReflectionSymmetry needs protocol['basis'] to transform "
                "molecular orbitals"
            )
        return protocol["basis"]

    @staticmethod
    @lru_cache(maxsize=16)
    def _ao_signs(basis_key):
        from pyqed.models.phenol_coordinates import (
            PHENOL_SPECIES,
            PhenolReactiveChart,
        )
        from pyqed.qchem import Molecule
        from pyqed.qchem.symmetry import _component_parity

        basis = basis_key
        chart = PhenolReactiveChart()
        geometry = chart.geometry(chart.equilibrium)
        molecule = Molecule(
            atom=list(zip(PHENOL_SPECIES, geometry)),
            unit="angstrom",
            basis=basis,
            charge=0,
            spin=0,
        ).topyscf()
        molecule.build(verbose=0)
        signs = []
        for _atom, _symbol, _shell, component in molecule.ao_labels(fmt=False):
            component = str(component).replace("^", "").strip()
            signs.append(_component_parity(component, (1, 1, -1)))
        return np.asarray(signs, dtype=float)

    def _reflection_signs(self, basis):
        if not isinstance(basis, str):
            raise TypeError(
                "PhenolReflectionSymmetry currently requires a named string basis"
            )
        return self._ao_signs(str(basis))

    @staticmethod
    def _reflect_cartesian(value):
        value = np.asarray(value)
        return np.einsum("...a,ba->...b", value, PhenolReflectionSymmetry.matrix)

    def transform_record(
        self,
        record,
        image,
        *,
        representative_geometry,
        requested_geometry,
        protocol,
    ):
        if image.operation == "identity":
            return record
        if image.operation != self.operation:
            raise ValueError(f"unsupported phenol orbit operation {image.operation!r}")
        if not isinstance(record, dict):
            raise TypeError(
                "PhenolReflectionSymmetry expects mapping electronic records"
            )

        representative_geometry = np.asarray(representative_geometry, dtype=float)
        requested_geometry = np.asarray(requested_geometry, dtype=float)
        reflected = representative_geometry @ self.matrix.T
        if not np.allclose(
            reflected,
            requested_geometry,
            atol=max(self.tolerance, 1.0e-10),
            rtol=0.0,
        ):
            raise ValueError(
                "the requested phenol geometry is not the sigma_xy image of its "
                "orbit representative"
            )

        transformed = dict(record)
        transformed["geometry"] = np.array(requested_geometry, copy=True)
        if "mo_coeff" in record:
            signs = self._reflection_signs(self._basis(protocol))
            coefficients = np.asarray(record["mo_coeff"])
            if coefficients.ndim != 2 or coefficients.shape[0] != signs.size:
                raise ValueError(
                    "mo_coeff does not match the AO basis used by the phenol protocol"
                )
            transformed["mo_coeff"] = signs[:, None] * coefficients

        for key in (
            "gradient",
            "gradients",
            "force",
            "forces",
            "dipole",
            "dipoles",
            "transition_dipole",
            "transition_dipoles",
            "nac",
            "nacs",
        ):
            if key in record and np.asarray(record[key]).shape[-1:] == (3,):
                transformed[key] = self._reflect_cartesian(record[key])
        transformed["sampling_symmetry"] = self.view_key(image)
        return transformed

    def metadata(self):
        return {
            "name": self.name,
            "coordinate_axis": self.torsion_axis,
            "odd_coordinate_axes": list(self.odd_axes),
            "canonical_domain": "first nonzero odd coordinate >= 0",
            "operation": self.operation,
            "cartesian_matrix": self.matrix.tolist(),
        }


__all__ = [
    "SamplingSymmetry",
    "SamplingSymmetryImage",
    "FiniteGroupSamplingSymmetry",
    "PhenolReflectionSymmetry",
]
