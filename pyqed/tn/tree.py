"""Finite open tree tensor networks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from operator import index

import numpy as np


def _integer(value, name: str) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be an integer.")
    try:
        return index(value)
    except TypeError as error:
        raise ValueError(f"{name} must be an integer.") from error


def _validated_dims(dims) -> tuple[int, ...]:
    dims = tuple(_integer(dim, "dimensions") for dim in dims)
    if not dims or any(dim < 1 for dim in dims):
        raise ValueError("dims must contain positive integers.")
    return dims


def _validated_parents(parents, nsites: int, root) -> tuple[int | None, ...]:
    parents = tuple(parents)
    if len(parents) != nsites:
        raise ValueError("parents must contain one entry per site.")
    values = tuple(
        None if parent is None else _integer(parent, "parent sites")
        for parent in parents
    )
    roots = tuple(site for site, parent in enumerate(values) if parent is None)
    if len(roots) != 1:
        raise ValueError("parents must contain exactly one root marked by None.")
    inferred_root = roots[0]
    if root is not None and _integer(root, "root") != inferred_root:
        raise ValueError("root is inconsistent with parents.")
    for site, parent in enumerate(values):
        if parent is not None and (parent < 0 or parent >= nsites or parent == site):
            raise ValueError(f"parent of site {site} is invalid.")

    for start in range(nsites):
        seen = set()
        site = start
        while values[site] is not None:
            if site in seen:
                raise ValueError("parents must define an acyclic tree.")
            seen.add(site)
            site = values[site]
        if site != inferred_root:
            raise ValueError("every site must be connected to the root.")
    return values


def _edge_dimension(bond_dim, child: int, parent: int) -> int:
    if isinstance(bond_dim, Mapping):
        if child in bond_dim:
            value = bond_dim[child]
        elif (parent, child) in bond_dim:
            value = bond_dim[parent, child]
        elif (child, parent) in bond_dim:
            value = bond_dim[child, parent]
        else:
            raise ValueError(f"bond_dim has no entry for edge {(parent, child)}.")
    elif (
        isinstance(bond_dim, Sequence)
        or (isinstance(bond_dim, np.ndarray) and bond_dim.ndim > 0)
    ) and not isinstance(bond_dim, (str, bytes)):
        try:
            value = bond_dim[child]
        except IndexError as error:
            raise ValueError(
                f"bond_dim has no entry for child site {child}."
            ) from error
    else:
        value = bond_dim
    value = _integer(value, "bond dimensions")
    if value < 1:
        raise ValueError("bond dimensions must be positive.")
    return value


class TTN:
    r"""A finite open tree tensor network with one physical leg per node.

    Tensor axes are ordered as ``(physical, bonds-to-neighbors...)``.  For each
    node, the parent bond comes first when present, followed by child bonds in
    increasing site order.  ``parents`` contains exactly one ``None`` entry,
    which identifies the root.

    Parameters
    ----------
    dims
        Physical dimension at every node.
    parents
        Parent site of every node, with ``None`` at the root.
    bond_dim
        A common positive bond dimension, a per-child sequence, or a mapping
        keyed by child site or undirected edge.  Ignored when ``tensors`` are
        supplied, because their edge dimensions are inferred.
    tensors
        Optional tensors in the axis convention described above.
    seed
        Seed used for random initialization.
    root
        Optional assertion for the root encoded by ``parents``.
    """

    def __init__(
        self,
        dims,
        parents,
        *,
        bond_dim=1,
        tensors=None,
        seed: int | None = None,
        root: int | None = None,
    ):
        self.dims = _validated_dims(dims)
        self.parents = _validated_parents(parents, len(self.dims), root)
        self.root = self.parents.index(None)

        children = [[] for _ in self.dims]
        for child, parent in enumerate(self.parents):
            if parent is not None:
                children[parent].append(child)
        self.children = tuple(tuple(sorted(value)) for value in children)
        self.neighbors = tuple(
            ((parent,) if parent is not None else ()) + self.children[site]
            for site, parent in enumerate(self.parents)
        )

        if tensors is None:
            edge_dims = {
                self._edge_key(parent, child): _edge_dimension(
                    bond_dim, child, parent
                )
                for child, parent in enumerate(self.parents)
                if parent is not None
            }
            rng = np.random.default_rng(seed)
            self.tensors = []
            for site, dim in enumerate(self.dims):
                shape = (dim,) + tuple(
                    edge_dims[self._edge_key(site, neighbor)]
                    for neighbor in self.neighbors[site]
                )
                tensor = rng.normal(size=shape) / np.sqrt(np.prod(shape))
                self.tensors.append(tensor)
        else:
            if len(tensors) != len(self.dims):
                raise ValueError("tensors must contain one entry per site.")
            self.tensors = [np.asarray(tensor).copy() for tensor in tensors]

        self.center: int | None = None
        self.validate()

    @property
    def nsites(self) -> int:
        return len(self.dims)

    @property
    def preorder(self) -> tuple[int, ...]:
        order = []
        stack = [self.root]
        while stack:
            site = stack.pop()
            order.append(site)
            stack.extend(reversed(self.children[site]))
        return tuple(order)

    @property
    def postorder(self) -> tuple[int, ...]:
        return tuple(reversed(self.preorder))

    @staticmethod
    def _edge_key(left: int, right: int) -> tuple[int, int]:
        return (left, right) if left < right else (right, left)

    def _bond_axis(self, site: int, neighbor: int) -> int:
        try:
            return 1 + self.neighbors[site].index(neighbor)
        except (IndexError, ValueError) as error:
            raise ValueError(f"sites {site} and {neighbor} are not neighbors.") from error

    @property
    def edge_dims(self) -> dict[tuple[int, int], int]:
        return {
            self._edge_key(child, parent): int(
                self.tensors[child].shape[self._bond_axis(child, parent)]
            )
            for child, parent in enumerate(self.parents)
            if parent is not None
        }

    def validate(self) -> None:
        """Validate physical axes, ranks, and dimensions on shared edges."""
        if len(self.tensors) != self.nsites:
            raise ValueError("tensors must contain one entry per site.")
        for site, tensor in enumerate(self.tensors):
            expected_ndim = 1 + len(self.neighbors[site])
            if tensor.ndim != expected_ndim:
                raise ValueError(
                    f"tensor {site} must have {expected_ndim} axes in "
                    "(physical, neighbor bonds...) order."
                )
            if tensor.shape[0] != self.dims[site]:
                raise ValueError(f"tensor {site} has the wrong physical dimension.")
            if any(size < 1 for size in tensor.shape):
                raise ValueError(f"tensor {site} cannot contain an empty axis.")
        for child, parent in enumerate(self.parents):
            if parent is None:
                continue
            child_dim = self.tensors[child].shape[self._bond_axis(child, parent)]
            parent_dim = self.tensors[parent].shape[self._bond_axis(parent, child)]
            if child_dim != parent_dim:
                raise ValueError(
                    f"bond dimension mismatch on edge {(parent, child)}: "
                    f"{parent_dim} != {child_dim}."
                )

    def copy(self) -> "TTN":
        """Return an independent copy."""
        result = type(self)(
            self.dims,
            self.parents,
            tensors=[tensor.copy() for tensor in self.tensors],
        )
        result.center = self.center
        return result

    def amplitude(self, configuration) -> complex:
        """Contract the amplitude of one product-basis configuration."""
        configuration = tuple(
            _integer(value, "configuration states") for value in configuration
        )
        if len(configuration) != self.nsites:
            raise ValueError("configuration must contain one value per site.")
        if any(value < 0 or value >= self.dims[site] for site, value in enumerate(configuration)):
            raise ValueError("configuration contains an out-of-range local state.")

        messages = {}
        for site in self.postorder:
            value = self.tensors[site][configuration[site]]
            for child in reversed(self.children[site]):
                value = np.tensordot(value, messages[child], axes=([-1], [0]))
            messages[site] = value
        return np.asarray(messages[self.root]).item()

    def state_vector(self, *, normalize: bool = False) -> np.ndarray:
        """Materialize the product-basis vector for reference calculations."""
        dtype = np.result_type(*[tensor.dtype for tensor in self.tensors])
        state = np.empty(self.dims, dtype=dtype)
        for configuration in np.ndindex(*self.dims):
            state[configuration] = self.amplitude(configuration)
        vector = state.reshape(-1)
        if normalize:
            norm = np.linalg.norm(vector)
            if norm <= 0.0:
                raise ValueError("cannot normalize a zero TTN state.")
            vector = vector / norm
        return vector

    def _norm_message(self, source: int, target: int, cache) -> np.ndarray:
        key = (source, target)
        if key in cache:
            return cache[key]
        tensor = self.tensors[source]
        degree = len(self.neighbors[source])
        bra_labels = [0] + list(range(1, degree + 1))
        ket_labels = [0] + list(range(degree + 1, 2 * degree + 1))
        arguments = [tensor, bra_labels, tensor.conj(), ket_labels]
        for position, neighbor in enumerate(self.neighbors[source]):
            if neighbor == target:
                continue
            message = self._norm_message(neighbor, source, cache)
            arguments.extend(
                [message, [bra_labels[1 + position], ket_labels[1 + position]]]
            )
        target_position = self.neighbors[source].index(target)
        output = [bra_labels[1 + target_position], ket_labels[1 + target_position]]
        result = np.einsum(*arguments, output, optimize=True)
        cache[key] = result
        return result

    def edge_message(self, source: int, target: int) -> np.ndarray:
        """Return the norm message directed from ``source`` to ``target``."""
        source = _integer(source, "source site")
        target = _integer(target, "target site")
        self._bond_axis(source, target)
        return np.array(self._norm_message(source, target, {}), copy=True)

    def _validated_operators(self, operators) -> dict[int, np.ndarray]:
        if operators is None:
            return {}
        if not isinstance(operators, Mapping):
            raise TypeError("operators must map site indices to local matrices.")
        result = {}
        for site, operator in operators.items():
            site = _integer(site, "operator site")
            if site < 0 or site >= self.nsites:
                raise IndexError("operator site is outside the TTN.")
            operator = np.asarray(operator)
            expected_shape = (self.dims[site], self.dims[site])
            if operator.shape != expected_shape:
                raise ValueError(
                    f"operator at site {site} must have shape {expected_shape}."
                )
            result[site] = operator
        return result

    def _operator_message(
        self,
        source: int,
        target: int,
        operators: Mapping[int, np.ndarray],
        cache,
        *,
        active_edges=None,
        identity_cache=None,
    ) -> np.ndarray:
        key = (source, target)
        if key in cache:
            return cache[key]
        if active_edges is not None and key not in active_edges:
            return self._operator_message(
                source,
                target,
                {},
                identity_cache,
            )
        tensor = self.tensors[source]
        degree = len(self.neighbors[source])
        bra_labels = [0] + list(range(1, degree + 1))
        operator = operators.get(source)
        if operator is None:
            ket_labels = [0] + list(range(degree + 1, 2 * degree + 1))
            arguments = [tensor.conj(), bra_labels, tensor, ket_labels]
        else:
            ket_labels = [degree + 1] + list(
                range(degree + 2, 2 * degree + 2)
            )
            arguments = [
                tensor.conj(),
                bra_labels,
                operator,
                [bra_labels[0], ket_labels[0]],
                tensor,
                ket_labels,
            ]
        for position, neighbor in enumerate(self.neighbors[source]):
            if neighbor == target:
                continue
            message = self._operator_message(
                neighbor,
                source,
                operators,
                cache,
                active_edges=active_edges,
                identity_cache=identity_cache,
            )
            arguments.extend(
                [message, [bra_labels[1 + position], ket_labels[1 + position]]]
            )
        target_position = self.neighbors[source].index(target)
        output = [bra_labels[1 + target_position], ket_labels[1 + target_position]]
        result = np.einsum(*arguments, output, optimize=True)
        cache[key] = result
        return result

    def _effective_product_operator(
        self,
        operators: Mapping[int, np.ndarray],
        center: int,
        identity_cache,
    ) -> np.ndarray:
        tensor = self.tensors[center]
        toward_center, _ = self._traversal_from(center)
        active_edges = set()
        for operator_site in operators:
            site = operator_site
            while site != center:
                target = toward_center[site]
                active_edges.add((site, target))
                site = target

        rank = tensor.ndim
        bra_labels = list(range(rank))
        ket_labels = list(range(rank, 2 * rank))
        local_operator = operators.get(center)
        if local_operator is None:
            local_operator = np.eye(self.dims[center], dtype=tensor.dtype)
        arguments = [
            local_operator,
            [bra_labels[0], ket_labels[0]],
        ]
        cache = {}
        for position, neighbor in enumerate(self.neighbors[center]):
            message = self._operator_message(
                neighbor,
                center,
                operators,
                cache,
                active_edges=active_edges,
                identity_cache=identity_cache,
            )
            arguments.extend(
                [message, [bra_labels[1 + position], ket_labels[1 + position]]]
            )
        effective = np.einsum(
            *arguments,
            bra_labels + ket_labels,
            optimize=True,
        )
        return effective.reshape(tensor.size, tensor.size)

    def effective_product_operator(
        self,
        operators=None,
        *,
        center: int | None = None,
    ) -> np.ndarray:
        r"""Contract a product operator into the open tensor space at ``center``.

        ``operators`` maps selected sites to local matrices; identity operators
        are implied elsewhere.  If the TTN is canonical at ``center``, the
        returned matrix is the ordinary one-site effective operator because the
        corresponding effective overlap is the identity.
        """
        operators = self._validated_operators(operators)
        if center is None:
            center = self.root
        else:
            center = _integer(center, "center site")
            if center < 0 or center >= self.nsites:
                raise IndexError("center is outside the TTN.")
        return self._effective_product_operator(operators, center, {})

    def effective_operator_sum(
        self,
        terms,
        *,
        center: int | None = None,
    ) -> np.ndarray:
        r"""Return the center effective matrix for a sum of product operators.

        Each term is a ``(coefficient, operators)`` pair.  Identity-branch
        messages are shared across the terms to avoid redundant contractions.
        """
        if center is None:
            center = self.root
        else:
            center = _integer(center, "center site")
            if center < 0 or center >= self.nsites:
                raise IndexError("center is outside the TTN.")
        identity_cache = {}
        effective = None
        for coefficient, operators in terms:
            operators = self._validated_operators(operators)
            contribution = coefficient * self._effective_product_operator(
                operators,
                center,
                identity_cache,
            )
            effective = contribution if effective is None else effective + contribution
        if effective is None:
            size = self.tensors[center].size
            dtype = np.result_type(*[tensor.dtype for tensor in self.tensors], float)
            effective = np.zeros((size, size), dtype=dtype)
        return effective

    def expectation_value(self, operators=None, *, normalize: bool = True):
        r"""Return the expectation of a product of local operators.

        ``operators`` maps site indices to local matrices and identity operators
        are implied on all omitted sites.
        """
        center = self.root
        tensor = self.tensors[center].reshape(-1)
        effective = self.effective_product_operator(operators, center=center)
        value = np.vdot(tensor, effective @ tensor)
        if normalize:
            norm_squared = self.norm_squared()
            if norm_squared <= 0.0:
                raise ValueError("cannot normalize an expectation for a zero TTN state.")
            value = value / norm_squared
        return np.real_if_close(value).item()

    def norm_squared(self) -> float:
        """Contract ``<psi|psi>`` exactly by tree messages."""
        root = self.root
        tensor = self.tensors[root]
        degree = len(self.neighbors[root])
        bra_labels = [0] + list(range(1, degree + 1))
        ket_labels = [0] + list(range(degree + 1, 2 * degree + 1))
        arguments = [tensor, bra_labels, tensor.conj(), ket_labels]
        cache = {}
        for position, neighbor in enumerate(self.neighbors[root]):
            message = self._norm_message(neighbor, root, cache)
            arguments.extend(
                [message, [bra_labels[1 + position], ket_labels[1 + position]]]
            )
        contraction = np.einsum(*arguments, [], optimize=True)
        value = contraction.item()
        real_dtype = contraction.real.dtype
        if not np.issubdtype(real_dtype, np.inexact):
            real_dtype = np.dtype(float)
        finfo = np.finfo(real_dtype)
        tolerance = 128.0 * finfo.eps * max(finfo.tiny, abs(value))
        if abs(np.imag(value)) > tolerance or np.real(value) < -tolerance:
            raise ValueError("TTN norm contraction produced a nonphysical value.")
        return max(0.0, float(np.real(value)))

    def norm(self) -> float:
        """Return the Hilbert-space norm."""
        return float(np.sqrt(self.norm_squared()))

    def normalize(self) -> "TTN":
        """Normalize the represented state in place and return ``self``."""
        norm = self.norm()
        if norm <= 0.0:
            raise ValueError("cannot normalize a zero TTN state.")
        scale_site = self.root if self.center is None else self.center
        self.tensors[scale_site] = self.tensors[scale_site] / norm
        return self

    def _traversal_from(self, center: int):
        parents = {center: None}
        preorder = []
        stack = [center]
        while stack:
            site = stack.pop()
            preorder.append(site)
            children = [neighbor for neighbor in self.neighbors[site] if neighbor != parents[site]]
            for child in reversed(children):
                parents[child] = site
                stack.append(child)
        return parents, preorder

    def _gauge_toward(self, source: int, target: int) -> None:
        axis = self._bond_axis(source, target)
        tensor = self.tensors[source]
        permutation = tuple(index for index in range(tensor.ndim) if index != axis) + (axis,)
        moved = tensor.transpose(permutation)
        row_shape = moved.shape[:-1]
        matrix = moved.reshape(-1, moved.shape[-1])
        q, r = np.linalg.qr(matrix, mode="reduced")
        rank = q.shape[1]
        q_tensor = q.reshape(row_shape + (rank,))
        inverse = np.argsort(permutation)
        self.tensors[source] = q_tensor.transpose(inverse)

        target_axis = self._bond_axis(target, source)
        absorbed = np.tensordot(r, self.tensors[target], axes=(1, target_axis))
        self.tensors[target] = np.moveaxis(absorbed, 0, target_axis)

    def canonicalize(self, center: int) -> "TTN":
        """Put every branch in an isometric gauge directed at ``center``."""
        center = _integer(center, "center site")
        if center < 0 or center >= self.nsites:
            raise IndexError("center is outside the TTN.")
        parents, preorder = self._traversal_from(center)
        for source in reversed(preorder[1:]):
            self._gauge_toward(source, parents[source])
        self.center = center
        self.validate()
        return self

    def move_center(self, site: int) -> "TTN":
        """Canonicalize the network toward ``site``."""
        return self.canonicalize(site)
