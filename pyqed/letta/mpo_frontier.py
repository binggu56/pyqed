"""Directional frontier messages for graph-tied tensors and an MPO."""

from __future__ import annotations

import numpy as np
from opt_einsum import contract_expression


def _factor_physical_operator(operator, physical_dims):
    """Factor one fused physical operator into exact sequential local cores."""
    operator = np.asarray(operator)
    physical_dims = tuple(int(dim) for dim in physical_dims)
    physical_dim = int(np.prod(physical_dims))
    if operator.shape != (physical_dim, physical_dim):
        raise ValueError("physical_dims are inconsistent with the operator.")
    nphysical = len(physical_dims)
    reshaped = operator.reshape(*physical_dims, *physical_dims)
    axes = tuple(
        axis
        for pair in zip(
            range(nphysical),
            range(nphysical, 2 * nphysical),
        )
        for axis in pair
    )
    pair_dims = tuple(dim * dim for dim in physical_dims)
    work = reshaped.transpose(axes).reshape(1, *pair_dims, 1)
    cores = []
    previous_rank = 1
    for position, (dim, pair_dim) in enumerate(
        zip(physical_dims[:-1], pair_dims[:-1])
    ):
        matrix = work.reshape(previous_rank * pair_dim, -1)
        left, singular_values, right = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        threshold = (
            np.finfo(singular_values.dtype).eps
            * max(matrix.shape)
            * max(float(singular_values[0]), np.finfo(float).tiny)
        )
        rank = max(1, int(np.count_nonzero(singular_values > threshold)))
        cores.append(
            left[:, :rank]
            .reshape(previous_rank, dim, dim, rank)
            .transpose(0, 3, 1, 2)
        )
        work = (
            singular_values[:rank, None] * right[:rank]
        ).reshape(
            rank,
            *pair_dims[position + 1 :],
            1,
        )
        previous_rank = rank
    final_dim = physical_dims[-1]
    cores.append(
        work.reshape(
            previous_rank,
            final_dim,
            final_dim,
            1,
        ).transpose(0, 3, 1, 2)
    )
    return tuple(cores)


def _factor_mpo_physical_transitions(tensor, physical_dims):
    """Factor every nonzero MPO-channel operator independently."""
    tensor = np.asarray(tensor)
    groups = {}
    for left, right in zip(*np.nonzero(np.any(tensor != 0, axis=(2, 3)))):
        cores = _factor_physical_operator(
            tensor[left, right],
            physical_dims,
        )
        key = tuple(core.shape for core in cores)
        groups.setdefault(key, []).append((int(left), int(right), cores))
    result = []
    for entries in groups.values():
        result.append(
            (
                np.asarray([entry[0] for entry in entries], dtype=int),
                np.asarray([entry[1] for entry in entries], dtype=int),
                tuple(
                    np.stack([entry[2][position] for entry in entries])
                    for position in range(len(entries[0][2]))
                ),
            )
        )
    return tuple(result)


class MPOFrontier:
    r"""Cache exact left/right cut messages for a graph-tied double layer.

    A message at cut ``c`` carries the bra/ket virtual bond, the MPO bond, and
    both bra and ket values of every future physical variable referenced by a
    tensor to the left of the cut.  Its size is therefore controlled by the
    weighted physical-tie frontier, not by the full Hilbert-space dimension.
    """

    def __init__(
        self,
        dims,
        physical_sites,
        tensor_shapes,
        mpo_tensors,
        *,
        paired_sites=None,
        optimize="greedy",
        physical_factor_dims=None,
        physical_kernel="auto",
    ):
        self.dims = tuple(int(dim) for dim in dims)
        self.physical_groups = tuple(tuple(sites) for sites in physical_sites)
        self.tensor_shapes = tuple(tuple(shape) for shape in tensor_shapes)
        self.mpo_tensors = tuple(np.asarray(tensor) for tensor in mpo_tensors)
        if paired_sites is None:
            paired_sites = range(len(self.dims))
        self.paired_sites = frozenset(int(site) for site in paired_sites)
        if any(site < 0 or site >= len(self.dims) for site in self.paired_sites):
            raise ValueError("paired_sites contains an invalid site.")
        self.optimize = optimize
        self.physical_kernel = str(physical_kernel).lower().replace("-", "_")
        if self.physical_kernel not in {"auto", "fused", "sequential"}:
            raise ValueError(
                "physical_kernel must be 'auto', 'fused', or 'sequential'."
            )
        self.nsites = len(self.dims)
        if not (
            len(self.physical_groups)
            == len(self.tensor_shapes)
            == len(self.mpo_tensors)
            == self.nsites
        ):
            raise ValueError("frontier inputs must contain one entry per site.")
        if physical_factor_dims is None:
            self.physical_factor_dims = (None,) * self.nsites
        else:
            if len(physical_factor_dims) != self.nsites:
                raise ValueError(
                    "physical_factor_dims must contain one entry per site."
                )
            normalized = []
            for site, factors in enumerate(physical_factor_dims):
                if factors is None:
                    normalized.append(None)
                    continue
                factors = tuple(int(dim) for dim in factors)
                if not factors or any(dim < 1 for dim in factors):
                    raise ValueError(
                        "physical factor dimensions must be positive."
                    )
                if int(np.prod(factors)) != self.dims[site]:
                    raise ValueError(
                        f"physical factors do not multiply to dims[{site}]."
                    )
                normalized.append(factors)
            self.physical_factor_dims = tuple(normalized)

        self.virtual_bonds = (self.tensor_shapes[0][0],) + tuple(
            shape[1] for shape in self.tensor_shapes
        )
        self.mpo_bonds = (self.mpo_tensors[0].shape[0],) + tuple(
            tensor.shape[1] for tensor in self.mpo_tensors
        )
        for site, (shape, mpo) in enumerate(zip(self.tensor_shapes, self.mpo_tensors)):
            if shape[0] != self.virtual_bonds[site]:
                raise ValueError(f"virtual bond mismatch before site {site}.")
            if mpo.shape != (
                self.mpo_bonds[site],
                self.mpo_bonds[site + 1],
                self.dims[site],
                self.dims[site],
            ):
                raise ValueError(f"MPO tensor {site} has an invalid shape.")
            if site not in self.paired_sites:
                reconstructed = np.zeros_like(mpo)
                diagonal = np.arange(self.dims[site])
                reconstructed[:, :, diagonal, diagonal] = mpo[
                    :, :, diagonal, diagonal
                ]
                scale = max(float(np.linalg.norm(mpo)), 1.0)
                if not np.allclose(
                    mpo,
                    reconstructed,
                    rtol=0.0,
                    atol=256.0 * np.finfo(float).eps * scale,
                ):
                    raise ValueError(
                        f"unpaired MPO site {site} is not diagonal."
                    )
        if self.virtual_bonds[0] != 1 or self.virtual_bonds[-1] != 1:
            raise ValueError("frontier virtual boundary dimensions must be one.")
        if self.mpo_bonds[0] != 1 or self.mpo_bonds[-1] != 1:
            raise ValueError("frontier MPO boundary dimensions must be one.")

        first_reference = list(range(self.nsites))
        for left_site, group in enumerate(self.physical_groups):
            for physical_site in group[1:]:
                first_reference[physical_site] = min(
                    first_reference[physical_site], left_site
                )
        self.frontier_sites = tuple(
            tuple(
                physical_site
                for physical_site in range(cut, self.nsites)
                if first_reference[physical_site] < cut
            )
            for cut in range(self.nsites + 1)
        )

        start = 0
        self.bra_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.ket_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.operator_bonds = tuple(range(start, start + self.nsites + 1))
        start += self.nsites + 1
        self.bra_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.ket_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.local_ket_physical = tuple(range(start, start + self.nsites))
        start += self.nsites
        self.vector_batch_label = start
        self._identities = tuple(np.eye(dim) for dim in self.dims)
        self._physical_mpo_cores = tuple(
            (
                None
                if factors is None or len(factors) == 1
                else _factor_mpo_physical_transitions(tensor, factors)
            )
            for tensor, factors in zip(
                self.mpo_tensors,
                self.physical_factor_dims,
            )
        )
        self._expressions = {}

    @property
    def uses_sequential_physical_kernels(self):
        return any(cores is not None for cores in self._physical_mpo_cores)

    @property
    def plan_count(self):
        return len(self._expressions)

    @property
    def peak_message_elements(self):
        return max(self.message_elements(cut) for cut in range(self.nsites + 1))

    def message_elements(self, cut):
        return int(np.prod(self.message_shape(cut)))

    @property
    def total_message_elements(self):
        return sum(self.message_elements(cut) for cut in range(self.nsites + 1))

    def message_shape(self, cut):
        cut = int(cut)
        shape = [
            self.virtual_bonds[cut],
            self.virtual_bonds[cut],
            self.mpo_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            shape.append(self.dims[site])
            if site in self.paired_sites:
                shape.append(self.dims[site])
        return tuple(shape)

    def _message_labels(self, cut):
        labels = [
            self.bra_bonds[cut],
            self.ket_bonds[cut],
            self.operator_bonds[cut],
        ]
        for site in self.frontier_sites[cut]:
            labels.append(self.bra_physical[site])
            if site in self.paired_sites:
                labels.append(self.ket_physical[site])
        return tuple(labels)

    def _ket_label(self, site):
        return (
            self.ket_physical[site]
            if site in self.paired_sites
            else self.bra_physical[site]
        )

    def _tensor_labels(self, site, *, bra):
        bonds = self.bra_bonds if bra else self.ket_bonds
        physical = (
            self.bra_physical
            if bra
            else tuple(self._ket_label(index) for index in range(self.nsites))
        )
        return (
            bonds[site],
            bonds[site + 1],
            *(physical[index] for index in self.physical_groups[site]),
        )

    def left_boundary(self):
        return np.ones(self.message_shape(0))

    def right_boundary(self):
        return np.ones(self.message_shape(self.nsites))

    def _advance_expression(self, direction, site):
        key = ("advance", direction, int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        source_cut, target_cut = (
            (site, site + 1) if direction == "left" else (site + 1, site)
        )
        expression = contract_expression(
            self.message_shape(source_cut),
            self._message_labels(source_cut),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=True),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                self._ket_label(site),
            ),
            self._message_labels(target_cut),
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

    def advance_left(self, message, tensors, site):
        expression = self._advance_expression("left", site)
        tensor = tensors[int(site)]
        return expression(message, tensor.conj(), tensor, self.mpo_tensors[int(site)])

    def advance_right(self, message, tensors, site):
        expression = self._advance_expression("right", site)
        tensor = tensors[int(site)]
        return expression(message, tensor.conj(), tensor, self.mpo_tensors[int(site)])

    def build_left(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[0] = self.left_boundary()
        for site in range(self.nsites):
            messages[site + 1] = self.advance_left(messages[site], tensors, site)
        return messages

    def build_right(self, tensors):
        messages = [None] * (self.nsites + 1)
        messages[-1] = self.right_boundary()
        for site in range(self.nsites - 1, -1, -1):
            messages[site] = self.advance_right(messages[site + 1], tensors, site)
        return messages

    def scalar(self, tensors):
        value = self.left_boundary()
        for site in range(self.nsites):
            value = self.advance_left(value, tensors, site)
        return np.asarray(value).reshape(()).item()

    def _hole_expression(self, mode, site):
        key = ("hole", mode, int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        if mode not in {"matrix", "action"}:
            raise ValueError("hole mode must be 'matrix' or 'action'.")
        site = int(site)
        bra_labels = (
            self.bra_bonds[site],
            self.bra_bonds[site + 1],
            *(self.bra_physical[index] for index in self.physical_groups[site]),
        )
        ket_labels = (
            self.ket_bonds[site],
            self.ket_bonds[site + 1],
            *(
                (
                    self.ket_physical[index]
                    if index in self.paired_sites
                    else self.local_ket_physical[index]
                )
                for index in self.physical_groups[site]
            ),
        )
        own_ket_label = (
            self.ket_physical[site]
            if site in self.paired_sites
            else self.local_ket_physical[site]
        )
        arguments = [
            self.message_shape(site),
            self._message_labels(site),
            self.message_shape(site + 1),
            self._message_labels(site + 1),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                own_ket_label,
            ),
        ]
        for physical_site in self.physical_groups[site][1:]:
            if physical_site not in self.paired_sites:
                arguments.extend(
                    (
                        (self.dims[physical_site], self.dims[physical_site]),
                        (
                            self.bra_physical[physical_site],
                            self.local_ket_physical[physical_site],
                        ),
                    )
                )
        if mode == "action":
            arguments.extend((self.tensor_shapes[site], ket_labels))
            output_labels = bra_labels
        else:
            output_labels = bra_labels + ket_labels
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

    def hole_matrix(self, site, left, right):
        site = int(site)
        expression = self._hole_expression("matrix", site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_groups[site][1:]
            if physical_site not in self.paired_sites
        ]
        value = expression(left, right, self.mpo_tensors[site], *identities)
        size = int(np.prod(self.tensor_shapes[site]))
        return np.asarray(value).reshape(size, size)

    def _hole_block_expression(self, site):
        """Contract one fixed pair of local physical configurations."""
        key = ("hole", "physical_block", int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        arguments = [
            self.message_shape(site),
            self._message_labels(site),
            self.message_shape(site + 1),
            self._message_labels(site + 1),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                (
                    self.ket_physical[site]
                    if site in self.paired_sites
                    else self.local_ket_physical[site]
                ),
            ),
        ]
        for physical_site in self.physical_groups[site][1:]:
            if physical_site not in self.paired_sites:
                arguments.extend(
                    (
                        (self.dims[physical_site], self.dims[physical_site]),
                        (
                            self.bra_physical[physical_site],
                            self.local_ket_physical[physical_site],
                        ),
                    )
                )
        for physical_site in self.physical_groups[site]:
            ket_label = (
                self.ket_physical[physical_site]
                if physical_site in self.paired_sites
                else self.local_ket_physical[physical_site]
            )
            arguments.extend(
                (
                    (self.dims[physical_site],),
                    (self.bra_physical[physical_site],),
                    (self.dims[physical_site],),
                    (ket_label,),
                )
            )
        output_labels = (
            self.bra_bonds[site],
            self.bra_bonds[site + 1],
            self.ket_bonds[site],
            self.ket_bonds[site + 1],
        )
        expression = contract_expression(
            *arguments,
            output_labels,
            optimize=self.optimize,
        )
        self._expressions[key] = expression
        return expression

    def _validated_physical_configuration(self, site, configuration, *, name):
        configuration = tuple(int(value) for value in configuration)
        physical_sites = self.physical_groups[int(site)]
        if len(configuration) != len(physical_sites):
            raise ValueError(
                f"{name} must contain {len(physical_sites)} physical values."
            )
        if any(
            value < 0 or value >= self.dims[physical_site]
            for value, physical_site in zip(configuration, physical_sites)
        ):
            raise ValueError(f"{name} contains an out-of-range physical value.")
        return configuration

    def hole_block(self, site, left, right, bra_configuration, ket_configuration):
        r"""Return one virtual block of the local effective operator.

        The local tensor is viewed as a matrix with combined virtual index
        ``(left, right)`` and physical-slice index ``(s_k, s_{P_k})``.  This
        method fixes the bra and ket physical slices before contraction, so it
        never constructs the full local matrix.
        """
        site = int(site)
        bra_configuration = self._validated_physical_configuration(
            site, bra_configuration, name="bra_configuration"
        )
        ket_configuration = self._validated_physical_configuration(
            site, ket_configuration, name="ket_configuration"
        )
        expression = self._hole_block_expression(site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_groups[site][1:]
            if physical_site not in self.paired_sites
        ]
        selectors = []
        for physical_site, bra_value, ket_value in zip(
            self.physical_groups[site], bra_configuration, ket_configuration
        ):
            selectors.extend(
                (
                    self._identities[physical_site][bra_value],
                    self._identities[physical_site][ket_value],
                )
            )
        value = expression(
            left,
            right,
            self.mpo_tensors[site],
            *identities,
            *selectors,
        )
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        return np.asarray(value).reshape(virtual_size, virtual_size)

    def hole_blocks(self, site, left, right, configuration_pairs):
        """Return several fixed-physical blocks while reusing local setup."""
        site = int(site)
        expression = self._hole_block_expression(site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_sites[site][1:]
            if physical_site not in self.paired_sites
        ]
        virtual_size = self.virtual_bonds[site] * self.virtual_bonds[site + 1]
        result = {}
        for row, column, bra_configuration, ket_configuration in configuration_pairs:
            bra_configuration = self._validated_physical_configuration(
                site, bra_configuration, name="bra_configuration"
            )
            ket_configuration = self._validated_physical_configuration(
                site, ket_configuration, name="ket_configuration"
            )
            selectors = []
            for physical_site, bra_value, ket_value in zip(
                self.physical_sites[site], bra_configuration, ket_configuration
            ):
                selectors.extend(
                    (
                        self._identities[physical_site][bra_value],
                        self._identities[physical_site][ket_value],
                    )
                )
            value = expression(
                left,
                right,
                self.mpo_tensors[site],
                *identities,
                *selectors,
            )
            result[(int(row), int(column))] = np.asarray(value).reshape(
                virtual_size, virtual_size
            )
        return result

    def hole_action(self, site, left, right, vector):
        site = int(site)
        sequential = self._sequential_hole_action(
            site,
            left,
            right,
            vector,
        )
        if sequential is not None:
            return sequential
        expression = self._hole_expression("action", site)
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_groups[site][1:]
            if physical_site not in self.paired_sites
        ]
        value = expression(
            left,
            right,
            self.mpo_tensors[site],
            *identities,
            np.asarray(vector).reshape(self.tensor_shapes[site]),
        )
        return np.asarray(value).reshape(-1)

    def hole_action_components(self, site, left, right, vector):
        """Yield independently compressible pieces of one local action.

        A compressed MPO has already mixed its automaton channels, so its
        exact local action is the smallest useful component.  Identity-block
        and termwise frontiers override this with finer streamed pieces.
        """
        yield self.hole_action(site, left, right, vector)

    def hole_action_component_count(self, site):
        return 1

    def _enrichment_expression(self, direction, site):
        """Plan a dense-frontier single-layer ``L W A`` or ``A W R``."""
        key = ("enrichment", str(direction), int(site))
        cached = self._expressions.get(key)
        if cached is not None:
            return cached
        site = int(site)
        if direction == "left":
            cut = site
            row_labels = (
                self.bra_bonds[site],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site + 1],
                self.operator_bonds[site + 1],
            ]
        elif direction == "right":
            cut = site + 1
            row_labels = (
                self.bra_bonds[site + 1],
                *(self.bra_physical[index] for index in self.physical_groups[site]),
            )
            column_labels = [
                self.ket_bonds[site],
                self.operator_bonds[site],
            ]
        else:
            raise ValueError("direction must be 'left' or 'right'.")
        local = set(self.physical_groups[site])
        for physical_site in self.frontier_sites[cut]:
            if physical_site in local:
                continue
            column_labels.append(self.bra_physical[physical_site])
            if physical_site in self.paired_sites:
                column_labels.append(self.ket_physical[physical_site])
        arguments = [
            self.message_shape(cut),
            self._message_labels(cut),
            self.tensor_shapes[site],
            self._tensor_labels(site, bra=False),
            self.mpo_tensors[site].shape,
            (
                self.operator_bonds[site],
                self.operator_bonds[site + 1],
                self.bra_physical[site],
                self._ket_label(site),
            ),
        ]
        identities = []
        for physical_site in self.physical_groups[site][1:]:
            identity = self._identities[physical_site]
            identities.append(identity)
            arguments.extend(
                (
                    identity.shape,
                    (
                        self.bra_physical[physical_site],
                        self._ket_label(physical_site),
                    ),
                )
            )
        expression = contract_expression(
            *arguments,
            row_labels + tuple(column_labels),
            optimize=self.optimize,
        )
        row_size = int(
            self.virtual_bonds[site if direction == "left" else site + 1]
            * np.prod(
                [self.dims[index] for index in self.physical_groups[site]],
                dtype=np.int64,
            )
        )
        cached = (expression, tuple(identities), row_size)
        self._expressions[key] = cached
        return cached

    def left_enrichment_components(self, site, left, vector):
        """Yield the dense open ``L W A`` range."""
        site = int(site)
        expression, identities, row_size = self._enrichment_expression("left", site)
        value = expression(
            left,
            np.asarray(vector).reshape(self.tensor_shapes[site]),
            self.mpo_tensors[site],
            *identities,
        )
        yield np.asarray(value).reshape(row_size, -1)

    def right_enrichment_components(self, site, right, vector):
        """Yield the dense open ``A W R`` range."""
        site = int(site)
        expression, identities, row_size = self._enrichment_expression("right", site)
        value = expression(
            right,
            np.asarray(vector).reshape(self.tensor_shapes[site]),
            self.mpo_tensors[site],
            *identities,
        )
        yield np.asarray(value).reshape(row_size, -1)

    def enrichment_component_count(self, site):
        return 1

    def hole_actions(self, site, left, right, vectors):
        """Apply one hole operator to multiple local vectors together."""
        site = int(site)
        vectors = np.asarray(vectors)
        size = int(np.prod(self.tensor_shapes[site]))
        if vectors.ndim != 2 or vectors.shape[1] != size:
            raise ValueError(f"vectors must have shape (batch, {size}).")
        batch = vectors.shape[0]
        key = ("hole", "actions", site, batch)
        expression = self._expressions.get(key)
        if expression is None:
            bra_labels = self._tensor_labels(site, bra=True)
            ket_labels = self._tensor_labels(site, bra=False)
            own_ket_label = (
                self.ket_physical[site]
                if site in self.paired_sites
                else self.local_ket_physical[site]
            )
            arguments = [
                self.message_shape(site),
                self._message_labels(site),
                self.message_shape(site + 1),
                self._message_labels(site + 1),
                self.mpo_tensors[site].shape,
                (
                    self.operator_bonds[site],
                    self.operator_bonds[site + 1],
                    self.bra_physical[site],
                    own_ket_label,
                ),
            ]
            for physical_site in self.physical_groups[site][1:]:
                if physical_site not in self.paired_sites:
                    arguments.extend(
                        (
                            (self.dims[physical_site], self.dims[physical_site]),
                            (
                                self.bra_physical[physical_site],
                                self.local_ket_physical[physical_site],
                            ),
                        )
                    )
            arguments.extend(
                (
                    (batch, *self.tensor_shapes[site]),
                    (self.vector_batch_label, *ket_labels),
                )
            )
            expression = contract_expression(
                *arguments,
                (self.vector_batch_label, *bra_labels),
                optimize=self.optimize,
            )
            self._expressions[key] = expression
        identities = [
            self._identities[physical_site]
            for physical_site in self.physical_groups[site][1:]
            if physical_site not in self.paired_sites
        ]
        value = expression(
            left,
            right,
            self.mpo_tensors[site],
            *identities,
            vectors.reshape(batch, *self.tensor_shapes[site]),
        )
        return np.asarray(value).reshape(batch, size)

    def _sequential_hole_action(self, site, left, right, vector):
        """Apply a fused block MPO through its unfused physical cores."""
        transitions = self._physical_mpo_cores[int(site)]
        if (
            transitions is None
            or self.physical_kernel == "fused"
            or (
                self.physical_kernel == "auto"
                and len(transitions) > 2
            )
        ):
            return None
        site = int(site)
        spectators = self.physical_groups[site][1:]
        if self.frontier_sites[site] or tuple(self.frontier_sites[site + 1]) != tuple(
            spectators
        ):
            return None
        if any(spectator not in self.paired_sites for spectator in spectators):
            return None

        factor_dims = self.physical_factor_dims[site]
        spectator_dims = tuple(self.dims[index] for index in spectators)
        unfused_shape = (
            self.virtual_bonds[site],
            self.virtual_bonds[site + 1],
            *factor_dims,
            *spectator_dims,
        )
        unfused = np.asarray(vector).reshape(unfused_shape)
        result = np.zeros(
            unfused_shape,
            dtype=np.result_type(
                np.asarray(vector).dtype,
                self.mpo_tensors[site].dtype,
            ),
        )
        for left_channels, right_channels, cores in transitions:
            batch_size = len(left_channels)
            core_shapes = tuple(core.shape[1:] for core in cores)
            key = (
                "hole",
                "sequential_transition_action",
                site,
                batch_size,
                core_shapes,
            )
            expression = self._expressions.get(key)
            if expression is None:
                bra_left, bra_right = 0, 1
                ket_left, ket_right = 2, 3
                batch = 4
                rank_labels = tuple(range(10, 10 + len(cores) + 1))
                bra_physical = tuple(range(100, 100 + len(cores)))
                ket_physical = tuple(range(200, 200 + len(cores)))
                bra_spectators = tuple(range(300, 300 + len(spectators)))
                ket_spectators = tuple(range(400, 400 + len(spectators)))
                arguments = [
                    (
                        batch_size,
                        self.virtual_bonds[site],
                        self.virtual_bonds[site],
                    ),
                    (batch, bra_left, ket_left),
                    (
                        batch_size,
                        self.virtual_bonds[site + 1],
                        self.virtual_bonds[site + 1],
                        *(
                            dim
                            for spectator_dim in spectator_dims
                            for dim in (spectator_dim, spectator_dim)
                        ),
                    ),
                    (
                        batch,
                        bra_right,
                        ket_right,
                        *(
                            label
                            for pair in zip(
                                bra_spectators,
                                ket_spectators,
                            )
                            for label in pair
                        ),
                    ),
                ]
                for position, core in enumerate(cores):
                    arguments.extend(
                        (
                            core.shape,
                            (
                                batch,
                                rank_labels[position],
                                rank_labels[position + 1],
                                bra_physical[position],
                                ket_physical[position],
                            ),
                        )
                    )
                arguments.extend(
                    (
                        unfused_shape,
                        (
                            ket_left,
                            ket_right,
                            *ket_physical,
                            *ket_spectators,
                        ),
                    )
                )
                output_labels = (
                    bra_left,
                    bra_right,
                    *bra_physical,
                    *bra_spectators,
                )
                expression = contract_expression(
                    *arguments,
                    output_labels,
                    optimize=self.optimize,
                )
                self._expressions[key] = expression
            result += expression(
                np.moveaxis(
                    np.take(left, left_channels, axis=2),
                    2,
                    0,
                ),
                np.moveaxis(
                    np.take(right, right_channels, axis=2),
                    2,
                    0,
                ),
                *cores,
                unfused,
            )
        return result.reshape(-1)


__all__ = ["MPOFrontier"]
