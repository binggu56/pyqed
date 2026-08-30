"""Reduced-channel qchem Hamiltonian builders for spatial-orbital DMRG."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from pyqed.mps.nonabelian import (
    AutoMPO,
    FullyReducedSpatialOrbitalSite,
    SpatialSpinFreeERIBuilder,
    add_cpp_spatial_spinfree_family_terms,
    add_spatial_one_body_terms,
    physical_leg_from_spatial_orbital,
    sum_mpo_chains,
)
from pyqed.mps.nonabelian.coupling import ordered_two_m_values
from pyqed.mps.nonabelian.mpo import (
    RankCoupledChannelTerm,
    RankCoupledMPO,
    SparseVirtualBlock,
)
from pyqed.mps.su2 import SU2Irrep


_NORMAL_COMPLEMENTARY_OPERATOR_TWO_J = (
    0, 0, 1, 1, 1, 1, 1, 1, 0, 2,
    0, 2, 0, 2, 0, 2, 0, 2, 0, 2,
)
_NORMAL_COMPLEMENTARY_OPERATOR_CHARGE = (
    0, 0, 1, -1, 1, -1, 1, -1, 2, 2,
    -2, -2, 0, 0, 2, 2, -2, -2, 0, 0,
)
_NORMAL_COMPLEMENTARY_FAMILY_NAMES = ("S", "R", "A", "P", "B", "Q")


class _NormalComplementaryPrimitiveOperator:
    """Lightweight physical view of one C++-owned reduced local primitive."""

    def __init__(self, values, operator_id, phys_leg, *, right_parity=False):
        self.values = np.ascontiguousarray(values, dtype=float)
        self.operator_id = int(operator_id)
        self.right_parity = bool(right_parity)
        self.phys_out_leg = phys_leg
        self.phys_in_leg = phys_leg
        self.rank_irrep = SU2Irrep(
            _NORMAL_COMPLEMENTARY_OPERATOR_TWO_J[self.operator_id]
        )
        self._component_cache = {}

    @property
    def dtype(self):
        return self.values.dtype

    @property
    def components(self):
        return ordered_two_m_values(self.rank_irrep)

    def component_block(self, two_m_component, q_out, q_in):
        key = (int(two_m_component), q_out, q_in)
        if key in self._component_cache:
            return self._component_cache[key]
        output_charge = int(q_out.charge)
        input_charge = int(q_in.charge)
        if (
            output_charge < 0
            or output_charge >= self.values.shape[0]
            or input_charge < 0
            or input_charge >= self.values.shape[1]
        ):
            self._component_cache[key] = None
            return None
        component_index = int(two_m_component) + 2
        if component_index < 0 or component_index >= self.values.shape[2]:
            self._component_cache[key] = None
            return None
        coefficient = self.values[
            output_charge,
            input_charge,
            component_index,
        ]
        if self.right_parity and input_charge & 1:
            coefficient = -coefficient
        if coefficient == 0.0:
            self._component_cache[key] = None
            return None
        if (
            self.phys_out_leg.sector_dim(q_out) == 1
            and self.phys_in_leg.sector_dim(q_in) == 1
        ):
            block = np.asarray([[coefficient]], dtype=float)
        else:
            from pyqed.mps.nonabelian.environment import _physical_component_matrix

            block = coefficient * _physical_component_matrix(
                q_out,
                q_in,
                self.rank_irrep,
                int(two_m_component),
                self.rank_irrep.two_j == 0,
            )
        block = np.ascontiguousarray(block, dtype=float)
        self._component_cache[key] = block
        return block


def build_su2_normal_complementary_mpo(
    moving_environment,
    *,
    fully_reduced=False,
    materialize_reduced_terms=True,
):
    """Build lightweight MPO views over the persistent C++ NC route arenas."""

    site_descriptor = (
        FullyReducedSpatialOrbitalSite()
        if fully_reduced
        else None
    )
    phys_leg = physical_leg_from_spatial_orbital(site_descriptor)
    n_sites = int(moving_environment.system_stats["n_sites"])
    factors = []
    for site in range(n_sites):
        plan = moving_environment.normal_complementary_plan(site)
        primitives = moving_environment.normal_complementary_primitives(site)
        plan["_primitive_nonzero"] = np.any(primitives != 0.0, axis=3)
        left_qn = np.asarray(
            plan["left_channel_quantum_numbers"],
            dtype=np.int64,
        )
        right_qn = np.asarray(
            plan["right_channel_quantum_numbers"],
            dtype=np.int64,
        )
        if fully_reduced and not materialize_reduced_terms:
            factors.append(
                RankCoupledMPO(
                    dense_blocks={},
                    phys_out_leg=phys_leg,
                    phys_in_leg=phys_leg,
                    left_channel_irreps=tuple(
                        SU2Irrep(int(two_j)) for two_j in left_qn[:, 1]
                    ),
                    right_channel_irreps=tuple(
                        SU2Irrep(int(two_j)) for two_j in right_qn[:, 1]
                    ),
                    left_channel_charges=tuple(int(x) for x in left_qn[:, 0]),
                    right_channel_charges=tuple(int(x) for x in right_qn[:, 0]),
                    reduced_terms=(),
                    symbolic_transitions=(),
                    normal_complementary_site=site,
                    normal_complementary_plan=plan,
                    normal_complementary_owner=moving_environment,
                    normal_complementary_fully_reduced=True,
                )
            )
            continue
        source = np.asarray(plan["source"], dtype=np.int64)
        target = np.asarray(plan["target"], dtype=np.int64)
        operator_ids = np.asarray(plan["operator"], dtype=np.int64)
        coefficients = np.asarray(plan["coefficient"], dtype=float)
        family_masks = np.asarray(plan["family_mask"], dtype=np.uint64)

        def component_orientation(transition, *, left):
            transition = int(transition)
            left_channel = int(source[transition])
            right_channel = int(target[transition])
            left_two_j = int(left_qn[left_channel, 1])
            right_two_j = int(right_qn[right_channel, 1])
            right_charge = int(right_qn[right_channel, 0])
            operator_charge = _NORMAL_COMPLEMENTARY_OPERATOR_CHARGE[
                int(operator_ids[transition])
            ]
            if (
                left_two_j == 1
                and right_two_j == 2
                and abs(right_charge) == 2
                and operator_charge
            ):
                return -1 if operator_charge > 0 else 1
            if (
                not left
                and left_two_j == 0
                and right_two_j != 0
                and operator_charge
            ):
                return 1 if operator_charge > 0 else -1
            if (
                left
                and right_two_j == 0
                and left_two_j != 0
                and operator_charge
            ):
                return -1 if operator_charge > 0 else 1
            quantum_numbers = left_qn if left else right_qn
            channel = left_channel if left else right_channel
            return -1 if int(quantum_numbers[channel, 0]) < 0 else 1

        reduced_terms = []
        for operator_id in range(len(_NORMAL_COMPLEMENTARY_OPERATOR_TWO_J)):
            operator_transitions = np.flatnonzero(operator_ids == operator_id)
            for right_parity in (False, True):
                parity_selected = operator_transitions[
                    np.asarray(
                        [
                            bool(
                                int(right_qn[int(target[transition]), 0]) & 1
                            )
                            is right_parity
                            for transition in operator_transitions
                        ],
                        dtype=bool,
                    )
                ]
                for left_orientation in (-1, 1):
                    for right_orientation in (-1, 1):
                        oriented_selected = parity_selected[
                            np.asarray(
                                [
                                    component_orientation(
                                        transition,
                                        left=True,
                                    )
                                    == left_orientation
                                    and (
                                        component_orientation(
                                            transition,
                                            left=False,
                                        )
                                    )
                                    == right_orientation
                                    for transition in parity_selected
                                ],
                                dtype=bool,
                            )
                        ]
                        for orient_virtual_coupling in (False, True):
                            coupling_selected = oriented_selected[
                                np.asarray(
                                    [
                                        (
                                            int(
                                                left_qn[
                                                    int(source[transition]),
                                                    0,
                                                ]
                                            )
                                            * int(
                                                right_qn[
                                                    int(target[transition]),
                                                    0,
                                                ]
                                            )
                                            < 0
                                        )
                                        is orient_virtual_coupling
                                        for transition in oriented_selected
                                    ],
                                    dtype=bool,
                                )
                            ]
                            for dual_right_coupling in (False, True):
                                dual_selected = coupling_selected[
                                    np.asarray(
                                        [
                                            (
                                                (
                                                    (
                                                        operator_id
                                                        in (12, 13, 18, 19)
                                                        or (
                                                            operator_id == 0
                                                            and abs(int(
                                                                left_qn[
                                                                    int(source[transition]),
                                                                    0,
                                                                ]
                                                            ))
                                                            != 2
                                                        )
                                                    )
                                                    and int(
                                                        left_qn[
                                                            int(source[transition]),
                                                            0,
                                                        ]
                                                    )
                                                    != 0
                                                    and int(
                                                        left_qn[
                                                            int(source[transition]),
                                                            0,
                                                        ]
                                                    )
                                                    == int(
                                                        right_qn[
                                                            int(target[transition]),
                                                            0,
                                                        ]
                                                    )
                                                )
                                                or (
                                                    int(
                                                        left_qn[
                                                            int(source[transition]),
                                                            1,
                                                        ]
                                                    )
                                                    == 2
                                                    and int(
                                                        right_qn[
                                                            int(target[transition]),
                                                            1,
                                                        ]
                                                    )
                                                    == 1
                                                    and abs(
                                                        int(
                                                            left_qn[
                                                                int(source[transition]),
                                                                0,
                                                            ]
                                                        )
                                                    )
                                                    == 2
                                                )
                                            )
                                            is dual_right_coupling
                                            for transition in coupling_selected
                                        ],
                                        dtype=bool,
                                    )
                                ]
                                for scalar_source_phase in (False, True):
                                    source_phase_selected = dual_selected[
                                        np.asarray(
                                            [
                                                (
                                                    abs(int(
                                                        left_qn[
                                                            int(source[transition]),
                                                            0,
                                                        ]
                                                    ))
                                                    == 2
                                                    and int(
                                                        left_qn[
                                                            int(source[transition]),
                                                            1,
                                                        ]
                                                    )
                                                    == 0
                                                    and int(
                                                        right_qn[
                                                            int(target[transition]),
                                                            1,
                                                        ]
                                                    )
                                                    != 0
                                                    and _NORMAL_COMPLEMENTARY_OPERATOR_TWO_J[
                                                        operator_id
                                                    ]
                                                    != 0
                                                )
                                                is scalar_source_phase
                                                for transition in dual_selected
                                            ],
                                            dtype=bool,
                                        )
                                    ]
                                    for pair_target_phase in (False, True):
                                        selected = source_phase_selected[
                                            np.asarray(
                                                [
                                                    (
                                                        abs(int(
                                                            right_qn[
                                                                int(target[transition]),
                                                                0,
                                                            ]
                                                        ))
                                                        == 2
                                                        and int(
                                                            left_qn[
                                                                int(source[transition]),
                                                                1,
                                                            ]
                                                        )
                                                        != 0
                                                        and _NORMAL_COMPLEMENTARY_OPERATOR_TWO_J[
                                                            operator_id
                                                        ]
                                                        != 0
                                                    )
                                                    is pair_target_phase
                                                    for transition in source_phase_selected
                                                ],
                                                dtype=bool,
                                            )
                                        ]
                                        if selected.size == 0:
                                            continue
                                        routes = {}
                                        route_transitions = {}
                                        for transition in selected:
                                            key = (
                                                int(source[transition]),
                                                int(target[transition]),
                                            )
                                            routes[key] = routes.get(key, 0.0) + float(
                                                coefficients[transition]
                                            )
                                            route_transitions.setdefault(key, []).append(
                                                int(transition)
                                            )
                                        virtual_block = SparseVirtualBlock.from_entries(
                                            (
                                                int(plan["left_channels"]),
                                                int(plan["right_channels"]),
                                            ),
                                            routes,
                                            dtype=float,
                                            retain_zeros=True,
                                        )
                                        term = RankCoupledChannelTerm(
                                            reduced_operator=_NormalComplementaryPrimitiveOperator(
                                                primitives[:, :, operator_id, :],
                                                operator_id,
                                                phys_leg,
                                                right_parity=right_parity,
                                            ),
                                            visible_virtual_block=virtual_block,
                                            use_cg_coupling=True,
                                            left_component_orientation=left_orientation,
                                            right_component_orientation=right_orientation,
                                            orient_virtual_coupling=orient_virtual_coupling,
                                            dual_right_coupling=dual_right_coupling,
                                            phase_from_charged_scalar_source=(
                                                scalar_source_phase
                                            ),
                                            phase_to_charged_pair_target=(
                                                pair_target_phase
                                            ),
                                        )
                                        object.__setattr__(
                                            term,
                                            "_normal_complementary_route_transitions",
                                            tuple(
                                                tuple(route_transitions[(int(row), int(col))])
                                                for row, col in zip(
                                                    virtual_block.rows,
                                                    virtual_block.cols,
                                                )
                                            ),
                                        )
                                        reduced_terms.append(term)
        factors.append(
            RankCoupledMPO(
                dense_blocks={},
                phys_out_leg=phys_leg,
                phys_in_leg=phys_leg,
                left_channel_irreps=tuple(
                    SU2Irrep(int(two_j)) for two_j in left_qn[:, 1]
                ),
                right_channel_irreps=tuple(
                    SU2Irrep(int(two_j)) for two_j in right_qn[:, 1]
                ),
                left_channel_charges=tuple(int(x) for x in left_qn[:, 0]),
                right_channel_charges=tuple(int(x) for x in right_qn[:, 0]),
                reduced_terms=tuple(reduced_terms),
                symbolic_transitions=(),
                normal_complementary_site=site,
                normal_complementary_plan=plan,
                normal_complementary_owner=moving_environment,
                normal_complementary_fully_reduced=bool(fully_reduced),
            )
        )
    return factors


def refresh_su2_normal_complementary_mpo(moving_environment, factors):
    """Refresh all NC integral coefficients without replacing MPO core objects."""

    factors = tuple(factors)
    n_sites = int(moving_environment.system_stats["n_sites"])
    if len(factors) != n_sites:
        raise ValueError("The NC MPO refresh requires one factor per active orbital.")
    topology_keys = (
        "source",
        "target",
        "operator",
        "first_index",
        "second_index",
        "family_mask",
    )
    numeric_plan_keys = (
        "coefficient",
        "component_transition",
        "component_source",
        "component_target",
        "component_local_two_m",
        "component_coefficient",
        "family_transition_counts",
    )
    for site, factor in enumerate(factors):
        if (
            getattr(factor, "normal_complementary_owner", None)
            is not moving_environment
            or int(getattr(factor, "normal_complementary_site", -1)) != site
        ):
            raise ValueError("The installed MPO is not owned by this NC environment.")
        installed_plan = factor.normal_complementary_plan
        refreshed_plan = moving_environment.normal_complementary_plan(site)
        for key in topology_keys:
            if not np.array_equal(installed_plan[key], refreshed_plan[key]):
                raise ValueError("An NC numeric refresh changed the MPO transition topology.")

        coefficients = np.asarray(refreshed_plan["coefficient"], dtype=float)
        primitives = moving_environment.normal_complementary_primitives(site)
        primitive_nonzero = np.any(primitives != 0.0, axis=3)
        if not np.array_equal(
            installed_plan.get("_primitive_nonzero"),
            primitive_nonzero,
        ):
            installed_plan.pop("_boundary_action_cache", None)
        installed_plan["_primitive_nonzero"] = primitive_nonzero
        for key in numeric_plan_keys:
            value = refreshed_plan[key]
            installed_plan[key] = value.copy() if hasattr(value, "copy") else value

        for term in factor.reduced_terms:
            route_transitions = getattr(
                term,
                "_normal_complementary_route_transitions",
                None,
            )
            if route_transitions is None:
                raise ValueError("An NC term is missing its numeric refresh map.")
            block = term.visible_virtual_block
            if len(route_transitions) != block.values.size:
                raise ValueError("An NC term refresh map has an invalid size.")
            block.values[...] = np.asarray(
                [
                    np.sum(coefficients[np.asarray(indices, dtype=np.int64)])
                    for indices in route_transitions
                ],
                dtype=block.values.dtype,
            )
            operator = term.reduced_operator
            operator.values[...] = primitives[:, :, operator.operator_id, :]
            operator._component_cache.clear()

        factor._reduced_block_cache.clear()
        factor._block_cache.clear()
        object.__setattr__(factor, "_reduced_action_cache", None)
    if not moving_environment.refresh_contextual_cores(factors):
        for factor in factors:
            contextual_cache = getattr(
                factor,
                "_contextual_angular_core_cache",
                None,
            )
            if contextual_cache is not None:
                contextual_cache.clear()
    return factors


class ComplementaryEntries(Mapping):
    """Read-only sparse-family view backed by contiguous C++ exports."""

    def __init__(self, indices, values, *, moving_environment=None, family_name=None):
        self.indices = np.ascontiguousarray(indices, dtype=np.int64)
        self.values = np.ascontiguousarray(values, dtype=float)
        self.moving_environment = moving_environment
        self.family_name = None if family_name is None else str(family_name)
        if self.indices.ndim != 2 or self.values.shape != (self.indices.shape[0],):
            raise ValueError("C++ complementary-family arrays are inconsistent.")

    def __len__(self):
        return int(self.values.size)

    def __iter__(self):
        for row in self.indices:
            yield tuple(int(index) for index in row)

    def __getitem__(self, key):
        key = np.asarray(tuple(key), dtype=np.int64)
        if key.shape != (self.indices.shape[1],):
            raise KeyError(tuple(key))
        matches = np.flatnonzero(np.all(self.indices == key, axis=1))
        if matches.size == 0:
            raise KeyError(tuple(int(index) for index in key))
        return complex(self.values[int(matches[0])])

    def items(self):
        for row, value in zip(self.indices, self.values):
            yield tuple(int(index) for index in row), complex(value)

    def partition_counts(self, side, bond):
        if self.moving_environment is None or self.family_name is None:
            return None
        return self.moving_environment.family_partition_counts(
            self.family_name,
            side,
            int(bond),
        )

    def __eq__(self, other):
        if isinstance(other, ComplementaryEntries):
            return bool(
                np.array_equal(self.indices, other.indices)
                and np.array_equal(self.values, other.values)
            )
        if isinstance(other, Mapping):
            return dict(self.items()) == dict(other.items())
        return NotImplemented


@dataclass(frozen=True)
class ComplementaryOperatorFamily:
    """
    Sparse block2-style complementary operator family.

    The entries store integral-side coefficients and channel labels, not the
    renormalized block tensors themselves.  Sweep code can use these families
    to build persistent complementary renormalized operators without rewalking
    the raw four-index ERI tensor.

    :param name: Family label, for example ``"S"``, ``"R"``, ``"A"``,
        ``"P"``, ``"B"``, or ``"Q"``.
    :param rank: Tensor rank or operator arity represented by the family.
    :param entries: Sparse mapping from orbital-index tuples to coefficients.
    :param description: Short description of the represented operator channel.
    """

    name: str
    rank: int
    entries: Mapping
    description: str = ""

    @property
    def n_terms(self):
        """
        Return the number of stored sparse entries.

        :returns: Sparse entry count.
        """

        return int(len(self.entries))

    @property
    def index_shape(self):
        """
        Return the index tuple lengths represented by this family.

        :returns: Sorted tuple of key lengths.
        """

        return tuple(sorted({len(tuple(key)) for key in self.entries}))

    def as_metadata(self):
        """
        Return lightweight diagnostics for this family.

        :returns: Dictionary suitable for Hamiltonian ``info`` metadata.
        """

        return {
            "name": str(self.name),
            "rank": int(self.rank),
            "n_terms": int(self.n_terms),
            "index_shape": self.index_shape,
            "description": str(self.description),
        }


@dataclass(frozen=True)
class SpatialComplementaryOperatorFamilies:
    """
    Sparse spatial-orbital complementary operator families.

    These are the chemistry-side block2 channels.  ``S``/``A``/``B`` describe
    structural one-particle, pair, and particle-hole operator channels, while
    ``R``/``P``/``Q`` carry screened integral coefficients for the one-body,
    two-generator, and exchange/correction complementary contractions.

    :param families: Mapping from family name to
        :class:`ComplementaryOperatorFamily`.
    :param n_sites: Number of spatial active orbitals.
    :param cutoff: Screening threshold used to create sparse entries.
    :param include_half: Whether ERI entries include the conventional
        two-electron ``1/2`` prefactor.
    :param prefer_direct_orthonormal_projection: Prefer the experimental
        component-direct factorized projection in the local Davidson kernel.
        The default stays ``False`` because transformed component tables are
        currently faster for small and medium chemistry benchmarks.
    :param prefer_direct_component_transform: Prefer direct ``X^H L R X``
        transformed-kernel construction.  This is experimental and defaults to
        ``False`` because the parent-block transform path is faster on H6.
    :param prefer_recursive_operator_matvec: Prefer the matrix-free recursive
        complementary-operator matvec path.  This avoids building transformed
        local Hamiltonian kernels entirely, but is currently slower than the
        compiled parent-block backend on the default SU(2) qchem benchmarks,
        so it remains opt-in.
    :param prefer_complementary_payload_tensor_matvec: Prefer the experimental
        payload/family tensor matvec path.  This mirrors the complementary
        family ownership used by block2, but is currently slower than the
        compiled parent-block path for small and medium qchem examples, so it
        remains opt-in.
    """

    families: dict
    n_sites: int
    cutoff: float
    include_half: bool = True
    prefer_direct_orthonormal_projection: bool = False
    prefer_direct_component_transform: bool = False
    prefer_recursive_operator_matvec: bool = False
    prefer_complementary_payload_tensor_matvec: bool = False
    prefer_precontracted_family_environment: bool = True
    boundary_table_max_dim: int = 32
    debug_complementary_action_check: bool = False
    debug_complementary_action_check_tol: float = 1.0e-10
    debug_complementary_action_check_limit: int = 32
    exact_component_compression_policy: str = "auto"
    exact_component_compression_validate: bool = True
    exact_component_compression_validation_vectors: int = 1
    exact_component_compression_min_reduction: int = 1
    exact_component_compression_max_group_size: int = 64
    enable_cpp_boundary_r: bool = False
    validate_cpp_boundary_r: bool = True
    enable_cpp_boundary_p: bool = True
    validate_cpp_boundary_p: bool = False
    cpp_boundary_p_validation_policy: str = "off"
    direct_operator_batch_min_entries: int = 2

    def __getitem__(self, name):
        """Return a named family such as ``"P"`` or ``"Q"``."""

        return self.families[str(name)]

    def get(self, name, default=None):
        """Return a named family or ``default`` when absent."""

        return self.families.get(str(name), default)

    @property
    def names(self):
        """
        Return available family names.

        :returns: Tuple of family labels in insertion order.
        """

        return tuple(self.families)

    @property
    def n_terms(self):
        """
        Return the total sparse entry count across all families.

        :returns: Total sparse entry count.
        """

        return int(sum(family.n_terms for family in self.families.values()))

    def as_metadata(self):
        """
        Return compact diagnostics for Hamiltonian metadata.

        :returns: Dictionary describing family availability and term counts.
        """

        return {
            "enabled": True,
            "n_sites": int(self.n_sites),
            "cutoff": float(self.cutoff),
            "include_half": bool(self.include_half),
            "prefer_direct_orthonormal_projection": bool(
                self.prefer_direct_orthonormal_projection
            ),
            "prefer_direct_component_transform": bool(
                self.prefer_direct_component_transform
            ),
            "prefer_recursive_operator_matvec": bool(
                self.prefer_recursive_operator_matvec
            ),
            "prefer_complementary_payload_tensor_matvec": bool(
                self.prefer_complementary_payload_tensor_matvec
            ),
            "prefer_precontracted_family_environment": bool(
                self.prefer_precontracted_family_environment
            ),
            "boundary_table_max_dim": int(self.boundary_table_max_dim),
            "debug_complementary_action_check": bool(
                self.debug_complementary_action_check
            ),
            "debug_complementary_action_check_tol": float(
                self.debug_complementary_action_check_tol
            ),
            "debug_complementary_action_check_limit": int(
                self.debug_complementary_action_check_limit
            ),
            "exact_component_compression_policy": str(
                self.exact_component_compression_policy
            ),
            "exact_component_compression_validate": bool(
                self.exact_component_compression_validate
            ),
            "exact_component_compression_validation_vectors": int(
                self.exact_component_compression_validation_vectors
            ),
            "exact_component_compression_min_reduction": int(
                self.exact_component_compression_min_reduction
            ),
            "exact_component_compression_max_group_size": int(
                self.exact_component_compression_max_group_size
            ),
            "enable_cpp_boundary_r": bool(self.enable_cpp_boundary_r),
            "validate_cpp_boundary_r": bool(self.validate_cpp_boundary_r),
            "enable_cpp_boundary_p": bool(self.enable_cpp_boundary_p),
            "validate_cpp_boundary_p": bool(self.validate_cpp_boundary_p),
            "cpp_boundary_p_validation_policy": str(
                self.cpp_boundary_p_validation_policy
            ),
            "direct_operator_batch_min_entries": int(
                self.direct_operator_batch_min_entries
            ),
            "families": {
                name: family.as_metadata()
                for name, family in self.families.items()
            },
            "family_names": self.names,
            "n_terms": int(self.n_terms),
        }


_COMPLEMENTARY_FAMILY_DESCRIPTIONS = {
    "S": "single-orbital spinor source channels",
    "R": "effective one-body complementary coefficients",
    "A": "pair/scalar-generator structural channels",
    "P": "two-generator ERI complementary coefficients",
    "B": "particle-hole scalar-generator structural channels",
    "Q": "delta-contracted one-body correction complementary coefficients",
}


def _build_su2_moving_environment(
    h_spatial,
    eri_spatial,
    *,
    n_elec,
    spin,
    ecore,
    orb_sym,
    cutoff,
):
    """Construct the persistent handwritten C++ SU(2) system owner."""

    if eri_spatial is None:
        n_sites = int(np.asarray(h_spatial).shape[0])
        eri_spatial = np.zeros((n_sites,) * 4, dtype=float)
    if np.iscomplexobj(h_spatial) or np.iscomplexobj(eri_spatial):
        return None
    try:
        from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    except ImportError:
        return None
    return SU2MovingEnvironment(
        np.ascontiguousarray(h_spatial, dtype=float),
        np.ascontiguousarray(eri_spatial, dtype=float),
        int(n_elec),
        two_s=int(spin),
        ecore=float(ecore),
        orb_sym=(
            np.zeros(np.asarray(h_spatial).shape[0], dtype=np.int64)
            if orb_sym is None
            else np.asarray(orb_sym, dtype=np.int64)
        ),
        cutoff=float(cutoff),
        include_half=True,
    )


def _complementary_families_from_cpp(moving_environment):
    """Materialize lightweight Python metadata from C++ family storage."""

    families = {}
    for name in ("S", "R", "A", "P", "B", "Q"):
        payload = moving_environment.family(name)
        indices = np.asarray(payload["indices"], dtype=np.int64)
        values = np.asarray(payload["values"], dtype=float)
        families[name] = ComplementaryOperatorFamily(
            name=name,
            rank=int(payload["rank"]),
            entries=ComplementaryEntries(
                indices,
                values,
                moving_environment=moving_environment,
                family_name=name,
            ),
            description=_COMPLEMENTARY_FAMILY_DESCRIPTIONS[name],
        )
    return families


def build_spatial_complementary_operator_families(
    h1e,
    eri=None,
    *,
    cutoff=1.0e-10,
    include_half=True,
    prefer_complementary_payload_tensor_matvec=False,
    prefer_precontracted_family_environment=True,
    boundary_table_max_dim=32,
    debug_complementary_action_check=False,
    debug_complementary_action_check_tol=1.0e-10,
    debug_complementary_action_check_limit=32,
    exact_component_compression_policy="auto",
    exact_component_compression_validate=True,
    exact_component_compression_validation_vectors=1,
    exact_component_compression_min_reduction=1,
    exact_component_compression_max_group_size=64,
    enable_cpp_boundary_r=False,
    validate_cpp_boundary_r=True,
    enable_cpp_boundary_p=True,
    validate_cpp_boundary_p=False,
    cpp_boundary_p_validation_policy="off",
    direct_operator_batch_min_entries=2,
    moving_environment=None,
):
    """
    Build sparse block2-style ``S/R/A/P/B/Q`` families from active integrals.

    The present representation is spin-free and spatial-orbital based.  It
    exposes the same complementary-operator ownership boundary used by block2:
    raw one- and two-electron integrals are grouped into named operator
    families before the moving environment constructs renormalized left/right
    operator stacks.

    ``P`` stores the scalar-generator coupling coefficients for
    ``E_pq E_rs``.  ``Q`` stores the corresponding ``-delta_qr E_ps``
    correction channels.  ``R`` stores the effective one-body coefficients
    ``h_ps + sum_q Q_psq``.  ``S``, ``A``, and ``B`` are structural channel
    families used by later renormalized-operator construction.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved array.
    :param eri: Optional restricted spin-resolved ERI tensor.
    :param cutoff: Absolute screening threshold.
    :param include_half: Whether to apply the conventional two-electron
        prefactor ``1/2`` to ERI coefficients.
    :param prefer_complementary_payload_tensor_matvec: Whether downstream
        sweep kernels should prefer the family/payload boundary matvec over
        the generic residual split.
    :param prefer_precontracted_family_environment: Whether downstream sweep
        kernels should precontract named family environments before the local
        center action.  This is useful for profiling but not the default on
        small and medium Abelian benchmarks.
    :param boundary_table_max_dim: Maximum local layout dimension for the
        long-term family-operator boundary action table.
    :param debug_complementary_action_check: Enable live local-action checks
        against the exact MPO action in sweep kernels.
    :returns: :class:`SpatialComplementaryOperatorFamilies`.
    """

    h_spatial = _restricted_spatial_h1e(h1e)
    n_sites = int(h_spatial.shape[0])
    cutoff = float(cutoff)
    if moving_environment is not None:
        families = _complementary_families_from_cpp(moving_environment)
        return SpatialComplementaryOperatorFamilies(
            families=families,
            n_sites=n_sites,
            cutoff=cutoff,
            include_half=bool(include_half),
            prefer_complementary_payload_tensor_matvec=bool(
                prefer_complementary_payload_tensor_matvec
            ),
            prefer_precontracted_family_environment=bool(
                prefer_precontracted_family_environment
            ),
            boundary_table_max_dim=int(boundary_table_max_dim),
            debug_complementary_action_check=bool(
                debug_complementary_action_check
            ),
            debug_complementary_action_check_tol=float(
                debug_complementary_action_check_tol
            ),
            debug_complementary_action_check_limit=int(
                debug_complementary_action_check_limit
            ),
            exact_component_compression_policy=str(
                exact_component_compression_policy
            ),
            exact_component_compression_validate=bool(
                exact_component_compression_validate
            ),
            exact_component_compression_validation_vectors=int(
                exact_component_compression_validation_vectors
            ),
            exact_component_compression_min_reduction=int(
                exact_component_compression_min_reduction
            ),
            exact_component_compression_max_group_size=int(
                exact_component_compression_max_group_size
            ),
            enable_cpp_boundary_r=bool(enable_cpp_boundary_r),
            validate_cpp_boundary_r=bool(validate_cpp_boundary_r),
            enable_cpp_boundary_p=bool(enable_cpp_boundary_p),
            validate_cpp_boundary_p=bool(validate_cpp_boundary_p),
            cpp_boundary_p_validation_policy=str(
                cpp_boundary_p_validation_policy
            ),
            direct_operator_batch_min_entries=int(
                direct_operator_batch_min_entries
            ),
        )
    h_entries = {
        (int(p), int(q)): complex(h_spatial[p, q])
        for p, q in np.argwhere(np.abs(h_spatial) > cutoff)
    }
    p_entries = {}
    q_entries = {}
    if eri is not None:
        eri_arr = np.asarray(eri)
        if eri_arr.ndim == 6:
            if eri_arr.shape[0] < 1 or eri_arr.shape[1] < 1:
                raise ValueError("eri must have shape (spin, spin, n, n, n, n).")
            eri_spatial = eri_arr[0, 0]
        elif eri_arr.ndim == 4:
            eri_spatial = eri_arr
        else:
            raise ValueError("eri must be a spatial ERI tensor or spin-resolved ERI tensor.")
        if eri_spatial.shape != (n_sites, n_sites, n_sites, n_sites):
            raise ValueError(
                f"eri spatial shape {eri_spatial.shape!r} does not match h1e dimension {n_sites}."
            )
        values = 0.5 * eri_spatial if include_half else eri_spatial
        for p, q, r, s in np.argwhere(np.abs(values) > cutoff):
            val = complex(values[p, q, r, s])
            p_entries[(int(p), int(q), int(r), int(s))] = val
            if int(q) == int(r):
                key = (int(p), int(s), int(q))
                q_entries[key] = q_entries.get(key, 0.0) - val
                if abs(q_entries[key]) <= cutoff:
                    q_entries.pop(key, None)

    r_entries = dict(h_entries)
    for (p, s, _q), val in q_entries.items():
        key = (int(p), int(s))
        r_entries[key] = r_entries.get(key, 0.0) + complex(val)
        if abs(r_entries[key]) <= cutoff:
            r_entries.pop(key, None)

    active_orbitals = set()
    for entries in (h_entries, p_entries, q_entries, r_entries):
        for key in entries:
            active_orbitals.update(int(idx) for idx in key)
    if not active_orbitals:
        active_orbitals = set(range(n_sites))
    active_orbitals = tuple(sorted(active_orbitals))

    generator_pairs = {
        (int(p), int(q)): 1.0
        for p, q in set(h_entries) | {(p, q) for p, q, _r, _s in p_entries} | set(r_entries)
    }
    pair_channels = {}
    for p, q, r, s in p_entries:
        pair_channels[(int(p), int(q))] = 1.0
        pair_channels[(int(r), int(s))] = 1.0
    families = {
        "S": ComplementaryOperatorFamily(
            name="S",
            rank=1,
            entries={(int(p),): 1.0 for p in active_orbitals},
            description="single-orbital spinor source channels",
        ),
        "R": ComplementaryOperatorFamily(
            name="R",
            rank=2,
            entries=r_entries,
            description="effective one-body complementary coefficients",
        ),
        "A": ComplementaryOperatorFamily(
            name="A",
            rank=2,
            entries=pair_channels,
            description="pair/scalar-generator structural channels",
        ),
        "P": ComplementaryOperatorFamily(
            name="P",
            rank=4,
            entries=p_entries,
            description="two-generator ERI complementary coefficients",
        ),
        "B": ComplementaryOperatorFamily(
            name="B",
            rank=2,
            entries=generator_pairs,
            description="particle-hole scalar-generator structural channels",
        ),
        "Q": ComplementaryOperatorFamily(
            name="Q",
            rank=3,
            entries=q_entries,
            description="delta-contracted one-body correction complementary coefficients",
        ),
    }
    return SpatialComplementaryOperatorFamilies(
        families=families,
        n_sites=n_sites,
        cutoff=cutoff,
        include_half=bool(include_half),
        prefer_complementary_payload_tensor_matvec=bool(
            prefer_complementary_payload_tensor_matvec
        ),
        prefer_precontracted_family_environment=bool(
            prefer_precontracted_family_environment
        ),
        boundary_table_max_dim=int(boundary_table_max_dim),
        debug_complementary_action_check=bool(debug_complementary_action_check),
        debug_complementary_action_check_tol=float(debug_complementary_action_check_tol),
        debug_complementary_action_check_limit=int(debug_complementary_action_check_limit),
        exact_component_compression_policy=str(exact_component_compression_policy),
        exact_component_compression_validate=bool(exact_component_compression_validate),
        exact_component_compression_validation_vectors=int(
            exact_component_compression_validation_vectors
        ),
        exact_component_compression_min_reduction=int(
            exact_component_compression_min_reduction
        ),
        exact_component_compression_max_group_size=int(
            exact_component_compression_max_group_size
        ),
        enable_cpp_boundary_r=bool(enable_cpp_boundary_r),
        validate_cpp_boundary_r=bool(validate_cpp_boundary_r),
        enable_cpp_boundary_p=bool(enable_cpp_boundary_p),
        validate_cpp_boundary_p=bool(validate_cpp_boundary_p),
        cpp_boundary_p_validation_policy=str(cpp_boundary_p_validation_policy),
        direct_operator_batch_min_entries=int(direct_operator_batch_min_entries),
    )


@dataclass(frozen=True)
class ReducedSpatialHamiltonian:
    """
    Block-style reduced-channel qchem Hamiltonian.

    The object mirrors the system/MPO boundary used by block2: chemistry code
    initializes a quantum-chemistry system once, then DMRG consumes the
    already-built MPO together with target quantum numbers and scalar core
    energy.  Existing callers can keep using ``factors`` and ``info``.

    :param factors: Rank-coupled MPO cores for the active-space Hamiltonian.
    :param info: Assembly metadata and diagnostics.
    :param n_sites: Number of spatial active orbitals.
    :param nelec: Target active-electron count, if known.
    :param spin: Target doubled spin ``2S``.
    :param ecore: Scalar core energy added outside the active MPO.
    :param orb_sym: Optional orbital symmetry labels.
    :param symmetry: Symmetry backend label.
    :param complementary_operators: Optional block2-style complementary
        operator families derived from active integrals.
    """

    factors: list
    info: dict
    n_sites: int
    nelec: int | None = None
    spin: int = 0
    ecore: float = 0.0
    orb_sym: tuple | None = None
    symmetry: str = "su2"
    complementary_operators: SpatialComplementaryOperatorFamilies | None = None
    moving_environment: object | None = None
    h1e: object | None = None
    eri: object | None = None
    cutoff: float = 1.0e-10
    fully_reduced: bool = False

    @property
    def mpo(self):
        """
        Return the active-space MPO cores.

        :returns: Rank-coupled MPO factor list.
        """

        return self.factors

    @property
    def ncas(self):
        """
        Return the number of active spatial orbitals.

        :returns: Active-space site count.
        """

        return int(self.n_sites)

    def initialize_system_kwargs(self):
        """
        Return block2-style system initialization metadata.

        :returns: Dictionary with ``n_sites``, ``n_elec``, ``spin``, and
            ``orb_sym`` fields.
        """

        return {
            "n_sites": int(self.n_sites),
            "n_elec": None if self.nelec is None else int(self.nelec),
            "spin": int(self.spin),
            "orb_sym": None if self.orb_sym is None else tuple(self.orb_sym),
        }

    def materialize_transition_factors(self):
        """Build the exact reduced MPO view used for arbitrary bra/ket contractions.

        Production SU(2)-DMRG contracts the compact normal/complementary
        Hamiltonian through ``moving_environment`` and never calls this method.
        Optimizers such as LETTA need arbitrary transition matrix elements, so
        they use this explicit Wigner--Eckart carrier reconstructed from the
        canonical spatial integrals.
        """
        if self.info.get("python_reduced_terms_materialized", True):
            return list(self.factors)
        if self.h1e is None:
            raise ValueError(
                "Native SU(2) Hamiltonian has no integral recipe for transition contractions."
            )
        eri = None
        if self.eri is not None:
            eri = np.asarray(self.eri)[None, None, ...]
        explicit = SpatialReducedHamiltonianBuilder(
            np.asarray(self.h1e),
            eri=eri,
            cutoff=float(self.cutoff),
            fully_reduced=bool(self.fully_reduced),
            # Transition factors are operator data, independent of the target
            # state. Keep this rebuild on the explicit reduced compiler rather
            # than creating a second sector-specific native C++ owner.
            nelec=None,
            spin=0,
            ecore=0.0,
            orb_sym=self.orb_sym,
        ).build()
        return list(explicit.factors)

    def with_info(self, **updates):
        """
        Return a copy with updated metadata.

        :param updates: Metadata values merged into ``info``.
        :returns: Updated :class:`ReducedSpatialHamiltonian`.
        """

        info = dict(self.info)
        info.update(updates)
        return ReducedSpatialHamiltonian(
            factors=list(self.factors),
            info=info,
            n_sites=self.n_sites,
            nelec=self.nelec,
            spin=self.spin,
            ecore=self.ecore,
            orb_sym=self.orb_sym,
            symmetry=self.symmetry,
            complementary_operators=self.complementary_operators,
            moving_environment=self.moving_environment,
            h1e=self.h1e,
            eri=self.eri,
            cutoff=self.cutoff,
            fully_reduced=self.fully_reduced,
        )


@dataclass(frozen=True)
class SpatialReducedHamiltonianBuilder:
    """
    Build qchem active-space Hamiltonians as reduced spatial MPO chains.

    This class is the chemistry-to-MPO ownership boundary.  It normalizes the
    restricted active-space integrals and creates the persistent C++ SU(2)
    system.  Canonical production builds derive their temporary reference
    carrier from the C++ ``P/Q`` families; the Python
    :class:`SpatialSpinFreeERIBuilder` remains the fallback/reference compiler.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved array
        with shape ``(spin, n, n)``.
    :param eri: Optional spin-resolved ERI tensor with shape
        ``(spin, spin, n, n, n, n)``.
    :param cutoff: Absolute screening threshold used by the MPO builders.
    """

    h1e: object
    eri: object | None = None
    cutoff: float = 1.0e-10
    fully_reduced: bool = False
    nelec: int | None = None
    spin: int = 0
    ecore: float = 0.0
    orb_sym: tuple | None = None
    reuse: ReducedSpatialHamiltonian | None = None

    @property
    def h_spatial(self):
        """
        Return the restricted spatial one-electron active-space matrix.

        :returns: Square ``(n, n)`` one-electron matrix.
        """
        return _restricted_spatial_h1e(self.h1e)

    @property
    def eri_spatial(self):
        """
        Return the restricted spatial ERI tensor when available.

        :returns: ``None`` or the ``(n, n, n, n)`` ERI block.
        """
        if self.eri is None:
            return None
        eri_arr = np.asarray(self.eri)
        if eri_arr.ndim != 6 or eri_arr.shape[0] < 1 or eri_arr.shape[1] < 1:
            raise ValueError("eri must have shape (spin, spin, n, n, n, n).")
        return eri_arr[0, 0]

    def build(self):
        """
        Build the reduced qchem Hamiltonian MPO.

        :returns: :class:`ReducedSpatialHamiltonian` carrying MPO factors and
            assembly metadata.
        """
        h_spatial = self.h_spatial
        if h_spatial.shape[0] < 2:
            raise NotImplementedError("Reduced spatial Hamiltonian MPO currently requires at least two active orbitals.")
        eri_spatial = self.eri_spatial
        effective_eri = (
            np.zeros((h_spatial.shape[0],) * 4, dtype=float)
            if eri_spatial is None
            else eri_spatial
        )
        moving_environment = None
        reused_factors = None
        reused_runtime = False
        reusable = self.reuse
        if (
            reusable is not None
            and reusable.moving_environment is not None
            and int(reusable.n_sites) == int(h_spatial.shape[0])
            and reusable.nelec == self.nelec
            and int(reusable.spin) == int(self.spin)
            and reusable.orb_sym == self.orb_sym
            and bool(
                reusable.info.get("spatial_site_basis")
                == "fully_reduced_su2"
            ) == bool(self.fully_reduced)
        ):
            candidate = reusable.moving_environment
            try:
                candidate.update_integrals(
                    np.ascontiguousarray(h_spatial, dtype=float),
                    np.ascontiguousarray(effective_eri, dtype=float),
                    float(self.ecore),
                )
                reused_factors = list(reusable.factors)
                refresh_su2_normal_complementary_mpo(candidate, reused_factors)
                moving_environment = candidate
                reused_runtime = True
            except (AttributeError, TypeError, ValueError):
                moving_environment = None
                reused_factors = None
        if moving_environment is None:
            moving_environment = _build_su2_moving_environment(
                h_spatial,
                eri_spatial,
                n_elec=0 if self.nelec is None else self.nelec,
                spin=self.spin,
                ecore=self.ecore,
                orb_sym=self.orb_sym,
                cutoff=self.cutoff,
            )
        complementary = build_spatial_complementary_operator_families(
            h_spatial,
            eri_spatial,
            cutoff=self.cutoff,
            include_half=True,
            moving_environment=moving_environment,
        )

        site_descriptor = FullyReducedSpatialOrbitalSite() if self.fully_reduced else None
        site_legs = [
            physical_leg_from_spatial_orbital(site_descriptor)
            for _ in range(h_spatial.shape[0])
        ]
        two_body_info = {
            "total_terms": 0,
            "we_product_terms": 0,
            "scalar_product_terms": 0,
            "one_body_correction_terms": 0,
        }
        production_normal_complementary = bool(
            moving_environment is not None
            and self.nelec is not None
        )
        if production_normal_complementary:
            factors = (
                reused_factors
                if reused_factors is not None
                else build_su2_normal_complementary_mpo(
                    moving_environment,
                    fully_reduced=self.fully_reduced,
                    materialize_reduced_terms=not self.fully_reduced,
                )
            )
            has_integrals = bool(
                eri_spatial is not None
                and np.any(np.abs(eri_spatial) > self.cutoff)
            )
            if has_integrals:
                family_actions = sum(
                    sum(
                        moving_environment.normal_complementary_plan(site)[
                            "family_transition_counts"
                        ].values()
                    )
                    for site in range(int(h_spatial.shape[0]))
                )
                two_body_info.update(
                    total_terms=int(family_actions),
                    scalar_product_terms=int(family_actions),
                )
            two_body_builder = "SU2System[NC]" if has_integrals else "none"
        else:
            autompo = AutoMPO(site_legs)
            add_spatial_one_body_terms(autompo, h_spatial, cutoff=self.cutoff)
            one_body_factors = autompo.build()
            two_body_factors = []
            if eri_spatial is not None and np.any(np.abs(eri_spatial) > self.cutoff):
                if moving_environment is not None and not self.fully_reduced:
                    eri_autompo = AutoMPO(site_legs)
                    two_body_info = add_cpp_spatial_spinfree_family_terms(
                        eri_autompo,
                        complementary,
                        cutoff=self.cutoff,
                        return_info=True,
                    )
                    two_body_factors = (
                        eri_autompo.build()
                        if two_body_info["total_terms"]
                        else []
                    )
                    two_body_builder = "SU2System[P/Q]"
                else:
                    eri_builder = SpatialSpinFreeERIBuilder(
                        site_legs,
                        eri_spatial,
                        cutoff=self.cutoff,
                    )
                    two_body_factors, two_body_info = eri_builder.build(return_info=True)
                    two_body_builder = "SpatialSpinFreeERIBuilder"
            else:
                two_body_builder = "none"
            factors = sum_mpo_chains(
                one_body_factors,
                two_body_factors,
                phys_leg=physical_leg_from_spatial_orbital(site_descriptor),
                cutoff=self.cutoff,
            )
        two_body_term_count = int(two_body_info["total_terms"])
        we_product_terms = int(two_body_info["we_product_terms"])
        scalar_product_terms = int(two_body_info["scalar_product_terms"])
        fully_reduced_density_terms = int(two_body_info.get("fully_reduced_density_terms", 0))
        fully_reduced_density_bilinear_terms = int(two_body_info.get("fully_reduced_density_bilinear_terms", 0))
        fully_reduced_pair_terms = int(two_body_info.get("fully_reduced_pair_terms", 0))
        fully_reduced_exchange_terms = int(two_body_info.get("fully_reduced_exchange_terms", 0))
        one_body_correction_terms = int(two_body_info["one_body_correction_terms"])
        has_two_body = bool(two_body_term_count)
        info = {
            "block_hamiltonian": True,
            "block_hamiltonian_class": "ReducedSpatialHamiltonian",
            "representation": (
                "spatial_reduced_spinfree_mpo"
                if two_body_term_count
                else "spatial_reduced_mixed_mpo"
            ),
            "site": "spatial",
            "spatial_site_basis": "fully_reduced_su2" if self.fully_reduced else "canonical_su2",
            "ncas": int(h_spatial.shape[0]),
            "n_sites": int(h_spatial.shape[0]),
            "n_elec": None if self.nelec is None else int(self.nelec),
            "spin": int(self.spin),
            "ecore": float(self.ecore),
            "orb_sym": None if self.orb_sym is None else tuple(self.orb_sym),
            "one_body_reduced": True,
            "one_body_reduced_source": True,
            "final_mpo_reduced_metadata": True,
            "pipeline": "cpp_integrals->su2_system->reduced_complementary_families",
            "su2_system": (
                None if moving_environment is None else moving_environment.system_stats
            ),
            "normal_complementary_routes": (
                None
                if moving_environment is None
                else {
                    "owner": "cpp_system",
                    "layout": "su2_normal_complementary",
                    "transition_count": int(
                        moving_environment.system_stats.get(
                            "normal_complementary_transition_count", 0
                        )
                    ),
                    "memory_bytes": int(
                        moving_environment.system_stats.get(
                            "normal_complementary_memory_bytes", 0
                        )
                    ),
                    "reference_carrier_required": not production_normal_complementary,
                }
            ),
            "complementary_operator_families": complementary.as_metadata(),
            "complementary_operator_family_names": complementary.names,
            "complementary_operator_total_terms": int(complementary.n_terms),
            "complementary_operator_builder": "spatial_spinfree_sparse_S/R/A/P/B/Q",
            "two_body": has_two_body,
            "two_body_builder": two_body_builder if has_two_body else "none",
            "reference_carrier": bool(
                has_two_body and not production_normal_complementary
            ),
            "reference_carrier_source": (
                None
                if production_normal_complementary
                else (
                    "cpp_complementary_P/Q"
                    if two_body_builder == "SU2System[P/Q]"
                    else two_body_builder
                )
            ),
            "normal_complementary_production": production_normal_complementary,
            "su2_runtime_reused": bool(reused_runtime),
            "python_reduced_terms_materialized": bool(
                not (production_normal_complementary and self.fully_reduced)
            ),
            "includes_core_energy": production_normal_complementary,
            "two_body_representation": "+".join(
                part
                for part, enabled in (
                    ("we_general_reduced_strings", we_product_terms),
                    ("fully_reduced_density_eri", fully_reduced_density_terms),
                    ("fully_reduced_density_bilinear_eri", fully_reduced_density_bilinear_terms),
                    ("fully_reduced_pair_eri", fully_reduced_pair_terms),
                    ("fully_reduced_exchange_eri", fully_reduced_exchange_terms),
                    ("spinfree_scalar_coupled_eri", scalar_product_terms or one_body_correction_terms),
                )
                if enabled
            )
            or "none",
            "two_body_reduced_string_terms": int(we_product_terms),
            "two_body_scalar_density_terms": 0,
            "two_body_compressed_pair_terms": 0,
            "two_body_compressed_pair_input_terms": 0,
            "two_body_scalar_product_terms": int(scalar_product_terms),
            "two_body_fully_reduced_density_terms": int(fully_reduced_density_terms),
            "two_body_fully_reduced_density_bilinear_terms": int(fully_reduced_density_bilinear_terms),
            "two_body_fully_reduced_pair_terms": int(fully_reduced_pair_terms),
            "two_body_fully_reduced_exchange_terms": int(fully_reduced_exchange_terms),
            "two_body_one_body_correction_terms": int(one_body_correction_terms),
            "two_body_symbolic_terms": 0,
            "mpo_max_bond": int(max(core.right_dim for core in factors)),
        }
        return ReducedSpatialHamiltonian(
            factors=list(factors),
            info=info,
            n_sites=int(h_spatial.shape[0]),
            nelec=self.nelec,
            spin=int(self.spin),
            ecore=float(self.ecore),
            orb_sym=None if self.orb_sym is None else tuple(self.orb_sym),
            symmetry="su2",
            complementary_operators=complementary,
            moving_environment=moving_environment,
            h1e=np.array(h_spatial, copy=True),
            eri=None if eri_spatial is None else np.array(eri_spatial, copy=True),
            cutoff=float(self.cutoff),
            fully_reduced=bool(self.fully_reduced),
        )


def _restricted_spatial_h1e(h1e):
    arr = np.asarray(h1e)
    if arr.ndim == 2:
        h_spatial = arr
    elif arr.ndim == 3 and arr.shape[0] >= 1:
        h_spatial = arr[0]
        if arr.shape[0] > 1 and not np.allclose(arr[0], arr[1], atol=1.0e-10, rtol=1.0e-10):
            raise NotImplementedError(
                "Reduced spatial Hamiltonian builder currently expects restricted alpha/beta one-electron integrals."
            )
    else:
        raise ValueError("h1e must be a spatial matrix or spin-resolved array with shape (spin, n, n).")
    if h_spatial.ndim != 2 or h_spatial.shape[0] != h_spatial.shape[1]:
        raise ValueError("h1e spatial block must be square.")
    return np.asarray(h_spatial)


def build_spatial_reduced_hamiltonian_mpo(
    h1e,
    eri=None,
    *,
    cutoff=1.0e-10,
    fully_reduced=False,
    nelec=None,
    spin=0,
    ecore=0.0,
    orb_sym=None,
    reuse=None,
):
    """
    Build a qchem spatial Hamiltonian MPO using reduced SU(2) channels.

    This covers the restricted active-space Hamiltonian

        sum_pq,sigma h[p,q] c^dagger[p,sigma] c[q,sigma]
        + 1/2 sum_pqrs,sigma,tau (pq|rs)
          c^dagger[p,sigma] c^dagger[r,tau] c[s,tau] c[q,sigma]

    The one-electron part is generated with reduced rank-1/2 endpoint tensors.
    The two-electron reference carrier is generated from the C++ ``P/Q``
    family records representing ``E_pq E_rs - delta_qr E_ps``.  The final
    Hamiltonian sum preserves visible reduced virtual-channel metadata by
    direct-summing :class:`RankCoupledMPO` cores instead of expanding the
    cores first.

    :param h1e: Spatial one-electron matrix or restricted spin-resolved
        one-electron integrals.
    :param eri: Optional restricted spin-resolved two-electron integrals.
    :param cutoff: Absolute screening threshold.
    :param fully_reduced: Whether to use fully reduced spatial SU(2) sites.
    :param nelec: Target active-electron count, used as system metadata.
    :param spin: Target doubled spin ``2S``.
    :param ecore: Scalar core energy outside the active-space MPO.
    :param orb_sym: Optional orbital symmetry labels.
    :returns: :class:`ReducedSpatialHamiltonian`.
    """
    return SpatialReducedHamiltonianBuilder(
        h1e,
        eri=eri,
        cutoff=cutoff,
        fully_reduced=fully_reduced,
        nelec=nelec,
        spin=spin,
        ecore=ecore,
        orb_sym=None if orb_sym is None else tuple(orb_sym),
        reuse=reuse,
    ).build()
