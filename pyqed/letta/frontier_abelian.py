"""Abelian charge sectors for graph-tied frontier LETTA states.

Every physical site is *owned* by exactly one graph-LETTA tensor.  Physical
parent legs are hard-copy conditioning labels and are therefore neutral
spectators in the charge-flow equations.  Counting their physical charge a
second time would make the total charge depend on the chosen tie graph.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import dataclass, replace

import numpy as np

from .abelian import _add_charges, _as_charge, _sub_charges
from .core import _lowest_generalized_eigenpair, _lowest_hermitian_eigenpair
from .cp_tying import _validated_parent_sets
from .frontier_tying import (
    FrontierGaugeUpdate,
    FrontierSiteUpdate,
    FrontierTiedLETTA,
)
from .matrix_free import lowest_generalized_davidson
from .initialization import _validated_mps_tensors


def _charge_counts(local_qns, *, initial):
    """Return prefix charge multiplicities after every local site."""
    counts = [Counter({_as_charge(initial): 1})]
    for site_qns in local_qns:
        following = Counter()
        for prefix, multiplicity in counts[-1].items():
            for charge in site_qns:
                following[_add_charges(prefix, charge)] += multiplicity
        counts.append(following)
    return counts


def _suffix_charge_counts(local_qns, *, rank):
    """Return suffix-sum multiplicities beginning at every local site."""
    zero = tuple(0 for _ in range(int(rank)))
    counts = [None] * (len(local_qns) + 1)
    counts[-1] = Counter({zero: 1})
    for site in range(len(local_qns) - 1, -1, -1):
        current = Counter()
        for suffix, multiplicity in counts[site + 1].items():
            for charge in local_qns[site]:
                current[_add_charges(charge, suffix)] += multiplicity
        counts[site] = current
    return counts


def _normalized_bond_dims(nsites, bond_dims):
    """Normalize scalar, internal-only, or boundary-inclusive bond sizes."""
    nsites = int(nsites)
    if np.isscalar(bond_dims):
        dimension = int(bond_dims)
        dims = (1,) + (dimension,) * max(0, nsites - 1) + (1,)
    else:
        dims = tuple(int(dimension) for dimension in bond_dims)
        if len(dims) == nsites - 1:
            dims = (1,) + dims + (1,)
        elif len(dims) != nsites + 1:
            raise ValueError(
                "bond_dims must be scalar, contain nsites-1 internal sizes, "
                "or contain nsites+1 boundary-inclusive sizes."
            )
    if any(dimension < 1 for dimension in dims):
        raise ValueError("all bond dimensions must be positive.")
    if dims[0] != 1 or dims[-1] != 1:
        raise ValueError("open-boundary charge layouts require unit end bonds.")
    return dims


def _allocate_sector_labels(feasible, dimension, *, center):
    """Allocate a finite bond degeneracy among the dominant charge sectors."""
    dimension = int(dimension)
    center = np.asarray(center, dtype=float)
    ranked = sorted(
        feasible.items(),
        key=lambda item: (
            -int(item[1]),
            float(np.sum((np.asarray(item[0], dtype=float) - center) ** 2)),
            item[0],
        ),
    )
    selected = ranked[:dimension]
    if not selected:
        raise ValueError("the requested target charge has no feasible bond sector.")

    degeneracies = {charge: 1 for charge, _weight in selected}
    weights = {charge: int(weight) for charge, weight in selected}
    while sum(degeneracies.values()) < dimension:
        charge = max(
            degeneracies,
            key=lambda label: (
                weights[label] / degeneracies[label],
                -float(
                    np.sum((np.asarray(label, dtype=float) - center) ** 2)
                ),
            ),
        )
        degeneracies[charge] += 1
    return tuple(
        charge
        for charge in sorted(degeneracies)
        for _ in range(degeneracies[charge])
    )


@dataclass(frozen=True)
class FrontierAbelianLayout:
    r"""Fixed-target Abelian layout for graph-tied LETTA tensors.

    ``bond_qns[k]`` labels the virtual bond immediately before site ``k``;
    hence there are ``nsites + 1`` bond-label lists.  An entry of the tensor at
    site ``k`` is allowed when

    .. math::

        q_{k+1} = q_k + q(s_k).

    Charges of the tied parent states do not enter this equation: those axes
    refer to physical sites whose charge is already counted by their owning
    tensor.
    """

    local_qns: tuple[tuple[tuple[int, ...], ...], ...]
    bond_qns: tuple[tuple[tuple[int, ...], ...], ...]
    target: tuple[int, ...]

    def __post_init__(self):
        local_qns = tuple(
            tuple(_as_charge(charge) for charge in site)
            for site in self.local_qns
        )
        bond_qns = tuple(
            tuple(_as_charge(charge) for charge in bond)
            for bond in self.bond_qns
        )
        target = _as_charge(self.target)
        if not local_qns:
            raise ValueError("a frontier Abelian layout needs at least one site.")
        if len(bond_qns) != len(local_qns) + 1:
            raise ValueError("bond_qns must contain one label list per chain cut.")
        if any(not site for site in local_qns):
            raise ValueError("each site must contain at least one local charge.")
        if any(not bond for bond in bond_qns):
            raise ValueError("each bond must contain at least one charge label.")
        if len(bond_qns[0]) != 1 or len(bond_qns[-1]) != 1:
            raise ValueError("open-boundary charge layouts require unit end bonds.")
        rank = len(target)
        charges = [target]
        charges.extend(charge for site in local_qns for charge in site)
        charges.extend(charge for bond in bond_qns for charge in bond)
        if any(len(charge) != rank for charge in charges):
            raise ValueError("all Abelian charges must have the target charge rank.")
        if bond_qns[-1][0] != target:
            raise ValueError("the right-boundary bond charge must equal target.")
        object.__setattr__(self, "local_qns", local_qns)
        object.__setattr__(self, "bond_qns", bond_qns)
        object.__setattr__(self, "target", target)

    @property
    def nsites(self):
        return len(self.local_qns)

    @property
    def dims(self):
        return tuple(len(site) for site in self.local_qns)

    @property
    def bond_dims(self):
        return tuple(len(bond) for bond in self.bond_qns)

    @classmethod
    def from_local_charges(
        cls,
        local_qns,
        *,
        target,
        bond_dims,
        left_boundary=None,
        bond_qns=None,
    ):
        """Build a compact fixed-target layout with finite sector degeneracy.

        When explicit ``bond_qns`` are omitted, feasible charge sectors are
        ranked by their exact prefix-times-suffix configuration multiplicity.
        The available bond dimension is then distributed among the dominant
        sectors.  Supplying ``bond_qns`` remains the preferred route when a
        charge-resolved MPS warm start fixes a particular degeneracy layout.
        """
        local_qns = tuple(
            tuple(_as_charge(charge) for charge in site)
            for site in local_qns
        )
        target = _as_charge(target)
        if left_boundary is None:
            left_boundary = tuple(0 for _ in target)
        left_boundary = _as_charge(left_boundary)
        if len(left_boundary) != len(target):
            raise ValueError("left_boundary and target must have the same rank.")
        dims = _normalized_bond_dims(len(local_qns), bond_dims)
        if bond_qns is not None:
            bond_qns = tuple(
                tuple(_as_charge(charge) for charge in bond)
                for bond in bond_qns
            )
            if tuple(len(bond) for bond in bond_qns) != dims:
                raise ValueError("explicit bond_qns do not match bond_dims.")
            return cls(local_qns, bond_qns, target)

        prefix = _charge_counts(local_qns, initial=left_boundary)
        suffix = _suffix_charge_counts(local_qns, rank=len(target))
        labels = [(left_boundary,)]
        for cut in range(1, len(local_qns)):
            feasible = {}
            for charge, left_count in prefix[cut].items():
                needed = _sub_charges(target, charge)
                right_count = suffix[cut].get(needed, 0)
                if right_count:
                    feasible[charge] = int(left_count) * int(right_count)
            fraction = cut / len(local_qns)
            center = tuple(
                (1.0 - fraction) * left + fraction * right
                for left, right in zip(left_boundary, target)
            )
            labels.append(
                _allocate_sector_labels(
                    feasible,
                    dims[cut],
                    center=center,
                )
            )
        labels.append((target,))
        layout = cls(local_qns, tuple(labels), target)
        layout._validate_charge_paths()
        return layout

    @classmethod
    def spin_half(
        cls,
        nsites,
        *,
        target_two_sz=0,
        bond_dims=1,
    ):
        """Build a spin-1/2 total-``2*S^z`` layout in the ``up, down`` basis."""
        nsites = int(nsites)
        if nsites < 1:
            raise ValueError("nsites must be positive.")
        target_two_sz = int(target_two_sz)
        if abs(target_two_sz) > nsites or (target_two_sz - nsites) % 2:
            raise ValueError("target_two_sz is incompatible with nsites spin halves.")
        local_qns = (((1,), (-1,)),) * nsites
        return cls.from_local_charges(
            local_qns,
            target=(target_two_sz,),
            bond_dims=bond_dims,
        )

    def _validate_charge_paths(self):
        for site, site_qns in enumerate(self.local_qns):
            left = self.bond_qns[site]
            right = self.bond_qns[site + 1]
            allowed_left = {
                q_left
                for q_left in left
                if any(
                    _add_charges(q_left, q_site) == q_right
                    for q_site in site_qns
                    for q_right in right
                )
            }
            allowed_right = {
                q_right
                for q_right in right
                if any(
                    _add_charges(q_left, q_site) == q_right
                    for q_left in left
                    for q_site in site_qns
                )
            }
            if set(left) != allowed_left or set(right) != allowed_right:
                raise ValueError(
                    f"automatically selected charge sectors disconnect at site {site}; "
                    "provide explicit bond_qns."
                )

    def local_masks(self, physical_sites):
        """Return symmetry masks in native ``(left,right,site,parents...)`` order."""
        physical_sites = tuple(tuple(int(index) for index in sites) for sites in physical_sites)
        if len(physical_sites) != self.nsites:
            raise ValueError("physical_sites must contain one entry per tensor.")
        masks = []
        for site, sites in enumerate(physical_sites):
            if not sites or sites[0] != site:
                raise ValueError("each graph tensor must own its leading physical site.")
            shape = (
                len(self.bond_qns[site]),
                len(self.bond_qns[site + 1]),
                *(self.dims[index] for index in sites),
            )
            mask = np.zeros(shape, dtype=bool)
            spectator = (slice(None),) * (len(sites) - 1)
            for left, q_left in enumerate(self.bond_qns[site]):
                for state, q_site in enumerate(self.local_qns[site]):
                    needed = _add_charges(q_left, q_site)
                    for right, q_right in enumerate(self.bond_qns[site + 1]):
                        if q_right == needed:
                            mask[(left, right, state, *spectator)] = True
            if not np.any(mask):
                raise ValueError(f"charge layout removes every entry at site {site}.")
            masks.append(mask)
        return tuple(masks)

    def structural_support_sizes(self, physical_sites):
        return tuple(
            int(np.count_nonzero(mask)) for mask in self.local_masks(physical_sites)
        )

    def with_expanded_bond(self, cut, labels):
        """Return a layout with appended charge-degeneracy labels at one cut."""
        cut = int(cut)
        if cut <= 0 or cut >= self.nsites:
            raise ValueError("cut must be an internal virtual bond.")
        labels = tuple(_as_charge(label) for label in labels)
        if any(len(label) != len(self.target) for label in labels):
            raise ValueError("new bond labels must have the target charge rank.")
        bonds = list(self.bond_qns)
        bonds[cut] = tuple(bonds[cut]) + labels
        result = type(self)(self.local_qns, tuple(bonds), self.target)
        result._validate_charge_paths()
        return result


class AbelianFrontierTiedLETTA(FrontierTiedLETTA):
    r"""Graph-tied frontier LETTA restricted to a fixed Abelian charge sector.

    The contractor remains dense in charge labels for now, while tensors and
    every local eigensolve are projected onto symmetry-compatible entries.
    This already removes forbidden variational parameters and prevents local
    optimization from leaking out of the requested total-charge sector.
    """

    def __init__(
        self,
        hamiltonian,
        dims,
        parent_sets,
        *,
        abelian_layout: FrontierAbelianLayout,
        **kwargs,
    ):
        if not isinstance(abelian_layout, FrontierAbelianLayout):
            raise TypeError("abelian_layout must be a FrontierAbelianLayout.")
        if tuple(int(dim) for dim in dims) != abelian_layout.dims:
            raise ValueError("abelian_layout local dimensions do not match dims.")
        self.abelian_layout = abelian_layout
        if "bond_dims" not in kwargs:
            if "bond_dim" in kwargs:
                requested = _normalized_bond_dims(
                    len(abelian_layout.local_qns), kwargs["bond_dim"]
                )
                if requested != abelian_layout.bond_dims:
                    raise ValueError(
                        "bond_dim is inconsistent with abelian_layout; use the "
                        "layout bond dimensions or supply matching bond_dims."
                    )
            else:
                kwargs["bond_dims"] = abelian_layout.bond_dims
        super().__init__(hamiltonian, dims, parent_sets, **kwargs)
        if tuple(self._bond_dims()) != abelian_layout.bond_dims:
            raise ValueError(
                "abelian_layout bond dimensions do not match the frontier state."
            )
        self.local_masks = abelian_layout.local_masks(self.physical_sites)
        self._apply_local_masks()
        self.balance_gauges()
        self.energy = self.expectation()

    @property
    def nparameters(self):
        """Number of independent symmetry-allowed dense tensor entries."""
        return int(sum(np.count_nonzero(mask) for mask in self.local_masks))

    @property
    def dense_nparameters(self):
        """Parameter count of the same graph tensors without symmetry masks."""
        return int(sum(tensor.size for tensor in self.tensors))

    def local_support_sizes(self):
        return tuple(
            (int(np.count_nonzero(mask)), int(mask.size))
            for mask in self.local_masks
        )

    def _apply_local_masks(self):
        for site, mask in enumerate(self.local_masks):
            self.tensors[site] = np.where(mask, self.tensors[site], 0)

    def copy(self):
        result = type(self)(
            self.hamiltonian,
            self.dims,
            self.parent_sets,
            abelian_layout=self.abelian_layout,
            bond_dim=self.bond_dim,
            bond_dims=self._bond_dims(),
            tensors=[tensor.copy() for tensor in self.tensors],
            frontier_backend=self.frontier_backend,
            path_optimizer=self.path_optimizer,
            tt_max_rank=self.tt_options["max_rank"],
            tt_rtol=self.tt_options["rtol"],
            tt_atol=self.tt_options["atol"],
            tt_transfer_max_rank=self.tt_options["transfer_max_rank"],
            tt_transfer_rtol=self.tt_options["transfer_rtol"],
            tt_transfer_atol=self.tt_options["transfer_atol"],
            tt_absorption=self.tt_options["absorption"],
            tt_norm_backend=self.tt_norm_backend,
            tt_hermitize=self.tt_hermitize,
        )
        result.tensors = [tensor.copy() for tensor in self.tensors]
        result.history = list(self.history)
        result.energy = result.expectation()
        result.converged = self.converged
        result.rng.bit_generator.state = deepcopy(self.rng.bit_generator.state)
        return result

    def _automatic_expansion_labels(self, cut, count):
        """Choose charge labels connected to both neighboring fixed layouts."""
        cut = int(cut)
        count = int(count)
        left_labels = self.abelian_layout.bond_qns[cut - 1]
        current_labels = self.abelian_layout.bond_qns[cut]
        right_labels = self.abelian_layout.bond_qns[cut + 1]
        incoming_qns = self.abelian_layout.local_qns[cut - 1]
        outgoing_qns = self.abelian_layout.local_qns[cut]
        incoming = Counter(
            _add_charges(q_left, q_site)
            for q_left in left_labels
            for q_site in incoming_qns
        )
        outgoing = Counter()
        for charge in incoming:
            outgoing[charge] = sum(
                1
                for q_site in outgoing_qns
                for q_right in right_labels
                if _add_charges(charge, q_site) == q_right
            )
        candidates = {
            charge: incoming[charge] * outgoing[charge]
            for charge in incoming
            if outgoing[charge]
        }
        if not candidates and count:
            raise ValueError(
                "no charge sector connects the two neighboring bond layouts."
            )
        degeneracies = Counter(current_labels)
        labels = []
        for _ in range(count):
            charge = max(
                candidates,
                key=lambda candidate: (
                    candidates[candidate] / (degeneracies[candidate] + 1),
                    -sum(abs(value) for value in candidate),
                    tuple(-value for value in candidate),
                ),
            )
            labels.append(charge)
            degeneracies[charge] += 1
        return tuple(labels)

    def expand_bond(
        self,
        cut,
        new_dimension,
        *,
        new_charge_labels=None,
        direction="right",
        strategy="residual",
        scale=1.0e-3,
        seed=None,
    ):
        """Expand one cut while appending symmetry-compatible charge labels.

        The unmatched side of every new channel is initialized to zero by the
        base expansion, so the represented state is unchanged.  Residual or
        random seeds on the other side are then projected by the new Abelian
        masks.  Explicit ``new_charge_labels`` can reproduce the sector layout
        of a charge-resolved warm start; otherwise connected sectors are
        allocated deterministically.
        """
        cut = int(cut)
        new_dimension = int(new_dimension)
        if cut <= 0 or cut >= len(self.dims):
            raise ValueError("cut must be an internal virtual bond.")
        old_dimension = self._bond_dims()[cut]
        added = new_dimension - old_dimension
        if added < 0:
            raise ValueError("expand_bond only supports increasing dimensions.")
        if new_charge_labels is None:
            labels = self._automatic_expansion_labels(cut, added)
        else:
            labels = tuple(_as_charge(label) for label in new_charge_labels)
            if len(labels) != added:
                raise ValueError(
                    "new_charge_labels must contain one label per added channel."
                )
        new_layout = self.abelian_layout.with_expanded_bond(cut, labels)
        record = super().expand_bond(
            cut,
            new_dimension,
            direction=direction,
            strategy=strategy,
            scale=scale,
            seed=seed,
        )
        self.abelian_layout = new_layout
        self.local_masks = new_layout.local_masks(self.physical_sites)
        self._apply_local_masks()
        energy_after = self.expectation()
        normalized_direction = str(direction).lower().replace("_", "-")
        expands_right = normalized_direction in {
            "right",
            "lr",
            "left-to-right",
            "forward",
        }
        seeded = (
            sum(
                np.linalg.norm(self.tensors[cut - 1][:, channel]) > 0.0
                for channel in range(old_dimension, new_dimension)
            )
            if expands_right
            else sum(
                np.linalg.norm(self.tensors[cut][channel]) > 0.0
                for channel in range(old_dimension, new_dimension)
            )
        )
        self.energy = energy_after
        return replace(
            record,
            seeded_directions=int(seeded),
            energy=energy_after,
        )

    def canonicalize_virtual(self, direction):
        raise NotImplementedError(
            "dense virtual QR can mix Abelian sectors; use the sector-preserving "
            "frontier gauge or disable virtual canonicalization."
        )

    def natural_gradient_step(self, **kwargs):
        raise NotImplementedError(
            "the unrestricted block natural gradient can leave the Abelian "
            "support; symmetry-projected natural gradients are not implemented."
        )

    def _support_indices(self, site):
        site = self._validated_site(site)
        return np.flatnonzero(self.local_masks[site].reshape(-1))

    @staticmethod
    def _embed_support(vector, indices, size, *, dtype):
        embedded = np.zeros(int(size), dtype=np.result_type(vector, dtype))
        embedded[indices] = vector
        return embedded

    def optimize_site(
        self,
        site,
        *,
        metric_tol=1.0e-12,
        solver="auto",
        matrix_free_threshold=256,
        block_sparse_max_elements=4_000_000,
        eig_tol=1.0e-10,
        maxiter=None,
        max_subspace=32,
        energy_before=None,
        environment=None,
    ):
        """Minimize one tensor strictly inside its Abelian support."""
        site = self._validated_site(site)
        solver = str(solver).lower().replace("-", "_")
        if solver in {"block", "physical_block", "physical_blocks"}:
            solver = "block_sparse"
        if solver in {"canonical", "identity_metric", "local_canonical", "s_identity"}:
            solver = "whitened"
        if solver not in {"auto", "direct", "whitened", "matrix_free", "block_sparse"}:
            raise ValueError(
                "solver must be 'auto', 'direct', 'whitened', "
                "'matrix_free', or 'block_sparse'."
            )
        if not self.norm_contraction_is_exact:
            raise ValueError("variational optimization requires an exact norm contraction.")
        if not self.hamiltonian_action_is_hermitian:
            raise ValueError("variational optimization requires a Hermitian action.")
        metric_tol = float(metric_tol)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        old_tensor = self.tensors[site].copy()
        support = self._support_indices(site)
        if support.size == 0:
            raise ValueError("the Abelian mask removes every local tensor entry.")
        if energy_before is None:
            energy_before = self.expectation()
        energy_before = float(energy_before)
        environment = self._resolved_environment(site, environment)

        selected_solver = solver
        if selected_solver == "auto":
            selected_solver = (
                "direct"
                if support.size < int(matrix_free_threshold)
                and self.frontier_backend != "tensor_train"
                else "matrix_free"
            )
        if selected_solver in {"direct", "whitened", "block_sparse"} and self.frontier_backend == "tensor_train":
            if selected_solver == "block_sparse":
                selected_solver = "matrix_free"
            else:
                raise ValueError(
                    f"solver='{selected_solver}' is unavailable for tensor-train frontiers."
                )

        accepted = False
        energy_after = energy_before
        metric_rank = 0
        hamiltonian_matvecs = 0
        metric_matvecs = 0
        iterations = 0
        residual_norm = float("inf")
        solver_converged = False
        message = "local solve not attempted"
        physical_blocks = 0
        hamiltonian_blocks = 0
        block_component_sizes = ()
        stored_operator_elements = 0
        solver_metric_is_identity = False
        solver_metric_identity_error = float("nan")
        solver_coordinate_residual_norm = float("nan")
        full_size = old_tensor.size

        def embed(vector):
            return self._embed_support(
                vector,
                support,
                full_size,
                dtype=old_tensor.dtype,
            )

        try:
            if selected_solver in {"direct", "whitened"}:
                metric, effective = self.local_operators(site, environment=environment)
                reduced_metric = metric[np.ix_(support, support)]
                reduced_effective = effective[np.ix_(support, support)]
                if selected_solver == "direct":
                    eigenvalues = np.linalg.eigvalsh(reduced_metric)
                    scale = max(
                        float(np.linalg.norm(reduced_metric, ord=np.inf)),
                        np.finfo(float).tiny,
                    )
                    metric_rank = int(
                        np.count_nonzero(eigenvalues > metric_tol * scale)
                    )
                    energy_after, reduced_vector = _lowest_generalized_eigenpair(
                        reduced_effective,
                        reduced_metric,
                        metric_tol=metric_tol,
                    )
                    message = "converged in the Abelian support"
                else:
                    basis, hamiltonian, frame = self._whiten_local_operators(
                        reduced_metric,
                        reduced_effective,
                        metric_tol=metric_tol,
                    )
                    energy, solver_vector = _lowest_hermitian_eigenpair(hamiltonian)
                    solver_coordinate_residual_norm = float(
                        np.linalg.norm(hamiltonian @ solver_vector - energy * solver_vector)
                    )
                    reduced_vector = basis @ solver_vector
                    metric_rank = int(frame["metric_rank"])
                    solver_metric_is_identity = True
                    solver_metric_identity_error = float(frame["identity_metric_error"])
                    message = "converged in the Abelian local S=I frame"
                vector = embed(reduced_vector)
                metric_vector = reduced_metric @ reduced_vector
                hamiltonian_vector = reduced_effective @ reduced_vector
                denominator = np.vdot(reduced_vector, metric_vector)
                energy_after = float(
                    np.real(np.vdot(reduced_vector, hamiltonian_vector) / denominator)
                )
                residual_norm = float(
                    np.linalg.norm(
                        hamiltonian_vector - energy_after * metric_vector
                    )
                )
                solver_converged = True
            else:
                if selected_solver == "block_sparse":
                    problem = self.local_block_problem(site, environment=environment)

                    def hamiltonian_action(trial):
                        return problem.hamiltonian @ trial

                    def metric_action(trial):
                        return problem.metric @ trial

                    physical_blocks = len(problem.metric.blocks)
                    hamiltonian_blocks = len(problem.hamiltonian.blocks)
                    stored_operator_elements = problem.stored_elements
                else:
                    def hamiltonian_action(trial):
                        return self.hamiltonian_action(
                            site, trial, environment=environment
                        )

                    def metric_action(trial):
                        return self.metric_action(
                            site, trial, environment=environment
                        )

                def projected(action, trial):
                    return action(embed(trial))[support]

                energy_after, reduced_vector, diagnostics = lowest_generalized_davidson(
                    lambda trial: projected(hamiltonian_action, trial),
                    lambda trial: projected(metric_action, trial),
                    old_tensor.reshape(-1)[support],
                    tol=eig_tol,
                    metric_tol=metric_tol,
                    maxiter=maxiter,
                    max_subspace=max_subspace,
                    random_seed=site,
                )
                vector = embed(reduced_vector)
                metric_rank = diagnostics.projected_rank
                hamiltonian_matvecs = diagnostics.hamiltonian_matvecs
                metric_matvecs = diagnostics.metric_matvecs
                iterations = diagnostics.iterations
                residual_norm = diagnostics.residual_norm
                solver_converged = diagnostics.converged
                message = f"{diagnostics.message}; Abelian support projected"
                if not diagnostics.converged:
                    raise ValueError(diagnostics.message)

            tolerance = 256.0 * np.finfo(float).eps * max(1.0, abs(energy_before))
            candidate = np.real_if_close(vector.reshape(old_tensor.shape))
            candidate = np.where(self.local_masks[site], candidate, 0)
            accepted = (
                np.isfinite(energy_after)
                and energy_after <= energy_before + tolerance
            )
            if accepted:
                self.tensors[site] = candidate.astype(
                    np.result_type(old_tensor.dtype, candidate.dtype), copy=False
                )
                if not self.hamiltonian_contraction_is_exact:
                    checked_energy = self.expectation()
                    accepted = (
                        np.isfinite(checked_energy)
                        and checked_energy <= energy_before + tolerance
                    )
                    if accepted:
                        energy_after = float(checked_energy)
                        message = f"{message}; accepted by global TT energy check"
                    else:
                        message = f"{message}; rejected by global TT energy check"
                        energy_after = energy_before
            else:
                energy_after = energy_before
        except (ValueError, np.linalg.LinAlgError) as error:
            accepted = False
            solver_converged = False
            if message == "local solve not attempted":
                message = str(error)
            else:
                message = f"{message}; {error}"
        if not accepted:
            self.tensors[site] = old_tensor
            energy_after = energy_before
        self.energy = float(energy_after)
        return FrontierSiteUpdate(
            site=site,
            raw_dim=old_tensor.size,
            metric_rank=metric_rank,
            metric_rank_is_projected=(selected_solver in {"matrix_free", "block_sparse"}),
            solver=selected_solver,
            solver_converged=solver_converged,
            message=message,
            hamiltonian_matvecs=hamiltonian_matvecs,
            metric_matvecs=metric_matvecs,
            iterations=iterations,
            residual_norm=residual_norm,
            energy_before=energy_before,
            energy=float(energy_after),
            accepted=bool(accepted),
            physical_blocks=physical_blocks,
            hamiltonian_blocks=hamiltonian_blocks,
            block_component_sizes=block_component_sizes,
            stored_operator_elements=stored_operator_elements,
            solver_metric_is_identity=solver_metric_is_identity,
            solver_metric_identity_error=solver_metric_identity_error,
            solver_coordinate_residual_norm=solver_coordinate_residual_norm,
        )

    def canonicalize_frontier_gauge(
        self,
        *,
        metric_tol=1.0e-12,
        max_condition=1.0e8,
        weighting="uniform",
    ):
        """Balance frontier Grams with gauges block diagonal in charge."""
        if self.frontier_backend == "tensor_train":
            raise NotImplementedError(
                "sector-preserving frontier gauges require exact dense norm messages."
            )
        metric_tol = float(metric_tol)
        max_condition = float(max_condition)
        if not np.isfinite(metric_tol) or metric_tol < 0.0:
            raise ValueError("metric_tol must be finite and nonnegative.")
        if not np.isfinite(max_condition) or max_condition < 1.0:
            raise ValueError("max_condition must be finite and at least one.")
        left_messages = self._norm_frontier.build_left(self.tensors)
        right_messages = self._norm_frontier.build_right(self.tensors)
        updates = []
        tiny = np.finfo(float).tiny
        relative_floor = max(metric_tol, 128.0 * np.finfo(float).eps)
        for cut, labels in enumerate(self.abelian_layout.bond_qns[1:-1], start=1):
            left, right = self.frontier_bond_grams(
                cut,
                left_messages=left_messages,
                right_messages=right_messages,
                weighting=weighting,
            )
            dimension = len(labels)
            gauge = np.eye(dimension, dtype=np.result_type(left, right))
            left_rank = 0
            right_rank = 0
            applied = True
            messages = []
            for charge in sorted(set(labels)):
                indices = np.asarray(
                    [index for index, label in enumerate(labels) if label == charge],
                    dtype=np.intp,
                )
                left_block = self._hermitian_part(left[np.ix_(indices, indices)])
                right_block = self._hermitian_part(right[np.ix_(indices, indices)])
                left_values, left_vectors = np.linalg.eigh(left_block)
                right_values = np.linalg.eigvalsh(right_block)
                left_scale = max(float(np.max(left_values, initial=0.0)), tiny)
                right_scale = max(float(np.max(right_values, initial=0.0)), tiny)
                active_left = left_values > relative_floor * left_scale
                active_right = right_values > relative_floor * right_scale
                left_rank += int(np.count_nonzero(active_left))
                right_rank += int(np.count_nonzero(active_right))
                if not np.all(active_left) or not np.all(active_right):
                    applied = False
                    messages.append(f"rank-deficient sector {charge}")
                    continue
                left_trace = float(np.trace(left_block).real / len(indices))
                right_trace = float(np.trace(right_block).real / len(indices))
                normalized_values = left_values / left_trace
                left_half = (
                    left_vectors * np.sqrt(normalized_values)
                ) @ left_vectors.conj().T
                left_inverse_half = (
                    left_vectors * (1.0 / np.sqrt(normalized_values))
                ) @ left_vectors.conj().T
                center = self._hermitian_part(
                    left_half @ (right_block / right_trace) @ left_half
                )
                center_values, center_vectors = np.linalg.eigh(center)
                if np.any(center_values <= relative_floor * max(float(center_values[-1]), tiny)):
                    applied = False
                    messages.append(f"singular balanced sector {charge}")
                    continue
                sector_gauge = (
                    (right_trace / left_trace) ** 0.25
                    * left_inverse_half
                    @ ((center_vectors * center_values**0.25) @ center_vectors.conj().T)
                )
                if np.linalg.cond(sector_gauge) > max_condition:
                    applied = False
                    messages.append(f"ill-conditioned sector {charge}")
                    continue
                gauge[np.ix_(indices, indices)] = sector_gauge

            denominator = max(float(np.linalg.norm(left) + np.linalg.norm(right)), tiny)
            imbalance_before = float(2.0 * np.linalg.norm(left - right) / denominator)
            if applied:
                left_tensor = self.tensors[cut - 1]
                transformed_left = np.tensordot(left_tensor, gauge, axes=(1, 0))
                self.tensors[cut - 1] = np.moveaxis(transformed_left, -1, 1)
                right_tensor = self.tensors[cut]
                transformed_right = np.linalg.solve(
                    gauge,
                    right_tensor.reshape(dimension, -1),
                )
                self.tensors[cut] = transformed_right.reshape(right_tensor.shape)
                self._apply_local_masks()
                inverse = np.linalg.inv(gauge)
                balanced_left = self._hermitian_part(gauge.conj().T @ left @ gauge)
                balanced_right = self._hermitian_part(
                    inverse @ right @ inverse.conj().T
                )
                values = np.linalg.eigvalsh(balanced_left)
                positive = values[values > relative_floor * max(float(values[-1]), tiny)]
                balanced_condition = (
                    float(positive[-1] / positive[0]) if positive.size else float("inf")
                )
                balanced_denominator = max(
                    float(np.linalg.norm(balanced_left) + np.linalg.norm(balanced_right)),
                    tiny,
                )
                imbalance_after = float(
                    2.0 * np.linalg.norm(balanced_left - balanced_right) / balanced_denominator
                )
                message = "sector-balanced"
            else:
                balanced_condition = float("inf")
                imbalance_after = imbalance_before
                message = "; ".join(messages)
            left_values = np.linalg.eigvalsh(left)
            right_values = np.linalg.eigvalsh(right)
            positive_left = left_values[left_values > relative_floor * max(float(left_values[-1]), tiny)]
            positive_right = right_values[right_values > relative_floor * max(float(right_values[-1]), tiny)]
            updates.append(
                FrontierGaugeUpdate(
                    cut=cut,
                    frontier_sites=tuple(self._norm_frontier.frontier_sites[cut]),
                    applied=applied,
                    message=message,
                    left_rank=left_rank,
                    right_rank=right_rank,
                    left_condition=(
                        float(positive_left[-1] / positive_left[0])
                        if positive_left.size
                        else float("inf")
                    ),
                    right_condition=(
                        float(positive_right[-1] / positive_right[0])
                        if positive_right.size
                        else float("inf")
                    ),
                    balanced_condition=balanced_condition,
                    gauge_condition=float(np.linalg.cond(gauge)),
                    imbalance_before=imbalance_before,
                    imbalance_after=imbalance_after,
                )
            )
        return tuple(updates)


def abelian_frontier_tied_letta_from_mps(
    hamiltonian,
    parent_sets,
    mps_tensors,
    *,
    abelian_layout=None,
    local_qns=None,
    bond_qns=None,
    target=None,
    tie_noise=0.0,
    symmetry_atol=1.0e-12,
    seed=None,
    **kwargs,
):
    r"""Lift a charge-resolved dense MPS exactly into graph-tied LETTA.

    MPS cores have ``(left, physical, right)`` ordering and may have
    nonuniform bond dimensions.  ``bond_qns`` labels those dense bond bases,
    including the two unit boundaries.  Equivalently, callers can pass a
    prebuilt :class:`FrontierAbelianLayout`.

    With ``tie_noise=0`` the cores are merely transposed and broadcast over
    every tied-parent configuration, so the MPS state is represented exactly.
    Every core is checked against ``q_right = q_left + q(physical)`` before
    construction; forbidden entries larger than ``symmetry_atol`` are rejected
    rather than silently projected away.
    """
    if "tensors" in kwargs or "bond_dims" in kwargs or "bond_dim" in kwargs:
        raise TypeError(
            "tensors and bond dimensions are determined by the charge-resolved MPS."
        )
    mps_tensors = _validated_mps_tensors(mps_tensors)
    dims = tuple(int(tensor.shape[1]) for tensor in mps_tensors)
    parent_sets = _validated_parent_sets(dims, parent_sets)
    mps_bond_dims = (mps_tensors[0].shape[0],) + tuple(
        tensor.shape[2] for tensor in mps_tensors
    )
    if abelian_layout is None:
        if local_qns is None or bond_qns is None or target is None:
            raise TypeError(
                "supply abelian_layout or all of local_qns, bond_qns, and target."
            )
        abelian_layout = FrontierAbelianLayout(
            tuple(tuple(_as_charge(charge) for charge in site) for site in local_qns),
            tuple(tuple(_as_charge(charge) for charge in bond) for bond in bond_qns),
            _as_charge(target),
        )
    elif any(value is not None for value in (local_qns, bond_qns, target)):
        raise TypeError(
            "local_qns, bond_qns, and target cannot accompany abelian_layout."
        )
    if not isinstance(abelian_layout, FrontierAbelianLayout):
        raise TypeError("abelian_layout must be a FrontierAbelianLayout.")
    if abelian_layout.dims != dims:
        raise ValueError("abelian_layout local dimensions do not match the MPS.")
    if abelian_layout.bond_dims != mps_bond_dims:
        raise ValueError("abelian_layout bond dimensions do not match the MPS cores.")
    tie_noise = float(tie_noise)
    symmetry_atol = float(symmetry_atol)
    if not np.isfinite(tie_noise) or tie_noise < 0.0:
        raise ValueError("tie_noise must be finite and nonnegative.")
    if not np.isfinite(symmetry_atol) or symmetry_atol < 0.0:
        raise ValueError("symmetry_atol must be finite and nonnegative.")

    physical_sites = tuple(
        (site,) + parents for site, parents in enumerate(parent_sets)
    )
    masks = abelian_layout.local_masks(physical_sites)
    rng = np.random.default_rng(seed)
    is_complex = any(np.iscomplexobj(tensor) for tensor in mps_tensors)
    tensors = []
    for site, (core, parents, mask) in enumerate(
        zip(mps_tensors, parent_sets, masks)
    ):
        local = core.transpose(0, 2, 1)
        parent_shape = tuple(dims[parent] for parent in parents)
        tensor = np.broadcast_to(
            local.reshape(local.shape + (1,) * len(parent_shape)),
            local.shape + parent_shape,
        ).copy()
        scale = max(float(np.max(np.abs(tensor), initial=0.0)), 1.0)
        forbidden = float(np.max(np.abs(tensor[~mask]), initial=0.0))
        if forbidden > symmetry_atol * scale:
            raise ValueError(
                f"MPS core {site} has entries outside its Abelian charge support."
            )
        tensor[~mask] = 0
        if tie_noise > 0.0 and np.prod(parent_shape, dtype=int) > 1:
            noise = rng.normal(size=tensor.shape)
            if is_complex:
                noise = (noise + 1.0j * rng.normal(size=tensor.shape)) / np.sqrt(2.0)
            parent_axes = tuple(range(3, tensor.ndim))
            noise -= np.mean(noise, axis=parent_axes, keepdims=True)
            noise = np.where(mask, noise, 0)
            noise_rms = float(np.sqrt(np.mean(np.abs(noise[mask]) ** 2)))
            core_rms = float(np.sqrt(np.mean(np.abs(core) ** 2)))
            if noise_rms > 0.0 and core_rms > 0.0:
                tensor += tie_noise * core_rms * noise / noise_rms
        tensors.append(tensor)
    return AbelianFrontierTiedLETTA(
        hamiltonian,
        dims,
        parent_sets,
        abelian_layout=abelian_layout,
        bond_dims=abelian_layout.bond_dims,
        tensors=tensors,
        seed=seed,
        **kwargs,
    )


__all__ = [
    "AbelianFrontierTiedLETTA",
    "FrontierAbelianLayout",
    "abelian_frontier_tied_letta_from_mps",
]
