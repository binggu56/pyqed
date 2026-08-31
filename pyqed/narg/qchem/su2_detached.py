"""Detached conditional frames in reduced U(1) x SU(2) multiplicity space."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from pyqed.narg.irrep_tensor import Leg, IrrepTensor, OpIrrep

from .su2_core import local_su2_branches, su2_product_symmetry
from .su2_three_site import coupled_product_states
from .su2_two_site import SectorRoot, TruncatedSU2NARG


def _orthonormal_columns(matrix, *, atol=1.0e-12):
    """Incremental QR without forming or diagonalizing a wide Gram matrix."""
    matrix = np.asarray(matrix)
    if matrix.size == 0:
        return np.zeros((matrix.shape[0], 0), dtype=np.result_type(matrix, complex))
    columns = []
    scale = max(float(np.linalg.norm(matrix, axis=0).max(initial=0.0)), 1.0)
    threshold = max(float(atol), np.finfo(float).eps * max(matrix.shape)) * scale
    for column in matrix.T:
        vector = np.asarray(column, dtype=np.result_type(matrix, complex)).copy()
        if columns:
            basis = np.column_stack(columns)
            for _ in range(2):
                vector -= basis @ (basis.conj().T @ vector)
        norm = float(np.linalg.norm(vector))
        if norm > threshold:
            columns.append(vector / norm)
    if not columns:
        return np.zeros((matrix.shape[0], 0), dtype=np.result_type(matrix, complex))
    return np.column_stack(columns)


def _lowest_subspace(
    hamiltonian,
    rank,
    *,
    constraints=None,
    tol=1.0e-10,
    maxiter=400,
):
    r"""Minimize a rank-``rank`` Rayleigh trace using only small Rayleigh solves.

    The ambient Hamiltonian may be much larger than ``rank``.  It is only
    applied to tall vectors.  Every eigendecomposition below has order at most
    ``rank``; orthogonality is restored by incremental QR.
    """
    hamiltonian = np.asarray(hamiltonian)
    ambient = int(hamiltonian.shape[0])
    if hamiltonian.shape != (ambient, ambient):
        raise ValueError("conditional Hamiltonian must be square")
    constraints = (
        np.zeros((ambient, 0), dtype=np.result_type(hamiltonian, complex))
        if constraints is None
        else _orthonormal_columns(constraints)
    )
    rank = min(int(rank), ambient - constraints.shape[1])
    if rank <= 0:
        return np.zeros(0), np.zeros((ambient, 0), dtype=complex), 0.0, 0

    diagonal_order = np.argsort(np.real(np.diag(hamiltonian)))
    trial = np.eye(ambient, dtype=np.result_type(hamiltonian, complex))[:, diagonal_order]
    if constraints.shape[1]:
        trial -= constraints @ (constraints.conj().T @ trial)
    vectors = _orthonormal_columns(trial)[:, :rank]
    if vectors.shape[1] != rank:
        raise RuntimeError("conditional constraints leave too few independent directions")

    diagonal = np.real(np.diag(hamiltonian))
    off_diagonal = np.sum(np.abs(hamiltonian), axis=1) - np.abs(diagonal)
    upper = float(np.max(diagonal + off_diagonal, initial=1.0))
    lower = float(np.min(diagonal - off_diagonal, initial=-1.0))
    shift = upper + 0.05 * max(upper - lower, 1.0)
    residual_norm = np.inf
    iteration = 0
    for iteration in range(1, int(maxiter) + 1):
        applied = hamiltonian @ vectors
        rayleigh = vectors.conj().T @ applied
        values, rotation = np.linalg.eigh(0.5 * (rayleigh + rayleigh.conj().T))
        vectors = vectors @ rotation
        applied = applied @ rotation
        residual = applied - vectors * values[None, :]
        if constraints.shape[1]:
            residual -= constraints @ (constraints.conj().T @ residual)
        residual -= vectors @ (vectors.conj().T @ residual)
        residual_norm = float(np.linalg.norm(residual, ord="fro"))
        if residual_norm <= float(tol) * max(1.0, float(np.linalg.norm(values))):
            break
        candidate = shift * vectors - applied
        if constraints.shape[1]:
            candidate -= constraints @ (constraints.conj().T @ candidate)
        candidate = _orthonormal_columns(candidate)
        if candidate.shape[1] != rank:
            raise RuntimeError("conditional subspace iteration lost rank")
        vectors = candidate

    applied = hamiltonian @ vectors
    rayleigh = vectors.conj().T @ applied
    values, rotation = np.linalg.eigh(0.5 * (rayleigh + rayleigh.conj().T))
    vectors = vectors @ rotation
    residual = hamiltonian @ vectors - vectors * values[None, :]
    residual_norm = float(np.linalg.norm(residual, ord="fro"))
    return np.real(values), vectors, residual_norm, iteration


def _product_states_by_irrep(narg):
    grouped = defaultdict(list)
    for state in coupled_product_states(narg.source_block):
        if state.total_irrep in narg.site.dims:
            grouped[state.total_irrep].append(state)
    for irrep in narg.site.irreps:
        if len(grouped[irrep]) != narg.site.sector_dim(irrep):
            raise RuntimeError(
                "reduced product labels do not match the grown SU2 sector dimension"
            )
    return dict(grouped)


def _strict_D_baseline(narg, D):
    """Build the ordinary D-column anchor without an enlarged eigensolve."""
    D = int(D)
    required_irrep = getattr(narg, "_su2_detached_required_irrep", None)
    provisional = {}
    maximum_order = 0
    maximum_ambient = 0
    maximum_residual = 0.0
    if required_irrep is not None:
        if required_irrep not in narg.site.dims:
            raise ValueError(f"detached target sector {required_irrep.charge} is absent")
        irreps = (required_irrep,)
    else:
        irreps = narg.site.irreps
    scores = []
    for irrep in irreps:
        block = narg.hamiltonian.block(irrep, irrep)
        rank = min(D, block.shape[0])
        values, vectors, residual, _iterations = _lowest_subspace(block, rank)
        provisional[irrep] = (values, vectors)
        scores.extend(
            (float(energy), irrep, index)
            for index, energy in enumerate(values)
        )
        maximum_order = max(maximum_order, vectors.shape[1])
        maximum_ambient = max(maximum_ambient, block.shape[0])
        maximum_residual = max(maximum_residual, float(residual))

    selected = defaultdict(list)
    for _energy, irrep, index in sorted(
        scores,
        key=lambda item: (item[0], item[1].charge, item[2]),
    )[:D]:
        selected[irrep].append(index)

    dims = {}
    transforms = {}
    hamiltonians = {}
    bases = {}
    roots = []
    for irrep, indices in selected.items():
        all_values, all_vectors = provisional[irrep]
        values = all_values[indices]
        vectors = all_vectors[:, indices]
        dims[irrep] = vectors.shape[1]
        transforms[(irrep, irrep)] = vectors
        hamiltonians[(irrep, irrep)] = np.diag(values)
        source_basis = narg.bases.get(irrep)
        if source_basis is not None:
            bases[irrep] = source_basis @ vectors
        roots.extend(
            SectorRoot(
                energy=float(energy),
                irrep=irrep,
                local_index=index,
                vector=vectors[:, index].copy(),
            )
            for index, energy in enumerate(values)
        )

    site = Leg(dims, symmetry=su2_product_symmetry())
    baseline = TruncatedSU2NARG(
        source=narg,
        kept_roots=sorted(
            roots,
            key=lambda root: (root.energy, root.irrep.charge, root.local_index),
        ),
        site=site,
        bases=bases,
        transform=IrrepTensor(narg.site, site, OpIrrep((0, 0)), transforms),
        hamiltonian=IrrepTensor(site, site, OpIrrep((0, 0)), hamiltonians),
    )
    baseline._su2_maximum_eigensolve_order = maximum_order
    baseline._su2_maximum_ambient_dimension = maximum_ambient
    baseline._su2_residual_norm = maximum_residual
    return baseline


def _conditional_root_data(narg, states_by_irrep, frame_dim):
    branch_names = tuple(branch.name for branch in local_su2_branches())
    candidates = {branch: [] for branch in branch_names}
    maximum_order = 0
    maximum_ambient = 0
    for total_irrep, states in states_by_irrep.items():
        hamiltonian = narg.hamiltonian.block(total_irrep, total_irrep)
        channels = defaultdict(list)
        for row, state in enumerate(states):
            channels[(state.branch, state.block_irrep)].append(row)
        for (branch, block_irrep), rows in channels.items():
            rows = np.asarray(rows, dtype=int)
            block = hamiltonian[np.ix_(rows, rows)]
            values, vectors, _residual, _iterations = _lowest_subspace(
                0.5 * (block + block.conj().T),
                min(int(frame_dim), len(rows)),
            )
            maximum_order = max(maximum_order, vectors.shape[1])
            maximum_ambient = max(maximum_ambient, block.shape[0])
            for column, energy in enumerate(values):
                candidates[branch].append(
                    (
                        float(np.real(energy)),
                        total_irrep,
                        block_irrep,
                        rows,
                        vectors[:, column].copy(),
                    )
                )

    for branch in branch_names:
        candidates[branch] = sorted(
            candidates[branch],
            key=lambda item: (item[0], item[1].charge, item[2].charge),
        )
    if not any(candidates.values()):
        raise ValueError("no feasible SU2 detached-frame anchors were found")
    narg._su2_detached_conditional_maximum_order = maximum_order
    narg._su2_detached_conditional_maximum_ambient = maximum_ambient
    return branch_names, candidates


def _orthogonal_branch_frames(source_site, branch_names, candidates, frame_dim):
    columns = defaultdict(list)
    selected = []
    branch_ranks = []
    for branch in branch_names:
        accepted = 0
        for energy, total_irrep, block_irrep, rows, source_vector in candidates[branch]:
            vector = np.asarray(source_vector, dtype=complex).copy()
            previous = [
                column
                for (other_branch, other_irrep), vectors in columns.items()
                if other_irrep == block_irrep
                for column in vectors
            ]
            if previous:
                basis = np.column_stack(previous)
                for _ in range(2):
                    vector -= basis @ (basis.conj().T @ vector)
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-12:
                continue
            vector /= norm
            columns[(branch, block_irrep)].append(vector)
            selected.append(
                (branch, energy, total_irrep, block_irrep, rows, vector.copy())
            )
            accepted += 1
            if accepted >= int(frame_dim):
                break
        branch_ranks.append(accepted)

    frames = {
        channel: np.column_stack(vectors)
        for channel, vectors in columns.items()
        if vectors
    }
    for (_branch, block_irrep), frame in frames.items():
        if frame.shape[0] != source_site.sector_dim(block_irrep):
            raise RuntimeError("detached frame has an inconsistent old-sector dimension")
    return frames, selected, tuple(branch_ranks)


def _candidate_data(
    narg,
    states_by_irrep,
    frames,
    conditional_roots,
    baseline,
    protect_per_branch,
):
    candidates = {}
    anchor_matrices = {}
    channels_by_irrep = {}
    anchors_by_irrep = defaultdict(list)
    protected_by_branch = defaultdict(int)
    for branch, _energy, total_irrep, _block_irrep, rows, vector in conditional_roots:
        if protected_by_branch[branch] >= int(protect_per_branch):
            continue
        full = np.zeros(
            narg.site.sector_dim(total_irrep),
            dtype=np.result_type(vector, complex),
        )
        full[rows] = vector
        anchors_by_irrep[total_irrep].append(full)
        protected_by_branch[branch] += 1
    if baseline is not None:
        for irrep in baseline.site.irreps:
            block = baseline.transform.blocks.get((irrep, irrep))
            if block is not None:
                anchors_by_irrep[irrep].extend(
                    block[:, column].copy() for column in range(block.shape[1])
                )

    for total_irrep, states in states_by_irrep.items():
        sector_dim = narg.site.sector_dim(total_irrep)
        channels = defaultdict(list)
        for row, state in enumerate(states):
            channels[(state.branch, state.block_irrep)].append((row, state))
        channel_columns = []
        channel_metadata = []
        for (physical_branch, block_irrep), entries in channels.items():
            rows = []
            old_indices = []
            for row, state in entries:
                rows.append(row)
                old_indices.append(state.block_local_index)
            channel_metadata.append(
                (
                    physical_branch,
                    block_irrep,
                    np.asarray(rows, dtype=int),
                    np.asarray(old_indices, dtype=int),
                )
            )
            for (frame_branch, frame_irrep), frame in frames.items():
                if frame_irrep != block_irrep:
                    continue
                block = np.zeros(
                    (sector_dim, frame.shape[1]),
                    dtype=np.result_type(frame, complex),
                )
                for row, state in entries:
                    block[row] = frame[state.block_local_index]
                channel_columns.append(block)
        baseline_block = None
        if baseline is not None:
            baseline_block = baseline.transform.blocks.get((total_irrep, total_irrep))
        if not channel_columns and baseline_block is None:
            continue
        pieces = list(channel_columns)
        if baseline_block is not None:
            pieces.append(baseline_block)
        basis = _orthonormal_columns(np.column_stack(pieces))
        error = float(
            np.linalg.norm(
                basis.conj().T @ basis - np.eye(basis.shape[1]),
                ord=np.inf,
            )
        )
        if error > 1.0e-9:
            raise RuntimeError(
                f"SU2 detached candidate basis is not orthonormal; error={error:.3e}"
            )
        candidates[total_irrep] = basis
        channels_by_irrep[total_irrep] = channel_metadata
        anchor_matrices[total_irrep] = (
            np.column_stack(anchors_by_irrep[total_irrep])
            if anchors_by_irrep[total_irrep]
            else np.zeros((sector_dim, 0), dtype=complex)
        )
    return candidates, anchor_matrices, channels_by_irrep


def _solve_detached_space(
    narg,
    states_by_irrep,
    frames,
    conditional_roots,
    baseline,
    frame_dim,
    chi,
    protect_per_branch,
):
    candidates, anchor_matrices, channels = _candidate_data(
        narg,
        states_by_irrep,
        frames,
        conditional_roots,
        baseline,
        protect_per_branch,
    )
    sector_data = {}
    anchor_rank = 0
    anchor_lowest = np.inf
    anchor_inclusion_error = 0.0
    score_table = []
    for irrep, basis in candidates.items():
        hamiltonian = narg.hamiltonian.block(irrep, irrep)
        projected = basis.conj().T @ (hamiltonian @ basis)
        projected = 0.5 * (projected + projected.conj().T)
        anchor = anchor_matrices[irrep]
        coefficients = basis.conj().T @ anchor
        anchor_inclusion_error = max(
            anchor_inclusion_error,
            float(np.linalg.norm(anchor - basis @ coefficients, ord="fro")),
        )
        anchor_basis = _orthonormal_columns(coefficients)
        anchor_rank += anchor_basis.shape[1]
        if anchor_basis.shape[1]:
            anchor_h = anchor_basis.conj().T @ (projected @ anchor_basis)
            anchor_lowest = min(
                anchor_lowest,
                float(np.min(np.real(np.diag(anchor_h)))),
            )
        sector_data[irrep] = {
            "basis": basis,
            "projected": projected,
            "anchor": anchor,
        }
        score_table.extend(
            (float(np.real(projected[column, column])), irrep, column)
            for column in range(projected.shape[0])
        )

    parent_dim = sum(data["basis"].shape[1] for data in sector_data.values())
    if parent_dim > int(chi):
        raise ValueError(
            "factorized SU2 detached parent exceeds chi: "
            f"required {parent_dim}, received {int(chi)}"
        )

    # ``chi`` bounds the carried tensor parent.  The physical target always has
    # the requested conditional rank D (or the complete parent when smaller).
    target_rank = min(parent_dim, int(frame_dim))
    target_counts = defaultdict(int)
    required_irrep = getattr(narg, "_su2_detached_required_irrep", None)
    if baseline is not None:
        for irrep, rank in baseline.site.dims.items():
            target_counts[irrep] = min(int(rank), sector_data[irrep]["basis"].shape[1])
    elif required_irrep is not None:
        required = sector_data.get(required_irrep)
        if required is None:
            raise ValueError(f"detached target sector {required_irrep.charge} is absent")
        target_counts[required_irrep] = min(target_rank, required["basis"].shape[1])
    else:
        for _score, irrep, _column in sorted(
            score_table,
            key=lambda item: (item[0], item[1].charge, item[2]),
        )[:target_rank]:
            target_counts[irrep] += 1

    parent_dims = {}
    parent_transform_blocks = {}
    parent_hamiltonian_blocks = {}
    parent_bases = {}
    parent_roots = []
    target_dims = {}
    target_transform_blocks = {}
    target_hamiltonian_blocks = {}
    target_bases = {}
    target_roots = []
    target_physical = {}
    target_residual = 0.0
    target_iterations = 0
    lowest = np.inf
    for irrep, data in sector_data.items():
        basis = data["basis"]
        projected = data["projected"]
        parent_dims[irrep] = basis.shape[1]
        parent_transform_blocks[(irrep, irrep)] = basis
        parent_hamiltonian_blocks[(irrep, irrep)] = projected
        source_basis = narg.bases.get(irrep)
        if source_basis is not None:
            parent_bases[irrep] = source_basis @ basis
        identity = np.eye(basis.shape[1], dtype=complex)
        parent_roots.extend(
            SectorRoot(
                energy=float(np.real(projected[column, column])),
                irrep=irrep,
                local_index=column,
                vector=basis[:, column].copy(),
            )
            for column in range(basis.shape[1])
        )

        rank = int(target_counts.get(irrep, 0))
        if rank == 0:
            continue
        values, coefficients, residual, iterations = _lowest_subspace(projected, rank)
        physical = basis @ coefficients
        target_physical[irrep] = physical
        target_dims[irrep] = rank
        target_transform_blocks[(irrep, irrep)] = physical
        target_hamiltonian_blocks[(irrep, irrep)] = np.diag(values)
        if source_basis is not None:
            target_bases[irrep] = source_basis @ physical
        target_residual = max(target_residual, float(residual))
        target_iterations = max(target_iterations, int(iterations))
        lowest = min(lowest, float(values[0]))
        target_roots.extend(
            SectorRoot(
                energy=float(np.real(energy)),
                irrep=irrep,
                local_index=local_index,
                vector=physical[:, local_index].copy(),
            )
            for local_index, energy in enumerate(values)
        )

    if not target_roots:
        raise ValueError("no states remain in the SU2 detached-frame space")
    parent_site = Leg(parent_dims, symmetry=su2_product_symmetry())
    parent = TruncatedSU2NARG(
        source=narg,
        kept_roots=sorted(
            parent_roots,
            key=lambda root: (root.energy, root.irrep.charge, root.local_index),
        ),
        site=parent_site,
        bases=parent_bases,
        transform=IrrepTensor(
            narg.site,
            parent_site,
            OpIrrep((0, 0)),
            parent_transform_blocks,
        ),
        hamiltonian=IrrepTensor(
            parent_site,
            parent_site,
            OpIrrep((0, 0)),
            parent_hamiltonian_blocks,
        ),
    )
    target_site = Leg(target_dims, symmetry=su2_product_symmetry())
    target = TruncatedSU2NARG(
        source=narg,
        kept_roots=sorted(
            target_roots,
            key=lambda root: (root.energy, root.irrep.charge, root.local_index),
        ),
        site=target_site,
        bases=target_bases,
        transform=IrrepTensor(
            narg.site,
            target_site,
            OpIrrep((0, 0)),
            target_transform_blocks,
        ),
        hamiltonian=IrrepTensor(
            target_site,
            target_site,
            OpIrrep((0, 0)),
            target_hamiltonian_blocks,
        ),
    )
    parent._su2_target_truncated = target
    return parent, {
        "anchor_rank": anchor_rank,
        "anchor_inclusion_error": anchor_inclusion_error,
        "retained_anchor_error": 0.0,
        "anchor_lowest_energy": float(anchor_lowest),
        "lowest_energy": float(lowest),
        "detached_improvement": float(anchor_lowest - lowest) if np.isfinite(anchor_lowest) else 0.0,
        "detached_dim": int(sum(basis.shape[1] for basis in candidates.values())),
        "retained_dim": len(parent_roots),
        "target_dim": len(target_roots),
        "parent_dim": len(parent_roots),
        "maximum_eigensolve_order": max(
            max(target_counts.values(), default=0),
            int(getattr(narg, "_su2_detached_conditional_maximum_order", 0)),
            int(getattr(baseline, "_su2_maximum_eigensolve_order", 0)),
        ),
        "maximum_ambient_dimension": max(
            max((data["projected"].shape[0] for data in sector_data.values()), default=0),
            int(getattr(narg, "_su2_detached_conditional_maximum_ambient", 0)),
            int(getattr(baseline, "_su2_maximum_ambient_dimension", 0)),
        ),
        "target_residual_norm": target_residual,
        "target_iterations": target_iterations,
    }, target_physical, channels


def _residual_expansion(narg, states_by_irrep, frames, physical_by_irrep, channels):
    grouped = defaultdict(list)
    residual_sq = 0.0
    state_count = 0
    for irrep, physical in physical_by_irrep.items():
        hamiltonian = narg.hamiltonian.block(irrep, irrep)
        residual = hamiltonian @ physical
        residual -= physical @ (physical.conj().T @ residual)
        residual_sq += float(np.linalg.norm(residual, ord="fro") ** 2)
        state_count += physical.shape[1]
        for _physical_branch, block_irrep, rows, old_indices in channels[irrep]:
            old_dim = narg.source_block.site.sector_dim(block_irrep)
            block_residual = np.zeros(
                (old_dim, physical.shape[1]),
                dtype=np.result_type(residual, complex),
            )
            block_residual[old_indices] = residual[rows]
            frame_columns = [
                frame
                for (_frame_branch, frame_irrep), frame in frames.items()
                if frame_irrep == block_irrep
            ]
            if frame_columns:
                frame_union = np.column_stack(frame_columns)
                block_residual -= frame_union @ (
                    frame_union.conj().T @ block_residual
                )
            if np.linalg.norm(block_residual) > 1.0e-14:
                grouped[block_irrep].append(block_residual)

    additions = []
    for block_irrep, matrices in grouped.items():
        matrix = np.column_stack(matrices)
        ordered = sorted(
            (
                (float(np.linalg.norm(column)), index, column.copy())
                for index, column in enumerate(matrix.T)
            ),
            key=lambda item: (item[0], item[1]),
        )
        basis = []
        for value, _index, vector in reversed(ordered):
            if basis:
                previous = np.column_stack(basis)
                vector -= previous @ (previous.conj().T @ vector)
            norm = float(np.linalg.norm(vector))
            if norm > 1.0e-14:
                vector /= norm
                basis.append(vector)
                additions.append((value, block_irrep, vector.copy()))
    additions.sort(key=lambda item: item[0], reverse=True)
    return np.sqrt(residual_sq / max(state_count, 1)), additions


def detached_frame_truncation(
    narg,
    *,
    frame_dim,
    chi,
    baseline=None,
    adapt_tol=None,
    max_frame_rank=None,
    expand_dim=1,
    protect_per_branch=None,
):
    r"""Construct an SU(2)-equivariant detached-frame truncation.

    Mutually orthogonal frames are formed for every exposed-boundary label
    ``t`` inside each old-parent ``(Ne, j2)`` multiplicity sector.  Every new
    physical local branch ``s`` is then combined with every compatible frame
    ``t``.  This is the reduced SU(2) counterpart of the rolling
    ``|s>|t alpha>`` space.  The returned parent transform is scalar and block
    diagonal in total ``(Ne, j2)``.  The enlarged parent Hamiltonian is never
    diagonalized: it is applied to tall rank-``D`` tensors, and only their
    ``D x D`` Rayleigh matrices are solved.
    When ``baseline`` is supplied, its ordinary
    ``frame_dim``-multiplet space is retained as a mandatory anchor, so the
    detached recursion variationally contains the corresponding regular NARG
    recursion.
    """
    frame_dim = int(frame_dim)
    chi = int(chi)
    expand_dim = int(expand_dim)
    protect_per_branch = (
        0
        if protect_per_branch is None
        else int(protect_per_branch)
    )
    if frame_dim < 1 or chi < 1 or expand_dim < 1:
        raise ValueError("frame_dim, chi, and expand_dim must be positive")
    if protect_per_branch < 0:
        raise ValueError("protect_per_branch must be non-negative")
    if adapt_tol is not None and float(adapt_tol) < 0.0:
        raise ValueError("adapt_tol must be non-negative")

    if baseline is None:
        baseline = _strict_D_baseline(narg, frame_dim)

    states_by_irrep = _product_states_by_irrep(narg)
    branch_names, conditional_candidates = _conditional_root_data(
        narg,
        states_by_irrep,
        frame_dim,
    )
    frames, conditional_roots, branch_ranks = _orthogonal_branch_frames(
        narg.source_block.site,
        branch_names,
        conditional_candidates,
        frame_dim,
    )
    old_dim = sum(narg.source_block.site.dims.values())
    maximum_rank = old_dim
    max_frame_rank = (
        maximum_rank
        if max_frame_rank is None
        else min(int(max_frame_rank), maximum_rank)
    )
    if max_frame_rank < sum(frame.shape[1] for frame in frames.values()):
        raise ValueError("max_frame_rank is smaller than the conditional anchor frame")

    initial_frame_rank = sum(frame.shape[1] for frame in frames.values())
    residual_history = []
    adapt_iterations = 0
    while True:
        truncated, diagnostics, physical, channels = _solve_detached_space(
            narg,
            states_by_irrep,
            frames,
            conditional_roots,
            baseline,
            frame_dim,
            chi,
            protect_per_branch,
        )
        residual_norm, additions = _residual_expansion(
            narg,
            states_by_irrep,
            frames,
            physical,
            channels,
        )
        residual_history.append(float(residual_norm))
        frame_rank = sum(frame.shape[1] for frame in frames.values())
        if (
            adapt_tol is None
            or residual_norm <= float(adapt_tol)
            or frame_rank >= max_frame_rank
            or not additions
        ):
            break

        added = defaultdict(list)
        room = max_frame_rank - frame_rank
        for _weight, block_irrep, vector in additions:
            frame_columns = [
                frame
                for (_branch, frame_irrep), frame in frames.items()
                if frame_irrep == block_irrep
            ]
            if frame_columns:
                frame_union = np.column_stack(frame_columns)
                vector -= frame_union @ (frame_union.conj().T @ vector)
            if added[block_irrep]:
                extra = np.column_stack(added[block_irrep])
                vector -= extra @ (extra.conj().T @ vector)
            norm = float(np.linalg.norm(vector))
            if norm <= 1.0e-12:
                continue
            added[block_irrep].append(vector / norm)
            if sum(len(vectors) for vectors in added.values()) >= min(expand_dim, room):
                break
        if not added:
            break
        for block_irrep, vectors in added.items():
            frame_branch = min(
                branch_names,
                key=lambda branch: sum(
                    frame.shape[1]
                    for (owner, _irrep), frame in frames.items()
                    if owner == branch
                ),
            )
            channel = (frame_branch, block_irrep)
            if channel in frames:
                frames[channel] = np.column_stack((frames[channel], *vectors))
            else:
                frames[channel] = np.column_stack(vectors)
        adapt_iterations += 1

    frame_rank = sum(frame.shape[1] for frame in frames.values())
    frame_unions = defaultdict(list)
    for (_branch, block_irrep), frame in frames.items():
        frame_unions[block_irrep].append(frame)
    orthogonality_error = max(
        (
            float(
                np.linalg.norm(
                    union.conj().T @ union - np.eye(union.shape[1]),
                    ord=np.inf,
                )
            )
            for pieces in frame_unions.values()
            for union in (np.column_stack(pieces),)
        ),
        default=0.0,
    )
    diagnostics.update(
        {
            "frame_dim": frame_dim,
            "chi": chi,
            "baseline_rank": (
                0 if baseline is None else sum(baseline.site.dims.values())
            ),
            "branch_ranks": branch_ranks,
            "protected_per_branch": protect_per_branch,
            "frame_rank": frame_rank,
            "frame_union_rank": frame_rank,
            "cross_product_basis": True,
            "strict_D_rayleigh": True,
            "initial_frame_rank": initial_frame_rank,
            "adapted_rank": frame_rank - initial_frame_rank,
            "adapt_iterations": adapt_iterations,
            "orthogonality_error": orthogonality_error,
            "frame_residual_norm": residual_history[-1],
            "frame_residual_history": tuple(residual_history),
        }
    )
    truncated._su2_detached_diagnostics = diagnostics
    return truncated
