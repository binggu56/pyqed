"""Cross-calculation overlaps for direct-reduced SU(2)-NARG states.

The orbital biorthogonalization follows Malmqvist, Int. J. Quantum Chem. 30,
479 (1986), https://doi.org/10.1002/qua.560300404. The reduced-MPS orbital
transformation follows the nonorthogonal MPS state-interaction construction of
Knecht et al., J. Chem. Theory Comput. 12, 5881 (2016),
https://doi.org/10.1021/acs.jctc.6b00889. This module adapts those components to
the conditional truncation tensors stored by SU(2)-NARG; it is not a direct
reproduction of either reference implementation.
"""

from __future__ import annotations

import numpy as np

from pyqed.mps.nonabelian.environment import contract_chain_expectation
from pyqed.mps.nonabelian.mps import MPS
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
from pyqed.mps.nonabelian.states import (
    FullyReducedSpatialOrbitalSite,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.tensor import NonabelianTensor
from pyqed.qchem.mcscf.casci import _prepare_biorthogonal_overlap

from .su2_core import couple_multiplets, local_site_multiplets, local_su2_branches
from .su2_three_site import product_states_by_irrep


def _sector(irrep):
    nelec, j2 = irrep.charge
    return spatial_target_sector(int(nelec), int(j2))


def _bond_qns(truncated):
    return [
        _sector(irrep)
        for irrep in truncated.site.irreps
        for _ in range(int(truncated.site.sector_dim(irrep)))
    ]


def _root_positions(truncated):
    positions = {}
    counts = {}
    for root in truncated.kept_roots:
        position = counts.get(root.irrep, 0)
        counts[root.irrep] = position + 1
        positions[(root.irrep, int(root.local_index))] = position
    return positions


def _site_tensor(data, left_qns, physical_qns, right_qns):
    # NARG's local |up down> state differs by a minus sign from the canonical
    # spatial-orbital Fock basis used by the reduced-MPS Gaussian circuit.
    canonical_data = {
        key: (-np.asarray(block) if int(key[1].charge) == 2 else np.asarray(block))
        for key, block in data.items()
    }
    return NonabelianTensor(
        data=canonical_data,
        qns=[list(left_qns), list(physical_qns), list(right_qns)],
        dirs=[-1, 1, 1],
        metadata={"physical_basis": "fully_reduced_su2"},
    )


def _first_site_tensor():
    physical = FullyReducedSpatialOrbitalSite()
    vacuum = spatial_target_sector(0, 0)
    data = {
        (vacuum, q_phys, q_phys): np.ones((1, 1, 1), dtype=complex)
        for q_phys in physical.qn
    }
    return _site_tensor(data, [vacuum], physical.qn, physical.qn)


def _two_site_source_labels():
    grouped = {}
    for branch in local_su2_branches():
        q_phys = spatial_target_sector(branch.nelec, branch.j2)
        for left in local_site_multiplets():
            q_left = spatial_target_sector(left.nelec, left.j2)
            for coupled in couple_multiplets(left, branch.multiplet):
                irrep_key = (int(coupled.nelec), int(coupled.j2))
                grouped.setdefault(irrep_key, []).append((q_left, q_phys))
    return grouped


def _seed_second_site_tensor(truncated, terminal=None):
    physical = FullyReducedSpatialOrbitalSite()
    source_labels = _two_site_source_labels()
    terminal_sector = None
    terminal_vectors = None
    if terminal is None:
        right_qns = _bond_qns(truncated)
    else:
        terminal_sector, terminal_vectors = terminal
        terminal_vectors = np.asarray(terminal_vectors, dtype=complex)
        if terminal_vectors.ndim == 1:
            terminal_vectors = terminal_vectors[:, None]
        right_qns = [terminal_sector] * terminal_vectors.shape[1]
    data = {}
    for irrep in truncated.site.irreps:
        q_right = _sector(irrep)
        labels = source_labels.get(tuple(int(x) for x in irrep.charge), ())
        transform = np.asarray(truncated.transform.block(irrep, irrep), dtype=complex)
        if terminal is not None:
            if q_right != terminal_sector:
                continue
            transform = transform @ terminal_vectors
        for row, (q_left, q_phys) in enumerate(labels):
            if row >= transform.shape[0]:
                raise ValueError("two-site NARG source labels do not match the seed transform")
            key = (q_left, q_phys, q_right)
            block = data.setdefault(
                key,
                np.zeros((1, 1, transform.shape[1]), dtype=complex),
            )
            block[0, 0, :] = transform[row, :]
        if len(labels) != transform.shape[0]:
            raise ValueError("two-site NARG seed transform has an unexpected source dimension")
    return _site_tensor(data, physical.qn, physical.qn, right_qns)


def _conditional_site_tensor(previous, current):
    physical = FullyReducedSpatialOrbitalSite()
    grouped = product_states_by_irrep(previous)
    left_positions = _root_positions(previous)
    data = {}
    for irrep in current.site.irreps:
        states = grouped.get(irrep, ())
        transform = np.asarray(current.transform.block(irrep, irrep), dtype=complex)
        if len(states) != transform.shape[0]:
            raise ValueError("NARG product-state labels do not match the truncation transform")
        q_right = _sector(irrep)
        d_right = int(current.site.sector_dim(irrep))
        for row, state in enumerate(states):
            q_left = _sector(state.block_irrep)
            q_phys = _sector(state.local_irrep)
            left_position = left_positions[
                (state.block_irrep, int(state.block_local_index))
            ]
            d_left = int(previous.site.sector_dim(state.block_irrep))
            key = (q_left, q_phys, q_right)
            block = data.setdefault(
                key,
                np.zeros((d_left, 1, d_right), dtype=complex),
            )
            block[left_position, 0, :] = transform[row, :]
    return _site_tensor(data, _bond_qns(previous), physical.qn, _bond_qns(current))


def _terminal_site_tensor(previous, target_irrep, root_vectors):
    physical = FullyReducedSpatialOrbitalSite()
    grouped = product_states_by_irrep(
        previous,
        allowed_nelec={int(target_irrep[0])},
    )
    from pyqed.narg.irrep_tensor import Irrep

    irrep = Irrep(tuple(int(x) for x in target_irrep))
    states = grouped.get(irrep, ())
    vectors = np.asarray(root_vectors, dtype=complex)
    if vectors.ndim == 1:
        vectors = vectors[:, None]
    if len(states) != vectors.shape[0]:
        raise ValueError("terminal NARG root does not match its coupled product basis")
    left_positions = _root_positions(previous)
    q_right = _sector(irrep)
    data = {}
    for row, state in enumerate(states):
        q_left = _sector(state.block_irrep)
        q_phys = _sector(state.local_irrep)
        left_position = left_positions[
            (state.block_irrep, int(state.block_local_index))
        ]
        d_left = int(previous.site.sector_dim(state.block_irrep))
        key = (q_left, q_phys, q_right)
        block = data.setdefault(
            key,
            np.zeros((d_left, 1, vectors.shape[1]), dtype=complex),
        )
        block[left_position, 0, :] = vectors[row, :]
    return _site_tensor(
        data,
        _bond_qns(previous),
        physical.qn,
        [q_right] * vectors.shape[1],
    )


def _selected_state_ids(solver, state_ids):
    roots = np.asarray(solver.root_vectors)
    if roots.ndim == 1:
        roots = roots[:, None]
    if state_ids is None:
        ids = list(range(roots.shape[1]))
    elif isinstance(state_ids, int):
        ids = [int(state_ids)]
    else:
        ids = [int(state_id) for state_id in state_ids]
    if not ids:
        raise ValueError("At least one NARG root must be selected for overlap.")
    for state_id in ids:
        if state_id < 0 or state_id >= roots.shape[1]:
            raise IndexError(f"state_id={state_id} is outside the available NARG roots")
    return ids, roots[:, ids]


def narg_reduced_mps_root_batch(solver, state_ids=None):
    """Convert selected roots to one reduced MPS with an open root boundary."""
    if solver.chain is None or solver.root_vectors is None or solver.target_irrep is None:
        raise ValueError("Run SU2-NARG before requesting overlap states.")
    state_ids, root_vectors = _selected_state_ids(solver, state_ids)
    nsites = int(np.asarray(solver.h1e).shape[0])
    required_source_size = 2 if nsites == 2 else nsites - 1
    if required_source_size not in solver.chain.blocks:
        last_source_size = max(int(size) for size in solver.chain.blocks)
        grown_sites = 2 if nsites == 2 else last_source_size + 1
        raise ValueError(
            "SU2-NARG cross overlap requires a chain grown through every active orbital: "
            f"got final_size={grown_sites} for ncas={nsites}."
        )
    first = _first_site_tensor()
    seed = solver.chain.blocks[2].truncated
    target_sector = spatial_target_sector(*solver.target_irrep)
    if nsites == 2:
        sites = [
            first,
            _seed_second_site_tensor(
                seed,
                terminal=(target_sector, root_vectors),
            ),
        ]
    else:
        sites = [first, _seed_second_site_tensor(seed)]
        for size in range(3, nsites):
            sites.append(
                _conditional_site_tensor(
                    solver.chain.blocks[size - 1].truncated,
                    solver.chain.blocks[size].truncated,
                )
            )
        sites.append(
            _terminal_site_tensor(
                solver.chain.blocks[nsites - 1].truncated,
                solver.target_irrep,
                root_vectors,
            )
        )
    state = MPS.from_sites(
        sites,
        center=nsites - 1,
        target_sector=target_sector,
    )
    state.root_ids = tuple(state_ids)
    return state


def _root_state_from_batch(batch, root_index):
    root_index = int(root_index)
    terminal = batch.sites[-1]
    right_qns = list(terminal.qns[2])
    if root_index < 0 or root_index >= len(right_qns):
        raise IndexError(f"root_index={root_index} is outside the batched root boundary")
    terminal_one = NonabelianTensor(
        {
            key: np.asarray(block)[..., root_index : root_index + 1]
            for key, block in terminal.data.items()
        },
        [list(terminal.qns[0]), list(terminal.qns[1]), [right_qns[root_index]]],
        list(terminal.dirs),
        fusion_legs=terminal.fusion_legs,
        metadata=terminal.metadata,
    )
    return MPS.from_sites(
        [*(site.copy() for site in batch.sites[:-1]), terminal_one],
        center=batch.center,
        target_sector=batch.target_sector,
    )


def narg_reduced_mps_states(solver, state_ids=None):
    """Convert selected SU(2)-NARG roots to fully reduced SU(2) MPS states."""
    batch = narg_reduced_mps_root_batch(solver, state_ids)
    return [
        _root_state_from_batch(batch, root_index)
        for root_index in range(len(batch.root_ids))
    ]


def _base_molecule(solver):
    active = getattr(solver, "active_space", None)
    if active is not None:
        return active.base_mol
    mol = getattr(solver, "mol", None)
    return getattr(mol, "base_mol", mol)


def _active_space_shape(solver):
    ncas = int(solver.ncas) if solver.ncas is not None else int(np.asarray(solver.h1e).shape[0])
    ncore = int(solver.ncore or 0)
    return ncore, ncas


def _ordered_mo_coeff(solver):
    ncore, ncas = _active_space_shape(solver)
    if solver.active_space is not None:
        core = np.asarray(solver.mo_core)
        active = np.asarray(solver.mo_cas)
    else:
        coeff = getattr(solver.mf, "mo_coeff", None)
        if coeff is None:
            raise ValueError(
                "Automatic NARG cross overlap needs MO coefficients; pass mo_overlap explicitly."
            )
        coeff = np.asarray(coeff)
        core = coeff[:, :ncore]
        active = coeff[:, ncore : ncore + ncas]
    order = tuple(getattr(solver, "orbital_order", range(ncas)))
    if len(order) != ncas:
        raise ValueError("NARG orbital order is inconsistent with the active space")
    return np.column_stack((core, active[:, order]))


def _ordered_mo_overlap(bra, ket, *, ao_overlap=None, mo_overlap=None):
    bra_ncore, bra_ncas = _active_space_shape(bra)
    ket_ncore, ket_ncas = _active_space_shape(ket)
    if bra_ncore != ket_ncore or bra_ncas != ket_ncas:
        raise ValueError(
            "NARG overlap requires matching active-space sizes: "
            f"(ncore, ncas)=({bra_ncore}, {bra_ncas}) vs "
            f"({ket_ncore}, {ket_ncas})."
        )
    if ao_overlap is not None and mo_overlap is not None:
        raise ValueError("Pass either ao_overlap or mo_overlap, not both.")
    if mo_overlap is None:
        if ao_overlap is None:
            from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix

            ao_overlap = _cross_ao_overlap_matrix(
                _base_molecule(bra),
                _base_molecule(ket),
            )
        left = _ordered_mo_coeff(bra)
        right = _ordered_mo_coeff(ket)
        return left.conj().T @ np.asarray(ao_overlap) @ right

    overlap = np.asarray(mo_overlap, dtype=complex)
    full_dim = bra_ncore + bra_ncas
    if overlap.shape != (full_dim, full_dim):
        if bra_ncore == 0 and overlap.shape == (bra_ncas, bra_ncas):
            full_dim = bra_ncas
        else:
            raise ValueError(
                f"mo_overlap must have shape {(full_dim, full_dim)}, got {overlap.shape}."
            )
    bra_order = np.asarray(
        [*range(bra_ncore), *(bra_ncore + np.asarray(bra.orbital_order, dtype=int))]
    )
    ket_order = np.asarray(
        [*range(ket_ncore), *(ket_ncore + np.asarray(ket.orbital_order, dtype=int))]
    )
    return overlap[np.ix_(bra_order, ket_order)]


def _align_orbital_block(reference, target, ao_overlap, method):
    reference = np.asarray(reference, dtype=complex)
    target = np.asarray(target, dtype=complex)
    if reference.shape[1] != target.shape[1]:
        raise ValueError("Parallel-transport orbital blocks must have equal sizes.")
    size = reference.shape[1]
    if size == 0:
        return target.copy(), np.empty((0, 0), dtype=complex), {
            "singular_values": np.empty(0),
            "offdiagonal_norm_before": 0.0,
            "offdiagonal_norm_after": 0.0,
        }
    overlap = reference.conj().T @ np.asarray(ao_overlap) @ target
    method = str(method).lower().replace("-", "_")
    if method in {"polar", "procrustes"}:
        left, singular_values, right_h = np.linalg.svd(overlap, full_matrices=False)
        rotation = right_h.conj().T @ left.conj().T
    elif method in {"match", "assignment", "phase"}:
        from scipy.optimize import linear_sum_assignment

        rows, columns = linear_sum_assignment(-np.abs(overlap))
        order = columns[np.argsort(rows)]
        rotation = np.zeros((size, size), dtype=complex)
        for row, column in enumerate(order):
            value = overlap[row, column]
            phase = 1.0 if abs(value) == 0.0 else np.conj(value) / abs(value)
            rotation[column, row] = phase
        singular_values = np.linalg.svd(overlap, compute_uv=False)
    else:
        raise ValueError("transport method must be 'polar' or 'match'.")
    aligned = target @ rotation
    aligned_overlap = overlap @ rotation

    def offdiagonal_norm(matrix):
        return float(np.linalg.norm(matrix - np.diag(np.diag(matrix))))

    return aligned, rotation, {
        "singular_values": np.asarray(singular_values),
        "overlap_before": overlap,
        "overlap_after": aligned_overlap,
        "offdiagonal_norm_before": offdiagonal_norm(overlap),
        "offdiagonal_norm_after": offdiagonal_norm(aligned_overlap),
    }


def parallel_transport_narg_orbitals(
    reference,
    target_mf,
    *,
    mo_coeff=None,
    ao_overlap=None,
    method="polar",
    transport_core=True,
    return_info=False,
):
    """Align target MOs to a completed NARG calculation before the next run.

    ``method='polar'`` performs a full unitary Procrustes alignment. The
    localization-preserving ``method='match'`` only permutes and phase-aligns
    orbitals. Core and active spaces are never mixed. This is a PyQED
    adaptation of the polar/biorthogonal orbital alignment used in
    nonorthogonal state-interaction workflows, not a reproduction of a
    particular program.
    """
    if reference.chain is None or reference.mo_cas is None:
        raise ValueError("Run the reference molecular SU2-NARG calculation first.")
    coeff = np.asarray(
        getattr(target_mf, "mo_coeff", None) if mo_coeff is None else mo_coeff,
        dtype=complex,
    )
    if coeff.ndim != 2:
        raise ValueError("Target parallel transport requires a two-dimensional mo_coeff.")
    ncore, ncas = _active_space_shape(reference)
    if coeff.shape[1] < ncore + ncas:
        raise ValueError("Target mo_coeff does not contain the reference core and active spaces.")
    if ao_overlap is None:
        from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix

        target_mol = getattr(target_mf, "mol", None)
        target_mol = getattr(target_mol, "base_mol", target_mol)
        ao_overlap = _cross_ao_overlap_matrix(_base_molecule(reference), target_mol)
    ao_overlap = np.asarray(ao_overlap)

    out = np.array(coeff, copy=True)
    active_slice = slice(ncore, ncore + ncas)
    aligned_active, active_rotation, active_info = _align_orbital_block(
        np.asarray(reference.mo_cas),
        coeff[:, active_slice],
        ao_overlap,
        method,
    )
    out[:, active_slice] = aligned_active
    if transport_core and ncore:
        aligned_core, core_rotation, core_info = _align_orbital_block(
            np.asarray(reference.mo_core),
            coeff[:, :ncore],
            ao_overlap,
            method,
        )
        out[:, :ncore] = aligned_core
    else:
        core_rotation = np.eye(ncore, dtype=complex)
        core_info = None
    info = {
        "method": str(method).lower().replace("-", "_"),
        "ncore": ncore,
        "ncas": ncas,
        "transport_core": bool(transport_core),
        "core_rotation": core_rotation,
        "active_rotation": active_rotation,
        "core": core_info,
        "active": active_info,
    }
    return (out, info) if return_info else out


def narg_overlap_orbital_order(
    bra,
    ket,
    *,
    ao_overlap=None,
    mo_overlap=None,
    exact_limit=18,
    return_info=False,
):
    """Suggest a common active-orbital chain order from overlap cut weights."""
    from pyqed.qchem.orbital_clustering import (
        orbital_boundary_cut_cost,
        order_orbital_clusters,
    )

    full_overlap = _ordered_mo_overlap(
        bra,
        ket,
        ao_overlap=ao_overlap,
        mo_overlap=mo_overlap,
    )
    ncore, ncas = _active_space_shape(bra)
    prep = _prepare_biorthogonal_overlap(
        full_overlap,
        ncore,
        ncore,
        ncas,
        ncas,
        np.dtype(np.result_type(full_overlap, complex)),
    )
    matrix = prep.saa_eff
    graph = np.abs(matrix) ** 2 + np.abs(matrix.T) ** 2
    np.fill_diagonal(graph, 0.0)
    singleton = [(index,) for index in range(ncas)]
    ordered = order_orbital_clusters(
        graph,
        singleton,
        exact_limit=exact_limit,
    )
    order = tuple(block[0] for block in ordered)
    info = {
        "order": order,
        "graph": graph,
        "cut_cost_before": orbital_boundary_cut_cost(graph, singleton),
        "cut_cost_after": orbital_boundary_cut_cost(graph, ordered),
        "exact_ordering": ncas <= int(exact_limit),
    }
    return (order, info) if return_info else order


def _overlap_matrix(bra_states, ket_states):
    out = np.empty((len(bra_states), len(ket_states)), dtype=complex)
    for i, bra_state in enumerate(bra_states):
        for j, ket_state in enumerate(ket_states):
            identity = _identity_mpo_factors_for_sites_and_mpo(
                ket_state.sites,
                [None] * len(ket_state.sites),
            )
            out[i, j] = contract_chain_expectation(
                ket_state.sites,
                identity,
                bra_sites=bra_state.sites,
            )
    return out


def _batched_overlap_matrix(bra_state, ket_state, nbra, nket):
    """Contract an identity MPO while leaving both root boundaries open."""
    from pyqed.mps.nonabelian.environment import (
        _contract_from_left_blocks_rank_coupled,
        _initial_left_env_blocks_rank_coupled,
        _is_rank_coupled_chain,
        _normalize_block_sparse_mpo_factors,
        _tensor_dense_layout,
    )

    identity = _identity_mpo_factors_for_sites_and_mpo(
        ket_state.sites,
        [None] * len(ket_state.sites),
    )
    layouts = [_tensor_dense_layout(site) for site in ket_state.sites]
    factors = _normalize_block_sparse_mpo_factors(identity, site_layouts=layouts)
    if not _is_rank_coupled_chain(factors):
        raise TypeError("Batched SU(2)-NARG overlap requires a fully reduced identity MPO.")
    env = _initial_left_env_blocks_rank_coupled(layouts[0], factors[0])
    for factor, bra_site, ket_site in zip(
        factors,
        bra_state.sites,
        ket_state.sites,
    ):
        env = _contract_from_left_blocks_rank_coupled(
            factor,
            bra_site,
            env,
            ket_site,
        )

    out = np.zeros((int(nbra), int(nket)), dtype=complex)
    for channel_blocks in env.values():
        block = channel_blocks.get(0)
        if block is not None:
            out += np.asarray(block).sum(axis=0)
    return out


def _reduced_bond_dimension(state):
    return max([1] + [len(site.qns[2]) for site in state.sites[:-1]])


def _orbital_map_locality(one_particle_map):
    matrix = np.asarray(one_particle_map, dtype=complex)
    graph = np.abs(matrix) ** 2 + np.abs(matrix.T) ** 2
    np.fill_diagonal(graph, 0.0)
    cut_weights = np.asarray(
        [
            np.sum(graph[:cut, cut:])
            for cut in range(1, matrix.shape[0])
        ],
        dtype=float,
    )
    return {
        "sum_cut_weight": float(np.sum(cut_weights)),
        "max_cut_weight": float(np.max(cut_weights, initial=0.0)),
        "cut_weights": cut_weights,
        "offdiagonal_frobenius_norm": float(
            np.linalg.norm(matrix - np.diag(np.diag(matrix)))
        ),
    }


def _split_maps(prep):
    identity = np.eye(prep.saa_eff.shape[0], dtype=complex)
    return {
        "balanced": (
            np.linalg.inv(prep.x_left),
            np.linalg.inv(prep.x_right),
        ),
        "bra_only": (prep.saa_eff.conj().T, identity),
        "ket_only": (identity, prep.saa_eff),
    }


def _select_orbital_split(
    prep,
    bra_state,
    ket_state,
    requested,
    *,
    condition_limit,
):
    aliases = {
        "bra": "bra_only",
        "left": "bra_only",
        "ket": "ket_only",
        "right": "ket_only",
        "svd": "balanced",
    }
    requested = aliases.get(str(requested).lower().replace("-", "_"), requested)
    requested = str(requested).lower().replace("-", "_")
    maps = _split_maps(prep)
    if requested not in {"auto", *maps}:
        raise ValueError(
            "orbital_split must be 'auto', 'balanced', 'bra_only', or 'ket_only'."
        )

    bra_scale = float(_reduced_bond_dimension(bra_state) ** 3)
    ket_scale = float(_reduced_bond_dimension(ket_state) ** 3)
    diagnostics = {}
    for name, (bra_map, ket_map) in maps.items():
        bra_locality = _orbital_map_locality(bra_map)
        ket_locality = _orbital_map_locality(ket_map)
        diagnostics[name] = {
            "bra": bra_locality,
            "ket": ket_locality,
            "estimated_cost": (
                bra_scale * bra_locality["sum_cut_weight"]
                + ket_scale * ket_locality["sum_cut_weight"]
            ),
        }

    condition = float(np.linalg.cond(prep.saa_eff))
    if requested == "auto":
        allowed = list(maps)
        if not np.isfinite(condition) or condition > float(condition_limit):
            allowed = ["balanced"]
        selected = min(
            allowed,
            key=lambda name: (
                diagnostics[name]["estimated_cost"],
                0 if name == "ket_only" else 1 if name == "bra_only" else 2,
            ),
        )
    else:
        selected = requested
    return selected, maps[selected], {
        "requested": requested,
        "selected": selected,
        "active_overlap_condition_number": condition,
        "condition_limit": float(condition_limit),
        "candidates": diagnostics,
    }


def _identity_transform_info(
    state,
    *,
    cutoff,
    max_bond,
    discarded_weight_budget,
    adaptive_max_bond,
    orbital_blocks=None,
):
    input_bond = _reduced_bond_dimension(state)
    adaptive = (
        isinstance(max_bond, str) and max_bond.lower() == "adaptive"
    )
    return {
        "method": "identity_skip",
        "exact": True,
        "cutoff": float(cutoff),
        "max_bond": max_bond,
        "requested_max_bond": max_bond,
        "adaptive": adaptive,
        "discarded_weight_budget": (
            float(discarded_weight_budget) if adaptive else None
        ),
        "adaptive_max_bond": (
            int(adaptive_max_bond) if adaptive else None
        ),
        "input_reduced_bond_dimension": input_bond,
        "adjacent_gate_count": 0,
        "orbital_block_count": len(orbital_blocks or [range(len(state))]),
        "orbital_factorization": "identity",
        "orbital_block_factorizations": [],
        "unitarity_residual": 0.0,
        "peak_reduced_bond_dimension": input_bond,
        "sum_gate_discarded_weight": 0.0,
        "max_gate_discarded_weight": 0.0,
        "truncated_gate_count": 0,
        "adaptive_budget_satisfied": True if adaptive else None,
        "gate_bonds": [],
        "gate_discarded_weight_budgets": [],
        "gate_kept_reduced_bonds": [],
        "determinant_expansion": False,
        "component_expansion": False,
        "skipped_identity": True,
    }


def _graph_block_orbital_map(one_particle_map, threshold):
    matrix = np.asarray(one_particle_map, dtype=complex)
    threshold = float(threshold)
    if threshold < 0.0:
        raise ValueError("orbital_map_threshold must be non-negative.")
    approximated = np.array(matrix, copy=True)
    if threshold > 0.0:
        offdiagonal = ~np.eye(matrix.shape[0], dtype=bool)
        weak = offdiagonal & (np.abs(approximated) <= threshold)
        approximated[weak] = 0.0
    boundaries = [0]
    for cut in range(1, matrix.shape[0]):
        crossing = max(
            np.max(np.abs(approximated[:cut, cut:]), initial=0.0),
            np.max(np.abs(approximated[cut:, :cut]), initial=0.0),
        )
        if crossing == 0.0:
            boundaries.append(cut)
    boundaries.append(matrix.shape[0])
    blocks = [
        tuple(range(start, stop))
        for start, stop in zip(boundaries[:-1], boundaries[1:])
    ]
    labels = np.empty(matrix.shape[0], dtype=int)
    for label, block in enumerate(blocks):
        labels[np.asarray(block, dtype=int)] = label
    approximated[labels[:, None] != labels[None, :]] = 0.0
    residual = float(np.linalg.norm(matrix - approximated, ord=2))
    if not np.isfinite(np.linalg.cond(approximated)):
        raise np.linalg.LinAlgError(
            "Graph-thresholded orbital map is singular; reduce orbital_map_threshold."
        )
    return approximated, blocks, residual


def _apply_one_particle_map(
    state,
    one_particle_map,
    *,
    cutoff,
    max_bond,
    discarded_weight_budget,
    adaptive_max_bond,
    map_threshold,
):
    applied_map, orbital_blocks, map_residual = _graph_block_orbital_map(
        one_particle_map,
        map_threshold,
    )
    identity = np.eye(len(state), dtype=complex)
    if np.allclose(applied_map, identity, atol=1.0e-14, rtol=1.0e-14):
        info = _identity_transform_info(
            state,
            cutoff=cutoff,
            max_bond=max_bond,
            discarded_weight_budget=discarded_weight_budget,
            adaptive_max_bond=adaptive_max_bond,
            orbital_blocks=orbital_blocks,
        )
        info["orbital_map_residual"] = map_residual
        info["exact"] = map_residual == 0.0
        return state, info, applied_map
    transformed, info = apply_spatial_orbital_transform(
        state,
        applied_map,
        inverse=False,
        orbital_blocks=orbital_blocks,
        cutoff=cutoff,
        max_bond=max_bond,
        discarded_weight_budget=discarded_weight_budget,
        adaptive_max_bond=adaptive_max_bond,
        return_info=True,
    )
    info["orbital_map_residual"] = map_residual
    info["exact"] = info["exact"] and map_residual == 0.0
    return transformed, info, applied_map


def su2_narg_overlap(
    bra,
    ket,
    *,
    ao_overlap=None,
    mo_overlap=None,
    bra_state_ids=None,
    ket_state_ids=None,
    orbital_split="auto",
    split_condition_limit=1.0e8,
    orbital_map_threshold=0.0,
    cutoff=1.0e-10,
    max_bond="auto",
    discarded_weight_budget=1.0e-6,
    adaptive_max_bond=8192,
    return_info=False,
):
    """Return cross-calculation overlaps between SU(2)-NARG roots.

    ``cutoff=0``, ``max_bond=None``, and ``orbital_map_threshold=0`` make the
    orbital transformation exact up to floating-point roundoff. A positive map
    threshold drops weak graph edges and factors disconnected contiguous
    orbital blocks independently; its spectral-norm residual is returned in
    the diagnostics. With ``max_bond="adaptive"``, each gate receives an equal
    share of the remaining ``discarded_weight_budget`` and keeps the smallest
    reduced rank meeting it, up to ``adaptive_max_bond``. The reported sum of
    relative gate errors is a conservative compression diagnostic, not a bound
    on the final overlap error.
    """
    bra_batch = narg_reduced_mps_root_batch(bra, bra_state_ids)
    ket_batch = narg_reduced_mps_root_batch(ket, ket_state_ids)
    nbra = len(bra_batch.root_ids)
    nket = len(ket_batch.root_ids)
    s_mo = _ordered_mo_overlap(
        bra,
        ket,
        ao_overlap=ao_overlap,
        mo_overlap=mo_overlap,
    )
    ncore, ncas = _active_space_shape(bra)
    prep = _prepare_biorthogonal_overlap(
        s_mo,
        ncore,
        ncore,
        ncas,
        ncas,
        np.dtype(np.result_type(s_mo, complex)),
    )

    selected_split, (bra_map, ket_map), split_info = _select_orbital_split(
        prep,
        bra_batch,
        ket_batch,
        orbital_split,
        condition_limit=split_condition_limit,
    )
    transformed_bra, bra_info, applied_bra_map = _apply_one_particle_map(
        bra_batch,
        bra_map,
        cutoff=cutoff,
        max_bond=max_bond,
        discarded_weight_budget=discarded_weight_budget,
        adaptive_max_bond=adaptive_max_bond,
        map_threshold=orbital_map_threshold,
    )
    transformed_ket, ket_info, applied_ket_map = _apply_one_particle_map(
        ket_batch,
        ket_map,
        cutoff=cutoff,
        max_bond=max_bond,
        discarded_weight_budget=discarded_weight_budget,
        adaptive_max_bond=adaptive_max_bond,
        map_threshold=orbital_map_threshold,
    )
    transform_info = {"bra": [bra_info], "ket": [ket_info]}

    value = prep.core_factor * _batched_overlap_matrix(
        transformed_bra,
        transformed_ket,
        nbra,
        nket,
    )
    if not return_info:
        return value
    return value, {
        "backend": "recursive_su2",
        "exact": all(
            info["exact"]
            for side in transform_info.values()
            for info in side
        ),
        "sector_preserving": True,
        "determinant_expansion": False,
        "component_expansion": False,
        "batched_roots": True,
        "root_batch_sizes": {"bra": nbra, "ket": nket},
        "orbital_split": selected_split,
        "orbital_split_diagnostics": split_info,
        "orbital_map_threshold": float(orbital_map_threshold),
        "orbital_map_residual": float(
            np.linalg.norm(
                applied_bra_map.conj().T @ applied_ket_map - prep.saa_eff,
                ord=2,
            )
        ),
        "orbital_transform_calls": sum(
            info["method"] != "identity_skip"
            for info in (bra_info, ket_info)
        ),
        "overlap_contractions": 1,
        "core_factor": prep.core_factor,
        "mo_overlap": s_mo,
        "transforms": transform_info,
    }
