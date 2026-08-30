import numpy as np

from examples.mps.adaptive_cp_letta_j1j2_square import square_j1_j2_bonds
from examples.mps.frontier_tied_letta_j1j2_all_nn import (
    heisenberg_local_hamiltonian,
)
from examples.mps.benchmark_plaquette_frontier_letta_j1j2 import (
    plaquette_frontier_problem,
)
from pyqed.letta import AbelianFrontierTiedLETTA, FrontierAbelianLayout
from pyqed.letta.plaquette_blocks import (
    block_local_hamiltonian,
    block_state_vector,
    blocked_local_charges,
    interplaquette_edges,
    plaquette_site_order,
    remap_site_edges,
    square_plaquette_blocks,
    unblock_state_vector,
    unfused_block_tensor,
)


def _hamiltonian(nrows, ncols, j2=0.5):
    nearest, diagonals = square_j1_j2_bonds(nrows, ncols)
    weighted = tuple((left, right, 1.0) for left, right in nearest)
    weighted += tuple((left, right, j2) for left, right in diagonals)
    return heisenberg_local_hamiltonian(nrows * ncols, weighted)


def test_square_plaquette_blocks_follow_block_snake_and_partition_sites():
    blocks = square_plaquette_blocks(4, 4)

    assert blocks == (
        (0, 1, 7, 6),
        (2, 3, 5, 4),
        (10, 11, 13, 12),
        (8, 9, 15, 14),
    )
    assert sorted(site for block in blocks for site in block) == list(range(16))


def test_interplaquette_ties_use_microscopic_boundary_legs_only():
    blocks = square_plaquette_blocks(4, 4)
    nearest, diagonals = square_j1_j2_bonds(4, 4)
    order = plaquette_site_order(blocks)
    nearest_boundary = interplaquette_edges(blocks, nearest)
    all_boundary = interplaquette_edges(blocks, nearest + diagonals)
    remapped = remap_site_edges(nearest_boundary, order)

    assert order == tuple(site for block in blocks for site in block)
    assert len(nearest_boundary) == 8
    assert len(all_boundary) == 18
    assert len(remapped) == 8
    owner = {
        site: block_index
        for block_index, block in enumerate(blocks)
        for site in block
    }
    assert all(owner[left] != owner[right] for left, right in nearest_boundary)


def test_plaquette_frontier_problem_has_real_ties_and_exact_reordering():
    problem = plaquette_frontier_problem(2, 4, j2=0.5, tie_graph="nearest")
    reference = _hamiltonian(2, 4)
    rng = np.random.default_rng(13)
    vector = rng.normal(size=reference.shape[0])

    expected = block_state_vector(
        reference.matvec(vector),
        reference.dims,
        problem["original_blocks"],
    )
    reordered = block_state_vector(
        vector,
        reference.dims,
        problem["original_blocks"],
    )
    actual = problem["hamiltonian"].matvec(reordered)

    assert problem["tie_edges"]
    assert any(problem["parent_sets"])
    assert problem["hamiltonian"].dims == (2,) * 8
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)


def test_u1_identity_blocks_respect_channel_specific_charge_pair_maps():
    problem = plaquette_frontier_problem(2, 4, j2=0.5, tie_graph="nearest")
    hamiltonian = problem["hamiltonian"]
    layout = FrontierAbelianLayout.spin_half(
        8,
        target_two_sz=0,
        bond_dims=2,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        problem["parent_sets"],
        abelian_layout=layout,
        seed=19,
        frontier_backend="identity_block",
    )
    compressed = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        problem["parent_sets"],
        abelian_layout=layout,
        tensors=state.tensors,
        frontier_backend="compressed",
    )
    vector = state.state_vector()
    dense_energy = float(
        np.real(np.vdot(vector, hamiltonian.matvec(vector)))
        / np.real(np.vdot(vector, vector))
    )

    np.testing.assert_allclose(state.expectation(), dense_energy, atol=2.0e-12)
    np.testing.assert_allclose(
        state.expectation(),
        compressed.expectation(),
        atol=2.0e-12,
    )


def test_blocked_hamiltonian_matvec_is_an_exact_basis_permutation():
    hamiltonian = _hamiltonian(2, 4)
    blocks = square_plaquette_blocks(2, 4)
    blocked = block_local_hamiltonian(hamiltonian, blocks)
    rng = np.random.default_rng(17)
    vector = rng.normal(size=hamiltonian.shape[0]) + 1.0j * rng.normal(
        size=hamiltonian.shape[0]
    )

    expected = block_state_vector(
        hamiltonian.matvec(vector),
        hamiltonian.dims,
        blocks,
    )
    actual = blocked.matvec(
        block_state_vector(vector, hamiltonian.dims, blocks)
    )

    assert blocked.dims == (16, 16)
    np.testing.assert_allclose(actual, expected, atol=2.0e-13)
    np.testing.assert_allclose(
        unblock_state_vector(
            block_state_vector(vector, hamiltonian.dims, blocks),
            hamiltonian.dims,
            blocks,
        ),
        vector,
        atol=0.0,
    )


def test_spin_half_plaquette_charges_and_unfused_tensor_have_four_legs():
    blocks = square_plaquette_blocks(2, 2)
    charges = blocked_local_charges((((1,), (-1,)),) * 4, blocks)
    values, counts = np.unique(
        [charge[0] for charge in charges[0]],
        return_counts=True,
    )

    np.testing.assert_array_equal(values, (-4, -2, 0, 2, 4))
    np.testing.assert_array_equal(counts, (1, 4, 6, 4, 1))
    tensor = np.arange(3 * 5 * 16).reshape(3, 5, 16)
    unfused = unfused_block_tensor(tensor, (2, 2, 2, 2))
    assert unfused.shape == (3, 5, 2, 2, 2, 2)
    assert np.shares_memory(unfused, tensor)


def test_u1_identity_block_frontier_accepts_plaquette_charge_degeneracies():
    microscopic = _hamiltonian(2, 4)
    blocks = square_plaquette_blocks(2, 4)
    hamiltonian = block_local_hamiltonian(microscopic, blocks)
    local_qns = blocked_local_charges((((1,), (-1,)),) * 8, blocks)
    layout = FrontierAbelianLayout.from_local_charges(
        local_qns,
        target=(0,),
        bond_dims=4,
    )
    parent_sets = ((),) * len(blocks)
    blocked = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        abelian_layout=layout,
        seed=23,
        frontier_backend="identity_block",
    )
    compressed = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        parent_sets,
        abelian_layout=layout,
        tensors=blocked.tensors,
        frontier_backend="compressed",
    )

    np.testing.assert_allclose(
        blocked.expectation(),
        compressed.expectation(),
        atol=2.0e-12,
    )


def test_sequential_plaquette_pair_action_matches_fused_mpo_kernel():
    microscopic = _hamiltonian(2, 4)
    blocks = square_plaquette_blocks(2, 4)
    hamiltonian = block_local_hamiltonian(microscopic, blocks)
    local_qns = blocked_local_charges((((1,), (-1,)),) * 8, blocks)
    layout = FrontierAbelianLayout.from_local_charges(
        local_qns,
        target=(0,),
        bond_dims=4,
    )
    state = AbelianFrontierTiedLETTA(
        hamiltonian,
        hamiltonian.dims,
        ((),) * len(blocks),
        abelian_layout=layout,
        seed=31,
        frontier_backend="identity_block",
    )
    plan = state._pair_plan(0)
    environment = state.pair_environment(0)
    rng = np.random.default_rng(41)
    vector = rng.normal(size=int(np.prod(plan.merged_shape)))
    bra = tuple(
        tuple(int(value) for value in row)
        for row in rng.integers(0, 16, size=(12, 2))
    )
    ket = tuple(
        tuple(int(value) for value in row)
        for row in rng.integers(0, 16, size=(12, 2))
    )

    batched = plan.hamiltonian_engine.hole_blocks(
        0,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        bra,
        ket,
    )
    scalar = np.stack(
        [
            plan.hamiltonian_engine.hole_block(
                0,
                environment.hamiltonian_left,
                environment.hamiltonian_right,
                bra_configuration,
                ket_configuration,
            )
            for bra_configuration, ket_configuration in zip(bra, ket)
        ]
    )
    np.testing.assert_allclose(batched, scalar, atol=3.0e-12)

    assert plan.hamiltonian_engine.uses_sequential_physical_kernels
    plan.hamiltonian_engine.physical_kernel = "sequential"
    sequential = state.pair_hamiltonian_action(
        0,
        vector,
        environment=environment,
    )
    saved = plan.hamiltonian_engine._physical_mpo_cores
    plan.hamiltonian_engine._physical_mpo_cores = (None,) * len(saved)
    try:
        fused = state.pair_hamiltonian_action(
            0,
            vector,
            environment=environment,
        )
    finally:
        plan.hamiltonian_engine._physical_mpo_cores = saved

    np.testing.assert_allclose(sequential, fused, atol=3.0e-12)
