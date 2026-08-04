import numpy as np
import pytest

import pyqed.letta as letta_module
from pyqed.letta import (
    DenseTiedLETTA,
    FrontierTiedLETTA,
    LETTA,
    LocalHamiltonian,
    LocalTerm,
    LocalMPOProduct,
    SectorProjectedLETTA,
    fixed_charge_projector_mpo,
)
from pyqed.letta.block_mpo_frontier import BlockMPOFrontier


DIMS = (2, 2, 2, 2)
CHARGES = ((1, -1),) * 4
# Site 3 is a tied physical leg in every preceding tensor.  The charge
# projector must nevertheless count that unique physical site only once.
PARENTS = ((1, 3), (2, 3), (3,), ())


def _spin_operators():
    sx = 0.5 * np.array([[0.0, 1.0], [1.0, 0.0]])
    sy = 0.5 * np.array([[0.0, -1.0j], [1.0j, 0.0]])
    sz = 0.5 * np.array([[1.0, 0.0], [0.0, -1.0]])
    exchange = (
        np.kron(sx, sx)
        + np.kron(sy, sy)
        + 0.73 * np.kron(sz, sz)
    )
    dm = np.kron(sx, sy) - np.kron(sy, sx)
    return sx, sz, exchange + 0.19 * dm


def _conserving_hamiltonian():
    _sx, sz, bond = _spin_operators()
    return LocalHamiltonian(
        DIMS,
        (
            LocalTerm((0,), 0.11 * sz),
            LocalTerm((2,), -0.07 * sz),
            LocalTerm((0, 1), bond),
            LocalTerm((1, 2), 0.83 * bond),
            LocalTerm((2, 3), -0.41 * bond),
            LocalTerm((0, 3), 0.13 * bond),
        ),
        constant=0.037,
    )


def _dense_charge_projector(dims=DIMS, charges=CHARGES, target=0):
    mask = np.fromiter(
        (
            sum(charges[site][physical] for site, physical in enumerate(config))
            == target
            for config in np.ndindex(*dims)
        ),
        dtype=bool,
        count=int(np.prod(dims)),
    )
    return np.diag(mask.astype(float))


def _projected_state(seed=7):
    hamiltonian = _conserving_hamiltonian()
    unrestricted = FrontierTiedLETTA(
        hamiltonian,
        DIMS,
        PARENTS,
        bond_dim=2,
        seed=seed,
    )
    projected = SectorProjectedLETTA.from_unrestricted(
        unrestricted,
        local_charges=CHARGES,
        target=0,
    )
    return projected, unrestricted


def _dense_local_reference(state):
    dense = DenseTiedLETTA(
        state.hamiltonian.to_dense(),
        state.dims,
        state.parent_sets,
        bond_dim=state.bond_dim,
        tensors=state.tensors,
    )
    # DenseTiedLETTA construction balances gauges.  Restore the exact tensors
    # whose frontier local operators are being checked.
    dense.tensors = [tensor.copy() for tensor in state.tensors]
    return dense


def _projected_dense_energy(state):
    vector = state.state_vector()
    norm = np.vdot(vector, vector)
    return float(np.real(np.vdot(vector, state.hamiltonian.to_dense() @ vector) / norm))


def test_fixed_charge_projector_mpo_matches_dense_projector_and_is_idempotent():
    mpo = fixed_charge_projector_mpo(DIMS, CHARGES, target=0)
    actual = mpo.to_dense()
    expected = _dense_charge_projector()

    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual.T.conj(), actual)
    np.testing.assert_array_equal(actual @ actual, actual)
    assert mpo.bond_dims == (1, 2, 3, 2, 1)
    assert int(np.trace(actual)) == 6


def test_projected_state_norm_energy_and_local_operators_match_dense_reference():
    state, _unrestricted = _projected_state()
    projector = _dense_charge_projector()
    raw = state.raw_state_vector()
    expected_state = projector @ raw
    hamiltonian = state.hamiltonian.to_dense()

    np.testing.assert_allclose(state.state_vector(), expected_state, atol=2.0e-14)
    np.testing.assert_allclose(
        state.norm(),
        np.vdot(expected_state, expected_state).real,
        atol=3.0e-13,
    )
    expected_energy = np.vdot(
        expected_state,
        hamiltonian @ expected_state,
    ) / np.vdot(expected_state, expected_state)
    np.testing.assert_allclose(state.expectation(), expected_energy.real, atol=4.0e-13)

    dense = _dense_local_reference(state)
    for site, tensor in enumerate(state.tensors):
        raw_local_map = dense.local_projector(site)
        projected_local_map = projector @ raw_local_map
        expected_metric = projected_local_map.T.conj() @ projected_local_map
        expected_effective = (
            projected_local_map.T.conj()
            @ hamiltonian
            @ projected_local_map
        )
        metric, effective = state.local_operators(site)
        np.testing.assert_allclose(metric, expected_metric, atol=8.0e-13)
        np.testing.assert_allclose(effective, expected_effective, atol=8.0e-13)

        vector = np.linspace(-0.8, 0.9, tensor.size).astype(complex)
        np.testing.assert_allclose(
            state.metric_action(site, vector),
            expected_metric @ vector,
            atol=8.0e-13,
        )
        np.testing.assert_allclose(
            state.hamiltonian_action(site, vector),
            expected_effective @ vector,
            atol=8.0e-13,
        )


def test_sparse_product_frontier_matches_materialized_projected_mpo():
    state, _unrestricted = _projected_state(seed=9)
    sparse = state._hamiltonian_frontier

    assert isinstance(state.objective_mpo, LocalMPOProduct)
    assert sparse.factorized_mpo
    assert sparse.stored_mpo_elements < sparse.dense_mpo_elements
    assert state.projected_hamiltonian_mpo_diagnostics["representation"] == (
        "renormalized_complementary_operators"
    )

    materialized = state.objective_mpo.materialize()
    dense = BlockMPOFrontier(
        state.dims,
        state.physical_sites,
        tuple(tensor.shape for tensor in state.tensors),
        materialized.tensors,
    )
    sparse_left = sparse.build_left(state.tensors)
    sparse_right = sparse.build_right(state.tensors)
    dense_left = dense.build_left(state.tensors)
    dense_right = dense.build_right(state.tensors)

    np.testing.assert_allclose(
        sparse.scalar(state.tensors),
        dense.scalar(state.tensors),
        atol=8.0e-13,
    )
    for sparse_message, dense_message in zip(sparse_left, dense_left):
        for sparse_block, dense_block in zip(
            sparse_message.blocks,
            dense_message.blocks,
        ):
            np.testing.assert_allclose(sparse_block, dense_block, atol=8.0e-13)
    for sparse_message, dense_message in zip(sparse_right, dense_right):
        for sparse_block, dense_block in zip(
            sparse_message.blocks,
            dense_message.blocks,
        ):
            np.testing.assert_allclose(sparse_block, dense_block, atol=8.0e-13)

    for site, tensor in enumerate(state.tensors):
        np.testing.assert_allclose(
            sparse.hole_matrix(
                site,
                sparse_left[site],
                sparse_right[site + 1],
            ),
            dense.hole_matrix(
                site,
                dense_left[site],
                dense_right[site + 1],
            ),
            atol=8.0e-13,
        )
        vector = np.linspace(-0.7, 0.8, tensor.size)
        np.testing.assert_allclose(
            sparse.hole_action(
                site,
                sparse_left[site],
                sparse_right[site + 1],
                vector,
            ),
            dense.hole_action(
                site,
                dense_left[site],
                dense_right[site + 1],
                vector,
            ),
            atol=8.0e-13,
        )


def test_projected_construction_never_materializes_the_mpo_product(monkeypatch):
    hamiltonian = _conserving_hamiltonian()
    unrestricted = FrontierTiedLETTA(
        hamiltonian,
        DIMS,
        PARENTS,
        bond_dim=2,
        seed=10,
    )

    def fail_materialization(*_args, **_kwargs):
        raise AssertionError("exact projected construction fused the MPO product")

    monkeypatch.setattr(type(hamiltonian.to_mpo()), "compose", fail_materialization)
    state = SectorProjectedLETTA.from_unrestricted(
        unrestricted,
        local_charges=CHARGES,
        target=0,
    )

    assert state._hamiltonian_frontier.factorized_mpo
    assert state.nparameters == unrestricted.nparameters


def test_projected_pair_operators_match_a_materialized_product_frontier():
    state, _unrestricted = _projected_state(seed=12)
    reference = FrontierTiedLETTA(
        state.hamiltonian,
        state.dims,
        state.parent_sets,
        bond_dims=state.bond_dims,
        tensors=[tensor.copy() for tensor in state.tensors],
        frontier_backend="identity_block",
        _norm_mpo=state.norm_mpo,
        _objective_mpo=state.objective_mpo.materialize(),
        _balance_initial_gauges=False,
    )

    sparse_metric, sparse_hamiltonian = state.pair_local_operators(1)
    dense_metric, dense_hamiltonian = reference.pair_local_operators(1)

    np.testing.assert_allclose(sparse_metric, dense_metric, atol=1.0e-12)
    np.testing.assert_allclose(
        sparse_hamiltonian,
        dense_hamiltonian,
        atol=1.0e-12,
    )


def test_projected_pair_blocks_contract_one_hermitian_orientation(monkeypatch):
    state, _unrestricted = _projected_state(seed=12)
    plan = state._pair_plan(1)
    original = plan.hamiltonian_engine.hole_blocks
    requested = []

    def record_requests(site, left, right, configuration_pairs):
        requested.extend(configuration_pairs)
        return original(site, left, right, configuration_pairs)

    monkeypatch.setattr(
        plan.hamiltonian_engine,
        "hole_blocks",
        record_requests,
    )
    problem = state.pair_local_block_problem(1)
    metric, effective = state.pair_local_operators(1)

    assert requested
    assert all(row <= column for row, column, _bra, _ket in requested)
    np.testing.assert_allclose(problem.metric.to_dense(), metric, atol=1.0e-12)
    np.testing.assert_allclose(
        problem.hamiltonian.to_dense(),
        effective,
        atol=1.0e-12,
    )


def test_sector_projection_keeps_every_unrestricted_tensor_parameter():
    state, unrestricted = _projected_state(seed=11)

    assert not hasattr(state, "local_masks")
    assert state.nparameters == unrestricted.nparameters
    assert state.dense_nparameters == unrestricted.nparameters
    assert state.local_support_sizes() == tuple(
        (tensor.size, tensor.size) for tensor in state.tensors
    )
    for projected_tensor, unrestricted_tensor in zip(
        state.tensors,
        unrestricted.tensors,
    ):
        np.testing.assert_array_equal(projected_tensor, unrestricted_tensor)

    raw = state.raw_state_vector()
    projected = state.state_vector()
    assert np.count_nonzero(raw) == raw.size
    assert np.count_nonzero(projected) == 6
    # Repeated appearances of site 3 in tied tensors must not change the
    # six-dimensional total-charge-zero physical sector.
    np.testing.assert_array_equal(projected != 0, np.diag(_dense_charge_projector()) != 0)


def test_projected_one_site_update_is_variational_and_matches_dense_energy():
    state, _unrestricted = _projected_state(seed=13)
    shapes_before = tuple(tensor.shape for tensor in state.tensors)
    parameters_before = state.nparameters
    energy_before = state.expectation()

    update = state.optimize_site(1, solver="direct")

    assert update.accepted
    assert update.energy <= energy_before + 2.0e-12
    assert tuple(tensor.shape for tensor in state.tensors) == shapes_before
    assert state.nparameters == parameters_before
    assert "exact local energy check" in update.message
    np.testing.assert_allclose(
        state.expectation(),
        _projected_dense_energy(state),
        atol=8.0e-13,
    )
    np.testing.assert_allclose(update.energy, state.expectation(), atol=8.0e-13)


def test_projected_pair_update_preserves_representable_metric_null_component(
    monkeypatch,
):
    state, _unrestricted = _projected_state(seed=23)
    site = 1
    plan = state._pair_plan(site)
    block_problem = state.pair_local_block_problem(site)
    metric = block_problem.metric.to_dense()
    metric_blocks = tuple(
        block_problem.metric.blocks[(block, block)]
        for block in range(block_problem.layout.nblocks)
    )
    for variable in ("left", "right"):
        design = state._pair_factor_design(
            site,
            plan.union_sites,
            state.tensors[site],
            state.tensors[site + 1],
            variable=variable,
        )
        factor_metric = state._pair_factor_metric_operator(
            site,
            plan.union_sites,
            state.tensors[site],
            state.tensors[site + 1],
            metric_blocks,
            variable=variable,
        )
        np.testing.assert_allclose(
            factor_metric.to_dense(),
            design.T.conj() @ metric @ design,
            atol=8.0e-13,
        )
    values, vectors = np.linalg.eigh(metric)
    scale = max(
        float(np.linalg.norm(metric, ord=np.inf)),
        float(np.max(np.abs(values), initial=0.0)),
    )
    null_vectors = vectors[
        :,
        values <= 64.0 * np.finfo(float).eps * scale,
    ]
    old_merged = state._merge_pair_factors(
        site,
        plan.union_sites,
        state.tensors[site],
        state.tensors[site + 1],
    ).reshape(-1)
    old_null = null_vectors.T.conj() @ old_merged
    assert null_vectors.shape[1] > 0
    assert np.linalg.norm(old_null) > 1.0e-3
    shapes_before = tuple(tensor.shape for tensor in state.tensors)
    parameters_before = state.nparameters
    energy_before = state.expectation()

    def fail_dense_pair(*_args, **_kwargs):
        raise AssertionError("the block pair path materialized dense pair operators")

    monkeypatch.setattr(state, "pair_local_operators", fail_dense_pair)
    update = state.optimize_two_sites(
        site,
        pair_operator_backend="block",
        split_strategy="variational",
        factor_solver="matrix_free",
        metric_support="numerical",
        outer_cycles=1,
        split_metric_sweeps=2,
        split_variational_sweeps=2,
        verify_global=True,
    )

    new_merged = state._merge_pair_factors(
        site,
        plan.union_sites,
        state.tensors[site],
        state.tensors[site + 1],
    ).reshape(-1)
    assert update.accepted
    assert update.energy < energy_before
    assert update.pair_operator_backend == "block"
    assert update.factor_sweeps > 0
    assert update.factor_accepted_updates > 0
    assert update.relative_truncation_error < 1.0e-13
    assert tuple(tensor.shape for tensor in state.tensors) == shapes_before
    assert state.nparameters == parameters_before
    np.testing.assert_allclose(
        null_vectors.T.conj() @ new_merged,
        old_null,
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        state.expectation(),
        _projected_dense_energy(state),
        atol=8.0e-13,
    )


def test_projected_pair_dense_metric_eigensystem_reuses_null_completion(
    monkeypatch,
):
    state, _unrestricted = _projected_state(seed=23)
    site = 1
    metric, _effective = state.pair_local_operators(site)
    eigensystem = state._dense_pair_metric_eigensystem(metric)
    old_vector = state._merged_pair_tensor(site)[0].reshape(-1)
    vector = np.linspace(-0.7, 0.9, old_vector.size)
    reference = state._complete_pair_metric_solution(
        old_vector,
        vector,
        metric=metric,
    )
    assert reference is not None

    def forbidden_eigh(*_args, **_kwargs):
        raise AssertionError("cached pair completion diagonalized the metric")

    monkeypatch.setattr(np.linalg, "eigh", forbidden_eigh)
    completed = state._complete_pair_metric_solution(
        old_vector,
        vector,
        metric=metric,
        metric_eigensystem=eigensystem,
    )

    np.testing.assert_allclose(completed, reference, atol=2.0e-13)
    values, vectors = eigensystem
    scale = max(
        float(np.linalg.norm(metric, ord=np.inf)),
        float(np.max(np.abs(values), initial=0.0)),
    )
    null_vectors = vectors[
        :,
        values <= 64.0 * np.finfo(float).eps * scale,
    ]
    assert null_vectors.shape[1] > 0
    np.testing.assert_allclose(
        null_vectors.T.conj() @ completed,
        null_vectors.T.conj() @ old_vector,
        atol=2.0e-13,
    )


def test_projected_pair_completion_preserves_only_true_metric_null_space():
    state, _unrestricted = _projected_state(seed=23)
    metric = np.diag([2.0, 5.0e-14, 0.0])
    old_vector = np.array([1.0, 3.0, 7.0])
    vector = np.array([2.0, 5.0, 11.0])
    eigensystem = state._dense_pair_metric_eigensystem(metric)

    completed = state._complete_pair_metric_solution(
        old_vector,
        vector,
        metric=metric,
        metric_eigensystem=eigensystem,
    )

    np.testing.assert_allclose(completed, [2.0, 5.0, 7.0], atol=2.0e-13)


def test_projected_pair_full_rank_completion_is_a_noop():
    state, _unrestricted = _projected_state(seed=23)
    metric = np.diag([2.0, 0.5, 0.1])
    eigensystem = state._dense_pair_metric_eigensystem(metric)

    completed = state._complete_pair_metric_solution(
        np.array([1.0, 3.0, 7.0]),
        np.array([2.0, 5.0, 11.0]),
        metric=metric,
        metric_eigensystem=eigensystem,
    )

    assert completed is None


def test_projected_dense_pair_update_factorizes_metric_once(monkeypatch):
    state, _unrestricted = _projected_state(seed=29)
    original_eigensystem = state._dense_pair_metric_eigensystem
    original_completion = state._complete_pair_metric_solution
    eigensystems = []
    completion_eigensystems = []

    def record_eigensystem(metric):
        eigensystem = original_eigensystem(metric)
        eigensystems.append(eigensystem)
        return eigensystem

    def record_completion(old_vector, vector, **kwargs):
        eigensystem = kwargs.get("metric_eigensystem")
        if eigensystem is not None:
            completion_eigensystems.append(eigensystem)
        return original_completion(old_vector, vector, **kwargs)

    monkeypatch.setattr(
        state,
        "_dense_pair_metric_eigensystem",
        record_eigensystem,
    )
    monkeypatch.setattr(
        state,
        "_complete_pair_metric_solution",
        record_completion,
    )
    update = state.optimize_two_sites(
        1,
        pair_operator_backend="dense",
        factor_solver="dense",
        metric_support="numerical",
        outer_cycles=1,
        split_metric_sweeps=0,
        split_variational_sweeps=0,
        split_random_starts=0,
        verify_global=True,
    )

    assert update.accepted
    assert len(eigensystems) == 1
    assert completion_eigensystems
    assert all(
        eigensystem is eigensystems[0]
        for eigensystem in completion_eigensystems
    )


def test_letta_factory_constructs_projected_u1_without_reduced_layout():
    hamiltonian = _conserving_hamiltonian()
    state = LETTA(
        hamiltonian,
        parents=PARENTS,
        symmetry="u1",
        charges=CHARGES,
        target=0,
        bond_dim=2,
        seed=17,
    )

    assert isinstance(state, SectorProjectedLETTA)
    assert state.symmetry == "u1"
    assert state.projection.target == (0,)
    assert state.projection.max_mpo_bond == 3
    assert state.dense_nparameters == sum(tensor.size for tensor in state.tensors)
    assert not hasattr(state, "abelian_layout")
    np.testing.assert_allclose(
        state.expectation(),
        _projected_dense_energy(state),
        atol=8.0e-13,
    )


def test_reduced_graph_u1_routes_are_not_available():
    hamiltonian = _conserving_hamiltonian()

    with pytest.raises(ValueError, match="None or 'u1'"):
        LETTA(
            hamiltonian,
            parents=PARENTS,
            symmetry="reduced_u1",
            charges=CHARGES,
        )
    for keyword in ("layout", "symmetry_layout", "abelian_layout"):
        with pytest.raises(TypeError, match="removed locally masked"):
            LETTA(
                hamiltonian,
                parents=PARENTS,
                symmetry="u1",
                charges=CHARGES,
                **{keyword: object()},
            )
    for name in (
        "AbelianFrontierTiedLETTA",
        "FrontierAbelianLayout",
        "ReducedSymmetricLETTA",
        "SymmetricLETTA",
        "SymmetryLayout",
        "abelian_frontier_tied_letta_from_mps",
    ):
        assert not hasattr(letta_module, name)


def test_sector_projected_letta_rejects_a_nonconserving_hamiltonian():
    sx, _sz, _bond = _spin_operators()
    hamiltonian = LocalHamiltonian(
        DIMS,
        (LocalTerm((1,), 0.2 * sx),),
    )

    with pytest.raises(ValueError, match="does not conserve charge"):
        SectorProjectedLETTA(
            hamiltonian,
            DIMS,
            PARENTS,
            local_charges=CHARGES,
            target=0,
            bond_dim=2,
            seed=3,
        )


def test_copy_and_bond_rebuild_preserve_the_exact_sector_projector():
    state, _unrestricted = _projected_state(seed=19)
    copied = state.copy()

    assert isinstance(copied, SectorProjectedLETTA)
    assert copied.symmetry == state.symmetry
    assert copied.projection == state.projection
    assert copied._hamiltonian_frontier.factorized_mpo
    np.testing.assert_array_equal(
        copied.norm_mpo.to_dense(),
        state.norm_mpo.to_dense(),
    )
    np.testing.assert_allclose(
        copied.objective_mpo.to_dense(),
        state.objective_mpo.to_dense(),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(copied.state_vector(), state.state_vector(), atol=2.0e-14)
    np.testing.assert_allclose(copied.expectation(), state.expectation(), atol=5.0e-13)

    vector_before = copied.state_vector()
    energy_before = copied.expectation()
    update = copied.expand_bond(2, 3, strategy="zero")

    assert update.new_dimension == 3
    assert copied.symmetry == "u1"
    assert copied.projection == state.projection
    assert copied.bond_dims[2] == 3
    assert copied._hamiltonian_frontier.factorized_mpo
    assert copied.nparameters == sum(tensor.size for tensor in copied.tensors)
    assert copied.local_support_sizes() == tuple(
        (tensor.size, tensor.size) for tensor in copied.tensors
    )
    np.testing.assert_array_equal(
        copied.norm_mpo.to_dense(),
        _dense_charge_projector(),
    )
    np.testing.assert_allclose(
        copied.objective_mpo.to_dense(),
        copied.hamiltonian.to_dense() @ _dense_charge_projector(),
        atol=2.0e-13,
    )
    np.testing.assert_allclose(copied.state_vector(), vector_before, atol=3.0e-14)
    np.testing.assert_allclose(copied.expectation(), energy_before, atol=8.0e-13)
