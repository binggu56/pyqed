import numpy as np
import pytest

from pyqed.lattice import Site, SpinHalfSite
from pyqed.letta import (
    AbelianFrontierTiedLETTA,
    FrontierLETTA,
    FrontierTiedLETTA,
    NonAbelianFrontierLETTA,
    SU2LETTA,
)
from pyqed.mps.nonabelian import NonabelianTensor, RankCoupledMPO, spatial_target_sector
from pyqed.mps.su2 import SU2Irrep
from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo
from pyqed.symmetry import Leg
from pyqed.tn import Hamiltonian, LocalTerm


def _heisenberg(sites, edges):
    hamiltonian = Hamiltonian(sites)
    for left, right in edges:
        for name in ("Sx", "Sy", "Sz"):
            hamiltonian.add_product(
                1.0,
                (left, name),
                (right, name),
            )
    return hamiltonian


def test_frontier_letta_infers_sites_and_orients_undirected_graph():
    sites = (Site(2),) * 4
    hamiltonian = Hamiltonian(sites)

    state = FrontierLETTA(
        hamiltonian,
        graph=[(3, 0), (2, 1), (0, 3)],
        D=1,
        seed=3,
    )

    assert isinstance(state, FrontierTiedLETTA)
    assert not isinstance(state, AbelianFrontierTiedLETTA)
    assert state.sites == sites
    assert state.graph == ((0, 3), (1, 2))
    assert state.parent_sets == ((3,), (2,), (), ())
    assert state.ordering == (0, 1, 2, 3)
    assert state.target_charge is None


def test_frontier_letta_dispatches_qchem_rank_coupled_mpo_to_su2_backend():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        fully_reduced=True,
        nelec=2,
        spin=0,
    )

    state = FrontierLETTA(hamiltonian, graph=[(0, 1)], D=1, seed=4)

    assert isinstance(state, SU2LETTA)
    assert state.tie == "physical"
    assert state.parent_sets == ((1,), ())
    assert state.target_sector == spatial_target_sector(2, 0)
    state.close()


def test_frontier_letta_exposes_adaptive_su2_multiplet_growth():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        fully_reduced=True,
        nelec=2,
        spin=0,
    )

    state = FrontierLETTA(
        hamiltonian,
        graph=[(0, 1)],
        D=3,
        adaptive_bond=True,
        seed=4,
    )

    assert isinstance(state, SU2LETTA)
    assert state.adaptive_bond
    assert state.D == 3
    assert state.initial_D == 2
    assert set(state.reduced_bond_multiplicities(0).values()) == {2}
    state.close()


def test_frontier_letta_uses_fusion_ties_for_fixed_local_irreps():
    vacuum = spatial_target_sector(0, 0)
    spinor = spatial_target_sector(1, 1)
    singlet_pair = spatial_target_sector(2, 0)
    triplet_pair = spatial_target_sector(2, 2)
    target = spatial_target_sector(3, 1)

    def tensor(data, qns):
        return NonabelianTensor(
            data,
            qns,
            [-1, 1, 1],
            metadata={"physical_basis": "fully_reduced_su2"},
        )

    sites = (
        tensor(
            {(vacuum, spinor, spinor): np.ones((1, 1, 1))},
            [[vacuum], [spinor], [spinor]],
        ),
        tensor(
            {
                (spinor, spinor, singlet_pair): np.ones((1, 1, 1)),
                (spinor, spinor, triplet_pair): np.ones((1, 1, 1)),
            },
            [[spinor], [spinor], [singlet_pair, triplet_pair]],
        ),
        tensor(
            {
                (singlet_pair, spinor, target): np.ones((1, 1, 1)),
                (triplet_pair, spinor, target): np.ones((1, 1, 1)),
            },
            [[singlet_pair, triplet_pair], [spinor], [target]],
        ),
    )
    physical_leg = Leg.from_dims({spinor: 1})
    scalar = SU2Irrep(0)
    identity_core = RankCoupledMPO(
        dense_blocks={(spinor, spinor): np.ones((1, 1, 1, 1))},
        phys_out_leg=physical_leg,
        phys_in_leg=physical_leg,
        left_channel_irreps=(scalar,),
        right_channel_irreps=(scalar,),
    )

    state = FrontierLETTA(
        (identity_core,) * 3,
        sites=sites,
        target_sector=target,
        graph=[(0, 2)],
        D=1,
    )

    assert type(state) is NonAbelianFrontierLETTA
    assert state.tie == "fusion"
    assert state.tie_domains[2] == (singlet_pair, triplet_pair)
    assert state.frontier_states == (1, 2, 2, 1)
    np.testing.assert_allclose(state.expectation(), 1.0, atol=1.0e-12)
    state.run(nsweeps=1, tol=0.0, max_local_parameters=32)
    np.testing.assert_allclose(state.energy, 1.0, atol=1.0e-12)
    assert all(
        update["fully_wigner_eckart_reduced"]
        for update in state.history[0]["updates"]
    )
    state.close()


def test_frontier_letta_infers_default_graph_from_hamiltonian_supports():
    sites = (Site(2),) * 4
    zz = np.diag([1.0, -1.0])
    three_site = np.kron(np.kron(zz, zz), zz)
    hamiltonian = Hamiltonian(
        sites,
        (
            LocalTerm((0, 2), np.kron(zz, zz)),
            LocalTerm((1, 2, 3), three_site),
        ),
    )

    state = FrontierLETTA(hamiltonian, seed=4)

    assert state.graph == ((0, 2), (1, 2), (1, 3), (2, 3))
    assert state.parent_sets == ((2,), (2, 3), (3,), ())


def test_frontier_letta_auto_order_preserves_the_hamiltonian():
    sites = (SpinHalfSite(),) * 4
    edges = ((0, 1), (0, 2), (0, 3))
    hamiltonian = _heisenberg(sites, edges)
    state = FrontierLETTA(
        hamiltonian,
        graph=edges,
        ordering="auto",
        D=1,
        seed=3,
    )

    assert sorted(state.ordering) == list(range(4))
    configurations = tuple(np.ndindex((2,) * 4))
    old_indices = [
        np.ravel_multi_index(
            tuple(configuration[state.inverse_ordering[site]] for site in range(4)),
            (2,) * 4,
        )
        for configuration in configurations
    ]
    expected = hamiltonian.to_dense()[np.ix_(old_indices, old_indices)]
    np.testing.assert_allclose(state.hamiltonian.to_dense(), expected, atol=1.0e-12)


def test_frontier_letta_mixed_precision_search_verifies_in_full_precision():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    exact = FrontierLETTA(
        hamiltonian,
        D=2,
        frontier_backend="identity_block",
        seed=5,
    )
    mixed = FrontierLETTA(
        hamiltonian,
        D=2,
        frontier_backend="identity_block",
        compute_dtype="float32",
        seed=5,
    )
    exact.run(nsweeps=1, solver="matrix_free", gauge=None)
    mixed.run(nsweeps=1, solver="matrix_free", gauge=None)

    assert mixed.compute_dtype == np.dtype("float32")
    assert mixed.history[0]["solver_failures"] == 0
    np.testing.assert_allclose(mixed.energy, exact.energy, atol=1.0e-10)


def test_frontier_letta_adaptive_solver_tightens_recorded_tolerance():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        D=2,
        frontier_backend="identity_block",
        seed=11,
    )
    state.run(
        nsweeps=2,
        tol=0.0,
        solver="matrix_free",
        gauge=None,
        adaptive_solver=True,
        eig_tol_initial=1.0e-4,
        eig_tol=1.0e-10,
    )

    assert state.history[0]["eig_tol"] == pytest.approx(1.0e-4)
    assert 1.0e-10 <= state.history[1]["eig_tol"] <= 1.0e-4


def test_frontier_letta_can_leave_backbone_edges_to_virtual_bonds():
    sites = (SpinHalfSite(),) * 4
    edges = ((0, 1), (0, 2), (1, 2), (2, 3))
    hamiltonian = _heisenberg(sites, edges)

    state = FrontierLETTA(
        hamiltonian,
        graph=edges,
        D=2,
        tie_backbone=False,
        seed=21,
    )

    assert state.graph == ((0, 2),)
    assert state.parent_sets == ((2,), (), (), ())
    assert state.tie_backbone is False


def test_frontier_letta_protected_alias_keeps_exact_norm():
    sites = (SpinHalfSite(),) * 3
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2)))

    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2),),
        D=2,
        frontier_backend="protected",
        max_rank=2,
        rtol=1.0e-8,
        seed=22,
    )

    assert state.frontier_backend == "tensor_train"
    assert state.norm_contraction_is_exact
    assert not state.hamiltonian_contraction_is_exact


def test_frontier_letta_target_charge_selects_abelian_state():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))

    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=2,
        seed=5,
    )

    assert isinstance(state, AbelianFrontierTiedLETTA)
    assert state.sites == sites
    assert state.parent_sets == ((2,), (3,), (), ())
    assert state.abelian_layout.target == (0,)
    assert state.target_charge == {"2sz": 0}
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert np.count_nonzero(tensor[~mask]) == 0


def test_frontier_letta_accepts_physical_sz_units():
    sites = (SpinHalfSite(),) * 2
    hamiltonian = _heisenberg(sites, ((0, 1),))

    state = FrontierLETTA(
        hamiltonian,
        graph=[],
        target_charge={"Sz": 1},
        D=1,
        seed=6,
    )

    assert state.abelian_layout.target == (2,)
    assert state.target_charge == {"2sz": 2}


def test_frontier_letta_rejects_unknown_charge_and_directed_graph():
    sites = (SpinHalfSite(),) * 2
    hamiltonian = _heisenberg(sites, ((0, 1),))

    with pytest.raises(ValueError, match="missing '2sz'"):
        FrontierLETTA(hamiltonian, target_charge={"N": 1})

    class DirectedGraph:
        def is_directed(self):
            return True

    with pytest.raises(ValueError, match="must be undirected"):
        FrontierLETTA(hamiltonian, graph=DirectedGraph())


def test_frontier_letta_adaptive_bond_grows_only_to_D():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=4,
        adaptive_bond=True,
        seed=8,
    )

    assert state.bond_dims == (1, 2, 2, 2, 1)
    state.run(nsweeps=1, tol=0.0)

    assert state.bond_dims == (1, 4, 4, 4, 1)
    assert state.history[0]["bond_dims"] == state.bond_dims
    assert len(state.history[0]["bond_expansions"]) == 3
    assert max(state.bond_dims) <= 4


@pytest.mark.parametrize("frontier_backend", ["compressed", "identity_block", "termwise"])
def test_frontier_letta_amen_bootstraps_bonds_before_sweep(frontier_backend):
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        D=4,
        adaptive_bond=True,
        frontier_backend=frontier_backend,
        chunk_size=2,
        seed=8,
    )
    initial_energy = state.energy

    state.run(
        nsweeps=1,
        tol=0.0,
        enrich="amen",
        enrich_rank=1,
        environment_cache="checkpointed",
    )

    record = state.history[0]
    expansions = record["bond_expansions"]
    assert record["enrich"] == "amen"
    assert [update.cut for update in expansions] == [1, 2, 3]
    assert all(update.strategy == "residual" for update in expansions)
    assert all(update.old_dimension == 2 for update in expansions)
    assert state.bond_dims == (1, 4, 4, 4, 1)
    # Both bonds adjacent to site 1 are available from its first local solve.
    assert record["updates"][1].raw_dim == (
        4 * 4 * int(np.prod(state.tensors[1].shape[2:]))
    )
    assert state.energy < initial_energy


def test_frontier_letta_amen_preserves_u1_layout_and_masks():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=4,
        adaptive_bond=True,
        frontier_backend="termwise",
        chunk_size=2,
        seed=8,
    )

    state.run(nsweeps=1, tol=0.0, enrich="amen", enrich_rank=1)

    assert state.abelian_layout.bond_dims == state.bond_dims
    assert state.history[0]["bond_expansions"]
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert tensor.shape == mask.shape
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_amen_uses_weighted_residual_and_is_backend_stable():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    directions = {}
    for backend in ("compressed", "identity_block", "termwise"):
        state = FrontierLETTA(
            hamiltonian,
            graph=((0, 2), (1, 3)),
            D=3,
            adaptive_bond=True,
            frontier_backend=backend,
            chunk_size=2,
            seed=8,
        )
        site = 1
        environment = state.site_environment(site)
        tensor = state.tensors[site]
        axes = (0, *range(2, tensor.ndim), 1)
        matrix = tensor.transpose(axes).reshape(-1, tensor.shape[1])
        occupied, singular_values, _right = np.linalg.svd(
            matrix,
            full_matrices=False,
        )
        threshold = (
            256.0
            * np.finfo(float).eps
            * max(float(np.max(singular_values, initial=0.0)), 1.0)
        )
        occupied = occupied[:, singular_values > threshold]
        components = tuple(
            state._hamiltonian_frontier.left_enrichment_components(
                site,
                environment.hamiltonian_left,
                tensor.reshape(-1),
            )
        )
        expected_weight = 0.0
        for component in components:
            residual = np.asarray(component)
            residual -= occupied @ (occupied.conj().T @ residual)
            expected_weight += float(np.linalg.norm(residual) ** 2)

        direction, source_norm, count, _discarded = (
            state._amen_enrichment_directions(
                site,
                environment=environment,
                direction="right",
                rank=1,
                rtol=0.0,
            )
        )

        assert count == len(components)
        np.testing.assert_allclose(source_norm**2, expected_weight, rtol=2.0e-13)
        directions[backend] = direction[:, 0] / np.linalg.norm(direction[:, 0])

    assert abs(np.vdot(directions["identity_block"], directions["termwise"])) > 0.995
    assert abs(np.vdot(directions["compressed"], directions["identity_block"])) > 0.95
    assert abs(np.vdot(directions["compressed"], directions["termwise"])) > 0.95


@pytest.mark.parametrize(("site", "direction"), [(1, "right"), (1, "left")])
def test_amen_residual_is_projected_per_shared_tie_configuration(site, direction):
    sites = (SpinHalfSite(),) * 4
    edges = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
    state = FrontierLETTA(
        _heisenberg(sites, edges),
        graph=edges,
        D=2,
        adaptive_bond=True,
        frontier_backend="identity_block",
        seed=18,
    )
    tensor = state.tensors[site]
    if direction == "right":
        axes = (0, *range(2, tensor.ndim), 1)
        occupied_matrix = tensor.transpose(axes).reshape(-1, tensor.shape[1])
    else:
        occupied_matrix = tensor.reshape(tensor.shape[0], -1).T
    rows_by_condition = state._amen_condition_rows(site, direction)

    directions, source_norm, _components, _discarded = (
        state._amen_enrichment_directions(
            site,
            environment=state.site_environment(site),
            direction=direction,
            rank=1,
            rtol=0.0,
        )
    )

    assert len(rows_by_condition) > 1
    direction_columns = directions if direction == "right" else directions.T
    assert direction_columns.shape[1] == 1
    assert source_norm > 0.0
    for rows in rows_by_condition:
        occupied, values, _right = np.linalg.svd(
            occupied_matrix[rows],
            full_matrices=False,
        )
        threshold = (
            256.0
            * np.finfo(float).eps
            * max(float(np.max(values, initial=0.0)), 1.0)
        )
        occupied = occupied[:, values > threshold]
        np.testing.assert_allclose(
            occupied.conj().T @ direction_columns[rows],
            0.0,
            atol=2.0e-13,
        )


@pytest.mark.parametrize("target_charge", [None, {"Sz": 0}])
def test_frontier_letta_amen_qr_expansion_preserves_state(target_charge):
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge=target_charge,
        D=4,
        adaptive_bond=True,
        frontier_backend="identity_block",
        seed=8,
    )
    vector = state.state_vector()
    energy = state.energy

    update, refresh = state._amen_expand_after_site(
        0,
        environment=state.site_environment(0),
        direction="right",
        rank=1,
        rtol=1.0e-7,
        scale=1.0e-3,
    )

    assert update is not None
    assert refresh is None
    assert update.seeded_directions == 1
    np.testing.assert_allclose(state.state_vector(), vector, atol=3.0e-15)
    np.testing.assert_allclose(state.expectation(), energy, atol=3.0e-14)


@pytest.mark.parametrize("target_charge", [None, {"Sz": 0}])
def test_saturated_amen_temporarily_expands_and_retracts_to_cap(target_charge):
    sites = (SpinHalfSite(),) * 4
    edges = ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3))
    hamiltonian = _heisenberg(sites, edges)
    state = FrontierLETTA(
        hamiltonian,
        graph=edges,
        target_charge=target_charge,
        D=2,
        adaptive_bond=True,
        frontier_backend="identity_block",
        seed=12,
    )
    initial_energy = state.energy

    state.run(
        nsweeps=2,
        tol=0.0,
        solver="metric_orthonormal",
        enrich="amen",
        enrich_rank=1,
        enrich_every=1,
        enrich_trigger=None,
    )

    refreshes = [
        refresh
        for record in state.history
        for refresh in record["bond_refreshes"]
    ]
    assert refreshes
    assert all(record["amen_refresh_accepted"] for record in state.history)
    assert not any(
        record["permanent_bond_expansions"] for record in state.history
    )
    assert all(
        refresh.temporary_dimension > refresh.target_dimension
        for refresh in refreshes
    )
    assert any(refresh.overlap_sites for refresh in refreshes)
    assert any(
        refresh.accepted and refresh.subspace_change > 1.0e-8
        for refresh in refreshes
    )
    assert all(dimension <= 2 for dimension in state.bond_dims)
    assert state.energy <= initial_energy + 2.0e-13
    assert all(
        update.solver_converged
        for record in state.history
        for update in record["updates"]
    )
    if target_charge is not None:
        assert state.abelian_layout.bond_dims == state.bond_dims
        for tensor, mask in zip(state.tensors, state.local_masks):
            assert tensor.shape == mask.shape
            np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_saturated_amen_waits_for_stagnation_without_changing_sweep():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    initial = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        D=2,
        adaptive_bond=True,
        frontier_backend="identity_block",
        seed=12,
    )
    ordinary = initial.copy()
    adaptive_amen = initial.copy()

    ordinary.run(nsweeps=1, tol=0.0, solver="metric_orthonormal")
    adaptive_amen.run(
        nsweeps=1,
        tol=0.0,
        solver="metric_orthonormal",
        enrich="amen",
    )

    record = adaptive_amen.history[0]
    assert record["amen_refresh_scheduled"]
    assert not record["amen_refresh_due"]
    assert not record["bond_refreshes"]
    np.testing.assert_allclose(adaptive_amen.energy, ordinary.energy, atol=3.0e-13)
    overlap = np.vdot(
        ordinary.state_vector(normalize=True),
        adaptive_amen.state_vector(normalize=True),
    )
    np.testing.assert_allclose(abs(overlap), 1.0, atol=3.0e-13)


def test_frontier_letta_amen_requires_adaptive_bond_cap():
    sites = (SpinHalfSite(),) * 2
    hamiltonian = _heisenberg(sites, ((0, 1),))
    state = FrontierLETTA(hamiltonian, D=2, seed=3)

    with pytest.raises(ValueError, match="adaptive_bond=True"):
        state.run(nsweeps=1, enrich="amen")


def test_adaptive_sweep_reduces_null_cut_without_immediate_regrowth():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=4,
        adaptive_bond=True,
        seed=10,
    )
    state.expand_bond(2, 4, strategy="zero")

    state.run(nsweeps=1, tol=0.0)

    record = state.history[0]
    assert [update.cut for update in record["bond_reductions"]] == [2]
    assert 2 not in {update.cut for update in record["bond_expansions"]}
    assert state.bond_dims[2] == 2


def test_amen_null_cut_cooldown_allows_reverse_sweep_regrowth():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=4,
        adaptive_bond=True,
        seed=10,
    )
    state.expand_bond(2, 4, strategy="zero")

    state.run(
        nsweeps=2,
        tol=0.0,
        solver="matrix_free",
        enrich="amen",
        enrich_rank=1,
    )

    first, second = state.history
    assert [update.cut for update in first["bond_reductions"]] == [2]
    assert 2 not in {update.cut for update in first["bond_expansions"]}
    assert first["bond_regrowth_cooldown"] == (2,)
    assert 2 in {update.cut for update in second["bond_expansions"]}
    assert state.bond_dims[2] == 4


def test_u1_matrix_free_solver_uses_conditional_metric_blocks():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    initial = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=2,
        seed=18,
    )
    direct = initial.copy()
    iterative = initial.copy()

    direct_update = direct.optimize_site(1, solver="direct")
    iterative_update = iterative.optimize_site(
        1,
        solver="matrix_free",
        eig_tol=1.0e-11,
    )

    assert direct_update.accepted
    assert iterative_update.accepted
    assert iterative_update.solver_metric_is_identity
    assert iterative_update.physical_blocks > 0
    assert iterative_update.stored_operator_elements > 0
    assert iterative_update.stored_operator_elements < 2 * iterative.tensors[1].size**2
    np.testing.assert_allclose(
        iterative_update.energy,
        direct_update.energy,
        atol=2.0e-9,
    )
    for tensor, mask in zip(iterative.tensors, iterative.local_masks):
        np.testing.assert_array_equal(tensor[~mask], 0.0)


def test_termwise_plain_solver_retains_compiled_local_action_plans():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(
        sites,
        ((0, 1), (1, 2), (2, 3), (0, 2), (1, 3)),
    )
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        D=2,
        frontier_backend="termwise",
        chunk_size=2,
        workers=2,
        seed=28,
    )
    frontier = state._hamiltonian_frontier
    for engine in frontier._engines:
        engine.clear_contraction_plans()
    environment = state.site_environment(1)
    probe = state.tensors[1].reshape(-1)
    prepared = frontier.prepare_hole_action(
        1,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
    )
    reference = frontier.hole_action(
        1,
        environment.hamiltonian_left,
        environment.hamiltonian_right,
        probe,
    )
    first_count = frontier.plan_count
    np.testing.assert_allclose(prepared(probe), reference, atol=2.0e-13)
    np.testing.assert_allclose(prepared(probe), reference, atol=2.0e-13)
    batch = np.stack((probe, -0.4 * probe))
    np.testing.assert_allclose(
        prepared.many(batch),
        np.stack((reference, -0.4 * reference)),
        atol=2.0e-13,
    )
    assert first_count > 0
    batched_count = frontier.plan_count
    assert batched_count >= first_count
    prepared.many(batch)
    assert frontier.plan_count == batched_count
    state.close()


def test_u1_null_bond_reduction_preserves_charge_layout_and_state():
    sites = (SpinHalfSite(),) * 4
    hamiltonian = _heisenberg(sites, ((0, 1), (1, 2), (2, 3)))
    state = FrontierLETTA(
        hamiltonian,
        graph=((0, 2), (1, 3)),
        target_charge={"Sz": 0},
        D=2,
        seed=12,
    )
    vector = state.state_vector()
    energy = state.energy
    old_labels = state.abelian_layout.bond_qns[2]
    state.expand_bond(2, 4, strategy="zero")

    reductions = state.reduce_null_bonds()

    assert len(reductions) == 1
    assert reductions[0].cut == 2
    assert reductions[0].new_dimension == 2
    assert state.bond_dims == (1, 2, 2, 2, 1)
    assert state.abelian_layout.bond_qns[2] == old_labels
    for tensor, mask in zip(state.tensors, state.local_masks):
        assert np.count_nonzero(tensor[~mask]) == 0
    np.testing.assert_allclose(state.state_vector(), vector, atol=2.0e-14)
    np.testing.assert_allclose(state.energy, energy, atol=3.0e-14)


def test_frontier_letta_draw_writes_vector_figure(tmp_path):
    sites = (Site(2),) * 4
    state = FrontierLETTA(
        Hamiltonian(sites),
        graph=((0, 2), (1, 3)),
        D=2,
        seed=9,
    )
    path = tmp_path / "frontier-letta.pdf"

    figure, axes = state.draw(path)

    assert path.is_file()
    assert figure.axes == [axes]
    assert len(axes.patches) >= 6
