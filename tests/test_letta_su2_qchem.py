import copy

import numpy as np
import pytest

from pyqed.letta import SU2LETTA
from pyqed.letta.su2_qchem import _ChannelResolvedPairSpace
from pyqed.mps.nonabelian import (
    FullyReducedSpatialOrbitalSite,
    LocalTransitionPlan,
    clebsch_gordan,
    contract_chain_expectation,
    contract_chain_transition,
    expand_rank_coupled_mpo,
    merge_mps_sites,
)
from pyqed.mps.nonabelian.environment import (
    BlockSparseEnvironmentChain,
    DenseEnvironmentChain,
)
from pyqed.mps.nonabelian.solver import (
    _resolve_davidson_operator,
    pack_two_site_state,
)
from pyqed.mps.nonabelian.renormalized import RenormalizedBlockStack
from pyqed.mps.nonabelian.coupling import ordered_two_m_values
from pyqed.mps.nonabelian.states import (
    build_random_reduced_spatial_mps,
    spatial_target_sector,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.su2 import SpatialOrbitalSite
from pyqed.narg.qchem import LETTA
from pyqed.qchem.letta import LETTA as QChemLETTA
from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo


def test_qchem_letta_constructs_su2_state_from_mean_field():
    class Molecule:
        nelec = (1, 1)
        spin = 0

        @staticmethod
        def energy_nuc():
            return 0.7

    class MeanField:
        mol = Molecule()

        @staticmethod
        def get_hcore_mo():
            return np.array([[-1.0, -0.2], [-0.2, 0.5]])

        @staticmethod
        def get_eri_mo():
            return np.zeros((2, 2, 2, 2))

    mf = MeanField()
    state = QChemLETTA(mf, D=2, graph=[(0, 1)], seed=4)

    assert isinstance(state, SU2LETTA)
    assert state.mf is mf
    assert state.ncas == 2
    assert state.nelecas == 2
    assert state.ncore == 0
    assert state.D == 2
    assert state.is_native_su2
    state.run(nsweeps=1, algorithm="one_site", tol=0.0)
    assert np.isfinite(state.energy)
    state.close()


def test_su2_letta_defaults_to_nearest_neighbor_ties_for_dense_integrals():
    nsites = 4
    h1e = np.full((nsites, nsites), -0.05)
    np.fill_diagonal(h1e, np.linspace(-1.0, 0.5, nsites))
    eri = np.full((nsites, nsites, nsites, nsites), 1.0e-3)

    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        D=1,
        seed=4,
    )

    assert state.graph == ((0, 1), (1, 2), (2, 3))
    assert state.frontier_states == (1, 3, 3, 3, 1)
    state.close()


def test_su2_letta_accepts_nearest_neighbor_graph_alias():
    state = SU2LETTA.from_integrals(
        np.diag([-1.0, -0.5, 0.25, 0.75]),
        nelec=4,
        spin=0,
        graph="nn",
        D=1,
        seed=4,
    )

    assert state.graph == ((0, 1), (1, 2), (2, 3))
    assert state.frontier_states == (1, 3, 3, 3, 1)
    state.close()


def _dense_vector_from_reduced_spatial_mps(sites):
    descriptor = SpatialOrbitalSite()
    state_sector = [None] * descriptor.d
    state_two_m = [None] * descriptor.d
    for sector_index, sector in enumerate(descriptor.qn):
        for local_index, state_index in enumerate(descriptor.state_index[sector_index]):
            state_sector[state_index] = sector
            state_two_m[state_index] = ordered_two_m_values(sector.irrep)[local_index]

    vector = np.zeros(descriptor.d ** len(sites), dtype=complex)
    for basis_index in range(vector.size):
        physical_indices = np.unravel_index(basis_index, (descriptor.d,) * len(sites))
        boundary = {(sites[0].qns[0][0], 0, 0): 1.0 + 0.0j}
        for tensor, physical_index in zip(sites, physical_indices):
            q_phys = state_sector[physical_index]
            two_m_phys = state_two_m[physical_index]
            updated = {}
            for (q_left, left_slot, two_m_left), amplitude in boundary.items():
                for (block_left, block_phys, block_right), block in tensor.data.items():
                    if block_left != q_left or block_phys != q_phys:
                        continue
                    for right_slot in range(block.shape[2]):
                        for two_m_right in ordered_two_m_values(block_right.irrep):
                            coefficient = clebsch_gordan(
                                block_left.irrep,
                                block_phys.irrep,
                                block_right.irrep,
                                two_m_left,
                                two_m_phys,
                                two_m_right,
                            )
                            if coefficient:
                                key = (block_right, right_slot, two_m_right)
                                updated[key] = updated.get(key, 0.0) + (
                                    amplitude
                                    * block[left_slot, 0, right_slot]
                                    * coefficient
                                )
            boundary = updated
        target = sites[-1].qns[2][0]
        vector[basis_index] = boundary.get((target, 0, 0), 0.0)
    return vector


def _total_spin_squared(nsites):
    identity = np.eye(4)
    sz = np.diag([0.0, 0.5, -0.5, 0.0])
    plus = np.zeros((4, 4))
    plus[1, 2] = 1.0
    minus = plus.T

    def total(local):
        out = np.zeros((4**nsites, 4**nsites))
        for active in range(nsites):
            factor = np.array([[1.0]])
            for site in range(nsites):
                factor = np.kron(factor, local if site == active else identity)
            out += factor
        return out

    total_sz = total(sz)
    total_plus = total(plus)
    total_minus = total(minus)
    return total_sz @ total_sz + 0.5 * (
        total_plus @ total_minus + total_minus @ total_plus
    )


def test_su2_letta_neutral_ties_embed_reduced_mps_exactly():
    h1e = np.array([[-0.8, -0.15], [-0.15, 0.4]])
    hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        fully_reduced=True,
        n_elec=2,
        spin=0,
    )
    base = build_random_reduced_spatial_mps(
        2,
        target_sector=spatial_target_sector(2, 0),
        bond_multiplicity=2,
        seed=7,
    )
    state = SU2LETTA.from_mps(
        base,
        hamiltonian,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
    )

    identity = _identity_mpo_factors_for_sites_and_mpo(base, state.mpo)
    base_energy = (
        contract_chain_expectation(base, state.mpo)
        / contract_chain_expectation(base, identity)
    )

    assert state.is_native_su2
    assert state.nelec == 2
    assert not hasattr(state, "n_elec")
    assert state.parent_sets == ((1,), ())
    assert state.frontier_states == (1, 3, 1)
    assert all(
        block.shape[1] == 1
        for site in state.materialize()
        for block in site.data.values()
    )
    np.testing.assert_allclose(state.expectation(), base_energy, atol=2.0e-12)


def test_su2_letta_conditional_canonical_gauge_preserves_state():
    h1e = np.array(
        [
            [-0.8, -0.2, 0.0, 0.0],
            [-0.2, -0.3, -0.15, 0.0],
            [0.0, -0.15, 0.2, -0.1],
            [0.0, 0.0, -0.1, 0.7],
        ]
    )
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=1,
        seed=9,
    )
    before = _dense_vector_from_reduced_spatial_mps(state.materialize())
    energy_before = state.energy
    updates = state.canonicalize_conditional_center(0)
    after = _dense_vector_from_reduced_spatial_mps(state.materialize())

    assert state.supports_conditional_canonical_gauge
    assert any(update["applied"] for update in updates)
    assert max(
        update.get("canonical_error", 0.0)
        for update in updates
        if update["applied"]
    ) < 2.0e-12
    np.testing.assert_allclose(after, before, atol=2.0e-12)
    np.testing.assert_allclose(state.expectation(), energy_before, atol=2.0e-12)


def test_su2_letta_conditional_moving_environment_matches_rebuild_path():
    h1e = np.array(
        [
            [-0.8, -0.2, 0.0, 0.0],
            [-0.2, -0.3, -0.15, 0.0],
            [0.0, -0.15, 0.2, -0.1],
            [0.0, 0.0, -0.1, 0.7],
        ]
    )
    initial = SU2LETTA.from_integrals(
        h1e,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=1,
        seed=9,
    )
    rebuilt = copy.deepcopy(initial)
    moving = initial
    rebuilt.run(
        nsweeps=1,
        algorithm="two_site",
        tol=0.0,
        gauge=None,
        reuse_environments=False,
    )
    moving.run(
        nsweeps=1,
        algorithm="two_site",
        tol=0.0,
        gauge="conditional",
        reuse_environments=True,
    )

    np.testing.assert_allclose(moving.energy, rebuilt.energy, atol=2.0e-11)
    assert moving.history[0]["gauge"] == "conditional"
    assert moving.history[0]["environment_reuse"]
    assert (
        moving.history[0]["moving_environment_backend"]
        == "cpp_su2_local_routes_python_boundaries_metric"
    )
    moving_stats = moving.history[0]["moving_environment_stats"]
    assert moving_stats["hamiltonian_boundary_advances"] > 0
    assert moving_stats["su2_boundary_environment"] is None
    assert moving.history[0]["cpp_local_operator_stats"][
        "factor_route_matvec_calls"
    ] > 0
    pair_plans = [
        update.get("solver_info", {})
        .get("davidson", {})
        .get("pair_transition_plan", {})
        for update in moving.history[0]["updates"]
    ]
    assert any(plan.get("environment_reused", False) for plan in pair_plans)
    assert any(
        plan.get("hamiltonian_backend") == "su2-contextual-cpp"
        for plan in pair_plans
    )
    assert all(update["matrix_free"] for update in moving.history[0]["updates"])


def test_su2_letta_cpp_boundaries_match_recursive_reduced_blocks():
    h1e = np.array(
        [
            [-0.8, -0.2, 0.0, 0.0],
            [-0.2, -0.3, -0.15, 0.0],
            [0.0, -0.15, 0.2, -0.1],
            [0.0, 0.0, -0.1, 0.7],
        ]
    )
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=1,
        seed=9,
    )
    state.canonicalize_conditional_center(0)
    sites = state.materialize()
    recursive = BlockSparseEnvironmentChain.build(sites, state.mpo)
    stack = RenormalizedBlockStack(
        namespace="hamiltonian",
        su2_boundary_environment=state._su2_moving_environment,
    )
    compiled = BlockSparseEnvironmentChain.build(
        sites,
        state.mpo,
        renormalized_blocks=stack,
    )

    def flattened(block):
        return {
            (sector_pair, int(channel)): np.asarray(array)
            for sector_pair, channels in block.data.items()
            for channel, array in channels.items()
        }

    for expected_side, compiled_side in (
        (recursive.left_envs, compiled.left_envs),
        (recursive.right_envs, compiled.right_envs),
    ):
        for expected, actual in zip(expected_side, compiled_side):
            expected_blocks = flattened(expected)
            actual_blocks = flattened(actual)
            assert actual_blocks.keys() == expected_blocks.keys()
            for key, expected_block in expected_blocks.items():
                np.testing.assert_allclose(
                    actual_blocks[key], expected_block, atol=3.0e-13
                )

    cache_keys = tuple(
        tuple(core._environment_reduced_block_cache)
        for core in state.mpo
    )
    assert any(key[0] == "left" for keys in cache_keys for key in keys)
    assert any(key[0] == "right" for keys in cache_keys for key in keys)
    BlockSparseEnvironmentChain.build(sites, state.mpo)
    assert cache_keys == tuple(
        tuple(core._environment_reduced_block_cache)
        for core in state.mpo
    )


def test_qchem_su2_letta_sweep_reaches_two_orbital_one_body_reference():
    h1e = np.array([[-1.0, -0.2], [-0.2, 0.5]])
    state = LETTA.from_integrals(
        h1e,
        symmetry="su2",
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=4,
    )
    initial = state.energy
    state.run(nsweeps=1, tol=0.0, max_local_parameters=32)
    exact = 2.0 * np.linalg.eigvalsh(h1e)[0]

    assert state.energy < initial
    np.testing.assert_allclose(state.energy, exact, atol=2.0e-11)
    assert state.history[0]["complete_cycle"]
    assert all(update["native_su2"] for update in state.history[0]["updates"])

    vector = _dense_vector_from_reduced_spatial_mps(state.materialize())
    vector /= np.linalg.norm(vector)
    spin_squared = np.vdot(vector, _total_spin_squared(2) @ vector)
    np.testing.assert_allclose(spin_squared, 0.0, atol=2.0e-12)


def test_qchem_su2_letta_accepts_spatial_eri_and_reaches_hubbard_dimer():
    hopping = 1.0
    interaction = 4.0
    h1e = np.array([[0.0, -hopping], [-hopping, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = interaction
    eri[1, 1, 1, 1] = interaction
    state = LETTA.from_integrals(
        h1e,
        eri,
        symmetry="su2",
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=2,
    )
    state.run(nsweeps=1, tol=0.0, max_local_parameters=32)
    exact = 0.5 * (
        interaction - np.sqrt(interaction**2 + 16.0 * hopping**2)
    )

    np.testing.assert_allclose(state.energy, exact, atol=2.0e-11)


def test_qchem_su2_letta_counts_compiled_mpo_core_energy_once():
    core_energy = -3.25
    h1e = np.array([[-1.0, -0.2], [-0.2, 0.5]])
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        ecore=core_energy,
        seed=4,
    )
    state.run(nsweeps=1, tol=0.0, max_local_parameters=32)
    exact = core_energy + 2.0 * np.linalg.eigvalsh(h1e)[0]

    assert state.mpo_includes_core_energy
    assert state.core_energy == core_energy
    assert state.ecore == 0.0
    np.testing.assert_allclose(state.energy, exact, atol=2.0e-11)


def test_default_wigner_eckart_sweep_matches_polarization_reference():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 4.0
    initial = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=2,
    )
    automatic = copy.deepcopy(initial)
    wigner_eckart = copy.deepcopy(initial)
    reference = copy.deepcopy(initial)

    automatic.run(nsweeps=1, tol=0.0, max_local_parameters=32)
    wigner_eckart.run(
        nsweeps=1,
        tol=0.0,
        max_local_parameters=32,
        solver="wigner_eckart",
    )
    reference.run(
        nsweeps=1,
        tol=0.0,
        max_local_parameters=32,
        solver="polarization",
    )

    np.testing.assert_allclose(automatic.energy, reference.energy, atol=2.0e-11)
    np.testing.assert_allclose(wigner_eckart.energy, reference.energy, atol=2.0e-11)
    assert all(
        update["solver"] == "wigner_eckart"
        for update in automatic.history[0]["updates"]
    )
    assert all(
        update["requested_solver"] == "auto" and update["auto_selected"]
        for update in automatic.history[0]["updates"]
    )
    assert all(
        update["fully_wigner_eckart_reduced"]
        and update["environment_backend"] == "wigner_eckart_reduced"
        for update in wigner_eckart.history[0]["updates"]
    )


def test_wigner_eckart_local_transition_plan_and_grouped_routes_are_exact():
    h1e = np.array(
        [
            [0.0, -1.0, 0.0],
            [-1.0, 0.0, -0.7],
            [0.0, -0.7, 0.2],
        ]
    )
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 0, 0, 0] = 1.3
    eri[1, 1, 1, 1] = 0.8
    eri[2, 2, 2, 2] = 1.1
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=2,
        spin=0,
        graph=[(0, 1), (1, 2)],
        D=1,
        seed=2,
    )
    sites = state.materialize()
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, state.mpo)
    for factors in (state.mpo, identity):
        reference = contract_chain_transition(sites, factors, sites)
        for site in range(len(sites)):
            for direction in ("lr", "rl"):
                transition = LocalTransitionPlan.build(
                    sites, factors, site, direction=direction
                )
                np.testing.assert_allclose(
                    transition.contract(sites[site], sites[site]),
                    reference,
                    atol=2.0e-12,
                )

    assert LocalTransitionPlan.build(sites, state.mpo, 0).direction == "rl"
    assert LocalTransitionPlan.build(sites, state.mpo, 2).direction == "lr"
    cached_blocks = sum(
        len(core._reduced_block_cache) for core in state.mpo
    )
    LocalTransitionPlan.build(sites, state.mpo, 0).contract(sites[0], sites[0])
    assert sum(
        len(core._reduced_block_cache) for core in state.mpo
    ) == cached_blocks

    routes = state._wigner_eckart_route_plan(0)
    rng = np.random.default_rng(31)
    vector = rng.normal(size=state._pack_site(0).size)
    current = state._pack_site(0)
    try:
        state._set_site_vector(0, vector)
        reference = state.materialize_site(0)
    finally:
        state._set_site_vector(0, current)
    routed = routes.tensor(vector)
    assert routes.backend == "block-grouped-gemm"
    assert routes.nbytes > 0
    for key in reference.data:
        np.testing.assert_allclose(routed.data[key], reference.data[key], atol=1.0e-13)
    assert state._wigner_eckart_route_plan(0) is routes
    assert state.wigner_eckart_cache_stats["hits"] >= 1

    layout = _ChannelResolvedPairSpace(sites[0], sites[1])
    embedding_current, embedding = state._local_pair_embedding(
        0, sites, 0, layout
    )
    unit = np.zeros(embedding_current.size, dtype=complex)
    reference_columns = []
    try:
        for index in range(unit.size):
            unit[index] = 1.0
            state._set_site_vector(0, unit)
            reference_columns.append(
                layout.pack_sites(state.materialize_site(0), sites[1])
            )
            unit[index] = 0.0
    finally:
        state._set_site_vector(0, embedding_current)
    np.testing.assert_allclose(
        embedding,
        np.column_stack(reference_columns),
        atol=1.0e-13,
    )


def test_matrix_free_wigner_eckart_davidson_matches_dense_local_solve():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 4.0
    initial = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=2,
    )
    dense = copy.deepcopy(initial)
    matrix_free = copy.deepcopy(initial)
    dense_update = dense.optimize_site(
        0,
        solver="wigner_eckart",
        we_dense_dim=99,
    )
    matrix_free_update = matrix_free.optimize_site(
        0,
        solver="wigner_eckart",
        we_dense_dim=0,
        davidson_tol=1.0e-11,
    )

    np.testing.assert_allclose(
        matrix_free_update["energy_after"],
        dense_update["energy_after"],
        atol=2.0e-11,
    )
    assert matrix_free_update["matrix_free"]
    assert (
        matrix_free_update["local_linear_algebra"]
        == "metric_orthonormal_projected_davidson"
    )
    assert matrix_free_update["solver_info"]["davidson"]["davidson_converged"]
    assert (
        matrix_free_update["solver_info"]["davidson"]["orthonormality_error"]
        < 1.0e-11
    )
    assert matrix_free_update["solver_info"]["route_backend"] == "block-grouped-gemm"


def test_projected_tied_space_sweep_reuses_dmrg_environments_at_d2():
    norb = 4
    h1e = np.diag(np.linspace(-1.0, 1.0, norb))
    h1e += np.diag(np.full(norb - 1, -0.2), 1)
    h1e += np.diag(np.full(norb - 1, -0.2), -1)
    eri = np.zeros((norb, norb, norb, norb))
    indices = np.arange(norb)
    eri[indices, indices, indices, indices] = 1.0
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        graph="nn",
        D=2,
        seed=3,
    )
    before = state.expectation()
    state.run(
        nsweeps=1,
        algorithm="projected",
        reuse_environments=True,
        gauge="conditional",
        solver="wigner_eckart",
        davidson_tol=1.0e-8,
    )

    cycle = state.history[-1]
    assert state.energy < before
    assert cycle["algorithm"] == "projected"
    assert cycle["environment_reuse"]
    assert cycle["rejected_updates"] == 0
    assert all(
        update["local_linear_algebra"]
        == "metric_orthonormal_projected_davidson"
        for update in cycle["updates"]
    )
    assert all(
        update["solver_info"]["environment_reused"]
        for update in cycle["updates"]
    )
    assert max(
        update["solver_info"]["davidson"]["orthonormality_error"]
        for update in cycle["updates"]
    ) < 1.0e-10


def test_frontier_one_site_actions_match_full_chain_wigner_eckart_at_d2():
    norb = 4
    h1e = np.diag(np.linspace(-1.0, 1.0, norb))
    h1e += np.diag(np.full(norb - 1, -0.2), 1)
    h1e += np.diag(np.full(norb - 1, -0.2), -1)
    eri = np.zeros((norb, norb, norb, norb))
    eri[np.arange(norb), np.arange(norb), np.arange(norb), np.arange(norb)] = 1.0
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=2,
        seed=3,
    )
    sites = state.materialize()
    reference = state._wigner_eckart_matrix_free_problem(1, sites=sites)
    frontier = state._frontier_wigner_eckart_matrix_free_problem(
        1, 1, sites=sites
    )
    rng = np.random.default_rng(4)
    vector = rng.normal(size=reference[1].size) + 1j * rng.normal(
        size=reference[1].size
    )

    np.testing.assert_array_equal(frontier[1], reference[1])
    np.testing.assert_allclose(
        frontier[2](vector), reference[2](vector), atol=2.0e-12
    )
    np.testing.assert_allclose(
        frontier[3](vector), reference[3](vector), atol=2.0e-12
    )
    assert frontier[-1]["local_action_backend"] == "frontier_projected_pair"
    assert frontier[-1]["backend"] == "compiled_contextual_channel_resolved"
    state.close()


def test_native_two_site_su2_letta_reaches_hubbard_dimer_and_reports_truncation():
    hopping = 1.0
    interaction = 4.0
    h1e = np.array([[0.0, -hopping], [-hopping, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = interaction
    eri[1, 1, 1, 1] = interaction
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=2,
    )

    update = state.optimize_two_sites(0, davidson_tol=1.0e-11)
    exact = 0.5 * (
        interaction - np.sqrt(interaction**2 + 16.0 * hopping**2)
    )

    np.testing.assert_allclose(state.expectation(), exact, atol=2.0e-11)
    assert update["accepted"]
    assert update["fully_wigner_eckart_reduced"]
    assert update["environment_backend"] == "channel_resolved_pair_transitions"
    assert update["local_residual"] < 1.0e-10
    assert update["truncation_error"] < 1.0e-12
    assert update["fixed_reduced_bond_dim"] == 1


def test_native_two_site_su2_letta_accepts_an_empty_triplet_operator():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 4.0
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=2,
        spin=2,
        graph=[(0, 1)],
        D=1,
        seed=3,
    )

    state.run(nsweeps=1, algorithm="two_site", tol=0.0)

    np.testing.assert_allclose(state.energy, 0.0, atol=2.0e-12)
    assert state.target_sector.irrep.two_j == 2
    assert all(update["accepted"] for update in state.history[0]["updates"])
    assert state.history[0]["max_local_residual"] == 0.0


def test_metric_retraction_converges_despite_null_coefficient_directions():
    h1e = np.zeros((4, 4))
    h1e[np.arange(3), np.arange(1, 4)] = -1.0
    h1e[np.arange(1, 4), np.arange(3)] = -1.0
    eri = np.zeros((4, 4, 4, 4))
    eri[np.arange(4), np.arange(4), np.arange(4), np.arange(4)] = 4.0
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=1,
        seed=7,
    )

    state.run(
        nsweeps=3,
        algorithm="two_site",
        tol=1.0e-9,
        residual_tol=1.0e-8,
        truncation_tol=1.0e-7,
        consecutive_cycles=2,
    )

    exact = -1.9531453086845698
    np.testing.assert_allclose(state.energy, exact, atol=2.0e-11)
    assert state.converged
    assert state.convergence_summary["max_truncation_error"] < 1.0e-7
    central = [
        update
        for update in state.history[-1]["updates"]
        if update["bond"] == 1
    ]
    assert central
    assert all("coefficient_retraction_error" in update for update in central)
    assert all(
        update["solver_info"]["davidson"]["metric_rank"]
        < update["solver_info"]["davidson"]["parent_dimension"]
        for update in central
    )
    assert max(update["truncation_error"] for update in central) < 1.0e-7


def test_channel_resolved_d2_pair_kernel_is_exact_and_matrix_free():
    h1e = np.zeros((4, 4))
    h1e[np.arange(3), np.arange(1, 4)] = -1.0
    h1e[np.arange(1, 4), np.arange(3)] = -1.0
    eri = np.zeros((4, 4, 4, 4))
    eri[np.arange(4), np.arange(4), np.arange(4), np.arange(4)] = 4.0
    state = SU2LETTA.from_integrals(
        h1e,
        eri,
        nelec=4,
        spin=0,
        graph=[(0, 1), (1, 2), (2, 3)],
        D=2,
        seed=2,
    )
    initial = state.expectation()

    update = state.optimize_two_sites(
        1,
        dense_dim=0,
        davidson_tol=1.0e-8,
        davidson_maxiter=40,
        davidson_max_space=24,
        retraction_maxiter=2,
        retraction_relax_sweeps=0,
    )

    np.testing.assert_allclose(update["energy_before"], initial, atol=2.0e-12)
    assert update["accepted"]
    assert update["energy_after"] < initial
    assert state.expectation() >= -1.95314530868457 - 1.0e-10
    assert update["matrix_free"]
    assert update["environment_backend"] == "channel_resolved_pair_transitions"
    davidson = update["solver_info"]["davidson"]
    assert davidson["davidson_converged"]
    assert davidson["metric_rank"] < davidson["parent_dimension"]
    assert davidson["pair_transition_plan"]["hamiltonian_transitions"] > 0
    assert update["local_residual"] < 1.0e-8


def test_su2_letta_checkpoint_resume_preserves_state_and_diagnostics(tmp_path):
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=4,
    )
    checkpoint = tmp_path / "su2-letta.chk"
    state.run(
        nsweeps=1,
        algorithm="two_site",
        tol=0.0,
        checkpoint=checkpoint,
    )
    restored = SU2LETTA.load_checkpoint(checkpoint)

    np.testing.assert_allclose(restored.expectation(), state.energy, atol=2.0e-12)
    assert restored.history == state.history
    assert restored.convergence_summary["cycles"] == 1
    assert restored.convergence_summary["max_local_residual"] < 1.0e-9
    assert restored.storage_nbytes > 0

    restored.run(
        nsweeps=1,
        algorithm="two_site",
        tol=1.0e-12,
        reset_history=False,
    )
    assert len(restored.history) == 2
    assert restored.history[-1]["sweep"] == 2


def test_auto_solver_always_selects_the_exact_reduced_path():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=2,
        spin=0,
        graph=[(0, 1)],
        D=1,
        seed=2,
    )
    assert state._select_local_solver("auto") == "wigner_eckart"
    with pytest.raises(ValueError, match="solver must be"):
        state.run(nsweeps=1, solver="projected")

    h1e_large = np.array(
        [
            [-1.0, -0.2, 0.0],
            [-0.2, 0.1, -0.15],
            [0.0, -0.15, 0.6],
        ]
    )
    mixed = SU2LETTA.from_integrals(
        h1e_large,
        nelec=2,
        spin=0,
        graph=[(0, 1), (1, 2)],
        D=2,
        seed=8,
    )
    mixed.run(nsweeps=1, tol=0.0, solver="auto", davidson_tol=1.0e-8)
    updates = mixed.history[0]["updates"]
    assert {update["solver"] for update in updates} == {"wigner_eckart"}
    assert all(update["requested_solver"] == "auto" for update in updates)
    assert all(
        update["solver_info"]["davidson"]["davidson_converged"]
        for update in updates
        if update["matrix_free"]
    )


def test_shared_worker_pool_preserves_wigner_eckart_updates():
    h1e = np.array([[0.0, -1.0], [-1.0, 0.0]])
    eri = np.zeros((2, 2, 2, 2))
    eri[0, 0, 0, 0] = 4.0
    eri[1, 1, 1, 1] = 4.0
    states = [
        SU2LETTA.from_integrals(
            h1e,
            eri,
            nelec=2,
            spin=0,
            graph=[(0, 1)],
            D=1,
            seed=2,
            workers=workers,
        )
        for workers in (1, 2)
    ]
    try:
        states[0].run(nsweeps=1, tol=0.0, solver="wigner_eckart")
        states[1].run(nsweeps=1, tol=0.0, solver="wigner_eckart")
        np.testing.assert_allclose(states[1].energy, states[0].energy, atol=2.0e-12)
        assert all(
            update["workers"] == 2
            for update in states[1].history[0]["updates"]
        )
    finally:
        for state in states:
            state.close()


def test_wigner_eckart_sweep_matches_three_site_polarization_reference():
    h1e = np.array(
        [
            [-1.0, -0.2, 0.0],
            [-0.2, 0.1, -0.15],
            [0.0, -0.15, 0.6],
        ]
    )
    initial = SU2LETTA.from_integrals(
        h1e,
        nelec=2,
        spin=0,
        graph=[(0, 1), (1, 2)],
        D=1,
        seed=8,
    )
    reduced = copy.deepcopy(initial)
    reference = copy.deepcopy(initial)

    reduced.run(nsweeps=1, tol=0.0, max_local_parameters=128)
    reference.run(
        nsweeps=1,
        tol=0.0,
        max_local_parameters=128,
        solver="polarization",
    )

    np.testing.assert_allclose(reduced.energy, reference.energy, atol=2.0e-11)
    assert len(reduced.history[0]["updates"]) == 2 * (reduced.nsites - 1)
    assert all(
        update["environment_backend"] == "wigner_eckart_reduced"
        for update in reduced.history[0]["updates"]
    )


def test_reduced_block_actions_match_component_expanded_reference():
    h1e = np.array(
        [
            [-1.0, -0.2, 0.0],
            [-0.2, 0.1, -0.15],
            [0.0, -0.15, 0.6],
        ]
    )
    state = SU2LETTA.from_integrals(
        h1e,
        nelec=2,
        spin=0,
        graph=[(0, 1), (1, 2)],
        D=1,
        seed=8,
    )
    sites = state.materialize()
    component_mpo = tuple(expand_rank_coupled_mpo(core) for core in state.mpo)
    sparse_chain = BlockSparseEnvironmentChain.build(sites, component_mpo)
    dense_chain = DenseEnvironmentChain.build(sites, state.mpo)
    rng = np.random.default_rng(19)

    for bond in range(state.nsites - 1):
        template = merge_mps_sites(sites[bond], sites[bond + 1])
        packed, layout = pack_two_site_state(template)
        sparse_action, _ = _resolve_davidson_operator(
            sparse_chain.bond_operator(bond, template), template, layout
        )
        dense_action, _ = _resolve_davidson_operator(
            dense_chain.bond_operator(bond, template), template, layout
        )
        probe = rng.normal(size=packed.size) + 1.0j * rng.normal(size=packed.size)

        np.testing.assert_allclose(
            sparse_action(probe),
            dense_action(probe),
            atol=2.0e-12,
            rtol=2.0e-12,
        )


def test_su2_letta_rejects_magnetic_multiplicity_physical_sites():
    h1e = np.diag([-0.5, 0.25])
    hamiltonian = build_spatial_reduced_hamiltonian_mpo(
        h1e,
        fully_reduced=True,
        n_elec=2,
        spin=0,
    )
    base = build_random_reduced_spatial_mps(2, seed=3)
    base[0].metadata["physical_basis"] = "canonical_su2"

    with np.testing.assert_raises(ValueError):
        SU2LETTA.from_mps(
            base,
            hamiltonian,
            nelec=2,
            spin=0,
        )
