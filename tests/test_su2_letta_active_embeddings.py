import numpy as np
import pytest

from pyqed.letta import SU2LETTA
from pyqed.letta.su2_qchem import _canonical_graph
from pyqed.qchem.dmrg.backends.reduced import (
    ComplementaryEntries,
    ComplementaryOperatorFamily,
)


@pytest.mark.parametrize("alias", ["nnnn", "third_neighbor", "third-neighbor"])
def test_nnnn_graph_includes_all_three_neighbor_ranges(alias):
    expected = tuple((i, j) for i in range(5) for j in range(i + 1, min(5, i + 4)))
    assert _canonical_graph(5, alias) == expected


def test_nnnn_projected_sweep_matches_explicit_mpo_energy():
    h = np.diag([-1., -.4, .2, .7])
    h += np.diag(np.full(3, -.2), 1) + np.diag(np.full(3, -.2), -1)
    eri = np.zeros((4,) * 4)
    eri[np.arange(4), np.arange(4), np.arange(4), np.arange(4)] = 1.
    direct = SU2LETTA.from_integrals(h, eri, nelec=4, graph="nnnn", D=1, seed=11)
    explicit = SU2LETTA.from_integrals(
        h, eri, nelec=4, graph="nnnn", D=1, seed=11,
        hamiltonian_representation="mpo",
    )
    try:
        assert direct.max_frontier_width == 3
        assert max(direct.frontier_states) == 27
        assert not direct.supports_conditional_canonical_gauge
        before = direct.energy
        np.testing.assert_allclose(before, explicit.expectation(), atol=2e-12)
        direct.run(nsweeps=1, algorithm="projected", tol=0., davidson_tol=1e-8)
        for site in range(4):
            explicit._set_site_vector(site, direct._pack_site(site))
        np.testing.assert_allclose(direct.energy, explicit.expectation(), atol=2e-10)
        assert direct.energy <= before + 1e-10
        assert not direct.history[-1]["conditional_gauge_supported"]
        assert all(update["solver_info"]["environment_reused"]
                   for update in direct.history[-1]["updates"])
    finally:
        direct.close()
        explicit.close()


@pytest.mark.parametrize("graph", ["nn", "nnn", "nnnn"])
@pytest.mark.parametrize("site", [1, 2])
def test_active_embedding_matches_unit_direction_reference(graph, site, monkeypatch):
    state = SU2LETTA.from_integrals(
        np.diag([-1., -.4, .2, .7]), nelec=4, spin=0,
        D=2, graph=graph, seed=11,
    )
    try:
        sites = state.materialize()
        _, layout = state._pair_coordinate_space(1, sites)
        routes = state._wigner_eckart_route_plan(site)
        for key in sites[site].data:
            active, blocks = routes.active_block_basis(key)
            full = routes.block_basis(key)
            restored = np.zeros_like(full)
            restored[active] = blocks
            np.testing.assert_array_equal(restored, full)

        reference = np.column_stack([
            layout.pack_sites(varied, sites[2]) if site == 1
            else layout.pack_sites(sites[1], varied)
            for varied in routes.basis
        ])

        def forbid_full_basis(*args, **kwargs):
            raise AssertionError("Production embedding must use active directions")

        monkeypatch.setattr(routes, "block_basis", forbid_full_basis)
        _, embedding = state._local_pair_embedding(site, sites, 1, layout)
        np.testing.assert_allclose(np.asarray(embedding), reference, atol=1e-13)
        assert all(blocks.shape[0] == indices.size
                   for indices, blocks in state._embedding_basis_cache.values())
        assert sum(blocks.nbytes for _, blocks in state._embedding_basis_cache.values()) < (
            sum(routes.nparameters * np.asarray(block).nbytes
                for block in sites[site].data.values())
        )
        assert state.storage_nbytes > 0
    finally:
        state.close()


@pytest.mark.parametrize("count", [0, 3])
def test_complementary_metadata_uses_array_shape_without_entry_scan(count):
    class NoIteration(ComplementaryEntries):
        def __iter__(self):
            raise AssertionError("Metadata must not scan integral entries")

    entries = NoIteration(np.zeros((count, 4), dtype=int), np.zeros(count))
    family = ComplementaryOperatorFamily("P", 2, entries)
    assert family.as_metadata()["index_shape"] == ((4,) if count else ())
    assert family.as_metadata()["n_terms"] == count
    mapped = ComplementaryOperatorFamily("P", 2, {(0,): 1., (0, 1): 2.})
    assert mapped.index_shape == (1, 2)


@pytest.mark.parametrize("budget", [0, 4 * 1024**2])
@pytest.mark.parametrize("spin", [0, 2])
@pytest.mark.parametrize("use_packer", [False, True])
def test_complementary_energy_reuses_bounded_boundary_routes(
    budget, spin, use_packer, monkeypatch,
):
    from pyqed.mps.nonabelian import environment

    monkeypatch.setattr(environment, "_BOUNDARY_ROUTE_CACHE_MAX_BYTES", budget)
    packing_calls = []
    pack = environment._pack_normal_complementary_boundary_routes_cpp

    def counted_pack(*args, **kwargs):
        packing_calls.append(kwargs["side"])
        return pack(*args, **kwargs) if use_packer else None

    monkeypatch.setattr(environment, "_pack_normal_complementary_boundary_routes_cpp",
                        counted_pack)
    n = 6
    h = np.diag(np.linspace(-1., 1., n))
    h += np.diag(np.full(n - 1, -.2), 1) + np.diag(np.full(n - 1, -.2), -1)
    rng = np.random.default_rng(19)
    factors = rng.normal(scale=.1, size=(n, n, 4))
    factors = (factors + factors.swapaxes(0, 1)) * .5
    eri = np.einsum("pqL,rsL->pqrs", factors, factors)
    direct = SU2LETTA.from_integrals(h, eri, nelec=n, spin=spin, D=2, seed=11)
    explicit = SU2LETTA.from_integrals(
        h, eri, nelec=n, spin=spin, D=2, seed=11,
        hamiltonian_representation="mpo",
    )
    try:
        first = direct.expectation()
        first_calls = len(packing_calls)
        assert first_calls > 0
        assert set(packing_calls) == {"right"}
        np.testing.assert_allclose(first, explicit.expectation(), atol=2e-12)
        np.testing.assert_allclose(direct.expectation(), first, atol=2e-12)
        if budget == 0:
            assert len(packing_calls) > first_calls
        else:
            assert len(packing_calls) == first_calls

        changed = direct._pack_site(2).copy()
        changed[0] += .2
        direct._set_site_vector(2, changed)
        explicit._set_site_vector(2, changed)
        np.testing.assert_allclose(direct.expectation(), explicit.expectation(), atol=2e-12)
        for core in direct.mpo:
            records = core.normal_complementary_plan.get("_boundary_action_cache", {})
            assert sum(record["routes"].nbytes for record in records.values()
                       if record.get("routes") is not None) <= budget
    finally:
        direct.close()
        explicit.close()


@pytest.mark.parametrize("spin", [0, 2])
def test_factorized_metric_projection_matches_dense_complex_reference(spin):
    state = SU2LETTA.from_integrals(
        np.diag([-1., -.4, .2, .7]), nelec=4, spin=spin,
        D=3, graph="nnn", seed=11,
    )
    try:
        sites = state.materialize()
        merged, layout = state._pair_coordinate_space(1, sites)
        _, dense, dense_info = state._reduced_pair_transition_actions(
            1, sites, merged, layout,
        )
        rng = np.random.default_rng(13)
        embedding = rng.normal(size=(layout.size, 5)) + 1j * rng.normal(
            size=(layout.size, 5)
        )
        expected = state._project_metric_components(
            embedding, dense.metric_blocks, site=1, bond=1,
        )
        _, compact, compact_info = state._reduced_pair_transition_actions(
            1, sites, merged, layout, compact_metric_blocks=True,
        )
        actual = state._project_metric_components(
            embedding, compact.metric_blocks, site=1, bond=1, factorized=True,
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=2e-12)
        assert compact_info["metric_nbytes"] < dense_info["metric_nbytes"]
        assert not compact_info["metric_blocks_materialized"]
        assert compact_info["metric_blocks_factorized"]
    finally:
        state.close()


@pytest.mark.parametrize("graph", ["nn", "nnn", "nnnn"])
@pytest.mark.parametrize("tie", ["physical", "fusion"])
@pytest.mark.parametrize("site", [1, 2])
def test_frontier_scatter_matches_materialized_tensor_and_pair_actions(graph, tie, site):
    state = SU2LETTA.from_integrals(
        np.diag([-1., -.4, .2, .7]), nelec=4, spin=0,
        D=2, graph=graph, tie=tie, seed=11,
    )
    try:
        routes = state._wigner_eckart_route_plan(site)
        original = state._pack_site(site)
        rng = np.random.default_rng(31)
        vector = rng.normal(size=original.size) + 1j * rng.normal(size=original.size)
        state._set_site_vector(site, vector)
        actual = state.materialize_site(site)
        predicted = routes.tensor(vector)
        assert actual.data.keys() == predicted.data.keys()
        for key in actual.data:
            np.testing.assert_allclose(predicted.data[key], actual.data[key], atol=1e-13)
        np.testing.assert_allclose(
            np.vdot(vector, routes.adjoint(actual.data)),
            sum(np.vdot(block, block) for block in actual.data.values()), atol=1e-11,
        )
        sites = state.materialize()
        _, layout = state._pair_coordinate_space(1, sites)
        _, structured = state._local_pair_embedding(site, sites, 1, layout)
        _, action = state._local_pair_embedding(site, sites, 1, layout, matrix_free=True)
        dense = np.asarray(structured)
        pair = rng.normal(size=layout.size) + 1j * rng.normal(size=layout.size)
        np.testing.assert_allclose(action.matvec(vector), dense @ vector, atol=1e-12)
        np.testing.assert_allclose(action.rmatvec(pair), dense.conj().T @ pair, atol=1e-12)
    finally:
        state.close()


@pytest.mark.parametrize("complex_transform", [False, True])
def test_projected_factor_routes_preserve_real_and_complex_actions(complex_transform, monkeypatch):
    h = np.diag(np.linspace(-1., 1., 6))
    h += np.diag(np.full(5, -.2), 1) + np.diag(np.full(5, -.2), -1)
    eri = np.zeros((6,) * 4)
    eri[np.arange(6), np.arange(6), np.arange(6), np.arange(6)] = .6
    state = SU2LETTA.from_integrals(
        h, eri, nelec=6, D=1, seed=11,
        graph=[(i, i+1) for i in range(5)] + [(0, 5)],
    )
    original = state._install_projected_factor_route
    tested = []

    def check_install(**kwargs):
        route = original(**kwargs)
        if route is None:
            return None
        design = kwargs['orthonormal_design']
        if complex_transform:
            design = design.compose(np.diag(np.exp(.3j * np.arange(design.shape[1]))))
            route = original(**dict(kwargs, orthonormal_design=design))
        assert route['real_projection'] == (not complex_transform)
        owner, key = route['owner'], route['projection_key']
        rng = np.random.default_rng(42)
        vector = rng.normal(size=design.shape[1]) + 1j * rng.normal(size=design.shape[1])
        reference = design.rmatvec(kwargs['parent_action'](design.matvec(vector)))
        compiled = owner.factor_route_projected_matvec(key, vector)
        np.testing.assert_allclose(compiled, reference, atol=1e-12)
        if not complex_transform:
            split = (owner.factor_route_projected_real_matvec(key, vector.real)
                     + 1j * owner.factor_route_projected_real_matvec(key, vector.imag))
            np.testing.assert_allclose(split, compiled, atol=1e-12)
        tested.append(True)
        return original(**kwargs)

    monkeypatch.setattr(state, '_install_projected_factor_route', check_install)
    try:
        state.run(nsweeps=1, algorithm='projected', reuse_environments=True,
                  tol=0., davidson_tol=1e-9)
        assert tested
    finally:
        state.close()
