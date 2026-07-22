import numpy as np

from pyqed.narg import (
    CoordinateTreeBasis,
    LocalCoordinateBasis,
    ManyElectronOrderedOperator,
    OrderedConfigurationSpace,
    ParticleGrowthLayer,
    ParticleGrowthNARGResult,
    ParticleGrowthState,
    PrefixCoordinateSpace,
    RecursiveCoordinateBasis,
    SparseBasis,
    coordinate_tree_basis,
    narg,
    narg_matrix_free,
    ordered_hamiltonian,
    ordered_operator,
    particle_growth_basis,
    particle_growth_layer,
    particle_growth_narg,
    sine_box_dvr,
    two_electron_first_quantized_narg,
    two_electron_wedge_hamiltonian,
)


def _soft_coulomb(x, y):
    return 1.0 / np.sqrt((x - y) ** 2 + 0.25)


def _nearest_neighbor_kinetic(ngrid, *, xmin=-4.0, xmax=4.0):
    grid = np.linspace(xmin, xmax, int(ngrid))
    dx = grid[1] - grid[0]
    kinetic = np.zeros((ngrid, ngrid))
    np.fill_diagonal(kinetic, 1.0 / dx**2)
    offdiag = -0.5 / dx**2
    for site in range(ngrid - 1):
        kinetic[site, site + 1] = offdiag
        kinetic[site + 1, site] = offdiag
    return grid, kinetic


def test_two_electron_first_quantized_narg_is_exact_when_branches_are_full():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    hamiltonian, pairs = two_electron_wedge_hamiltonian(
        kinetic,
        grid,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
        exchange="triplet",
    )

    result = two_electron_first_quantized_narg(
        hamiltonian,
        pairs,
        D=6,
        nstates=5,
        exchange="triplet",
    )

    exact = np.linalg.eigvalsh(hamiltonian)[:5]
    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    np.testing.assert_allclose(result.exact_energies, exact, atol=1e-10)


def test_two_electron_first_quantized_narg_is_variational_for_truncated_branches():
    grid, kinetic = sine_box_dvr(8, xmin=-5.0, xmax=5.0)
    hamiltonian, pairs = two_electron_wedge_hamiltonian(
        kinetic,
        grid,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
        exchange="singlet",
    )

    result = two_electron_first_quantized_narg(
        hamiltonian,
        pairs,
        D=2,
        nstates=4,
        exchange="singlet",
    )

    exact = np.linalg.eigvalsh(hamiltonian)[:4]
    assert result.branch_basis.shape[0] == pairs.shape[0]
    assert result.branch_basis.shape[1] < pairs.shape[0]
    assert np.all(result.energies >= exact - 1e-10)


def test_two_electron_first_quantized_narg_uses_only_ordered_wedge_states():
    grid, kinetic = sine_box_dvr(6)
    hamiltonian, pairs = two_electron_wedge_hamiltonian(kinetic, grid, exchange="triplet")
    result = two_electron_first_quantized_narg(hamiltonian, pairs, D=2, nstates=2)

    assert pairs.shape[0] == 6 * 5 // 2
    assert result.branch_basis.shape[0] == pairs.shape[0]
    assert np.all(pairs[:, 0] < pairs[:, 1])


def test_ordered_hamiltonian_matches_two_electron_triplet_builder():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    h2, pairs = two_electron_wedge_hamiltonian(
        kinetic,
        grid,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
        exchange="triplet",
    )
    hmany, configs = ordered_hamiltonian(
        kinetic,
        grid,
        nelec=2,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    np.testing.assert_array_equal(configs, pairs)
    np.testing.assert_allclose(hmany, h2, atol=1e-12)


def test_narg_is_exact_when_branches_are_full():
    grid, kinetic = sine_box_dvr(6, xmin=-4.0, xmax=4.0)
    hamiltonian, configs = ordered_hamiltonian(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    result = narg(hamiltonian, configs, D=10, nstates=5)

    exact = np.linalg.eigvalsh(hamiltonian)[:5]
    np.testing.assert_allclose(result.energies, exact, atol=1e-10)
    np.testing.assert_allclose(result.exact_energies, exact, atol=1e-10)


def test_narg_is_variational_when_truncated():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    hamiltonian, configs = ordered_hamiltonian(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    result = narg(hamiltonian, configs, D=2, nstates=4)
    exact = np.linalg.eigvalsh(hamiltonian)[:4]

    assert result.branch_basis.shape[0] == configs.shape[0]
    assert result.branch_basis.shape[1] < configs.shape[0]
    assert np.all(result.energies >= exact - 1e-10)



def test_ordered_operator_matches_dense_builder_and_matvec():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    assert isinstance(operator.configs, OrderedConfigurationSpace)
    assert not hasattr(operator, "index")
    dense, configs = ordered_hamiltonian(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    rng = np.random.default_rng(24)
    vector = rng.normal(size=operator.shape[0])
    matrix = rng.normal(size=(operator.shape[0], 3))

    np.testing.assert_array_equal(operator.configs, configs)
    np.testing.assert_allclose(operator.to_dense(), dense, atol=1e-12)
    np.testing.assert_allclose(operator.matvec(vector), dense @ vector, atol=1e-12)
    np.testing.assert_allclose(operator.matmat(matrix), dense @ matrix, atol=1e-12)


def test_ordered_operator_uses_sparse_kinetic_hops_and_lazy_potentials():
    grid = np.linspace(-1.0, 1.0, 6)
    kinetic = np.zeros((grid.size, grid.size))
    kinetic[np.arange(grid.size), np.arange(grid.size)] = 2.0
    kinetic[np.arange(grid.size - 1), np.arange(1, grid.size)] = -0.5
    kinetic[np.arange(1, grid.size), np.arange(grid.size - 1)] = -0.5

    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: x**2,
        interaction=lambda x, y: 1.0 / np.sqrt((x - y) ** 2 + 0.3),
    )

    assert operator._one_body_array is None
    assert operator._two_body_array is None
    np.testing.assert_array_equal(operator.kinetic_terms(2)[0], np.array([1, 2, 3]))
    np.testing.assert_array_equal(operator.kinetic_terms(2, transpose=True)[0], np.array([1, 2, 3]))

    rng = np.random.default_rng(36)
    vector = rng.normal(size=operator.shape[0])
    rows = np.arange(operator.shape[0])
    out_rows, out_values = operator.apply_sparse(rows, vector)
    sparse_out = np.zeros(operator.shape[0])
    sparse_out[out_rows] = out_values

    np.testing.assert_allclose(sparse_out, operator.matvec(vector), atol=1e-12)


def test_ordered_operator_accepts_analytic_kinetic_hop_callback():
    grid = np.linspace(-1.0, 1.0, 7)
    kinetic = np.zeros((grid.size, grid.size))
    kinetic[np.arange(grid.size), np.arange(grid.size)] = 2.0
    kinetic[np.arange(grid.size - 1), np.arange(1, grid.size)] = -0.5
    kinetic[np.arange(1, grid.size), np.arange(grid.size - 1)] = -0.5

    def kinetic_hops(site):
        cols = [site]
        values = [2.0]
        if site > 0:
            cols.append(site - 1)
            values.append(-0.5)
        if site + 1 < grid.size:
            cols.append(site + 1)
            values.append(-0.5)
        return cols, values

    matrix_operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.25 * x**2,
        interaction=_soft_coulomb,
    )
    callback_operator = ordered_operator(
        kinetic_hops,
        grid,
        nelec=3,
        external=lambda x: 0.25 * x**2,
        interaction=_soft_coulomb,
    )

    assert callback_operator.kinetic is None
    rng = np.random.default_rng(48)
    vector = rng.normal(size=matrix_operator.shape[0])

    np.testing.assert_allclose(callback_operator.matvec(vector), matrix_operator.matvec(vector), atol=1e-12)
    np.testing.assert_allclose(callback_operator.to_dense(), matrix_operator.to_dense(), atol=1e-12)


def test_ordered_operator_local_matvec_matches_local_dense_block():
    grid, kinetic = sine_box_dvr(8, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    rows = operator.prefix_rows((1,))

    rng = np.random.default_rng(52)
    vector = rng.normal(size=rows.size)
    matrix = rng.normal(size=(rows.size, 3))

    block = operator.submatrix(rows)
    np.testing.assert_allclose(operator.local_matvec(rows, vector), block @ vector, atol=1e-12)
    np.testing.assert_allclose(operator.local_matmat(rows, matrix), block @ matrix, atol=1e-12)
    np.testing.assert_allclose(operator.local_dense(rows), block, atol=1e-12)


def test_ordered_configuration_space_rank_unrank_matches_dense_order():
    space = OrderedConfigurationSpace(7, 3)
    dense = np.asarray(space)

    assert dense.shape == (35, 3)
    assert tuple(space[0]) == (0, 1, 2)
    assert tuple(space[-1]) == (4, 5, 6)
    for row, config in enumerate(dense):
        assert space.rank(config) == row
        np.testing.assert_array_equal(space.unrank(row), config)
    np.testing.assert_array_equal(space[[0, 4, 10], 1], dense[[0, 4, 10], 1])


def test_ordered_configuration_space_prefix_ranges_match_dense_filter():
    space = OrderedConfigurationSpace(8, 4)
    dense = np.asarray(space)
    for prefix in [(), (0,), (1,), (1, 3), (2, 4, 6)]:
        rows = space.prefix_rows(prefix)
        assert isinstance(rows, PrefixCoordinateSpace)
        expected = np.arange(dense.shape[0])
        for depth, value in enumerate(prefix):
            expected = expected[dense[expected, depth] == value]
        np.testing.assert_array_equal(rows, expected)
        if expected.size:
            assert rows.start == expected[0]
            assert rows.size == expected.size
        if len(prefix) < space.nelec:
            np.testing.assert_array_equal(space.child_values(prefix), np.unique(dense[rows, len(prefix)]))


def test_matrix_free_narg_matches_dense_narg():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    dense = operator.to_dense()

    dense_result = narg(dense, operator.configs, D=3, nstates=4)
    matrix_free_result = narg_matrix_free(operator, D=3, nstates=4, exact=True)

    assert isinstance(matrix_free_result.branch_basis, CoordinateTreeBasis)
    assert isinstance(matrix_free_result.vectors, RecursiveCoordinateBasis)
    np.testing.assert_allclose(matrix_free_result.energies, dense_result.energies, atol=1e-10)
    np.testing.assert_allclose(matrix_free_result.exact_energies, dense_result.exact_energies, atol=1e-10)
    assert matrix_free_result.branch_basis.shape[0] == operator.shape[0]
    assert matrix_free_result.vectors.shape[1] == 4


def test_coordinate_tree_basis_stores_recursive_branch_bases():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    def _unexpected_submatrix(_rows):
        raise AssertionError("coordinate tree leaves should use local matrix-free eigensolves")

    operator.submatrix = _unexpected_submatrix
    tree = coordinate_tree_basis(operator, D=2)

    for branch in tree.iter_branches():
        assert isinstance(branch.basis, RecursiveCoordinateBasis)
        assert isinstance(branch.rows, PrefixCoordinateSpace)
        assert branch.basis.rows is branch.rows
        assert branch.basis.local_shape[0] == branch.rows.size
        local_basis = branch.basis.to_local()
        assert isinstance(local_basis, LocalCoordinateBasis)
        assert local_basis.rows is branch.rows
        np.testing.assert_array_equal(local_basis.rows, branch.rows)
        assert branch.basis.shape[0] == operator.shape[0]
        if branch.children:
            assert branch.basis.children
            assert branch.basis.leaf_vectors is None
            if branch.prefix:
                assert branch.basis.coeff is not None
        else:
            assert branch.basis.leaf_vectors is not None

    sparse = tree.to_sparse()
    assert isinstance(sparse, SparseBasis)
    assert tree.root.basis.local_shape[0] == operator.shape[0]
    np.testing.assert_allclose(operator.project(tree.root.basis), operator.project(sparse), atol=1e-12)


def test_coordinate_tree_basis_can_share_electron_three_suffix_bases():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    tree = coordinate_tree_basis(operator, D=2, share_suffix=True)
    leaves = [branch for branch in tree.iter_branches() if not branch.children]
    shared_by_key = {}
    repeated = 0
    for leaf in leaves:
        assert leaf.basis.leaf_key is not None
        assert leaf.basis.coeff is not None
        assert leaf.basis.shape[1] <= 2
        key = leaf.basis.leaf_key
        previous = shared_by_key.setdefault(key, leaf.basis.leaf_vectors)
        if previous is leaf.basis.leaf_vectors:
            repeated += 1
        assert previous is leaf.basis.leaf_vectors

    assert repeated > len(shared_by_key)
    sparse = tree.to_sparse()
    np.testing.assert_allclose(tree.project(operator), operator.project(sparse), atol=1e-12)

    result = narg_matrix_free(operator, D=2, nstates=4, exact=True, share_suffix=True)
    assert isinstance(result.vectors, RecursiveCoordinateBasis)
    assert np.all(result.energies >= result.exact_energies - 1e-10)


def test_particle_growth_narg_is_exact_when_fixed_last_sectors_are_full():
    grid, kinetic = sine_box_dvr(6, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    result = particle_growth_narg(operator, D=10, nstates=5, exact=True)

    assert isinstance(result, ParticleGrowthNARGResult)
    assert isinstance(result.branch_basis, ParticleGrowthLayer)
    assert isinstance(result.vectors, ParticleGrowthState)
    assert result.branch_basis.shape == operator.shape
    assert result.vectors.shape == (operator.shape[0], 5)
    np.testing.assert_allclose(result.energies, result.exact_energies, atol=1e-10)


def test_particle_growth_narg_extends_to_four_electron_fixed_last_sectors():
    grid, kinetic = sine_box_dvr(6, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=4,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    result = particle_growth_narg(operator, D=10, nstates=4, exact=True)

    assert result.branch_basis.shape == operator.shape
    np.testing.assert_allclose(result.energies, result.exact_energies, atol=1e-10)


def test_particle_growth_basis_reuses_previous_layer_without_inner_eigensolves():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=4,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    original = ManyElectronOrderedOperator.local_lowest_eigenpairs

    def _unexpected_local_eigensolve(self, rows, nstates):
        raise AssertionError("adjacent particle growth should reuse previous layers")

    def _unexpected_full_project(_basis):
        raise AssertionError("adjacent particle growth should use structured layer projection")

    operator.project = _unexpected_full_project
    ManyElectronOrderedOperator.local_lowest_eigenpairs = _unexpected_local_eigensolve
    try:
        tree = particle_growth_basis(operator, D=2)
    finally:
        ManyElectronOrderedOperator.local_lowest_eigenpairs = original

    assert [branch.prefix for branch in tree.root.children] == [(3,), (4,), (5,), (6,)]
    assert all(branch.basis.shape[1] <= 2 for branch in tree.iter_branches() if branch.prefix)
    right_branch = tree.root.children[-1]
    assert [child.prefix for child in right_branch.children] == [(2, 6), (3, 6), (4, 6), (5, 6)]
    assert all(len(branch.prefix) == 4 for branch in tree.iter_branches() if not branch.children)


def test_particle_growth_narg_reoptimizes_fixed_last_sectors_variationally():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    tree = particle_growth_basis(operator, D=2)
    leaves = [branch for branch in tree.iter_branches() if not branch.children]
    assert [branch.prefix for branch in tree.root.children] == [(2,), (3,), (4,), (5,), (6,)]
    assert all(branch.basis.shape[1] <= 2 for branch in tree.root.children)
    assert all(len(branch.prefix) == 3 for branch in leaves)
    assert all(branch.basis.leaf_key[0] == "particle-site" for branch in leaves)
    assert tree.shape[1] < operator.shape[0]

    sparse = tree.to_sparse()
    np.testing.assert_allclose(tree.project(operator), operator.project(sparse), atol=1e-12)

    result = particle_growth_narg(operator, D=2, nstates=4, exact=True)
    assert isinstance(result.branch_basis, ParticleGrowthLayer)
    assert isinstance(result.vectors, ParticleGrowthState)
    assert np.all(result.energies >= result.exact_energies - 1e-10)


def test_particle_growth_layer_projects_like_compatibility_tree():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=4,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    layer = particle_growth_layer(operator, D=2)
    tree = layer.to_tree()

    assert isinstance(layer, ParticleGrowthLayer)
    np.testing.assert_allclose(layer.project(operator), tree.project(operator), atol=1e-12)
    coeff = np.eye(layer.shape[1], 3)
    state = layer.truncate(coeff)
    assert isinstance(state, ParticleGrowthState)
    np.testing.assert_allclose(state.to_dense(), tree.truncate(coeff).to_dense(), atol=1e-12)


def test_particle_growth_layer_fast_nearest_neighbor_projection_matches_tree():
    grid, kinetic = _nearest_neighbor_kinetic(8)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=4,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )

    layer = particle_growth_layer(operator, D=2)
    tree = layer.to_tree()

    np.testing.assert_allclose(layer.project(operator), tree.project(operator), atol=1e-12)


def test_particle_growth_narg_uses_layer_result_without_tree_conversion():
    grid, kinetic = _nearest_neighbor_kinetic(7)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    original = ParticleGrowthLayer.to_tree

    def _unexpected_to_tree(self):
        raise AssertionError("particle_growth_narg should keep the layer-only representation")

    ParticleGrowthLayer.to_tree = _unexpected_to_tree
    try:
        result = particle_growth_narg(operator, D=2, nstates=4, exact=True)
    finally:
        ParticleGrowthLayer.to_tree = original

    assert isinstance(result, ParticleGrowthNARGResult)
    assert isinstance(result.branch_basis, ParticleGrowthLayer)
    assert isinstance(result.vectors, ParticleGrowthState)
    assert np.all(result.energies >= result.exact_energies - 1e-10)


def test_coordinate_tree_basis_project_apply_and_truncate_match_sparse_basis():
    grid, kinetic = sine_box_dvr(7, xmin=-4.0, xmax=4.0)
    operator = ordered_operator(
        kinetic,
        grid,
        nelec=3,
        external=lambda x: 0.5 * x**2,
        interaction=_soft_coulomb,
    )
    tree = coordinate_tree_basis(operator, D=2)
    sparse = tree.to_sparse()

    def _unexpected_to_sparse():
        raise AssertionError("tree operations should not collapse through to_sparse")

    def _unexpected_to_local():
        raise AssertionError("tree projection should not expand recursive bases through to_local")

    tree.to_sparse = _unexpected_to_sparse
    tree.root.to_sparse = _unexpected_to_sparse
    tree.root.basis.to_sparse = _unexpected_to_sparse

    assert isinstance(tree, CoordinateTreeBasis)
    saved_to_local = []
    for branch in tree.iter_branches():
        saved_to_local.append((branch.basis, branch.basis.to_local))
        branch.basis.to_local = _unexpected_to_local
    np.testing.assert_allclose(tree.project(operator), operator.project(sparse), atol=1e-12)
    for basis, to_local in saved_to_local:
        basis.to_local = to_local

    applied = tree.apply(operator)
    dense_applied = operator.matmat(sparse.to_dense())
    np.testing.assert_allclose(applied.to_dense(), dense_applied, atol=1e-12)

    coeff = np.eye(tree.shape[1], 2)
    np.testing.assert_allclose(tree.truncate(coeff).to_dense(), sparse.combine(coeff).to_dense(), atol=1e-12)
