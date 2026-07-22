import numpy as np

from pyqed.smolyak.sg import SparseGridLDR


def test_sparse_grid_ldr_overlap_is_symmetric_positive_definite():
    sg = SparseGridLDR(ndim=2, level=3, domain=((0.0, 1.0), (0.0, 1.0)))

    S = sg.build_overlap().toarray()
    T = sg.build_kinetic().toarray()

    assert sg.npts == 17
    assert np.allclose(S, S.T)
    assert np.allclose(T, T.T)
    assert np.linalg.eigvalsh(S)[0] > 0.0


def test_sparse_grid_ldr_particle_in_box_low_energies():
    sg = SparseGridLDR(ndim=1, level=5, domain=((0.0, 1.0),), mass=1.0)

    evals, _ = sg.solve(nstates=4)
    expected = 0.5 * np.pi**2 * np.arange(1, 5) ** 2

    np.testing.assert_allclose(evals, expected, rtol=1.5e-2, atol=0.0)


def test_sparse_grid_ldr_multistate_hamiltonian_shape_and_hermiticity():
    sg = SparseGridLDR(ndim=2, level=2, domain=((-1.0, 1.0), (-1.0, 1.0)))

    def potential(nodes):
        values = np.zeros((len(nodes), 2, 2), dtype=float)
        values[:, 0, 0] = 0.5 * np.sum(nodes**2, axis=1)
        values[:, 1, 1] = values[:, 0, 0] + 0.1
        values[:, 0, 1] = values[:, 1, 0] = 0.05 * nodes[:, 0]
        return values

    H = sg.build_hamiltonian(potential).toarray()
    S = sg.overlap(nstates=2).toarray()

    assert H.shape == (2 * sg.npts, 2 * sg.npts)
    assert S.shape == H.shape
    np.testing.assert_allclose(H, H.T)
    assert np.linalg.eigvalsh(S)[0] > 0.0


def test_sparse_grid_ldr_nodal_values_to_coefficients_interpolates_nodes():
    sg = SparseGridLDR(ndim=2, level=3, domain=((-1.0, 1.0), (-1.0, 1.0)))
    values = np.sin(sg.nodes[:, 0]) * np.cos(sg.nodes[:, 1])

    coeffs = sg.nodal_values_to_coefficients(values)
    reconstructed = sg.interpolation_matrix() @ coeffs

    np.testing.assert_allclose(reconstructed, values, atol=1e-12)


def test_sparse_grid_ldr_tensor_index_rule_restores_full_hierarchical_grid():
    sg = SparseGridLDR(
        ndim=2,
        level=3,
        domain=((0.0, 1.0), (0.0, 1.0)),
        index_rule="tensor",
    )

    assert sg.npts == 7**2
    assert len({tuple(node) for node in sg.nodes}) == sg.npts


def test_sparse_grid_ldr_tensor_region_refinement_adds_selected_fine_functions():
    sg = SparseGridLDR(ndim=2, level=3, domain=((0.0, 1.0), (0.0, 1.0)))
    n_smolyak = sg.npts

    sg.refine_tensor_region(
        level=3,
        predicate=lambda point: abs(point[0] - point[1]) < 0.2,
    )

    assert n_smolyak < sg.npts < 7**2
    S = sg.build_overlap().toarray()
    assert np.linalg.eigvalsh(S)[0] > 0.0


def test_sparse_grid_ldr_cellwise_quadrature_respects_hat_breakpoints():
    sg = SparseGridLDR(ndim=2, level=3, domain=((0.0, 1.0), (0.0, 1.0)))

    V = sg.build_potential_quadrature(
        lambda points: np.ones(len(points)),
        order=2,
    ).toarray()
    S = sg.build_overlap().toarray()

    np.testing.assert_allclose(V, S, atol=1e-13)


def test_sparse_grid_ldr_cellwise_quadrature_is_order_stable():
    sg = SparseGridLDR(ndim=2, level=3, domain=((0.0, 1.0), (0.0, 1.0)))
    values = sg.nodes[:, 0] + 0.3 * sg.nodes[:, 1]

    V2 = sg.build_potential_quadrature(values, order=2).toarray()
    V5 = sg.build_potential_quadrature(values, order=5).toarray()

    np.testing.assert_allclose(V2, V5, atol=1e-13)


def test_sparse_grid_ldr_hamiltonian_callable_uses_callable_quadrature():
    sg = SparseGridLDR(ndim=1, level=3, domain=((0.0, 1.0),), mass=1.0)

    def potential(points):
        return points[:, 0] ** 2

    T = sg.build_kinetic()
    V = sg.build_potential_quadrature(potential, order=5)
    H = sg.build_hamiltonian(potential, quadrature_order=5)

    np.testing.assert_allclose(H.toarray(), (T + V).toarray(), atol=1e-13)


def test_sparse_grid_ldr_propagation_preserves_generalized_norm():
    sg = SparseGridLDR(ndim=1, level=4, domain=((0.0, 1.0),), mass=1.0)
    _, evecs = sg.solve(nstates=1)
    coeff = evecs[:, 0]
    S = sg.overlap().toarray()

    propagated = sg.propagate(coeff, dt=0.1, nt=5)
    norm0 = np.vdot(coeff, S @ coeff)
    norm1 = np.vdot(propagated, S @ propagated)

    np.testing.assert_allclose(norm1, norm0, atol=1e-12)
