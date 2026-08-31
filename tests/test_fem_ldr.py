import numpy as np
import scipy.linalg
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh

from pyqed.ldr import FEMLDR, TriangularMesh


def _mesh(order=2):
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    return TriangularMesh.from_vertices(
        nodes,
        [(0, 1, 2), (0, 2, 3)],
        order=order,
    )


def _potential(nodes):
    potential = np.empty((len(nodes), 2, 2), dtype=complex)
    for index, (x, y) in enumerate(nodes):
        potential[index] = np.array(
            [[0.2 + x, 0.1 + 0.05j * y], [0.1 - 0.05j * y, 0.4 - x]]
        )
    return potential


def test_quadratic_triangle_mass_and_stiffness_are_variational():
    mesh = _mesh(order=2)
    constant = np.ones(mesh.size)

    np.testing.assert_allclose(mesh.mass.toarray(), mesh.mass.toarray().T)
    np.testing.assert_allclose(
        mesh.stiffness.toarray(),
        mesh.stiffness.toarray().T,
    )
    np.testing.assert_allclose(constant @ mesh.mass @ constant, 1.0)
    np.testing.assert_allclose(mesh.stiffness @ constant, 0.0, atol=1.0e-13)
    assert np.linalg.eigvalsh(mesh.mass.toarray())[0] > 0.0
    assert np.linalg.eigvalsh(mesh.stiffness.toarray())[0] > -1.0e-12


def test_quadratic_cartesian_dirichlet_mesh_matches_interior_half_grid():
    mesh = TriangularMesh.cartesian(((-1.0, 1.0), (-1.0, 1.0)), 4, order=2)

    assert mesh.size == 49
    assert np.all(np.abs(mesh.nodes) < 1.0)
    assert np.linalg.eigvalsh(mesh.mass.toarray())[0] > 0.0
    assert np.linalg.eigvalsh(mesh.stiffness.toarray())[0] > 0.0


def test_quadratic_elements_improve_ground_laplacian_at_fixed_node_count():
    linear = TriangularMesh.cartesian(((0.0, 1.0), (0.0, 1.0)), 8, order=1)
    quadratic = TriangularMesh.cartesian(
        ((0.0, 1.0), (0.0, 1.0)),
        4,
        order=2,
    )
    exact = 2.0 * np.pi**2

    linear_value = eigsh(
        linear.stiffness,
        k=1,
        M=linear.mass,
        which="SM",
        return_eigenvectors=False,
    )[0]
    quadratic_value = eigsh(
        quadratic.stiffness,
        k=1,
        M=quadratic.mass,
        which="SM",
        return_eigenvectors=False,
    )[0]

    assert linear.size == quadratic.size == 49
    assert abs(quadratic_value - exact) < 0.1 * abs(linear_value - exact)


def test_red_green_refinement_is_conforming_and_preserves_area():
    mesh = _mesh(order=2)
    refined = mesh.refine([0])
    vertices = refined.nodes[: refined.vertex_count]
    coordinates = vertices[refined.vertex_triangles]
    first = coordinates[:, 1] - coordinates[:, 0]
    second = coordinates[:, 2] - coordinates[:, 0]
    signed_twice_area = (
        first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    )

    assert refined.order == 2
    assert refined.vertex_count == 7
    assert len(refined.vertex_triangles) == 6
    np.testing.assert_allclose(0.5 * np.sum(np.abs(signed_twice_area)), 1.0)
    np.testing.assert_allclose(
        refined.stiffness @ np.ones(refined.size),
        0.0,
        atol=1.0e-12,
    )


def test_dorfler_mark_selects_the_smallest_dominant_set():
    marked = TriangularMesh.dorfler_mark([1.0, 4.0, 2.0, 3.0], theta=0.6)

    np.testing.assert_array_equal(marked, [1, 3])


def test_target_size_refinement_selects_the_closest_available_mesh():
    mesh = _mesh(order=2)
    indicators = np.array([1.0, 2.0])
    refined, marked = mesh.refine_to_size(indicators, 15)
    ordering = np.argsort(indicators)[::-1]
    candidates = [mesh.refine(ordering[:count]).size for count in (0, 1, 2)]

    assert abs(refined.size - 15) == min(abs(size - 15) for size in candidates)
    assert len(marked) > 0


def test_gradient_indicator_tracks_nodal_field_localization():
    mesh = _mesh(order=2)
    field = np.exp(-20.0 * np.sum((mesh.nodes - [0.8, 0.2]) ** 2, axis=1))
    indicators = mesh.gradient_indicators(field)

    assert indicators.shape == (2,)
    assert indicators[0] > indicators[1]


def test_fem_ldr_is_exactly_gauge_equivalent_to_full_diabatic_fem():
    mesh = _mesh(order=2)
    potential = _potential(mesh.nodes)
    solver = FEMLDR(mesh, 2).set_diabatic(potential)
    identity = sp.eye(2, format="csr")
    global_mass = sp.kron(mesh.mass, identity, format="csr").toarray()
    global_stiffness = sp.kron(
        mesh.stiffness,
        identity,
        format="csr",
    ).toarray()
    global_potential = scipy.linalg.block_diag(*potential)
    global_hamiltonian = 0.5 * global_stiffness + 0.5 * (
        global_mass @ global_potential + global_potential @ global_mass
    )
    transform = scipy.linalg.block_diag(*solver.frames)

    np.testing.assert_allclose(
        solver.mass_matrix().toarray(),
        transform.conj().T @ global_mass @ transform,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        solver.hamiltonian().toarray(),
        transform.conj().T @ global_hamiltonian @ transform,
        atol=1.0e-12,
    )


def test_projector_indicator_is_invariant_to_local_eigenvector_phases():
    solver = FEMLDR(_mesh(order=2), 2).set_diabatic(_potential(_mesh().nodes))
    reference = solver.projector_indicators()
    phases = np.exp(1j * np.arange(solver.ngrid * solver.nstates)).reshape(
        solver.ngrid,
        solver.nstates,
    )
    solver.frames *= phases[:, None, :]

    np.testing.assert_allclose(solver.projector_indicators(), reference)


def test_residual_indicator_vanishes_for_a_constant_free_solution():
    mesh = _mesh(order=2)
    solver = FEMLDR(mesh, 1).set_diabatic(
        np.zeros((mesh.size, 1, 1), dtype=complex)
    )
    states = np.ones((3, mesh.size, 1), dtype=complex)
    indicators = solver.residual_indicators(states, [0.0, 0.1, 0.2])

    np.testing.assert_allclose(indicators, 0.0, atol=1.0e-24)


def test_residual_indicator_is_invariant_to_local_eigenvector_phases():
    mesh = _mesh(order=2)
    solver = FEMLDR(mesh, 2).set_diabatic(_potential(mesh.nodes))
    rng = np.random.default_rng(8)
    states = rng.normal(size=(3, mesh.size, 2)) + 1j * rng.normal(
        size=(3, mesh.size, 2)
    )
    times = np.array([0.0, 0.1, 0.2])
    reference = solver.residual_indicators(states, times)
    phases = np.exp(
        1j * np.arange(solver.ngrid * solver.nstates)
    ).reshape(solver.ngrid, solver.nstates)
    solver.frames *= phases[:, None, :]
    transformed = states * phases.conj()[None, :, :]

    np.testing.assert_allclose(
        solver.residual_indicators(transformed, times),
        reference,
        atol=1.0e-11,
    )


def test_generalized_crank_nicolson_preserves_mass_norm_and_populations():
    mesh = _mesh(order=2)
    solver = FEMLDR(mesh, 2).set_diabatic(_potential(mesh.nodes))
    rng = np.random.default_rng(4)
    state = rng.normal(size=(mesh.size, 2)) + 1j * rng.normal(
        size=(mesh.size, 2)
    )
    state = solver.normalize(state)

    solver.run(state, dt=0.02, nsteps=20, nout=2)
    adiabatic = solver.adiabatic_populations()
    diabatic = solver.diabatic_populations()

    np.testing.assert_allclose(solver.norm, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(adiabatic.sum(axis=1), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(diabatic.sum(axis=1), 1.0, atol=1.0e-12)
