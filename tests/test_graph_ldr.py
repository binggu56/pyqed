import numpy as np
import scipy.linalg

from pyqed.ldr import GraphLDR, GraphMesh


def _path_mesh():
    return GraphMesh.path(np.array([0.0, 0.7, 1.8, 3.0]))


def _random_unitary(rng, size):
    raw = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    unitary, _ = np.linalg.qr(raw)
    return unitary


def test_path_mesh_keo_is_hermitian_and_annihilates_a_constant_function():
    mesh = _path_mesh()
    kinetic = mesh.kinetic().toarray()
    constant_coefficients = np.sqrt(mesh.volumes)

    np.testing.assert_allclose(kinetic, kinetic.T)
    np.testing.assert_allclose(kinetic @ constant_coefficients, 0.0, atol=1.0e-14)
    assert np.linalg.eigvalsh(kinetic)[0] > -1.0e-13


def test_triangulated_fem_keo_is_hermitian_positive_and_constant_exact():
    nodes = np.array(
        [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
    )
    mesh = GraphMesh.triangulated(nodes, [(0, 1, 2), (0, 2, 3)])
    kinetic = mesh.kinetic().toarray()

    np.testing.assert_allclose(kinetic, kinetic.T, atol=1.0e-14)
    np.testing.assert_allclose(
        kinetic @ np.sqrt(mesh.volumes),
        0.0,
        atol=1.0e-14,
    )
    assert np.linalg.eigvalsh(kinetic)[0] > -1.0e-13


def test_polar_fem_collapses_center_and_covers_stiffness_connections():
    mesh = GraphMesh.polar_fem(np.linspace(0.0, 3.0, 7), 12)

    assert mesh.size == 1 + 6 * 12
    assert mesh._stiffness is not None
    np.testing.assert_allclose(
        mesh.kinetic() @ np.sqrt(mesh.volumes),
        0.0,
        atol=1.0e-12,
    )


def test_fourth_order_rectilinear_keo_has_fourth_order_low_energy_error():
    errors = []
    for size in (15, 31):
        spacing = np.pi / (size + 1)
        axis = spacing * np.arange(1, size + 1)
        mesh = GraphMesh.rectilinear_fourth_order(axis, axis)
        energy = scipy.linalg.eigh(
            mesh.kinetic().toarray(),
            subset_by_index=(0, 0),
            eigvals_only=True,
        )[0]
        errors.append(abs(energy - 1.0))

    assert errors[0] / errors[1] > 15.0
    assert mesh.nedges > GraphMesh.rectilinear(axis, axis).nedges


def test_graph_ldr_matches_a_global_diabatic_graph_hamiltonian():
    mesh = _path_mesh()
    potential = np.zeros((mesh.size, 2, 2), dtype=complex)
    for node, coordinate in enumerate(mesh.nodes[:, 0]):
        potential[node] = np.array(
            [
                [0.2 * coordinate**2, 0.1 + 0.03j * coordinate],
                [0.1 - 0.03j * coordinate, 0.4 + 0.05 * coordinate],
            ]
        )

    solver = GraphLDR(mesh, 2).set_diabatic(potential)
    global_hamiltonian = np.kron(
        mesh.kinetic().toarray(), np.eye(2, dtype=complex)
    )
    global_hamiltonian += scipy.linalg.block_diag(*potential)
    transform = scipy.linalg.block_diag(*solver.frames)

    expected = transform.conj().T @ global_hamiltonian @ transform
    np.testing.assert_allclose(solver.hamiltonian(), expected, atol=1.0e-12)


def test_graph_ldr_is_covariant_under_independent_node_gauges():
    mesh = _path_mesh()
    rng = np.random.default_rng(12)
    nstates = 2
    potential = np.empty((mesh.size, nstates, nstates), dtype=complex)
    for node in range(mesh.size):
        raw = rng.normal(size=(nstates, nstates))
        potential[node] = raw + raw.T
    overlaps = np.empty((mesh.nedges, nstates, nstates), dtype=complex)
    for edge in range(mesh.nedges):
        left = _random_unitary(rng, nstates)
        right = _random_unitary(rng, nstates)
        overlaps[edge] = left @ np.diag((0.98, 0.91)) @ right

    reference = GraphLDR(
        mesh,
        nstates,
        potential=potential,
        overlaps=overlaps,
    )
    gauges = np.asarray([_random_unitary(rng, nstates) for _ in range(mesh.size)])
    transformed_potential = np.einsum(
        "mai,mab,mbj->mij",
        gauges.conj(),
        potential,
        gauges,
        optimize=True,
    )
    transformed_overlaps = np.asarray(
        [
            gauges[left].conj().T @ overlaps[edge] @ gauges[right]
            for edge, (left, right) in enumerate(mesh.edges)
        ]
    )
    transformed = GraphLDR(
        mesh,
        nstates,
        potential=transformed_potential,
        overlaps=transformed_overlaps,
    )
    global_gauge = scipy.linalg.block_diag(*gauges)

    expected = global_gauge.conj().T @ reference.hamiltonian() @ global_gauge
    np.testing.assert_allclose(transformed.hamiltonian(), expected, atol=1.0e-12)
    np.testing.assert_allclose(
        transformed.hamiltonian(),
        transformed.hamiltonian().conj().T,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        transformed.overlap_singular_values(),
        reference.overlap_singular_values(),
        atol=1.0e-12,
    )
    np.testing.assert_array_equal(
        reference.poorly_resolved_edges(0.95),
        mesh.edges,
    )


def test_graph_ldr_sparse_propagation_preserves_norm():
    mesh = _path_mesh()
    energies = np.column_stack(
        (0.1 * mesh.nodes[:, 0] ** 2, 0.3 + 0.05 * mesh.nodes[:, 0])
    )
    angles = 0.2 * mesh.nodes[:, 0]
    frames = np.asarray(
        [
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
            for angle in angles
        ],
        dtype=complex,
    )
    overlaps = np.asarray(
        [frames[left].conj().T @ frames[right] for left, right in mesh.edges]
    )
    solver = GraphLDR(mesh, 2, energies=energies, overlaps=overlaps)
    rng = np.random.default_rng(3)
    state = rng.normal(size=(mesh.size, 2)) + 1j * rng.normal(
        size=(mesh.size, 2)
    )
    state /= np.linalg.norm(state)

    solver.run(state, dt=0.04, nsteps=5, nout=1)

    np.testing.assert_allclose(solver.norm, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        solver.hamiltonian(matrix_free=True) @ state.reshape(-1),
        solver.hamiltonian() @ state.reshape(-1),
        atol=1.0e-12,
    )
