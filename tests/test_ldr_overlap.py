import numpy as np

from pyqed.ldr import overlap


def test_layout_and_snake_cover_product_grid():
    indices, flat, edges = overlap.layout((2, 3, 2))

    assert indices.shape == (12, 3)
    assert flat[(1, 2, 1)] == 11
    assert len(edges) == 20

    snake = overlap.snake((2, 3, 2))
    assert set(snake) == set(map(tuple, indices))
    for left, right in zip(snake, snake[1:]):
        assert sum(abs(a - b) for a, b in zip(left, right)) == 1


def test_scalar_links_compose_and_form_hermitian_dense_overlap():
    shape = (2, 3)
    links = overlap.nearest(
        shape,
        lambda left, right: np.exp(
            1j * (0.1 + 0.2 * left[0] + 0.3 * left[1] + 0.4 * right[0])
        ),
    )

    value = overlap.between((0, 0), (1, 2), links)
    expected = links[(0, (0, 0))] * links[(1, (1, 0))] * links[(1, (1, 1))]
    np.testing.assert_allclose(value, expected)

    dense = overlap.dense(shape, links)
    np.testing.assert_allclose(dense, dense.conj().T)
    np.testing.assert_allclose(np.diag(dense), 1.0)


def test_matrix_path_average_uses_all_active_axis_orders():
    a = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    b = np.diag([1.0, 1j])
    c = np.diag([1j, 1.0])
    d = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    links = {
        (0, (0, 0)): a,
        (1, (1, 0)): b,
        (1, (0, 0)): c,
        (0, (0, 1)): d,
    }

    direct = overlap.between((0, 0), (1, 1), links, nstates=2)
    averaged = overlap.between(
        (0, 0),
        (1, 1),
        links,
        nstates=2,
        average_paths=True,
    )

    np.testing.assert_allclose(direct, a @ b)
    np.testing.assert_allclose(averaged, (a @ b + c @ d) / 2.0)


def test_full_overlap_and_link_serialization():
    objects = np.array([[0.0, 0.2], [0.3, 0.5]], dtype=object)

    def pair(left, right):
        angle = float(right) - float(left)
        return np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

    full = overlap.full(objects, pair, nstates=2)
    np.testing.assert_allclose(full, full.swapaxes(0, 2).swapaxes(1, 3).conj())

    links = overlap.nearest((2, 2), lambda left, right: pair(objects[left], objects[right]))
    axes, indices, data = overlap.pack(links, ndim=2, nstates=2)
    restored = overlap.unpack(axes, indices, data)
    assert restored.keys() == links.keys()
    for key, value in links.items():
        np.testing.assert_allclose(restored[key], value)


def test_unitary_returns_scalar_phase_and_matrix_polar_factor():
    assert overlap.unitary(2j) == 1j

    value = np.array([[2.0, 0.3], [0.0, 0.5]], dtype=complex)
    factor = overlap.unitary(value)
    np.testing.assert_allclose(factor.conj().T @ factor, np.eye(2), atol=1.0e-12)


def test_phase_gauge_cancels_local_adiabatic_signs():
    phases = np.asarray((1.0, -1.0, -1.0j, 1.0j), dtype=complex)
    links = overlap.nearest(
        (4,),
        lambda left, right: np.diag(
            phases[left[0]].conjugate() * phases[right[0]] * np.ones(2)
        ),
    )

    gauge = overlap.phase_gauge((4,), links, state=0, anchor=(1,))

    expected = phases[1] * phases.conjugate()
    np.testing.assert_allclose(gauge, expected)


def test_procrustes_factorization_supports_matrix_batches():
    singular = np.asarray((0.9, 0.35))
    left = np.asarray(
        [[1.0, 1.0j], [1.0j, 1.0]],
        dtype=complex,
    ) / np.sqrt(2.0)
    right = np.asarray(
        [[1.0, -1.0], [1.0, 1.0]],
        dtype=complex,
    ) / np.sqrt(2.0)
    matrix = left @ np.diag(singular) @ right.conj().T
    matrices = np.stack((matrix, matrix.conj()))

    rotation, positive, values = overlap.procrustes(matrices)

    np.testing.assert_allclose(
        rotation.conj().swapaxes(-1, -2) @ rotation,
        np.broadcast_to(np.eye(2), rotation.shape),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(rotation @ positive, matrices, atol=1.0e-12)
    np.testing.assert_allclose(
        positive,
        positive.conj().swapaxes(-1, -2),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(values, np.broadcast_to(singular, (2, 2)))


def test_state_tracking_and_positive_link_gauge_follow_permuted_channels():
    phases = np.asarray((1.0, -1.0, 1.0j))
    permutations = (
        np.eye(3),
        np.eye(3)[:, (0, 2, 1)],
        np.eye(3)[:, (2, 0, 1)] * phases,
    )
    links = np.asarray(
        [left.conj().T @ right for left, right in zip(permutations[:-1], permutations[1:])]
    )

    indices, selected = overlap.track_states(links, anchor=0, states=(0, 1))

    np.testing.assert_array_equal(indices, ((0, 1), (0, 2), (1, 2)))
    gauges, aligned = overlap.positive_link_gauge(selected, anchor=0)
    np.testing.assert_allclose(
        gauges.conj().swapaxes(-1, -2) @ gauges,
        np.broadcast_to(np.eye(2), gauges.shape),
        atol=1.0e-12,
    )
    for block in aligned:
        np.testing.assert_allclose(block, block.conj().T, atol=1.0e-12)
        assert np.min(np.linalg.eigvalsh(block)) > 1.0 - 1.0e-12


def test_graph_tracking_and_gauge_use_one_assignment_per_loop_vertex():
    points = ((0, 0), (0, 1), (1, 0), (1, 1))
    pairs = (
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
    )
    permutations = (
        np.eye(4),
        np.eye(4)[:, (0, 2, 1, 3)],
        np.eye(4)[:, (2, 0, 1, 3)],
        np.eye(4)[:, (1, 2, 0, 3)],
    )
    frames = dict(zip(points, permutations))
    links = np.asarray([frames[left].T @ frames[right] for left, right in pairs])

    roots, selected = overlap.track_states_graph(
        points, pairs, links, anchor=(0, 0), states=(0, 1, 2)
    )

    expected = np.asarray(
        [np.argmax(np.abs(frames[point]), axis=1)[:3] for point in points]
    )
    np.testing.assert_array_equal(roots, expected)
    gauges, aligned = overlap.synchronize_link_gauge(
        points, pairs, selected, anchor=(0, 0)
    )
    np.testing.assert_allclose(
        gauges.conj().swapaxes(-1, -2) @ gauges,
        np.broadcast_to(np.eye(3), gauges.shape),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(aligned, np.broadcast_to(np.eye(3), aligned.shape))


def test_graph_tracking_honors_multiple_fixed_root_assignments():
    points = ((0,), (1,), (2,))
    pairs = (((0,), (1,)), ((1,), (2,)))
    links = np.broadcast_to(np.eye(4), (2, 4, 4)).copy()
    roots, _selected = overlap.track_states_graph(
        points,
        pairs,
        links,
        anchor=(0,),
        states=(0, 1, 2),
        fixed={(2,): (1, 3, 0)},
    )
    np.testing.assert_array_equal(roots[0], (0, 1, 2))
    np.testing.assert_array_equal(roots[2], (1, 3, 0))
