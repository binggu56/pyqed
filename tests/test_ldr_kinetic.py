import numpy as np
import scipy.sparse as sp

from pyqed.dvr import DVR, ExponentialDVR, HermiteDVR, SineDVR
from pyqed.ldr import LDR
from pyqed.ldr import kinetic
from pyqed.ldr import overlap


def _links():
    return {
        (0, (0,)): np.array([[0.9, 0.1], [-0.1, 0.9]], dtype=complex),
        (0, (1,)): np.array([[0.8, 0.2j], [0.2j, 0.8]], dtype=complex),
    }


def test_dress_accepts_sparse_kinetic_and_exact_overlap_callback():
    nuclear = sp.csr_matrix(np.array([[1.0, -0.2], [-0.2, 0.7]]))
    blocks = {
        (0, 0): np.eye(2),
        (0, 1): np.array([[0.9, 0.1j], [0.1j, 0.8]]),
        (1, 0): np.array([[0.9, -0.1j], [-0.1j, 0.8]]),
        (1, 1): np.eye(2),
    }

    actual = kinetic.dress(
        nuclear,
        lambda i, j: blocks[(i, j)],
        nstates=2,
    )
    expected = np.zeros((2, 2, 2, 2), dtype=complex)
    for i in range(2):
        for j in range(2):
            expected[i, :, j, :] = nuclear[i, j] * blocks[(i, j)]

    np.testing.assert_allclose(actual.toarray(), expected.reshape(4, 4))


def test_linked_sparse_matrix_and_operator_agree():
    nuclear = np.array(
        [[1.0, 0.2, -0.05], [0.2, 1.1, 0.3], [-0.05, 0.3, 0.7]],
        dtype=complex,
    )
    links = _links()
    dense = kinetic.matrix(nuclear, (3,), 2, links=links)
    sparse = kinetic.linked(nuclear, (3,), links, nstates=2)
    vector = np.arange(6, dtype=float) + 0.1j * np.arange(6)

    np.testing.assert_allclose(sparse.toarray(), dense)
    np.testing.assert_allclose(
        kinetic.operator(nuclear, (3,), 2, links=links) @ vector,
        dense @ vector,
    )
    np.testing.assert_allclose(
        kinetic.operator(sp.csr_matrix(nuclear), (3,), 2, links=links) @ vector,
        dense @ vector,
    )


def test_periodic_linked_fourier_kinetic_is_gauge_covariant():
    size = 5
    nuclear = ExponentialDVR(npts=size, L=2.0 * np.pi).t()
    phases = np.exp(1j * np.asarray((0.1, -0.2, 0.35, 0.7, -0.4)))
    links = {
        (0, (index,)): np.exp(0.13j * (index + 1))
        for index in range(size)
    }
    transformed = {
        (0, (index,)): (
            phases[index].conjugate()
            * links[(0, (index,))]
            * phases[(index + 1) % size]
        )
        for index in range(size)
    }
    reference = kinetic.matrix(
        nuclear, (size,), 1, links=links, periodic_axes=(0,)
    )
    actual = kinetic.matrix(
        nuclear, (size,), 1, links=transformed, periodic_axes=(0,)
    )
    gauge = np.diag(phases)

    np.testing.assert_allclose(actual, gauge.conj().T @ reference @ gauge)
    np.testing.assert_allclose(actual, actual.conj().T)


def test_dense_overlap_and_link_representations_give_same_action():
    nuclear = np.array(
        [[0.6, -0.2, 0.1], [-0.2, 0.8, -0.15], [0.1, -0.15, 0.9]],
        dtype=complex,
    )
    links = _links()
    overlaps = overlap.dense((3,), links, nstates=2)
    linked_matrix = kinetic.matrix(nuclear, (3,), 2, links=links)
    overlap_matrix = kinetic.matrix(nuclear, (3,), 2, overlaps=overlaps)

    np.testing.assert_allclose(overlap_matrix, linked_matrix)
    vector = np.linspace(0.1, 1.2, 6).astype(complex)
    np.testing.assert_allclose(
        kinetic.operator(nuclear, (3,), 2, overlaps=overlaps) @ vector,
        overlap_matrix @ vector,
    )


def test_rotational_dense_and_sparse_matrix_free_actions_match():
    rng = np.random.default_rng(7)
    raw = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    nuclear = raw + raw.conj().T
    links = {(0, (0,)): np.array([[0.9, 0.1j], [0.1j, 0.85]])}
    dense = kinetic.matrix(nuclear, (2,), 2, links=links, nrot=2)
    vector = rng.normal(size=8) + 1j * rng.normal(size=8)

    np.testing.assert_allclose(
        kinetic.operator(nuclear, (2,), 2, links=links, nrot=2) @ vector,
        dense @ vector,
    )
    np.testing.assert_allclose(
        kinetic.operator(
            sp.csr_matrix(nuclear),
            (2,),
            2,
            links=links,
            nrot=2,
        )
        @ vector,
        dense @ vector,
    )


def test_prefix_mpos_match_dense_nonunitary_link_products():
    ngrid = 7
    nuclear = ExponentialDVR(npts=ngrid, L=8.0, mass=2.0).t()
    links = {}
    for index in range(ngrid - 1):
        angle = 0.03 * (index + 1)
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        links[(0, (index,))] = rotation @ np.diag([0.98, 0.94])

    components, info = kinetic.prefix_mpos(nuclear, links)
    actual = sum(component.to_dense() for component in components)
    expected = kinetic.matrix(
        nuclear,
        (ngrid,),
        2,
        links=links,
        symmetrize=False,
    )

    np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    fft = kinetic.PrefixFFT(nuclear, links)
    descriptor_fft = kinetic.PrefixFFT(
        ExponentialDVR(npts=ngrid, L=8.0, mass=2.0).kinetic_toeplitz(),
        links,
    )
    rng = np.random.default_rng(8)
    vector = rng.normal(size=ngrid * 2) + 1j * rng.normal(size=ngrid * 2)
    vectors = rng.normal(size=(ngrid * 2, 3))
    np.testing.assert_allclose(fft.matvec(vector), expected @ vector, atol=1.0e-11)
    np.testing.assert_allclose(
        descriptor_fft.matvec(vector), expected @ vector, atol=1.0e-11
    )
    np.testing.assert_allclose(fft.matmat(vectors), expected @ vectors, atol=1.0e-11)
    np.testing.assert_allclose(
        fft.aslinearoperator() @ vector,
        expected @ vector,
        atol=1.0e-11,
    )
    assert info["circulant_error"] < 1.0e-13
    assert fft.info["toeplitz_error"] < 1.0e-13
    assert descriptor_fft.info["descriptor"]
    assert info["max_link_condition"] > 1.0
    assert info["max_prefix_condition"] > info["max_link_condition"]
    assert max(info["operator_ranks"][1]) <= 4


def test_ldr_selects_prefix_fft_for_periodic_one_dimensional_grid():
    ngrid = 9
    axis = ExponentialDVR(npts=ngrid, L=8.0, mass=2.0)
    dvr = DVR.from_axes((axis,))
    axis.t = lambda: (_ for _ in ()).throw(AssertionError("dense KEO constructed"))
    links = {
        (0, (index,)): np.array(
            [[0.98, 0.01j * (index + 1)], [0.005j, 0.95]], dtype=complex
        )
        for index in range(ngrid - 1)
    }
    solver = LDR(dvr, 2, links=links)
    vector = np.random.default_rng(12).normal(size=solver.size)

    actual = solver.kinetic_operator() @ vector
    expected = solver.kinetic_matrix() @ vector

    np.testing.assert_allclose(actual, expected, atol=1.0e-11)
    assert isinstance(solver.kinetic, tuple)
    assert dvr._K is None
    assert solver.kinetic_info["backend"] == "prefix-fft"
    assert solver.kinetic_info["descriptor"]


def test_sine_prefix_fft_matches_dense_nonunitary_link_products_without_dense_keo():
    ngrid = 9
    axis = SineDVR(-2.0, 3.0, ngrid, mass=2.5)
    dense_kinetic = axis.t()
    links = {
        (0, (index,)): np.array(
            [
                [0.98 - 0.002 * index, 0.01j * (index + 1)],
                [0.004j * (index + 1), 0.95 + 0.001 * index],
            ],
            dtype=complex,
        )
        for index in range(ngrid - 1)
    }
    expected = kinetic.matrix(
        dense_kinetic,
        (ngrid,),
        2,
        links=links,
        symmetrize=False,
    )
    axis.T = None
    axis.t = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("dense sine KEO constructed")
    )
    solver = LDR(DVR.from_axes((axis,)), 2, links=links)
    rng = np.random.default_rng(15)
    vector = rng.normal(size=solver.size) + 1j * rng.normal(size=solver.size)

    np.testing.assert_allclose(
        solver.kinetic_operator() @ vector,
        expected @ vector,
        atol=2.0e-11,
    )
    np.testing.assert_allclose(solver.kinetic_matrix(), expected, atol=2.0e-11)
    np.testing.assert_allclose(
        solver._trace(),
        np.trace(expected),
        atol=2.0e-11,
    )
    assert solver.kinetic_info["backend"] == "prefix-fft"
    assert solver.kinetic_info["structure"] == "toeplitz-hankel"
    assert solver.kinetic_info["descriptor"]


def test_ldr_prefix_fft_can_be_selected_or_bypassed():
    ngrid = 7
    dvr = DVR.from_axes((ExponentialDVR(npts=ngrid, L=6.0),))
    links = {
        (0, (index,)): np.diag([0.98, 0.94])
        for index in range(ngrid - 1)
    }
    solver = LDR(dvr, 2, links=links, kinetic_backend="prefix-fft")
    vector = np.arange(solver.size, dtype=float)

    np.testing.assert_allclose(
        solver.kinetic_operator() @ vector,
        solver.kinetic_operator(backend="generic") @ vector,
        atol=1.0e-11,
    )
    assert solver.kinetic_info["backend"] == "generic"


def test_two_dimensional_prefix_fft_matches_full_ldr():
    shape = (5, 6)
    axes = (
        ExponentialDVR(npts=shape[0], L=7.0, mass=2.0),
        ExponentialDVR(npts=shape[1], L=9.0, mass=3.0),
    )
    descriptors = tuple(axis.kinetic_toeplitz() for axis in axes)
    matrices = tuple(axis.t() for axis in axes)
    nuclear = np.kron(matrices[0], np.eye(shape[1]))
    nuclear += np.kron(np.eye(shape[0]), matrices[1])

    links = {}
    for axis, axis_size in enumerate(shape):
        for point in np.ndindex(*shape):
            if point[axis] == axis_size - 1:
                continue
            phase = 0.025 * (axis + 1) * (1 + sum(point))
            rotation = np.array(
                [[np.cos(phase), -np.sin(phase)],
                 [np.sin(phase), np.cos(phase)]],
                dtype=complex,
            )
            contraction = np.diag(
                [0.985 - 0.002 * point[axis], 0.96 + 0.003 * point[1 - axis]]
            )
            links[(axis, point)] = rotation @ contraction

    reference = kinetic.matrix(
        nuclear,
        shape,
        2,
        links=links,
        symmetrize=False,
    )
    fft = kinetic.PrefixFFTND(descriptors, shape, links)
    rng = np.random.default_rng(21)
    vector = rng.normal(size=reference.shape[0]) + 1j * rng.normal(
        size=reference.shape[0]
    )
    vectors = rng.normal(size=(reference.shape[0], 3))

    np.testing.assert_allclose(fft.matvec(vector), reference @ vector, atol=1.0e-10)
    np.testing.assert_allclose(fft.matmat(vectors), reference @ vectors, atol=1.0e-10)
    np.testing.assert_allclose(
        fft.aslinearoperator() @ vector, reference @ vector, atol=1.0e-10
    )
    assert fft.info["backend"] == "prefix-fft-nd"
    assert all(info["descriptor"] for info in fft.info["axes"])


def test_multidimensional_sine_prefix_fft_matches_full_ldr():
    shape = (7, 6)
    axes = (
        SineDVR(-1.0, 2.0, shape[0], mass=2.0),
        ExponentialDVR(npts=shape[1], L=8.0, mass=3.0),
    )
    descriptors = (
        axes[0].kinetic_descriptor(),
        axes[1].kinetic_toeplitz(),
    )
    matrices = tuple(axis.t() for axis in axes)
    nuclear = np.kron(matrices[0], np.eye(shape[1]))
    nuclear += np.kron(np.eye(shape[0]), matrices[1])
    links = {}
    for axis, size in enumerate(shape):
        for point in np.ndindex(shape):
            if point[axis] == size - 1:
                continue
            angle = 0.018 * (1 + axis + sum(point))
            links[(axis, point)] = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=complex,
            ) @ np.diag([0.99, 0.95])
    reference = kinetic.matrix(
        nuclear, shape, 2, links=links, symmetrize=False
    )
    operator = kinetic.PrefixFFTND(descriptors, shape, links)
    rng = np.random.default_rng(25)
    vector = rng.normal(size=reference.shape[0]) + 1j * rng.normal(
        size=reference.shape[0]
    )
    vectors = rng.normal(size=(reference.shape[0], 3))

    np.testing.assert_allclose(operator.matvec(vector), reference @ vector, atol=2e-11)
    np.testing.assert_allclose(operator.matmat(vectors), reference @ vectors, atol=2e-11)
    assert [item["backend"] for item in operator.info["axes"]] == [
        "sine-fft",
        "fft",
    ]


def test_ldr_auto_selects_multidimensional_sine_prefix_fft():
    axes = (
        SineDVR(-2.0, 2.0, 4, mass=2.0),
        SineDVR(-1.5, 2.5, 5, mass=3.0),
    )
    grid = DVR.from_axes(axes)
    links = {}
    for axis, size in enumerate(grid.shape):
        for point in np.ndindex(grid.shape):
            if point[axis] == size - 1:
                continue
            angle = 0.02 * (1 + axis + sum(point))
            links[(axis, point)] = np.array(
                [
                    [np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)],
                ],
                dtype=complex,
            ) @ np.diag((0.99, 0.96))

    solver = LDR(grid, 2, links=links)
    vector = np.random.default_rng(31).normal(size=solver.size)
    assert solver.kinetic is None
    assert grid._K is None

    automatic = solver.kinetic_operator()
    automatic_info = dict(solver.kinetic_info)
    solver._trace()
    solver.run(vector / np.linalg.norm(vector), dt=0.01, nsteps=2)

    assert grid._K is None
    generic = solver.kinetic_operator(backend="generic")

    np.testing.assert_allclose(automatic @ vector, generic @ vector, atol=2.0e-11)
    assert grid._K is not None
    assert automatic_info["backend"] == "prefix-fft-nd"
    assert [item["backend"] for item in automatic_info["axes"]] == [
        "sine-fft",
        "sine-fft",
    ]


def test_mixed_periodic_and_hermite_prefix_operator_matches_full_ldr():
    shape = (5, 6)
    periodic = ExponentialDVR(npts=shape[0], L=2.0 * np.pi, mass=2.0)
    harmonic = HermiteDVR(npts=shape[1], mass=3.0, omega=0.4)
    axis_kinetics = (periodic.kinetic_toeplitz(), harmonic.t())
    nuclear = np.kron(periodic.t(), np.eye(shape[1]))
    nuclear += np.kron(np.eye(shape[0]), harmonic.t())
    links = {}
    for axis, axis_size in enumerate(shape):
        for point in np.ndindex(*shape):
            if point[axis] == axis_size - 1:
                continue
            angle = 0.02 * (1 + axis + sum(point))
            links[(axis, point)] = np.array(
                [[np.cos(angle), -np.sin(angle)],
                 [np.sin(angle), np.cos(angle)]],
                dtype=complex,
            ) @ np.diag([0.99, 0.96])

    reference = kinetic.matrix(
        nuclear, shape, 2, links=links, symmetrize=False
    )
    operator = kinetic.PrefixFFTND(axis_kinetics, shape, links)
    vector = np.random.default_rng(22).normal(size=reference.shape[0])

    np.testing.assert_allclose(
        operator.matvec(vector), reference @ vector, atol=1.0e-10
    )
    assert [info["backend"] for info in operator.info["axes"]] == [
        "fft",
        "direct",
    ]
