import numpy as np
import pytest

from pyqed.mps import FunctionalTT
from pyqed.mps.functional import PiecewisePCHIP, load_field_model
from pyqed.mps.cross import tt_cross
from pyqed.mps.decompose import tt_to_tensor


def _polynomial_surface(coordinates):
    x, y, z = np.asarray(coordinates).T
    return 1.2 + (1.0 + 0.3 * x) * (0.5 - y) * (1.0 + z**2) + 0.2 * x * y * z


def test_functional_tt_fits_scattered_low_rank_surface():
    rng = np.random.default_rng(2)
    training = rng.uniform(-1.0, 1.0, size=(500, 3))
    validation = rng.uniform(-1.0, 1.0, size=(150, 3))
    model = FunctionalTT(
        degrees=3,
        rank=3,
        bounds=((-1.0, 1.0),) * 3,
        regularization=1.0e-12,
        sweeps=20,
        rtol=1.0e-9,
        patience=4,
        random_state=3,
    ).fit(training, _polynomial_surface(training))

    predicted = model.predict(validation)

    np.testing.assert_allclose(
        predicted,
        _polynomial_surface(validation),
        atol=2.0e-6,
    )
    assert model.ranks_ == (1, 3, 3, 1)
    assert model.success


def test_functional_tt_batched_prediction_matches_materialized_core_values():
    rng = np.random.default_rng(41)
    training = rng.uniform(-1.0, 1.0, size=(240, 3))
    probes = rng.uniform(-1.0, 1.0, size=(37, 3))
    model = FunctionalTT(
        degrees=3,
        rank=3,
        bounds=((-1.0, 1.0),) * 3,
        regularization=1.0e-12,
        sweeps=8,
        random_state=9,
    ).fit(training, _polynomial_surface(training))
    reference = model._denormalize(
        model._contract(model._core_values(model._basis_matrices(probes)))
    ).reshape(len(probes))

    np.testing.assert_allclose(
        model.predict(probes, batch_size=7),
        reference,
        atol=1.0e-13,
        rtol=1.0e-13,
    )
    with pytest.raises(ValueError, match="batch_size"):
        model.predict(probes, batch_size=0)


def test_functional_tt_fourier_basis_is_periodic_and_serializable(tmp_path):
    rng = np.random.default_rng(5)
    coordinates = np.column_stack(
        (rng.uniform(-1.0, 1.0, 400), rng.uniform(0.0, 2.0 * np.pi, 400))
    )

    def surface(points):
        x, phi = np.asarray(points).T
        return 0.7 + (1.0 + 0.2 * x) * (np.cos(phi) + 0.3 * np.sin(2.0 * phi))

    model = FunctionalTT(
        bases=("legendre", "fourier"),
        degrees=(2, 2),
        rank=2,
        bounds=((-1.0, 1.0), (0.0, 2.0 * np.pi)),
        regularization=1.0e-12,
        sweeps=12,
        rtol=1.0e-9,
        random_state=2,
    ).fit(coordinates, surface(coordinates))
    filename = tmp_path / "potential_ft.npz"
    model.save(filename)
    loaded = FunctionalTT.load(filename)

    probes = np.array(((0.2, 0.0), (0.2, 2.0 * np.pi), (-0.3, 1.1)))

    np.testing.assert_allclose(model.predict(probes), surface(probes), atol=2.0e-8)
    np.testing.assert_allclose(loaded.predict(probes), model.predict(probes))
    np.testing.assert_allclose(model.predict(probes[0]), model.predict(probes[1]))


def test_piecewise_pchip_preserves_nodes_hermiticity_and_serializes(tmp_path):
    coordinates = np.asarray((0.0, 0.1, 0.4, 1.0))
    values = np.empty((len(coordinates), 2, 2), dtype=complex)
    values[:, 0, 0] = coordinates**2
    values[:, 1, 1] = 1.0 - 0.2 * coordinates
    values[:, 0, 1] = 0.1j * coordinates
    values[:, 1, 0] = values[:, 0, 1].conj()
    model = PiecewisePCHIP(hermitian=True).fit(coordinates, values)

    np.testing.assert_allclose(model.predict(coordinates[:, None]), values)
    dense = model.predict(np.linspace(0.0, 1.0, 101)[:, None])
    np.testing.assert_allclose(dense, dense.conj().swapaxes(-1, -2))
    with pytest.raises(ValueError, match="outside"):
        model.predict(np.asarray((1.1,)))

    filename = tmp_path / "piecewise.npz"
    model.save(filename)
    restored = load_field_model(filename)
    np.testing.assert_allclose(
        restored.predict(np.asarray(((0.25,), (0.75,)))),
        model.predict(np.asarray(((0.25,), (0.75,)))),
    )


def test_functional_tt_builds_diagonal_mpo_without_materializing_grid():
    rng = np.random.default_rng(8)
    coordinates = rng.uniform(-1.0, 1.0, size=(400, 3))
    model = FunctionalTT(
        degrees=3,
        rank=3,
        bounds=((-1.0, 1.0),) * 3,
        regularization=1.0e-12,
        sweeps=20,
        rtol=1.0e-9,
        random_state=4,
    ).fit(coordinates, _polynomial_surface(coordinates))
    grids = (
        np.linspace(-1.0, 1.0, 4),
        np.linspace(-1.0, 1.0, 3),
        np.linspace(-1.0, 1.0, 2),
    )
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack([axis.reshape(-1) for axis in mesh], axis=1)

    mpo = model.mpo(grids)
    dense = mpo.to_dense()

    assert mpo.L == len(grids)
    np.testing.assert_allclose(np.diag(dense), model.predict(points), atol=1.0e-12)
    np.testing.assert_allclose(dense, np.diag(np.diag(dense)), atol=1.0e-14)


def test_functional_tt_refines_tt_cross_cores_off_grid():
    grids = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 6),
        np.linspace(-1.0, 1.0, 5),
    )

    def surface(points):
        x, y, z = np.asarray(points).T
        return 0.4 + (1.0 + 0.2 * x) * (0.7 - y) * (1.0 + z**2) + 0.1 * x * z

    def oracle(index):
        point = np.array([grid[position] for grid, position in zip(grids, index)])
        return surface(point[None, :])[0]

    cores, info = tt_cross(
        tuple(len(grid) for grid in grids),
        oracle,
        max_rank=3,
        sweeps=5,
        rtol=1.0e-12,
        validation=80,
        seed=4,
        return_state=True,
    )
    indices = tuple(info["state"]["cache"])
    coordinates = np.array(
        [[grid[position] for grid, position in zip(grids, index)] for index in indices]
    )
    values = surface(coordinates)
    model = FunctionalTT(
        degrees=(2, 1, 2),
        rank=3,
        bounds=((-1.0, 1.0),) * 3,
    ).fit_from_cross(
        grids,
        cores,
        coordinates,
        values,
        sweeps=30,
        rtol=1.0e-10,
        atol=1.0e-7,
    )
    rng = np.random.default_rng(12)
    probes = rng.uniform(-1.0, 1.0, size=(100, 3))

    np.testing.assert_allclose(model.predict(probes), surface(probes), atol=2.0e-7)
    assert model.refinement["method"] == "canonical_als"
    assert model.refinement["converged"]
    assert model.refinement["final_error"] <= model.refinement["initial_error"]


def test_functional_tt_jointly_fits_complex_matrices_and_serializes(tmp_path):
    def matrices(points):
        x, y = np.asarray(points).T
        diagonal_0 = 1.0 + 0.2 * x + 0.1 * y**2
        coupling = 0.3 * x * y + 0.1j * (x - y)
        diagonal_1 = 0.7 - 0.3 * y + 0.2 * x**2
        output = np.empty((len(points), 2, 2), dtype=complex)
        output[:, 0, 0] = diagonal_0
        output[:, 0, 1] = coupling
        output[:, 1, 0] = coupling.conj()
        output[:, 1, 1] = diagonal_1
        return output

    rng = np.random.default_rng(17)
    training = rng.uniform(-1.0, 1.0, size=(500, 2))
    probes = rng.uniform(-1.0, 1.0, size=(6, 7, 2))
    model = FunctionalTT(
        degrees=2,
        rank=4,
        bounds=((-1.0, 1.0),) * 2,
        regularization=1.0e-12,
        sweeps=12,
        rtol=1.0e-10,
        random_state=9,
    ).fit(training, matrices(training))

    predicted = model.predict(probes)

    assert predicted.shape == (6, 7, 2, 2)
    assert model.normalization == "frobenius"
    assert model.hermitian_
    assert model.output_shape_ == (2, 2)
    assert model.output_core.shape == (model.ranks_[-1], 4)
    assert np.isrealobj(model.output_core)
    assert all(np.isrealobj(core) for core in model.cores)
    training_values = matrices(training).reshape(len(training), -1)
    centered = training_values - np.mean(training_values, axis=0)
    expected_scale = np.sqrt(np.mean(np.sum(np.abs(centered) ** 2, axis=1)))
    np.testing.assert_allclose(model.scale, expected_scale)
    encoded = model._encode_values(matrices(training))
    np.testing.assert_allclose(
        np.linalg.norm(encoded),
        np.linalg.norm(matrices(training)),
    )
    expected = matrices(probes.reshape(-1, 2)).reshape(predicted.shape)
    np.testing.assert_allclose(predicted, expected, atol=1.0e-11)
    np.testing.assert_allclose(
        predicted,
        predicted.conj().swapaxes(-1, -2),
        atol=0.0,
    )

    filename = tmp_path / "matrix_functional_tt.npz"
    model.save(filename)
    loaded = FunctionalTT.load(filename)
    assert loaded.normalization == "frobenius"
    assert loaded.hermitian_
    np.testing.assert_allclose(loaded.predict(probes), predicted, atol=1.0e-13)


def test_complex_feature_tt_realification_preserves_real_gram_and_error_bound():
    rng = np.random.default_rng(31)
    training = rng.uniform(-1.0, 1.0, size=(300, 2))

    def features(points):
        x, y = np.asarray(points).T
        output = np.empty((len(points), 2, 2), dtype=complex)
        output[:, 0, 0] = 1.0 + 0.2 * x + 0.1j * y
        output[:, 0, 1] = 0.1 * y - 0.2j * x
        output[:, 1, 0] = -0.15 * x + 0.25j * y
        output[:, 1, 1] = 0.8 - 0.1 * y + 0.05j * x
        return output

    model = FunctionalTT(
        degrees=1,
        rank=4,
        bounds=((-1.0, 1.0),) * 2,
        regularization=1.0e-12,
        sweeps=12,
        rtol=1.0e-10,
        hermitian=False,
        random_state=7,
    ).fit(training, features(training))
    real_model = model.realify_features()
    left = rng.uniform(-1.0, 1.0, size=(24, 2))
    right = rng.uniform(-1.0, 1.0, size=(24, 2))
    complex_left = model.predict(left)
    complex_right = model.predict(right)
    real_left = real_model.predict(left)
    real_right = real_model.predict(right)
    complex_gram = np.einsum(
        "nra,nrb->nab", complex_left.conj(), complex_right
    )
    real_gram = np.einsum("nra,nrb->nab", real_left, real_right)

    assert not np.iscomplexobj(real_model.predict(left))
    assert real_model.output_shape_ == (4, 2)
    np.testing.assert_allclose(real_gram, complex_gram.real, atol=2.0e-12)
    real_target = rng.standard_normal(complex_gram.shape)
    complex_error = np.linalg.norm(complex_gram - real_target, axis=(-2, -1))
    real_error = np.linalg.norm(real_gram - real_target, axis=(-2, -1))
    assert np.all(real_error <= complex_error + 2.0e-12)


def test_functional_tt_matrix_grid_has_terminal_output_site():
    rng = np.random.default_rng(23)
    training = rng.uniform(-1.0, 1.0, size=(300, 2))

    def matrices(points):
        x, y = np.asarray(points).T
        output = np.empty((len(points), 2, 2))
        output[:, 0, 0] = 1.0 + x
        output[:, 0, 1] = x * y
        output[:, 1, 0] = x * y
        output[:, 1, 1] = 2.0 - y
        return output

    model = FunctionalTT(
        degrees=1,
        rank=4,
        bounds=((-1.0, 1.0),) * 2,
        normalization="elementwise",
        regularization=1.0e-12,
        sweeps=12,
        random_state=2,
    ).fit(training, matrices(training))
    grids = (np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 5))
    cores = model.tensor_cores(grids)
    dense = tt_to_tensor(cores).reshape(4, 5, 2, 2)
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack(mesh, axis=-1)

    assert len(cores) == 3
    assert model.normalization == "elementwise"
    assert model.hermitian_
    assert cores[-1].shape[1] == 4
    assert cores[-1].shape[0] <= model.ranks_[-1] + 1
    np.testing.assert_allclose(dense, model.predict(points), atol=1.0e-12)
    np.testing.assert_allclose(dense, dense.conj().swapaxes(-1, -2), atol=1.0e-14)

    refined = (
        np.linspace(-1.0, 1.0, 7),
        np.linspace(-1.0, 1.0, 6),
    )
    refined_mesh = np.meshgrid(*refined, indexing="ij")
    refined_points = np.stack(refined_mesh, axis=-1)
    mpo = model.mpo(refined)
    expected = np.zeros((7, 6, 2, 7, 6, 2), dtype=complex)
    values = model.predict(refined_points)
    for index in np.ndindex(7, 6):
        expected[*index, :, *index, :] = values[index]
    np.testing.assert_allclose(
        mpo.to_dense(),
        expected.reshape(7 * 6 * 2, 7 * 6 * 2),
        atol=1.0e-12,
    )


def test_functional_tt_fits_hermitian_product_grid_by_tt_svd():
    grids = (np.linspace(-1.0, 1.0, 4), np.linspace(-1.0, 1.0, 5))
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack(mesh, axis=-1)

    def matrices(samples):
        x, y = np.asarray(samples).reshape(-1, 2).T
        coupling = x * y + 0.2j * (x - y)
        output = np.empty((len(x), 2, 2), dtype=complex)
        output[:, 0, 0] = 1.0 + x
        output[:, 0, 1] = coupling
        output[:, 1, 0] = coupling.conj()
        output[:, 1, 1] = 2.0 - y
        return output.reshape(*np.asarray(samples).shape[:-1], 2, 2)

    model = FunctionalTT(
        degrees=(1, 1),
        rank=(2, 4),
        bounds=((-1.0, 1.0),) * 2,
        hermitian=True,
    ).fit_grid(grids, matrices(points))
    rng = np.random.default_rng(31)
    probes = rng.uniform(-1.0, 1.0, size=(80, 2))

    assert model.n_sweeps == 0
    assert np.isrealobj(model.output_core)
    np.testing.assert_allclose(model.predict(probes), matrices(probes), atol=1.0e-12)
    np.testing.assert_allclose(model.error, 0.0, atol=1.0e-12)


def test_functional_tt_does_not_constrain_rectangular_matrix_fields():
    rng = np.random.default_rng(29)
    training = rng.uniform(-1.0, 1.0, size=(300, 2))

    def features(points):
        x, y = np.asarray(points).T
        output = np.empty((len(points), 3, 2), dtype=complex)
        output[:, 0, 0] = 1.0 + x
        output[:, 0, 1] = 0.2j * y
        output[:, 1, 0] = x * y
        output[:, 1, 1] = 0.5 - y
        output[:, 2, 0] = x - 0.3j * y
        output[:, 2, 1] = 1.0
        return output

    model = FunctionalTT(
        degrees=1,
        rank=4,
        bounds=((-1.0, 1.0),) * 2,
        regularization=1.0e-12,
        random_state=4,
    ).fit(training, features(training))
    probes = rng.uniform(-1.0, 1.0, size=(40, 2))

    assert not model.hermitian_
    assert model.predict(probes).shape == (40, 3, 2)
    np.testing.assert_allclose(model.predict(probes), features(probes), atol=1.0e-11)


def test_functional_tt_rejects_nonhermitian_data_when_required():
    coordinates = np.array(((-1.0,), (0.0,), (1.0,)))
    values = np.zeros((3, 2, 2), dtype=complex)
    values[:, 0, 1] = 1.0j

    with pytest.raises(ValueError, match="are not Hermitian"):
        FunctionalTT(
            degrees=1,
            rank=2,
            bounds=((-1.0, 1.0),),
            hermitian=True,
        ).fit(coordinates, values)
