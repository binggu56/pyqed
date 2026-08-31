import numpy as np
import pytest

from pyqed.ldr import AbInitioFit
from pyqed.ldr.oracle import (
    FeatureOracle,
    Frames,
    ProcrustesOracle,
    synchronize_features,
)
from pyqed.ldr.ttfit import (
    FiberSampler,
    KineticSampler,
    FeatureSampler,
    HermitianSampler,
    LPAFeatureOracle,
    LinkSampler,
    LinkPath,
    HamiltonianSampler,
    assemble,
    adaptive_feature_points,
    build_ey,
    build_mpo,
    corewise_link_mpo_kinetic,
    coordinate_fiber_points,
    corewise_link_mpo_components,
    endpoint_feature_mpo_kinetic,
    fiber_kernel,
    fit_mpo,
    fit_overlap,
    fit_cross,
    fit_cur,
    fit_aligned,
    fit_adaptive_sync,
    fit_block_cross,
    fit_ey,
    fit_features,
    fit_hamiltonian,
    fit_kinetic,
    fit_links,
    fit_sparse,
    fit_svd,
    fit_sync,
    grid_links,
    group_kinetic_terms,
    interpolate,
    interpolate_fiber,
    interpolation_matrix,
    kernel_fiber,
    link_mpo_kinetic,
    sample_graph,
)
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.functional import FunctionalTT
from pyqed.mps.mpo import sop_to_mpo
from pyqed.mps.mps import MPS, MPO
from pyqed.mps.tdvp import (
    TDVPEngine,
    one_site_tdvp_step,
    one_site_tdvp_sum_step,
    two_site_tdvp_step,
    two_site_tdvp_sum_step,
)
from pyqed.namd.ttldr import TTLDR


class ArrayOracle:
    def __init__(self, local, overlap):
        self.local = np.asarray(local)
        self.overlap = np.asarray(overlap)
        self.shape = self.local.shape[:-2]

    def hamiltonian_many(self, indices):
        return np.asarray([self.local[index] for index in indices])

    def overlap_many(self, pairs):
        return np.asarray(
            [
                self.overlap[
                    np.ravel_multi_index(left, self.shape),
                    :,
                    np.ravel_multi_index(right, self.shape),
                    :,
                ]
                for left, right in pairs
            ]
        )


def test_sampled_graph_features_use_one_global_synchronization():
    shape = (5, 5)
    frames = np.empty((*shape, 2, 2))
    for index in np.ndindex(shape):
        angle = 0.09 * (index[0] - 2) - 0.06 * (index[1] - 2)
        frames[index] = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    flat = frames.reshape(-1, 2, 2)
    overlap = np.einsum("pai,qaj->piqj", flat, flat, optimize=True)
    local = np.zeros((*shape, 2, 2))
    oracle = ArrayOracle(local, overlap)
    points = tuple((i, j) for i in (0, 2, 4) for j in (0, 2, 4))
    pairs = sample_graph(points, shape, neighbors=2)

    features, info = synchronize_features(
        oracle,
        points,
        pairs,
        2,
        anchor=(2, 2),
        penalty=50.0,
        maxiter=500,
        gtol=1.0e-10,
        seed=3,
    )

    point_ids = {point: index for index, point in enumerate(points)}
    reconstructed = np.asarray(
        [
            features[point_ids[left]].conj().T @ features[point_ids[right]]
            for left, right in pairs
        ]
    )
    np.testing.assert_allclose(reconstructed, oracle.overlap_many(pairs), atol=2.0e-6)
    gram = features.conj().swapaxes(-1, -2) @ features
    np.testing.assert_allclose(
        gram, np.broadcast_to(np.eye(2), gram.shape), atol=2.0e-14
    )
    assert info["isometry"] == "exact-polar-retraction"
    assert info["maximum_orthogonality_defect"] < 2.0e-14
    assert not np.iscomplexobj(features)
    assert info["real_valued"]
    assert info["points"] == 9
    assert info["pairs"] < 36


def test_nystrom_feature_strategy_uses_distributed_procrustes_landmarks():
    shape = (3, 3)
    points = tuple(np.ndindex(shape))
    rng = np.random.default_rng(81)
    frames = []
    for point in points:
        base = rng.normal(size=(6, 2))
        base += 0.2 * point[0] - 0.1 * point[1]
        frame, _ = np.linalg.qr(base)
        frames.append(frame)
    overlap = np.empty((9, 2, 9, 2))
    for left, first in enumerate(frames):
        for right, second in enumerate(frames):
            overlap[left, :, right, :] = first.T @ second
    oracle = ArrayOracle(np.zeros((*shape, 2, 2)), overlap)
    pairs = sample_graph(points, shape, neighbors=2)
    grids = (np.linspace(-1.0, 1.0, 3),) * 2

    _energy, feature, info = fit_sync(
        oracle,
        grids,
        2,
        points,
        pairs=pairs,
        anchor=(1, 1),
        max_rank=8,
        feature_rank=6,
        degrees=2,
        sweeps=4,
        rtol=1.0e-10,
        variational_maxiter=0,
        feature_strategy="nystrom",
    )

    mesh = np.meshgrid(*grids, indexing="ij")
    coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
    values = np.asarray(feature.predict(coordinates))
    point_ids = {point: index for index, point in enumerate(points)}
    predicted = np.asarray(
        [
            values[point_ids[left]].T @ values[point_ids[right]]
            for left, right in pairs
        ]
    )
    relative = np.linalg.norm(predicted - oracle.overlap_many(pairs))
    relative /= np.linalg.norm(oracle.overlap_many(pairs))
    assert relative < 1.0e-8
    assert info["feature_strategy"] == "nystrom"
    assert info["synchronization"]["backend"] == (
        "procrustes-nystrom-feature-synchronization"
    )
    assert info["synchronization"]["landmark_count"] == len(points)
    assert info["variational"]["backend"] == "procrustes-nystrom-links"


def test_ab_initio_sync_fits_one_feature_field_from_selected_points():
    built = []

    def builder(index):
        built.append(tuple(index))
        x, y = index
        angle = 0.08 * (x - 2) - 0.05 * (y - 2)
        frame = np.asarray(
            [[np.cos(angle), 0.0], [0.0, 1.0], [np.sin(angle), 0.0]]
        )
        return frame, np.asarray([0.02 * x + 0.01 * y, 0.4 + 0.01 * y])

    grids = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 5))
    points = tuple((i, j) for i in (0, 2, 4) for j in (0, 2, 4))
    with AbInitioFit(
        grids,
        2,
        builder,
        anchor=(2, 2),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    ) as fit:
        fit.run(
            representation="sync",
            points=points,
            neighbors=2,
            rank=3,
            degrees=2,
            fit_sweeps=2,
            feature_rank=3,
            feature_penalty=50.0,
            feature_maxiter=500,
            validation=2,
            seed=4,
        )

        assert fit.success
        assert fit.energy.output_shape_ == (2, 2)
        assert fit.feature.output_shape_ == (3, 2)
        assert fit.info["unique_geometries"] == 9
        assert fit.info["variational"]["rms_relative_link_error"] < 0.1
        assert fit.info["synchronization"]["real_valued"]
        assert not np.iscomplexobj(fit.feature.predict([[0.1, 0.1]]))
        mesh = np.meshgrid(*grids, indexing="ij")
        coordinates = np.stack([axis.reshape(-1) for axis in mesh], axis=1)
        predicted = fit.feature.predict(coordinates).reshape(5, 5, 3, 2)
        pairs = sample_graph(points, (5, 5), neighbors=2)
        predicted_links = np.asarray(
            [predicted[left].T @ predicted[right] for left, right in pairs]
        )
        target_links = fit.oracle.overlap_many(pairs)
        final_error = np.linalg.norm(predicted_links - target_links) / np.linalg.norm(
            target_links
        )
        assert final_error <= fit.info["variational"][
            "initial_relative_link_error"
        ] + 1.0e-10
        assert len(set(built)) == 9


def test_feature_synchronization_retains_genuinely_complex_overlaps():
    shape = (3,)
    phases = np.exp(1j * np.asarray([0.0, 0.3, 0.7]))
    overlap = np.empty((3, 1, 3, 1), dtype=complex)
    for left in range(3):
        for right in range(3):
            overlap[left, 0, right, 0] = phases[left].conj() * phases[right]
    oracle = ArrayOracle(np.zeros((3, 1, 1)), overlap)
    points = ((0,), (1,), (2,))
    pairs = (((0,), (1,)), ((1,), (2,)))

    features, info = synchronize_features(
        oracle, points, pairs, 1, anchor=(0,), maxiter=300, gtol=1.0e-10
    )

    assert np.iscomplexobj(features)
    assert not info["real_valued"]
    np.testing.assert_allclose(
        features.conj().swapaxes(-1, -2) @ features,
        np.ones((3, 1, 1)),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        features[0].conj().T @ features[1], overlap[0, :, 1, :], atol=2.0e-6
    )


def test_adaptive_feature_sampling_uses_model_defect_and_bounded_pool():
    class Feature:
        output_shape_ = (2, 1)

        @staticmethod
        def predict(coordinates):
            coordinates = np.asarray(coordinates)
            values = np.zeros((len(coordinates), 2, 1))
            values[:, 0, 0] = 1.0 + 2.0 * coordinates[:, 0] ** 2
            return values

    grids = (np.linspace(-1.0, 1.0, 101), np.linspace(-1.0, 1.0, 101))
    selected, info = adaptive_feature_points(
        Feature(),
        grids,
        ((50, 50),),
        4,
        candidate_pool=37,
        seed=8,
    )

    assert len(selected) == 4
    assert len(set(selected)) == 4
    assert info["candidate_pool"] == 37
    assert all(point != (50, 50) for point in selected)
    assert max(abs(point[0] - 50) for point in selected) >= 40

    weighted, weighted_info = adaptive_feature_points(
        Feature(),
        grids,
        ((50, 50),),
        4,
        candidate_pool=37,
        importance=np.exp(
            -0.02
            * (
                (np.arange(101)[:, None] - 50) ** 2
                + (np.arange(101)[None, :] - 50) ** 2
            )
        ),
        importance_floor=0.0,
        seed=8,
    )
    assert weighted_info["importance_weighted"]
    assert max(abs(point[0] - 50) for point in weighted) < 40

    called = []

    def local_importance(coordinates):
        called.append(len(coordinates))
        return np.exp(-np.sum(coordinates**2, axis=1))

    callable_weighted, callable_info = adaptive_feature_points(
        Feature(), grids, ((50, 50),), 4,
        candidate_pool=37, importance=local_importance, seed=8,
    )
    assert len(callable_weighted) == 4
    assert called == [37]
    assert callable_info["importance_weighted"]


def test_coordinate_fiber_seed_scales_linearly_with_dimension():
    points = coordinate_fiber_points((9, 9, 9, 9), points_per_axis=5)
    assert len(points) == 1 + 4 * (5 - 1)
    assert points[0] == (4, 4, 4, 4)
    assert all(sum(a != b for a, b in zip(point, points[0])) <= 1 for point in points)


def test_adaptive_sync_adds_only_selected_geometries():
    shape = (5, 5)
    frames = np.empty((*shape, 2, 2))
    for index in np.ndindex(shape):
        angle = 0.08 * index[0] - 0.05 * index[1]
        frames[index] = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
    flat = frames.reshape(-1, 2, 2)
    overlap = np.einsum("pai,qaj->piqj", flat, flat, optimize=True)
    local = np.zeros((*shape, 2, 2))
    oracle = ArrayOracle(local, overlap)
    initial = tuple((i, j) for i in (0, 2, 4) for j in (0, 2, 4))

    energy, feature, info = fit_adaptive_sync(
        oracle,
        (np.linspace(-1.0, 1.0, 5),) * 2,
        2,
        initial,
        target_points=13,
        batch_size=2,
        candidate_pool=7,
        anchor=(2, 2),
        max_rank=3,
        feature_rank=2,
        neighbors=2,
        degrees=2,
        sweeps=2,
        feature_penalty=50.0,
        feature_maxiter=200,
        variational_maxiter=0,
        seed=3,
    )

    assert energy.output_shape_ == (2, 2)
    assert feature.output_shape_ == (2, 2)
    assert info["unique_geometries"] == 13
    assert info["initial_geometries"] == 9
    assert info["adaptive_rounds"] == 2
    assert len(info["points"]) == 13
    assert not info["history"][0]["warm_started"]
    assert info["history"][1]["warm_started"]


def test_adaptive_sync_stops_after_converged_batch_is_refitted():
    shape = (5, 5)
    frames = np.broadcast_to(np.eye(2), (*shape, 2, 2)).copy()
    flat = frames.reshape(-1, 2, 2)
    overlap = np.einsum("pai,qaj->piqj", flat, flat, optimize=True)
    oracle = ArrayOracle(np.zeros((*shape, 2, 2)), overlap)
    initial = tuple((i, j) for i in (0, 2, 4) for j in (0, 2, 4))

    _energy, _feature, info = fit_adaptive_sync(
        oracle,
        (np.linspace(-1.0, 1.0, 5),) * 2,
        2,
        initial,
        target_points=25,
        batch_size=2,
        candidate_pool=7,
        energy_atol=1.0e-12,
        link_rtol=1.0e6,
        patience=1,
        minimum_rounds=1,
        anchor=(2, 2),
        max_rank=3,
        feature_rank=2,
        neighbors=2,
        degrees=2,
        sweeps=2,
        feature_penalty=50.0,
        feature_maxiter=200,
        variational_maxiter=0,
        seed=3,
    )

    assert info["converged"]
    assert info["stop_reason"] == "converged"
    assert info["unique_geometries"] == 11
    assert info["adaptive_rounds"] == 1
    assert info["history"][0]["validation"]["passed"]
    assert info["history"][1]["convergence_refit"]
    assert set(info["history"][0]["validation"]["points"]).issubset(
        info["points"]
    )


def parallel_record(index):
    return tuple(index), int(np.ravel_multi_index(index, (2, 3)))


def test_ttldr_dresses_direct_nuclear_mpo_with_scalar_overlap():
    class Solver:
        nx = (2,)
        nstates = 1
        apes = np.asarray([[0.2], [0.5]])
        overlap_matrix = None
        overlap_links = {(0, (0,)): np.asarray([[0.8]])}
        overlap_path_average = False

    kinetic = np.asarray([[0.7, -0.3], [-0.3, 0.9]])
    nuclear_keo = MPO([kinetic[None, None]])
    pes_mpo = MPO([np.diag(Solver.apes[:, 0])[None, None]])
    driver = TTLDR(
        Solver(),
        nuclear_keo=nuclear_keo,
        pes_mpo=pes_mpo,
        gauge_sync=False,
        overlap_method="dense",
        overlap_rank=4,
        operator_rank=None,
    )

    expected = kinetic * np.asarray([[1.0, 0.8], [0.8, 1.0]])
    expected += np.diag(Solver.apes[:, 0])
    np.testing.assert_allclose(driver.hamiltonian.to_dense(), expected)


def test_ttldr_consumes_aligned_energy_and_links_without_solver():
    class Solver:
        nx = (2,)
        nstates = 1

        @staticmethod
        def buildK_product_terms(symmetrize=True):
            assert symmetrize
            kinetic = np.asarray([[0.7, -0.3], [-0.3, 0.9]])
            return [(1.0, (kinetic,))]

    class EnergyFit:
        output_shape_ = (1, 1)

        @staticmethod
        def mpo(grids):
            values = np.asarray([0.2, 0.5])
            nuclear = np.diag(values).reshape(1, 1, 2, 2)
            electronic = np.ones((1, 1, 1, 1))
            return MPO([nuclear, electronic])

    class LinkFit:
        output_shape_ = (1, 1)

        @staticmethod
        def tensor_cores(grids):
            assert tuple(len(grid) for grid in grids) == (1,)
            return [np.ones((1, 1, 1)), np.full((1, 1, 1), 0.8)]

    driver = TTLDR(
        energy=EnergyFit(),
        links=(LinkFit(),),
        grids=(np.asarray([-1.0, 1.0]),),
        keo=Solver.buildK_product_terms(),
        overlap_rank=4,
        overlap_sweeps=4,
        overlap_validation=16,
        operator_rank=None,
    )

    kinetic = np.asarray([[0.7, -0.3], [-0.3, 0.9]])
    expected = kinetic * np.asarray([[1.0, 0.8], [0.8, 1.0]])
    expected += np.diag([0.2, 0.5])
    assert len(driver.components) == 2
    assert driver._hamiltonian is None
    np.testing.assert_allclose(driver.hamiltonian.to_dense(), expected, atol=1.0e-12)

    state = driver.state(np.asarray([[1.0], [0.0]]))
    driver.run(
        state,
        dt=0.01,
        steps=1,
        max_bond=2,
        progress=False,
        workers=2,
    )
    assert driver.final_state is not None
    np.testing.assert_allclose(driver.norms, 1.0, atol=1.0e-12)


def test_ttldr_split_local_cap_has_exact_channel_resolved_norm_loss():
    dimensions = (3, 2)
    zero = MPO(
        [
            np.zeros((1, 1, dimensions[0], dimensions[0]), dtype=complex),
            np.eye(dimensions[1], dtype=complex)[None, None],
        ]
    )
    driver = object.__new__(TTLDR)
    driver.dims = dimensions
    driver.nstates = dimensions[-1]
    driver.components = (zero,)
    driver.is_hermitian = True
    driver.fitted_fields = False

    values = np.asarray(
        [[0.5, 0.0], [0.0, np.sqrt(0.35)], [np.sqrt(0.20), np.sqrt(0.20)]],
        dtype=complex,
    )
    values /= np.linalg.norm(values)
    state = MPS(decompose(values, rank=2))
    projectors = []
    for channel in range(dimensions[-1]):
        electronic = np.zeros((2, 2))
        electronic[channel, channel] = 1.0
        projectors.append(
            MPO(
                [
                    np.eye(dimensions[0])[None, None],
                    electronic[None, None],
                ]
            )
        )

    absorber = np.asarray([0.0, 0.15, 0.30])
    dt = 0.2
    steps = 4
    driver.run(
        state,
        dt=dt,
        steps=steps,
        interval=1,
        integrator="tdvp",
        normalize=False,
        progress=False,
        e_ops=projectors,
        absorber=absorber,
    )

    probabilities = np.abs(values) ** 2
    times = np.arange(steps + 1) * dt
    survival = np.asarray(
        [np.sum(probabilities * np.exp(-2.0 * time * absorber[:, None])) for time in times]
    )
    expected_yields = np.asarray(
        [
            np.sum(
                probabilities
                * (1.0 - np.exp(-2.0 * time * absorber[:, None])),
                axis=0,
            )
            for time in times
        ]
    )
    expected_cap = np.asarray(
        [
            np.sum(
                probabilities
                * np.exp(-2.0 * time * absorber[:, None])
                * absorber[:, None],
                axis=0,
            )
            for time in times
        ]
    )

    np.testing.assert_allclose(driver.norms, survival, atol=2.0e-12)
    np.testing.assert_allclose(driver.absorber_yields, expected_yields, atol=2.0e-12)
    np.testing.assert_allclose(
        driver.absorber_expectations, expected_cap, atol=2.0e-12
    )
    np.testing.assert_allclose(driver.absorption_closure, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(
        np.sum(driver.populations, axis=1), driver.norms, atol=2.0e-12
    )

    with pytest.raises(ValueError, match="must not normalize"):
        driver.run(
            state,
            dt=dt,
            steps=1,
            normalize=True,
            progress=False,
            e_ops=projectors,
            absorber=absorber,
        )


def test_ttldr_split_cap_accepts_all_nuclear_sites():
    dimensions = (2, 3, 2)
    zero = MPO(
        [
            np.zeros((1, 1, dimensions[0], dimensions[0]), dtype=complex),
            np.eye(dimensions[1], dtype=complex)[None, None],
            np.eye(dimensions[2], dtype=complex)[None, None],
        ]
    )
    driver = object.__new__(TTLDR)
    driver.dims = dimensions
    driver.nstates = dimensions[-1]
    driver.components = (zero,)
    driver.is_hermitian = True
    driver.fitted_fields = False

    values = np.arange(1, np.prod(dimensions) + 1, dtype=float).reshape(dimensions)
    values = values.astype(complex) / np.linalg.norm(values)
    state = MPS(decompose(values, rank=6))
    projectors = []
    for channel in range(dimensions[-1]):
        electronic = np.zeros((2, 2))
        electronic[channel, channel] = 1.0
        projectors.append(
            MPO(
                [
                    np.eye(dimensions[0])[None, None],
                    np.eye(dimensions[1])[None, None],
                    electronic[None, None],
                ]
            )
        )

    caps = {0: np.asarray((0.0, 0.20)), 1: np.asarray((0.10, 0.0, 0.30))}
    dt = 0.15
    steps = 3
    driver.run(
        state,
        dt=dt,
        steps=steps,
        interval=1,
        integrator="tdvp",
        normalize=False,
        progress=False,
        e_ops=projectors,
        absorber=caps,
    )

    probability = np.abs(values) ** 2
    total_cap = caps[0][:, None, None] + caps[1][None, :, None]
    times = np.arange(steps + 1) * dt
    expected_norms = np.asarray(
        [np.sum(probability * np.exp(-2.0 * time * total_cap)) for time in times]
    )
    expected_yields = np.asarray(
        [
            np.sum(
                probability * (1.0 - np.exp(-2.0 * time * total_cap)),
                axis=(0, 1),
            )
            for time in times
        ]
    )
    expected_cap = np.asarray(
        [
            np.sum(
                probability
                * np.exp(-2.0 * time * total_cap)
                * total_cap,
                axis=(0, 1),
            )
            for time in times
        ]
    )

    np.testing.assert_allclose(driver.norms, expected_norms, atol=2.0e-12)
    np.testing.assert_allclose(driver.absorber_yields, expected_yields, atol=2.0e-12)
    np.testing.assert_allclose(
        driver.absorber_expectations, expected_cap, atol=2.0e-12
    )
    np.testing.assert_allclose(driver.absorption_closure, 0.0, atol=2.0e-12)


def test_ttldr_cap_does_not_count_tdvp2_truncation_as_absorption():
    dimensions = (2, 2)
    pauli_x = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    coupling = MPO([pauli_x[None, None], pauli_x[None, None]])
    driver = object.__new__(TTLDR)
    driver.dims = dimensions
    driver.nstates = dimensions[-1]
    driver.components = (coupling,)
    driver.is_hermitian = True
    driver.fitted_fields = False

    zero = np.asarray([1.0, 0.0], dtype=complex)
    state = MPS([zero.reshape(1, 2, 1), zero.reshape(1, 2, 1)])
    projectors = []
    for channel in range(dimensions[-1]):
        electronic = np.zeros((2, 2))
        electronic[channel, channel] = 1.0
        projectors.append(
            MPO([np.eye(dimensions[0])[None, None], electronic[None, None]])
        )

    driver.run(
        state,
        dt=0.3,
        steps=1,
        interval=1,
        max_bond=1,
        integrator="tdvp2",
        cutoff=0.0,
        normalize=False,
        progress=False,
        e_ops=projectors,
        absorber=np.zeros(dimensions[0]),
    )

    assert driver.tdvp_truncation_errors[-1] > 0.0
    assert driver.tdvp_norm_defects[-1] > 0.0
    np.testing.assert_allclose(driver.norms, 1.0, atol=2.0e-12)
    np.testing.assert_allclose(driver.absorber_yields, 0.0, atol=2.0e-12)
    np.testing.assert_allclose(driver.absorption_closure, 0.0, atol=2.0e-12)


def test_ttldr_contracts_working_frame_populations_in_one_pass(monkeypatch):
    dimensions = (3, 2)
    zero = MPO(
        [
            np.zeros((1, 1, dimensions[0], dimensions[0]), dtype=complex),
            np.eye(dimensions[1], dtype=complex)[None, None],
        ]
    )
    driver = object.__new__(TTLDR)
    driver.dims = dimensions
    driver.nx = dimensions[:-1]
    driver.nstates = dimensions[-1]
    driver.components = (zero,)
    driver.is_hermitian = True
    driver.fitted_fields = True
    driver.gauge_sync = True
    driver._working_projectors = None

    values = np.asarray(
        [[0.3, 0.2j], [0.4, -0.1], [0.5j, 0.6]], dtype=complex
    )
    values /= np.linalg.norm(values)
    state = MPS(decompose(values, rank=2))
    projectors = driver.projectors()
    expected = np.sum(np.abs(values) ** 2, axis=0)
    np.testing.assert_allclose(driver.working_frame_populations(state), expected)
    assert all(max(projector.bond_orders(), default=1) == 1 for projector in projectors)

    def reject_separate_expectations(self, operator):
        raise AssertionError("working populations should share one contraction")

    monkeypatch.setattr(MPS, "expectation", reject_separate_expectations)
    driver.run(
        state,
        dt=0.01,
        steps=1,
        integrator="tdvp",
        progress=False,
        e_ops=projectors,
        absorber=np.zeros(dimensions[0]),
    )
    np.testing.assert_allclose(driver.populations[0], expected)
    np.testing.assert_allclose(driver.populations[1], expected, atol=1.0e-12)


def test_ttldr_restores_prebuilt_fitted_components_without_rebuilding():
    dimensions = (3, 2)
    zero = MPO(
        [
            np.zeros((1, 1, dimensions[0], dimensions[0]), dtype=complex),
            np.eye(dimensions[1], dtype=complex)[None, None],
        ]
    )
    driver = TTLDR.from_components(
        (zero,),
        grids=(np.linspace(-1.0, 1.0, dimensions[0]),),
        overlap_info={"backend": "cached", "fields": []},
    )
    values = np.arange(1, 7, dtype=float).reshape(dimensions).astype(complex)
    values /= np.linalg.norm(values)
    state = driver.state(values, max_rank=2)

    assert driver.components[0] is zero
    assert driver.dims == dimensions
    assert driver.fitted_fields
    assert not driver.gauge_sync
    assert driver.overlap_info["backend"] == "cached"
    np.testing.assert_allclose(driver.dense(state), values, atol=1.0e-14)

    driver.run(
        state,
        dt=0.01,
        steps=1,
        integrator="tdvp",
        progress=False,
        e_ops=driver.projectors(),
        absorber=np.zeros(dimensions[0]),
    )
    np.testing.assert_allclose(driver.dense(driver.final_state), values, atol=1.0e-12)


def test_fitted_ttldr_builds_matrix_free_adiabatic_projector_and_state():
    class EnergyFit:
        output_shape_ = (2, 2)

        @staticmethod
        def predict(coordinates):
            x = np.asarray(coordinates)[:, 0]
            blocks = np.zeros((len(x), 2, 2), dtype=complex)
            blocks[:, 0, 0] = -0.4
            blocks[:, 1, 1] = 0.4
            blocks[:, 0, 1] = blocks[:, 1, 0] = 0.15 * x
            return blocks

        @staticmethod
        def mpo(grids):
            grid = np.asarray(grids[0])
            values = EnergyFit.predict(grid[:, None])
            factors = np.zeros((1, 1, len(grid), len(grid)), dtype=complex)
            # This method is not used by the projector construction.
            factors[0, 0] = np.eye(len(grid))
            electronic = np.eye(2)[None, None]
            return MPO([factors, electronic])

    class FeatureFit:
        output_shape_ = (2, 2)

        @staticmethod
        def tensor_cores(grids):
            size = len(grids[0])
            nuclear = np.ones((1, size, 1))
            output = np.eye(2).reshape(1, 4, 1)
            return [nuclear, output]

    grid = np.linspace(-1.0, 1.0, 7)
    kinetic = np.zeros((7, 7))
    driver = TTLDR(
        energy=EnergyFit(),
        feature=FeatureFit(),
        grids=(grid,),
        keo=((1.0, (kinetic,)),),
        operator_rank=None,
    )
    projector, info = driver.adiabatic_projector(
        1, max_rank=4, sweeps=4, validation=16, seed=3
    )
    dense_projector, dense_info = driver.adiabatic_projector(
        1, method="dense", max_rank=None
    )
    packet = np.exp(-4.0 * grid**2)
    state, cached, _state_info = driver.matched_state(
        (packet,), 1, max_bond=4, projector_rank=4,
        projector_sweeps=4, projector_validation=16,
    )

    assert projector.L == 2
    assert dense_projector.L == 2
    assert dense_info["backend"] == "dense-local-projector-mpo"
    assert dense_info["validation_error"] < 1.0e-12
    assert cached.L == 2
    assert info["samples"] < 7 * 4 * 8
    np.testing.assert_allclose(state.norm_squared(), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(
        state.expectation(projector).real, 1.0, atol=3.0e-4
    )
    dense = np.asarray(
        tt_to_tensor([state._get_std_B(site) for site in range(state.L)])
    ).reshape(7, 2)
    np.testing.assert_allclose(
        np.linalg.norm(dense, axis=1) / np.linalg.norm(dense),
        np.abs(packet) / np.linalg.norm(packet),
        atol=3.0e-4,
    )


def test_overlap_fiber_round_trip_and_hamiltonian_assembly():
    shape = (2, 3)
    nstates = 2
    ngrid = int(np.prod(shape))
    rng = np.random.default_rng(4)
    kernel = rng.normal(size=(ngrid, nstates, ngrid, nstates))
    active = (0,)
    fiber = kernel_fiber(kernel, shape, active)
    restored = fiber_kernel(fiber, shape, nstates, active)

    for left in np.ndindex(shape):
        for right in np.ndindex(shape):
            i = np.ravel_multi_index(left, shape)
            j = np.ravel_multi_index(right, shape)
            if left[1] == right[1]:
                np.testing.assert_allclose(restored[i, :, j, :], kernel[i, :, j, :])
            else:
                np.testing.assert_allclose(restored[i, :, j, :], 0.0)

    off_diagonal = np.asarray([[0.0, -0.2], [-0.2, 0.0]])
    diagonal = np.diag([0.3, 0.5, 0.7])
    groups = group_kinetic_terms([(1.0, off_diagonal, diagonal)], shape)
    local = np.zeros((*shape, nstates, nstates))
    hamiltonian = assemble(groups, {active: fiber}, local, hermitize=False)
    expected = np.kron(off_diagonal, diagonal)[:, None, :, None] * restored
    np.testing.assert_allclose(
        hamiltonian,
        expected.reshape(ngrid * nstates, ngrid * nstates),
    )


def test_blockwise_tt_cores_build_exact_ldr_mpo():
    shape = (2, 3)
    nstates = 2
    ngrid = int(np.prod(shape))
    rng = np.random.default_rng(11)
    off_diagonal = np.asarray([[0.0, -0.2], [-0.2, 0.0]])
    diagonal = np.diag([0.3, 0.5, 0.7])
    terms = [(1.0, (off_diagonal, diagonal))]
    groups = group_kinetic_terms(terms, shape)
    active = (0,)
    overlap = rng.normal(size=(ngrid, nstates, ngrid, nstates))
    fiber = kernel_fiber(overlap, shape, active)
    fiber_blocks = fiber.reshape(*fiber.shape[:-1], nstates, nstates)
    local = rng.normal(size=(*shape, nstates, nstates))

    local_cores = {
        (alpha, beta): decompose(local[..., alpha, beta], rank=8)
        for alpha in range(nstates)
        for beta in range(nstates)
    }
    overlap_cores = {
        active: {
            (alpha, beta): decompose(fiber_blocks[..., alpha, beta], rank=8)
            for alpha in range(nstates)
            for beta in range(nstates)
        }
    }
    mpo = build_mpo(
        terms,
        local_cores,
        overlap_cores,
        shape,
        nstates,
        hermitize=False,
    )
    expected = assemble(groups, {active: fiber}, local, hermitize=False)
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=1.0e-12)
    components = build_mpo(
        terms,
        local_cores,
        overlap_cores,
        shape,
        nstates,
        hermitize=False,
        split=True,
    )
    np.testing.assert_allclose(
        sum(component.to_dense() for component in components),
        expected,
        atol=1.0e-12,
    )


def test_sum_tdvp_matches_direct_sum_mpo_step():
    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    first = sop_to_mpo((2, 2), [(0.3, (sx, sx))])
    second = sop_to_mpo((2, 2), [(0.2, (sz, np.eye(2)))])
    values = np.asarray([[1.0, 0.2j], [-0.1, 0.3]], dtype=complex)
    state = MPS(decompose(values / np.linalg.norm(values), rank=2))

    reference = two_site_tdvp_step(
        state,
        first + second,
        0.03,
        max_bond=2,
        canonicalize=True,
        normalize=True,
    )
    fitted = two_site_tdvp_sum_step(
        state,
        (first, second),
        0.03,
        max_bond=2,
        canonicalize=True,
        normalize=True,
    )
    engine = TDVPEngine((first, second), max_bond=2, workers=2)
    threaded, engine_info = engine.step(state, 0.03)
    engine.close()
    assert engine_info["operator_mode"] == "sum"
    assert engine_info["components"] == 2
    assert engine_info["workers"] == 2
    reference_values = tt_to_tensor(
        [reference._get_std_B(site) for site in range(reference.L)]
    )
    fitted_values = tt_to_tensor([fitted._get_std_B(site) for site in range(fitted.L)])
    threaded_values = tt_to_tensor(
        [threaded._get_std_B(site) for site in range(threaded.L)]
    )
    phase = np.vdot(reference_values, fitted_values)
    fitted_values *= np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(fitted_values, reference_values, atol=1.0e-11)
    phase = np.vdot(reference_values, threaded_values)
    threaded_values *= np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(threaded_values, reference_values, atol=1.0e-11)

    fixed_reference = one_site_tdvp_step(
        state,
        first + second,
        0.03,
        canonicalize=True,
        normalize=True,
    )
    fixed_sum = one_site_tdvp_sum_step(
        state,
        (first, second),
        0.03,
        canonicalize=True,
        normalize=True,
    )
    fixed_reference_values = tt_to_tensor(fixed_reference.factors)
    fixed_sum_values = tt_to_tensor(fixed_sum.factors)
    phase = np.vdot(fixed_reference_values, fixed_sum_values)
    fixed_sum_values *= np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(
        fixed_sum_values, fixed_reference_values, atol=1.0e-11
    )

    single_engine = TDVPEngine(first, max_bond=2)
    _, single_info = single_engine.step(state, 0.03)
    single_engine.close()
    assert single_info["operator_mode"] == "single"
    assert single_info["components"] == 1
    assert single_info["workers"] == 1


def test_compiled_sum_tdvp_sweep_reuses_valid_right_environments():
    from pyqed.mps import tdvp_cpp

    if not (
        tdvp_cpp.CPP_TDVP_AVAILABLE
        and tdvp_cpp.one_site_lanczos_sum_sweep is not None
    ):
        pytest.skip("compiled sum-TDVP sweep is unavailable")

    sx = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    sz = np.asarray([[1.0, 0.0], [0.0, -1.0]])
    operators = (
        sop_to_mpo((2, 2), [(0.3, (sx, sx))]),
        sop_to_mpo((2, 2), [(0.2, (sz, np.eye(2)))]),
    )
    values = np.asarray([[1.0, 0.2j], [-0.1, 0.3]], dtype=complex)
    state = MPS(decompose(values / np.linalg.norm(values), rank=2))
    state = state.right_canonicalize()
    engine = TDVPEngine(
        operators,
        integrator="tdvp",
        canonicalize_first=False,
        workers=2,
    )
    first, first_info = engine.step(state, 0.03, normalize=False)
    assert first_info["backend"] == "compiled-sum-tdvp-sweep"
    assert not first_info["right_environments_reused"]

    damped = first.copy()
    damped.factors[0] *= np.asarray([0.9, 1.0]).reshape(1, 2, 1)
    cached, cached_info = engine.step(damped, 0.03, normalize=False)
    assert cached_info["right_environments_reused"]
    reference = one_site_tdvp_sum_step(
        damped,
        operators,
        0.03,
        canonicalize=False,
        normalize=False,
    )
    np.testing.assert_allclose(
        tt_to_tensor(cached.factors),
        tt_to_tensor(reference.factors),
        atol=1.0e-11,
    )

    changed_right = cached.copy()
    changed_right.factors[1] *= 1.0001
    _, changed_info = engine.step(changed_right, 0.03, normalize=False)
    assert not changed_info["right_environments_reused"]
    engine.close()


def test_labelled_mpo_keo_components_use_their_active_link_paths():
    axes = (np.linspace(-0.4, 0.5, 3), np.linspace(-0.2, 0.3, 3))
    links = []
    for active in range(2):
        edge_axes = list(axes)
        edge_axes[active] = 0.5 * (
            edge_axes[active][:-1] + edge_axes[active][1:]
        )
        values = np.broadcast_to(
            np.eye(2),
            (*tuple(len(axis) for axis in edge_axes), 2, 2),
        ).copy()
        links.append(
            FunctionalTT(
                degrees=tuple(len(axis) - 1 for axis in edge_axes),
                rank=2,
                hermitian=False,
            ).fit_grid(tuple(edge_axes), values)
        )

    hopping = np.asarray(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]
    )
    cross = sop_to_mpo((3, 3), [(0.3, (hopping, hopping))])
    diagonal = sop_to_mpo((3, 3), [(0.2, (np.eye(3), np.eye(3)))])
    dressed, info = corewise_link_mpo_components(
        links,
        axes,
        (((0, 1), cross), ((), diagonal)),
        2,
        max_rank=4,
        operator_rank=None,
        split=True,
    )
    result = sum(component.to_dense() for component in dressed)
    expected = np.kron(cross.to_dense() + diagonal.to_dense(), np.eye(2))

    np.testing.assert_allclose(result, expected, atol=1.0e-11)
    assert info["backend"] == "corewise-directional-link-labelled-mpo"
    assert info["fields"][0]["active"] == (0, 1)

    from types import SimpleNamespace

    local = np.broadcast_to(
        np.diag((0.05, 0.10)),
        (3, 3, 2, 2),
    ).copy()
    energy = FunctionalTT(
        degrees=(2, 2), rank=4, hermitian=True
    ).fit_grid(axes, local)
    fit = SimpleNamespace(
        success=True,
        energy=energy,
        links=tuple(links),
        feature=None,
        grids=axes,
    )
    driver = TTLDR.from_fit(
        fit,
        keo=(((0, 1), cross), ((), diagonal)),
        overlap_rank=4,
        potential_rank=None,
        operator_rank=None,
    )
    expected += np.kron(np.eye(9), np.diag((0.05, 0.10)))
    np.testing.assert_allclose(driver.hamiltonian.to_dense(), expected, atol=1.0e-11)
    assert driver.overlap_info["backend"] == "corewise-directional-link-labelled-mpo"


def test_native_sum_tdvp_site_and_two_site_kernels(monkeypatch):
    from scipy.linalg import expm

    from pyqed.mps import tdvp as tdvp_module
    from pyqed.mps import tdvp_cpp

    if not (
        tdvp_cpp.CPP_TDVP_AVAILABLE
        and tdvp_cpp.CPP_TDVP_HAS_BLAS
        and tdvp_cpp.site_lanczos_sum is not None
        and tdvp_cpp.two_site_lanczos_sum is not None
    ):
        pytest.skip("native sum-TDVP kernels are unavailable")

    rng = np.random.default_rng(41)

    def hermitian(size):
        matrix = rng.normal(size=(size, size))
        matrix = matrix + 1j * rng.normal(size=(size, size))
        return matrix + matrix.conj().T

    site_shape = (2, 3, 2)
    site_theta = rng.normal(size=site_shape) + 1j * rng.normal(size=site_shape)
    site_environments = tuple(
        (
            hermitian(site_shape[0])[:, None, :],
            hermitian(site_shape[1])[None, None, :, :],
            hermitian(site_shape[2])[:, None, :],
        )
        for _ in range(2)
    )

    def apply_site(vector):
        local = vector.reshape(site_shape)
        return sum(
            tdvp_module._apply_site_heff(local, *environment)
            for environment in site_environments
        ).ravel()

    site_dim = int(np.prod(site_shape))
    site_matrix = np.column_stack(
        [apply_site(np.eye(site_dim, dtype=complex)[:, column]) for column in range(site_dim)]
    )
    site_reference = (
        expm(-1j * 0.02 * site_matrix) @ site_theta.ravel()
    ).reshape(site_shape)
    calls = {"site": 0, "two_site": 0}
    original_site = tdvp_cpp.site_lanczos_sum

    def site_wrapper(*args):
        calls["site"] += 1
        return original_site(*args)

    monkeypatch.setattr(tdvp_cpp, "site_lanczos_sum", site_wrapper)
    site_result = tdvp_module._evolve_site_sum(
        site_theta,
        site_environments,
        0.02,
        krylov_dim=20,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
    )
    np.testing.assert_allclose(site_result, site_reference, atol=1.0e-11)

    two_shape = (2, 2, 3, 2)
    two_theta = rng.normal(size=two_shape) + 1j * rng.normal(size=two_shape)
    two_environments = tuple(
        (
            hermitian(two_shape[0])[:, None, :],
            hermitian(two_shape[1])[None, None, :, :],
            hermitian(two_shape[2])[None, None, :, :],
            hermitian(two_shape[3])[:, None, :],
        )
        for _ in range(2)
    )

    def apply_two(vector):
        local = vector.reshape(two_shape)
        return sum(
            np.einsum(
                "amb,mnpq,nors,cod,bqsd->aprc",
                *environment,
                local,
                optimize=True,
            )
            for environment in two_environments
        ).ravel()

    two_dim = int(np.prod(two_shape))
    two_matrix = np.column_stack(
        [apply_two(np.eye(two_dim, dtype=complex)[:, column]) for column in range(two_dim)]
    )
    two_reference = (
        expm(-1j * 0.01 * two_matrix) @ two_theta.ravel()
    ).reshape(two_shape)
    original_two_site = tdvp_cpp.two_site_lanczos_sum

    def two_site_wrapper(*args):
        calls["two_site"] += 1
        return original_two_site(*args)

    monkeypatch.setattr(tdvp_cpp, "two_site_lanczos_sum", two_site_wrapper)
    two_result = tdvp_module._evolve_two_site_sum(
        two_theta,
        two_environments,
        0.01,
        krylov_dim=30,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
    )
    np.testing.assert_allclose(two_result, two_reference, atol=1.0e-11)
    assert calls == {"site": 1, "two_site": 1}


def test_native_sum_tdvp_adaptive_lanczos_reduces_matvecs():
    from pyqed.mps import tdvp_cpp

    if not (
        tdvp_cpp.CPP_TDVP_AVAILABLE
        and tdvp_cpp.CPP_TDVP_HAS_BLAS
        and tdvp_cpp.site_lanczos_sum is not None
        and tdvp_cpp.reset_kernel_stats is not None
    ):
        pytest.skip("native sum-TDVP kernel statistics are unavailable")

    rng = np.random.default_rng(53)
    shape = (4, 5, 4)
    theta = rng.normal(size=shape) + 1j * rng.normal(size=shape)
    diagonal = np.linspace(-0.2, 0.3, shape[1])
    environments = (
        np.eye(shape[0], dtype=complex)[:, None, :],
        np.diag(diagonal)[None, None, :, :],
        np.eye(shape[2], dtype=complex)[:, None, :],
    )
    arguments = (
        theta,
        [environments[0]],
        [environments[1]],
        [environments[2]],
        0.05,
        12,
    )

    tdvp_cpp.reset_kernel_stats()
    reference = tdvp_cpp.site_lanczos_sum(*arguments, 0.0, 20_000_000, 1)
    full_stats = dict(tdvp_cpp.kernel_stats())
    tdvp_cpp.reset_kernel_stats()
    adaptive = tdvp_cpp.site_lanczos_sum(*arguments, 1.0e-12, 20_000_000, 1)
    adaptive_stats = dict(tdvp_cpp.kernel_stats())
    tdvp_cpp.reset_kernel_stats()
    low_memory = tdvp_cpp.site_lanczos_sum(*arguments, 1.0e-12, 1, 2)
    low_memory_stats = dict(tdvp_cpp.kernel_stats())

    np.testing.assert_allclose(adaptive, reference, atol=1.0e-12, rtol=1.0e-12)
    np.testing.assert_allclose(low_memory, adaptive, atol=1.0e-12, rtol=1.0e-12)
    assert adaptive_stats["lanczos_matvecs"] < full_stats["lanczos_matvecs"]
    assert low_memory_stats["low_memory_site_sum_calls"] == 1


def test_fit_mpo_keeps_oracle_fields_in_tensor_train_form():
    shape = (3, 2)
    nstates = 2
    ngrid = int(np.prod(shape))
    local = np.zeros((*shape, nstates, nstates))
    overlap = np.zeros((ngrid, nstates, ngrid, nstates))
    for left in np.ndindex(shape):
        i = np.ravel_multi_index(left, shape)
        local[left] = np.diag([0.1 * left[0], 0.2 + 0.1 * left[1]])
        for right in np.ndindex(shape):
            j = np.ravel_multi_index(right, shape)
            overlap[i, :, j, :] = np.eye(nstates)
    oracle = ArrayOracle(local, overlap)
    hopping = np.asarray([[0.0, -0.1, 0.0], [-0.1, 0.0, -0.1], [0.0, -0.1, 0.0]])
    terms = [(1.0, (hopping, np.eye(2)))]
    mpo, info = fit_mpo(
        oracle,
        terms,
        shape,
        nstates,
        max_rank=2,
        operator_rank=None,
        sweeps=3,
        validation=24,
    )
    groups = group_kinetic_terms(terms, shape)
    fiber = kernel_fiber(overlap, shape, (0,))
    expected = assemble(groups, {(0,): fiber}, local)
    np.testing.assert_allclose(mpo.to_dense(), expected, atol=1.0e-10)
    assert info["backend"] == "tt-cross-mpo"
    assert info["unique_geometries"] <= ngrid


def test_feature_oracle_and_ey_cross_reconstruct_a_low_rank_gram_kernel():
    shape = (2, 2)
    nstates = 2
    points = list(np.ndindex(shape))
    rng = np.random.default_rng(18)
    frames = []
    for _point in points:
        matrix = rng.normal(size=(4, nstates)) + 1j * rng.normal(size=(4, nstates))
        frame, _ = np.linalg.qr(matrix)
        frames.append(frame)
    overlap = np.empty((4, nstates, 4, nstates), dtype=complex)
    for left, first in enumerate(frames):
        for right, second in enumerate(frames):
            overlap[left, :, right, :] = first.conj().T @ second
    local = np.empty((*shape, nstates, nstates), dtype=complex)
    for point in points:
        matrix = rng.normal(size=(nstates, nstates))
        local[point] = matrix + matrix.T
    oracle = ArrayOracle(local, overlap)
    feature = FeatureOracle(oracle, points)
    assert feature.rank == 4
    pairs = [(left, right) for left in points for right in points]
    np.testing.assert_allclose(
        feature.overlap_many(pairs),
        oracle.overlap_many(pairs),
        atol=1.0e-12,
    )

    hopping = np.asarray([[0.0, -0.15], [-0.15, 0.0]])
    terms = [(1.0, (hopping, np.eye(2)))]
    groups = group_kinetic_terms(terms, shape)
    exact_fiber = kernel_fiber(overlap, shape, (0,))
    expected = assemble(groups, {(0,): exact_fiber}, local)

    feature_values = feature.feature_many(points).reshape(*shape, -1)
    energy_cores = decompose(local.reshape(*shape, nstates**2), rank=16)
    feature_cores = decompose(feature_values, rank=16)
    direct = build_ey(
        terms,
        energy_cores,
        feature_cores,
        shape,
        nstates,
        feature.rank,
    )
    np.testing.assert_allclose(direct.to_dense(), expected, atol=1.0e-11)

    class FeatureModel:
        output_shape_ = (feature.rank, nstates)

        @staticmethod
        def tensor_cores(_grids):
            return feature_cores

    direct_components = build_ey(
        terms,
        energy_cores,
        feature_cores,
        shape,
        nstates,
        feature.rank,
        split=True,
    )
    endpoint, endpoint_info = endpoint_feature_mpo_kinetic(
        FeatureModel(),
        (np.arange(2.0), np.arange(2.0)),
        [((0,), sop_to_mpo(shape, terms))],
        nstates,
        labelled=True,
    )
    np.testing.assert_allclose(
        endpoint[0].to_dense(), direct_components[0].to_dense(), atol=1.0e-11
    )
    assert endpoint_info["backend"] == "endpoint-feature-gram-mpo"
    assert not endpoint_info["nearest_link_products"]

    crossed, info = fit_ey(
        oracle,
        terms,
        shape,
        nstates,
        points,
        max_rank=16,
        sweeps=5,
        validation=64,
    )
    np.testing.assert_allclose(crossed.to_dense(), expected, atol=1.0e-10)
    assert info["backend"] == "tt-cross-ey"
    assert info["feature_rank"] == 4


def test_lpa_feature_oracle_uses_only_neighbor_features_for_long_links():
    class Features:
        shape = (3,)
        nstates = 1

        def feature_many(self, indices):
            angles = np.asarray([0.0, 0.4, 1.1])
            values = []
            for index in indices:
                angle = angles[index[0]]
                values.append([[np.cos(angle)], [np.sin(angle)]])
            return np.asarray(values)

    oracle = LPAFeatureOracle(Features())
    result = oracle.overlap_many([((0,), (2,))])[0, 0, 0]
    expected = np.cos(0.4) * np.cos(0.7)
    np.testing.assert_allclose(result, expected)
    assert set(oracle.links) == {(0, (0,)), (0, (1,))}
    assert not np.isclose(result, np.cos(1.1))


def test_svd_and_cross_fit_low_rank_matrix_field():
    shape = (5, 4, 3)
    values = np.empty(shape)
    for index in np.ndindex(shape):
        values[index] = np.prod([1.0 + 0.2 * item for item in index])

    _cores, fitted, info = fit_svd(values, 1)
    np.testing.assert_allclose(fitted, values, atol=1.0e-12)
    assert info["ranks"] == (1, 1, 1, 1)

    _cores, fitted, info = fit_cross(
        shape,
        lambda index: values[index],
        max_rank=1,
        sweeps=2,
        validation=20,
    )
    np.testing.assert_allclose(fitted, values, atol=1.0e-12)
    assert info["samples"] < values.size


def test_anchor_interpolation_handles_local_and_two_point_fibers():
    coarse = np.asarray([-1.0, 0.0, 1.0])
    fine = np.linspace(-1.0, 1.0, 5)
    matrix = interpolation_matrix(coarse, fine)
    spline = interpolation_matrix(coarse, fine, degree=3)
    local = coarse[:, None] ** 2 + 2.0 * coarse[None, :]
    fitted_local = interpolate(local, (matrix, matrix))
    expected_local = fine[:, None] ** 2 + 2.0 * fine[None, :]
    np.testing.assert_allclose(fitted_local, expected_local, atol=1.0e-13)
    np.testing.assert_allclose(
        interpolate(local, (spline, spline)),
        expected_local,
        atol=1.0e-13,
    )

    bra, ket, spectator = np.meshgrid(coarse, coarse, coarse, indexing="ij")
    fiber = (bra * ket + spectator**2).reshape(9, 3)
    fitted_fiber = interpolate_fiber(fiber, (matrix, matrix), active=(0,))
    bra, ket, spectator = np.meshgrid(fine, fine, fine, indexing="ij")
    expected_fiber = (bra * ket + spectator**2).reshape(25, 5)
    np.testing.assert_allclose(fitted_fiber, expected_fiber, atol=1.0e-13)


def test_matrix_field_samplers_decode_electronic_and_paired_indices():
    shape = (2, 3)
    nstates = 2
    ngrid = int(np.prod(shape))
    local = np.arange(ngrid * nstates**2).reshape(*shape, nstates, nstates)
    overlap = np.arange((ngrid * nstates) ** 2).reshape(ngrid, nstates, ngrid, nstates)
    oracle = ArrayOracle(local, overlap)

    local_sampler = HamiltonianSampler(oracle, nstates)
    assert local_sampler((1, 2, 3)) == local[1, 2, 1, 1]
    element_sampler = HamiltonianSampler(oracle, nstates, element=(1, 0))
    assert element_sampler((1, 2)) == local[1, 2, 1, 0]

    fiber_sampler = FiberSampler(oracle, shape, nstates, active=(0,))
    index = (1 * shape[0] + 0, 2, 1 * nstates + 0)
    assert (
        fiber_sampler(index)
        == overlap[
            np.ravel_multi_index((1, 2), shape),
            1,
            np.ravel_multi_index((0, 2), shape),
            0,
        ]
    )
    assert fiber_sampler.pairs == {((1, 2), (0, 2))}
    element_fiber = FiberSampler(oracle, shape, nstates, active=(0,), element=(1, 0))
    assert (
        element_fiber(index[:-1])
        == overlap[
            np.ravel_multi_index((1, 2), shape),
            1,
            np.ravel_multi_index((0, 2), shape),
            0,
        ]
    )


def test_hermitian_sampler_caches_whole_matrices_per_geometry():
    class Oracle:
        def __init__(self):
            self.requests = []

        def hamiltonian_many(self, indices):
            self.requests.append(tuple(indices))
            output = []
            for (index,) in indices:
                coupling = 0.2 * index + 0.1j
                output.append([[index, coupling], [coupling.conjugate(), index + 1]])
            return np.asarray(output)

    oracle = Oracle()
    sampler = HermitianSampler(oracle, 2)
    indices = np.asarray(((2, 0), (2, 1), (2, 2), (3, 0), (2, 3)))
    values = sampler.batch(indices)

    assert len(oracle.requests) == 1
    assert set(oracle.requests[0]) == {(2,), (3,)}
    assert sampler.points == {(2,), (3,)}
    assert np.all(np.isfinite(values))
    sampler.batch(np.asarray(((2, 0), (3, 1))))
    assert len(oracle.requests) == 1


def test_hermitian_functional_cross_uses_selected_geometries():
    shape = (7, 7, 7)
    grids = tuple(np.linspace(-1.0, 1.0, size) for size in shape)
    x, y, z = np.meshgrid(*grids, indexing="ij")
    values = np.empty((*shape, 2, 2), dtype=complex)
    values[..., 0, 0] = 1.0 + 0.2 * x + 0.1 * y * z
    values[..., 1, 1] = 0.6 - 0.1 * y + 0.05 * x * z
    coupling = (
        0.2 * (1.0 + x) * (1.0 - 0.3 * y) * (1.0 + 0.1 * z)
        + 0.05j * (x - y) * (1.0 + 0.2 * z)
    )
    values[..., 0, 1] = coupling
    values[..., 1, 0] = coupling.conj()

    class Oracle:
        def hamiltonian_many(self, indices):
            return np.asarray([values[tuple(index)] for index in indices])

    model, info = fit_hamiltonian(
        Oracle(),
        grids,
        2,
        max_rank=4,
        degrees=2,
        sweeps=5,
        rtol=1.0e-10,
        validation=40,
        seed=2,
        start_rank=1,
        kick_rank=1,
    )
    points = np.stack([axis.reshape(-1) for axis in (x, y, z)], axis=1)
    predicted = model.predict(points).reshape(values.shape)

    np.testing.assert_allclose(predicted, values, atol=1.0e-12)
    assert info["backend"] == "hermitian-functional-tt-cross"
    assert info["unique_geometries"] < np.prod(shape)
    assert info["unique_geometries"] < info["samples"]


def test_feature_sampler_caches_whole_matrices_per_geometry():
    class Oracle:
        rank = 3
        nstates = 2

        def __init__(self):
            self.requests = []

        def feature_many(self, indices):
            self.requests.append(tuple(indices))
            return np.asarray([
                np.arange(6).reshape(3, 2) + index[0] * (1.0 + 0.2j)
                for index in indices
            ])

    oracle = Oracle()
    sampler = FeatureSampler(oracle)
    indices = np.asarray(((2, 0), (2, 3), (3, 1), (2, 5)))
    values = sampler.batch(indices)

    assert len(oracle.requests) == 1
    assert set(oracle.requests[0]) == {(2,), (3,)}
    assert sampler.points == {(2,), (3,)}
    assert np.iscomplexobj(values)
    sampler.batch(np.asarray(((2, 1), (3, 4))))
    assert len(oracle.requests) == 1


def test_feature_functional_cross_uses_selected_geometries():
    shape = (7, 7, 7)
    grids = tuple(np.linspace(-1.0, 1.0, size) for size in shape)
    x, y, z = np.meshgrid(*grids, indexing="ij")
    values = np.empty((*shape, 3, 2), dtype=complex)
    values[..., 0, 0] = (1.0 + 0.2 * x) * (1.0 - 0.1 * y)
    values[..., 0, 1] = 0.3j * (1.0 + z)
    values[..., 1, 0] = (x - 0.2j * y) * (1.0 + 0.1 * z)
    values[..., 1, 1] = 0.4 + 0.1 * x * y
    values[..., 2, 0] = (1.0 - x) * (0.2 + 0.1j * z)
    values[..., 2, 1] = y + 0.2j * z

    class Oracle:
        rank = 3
        nstates = 2

        def feature_many(self, indices):
            return np.asarray([values[tuple(index)] for index in indices])

    model, info = fit_features(
        Oracle(),
        grids,
        max_rank=6,
        degrees=2,
        sweeps=5,
        rtol=1.0e-10,
        validation=40,
        seed=3,
        start_rank=6,
        kick_rank=1,
    )
    points = np.stack([axis.reshape(-1) for axis in (x, y, z)], axis=1)
    predicted = model.predict(points).reshape(values.shape)

    np.testing.assert_allclose(predicted, values, atol=1.0e-12)
    assert not model.hermitian_
    assert model.output_shape_ == (3, 2)
    assert info["backend"] == "feature-functional-tt-cross"
    assert info["unique_geometries"] < np.prod(shape)
    assert info["unique_geometries"] < info["samples"]


def test_link_sampler_caches_matrices_and_endpoint_geometries():
    class Oracle:
        def __init__(self):
            self.requests = []

        def overlap_many(self, pairs):
            self.requests.append(tuple(pairs))
            return np.asarray([
                [[left[0] + right[1], 0.2j], [-0.1j, left[1] + right[0]]]
                for left, right in pairs
            ])

    oracle = Oracle()
    sampler = LinkSampler(oracle, (3, 4), axis=0, nstates=2)
    indices = np.asarray(((1, 2, 0), (1, 2, 3), (0, 0, 1)))
    values = sampler.batch(indices)

    assert len(oracle.requests) == 1
    assert set(oracle.requests[0]) == {
        ((1, 2), (2, 2)),
        ((0, 0), (1, 0)),
    }
    assert sampler.points == {(0, 0), (1, 0), (1, 2), (2, 2)}
    assert np.iscomplexobj(values)
    sampler.batch(np.asarray(((1, 2, 1), (0, 0, 2))))
    assert len(oracle.requests) == 1


def test_directional_link_cross_reconstructs_complex_matrix_fields():
    grids = (np.linspace(-1.0, 1.0, 7), np.linspace(-0.8, 0.8, 6))

    def field(axis, left, right):
        x = 0.5 * (grids[0][left[0]] + grids[0][right[0]])
        y = 0.5 * (grids[1][left[1]] + grids[1][right[1]])
        return np.asarray(
            [
                [1.0 + 0.1 * x + 0.03 * axis * y, 0.2 * x + 0.1j * y],
                [-0.1 * y + 0.05j * x, 0.8 - 0.05 * y + 0.02 * axis * x],
            ]
        )

    class Oracle:
        def overlap_many(self, pairs):
            values = []
            for left, right in pairs:
                axis = int(np.flatnonzero(np.asarray(right) - left)[0])
                values.append(field(axis, left, right))
            return np.asarray(values)

    models, info = fit_links(
        Oracle(),
        grids,
        2,
        max_rank=6,
        degrees=2,
        sweeps=5,
        rtol=1.0e-11,
        validation=40,
        seed=6,
        start_rank=6,
        kick_rank=1,
    )
    for axis, model in enumerate(models):
        np.testing.assert_allclose(
            model.bounds_,
            [(grid[0], grid[-1]) for grid in grids],
        )
        edge_grids = list(grids)
        edge_grids[axis] = 0.5 * (
            grids[axis][:-1] + grids[axis][1:]
        )
        mesh = np.meshgrid(*edge_grids, indexing="ij")
        coordinates = np.stack([item.reshape(-1) for item in mesh], axis=1)
        predicted = model.predict(coordinates).reshape(
            *(len(grid) for grid in edge_grids), 2, 2
        )
        exact = np.empty_like(predicted)
        for left in np.ndindex(predicted.shape[:-2]):
            right = list(left)
            right[axis] += 1
            exact[left] = field(axis, left, tuple(right))
        np.testing.assert_allclose(predicted, exact, atol=1.0e-12)
        assert not model.hermitian_
        refined = tuple(
            np.linspace(grid[0], grid[-1], 2 * len(grid) - 1)
            for grid in grids
        )
        refined_edges = list(refined)
        refined_edges[axis] = 0.5 * (
            refined[axis][:-1] + refined[axis][1:]
        )
        refined_mesh = np.meshgrid(*refined_edges, indexing="ij")
        refined_coordinates = np.stack(
            [item.reshape(-1) for item in refined_mesh], axis=1
        )
        refined_predicted = model.predict(refined_coordinates)
        refined_exact = []
        for point in refined_coordinates:
            refined_exact.append(
                np.asarray(
                    [
                        [
                            1.0 + 0.1 * point[0] + 0.03 * axis * point[1],
                            0.2 * point[0] + 0.1j * point[1],
                        ],
                        [
                            -0.1 * point[1] + 0.05j * point[0],
                            0.8 - 0.05 * point[1] + 0.02 * axis * point[0],
                        ],
                    ]
                )
            )
        np.testing.assert_allclose(
            refined_predicted,
            refined_exact,
            atol=1.0e-12,
        )
    assert info["backend"] == "directional-link-functional-tt-cross"
    assert info["unique_geometries"] <= np.prod(tuple(map(len, grids)))
    assert len(info["directions"]) == 2


def test_matrix_block_cur_reconstructs_low_rank_energy_and_links():
    grids = tuple(np.linspace(-1.0, 1.0, size) for size in (6, 5, 6))

    def coordinate(index):
        return tuple(grids[axis][value] for axis, value in enumerate(index))

    def energy(index):
        x, y, z = coordinate(index)
        coupling = 0.04 * x * y + 0.02j * z
        return np.asarray(
            [[0.2 + 0.1 * x + 0.03 * z, coupling],
             [coupling.conjugate(), 0.7 - 0.05 * y]]
        )

    def link(axis, left):
        x, y, z = coordinate(left)
        return np.asarray(
            [[0.9 + 0.02 * axis * x, 0.03 * x * y + 0.01j * z],
             [-0.02 * y + 0.01j * x, 0.8 - 0.04 * y * z]]
        )

    class Oracle:
        def hamiltonian_many(self, indices):
            return np.asarray([energy(tuple(index)) for index in indices])

        def overlap_many(self, pairs):
            blocks = []
            for left, right in pairs:
                left = tuple(left)
                axis = int(np.flatnonzero(np.asarray(right) - left)[0])
                blocks.append(link(axis, left))
            return np.asarray(blocks)

    energy_fit, link_fits, info = fit_cur(
        Oracle(),
        grids,
        2,
        rank=6,
        degrees=2,
        axis=1,
        slabs=2,
        probes=6,
        seed=5,
    )
    mesh = np.meshgrid(*grids, indexing="ij")
    points = np.stack([values.reshape(-1) for values in mesh], axis=1)
    exact_energy = np.asarray([energy(index) for index in np.ndindex((6, 5, 6))])
    np.testing.assert_allclose(
        energy_fit.predict(points).reshape(6, 5, 6, 2, 2),
        exact_energy.reshape(6, 5, 6, 2, 2),
        atol=1.0e-10,
    )
    fitted_links = grid_links(link_fits, grids)
    for axis in range(3):
        edge_shape = [6, 5, 6]
        edge_shape[axis] -= 1
        for left in np.ndindex(tuple(edge_shape)):
            np.testing.assert_allclose(
                fitted_links[(axis, left)],
                link(axis, left),
                atol=1.0e-10,
            )
    assert info["backend"] == "matrix-block-cur"
    for item in info["links"]:
        edge_shape = [6, 5, 6]
        edge_shape[item["direction"]] -= 1
        assert item["sampled_links"] < np.prod(edge_shape)


def test_link_path_uses_fixed_order_and_reverse_adjoint():
    first = np.asarray([[1.0, 0.2], [0.1j, 0.8]])
    second = np.asarray([[0.9, -0.2j], [0.1, 1.1]])
    links = {
        (0, (0, 0)): first,
        (0, (0, 1)): np.eye(2),
        (1, (0, 0)): np.eye(2),
        (1, (1, 0)): second,
    }
    oracle = LinkPath((2, 2), 2, links, order=(0, 1))
    forward, reverse, diagonal = oracle.overlap_many(
        [((0, 0), (1, 1)), ((1, 1), (0, 0)), ((1, 0), (1, 0))]
    )

    np.testing.assert_allclose(forward, first @ second)
    np.testing.assert_allclose(reverse, forward.conj().T)
    np.testing.assert_allclose(diagonal, np.eye(2))
    assert oracle.used_links == {(0, (0, 0)), (1, (1, 0))}


def test_fit_overlap_builds_full_nuclear_electronic_mpo():
    shape = (3, 3)
    nstates = 2
    links = {}
    for axis in range(2):
        link_shape = list(shape)
        link_shape[axis] -= 1
        for index in np.ndindex(tuple(link_shape)):
            links[(axis, index)] = np.eye(nstates)
    oracle = LinkPath(shape, nstates, links)
    overlap, info = fit_overlap(
        oracle,
        shape,
        nstates,
        max_rank=1,
        sweeps=2,
        rtol=1.0e-12,
        validation=32,
    )

    expected = np.kron(np.ones((np.prod(shape),) * 2), np.eye(nstates))
    np.testing.assert_allclose(overlap.to_dense(), expected, atol=1.0e-12)
    assert info["backend"] == "path-overlap-tt-cross"
    assert info["unique_overlap_blocks"] < np.prod(shape) ** 2


def test_kinetic_sampler_skips_zero_kinetic_pairs():
    class Oracle:
        def __init__(self):
            self.pairs = []

        def overlap_many(self, pairs):
            self.pairs.extend(pairs)
            return np.repeat(np.eye(2)[None], len(pairs), axis=0)

    hopping = np.asarray([[0.0, -0.3], [-0.3, 0.0]])
    sampler = KineticSampler(
        Oracle(),
        [(1.0, (hopping, np.eye(2)))],
        (2, 2),
        2,
        (0,),
    )
    values = sampler.batch(
        np.asarray(
            [
                (0, 0, 0),
                (1, 0, 0),
                (1, 0, 3),
            ]
        )
    )

    np.testing.assert_allclose(values, [0.0, -0.3, -0.3])
    assert sampler.pairs == {
        ((0, 0), (0, 0)),
        ((0, 0), (1, 0)),
    }
    assert sampler.transport_pairs == {((0, 0), (1, 0))}
    assert sampler.oracle.pairs == [((0, 0), (1, 0))]


def test_fit_kinetic_directly_builds_dressed_mpo():
    shape = (3, 2)
    nstates = 2
    links = {}
    for axis in range(2):
        link_shape = list(shape)
        link_shape[axis] -= 1
        for index in np.ndindex(tuple(link_shape)):
            links[(axis, index)] = np.eye(nstates)
    oracle = LinkPath(shape, nstates, links)
    hopping0 = np.asarray(
        [[0.0, -0.2, 0.0], [-0.2, 0.0, -0.2], [0.0, -0.2, 0.0]]
    )
    hopping1 = np.asarray([[0.0, -0.1], [-0.1, 0.0]])
    diagonal0 = np.diag([0.8, 1.0, 1.2])
    diagonal1 = np.diag([0.7, 1.1])
    terms = [
        (1.0, (hopping0, diagonal1)),
        (0.4, (diagonal0, hopping1)),
        (0.2, (diagonal0, diagonal1)),
    ]
    kinetic, info = fit_kinetic(
        oracle,
        terms,
        shape,
        nstates,
        max_rank=8,
        sweeps=4,
        rtol=1.0e-12,
        validation=64,
        start_rank=8,
        kick_rank=1,
    )

    groups = group_kinetic_terms(terms, shape)
    bare = sum(groups.values())
    expected = np.kron(bare, np.eye(nstates))
    np.testing.assert_allclose(kinetic.to_dense(), expected, atol=1.0e-11)
    assert info["backend"] == "dressed-kinetic-tt-cross"
    assert info["groups"] == 3
    assert info["unique_transport_pairs"] < info["unique_sampled_pairs"]


def test_link_mpo_kinetic_builds_nonunitary_path_products_directly():
    shape = (3, 2)
    nstates = 2
    links = {
        (0, (0, 0)): np.asarray([[0.92, 0.08], [-0.03, 0.85]]),
        (0, (1, 0)): np.asarray([[0.88, -0.05], [0.04, 0.81]]),
        (0, (0, 1)): np.asarray([[0.86, 0.02], [0.07, 0.90]]),
        (0, (1, 1)): np.asarray([[0.83, -0.06], [0.01, 0.87]]),
        (1, (0, 0)): np.asarray([[0.91, 0.03], [-0.02, 0.84]]),
        (1, (1, 0)): np.asarray([[0.89, -0.04], [0.05, 0.82]]),
        (1, (2, 0)): np.asarray([[0.87, 0.01], [0.06, 0.80]]),
    }
    path = LinkPath(shape, nstates, links)
    kinetic0 = np.asarray(
        [[0.7, -0.2, 0.05], [-0.2, 0.8, -0.15], [0.05, -0.15, 0.9]]
    )
    kinetic1 = np.asarray([[0.4, -0.1], [-0.1, 0.6]])
    terms = (
        (1.0, (kinetic0, np.eye(2))),
        (1.0, (np.eye(3), kinetic1)),
    )
    components, info = link_mpo_kinetic(
        path,
        terms,
        shape,
        nstates,
        max_rank=32,
        operator_rank=None,
        split=True,
    )

    expected = np.zeros((np.prod(shape) * nstates,) * 2, dtype=complex)
    for left in np.ndindex(shape):
        left_flat = np.ravel_multi_index(left, shape)
        for right in np.ndindex(shape):
            differing = [axis for axis in range(2) if left[axis] != right[axis]]
            value = 0.0
            if not differing:
                value = kinetic0[left[0], right[0]] + kinetic1[left[1], right[1]]
            elif differing == [0]:
                value = kinetic0[left[0], right[0]]
            elif differing == [1]:
                value = kinetic1[left[1], right[1]]
            if value == 0.0:
                continue
            right_flat = np.ravel_multi_index(right, shape)
            block = path.between(left, right)
            expected[
                left_flat * nstates : (left_flat + 1) * nstates,
                right_flat * nstates : (right_flat + 1) * nstates,
            ] += value * block
    expected = 0.5 * (expected + expected.conj().T)

    actual = sum(component.to_dense() for component in components)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    assert info["backend"] == "directional-link-overlap-mpo"
    assert all(
        field["backend"] == "directional-link-overlap-tt-svd"
        for field in info["fields"].values()
    )


def test_corewise_link_mpo_uses_fitted_cores_for_nonunitary_intervals():
    shape = (3, 2)
    nstates = 2
    links = {
        (0, (0, 0)): np.asarray([[0.92, 0.08], [-0.03, 0.85]]),
        (0, (1, 0)): np.asarray([[0.88, -0.05], [0.04, 0.81]]),
        (0, (0, 1)): np.asarray([[0.86, 0.02], [0.07, 0.90]]),
        (0, (1, 1)): np.asarray([[0.83, -0.06], [0.01, 0.87]]),
        (1, (0, 0)): np.asarray([[0.91, 0.03], [-0.02, 0.84]]),
        (1, (1, 0)): np.asarray([[0.89, -0.04], [0.05, 0.82]]),
        (1, (2, 0)): np.asarray([[0.87, 0.01], [0.06, 0.80]]),
    }

    class CoreOnlyLink:
        output_shape_ = (nstates, nstates)

        def __init__(self, values):
            self.values = np.asarray(values)

        def tensor_cores(self, grids):
            expected = self.values.shape[:-2]
            assert tuple(len(grid) for grid in grids) == expected
            tensor = self.values.reshape(*expected, nstates * nstates)
            return decompose(tensor, rank=64)

    fields = []
    for axis in range(2):
        edge_shape = list(shape)
        edge_shape[axis] -= 1
        values = np.empty((*edge_shape, nstates, nstates), dtype=complex)
        for index in np.ndindex(tuple(edge_shape)):
            values[index] = links[(axis, index)]
        fields.append(CoreOnlyLink(values))

    kinetic0 = np.asarray(
        [[0.7, -0.2, 0.05], [-0.2, 0.8, -0.15], [0.05, -0.15, 0.9]]
    )
    kinetic1 = np.asarray([[0.4, -0.1], [-0.1, 0.6]])
    terms = (
        (1.0, (kinetic0, np.eye(2))),
        (1.0, (np.eye(3), kinetic1)),
    )
    components, info = corewise_link_mpo_kinetic(
        fields,
        tuple(np.arange(size, dtype=float) for size in shape),
        terms,
        nstates,
        max_rank=64,
        operator_rank=None,
        split=True,
    )

    path = LinkPath(shape, nstates, links)
    expected = np.zeros((np.prod(shape) * nstates,) * 2, dtype=complex)
    for left in np.ndindex(shape):
        left_flat = np.ravel_multi_index(left, shape)
        for right in np.ndindex(shape):
            differing = [axis for axis in range(2) if left[axis] != right[axis]]
            if not differing:
                coefficient = (
                    kinetic0[left[0], right[0]]
                    + kinetic1[left[1], right[1]]
                )
            elif differing == [0]:
                coefficient = kinetic0[left[0], right[0]]
            elif differing == [1]:
                coefficient = kinetic1[left[1], right[1]]
            else:
                coefficient = 0.0
            if coefficient == 0.0:
                continue
            right_flat = np.ravel_multi_index(right, shape)
            block = path.between(left, right)
            expected[
                left_flat * nstates : (left_flat + 1) * nstates,
                right_flat * nstates : (right_flat + 1) * nstates,
            ] += coefficient * block

    actual = sum(component.to_dense() for component in components)
    np.testing.assert_allclose(actual, expected, atol=2.0e-12)
    assert info["backend"] == "corewise-directional-link-mpo"
    assert not info["materialized_link_grid"]
    assert not info["materialized_overlap_fiber"]
    assert all(
        scan["backend"] == "functional-tt-corewise-scan"
        for scan in info["axis_scans"].values()
    )


def test_frame_and_procrustes_oracles_batch_cached_records(tmp_path):
    built = []

    def builder(index):
        built.append(index)
        angle = 0.1 * index[0]
        rotation = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return rotation, np.asarray([index[0], index[0] + 0.5])

    frames = Frames((4,), builder, cache_dir=tmp_path, workers=1)
    oracle = ProcrustesOracle(
        frames,
        (0,),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    )
    local = oracle.hamiltonian_many([(1,), (3,), (1,)])
    blocks = oracle.overlap_many([((1,), (3,)), ((3,), (1,))])

    assert sorted(built) == [(0,), (1,), (3,)]
    expected = []
    for position in (1, 3, 1):
        angle = 0.1 * position
        gauge = np.asarray(
            [
                [np.cos(angle), np.sin(angle)],
                [-np.sin(angle), np.cos(angle)],
            ]
        )
        expected.append(gauge.T @ np.diag([position, position + 0.5]) @ gauge)
    np.testing.assert_allclose(local, expected)
    np.testing.assert_allclose(
        blocks, np.broadcast_to(np.eye(2), (2, 2, 2)), atol=1.0e-13
    )
    frames.close()

    restored = Frames((4,), cache_dir=tmp_path)
    assert restored.get((3,))[1][0] == 3.0
    assert restored.stats["restored"] == 1


def test_frames_accept_boolean_progress(capsys):
    frames = Frames((2,), lambda index: index, progress=True)
    frames.get_many(((0,), (1,)))
    assert "electronic point 2" in capsys.readouterr().out


def test_aligned_fit_reuses_frames_between_energy_and_link_crosses(tmp_path):
    built = []

    def builder(index):
        built.append(index)
        x, y = index
        angle = 0.07 * x - 0.03 * y
        frame = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return frame, np.asarray([0.2 * x + 0.1 * y, 0.5 + 0.1 * x])

    grids = (np.linspace(-1.0, 1.0, 3), np.linspace(-0.5, 0.5, 3))
    frames = Frames((3, 3), builder, cache_dir=tmp_path)
    oracle = ProcrustesOracle(
        frames,
        (1, 1),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    )
    energy, links, info = fit_aligned(
        oracle,
        grids,
        2,
        max_rank=3,
        sweeps=2,
        validation=8,
        seed=4,
    )

    assert energy.output_shape_ == (2, 2)
    assert len(links) == 2
    assert len(built) == len(set(built))
    assert info["quantum_chemistry_calls"] == len(built)
    assert info["unique_geometries"] == len(built)
    assert info["frame_sampling"]["links"]["memory_hits"] > 0
    assert frames.stats["unique_requested"] == len(built)
    frames.close()


def test_block_cross_selects_shared_vertices_for_complete_matrices(tmp_path):
    built = []

    def builder(index):
        built.append(index)
        x, y = index
        angle = 0.03 * x - 0.02 * y
        frame = np.asarray(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        return frame, np.asarray([0.1 * x + 0.03 * y, 0.5 + 0.02 * x])

    grids = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.5, 0.5, 5))
    frames = Frames((5, 5), builder, cache_dir=tmp_path)
    oracle = ProcrustesOracle(
        frames,
        (2, 2),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    )
    energy, links, info = fit_block_cross(
        oracle,
        grids,
        2,
        rank=2,
        degrees=2,
        sweeps=3,
        validation=8,
        seed=4,
    )

    assert energy.output_shape_ == (2, 2)
    assert len(links) == 2
    assert info["backend"] == "shared-block-functional-tt-cross"
    assert info["selected_vertices"] < 25
    assert info["quantum_chemistry_calls"] == len(set(built))
    assert all(count > 0 for count in info["link_samples"])
    frames.close()


def test_ab_initio_fit_owns_sampling_and_persistence(tmp_path):
    built = []

    def builder(index):
        built.append(index)
        x, y = index
        angle = 0.04 * x - 0.02 * y
        frame = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return frame, np.asarray([0.1 * x + 0.03 * y, 0.5 + 0.02 * x])

    grids = (np.linspace(-1.0, 1.0, 3), np.linspace(-0.5, 0.5, 3))
    output = tmp_path / "fit"
    with AbInitioFit(
        grids,
        2,
        builder,
        anchor=(1, 1),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
        cache=tmp_path / "frames",
    ) as fit:
        fit.run(rank=3, degrees=2, sweeps=2, validation=8, seed=5)
        mesh = np.meshgrid(*grids, indexing="ij")
        refinement_coordinates = np.stack(
            [values.reshape(-1) for values in mesh], axis=1
        )
        refinement_values = fit.energy.predict(refinement_coordinates)
        fit.refine_hamiltonian(
            refinement_coordinates,
            refinement_values,
            degrees=2,
            rank=4,
            sweeps=8,
            rtol=1.0e-12,
        )
        assert fit.hamiltonian_refinement["representation"] == (
            "full-procrustes-gauged-hermitian-matrix"
        )
        assert fit.hamiltonian_refinement["relative_refinement_error"] < 1.0e-8
        fit.save(output, labels=("x", "y"), metadata={"method": "test"})
        expected = fit.energy.predict(np.asarray([[0.0, 0.0]]))
        identity = np.eye(3)
        driver = TTLDR.from_fit(
            fit,
            keo=[(1.0, (identity, identity))],
            overlap_rank=3,
            overlap_sweeps=2,
            overlap_validation=8,
            operator_rank=None,
        )
        assert fit.success
        assert driver.solver is None
        assert driver.dims == (3, 3, 2)
        assert driver._hamiltonian is None
        for component in driver.components:
            dense = component.to_dense()
            np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-11)
        values = np.zeros(driver.dims, dtype=complex)
        values[1, 1, 0] = 1.0
        state = driver.state(values, physical=False)
        with pytest.raises(ValueError, match="explicit e_ops"):
            driver.run(state, dt=0.01, steps=0, progress=False)
        driver.run(
            state,
            dt=0.001,
            steps=1,
            max_bond=8,
            progress=False,
            e_ops=driver.projectors(),
        )
        np.testing.assert_allclose(driver.norms, 1.0, atol=1.0e-10)
        assert fit.stats["fit"]["quantum_chemistry_calls"] == len(set(built))

    restored = AbInitioFit.load(output)
    np.testing.assert_allclose(
        restored.energy.predict(np.asarray([[0.0, 0.0]])),
        expected,
    )
    assert restored.shape == (3, 3)
    assert restored.anchor == (1, 1)
    assert len(restored.links) == 2
    assert restored.paths["links"][0].name == "bar_l_x.npz"
    restored.close()


def test_fitted_feature_regenerates_links_on_a_refined_grid(tmp_path):
    def builder(index):
        x, y = index
        angle = 0.18 * x - 0.11 * y
        frame = np.asarray(
            [
                [np.cos(angle), 0.0],
                [0.0, 1.0],
                [np.sin(angle), 0.0],
            ]
        )
        return frame, np.asarray([0.03 * x + 0.01 * y, 0.4 + 0.02 * y])

    training = (np.linspace(-1.0, 1.0, 3), np.linspace(-0.5, 0.5, 3))
    refined = (np.linspace(-1.0, 1.0, 4), np.linspace(-0.5, 0.5, 4))
    output = tmp_path / "feature-fit"
    with AbInitioFit(
        training,
        2,
        builder,
        anchor=(1, 1),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    ) as fit:
        fit.run(
            representation="features",
            feature_rank=3,
            feature_penalty=50.0,
            feature_maxiter=300,
            rank=8,
            degrees=2,
            sweeps=3,
            validation=8,
            seed=9,
        )
        fit.save(output, labels=("x", "y"))
        hopping = np.asarray(
            [
                [0.7, -0.2, 0.03, 0.0],
                [-0.2, 0.8, -0.15, 0.02],
                [0.03, -0.15, 0.9, -0.1],
                [0.0, 0.02, -0.1, 1.0],
            ]
        )
        identity = np.eye(4)
        terms = (
            (1.0, (hopping, identity)),
            (1.0, (identity, hopping)),
        )
        driver = TTLDR.from_fit(
            fit,
            grids=refined,
            keo=terms,
            overlap_rank=16,
            operator_rank=None,
        )

        mesh = np.meshgrid(*refined, indexing="ij")
        points = np.stack([value.reshape(-1) for value in mesh], axis=1)
        features = fit.feature.predict(points).reshape(4, 4, 3, 2)
        links = {}
        for axis in range(2):
            edge_shape = [4, 4]
            edge_shape[axis] -= 1
            for left in np.ndindex(tuple(edge_shape)):
                right = list(left)
                right[axis] += 1
                links[(axis, left)] = (
                    features[left].conj().T @ features[tuple(right)]
                )
        path = LinkPath((4, 4), 2, links)
        expected = np.zeros((32, 32), dtype=complex)
        for left in np.ndindex(4, 4):
            left_flat = np.ravel_multi_index(left, (4, 4))
            for right in np.ndindex(4, 4):
                differing = [axis for axis in range(2) if left[axis] != right[axis]]
                if not differing:
                    coefficient = hopping[left[0], right[0]] + hopping[left[1], right[1]]
                elif differing == [0]:
                    coefficient = hopping[left[0], right[0]]
                elif differing == [1]:
                    coefficient = hopping[left[1], right[1]]
                else:
                    coefficient = 0.0
                if coefficient == 0.0:
                    continue
                right_flat = np.ravel_multi_index(right, (4, 4))
                expected[
                    2 * left_flat : 2 * left_flat + 2,
                    2 * right_flat : 2 * right_flat + 2,
                ] += coefficient * path.between(left, right)

        actual = sum(component.to_dense() for component in driver.kinetic)
        np.testing.assert_allclose(actual, expected, atol=2.0e-10)
        assert driver.overlap_info["feature_links"]["backend"] == (
            "feature-endpoint-link-cores"
        )
        assert not driver.overlap_info["feature_links"]["materialized_link_grid"]
        assert driver.overlap_info["action"] == "linked-product-approximation"
        assert not driver.overlap_info["unitarized"]
        assert driver.transports["linked_product_approximation"]

    restored = AbInitioFit.load(output)
    assert restored.links is None
    assert restored.feature.output_shape_ == (3, 2)
    assert restored.paths["feature"].name == "y.npz"
    restored.close()


def test_sparse_aligned_fit_completes_links_between_sampled_vertices(tmp_path):
    built = []

    def builder(index):
        built.append(index)
        x, y = index
        angle = 0.04 * x + 0.02 * y
        frame = np.asarray(
            [
                [np.cos(angle), -np.sin(angle)],
                [np.sin(angle), np.cos(angle)],
            ]
        )
        return frame, np.asarray([0.1 * x + 0.05 * y, 0.4 + 0.03 * x])

    grids = (np.linspace(-1.0, 1.0, 5), np.linspace(-0.8, 0.8, 5))
    frames = Frames((5, 5), builder, cache_dir=tmp_path)
    oracle = ProcrustesOracle(
        frames,
        (2, 2),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
    )
    energy, links, info = fit_sparse(
        oracle,
        grids,
        2,
        rank=2,
        degrees=2,
        initial=8,
        validation=4,
        rounds=2,
        rtol=1.0e-5,
        sweeps=8,
        seed=3,
    )

    assert energy.output_shape_ == (2, 2)
    assert len(links) == 2
    assert info["tested_vertices"] <= 16
    assert info["quantum_chemistry_calls"] == len(set(built))
    assert info["unique_geometries"] < 25
    assert info["model_training_geometries"] == info["energy_samples"]
    assert info["sequence"] == "halton"
    assert all(count > 0 for count in info["link_samples"])
    frames.close()


def test_ab_initio_fit_exposes_matrix_block_cur_sampler(tmp_path):
    grids = tuple(np.linspace(-1.0, 1.0, 4) for _ in range(3))

    def builder(index):
        angle = 0.02 * sum(index)
        frame = np.asarray(
            [[np.cos(angle), -np.sin(angle)],
             [np.sin(angle), np.cos(angle)]]
        )
        return frame, np.asarray([0.05 * sum(index), 0.4 + 0.02 * index[1]])

    with AbInitioFit(
        grids,
        2,
        builder,
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=lambda left, right: left.T @ right,
        cache=tmp_path,
    ) as fit:
        fit.run(
            sampler="cur",
            rank=3,
            degrees=2,
            cur_axis=1,
            cur_slabs=2,
            cur_probes=3,
            seed=7,
        )

    assert fit.success
    assert fit.info["backend"] == "matrix-block-cur"
    assert fit.config["cur_axis"] == 1
    assert fit.config["cur_slabs"] == 2
    assert fit.config["cur_probes"] == 3


def test_frames_parallelize_one_batch_and_persist_results(tmp_path):
    with Frames(
        (2, 3),
        parallel_record,
        cache_dir=tmp_path,
        workers=2,
    ) as oracle:
        indices = [(0, 0), (1, 2), (0, 2), (1, 2)]
        try:
            records = oracle.get_many(indices)
        except (NotImplementedError, PermissionError) as error:
            pytest.skip(f"process pools are unavailable: {error}")
        assert records == [((0, 0), 0), ((1, 2), 5), ((0, 2), 2)]
        assert oracle.stats["built"] == 3
        assert oracle.stats["batches"] == 1

    restored = Frames((2, 3), cache_dir=tmp_path)
    assert restored.get((1, 2)) == ((1, 2), 5)
