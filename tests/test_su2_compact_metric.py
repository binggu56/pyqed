import numpy as np
import pytest

from pyqed.mps.nonabelian import solver as solver_module
from pyqed.mps.nonabelian.basis import (
    LocalLayoutEntry,
    TwoSiteBasis,
)
from pyqed.symmetry import Leg
from pyqed.mps.nonabelian.decompose import (
    _build_side_pipe,
    _factor_metric_weighted_projected_svd,
)
from pyqed.mps.nonabelian.linalg import (
    ReducedProjectedSector,
    project_reduced_sector,
)
from pyqed.mps.nonabelian.local_operator import build_identity_mpo_local_actions
from pyqed.mps.nonabelian.mpo import SparseVirtualBlock
from pyqed.mps.nonabelian.renormalized import (
    FactorizedRouteMetricBlock,
    KroneckerMetricBlock,
    KroneckerMetricTransform,
    RenormalizedOperatorStack,
)
from pyqed.mps.nonabelian.solver import _factorized_route_metric_transform
from pyqed.mps.nonabelian.tensor import (
    FusionPipe,
    FusionPipeEntry,
    IdentityBasisTransform,
)


def test_packed_generalized_davidson_caps_owned_basis_workspace(monkeypatch):
    size = 100
    diagonal = np.arange(size, dtype=float)
    guess = np.ones(size, dtype=float)
    bytes_per_column = (
        size
        * np.dtype(np.complex128).itemsize
        * solver_module._PACKED_DAVIDSON_OWNED_BASIS_ARRAYS
    )
    monkeypatch.setattr(
        solver_module,
        "_PACKED_DAVIDSON_BASIS_MAX_BYTES",
        2 * bytes_per_column,
    )

    theta, _vector, objective = (
        solver_module._solve_packed_generalized_davidson(
            guess,
            lambda vector: diagonal * vector,
            h_diag=diagonal,
            N=lambda vector: vector,
            n_diag=np.ones(size),
            tol=1.0e-12,
            tol_residual=1.0e-10,
            itermax=20,
            max_space=20,
            profile=True,
        )
    )

    assert abs(theta) < 1.0e-12
    assert objective["packed_dimension"] == size
    assert objective["requested_max_space"] == 20
    assert objective["workspace_max_space"] == 2
    assert objective["workspace_limited"] is True
    assert objective["estimated_basis_workspace_bytes"] == 2 * bytes_per_column


def test_packed_davidson_adaptive_confirmation_removes_floor_16():
    size = 24
    diagonal = np.arange(size, dtype=float)
    guess = np.zeros(size)
    guess[0] = 1.0
    common = dict(
        H=lambda vector: diagonal * vector,
        h_diag=diagonal,
        tol=1.0e-12,
        tol_residual=1.0e-12,
        itermax=30,
        max_space=24,
    )

    _, _, legacy = solver_module._solve_packed_generalized_davidson(
        guess,
        minimum_explored_dimension=16,
        convergence_confirmations=1,
        **common,
    )
    theta, _, adaptive = solver_module._solve_packed_generalized_davidson(
        guess,
        minimum_explored_dimension=2,
        convergence_confirmations=2,
        **common,
    )

    assert abs(theta) < 1.0e-12
    assert adaptive["davidson_converged"]
    assert adaptive["davidson_iterations"] < legacy["davidson_iterations"]
    assert adaptive["convergence_streak"] == 2
    assert adaptive["root_overlap"] == pytest.approx(1.0)


def test_kronecker_metric_and_transform_match_dense_reference():
    rng = np.random.default_rng(421)
    left_factor = rng.normal(size=(3, 3))
    right_factor = rng.normal(size=(2, 2))
    left_metric = left_factor @ left_factor.T + np.eye(3)
    right_metric = right_factor @ right_factor.T + np.eye(2)
    left_values, left_vectors = np.linalg.eigh(left_metric)
    right_values, right_vectors = np.linalg.eigh(right_metric)
    left_transform = left_vectors / np.sqrt(left_values)[None, :]
    right_transform = right_vectors / np.sqrt(right_values)[None, :]

    metric = KroneckerMetricBlock(left_metric, right_metric, (2, 3))
    transform = KroneckerMetricTransform(
        left_transform,
        right_transform,
        (2, 3),
    )
    vector = rng.normal(size=metric.shape[0])

    np.testing.assert_allclose(metric @ vector, np.asarray(metric) @ vector)
    np.testing.assert_allclose(
        transform.T @ (transform @ vector),
        np.asarray(transform).T @ (np.asarray(transform) @ vector),
    )
    np.testing.assert_allclose(
        np.asarray(transform).T @ np.asarray(metric) @ np.asarray(transform),
        np.eye(transform.shape[1]),
        atol=1.0e-12,
    )
    assert metric.stored_elements < metric.shape[0] ** 2


def test_factorized_route_metric_matches_dense_reference():
    rng = np.random.default_rng(422)
    in_shape = (2, 2, 1, 3)
    out_shape = (3, 2, 1, 2)
    in_size = int(np.prod(in_shape))
    out_size = int(np.prod(out_shape))
    dim = in_size + out_size
    left = rng.normal(size=(out_shape[0], in_shape[0]))
    right = rng.normal(size=(out_shape[3], in_shape[3]))
    routes = (
        (
            slice(0, in_size),
            slice(in_size, dim),
            in_shape,
            out_shape,
            left,
            right,
        ),
        (
            slice(in_size, dim),
            slice(0, in_size),
            out_shape,
            in_shape,
            left.T,
            right.T,
        ),
    )
    metric = FactorizedRouteMetricBlock(dim=dim, routes=routes)
    vectors = rng.normal(size=(dim, 3))

    np.testing.assert_allclose(metric @ vectors, np.asarray(metric) @ vectors)
    assert metric.stored_elements < dim**2


def test_factorized_route_metric_owned_cholesky_transform_is_exact():
    left = np.asarray([[2.0, 0.3], [0.3, 1.5]])
    right = np.asarray([[1.2, -0.1], [-0.1, 1.8]])
    shape = (2, 2, 1, 2)
    dim = int(np.prod(shape))
    metric = FactorizedRouteMetricBlock(
        dim=dim,
        routes=((slice(0, dim), slice(0, dim), shape, shape, left, right),),
    )
    transform = _factorized_route_metric_transform(metric, tol=1.0e-12)

    np.testing.assert_allclose(
        transform.conj().T @ np.asarray(metric) @ transform,
        np.eye(transform.shape[1]),
        atol=1.0e-12,
    )


def test_identity_metric_scans_cross_sector_routes_after_dense_diagonal_block():
    left0, left1, phys, right = "left0", "left1", "phys", "right"
    basis = TwoSiteBasis(
        left=Leg((left0, left1), {left0: 2, left1: 1}),
        phys1=Leg((phys,), {phys: 1}),
        phys2=Leg((phys,), {phys: 1}),
        right=Leg((right,), {right: 1}),
        entries=(
            LocalLayoutEntry((left0, phys, phys, right), (2, 1, 1, 1), 0, 2),
            LocalLayoutEntry((left1, phys, phys, right), (1, 1, 1, 1), 2, 1),
        ),
    )
    full_left = np.asarray(
        [
            [1.3, 0.2, 0.4],
            [0.2, 0.9, -0.1],
            [0.4, -0.1, 1.1],
        ]
    )
    left_environment = {
        (left0, left0): full_left[:2, :2][None, ...],
        (left1, left1): full_left[2:, 2:][None, ...],
        (left0, left1): full_left[:2, 2:][None, ...],
        (left1, left0): full_left[2:, :2][None, ...],
    }
    right_environment = {(right, right): np.ones((1, 1, 1))}

    _tensor, _reduced, packed, _diag, _identity_like = (
        build_identity_mpo_local_actions(
            left_environment,
            right_environment,
            basis,
            base_dtype=float,
        )
    )
    vector = np.asarray([0.7, -0.3, 1.2])

    np.testing.assert_allclose(packed(vector), full_left @ vector)
    assert packed.factorized_metric_cross_sector_blocks == 2


def test_sparse_virtual_block_stores_only_active_routes():
    dense = np.zeros((128, 96, 2, 2))
    dense[2, 4] = np.eye(2)
    dense[90, 7] = 2.0 * np.eye(2)
    block = SparseVirtualBlock.from_dense(dense)

    assert block.nnz == 2
    assert block.values.size == 8
    np.testing.assert_array_equal(np.asarray(block), dense)


def test_single_entry_numeric_problem_cache_releases_before_rebuild():
    cache = RenormalizedOperatorStack(max_size=1)
    cache.put("old", np.ones(1024))

    cache.prepare_miss("new")

    assert len(cache) == 0


def test_product_side_pipe_keeps_identity_basis_transform_compact():
    block = np.arange(24.0).reshape(2, 1, 1, 12)
    entries = [(("left", "p1", "p2", "right"), block)]
    left_pipe, left_basis, _channels = _build_side_pipe(
        entries,
        "mid",
        side="left",
    )
    right_pipe, right_basis, _channels = _build_side_pipe(
        entries,
        "mid",
        side="right",
    )

    assert all(isinstance(value, IdentityBasisTransform) for value in left_basis.values())
    assert all(isinstance(value, IdentityBasisTransform) for value in right_basis.values())
    projected = project_reduced_sector(
        entries,
        "mid",
        left_pipe,
        right_pipe,
        left_basis,
        right_basis,
    )
    np.testing.assert_array_equal(projected.as_matrix(), block.reshape(2, 12))


def test_factor_metric_svd_preserves_state_and_exact_norm():
    left_entry = FusionPipeEntry(
        child_sectors=("left", "p1"),
        fused_sector="mid",
        slot=0,
        offset=0,
        local_dim=2,
        selected_shape=(2, 1),
    )
    right_entry = FusionPipeEntry(
        child_sectors=("p2", "right"),
        fused_sector="mid",
        slot=0,
        offset=0,
        local_dim=3,
        selected_shape=(1, 3),
    )
    left_pipe = FusionPipe.from_entries(
        child_legs=(0, 1),
        child_sector_lists=(("left",), ("p1",)),
        child_dirs=(-1, 1),
        fused_sectors=("mid",),
        entries=(left_entry,),
        coupling="left",
    )
    right_pipe = FusionPipe.from_entries(
        child_legs=(2, 3),
        child_sector_lists=(("p2",), ("right",)),
        child_dirs=(1, 1),
        fused_sectors=("mid",),
        entries=(right_entry,),
        coupling="left",
    )
    matrix = np.asarray([[0.2, -0.4, 0.7], [1.1, 0.3, -0.2]])
    identity_maps = {
        (left_entry.child_sectors, left_entry.selected_shape, 0):
            IdentityBasisTransform(2),
    }
    right_identity_maps = {
        (right_entry.child_sectors, right_entry.selected_shape, 0):
            IdentityBasisTransform(3),
    }
    projection = ReducedProjectedSector(
        sector="mid",
        left_pipe=left_pipe,
        right_pipe=right_pipe,
        left_basis_map=identity_maps,
        right_basis_map=right_identity_maps,
        blocks={
            (
                (left_entry.child_sectors, left_entry.selected_shape, 0),
                (right_entry.child_sectors, right_entry.selected_shape, 0),
            ): matrix,
        },
        dtype=float,
    )
    left_metric = np.asarray([[1.7, 0.2], [0.2, 1.1]])
    right_metric = np.asarray(
        [[1.2, -0.1, 0.0], [-0.1, 1.6, 0.3], [0.0, 0.3, 1.4]]
    )

    result = _factor_metric_weighted_projected_svd(
        projection,
        left_factors={("left", "left"): left_metric},
        right_factors={("right", "right"): right_metric},
    )

    np.testing.assert_allclose(
        result.U @ np.diag(result.singular_values) @ result.Vh,
        matrix,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        np.sum(result.singular_values**2),
        np.trace(matrix.T @ left_metric @ matrix @ right_metric.T),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(result.U.T @ left_metric @ result.U, np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(
        result.Vh @ right_metric.T @ result.Vh.T,
        np.eye(2),
        atol=1.0e-12,
    )


def test_generalized_davidson_retains_overlap_after_imperfect_orthogonalization(monkeypatch):
    orthogonalize = solver_module._metric_orthonormalize_packed_columns
    def imperfect(*args, **kwargs):
        vectors, images = orthogonalize(*args, **kwargs)
        return vectors*1.0001, images*1.0001
    monkeypatch.setattr(solver_module, '_metric_orthonormalize_packed_columns', imperfect)
    h = np.array([[-2., .3], [.3, -1.]])
    theta, vector, _ = solver_module._solve_packed_generalized_davidson(
        np.array([1., .4]), lambda x: h@x, h_diag=np.diag(h),
        N=lambda x: x, n_diag=np.ones(2), tol=1e-11, tol_residual=1e-10,
        max_space=2, itermax=20)
    assert abs(theta-np.linalg.eigvalsh(h)[0]) < 1e-10
    assert np.linalg.norm(h@vector-theta*vector)/np.linalg.norm(vector) < 1e-10


def test_generalized_davidson_checks_actual_returned_vector(monkeypatch):
    canonicalize = solver_module._canonicalize_eigenvector

    def perturbed(vector, **kwargs):
        result = np.array(canonicalize(vector, **kwargs), copy=True)
        result[-1] += 1e-5
        return result

    monkeypatch.setattr(solver_module, '_canonicalize_eigenvector', perturbed)
    h = np.array([[-2., .3], [.3, -1.]])
    theta, vector, info = solver_module._solve_packed_generalized_davidson(
        np.array([1., .4]), lambda x: h @ x, h_diag=np.diag(h),
        N=lambda x: x, n_diag=np.ones(2), tol=1e-11,
        tol_residual=1e-10, max_space=2, itermax=20,
    )
    residual = np.linalg.norm(h @ vector - theta * vector)
    assert residual > 1e-6
    assert info['residual'] == pytest.approx(residual, rel=1e-12)
    assert not info['davidson_converged']


def test_metric_orthogonalization_refreshes_cancelled_action():
    basis = np.array([[1.], [0.], [0.]])
    cached_images = np.array([[1.], [1e-7], [0.]])
    vector = np.array([1., 1e-4, 1e-4])
    result, image = solver_module._metric_orthogonalize_packed_vector(
        vector, vector.copy(), basis, cached_images,
        tol=1e-12, metric_operator=lambda x: x,
    )
    np.testing.assert_allclose(image, result, atol=1e-15)
    np.testing.assert_allclose(np.vdot(result, image), 1., atol=1e-15)
    np.testing.assert_allclose(basis.T @ result, 0., atol=1e-15)


def test_generalized_davidson_measures_both_projected_triangles():
    from scipy.linalg import eigh

    h = np.array([[-2., .300001], [.3, -1.]])
    n = np.diag([1., 2.])
    theta, _, info = solver_module._solve_packed_generalized_davidson(
        np.array([1., 0.]), lambda x: h @ x, h_diag=np.diag(h),
        N=lambda x: n @ x, n_diag=np.diag(n), tol=1e-12,
        tol_residual=1e-12, max_space=2, itermax=4,
    )
    expected = eigh((h+h.T)/2, n, eigvals_only=True)[0]
    assert theta == pytest.approx(expected, abs=1e-12)
    assert not info['davidson_converged']
