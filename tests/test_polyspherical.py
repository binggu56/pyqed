from functools import reduce

import numpy as np
import pytest

from pyqed.dvr.dvr_1d import ExponentialDVR, SineDVR
from pyqed.mps.decompose import decompose
from pyqed.mps.mps import _mpo_to_dense_operator
from pyqed.namd.polyspherical import (
    _boundary_complete_metric_derivative,
    PolysphericalTree,
    analytic_keo_terms,
    build_analytic_keo_mpo,
    build_keo_mpo,
    build_keo_mpo_cross,
    metric_keo_mpo,
    metric_tt_keo_components,
    metric_tt_keo_mpo,
    sample_analytic_metric,
)
from pyqed.namd.triatomic import Triatom


class _DVR:
    def __init__(self, momentum):
        self._momentum = np.asarray(momentum, dtype=complex)
        self.npts = self._momentum.shape[0]

    def momentum(self):
        return self._momentum


@pytest.mark.parametrize("npts", (5, 6, 7))
def test_boundary_complete_metric_derivative_recovers_sine_kinetic(npts):
    dvr = SineDVR(-0.8, 1.1, npts)
    derivative = _boundary_complete_metric_derivative(dvr)

    np.testing.assert_allclose(
        0.5 * derivative.conj().T @ derivative,
        dvr.t(),
        atol=2.0e-12,
    )
    assert np.linalg.svd(derivative, compute_uv=False)[-1] > 1.0e-8

    metric = np.diag(1.0 + 0.2 * dvr.x)
    variable_kinetic = 0.5 * derivative.conj().T @ metric @ derivative
    assert np.linalg.eigvalsh(variable_kinetic)[0] > 0.0


def test_boundary_complete_metric_derivative_preserves_periodic_kinetic():
    dvr = ExponentialDVR(npts=9, L=2.0 * np.pi)
    derivative = _boundary_complete_metric_derivative(dvr)

    np.testing.assert_allclose(
        0.5 * derivative.conj().T @ derivative,
        dvr.t(),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(derivative, dvr.momentum(), atol=2.0e-8)


def test_tetraatomic_tree_recovers_jacobi_vectors_and_center_of_mass():
    masses = np.array([1.0, 2.0, 3.0, 4.0])
    tree = PolysphericalTree(((0, 1), (2, 3)), masses)
    coordinates = np.array([5.0, 1.2, 1.1, 1.5, 0.8, -0.4])

    geometry = tree.cartesian(coordinates)
    expected_vectors = tree.jacobi_vectors(coordinates)

    np.testing.assert_allclose(
        masses @ geometry / masses.sum(), np.zeros(3), atol=1.0e-14
    )
    np.testing.assert_allclose(
        tree.vectors_from_cartesian(geometry), expected_vectors, atol=1.0e-14
    )
    assert tree.coordinate_labels == (
        "r0", "r1", "theta1", "r2", "theta2", "phi2"
    )
    np.testing.assert_allclose(tree.reduced_masses, [2.1, 2.0 / 3.0, 12.0 / 7.0])


def test_tree_rejects_missing_or_duplicate_atoms():
    masses = [1.0, 1.0, 1.0]
    for invalid in (((0, 1), 1), (0, 1)):
        try:
            PolysphericalTree(invalid, masses)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid Jacobi tree was accepted")


def test_metric_keo_mpo_matches_dense_derivative_sandwich():
    rng = np.random.default_rng(17)
    dims = (3, 2, 2)
    momenta = []
    for dim in dims:
        raw = rng.normal(size=(dim, dim)) + 1j * rng.normal(size=(dim, dim))
        momenta.append(0.5 * (raw + raw.conj().T))
    dvrs = [_DVR(momentum) for momentum in momenta]

    grid = np.meshgrid(
        np.linspace(-0.4, 0.5, dims[0]),
        np.linspace(-0.2, 0.3, dims[1]),
        np.linspace(0.1, 0.6, dims[2]),
        indexing="ij",
    )
    metric = np.zeros((*dims, 3, 3))
    metric[..., 0, 0] = 1.0 + 0.1 * grid[0]
    metric[..., 1, 1] = 1.2 + 0.2 * grid[1]
    metric[..., 2, 2] = 0.9 + 0.1 * grid[2]
    metric[..., 0, 1] = metric[..., 1, 0] = 0.03 * grid[0] * grid[1]
    metric[..., 1, 2] = metric[..., 2, 1] = 0.02 * grid[1] * grid[2]
    pseudopotential = 0.04 * grid[0] * grid[2]

    mpo = metric_keo_mpo(
        dvrs, metric, pseudopotential, field_rtol=0.0
    )
    dense = _mpo_to_dense_operator(mpo)

    identities = [np.eye(dim) for dim in dims]
    derivative = []
    for axis in range(3):
        operators = [
            momenta[site] if site == axis else identities[site]
            for site in range(3)
        ]
        derivative.append(reduce(np.kron, operators))
    reference = np.diag(pseudopotential.reshape(-1).astype(complex))
    for first in range(3):
        for second in range(3):
            coefficient = np.diag(metric[..., first, second].reshape(-1))
            reference += 0.5 * (
                derivative[first].conj().T @ coefficient @ derivative[second]
            )

    np.testing.assert_allclose(dense, reference, atol=2.0e-12)


def test_metric_tt_keo_components_reconstruct_the_full_operator():
    momenta = (
        np.asarray([[0.2, 0.3j], [-0.3j, -0.1]]),
        np.asarray(
            [[0.1, 0.2j, 0.0], [-0.2j, 0.0, 0.15j], [0.0, -0.15j, -0.1]]
        ),
    )
    dvrs = tuple(_DVR(momentum) for momentum in momenta)
    shape = tuple(dvr.npts for dvr in dvrs)
    mesh = np.meshgrid(
        np.linspace(-0.3, 0.4, shape[0]),
        np.linspace(-0.2, 0.5, shape[1]),
        indexing="ij",
    )
    fields = {
        (0, 0): 1.1 + 0.1 * mesh[0],
        (0, 1): 0.04 * mesh[0] * mesh[1],
        (1, 1): 0.8 + 0.2 * mesh[1],
    }
    metric_cores = {
        label: decompose(values, rank=4) for label, values in fields.items()
    }
    pseudo = decompose(0.03 * mesh[0] - 0.02 * mesh[1], rank=4)

    components = metric_tt_keo_components(dvrs, metric_cores, pseudo)
    reconstructed = sum(
        (_mpo_to_dense_operator(component) for _active, component in components),
        np.zeros((np.prod(shape), np.prod(shape)), dtype=complex),
    )
    full = _mpo_to_dense_operator(
        metric_tt_keo_mpo(dvrs, metric_cores, pseudo)
    )

    assert [active for active, _component in components] == [
        (0,), (0, 1), (1,), ()
    ]
    np.testing.assert_allclose(reconstructed, full, atol=2.0e-12)
    np.testing.assert_allclose(full, full.conj().T, atol=2.0e-12)


def test_polyspherical_mpo_matches_jax_dense_keo():
    pytest.importorskip("jax")
    from pyqed.namd.keo import calculate_exact_keo

    masses = np.array([1.0, 16.0, 1.0])
    tree = PolysphericalTree(((0, 1), 2), masses)
    dvrs = [
        SineDVR(2.0, 3.0, 2),
        SineDVR(1.4, 2.2, 2),
        SineDVR(0.7, 1.5, 2),
    ]

    mpo = build_keo_mpo(tree, dvrs, field_rtol=0.0)
    dense = _mpo_to_dense_operator(mpo)
    reference = calculate_exact_keo(
        dvrs,
        masses,
        tree.jax_map(),
        mode="vib",
        verbose=False,
    )

    np.testing.assert_allclose(dense, reference, atol=1.0e-12)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-12)


def test_jacobi_metric_and_pseudopotential_match_closed_form():
    pytest.importorskip("jax")
    from pyqed.namd.polyspherical import sample_metric

    masses = np.array([1.0, 16.0, 1.0])
    tree = PolysphericalTree(((1, 2), 0), masses)
    dvrs = [
        SineDVR(2.0, 3.0, 3),
        SineDVR(1.4, 2.2, 3),
        SineDVR(0.7, 1.5, 3),
    ]
    metric, pseudopotential = sample_metric(
        dvrs, masses, tree.jax_map()
    )
    radial_R, radial_r, gamma = np.meshgrid(
        *(dvr.x for dvr in dvrs), indexing="ij"
    )
    mu_r = masses[1] * masses[2] / (masses[1] + masses[2])
    mu_R = masses[0] * (masses[1] + masses[2]) / masses.sum()
    angular = 1.0 / (mu_R * radial_R**2) + 1.0 / (mu_r * radial_r**2)
    expected_metric = np.zeros_like(metric)
    expected_metric[..., 0, 0] = 1.0 / mu_R
    expected_metric[..., 1, 1] = 1.0 / mu_r
    expected_metric[..., 2, 2] = angular
    expected_pseudopotential = (
        -0.125 * angular * (1.0 + 1.0 / np.sin(gamma) ** 2)
    )

    np.testing.assert_allclose(metric, expected_metric, atol=5.0e-14)
    np.testing.assert_allclose(
        pseudopotential, expected_pseudopotential, atol=5.0e-14
    )


@pytest.mark.parametrize("npts", (2, 3, 4, 5))
def test_analytic_jacobi_sop_matches_ad_generated_keo(npts):
    pytest.importorskip("jax")
    masses = np.array([1.0, 16.0, 1.0])
    tree = PolysphericalTree(((1, 2), 0), masses)
    dvrs = [
        SineDVR(2.0, 3.0, npts),
        SineDVR(1.4, 2.2, npts),
        SineDVR(0.7, 1.5, npts),
    ]

    analytical = _mpo_to_dense_operator(build_analytic_keo_mpo(tree, dvrs))
    generated = _mpo_to_dense_operator(
        build_keo_mpo(tree, dvrs, field_rtol=0.0, method="ad")
    )
    dispatched = _mpo_to_dense_operator(
        build_keo_mpo(tree, dvrs, method="analytic")
    )

    np.testing.assert_allclose(analytical, generated, atol=2.0e-11)
    np.testing.assert_allclose(dispatched, analytical, atol=2.0e-12)
    assert len(analytic_keo_terms(tree, dvrs)) == 6


def test_analytic_diatomic_sop_is_reduced_mass_radial_kinetic():
    masses = np.array([1.0, 2.0])
    tree = PolysphericalTree((0, 1), masses)
    dvr = SineDVR(0.8, 4.0, 7)

    analytical = _mpo_to_dense_operator(
        build_analytic_keo_mpo(tree, [dvr])
    )
    expected = dvr.t() * dvr.mass / tree.reduced_masses[0]

    np.testing.assert_allclose(analytical, expected, atol=2.0e-13)


def test_analytic_tetraatomic_sop_matches_ad_generated_keo():
    pytest.importorskip("jax")
    masses = np.array([1.0, 2.0, 3.0, 4.0])
    tree = PolysphericalTree(((0, 1), (2, 3)), masses)
    domains = {
        "radial": (1.2, 2.1),
        "theta": (0.5, 1.4),
        "phi": (-1.0, 0.7),
    }
    dvrs = [
        SineDVR(*domains[
            "radial" if label.startswith("r")
            else "theta" if label.startswith("theta")
            else "phi"
        ], 2)
        for label in tree.coordinate_labels
    ]

    analytical = _mpo_to_dense_operator(
        build_keo_mpo(tree, dvrs, method="analytic")
    )
    generated = _mpo_to_dense_operator(
        build_keo_mpo(tree, dvrs, method="ad", field_rtol=0.0)
    )

    np.testing.assert_allclose(analytical, generated, atol=1.0e-11)
    np.testing.assert_allclose(
        analytical, analytical.conj().T, atol=2.0e-12
    )
    assert len(analytic_keo_terms(tree, dvrs)) == 27


def test_analytic_pentaatomic_fields_match_ad_fields():
    pytest.importorskip("jax")
    from pyqed.namd.polyspherical import sample_metric

    masses = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    tree = PolysphericalTree((((0, 1), 2), (3, 4)), masses)
    domains = {
        "radial": (1.1, 2.0),
        "theta": (0.45, 1.35),
        "phi": (-0.8, 0.7),
    }
    dvrs = [
        SineDVR(*domains[
            "radial" if label.startswith("r")
            else "theta" if label.startswith("theta")
            else "phi"
        ], 2)
        for label in tree.coordinate_labels
    ]

    analytical_metric, analytical_pseudo = sample_analytic_metric(tree, dvrs)
    generated_metric, generated_pseudo = sample_metric(
        dvrs, masses, tree.jax_map()
    )

    np.testing.assert_allclose(
        analytical_metric, generated_metric, atol=5.0e-13
    )
    np.testing.assert_allclose(
        analytical_pseudo, generated_pseudo, atol=5.0e-13
    )


def test_default_diatomic_builder_uses_nonsingular_analytic_metric():
    masses = np.array([1.0, 2.0])
    tree = PolysphericalTree((0, 1), masses)
    dvr = SineDVR(0.8, 4.0, 5)

    default = _mpo_to_dense_operator(build_keo_mpo(tree, [dvr]))
    analytical = _mpo_to_dense_operator(
        build_keo_mpo(tree, [dvr], method="analytic")
    )

    np.testing.assert_allclose(default, analytical, atol=2.0e-13)


def test_triatom_jacobi_keo_matches_polyspherical_mpo_after_reordering():
    pytest.importorskip("jax")
    masses = np.array([1.0, 16.0, 1.0])
    radial_R = SineDVR(2.0, 3.0, 3)
    radial_r = SineDVR(1.4, 2.2, 3)
    gamma = SineDVR(0.7, 1.5, 3)

    triatom = Triatom.__new__(Triatom)
    triatom.mass = masses
    triatom.J = 0
    triatom.dvrs = [radial_r, radial_R, gamma]
    triatom.x = [dvr.x for dvr in triatom.dvrs]
    triatom.nx = [dvr.npts for dvr in triatom.dvrs]
    analytic = triatom._buildK_jacobi_h_oh()
    nr, nR, ng = triatom.nx
    analytic_Rrg = analytic.reshape(
        nr, nR, ng, nr, nR, ng
    ).transpose(1, 0, 2, 4, 3, 5).reshape(nR * nr * ng, -1)

    tree = PolysphericalTree(((1, 2), 0), masses)
    mpo = build_keo_mpo(
        tree, [radial_R, radial_r, gamma], field_rtol=0.0
    )
    polyspherical = _mpo_to_dense_operator(mpo)

    np.testing.assert_allclose(polyspherical, analytic_Rrg, atol=2.0e-12)
    assert np.linalg.eigvalsh(polyspherical)[0] > 0.0


@pytest.mark.parametrize("npts", range(2, 8))
def test_jacobi_sine_weak_form_is_positive_for_odd_and_even_grids(npts):
    masses = np.array([1.0, 16.0, 1.0])
    triatom = Triatom.__new__(Triatom)
    triatom.mass = masses
    triatom.J = 0
    triatom.dvrs = [
        SineDVR(1.4, 2.2, npts),
        SineDVR(2.0, 3.0, npts),
        SineDVR(0.7, 1.5, npts),
    ]
    triatom.x = [dvr.x for dvr in triatom.dvrs]
    triatom.nx = [dvr.npts for dvr in triatom.dvrs]

    kinetic = triatom._buildK_jacobi_h_oh()
    sparse_kinetic = triatom._buildK_jacobi_h_oh(sparse=True)

    np.testing.assert_allclose(kinetic, kinetic.conj().T, atol=2.0e-12)
    np.testing.assert_allclose(sparse_kinetic.toarray(), kinetic, atol=2.0e-12)
    assert np.linalg.eigvalsh(kinetic)[0] > 0.0


def test_tt_cross_jacobi_keo_matches_full_grid_and_compresses_hermitian():
    pytest.importorskip("jax")
    masses = np.array([1.0, 16.0, 1.0])
    tree = PolysphericalTree(((1, 2), 0), masses)
    dvrs = [
        SineDVR(2.0, 3.0, 4),
        SineDVR(1.4, 2.2, 4),
        SineDVR(0.7, 1.5, 4),
    ]
    reference = _mpo_to_dense_operator(
        build_keo_mpo(tree, dvrs, field_rtol=0.0)
    )
    sampled, info = build_keo_mpo_cross(
        tree,
        dvrs,
        cross_max_rank=8,
        cross_sweeps=3,
        cross_rtol=1.0e-11,
        cross_validation=64,
        return_info=True,
    )
    sampled_dense = _mpo_to_dense_operator(sampled)

    np.testing.assert_allclose(sampled_dense, reference, atol=2.0e-11)
    assert info["point_samples"] <= info["grid_size"]

    compressed = sampled.compress_hermitian(12)
    compressed_dense = _mpo_to_dense_operator(compressed)
    relative_error = np.linalg.norm(compressed_dense - reference) / np.linalg.norm(
        reference
    )
    np.testing.assert_allclose(
        compressed_dense, compressed_dense.conj().T, atol=2.0e-12
    )
    assert max(compressed.bond_orders()) <= 12
    assert relative_error < 1.0e-2
