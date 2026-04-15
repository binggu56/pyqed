import numpy as np
import pytest

from pyqed.dvr import (
    AnisotropicGaussianWavepacketFBR,
    ComplexGaussianWavepacketFBR,
    GaussianWavepacketFBR,
)


def test_gaussian_wavepacket_fbr_builds_symmetric_overlap_and_coordinates():
    fbr = GaussianWavepacketFBR(
        centers=[[-1.0, 0.5], [0.0, -0.25], [1.2, 1.0]],
        widths=[[0.8, 1.1], [1.0, 0.9], [1.3, 0.7]],
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr.overlap, fbr.overlap.T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(fbr.overlap)) > 0.0

    for op in fbr.coordinate_ops:
        np.testing.assert_allclose(op, op.T, atol=1e-12)


def test_gaussian_wavepacket_fbr_to_sddvr_generalizes_to_any_dimension():
    fbr = GaussianWavepacketFBR(
        centers=[
            [-1.0, 0.2, 0.0],
            [0.0, -0.3, 0.5],
            [1.0, 0.8, -0.5],
            [0.4, -0.7, 1.2],
        ],
        widths=0.9,
        labels=['x', 'y', 'z'],
    )

    sd = fbr.to_sddvr()
    orth_ops = fbr.orthonormal_coordinate_ops()
    before = sum(np.linalg.norm(op - np.diag(np.diag(op))) for op in orth_ops)
    after = sd.diagonal_error()

    assert sd.grid.shape == (fbr.nbasis, fbr.ndim)
    assert sd.labels == ['x', 'y', 'z']
    assert np.isfinite(after)
    assert after < before

    for idx in range(fbr.ndim):
        before_i = np.linalg.norm(orth_ops[idx] - np.diag(np.diag(orth_ops[idx])))
        after_i = np.linalg.norm(
            sd.coordinate(idx) - np.diag(np.diag(sd.coordinate(idx)))
        )
        assert after_i <= before_i


def test_random_ho_sampling_is_reproducible_and_well_conditioned():
    fbr1 = GaussianWavepacketFBR.random_ho(
        nbasis=8,
        omega=[1.0, 1.5],
        seed=7,
        overlap_cutoff=0.85,
        labels=['x', 'y'],
    )
    fbr2 = GaussianWavepacketFBR.random_ho(
        nbasis=8,
        omega=[1.0, 1.5],
        seed=7,
        overlap_cutoff=0.85,
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr1.centers, fbr2.centers)
    np.testing.assert_allclose(fbr1.widths, fbr2.widths)
    assert fbr1.nbasis == 8
    assert fbr1.ndim == 2
    assert np.min(np.linalg.eigvalsh(fbr1.overlap)) > fbr1.s_thresh

    mask = ~np.eye(fbr1.nbasis, dtype=bool)
    assert np.max(np.abs(fbr1.overlap[mask])) < 0.85 + 1e-12


def test_random_ho_basis_gives_reasonable_2d_oscillator_spectrum():
    wx, wy = 1.0, 1.5
    fbr = GaussianWavepacketFBR.random_ho(
        nbasis=20,
        omega=[wx, wy],
        seed=11,
        overlap_cutoff=0.8,
        center_scale=1.2,
        labels=['x', 'y'],
    )

    t = fbr.orthonormal_kinetic(mass=[1.0, 1.0])
    qx, qy = fbr.orthonormal_coordinate_ops()
    v = 0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy))
    e, _ = np.linalg.eigh(t + v)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    np.testing.assert_allclose(e[:4], exact[:4], atol=7e-1)


def test_anisotropic_gaussian_wavepacket_fbr_builds_spd_overlap():
    widths = np.array([
        [[1.0, 0.2], [0.2, 1.4]],
        [[0.8, -0.1], [-0.1, 1.2]],
        [[1.3, 0.15], [0.15, 0.9]],
    ])
    fbr = AnisotropicGaussianWavepacketFBR(
        centers=[[-1.0, 0.0], [0.1, -0.4], [1.2, 0.8]],
        width_mats=widths,
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr.overlap, fbr.overlap.conj().T, atol=1e-12)
    assert np.min(np.linalg.eigvalsh(fbr.overlap)) > 0.0
    for op in fbr.coordinate_ops:
        np.testing.assert_allclose(op, op.conj().T, atol=1e-12)


def test_complex_gwp_zero_momentum_reduces_to_real_anisotropic_case():
    widths = np.array([
        [[1.0, 0.0], [0.0, 1.5]],
        [[0.9, 0.1], [0.1, 1.2]],
    ])
    centers = [[-0.5, 0.0], [0.8, -0.3]]

    real_fbr = AnisotropicGaussianWavepacketFBR(centers=centers, width_mats=widths)
    complex_fbr = ComplexGaussianWavepacketFBR(
        centers=centers,
        width_mats=widths,
        momenta=[[0.0, 0.0], [0.0, 0.0]],
    )

    np.testing.assert_allclose(complex_fbr.overlap.real, real_fbr.overlap.real, atol=1e-12)
    np.testing.assert_allclose(complex_fbr.coordinate_ops.real, real_fbr.coordinate_ops.real, atol=1e-12)
    np.testing.assert_allclose(complex_fbr.kinetic().real, real_fbr.kinetic().real, atol=1e-12)


def test_complex_gwp_builds_hermitian_operators_and_blocks_sddvr():
    fbr = ComplexGaussianWavepacketFBR(
        centers=[[-0.5, 0.0], [0.7, -0.4], [1.1, 0.6]],
        width_mats=[
            [[1.0, 0.1], [0.1, 1.4]],
            [[0.9, -0.05], [-0.05, 1.1]],
            [[1.2, 0.08], [0.08, 0.95]],
        ],
        momenta=[[0.2, 0.0], [-0.1, 0.15], [0.05, -0.25]],
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr.overlap, fbr.overlap.conj().T, atol=1e-12)
    for op in fbr.coordinate_ops:
        np.testing.assert_allclose(op, op.conj().T, atol=1e-12)
    np.testing.assert_allclose(fbr.kinetic(), fbr.kinetic().conj().T, atol=1e-12)

    with pytest.raises(NotImplementedError):
        fbr.to_sddvr()


def test_complex_gwp_importance_sampled_ho_is_reproducible():
    fbr1 = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=10,
        omega=[1.0, 1.5],
        seed=17,
        overlap_cutoff=0.8,
        center_scale=1.0,
        momentum_scale=0.02,
        labels=['x', 'y'],
    )
    fbr2 = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=10,
        omega=[1.0, 1.5],
        seed=17,
        overlap_cutoff=0.8,
        center_scale=1.0,
        momentum_scale=0.02,
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr1.centers, fbr2.centers)
    np.testing.assert_allclose(fbr1.momenta, fbr2.momenta)
    np.testing.assert_allclose(fbr1.width_mats, fbr2.width_mats)
    assert np.min(np.linalg.eigvalsh(fbr1.overlap)) > fbr1.s_thresh
    assert np.max(np.abs(fbr1.momenta)) > 0.0


def test_complex_gwp_importance_sampled_ho_supports_sobol_sampling():
    fbr1 = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=10,
        omega=[1.0, 1.5],
        seed=17,
        overlap_cutoff=0.8,
        center_scale=1.0,
        momentum_scale=0.02,
        sampling='sobol',
        labels=['x', 'y'],
    )
    fbr2 = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=10,
        omega=[1.0, 1.5],
        seed=17,
        overlap_cutoff=0.8,
        center_scale=1.0,
        momentum_scale=0.02,
        sampling='sobol',
        labels=['x', 'y'],
    )

    np.testing.assert_allclose(fbr1.centers, fbr2.centers)
    np.testing.assert_allclose(fbr1.momenta, fbr2.momenta)
    assert np.max(np.abs(fbr1.momenta)) > 0.0
    assert np.min(np.linalg.eigvalsh(fbr1.overlap)) > fbr1.s_thresh


def test_complex_gwp_importance_sampled_ho_gives_reasonable_2d_oscillator_spectrum():
    wx, wy = 1.0, 1.5
    fbr = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=20,
        omega=[wx, wy],
        seed=17,
        overlap_cutoff=0.8,
        center_scale=1.0,
        momentum_scale=0.02,
        labels=['x', 'y'],
    )

    t = fbr.orthonormal_kinetic(mass=[1.0, 1.0])
    qx, qy = fbr.orthonormal_coordinate_ops()
    v = 0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy))
    e, _ = np.linalg.eigh(t + v)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    assert np.max(np.abs(np.stack((qx, qy)).imag)) > 1e-3
    np.testing.assert_allclose(e[:6], exact[:6], atol=3.7e-1)


def test_complex_gwp_sobol_sampling_gives_reasonable_4d_oscillator_spectrum():
    omegas = np.array([1.0, 1.2, 0.8, 1.5])
    fbr = ComplexGaussianWavepacketFBR.importance_sampled_ho(
        nbasis=32,
        omega=omegas,
        seed=17,
        overlap_cutoff=0.75,
        center_scale=1.0,
        momentum_scale=0.01,
        sampling='sobol',
        labels=['q1', 'q2', 'q3', 'q4'],
    )

    t = fbr.orthonormal_kinetic(mass=np.ones(4))
    q = fbr.orthonormal_coordinate_ops()
    v = sum(0.5 * omegas[i] ** 2 * (q[i] @ q[i]) for i in range(4))
    e, _ = np.linalg.eigh(t + v)

    exact = []
    for n0 in range(4):
        for n1 in range(4):
            for n2 in range(4):
                for n3 in range(4):
                    exact.append(np.dot(np.array([n0, n1, n2, n3]) + 0.5, omegas))
    exact = np.sort(np.array(exact))

    assert np.max(np.abs(q.imag)) > 1e-3
    np.testing.assert_allclose(e[:8], exact[:8], atol=4.5e-1)


def test_complex_gwp_gives_reasonable_2d_oscillator_spectrum():
    wx, wy = 1.0, 1.5
    centers_1d = np.linspace(-2.5, 2.5, 5)
    centers = np.array(
        [[x, y] for x in centers_1d for y in centers_1d],
        dtype=float,
    )
    width_mats = np.tile(np.diag([wx, wy])[None, :, :], (len(centers), 1, 1))
    momenta = np.array([[0.08 * y, -0.08 * x] for x, y in centers], dtype=float)

    fbr = ComplexGaussianWavepacketFBR(
        centers=centers,
        width_mats=width_mats,
        momenta=momenta,
        labels=['x', 'y'],
    )

    t = fbr.orthonormal_kinetic(mass=[1.0, 1.0])
    qx, qy = fbr.orthonormal_coordinate_ops()
    v = 0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy))
    e, _ = np.linalg.eigh(t + v)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    assert np.max(np.abs(np.stack((qx, qy)).imag)) > 1e-3
    np.testing.assert_allclose(e[:6], exact[:6], atol=8e-2)


def test_complex_gwp_supports_diagonal_local_potential_approximation():
    wx, wy = 1.0, 1.5
    centers_1d = np.linspace(-2.5, 2.5, 5)
    centers = np.array(
        [[x, y] for x in centers_1d for y in centers_1d],
        dtype=float,
    )
    width_mats = np.tile(np.diag([wx, wy])[None, :, :], (len(centers), 1, 1))
    momenta = np.array([[0.08 * y, -0.08 * x] for x, y in centers], dtype=float)

    fbr = ComplexGaussianWavepacketFBR(
        centers=centers,
        width_mats=width_mats,
        momenta=momenta,
        labels=['x', 'y'],
    )

    t = fbr.orthonormal_kinetic(mass=[1.0, 1.0])
    v_diag = fbr.diagonal_local_operator(
        lambda x, y: 0.5 * (wx ** 2 * x ** 2 + wy ** 2 * y ** 2)
    )
    e_diag, _ = np.linalg.eigh(t + v_diag)

    qx, qy = fbr.orthonormal_coordinate_ops()
    v_exact = 0.5 * (wx ** 2 * (qx @ qx) + wy ** 2 * (qy @ qy))
    e_exact, _ = np.linalg.eigh(t + v_exact)

    exact = []
    for nx in range(4):
        for ny in range(4):
            exact.append((nx + 0.5) * wx + (ny + 0.5) * wy)
    exact = np.sort(np.array(exact))

    np.testing.assert_allclose(v_diag, v_diag.conj().T, atol=1e-12)
    np.testing.assert_allclose(fbr.diagonal_grid().imag, 0.0, atol=1e-12)
    np.testing.assert_allclose(e_diag[:6], exact[:6], atol=2.8e-1)
    assert np.max(np.abs(e_diag[:6] - exact[:6])) > np.max(np.abs(e_exact[:6] - exact[:6]))
