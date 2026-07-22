import numpy as np
import pytest
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import (
    AnisotropicGaussianWavepacketFBR,
    ComplexGaussianWavepacketFBR,
    GaussianWavepacketFBR,
    PeriodicVonNeumannBasis,
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


def test_periodic_von_neumann_basis_builds_biorthogonal_partner():
    pvb = PeriodicVonNeumannBasis(
        n_position=5,
        n_momentum=5,
        length=12.0,
    )

    metric = pvb.dx * pvb.biorthogonal_values.conj().T @ pvb.values

    assert pvb.values.shape == (25, 25)
    assert pvb.biorthogonal_values.shape == (25, 25)
    assert np.linalg.cond(pvb.overlap) < 20.0
    np.testing.assert_allclose(metric, np.eye(pvb.nbasis), atol=1e-11)


def test_periodic_von_neumann_biorthogonal_exchange_reconstructs_grid_state():
    pvb = PeriodicVonNeumannBasis(
        n_position=5,
        n_momentum=5,
        length=10.0,
    )
    rng = np.random.default_rng(19)
    psi = rng.normal(size=pvb.nbasis) + 1j * rng.normal(size=pvb.nbasis)
    psi /= np.sqrt(pvb.dx * np.vdot(psi, psi).real)

    exchanged = pvb.biorthogonal_exchange(psi)
    pvn_coeff = pvb.pvn_coefficients(psi)
    bvn_coeff = pvb.biorthogonal_coefficients(psi)

    np.testing.assert_allclose(exchanged, psi, atol=1e-11)
    np.testing.assert_allclose(
        pvb.reconstruct_from_biorthogonal_coefficients(bvn_coeff),
        psi,
        atol=1e-11,
    )
    assert np.linalg.norm(pvn_coeff) > 0.0


def _smooth_two_state_vibronic_model(x, alpha=0.03, kappa=0.03, delta=0.35):
    x = np.asarray(x, dtype=float)
    scalar = 0.5 * alpha * x**2
    z = kappa * x
    rho = np.sqrt(z**2 + delta**2)
    theta = 0.5 * np.arctan2(delta, z)
    c = np.cos(theta)
    s = np.sin(theta)

    vectors = np.empty((x.size, 2, 2))
    vectors[:, :, 0] = np.stack((-s, c), axis=-1)
    vectors[:, :, 1] = np.stack((c, s), axis=-1)
    energies = np.stack((scalar - rho, scalar + rho), axis=-1)

    diabatic_h = np.zeros((x.size, 2, 2))
    diabatic_h[:, 0, 0] = scalar + z
    diabatic_h[:, 1, 1] = scalar - z
    diabatic_h[:, 0, 1] = delta
    diabatic_h[:, 1, 0] = delta
    return diabatic_h, energies, vectors


def _pvb_ldr_spectrum_errors(n_position):
    matrices = _build_pvb_ldr_matrices(n_position)
    coord_eval = np.sort(np.linalg.eigvalsh(matrices["h_coord"]).real)
    exact_eval = np.sort(np.linalg.eigvals(matrices["h_exact"]).real)
    diag_eval = np.sort(np.linalg.eigvals(matrices["h_diag"]).real)
    return {
        "exact": np.max(np.abs(exact_eval[:6] - coord_eval[:6])),
        "diagonal": np.max(np.abs(diag_eval[:6] - coord_eval[:6])),
    }


def _build_pvb_ldr_matrices(n_position):
    pvb = PeriodicVonNeumannBasis(
        n_position=n_position,
        n_momentum=5,
        length=14.0,
        x_min=-7.0,
    )
    ngrid = pvb.nbasis
    nstates = 2
    mass = 8.0
    frame_positions = np.repeat(pvb.positions, pvb.n_momentum)

    t_grid = pvb.kinetic_grid_operator(mass=mass)
    t_gb = pvb.kinetic_operator(mass=mass)
    h_grid_points, _, grid_vectors = _smooth_two_state_vibronic_model(pvb.grid)
    _, frame_energies, frame_vectors = _smooth_two_state_vibronic_model(frame_positions)
    electronic_overlap = np.einsum("ida,jdb->iajb", frame_vectors, frame_vectors)

    kinetic_part = t_gb[:, None, :, None] * electronic_overlap
    diagonal_v = np.zeros_like(kinetic_part)
    h_diag = kinetic_part.copy()
    for i in range(ngrid):
        for a in range(nstates):
            h_diag[i, a, i, a] += frame_energies[i, a]
            diagonal_v[i, a, i, a] = frame_energies[i, a]

    v_exact = np.zeros_like(h_diag)
    for i in range(ngrid):
        for j in range(ngrid):
            nuclear_weight = (
                pvb.dx
                * pvb.values[:, i].conj()
                * pvb.biorthogonal_values[:, j]
            )
            for a in range(nstates):
                left = frame_vectors[i, :, a]
                for b in range(nstates):
                    right = frame_vectors[j, :, b]
                    electronic_matrix = np.einsum(
                        "d,xde,e->x",
                        left,
                        h_grid_points,
                        right,
                        optimize=True,
                    )
                    v_exact[i, a, j, b] = np.dot(nuclear_weight, electronic_matrix)

    h_exact = kinetic_part + v_exact
    h_dvr_transform = np.kron(t_gb, np.eye(nstates, dtype=complex))
    h_dvr_transform += pvb.local_matrix_operator(h_grid_points).reshape(
        ngrid * nstates,
        ngrid * nstates,
    )
    h_coord = np.kron(t_grid, np.eye(nstates, dtype=complex))
    for xidx in range(ngrid):
        sl = slice(nstates * xidx, nstates * (xidx + 1))
        h_coord[sl, sl] += h_grid_points[xidx]

    nuclear_metric = pvb.dx * (
        pvb.biorthogonal_values.conj().T @ pvb.biorthogonal_values
    )
    diabatic_metric = np.kron(nuclear_metric, np.eye(nstates, dtype=complex))
    metric = (nuclear_metric[:, None, :, None] * electronic_overlap).reshape(
        ngrid * nstates,
        ngrid * nstates,
    )
    kinetic_part = kinetic_part.reshape(ngrid * nstates, ngrid * nstates)
    diagonal_v = diagonal_v.reshape(ngrid * nstates, ngrid * nstates)
    diagonal_v_metric = np.linalg.solve(
        metric,
        0.5 * (metric @ diagonal_v + diagonal_v.conj().T @ metric),
    )
    return {
        "pvb": pvb,
        "grid_vectors": grid_vectors,
        "frame_vectors": frame_vectors,
        "metric": metric,
        "h_coord": h_coord,
        "h_exact": h_exact.reshape(ngrid * nstates, ngrid * nstates),
        "h_dvr_transform": h_dvr_transform,
        "diabatic_metric": diabatic_metric,
        "h_diag": h_diag.reshape(ngrid * nstates, ngrid * nstates),
        "h_metric_diag": kinetic_part + diagonal_v_metric,
    }


def test_periodic_von_neumann_diagonal_potential_approximation_on_vibronic_model():
    coarse = _pvb_ldr_spectrum_errors(n_position=5)
    refined = _pvb_ldr_spectrum_errors(n_position=9)

    assert coarse["exact"] < 1e-10
    assert refined["exact"] < 1e-10
    assert refined["diagonal"] < coarse["diagonal"]
    assert refined["diagonal"] < 3.0e-2


def _initial_vibronic_packet(pvb, grid_vectors, frame_vectors):
    envelope = np.exp(
        -0.5 * ((pvb.grid + 2.4) / 0.75) ** 2
        + 1j * 1.0 * (pvb.grid + 2.4)
    )
    psi = envelope[:, None] * grid_vectors[:, :, 0]
    psi /= np.sqrt(pvb.dx * np.vdot(psi.ravel(), psi.ravel()).real)
    coeff = np.einsum(
        "xi,x,ida,xd->ia",
        pvb.values.conj(),
        np.full(pvb.nbasis, pvb.dx),
        frame_vectors,
        psi,
        optimize=True,
    )
    return psi.ravel(), coeff.ravel()


def _reconstruct_vibronic_packet(pvb, frame_vectors, coeff):
    return np.einsum(
        "xi,ida,ia->xd",
        pvb.biorthogonal_values,
        frame_vectors,
        coeff.reshape(pvb.nbasis, 2),
        optimize=True,
    ).ravel()


def _initial_diabatic_pvb_coefficients(pvb, psi):
    return np.einsum(
        "xi,x,xa->ia",
        pvb.values.conj(),
        np.full(pvb.nbasis, pvb.dx),
        psi.reshape(pvb.nbasis, 2),
        optimize=True,
    ).ravel()


def _reconstruct_diabatic_pvb_packet(pvb, coeff):
    return np.einsum(
        "xi,ia->xa",
        pvb.biorthogonal_values,
        coeff.reshape(pvb.nbasis, 2),
        optimize=True,
    ).ravel()


def test_periodic_von_neumann_diagonal_potential_dynamics_on_vibronic_model():
    matrices = _build_pvb_ldr_matrices(n_position=9)
    pvb = matrices["pvb"]
    psi0, coeff0 = _initial_vibronic_packet(
        pvb,
        matrices["grid_vectors"],
        matrices["frame_vectors"],
    )
    times = np.linspace(0.0, 12.0, 61)

    coord_states = expm_multiply(
        -1j * matrices["h_coord"],
        psi0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )
    exact_coeffs = expm_multiply(
        -1j * matrices["h_exact"],
        coeff0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )
    naive_coeffs = expm_multiply(
        -1j * matrices["h_diag"],
        coeff0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )
    metric_coeffs = expm_multiply(
        -1j * matrices["h_metric_diag"],
        coeff0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )

    exact_final = _reconstruct_vibronic_packet(
        pvb,
        matrices["frame_vectors"],
        exact_coeffs[-1],
    )
    naive_final = _reconstruct_vibronic_packet(
        pvb,
        matrices["frame_vectors"],
        naive_coeffs[-1],
    )
    metric_final = _reconstruct_vibronic_packet(
        pvb,
        matrices["frame_vectors"],
        metric_coeffs[-1],
    )
    coord_final = coord_states[-1]

    exact_error = np.sqrt(
        pvb.dx * np.vdot(exact_final - coord_final, exact_final - coord_final).real
    )
    naive_error = np.sqrt(
        pvb.dx * np.vdot(naive_final - coord_final, naive_final - coord_final).real
    )
    metric_error = np.sqrt(
        pvb.dx * np.vdot(metric_final - coord_final, metric_final - coord_final).real
    )
    naive_norm = pvb.dx * np.vdot(naive_final, naive_final).real
    metric_norm = pvb.dx * np.vdot(metric_final, metric_final).real

    assert exact_error < 1e-10
    assert metric_error < naive_error
    assert abs(metric_norm - 1.0) < 1e-10
    assert naive_norm > 1.5


def test_periodic_von_neumann_dvr_potential_transform_dynamics_on_vibronic_model():
    matrices = _build_pvb_ldr_matrices(n_position=9)
    pvb = matrices["pvb"]
    psi0, _ = _initial_vibronic_packet(
        pvb,
        matrices["grid_vectors"],
        matrices["frame_vectors"],
    )
    coeff0 = _initial_diabatic_pvb_coefficients(pvb, psi0)
    times = np.linspace(0.0, 12.0, 61)

    coord_states = expm_multiply(
        -1j * matrices["h_coord"],
        psi0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )
    pvb_coeffs = expm_multiply(
        -1j * matrices["h_dvr_transform"],
        coeff0,
        start=times[0],
        stop=times[-1],
        num=times.size,
        endpoint=True,
    )
    pvb_final = _reconstruct_diabatic_pvb_packet(pvb, pvb_coeffs[-1])
    final_error = np.sqrt(
        pvb.dx
        * np.vdot(pvb_final - coord_states[-1], pvb_final - coord_states[-1]).real
    )

    metric = matrices["diabatic_metric"]
    metric_residual = np.linalg.norm(
        matrices["h_dvr_transform"].conj().T @ metric
        - metric @ matrices["h_dvr_transform"]
    )
    metric_residual /= np.linalg.norm(metric @ matrices["h_dvr_transform"])
    assert metric_residual < 1e-10
    assert final_error < 1e-10


def test_periodic_von_neumann_high_low_exchange_effective_mode_gaps_compare_exact():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    pvb = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )

    normal_modes = np.linalg.eigvalsh(
        np.array(
            [
                [omega_low, coupling],
                [coupling, omega_high],
            ]
        )
    )
    exact_low_frequency = normal_modes[0]

    scale = 1.0 - coupling**2 / (omega_high * omega_low)
    effective_frequency = omega_low * scale
    effective_h = scale * pvb.kinetic_operator(mass=1.0)
    effective_h += 0.5 * omega_low**2 * scale * pvb.local_operator(pvb.grid**2)
    effective_gaps = np.sort(np.linalg.eigvals(effective_h).real)[:6]
    effective_gaps -= effective_gaps[0]

    expected_effective_gaps = np.arange(6) * effective_frequency
    exact_low_branch_gaps = np.arange(6) * exact_low_frequency

    np.testing.assert_allclose(effective_gaps, expected_effective_gaps, atol=1e-10)
    assert np.max(np.abs(effective_gaps - exact_low_branch_gaps)) < 1.0e-2


def _ladder_operator(nbasis):
    op = np.zeros((nbasis, nbasis), dtype=complex)
    for n in range(1, nbasis):
        op[n - 1, n] = np.sqrt(n)
    return op


def _high_low_exchange_exact_gaps(omega_high, omega_low, coupling, nlevels):
    normal_modes = np.linalg.eigvalsh(
        np.array(
            [
                [omega_low, coupling],
                [coupling, omega_high],
            ],
            dtype=float,
        )
    )
    levels = []
    for n_high in range(nlevels):
        for n_low in range(nlevels):
            levels.append(n_low * normal_modes[0] + n_high * normal_modes[1])
    levels = np.sort(np.asarray(levels))
    return levels[:nlevels] - levels[0]


def _high_low_exchange_pvb_multistate_gaps(
    omega_high,
    omega_low,
    coupling,
    n_high_states,
    nlevels,
):
    pvb = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    nfock = 16
    destroy = _ladder_operator(nfock)
    create = destroy.conj().T
    high_h = omega_high * (create @ destroy)
    high_x = (destroy + create) / np.sqrt(2.0 * omega_high)
    high_p = -1j * np.sqrt(omega_high / 2.0) * (destroy - create)

    q_centers = np.repeat(pvb.positions, pvb.n_momentum)
    p_centers = np.tile(pvb.momenta, pvb.n_position)
    high_states = np.empty((pvb.nbasis, nfock, n_high_states), dtype=complex)
    for i, (q, p) in enumerate(zip(q_centers, p_centers)):
        conditional_h = high_h + coupling * (
            np.sqrt(omega_high * omega_low) * q * high_x
            + p * high_p / np.sqrt(omega_high * omega_low)
        )
        _, vecs = np.linalg.eigh(conditional_h)
        high_states[i] = vecs[:, :n_high_states]

    overlap = np.einsum(
        "iam,jan->imjn",
        high_states.conj(),
        high_states,
        optimize=True,
    )
    x_high = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_x,
        high_states,
        optimize=True,
    )
    p_high = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_p,
        high_states,
        optimize=True,
    )
    h_high = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_h,
        high_states,
        optimize=True,
    )

    kinetic = pvb.kinetic_operator(mass=1.0)
    potential = 0.5 * omega_low**2 * pvb.local_operator(pvb.grid**2)
    position = pvb.local_operator(pvb.grid)
    momentum = pvb.pvb_operator(pvb.momentum_grid_operator())

    h_pvb = (kinetic + potential)[:, None, :, None] * overlap
    h_pvb += np.eye(pvb.nbasis)[:, None, :, None] * h_high
    h_pvb += (
        coupling
        * np.sqrt(omega_high * omega_low)
        * position[:, None, :, None]
        * x_high
    )
    h_pvb += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * momentum[:, None, :, None]
        * p_high
    )

    levels = np.sort(
        np.linalg.eigvals(
            h_pvb.reshape(
                pvb.nbasis * n_high_states,
                pvb.nbasis * n_high_states,
            )
        ).real
    )
    return levels[:nlevels] - levels[0]


def _high_low_exchange_standard_dvr_gaps(
    omega_high,
    omega_low,
    coupling,
    n_high_fock,
    nlevels,
):
    pvb = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    destroy = _ladder_operator(n_high_fock)
    create = destroy.conj().T
    high_h = omega_high * (create @ destroy)
    high_x = (destroy + create) / np.sqrt(2.0 * omega_high)
    high_p = -1j * np.sqrt(omega_high / 2.0) * (destroy - create)

    low_h = pvb.kinetic_grid_operator(mass=1.0)
    low_h += 0.5 * omega_low**2 * np.diag(pvb.grid**2)
    h_dvr = np.kron(low_h, np.eye(n_high_fock, dtype=complex))
    h_dvr += np.kron(np.eye(pvb.nbasis, dtype=complex), high_h)
    h_dvr += (
        coupling
        * np.sqrt(omega_high * omega_low)
        * np.kron(np.diag(pvb.grid), high_x)
    )
    h_dvr += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * np.kron(pvb.momentum_grid_operator(), high_p)
    )

    levels = np.sort(np.linalg.eigvalsh(h_dvr).real)
    return levels[:nlevels] - levels[0]


def _high_low_exchange_coordinate_dvr_ldr_gaps(
    omega_high,
    omega_low,
    coupling,
    n_high_states,
    nlevels,
):
    dvr = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    nfock = 16
    destroy = _ladder_operator(nfock)
    create = destroy.conj().T
    high_h = omega_high * (create @ destroy)
    high_x = (destroy + create) / np.sqrt(2.0 * omega_high)
    high_p = -1j * np.sqrt(omega_high / 2.0) * (destroy - create)

    high_states = np.empty((dvr.nbasis, nfock, n_high_states), dtype=complex)
    high_energies = np.empty((dvr.nbasis, n_high_states))
    for i, x in enumerate(dvr.grid):
        conditional_h = (
            high_h
            + coupling * np.sqrt(omega_high * omega_low) * x * high_x
        )
        eigvals, eigvecs = np.linalg.eigh(conditional_h)
        high_energies[i] = eigvals[:n_high_states]
        high_states[i] = eigvecs[:, :n_high_states]

    overlap = np.einsum(
        "iam,jan->imjn",
        high_states.conj(),
        high_states,
        optimize=True,
    )
    p_high = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_p,
        high_states,
        optimize=True,
    )

    h_dvr_ldr = dvr.kinetic_grid_operator(mass=1.0)[:, None, :, None] * overlap
    h_dvr_ldr += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * dvr.momentum_grid_operator()[:, None, :, None]
        * p_high
    )
    low_potential = 0.5 * omega_low**2 * dvr.grid**2
    for i in range(dvr.nbasis):
        for m in range(n_high_states):
            h_dvr_ldr[i, m, i, m] += low_potential[i] + high_energies[i, m]

    levels = np.sort(
        np.linalg.eigvals(
            h_dvr_ldr.reshape(
                dvr.nbasis * n_high_states,
                dvr.nbasis * n_high_states,
            )
        ).real
    )
    return levels[:nlevels] - levels[0]


def _anharmonic_high_mode_operators(omega_high, n_high_fock, quartic):
    destroy = _ladder_operator(n_high_fock)
    create = destroy.conj().T
    high_x = (destroy + create) / np.sqrt(2.0 * omega_high)
    high_p = -1j * np.sqrt(omega_high / 2.0) * (destroy - create)
    high_h = omega_high * (create @ destroy) + quartic * (
        high_x @ high_x @ high_x @ high_x
    )
    return high_h, high_x, high_p


def _low_phase_space_gaussian_operator(basis, q0, p0, sigma_q, sigma_p):
    q_half = np.diag(np.exp(-0.25 * ((basis.grid - q0) / sigma_q) ** 2))
    p_window = basis._fourier_grid_operator(
        np.exp(-0.5 * ((basis.hbar * basis.wave_numbers - p0) / sigma_p) ** 2)
    )
    operator = q_half @ p_window @ q_half
    return 0.5 * (operator + operator.conj().T)


def _phase_space_gaussian_values(q, p, q0, p0, sigma_q, sigma_p):
    return np.exp(
        -0.5 * ((q - q0) / sigma_q) ** 2
        - 0.5 * ((p - p0) / sigma_p) ** 2
    )


def _high_low_exchange_anharmonic_exact_gaps(
    omega_high,
    omega_low,
    coupling,
    quartic,
    n_high_fock,
    nlevels,
    nonlinear=0.0,
    momentum_nonlinear=0.0,
    phase_space_coupling=None,
):
    dvr = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    high_h, high_x, high_p = _anharmonic_high_mode_operators(
        omega_high,
        n_high_fock,
        quartic,
    )
    low_h = dvr.kinetic_grid_operator(mass=1.0)
    low_h += 0.5 * omega_low**2 * np.diag(dvr.grid**2)
    h_exact = np.kron(low_h, np.eye(n_high_fock, dtype=complex))
    h_exact += np.kron(np.eye(dvr.nbasis, dtype=complex), high_h)
    h_exact += (
        coupling
        * np.sqrt(omega_high * omega_low)
        * np.kron(np.diag(dvr.grid), high_x)
    )
    h_exact += nonlinear * np.kron(np.diag(dvr.grid**2), high_x)
    p_grid = dvr.momentum_grid_operator()
    h_exact += momentum_nonlinear * np.kron(p_grid @ p_grid, high_x)
    if phase_space_coupling is not None:
        kappa, q0, p0, sigma_q, sigma_p = phase_space_coupling
        phase_space_op = _low_phase_space_gaussian_operator(
            dvr,
            q0,
            p0,
            sigma_q,
            sigma_p,
        )
        h_exact += kappa * np.kron(phase_space_op, high_x)
    h_exact += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * np.kron(p_grid, high_p)
    )
    levels = np.sort(np.linalg.eigvalsh(h_exact).real)
    return levels[:nlevels] - levels[0]


def _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
    omega_high,
    omega_low,
    coupling,
    quartic,
    n_high_states,
    n_high_fock,
    nlevels,
    nonlinear=0.0,
    momentum_nonlinear=0.0,
    phase_space_coupling=None,
):
    dvr = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    high_h, high_x, high_p = _anharmonic_high_mode_operators(
        omega_high,
        n_high_fock,
        quartic,
    )
    high_states = np.empty((dvr.nbasis, n_high_fock, n_high_states), dtype=complex)
    for i, x in enumerate(dvr.grid):
        conditional_h = (
            high_h
            + coupling * np.sqrt(omega_high * omega_low) * x * high_x
            + nonlinear * x**2 * high_x
        )
        _, eigvecs = np.linalg.eigh(conditional_h)
        high_states[i] = eigvecs[:, :n_high_states]

    overlap = np.einsum(
        "iam,jan->imjn",
        high_states.conj(),
        high_states,
        optimize=True,
    )
    high_h_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_h,
        high_states,
        optimize=True,
    )
    high_x_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_x,
        high_states,
        optimize=True,
    )
    high_p_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_p,
        high_states,
        optimize=True,
    )
    low_h = dvr.kinetic_grid_operator(mass=1.0)
    low_h += 0.5 * omega_low**2 * np.diag(dvr.grid**2)
    h_ldr = low_h[:, None, :, None] * overlap
    h_ldr += np.eye(dvr.nbasis)[:, None, :, None] * high_h_mat
    h_ldr += (
        coupling
        * np.sqrt(omega_high * omega_low)
        * np.diag(dvr.grid)[:, None, :, None]
        * high_x_mat
    )
    h_ldr += nonlinear * np.diag(dvr.grid**2)[:, None, :, None] * high_x_mat
    p_grid = dvr.momentum_grid_operator()
    h_ldr += momentum_nonlinear * (p_grid @ p_grid)[:, None, :, None] * high_x_mat
    if phase_space_coupling is not None:
        kappa, q0, p0, sigma_q, sigma_p = phase_space_coupling
        phase_space_op = _low_phase_space_gaussian_operator(
            dvr,
            q0,
            p0,
            sigma_q,
            sigma_p,
        )
        h_ldr += kappa * phase_space_op[:, None, :, None] * high_x_mat
    h_ldr += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * p_grid[:, None, :, None]
        * high_p_mat
    )
    levels = np.sort(
        np.linalg.eigvals(
            h_ldr.reshape(
                dvr.nbasis * n_high_states,
                dvr.nbasis * n_high_states,
            )
        ).real
    )
    return levels[:nlevels] - levels[0]


def _high_low_exchange_anharmonic_pvb_ldr_gaps(
    omega_high,
    omega_low,
    coupling,
    quartic,
    n_high_states,
    n_high_fock,
    nlevels,
    nonlinear=0.0,
    momentum_nonlinear=0.0,
    phase_space_coupling=None,
):
    pvb = PeriodicVonNeumannBasis(
        n_position=21,
        n_momentum=5,
        length=34.0,
        x_min=-17.0,
    )
    high_h, high_x, high_p = _anharmonic_high_mode_operators(
        omega_high,
        n_high_fock,
        quartic,
    )
    q_centers = np.repeat(pvb.positions, pvb.n_momentum)
    p_centers = np.tile(pvb.momenta, pvb.n_position)
    high_states = np.empty((pvb.nbasis, n_high_fock, n_high_states), dtype=complex)
    for i, (q, p) in enumerate(zip(q_centers, p_centers)):
        conditional_h = high_h + coupling * (
            np.sqrt(omega_high * omega_low) * q * high_x
            + p * high_p / np.sqrt(omega_high * omega_low)
        )
        conditional_h += nonlinear * q**2 * high_x
        conditional_h += momentum_nonlinear * p**2 * high_x
        if phase_space_coupling is not None:
            kappa, q0, p0, sigma_q, sigma_p = phase_space_coupling
            conditional_h += (
                kappa
                * _phase_space_gaussian_values(q, p, q0, p0, sigma_q, sigma_p)
                * high_x
            )
        _, eigvecs = np.linalg.eigh(conditional_h)
        high_states[i] = eigvecs[:, :n_high_states]

    overlap = np.einsum(
        "iam,jan->imjn",
        high_states.conj(),
        high_states,
        optimize=True,
    )
    high_h_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_h,
        high_states,
        optimize=True,
    )
    high_x_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_x,
        high_states,
        optimize=True,
    )
    high_p_mat = np.einsum(
        "iam,ab,jbn->imjn",
        high_states.conj(),
        high_p,
        high_states,
        optimize=True,
    )
    low_h = pvb.kinetic_operator(mass=1.0)
    low_h += 0.5 * omega_low**2 * pvb.local_operator(pvb.grid**2)
    h_ldr = low_h[:, None, :, None] * overlap
    h_ldr += np.eye(pvb.nbasis)[:, None, :, None] * high_h_mat
    h_ldr += (
        coupling
        * np.sqrt(omega_high * omega_low)
        * pvb.local_operator(pvb.grid)[:, None, :, None]
        * high_x_mat
    )
    h_ldr += (
        nonlinear
        * pvb.local_operator(pvb.grid**2)[:, None, :, None]
        * high_x_mat
    )
    p_grid = pvb.momentum_grid_operator()
    h_ldr += (
        momentum_nonlinear
        * pvb.pvb_operator(p_grid @ p_grid)[:, None, :, None]
        * high_x_mat
    )
    if phase_space_coupling is not None:
        kappa, q0, p0, sigma_q, sigma_p = phase_space_coupling
        phase_space_op = _low_phase_space_gaussian_operator(
            pvb,
            q0,
            p0,
            sigma_q,
            sigma_p,
        )
        h_ldr += (
            kappa
            * pvb.pvb_operator(phase_space_op)[:, None, :, None]
            * high_x_mat
        )
    h_ldr += (
        coupling
        / np.sqrt(omega_high * omega_low)
        * pvb.pvb_operator(p_grid)[:, None, :, None]
        * high_p_mat
    )
    levels = np.sort(
        np.linalg.eigvals(
            h_ldr.reshape(
                pvb.nbasis * n_high_states,
                pvb.nbasis * n_high_states,
            )
        ).real
    )
    return levels[:nlevels] - levels[0]


def test_periodic_von_neumann_high_low_exchange_multistate_branches_improve_spectrum():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    nlevels = 16
    exact = _high_low_exchange_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        nlevels,
    )

    keep1 = _high_low_exchange_pvb_multistate_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=1,
        nlevels=nlevels,
    )
    keep2 = _high_low_exchange_pvb_multistate_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=2,
        nlevels=nlevels,
    )
    keep4 = _high_low_exchange_pvb_multistate_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=4,
        nlevels=nlevels,
    )

    err1 = np.max(np.abs(keep1 - exact))
    err2 = np.max(np.abs(keep2 - exact))
    err4 = np.max(np.abs(keep4 - exact))

    assert err2 < err1
    assert err4 < err2
    assert err4 < 1.0e-5


def test_standard_dvr_high_low_exchange_product_basis_matches_exact_spectrum():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    nlevels = 16
    exact = _high_low_exchange_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        nlevels,
    )
    dvr = _high_low_exchange_standard_dvr_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_fock=8,
        nlevels=nlevels,
    )

    np.testing.assert_allclose(dvr, exact, atol=1e-10)


def test_coordinate_dvr_ldr_high_low_exchange_multistate_branches_match_exact_spectrum():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    nlevels = 16
    exact = _high_low_exchange_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        nlevels,
    )

    keep1 = _high_low_exchange_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=1,
        nlevels=nlevels,
    )
    keep2 = _high_low_exchange_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=2,
        nlevels=nlevels,
    )
    keep4 = _high_low_exchange_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        n_high_states=4,
        nlevels=nlevels,
    )

    err1 = np.max(np.abs(keep1 - exact))
    err2 = np.max(np.abs(keep2 - exact))
    err4 = np.max(np.abs(keep4 - exact))

    assert err2 < err1
    assert err4 < err2
    assert err4 < 2.0e-6


def test_high_low_exchange_with_anharmonic_high_mode_multistate_converges():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    quartic = 0.1
    n_high_fock = 16
    nlevels = 16
    exact = _high_low_exchange_anharmonic_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_fock,
        nlevels,
    )

    coord_keep2 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
    )
    coord_keep4 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=4,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
    )
    pvb_keep2 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
    )
    pvb_keep6 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=6,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
    )

    coord_err2 = np.max(np.abs(coord_keep2 - exact))
    coord_err4 = np.max(np.abs(coord_keep4 - exact))
    pvb_err2 = np.max(np.abs(pvb_keep2 - exact))
    pvb_err6 = np.max(np.abs(pvb_keep6 - exact))

    assert coord_err4 < coord_err2
    assert pvb_err6 < pvb_err2
    assert coord_err4 < 5.0e-6
    assert pvb_err6 < 1.0e-6


def test_high_low_exchange_with_nonlinear_coupling_multistate_converges():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    quartic = 0.1
    nonlinear = 0.05
    n_high_fock = 16
    nlevels = 16
    exact = _high_low_exchange_anharmonic_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_fock,
        nlevels,
        nonlinear=nonlinear,
    )

    coord_keep2 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
    )
    coord_keep4 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=4,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
    )
    pvb_keep2 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
    )
    pvb_keep6 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=6,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
    )

    coord_err2 = np.max(np.abs(coord_keep2 - exact))
    coord_err4 = np.max(np.abs(coord_keep4 - exact))
    pvb_err2 = np.max(np.abs(pvb_keep2 - exact))
    pvb_err6 = np.max(np.abs(pvb_keep6 - exact))

    assert coord_err4 < coord_err2
    assert pvb_err6 < pvb_err2
    assert coord_err4 < 5.0e-6
    assert pvb_err6 < 5.0e-9


def test_high_low_exchange_with_momentum_squared_coupling_multistate_converges():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.35
    quartic = 0.1
    nonlinear = 0.05
    momentum_nonlinear = 0.002
    n_high_fock = 16
    nlevels = 16
    exact = _high_low_exchange_anharmonic_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_fock,
        nlevels,
        nonlinear=nonlinear,
        momentum_nonlinear=momentum_nonlinear,
    )

    coord_keep2 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
        momentum_nonlinear=momentum_nonlinear,
    )
    coord_keep4 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=4,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
        momentum_nonlinear=momentum_nonlinear,
    )
    pvb_keep2 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
        momentum_nonlinear=momentum_nonlinear,
    )
    pvb_keep6 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=6,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        nonlinear=nonlinear,
        momentum_nonlinear=momentum_nonlinear,
    )

    coord_err2 = np.max(np.abs(coord_keep2 - exact))
    coord_err4 = np.max(np.abs(coord_keep4 - exact))
    pvb_err2 = np.max(np.abs(pvb_keep2 - exact))
    pvb_err6 = np.max(np.abs(pvb_keep6 - exact))

    assert coord_err4 < coord_err2
    assert pvb_err6 < pvb_err2
    assert coord_err4 < 1.0e-5
    assert pvb_err6 < 1.0e-7


def test_phase_space_gaussian_coupling_favors_pvb_in_compressed_basis():
    omega_high = 6.0
    omega_low = 0.5
    coupling = 0.0
    quartic = 0.05
    phase_space_coupling = (5.0, 0.0, 3.879991, 2.0, 1.2)
    n_high_fock = 16
    nlevels = 16
    exact = _high_low_exchange_anharmonic_exact_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_fock,
        nlevels,
        phase_space_coupling=phase_space_coupling,
    )

    coord_keep2 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        phase_space_coupling=phase_space_coupling,
    )
    coord_keep4 = _high_low_exchange_anharmonic_coordinate_dvr_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=4,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        phase_space_coupling=phase_space_coupling,
    )
    pvb_keep2 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=2,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        phase_space_coupling=phase_space_coupling,
    )
    pvb_keep6 = _high_low_exchange_anharmonic_pvb_ldr_gaps(
        omega_high,
        omega_low,
        coupling,
        quartic,
        n_high_states=6,
        n_high_fock=n_high_fock,
        nlevels=nlevels,
        phase_space_coupling=phase_space_coupling,
    )

    coord_err2 = np.max(np.abs(coord_keep2 - exact))
    coord_err4 = np.max(np.abs(coord_keep4 - exact))
    pvb_err2 = np.max(np.abs(pvb_keep2 - exact))
    pvb_err6 = np.max(np.abs(pvb_keep6 - exact))

    assert pvb_err2 < 0.5 * coord_err2
    assert coord_err4 < coord_err2
    assert pvb_err6 < pvb_err2
    assert coord_err4 < 1.0e-6
    assert pvb_err6 < 1.0e-8
