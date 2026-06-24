import types

import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.models.heisenberg import Heisenberg
from pyqed.models.impurity.sbm import SBM
import pyqed.mps.tdvp as tdvp_module
from pyqed.mps.decompose import decompose, tt_to_tensor
from pyqed.mps.mps import MPS, MPO, _mpo_to_dense_operator
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.tdvp import two_site_tdvp_step


def _cpp_tdvp_or_skip():
    from pyqed.mps import tdvp_cpp

    if not tdvp_cpp.CPP_TDVP_AVAILABLE:
        pytest.skip(f"C++ TDVP backend unavailable: {tdvp_cpp.CPP_TDVP_BUILD_ERROR}")
    if not tdvp_cpp.CPP_TDVP_HAS_BLAS:
        pytest.skip("C++ TDVP backend built without BLAS contractions")
    return tdvp_cpp


class _IdentityOp:
    def __matmul__(self, psi):
        return psi


def test_tdmps_run_uses_checkpoint_times_for_tail_interval():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    td = TDMPS(H, D=8)
    td.build_propagator = types.MethodType(lambda self, dt, order=2, scale=0: None, td)
    td.step = types.MethodType(lambda self, psi, **kwargs: psi, td)

    td.run(psi0, dt=0.1, steps=5, e_ops=[H], interval=2)

    np.testing.assert_allclose(td.times, np.array([0.2, 0.4, 0.5]))
    assert td.observables.shape == (3, 1)
    np.testing.assert_allclose(td.observables[:, 0], td.observables[0, 0])


def test_sbm_tddmrg_builds_hamiltonian_before_returning():
    model = SBM(Himp=None, alpha=0.1, delta=1.0, epsilon=0.0)
    model.nmodes = 1
    model.t0 = 0.0
    model.onsite = np.array([0.0])
    model.hopping = np.array([], dtype=float)

    td = model.TDDMRG(D=8, nb=4)

    assert model.H is not None
    assert td.H is model.H


def test_tdmps_dynamic_run_uses_split_propagation_without_full_rebuild():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    td = TDMPS(H, D=8, interaction_mpo=H, field=lambda t: 1.0)

    def _fail_build_propagator(self, dt, order=2, scale=0, time=0.0, field=None):
        raise AssertionError("full propagator rebuild should not be used in dynamic split mode")

    def _fake_static(self, dt, order=2, scale=0):
        self.U_static = _IdentityOp()
        self.U_static_half = _IdentityOp()
        return self.U_static, self.U_static_half

    def _fake_interaction(self, dt, time=0.0, field=None, order=2, scale=0):
        return _IdentityOp()

    td.build_propagator = types.MethodType(_fail_build_propagator, td)
    td.build_static_propagators = types.MethodType(_fake_static, td)
    td.build_interaction_propagator = types.MethodType(_fake_interaction, td)

    td.run(psi0, dt=0.1, steps=3, e_ops=[], interval=1, field=lambda t: 1.0)

    np.testing.assert_allclose(td.times, np.array([0.1, 0.2, 0.3]))
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(3))
    assert len(td.substep_pre_normalization_norms) == 3
    assert len(td.substep_pre_normalization_norms[0]) == 3
    assert td.static_energies.shape == (4,)
    assert td.energy_drift.shape == (4,)


def test_tdmps_affine_hamiltonian_cache_matches_mpo_addition():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    td = TDMPS(H, D=8, interaction_mpo=H)

    eff = td.hamiltonian(time=0.0, field=lambda t: 0.25)
    expected = H + (-0.25) * H
    np.testing.assert_allclose(
        _mpo_to_dense_operator(eff),
        _mpo_to_dense_operator(expected),
        atol=1.0e-12,
    )

    eff2 = td.hamiltonian(time=0.0, field=lambda t: 0.5)
    expected2 = H + (-0.5) * H
    np.testing.assert_allclose(
        _mpo_to_dense_operator(eff2),
        _mpo_to_dense_operator(expected2),
        atol=1.0e-12,
    )
    assert len(td._affine_hamiltonian_cache) == 1
    assert eff.factors[-1] is eff2.factors[-1]


def test_tdmps_tdvp_matches_exact_dense_for_full_two_site_manifold():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()

    rng = np.random.default_rng(3)
    vec0 = rng.normal(size=4) + 1j * rng.normal(size=4)
    vec0 = vec0 / np.linalg.norm(vec0)
    psi0 = MPS(decompose(vec0.reshape(2, 2), rank=2), labels=["lv", "p", "rv"]).normalize()

    dt = 0.03
    td = TDMPS(H, D=8)
    td.run(psi0, dt=dt, steps=1, e_ops=[H], integrator="tdvp")

    h_dense = _mpo_to_dense_operator(H)
    exact = expm(-1j * dt * h_dense) @ vec0
    exact = exact / np.linalg.norm(exact)
    actual = np.asarray(tt_to_tensor(td.final_state.factors), dtype=complex).reshape(-1)

    overlap = np.vdot(exact, actual)
    np.testing.assert_allclose(abs(overlap), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(1), atol=1.0e-12)
    reversal = td.time_reversal_error(psi0, dt=dt, steps=1, integrator="tdvp")
    assert reversal["state_error"] < 1.0e-7


def test_tdmps_two_site_tdvp_grows_product_state_bond_and_matches_exact():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    dt = 0.08
    td = TDMPS(H, D=8)
    td.run(psi0, dt=dt, steps=1, e_ops=[H], integrator="tdvp2")

    h_dense = _mpo_to_dense_operator(H)
    vec0 = np.asarray(tt_to_tensor(psi0.factors), dtype=complex).reshape(-1)
    exact = expm(-1j * dt * h_dense) @ vec0
    exact = exact / np.linalg.norm(exact)
    actual = np.asarray(tt_to_tensor(td.final_state.factors), dtype=complex).reshape(-1)

    overlap = np.vdot(exact, actual)
    np.testing.assert_allclose(abs(overlap), 1.0, atol=1.0e-12)
    assert max(td.final_state.bond_orders()) > 1
    np.testing.assert_allclose(td.tdvp_truncation_errors, np.zeros(1), atol=1.0e-14)

    reversal = td.time_reversal_error(psi0, dt=dt, steps=1, integrator="tdvp2")
    assert reversal["state_error"] < 1.0e-7


def test_tdmps_lanczos_krylov_matches_arnoldi_backend():
    model = Heisenberg(L=3)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    states = {}
    for method in ("arnoldi", "lanczos"):
        td = TDMPS(H, D=8)
        td.run(
            psi0,
            dt=0.03,
            steps=2,
            e_ops=[],
            integrator="tdvp2",
            krylov_dim=8,
            krylov_method=method,
            measure_observables=False,
            track_energy=False,
            progress=False,
        )
        states[method] = td.final_state

    diagnostic = TDMPS.overlap_diagnostic(
        TDMPS.state_overlap(states["arnoldi"], states["lanczos"]),
        states["arnoldi"].norm(),
        states["lanczos"].norm(),
    )
    assert diagnostic["state_error"] < 1.0e-8


def test_cpp_tdvp_site_lanczos_matches_python_backend(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    rng = np.random.default_rng(31)
    phys_dim = 4
    raw = rng.normal(size=(phys_dim, phys_dim)) + 1j * rng.normal(size=(phys_dim, phys_dim))
    h_local = 0.5 * (raw + raw.conj().T)
    theta = rng.normal(size=(1, phys_dim, 1)) + 1j * rng.normal(size=(1, phys_dim, 1))
    left = np.ones((1, 1, 1), dtype=complex)
    right = np.ones((1, 1, 1), dtype=complex)
    W = h_local.reshape(1, 1, phys_dim, phys_dim)

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    expected = tdvp_module._evolve_site(
        theta,
        left,
        W,
        right,
        0.07,
        krylov_dim=phys_dim,
        krylov_tol=1.0e-14,
    )

    actual = tdvp_cpp.site_lanczos(theta, left, W, right, 0.07, phys_dim, 1.0e-14)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_cpp_tdvp_sparse_site_lanczos_matches_python_backend(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    rng = np.random.default_rng(34)
    bond_dim = 2
    phys_dim = 2
    mpo_dim = 3
    raw = rng.normal(size=(phys_dim, phys_dim)) + 1j * rng.normal(size=(phys_dim, phys_dim))
    h_local = 0.5 * (raw + raw.conj().T)
    theta = rng.normal(size=(bond_dim, phys_dim, bond_dim)) + 1j * rng.normal(
        size=(bond_dim, phys_dim, bond_dim)
    )
    left = np.zeros((bond_dim, mpo_dim, bond_dim), dtype=complex)
    right = np.zeros((bond_dim, mpo_dim, bond_dim), dtype=complex)
    for i in range(bond_dim):
        left[i, 0, i] = 1.0
        right[i, 0, i] = 1.0
    W = np.zeros((mpo_dim, mpo_dim, phys_dim, phys_dim), dtype=complex)
    W[0, 0] = h_local

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    expected = tdvp_module._evolve_site(
        theta,
        left,
        W,
        right,
        0.04,
        krylov_dim=bond_dim * phys_dim * bond_dim,
        krylov_tol=1.0e-14,
    )

    actual = tdvp_cpp.site_lanczos(
        theta,
        left,
        W,
        right,
        0.04,
        bond_dim * phys_dim * bond_dim,
        1.0e-14,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_cpp_tdvp_bond_lanczos_matches_python_backend(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    rng = np.random.default_rng(32)
    bond_dim = 4
    center = rng.normal(size=(bond_dim, bond_dim)) + 1j * rng.normal(size=(bond_dim, bond_dim))
    left_diag = rng.normal(size=bond_dim)
    right_diag = rng.normal(size=bond_dim)
    left = np.zeros((bond_dim, 1, bond_dim), dtype=complex)
    right = np.zeros((bond_dim, 1, bond_dim), dtype=complex)
    for i in range(bond_dim):
        left[i, 0, i] = left_diag[i]
        right[i, 0, i] = right_diag[i]

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    expected = tdvp_module._evolve_bond(
        center,
        left,
        right,
        0.05,
        krylov_dim=bond_dim * bond_dim,
        krylov_tol=1.0e-14,
    )

    actual = tdvp_cpp.bond_lanczos(center, left, right, -0.05, bond_dim * bond_dim, 1.0e-14)
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)


def test_tdmps_one_site_tdvp_cpp_backend_matches_python(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    model = Heisenberg(L=3)
    H = model.build_H_mpo()
    rng = np.random.default_rng(33)
    vec0 = rng.normal(size=8) + 1j * rng.normal(size=8)
    vec0 = vec0 / np.linalg.norm(vec0)
    psi0 = MPS(decompose(vec0.reshape(2, 2, 2), rank=8), labels=["lv", "p", "rv"]).normalize()

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    python_td = TDMPS(H, D=8)
    python_td.run(
        psi0,
        dt=0.04,
        steps=2,
        e_ops=[],
        integrator="tdvp",
        krylov_dim=8,
        krylov_method="lanczos",
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", tdvp_cpp)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    cpp_td = TDMPS(H, D=8)
    cpp_td.run(
        psi0,
        dt=0.04,
        steps=2,
        e_ops=[],
        integrator="tdvp",
        krylov_dim=8,
        krylov_method="lanczos",
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    diagnostic = TDMPS.overlap_diagnostic(
        TDMPS.state_overlap(python_td.final_state, cpp_td.final_state),
        python_td.final_state.norm(),
        cpp_td.final_state.norm(),
    )
    assert diagnostic["state_error"] < 1.0e-7


def test_tdmps_can_skip_measurements_for_fair_propagation_timing():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    td = TDMPS(H, D=8)
    td.run(
        psi0,
        dt=0.02,
        steps=2,
        e_ops=[H],
        integrator="tdvp2",
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    assert td.observables.shape == (2, 1)
    assert np.all(np.isnan(td.observables))
    assert np.all(np.isnan(td.static_energies))
    assert td.final_state is not None


def test_tdmps_stateful_tdvp_matches_canonicalize_each_step():
    model = Heisenberg(L=4)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    fast = TDMPS(H, D=8)
    fast.run(
        psi0,
        dt=0.02,
        steps=3,
        e_ops=[],
        integrator="tdvp2",
        krylov_dim=8,
        canonicalize_each_step=False,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    safe = TDMPS(H, D=8)
    safe.run(
        psi0,
        dt=0.02,
        steps=3,
        e_ops=[],
        integrator="tdvp2",
        krylov_dim=8,
        canonicalize_each_step=True,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    diagnostic = TDMPS.overlap_diagnostic(
        TDMPS.state_overlap(fast.final_state, safe.final_state),
        fast.final_state.norm(),
        safe.final_state.norm(),
    )
    assert diagnostic["state_error"] < 1.0e-8


def test_two_site_tdvp_diagonal_mpo_fast_path_matches_dense_path():
    rng = np.random.default_rng(11)
    nsites = 4
    phys_dim = 2
    bond_dim = 3
    vec = rng.normal(size=phys_dim**nsites) + 1j * rng.normal(size=phys_dim**nsites)
    vec = vec / np.linalg.norm(vec)
    psi = MPS(
        decompose(vec.reshape((phys_dim,) * nsites), rank=phys_dim**nsites),
        labels=["lv", "p", "rv"],
    ).normalize()

    mpo_factors = []
    for site in range(nsites):
        left_dim = 1 if site == 0 else bond_dim
        right_dim = 1 if site == nsites - 1 else bond_dim
        core = np.zeros((left_dim, right_dim, phys_dim, phys_dim), dtype=complex)
        diag = rng.normal(size=(left_dim, right_dim, phys_dim))
        for p in range(phys_dim):
            core[:, :, p, p] = diag[:, :, p]
        mpo_factors.append(core)
    H = MPO(mpo_factors, homogenous=False)

    fast = two_site_tdvp_step(
        psi,
        H,
        0.02,
        max_bond=8,
        krylov_dim=8,
        diagonal_fast_path=True,
    )
    original_detector = tdvp_module._physical_diagonal_blocks
    tdvp_module._physical_diagonal_blocks = lambda W, *, cutoff=1.0e-14: None
    try:
        dense = two_site_tdvp_step(psi, H, 0.02, max_bond=8, krylov_dim=8)
    finally:
        tdvp_module._physical_diagonal_blocks = original_detector

    np.testing.assert_allclose(
        tt_to_tensor(fast.factors),
        tt_to_tensor(dense.factors),
        atol=1.0e-10,
        rtol=1.0e-10,
    )
