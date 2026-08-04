import sys
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
from pyqed.mps.tdvp import SymmetricTDVP, spatial_fermion_number_sz_sectors, two_site_tdvp_step


def _cpp_tdvp_or_skip():
    from pyqed.mps import tdvp_cpp

    if not tdvp_cpp.CPP_TDVP_AVAILABLE:
        pytest.skip(f"C++ TDVP backend unavailable: {tdvp_cpp.CPP_TDVP_BUILD_ERROR}")
    if not tdvp_cpp.CPP_TDVP_HAS_BLAS:
        pytest.skip("C++ TDVP backend built without BLAS contractions")
    return tdvp_cpp


def _cpp_davidson_or_skip():
    try:
        from pyqed.mps import cpp_davidson
    except ImportError as exc:
        pytest.skip(f"C++ Davidson backend module unavailable: {exc}")

    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE:
        pytest.skip(f"C++ Davidson backend unavailable: {cpp_davidson.CPP_DAVIDSON_BUILD_ERROR}")
    if cpp_davidson.abelian_tdvp_site_heff_data is None:
        pytest.skip("C++ Abelian TDVP Heff kernels unavailable")
    if cpp_davidson.AbelianTDVPSiteHeffPlan is None:
        pytest.skip("C++ Abelian TDVP Heff plan kernels unavailable")
    return cpp_davidson


def _assert_abelian_data_allclose(actual, expected, *, atol=1.0e-12):
    assert actual.qns == expected.qns
    assert actual.dirs == expected.dirs
    assert set(actual.data) == set(expected.data)
    for key, block in actual.data.items():
        np.testing.assert_allclose(block, expected.data[key], atol=atol)


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
    assert td.substep_pre_normalization_norms is None
    assert td.static_energies.shape == (4,)
    assert td.energy_drift.shape == (4,)


def test_tdmps_callable_field_is_precomputed_before_step_loop():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()
    field_times = []
    step_fields = []

    def field(t):
        field_times.append(float(t))
        return np.array([0.0, 0.0, t])

    td = TDMPS(H, D=8, interaction_mpo=H, field=field)

    def _fake_step(self, psi, **kwargs):
        value = kwargs.get("field")
        assert not callable(value)
        step_fields.append(np.asarray(value, dtype=float))
        self._last_step_pre_normalization_norms = (1.0,)
        self._last_step_pre_normalization_norm2 = (1.0,)
        self._last_step_tdvp_truncation_error = 0.0
        return psi

    td.step = types.MethodType(_fake_step, td)
    td.run(
        psi0,
        dt=0.1,
        steps=3,
        e_ops=[],
        interval=2,
        integrator="tdvp",
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    np.testing.assert_allclose(field_times, [0.05, 0.15, 0.25, 0.2, 0.3])
    np.testing.assert_allclose([vec[2] for vec in step_fields], [0.05, 0.15, 0.25])
    np.testing.assert_allclose(td.fields[:, 2], [0.2, 0.3])


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


def test_affine_block_sparse_mpo_reuses_shared_tail(monkeypatch):
    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)
    td = TDMPS(h_mpo, D=8, interaction_mpo=h_mpo)
    site_qn_maps, _target_qn = tdvp_module._block_sparse_site_qn_maps(
        [0, 1],
        2,
        (2, 2),
        1,
    )

    tdvp_module._AFFINE_BLOCK_SPARSE_MPO_CACHE.clear()
    calls = []
    original = tdvp_module.dense_to_symmetric_mpo

    def _counting_dense_to_symmetric_mpo(dense_mpo_list, *args, **kwargs):
        calls.append(len(dense_mpo_list))
        return original(dense_mpo_list, *args, **kwargs)

    monkeypatch.setattr(
        tdvp_module,
        "dense_to_symmetric_mpo",
        _counting_dense_to_symmetric_mpo,
    )
    eff1 = td.hamiltonian(time=0.0, field=lambda t: 0.25)
    eff2 = td.hamiltonian(time=0.0, field=lambda t: 0.5)

    tdvp_module._as_block_sparse_mpo(eff1, site_qn_maps)
    cached = tdvp_module._as_block_sparse_mpo(eff2, site_qn_maps)

    assert calls == [2]
    full = original(
        [np.asarray(w) for w in eff2.factors],
        site_qn_maps,
        native_site_storage=True,
    )
    for actual, expected in zip(cached, full):
        _assert_abelian_data_allclose(actual, expected)


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


def test_tdmps_compressed_taylor_step_matches_exact_dense():
    model = Heisenberg(L=2)
    H = model.build_H_mpo()
    rng = np.random.default_rng(37)
    vec0 = rng.normal(size=4) + 1j * rng.normal(size=4)
    vec0 /= np.linalg.norm(vec0)
    psi0 = MPS(decompose(vec0.reshape(2, 2), rank=2))

    dt = 0.03
    td = TDMPS(H, D=8)
    td.run(
        psi0,
        dt=dt,
        steps=1,
        e_ops=[],
        integrator="taylor",
        order=4,
        scale=2,
        progress=False,
    )

    exact = expm(-1j * dt * _mpo_to_dense_operator(H)) @ vec0
    actual = np.asarray(tt_to_tensor(td.final_state.factors), dtype=complex).reshape(-1)
    np.testing.assert_allclose(actual, exact, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(td.pre_normalization_norms, np.ones(1), atol=1.0e-10)


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


def test_tdmps_two_site_cutoff_adapts_bonds_below_hard_limit():
    model = Heisenberg(L=6)
    H = model.build_H_mpo()
    psi0 = model.build_neel_state()

    full = TDMPS(H, D=64).run(
        psi0,
        dt=0.1,
        steps=2,
        integrator="tdvp2",
        cutoff=0.0,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )
    adaptive = TDMPS(H, D=64).run(
        psi0,
        dt=0.1,
        steps=2,
        integrator="tdvp2",
        cutoff=1.0e-6,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    assert max(adaptive.final_state.bond_orders()) < max(
        full.final_state.bond_orders()
    )
    assert np.max(adaptive.bond_dimensions) <= 64
    assert adaptive.bond_dimensions.shape == (3, 6)
    assert np.max(adaptive.tdvp_truncation_errors) > 0.0


def test_tdmps_hybrid_integrator_records_warmup_and_enrichment_schedule():
    model = Heisenberg(L=4)
    driver = TDMPS(model.build_H_mpo(), D=8).run(
        model.build_neel_state(),
        dt=0.02,
        steps=8,
        integrator="hybrid",
        hybrid_warmup_steps=2,
        hybrid_tdvp2_interval=3,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    assert driver.integrator_history.tolist() == [
        "tdvp2",
        "tdvp2",
        "tdvp",
        "tdvp",
        "tdvp2",
        "tdvp",
        "tdvp",
        "tdvp2",
    ]
    np.testing.assert_allclose(driver.final_state.norm(), 1.0, atol=1.0e-12)
    assert max(driver.final_state.bond_orders()) > 1


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


def test_cpp_tdvp_two_site_lanczos_matches_python_backend(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    rng = np.random.default_rng(35)
    bond_dim = 2
    phys_dim = 3
    raw_left = rng.normal(size=(phys_dim, phys_dim)) + 1j * rng.normal(
        size=(phys_dim, phys_dim)
    )
    raw_right = rng.normal(size=(phys_dim, phys_dim)) + 1j * rng.normal(
        size=(phys_dim, phys_dim)
    )
    h_left = 0.5 * (raw_left + raw_left.conj().T)
    h_right = 0.5 * (raw_right + raw_right.conj().T)
    identity = np.eye(phys_dim)
    W_left = np.stack((h_left, identity), axis=0).reshape(
        1, 2, phys_dim, phys_dim
    )
    W_right = np.stack((identity, h_right), axis=0).reshape(
        2, 1, phys_dim, phys_dim
    )
    theta = rng.normal(
        size=(bond_dim, phys_dim, phys_dim, bond_dim)
    ) + 1j * rng.normal(size=(bond_dim, phys_dim, phys_dim, bond_dim))
    raw_left_env = rng.normal(size=(bond_dim, bond_dim)) + 1j * rng.normal(
        size=(bond_dim, bond_dim)
    )
    raw_right_env = rng.normal(size=(bond_dim, bond_dim)) + 1j * rng.normal(
        size=(bond_dim, bond_dim)
    )
    left = (0.5 * (raw_left_env + raw_left_env.conj().T))[:, None, :]
    right = (0.5 * (raw_right_env + raw_right_env.conj().T))[:, None, :]

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    krylov_dim = int(np.prod(theta.shape))
    expected = tdvp_module._evolve_two_site(
        theta,
        left,
        W_left,
        W_right,
        right,
        0.03,
        krylov_dim=krylov_dim,
        krylov_tol=1.0e-14,
    )

    actual = tdvp_cpp.two_site_lanczos(
        theta,
        left,
        W_left,
        W_right,
        right,
        0.03,
        krylov_dim,
        1.0e-14,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-11, rtol=1.0e-11)


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


def test_tdmps_two_site_tdvp_cpp_backend_matches_python(monkeypatch):
    tdvp_cpp = _cpp_tdvp_or_skip()
    model = Heisenberg(L=3)
    H = model.build_H_mpo()
    rng = np.random.default_rng(36)
    vec0 = rng.normal(size=8) + 1j * rng.normal(size=8)
    vec0 /= np.linalg.norm(vec0)
    psi0 = MPS(
        decompose(vec0.reshape(2, 2, 2), rank=8),
        labels=["lv", "p", "rv"],
    ).normalize()

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", None)
    monkeypatch.setattr(tdvp_module, "_tdvp_cpp_tried", True)
    python_td = TDMPS(H, D=8)
    python_td.run(
        psi0,
        dt=0.04,
        steps=2,
        integrator="tdvp2",
        krylov_dim=8,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    monkeypatch.setattr(tdvp_module, "_tdvp_cpp", tdvp_cpp)
    cpp_td = TDMPS(H, D=8)
    cpp_td.run(
        psi0,
        dt=0.04,
        steps=2,
        integrator="tdvp2",
        krylov_dim=8,
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    diagnostic = TDMPS.overlap_diagnostic(
        TDMPS.state_overlap(python_td.final_state, cpp_td.final_state),
        python_td.final_state.norm(),
        cpp_td.final_state.norm(),
    )
    assert diagnostic["state_error"] < 1.0e-8


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


def test_symmetric_tdvp_projects_dense_mps_to_target_sector():
    rng = np.random.default_rng(41)
    nsites = 3
    phys_dim = 2
    vec = rng.normal(size=phys_dim**nsites) + 1j * rng.normal(size=phys_dim**nsites)
    psi = MPS(
        decompose(vec.reshape((phys_dim,) * nsites), rank=phys_dim**nsites),
        labels=["lv", "p", "rv"],
    )
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=[0, 1],
        target_sector=1,
        max_bond=phys_dim**nsites,
    )

    projected, info = engine.project(psi, return_info=True)
    tensor = np.asarray(tt_to_tensor(projected.factors), dtype=complex).reshape((phys_dim,) * nsites)

    for index in np.ndindex(tensor.shape):
        if sum(index) != 1:
            assert abs(tensor[index]) < 1.0e-12
    np.testing.assert_allclose(projected.norm(), 1.0, atol=1.0e-12)
    assert 0.0 < info["sector_weight"] < 1.0
    assert info["backend"] == "sector-mpo"
    assert info["max_projector_bond"] == 2


def test_symmetric_tdvp_one_site_step_preserves_target_sector():
    nsites = 3
    phys_dim = 2
    vec = np.zeros((phys_dim,) * nsites, dtype=complex)
    vec[1, 0, 0] = 1.0 / np.sqrt(2.0)
    vec[0, 1, 0] = 1.0j / np.sqrt(2.0)
    psi = MPS(
        decompose(vec, rank=phys_dim**nsites),
        labels=["lv", "p", "rv"],
    ).normalize()
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=[0, 1],
        target_sector=1,
        max_bond=phys_dim**nsites,
        krylov_dim=4,
    )

    out, info = engine.step(psi, 0.05, return_info=True)
    actual = np.asarray(tt_to_tensor(out.factors), dtype=complex).reshape((phys_dim,) * nsites)

    np.testing.assert_allclose(actual, vec, atol=1.0e-12)
    np.testing.assert_allclose(out.norm(), 1.0, atol=1.0e-12)
    assert info["input_sector_weight"] == pytest.approx(1.0)
    assert info["output_sector_weight"] == pytest.approx(1.0)
    assert info["integrator"] == "tdvp"
    assert info["projection_backend"] == "sector-mpo"


def test_symmetric_tdvp_block_sparse_step_matches_exact_fixed_sector():
    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 0.8
    vec[0, 1] = 0.6j
    vec = vec / np.linalg.norm(vec.reshape(-1))
    psi = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()
    engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=1,
        projection_backend="block-sparse",
        krylov_dim=8,
    )

    out, info = engine.step(psi, 0.1, return_info=True)
    dense_out = tdvp_module.symmetric_to_dense(out)
    actual = np.asarray(tt_to_tensor(dense_out.factors), dtype=complex).reshape(-1)
    exact = expm(-1j * 0.1 * _mpo_to_dense_operator(h_mpo)) @ vec.reshape(-1)

    assert hasattr(out.factors[0], "qns")
    assert info["projection_backend"] == "block-sparse"
    np.testing.assert_allclose(out.norm(), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(abs(np.vdot(exact, actual)), 1.0, atol=1.0e-12)


def test_block_sparse_tdvp_canonicalizes_initial_state_and_conserves_energy():
    nsites = 6
    bond = 4
    rng = np.random.default_rng(17)
    vec = np.zeros(2**nsites, dtype=complex)
    for index in range(vec.size):
        if index.bit_count() == nsites // 2:
            vec[index] = rng.normal() + 1j * rng.normal()
    vec /= np.linalg.norm(vec)

    psi = MPS(
        decompose(vec.reshape((2,) * nsites), rank=bond),
        labels=["lv", "p", "rv"],
    )
    h_mpo = Heisenberg(nsites).build_H_mpo()
    h_dense = _mpo_to_dense_operator(h_mpo)
    engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=nsites // 2,
        projection_backend="block-sparse",
        max_bond=bond,
        krylov_dim=12,
        krylov_tol=1.0e-13,
    )

    state = engine.project(psi)
    dense = tdvp_module.symmetric_to_dense(state)
    initial = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
    energy0 = np.vdot(initial, h_dense @ initial)
    for _ in range(3):
        state, _ = engine.step(state, 0.01, return_info=True)
    dense = tdvp_module.symmetric_to_dense(state)
    final = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
    energy1 = np.vdot(final, h_dense @ final)

    assert engine.canonicalize_first is True
    np.testing.assert_allclose(np.linalg.norm(final), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(energy1, energy0, atol=1.0e-11)


def test_block_sparse_tdvp_native_one_site_sweep_matches_python(monkeypatch):
    _cpp_davidson_or_skip()

    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 0.8
    vec[0, 1] = 0.6j
    vec = vec / np.linalg.norm(vec.reshape(-1))
    psi0 = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()

    def run(native):
        monkeypatch.setattr(tdvp_module, "_BLOCK_ONE_SITE_CPP_ENGINE", int(native))
        engine = SymmetricTDVP(
            h_mpo,
            local_sectors=[0, 1],
            target_sector=1,
            projection_backend="block-sparse",
            krylov_dim=8,
        )
        psi = psi0.copy()
        info = {}
        for _ in range(2):
            psi, info = engine.step(psi, 0.07, return_info=True)
        dense = tdvp_module.symmetric_to_dense(psi)
        tensor = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
        return tensor, info

    py_tensor, py_info = run(False)
    native_tensor, native_info = run(True)

    assert py_info["cpp_one_site_engine"] is False
    assert native_info["cpp_one_site_engine"] is True
    assert native_info["cpp_one_site_engine_native_kernels"] is True
    np.testing.assert_allclose(native_tensor, py_tensor, atol=1.0e-12)


def test_block_sparse_tdvp_native_adaptive_krylov_matches_full_dimension():
    _cpp_davidson_or_skip()

    nsites = 6
    bond = 4
    rng = np.random.default_rng(17)
    vec = np.zeros(2**nsites, dtype=complex)
    for index in range(vec.size):
        if index.bit_count() == nsites // 2:
            vec[index] = rng.normal() + 1j * rng.normal()
    vec /= np.linalg.norm(vec)
    psi = MPS(
        decompose(vec.reshape((2,) * nsites), rank=bond),
        labels=["lv", "p", "rv"],
    )
    h_mpo = Heisenberg(nsites).build_H_mpo()

    def run(krylov_tol):
        engine = SymmetricTDVP(
            h_mpo,
            local_sectors=[0, 1],
            target_sector=nsites // 2,
            projection_backend="block-sparse",
            max_bond=bond,
            krylov_dim=12,
            krylov_tol=krylov_tol,
        )
        state = engine.project(psi)
        info = {}
        for _ in range(2):
            state, info = engine.step(state, 0.01, return_info=True)
        dense = tdvp_module.symmetric_to_dense(state)
        tensor = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
        return tensor, info

    full, full_info = run(0.0)
    adaptive, adaptive_info = run(1.0e-13)
    phase = np.vdot(full, adaptive)
    phase /= abs(phase)

    np.testing.assert_allclose(adaptive, phase * full, atol=1.0e-12)
    assert adaptive_info["cpp_one_site_engine_site_matvecs"] < full_info[
        "cpp_one_site_engine_site_matvecs"
    ]
    assert adaptive_info["cpp_one_site_engine_bond_matvecs"] < full_info[
        "cpp_one_site_engine_bond_matvecs"
    ]


def test_block_sparse_two_site_tdvp_grows_bond_and_matches_exact():
    model = Heisenberg(2)
    psi = model.build_neel_state()
    h_mpo = model.build_H_mpo()
    initial = np.asarray(tt_to_tensor(psi.factors), dtype=complex).reshape(-1)
    engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=1,
        integrator="tdvp2",
        projection_backend="block-sparse",
        max_bond=2,
        krylov_dim=8,
    )

    out, info = engine.step(psi, 0.1, return_info=True)
    dense = tdvp_module.symmetric_to_dense(out)
    actual = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
    exact = expm(-0.1j * _mpo_to_dense_operator(h_mpo)) @ initial

    assert hasattr(out.factors[0], "qns")
    assert out.bond_orders()[0] == 2
    assert info["integrator"] == "tdvp2"
    assert info["max_kept_states"] == 2
    np.testing.assert_allclose(actual, exact, atol=1.0e-12)


def test_block_sparse_two_site_tdvp_matches_dense_two_site_sweep():
    model = Heisenberg(4)
    psi = model.build_neel_state()
    h_mpo = model.build_H_mpo()
    dense_engine = tdvp_module.TDVPEngine(
        h_mpo,
        integrator="tdvp2",
        max_bond=4,
        krylov_dim=12,
        krylov_tol=1.0e-13,
    )
    block_engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=2,
        integrator="tdvp2",
        projection_backend="block-sparse",
        max_bond=4,
        krylov_dim=12,
        krylov_tol=1.0e-13,
    )

    dense, _ = dense_engine.step(psi, 0.03, return_info=True)
    block, info = block_engine.step(psi, 0.03, return_info=True)
    dense_vector = np.asarray(tt_to_tensor(dense.factors), dtype=complex).reshape(-1)
    block_vector = np.asarray(
        tt_to_tensor(tdvp_module.symmetric_to_dense(block).factors),
        dtype=complex,
    ).reshape(-1)

    np.testing.assert_allclose(block_vector, dense_vector, atol=1.0e-12)
    assert block.bond_orders() == dense.bond_orders()
    assert info["truncation_error"] == pytest.approx(0.0, abs=1.0e-14)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="dense native TDVP workspace uses Accelerate",
)
def test_dense_tdvp_workspace_matches_local_kernel_and_reuses_mpo_plan():
    _cpp_davidson_or_skip()
    rng = np.random.default_rng(31)
    factors = [
        rng.normal(size=(1, 2, 3)) + 1j * rng.normal(size=(1, 2, 3)),
        rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3)),
        rng.normal(size=(3, 2, 3)) + 1j * rng.normal(size=(3, 2, 3)),
        rng.normal(size=(3, 2, 1)) + 1j * rng.normal(size=(3, 2, 1)),
    ]
    psi = MPS(factors, labels=["lv", "p", "rv"]).right_canonicalize()
    local = np.array([[1.1, 0.2 - 0.1j], [0.2 + 0.1j, -0.7]], dtype=complex)
    mpo = [local.reshape(1, 1, 2, 2) for _ in range(4)]
    reference = tdvp_module.TDVPEngine(
        mpo,
        integrator="tdvp2",
        max_bond=3,
        krylov_dim=8,
        dense_cpp_workspace="off",
    )
    cached = tdvp_module.TDVPEngine(
        mpo,
        integrator="tdvp2",
        max_bond=3,
        krylov_dim=8,
        dense_cpp_workspace="on",
    )
    expected, _ = reference.step(psi, 0.02)
    actual, info = cached.step(psi, 0.02)
    np.testing.assert_allclose(
        np.asarray(tt_to_tensor(actual.factors), dtype=complex),
        np.asarray(tt_to_tensor(expected.factors), dtype=complex),
        atol=1.0e-12,
    )
    assert info["dense_cpp_workspace_two_site_evolutions"] == 6, info["dense_cpp_workspace_last_error"]
    assert info["dense_cpp_workspace_static_w_reuses"] >= 2


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="dense native environment updates use Accelerate",
)
def test_dense_tdvp_cpp_environment_updates_match_einsum():
    _cpp_davidson_or_skip()
    rng = np.random.default_rng(47)
    left = rng.normal(size=(4, 2, 4)) + 1j * rng.normal(size=(4, 2, 4))
    right = rng.normal(size=(5, 2, 5)) + 1j * rng.normal(size=(5, 2, 5))
    site = rng.normal(size=(4, 3, 5)) + 1j * rng.normal(size=(4, 3, 5))
    w = rng.normal(size=(2, 2, 3, 3)) + 1j * rng.normal(size=(2, 2, 3, 3))
    np.testing.assert_allclose(
        tdvp_module._update_left_env(left, site, w, dense_cpp=True),
        tdvp_module._update_left_env(left, site, w),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        tdvp_module._update_right_env(right, site, w, dense_cpp=True),
        tdvp_module._update_right_env(right, site, w),
        atol=1.0e-12,
    )


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="native Abelian sector SVD currently uses Accelerate",
)
def test_block_sparse_two_site_tdvp_uses_cpp_moving_environment_sweep():
    _cpp_davidson_or_skip()
    model = Heisenberg(4)
    engine = SymmetricTDVP(
        model.build_H_mpo(),
        local_sectors=[0, 1],
        target_sector=2,
        integrator="tdvp2",
        projection_backend="block-sparse",
        max_bond=4,
    )
    state = model.build_neel_state()
    info = {}
    for _ in range(2):
        state, info = engine.step(state, 0.03, return_info=True)

    assert info["cpp_two_site_engine"] is True
    assert info["cpp_two_site_engine_native_kernels"] is True
    assert info["cpp_two_site_engine_two_site_evolutions"] == 6
    assert info["cpp_two_site_engine_splits"] == 6
    assert info["cpp_two_site_engine_environment_advances"] == 10
    assert info["cpp_two_site_tdvp_sweep_failures"] == 0
    dense_engine = tdvp_module.TDVPEngine(
        model.build_H_mpo(),
        integrator="tdvp2",
        max_bond=4,
    )
    dense_state = model.build_neel_state()
    for _ in range(2):
        dense_state, _ = dense_engine.step(dense_state, 0.03)
    block_vector = np.asarray(
        tt_to_tensor(tdvp_module.symmetric_to_dense(state).factors),
        dtype=complex,
    ).reshape(-1)
    dense_vector = np.asarray(
        tt_to_tensor(dense_state.factors),
        dtype=complex,
    ).reshape(-1)
    np.testing.assert_allclose(block_vector, dense_vector, atol=1.0e-12)


def test_block_sparse_two_site_tdvp_cutoff_reduces_bond_dimension():
    model = Heisenberg(6)
    h_mpo = model.build_H_mpo()

    def run(cutoff):
        state = model.build_neel_state()
        engine = SymmetricTDVP(
            h_mpo,
            local_sectors=[0, 1],
            target_sector=3,
            integrator="tdvp2",
            projection_backend="block-sparse",
            max_bond=32,
            cutoff=cutoff,
            krylov_dim=10,
        )
        for _ in range(2):
            state, info = engine.step(state, 0.05, return_info=True)
        return state, info

    full, _ = run(0.0)
    adaptive, info = run(1.0e-5)

    assert max(adaptive.bond_orders()) < max(full.bond_orders())
    assert max(adaptive.bond_orders()) <= 32
    assert info["max_kept_states"] < 8
    np.testing.assert_allclose(adaptive.norm(), 1.0, atol=1.0e-12)


def test_tdmps_routes_symmetric_tdvp2_to_block_sparse_engine():
    model = Heisenberg(2)
    driver = TDMPS(
        model.build_H_mpo(),
        D=2,
        local_sectors=[0, 1],
        target_sector=1,
        projection="block-sparse",
    ).run(
        model.build_neel_state(),
        dt=0.05,
        steps=1,
        integrator="tdvp2",
        measure_observables=False,
        track_energy=False,
        progress=False,
    )

    assert hasattr(driver.final_state.factors[0], "qns")
    assert driver.final_state.bond_orders()[0] == 2
    assert driver.integrator_history.tolist() == ["tdvp2"]


def test_symmetric_tdvp_block_sparse_accepts_spatial_sector_tuples():
    nsites = 2
    phys_dim = 4
    vec = np.zeros((phys_dim, phys_dim), dtype=complex)
    vec[1, 2] = 1.0
    psi = MPS(decompose(vec, rank=4), labels=["lv", "p", "rv"]).normalize()
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=spatial_fermion_number_sz_sectors(),
        target_sector=(2, 0),
        projection_backend="block-sparse",
        krylov_dim=4,
    )

    out, info = engine.step(psi, 0.02, return_info=True)
    dense_out = tdvp_module.symmetric_to_dense(out)
    actual = np.asarray(tt_to_tensor(dense_out.factors), dtype=complex).reshape(vec.shape)

    assert hasattr(out.factors[0], "qns")
    assert info["target_qn"].components == (2, 0)
    np.testing.assert_allclose(actual, vec, atol=1.0e-12)
    np.testing.assert_allclose(out.norm(), 1.0, atol=1.0e-12)


def test_symmetric_tdvp_block_sparse_reuses_native_mpo(monkeypatch):
    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 1.0
    psi = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()

    calls = {"count": 0}
    original = tdvp_module._as_block_sparse_mpo

    def _counting_as_block_sparse_mpo(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(tdvp_module, "_as_block_sparse_mpo", _counting_as_block_sparse_mpo)
    engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=1,
        projection_backend="block-sparse",
        krylov_dim=4,
    )

    out, info1 = engine.step(psi, 0.01, return_info=True)
    out, info2 = engine.step(out, 0.01, return_info=True)

    assert calls["count"] == 1
    assert info1["mpo_cached"] is True
    assert info2["mpo_cached"] is True
    assert engine.canonicalize_first is True
    assert hasattr(out.factors[0], "qns")


def test_block_sparse_tdvp_reuses_global_static_mpo_cache(monkeypatch):
    identity = np.eye(2, dtype=complex)
    w0 = np.zeros((1, 1, 2, 2), dtype=complex)
    w1 = np.zeros((1, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w1[0, 0] = identity
    h_mpo = MPO([w0, w1], homogenous=False)
    h_mpo._pyqed_cache_key = ("test-static-mpo-cache", 2)

    site_qn_maps, _target_qn = tdvp_module._block_sparse_site_qn_maps(
        [0, 1],
        2,
        (2, 2),
        1,
    )
    tdvp_module._BLOCK_SPARSE_MPO_CACHE.clear()
    calls = {"count": 0}
    original = tdvp_module.dense_to_symmetric_mpo

    def _counting_dense_to_symmetric_mpo(*args, **kwargs):
        calls["count"] += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(
        tdvp_module,
        "dense_to_symmetric_mpo",
        _counting_dense_to_symmetric_mpo,
    )

    first = tdvp_module._as_block_sparse_mpo(h_mpo, site_qn_maps)
    second = tdvp_module._as_block_sparse_mpo(h_mpo, site_qn_maps)

    assert calls["count"] == 1
    assert first[0] is second[0]
    tdvp_module._BLOCK_SPARSE_MPO_CACHE.clear()


def test_block_sparse_tdvp_cpp_heff_kernels_match_python():
    cpp_davidson = _cpp_davidson_or_skip()

    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 0.8
    vec[0, 1] = 0.6j
    psi = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()

    site_qn_maps, target_qn = tdvp_module._block_sparse_site_qn_maps(
        [0, 1],
        2,
        (2, 2),
        1,
    )
    factors = tdvp_module._as_block_sparse_factors(psi, site_qn_maps)
    mpo = tdvp_module._as_block_sparse_mpo(h_mpo, site_qn_maps)
    right_envs = tdvp_module._build_block_right_envs(factors, mpo, target_qn)
    left = tdvp_module.initial_E(mpo[0])

    site_cpp = tdvp_module._cpp_payload_to_abelian_tensor(
        cpp_davidson.abelian_tdvp_site_heff_data(
            factors[0],
            left,
            mpo[0],
            right_envs[1],
        )
    )
    tmp = tdvp_module.abelian_tensor_data_tensordot(left, factors[0], ([2], [0]))
    tmp = tdvp_module.abelian_tensor_data_tensordot(tmp, mpo[0], ([0, 3], [0, 3]))
    tmp = tdvp_module.abelian_tensor_data_tensordot(tmp, right_envs[1], ([2, 1], [0, 2]))
    site_ref = tdvp_module.abelian_transpose_tensor_data(
        tmp,
        (0, 2, 1),
        carrier=tdvp_module.AbelianSiteTensorData,
    )
    _assert_abelian_data_allclose(site_cpp, site_ref)
    site_plan = cpp_davidson.AbelianTDVPSiteHeffPlan.from_tensors(
        factors[0],
        left,
        mpo[0],
        right_envs[1],
    )
    assert site_plan.route_count() > 0
    site_planned = tdvp_module._cpp_payload_to_abelian_tensor(
        site_plan.apply(factors[0], left, mpo[0], right_envs[1])
    )
    _assert_abelian_data_allclose(site_planned, site_ref)

    q, center = tdvp_module._block_left_qr(factors[0])
    left_next = tdvp_module.contract_from_left(mpo[0], q, left, q)
    if cpp_davidson.MovingEnvironment is not None:
        owner = cpp_davidson.MovingEnvironment()
        planned_left_next = tdvp_module._advance_block_environment(
            "left",
            mpo[0],
            q,
            left,
            q,
            moving_environment=owner,
            plan_key="test-left-env",
        )
        _assert_abelian_data_allclose(planned_left_next, left_next)
        planned_left_next_again = tdvp_module._advance_block_environment(
            "left",
            mpo[0],
            q,
            left,
            q,
            moving_environment=owner,
            plan_key="test-left-env",
        )
        _assert_abelian_data_allclose(planned_left_next_again, left_next)
        stats = dict(owner.stats())
        assert stats["environment_plan_builds"] == 1
        assert stats["environment_plan_cache_hits"] >= 1

    bond_cpp = tdvp_module._cpp_payload_to_abelian_tensor(
        cpp_davidson.abelian_tdvp_bond_heff_data(center, left_next, right_envs[1])
    )
    tmp = tdvp_module.abelian_tensor_data_tensordot(left_next, center, ([2], [0]))
    bond_ref = tdvp_module.abelian_tensor_data_tensordot(
        tmp,
        right_envs[1],
        ([0, 2], [0, 2]),
    )
    _assert_abelian_data_allclose(bond_cpp, bond_ref)
    bond_plan = cpp_davidson.AbelianTDVPBondHeffPlan.from_tensors(
        center,
        left_next,
        right_envs[1],
    )
    assert bond_plan.route_count() > 0
    bond_planned = tdvp_module._cpp_payload_to_abelian_tensor(
        bond_plan.apply(center, left_next, right_envs[1])
    )
    _assert_abelian_data_allclose(bond_planned, bond_ref)


def test_cpp_lapack_qr_matches_numpy_reduced_qr():
    cpp_davidson = _cpp_davidson_or_skip()
    if getattr(cpp_davidson, "lapack_qr", None) is None:
        pytest.skip("C++ LAPACK QR wrapper unavailable")

    rng = np.random.default_rng(123)
    matrix = rng.normal(size=(7, 4)) + 1j * rng.normal(size=(7, 4))
    q_cpp, r_cpp = cpp_davidson.lapack_qr(matrix)
    q_np, r_np = np.linalg.qr(matrix, mode="reduced")

    np.testing.assert_allclose(q_cpp @ r_cpp, matrix, atol=1.0e-12)
    np.testing.assert_allclose(q_cpp.conj().T @ q_cpp, np.eye(4), atol=1.0e-12)
    np.testing.assert_allclose(np.abs(np.diag(r_cpp)), np.abs(np.diag(r_np)), atol=1.0e-12)


def test_block_sparse_symmetric_tdvp_reuses_native_state_after_first_step(monkeypatch):
    monkeypatch.setattr(tdvp_module, "_BLOCK_ONE_SITE_CPP_ENGINE", 1)
    identity = np.eye(2, dtype=complex)
    w0 = np.zeros((1, 1, 2, 2), dtype=complex)
    w1 = np.zeros((1, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w1[0, 0] = identity
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 1.0
    psi = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()
    engine = SymmetricTDVP(
        h_mpo,
        local_sectors=[0, 1],
        target_sector=1,
        projection_backend="block-sparse",
        krylov_dim=4,
    )

    out, info1 = engine.step(psi, 0.01, return_info=True)
    _out2, info2 = engine.step(out, 0.01, return_info=True)

    assert info1["state_copied"] is True
    assert info2["state_copied"] is False
    if info2.get("cpp_moving_environment"):
        assert info1["cpp_one_site_engine"] is True
        assert info2["cpp_one_site_engine"] is True
        assert info1["cpp_one_site_engine_native_kernels"] is True
        assert info2["cpp_one_site_engine_native_kernels"] is True
        assert info1["cpp_one_site_tdvp_sweep_calls"] == 1
        assert info1["cpp_environment_plan_advance_calls"] > 0
        assert info2["cpp_environment_plan_cache_hits"] > 0


def test_tdmps_block_sparse_observables_do_not_densify(monkeypatch):
    import pyqed.mps.tdmps as tdmps_module

    identity = np.eye(2, dtype=complex)
    create = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
    destroy = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
    w0 = np.zeros((1, 3, 2, 2), dtype=complex)
    w1 = np.zeros((3, 1, 2, 2), dtype=complex)
    w0[0, 0] = identity
    w0[0, 1] = create
    w0[0, 2] = destroy
    w1[1, 0] = destroy
    w1[2, 0] = create
    h_mpo = MPO([w0, w1], homogenous=False)

    vec = np.zeros((2, 2), dtype=complex)
    vec[1, 0] = 0.8
    vec[0, 1] = 0.6j
    vec = vec / np.linalg.norm(vec.reshape(-1))
    psi = MPS(decompose(vec, rank=2), labels=["lv", "p", "rv"]).normalize()

    def _fail_densify(*args, **kwargs):
        raise AssertionError("block-sparse observables should not densify the MPS")

    monkeypatch.setattr(tdmps_module, "symmetric_to_dense", _fail_densify)
    td = TDMPS(
        h_mpo,
        D=2,
        local_sectors=[0, 1],
        target_sector=1,
        projection="block-sparse",
    )
    td.run(
        psi,
        dt=0.05,
        steps=1,
        e_ops=[h_mpo],
        integrator="tdvp",
        krylov_dim=8,
        measure_observables=True,
        track_energy=True,
        progress=False,
    )

    assert hasattr(td.final_state.factors[0], "qns")
    np.testing.assert_allclose(td.observables[0, 0], td.static_energies[-1], atol=1.0e-12)
    np.testing.assert_allclose(
        TDMPS.state_overlap(td.final_state, td.final_state),
        td.final_state.norm(),
        atol=1.0e-12,
    )


def test_symmetric_tdvp_mpo_projector_matches_dense_reference():
    rng = np.random.default_rng(43)
    nsites = 4
    phys_dim = 2
    vec = rng.normal(size=phys_dim**nsites) + 1j * rng.normal(size=phys_dim**nsites)
    psi = MPS(
        decompose(vec.reshape((phys_dim,) * nsites), rank=phys_dim**nsites),
        labels=["lv", "p", "rv"],
    )
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=[0, 1],
        target_sector=2,
        max_bond=phys_dim**nsites,
    )

    projected_mpo, mpo_info = engine.project(psi, return_info=True)
    projected_dense, dense_info = engine.project_dense(psi, return_info=True)

    np.testing.assert_allclose(
        tt_to_tensor(projected_mpo.factors),
        tt_to_tensor(projected_dense.factors),
        atol=1.0e-12,
    )
    assert mpo_info["backend"] == "sector-mpo"
    assert dense_info["backend"] == "dense-sector"


def test_symmetric_tdvp_mpo_projector_does_not_need_dense_guard():
    nsites = 5
    phys_dim = 2
    factors = []
    for site in range(nsites):
        core = np.zeros((1, phys_dim, 1), dtype=complex)
        core[0, 1 if site in {1, 3} else 0, 0] = 1.0
        factors.append(core)
    psi = MPS(factors, labels=["lv", "p", "rv"])
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=[0, 1],
        target_sector=2,
        max_dense_sites=2,
    )

    projected, info = engine.project(psi, return_info=True)

    np.testing.assert_allclose(projected.norm(), 1.0, atol=1.0e-12)
    assert info["sector_weight"] == pytest.approx(1.0)
    assert info["backend"] == "sector-mpo"


def test_symmetric_tdvp_accepts_per_site_scalar_sector_tables():
    nsites = 2
    phys_dim = 2
    zero_h = MPO(
        [np.zeros((1, 1, phys_dim, phys_dim), dtype=complex) for _ in range(nsites)],
        homogenous=False,
    )
    engine = SymmetricTDVP(
        zero_h,
        local_sectors=[[0, 1], [0, 2]],
        target_sector=2,
        max_bond=phys_dim**nsites,
    )

    mask = engine.sector_mask((phys_dim, phys_dim))

    expected = np.array([[False, True], [False, False]])
    np.testing.assert_array_equal(mask, expected)


def test_spatial_fermion_number_sz_sector_helper():
    assert spatial_fermion_number_sz_sectors() == [(0, 0), (1, 1), (1, -1), (2, 0)]


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
