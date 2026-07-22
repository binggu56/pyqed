import numpy as np
import pytest
import sympy as sp
from scipy.integrate import solve_ivp as scipy_solve_ivp

from pyqed import pauli
import pyqed.heom.deom as heom_deom
from pyqed.heom import Bath, HEOM, prony
from pyqed.heom.deom import (
    decompose_spectrum_pade,
    fit_spectrum_prony,
    gamma_smolyak_weights,
    generate_time_dense,
    generate_smolyak_keys,
    native_rhs_total,
    smolyak_hierarchy_score,
)
from pyqed.superoperator import liouvillian, operator_to_superoperator


def test_smolyak_hierarchy_keys_are_downward_closed():
    keys = generate_smolyak_keys(lmax=2, nexp=3)
    key_set = {tuple(key) for key in keys}

    assert len(keys) == 13
    assert tuple(keys[0]) == (0, 0, 0)
    assert all(smolyak_hierarchy_score(key) <= 2 for key in keys)

    for key in keys:
        for pos, occupation in enumerate(key):
            if occupation == 0:
                continue
            lower = key.copy()
            lower[pos] -= 1
            assert tuple(lower) in key_set


def test_weighted_smolyak_penalizes_fast_gamma():
    weights = gamma_smolyak_weights(np.array([1.0, 6.3, 19.5]))
    keys = generate_smolyak_keys(lmax=4, nexp=3, weights=weights)
    key_set = {tuple(key) for key in keys}

    np.testing.assert_array_equal(weights, np.array([1, 3, 5]))
    assert (15, 0, 0) in key_set
    assert (0, 1, 0) in key_set
    assert (0, 0, 1) not in key_set
    assert all(smolyak_hierarchy_score(key, weights) <= 4 for key in keys)


def test_gamma_smolyak_weights_ignore_pure_oscillation_frequency():
    weights = gamma_smolyak_weights(np.array([0.0 + 10.0j, 0.2 + 4.0j, 3.2 + 1.0j]))

    np.testing.assert_array_equal(weights, np.array([1, 1, 5]))


def test_prony_decomposition_accepts_nexp_and_can_return_bath(capsys):
    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})

    bath = prony(
        spectrum,
        w,
        beta=1.0,
        nexp=2,
        scale=20,
        n=80,
        npsd=4,
        as_bath=True,
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert isinstance(bath, Bath)
    assert len(bath.expn) == 2
    assert bath.etal.shape == bath.expn.shape


def test_prony_decomposition_does_not_mutate_nind_list():
    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    nind = [1, 1]

    etal, etar, etaa, expn = fit_spectrum_prony(
        spectrum,
        w,
        beta=1.0,
        nind=nind,
        scale=20,
        n=80,
        npsd=4,
    )

    assert nind == [1, 1]
    assert len(expn) == 2
    assert etal.shape == etar.shape == etaa.shape == expn.shape


def test_prony_handles_subohmic_nonrational_spectrum():
    w = sp.symbols("omega", real=True)
    spectrum = 0.01 * w**0.5 * sp.exp(-w)

    bath = prony(
        spectrum,
        w,
        beta=5.0,
        nexp=5,
        scale=8,
        n=160,
        npsd=4,
        n_omega=800,
        as_bath=True,
    )

    assert isinstance(bath, Bath)
    assert 1 <= len(bath.expn) <= 5
    assert np.all(np.isfinite(bath.expn))
    assert np.all(np.real(bath.expn) > 0.0)


def test_prony_hybrid_fit_for_subohmic_nonrational_spectrum():
    w = sp.symbols("omega", real=True)
    spectrum = 0.01 * w**0.5 * sp.exp(-w)

    etal, etar, etaa, expn = prony(
        spectrum,
        w,
        beta=5.0,
        nexp=5,
        scale=8,
        n=160,
        npsd=4,
        n_omega=800,
        fit="hybrid",
        refine_n_omega=64,
        refine_max_nfev=20,
    )

    assert 1 <= len(expn) <= 5
    assert etal.shape == etar.shape == etaa.shape == expn.shape
    assert np.all(np.isfinite(etal))
    assert np.all(np.isfinite(expn))
    assert np.all(np.real(expn) > 0.0)


def test_direct_exponential_bath_hierarchy_liouvillian_lmax_zero_matches_unitary_block():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.25 * sz
    bath = Bath.from_exponential_terms([1.7], [0.2], mode=[0])
    solver = HEOM(
        system=hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=0,
    )

    liouvillian = solver.hierarchy_liouvillian()
    expected = (-1j * operator_to_superoperator(hamiltonian)).toarray()

    assert solver.nmax == 1
    np.testing.assert_allclose(liouvillian, expected, atol=1.0e-12)


def test_direct_exponential_heom_accepts_custom_base_liouvillian():
    _, _sx, _, sz = pauli()
    base = np.diag([0.0, -0.1, -0.2, -0.3]).astype(np.complex128)
    bath = Bath.from_exponential_terms([1.3], [0.4], mode=[0])
    solver = HEOM(
        system=np.zeros((2, 2), dtype=np.complex128),
        system_liouvillian=base,
        bath=bath,
        coupling=sz,
        lmax=1,
    )

    liouvillian = solver.hierarchy_liouvillian()

    assert solver.nmax == 2
    np.testing.assert_allclose(liouvillian[:4, :4], base, atol=1.0e-12)
    assert np.linalg.norm(liouvillian[:4, 4:]) > 0.0
    assert np.linalg.norm(liouvillian[4:, :4]) > 0.0


def test_heom_correlation_2p_1t_uses_zero_tier_propagation():
    _, _sx, _, sz = pauli()
    base = np.diag([0.0, -0.2, -0.5, -0.7]).astype(np.complex128)
    bath = Bath.from_exponential_terms([1.3], [0.4], mode=[0])
    solver = HEOM(
        system=np.zeros((2, 2), dtype=np.complex128),
        system_liouvillian=base,
        bath=bath,
        coupling=sz,
        lmax=0,
    )
    rho0 = np.array([[0.7, 0.2], [0.2, 0.3]], dtype=np.complex128)
    a_op = np.array([[0.1, 0.4], [0.5, -0.2]], dtype=np.complex128)
    b_op = np.array([[0.3, -0.1], [0.2, 0.6]], dtype=np.complex128)
    distances = np.array([0.0, 0.4, 1.2])

    values = solver.correlation_2p_1t(
        rho0,
        [a_op, b_op],
        distances,
    )

    vector0 = (b_op @ rho0).reshape(-1)
    trace_row = a_op.T.reshape(-1)
    expected = np.array(
        [trace_row @ (np.exp(np.diag(base) * distance) * vector0) for distance in distances]
    )
    np.testing.assert_allclose(values, expected, atol=1.0e-12)


def test_heom_steady_state_solves_trace_constrained_hierarchy():
    _, _sx, _, sz = pauli()
    collapse = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=np.complex128)
    base = liouvillian(np.zeros((2, 2), dtype=np.complex128), [collapse]).toarray()
    bath = Bath.from_exponential_terms([1.3], [0.1], mode=[0])
    solver = HEOM(
        system=np.zeros((2, 2), dtype=np.complex128),
        system_liouvillian=base,
        bath=bath,
        coupling=sz,
        lmax=1,
    )

    hierarchy = solver.steady_state()
    vector = hierarchy.reshape(-1)
    generator = solver.hierarchy_liouvillian()

    assert hierarchy.shape == (2, 2, 2)
    np.testing.assert_allclose(np.trace(hierarchy[0]), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(generator @ vector, np.zeros_like(vector), atol=1.0e-10)
    assert solver.steady_state_residual < 1.0e-10
    assert solver.steady_state_trace_error < 1.0e-12


def test_heom_correlation_accepts_full_steady_hierarchy():
    _, _sx, _, sz = pauli()
    base = np.diag([0.0, -0.2, -0.5, -0.7]).astype(np.complex128)
    bath = Bath.from_exponential_terms([1.3], [0.4], mode=[0])
    solver = HEOM(
        system=np.zeros((2, 2), dtype=np.complex128),
        system_liouvillian=base,
        bath=bath,
        coupling=sz,
        lmax=0,
    )
    rho0 = solver.steady_state()
    a_op = np.array([[0.1, 0.4], [0.5, -0.2]], dtype=np.complex128)
    b_op = np.array([[0.3, -0.1], [0.2, 0.6]], dtype=np.complex128)
    distances = np.array([0.0, 0.4, 1.2])

    full = solver.correlation_2p_1t(
        rho0,
        [a_op, b_op],
        distances,
    )
    reduced = solver.correlation_2p_1t(
        rho0[0],
        [a_op, b_op],
        distances,
    )

    np.testing.assert_allclose(full, reduced, atol=1.0e-12)


def test_heom_uses_hierarchy_liouvillian_name_only():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx
    bath = Bath.from_exponential_terms([1.1], [0.2], mode=[0])
    solver = HEOM(
        system=hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=0,
    )

    assert hasattr(solver, "hierarchy_liouvillian")
    assert not hasattr(solver, "generate_propagator")
    assert not hasattr(solver, "gen_generate_propgator")
    liouvillian = solver.hierarchy_liouvillian()

    assert liouvillian.shape == (4, 4)
    np.testing.assert_allclose(solver.hierarchy_liouvillian_matrix, liouvillian)


def test_user_weighted_smolyak_keys_are_anisotropic():
    weights = np.array([1, 2, 3])
    keys = generate_smolyak_keys(lmax=3, nexp=3, weights=weights)
    key_set = {tuple(key) for key in keys}

    assert (7, 0, 0) in key_set
    assert (0, 1, 0) in key_set
    assert (0, 0, 1) in key_set
    assert (0, 2, 0) not in key_set
    assert (0, 0, 2) not in key_set
    assert all(smolyak_hierarchy_score(key, weights) <= 3 for key in keys)


def test_smolyak_heom_spin_boson_short_run_preserves_trace():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(
        system=hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=2,
        hierarchy_truncation="smolyak",
    )

    _, rhos = solver.run(rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    traces = np.array([np.trace(rho) for rho in rhos])

    assert solver.nmax == 13
    np.testing.assert_allclose(traces, 1.0, atol=1.0e-12)


def test_weighted_smolyak_heom_spin_boson_short_run_preserves_trace():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(
        hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=3,
        hierarchy_truncation="weighted-smolyak",
        hierarchy_weights=[1, 2, 3],
    )

    _, rhos = solver.run(rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    traces = np.array([np.trace(rho) for rho in rhos])

    np.testing.assert_array_equal(solver.hierarchy_weights, np.array([1, 2, 3]))
    assert all(smolyak_hierarchy_score(key, solver.hierarchy_weights) <= 3 for key in solver.keys)
    np.testing.assert_allclose(traces, 1.0, atol=1.0e-12)


def test_weighted_smolyak_defaults_to_gamma_weights():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(
        hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=4,
        hierarchy_truncation="weighted-smolyak",
    )

    _, rhos = solver.run(rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    traces = np.array([np.trace(rho) for rho in rhos])

    np.testing.assert_array_equal(solver.hierarchy_weights, np.array([1, 3, 5]))
    assert all(smolyak_hierarchy_score(key, solver.hierarchy_weights) <= 4 for key in solver.keys)
    np.testing.assert_allclose(traces, 1.0, atol=1.0e-12)


def test_weighted_smolyak_accepts_gamma_weight_alias():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(
        hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=4,
        hierarchy_truncation="smolyak",
        hierarchy_weights="gamma",
    )

    solver.run(rho0=rho0.copy(), dt=0.01, nt=1, method="rk4", p1=None)

    np.testing.assert_array_equal(solver.hierarchy_weights, np.array([1, 3, 5]))


def test_gamma_smolyak_heom_spin_boson_short_run_preserves_trace():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(
        system=hamiltonian,
        bath=bath,
        coupling=sz,
        lmax=4,
        hierarchy_truncation="gamma-smolyak",
    )

    _, rhos = solver.run(rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    traces = np.array([np.trace(rho) for rho in rhos])

    np.testing.assert_array_equal(solver.hierarchy_weights, np.array([1, 3, 5]))
    np.testing.assert_allclose(traces, 1.0, atol=1.0e-12)


def test_heom_constructor_defaults_fill_static_operators():
    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    bath = Bath(
        [spectrum],
        w,
        [1.0],
        [2],
        [0, 0, 0],
        [decompose_spectrum_pade],
    )
    solver = HEOM(hamiltonian, bath=bath, coupling=sz, lmax=1)

    solver.run(rho0=rho0.copy(), dt=0.01, nt=1, method="rk4")

    assert len(solver.coupling) == 1
    np.testing.assert_allclose(solver.system_dipole, 0.0)
    np.testing.assert_allclose(solver.coupling_dipole[0], 0.0)


def test_native_total_heom_matches_python_fallback():
    if heom_deom._get_heom_cpp() is None:
        pytest.skip("optional HEOM C++ extension is not built")

    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    def make_solver():
        bath = Bath(
            [spectrum],
            w,
            [1.0],
            [2],
            [0, 0, 0],
            [decompose_spectrum_pade],
        )
        return HEOM(
            system=hamiltonian,
            bath=bath,
            coupling=sz,
            lmax=2,
            hierarchy_truncation="total",
        )

    native_solver = make_solver()
    _, native_rhos = native_solver.run(
        rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    assert native_solver._use_native_total

    native_module = heom_deom._heom_cpp
    heom_deom._heom_cpp = None
    try:
        fallback_solver = make_solver()
        _, fallback_rhos = fallback_solver.run(
            rho0=rho0.copy(), dt=0.01, nt=3, method="rk4", p1=None)
    finally:
        heom_deom._heom_cpp = native_module

    assert not fallback_solver._use_native_total
    fallback_final = (
        fallback_rhos[-1].toarray()
        if hasattr(fallback_rhos[-1], "toarray")
        else np.asarray(fallback_rhos[-1])
    )
    np.testing.assert_allclose(native_rhos[-1], fallback_final, atol=1.0e-12)


def test_native_dop853_dense_output_and_pulse_match_scipy():
    heom_cpp = heom_deom._get_heom_cpp()
    if heom_cpp is None or not hasattr(heom_cpp, "dop853_by_index"):
        pytest.skip("native HEOM DOP853 extension is not built")

    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    bath = Bath.from_exponential_terms(
        expn=np.array([0.35 + 0.2j, 1.7 - 0.1j]),
        etal=np.array([0.08 + 0.02j, 0.03 - 0.01j]),
    )
    solver = HEOM(
        system=hamiltonian,
        system_dipole=0.2 * sz,
        pulse_system_func=lambda t: np.sin(1.3 * t),
        bath=bath,
        coupling=sz,
        lmax=2,
    )
    t, native_rhos = solver.run(
        rho0=rho0,
        dt=0.01,
        nt=100,
        method="dop853",
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    shape = (solver.nmax, solver.nsys, solver.nsys)
    initial = np.zeros(shape, dtype=np.complex128)
    initial[0] = rho0

    def rhs(time, state):
        system_t, coupling_t = generate_time_dense(
            solver.system,
            solver.system_dipole,
            solver.pulse_system_func,
            solver.coupling,
            solver.coupling_dipole,
            solver.pulse_coupling_func,
            time,
        )
        return native_rhs_total(
            state.reshape(shape),
            solver._native_keys,
            solver._native_minus_index,
            solver._native_plus_index,
            solver._native_bath_list,
            solver._native_mode,
            system_t,
            coupling_t,
        ).reshape(-1)

    reference = scipy_solve_ivp(
        rhs,
        (t[0], t[-1]),
        initial.reshape(-1),
        method="DOP853",
        t_eval=t,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    reference_rhos = reference.y.T.reshape(len(t), *shape)[:, 0]

    assert solver.method == "dop853"
    assert solver.success
    assert solver.n_steps > 0
    assert solver.nfev == reference.nfev
    np.testing.assert_allclose(native_rhos, reference_rhos, atol=2.0e-12)
    np.testing.assert_allclose(solver.ddos.reshape(-1), reference.y[:, -1], atol=2.0e-12)


def test_native_dop853_uses_fast_edge_tables_by_default():
    heom_cpp = heom_deom._get_heom_cpp()
    if heom_cpp is None or not hasattr(heom_cpp, "dop853_by_index"):
        pytest.skip("native HEOM DOP853 extension is not built")

    _, sx, _, sz = pauli()
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0
    expn = np.array([0.35 + 0.05j, 0.7 + 0.12j, 1.3 - 0.08j], dtype=np.complex128)
    etal = np.array([0.04 + 0.01j, 0.025 - 0.006j, 0.012 + 0.003j], dtype=np.complex128)

    def make_solver():
        return HEOM(
            system=-0.5 * sx - 0.2 * sz,
            bath=Bath.from_exponential_terms(expn=expn, etal=etal),
            coupling=sz,
            lmax=3,
            hierarchy_truncation="total",
        )

    default_solver = make_solver()
    t_eval, fast_rhos = default_solver.run(
        rho0=rho0.copy(),
        dt=0.002,
        nt=3,
        method="dop853",
        rtol=1.0e-10,
        atol=1.0e-12,
        threads=None,
    )

    assert default_solver.method == "dop853"
    assert default_solver.threads == 1
    assert default_solver._use_native_total
    assert default_solver._native_edge_tables is not None
    assert default_solver._native_edge_tables[1].size > 0
    assert default_solver._native_edge_tables[6].size > 0

    legacy_solver = make_solver()
    legacy_solver.check_()
    legacy_solver.init_()
    ddos = np.zeros((legacy_solver.nmax, legacy_solver.nsys, legacy_solver.nsys), dtype=np.complex128)
    ddos[0] = rho0
    expn_native, etal_native, etar_native, etaa_native = legacy_solver._native_bath_list
    legacy_rhos, legacy_nfev, _, _ = heom_cpp.dop853_by_index(
        ddos,
        legacy_solver._native_keys,
        legacy_solver._native_minus_index,
        legacy_solver._native_plus_index,
        expn_native,
        etal_native,
        etar_native,
        etaa_native,
        legacy_solver._native_mode,
        np.ascontiguousarray(legacy_solver.system, dtype=np.complex128),
        np.ascontiguousarray(legacy_solver.system_dipole, dtype=np.complex128),
        np.ascontiguousarray(legacy_solver.coupling, dtype=np.complex128),
        np.ascontiguousarray(legacy_solver.coupling_dipole, dtype=np.complex128),
        None,
        None,
        t_eval,
        1.0e-10,
        1.0e-12,
        1,
    )

    assert default_solver.nfev == legacy_nfev
    np.testing.assert_allclose(fast_rhos, legacy_rhos, atol=1.0e-13)
    np.testing.assert_allclose(default_solver.ddos, ddos, atol=1.0e-13)


def test_native_dop853_t_span_returns_accepted_step_output():
    heom_cpp = heom_deom._get_heom_cpp()
    if (
        heom_cpp is None
        or not hasattr(heom_cpp, "dop853_by_index")
        or not hasattr(heom_cpp, "dop853_adaptive_by_index")
    ):
        pytest.skip("native HEOM DOP853 extension with adaptive output is not built")

    _, sx, _, sz = pauli()
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0
    bath = Bath.from_exponential_terms(
        expn=np.array([0.35 + 0.05j, 0.7 + 0.12j, 1.3 - 0.08j], dtype=np.complex128),
        etal=np.array([0.04 + 0.01j, 0.025 - 0.006j, 0.012 + 0.003j], dtype=np.complex128),
    )

    def make_solver():
        return HEOM(
            system=-0.5 * sx - 0.2 * sz,
            bath=bath,
            coupling=sz,
            lmax=3,
            hierarchy_truncation="total",
        )

    adaptive_solver = make_solver()
    t_adaptive, adaptive_rhos = adaptive_solver.run(
        rho0=rho0.copy(),
        method="dop853",
        t_span=(0.0, 0.006),
        rtol=1.0e-10,
        atol=1.0e-12,
        threads=1,
    )
    fixed_solver = make_solver()
    t_fixed, fixed_rhos = fixed_solver.run(
        rho0=rho0.copy(),
        method="dop853",
        t_eval=np.array([0.0, 0.006]),
        rtol=1.0e-10,
        atol=1.0e-12,
        threads=1,
    )

    assert adaptive_solver.method == "dop853"
    assert len(t_adaptive) == adaptive_solver.n_steps + 1
    assert len(t_adaptive) >= 2
    assert t_adaptive[0] == pytest.approx(0.0)
    assert t_adaptive[-1] == pytest.approx(0.006)
    assert np.all(np.diff(t_adaptive) > 0.0)
    np.testing.assert_allclose(t_fixed, np.array([0.0, 0.006]))
    np.testing.assert_allclose(adaptive_rhos[-1], fixed_rhos[-1], atol=1.0e-13)


def test_native_total_integrator_options_agree_on_short_run():
    if heom_deom._get_heom_cpp() is None:
        pytest.skip("optional HEOM C++ extension is not built")

    _, sx, _, sz = pauli()
    hamiltonian = -0.5 * sx - 0.5 * sz
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0

    w, lam, gam = sp.symbols("omega lambda gamma", real=True)
    spectrum = (2 * lam * gam * w / (gam**2 + w**2)).subs({lam: 0.2, gam: 1.0})
    def make_solver():
        bath = Bath(
            [spectrum],
            w,
            [1.0],
            [2],
            [0, 0, 0],
            [decompose_spectrum_pade],
        )
        return HEOM(
            system=hamiltonian,
            bath=bath,
            coupling=sz,
            lmax=2,
            hierarchy_truncation="total",
        )

    _, rk4_rhos = make_solver().run(rho0=rho0.copy(), dt=0.002, nt=5, method="rk4", p1=None)
    dop_solver = make_solver()
    _, dop_rhos = dop_solver.run(
        rho0=rho0.copy(),
        dt=0.002,
        nt=5,
        method="dop853",
        rtol=1.0e-10,
        atol=1.0e-12,
        p1=None,
    )
    krylov_solver = make_solver()
    _, krylov_rhos = krylov_solver.run(
        rho0=rho0.copy(),
        dt=0.002,
        nt=5,
        method="krylov",
        krylov_dim=20,
        p1=None,
    )

    assert dop_solver.success
    assert krylov_solver.success
    assert dop_solver.n_steps > 0
    np.testing.assert_allclose(np.trace(dop_rhos[-1]), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(np.trace(krylov_rhos[-1]), 1.0, atol=1.0e-12)
    np.testing.assert_allclose(dop_rhos, rk4_rhos, atol=1.0e-8)
    np.testing.assert_allclose(krylov_rhos, dop_rhos, atol=1.0e-8)


def test_native_dop853_threads_match_serial():
    if heom_deom._get_heom_cpp() is None:
        pytest.skip("optional HEOM C++ extension is not built")

    _, sx, _, sz = pauli()
    rho0 = np.zeros((2, 2), dtype=np.complex128)
    rho0[1, 1] = 1.0
    expn = np.linspace(0.4, 1.3, 10, dtype=np.float64).astype(np.complex128)
    etal = (0.002 * np.exp(-np.arange(10))).astype(np.complex128)

    def make_solver():
        return HEOM(
            system=-0.5 * sx - 0.2 * sz,
            bath=Bath.from_exponential_terms(expn=expn, etal=etal),
            coupling=sz,
            lmax=6,
            hierarchy_truncation="total",
        )

    serial = make_solver()
    _, serial_rhos = serial.run(
        rho0=rho0.copy(),
        dt=1.0e-4,
        nt=1,
        method="dop853",
        rtol=1.0e-9,
        atol=1.0e-11,
        threads=1,
    )
    parallel = make_solver()
    _, parallel_rhos = parallel.run(
        rho0=rho0.copy(),
        dt=1.0e-4,
        nt=1,
        method="dop853",
        rtol=1.0e-9,
        atol=1.0e-11,
        threads=2,
    )

    assert serial.nmax > 4096
    assert parallel.threads == 2
    np.testing.assert_allclose(parallel_rhos, serial_rhos, atol=1.0e-13)
