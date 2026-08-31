import numpy as np
import pytest

from pyqed.models.heisenberg import Heisenberg
from pyqed.mps import DMRG
from pyqed.mps import cpp_davidson
from pyqed.mps.mps import (
    DenseLocalProblem,
    MovingEnvironment,
    coarse_grain_MPO,
    coarse_grain_MPS,
    contract_from_left,
    contract_from_right,
    initial_E,
    initial_F,
)


def test_dense_dmrg_uses_moving_environment_with_old_path_parity():
    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()

    common = dict(
        D=4,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
    )
    moved = DMRG(hamiltonian, **common).run()
    direct = DMRG(
        hamiltonian,
        **{
            **common,
            "abelian_matvec_options": {"moving_environment": False},
        },
    ).run()

    assert np.allclose(moved.e_tot, direct.e_tot, atol=1.0e-12)

    profile = moved.sweep_history[-1]["environment_profile"]["moving_environment"]
    assert profile["dense_local_operator_builds"] >= 1
    assert profile["dense_solve_local_calls"] >= 1

    last_update = moved.sweep_history[-1]["updates"][-1]
    matvec_profile = last_update["matvec_profile"]
    assert matvec_profile["dominant_path"] in {
        "dense_numpy_tensordot",
        "dense_cpp_matvec",
    }
    assert matvec_profile["local_solver"]["kind"] in {"eigsh", "dense_fallback"}


def test_dense_dmrg_requires_mpo_and_mps_owners():
    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()

    with pytest.raises(TypeError, match="MPO Hamiltonian"):
        DMRG(hamiltonian.factors, D=4, init_guess=initial)
    with pytest.raises(TypeError, match="MPS initial guess"):
        DMRG(hamiltonian, D=4, init_guess=initial.factors)


def test_dense_cpp_matvec_matches_numpy_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.dense_two_site_matvec is None
    ):
        pytest.skip("C++ dense matvec backend is unavailable")

    rng = np.random.default_rng(7)
    e = rng.normal(size=(2, 3, 3)) + 1j * rng.normal(size=(2, 3, 3))
    w = rng.normal(size=(2, 5, 4, 4)) + 1j * rng.normal(size=(2, 5, 4, 4))
    f = rng.normal(size=(5, 6, 6)) + 1j * rng.normal(size=(5, 6, 6))
    v = rng.normal(size=(3 * 4 * 6,)) + 1j * rng.normal(size=(3 * 4 * 6,))

    py_op = DenseLocalProblem(
        e,
        w,
        f,
        matvec_options={"moving_environment_dense_cpp_matvec": False},
    )
    cpp_op = DenseLocalProblem(
        e,
        w,
        f,
        matvec_options={"moving_environment_dense_cpp_matvec": True},
    )

    np.testing.assert_allclose(cpp_op.matvec(v), py_op.matvec(v), atol=1.0e-10)
    assert cpp_op.profile_summary()["dominant_path"] == "dense_cpp_matvec"


def test_dense_cpp_openmp_matvec_matches_serial_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseDavidsonWorkspace is None
        or not callable(cpp_davidson.openmp_available)
        or not cpp_davidson.openmp_available()
    ):
        pytest.skip("C++ OpenMP DMRG backend is unavailable")

    rng = np.random.default_rng(29)
    e = rng.normal(size=(3, 7, 7)) + 1j * rng.normal(size=(3, 7, 7))
    w = rng.normal(size=(3, 4, 4, 4)) + 1j * rng.normal(size=(3, 4, 4, 4))
    f = rng.normal(size=(4, 8, 8)) + 1j * rng.normal(size=(4, 8, 8))
    v = rng.normal(size=7 * 4 * 8) + 1j * rng.normal(size=7 * 4 * 8)
    previous = cpp_davidson.get_num_threads()
    try:
        workspace = cpp_davidson.DenseDavidsonWorkspace()
        workspace.bind(e, w, f)
        cpp_davidson.set_num_threads(1)
        serial = workspace.matvec(v, "loop")
        cpp_davidson.set_num_threads(2)
        parallel = workspace.matvec(v, "openmp")
        np.testing.assert_allclose(parallel, serial, atol=1.0e-12)
        stats = dict(workspace.stats())
        assert stats["last_matvec_backend"] == "openmp"
        assert stats["openmp_matvec_calls"] == 1
        assert stats["openmp"]["threads"] == 2
    finally:
        cpp_davidson.set_num_threads(previous)


def test_dmrg_n_threads_configures_native_and_numba_backends():
    model = Heisenberg(L=4)
    native_previous = (
        cpp_davidson.get_num_threads()
        if callable(cpp_davidson.get_num_threads)
        else None
    )
    from numba import get_num_threads, set_num_threads

    numba_previous = get_num_threads()
    try:
        dmrg = DMRG(
            model.build_H_mpo(),
            D=4,
            init_guess=model.build_neel_state(),
            nsweeps=1,
            not_conv_err=False,
            performance="dense",
            n_threads=2,
        )
        assert dmrg.n_threads == 2
        assert dmrg.threading_info["numba"]["threads"] == 2
        native = dmrg.threading_info["native_openmp"]
        if native.get("available", False):
            assert native["threads"] == 2
            assert (
                dmrg.abelian_matvec_options[
                    "moving_environment_dense_cpp_davidson_backend"
                ]
                == "openmp"
            )
            dmrg.run()
            update = dmrg.sweep_history[-1]["updates"][-1]
            assert update["matvec_profile"]["dominant_path"].endswith(
                "_openmp"
            )
    finally:
        set_num_threads(numba_previous)
        if native_previous is not None:
            cpp_davidson.set_num_threads(native_previous)


def test_dmrg_rejects_invalid_thread_count():
    model = Heisenberg(L=4)
    with pytest.raises(ValueError, match="n_threads"):
        DMRG(
            model.build_H_mpo(),
            D=4,
            init_guess=model.build_neel_state(),
            n_threads=0,
        )


def test_dense_cpp_tensor_primitives_match_numpy_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.dense_coarse_grain_mpo is None
        or cpp_davidson.dense_coarse_grain_mps is None
        or cpp_davidson.dense_environment_update_left is None
        or cpp_davidson.dense_environment_update_right is None
    ):
        pytest.skip("C++ dense tensor primitives are unavailable")

    rng = np.random.default_rng(11)

    def rand(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    w1 = rand((2, 3, 2, 2))
    w2 = rand((3, 4, 3, 3))
    np.testing.assert_allclose(
        cpp_davidson.dense_coarse_grain_mpo(w1, w2),
        coarse_grain_MPO(w1, w2),
        atol=1.0e-10,
    )

    a = rand((2, 3, 5))
    b = rand((5, 2, 4))
    np.testing.assert_allclose(
        cpp_davidson.dense_coarse_grain_mps(a, b),
        coarse_grain_MPS(a, b),
        atol=1.0e-10,
    )

    w = rand((2, 3, 4, 5))
    left_a = rand((6, 4, 7))
    left_b = rand((8, 5, 9))
    left_e = rand((2, 6, 8))
    np.testing.assert_allclose(
        cpp_davidson.dense_environment_update_left(w, left_a, left_e, left_b),
        contract_from_left(w, left_a, left_e, left_b),
        atol=1.0e-10,
    )

    right_a = rand((6, 4, 7))
    right_b = rand((8, 5, 9))
    right_f = rand((3, 7, 9))
    np.testing.assert_allclose(
        cpp_davidson.dense_environment_update_right(w, right_a, right_f, right_b),
        contract_from_right(w, right_a, right_f, right_b),
        atol=1.0e-10,
    )


def test_dense_cpp_environment_update_is_explicit_opt_in_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.dense_environment_update_left is None
    ):
        pytest.skip("C++ dense environment update is unavailable")

    rng = np.random.default_rng(13)

    def rand(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    w = rand((2, 3, 4, 5))
    a = rand((6, 4, 7))
    b = rand((8, 5, 9))
    e = rand((2, 6, 8))

    env_default = MovingEnvironment(
        matvec_options={"moving_environment_dense_cpp_davidson": True}
    )
    np.testing.assert_allclose(
        env_default.compiled_backend.update_left_environment(w, a, e, b),
        contract_from_left(w, a, e, b),
        atol=1.0e-10,
    )
    assert env_default.moving_profile_stats["dense_cpp_environment_update_calls"] == 0

    env_opt_in = MovingEnvironment(
        matvec_options={"moving_environment_dense_cpp_environment_update": True}
    )
    np.testing.assert_allclose(
        env_opt_in.compiled_backend.update_left_environment(w, a, e, b),
        contract_from_left(w, a, e, b),
        atol=1.0e-10,
    )
    assert env_opt_in.moving_profile_stats["dense_cpp_environment_update_calls"] == 1


def test_dense_cpp_davidson_workspace_matches_exact_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseDavidsonWorkspace is None
    ):
        pytest.skip("C++ dense Davidson workspace is unavailable")

    rng = np.random.default_rng(17)

    def hermitian(n):
        mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return 0.5 * (mat + mat.conj().T)

    e = hermitian(3)[None, :, :]
    w = hermitian(2)[None, None, :, :]
    f = hermitian(4)[None, :, :]
    nloc = 3 * 2 * 4
    v0 = rng.normal(size=nloc) + 1j * rng.normal(size=nloc)

    py_op = DenseLocalProblem(e, w, f)
    h_dense = np.column_stack(
        [
            py_op.matvec(np.eye(nloc, dtype=np.complex128)[:, col])
            for col in range(nloc)
        ]
    )
    h_dense = 0.5 * (h_dense + h_dense.T.conj())
    evals, _ = np.linalg.eigh(h_dense)

    workspace = cpp_davidson.DenseDavidsonWorkspace()
    workspace.bind(e, w, f)
    np.testing.assert_allclose(workspace.diagonal(), np.diag(h_dense), atol=1.0e-12)
    result = workspace.solve_bound(v0, 1.0e-10, 200, 32, True, "blas")
    assert result["accepted"]
    assert result["backend"] in {"blas", "loop"}
    assert abs(float(result["energy"]) - evals[0]) < 1.0e-8

    cpp_op = DenseLocalProblem(
        e,
        w,
        f,
        matvec_options={
            "moving_environment_dense_cpp_davidson": True,
            "moving_environment_dense_cpp_davidson_accept_unconverged": True,
        },
    )
    energies, _ = cpp_op.solve(v0, 1, tol=1.0e-10, maxiter=200)
    assert abs(float(energies[0]) - evals[0]) < 1.0e-8
    assert cpp_op.profile_summary()["local_solver"]["kind"] == "cpp_dense_davidson"


def test_dense_cpp_block_davidson_workspace_matches_exact_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseDavidsonWorkspace is None
        or not hasattr(cpp_davidson.DenseDavidsonWorkspace(), "solve_bound_block")
    ):
        pytest.skip("C++ dense block Davidson workspace is unavailable")

    rng = np.random.default_rng(18)

    def hermitian(n):
        mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return 0.5 * (mat + mat.conj().T)

    e = hermitian(3)[None, :, :]
    w = hermitian(2)[None, None, :, :]
    f = hermitian(4)[None, :, :]
    nloc = 3 * 2 * 4
    v0 = rng.normal(size=nloc) + 1j * rng.normal(size=nloc)

    py_op = DenseLocalProblem(e, w, f)
    h_dense = np.column_stack(
        [
            py_op.matvec(np.eye(nloc, dtype=np.complex128)[:, col])
            for col in range(nloc)
        ]
    )
    h_dense = 0.5 * (h_dense + h_dense.T.conj())
    evals, _ = np.linalg.eigh(h_dense)

    workspace = cpp_davidson.DenseDavidsonWorkspace()
    workspace.bind(e, w, f)
    result = workspace.solve_bound_block(v0, 1.0e-10, 200, 32, True, "blas", 3)
    assert result["accepted"]
    assert result["kind"] == "cpp_dense_block_davidson"
    assert result["block_davidson"]
    assert int(result["block_size"]) == 3
    assert abs(float(result["energy"]) - evals[0]) < 1.0e-8
    stats = workspace.stats()
    assert stats["block_solve_calls"] == 1
    assert stats["batched_matvec_calls"] >= 1
    assert stats["batched_matvec_vectors"] >= int(result["matvec_calls"])

    cpp_op = DenseLocalProblem(
        e,
        w,
        f,
        matvec_options={
            "moving_environment_dense_cpp_davidson": True,
            "moving_environment_dense_cpp_block_davidson": True,
            "moving_environment_dense_cpp_block_davidson_size": 3,
            "moving_environment_dense_cpp_davidson_accept_unconverged": True,
        },
    )
    energies, _ = cpp_op.solve(v0, 1, tol=1.0e-10, maxiter=200)
    assert abs(float(energies[0]) - evals[0]) < 1.0e-8
    assert cpp_op.profile_summary()["local_solver"]["kind"] == (
        "cpp_dense_block_davidson"
    )


def test_dense_cpp_sweep_workspace_reuses_bond_records_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseSweepWorkspace is None
    ):
        pytest.skip("C++ dense sweep workspace is unavailable")

    rng = np.random.default_rng(19)

    def hermitian(n):
        mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return 0.5 * (mat + mat.conj().T)

    e = hermitian(2)[None, :, :]
    w = hermitian(2)[None, None, :, :]
    f = hermitian(3)[None, :, :]
    nloc = 2 * 2 * 3
    v0 = rng.normal(size=nloc) + 1j * rng.normal(size=nloc)

    py_op = DenseLocalProblem(e, w, f)
    h_dense = np.column_stack(
        [
            py_op.matvec(np.eye(nloc, dtype=np.complex128)[:, col])
            for col in range(nloc)
        ]
    )
    h_dense = 0.5 * (h_dense + h_dense.T.conj())
    evals, _ = np.linalg.eigh(h_dense)

    owner = cpp_davidson.DenseSweepWorkspace()
    owner.bind("bond:0", e, w, f)
    first = owner.solve_bound("bond:0", v0, 1.0e-10, 200, 24, True, "blas")
    owner.bind_boundaries("bond:0", e, f)
    second = owner.solve_bound("bond:0", v0, 1.0e-10, 200, 24, True, "blas")

    assert first["accepted"]
    assert second["accepted"]
    assert bool(second["workspace_reused"])
    assert abs(float(second["energy"]) - evals[0]) < 1.0e-8
    assert owner.stats()["records"] == 1
    assert owner.stats()["boundary_bind_calls"] == 1


def test_dense_cpp_sweep_workspace_two_site_solve_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseSweepWorkspace is None
        or not hasattr(cpp_davidson.DenseSweepWorkspace(), "solve_two_site")
    ):
        pytest.skip("C++ dense two-site workspace solve is unavailable")

    rng = np.random.default_rng(23)

    def hermitian(n):
        mat = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
        return 0.5 * (mat + mat.conj().T)

    w1 = hermitian(2)[None, None, :, :]
    w2 = hermitian(2)[None, None, :, :]
    e = np.eye(2, dtype=np.complex128)[None, :, :]
    f = np.eye(3, dtype=np.complex128)[None, :, :]
    a = rng.normal(size=(2, 2, 4)) + 1j * rng.normal(size=(2, 2, 4))
    b = rng.normal(size=(4, 2, 3)) + 1j * rng.normal(size=(4, 2, 3))
    aa = coarse_grain_MPS(a, b)
    w = coarse_grain_MPO(w1, w2)
    nloc = int(aa.size)

    py_op = DenseLocalProblem(e, w, f)
    h_dense = np.column_stack(
        [
            py_op.matvec(np.eye(nloc, dtype=np.complex128)[:, col])
            for col in range(nloc)
        ]
    )
    h_dense = 0.5 * (h_dense + h_dense.T.conj())
    evals, _ = np.linalg.eigh(h_dense)

    owner = cpp_davidson.DenseSweepWorkspace()
    first = owner.solve_two_site(
        "bond:0",
        e,
        w1,
        w2,
        f,
        a,
        b,
        1.0e-10,
        200,
        24,
        True,
        "blas",
        True,
    )
    second = owner.solve_two_site(
        "bond:0",
        e,
        w1,
        w2,
        f,
        a,
        b,
        1.0e-10,
        200,
        24,
        True,
        "blas",
        True,
    )

    assert first["accepted"]
    assert second["accepted"]
    assert bool(second["two_site_static_w_reused"])
    assert abs(float(first["energy"]) - evals[0]) < 1.0e-8
    assert abs(float(second["energy"]) - evals[0]) < 1.0e-8
    assert owner.stats()["records"] == 1
    assert owner.stats()["two_site_mpo_builds"] == 1
    assert owner.stats()["two_site_mps_builds"] == 2
    assert owner.stats()["two_site_static_w_reuses"] == 1


def test_dense_cpp_workspace_runs_complete_dmrg_half_sweeps_when_available():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseSweepWorkspace is None
        or not hasattr(cpp_davidson.DenseSweepWorkspace(), "dmrg_half_sweep")
    ):
        pytest.skip("C++ dense DMRG sweep controller is unavailable")

    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()
    reference = DMRG(
        hamiltonian,
        D=4,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
        recenter_final=False,
        final_expectation=False,
    ).run()

    owner = cpp_davidson.DenseSweepWorkspace()
    left = owner.dmrg_half_sweep(
        initial.factors,
        hamiltonian.factors,
        initial_E(hamiltonian.factors[0]),
        initial_F(hamiltonian.factors[-1]),
        "lr",
        4,
        1.0e-9,
        5000,
        64,
        False,
        "blas",
        True,
        chain_key="test-dmrg-chain",
    )
    right = owner.dmrg_half_sweep(
        left["factors"],
        hamiltonian.factors,
        initial_E(hamiltonian.factors[0]),
        initial_F(hamiltonian.factors[-1]),
        "rl",
        4,
        1.0e-9,
        5000,
        64,
        False,
        "blas",
        True,
        chain_key="test-dmrg-chain",
    )
    final = owner.dmrg_half_sweep(
        right["factors"],
        hamiltonian.factors,
        initial_E(hamiltonian.factors[0]),
        initial_F(hamiltonian.factors[-1]),
        "lr",
        4,
        1.0e-9,
        5000,
        64,
        False,
        "blas",
        True,
        chain_key="test-dmrg-chain",
    )

    assert len(left["updates"]) == len(initial.factors) - 2
    assert len(left["left_environments"]) == len(initial.factors) - 1
    assert len(left["right_environments"]) == 1
    assert len(right["left_environments"]) == 1
    assert len(right["right_environments"]) == len(initial.factors) - 1
    assert np.allclose(final["energy"], reference.e_tot, atol=1.0e-10)
    stats = owner.stats()
    assert stats["dmrg_half_sweep_calls"] == 3
    assert stats["dmrg_half_sweep_bonds"] == 3 * (len(initial.factors) - 2)
    assert stats["dmrg_chain_installs"] == 1
    assert stats["dmrg_chain_reuses"] == 2
    assert stats["dmrg_environment_chain_reuses"] == 2
    assert stats["dmrg_environment_buffer_reuses"] >= 1


def test_dense_dmrg_can_use_cpp_davidson_workspace_when_enabled():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseDavidsonWorkspace is None
    ):
        pytest.skip("C++ dense Davidson workspace is unavailable")

    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()
    common = dict(
        D=4,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
    )
    reference = DMRG(hamiltonian, **common).run()
    moved = DMRG(
        hamiltonian,
        **{
            **common,
            "abelian_matvec_options": {
                "moving_environment_dense_cpp_davidson": True,
                "moving_environment_dense_cpp_two_site_solve": False,
                "moving_environment_dense_cpp_davidson_backend": "blas",
            },
        },
    ).run()

    assert np.allclose(moved.e_tot, reference.e_tot, atol=1.0e-12)
    profile = moved.sweep_history[-1]["updates"][-1]["matvec_profile"]
    assert profile["dominant_path"] in {
        "dense_cpp_davidson_blas",
        "dense_cpp_davidson_loop",
    }
    assert profile["local_solver"]["kind"] == "cpp_dense_davidson"
    assert profile["operatorless"]
    assert profile["local_solver"]["operatorless"]
    moving_profile = moved.sweep_history[-1]["environment_profile"]["moving_environment"]
    assert moving_profile["dense_local_operator_builds"] == 0
    assert moving_profile["dense_local_operator_reuses"] == 0
    assert moving_profile["dense_operatorless_local_problem_binds"] >= 1
    assert moving_profile["dense_operatorless_local_problem_solve_calls"] >= 1
    assert moving_profile["dense_operatorless_local_problem_solve_accepts"] >= 1
    assert moving_profile["dense_cpp_split_calls"] >= 1
    assert moving_profile["dense_cpp_split_accepts"] >= 1
    assert moving_profile["dense_cpp_split_failures"] == 0
    assert moving_profile["dense_cpp_sweep_workspace_enabled"]
    assert moving_profile["dense_cpp_sweep_workspace_records"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_solve_calls"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_binds"] == moving_profile[
        "dense_cpp_sweep_workspace_records"
    ]
    assert moving_profile["dense_cpp_sweep_workspace_boundary_binds"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_static_w_hits"] >= 1
    assert moving_profile["dense_cpp_tensor_primitive_calls"] >= 1
    assert moving_profile["dense_cpp_coarse_grain_mpo_calls"] >= 1
    assert moving_profile["dense_cpp_coarse_grain_mps_calls"] >= 1


def test_dense_dmrg_can_use_fused_cpp_two_site_solve_when_enabled():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseSweepWorkspace is None
        or not hasattr(cpp_davidson.DenseSweepWorkspace(), "solve_two_site")
    ):
        pytest.skip("C++ dense fused two-site solve is unavailable")

    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()
    common = dict(
        D=4,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
    )
    reference = DMRG(hamiltonian, **common).run()
    moved = DMRG(
        hamiltonian,
        **{
            **common,
            "abelian_matvec_options": {
                "moving_environment_dense_cpp_davidson": True,
                "moving_environment_dense_cpp_two_site_solve": True,
                "moving_environment_dense_cpp_davidson_backend": "blas",
            },
        },
    ).run()

    assert np.allclose(moved.e_tot, reference.e_tot, atol=1.0e-12)
    profile = moved.sweep_history[-1]["updates"][-1]["matvec_profile"]
    assert profile["local_solver"]["kind"] == "cpp_dense_davidson"
    assert profile["local_solver"]["two_site_solver"]
    moving_profile = moved.sweep_history[-1]["environment_profile"]["moving_environment"]
    assert moving_profile["dense_cpp_sweep_workspace_two_site_solve_calls"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_two_site_solve_accepts"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_two_site_static_w_reuses"] >= 1


def test_dense_dmrg_can_use_fused_cpp_block_davidson_when_enabled():
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or cpp_davidson.DenseSweepWorkspace is None
        or not hasattr(cpp_davidson.DenseSweepWorkspace(), "solve_two_site_block")
    ):
        pytest.skip("C++ dense fused block Davidson solve is unavailable")

    model = Heisenberg(L=4)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()
    common = dict(
        D=4,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
    )
    reference = DMRG(hamiltonian, **common).run()
    moved = DMRG(
        hamiltonian,
        **{
            **common,
            "abelian_matvec_options": {
                "moving_environment_dense_cpp_davidson": True,
                "moving_environment_dense_cpp_two_site_solve": True,
                "moving_environment_dense_cpp_block_davidson": True,
                "moving_environment_dense_cpp_block_davidson_size": 2,
                "moving_environment_dense_cpp_davidson_backend": "blas",
            },
        },
    ).run()

    assert np.allclose(moved.e_tot, reference.e_tot, atol=1.0e-12)
    profile = moved.sweep_history[-1]["updates"][-1]["matvec_profile"]
    assert profile["local_solver"]["kind"] == "cpp_dense_block_davidson"
    assert profile["local_solver"]["two_site_solver"]
    assert profile["local_solver"]["block_davidson"]
    assert profile["local_solver"]["block_size"] == 2
    moving_profile = moved.sweep_history[-1]["environment_profile"][
        "moving_environment"
    ]
    assert moving_profile["dense_cpp_sweep_workspace_two_site_solve_calls"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_two_site_block_solve_calls"] >= 1
    assert moving_profile["dense_cpp_sweep_workspace_block_davidson_accepts"] >= 1


def test_dense_dmrg_accepts_mpo_wrapper_when_final_gauge_is_left():
    model = Heisenberg(L=12)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()

    dmrg = DMRG(
        hamiltonian,
        D=32,
        nsweeps=1,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="reference",
        abelian_matvec_options={
            "moving_environment_dense_cpp_davidson": True,
            "moving_environment_dense_cpp_davidson_backend": "blas",
        },
    ).run()

    assert np.isfinite(dmrg.e_tot)
    assert dmrg.gauge.lower() == "left"
    assert dmrg.ground_state.center == len(hamiltonian.factors) - 1
    assert dmrg.ground_state.sites == hamiltonian.sites == initial.sites
    assert dmrg.ground_state.legs == tuple(site.leg for site in model.sites)


def test_dense_dmrg_dense_policy_uses_fused_local_solve():
    model = Heisenberg(L=8)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()

    dmrg = DMRG(
        hamiltonian,
        D=12,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
        performance="dense",
    ).run()

    moving_profile = dmrg.sweep_history[-1]["environment_profile"][
        "moving_environment"
    ]
    last_profile = dmrg.sweep_history[-1]["updates"][-1]["matvec_profile"]
    assert np.isfinite(dmrg.e_tot)
    assert last_profile["local_solver"]["kind"] == "cpp_dense_davidson"
    assert last_profile["local_solver"]["two_site_solver"]
    assert moving_profile["dense_cpp_sweep_workspace_two_site_solve_calls"] >= 1
    assert moving_profile["dense_cpp_environment_update_calls"] == 0
    assert moving_profile["dense_cpp_dmrg_sweep_accepts"] >= 1
    assert moving_profile["dense_cpp_dmrg_sweep_backend_actual"] == (
        "cpp_dense_dmrg_half_sweep"
    )


def test_dense_dmrg_auto_policy_uses_dense_backend_for_dense_mpo():
    model = Heisenberg(L=12)
    hamiltonian = model.build_H_mpo()
    initial = model.build_neel_state()

    dmrg = DMRG(
        hamiltonian,
        D=12,
        nsweeps=2,
        init_guess=initial,
        not_conv_err=False,
        verbose=0,
    ).run()

    moving_profile = dmrg.sweep_history[-1]["environment_profile"][
        "moving_environment"
    ]
    last_profile = dmrg.sweep_history[-1]["updates"][-1]["matvec_profile"]
    assert dmrg.performance == "auto"
    assert dmrg.resolved_performance == "dense"
    assert last_profile["local_solver"]["kind"] == "cpp_dense_davidson"
    assert last_profile["local_solver"]["two_site_solver"]
    assert moving_profile["dense_cpp_sweep_workspace_two_site_solve_calls"] >= 1


def test_symmetric_dmrg_auto_policy_uses_symmetric_backend():
    model = Heisenberg(L=4)
    dmrg = DMRG(
        model.build_H_mpo(),
        D=4,
        init_guess=model.build_neel_state(),
        symmetry=True,
    )

    assert dmrg.performance == "auto"
    assert dmrg.resolved_performance == "symmetric"
