import io
import contextlib
import logging

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem import basis as basis_module
from pyqed.qchem.basis import (
    ContractedGaussian,
    _basis_signature,
    _cart2sph_unit_block,
    _compute_dense_eri_spherical_shellblocked,
    _compute_dense_eri_with_backend,
    _compute_pair_bounds,
    _shell,
    direct_jk_spherical_cpp,
)
from pyqed.qchem.hf import RHF
from pyqed.qchem.hf.rhf import get_jk


def test_builtin_spherical_matches_pyscf_rhf_energy():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    logging.disable(logging.CRITICAL)

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    basis = "def2-svp"

    mol = Molecule(atom=atom, basis=basis, unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={"coord_type": "spherical", "eri_representation": "dense"})
        mf = RHF(mol).run(tol=1e-9, max_cycle=100)

    pmol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=False, verbose=0)
    pmf = scf.RHF(pmol)
    pmf.conv_tol = 1e-9
    pmf.max_cycle = 100
    pmf.kernel()

    assert mol.nao == pmol.nao_nr()
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1e-9)


def test_builtin_molecular_default_is_spherical():
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11",
        basis="def2-svp",
        unit="bohr",
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={"eri_representation": "direct"})

    assert mol.builtin_coord_type == "spherical"
    assert mol._builtin_build_info["coord_type"] == "spherical"
    assert mol._builtin_direct_jk_data["kernel"].endswith("spherical-direct-jk")
    assert mol._builtin_direct_jk_data["aosym"] == "s8"
    assert mol._builtin_build_info["direct_jk"]["aosym"] == "s8"


def test_builtin_spherical_matches_pyscf_overlap_and_hcore_for_def2_tzvp():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto

    logging.disable(logging.CRITICAL)

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    basis = "def2-tzvp"

    mol = Molecule(atom=atom, basis=basis, unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={"coord_type": "spherical", "eri_representation": "dense"})

    pmol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=False, verbose=0)

    assert mol.nao == pmol.nao_nr()
    np.testing.assert_allclose(mol.overlap, pmol.intor("int1e_ovlp"), atol=1e-12)
    np.testing.assert_allclose(
        mol.hcore,
        pmol.intor("int1e_kin") + pmol.intor("int1e_nuc"),
        atol=1e-10,
    )


def test_builtin_spherical_f_shell_eri_matches_pyscf():
    pytest.importorskip("pyscf")
    from pyscf import gto

    logging.disable(logging.CRITICAL)
    mol = Molecule(atom="O 0 0 0", basis="def2-tzvp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "dense",
                "aosym": "s1",
                "eri_screen_tol": 0.0,
            },
        )

    pmol = gto.M(atom="O 0 0 0", basis="def2-tzvp", unit="Bohr", cart=False, verbose=0)
    np.testing.assert_allclose(mol.eri, pmol.intor("int2e_sph", aosym="s1"), atol=2.0e-12)


def test_builtin_spherical_dense_build_never_allocates_full_cartesian_eri(monkeypatch):
    def reject_cartesian_dense(*args, **kwargs):
        raise AssertionError("spherical dense build requested a full Cartesian ERI tensor")

    monkeypatch.setattr(basis_module, "_compute_dense_eri_with_backend", reject_cartesian_dense)
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11",
        basis="def2-svp",
        unit="bohr",
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={"coord_type": "spherical", "eri_representation": "dense"},
        )

    info = mol._builtin_build_info
    assert info["dense_eri_working_basis"] == "spherical"
    assert info["allocated_cartesian_dense_eri"] is False
    assert "spherical-shellblocked" in info["dense_builder"]
    assert info["max_cartesian_shell_quartet_elements"] < info["cartesian_dense_eri_elements"]


def test_builtin_spherical_cd_is_matrix_free_and_matches_dense_jk(monkeypatch):
    atom = "O 0 0 0"
    dense = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dense.build(options={
                "coord_type": "spherical",
                "eri_representation": "dense",
                "aosym": "s1",
                "eri_screen_tol": 0.0,
            },
        )

    def fail_dense(*_args, **_kwargs):
        raise AssertionError("spherical CD must not construct a dense ERI source")

    monkeypatch.setattr(
        basis_module, "_compute_dense_eri_spherical_shellblocked", fail_dense
    )
    monkeypatch.setattr(basis_module, "_compute_dense_eri_with_backend", fail_dense)
    cd = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        cd.build(eri="cd",
            options={
                "coord_type": "spherical",
                "low_rank_tol": 1.0e-10,
                "eri_screen_tol": 0.0,
            },
        )

    rng = np.random.default_rng(20260825)
    dm = rng.normal(size=(dense.nao, dense.nao))
    dm += dm.T
    reference_j, reference_k = get_jk(dense, dm)
    cd_j, cd_k = get_jk(cd, dm)
    info = cd._builtin_build_info

    assert cd.eri is None
    assert cd.eri_s4 is None
    assert cd.eri_s8 is None
    assert info["dense_builder"] is None
    assert info["dense_eri_working_basis"] is None
    assert info["allocated_cartesian_dense_eri"] is False
    assert info["cartesian_dense_eri_elements"] is None
    assert info["factor_storage"] == "packed-pair"
    assert info["cd"]["algorithm"] == "matrix-free-pivoted-cholesky"
    assert info["cd"]["working_basis"] == "spherical"
    assert info["cd"]["allocated_dense_eri"] is False
    assert "matrix-free" in info["factor_builder"]
    np.testing.assert_allclose(cd_j, reference_j, atol=2.0e-8, rtol=2.0e-8)
    np.testing.assert_allclose(cd_k, reference_k, atol=2.0e-8, rtol=2.0e-8)


def test_native_g_shell_spherical_eri_matches_global_cartesian_transform():
    if basis_module._integrals_cpp is None or not hasattr(
        basis_module._integrals_cpp, "compute_dense_eri_spherical"
    ):
        pytest.skip("native spherical ERI extension is unavailable")
    cart_basis = [
        ContractedGaussian(shell=shell, exps=[1.0], coefs=[1.0])
        for shell in _shell(4)
    ]
    signatures = tuple(_basis_signature(function) for function in cart_basis)
    pair_bounds = _compute_pair_bounds(signatures)
    (spherical, _computed, _skipped), builder = (
        _compute_dense_eri_spherical_shellblocked(
            signatures,
            pair_bounds,
            0.0,
            backend="cpp",
            workers=2,
        )
    )
    (cartesian, _computed, _skipped), _builder = _compute_dense_eri_with_backend(
        signatures,
        pair_bounds,
        0.0,
        backend="cpp",
    )
    transform = _cart2sph_unit_block(4)
    reference = np.einsum(
        "pa,qb,rc,sd,pqrs->abcd",
        transform,
        transform,
        transform,
        transform,
        cartesian,
        optimize=True,
    )

    assert "spherical-shellblocked" in builder
    np.testing.assert_allclose(spherical, reference, atol=2.0e-14, rtol=2.0e-14)


def test_builtin_direct_spherical_jk_consumes_spherical_density():
    if basis_module._integrals_cpp is None or not hasattr(
        basis_module._integrals_cpp, "direct_jk_spherical"
    ):
        pytest.skip("native direct spherical J/K extension is unavailable")
    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    dense = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    direct = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dense.build(options={
                "coord_type": "spherical",
                "eri_representation": "dense",
                "aosym": "s1",
                "eri_screen_tol": 0.0,
            },
        )
        direct.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_screen_tol": 0.0,
            },
        )

    rng = np.random.default_rng(71)
    dm = rng.normal(size=(dense.nao, dense.nao))
    dm += dm.T
    vj, vk = get_jk(direct, dm)
    reference_j = np.einsum("lk,ijkl->ij", dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", dm, dense.eri, optimize=True)

    assert direct._builtin_direct_jk_data["kernel"] == "cpp-spherical-direct-jk"
    assert direct._builtin_direct_jk_data["last_mode"] == "jk-cpp-spherical"
    np.testing.assert_allclose(vj, reference_j, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(vk, reference_k, atol=3.0e-12, rtol=3.0e-12)
    data = direct._builtin_direct_jk_data
    assert data.get("native_plan") is not None
    common = (
        data["shells"],
        data["origins"],
        data["exps"],
        data["weights"],
        data["nprim"],
        data["pair_bounds"],
        data["transform"],
    )
    for workers in (1, 2, 4, 2, 1, 4):
        parallel = direct_jk_spherical_cpp(*common, dm, 0.0, workers=workers)
        assert parallel is not None
        parallel_j, parallel_k, computed, _skipped = parallel
        assert computed > 0
        np.testing.assert_allclose(parallel_j, reference_j, atol=3.0e-12, rtol=3.0e-12)
        np.testing.assert_allclose(parallel_k, reference_k, atol=3.0e-12, rtol=3.0e-12)
    planned = direct_jk_spherical_cpp(
        *common,
        dm,
        0.0,
        workers=1,
        native_plan=data["native_plan"],
        symmetric_density=True,
    )
    np.testing.assert_allclose(planned[0], reference_j, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(planned[1], reference_k, atol=3.0e-12, rtol=3.0e-12)

    tiny_dm = np.eye(dense.nao) * 1.0e-12
    screened = direct_jk_spherical_cpp(*common, tiny_dm, 1.0e-10, workers=4)
    assert screened is not None
    screened_j, screened_k, screened_computed, screened_skipped = screened
    tiny_reference_j = np.einsum("lk,ijkl->ij", tiny_dm, dense.eri, optimize=True)
    tiny_reference_k = np.einsum("lk,ilkj->ij", tiny_dm, dense.eri, optimize=True)
    assert screened_computed == 0
    assert screened_skipped > 0
    np.testing.assert_allclose(screened_j, tiny_reference_j, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(screened_k, tiny_reference_k, atol=1.0e-10, rtol=0.0)

    dm_nonsymmetric = rng.normal(size=(dense.nao, dense.nao))
    vj, vk = get_jk(direct, dm_nonsymmetric)
    reference_j = np.einsum("lk,ijkl->ij", dm_nonsymmetric, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", dm_nonsymmetric, dense.eri, optimize=True)
    np.testing.assert_allclose(vj, reference_j, atol=3.0e-12, rtol=3.0e-12)
    np.testing.assert_allclose(vk, reference_k, atol=3.0e-12, rtol=3.0e-12)


def test_builtin_direct_spherical_rys_uses_native_cpp_through_d_shells():
    if (
        basis_module._integrals_cpp is None
        or not hasattr(basis_module._integrals_cpp, "direct_jk_spherical")
    ):
        pytest.skip("native C++ Rys direct J/K is unavailable")
    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    dense = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    direct = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dense.build(options={"coord_type": "spherical", "eri_representation": "dense", "aosym": "s1"},
        )
        direct.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": "rys",
                "rys_cache_mib": 1,
                "parallel": True,
                "eri_workers": 2,
                "parallel_min_nao": 0,
            },
        )

    rng = np.random.default_rng(2718)
    dm = rng.normal(size=(dense.nao, dense.nao))
    dm += dm.T
    vj, vk = get_jk(direct, dm)
    reference_j = np.einsum("lk,ijkl->ij", dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", dm, dense.eri, optimize=True)

    data = direct._builtin_direct_jk_data
    assert data.get("native_plan") is not None
    stats = basis_module._integrals_cpp.spherical_direct_jk_plan_stats(
        data["native_plan"]
    )
    assert stats["tasks"] == sum(stats["rank_tasks"])
    assert stats["recurrence_cache_max_bytes"] == 1024**2
    assert stats["spherical_plan_cache_max_bytes"] == 0
    assert (
        stats["spherical_plan_cache_bytes"]
        <= stats["spherical_plan_cache_max_bytes"]
    )
    assert stats["recurrence_cache_bytes"] > 0
    assert all(
        cached <= total
        for cached, total in zip(
            stats["rank_recurrence_cached"], stats["rank_tasks"]
        )
    )
    assert data["kernel"] == "rys-cpp-spherical-direct-jk"
    assert data["last_mode"] == "jk-rys-cpp-spherical"
    assert data["last_computed"] > 0
    np.testing.assert_allclose(vj, reference_j, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(vk, reference_k, atol=1.0e-10, rtol=1.0e-10)

    worker_results = []
    for workers in (1, 2, 4):
        result = direct_jk_spherical_cpp(
            data["shells"], data["origins"], data["exps"], data["weights"],
            data["nprim"], data["pair_bounds"], data["transform"], dm,
            workers=workers,
            rys_max_rank=12,
            native_plan=data["native_plan"],
            symmetric_density=True,
        )
        assert result is not None
        worker_results.append(result)
        np.testing.assert_allclose(result[0], reference_j, atol=1.0e-10, rtol=1.0e-10)
        np.testing.assert_allclose(result[1], reference_k, atol=1.0e-10, rtol=1.0e-10)
    assert [result[2:] for result in worker_results] == [worker_results[0][2:]] * 3

    runtime_screen_tol = max(1.0e-10, 10.0 * data["screen_tol"])
    planned_screened = direct_jk_spherical_cpp(
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], dm,
        runtime_screen_tol,
        workers=2,
        rys_max_rank=12,
        native_plan=data["native_plan"],
        symmetric_density=True,
    )
    unplanned_screened = direct_jk_spherical_cpp(
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], dm,
        runtime_screen_tol,
        workers=2,
        rys_max_rank=12,
        symmetric_density=True,
    )
    np.testing.assert_allclose(
        planned_screened[0], unplanned_screened[0], atol=1.0e-12, rtol=1.0e-12
    )
    np.testing.assert_allclose(
        planned_screened[1], unplanned_screened[1], atol=1.0e-12, rtol=1.0e-12
    )
    assert planned_screened[2] >= unplanned_screened[2]

    os_result = direct_jk_spherical_cpp(
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], dm,
        workers=1,
        rys_max_rank=-1,
    )
    for max_rank in range(13):
        partial_rys = direct_jk_spherical_cpp(
            data["shells"], data["origins"], data["exps"], data["weights"],
            data["nprim"], data["pair_bounds"], data["transform"], dm,
            workers=1,
            rys_max_rank=max_rank,
        )
        np.testing.assert_allclose(partial_rys[0], os_result[0], atol=1.0e-11, rtol=1.0e-11)
        np.testing.assert_allclose(partial_rys[1], os_result[1], atol=1.0e-11, rtol=1.0e-11)
        assert partial_rys[2:] == os_result[2:]

    nonsymmetric_dm = rng.normal(size=(dense.nao, dense.nao))
    vj, vk = get_jk(direct, nonsymmetric_dm)
    reference_j = np.einsum("lk,ijkl->ij", nonsymmetric_dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", nonsymmetric_dm, dense.eri, optimize=True)
    np.testing.assert_allclose(vj, reference_j, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(vk, reference_k, atol=1.0e-10, rtol=1.0e-10)

    screened = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        screened.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": "rys",
                    "direct_scf_tol": 1.0e-10,
            },
        )
    tiny_dm = np.eye(dense.nao) * 1.0e-12
    vj, vk = get_jk(screened, tiny_dm)
    reference_j = np.einsum("lk,ijkl->ij", tiny_dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", tiny_dm, dense.eri, optimize=True)
    assert screened._builtin_direct_jk_data["last_computed"] == 0
    assert screened._builtin_direct_jk_data["last_skipped"] > 0
    np.testing.assert_allclose(vj, reference_j, atol=1.0e-10, rtol=0.0)
    np.testing.assert_allclose(vk, reference_k, atol=1.0e-10, rtol=0.0)


def test_planned_spherical_rys_accepts_zero_screen_override():
    if (
        basis_module._integrals_cpp is None
        or not hasattr(basis_module._integrals_cpp, "direct_jk_spherical")
    ):
        pytest.skip("native C++ Rys direct J/K is unavailable")
    atom = "O 0 0 0"
    dense = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    direct = Molecule(atom=atom, basis="def2-svp", unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        dense.build(options={"coord_type": "spherical", "eri_representation": "dense", "aosym": "s1"},
        )
        direct.build(options={
                "coord_type": "spherical",
                "eri_representation": "direct",
                "eri_backend": "rys",
                "eri_screen_tol": 1.0,
            },
        )

    data = direct._builtin_direct_jk_data
    rng = np.random.default_rng(314159)
    dm = rng.normal(size=(direct.nao, direct.nao))
    dm += dm.T
    result = direct_jk_spherical_cpp(
        data["shells"], data["origins"], data["exps"], data["weights"],
        data["nprim"], data["pair_bounds"], data["transform"], dm,
        screen_tol=0.0,
        workers=1,
        rys_max_rank=data["rys_max_rank"],
        native_plan=data["native_plan"],
        symmetric_density=True,
    )
    reference_j = np.einsum("lk,ijkl->ij", dm, dense.eri, optimize=True)
    reference_k = np.einsum("lk,ilkj->ij", dm, dense.eri, optimize=True)
    np.testing.assert_allclose(result[0], reference_j, atol=1.0e-10, rtol=1.0e-10)
    np.testing.assert_allclose(result[1], reference_k, atol=1.0e-10, rtol=1.0e-10)


def test_spherical_rys_plan_prunes_zero_coefficient_primitive_pairs():
    cpp = basis_module._integrals_cpp
    if cpp is None or not hasattr(cpp, "build_spherical_direct_jk_plan"):
        pytest.skip("native C++ Rys direct J/K plan is unavailable")

    shells = np.array([[0, 0, 0]], dtype=np.int64)
    origins = np.zeros((1, 3), dtype=np.float64)
    exps = np.array([[1.0, 0.5]], dtype=np.float64)
    weights = np.array([[1.0, 0.0]], dtype=np.float64)
    nprim = np.array([2], dtype=np.int64)
    pair_bounds = np.ones((1, 1), dtype=np.float64)
    transform = np.ones((1, 1), dtype=np.float64)
    plan = cpp.build_spherical_direct_jk_plan(
        shells, origins, exps, weights, nprim, pair_bounds, transform,
        0.0, 6, 12, 1024**2,
    )
    stats = cpp.spherical_direct_jk_plan_stats(plan)
    assert stats["primitive_pair_terms"] == 1
    assert stats["primitive_pair_terms_unpruned"] == 4

    dm = np.array([[0.7]], dtype=np.float64)
    pruned = direct_jk_spherical_cpp(
        shells, origins, exps, weights, nprim, pair_bounds, transform, dm,
        workers=1, rys_max_rank=12, native_plan=plan, symmetric_density=True,
    )
    reference = direct_jk_spherical_cpp(
        shells, origins, exps[:, :1], weights[:, :1],
        np.array([1], dtype=np.int64), pair_bounds, transform, dm,
        workers=1, rys_max_rank=12, symmetric_density=True,
    )
    np.testing.assert_allclose(pruned[0], reference[0], atol=0.0, rtol=0.0)
    np.testing.assert_allclose(pruned[1], reference[1], atol=0.0, rtol=0.0)


def test_spherical_rys_plan_batches_general_contractions():
    cpp = basis_module._integrals_cpp
    if cpp is None or not hasattr(cpp, "build_spherical_direct_jk_plan"):
        pytest.skip("native C++ Rys direct J/K plan is unavailable")

    p_shell = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.int64)
    shells = np.vstack((p_shell, p_shell))
    origins = np.zeros((6, 3), dtype=np.float64)
    exps = np.tile([1.0, 0.4], (6, 1))
    weights = np.vstack(
        (
            np.tile([1.0, 0.2], (3, 1)),
            np.tile([0.6, -0.1], (3, 1)),
        )
    )
    nprim = np.full(6, 2, dtype=np.int64)
    pair_bounds = np.ones((6, 6), dtype=np.float64)
    transform = np.eye(6, dtype=np.float64)
    plan = cpp.build_spherical_direct_jk_plan(
        shells, origins, exps, weights, nprim, pair_bounds, transform,
        0.0, 6, 12, 1024**2,
    )
    stats = cpp.spherical_direct_jk_plan_stats(plan)
    assert stats["task_entry_bytes"] == 32
    assert stats["task_bytes"] == stats["tasks"] * stats["task_entry_bytes"]
    assert stats["execution_tasks"] < stats["tasks"]
    assert stats["contraction_batches"] == 1
    assert stats["batched_tasks"] == 6
    assert stats["spherical_plans"] == 0

    rng = np.random.default_rng(20260824)
    dm = rng.normal(size=(6, 6))
    dm += dm.T
    common = (
        shells, origins, exps, weights, nprim, pair_bounds, transform, dm,
    )
    batched = direct_jk_spherical_cpp(
        *common,
        workers=2,
        rys_max_rank=12,
        native_plan=plan,
        symmetric_density=True,
    )
    unbatched = direct_jk_spherical_cpp(
        *common,
        workers=1,
        rys_max_rank=12,
        symmetric_density=True,
    )
    rank_limited = direct_jk_spherical_cpp(
        *common,
        workers=1,
        rys_max_rank=0,
        native_plan=plan,
        symmetric_density=True,
    )
    assert batched[2:] == unbatched[2:]
    np.testing.assert_allclose(batched[0], unbatched[0], atol=5.0e-15, rtol=5.0e-15)
    np.testing.assert_allclose(batched[1], unbatched[1], atol=5.0e-15, rtol=5.0e-15)
    np.testing.assert_allclose(rank_limited[0], unbatched[0], atol=5.0e-13, rtol=5.0e-13)
    np.testing.assert_allclose(rank_limited[1], unbatched[1], atol=5.0e-13, rtol=5.0e-13)


def test_builtin_ri_factors_are_constructed_in_spherical_pair_space():
    mol = Molecule(
        atom="O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11",
        basis="def2-svp",
        unit="bohr",
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(options={
                "coord_type": "spherical",
                "eri_representation": "ri",
                "ri_cache": False,
                "eri_screen_tol": 0.0,
            },
        )

    info = mol._builtin_build_info["ri"]
    assert info["working_basis"] == "spherical"
    assert info["primary_nao"] == mol.nao
    assert info["pair_shape"][1] == mol.nao * (mol.nao + 1) // 2
    assert "spherical-pair-blocked" in info["tensor_builder"]
    assert mol.eri_factors.shape[1:] == (mol.nao, mol.nao)
