import numpy as np
import pytest

from examples.mps.hubbard_2d_mps_vs_letta import (
    hubbard_2d_dense_mpo,
    random_fixed_sector_abelian_mps,
    site_qn_maps,
)
from pyqed.mps import DMRG, MPO, dense_to_symmetric, dense_to_symmetric_mpo
from pyqed.mps import cpp_davidson
from pyqed.mps._dmrg3s import (
    _absorb_center_left,
    _absorb_center_right,
    _absorb_enriched_center_left,
    _absorb_enriched_center_right,
    _compress_left_expansion,
    _compress_right_expansion,
    _left_enriched_factorization,
    _native_expansion,
    _one_site_davidson,
    _pad_site_left,
    _pad_site_right,
    _partial_left_expansion_python,
    _partial_right_expansion_python,
    _right_enriched_factorization,
)
from pyqed.mps.abelian_direct import AbelianSiteTensorData
from pyqed.mps.mps import initial_E, initial_F
from pyqed.mps.symmetry import AbelianSector, SymmetryManager


def _hermitian(rng, size):
    matrix = rng.normal(size=(size, size)) + 1j * rng.normal(size=(size, size))
    return 0.5 * (matrix + matrix.conj().T)


def _assert_tensor_data_close(actual, expected, *, atol=1.0e-12):
    assert set(actual.data) == set(expected.data)
    for key in actual.data:
        assert np.asarray(actual.data[key]).shape == np.asarray(expected.data[key]).shape
        assert np.allclose(actual.data[key], expected.data[key], atol=atol, rtol=0.0)


def _two_site_hubbard_native_problem():
    dense_mpo, _ = hubbard_2d_dense_mpo(2, 1, hopping=1.0, hubbard_u=4.0)
    qn_maps = site_qn_maps(2)
    mpo = dense_to_symmetric_mpo(
        dense_mpo,
        qn_maps,
        native_site_storage=True,
    )
    factors = random_fixed_sector_abelian_mps(
        2,
        1,
        1,
        max_bond_dim=1,
        qn_maps=qn_maps,
        native_site_storage=True,
        seed=4,
    )
    target = SymmetryManager(["charge", "sz"]).get_target_qn(2, 0)
    return factors, mpo, target


def test_packed_cpp_one_site_davidson_matches_exact_local_problem():
    plan_cls = cpp_davidson.AbelianTDVPSiteHeffPlan
    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE or not hasattr(plan_cls, "davidson"):
        pytest.skip("packed C++ one-site Davidson is unavailable")

    rng = np.random.default_rng(7)
    q0 = AbelianSector(("charge",), (0,))
    key3 = (q0, q0, q0)
    key4 = (q0, q0, q0, q0)
    left_matrix = _hermitian(rng, 2)
    right_matrix = _hermitian(rng, 3)
    local_matrix = _hermitian(rng, 2)
    site = AbelianSiteTensorData(
        {key3: rng.normal(size=(2, 3, 2)) + 1j * rng.normal(size=(2, 3, 2))},
        [[q0], [q0], [q0]],
        [-1, 1, 1],
    )
    left = AbelianSiteTensorData(
        {key3: left_matrix.reshape(1, 2, 2)},
        [[q0], [q0], [q0]],
        [-1, 1, 1],
    )
    mpo = AbelianSiteTensorData(
        {key4: local_matrix.reshape(1, 1, 2, 2)},
        [[q0], [q0], [q0], [q0]],
        [-1, 1, 1, -1],
    )
    right = AbelianSiteTensorData(
        {key3: right_matrix.reshape(1, 3, 3)},
        [[q0], [q0], [q0]],
        [-1, 1, 1],
    )
    kwargs = dict(tol=1.0e-10, max_iter=40, restart_dim=16)

    energy_cpp, optimized, info = _one_site_davidson(
        site, left, mpo, right, backend="cpp", **kwargs
    )
    energy_python, _optimized_python, _python_info = _one_site_davidson(
        site, left, mpo, right, backend="python", **kwargs
    )
    exact_matrix = np.kron(left_matrix, np.kron(right_matrix, local_matrix))
    exact_energy = np.linalg.eigvalsh(exact_matrix)[0]

    assert energy_cpp == pytest.approx(exact_energy, abs=1.0e-9)
    assert energy_cpp == pytest.approx(energy_python, abs=1.0e-9)
    assert info["backend"] == "cpp-u1-site-davidson"
    assert info["routes"] == 1
    assert info["converged"] is True
    assert np.linalg.norm(next(iter(optimized.data.values()))) == pytest.approx(1.0)

    _energy_again, _optimized_again, reused_info = _one_site_davidson(
        site, left, mpo, right, backend="cpp", **kwargs
    )
    assert reused_info["workspace_reused"] is True


def test_grouped_gemm_environment_updates_match_complex_reference():
    left_kernel = cpp_davidson.abelian_left_environment_advance_data
    right_kernel = cpp_davidson.abelian_right_environment_advance_data
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or left_kernel is None
        or right_kernel is None
    ):
        pytest.skip("native Abelian environment kernels are unavailable")

    rng = np.random.default_rng(17)
    q0 = AbelianSector(("charge",), (0,))
    key3 = (q0, q0, q0)
    key4 = (q0, q0, q0, q0)

    def random_complex(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    def tensor3(block):
        return AbelianSiteTensorData(
            {key3: block},
            [[q0], [q0], [q0]],
            [-1, 1, 1],
            copy=False,
        )

    def tensor4(block):
        return AbelianSiteTensorData(
            {key4: block},
            [[q0], [q0], [q0], [q0]],
            [-1, 1, 1, -1],
            copy=False,
        )

    environment = random_complex((2, 16, 14))
    left_bra = random_complex((16, 12, 1))
    left_mpo = random_complex((2, 3, 1, 1))
    left_ket = random_complex((14, 10, 1))
    _keys, blocks, _qns, _dirs = left_kernel(
        tensor4(left_mpo),
        tensor3(left_bra),
        tensor3(environment),
        tensor3(left_ket),
    )
    left_reference = np.einsum(
        "xij,iau,xyuv,jbv->yab",
        environment,
        left_bra.conj(),
        left_mpo,
        left_ket,
        optimize=True,
    )
    assert np.allclose(blocks[0], left_reference, atol=2.0e-12, rtol=2.0e-12)

    right_bra = random_complex((12, 16, 1))
    right_mpo = random_complex((3, 2, 1, 1))
    right_ket = random_complex((10, 14, 1))
    _keys, blocks, _qns, _dirs = right_kernel(
        tensor4(right_mpo),
        tensor3(right_bra),
        tensor3(environment),
        tensor3(right_ket),
    )
    right_reference = np.einsum(
        "aip,xij,yxpv,bjv->yab",
        right_bra.conj(),
        environment,
        right_mpo,
        right_ket,
        optimize=True,
    )
    assert np.allclose(blocks[0], right_reference, atol=2.0e-12, rtol=2.0e-12)


def test_native_dmrg3s_expansions_and_fused_absorption_match_reference():
    left_kernel = cpp_davidson.abelian_dmrg3s_left_expansion_data
    right_kernel = cpp_davidson.abelian_dmrg3s_right_expansion_data
    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE or left_kernel is None or right_kernel is None:
        pytest.skip("native DMRG3S expansion kernels are unavailable")

    factors, mpo, target = _two_site_hubbard_native_problem()
    left = initial_E(mpo[0])
    right = initial_F(mpo[-1], target_qn=target)

    left_reference = _partial_left_expansion_python(factors[0], mpo[0], left)
    left_native = _native_expansion(left_kernel(factors[0], mpo[0], left), factors[0])
    _assert_tensor_data_close(left_native, left_reference)

    right_reference = _partial_right_expansion_python(factors[1], mpo[1], right)
    right_native = _native_expansion(right_kernel(factors[1], mpo[1], right), factors[1])
    _assert_tensor_data_close(right_native, right_reference)

    _left_site, left_center, left_dims, _info = _left_enriched_factorization(
        factors[0], left_native, 4, 0.1, 1.0e-14
    )
    padded_left = _absorb_center_left(
        left_center,
        _pad_site_left(factors[1], left_dims),
    )
    fused_left = _absorb_enriched_center_left(left_center, factors[1], left_dims)
    _assert_tensor_data_close(fused_left, padded_left)

    right_center, _right_site, right_dims, _info = _right_enriched_factorization(
        factors[1], right_native, 4, 0.1, 1.0e-14
    )
    padded_right = _absorb_center_right(
        _pad_site_right(factors[0], right_dims),
        right_center,
    )
    fused_right = _absorb_enriched_center_right(factors[0], right_center, right_dims)
    _assert_tensor_data_close(fused_right, padded_right)


def test_streamed_dmrg3s_sketch_has_a_global_enrichment_rank_cap():
    left_kernel = cpp_davidson.abelian_dmrg3s_left_expansion_data
    right_kernel = cpp_davidson.abelian_dmrg3s_right_expansion_data
    if not cpp_davidson.CPP_DAVIDSON_AVAILABLE or left_kernel is None or right_kernel is None:
        pytest.skip("native DMRG3S expansion kernels are unavailable")

    factors, mpo, target = _two_site_hubbard_native_problem()
    left = initial_E(mpo[0])
    right = initial_F(mpo[-1], target_qn=target)
    left_sketch = _native_expansion(
        left_kernel(factors[0], mpo[0], left, 2, 19),
        factors[0],
    )
    right_sketch = _native_expansion(
        right_kernel(factors[1], mpo[1], right, 2, 23),
        factors[1],
    )
    left_low_rank, left_info = _compress_left_expansion(
        factors[0], left_sketch, 1, 0.0
    )
    right_low_rank, right_info = _compress_right_expansion(
        factors[1], right_sketch, 1, 0.0
    )

    assert (
        sum(
            {
                key[1]: block.shape[1]
                for key, block in left_low_rank.data.items()
            }.values()
        )
        <= 1
    )
    assert (
        sum(
            {
                key[0]: block.shape[0]
                for key, block in right_low_rank.data.items()
            }.values()
        )
        <= 1
    )
    assert left_info["enrichment_states"] <= 1
    assert right_info["enrichment_states"] <= 1
    assert left_info["enrichment_mode"] == "streamed_low_rank"
    assert right_info["enrichment_mode"] == "streamed_low_rank"


def test_streamed_dmrg3s_count_sketch_gemm_matches_full_expansion():
    left_kernel = cpp_davidson.abelian_dmrg3s_left_expansion_data
    right_kernel = cpp_davidson.abelian_dmrg3s_right_expansion_data
    if (
        not cpp_davidson.CPP_DAVIDSON_AVAILABLE
        or left_kernel is None
        or right_kernel is None
    ):
        pytest.skip("native DMRG3S expansion kernels are unavailable")

    mask = (1 << 64) - 1

    def splitmix64(value):
        value = (int(value) + 0x9E3779B97F4A7C15) & mask
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
        return value ^ (value >> 31)

    def project_full(full, rank, seed, axis):
        sector_seed = splitmix64(int(seed) ^ 1)
        projected_shape = list(full.shape)
        projected_shape[axis] = int(rank)
        projected = np.zeros(projected_shape, dtype=complex)
        for fused in range(full.shape[axis]):
            hashed = splitmix64(sector_seed ^ fused)
            bucket = hashed % int(rank)
            sign = -1.0 if hashed >> 63 else 1.0
            source = [slice(None)] * full.ndim
            target = [slice(None)] * full.ndim
            source[axis] = fused
            target[axis] = bucket
            projected[tuple(target)] += sign * full[tuple(source)]
        return projected

    rng = np.random.default_rng(29)
    q0 = AbelianSector(("charge",), (0,))
    key3 = (q0, q0, q0)
    key4 = (q0, q0, q0, q0)

    def tensor3(block):
        return AbelianSiteTensorData(
            {key3: block}, [[q0], [q0], [q0]], [-1, 1, 1], copy=False
        )

    def tensor4(block):
        return AbelianSiteTensorData(
            {key4: block},
            [[q0], [q0], [q0], [q0]],
            [-1, 1, 1, -1],
            copy=False,
        )

    def random_complex(shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)

    environment = random_complex((2, 16, 14))
    left_site = random_complex((14, 12, 1))
    left_mpo = random_complex((2, 3, 1, 1))
    exact_left = np.einsum(
        "xik,kou,xmvu->iomv",
        environment,
        left_site,
        left_mpo,
        optimize=True,
    ).reshape(16, 36, 1)
    left_payload = left_kernel(
        tensor3(left_site), tensor4(left_mpo), tensor3(environment), 5, 31
    )
    left_block = np.asarray(next(iter(dict(left_payload).values())))
    assert np.allclose(
        left_block,
        project_full(exact_left, 5, 31, 1),
        atol=2.0e-11,
        rtol=2.0e-11,
    )

    right_site = random_complex((12, 14, 1))
    right_mpo = random_complex((3, 2, 1, 1))
    exact_right = np.einsum(
        "oip,xji,mxvp->omjv",
        right_site,
        environment,
        right_mpo,
        optimize=True,
    ).reshape(36, 16, 1)
    right_payload = right_kernel(
        tensor3(right_site), tensor4(right_mpo), tensor3(environment), 5, 37
    )
    right_block = np.asarray(next(iter(dict(right_payload).values())))
    assert np.allclose(
        right_block,
        project_full(exact_right, 5, 37, 0),
        atol=2.0e-11,
        rtol=2.0e-11,
    )


def test_dmrg3s_charge_sector_uses_rank_three_local_problems():
    q0 = AbelianSector(("charge",), (0,))
    q1 = AbelianSector(("charge",), (1,))
    qn_maps = [{0: q0, 1: q1} for _ in range(2)]
    annihilation = np.array([[0.0, 1.0], [0.0, 0.0]])
    creation = annihilation.T
    parity = np.diag([1.0, -1.0])
    identity = np.eye(2)
    mpo_left = np.zeros((1, 4, 2, 2))
    mpo_right = np.zeros((4, 1, 2, 2))
    mpo_left[0, 0] = identity
    mpo_right[3, 0] = identity
    for channel, (left, right) in enumerate(
        [(-creation @ parity, annihilation), (-parity @ annihilation, creation)],
        start=1,
    ):
        mpo_left[0, channel] = left
        mpo_right[channel, 0] = right

    mpo = dense_to_symmetric_mpo([mpo_left, mpo_right], qn_maps)
    initial = dense_to_symmetric(
        [
            np.array([0.0, 1.0]).reshape(1, 2, 1),
            np.array([1.0, 0.0]).reshape(1, 2, 1),
        ],
        phys_qns=[q0, q1],
    )
    symmetry = SymmetryManager(["charge"])
    solver = DMRG(
        mpo,
        D=4,
        init_guess=initial,
        nsweeps=3,
        opt="3s",
        target_qn=symmetry.get_target_qn(1),
        site_qn_maps=qn_maps,
        not_conv_err=False,
        sweep_tol=1.0e-12,
        davidson_tol=1.0e-10,
        enrichment=1.0e-2,
    ).run()

    assert solver.energy == pytest.approx(-1.0, abs=1.0e-10)
    assert solver.symmetry is True
    assert solver.sym_mgr.sym_types == ("charge",)
    assert solver.sweep_history[-1]["algorithm"] == "dmrg3s"
    assert all(row["local_tensor_rank"] == 3 for row in solver.sweep_history)
    assert all(
        update["dimension"] <= 8
        for row in solver.sweep_history
        for update in row["updates"]
    )
    if cpp_davidson.CPP_DAVIDSON_AVAILABLE and hasattr(
        cpp_davidson.AbelianTDVPSiteHeffPlan, "davidson"
    ):
        assert all(
            update["backend"] == "cpp-u1-site-davidson"
            for row in solver.sweep_history
            for update in row["updates"]
        )
    assert all(isinstance(site, AbelianSiteTensorData) for site in solver.state.factors)


def test_dmrg3s_u1xu1_enrichment_grows_product_state_bond():
    dense_mpo, _ = hubbard_2d_dense_mpo(
        2,
        1,
        hopping=1.0,
        hubbard_u=4.0,
    )
    qn_maps = site_qn_maps(2)
    mpo = dense_to_symmetric_mpo(
        dense_mpo,
        qn_maps,
        native_site_storage=True,
    )
    initial = random_fixed_sector_abelian_mps(
        2,
        1,
        1,
        max_bond_dim=1,
        qn_maps=qn_maps,
        native_site_storage=True,
        seed=4,
    )
    symmetry = SymmetryManager(["charge", "sz"])
    solver = DMRG(
        MPO(mpo),
        D=4,
        init_guess=initial,
        nsweeps=5,
        opt="3s",
        target_qn=symmetry.get_target_qn(2, 0),
        site_qn_maps=qn_maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-10,
        davidson_max_iter=50,
        enrichment=0.1,
        enrichment_decay=0.5,
        workers=2,
    ).run()

    exact = 0.5 * (4.0 - np.sqrt(4.0**2 + 16.0))
    assert solver.energy == pytest.approx(exact, abs=1.0e-9)
    assert solver.sym_mgr.sym_types == ("charge", "sz")
    assert solver.state.factors[0].shape[1] == 4
    assert any(
        update.get("expanded_states", 0) > 0
        for row in solver.sweep_history
        for update in row["updates"]
    )
    assert solver.environment_profile["factorization_workers"] == 2
    assert solver.environment_profile["enrich_rank"] == 32
    assert solver.environment_profile["enrich_seed"] == 0
    assert solver.environment_profile["energy_source"] == "canonical_boundary_local_problem"
    assert solver.environment_profile["right_environment_builds"] == 1
    assert solver.environment_profile["right_environment_reuses"] == max(
        0, len(solver.sweep_history) - 1
    )
    assert all(
        row["energy_source"] == "canonical_boundary_local_problem"
        for row in solver.sweep_history
    )
    enriched_updates = [
        update
        for row in solver.sweep_history
        for update in row["updates"]
        if "enrichment_mode" in update
    ]
    assert enriched_updates
    assert all(update["enrichment_mode"] == "streamed_low_rank" for update in enriched_updates)
    assert all(update["enrichment_states"] <= 32 for update in enriched_updates)


def test_dmrg_one_site_alias_selects_dmrg3s_validation():
    with pytest.raises(ValueError, match="symmetry=True"):
        DMRG(
            MPO([np.zeros((1, 1, 2, 2))]),
            D=1,
            init_guess=[np.ones((1, 2, 1))],
            nsweeps=1,
            opt="1site",
            symmetry=False,
            not_conv_err=False,
        ).run()
