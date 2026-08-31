import numpy as np
import pytest


pytest.importorskip("pyqed.mps.nonabelian._su2_kernel")
from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
from pyqed.qchem.dmrg.dmrg import DMRG as QCDMRG
from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF


def _environment():
    return SU2MovingEnvironment(
        np.zeros((2, 2)),
        np.zeros((2, 2, 2, 2)),
        2,
    )


def _packed_pool(arrays):
    arrays = tuple(np.ascontiguousarray(value, dtype=float) for value in arrays)
    return (
        np.concatenate([value.reshape(-1) for value in arrays]),
        np.cumsum([0, *(value.size for value in arrays)], dtype=np.int64),
        np.cumsum([0, *(value.ndim for value in arrays)], dtype=np.int64),
        np.asarray(
            [dimension for value in arrays for dimension in value.shape],
            dtype=np.int64,
        ),
    )


def _raw_source(boundary, local_operator):
    boundary_pool = _packed_pool((boundary,))
    operator_pool = _packed_pool((local_operator,))
    return (
        np.zeros(1, dtype=np.int64),
        np.zeros(1, dtype=np.int64),
        boundary_pool[1],
        boundary_pool[2],
        boundary_pool[3],
        boundary_pool[0],
        operator_pool[1],
        operator_pool[2],
        operator_pool[3],
        operator_pool[0],
    )


def _install_raw_routes(environment, in_indices, out_indices):
    rng = np.random.default_rng(29)
    left_source = _raw_source(
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(2, 1, 1, 1)),
    )
    right_source = _raw_source(
        rng.normal(size=(2, 2, 2)),
        rng.normal(size=(1, 2, 1, 1)),
    )
    n_entries = max(max(in_indices), max(out_indices)) + 1
    offsets = 4 * np.arange(n_entries, dtype=np.int64)
    shapes = np.tile([2, 1, 1, 2], (n_entries, 1))
    n_routes = len(in_indices)
    environment.install_raw_factor_routes(
        "wave-test",
        np.asarray(in_indices, dtype=np.int32),
        np.asarray(out_indices, dtype=np.int32),
        np.zeros(n_routes, dtype=np.int64),
        np.zeros(n_routes, dtype=np.int64),
        offsets,
        shapes,
        np.zeros(1, dtype=np.int64),
        left_source,
        np.zeros(1, dtype=np.int64),
        right_source,
        4 * n_entries,
        401,
        402,
    )


def test_su2_openmp_local_matvec_matches_serial():
    rng = np.random.default_rng(17)
    dimension = 96
    blocks = [
        rng.normal(size=(64, 48)) + 1j * rng.normal(size=(64, 48)),
        rng.normal(size=(48, 64)) + 1j * rng.normal(size=(48, 64)),
    ]
    vector = rng.normal(size=dimension) + 1j * rng.normal(size=dimension)
    environment = _environment()
    environment.install_local_operator(
        "openmp-test",
        blocks,
        input_starts=[0, 16],
        output_starts=[16, 0],
        dimension=dimension,
    )

    environment.set_num_threads(1)
    expected = environment.local_matvec("openmp-test", vector)
    environment.set_num_threads(4)
    actual = environment.local_matvec("openmp-test", vector)

    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
    info = environment.threading_info
    assert info["backend"] in {"openmp", "serial"}
    assert info["n_threads"] == (4 if info["available"] else 1)
    if info["available"]:
        assert info["parallel_regions"] >= 2
        assert info["tasks"] >= sum(block.shape[0] for block in blocks)


def test_su2_dense_pair_scheduler_parallelizes_disjoint_outputs():
    environment = _environment()
    _install_raw_routes(environment, range(8), range(8))
    stats = environment.stats
    assert stats["dense_pair_scheduler"] == "dependency_waves"
    assert stats["dense_pair_execution_count"] == 8
    assert stats["dense_pair_wave_count"] == 1
    assert stats["dense_pair_max_wave_width"] == 8

    vector = np.random.default_rng(31).normal(size=32)
    environment.set_num_threads(1)
    expected = environment.factor_route_real_matvec("wave-test", vector)
    environment.set_num_threads(4)
    actual = environment.factor_route_real_matvec("wave-test", vector)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)
    if environment.threading_info["available"]:
        assert environment.threading_info["parallel_regions"] >= 1


def test_su2_dense_pair_scheduler_serializes_conflicting_outputs():
    environment = _environment()
    _install_raw_routes(
        environment,
        [0, 0, 1, 1, 2, 2, 3, 3],
        [0, 1, 2, 3, 0, 2, 1, 3],
    )
    stats = environment.stats
    assert stats["dense_pair_execution_count"] == 4
    assert stats["dense_pair_wave_count"] == 2
    assert stats["dense_pair_max_wave_width"] == 2

    vector = np.random.default_rng(37).normal(size=16)
    environment.set_num_threads(1)
    expected = environment.factor_route_real_matvec("wave-test", vector)
    environment.set_num_threads(4)
    actual = environment.factor_route_real_matvec("wave-test", vector)
    np.testing.assert_allclose(actual, expected, rtol=2.0e-14, atol=2.0e-14)


@pytest.mark.parametrize("value", [0, -1])
def test_su2_thread_count_must_be_positive(value):
    with pytest.raises(ValueError, match="positive integer"):
        _environment().set_num_threads(value)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_su2_thread_count_must_be_an_integer(value):
    with pytest.raises(TypeError, match="positive integer"):
        _environment().set_num_threads(value)


@pytest.mark.parametrize("value", [True, 1.5, "2"])
def test_qchem_dmrg_rejects_noninteger_thread_count_before_build(value):
    dmrg = object.__new__(QCDMRG)
    with pytest.raises(TypeError, match="positive integer"):
        dmrg.run(n_threads=value)


@pytest.mark.parametrize("value", [0, -2])
def test_qchem_dmrg_rejects_nonpositive_thread_count_before_build(value):
    dmrg = object.__new__(QCDMRG)
    with pytest.raises(ValueError, match="positive"):
        dmrg.run(n_threads=value)


def test_qchem_su2_threads_reach_native_moving_environment():
    molecule = Molecule(
        atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g"
    )
    molecule.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mean_field = RHF(molecule).run()
    dmrg = QCDMRG(
        mean_field,
        ncas=2,
        nelecas=2,
        D=8,
        init_guess="hf",
        symmetry="su2",
        verbose=0,
    )

    dmrg.run(nsweeps=2, n_threads=2, require_convergence=False)

    np.testing.assert_allclose(dmrg.energy, -1.1372759437827158, atol=1.0e-10)
    threading = dmrg.dmrg.diagnostics["threading"]
    assert threading["n_threads"] == (2 if threading["available"] else 1)
    assert dmrg.history[-1]["threading"] == threading


def test_qchem_su2_fused_output_openmp_matches_serial():
    atom = "; ".join(f"H 0 0 {1.6 * index}" for index in range(6))
    molecule = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    molecule.build(eri="dense", aosym="s1", options={"eri_backend": "cpp"})
    mean_field = RHF(molecule).run()

    def run(n_threads):
        solver = QCDMRG(
            mean_field,
            ncas=6,
            nelecas=6,
            D=32,
            init_guess="cid",
            symmetry="su2",
            spatial_site_basis="fully_reduced",
            verbose=0,
        )
        engine = solver.run(
            nsweeps=1,
            n_threads=n_threads,
            require_convergence=False,
            su2_kernel_backend="cpp",
            conv_tol=-1.0,
            bond_multiplicity=4,
            seed=20260828,
            davidson_tol=1.0e-4,
            davidson_max_iter=40,
            mixer_nsweeps=0,
            mixer_zero_block_noise_scale=0.0,
        )
        half_sweep_energies = [
            float(row["energy"])
            for row in engine.history
            if row.get("direction") in {"lr", "rl"}
        ]
        stats = engine.history[-1]["moving_environment_stats"][
            "su2_moving_environment"
        ]
        return solver.energy, half_sweep_energies, engine.diagnostics, stats

    serial = run(1)
    parallel = run(4)

    np.testing.assert_allclose(parallel[0], serial[0], rtol=0.0, atol=1.0e-11)
    np.testing.assert_allclose(parallel[1], serial[1], rtol=0.0, atol=1.0e-11)
    for key in (
        "peak_persistent_output_batch_count",
        "peak_persistent_output_task_count",
        "peak_persistent_output_group_count",
    ):
        assert parallel[3][key] >= serial[3][key]
    assert parallel[3]["peak_persistent_output_task_count"] >= parallel[3][
        "peak_persistent_output_group_count"
    ]
    threading = parallel[2]["threading"]
    assert threading["n_threads"] == (4 if threading["available"] else 1)
    assert (threading["parallel_regions"] == 0) == (
        threading["tasks"] == 0
    )
