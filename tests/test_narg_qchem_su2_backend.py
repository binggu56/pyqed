import numpy as np
import pytest
from types import SimpleNamespace

from pyqed.narg import NARG as PublicNARG
from pyqed.narg.irrep_tensor import Irrep
from pyqed.narg.qchem import NARG, SU2NARG
from pyqed.narg.qchem import su2_three_site as su2_three_site_module
from pyqed.narg.qchem import su2_backend as su2_backend_module
from pyqed.narg.qchem.su2_backend import SU2NARGBackend, resolve_su2_narg_backend
from pyqed.narg.qchem.su2_core import su2_projected_roots
from pyqed.narg.qchem.su2_chain import (
    LowRankERI,
    block_identity_reduced_tensor,
    diagonalize_block,
    env_spin_can_couple,
    feasible_target_irreps,
    run_su2_narg_chain,
)
from pyqed.narg.qchem.su2_chain import (
    grow_one_site_direct_reduced,
    grown_component_v1_packages,
    grown_coupling_operators,
    grown_reduced_v1_packages,
    seed_exact_pair_composites,
)
from pyqed.narg.qchem.su2_three_site import (
    PackedBilinearEntries,
    accumulate_bilinear_entries,
    coalesce_bilinear_entries,
    local_reduced_operator,
    product_tensor_angular_terms,
    reduced_product_tensor_irrep,
    reduced_scalar_product_irrep_tensor,
    rotate_reduced_tensor_to_truncated,
    rotate_reduced_tensors_to_truncated,
)
from pyqed.narg.qchem.su2_two_site import (
    AdaptiveD,
    build_renormalized_two_site_block,
    build_two_site_su2_narg,
    diagonalize_all_sectors,
    truncate_to_D,
)
from pyqed.mps.nonabelian.coupling import clebsch_gordan
from pyqed.mps.su2 import SU2Irrep


def _hubbard_integrals(nsites: int, *, t: float = 0.7, u: float = 2.0):
    h1e = np.zeros((nsites, nsites), dtype=float)
    for site in range(nsites - 1):
        h1e[site, site + 1] = h1e[site + 1, site] = -float(t)
    eri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    for site in range(nsites):
        eri[site, site, site, site] = float(u)
    return h1e, eri


def _physical_random_integrals(nsites: int, *, seed: int = 23):
    rng = np.random.default_rng(seed)
    h1e = rng.normal(scale=0.4, size=(nsites, nsites))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(scale=0.08, size=(nsites, nsites, nsites, nsites))
    eri = 0.25 * (
        eri
        + eri.swapaxes(0, 1)
        + eri.swapaxes(2, 3)
        + eri.swapaxes(0, 1).swapaxes(2, 3)
    )
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1e, eri


def test_native_clebsch_gordan_matches_python_for_general_spins():
    from pyqed.narg.qchem import su2_native

    native = su2_native.clebsch_gordan_doubled
    if native is None:
        pytest.skip("optional SU(2)-NARG C++ extension is unavailable")
    for left_j2, right_j2, fused_j2 in ((3, 4, 5), (5, 3, 4), (4, 4, 6)):
        left = SU2Irrep(left_j2)
        right = SU2Irrep(right_j2)
        fused = SU2Irrep(fused_j2)
        for left_m2 in range(-left_j2, left_j2 + 1, 2):
            for right_m2 in range(-right_j2, right_j2 + 1, 2):
                fused_m2 = left_m2 + right_m2
                expected = clebsch_gordan(
                    left,
                    right,
                    fused,
                    left_m2,
                    right_m2,
                    fused_m2,
                )
                assert native(
                    left_j2,
                    right_j2,
                    fused_j2,
                    left_m2,
                    right_m2,
                    fused_m2,
                ) == pytest.approx(expected, abs=1.0e-13)


def _su2_chain_sector_roots(h1e, eri, *, D: int, nroots: int = 8, **kwargs):
    final_size = int(np.asarray(h1e).shape[0])
    D_by_size = {2: min(10, int(D))}
    D_by_size.update({nsites: int(D) for nsites in range(3, final_size)})
    chain = run_su2_narg_chain(
        h1e,
        eri,
        D_by_size,
        final_size=final_size,
        target_nelec=final_size,
        target_j2=0,
        backend="python",
        **kwargs,
    )
    roots, _ = diagonalize_block(
        chain.final,
        nelec=final_size,
        j2=0,
        nroots=nroots,
        backend="python",
    )
    return roots


def _assert_reduced_tensors_allclose(actual, expected, *, atol=1.0e-12):
    assert set(actual) == set(expected)
    for key, tensor in expected.items():
        assert set(actual[key].blocks) == set(tensor.blocks)
        for block_key, block in tensor.blocks.items():
            np.testing.assert_allclose(actual[key].blocks[block_key], block, atol=atol)


def test_grown_reduced_v1_packages_match_projected_components_random_and_hubbard():
    for h1e, eri in (_physical_random_integrals(4), _hubbard_integrals(4)):
        source = build_renormalized_two_site_block(
            h1e[:2, :2],
            eri[:2, :2, :2, :2],
            D=8,
            backend="python",
        )
        seed_exact_pair_composites(source, site_count=2)
        grown_narg = grow_one_site_direct_reduced(
            h1e[:3, :3],
            eri[:3, :3, :3, :3],
            source,
            target_nelec=None,
            build_branch_basis=False,
        )
        grown = grown_coupling_operators(grown_narg, include_even_composites=False)

        reduced = grown_reduced_v1_packages(source, grown, h1e, eri, future_sites=(3,))
        projected = grown_component_v1_packages(source, h1e, eri, future_sites=(3,))

        _assert_reduced_tensors_allclose(reduced, projected)


def test_reduced_composite_v1_spectrum_matches_projected_growth_random_and_hubbard():
    for h1e, eri in (_physical_random_integrals(6), _hubbard_integrals(6)):
        recursive = _su2_chain_sector_roots(
            h1e,
            eri,
            D=14,
            project_growth_hamiltonian=False,
            project_v1_packages=True,
        )
        projected = _su2_chain_sector_roots(
            h1e,
            eri,
            D=14,
            project_growth_hamiltonian=True,
            project_v1_packages=False,
        )
        np.testing.assert_allclose(recursive, projected, atol=1.0e-12)


def test_su2_python_backend_sector_matvec_and_diagonalization():
    backend = resolve_su2_narg_backend("python")
    block = np.array([[2.0, 0.25j], [-0.25j, 3.0]], dtype=complex)
    vector = np.array([1.0 - 0.2j, 0.3 + 0.4j], dtype=complex)

    np.testing.assert_allclose(backend.sector_matvec(block, vector), block @ vector)

    result = backend.diagonalize_sector(block, nroots=1)
    np.testing.assert_allclose(result.values, np.linalg.eigvalsh(block)[:1])
    assert result.vectors.shape == (2, 1)
    assert backend.summary()["name"] == "python"


def test_su2_native_openmp_operator_rotation_matches_numpy():
    from pyqed.narg.qchem import su2_native

    if su2_native.rotate_operator_blocks is None:
        pytest.skip("optional SU(2)-NARG C++ extension is unavailable")
    rng = np.random.default_rng(71)
    specs = []
    expected = []
    for _ in range(8):
        u_bra = rng.normal(size=(48, 24)) + 1j * rng.normal(size=(48, 24))
        block = rng.normal(size=(48, 52)) + 1j * rng.normal(size=(48, 52))
        u_ket = rng.normal(size=(52, 20)) + 1j * rng.normal(size=(52, 20))
        specs.append((u_bra, block, u_ket))
        expected.append(u_bra.conj().T @ block @ u_ket)

    threads = 4 if su2_native.openmp_available() else 1
    before = su2_native.openmp_info()
    try:
        assert su2_native.set_num_threads(threads) == threads
        actual = su2_native.rotate_operator_blocks(specs)
    finally:
        su2_native.set_num_threads(1)

    for block, reference in zip(actual, expected):
        np.testing.assert_allclose(block, reference, atol=5.0e-13)
    after = su2_native.openmp_info()
    if threads > 1:
        assert after["parallel_regions"] > before["parallel_regions"]
        assert after["tasks"] >= before["tasks"] + len(specs)


def test_su2_native_openmp_bilinear_wave_matches_numpy():
    from pyqed.narg.qchem import su2_native

    if su2_native.accumulate_bilinear_wave is None:
        pytest.skip("optional SU(2)-NARG growth-wave kernel is unavailable")
    rng = np.random.default_rng(83)
    specs = []
    expected = []
    for _ in range(8):
        size = 512
        rows = rng.integers(0, 32, size=size, dtype=np.int64)
        cols = rng.integers(0, 32, size=size, dtype=np.int64)
        block_rows = rng.integers(0, 16, size=size, dtype=np.int64)
        block_cols = rng.integers(0, 16, size=size, dtype=np.int64)
        local_rows = rng.integers(0, 4, size=size, dtype=np.int64)
        local_cols = rng.integers(0, 4, size=size, dtype=np.int64)
        coeffs = rng.normal(size=size) + 1j * rng.normal(size=size)
        block = rng.normal(size=(16, 16)) + 1j * rng.normal(size=(16, 16))
        local = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
        prefactor = 0.75 - 0.2j
        group = (
            rows,
            cols,
            block_rows,
            block_cols,
            local_rows,
            local_cols,
            coeffs,
            block,
            local,
            prefactor,
        )
        specs.append((32, 32, (group,)))
        reference = np.zeros((32, 32), dtype=complex)
        np.add.at(
            reference,
            (rows, cols),
            prefactor
            * coeffs
            * block[block_rows, block_cols]
            * local[local_rows, local_cols],
        )
        expected.append(reference)

    threads = 4 if su2_native.openmp_available() else 1
    before = su2_native.openmp_info()
    try:
        su2_native.set_num_threads(threads)
        actual = su2_native.accumulate_bilinear_wave(specs)
    finally:
        su2_native.set_num_threads(1)

    for value, reference in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=2.0e-13)
    after = su2_native.openmp_info()
    if threads > 1:
        assert after["parallel_regions"] > before["parallel_regions"]
        assert after["tasks"] >= before["tasks"] + len(specs)


def test_compiled_su2_backend_threads_preserve_chain_energy():
    compiled = resolve_su2_narg_backend("compiled", threads=1)
    if not compiled.capabilities.openmp:
        pytest.skip("compiled SU(2)-NARG backend has no OpenMP support")
    h1e, eri = _hubbard_integrals(4)
    try:
        serial = run_su2_narg_chain(
            h1e,
            eri,
            {2: 8, 3: 12},
            final_size=4,
            target_nelec=4,
            target_j2=0,
            backend=compiled,
            threads=1,
        )
        parallel = run_su2_narg_chain(
            h1e,
            eri,
            {2: 8, 3: 12},
            final_size=4,
            target_nelec=4,
            target_j2=0,
            backend=compiled,
            threads=4,
        )
        serial_energy, _ = diagonalize_block(
            serial.final, nelec=4, j2=0, nroots=2, backend=compiled
        )
        parallel_energy, _ = diagonalize_block(
            parallel.final, nelec=4, j2=0, nroots=2, backend=compiled
        )
    finally:
        compiled.configure_threads(1)

    np.testing.assert_allclose(parallel_energy, serial_energy, atol=1.0e-12)
    assert parallel.backend["threads"] == 4
    assert parallel.backend["openmp"] is True


def test_public_su2_narg_propagates_threads_to_backend():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        @staticmethod
        def energy_nuc():
            return 0.0

    backend = resolve_su2_narg_backend("compiled", threads=1)
    if not backend.capabilities.openmp:
        pytest.skip("compiled SU(2)-NARG backend has no OpenMP support")
    h1e, eri = _hubbard_integrals(4)
    try:
        solver = NARG(
            SimpleNamespace(mol=DummyMol()),
            mol=DummyMol(),
            symmetry="spin",
            h1e=h1e,
            eri=eri,
            D=8,
            nstates=1,
            threads=3,
        ).run()
    finally:
        backend.configure_threads(1)

    assert solver.timings["threads"] == 3
    assert solver.backend["threads"] == 3
    assert solver.backend["openmp"] is True
    assert np.isfinite(solver.e_tot[0])


def test_public_su2_narg_uses_constructor_integrals():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4)
    mol = DummyMol()
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        symmetry="su2",
        h1e=h1e,
        eri=eri,
        D=8,
        nstates=1,
        final_size=4,
        target_nelec=4,
        target_j2=0,
        su2_backend="python",
    )

    returned = solver.run()
    energies, block = solver
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8, 3: 8},
        final_size=4,
        target_nelec=4,
        target_j2=0,
        backend="python",
    )
    reference, _ = diagonalize_block(
        chain.final,
        nelec=4,
        j2=0,
        nroots=1,
        backend="python",
    )

    assert isinstance(solver, SU2NARG)
    assert returned is solver
    assert block.size
    np.testing.assert_allclose(energies, reference, atol=1.0e-12)


def test_public_su2_detached_frames_choose_seed_and_parent_capacity():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        @staticmethod
        def energy_nuc():
            return 0.0

    h1e, eri = _hubbard_integrals(6)
    mol = DummyMol()
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        symmetry="su2",
        h1e=h1e,
        eri=eri,
        D=2,
        nstates=1,
        target_nelec=6,
        target_j2=0,
        su2_backend="python",
        project_v1_packages=False,
        dressing="detached_frames",
    ).run()

    assert solver.n0 == 3
    assert solver.chi == 32
    assert set(solver.timings["detached_by_size"]) == {4, 5, 6}
    assert np.isfinite(solver.e_tot[0])
    rdm1 = solver.make_rdm1()
    np.testing.assert_allclose(np.trace(rdm1), 6.0, atol=1.0e-8)


def test_top_level_public_narg_dispatches_to_su2_qchem_driver():
    class DummyMol:
        nelec = (1, 1)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(2)
    mol = DummyMol()

    solver = PublicNARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        symmetry="su2",
        h1e=h1e,
        eri=eri,
        D=8,
        nstates=1,
        final_size=2,
        target_nelec=2,
        target_j2=0,
        su2_backend="python",
    ).run()

    assert isinstance(solver, SU2NARG)
    assert solver.e_tot.shape == (1,)
    assert solver.block.size


def test_public_su2_narg_uses_reduced_composite_path_by_default():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4)
    exact, _, _, _ = su2_projected_roots(h1e, eri, nelec=4, j2=0, nroots=1)
    mol = DummyMol()

    solver = PublicNARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        symmetry="su2",
        h1e=h1e,
        eri=eri,
        D=128,
        nstates=1,
        final_size=4,
        target_nelec=4,
        target_j2=0,
        su2_backend="python",
    ).run()

    assert solver.timings["variational"] is False
    assert solver.timings["project_growth_hamiltonian"] is False
    assert solver.timings["project_v1_packages"] is True
    np.testing.assert_allclose(solver.e_tot, exact, atol=1.0e-10)


def test_public_su2_projected_growth_makes_recursive_block_variational():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4)
    exact, _, _, _ = su2_projected_roots(h1e, eri, nelec=4, j2=0, nroots=1)
    mol = DummyMol()

    solver = PublicNARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        symmetry="su2",
        h1e=h1e,
        eri=eri,
        D=128,
        nstates=1,
        final_size=4,
        target_nelec=4,
        target_j2=0,
        su2_backend="python",
        variational=False,
        project_growth_hamiltonian=True,
    ).run()

    assert solver.timings["variational"] is False
    assert solver.timings["project_growth_hamiltonian"] is True
    np.testing.assert_allclose(solver.e_tot, exact, atol=1.0e-10)


def test_su2_backend_batched_rotation_matches_block_rotation():
    backend = SU2NARGBackend()
    old_threshold = su2_backend_module.ROTATION_BATCH_MIN_BLOCKS
    su2_backend_module.ROTATION_BATCH_MIN_BLOCKS = 2
    try:
        u_bra = np.array([[1.0, 0.0], [0.2j, 0.9]], dtype=complex)
        u_ket = np.array([[0.7, 0.1j], [0.0, 1.1]], dtype=complex)
        blocks = [
            np.array([[1.0, 0.3], [0.2j, 2.0]], dtype=complex),
            np.array([[0.2, -0.1j], [1.4, -0.5]], dtype=complex),
        ]
        specs = [(index, u_bra, block, u_ket) for index, block in enumerate(blocks)]

        rotated = backend.rotate_operator_blocks(specs)
    finally:
        su2_backend_module.ROTATION_BATCH_MIN_BLOCKS = old_threshold

    assert [key for key, _ in rotated] == [0, 1]
    for (_, block), source in zip(rotated, blocks):
        np.testing.assert_allclose(block, u_bra.conj().T @ source @ u_ket)


def test_su2_truncation_uses_backend_boundary():
    h1e, eri = _hubbard_integrals(2)
    narg = build_two_site_su2_narg(h1e, eri)
    backend = resolve_su2_narg_backend("python")

    direct = truncate_to_D(narg, D=6, allowed_nelec={1, 2, 3}, backend="python")
    resolved = truncate_to_D(narg, D=6, allowed_nelec={1, 2, 3}, backend=backend)

    assert [root.irrep for root in direct.kept_roots] == [root.irrep for root in resolved.kept_roots]
    np.testing.assert_allclose(direct.hamiltonian.to_dense(), resolved.hamiltonian.to_dense())


def test_su2_adaptive_truncation_uses_energy_window():
    h1e, eri = _hubbard_integrals(2)
    narg = build_two_site_su2_narg(h1e, eri)
    spec = AdaptiveD(D_min=3, D_max=8, energy_window=0.5)

    truncated = truncate_to_D(narg, D=spec, allowed_nelec={1, 2, 3}, backend="python")
    roots = diagonalize_all_sectors(
        narg,
        allowed_nelec={1, 2, 3},
        nroots=spec.D_max,
        backend="python",
    )
    inside_window = sum(root.energy <= roots[0].energy + spec.energy_window + 1.0e-12 for root in roots)
    expected = min(spec.D_max, max(spec.D_min, inside_window), len(roots))

    assert len(truncated.kept_roots) == expected
    np.testing.assert_allclose(
        [root.energy for root in truncated.kept_roots],
        [root.energy for root in roots[:expected]],
    )


def test_su2_spin_reachability_uses_remaining_sites():
    assert env_spin_can_couple(block_j2=1, env_nelec=1, env_sites=1, target_j2=0)
    assert not env_spin_can_couple(block_j2=2, env_nelec=1, env_sites=1, target_j2=0)
    assert env_spin_can_couple(block_j2=2, env_nelec=2, env_sites=2, target_j2=2)


def test_su2_fixed_D_chain_filters_seed_block_by_target_spin():
    h1e, eri = _hubbard_integrals(3)

    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8},
        final_size=3,
        target_nelec=3,
        target_j2=3,
        backend="python",
    )

    assert feasible_target_irreps(
        block_sites=2,
        final_sites=3,
        target_nelec=3,
        target_j2=3,
    ) == {chain.blocks[2].truncated.kept_roots[0].irrep}
    assert {root.irrep.charge for root in chain.blocks[2].truncated.kept_roots} == {(2, 2)}
    roots, block = diagonalize_block(
        chain.final,
        nelec=3,
        j2=3,
        nroots=1,
        backend="python",
    )
    assert block.size
    assert roots.size


def test_su2_two_site_block_builds_with_backend():
    h1e, eri = _hubbard_integrals(2)
    block = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")

    assert block.truncated.kept_roots
    assert ("Cdag", 0) in block.reduced_operators
    assert ("Ctilde", 1) in block.reduced_operators


def test_su2_chain_final_size_two_uses_seed_block():
    h1e, eri = _hubbard_integrals(2)
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8},
        final_size=2,
        target_nelec=2,
        target_j2=0,
        backend="python",
    )
    roots, block = diagonalize_block(
        chain.final,
        nelec=2,
        j2=0,
        nroots=1,
        backend="python",
    )

    assert chain.final is chain.blocks[2]
    assert block.size
    assert np.all(np.isfinite(roots))


def test_su2_packed_bilinear_accumulation_matches_raw_entries():
    old_flag = su2_three_site_module.SU2_COMPILED_ANGULAR
    try:
        su2_three_site_module.SU2_COMPILED_ANGULAR = False
        h1e, eri = _hubbard_integrals(2)
        block = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
        block_tensor = block.reduced_operators[("Cdag", 0)]
        local_tensor = local_reduced_operator("JW")
        site, _, terms = product_tensor_angular_terms(
            block,
            block_tensor.op,
            local_tensor.op,
            total_rank2=1,
        )
    finally:
        su2_three_site_module.SU2_COMPILED_ANGULAR = old_flag

    packed = next(
        entries for entries in terms.values() if isinstance(entries, PackedBilinearEntries)
    )
    bra_irrep, ket_irrep = next(key for key, value in terms.items() if value is packed)
    shape = (site.sector_dim(bra_irrep), site.sector_dim(ket_irrep))
    raw_block = np.zeros(shape, dtype=complex)
    packed_block = np.zeros(shape, dtype=complex)

    accumulate_bilinear_entries(raw_block, packed.entries, block_tensor, local_tensor)
    accumulate_bilinear_entries(packed_block, packed, block_tensor, local_tensor)

    np.testing.assert_allclose(packed_block, raw_block, atol=1.0e-12)


def test_su2_bilinear_coalescing_sums_duplicate_addresses():
    old_flag = su2_three_site_module.SU2_COMPILED_ANGULAR
    try:
        su2_three_site_module.SU2_COMPILED_ANGULAR = False
        h1e, eri = _hubbard_integrals(2)
        block = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
        block_tensor = block.reduced_operators[("Cdag", 0)]
        local_tensor = local_reduced_operator("JW")
        _, _, terms = product_tensor_angular_terms(
            block,
            block_tensor.op,
            local_tensor.op,
            total_rank2=1,
        )
    finally:
        su2_three_site_module.SU2_COMPILED_ANGULAR = old_flag

    packed = next(
        entries for entries in terms.values() if isinstance(entries, PackedBilinearEntries)
    )
    entry = packed.entries[0]
    duplicate_entries = (entry, entry[:2] + (-entry[2],) + entry[3:])

    assert coalesce_bilinear_entries(duplicate_entries) == ()


def test_su2_compiled_product_tensor_matches_python_when_available():
    if not su2_three_site_module.SU2_COMPILED_ANGULAR:
        return

    h1e, eri = _hubbard_integrals(2)
    block_python = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
    block_compiled = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
    local_tensor = local_reduced_operator("JW")

    old_flag = su2_three_site_module.SU2_COMPILED_ANGULAR
    old_threshold = su2_three_site_module.SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS
    try:
        su2_three_site_module.SU2_COMPILED_ANGULAR = False
        python_tensor = reduced_product_tensor_irrep(
            block_python,
            block_python.reduced_operators[("Cdag", 0)],
            local_tensor,
            total_rank2=1,
        )
        su2_three_site_module.SU2_COMPILED_ANGULAR = True
        su2_three_site_module.SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS = 0
        compiled_tensor = reduced_product_tensor_irrep(
            block_compiled,
            block_compiled.reduced_operators[("Cdag", 0)],
            local_tensor,
            total_rank2=1,
        )
    finally:
        su2_three_site_module.SU2_COMPILED_ANGULAR = old_flag
        su2_three_site_module.SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS = old_threshold

    assert set(compiled_tensor.blocks) == set(python_tensor.blocks)
    for key, block in compiled_tensor.blocks.items():
        np.testing.assert_allclose(block, python_tensor.blocks[key], atol=1.0e-12)


def test_su2_compiled_scalar_product_matches_python_when_available():
    h1e, eri = _hubbard_integrals(2)
    python_block = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
    compiled_block = build_renormalized_two_site_block(h1e, eri, D=8, backend="python")
    local_tensor = local_reduced_operator("JWCtilde")

    old_flag = su2_three_site_module.SU2_COMPILED_ANGULAR
    try:
        su2_three_site_module.SU2_COMPILED_ANGULAR = False
        python_tensor = reduced_scalar_product_irrep_tensor(
            python_block,
            python_block.reduced_operators[("Cdag", 0)],
            local_tensor,
        )
        su2_three_site_module.SU2_COMPILED_ANGULAR = True
        compiled_tensor = reduced_scalar_product_irrep_tensor(
            compiled_block,
            compiled_block.reduced_operators[("Cdag", 0)],
            local_tensor,
        )
    finally:
        su2_three_site_module.SU2_COMPILED_ANGULAR = old_flag

    assert set(compiled_tensor.blocks) == set(python_tensor.blocks)
    for key, value in python_tensor.blocks.items():
        np.testing.assert_allclose(compiled_tensor.blocks[key], value, atol=1.0e-12)


def test_su2_growth_wave_matches_individual_reduced_products():
    from pyqed.narg.qchem import su2_native

    h1e, eri = _hubbard_integrals(3)
    source = build_renormalized_two_site_block(
        h1e[:2, :2],
        eri[:2, :2, :2, :2],
        D=8,
        backend="python",
    )
    grown_narg = grow_one_site_direct_reduced(
        h1e,
        eri,
        source,
        target_nelec=None,
        build_branch_basis=False,
    )
    before = su2_native.openmp_info() if su2_native.openmp_info else None
    actual = grown_coupling_operators(
        grown_narg,
        include_even_composites=False,
    )
    after = su2_native.openmp_info() if su2_native.openmp_info else None
    if su2_native.reduced_growth_graph is not None:
        assert after["growth_graph_calls"] > before["growth_graph_calls"]
        assert after["growth_graph_plans"] > before["growth_graph_plans"]

    identity = block_identity_reduced_tensor(source)
    expected = {}
    for key in actual:
        name, site = key
        if site < 2:
            block_tensor = source.reduced_operators[key]
            local_tensor = local_reduced_operator("JW")
        else:
            block_tensor = identity
            local_tensor = local_reduced_operator(name)
        expected[key] = reduced_product_tensor_irrep(
            source,
            block_tensor,
            local_tensor,
            total_rank2=1,
        )

    assert set(actual) == set(expected)
    for key, expected_tensor in expected.items():
        assert set(actual[key].blocks) == set(expected_tensor.blocks)
        for block_key, expected_block in expected_tensor.blocks.items():
            np.testing.assert_allclose(
                actual[key].blocks[block_key],
                expected_block,
                atol=1.0e-12,
            )


def test_su2_operator_rotation_uses_backend_batch_hook():
    class TrackingBackend(SU2NARGBackend):
        def __init__(self):
            self.rotate_batches = 0
            self.rotated_blocks = 0

        def rotate_operator_blocks(self, block_specs):
            self.rotate_batches += 1
            self.rotated_blocks += len(block_specs)
            return super().rotate_operator_blocks(block_specs)

    h1e, eri = _hubbard_integrals(2)
    backend = TrackingBackend()
    block = build_renormalized_two_site_block(h1e, eri, D=8, backend=backend)

    assert block.reduced_operators
    assert backend.rotate_batches > 0
    assert backend.rotated_blocks > 0


def test_su2_bulk_operator_projection_matches_individual_projection():
    class TrackingBackend(SU2NARGBackend):
        def __init__(self):
            self.rotate_batches = 0
            self.rotated_blocks = 0

        def rotate_operator_blocks(self, block_specs):
            self.rotate_batches += 1
            self.rotated_blocks += len(block_specs)
            return super().rotate_operator_blocks(block_specs)

    h1e, eri = _hubbard_integrals(3)
    source = build_renormalized_two_site_block(
        h1e[:2, :2],
        eri[:2, :2, :2, :2],
        D=8,
        backend="python",
    )
    grown_narg = grow_one_site_direct_reduced(
        h1e,
        eri,
        source,
        target_nelec=None,
        build_branch_basis=False,
    )
    truncated = truncate_to_D(grown_narg, D=16, allowed_nelec={1, 2, 3}, backend="python")
    tensors = grown_coupling_operators(grown_narg)

    individual = {
        key: rotate_reduced_tensor_to_truncated(truncated, tensor, backend="python")
        for key, tensor in tensors.items()
    }
    backend = TrackingBackend()
    bulk = rotate_reduced_tensors_to_truncated(truncated, tensors, backend=backend)

    assert backend.rotate_batches == 1
    assert backend.rotated_blocks > 0
    assert set(bulk) == set(individual)
    for key, tensor in individual.items():
        assert set(bulk[key].blocks) == set(tensor.blocks)
        for block_key, block in tensor.blocks.items():
            np.testing.assert_allclose(bulk[key].blocks[block_key], block)


def test_su2_chain_accepts_backend():
    h1e, eri = _hubbard_integrals(3)

    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8},
        final_size=3,
        target_nelec=3,
        backend="python",
    )

    assert chain.backend["name"] == "python"
    roots, block = diagonalize_block(
        chain.final,
        nelec=3,
        j2=1,
        nroots=2,
        backend="python",
    )
    assert block.size
    assert roots.size
    assert np.all(np.isfinite(roots))


def test_su2_chain_accepts_adaptive_D_and_reports_kept_counts():
    h1e, eri = _hubbard_integrals(4)
    spec = AdaptiveD(D_min=4, D_max=12, energy_window=0.25)

    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8, 3: spec},
        final_size=4,
        target_nelec=4,
        backend="python",
    )

    assert chain.timings["D_by_size"][3]["adaptive"] is True
    assert 4 <= chain.timings["kept_by_size"][3] <= 12
    roots, block = diagonalize_block(
        chain.final,
        nelec=4,
        j2=0,
        nroots=2,
        backend="python",
    )
    assert block.size
    assert roots.size
    assert np.all(np.isfinite(roots))


def test_su2_future_cc_uses_reduced_future_couplings_and_preserves_sectors():
    h1e, eri = _hubbard_integrals(5)
    common = dict(
        D_by_size={2: 3, 3: 3, 4: 3},
        final_size=5,
        target_nelec=5,
        target_j2=1,
        backend="python",
        project_v1_packages=False,
    )
    plain = run_su2_narg_chain(h1e, eri, **common)
    dressed = run_su2_narg_chain(
        h1e,
        eri,
        **common,
        dressing="future_cc",
        future_cc_strength=0.2,
    )
    plain_energy = diagonalize_block(
        plain.final,
        nelec=5,
        j2=1,
        nroots=1,
        backend="python",
    )[0][0]
    dressed_energy = diagonalize_block(
        dressed.final,
        nelec=5,
        j2=1,
        nroots=1,
        backend="python",
    )[0][0]

    assert dressed_energy < plain_energy - 1.0e-3
    assert dressed.timings["dressing"] == "future_cc"
    assert dressed.timings["future_cc_by_size"]
    assert any(
        item["discarded_source_norm"] > 1.0e-8
        for item in dressed.timings["future_cc_by_size"].values()
    )
    assert all(
        item["maximum_response_residual"] < 1.0e-8
        for item in dressed.timings["future_cc_by_size"].values()
    )
    for block in dressed.blocks.values():
        for (bra, ket), transform in block.transform.blocks.items():
            assert bra == ket
            np.testing.assert_allclose(
                transform.conj().T @ transform,
                np.eye(transform.shape[1]),
                atol=1.0e-10,
            )


def test_su2_detached_frames_keep_all_post_seed_rayleigh_solves_at_most_D():
    h1e, eri = _hubbard_integrals(4)
    exact = su2_projected_roots(h1e, eri, nelec=4, j2=0, nroots=1)[0][0]
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {2: 2, 3: 2},
        final_size=4,
        target_nelec=4,
        target_j2=0,
        backend="python",
        project_v1_packages=False,
        dressing="detached_frames",
    )
    energy = diagonalize_block(
        chain.final,
        nelec=4,
        j2=0,
        nroots=1,
        backend="python",
    )[0][0]

    assert energy >= exact - 1.0e-10
    assert np.isfinite(energy)
    assert chain.timings["dressing"] == "detached_frames"
    assert set(chain.timings["detached_by_size"]) == {3, 4}
    for diagnostics in chain.timings["detached_by_size"].values():
        assert diagnostics["branch_ranks"] == (2, 2, 2)
        assert diagnostics["baseline_rank"] == 2
        assert diagnostics["protected_per_branch"] == 0
        assert diagnostics["cross_product_basis"] is True
        assert diagnostics["strict_D_rayleigh"] is True
        assert diagnostics["initial_frame_rank"] == sum(
            diagnostics["branch_ranks"]
        )
        assert diagnostics["detached_dim"] >= diagnostics["frame_union_rank"]
        assert diagnostics["orthogonality_error"] < 1.0e-10
        assert diagnostics["target_dim"] == 2
        assert diagnostics["maximum_eigensolve_order"] <= 2
        assert diagnostics["retained_dim"] <= diagnostics["chi"]
    assert any(
        diagnostics["maximum_ambient_dimension"] > 2
        for diagnostics in chain.timings["detached_by_size"].values()
    )
    assert chain.timings["final_target_dim"] == 2
    assert chain.final.hamiltonian.block(Irrep((4, 0)), Irrep((4, 0))).shape == (2, 2)
    for block in chain.blocks.values():
        for (bra, ket), transform in block.transform.blocks.items():
            assert bra == ket
            np.testing.assert_allclose(
                transform.conj().T @ transform,
                np.eye(transform.shape[1]),
                atol=1.0e-10,
            )

    combined = run_su2_narg_chain(
        h1e,
        eri,
        {2: 2, 3: 2},
        final_size=4,
        target_nelec=4,
        target_j2=0,
        backend="python",
        project_v1_packages=False,
        dressing="detached+cc",
    )
    combined_energy = diagonalize_block(
        combined.final,
        nelec=4,
        j2=0,
        nroots=1,
        backend="python",
    )[0][0]
    assert combined_energy >= exact - 1.0e-10
    assert set(combined.timings["cc_by_size"]) == {3}
    cc = combined.timings["cc_by_size"][3]
    assert cc["response_rank"] > 0
    assert cc["sector_energy_gain"] >= -1.0e-12
    assert cc["maximum_response_residual"] < 1.0e-8
    assert cc["iterative_fallbacks"] == 0


def test_su2_detached_frames_use_exact_seed_and_rolling_parent():
    h1e, eri = _hubbard_integrals(6)
    chain = run_su2_narg_chain(
        h1e,
        eri,
        {size: 2 for size in range(2, 6)},
        final_size=6,
        target_nelec=6,
        target_j2=0,
        backend="python",
        project_v1_packages=False,
        dressing="detached_frames",
    )

    assert chain.timings["n0"] == 3
    assert chain.timings["chi"] == 32
    assert set(chain.timings["detached_by_size"]) == {4, 5, 6}
    assert chain.timings["kept_by_size"][3] > 2
    for size in (4, 5):
        diagnostics = chain.timings["detached_by_size"][size]
        assert diagnostics["branch_ranks"] == (2, 2, 2)
        assert diagnostics["target_dim"] == 2
        assert diagnostics["parent_dim"] > diagnostics["target_dim"]
        assert len(chain.blocks[size]._su2_target_truncated.kept_roots) == 2
        assert len(chain.blocks[size].truncated.kept_roots) == diagnostics["parent_dim"]


def test_su2_low_rank_eri_chain_matches_dense_v1_packages():
    h1e, eri = _hubbard_integrals(4)

    dense = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8, 3: 8},
        final_size=4,
        target_nelec=4,
        backend="python",
    )
    factorized = run_su2_narg_chain(
        h1e,
        eri,
        {2: 8, 3: 8},
        final_size=4,
        target_nelec=4,
        backend="python",
        low_rank_eri=LowRankERI.from_dense(eri, tol=1e-12),
    )

    np.testing.assert_allclose(
        factorized.final.hamiltonian.to_dense(),
        dense.final.hamiltonian.to_dense(),
        atol=1e-10,
    )
