import numpy as np
from types import SimpleNamespace

from pyqed.narg import NARG as PublicNARG
from pyqed.narg.qchem import NARG, SU2NARG
from pyqed.narg.qchem import su2_three_site as su2_three_site_module
from pyqed.narg.qchem import su2_backend as su2_backend_module
from pyqed.narg.qchem.su2_backend import SU2NARGBackend, resolve_su2_narg_backend
from pyqed.narg.qchem.su2_core import (
    scalar_hamiltonian_irrep_tensor,
    su2_product_symmetry,
    su2_projected_roots,
)
from pyqed.narg.qchem.su2_chain import (
    LowRankERI,
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
    assembled_hamiltonian_irrep_tensor,
    coalesce_bilinear_entries,
    local_reduced_operator,
    product_operator_irrep_tensor,
    product_tensor_angular_terms,
    reduced_product_tensor_irrep,
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
from pyqed.symmetry import Irrep, Leg


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


def test_projected_operator_helpers_preserve_the_supplied_leg():
    irrep = Irrep((0, 0))
    leg = Leg({irrep: 1}, symmetry=su2_product_symmetry())
    basis = {irrep: np.ones((1, 1))}

    scalar = scalar_hamiltonian_irrep_tensor(np.array([[2.0]]), leg, basis)
    block_basis = np.ones((1, 1))
    primitive_bases = {irrep: np.array([[1.0], [0.0], [0.0], [0.0]])}
    assembled = assembled_hamiltonian_irrep_tensor(
        np.diag([3.0, 0.0, 0.0, 0.0]),
        block_basis,
        leg,
        primitive_bases,
    )
    product = product_operator_irrep_tensor(
        np.diag([4.0, 0.0, 0.0, 0.0]),
        block_basis,
        leg,
        primitive_bases,
    )

    for tensor in (scalar, assembled, product):
        assert tensor.bra is leg
        assert tensor.ket is leg


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
