import numpy as np
import pytest

from pyqed.qchem.basis import ERI, S, T, point_charge
from pyqed.qchem.pbc.ewald import (
    ao_pair_ft_matrix_s,
    ewald_nuclear_repulsion_1d_inf_vacuum,
    gaussian_pair_ft,
    short_range_eri,
    short_range_eri_s,
    short_range_point_charge,
)
from pyqed.qchem.pbc import Cell, Chain, EwaldRHF, KRHF, RHF


def test_pbc_cell_builds_native_1d_and_makes_kpts():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    assert cell.built
    assert cell.nao == 1
    assert cell.nelectron == 2
    assert cell.low_dim_ft_type == "inf_vacuum"
    np.testing.assert_allclose(
        cell.lattice_vectors,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
    )

    kpts = cell.make_kpts(3)
    assert kpts.shape == (3, 3)
    np.testing.assert_allclose(kpts[1], np.zeros(3), atol=1e-12)


def test_pbc_chain_is_explicit_1d_api():
    chain = Chain(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        vacuum=20.0,
    ).build()

    assert isinstance(chain, Cell)
    assert chain.dimension == 1
    assert chain.lattice_constant == 3.0
    np.testing.assert_allclose(
        chain.lattice_vectors,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
    )


def test_pbc_chain_rhf_uses_native_1d_path():
    chain = Chain(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        vacuum=20.0,
    ).build()

    mf = chain.RHF(nimages=1, nk=3).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert mf.nkpts == 3
    assert len(mf.dm) == 3


def test_native_pbc_rhf_runs_gamma_and_kpoint_1d():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf_gamma = RHF(cell, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_gamma.e_tot)
    assert mf_gamma.nkpts == 1
    assert mf_gamma.dm.shape == (1, 1)
    assert np.isfinite(mf_gamma.e_nuc)

    kpts = cell.make_kpts(3)
    mf_k = RHF(cell, kpts=kpts, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_k.e_tot)
    assert mf_k.nkpts == 3
    assert len(mf_k.dm) == 3
    assert all(d.shape == (1, 1) for d in mf_k.dm)
    assert np.isfinite(mf_k.e_nuc)


def test_native_pbc_rhf_reports_unconverged_gamma_runs():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = RHF(cell, nimages=1).run(max_cycle=1, conv_tol=1e-12, conv_tol_dm=1e-12)
    assert not mf.converged


def test_native_pbc_rhf_accepts_nk_convenience_mesh():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(nimages=1, nk=3).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert mf.nkpts == 3
    assert len(mf.dm) == 3


def test_pbc_cell_has_native_ewald_nuclear_repulsion():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    e1 = cell.ewald_nuclear_repulsion(real_cut=3, recip_cut=6)
    e2 = cell.ewald_nuclear_repulsion(real_cut=4, recip_cut=8)
    assert np.isfinite(e1)
    assert np.isfinite(e2)
    assert abs(e2 - e1) < 5e-2


def test_s_gaussian_pair_fourier_matches_overlap_at_zero_g():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    pair_g0 = ao_pair_ft_matrix_s(cell.unit_molecule._bas, np.zeros(3))
    np.testing.assert_allclose(pair_g0, cell.unit_molecule.overlap, atol=1e-12)


def test_s_gaussian_pair_fourier_has_real_density_symmetry():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    gvec = np.asarray([0.3, -0.2, 0.1])
    pair_g = ao_pair_ft_matrix_s(cell.unit_molecule._bas, gvec)
    pair_minus_g = ao_pair_ft_matrix_s(cell.unit_molecule._bas, -gvec)
    np.testing.assert_allclose(pair_minus_g, pair_g.conj(), atol=1e-12)


def test_ewald_pair_fourier_keeps_finite_g_sp_terms_against_pyscf():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto
    from pyscf.pbc.df import ft_ao

    atom = "Li 0 0 0; H 3.0 0 0"
    lattice = np.diag([8.0, 8.0, 8.0])
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=2,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf._validate()
    mf._periodic_setup()

    g = np.pi / 4.0
    gvecs = np.asarray(
        [
            [0.0, 0.0, -g],
            [-g, 0.0, 0.0],
            [g, 0.0, 0.0],
            [0.0, -g, 0.0],
            [0.0, 0.0, g],
            [0.0, g, 0.0],
        ],
        dtype=float,
    )
    kpt = np.asarray(mf.kpts[0], dtype=float)
    native = mf._periodic_pair_ft_batch(gvecs, kpt)

    pyscf_cell = gto.Cell()
    pyscf_cell.atom = atom
    pyscf_cell.a = lattice
    pyscf_cell.basis = "sto-3g"
    pyscf_cell.unit = "B"
    pyscf_cell.spin = 0
    pyscf_cell.verbose = 0
    pyscf_cell.cart = True
    pyscf_cell.build()
    pyscf_ref = np.asarray(
        ft_ao.ft_aopair(
            pyscf_cell,
            gvecs,
            aosym="s1",
            kpti_kptj=np.asarray([kpt, kpt]),
        )
    )

    np.testing.assert_allclose(native[:, 0, 2:5], pyscf_ref[:, 0, 2:5], atol=5.0e-6)
    np.testing.assert_allclose(native, pyscf_ref, atol=5.0e-6)


def test_ewald_pair_fourier_compiled_block_matches_direct_shift_loop():
    from pyqed.qchem.fourier import has_compiled_ao_ft

    if not has_compiled_ao_ft():
        pytest.skip("compiled AO-pair Fourier backend is not available")

    cell = Cell(
        atom="Li 0 0 0; H 3.0 0 0",
        a=np.diag([8.0, 8.0, 8.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=1,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf._validate()
    mf._periodic_setup()

    assert mf._pair_ft_block_plan is not None
    gvecs = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 4.0, 0.0, 0.0],
            [0.0, -np.pi / 4.0, np.pi / 5.0],
        ],
        dtype=float,
    )
    kpt = np.asarray(mf.kpts[0], dtype=float)
    fast = mf._periodic_pair_ft_batch(gvecs, kpt)
    direct = mf._periodic_pair_ft_batch_direct(gvecs, kpt)

    np.testing.assert_allclose(fast, direct, atol=1.0e-12)


def test_pbc_cell_has_native_reciprocal_electronic_matrices():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    vne = cell.reciprocal_nuclear_attraction_matrix(recip_cut=8)
    dm = np.eye(cell.nao)
    jmat = cell.reciprocal_hartree_matrix(dm, recip_cut=8)

    assert vne.shape == (cell.nao, cell.nao)
    assert jmat.shape == (cell.nao, cell.nao)
    np.testing.assert_allclose(vne, vne.conj().T, atol=1e-12)
    np.testing.assert_allclose(jmat, jmat.conj().T, atol=1e-12)
    assert np.all(np.isfinite(vne.real))
    assert np.all(np.isfinite(jmat.real))


def test_native_reciprocal_electronic_matrices_are_cutoff_stable():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    eta = 0.5
    vne6 = cell.reciprocal_nuclear_attraction_matrix(recip_cut=6, eta=eta)
    vne8 = cell.reciprocal_nuclear_attraction_matrix(recip_cut=8, eta=eta)
    j6 = cell.reciprocal_hartree_matrix(np.eye(cell.nao), recip_cut=6, eta=eta)
    j8 = cell.reciprocal_hartree_matrix(np.eye(cell.nao), recip_cut=8, eta=eta)

    assert np.linalg.norm(vne8 - vne6) < 1e-2
    assert np.linalg.norm(j8 - j6) < 1e-2


def test_short_range_nuclear_attraction_matches_full_at_eta_zero():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    basis = cell.unit_molecule._bas
    kinetic = np.asarray([[T(a, b) for b in basis] for a in basis])
    full_vne = cell.unit_molecule.hcore - kinetic
    sr_vne = cell.short_range_nuclear_attraction_matrix(eta=0.0, real_cut=0)

    np.testing.assert_allclose(sr_vne, full_vne, atol=1e-10)


def test_short_range_nuclear_attraction_decays_with_eta():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_small_eta = cell.short_range_nuclear_attraction_matrix(eta=0.3, real_cut=1)
    sr_large_eta = cell.short_range_nuclear_attraction_matrix(eta=1.0, real_cut=1)
    assert np.linalg.norm(sr_large_eta) < np.linalg.norm(sr_small_eta)


def test_short_range_eri_matches_full_at_eta_zero():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_eri = cell.short_range_eri_tensor(eta=0.0)
    np.testing.assert_allclose(sr_eri, cell.unit_molecule.eri, atol=1e-10)


def test_short_range_eri_decays_with_eta():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_small_eta = cell.short_range_eri_tensor(eta=0.3)
    sr_large_eta = cell.short_range_eri_tensor(eta=1.0)
    assert np.linalg.norm(sr_large_eta) < np.linalg.norm(sr_small_eta)


def test_reciprocal_eri_tensor_has_basic_permutation_symmetry():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    eri = cell.reciprocal_eri_tensor(recip_cut=6, eta=0.5)
    assert eri.shape == (cell.nao, cell.nao, cell.nao, cell.nao)
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)


def test_native_ewald_rhf_runs_s_gaussian_gamma():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(method="ewald", eta=0.5, real_cut=3, recip_cut=6, mesh=(9, 10, 10)).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    assert mf.converged
    assert np.isfinite(mf.e_tot)
    assert np.isfinite(mf.e_nuc)
    assert mf.dm.shape == (cell.nao, cell.nao)
    assert np.isfinite(mf.madelung)


def test_native_ewald_rhf_runs_s_gaussian_kpoints():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(method="ewald", nk=3, eta=0.5, real_cut=1, recip_cut=4, mesh=(7, 8, 8)).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    assert mf.converged
    assert mf.nkpts == 3
    assert len(mf.dm) == 3
    assert all(d.shape == (cell.nao, cell.nao) for d in mf.dm)
    assert all(np.allclose(f, f.conj().T, atol=1e-10) for f in mf.fock)


def test_pbc_exposes_krhf_alias_for_ewald_solver():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    assert KRHF is EwaldRHF
    assert isinstance(cell.KRHF(nk=(1, 1, 1), eta=0.5), EwaldRHF)
    assert isinstance(cell.RHF(method="krhf", nk=(1, 1, 1), eta=0.5), EwaldRHF)


def test_native_ewald_krhf_uses_global_kpoint_occupations():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf = cell.KRHF(kpts=cell.make_kpts((2, 1, 1)), eta=0.5)
    fock = [
        np.diag([-2.0, -1.0]),
        np.diag([0.0, 10.0]),
    ]
    overlap = [np.eye(cell.nao), np.eye(cell.nao)]

    _mo_energy, _mo_coeff, mo_occ, dm = mf._solve_fock(fock, overlap)

    np.testing.assert_allclose(mo_occ[0], [2.0, 2.0])
    np.testing.assert_allclose(mo_occ[1], [0.0, 0.0])
    electron_count = sum(np.trace(d).real for d in dm) / mf.nkpts
    np.testing.assert_allclose(electron_count, cell.nelectron, atol=1e-12)


def test_native_ewald_krhf_fractionally_occupies_degenerate_frontier():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf = cell.KRHF(kpts=cell.make_kpts((2, 1, 1)), eta=0.5)
    fock = [
        np.diag([-2.0, 0.0]),
        np.diag([0.0, 10.0]),
    ]
    overlap = [np.eye(cell.nao), np.eye(cell.nao)]

    _mo_energy, mo_coeff, mo_occ, dm = mf._solve_fock(fock, overlap)
    rebuilt_dm = mf.make_rdm1(mo_coeff, mo_occ)

    np.testing.assert_allclose(mo_occ[0], [2.0, 1.0])
    np.testing.assert_allclose(mo_occ[1], [1.0, 0.0])
    for actual, rebuilt in zip(dm, rebuilt_dm):
        np.testing.assert_allclose(actual, rebuilt, atol=1e-12)
    electron_count = sum(np.trace(d).real for d in dm) / mf.nkpts
    np.testing.assert_allclose(electron_count, cell.nelectron, atol=1e-12)


def test_native_3d_hydrogen_cell_builds_and_makes_kpts():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    assert cell.built
    assert cell.dimension == 3
    assert cell.nao == 2
    kpts = cell.make_kpts((2, 2, 2))
    assert kpts.shape == (8, 3)
    assert np.all(np.isfinite(kpts))


def test_native_3d_ewald_rhf_runs_gamma_and_kpoints():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf_gamma = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    mf_nk1 = cell.RHF(method="ewald", nk=(1, 1, 1), eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    mf_k = cell.RHF(method="ewald", nk=(2, 2, 2), eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    assert mf_gamma.converged
    assert mf_nk1.converged
    assert mf_k.converged
    assert np.isfinite(mf_gamma.e_tot)
    assert np.isfinite(mf_k.e_tot)
    np.testing.assert_allclose(mf_gamma.e_tot, mf_nk1.e_tot, atol=1e-10)
    assert mf_k.nkpts == 8
    assert len(mf_k.dm) == 8
    assert all(d.shape == (cell.nao, cell.nao) for d in mf_k.dm)
    assert all(np.allclose(f, f.conj().T, atol=1e-10) for f in mf_k.fock)


def test_native_3d_hydrogen_band_structure_shapes():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=2).run(
        max_cycle=50,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    path = np.column_stack([np.linspace(-0.5, 0.5, 3), np.zeros(3), np.zeros(3)])

    bands = mf.band_structure(scaled_kpts=path, exchange="average")
    overlap_sorted = mf.band_structure(
        scaled_kpts=path,
        exchange="average",
        sort_bands="overlap",
    )

    assert bands["kpts"].shape == (3, 3)
    assert bands["mo_energy"].shape == (3, cell.nao)
    assert bands["mo_energy_reference"].shape == (3, cell.nao)
    assert overlap_sorted["mo_energy"].shape == (3, cell.nao)
    assert np.isfinite(bands["e_fermi"])
    assert np.all(np.isfinite(bands["mo_energy"]))
    assert np.all(np.isfinite(overlap_sorted["mo_energy"]))


def test_native_3d_hydrogen_mesh_interpolated_bands():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(
        method="ewald",
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="reciprocal",
    ).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    mesh_bands = mf.band_structure(kpts=mf.kpts, exchange="mesh")
    interp_at_mesh = mf.band_structure(kpts=mf.kpts, exchange="mesh_interpolate")
    path = np.column_stack([np.linspace(-0.5, 0.5, 5), np.zeros(5), np.zeros(5)])
    interp_path = mf.band_structure(scaled_kpts=path, exchange="mesh_interpolate")

    assert interp_at_mesh["interpolated"]
    np.testing.assert_allclose(
        interp_at_mesh["mo_energy"],
        mesh_bands["mo_energy"],
        atol=1e-10,
    )
    assert interp_path["mo_energy"].shape == (5, cell.nao)
    assert np.all(np.isfinite(interp_path["mo_energy"]))
    with pytest.raises(ValueError, match="self-consistent SCF k-points"):
        mf.band_structure(scaled_kpts=path, exchange="mesh")


def test_native_3d_reciprocal_jk_accepts_larger_pair_cut():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.RHF(
        method="ewald",
        kpts=kpts,
        eta=0.5,
        real_cut=0,
        pair_cut=1,
        recip_cut=2,
        jk_builder="reciprocal",
    ).run(max_cycle=80, conv_tol=1e-10, conv_tol_dm=1e-8)

    assert mf.converged
    assert mf.nkpts == 2
    assert all(np.allclose(fock, fock.conj().T, atol=1e-10) for fock in mf.fock)
    assert np.isfinite(mf.e_tot)


def test_optional_pyscf_3d_hydrogen_gamma_reference_scale():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.RHF(ref_cell).run(conv_tol=1e-10)

    assert ref_mf.converged
    assert mf.converged
    assert np.isfinite(ref_mf.e_tot)
    assert np.isfinite(mf.e_tot)
    assert abs(mf.e_tot - ref_mf.e_tot) < 1.0


def test_optional_pyscf_3d_hydrogen_centered_krhf_reference_scale():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.RHF(
        method="ewald",
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
        jk_builder="reciprocal",
    ).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.KRHF(ref_cell, kpts=kpts).run(conv_tol=1e-10)

    assert ref_mf.converged
    assert mf.converged
    assert abs(mf.e_tot - ref_mf.e_tot) < 5e-6

    scaled_path = np.column_stack(
        [np.linspace(-0.5, 0.5, 5), np.zeros(5), np.zeros(5)]
    )
    native_bands = mf.band_structure(scaled_kpts=scaled_path, exchange="finite_q")
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    ref_bands, _ = ref_mf.get_bands(scaled_path @ recip)
    assert np.max(np.abs(native_bands["mo_energy"] - ref_bands)) < 1e-5


def test_optional_pyscf_3d_hydrogen_one_body_reference():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    from pyqed.qchem.pbc.hf.ewald_rhf import EwaldRHF

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = EwaldRHF(cell, eta=0.5, real_cut=1, recip_cut=5)
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()
    overlap = mf._fourier_sum(mf._s_r, np.zeros(3))
    hcore = (
        mf._fourier_sum(mf._t_r, np.zeros(3))
        + mf._fourier_sum(mf._vne_sr_r, np.zeros(3))
        + mf._reciprocal_nuclear_attraction(np.zeros(3))
        + mf._nuclear_background_hcore(overlap)
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.RHF(ref_cell)

    assert np.linalg.norm(overlap - ref_mf.get_ovlp()) < 2e-3
    assert np.linalg.norm(hcore - ref_mf.get_hcore()) < 3e-3


def test_optional_pyscf_3d_hydrogen_madelung_reference():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_tools = pytest.importorskip("pyscf.pbc.tools.pbc")

    from pyqed.qchem.pbc.hf.ewald_rhf import EwaldRHF

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = EwaldRHF(cell, eta=0.5, real_cut=1, recip_cut=5)
    mf._validate()

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()

    np.testing.assert_allclose(
        mf._madelung(),
        pyscf_pbc_tools.madelung(ref_cell, np.zeros((1, 3))),
        atol=6e-4,
    )


def test_native_1d_inf_vacuum_probe_madelung_matches_reference_value():
    lattice = np.diag([4.0, 20.0, 20.0])
    probe_energy = ewald_nuclear_repulsion_1d_inf_vacuum(
        np.asarray([1.0]),
        np.zeros((1, 3)),
        lattice,
        eta=0.31622776601683794,
        real_cut=5,
        mesh=(15, 18, 18),
    )
    madelung = -2.0 * probe_energy
    np.testing.assert_allclose(madelung, -5.585196565523321, atol=1e-12)


def test_cartesian_p_d_pair_fourier_matches_overlap_at_zero_g():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)

    np.testing.assert_allclose(gaussian_pair_ft(p_fn, d_fn, np.zeros(3)), S(p_fn, d_fn), atol=1e-12)


def test_cartesian_p_d_short_range_point_charge_matches_full_at_eta_zero():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)
    center = np.asarray([0.2, -0.1, 0.3])

    sr = short_range_point_charge(p_fn, d_fn, center, eta=0.0)
    full = point_charge(p_fn, d_fn, center)
    np.testing.assert_allclose(sr, full, atol=1e-10)


def test_short_range_eri_s_shortcut_matches_generic_cartesian_integral():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    basis = cell.unit_molecule._bas

    expected = short_range_eri(basis[0], basis[1], basis[0], basis[1], eta=0.5)
    actual = short_range_eri_s(basis[0], basis[1], basis[0], basis[1], eta=0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_cartesian_p_d_short_range_eri_matches_full_at_eta_zero():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    s_fn = next(fn for fn in basis if sum(fn.shell) == 0)
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)

    sr = short_range_eri(d_fn, p_fn, s_fn, d_fn, eta=0.0)
    full = ERI(d_fn, p_fn, s_fn, d_fn)
    np.testing.assert_allclose(sr, full, atol=1e-10)
