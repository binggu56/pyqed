import numpy as np

from pyqed.qchem.basis import ERI, S, T, point_charge
from pyqed.qchem.pbc.ewald import (
    ao_pair_ft_matrix_s,
    ewald_nuclear_repulsion_1d_inf_vacuum,
    gaussian_pair_ft,
    short_range_eri,
    short_range_point_charge,
)
from pyqed.qchem.pbc import Cell, Chain, RHF


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


def test_native_ewald_rhf_rejects_kpoints_for_now():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    try:
        cell.RHF(method="ewald", nk=3)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("method='ewald' should reject k-points until implemented.")


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
