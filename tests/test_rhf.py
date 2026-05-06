import numpy as np
import pytest
import matplotlib

matplotlib.use("Agg", force=True)

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF, RHFAnalysis
from pyqed.qchem.hf.rhf import get_or_build_low_rank_eri_factors, get_jk
from pyqed.qchem.tools import cubegen

try:
    import pyvista as _pv
except Exception:
    _pv = None

_HAS_PYVISTA_PLOTTING = bool(_pv is not None and _pv.system_supports_plotting())


def test_rhf_verbose_zero_is_clean_and_verbose_one_reports_energy(capsys):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')
    capsys.readouterr()

    mol.RHF().run()
    quiet = capsys.readouterr()
    assert "E(HF)" not in quiet.out
    assert "E_nclr" not in quiet.out

    mol.RHF(verbose=1).run()
    verbose = capsys.readouterr()
    assert "E_nclr" in verbose.out
    assert "E(HF)" in verbose.out


def test_rhf_density_fit_matches_conventional_energy_and_jk():
    scf = pytest.importorskip('pyscf.scf')

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_df = RHF(mol).run(density_fit=True)
    mf_direct = RHF(mol).run()

    pmol = mol.topyscf()
    pmol.build(verbose=0)
    pmf_df = scf.RHF(pmol).density_fit()
    pmf_df.conv_tol = 1e-8
    pmf_df.kernel()

    np.testing.assert_allclose(mf_df.e_tot, pmf_df.e_tot, atol=1e-10)
    np.testing.assert_allclose(mf_df.mo_energy, pmf_df.mo_energy, atol=1e-6)
    np.testing.assert_allclose(mf_df.dm, pmf_df.make_rdm1(), atol=1e-5)
    assert abs(mf_df.e_tot - mf_direct.e_tot) < 1e-3

    vj_df, vk_df = mf_df.get_jk()
    vj_ref, vk_ref = pmf_df.get_jk(dm=mf_df.dm)
    np.testing.assert_allclose(vj_df, vj_ref, atol=1e-10)
    np.testing.assert_allclose(vk_df, vk_ref, atol=1e-10)

    assert mf_df.density_fit
    assert mf_df._pyscf_mf is not None


def test_rhf_localize_orbitals_ibo_occ():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    c_ibo, occ_idx, info = mf.localize_orbitals(
        method='ibo',
        space='occ',
        return_indices=True,
        return_info=True,
    )

    expected_idx = np.flatnonzero(mf.mo_occ > 0.5)
    np.testing.assert_array_equal(occ_idx, expected_idx)
    assert c_ibo.shape == (mol.nao, mf.nocc)
    assert info['method'] == 'ibo'
    assert info['backend'] == 'native'
    assert info['final_objective'] >= info['initial_objective'] - 1e-10

    s = mf.get_ovlp()
    np.testing.assert_allclose(c_ibo.T @ s @ c_ibo, np.eye(mf.nocc), atol=1e-8)


def test_rhf_localize_orbitals_rejects_unsupported_space():
    mol = Molecule(atom='H 0 0 0; H 0 0 0.74', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    with pytest.raises(ValueError, match="space='occ'"):
        mf.localize_orbitals(method='ibo', space='vir')


def test_rhf_localize_orbitals_ibo_custom_coeff_native_gbasis():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    coeff = mf.mo_coeff[:, :mf.nocc]

    c_ibo, info = mf.localize_orbitals(
        method='ibo',
        mo_coeff=coeff,
        return_info=True,
    )

    assert c_ibo.shape == coeff.shape
    assert info['backend'] == 'native'
    assert info['final_objective'] >= info['initial_objective'] - 1e-10

    s = mf.get_ovlp()
    np.testing.assert_allclose(c_ibo.T @ s @ c_ibo, np.eye(mf.nocc), atol=1e-8)


def test_rhf_localize_orbitals_lm_custom_coeff_native_gbasis():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    coeff = mf.mo_coeff[:, :mf.nocc]

    c_lm, info = mf.localize_orbitals(
        method='lm',
        mo_coeff=coeff,
        return_info=True,
    )

    assert c_lm.shape == coeff.shape
    assert info['method'] == 'lm'
    assert info['backend'] == 'native'
    assert info['population_metric'] == 'lowdin'
    assert info['final_objective'] >= info['initial_objective'] - 1e-10

    s = mf.get_ovlp()
    np.testing.assert_allclose(c_lm.T @ s @ c_lm, np.eye(mf.nocc), atol=1e-8)


def test_rhf_localize_orbitals_pm_custom_coeff_native_gbasis():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    coeff = mf.mo_coeff[:, :mf.nocc]

    c_pm, info = mf.localize_orbitals(
        method='pm',
        mo_coeff=coeff,
        return_info=True,
    )

    assert c_pm.shape == coeff.shape
    assert info['method'] == 'pm'
    assert info['backend'] == 'native'
    assert info['population_metric'] == 'mulliken'
    assert info['final_objective'] >= info['initial_objective'] - 1e-10

    s = mf.get_ovlp()
    np.testing.assert_allclose(c_pm.T @ s @ c_pm, np.eye(mf.nocc), atol=1e-8)


def test_rhf_localize_orbitals_pipek_mezey_alias_native_gbasis():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    coeff = mf.mo_coeff[:, :mf.nocc]

    c_pm, info = mf.localize_orbitals(
        method='pipek-mezey',
        mo_coeff=coeff,
        return_info=True,
    )

    assert c_pm.shape == coeff.shape
    assert info['method'] == 'pm'
    assert info['population_metric'] == 'mulliken'


def test_rhf_localize_orbitals_boys_custom_coeff_native_gbasis():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis')

    mf = RHF(mol).run()
    coeff = mf.mo_coeff[:, :mf.nocc]

    c_boys, info = mf.localize_orbitals(
        method='boys',
        mo_coeff=coeff,
        return_info=True,
    )

    assert c_boys.shape == coeff.shape
    assert info['method'] == 'boys'
    assert info['final_objective'] >= info['initial_objective'] - 1e-10

    s = mf.get_ovlp()
    np.testing.assert_allclose(c_boys.T @ s @ c_boys, np.eye(mf.nocc), atol=1e-8)

    r_ao = mol.moment_integral(center=np.zeros(3))
    if r_ao.shape[-1] == 3:
        r_ao = np.moveaxis(r_ao, -1, 0)
    r_before = np.einsum('xij,ip,jq->xpq', r_ao, coeff, coeff, optimize=True)
    r_after = np.einsum('xij,ip,jq->xpq', r_ao, c_boys, c_boys, optimize=True)
    centers_before = np.diagonal(r_before, axis1=1, axis2=2).T
    centers_after = np.diagonal(r_after, axis1=1, axis2=2).T
    obj_before = np.sum(centers_before * centers_before)
    obj_after = np.sum(centers_after * centers_after)
    assert obj_after >= obj_before - 1e-10


def test_cholesky_jk_factors_reproduce_dense_jk_for_tight_tolerance():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf = RHF(mol).run()
    factors = get_or_build_low_rank_eri_factors(mol, tol=1e-12)

    vj_dense, vk_dense = get_jk(mol, mf.dm)
    vj_lr, vk_lr = get_jk(mol, mf.dm, eri_factors=factors)

    np.testing.assert_allclose(vj_lr, vj_dense, atol=1e-8)
    np.testing.assert_allclose(vk_lr, vk_dense, atol=1e-8)


def test_rhf_cholesky_jk_matches_direct_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_direct = RHF(mol).run()
    mf_lr = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    np.testing.assert_allclose(mf_lr.e_tot, mf_direct.e_tot, atol=1e-7)
    np.testing.assert_allclose(mf_lr.dm, mf_direct.dm, atol=1e-6)

    assert mf_lr.cholesky_jk
    assert mf_lr.cholesky_tol == 1e-10
    assert mf_lr.low_rank_jk
    assert mf_lr.eri_factors is not None
    assert mf_lr.eri_factors.shape[0] <= mol.nao * mol.nao


def test_rhf_auto_prefers_prebuilt_dense_plus_factors():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(
        driver='builtin',
        options={'eri_representation': 'dense+factors', 'low_rank_tol': 1e-10},
    )

    mf_auto = RHF(mol).run()
    mf_explicit = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    assert mol.eri is not None
    assert mol.eri_factors is not None
    assert mf_auto.cholesky_jk
    assert mf_auto.low_rank_jk
    np.testing.assert_allclose(mf_auto.e_tot, mf_explicit.e_tot, atol=1e-10)
    np.testing.assert_allclose(mf_auto.dm, mf_explicit.dm, atol=1e-8)


def test_low_rank_aliases_match_cholesky_options():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf_alias = RHF(mol).run(low_rank_jk=True, low_rank_tol=1e-10)

    assert mf_alias.cholesky_jk
    assert mf_alias.low_rank_jk
    assert mf_alias.cholesky_tol == mf_alias.low_rank_tol == 1e-10


def test_low_rank_factor_cache_reuses_exact_geometry_after_rebuild():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    factors_first = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert mol._low_rank_eri_last_info['mode'] == 'cold'

    mol.build(driver='gbasis-pyscf')
    factors_second = get_or_build_low_rank_eri_factors(mol, tol=1e-10)

    assert factors_second is factors_first
    assert mol._low_rank_eri_last_info['mode'] == 'exact'


def test_low_rank_factor_cache_warm_starts_after_geometry_change():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    factors_first = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert mol._low_rank_eri_last_info['mode'] == 'cold'

    coords = mol.atom_coords().copy()
    coords[1, 2] += 0.1 / 0.52917721092
    mol.set_geom(coords)
    mol.build(driver='gbasis-pyscf')

    factors_second = get_or_build_low_rank_eri_factors(mol, tol=1e-10)
    assert factors_second is not factors_first
    assert mol._low_rank_eri_last_info['mode'] == 'warm'

    mf = RHF(mol).run()
    vj_dense, vk_dense = get_jk(mol, mf.dm)
    vj_lr, vk_lr = get_jk(mol, mf.dm, eri_factors=factors_second)
    np.testing.assert_allclose(vj_lr, vj_dense, atol=1e-7)
    np.testing.assert_allclose(vk_lr, vk_dense, atol=1e-7)


def test_low_rank_scanner_reuses_density_and_factor_history():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='gbasis-pyscf')

    mf0 = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)
    scanner = mf0.as_scanner()

    coords = mol.atom_coords().copy()
    coords[1, 2] += 0.05 / 0.52917721092
    e_scan = scanner(coords)

    mol_ref = Molecule(atom='Li 0 0 0; H 0 0 1.65', unit='angstrom', basis='sto-3g')
    mol_ref.build(driver='gbasis-pyscf')
    e_ref = RHF(mol_ref).run(cholesky_jk=True, cholesky_tol=1e-10).e_tot

    np.testing.assert_allclose(e_scan, e_ref, atol=1e-8)
    assert scanner.mf.dm is not None
    assert scanner.mf.eri_factors is not None
    assert mol._low_rank_eri_last_info['mode'] in {'warm', 'exact'}


def test_builtin_ao_labels_are_available_without_pyscf():
    mol = Molecule(atom='O 0 0 0; H 0 0 1.8', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')

    labels = mol.ao_labels()

    assert len(labels) == mol.nao
    assert labels == [
        '0 O 1s',
        '0 O 2s',
        '0 O 2px',
        '0 O 2py',
        '0 O 2pz',
        '1 H 1s',
    ]


def test_rhf_mo_components_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    analysis = mf.mo_components(mo_indices=0, metric='mulliken')

    assert len(analysis) == 1
    mo0 = analysis[0]
    assert mo0['mo_index'] == 0
    np.testing.assert_allclose(mo0['contribution_sum'], 1.0, atol=1e-8)
    assert [entry['label'] for entry in mo0['components']] == ['0 H 1s', '1 H 1s']
    np.testing.assert_allclose(
        [entry['contribution'] for entry in mo0['components']],
        [0.5, 0.5],
        atol=1e-6,
    )


def test_rhf_print_mo_components_builtin_h2(capsys):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()
    capsys.readouterr()

    text = mf.print_mo_components(mo_indices=0, metric='mulliken')
    captured = capsys.readouterr()

    assert captured.out == text + '\n'
    assert 'MO 0:' in text
    assert '0 H 1s' in text
    assert '1 H 1s' in text
    assert 'contribution=+0.5000000000' in text


def test_rhf_analyze_returns_rhfanalysis():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    analysis = mf.analyze()

    assert isinstance(analysis, RHFAnalysis)


def test_rhf_mulliken_charges_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    data = mf.mulliken_charges()

    np.testing.assert_allclose(data['atom_populations'], [1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(data['charges'], [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(data['total_charge'], 0.0, atol=1e-8)


def test_rhf_mulliken_charges_builtin_h2o():
    mol = Molecule(
        atom='O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266',
        unit='bohr',
        basis='sto-3g',
    )
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    data = mf.mulliken_charges()

    np.testing.assert_allclose(np.sum(data['atom_populations']), mol.nelec, atol=1e-8)
    np.testing.assert_allclose(data['total_charge'], float(mol.charge), atol=1e-8)
    assert data['charges'][0] < 0.0
    assert data['charges'][1] > 0.0
    assert data['charges'][2] > 0.0


def test_rhf_print_mulliken_charges_builtin_h2(capsys):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()
    capsys.readouterr()

    text = mf.print_mulliken_charges()
    captured = capsys.readouterr()

    assert captured.out == text + '\n'
    assert 'Mulliken charges:' in text
    assert 'Atom   0' in text
    assert 'Atom   1' in text
    assert 'Total charge = +0.0000000000' in text


def test_rhf_lowdin_charges_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    data = mf.lowdin_charges()

    np.testing.assert_allclose(data['atom_populations'], [1.0, 1.0], atol=1e-8)
    np.testing.assert_allclose(data['charges'], [0.0, 0.0], atol=1e-8)
    np.testing.assert_allclose(data['total_charge'], 0.0, atol=1e-8)


def test_rhf_mayer_bond_orders_builtin_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    data = mf.mayer_bond_orders()

    assert data['bond_orders'].shape == (2, 2)
    np.testing.assert_allclose(data['bond_orders'], data['bond_orders'].T, atol=1e-12)
    assert data['bond_orders'][0, 1] > 0.5


def test_rhf_wiberg_bond_orders_builtin_h2(capsys):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    data = mf.wiberg_bond_orders()

    assert data['bond_orders'].shape == (2, 2)
    np.testing.assert_allclose(data['bond_orders'], data['bond_orders'].T, atol=1e-12)
    assert data['bond_orders'][0, 1] > 0.5

    capsys.readouterr()
    text = mf.print_wiberg_bond_orders(min_bond_order=0.1)
    captured = capsys.readouterr()
    assert captured.out == text + '\n'
    assert 'Wiberg bond orders:' in text
    assert 'Bond ( 0 H)-( 1 H)' in text


def test_rhf_mo_composition_atom_shell_builtin_h2o():
    mol = Molecule(
        atom='O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266',
        unit='bohr',
        basis='sto-3g',
    )
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    analysis = mf.mo_composition(mo_indices=4, metric='mulliken', group_by='atom+shell')

    assert len(analysis) == 1
    mo = analysis[0]
    assert mo['group_by'] == 'atom+shell'
    np.testing.assert_allclose(mo['contribution_sum'], 1.0, atol=1e-8)
    assert mo['components'][0]['label'] == '0 O 2p'
    np.testing.assert_allclose(mo['components'][0]['contribution'], 1.0, atol=1e-8)


def test_rhf_mo_overlap_identity_and_nearby_geometry():
    mol1 = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol1.build(driver='builtin')
    mf1 = RHF(mol1).run()

    s11 = mf1.mo_overlap(mf1)
    np.testing.assert_allclose(s11, np.eye(2), atol=1e-8)

    mol2 = Molecule(atom='H 0 0 0; H 0 0 1.5', unit='bohr', basis='sto-3g')
    mol2.build(driver='builtin')
    mf2 = RHF(mol2).run()

    s12 = mf1.mo_overlap(mf2)
    assert s12.shape == (2, 2)
    assert abs(s12[0, 0]) > 0.9


def test_rhf_sample_mo_grid_and_plot_3d_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    grid = mf.analyze().sample_mo_grid(0, nx=16, ny=15, nz=14, margin=2.0)

    assert grid['mo_index'] == 0
    assert grid['values'].shape == (16, 15, 14)
    assert np.isfinite(grid['values']).all()
    assert np.max(np.abs(grid['values'])) > 0.0

    out = tmp_path / 'h2_mo0.png'
    result = mf.plot_mo_3d(0, nx=16, ny=15, nz=14, margin=2.0, save=out)

    assert result['isovalue'] > 0.0
    assert result['save_path'] == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_rhf_orbital_cube_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_mo0.cube'
    result = mf.orbital_cube(0, out, nx=12, ny=11, nz=10, margin=2.0)

    assert result['cube_path'] == str(out)
    assert result['orbital_index'] == 0
    assert result['coeff_source'] == 'mo'
    assert result['shape'] == (12, 11, 10)
    assert out.exists()
    assert out.stat().st_size > 0

    lines = out.read_text(encoding='utf-8').splitlines()
    assert 'pyqed orbital cube: source=mo index=0' in lines[0]
    assert 'OUTER LOOP: X, MIDDLE LOOP: Y, INNER LOOP: Z' == lines[1]
    header = lines[2].split()
    assert int(header[0]) == 2
    assert int(lines[3].split()[0]) == 12
    assert int(lines[4].split()[0]) == 11
    assert int(lines[5].split()[0]) == 10


def test_rhf_orbital_cube_custom_coeff_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_custom.cube'
    coeff = np.asarray(mf.mo_coeff[:, 0], dtype=float)
    result = mf.orbital_cube(None, out, coeff=coeff, nx=10, ny=10, nz=10, margin=2.0)

    assert result['cube_path'] == str(out)
    assert result['orbital_index'] is None
    assert result['coeff_source'] == 'custom'
    assert out.exists()
    assert out.stat().st_size > 0
    assert 'pyqed orbital cube: source=custom index=custom' in out.read_text(encoding='utf-8').splitlines()[0]


def test_cubegen_orbital_module_function_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_module_orbital.cube'
    result = cubegen.orbital(mf, out, orbital_index=0, nx=10, ny=10, nz=10, margin=2.0)

    assert result['cube_path'] == str(out)
    assert result['shape'] == (10, 10, 10)
    assert out.exists()
    assert out.stat().st_size > 0


def test_cubegen_density_module_function_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_density.cube'
    result = cubegen.density(mf, out, nx=10, ny=9, nz=8, margin=2.0)

    assert result['cube_path'] == str(out)
    assert result['shape'] == (10, 9, 8)
    assert out.exists()
    assert out.stat().st_size > 0
    lines = out.read_text(encoding='utf-8').splitlines()
    assert lines[0] == 'pyqed electron density cube'
    assert int(lines[3].split()[0]) == 10
    assert int(lines[4].split()[0]) == 9
    assert int(lines[5].split()[0]) == 8


def test_rhf_sample_density_grid_and_plot_3d_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    grid = mf.sample_density_grid(nx=14, ny=13, nz=12, margin=2.0)

    assert grid['values'].shape == (14, 13, 12)
    assert np.isfinite(grid['values']).all()
    assert np.max(grid['values']) > 0.0

    out = tmp_path / 'h2_density.png'
    result = mf.plot_density_3d(nx=14, ny=13, nz=12, margin=2.0, style='bold', save=out)

    assert result['isovalue'] > 0.0
    assert len(result['isovalues']) >= 1
    assert result['isovalue'] == result['isovalues'][0]
    assert result['save_path'] == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


def test_rhf_plot_frontier_mos_3d_publication_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_frontier.png'
    result = mf.plot_frontier_mos_3d(
        nx=16,
        ny=16,
        nz=16,
        margin=2.0,
        style='publication',
        save=out,
    )

    assert result['mo_indices'] == (0, 1)
    assert result['save_path'] == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not _HAS_PYVISTA_PLOTTING, reason="pyvista plotting is not supported")
def test_rhf_plot_mo_3d_pyvista_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_mo0_pyvista.png'
    result = mf.plot_mo_3d(
        0,
        nx=14,
        ny=14,
        nz=14,
        margin=2.0,
        backend='pyvista',
        style='publication',
        save=out,
    )

    assert result['backend'] == 'pyvista'
    assert result['save_path'] == str(out)
    assert out.exists()
    assert out.stat().st_size > 0


@pytest.mark.skipif(not _HAS_PYVISTA_PLOTTING, reason="pyvista plotting is not supported")
def test_rhf_plot_density_3d_pyvista_builtin_h2(tmp_path):
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin')
    mf = RHF(mol).run()

    out = tmp_path / 'h2_density_pyvista.png'
    result = mf.plot_density_3d(
        nx=16,
        ny=15,
        nz=14,
        margin=2.0,
        style='bold',
        backend='pyvista',
        save=out,
    )

    assert result['backend'] == 'pyvista'
    assert result['isovalue'] > 0.0
    assert len(result['isovalues']) >= 1
    assert result['smooth_sigma'] > 0.0
    assert result['save_path'] == str(out)
    assert out.exists()
    assert out.stat().st_size > 0
