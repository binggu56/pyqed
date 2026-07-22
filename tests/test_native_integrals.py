import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.basis import (
    PackedRIFactors,
    _basis_cy,
    _integrals_cpp,
    _rys_cy,
    _basis_signature,
    _builtin_worker_count,
    _cart_shell_blocks,
    _compute_cartesian_shell_quartet_block_cython,
    _compute_dense_eri_serial,
    _compute_dense_eri_serial_cpp_cartesian,
    _compute_dense_eri_serial_cpp_ssss,
    _compute_eri_s8_cpp_cartesian,
    _compute_dense_eri_serial_aopairs,
    _compute_dense_eri_serial_cython_blocked,
    _compute_aux_coulomb_metric,
    _compute_native_ri_pair_tensors_cpp,
    _compute_one_electron_shellblocked,
    _compute_one_electron_shellblocked_cython,
    _compute_pair_bounds,
    _compute_three_center_pair_tensor_from_signatures,
    _shell,
    make_contractions,
    parse_gbs,
    _basis_path,
    contract_jk_s4,
    contract_jk_s8,
    contract_jk_ri,
    direct_jk_cartesian_cpp,
    direct_veff_cartesian_cpp,
    pack_eri_s8,
    contract_veff_s8,
    contract_veff_s8_mo,
    unpack_eri_s4,
    unpack_eri_s8,
)

try:
    from pyscf import gto, scf
except ImportError:  # pragma: no cover - optional dependency in some envs
    gto = None
    scf = None


def test_native_build_is_default_and_produces_ao_tensors():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    assert mol._build_driver in {'native', 'builtin'}
    assert mol.nao == 2
    assert mol.overlap.shape == (2, 2)
    assert mol.hcore.shape == (2, 2)
    assert mol.eri is None
    assert mol.eri_s8 is not None
    assert mol.eri_factors is None
    assert mol.builtin_low_rank_tol == 1e-10
    assert mol._builtin_build_info['requested_representation'] == 'auto'
    assert mol._builtin_build_info['representation'] == 'dense'
    assert mol._builtin_build_info['aosym'] == 's8'
    assert mol._builtin_build_info['dense_storage'] == 's8'
    assert mol._builtin_build_info['factor_rank'] is None
    np.testing.assert_allclose(np.diag(mol.overlap), np.ones(2), atol=1e-12)


def test_native_build_runs_rhf_without_external_integral_backends():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='native')

    mf = mol.RHF().run(max_cycle=60)
    assert np.isfinite(mf.e_tot)


def test_builtin_build_accepts_short_eri_keyword_for_factors():
    for eri in ('factors', 'cd'):
        mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
        mol.build(driver='builtin', eri=eri)

        assert mol.builtin_eri_representation == 'factors'
        assert mol.builtin_aosym == 's8'
        assert mol._builtin_build_info['aosym'] == 's8'
        assert mol.eri is None
        assert mol.eri_factors is not None

        mf = mol.RHF().run()
        assert mf.cholesky_jk
        assert mf.eri_factors is not None


def test_native_shell_generator_includes_f_cartesian_components():
    assert _shell(2) == [
        (2, 0, 0),
        (1, 1, 0),
        (1, 0, 1),
        (0, 2, 0),
        (0, 1, 1),
        (0, 0, 2),
    ]
    assert len(_shell(3)) == 10
    assert _shell(3)[0] == (3, 0, 0)
    assert _shell(3)[-1] == (0, 0, 3)


def test_native_build_supports_d_shells_in_cartesian_basis():
    if gto is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    basis = '6-31g(d,p)'

    mol = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol.build(driver='builtin')
    mf = mol.RHF().run(max_cycle=80)

    ref = scf.RHF(gto.M(atom=atom, basis=basis, unit='angstrom', cart=True)).run(conv_tol=1e-12)

    assert mol.nao == ref.mol.nao
    assert np.isfinite(mf.e_tot)
    np.testing.assert_allclose(mf.e_tot, ref.e_tot, atol=1e-8, rtol=1e-8)


def test_builtin_rys_backend_matches_default_dense_builder_for_sp_basis():
    atom = 'O 0 0 0; H 0 0 1.8; H 0 1.7 0'
    basis = 'sto-3g'

    mol_default = Molecule(atom=atom, unit='bohr', basis=basis)
    mol_default.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})

    mol_rys = Molecule(atom=atom, unit='bohr', basis=basis)
    mol_rys.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1', 'eri_backend': 'rys'})

    np.testing.assert_allclose(mol_rys.overlap, mol_default.overlap, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.hcore, mol_default.hcore, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.eri, mol_default.eri, atol=1e-11, rtol=1e-11)
    if _integrals_cpp is not None:
        assert mol_default._builtin_build_info['dense_builder'] == 'cpp-cartesian-lmax6'
    elif _rys_cy is not None:
        assert mol_default._builtin_build_info['dense_builder'] == 'rys-cython-blocked-auto'
    expected_builder = 'rys-cython-blocked' if _rys_cy is not None else 'rys-screened-mixed'
    assert mol_rys._builtin_build_info['dense_builder'] == expected_builder

    e_default = mol_default.RHF().run(max_cycle=80).e_tot
    e_rys = mol_rys.RHF().run(max_cycle=80).e_tot
    np.testing.assert_allclose(e_rys, e_default, atol=1e-10, rtol=1e-10)


def test_builtin_parallel_option_keeps_compiled_rys_for_sp_basis():
    atom = 'O 0 0 0; H 0 0 1.8; H 0 1.7 0'
    basis = 'sto-3g'

    mol_serial = Molecule(atom=atom, unit='bohr', basis=basis)
    mol_serial.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})

    mol_parallel = Molecule(atom=atom, unit='bohr', basis=basis)
    mol_parallel.build(
        driver='builtin',
        options={
            'eri_representation': 'dense',
            'aosym': 's1',
            'parallel': True,
            'eri_workers': 2,
        },
    )

    if _integrals_cpp is not None:
        assert mol_parallel._builtin_build_info['dense_builder'] == 'cpp-cartesian-lmax6'
    elif _rys_cy is not None:
        assert mol_parallel._builtin_build_info['dense_builder'] == 'rys-cython-blocked-auto'
    np.testing.assert_allclose(mol_parallel.eri, mol_serial.eri, atol=1e-11, rtol=1e-11)


def test_builtin_rys_backend_matches_default_dense_builder_for_d_basis():
    atom = 'H 0 0 0; F 0 0 0.9'
    basis = '6-31g(d,p)'

    mol_default = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_default.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})

    mol_rys = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_rys.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1', 'eri_backend': 'rys'})

    np.testing.assert_allclose(mol_rys.overlap, mol_default.overlap, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.hcore, mol_default.hcore, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.eri, mol_default.eri, atol=1e-9, rtol=1e-9)
    expected_builder = (
        'cython-shell-os-blocked-mixed-d-fallback'
        if _basis_cy is not None
        else 'python-serial-mixed-d-fallback'
    )
    assert mol_rys._builtin_build_info['dense_builder'] == expected_builder

    e_default = mol_default.RHF().run(max_cycle=80).e_tot
    e_rys = mol_rys.RHF().run(max_cycle=80).e_tot
    np.testing.assert_allclose(e_rys, e_default, atol=1e-9, rtol=1e-9)


def test_cpp_ssss_dense_helper_matches_existing_dense_builder():
    if _integrals_cpp is None:
        return

    basis = make_contractions(
        parse_gbs(_basis_path('sto-3g')),
        ['H', 'H'],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=float),
        coord_types='c',
    )
    signatures = tuple(_basis_signature(b) for b in basis)
    pair_bounds = _compute_pair_bounds(signatures)

    cpp_eri, cpp_computed, cpp_skipped = _compute_dense_eri_serial_cpp_ssss(
        signatures,
        pair_bounds,
        0.0,
    )
    ref_eri, ref_computed, ref_skipped = _compute_dense_eri_serial_aopairs(
        signatures,
        pair_bounds,
        0.0,
    )

    assert cpp_computed == ref_computed
    assert cpp_skipped == ref_skipped
    np.testing.assert_allclose(cpp_eri, ref_eri, atol=1e-12, rtol=1e-12)


def test_cpp_cartesian_helper_supports_p_d_f_shells():
    if _integrals_cpp is None:
        return

    signatures = (
        ((0, 0, 0), (0.0, 0.0, 0.0), (0.7, 1.4), (0.8, 0.3)),
        ((1, 0, 0), (0.1, -0.2, 0.3), (0.9,), (0.6,)),
        ((0, 1, 0), (-0.2, 0.4, -0.1), (1.2,), (-0.5,)),
        ((0, 0, 1), (0.3, 0.2, -0.4), (0.8,), (0.7,)),
        ((1, 1, 0), (-0.3, 0.2, 0.4), (1.1,), (0.5,)),
        ((0, 1, 1), (0.2, -0.5, 0.1), (1.4,), (-0.25,)),
        ((2, 0, 1), (0.4, 0.1, -0.2), (1.3,), (0.4,)),
    )
    pair_bounds = _compute_pair_bounds(signatures)

    for screen_tol in (0.0, 1e-6):
        cpp_eri, cpp_computed, cpp_skipped = _compute_dense_eri_serial_cpp_cartesian(
            signatures,
            pair_bounds,
            screen_tol,
        )
        ref_eri, ref_computed, ref_skipped = _compute_dense_eri_serial_aopairs(
            signatures,
            pair_bounds,
            screen_tol,
        )

        assert cpp_computed == ref_computed
        assert cpp_skipped == ref_skipped
        np.testing.assert_allclose(cpp_eri, ref_eri, atol=1e-9, rtol=1e-9)


def test_cpp_cartesian_s8_helper_matches_packed_dense_reference():
    if _integrals_cpp is None or not hasattr(_integrals_cpp, "compute_eri_s8_cartesian"):
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    basis = make_contractions(
        parse_gbs(_basis_path('6-31g(d,p)')),
        ['H', 'F'],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.9]], dtype=float),
        coord_types='c',
    )
    signatures = tuple(_basis_signature(b) for b in basis)
    pair_bounds = _compute_pair_bounds(signatures)

    eri_s8, computed_s8, skipped_s8 = _compute_eri_s8_cpp_cartesian(
        signatures,
        pair_bounds,
        0.0,
    )
    ref_eri, computed_ref, skipped_ref = _compute_dense_eri_serial_aopairs(
        signatures,
        pair_bounds,
        0.0,
    )

    assert computed_s8 == computed_ref
    assert skipped_s8 == skipped_ref
    np.testing.assert_allclose(eri_s8, pack_eri_s8(ref_eri), atol=1e-9, rtol=1e-9)


def test_cpp_cartesian_helper_supports_high_l_scalar_fallback():
    if _integrals_cpp is None:
        return

    signatures = (
        ((0, 0, 0), (0.0, 0.0, 0.0), (0.8,), (1.0,)),
        ((4, 0, 0), (0.2, -0.1, 0.3), (1.1,), (0.4,)),
        ((0, 0, 5), (-0.3, 0.4, -0.2), (1.3,), (-0.2,)),
    )
    pair_bounds = _compute_pair_bounds(signatures)

    cpp_eri, cpp_computed, cpp_skipped = _compute_dense_eri_serial_cpp_cartesian(
        signatures,
        pair_bounds,
        0.0,
    )
    ref_eri, ref_computed, ref_skipped = _compute_dense_eri_serial_aopairs(
        signatures,
        pair_bounds,
        0.0,
    )

    assert cpp_computed == ref_computed
    assert cpp_skipped == ref_skipped
    np.testing.assert_allclose(cpp_eri, ref_eri, atol=1e-8, rtol=1e-8)


def test_dense_dispatch_falls_back_beyond_cpp_high_l_limit():
    signatures = (
        ((0, 0, 0), (0.0, 0.0, 0.0), (0.8,), (1.0,)),
        ((7, 0, 0), (0.2, -0.1, 0.3), (1.1,), (0.4,)),
    )
    pair_bounds = _compute_pair_bounds(signatures)

    eri, computed, skipped = _compute_dense_eri_serial(signatures, pair_bounds, 0.0)
    ref_eri, ref_computed, ref_skipped = _compute_dense_eri_serial_aopairs(
        signatures,
        pair_bounds,
        0.0,
    )

    assert computed == ref_computed
    assert skipped == ref_skipped
    np.testing.assert_allclose(eri, ref_eri, atol=1e-8, rtol=1e-8)


def test_builtin_cpp_backend_builds_s_shell_dense_eri():
    if _integrals_cpp is None:
        return

    mol_cpp = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol_cpp.build(
        driver='builtin',
        eri='dense',
        aosym='s1',
        options={'eri_backend': 'cpp'},
    )

    mol_ref = Molecule(atom=mol_cpp.atom, unit='bohr', basis='sto-3g')
    mol_ref.build(driver='builtin', eri='dense', aosym='s1')

    assert mol_cpp._builtin_build_info['eri_backend'] == 'cpp'
    assert mol_cpp._builtin_build_info['dense_builder'] == 'cpp-cartesian-lmax6'
    np.testing.assert_allclose(mol_cpp.overlap, mol_ref.overlap, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_cpp.hcore, mol_ref.hcore, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_cpp.eri, mol_ref.eri, atol=1e-12, rtol=1e-12)


def test_builtin_cpp_backend_builds_factor_only_cholesky():
    if _integrals_cpp is None:
        return

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(
        driver='builtin',
        eri='factors',
        options={'eri_backend': 'cpp', 'low_rank_tol': 1e-12},
    )

    assert mol.eri is None
    assert mol.eri_factors is not None
    assert isinstance(mol.eri_factors, PackedRIFactors)
    assert mol.eri_factors.pair_shape == (3, 3)
    assert mol._builtin_build_info['dense_builder'] == 'cpp-cartesian-s8-lmax6-factor-source'
    assert mol._builtin_build_info['factor_builder'] == 'cpp-s8-pair-pivoted-cholesky'
    assert mol._builtin_build_info['factor_storage'] == 'packed-pair'
    assert mol._builtin_build_info['factor_pair_shape'] == (3, 3)

    mol_full = Molecule(atom=mol.atom, unit='bohr', basis='sto-3g')
    mol_full.build(
        driver='builtin',
        eri='factors',
        aosym='s1',
        options={'eri_backend': 'cpp', 'low_rank_tol': 1e-12},
    )
    rng = np.random.default_rng(135)
    dm = rng.normal(size=(mol.nao, mol.nao))
    dm = dm + dm.T
    vj, vk = contract_jk_ri(mol.eri_factors, dm, mol.nao)
    vj_full, vk_full = contract_jk_ri(mol_full.eri_factors, dm, mol_full.nao)
    np.testing.assert_allclose(vj, vj_full, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(vk, vk_full, atol=1e-12, rtol=1e-12)


def test_cpp_ri_tensor_matches_python_reference():
    if _integrals_cpp is None:
        return

    basis = make_contractions(
        parse_gbs(_basis_path('sto-3g')),
        ['H', 'H'],
        np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]], dtype=float),
        coord_types='c',
    )
    signatures = tuple(_basis_signature(b) for b in basis)
    aux_signatures = signatures
    pair_bounds = _compute_pair_bounds(signatures)

    for screen_tol in (0.0, 0.5):
        cpp = _compute_native_ri_pair_tensors_cpp(
            signatures,
            aux_signatures,
            pair_bounds,
            screen_tol,
        )
        assert cpp is not None
        metric_cpp, j3_cpp, computed_cpp, skipped_cpp = cpp
        metric_ref = _compute_aux_coulomb_metric(aux_signatures)
        j3_ref, computed_ref, skipped_ref = _compute_three_center_pair_tensor_from_signatures(
            signatures,
            aux_signatures,
            pair_bounds=pair_bounds,
            ri_screen_tol=screen_tol,
        )

        assert computed_cpp == computed_ref
        assert skipped_cpp == skipped_ref
        np.testing.assert_allclose(metric_cpp, metric_ref, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(j3_cpp, j3_ref, atol=1e-11, rtol=1e-11)


def test_builtin_ri_accepts_cpp_tensor_backend_request():
    if _integrals_cpp is None:
        return

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(
        driver='builtin',
        eri='ri',
        options={
            'eri_backend': 'cpp',
            'ri_tensor_backend': 'cpp',
            'ri_cache': False,
        },
    )

    ri_info = mol._builtin_build_info['ri']
    assert mol.eri is None
    assert mol.eri_factors is not None
    assert mol._builtin_build_info['factor_builder'] == 'native-ri'
    assert ri_info['tensor_backend'] == 'cpp'
    assert ri_info['effective_tensor_backend'] == 'cpp'
    assert ri_info['tensor_builder'] == 'cpp-kernel-packed'
    assert ri_info['tensor_kernel'] == 'cpp-shell-block-vrr-hrr'


def test_builtin_basis_aliases_resolve_existing_gbs_files():
    assert _basis_path("cc-pvtz").endswith("cc-pvtz.0.gbs")
    assert _basis_path("aug-cc-pvdz").endswith("aug-cc-pvdz.0.gbs")


def test_builtin_s4_storage_matches_dense_rhf_energy():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol_dense = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense', aosym='s1')
    e_dense = mol_dense.RHF().run(max_cycle=80).e_tot

    mol_s4 = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_s4.build(driver='builtin', eri='s4')
    e_s4 = mol_s4.RHF().run(max_cycle=80).e_tot

    assert mol_s4.eri is None
    assert mol_s4.eri_s4 is not None
    np.testing.assert_allclose(unpack_eri_s4(mol_s4.eri_s4, mol_s4.nao), mol_dense.eri, atol=1e-12)
    np.testing.assert_allclose(e_s4, e_dense, atol=1e-10, rtol=1e-10)


def test_builtin_s8_storage_matches_dense_rhf_energy():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol_dense = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense', aosym='s1')
    e_dense = mol_dense.RHF().run(max_cycle=80).e_tot

    mol_s8 = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_s8.build(driver='builtin', eri='s8')
    e_s8 = mol_s8.RHF().run(max_cycle=80).e_tot

    npair = mol_s8.nao * (mol_s8.nao + 1) // 2
    assert mol_s8.eri is None
    assert mol_s8.eri_s8 is not None
    assert mol_s8.eri_s8.shape == (npair * (npair + 1) // 2,)
    np.testing.assert_allclose(unpack_eri_s8(mol_s8.eri_s8, mol_s8.nao), mol_dense.eri, atol=1e-12)
    np.testing.assert_allclose(e_s8, e_dense, atol=1e-10, rtol=1e-10)


def test_builtin_cpp_backend_builds_direct_s8_storage():
    if _integrals_cpp is None or not hasattr(_integrals_cpp, "compute_eri_s8_cartesian"):
        return

    mol = Molecule(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 0', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='dense', aosym='s8', options={'eri_backend': 'cpp'})

    mol_ref = Molecule(atom=mol.atom, unit='bohr', basis='sto-3g')
    mol_ref.build(driver='builtin', eri='dense', aosym='s1')

    assert mol.eri is None
    assert mol.eri_s8 is not None
    assert mol._builtin_build_info['dense_builder'] == 'cpp-cartesian-s8-lmax6'
    np.testing.assert_allclose(unpack_eri_s8(mol.eri_s8, mol.nao), mol_ref.eri, atol=1e-11, rtol=1e-11)


def test_builtin_dense_defaults_to_cpp_s8_storage():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='dense')

    info = mol._builtin_build_info
    assert info['driver'] == 'builtin'
    assert info['basis'] == 'sto-3g'
    assert info['input_unit'] == 'bohr'
    assert info['coordinate_unit'] == 'bohr'
    assert info['atom_symbols'] == ['H', 'H']
    assert info['geometry_hash'] == mol.geometry_hash()
    assert mol.builtin_aosym == 's8'
    assert info['aosym'] == 's8'
    assert mol.eri is None
    assert mol.eri_s8 is not None
    if _integrals_cpp is not None and hasattr(_integrals_cpp, "compute_eri_s8_cartesian"):
        assert info['dense_builder'] == 'cpp-cartesian-s8-lmax6'


def test_builtin_dense_pyrazine_631g_matches_pyscf_rhf():
    if gto is None:
        return

    atom = '''
N   0.000000   1.397000   0.000000
C   1.209000   0.698500   0.000000
C   1.209000  -0.698500   0.000000
N   0.000000  -1.397000   0.000000
C  -1.209000  -0.698500   0.000000
C  -1.209000   0.698500   0.000000
H   2.147000   1.240000   0.000000
H   2.147000  -1.240000   0.000000
H  -2.147000  -1.240000   0.000000
H  -2.147000   1.240000   0.000000
'''

    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g')
    mol.build(driver='builtin', eri='dense')
    mf = mol.RHF().run(tol=1e-9, max_cycle=100)

    ref_mol = gto.M(atom=atom, basis='6-31g', unit='angstrom', cart=mol.cart, verbose=0)
    ref = scf.RHF(ref_mol)
    ref.conv_tol = 1e-9
    ref.max_cycle = 100
    ref.kernel()

    assert ref.converged
    assert mf.converged
    assert mol._builtin_build_info['basis'] == '6-31g'
    assert mol._builtin_build_info['input_unit'] == 'angstrom'
    assert mol._builtin_build_info['geometry_hash'] == mol.geometry_hash()
    np.testing.assert_allclose(mf.e_tot, ref.e_tot, atol=2e-8, rtol=0.0)


def test_builtin_cpp_s8_parallel_matches_serial_storage():
    if _integrals_cpp is None or not hasattr(_integrals_cpp, "compute_eri_s8_cartesian"):
        return

    atom = 'O 0 0 0; H 0 0 1.8; H 0 1.7 0'
    serial = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    serial.build(driver='builtin', eri='dense')

    parallel = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    parallel.build(
        driver='builtin',
        eri='dense',
        options={
            'parallel': True,
            'eri_workers': 2,
            'parallel_min_nao': 0,
        },
    )

    assert parallel._builtin_build_info['workers'] == 2
    assert parallel._builtin_build_info['dense_builder'] == 'cpp-cartesian-s8-lmax6'
    assert parallel._builtin_build_info['quartets_computed'] == serial._builtin_build_info['quartets_computed']
    assert parallel._builtin_build_info['quartets_screened'] == serial._builtin_build_info['quartets_screened']
    np.testing.assert_allclose(parallel.eri_s8, serial.eri_s8, atol=1e-13, rtol=1e-13)


def test_builtin_parallel_min_nao_default_is_twelve():
    mol = Molecule(
        atom='H 0 0 0; H 0 0 1.4',
        unit='bohr',
        basis='sto-3g',
    )

    assert mol.builtin_parallel_min_nao == 12
    mol.builtin_parallel = True
    mol.native_parallel = True
    mol.builtin_eri_workers = 3
    mol.native_eri_workers = 3
    assert _builtin_worker_count(mol, 11) == 1
    assert _builtin_worker_count(mol, 12) == 3


def test_builtin_aosym_s8_matches_legacy_eri_alias():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol_aosym = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_aosym.build(driver='builtin', eri='dense', aosym='s8')

    mol_legacy = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_legacy.build(driver='builtin', eri='s8')

    assert mol_aosym.builtin_eri_representation == 'dense'
    assert mol_aosym.builtin_aosym == 's8'
    assert mol_aosym._builtin_build_info['representation'] == 'dense'
    assert mol_aosym._builtin_build_info['aosym'] == 's8'
    expected_builder = (
        'cpp-cartesian-s8-lmax6'
        if _integrals_cpp is not None and hasattr(_integrals_cpp, "compute_eri_s8_cartesian")
        else ('cpp-cartesian-lmax6' if _integrals_cpp is not None else 'cython-s8-packed')
    )
    assert mol_aosym._builtin_build_info['dense_builder'] == expected_builder
    np.testing.assert_allclose(mol_aosym.eri_s8, mol_legacy.eri_s8, atol=1e-12)


def test_builtin_packed_jk_contractions_match_dense_tensor():
    mol_dense = Molecule(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 0', unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense', aosym='s1')

    mol_s4 = Molecule(atom=mol_dense.atom, unit='bohr', basis='sto-3g')
    mol_s4.build(driver='builtin', eri='s4')
    mol_s8 = Molecule(atom=mol_dense.atom, unit='bohr', basis='sto-3g')
    mol_s8.build(driver='builtin', eri='s8')

    rng = np.random.default_rng(123)
    dm = rng.normal(size=(mol_dense.nao, mol_dense.nao))
    dm = dm + dm.T
    vj_dense = np.einsum('lk,ijkl->ij', dm, mol_dense.eri, optimize=True)
    vk_dense = np.einsum('lk,ilkj->ij', dm, mol_dense.eri, optimize=True)

    vj_s4, vk_s4 = contract_jk_s4(mol_s4.eri_s4, dm, mol_s4.nao)
    vj_s8, vk_s8 = contract_jk_s8(mol_s8.eri_s8, dm, mol_s8.nao)

    np.testing.assert_allclose(vj_s4, vj_dense, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_s4, vk_dense, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vj_s8, vj_dense, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_s8, vk_dense, atol=1e-11, rtol=1e-11)
    if _integrals_cpp is not None and hasattr(_integrals_cpp, "contract_jk_s8"):
        vj_cpp, vk_cpp = _integrals_cpp.contract_jk_s8(
            np.ascontiguousarray(mol_s8.eri_s8, dtype=np.float64),
            np.ascontiguousarray(dm, dtype=np.float64),
            mol_s8.nao,
            2,
        )
        np.testing.assert_allclose(vj_cpp, vj_dense, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(vk_cpp, vk_dense, atol=1e-11, rtol=1e-11)
        vj_wrap, vk_wrap = contract_jk_s8(mol_s8.eri_s8, dm, mol_s8.nao, workers=2)
        np.testing.assert_allclose(vj_wrap, vj_dense, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(vk_wrap, vk_dense, atol=1e-11, rtol=1e-11)


def test_cpp_packed_ri_jk_and_ao2mo_match_numpy_references():
    if (
        _integrals_cpp is None
        or not hasattr(_integrals_cpp, "contract_jk_ri_occ_packed")
        or not (
            hasattr(_integrals_cpp, "mo_pair_factors")
            or hasattr(_integrals_cpp, "transform_ri_factors_to_mo_pair")
        )
    ):
        return

    nao = 5
    naux = 7
    nmo = 4
    npair = nao * (nao + 1) // 2
    rng = np.random.default_rng(456)
    pair_factors = rng.normal(size=(naux, npair))
    mo_coeff = rng.normal(size=(nao, nmo))
    mo_occ = np.asarray([2.0, 1.0, 0.5, 0.0])

    dm = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
    vj_ref, vk_ref = contract_jk_ri(pair_factors, dm, nao)
    vj_cpp, vk_cpp = _integrals_cpp.contract_jk_ri_occ_packed(
        np.ascontiguousarray(pair_factors, dtype=np.float64),
        np.ascontiguousarray(mo_coeff, dtype=np.float64),
        np.ascontiguousarray(mo_occ, dtype=np.float64),
        nao,
    )
    vj_cpp_parallel, vk_cpp_parallel = _integrals_cpp.contract_jk_ri_occ_packed(
        np.ascontiguousarray(pair_factors, dtype=np.float64),
        np.ascontiguousarray(mo_coeff, dtype=np.float64),
        np.ascontiguousarray(mo_occ, dtype=np.float64),
        nao,
        2,
    )
    np.testing.assert_allclose(vj_cpp, vj_ref, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_cpp, vk_ref, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vj_cpp_parallel, vj_ref, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_cpp_parallel, vk_ref, atol=1e-11, rtol=1e-11)

    mo_right = rng.normal(size=(nao, 3))
    transform_cpp = getattr(_integrals_cpp, "mo_pair_factors", None)
    if transform_cpp is None:
        transform_cpp = _integrals_cpp.transform_ri_factors_to_mo_pair
    transformed = transform_cpp(
        np.ascontiguousarray(pair_factors, dtype=np.float64),
        np.ascontiguousarray(mo_coeff, dtype=np.float64),
        np.ascontiguousarray(mo_right, dtype=np.float64),
    )
    rows, cols = np.tril_indices(nao)
    full = np.zeros((naux, nao, nao))
    full[:, rows, cols] = pair_factors
    full[:, cols, rows] = pair_factors
    ref = np.einsum("Pij,ip,jq->Ppq", full, mo_coeff, mo_right, optimize=True)
    np.testing.assert_allclose(transformed, ref, atol=1e-11, rtol=1e-11)


def test_cpp_s8_ao2mo_matches_dense_einsum_reference():
    if _integrals_cpp is None or not hasattr(_integrals_cpp, "ao2mo_s8"):
        return

    mol = Molecule(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 0', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')

    rng = np.random.default_rng(789)
    coeff = rng.normal(size=(mol.nao, mol.nao))
    eri_mo = _integrals_cpp.ao2mo_s8(
        np.ascontiguousarray(mol.eri_s8, dtype=np.float64),
        np.ascontiguousarray(coeff, dtype=np.float64),
    )
    eri_dense = unpack_eri_s8(mol.eri_s8, mol.nao)
    ref = np.einsum("ijkl,ip,jq,kr,ls->pqrs", eri_dense, coeff, coeff, coeff, coeff, optimize=True)

    np.testing.assert_allclose(eri_mo, ref, atol=1e-11, rtol=1e-11)

    mf = mol.RHF()
    mf.mo_coeff = coeff
    np.testing.assert_allclose(mf.get_eri_mo(), ref, atol=1e-11, rtol=1e-11)


def test_cpp_s8_mo_veff_matches_density_jk_reference():
    if _integrals_cpp is None or not hasattr(_integrals_cpp, "contract_veff_s8_occ"):
        return

    mol = Molecule(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 0', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='s8')

    rng = np.random.default_rng(2468)
    coeff = rng.normal(size=(mol.nao, mol.nao))
    occ = np.zeros(mol.nao)
    occ[:3] = [2.0, 1.0, 0.5]

    veff_cpp = contract_veff_s8_mo(mol.eri_s8, coeff, occ, mol.nao, workers=2)
    dm = coeff @ np.diag(occ) @ coeff.T
    veff_dm_cpp = contract_veff_s8(mol.eri_s8, dm, mol.nao, workers=2)
    vj, vk = contract_jk_s8(mol.eri_s8, dm, mol.nao)
    ref = vj - 0.5 * vk

    assert veff_cpp is not None
    assert veff_dm_cpp is not None
    np.testing.assert_allclose(veff_cpp, ref, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(veff_dm_cpp, ref, atol=1e-11, rtol=1e-11)


def test_builtin_direct_jk_matches_dense_tensor_and_rhf_energy():
    atom = 'O 0 0 0; H 0 0 1.8; H 0 1.7 0'
    mol_dense = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense', aosym='s1')

    mol_direct = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_direct.build(driver='builtin', eri='direct')

    rng = np.random.default_rng(321)
    dm = rng.normal(size=(mol_dense.nao, mol_dense.nao))
    dm = dm + dm.T
    dm_ref = rng.normal(size=(mol_dense.nao, mol_dense.nao))
    dm_ref = dm_ref + dm_ref.T
    vj_dense = np.einsum('lk,ijkl->ij', dm, mol_dense.eri, optimize=True)
    vk_dense = np.einsum('lk,ilkj->ij', dm, mol_dense.eri, optimize=True)
    from pyqed.qchem.hf.rhf import get_jk, get_veff
    vj_direct, vk_direct = get_jk(mol_direct, dm)

    np.testing.assert_allclose(vj_direct, vj_dense, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_direct, vk_dense, atol=1e-11, rtol=1e-11)
    dm_nonsym = rng.normal(size=(mol_dense.nao, mol_dense.nao))
    vj_dense_nonsym = np.einsum('lk,ijkl->ij', dm_nonsym, mol_dense.eri, optimize=True)
    vk_dense_nonsym = np.einsum('lk,ilkj->ij', dm_nonsym, mol_dense.eri, optimize=True)
    vj_direct_nonsym, vk_direct_nonsym = get_jk(mol_direct, dm_nonsym)
    np.testing.assert_allclose(vj_direct_nonsym, vj_dense_nonsym, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_direct_nonsym, vk_dense_nonsym, atol=1e-11, rtol=1e-11)
    assert mol_direct._builtin_direct_jk_data is not None
    assert mol_direct.eri_s8 is None
    assert mol_direct._builtin_build_info['dense_builder'] in {
        'cpp-cartesian-direct-jk',
        'cython-direct-jk',
    }
    if _integrals_cpp is not None and hasattr(_integrals_cpp, "direct_jk_cartesian"):
        assert mol_direct._builtin_build_info['dense_builder'] == 'cpp-cartesian-direct-jk'
        assert mol_direct._builtin_build_info['direct_jk']['screening'] == 'schwarz+density'
        assert mol_direct._builtin_build_info['direct_jk']['task_cache'] == 'cpp-shell-pair-geometry+quartet-tasks'
        data = mol_direct._builtin_direct_jk_data
        direct_cpp = direct_jk_cartesian_cpp(
            data["shells"],
            data["origins"],
            data["exps"],
            data["weights"],
            data["nprim"],
            data["pair_bounds"],
            dm,
            data["screen_tol"],
            workers=2,
        )
        assert direct_cpp is not None
        vj_cpp, vk_cpp, computed, _skipped = direct_cpp
        assert computed > 0
        np.testing.assert_allclose(vj_cpp, vj_dense, atol=1e-11, rtol=1e-11)
        np.testing.assert_allclose(vk_cpp, vk_dense, atol=1e-11, rtol=1e-11)
        if hasattr(_integrals_cpp, "direct_veff_cartesian"):
            direct_veff_cpp = direct_veff_cartesian_cpp(
                data["shells"],
                data["origins"],
                data["exps"],
                data["weights"],
                data["nprim"],
                data["pair_bounds"],
                dm,
                data["screen_tol"],
                workers=2,
            )
            assert direct_veff_cpp is not None
            veff_cpp, veff_computed, _veff_skipped = direct_veff_cpp
            assert veff_computed == computed
            np.testing.assert_allclose(veff_cpp, vj_dense - 0.5 * vk_dense, atol=1e-11, rtol=1e-11)
    vhf_ref = get_veff(mol_direct, dm_ref)
    vhf_full = get_veff(mol_direct, dm)
    vhf_incr = get_veff(mol_direct, dm, dm_last=dm_ref, vhf_last=vhf_ref)
    np.testing.assert_allclose(vhf_incr, vhf_full, atol=1e-10, rtol=1e-10)
    assert mol_direct._builtin_direct_jk_data["last_mode"].startswith("veff-")
    assert mol_direct._builtin_direct_jk_data["last_skipped"] is not None
    e_dense = mol_dense.RHF().run(max_cycle=80).e_tot
    e_direct = mol_direct.RHF().run(max_cycle=80).e_tot
    np.testing.assert_allclose(e_direct, e_dense, atol=1e-10, rtol=1e-10)


def test_builtin_auto_prefers_ri_for_larger_native_builds():
    mol = Molecule(
        atom='O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266',
        unit='bohr',
        basis='def2-svp',
    )
    mol.build(driver='builtin', eri='auto')

    assert mol._builtin_build_info['representation'] == 'ri'
    assert mol.eri is None
    assert mol.eri_factors is not None


def test_rhf_exposes_factorized_ao2mo_for_ri_builds():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='ri')

    mf = mol.RHF().run(max_cycle=60)
    pair_factors = mf.mo_factors(mf.mo_coeff)
    pair_factors_alias = mf.get_eri_mo_factors(mf.mo_coeff)
    eri_from_factors = np.einsum('Ppq,Prs->pqrs', pair_factors, pair_factors, optimize=True)

    assert pair_factors.ndim == 3
    assert pair_factors.shape[1:] == (mol.nao, mol.nao)
    np.testing.assert_allclose(pair_factors_alias, pair_factors, atol=0.0, rtol=0.0)
    np.testing.assert_allclose(eri_from_factors, mf.get_eri_mo(), atol=1e-12, rtol=1e-12)


def test_builtin_auto_uses_packed_s8_for_small_exact_builds():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='auto')

    assert mol._builtin_build_info['representation'] == 'dense'
    assert mol._builtin_build_info['aosym'] == 's8'
    assert mol.eri is None
    assert mol.eri_s8 is not None
    assert mol.builtin_resolved_eri_representation == 'dense'
    assert mol.builtin_resolved_aosym == 's8'


def test_shell_blocked_dense_eri_matches_legacy_aopair_builder():
    atom = 'H 0 0 0; H 0 0 1.4'
    mol = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    basis_dict = parse_gbs(_basis_path(mol.basis))
    basis = make_contractions(basis_dict, mol.atom_symbols(), np.asarray(mol.atom_coords(), dtype=float), coord_types='p')
    signatures = tuple(_basis_signature(fn) for fn in basis)
    pair_bounds = _compute_pair_bounds(signatures)

    eri_shell, computed_shell, skipped_shell = _compute_dense_eri_serial(signatures, pair_bounds, 0.0)
    eri_pair, computed_pair, skipped_pair = _compute_dense_eri_serial_aopairs(signatures, pair_bounds, 0.0)

    np.testing.assert_allclose(eri_shell, eri_pair, atol=1e-12, rtol=1e-12)
    assert computed_shell == computed_pair
    assert skipped_shell == skipped_pair == 0


def test_cython_dense_eri_matches_legacy_aopair_builder_for_d_shell_case():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g(d,p)')
    mol.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})
    signatures = tuple(_basis_signature(fn) for fn in mol._bas)
    pair_bounds = _compute_pair_bounds(signatures)

    eri_compiled, computed_compiled, skipped_compiled = _compute_dense_eri_serial(signatures, pair_bounds, 0.0)
    eri_pair, computed_pair, skipped_pair = _compute_dense_eri_serial_aopairs(signatures, pair_bounds, 0.0)

    np.testing.assert_allclose(eri_compiled, eri_pair, atol=1e-9, rtol=1e-9)
    assert computed_compiled >= computed_pair
    assert skipped_compiled == skipped_pair == 0


def test_cython_blocked_dense_eri_matches_legacy_aopair_builder_for_d_shell_case():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g(d,p)')
    mol.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})
    signatures = tuple(_basis_signature(fn) for fn in mol._bas)
    pair_bounds = _compute_pair_bounds(signatures)

    blocked = _compute_dense_eri_serial_cython_blocked(signatures, pair_bounds, 0.0)
    assert blocked is not None
    eri_blocked, computed_blocked, skipped_blocked = blocked
    eri_pair, computed_pair, skipped_pair = _compute_dense_eri_serial_aopairs(signatures, pair_bounds, 0.0)

    np.testing.assert_allclose(eri_blocked, eri_pair, atol=1e-9, rtol=1e-9)
    assert computed_blocked >= computed_pair
    assert skipped_blocked == skipped_pair == 0


def test_cython_cartesian_shell_quartet_block_matches_dense_slice():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g(d,p)')
    mol.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})
    signatures = tuple(_basis_signature(fn) for fn in mol._bas)
    shell_blocks = _cart_shell_blocks(mol._bas)

    # Choose a nontrivial quartet that includes the d-shell block on F.
    a0, a1, _ = shell_blocks[1]
    b0, b1, _ = shell_blocks[2]
    c0, c1, _ = shell_blocks[-2]
    d0, d1, _ = shell_blocks[-1]
    block = _compute_cartesian_shell_quartet_block_cython(
        signatures, (a0, a1, b0, b1, c0, c1, d0, d1)
    )
    if block is None:
        return

    pair_bounds = _compute_pair_bounds(signatures)
    eri_dense, _, _ = _compute_dense_eri_serial(signatures, pair_bounds, 0.0)
    ref = eri_dense[a0:a1, b0:b1, c0:c1, d0:d1]
    np.testing.assert_allclose(block, ref, atol=1e-9, rtol=1e-9)


def test_cython_iterative_os_shell_quartet_matches_default_block():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g(d,p)')
    mol.build(driver='builtin', options={'eri_representation': 'dense', 'aosym': 's1'})
    signatures = tuple(_basis_signature(fn) for fn in mol._bas)
    shell_blocks = _cart_shell_blocks(mol._bas)

    a0, a1, _ = shell_blocks[1]
    b0, b1, _ = shell_blocks[2]
    c0, c1, _ = shell_blocks[-2]
    d0, d1, _ = shell_blocks[-1]
    shell_block = (a0, a1, b0, b1, c0, c1, d0, d1)
    block_default = _compute_cartesian_shell_quartet_block_cython(signatures, shell_block)
    block_iterative = _compute_cartesian_shell_quartet_block_cython(
        signatures, shell_block, use_iterative=True
    )

    if block_default is None or block_iterative is None:
        return
    np.testing.assert_allclose(block_iterative, block_default, atol=1e-12, rtol=1e-12)


def test_cython_one_electron_matches_python_builder():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    mol = Molecule(atom=atom, unit='angstrom', basis='6-31g(d,p)')
    basis_dict = parse_gbs(_basis_path(mol.basis))
    basis = make_contractions(
        basis_dict,
        mol.atom_symbols(),
        np.asarray(mol.atom_coords(), dtype=float),
        coord_types='c',
    )
    signatures = tuple(_basis_signature(fn) for fn in basis)
    py_overlap, py_kinetic, py_vnuc = _compute_one_electron_shellblocked(
        basis,
        np.asarray(mol.atom_coords(), dtype=float),
        np.asarray(mol.atom_charges(), dtype=float),
    )
    cy_result = _compute_one_electron_shellblocked_cython(
        signatures,
        np.asarray(mol.atom_coords(), dtype=float),
        np.asarray(mol.atom_charges(), dtype=float),
    )
    assert cy_result is not None
    cy_overlap, cy_kinetic, cy_vnuc = cy_result
    np.testing.assert_allclose(cy_overlap, py_overlap, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(cy_kinetic, py_kinetic, atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(cy_vnuc, py_vnuc, atol=1e-9, rtol=1e-9)


def test_factor_only_builtin_d_shell_matches_dense_plus_factors():
    if _basis_cy is None:
        return

    atom = 'H 0 0 0; F 0 0 0.9'
    basis = '6-31g(d,p)'

    mol_dense = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_dense.build(
        driver='builtin',
        options={'eri_representation': 'dense+factors', 'low_rank_tol': 1e-10},
    )
    mf_dense = mol_dense.RHF().run(cholesky_jk=True, cholesky_tol=1e-10, max_cycle=80)

    mol_fact = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_fact.build(
        driver='builtin',
        options={'eri_representation': 'factors', 'low_rank_tol': 1e-10},
    )
    mf_fact = mol_fact.RHF().run(cholesky_jk=True, cholesky_tol=1e-10, max_cycle=80)

    assert mol_fact.eri is None
    assert mol_fact.eri_factors is not None
    assert mol_fact._builtin_build_info['factor_builder'] in {
        'cpp-dense-pivoted-cholesky',
        'cpp-s8-pair-pivoted-cholesky',
        'cython-kernel',
        'cython-kernel-blocked',
        'cython-kernel-packed',
        'cython-kernel-blocked-packed',
        'python-oracle-packed',
    }
    np.testing.assert_allclose(mf_fact.e_tot, mf_dense.e_tot, atol=1e-8, rtol=1e-8)
