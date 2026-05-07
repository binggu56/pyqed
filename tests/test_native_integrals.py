import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.basis import (
    _basis_cy,
    _rys_cy,
    _basis_signature,
    _cart_shell_blocks,
    _compute_cartesian_shell_quartet_block_cython,
    _compute_dense_eri_serial,
    _compute_dense_eri_serial_aopairs,
    _compute_dense_eri_serial_cython_blocked,
    _compute_one_electron_shellblocked,
    _compute_one_electron_shellblocked_cython,
    _compute_pair_bounds,
    _shell,
    make_contractions,
    parse_gbs,
    _basis_path,
    contract_jk_s4,
    contract_jk_s8,
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
    assert mol._builtin_build_info['requested_representation'] == 'auto'
    assert mol._builtin_build_info['representation'] == 'dense'
    assert mol._builtin_build_info['aosym'] == 's8'
    np.testing.assert_allclose(np.diag(mol.overlap), np.ones(2), atol=1e-12)


def test_native_build_runs_rhf_without_external_integral_backends():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(driver='native')

    mf = mol.RHF().run(max_cycle=60)
    assert np.isfinite(mf.e_tot)


def test_builtin_build_accepts_short_eri_keyword_for_factors():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(driver='builtin', eri='factors')

    assert mol.builtin_eri_representation == 'factors'
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
    mol_default.build(driver='builtin', options={'eri_representation': 'dense'})

    mol_rys = Molecule(atom=atom, unit='bohr', basis=basis)
    mol_rys.build(driver='builtin', options={'eri_representation': 'dense', 'eri_backend': 'rys'})

    np.testing.assert_allclose(mol_rys.overlap, mol_default.overlap, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.hcore, mol_default.hcore, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.eri, mol_default.eri, atol=1e-11, rtol=1e-11)
    if _rys_cy is not None:
        assert mol_default._builtin_build_info['dense_builder'] == 'rys-cython-blocked-auto'
    expected_builder = 'rys-cython-blocked' if _rys_cy is not None else 'rys-screened-mixed'
    assert mol_rys._builtin_build_info['dense_builder'] == expected_builder

    e_default = mol_default.RHF().run(max_cycle=80).e_tot
    e_rys = mol_rys.RHF().run(max_cycle=80).e_tot
    np.testing.assert_allclose(e_rys, e_default, atol=1e-10, rtol=1e-10)


def test_builtin_rys_backend_matches_default_dense_builder_for_d_basis():
    atom = 'H 0 0 0; F 0 0 0.9'
    basis = '6-31g(d,p)'

    mol_default = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_default.build(driver='builtin', options={'eri_representation': 'dense'})

    mol_rys = Molecule(atom=atom, unit='angstrom', basis=basis)
    mol_rys.build(driver='builtin', options={'eri_representation': 'dense', 'eri_backend': 'rys'})

    np.testing.assert_allclose(mol_rys.overlap, mol_default.overlap, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.hcore, mol_default.hcore, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(mol_rys.eri, mol_default.eri, atol=1e-9, rtol=1e-9)
<<<<<<< HEAD
    expected_builder = (
        'cython-shell-os-blocked-mixed-d-fallback'
        if _basis_cy is not None
        else 'python-serial-mixed-d-fallback'
    )
    assert mol_rys._builtin_build_info['dense_builder'] == expected_builder
=======
    assert mol_rys._builtin_build_info['dense_builder'] == 'rys-screened-mixed'
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4

    e_default = mol_default.RHF().run(max_cycle=80).e_tot
    e_rys = mol_rys.RHF().run(max_cycle=80).e_tot
    np.testing.assert_allclose(e_rys, e_default, atol=1e-9, rtol=1e-9)


def test_builtin_basis_aliases_resolve_existing_gbs_files():
    assert _basis_path("cc-pvtz").endswith("cc-pvtz.0.gbs")
    assert _basis_path("aug-cc-pvdz").endswith("aug-cc-pvdz.0.gbs")


def test_builtin_s4_storage_matches_dense_rhf_energy():
    atom = 'H 0 0 0; H 0 0 1.4'

    mol_dense = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense')
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
    mol_dense.build(driver='builtin', eri='dense')
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
    assert mol_aosym._builtin_build_info['dense_builder'] == 'cython-s8-packed'
    np.testing.assert_allclose(mol_aosym.eri_s8, mol_legacy.eri_s8, atol=1e-12)


def test_builtin_packed_jk_contractions_match_dense_tensor():
    mol_dense = Molecule(atom='O 0 0 0; H 0 0 1.8; H 0 1.7 0', unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense')

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


def test_builtin_direct_jk_matches_dense_tensor_and_rhf_energy():
    atom = 'O 0 0 0; H 0 0 1.8; H 0 1.7 0'
    mol_dense = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_dense.build(driver='builtin', eri='dense')

    mol_direct = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol_direct.build(driver='builtin', eri='direct')

    rng = np.random.default_rng(321)
    dm = rng.normal(size=(mol_dense.nao, mol_dense.nao))
    dm = dm + dm.T
    vj_dense = np.einsum('lk,ijkl->ij', dm, mol_dense.eri, optimize=True)
    vk_dense = np.einsum('lk,ilkj->ij', dm, mol_dense.eri, optimize=True)
    from pyqed.qchem.hf.rhf import get_jk
    vj_direct, vk_direct = get_jk(mol_direct, dm)

    np.testing.assert_allclose(vj_direct, vj_dense, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_direct, vk_dense, atol=1e-11, rtol=1e-11)
    assert mol_direct._builtin_direct_jk_data is None
    assert mol_direct.eri_s8 is not None
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
    mol.build(driver='builtin', options={'eri_representation': 'dense'})
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
    mol.build(driver='builtin', options={'eri_representation': 'dense'})
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
    mol.build(driver='builtin', options={'eri_representation': 'dense'})
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
    mol.build(driver='builtin', options={'eri_representation': 'dense'})
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
    assert mol_fact._builtin_build_info['factor_builder'] in {'cython-kernel', 'cython-kernel-blocked'}
    np.testing.assert_allclose(mf_fact.e_tot, mf_dense.e_tot, atol=1e-8, rtol=1e-8)
