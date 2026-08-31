import ast
from pathlib import Path

import numpy as np

from pyqed.qchem import CISD, FCI, Molecule, UCISD
from pyqed.qchem.hf import RHF, UHF


def test_restricted_cisd_matches_fci_for_minimal_h2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    cisd = CISD(mf).run()
    fci = FCI(mf).run(nstates=1)

    np.testing.assert_allclose(cisd.e_tot, fci.e_tot[0], atol=1e-8)
    np.testing.assert_allclose(np.trace(cisd.make_rdm1()), mol.nelec, atol=1e-8)


def test_restricted_cisd_make_rdm12_matches_individual_calls():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    cisd = CISD(mf).run()

    dm1, dm2 = cisd.make_rdm12()

    np.testing.assert_allclose(dm1, cisd.make_rdm1(), atol=1e-10)
    np.testing.assert_allclose(dm2, cisd.make_rdm2(), atol=1e-10)


def test_restricted_cisd_direct_sigma_backend_matches_default_energy():
    mol = Molecule(
        atom='O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587',
        unit='angstrom',
        basis='sto-3g',
    )
    mol.build()

    mf = RHF(mol).run()
    ref = CISD(mf).run()
    direct = CISD(mf).run(sigma_backend='direct')

    np.testing.assert_allclose(direct.e_tot, ref.e_tot, atol=1e-10)
    np.testing.assert_allclose(np.trace(direct.make_rdm1()), mol.nelec, atol=1e-8)
    ref_dm1, ref_dm2 = ref.make_rdm12()
    direct_dm1, direct_dm2 = direct.make_rdm12()
    np.testing.assert_allclose(direct_dm1, ref_dm1, atol=1e-5)
    np.testing.assert_allclose(direct_dm2, ref_dm2, atol=1e-5)


def test_ucisd_uses_spin_resolved_occupations_for_open_shell_uhf():
    mol = Molecule(
        atom='H 0 0 0; H 0 0 1.5',
        unit='angstrom',
        basis='6-31g',
        spin=2,
    )
    mol.build()

    mf = UHF(mol).run(max_cycle=50)
    ucisd = UCISD(mf)

    assert ucisd.nocc == (mf.na, mf.nb)
    assert ucisd.nvir == (mf.nmo - mf.na, mf.nmo - mf.nb)

    h_ci = ucisd.buildH()
    assert h_ci.shape[0] == h_ci.shape[1]
    assert np.all(np.isfinite(h_ci))


def test_ci_modules_do_not_import_pyscf_at_runtime():
    root = Path(__file__).resolve().parents[1]
    for relpath in ('pyqed/qchem/ci/cisd.py', 'pyqed/qchem/ci/fci.py'):
        source = (root / relpath).read_text(encoding='utf-8')
        tree = ast.parse(source, filename=relpath)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
                assert all(name != 'pyscf' and not name.startswith('pyscf.') for name in names), relpath
            elif isinstance(node, ast.ImportFrom):
                assert node.module != 'pyscf' and not (
                    isinstance(node.module, str) and node.module.startswith('pyscf.')
                ), relpath
