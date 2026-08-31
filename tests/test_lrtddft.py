import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.dft import RKS, TDA, TDDFT
from pyqed.qchem.tddft import TDA as NativeTDA
from pyqed.qchem.tddft import TDDFT as CanonicalTDDFT
from pyqed.qchem.lrtddft import TDDFT as CompatTDDFT


def test_tddft_module_aliases_remain_compatible():
    assert CanonicalTDDFT is CompatTDDFT


def test_lrtddft_lda_smoke():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='lda').run()

    tda = TDA(mf).run(nstates=1)
    tddft = TDDFT(mf).run(nstates=1)

    assert tda.a.shape == (1, 1, 1, 1)
    assert tddft.a.shape == (1, 1, 1, 1)
    assert np.isfinite(tda.e[0])
    assert np.isfinite(tddft.e[0])
    assert tda.e[0] > 0.0
    assert tddft.e[0] > 0.0


def test_native_tddft_excited_state_gradients_match_pyscf():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import dft as pyscf_dft
    from pyscf import tdscf as pyscf_tdscf

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='lda').run()
    tda = TDA(mf).run(nstates=1)
    tddft = TDDFT(mf).run(nstates=1)

    g0 = tddft.nuc_grad_method().kernel(state=0)
    g_tda = tda.nuc_grad_method().kernel(state=1)
    g_tddft = tddft.nuc_grad_method().kernel(state=1)

    pmol = mol.topyscf()
    pmf = pyscf_dft.RKS(pmol)
    pmf.xc = 'lda'
    pmf.kernel(dm0=np.asarray(mf.dm))

    ptda = pyscf_tdscf.TDA(pmf)
    ptda.nstates = 1
    ptda.kernel()
    ptd = pyscf_tdscf.TDDFT(pmf)
    ptd.nstates = 1
    ptd.kernel()

    g0_ref = pmf.nuc_grad_method().kernel()
    g_tda_ref = ptda.nuc_grad_method().kernel(state=1)
    g_tddft_ref = ptd.nuc_grad_method().kernel(state=1)

    assert g0.shape == (mol.natom, 3)
    assert g_tda.shape == (mol.natom, 3)
    assert g_tddft.shape == (mol.natom, 3)
    assert np.allclose(g0, g0_ref, atol=1e-5, rtol=1e-5)
    assert np.allclose(g_tda, g_tda_ref, atol=1e-6, rtol=1e-6)
    assert np.allclose(g_tddft, g_tddft_ref, atol=1e-6, rtol=1e-6)


def test_native_tddft_native_gradient_backend_is_not_implemented():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RKS(mol, xc='lda').run()
    tddft = TDDFT(mf).run(nstates=1)

    with pytest.raises(NotImplementedError):
        tddft.nuc_grad_method(backend='native').kernel(state=1)


def _align_transition_rows(reference, values):
    aligned = np.array(values, copy=True)
    for state in range(aligned.shape[0]):
        if np.linalg.norm(reference[state] + aligned[state]) < np.linalg.norm(reference[state] - aligned[state]):
            aligned[state] *= -1.0
    return aligned


def test_native_tddft_transition_magnetic_dipoles_match_pyscf_tdhf():
    pytest.importorskip("pyscf")
    from pyscf import gto, scf, tdscf

    atom = 'O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266'

    mol = Molecule(atom=atom, unit='bohr', basis='sto-3g')
    mol.build(eri='dense')
    mf = mol.RHF().run(tol=1e-10, max_cycle=80)

    pyqed_tda = NativeTDA(mf).run(nstates=3)
    pyqed_tdhf = CanonicalTDDFT(mf).run(nstates=3)

    pmol = gto.M(atom=atom, unit='bohr', basis='sto-3g', charge=0, spin=0)
    pmf = scf.RHF(pmol).run(conv_tol=1e-10, verbose=0)
    pyscf_tda = tdscf.TDA(pmf).run(nstates=3, verbose=0)
    pyscf_tdhf = tdscf.TDHF(pmf).run(nstates=3, verbose=0)

    np.testing.assert_allclose(pyqed_tda.e, pyscf_tda.e, atol=5e-8)
    np.testing.assert_allclose(pyqed_tdhf.e, pyscf_tdhf.e, atol=5e-8)

    pyqed_tda_m = _align_transition_rows(
        pyscf_tda.transition_magnetic_dipole(),
        pyqed_tda.transition_magnetic_dipole(convention='pyscf'),
    )
    pyqed_tdhf_m = _align_transition_rows(
        pyscf_tdhf.transition_magnetic_dipole(),
        pyqed_tdhf.transition_magnetic_dipole(convention='pyscf'),
    )

    np.testing.assert_allclose(pyqed_tda_m, pyscf_tda.transition_magnetic_dipole(), atol=5e-8)
    np.testing.assert_allclose(pyqed_tdhf_m, pyscf_tdhf.transition_magnetic_dipole(), atol=5e-8)
    np.testing.assert_allclose(
        _align_transition_rows(
            0.5 * pyscf_tda.transition_magnetic_dipole(),
            pyqed_tda.transition_magnetic_dipole(),
        ),
        0.5 * pyscf_tda.transition_magnetic_dipole(),
        atol=5e-8,
    )
    np.testing.assert_allclose(
        _align_transition_rows(
            0.5 * pyscf_tdhf.transition_magnetic_dipole(),
            pyqed_tdhf.transition_magnetic_dipole(),
        ),
        0.5 * pyscf_tdhf.transition_magnetic_dipole(),
        atol=5e-8,
    )
