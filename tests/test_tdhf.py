import importlib.util
from pathlib import Path

import numpy as np
from pyscf import gto, scf, tdscf


def _load_legacy_tdhf():
    module_path = Path(__file__).resolve().parents[1] / 'pyqed' / 'qchem' / 'tdscf' / 'tdhf.py'
    spec = importlib.util.spec_from_file_location('pyqed_legacy_tdhf', module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.TDHF


def test_legacy_tdhf_matches_pyscf_rhf_singlet_and_triplet():
    TDHF = _load_legacy_tdhf()

    mol = gto.M(atom='H 0 0 0; H 0 0 1.4', unit='Bohr', basis='sto-3g')
    mf = scf.RHF(mol).run()

    ref_tda_singlet = tdscf.TDA(mf)
    ref_tdhf_singlet = tdscf.TDHF(mf)
    ref_tda_triplet = tdscf.TDA(mf)
    ref_tda_triplet.singlet = False
    ref_tdhf_triplet = tdscf.TDHF(mf)
    ref_tdhf_triplet.singlet = False

    tdhf = TDHF(mf)

    tdhf.singlet = True
    e_tda_singlet, _ = tdhf.run(using_tda=True, method='TDHF')
    e_tdhf_singlet, _ = tdhf.run(using_tda=False, using_casida=True, method='TDHF')

    tdhf.singlet = False
    e_tda_triplet, _ = tdhf.run(using_tda=True, method='TDHF')
    e_tdhf_triplet, _ = tdhf.run(using_tda=False, using_casida=True, method='TDHF')

    np.testing.assert_allclose(e_tda_singlet[:1], ref_tda_singlet.kernel(nstates=1)[0], atol=1e-8)
    np.testing.assert_allclose(e_tdhf_singlet[:1], ref_tdhf_singlet.kernel(nstates=1)[0], atol=1e-8)
    np.testing.assert_allclose(e_tda_triplet[:1], ref_tda_triplet.kernel(nstates=1)[0], atol=1e-8)
    np.testing.assert_allclose(e_tdhf_triplet[:1], ref_tdhf_triplet.kernel(nstates=1)[0], atol=1e-8)

