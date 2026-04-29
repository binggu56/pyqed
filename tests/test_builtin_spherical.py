import io
import contextlib
import logging

import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF


def test_builtin_spherical_matches_pyscf_rhf_energy():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto, scf

    logging.disable(logging.CRITICAL)

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    basis = "def2-svp"

    mol = Molecule(atom=atom, basis=basis, unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(driver="builtin", options={"coord_type": "spherical"})
        mf = RHF(mol).run(tol=1e-9, max_cycle=100)

    pmol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=False, verbose=0)
    pmf = scf.RHF(pmol)
    pmf.conv_tol = 1e-9
    pmf.max_cycle = 100
    pmf.kernel()

    assert mol.nao == pmol.nao_nr()
    np.testing.assert_allclose(mf.e_tot, pmf.e_tot, atol=1e-9)


def test_builtin_spherical_matches_pyscf_overlap_and_hcore_for_def2_tzvp():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import gto

    logging.disable(logging.CRITICAL)

    atom = "O 0 0 0; H 0 -1.43 1.11; H 0 1.43 1.11"
    basis = "def2-tzvp"

    mol = Molecule(atom=atom, basis=basis, unit="bohr")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(driver="builtin", options={"coord_type": "spherical", "eri_representation": "dense"})

    pmol = gto.M(atom=atom, basis=basis, unit="Bohr", cart=False, verbose=0)

    assert mol.nao == pmol.nao_nr()
    np.testing.assert_allclose(mol.overlap, pmol.intor("int1e_ovlp"), atol=1e-12)
    np.testing.assert_allclose(
        mol.hcore,
        pmol.intor("int1e_kin") + pmol.intor("int1e_nuc"),
        atol=1e-10,
    )
