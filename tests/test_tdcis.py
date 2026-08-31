from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem import Molecule, TDCIS, cis_determinant_basis
from pyqed.qchem.hf import RHF
from pyqed.qchem.tdcis import TDCIS as DirectTDCIS


def _h2_rhf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    return RHF(mol).run()


def test_tdcis_is_exported():
    assert TDCIS is DirectTDCIS


def test_cis_determinant_basis_contains_reference_and_singles():
    mf = _h2_rhf()
    binary = cis_determinant_basis(mf)

    nocc = int(np.count_nonzero(np.asarray(mf.mo_occ) > 0))
    nvir = int(mf.nmo - nocc)
    assert binary.shape == (1 + 2 * nocc * nvir, 2, mf.nmo)
    np.testing.assert_array_equal(binary[0, 0], np.array([1, 0], dtype=np.int8))
    np.testing.assert_array_equal(binary[0, 1], np.array([1, 0], dtype=np.int8))
    assert all(np.sum(det[0]) == nocc for det in binary)
    assert all(np.sum(det[1]) == nocc for det in binary)


def test_cis_determinant_basis_rejects_rohf_occupations():
    mf = SimpleNamespace(mo_occ=np.array([2.0, 1.0, 0.0]))

    with pytest.raises(ValueError, match="ROHF"):
        cis_determinant_basis(mf)


def test_tdcis_field_free_preserves_norm_and_populations():
    mf = _h2_rhf()
    td = TDCIS(mf, nstates=3)

    traj = td.run(dt=0.05, nsteps=6, ci0=0)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.populations[:, 0], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(traj.populations[:, 1:], 0.0, atol=1.0e-10)
    assert traj.ci.shape == (7, td.cis_binary.shape[0])


def test_tdcis_kick_generates_response():
    mf = _h2_rhf()
    td = TDCIS(mf, nstates=3)

    traj = td.run(
        dt=0.05,
        nsteps=8,
        ci0=0,
        kick={"strength": 1.0e-3, "axis": "z"},
    )

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(np.abs(traj.dipoles[:, 2])) > 1.0e-6
