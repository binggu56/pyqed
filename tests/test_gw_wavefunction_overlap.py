import numpy as np
import pytest

from pyqed.gw.bse import BSE, TDA
from pyqed.gw.gw import GW
from pyqed.qchem import Molecule
from pyqed.qchem.hf.rhf import RHF


def _h2_gw(distance):
    mol = Molecule(
        atom=f"H 0 0 0; H 0 0 {distance}",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    return GW(mf, screening="TDH", eta=1.0e-3).run()


def test_gw_tda_and_bse_wavefunction_overlap_use_stored_vectors():
    gw = _h2_gw(0.74)

    tda = TDA(gw).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)
    bse = BSE(gw).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)

    np.testing.assert_allclose(tda.wavefunction_overlap(tda), np.eye(1), atol=1.0e-10)
    np.testing.assert_allclose(bse.wavefunction_overlap(bse), np.eye(1), atol=1.0e-10)
    with pytest.raises(TypeError):
        tda.wavefunction_overlap(tda, tda.x, tda.x)


def test_gw_tda_and_bse_wavefunction_overlap_between_geometries():
    gw_bra = _h2_gw(0.74)
    gw_ket = _h2_gw(0.78)

    tda_bra = TDA(gw_bra).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)
    tda_ket = TDA(gw_ket).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)
    bse_bra = BSE(gw_bra).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)
    bse_ket = BSE(gw_ket).run(nroots=1, use_qp=False, low_rank=False, return_vectors=True)

    tda_overlap = tda_bra.wavefunction_overlap(tda_ket)
    bse_overlap = bse_bra.wavefunction_overlap(bse_ket)

    assert np.all(np.isfinite(tda_overlap))
    assert np.all(np.isfinite(bse_overlap))
    assert 0.8 < abs(tda_overlap[0, 0]) <= 1.05
    assert 0.8 < abs(bse_overlap[0, 0]) <= 1.05


def test_h2o_ccpvdz_wavefunction_overlap_keeps_cartesian_ao_metric():
    def run_h2o(delta):
        atom = (
            f"O 0 0 0; "
            f"H 0 -0.757 {0.587 + delta}; "
            f"H 0  0.757 0.587"
        )
        mol = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
        mol.build(driver="builtin", eri="dense")
        mf = RHF(mol).run(verbose=0)
        tda = TDA(mf, screening="TDH", eta=1.0e-3).run(
            nroots=2,
            use_qp=False,
            low_rank=False,
            return_vectors=True,
        )
        return tda

    tda_bra = run_h2o(0.0)
    tda_ket = run_h2o(0.05)

    assert tda_bra._scf.mo_coeff.shape[0] == 25
    assert tda_bra.mol.nao == 25
    overlap = tda_bra.wavefunction_overlap(tda_ket)

    assert overlap.shape == (2, 2)
    assert np.all(np.isfinite(overlap))
    assert np.max(np.abs(overlap)) <= 1.0 + 1.0e-10


def test_gw_bse_import_without_pyscf(monkeypatch):
    import builtins
    import importlib
    import sys

    original_import = builtins.__import__

    def block_pyscf(name, *args, **kwargs):
        if name == "pyscf" or name.startswith("pyscf."):
            raise ImportError("blocked pyscf")
        return original_import(name, *args, **kwargs)

    for name in list(sys.modules):
        if name == "pyqed.gw.gw" or name == "pyqed.gw.bse":
            del sys.modules[name]
    monkeypatch.setattr(builtins, "__import__", block_pyscf)

    gw_module = importlib.import_module("pyqed.gw.gw")
    bse_module = importlib.import_module("pyqed.gw.bse")

    assert gw_module.GW is not None
    assert bse_module.BSE is not None
