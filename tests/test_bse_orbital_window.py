"""Restricted excitation spaces retain the full reference dielectric screening."""

import importlib

import numpy as np
import pytest


@pytest.fixture(scope="module")
def reference():
    pyscf = pytest.importorskip("pyscf")
    from pyscf import df
    from pyqed.qchem.basis import PackedRIFactors
    from pyqed.gw.gw import GW
    mol = pyscf.gto.M(atom="O 0 0 0; H 0 -.757 .587; H 0 .757 .587",
                      basis="sto-3g", verbose=0)
    mf = pyscf.scf.RHF(mol)
    mf.chkfile, mf.conv_tol = None, 1e-12
    mf.kernel()
    fitted = df.DF(mol).build()
    mol.eri_factors = PackedRIFactors(np.concatenate(list(fitted.loop())), mol.nao)
    return GW(mf, eta=1e-6, freq_int="contour").run(nw=64)


@pytest.mark.parametrize("window", [(2, 1), (3, 2), (5, 2)])
def test_window_operator_equals_full_kernel_submatrix(reference, window, monkeypatch):
    b = importlib.import_module("pyqed.gw.bse")
    full = b.BSE(reference, screening_energies='hf')
    a, bb = b.bse_AB_matrices(full)
    no, nv = window
    selected = [i*(full.nso-full.nocc)+v
                for i in range(full.nocc-no, full.nocc) for v in range(nv)]
    expected_a, expected_b = [m[np.ix_(selected, selected)] for m in [a, bb]]
    restricted = b.BSE(reference, orbital_window=window, screening_energies='hf')
    assert restricted._static_screening.cholesky is reference._contour.static_screening.cholesky
    assert reference._contour.pairs.shape[1] == 10  # All five occupied x two virtual transitions.
    monkeypatch.setattr(b, "rpa", lambda *a, **k: pytest.fail("window screening rebuild"))
    actual_a, actual_b = b.bse_AB_matrices(restricted)
    np.testing.assert_allclose(actual_a, expected_a, atol=1e-12)
    np.testing.assert_allclose(actual_b, expected_b, atol=1e-12)
    rng = np.random.default_rng(36)
    vectors = rng.normal(size=(2*no*nv, 3))
    expected = np.block([[expected_a, expected_b], [-expected_b, -expected_a]]) @ vectors
    np.testing.assert_allclose(b._bse_full_matmat(restricted, vectors, 2), expected, atol=1e-12)


def test_partial_qp_window_and_full_window_limit(reference):
    from pyqed.gw.bse import BSE, TDA
    import copy
    partial = copy.copy(reference)
    partial.e_qp = reference.e_qp.copy()
    partial.e_qp[:2] = np.nan
    for cls in [BSE, TDA]:
        expected = cls(reference, screening_energies='hf', orbital_window=(3, 2)).run(nroots=3, low_rank=False).e
        actual = cls(partial, screening_energies='hf', orbital_window=(3, 2)).run(nroots=3, tol=1e-9).e
        np.testing.assert_allclose(actual, expected, atol=1e-8)
        full = cls(reference, screening_energies='hf').run(nroots=3, low_rank=False).e
        limit = cls(reference, screening_energies='hf', orbital_window=(5, 2)).run(nroots=3, low_rank=False).e
        np.testing.assert_allclose(limit, full, atol=1e-12)
    with pytest.raises(ValueError, match="every orbital"):
        BSE(partial, orbital_window=(4, 2))


@pytest.mark.parametrize("window", [(0, 2), (2, 0), (6, 2), (2, 3), (True, 1), (2.5, 1)])
def test_invalid_windows(reference, window):
    from pyqed.gw.bse import BSE
    with pytest.raises(ValueError, match="orbital_window"):
        BSE(reference, orbital_window=window)


@pytest.mark.parametrize('tda', [False, True])
def test_scanner_retains_window_with_partial_qp_energies(reference, monkeypatch, tda):
    import copy
    from pyqed.gw.bse import BSE, TDA
    partial = copy.copy(reference)
    partial.e_qp = reference.e_qp.copy()
    partial.e_qp[:2] = np.nan
    cls = TDA if tda else BSE
    base = cls(partial, screening_energies='hf', orbital_window=(3, 2)).run(nroots=2, tol=1e-9)
    scanner = base.as_scanner(nroots=2, return_object=True, run_kwargs={'tol': 1e-9})
    monkeypatch.setattr(scanner, '_prepare_mol', lambda value: base.mol)
    monkeypatch.setattr(scanner, '_run_mf', lambda mol: base._scf)
    monkeypatch.setattr(scanner, '_run_gw', lambda mf: partial)
    scanned = scanner(base.mol)
    assert scanned.orbital_window == (3, 2)
    np.testing.assert_array_equal(scanned.orbital_indices, base.orbital_indices)
    np.testing.assert_allclose(scanned.e, base.e, atol=1e-10)
