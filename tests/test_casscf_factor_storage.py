from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF


@pytest.mark.parametrize('occupied', [[], [0, 2], list(range(7))])
@pytest.mark.parametrize('complex_values', [False, True])
def test_factor_veff_matches_full_contraction(occupied, complex_values):
    rng = np.random.default_rng(39)
    b = rng.normal(size=(13, 7, 7))
    if complex_values:
        b = b + 1j*rng.normal(size=b.shape)
    dm = np.zeros((7, 7), dtype=b.dtype)
    d = rng.normal(size=(len(occupied), len(occupied)))
    if complex_values:
        d = d + 1j*rng.normal(size=d.shape)
    dm[np.ix_(occupied, occupied)] = d
    parent = SimpleNamespace(mol=None, mo_occ=np.array([2, 2, 0, 0, 0, 0, 0]),
                             energy_nuc=lambda: 0.)
    frozen = SecondOrderCASSCF._FrozenFactorRHF(parent, np.eye(7), b, np.eye(7))
    assert np.shares_memory(frozen.eri_factors, b)
    assert not frozen.eri_factors.flags.writeable
    ref = (np.einsum('Pkl,lk,Pij->ij', b, dm, b, optimize=True)
           -.5*np.einsum('Pil,lk,Pkj->ij', b, dm, b, optimize=True))
    np.testing.assert_allclose(frozen.get_veff(dm), ref, atol=1e-11)
    solver = object.__new__(SecondOrderCASSCF)
    u = rng.normal(size=(7, 7))
    if complex_values:
        u = u + 1j*rng.normal(size=u.shape)
    _, rotated = solver._transform_frozen_factor_integrals(np.eye(7), b, u)
    np.testing.assert_allclose(rotated,
        np.einsum('pi,Ppq,qj->Pij', u.conj(), b, u, optimize=True), atol=1e-11)
