import warnings

import numpy as np
import pytest

from pyqed.qchem.basis import unpack_eri_s8


@pytest.mark.parametrize("fallback", [False, True])
@pytest.mark.parametrize("noccupied", [1, 3, 6])
def test_excitation_pairs_and_normal_ordering(noccupied, fallback, monkeypatch):
    import importlib

    module = importlib.import_module("pyqed.qchem.ao2mo")
    from pyqed.qchem.mcscf.perturber_templates import (
        ReferenceIntegrals,
        _hamiltonian_coefficients,
    )

    if fallback:
        monkeypatch.setattr(module, "_ao2mo", None)
    rng = np.random.default_rng(734)
    nao, nmo = 6, 7
    npair = nao * (nao + 1) // 2
    packed = rng.normal(size=npair * (npair + 1) // 2)
    c = rng.normal(size=(nao, nmo))
    dense = unpack_eri_s8(packed, nao)
    reference = np.einsum("abcd,ap,bi,cq,dj->piqj", dense, c, c, c, c, optimize=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        pair = module.ao2mo_excitation(packed, c, noccupied, workers=2)
    np.testing.assert_allclose(
        pair, reference[:, :noccupied, :, :noccupied], atol=2e-11, rtol=2e-12
    )
    ncore = min(2, noccupied - 1)
    core = sum(
        (2 * reference[:, :, i, i] - reference[:, i, i, :] for i in range(ncore)),
        np.zeros((nmo, nmo)),
    )
    h1 = rng.normal(size=(nmo, nmo))
    compact = ReferenceIntegrals(pair, core)
    for frozen in range(ncore + 1):
        actual = _hamiltonian_coefficients(
            h1, compact, ncore, noccupied - ncore, frozen
        )
        expected = _hamiltonian_coefficients(
            h1, reference, ncore, noccupied - ncore, frozen
        )
        for a, b in zip(actual, expected):
            np.testing.assert_allclose(a, b, atol=3e-11, rtol=3e-12)
