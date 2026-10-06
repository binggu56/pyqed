"""Exact inactive-core contractions against the full embedded CAS density."""
import numpy as np
import pytest

from pyqed.optimize import OrbitalContractionPlan, gradient as full_gradient
from pyqed.qchem.mcscf.cocas import energy as full_energy


@pytest.mark.parametrize("ncore", [1, 3])
@pytest.mark.parametrize("symmetric_factors", [False, True])
def test_inactive_core_energy_and_gradient_match_full_density(ncore, symmetric_factors):
    rng = np.random.default_rng(409 + ncore)
    nactive, nao, naux = 2, ncore + 5, 7
    ncol = ncore + nactive
    h = rng.normal(size=(nao, nao))
    h = h + h.T
    factors = rng.normal(size=(naux, nao, nao))
    if symmetric_factors:
        factors = factors + factors.swapaxes(1, 2)
    orbitals = np.linalg.qr(rng.normal(size=(nao, ncol)))[0]
    dm1 = np.zeros((ncol, ncol))
    dm1[:ncore, :ncore] = 2 * np.eye(ncore)
    active_dm1 = rng.normal(size=(nactive, nactive))
    active_dm1 += active_dm1.T
    dm1[ncore:, ncore:] = active_dm1
    dm2 = np.zeros((ncol,) * 4)
    eye = np.eye(ncore)
    dm2[:ncore, :ncore, :ncore, :ncore] = (
        4 * np.einsum('ij,kl->ijkl', eye, eye)
        - 2 * np.einsum('il,kj->ijkl', eye, eye)
    )
    for i in range(ncore):
        dm2[i, i, ncore:, ncore:] = 2 * active_dm1
        dm2[ncore:, ncore:, i, i] = 2 * active_dm1
        dm2[i, ncore:, i, ncore:] = -active_dm1
        dm2[ncore:, i, ncore:, i] = -active_dm1
    dm2[ncore:, ncore:, ncore:, ncore:] = rng.normal(size=(nactive,) * 4)
    full = OrbitalContractionPlan(h, factors, orbitals.shape, dm1.shape, dm2.shape)
    reduced = OrbitalContractionPlan(h, factors, orbitals.shape, dm1.shape, dm2.shape, ncore=ncore)
    args = h, factors, dm1, dm2
    np.testing.assert_allclose(reduced.energy(orbitals, *args), full.energy(orbitals, *args), atol=1e-10, rtol=1e-12)
    gradient = reduced.gradient(orbitals, *args)
    np.testing.assert_allclose(gradient, full.gradient(orbitals, *args), atol=1e-10, rtol=1e-12)
    direction = rng.normal(size=orbitals.shape)
    step = 1e-5
    finite_difference = (reduced.energy(orbitals+step*direction, *args)
                         - reduced.energy(orbitals-step*direction, *args)) / (2*step)
    np.testing.assert_allclose(np.sum(gradient*direction), finite_difference, atol=2e-6, rtol=2e-8)

    orbitals[0, 0] += 0.01
    np.testing.assert_allclose(reduced.energy(orbitals, *args), full_energy(orbitals, *args), atol=1e-10, rtol=1e-12)
    np.testing.assert_allclose(reduced.gradient(orbitals, *args), full_gradient(orbitals, *args), atol=1e-10, rtol=1e-12)
