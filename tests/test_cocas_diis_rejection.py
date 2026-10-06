"""CO must recover from a bad DIIS direction below the trust-radius floor."""

from types import SimpleNamespace

import numpy as np
import pytest

import pyqed.qchem.mcscf.cocas as co


@pytest.mark.parametrize("state_average", [False, True])
def test_energy_increasing_diis_retries_ordinary_orbital_update(monkeypatch, state_average):
    matrix = np.diag([0.0, 1.0])
    initial = np.array([[1.0], [0.1]])
    initial /= np.linalg.norm(initial)

    class Plan:
        def __init__(self, *_args, ncore=0):
            pass

        def energy(self, orbitals, *_args):
            return float((orbitals.T @ matrix @ orbitals).item())

        def gradient(self, orbitals, *_args):
            return 2 * matrix @ orbitals

    plan = Plan()

    def make_state(orbitals):
        energy = plan.energy(orbitals)
        return SimpleNamespace(
            ncore=0, nstates=2 if state_average else 1, verbose=0,
            e_tot=np.array([energy, energy + 0.5]) if state_average else energy,
            make_rdm12=lambda *_args, **_kwargs: (np.ones((1, 1)), np.ones((1, 1, 1, 1))),
        )

    trials = []

    def solve(_mc, *_args, mo_coeff, **_kwargs):
        trials.append(plan.energy(mo_coeff))
        return make_state(mo_coeff)

    def bad_diis(_diis, base, _candidate, **_kwargs):
        expected = co._physical_orbital_gradient(
            base, plan.gradient(base), 0, 1, active_active=False)
        np.testing.assert_allclose(_kwargs['residual'], expected)
        candidate_gradient = co._physical_orbital_gradient(
            _candidate, plan.gradient(_candidate), 0, 1, active_active=False)
        assert np.linalg.norm(_kwargs['residual']) > 100 * np.linalg.norm(candidate_gradient)
        trial = base + np.array([[0.0], [0.01]])
        return trial / np.linalg.norm(trial)

    monkeypatch.setattr(co, "OrbitalContractionPlan", Plan)
    monkeypatch.setattr(co, "_run_macro_casci", solve)
    monkeypatch.setattr(co, "_apply_orbital_diis", bad_diis)
    events = []
    args = (initial, 2, 1, np.eye(2), matrix, np.zeros((2, 2, 2, 2)))
    options = dict(max_cycles=1, tol=1e-14, optimizer="LBFGS", diis_residual="gradient", optimizer_tol=1e-8,
                   optimizer_max_steps=20, macro_trust_radius=0.05, macro_trust_min=0.05,
                   macro_callback=events.append, raise_on_nonconvergence=False)
    if state_average:
        orbitals, result = co.kernel_state_average(make_state(initial), np.array([0.5, 0.5]),
                                                   *args, **options)
    else:
        orbitals, result = co.kernel(make_state(initial), *args, **options)
    assert len(trials) == 2
    assert trials[0] > plan.energy(initial)
    assert trials[1] < plan.energy(initial)
    assert plan.energy(orbitals) == pytest.approx(trials[1])
    assert events[-1]["diagnostics"]["diis_rejected"]
    assert events[-1]["diagnostics"]["accepted"]
    np.testing.assert_allclose(orbitals.T @ orbitals, np.eye(1), atol=1e-12)
