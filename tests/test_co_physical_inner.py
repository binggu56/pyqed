"""Physical inner directions and transport of retained manifold secants."""

import numpy as np
import pytest

from pyqed.optimize import grad, lbfgs_direction, minimize, norm, project, update_lbfgs_history
from pyqed.qchem.mcscf.cocas import RelaxedOrbitalLBFGS, _physical_orbital_gradient


def test_retained_secants_follow_the_current_tangent_space():
    old = np.eye(5, 3)
    rng = np.random.default_rng(4)
    step = grad(old, rng.normal(size=old.shape))
    s_history, y_history = [step.copy()], [2 * step.copy()]
    current = project(old + 0.2 * rng.normal(size=old.shape))
    expected = grad(current, step)
    # A rejected new pair must still transport all the retained old pairs.
    update_lbfgs_history(s_history, y_history, current, step, -step, 7)
    assert len(s_history) == 1
    np.testing.assert_allclose(s_history[0], expected, atol=1e-13)
    np.testing.assert_allclose(y_history[0], 2 * expected, atol=1e-13)
    np.testing.assert_allclose(current.T @ s_history[0] + s_history[0].T @ current,
                               0, atol=1e-13)


@pytest.mark.parametrize("algorithm", ["RCG", "LBFGS"])
@pytest.mark.parametrize("active_active", [False, True])
def test_redundant_active_rotations_are_removed_only_for_exact_cas(algorithm, active_active):
    initial = np.eye(3)
    target = initial.copy()
    angle = 0.25
    target[1:, 1:] = [[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]]
    projection = lambda x, g: _physical_orbital_gradient(
        x, g, 1, 2, active_active=active_active)
    final, value = minimize(
        lambda x: float(np.sum((x - target) ** 2)), initial,
        gradient_fn=lambda x: 2 * (x - target), projection_fn=projection,
        algorithm=algorithm, tau=1, epsilon=1e-9, max_iterations=50,
    )
    np.testing.assert_allclose(final.T @ final, np.eye(3), atol=1e-12)
    if active_active:
        assert value < 1e-17
    else:
        np.testing.assert_allclose(final, initial, atol=1e-13)
        assert value > 0.01


def test_physical_inner_minimizes_a_multiorbital_quotient_objective():
    matrix = np.diag([0., 1., 3., 10., 100.])
    occupations = np.diag([2., 1., 1.])
    initial = project(np.eye(5, 3) + 0.05 * np.random.default_rng(5).normal(size=(5, 3)))
    projection = lambda x, g: _physical_orbital_gradient(x, g, 1, 2, active_active=False)
    final, value = minimize(
        lambda x: float(np.trace(x.T @ matrix @ x @ occupations)), initial,
        gradient_fn=lambda x: 2 * matrix @ x @ occupations,
        projection_fn=projection, algorithm="LBFGS", tau=1,
        epsilon=1e-7, max_iterations=150,
    )
    assert value == pytest.approx(4., abs=1e-10)
    assert norm(projection(final, 2 * matrix @ final @ occupations)) < 1e-7


def test_physical_secant_transport_removes_redundant_blocks():
    rng = np.random.default_rng(8)
    current = project(rng.normal(size=(6, 4)))
    projection = lambda x, g: _physical_orbital_gradient(x, g, 2, 2, active_active=False)
    step = rng.normal(size=current.shape)
    s_history, y_history = [step.copy()], [2 * step.copy()]
    update_lbfgs_history(s_history, y_history, current, step, 3 * step, 7,
                         projection_fn=projection)
    for s, y in zip(s_history, y_history):
        for vector in (s, y):
            vertical = current.T @ vector
            np.testing.assert_allclose(vertical[:2, :2], 0, atol=1e-12)
            np.testing.assert_allclose(vertical[2:, 2:], 0, atol=1e-12)


def test_lbfgs_with_positive_inverse_model_matches_dense_bfgs():
    rng = np.random.default_rng(22)
    curvature = np.diag([1., 3., 50., 1000.])
    inverse = np.diag([1., 1/3, 1/50, 1/1000])
    s = [rng.normal(size=(4, 1)) for _ in range(3)]
    y = [curvature @ vector for vector in s]
    gradient = rng.normal(size=(4, 1))
    # Independently assemble inverse BFGS rank-two updates.
    h = inverse.copy()
    for step, change in zip(s, y):
        rho = 1 / float((step.T @ change).item())
        left = np.eye(4) - rho * step @ change.T
        h = left @ h @ left.T + rho * step @ step.T
    actual = lbfgs_direction(gradient, s, y, initial_inverse=lambda v: inverse @ v)
    np.testing.assert_allclose(actual, h @ gradient, atol=1e-12)
    np.testing.assert_allclose(actual, inverse @ gradient, atol=1e-12)


@pytest.mark.parametrize("algorithm", ["AH", "NEWTON"])
def test_custom_projection_rejects_incompatible_hessian(algorithm):
    with pytest.raises(ValueError, match="custom tangent projection"):
        minimize(lambda x: float(np.sum(x*x)), np.eye(2),
                 algorithm=algorithm, projection_fn=grad)


def test_relaxed_preconditioner_is_covariant_in_a_noncanonical_reference_basis():
    rng = np.random.default_rng(12)
    base = project(rng.normal(size=(6, 3)))
    change = project(rng.normal(size=(6, 6)))
    fock = np.diag([-20., -2., -.3, .1, 1., 3.])
    density = np.diag([2., 1., 1.])
    gradient = rng.normal(size=base.shape)
    plain = RelaxedOrbitalLBFGS(1, 2, reference_fock=fock)
    transformed = RelaxedOrbitalLBFGS(1, 2, reference_fock=change.T @ fock @ change)
    for _ in range(3):
        candidate = plain.update(base, gradient, density)
        actual = transformed.update(change.T @ base, change.T @ gradient, density)
        np.testing.assert_allclose(change @ actual, candidate, atol=1e-12)
        np.testing.assert_allclose(candidate.T @ candidate, np.eye(3), atol=1e-12)
        assert norm(candidate-base) <= 0.25 + 1e-12
        base = candidate
        gradient = rng.normal(size=base.shape)


@pytest.mark.parametrize("active_active", [False, True])
def test_relaxed_update_retains_finite_bond_active_variables(active_active):
    base = np.eye(2)
    gradient = np.array([[0., 1.], [-1., 0.]])
    helper = RelaxedOrbitalLBFGS(0, 2, active_active=active_active,
                                reference_fock=np.diag([-1., 1.]))
    candidate = helper.update(base, gradient, np.eye(2))
    if active_active:
        assert norm(candidate - base) > 0.01
        assert np.sum(gradient * (candidate-base)) < 0
    else:
        np.testing.assert_allclose(candidate, base, atol=1e-13)


@pytest.mark.parametrize("factorized", [False, True])
def test_relaxed_cocas_lih_matches_fixed_rdm_minimum(factorized):
    from pyqed.qchem import COCAS, Molecule
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    if factorized:
        mol.build(eri="cd", options={"low_rank_tol": 1e-10, "eri_screen_tol": 0.})
    else:
        mol.build()
    mf = mol.RHF().run(tol=1e-10)
    ordinary = COCAS(mf, ncas=2, nelecas=2, optimizer="LBFGS", max_cycles=40,
                      use_cholesky=factorized).run()
    relaxed = COCAS(mf, ncas=2, nelecas=2, orbital_update="relaxed_lbfgs",
                     diis=False, max_cycles=100, orb_grad_tol=1e-5,
                     use_cholesky=factorized).run()
    assert relaxed.converged
    np.testing.assert_allclose(relaxed.e_tot, ordinary.e_tot, atol=1e-7, rtol=0)
    final = relaxed.macro_diagnostics[-1]
    assert final["gn"] < 1e-5
    assert final["solver"]
    np.testing.assert_allclose(relaxed.mo_coeff.T @ mf.get_ovlp() @ relaxed.mo_coeff,
                               np.eye(3), atol=1e-11)
