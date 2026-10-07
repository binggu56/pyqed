"""Post-CI residuals, gauge covariance and conditioning of outer CO DIIS."""

import numpy as np
import pytest

from pyqed.qchem.mcscf.cocas import (
    OrbitalDIIS, _orthonormalize_columns, _physical_orbital_gradient,
)


def rotation(angle):
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


@pytest.mark.parametrize('residual_kind', ['transported_step', 'gradient'])
def test_history_is_covariant_under_independent_redundant_gauges(residual_kind):
    rng = np.random.default_rng(17)
    base = np.eye(6, 4)
    plain = OrbitalDIIS(residual_kind=residual_kind)
    gauged = OrbitalDIIS(residual_kind=residual_kind)
    for step in range(4):
        base = _orthonormalize_columns(base + 0.02 * rng.normal(size=base.shape))
        error = _physical_orbital_gradient(base, rng.normal(size=base.shape),
                                         2, 2, active_active=False)
        candidate = _orthonormalize_columns(base - 0.03 * error)
        gauge = np.zeros((4, 4))
        gauge[:2, :2] = rotation(0.7 * (step + 1))
        gauge[2:, 2:] = rotation(-0.4 * (step + 1))
        expected = plain.update(base, candidate, residual=error, ncore=2, ncas=2,
                                active_active=False)
        actual = gauged.update(base @ gauge, candidate @ gauge, residual=error @ gauge,
                               ncore=2, ncas=2, active_active=False)
        np.testing.assert_allclose(actual, expected @ gauge, atol=2e-12)
        np.testing.assert_allclose(actual.T @ actual, np.eye(4), atol=2e-12)
    assert plain.last_info['diis_used']
    assert gauged.last_info['diis_used']


@pytest.mark.parametrize('scale', [1.0, 1e-9, 1e-90])
def test_regularization_is_relative_to_residual_scale(scale):
    base = np.array([[1.0], [0.0]])
    candidates = [np.array([[np.cos(a)], [np.sin(a)]]) for a in (0.1, -0.05)]
    diis = OrbitalDIIS(residual_kind='gradient')
    for coefficient, candidate in zip((1.0, -0.5), candidates):
        result = diis.update(base, candidate, residual=np.array([[0.0], [coefficient * scale]]),
                             ncore=0, ncas=1, active_active=False)
    expected = _orthonormalize_columns(candidates[0] / 3 + 2 * candidates[1] / 3)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert diis.last_info['diis_used']
    assert diis.last_info['diis_max_weight'] == pytest.approx(2 / 3)


def test_finite_bond_active_rotations_remain_in_diis_residual():
    base = np.eye(2)
    error = np.array([[0.0, -1.0], [1.0, 0.0]])
    diis = OrbitalDIIS(residual_kind='gradient')
    for coefficient, angle in ((1.0, 0.1), (-0.5, -0.05)):
        result = diis.update(base, rotation(angle), residual=coefficient * error,
                             ncore=0, ncas=2, active_active=True)
    expected = _orthonormalize_columns(rotation(0.1) / 3 + 2 * rotation(-0.05) / 3)
    np.testing.assert_allclose(result, expected, atol=1e-10)
    assert diis.last_info['diis_used']


def test_ill_conditioned_history_falls_back_without_large_coefficients():
    base = np.array([[1.0], [0.0]])
    diis = OrbitalDIIS(residual_kind='gradient')
    candidate = _orthonormalize_columns(base + np.array([[0.0], [-0.01]]))
    for coefficient in (1.0, 0.99):
        result = diis.update(base, candidate, residual=np.array([[0.0], [coefficient]]),
                             ncore=0, ncas=1, active_active=False)
    np.testing.assert_allclose(result, candidate)
    assert not diis.last_info['diis_used']
    diis.reset()
    assert not diis.bases and not diis.errors and not diis.vectors


@pytest.mark.parametrize('state_average', [False, True])
@pytest.mark.parametrize('physical_inner,orbital_update,gap_floor', [
    (False, 'fixed_rdm', None), (True, 'fixed_rdm', None),
    (False, 'relaxed_lbfgs', None), (True, 'relaxed_lbfgs', None),
    (False, 'fixed_rdm', 0.1),
])
def test_orbital_updates_use_current_ci_rdms(monkeypatch, state_average, physical_inner, orbital_update, gap_floor):
    from types import SimpleNamespace
    import pyqed.qchem.mcscf.cocas as co

    matrix = np.diag([0.0, 1.0])
    initial = _orthonormalize_columns(np.array([[1.0], [0.1]]))
    weights = np.array([0.25, 0.75])

    def density(orbitals, root=0):
        return np.ones((1, 1)) * (1 + orbitals[1, 0] + root)

    class Plan:
        def __init__(self, *_args, **_kwargs):
            pass

        def energy(self, orbitals, *_args):
            return float((orbitals.T @ matrix @ orbitals).item())

        def gradient(self, orbitals, _h, _eri, d1, _d2):
            return 2 * d1[0, 0] * matrix @ orbitals

    def state(orbitals):
        e = Plan().energy(orbitals)
        return SimpleNamespace(
            ncore=0, nstates=2 if state_average else 1, verbose=0,
            e_tot=np.array([e, e + 0.5]) if state_average else e,
            make_rdm12=lambda root, **_kwargs: (density(orbitals, root), np.ones((1, 1, 1, 1))),
        )

    seen = []

    def apply(_helper, base, candidate, *, residual, **_kwargs):
        d1 = density(base)
        if state_average:
            d1 = sum(weight * density(base, root) for root, weight in enumerate(weights))
        expected = _physical_orbital_gradient(
            base, Plan().gradient(base, None, None, d1, None), 0, 1, active_active=False)
        np.testing.assert_allclose(residual, expected, atol=1e-12)
        seen.append(base.copy())
        return candidate

    monkeypatch.setattr(co, 'OrbitalContractionPlan', Plan)
    monkeypatch.setattr(co, '_apply_orbital_diis', apply)
    monkeypatch.setattr(co, '_run_macro_casci', lambda _mc, *_a, mo_coeff, **_kw: state(mo_coeff))
    def inner(_f, base, *, projection_fn, isd_preconditioner, **_kw):
        if gap_floor is not None:
            skew = np.array([[0.0, 1.0], [-1.0, 0.0]])
            np.testing.assert_allclose(isd_preconditioner(skew), skew / 2)
        else:
            assert isd_preconditioner is None
        if physical_inner:
            vector = np.array([[0.2], [0.5]])
            np.testing.assert_allclose(projection_fn(base, vector),
                _physical_orbital_gradient(base, vector, 0, 1, active_active=False))
        else:
            assert projection_fn is None
        return _orthonormalize_columns(base + np.array([[0.0], [-0.02]])), 0.0
    monkeypatch.setattr(co, 'minimize', inner)
    def relaxed_update(_self, base, euclidean, d1):
        expected = density(base)
        if state_average:
            expected = sum(weight * density(base, root) for root, weight in enumerate(weights))
        np.testing.assert_allclose(d1, expected)
        candidate = _orthonormalize_columns(base + np.array([[0.0], [-0.02]]))
        return apply(None, base, candidate, residual=_physical_orbital_gradient(
            base, euclidean, 0, 1, active_active=False))
    monkeypatch.setattr(co.RelaxedOrbitalLBFGS, 'update', relaxed_update)
    args = (initial, 2, 1, np.eye(2), matrix, np.zeros((2, 2, 2, 2)))
    options = dict(max_cycles=2, tol=0, diis_residual='gradient',
                   physical_inner=physical_inner, orbital_update=orbital_update,
                   diis=orbital_update == 'fixed_rdm', raise_on_nonconvergence=False)
    if gap_floor is not None:
        options.update(optimizer='ISD', isd_gap_floor=gap_floor, reference_fock=matrix)
    if state_average:
        co.kernel_state_average(state(initial), weights, *args, **options)
    else:
        co.kernel(state(initial), *args, **options)
    assert len(seen) >= 2
    assert not np.allclose(seen[0], seen[1])


def test_transported_step_diis_accelerates_a_slow_fixed_point_map():
    diis = OrbitalDIIS(residual_kind='transported_step')
    for angle in (0.12, 0.06):
        base = np.array([[np.cos(angle)], [np.sin(angle)]])
        candidate = np.array([[np.cos(angle / 2)], [np.sin(angle / 2)]])
        result = diis.update(base, candidate, ncore=0, ncas=1, active_active=False)
    # The unaccelerated map has contraction factor 1/2; DIIS cancels its
    # linear error, leaving only the small manifold-curvature error.
    assert abs(result[1, 0]) < 1e-4
    assert abs(candidate[1, 0]) > 0.02
    np.testing.assert_allclose(result.T @ result, np.eye(1), atol=1e-12)
    assert diis.last_info['diis_used']


@pytest.mark.parametrize('residual_kind', ['step', 'transported_step', 'gradient'])
def test_rank_deficient_extrapolation_keeps_a_valid_candidate(residual_kind):
    base = np.eye(2)
    error = np.array([[0.0, -1.0], [1.0, 0.0]])
    diis = OrbitalDIIS(residual_kind=residual_kind)
    candidate = rotation(0.1)
    diis.update(base, candidate, residual=error, ncore=0, ncas=2, active_active=True)
    result = diis.update(-base, -candidate, residual=-error,
                         ncore=0, ncas=2, active_active=True)
    # Opposite errors produce equal weights, which would cancel both columns.
    np.testing.assert_allclose(result, -candidate)
    np.testing.assert_allclose(result.T @ result, np.eye(2), atol=1e-12)
    assert not diis.last_info['diis_used']
