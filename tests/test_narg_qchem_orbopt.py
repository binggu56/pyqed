import numpy as np
import pytest
from scipy.linalg import expm

from pyqed import Molecule
from pyqed.narg.qchem import NARG, NARGOpt, NARGSCF
from pyqed.narg.qchem.orbopt import (
    orbital_rotation_pairs,
    pack_orbital_pairs,
    unpack_orbital_pairs,
)


def test_active_active_orbital_pairs_are_explicit():
    pairs = orbital_rotation_pairs("active_active", ncore=2, ncas=3, nmo=8)

    assert pairs == [(2, 3), (2, 4), (3, 4)]


def test_casscf_orbital_pairs_exclude_active_active():
    pairs = orbital_rotation_pairs("casscf", ncore=1, ncas=3, nmo=6)
    dmrg_pairs = orbital_rotation_pairs("dmrgscf", ncore=1, ncas=3, nmo=6)

    assert pairs == dmrg_pairs
    assert (1, 2) not in pairs
    assert (0, 1) in pairs
    assert (1, 4) in pairs


def test_short_narg_orbital_optimizer_names_are_public():
    assert issubclass(NARGSCF, NARGOpt)


def test_nargscf_accepts_second_order_optimizer_alias():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="second_order",
        max_cycle=0,
    )

    assert mc.optimizer == "AH"


def test_nargscf_accepts_constrained_optimizer_alias():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="constrained",
        max_cycle=0,
    )

    assert mc.optimizer == "CONSTRAINED"
    assert mc.constrained_method == "L-BFGS-B"


def test_nargscf_accepts_lbfgs_trust_optimizer_alias():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="lbfgs-tr",
        lbfgs_trust_region=False,
        max_cycle=0,
    )

    assert mc.optimizer == "LBFGS"
    assert mc.lbfgs_trust_region is True


def test_nargscf_accepts_recursive_macro_optimizer_alias():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    default = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="recursive-lbfgs",
        max_cycle=0,
    )
    explicit_step = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="recursive",
        max_step=0.02,
        max_cycle=0,
    )

    assert default.optimizer == "LBFGS"
    assert default.gradient == "recursive"
    assert default._use_recursive_gradient() is True
    assert default.lbfgs_trust_region is True
    assert default.max_step == 0.01
    assert explicit_step.optimizer == "LBFGS"
    assert explicit_step.gradient == "recursive"
    assert explicit_step._use_recursive_gradient() is True
    assert explicit_step.max_step == 0.02


def test_nargscf_gradient_mode_selection():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    auto_lbfgs = NARGSCF(DummyMF(), ncas=2, nelecas=2, max_cycle=0)
    auto_ah = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="ah",
        ah_hessian="recursive_response",
        max_cycle=0,
    )
    recursive_lbfgs = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="lbfgs",
        gradient="recursive_response",
        max_cycle=0,
    )
    rdm_ah = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="ah",
        ah_hessian="recursive_response",
        gradient="rdm",
        max_cycle=0,
    )

    assert auto_lbfgs.gradient == "auto"
    assert auto_lbfgs._use_recursive_gradient() is False
    assert auto_ah._use_recursive_gradient() is True
    assert recursive_lbfgs.gradient == "recursive"
    assert recursive_lbfgs._use_recursive_gradient() is True
    assert rdm_ah._use_recursive_gradient() is False
    with pytest.raises(ValueError):
        NARGSCF(DummyMF(), ncas=2, nelecas=2, gradient="bogus", max_cycle=0)


def test_nargscf_rejects_mo_coeff_from_different_overlap():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

        def get_ovlp(self):
            return np.eye(4)

    mc = NARGSCF(DummyMF(), ncas=2, nelecas=2, max_cycle=0)
    bad_mo = np.eye(4)
    bad_mo[0, 0] = 1.1

    with pytest.raises(ValueError, match="not orthonormal"):
        mc.run(mo_coeff=bad_mo)


def test_nargscf_recursive_response_uses_true_gradient_fd_path():
    class DummyMF:
        nelec = (2, 1)
        mo_coeff = np.eye(3)

    class DummySolver:
        h1e = np.zeros((3, 3))

    default = NARGSCF(
        DummyMF(),
        ncas=3,
        nelecas=(2, 1),
        optimizer="ah",
        ah_hessian="recursive_response",
        max_cycle=0,
    )
    enabled = NARGSCF(
        DummyMF(),
        ncas=3,
        nelecas=(2, 1),
        optimizer="ah",
        ah_hessian="recursive_response",
        ah_recursive_response_blocks=True,
        max_cycle=0,
    )

    assert default._use_recursive_response_blocks(DummySolver()) is False
    assert enabled._use_recursive_response_blocks(DummySolver()) is False
    assert default._recursive_response_disabled_reason(DummySolver()) == (
        "recursive_gradient_fd"
    )


def test_nargscf_defaults_to_casscf_rotation_space():
    class DummyMF:
        nelec = (3, 3)
        mo_coeff = np.eye(6)

    opt = NARGSCF(DummyMF(), ncas=3, nelecas=4, max_cycle=0)
    expected = orbital_rotation_pairs(
        "casscf",
        ncore=opt.ncore,
        ncas=opt.ncas,
        nmo=opt.nmo,
    )

    assert opt.rotation_space == "casscf"
    assert opt._ordered_pairs() == expected
    assert (1, 2) not in expected
    assert (0, 1) in expected
    assert (1, 4) in expected


def test_nargscf_keeps_explicit_full_rotation_space():
    class DummyMF:
        nelec = (3, 3)
        mo_coeff = np.eye(6)

    opt = NARGSCF(
        DummyMF(),
        ncas=3,
        nelecas=4,
        rotation_space="full",
        max_cycle=0,
    )
    expected = orbital_rotation_pairs(
        "full",
        ncore=opt.ncore,
        ncas=opt.ncas,
        nmo=opt.nmo,
    )

    assert opt._ordered_pairs() == expected
    assert (1, 2) in expected
    assert (0, 1) in expected
    assert (1, 4) in expected


@pytest.mark.parametrize("old_space", ["narg", "casscf_plus_active", "full_narg"])
def test_old_rotation_space_names_are_removed(old_space):
    with pytest.raises(ValueError):
        orbital_rotation_pairs(old_space, ncore=1, ncas=2, nmo=5)


def test_orbital_pair_packers_include_active_active_pairs():
    pairs = [(0, 2), (2, 3)]
    grad = np.zeros((4, 4))
    grad[0, 2] = -0.25
    grad[2, 3] = 0.75

    vec = pack_orbital_pairs(grad, pairs)
    kappa = unpack_orbital_pairs(vec, pairs, nmo=4)

    np.testing.assert_allclose(vec, [-0.25, 0.75])
    np.testing.assert_allclose(kappa[0, 2], -0.25)
    np.testing.assert_allclose(kappa[2, 0], 0.25)
    np.testing.assert_allclose(kappa[2, 3], 0.75)
    np.testing.assert_allclose(kappa[3, 2], -0.75)


def test_narg_orbital_optimizer_lowers_core_active_distortion():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    kappa = np.zeros_like(mf.mo_coeff)
    kappa[0, 1] = 0.15
    kappa[1, 0] = -0.15
    mo0 = mf.mo_coeff @ expm(kappa)

    base = NARG(
        mf,
        symmetry="su2",
        ncas=2,
        nelecas=2,
        mo_coeff=mo0,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
    ).run()
    opt = NARGOpt(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        rotation_space=[(0, 1)],
        max_cycle=1,
        initial_step=0.15,
    ).run(mo_coeff=mo0)

    assert opt.e_tot[0] <= base.e_tot[0] + 1.0e-12
    assert opt.history[0]["accepted_pairs"] == 1
    np.testing.assert_allclose(opt.e_tot[0], -7.862128847738409, atol=1.0e-10)


def test_nargscf_rdm_gradient_lowers_core_active_distortion():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    kappa = np.zeros_like(mf.mo_coeff)
    kappa[0, 1] = 0.15
    kappa[1, 0] = -0.15
    mo0 = mf.mo_coeff @ expm(kappa)

    base = NARG(
        mf,
        symmetry="su2",
        ncas=2,
        nelecas=2,
        mo_coeff=mo0,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
    ).run()
    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        max_cycle=1,
        max_step=0.05,
    ).run(mo_coeff=mo0)

    assert mc.e_tot[0] < base.e_tot[0] - 1.0e-6
    assert mc.history[0]["accepted"] is True
    assert mc.history[0]["gradient_max"] > 0.0
    assert mc.history[0]["optimizer"] in {"DIAG", "LBFGS", "STEEPEST"}
    assert mc.history[0]["accepted_step_max"] > 0.0
    assert mc.history[0]["converged"] is False
    assert mc.convergence_reason == "max_cycle"
    assert mc.history[0]["pair_count"] == len(
        orbital_rotation_pairs(
            "casscf",
            ncore=mc.ncore,
            ncas=mc.ncas,
            nmo=mc.nmo,
        )
    )


def test_nargscf_converges_on_gradient_tolerance():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = mol.RHF().run()

    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        target_j2=0,
        su2_backend="python",
        max_cycle=4,
        conv_tol_grad=1.0,
    ).run()

    assert mc.converged is True
    assert mc.convergence_reason == "gradient"
    assert len(mc.history) == 1
    assert mc.history[0]["converged"] is True
    assert mc.history[0]["convergence_reason"] == "gradient"
    assert mc.history[0]["accepted"] is False


def test_nargscf_retries_with_smaller_trust_radius_after_rejection():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    class DummySolver:
        e_tot = np.array([0.0])

    solver = DummySolver()
    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        D=4,
        optimizer="diag",
        max_cycle=1,
        max_step=0.04,
        retry_on_rejection=True,
        max_rejection_retries=2,
        rejection_shrink=0.25,
    )
    calls = []

    mc._evaluate = lambda mo_coeff: (0.0, solver)

    def fake_evaluate_with_gradient(mo_coeff, *, pairs, energy=None, solver=None):
        fock = np.diag(np.arange(4.0))
        grad_vec = np.ones(len(pairs), dtype=float) * 0.1
        return float(0.0 if energy is None else energy), solver, fock, None, grad_vec

    def fake_gradient_step(grad_vec, fock, pairs, lbfgs_s, lbfgs_y, *, max_step=None):
        step = np.zeros_like(grad_vec)
        step[0] = float(max_step)
        return step, "LBFGS" if lbfgs_s else "DIAG"

    def fake_line_search(mo_coeff, energy, step_vec, pairs):
        calls.append(float(step_vec[0]))
        if len(calls) == 1:
            return False, mo_coeff, energy, 0.0, None
        return True, mo_coeff.copy(), energy - 1.0e-4, 1.0, solver

    mc._evaluate_with_gradient = fake_evaluate_with_gradient
    mc._gradient_step = fake_gradient_step
    mc._gradient_line_search = fake_line_search

    mc.run()

    assert mc.e_tot[0] == 0.0
    assert calls == [0.04, 0.01]
    assert mc.convergence_reason == "max_cycle"
    assert mc.history[0]["accepted"] is True
    assert mc.history[0]["energy_initial"] == 0.0
    np.testing.assert_allclose(mc.history[0]["energy"], -1.0e-4)
    np.testing.assert_allclose(mc.history[0]["trial_energy"], -1.0e-4)
    assert mc.history[0]["retry_count"] == 1
    assert mc.history[0]["trust_radius"] == 0.01
    assert [item["accepted"] for item in mc.history[0]["retry_history"]] == [
        False,
        True,
    ]


def test_nargscf_lbfgs_trust_region_rejects_poor_ratio_and_retries():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(2)

    class DummySolver:
        def __init__(self, energy):
            self.e_tot = np.array([energy])

    mc = NARGSCF(
        DummyMF(),
        ncas=1,
        nelecas=2,
        D=4,
        optimizer="lbfgs",
        max_cycle=1,
        max_step=0.04,
        accept_delta=0.0,
        lbfgs_trust_eta=0.10,
        retry_on_rejection=True,
        max_rejection_retries=2,
        rejection_shrink=0.25,
    )

    def fake_evaluate(mo_coeff):
        theta = float(np.arctan2(mo_coeff[0, 1], mo_coeff[0, 0]))
        if abs(theta) < 1.0e-12:
            energy = 0.0
        elif abs(theta) > 0.02:
            energy = -1.0e-4
        else:
            energy = -5.0e-4
        return energy, DummySolver(energy)

    def fake_evaluate_with_gradient(mo_coeff, *, pairs, energy=None, solver=None):
        del mo_coeff, pairs
        if energy is None:
            energy, solver = fake_evaluate(np.eye(2))
        fock = np.diag([0.0, 1.0])
        grad_vec = np.array([0.1])
        return float(energy), solver, fock, None, grad_vec

    mc._evaluate = fake_evaluate
    mc._evaluate_with_gradient = fake_evaluate_with_gradient

    mc.run()

    record = mc.history[0]
    assert record["trust_region"] is True
    assert record["accepted"] is True
    assert record["retry_count"] == 1
    np.testing.assert_allclose(record["trust_radius"], 0.01)
    np.testing.assert_allclose(record["energy"], -5.0e-4)
    assert record["retry_history"][0]["trust_ratio"] < mc.lbfgs_trust_eta
    assert record["retry_history"][1]["trust_ratio"] > mc.lbfgs_trust_eta
    np.testing.assert_allclose(mc.e_tot[0], -5.0e-4)


def test_nargscf_line_search_can_accept_opposite_rotation_direction():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(3)

    class DummySolver:
        e_tot = np.array([1.0])

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        D=4,
        max_cycle=0,
        accept_delta=0.0,
    )
    calls = []

    def fake_evaluate(mo_coeff):
        calls.append(float(mo_coeff[0, 1]))
        return 1.0 + float(mo_coeff[0, 1]), DummySolver()

    mc._evaluate = fake_evaluate
    accepted, _mo, energy, scale, _solver = mc._gradient_line_search(
        np.eye(3),
        1.0,
        np.array([0.1]),
        [(0, 1)],
    )

    assert accepted is True
    assert energy < 1.0
    assert scale < 0.0
    assert any(value > 0.0 for value in calls)
    assert any(value < 0.0 for value in calls)


def test_nargscf_constrained_step_minimizes_bounded_local_rotation():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(2)

    class DummySolver:
        e_tot = np.array([0.0])

    target = 0.03
    mc = NARGSCF(
        DummyMF(),
        ncas=1,
        nelecas=2,
        optimizer="constrained",
        constrained_maxiter=20,
        max_cycle=0,
    )
    pairs = [(0, 1)]
    mc._last_gradient_context = {
        "mo_coeff": np.eye(2),
        "energy": target * target,
        "solver": DummySolver(),
    }

    def fake_evaluate_with_gradient(mo_coeff, *, pairs, energy=None, solver=None):
        del energy, solver
        theta = float(np.arctan2(mo_coeff[0, 1], mo_coeff[0, 0]))
        value = (theta - target) ** 2
        grad_vec = np.array([2.0 * (theta - target)])
        return value, DummySolver(), np.eye(2), None, grad_vec

    mc._evaluate_with_gradient = fake_evaluate_with_gradient

    step_vec, direction = mc._gradient_step(
        np.array([-2.0 * target]),
        np.eye(2),
        pairs,
        [],
        [],
        max_step=0.05,
    )

    assert direction == "CONSTRAINED"
    assert abs(step_vec[0]) <= 0.05
    np.testing.assert_allclose(step_vec[0], target, atol=1.0e-5)
    assert mc._last_constrained_trial is not None
    assert mc._last_step_info["constrained_nfev"] > 0
    assert mc._last_step_info["constrained_energy_drop"] > 0.0


def test_nargscf_constrained_trial_updates_without_line_search():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(2)

    class DummySolver:
        e_tot = np.array([0.0])

    solver = DummySolver()
    mc = NARGSCF(
        DummyMF(),
        ncas=1,
        nelecas=2,
        optimizer="constrained",
        max_cycle=1,
    )
    mc._evaluate = lambda mo_coeff: (1.0, solver)

    def fake_evaluate_with_gradient(mo_coeff, *, pairs, energy=None, solver=None):
        return float(1.0 if energy is None else energy), solver, np.eye(2), None, np.array([0.1])

    def fake_gradient_step(grad_vec, fock, pairs, lbfgs_s, lbfgs_y, *, max_step=None):
        step = np.array([-0.02])
        mc._last_step_info = {"constrained_nfev": 2}
        mc._last_constrained_trial = {
            "mo_coeff": np.eye(2),
            "energy": 0.75,
            "solver": solver,
            "step_vec": step,
            "grad_vec": np.array([0.0]),
        }
        return step, "CONSTRAINED"

    mc._evaluate_with_gradient = fake_evaluate_with_gradient
    mc._gradient_step = fake_gradient_step
    mc._gradient_line_search = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("constrained optimizer should not use line search")
    )

    mc.run()

    assert mc.history[0]["accepted"] is True
    assert mc.history[0]["energy"] == 0.75
    assert mc.history[0]["accepted_scale"] == 1.0
    assert mc.history[0]["constrained_nfev"] == 2
    assert mc.convergence_reason == "max_cycle"


def test_nargscf_lbfgs_skips_noisy_secant_update():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="lbfgs",
        lbfgs_curvature_tol=1.0e-2,
        max_cycle=0,
    )
    s_history = []
    y_history = []

    skipped = mc._append_lbfgs_history(
        s_history,
        y_history,
        np.array([1.0, 0.0]),
        np.array([1.0e-4, 1.0]),
    )
    accepted = mc._append_lbfgs_history(
        s_history,
        y_history,
        np.array([1.0, 0.0]),
        np.array([1.0, 0.1]),
    )

    assert skipped["accepted"] is False
    assert accepted["accepted"] is True
    assert len(s_history) == 1
    np.testing.assert_allclose(s_history[0], [1.0, 0.0])


def test_nargscf_lbfgs_step_reports_selected_descent_candidate():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="lbfgs",
        max_cycle=0,
    )
    pairs = [(0, 1), (0, 2)]
    grad_vec = np.array([0.2, -0.1])
    fock = np.diag([0.0, 2.0, 5.0, 9.0])
    s_history = [np.array([-0.02, 0.01])]
    y_history = [np.array([-0.08, 0.05])]

    step_vec, direction = mc._gradient_step(
        grad_vec,
        fock,
        pairs,
        s_history,
        y_history,
        max_step=0.05,
    )

    assert direction in {"DIAG", "LBFGS", "STEEPEST"}
    assert np.dot(step_vec, grad_vec) < 0.0
    assert np.max(np.abs(step_vec)) <= 0.05
    assert mc._last_step_info["lbfgs_candidate_count"] >= 1
    assert mc._last_step_info["lbfgs_selected"] == direction


def test_nargscf_lbfgs_uses_orbital_denominator_preconditioner():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        optimizer="lbfgs",
        max_cycle=0,
    )
    pairs = [(0, 1), (0, 2)]
    grad_vec = np.array([2.0, 2.0])
    fock = np.diag([0.0, 2.0, 10.0, 12.0])

    step_vec, direction = mc._gradient_step(
        grad_vec,
        fock,
        pairs,
        [],
        [],
        max_step=1.0,
    )

    assert direction == "DIAG"
    np.testing.assert_allclose(step_vec, [-0.5, -0.1])
    assert mc._last_step_info["lbfgs_preconditioner"] == "orbital_denominator"
    np.testing.assert_allclose(mc._last_step_info["lbfgs_h0_min_value"], 0.05)
    np.testing.assert_allclose(mc._last_step_info["lbfgs_h0_max_value"], 0.25)
    np.testing.assert_allclose(mc._last_step_info["lbfgs_hess_min_value"], 4.0)
    np.testing.assert_allclose(mc._last_step_info["lbfgs_hess_max_value"], 20.0)


def test_nargscf_ah_step_uses_dense_hessian_action():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(4)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        D=4,
        optimizer="ah",
        ah_dense_threshold=10,
        max_cycle=0,
    )
    pairs = [(0, 1), (0, 2)]
    grad_vec = np.array([0.2, -0.1])
    fock = np.diag([0.0, 2.0, 4.0, 6.0])
    hess = np.array([[4.0, 1.0], [1.0, 3.0]])
    calls = []

    def fake_hessian_action(context, pairs_arg, vec):
        assert context == {"sentinel": True}
        assert pairs_arg == pairs
        calls.append(np.asarray(vec, dtype=float).copy())
        return hess @ vec

    mc._last_gradient_context = {"sentinel": True}
    mc._pair_hessian_action = fake_hessian_action

    step_vec, direction = mc._gradient_step(
        grad_vec,
        fock,
        pairs,
        [],
        [],
        max_step=0.5,
    )

    assert direction == "AH-DENSE"
    assert len(calls) == len(pairs)
    assert np.dot(step_vec, grad_vec) < 0.0
    assert mc._last_step_info["ah_solver"] == "dense"
    assert mc._last_step_info["ah_converged"] is True
    assert mc._last_step_info["ah_subspace_dim"] == len(pairs)


def test_nargscf_ah_davidson_is_bounded_by_default():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(8)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        D=4,
        optimizer="ah",
        ah_dense_threshold=0,
        max_cycle=0,
    )
    pairs = [(0, i) for i in range(1, 7)]
    grad_vec = np.linspace(0.05, 0.2, len(pairs))
    fock = np.diag(np.linspace(0.0, 4.0, 8))
    calls = []

    def fake_hessian_action(context, pairs_arg, vec):
        assert context == {"sentinel": True}
        assert pairs_arg == pairs
        calls.append(np.asarray(vec, dtype=float).copy())
        return 2.0 * np.asarray(vec, dtype=float)

    mc._last_gradient_context = {"sentinel": True}
    mc._pair_hessian_action = fake_hessian_action

    step_vec, direction = mc._gradient_step(
        grad_vec,
        fock,
        pairs,
        [],
        [],
        max_step=0.02,
    )

    assert direction == "AH-DAVIDSON"
    assert mc.ah_max_cycle == 1
    assert mc.ah_max_subspace == 4
    assert np.dot(step_vec, grad_vec) < 0.0
    assert mc._last_step_info["ah_solver"] == "davidson"
    assert mc._last_step_info["ah_iterations"] <= 1
    assert mc._last_step_info["ah_matvec_count"] == len(calls)
    assert mc._last_step_info["ah_matvec_count"] <= 3


def test_nargscf_relaxed_fd_hessian_action_differentiates_relaxed_gradient():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(2)

    mc = NARGSCF(
        DummyMF(),
        ncas=1,
        nelecas=2,
        D=4,
        optimizer="ah",
        ah_hessian="relaxed_fd",
        ah_fd_step=1.0e-4,
        max_cycle=0,
    )
    pairs = [(0, 1)]
    context = {"mo_coeff": np.eye(2)}
    calls = []

    def fake_relaxed_gradient_at_mo(mo_coeff, pairs_arg):
        assert pairs_arg == pairs
        calls.append(np.asarray(mo_coeff, dtype=float).copy())
        theta = np.arcsin(float(np.asarray(mo_coeff)[0, 1]))
        return np.array([3.0 * theta], dtype=float)

    mc._relaxed_gradient_at_mo = fake_relaxed_gradient_at_mo

    action = mc._pair_hessian_action(context, pairs, np.array([2.0]))

    np.testing.assert_allclose(action, [6.0], atol=1.0e-10)
    assert len(calls) == 2
    assert context["_relaxed_fd_evaluations"] == 2


def test_nargscf_ah_relaxed_fd_reports_extra_gradient_evaluations():
    class DummyMF:
        nelec = (1, 1)
        mo_coeff = np.eye(8)

    mc = NARGSCF(
        DummyMF(),
        ncas=2,
        nelecas=2,
        D=4,
        optimizer="ah",
        ah_hessian="relaxed_fd",
        ah_fd_step=1.0e-4,
        ah_dense_threshold=0,
        max_cycle=0,
    )
    pairs = [(0, i) for i in range(1, 7)]
    grad_vec = np.linspace(0.05, 0.2, len(pairs))
    fock = np.diag(np.linspace(0.0, 4.0, 8))
    calls = []

    def fake_relaxed_gradient_at_mo(mo_coeff, pairs_arg):
        assert pairs_arg == pairs
        calls.append(np.asarray(mo_coeff, dtype=float).copy())
        return 2.0 * pack_orbital_pairs(mo_coeff - mo_coeff.T, pairs)

    mc._last_gradient_context = {"mo_coeff": np.eye(8)}
    mc._relaxed_gradient_at_mo = fake_relaxed_gradient_at_mo

    step_vec, direction = mc._gradient_step(
        grad_vec,
        fock,
        pairs,
        [],
        [],
        max_step=0.02,
    )

    assert direction == "AH-DAVIDSON"
    assert np.dot(step_vec, grad_vec) < 0.0
    assert mc._last_step_info["ah_hessian"] == "relaxed_fd"
    assert mc._last_step_info["ah_solver"] == "davidson"
    assert mc._last_step_info["ah_matvec_count"] <= 3
    assert len(calls) % 2 == 0
    assert mc._last_step_info["ah_relaxed_fd_evaluations"] == len(calls)
