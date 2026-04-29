import numpy as np
import pytest
from scipy.linalg import expm
from types import SimpleNamespace

from pyqed.qchem import (
    CASSCF,
    COCASCI,
    FirstOrderCASSCF,
    Molecule,
    SecondOrderCASSCF,
)
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.orbopt import (
    davidson_augmented_hessian_direction,
    shifted_hessian_trust_step,
)


def _distorted_lih_reference(angle=0.12):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[1, 3] = angle
    kappa[3, 1] = -angle
    return mf, mf.mo_coeff @ expm(kappa)


def _distorted_ethylene44_reference(angle=0.08):
    atom = [
        ["C", 0.00000000, 0.00000000, 0.66796400],
        ["H", 0.92288300, 0.00000000, 1.24294900],
        ["H", -0.92288300, 0.00000000, 1.24294900],
        ["C", 0.00000000, 0.00000000, -0.66796400],
        ["H", 0.54030916, 0.92288300, -0.86462045],
        ["H", 0.54030916, -0.92288300, -0.86462045],
    ]
    mol = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[6, 10] = angle
    kappa[10, 6] = -angle
    return mf, mf.mo_coeff @ expm(kappa)


def test_casscf_and_cocasci_are_distinct_public_apis():
    assert CASSCF is not COCASCI
    assert CASSCF is SecondOrderCASSCF
    assert FirstOrderCASSCF is not CASSCF
    assert CASSCF.__name__ == "SecondOrderCASSCF"
    assert FirstOrderCASSCF.__name__ == "FirstOrderCASSCF"
    assert SecondOrderCASSCF.__name__ == "SecondOrderCASSCF"
    assert COCASCI.__name__ in {"COCAS", "COCASCI"}


def test_public_second_order_casscf_accepts_max_cycles_alias():
    mf = SimpleNamespace(mol=SimpleNamespace(nao=4), nmo=4)

    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycles=40)

    assert mc.max_cycle == 40
    assert mc.max_cycles == 40
    assert mc.coupling == "full"


def test_public_second_order_casscf_rejects_conflicting_cycle_aliases():
    mf = SimpleNamespace(mol=SimpleNamespace(nao=4), nmo=4)

    with pytest.raises(ValueError, match="conflicting values"):
        CASSCF(mf, ncas=2, nelecas=2, max_cycle=20, max_cycles=40)


def test_second_order_wmk_orbital_update_is_unitary_and_second_order():
    mf = SimpleNamespace(mol=SimpleNamespace(nao=4), nmo=4)
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        orbital_parameterization="wmk",
    )

    kappa = np.array(
        [
            [0.0, 0.03, -0.02, 0.01],
            [-0.03, 0.0, 0.04, -0.01],
            [0.02, -0.04, 0.0, 0.02],
            [-0.01, 0.01, -0.02, 0.0],
        ]
    )
    unitary = mc._orbital_unitary(kappa)
    second_order = np.eye(4) + kappa + 0.5 * (kappa @ kappa)

    np.testing.assert_allclose(unitary.T @ unitary, np.eye(4), atol=1.0e-12)
    np.testing.assert_allclose(unitary, second_order, atol=2.0e-4)


def test_second_order_orbital_pspace_prioritizes_critical_rotations():
    mf = SimpleNamespace(mol=SimpleNamespace(nao=5), nmo=5)
    mc = SecondOrderCASSCF(mf, ncas=2, nelecas=2, ah_pspace_size=3)

    grad = np.array([0.01, 0.20, 0.03, 0.40, 0.02])
    hdiag = np.array([0.5, -0.1, 0.03, 2.0, -0.2])
    guess = mc._orbital_pspace_guess(grad, hdiag)
    selected = set(np.flatnonzero(np.any(np.abs(guess) > 0.5, axis=1)))

    assert {1, 4}.issubset(selected)
    assert len(selected) == 3


def test_davidson_augmented_hessian_can_return_diagnostics():
    grad = np.array([0.15, -0.05, 0.02])
    hdiag = np.array([0.8, 1.2, 1.5])

    step, info = davidson_augmented_hessian_direction(
        grad,
        hdiag,
        matvec=lambda vec: hdiag * vec,
        max_cycle=4,
        max_subspace=6,
        return_info=True,
    )

    assert step.shape == grad.shape
    assert np.dot(step, grad) < 0.0
    assert info["iterations"] >= 1
    assert info["subspace_dim"] >= 1
    assert np.isfinite(info["residual_norm"])
    assert info["used_fallback"] is False


def test_shifted_hessian_trust_step_respects_norm_radius():
    grad = np.array([1.0, -0.5, 0.25])
    hdiag = np.array([0.2, 0.3, 0.4])

    unconstrained, shift0 = shifted_hessian_trust_step(
        grad,
        hdiag,
        trust_radius=None,
        regularization=1.0e-6,
    )
    constrained, shift = shifted_hessian_trust_step(
        grad,
        hdiag,
        trust_radius=0.2,
        regularization=1.0e-6,
    )

    assert shift0 == 0.0
    assert np.linalg.norm(unconstrained) > 0.2
    assert shift > 0.0
    assert np.linalg.norm(constrained) <= 0.2 * (1.0 + 1.0e-10)
    assert np.dot(constrained, grad) < 0.0


def test_second_order_internal_preopt_lowers_core_active_distortion():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[0, 1] = 0.18
    kappa[1, 0] = -0.18
    mo_guess = mf.mo_coeff @ expm(kappa)
    initial = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        internal_preopt_steps=1,
        internal_preopt_max_step=0.1,
        internal_preopt_hessian="finite_difference",
    )
    mc.nstates = 1
    mc.state_id = 0
    mo_preopt, ci_guess = mc._internal_preopt(mo_guess, None, macro=1)
    final = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_preopt,
        method="direct_ci",
        ci0=ci_guess,
    )

    assert len(mc.internal_preopt_history) >= 1
    assert mc.internal_preopt_history[0]["accepted"] is True
    assert mc.internal_preopt_history[0]["hessian"] == "finite_difference"
    assert mc.internal_preopt_history[0]["space"] == "core_active"
    assert mc.internal_preopt_history[0]["hessian_dim"] >= 1
    assert final.e_tot[0] < initial.e_tot[0]


def test_second_order_internal_preopt_can_use_all_nonredundant_rotations():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        internal_preopt_steps=1,
        internal_preopt_hessian="diagonal",
        internal_preopt_space="nonredundant",
    )
    mc.nstates = 1
    mc.state_id = 0

    casci = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    full_mask = mc._internal_preopt_mask(casci.ncore, casci.ncas, mf.nmo)
    core_active_mask = mc._core_active_mask(casci.ncore, casci.ncas, mf.nmo)

    assert np.count_nonzero(full_mask) > np.count_nonzero(core_active_mask)

    mo_preopt, ci_guess = mc._internal_preopt(mf.mo_coeff, None, macro=1)

    assert mo_preopt.shape == mf.mo_coeff.shape
    assert ci_guess is not None
    assert len(mc.internal_preopt_history) >= 1
    assert mc.internal_preopt_history[0]["space"] == "nonredundant"
    assert mc.internal_preopt_history[0]["hessian_dim"] == int(np.count_nonzero(full_mask))


def test_second_order_full_internal_optimization_converges_internal_loop():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[0, 1] = 0.18
    kappa[1, 0] = -0.18
    mo_guess = mf.mo_coeff @ expm(kappa)
    initial = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        internal_preopt_steps=0,
        internal_optimization=True,
        internal_max_cycle=4,
        internal_conv_tol_grad=4.0e-2,
        internal_preopt_hessian="diagonal",
        internal_preopt_space="nonredundant",
    )
    mc.nstates = 1
    mc.state_id = 0
    mo_opt, ci_guess = mc._internal_preopt(mo_guess, None, macro=1)
    final = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_opt,
        method="direct_ci",
        ci0=ci_guess,
    )

    assert mc.internal_optimization_converged is True
    assert len(mc.internal_preopt_history) == 1
    assert mc.internal_preopt_history[0]["internal_optimization"] is True
    assert mc.internal_preopt_history[0]["internal_stop_reason"] == "gradient"
    assert mc.internal_preopt_history[0]["post_gradient_norm"] < 4.0e-2
    assert final.e_tot[0] < initial.e_tot[0]


def test_second_order_internal_optimization_can_use_davidson_solver():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[0, 1] = 0.18
    kappa[1, 0] = -0.18
    mo_guess = mf.mo_coeff @ expm(kappa)
    initial = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        internal_optimization=True,
        internal_max_cycle=2,
        internal_preopt_hessian="analytic",
        internal_preopt_solver="davidson",
        internal_preopt_space="nonredundant",
        ah_max_cycle=3,
        ah_max_subspace=5,
    )
    mc.nstates = 1
    mc.state_id = 0
    mo_opt, ci_guess = mc._internal_preopt(mo_guess, None, macro=1)
    final = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_opt,
        method="direct_ci",
        ci0=ci_guess,
    )

    assert len(mc.internal_preopt_history) >= 1
    record = mc.internal_preopt_history[0]
    assert record["hessian"] == "analytic"
    assert record["solver"] == "davidson"
    assert record["solver_iterations"] >= 1
    assert record["solver_subspace_dim"] >= 1
    assert np.isfinite(record["solver_residual_norm"])
    assert final.e_tot[0] < initial.e_tot[0]


@pytest.mark.parametrize(
    ("internal_hessian", "orbital_hessian"),
    [
        ("coupled", "analytic"),
        ("coupled_fd", "finite_difference"),
    ],
)
def test_second_order_internal_preopt_can_use_coupled_ci_response(
    internal_hessian,
    orbital_hessian,
):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[0, 1] = 0.18
    kappa[1, 0] = -0.18
    mo_guess = mf.mo_coeff @ expm(kappa)
    initial = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        internal_preopt_steps=1,
        internal_preopt_hessian=internal_hessian,
        internal_preopt_space="nonredundant",
        coupled_ci_roots=1,
        coupled_qspace_cycles=1,
    )
    mc.nstates = 1
    mc.state_id = 0
    mo_preopt, ci_guess = mc._internal_preopt(mo_guess, None, macro=1)
    final = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_preopt,
        method="direct_ci",
        ci0=ci_guess,
    )

    assert len(mc.internal_preopt_history) >= 1
    assert mc.internal_preopt_history[0]["hessian"] == internal_hessian
    assert mc.internal_preopt_history[0]["coupled_orbital_hessian"] == orbital_hessian
    assert mc.internal_preopt_history[0]["coupled_ci_dim"] >= 1
    if internal_hessian == "coupled_fd":
        assert mc.internal_preopt_history[0]["coupled_fallback_diagonal"] is True
    assert final.e_tot[0] < initial.e_tot[0]


def test_second_order_internal_preopt_guard_rejects_worse_preview():
    mf = SimpleNamespace(mol=SimpleNamespace(nao=2), nmo=2)
    mc = SecondOrderCASSCF(
        mf,
        ncas=1,
        nelecas=0,
        internal_preopt_guard_cycles=2,
    )
    record = {}
    before = np.eye(2)
    after = np.array([[0.9, 0.1], [-0.1, 0.9]])

    def preview(mo_coeff, guard_cycles):
        assert guard_cycles == 2
        return -1.0 if mo_coeff is before else -0.9

    mc._internal_preopt_preview_energy = preview

    assert mc._internal_preopt_guard_accepts(before, after, record) is False
    assert record["guard_cycles"] == 2
    assert record["guard_before_energy"] == -1.0
    assert record["guard_after_energy"] == -0.9
    assert record["guard_accepted"] is False


def test_second_order_pspace_reaches_lower_ethylene44_stationary_point():
    mf, mo_guess = _distorted_ethylene44_reference(angle=0.08)

    mc = SecondOrderCASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=25,
        max_micro_cycle=8,
        conv_tol=1.0e-8,
        conv_tol_grad=1.0e-5,
        conv_tol_grad_relaxed=1.0e-4,
        max_step=0.1,
        coupling="qn",
        auto_active_restarts=False,
    ).run(mo_coeff=mo_guess)

    assert mc.e_tot[0] < -76.97215
    assert mc.history[-1]["gradient_norm"] < 1.0e-5


def test_first_order_casscf_lih_lowers_initial_casci_energy():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[1, 3] = 0.2
    kappa[3, 1] = -0.2
    mo_guess = mf.mo_coeff @ expm(kappa)

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    mc = FirstOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=12,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
    ).run(mo_coeff=mo_guess)

    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1


def test_first_order_casscf_diis_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    mc = FirstOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=8,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        diis=True,
        diis_space=4,
        diis_start=2,
    ).run()

    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1


def test_first_order_casscf_lbfgs_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    mc = FirstOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=8,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        optimizer="LBFGS",
        optimizer_history=5,
        diis=False,
    ).run()

    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1


def test_first_order_casscf_ah_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    mc = FirstOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=8,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        step_size=0.25,
        max_step=0.1,
        optimizer="AH",
        diis=False,
    ).run()

    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1


def test_second_order_casscf_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[1, 3] = 0.2
    kappa[3, 1] = -0.2
    mo_guess = mf.mo_coeff @ expm(kappa)

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=10,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        coupling="qn",
    ).run(mo_coeff=mo_guess)

    assert mc.optimizer == "AH"
    assert mc.diis is False
    assert mc.coupling == "qn"
    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1
    assert len(mc.micro_history) >= 1
    stepped_records = [
        record for record in mc.micro_history if "ah_residual_norm" in record
    ]
    assert len(stepped_records) >= 1
    assert stepped_records[0]["ah_iterations"] >= 1
    assert stepped_records[0]["ah_subspace_dim"] >= 1
    assert stepped_records[0]["ah_pspace_dim"] >= 1
    assert stepped_records[0]["ah_trust_radius"] > 0.0
    assert stepped_records[0]["ah_trust_metric"] == "component"
    assert stepped_records[0]["ah_adaptive_trust"] is False
    assert "ah_diagonal_shift" in stepped_records[0]
    assert stepped_records[0]["ah_predicted_reduction"] > 0.0
    assert stepped_records[0]["ah_actual_reduction"] >= 0.0
    assert np.isfinite(stepped_records[0]["ah_ratio"])
    assert np.isfinite(stepped_records[0]["ah_residual_norm"])


def test_second_order_wmk_parameterization_path_runs_on_lih():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=10,
        max_micro_cycle=4,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        orbital_parameterization="wmk",
        coupling="qn",
    ).run(mo_coeff=mo_guess)

    assert mc.orbital_parameterization == "wmk"
    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.micro_history) >= 1
    stepped = [record for record in mc.micro_history if "ah_residual_norm" in record]
    assert len(stepped) >= 1
    assert stepped[0]["orbital_parameterization"] == "wmk"
    assert stepped[0]["orbital_hessian_model"] == "analytic_wmk_second_order"


def test_casscf_can_reorder_explicit_active_orbitals():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)

    mc = FirstOrderCASSCF(mf, ncas=2, nelecas=2)
    reordered = mc.reorder_mo_for_active_orbitals(mo_guess, active_orbitals=(1, 3))

    np.testing.assert_allclose(reordered[:, mc._default_ncore()], mo_guess[:, 1])
    np.testing.assert_allclose(reordered[:, mc._default_ncore() + 1], mo_guess[:, 3])

    result = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=4,
        max_micro_cycle=2,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
    ).run(mo_coeff=mo_guess, active_orbitals=(1, 3))

    assert np.isfinite(result.e_tot[0])


def test_second_order_casscf_accepts_uncoupled_path():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=1,
        max_micro_cycle=1,
        coupling="uncoupled",
    )

    assert mc.coupling == "uncoupled"


def test_second_order_casscf_relaxed_fd_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[1, 3] = 0.12
    kappa[3, 1] = -0.12
    mo_guess = mf.mo_coeff @ expm(kappa)

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=10,
        max_micro_cycle=4,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        coupling="relaxed_fd",
        coupled_fd_step=1.0e-4,
        ah_max_cycle=2,
        ah_max_subspace=4,
    ).run(mo_coeff=mo_guess)

    assert mc.coupling == "relaxed_fd"
    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.micro_history) >= 1


def test_second_order_casscf_partial_coupled_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[1, 3] = 0.12
    kappa[3, 1] = -0.12
    mo_guess = mf.mo_coeff @ expm(kappa)

    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    mc = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=10,
        max_micro_cycle=4,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        coupling="partial",
        coupled_ci_roots=1,
    ).run(mo_coeff=mo_guess)

    assert mc.coupling == "partial"
    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.micro_history) >= 1


def test_second_order_partial_tracks_relaxed_fd_on_lih():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)

    common_kwargs = dict(
        ncas=2,
        nelecas=2,
        max_cycle=10,
        max_micro_cycle=4,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        ah_max_cycle=2,
        ah_max_subspace=4,
    )
    partial = SecondOrderCASSCF(
        mf,
        coupling="partial",
        **common_kwargs,
    ).run(mo_coeff=mo_guess)
    relaxed_fd = SecondOrderCASSCF(
        mf,
        coupling="relaxed_fd",
        coupled_fd_step=1.0e-4,
        **common_kwargs,
    ).run(mo_coeff=mo_guess)

    assert np.isfinite(partial.e_tot[0])
    assert np.isfinite(relaxed_fd.e_tot[0])
    assert abs(partial.e_tot[0] - relaxed_fd.e_tot[0]) < 2.0e-3


def test_second_order_full_coupled_path_runs_on_lih():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)

    full = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=10,
        max_micro_cycle=4,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        coupling="full",
        ah_max_cycle=2,
        ah_max_subspace=4,
    ).run(mo_coeff=mo_guess)

    assert full.coupling == "full"
    assert np.isfinite(full.e_tot[0])
    assert len(full.micro_history) >= 1


def test_second_order_casscf_no_core_full_active_space_runs_on_h2():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    driver = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=3,
        max_micro_cycle=1,
        conv_tol=1.0e-8,
        conv_tol_grad=1.0e-5,
        conv_tol_grad_relaxed=1.0e-4,
        coupling="qn",
    ).run(active_orbitals=(0, 1))

    assert np.isfinite(driver.e_tot[0])
    np.testing.assert_allclose(driver.e_tot[0], mc0.e_tot[0], atol=1.0e-8)


def test_second_order_casscf_factorized_qn_matches_dense_on_lih():
    atom = "Li 0 0 0; H 0 0 1.6"
    mol_dense = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol_dense.build(driver="gbasis-pyscf")
    mol_factor = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol_factor.build(driver="gbasis-pyscf")

    mf_dense = mol_dense.RHF().run()
    mf_factor = mol_factor.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)
    common = dict(
        ncas=2,
        nelecas=2,
        max_cycle=8,
        max_micro_cycle=3,
        conv_tol=1.0e-6,
        conv_tol_grad=1.0e-4,
        conv_tol_grad_relaxed=1.0e-3,
        coupling="qn",
        auto_active_restarts=False,
    )

    dense = SecondOrderCASSCF(mf_dense, **common).run(use_cholesky=False)
    factor = SecondOrderCASSCF(mf_factor, **common).run(use_cholesky=True)

    assert not dense.use_cholesky_integrals
    assert factor.use_cholesky_integrals
    assert factor.casci.use_cholesky_integrals
    assert str(factor.casci.solver_backend).startswith("direct_ci_factor_conn")
    np.testing.assert_allclose(factor.e_tot[0], dense.e_tot[0], atol=1.0e-6)


def test_second_order_casscf_factorized_qn_avoids_dense_mo_eri(monkeypatch):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis-pyscf")

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    def fail_get_eri_mo(*args, **kwargs):
        raise AssertionError("dense MO ERIs should not be built in factorized CASSCF")

    monkeypatch.setattr(mf, "get_eri_mo", fail_get_eri_mo)

    factor = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=8,
        max_micro_cycle=3,
        conv_tol=1.0e-6,
        conv_tol_grad=1.0e-4,
        conv_tol_grad_relaxed=1.0e-3,
        coupling="qn",
        auto_active_restarts=False,
    ).run()

    assert np.isfinite(factor.e_tot[0])
    assert factor.use_cholesky_integrals
    assert factor.casci.use_cholesky_integrals


def test_second_order_casscf_factorized_full_matches_dense_on_lih():
    atom = "Li 0 0 0; H 0 0 1.6"
    mol_dense = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol_dense.build(driver="gbasis-pyscf")
    mol_factor = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol_factor.build(driver="gbasis-pyscf")

    mf_dense = mol_dense.RHF().run()
    mf_factor = mol_factor.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)
    common = dict(
        ncas=2,
        nelecas=2,
        max_cycle=8,
        max_micro_cycle=3,
        conv_tol=1.0e-6,
        conv_tol_grad=1.0e-4,
        conv_tol_grad_relaxed=1.0e-3,
        coupling="full",
        auto_active_restarts=False,
    )

    dense = SecondOrderCASSCF(mf_dense, **common).run(use_cholesky=False)
    factor = SecondOrderCASSCF(mf_factor, **common).run()

    assert factor.use_cholesky_integrals
    assert factor.coupling == "full"
    assert str(factor.casci.solver_backend).startswith("direct_ci_factor_conn")
    np.testing.assert_allclose(factor.e_tot[0], dense.e_tot[0], atol=1.0e-8)


def test_second_order_casscf_factorized_full_default_avoids_dense_mo_eri(monkeypatch):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis-pyscf")

    mf = mol.RHF().run(cholesky_jk=True, cholesky_tol=1.0e-10)

    def fail_get_eri_mo(*args, **kwargs):
        raise AssertionError("dense MO ERIs should not be built in factorized full CASSCF")

    monkeypatch.setattr(mf, "get_eri_mo", fail_get_eri_mo)

    factor = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=8,
        max_micro_cycle=3,
        conv_tol=1.0e-6,
        conv_tol_grad=1.0e-4,
        conv_tol_grad_relaxed=1.0e-3,
        auto_active_restarts=False,
    ).run()

    assert factor.coupling == "full"
    assert np.isfinite(factor.e_tot[0])
    assert factor.use_cholesky_integrals
    assert factor.casci.use_cholesky_integrals


def test_second_order_simultaneous_reduced_alias_uses_partial_path():
    mf, _ = _distorted_lih_reference(angle=0.12)

    driver = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        coupling="simultaneous_reduced",
    )

    assert driver.coupling == "partial"


def test_second_order_full_coupled_path_can_fallback_on_untrusted_model():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    full = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=3,
        max_micro_cycle=1,
        conv_tol=1.0e-2,
        conv_tol_grad=1.0e-2,
        conv_tol_grad_relaxed=1.0e-1,
        coupling="full",
        coupled_accept_min_ratio=10.0,
        coupled_fallback=True,
        ah_max_cycle=2,
        ah_max_subspace=4,
    ).run(mo_coeff=mo_guess)

    records = [
        record
        for record in full.micro_history
        if record.get("coupled_step_attempted", False)
    ]
    assert records
    assert any(record["coupled_fallback_used"] is True for record in records)
    assert all(record["coupled_joint_trust_region"] is True for record in records)
    assert all(record["ah_actual_reduction"] > 0.0 for record in records)
    assert np.isfinite(full.e_tot[0])
    assert full.e_tot[0] < mc0.e_tot[0] - 1.0e-6


def test_second_order_joint_ci_orbital_trial_matches_casci_energy():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, coupling="simultaneous")
    driver.nstates = 1
    driver.state_id = 0
    driver.mo_coeff_ref = mo_guess
    h1_mo = mf.get_hcore_mo(mo_guess)
    eri_mo = mf.get_eri_mo(mo_guess, notation="chem")
    mc = driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    unitary = np.eye(mf.nmo)
    kappa = np.zeros((mf.nmo, mf.nmo))

    _, joint_energy, trial_mc = driver._joint_ci_orbital_trial(
        h1_mo,
        eri_mo,
        unitary,
        kappa,
        mc,
        mc.ci[0],
        mc.ci[0],
        scale=1.0,
    )

    assert trial_mc.ci[0].shape == mc.ci[0].shape
    np.testing.assert_allclose(joint_energy, mc.e_tot[0], atol=1.0e-10)


def test_second_order_joint_trust_region_scales_newton_model_terms():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    driver = SecondOrderCASSCF(mf, ncas=2, nelecas=2, coupling="simultaneous")
    driver.nstates = 1
    driver.state_id = 0
    driver.mo_coeff_ref = mo_guess
    h1_mo = mf.get_hcore_mo(mo_guess)
    eri_mo = mf.get_eri_mo(mo_guess, notation="chem")
    mc = driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    kappa = np.zeros((mf.nmo, mf.nmo))

    accepted, joint = driver._joint_trust_region_micro_search(
        h1_mo,
        eri_mo,
        np.eye(mf.nmo),
        kappa,
        mc.e_tot[0],
        mc,
        mc.ci[0],
        model_reduction=99.0,
        model_linear=-4.0,
        model_quadratic=2.0,
    )

    assert accepted is False
    assert joint[5] == pytest.approx(3.0)


def test_second_order_full_ci_orbital_response_matches_finite_difference():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    mc_driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4, coupling="full")
    mc_driver.nstates = 1
    mc_driver.state_id = 0
    mc_driver.mo_coeff_ref = mo_guess

    h1_mo = mf.get_hcore_mo(mo_guess)
    eri_mo = mf.get_eri_mo(mo_guess, notation="chem")
    casci = mc_driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    c0 = np.asarray(casci.ci[0])

    from pyqed.qchem.mcscf.orbopt import pack_nonredundant, unpack_nonredundant

    nvar = pack_nonredundant(
        np.zeros((mf.nmo, mf.nmo)),
        casci.ncore,
        casci.ncas,
        mf.nmo,
    ).size
    step = np.linspace(0.2, 1.0, nvar)
    step /= np.linalg.norm(step)
    analytic = mc_driver._ci_gradient_from_orbital_response(
        casci,
        h1_mo,
        eri_mo,
        c0,
        step,
    )

    eps = 1.0e-5
    kappa = unpack_nonredundant(step, casci.ncore, casci.ncas, mf.nmo)
    h1_plus, eri_plus = mc_driver._transform_frozen_integrals(
        h1_mo,
        eri_mo,
        expm(eps * kappa),
    )
    h1_minus, eri_minus = mc_driver._transform_frozen_integrals(
        h1_mo,
        eri_mo,
        expm(-eps * kappa),
    )
    sigma_plus = mc_driver._make_integral_sigma_casci(casci, h1_plus, eri_plus).ci_sigma(c0)
    sigma_minus = mc_driver._make_integral_sigma_casci(casci, h1_minus, eri_minus).ci_sigma(c0)
    finite_difference = mc_driver._project_ci_response(
        (sigma_plus - sigma_minus) / (2.0 * eps),
        [c0],
    )

    np.testing.assert_allclose(analytic, finite_difference, atol=1.0e-8, rtol=1.0e-8)


def test_second_order_full_coupling_blocks_are_symmetric():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    mc_driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4, coupling="full")
    mc_driver.nstates = 1
    mc_driver.state_id = 0
    mc_driver.mo_coeff_ref = mo_guess

    h1_mo = mf.get_hcore_mo(mo_guess)
    eri_mo = mf.get_eri_mo(mo_guess, notation="chem")
    casci = mc_driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    c0 = np.asarray(casci.ci[0])

    from pyqed.qchem.mcscf.orbopt import pack_nonredundant

    nvar = pack_nonredundant(
        np.zeros((mf.nmo, mf.nmo)),
        casci.ncore,
        casci.ncas,
        mf.nmo,
    ).size
    orb_step = np.linspace(0.1, 1.0, nvar)
    orb_step /= np.linalg.norm(orb_step)

    ci_step = np.eye(c0.size)[:, 0] + 0.3 * np.eye(c0.size)[:, min(2, c0.size - 1)]
    ci_step = mc_driver._project_ci_response(ci_step, [c0])
    ci_step /= np.linalg.norm(ci_step)

    hco_orb = mc_driver._ci_gradient_from_orbital_response(
        casci,
        h1_mo,
        eri_mo,
        c0,
        orb_step,
    )
    hoc_ci = mc_driver._orbital_gradient_from_ci_response_adjoint(
        casci,
        h1_mo,
        eri_mo,
        c0,
        ci_step,
    )

    np.testing.assert_allclose(
        np.dot(ci_step, hco_orb),
        np.dot(orb_step, hoc_ci),
        atol=1.0e-8,
        rtol=1.0e-8,
    )


def test_second_order_exact_orbital_gradient_matches_energy_finite_difference():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    mc_driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4, coupling="partial")
    mc_driver.nstates = 1
    mc_driver.state_id = 0
    mc_driver.mo_coeff_ref = mo_guess

    h1_mo = mf.get_hcore_mo(mo_guess)
    eri_mo = mf.get_eri_mo(mo_guess, notation="chem")
    casci = mc_driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    grad_vec = mc_driver._exact_orbital_gradient_vector(
        casci,
        h1_mo,
        eri_mo,
        casci.ci[0],
    )

    from pyqed.qchem.mcscf.orbopt import unpack_nonredundant

    step = np.linspace(0.2, 1.0, grad_vec.size)
    step /= np.linalg.norm(step)
    eps = 1.0e-5
    kappa = unpack_nonredundant(step, casci.ncore, casci.ncas, mf.nmo)

    e_plus = CASCI(mf, ncas=4, nelecas=4).run(
        nstates=1,
        mo_coeff=mo_guess @ expm(eps * kappa),
        method="direct_ci",
    ).e_tot[0]
    e_minus = CASCI(mf, ncas=4, nelecas=4).run(
        nstates=1,
        mo_coeff=mo_guess @ expm(-eps * kappa),
        method="direct_ci",
    ).e_tot[0]
    finite_difference = (e_plus - e_minus) / (2.0 * eps)

    np.testing.assert_allclose(np.dot(grad_vec, step), finite_difference, atol=1.0e-7)
