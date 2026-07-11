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
from pyqed.qchem.mcscf import direct_ci as direct_ci_module
from pyqed.qchem.mcscf.casci import _get_mf_cholesky_factors, transform_eri_factors_to_mo_pair
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.orbopt import (
    davidson_augmented_hessian_direction,
    pack_nonredundant,
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


def test_second_order_reuses_direct_ci_connectivity(monkeypatch):
    """Repeated frozen-integral CASCI solves should reuse determinant connectivity."""
    mf, mo_guess = _distorted_lih_reference()
    original = direct_ci_module.build_direct_connectivity
    calls = {"count": 0}

    def counted(binary):
        calls["count"] += 1
        return original(binary)

    monkeypatch.setattr(direct_ci_module, "build_direct_connectivity", counted)

    with pytest.raises(RuntimeError, match="Max macro steps reached"):
        SecondOrderCASSCF(
            mf,
            ncas=4,
            nelecas=4,
            max_cycle=3,
            max_micro_cycle=2,
            conv_tol=1.0e-12,
            conv_tol_grad=1.0e-12,
            conv_tol_grad_relaxed=1.0e-12,
            coupling="full",
            auto_active_restarts=False,
        ).run(mo_coeff=mo_guess)

    assert calls["count"] == 1


def test_second_order_batched_derivative_sigma_matches_direct_loop():
    mf, mo_guess = _distorted_lih_reference()
    mc = CASCI(mf, ncas=4, nelecas=4).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    so = SecondOrderCASSCF(mf, ncas=4, nelecas=4)
    h1_mo, eri_mo = so._get_integrals(mo_guess)
    ci = mc.ci[0]

    dh1_basis, deri_basis = so._active_integral_derivative_basis(mc, h1_mo, eri_mo)
    reference = np.asarray(
        [
            so._make_active_sigma_casci(mc, dh1_basis[i], deri_basis[i]).ci_sigma(ci)
            for i in range(dh1_basis.shape[0])
        ]
    )
    batched = so._derivative_sigma_basis(mc, h1_mo, eri_mo, ci)

    np.testing.assert_allclose(batched, reference, atol=1.0e-12)


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
    assert mc.internal_preopt_history[0]["hessian_dim"] >= 1
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


def test_second_order_full_state_average_tracks_partial_on_lih():
    mf, mo_guess = _distorted_lih_reference(angle=0.12)
    common = dict(
        ncas=4,
        nelecas=4,
        max_cycle=8,
        max_micro_cycle=3,
        conv_tol=1.0e-3,
        conv_tol_grad=1.0e-4,
        conv_tol_grad_relaxed=1.0e-3,
        ah_max_cycle=2,
        ah_max_subspace=5,
        auto_active_restarts=False,
    )

    partial = SecondOrderCASSCF(mf, coupling="partial", **common).state_average(
        [0.5, 0.5]
    ).run(nstates=2, mo_coeff=mo_guess)
    full = SecondOrderCASSCF(mf, coupling="full", **common).state_average(
        [0.5, 0.5]
    ).run(nstates=2, mo_coeff=mo_guess)

    assert full.coupling == "full"
    assert full.converged
    np.testing.assert_allclose(full.e_tot, partial.e_tot, atol=1.0e-7)


def test_second_order_frozen_casci_can_use_cholesky_pair_factors():
    mf, mo_guess = _distorted_lih_reference(angle=0.08)
    mf.cholesky_jk = True
    mf.cholesky_tol = 1.0e-10
    driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4)
    driver.nstates = 1
    driver.state_id = 0
    h1_mo, eri_mo = driver._get_integrals(mo_guess)
    pair_factors = transform_eri_factors_to_mo_pair(
        _get_mf_cholesky_factors(mf),
        mo_guess,
        mo_guess,
    )

    dense = driver._make_integral_casci(h1_mo, eri_mo, mo_guess, 1)
    factored = driver._make_integral_casci(
        h1_mo,
        eri_mo,
        mo_guess,
        1,
        pair_factors_mo=pair_factors,
    )

    assert factored.solver_backend == "direct_ci_factor_conn"
    np.testing.assert_allclose(factored.e_tot, dense.e_tot, atol=1.0e-8)


def test_second_order_pair_factor_full_mo_eri_matches_dense_rotation():
    mf, mo_guess = _distorted_lih_reference(angle=0.08)
    mf.cholesky_jk = True
    mf.cholesky_tol = 1.0e-10
    driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4)
    h1_ref, eri_ref = driver._get_integrals(mo_guess)
    pair_ref = transform_eri_factors_to_mo_pair(
        _get_mf_cholesky_factors(mf),
        mo_guess,
        mo_guess,
    )

    kappa = np.zeros((mf.mo_coeff.shape[1], mf.mo_coeff.shape[1]))
    kappa[0, 2] = 0.03
    kappa[2, 0] = -0.03
    U = expm(kappa)
    h1_dense, eri_dense = driver._transform_frozen_integrals(h1_ref, eri_ref, U)
    h1_factored, eri_factored, pair_cur = driver._current_frozen_integrals(
        h1_ref,
        None,
        pair_ref,
        U,
    )

    assert pair_cur is not None
    np.testing.assert_allclose(h1_factored, h1_dense, atol=1.0e-12)
    np.testing.assert_allclose(eri_factored, eri_dense, atol=1.0e-8)


def test_second_order_factorized_orbital_hessian_matches_dense():
    mf, mo_guess = _distorted_lih_reference(angle=0.08)
    mf.cholesky_jk = True
    mf.cholesky_tol = 1.0e-10
    driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4)
    h1_mo = mf.get_hcore_mo(mo_guess)
    pair_factors = transform_eri_factors_to_mo_pair(
        _get_mf_cholesky_factors(mf),
        mo_guess,
        mo_guess,
    )
    eri_mo = driver._assemble_eri_from_pair_factors(pair_factors)
    mc = CASCI(mf, ncas=4, nelecas=4).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )
    dm1, dm2 = driver._effective_rdms(mc, 0)
    nvar = pack_nonredundant(
        np.zeros((mf.nmo, mf.nmo)),
        mc.ncore,
        mc.ncas,
        mf.nmo,
    ).size
    vec = np.linspace(0.2, 1.0, nvar)
    vec /= np.linalg.norm(vec)

    dense = driver._analytic_orbital_hessian_action(h1_mo, eri_mo, dm1, dm2, mc, vec)
    factored = driver._analytic_orbital_hessian_action(
        h1_mo,
        eri_mo,
        dm1,
        dm2,
        mc,
        vec,
        pair_factors_mo=pair_factors,
    )

    np.testing.assert_allclose(factored, dense, atol=1.0e-12)


def test_second_order_factorized_active_core_derivatives_match_dense():
    mf, mo_guess = _distorted_lih_reference(angle=0.08)
    mf.cholesky_jk = True
    mf.cholesky_tol = 1.0e-10
    driver = SecondOrderCASSCF(mf, ncas=4, nelecas=4)
    h1_mo = mf.get_hcore_mo(mo_guess)
    pair_factors = transform_eri_factors_to_mo_pair(
        _get_mf_cholesky_factors(mf),
        mo_guess,
        mo_guess,
    )
    eri_mo = driver._assemble_eri_from_pair_factors(pair_factors)
    mc = CASCI(mf, ncas=4, nelecas=4).run(
        nstates=1,
        mo_coeff=mo_guess,
        method="direct_ci",
    )

    dh1_dense, deri_dense = driver._active_integral_derivative_basis(mc, h1_mo, eri_mo)
    core_dense = driver._core_energy_derivative_basis(mc, h1_mo, eri_mo)
    driver._full_derivative_cache = None
    dh1_factored, deri_factored = driver._active_integral_derivative_basis(
        mc,
        h1_mo,
        eri_mo,
        pair_factors_mo=pair_factors,
    )
    core_factored = driver._core_energy_derivative_basis(
        mc,
        h1_mo,
        eri_mo,
        pair_factors_mo=pair_factors,
    )

    np.testing.assert_allclose(dh1_factored, dh1_dense, atol=1.0e-12)
    np.testing.assert_allclose(deri_factored, deri_dense, atol=1.0e-12)
    np.testing.assert_allclose(core_factored, core_dense, atol=1.0e-12)


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
