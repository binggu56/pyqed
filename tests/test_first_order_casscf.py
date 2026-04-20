import numpy as np
from scipy.linalg import expm

from pyqed.qchem import CASSCF, COCASCI, FirstOrderCASSCF, Molecule
from pyqed.qchem.mcscf.direct_ci import CASCI


def test_casscf_and_cocasci_are_distinct_public_apis():
    assert CASSCF is not COCASCI
    assert CASSCF is FirstOrderCASSCF
    assert CASSCF.__name__ == "CASSCF"
    assert COCASCI.__name__ == "COCASCI"


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
    mc = CASSCF(mf, ncas=2, nelecas=2, max_cycle=8).run(mo_coeff=mo_guess)

    assert np.isfinite(mc.e_tot[0])
    assert mc.e_tot[0] < mc0.e_tot[0] - 1.0e-6
    assert len(mc.history) >= 1


def test_first_order_casscf_diis_path_runs_on_lih():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build(driver="gbasis")

    mf = mol.RHF().run()
    mc0 = CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method="direct_ci")
    mc = CASSCF(
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
    mc = CASSCF(
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
    mc = CASSCF(
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
