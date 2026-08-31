import math

import numpy as np

from pyqed.qchem import Molecule, FirstOrderRASSCF, RASCI, RASSCF, SecondOrderRASSCF
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.direct_ci import CASCI
from pyqed.qchem.mcscf.orbopt import nonredundant_pairs
from pyqed.qchem.mcscf.rasscf import (
    generate_ras_determinants,
    ras_occupations,
)


def _lih_reference():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    return RHF(mol).run()


def test_generate_ras_determinants_applies_hole_particle_limits():
    binary = generate_ras_determinants(
        4,
        (2, 2),
        ras_spaces=(1, 2, 1),
        max_holes=1,
        max_electrons=1,
    )
    holes, ras3_elec = ras_occupations(binary, (1, 2, 1))

    assert binary.shape[0] < math.comb(4, 2) ** 2
    assert np.all(holes <= 1)
    assert np.all(ras3_elec <= 1)


def test_rasci_cas_limit_matches_casci_energy():
    mf = _lih_reference()

    cas = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method="direct_ci")
    ras = RASCI(
        mf,
        ncas=4,
        nelecas=4,
        ras_spaces=(0, 4, 0),
        max_holes=0,
        max_electrons=0,
    ).run(nstates=2, method="direct_ci")

    assert ras.ras_ndet == cas.binary.shape[0]
    np.testing.assert_allclose(ras.e_tot, cas.e_tot, atol=1.0e-10)


def test_rasci_restricted_space_is_variational_subset():
    mf = _lih_reference()

    cas = CASCI(mf, ncas=4, nelecas=4).run(nstates=1, method="direct_ci")
    ras = RASCI(
        mf,
        ncas=4,
        nelecas=4,
        ras_spaces=(1, 2, 1),
        max_holes=1,
        max_electrons=1,
    ).run(nstates=1, method="direct_ci")

    assert 0 < ras.ras_ndet < ras.ras_full_ndet
    assert ras.e_tot[0] >= cas.e_tot[0] - 1.0e-10


def test_rasci_accepts_openmolcas_style_nactel_aliases():
    mf = _lih_reference()

    direct = RASCI(
        mf,
        ncas=4,
        nelecas=4,
        ras_spaces=(1, 2, 1),
        max_holes=1,
        max_electrons=1,
    ).run(nstates=1)
    molcas_style = RASCI.from_openmolcas(
        mf,
        ras1=1,
        ras2=2,
        ras3=1,
        nactel=(4, 1, 1),
    ).run(nstates=1)

    assert molcas_style.ras_spaces == (1, 2, 1)
    assert molcas_style.max_holes == 1
    assert molcas_style.max_electrons == 1
    np.testing.assert_allclose(molcas_style.e_tot, direct.e_tot, atol=1.0e-10)


def test_first_order_rasscf_includes_cross_ras_active_rotations_and_runs():
    mf = _lih_reference()
    mc = FirstOrderRASSCF(
        mf,
        ncas=4,
        nelecas=4,
        ras_spaces=(1, 2, 1),
        max_holes=1,
        max_electrons=1,
        max_cycle=1,
        conv_tol_grad=1.0e9,
        max_step=0.0,
        verbose=0,
    )
    mc.run(nstates=1)

    assert mc.converged
    assert 0 < mc.ras_ndet < mc.ras_full_ndet

    pairs = set(mc.orbital_rotation_pairs(ncore=mc.ncore, ncas=mc.ncas, nmo=mc.nmo))
    cas_pairs = set(nonredundant_pairs(mc.ncore, mc.ncas, mc.nmo))
    assert len(pairs) > len(cas_pairs)
    assert (mc.ncore, mc.ncore + 1) in pairs
    assert (mc.ncore + 2, mc.ncore + 3) in pairs


def test_second_order_rasscf_is_public_default_and_runs():
    mf = _lih_reference()
    mc = RASSCF(
        mf,
        ncas=4,
        nactel=(4, 1, 1),
        ras_spaces=(1, 2, 1),
        max_cycle=2,
        max_micro_cycle=1,
        conv_tol=1.0e9,
        conv_tol_grad=1.0e9,
        conv_tol_grad_relaxed=1.0e9,
        max_step=0.0,
        verbose=0,
    )
    assert isinstance(mc, SecondOrderRASSCF)

    mc.run(nstates=1)

    assert mc.converged
    assert 0 < mc.ras_ndet < mc.ras_full_ndet
    assert mc.micro_history


def test_second_order_rasscf_default_relaxed_hessian_converges_tightly():
    mf = _lih_reference()
    mc = RASSCF.from_openmolcas(
        mf,
        inactive=0,
        ras1=1,
        ras2=2,
        ras3=1,
        nactel=(4, 1, 1),
        max_cycle=12,
        max_micro_cycle=8,
        conv_tol=1.0e-10,
        conv_tol_grad=1.0e-6,
        conv_tol_grad_relaxed=1.0e-6,
        verbose=0,
    )

    assert mc.coupling == "relaxed_fd"

    mc.run(nstates=1)

    assert mc.converged
    assert mc.history[-1]["gradient_norm"] < 1.0e-6
    np.testing.assert_allclose(mc.e_tot[0], -7.8816443039, atol=5.0e-8)
