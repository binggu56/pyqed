import numpy as np

from pyqed.qchem import Molecule, TDCASCI
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.direct_ci import CASCI


def _h2_casci(nstates=4):
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    return CASCI(mf, ncas=2, nelecas=2, verbose=0).run(
        nstates=nstates,
        method="direct_ci",
    )


def test_tdcasci_is_exported_without_old_rt_names():
    from pyqed.qchem.tdcasci import TDCASCI as DirectTDCASCI
    import pyqed.qchem as qchem

    assert TDCASCI is DirectTDCASCI
    assert not hasattr(qchem, "RTCASCI")
    assert not hasattr(qchem, "RealTimeCASCI")


def test_tdcasci_field_free_preserves_norm_energy_phase_and_populations():
    mc = _h2_casci(nstates=4)
    td = TDCASCI(mc)

    traj = td.run(dt=0.07, nsteps=8, ci0=0)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(traj.energies, mc.e_tot[0], atol=1.0e-10)
    np.testing.assert_allclose(traj.populations[:, 0], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(traj.populations[:, 1:], 0.0, atol=1.0e-10)

    active_e = float(mc.e_tot[0] - mc.e_core)
    phase = np.exp(-1j * active_e * traj.times)
    reference = phase[:, None] * np.asarray(mc.ci[0], dtype=complex)[None, :]
    np.testing.assert_allclose(traj.ci, reference, atol=1.0e-10)


def test_tdcasci_state_basis_matches_determinant_basis_for_field_free_state():
    mc = _h2_casci(nstates=4)
    td = TDCASCI(mc)

    det_traj = td.run(dt=0.05, nsteps=6, ci0=1)
    state_traj = td.run(dt=0.05, nsteps=6, ci0=1, basis="state", nstates=4)

    np.testing.assert_allclose(state_traj.norms, 1.0, atol=1.0e-12)
    np.testing.assert_allclose(state_traj.populations[:, 1], 1.0, atol=1.0e-10)
    np.testing.assert_allclose(state_traj.ci, det_traj.ci, atol=1.0e-10)


def test_tdcasci_kick_and_spectrum_helpers():
    mc = _h2_casci(nstates=4)
    td = TDCASCI(mc)

    traj = td.run(
        dt=0.05,
        nsteps=8,
        ci0=0,
        kick={"strength": 1.0e-3, "axis": "z"},
    )

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert np.max(np.abs(traj.dipoles[:, 2])) > 1.0e-6
    omega, power = traj.dipole_spectrum(axis="z")
    assert omega.shape == power.shape
    assert np.all(power >= 0.0)
    corr_omega, corr_power = traj.autocorrelation_spectrum()
    assert corr_omega.shape == corr_power.shape


def test_tdcasci_accepts_general_time_dependent_one_body_drive():
    mc = _h2_casci(nstates=4)
    td = TDCASCI(mc)
    ncas = mc.ncas
    drive = np.zeros((ncas, ncas))
    drive[0, 1] = drive[1, 0] = 0.02

    def h1(time):
        return np.sin(0.3 * time) * drive

    traj = td.run(dt=0.05, nsteps=6, ci0=0, h1_mo=h1)

    np.testing.assert_allclose(traj.norms, 1.0, atol=1.0e-12)
    assert traj.ci.shape == (7, mc.binary.shape[0])
