from types import SimpleNamespace

import numpy as np

from pyqed.strong_field import HHG


class _Field:
    omega = 0.5
    duration = 0.6

    def __call__(self, time):
        return np.array([0.0, 0.0, np.sin(self.omega * time)])


class _Dynamics:
    def __init__(self):
        self.run_options = None

    def run(self, **options):
        self.run_options = options
        time = options["t0"] + options["dt"] * np.arange(options["nsteps"] + 1)
        dipole = np.zeros((time.size, 3))
        acceleration = np.zeros_like(dipole)
        dipole[:, 2] = np.sin(2.0 * time)
        acceleration[:, 2] = -4.0 * np.sin(2.0 * time)
        return SimpleNamespace(
            times=time,
            dipoles=dipole,
            dipole_accelerations=acceleration,
        )


class _Reference:
    def __init__(self):
        self.dynamics = _Dynamics()
        self.rt_options = None

    def RTTDHF(self, **options):
        self.rt_options = options
        return self.dynamics


class _RHF:
    def __init__(self, reference):
        self.reference = reference
        self.run_options = None

    def run(self, **options):
        self.run_options = options
        return self.reference


class _Molecule:
    def __init__(self):
        self.reference = _Reference()
        self.rhf = _RHF(self.reference)
        self.cap_options = None

    def RHF(self):
        return self.rhf

    def dipole_operator(self, axis):
        assert axis == "z"
        return np.diag([-1.0, 0.0, 1.0])

    def cap(self, **options):
        self.cap_options = options
        return np.diag([0.2, 0.0, 0.2])


def test_ab_initio_hhg_runs_reference_propagation_and_analysis():
    mol = _Molecule()
    hhg = HHG(
        mol,
        method="rt-tdhf",
        field=_Field(),
        cap={"width": 1.0, "strength": 0.2},
        reference_options={"newton": True, "sweeps": 4},
    ).run(dt=0.1)

    assert mol.rhf.run_options == {"newton": True, "sweeps": 4}
    assert mol.cap_options == {"width": 1.0, "strength": 0.2}
    assert mol.reference.rt_options["field"] is hhg.field
    assert hhg.trajectory is not None
    assert hhg.time.size == 7
    assert hhg.harmonic_order.shape == hhg.intensity.shape
    assert np.max(hhg.normalized_intensity) == 1.0


def test_hhg_accepts_prebuilt_reference_and_explicit_step_count():
    mol = _Molecule()
    hhg = HHG(
        mol,
        field=_Field(),
        reference=mol.reference,
        omega=0.5,
        cap=False,
        window="none",
    ).run(dt=0.1, nsteps=4)

    assert mol.rhf.run_options is None
    assert mol.cap_options is None
    assert hhg.time.size == 5
