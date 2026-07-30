#!/usr/bin/env python3
"""Minimal H4 RT-TDHF/RTLDR high-harmonic calculation."""

from pathlib import Path

import numpy as np

from pyqed.dvr import DVR
from pyqed.namd.rtldr.gdvr import GDVRFrame as RTTDHFFrame
from pyqed.namd.rtldr.gdvr import Solver as RTLDR
from pyqed.qchem.gdvr import AtomicChain, cap_operator_from_z


MASS_H = 1836.15267343
REFERENCE_SPACING = 1.5
Q_MIN = -0.2
Q_MAX = 0.2
NPOINTS = 3
LZ = 8.0
NZ = 63
M = 1
CAP_WIDTH = 2.0
CAP_STRENGTH = 0.005
OMEGA = 0.057
FIELD = 0.0534
CYCLES = 6.0
DT = 0.05
OUTPUT = "h4_rtldr_hhg.npz"


def pulse(amplitude, omega, cycles):
    duration = cycles * 2.0 * np.pi / omega

    def field(time):
        value = 0.0
        if 0.0 <= time <= duration:
            value = (
                amplitude
                * np.sin(np.pi * time / duration) ** 2
                * np.sin(omega * time)
            )
        return np.array([0.0, 0.0, value])

    field.duration = duration
    return field


def atomic_positions(q):
    q_in_phase, q_out_of_phase = np.asarray(q, dtype=float)
    z = (np.arange(4) - 1.5) * REFERENCE_SPACING
    z[1] += (q_in_phase - q_out_of_phase) / np.sqrt(2.0)
    z[2] += (q_in_phase + q_out_of_phase) / np.sqrt(2.0)
    return z


def collective_dvr():
    return DVR(
        domains=[(Q_MIN, Q_MAX)] * 2,
        npts=[NPOINTS] * 2,
        mass=MASS_H,
        names=("q_in_phase", "q_out_of_phase"),
    )


def frame(q, field):
    z = atomic_positions(q)
    mol = AtomicChain(
        elements=["H"] * 4,
        coords=[[0.0, 0.0, value] for value in z],
    )
    mol.build(
        Lz=LZ,
        Nz=NZ,
        M=M,
        verbose=False,
        dvr_method="sine",
    )
    mf = mol.RHF().run(conv=1.0e-8, max_iter=100, verbose=False)
    cap = cap_operator_from_z(
        mol.z,
        M=M,
        width=CAP_WIDTH,
        strength=CAP_STRENGTH,
    )
    return RTTDHFFrame(
        mf,
        field=field,
        interaction=mol.dipole_operator("z"),
        cap=cap,
        nuclear_dipole=np.array([0.0, 0.0, np.sum(z)]),
    )


def spectrum(times, dipole, omega0):
    dt = times[1] - times[0]
    acceleration = np.gradient(np.gradient(dipole - dipole[0], dt), dt)
    omega = 2.0 * np.pi * np.fft.rfftfreq(4 * times.size, dt)
    intensity = np.abs(
        np.fft.rfft(acceleration * np.hanning(times.size), 4 * times.size)
    ) ** 2
    scale = np.max(intensity[1:])
    if scale > 0.0:
        intensity[1:] /= scale
    return omega / omega0, intensity, acceleration


def run():
    field = pulse(FIELD, OMEGA, CYCLES)
    nuclear = collective_dvr()
    solver = RTLDR(
        nuclear=nuclear,
        electronic=[frame(q, field) for q in nuclear.points],
    )
    initial_state, ground_energy = solver.ground_state()
    trajectory = solver.run(
        initial_state,
        dt=DT,
        nsteps=int(np.ceil(field.duration / DT)),
        store_overlaps=False,
    )
    dipole = trajectory.weighted_dipole[:, 2]
    harmonic, intensity, acceleration = spectrum(
        trajectory.times,
        dipole,
        OMEGA,
    )

    output = Path(OUTPUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        time=trajectory.times,
        q_in_phase=nuclear.x[0],
        q_out_of_phase=nuclear.x[1],
        collective_points=nuclear.points,
        atomic_z=np.array([atomic_positions(q) for q in nuclear.points]),
        electronic_z=solver.frames[0].z,
        cap_profile=np.diag(solver.frames[0].cap).real,
        ground_energy=ground_energy,
        coefficients=trajectory.coefficients,
        nuclear_density=trajectory.coordinate_density,
        electronic_energies=trajectory.electronic_energies,
        dipole=dipole,
        dipole_acceleration=acceleration,
        electron_count=trajectory.weighted_electron_count,
        harmonic_order=harmonic,
        hhg_intensity=intensity,
    )


if __name__ == "__main__":
    run()
