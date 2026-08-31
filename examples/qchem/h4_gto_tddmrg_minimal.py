#!/usr/bin/env python3
"""Minimal two-cycle GTO TDDMRG calculation for linear H4."""

import matplotlib.pyplot as plt
import numpy as np

from pyqed.qchem import Molecule
from pyqed.qchem.dmrg import TDDMRG
from pyqed.qchem.hf import RHF


E0 = 0.05
OMEGA = 0.057
NCYCLES = 2
DT = 0.2
D = 16
NCAS = 4
NELECAS = 4
DURATION = NCYCLES * 2.0 * np.pi / OMEGA
NSTEPS = round(DURATION / DT)


def field(t):
    ez = 0.0
    if 0.0 <= t <= DURATION:
        ez = E0 * np.sin(np.pi * t / DURATION) ** 2 * np.sin(OMEGA * t)
    return np.array([0.0, 0.0, ez])


field.polarization = np.array([0.0, 0.0, 1.0])


mol = Molecule(
    atom="H 0 0 -3.6; H 0 0 -1.2; H 0 0 1.2; H 0 0 3.6",
    unit="bohr",
    basis="d-aug-cc-pvdz",
)
mol.build()
mf = RHF(mol).run()

td = TDDMRG(
    mf,
    ncas=NCAS,
    nelecas=NELECAS,
    init_guess="hf",
).build()
td.optimize_ground_state(
    D=D,
    nstates=1,
    nsweeps=10,
    symmetry_list=["charge", "sz"],
    compute_s2=False,
)

psi0 = td.export_ground_state(dense=True)
mu_z_mpo = td.get_interaction_mpo(axis=2)
mu0 = float(np.real(psi0.expectation(mu_z_mpo)))
zero_mpo = td._zero_mpo(2 * td.ncas, dtype=complex)
td.run(
    psi0=psi0,
    dt=DT,
    steps=NSTEPS,
    interval=1,
    field=field,
    e_ops=["mu_z"],
    interaction_mpo=(zero_mpo, zero_mpo, mu_z_mpo),
    tdvp_dynamic_mode="midpoint",
    D=D,
)

time = np.concatenate(([0.0], np.asarray(td.times)))
field_z = np.concatenate(([field(0.0)[2]], np.asarray(td.fields)[:, 2]))
dipole_z = np.concatenate(([mu0], np.real(td.observables[:, 0])))
energy = np.real(td.static_energies)
np.savez(
    "h4_gto_tddmrg_2cycle.npz",
    time=time,
    field=field_z,
    dipole=dipole_z,
    energy=energy,
)

induced_dipole = dipole_z - dipole_z[0]
acceleration = np.gradient(np.gradient(induced_dipole, DT), DT)
window = np.hanning(time.size)
frequency = 2.0 * np.pi * np.fft.rfftfreq(time.size, d=DT)
harmonic_order = frequency / OMEGA
hhg = np.abs(np.fft.rfft(acceleration * window)) ** 2
hhg /= np.max(hhg[harmonic_order >= 1.0])

fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.5))
axes[0].plot(time, field_z, color="tab:red")
axes[0].set_ylabel(r"$E_z(t)$ (a.u.)")
axes[1].plot(time, induced_dipole, color="tab:blue")
axes[1].set_xlabel("Time (a.u.)")
axes[1].set_ylabel(r"$\Delta\mu_z(t)$ (a.u.)")
axes[2].semilogy(harmonic_order, np.maximum(hhg, 1.0e-14), color="tab:purple")
axes[2].set_xlim(0.0, 50.0)
axes[2].set_ylim(1.0e-12, 2.0)
axes[2].set_xlabel(r"Harmonic order $\omega/\omega_0$")
axes[2].set_ylabel("Normalized HHG intensity")
fig.suptitle(rf"H$_4$ GTO TDDMRG: CAS({NELECAS}e, {NCAS}o), $D={D}$")
fig.tight_layout()
fig.savefig("h4_gto_tddmrg_2cycle.png", dpi=240)
plt.close(fig)
