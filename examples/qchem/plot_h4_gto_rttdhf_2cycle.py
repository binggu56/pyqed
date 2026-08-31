#!/usr/bin/env python3
"""Plot the two-cycle H4 GTO RT-TDHF response and HHG spectrum."""

import matplotlib.pyplot as plt
import numpy as np


OMEGA0 = 0.057
data = np.load("h4_gto_rttdhf_2cycle.npz")
time = data["time"]
field = data["field"][:, 2]
dipole = data["dipole"][:, 2]
induced_dipole = dipole - dipole[0]
energy = data["energy"]

fig, axes = plt.subplots(3, 1, sharex=True, figsize=(7.2, 7.2))
axes[0].plot(time, field, color="tab:red")
axes[0].set_ylabel(r"$E_z(t)$ (a.u.)")
axes[1].plot(time, induced_dipole, color="tab:blue")
axes[1].set_ylabel(r"$\Delta\mu_z(t)$ (a.u.)")
axes[2].plot(time, energy - energy[0], color="tab:green")
axes[2].set_ylabel(r"$E(t)-E(0)$ (a.u.)")
axes[2].set_xlabel("Time (a.u.)")
fig.suptitle(r"H$_4$ GTO RT-TDHF: two-cycle polarization response")
fig.tight_layout()
fig.savefig("h4_gto_rttdhf_2cycle_response.png", dpi=240)
plt.close(fig)

dt = time[1] - time[0]
acceleration = np.gradient(np.gradient(induced_dipole, dt), dt)
window = np.hanning(time.size)
angular_frequency = 2.0 * np.pi * np.fft.rfftfreq(time.size, d=dt)
harmonic_order = angular_frequency / OMEGA0
intensity = np.abs(np.fft.rfft(acceleration * window)) ** 2
positive = harmonic_order >= 1.0
intensity /= np.max(intensity[positive])

fig, ax = plt.subplots(figsize=(7.2, 4.2))
ax.semilogy(harmonic_order, np.maximum(intensity, 1.0e-14), color="tab:purple")
ax.set_xlim(0.0, 50.0)
ax.set_ylim(1.0e-12, 2.0)
ax.set_xlabel(r"Harmonic order $\omega/\omega_0$")
ax.set_ylabel("Normalized HHG intensity")
ax.set_title(r"H$_4$ GTO RT-TDHF: two-cycle HHG spectrum")
fig.tight_layout()
fig.savefig("h4_gto_rttdhf_2cycle_hhg.png", dpi=240)
plt.close(fig)
