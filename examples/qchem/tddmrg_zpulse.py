#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TDDMRG example: H2 driven by a z-polarized Gaussian pulse.
"""

import numpy as np
import matplotlib.pyplot as plt

from pyqed.qchem import Molecule, gaussian_pulse
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import TDDMRG


def main():
    out = "/Users/bingg/Library/CloudStorage/OneDrive-西湖大学/pyqed/examples/qchem/tddmrg_zpulse_response.png"

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()

    pulse = gaussian_pulse(
        amplitude=5e-3,
        center=2.0,
        width=0.5,
        frequency=0.0,
        phase=0.0,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(mf, ncas=2, nelecas=2, D=8, init_guess="cid").build()
    td.optimize_ground_state(
        nstates=1,
        nsweeps=6,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
    )
    td.run(
        dt=0.1,
        steps=80,
        interval=1,
        field=pulse,
        e_ops=["H", "mu_z"],
    )

    times = np.asarray(td.times, dtype=float)
    field_z = np.asarray(td.fields[:, 2], dtype=float)
    mu_z = np.real(td.observables[:, 1])
    polarization_z = mu_z - mu_z[0]
    energies = np.real(td.observables[:, 0]) + td.e_core

    print(f"Computed {len(times)} time samples.")
    print(f"max |E_z(t)| = {np.max(np.abs(field_z)):.6e} a.u.")
    print(f"max |Delta mu_z(t)| = {np.max(np.abs(polarization_z)):.6e} a.u.")
    print(f"max |E(t) - E(0)| = {np.max(np.abs(energies - energies[0])):.6e} Ha")
    print(f"Saved figure to: {out}")

    fig, axes = plt.subplots(2, 1, figsize=(7, 6), sharex=True)

    axes[0].plot(times, field_z, color="tab:red", lw=1.8, label="E_z(t)")
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].legend()

    axes[1].plot(times, polarization_z, color="tab:blue", lw=1.8, label=r"$\Delta \mu_z(t)$")
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_ylabel("Polarization (a.u.)")
    axes[1].legend()

    fig.suptitle("TDDMRG H2 response to a z-polarized Gaussian pulse")
    fig.tight_layout()
    fig.savefig(out, dpi=180)
    plt.show()


if __name__ == "__main__":
    main()
