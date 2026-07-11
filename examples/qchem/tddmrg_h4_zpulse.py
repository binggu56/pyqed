#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare H4 TDDMRG and RTTDHF under the same z-polarized Gaussian pulse.
"""

import numpy as np
import ultraplot as uplt

from pyqed.qchem import Molecule, RTTDHF, gaussian_pulse
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import TDDMRG
from pyqed.mps.mps import expect_mps


def main():
    out = "/Users/bingg/Library/CloudStorage/OneDrive-西湖大学/pyqed/examples/qchem/tddmrg_h4_zpulse_mu.png"

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.6; H 0 0 3.2; H 0 0 4.8",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    omega = 0.6

    pulse = gaussian_pulse(
        amplitude=2e-3,
        center=0.5,
        width=0.2,
        omega=omega,
        phase=0.0,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(mf, ncas=4, nelecas=4, D=12, td_bond_dim=12, init_guess="cid").build()
    td.optimize_ground_state(
        nstates=1,
        nsweeps=4,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
    )
    psi0 = td.export_ground_state(dense=True)
    mu_z_mpo = td.get_interaction_mpo(axis=2)
    center = mol.center_of_mass()
    nuclear_mu_z = np.sum(mol.atom_charges() * (mol.atom_coords()[:, 2] - center[2]))
    mu0_electronic = float(np.real(expect_mps(psi0.factors, mu_z_mpo.factors)))
    td.run(
        dt=0.05,
        steps=10,
        interval=1,
        field=pulse,
        e_ops=["mu_z"],
    )

    times = np.concatenate(([0.0], np.asarray(td.times, dtype=float)))
    electronic_mu_z_dmrg = np.concatenate(([mu0_electronic], np.real(td.observables[:, 0])))
    mu_z_dmrg = nuclear_mu_z + electronic_mu_z_dmrg
    field_z = np.concatenate(([td._field_vector(0.0, pulse)[2]], np.asarray(td.fields[:, 2], dtype=float)))

    rt = RTTDHF(mf, field=pulse).run(
        dt=0.05,
        nsteps=10,
        store_dm=False,
    )
    mu_z_rttdhf = nuclear_mu_z + np.asarray(rt.dipoles[:, 2], dtype=float)

    print(f"Computed {len(times)} time samples.")
    print(f"omega = {omega:.6f} a.u.")
    print("TDDMRG settings: D=12, td_bond_dim=12, nsweeps=4")
    print(f"mu_z^DMRG(t=0) = {mu_z_dmrg[0]:.6e} a.u.")
    print(f"mu_z^RTTDHF(t=0) = {mu_z_rttdhf[0]:.6e} a.u.")
    print(f"max |E_z(t)| = {np.max(np.abs(field_z)):.6e} a.u.")
    print(f"max |mu_z^DMRG(t) - mu_z^DMRG(0)| = {np.max(np.abs(mu_z_dmrg - mu_z_dmrg[0])):.6e} a.u.")
    print(f"max |mu_z^RTTDHF(t) - mu_z^RTTDHF(0)| = {np.max(np.abs(mu_z_rttdhf - mu_z_rttdhf[0])):.6e} a.u.")
    print(f"Saved figure to: {out}")

    fig, axes = uplt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))
    axes[0].plot(times, field_z, lw=2.0, color="tab:red", label=r"$E_z(t)$")
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].legend(loc="best", frame=False)

    axes[1].plot(times, mu_z_dmrg, lw=2.0, marker="o", ms=4, color="tab:blue", label=r"$\mu_z^{\mathrm{DMRG}}(t)$")
    axes[1].plot(rt.times, mu_z_rttdhf, lw=1.8, ls="--", color="tab:green", label=r"$\mu_z^{\mathrm{RTTDHF}}(t)$")
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_ylabel(r"$\mu_z(t)$ (a.u.)")
    axes[1].legend(loc="best", frame=False)

    fig.format(suptitle=rf"H4 response to a z-polarized Gaussian pulse ($\omega={omega:.2f}$ a.u.)")
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
