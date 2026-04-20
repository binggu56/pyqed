#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare H2 real-time dipole response from:
1. exact dense active-space propagation
2. TDDMRG
3. RTTDHF
"""

import io
import contextlib
import logging

import numpy as np
import ultraplot as uplt
from scipy.linalg import expm

from pyqed.qchem import Molecule, RTTDHF, gaussian_pulse
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import TDDMRG
from pyqed.qchem.dmrg.tddmrg import _mpo_to_dense_matrix
from pyqed.mps.decompose import tt_to_tensor
from pyqed.mps.mps import expect_mps


def main():
    logging.disable(logging.CRITICAL)

    out = "/Users/bingg/Library/CloudStorage/OneDrive-西湖大学/pyqed/examples/qchem/tddmrg_h2_threeway_compare.png"

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        mol.build(driver="gbasis")
        mf = RHF(mol).run()

    omega = 0.6
    dt = 0.1
    steps = 240
    pulse = gaussian_pulse(
        amplitude=2e-3,
        center=12.0,
        width=4.0,
        omega=omega,
        phase=0.0,
        polarization=(0.0, 0.0, 1.0),
    )

    td = TDDMRG(mf, ncas=2, nelecas=2, D=8, td_bond_dim=8, init_guess="cid")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        td.build()
        td.optimize_ground_state(
            nstates=1,
            nsweeps=4,
            symmetry_list=["charge", "sz"],
            compute_s2=False,
        )

    psi0 = td.export_initial_guess(dense=True)
    mu_mpo = td.get_interaction_mpo(axis=2)
    mu0 = float(np.real(expect_mps(psi0.factors, mu_mpo.factors)))

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        td.run(dt=dt, steps=steps, interval=1, field=pulse, e_ops=["mu_z"])
    times = np.concatenate(([0.0], td.times))
    mu_td = np.concatenate(([mu0], np.real(td.observables[:, 0])))

    vec = np.asarray(tt_to_tensor(psi0.factors), dtype=complex).reshape(-1)
    mu_mat = _mpo_to_dense_matrix(mu_mpo)
    h_mat = _mpo_to_dense_matrix(td._get_td_hamiltonian())
    u_half = expm(-0.5j * dt * h_mat)

    mu_exact = [float(np.real(np.vdot(vec, mu_mat @ vec)))]
    time = 0.0
    for _ in range(steps):
        ez = td._field_vector(time + 0.5 * dt, pulse)[2]
        vec = u_half @ vec
        vec = expm(+1j * dt * ez * mu_mat) @ vec
        vec = u_half @ vec
        vec = vec / np.linalg.norm(vec)
        mu_exact.append(float(np.real(np.vdot(vec, mu_mat @ vec))))
        time += dt
    mu_exact = np.asarray(mu_exact)

    rt = RTTDHF(mf, field=pulse)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        rt.run(dt=dt, nsteps=steps, store_dm=False)
    mu_rt = np.asarray(rt.dipoles[:, 2], dtype=float)

    field_z = np.array([td._field_vector(t, pulse)[2] for t in times], dtype=float)

    print("System: H2 / STO-3G / CAS(2,2)")
    print(f"omega = {omega:.6f} a.u.")
    print(f"max |mu_exact - mu_exact(0)| = {np.max(np.abs(mu_exact - mu_exact[0])):.6e}")
    print(f"max |mu_tddmrg - mu_tddmrg(0)| = {np.max(np.abs(mu_td - mu_td[0])):.6e}")
    print(f"max |mu_rttdhf - mu_rttdhf(0)| = {np.max(np.abs(mu_rt - mu_rt[0])):.6e}")
    print(f"max |mu_exact - mu_tddmrg| = {np.max(np.abs(mu_exact - mu_td)):.6e}")
    print(f"Saved figure to: {out}")

    fig, axes = uplt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))

    axes[0].plot(times, field_z, lw=2.0, color="tab:red", label=r"$E_z(t)$")
    axes[0].set_ylabel("Field (a.u.)")
    axes[0].legend(loc="best", frame=False)

    axes[1].plot(times, mu_exact, lw=2.2, color="black", label="Exact Dense")
    axes[1].plot(times, mu_td, lw=1.8, ls="--", color="tab:blue", marker="o", ms=3, label="TDDMRG")
    axes[1].plot(times, mu_rt, lw=1.8, ls="-.", color="tab:green", label="RTTDHF")
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_ylabel(r"$\mu_z(t)$ (a.u.)")
    axes[1].legend(loc="best", frame=False)

    fig.format(suptitle=rf"H2 response comparison ($\omega={omega:.2f}$ a.u.)")
    fig.savefig(out, dpi=200)


if __name__ == "__main__":
    main()
