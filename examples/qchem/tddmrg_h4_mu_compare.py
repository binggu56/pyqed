#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare H4 TD-DMRG mu_z(t) for the fixed-polarization builder and the
dense/exact-builder reference.
"""

import time
import numpy as np
import ultraplot as uplt
from scipy.linalg import expm

from pyqed.qchem import Molecule, gaussian_pulse
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg import TDDMRG
from pyqed.qchem.dmrg.overlap import _unitary_rotation_mpo
from pyqed.mps.tdmps import TDMPS


def main():

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

    D = 12
    dt = 0.05
    steps = 20

    td = TDDMRG(mf, ncas=4, nelecas=4, D=D, td_bond_dim=D, init_guess="cid").build()
    td.optimize_ground_state(
        nstates=1,
        nsweeps=2,
        symmetry_list=["charge", "sz"],
        compute_s2=False,
    )

    psi0 = td.export_initial_guess(dense=True)
    H = td._get_td_hamiltonian()
    mu_z = td.get_interaction_mpo(axis=2)
    interaction = td.get_interaction_mpo()
    spatial_z = td.get_interaction_spatial(axis=2)

    def old_exact_builder(dt, time=0.0, field=None, order=2, scale=0):
        del order, scale
        vec = td._field_vector(time, field)
        if not np.any(vec):
            return None
        h_int = -vec[2] * spatial_z
        orbital_transform = expm(-1j * dt * h_int)
        return _unitary_rotation_mpo(orbital_transform, mpo_bond_dim=td.td_bond_dim)

    fixed_solver = TDMPS(
        H,
        D=D,
        interaction_mpo=interaction,
        field=pulse,
        interaction_propagator_builder=td.build_interaction_unitary_mpo,
    )
    t0 = time.perf_counter()
    fixed_solver.run(psi0.copy(), dt=dt, steps=steps, e_ops=[mu_z], interval=1, field=pulse)
    t1 = time.perf_counter()

    dense_solver = TDMPS(
        H,
        D=D,
        interaction_mpo=interaction,
        field=pulse,
        interaction_propagator_builder=old_exact_builder,
    )
    t2 = time.perf_counter()
    dense_solver.run(psi0.copy(), dt=dt, steps=steps, e_ops=[mu_z], interval=1, field=pulse)
    t3 = time.perf_counter()

    times = np.asarray(fixed_solver.times, dtype=float)
    mu_fixed = np.real(fixed_solver.observables[:, 0])
    mu_dense = np.real(dense_solver.observables[:, 0])
    delta = mu_fixed - mu_dense

    print(f"D = {D}, dt = {dt}, steps = {steps}")
    print(f"fixed-polarization path: {t1 - t0:.3f} s")
    print(f"dense/exact path:        {t3 - t2:.3f} s")
    print(f"max |mu_fixed - mu_dense| = {np.max(np.abs(delta)):.6e} a.u.")

    fig, axes = uplt.subplots(nrows=2, ncols=1, sharex=True, figsize=(7, 6))

    axes[0].plot(times, mu_fixed, lw=2.0, label="Fixed polarization")
    axes[0].plot(times, mu_dense, lw=1.8, ls="--", label="Dense / exact builder")
    axes[0].set_ylabel(r"$\mu_z(t)$ (a.u.)")
    axes[0].legend(loc="best", ncols=1, frame=False)

    axes[1].plot(times, delta, color="black", lw=1.8)
    axes[1].set_xlabel("Time (a.u.)")
    axes[1].set_ylabel(r"$\Delta \mu_z(t)$")

    fig.format(suptitle=rf"H4 STO-3G TD-DMRG comparison ($D={D}$)")



if __name__ == "__main__":
    main()
