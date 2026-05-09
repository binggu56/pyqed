#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  8 08:21:45 2026

@author: Bing Gu (gubing at westlake dot edu dot cn)
"""

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg.nonabelian import DMRG

mol = Molecule(
    atom="Li 0 0 0; F 0 0 1.6",
    unit="bohr",
    basis="631g",
)
mol.build()

mf = RHF(mol).run()

dmrg = DMRG(
    mf,
    ncas=8,
    nelecas=8,
    D=16,
    init_guess="cid",
    symmetry="su2",
    verbose=1,
)

dmrg.run(
    nsweeps=2,
    local_basis_policy="block2_like",
    max_bond_mode="per_sector",
    mixer_zero_block_noise_scale=0.0,

    # New SU(2) family-kernel controls
    family_kernel_backend="auto",          # "auto", "dense", or "factor"
    family_dense_threshold=65536,          # max dense elements per local block
    family_dense_max_total_elements=0,     # 0 forces factor kernels by memory budget
)

print("E =", dmrg.e_tot)

for sweep in dmrg.dmrg.history:
    print("sweep", sweep["sweep"], "E", sweep.get("energy"))
    print("family policy:", sweep.get("family_kernel_policy"))

    for obj in sweep.get("bond_objectives", []):
        stats = obj.get("renormalized_operator_table_stats") or {}
        table = stats.get("complementary_family_table") or {}
        if table:
            print("backend:", table["backend"])
            print("dense elements:", table["native_kernel_elements"])
            print("factor elements:", table["factor_kernel_elements"])