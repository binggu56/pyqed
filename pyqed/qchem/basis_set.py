#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 00:01:55 2024

@author: bingg
"""

import numpy as np
from gbasis.parsers import parse_gbs, make_contractions
from gbasis.integrals.overlap import overlap_integral
from gbasis.integrals.kinetic_energy import kinetic_energy_integral
from gbasis.integrals.nuclear_electron_attraction import \
nuclear_electron_attraction_integral
from gbasis.integrals.electron_repulsion import electron_repulsion_integral





# Define atomic symbols and coordinates (i.e., basis function centers)
atoms = ["H", "H"]
atcoords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

def build(mol):
    """
    electronic integrals in AO

    Parameters
    ----------
    mol : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """

    if mol.basis in ['631g', '6-31g', '631G', '6-31G']:
        # Obtain basis functions from the basis set files
        basis_dict = parse_gbs("6-31g.1.gbs")
        basis = make_contractions(basis_dict, atoms, atcoords, coord_types="c")


    # compute overlap integrals in AO and MO basis
    mol.overlap = overlap_integral(basis)


    # olp_mo = overlap_integral(basis, transform=mo_coeffs.T)

    # compute kinetic energy integrals in AO basis
    k_int1e = kinetic_energy_integral(basis)
    print("Shape kinetic energy integral: ", k_int1e.shape, "(#AO, #AO)")


    # compute nuclear-electron attraction integrals in AO basis
    atnums = np.array([1,1])
    nuc_int1e = nuclear_electron_attraction_integral(
            basis, atcoords, atnums)
    print("Shape Nuclear-electron integral: ", nuc_int1e.shape, "(#AO, #AO)")

    mol.hcore = k_int1e + nuc_int1e

    #Compute e-e repulsion integral in MO basis, shape=(#MO, #MO, #MO, #MO)
    int2e_mo = electron_repulsion_integral(basis, notation='chemist')
    mol.eri = int2e_mo


def energy_nuc(atcoords, atnums):
    # Compute Nucleus-Nucleus repulsion
    rab = np.triu(np.linalg.norm(atcoords[:, None]- atcoords, axis=-1))
    at_charges = np.triu(atnums[:, None] * atnums)[np.where(rab > 0)]
    nn_e = np.sum(at_charges / rab[rab > 0])
    return nn_e

# kin_e = np.trace(dm.dot(k_int1e))
# print("Kinetic energy (Hartree):", kin_e)

# To obtain the total number of AOs we compute the cartesian components for each angular momentum
total_ao = 0
print(f"Number of generalized shells: {len(basis)}") # Output 6
for shell in basis:
    total_ao += shell.angmom_components_cart.shape[0]

print("Total number of AOs: ", total_ao) # output 10