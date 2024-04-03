#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 19:55:52 2023

@author: xiaozhu
"""

import numpy as np
from scipy.constants import atomic_mass, electron_mass
import scipy.constants as const
from pyscf import gto, scf, dft, tdscf, ci
from functools import reduce
from pyscf import hessian as hessian_module
import pyscf.hessian.rhf

# Build molecule
mol = gto.Mole()
mol.build(
    atom = '''
    N   -2.912892    0.444945    0.040859
    C   -3.191128   -0.790170    0.027261
    C   -4.458793   -1.235223   -0.013632
    N   -5.447661   -0.444953   -0.040858
    C   -5.169422    0.790163   -0.027260
    C   -3.901761    1.235215    0.013633
    H   -2.344820   -1.496317    0.050492
    H   -4.678296   -2.315552   -0.024843
    H   -6.015733    1.496310   -0.050491
    H   -3.682256    2.315544    0.024844
    ''',
    basis = 'sto-3g',
    symmetry = False,
)

# Perform RHF calculation
mf = scf.RHF(mol)
mf.kernel()
mf.run()

# Calculate Hessian
h_GS = mf.Hessian().kernel()
h_GS = h_GS.transpose(0, 2, 1, 3).reshape(mol.natm*3, mol.natm*3)

# Mass-weighted Hessian
atom_masses = mol.atom_mass_list(isotope_avg=True)

amu_to_au = atomic_mass / electron_mass
atom_masses_au = atom_masses * amu_to_au

atom_masses_au = np.repeat(atom_masses_au, 3)
# atom_masses = np.repeat(atom_masses, 3)
Mhalf = 1/np.sqrt(np.outer(atom_masses_au, atom_masses_au))
weighted_h_GS = h_GS* Mhalf

# Calculate eigenvalues and eigenvectors of the hessian
force_cst_GS, modes_GS = np.linalg.eigh(weighted_h_GS)

# Convert force constants to frequencies
freq_GS = np.sqrt(np.abs(force_cst_GS))

# Determine the number of normal modes to keep based on linearity
num_modes_to_keep = 3 * mol.natm - 6 if not mol.symmetry else 3 * mol.natm - 5

# Keep the last num_modes_to_keep frequencies and normal modes
freq_GS_last = freq_GS[-num_modes_to_keep:]

modes_GS_last = modes_GS[-num_modes_to_keep:,:]

# Choose two normal models
mode1 = modes_GS_last[2, :]  # third mode
mode2 = modes_GS_last[5, :]  # Sixth mode
freq1 = freq_GS_last[2] 
freq2 = freq_GS_last[5]

# Step 2: Set up the grid for mass-weighted normal coordinates Q in unit of square-root(mass)*length 
grid_x = np.linspace(-6, 6, 16)/ np.sqrt(freq1)
grid_y = np.linspace(-6, 6, 16)/ np.sqrt(freq2) # 33 points from -6 to 6
print(grid_x)
configurations = []
initial_coords = mol.atom_coords()

# # Step 3: Generate configurations: transform mass weighted coordinates to real coordinates
for displacement1 in grid_x:
    for displacement2 in grid_y:
        # Displace pyrazine along the two modes
        displacement_vector = displacement1 * mode1 + displacement2 * mode2
        scaled_displacement = displacement_vector / np.sqrt(atom_masses_au)
        new_config = initial_coords + scaled_displacement.reshape(-1, 3)
        configurations.append(new_config)

num_configs = len(configurations)
# num_configs = len(grid_x) * len(grid_y)

for i in range(num_configs):
    x_index, y_index = np.unravel_index(i, (len(grid_x), len(grid_y)))
    print(f"Config {i} corresponds grid_x[{x_index}] and grid_y[{y_index}]")
    
for i, config in enumerate(configurations):
    with open(f"config_{i}.xyz", 'w') as file:
        file.write(f"{len(config)}\n")  
        file.write(f"Configuration {i}\n")  
        for atom, (x, y, z) in enumerate(config):
            atom_symbol = mol._atom[atom][0]  # Get the symbol of the atom
            file.write(f'{atom_symbol} {x} {y} {z}\n')
        # for atom, coord in zip(mol.atom_symbols(), config):
        #     file.write(f"{atom} {coord[0]} {coord[1]} {coord[2]}\n")
