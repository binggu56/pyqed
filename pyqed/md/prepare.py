#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:30:04 2024

@author: bingg
"""

import numpy as np
from scipy import constants as cst

class Prepare:
    def __init__(self,
                ureg, # Pint unit registry
                number_atoms, # List - no unit
                epsilon, # List - Kcal/mol
                sigma, # List - Angstrom
                atom_mass,  # List - g/mol
                *args,
                **kwargs):
        self.ureg = ureg
        self.number_atoms = number_atoms
        self.epsilon = epsilon
        self.sigma = sigma
        self.atom_mass = atom_mass
        super().__init__(*args, **kwargs)