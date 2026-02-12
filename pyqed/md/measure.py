#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:32:16 2024

@author: bingg
"""

import numpy as np
from initialize import InitializeSimulation

# from Measurements import Measurements
# import numpy as np
import copy
import os


class Measurements(InitializeSimulation):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)






class MinimizeEnergy(Measurements):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)


import warnings
warnings.filterwarnings('ignore')


class MonteCarlo(Measurements):
    def __init__(self,
                *args,
                **kwargs):
        super().__init__(*args, **kwargs)


class MolecularDynamics(Measurements):
    def __init__(self,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)


# # Import the required modules
# from utility import Utilities
# # from MonteCarlo import MonteCarlo

# # Make sure that MonteCarlo correctly inherits from Utilities
# def test_montecarlo_inherits_from_utilities():
#     assert issubclass(MonteCarlo, Utilities), \
#         "MonteCarlo should inherit from Utilities"
#     print("MonteCarlo correctly inherits from Utilities")

# # Make sure that Utilities does not inherit from MonteCarlo
# def test_utilities_does_not_inherit_from_montecarlo():
#     assert not issubclass(Utilities, MonteCarlo), \
#         "Utilities should not inherit from MonteCarlo"
#     print("Utilities does not inherit from MonteCarlo, as expected")

# # In the script is launched with Python, call Pytest
# if __name__ == "__main__":
#     import pytest
#     pytest.main(["-s", __file__])

mc = MonteCarlo()