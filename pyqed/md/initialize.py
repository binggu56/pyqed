#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 17:31:40 2024

@author: bingg
"""

import os
import numpy as np
from prepare import Prepare
from utility import Utilities




class InitializeSimulation(Prepare, Utilities):
    def __init__(self,
                *args,
                **kwargs,
                ):
        super().__init__(*args, **kwargs)