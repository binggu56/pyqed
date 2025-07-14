#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  8 09:34:51 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

# if __name__=='__main__':
    # lista = ['0']
    # listb = ['1', '2']
    # print(composite_state_index(lista, listb))
import numpy as np
from pyqed.mps.abelian import ConservedSite

sa = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 2], [1, 3]], qmin=1, qmax=2)

sb = ConservedSite(qn=[0, 1], degeneracy=[2, 2], state_index=[[0, 1], [2, 3]])



for i in range(2):
    sa += sa
    sa.info()
    
    sb += sb

# sb.info()
