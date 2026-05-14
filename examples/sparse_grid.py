#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 16:06:06 2023

@author: Bing Gu (gubing@westlake.edu.cn)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.smolyak.sg import SparseGrid

if __name__=='__main__':
    
    level = 4
    dim = 2
    # Sparse grid solver
    sg = SparseGrid(ndim=dim, level=level)
    
    sg.generatePoints()
    # sg.print_points()
    
    # print(sg.indices)
    
    #
    #  Determine sg.gP with the coordinates of the points 
    #  associated with the sparse grid index set.
    #
    
    # print('index set for SGCT\n', sg.index_set)
    #
    #  Print the points in the grid.
    #

          

        
    #
    #  Did we compute the right number of grid points?
    #
    # assert(len(sg.indices) == 17)

    sg.plot_grid()
    
    #
    #  Evaluate the initial wavepacket at each grid point.
    #
    for i in range(len(sg.indices)):
        # sum = 1.0
        pos = sg.gP[tuple(sg.indices[i])].pos
        # for j in range(len(pos)):
            # sum *= 4.*pos[j]*(1.0-pos[j])
            
        # sg.gP[tuple(sg.indices[i])].fv = gwp(pos)
    #
    #  Convert to hierarchical values.
    #
    # sg.nodal2Hier()
    
    # x = np.linspace(-6, 6, 2**5, endpoint=False)[1:]

    index_set, c = sg.combination_technique() 
    
    print(index_set)
    print(c)
    
