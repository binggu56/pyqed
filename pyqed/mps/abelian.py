#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 23 10:47:12 2025

@author: Bing Gu (gubing@westlake.edu.cn)
"""

from pyqed.mps.mps import Site

from itertools import product

class ConservedSite:
    def __init__(self, qn=None, degenaracy=None, state_index=None):
        if qn is None:
            self.qn = [0, 1, 2] # quantum numbers of number operator
            self.degenaracy = [1, 2, 1] # degeneracy corresponding to the quantum numbers
            self.state_index = [[0], [1, 2], [3]]
            self.d = 4
        
        else:
            self.qn = qn 
            self.degenaracy = degenaracy
            self.state_index = state_index
    
    def __add__(self, other):
        qn = []
        degeneracy = []
        state_index = []
        
        for n in range(len(self.qn)):
            for m in range(len(other.qn)):
                c = self.qn[n] + other.qn[m]

                idx = composite_index(self.state_index[n], other.state_index[m])
                
                d = self.degenaracy[n] * other.degenaracy[m]
                
                
                if c not in qn: # if the charge does not exist yet 
                    qn.append(c)
                    degeneracy += [d]
                    state_index.append(idx)
                    
                else:
                    i = qn.index(c)
                    degeneracy[i] += d
                    state_index[i] += idx
        
        return ConservedSite(qn, degeneracy, state_index)
    
    def truncate(self, qn):
        
        i = self.qn.index(qn)
        
    def subblock(self, qn):
        pass
        
    
    def info(self):
        print('Quantum number', self.qn)
        print('Degeneracy', self.degenaracy)
        print('Total number of states', sum(self.degenaracy))
        print('state indices', self.state_index)
    
    
        


def composite_index(lista, listb):
    return [str(a) + str(b) for a, b in product(lista,listb)]

# lista = ['0']
# listb = ['1', '2']
# print(composite_state_index(lista, listb))

a = ConservedSite()
b = ConservedSite()
s = a + b + a 
s.info()