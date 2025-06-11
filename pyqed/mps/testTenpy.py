#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 23 15:14:08 2020

@author: bing
"""

import tenpy.linalg.np_conserved as npc

M = npc.Array.from_ndarray_trivial([[0., 1.], [1., 0.]])
v = npc.Array.from_ndarray_trivial([2., 4. + 1.j])

v[0] = 3.  # set indiviual entries like in numpy

print("|v> =", v.to_ndarray())
# |v> = [ 3.+0.j  4.+1.j]

M_v = npc.tensordot(M, v, axes=[1, 0])
print("M|v> =", M_v.to_ndarray())
# M|v> = [ 4.+1.j  3.+0.j]
print("<v|M|v> =", npc.inner(v.conj(), M_v, axes='range'))
# <v|M|v> = (24+0j)