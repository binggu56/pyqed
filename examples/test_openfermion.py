#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 22:56:47 2024

@author: bingg
"""

from openfermion import *

# Create some ladder operators
annihilate_2 = FermionOperator('2')
create_2 = FermionOperator('2^')
annihilate_5 = FermionOperator('5')
create_5 = FermionOperator('5^')

# Construct occupation number operators
num_2 = create_2 * annihilate_2
num_5 = create_5 * annihilate_5

# Map FermionOperators to QubitOperators using the JWT
annihilate_2_jw = jordan_wigner(annihilate_2)
create_2_jw = jordan_wigner(create_2)
annihilate_5_jw = jordan_wigner(annihilate_5)
create_5_jw = jordan_wigner(create_5)
num_2_jw = jordan_wigner(num_2)
num_5_jw = jordan_wigner(num_5)

print(annihilate_2_jw)

# Create QubitOperator versions of zero and identity
zero = QubitOperator()
identity = QubitOperator(())

# Check the canonical anticommutation relations
assert anticommutator(annihilate_5_jw, annihilate_2_jw) == zero
assert anticommutator(annihilate_5_jw, annihilate_5_jw) == zero
assert anticommutator(annihilate_5_jw, create_2_jw) == zero
assert anticommutator(annihilate_5_jw, create_5_jw) == identity

# Check that the occupation number operators commute
assert commutator(num_2_jw, num_5_jw) == zero

# Print some output
print("annihilate_2_jw = \n{}".format(annihilate_2_jw))
print('')
print("create_2_jw = \n{}".format(create_2_jw))
print('')
print("annihilate_5_jw = \n{}".format(annihilate_5_jw))
print('')
print("create_5_jw = \n{}".format(create_5_jw))
print('')
print("num_2_jw = \n{}".format(num_2_jw))
print('')
print("num_5_jw = \n{}".format(num_5_jw))