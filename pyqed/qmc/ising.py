#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 13:29:21 2025


Ising Model

Uses a small lattice (e.g. 20x20) and displays
only the final lattice state.
"""
from __future__ import division, print_function
import numpy as np
import random as rand
import ultraplot as plt

def deltaU(i, j):
    """
    For a hypothetical dipole flip, compute the potential energy
    change. Each dipole has four nearest neighbors--left, top, right
    and bottom.
    Notice the periodic boundary conditions.
    """
    top = s[size-1,j] if i==0 else s[i-1,j]
    bottom = s[0,j] if i==size-1 else s[i+1,j]
    left = s[i, size-1] if j==0 else s[i,j-1]
    right = s[i,0] if j==size-1 else s[i,j+1]
    Ediff = 2*s[i,j]*(top+bottom+left+right)
    return Ediff

def initialize():
    """
    Initialize the lattice by randomly selecting 1 or -1
    for each dipole.
    """
    for i in range(size):
        for j in range(size):
            s[i,j] = 1 if rand.random()<0.5 else -1

def corr(r):
    """
    The correlation function for distance r
    """
    avg = 0
    for i in range(size):
        for j in range(size):
            me = s[i,j] # self
            ab = s[(i-r)%size, j] # above
            be = s[(i+r)%size, j] # below
            le = s[i, (j-r)%size] # left
            ri = s[i, (j+r)%size] # right
            avg += me*ab + me*be + me*le + me*ri
    avg = avg/(4*size*size)
    M = np.sum(s)/(size*size) # Overall magnetization
    return avg - M*M


if __name__ == '__main__':

    size = 100 # Size of the square lattice
    T = 2 # Temperature in units of epsilon/k
    upperLimit = 1000*size*size # Each dipole is flipped an  average 1000 times
    s = np.zeros((size, size), dtype=int) # Create the lattice as a 2D array
    initialize() # Initialize the array
    
    Mlist = [] # List to store magnetization values
    
    for iteration in range(1, upperLimit): # The main iteration loop
        i = rand.randint(0, size-1) # Choose a random dipole
        j = rand.randint(0, size-1)
        
        Ediff = deltaU(i, j) # Compute the energy change of a
        
        # hypothetical flip
        if Ediff <= 0: # Flip the dipole if energy is reduced
            s[i,j] = -s[i,j]
        else: # Else, the Boltzmann factor gives
            if rand.random() < np.exp(-Ediff/T): # the probability of flipping
                s[i,j] = -s[i,j]
        
        Mlist += [np.abs(np.sum(s))]
                
        if iteration % 100000 == 0:
            print((iteration/upperLimit)*100, "% done")
    
    # print(np.abs(np.sum(s)))

    corrfun = []
    xlist = []
    for r in range(1,int(size/2)+1): # Compute the correlation function
        xlist += [r]
        corrfun += [corr(r)]
            
    fig, ax = plt.subplots() # Initialize the plot
    
    ax.imshow(s, interpolation='nearest') # Plot the final configuration
    # plt.show()
    
    fig, ax = plt.subplot() # and the correlation function
    ax.plot(xlist, corrfun)
    plt.show()
    
