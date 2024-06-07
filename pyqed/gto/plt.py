#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 23:39:17 2017

@author: binggu
"""

import numpy as np
from FCI import *
import matplotlib.pyplot as plt

def set_style():
    #plt.style.use(['classic'])
    #print(plt.style.available)
#    font = {'family' : 'Times',
#        'weight' : 'bold',
#        'size'   : 22}

    plt.rc('font',**{'family':'sans-serif','sans-serif':['Helvetica'],
                     'weight':'normal'})
    #plt.rc('text', usetex=True)

    SIZE = 12

    plt.rc('xtick', labelsize=SIZE)
    plt.rc('ytick', labelsize=SIZE)

    #plt.rc('font', size=SIZE, weight='bold')  # controls default text sizes
    plt.rc('axes', titlesize=SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=SIZE)  # fontsize of the x any y labels
    plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SIZE)  # legend fontsize
    plt.rc('figure', titlesize=SIZE)  # # size of the figure title
    plt.rc('axes', linewidth=1)

    plt.rcParams['axes.labelweight'] = 'normal'




    # the axes attributes need to be set before the call to subplot
    plt.rc('xtick.major', size=6, pad=7)
    plt.rc('ytick.major', size=6, pad=7)

    # using aliases for color, linestyle and linewidth; gray, solid, thick
    plt.rc('grid', c='0.5', ls='-', lw=1)
    plt.rc('lines', lw=1, color='g')
    plt.rc('savefig',dpi=100)

def h2_pes(method):

    if method == 'FCI':
        file = open("h2_pes_FCI.dat", "w")
        for r in np.linspace(0.7, 20,40):
            electronic_energy = configuration_interaction([0., r], [1, 1])
            file.write( '{} {} {} \n'.format(r, *electronic_energy))

        file.close()

    elif method == 'HF':
        file = open("h2_pes_HF.dat", "w")
        for r in np.linspace(0.7, 20,40):
            total_energy = hartree_fock([0., r], [1, 1])
            file.write( '{} {} \n'.format(r, total_energy))

        file.close()


def plt_PES():

    fig, ax = plt.subplots()
    R, PES0, PES1 = np.genfromtxt('h2_pes_FCI.dat',unpack=True)
    ax.plot(R, PES0, 'o-',  label='FCI')
    ax.plot(R, PES1, 'o-',  label='FCI')

    R, PES_HF = np.genfromtxt('h2_pes_HF.dat',unpack=True)

    ax.plot(R, PES_HF, 'o--', label='HF')

    ax.set_xlim(0,10)
    ax.legend()
    ax.set_xlabel('R (Bohr)')
    ax.set_ylabel('Energy (a.u.)')
    plt.savefig('H2.eps',dpi=1200)
    plt.draw()


set_style()

plt_PES()
plt.show()