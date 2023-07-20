#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Extract current density field given single-particle density matrix (1DM)
in atomic orbitals (AO). 

@author: Stefano M. Cavalleto, Shichao Sun, Bing Gu

TODO: directly extract 1DM from  Pyscf computations (Done.)

@data: Dec 16, 2021

"""
import numpy
import pyscf
import os
from pyscf import gto, scf, mcscf, tools

import numpy as np
import time
from pyscf import lib
from pyscf.dft import numint
from pyscf import __config__


Nval = 4

# LOAD GEOM
fname='TCD/mgpc_geom.dat'
with open(fname) as f:     # Open the file as f. 
    geom = f.read().splitlines()  # The function readlines() reads the file.

# SETTING UP MOL
mol = gto.M(atom = geom, basis = '6-31G*')
NumBasis = len(mol.ao_labels())
#print('numbasis')
#print(NumBasis)


#%%

#LOAD TDM
#for i in range(1,Nval+1):
#    for j in range(i,Nval+1):
def LoadMOLPROTDM(TDMfname):
    #fname='dm_%d_%d.dat' % (i,j)
    print('Loading '+ TDMfname)
    with open(TDMfname) as f:     # Open the file as f. 
        tdm = f.read().splitlines()  # The function readlines() reads the file.

    #print(tdm)
    #tdm = tdm[2:-1]
    #print(tdm)

    #print(tdm(1))

    tdm = " ".join(tdm).split()
    tdm = numpy.reshape(tdm, (NumBasis, NumBasis)).astype(float)  # change to float
    return tdm

def CreateCube(mol,nx,ny,nz):
    cube=tools.cubegen.Cube(mol, nx, ny, nz) # Creating a cube object
    coords = cube.get_coords() # create a 3d grid to evaluate the basis in real space. chi_mu(r)
    return cube, coords

def WriteCube(cube,rho,outfile):
    cube.write(rho, outfile, comment='Electron density in real space (e/Bohr^3)')


#%%
def eval_nabla_ao(mol,coords):
    nabla_ao_eval = pyscf.gto.eval_gto(mol,'GTOval_ip_sph',coords) # Evaluate nabla |AO(r)> on real space. Output: [3,nx*ny*nz,nao]
    return nabla_ao_eval

def eval_ao(mol,coords):
    ao_eval = pyscf.gto.eval_gto(mol,'GTOval_sph',coords)  # Evaluate |AO(r)> on real space. Output: [nx*ny*nz,nao]
    return ao_eval


def eval_rho_tcurdens(mol,tdm,coords):
    nabla_ao_eval = eval_nabla_ao(mol,coords)
    ao_eval = eval_ao(mol,coords)

    rho_tcurdens = np.array([numpy.einsum('pq,gp,gq->g',(tdm-tdm.T),ao_eval,nabla_ao_eval[idx]).reshape(nx,ny,nz) for idx in np.arange(3)])
    return rho_tcurdens

def eval_rho_tchgdens(mol,tdm,coords):
    ao_eval = eval_ao(mol,coords)
    rho_tchgdens = np.array(numpy.einsum('pq,gp,gq->g',(tdm),ao_eval,ao_eval).reshape(nx,ny,nz))
    return rho_tchgdens

def eval_rho_ao(mol,tdm,ao_eval_bra,ao_eval_ket,coords):
    #ao_eval = eval_ao(mol,coords)
    rho_tchgdens = np.array(numpy.einsum('pq,gp,gq->g',(tdm),ao_eval_bra,ao_eval_ket).reshape(nx,ny,nz))
    return rho_tchgdens


#%%
# electronic structure computations 


# get the 1-particle density matrix


# generate cube file for the current density field operator

ngrid=80;
nx=ngrid;ny=ngrid;nz=ngrid;

[cube, coords] = CreateCube(mol,nx,ny,nz)

arr=[]
arr = [1.0 for i in range(ngrid)] 


nabla_ao_eval = eval_nabla_ao(mol,coords)
ao_eval = eval_ao(mol,coords)
#print(nabla_ao_eval)

#print('shape ',np.shape(nabla_ao_eval))

# number of many-electron states 
nstates = 23

for ii in range(nstates):
    for jj in range(ii):


        TDMfname = 'transitionDM/state'+str(ii)+'tostate'+str(jj)+'tranden.txt'

        tdm = LoadMOLPROTDM(TDMfname)

        #print('transition density matrix',tdm)
        #print('shape',np.shape(tdm))

        intermediate = ao_eval@(tdm.T-tdm) 
        rho_tcurdens = np.array([ np.array([ np.dot( intermediate[igrid,:] , nabla_ao_eval[idx,igrid,:] )  for igrid in np.arange(nx*ny*nz) ]) for idx in np.arange(3)])
        #rho_tcurdens = np.array([numpy.einsum('gq,gq->g',intermediate,nabla_ao_eval[idx]) for idx in np.arange(3)])
        
        #print('shapeof result',np.shape(rho_tcurdens))

        rho_tcurdens = rho_tcurdens.real

        #first coordinate: Jx Jy Jz
        #second: x third: y fourth: z
        #%% int Transition Current Densities
        outfile_tcurdens=['TCD/state'+str(ii)+'state'+str(jj)+'_x.cube','TCD/state'+str(ii)+'state'+str(jj)+'_y.cube','TCD/state'+str(ii)+'state'+str(jj)+'_z.cube']
        for idx in np.arange(3): 
            outfile_tdx=open(outfile_tcurdens[idx],"w")
            np.savetxt( outfile_tdx , rho_tcurdens[idx] ) 
            outfile_tdx.close()



#%% vector plot to be in MATHEMATICA
