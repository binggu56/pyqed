#!/usr/bin/python

############################
# Code by:Erica Chong and Winston Wright
# Last modified by Bing Gu: Mar, 2021
#
# Diagonalize a Hamiltonian matrix
# in the DVR (pseudo-spectral) basis via
# Gauss-Hermite DVR, and print eigenfunction in coordinate
# representation.
#
# For more info on implementation, see John Light and Tucker Carington
# "Discrete Variable Representations and their Utiliazation"
#
############################

from __future__ import division
import math
from math import sqrt
import cmath
import sys
import numpy as np
from  mpmath import *
from scipy import linalg
from scipy.integrate import simps
from scipy.special import h_roots
from time import strftime

from lime.style import subplots

def eig(a):
    evals, evecs = linalg.eig(a)

    idx = evals.argsort()
    evals = evals[idx]
    evecs = evecs[:,idx]
    return evals, evecs

n = 20  # sets both number of grid points for the DVR (i.e. the highest order hermite polynomial in the spectral basis)
mass = 1.0      # electron mass in au
hBar = 1.0 	# au
PIM4 = math.pow(math.pi,(-1/4))
mp.dps = 8 # Sets the (decimal) precision for the mp floats (set this smallest that prevents overflows/div by zero in Herms and Hermsatroot builds)
xscale = 1.6 # factor to scale the coordinates (in AU) by.  Useful if using small n to cover a large system, and very high grid point resolution isn't needed

#print 'n = ' + str(n)
nele = 1  #float(sys.argv[1])
nprot = 1 #float(sys.argv[2])
RQD = 1 #float(sys.argv[3])
l = 0 #float(sys.argv[4])


# outfile = str(int(nele))+'_elect_'+str(int(nprot))+'_prot_'+str(RQD)+'_Bohr_QD_l_'+str(int(l))
# o1 = str( str(outfile) + '_output.txt')
# output4 = open(o1,"w")
# output4.write(strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write('number electron, protons, and QD radius (Bohr), \
#    orbital angular momentum (l):'+str(int(nele))+str(int(nprot))+str(RQD)+str(int(l)))
print("number of DVR points = " + str(n) + '\n')


### DEFINE THE POTENTIAL

# Harmonic oscillator and parabolic doublewell tests
#
def potential(x):
    omega = 1. 		# sqrt force constant
    x *= xscale
# DBL test of coordinate rescaling
    #x = 0.1*x # convert from bohr to nm for units of x
    #return (0.5*mass*omega*omega*x*x)
#  parabolic double well potential:
    return 1.0/4.0*(abs(x/2.0) - 2.0)**2


# def potential(x):
# # Parameters for the potential
#   R = RQD #QD radius in Bohr
#   x = xscale*x
# #  epsOut = 7.56 # THF permitivity
#   epsOut = 1.0 # Vacuum permitivity
#   epsRel =  10.0 # epsilon_r for ZnO
#   epsR = 1 + (epsRel-1)/(1+(1.7)/(R**1.8))
#   if abs(x) < R:
#     return -((nele-1)*x**2)/(2*epsR*R**3)+(nele-1)/(2*epsR*R)+(l**2+l)/(2*x**2)
# #    return -((nele-1)*x**2)/(2*epsR*R**3)+(nele-1)/(2*epsR*R)
#   else:
#     return -(nprot-(nele-1))/(epsOut*abs(x))+(nprot-(nele-1))/(epsOut*R)+(l**2+l)/(2*x**2)
# #    return -(nprot-(nele-1))/(epsOut*abs(x))+(nprot-(nele-1))/(epsOut*R)


# orthonormal set of Hermite polynomials (physicistsi form) with arbitrary precision
def hermite(n,x=0):
  if n == -1:
      return mpf(0)
  if n == 0:
      return mpf(math.pow(math.pi,(-1/4)))

  return mpf(x* sqrt(2/n)*hermite(n-1,x) - sqrt((n-1)/n)*hermite(n-2,x))


# weighting function for Hermite polynomials
def w(x):
  return (np.exp(-x*x))

# Step 0. Setting up the DVR gridpoints and weights

#print "Finding the roots of Hermite polynomials (and associated weights)"
xVals, wVals = h_roots(n,False)

xmax = xVals[-1]
xmin = xVals[0]
#print "xmin, ", xmin*xscale, " xmax, ", xmax*xscale
#print "Evaluating the Hermite polynomials @ roots of the n'th order one"
# Evaluate H0 and H1 on a grid and use forward recursion with the value of the functions at those points.
HermsAtRoots = np.zeros((n,n))*mpf(1.0) # initialize an arbitrary precision array
for a in range(n):
  HermsAtRoots[a,0] = hermite(0,xVals[a])
  HermsAtRoots[a,1] = hermite(1,xVals[a])
i = 2
while i < n:
  for a in range(n):
    HermsAtRoots[a,i] = mpf(xVals[a]* sqrt(2/(i))*HermsAtRoots[a,i-1]\
                            - sqrt((i-1)/(i))*HermsAtRoots[a,i-2])# need arbitrary precisions here since these can get very small (not representable by double-precision floats)
  i = i + 1

# Step 1. Set up the uniformly spaced grid to evaluate the potential on for printing

print("Forming potential matrix in the DVR")
# Step 1.5 Build the potential matrix in the pseudo-spectral (DVR) basis (and print)

potvec = potential(xVals)

OutputV = open("potential.txt", "w")
np.savetxt(OutputV, potvec) # in au
OutputV.close

fig, ax = subplots()
ax.plot(xVals, potvec)
potentialDVR = np.diag(potvec)

# Step 2. Form the kinetic energy matrix directly in pseudo-spectral basis

#print "Forming kinetic matrix in the DVR"

def md2dxn(j,k,roots,n):
  if j==k:
    return (-2.0*(n-1)/3.0)-0.5+(roots[j]**2/3.0)
  elif (j+k)%2!=0:
    return -1*(0.5-2.0/((roots[k]-roots[j])**2))
  else:
    return 0.5-2.0/((roots[k]-roots[j])**2)

kineticDVR = np.zeros((n,n))
for M in range(n):
  for N in range(n):
    kineticDVR[M,N] = (-hBar*hBar/(2.0*mass))*md2dxn(M,N,xVals,n)

# unit conversion for the second derivative in the kinetic energy
kineticDVR = kineticDVR/xscale**2

# Step 3. Form full Hamiltonian Matrix

hamiltonianDVR = potentialDVR + kineticDVR

# Step 4. Diagonalize Hamiltonain Matrix

print("Diagonalizing Hamiltonian in the DVR")

evals, evecs = eig(hamiltonianDVR)


# DBL test of printing without back-transformation to the hermite polynomial basis (uneven grid points)

# OutputGrid = open("gridpoints.txt","w")
# np.savetxt(OutputGrid, xVals) # in au
# OutputGrid.close
print('DVR grid = ', xVals)

# Step 5.  Print out the results (eigenvalues and eigenfunctions)

# OutputEvals = open(str(str(outfile) +"_evals.txt"),"w")
# np.savetxt(OutputEvals, evals)
# OutputEvals.close

print(evals)

for i in range(n):
  if wVals[i]>1.e-300:
    evecs[i,:] = evecs[i,:] * sqrt(w(xVals[i])/wVals[i])


#print 'Integral of ground state wavefunction mod. sq.: '+str(simps(evecs[:,0]**2/xscale,xVals*xscale))

# OutputEfunct = open(str(str(outfile) +"_efunctions.txt"),"w")
# np.savetxt(OutputEfunct, evecs) # in au
# OutputEfunct.close

fig, ax = subplots()
for j in range(5):
    ax.plot(xVals, evecs[:,j])

# DBL print the RDF, assuming the 1D potential here is a radial component of a 3d problem

# rdf = np.zeros(n)
# for i in range(n):
#   rdf[i] = evecs[i,0]**2*(xVals[i]*xscale)**2
# rdfnorm = simps(rdf,xVals*xscale)
# OutputRDF = open(str(str(outfile) +"_RDF_of_groundstate.txt"),"w")
# np.savetxt(OutputRDF,rdf/rdfnorm)
# OutputRDF.close


# output4.write("Printing data  " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write("xvals" + '\n' + "   ".join(str(item) for item in xVals) + '\n')
# output4.write("wVals" + '\n' + "   ".join(str(item) for item in wVals) + '\n')

# output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write("potentialDVR" + '\n')
# for row in potentialDVR:
#   output4.write('  '.join(str(elem) for elem in row) + '\n')

# output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write('\n' + "kineticDVR" + '\n')
# for row in kineticDVR:
#   output4.write('  '.join(str(elem) for elem in row) + '\n')

# output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write('\n' + "hamiltonianDVR" + '\n')
# for row in hamiltonianDVR:
#   output4.write('  '.join(str(elem) for elem in row) + '\n')
# output4.write('\n' + "eigenvalues" + '\n' + "   ".join(str(item) for item in evals) + '\n')

# output4.write('\n' + strftime("%Y-%m-%d %H:%M:%S") + '\n')
# output4.write('\n' + "eigenvectors" + '\n')
# for row in evecs:
#   output4.write('  '.join(str(elem) for elem in row) + '\n')

print("Finished!!  " + strftime("%Y-%m-%d %H:%M:%S") + '\n')
