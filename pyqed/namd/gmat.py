#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:06:22 2022

Compute the G-matrix

@author: bing
"""


from pyqed.coordinates import readxyz, eckart
from numpy import sqrt
from scipy.linalg import inv

def buildGmat_linear(geom, mass, coord, ndim=2):
    """
    Input
    =====
    geom: 2darray
        input reference geometry (3,N) for Eckart frame
    mass: masses according to geom (1,N) in [u]
    coord: Struct containing:
    %       .ndims: number of dimensions
    %       .dimn: (3,N) vector for n-th dimension

    Return
    ======
    %gmat: G-matrix
    """
    # Convert mass tu a.u.
    mass=mass*1822.89

    # initialize inverse G-Matrix
    gmat=zeros(coord.ndims,coord.ndims);

    dq=0.001 # dq for derivative


    for i in range(ndim):

        veci=eval(sprintf('coord.dim%d',i)) #get vector for dimension i

        # central difference derivative for dimension i
        geomi1 = geom - dq*veci
        geomi2 = geom + dq*veci
        dxdqi=(geomi1-geomi2)/(2*dq)

        for j in range(ndim):

            vecj=eval(sprintf('coord.dim%d',j)) # get vector for dimension j

            # central difference derivative for dimension j
            geomj1=geom-dq*vecj
            geomj2=geom+dq*vecja
            dxdqj=(geomj1-geomj2)/(2*dq)

            # Assign inverse  G-Matrix element
            gmat[i, j] = sum(mass*(dxdqi*dxdqj))


    gmat = inv(gmat) #invert to get G-matrix


    return gmat


def buildG_curvilinear(reference, geom, ndim=2, frame='eckart'):
    """
    Compute G matrix for curvilinear coordinates using Eckart frame.

    Useful for small amplitude motion e.g. normal vibrations.

    Parameters
    ----------
    reference : TYPE
        DESCRIPTION.
    geom : ndarray [nx, 3, Natom] or [nx, ny, 3, Natom]
        all xyz of geometries in a grid

    Returns
    -------
    None.

    """
    assert(geom.ndim == 4)

    if frame != 'eckart':
        raise ValueError('Molecular frame has to be eckart.')

    if ndim > 2:
        raise NotImplementedError('{} dimensions of reduced coordinates\
                                  not supported. Try 2.'.format(ndim))
    # initialize inverse G-Matrix
    nx, ny = geom.shape[:2] # grid points
    gmat=np.zeros(nx, ny, ndim, ndim)

    dq=0.001 # dq for derivative

    if ndim == 1:
        nx = geom.shape[0]
        gmat = np.zeros(nx)

        for i in range(1, nx-1):

            geomi1 = eckart(reference, geom[n+1, m], mass)
            geomi2 = eckart(reference, geom[n-1, m], mass)
            dxdq = (geomi1-geomi2)/(2*dq)

            gmat[i] = sum(mass*(dxdqi*dxdqj))

        gmat = 1./gmat

    elif ndim == 2:


        for n in range(1, nx-1):
            for m in range(1, ny-1):

                for i in range(ndim):

                    veci=eval(sprintf('coord.dim%d',i)) #get vector for dimension i

                    # central difference derivative for dimension i
                    geomi1 = eckart(reference, geom[n+1, m], mass)
                    geomi2 = eckart(reference, geom[n-1, m], mass) #
                    dxdqi=(geomi1-geomi2)/(2*dq)

                    for j in range(ndim):

                        vecj=eval(sprintf('coord.dim%d',j)) # get vector for dimension j

                        # central difference derivative for dimension j
                        geomj1=eckart(reference, geom[n, m+1], mass)
                        geomj2=eckart(reference, geom[n, m-1], mass)

                        dxdqj=(geomj1-geomj2)/(2*dq)

                        # Assign inverse  G-Matrix element
                        gmat[i, j] = sum(mass*(dxdqi*dxdqj))

        # 4 boundaries

        gmat = inv(gmat) #invert to get G-matrix

    return gmat


s0min,mass = readxyz('s0min.xyz',2)

s0min = s0min/0.5291772083

s2min = readXyz('s2min.xyz',2)
s2min = eckart(s0min,s2min,mass[1,:])/0.5291772083

coin = readxyz('s2s1coin.xyz',2)
coin = eckart(s0min,coin,mass[1,:])/0.5291772083;

v1 = coin - s0min
v1 = v1 / sqrt( sum(sum(v1*v1)) )

v2 = s2min - s0min
v2 = v2 - sum(sum(v2*v1))*v1
v2 = v2 / sqrt( sum(sum(v2*v2)) )

coord=struct
coord.ndims = 2
coord.dim1 = v1
coord.dim2 = v2

gmatu = BuildGmatND(s0min,mass(1,:),coord)


# Points necessary in each dimension
# q1 = (-0.5:0.1:1)./0.5291772083;
# q2 = (-0.5:0.1:0.5)./0.5291772083;

# Tmax = 3.5 ;                % potential energy gained by the wp [eV] x 1.5
# Tmax = Tmax / 27.21;        % [eV] to [au]
# k1m = sqrt((2*Tmax) ./ ( gmatu(1,1) - gmatu(1,2).^2./gmatu(2,2) ));
# k2m = sqrt((2*Tmax) ./ ( gmatu(2,2) - gmatu(1,2).^2./gmatu(1,1) ));
# dx1 = pi / k1m;
# dx2 = pi / k2m;
# Nx1 = (((q1(end)-q1(1))) / dx1) + 1
# Nx2 = (((q2(end)-q2(1))) / dx2) + 1
