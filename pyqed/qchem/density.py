# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 10:09:59 2022

Compute the charge and current density in real space

@author: Bing
"""

import numpy as np
import pyscf
from pyscf import tools


def eval_nabla_ao(mol, coords):
    nabla_ao_eval = pyscf.gto.eval_gto(mol,'GTOval_ip_sph',coords)
    # Evaluate nabla |AO(r)> on real space. Output: [3,nx*ny*nz,nao]
    return nabla_ao_eval


class Analyze:
    """
    analyze TDA results; compute charge density, current density in 3D
    """
    def __init__(self, td=None):
        """
        visualize the time-dependent charge density and current density

        Parameters
        ----------
        mol : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        self.td = td
        self.mf = td._scf
        self.nx = None
        self.ny = None
        self.nz = None
        self.mol = td.mol
        self.cube = None
        self.coords = None
        self.ngrids = None
        self._ao = None
        self._nabla_ao = None


    def CreateCube(self, nx=80,ny=80,nz=80):

        cube = tools.cubegen.Cube(self.mol, nx, ny, nz) # Creating a cube object
        # create a 3d grid to evaluate the basis in real space. chi_mu(r)
        self.coords = cube.get_coords() # [ngrids, 3]

        self.ngrids = cube.get_ngrids()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.cube = cube
        return

    def build_dm(self, state_id):
        '''
        Taking the TDA amplitudes as the CIS coefficients, calculate the density
        matrix in AO basis of the excited states

        Parameters
        ----------
        state_id: int
            state number, 1 is the first excited state.
        '''
        td = self.td

        if state_id == 0: # ground state
            return td._scf.make_rdm1()

        else: # excited states

            cis_t1 = td.xy[state_id-1][0] # [no, nv]
            dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
            dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

            # The ground state density matrix in mo_basis
            mf = td._scf
            dm = np.diag(mf.mo_occ)

            # Add CIS contribution
            nocc = cis_t1.shape[0]
            # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
            # include the spin-down contribution
            dm[:nocc,:nocc] += dm_oo * 2
            dm[nocc:,nocc:] += dm_vv * 2

            # Transform density matrix to AO basis
            mo = mf.mo_coeff
            dm = np.einsum('pi,ij,qj->pq', mo, dm, mo.conj())
        return dm


    def transition_current_density(self, state=1):
        """
        transition_current_density

        Parameters
        ----------
        state : TYPE, optional
            DESCRIPTION. The default is 1.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        if self.cube is None:
            self.CreateCube()

        if self._ao is None:
            self.eval_ao()

        if self._nabla_ao is None:
            self.eval_nabla_ao()

        # compute the transition density matrix
        tdm = self.build_tdm(state-1)

        return transition_current_density(self.cube, tdm, self._ao, self._nabla_ao)

    def build_tcd_full(self, tdm):
        """
        Compute the transition charge density between all states from the full
        transition density matrix
        :math:` T_{ia}^{JI} = \langle \Psi_J | i a^\dagger |\Psi_I \rangle`

        For currents, the diagonal elements (I = J) do not contribute for time-reversal
        invariant systems.

        Parameters
        ----------
        tdm : ndarray [nao, nao, nstates, nstates]
            full transition density matrix in AO

        Returns
        -------
        tcd : ndarray [3, nx*ny*nz, nstates, nstates]
            The full transition current density matrix. The last dimension refers to
            x, y, z.

        """
        nstates = tdm.shape[-1] # number of all states
        # nmo = nocc + nvir
        ngrids = np.size(self.coords)
        J = np.zeros((3, ngrids, nstates, nstates))

        nabla_ao = self.nabla_ao()
        ao_eval = self.eval_ao()


        for ii in range(nstates):
            for jj in range(ii):


                # TDMfname = 'transitionDM/state'+str(ii)+'tostate'+str(jj)+'tranden.txt'

                # tdm = LoadMOLPROTDM(TDMfname)
                _tdm = tdm[:, :, jj, ii]
                #print('transition density matrix',tdm)
                #print('shape',np.shape(tdm))

                intermediate = ao_eval@(_tdm.T - _tdm) # this tdm is for T^{JI}
                # rho_tcurdens = np.array([ np.array([ np.dot(intermediate[igrid,:] , \
                #                                             nabla_ao_eval[idx,igrid,:] )\
                    #for igrid in np.arange(nx*ny*nz) ]) for idx in np.arange(3)])
                rho_tcurdens = np.einsum('gq, agq -> ag', intermediate, nabla_ao)

                J[:, :, jj, ii] = rho_tcurdens.real

                #first coordinate: Jx Jy Jz
                #second: x third: y fourth: z
                # int Transition Current Densities
                # outfile_tcurdens=['TCD/state'+str(ii)+'state'+str(jj)+'_x.cube',
                #                   'TCD/state'+str(ii)+'state'+str(jj)+'_y.cube',
                #                   'TCD/state'+str(ii)+'state'+str(jj)+'_z.cube']
                # for idx in np.arange(3):
                #     outfile_tdx=open(outfile_tcurdens[idx],"w")
                #     np.savetxt( outfile_tdx , rho_tcurdens[idx] )
                #     outfile_tdx.close()
        return J

    def density(self, state=0, do_diff=True, output=None):
        """
        real-space image of the charge density for a given state.
        ..math::
            n(\mathbf r) = D_{\mu \nu} \chi_\mu(\bf r) \chi_\nu(\bf r)
        where D is the one-electron density matrix.

        Parameters
        ----------
        state : int, optional
            state id. The default is 0 (ground state).
        coeff : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if self._ao is None:
            self.eval_ao() # [ngrids, nao]
        ao = self._ao

        dm = self.build_dm(state)
        nx, ny, nz = self.nx, self.ny, self.nz

        # for real AOs
        d = np.einsum('gu, uv, gv -> g', ao, dm, ao).reshape(nx, ny, nz)


        if do_diff:
            dm0 = self.build_dm(0)
            d0 = np.einsum('gu, uv, gv -> g', ao, dm0, ao).reshape(nx, ny, nz)

            d = d - d0

        if output is not None: np.save(output, d)

        from mayavi.mlab import contour3d
        contour3d(d, contours=4, transparent=True, opacity=0.5, colormap='viridis')

        return d

    # def realtime_density(self, coeff):
    #     # generate density from expansion coefficients
    #     pass

    def eval_nabla_ao(self):
        """
        Evaluate nabla |AO(r)> on real space.

        Returns
        ---------
        Output: [3,nx*ny*nz,nao]
        """
        mol = self.mol
        nabla_ao_eval = pyscf.gto.eval_gto(mol,'GTOval_ip_sph', self.coords)

        self._nabla_ao = nabla_ao_eval

        return nabla_ao_eval


    def eval_ao(self):
        """
        Evaluate |AO(r)> on real space.

        Returns
        -------
        Output: [nx*ny*nz,nao]
        """
        mol = self.mol
        if self.coords is None:
            raise ValueError('Call CreateCube() first to create Cube.')

        ao_eval = pyscf.gto.eval_gto(mol,'GTOval_sph',self.coords)
        self._ao = ao_eval
        return ao_eval

    def transition_density_matrix(self, state):
        return build_tdm(self.td.xy[state][0], self.mf.mo_coeff)

    def build_tdm(self, state):
        """
        Deprecated. Use transition_density_matrix()

        Parameters
        ----------
        state : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return build_tdm(self.td.xy[state][0], self.mf.mo_coeff)


def build_rdm1(mo_coeff, mo_occ, x=None):
    '''One-particle density matrix in AO representation

    Taking the TDA amplitudes as the CIS coefficients, calculate the density
    matrix in AO basis of the excited states

    Parameters
    ----------
    state_id: int
        state number, 1 is the first excited state.

    Args:
        mo_coeff : 2D ndarray
            Orbital coefficients. Each column is one orbital.
        mo_occ : 1D ndarray
            Occupancy
        x: TDA coeff
            if x is None, returns the ground state rdm.
    '''
    if x is None:

        mocc = mo_coeff[:,mo_occ>0]

        dm = np.dot(mocc*mo_occ[mo_occ>0], mocc.conj().T)

    else: # excited state rdm

        cis_t1 = x # [no, nv]
        dm_oo = -np.einsum('ia,ka->ik', cis_t1.conj(), cis_t1)
        dm_vv = np.einsum('ia,ic->ac', cis_t1, cis_t1.conj())

        # The ground state density matrix in mo_basis
        dm = np.diag(mo_occ)

        # Add CIS contribution
        nocc = cis_t1.shape[0]
        # Note that dm_oo and dm_vv correspond to spin-up contribution. "*2" to
        # include the spin-down contribution
        dm[:nocc,:nocc] += dm_oo * 2
        dm[nocc:,nocc:] += dm_vv * 2

        # Transform density matrix to AO basis
        dm = np.einsum('pi,ij,qj->pq', mo_coeff, dm, mo_coeff.conj())

    return dm


def build_tdm(x, mo_coeff):
    """
    Compute the transition density matrix in AO basis for TDA

    .. math::
        T_{\mu \nu} = \langle \Phi_n |  \nu^\dag \mu |\Phi_0\rangle

    Parameters
    ----------
    x : ndarray [nocc, nvir]
        TDA coeff
    mo_coeff: ndarray
        mo coeff

    Returns
    -------
    None.

    """

    nocc, nvir = x.shape

    return np.einsum('ui, ia, va ->uv', mo_coeff[:, :nocc], \
                     x, mo_coeff[:, nocc:].conj())


def density(cube, rdm, ao, mo_coeff=None, representation='ao',\
            output='density'):
    """
    real-space image of the charge density for a given state.

    ..math::
        n(\mathbf r) = D_{\mu \nu} \chi_\mu(\bf r) \chi_\nu(\bf r)

    where D is the one-electron density matrix in AO.

    Parameters
    ----------
    state : int, optional
        state id. The default is 0 (ground state).
    coeff : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """


    nx, ny, nz = cube.nx, cube.ny, cube.nz

    # for real AOs
    if representation == 'ao':

        d = np.einsum('gu, uv, gv -> g', ao, rdm, ao).reshape(nx, ny, nz)

    else:

        mo = np.einsum('gu, up -> qp', ao, mo_coeff)

        d = np.einsum('gu, uv, gv -> g', mo, rdm, mo).reshape(nx, ny, nz)


    # if do_diff:
    #     dm0 = self.build_dm(0)
    #     d0 = np.einsum('gu, uv, gv -> g', ao, dm0, ao).reshape(nx, ny, nz)

    #     d = d - d0

    np.save(output, d)

    return d


def transition_charge_density(cube, tdm, ao):
    """
    

    Parameters
    ----------
    tdm : TYPE
        DESCRIPTION.
    ao : AOs in real space 

    Returns
    -------
    None.

    """
    nx, ny, nz = cube.nx, cube.ny, cube.nz

    # \sigma(r) = \braket{\Phi_n | \hat{\sigma}(\bf r)|\Phi_0}
    rho = np.einsum('gu, uv, gv -> g', ao, tdm, ao.conj()).reshape(nx, ny, nz)

    return rho

def transition_current_density(cube, tdm, ao, nabla_ao, mo_coeff=None, basis='ao',\
            output='tcd'):
    """
    real-space image of the transition current density for a given state.

    .. math::

        \bf j(\mathbf r) = \braket{\Phi_n| \hat{j}(\bf r) |\Phi_0}
            = T_{\mu \nu} \bf j_{\mu \nu}(\bf r)

            \chi_\mu(\bf r) \chi_\nu(\bf r)


    where D is the one-electron density matrix in AO.

    Parameters
    ----------
    state : int, optional
        state id. The default is 0 (ground state).
    tdm : ndarray
        TDM in AO
    coeff : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """
    if basis == 'mo':
        raise NotImplementedError('Please provide density matrix in AO representation.')

    nx, ny, nz = cube.nx, cube.ny, cube.nz

    # compute \chi_\mu(r) \Gamma_{\mu \nu}

    tmp = ao @ (tdm - tdm.T) # [ng, nao]


    # j = 0.5j * np.einsum('xgu, gu -> xg', nabla_ao, tmp)
    # j = j.reshape(3, nx, ny, nz)

    tcd = [0.5j * np.einsum('gu, gu -> g', nabla_ao[i], tmp).reshape(nx, ny, nz)
           for i in range(3)]

    # d = np.einsum('gu, uv, gv -> g', ao, rdm, ao).reshape(nx, ny, nz)

    np.save(output, tcd)

    return tcd


if __name__ == '__main__':

    from pyscf import gto, dft, scf, tddft
    from pyqed.style import vector_field

    mol = gto.M(atom =
    '''
    H      1.2194     -0.1652      2.1600
    C      0.6825     -0.0924      1.2087
    C     -0.7075     -0.0352      1.1973
    H     -1.2644     -0.0630      2.1393
    C     -1.3898      0.0572     -0.0114
    H     -2.4836      0.1021     -0.0204
    C     -0.6824      0.0925     -1.2088
    H     -1.2194      0.1652     -2.1599
    C      0.7075      0.0352     -1.1973
    H      1.2641      0.0628     -2.1395
    C      1.3899     -0.0572      0.0114
    H      2.4836     -0.1022      0.0205
    ''',
    basis = '321g',
    symmetry=True)

    mf = dft.RKS(mol)
    mf.xc = 'b3lyp'
    mf.kernel()

    td = tddft.TDA(mf)
    td.run(nstates=3)

    # TCD
    analyzer = Analyze(td)
    J = analyzer.transition_current_density()


