#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 23:25:51 2017


Ehrenfest dynamics for model Hamiltonians

@author: binggu

@status: not finished, work on single Ehrenfest trajectory first
"""


import numpy as np
import numba
import sys
import math
import tqdm
from copy import deepcopy
from pyqed import au2k, au2angstrom, au2wavenumber, ket2dm
from opt_einsum import contract
from scipy.linalg import expm
from pyqed import Molecule



def M1mat(a, Nb):

    M1 = np.zeros((Nb,Nb))

    for m in range(Nb-1):
        M1[m,m+1] = np.sqrt(float(m+1)/2.0/a)

    M1 = Sym(M1)

    return M1

def M2mat(a, Nb):

    M2 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M2[m,m] = (float(m) + 0.5)/a

    if Nb > 1:
        for m in range(Nb-2):
            M2[m,m+2] = np.sqrt(float((m+1)*(m+2)))/2.0/a

    M2 = Sym(M2)

    return M2

def M3mat(a, Nb):

    M3 = np.zeros((Nb,Nb))

    for m in range(Nb-1):
        M3[m,m+1] = 3.0 * (float(m+1)/2./a)**1.5

    if Nb > 2:
        for m in range(Nb-3):
            M3[m,m+3] = np.sqrt(float((m+1)*(m+2)*(m+3))) / (2.0*a)**1.5

    M3 = Sym(M3)

    return M3

def M4mat(a, Nb):

    M4 = np.zeros((Nb,Nb))

    for m in range(Nb):
        M4[m,m] =  float(3.0 * m**2 + 3.0 * (m+1)**2) / (2.*a)**2

    if Nb > 1:
        for m in range(Nb-2):
            M4[m,m+2] = (4.0*m + 6.0) * np.sqrt(float((m+1)*(m+2))) / (2.*a)**2

    if Nb > 3:
        for m in range(Nb-4):
            M4[m,m+4] = np.sqrt(float((m+1)*(m+2)*(m+3)*(m+4))) / (2.0*a)**2

    M4 = Sym(M4)

    if Nb > 1:
        if not M4[0,1] == M4[1,0]:
            print(M4)
            print('\n ERROR: Not symmetric matrix M4.\n')
            sys.exit()
    return M4


def Hermite(x, Nb):
    """
    Hermite polynomials

    Parameters
    ----------
    x : TYPE
        DESCRIPTION.
    Nb : TYPE
        DESCRIPTION.

    Returns
    -------
    H : TYPE
        DESCRIPTION.

    """

    cons = np.array([1. / np.sqrt(float(2**n) * float(math.factorial(n))) for n in range(Nb)])

    H = []
    H.append(1.0)
    H.append( x * 2.0 )
    if Nb > 2:
        for n in range(2,Nb):
            Hn = 2.0 * x * H[n-1] - 2.0*(n-1) * H[n-2]
            H.append(Hn)

    for n in range(Nb):
        H[n] = H[n]*cons[n]

    return H


def Vx(x):

    g = 0.1
    return  x**2/2.0 + g * x**4 / 4.0

def Kmat(alpha,pAve, Nb, mass=1):

    K = np.zeros((Nb,Nb),dtype=complex)

    ar = alpha.real

    for j in range(Nb):
        K[j,j] = np.abs(alpha)**2 / ar * (2. * j + 1.)/2. +  pAve**2

    for j in range(1,Nb):
        K[j-1,j] = -1j*np.conj(alpha) * pAve * np.sqrt(2. * j / ar)
        K[j,j-1] = np.conj(K[j-1,j])

    if Nb > 2:
        for j in range(2,Nb):
            K[j-2,j] = - np.sqrt(float((j-1)*j)) * np.conj(alpha)**2 / 2. / ar
            K[j,j-2] = np.conj(K[j-2,j])


    #K[0,0] = np.abs(alpha)**2/alpha.real / 2. + pAve**2
    #K[1,1] = np.abs(alpha)**2/alpha.real * 3.0 / 2. + pAve**2

    #K[0,1] = -1j*np.conj(alpha) * pAve * np.sqrt(2.*j/alpha.real)
    #K[1,0] = np.conj(K[0,1])
    K = K / (2.* mass)

    return K

def Sym(V):
    n = V.shape[-1]

    for i in range(n):
        for j in range(i):
            V[i,j] = V[j,i]
    return V

# @numba.autojit
def Vint(x,y):
    """
    interaction potential between x and y
    """

    PES = 'HO'

    if PES == 'Morse':

        a, x0 = 1.02, 1.4
        De = 0.176 / 100.0

        d = (1.0-np.exp(-a*x))

        v0 = De*d**2

        dv = 2. * De * d * a * np.exp(-a*x)

    elif PES == 'HO':

        v0 = x**2/2.0  + y**2/2.0


    elif PES == 'AHO':

        eps = 0.4

        v0 = x**2/2.0 + eps * x**4/4.0
        dv = x + eps * x**3
        #ddv = 2.0 * De * (-d*np.exp(-a*((x-x0)))*a**2 + (np.exp(-a*(x-x0)))**2*a**2)

#    elif PES == 'pH2':
#
#        dx = 1e-4
#
#        v0 = np.zeros(Ntraj)
#        dv = np.zeros(Ntraj)
#
#        for i in range(Ntraj):
#            v0[i] = vpot(x[i])
#            dv[i] = ( vpot(x[i] + dx) - v0[i])/dx

    return v0

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle(v1, v2):
    """
    Returns the angle in radians between vectors 'v1' and 'v2'

    Parameters
    ----------
    v1 : TYPE
        DESCRIPTION.
    v2 : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))



def ground(x):
    return 0.5 * np.sum(x**2), x

def excited(x):
    return 0.5 * np.sum((x-1.0)**2) + 1.0, x - 1.0

# @numba.autojit
# def mean_field_force(y,c):

#     V0, dV0 = ground(y)
#     V1, dV1 = excited(y)

#     Vmf = abs(c[:,0])**2 * V0 + abs(c[:, 1])**2 * V1
#     dVmf = abs(c[:, 0])**2 * dV0 + abs(c[:, 1])**2 * dV1

#     return Vmf, dVmf


# class Ehrenfest:
#     def __init__(self, ntraj, ndim, nstates):
#         self.ntraj = ntraj
#         self.ndim = ndim
#         self.nstates = nstates
#         self.c = np.zeros((ntraj,nstates),dtype=np.complex128)

#         self.x = None # nuclear position
#         self.p = None # nuclear momentum
#         self.w = None # weight of each trajectory

#     def sample(self, temperature=300, unit='K'):

#         if unit == 'K':
#             temperature = temperature/au2k
#         elif unit == 'au':
#             temperature = temperature
#         else:
#             raise ValueError(f"Invalid unit: {unit}")

#         self.x = np.random.randn(self.ntraj, self.ndim)
#         self.x = self.x / np.sqrt(2.0 * self.ax) + self.x0

#         self.p = np.zeros(self.ntraj, self.ndim)

#         self.w = np.array([1./self.ntraj]*self.ntraj)

#     def run(self, dt=0.002, nt=200):
#         pass

class EhrenfestTrajectory:
    def __init__(self, x, p, c, mass=None, energy=None, grad=None, nac=None):
        self.x = x
        self.p = p
        if mass is not None:
            self.v = p/mass
        else:
            self.v = None

        self.c = c
        self.nac = nac
        self.energy = energy
        self.grad = grad

        self.force = None

        self.p_prev = None # p(t-dt) momentum as previous time step
        self.nac_prev = None # NAC previous time step

    def rdm(self):
        return ket2dm(self.c)

class TDDFTTrajectory(EhrenfestTrajectory, Molecule):
    def __init__(self, atom_coords, p, c, *args, **kwargs):
        atom_coords = np.asarray(atom_coords, dtype=float)
        if atom_coords.ndim != 2 or atom_coords.shape[1] != 3:
            raise ValueError("atom_coords must have shape (natom, 3).")

        self.natom = atom_coords.shape[0]
        super().__init__(atom_coords.reshape(-1).copy(),
                         np.asarray(p, dtype=float).reshape(-1).copy(),
                         c, *args, **kwargs)

    @property
    def atom_coords(self):
        return self.x.reshape(self.natom, 3)

    @atom_coords.setter
    def atom_coords(self, coords):
        coords = np.asarray(coords, dtype=float)
        if coords.shape != (self.natom, 3):
            raise ValueError(f"atom_coords must have shape ({self.natom}, 3), got {coords.shape}.")
        self.x = coords.reshape(-1).copy()




    # def evolve_x(self, dt):
    #     pass

class TDDFTDriver:
    def __init__(
        self,
        mol,
        nstates,
        xc='lda,vwn',
        nac_method='none',
        point_charge_coords=None,
        point_charges=None,
    ):
        """
        Thin backend adapter for TDDFT state data used by ``TDDFTEhrenfest``.

        Public responsibilities are:
        - ``evaluate(coords)`` / ``single_point(coords)``
        - ``as_scanner()``
        - ``normal_modes()``

        ``point_charge_coords`` and ``point_charges`` provide a fixed MM
        electrostatic embedding bath in Bohr and electron-charge units.
        """
        self.mol = mol
        self.xc = xc
        self.nstates = nstates
        self.nexc = max(0, nstates - 1)
        self.fd_step = 1e-3
        self.nac_fd_step = 1e-3
        self.nac_method = nac_method
        self.point_charge_coords, self.point_charges = _as_point_charges(
            point_charge_coords,
            point_charges,
        )
        self.backend = None
        self.ks = None
        self.td = None

        self._initialize_backend()

    def _initialize_backend(self):
        if hasattr(self.mol, 'apply'):
            self.backend = 'pyscf'
            self.ks, self.td = self._build_pyscf_point(self.mol)
            return

        if hasattr(self.mol, 'build'):
            self.backend = 'pyqed'
            if getattr(self.mol, 'nao', None) is None or getattr(self.mol, '_bas', None) is None:
                self.mol.build()
            self.ks, self.td = self._build_pyqed_point(self.mol)
            return

        raise TypeError("TDDFTDriver expects a PySCF Molecule or a pyqed Molecule.")
    
    def _reshape_coords(self, coords):
        coords = np.asarray(coords, dtype=float)
        return coords.reshape(_get_natom(self.mol), 3)

    def _copy_with_coords(self, coords):
        coords = self._reshape_coords(coords)
        if self.backend == 'pyscf':
            mol = self.mol.copy()
            mol.set_geom_(coords, unit='Bohr')
            return mol

        mol = Molecule(
            atom=deepcopy(self.mol.atom),
            charge=self.mol.charge,
            spin=self.mol.spin,
            basis=self.mol.basis,
            unit='bohr',
        )
        mol.set_geom(coords)
        mol.build()
        return mol

    def _build_pyscf_point(self, mol):
        ks = mol.RKS()
        if self.point_charge_coords is not None:
            from pyscf import qmmm

            ks = qmmm.mm_charge(
                ks,
                self.point_charge_coords,
                self.point_charges,
                unit='Bohr',
            )
        ks.xc = self.xc
        ks.kernel()
        td = ks.apply("TDRKS")
        td.nstates = self.nexc
        if self.nexc > 0:
            td.kernel()
        else:
            td = None
        return ks, td

    def _build_pyqed_point(self, mol):
        from pyqed.qchem.dft import RKS

        ks = RKS(mol, xc=self.xc)
        if self.point_charge_coords is not None:
            from pyqed.qchem import embed_point_charges

            embedded = embed_point_charges(
                ks,
                self.point_charge_coords,
                self.point_charges,
                run_kwargs={'verbose': 0},
            )
            embedded.run()
            ks = embedded.mf
        else:
            ks.run(verbose=0)
        td = ks.TDDFT().run(nstates=self.nexc) if self.nexc > 0 else None
        return ks, td

    def _build_state_point(self, coords):
        coords = self._reshape_coords(coords)
        mol = self._copy_with_coords(coords)

        if self.backend == 'pyscf':
            ks, td = self._build_pyscf_point(mol)
        else:
            ks, td = self._build_pyqed_point(mol)

        self.ks = ks
        self.td = td
        return mol, ks, td

    def _state_energies(self, ks, td):
        energies = np.empty(self.nstates, dtype=float)
        energies[0] = float(ks.e_tot)
        if self.nexc > 0:
            ex = np.asarray(td.e, dtype=float)
            if ex.shape[0] < self.nexc:
                raise ValueError(
                    f"Requested {self.nexc} excited states but TDDFT returned {ex.shape[0]}."
                )
            energies[1:] = energies[0] + ex[:self.nexc]
        return energies

    def _pyscf_gradients(self, ks, td):
        grads = np.zeros((self.nstates, _get_natom(self.mol) * 3), dtype=float)
        grads[0] = np.asarray(ks.nuc_grad_method().kernel(), dtype=float).reshape(-1)
        for n in range(self.nexc):
            grads[n + 1] = np.asarray(
                td.nuc_grad_method().kernel(state=n + 1), dtype=float
            ).reshape(-1)
        return grads

    def _finite_difference_gradients(self, coords):
        coords = self._reshape_coords(coords)
        natm = coords.shape[0]
        ndof = 3 * natm
        grads = np.zeros((self.nstates, ndof), dtype=float)

        for a in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[a] = self.fd_step
            _, ks_p, td_p = self._build_state_point(coords.reshape(-1) + disp)
            _, ks_m, td_m = self._build_state_point(coords.reshape(-1) - disp)
            e_p = self._state_energies(ks_p, td_p)
            e_m = self._state_energies(ks_m, td_m)
            grads[:, a] = (e_p - e_m) / (2.0 * self.fd_step)

        return grads

    def _zero_nac(self, coords):
        return np.zeros((self.nstates, self.nstates, self._reshape_coords(coords).size), dtype=float)

    def _state_vectors(self, ks, td):
        try:
            from pyscf import ci as pyscf_ci
        except Exception as exc:
            raise NotImplementedError(
                "Overlap-based NACs require a local PySCF installation."
            ) from exc

        mo_occ = np.asarray(ks.mo_occ)
        nocc = int(np.count_nonzero(mo_occ > 0))
        nmo = int(np.asarray(ks.mo_coeff).shape[1])
        nvir = nmo - nocc
        vecdim = 1 + nocc * nvir + nocc * nocc * nvir * nvir

        states = []
        ground = np.zeros(vecdim, dtype=float)
        ground[0] = 1.0
        states.append(ground)

        if td is None:
            return states

        ident = np.eye(nmo)
        for x, y in td.xy[:self.nexc]:
            vec = np.zeros(vecdim, dtype=float)
            vec[1:1 + nocc * nvir] = np.asarray(x, dtype=float).reshape(-1)
            norm = pyscf_ci.cisd.overlap(vec, vec, nmo, nocc, ident)
            if abs(norm) > 1e-14:
                vec /= np.sqrt(abs(norm))
            states.append(vec)
        return states

    def _state_overlap_matrix(self, ks_ref, td_ref, ks_other, td_other):
        try:
            from pyscf import ci as pyscf_ci
            from pyscf import gto as pyscf_gto
        except Exception as exc:
            raise NotImplementedError(
                "Overlap-based NACs require a local PySCF installation."
            ) from exc

        mol_ref = ks_ref.mol if hasattr(ks_ref.mol, 'intor') else ks_ref.mol.topyscf()
        mol_other = ks_other.mol if hasattr(ks_other.mol, 'intor') else ks_other.mol.topyscf()
        sao = pyscf_gto.intor_cross('int1e_ovlp', mol_ref, mol_other)

        cref = np.asarray(ks_ref.mo_coeff)
        cother = np.asarray(ks_other.mo_coeff)
        smo = np.dot(cref.T.conj(), np.dot(sao, cother))

        mo_occ = np.asarray(ks_ref.mo_occ)
        nocc = int(np.count_nonzero(mo_occ > 0))
        nmo = int(cref.shape[1])

        states_ref = self._state_vectors(ks_ref, td_ref)
        states_other = self._state_vectors(ks_other, td_other)

        ovlp = np.zeros((len(states_ref), len(states_other)), dtype=complex)
        for i, bra in enumerate(states_ref):
            for j, ket in enumerate(states_other):
                ovlp[i, j] = pyscf_ci.cisd.overlap(bra, ket, nmo, nocc, s=smo)

        for j in range(min(ovlp.shape[0], ovlp.shape[1])):
            if abs(ovlp[j, j]) > 1e-14:
                phase = np.exp(-1j * np.angle(ovlp[j, j]))
                ovlp[:, j] *= phase
        return ovlp

    def _overlap_nac(self, coords, ks, td):
        coords = self._reshape_coords(coords)
        ndof = coords.size
        nac = np.zeros((self.nstates, self.nstates, ndof), dtype=float)
        if self.nstates <= 1:
            return nac

        for a in range(ndof):
            disp = np.zeros(ndof, dtype=float)
            disp[a] = self.nac_fd_step
            _, ks_p, td_p = self._build_state_point(coords.reshape(-1) + disp)
            _, ks_m, td_m = self._build_state_point(coords.reshape(-1) - disp)

            ovlp_p = self._state_overlap_matrix(ks, td, ks_p, td_p)
            ovlp_m = self._state_overlap_matrix(ks, td, ks_m, td_m)
            nac[:, :, a] = ((ovlp_p - ovlp_m) / (2.0 * self.nac_fd_step)).real

        nac = 0.5 * (nac - np.transpose(nac, (1, 0, 2)))
        for i in range(self.nstates):
            nac[i, i, :] = 0.0
        return nac

    def _evaluate_pyscf(self, coords):
        coords = self._reshape_coords(coords)
        _, ks, td = self._build_state_point(coords)
        energies = self._state_energies(ks, td)
        grads = self._pyscf_gradients(ks, td)
        if self.nac_method == 'overlap_fd':
            try:
                nac = self._overlap_nac(coords, ks, td)
            except NotImplementedError:
                nac = self._zero_nac(coords)
        else:
            nac = self._zero_nac(coords)
        return energies, grads, nac

    def _evaluate_pyqed(self, coords):
        coords = self._reshape_coords(coords)
        _, ks, td = self._build_state_point(coords)
        energies = self._state_energies(ks, td)
        grads = self._pyqed_gradients(coords, ks, td)
        return energies, grads, self._zero_nac(coords)

    def _pyqed_gradients(self, coords, ks, td):
        if self.point_charge_coords is not None:
            return self._finite_difference_gradients(coords)
        try:
            grads = np.zeros((self.nstates, coords.size), dtype=float)
            grads[0] = np.asarray(ks.nuc_grad_method().kernel(), dtype=float).reshape(-1)
            if self.nexc > 0:
                td_grad = td.nuc_grad_method()
                for n in range(self.nexc):
                    grads[n + 1] = np.asarray(
                        td_grad.kernel(state=n + 1), dtype=float
                    ).reshape(-1)
        except (AttributeError, ImportError, NotImplementedError):
            grads = self._finite_difference_gradients(coords)
        return grads

    def evaluate(self, coords):
        """
        Evaluate state energies, gradients, and NACs at one geometry.
        """
        if self.backend == 'pyscf':
            return self._evaluate_pyscf(coords)
        return self._evaluate_pyqed(coords)

    def single_point(self, coords):
        """
        Backward-compatible alias for :meth:`evaluate`.
        """
        return self.evaluate(coords)

    def grad(self, coords=None):
        if coords is None:
            coords = _get_atom_coords(self.mol)
        _, g, _ = self.evaluate(coords)
        return g

    def nonadiabatic_coupling(self):
        pass
    
    def as_scanner(self):
        """
        Return a callable scanner with the contract
        ``scanner(coords) -> (energy, grad, nac)``.
        """
        def scanner(coords, *args):
            if args:
                raise TypeError("TDDFTDriver scanner does not accept extra positional arguments.")
            return self.evaluate(coords)
        return scanner

    def point_data(self, coords):
        """
        Return the backend state objects and adiabatic data at one geometry.
        """
        coords = self._reshape_coords(coords)
        _, ks, td = self._build_state_point(coords)
        energies = self._state_energies(ks, td)
        if self.backend == 'pyscf':
            grads = self._pyscf_gradients(ks, td)
        else:
            grads = self._pyqed_gradients(coords, ks, td)
        return {
            'coords': coords,
            'ks': ks,
            'td': td,
            'energies': energies,
            'grads': grads,
        }

    def state_overlap(self, coords_ref, coords_other, ref_data=None, other_data=None):
        """
        Overlap matrix between adiabatic states at two nearby geometries.
        """
        if ref_data is None:
            ref_data = self.point_data(coords_ref)
        if other_data is None:
            other_data = self.point_data(coords_other)
        return self._state_overlap_matrix(
            ref_data['ks'],
            ref_data['td'],
            other_data['ks'],
            other_data['td'],
        )

    def normal_modes(self):
        if self.backend == 'pyscf':
            from pyqed.qchem.hessian import Hessian
            hessian = Hessian(self.ks)
            frequencies, modes, reduced_masses = hessian.run()
            equilibrium_geometry = _get_atom_coords(hessian.mol)
            return frequencies, modes, reduced_masses, equilibrium_geometry

        hessian = self.ks.Hessian()
        hessian.run()
        vib = hessian.vibrational_analysis()
        return (
            np.asarray(vib['freq_au']),
            np.asarray(vib['modes']),
            np.asarray(vib['reduced_mass_au']),
            _get_atom_coords(self.ks.mol),
        )

class Ehrenfest:
    """
    Ehrenfest dynamics for model Hamiltonians
    """
    def __init__(self, ndim, ntraj, nstates, mass=1, nac_driver=None):
        """
        Ehrenfest dynamics for model Hamiltonians

        .. math::

            i \dot{c}_i(t) = \sum_{j = 0}^{N-1} (E_i \delta_{ij} - i d_{ij} v_i ) c_j

            \dot{P} = F/M

            \dot{X} = P

        where i,j labels the electronic states, X, P are respectively, the nuclear coordinate and
        momentum.

        Parameters
        ----------
        ndim : TYPE
            DESCRIPTION.
        ntraj : TYPE
            DESCRIPTION.
        nstates : TYPE
            DESCRIPTION.
        mass : TYPE, optional
            DESCRIPTION. The default is 1.
        model : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        self.ntraj    = ntraj
        self.nstates  = nstates
        self.mass     = mass

        # self.interp   = interp
        # self.grid     = pes_grid
        self.ndim     = ndim  #

        self.nac_driver = nac_driver
        self.trajs = None
        self.times = None
        self.x_history = None
        self.rho_history = None
        self.energy_history = None
        self.norm_history = None



        # electronic coeffs, positions, momenta
        # self.c  = np.zeros((ntraj, nstates), dtype=np.complex128)
        # self.y  = np.zeros((ntraj, len(pes_grid)))
        # self.py = np.zeros_like(self.y)
        # self.w  = np.ones(ntraj)/ntraj

        # NAC driver
        # self.nac_driver = model #or H3CASSCF_NAC(nstates=nstates)

    def sample(
        self,
        init_state=None,
        distribution='gaussian',
        ax=1,
        x0=0,
        p0=0,
        ap=None,
        c0=None,
        sample_momentum=None,
        temperature=300,
        unit='K',
    ):
        """


        Parameters
        ----------
        distribution : TYPE, optional
            DESCRIPTION. The default is 'gaussian'.
        ax : TYPE, optional
            DESCRIPTION. The default is 1.
        x0 : TYPE, optional
            DESCRIPTION. The default is 0.
        p0 : TYPE, optional
            Initial momentum center. Scalar or shape ``(ndim,)``.
        ap : TYPE, optional
            Momentum-space Gaussian precision. If omitted, a minimum-uncertainty
            value matched to ``ax`` is used.
        c0 : TYPE, optional
            Initial electronic amplitude vector of shape ``(nstates,)``.
        sample_momentum : bool, optional
            Whether to sample the initial momentum distribution. Defaults to
            ``True`` for ``distribution='wigner'`` and ``False`` otherwise.
        temperature : TYPE, optional
            DESCRIPTION. The default is 300.
        unit : TYPE, optional
            DESCRIPTION. The default is 'K'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        x : TYPE
            DESCRIPTION.
        p : TYPE
            DESCRIPTION.
        c : TYPE
            DESCRIPTION.

        """
        ntraj = self.ntraj

        if unit == 'K':
            temperature = temperature/au2k
        elif unit == 'au':
            temperature = temperature
        else:
            raise ValueError(f"Invalid unit: {unit}")


        if np.isscalar(ax):
            ax = np.full(self.ndim, float(ax), dtype=float)
        else:
            ax = np.asarray(ax, dtype=float)
            if ax.shape != (self.ndim,):
                raise ValueError(f"ax must be scalar or shape ({self.ndim},), got {ax.shape}.")

        if np.isscalar(x0):
            x0 = np.full(self.ndim, float(x0), dtype=float)
        else:
            x0 = np.asarray(x0, dtype=float)
            if x0.shape != (self.ndim,):
                raise ValueError(f"x0 must be scalar or shape ({self.ndim},), got {x0.shape}.")

        if np.isscalar(p0):
            p0 = np.full(self.ndim, float(p0), dtype=float)
        else:
            p0 = np.asarray(p0, dtype=float)
            if p0.shape != (self.ndim,):
                raise ValueError(f"p0 must be scalar or shape ({self.ndim},), got {p0.shape}.")

        if ap is None:
            ap = 0.5 * ax
        elif np.isscalar(ap):
            ap = np.full(self.ndim, float(ap), dtype=float)
        else:
            ap = np.asarray(ap, dtype=float)
            if ap.shape != (self.ndim,):
                raise ValueError(f"ap must be scalar or shape ({self.ndim},), got {ap.shape}.")

        distribution = distribution.lower()
        if sample_momentum is None:
            sample_momentum = distribution == 'wigner'

        self.w = np.array([1./self.ntraj]*self.ntraj)

        if distribution not in ('gaussian', 'wigner'):
            raise ValueError("distribution must be 'gaussian' or 'wigner'.")

        x = np.random.randn(self.ntraj, self.ndim)
        p = np.zeros((self.ntraj, self.ndim), dtype=float)

        for j in range(self.ndim):
            x[:, j] = x[:, j] / np.sqrt(2.0 * ax[j]) + x0[j]
            if sample_momentum:
                p[:, j] = np.random.randn(self.ntraj) * np.sqrt(ap[j]) + p0[j]
            else:
                p[:, j] = p0[j]

        # initial electronic state
        if c0 is not None:
            c0 = np.asarray(c0, dtype=complex)
            if c0.shape != (self.nstates,):
                raise ValueError(f"c0 must have shape ({self.nstates},), got {c0.shape}.")
            c0_norm = np.linalg.norm(c0)
            if c0_norm == 0:
                raise ValueError("c0 must not be the zero vector.")
            c0 = c0 / c0_norm
            c = np.tile(c0, (self.ntraj, 1))
        else:
            if init_state is None:
                raise ValueError("Specify either init_state or c0.")
            c = np.zeros((self.ntraj, self.nstates), dtype=complex)
            c[:, init_state] = 1

        trajs = []
        for n in range(ntraj):
            traj = EhrenfestTrajectory(x[n], p[n], c[n], mass=self.mass)
            trajs.append(traj)

        self.trajs = trajs

        return trajs

    # def initialize(self, y0, width, E_kin_au, mix_state=None):
    #     # self.ndim = len(y0)
    #     # sample positions around y0

    #     self.y = np.random.randn(self.ntraj, self.ndim)/np.sqrt(2*width) + y0
    #     self.py = np.zeros_like(self.y)
    #     if mix_state is None:
    #         mix_state = np.ones(self.nstates)/np.sqrt(self.nstates)
    #     for k in range(self.ntraj):
    #         self.c[k,:] = mix_state


    #     p_magnitude = np.sqrt(2 * self.mass * E_kin_au)
    #     directions = np.random.randn(self.ntraj, self.ndim)
    #     directions /= np.linalg.norm(directions, axis=1)[:, None]
    #     self.py = p_magnitude * directions

    # def _get_energies(self, yk):

    #     eps, _ = self.nac_driver.adiabatic_energies(yk)
    #     return eps


    # def _get_all_gradients(self, yk):
    #     nst, ndim = self.nstates, len(yk)
    #     grads = np.zeros((nst, ndim))
    #     delta = 1e-3
    #     for i in range(nst):
    #         for a in range(ndim):
    #             y_plus  = yk.copy(); y_plus[a]  += delta
    #             y_minus = yk.copy(); y_minus[a] -= delta
    #             e_plus  = self._get_energies(y_plus)[i]
    #             e_minus = self._get_energies(y_minus)[i]
    #             # grads[i,a] = (e_plus - e_minus)/(2*delta)
    #             grads[i,a] = (float(e_plus) - float(e_minus))/(2*delta)
    #     return grads


    # def _get_nacs(self, yk):
    #     """
    #     Compute nonadiabatic couplings d_{ji}^alpha using:
    #         d_{ji}^alpha = ⟨phi_j | ∂H/∂R_alpha | phi_i⟩ / (E_i - E_j)

    #     Only for 2 electronic states (nstates=2).
    #     """

    #     if self.nac_driver is not None:
    #         # Use a NAC driver to compute the couplings
    #         return self.nac_driver.nacs(yk)
    #         # return self.nac_driver.get_nacs(yk[0])

    #     else: print("Using finite difference to compute NACs")

    # def _get_grads(self, yk):
    #     return self.nac_driver.gradients(yk)


    def _dc(self, c, v, energy, nac):
        """
        equation of motion for C

        .. math::

            dC/dt = -i \mathbf{H}_\text{eff}  C,

            H_eff[i,i] = E_i,   H_eff[j,i] = - i (v · d_{ji})
        """

        # energies = self._get_energies(x)

        # d = self._get_nacs(yk)
        # nst = self.nstates

        H = np.diag(energy) - 1j * contract('a, ija', v, nac)

        # for i in range(nst):
        #     H_eff[i,i] = energies[i]
        #     for j in range(nst):
        #         if j!=i:
        #             coupling = np.dot(vk, d[j,i])
        #             H_eff[j,i] = -1j * coupling

        return -1j * H @ c

    def H(self, v, energy, nac):
        """
        equation of motion for C

        .. math::

            dC/dt = -i \mathbf{H}_\text{eff}  C,

            H_eff[i,i] = E_i,   H_eff[j,i] = - i (v · d_{ji})
        """

        H = np.diag(energy) - 1j * contract('a, ija', v, nac)

        return H

    def mean_field_force(self, x, c, energy=None, grad=None, nac=None, return_electronic_data=False):

        """
        Mean field force
        .. math::

            F_\text{MF} = -\sum_j |c_j|^2 \partial_\alpha E_j
                         + \sum_{i, j} c_i^* c_j (E_i - E_j) d_{ji}


        Refs
            J. Chem. Phys. 150, 204124 (2019)
        """
        # if energy is None:
        #     energy, grad = self.model.adiabatic_energy(x)

        if nac is None:
            # nac = self.model.nac(x)

            energy, grad, nac = self.nac_driver(x)

        # C = self.mo_coeff

        # diagonal part
        F_diag = - contract('a, ai -> i', np.abs(c)**2, grad)
        dE  = energy[:, None] - energy[None, :]
        F_non = contract('i, j, ij, ija -> a', c.conj(), c, dE, nac)
        F = F_diag + F_non

        if return_electronic_data:
            return np.real(F), energy, grad, nac
        else:
            return np.real(F)

        # # version 1 in the loop
        # F = np.zeros(ndim)
        # # diagonal term
        # for j in range(nst):
        #     F -= np.abs(ck[j])**2 * grads[j]
        # # nonadiabatic term
        # for i in range(nst):
        #     for j in range(nst):
        #         if i!=j:
        #             coeff = (ck[j].conj()*ck[i]*(energies[i]-energies[j])).real
        #             # F -= coeff * d[i,j]
        #             F -= coeff * d[j,i]
        # return F

    def run(self, dt=0.01, nt=10, nout=1, method='euler', force_driver=None, *args):
        """
        using velocity verlet method to propagate the nuclei

        .. math::

            X(t + \Delta t) = P(t+ dt/2) * dt

        """

        if self.trajs is None:
            raise ValueError("No trajectories initialized. Call sample(...) before run(...).")
        if nout <= 0:
            raise ValueError("nout must be a positive integer.")

        mass = np.asarray(self.mass, dtype=float)
        if mass.ndim == 0:
            mass = np.full(self.ndim, float(mass), dtype=float)
        elif mass.shape != (self.ndim,):
            raise ValueError(f"mass must be scalar or shape ({self.ndim},), got {mass.shape}.")

        dt2 = dt/2.

        # trajs = self.trajs

        force = self.mean_field_force

        # get electronic data at t0
        for traj in self.trajs:
            traj.force, traj.energy, traj.grad, traj.nac = force(traj.x, traj.c, return_electronic_data=True)
            traj.v = traj.p / mass

            # print('nac', traj.nac.shape)

        times = [0.0]
        x_history = [self.xAve().copy()]
        rho_history = [self.rdm().copy()]
        energy_history = [self.total_energy()]
        norm_history = [self.norm()]

        completed_steps = 0
        while completed_steps < nt:
        # for step in tqdm(range(nt),desc="processing"):
            chunk_steps = min(nout, nt - completed_steps)

            for k in range(chunk_steps):

                for traj in self.trajs:

                    traj.p_prev = traj.p.copy()
                    traj.nac_prev = traj.nac.copy()

                    # half-step momentum
                    traj.p += dt2 * traj.force

                    # half-step position
                    traj.x += dt2 * traj.p / mass

                    # full-step c
                    v = traj.p/mass
                    traj.v = v

                    # print(self._dc(traj.c, v, traj.energy, traj.nac))

                    # traj.c = traj.c + dt * self._dc(traj.c, v, traj.energy, traj.nac)

                    H = self.H(v, traj.energy, traj.nac)

                    traj.c = expm(-1j * H * dt) @ traj.c
                    c_norm = np.linalg.norm(traj.c)
                    if c_norm > 0:
                        traj.c = traj.c / c_norm

                    # force at t + dt
                    traj.energy, traj.grad, traj.nac = self.nac_driver(traj.x, *args)

                    # TODO: parrallel transport gauge
                    # for n in range(1, self.nstates):
                    #     if angle(traj.nac_prev, traj.nac)
                    
                    # print(angle(traj.nac_prev[0, 1], traj.nac[0,1]))
                    
                    # half-step x
                    traj.x += dt2 * traj.p / mass
                    
                    # half p
                    traj.force = force(traj.x, traj.c, traj.energy, traj.grad, traj.nac)
                    traj.p += dt2 * traj.force
                    traj.v = traj.p / mass
            completed_steps += chunk_steps
            times.append(completed_steps * dt)
            x_history.append(self.xAve().copy())
            rho_history.append(self.rdm().copy())
            energy_history.append(self.total_energy())
            norm_history.append(self.norm())

        self.times = np.asarray(times, dtype=float)
        self.x_history = np.asarray(x_history)
        self.rho_history = np.asarray(rho_history)
        self.energy_history = np.asarray(energy_history, dtype=float)
        self.norm_history = np.asarray(norm_history, dtype=float)
        return self

    def rdm(self):
        """
            compute reduced electronic density matrix
        """
        if self.trajs is None:
            raise ValueError("No trajectories initialized.")
        n = self.nstates
        rho = np.zeros((n, n), dtype=np.complex128)

        for traj in self.trajs:
            rho += traj.rdm()

        return rho/self.ntraj


    def xAve(self):
        if self.trajs is None:
            raise ValueError("No trajectories initialized.")
        return sum([traj.x for traj in self.trajs])/self.ntraj




    def total_energy(self):
        if self.trajs is None:
            raise ValueError("No trajectories initialized.")
        mass = np.asarray(self.mass, dtype=float)
        if mass.ndim == 0:
            mass = np.full(self.ndim, float(mass), dtype=float)
        kinetic = 0.0
        electronic = 0.0
        for traj in self.trajs:
            kinetic += 0.5 * np.sum((traj.p**2) / mass)
            electronic += np.real(np.vdot(traj.c, np.asarray(traj.energy) * traj.c))
        return float((kinetic + electronic) / self.ntraj)

    def norm(self):
        if self.trajs is None:
            raise ValueError("No trajectories initialized.")
        return float(np.mean([np.vdot(traj.c, traj.c).real for traj in self.trajs]))

class GeometricEhrenfest(Ehrenfest):
    pass


def _get_natom(mol):
    for attr in ('natom', 'natm'):
        if hasattr(mol, attr):
            return int(getattr(mol, attr))
    raise AttributeError("Molecule-like object must define natom or natm.")


def _get_atom_coords(mol):
    if not hasattr(mol, 'atom_coords'):
        raise AttributeError("Molecule-like object must define atom_coords().")
    atom_coords = mol.atom_coords
    try:
        coords = atom_coords(unit='Bohr')
    except TypeError:
        coords = atom_coords()
    coords = np.asarray(coords, dtype=float)
    natom = _get_natom(mol)
    return coords.reshape(natom, 3)


def _get_atom_masses(mol):
    if not hasattr(mol, 'atom_mass_list'):
        raise AttributeError("Molecule-like object must define atom_mass_list().")
    atom_mass_list = mol.atom_mass_list
    try:
        masses = atom_mass_list(isotope_avg=True)
    except TypeError:
        masses = atom_mass_list()
    masses = np.asarray(masses, dtype=float)
    natom = _get_natom(mol)
    if masses.shape != (natom,):
        raise ValueError(f"atom_mass_list must return shape ({natom},), got {masses.shape}.")
    return masses


def _as_point_charges(coords, charges):
    if coords is None and charges is None:
        return None, None
    if coords is None or charges is None:
        raise ValueError("Provide point_charge_coords and point_charges together.")

    coords = np.asarray(coords, dtype=float)
    if coords.ndim == 1:
        if coords.size != 3:
            raise ValueError("point_charge_coords must have shape (ncharge, 3).")
        coords = coords.reshape(1, 3)
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("point_charge_coords must have shape (ncharge, 3).")

    charges = np.asarray(charges, dtype=float).reshape(-1)
    if charges.shape != (coords.shape[0],):
        raise ValueError(
            f"point_charges must have shape ({coords.shape[0]},), got {charges.shape}."
        )
    return coords.copy(), charges.copy()


def _is_linear_geometry(coords, tol=1e-10):
    coords = np.asarray(coords, dtype=float)
    natom = coords.shape[0]
    if natom <= 2:
        return True

    ref = None
    origin = coords[0]
    for vec in coords[1:] - origin:
        if np.linalg.norm(vec) > tol:
            ref = vec
            break

    if ref is None:
        return True

    ref_norm = np.linalg.norm(ref)
    for vec in coords[1:] - origin:
        vec_norm = np.linalg.norm(vec)
        if vec_norm <= tol:
            continue
        if np.linalg.norm(np.cross(ref, vec)) > tol * ref_norm * vec_norm:
            return False
    return True


def _expected_vibrational_mode_count(coords):
    natom = np.asarray(coords).reshape(-1, 3).shape[0]
    if natom <= 1:
        return 0
    ncart = 3 * natom
    if _is_linear_geometry(coords):
        return max(0, ncart - 5)
    return max(0, ncart - 6)


def _select_vibrational_mode_indices(frequencies, coords, mode_indices=None, freq_tol=1e-8):
    frequencies = np.asarray(frequencies)
    nmodes = frequencies.shape[0]
    if mode_indices is not None:
        indices = np.asarray(mode_indices, dtype=int)
        if indices.ndim != 1:
            raise ValueError("mode_indices must be a one-dimensional index array.")
    else:
        nvib = _expected_vibrational_mode_count(coords)
        if nvib == 0:
            return np.zeros(0, dtype=int)
        real_mask = np.abs(np.imag(frequencies)) <= freq_tol
        positive = np.where(real_mask & (np.real(frequencies) > freq_tol))[0]
        if positive.size < nvib:
            raise ValueError(
                f"Need {nvib} positive vibrational frequencies, found {positive.size}."
            )
        positive = positive[np.argsort(np.real(frequencies[positive]))]
        indices = positive[-nvib:]

    if np.any(indices < 0) or np.any(indices >= nmodes):
        raise ValueError("mode_indices contain out-of-range entries.")

    selected = frequencies[indices]
    if np.any(np.abs(np.imag(selected)) > freq_tol):
        raise ValueError("Selected normal modes contain imaginary frequencies.")
    if np.any(np.real(selected) <= freq_tol):
        raise ValueError("Selected normal modes must have positive frequencies.")
    return np.sort(indices)


def _broadcast_mode_parameter(value, nmodes, name):
    if np.isscalar(value):
        return np.full(nmodes, float(value), dtype=float)
    value = np.asarray(value, dtype=float)
    if value.shape != (nmodes,):
        raise ValueError(f"{name} must be scalar or shape ({nmodes},), got {value.shape}.")
    return value


def _validate_mode_variances(variances, nmodes, name):
    variances = _broadcast_mode_parameter(variances, nmodes, name)
    if np.any(variances < 0.0):
        raise ValueError(f"{name} must be non-negative.")
    return variances


def _thermal_wigner_variances(frequencies, temperature_au):
    frequencies = np.asarray(frequencies, dtype=float)
    if np.any(frequencies <= 0.0):
        raise ValueError("Wigner sampling requires positive vibrational frequencies.")
    if temperature_au <= 0.0:
        coth = np.ones_like(frequencies)
    else:
        x = 0.5 * frequencies / temperature_au
        coth = 1.0 / np.tanh(x)
    q_var = 0.5 * coth / frequencies
    p_var = 0.5 * coth * frequencies
    return q_var, p_var

class CoupledOscillatorModel:
    """
    Two-state, two-dimensional coupled harmonic oscillator:

    .. math::

        H = 1/2 \omega_1 x^2 + 1/2 \omega_2 y^2  + 1/2 g x y


    """
    def __init__(self, omega1, omega2, g, x):
        self.omega1  = omega1
        self.omega2  = omega2
        self.g       = g
        self.nstates = 2
        self.ndim    = 1
        self.x = 1 # fixed at x=1


        self.E = None
        self.U = None


    def H_diab(self, R):
        """
        single point calculation

        ### THIS IS WRONG!

        Parameters
        ----------
        R : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        x = self.x
        # y = R
        y = float(R[0] if hasattr(R, '__len__') else R)
        E1 = 0.5 * self.omega1*x*x
        E2 = 0.5 * self.omega2 * y**2
        V12 = 0.5 * self.g * y * x
        return np.array([[E1, V12], [V12, E2]], dtype=float)

        # return np.array([
        #     [0.5*self.omega1*x*x,  0.5*self.g*x*y],
        #     [0.5*self.g*x*y,       0.5*self.omega2*y*y]
        # ], dtype=float)

    # def pes(self, y_grid):
    #     """
    #     For each y in y_grid, construct the harmonic oscillator Hamiltonian
    #     in x and extract nstates adiabatic energies \\varpsilon_j(y).
    #     """
    #     self.y_grid = y_grid
    #     ny = len(y_grid)
    #     energies = np.zeros((ny, self.nstates))

    #     for i in range(ny):
    #         n = np.arange(self.nstates)
    #         energies[i, :] = omega1 * np.sqrt(1 ) * (n + 0.5)+0.5*omega2 * y[
    #             i]**2 -(0.125 *g**2 *y[i]**2)/(omega1 )


    #     self.energies = energies
    #     self.interpolators = [RegularGridInterpolator((y_grid,), energies[:, i]) for i in range(self.nstates)]

    #     return energies

    def adiabatic_energy(self, R, return_grad=True):

        H = self.H_diab(R)
        E, U = np.linalg.eigh(H)

        self.E = E
        self.U = U

        if return_grad:
            grad = self.dH(R)

            return E, grad
        else:
            return E, U

    def dH(self, R):
        """
        retrun dH/dx, dH/dy two matrix
        """
        # y = R
        y = float(R[0] if hasattr(R, '__len__') else R)
        x = self.x
        # dH_dx = np.array([
        #     [self.omega1*x,    0.5*self.g*y],
        #     [0.5*self.g*y,     0.0]
        # ])
        # dH_dy = np.array([
        #     [0.0,              0.5*self.g*x],
        #     [0.5*self.g*x,     self.omega2*y]
        # ])
        dH_dy = np.array([[0.0, 0.5*self.g*x], [0.5*self.g*x, self.omega2*y]], dtype=float)
        return [dH_dy]
        # return [dH_dx, dH_dy]

    def nac(self, R):
        """
        Calculate NAC matrix

        .. math::

            d_{ij}^\alpha = <\\phi_i | \partial_\alpha H | \\phi_j> / (E_j - E_i)

        for i \ne j, \alpha is the nuclear degrees of freedom.

        return (nstates,nstates,ndim)
        """
        # eps, U = self.adiabatic_energy(R, return_grad=False)
        # if self.E is not None:
        #     E, U = self.E, self.U
        # else:
        E, U = self.adiabatic_energy(R, return_grad=False)


        dH = np.array(self.dH(R))

        # n, dim = self.nstates, self.ndim

        d = np.zeros((self.nstates, self.nstates, self.ndim), dtype=float)

        # for a in range(self.ndim):
        #     M = dH_list[a]
        #     for i in range(self.nstates):
        #         for j in range(self.nstates):
        #             if i != j:
        #                 num = U[:,i].conj() @ M @ U[:,j]
        #                 de = E[j] - E[i]
        #                 d[i,j,a] = num/de

        num = contract('ui, auv, vj -> ija', U.conj(), dH, U)

        with np.errstate(divide='ignore'):
            de_inv = 1/np.subtract.outer(E, E)


        de_inv = np.nan_to_num(de_inv, nan=0, posinf=0, neginf=0)

        d = -contract('ija, ij -> ija',  num, de_inv)

        return E, d



    # def gradients(self, R):
    #     """
    #     Return the gradient of each adiabatic energy level with respect to nuclear coordinates
    #     \\partial_alpha \\varpsilon_i = <\\phi_i | \\partial_alpha H | \\phi_i>
    #     shape (nstates, ndim)
    #     """
    #     eps, U = self.adiabatic_energies(R)
    #     dH_list = self.dH(R)
    #     grads = np.zeros((self.nstates, self.ndim), dtype=float)

    #     for a in range(self.ndim):
    #         M = dH_list[a]
    #         for i in range(self.nstates):
    #             grads[i,a] = (U[:,i].conj() @ (M @ U[:,i])).real
    #     return grads



class TDDFTEhrenfest(Ehrenfest):
    def __init__(self, mol, ntraj, nstates, nac_driver=None):

        self.mol = mol
        self.natom = _get_natom(mol)
        self.equilibrium_geometry = _get_atom_coords(mol)
        self.atomic_masses = _get_atom_masses(mol)
        cartesian_masses = np.repeat(self.atomic_masses, 3)

        self._raw_nac_driver = nac_driver
        wrapped_driver = self._wrap_nac_driver(nac_driver) if nac_driver is not None else None

        super().__init__(ndim=self.natom * 3,
                         ntraj=ntraj,
                         nstates=nstates,
                         mass=cartesian_masses,
                         nac_driver=wrapped_driver)

        self.normal_mode_indices = None
        self.normal_modes = None
        self.vibrational_frequencies = None
        self.reduced_masses = None

        self.mols = None
        self.configurations = None

    @staticmethod
    def _unitarize_overlap(overlap):
        u, _, vh = np.linalg.svd(np.asarray(overlap, dtype=complex), full_matrices=False)
        return u @ vh

    @staticmethod
    def _ld_hamiltonian(transform, energies):
        return transform @ np.diag(np.asarray(energies, dtype=float)) @ transform.conj().T

    @staticmethod
    def _ld_gradient_matrices(transform, grads):
        grads = np.asarray(grads, dtype=float)
        nstates, ndim = grads.shape
        out = np.zeros((nstates, nstates, ndim), dtype=complex)
        for a in range(ndim):
            out[:, :, a] = transform @ np.diag(grads[:, a]) @ transform.conj().T
        return out

    @staticmethod
    def _ld_force(c, grad_ld):
        rho = np.outer(np.asarray(c, dtype=complex), np.asarray(c, dtype=complex).conj())
        return -np.real(np.einsum('ij,ija->a', rho, grad_ld, optimize=True))

    def _wrap_nac_driver(self, nac_driver):
        if nac_driver is None:
            return None
        if hasattr(nac_driver, 'as_scanner'):
            try:
                scanner = nac_driver.as_scanner()
            except NotImplementedError:
                scanner = None
        else:
            scanner = nac_driver
        if scanner is None and isinstance(nac_driver, TDDFTDriver):
            return None
        if scanner is None or not callable(scanner):
            raise TypeError("nac_driver must be callable or provide as_scanner().")

        def wrapped(coords, *args):
            coords = np.asarray(coords, dtype=float).reshape(self.natom, 3)
            energy, grad, nac = scanner(coords, *args)
            energy = np.asarray(energy, dtype=float)
            grad = np.asarray(grad, dtype=float).reshape(self.nstates, self.ndim)
            nac = np.asarray(nac, dtype=float).reshape(self.nstates, self.nstates, self.ndim)
            return energy, grad, nac

        return wrapped

    def _build_hessian_normal_modes(self, hessian=None):
        if hessian is None:
            if isinstance(self._raw_nac_driver, TDDFTDriver):
                return self._raw_nac_driver.normal_modes()
            else:
                raise ValueError(
                    "Provide hessian or explicit frequencies/normal_modes when TDDFTDriver is unavailable."
                )
        frequencies, modes, reduced_masses = hessian.run()
        equilibrium_geometry = _get_atom_coords(hessian.mol)
        return frequencies, modes, reduced_masses, equilibrium_geometry

    def set_normal_modes(
        self,
        frequencies,
        normal_modes,
        reduced_masses=None,
        equilibrium_geometry=None,
        mode_indices=None,
        freq_tol=1e-8,
    ):
        frequencies = np.asarray(frequencies)
        normal_modes = np.asarray(normal_modes, dtype=float)
        if normal_modes.ndim != 3 or normal_modes.shape[1:] != (self.natom, 3):
            raise ValueError(
                f"normal_modes must have shape (nmodes, {self.natom}, 3), got {normal_modes.shape}."
            )
        if frequencies.shape != (normal_modes.shape[0],):
            raise ValueError(
                f"frequencies must have shape ({normal_modes.shape[0]},), got {frequencies.shape}."
            )

        if equilibrium_geometry is None:
            equilibrium_geometry = self.equilibrium_geometry
        equilibrium_geometry = np.asarray(equilibrium_geometry, dtype=float).reshape(self.natom, 3)

        indices = _select_vibrational_mode_indices(
            frequencies, equilibrium_geometry, mode_indices=mode_indices, freq_tol=freq_tol
        )

        if reduced_masses is None:
            reduced_masses = np.full(frequencies.shape[0], np.nan, dtype=float)
        else:
            reduced_masses = np.asarray(reduced_masses, dtype=float)
            if reduced_masses.shape != frequencies.shape:
                raise ValueError(
                    f"reduced_masses must have shape {frequencies.shape}, got {reduced_masses.shape}."
                )

        self.equilibrium_geometry = equilibrium_geometry
        self.normal_mode_indices = indices
        self.normal_modes = normal_modes[indices].copy()
        self.vibrational_frequencies = np.real(frequencies[indices]).copy()
        self.reduced_masses = reduced_masses[indices].copy()
        return self

    def _ensure_normal_modes(
        self,
        frequencies=None,
        normal_modes=None,
        reduced_masses=None,
        equilibrium_geometry=None,
        mode_indices=None,
        hessian=None,
    ):
        if frequencies is not None or normal_modes is not None or hessian is not None:
            if hessian is not None:
                frequencies, normal_modes, reduced_masses, equilibrium_geometry = \
                    self._build_hessian_normal_modes(hessian=hessian)
            elif frequencies is None or normal_modes is None:
                raise ValueError("Provide both frequencies and normal_modes together.")

            self.set_normal_modes(
                frequencies=frequencies,
                normal_modes=normal_modes,
                reduced_masses=reduced_masses,
                equilibrium_geometry=equilibrium_geometry,
                mode_indices=mode_indices,
            )
        elif self.normal_modes is None or self.vibrational_frequencies is None:
            frequencies, normal_modes, reduced_masses, equilibrium_geometry = \
                self._build_hessian_normal_modes(hessian=hessian)
            self.set_normal_modes(
                frequencies=frequencies,
                normal_modes=normal_modes,
                reduced_masses=reduced_masses,
                equilibrium_geometry=equilibrium_geometry,
                mode_indices=mode_indices,
            )

        return self.normal_modes, self.vibrational_frequencies

    def sample(
        self,
        init_state=None,
        distribution='thermal_wigner',
        q0=0.0,
        p0=0.0,
        q_var=None,
        p_var=None,
        c0=None,
        sample_momentum=None,
        temperature=300.0,
        unit='K',
        frequencies=None,
        normal_modes=None,
        reduced_masses=None,
        equilibrium_geometry=None,
        mode_indices=None,
        hessian=None,
    ):
        if unit == 'K':
            temperature_au = temperature/au2k
        elif unit == 'au':
            temperature_au = float(temperature)
        else:
            raise ValueError(f"Invalid unit: {unit}")

        modes, vib_freq = self._ensure_normal_modes(
            frequencies=frequencies,
            normal_modes=normal_modes,
            reduced_masses=reduced_masses,
            equilibrium_geometry=equilibrium_geometry,
            mode_indices=mode_indices,
            hessian=hessian,
        )

        nmodes = modes.shape[0]
        if nmodes == 0:
            raise ValueError("No vibrational normal modes available for sampling.")

        q0 = _broadcast_mode_parameter(q0, nmodes, 'q0')
        p0 = _broadcast_mode_parameter(p0, nmodes, 'p0')

        distribution = distribution.lower()
        if distribution == 'wigner':
            distribution = 'thermal_wigner'
        if sample_momentum is None:
            sample_momentum = distribution == 'thermal_wigner'
        if distribution not in ('gaussian', 'thermal_wigner'):
            raise ValueError("distribution must be 'gaussian', 'wigner', or 'thermal_wigner'.")

        if distribution == 'thermal_wigner':
            q_var_default, p_var_default = _thermal_wigner_variances(vib_freq, temperature_au)
            if q_var is None:
                q_mode_var = q_var_default
            else:
                q_mode_var = _validate_mode_variances(q_var, nmodes, 'q_var')
            if p_var is None:
                p_mode_var = p_var_default
            else:
                p_mode_var = _validate_mode_variances(p_var, nmodes, 'p_var')
        else:
            if q_var is None:
                q_mode_var = 0.5 / vib_freq
            else:
                q_mode_var = _validate_mode_variances(q_var, nmodes, 'q_var')
            if p_var is None:
                p_mode_var = 0.5 * vib_freq
            else:
                p_mode_var = _validate_mode_variances(p_var, nmodes, 'p_var')

        q = np.random.randn(self.ntraj, nmodes) * np.sqrt(q_mode_var)[None, :] + q0[None, :]
        p_mode = np.tile(p0, (self.ntraj, 1))
        if sample_momentum:
            p_mode += np.random.randn(self.ntraj, nmodes) * np.sqrt(p_mode_var)[None, :]

        modes_flat = modes.reshape(nmodes, self.ndim)
        xeq_flat = self.equilibrium_geometry.reshape(-1)
        x_cart = q @ modes_flat + xeq_flat[None, :]
        p_cart = p_mode @ (modes_flat * np.asarray(self.mass, dtype=float)[None, :])

        if c0 is not None:
            c0 = np.asarray(c0, dtype=complex)
            if c0.shape != (self.nstates,):
                raise ValueError(f"c0 must have shape ({self.nstates},), got {c0.shape}.")
            c0_norm = np.linalg.norm(c0)
            if c0_norm == 0:
                raise ValueError("c0 must not be the zero vector.")
            c = np.tile(c0 / c0_norm, (self.ntraj, 1))
        else:
            if init_state is None:
                raise ValueError("Specify either init_state or c0.")
            c = np.zeros((self.ntraj, self.nstates), dtype=complex)
            c[:, init_state] = 1.0

        self.w = np.full(self.ntraj, 1.0 / self.ntraj, dtype=float)
        trajs = []
        for n in range(self.ntraj):
            traj = TDDFTTrajectory(
                x_cart[n].reshape(self.natom, 3),
                p_cart[n],
                c[n],
                mass=self.mass,
            )
            traj.q = q[n].copy()
            traj.p_mode = p_mode[n].copy()
            trajs.append(traj)

        self.trajs = trajs
        self.configurations = np.asarray([traj.atom_coords for traj in trajs], dtype=float)
        return trajs

    def run(self, dt=0.01, nt=10, nout=1, method='euler', force_driver=None,
            electronic_representation='adiabatic_nac', *args):
        if electronic_representation not in ('overlap', 'local_diabatic_overlap'):
            return super().run(dt=dt, nt=nt, nout=nout, method=method, force_driver=force_driver, *args)

        driver = self._raw_nac_driver
        if not isinstance(driver, TDDFTDriver):
            raise TypeError(
                "overlap propagation requires nac_driver to be a TDDFTDriver."
            )

        if self.trajs is None:
            raise ValueError("No trajectories initialized. Call sample(...) before run(...).")
        if nout <= 0:
            raise ValueError("nout must be a positive integer.")

        mass = np.asarray(self.mass, dtype=float)
        if mass.ndim == 0:
            mass = np.full(self.ndim, float(mass), dtype=float)
        elif mass.shape != (self.ndim,):
            raise ValueError(f"mass must be scalar or shape ({self.ndim},), got {mass.shape}.")

        dt2 = dt / 2.0

        for traj in self.trajs:
            point = driver.point_data(traj.x.reshape(self.natom, 3))
            traj.energy = point['energies']
            traj.grad = point['grads']
            traj.state_point = point
            traj.nac = np.zeros((self.nstates, self.nstates, self.ndim), dtype=float)
            traj.force = self._ld_force(traj.c, self._ld_gradient_matrices(np.eye(self.nstates), traj.grad))
            traj.v = traj.p / mass

        times = [0.0]
        x_history = [self.xAve().copy()]
        rho_history = [self.rdm().copy()]
        energy_history = [self.total_energy()]
        norm_history = [self.norm()]

        completed_steps = 0
        while completed_steps < nt:
            chunk_steps = min(nout, nt - completed_steps)
            for _ in range(chunk_steps):
                for traj in self.trajs:
                    traj.p += dt2 * traj.force
                    traj.x += dt * traj.p / mass
                    traj.v = traj.p / mass

                    prev_state = traj.state_point
                    curr_state = driver.point_data(traj.x.reshape(self.natom, 3))
                    state_overlap = driver.state_overlap(
                        prev_state['coords'],
                        curr_state['coords'],
                        ref_data=prev_state,
                        other_data=curr_state,
                    )
                    ld_transform = self._unitarize_overlap(state_overlap)

                    h_prev = np.diag(np.asarray(prev_state['energies'], dtype=float))
                    h_curr_ld = self._ld_hamiltonian(ld_transform, curr_state['energies'])
                    h_step = 0.5 * (h_prev + h_curr_ld)

                    c_step = expm(-1j * h_step * dt) @ traj.c
                    grad_curr_ld = self._ld_gradient_matrices(ld_transform, curr_state['grads'])
                    traj.force = self._ld_force(c_step, grad_curr_ld)

                    traj.c = ld_transform.conj().T @ c_step
                    c_norm = np.linalg.norm(traj.c)
                    if c_norm > 0:
                        traj.c = traj.c / c_norm

                    traj.energy = curr_state['energies']
                    traj.grad = curr_state['grads']
                    traj.state_point = curr_state
                    traj.overlap = state_overlap
                    traj.ld_transform = ld_transform
                    traj.v = traj.p / mass
                    traj.p += dt2 * traj.force

            completed_steps += chunk_steps
            times.append(completed_steps * dt)
            x_history.append(self.xAve().copy())
            rho_history.append(self.rdm().copy())
            energy_history.append(self.total_energy())
            norm_history.append(self.norm())

        self.times = np.asarray(times, dtype=float)
        self.x_history = np.asarray(x_history)
        self.rho_history = np.asarray(rho_history)
        self.energy_history = np.asarray(energy_history, dtype=float)
        self.norm_history = np.asarray(norm_history, dtype=float)
        return self


# Backward compatibility alias.  Prefer TDDFTEhrenfest for new code.
AbInitioEhrenfest = TDDFTEhrenfest
        


    # def mean_field_force(self, x, c, energy=None, grad=None, nac=None):
    #     """
    #     Mean field force
    #     .. math::

    #         F_\text{MF} = -\sum_j |c_j|^2 \partial_\alpha E_j
    #                      + \sum_{i, j} c_i^* c_j (E_i - E_j) d_{ji}


    #     Refs
    #         J. Chem. Phys. 150, 204124 (2019)
    #     """
    #     # if energy is None:
    #     #     energy, grad = self.model.adiabatic_energy(x)

    #     if nac is None:
    #         # nac = self.model.nac(x)

    #         energy, grad, nac = self.nac_driver(x)

    #     # C = self.mo_coeff

    #     # diagonal part
    #     F_diag = - contract('a, ai -> i', np.abs(c)**2, grad)
    #     dE  = energy[:, None] - energy[None, :]
    #     F_non = contract('i, j, ij, ija -> a', c.conj(), c, dE, nac)
    #     F = F_diag + F_non


    #     return np.real(F)
    
    # def run(self, dt, nt):

    #     mass = self.mass

    #     x, p, c = self.sample()

    #     for mol in self.mols:





if __name__ == '__main__':

    import ultraplot as plt

    from pyqed.models.ShinMetiu import ShinMetiu2
    from pyqed import proton_mass as mp


    mol = ShinMetiu2()

    mol.build(domain=[[-10, 10], ] * 2, npts=[31, 31])

    ed = Ehrenfest(ndim=mol.ndim, ntraj=1, nstates=mol.nstates, mass=[mp, ] * 2)
    
    ed.nac_driver = mol.nonadiabatic_coupling
    
    ed.sample(init_state=2, x0=[0, 1.3], ax=18)
    ed.run(dt=0.5, nt=400, nout=2)

    rho = ed.rdm()

    print(rho)

    #######################
    ### Ehrenfest dynamics
    #######################
    # ehrenfest = Ehrenfest(ntraj=10, ndim = 1, mass=mass_nuc, # x is quantum , y is classical , dimension is 1
    #                          nstates=2, model=model)

    # # ehrenfest.sample(initial_state=1)
    # dt = 0.1
    # nt = 10
    # y_steps, py_steps, c_steps = ehrenfest.run(dt=dt, nt = nt, method='euler')


    # np.savez(f'dyn_2d_coupledoscillator_p{E_kin_au}_traj{ntraj}_dt{dt}_Nt{Nt}.npz',
    #          y_steps=y_steps, py_steps=py_steps, c_steps=c_steps)#, py=dyn.py, c=dyn.c, y=dyn.y)

    # # initialization
    # # for nuclear DOF  : an ensemble of trajectories
    # # for electronic DOF  : for each trajectory associate a complex vector c of dimension M

    # ntraj = Ntraj = 10
    # M = nstates = 2
    # #nfit = 5
    # #ax = 1.0 # width of the GH basis
    # ay0 = 16.0
    # y0 = 0.1

    # # initial conditions for c
    # c = np.zeros((Ntraj,M),dtype=np.complex128)

    # # mixture of ground and first excited state

    # c[:,0] = 1.0/np.sqrt(2.0)+0j
    # c[:,1] = 1.0/np.sqrt(2.0)+0j
    # #for i in range(2,M):
    # #    c[:,i] = 0.0+0.0j

    # # coherent state
    # #z = 1.0/np.sqrt(2.0) * x0 * np.sqrt(ax)
    # #for i in range(M):
    # #    c[:,i] = np.exp(-0.5 * np.abs(z)**2) * z**i / np.sqrt(math.factorial(i))

    # print('initial occupation \n',c[0,:])
    # print('trace of density matrix',np.vdot(c[0,:], c[0,:]))
    # # ---------------------------------
    # # initial conditions for nuclear trajectory

    # # ensemble of trajectories
    # y = np.random.randn(ntraj)
    # y = y / np.sqrt(2.0 * ay0) + y0
    # print('trajectory range {}, {}'.format(min(y),max(y)))

    # print('intial nuclear position',y)
    # py = np.zeros(Ntraj)
    # # ry = - ay0 * (y-y0)

    # w = np.array([1./Ntraj]*Ntraj)

    # # -------------------------------

    # amx = 1.0
    # amy = 1836.15

    # f_MSE = open('rMSE.out','w')
    # nout = 1       # number of trajectories to print
    # fmt =  ' {}' * (nout+1)  + '\n'
    # #Eu = 0.

    # Ndim = 1           # dimensionality of the nuclei
    # fric_cons = 0.0      # friction constant


    # Nt = 20000
    # dt = 0.002
    # dt2 = dt/2.0
    # t = 0.0

    # print('time range for propagation is [0,{}]'.format(Nt*dt))
    # print('timestep  = {}'.format(dt))

    # # construct the Hamiltonian matrix for anharmonic oscilator
    # #g = 0.0
    # #V = 0.5 * M2mat(ax,M) + g* M4mat(ax,M)
    # #K = Kmat(ax,0.0,M)
    # #H = K+V

    # #print('Hamiltonian matrix in DOF x = \n')
    # #print(H)
    # #print('\n')

    # #eps = 0.5 # nonlinear coupling Vint = eps*x**2*y**2

    # # @numba.autojit
    # def den(c,w):
    #     """
    #         compute reduced density matrix elements
    #     """
    #     rho = np.zeros((M,M),dtype=np.complex128)
    #     for k in range(Ntraj):
    #         for i in range(M):
    #             for j in range(M):
    #                 rho[i,j] += c[k,i] * np.conjugate(c[k,j]) * w[k]

    #     rho2 = np.dot(rho,rho)

    #     purity = 0.0+0.0j
    #     for i in range(M):
    #         purity += rho2[i,i]

    #     return rho[0,1], purity.real

    # # @numba.autojit
    # def norm(c,w):

    #     anm = 0.0

    #     for k in range(Ntraj):
    #         anm += np.vdot(c[k,:], c[k,:]).real * w[k]
    #     return anm

    # # # @numba.autojit
    # # def fit_c(c,y):
    # #     """
    # #     global approximation of c vs y to obtain the derivative c'',c'
    # #     """
    # #     dc = np.zeros((Ntraj,M),dtype=np.complex128)
    # #     ddc = np.zeros((Ntraj,M),dtype=np.complex128)

    # #     for j in range(M):

    # #         z = c[:,j]
    # #         pars = np.polyfit(y,z,nfit)
    # #         p0 = np.poly1d(pars)
    # #         p1 = np.polyder(p0)
    # #         p2 = np.polyder(p1)
    # # #for k in range(Ntraj):
    # #         dc[:,j] = p1(y)
    # #         ddc[:,j] = p2(y)

    # #     return dc, ddc

    # # @numba.autojit
    # def prop_c(y):

    #     # dc, ddc = fit_c(c,y)

    #     dcdt = np.zeros([ntraj,M],dtype=np.complex128)


    #     #X1 = M1mat(ax,M)
    #     for k in range(ntraj):

    #         H = np.zeros((nstates, nstates))
    #         H[0,0] = ground(y[k])[0]
    #         H[0,1] = H[1,0] = 0.0
    #         H[1,1] = excited(y[k])[0]

    #         # anharmonic term in the bath potential
    #         #Va = y[k]**4 * 1.0

    #         tmp = H.dot(c[k,:])

    #         dcdt[k,:] = -1j * tmp

    #     return dcdt

    # # @numba.autojit
    # def xAve(c,y,w):
    #     """
    #     compute expectation value of x
    #     """
    #     Xmat = M1mat(ax,M)

    #     x_ave = 0.0+0.0j
    #     for k in range(Ntraj):
    #         for m in range(M):
    #             for n in range(M):
    #                 x_ave += Xmat[m,n] * np.conjugate(c[k,m]) * c[k,n] * w[k]

    #     return x_ave.real

    # # propagate the QTs for y


    # # update the coeffcients for each trajectory
    # fmt_c = ' {} '* (M+1)

    # f = open('traj.dat','w')
    # fe = open('en.out','w')
    # fc = open('c.dat','w')
    # fx = open('xAve.dat','w')
    # fnorm = open('norm.dat', 'w')
    # fden = open('den.dat','w')


    # v0, dv = mean_field_force(y,c)

    # cold = c
    # dcdt = prop_c(y)
    # c = c + dcdt * dt

    # for k in range(Nt):

    #     t = t + dt

    #     py += - dv * dt2

    #     y +=  py*dt/amy

    #     # force field

    #     # x_ave = xAve(c,y,w)
    #     v0, dv = mean_field_force(y,c)

    #     py += - dv * dt2

    #     # renormalization

    #     #anm = norm(c,w)
    #     #c /= np.sqrt(anm)

    #     # update c

    #     dcdt = prop_c(y)
    #     cnew = cold + dcdt * dt * 2.0
    #     cold = c
    #     c = cnew


    #     #  output data for each timestep
    # #    d = c
    # #    for k in range(Ntraj):
    # #        for i in range(M):
    # #            d[k,i] = np.exp(-1j*t*H[i,i])*c[k,i]


    #     # fx.write('{} {} \n'.format(t,x_ave))

    #     f.write(fmt.format(t,*y[0:nout]))

    #     #fnorm.write(' {} {} \n'.format(t,anm))

    #     # output density matrix elements
    #     # rho, purity = den(c,w)
    #     # fden.write(' {} {} {} \n'.format(t,rho, purity))

    #     Ek = np.dot(py*py,w)/2./amy
    #     Ev = np.dot(v0,w)

    #     Etot = Ek + Ev

    #     fe.write('{} {} {} {} \n'.format(t,Ek,Ev,Etot))


    # print('The total energy = {} Hartree. \n'.format(Etot))

    # # print trajectory and coefficients
    # for k in range(Ntraj):
    #     fc.write( '{} {} {} \n'.format(y[k], c[k,0],c[k,-1]))

    # fe.close()
    # f.close()
    # fc.close()
    # fx.close()


#a, x0, De = 1.02, 1.4, 0.176/100
#print('The well depth = {} cm-1. \n'.format(De * hartree_wavenumber))
#
#omega  = a * np.sqrt(2. * De / am )
#E0 = omega/2. - omega**2/16./De
#dE = (Etot-E0) * hartree_wavenumber
#print('Exact ground-state energy = {} Hartree. \nEnergy deviation = {} cm-1. \n'.format(E0,dE))
#
