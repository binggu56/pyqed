import numpy
import numpy as np
import scipy

PI = np.pi

from pyqed import Molecule
from pyqed.qchem.atomic_data import atom_names, bohr_to_angs, vdw_radii
from pyqed.qchem.dft.grid import lebedev_sphere
from pyqed.qchem.pcm_integrals import (
    nuclear_potential_at_surface,
    surface_charge_ao_coulomb,
)
from pyqed.qchem.solvent import _attach_solvent

def pcm_for_scf(mf, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mf.mol)
    return _attach_solvent._for_scf(mf, solvent_obj, dm)

def pcm_for_casci(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casci(mc, solvent_obj, dm)

def pcm_for_casscf(mc, solvent_obj=None, dm=None):
    if solvent_obj is None:
        if isinstance(getattr(mc._scf, 'with_solvent', None), PCM):
            solvent_obj = mc._scf.with_solvent
        else:
            solvent_obj = PCM(mc.mol)
    return _attach_solvent._for_casscf(mc, solvent_obj, dm)

def pcm_for_tdscf(td, solvent_obj=None, dm=None, equilibrium_solvation=False):
    return _attach_solvent._for_tdscf(
        td,
        solvent_obj=solvent_obj,
        dm=dm,
        equilibrium_solvation=equilibrium_solvation,
    )



# TABLE II,  J. Chem. Phys. 122, 194110 (2005)
XI = {
    6: 4.84566077868,
    14: 4.86458714334,
    26: 4.85478226219,
    38: 4.90105812685,
    50: 4.89250673295,
    86: 4.89741372580,
    110: 4.90101060987,
    146: 4.89825187392,
    170: 4.90685517725,
    194: 4.90337644248,
    302: 4.90498088169,
    350: 4.86879474832,
    434: 4.90567349080,
    590: 4.90624071359,
    770: 4.90656435779,
    974: 4.90685167998,
    1202: 4.90704098216,
    1454: 4.90721023869,
    1730: 4.90733270691,
    2030: 4.90744499142,
    2354: 4.90753082825,
    2702: 4.90760972766,
    3074: 4.90767282394,
    3470: 4.90773141371,
    3890: 4.90777965981,
    4334: 4.90782469526,
    4802: 4.90749125553,
    5294: 4.90762073452,
    5810: 4.90792902522,
}

LEBEDEV_ORDER = {
    3: 6,
    5: 14,
    7: 26,
    9: 38,
    11: 50,
    13: 74,
    15: 86,
    17: 110,
    19: 146,
    21: 170,
    23: 194,
    25: 230,
    27: 266,
    29: 302,
    31: 350,
    35: 434,
    41: 590,
    47: 770,
    53: 974,
    59: 1202,
    65: 1454,
    71: 1730,
    77: 2030,
    83: 2354,
    89: 2702,
    95: 3074,
    101: 3470,
    107: 3890,
    113: 4334,
    119: 4802,
    125: 5294,
    131: 5810,
}


def _modified_bondi_radii():
    """Return a PySCF-compatible vdW radius table indexed by nuclear charge."""
    fallback = 2.0 / bohr_to_angs
    table = numpy.full(len(atom_names) + 1, fallback)
    table[0] = 0.0
    for charge, symbol in enumerate(atom_names, start=1):
        table[charge] = vdw_radii.get(symbol, fallback)
    table[1] = 1.1 / bohr_to_angs
    return table


def _lebedev_order_to_npoints(order):
    if order in LEBEDEV_ORDER:
        return LEBEDEV_ORDER[order]
    if order in XI:
        return order
    raise ValueError(f"Unsupported Lebedev order/grid size: {order}")


def _prange(start, stop, step):
    for p0 in range(start, stop, step):
        yield p0, min(p0 + step, stop)


def _current_memory_mb():
    # The native PCM code only needs an approximate allocator budget for chunking.
    return 0.0


modified_Bondi = _modified_bondi_radii()


def switch_h(x):
    '''
    switching function (eq. 3.19)
    J. Chem. Phys. 133, 244111 (2010)
    notice the typo in the paper
    '''
    y = x**3 * (10.0 - 15.0*x + 6.0*x**2)
    y[x<0] = 0.0
    y[x>1] = 1.0
    return y

def gen_surface(mol, ng=302, rad=modified_Bondi, vdw_scale=1.2):
    '''J. Phys. Chem. A 1999, 103, 11060-11079'''
    directions, angular_weights = lebedev_sphere(ng)
    atom_coords = mol.atom_coords()
    charges = mol.atom_charges()
    N_J = ng * numpy.ones(mol.natom)
    R_J = numpy.asarray([rad[chg] for chg in charges])
    R_sw_J = R_J * (14.0 / N_J)**0.5
    alpha_J = 1.0/2.0 + R_J/R_sw_J - ((R_J/R_sw_J)**2 - 1.0/28)**0.5
    R_in_J = R_J - alpha_J * R_sw_J

    grid_coords = []
    weights = []
    charge_exp = []
    switch_fun = []
    R_vdw = []
    norm_vec = []
    area = []
    gslice_by_atom = []
    p0 = p1 = 0
    for ia in range(mol.natom):
        symb = mol.atom_symbol(ia)
        # chg = gto.charge(symb)
        chg = mol.atom_charge(ia)
        r_vdw = rad[chg]

        atom_grid = r_vdw * directions + atom_coords[ia,:]
        riJ = scipy.spatial.distance.cdist(atom_grid[:,:3], atom_coords)
        diJ = (riJ - R_in_J) / R_sw_J
        diJ[:,ia] = 1.0
        diJ[diJ < 1e-8] = 0.0
        fiJ = switch_h(diJ)

        w = angular_weights
        swf = numpy.prod(fiJ, axis=1)
        idx = w*swf > 1e-16

        p0, p1 = p1, p1+sum(idx)
        gslice_by_atom.append([p0,p1])
        grid_coords.append(atom_grid[idx,:3])
        weights.append(w[idx])
        switch_fun.append(swf[idx])
        norm_vec.append(directions[idx,:3])
        xi = XI[ng] / (r_vdw * w[idx]**0.5)
        charge_exp.append(xi)
        R_vdw.append(numpy.ones(sum(idx)) * r_vdw)
        area.append(w[idx]*r_vdw**2*swf[idx])

    grid_coords = numpy.vstack(grid_coords)
    norm_vec = numpy.vstack(norm_vec)
    weights = numpy.concatenate(weights)
    charge_exp = numpy.concatenate(charge_exp)
    switch_fun = numpy.concatenate(switch_fun)
    area = numpy.concatenate(area)
    R_vdw = numpy.concatenate(R_vdw)

    surface = {
        'ng': ng,
        'gslice_by_atom': gslice_by_atom,
        'grid_coords': grid_coords,
        'weights': weights,
        'charge_exp': charge_exp,
        'switch_fun': switch_fun,
        'R_vdw': R_vdw,
        'norm_vec': norm_vec,
        'area': area,
        'R_in_J': R_in_J,
        'R_sw_J': R_sw_J,
        'atom_coords': atom_coords
    }
    return surface
    
def get_F_A(surface):
    '''
    generate F and A matrix in  J. Chem. Phys. 133, 244111 (2010)
    '''
    R_vdw = surface['R_vdw']
    switch_fun = surface['switch_fun']
    weights = surface['weights']
    A = weights*R_vdw**2*switch_fun
    return switch_fun, A

def get_D_S(surface, with_S=True, with_D=False):
    '''
    generate D and S matrix in  J. Chem. Phys. 133, 244111 (2010)
    The diagonal entries of S is not filled
    '''
    charge_exp  = surface['charge_exp']
    grid_coords = surface['grid_coords']
    switch_fun  = surface['switch_fun']
    norm_vec    = surface['norm_vec']
    R_vdw       = surface['R_vdw']

    xi_i, xi_j = numpy.meshgrid(charge_exp, charge_exp, indexing='ij')
    xi_ij = xi_i * xi_j / (xi_i**2 + xi_j**2)**0.5
    rij = scipy.spatial.distance.cdist(grid_coords, grid_coords)
    xi_r_ij = xi_ij * rij
    numpy.fill_diagonal(rij, 1)
    S = scipy.special.erf(xi_r_ij) / rij
    numpy.fill_diagonal(S, charge_exp * (2.0 / PI)**0.5 / switch_fun)

    D = None
    if with_D:
        drij = numpy.expand_dims(grid_coords, axis=1) - grid_coords
        nrij = numpy.sum(drij * norm_vec, axis=-1)

        D = S*nrij/rij**2 -2.0*xi_r_ij/PI**0.5*numpy.exp(-xi_r_ij**2)*nrij/rij**3
        numpy.fill_diagonal(D, -charge_exp * (2.0 / PI)**0.5 / (2.0 * R_vdw))

    return D, S

class PCM:
    '''
    PCM Solvent Model

    This class implements the Polarizable Continuum Model (PCM) for solvent effects.

    Input Attributes:
    -----------------
    method : str
        The PCM model. Options include 'C-PCM', 'IEF-PCM', 'COSMO', and 'SS(V)PE'.
        Default is 'C-PCM'.

    vdw_scale : float
        A scaling factor for van der Waals radii. Default is 1.2, consistent with Q-Chem settings.

    r_probe : float
        An additional radius (in Angstrom) added to the van der Waals radii.
        Default is 0.0.

    radii_table : dict
        Custom van der Waals radii for each element. By default, scaled van der Waals radii
        from `vdw_scale` and `r_probe` are used.

    lebedev_order : int
        The order of the Lebedev mesh used for the cavity sphere. Default is 29 (302 grids).

    eps : float
        The dielectric constant of the solvent. Default is 78.3553, the dielectric constant
        for water.

    frozen : bool
        Whether to freeze the potential produced by the solvent during SCF iterations or
        other convergence processes. When frozen=True is set, the solvent is
        assumed to respond slowly, while the electron density relaxes quickly.
        Default is False.

    max_cycle : int
        The maximum number of iterations to relax the solvent.

    conv_tol : float
        The convergence tolerance for total energy during solvent relaxation.

    equilibrium_solvation : bool
        Affects TDDFT and other excited state computations. Controls whether the solvent
        relaxes rapidly with respect to the electron density of the excited state.
        For vertical excitations, it is recommended to set this to False, as the solvent
        typically does not fully relax. In some software packages (e.g., Q-Chem),
        non-equilibrium solvation is applied with an optical dielectric constant of
        eps=1.78. Default is False.

    state_id : int
        Specifies the target state in excited state calculations.
        `state_id=0` corresponds to the ground state, while `state_id=1` corresponds
        to the first excited state. Default is 0.

    state_average : bool
        If True, update the solvent from an equal-weight average of all requested
        electronic-state densities. This is useful for state-averaged CASSCF/CASCI
        benchmarks.

    state_weights : array_like or None
        Optional explicit state weights for the solvent density. If set, this
        overrides state_average.

    Saved Results:
    --------------
    e_tot : float
        The energy contribution from the solvent.

    v : ndarray
        The potential matrix generated by the solvent.

    Intermediate Attributes:
    ------------------------
    These attributes are generated during calculations and should not be modified.
    Additionally, they may not be compatible between GPU and CPU implementations.

    - surface
    - _intermediates
    - v_grids_n
    '''

    _keys = {
        'method', 'vdw_scale', 'surface', 'r_probe',
        'mol', 'radii_table', 'lebedev_order',
        'eps', 'max_cycle', 'conv_tol', 'state_id', 'state_average',
        'state_weights', 'frozen',
        'equilibrium_solvation', 'integral_backend', 'e', 'v', 'v_grids_n',
    }

    def kernel(self, dm):
        """Compute the PCM energy and AO potential for a density matrix."""
        self._dm = numpy.asarray(dm)
        self.e, self.v = self._get_vind(self._dm)
        return self.e, self.v

    def __init__(self, mol):
        self.mol = mol
        # self.stdout = mol.stdout
        # self.verbose = mol.verbose
        self.verbose = 3
        # self.max_memory = mol.max_memory
        self.max_memory = 1000
        self.method = 'C-PCM'

        self.vdw_scale = 1.2 # default value in qchem
        self.r_probe = 0.0
        self.radii_table = None
        self.lebedev_order = 29
        self.eps = 78.3553

        self.max_cycle = 20
        self.conv_tol = 1e-7
        self.state_id = 0
        self.state_average = False
        self.state_weights = None

        self.frozen = False
        self.equilibrium_solvation = False
        self.integral_backend = 'auto'

        self.surface = {}
        self._intermediates = {}
        self.v_grids_n = None # nuclear potential on grids

        self.e = None
        self.v = None
        self._dm = None

    def _resolve_integral_backend(self):
        backend = str(self.integral_backend).lower()
        if backend in {'native', 'pyscf'}:
            return backend
        if backend == 'auto':
            return 'native'
        else:
            raise ValueError("PCM integral_backend must be 'auto', 'native', or 'pyscf'.")


    # def dump_flags(self, verbose=None):
    #     logger.info(self, '******** %s (In testing) ********', self.__class__)
    #     logger.warn(self, 'PCM is an experimental feature. It is '
    #                 'still in testing.\nFeatures and APIs may be changed '
    #                 'in the future.')
    #     logger.info(self, 'lebedev_order = %s (%d grids per sphere)',
    #                 self.lebedev_order, _lebedev_order_to_npoints(self.lebedev_order))
    #     logger.info(self, 'eps = %s'          , self.eps)
    #     logger.info(self, 'frozen = %s'       , self.frozen)
    #     #logger.info(self, 'equilibrium_solvation = %s', self.equilibrium_solvation)
    #     return self

    # def to_gpu(self):
    #     from pyscf.lib import to_gpu
    #     obj = to_gpu(self)
    #     return obj.reset()

    def reset(self, mol=None):
        if mol is not None:
            self.mol = mol
        self._intermediates = None
        self.surface = None
        self.v_grids_n = None
        return self

    def build(self, ng=None):
        if self.radii_table is None:
            vdw_scale = self.vdw_scale
            radii_table = vdw_scale * modified_Bondi + self.r_probe/bohr_to_angs
        else:
            radii_table = self.radii_table
        # logger.debug2(self, 'radii_table %s', radii_table)
        mol = self.mol
        if ng is None:
            ng = _lebedev_order_to_npoints(self.lebedev_order)

        self.surface = gen_surface(mol, rad=radii_table, ng=ng)
        self._intermediates = {}
        backend = self._resolve_integral_backend()
        self._intermediates['integral_backend'] = backend
        F, A = get_F_A(self.surface)
        D, S = get_D_S(self.surface, with_S=True, with_D=True)

        epsilon = self.eps
        if self.verbose:
            print('eps', self.eps)
        if self.method.upper() in ['C-PCM', 'CPCM']:
            f_epsilon = (epsilon-1.)/epsilon if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
            if self.verbose:
                print('f_epsilon', f_epsilon)
                print('R[0]', R[0,0])
        elif self.method.upper() == 'COSMO':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0/2.0) if epsilon != float('inf') else 1.0
            K = S
            R = -f_epsilon * numpy.eye(K.shape[0])
        elif self.method.upper() in ['IEF-PCM', 'IEFPCM']:
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(2.0*PI) * DAS
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        elif self.method.upper() == 'SS(V)PE':
            f_epsilon = (epsilon - 1.0)/(epsilon + 1.0) if epsilon != float('inf') else 1.0
            DA = D*A
            DAS = numpy.dot(DA, S)
            K = S - f_epsilon/(4.0*PI) * (DAS + DAS.T)
            R = -f_epsilon * (numpy.eye(K.shape[0]) - 1.0/(2.0*PI)*DA)
        else:
            raise RuntimeError(f"Unknown implicit solvent model: {self.method}")

        intermediates = {
            'S': S,
            'D': D,
            'A': A,
            'K': K,
            'R': R,
            'f_epsilon': f_epsilon
        }
        self._intermediates.update(intermediates)

        charge_exp  = self.surface['charge_exp']
        grid_coords = self.surface['grid_coords']
        atom_coords = mol.atom_coords()
        atom_charges = mol.atom_charges()

        if backend == 'native':
            self.v_grids_n = nuclear_potential_at_surface(
                atom_coords,
                atom_charges,
                grid_coords,
                charge_exp**2,
            )
        elif backend == 'pyscf':
            from pyqed.qchem.mol import fakemol_for_charges, intor_cross

            int2c2e = mol._add_suffix('int2c2e')
            fakemol = fakemol_for_charges(grid_coords, expnt=charge_exp**2)
            fakemol_nuc = fakemol_for_charges(atom_coords)
            v_ng = intor_cross(int2c2e, fakemol_nuc, fakemol)
            self.v_grids_n = numpy.dot(atom_charges, v_ng)
        else:
            raise ValueError("PCM integral_backend must be 'native' or 'pyscf'.")

    def _surface_coulomb_tensor(self):
        v_nj = self._intermediates.get('surface_coulomb_tensor')
        if v_nj is None:
            backend = self._intermediates.get('integral_backend', self._resolve_integral_backend())
            if backend == 'native':
                v_nj = surface_charge_ao_coulomb(
                    self.mol,
                    self.surface['grid_coords'],
                    self.surface['charge_exp']**2,
                )
            elif backend == 'pyscf':
                from pyscf import df
                from pyqed.qchem.mol import fakemol_for_charges

                grid_coords = self.surface['grid_coords']
                exponents = self.surface['charge_exp']
                int3c2e = self.mol._add_suffix('int3c2e')
                fakemol = fakemol_for_charges(grid_coords, expnt=exponents**2)
                v_nj = df.incore.aux_e2(self.mol, fakemol, intor=int3c2e, aosym='s1')
            else:
                raise ValueError("PCM integral_backend must be 'native' or 'pyscf'.")
            self._intermediates['surface_coulomb_tensor'] = v_nj
        return v_nj

    def _get_vind(self, dms):
        if not self._intermediates:
            self.build()

        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)
        if dms.shape[0] == 2:
            dms = (dms[0] + dms[1]).reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids_e = self._get_v(dms)
        # print('v_grids_e', v_grids_e)
        v_grids = self.v_grids_n - v_grids_e
        # print('v_grids[0]', v_grids[0])

        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T
        # print('R', R) # not equal
        # print('b', b[:10]) # MOT EQUAL
        # print('q[0]', q[0])  # not equal to pyscf

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        epcm = 0.5 * numpy.dot(q_sym[0], v_grids[0])

        self._intermediates['q'] = q[0]
        self._intermediates['q_sym'] = q_sym[0]
        self._intermediates['v_grids'] = v_grids[0]
        self._intermediates['dm'] = dms
        return epcm, vmat[0]

    def _get_v(self, dms):
        '''
        return electrostatic potential on surface
        '''
        nao = dms.shape[-1]
        grid_coords = self.surface['grid_coords']
        ngrids = grid_coords.shape[0]
        nset = dms.shape[0]
        v_grids_e = numpy.empty([nset, ngrids])
        v_nj_all = self._surface_coulomb_tensor()
        max_memory = self.max_memory - _current_memory_mb()
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))
        for p0, p1 in _prange(0, ngrids, blksize):
            v_nj = v_nj_all[:, :, p0:p1]
            for i in range(nset):
                v_grids_e[i,p0:p1] = numpy.einsum('ijL,ij->L',v_nj, dms[i])

        return v_grids_e

    def _get_vmat(self, q):
        mol = self.mol
        nao = mol.nao
        grid_coords = self.surface['grid_coords']
        ngrids = grid_coords.shape[0]
        q = q.reshape([-1,ngrids])
        nset = q.shape[0]
        vmat = numpy.zeros([nset,nao,nao])
        v_nj_all = self._surface_coulomb_tensor()
        max_memory = self.max_memory - _current_memory_mb()
        blksize = int(max(max_memory*.9e6/8/nao**2, 400))

        for p0, p1 in _prange(0, ngrids, blksize):
            v_nj = v_nj_all[:, :, p0:p1]
            for i in range(nset):
                vmat[i] += -numpy.einsum('ijL,L->ij', v_nj, q[i,p0:p1])
        return vmat

    # def nuc_grad_method(self, grad_method):
    #     raise DeprecationWarning('Use the make_grad_object function from '
    #                              'pyscf.solvent.grad.pcm or '
    #                              'pyscf.solvent._ddcosmo_tdscf_grad instead.')

    # def grad(self, dm):
    #     '''Computes the Jacobian for the energy associated with the solvent,
    #     including the derivatives of the solvent itsself and the interactions
    #     between the solvent and the charge density of the solute.
    #     '''
    #     from pyscf.solvent.grad.pcm import grad_qv, grad_nuc, grad_solver
    #     de_solvent = grad_qv(self, dm)
    #     de_solvent+= grad_nuc(self, dm)
    #     de_solvent+= grad_solver(self, dm)
    #     return de_solvent

    # def Hessian(self, hess_method):
    #     raise DeprecationWarning('Use the make_hessian_object function from '
    #                              'pyscf.solvent.hessian.pcm instead.')

    # def hess(self, dm):
    #     '''Computes the Hessian for the energy associated with the solvent,
    #     including the derivatives of the solvent itsself and the interactions
    #     between the solvent and the charge density of the solute.
    #     '''
    #     from pyscf.solvent.hessian.pcm import (
    #         analytical_hess_nuc, analytical_hess_qv, analytical_hess_solver)
    #     de_solvent  =    analytical_hess_nuc(self, dm, verbose=self.verbose)
    #     de_solvent +=     analytical_hess_qv(self, dm, verbose=self.verbose)
    #     de_solvent += analytical_hess_solver(self, dm, verbose=self.verbose)
    #     return de_solvent

    def _B_dot_x(self, dms):
        if not self._intermediates:
            self.build()
        out_shape = dms.shape
        nao = dms.shape[-1]
        dms = dms.reshape(-1,nao,nao)

        K = self._intermediates['K']
        R = self._intermediates['R']
        v_grids = -self._get_v(dms)
        # print('v_grids')

        b = numpy.dot(R, v_grids.T)
        q = numpy.linalg.solve(K, b).T

        vK_1 = numpy.linalg.solve(K.T, v_grids.T)
        qt = numpy.dot(R.T, vK_1).T
        q_sym = (q + qt)/2.0

        vmat = self._get_vmat(q_sym)
        return vmat.reshape(out_shape)

    # def run(self,dm):
    #     from pyscf.solvent.ddcosmo import ddCOSMO
    #     self.e, self.v = ddCOSMO._get_vind(self, dm)


if __name__ == '__main__':

    # from pyscf import scf, gto, lib

    # mol = gto.Mole()
    # mol.atom = 'Li 0 0 0; H 0 0 1.4' 
    # mol.basis = 'sto-3g' 
    # mol.unit = 'Bohr'
    # mol.build()

    # mf = scf.RHF(mol)
    # mf.run()
    # dm = mf.make_rdm1()



    from pyqed import Molecule
    from pyqed.qchem import solvent
    # from pyscf import gto, solvent

    mol = Molecule(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    # mol = gto.M(atom='Li 0 0 0; H 0 0 1.4', unit='b', basis='sto3g')
    mol.build()

    mf = mol.RHF()

    mf = solvent.PCM(mf)
    # pcm = PCM(mol)

    mf.run()
    # print('ener', pcm.e)
    # e_with_solvent = mf.e_tot + pcm.e
    # print('e_with_solvent', e_with_solvent)
