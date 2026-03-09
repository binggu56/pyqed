import re
import numpy as np
from collections import Counter
from pyqed.units import atomic_mass, amu2au
from jax import numpy as jnp
from eckart import eckart
from kinetic import *
# from pyqed.ldr.ldr import *


class Triatom:
    """
    nonadiabatic geometric quantum dynamics for triatomic molecules ABC
    (vibrational, rovibrational, rovibronic)
    """
    def __init__(self, symbols, coord='eckart', with_rotation=False):
        
        # Parse chemical formula into a list of atoms
        atoms = self._parse_formula(symbols)
        if len(atoms) != 3:
            raise ValueError(f"A triatomic molecule is required, but {len(atoms)} atoms were parsed: {atoms}")

        count = Counter(atoms)

        if len(count) == 2 and set(count.values()) == {1, 2}:
            # X2Y or XY2 pattern
            for elem, c in count.items():
                if c == 2:
                    X_symbol = elem
                else:
                    Y_symbol = elem

            # Mass order: [central Y, X, X]
            # corresponding to r1 = Y-X, r2 = Y-X, theta = X-Y-X
            self.M_center = atomic_mass[Y_symbol.upper()] * amu2au
            self.M_end1 = atomic_mass[X_symbol.upper()] * amu2au
            self.M_end2 = atomic_mass[X_symbol.upper()] * amu2au
            self.masses = [self.M_center, self.M_end1, self.M_end2]
            print(f"X2Y molecule: X={X_symbol}(×2), Y={Y_symbol} (center)")

        elif len(count) == 3:
            # ABC pattern is not yet supported: use X2Y with the central atom in the middle position
            # A_symbol, B_symbol, C_symbol = atoms[0], atoms[1], atoms[2]
            # self.M_end1 = atomic_mass[A_symbol.upper()] * amu2au   # A (terminal atom 1)
            # self.M_center = atomic_mass[B_symbol.upper()] * amu2au  # B (central atom)
            # self.M_end2 = atomic_mass[C_symbol.upper()] * amu2au   # C (terminal atom 2)
            # Mass order: [central B, terminal A, terminal C]
            # corresponding to r1 = B-A, r2 = B-C, theta = A-B-C
            # self.masses = [self.M_center, self.M_end1, self.M_end2]
            print("ABC is not supported yet. Please use X2Y format with the central atom in the middle position, e.g., 'H2O'")
        else:
            raise ValueError(f"Unsupported triatomic molecule type: {symbols}")

        # Reduced masses (for G-matrix)
        self.mu1 = (self.M_center * self.M_end1) / (self.M_center + self.M_end1)
        self.mu2 = (self.M_center * self.M_end2) / (self.M_center + self.M_end2)

        print(f"M_center = {self.M_center:.3f} a.u., M_end1 = {self.M_end1:.3f} a.u., M_end2 = {self.M_end2:.3f} a.u.")

        # dvrr1, dvrr2, dvrth = dvrs
        # self.r1_grid, self.p_r1 = dvrr1.x, dvrr1.momentum()
        # self.r2_grid, self.p_r2 = dvrr2.x, dvrr2.momentum()
        # self.th_grid, self.p_th = dvrth.x, dvrth.momentum()
        # self.N_r1 = len(self.r1_grid)
        # self.N_r2 = len(self.r2_grid)
        # self.N_th = len(self.th_grid)
        # self.hbar = 1.0

        # coordinates
        assert coord.lower() in ['eckart', 'jacobi', 'cartesian']
        self.coord = coord

        if not with_rotation:
            self.dim = 3 # vibration only
        else:
            self.dim = 6 # rotation + vibration

        self.x = None # list of grids




    @staticmethod
    def _parse_formula(formula):
        """Parse chemical formula string into a list of atoms.

        Example:
            'H2O' -> ['H', 'H', 'O']
        """
        tokens = re.findall(r'([A-Z][a-z]?")(\d*)', formula)
        atoms = []
        for elem, num in tokens:
            if elem == '':
                continue
            n = int(num) if num else 1
            atoms.extend([elem] * n)
        return atoms

    def buildK(self):
        """
        build the analytical kinetic energy operator in bond distance and
        angle coordinates
             H
        r1  /
           / \
          O  | theta
           \/
        r2  \
             H

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        M_Y = self.M_center
        M_X1 = self.M_end1
        M_X2 = self.M_end2

        p_r1_1d = self.p_r1
        p_r2_1d = self.p_r2
        p_th_1d = self.p_th
        hbar = self.hbar

        r1_grid = self.r1_grid
        r2_grid = self.r2_grid
        th_grid = self.th_grid

        N_r1 = self.N_r1
        N_r2 = self.N_r2
        N_th = self.N_th

        I_r1 = np.eye(N_r1)
        I_r2 = np.eye(N_r2)
        I_th = np.eye(N_th)

        P1 = np.kron(p_r1_1d, np.kron(I_r2, I_th))
        P2 = np.kron(I_r1, np.kron(p_r2_1d, I_th))
        P_th_op = np.kron(I_r1, np.kron(I_r2, p_th_1d))

        R1_mesh, R2_mesh, Th_mesh = np.meshgrid(r1_grid, r2_grid, th_grid, indexing='ij')
        r1_vec = R1_mesh.flatten()
        r2_vec = R2_mesh.flatten()
        th_vec = Th_mesh.flatten()

        val_G11 = 1.0 / M_X1 + 1.0 / M_Y
        val_G22 = 1.0 / M_X2 + 1.0 / M_Y
        val_G12 = np.cos(th_vec) / M_Y
        val_G1th = -np.sin(th_vec) / (M_Y * r2_vec)
        val_G2th = -np.sin(th_vec) / (M_Y * r1_vec)

        inv_mu1 = 1.0 / M_X1 + 1.0 / M_Y
        inv_mu2 = 1.0 / M_X2 + 1.0 / M_Y
        val_Gthth = inv_mu1 / r1_vec ** 2 + inv_mu2 / r2_vec ** 2 - 2 * np.cos(th_vec) / (M_Y * r1_vec * r2_vec)

        csc_th = 1.0 / np.sin(th_vec)
        V_metric_val = (-(hbar ** 2 / 8.0) * val_Gthth * (1 + csc_th ** 2)
                        - hbar ** 2 / 2.0 / M_Y * (np.cos(th_vec) / (r1_vec * r2_vec)))
        V_metric_mat = np.diag(V_metric_val)

        def sandwich(P_left, G_vals, P_right):
            return P_left @ np.diag(G_vals) @ P_right

        T_11 = sandwich(P1, np.full_like(r1_vec, val_G11), P1)
        T_22 = sandwich(P2, np.full_like(r1_vec, val_G22), P2)
        T_thth = sandwich(P_th_op, val_Gthth, P_th_op)
        T_12 = sandwich(P1, val_G12, P2) + sandwich(P2, val_G12, P1)
        T_1th = sandwich(P1, val_G1th, P_th_op) + sandwich(P_th_op, val_G1th, P1)
        T_2th = sandwich(P2, val_G2th, P_th_op) + sandwich(P_th_op, val_G2th, P2)

        T_total = 0.5 * (T_11 + T_22 + T_thth + T_12 + T_1th + T_2th) + V_metric_mat

        print(">>> T_total computed.")
        return T_total

    def calculate_exact_keo(dvrs, masses, internal_to_cartesian, mode='T', verbose=True):
        """Exact KEO Calculator.

        Args:
            dvrs (list): List of DVR objects (e.g., [dvr_r1, dvr_theta]).
            masses (list): List of atomic masses.
            internal_to_cartesian (function): Coordinate mapping function (JAX compatible).
            mode (str): 'T' returns kinetic energy matrix (Hamiltonian term), 'G' returns G-matrix values.
            verbose (bool): Whether to print construction process information.

        Returns:
            np.ndarray:
                If mode='T': Kinetic energy matrix with shape (N_tot, N_tot).
                If mode='G': G-matrix value array with shape (N_tot, N_dim, N_dim).
        """

        @functools.partial(jax.jit, static_argnums=(2,))
        def pseudo(
                q: np.ndarray,
                masses: np.ndarray,
                internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
        ):
            """Pseudopotential (or extrapotential) implementation according to Eq. (21)
            in Edit Mátyus, Gábor Czakó, and Attila G. Császár,
            J. Chem. Phys. 130, 134112 (2009)
            http://dx.doi.org/10.1063/1.3076742
            """
            nq = len(q)
            G = Gmat(q, masses, internal_to_cartesian)[:nq, :nq]
            dG = jac_Gmat_vib(q, masses, internal_to_cartesian)
            k = jnp.arange(nq)
            dG = dG[k, :, k]
            dlogdet = jac_log_abs_det_gmat(q, masses, internal_to_cartesian)
            hlogdet = hess_log_abs_det_gmat(q, masses, internal_to_cartesian)
            pseudo1 = dlogdet @ G @ dlogdet
            pseudo2 = jnp.sum(dG @ dlogdet) + jnp.sum(G * hlogdet)
            return (pseudo1 + 4 * pseudo2) / 32.0

        @functools.partial(jax.jit, static_argnums=2)
        def gmat(q, masses, internal_to_cartesian):
            # xyz_g = jax.jacfwd(internal_to_cartesian)(jnp.asarray(q))
            xyz_g = jax.jacrev(internal_to_cartesian)(jnp.asarray(q))
            tvib = xyz_g
            xyz = internal_to_cartesian(jnp.asarray(q))
            trot = jnp.transpose(EPS @ xyz.T, (2, 0, 1))
            ttra = jnp.array([jnp.eye(3, dtype=jnp.float64) for _ in range(len(xyz))])
            tvec = jnp.concatenate((tvib, trot, ttra), axis=2)
            masses_sq = jnp.sqrt(jnp.asarray(masses))
            tvec = tvec * masses_sq[:, None, None]
            tvec = jnp.reshape(tvec, (len(xyz) * 3, len(q) + 6))

            return tvec.T @ tvec

        @functools.partial(jax.jit, static_argnums=2)
        def Gmat(
                q: np.ndarray,
                masses: np.ndarray,
                internal_to_cartesian: Callable[[jnp.ndarray], jnp.ndarray],
        ):
            """Compute the kinetic energy G-matrix for a molecular system.
            
            .. math::
                
                T = 0.5  \sum_{i,j} p_i  G_{ij}(q)  p_j


            Args:
                q (np.ndarray): Internal coordinates with shape (3N-6,),
                    where N is the number of atoms. Bond lengths are in Å,
                    and angles are in radians.
                masses (np.ndarray): 1D array of atomic masses. The order of atoms
                    in `masses` must match the order of atoms in the output of
                    `internal_to_cartesian`.
                internal_to_cartesian (Callable): Converts internal coordinates `q`
                    into Cartesian coordinates, returning an array of shape
                    (number of atoms, 3).

            Returns:
                np.ndarray: Square matrix of shape (ncoo+3+3, ncoo+3+3), representing
                the elements of the kinetic energy G-matrix. The first `ncoo` rows
                and columns correspond to vibrational coordinates, followed by three
                rotational and three translational coordinates. Units: cm^-1.
            """
            return inv(gmat(q, masses, internal_to_cartesian))

        if verbose: print(f"[KEO] Starting calculation with {len(dvrs)} dimensions...")
        global pesudo_all

        grids = [d.x for d in dvrs]
        mesh = jnp.meshgrid(*grids, indexing='ij')

        q_batch = jnp.stack([m.flatten() for m in mesh], axis=1)
        n_tot = q_batch.shape[0]
        n_dim = len(dvrs)

        if verbose: print(f"[KEO] Total grid points: {n_tot} (Shape: {q_batch.shape})")

        if verbose: print("[KEO] Computing exact G-matrix via JAX AD...")

        batch_Gmat_fn = jax.vmap(Gmat, in_axes=(0, None, None))
        G_all = batch_Gmat_fn(q_batch, masses, internal_to_cartesian)

        batch_pseudo_fn = jax.vmap(pseudo, in_axes=(0, None, None))
        pesudo_all = batch_pseudo_fn(q_batch, masses, internal_to_cartesian)

        print(np.array(G_all).shape)

        G_vib = G_all[1, :n_dim, :n_dim]
        G_rot = G_all[1, n_dim:n_dim + 3, n_dim:n_dim + 3]
        G_tra = G_all[1, n_dim + 3:, n_dim + 3:]
        G_rot_vib = G_all[1, :n_dim, n_dim:n_dim + 3]
        G_tra_vib = G_all[1, :n_dim, n_dim + 3:]
        G_vib_rot = G_all[1, n_dim:n_dim + 3, :n_dim]
        G_vib_tra = G_all[1, n_dim + 3:, :n_dim]

        print(np.array(G_all[1, :, :]))
        print(np.array(G_vib))
        print(np.array(G_rot))
        print(np.array(G_tra))

        if mode == 'G':
            if verbose: print("[KEO] Returning G-matrix values only.")
            return np.array(G_vib)


        if verbose: print("[KEO] Assembling full Hamiltonian matrix T...")

        Ids = [np.eye(d.npts) for d in dvrs]
        D1s = [d.momentum() for d in dvrs]

        T_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)

        for i in range(n_dim):
            for j in range(n_dim):

                # D_i_full = I x I x ... x D_i x ... x I
                ops_i = [D1s[k] if k == i else Ids[k] for k in range(n_dim)]
                D_i_full = reduce(kron, ops_i)

                if i == j:
                    D_j_full = D_i_full
                else:
                    ops_j = [D1s[k] if k == j else Ids[k] for k in range(n_dim)]
                    D_j_full = reduce(kron, ops_j)

                g_diag_values = np.array(G_vib[:, i, j])
                # print('Point',i,j,'G_ij matrix:',g_diag_values.min())
                G_op = np.diag(g_diag_values)

                term = 0.5 * (D_i_full.conj().T @ G_op @ D_j_full)

                T_mat += term

        T_mat += np.diag(pesudo_all)
        if verbose: print(f"[KEO] T matrix assembled. Shape: {T_mat.shape}")
        return T_mat

    def internal_to_xyz(self):
        pass

    def run(self, domain, npts=[15, 15, 15], dvr_type='sine'):

        from pyqed.dvr.dvr_1d import SineDVR

        # generate the DVR grids

        N_r1, N_r2, N_th = npts

        dvr_r1 = SineDVR(1.0, 4.0, N_r1)    # r1 (bond length 1)
        dvr_r2 = SineDVR(1.0, 4.0, N_r2)    # r2 (bond length 2)
        dvr_th = SineDVR(1.2, 2.8, N_th)    # theta (bond angle)

        self.dvr = [dvr_r1, dvr_r2, dvr_th]
        self.x = [dvr.x for dvr in self.dvr]



        # compute the KEO in the DVR
        T = self.buildK()

        # build the potential energy and overlap matrix

        #


        pass


if __name__ == '__main__':

    import scipy.linalg
    from pyqed.phys import gwp, discretize
    from pyqed.units import au2fs, au2wavenumber
    import time

    # ===== 1. Define system and DVR basis =====
    N_r1, N_r2, N_th =7,7,7
    dvr_r1 = SineDVR(1.0, 4.0, N_r1)    # r1 (bond length 1)
    dvr_r2 = SineDVR(1.0, 4.0, N_r2)    # r2 (bond length 2)
    dvr_th = SineDVR(1.2, 2.8, N_th)    # theta (bond angle)
    dvrs = [dvr_r1, dvr_r2, dvr_th]

    nstates = 2

    # ===== 2. Initialize Triatom class =====
    mol = Triatom('H2O')

    # ===== 3. Triatom builds full kinetic energy matrix T_total, pass to LDRN =====
    start_time = time.time()
    T_total = mol.buildK()
    print(f"T_total built in {time.time() - start_time:.2f} s, shape = {T_total.shape}")


    domains = [[1.0, 4.0], [1.0, 4.0], [1.2, 2.8]]
    levels = [3,3,3]  # so that 2^level - 1 = N per dimension

    solver = LDRN(domains, levels, nstates=nstates,
                  mass=[mol.mu1, mol.mu2, 1.0], ndim=3)

    nx_list = solver.nx  # e.g. [N_r1, N_r2, N_th]
    x = solver.x         # [r1_grid, r2_grid, th_grid]
    dx = solver.dx


    solver.apes = np.zeros((*nx_list, nstates))
    solver.adiabatic_states = np.zeros((*nx_list, nstates, nstates))



    dt = 0.2 / au2fs
    nt = 100
    nout = 1

    print("Computing exp(-i T_total dt) ...")
    start_time = time.time()
    exp_T_full = scipy.linalg.expm(-1j * T_total * dt)
    print(f"exp(T) computed in {time.time() - start_time:.2f} s, shape = {exp_T_full.shape}")


    N_tot = N_r1 * N_r2 * N_th
    exp_T = exp_T_full.reshape(N_r1, N_r2, N_th, N_r1, N_r2, N_th)

    nr1, nr2, nth = nx_list
    psi0 = np.zeros((nr1, nr2, nth, nstates), dtype=complex)
    for i in range(nr1):
        for j in range(nr2):
            for k in range(nth):
                psi0[i, j, k, 0] = gwp(np.array([x[0][i], x[1][j], x[2][k]]),
                                        x0=[2.0, 2.0, 2.0], ndim=3)


    psi0 = np.einsum('ijka, ijkab -> ijkb', psi0, solver.adiabatic_states)

    # ===== 6. Build potential propagator =====
    solver.buildV(dt)



    print('Starting time evolution with LDRN...')
    result = solver.run(psi0=psi0, dt=dt, nt=nt, nout=nout)
    print('Time evolution completed.')