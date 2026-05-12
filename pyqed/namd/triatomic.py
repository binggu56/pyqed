import string
import functools
from collections import Counter
from functools import reduce

import numpy as np
import scipy.linalg
from periodictable import elements
import jax
import jax.numpy as jnp

from pyqed.units import amu2au, au2fs, au2wavenumber
from pyqed.dvr.dvr_1d import SineDVR
from pyqed import interval, au2angstrom
from pyqed.phys import gwp
from pyqed.qchem.mol import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.namd.keo import (
    EPS,
    calculate_exact_keo as calculate_rovibrational_keo,
    hess_log_abs_det_gmat,
    inv,
    jac_Gmat_vib,
    jac_log_abs_det_gmat,
    kron,
)


class Triatom(Molecule):
    """
    Nonadiabatic geometric quantum dynamics for triatomic molecules ABC.
    (vibrational, rovibrational, rovibronic)

    Inherits from Molecule for electronic structure capabilities, and
    integrates LDR-based split-operator propagation directly.
    """

    def __init__(self,
                 atom: str | list[str],
                 basis: str | dict[str, str] = 'sto-3g',
                 nstates: int = 1,
                 charge: int = 0,
                 spin: int = 0,
                 unit: str = 'Angstrom',
                 dvr_type: str = 'sine',
                 J: int = 0,
                 Jz: int | None = None):  # 新增 dvr_type 参数

        # 调用父类初始化
        super().__init__(atom=atom, basis=basis, charge=charge, spin=spin, unit=unit)

        self.nstates = nstates
        self.dvr_type = dvr_type  # 修复: 添加 dvr_type 属性
        self.overlap_matrix=None
        self.ndim=3
        # 初始化其他属性，防止 AttributeError
        self.masses = [1.0, 1.0, 1.0]
        self.x = None
        self.dvrs = None
        self.apes = None
        self.adiabatic_states = None
        self.non_adiabatic_couplings = None
        self.mass = self.atom_mass_list()*amu2au
        self.J = int(J)
        self.Jz = self._validate_Jz(Jz, self.J)
        self.nrot = self._rotational_dimension()
        # 用于存储动力学结果
        self.psi_t = None

    def atom_mass_list(mol):
        '''
        A list of mass for all atoms in the molecule
        '''
        return np.array([elements.isotope(mol.atom_symbol(i)).mass \
                         for i in range(mol.natom)])

    @staticmethod
    def _validate_Jz(Jz, J):
        if Jz is None:
            return None
        Jz = int(Jz)
        if Jz < -J or Jz > J:
            raise ValueError(f"Jz={Jz} is outside the allowed range [-J, J] for J={J}.")
        return Jz

    def _rotational_dimension(self):
        if self.J == 0:
            return 1
        if self.Jz is not None:
            return int(2 * self.J + 1)
        return int((2 * self.J + 1) ** 2)

    def set_rotation(self, J=1, Jz=None):
        """Set the fixed-J rotational basis for rovibronic propagation.

        The rotational basis is the body-fixed ``|J,K,M>`` basis used by
        :func:`pyqed.namd.keo.calculate_exact_keo`, with dimension
        ``(2J + 1)^2``.  If ``Jz`` is supplied, field-free Jz conservation is
        used and only a single fixed-M block is built, with dimension
        ``2J + 1``.
        """
        J = int(J)
        if J < 0:
            raise ValueError("J must be non-negative.")
        self.J = J
        self.Jz = self._validate_Jz(Jz, J)
        self.nrot = self._rotational_dimension()
        return self

    def _rotation_enabled(self):
        return self.J > 0

    def buildK_todo(self,R_1e,R_2e,theta_e,eta_e):
        """
        Build the analytical kinetic energy operator for ABC
        Where mass are refered as M_1,M_0,M_2, and coordinates are r1,r2,theta (Jacobi coordinates in Eckart frame).
        \\eta_e is the Eckart embedding angle, between the Eckart I2 axis and R1

        J. Chem. Phys. 107, 9493–9501 (1997)
        J. Chem. Phys. 107, 2813–2818 (1997)
        """

        M_0, M_1, M_2 = self.M_center, self.M_end1, self.M_end2
        dvrs = self.dvr


        p_r1 = dvrs[0].momentum()
        p_r2 = dvrs[1].momentum()
        p_th = dvrs[2].momentum()
        hbar = 1.0

        r1_grid, r2_grid, th_grid = self.x
        N_r1, N_r2, N_th = self.nx

        I_r1, I_r2, I_th = np.eye(N_r1), np.eye(N_r2), np.eye(N_th)

        P1 = np.kron(p_r1, np.kron(I_r2, I_th))
        P2 = np.kron(I_r1, np.kron(p_r2, I_th))
        P_th_op = np.kron(I_r1, np.kron(I_r2, p_th))

        R1, R2, Th = np.meshgrid(r1_grid, r2_grid, th_grid, indexing='ij')
        r1v, r2v, thv = R1.flatten(), R2.flatten(), Th.flatten()

        eps0 = 1
        eps1 = M_1 / M_0
        eps2 = M_2 / M_0
        eps12 = eps1 + eps2;
        eps11 = eps1 + eps1;
        eps22 = eps2 + eps2;
        eps01 = eps0 + eps1;
        eps02 = eps0 + eps2;
        eps012 = eps1 + eps2 + 1
        rho = R_2e / R_1e
        mu1 = M_0 * M_1 / (M_0 + M_1)
        mu2 = M_0 * M_2 / (M_0 + M_2)
        coste = np.cos(theta_e)
        sinte = np.sin(theta_e)
        sint = np.sin(thv)
        cost = np.cos(thv)
        from numpy import sin, cos
        c = cost
        Laminv = M_0 * (eps1 ** 2 * r1v ** 2 * (eps2 ** 2 * rho ** 2 + eps02 ** 2 - 2 * eps2 * eps02 * rho * coste) + eps2 ** 2 * r2v ** 2 * (
                    eps1 ** 2 + eps01 ** 2 * rho ** 2 - 2 * eps1 * eps01 * rho * coste) +
                        2 * eps1 * eps2 * r1v * r2v * (
                                    eps01 * eps02 * rho * np.cos(thv - theta_e) + eps1 * eps2 * rho * np.cos(thv + theta_e) - (eps1 * eps02 + eps2 * eps01 * rho ** 2) * np.cos(thv)))
        Lam = 1 / Laminv

        G_r1y = Lam * eps012 * eps2 * (r2v * (eps1 * sint - eps01 * rho * np.sin(thv - theta_e)) - r1v * eps1 * rho * sinte)

        G_r2y = Lam * eps012 * eps1 * rho * (r1v * (-eps2 * rho * np.sin(thv) + eps02 * np.sin(thv - theta_e)) + r2v * eps2 * sinte)

        G_cy = Lam * eps012 * sint * (
                (r1v / r2v) * eps1 * (-eps02 * rho * np.cos(thv - theta_e) + eps2 * rho ** 2 * cost)
                - eps2 * eps01 * rho ** 2
                + eps1 * eps02
                + (r2v / r1v) * eps2 * (eps01 * rho * np.cos(thv - theta_e) - eps1 * cost)
        )
        # ---------- 提前确保你定义了 eta_e ----------
        # eta_e = 0.0  # (举例：你需要在这里指定你的 Eckart 嵌入角)

        # 计算辅助变量 S_1, S_2, C_1, C_2 (Eq. 18 - 21)
        S_1 = eps1 * (-eps2 * rho * np.sin(thv + theta_e - eta_e) + eps02 * np.sin(thv - eta_e)) \
              + (eps2 * r2v / r1v) * (eps01 * rho * np.sin(theta_e - eta_e) + eps1 * np.sin(eta_e))

        S_2 = eps2 * (eps01 * rho * np.sin(thv - theta_e + eta_e) - eps1 * np.sin(thv + eta_e)) \
              + (eps1 * r1v / r2v) * (eps2 * rho * np.sin(theta_e - eta_e) + eps02 * np.sin(eta_e))

        C_1 = eps1 * (-eps2 * rho * np.cos(thv + theta_e - eta_e) + eps02 * np.cos(thv - eta_e)) \
              + (eps2 * r2v / r1v) * (eps01 * rho * np.cos(theta_e - eta_e) - eps1 * np.cos(eta_e))

        C_2 = eps2 * (eps01 * rho * np.cos(thv - theta_e + eta_e) - eps1 * np.cos(thv + eta_e)) \
              + (eps1 * r1v / r2v) * (-eps2 * rho * np.cos(theta_e - eta_e) + eps02 * np.cos(eta_e))

        inv_sin2 = 1.0 / (1 - cost ** 2)

        G_zz = Lam * inv_sin2 * (S_1 ** 2 / eps1 + S_2 ** 2 / eps2 + (S_1 + S_2) ** 2)

        G_xx = Lam * inv_sin2 * (C_1 ** 2 / eps1 + C_2 ** 2 / eps2 + (C_1 - C_2) ** 2)

        G_zx = -Lam * inv_sin2 * (S_1 * C_1 / eps1 - S_2 * C_2 / eps2 + (S_1 + S_2) * (C_1 - C_2))

        G_yy = Lam * eps012 * (eps1 * eps02 + eps2 * eps01 * rho ** 2 - 2 * eps1 * eps2 * rho * coste)

        def sandwich(Pl, g, Pr):
            return Pl @ np.diag(g) @ Pr
        Tvib=( sandwich(P1, np.full_like(r1v, 1/mu1), P1)+sandwich(P2, np.full_like(r1v, 1/mu2), P2)+sandwich(P1, np.full_like(r1v, 2*cost/M_0), P1))


        csc_th = 1.0 / np.sin(thv)




        T = 0.5 * (1        ) + 1

        return T

    def _internal_to_xyz_jax(self, q):
        r1, r2, theta = q
        zero = jnp.asarray(0.0, dtype=q.dtype)
        B = jnp.array([zero, zero, zero], dtype=q.dtype)
        A = jnp.array([r1, zero, zero], dtype=q.dtype)
        C = jnp.array([r2 * jnp.cos(theta), r2 * jnp.sin(theta), zero], dtype=q.dtype)
        return jnp.stack([A, B, C], axis=0)

    def build_rovibrational_keo(self, J=None, verbose=True):
        """Build the full vibrational + rotational + Coriolis KEO."""
        if self.dvrs is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        J = self.J if J is None else int(J)
        if J < 0:
            raise ValueError("J must be non-negative.")
        Jz = self._validate_Jz(self.Jz, J)
        return calculate_rovibrational_keo(
            self.dvrs,
            np.asarray(self.mass, dtype=float),
            self._internal_to_xyz_jax,
            mode='all',
            J_val=J,
            M_val=Jz,
            verbose=verbose,
        )

    def buildK(self, J=None):
        """Build the analytical kinetic energy operator in bond-distance /
        angle coordinates (Eckart frame).

             H
        r1  /
           / \\
          O  | theta
           \\/
        r2  \\
             H

        Ref:
        1.J. Chem. Phys. 97, 3029–3037 (1992)[An error in Eq.2]
        2.J. Chem. Phys. 88, 4171–4185 (1988)
        """
        J = self.J if J is None else int(J)
        if J > 0:
            return self.build_rovibrational_keo(J=J, verbose=True)

        self.M_end1, self.M_center, self.M_end2 = self.mass[0], self.mass[1], self.mass[2]
        M_Y, M_X1, M_X2 = self.M_center, self.M_end1, self.M_end2
        dvrs = self.dvrs

        p_r1 = dvrs[0].momentum()
        p_r2 = dvrs[1].momentum()
        p_th = dvrs[2].momentum()
        hbar = 1.0

        r1_grid, r2_grid, th_grid = self.x
        N_r1, N_r2, N_th = self.nx

        I_r1, I_r2, I_th = np.eye(N_r1), np.eye(N_r2), np.eye(N_th)

        P1 = np.kron(p_r1, np.kron(I_r2, I_th))
        P2 = np.kron(I_r1, np.kron(p_r2, I_th))
        P_th_op = np.kron(I_r1, np.kron(I_r2, p_th))

        R1, R2, Th = np.meshgrid(r1_grid, r2_grid, th_grid, indexing='ij')
        r1v, r2v, thv = R1.flatten(), R2.flatten(), Th.flatten()

        val_G11 = 1.0 / M_X1 + 1.0 / M_Y
        val_G22 = 1.0 / M_X2 + 1.0 / M_Y
        val_G12 = np.cos(thv) / M_Y
        val_G1th = -np.sin(thv) / (M_Y * r2v)
        val_G2th = -np.sin(thv) / (M_Y * r1v)

        inv_mu1 = 1.0 / M_X1 + 1.0 / M_Y
        inv_mu2 = 1.0 / M_X2 + 1.0 / M_Y
        val_Gthth = inv_mu1 / r1v**2 + inv_mu2 / r2v**2 \
                    - 2 * np.cos(thv) / (M_Y * r1v * r2v)

        csc_th = 1.0 / np.sin(thv)
        V_metric = np.diag(
            -(hbar**2 / 8.0) * val_Gthth * (1 + csc_th**2)
            - hbar**2 / 2.0 / M_Y * (np.cos(thv) / (r1v * r2v)))

        def sandwich(Pl, g, Pr):
            return Pl @ np.diag(g) @ Pr

        T = 0.5 * (
            sandwich(P1, np.full_like(r1v, val_G11), P1)
            + sandwich(P2, np.full_like(r1v, val_G22), P2)
            + sandwich(P_th_op, val_Gthth, P_th_op)
            + sandwich(P1, val_G12, P2) + sandwich(P2, val_G12, P1)
            + sandwich(P1, val_G1th, P_th_op) + sandwich(P_th_op, val_G1th, P1)
            + sandwich(P2, val_G2th, P_th_op) + sandwich(P_th_op, val_G2th, P2)
        ) + V_metric

        #print(">>> T_total computed.")
        return T

    def calculate_exact_keo(self, dvrs, masses, internal_to_cartesian,
                            mode='T', verbose=True):
        """Exact KEO via JAX automatic differentiation.

        Parameters
        ----------
        dvrs : list of DVR objects
        masses : array-like of atomic masses
        internal_to_cartesian : callable
        mode : 'T' or 'G'
        verbose : bool
        """

        @functools.partial(jax.jit, static_argnums=(2,))
        def pseudo(q, masses, internal_to_cartesian):
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
        def Gmat(q, masses, internal_to_cartesian):
            return inv(gmat(q, masses, internal_to_cartesian))

        if verbose:
            print(f"[KEO] Starting calculation with {len(dvrs)} dimensions...")

        grids = [d.x for d in dvrs]
        mesh = jnp.meshgrid(*grids, indexing='ij')
        q_batch = jnp.stack([m.flatten() for m in mesh], axis=1)
        n_tot = q_batch.shape[0]
        n_dim = len(dvrs)

        if verbose:
            print(f"[KEO] Total grid points: {n_tot}")

        batch_Gmat_fn = jax.vmap(Gmat, in_axes=(0, None, None))
        G_all = batch_Gmat_fn(q_batch, masses, internal_to_cartesian)

        batch_pseudo_fn = jax.vmap(pseudo, in_axes=(0, None, None))
        pseudo_all = batch_pseudo_fn(q_batch, masses, internal_to_cartesian)

        G_vib = G_all[:, :n_dim, :n_dim]

        if mode == 'G':
            return np.array(G_vib)

        if verbose:
            print("[KEO] Assembling full T matrix...")

        Ids = [np.eye(d.npts) for d in dvrs]
        D1s = [d.momentum() for d in dvrs]

        T_mat = np.zeros((n_tot, n_tot), dtype=np.complex128)
        for i in range(n_dim):
            for j in range(n_dim):
                ops_i = [D1s[k] if k == i else Ids[k] for k in range(n_dim)]
                D_i_full = reduce(kron, ops_i)
                if i == j:
                    D_j_full = D_i_full
                else:
                    ops_j = [D1s[k] if k == j else Ids[k] for k in range(n_dim)]
                    D_j_full = reduce(kron, ops_j)
                G_op = np.diag(np.array(G_vib[:, i, j]))
                T_mat += 0.5 * (D_i_full.conj().T @ G_op @ D_j_full)

        T_mat += np.diag(np.array(pseudo_all))
        if verbose:
            print(f"[KEO] T matrix assembled. Shape: {T_mat.shape}")
        return T_mat



    def buildV(self, dt):
        """Build the potential energy propagator (split-operator)."""
        exp_V = np.exp(-1j * dt * self.apes)
        exp_V_half = np.exp(-1j * 0.5 * dt * self.apes)
        if self._rotation_enabled():
            exp_V = np.expand_dims(exp_V, axis=self.ndim)
            exp_V_half = np.expand_dims(exp_V_half, axis=self.ndim)
        self.exp_V = exp_V
        self.exp_V_half = exp_V_half

    def buildH(self, dt):
        """Build the full kinetic propagator exp(-i T dt).

        Uses the analytical KEO in bond/angle coordinates.
        """
        import time
        print("Building T_total ...")
        t0 = time.time()
        has_rotation = self._rotation_enabled()
        T_total = self.buildK()
        print(f"T_total built in {time.time() - t0:.2f} s, shape = {T_total.shape}")

        print("Computing exp(-i T dt) ...")
        t0 = time.time()
        exp_T_full = scipy.linalg.expm(-1j * T_total * dt)
        #print(f"exp(T) computed in {time.time() - t0:.2f} s")
        state_eye = np.eye(self.nstates, dtype=complex)
        if has_rotation:
            exp_T = exp_T_full.reshape(*self.nx, self.nrot, *self.nx, self.nrot)
            idx1 = string.ascii_lowercase[:self.ndim]
            idx2 = string.ascii_lowercase[self.ndim:2 * self.ndim]
            if self.overlap_matrix is None:
                self.exp_T = np.expand_dims(exp_T, axis=self.ndim + 1)
                self.exp_T = np.expand_dims(self.exp_T, axis=-1)
                eye_shape = [1] * self.exp_T.ndim
                eye_shape[self.ndim + 1] = self.nstates
                eye_shape[-1] = self.nstates
                self.exp_T = self.exp_T * state_eye.reshape(eye_shape)
            else:
                expected = (*self.nx, self.nstates, *self.nx, self.nstates)
                if self.overlap_matrix.shape != expected:
                    raise ValueError(
                        f"overlap_matrix shape {self.overlap_matrix.shape} != expected {expected}"
                    )
                self.exp_T = np.einsum(
                    f'{idx1}r{idx2}s,{idx1}y{idx2}x->{idx1}ry{idx2}sx',
                    exp_T,
                    self.overlap_matrix,
                )
        else:
            exp_T = exp_T_full.reshape(*self.nx, *self.nx)
            if self.overlap_matrix is None:
                self.exp_T = exp_T[:, :, :, None, :, :, :, None] * state_eye[
                    None, None, None, :, None, None, None, :
                ]
            else:
                expected = (*self.nx, self.nstates, *self.nx, self.nstates)
                if self.overlap_matrix.shape != expected:
                    raise ValueError(
                        f"overlap_matrix shape {self.overlap_matrix.shape} != expected {expected}"
                    )
                self.exp_T = np.einsum('abcdef,abcidefj->abcidefj', exp_T, self.overlap_matrix)

        self.H = T_total
        return self.exp_T


    # ------------------------------------------------------------------ #
    #  Time propagation (split-operator)
    # ------------------------------------------------------------------ #

    def run(self, psi0, dt, nt, nout=1, t0=0):
        """Run wavepacket propagation using split-operator.

        Parameters
        ----------
        psi0 : ndarray, shape (*nx, nstates)
            Initial wavefunction.
        dt : float
            Time step.
        nt : int
            Total number of time steps.
        nout : int
            Output every `nout` steps.
        t0 : float
            Initial time.

        Returns
        -------
        result : dict with 'times' and 'psilist'.
        """
        has_rotation = self._rotation_enabled()
        expected = (*self.nx, self.nrot, self.nstates) if has_rotation else (*self.nx, self.nstates)
        if psi0.shape != expected:
            raise ValueError(f"psi0 shape {psi0.shape} != expected {expected}")

        # --- Build required operators if not yet done ---
        if self.apes is None:
            raise RuntimeError("APES not built. Call build_apes() first.")

        self.buildV(dt)


        self.buildH(dt)


        # --- Set up einsum contraction for propagation ---
        D = self.ndim
        alphabet = list(string.ascii_lowercase)
        idx1 = "".join(alphabet[:D])
        idx2 = "".join(alphabet[D:2 * D])
        if has_rotation:
            kin_string = f"{idx1}ry{idx2}sx,{idx2}sx->{idx1}ry"
        else:
            kin_string = f"{idx1}y{idx2}x,{idx2}x->{idx1}y"

        # --- Propagate ---
        psilist = [psi0.copy()]
        psi = psi0.copy()



        for k in range(nt // nout):
            for _ in range(nout):
                psi = self.exp_V_half * psi
                psi = np.einsum(kin_string, self.exp_T, psi)
                psi = self.exp_V_half * psi

            psilist.append(psi.copy())
            print(f"Time {t0 + (k + 1) * nout * dt*au2fs:.2f} fs: step {k + 1}/{nt // nout}")
        # final half-step correction


        times = t0 + dt * nout * np.arange(len(psilist))
        return {'times': times, 'psilist': psilist}



    def internal_to_xyz(self, r1, r2, theta):
        """Convert internal coordinates (r1, r2, theta) to Cartesian XYZ.

        Convention:  center atom B at origin,
                     end atom A along +x at distance r1,
                     end atom C at distance r2, angle theta from BA.

        Parameters
        ----------
        r1, r2 : float   Bond lengths (a.u.)
        theta  : float   Bond angle (rad)

        Returns
        -------
        coords : ndarray, shape (3, 3)
            Cartesian coordinates of [A, B, C].
        """
        # B (center) at origin
        B = np.array([0.0, 0.0, 0.0])
        # A along +x
        A = np.array([r1, 0.0, 0.0])
        # C at angle theta from BA
        C = np.array([r2 * np.cos(theta), r2 * np.sin(theta), 0.0])
        return np.array([A, B, C])  # order: end1, center, end2

    def set_dvr(self, dvrs=None, domains=None, npts=None, dvr_type='sine'):
        """
        Set DVRs for internal coordinates (r1, r2, theta).
        Can be called with either existing DVR setup `dvrs` OR `domains` and `npts`.

        Parameters
        ----------
        dvrs : list of DVR objects, optional
            已构建好的 DVR 对象列表。
        domains : list of [min, max], optional
            每个维度的定义域范围，例如 [[1.0, 4.0], ...]。
        npts : list of int, optional
            每个维度的格点数量，例如 [10, 10, 10]。
        dvr_type : str, optional (default 'sine')
            DVR 类型。
        """
        # 情况 1: 通过 domains 和 npts 初始化新的 DVR
        if domains is not None and npts is not None:
            if len(domains) != len(npts):
                raise ValueError("Length of 'domains' and 'npts' must match.")

            self.dvrs = []
            # 简单工厂模式创建 DVR
            for dom, n in zip(domains, npts):
                if dvr_type == 'sine':
                    self.dvrs.append(SineDVR(dom[0], dom[1], n))
                else:
                    self.dvrs.append(SineDVR(dom[0], dom[1], n))

            self.dvr_type = dvr_type

        # 情况 2: 传入已有的 DVR 对象列表
        elif dvrs is not None:
            self.dvrs = dvrs
            if hasattr(dvrs[0], 'type'):
                self.dvr_type = dvrs[0].type
            else:
                self.dvr_type = 'sine'

        else:
            raise ValueError("Must provide either 'dvrs' or both 'domains' and 'npts'.")

        # 统一设置网格坐标和点数属性
        self.x = [d.x for d in self.dvrs]
        self.nx = [len(x) for x in self.x]
        self.dx=[d.dx for d in self.dvrs]
        self.dv=np.prod(self.dx)
        # 兼容旧代码，将属性也暴露为 domain
        if domains is not None:
            self.domain = domains

        return self

    def scan_pes(self, basis="631g*", nstates=None, verbose=0, ncas=6, nelecas=6):
        """
        Scan the adiabatic PES over the DVR grid and compute the non-adiabatic
        overlap matrix. Results are always saved to disk as three files:
            apes.npz            -- adiabatic potential energy surfaces
            overlap_matrix.npz  -- full pairwise overlap matrix
            grid_mc_objects.npz -- raw CASCI objects for every grid point

        If externally provided (self.apes / self.overlap_matrix are already set),
        the scan is skipped entirely.

        Parameters
        ----------
        basis : str
            Gaussian basis set string (default: '631g*').
        nstates : int, optional
            Number of electronic states. Defaults to self.nstates.
        verbose : int
            Verbosity level (currently unused internally).
        ncas : int
            Number of active-space orbitals for CASCI.
        nelecas : int
            Number of active-space electrons for CASCI.

        Returns
        -------
        apes : ndarray, shape (*nx, nstates)
            Adiabatic potential energy surfaces on the DVR grid.
        overlap_matrix : ndarray, shape (*nx, nstates, *nx, nstates)
            Full pairwise overlap matrix between all grid points and states.
        """
        if self.x is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        if not hasattr(self, "internal_to_xyz"):
            raise NotImplementedError(
                "Method 'internal_to_xyz(r1, r2, theta)' must be implemented."
            )

        if nstates is None:
            nstates = self.nstates
        nx = self.nx  # [N_r1, N_r2, N_theta]
        meta = dict(nx=nx, nstates=nstates, basis=basis, ncas=ncas, nelecas=nelecas)

        # --- 0. Skip if external inputs are already provided -------------------
        if self.apes is not None and self.overlap_matrix is not None:
            print("[scan_pes] External PES and overlap matrix already set. Skipping.")
            return self.apes, self.overlap_matrix

        # --- 1. Scan: compute CASCI at every grid point ------------------------
        print(f"[scan_pes] Scanning PES (basis={basis}, ncas={ncas}, nelecas={nelecas}) ...")
        r1_grid, r2_grid, th_grid = self.x
        total = np.prod(nx)
        count = 0

        grid_mc_objects = np.empty(nx, dtype=object)
        self.apes = np.zeros((*nx, nstates))

        for i, r1 in enumerate(r1_grid):
            for j, r2 in enumerate(r2_grid):
                for k, theta in enumerate(th_grid):
                    xyz = self.internal_to_xyz(r1, r2, theta)
                    atom_spec = [[s, c] for s, c in zip(self.atom_symbols(), xyz)]

                    pmol = Molecule(
                        atom=atom_spec, basis=basis,
                        charge=self.charge, spin=self.spin, unit=self.unit,
                    )
                    pmol.build()
                    mc = CASCI(RHF(pmol).run(), ncas=ncas, nelecas=nelecas)
                    mc.run(nstates=nstates)

                    grid_mc_objects[i, j, k] = mc
                    self.apes[i, j, k, :] = np.atleast_1d(np.asarray(mc.e_tot))[:nstates]

                    count += 1
                    if count % max(1, total // 10) == 0:
                        print(f"  ... {count}/{total} points computed")

        np.savez("grid_mc_objects.npz", data=grid_mc_objects, meta=np.array(meta))
        np.savez("apes.npz", data=self.apes, meta=np.array(meta))
        print("[scan_pes] Saved grid_mc_objects.npz and apes.npz")

        # --- 2. Compute full overlap matrix ------------------------------------
        print("[scan_pes] Computing full overlap matrix ...")
        flat_mc = grid_mc_objects.flatten()
        N = len(flat_mc)
        S = np.zeros((N, nstates, N, nstates))

        for a in range(N):
            for b in range(a, N):
                ov_ab = overlap(flat_mc[a], flat_mc[b])
                S[a, :, b, :] = ov_ab
                if a != b:
                    S[b, :, a, :] = ov_ab.T

        self.overlap_matrix = S.reshape((*nx, nstates, *nx, nstates))
        np.savez("overlap_matrix.npz", data=self.overlap_matrix, meta=np.array(meta))
        print("[scan_pes] Saved overlap_matrix.npz")

        return self.apes, self.overlap_matrix

    def get_population(self, result, plot=True):
        """
        计算并绘制各电子态的布局数 (Population)。

        Parameters
        ----------
        result : dict
            run() 函数的返回结果，必须包含 'psilist' 和 'times'。
        plot : bool, optional
            是否绘制布局数随时间变化的图像，默认为 True。

        Returns
        -------
        pops : ndarray
            形状为 (n_steps, nstates) 的布局数数组。
        """
        psilist = result['psilist']
        times = result['times']

        pops = []
        for psi in psilist:
            # 自动识别空间维度：由于 psi 的形状是 (*nx, nstates)，
            # 我们对除了最后一个维度（电子态）以外的所有维度求和
            spatial_axes = tuple(range(psi.ndim - 1))

            prob = np.sum(np.abs(psi) ** 2, axis=spatial_axes)*self.dv
            pops.append(prob)

        pops = np.array(pops)

        if plot:
            import ultraplot as uplt
            fig,ax=uplt.subplots(figsize=(8, 6))

            nstates = pops.shape[1]
            for s in range(nstates):
                ax.plot(times, pops[:, s], label=f'State {s}', linewidth=2)
            #uplt.xlabel('Time (fs)')
            #uplt.ylabel('Population')
            #uplt.legend()
            #uplt.grid(False)
            #plt.tight_layout()
            fig.show()

        return pops


# ====================================================================== #
#  Main script
# ====================================================================== #

if __name__ == '__main__':
    import time

    nstates = 2

    import numpy as np

    r_OH = 1/au2angstrom # bohr
    theta = 104.5 * np.pi / 180.0

    atom_H3 = [['H', (r_OH, 0., 0.)],
               ['H', (0., 0., 0.)],
               ['H', (r_OH * np.cos(theta), r_OH * np.sin(theta), 0.)]]

    mol = Triatom(atom_H3, nstates=nstates, basis='631g*', charge=1, spin=0)
    print(mol.mass)
    # --- 2. DVR grids (r1, r2, theta) ---
    mol.set_dvr(domains=[[1.0, 2.2], [1.0, 2.2], [1.0, 2.2]],
                npts=[5, 5,5])

    # --- 3. Scan adiabatic PES via CASCI ---
    t0 = time.time()
    mol.overlap_matrix = np.load("overlap_matrix.npz")['data']
    mol.apes = np.load("apes.npz")['data']
    #二选一
    #mol.scan_pes(basis='631g', ncas=3, nelecas=3)
    print(f"PES scan completed in {time.time() - t0:.1f} s")


    # --- 4. Initial wavepacket ---
    nx = mol.nx
    a=b=c=3
    # 根据索引 a, b, c 获取中心坐标
    center = [mol.x[0][a], mol.x[1][b], mol.x[2][c]]

    # --- 逐点调用 gwp 构造 3D 高斯波包 ---
    psi0 = np.zeros((*nx, nstates), dtype=complex)
    for i in range(nx[0]):
        for j in range(nx[1]):
            for k in range(nx[2]):
                pt = np.array([mol.x[0][i], mol.x[1][j], mol.x[2][k]])
                # 传入 a_matrix 和动态获取的 center
                psi0[i, j, k,1] = gwp(pt, x0=center, ndim=3)


    # psi0 = np.einsum('ijka,ijkab->ijkb', psi0, mol.adiabatic_states)
    print('The norm of psi0', np.sqrt(np.sum(np.abs(psi0) ** 2) * mol.dv))
    psi0_new = np.einsum('ijlk,ijl->ijlk', mol.overlap_matrix[:, :, :, :, 1, 1, 1, 1], psi0[:, :, :, 1])



    # --- 5. Propagate ---
    dt = 0.1 / au2fs
    result = mol.run(psi0=psi0, dt=dt, nt=100, nout=1)
    print(f"Done. Snapshots saved: {len(result['psilist'])}")
    p=mol.get_population(result, plot=True)
