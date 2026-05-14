import string
import functools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import reduce

import numpy as np
import scipy.linalg
from periodictable import elements
import jax
import jax.numpy as jnp

from pyqed.units import amu2au, au2fs, au2wavenumber
from pyqed.dvr.dvr_1d import LegendreDVR, PODVR, SineDVR
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

warnings.filterwarnings("ignore", message="AM1 model is under testing")


def _normalize_triatomic_electronic_method(method):
    method = str(method).lower().replace("_", "-")
    aliases = {
        "cas": "casci",
        "rhf-casci": "casci",
        "am1": "am1-meci",
        "meci": "am1-meci",
        "am1/meci": "am1-meci",
    }
    return aliases.get(method, method)


def _set_worker_thread_limits(worker_threads):
    if worker_threads is None:
        return
    value = str(int(worker_threads))
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = value


def _triatomic_scan_point_worker(task):
    (
        idx,
        xyz,
        atom_symbols,
        basis,
        charge,
        spin,
        unit,
        ncas,
        nelecas,
        nstates,
        electronic_method,
        electronic_options,
    ) = task
    atom_spec = [[symbol, tuple(coord)] for symbol, coord in zip(atom_symbols, xyz)]
    pmol = Molecule(atom=atom_spec, basis=basis, charge=charge, spin=spin, unit=unit)
    electronic_method = _normalize_triatomic_electronic_method(electronic_method)

    if electronic_method == "casci":
        pmol.build()
        mf = RHF(pmol).run()
        mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
        mc.run(nstates=nstates)
        energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))[:nstates]
        return idx, energies, mc

    if electronic_method == "am1-meci":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyqed.qchem.semiempirical.am1 import RAM1

        mf = RAM1(pmol).run(
            conv_tol=float(electronic_options.get("scf_tol", 1.0e-9)),
            max_cycle=int(electronic_options.get("max_cycle", 120)),
            verbose=int(electronic_options.get("verbose", 0)),
            damping=float(electronic_options.get("damping", 0.0)),
        )
        mc = mf.MECI(nstates=nstates, ncas=ncas).run()
        energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))[:nstates]
        return idx, energies, mc

    raise ValueError(
        "electronic_method must be 'casci' or 'am1/meci' "
        f"(got {electronic_method!r})."
    )


def _electronic_state_overlap(left, right):
    """Return an electronic-state overlap matrix for LDR scans."""
    if hasattr(left, "wavefunction_overlap"):
        return left.wavefunction_overlap(right)
    return overlap(left, right)


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
                 dvr_type: str = 'default',
                 J: int = 0,
                 Jz: int | None = None):  # 新增 dvr_type 参数

        # 调用父类初始化
        super().__init__(atom=atom, basis=basis, charge=charge, spin=spin, unit=unit)

        self.nstates = nstates
        self.dvr_type = dvr_type  # 修复: 添加 dvr_type 属性
        self.overlap_matrix=None
        self.overlap_links = None
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

    def _build_flat_kinetic_matrix(self, T_total):
        """Build the flat LDR kinetic matrix in the propagation basis."""
        has_rotation = self._rotation_enabled()
        state_eye = np.eye(self.nstates, dtype=complex)
        overlap_links = getattr(self, "overlap_links", None)

        if has_rotation:
            ng = int(np.prod(self.nx))
            if self.overlap_matrix is None:
                if overlap_links is not None:
                    return self._build_linked_flat_kinetic_matrix(T_total, overlap_links)
                return np.kron(T_total, state_eye)

            expected = (*self.nx, self.nstates, *self.nx, self.nstates)
            if self.overlap_matrix.shape != expected:
                raise ValueError(
                    f"overlap_matrix shape {self.overlap_matrix.shape} != expected {expected}"
                )
            T_rs = T_total.reshape(ng, self.nrot, ng, self.nrot)
            A = self.overlap_matrix.reshape(ng, self.nstates, ng, self.nstates)
            K = np.einsum('irms,iamb->iramsb', T_rs, A, optimize=True)
            K = K.reshape(ng * self.nrot * self.nstates, ng * self.nrot * self.nstates)
            return 0.5 * (K + K.conj().T)

        if self.overlap_matrix is None:
            if overlap_links is not None:
                return self._build_linked_flat_kinetic_matrix(T_total, overlap_links)
            return np.kron(T_total, state_eye)

        expected = (*self.nx, self.nstates, *self.nx, self.nstates)
        if self.overlap_matrix.shape != expected:
            raise ValueError(
                f"overlap_matrix shape {self.overlap_matrix.shape} != expected {expected}"
            )
        ng = int(np.prod(self.nx))
        A = self.overlap_matrix.reshape(ng, self.nstates, ng, self.nstates)
        K = np.einsum('im,iamb->iamb', T_total, A, optimize=True)
        K = K.reshape(ng * self.nstates, ng * self.nstates)
        return 0.5 * (K + K.conj().T)

    def _build_linked_flat_kinetic_matrix(self, T_total, links, threshold=0.0):
        """Build the flat LDR kinetic matrix directly from nearest-neighbor links.

        This avoids materializing the full ``overlap_matrix`` tensor.  For dense
        DVR kinetic operators the final flat kinetic matrix is still dense, but
        the electronic overlap blocks are generated on demand from link products.
        """
        indices = self._grid_indices()
        ng = len(indices)
        nstates = self.nstates
        state_eye = np.eye(nstates, dtype=complex)
        has_rotation = self._rotation_enabled()

        def linked_block(i, j, bra_idx, ket_idx):
            if i == j:
                return state_eye
            if i < j:
                return self._linked_overlap_between(bra_idx, ket_idx, links, nstates)
            return self._linked_overlap_between(ket_idx, bra_idx, links, nstates).conj().T

        if has_rotation:
            T_rs = T_total.reshape(ng, self.nrot, ng, self.nrot)
            K = np.zeros((ng, self.nrot, nstates, ng, self.nrot, nstates), dtype=complex)
            for i, bra_idx in enumerate(indices):
                for j, ket_idx in enumerate(indices):
                    block = T_rs[i, :, j, :]
                    if threshold > 0.0 and np.max(np.abs(block)) <= threshold:
                        continue
                    Aij = linked_block(i, j, bra_idx, ket_idx)
                    K[i, :, :, j, :, :] = (
                        block[:, None, :, None] * Aij[None, :, None, :]
                    )
            K = K.reshape(ng * self.nrot * nstates, ng * self.nrot * nstates)
            return 0.5 * (K + K.conj().T)

        K = np.zeros((ng, nstates, ng, nstates), dtype=complex)
        for i, bra_idx in enumerate(indices):
            for j, ket_idx in enumerate(indices):
                Tij = T_total[i, j]
                if threshold > 0.0 and abs(Tij) <= threshold:
                    continue
                Aij = linked_block(i, j, bra_idx, ket_idx)
                K[i, :, j, :] = Tij * Aij
        K = K.reshape(ng * nstates, ng * nstates)
        return 0.5 * (K + K.conj().T)

    def _linked_overlap_block(self, i, j, bra_idx, ket_idx, links, nstates):
        """Return the linked-product overlap block using dense-A convention."""
        if i == j:
            return np.eye(nstates, dtype=complex)
        if i < j:
            return self._linked_overlap_between(bra_idx, ket_idx, links, nstates)
        return self._linked_overlap_between(ket_idx, bra_idx, links, nstates).conj().T

    def _build_kinetic_linear_operator(self, T_total, threshold=0.0):
        """Build a matrix-free LDR kinetic operator.

        The returned ``LinearOperator`` applies the flat kinetic matrix without
        constructing that matrix explicitly.  With ``overlap_links`` this also
        avoids materializing the full pairwise electronic overlap tensor.
        """
        from scipy.sparse.linalg import LinearOperator

        has_rotation = self._rotation_enabled()
        indices = self._grid_indices()
        ng = len(indices)
        nstates = self.nstates
        links = getattr(self, "overlap_links", None)
        nflat = ng * (self.nrot if has_rotation else 1) * nstates
        dtype = np.result_type(T_total.dtype, complex)

        if has_rotation:
            T_rs = np.asarray(T_total).reshape(ng, self.nrot, ng, self.nrot)

            if self.overlap_matrix is not None:
                A = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, self.nrot, nstates)
                    out = np.einsum("irjs,iajb,jsb->ira", T_rs, A, psi, optimize=True)
                    return out.reshape(-1)

            elif links is not None:

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, self.nrot, nstates)
                    out = np.zeros_like(psi, dtype=dtype)
                    for i, bra_idx in enumerate(indices):
                        for j, ket_idx in enumerate(indices):
                            block = T_rs[i, :, j, :]
                            if threshold > 0.0 and np.max(np.abs(block)) <= threshold:
                                continue
                            Aij = self._linked_overlap_block(
                                i, j, bra_idx, ket_idx, links, nstates
                            )
                            transported = psi[j] @ Aij.T
                            out[i] += np.einsum("rs,sa->ra", block, transported)
                    return out.reshape(-1)

            else:

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, self.nrot, nstates)
                    out = np.einsum("irjs,jsa->ira", T_rs, psi, optimize=True)
                    return out.reshape(-1)

        else:
            T = np.asarray(T_total)

            if self.overlap_matrix is not None:
                A = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, nstates)
                    out = np.einsum("ij,iajb,jb->ia", T, A, psi, optimize=True)
                    return out.reshape(-1)

            elif links is not None:

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, nstates)
                    out = np.zeros_like(psi, dtype=dtype)
                    for i, bra_idx in enumerate(indices):
                        for j, ket_idx in enumerate(indices):
                            Tij = T[i, j]
                            if threshold > 0.0 and abs(Tij) <= threshold:
                                continue
                            Aij = self._linked_overlap_block(
                                i, j, bra_idx, ket_idx, links, nstates
                            )
                            out[i] += Tij * (Aij @ psi[j])
                    return out.reshape(-1)

            else:

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, nstates)
                    out = T @ psi
                    return out.reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        return LinearOperator(
            shape=(nflat, nflat),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=dtype,
        )

    def _kinetic_trace_from_nuclear_operator(self, T_total):
        """Trace of the flat electronic/rotational kinetic operator."""
        ng = int(np.prod(self.nx))
        if self._rotation_enabled():
            T_rs = np.asarray(T_total).reshape(ng, self.nrot, ng, self.nrot)
            return self.nstates * np.trace(T_rs.reshape(ng * self.nrot, ng * self.nrot))
        return self.nstates * np.trace(T_total)

    @staticmethod
    def _canonical_kinetic_propagator(kinetic_propagator):
        kinetic_propagator = kinetic_propagator.lower()
        aliases = {
            "krylov": "expm_multiply",
            "expm-multiply": "expm_multiply",
            "cheb": "chebyshev",
            "cheby": "chebyshev",
        }
        kinetic_propagator = aliases.get(kinetic_propagator, kinetic_propagator)
        if kinetic_propagator not in ("dense", "expm_multiply", "chebyshev"):
            raise ValueError(
                "kinetic_propagator must be 'dense', 'expm_multiply', or 'chebyshev'."
            )
        return kinetic_propagator

    def _estimate_chebyshev_bounds(self, method="endpoints", margin=1e-12):
        from scipy.sparse.linalg import LinearOperator

        method = method.lower()
        n = self.H.shape[0]
        if isinstance(self.H, LinearOperator) and method != "eigsh":
            method = "eigsh"

        if method == "exact":
            evals = scipy.linalg.eigvalsh(self.H)
            emin = float(evals[0])
            emax = float(evals[-1])
        elif method in ("endpoints", "subset"):
            emin = float(scipy.linalg.eigh(
                self.H,
                eigvals_only=True,
                subset_by_index=[0, 0],
            )[0])
            emax = float(scipy.linalg.eigh(
                self.H,
                eigvals_only=True,
                subset_by_index=[n - 1, n - 1],
            )[0])
        elif method == "eigsh":
            from scipy.sparse.linalg import eigsh

            emin = float(eigsh(self.H, k=1, which="SA", return_eigenvectors=False)[0])
            emax = float(eigsh(self.H, k=1, which="LA", return_eigenvectors=False)[0])
        elif method == "gershgorin":
            diag = np.real(np.diag(self.H))
            radii = np.sum(np.abs(self.H), axis=1) - np.abs(np.diag(self.H))
            emin = float(np.min(diag - radii))
            emax = float(np.max(diag + radii))
        else:
            raise ValueError(
                "chebyshev_bounds must be 'endpoints', 'exact', 'eigsh', or 'gershgorin'."
            )

        span = emax - emin
        if span > 0:
            pad = max(float(margin) * span, np.finfo(float).eps * max(abs(emin), abs(emax), 1.0))
            emin -= pad
            emax += pad
        return emin, emax

    def _prepare_chebyshev_kinetic(
        self,
        dt,
        tol=1e-12,
        max_order=4096,
        bounds="gershgorin",
        bounds_margin=1e-12,
    ):
        from scipy.special import jv

        emin, emax = self._estimate_chebyshev_bounds(method=bounds, margin=bounds_margin)
        center = 0.5 * (emax + emin)
        radius = 0.5 * (emax - emin)
        if radius == 0:
            order = 0
            coeffs = np.array([np.exp(-1j * center * dt)], dtype=complex)
        else:
            dt_sign = 1.0 if dt >= 0 else -1.0
            z = radius * abs(dt)
            coeffs_list = [jv(0, z)]
            small_tail = 0
            order = 0
            for k in range(1, max_order + 1):
                coeff = 2.0 * ((-1j * dt_sign) ** k) * jv(k, z)
                coeffs_list.append(coeff)
                order = k
                if abs(coeff) < tol and k > z:
                    small_tail += 1
                    if small_tail >= 8:
                        break
                else:
                    small_tail = 0
            else:
                raise RuntimeError(
                    f"Chebyshev expansion did not converge within max_order={max_order}."
                )
            coeffs = np.asarray(coeffs_list, dtype=complex)

        self.chebyshev_center = center
        self.chebyshev_radius = radius
        self.chebyshev_coeffs = coeffs
        self.chebyshev_phase = np.exp(-1j * center * dt)
        self.chebyshev_order = order
        self.chebyshev_bounds = (emin, emax)
        self.chebyshev_bounds_method = bounds

    def buildH(
        self,
        dt,
        kinetic_propagator="dense",
        chebyshev_tol=1e-12,
        chebyshev_max_order=4096,
        chebyshev_bounds="gershgorin",
        matrix_free_kinetic=False,
    ):
        """Build the full kinetic propagator exp(-i T dt).

        Uses the analytical KEO in bond/angle coordinates.
        """
        import time
        kinetic_propagator = self._canonical_kinetic_propagator(kinetic_propagator)

        print("Building T_total ...")
        t0 = time.time()
        has_rotation = self._rotation_enabled()
        T_total = self.buildK()
        T_total = 0.5 * (T_total + T_total.conj().T)
        self.kinetic_trace = self._kinetic_trace_from_nuclear_operator(T_total)
        print(f"T_total built in {time.time() - t0:.2f} s, shape = {T_total.shape}")

        if matrix_free_kinetic:
            if kinetic_propagator == "dense":
                raise ValueError(
                    "matrix_free_kinetic=True requires kinetic_propagator="
                    "'expm_multiply' or 'chebyshev'."
                )
            print("Building matrix-free kinetic LinearOperator ...")
            t0 = time.time()
            self.H = self._build_kinetic_linear_operator(T_total)
            self.T_total = T_total
            print(
                f"LinearOperator built in {time.time() - t0:.2f} s, "
                f"shape = {self.H.shape}"
            )
        else:
            print("Building flat kinetic matrix ...")
            t0 = time.time()
            self.H = np.ascontiguousarray(self._build_flat_kinetic_matrix(T_total))
            print(f"Flat kinetic matrix built in {time.time() - t0:.2f} s, shape = {self.H.shape}")

        if kinetic_propagator == "chebyshev":
            print("Preparing Chebyshev kinetic expansion ...")
            t0 = time.time()
            self._prepare_chebyshev_kinetic(
                dt,
                tol=chebyshev_tol,
                max_order=chebyshev_max_order,
                bounds=chebyshev_bounds,
            )
            print(
                "Chebyshev expansion prepared in "
                f"{time.time() - t0:.2f} s, order = {self.chebyshev_order}, "
                f"bounds = {self.chebyshev_bounds_method}"
            )
            self.exp_T = None
            self.kinetic_propagator = kinetic_propagator
            return self.exp_T

        if kinetic_propagator == "expm_multiply":
            self.exp_T = None
            self.kinetic_propagator = kinetic_propagator
            return self.exp_T

        print("Computing exp(-i T dt) ...")
        t0 = time.time()
        exp_T_full = scipy.linalg.expm(-1j * self.H * dt)
        if has_rotation:
            self.exp_T = exp_T_full.reshape(*self.nx, self.nrot, self.nstates,
                                             *self.nx, self.nrot, self.nstates)
        else:
            self.exp_T = exp_T_full.reshape(*self.nx, self.nstates,
                                             *self.nx, self.nstates)
        print(f"exp(-i T dt) computed in {time.time() - t0:.2f} s")
        self.kinetic_propagator = kinetic_propagator
        return self.exp_T

    def _apply_kinetic_expm_multiply(self, psi, dt):
        from scipy.sparse.linalg import expm_multiply

        kwargs = {}
        if hasattr(self, "kinetic_trace"):
            kwargs["traceA"] = -1j * dt * self.kinetic_trace
        propagated = expm_multiply(-1j * dt * self.H, psi.reshape(-1), **kwargs)
        return propagated.reshape(psi.shape)

    def _apply_kinetic_chebyshev(self, psi):
        vec0 = psi.reshape(-1)
        coeffs = self.chebyshev_coeffs
        if self.chebyshev_radius == 0:
            return (coeffs[0] * vec0).reshape(psi.shape)

        def scaled_matvec(vec):
            return (self.H @ vec - self.chebyshev_center * vec) / self.chebyshev_radius

        accum = coeffs[0] * vec0
        if len(coeffs) > 1:
            tkm1 = vec0
            tk = scaled_matvec(vec0)
            accum = accum + coeffs[1] * tk
            for k in range(2, len(coeffs)):
                tkp1 = 2.0 * scaled_matvec(tk) - tkm1
                accum = accum + coeffs[k] * tkp1
                tkm1, tk = tk, tkp1

        return (self.chebyshev_phase * accum).reshape(psi.shape)


    # ------------------------------------------------------------------ #
    #  Time propagation (split-operator)
    # ------------------------------------------------------------------ #

    def run(
        self,
        psi0,
        dt,
        nt,
        nout=1,
        t0=0,
        kinetic_propagator="dense",
        chebyshev_tol=1e-12,
        chebyshev_max_order=4096,
        chebyshev_bounds="gershgorin",
        matrix_free_kinetic=False,
    ):
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


        kinetic_propagator = self._canonical_kinetic_propagator(kinetic_propagator)
        self.buildH(
            dt,
            kinetic_propagator=kinetic_propagator,
            chebyshev_tol=chebyshev_tol,
            chebyshev_max_order=chebyshev_max_order,
            chebyshev_bounds=chebyshev_bounds,
            matrix_free_kinetic=matrix_free_kinetic,
        )


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
                if kinetic_propagator == "expm_multiply":
                    psi = self._apply_kinetic_expm_multiply(psi, dt)
                elif kinetic_propagator == "chebyshev":
                    psi = self._apply_kinetic_chebyshev(psi)
                else:
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

    def set_dvr(self, dvrs=None, domains=None, npts=None, dvr_type=None, dvr_params=None):
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
        dvr_type : str or sequence, optional
            DVR type.  The default triatomic setup is
            ``['podvr', 'podvr', 'legendre']``: Morse-reference PODVRs for
            the two bond stretches and Legendre DVR for the bending angle.
            Use ``'sine'`` to recover the old uniform sine DVR behavior.
        dvr_params : dict or sequence of dict, optional
            Per-coordinate keyword arguments passed to the DVR constructor.
        """
        # 情况 1: 通过 domains 和 npts 初始化新的 DVR
        if domains is not None and npts is not None:
            if len(domains) != len(npts):
                raise ValueError("Length of 'domains' and 'npts' must match.")

            self.dvrs = []
            if dvr_type is None:
                dvr_type = self.dvr_type
            if isinstance(dvr_type, str):
                if dvr_type == 'default':
                    dvr_types = ['podvr', 'podvr', 'legendre']
                else:
                    dvr_types = [dvr_type] * len(npts)
            else:
                dvr_types = list(dvr_type)
            if len(dvr_types) != len(npts):
                raise ValueError("Length of 'dvr_type' and 'npts' must match.")

            if dvr_params is None:
                param_list = [{} for _ in npts]
            elif isinstance(dvr_params, dict):
                param_list = [dvr_params.copy() for _ in npts]
            else:
                param_list = [dict(params) for params in dvr_params]
            if len(param_list) != len(npts):
                raise ValueError("Length of 'dvr_params' and 'npts' must match.")

            for dom, n, kind, params in zip(domains, npts, dvr_types, param_list):
                kind = kind.lower()
                if kind == 'sine':
                    self.dvrs.append(SineDVR(dom[0], dom[1], n, **params))
                elif kind == 'podvr':
                    self.dvrs.append(PODVR(dom[0], dom[1], n, **params))
                elif kind in ('legendre', 'legendre_dvr'):
                    self.dvrs.append(LegendreDVR(dom[0], dom[1], n, **params))
                else:
                    raise ValueError(f"Unsupported DVR type: {kind!r}.")

            self.dvr_type = dvr_types

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
        self.w = [getattr(d, 'w', np.ones(len(d.x)) * d.dx) for d in self.dvrs]
        self.grid_weights = self._build_grid_weights()
        self.sqrt_grid_weights = np.sqrt(self.grid_weights)
        # 兼容旧代码，将属性也暴露为 domain
        if domains is not None:
            self.domain = domains

        return self

    def _build_grid_weights(self):
        weights = np.asarray(self.w[0], dtype=float)
        for w in self.w[1:]:
            weights = np.multiply.outer(weights, np.asarray(w, dtype=float))
        return weights.reshape(self.nx)

    def _weight_broadcast_shape(self, psi):
        return (*self.nx, *([1] * (psi.ndim - self.ndim)))

    def to_quadrature_normalized(self, psi_values):
        """Convert raw grid values to orthonormal DVR coefficients.

        In the quadrature-normalized representation used by propagation,
        ``sum(abs(coeffs)**2)`` is the nuclear/electronic norm.  If a
        wavefunction is instead stored as raw values on the grid, multiply by
        ``sqrt(grid_weights)`` before propagation.
        """
        psi_values = np.asarray(psi_values)
        return psi_values * self.sqrt_grid_weights.reshape(self._weight_broadcast_shape(psi_values))

    def from_quadrature_normalized(self, coeffs):
        """Convert orthonormal DVR coefficients back to raw grid values."""
        coeffs = np.asarray(coeffs)
        return coeffs / self.sqrt_grid_weights.reshape(self._weight_broadcast_shape(coeffs))

    def norm(self, psi):
        """Return the norm of quadrature-normalized DVR coefficients."""
        return float(np.sqrt(np.sum(np.abs(psi) ** 2)))

    def _as_state_overlap_matrix(self, ov, nstates):
        ov = np.asarray(ov, dtype=complex)
        if ov.ndim == 0:
            ov = ov.reshape(1, 1)
        if ov.shape != (nstates, nstates):
            raise ValueError(f"State overlap shape {ov.shape} != {(nstates, nstates)}.")
        return ov

    def _grid_indices(self):
        return list(np.ndindex(*self.nx))

    def _build_full_overlap_matrix(self, grid_mc_objects, nstates, overlap_fn=overlap):
        nx = self.nx
        flat_mc = grid_mc_objects.flatten()
        ngrid = len(flat_mc)
        S = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
        state_eye = np.eye(nstates, dtype=complex)

        for a in range(ngrid):
            S[a, :, a, :] = state_eye
            for b in range(a + 1, ngrid):
                ov_ab = self._as_state_overlap_matrix(overlap_fn(flat_mc[a], flat_mc[b]), nstates)
                S[a, :, b, :] = ov_ab
                S[b, :, a, :] = ov_ab.conj().T

        return S.reshape((*nx, nstates, *nx, nstates))

    @staticmethod
    def _polar_unitary(mat):
        """Return the unitary polar factor of a square overlap matrix."""
        u, _, vh = np.linalg.svd(np.asarray(mat, dtype=complex), full_matrices=False)
        return u @ vh

    def _compute_overlap_links(
        self,
        grid_mc_objects,
        nstates,
        overlap_fn=overlap,
        unitarize=False,
    ):
        indices = self._grid_indices()
        links = {}

        for idx in indices:
            for axis in range(self.ndim):
                if idx[axis] + 1 >= self.nx[axis]:
                    continue
                nxt = list(idx)
                nxt[axis] += 1
                nxt = tuple(nxt)
                link = self._as_state_overlap_matrix(
                    overlap_fn(grid_mc_objects[idx], grid_mc_objects[nxt]),
                    nstates,
                )
                if unitarize:
                    link = self._polar_unitary(link)
                links[(axis, idx)] = link

        return links

    def _pack_overlap_links(self, links):
        """Convert an overlap-link dictionary to arrays suitable for np.savez."""
        items = sorted(links.items(), key=lambda item: (item[0][0], item[0][1]))
        if not items:
            return (
                np.empty(0, dtype=int),
                np.empty((0, self.ndim), dtype=int),
                np.empty((0, self.nstates, self.nstates), dtype=complex),
            )
        axes = np.asarray([axis for (axis, _), _ in items], dtype=int)
        indices = np.asarray([idx for (_, idx), _ in items], dtype=int)
        data = np.asarray([mat for _, mat in items], dtype=complex)
        return axes, indices, data

    def _unpack_overlap_links(self, axes, indices, data):
        """Rebuild an overlap-link dictionary from packed arrays."""
        return {
            (int(axis), tuple(int(i) for i in idx)): np.asarray(mat, dtype=complex)
            for axis, idx, mat in zip(axes, indices, data)
        }

    def _linked_overlap_between(self, bra_idx, ket_idx, links, nstates):
        current = list(bra_idx)
        mat = np.eye(nstates, dtype=complex)

        for axis in range(self.ndim):
            while current[axis] < ket_idx[axis]:
                src = tuple(current)
                mat = mat @ links[(axis, src)]
                current[axis] += 1
            while current[axis] > ket_idx[axis]:
                current[axis] -= 1
                src = tuple(current)
                mat = mat @ links[(axis, src)].conj().T

        return mat

    def _build_linked_overlap_from_links(self, links, nstates):
        indices = self._grid_indices()
        ngrid = len(indices)
        flat_index = {idx: i for i, idx in enumerate(indices)}
        S = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
        state_eye = np.eye(nstates, dtype=complex)

        for i, bra_idx in enumerate(indices):
            S[i, :, i, :] = state_eye
            for ket_idx in indices[i + 1:]:
                j = flat_index[ket_idx]
                ov_ij = self._linked_overlap_between(bra_idx, ket_idx, links, nstates)
                S[i, :, j, :] = ov_ij
                S[j, :, i, :] = ov_ij.conj().T

        return S.reshape((*self.nx, nstates, *self.nx, nstates))

    def _build_linked_overlap_matrix(self, grid_mc_objects, nstates, overlap_fn=overlap):
        links = self._compute_overlap_links(grid_mc_objects, nstates, overlap_fn=overlap_fn)
        return self._build_linked_overlap_from_links(links, nstates)

    def _electronic_structure_tasks(
        self,
        basis,
        ncas,
        nelecas,
        nstates,
        electronic_method="casci",
        electronic_options=None,
    ):
        if electronic_options is None:
            electronic_options = {}
        atom_symbols = tuple(self.atom_symbols())
        tasks = []
        for idx in self._grid_indices():
            xyz = self.internal_to_xyz(*(self.x[axis][idx[axis]] for axis in range(self.ndim)))
            tasks.append((
                idx,
                np.asarray(xyz, dtype=float),
                atom_symbols,
                basis,
                self.charge,
                self.spin,
                self.unit,
                ncas,
                nelecas,
                nstates,
                electronic_method,
                dict(electronic_options),
            ))
        return tasks

    def _run_electronic_structure_scan(
        self,
        tasks,
        nstates,
        n_workers=1,
        worker_threads=1,
    ):
        total = len(tasks)
        grid_mc_objects = np.empty(self.nx, dtype=object)
        apes = np.zeros((*self.nx, nstates))

        if n_workers is None:
            n_workers = 1
        n_workers = int(n_workers)
        if n_workers < 1:
            raise ValueError("n_workers must be >= 1.")

        report_every = max(1, total // 10)
        if n_workers == 1:
            _set_worker_thread_limits(worker_threads)
            for count, task in enumerate(tasks, start=1):
                idx, energies, mc = _triatomic_scan_point_worker(task)
                grid_mc_objects[idx] = mc
                apes[idx] = energies
                if count % report_every == 0 or count == total:
                    print(f"  ... {count}/{total} points computed")
            return apes, grid_mc_objects

        with ProcessPoolExecutor(
            max_workers=n_workers,
            initializer=_set_worker_thread_limits,
            initargs=(worker_threads,),
        ) as executor:
            futures = [executor.submit(_triatomic_scan_point_worker, task) for task in tasks]
            for count, future in enumerate(as_completed(futures), start=1):
                idx, energies, mc = future.result()
                grid_mc_objects[idx] = mc
                apes[idx] = energies
                if count % report_every == 0 or count == total:
                    print(f"  ... {count}/{total} points computed")

        return apes, grid_mc_objects

    def scan_pes(
        self,
        basis="631g*",
        nstates=None,
        verbose=0,
        ncas=6,
        nelecas=6,
        overlap_method="linked",
        unitarize_overlap_links=False,
        n_workers=1,
        worker_threads=1,
        electronic_method="casci",
        scf_tol=1.0e-9,
        max_cycle=120,
        damping=0.0,
    ):
        """
        Scan the adiabatic PES over the DVR grid and compute the non-adiabatic
        overlap matrix. Results are always saved to disk as three files:
            apes.npz            -- adiabatic potential energy surfaces
            overlap_matrix.npz  -- LDR electronic overlap matrix, unless
                                   overlap_method='link-only'
            overlap_links.npz   -- nearest-neighbor LDR links, for linked
                                   and link-only overlap methods
            grid_mc_objects.npz -- raw CASCI objects for every grid point

        If externally provided (self.apes and self.overlap_matrix or
        self.overlap_links are already set), the scan is skipped entirely.

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
        overlap_method : {'linked', 'full', 'link-only'}
            How to build the LDR electronic overlap matrix.  ``'linked'`` is
            the default linked-product approximation from nearest-neighbor
            overlaps.  ``'full'`` computes every pair directly.
            ``'link-only'`` stores only nearest-neighbor links and avoids
            constructing the full overlap matrix.
        unitarize_overlap_links : bool, optional
            If True, replace each nearest-neighbor overlap by its unitary polar
            factor before using it in linked-product propagation.
        n_workers : int, optional
            Number of worker processes for the independent electronic
            structure jobs.  Use ``1`` for serial execution.
        worker_threads : int or None, optional
            BLAS/OpenMP thread count set inside each worker process.  The
            default ``1`` avoids oversubscription when many workers are used.
            Set to ``None`` to leave the environment untouched.
        electronic_method : {'casci', 'am1/meci'}, optional
            Electronic-structure backend used at each grid point.  ``'casci'``
            keeps the original native RHF/CASCI path.  ``'am1/meci'`` uses the
            native semiempirical AM1 reference followed by full MECI in the
            frontier active space specified by ``ncas``.
        scf_tol, max_cycle, damping
            AM1 SCF controls used when ``electronic_method='am1/meci'``.

        Returns
        -------
        apes : ndarray, shape (*nx, nstates)
            Adiabatic potential energy surfaces on the DVR grid.
        overlap_matrix_or_links : ndarray or dict
            Full pairwise overlap matrix for ``'full'`` and ``'linked'``;
            nearest-neighbor link dictionary for ``'link-only'``.
        """
        if self.x is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        if not hasattr(self, "internal_to_xyz"):
            raise NotImplementedError(
                "Method 'internal_to_xyz(r1, r2, theta)' must be implemented."
            )

        if nstates is None:
            nstates = self.nstates
        electronic_method = _normalize_triatomic_electronic_method(electronic_method)
        if electronic_method not in ("casci", "am1-meci"):
            raise ValueError("electronic_method must be 'casci' or 'am1/meci'.")
        nx = self.nx  # [N_r1, N_r2, N_theta]
        overlap_method = overlap_method.lower().replace("_", "-")
        if overlap_method == "links":
            overlap_method = "link-only"
        if overlap_method not in ("linked", "full", "link-only"):
            raise ValueError(
                "overlap_method must be 'linked', 'full', or 'link-only'."
            )
        meta = dict(
            nx=nx,
            nstates=nstates,
            basis=basis,
            ncas=ncas,
            nelecas=nelecas,
            electronic_method=electronic_method,
            overlap_method=overlap_method,
            unitarize_overlap_links=unitarize_overlap_links,
            n_workers=n_workers,
            worker_threads=worker_threads,
            scf_tol=scf_tol,
            max_cycle=max_cycle,
            damping=damping,
        )

        # --- 0. Skip if external inputs are already provided -------------------
        if self.apes is not None and (
            self.overlap_matrix is not None or self.overlap_links is not None
        ):
            print("[scan_pes] External PES and LDR overlaps already set. Skipping.")
            overlap_data = (
                self.overlap_matrix
                if self.overlap_matrix is not None
                else self.overlap_links
            )
            return self.apes, overlap_data

        # --- 1. Scan: compute electronic states at every grid point ------------
        print(
            f"[scan_pes] Scanning PES "
            f"(method={electronic_method}, basis={basis}, ncas={ncas}, "
            f"nelecas={nelecas}, n_workers={n_workers}) ..."
        )
        total = np.prod(nx)
        electronic_options = dict(
            scf_tol=scf_tol,
            max_cycle=max_cycle,
            damping=damping,
            verbose=verbose,
        )
        tasks = self._electronic_structure_tasks(
            basis,
            ncas,
            nelecas,
            nstates,
            electronic_method=electronic_method,
            electronic_options=electronic_options,
        )
        if len(tasks) != total:
            raise RuntimeError("Internal error while preparing electronic-structure tasks.")
        self.apes, grid_mc_objects = self._run_electronic_structure_scan(
            tasks,
            nstates,
            n_workers=n_workers,
            worker_threads=worker_threads,
        )

        np.savez("grid_mc_objects.npz", data=grid_mc_objects, meta=np.array(meta))
        np.savez("apes.npz", data=self.apes, meta=np.array(meta))
        print("[scan_pes] Saved grid_mc_objects.npz and apes.npz")

        # --- 2. Compute LDR overlap matrix -------------------------------------
        if overlap_method == "linked":
            print("[scan_pes] Computing overlap matrix with linked-product approximation ...")
            self.overlap_links = self._compute_overlap_links(
                grid_mc_objects,
                nstates,
                overlap_fn=_electronic_state_overlap,
                unitarize=unitarize_overlap_links,
            )
            axes, link_indices, link_data = self._pack_overlap_links(self.overlap_links)
            np.savez(
                "overlap_links.npz",
                axes=axes,
                indices=link_indices,
                data=link_data,
                meta=np.array(meta),
            )
            print("[scan_pes] Saved overlap_links.npz")
            self.overlap_matrix = self._build_linked_overlap_from_links(
                self.overlap_links,
                nstates,
            )
        elif overlap_method == "link-only":
            print("[scan_pes] Computing nearest-neighbor overlap links only ...")
            self.overlap_links = self._compute_overlap_links(
                grid_mc_objects,
                nstates,
                overlap_fn=_electronic_state_overlap,
                unitarize=unitarize_overlap_links,
            )
            self.overlap_matrix = None
            axes, link_indices, link_data = self._pack_overlap_links(self.overlap_links)
            np.savez(
                "overlap_links.npz",
                axes=axes,
                indices=link_indices,
                data=link_data,
                meta=np.array(meta),
            )
            print("[scan_pes] Saved overlap_links.npz")
            return self.apes, self.overlap_links
        else:
            print("[scan_pes] Computing full pairwise overlap matrix ...")
            self.overlap_matrix = self._build_full_overlap_matrix(
                grid_mc_objects,
                nstates,
                overlap_fn=_electronic_state_overlap,
            )

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
            if self._rotation_enabled():
                prob = np.sum(np.abs(psi) ** 2, axis=tuple(range(self.ndim + 1)))
            else:
                prob = np.sum(np.abs(psi) ** 2, axis=tuple(range(self.ndim)))
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
    psi0 = mol.to_quadrature_normalized(psi0)
    print('The norm of psi0', mol.norm(psi0))
    psi0_new = np.einsum('ijlk,ijl->ijlk', mol.overlap_matrix[:, :, :, :, 1, 1, 1, 1], psi0[:, :, :, 1])



    # --- 5. Propagate ---
    dt = 0.1 / au2fs
    result = mol.run(psi0=psi0, dt=dt, nt=100, nout=1)
    print(f"Done. Snapshots saved: {len(result['psilist'])}")
    p=mol.get_population(result, plot=True)
