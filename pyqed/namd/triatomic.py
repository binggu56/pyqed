import string
import functools
import itertools
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from functools import reduce

import numpy as np
import scipy.linalg
import scipy.sparse as sp
from periodictable import elements
import jax
import jax.numpy as jnp

try:
    from numba import njit
except Exception:  # pragma: no cover - optional acceleration dependency
    njit = None

from pyqed.units import amu2au, au2fs, au2wavenumber
from pyqed.dvr.dvr_1d import FEDVR, LegendreDVR, PODVR, SineDVR
from pyqed import interval, au2angstrom
from pyqed.phys import gwp
from pyqed.qchem.mol import Molecule
from pyqed.qchem.hf import RHF, ROHF
from pyqed.qchem.mcscf.casci import CASCI, overlap
from pyqed.namd.keo import (
    EPS,
    calculate_exact_keo as calculate_rovibrational_keo,
    Gmat,
    pseudo,
    build_J_matrices,
    hess_log_abs_det_gmat,
    inv,
    jac_Gmat_vib,
    jac_log_abs_det_gmat,
    kron,
)

warnings.filterwarnings("ignore", message="AM1 model is under testing")


if njit is not None:
    @njit(cache=True)
    def _compiled_rovibronic_block_matvec(
        vec,
        edge_bra,
        edge_ket,
        rot_blocks,
        overlap_blocks,
        ng,
        nrot,
        nstates,
    ):
        out = np.zeros(vec.shape, dtype=np.complex128)
        nedges = edge_bra.shape[0]
        for edge in range(nedges):
            bra = edge_bra[edge]
            ket = edge_ket[edge]
            for r in range(nrot):
                for a in range(nstates):
                    acc = 0.0 + 0.0j
                    for s in range(nrot):
                        rot_coeff = rot_blocks[edge, r, s]
                        if rot_coeff != 0.0:
                            base = (ket * nrot + s) * nstates
                            for b in range(nstates):
                                acc += (
                                    rot_coeff
                                    * vec[base + b]
                                    * overlap_blocks[edge, a, b]
                                )
                    out[(bra * nrot + r) * nstates + a] += acc
        return out
else:
    _compiled_rovibronic_block_matvec = None


def _normalize_triatomic_electronic_method(method):
    method = str(method).lower().replace("_", "-")
    aliases = {
        "cas": "casci",
        "rhf-casci": "casci",
        "rohf-casci": "rohf-casci",
        "rohf/casci": "rohf-casci",
        "am1": "am1-meci",
        "meci": "am1-meci",
        "am1/meci": "am1-meci",
        "uam1": "uam1-meci",
        "uam1/meci": "uam1-meci",
        "uhf-am1": "uam1-meci",
        "uhf-am1/meci": "uam1-meci",
    }
    return aliases.get(method, method)


def _normalize_rovibronic_kinetic_method(method):
    if method is None or method is False:
        return None
    if method is True:
        return "compiled"
    key = str(method).strip().lower().replace("_", "-")
    aliases = {
        "compiled": "compiled",
        "compile": "compiled",
        "numba": "compiled",
        "jit": "compiled",
        "matrix-free": "compiled",
        "matrixfree": "compiled",
        "sparse": "sparse",
        "bsr": "sparse",
        "sparse-bsr": "sparse",
        "python": "python",
        "py": "python",
        "fused": "python",
        "none": None,
        "false": None,
        "off": None,
    }
    if key not in aliases:
        raise ValueError(
            "rovibronic_kinetic must be one of None, 'compiled', 'sparse', or 'python'."
        )
    return aliases[key]


def _normalize_kinetic_action(method):
    if method is None or method is False:
        return "dense"
    if method is True:
        return "matrix-free"
    key = str(method).strip().lower().replace("_", "-")
    aliases = {
        "dense": "dense",
        "flat": "dense",
        "matrix-free": "matrix-free",
        "matrixfree": "matrix-free",
        "linear-operator": "matrix-free",
        "linearoperator": "matrix-free",
        "operator": "matrix-free",
        "none": "dense",
        "false": "dense",
        "off": "dense",
    }
    if key not in aliases:
        raise ValueError("kinetic_action must be one of 'dense' or 'matrix-free'.")
    return aliases[key]


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


def _run_native_rhf_with_retries(pmol, electronic_options):
    init_guess = str(electronic_options.get("init_guess", "hcore"))
    scf_tol = float(electronic_options.get("scf_tol", 1.0e-9))
    conv_tol_dm = float(electronic_options.get("conv_tol_dm", 1.0e-6))
    max_cycle = int(electronic_options.get("max_cycle", 120))
    verbose = int(electronic_options.get("verbose", 0))
    diis_start_cycle = int(electronic_options.get("diis_start_cycle", 2))
    diis_space = int(electronic_options.get("diis_space", 8))
    base_damping = float(electronic_options.get("damping", 0.0))
    base_level_shift = float(electronic_options.get("level_shift", 0.0))
    base_diis = bool(electronic_options.get("diis", True))
    retry_ladder = bool(electronic_options.get("rhf_retry_ladder", True))
    select_lowest = bool(electronic_options.get("rhf_retry_select_lowest", False))

    attempts = [
        dict(
            init_guess=init_guess,
            damping=base_damping,
            level_shift=base_level_shift,
            diis=base_diis,
            conv_tol_dm=conv_tol_dm,
        )
    ]
    if retry_ladder:
        ladder = [
            (init_guess, 0.05, 0.05, True, conv_tol_dm),
            (init_guess, 0.10, 0.10, True, conv_tol_dm),
            (init_guess, 0.25, 0.2, True, conv_tol_dm),
            (init_guess, 0.50, 0.5, True, max(conv_tol_dm, 5.0e-6)),
            ("minao", 0.30, 0.3, True, max(conv_tol_dm, 5.0e-6)),
            ("minao", 0.50, 0.7, False, max(conv_tol_dm, 1.0e-5)),
            ("atom", 0.50, 0.7, False, max(conv_tol_dm, 1.0e-5)),
            ("hcore", 0.50, 0.7, False, max(conv_tol_dm, 1.0e-5)),
        ]
        seen = {
            (
                attempts[0]["init_guess"],
                attempts[0]["damping"],
                attempts[0]["level_shift"],
                attempts[0]["diis"],
                attempts[0]["conv_tol_dm"],
            )
        }
        for guess, damping, level_shift, diis, dm_tol in ladder:
            step = (guess, damping, level_shift, diis, dm_tol)
            if step in seen:
                continue
            seen.add(step)
            attempts.append(
                dict(
                    init_guess=guess,
                    damping=damping,
                    level_shift=level_shift,
                    diis=diis,
                    conv_tol_dm=dm_tol,
                )
            )

    errors = []
    successes = []
    for attempt_id, attempt in enumerate(attempts, start=1):
        try:
            mf = RHF(
                pmol,
                init_guess=attempt["init_guess"],
                verbose=verbose,
            ).run(
                tol=scf_tol,
                conv_tol_dm=attempt["conv_tol_dm"],
                max_cycle=max_cycle,
                damping=attempt["damping"],
                level_shift=attempt["level_shift"],
                diis=attempt["diis"],
                diis_start_cycle=diis_start_cycle,
                diis_space=diis_space,
            )
            mf.scf_attempt = attempt
            mf.scf_attempt_id = attempt_id
            if not select_lowest:
                return mf
            successes.append(mf)
        except KeyboardInterrupt:
            raise
        except BaseException as exc:
            errors.append((attempt, type(exc).__name__, str(exc)))

    if successes:
        return min(successes, key=lambda mf: float(mf.e_tot))

    details = "; ".join(
        f"{attempt['init_guess']}/damp={attempt['damping']}/shift={attempt['level_shift']}"
        f"/diis={attempt['diis']} -> {name}: {msg}"
        for attempt, name, msg in errors
    )
    raise RuntimeError(f"Native RHF failed after {len(attempts)} attempts: {details}")


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
        mf = _run_native_rhf_with_retries(pmol, electronic_options)
        mc = CASCI(mf, ncas=ncas, nelecas=nelecas)
        mc.run(nstates=nstates)
        output_nstates = int(electronic_options.get("output_nstates", nstates))
        energies = _finalize_casci_roots(mc, output_nstates, spin, electronic_options)
        return idx, energies, mc

    if electronic_method == "rohf-casci":
        pmol.build()
        mf = ROHF(pmol).run(
            conv_tol=float(electronic_options.get("scf_tol", 1.0e-9)),
            conv_tol_dm=float(electronic_options.get("conv_tol_dm", 1.0e-6)),
            max_cycle=int(electronic_options.get("max_cycle", 120)),
            verbose=int(electronic_options.get("verbose", 0)),
            damping=float(electronic_options.get("damping", 0.25)),
            diis=bool(electronic_options.get("diis", True)),
            diis_start_cycle=int(electronic_options.get("diis_start_cycle", 2)),
            diis_space=int(electronic_options.get("diis_space", 8)),
        )
        active_nelecas = ncas if nelecas is None else nelecas
        mc = CASCI(mf, ncas=ncas, nelecas=active_nelecas, spin=spin)
        mc.run(nstates=nstates, method=electronic_options.get("casci_method", "direct_ci"))
        output_nstates = int(electronic_options.get("output_nstates", nstates))
        energies = _finalize_casci_roots(mc, output_nstates, spin, electronic_options)
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
        meci_kwargs = {}
        for key in ("spin_penalty", "target_spin", "target_s2"):
            if electronic_options.get(key) is not None:
                meci_kwargs[key] = electronic_options[key]
        mc = mf.MECI(nstates=nstates, ncas=ncas, **meci_kwargs).run()
        energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))[:nstates]
        return idx, energies, mc

    if electronic_method == "uam1-meci":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from pyqed.qchem.semiempirical.am1 import UAM1

        mf = UAM1(pmol).run(
            conv_tol=float(electronic_options.get("scf_tol", 1.0e-9)),
            max_cycle=int(electronic_options.get("max_cycle", 160)),
            verbose=int(electronic_options.get("verbose", 0)),
            damping=float(electronic_options.get("damping", 0.25)),
        )
        meci_kwargs = {}
        for key in ("spin_penalty", "target_spin", "target_s2"):
            if electronic_options.get(key) is not None:
                meci_kwargs[key] = electronic_options[key]
        mc = mf.MECI(nstates=nstates, ncas=ncas, **meci_kwargs).run()
        energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))[:nstates]
        return idx, energies, mc

    raise ValueError(
        "electronic_method must be 'casci', 'rohf/casci', 'am1/meci', or 'uam1/meci' "
        f"(got {electronic_method!r})."
    )


def _electronic_state_overlap(left, right):
    """Return an electronic-state overlap matrix for LDR scans."""
    if hasattr(left, "wavefunction_overlap"):
        return left.wavefunction_overlap(right)
    return overlap(left, right)


def _finalize_casci_roots(mc, nstates, spin, electronic_options):
    """Trim or spin-filter CASCI roots in place and return selected energies."""
    nstates = int(nstates)
    spin_filter = electronic_options.get("spin_filter", None)
    if spin_filter is None:
        spin_filter = "none"
    spin_filter = str(spin_filter).lower().replace("_", "-")
    all_energies = np.atleast_1d(np.asarray(mc.e_tot, dtype=float))

    if spin_filter in ("none", "off", "false", "0"):
        selected = np.arange(min(nstates, len(all_energies)))
        all_s2 = None
    elif spin_filter in ("target", "s2", "doublet"):
        target_s2 = electronic_options.get("spin_filter_target_s2", None)
        if target_s2 is None:
            target_s2 = electronic_options.get("target_s2", None)
        if target_s2 is None:
            target_spin = electronic_options.get("target_spin", None)
            if target_spin is None:
                target_spin = abs(float(spin)) / 2.0
            target_s2 = float(target_spin) * (float(target_spin) + 1.0)
        target_s2 = float(target_s2)
        tol = float(electronic_options.get("spin_filter_tol", electronic_options.get("spin_tol", 1.0e-3)))
        all_s2 = np.array([float(mc.spin_square(root)) for root in range(len(all_energies))])
        candidates = np.flatnonzero(np.abs(all_s2 - target_s2) <= tol)
        if len(candidates) < nstates:
            raise RuntimeError(
                f"Only {len(candidates)} CASCI roots match <S^2>={target_s2:.6f} "
                f"+/- {tol:.3g}; available S2={np.array2string(all_s2, precision=6)}"
            )
        selected = candidates[np.argsort(all_energies[candidates])[:nstates]]
        selected = selected[np.argsort(all_energies[selected])]
    else:
        raise ValueError(f"Unknown spin_filter={spin_filter!r}.")

    mc.e_tot = all_energies[selected]
    mc.ci = [mc.ci[int(root)] for root in selected]
    mc.nstates = len(selected)
    mc.selected_roots = np.asarray(selected, dtype=int)
    if all_s2 is not None:
        mc.selected_s2 = all_s2[selected]
    return np.asarray(mc.e_tot, dtype=float)


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
                 driver=None,
                 J: int = 0,
                 Jz: int | None = None):  # 新增 dvr_type 参数

        # 调用父类初始化
        if driver is not None and basis == 'sto-3g':
            driver_ref = getattr(driver, "template", driver)
            driver_mol = getattr(driver_ref, "mol", None)
            basis = getattr(driver_mol, "basis", basis)
        super().__init__(atom=atom, basis=basis, charge=charge, spin=spin, unit=unit)

        self.driver = driver
        self.nstates = nstates
        self.dvr_type = dvr_type  # 修复: 添加 dvr_type 属性
        self.overlap_matrix=None
        self.overlap_links = None
        self.overlap_path_average = False
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

    def build_factorized_rovibrational_keo(self, J=None, *, verbose=True):
        """Return a matrix-free factorized rovibrational KEO LinearOperator.

        The operator is equivalent to ``calculate_exact_keo(..., mode='all')``
        but is applied as tensor contractions:

        ``D_i^dagger G_ij(q) D_j`` for vibration,
        ``G_ab(q) J_a J_b`` for rotation, and
        ``0.5 * (D_i^dagger G_ai(q) + G_ai(q) D_i) J_a`` for Coriolis terms.

        This avoids materializing the dense rovibrational ``T_all`` matrix.
        It is a KEO operator on the nuclear/rotational space only; linked-LDR
        electronic transport is handled separately by the LDR kinetic action.
        """
        from scipy.sparse.linalg import LinearOperator

        if self.dvrs is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        J = self.J if J is None else int(J)
        if J < 0:
            raise ValueError("J must be non-negative.")

        n_dim = self.ndim
        grids = [d.x for d in self.dvrs]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        q_batch = jnp.stack([m.flatten() for m in mesh], axis=1)
        ng = int(np.prod(self.nx))
        Jz = self._validate_Jz(self.Jz, J)
        nrot = self._rotational_dimension() if J > 0 else 1

        if verbose:
            print(
                "[KEO-factorized] Building metric tensors for "
                f"{ng} grid points, J={J}, Jz={Jz}, nrot={nrot}"
            )
        batch_Gmat_fn = jax.vmap(Gmat, in_axes=(0, None, None))
        G_raw = np.asarray(
            batch_Gmat_fn(q_batch, np.asarray(self.mass, dtype=float), self._internal_to_xyz_jax)
        )
        G_all = G_raw.reshape(*self.nx, *G_raw.shape[1:])
        batch_pseudo_fn = jax.vmap(pseudo, in_axes=(0, None, None))
        pseudo_grid = np.asarray(
            batch_pseudo_fn(q_batch, np.asarray(self.mass, dtype=float), self._internal_to_xyz_jax)
        ).reshape(*self.nx)

        D = [np.asarray(d.momentum(), dtype=complex) for d in self.dvrs]
        Ddag = [op.conj().T for op in D]

        if J > 0:
            J_ops_dict = build_J_matrices(J, M=Jz)
            J_ops = [
                np.asarray(J_ops_dict["jx"], dtype=complex),
                np.asarray(J_ops_dict["jy"], dtype=complex),
                np.asarray(J_ops_dict["jz"], dtype=complex),
            ]
        else:
            J_ops = []

        def apply_axis(arr, op, axis):
            moved = np.moveaxis(arr, axis, 0)
            old_shape = moved.shape
            flat = moved.reshape(old_shape[0], -1)
            applied = op @ flat
            applied = np.asarray(applied).reshape((op.shape[0],) + old_shape[1:])
            return np.moveaxis(applied, 0, axis)

        def apply_rot(arr, op):
            return np.einsum("...s,rs->...r", arr, op, optimize=True)

        def matvec(vec):
            psi = np.asarray(vec, dtype=complex).reshape(*self.nx, nrot)
            out = np.zeros_like(psi, dtype=complex)

            G_vib = G_all[..., :n_dim, :n_dim]
            for i in range(n_dim):
                for j in range(n_dim):
                    dpsi = apply_axis(psi, D[j], j)
                    weighted = G_vib[..., i, j, None] * dpsi
                    out += 0.5 * apply_axis(weighted, Ddag[i], i)

            out += pseudo_grid[..., None] * psi

            if J > 0:
                G_rot = G_all[..., n_dim:n_dim + 3, n_dim:n_dim + 3]
                for a in range(3):
                    for b in range(3):
                        J_ab = J_ops[a] @ J_ops[b]
                        out += 0.5 * G_rot[..., a, b, None] * apply_rot(psi, J_ab)

                G_cor = G_all[..., n_dim:n_dim + 3, :n_dim]
                for a in range(3):
                    rot_psi = apply_rot(psi, J_ops[a])
                    for i in range(n_dim):
                        G_ai = G_cor[..., a, i, None]
                        out += 0.5 * apply_axis(G_ai * rot_psi, Ddag[i], i)
                        out += 0.5 * G_ai * apply_axis(rot_psi, D[i], i)

            return out.reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        return LinearOperator(
            shape=(ng * nrot, ng * nrot),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=complex,
        )

    def build_factorized_rovibronic_ldr_action(
        self,
        *,
        threshold=0.0,
        cache_columns=False,
        verbose=True,
    ):
        """Return an exact matrix-free rovibronic LDR kinetic action.

        This combines the factorized rovibrational KEO with dense or linked
        electronic LDR overlaps without building either the dense
        rovibrational KEO matrix or the dense flat LDR kinetic matrix.

        The implementation streams columns of the factorized rovibrational KEO.
        It is intended as a correctness bridge and small-grid validation path;
        a production implementation should fuse the product terms with linked
        electronic transport directly.
        """
        from scipy.sparse.linalg import LinearOperator

        if not self._rotation_enabled():
            raise RuntimeError("Factorized rovibronic LDR action is intended for J > 0.")
        indices = self._grid_indices()
        ng = len(indices)
        nrot = self.nrot
        nstates = self.nstates
        nrovib = ng * nrot
        nflat = nrovib * nstates
        links = getattr(self, "overlap_links", None)
        state_eye = np.eye(nstates, dtype=complex)
        dense_overlap = None
        if self.overlap_matrix is not None:
            dense_overlap = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

        T_fac = self.build_factorized_rovibrational_keo(verbose=verbose)
        column_cache = {} if cache_columns else None
        unit = np.zeros(nrovib, dtype=complex)

        def overlap_block(i, j):
            if dense_overlap is not None:
                return dense_overlap[i, :, j, :]
            if links is not None:
                return self._linked_overlap_block(
                    i, j, indices[i], indices[j], links, nstates
                )
            return state_eye

        def rovib_column(col):
            if column_cache is not None and col in column_cache:
                return column_cache[col]
            unit[:] = 0.0
            unit[col] = 1.0
            values = np.asarray(T_fac @ unit)
            if threshold > 0.0:
                rows = np.flatnonzero(np.abs(values) > threshold)
            else:
                rows = np.flatnonzero(values)
            data = values[rows]
            if column_cache is not None:
                column_cache[col] = (rows, data)
            return rows, data

        def matvec(vec):
            psi = np.asarray(vec, dtype=complex).reshape(ng, nrot, nstates)
            out = np.zeros_like(psi, dtype=complex)
            for col in range(nrovib):
                j, s = divmod(col, nrot)
                ket = psi[j, s]
                if not np.any(ket):
                    continue
                rows, data = rovib_column(col)
                for row, Tij in zip(rows, data):
                    i, r = divmod(int(row), nrot)
                    out[i, r] += Tij * (overlap_block(i, j) @ ket)
            return out.reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        return LinearOperator(
            shape=(nflat, nflat),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=complex,
        )

    def build_fused_factorized_rovibronic_ldr_action(
        self,
        *,
        threshold=0.0,
        sparse=False,
        compiled=False,
        verbose=True,
    ):
        """Return a fused factorized rovibronic LDR kinetic action.

        Unlike ``build_factorized_rovibronic_ldr_action``, this does not
        stream columns of the rovibrational KEO.  It loops directly over the
        vibrational, rotational, and Coriolis product-action couplings and
        applies the local/linked electronic overlap block on each edge.
        """
        from scipy.sparse.linalg import LinearOperator

        if sparse and compiled:
            raise ValueError("sparse=True and compiled=True are mutually exclusive.")
        if compiled and _compiled_rovibronic_block_matvec is None:
            raise RuntimeError("compiled=True requires numba to be installed.")
        if not self._rotation_enabled():
            raise RuntimeError("Fused factorized rovibronic LDR action requires J > 0.")
        if self.dvrs is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")

        indices = self._grid_indices()
        ng = len(indices)
        nrot = self.nrot
        nstates = self.nstates
        block_size = nrot * nstates
        nflat = ng * block_size
        n_dim = self.ndim
        links = getattr(self, "overlap_links", None)
        state_eye = np.eye(nstates, dtype=complex)
        dense_overlap = None
        if self.overlap_matrix is not None:
            dense_overlap = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

        if verbose:
            print(
                "[KEO-factorized-LDR] Building fused metric tensors for "
                f"{ng} grid points, nrot={nrot}, nstates={nstates}"
            )
        grids = [d.x for d in self.dvrs]
        mesh = jnp.meshgrid(*grids, indexing="ij")
        q_batch = jnp.stack([m.flatten() for m in mesh], axis=1)
        batch_Gmat_fn = jax.vmap(Gmat, in_axes=(0, None, None))
        G_raw = np.asarray(
            batch_Gmat_fn(
                q_batch,
                np.asarray(self.mass, dtype=float),
                self._internal_to_xyz_jax,
            )
        )
        G_all = G_raw.reshape(*self.nx, *G_raw.shape[1:])
        batch_pseudo_fn = jax.vmap(pseudo, in_axes=(0, None, None))
        pseudo_grid = np.asarray(
            batch_pseudo_fn(
                q_batch,
                np.asarray(self.mass, dtype=float),
                self._internal_to_xyz_jax,
            )
        ).reshape(*self.nx)

        D = [np.asarray(d.momentum(), dtype=complex) for d in self.dvrs]
        Ddag = [op.conj().T for op in D]
        J_ops_dict = build_J_matrices(self.J, M=self.Jz)
        J_ops = [
            np.asarray(J_ops_dict["jx"], dtype=complex),
            np.asarray(J_ops_dict["jy"], dtype=complex),
            np.asarray(J_ops_dict["jz"], dtype=complex),
        ]

        flat_index = {idx: pos for pos, idx in enumerate(indices)}
        overlap_cache = {}
        axis_ranges = [range(n) for n in self.nx]

        def matrix_entries(mat):
            rows, cols = np.nonzero(np.abs(mat) > threshold)
            data = mat[rows, cols]
            return [(int(r), int(c), complex(v)) for r, c, v in zip(rows, cols, data)]

        rot_identity_entries = [(r, r, 1.0 + 0.0j) for r in range(nrot)]
        rot_entries = {}
        for a in range(3):
            rot_entries[("J", a)] = matrix_entries(J_ops[a])
            for b in range(3):
                rot_entries[("JJ", a, b)] = matrix_entries(J_ops[a] @ J_ops[b])

        def overlap_block(i, j):
            key = (i, j)
            block = overlap_cache.get(key)
            if block is not None:
                return block
            if dense_overlap is not None:
                block = dense_overlap[i, :, j, :]
            elif links is not None:
                block = self._linked_overlap_block(
                    i, j, indices[i], indices[j], links, nstates
                )
            else:
                block = state_eye
            overlap_cache[key] = block
            return block

        def add_edge_term(edge_terms, bra_flat, ket_flat, coeff, entries):
            if threshold > 0.0 and abs(coeff) <= threshold:
                return
            rot_block = edge_terms.get((bra_flat, ket_flat))
            if rot_block is None:
                rot_block = np.zeros((nrot, nrot), dtype=complex)
                edge_terms[(bra_flat, ket_flat)] = rot_block
            for r, s, rot_coeff in entries:
                value = coeff * rot_coeff
                if threshold > 0.0 and abs(value) <= threshold:
                    continue
                rot_block[r, s] += value

        def with_axis(idx, axis, value):
            items = list(idx)
            items[axis] = int(value)
            return tuple(items)

        def with_two_axes(idx, axis_i, value_i, axis_j, value_j):
            items = list(idx)
            items[axis_i] = int(value_i)
            items[axis_j] = int(value_j)
            return tuple(items)

        edge_terms = {}

        for idx in indices:
            flat = flat_index[idx]
            coeff = complex(pseudo_grid[idx])
            add_edge_term(edge_terms, flat, flat, coeff, rot_identity_entries)

        G_vib = G_all[..., :n_dim, :n_dim]
        for i_axis in range(n_dim):
            for j_axis in range(n_dim):
                Di_dag = Ddag[i_axis]
                Dj = D[j_axis]
                if i_axis == j_axis:
                    other_axes = [ax for ax in range(n_dim) if ax != i_axis]
                    other_shape = tuple(self.nx[ax] for ax in other_axes)
                    for other_values in np.ndindex(*other_shape):
                        base = [0] * n_dim
                        for ax, value in zip(other_axes, other_values):
                            base[ax] = int(value)
                        for m in axis_ranges[i_axis]:
                            mid = with_axis(base, i_axis, m)
                            gval = complex(G_vib[mid][i_axis, j_axis])
                            if threshold > 0.0 and abs(gval) <= threshold:
                                continue
                            for p in axis_ranges[i_axis]:
                                left = complex(Di_dag[p, m])
                                if threshold > 0.0 and abs(left) <= threshold:
                                    continue
                                bra_flat = flat_index[with_axis(base, i_axis, p)]
                                for q in axis_ranges[i_axis]:
                                    coeff = 0.5 * left * gval * complex(Dj[m, q])
                                    ket_flat = flat_index[with_axis(base, i_axis, q)]
                                    add_edge_term(
                                        edge_terms,
                                        bra_flat,
                                        ket_flat,
                                        coeff,
                                        rot_identity_entries,
                                    )
                else:
                    other_axes = [ax for ax in range(n_dim) if ax not in (i_axis, j_axis)]
                    other_shape = tuple(self.nx[ax] for ax in other_axes)
                    for other_values in np.ndindex(*other_shape):
                        base = [0] * n_dim
                        for ax, value in zip(other_axes, other_values):
                            base[ax] = int(value)
                        for mi in axis_ranges[i_axis]:
                            for mj in axis_ranges[j_axis]:
                                mid = with_two_axes(base, i_axis, mi, j_axis, mj)
                                gval = complex(G_vib[mid][i_axis, j_axis])
                                if threshold > 0.0 and abs(gval) <= threshold:
                                    continue
                                for p in axis_ranges[i_axis]:
                                    left = complex(Di_dag[p, mi])
                                    if threshold > 0.0 and abs(left) <= threshold:
                                        continue
                                    bra = with_two_axes(base, i_axis, p, j_axis, mj)
                                    bra_flat = flat_index[bra]
                                    for q in axis_ranges[j_axis]:
                                        coeff = 0.5 * left * gval * complex(Dj[mj, q])
                                        ket = with_two_axes(base, i_axis, mi, j_axis, q)
                                        add_edge_term(
                                            edge_terms,
                                            bra_flat,
                                            flat_index[ket],
                                            coeff,
                                            rot_identity_entries,
                                        )

        G_rot = G_all[..., n_dim:n_dim + 3, n_dim:n_dim + 3]
        for idx in indices:
            flat = flat_index[idx]
            for a in range(3):
                for b in range(3):
                    add_edge_term(
                        edge_terms,
                        flat,
                        flat,
                        0.5 * complex(G_rot[idx][a, b]),
                        rot_entries[("JJ", a, b)],
                    )

        G_cor = G_all[..., n_dim:n_dim + 3, :n_dim]
        for a in range(3):
            entries = rot_entries[("J", a)]
            for i_axis in range(n_dim):
                Di = D[i_axis]
                Di_dag = Ddag[i_axis]
                other_axes = [ax for ax in range(n_dim) if ax != i_axis]
                other_shape = tuple(self.nx[ax] for ax in other_axes)
                for other_values in np.ndindex(*other_shape):
                    base = [0] * n_dim
                    for ax, value in zip(other_axes, other_values):
                        base[ax] = int(value)
                    for m in axis_ranges[i_axis]:
                        ket = with_axis(base, i_axis, m)
                        ket_flat = flat_index[ket]
                        gval = complex(G_cor[ket][a, i_axis])
                        if threshold > 0.0 and abs(gval) <= threshold:
                            continue
                        for p in axis_ranges[i_axis]:
                            coeff = 0.5 * complex(Di_dag[p, m]) * gval
                            bra = with_axis(base, i_axis, p)
                            add_edge_term(
                                edge_terms,
                                flat_index[bra],
                                ket_flat,
                                coeff,
                                entries,
                            )
                    for m in axis_ranges[i_axis]:
                        bra = with_axis(base, i_axis, m)
                        bra_flat = flat_index[bra]
                        gval = complex(G_cor[bra][a, i_axis])
                        if threshold > 0.0 and abs(gval) <= threshold:
                            continue
                        for q in axis_ranges[i_axis]:
                            coeff = 0.5 * gval * complex(Di[m, q])
                            ket = with_axis(base, i_axis, q)
                            add_edge_term(
                                edge_terms,
                                bra_flat,
                                flat_index[ket],
                                coeff,
                                entries,
                            )

        edge_items = []
        for (bra_flat, ket_flat), rot_block in edge_terms.items():
            if threshold > 0.0 and np.max(np.abs(rot_block)) <= threshold:
                continue
            edge_items.append(
                (
                    bra_flat,
                    ket_flat,
                    rot_block,
                    overlap_block(bra_flat, ket_flat),
                )
            )
        if verbose:
            print(f"[KEO-factorized-LDR] Cached {len(edge_items)} product edges")
        trace_value = sum(
            np.trace(rot_block) * np.trace(Aij)
            for bra_flat, ket_flat, rot_block, Aij in edge_items
            if bra_flat == ket_flat
        )

        if sparse:
            block_rows = np.empty(len(edge_items), dtype=np.int64)
            block_cols = np.empty(len(edge_items), dtype=np.int64)
            data = np.empty((len(edge_items), block_size, block_size), dtype=complex)
            for item, (bra_flat, ket_flat, rot_block, Aij) in enumerate(edge_items):
                block_rows[item] = bra_flat
                block_cols[item] = ket_flat
                data[item] = np.kron(rot_block, Aij)

            order = np.lexsort((block_cols, block_rows))
            block_rows = block_rows[order]
            block_cols = block_cols[order]
            data = data[order]
            indptr = np.zeros(ng + 1, dtype=np.int64)
            np.add.at(indptr, block_rows + 1, 1)
            indptr = np.cumsum(indptr)
            matrix = sp.bsr_matrix(
                (data, block_cols, indptr),
                shape=(nflat, nflat),
                dtype=complex,
            )
            matrix.sum_duplicates()
            matrix.trace_value = trace_value
            matrix.edge_count = len(edge_items)
            return matrix

        if compiled:
            edge_bra = np.empty(len(edge_items), dtype=np.int64)
            edge_ket = np.empty(len(edge_items), dtype=np.int64)
            rot_blocks = np.empty((len(edge_items), nrot, nrot), dtype=np.complex128)
            overlap_blocks = np.empty(
                (len(edge_items), nstates, nstates),
                dtype=np.complex128,
            )
            for item, (bra_flat, ket_flat, rot_block, Aij) in enumerate(edge_items):
                edge_bra[item] = bra_flat
                edge_ket[item] = ket_flat
                rot_blocks[item] = rot_block
                overlap_blocks[item] = Aij

            def matvec(vec):
                vec = np.ascontiguousarray(np.asarray(vec, dtype=np.complex128).reshape(-1))
                return _compiled_rovibronic_block_matvec(
                    vec,
                    edge_bra,
                    edge_ket,
                    rot_blocks,
                    overlap_blocks,
                    ng,
                    nrot,
                    nstates,
                )

            def matmat(mat):
                mat = np.asarray(mat)
                return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

            operator = LinearOperator(
                shape=(nflat, nflat),
                matvec=matvec,
                rmatvec=matvec,
                matmat=matmat,
                dtype=complex,
            )
            operator.trace_value = trace_value
            operator.edge_count = len(edge_items)
            operator.block_storage_mib = (
                edge_bra.nbytes
                + edge_ket.nbytes
                + rot_blocks.nbytes
                + overlap_blocks.nbytes
            ) / 1024**2
            return operator

        def matvec(vec):
            psi = np.asarray(vec, dtype=complex).reshape(ng, nrot, nstates)
            out = np.zeros_like(psi, dtype=complex)
            for bra_flat, ket_flat, rot_block, Aij in edge_items:
                out[bra_flat] += rot_block @ psi[ket_flat] @ Aij.T
            return out.reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        operator = LinearOperator(
            shape=(nflat, nflat),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=complex,
        )
        operator.trace_value = trace_value
        operator.edge_count = len(edge_items)
        return operator

    def build_sparse_factorized_rovibronic_ldr_matrix(
        self,
        *,
        threshold=0.0,
        verbose=True,
    ):
        """Return the factorized rovibronic LDR kinetic action as sparse CSR.

        The matrix is assembled from cached grid-point blocks of size
        ``nrot * nstates``.  It avoids forming the dense rovibrational KEO but
        lets SciPy use optimized sparse matvecs during Krylov/Chebyshev
        propagation.
        """
        return self.build_fused_factorized_rovibronic_ldr_action(
            threshold=threshold,
            sparse=True,
            verbose=verbose,
        )

    def build_compiled_factorized_rovibronic_ldr_action(
        self,
        *,
        threshold=0.0,
        verbose=True,
    ):
        """Return the factorized rovibronic LDR kinetic action as a compiled matvec.

        The returned ``LinearOperator`` stores edge-wise rotational and
        electronic blocks and applies the product action in a Numba-compiled
        loop.  It is intended for larger ``J`` values where explicit BSR
        storage becomes expensive.
        """
        return self.build_fused_factorized_rovibronic_ldr_action(
            threshold=threshold,
            compiled=True,
            verbose=verbose,
        )

    def buildK(self, J=None, sparse=False):
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
            if sparse:
                raise NotImplementedError(
                    "Sparse rovibrational KEO is not implemented for J > 0."
                )
            return self.build_rovibrational_keo(J=J, verbose=True)

        self.M_end1, self.M_center, self.M_end2 = self.mass[0], self.mass[1], self.mass[2]
        M_Y, M_X1, M_X2 = self.M_center, self.M_end1, self.M_end2
        dvrs = self.dvrs

        def momentum_matrix(dvr):
            if sparse:
                try:
                    return dvr.momentum(sparse=True)
                except TypeError:
                    return sp.csr_matrix(dvr.momentum())
            return dvr.momentum()

        p_r1 = momentum_matrix(dvrs[0])
        p_r2 = momentum_matrix(dvrs[1])
        p_th = momentum_matrix(dvrs[2])
        hbar = 1.0

        r1_grid, r2_grid, th_grid = self.x
        N_r1, N_r2, N_th = self.nx

        if sparse:
            I_r1 = sp.eye(N_r1, format="csr", dtype=complex)
            I_r2 = sp.eye(N_r2, format="csr", dtype=complex)
            I_th = sp.eye(N_th, format="csr", dtype=complex)

            P1 = sp.kron(p_r1, sp.kron(I_r2, I_th, format="csr"), format="csr")
            P2 = sp.kron(I_r1, sp.kron(p_r2, I_th, format="csr"), format="csr")
            P_th_op = sp.kron(I_r1, sp.kron(I_r2, p_th, format="csr"), format="csr")
        else:
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
        v_metric = (
            -(hbar**2 / 8.0) * val_Gthth * (1 + csc_th**2)
            - hbar**2 / 2.0 / M_Y * (np.cos(thv) / (r1v * r2v))
        )
        V_metric = sp.diags(v_metric, format="csr") if sparse else np.diag(v_metric)

        def sandwich(Pl, g, Pr):
            if sparse:
                return Pl.conj().T @ sp.diags(g, format="csr") @ Pr
            return Pl.conj().T @ np.diag(g) @ Pr

        T = 0.5 * (
            sandwich(P1, np.full_like(r1v, val_G11), P1)
            + sandwich(P2, np.full_like(r1v, val_G22), P2)
            + sandwich(P_th_op, val_Gthth, P_th_op)
            + sandwich(P1, val_G12, P2) + sandwich(P2, val_G12, P1)
            + sandwich(P1, val_G1th, P_th_op) + sandwich(P_th_op, val_G1th, P1)
            + sandwich(P2, val_G2th, P_th_op) + sandwich(P_th_op, val_G2th, P2)
        ) + V_metric

        #print(">>> T_total computed.")
        return T.tocsr() if sparse else T

    def buildK_product_terms(self, J=None, sparse=False, symmetrize=False):
        """Return the J=0 analytical kinetic operator as product terms.

        Each term is ``(label, coefficient, A_r1, A_r2, A_theta)`` and
        represents ``coefficient * kron(A_r1, kron(A_r2, A_theta))``.  This is
        the TT/MPO-ready form of the same sandwich KEO used by ``buildK``.
        """
        J = self.J if J is None else int(J)
        if J > 0:
            raise NotImplementedError(
                "Product-term rovibrational KEO is not implemented for J > 0."
            )

        self.M_end1, self.M_center, self.M_end2 = self.mass[0], self.mass[1], self.mass[2]
        M_Y, M_X1, M_X2 = self.M_center, self.M_end1, self.M_end2
        dvrs = self.dvrs

        def momentum_matrix(dvr):
            if sparse:
                try:
                    return dvr.momentum(sparse=True)
                except TypeError:
                    return sp.csr_matrix(dvr.momentum())
            return dvr.momentum()

        def eye(n):
            if sparse:
                return sp.eye(n, format="csr", dtype=complex)
            return np.eye(n, dtype=complex)

        def diag(values):
            values = np.asarray(values, dtype=complex)
            if sparse:
                return sp.diags(values, format="csr")
            return np.diag(values)

        def matmul(*ops):
            out = ops[0]
            for op in ops[1:]:
                out = out @ op
            return out

        def adjoint(op):
            return op.getH() if sp.issparse(op) else op.conj().T

        p_r1 = momentum_matrix(dvrs[0])
        p_r2 = momentum_matrix(dvrs[1])
        p_th = momentum_matrix(dvrs[2])

        r1_grid, r2_grid, th_grid = self.x
        I_r1 = eye(len(r1_grid))
        I_r2 = eye(len(r2_grid))
        I_th = eye(len(th_grid))

        D_r1_inv = diag(1.0 / np.asarray(r1_grid))
        D_r2_inv = diag(1.0 / np.asarray(r2_grid))
        D_r1_inv2 = diag(1.0 / np.asarray(r1_grid) ** 2)
        D_r2_inv2 = diag(1.0 / np.asarray(r2_grid) ** 2)
        D_cos = diag(np.cos(th_grid))
        D_sin = diag(np.sin(th_grid))
        D_csc_metric = diag(1.0 + 1.0 / np.sin(th_grid) ** 2)

        inv_mu1 = 1.0 / M_X1 + 1.0 / M_Y
        inv_mu2 = 1.0 / M_X2 + 1.0 / M_Y
        terms = []

        def add(label, coef, A, B, C):
            terms.append((label, complex(coef), A, B, C))

        add("p1_G11_p1", 0.5 * inv_mu1, matmul(adjoint(p_r1), p_r1), I_r2, I_th)
        add("p2_G22_p2", 0.5 * inv_mu2, I_r1, matmul(adjoint(p_r2), p_r2), I_th)

        add("pth_Gthth_r1_pth", 0.5 * inv_mu1, D_r1_inv2, I_r2, matmul(adjoint(p_th), p_th))
        add("pth_Gthth_r2_pth", 0.5 * inv_mu2, I_r1, D_r2_inv2, matmul(adjoint(p_th), p_th))
        add(
            "pth_Gthth_r12_pth",
            -1.0 / M_Y,
            D_r1_inv,
            D_r2_inv,
            matmul(adjoint(p_th), D_cos, p_th),
        )

        add("p1_G12_p2", 0.5 / M_Y, adjoint(p_r1), p_r2, D_cos)
        add("p2_G12_p1", 0.5 / M_Y, p_r1, adjoint(p_r2), D_cos)

        add(
            "p1_G1th_pth",
            -0.5 / M_Y,
            adjoint(p_r1),
            D_r2_inv,
            matmul(D_sin, p_th),
        )
        add(
            "pth_G1th_p1",
            -0.5 / M_Y,
            p_r1,
            D_r2_inv,
            matmul(adjoint(p_th), D_sin),
        )
        add(
            "p2_G2th_pth",
            -0.5 / M_Y,
            D_r1_inv,
            adjoint(p_r2),
            matmul(D_sin, p_th),
        )
        add(
            "pth_G2th_p2",
            -0.5 / M_Y,
            D_r1_inv,
            p_r2,
            matmul(adjoint(p_th), D_sin),
        )

        add("V_metric_r1", -0.125 * inv_mu1, D_r1_inv2, I_r2, D_csc_metric)
        add("V_metric_r2", -0.125 * inv_mu2, I_r1, D_r2_inv2, D_csc_metric)
        add(
            "V_metric_r12_csc",
            0.25 / M_Y,
            D_r1_inv,
            D_r2_inv,
            matmul(D_cos, D_csc_metric),
        )
        add("V_metric_r12_cos", -0.5 / M_Y, D_r1_inv, D_r2_inv, D_cos)

        if symmetrize:
            hermitian_terms = []

            for label, coef, A, B, C in terms:
                hermitian_terms.append((label, 0.5 * coef, A, B, C))
                hermitian_terms.append(
                    (
                        f"{label}_adjoint",
                        0.5 * coef.conjugate(),
                        adjoint(A),
                        adjoint(B),
                        adjoint(C),
                    )
                )
            terms = hermitian_terms

        return terms

    def buildK_from_product_terms(self, J=None, sparse=False, symmetrize=False):
        """Materialize the product-term kinetic operator for validation."""
        terms = self.buildK_product_terms(J=J, sparse=sparse, symmetrize=symmetrize)

        def kron3(A, B, C):
            if sparse:
                return sp.kron(A, sp.kron(B, C, format="csr"), format="csr")
            return np.kron(A, np.kron(B, C))

        T = None
        for _, coef, A, B, C in terms:
            block = coef * kron3(A, B, C)
            T = block if T is None else T + block
        return T.tocsr() if sparse else T

    def applyK_product_terms(
        self,
        psi,
        J=None,
        sparse=False,
        terms=None,
        symmetrize=False,
    ):
        """Apply the product-term J=0 kinetic operator without assembling it.

        ``psi`` may have shape ``(n_r1, n_r2, n_theta)`` or additional trailing
        dimensions, for example electronic-state amplitudes.  The kinetic
        action is applied only to the first three nuclear axes.
        """
        psi = np.asarray(psi)
        if psi.shape[:3] != tuple(self.nx):
            raise ValueError(f"psi leading shape {psi.shape[:3]} != {tuple(self.nx)}")
        if terms is None:
            terms = self.buildK_product_terms(
                J=J,
                sparse=sparse,
                symmetrize=symmetrize,
            )

        def apply_axis(arr, op, axis):
            moved = np.moveaxis(arr, axis, 0)
            old_shape = moved.shape
            flat = moved.reshape(old_shape[0], -1)
            applied = op @ flat
            applied = np.asarray(applied).reshape((op.shape[0],) + old_shape[1:])
            return np.moveaxis(applied, 0, axis)

        out = np.zeros_like(psi, dtype=np.result_type(psi.dtype, complex))
        for _, coef, A, B, C in terms:
            term = apply_axis(psi, C, 2)
            term = apply_axis(term, B, 1)
            term = apply_axis(term, A, 0)
            out += coef * term
        return out

    def applyK_product_terms_ldr(
        self,
        psi,
        J=None,
        sparse=False,
        terms=None,
        threshold=0.0,
        symmetrize_terms=True,
    ):
        """Apply the LDR kinetic action using analytical product terms.

        This evaluates

        ``sum_jb T[i,j] A[i,a,j,b] psi[j,b]``

        without materializing the dense nuclear kinetic matrix ``T``.  If
        ``overlap_links`` are present, overlap blocks are linked transports
        evaluated on demand and cached.  If ``overlap_matrix`` is present, its
        blocks are used directly.  With neither, the electronic overlap is the
        identity.
        """
        J = self.J if J is None else int(J)
        if J > 0:
            raise NotImplementedError(
                "Product-term LDR kinetic action is not implemented for J > 0."
            )
        psi = np.asarray(psi)
        expected = (*self.nx, self.nstates)
        if psi.shape != expected:
            raise ValueError(f"psi shape {psi.shape} != expected {expected}")
        if terms is None:
            terms = self.buildK_product_terms(
                J=J,
                sparse=sparse,
                symmetrize=symmetrize_terms,
            )

        indices = self._grid_indices()
        flat_index = {idx: i for i, idx in enumerate(indices)}
        nstates = self.nstates
        state_eye = np.eye(nstates, dtype=complex)
        dense_overlap = None
        links = getattr(self, "overlap_links", None)
        if self.overlap_matrix is not None:
            dense_overlap = self.overlap_matrix.reshape(
                len(indices),
                nstates,
                len(indices),
                nstates,
            )
        linked_cache = {}

        def matrix_entries(op):
            if sp.issparse(op):
                coo = op.tocoo()
                rows = coo.row
                cols = coo.col
                data = coo.data
            else:
                arr = np.asarray(op)
                rows, cols = np.nonzero(np.abs(arr) > threshold)
                data = arr[rows, cols]
            if threshold > 0.0:
                keep = np.abs(data) > threshold
                rows = rows[keep]
                cols = cols[keep]
                data = data[keep]
            return rows.astype(int), cols.astype(int), data

        def overlap_block(bra_idx, ket_idx):
            if bra_idx == ket_idx:
                return state_eye
            if dense_overlap is not None:
                return dense_overlap[
                    flat_index[bra_idx],
                    :,
                    flat_index[ket_idx],
                    :,
                ]
            if links is None:
                return state_eye
            key = (bra_idx, ket_idx)
            block = linked_cache.get(key)
            if block is None:
                block = self._linked_overlap_block(
                    flat_index[bra_idx],
                    flat_index[ket_idx],
                    bra_idx,
                    ket_idx,
                    links,
                    nstates,
                )
                linked_cache[key] = block
            return block

        out = np.zeros_like(psi, dtype=np.result_type(psi.dtype, complex))
        for _, coef, A, B, C in terms:
            ai, aj, av = matrix_entries(A)
            bi, bj, bv = matrix_entries(B)
            ci, cj, cv = matrix_entries(C)
            for i1, j1, aij in zip(ai, aj, av):
                for i2, j2, bij in zip(bi, bj, bv):
                    for i3, j3, cij in zip(ci, cj, cv):
                        value = coef * aij * bij * cij
                        if threshold > 0.0 and abs(value) <= threshold:
                            continue
                        bra_idx = (int(i1), int(i2), int(i3))
                        ket_idx = (int(j1), int(j2), int(j3))
                        out[bra_idx] += value * (
                            overlap_block(bra_idx, ket_idx) @ psi[ket_idx]
                        )
        return out

    def build_product_term_ldr_edges(
        self,
        J=None,
        sparse=False,
        terms=None,
        threshold=0.0,
        symmetrize_terms=True,
    ):
        """Precompute edge data for fast product-term LDR matvecs.

        Returns a dictionary with nuclear row/column indices, merged kinetic
        coefficients, and the electronic overlap block for each coupled pair.
        """
        J = self.J if J is None else int(J)
        if J > 0:
            raise NotImplementedError(
                "Product-term LDR edges are not implemented for J > 0."
            )
        if terms is None:
            terms = self.buildK_product_terms(
                J=J,
                sparse=sparse,
                symmetrize=symmetrize_terms,
            )

        n1, n2, n3 = (int(n) for n in self.nx)
        ng = int(np.prod(self.nx))
        nstates = int(self.nstates)

        def matrix_entries(op):
            if sp.issparse(op):
                coo = op.tocoo()
                rows = coo.row.astype(int)
                cols = coo.col.astype(int)
                data = coo.data
            else:
                arr = np.asarray(op)
                rows, cols = np.nonzero(np.abs(arr) > threshold)
                data = arr[rows, cols]
            if threshold > 0.0:
                keep = np.abs(data) > threshold
                rows = rows[keep]
                cols = cols[keep]
                data = data[keep]
            return rows.astype(int), cols.astype(int), np.asarray(data, dtype=complex)

        all_rows = []
        all_cols = []
        all_vals = []
        for _, coef, A, B, C in terms:
            ai, aj, av = matrix_entries(A)
            bi, bj, bv = matrix_entries(B)
            ci, cj, cv = matrix_entries(C)
            if len(av) == 0 or len(bv) == 0 or len(cv) == 0:
                continue

            rows = (
                (ai[:, None, None] * n2 + bi[None, :, None]) * n3
                + ci[None, None, :]
            ).reshape(-1)
            cols = (
                (aj[:, None, None] * n2 + bj[None, :, None]) * n3
                + cj[None, None, :]
            ).reshape(-1)
            vals = (
                coef
                * av[:, None, None]
                * bv[None, :, None]
                * cv[None, None, :]
            ).reshape(-1)
            if threshold > 0.0:
                keep = np.abs(vals) > threshold
                rows = rows[keep]
                cols = cols[keep]
                vals = vals[keep]
            all_rows.append(rows.astype(np.int64, copy=False))
            all_cols.append(cols.astype(np.int64, copy=False))
            all_vals.append(vals.astype(complex, copy=False))

        if not all_rows:
            return {
                "rows": np.array([], dtype=np.int64),
                "cols": np.array([], dtype=np.int64),
                "values": np.array([], dtype=complex),
                "blocks": None,
                "shape": (*self.nx, nstates),
            }

        rows = np.concatenate(all_rows)
        cols = np.concatenate(all_cols)
        vals = np.concatenate(all_vals)
        keys = rows * ng + cols
        order = np.argsort(keys, kind="mergesort")
        keys = keys[order]
        vals = vals[order]
        unique_keys, start = np.unique(keys, return_index=True)
        vals = np.add.reduceat(vals, start)
        rows = (unique_keys // ng).astype(np.int64)
        cols = (unique_keys % ng).astype(np.int64)
        keep = np.abs(vals) > threshold
        rows = rows[keep]
        cols = cols[keep]
        vals = vals[keep]

        dense_overlap = None
        links = getattr(self, "overlap_links", None)
        if self.overlap_matrix is not None:
            dense_overlap = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

        if dense_overlap is not None:
            blocks = np.asarray(dense_overlap[rows, :, cols, :], dtype=complex)
        elif links is not None:
            indices = self._grid_indices()
            blocks = np.empty((len(rows), nstates, nstates), dtype=complex)
            for edge, (row, col) in enumerate(zip(rows, cols)):
                bra_idx = indices[int(row)]
                ket_idx = indices[int(col)]
                blocks[edge] = self._linked_overlap_block(
                    int(row),
                    int(col),
                    bra_idx,
                    ket_idx,
                    links,
                    nstates,
                )
        else:
            blocks = None

        return {
            "rows": rows,
            "cols": cols,
            "values": vals,
            "blocks": blocks,
            "shape": (*self.nx, nstates),
        }

    def apply_product_term_ldr_edges(self, psi, edges):
        """Apply precomputed product-term LDR edge data."""
        psi = np.asarray(psi)
        shape = tuple(edges["shape"])
        if psi.shape == (int(np.prod(shape)),):
            psi = psi.reshape(shape)
        if psi.shape != shape:
            raise ValueError(f"psi shape {psi.shape} != expected {shape}")

        rows = edges["rows"]
        cols = edges["cols"]
        vals = edges["values"]
        blocks = edges["blocks"]
        nstates = shape[-1]
        psi_flat = psi.reshape(-1, nstates)
        out = np.zeros_like(psi_flat, dtype=np.result_type(psi.dtype, complex))
        if len(rows) == 0:
            return out.reshape(shape)
        if blocks is None:
            transported = psi_flat[cols]
        else:
            transported = np.einsum(
                "eab,eb->ea",
                blocks,
                psi_flat[cols],
                optimize=True,
            )
        np.add.at(out, rows, vals[:, None] * transported)
        return out.reshape(shape)

    def build_product_term_ldr_kinetic_operator(
        self,
        J=None,
        sparse=False,
        threshold=0.0,
        symmetrize_terms=True,
    ):
        """Return a LinearOperator for product-term linked/full LDR kinetics."""
        from scipy.sparse.linalg import LinearOperator

        J = self.J if J is None else int(J)
        if J > 0:
            raise NotImplementedError(
                "Product-term LDR kinetic operator is not implemented for J > 0."
            )
        shape = (*self.nx, self.nstates)
        nflat = int(np.prod(shape))
        edges = self.build_product_term_ldr_edges(
            J=J,
            sparse=sparse,
            threshold=threshold,
            symmetrize_terms=symmetrize_terms,
        )

        def matvec(vec):
            return self.apply_product_term_ldr_edges(vec, edges).reshape(-1)

        def matmat(mat):
            mat = np.asarray(mat)
            return np.column_stack([matvec(mat[:, col]) for col in range(mat.shape[1])])

        return LinearOperator(
            shape=(nflat, nflat),
            matvec=matvec,
            rmatvec=matvec,
            matmat=matmat,
            dtype=complex,
        )

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

    def build_ttldr_bundle(self, *, max_rank=None, atol=0.0, prefer_links=True):
        """Return structured TT-LDR operator pieces for this model.

        This is a lightweight adapter for experiments with tensor-train/MPO
        representations of APES, electronic overlaps, and future rotational
        kinetic terms.  It does not alter the current propagation path.
        """
        from pyqed.namd.ttldr import build_bundle

        return build_bundle(
            self,
            max_rank=max_rank,
            atol=atol,
            prefer_links=prefer_links,
        )

    def build_ttldr_action(
        self,
        T_total=None,
        *,
        sparse=False,
        threshold=0.0,
        max_rank=None,
        atol=0.0,
        prefer_links=True,
    ):
        """Return a matrix-free TT-LDR action for dense wavepacket tensors."""
        from pyqed.namd.ttldr import build_action

        return build_action(
            self,
            T=T_total,
            sparse=sparse,
            threshold=threshold,
            max_rank=max_rank,
            atol=atol,
            prefer_links=prefer_links,
        )

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

        if sp.issparse(T_total):
            T_csr = T_total.tocsr()
            linked_cache = {}

            def linked_cached(i, j):
                key = (i, j)
                block = linked_cache.get(key)
                if block is None:
                    block = self._linked_overlap_block(
                        i, j, indices[i], indices[j], links, nstates
                    )
                    linked_cache[key] = block
                return block

            if has_rotation:
                A = None
                if self.overlap_matrix is not None:
                    A = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, self.nrot, nstates)
                    out = np.zeros_like(psi, dtype=dtype)
                    for row in range(T_csr.shape[0]):
                        i, r = divmod(row, self.nrot)
                        start, stop = T_csr.indptr[row], T_csr.indptr[row + 1]
                        for ptr in range(start, stop):
                            Tij = T_csr.data[ptr]
                            if threshold > 0.0 and abs(Tij) <= threshold:
                                continue
                            col = T_csr.indices[ptr]
                            j, s = divmod(col, self.nrot)
                            if A is not None:
                                Aij = A[i, :, j, :]
                                out[i, r] += Tij * (Aij @ psi[j, s])
                            elif links is not None:
                                out[i, r] += Tij * (linked_cached(i, j) @ psi[j, s])
                            else:
                                out[i, r] += Tij * psi[j, s]
                    return out.reshape(-1)

            else:
                A = None
                if self.overlap_matrix is not None:
                    A = self.overlap_matrix.reshape(ng, nstates, ng, nstates)

                rows = np.repeat(np.arange(ng), np.diff(T_csr.indptr))
                cols = T_csr.indices.copy()
                data = T_csr.data.copy()
                if threshold > 0.0:
                    keep = np.abs(data) > threshold
                    rows = rows[keep]
                    cols = cols[keep]
                    data = data[keep]

                if A is not None:
                    edge_blocks = A[rows, :, cols, :]
                elif links is not None:
                    edge_blocks = np.asarray(
                        [linked_cached(int(i), int(j)) for i, j in zip(rows, cols)],
                        dtype=complex,
                    )
                else:
                    edge_blocks = None
                    T_action = sp.csr_matrix((data, (rows, cols)), shape=T_csr.shape)

                def matvec(vec):
                    psi = np.asarray(vec).reshape(ng, nstates)
                    if edge_blocks is None:
                        return (T_action @ psi).reshape(-1)
                    transported = np.einsum(
                        "eab,eb->ea",
                        edge_blocks,
                        psi[cols],
                        optimize=True,
                    )
                    out = np.zeros_like(psi, dtype=dtype)
                    np.add.at(out, rows, data[:, None] * transported)
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

        if has_rotation:
            T_rs = np.asarray(T_total).reshape(ng, self.nrot, ng, self.nrot)
            linked_cache = {}

            def linked_cached(i, j, bra_idx, ket_idx):
                key = (i, j)
                block = linked_cache.get(key)
                if block is None:
                    block = self._linked_overlap_block(
                        i, j, bra_idx, ket_idx, links, nstates
                    )
                    linked_cache[key] = block
                return block

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
                            Aij = linked_cached(i, j, bra_idx, ket_idx)
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
            linked_cache = {}

            def linked_cached(i, j, bra_idx, ket_idx):
                key = (i, j)
                block = linked_cache.get(key)
                if block is None:
                    block = self._linked_overlap_block(
                        i, j, bra_idx, ket_idx, links, nstates
                    )
                    linked_cache[key] = block
                return block

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
                            Aij = linked_cached(i, j, bra_idx, ket_idx)
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
        if sp.issparse(T_total):
            return self.nstates * np.sum(T_total.diagonal())
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
        kinetic_action=None,
        rovibronic_kinetic=None,
    ):
        """Build the full kinetic propagator exp(-i T dt).

        Uses the analytical KEO in bond/angle coordinates.
        """
        import time
        kinetic_propagator = self._canonical_kinetic_propagator(kinetic_propagator)
        has_rotation = self._rotation_enabled()
        rovibronic_kinetic = _normalize_rovibronic_kinetic_method(rovibronic_kinetic)
        use_operator_kinetic = _normalize_kinetic_action(kinetic_action) == "matrix-free"

        if rovibronic_kinetic is not None:
            if not has_rotation:
                raise ValueError("rovibronic_kinetic requires J > 0.")
            if kinetic_propagator == "dense":
                raise ValueError(
                    "rovibronic_kinetic requires kinetic_propagator='expm_multiply' "
                    "or 'chebyshev'."
                )
            if rovibronic_kinetic == "sparse":
                print("Building sparse factorized rovibronic LDR matrix ...")
            elif rovibronic_kinetic == "compiled":
                print("Building compiled factorized rovibronic LDR LinearOperator ...")
            else:
                print("Building factorized rovibronic LDR LinearOperator ...")
            t0 = time.time()
            if rovibronic_kinetic == "sparse":
                self.H = self.build_sparse_factorized_rovibronic_ldr_matrix(verbose=True)
            elif rovibronic_kinetic == "compiled":
                self.H = self.build_compiled_factorized_rovibronic_ldr_action(verbose=True)
            else:
                self.H = self.build_fused_factorized_rovibronic_ldr_action(verbose=True)
            self.T_total = None
            self.kinetic_trace = (
                self.H.diagonal().sum()
                if sp.issparse(self.H)
                else getattr(self.H, "trace_value", None)
            )
            print(
                f"Factorized rovibronic kinetic built in {time.time() - t0:.2f} s, "
                f"shape = {self.H.shape}"
            )
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
            else:
                self.exp_T = None
            self.kinetic_propagator = kinetic_propagator
            return self.exp_T

        print("Building T_total ...")
        t0 = time.time()
        try:
            if has_rotation and use_operator_kinetic:
                T_total = self.buildK(sparse=False)
            else:
                T_total = self.buildK(sparse=use_operator_kinetic)
        except TypeError:
            T_total = self.buildK()
        if sp.issparse(T_total):
            T_total = 0.5 * (T_total + T_total.getH())
            T_total = T_total.tocsr()
        else:
            T_total = 0.5 * (T_total + T_total.conj().T)
        self.kinetic_trace = self._kinetic_trace_from_nuclear_operator(T_total)
        print(f"T_total built in {time.time() - t0:.2f} s, shape = {T_total.shape}")

        if use_operator_kinetic:
            if kinetic_propagator == "dense":
                raise ValueError(
                    "kinetic_action='matrix-free' requires kinetic_propagator="
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
        if getattr(self, "kinetic_trace", None) is not None:
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
        kinetic_action=None,
        rovibronic_kinetic=None,
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
            kinetic_action=kinetic_action,
            rovibronic_kinetic=rovibronic_kinetic,
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
        result = {'times': times, 'psilist': psilist}
        self.ldr_result = result
        self.ued_result = result
        return result



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

    def cartesian_grid(self, copy=True):
        """Cartesian atom coordinates on the current internal-coordinate grid.

        Returns
        -------
        coords : ndarray, shape (*nx, natom, 3)
            Body-fixed Cartesian coordinates for each DVR grid point.  The
            array is cached after the first call and invalidated by set_dvr().
        """
        if self.x is None or self.nx is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        cached = getattr(self, "_cartesian_grid_cache", None)
        if cached is None:
            axes = [np.asarray(axis, dtype=float) for axis in self.x]
            R1, R2, Theta = np.meshgrid(*axes, indexing="ij")
            coords = np.zeros((*self.nx, self.natom, 3), dtype=float)
            coords[..., 0, 0] = R1
            coords[..., 2, 0] = R2 * np.cos(Theta)
            coords[..., 2, 1] = R2 * np.sin(Theta)
            self._cartesian_grid_cache = coords
            cached = coords
        return cached.copy() if copy else cached

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
                elif kind in ('fedvr', 'fe-dvr', 'fe_dvr'):
                    params = dict(params)
                    n_elements = int(params.pop("n_elements", n))
                    n_lobatto = int(params.pop("n_lobatto", 5))
                    self.dvrs.append(
                        FEDVR(dom[0], dom[1], n_elements, n_lobatto, **params)
                    )
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
        self._cartesian_grid_cache = None
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

    def reference_projector(
        self,
        state,
        reference_index=None,
    ):
        """Project one reference electronic state into every local LDR basis.

        Returns an array of shape ``(*nx, nstates)`` containing
        ``<phi_a(q)|phi_state(q_ref)>``.  The projection uses the dense overlap
        tensor when present, otherwise it uses nearest-neighbor linked overlap
        transports directly.
        """
        state = int(state)
        if not 0 <= state < self.nstates:
            raise ValueError(f"state {state} outside [0, {self.nstates}).")
        if reference_index is None:
            reference_index = tuple(len(axis) // 2 for axis in self.x)
        else:
            reference_index = tuple(int(i) for i in reference_index)
        if len(reference_index) != self.ndim:
            raise ValueError("reference_index must have one entry per coordinate.")
        if any(i < 0 or i >= n for i, n in zip(reference_index, self.nx)):
            raise ValueError(
                f"reference_index {reference_index} outside grid shape {tuple(self.nx)}."
            )

        indices = self._grid_indices()
        flat_index = {idx: i for i, idx in enumerate(indices)}
        ref_flat = flat_index[reference_index]
        projector = np.zeros((*self.nx, self.nstates), dtype=complex)

        if self.overlap_matrix is not None:
            overlap = self.overlap_matrix.reshape(
                len(indices),
                self.nstates,
                len(indices),
                self.nstates,
            )
            for idx in indices:
                projector[idx] = overlap[flat_index[idx], :, ref_flat, state]
            return projector

        links = getattr(self, "overlap_links", None)
        if links is None:
            raise RuntimeError(
                "Reference-state projection requires overlap_matrix or overlap_links."
            )
        for idx in indices:
            block = self._linked_overlap_block(
                flat_index[idx],
                ref_flat,
                idx,
                reference_index,
                links,
                self.nstates,
            )
            projector[idx] = block[:, state]
        return projector

    def projected_initial_packet(
        self,
        state,
        width=80.0,
        widths=None,
        center=None,
        reference_index=None,
        momenta=None,
    ):
        """Build a normalized Gaussian packet with LDR reference projection.

        ``momenta`` gives the conjugate momenta for the internal coordinates
        and adds the phase ``exp(i p . (q - center))`` before normalization.
        """
        if center is None:
            center = np.array([axis[len(axis) // 2] for axis in self.x], dtype=float)
        else:
            center = np.asarray(center, dtype=float)
        if center.shape != (self.ndim,):
            raise ValueError("center must have one value per coordinate.")
        if widths is None:
            widths = np.full(self.ndim, float(width))
        else:
            widths = np.asarray(widths, dtype=float)
        if widths.shape != (self.ndim,):
            raise ValueError("widths must have one value per coordinate.")
        if momenta is None:
            momenta = np.zeros(self.ndim, dtype=float)
        else:
            momenta = np.asarray(momenta, dtype=float)
        if momenta.shape != (self.ndim,):
            raise ValueError("momenta must have one value per coordinate.")

        projector = self.reference_projector(
            state=state,
            reference_index=reference_index,
        )
        psi_values = np.zeros((*self.nx, self.nstates), dtype=complex)
        for idx in np.ndindex(*self.nx):
            q = np.array([self.x[axis][idx[axis]] for axis in range(self.ndim)])
            amp = np.exp(-np.sum(widths * (q - center) ** 2))
            phase = np.exp(1j * float(np.dot(momenta, q - center)))
            psi_values[idx] = amp * phase * projector[idx]

        psi = self.to_quadrature_normalized(psi_values)
        norm = self.norm(psi)
        if norm == 0.0:
            raise RuntimeError("Projected initial packet has zero norm.")
        return psi / norm

    def _as_state_overlap_matrix(self, ov, nstates):
        ov = np.asarray(ov, dtype=complex)
        if ov.ndim == 0:
            ov = ov.reshape(1, 1)
        if ov.shape != (nstates, nstates):
            raise ValueError(f"State overlap shape {ov.shape} != {(nstates, nstates)}.")
        return ov

    def _grid_indices(self):
        return list(np.ndindex(*self.nx))

    def _snake_grid_indices(self):
        """Return a nearest-neighbor-friendly serpentine traversal of the grid."""
        shape = tuple(int(n) for n in self.nx)

        def build(axis, reverse=False):
            values = range(shape[axis] - 1, -1, -1) if reverse else range(shape[axis])
            out = []
            for count, value in enumerate(values):
                if axis == len(shape) - 1:
                    out.append((value,))
                else:
                    child_reverse = bool(count % 2)
                    out.extend((value,) + tail for tail in build(axis + 1, child_reverse))
            return out

        return build(0, False)

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

    def _linked_overlap_between_path(self, bra_idx, ket_idx, links, nstates, axes):
        current = list(bra_idx)
        mat = np.eye(nstates, dtype=complex)

        for axis in axes:
            while current[axis] < ket_idx[axis]:
                src = tuple(current)
                mat = mat @ links[(axis, src)]
                current[axis] += 1
            while current[axis] > ket_idx[axis]:
                current[axis] -= 1
                src = tuple(current)
                mat = mat @ links[(axis, src)].conj().T

        return mat

    def _linked_overlap_between(self, bra_idx, ket_idx, links, nstates):
        active_axes = [
            axis for axis in range(self.ndim)
            if int(bra_idx[axis]) != int(ket_idx[axis])
        ]
        if not getattr(self, "overlap_path_average", False) or len(active_axes) <= 1:
            return self._linked_overlap_between_path(
                bra_idx,
                ket_idx,
                links,
                nstates,
                range(self.ndim),
            )

        paths = list(itertools.permutations(active_axes))
        out = np.zeros((nstates, nstates), dtype=complex)
        for axes in paths:
            out += self._linked_overlap_between_path(
                bra_idx,
                ket_idx,
                links,
                nstates,
                axes,
            )
        return out / len(paths)

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
        indices=None,
    ):
        if electronic_options is None:
            electronic_options = {}
        atom_symbols = tuple(self.atom_symbols())
        tasks = []
        if indices is None:
            indices = self._grid_indices()
        for idx in indices:
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

    def _run_electronic_structure_scan_with_scanner(
        self,
        tasks,
        nstates,
        worker_threads=1,
    ):
        """Run a serial electronic scan reusing the previous SCF solution."""
        total = len(tasks)
        grid_mc_objects = np.empty(self.nx, dtype=object)
        apes = np.zeros((*self.nx, nstates))
        report_every = max(1, total // 10)
        _set_worker_thread_limits(worker_threads)

        scanner = None
        electronic_method = None

        def retry_rohf(pmol, scanner_obj, previous_mf, options):
            if not bool(options.get("scanner_retry_ladder", True)):
                return previous_mf
            if getattr(previous_mf, "converged", False):
                return previous_mf

            base_max_cycle = int(options.get("max_cycle", 120))
            retry_max_cycle = int(options.get("scanner_retry_max_cycle", max(200, base_max_cycle)))
            conv_tol = float(options.get("scf_tol", 1.0e-9))
            conv_tol_dm = float(options.get("conv_tol_dm", 1.0e-6))
            diis = bool(options.get("diis", True))
            diis_start_cycle = int(options.get("diis_start_cycle", 2))
            diis_space = int(options.get("diis_space", 8))
            verbose = int(options.get("verbose", 0))
            dampings = options.get("scanner_retry_dampings", (0.25, 0.5, 0.1))
            if isinstance(dampings, str):
                dampings = [float(x) for x in dampings.split(",") if x.strip()]
            retry_conv_tol_dms = options.get(
                "scanner_retry_conv_tol_dms",
                (conv_tol_dm, max(conv_tol_dm, 5.0e-5), max(conv_tol_dm, 1.0e-4)),
            )
            if isinstance(retry_conv_tol_dms, str):
                retry_conv_tol_dms = [
                    float(x) for x in retry_conv_tol_dms.split(",") if x.strip()
                ]
            retry_diis_values = options.get("scanner_retry_diis", (diis, False))
            if isinstance(retry_diis_values, str):
                retry_diis_values = [
                    x.strip().lower() not in ("0", "false", "no", "off")
                    for x in retry_diis_values.split(",")
                    if x.strip()
                ]

            best_mf = previous_mf
            best_energy = getattr(previous_mf, "e_tot", np.inf)
            dm0 = scanner_obj._initial_density(pmol)
            steps = []
            for dm_tol in retry_conv_tol_dms:
                for damping_value in dampings:
                    for use_diis in retry_diis_values:
                        step = (float(damping_value), float(dm_tol), bool(use_diis))
                        if step not in steps:
                            steps.append(step)

            for damping_value, dm_tol, use_diis in steps:
                trial = ROHF(
                    pmol,
                    init_guess=getattr(previous_mf, "init_guess", "minao"),
                    verbose=verbose,
                )
                trial.run(
                    dm0=dm0,
                    init_guess="dm" if dm0 is not None else getattr(previous_mf, "init_guess", "minao"),
                    max_cycle=retry_max_cycle,
                    conv_tol=conv_tol,
                    conv_tol_dm=dm_tol,
                    damping=float(damping_value),
                    diis=use_diis,
                    diis_start_cycle=diis_start_cycle,
                    diis_space=diis_space,
                    verbose=verbose,
                )
                trial_energy = getattr(trial, "e_tot", np.inf)
                if np.isfinite(trial_energy) and (
                    not np.isfinite(best_energy) or trial_energy < best_energy or trial.converged
                ):
                    best_mf = trial
                    best_energy = trial_energy
                if trial.converged:
                    break

            scanner_obj.mf = best_mf
            scanner_obj.mol = best_mf.mol
            if getattr(best_mf, "converged", False):
                scanner_obj._guess_mf = best_mf
            return best_mf

        for count, task in enumerate(tasks, start=1):
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
                nstates_task,
                electronic_method_task,
                electronic_options,
            ) = task
            electronic_method_task = _normalize_triatomic_electronic_method(electronic_method_task)
            if electronic_method is None:
                electronic_method = electronic_method_task
            elif electronic_method != electronic_method_task:
                raise ValueError("Scanner scan received mixed electronic methods.")
            if electronic_method_task not in ("casci", "rohf-casci"):
                raise NotImplementedError(
                    "scan_pes(use_scanner=True) currently supports 'casci' and "
                    "'rohf/casci'."
                )

            atom_spec = [[symbol, tuple(coord)] for symbol, coord in zip(atom_symbols, xyz)]
            pmol = Molecule(atom=atom_spec, basis=basis, charge=charge, spin=spin, unit=unit)
            if scanner is None:
                pmol.build()
                if electronic_method_task == "casci":
                    mf = _run_native_rhf_with_retries(pmol, electronic_options)
                else:
                    mf = ROHF(pmol).run(
                        conv_tol=float(electronic_options.get("scf_tol", 1.0e-9)),
                        conv_tol_dm=float(electronic_options.get("conv_tol_dm", 1.0e-6)),
                        max_cycle=int(electronic_options.get("max_cycle", 120)),
                        verbose=int(electronic_options.get("verbose", 0)),
                        damping=float(electronic_options.get("damping", 0.25)),
                        diis=bool(electronic_options.get("diis", True)),
                        diis_start_cycle=int(electronic_options.get("diis_start_cycle", 2)),
                        diis_space=int(electronic_options.get("diis_space", 8)),
                    )
                scanner = mf.as_scanner()
                if electronic_method_task == "rohf-casci":
                    mf = retry_rohf(pmol, scanner, mf, electronic_options)
            else:
                scanner(pmol)
                mf = scanner.mf
                if electronic_method_task == "rohf-casci":
                    mf = retry_rohf(pmol, scanner, mf, electronic_options)

            active_nelecas = ncas if nelecas is None else nelecas
            mc = CASCI(mf, ncas=ncas, nelecas=active_nelecas, spin=spin)
            if electronic_method_task == "casci":
                mc.run(nstates=nstates_task)
            else:
                mc.run(
                    nstates=nstates_task,
                    method=electronic_options.get("casci_method", "direct_ci"),
                )
            output_nstates = int(electronic_options.get("output_nstates", nstates))
            energies = _finalize_casci_roots(mc, output_nstates, spin, electronic_options)
            grid_mc_objects[idx] = mc
            apes[idx] = energies
            if count % report_every == 0 or count == total:
                converged = getattr(mf, "converged", None)
                suffix = "" if converged is None else f", scf_converged={bool(converged)}"
                print(f"  ... {count}/{total} points computed{suffix}")

        return apes, grid_mc_objects

    @staticmethod
    def _driver_point_object(point):
        if isinstance(point, dict):
            for key in ("mc", "casci", "result", "object"):
                if key in point:
                    return point[key]
        return point

    @staticmethod
    def _driver_point_energies(point, nstates):
        if isinstance(point, dict):
            for key in ("energies", "e_tot", "energy"):
                if key in point:
                    energies = point[key]
                    break
            else:
                raise ValueError("Electronic driver result dictionary has no energies.")
        elif hasattr(point, "e_tot"):
            energies = point.e_tot
        elif isinstance(point, (tuple, list)) and point:
            energies = point[0]
        else:
            energies = point

        energies = np.atleast_1d(np.asarray(energies, dtype=float))
        if len(energies) < nstates:
            raise ValueError(
                f"Electronic driver returned {len(energies)} energies, expected {nstates}."
            )
        return energies[:nstates]

    def _run_electronic_driver_scan(
        self,
        driver,
        indices,
        nstates,
        worker_threads=1,
    ):
        """Run a serial scan using a user-supplied electronic driver/scanner."""
        _set_worker_thread_limits(worker_threads)
        if hasattr(driver, "as_scanner"):
            try:
                scanner = driver.as_scanner(nstates=nstates)
            except TypeError:
                scanner = driver.as_scanner()
        else:
            scanner = driver
        if not callable(scanner):
            raise TypeError("driver must be callable or provide as_scanner().")

        total = len(indices)
        grid_objects = np.empty(self.nx, dtype=object)
        apes = np.zeros((*self.nx, nstates))
        report_every = max(1, total // 10)

        for count, idx in enumerate(indices, start=1):
            q = [self.x[axis][idx[axis]] for axis in range(self.ndim)]
            xyz = np.asarray(self.internal_to_xyz(*q), dtype=float)
            point = scanner(xyz)
            obj = self._driver_point_object(point)
            grid_objects[idx] = obj
            apes[idx] = self._driver_point_energies(point, nstates)
            if count % report_every == 0 or count == total:
                converged = getattr(getattr(scanner, "mf", None), "converged", None)
                suffix = "" if converged is None else f", scf_converged={bool(converged)}"
                print(f"  ... {count}/{total} points computed{suffix}")

        return apes, grid_objects, scanner

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
        scan_roots=None,
        spin_filter="none",
        spin_filter_tol=1.0e-3,
        spin_filter_target_s2=None,
        scf_tol=1.0e-9,
        conv_tol_dm=1.0e-6,
        max_cycle=120,
        damping=0.0,
        level_shift=0.0,
        diis=True,
        diis_start_cycle=2,
        diis_space=8,
        spin_penalty=None,
        target_spin=None,
        target_s2=None,
        init_guess="hcore",
        rhf_retry_ladder=True,
        rhf_retry_select_lowest=False,
        use_scanner=False,
        scanner_order="snake",
        scanner_retry_ladder=True,
        scanner_retry_dampings=(0.25, 0.5, 0.1),
        scanner_retry_conv_tol_dms=None,
        scanner_retry_diis=(True, False),
        scanner_retry_max_cycle=None,
        driver=None,
    ):
        """
        Scan the adiabatic PES over the DVR grid and compute the non-adiabatic
        overlap matrix. Results are always saved to disk as three files:
            apes.npz            -- adiabatic potential energy surfaces
            overlap_matrix.npz  -- LDR electronic overlap matrix, unless
                                   overlap_method='link-only'
            overlap_links.npz   -- nearest-neighbor LDR links, for linked
                                   and link-only overlap methods
            electronic_data.npz -- AO density matrices and AO-origin geometry
                                   when the electronic backend provides them

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
        overlap_method : {'linked', 'full', 'link-only', 'none'}
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
        electronic_method : {'casci', 'rohf/casci', 'am1/meci', 'uam1/meci'}, optional
            Electronic-structure backend used at each grid point.  ``'casci'``
            keeps the original native RHF/CASCI path.  ``'am1/meci'`` uses the
            native semiempirical RHF-AM1 reference followed by full MECI in the
            frontier active space specified by ``ncas``.  ``'uam1/meci'`` uses
            the unrestricted AM1 reference for open-shell systems such as NO2.
            ``'rohf/casci'`` uses the native restricted open-shell HF
            reference followed by native CASCI.
        scf_tol, max_cycle, damping
            SCF controls used by native ROHF and AM1-family backends.
        conv_tol_dm : float, optional
            Density-matrix convergence tolerance for native ROHF scans.
        diis, diis_start_cycle, diis_space
            DIIS controls for native ROHF scans.
        spin_penalty : float or None, optional
            If set for AM1/MECI or UAM1/MECI, diagonalize
            ``H + spin_penalty * (S^2 - target_s2)^2`` while reporting
            physical CI energy expectations for the selected vectors.
        target_spin, target_s2 : float or None, optional
            Spin target for the penalty.  ``target_spin`` is the spin quantum
            number ``S``.  If both are omitted, the target is inferred from
            the fixed spin projection.
        use_scanner : bool, optional
            If True, run a serial ordered scan that reuses the previous SCF
            density/orbitals through the backend scanner.  This is useful for
            ROHF/CASCI grids where independent ``minao`` guesses can converge
            to discontinuous open-shell solutions.
        scanner_order : {'snake', 'lexicographic'}, optional
            Grid traversal used when ``use_scanner=True``.  ``'snake'`` keeps
            consecutive points adjacent on the tensor grid.
        scanner_retry_ladder : bool, optional
            If True, retry an unconverged ROHF scanner point from the last
            converged scanner density using the damping values in
            ``scanner_retry_dampings``.
        scanner_retry_dampings : sequence of float, optional
            Damping values used by the ROHF scanner retry ladder.
        scanner_retry_conv_tol_dms : sequence of float or None, optional
            Density residual tolerances used by retry attempts.  If omitted,
            retries use the requested tolerance plus looser ``5e-5`` and
            ``1e-4`` thresholds.
        scanner_retry_diis : sequence of bool, optional
            DIIS settings tried by the retry ladder.
        scanner_retry_max_cycle : int or None, optional
            Maximum ROHF iterations for retry attempts.  Defaults to at least
            200 cycles.
        Returns
        -------
        apes : ndarray, shape (*nx, nstates)
            Adiabatic potential energy surfaces on the DVR grid.
        overlap_matrix_or_links : ndarray or dict
            Full pairwise overlap matrix for ``'full'`` and ``'linked'``;
            nearest-neighbor link dictionary for ``'link-only'``; or ``None``
            for ``'none'``.
        electronic_data : dict
            Electronic-structure scan data for downstream observables,
            including AO density matrices, transition density matrices, and
            Cartesian geometries when the backend provides them.
        """
        if self.x is None:
            raise RuntimeError("DVR grids not set. Call set_dvr() first.")
        if not hasattr(self, "internal_to_xyz"):
            raise NotImplementedError(
                "Method 'internal_to_xyz(r1, r2, theta)' must be implemented."
            )

        driver = self.driver if driver is None else driver
        using_driver = driver is not None
        if using_driver:
            driver_ref = getattr(driver, "template", driver)
            driver_mol = getattr(driver_ref, "mol", None)
            basis = getattr(driver_mol, "basis", basis)
            ncas = getattr(driver_ref, "ncas", ncas)
            nelecas = getattr(driver_ref, "nelecas", nelecas)

        if nstates is None:
            driver_nstates = getattr(driver, "nstates", None)
            nstates = self.nstates if driver_nstates is None else driver_nstates
        nstates = int(nstates)
        if scan_roots is None:
            scan_roots = nstates
        scan_roots = int(scan_roots)
        if scan_roots < nstates:
            raise ValueError("scan_roots must be >= nstates.")
        if using_driver:
            electronic_method = "driver"
        else:
            electronic_method = _normalize_triatomic_electronic_method(electronic_method)
            if electronic_method not in ("casci", "rohf-casci", "am1-meci", "uam1-meci"):
                raise ValueError(
                    "electronic_method must be 'casci', 'rohf/casci', 'am1/meci', or 'uam1/meci'."
                )
        nx = self.nx  # [N_r1, N_r2, N_theta]
        overlap_method = overlap_method.lower().replace("_", "-")
        if overlap_method == "links":
            overlap_method = "link-only"
        if overlap_method not in ("linked", "full", "link-only", "none"):
            raise ValueError(
                "overlap_method must be 'linked', 'full', 'link-only', or 'none'."
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
            scan_roots=scan_roots,
            spin_filter=spin_filter,
            spin_filter_tol=spin_filter_tol,
            spin_filter_target_s2=spin_filter_target_s2,
            scf_tol=scf_tol,
            conv_tol_dm=conv_tol_dm,
            max_cycle=max_cycle,
            damping=damping,
            level_shift=level_shift,
            diis=diis,
            diis_start_cycle=diis_start_cycle,
            diis_space=diis_space,
            spin_penalty=spin_penalty,
            target_spin=target_spin,
            target_s2=target_s2,
            init_guess=init_guess,
            rhf_retry_ladder=rhf_retry_ladder,
            rhf_retry_select_lowest=rhf_retry_select_lowest,
            use_scanner=use_scanner,
            scanner_order=scanner_order,
            scanner_retry_ladder=scanner_retry_ladder,
            scanner_retry_dampings=scanner_retry_dampings,
            scanner_retry_conv_tol_dms=scanner_retry_conv_tol_dms,
            scanner_retry_diis=scanner_retry_diis,
            scanner_retry_max_cycle=scanner_retry_max_cycle,
            driver=None if driver is None else driver.__class__.__name__,
        )

        def make_electronic_data(grid_mc_objects):
            coords = None
            if hasattr(self, "cartesian_grid"):
                try:
                    coords = self.cartesian_grid(copy=True)
                except Exception:
                    coords = None
            first_mc = None
            if grid_mc_objects is not None:
                first_mc = next((mc for mc in grid_mc_objects.flat if mc is not None), None)
            mol0 = getattr(getattr(first_mc, "mf", None), "mol", None)
            if mol0 is None:
                mol0 = first_mc.mol if hasattr(first_mc, "mol") else None

            data = {
                "coords": coords,
                "symbols": tuple(self.atom_symbols()),
                "basis": getattr(mol0, "basis", basis),
                "charge": getattr(mol0, "charge", self.charge),
                "spin": getattr(mol0, "spin", self.spin),
                "unit": getattr(mol0, "unit", self.unit),
                "nstates": nstates,
                "ncas": ncas,
                "nelecas": nelecas,
                "electronic_method": electronic_method,
            }
            if grid_mc_objects is None:
                return data

            if (
                first_mc is None
                or not hasattr(first_mc, "make_rdm1")
                or not hasattr(first_mc, "make_tdm1")
                or not hasattr(first_mc, "mf")
                or not hasattr(first_mc.mf, "mol")
            ):
                return data

            mol0 = first_mc.mf.mol
            nao = int(mol0.nao)
            dm1_ao = np.empty((*self.nx, nstates, nao, nao), dtype=complex)
            tdm1_ao = np.empty((*self.nx, nstates, nstates, nao, nao), dtype=complex)
            for idx in np.ndindex(*self.nx):
                mc = grid_mc_objects[idx]
                for state in range(nstates):
                    dm1_ao[idx + (state,)] = mc.make_rdm1(
                        state,
                        with_core=True,
                        representation="ao",
                    )
                for bra in range(nstates):
                    for ket in range(nstates):
                        tdm1_ao[idx + (bra, ket)] = mc.make_tdm1(
                            bra,
                            ket,
                            with_core=True,
                            representation="ao",
                        )

            data["dm1_ao"] = dm1_ao
            data["tdm1_ao"] = tdm1_ao
            data["nao"] = nao

            try:
                from pyqed.qchem.fourier import AOPairFTPlan

                plan = AOPairFTPlan.from_molecule(mol0)
                data["ao_ft_plan"] = plan
                if coords is not None:
                    ao_origins = np.empty((*self.nx, plan.ncart, 3), dtype=float)
                    for idx in np.ndindex(*self.nx):
                        ao_origins[idx] = plan.origins_from_atom_coords(coords[idx])
                    data["ao_origins"] = ao_origins
            except Exception:
                pass

            return data

        def save_electronic_data(electronic_data):
            arrays = {
                key: electronic_data[key]
                for key in ("coords", "dm1_ao", "tdm1_ao", "ao_origins")
                if key in electronic_data and electronic_data[key] is not None
            }
            if "dm1_ao" not in arrays or "tdm1_ao" not in arrays:
                return
            arrays["meta"] = np.array(meta)
            np.savez("electronic_data.npz", **arrays)
            print("[scan_pes] Saved electronic_data.npz")

        def scan_return(apes, overlap_data, grid_mc_objects, save_electronic=False):
            electronic_data = make_electronic_data(grid_mc_objects)
            self.ed = electronic_data
            self.electronic_data = electronic_data
            self.ued_data = electronic_data
            if save_electronic:
                save_electronic_data(electronic_data)
            return apes, overlap_data, electronic_data

        # --- 0. Skip if external inputs are already provided -------------------
        if self.apes is not None and (
            self.overlap_matrix is not None or self.overlap_links is not None
        ):
            print("[scan_pes] External PES and LDR overlaps already set. Skipping.")
            grid_mc_objects = getattr(self, "grid_mc_objects", None)
            overlap_data = (
                self.overlap_matrix
                if self.overlap_matrix is not None
                else self.overlap_links
            )
            return scan_return(self.apes, overlap_data, grid_mc_objects)

        # --- 1. Scan: compute electronic states at every grid point ------------
        print(
            f"[scan_pes] Scanning PES "
            f"(method={electronic_method}, basis={basis}, ncas={ncas}, "
            f"nelecas={nelecas}, n_workers={n_workers}, use_scanner={use_scanner or using_driver}) ..."
        )
        total = np.prod(nx)
        electronic_options = dict(
            scf_tol=scf_tol,
            max_cycle=max_cycle,
            damping=damping,
            level_shift=level_shift,
            verbose=verbose,
            conv_tol_dm=conv_tol_dm,
            diis=diis,
            diis_start_cycle=diis_start_cycle,
            diis_space=diis_space,
            spin_penalty=spin_penalty,
            target_spin=target_spin,
            target_s2=target_s2,
            init_guess=init_guess,
            rhf_retry_ladder=rhf_retry_ladder,
            rhf_retry_select_lowest=rhf_retry_select_lowest,
            output_nstates=nstates,
            spin_filter=spin_filter,
            spin_filter_tol=spin_filter_tol,
            spin_filter_target_s2=spin_filter_target_s2,
            scanner_retry_ladder=scanner_retry_ladder,
            scanner_retry_dampings=scanner_retry_dampings,
            scanner_retry_diis=scanner_retry_diis,
        )
        if scanner_retry_conv_tol_dms is not None:
            electronic_options["scanner_retry_conv_tol_dms"] = scanner_retry_conv_tol_dms
        if scanner_retry_max_cycle is not None:
            electronic_options["scanner_retry_max_cycle"] = scanner_retry_max_cycle
        scanner_order = str(scanner_order).lower().replace("_", "-")
        if scanner_order not in ("snake", "lexicographic"):
            raise ValueError("scanner_order must be 'snake' or 'lexicographic'.")
        indices = (
            self._snake_grid_indices()
            if (use_scanner or using_driver) and scanner_order == "snake"
            else self._grid_indices()
        )
        driver_scanner = None
        if using_driver:
            if n_workers is not None and int(n_workers) != 1:
                warnings.warn(
                    "scan_pes(driver=...) is serial; ignoring n_workers > 1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.apes, grid_mc_objects, driver_scanner = self._run_electronic_driver_scan(
                driver,
                indices,
                nstates,
                worker_threads=worker_threads,
            )
        else:
            tasks = self._electronic_structure_tasks(
                basis,
                ncas,
                nelecas,
                scan_roots,
                electronic_method=electronic_method,
                electronic_options=electronic_options,
                indices=indices,
            )
            if len(tasks) != total:
                raise RuntimeError("Internal error while preparing electronic-structure tasks.")
        if (not using_driver) and use_scanner:
            if n_workers is not None and int(n_workers) != 1:
                warnings.warn(
                    "scan_pes(use_scanner=True) is serial; ignoring n_workers > 1.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.apes, grid_mc_objects = self._run_electronic_structure_scan_with_scanner(
                tasks,
                nstates,
                worker_threads=worker_threads,
            )
        elif not using_driver:
            self.apes, grid_mc_objects = self._run_electronic_structure_scan(
                tasks,
                nstates,
                n_workers=n_workers,
                worker_threads=worker_threads,
            )
        self.grid_mc_objects = grid_mc_objects
        overlap_fn = (
            driver_scanner.overlap
            if driver_scanner is not None and hasattr(driver_scanner, "overlap")
            else _electronic_state_overlap
        )

        np.savez("apes.npz", data=self.apes, meta=np.array(meta))
        print("[scan_pes] Saved apes.npz")

        # --- 2. Compute LDR overlap matrix -------------------------------------
        if overlap_method == "none":
            self.overlap_links = None
            self.overlap_matrix = None
            return scan_return(
                self.apes,
                None,
                grid_mc_objects,
                save_electronic=True,
            )

        if overlap_method == "linked":
            print("[scan_pes] Computing overlap matrix with linked-product approximation ...")
            self.overlap_links = self._compute_overlap_links(
                grid_mc_objects,
                nstates,
                overlap_fn=overlap_fn,
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
                overlap_fn=overlap_fn,
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
            return scan_return(
                self.apes,
                self.overlap_links,
                grid_mc_objects,
                save_electronic=True,
            )
        else:
            print("[scan_pes] Computing full pairwise overlap matrix ...")
            self.overlap_matrix = self._build_full_overlap_matrix(
                grid_mc_objects,
                nstates,
                overlap_fn=overlap_fn,
            )

        np.savez("overlap_matrix.npz", data=self.overlap_matrix, meta=np.array(meta))
        print("[scan_pes] Saved overlap_matrix.npz")

        return scan_return(
            self.apes,
            self.overlap_matrix,
            grid_mc_objects,
            save_electronic=True,
        )

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


Triatomic = Triatom


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
