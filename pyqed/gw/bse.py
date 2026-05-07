
# -*- coding: utf-8 -*-
"""

Dense molecular GW/BSE

Created on Sat Jan  5 23:55:43 2019
@author: Bing Gu

@description:
    The theory for the code can be found in [Fabien Bruneval, J. Chem. Phys. 2012, 136, 194107]

Based on the PySCF implementation
"""

import numpy as np
import scipy.linalg
import sys
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from pyscf import dft
from functools import reduce

from pyqed import is_positive_def
from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix


def _nelectron(mol):
    if hasattr(mol, 'nelectron'):
        return mol.nelectron
    if hasattr(mol, 'nelec'):
        return mol.nelec
    raise AttributeError("GW-BSE requires mol.nelectron or mol.nelec.")


def _get_k(mf):
    if hasattr(mf, 'get_k'):
        return mf.get_k()
    if hasattr(mf, 'get_jk'):
        return mf.get_jk()[1]
    raise AttributeError("GW-BSE requires an HF object with get_k() or get_jk().")


def _get_mo_eri(mf, mo_coeff, ao2mofn):
    if hasattr(mf, 'get_eri_mo'):
        return np.asarray(mf.get_eri_mo(mo_coeff=mo_coeff, notation='chem'))
    nmo = len(mf.mo_energy)
    return ao2mofn(mf.mol, (mo_coeff, mo_coeff, mo_coeff, mo_coeff),
                   compact=False).reshape(nmo, nmo, nmo, nmo)


def _get_ao_eri_factors(mf):
    eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(mf.mol, 'eri_factors', None)
    if eri_factors is None:
        return None
    return np.asarray(eri_factors, dtype=float)


def _get_mo_pair_factors(mf, mo_coeff):
    eri_factors = _get_ao_eri_factors(mf)
    if eri_factors is None:
        return None
    return np.einsum('Pmn,mp,nq->Ppq', eri_factors, mo_coeff, mo_coeff, optimize=True)


def _is_gw_reference(obj):
    return (
        hasattr(obj, '_scf')
        and hasattr(obj, 'e_qp')
        and hasattr(obj, '_qp_energy_so')
        and hasattr(obj, 'mo_coeff')
    )


def _spatial_eri_from_gw_reference(gw_ref, ao2mofn):
    if getattr(gw_ref, '_pair_factors', None) is not None:
        return None
    if getattr(gw_ref, 'eri', None) is not None:
        eri = np.asarray(gw_ref.eri)
        nmo = np.asarray(gw_ref.mo_coeff).shape[1]
        if eri.shape[0] == 2 * nmo:
            return eri[0::2, 0::2, 0::2, 0::2].real
        return eri.real
    return _get_mo_eri(gw_ref._scf, gw_ref.mo_coeff, ao2mofn)


def _copy_mf_scan_options(mf):
    opts = {
        "verbose": 0,
        "dm0": None if getattr(mf, "dm", None) is None else np.array(mf.dm, copy=True),
        "init_guess": "hcore",
    }
    if getattr(mf, "density_fit", False):
        opts["density_fit"] = True
        opts["auxbasis"] = getattr(mf, "auxbasis", None)
    elif getattr(mf, "cholesky_jk", False):
        opts["cholesky_jk"] = True
        opts["cholesky_tol"] = getattr(mf, "cholesky_tol", None)
        opts["cholesky_max_rank"] = getattr(mf, "cholesky_max_rank", None)
    return opts


def _rebuild_scan_mol(mol, build_driver=None):
    driver = build_driver or getattr(mol, "_build_driver", None) or "builtin"
    if driver == "builtin":
        options = getattr(mol, "builtin_options", None)
        mol.build(driver=driver, options=options)
    else:
        mol.build(driver=driver)
    return mol


def _pair_factor(gw, p, q):
    if getattr(gw, '_pair_factors', None) is None:
        return None
    return gw._pair_factors[:, p, q]


def _eri(gw, p, q, r, s):
    if gw.eri is not None:
        return gw.eri[p, q, r, s]
    pq = _pair_factor(gw, p, q)
    rs = _pair_factor(gw, r, s)
    if pq is None or rs is None:
        return 0.0
    return float(np.dot(pq, rs))


def _symmetrize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + mat.T.conjugate())


def _positive_matrix_power(mat, power, name, thresh=1e-10):
    evals, evecs = scipy.linalg.eigh(_symmetrize(mat))
    if evals[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"{name} is not positive semidefinite; lowest eigenvalue = {evals[0]:.6e}."
        )
    evals = np.clip(evals, 0.0, None)
    if power < 0 and np.any(evals <= thresh):
        raise np.linalg.LinAlgError(
            f"{name} is numerically singular; cannot form inverse power."
        )
    powered = evals ** power
    return (evecs * powered) @ evecs.T.conjugate(), evals


def _casida_eigh(A, B, thresh=1e-10):
    """Stable dense RPA/Casida diagonalization without scipy.sqrtm."""
    a_minus_b = _symmetrize(A - B)
    a_plus_b = _symmetrize(A + B)
    sqrt_a_minus_b, _ = _positive_matrix_power(a_minus_b, 0.5, "A-B", thresh=thresh)
    casida_h = _symmetrize(sqrt_a_minus_b @ a_plus_b @ sqrt_a_minus_b)
    omega2, t = scipy.linalg.eigh(casida_h)
    if omega2[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"Casida matrix has negative eigenvalue = {omega2[0]:.6e}."
        )
    omega = np.sqrt(np.clip(omega2, 0.0, None))
    order = np.argsort(omega)
    return omega[order], t[:, order]


def _pseudo_normalize_full_bse_vectors(vectors, dim, thresh=1e-14):
    vectors = np.array(vectors, dtype=float, copy=True)
    for root in range(vectors.shape[1]):
        x = vectors[:dim, root]
        y = vectors[dim:, root]
        norm = float(np.dot(x, x) - np.dot(y, y))
        if abs(norm) < thresh:
            continue
        if norm < 0.0:
            vectors[:, root] *= -1.0
            norm = -norm
        vectors[:, root] /= np.sqrt(norm)
    return vectors


def _metric_orthonormalize_full_bse_vectors(vectors, dim, thresh=1e-10):
    """Orthonormalize full BSE vectors under the X^T X - Y^T Y metric."""
    vectors = _pseudo_normalize_full_bse_vectors(vectors, dim)
    metric = np.r_[np.ones(dim), -np.ones(dim)]
    basis = []
    for root in range(vectors.shape[1]):
        vec = vectors[:, root].copy()
        for prev in basis:
            vec -= prev * np.dot(prev * metric, vec)
        norm = float(np.dot(vec * metric, vec))
        if abs(norm) < thresh:
            continue
        if norm < 0.0:
            vec *= -1.0
            norm = -norm
        basis.append(vec / np.sqrt(norm))
    if len(basis) != vectors.shape[1]:
        raise RuntimeError(
            "Full BSE eigenvectors are linearly dependent under the BSE metric; "
            "try dense Casida BSE or request fewer roots."
        )
    return np.column_stack(basis)


def _full_bse_vectors_from_casida(A, B, nroots):
    """Solve Hermitian Casida BSE and reconstruct metric-orthonormal X/Y."""
    a_minus_b = A - B
    sqrt_a_minus_b, _ = _positive_matrix_power(a_minus_b, 0.5, "A-B")
    invsqrt_a_minus_b, _ = _positive_matrix_power(a_minus_b, -0.5, "A-B")
    casida_h = _symmetrize(sqrt_a_minus_b @ _symmetrize(A + B) @ sqrt_a_minus_b)
    omega2, z = scipy.linalg.eigh(casida_h)
    order = np.argsort(omega2)
    omega2 = omega2[order]
    z = z[:, order]
    positive = np.where(omega2 > 0.0)[0][:nroots]
    if positive.size < nroots:
        raise RuntimeError(
            f"Casida full BSE found only {positive.size} positive roots; requested {nroots}."
        )

    omega = np.sqrt(omega2[positive])
    z = z[:, positive]
    x_plus_y = sqrt_a_minus_b @ z
    x_minus_y = invsqrt_a_minus_b @ (z * omega[None, :])
    x = 0.5 * (x_plus_y + x_minus_y)
    y = 0.5 * (x_plus_y - x_minus_y)
    vectors = np.vstack((x, y)) / np.sqrt(omega)[None, :]
    return omega, vectors


class BSE(object):
    def __init__(self, gw_or_mf, ao2mofn=pyscf.ao2mo.outcore.general_iofree,
                 screening='TDH', eta=1e-2):

        gw_ref = gw_or_mf if _is_gw_reference(gw_or_mf) else None
        mf = gw_or_mf
        if gw_ref is not None:
            mf = gw_ref._scf
            screening = getattr(gw_ref, 'screening', screening)
            eta = getattr(gw_ref, 'eta', eta)

        assert screening in ('TDH', 'TDHF', 'TDDFT')
        
        self.mol = mf.mol
        self._scf = mf
        self.gw = gw_ref
        self.reference = gw_ref if gw_ref is not None else mf
        self.verbose = getattr(self.mol, 'verbose', getattr(mf, 'verbose', 0))
        self.stdout = getattr(self.mol, 'stdout', getattr(mf, 'stdout', sys.stdout))
        self.max_memory = getattr(mf, 'max_memory',
                                  getattr(self.mol, 'max_memory', 4000))

        self.nocc = _nelectron(self.mol)//2
        if gw_ref is not None:
            self.nso = np.asarray(gw_ref.mo_coeff).shape[1]
            self.e_mf = np.asarray(gw_ref.e_mf[0::2], dtype=float)
            self.mo_coeff = np.asarray(gw_ref.mo_coeff, dtype=float)
            self.v_mf = np.asarray(gw_ref.v_mf[0::2, 0::2], dtype=float)
            self._pair_factors = getattr(gw_ref, '_pair_factors', None)
            self.eri = _spatial_eri_from_gw_reference(gw_ref, ao2mofn)
        else:
            try:
                # DFT
                mf.xc = mf.xc
                v_mf = mf.get_veff() - mf.get_j()

            except AttributeError:
                # HF
                v_mf = -_get_k(mf)

        if gw_ref is not None:
            pass
        elif mf.mo_occ[0] == 2:
            # RHF, convert to spin-orbitals
#            nso = 2*len(mf.mo_energy)
#            self.nso = nso
#            self.e_mf = np.zeros(nso)
#            self.e_mf[0::2] = self.e_mf[1::2] = mf.mo_energy
#            b = np.zeros((nso//2,nso))
#            b[:,0::2] = b[:,1::2] = mf.mo_coeff
#            self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
#            self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0
#            eri = ao2mofn(mf.mol, (b,b,b,b),
#                          compact=False).reshape(nso,nso,nso,nso)
#            eri[::2,1::2] = eri[1::2,::2] = eri[:,:,::2,1::2] = eri[:,:,1::2,::2] = 0
#            # Integrals are in "chemist's notation"
#            # eri[i,j,k,l] = (ij|kl) = \int i(1) j(1) 1/r12 k(r2) l(r2)
#            print("Imag part of ERIs =", np.linalg.norm(eri.imag))
#            self.eri = eri.real

            nso = len(mf.mo_energy)
            self.nso = nso
            self.e_mf = mf.mo_energy
            b = mf.mo_coeff
            self.mo_coeff = np.asarray(b, dtype=float)
            self.v_mf = reduce(np.dot, (b.T, v_mf, b))
            self._pair_factors = _get_mo_pair_factors(mf, b)
            self.eri = None if self._pair_factors is not None else _get_mo_eri(mf, b, ao2mofn)

        else:
            # ROHF or UHF, these are already spin-orbitals
            print("\n*** Only supporting restricted calculations right now! ***\n")
            raise NotImplementedError
            nso = len(mf.mo_energy)
            self.nso = nso
            self.e_mf = mf.mo_energy
            b = mf.mo_coeff
            self.v_mf = reduce(np.dot, (b.T, v_mf, b))
            eri = ao2mofn(mf.mol, (b,b,b,b),
                          compact=False).reshape(nso,nso,nso,nso)
            self.eri = eri

        print("There are %d spatial orbitals"%(self.nso))

        self.screening = screening
        self.eta = eta
        self._M = None

        self._e_qp = None
        if gw_ref is not None and gw_ref.e_qp is not None:
            self.e_qp = gw_ref.e_qp
        self.e_rpa = None
        self._bse_tda_info = None
        self.excitation_energies = None
        self.e = None
        self.info = None
        self._xy = None
        self._x = None
        self._y = None
        self.bse_metric = None

    @property
    def e_qp(self):
        """Quasiparticle energies used by BSE/TDA.

        BSE/TDA ``.e`` stores excitation energies, so QP energies live under
        ``.e_qp``.  ``egw`` is kept as a backward-compatible alias.
        """
        return self._e_qp

    @e_qp.setter
    def e_qp(self, value):
        self._e_qp = value

    @property
    def egw(self):
        return self.e_qp

    @egw.setter
    def egw(self, value):
        self.e_qp = value

    def _store_excitations(self, energies, vectors, metric):
        self.excitation_energies = energies
        self.e = energies
        self.bse_metric = str(metric)
        if self.bse_metric == 'tda':
            self._xy = None
            if hasattr(self, 'XY'):
                del self.XY
            self.x = vectors
            self._y = None
        else:
            self.xy = vectors

    @property
    def xy(self):
        if self.bse_metric == 'tda':
            raise AttributeError("TDA amplitudes are stored as .x, not .xy.")
        return self._xy

    @xy.setter
    def xy(self, value):
        if self.bse_metric == 'tda':
            raise AttributeError("TDA amplitudes are stored as .x, not .xy.")
        self._xy = value
        self.XY = value  # Backward-compatible alias; prefer lowercase .xy.
        if self.bse_metric == 'full':
            self._x = None
            self._y = None

    @property
    def excitation_vectors(self):
        """Generic compatibility alias: ``x`` for TDA, stacked ``xy`` for BSE."""
        if self.bse_metric == 'tda':
            return self.x
        return self.xy

    @excitation_vectors.setter
    def excitation_vectors(self, value):
        if self.bse_metric == 'tda':
            self.x = value
        else:
            self.xy = value

    @property
    def x(self):
        """Excitation X amplitudes.

        For TDA this is the stored TDA vector.  For full BSE this is the upper
        half of the stacked ``xy`` vector.
        """
        if self.bse_metric != 'full':
            return self._x
        if self.xy is None:
            return None
        dim = self.nocc * (self.nso - self.nocc)
        return self.xy[:dim]

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def y(self):
        """De-excitation Y amplitudes for full BSE; ``None`` for TDA."""
        if self.bse_metric != 'full':
            return self._y
        if self.xy is None:
            return None
        dim = self.nocc * (self.nso - self.nocc)
        return self.xy[dim:]

    @y.setter
    def y(self, value):
        self._y = value

    def kernel(self, mo_energy=None, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        self.e_qp = kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
        logger.log(self, 'GW bandgap = %.15g', self.e_qp[self.nocc//2]-self.e_qp[self.nocc//2-1])
        return self.e_qp

    def sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
        return sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn)

    def g0(self, omega):
        return g0(self, omega)

    def get_m_rpa(self, e_rpa, t_rpa):
        return get_m_rpa(self, e_rpa, t_rpa)

    def rpa(self, using_tda=False, using_casida=True, method='TDH'):
        self.e_rpa, t = rpa(self, using_tda, using_casida, method)
        return self.e_rpa, t

    def rpa_AB_matrices(self, method='TDH'):
        return rpa_AB_matrices(self, method)

    def bse(self, using_tda=False, using_casida=True):
        return bse(self, using_tda, using_casida)

    def _ensure_screening(self):
        if self.e_rpa is None or self._M is None:
            e_rpa, t_rpa = self.rpa(method=self.screening)
            self._M = self.get_m_rpa(e_rpa, t_rpa)

    def _run_tda(
        self,
        nroots=5,
        low_rank='auto',
        use_qp=True,
        tol=1e-8,
        max_cycle=80,
        max_space=None,
        return_info=False,
        return_vectors=False,
    ):
        if not use_qp or self.e_qp is None:
            self.e_qp = self.e_mf.copy()

        use_low_rank = (
            low_rank is True
            or (low_rank == 'auto' and getattr(self, '_pair_factors', None) is not None)
        )
        if use_low_rank:
            self.bse_tda_low_rank(
                nroots=nroots,
                tol=tol,
                max_cycle=max_cycle,
                max_space=max_space,
                return_info=return_info,
                return_vectors=return_vectors,
            )
            return self

        self._ensure_screening()
        e, vec = self.bse(using_tda=True, using_casida=False)
        e = e[:nroots]
        vec = vec[:, :nroots]
        self._store_excitations(e, vec, 'tda')
        self.info = {"solver": "dense_tda", "converged": True}
        return self

    def run(
        self,
        nroots=5,
        low_rank='auto',
        use_qp=True,
        tol=1e-8,
        max_cycle=80,
        return_info=False,
        return_vectors=False,
    ):
        '''Run full BSE excitations and return ``self`` for a chainable API.

        ``kernel()`` retains the historical meaning of computing GW
        quasiparticle energies.  ``run()`` computes full BSE excitation
        energies and stores them in ``.e``.  Full BSE amplitudes are stored in
        ``.xy`` with ``.x``/``.y`` views.  Use :class:`TDA` for the
        Tamm-Dancoff approximation.
        '''
        if not use_qp or self.e_qp is None:
            self.e_qp = self.e_mf.copy()

        use_low_rank = (
            low_rank is True
            or (low_rank == 'auto' and getattr(self, '_pair_factors', None) is not None)
        )
        if use_low_rank:
            self.bse_full_low_rank(
                nroots=nroots,
                tol=tol,
                max_cycle=max_cycle,
                return_info=return_info,
                return_vectors=return_vectors,
            )
            return self

        self._ensure_screening()
        A, B = bse_AB_matrices(self)
        e, vec = _full_bse_vectors_from_casida(A, B, nroots)
        self._store_excitations(e, vec, 'full')
        self.info = {"solver": "dense_full_bse", "converged": True}
        return self

    def bse_full_low_rank(
        self,
        nroots=5,
        tol=1e-8,
        max_cycle=80,
        return_info=False,
        return_vectors=True,
    ):
        return bse_full_low_rank(
            self,
            nroots=nroots,
            tol=tol,
            max_cycle=max_cycle,
            return_info=return_info,
            return_vectors=return_vectors,
        )

    def wavefunction_overlap(
        self,
        other,
        bra_vectors=None,
        ket_vectors=None,
        ao_overlap=None,
        metric='auto',
    ):
        if bra_vectors is None:
            bra_vectors = self.excitation_vectors
        if ket_vectors is None and other is not self:
            ket_vectors = other.excitation_vectors
        if bra_vectors is None:
            raise ValueError("No BSE/TDA vectors supplied. Run with return_vectors=True first.")
        if ket_vectors is None and other is not self:
            raise ValueError("No ket BSE/TDA vectors supplied. Run with return_vectors=True first.")
        return bse_wavefunction_overlap(
            self,
            other,
            bra_vectors,
            ket_vectors=ket_vectors,
            ao_overlap=ao_overlap,
            metric=metric,
        )


    def bse_tda_low_rank(
        self,
        nroots=5,
        tol=1e-8,
        max_cycle=80,
        max_space=None,
        return_info=False,
        return_vectors=True,
    ):
        return bse_tda_low_rank(
            self,
            nroots=nroots,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            return_info=return_info,
            return_vectors=return_vectors,
        )

    def as_scanner(
        self,
        nroots=None,
        energy="pes",
        build_driver=None,
        gw_method=None,
        gw_kwargs=None,
        mf_kwargs=None,
        run_kwargs=None,
        return_object=False,
    ):
        """Return a callable scanner for GW/BSE potential-energy scans.

        The default return value is a total-energy PES array
        ``[E0, E0 + Omega_1, ...]`` using the SCF ground-state reference.
        Use ``energy="excitation"`` for only BSE/TDA excitation energies, or
        ``energy="rpa"`` for an RPA-shifted ground-state reference.
        """
        return GWBSEScanner(
            self,
            nroots=nroots,
            energy=energy,
            build_driver=build_driver,
            gw_method=gw_method,
            gw_kwargs=gw_kwargs,
            mf_kwargs=mf_kwargs,
            run_kwargs=run_kwargs,
            return_object=return_object,
            solver_cls=self.__class__,
        )


class TDA(BSE):
    """Tamm-Dancoff approximation to BSE.

    This class is a convenience wrapper around :class:`BSE` that always solves
    the TDA eigenproblem and uses the TDA wavefunction-overlap metric by
    default.
    """

    def run(
        self,
        nroots=5,
        low_rank='auto',
        use_qp=True,
        tol=1e-8,
        max_cycle=80,
        max_space=None,
        return_info=False,
        return_vectors=False,
    ):
        return self._run_tda(
            nroots=nroots,
            low_rank=low_rank,
            use_qp=use_qp,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            return_info=return_info,
            return_vectors=return_vectors,
        )

    def wavefunction_overlap(
        self,
        other,
        bra_vectors=None,
        ket_vectors=None,
        ao_overlap=None,
        metric='tda',
    ):
        return super().wavefunction_overlap(
            other,
            bra_vectors=bra_vectors,
            ket_vectors=ket_vectors,
            ao_overlap=ao_overlap,
            metric=metric,
        )


class GWBSEScanner:
    """Callable GW/BSE scanner for geometry-dependent excitation energies."""

    def __init__(
        self,
        base,
        nroots=None,
        energy="pes",
        build_driver=None,
        gw_method=None,
        gw_kwargs=None,
        mf_kwargs=None,
        run_kwargs=None,
        return_object=False,
        solver_cls=BSE,
    ):
        self.base = base
        self.mol = base.mol
        self.mf = base._scf
        self.gw = base.gw
        self.bse = base
        self.solver_cls = solver_cls
        self.nroots = nroots
        self.energy = str(energy).lower()
        self.build_driver = build_driver
        self.gw_method = gw_method or getattr(base.gw, "method", None) or "g0w0"
        self.gw_kwargs = {} if gw_kwargs is None else dict(gw_kwargs)
        self.mf_kwargs = {} if mf_kwargs is None else dict(mf_kwargs)
        self.run_kwargs = {} if run_kwargs is None else dict(run_kwargs)
        self.return_object = return_object
        self.e_scf = None
        self.e0 = None
        self.e = None

    def _prepare_mol(self, mol_or_geom):
        if isinstance(mol_or_geom, np.ndarray):
            mol = self.mol
            mol.set_geom(np.asarray(mol_or_geom, dtype=float).reshape(mol.natom, 3))
            return _rebuild_scan_mol(mol, self.build_driver)

        mol = mol_or_geom
        if getattr(mol, "hcore", None) is None or (
            getattr(mol, "eri", None) is None and getattr(mol, "eri_factors", None) is None
        ):
            return _rebuild_scan_mol(mol, self.build_driver)
        return mol

    def _run_mf(self, mol):
        from pyqed.qchem.hf.rhf import RHF

        kwargs = _copy_mf_scan_options(self.mf)
        kwargs.update(self.mf_kwargs)
        mf = RHF(mol, init_guess=getattr(self.mf, "init_guess", "hcore"))
        mf.max_cycle = getattr(self.mf, "max_cycle", mf.max_cycle)
        mf.run(**kwargs)
        return mf

    def _run_gw(self, mf):
        from pyqed.gw.gw import GW

        freq_int = getattr(self.gw, "freq_int", "exact") if self.gw is not None else "exact"
        gw = GW(mf, screening=self.base.screening, eta=self.base.eta, freq_int=freq_int)
        gw.run(method=self.gw_method, **self.gw_kwargs)
        return gw

    def __call__(self, mol_or_geom):
        mol = self._prepare_mol(mol_or_geom)
        mf = self._run_mf(mol)
        gw = self._run_gw(mf)

        nroots = self.nroots
        if nroots is None:
            nroots = len(self.bse.e) if self.bse.e is not None else 5
        run_kwargs = dict(self.run_kwargs)
        run_kwargs.setdefault("nroots", nroots)

        bse = self.solver_cls(gw).run(**run_kwargs)

        self.mol = mol
        self.mf = mf
        self.gw = gw
        self.bse = bse
        self.e_scf = float(mf.e_tot)

        if self.energy in ("excitation", "excited", "omega"):
            self.e0 = None
            self.e = np.asarray(bse.e, dtype=float)
        elif self.energy in ("pes", "scf", "total"):
            self.e0 = self.e_scf
            self.e = np.r_[self.e0, self.e0 + np.asarray(bse.e, dtype=float)]
        elif self.energy in ("rpa", "rpa_pes", "rpa-total", "rpa_total"):
            self.e0 = float(gw.total_energy(method="rpa"))
            self.e = np.r_[self.e0, self.e0 + np.asarray(bse.e, dtype=float)]
        else:
            raise ValueError("energy must be 'pes', 'excitation', or 'rpa'.")

        return bse if self.return_object else self.e




def g0(gw, omega):
    '''
    Return the 0th order GF matrix [G0]_{pq} in the basis of single-particle
    orbitals (MF eigenvectors).
    '''
    g0 = np.zeros((gw.nso,gw.nso), dtype=np.complex128)
    for p in range(gw.nso):
        if p < gw.nocc: sgn = -1
        else: sgn = +1
        g0[p,p] = 1.0/(omega - gw.e_mf[p] + 1j*sgn*gw.eta)
    return g0

def rpa_AB_matrices(gw, method='TDHF'):
    '''Get the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    assert method in ('TDH','TDHF','TDDFT')
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa,dim_rpa))
    B = np.zeros((dim_rpa,dim_rpa))

    ia = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            A[ia, ia] = gw.e_mf[a] - gw.e_mf[i]
            jb = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ia,jb] += 2.*_eri(gw, a, i, b, j)
                    B[ia,jb] += 2.*_eri(gw, i, a, j, b)
                    
                    if method == 'TDHF':
                        A[ia,jb] -= _eri(gw, a, b, i, j)
                        B[ia,jb] -= _eri(gw, a, j, i, b)
                    jb += 1
            ia += 1

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    return A, B

def rpa(gw, using_tda=False, using_casida=True, method='TDHF'):
    r'''Get the RPA eigenvalues and eigenvectors.

    Q^\dagger = \sum_{ia} X_{ia} a^+ i - Y_{ia} i^+ a
    Leads to the RPA eigenvalue equations:
      [ A  B ][X] = omega [ 1  0 ][X]
      [ B  A ][Y]         [ 0 -1 ][Y]
    which is equivalent to
      [ A  B ][X] = omega [ 1  0 ][X]
      [-B -A ][Y] =       [ 0  1 ][Y]

    See, e.g. Stratmann, Scuseria, and Frisch,
              J. Chem. Phys., 109, 8218 (1998)
    '''
    A, B = rpa_AB_matrices(gw, method=method)

    if using_tda:
        ham_rpa = A
        e, x = eig(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            return _casida_eigh(A, B)


def get_m_rpa(gw, e_rpa, t_rpa):
    r'''
    Get the (intermediate) M_{pq,L} tensor.
    The M (or w) is needed to construct the screened Coulomb interaction W
    
    .. math::
        
        M_{pq,L} = \sum_{ia} ( (eps_a-eps_i)/erpa_L )^{1/2} T_{ai,L} (ai|pq)
    '''
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc
    t_by_e = t_rpa.copy()
    for L in range(len(e_rpa)):
        t_by_e[:,L] /= np.sqrt(e_rpa[L])
    sqrt_eps = np.zeros(nocc*nvir)
    if gw.eri is None:
        pair_ai = np.zeros((nocc*nvir, gw._pair_factors.shape[0]))
    else:
        eri_product = np.zeros((nocc*nvir, nso, nso))
    ai = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            sqrt_eps[ai] = np.sqrt(gw.e_mf[a]-gw.e_mf[i])
            if gw.eri is None:
                pair_ai[ai, :] = _pair_factor(gw, a, i)
            else:
                eri_product[ai,:,:] = gw.eri[a,i,:,:]
            ai += 1
    if gw.eri is None:
        weighted_pairs = np.einsum('a,al,aP->Pl', sqrt_eps, t_by_e, pair_ai, optimize=True)
        M = np.einsum('Pl,Ppq->pql', weighted_pairs, gw._pair_factors, optimize=True)
    else:
        M = np.einsum('a,al,apq->pql', sqrt_eps, t_by_e, eri_product, optimize=True)
    return M



def sigma(gw, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
    r'''
    self energy 
    .. math::
        
        \Sigma_{pq} = i [GW]_{pq}
    '''
    if not isinstance(omegas, (list,tuple,np.ndarray)):
        single_point = True
        omegas = [omegas]
    else:
        single_point = False

    # This usually takes the longest:
    if gw._M is None:
        gw._M = get_m_rpa(gw, e_rpa, t_rpa)

    nso = gw.nso
    nocc = gw.nocc

    sigma_c = []
    sigma_x = []
    for omega in omegas:
        sigma_cw = 0.
        sigma_xw = 0.
        for L in range(len(e_rpa)):
            for i in range(nocc):
                sigma_cw += gw._M[i,q,L]*gw._M[i,p,L]/(
                            omega - gw.e_mf[i] + e_rpa[L] - 1j*gw.eta )
            for a in range(nocc, nso):
                sigma_cw += gw._M[a,q,L]*gw._M[a,p,L]/(
                            omega - gw.e_mf[a] - e_rpa[L] + vir_sgn*1j*gw.eta )
        for i in range(nocc):
            sigma_xw += -_eri(gw, p, i, i, q)

        sigma_c.append(sigma_cw)
        sigma_x.append(sigma_xw)

    if single_point:
        return sigma_c[0], sigma_x[0]
    else:
        return sigma_c, sigma_x

#def sigma(gw, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
#    '''
#    self energy sigma_{pq} = i [GW]_{pq}
#    '''
#    if not isinstance(omegas, (list,tuple,np.ndarray)):
#        single_point = True
#        omegas = [omegas]
#    else:
#        single_point = False
#
#    # This usually takes the longest:
#    if gw._M is None:
#        gw._M = get_m_rpa(gw, e_rpa, t_rpa)
#
#    nso = gw.nso
#    nocc = gw.nocc
#
#    sigma_c = []
#    sigma_x = []
#    for omega in omegas:
#        sigma_cw = 0.
#        sigma_xw = 0.
#        for L in range(len(e_rpa)):
#            for i in range(nocc):
#                sigma_cw += gw._M[i,q,L]*gw._M[i,p,L]/(
#                            omega - gw.e_mf[i] + e_rpa[L] - 1j*gw.eta )
#            for a in range(nocc, nso):
#                sigma_cw += gw._M[a,q,L]*gw._M[a,p,L]/(
#                            omega - gw.e_mf[a] - e_rpa[L] + vir_sgn*1j*gw.eta )
#        for i in range(nocc):
#            sigma_xw += -gw.eri[p,i,i,q]
#
#        sigma_c.append(sigma_cw)
#        sigma_x.append(sigma_xw)
#
#    if single_point:
#        return sigma_c[0], sigma_x[0]
#    else:
#        return sigma_c, sigma_x

def kernel(gw, so_energy, so_coeff, verbose=logger.NOTE):
    '''Get the GW-corrected spatial orbital energies.

    Note: Works in spin-orbitals but returns energies for spatial orbitals.

    Args:
        gw : instance of :class:`GW`
        so_energy : (nso,) ndarray
        so_coeff : (nso,nso) ndarray

    Returns:
        egw : (nso/2,) ndarray
            The GW-corrected spatial orbital energies.
    '''
    print("# --- Performing RPA calculation ...")
    e_rpa, t_rpa = rpa(gw, method=gw.screening)

    print('RPA eigenvalues = ', e_rpa)


    # store the RPA eigvalues for BSE calculations
    gw.e_rpa = e_rpa

    print("done.")
    print("# --- Calculating GW QP corrections ...")
    egw = np.zeros(int(gw.nso))
    for p in range(0,gw.nso):
        def quasiparticle(omega):
            sigma_c_ppw, sigma_x_ppw = sigma(gw, p, p, omega, e_rpa, t_rpa)
            sigma_ppw = sigma_c_ppw + sigma_x_ppw
            return omega - gw.e_mf[p] - (2.*sigma_ppw.real - gw.v_mf[p,p])
        try:
            egw[p] = newton(quasiparticle, gw.e_mf[p], tol=1e-6, maxiter=100)
        except RuntimeError:
            print("Newton-Raphson unconverged, setting GW eval to MF eval.")
            egw[p] = gw.e_mf[p]
        print(egw[p])
    print("done.")

    return egw

def bse_AB_matrices(gw):
    '''Get the BSE A and B matrices, using the quasiparticle energies computed
    from GW, and screened interaction W computed from RPA
    '''
    method = gw.screening
    assert method in ('TDH','TDHF','TDDFT')

    # restricted calculations only
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa,dim_rpa))
    B = np.zeros((dim_rpa,dim_rpa))

    ia = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            # with GW corrected quasiparticle energy
            A[ia, ia] = gw.e_qp[a] - gw.e_qp[i]
            # with HF orbital energy
            #A[ia, ia] = gw._scf.mo_energy[a] - gw._scf.mo_energy[i]

            jb = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ia,jb] += 2.*_eri(gw, a, i, b, j) - _eri(gw, a, b, i, j)
                    B[ia,jb] += 2.*_eri(gw, a, i, j, b) - _eri(gw, a, j, i, b)
                    #if method == 'TDHF':
                    #    A[ia,jb] -= gw.eri[a,b,i,j]
                    #    B[ia,jb] -= gw.eri[a,j,i,b]
                    for L in range(len(gw.e_rpa)):
                        # MOLGW's no-RI SCREENED_COULOMB stores the induced
                        # interaction as -2 w_s w_s / pole.  With the Casida
                        # amplitudes used here this corresponds to +4 M M / w.
                        A[ia, jb] += 4.*gw._M[i,j,L] * gw._M[a,b,L]/ gw.e_rpa[L]
                        B[ia, jb] += 4.*gw._M[i,b,L] * gw._M[a,j,L]/ gw.e_rpa[L]

                    jb += 1

            ia += 1

    #assert np.allclose(A, A.transpose())
    #assert np.allclose(B, B.transpose())

    return A, B


def _bse_tda_diag(gw):
    nocc = gw.nocc
    energy = gw.e_qp if gw.e_qp is not None else gw.e_mf
    return (energy[nocc:][None, :] - energy[:nocc, None]).reshape(-1)


def _bse_tda_matvec(gw, x):
    nocc = gw.nocc
    nvir = gw.nso - nocc
    xmat = np.asarray(x).reshape(nocc, nvir)
    diag = _bse_tda_diag(gw).reshape(nocc, nvir)
    y = diag * xmat

    pair_factors = getattr(gw, '_pair_factors', None)
    if pair_factors is not None:
        occ_occ = pair_factors[:, :nocc, :nocc]
        vir_vir = pair_factors[:, nocc:, nocc:]
        vir_occ = pair_factors[:, nocc:, :nocc]

        direct_projected = np.einsum('Pbj,jb->P', vir_occ, xmat, optimize=True)
        y += 2.0 * np.einsum('Pai,P->ia', vir_occ, direct_projected, optimize=True)

        exchange_projected = np.einsum('Pij,jb->Pib', occ_occ, xmat, optimize=True)
        y -= np.einsum('Pab,Pib->ia', vir_vir, exchange_projected, optimize=True)
    else:
        y += 2.0 * np.einsum(
            'aibj,jb->ia',
            gw.eri[nocc:, :nocc, nocc:, :nocc],
            xmat,
            optimize=True,
        )
        y -= np.einsum(
            'abij,jb->ia',
            gw.eri[nocc:, nocc:, :nocc, :nocc],
            xmat,
            optimize=True,
        )

    if gw.e_rpa is None or gw._M is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)
        gw._M = gw.get_m_rpa(e_rpa, t_rpa)

    y += 4.0 * np.einsum(
        'ijL,abL,jb,L->ia',
        gw._M[:nocc, :nocc, :],
        gw._M[nocc:, nocc:, :],
        xmat,
        1.0 / gw.e_rpa,
        optimize=True,
    )
    return y.reshape(-1)


def _bse_b_matvec(gw, x):
    nocc = gw.nocc
    nvir = gw.nso - nocc
    xmat = np.asarray(x).reshape(nocc, nvir)
    y = np.zeros_like(xmat)

    pair_factors = getattr(gw, '_pair_factors', None)
    if pair_factors is not None:
        vir_occ = pair_factors[:, nocc:, :nocc]
        occ_vir = pair_factors[:, :nocc, nocc:]

        direct_projected = np.einsum('Pjb,jb->P', occ_vir, xmat, optimize=True)
        y += 2.0 * np.einsum('Pai,P->ia', vir_occ, direct_projected, optimize=True)

        exchange_projected = np.einsum('Pib,jb->Pij', occ_vir, xmat, optimize=True)
        y -= np.einsum('Paj,Pij->ia', vir_occ, exchange_projected, optimize=True)
    else:
        y += 2.0 * np.einsum(
            'aijb,jb->ia',
            gw.eri[nocc:, :nocc, :nocc, nocc:],
            xmat,
            optimize=True,
        )
        y -= np.einsum(
            'ajib,jb->ia',
            gw.eri[nocc:, :nocc, :nocc, nocc:],
            xmat,
            optimize=True,
        )

    if gw.e_rpa is None or gw._M is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)
        gw._M = gw.get_m_rpa(e_rpa, t_rpa)

    y += 4.0 * np.einsum(
        'ibL,ajL,jb,L->ia',
        gw._M[:nocc, nocc:, :],
        gw._M[nocc:, :nocc, :],
        xmat,
        1.0 / gw.e_rpa,
        optimize=True,
    )
    return y.reshape(-1)


def _bse_full_matvec(gw, xy):
    dim = gw.nocc * (gw.nso - gw.nocc)
    x = xy[:dim]
    y = xy[dim:]
    ax_by = _bse_tda_matvec(gw, x) + _bse_b_matvec(gw, y)
    bx_ay = _bse_b_matvec(gw, x) + _bse_tda_matvec(gw, y)
    return np.concatenate((ax_by, -bx_ay))


def bse_tda_low_rank(
    gw,
    nroots=5,
    tol=1e-8,
    max_cycle=80,
    max_space=None,
    return_info=False,
    return_vectors=True,
):
    '''Lowest TDA-BSE roots from a factorized/matrix-free Davidson solve.'''
    from pyqed.davidson import davidson

    diag = _bse_tda_diag(gw)
    nroots = min(int(nroots), diag.size)
    if nroots < 1:
        raise ValueError("nroots must be positive.")

    if gw.e_rpa is None or gw._M is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)
        gw._M = gw.get_m_rpa(e_rpa, t_rpa)

    eigvals, eigvecs, info = davidson(
        lambda vec: _bse_tda_matvec(gw, vec),
        neigen=nroots,
        tol=tol,
        itermax=max_cycle,
        diag=diag,
        max_space=max_space,
        return_info=True,
        return_partial=True,
    )
    gw._bse_tda_info = info
    gw.info = info
    if hasattr(gw, '_store_excitations'):
        gw._store_excitations(eigvals, eigvecs, 'tda')
    else:
        gw.excitation_energies = eigvals
        gw.excitation_vectors = eigvecs
        gw.xy = eigvecs
        gw.XY = eigvecs
        gw.bse_metric = 'tda'

    if not return_vectors:
        if return_info:
            return eigvals, info
        return eigvals
    if return_info:
        return eigvals, eigvecs, info
    return eigvals, eigvecs


def bse_full_low_rank(
    gw,
    nroots=5,
    tol=1e-8,
    max_cycle=80,
    return_info=False,
    return_vectors=True,
):
    '''Lowest positive full BSE roots from a matrix-free low-rank block solve.'''
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    dim = gw.nocc * (gw.nso - gw.nocc)
    nroots = min(int(nroots), dim)
    if nroots < 1:
        raise ValueError("nroots must be positive.")

    if gw.e_rpa is None or gw._M is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)
        gw._M = gw.get_m_rpa(e_rpa, t_rpa)

    shape = (2 * dim, 2 * dim)
    operator = LinearOperator(
        shape,
        matvec=lambda vec: _bse_full_matvec(gw, vec),
        dtype=float,
    )
    k = min(max(2 * nroots + 4, nroots + 2), shape[0] - 2)
    info = {
        "solver": "low_rank_full_bse",
        "converged": True,
        "nroots": nroots,
        "arnoldi_roots": k,
    }
    try:
        eigvals, eigvecs = eigs(
            operator,
            k=k,
            which='SM',
            tol=tol,
            maxiter=max_cycle,
        )
    except ArpackNoConvergence as err:
        eigvals = err.eigenvalues
        eigvecs = err.eigenvectors
        info["converged"] = False
        info["message"] = str(err)

    eigvals = np.asarray(eigvals)
    eigvecs = np.asarray(eigvecs)
    real_mask = np.abs(eigvals.imag) < max(1e-8, 100 * tol)
    pos = np.where(real_mask & (eigvals.real > 0.0))[0]
    order = pos[np.argsort(eigvals.real[pos])]
    if order.size < nroots:
        raise RuntimeError(
            f"Low-rank full BSE found only {order.size} positive real roots; requested {nroots}."
        )
    order = order[:nroots]
    roots = eigvals.real[order]
    vectors = eigvecs[:, order].real
    vectors = _metric_orthonormalize_full_bse_vectors(vectors, dim)
    gw._bse_full_info = info
    gw.info = info
    if hasattr(gw, '_store_excitations'):
        gw._store_excitations(roots, vectors, 'full')
    else:
        gw.excitation_energies = roots
        gw.excitation_vectors = vectors
        gw.xy = vectors
        gw.XY = vectors
        gw.bse_metric = 'full'

    if not return_vectors:
        if return_info:
            return roots, info
        return roots
    if return_info:
        return roots, vectors, info
    return roots, vectors


def _as_state_matrix(vectors, name):
    arr = np.asarray(vectors)
    was_vector = arr.ndim == 1
    if was_vector:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a vector or a 2D array of column vectors.")
    return arr, was_vector


def _same_molecule(bra, ket):
    if bra is ket:
        return True
    mol_bra = getattr(bra, "mol", None)
    mol_ket = getattr(ket, "mol", None)
    return mol_bra is not None and mol_bra is mol_ket


def _same_geometry_ao_overlap(bse_obj):
    mf = bse_obj._scf
    if hasattr(mf, "get_ovlp"):
        return np.asarray(mf.get_ovlp(), dtype=float)
    overlap = getattr(bse_obj.mol, "overlap", None)
    if overlap is None:
        raise ValueError("AO overlap was not supplied and cannot be inferred.")
    return np.asarray(overlap, dtype=float)


def _cross_geometry_ao_overlap(bra, ket):
    try:
        return _cross_ao_overlap_matrix(bra.mol, ket.mol)
    except Exception:
        try:
            from pyscf import gto

            return np.asarray(gto.intor_cross("int1e_ovlp", bra.mol, ket.mol), dtype=float)
        except Exception as err:
            raise ValueError(
                "AO overlap was not supplied and could not be built for the two geometries."
            ) from err


def _bse_cross_mo_overlap(bra, ket, ao_overlap=None):
    if ao_overlap is None:
        ao_overlap = (
            _same_geometry_ao_overlap(bra)
            if _same_molecule(bra, ket)
            else _cross_geometry_ao_overlap(bra, ket)
        )
    return (
        np.asarray(bra._scf.mo_coeff).conj().T
        @ np.asarray(ao_overlap)
        @ np.asarray(ket._scf.mo_coeff)
    )


def _single_excitation_list(nocc, nmo):
    occ = np.arange(nocc, dtype=int)
    excitations = []
    for i in range(nocc):
        for a in range(nocc, nmo):
            exc = occ.copy()
            exc[i] = a
            excitations.append((i, a, exc))
    return occ, excitations


def _singlet_single_excitation_metric(s_mo, nocc):
    nmo_bra, nmo_ket = s_mo.shape
    if nocc > nmo_bra or nocc > nmo_ket:
        raise ValueError("nocc is larger than one of the MO spaces.")

    occ_bra, bra_excitations = _single_excitation_list(nocc, nmo_bra)
    occ_ket, ket_excitations = _single_excitation_list(nocc, nmo_ket)
    occ_det = np.linalg.det(s_mo[np.ix_(occ_bra, occ_ket)])

    metric = np.zeros((len(bra_excitations), len(ket_excitations)), dtype=s_mo.dtype)
    for p, (_i, _a, bra_exc) in enumerate(bra_excitations):
        det_bra_exc_occ = np.linalg.det(s_mo[np.ix_(bra_exc, occ_ket)])
        for q, (_j, _b, ket_exc) in enumerate(ket_excitations):
            det_exc_exc = np.linalg.det(s_mo[np.ix_(bra_exc, ket_exc)])
            det_occ_ket_exc = np.linalg.det(s_mo[np.ix_(occ_bra, ket_exc)])
            metric[p, q] = occ_det * det_exc_exc + det_bra_exc_occ * det_occ_ket_exc
    return metric


def bse_wavefunction_overlap(
    bra,
    ket,
    bra_vectors,
    ket_vectors=None,
    ao_overlap=None,
    metric="auto",
):
    """Overlap between BSE/TDA singlet wavefunctions at different geometries.

    The state vectors are interpreted as coefficients in the spin-adapted
    single-excitation basis.  For full BSE vectors, the pseudo-overlap uses
    the standard ``X^dagger S X - Y^dagger S Y`` metric.
    """
    if ket_vectors is None:
        ket_vectors = bra_vectors
    if bra.nocc != ket.nocc:
        raise ValueError("BSE wavefunction overlap requires the same electron count.")

    s_mo = _bse_cross_mo_overlap(bra, ket, ao_overlap=ao_overlap)
    config_metric = _singlet_single_excitation_metric(s_mo, bra.nocc)
    bra_arr, bra_was_vector = _as_state_matrix(bra_vectors, "bra_vectors")
    ket_arr, ket_was_vector = _as_state_matrix(ket_vectors, "ket_vectors")

    dim_bra, dim_ket = config_metric.shape
    mode = str(metric).lower()
    if mode == "auto":
        if bra_arr.shape[0] == dim_bra and ket_arr.shape[0] == dim_ket:
            mode = "tda"
        elif bra_arr.shape[0] == 2 * dim_bra and ket_arr.shape[0] == 2 * dim_ket:
            mode = "full"
        else:
            raise ValueError("Cannot infer BSE overlap metric from vector dimensions.")

    if mode == "tda":
        if bra_arr.shape[0] != dim_bra or ket_arr.shape[0] != dim_ket:
            raise ValueError("TDA BSE overlap expects vectors of length nocc*nvir.")
        overlap = bra_arr.conj().T @ config_metric @ ket_arr
    elif mode in {"full", "xy", "symplectic"}:
        if bra_arr.shape[0] != 2 * dim_bra or ket_arr.shape[0] != 2 * dim_ket:
            raise ValueError("Full BSE overlap expects vectors of length 2*nocc*nvir.")
        xb, yb = bra_arr[:dim_bra], bra_arr[dim_bra:]
        xk, yk = ket_arr[:dim_ket], ket_arr[dim_ket:]
        overlap = xb.conj().T @ config_metric @ xk - yb.conj().T @ config_metric @ yk
    else:
        raise ValueError("metric must be 'auto', 'tda', or 'full'.")

    if bra_was_vector and ket_was_vector:
        return overlap[0, 0].item()
    return overlap


def bse(gw, using_tda=False, using_casida=True):
    r'''Get the RPA eigenvalues and eigenvectors.

    Q^\dagger = \sum_{ia} X_{ia} a^+ i - Y_{ia} i^+ a
    Leads to the RPA eigenvalue equations:
      [ A  B ][X] = omega [ 1  0 ][X]
      [ B  A ][Y]         [ 0 -1 ][Y]
    which is equivalent to
      [ A  B ][X] = omega [ 1  0 ][X]
      [-B -A ][Y] =       [ 0  1 ][Y]

    See, e.g. Stratmann, Scuseria, and Frisch,
              J. Chem. Phys., 109, 8218 (1998)
    '''

    A, B = bse_AB_matrices(gw)

    if using_tda:
        ham_rpa = A
        e, x = eig(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            return _casida_eigh(A, B)


def eig(h, s=None):
    e, c = scipy.linalg.eigh(h,s)
    return e, c


def eig_asymm(h):
    '''Diagonalize a real, *asymmetrix* matrix and return sorted results.

    Return the eigenvalues and eigenvectors (column matrix)
    sorted from lowest to highest eigenvalue.
    '''
    e, c = np.linalg.eig(h)
    if np.allclose(e.imag, 0*e.imag):
        e = np.real(e)
    else:
        print("WARNING: Eigenvalues are complex, will be returned as such.")

    idx = e.argsort()
    e = e[idx]
    c = c[:,idx]

    return e, c


# def is_positive_def(A):
#     try:
#         np.linalg.cholesky(A)
#         return True
#     except np.linalg.LinAlgError:
#         return False




def pes():

    mol = gto.Mole()
    mol.verbose = 2
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '3-21G'}
    # This is from G2/97 i.e. MP2/6-31G*

    R = np.linspace(0.2, 4, 10)
    ex_energy = np.zeros(len(R))

    f = open('excite_energy.dat','w')
    for i in range(len(R)):

        mol.atom = [['H' , (0.,  0., 0.)],
                    ['H' , (R[i], 0., 0.)]]
        mol.basis = '321g'
        mol.build()
        mf = scf.RHF(mol)

        mf.kernel()

        gw = BSE(mf, screening='TDHF')
        egw = gw.kernel()

        print('HF    vs.   GW ')
        for emf, eqp in zip(mf.mo_energy, egw):
            print("%0.6f %0.6f"%(emf, eqp))

        nocc = mol.nelectron//2
        ehomo = egw[nocc-1]
        print("GW -IP = GW HOMO =", ehomo, "au =", ehomo*27.211, "eV")


        #print('GW  spacial orbital energies (eV) = ', gw.e_qp*27.211)

        ex = bse(gw, using_tda=True, using_casida=False)[0]
        ex_energy[i] = ex[0]

        print("BSE Excitation energy =", ex ) #* au2ev)
        print("RPA eigenvalue = ", gw.e_rpa[0])
        f.write('{} {} {} \n'.format(R[i], ex_energy[i], gw.e_rpa[0]))

    f.close()

    return


if __name__ == '__main__':

    from pyscf import scf, gto
    from pyqed import au2ev

    mol = gto.Mole()
    mol.verbose = 2
    mol_name = 'H2'

#    if mol_name == 'H2':
##    #mol.atom = [['Ne' , (0., 0., 0.)]]
##    #mol.basis = {'Ne': '3-21G'}
##    # This is from G2/97 i.e. MP2/6-31G*
#        mol.atom = [['H' , (0.,  0., 0.)],
#                ['H' , (1.6, 0., 0.)]]
#    elif mol_name == 'benzene':
#        mol.atom = '''
#       C        0.70021        1.21284        0.00000
#       C        1.40045       -0.00002       -0.00000
#       C        0.70024       -1.21282        0.00000
#       C       -0.70024       -1.21282        0.00000
#       C       -1.40045       -0.00002        0.00000
#       C       -0.70021        1.21284       -0.00000
#       H       -1.24315        2.15309        0.00000
#       H        1.24315        2.15309        0.00000
#       H        2.48621        0.00006        0.00000
#       H        1.24305       -2.15315       -0.00000
#       H       -1.24305       -2.15315        0.00000
#       H       -2.48621        0.00006        0.00000
#       '''
#    else:
#        raise NotImplementedError

    # mol.atom = [
    #     [8 , (0. , 0.     , 0.)],
    #     [1 , (0. , -0.757 , 0.587)],
    #     [1 , (0. , 0.757  , 0.587)]]
    
    #!/usr/bin/env python


    mol = gto.M(
        atom = 'H 0 0 0; F 0 0 1.1',
        basis = '631g')
    
    nocc = mol.nelectron//2
    
    # By default, GW is done with analytic continuation
    # gw = gw.GW(mf)
    # same as gw = gw.GW(mf, freq_int='ac')
    # gw.kernel(orbs=range(nocc-3,nocc+3))

    # mol.basis = 'cc-pvdz'
#    mol.atom = '''
#    H   0.000000   0.934473    -0.588078
#    H   0.000000   -0.934473   -0.588078
#    C   0.000000   0.000000    0.000000
#    O   0.000000   0.000000    1.221104
#    '''
    # mol.basis = 'sto3g'
    mol.build()

    mf = scf.RHF(mol)
    #print(mf.scf())
    mf.kernel()

    gw = BSE(mf, screening='TDH')
    egw = gw.kernel()
    print('HF    vs.   GW ')
    for emf, eqp in zip(mf.mo_energy, egw):
        print("%0.6f %0.6f"%(emf, eqp))

    nocc = mol.nelectron//2
    ehomo = egw[nocc-1]
    elumo = egw[nocc]
    print("GW -IP = GW HOMO =", ehomo, "au =", ehomo*27.211, "eV")
    print("GW EA = GW LUMO =", elumo, "au =", elumo*27.211, "eV")

#
#
    # print('GW  spacial orbital energies (eV) = ', gw.e_qp*27.211)

    #excite = bse(gw, using_tda=True, using_casida=False)
    #print("BSE Excitation energy =", excite[0] ) #* au2ev)

    # pes()

    import matplotlib.pyplot as plt

    R,E, e_rpa = np.genfromtxt('excite_energy.dat', dtype=float, unpack=True)
    #print(R,E)
    #print(R, e_rpa)
    # R /= 0.529177
    plt.plot(R, E)
    # plt.plot(R, e_rpa)
