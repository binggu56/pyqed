
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
from scipy.linalg import eigh as eig
from pyqed.phys import eig_asymm
import sys

from functools import reduce

from pyqed import is_positive_def
from pyqed.qchem.hf.rhf import _cross_ao_overlap_matrix
from pyqed.gw.screening import FactorizedCouplings, StaticScreening

try:
    from pyqed.gw import _bse_contractions
except ImportError:
    _bse_contractions = None


def _ensure_bse_screening(gw):
    if getattr(gw, 'screening_energies', 'hf') == 'qp':
        energy = gw.e_mf if gw.e_qp is None else np.asarray(gw.e_qp)
        if not np.array_equal(energy, getattr(gw, '_screening_qp', None)):
            gw._static_screening = gw._M = gw.e_rpa = None
            full_energy = gw._screening_reference_energy.copy()
            if energy.shape == full_energy.shape:
                full_energy[:] = energy
            else:
                full_energy[gw.orbital_indices] = energy
            if not np.all(np.isfinite(full_energy)):
                raise ValueError("QP-gap BSE screening requires all GW energies; use screening_energies='hf' explicitly for partial GW windows")
            if np.any(full_energy[gw.full_nocc:, None] <= full_energy[None, :gw.full_nocc]):
                raise ValueError('BSE screening requires positive occupied-virtual gaps')
            if gw._screening_factors is not None and gw.screening == 'TDH':
                from types import SimpleNamespace
                from .contour import ContourDeformation
                reference = SimpleNamespace(screening='TDH', _pair_factors=gw._screening_factors,
                    mo_coeff=gw.mo_coeff, nocc=2*gw.full_nocc, eta=0., threads=1, max_memory=gw.max_memory)
                response = ContourDeformation(reference, full_energy, [0], nw=8, prepare=False)
                chol = scipy.linalg.cho_factor(response.dielectric(0., imaginary=True), lower=True, check_finite=False)
                gw._static_screening = StaticScreening(gw._pair_factors, chol, gw.max_memory)
            else:
                previous = gw.e_mf
                try:
                    gw.e_mf = energy
                    e_rpa, t_rpa = gw.rpa(method=gw.screening)
                    gw._M = gw.get_m_rpa(e_rpa, t_rpa)
                finally:
                    gw.e_mf = previous
            gw._screening_qp = energy.copy()
    if getattr(gw, '_static_screening', None) is not None:
        return
    if gw.e_rpa is None or gw._M is None:
        e_rpa, t_rpa = gw.rpa(method=gw.screening)
        gw._M = gw.get_m_rpa(e_rpa, t_rpa)


def _bse_screening(gw):
    static = getattr(gw, '_static_screening', None)
    return static if static is not None else gw._M


def _static_b_action(screening, x, nocc):
    occupied_virtual = screening.static_ov(nocc)
    out = np.zeros_like(x, dtype=np.result_type(x, screening.dtype))
    for first in range(0, len(screening.factors), 64):
        s = slice(first, first + 64)
        projected = np.einsum("Pib,jbk->Pijk", occupied_virtual[s], x, optimize=True)
        out += 4 * np.einsum(
            "Paj,Pijk->iak",
            screening.factors[s, nocc:, :nocc],
            projected,
            optimize=True,
        )
    return out


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

    if getattr(mf, "eri", None) is not None:
        eri_ao = np.asarray(mf.eri)
    elif getattr(mf.mol, "eri", None) is not None:
        eri_ao = np.asarray(mf.mol.eri)
    else:
        eri_ao = None

    if eri_ao is not None:
        return np.einsum(
            "mnkl,mp,nq,kr,ls->pqrs",
            eri_ao,
            mo_coeff.conj(),
            mo_coeff,
            mo_coeff.conj(),
            mo_coeff,
            optimize=True,
        )

    if ao2mofn is None:
        raise ValueError("BSE dense ERI build requires native dense AO ERIs or factorized ERIs.")
    nmo = len(mf.mo_energy)
    return ao2mofn(mf.mol, (mo_coeff, mo_coeff, mo_coeff, mo_coeff),
                   compact=False).reshape(nmo, nmo, nmo, nmo)


def _get_ao_eri_factors(mf):
    eri_factors = getattr(mf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(mf.mol, 'eri_factors', None)
    if eri_factors is None:
        return None
    return eri_factors


def _get_mo_pair_factors(mf, mo_coeff):
    eri_factors = _get_ao_eri_factors(mf)
    if eri_factors is None:
        return None
    from pyqed.qchem.basis import mo_pair_factors
    return mo_pair_factors(eri_factors, mo_coeff)


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


def _rebuild_scan_mol(mol):
    mol.build(options=getattr(mol, "builtin_options", None))
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
    """Restricted static-screened BSE, with full A/B or TDA response.

    Explicit HF-gap screening can reuse the matching contour GW reference's
    zero-frequency auxiliary dielectric factor. This is an exact reformulation of the static kernel,
    not a dynamical BSE approximation; see :class:`StaticScreening` and
    Bruneval, Hamed and Neaton, J. Chem. Phys. 142, 244101 (2015),
    doi:10.1063/1.4922489.
    By default, static direct-RPA screening uses GW quasiparticle gaps,
    matching PySCF 2.14 BSE (auxiliary formulation doi:10.1002/jcc.24688).
    This adaptation retains independent physical residual, metric and
    reference-stability checks. ``screening_energies='hf'`` explicitly uses
    reference gaps and permits reusing GW screening. QP screening requires
    finite QPs for the complete screening space, even with an orbital window.
    Default full BSE requests ten roots, 100 cycles and a 12*nroots subspace.
    ``tol=1e-4`` bounds physical XY residuals and response residual norms
    (the latter corresponds to PySCF's squared threshold 1e-8).

    ``orbital_window=(occupied, virtual)`` projects only the electron-hole
    excitation space onto frontier orbitals, retaining the reference's full
    static screening. This is an orbital-window approximation to molecular
    static BSE, not a reproduction of a converged complete-space calculation;
    enlarge the window to converge energies and state characters. See also
    Bruneval, Hamed and Neaton, J. Chem. Phys. 142, 244101 (2015),
    doi:10.1063/1.4922489. Matching factorized contour or spectral GW is required.

    ``threads`` controls auxiliary-tile OpenMP in factorized full-BSE and
    TDA actions (default one), independently of GW's thread setting. This
    exactly reassociates the static RI kernel of Bruneval, J. Chem. Phys.
    136, 194107 (2012), doi:10.1063/1.4718428; it adds no screening-pole or
    excitation-space truncation. Real double-precision factors and matching
    auxiliary static screening are required for multiple threads. Complex
    trials batch their real and imaginary components in the same action. Workers use
    private bounded scratch and fixed-order tile reduction, without
    transition matrices. Large matrix products use single-threaded SciPy BLAS
    within each OpenMP worker; tiny products retain the row-loop kernel.
    Full response batches X and Y together in each A/B factor product;
    this reuses factors through wider BLAS products without changing the
    operator. Final physical residual checks use the same column batching.
    Exactly zero trial components are skipped. One-entry Davidson seeds
    use direct RI columns in the same OpenMP team. This changes no screening
    or BSE approximation. ``max_memory`` can reduce observed parallelism; at least one trial batch must fit even below that budget.
    ``threading_info`` records requested/observed workers and action calls.
    ``batch_columns='auto'`` defaults to a cap of eight trial columns,
    reduced to fit estimated scratch at the requested thread count.
    Auxiliary tiles supply parallelism independently of vector batch width.
    Explicit integers override the cap; None selects scalar callbacks.
    ``batch_columns`` and ``info['batch_columns']`` store the selected width.
    This control does not parallelize screening preparation, projected
    eigensolvers, or dense reference solves. Existing iterative convergence
    and physical residual/metric gates remain applicable.
    """
    def __init__(self, gw_or_mf, ao2mofn=None,
                 screening='TDH', eta=1e-2, orbital_window=None, *, threads=1,
                 screening_energies='qp'):

        if screening_energies not in ('qp', 'hf'):
            raise ValueError("screening_energies must be 'qp' or 'hf'")
        self.screening_energies = screening_energies

        if isinstance(threads, (bool, np.bool_)) or not isinstance(threads, (int, np.integer)) or threads < 1:
            raise ValueError('threads must be a positive integer')
        if threads > 1 and (_bse_contractions is None or not _bse_contractions.openmp_available()):
            raise RuntimeError('Build BSE contractions with OpenMP to use threads > 1')
        self.threads = int(threads)
        self.threading_info = None

        gw_ref = gw_or_mf if _is_gw_reference(gw_or_mf) else None
        if gw_ref is not None and gw_ref.e_qp is None:
            raise ValueError("Run GW before constructing BSE; use the SCF reference for direct BSE")
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
        self.full_nocc = self.nocc
        self.orbital_window = orbital_window
        nmo = np.asarray(mf.mo_coeff).shape[1]
        self.orbital_indices = np.arange(nmo)
        if orbital_window is not None:
            if gw_ref is None:
                raise ValueError('An orbital window requires a GW reference with full screening')
            if len(orbital_window) != 2 or any(
                isinstance(v, (bool, np.bool_)) or int(v) != v or v < 1
                for v in orbital_window
            ):
                raise ValueError('orbital_window requires positive occupied and virtual counts')
            no, nv = map(int, orbital_window)
            if no > self.nocc or nv > nmo-self.nocc:
                raise ValueError('orbital_window exceeds the reference occupied/virtual space')
            self.orbital_window = (no, nv)
            self.orbital_indices = np.arange(self.nocc-no, self.nocc+nv)
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
            if not np.all(np.isfinite(np.asarray(gw_ref.e_qp)[self.orbital_indices])):
                if orbital_window is not None:
                    raise ValueError('BSE requires finite QP energies for every orbital in its window')
                raise ValueError('BSE requires QP energies for all spatial orbitals; run GW without an orbs subset')
            self.e_qp = np.array(gw_ref.e_qp, copy=True)
        self.e_rpa = None
        self._static_screening = None
        contour = getattr(gw_ref, '_ac_screening', None) or getattr(gw_ref, '_contour', None)
        if (screening_energies == 'hf' and screening == 'TDH' and contour is not None
                and contour.factors is self._pair_factors
                and np.array_equal(contour.energy, self.e_mf)
                and np.array_equal(contour.mo_coeff, self.mo_coeff)):
            self._static_screening = contour.static_screening
        charge = getattr(gw_ref, '_charge_screening', None)
        if (screening_energies == 'hf' and screening == 'TDH' and charge is not None
                and np.array_equal(charge['energy'], self.e_mf)
                and np.array_equal(charge['mo_coeff'], self.mo_coeff)):
            self.e_rpa = charge['poles'].copy()
            self._M = charge['couplings']
        self._bse_tda_info = None
        self.excitation_energies = None
        self.e = None
        self.info = None
        self._xy = None
        self._x = None
        self._y = None
        self.bse_metric = None
        self._screening_mf_energy = self.e_mf.copy()
        self._screening_factors = self._pair_factors
        self._screening_reference_energy = np.asarray(self.e_mf if self.e_qp is None else self.e_qp).copy()
        self._screening_qp = None
        if screening_energies == 'qp':
            _ensure_bse_screening(self)
        if self.orbital_window is not None:
            if self._pair_factors is None or (self._static_screening is None
                                             and not isinstance(self._M, FactorizedCouplings)):
                raise NotImplementedError('Orbital windows require matching full factorized GW screening')
            first, stop = self.orbital_indices[0], self.orbital_indices[-1]+1
            self._pair_factors = self._pair_factors[:, first:stop, first:stop]
            if self._static_screening is not None:
                self._static_screening = StaticScreening(
                    self._pair_factors, self._static_screening.cholesky, self.max_memory)
            else:
                self._M = FactorizedCouplings(self._pair_factors, self._M.projection, self._M.poles)
            self.mo_coeff = self.mo_coeff[:, first:stop]
            self.e_mf = self.e_mf[first:stop]
            self.v_mf = self.v_mf[first:stop, first:stop]
            if self.e_qp is not None:
                self.e_qp = np.asarray(self.e_qp)[first:stop]
            self.nocc, nv = self.orbital_window
            self.nso = self.nocc+nv
            if self._screening_qp is not None:
                self._screening_qp = self._screening_qp[first:stop].copy()

    def save(self, path):
        """Export available results to NPZ (energies in Hartree, no pickle).

        Missing amplitudes are omitted. This is a results export; use
        :meth:`checkpoint` to save the static BSE operator for reruns.
        """
        import json
        from pathlib import Path
        from .checkpoint import json_value

        if self.e is None:
            raise ValueError('Run BSE before saving results')
        arrays = dict(excitation_energy=self.e, energy_unit=np.asarray('hartree'),
                      metric=np.asarray(self.bse_metric),
                      info_json=np.asarray(json.dumps(json_value(self.info))))
        for key in ('x', 'y', 'e_qp', 'orbital_indices'):
            value = getattr(self, key, None)
            if value is not None:
                arrays[key] = value
        path = Path(path)
        with path.open('wb') as stream:
            np.savez(stream, **arrays)
        return path

    def checkpoint(self, path):
        """Save a self-contained factorized static-screening BSE directory.

        Includes QPs, RI factors, static dielectric Cholesky factor and any
        prepared screening caches/results. Restarts the eigensolve, not its
        Davidson iteration history. Does not checkpoint RHF/GW or scanners.
        The destination must not exist; large arrays are stored as NPY files.
        """
        from .checkpoint import save_checkpoint
        return save_checkpoint(self, path)

    @classmethod
    def from_checkpoint(cls, path, *, threads=1, max_memory=None):
        """Restore a static BSE problem with memory-mapped arrays, without SCF/GW.

        Use ``run(...)`` to solve again with the desired solver settings.
        Thread count defaults to one for portability across machines.
        """
        from .checkpoint import load_checkpoint
        return load_checkpoint(cls, path, threads=threads, max_memory=max_memory)

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

    def get_m_rpa(self, e_rpa, t_rpa, *, factorized=True):
        return get_m_rpa(self, e_rpa, t_rpa, factorized=factorized)

    def rpa(self, using_tda=False, using_casida=True, method='TDH'):
        self.e_rpa, t = rpa(self, using_tda, using_casida, method)
        return self.e_rpa, t

    def rpa_AB_matrices(self, method='TDH'):
        return rpa_AB_matrices(self, method)

    def bse(self, using_tda=False, using_casida=True):
        return bse(self, using_tda, using_casida)

    def _ensure_screening(self):
        _ensure_bse_screening(self)

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
        batch_columns='auto',
    ):
        if not use_qp or self.e_qp is None:
            self.e_qp = self.e_mf.copy()
            if hasattr(self, "_screening_mf_energy"):
                self._screening_reference_energy = self._screening_mf_energy.copy()

        use_low_rank = (
            low_rank is True
            or (low_rank == 'auto' and getattr(self, '_pair_factors', None) is not None)
        )
        if use_low_rank:
            self.solve_tda(
                nroots=nroots,
                tol=tol,
                max_cycle=max_cycle,
                max_space=max_space,
                return_info=return_info,
                return_vectors=return_vectors,
                batch_columns=batch_columns,
            )
            return self

        if getattr(self, 'threads', 1) > 1:
            raise ValueError('BSE threads require the factorized iterative path; set low_rank=True')
        if batch_columns == 'auto':
            batch_columns = None
        if batch_columns is not None:
            raise ValueError('batch_columns requires the iterative TDA path; set low_rank=True')
        self._ensure_screening()
        e, vec = self.bse(using_tda=True, using_casida=False)
        e = e[:nroots]
        vec = vec[:, :nroots]
        self._store_excitations(e, vec, 'tda')
        self.info = {"solver": "dense_tda", "converged": True}
        return self

    def run(
        self,
        nroots=10,
        low_rank='auto',
        use_qp=True,
        tol=1e-4,
        max_cycle=100,
        return_info=False,
        return_vectors=False,
        eigensolver=None,
        max_space=None,
        batch_columns='auto',
    ):
        '''Run full BSE excitations and return ``self`` for a chainable API.

        Supply a completed ``GW(mf).run()`` calculation for quasiparticle
        energies, or an SCF reference for direct BSE with mean-field gaps.
        This method computes excitation energies and stores them in ``.e``.  Full BSE amplitudes are stored in
        ``.xy`` with ``.x``/``.y`` views.  Use :class:`TDA` for the
        Tamm-Dancoff approximation. The matrix-free path defaults to
        real response Davidson; select ``eigensolver="davidson"`` for the
        general nonsymmetric solver or ``eigensolver="arpack"`` for Arnoldi. ``max_space``
        and ``batch_columns`` are documented in :func:`solve_bse`, together
        with method references and limitations.
        '''
        if eigensolver not in (None, "arpack", "davidson", "response", "lanczos"):
            raise ValueError("eigensolver must be response, lanczos, arpack or davidson")
        if not use_qp or self.e_qp is None:
            self.e_qp = self.e_mf.copy()
            if hasattr(self, "_screening_mf_energy"):
                self._screening_reference_energy = self._screening_mf_energy.copy()

        use_low_rank = (
            low_rank is True
            or (low_rank == 'auto' and getattr(self, '_pair_factors', None) is not None)
        )
        if use_low_rank:
            self.solve_bse(
                nroots=nroots,
                tol=tol,
                max_cycle=max_cycle,
                return_info=return_info,
                return_vectors=return_vectors,
                eigensolver="response" if eigensolver is None else eigensolver,
                max_space=max_space, batch_columns=batch_columns,
            )
            return self

        if getattr(self, 'threads', 1) > 1:
            raise ValueError('BSE threads require the factorized iterative path; set low_rank=True')

        if batch_columns == 'auto':
            batch_columns = None

        if eigensolver is not None or max_space is not None or batch_columns is not None:
            raise ValueError("Iterative full BSE options require low_rank=True")
        self._ensure_screening()
        A, B = bse_AB_matrices(self)
        e, vec = _full_bse_vectors_from_casida(A, B, nroots)
        self._store_excitations(e, vec, 'full')
        self.info = {"solver": "dense_full_bse", "converged": True}
        return self

    def solve_bse(
        self,
        nroots=10,
        tol=1e-4,
        max_cycle=100,
        return_info=False,
        return_vectors=True,
        eigensolver="response",
        max_space=None,
        batch_columns='auto',
    ):
        return solve_bse(
            self,
            nroots=nroots,
            tol=tol,
            max_cycle=max_cycle,
            return_info=return_info,
            return_vectors=return_vectors,
            eigensolver=eigensolver, max_space=max_space, batch_columns=batch_columns,
        )

    def wavefunction_overlap(
        self,
        other,
        *,
        ao_overlap=None,
        metric='auto',
    ):
        bra_vectors = self.excitation_vectors
        ket_vectors = self.excitation_vectors if other is self else other.excitation_vectors
        if bra_vectors is None:
            raise ValueError("No bra BSE/TDA vectors stored. Run with return_vectors=True first.")
        if ket_vectors is None:
            raise ValueError("No ket BSE/TDA vectors stored. Run with return_vectors=True first.")
        return bse_wavefunction_overlap(
            self,
            other,
            bra_vectors,
            ket_vectors=ket_vectors,
            ao_overlap=ao_overlap,
            metric=metric,
        )


    def solve_tda(
        self,
        nroots=10,
        tol=1e-4,
        max_cycle=100,
        max_space=None,
        return_info=False,
        return_vectors=True,
        batch_columns='auto',
    ):
        return solve_tda(
            self,
            nroots=nroots,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            return_info=return_info,
            return_vectors=return_vectors,
            batch_columns=batch_columns,
        )

    def as_scanner(
        self,
        nroots=None,
        energy="pes",
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
            gw_method=gw_method,
            gw_kwargs=gw_kwargs,
            mf_kwargs=mf_kwargs,
            run_kwargs=run_kwargs,
            return_object=return_object,
            solver_cls=self.__class__,
        )


    def clear_derivative_cache(self):
        """Release the geometry-local adjoint workspace and its disk files."""
        from .adjoint_workspace import clear
        clear(self)
        self._first_response_cache = None

    def derivatives(self, state_ids=None, modes=None, *, order=2, **kwargs):
        """Return a gradient (order=1) or (gradient, Hessian) (order=2).

        Uses the methods, units and restrictions documented by gradient()
        and hessian(). The second-order path computes both arrays in one
        Hessian evaluation, without a separate gradient call. Existing
        compatible intermediates may be reused; derivative caches and their
        temporary files are released on return or failure. Output arrays
        remain available as energy_gradient and energy_hessian. Selected
        Hessian columns retain the gradient along all requested modes.
        """
        try:
            if isinstance(order, (bool, np.bool_)) or order not in (1, 2):
                raise ValueError('Derivative order must be 1 or 2.')
            if order == 1:
                return self.gradient(state_ids, modes, **kwargs)
            hessian = self.hessian(state_ids, modes, **kwargs)
            return self.energy_gradient, hessian
        finally:
            self.clear_derivative_cache()

    def gradient(self, state_ids=None, modes=None, *, method="adjoint", **kwargs):
        """Return relaxed total-energy gradients, shape (nstates, nmodes).

        States index excitations from zero; energies are RHF plus excitation.
        Modes are Cartesian dR/dQ in bohr. Without modes the final axis is
        flattened Cartesian (3*natom). Results are stored in energy_gradient.
        Uses the analytic response adaptation of Tölle, Kitsaras and
        Loos, JPCL 16, 11134–11143 (2025), doi:10.1021/acs.jpclett.5c02219.
        Exact spectral RHF-G0W0 and full orbital spaces are required. See
        vibronic_couplings for reference fidelity and response restrictions.
        Individual surface derivatives are NaN at electronic degeneracies.
        The default method="adjoint" uses state-specific Z-vectors (optional PyTorch),
        currently requiring nondegenerate HF orbitals; see gw.adjoint for
        fidelity and limits. Geometry-local spectra, Z-vectors, orbital solver
        and raw first integral derivatives are retained for an adjoint Hessian
        with matching response settings. Directional electronic responses are
        computed only by hessian(), never by gradient(). Reference/state or
        response-setting changes invalidate the workspace; clear_derivative_cache()
        releases it explicitly. Saved graph tensors use temporary disk storage;
        storage="memory" retains them in RAM.
        Explicit method="forward" computes only first derivatives.
        Integral, RHF, GW and BSE first-order
        responses are cached for subsequent gradient/hessian calls. Content checks
        invalidate the cache when the reference or requested directions change.
        Above the memory budget, directions are processed individually and
        reusable first responses are stored in temporary disk files.
        """
        if method == 'adjoint':
            from .adjoint import gradients
            return gradients(self, state_ids, modes, **kwargs)
        if method != 'forward':
            raise ValueError("Gradient method must be forward or adjoint.")
        from .response import energy_derivatives
        return energy_derivatives(self, state_ids, modes, order=1, **kwargs)[0]

    def hessian(self, state_ids=None, modes=None, *, method='adjoint', **kwargs):
        """Return relaxed total-energy Hessians, shape (nstates, nmodes, nmodes).

        Includes response through the complete excitation space. These are
        surface curvatures, distinct from diagonal local-Hamiltonian G.
        See gradient for units, method reference, and restrictions. Results
        are stored in energy_hessian (energy_gradient is also computed).
        The default method='adjoint' analytically differentiates the Z-vector gradient,
        reusing first RHF response and mixed integral curvature; no second
        electronic response tensors or finite differences. This extends the
        gradient adaptation of Tölle, Kitsaras and Loos (2025),
        doi:10.1021/acs.jpclett.5c02219; see gw.adjoint.hessians for fidelity
        and restrictions. Saved tensors default to storage='disk'.
        excitation_response='projected' uses the existing matrix-free BSE
        actions and projected GMRES for eigenvector tangents; 'dense' remains
        the default reference. GW screening and square-root derivatives remain
        dense. excitation_response='physical' uses the stationary XY form
        and exact static dielectric contractions to remove BSE square-root
        derivatives; dynamic GW remains dense. See gw.adjoint.hessians for
        references, fidelity, residual and degeneracy checks.
        gw_response='spectral' replaces nested dynamic-GW root/inverse
        derivatives with full-pole divided differences and one Sylvester
        pullback. It retains the dense reference spectrum and response
        contractions; it is not a matrix-free or truncated-pole method.
        For adjoint response, columns=[...] selects direction indices and
        returns shape (nstates, nmodes, nselected), retaining all rows.
        For memory-bounded requests, method='polarization' uses summed
        directional curvatures. method='mixed' propagates one direct mixed
        second derivative from two disk-backed first responses and retains
        Hessian-only first frequency responses after diagonal passes. Both
        keep all mixed entries without finite differences. Small requests use
        the full analytic response tensor with either selection.
        """
        if method == "adjoint":
            from .adjoint import hessians
            return hessians(self, state_ids, modes, **kwargs)
        from .response import energy_derivatives
        return energy_derivatives(self, state_ids, modes, hessian_method=method, **kwargs)[1]

    def vibronic_couplings(self, state_ids=None, modes=None, frame='relaxed',
                           *, backend='auto', max_memory=None, degeneracy_tol=1e-7):
        r"""Analytic relaxed local Hamiltonian derivatives F and G.

        Forward first/second differentiation of RHF CPHF, charge RPA,
        zero-broadening diagonal G0W0 Dyson roots and static singlet BSE.
        This adapts the fully relaxed gradient formulation of Tölle,
        Kitsaras and Loos, JPCL 16, 11134–11143 (2025),
        doi:10.1021/acs.jpclett.5c02219, using forward response instead of
        their Lagrangian. Integral response reuses PyQED's analytic Gaussian
        and fixed-rank RI/CD implementations. No finite differences.

        Builtin real closed-shell RHF, exact spectral G0W0, TDH screening
        and complete orbital spaces are required. Degenerate orbital blocks
        use a parallel subspace gauge; repeated screening poles use analytic
        matrix square roots/resolvents. ``degeneracy_tol`` groups reference
        orbital energies in Hartree (default 1e-7). Occupied/virtual subspaces
        must remain separated by a gap. RI requires an unscreened
        Cholesky metric; CD retains fixed pivots/rank. Response is dense in
        transition space, with an explicit memory gate; factors stay factored.

        Inside a degenerate block, full self-energy response replaces the
        basis-dependent diagonal-only derivative. The symmetry-protected QP
        root and its frequency residue must be scalar within that block.
        Second-order Hermitian Dyson partitioning adapts Löwdin, JCP 19,
        1396 (1951), doi:10.1063/1.1748067; matrix-function response adapts
        Higham and Relton, SIMAX 35, 1019 (2014), doi:10.1137/130945259.
        Couplings between distinct reference orbital blocks are omitted.
        This is a block extension of diagonal G0W0, not full matrix GW.

        ``frame='relaxed'`` freezes reference exciton amplitudes in the
        smoothly continued canonical excitation frame (TDA), or the
        positive-frequency Hermitian Casida frame (full BSE). These are
        effective-Hamiltonian derivatives, not fixed real-space many-electron
        matrix elements or exact nonadiabatic couplings. The moving-frame
        transverse kinetic connection is not included. Use in RSH/CGLDR is
        the explicitly Q-independent-link approximation documented there.

        F has shape (nstates,nstates,nmodes), G has shape
        (nstates,nstates,nmodes,nmodes); without modes the trailing axes are
        (natom,3) and (natom,3,natom,3). Modes are dR/dQ in bohr.
        State indices are zero-based EXCITATIONS; no ground state is included.
        Total energies use E_RHF + excitation energy, with the corresponding
        common relaxed RHF derivatives. For isolated states diagonal F equals surface gradients;
        diagonal G is not a surface Hessian. ``self.energy_gradient`` and
        ``self.energy_hessian`` store mode-space surface derivatives, including
        response through ALL excitation states. Surface gradients/Hessians are NaN at electronic
        degeneracies (F/G remain defined); ``derivative_info`` records this.
        ``subspace_gradient``/``subspace_hessian`` instead differentiate the
        retained energy trace, including omitted-state response; these are
        defined for complete multiplets separated from the discarded states.
        F supplies LVC coefficients, G supplies QVC coefficients in this
        effective frame. Keep entire electronic multiplets in a vibronic model.
        """
        from .response import vibronic_couplings
        return vibronic_couplings(self, state_ids, modes, frame,
                                  backend=backend, max_memory=max_memory,
                                  degeneracy_tol=degeneracy_tol)

    @property
    def e_tot(self):
        """RHF reference plus neutral excitation energies (excited roots only)."""
        if self.e is None:
            return None
        return float(self._scf.e_tot) + np.asarray(self.e)


class TDA(BSE):
    """Tamm-Dancoff approximation to BSE.

    This class is a convenience wrapper around :class:`BSE` that always solves
    the TDA eigenproblem and uses the TDA wavefunction-overlap metric by
    default.
    """

    def run(
        self,
        nroots=10,
        low_rank='auto',
        use_qp=True,
        tol=1e-4,
        max_cycle=100,
        max_space=None,
        return_info=False,
        return_vectors=False,
        batch_columns='auto',
    ):
        """Solve TDA; ``batch_columns`` enables batched iterative actions.

        With factorized integrals, ``low_rank='auto'`` selects this path.
        Otherwise set ``low_rank=True`` explicitly; dense solves reject batching.
        """
        return self._run_tda(
            nroots=nroots,
            low_rank=low_rank,
            use_qp=use_qp,
            tol=tol,
            max_cycle=max_cycle,
            max_space=max_space,
            return_info=return_info,
            return_vectors=return_vectors,
            batch_columns=batch_columns,
        )

    def wavefunction_overlap(
        self,
        other,
        *,
        ao_overlap=None,
        metric='tda',
    ):
        return super().wavefunction_overlap(
            other,
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
            return _rebuild_scan_mol(mol)

        mol = mol_or_geom
        if getattr(mol, "hcore", None) is None or (
            getattr(mol, "eri", None) is None and getattr(mol, "eri_factors", None) is None
        ):
            return _rebuild_scan_mol(mol)
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

        freq_int = getattr(self.gw, "freq_int", "ac") if self.gw is not None else "ac"
        gw = GW(mf, screening=self.base.screening, eta=self.base.eta, freq_int=freq_int,
                threads=getattr(self.gw, 'threads', 1), vhf_df=getattr(self.gw, 'vhf_df', False))
        gw.run(method=self.gw_method, **self.gw_kwargs)
        return gw

    def __call__(self, mol_or_geom):
        mol = self._prepare_mol(mol_or_geom)
        mf = self._run_mf(mol)
        gw = self._run_gw(mf)

        nroots = self.nroots
        if nroots is None:
            nroots = len(self.bse.e) if self.bse.e is not None else 10
        run_kwargs = dict(self.run_kwargs)
        run_kwargs.setdefault("nroots", nroots)

        bse = self.solver_cls(
            gw, screening=self.base.screening, eta=self.base.eta,
            orbital_window=getattr(self.base, 'orbital_window', None),
            threads=getattr(self.base, 'threads', 1),
            screening_energies=self.base.screening_energies,
        ).run(**run_kwargs)

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


def get_m_rpa(gw, e_rpa, t_rpa, *, factorized=True):
    r'''
    Get the (intermediate) M_{pq,L} tensor.
    The M (or w) is needed to construct the screened Coulomb interaction W
    
    .. math::
        
        M_{pq,L} = \sum_{ia} ( (eps_a-eps_i)/erpa_L )^{1/2} T_{ai,L} (ai|pq)

    RI/CD returns exact factorized couplings by default, reusing the
    static RI reassociation of Bruneval, JCP 136, 194107 (2012),
    doi:10.1063/1.4718428. No poles or auxiliary factors are discarded.
    ``factorized=False`` materializes couplings for explicit references.
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
        from .gw import _screening_storage
        if factorized:
            projection = _screening_storage(gw, weighted_pairs.shape, weighted_pairs.dtype)
            projection[:] = weighted_pairs
            return FactorizedCouplings(gw._pair_factors, projection, e_rpa)
        M = _screening_storage(gw, (nso, nso, len(e_rpa)), gw._pair_factors.dtype)
        for p in range(nso):
            M[p] = gw._pair_factors[:, p, :].T @ weighted_pairs
    else:
        M = np.einsum('a,al,apq->pql', sqrt_eps, t_by_e, eri_product, optimize=True)
    return M



def bse_AB_matrices(gw):
    '''Get the BSE A and B matrices, using the quasiparticle energies computed
    from GW, and screened interaction W computed from RPA
    '''
    method = gw.screening
    assert method in ('TDH','TDHF','TDDFT')
    _ensure_bse_screening(gw)
    static = getattr(gw, '_static_screening', None)
    if static is not None:
        occupied = static.static_occupied(gw.nocc)
        occupied_virtual = static.static_ov(gw.nocc)

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
                    if static is not None:
                        A[ia, jb] += 4*np.dot(occupied[:,i,j], static.factors[:,a,b])
                        B[ia, jb] += 4*np.dot(occupied_virtual[:,i,b-nocc], static.factors[:,a,j])
                    else:
                        for L in range(len(gw.e_rpa)):
                            # The Casida convention gives induced screening +4 M M / w.
                            A[ia, jb] += 4.*gw._M[i,j,L] * gw._M[a,b,L]/gw.e_rpa[L]
                            B[ia, jb] += 4.*gw._M[i,b,L] * gw._M[a,j,L]/gw.e_rpa[L]

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
    return _bse_tda_matmat(gw, np.asarray(x).reshape(-1, 1), 1)[:, 0]


def _factorized_bse_action(gw, vectors, batch_columns, kind):
    """Stream matching real RI/static factors without transition matrices."""
    threads = getattr(gw, 'threads', 1)
    factors, screened = getattr(gw, '_pair_factors', None), _bse_screening(gw)
    supported = (factors is not None and factors.dtype == np.float64
                 and isinstance(screened, (FactorizedCouplings, StaticScreening))
                 and screened.factors is factors and screened.dtype == np.float64
                 and factors.strides[2] == factors.itemsize
                 and factors.strides[1] >= factors.shape[2]*factors.itemsize
                 and factors.strides[0] >= factors.shape[1]*factors.strides[1])
    if _bse_contractions is None or not supported:
        if threads > 1:
            raise NotImplementedError('BSE threads require compiled contractions and matching real factorized static screening with positive row-contiguous factors')
        return None
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1, user_api='blas'):
        occupied = screened.static_occupied(gw.nocc)
        ov = screened.static_ov(gw.nocc) if kind != 'a' else None
    no, nv = gw.nocc, gw.nso-gw.nocc
    if ov is None:
        # A ignores the ov data; a broadcast view avoids preparing B screening.
        ov = np.broadcast_to(np.zeros((1, no, nv)), (len(factors), no, nv))
    out = np.empty(vectors.shape, dtype=np.result_type(vectors, float))
    info = getattr(gw, 'threading_info', None)
    if (info is None or info.get('algorithm') != 'auxiliary-tile BSE contractions'
            or info.get('requested_threads') != threads):
        info = dict(algorithm='auxiliary-tile BSE contractions', openmp=_bse_contractions.openmp_available(),
                    requested_threads=threads, actual_threads=0, action_calls=0)
        gw.threading_info = info
    for first in range(0, vectors.shape[1], batch_columns):
        block = vectors[:, first:first+batch_columns]
        columns = block.shape[1]
        complex_block = np.iscomplexobj(block) and np.any(block.imag)
        work_columns = columns*(2 if complex_block else 1)
        products, buffers = (4, 18) if kind == 'full' else (2, 11)
        scratch = 8*(work_columns*(products*no*max(no, nv)+buffers*no*nv)+no*max(no, nv))
        workers = max(1, min(threads, int(max(1e-6, getattr(gw, 'max_memory', 4000))*1e6/scratch)))
        def apply(values):
            with threadpool_limits(limits=1, user_api='blas'):
                result, used, tile = _bse_contractions.action(factors, occupied, ov,
                    np.ascontiguousarray(values, dtype=float), no, kind, workers)
            info.update(actual_threads=max(info['actual_threads'], used), auxiliary_tile=tile)
            info['action_calls'] += 1
            return result
        if complex_block:
            value = apply(np.concatenate((block.real, block.imag), axis=1))
            value = value[:, :columns]+1j*value[:, columns:]
        else:
            value = apply(block.real)
        out[:, first:first+columns] = value
    diagonal = _bse_tda_diag(gw)[:, None]
    if kind == 'a':
        out += diagonal*vectors
    elif kind == 'full':
        dim = no*nv
        out[:dim] += diagonal*vectors[:dim]
        out[dim:] -= diagonal*vectors[dim:]
    return out


def _bse_tda_matmat(gw, vectors, batch_columns=8):
    """Exact column-batched version of the existing molecular TDA action.

    No change to its static screened kernel or excitation space. Explicit
    two-stage contractions avoid forming a transition-space matrix; scratch
    scales with the factor count times occupied/virtual sizes times the batch.
    Auxiliary factors and screening poles are blocked as well as trial
    columns, avoiding full virtual-virtual pole weighting and quartic
    contraction intermediates. The scalar action uses this same routine.
    Reused factorized charge screening contracts the exact auxiliary static
    metric, without materializing orbital-pair/pole couplings.
    """
    vectors = np.asarray(vectors)
    nocc = gw.nocc
    nvir = gw.nso - nocc
    if vectors.ndim != 2 or vectors.shape[0] != nocc * nvir or batch_columns < 1:
        raise ValueError("Invalid TDA trial block or batch size")
    _ensure_bse_screening(gw)
    compiled = _factorized_bse_action(gw, vectors, batch_columns, 'a')
    if compiled is not None:
        return compiled
    screened = _bse_screening(gw)
    factors = getattr(gw, "_pair_factors", None)
    out = np.empty(vectors.shape, dtype=np.result_type(vectors, screened.dtype, float))
    diagonal = _bse_tda_diag(gw).reshape(nocc, nvir, 1)
    for start in range(0, vectors.shape[1], batch_columns):
        stop = min(start + batch_columns, vectors.shape[1])
        x = vectors[:, start:stop].reshape(nocc, nvir, -1)
        y = diagonal * x
        if factors is not None:
            for first in range(0, len(factors), 64):
                f = factors[first : first + 64]
                vo = f[:, nocc:, :nocc]
                projected = np.einsum("Pbj,jbk->Pk", vo, x, optimize=True)
                y += 2 * np.einsum("Pai,Pk->iak", vo, projected, optimize=True)
                projected = np.einsum(
                    "Pij,jbk->Pibk", f[:, :nocc, :nocc], x, optimize=True
                )
                y -= np.einsum(
                    "Pab,Pibk->iak", f[:, nocc:, nocc:], projected, optimize=True
                )
        else:
            y += 2 * np.einsum(
                "aibj,jbk->iak", gw.eri[nocc:, :nocc, nocc:, :nocc], x, optimize=True
            )
            y -= np.einsum(
                "abij,jbk->iak", gw.eri[nocc:, nocc:, :nocc, :nocc], x, optimize=True
            )
        if isinstance(screened, (FactorizedCouplings, StaticScreening)):
            occupied = screened.static_occupied(nocc)
            for first in range(0, len(screened.factors), 64):
                s = slice(first, first + 64)
                projected = np.einsum("Pij,jbk->Pibk", occupied[s], x, optimize=True)
                y += 4 * np.einsum(
                    "Pab,Pibk->iak",
                    screened.factors[s, nocc:, nocc:],
                    projected,
                    optimize=True,
                )
        else:
            for first in range(0, len(gw.e_rpa), 64):
                s = slice(first, first + 64)
                projected = np.einsum(
                    "ijL,jbk->Libk", gw._M[:nocc, :nocc, s], x, optimize=True
                )
                projected /= gw.e_rpa[s, None, None, None]
                y += 4 * np.einsum(
                    "abL,Libk->iak", gw._M[nocc:, nocc:, s], projected, optimize=True
                )
        out[:, start:stop] = y.reshape(nocc * nvir, -1)
    return out


def _bse_b_matmat(gw, vectors, batch_columns=8):
    """Apply the B block with bounded auxiliary/pole and trial-vector batches."""
    vectors = np.asarray(vectors)
    nocc, nvir = gw.nocc, gw.nso - gw.nocc
    dim = nocc * nvir
    if vectors.ndim != 2 or vectors.shape[0] != dim or batch_columns < 1:
        raise ValueError("Invalid BSE B trial block or batch size")
    _ensure_bse_screening(gw)
    compiled = _factorized_bse_action(gw, vectors, batch_columns, 'b')
    if compiled is not None:
        return compiled
    screened = _bse_screening(gw)
    factors = getattr(gw, "_pair_factors", None)
    out = np.empty(vectors.shape, dtype=np.result_type(vectors, screened.dtype, float))
    for start in range(0, vectors.shape[1], batch_columns):
        stop = min(start + batch_columns, vectors.shape[1])
        x = vectors[:, start:stop].reshape(nocc, nvir, -1)
        if factors is None:
            y = 2 * np.einsum(
                "aijb,jbk->iak", gw.eri[nocc:, :nocc, :nocc, nocc:], x, optimize=True
            )
            y -= np.einsum(
                "ajib,jbk->iak", gw.eri[nocc:, :nocc, :nocc, nocc:], x, optimize=True
            )
        else:
            y = np.zeros_like(x, dtype=out.dtype)
            for first in range(0, len(factors), 64):
                f = factors[first : first + 64]
                projected = np.einsum(
                    "Pjb,jbk->Pk", f[:, :nocc, nocc:], x, optimize=True
                )
                y += 2 * np.einsum(
                    "Pai,Pk->iak", f[:, nocc:, :nocc], projected, optimize=True
                )
                projected = np.einsum(
                    "Pib,jbk->Pijk", f[:, :nocc, nocc:], x, optimize=True
                )
                y -= np.einsum(
                    "Paj,Pijk->iak", f[:, nocc:, :nocc], projected, optimize=True
                )
        if isinstance(screened, (StaticScreening, FactorizedCouplings)):
            y += _static_b_action(screened, x, nocc)
        else:
            for first in range(0, len(gw.e_rpa), 64):
                poles = slice(first, first + 64)
                projected = np.einsum(
                    "ibL,jbk->Lijk", screened[:nocc, nocc:, poles], x, optimize=True
                )
                projected /= gw.e_rpa[poles, None, None, None]
                y += 4 * np.einsum(
                    "ajL,Lijk->iak",
                    screened[nocc:, :nocc, poles],
                    projected,
                    optimize=True,
                )
        out[:, start:stop] = y.reshape(dim, -1)
    return out


def _bse_b_matvec(gw, vector):
    return _bse_b_matmat(gw, np.asarray(vector).reshape(-1, 1), 1)[:, 0]


def _bse_full_matvec(gw, vector):
    return _bse_full_matmat(gw, np.asarray(vector).reshape(-1, 1), 1)[:, 0]


def _bse_full_matmat(gw, vectors, batch_columns=8):
    """Apply [A B; -B -A], retaining only one trial batch of each block."""
    vectors = np.asarray(vectors)
    dim = gw.nocc * (gw.nso - gw.nocc)
    if vectors.ndim != 2 or vectors.shape[0] != 2 * dim or batch_columns < 1:
        raise ValueError("Invalid full BSE block or batch size")
    _ensure_bse_screening(gw)
    compiled = _factorized_bse_action(gw, vectors, batch_columns, 'full')
    if compiled is not None:
        return compiled
    out = np.empty(
        vectors.shape, dtype=np.result_type(vectors, _bse_screening(gw).dtype, float)
    )
    x, y = vectors[:dim], vectors[dim:]
    for start in range(0, vectors.shape[1], batch_columns):
        columns = slice(start, start + batch_columns)
        for a_source, b_source, target, sign in (
            (x, y, out[:dim], 1),
            (y, x, out[dim:], -1),
        ):
            action = _bse_tda_matmat(gw, a_source[:, columns], batch_columns)
            action += _bse_b_matmat(gw, b_source[:, columns], batch_columns)
            target[:, columns] = sign * action
    return out


def _resolve_bse_batch(gw, batch_columns, eigensolver='davidson', full=True):
    if batch_columns == 'auto':
        if eigensolver == 'arpack':
            batch_columns = None
        else:
            no, nv = gw.nocc, gw.nso-gw.nocc
            if no < 1 or nv < 1:
                raise ValueError('BSE requires a nonempty excitation space')
            fixed = 8*no*max(no, nv)
            products, buffers = (4, 18) if full else (2, 11)
            per_column = 8*(products*no*max(no, nv)+buffers*no*nv)*(2 if full else 1)
            budget = max(1e-6, getattr(gw, 'max_memory', 4000))*1e6/max(1, getattr(gw, 'threads', 1))-fixed
            batch_columns = max(1, min(8, int(budget/per_column)))
    if batch_columns is not None:
        if (isinstance(batch_columns, (bool, np.bool_)) or int(batch_columns) != batch_columns
                or batch_columns < 1):
            raise ValueError('batch_columns must be auto, a positive integer, or None')
        batch_columns = int(batch_columns)
    gw.batch_columns = batch_columns
    return batch_columns


def solve_tda(
    gw,
    nroots=10,
    tol=1e-4,
    max_cycle=100,
    max_space=None,
    return_info=False,
    return_vectors=True,
    batch_columns='auto',
):
    '''Lowest TDA-BSE roots from a factorized/matrix-free Davidson solve.

    ``batch_columns='auto'`` uses a memory-limited cap of eight columns,
    accounting for requested threads without equating columns with threads.
    Explicit positive integers override it; None retains scalar callbacks.
    This changes contraction order only, not screening, orbital gaps or the
    Davidson formulation. The public Davidson docstring documents its method
    references and adaptation limits. No universal speedup is guaranteed.
    '''
    from pyqed.davidson import davidson

    batch_columns = _resolve_bse_batch(gw, batch_columns, full=False)

    diag = _bse_tda_diag(gw)
    nroots = min(int(nroots), diag.size)
    if nroots < 1:
        raise ValueError("nroots must be positive.")

    _ensure_bse_screening(gw)

    eigvals, eigvecs, info = davidson(
        lambda vec: _bse_tda_matvec(gw, vec),
        roots=nroots,
        tolerance=tol,
        iterations=max_cycle,
        diag=diag,
        space=max_space,
        return_info=True,
        return_partial=True,
        matmat=None if batch_columns is None else lambda block: _bse_tda_matmat(gw,block,batch_columns),
        dtype=np.result_type(diag, _bse_screening(gw).dtype),
    )
    gw._bse_tda_info = info
    info['batch_columns'] = batch_columns
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


def _response_davidson(action, diagonal, nroots, tol, max_cycle, max_space, memory):
    """Real common-subspace Davidson for stable full response.

    Adaptation of the projected (A+B)/(A-B) response formulation of
    Stratmann, Scuseria and Frisch, JCP 109, 8218 (1998),
    doi:10.1063/1.477483. Uses the existing Casida reconstruction, gap
    preconditioning and thick restarts. Requires positive projected A-B
    and A+B; this is not a global stability certificate. No TDA or
    excitation-space truncation is introduced. Physical XY residuals
    determine convergence, rather than squared-frequency residuals.
    """
    dim = len(diagonal)
    space = min(dim, 12*nroots if max_space is None else int(max_space))
    if space < nroots or (space == nroots and space < dim):
        raise ValueError('Response subspace must exceed the requested roots')
    estimate = 8*(5*dim*space+16*dim*nroots+12*space*space)
    if estimate > memory:
        raise MemoryError('Response workspace exceeds memory limit')
    basis = np.zeros((dim, nroots))
    basis[np.argsort(diagonal)[:nroots], np.arange(nroots)] = 1.
    av = np.empty((dim, 0)); bv = av.copy()
    info = dict(iterations=0, restarts=0, operator_columns=0,
                estimated_workspace_bytes=estimate, history=[])

    def orthogonalize(vectors, against):
        for _ in range(2):
            vectors = vectors-against@(against.T@vectors)
        q, r, _ = scipy.linalg.qr(vectors, mode='economic', pivoting=True)
        keep = np.abs(np.diag(r)) > 1e-11
        return q[:, keep]

    for cycle in range(max_cycle):
        pending = basis[:, av.shape[1]:]
        applied = action(np.vstack((pending, np.zeros_like(pending))))
        if np.iscomplexobj(applied):
            if np.any(applied.imag):
                raise ValueError('Response solver requires a real operator')
            applied = applied.real
        av = np.column_stack((av, applied[:dim]))
        bv = np.column_stack((bv, -applied[dim:]))
        info['operator_columns'] += pending.shape[1]
        a, b = basis.T@av, basis.T@bv
        for matrix in (a, b):
            if not np.allclose(matrix, matrix.T, atol=1e-10, rtol=1e-10):
                raise RuntimeError('Response solver requires real symmetric A and B')
        a, b = _symmetrize(a), _symmetrize(b)
        if scipy.linalg.eigvalsh(a+b, subset_by_index=[0, 0])[0] <= 0:
            raise RuntimeError('Response A+B is not positive definite; unstable reference')
        roots, xy = _full_bse_vectors_from_casida(a, b, nroots)
        size = basis.shape[1]
        cx, cy = xy[:size], xy[size:]
        x, y = basis@cx, basis@cy
        rx = av@cx+bv@cy-x*roots
        ry = -bv@cx-av@cy-y*roots
        residual = np.sqrt(np.sum(rx*rx+ry*ry, axis=0))
        info.update(iterations=cycle+1, residual_norms=residual, subspace=size)
        info['history'].append(dict(iteration=cycle+1, subspace=size,
                                    residual_norms=residual.tolist(), energies=roots.tolist()))
        response_residual = np.maximum(np.linalg.norm(rx+ry, axis=0), np.linalg.norm(rx-ry, axis=0))
        info["response_residual_squared"] = response_residual**2
        if np.max(residual) <= .8*tol and np.max(response_residual) <= tol:
            return roots, np.vstack((x, y)), info
        active = (residual > .8*tol) | (response_residual > tol)
        # Residuals in P=X+Y and Q=X-Y share the orbital-gap preconditioner.
        denominator = roots[active][None, :]-diagonal[:, None]
        denominator = np.where(abs(denominator)<1e-8,
                               np.where(denominator<0, -1e-8, 1e-8), denominator)
        corrections = np.column_stack(((rx[:, active]-ry[:, active])/denominator,
                                       (rx[:, active]+ry[:, active])/denominator))
        extra = orthogonalize(corrections, basis)
        if not extra.shape[1]:
            raise RuntimeError('Response Davidson stagnated before convergence')
        if size+extra.shape[1] > space:
            restart = orthogonalize(np.column_stack((x, y)), np.empty((dim, 0)))
            if restart.shape[1] >= space:
                restart = restart[:, :max(nroots, space//2)]
            basis = restart
            av = np.empty((dim, 0)); bv = av.copy()
            info['restarts'] += 1
            extra = orthogonalize(corrections, basis)
        basis = np.column_stack((basis, extra[:, :space-basis.shape[1]]))
    raise RuntimeError('Response Davidson did not converge')


def solve_bse(
    gw, nroots=10, tol=1e-4, max_cycle=100, return_info=False,
    return_vectors=True, eigensolver="response", max_space=None, batch_columns='auto',
):
    """Lowest positive molecular full-BSE roots from a matrix-free action.

    ``eigensolver="response"`` (default) uses real common-subspace Davidson
    for A+B and A-B, following Stratmann, Scuseria and Frisch, J. Chem.
    Phys. 109, 8218–8224 (1998), doi:10.1063/1.477483. This adaptation uses
    the existing Casida reconstruction, gap preconditioning and thick restarts.
    Positive projected A+B and A-B are required; their positivity does not
    certify global reference stability. Full XY residuals determine convergence.
    This changes no screening kernel or excitation space. See
    :func:`_response_davidson` for implementation details.

    ``eigensolver="arpack"`` uses SciPy Arnoldi; ``"davidson"``
    uses :func:`pyqed.linalg.davidson_nonsymmetric` with positive-real harmonic
    Ritz selection and orbital-gap diagonal preconditioning. This adapts
    Davidson, J. Comput. Phys. 17, 87–94 (1975),
    doi:10.1016/0021-9991(75)90065-0, and Morgan, Linear Algebra Appl. 154–156,
    289–309 (1991), doi:10.1016/0024-3795(91)90381-6. It is not a
    structure-preserving BSE solver, stability test, or guarantee that all
    lowest positive roots have been found. Screening and the static BSE
    kernel are unchanged. Only the existing real-orbital molecular operator
    is supported. Returned vectors must have positive BSE metric and pass
    absolute residual checks after metric orthonormalization. Degenerate
    complex eigenvector mixtures are converted to an independent real basis
    using SVD of their real and imaginary parts; the physical residual gate
    is applied after that conversion. No partial
    solutions or silent solver fallback are accepted.

    ``eigensolver="lanczos"`` uses the unrestarted, fully reorthogonalized
    metric block Lanczos adaptation of Shao et al., SIAM J. Matrix Anal.
    Appl. 39, 683–711 (2018), doi:10.1137/16M1102641. See
    :func:`pyqed.gw.bse_lanczos.lanczos` for restrictions and fidelity.

    ``max_space`` controls the stored iterative space or ARPACK ncv. ``batch_columns``
    defaults to 'auto': a cap of eight reduced by estimated per-thread
    scratch and max_memory. Explicit integers override this policy; None
    selects scalar callbacks. ARPACK and dense paths use scalar/no batching
    for 'auto'. The selected width is recorded in info and the solver.
    """
    from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigs

    if eigensolver not in ("arpack", "davidson", "response", "lanczos"):
        raise ValueError("eigensolver must be response, lanczos, arpack or davidson")
    if not np.isfinite(tol) or tol <= 0 or int(max_cycle) != max_cycle or max_cycle < 1:
        raise ValueError("Require positive tolerance and integer max_cycle")
    batch_columns = _resolve_bse_batch(gw, batch_columns, eigensolver)
    if batch_columns is not None and eigensolver not in ("davidson", "response", "lanczos"):
        raise ValueError("batch_columns requires eigensolver='davidson' or 'response'")
    dim = gw.nocc * (gw.nso - gw.nocc)
    if isinstance(nroots, bool) or int(nroots) != nroots or nroots < 1 or dim < 1:
        raise ValueError("nroots must be positive and the excitation space nonempty")
    nroots = min(int(nroots), dim)
    if max_space is not None and (
        isinstance(max_space, bool) or int(max_space) != max_space or max_space < 1
    ):
        raise ValueError("max_space must be a positive integer")
    _ensure_bse_screening(gw)

    info = dict(solver="low_rank_full_bse", eigensolver=eigensolver,
                converged=False, nroots=nroots, batch_columns=batch_columns)
    gw.info = gw._bse_full_info = info
    calls = 0
    def action(vec):
        nonlocal calls
        calls += 1
        return _bse_full_matvec(gw, vec)
    def block_action(vectors):
        nonlocal calls
        calls += vectors.shape[1]
        return _bse_full_matmat(gw, vectors, int(batch_columns))
    try:
        if eigensolver in ("response", "lanczos"):
            from threadpoolctl import threadpool_limits
            def response_action(vectors):
                return (block_action(vectors) if batch_columns is not None else
                        np.column_stack([action(v) for v in vectors.T]))
            # Tall, narrow basis updates and small projected solves benefit
            # from serial BLAS; factor contractions retain their OpenMP team.
            with threadpool_limits(limits=1, user_api='blas'):
                method = _response_davidson
                extra = {}
                if eigensolver == "lanczos":
                    from .bse_lanczos import lanczos
                    method = lanczos
                    extra["info"] = info.setdefault("lanczos", {})
                eigvals, eigvecs, diagnostics = method(
                    response_action, _bse_tda_diag(gw), nroots, tol, max_cycle, max_space,
                    int(max(1e-6, getattr(gw, 'max_memory', 4000))*1e6), **extra,
                )
            info[eigensolver] = diagnostics
        elif eigensolver == "davidson":
            from pyqed.linalg import davidson_nonsymmetric
            diagonal = _bse_tda_diag(gw)
            eigvals, eigvecs, diagnostics = davidson_nonsymmetric(
                action, nroots, diag=np.r_[diagonal, -diagonal],
                matmat=block_action if batch_columns is not None else None,
                selection="positive_real", tolerance=tol, iterations=max_cycle,
                space=max_space,
                memory_limit=int(max(1e-6, getattr(gw, 'max_memory', 4000))*1e6),
            )
            info["davidson"] = diagnostics
        else:
            k = min(max(2*nroots+4, nroots+2), 2*dim-2)
            if k < 1:
                raise ValueError("ARPACK full BSE needs a larger space; use davidson or dense BSE")
            info["arnoldi_roots"] = k
            operator = LinearOperator((2*dim, 2*dim), matvec=action, dtype=float)
            eigvals, eigvecs = eigs(operator, k=k, which='SM', tol=tol,
                                    maxiter=max_cycle, ncv=max_space)

        eigvals = np.asarray(eigvals)
        eigvecs = np.asarray(eigvecs)
        real_mask = np.abs(eigvals.imag) < max(1e-8, 100*tol)
        pos = np.where(real_mask & (eigvals.real > 0.0))[0]
        order = pos[np.argsort(eigvals.real[pos])][:nroots]
        if len(order) != nroots:
            raise RuntimeError(f"Full BSE found only {len(order)} positive real roots; requested {nroots}")
        roots = eigvals.real[order]
        vectors = np.array(eigvecs[:, order], dtype=complex, copy=True)
        # A repeated real eigenvalue can have arbitrary complex mixtures,
        # not just per-vector phases. Recover a real basis within each cluster.
        real_vectors = np.empty(vectors.shape, dtype=float)
        cluster_tol = max(.01*tol, 100*np.finfo(float).eps*max(1., max(abs(roots))))
        start = 0
        while start < nroots:
            stop = start+1
            while stop < nroots and roots[stop]-roots[start] <= cluster_tol:
                stop += 1
            block = vectors[:, start:stop]
            u, singular, _ = np.linalg.svd(np.column_stack((block.real, block.imag)), full_matrices=False)
            count = stop-start
            if len(singular) < count or singular[count-1] <= 1e-12*singular[0]:
                raise RuntimeError("Full BSE selected vectors lack an independent real basis")
            real_vectors[:, start:stop] = u[:, :count]
            start = stop
        vectors = real_vectors
        metric = np.r_[np.ones(dim), -np.ones(dim)]
        gram = vectors.T @ (metric[:, None]*vectors)
        if np.linalg.eigvalsh((gram+gram.T)/2)[0] <= 0:
            raise RuntimeError("Positive BSE roots have nonpositive metric; possible unstable reference")
        vectors = _metric_orthonormalize_full_bse_vectors(vectors, dim)
        applied = (block_action(vectors) if batch_columns is not None else
                   np.column_stack([action(v) for v in vectors.T]))
        residuals = np.linalg.norm(applied-vectors*roots, axis=0)
        metric_error = float(np.linalg.norm(vectors.T@(metric[:, None]*vectors)-np.eye(nroots), ord=np.inf))
        info.update(residual_norms=residuals, metric_error=metric_error, operator_columns=calls)
        if not np.all(np.isfinite(residuals)) or np.max(residuals) > tol or metric_error > 1e-8:
            raise RuntimeError("Full BSE failed residual/metric checks after normalization")
        info["converged"] = True
    except (RuntimeError, ValueError, np.linalg.LinAlgError, ArpackNoConvergence) as exc:
        info.update(message=str(exc), operator_columns=calls)
        raise
    if hasattr(gw, '_store_excitations'):
        gw._store_excitations(roots, vectors, 'full')
    else:
        gw.excitation_energies = roots
        gw.excitation_vectors = vectors
        gw.xy = vectors
        gw.XY = vectors
        gw.bse_metric = 'full'
    if not return_vectors:
        return (roots, info) if return_info else roots
    return (roots, vectors, info) if return_info else (roots, vectors)


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
    if not callable(overlap):
        raise ValueError("AO overlap was not supplied and cannot be inferred.")
    return np.asarray(overlap(), dtype=float)


def _cross_geometry_ao_overlap(bra, ket):
    try:
        return _cross_ao_overlap_matrix(bra.mol, ket.mol)
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
