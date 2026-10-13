"""Auxiliary-basis contour-deformation molecular G0W0."""

from collections import OrderedDict
from functools import lru_cache

import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres

from .screening import StaticScreening

try:
    from . import _contour_contractions
except ImportError:
    _contour_contractions = None


class ContourDeformation:
    """Restricted direct-RPA contour integration with real CD/RI factors.

    Molecular adaptation of Zhu and Chan, JCTC 17, 727–741 (2021),
    doi:10.1021/acs.jctc.0c00704: imaginary-axis quadrature plus residues,
    without analytic continuation or screening-pole truncation. A static
    subtraction integrates the orbital-pole discontinuity analytically.
    Only diagonal G0W0 is supported, with a gapped, real restricted reference.
    Finite quadrature and the real-frequency eta regulator must be converged;
    agreement with the spectral solver is in the eta -> 0 limit. This is not
    a reproduction of the periodic, unrestricted, or self-consistent methods.
    Dielectric matrices are built one frequency at a time in auxiliary space.
    Real RI contractions are split into real and imaginary parts; residue
    solves use complex symmetry (not Hermitian symmetry). Transition blocking
    changes evaluation order only, without dropping transitions or poles.
    Optional real-frequency matrix-free screening uses SciPy restarted GMRES
    (Saad and Schultz, SIAM J. Sci. Stat. Comput. 7, 856–869 (1986),
    doi:10.1137/0907058), adapted with the static dielectric Cholesky factor
    as a left preconditioner. All transitions remain; unconverged solves
    raise rather than switching to a different approximation. Direct solves
    also check the dielectric residual, scaled by ||epsilon||_F ||x||_2
    + ||rhs||_2, against screening_tol. This backward-error check tolerates
    cancellation at independent-particle poles; it does not bound forward
    error. The unscaled RHS-relative residual is also recorded.
    A small linear
    residual alone does not guarantee self-energy accuracy near a pole.
    The zero-frequency Cholesky factor is retained for exact reuse by static
    full BSE and TDA; no screening eigenvectors are required for those kernels.
    Nearby residue solves reuse a converged vector only as a GMRES initial
    guess; every solution still passes the same unpreconditioned residual gate.
    Contiguous auxiliary blocks combine real and imaginary contractions.
    ``residue_solver="lanczos"`` instead reuses inverse-Casida response
    chains with separately controlled value/derivative estimates; see
    :class:`pyqed.gw.lanczos_response.LanczosResponse` for its references,
    adaptation, convergence controls and finite-precision limitations.

    ``GW(..., freq_int='contour_deformation', threads=N)`` uses OpenMP
    output tiles for dielectric construction, matrix-free screening actions
    and pair projections, plus compiled quadrature. This is an algebraic
    evaluation change to the above adaptation, with all transitions retained
    and the same regulator, root continuation and residual gates. Native
    tiles have fixed summation order; complex matrices are mirrored without
    conjugation. Serial large products retain NumPy BLAS; small tasks use
    one worker. Workers call no BLAS. SciPy Cholesky, direct solves,
    preconditioning, GMRES/Lanczos control and orbital root continuation
    retain their existing implementations and thread controls. max_memory
    bounds estimated contraction scratch, not total RSS or mandatory direct
    auxiliary matrices; a minimum serial workspace remains. threading_info
    records requested and observed teams per kernel. Parallel contractions
    require row-contiguous real double-precision transition factors. No
    universal scaling or new quadrature/accuracy guarantee follows.
    """

    def __init__(self, gw, energy, orbs, nw=64, quadrature_scale=0.5,
                 residue_solver="direct", screening_tol=1e-11,
                 screening_maxiter=1000, screening_restart=128, prepare=True,
                 target_factors=None, transition_factors=None):
        from pyqed.gw.gw import _screening_storage

        if gw.screening != "TDH" or gw._pair_factors is None:
            raise NotImplementedError(
                "Contour GW requires TDH screening and CD/RI factors"
            )
        if isinstance(nw, bool) or int(nw) != nw or nw < 8:
            raise ValueError("nw must be an integer of at least 8")
        if not np.isfinite(quadrature_scale) or quadrature_scale <= 0:
            raise ValueError("quadrature_scale must be positive and finite")
        if not np.isfinite(gw.eta) or gw.eta < 0:
            raise ValueError("eta must be finite and nonnegative")
        if residue_solver not in ("direct", "gmres", "lanczos"):
            raise ValueError("residue_solver must be direct, gmres or lanczos")
        if not np.isfinite(screening_tol) or not 0 < screening_tol < 1:
            raise ValueError("screening_tol must be between zero and one")
        for value in (screening_maxiter, screening_restart):
            if isinstance(value, bool) or int(value) != value or value < 1:
                raise ValueError("Screening iteration limits must be positive integers")
        self.residue_solver, self.screening_tol = residue_solver, screening_tol
        self.threads = getattr(gw, 'threads', 1)
        self.max_memory = gw.max_memory
        if self.threads > 1 and (_contour_contractions is None or not _contour_contractions.openmp_available()):
            raise RuntimeError('Build contour contractions with OpenMP to use threads > 1')
        self.screening_maxiter, self.screening_restart = int(screening_maxiter), int(screening_restart)
        self.screening_info = dict(solver=residue_solver, solves=0, iterations=0,
                                   max_iterations=0, max_relative_residual=0., failed_solves=0)
        self.response_cache_bytes = max(1024*1024, int(gw.max_memory*1e6/8))
        self.energy = np.asarray(energy, dtype=float).copy()
        self.mo_coeff = gw.mo_coeff.copy()
        self.nocc = gw.nocc // 2
        self.factors = gw._pair_factors
        if np.iscomplexobj(self.factors):
            raise NotImplementedError("Contour GW requires real orbital factors")
        self.gaps = (self.energy[self.nocc :] - self.energy[: self.nocc, None]).ravel()
        if (
            not len(self.gaps)
            or np.any(self.gaps <= 0)
            or not np.all(np.isfinite(self.energy))
        ):
            raise ValueError(
                "Contour GW requires finite energies and positive occupied-virtual gaps"
            )
        if transition_factors is not None and (
            np.shape(transition_factors) != (len(self.factors), len(self.gaps))
            or np.iscomplexobj(transition_factors)
        ):
            raise ValueError('Transition factors must retain every occupied-virtual pair')
        self.pairs = (self.factors[:, : self.nocc, self.nocc :].reshape(len(self.factors), -1)
                      if transition_factors is None else transition_factors)
        self.orbs = np.asarray(orbs)
        if (
            self.orbs.ndim != 1
            or not len(self.orbs)
            or self.orbs.dtype.kind not in "iu"
            or len(np.unique(self.orbs)) != len(self.orbs)
            or np.any(self.orbs < 0)
            or np.any(self.orbs >= len(self.energy))
        ):
            raise ValueError("orbs must contain distinct spatial orbital indices")
        self.columns = {int(p): j for j, p in enumerate(self.orbs)}
        self.target_factors = target_factors
        if target_factors is not None and (
            np.shape(target_factors) != (len(self.factors), len(self.orbs), len(self.energy))
            or np.iscomplexobj(target_factors)
        ):
            raise ValueError('Target factors must retain every intermediate MO for each selected orbital')
        self.eta = gw.eta
        self.quadrature_scale = float(quadrature_scale)
        self.block = max(
            1, min(4096, int(max(1e-6, gw.max_memory) * 1e6 / (64 * len(self.factors))))
        )
        nodes, weights = np.polynomial.legendre.leggauss(int(nw))
        self.freqs = quadrature_scale * (1 + nodes) / (1 - nodes)
        self.weights = weights * 2 * quadrature_scale / (1 - nodes) ** 2
        self.w0 = np.empty((len(self.orbs), len(self.energy)))
        self.w_imag = _screening_storage(gw, (*self.w0.shape, int(nw)), float)
        self.prepared = False
        self._prepared_frequencies = set()
        if not prepare:
            return
        for k, frequency in enumerate(np.r_[0.0, self.freqs]):
            chol = la.cho_factor(
                self.dielectric(frequency, imaginary=True),
                lower=True,
                overwrite_a=True,
                check_finite=False,
            )
            elements = self.screened_elements(chol, self.orbs)
            for j, p in enumerate(self.orbs):
                values = elements[j]
                if k == 0:
                    self.w0[j] = values
                else:
                    self.w_imag[j, :, k - 1] = values
            if k == 0:
                self.static_screening = StaticScreening(self.factors, chol, gw.max_memory)
            self._prepared_frequencies.add(k)
        self.prepared = True

    def screened_elements(self, cholesky, orbs):
        """Diagonal pair matrix elements for selected targets and every intermediate."""
        elements = np.empty((len(orbs), len(self.energy)))
        for j, p in enumerate(orbs):
            targets = getattr(self, 'target_factors', None)
            # Real MO pair factors are symmetric; rows avoid strided disk reads.
            rhs = self.factors[:, p, :] if targets is None else targets[:, self.columns[int(p)], :]
            screened = la.cho_solve(cholesky, rhs, check_finite=False)-rhs
            elements[j] = np.sum(rhs*screened, axis=0)
        return elements

    def set_screening_frequency(self, index, elements, cholesky=None, max_memory=4000):
        """Collect an independently computed quadrature frequency into prepared screening."""
        if (index < 0 or index > len(self.freqs) or index in self._prepared_frequencies
                or np.shape(elements) != self.w0.shape or not np.all(np.isfinite(elements))):
            raise ValueError('Invalid or duplicate prepared screening frequency')
        if index == 0:
            if cholesky is None or np.shape(cholesky[0]) != (len(self.factors),)*2:
                raise ValueError('Static screening requires its complete auxiliary Cholesky factor')
            self.static_screening = StaticScreening(self.factors, cholesky, max_memory)
            self.w0[:] = elements
        else:
            self.w_imag[:, :, index-1] = elements
        self._prepared_frequencies.add(index)
        self.prepared = len(self._prepared_frequencies) == len(self.freqs)+1

    def dielectric(self, frequency, *, imaginary=False):
        """Return 1-Pi with transition-blocked direct-RPA polarization."""
        z = frequency + 1j * self.eta
        weights = (
            -4 * self.gaps / (frequency**2 + self.gaps**2)
            if imaginary
            else 4 * self.gaps / (z * z - self.gaps**2)
        )
        compiled = self._use_contractions()
        workers = self._contraction_threads(matrix=True) if compiled else 1
        if compiled and workers > 1 and len(self.factors)**2*len(self.gaps) >= 262144:
            function = (_contour_contractions.imaginary_dielectric if imaginary
                        else _contour_contractions.dielectric)
            matrix, used = function(self.pairs, weights, workers)
            self._record_threads('dielectric', used)
            return matrix
        matrix = np.eye(len(self.factors), dtype=weights.dtype, order="F")
        for start in range(0, len(self.gaps), self.block):
            s = slice(start, start + self.block)
            pairs = self.pairs[:, s]
            if imaginary:
                # Imaginary-axis response is a positive symmetric rank update.
                # SYRK evaluates one triangle and avoids a second dense matrix.
                scaled = np.asarray(pairs * np.sqrt(-weights[s]), order='F')
                matrix = la.blas.dsyrk(1., scaled, beta=1., c=matrix,
                                       lower=1, overwrite_c=1)
            else:
                # RI factors are real: avoid promoting both matrix operands
                # to complex for every transition block.
                matrix.real -= (pairs * weights[s].real) @ pairs.T
                matrix.imag -= (pairs * weights[s].imag) @ pairs.T
        if imaginary:
            for column in range(len(matrix)):
                matrix[column, column+1:] = matrix[column+1:, column]
        if compiled:
            self._record_threads('dielectric_serial', 1)
        return matrix

    def _use_contractions(self):
        supported = (self.pairs.dtype == np.float64 and self.pairs.strides[1] == 8
                     and self.pairs.strides[0] >= self.pairs.shape[1]*8)
        if getattr(self, 'threads', 1) > 1 and not supported:
            raise NotImplementedError('Contour OpenMP requires row-contiguous real double-precision transition factors')
        return _contour_contractions is not None and supported

    def _contraction_threads(self, *, matrix=False):
        requested = getattr(self, 'threads', 1)
        if matrix:
            per_thread = 16*len(self.factors)*16+16*64*16
            shared = 64*len(self.factors)*8
        else:
            per_thread, shared = 4096, len(self.gaps)*16
        budget = max(1e-6, getattr(self, 'max_memory', 4000))*1e6-shared
        return max(1, min(requested, int(budget/per_thread)))

    def _record_threads(self, kind, used):
        if (getattr(self, 'threading_info', None) is None
                or self.threading_info['requested_threads'] != getattr(self, 'threads', 1)):
            self.threading_info = dict(algorithm='contour contractions',
                requested_threads=getattr(self, 'threads', 1), actual_threads=0,
                openmp=_contour_contractions.openmp_available(), kernels={}, calls={})
        info = self.threading_info
        info['actual_threads'] = max(info['actual_threads'], used)
        info['kernels'][kind] = max(info['kernels'].get(kind, 0), used)
        info['calls'][kind] = info['calls'].get(kind, 0)+1

    def _pair_projection(self, vector):
        compiled = self._use_contractions()
        workers = self._contraction_threads() if compiled else 1
        if compiled and (workers > 1 or self.pairs.size < 262144):
            value, used = _contour_contractions.project(self.pairs, vector, workers)
            self._record_threads('project', used)
            return value if np.iscomplexobj(vector) else value.real.copy()
        # Auxiliary-row blocks are contiguous even for disk-backed pairs.
        # Contract both components together to read each block only once.
        if not np.iscomplexobj(vector):
            projected = np.zeros(len(self.gaps))
            for first in range(0, len(self.factors), 128):
                s = slice(first, first+128)
                projected += self.pairs[s].T @ vector[s]
            if compiled:
                self._record_threads('project_serial', 1)
            return projected
        components = np.column_stack((vector.real, vector.imag))
        projection = np.zeros((len(self.gaps), 2))
        for first in range(0, len(self.factors), 128):
            s = slice(first, first+128)
            projection += self.pairs[s].T @ components[s]
        if compiled:
            self._record_threads('project_serial', 1)
        return projection[:, 0] + 1j*projection[:, 1]

    def _dielectric_action(self, weights, vector):
        compiled = self._use_contractions()
        workers = self._contraction_threads() if compiled else 1
        if compiled and (workers > 1 or self.pairs.size < 262144):
            value, used = _contour_contractions.action(self.pairs, weights, vector, workers)
            self._record_threads('action', used)
            return value
        projected = self._pair_projection(vector)
        weighted = weights * projected
        components = np.column_stack((weighted.real, weighted.imag))
        result = np.empty_like(vector, dtype=complex)
        for first in range(0, len(self.factors), 128):
            s = slice(first, first+128)
            back = self.pairs[s] @ components
            result[s] = vector[s] - (back[:, 0] + 1j*back[:, 1])
        if compiled:
            self._record_threads('action_serial', 1)
        return result

    def dielectric_action(self, frequency, vector):
        """Apply the complete real-frequency dielectric matrix without building it."""
        z = frequency + 1j*self.eta
        weights = 4*self.gaps/(z*z-self.gaps**2)
        return self._dielectric_action(weights, np.asarray(vector))

    def screening_solution(self, frequency, rhs, initial=None):
        """Solve epsilon(frequency) x = rhs with direct or residual-checked GMRES."""
        if self.residue_solver == 'lanczos':
            raise NotImplementedError('Lanczos response reuse evaluates residues, not auxiliary solutions')
        self.screening_info["solves"] += 1
        if self.residue_solver == "direct":
            matrix = self.dielectric(frequency)
            matrix_norm = np.linalg.norm(matrix)
            solution = la.solve(matrix, rhs, assume_a="sym",
                                overwrite_a=True, check_finite=False)
            residual = self.dielectric_action(frequency, solution)-rhs
            relative = np.linalg.norm(residual)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
            # At independent-particle poles, large epsilon causes cancellation.
            # Scale by the equation norm rather than the small RHS alone.
            backward = np.linalg.norm(residual)/max(
                matrix_norm*np.linalg.norm(solution)+np.linalg.norm(rhs), np.finfo(float).tiny)
            info = self.screening_info
            info['max_backward_error'] = max(info.get('max_backward_error', 0.), float(backward))
            info['max_relative_residual'] = max(info['max_relative_residual'], float(relative))
            if not np.all(np.isfinite(solution)) or not np.isfinite(backward) or backward > self.screening_tol:
                info['failed_solves'] += 1
                raise np.linalg.LinAlgError(
                    f"Screening direct solve failed at frequency {frequency:.12g}: "
                    f"scaled backward error={backward:.3g}, relative residual={relative:.3g}")
            return solution
        z = frequency + 1j*self.eta
        weights = 4*self.gaps/(z*z-self.gaps**2)
        action = lambda v: self._dielectric_action(weights, v)
        shape = (len(self.factors), len(self.factors))
        operator = LinearOperator(shape, matvec=action, dtype=complex)
        chol = self.static_screening.cholesky
        def precondition(v):
            return (la.cho_solve(chol, v.real, check_finite=False)
                    + 1j*la.cho_solve(chol, v.imag, check_finite=False))
        preconditioner = LinearOperator(shape, matvec=precondition, dtype=complex)
        iterations = 0
        def count(_):
            nonlocal iterations
            iterations += 1
        solution, status = gmres(
            operator, rhs, x0=initial, M=preconditioner, rtol=self.screening_tol, atol=0.,
            restart=min(self.screening_restart, len(rhs)), maxiter=self.screening_maxiter,
            callback=count, callback_type="legacy",
        )
        residual = np.linalg.norm(action(solution)-rhs)
        relative = residual/max(np.linalg.norm(rhs), np.finfo(float).tiny)
        info = self.screening_info
        info["iterations"] += iterations
        info["max_iterations"] = max(info["max_iterations"], iterations)
        info["max_relative_residual"] = max(info["max_relative_residual"], float(relative))
        if status or not np.all(np.isfinite(solution)) or relative > self.screening_tol:
            if initial is not None:
                # A nearby frequency can still cross a poorly conditioned pole.
                # Retry the original zero-start equation, with the same gate.
                info['warm_retries'] = info.get('warm_retries', 0)+1
                return self.screening_solution(frequency, rhs)
            info["failed_solves"] += 1
            raise np.linalg.LinAlgError(f"Screening GMRES failed at frequency {frequency:.12g}: "
                               f"status={status}, iterations={iterations}, relative residual={relative:.3g}")
        return solution

    def residue(self, frequency, orbital, intermediate):
        """Screened matrix element and its derivative at a positive real frequency."""
        rhs = self.factors[:, intermediate, orbital]
        if self.target_factors is not None and orbital in self.columns:
            rhs = self.target_factors[:, self.columns[orbital], intermediate]
        if self.residue_solver == 'lanczos':
            from .lanczos_response import LanczosResponse
            if not hasattr(self, '_lanczos_response'):
                self._lanczos_response = LanczosResponse(self)
            return self._lanczos_response.residue(frequency, (int(orbital), int(intermediate)), rhs)
        initial = None
        if self.residue_solver == "gmres":
            if not hasattr(self, '_residue_guesses'):
                self._residue_guesses = OrderedDict()
            key = (int(orbital), int(intermediate))
            previous = self._residue_guesses.get(key)
            if previous is not None and abs(frequency-previous[0]) <= max(1e-4, .1*frequency):
                initial = previous[1]
                self.screening_info['warm_starts'] = self.screening_info.get('warm_starts', 0)+1
        solution = self.screening_solution(frequency, rhs, initial=initial)
        if self.residue_solver == "gmres":
            self._residue_guesses[key] = (frequency, solution)
            self._residue_guesses.move_to_end(key)
            if len(self._residue_guesses) > 256:
                self._residue_guesses.popitem(last=False)
        value = rhs @ (solution - rhs)
        z = frequency + 1j * self.eta
        weights = -8 * z * self.gaps / (z * z - self.gaps**2) ** 2
        projected = self._pair_projection(solution)
        derivative = np.sum(weights * projected**2)
        return value, derivative

    def evaluate(self, orbital, omega):
        """Return diagonal correlation self-energy and its frequency derivative."""
        if not self.prepared:
            raise RuntimeError('Contour screening preparation is incomplete')
        j = self.columns[orbital]
        delta = float(omega) - self.energy
        if self._use_contractions():
            value, derivative, used = _contour_contractions.quadrature(self.energy,
                self.w0[j], self.w_imag[j], self.freqs, self.weights,
                self.nocc, float(omega), self._contraction_threads())
            self._record_threads('quadrature', used)
        else:
            value, derivative = self._quadrature(j, delta)
        occupation = np.arange(len(delta)) < self.nocc
        active = np.where(np.where(occupation, delta < 0, delta > 0))[0]
        for m in active:
            screened, slope = self.residue(abs(delta[m]), orbital, m)
            value += (-1 if occupation[m] else 1) * (screened - self.w0[j, m])
            derivative += slope
        return value, derivative

    def _quadrature(self, j, delta):
        difference = self.w_imag[j] - self.w0[j, :, None]
        denominator = delta[:, None] ** 2 + self.freqs**2
        occupation = np.arange(len(delta)) < self.nocc
        value = np.sum((0.5 - occupation) * self.w0[j])
        value -= (
            np.sum(difference * self.weights * delta[:, None] / denominator) / np.pi
        )
        derivative = (
            -np.sum(
                difference
                * self.weights
                * (self.freqs**2 - delta[:, None] ** 2)
                / denominator**2
            )
            / np.pi
        )
        return value, derivative


def kernel(
    gw, mo_energy, mo_coeff, *, orbs=None, nw=64, quadrature_scale=0.5, tol=1e-9,
    residue_solver="direct", screening_tol=1e-11, screening_maxiter=1000,
    screening_restart=128, reuse_screening=False,
):
    """Solve diagonal restricted G0W0 by :class:`ContourDeformation`.

    Unrequested spatial orbitals have NaN energies, weights and residuals.
    All intermediate occupied/virtual states remain in the screening and
    self-energy. QP roots use the spectral driver's continuation prescription.
    """
    from pyqed.gw.gw import _continue_qp_root, _set_rhf_orbitals, _sigma_x_diagonal

    if not np.isfinite(tol) or tol <= 0:
        raise ValueError("tol must be positive and finite")
    if not np.array_equal(mo_coeff, gw.mo_coeff):
        _set_rhf_orbitals(gw, mo_energy, mo_coeff)
    gw._M = gw._charge_screening = None
    gw._qp_energy_so = np.repeat(mo_energy, 2)
    orbs = list(range(len(mo_energy))) if orbs is None else list(orbs)
    if reuse_screening:
        contour = gw._contour
        if (contour is None or not contour.prepared or contour.factors is not gw._pair_factors
                or not np.array_equal(contour.energy, mo_energy)
                or not np.array_equal(contour.mo_coeff, mo_coeff)
                or contour.eta != gw.eta or len(contour.freqs) != nw
                or contour.quadrature_scale != quadrature_scale
                or contour.residue_solver != residue_solver
                or contour.screening_tol != screening_tol
                or contour.screening_maxiter != screening_maxiter
                or contour.screening_restart != screening_restart
                or not set(orbs).issubset(contour.columns)):
            raise ValueError('Prepared contour screening does not match the requested reference/controls/orbitals')
    else:
        gw._contour = None
        contour = ContourDeformation(gw, mo_energy, orbs, nw, quadrature_scale,
                                     residue_solver, screening_tol, screening_maxiter, screening_restart)
    gw._contour = contour
    contour.threads = getattr(gw, 'threads', 1)
    exchange = _sigma_x_diagonal(gw, orbs)
    previous = getattr(gw, 'e_qp', None) if reuse_screening else None
    energies = np.full(len(mo_energy), np.nan) if previous is None else np.array(previous, copy=True)
    if previous is None:
        gw.qp_weights = np.full_like(energies, np.nan)
        gw.qp_residuals = np.full_like(energies, np.nan)
    for index, p in enumerate(orbs):
        static = exchange[index] - gw.v_mf[2 * p, 2 * p]
        evaluate = lru_cache(maxsize=4)(lambda w: contour.evaluate(int(p), w))
        correction = lambda w: evaluate(float(w))[0] + static
        derivative = lambda w: evaluate(float(w))[1]
        energies[p] = _continue_qp_root(
            correction, derivative, float(mo_energy[p]), tol=tol
        )
        gw.qp_weights[p] = 1 / (1 - derivative(energies[p]).real)
        gw.qp_residuals[p] = abs(
            energies[p] - mo_energy[p] - correction(energies[p]).real
        )
    gw.orbital_indices = (np.asarray(orbs, dtype=int) if not reuse_screening else
                          np.array([p for p in contour.orbs if np.isfinite(energies[p])]))
    gw.contour_info = dict(
        nw=int(nw),
        quadrature_scale=quadrature_scale,
        orbs=gw.orbital_indices.tolist(),
        eta=gw.eta,
        screening="auxiliary direct RPA",
        static_subtraction=True,
        residue_solver=residue_solver,
        screening_tol=screening_tol,
        screening_maxiter=int(screening_maxiter),
        screening_restart=int(screening_restart),
        screening_info=contour.screening_info.copy(),
    )
    gw.threading_info = getattr(contour, 'threading_info', dict(
        algorithm='NumPy contour contractions', openmp=False, requested_threads=1, actual_threads=1))
    return energies
