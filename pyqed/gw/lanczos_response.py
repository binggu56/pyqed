"""Reusable inverse-Casida Lanczos resolvents for complete RI direct RPA."""
from collections import OrderedDict
import time

import numpy as np
import scipy.linalg as la


class LanczosResponse:
    r"""Adapted full-frequency response reuse, retaining every RI transition.

    References: J. Laflamme Janssen, B. Rousseau and M. Cote, Phys. Rev. B
    91, 125120 (2015), doi:10.1103/PhysRevB.91.125120; W. Gao et al.,
    Phys. Rev. Lett. 132, 126402 (2024), doi:10.1103/PhysRevLett.132.126402.

    This is an adaptation for real, gapped restricted direct RPA. Unlike
    Gao et al.'s polynomial square-root/self-energy formulation, it reuses
    the inverse squared Casida response for contour residues only. With
    D=diag(gaps), C=D^2+4 sqrt(D) F.T F sqrt(D), K=C^{-1}, and
    b=2 sqrt(D) F.T rhs, W_c=-b.T K (I-z^2 K)^{-1} b. Woodbury applies K
    using the complete static dielectric Cholesky factor. Imaginary-axis
    quadrature, eta, intermediate orbitals and BSE screening are unchanged.

    Two-pass reorthogonalized Lanczos chains are cached by orbital pair.
    Value and derivative error bounds use 0 <= K <= 1/min(gaps)^2 and the
    Krylov residual, including a floating-point relation-defect allowance.
    These are exact-arithmetic resolvent bounds, not guarantees against
    roundoff or an inaccurate input Cholesky factor. Complete/invariant
    chains additionally check the original dielectric residual. Near poles
    the bounds may be conservative; iteration/memory exhaustion raises.
    No orbital/pole truncation, continuation fit, or hidden GMRES fallback
    is used. Numerical tolerances must still be validated against reference
    GW energies and derivatives.
    """

    def __init__(self, contour):
        self.contour = contour
        self.gaps = contour.gaps
        self.inverse_power = self.gaps**-1.5
        self.upper = float(np.min(self.gaps)**-2)
        self.cache = OrderedDict()
        self.budget = getattr(contour, 'response_cache_bytes', 128*1024**2)

    def inverse_action(self, vector):
        c = self.contour
        weighted = self.inverse_power*vector
        aux = np.empty(len(c.factors))
        for first in range(0, len(aux), 128):
            s = slice(first, first+128)
            aux[s] = c.pairs[s]@weighted
        aux = la.cho_solve(c.static_screening.cholesky, aux, check_finite=False)
        return vector/self.gaps**2 - 4*self.inverse_power*c._pair_projection(aux)

    def _extend(self, chain):
        q = chain['next']
        chain['basis'].append(q)
        w = self.inverse_action(q)
        alpha = float(q@w)
        w -= alpha*q
        if len(chain['basis']) > 1:
            w -= chain['beta'][-1]*chain['basis'][-2]
        defect = 0.
        for _ in range(2):
            for v in chain['basis']:
                coefficient = float(v@w)
                defect += coefficient**2
                w -= coefficient*v
        beta = float(np.linalg.norm(w))
        chain['alpha'].append(alpha)
        chain['beta'].append(beta)
        chain['defect2'] += defect
        chain['exhausted'] = (len(chain['basis']) == len(self.gaps)
                              or beta <= 32*np.finfo(float).eps*self.upper)
        chain['next'] = None if chain['exhausted'] else w/beta
        chain['spectrum'] = None
        c = self.contour
        c.screening_info['iterations'] += 1
        c.screening_info['max_iterations'] = max(c.screening_info['max_iterations'], len(chain['alpha']))

    def _evaluate(self, chain, z):
        if chain['spectrum'] is None:
            chain['spectrum'] = la.eigh_tridiagonal(np.array(chain['alpha']),
                np.array(chain['beta'][:-1]), check_finite=False)
        eigen, vectors = chain['spectrum']
        denominator = 1-z*z*eigen
        amplitudes = vectors[0]/denominator
        d = chain['norm']*(vectors@amplitudes)
        dp = chain['norm']*(vectors@(2*z*eigen*vectors[0]/denominator**2))
        value = -chain['norm']**2*np.sum(vectors[0]**2*eigen/denominator)
        slope = -2*z*chain['norm']**2*np.sum(vectors[0]**2*eigen**2/denominator**2)
        lam = 0. if z == 0 else np.clip((z*z).real/abs(z*z)**2, 0., self.upper)
        distance = abs(1-z*z*lam)
        if distance == 0:
            return value, slope, np.inf, np.inf, d
        defect = (np.sqrt(2*chain['defect2'])
                  + 32*np.finfo(float).eps*self.upper*np.sqrt(len(d)))
        tail = chain['beta'][-1]*abs(d[-1])+defect*np.linalg.norm(d)
        tailp = chain['beta'][-1]*abs(dp[-1])+defect*np.linalg.norm(dp)
        error_s = abs(z*z)*tail/distance
        error_sp = (2*abs(z)*tail+abs(z*z)*tailp+2*abs(z)*self.upper*error_s)/distance
        return (value, slope, chain['norm']*self.upper*error_s,
                chain['norm']*self.upper*error_sp, d)

    def _certify_complete(self, chain, z, rhs, d):
        # Complete/invariant chains can beat the conservative interval bound.
        eigen, vectors = chain['spectrum']
        td = vectors@(eigen*(vectors.T@d))
        y = np.zeros(len(self.gaps), dtype=complex)
        for q, coefficient in zip(chain['basis'], td):
            y -= coefficient*q
        c = self.contour
        weighted = np.sqrt(self.gaps)*y
        solution = np.array(rhs, dtype=complex)
        for first in range(0, len(rhs), 128):
            s = slice(first, first+128)
            block = c.pairs[s]
            solution[s] += 2*(block@weighted.real+1j*(block@weighted.imag))
        relative = np.linalg.norm(c.dielectric_action(z.real, solution)-rhs)/max(np.linalg.norm(rhs), np.finfo(float).tiny)
        if not np.isfinite(relative) or relative > c.screening_tol:
            raise np.linalg.LinAlgError(f'Screening Lanczos failed: complete-chain dielectric residual={relative:.3g}')
        c.screening_info['max_relative_residual'] = max(c.screening_info['max_relative_residual'], float(relative))
        c.screening_info['dielectric_residual_checks'] = c.screening_info.get('dielectric_residual_checks', 0)+1
        projection = c._pair_projection(solution)
        derivative_weights = -8*z*self.gaps/(z*z-self.gaps**2)**2
        return rhs@(solution-rhs), np.sum(derivative_weights*projection**2)

    def residue(self, frequency, key, rhs):
        started = time.perf_counter()
        c = self.contour
        c.screening_info['solves'] += 1
        chain = self.cache.get(key)
        if chain is None:
            b = 2*np.sqrt(self.gaps)*c._pair_projection(rhs)
            norm = float(np.linalg.norm(b))
            if norm == 0:
                return 0j, 0j
            chain = dict(norm=norm, basis=[], alpha=[], beta=[], next=b/norm,
                         defect2=0., exhausted=False, spectrum=None)
            self.cache[key] = chain
            c.screening_info['response_chains_created'] = c.screening_info.get('response_chains_created', 0)+1
        self.cache.move_to_end(key)
        before = len(chain['alpha'])
        z = frequency+1j*c.eta
        try:
            while True:
                if not chain['alpha']:
                    self._extend(chain)
                value, slope, ev, ed, d = self._evaluate(chain, z)
                if chain['exhausted']:
                    value, slope = self._certify_complete(chain, z, rhs, d)
                    break
                if ev <= c.screening_tol*max(1., abs(value)) and ed <= c.screening_tol*max(1., abs(slope)):
                    info = c.screening_info
                    info['max_value_error_bound'] = max(info.get('max_value_error_bound', 0.), float(ev))
                    info['max_derivative_error_bound'] = max(info.get('max_derivative_error_bound', 0.), float(ed))
                    break
                if len(chain['alpha']) >= c.screening_maxiter:
                    raise np.linalg.LinAlgError(f'Screening Lanczos failed at {frequency:.12g}: '
                        f'steps={len(chain["alpha"])}, value bound={ev:.3g}, derivative bound={ed:.3g}')
                if (len(chain['basis'])+2)*len(self.gaps)*8 > self.budget:
                    raise MemoryError('Lanczos response chain exceeds its cache memory budget')
                self._extend(chain)
        except (np.linalg.LinAlgError, MemoryError):
            c.screening_info['failed_solves'] += 1
            raise
        if len(chain['alpha']) == before:
            c.screening_info['reused_responses'] = c.screening_info.get('reused_responses', 0)+1
            timing = 'reused_response_seconds'
        else:
            c.screening_info['extended_responses'] = c.screening_info.get('extended_responses', 0)+1
            timing = 'response_build_seconds'
        def size():
            return sum((len(t['basis'])+1)*len(self.gaps)*8 for t in self.cache.values())
        while len(self.cache) > 1 and size() > self.budget:
            self.cache.popitem(last=False)
            c.screening_info['response_cache_evictions'] = c.screening_info.get('response_cache_evictions', 0)+1
        c.screening_info[timing] = c.screening_info.get(timing, 0.)+time.perf_counter()-started
        return value, slope
