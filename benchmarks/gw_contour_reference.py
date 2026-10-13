"""Residual-checked acceleration of the PySCF contour reference.

The full RPA eigenbasis supplies an inverse preconditioner, not a truncated
screening model. Every solve checks the independently assembled PySCF
polarizability action, and failed iterative solves use PySCF's dense response.
This is an adapted reference driver, not a timing of stock GWCD.kernel.
Contour equations: Zhu and Chan, JCTC 17, 727–741 (2021),
doi:10.1021/acs.jctc.0c00704. GMRES: Saad and Schultz, SIAM J. Sci.
Stat. Comput. 7, 856–869 (1986), doi:10.1137/0907058.
"""
import numpy as np
import scipy.linalg as la
from scipy.sparse.linalg import LinearOperator, gmres
from pyscf.gw import gw_cd


class ContourReference:
    """PySCF residue formula with a full-spectrum inverse preconditioner.

    Zero regulator only. All auxiliary functions, MO intermediates and RPA
    modes remain. The RPA eigensystem may come from the tested implementation:
    its output is only an initial guess/preconditioner; acceptance requires
    the independent, complete dielectric equation's residual. Dense fallback
    solves may use a scaled backward-error check at the roundoff floor, with
    an additional RHS-relative cap of 10*tol. This is not a forward-error bound.
    This checks equation agreement, not independent root selection.
    """
    def __init__(self, factors, energy, nocc, poles, projection, tol=1e-10):
        self.factors = factors
        self.energy = np.asarray(energy)
        self.nocc = nocc
        self.poles = np.asarray(poles)
        self.projection = np.asarray(projection)
        self.pairs = np.ascontiguousarray(factors[:, :nocc, nocc:].reshape(len(factors), -1))
        self.gaps = (energy[nocc:]-energy[:nocc,None]).ravel()
        self.tol = tol
        self.info = dict(solves=0, refinements=0, dense_fallbacks=0,
                         max_relative_residual=0.,max_backward_error=0.,dense_checks=[])
        self._orbital = None

    def select(self, orbital):
        if self._orbital != orbital:
            self._amplitudes = self.projection.T @ self.factors[:, orbital, :]
            self._orbital = orbital

    def solution(self, frequency, rhs, amplitudes=None):
        # PySCF get_rho_response_R at eta=0; keep its two-term expression.
        coefficients = 2*(1/(frequency-self.gaps)+1/(-frequency-self.gaps))
        response = lambda x: x-self.pairs @ (coefficients*(self.pairs.T @ x))
        weights = 4*self.poles/(frequency**2-self.poles**2)
        inverse = lambda x: x+self.projection @ (weights*(self.projection.T @ x))
        if amplitudes is None:
            amplitudes = self.projection.T @ rhs
        x = rhs+self.projection @ (weights*amplitudes)
        norm = max(la.norm(rhs), np.finfo(float).tiny)
        relative = la.norm(response(x)-rhs)/norm
        backward = None
        certified = False
        self.info['solves'] += 1
        if relative > self.tol:
            self.info['refinements'] += 1
            operator = LinearOperator((len(rhs),)*2, matvec=response, dtype=float)
            preconditioner = LinearOperator(operator.shape, matvec=inverse, dtype=float)
            x, status = gmres(operator, rhs, x0=x, M=preconditioner,
                              rtol=self.tol*.1, atol=0., restart=10, maxiter=3)
            relative = la.norm(response(x)-rhs)/norm
            if status or relative > self.tol:
                self.info['dense_fallbacks'] += 1
                pi = gw_cd.get_rho_response_R(-self.gaps.reshape(self.nocc,-1),
                    frequency, self.pairs.reshape(len(rhs),self.nocc,-1), 0.)
                matrix = np.eye(len(rhs))-pi.real
                x = la.solve(matrix, rhs, assume_a='sym', check_finite=False)
                residual = la.norm(response(x)-rhs)
                relative = residual/norm
                backward = residual/(la.norm(matrix)*la.norm(x)+norm)
                certified = relative <= 10*self.tol and backward <= self.tol*.001
                self.info['max_backward_error'] = max(self.info['max_backward_error'],float(backward))
                self.info['dense_checks'].append(dict(frequency_Ha=float(frequency),
                    relative_residual=float(relative),backward_error=float(backward)))
        self.info['max_relative_residual'] = max(self.info['max_relative_residual'],float(relative))
        if not np.isfinite(relative) or (relative > self.tol and not certified):
            raise np.linalg.LinAlgError(f'Independent dielectric residual {relative:.3g} '
                                       f'exceeds {self.tol}; dense backward error={backward}')
        return x

    def residues(self, mo_energy, omega, orbp, ef, Lpq, Lia, eta):
        if eta != 0 or not np.array_equal(mo_energy, self.energy):
            raise ValueError('Reference cache requires identical energies and eta=0')
        self.select(orbp)
        if omega > ef:
            sign = 1
            active = np.flatnonzero((mo_energy < omega)&(mo_energy > ef))
        else:
            sign = -1
            active = np.flatnonzero((mo_energy > omega)&(mo_energy < ef))
        value = 0.
        for m in active:
            rhs = self.factors[:,orbp,m]
            solution = self.solution(abs(mo_energy[m]-omega), rhs, self._amplitudes[:,m])
            value += sign*np.dot(self.factors[:,m,orbp],solution-rhs)
        return value

    def imaginary_integral(self, omega, screened, freqs, weights, static):
        """Static-subtracted PySCF imaginary integral plus its residue formula.

        The integral of the constant W(0) term is analytic at eta=0. This
        avoids an unresolved narrow Green-function peak near an HF energy;
        it changes quadrature evaluation, not the contour equation. Stock
        GWCD uses the unsubtracted quadrature and may require more points.
        """
        delta = float(omega)-self.energy
        difference = screened-static[None,:]
        integral = -np.sum(weights[:,None]*delta[None,:]*difference/
                           (delta[None,:]**2+freqs[:,None]**2))/np.pi
        integral -= .5*np.dot(np.sign(delta),static)
        # A Green-function pole on the contour contributes half its residue.
        boundary = delta == 0
        occupation = np.arange(len(delta)) < self.nocc
        integral += np.sum((.5-occupation[boundary])*static[boundary])
        return integral
