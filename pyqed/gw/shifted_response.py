"""Matrix-free full-transition GW self-energy and directional first response."""
import numpy as np
from scipy.sparse.linalg import LinearOperator, gmres
from scipy.special import roots_legendre
from .lanczos_response import casida_action


class ShiftedGW:
    r"""Shifted-solve prototype for real restricted, zero-broadening direct-RPA GW.

    Reuses the factorized Casida action (Bruneval, JCP 136, 194107 (2012),
    doi:10.1063/1.4718428) and the SciPy restarted GMRES backend used by
    contour/orbital response (Saad and Schultz, SISC 7, 856 (1986),
    doi:10.1137/0907058). This is an independent resolvent adaptation,
    not a reproduction of either paper's GW algorithm.

    For a=abs(w-e_q), integrate h(C)b=(2a/pi) integral_0^infinity
    (C+t^2 I)^-1 b/(a^2+t^2) dt. The orbital-pair function is s*h(C)
    when s*(w-e_q)>0, or s*(2*(C-a^2 I)^-1-h(C)) otherwise (s=+1
    occupied, -1 virtual). The continuous a=0 limit is treated explicitly.
    This avoids artificial opposite-sign poles. All transitions/intermediate
    orbitals remain. Quadrature and solves are numerical approximations:
    order doubling checks both value and tangent, not a rigorous error bound.
    True self-energy poles remain ill-conditioned; failed solves raise with
    no dense fallback. Residuals alone cannot certify forward accuracy there.
    Inputs are spatial MO energies and CD/RI factors.

    First response includes supplied energy, factor, and frequency tangents;
    it does not construct nuclear/orbital response or static exchange terms.
    No QP root selection, finite broadening, Hessians, or BSE integration yet.
    Resident factors plus bounded restarted Krylov vectors are required; no
    dense transition/auxiliary matrix, square root, or spectrum is constructed.
    """
    def __init__(self, energy, factors, nocc, *, solve_tol=1e-12,
                 tolerance=1e-9, initial_order=16, max_order=256,
                 restart=40, maxiter=400):
        self.energy = np.asarray(energy)
        self.factors = np.asarray(factors)
        n = len(self.energy)
        if (self.energy.shape != (n,) or self.factors.ndim != 3
                or self.factors.shape[1:] != (n,n) or not 0 < nocc < n
                or isinstance(nocc,bool) or not isinstance(nocc, (int,np.integer))
                or np.iscomplexobj(self.energy) or np.iscomplexobj(self.factors)
                or not np.isfinite(self.energy).all() or not np.isfinite(self.factors).all()):
            raise ValueError('Require real finite spatial energies/factors and a valid occupied count')
        for name,value in [('initial_order',initial_order),('max_order',max_order),
                           ('restart',restart),('maxiter',maxiter)]:
            if isinstance(value,bool) or not isinstance(value,(int,np.integer)) or value < 1:
                raise ValueError(name+' must be a positive integer')
        if (not np.isfinite([solve_tol,tolerance]).all() or solve_tol <= 0
                or tolerance <= 0 or max_order < 2*initial_order):
            raise ValueError('Positive tolerances and at least two quadrature orders required')
        self.nocc = int(nocc)
        self.gaps = (self.energy[nocc:][None,:]-self.energy[:nocc,None]).ravel()
        if np.min(self.gaps) <= 0:
            raise ValueError('Positive occupied-virtual gaps required')
        self.pairs = self.factors[:,:nocc,nocc:].reshape(len(factors),-1)
        self.root = np.sqrt(self.gaps)
        self.diagonal = self.gaps**2+4*self.gaps*np.einsum('Pi,Pi->i',self.pairs,self.pairs)
        self.solve_tol,self.tolerance = solve_tol,tolerance
        self.initial_order,self.max_order = initial_order,max_order
        self.restart,self.maxiter = restart,maxiter
        self.info = {}

    def action(self, vector):
        """Apply C through the shared full-transition Casida kernel."""
        return casida_action(self.gaps,self.pairs,np.asarray(vector))

    def action_response(self, vector, energy_tangent, factor_tangent):
        """Apply C' to a fixed vector; supports vector batches."""
        dg = (energy_tangent[self.nocc:][None,:]-energy_tangent[:self.nocc,None]).ravel()
        dp = factor_tangent[:,:self.nocc,self.nocc:].reshape(self.pairs.shape)
        dr = .5*dg/self.root
        root = self.root if vector.ndim == 1 else self.root[:,None]
        dr = dr if vector.ndim == 1 else dr[:,None]
        diagonal = 2*self.gaps*dg
        if vector.ndim != 1: diagonal = diagonal[:,None]
        rv = root*vector
        projected = self.pairs@rv
        return (diagonal*vector+4*dr*(self.pairs.T@projected)
                +4*root*(dp.T@projected+self.pairs.T@(dp@rv+self.pairs@(dr*vector))))

    def solve(self, shift, rhs, *, sign=1):
        """Solve (sign C + shift I)x=b with independently checked GMRES residual."""
        rhs = np.asarray(rhs)
        if rhs.shape != self.gaps.shape or not np.isfinite(rhs).all() or not np.isfinite(shift):
            raise ValueError('Finite shift and transition-space vector required')
        action = lambda v: sign*self.action(v)+shift*v
        operator = LinearOperator((len(rhs),)*2,matvec=action,dtype=float)
        diag = sign*self.diagonal+shift
        scale = np.where(abs(diag)>1e-12,diag,np.where(diag<0,-1e-12,1e-12))
        preconditioner = LinearOperator(operator.shape,matvec=lambda v:v/scale,dtype=float)
        count = 0
        def callback(_):
            nonlocal count
            count += 1
        x,status = gmres(operator,rhs,M=preconditioner,rtol=self.solve_tol*.1,atol=0.,
            restart=min(self.restart,len(rhs)),maxiter=self.maxiter,
            callback=callback,callback_type='legacy')
        residual = np.linalg.norm(action(x)-rhs)/max(np.linalg.norm(rhs),np.finfo(float).tiny)
        self.info['solves'] = self.info.get('solves',0)+1
        self.info['iterations'] = self.info.get('iterations',0)+count
        self.info['max_relative_residual'] = max(self.info.get('max_relative_residual',0.),float(residual))
        if status or not np.isfinite(x).all() or not np.isfinite(residual) or residual > self.solve_tol:
            raise np.linalg.LinAlgError(f'Shifted GW solve failed: status={status}, relative residual={residual:.3e}')
        return x

    def _evaluate_independent(self, orbital, frequency, *, energy_tangent=None,
                 factor_tangent=None, frequency_tangent=0.):
        """Return correlation self-energy and its supplied directional derivative.

        Set frequency_tangent=1 with other tangents zero for dSigma/domega.
        ``info`` records quadrature convergence and checked solve residuals.
        """
        if (not isinstance(orbital,(int,np.integer)) or not 0 <= orbital < len(self.energy)
                or np.iscomplexobj([frequency,frequency_tangent])
                or not np.isfinite([frequency,frequency_tangent]).all()):
            raise ValueError('Valid orbital and finite real frequency/tangent required')
        de = np.zeros_like(self.energy) if energy_tangent is None else np.asarray(energy_tangent)
        df = np.zeros_like(self.factors) if factor_tangent is None else np.asarray(factor_tangent)
        if (de.shape != self.energy.shape or df.shape != self.factors.shape
                or np.iscomplexobj(de) or np.iscomplexobj(df)
                or not np.isfinite(de).all() or not np.isfinite(df).all()):
            raise ValueError('Real finite tangents with matching input shapes required')
        self.info = dict(converged=False,solves=0,iterations=0,max_relative_residual=0.,history=[])
        dg = (de[self.nocc:][None,:]-de[:self.nocc,None]).ravel()
        dp = df[:,:self.nocc,self.nocc:].reshape(self.pairs.shape)
        dc = lambda v:self.action_response(v,de,df)
        previous = None
        order = self.initial_order
        while order <= self.max_order:
            nodes,weights = roots_legendre(order)
            theta = (nodes+1)*np.pi/4
            value = derivative = 0.
            for q in range(len(self.energy)):
                f = self.factors[:,orbital,q]
                d = df[:,orbital,q]
                projected = self.pairs.T@f
                b = self.root*projected
                db = .5*dg/self.root*projected+self.root*(dp.T@f+self.pairs.T@d)
                delta = frequency-self.energy[q]
                dd = frequency_tangent-de[q]
                sign = 1 if q < self.nocc else -1
                magnitude = abs(delta)
                scale = np.sqrt(magnitude*np.min(self.gaps)) if magnitude else float(np.min(self.gaps))
                shifts = (scale*np.tan(theta))**2
                mapped = weights*scale/(2*np.cos(theta)**2)
                y,dy = np.zeros_like(b),np.zeros_like(b)
                for shift,weight in zip(shifts,mapped):
                    x = self.solve(shift,b)
                    dx = self.solve(shift,db-dc(x))
                    if magnitude:
                        kernel = magnitude/(magnitude**2+shift)
                        kernel_d = np.sign(delta)*dd*(shift-magnitude**2)/(magnitude**2+shift)**2
                        y += weight*kernel*x
                        dy += weight*(kernel*dx+kernel_d*x)
                    else:
                        y += weight*x
                if not magnitude:
                    # Continuous delta=0 limit, including its frequency derivative.
                    z = sign*self.solve(0.,b)
                    dz = self.solve(0.,sign*db-dc(z)-dd*y)
                elif delta*sign > 0:
                    z,dz = sign*y,sign*dy
                else:
                    x = self.solve(-delta**2,b)
                    dx = self.solve(-delta**2,db-dc(x)+2*delta*dd*x)
                    z,dz = sign*(2*x-y),sign*(2*dx-dy)
                value += 2*b@z
                derivative += 2*(db@z+b@dz)
            current = np.array([value,derivative])
            self.info['history'].append(dict(order=order,value=float(value),derivative=float(derivative)))
            if previous is not None and np.all(abs(current-previous) <= self.tolerance):
                self.info.update(converged=True,order=order,change=abs(current-previous).tolist())
                return tuple(current)
            previous = current
            order *= 2
        raise np.linalg.LinAlgError('Shifted GW quadrature failed to converge value and first response')

    def evaluate(self, orbital, frequency, *, energy_tangent=None,
                 factor_tangent=None, frequency_tangent=0., reuse=True):
        """Return self-energy and its directional derivative.

        The default scalar-adjoint path reuses each solution for the nuclear
        and frequency responses. ``reuse=False`` is the independent-solve
        validation control. No solutions persist between evaluations.
        """
        if not np.isreal(frequency_tangent) or not np.isfinite(frequency_tangent):
            raise ValueError('Finite real frequency tangent required')
        if not reuse:
            return self._evaluate_independent(orbital,frequency,
                energy_tangent=energy_tangent,factor_tangent=factor_tangent,
                frequency_tangent=frequency_tangent)
        value,nuclear,slope = self.evaluate_many(orbital,[frequency],
            energy_tangent=energy_tangent,factor_tangent=factor_tangent)[0]
        return value,nuclear+frequency_tangent*slope

    def evaluate_many(self, orbital, frequencies, *, energy_tangent=None,
                      factor_tangent=None):
        r"""Return columns (Sigma, supplied structural response, dSigma/domega).

        All frequencies share each positive-shift solve. Symmetry gives the
        exact scalar adjoint 2 db.T x - x.T dC x, eliminating tangent solves.
        This algebraic adaptation introduces no transition truncation or new
        tolerances; references/restrictions are those of :class:`ShiftedGW`.
        Each value and both derivatives must converge under order doubling.
        Shared quadrature uses the smallest nonzero frequency distance in
        each orbital pair to set its local scale. Only the current solution
        and scalar accumulators are retained, never a shift-by-vector cache.
        True-pole corrections still require separate frequency-dependent
        solves. No frequency interpolation or persistent reference cache.
        """
        frequencies = np.asarray(frequencies)
        if (not isinstance(orbital,(int,np.integer)) or not 0 <= orbital < len(self.energy)
                or frequencies.ndim != 1 or not len(frequencies)
                or np.iscomplexobj(frequencies) or not np.isfinite(frequencies).all()):
            raise ValueError('Valid orbital and finite real frequency vector required')
        de = np.zeros_like(self.energy) if energy_tangent is None else np.asarray(energy_tangent)
        df = np.broadcast_to(0.,self.factors.shape) if factor_tangent is None else np.asarray(factor_tangent)
        if (de.shape != self.energy.shape or df.shape != self.factors.shape
                or np.iscomplexobj(de) or np.iscomplexobj(df)
                or not np.isfinite(de).all() or not np.isfinite(df).all()):
            raise ValueError('Real finite tangents with matching input shapes required')
        structural = bool(np.any(de) or np.any(df))
        self.info = dict(converged=False,solves=0,iterations=0,max_relative_residual=0.,history=[],
                         frequencies=len(frequencies),derivative_solves=0)
        dg = (de[self.nocc:][None,:]-de[:self.nocc,None]).ravel()
        dp = df[:,:self.nocc,self.nocc:].reshape(self.pairs.shape)
        dc = lambda v:self.action_response(v,de,df)
        previous = None
        order = self.initial_order
        while order <= self.max_order:
            nodes,weights = roots_legendre(order)
            theta = (nodes+1)*np.pi/4
            result = np.zeros((len(frequencies),3))
            for q in range(len(self.energy)):
                f = self.factors[:,orbital,q]
                projected = self.pairs.T@f
                b = self.root*projected
                db = (.5*dg/self.root*projected+self.root*(dp.T@f+self.pairs.T@df[:,orbital,q])
                      if structural else None)
                delta = frequencies-self.energy[q]
                magnitude = abs(delta)
                sign = 1 if q < self.nocc else -1
                zero = magnitude == 0
                positive = magnitude[~zero]
                scale = (np.sqrt(np.min(positive)*np.min(self.gaps)) if len(positive)
                         else float(np.min(self.gaps)))
                shifts = (scale*np.tan(theta))**2
                mapped = weights*scale/(2*np.cos(theta)**2)
                # Only the continuous zero-delta limit needs a second vector.
                xzero = self.solve(0.,b) if np.any(zero) else None
                if xzero is not None:
                    result[zero,0] += sign*(b@xzero)
                    if structural:
                        result[zero,1] += sign*(2*db@xzero-xzero@dc(xzero))
                for shift,weight in zip(shifts,mapped):
                    x = self.solve(shift,b)
                    bilinear = b@x
                    tangent = 2*db@x-x@dc(x) if structural else 0.
                    kernel = magnitude/(magnitude**2+shift)
                    slope = np.sign(delta)*(shift-magnitude**2)/(magnitude**2+shift)**2
                    orientation = np.where(delta*sign>0,sign,-sign)
                    result[:,0] += weight*orientation*kernel*bilinear
                    result[:,1] += weight*orientation*(kernel*tangent-de[q]*slope*bilinear)
                    result[:,2] += weight*orientation*slope*bilinear
                    if xzero is not None:
                        value = weight*(xzero@x)
                        result[zero,1] += de[q]*value
                        result[zero,2] -= value
                for j in np.flatnonzero(delta*sign<0):
                    x = self.solve(-delta[j]**2,b)
                    slope = 2*delta[j]*(x@x)
                    tangent = 2*db@x-x@dc(x) if structural else 0.
                    result[j] += 2*sign*np.array([b@x,tangent-de[q]*slope,slope])
            result *= 2
            self.info['history'].append(dict(order=order,values=result.tolist()))
            if previous is not None and np.all(abs(result-previous) <= self.tolerance):
                self.info.update(converged=True,order=order,change=abs(result-previous).tolist())
                return result
            previous = result
            order *= 2
        raise np.linalg.LinAlgError('Shifted GW quadrature failed to converge values and first responses')
