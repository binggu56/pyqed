"""Dominant poles of the diagonal molecular GW Green function."""
import numpy as np
from threadpoolctl import threadpool_limits
from .contour import ContourDeformation


class PoleSpectrum:
    """Select the largest-weight Dyson pole, rather than a continuation branch.

    Diagonal restricted direct-RPA GW, using the full spectral self-energy
    (Bruneval, JCP 136, 194107, 2012, doi:10.1063/1.4718428). Largest-weight
    selection follows the quasiparticle definition discussed by Marie and
    Loos, JCTC 19, 3943–3957 (2023), doi:10.1021/acs.jctc.3c00281; this is not their SRG
    approximation. The arrowhead secular equation is monotone between poles.
    Interval bounds on spectral weights discard intervals only when they
    cannot beat the selected root. No finite real-frequency search window.

    Roots use the zero-broadening limit. Numerically negligible couplings
    with residue <= tol**2/(100*N) are omitted in selection only: their total
    arrowhead perturbation norm is <= tol/10 Ha. This bounds eigenvalue shifts,
    not weights of nearly degenerate peaks. Original couplings remain in
    self-energy and contour evaluations. Small maximum weights signal genuine
    fragmentation; largest-weight selection does not make a sharp QP exist.
    Full charge-RPA diagonalization costs quadratic transition-space storage.
    Couplings remain factorized and are formed only in orbital batches.
    Selected roots are then refined against all residues to at least 1e-12 Ha
    residual tolerance (or floating-point accuracy), without reselecting branches;
    this tighter reference is needed by analytic derivatives.
    """
    def __init__(self, gw, poles=None, vectors=None):
        from .gw import _charge_rpa, _charge_spatial_couplings, _active_energy
        self.energy = _active_energy(gw)[::2].copy()
        self.reference = gw.e_mf[::2].copy()
        self.factors = gw._pair_factors
        self.mo_coeff = gw.mo_coeff.copy()
        self.nocc = gw.nocc//2
        with threadpool_limits(limits=gw.threads,user_api='blas'):
            if poles is None:
                poles,vectors = _charge_rpa(gw)
            self.poles = np.asarray(poles)
            self.couplings = _charge_spatial_couplings(gw,poles,vectors)
        self.centers = np.r_[self.energy[:self.nocc,None]-poles,
                             self.energy[self.nocc:,None]+poles].ravel()
        self.roots = np.full(len(self.energy),np.nan)
        self.brackets = np.full((len(self.energy),2),np.nan)
        self.weights = self.roots.copy()
        self.residuals = self.roots.copy()
        self.excluded_bounds = self.roots.copy()
        self.tolerances = np.full(len(self.energy),np.inf)
        self.static = np.full(len(self.energy),np.nan)
        self.searched = np.zeros(len(self.energy),dtype=int)

    def refine(self, gw, tol=1e-12):
        """Refine the selected branches against the full zero-width self-energy.

        Dominant-root selection may omit negligible residues within its energy
        tolerance. Analytic derivatives can be more sensitive than energies to
        the remaining root error. This Newton refinement keeps the selected
        pole intervals and all residues; it does not repeat or change the
        largest-weight search described in the class references.
        """
        from scipy.optimize import newton
        if not np.isfinite(tol) or tol <= 0:
            raise ValueError('Root refinement tolerance must be positive.')
        roots, weights, residuals = self.roots.copy(), self.weights.copy(), self.residuals.copy()
        for p in np.flatnonzero(np.isfinite(roots)):
            residue = (2*self.couplings[p]**2).ravel()
            active = residue > 0
            centers, residue = self.centers[active], residue[active]
            def equation(x):
                return x-self.reference[p]-self.static[p]-np.sum(residue/(x-centers))
            def slope(x):
                return 1+np.sum(residue/(x-centers)**2)
            root = newton(equation,roots[p],fprime=slope,tol=tol/10,maxiter=50)
            if not self.brackets[p,0] < root < self.brackets[p,1]:
                raise RuntimeError(f'QP refinement crossed a pole interval for orbital {p}.')
            error = abs(equation(root))
            if error > max(tol,100*np.finfo(float).eps*max(1.,abs(root))):
                raise RuntimeError(f'QP refinement residual for orbital {p}: {error:g}')
            roots[p], weights[p], residuals[p] = root, 1/slope(root), error
        self.roots, self.weights, self.residuals = roots, weights, residuals
        gw.qp_weights, gw.qp_residuals = weights.copy(), residuals.copy()
        gw.qp_selection_info = dict(gw.qp_selection_info, refinement='full-residue fixed-branch Newton')
        return roots.copy()

    def solve(self,gw,orbs,tol=1e-9):
        from .gw import _dyson,_sigma_x_diagonal
        if _dyson is None or not hasattr(_dyson,'dominant'):
            raise RuntimeError('Build the GW Dyson kernel for dominant quasiparticle selection')
        ids=np.asarray(list(orbs))
        if (ids.ndim!=1 or not len(ids) or ids.dtype.kind not in 'iu'
                or len(np.unique(ids))!=len(ids) or np.any(ids<0) or np.any(ids>=len(self.energy))):
            raise ValueError('Require unique valid spatial orbital indices')
        if not np.isfinite(tol) or tol<=0:
            raise ValueError('Require a positive finite root tolerance')
        all_static=_sigma_x_diagonal(gw,ids)-np.diag(gw.v_mf)[2*ids]
        stale=(~np.isfinite(self.roots[ids]) | (self.tolerances[ids]>tol)
               | (self.static[ids]!=all_static))
        missing=ids[stale]
        static=all_static[stale]
        batch=max(1,min(gw.threads,int(max(1e-6,gw.max_memory)*1e6/(128*len(self.centers)))))
        actual=1
        for first in range(0,len(missing),batch):
            targets=missing[first:first+batch]
            with threadpool_limits(limits=gw.threads,user_api='blas'):
                residue=np.stack([(2*self.couplings[int(p)]**2).ravel() for p in targets])
            result=_dyson.dominant(self.centers,residue,self.reference[targets],
                                  static[first:first+batch],gw.threads,tol)
            roots,weights,residuals,used,bounds,searched,brackets=result
            self.roots[targets],self.weights[targets],self.residuals[targets]=roots,weights,residuals
            self.excluded_bounds[targets],self.searched[targets],self.brackets[targets]=bounds,searched,brackets
            self.tolerances[targets]=tol
            self.static[targets]=static[first:first+batch]
            actual=max(actual,used)
        gw.qp_weights=self.weights.copy();gw.qp_residuals=self.residuals.copy()
        gw.qp_selection_info=dict(method='largest spectral weight',zero_broadening=True,
            weight_bounds_checked=True,orbitals=ids.tolist(),
            excluded_weight_bounds=self.excluded_bounds[ids].tolist(),
            searched_intervals=self.searched[ids].tolist(),coupling_energy_error_bound_Ha=tol/10,
            fragmented_orbitals=ids[self.weights[ids]<.5].tolist())
        gw.threading_info=dict(algorithm='dominant Dyson poles',openmp=_dyson.openmp_available(),
            requested_threads=gw.threads,actual_threads=actual,batch_orbitals=batch)
        return self.refine(gw, tol=min(tol,1e-12)) if len(missing) else self.roots.copy()


def spectrum(gw):
    """Reuse poles only for the identical screening and mean-field reference."""
    from .gw import _active_energy
    obj=getattr(gw,'_qp_spectrum',None)
    if (obj is None or obj.factors is not gw._pair_factors
            or not np.array_equal(obj.energy,_active_energy(gw)[::2])
            or not np.array_equal(obj.reference,gw.e_mf[::2])
            or not np.array_equal(obj.mo_coeff,gw.mo_coeff)):
        obj=gw._qp_spectrum=PoleSpectrum(gw)
    return obj


class PoleContour(ContourDeformation):
    """Same contour quadrature with the complete spectral dielectric response.

    Uses the contour formulation and limitations of ContourDeformation and
    the full-pole screening of PoleSpectrum. No RPA poles are removed from W.
    Scratch is one target orbital's three-index couplings, not a four-index ERI.
    """
    def __init__(self,gw,energy,orbs,nw=64,quadrature_scale=.5,**kwargs):
        super().__init__(gw,energy,orbs,nw,quadrature_scale,prepare=False,**kwargs)
        self.residue_solver="spectral"
        self.screening_info["solver"]="spectral"
        self.spectrum=spectrum(gw)
        self.static_screening=self.spectrum.couplings
        self.poles=self.spectrum.poles
        self._squared_orbital=None
        self._squared=None
        weights=-4*self.poles[:,None]/(self.poles[:,None]**2+np.r_[0.,self.freqs][None,:]**2)
        with threadpool_limits(limits=gw.threads,user_api='blas'):
            for j,p in enumerate(self.orbs):
                values=self.spectrum.couplings[int(p)]**2@weights
                self.w0[j]=values[:,0];self.w_imag[j]=values[:,1:]
        self.prepared=True

    def _residue(self,frequency,orbital,intermediate):
        square=(self._squared[intermediate] if self._squared_orbital==orbital
                else self.spectrum.couplings[int(orbital),int(intermediate)]**2)
        z=frequency+1j*self.eta
        d=z*z-self.poles**2
        weighted=4*self.poles*square
        return np.sum(weighted/d),np.sum(-2*z*weighted/d**2)

    def evaluate(self,orbital,omega):
        if self._squared_orbital!=orbital:
            self._squared=self.spectrum.couplings[int(orbital)]**2
            self._squared_orbital=orbital
        j=self.columns[orbital]
        delta=float(omega)-self.energy
        value,derivative=self._quadrature(j,delta)
        occupied=np.arange(len(delta))<self.nocc
        active=np.flatnonzero(np.where(occupied,delta<0,delta>0))
        if len(active):
            z=abs(delta[active])+1j*self.eta
            d=z[:,None]**2-self.poles[None,:]**2
            weighted=4*self.poles*self._squared[active]
            screened=np.sum(weighted/d,axis=1)
            value+=np.sum(np.where(occupied[active],-1,1)*(screened-self.w0[j,active]))
            derivative+=np.sum(-2*z[:,None]*weighted/d**2)
        return value,derivative


def refine_contour(response,p,seed,bracket,energy,static,tol):
    """Refine the selected pole without crossing either neighboring pole."""
    x=float(seed)
    for _ in range(30):
        sigma,derivative=response.evaluate(int(p),x)
        residual=x-energy-static-sigma.real
        slope=1-derivative.real
        if slope<=0 or not np.isfinite([residual,slope]).all():
            raise RuntimeError('Selected contour root lost positive spectral weight; reduce eta')
        if abs(residual)<=tol:
            return x,1/slope,abs(residual)
        trial=x-residual/slope
        if not bracket[0]<trial<bracket[1]:
            raise RuntimeError('Contour refinement would cross a self-energy pole; reduce eta or refine quadrature')
        x=trial
    raise RuntimeError('Selected contour root did not converge')
