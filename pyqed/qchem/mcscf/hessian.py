"""Relaxed nuclear curvature for conventional CASCI and CASSCF."""
import copy
from time import perf_counter
import numpy as np
from scipy.sparse.linalg import LinearOperator, minres


class CASCIHessian:
    """Analytic isolated-state CASCI Hessian on native RHF orbitals.

    Uses projected Rayleigh--Schrodinger eigenvector response (see Helgaker,
    Jorgensen and Olsen, Molecular Electronic-Structure Theory, Wiley, 2000,
    doi:10.1002/9781119019572), combined with PyQED's first/second canonical
    RHF orbital response. This is an implementation of that response
    formulation, not a CASSCF Hessian. No excited-root truncation is used.
    Currently restricted to real, conventional direct-CI, all-electron RHF
    references without spatial-symmetry filtering. Integral derivatives are
    dense; this is not a scalable tensor-network or density-fitting driver.
    Degenerate-state surfaces and nonlinear-coordinate curvature are excluded.
    """

    def __init__(self, mc):
        self.base = mc
        self.success = False
        self.hessian = None

    def run(self, state=0, modes=None, *, tolerance=1e-9, maxiter=1000,
               backend="native"):
        """Compute curvature in ``hessian`` and return this driver.

        Output is Cartesian (atom,xyz,atom,xyz) or a directional matrix.

        Directions are constant Cartesian displacement vectors in bohr.
        ``response_residuals`` stores absolute projected-equation residuals.
        Failure to converge raises rather than returning an approximate Hessian.
        """
        from pyqed.qchem.hf import RHF
        from pyqed.qchem.geometric import _active_integral_derivatives_from_full_mo
        from pyqed.qchem.nac.sacasscf import FixedOrbitalCASCIDriver

        self.success = False
        mc = self.base
        self.hessian = None
        solver = mc if hasattr(mc, "ci_sigma") else getattr(mc, "_direct_solver", None)
        if not isinstance(mc.mf, RHF) or hasattr(mc, "casci"):
            raise NotImplementedError("Requires a native RHF-reference CASCI, not CASSCF.")
        if solver is None or solver.ci is not mc.ci:
            raise NotImplementedError("Run CASCI with method='direct_ci' first.")
        factorized = (getattr(mc, 'use_cholesky', False)
                      or any(getattr(mc.mf, flag, False) for flag in
                             ('density_fit', 'low_rank_jk', 'cholesky_jk')))
        if factorized or getattr(mc.mf, 'with_solvent', None):
            raise NotImplementedError("Factorized or solvent references are unsupported.")
        if getattr(mc.mol, 'ecp', None) or getattr(mc.mol, '_ecp', None):
            raise NotImplementedError("ECP Hessians are unsupported.")
        energies = np.asarray(mc.e_tot)
        if not isinstance(state, (int, np.integer)) or not 0 <= state < len(energies):
            raise ValueError("state must select a saved CI root.")
        gaps = np.abs(energies - energies[state])
        if np.any(np.delete(gaps, state) < 1e-7):
            raise ValueError("Degenerate roots require multistate response.")
        if getattr(mc, "wfnsym", None) is not None:
            raise NotImplementedError("Symmetry-filtered CI response is unsupported.")
        if tolerance <= 0 or maxiter < 1:
            raise ValueError("Positive tolerance and maxiter required.")
        c = np.asarray(mc.ci[state])
        if np.iscomplexobj(c) or np.iscomplexobj(mc.mo_coeff):
            raise NotImplementedError("Only real orbitals and CI vectors are supported.")
        c = c / np.linalg.norm(c)
        energy = float(mc.e_tot[state] - mc.e_core)
        project = lambda v: v - c * np.dot(c, v)
        sigma = solver.ci_sigma
        if np.linalg.norm(sigma(c) - energy * c) > 10 * tolerance:
            raise RuntimeError("Reference CI eigenvector is not converged.")
        first, second, terms = mc.vibronic_couplings(
            state_ids=[state], modes=modes, frame="rhf-relaxed",
            backend=backend, return_terms=True)
        h1, eri1 = _active_integral_derivatives_from_full_mo(
            terms.h1_mo_cartesian, terms.eri1_mo_cartesian, mc.ncore, mc.ncas)
        adapter = FixedOrbitalCASCIDriver(mc.mf, mc.mo_coeff, mc.ncore, mc.ncas)
        rhs = np.array([project(adapter._make_active_sigma_casci(solver, h, v).ci_sigma(c))
                        for h, v in zip(h1, eri1)])
        def action(v):
            q = project(v)
            return project(sigma(q) - energy * q) + c * np.dot(c, v)
        operator = LinearOperator((c.size, c.size), matvec=action, dtype=float)
        response = []
        residuals = []
        for b in rhs:
            x, info = minres(operator, -b, rtol=tolerance * 0.01, maxiter=maxiter)
            residual = np.linalg.norm(action(x) + b)
            if info != 0 or residual > tolerance:
                raise RuntimeError(f"CI response failed: info={info}, residual={residual:g}")
            response.append(project(x))
            residuals.append(residual)
        correction = rhs @ np.asarray(response).T
        n = len(rhs)
        self.hessian = np.real_if_close(second[0, 0].reshape(n, n) + correction + correction.T)
        self.gradient = first[0, 0]
        self.response_residuals = np.asarray(residuals)
        if modes is None:
            self.hessian = self.hessian.reshape(mc.mol.natom, 3, mc.mol.natom, 3)
        self.success = True
        return self


class CASSCFHessian:
    """Analytic ground-state, state-specific CASSCF nuclear Hessian.

    Adapts the symmetric-orthonormalization/exponential-response formulation
    of Helgaker, Almlof, Jensen and Jorgensen, J. Chem. Phys. 84, 6266--6279
    (1986), doi:10.1063/1.450771. Uses PyQED Gaussian derivatives and a joint
    orbital/CI MINRES solve, not the paper's integral/storage implementation.
    First-order coupled response suffices at a stationary wavefunction.
    Real all-electron conventional direct CI only; no state averaging,
    reduced tensor-network response, solvent, ECP, or factorized integrals.
    Second derivative ERIs are shell-contracted; first derivatives are streamed.
    Dense reference integrals still limit basis size. The caller must supply an isolated ground-state stationary point;
    this driver does not certify electronic stability or root identity.
    """

    def __init__(self, mc):
        self.base = mc
        self.hessian = None
        self.success = False

    def run(self, modes=None, *, tolerance=1e-9, stationarity_tolerance=1e-7,
            maxiter=1000, backend="native", derivatives="streamed", precondition=True):
        """Populate ``hessian`` and ``gradient`` and return this driver.

        Constant Cartesian directions give a square directional Hessian;
        without directions the shape is (natom,3,natom,3). Nonlinear coordinate
        maps require additional coordinate-curvature terms. No level shift or
        truncation of the CI response is used. Failed response raises.
        """
        from pyqed.qchem.hf import RHF
        from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
        from pyqed.qchem.geometric import _active_integral_derivatives_from_full_mo
        from pyqed.qchem.nac.sacasscf import FixedOrbitalCASCIDriver

        self.success = False
        started = perf_counter()
        self.timings = {}
        self.hessian = self.gradient = self.response_residuals = None
        base = self.base
        if derivatives not in ('streamed', 'dense-reference'):
            raise ValueError("derivatives must be 'streamed' or 'dense-reference'.")
        if derivatives == 'streamed' and backend not in ('native', 'auto'):
            raise NotImplementedError("Streamed curvature requires the compiled integral engine.")
        if not isinstance(base, SecondOrderCASSCF) or not isinstance(base.mf, RHF):
            raise NotImplementedError("Requires native SecondOrderCASSCF on RHF orbitals.")
        if not base.converged or base.casci is None:
            raise ValueError("Converge CASSCF before computing its Hessian.")
        if base.nstates != 1 or base.state_id != 0 or base.weights is not None:
            raise NotImplementedError("Only state-specific ground-state CASSCF is supported.")
        if (base.use_cholesky or any(getattr(base.mf, flag, False) for flag in
                ('density_fit', 'low_rank_jk', 'cholesky_jk', 'with_solvent'))
                or getattr(base.mol, 'ecp', None) or getattr(base.mol, '_ecp', None)):
            raise NotImplementedError("Requires unfactorized all-electron vacuum integrals.")
        mc = base.casci
        if not hasattr(mc, 'ci_sigma'):
            raise NotImplementedError("Conventional direct-CI response is required.")
        if getattr(mc, 'wfnsym', None) is not None:
            raise NotImplementedError("Symmetry-filtered CI response is unsupported.")
        if tolerance <= 0 or stationarity_tolerance <= 0 or maxiter < 1:
            raise ValueError("Positive tolerances and maxiter required.")
        if np.iscomplexobj(base.mo_coeff) or np.iscomplexobj(mc.ci[0]):
            raise NotImplementedError("Only real orbitals and CI vectors are supported.")
        # Keep response caches separate from the optimizer's cached keyframes.
        driver = copy.copy(base)
        driver._full_derivative_cache = driver._full_derivative_sigma_cache = None
        driver._integral_hessian_cache = None
        c = np.asarray(mc.ci[0], dtype=float)
        c = c / np.linalg.norm(c)
        project = lambda v: v - c * np.dot(c, v)
        h, eri = driver._get_integrals(base.mo_coeff)
        energy = float(mc.e_tot[0] - mc.e_core)
        from .orbopt import generalized_fock, orbital_gradient as orbital_energy_gradient
        dm1, dm2 = driver._effective_rdms(mc, 0)
        orbital_gradient = driver._pack_orbitals(
            orbital_energy_gradient(generalized_fock(h, eri, dm1, dm2)),
            mc.ncore, mc.ncas, driver.nmo)
        self.stationarity_residual = max(np.linalg.norm(orbital_gradient, ord=np.inf)
                                        if orbital_gradient.size else 0.,
                                        np.linalg.norm(mc.ci_sigma(c) - energy*c))
        if self.stationarity_residual > stationarity_tolerance:
            raise RuntimeError(f"CASSCF is not stationary: {self.stationarity_residual:g}")
        norb = orbital_gradient.size
        from .nuclear_derivatives import orbital_ci_coupling
        coupling = orbital_ci_coupling(driver, mc, h, eri, c)
        coupling = coupling.reshape(norb, c.size)
        coupling -= (coupling @ c)[:, None] * c

        def action(v):
            orb, ci = v[:norb], project(v[norb:])
            oo = driver._analytic_orbital_hessian_action(h, eri, dm1, dm2, mc, orb)
            # CI normalization gives 2(H-E), and both cross blocks carry 2.
            oc = 2 * coupling @ ci
            cc = 2 * project(mc.ci_sigma(ci) - energy * ci)
            return np.concatenate((oo + oc, cc + 2*coupling.T @ orb + c*np.dot(c, v[norb:])))

        self.timings['electronic_setup'] = perf_counter() - started
        stamp = perf_counter()
        if derivatives == 'streamed':
            from .nuclear_derivatives import stationary_nuclear_terms
            gradient, explicit, rhs = stationary_nuclear_terms(
                driver, mc, h, eri, dm1, dm2, c, modes)
        else:
            first, second, terms = base.vibronic_couplings(
                state_ids=[0], modes=modes, frame='symmetric', return_terms=True,
                backend=backend)
            dh, dv = terms.h1_mo_cartesian, terms.eri1_mo_cartesian
            ah, av = _active_integral_derivatives_from_full_mo(dh, dv, mc.ncore, mc.ncas)
            adapter = FixedOrbitalCASCIDriver(base.mf, base.mo_coeff, mc.ncore, mc.ncas)
            rhs = []
            for k in range(len(dh)):
                driver._full_derivative_cache = driver._full_derivative_sigma_cache = None
                orb = driver._exact_orbital_gradient_vector(mc, dh[k], dv[k], c)
                ci = 2 * project(adapter._make_active_sigma_casci(mc, ah[k], av[k]).ci_sigma(c))
                rhs.append(np.concatenate((orb, ci)))
            rhs = np.asarray(rhs)
            gradient, explicit = first[0, 0].real, second[0, 0].real.reshape(len(rhs), len(rhs))
        self.timings['nuclear_derivatives'] = perf_counter() - stamp
        stamp = perf_counter()
        size = norb + c.size
        operator = LinearOperator((size, size), matvec=action, dtype=float)
        preconditioner = None
        if precondition:
            from .orbopt import generalized_fock, diagonal_hessian
            orbital_diag = diagonal_hessian(generalized_fock(h, eri, dm1, dm2),
                                            mc.ncore, mc.ncas, absolute=True)
            inverse = 1/np.maximum(np.abs(np.concatenate(
                (orbital_diag, 2*(mc.ci_diagonal()-energy)))), 1e-4)
            preconditioner = LinearOperator((size, size), matvec=lambda v: inverse*v, dtype=float)
        response, residuals, iterations = [], [], []
        for b in rhs:
            count = [0]
            def count_iteration(_):
                count[0] += 1
            x, info = minres(operator, -b, rtol=tolerance*.01, maxiter=maxiter,
                             M=preconditioner, callback=count_iteration)
            remainder = action(x) + b
            # MINRES stops in its preconditioned metric. Refine against the
            # physical equation before accepting a nuclear response vector.
            for _ in range(3):
                if info != 0 or np.linalg.norm(remainder) <= tolerance:
                    break
                delta, info = minres(operator, -remainder,
                    rtol=min(tolerance*.01, 1e-12), maxiter=maxiter,
                    M=preconditioner, callback=count_iteration)
                x += delta
                remainder = action(x) + b
            residual = np.linalg.norm(remainder)
            if info != 0 or residual > tolerance:
                raise RuntimeError(f"Coupled response failed: info={info}, residual={residual:g}")
            response.append(x)
            residuals.append(residual)
            iterations.append(count[0])
        correction = rhs @ np.asarray(response).T
        self.response_asymmetry = float(np.max(np.abs(correction - correction.T)))
        if self.response_asymmetry > 10*tolerance:
            raise RuntimeError("Coupled response curvature is not symmetric.")
        n = len(rhs)
        self.hessian = np.real_if_close(explicit.reshape(n, n)
                                       + .5*(correction + correction.T))
        self.gradient = np.real_if_close(gradient)
        self.response_residuals = np.asarray(residuals)
        self.response_iterations = np.asarray(iterations)
        if modes is None:
            self.hessian = self.hessian.reshape(base.mol.natom, 3, base.mol.natom, 3)
            self.gradient = self.gradient.reshape(base.mol.natom, 3)
        self.timings['coupled_response'] = perf_counter() - stamp
        self.timings['total'] = perf_counter() - started
        self.success = True
        return self
