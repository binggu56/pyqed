"""RHF and pure-LDA orbital operators with shared adjoint solves."""
import numpy as np
from scipy.linalg import lu_factor, lu_solve
from scipy.sparse.linalg import LinearOperator, gmres


class OrbitalResponse:
    """Orbital operator in (virtual,occupied) order, with shared transpose solves.

    Adapts the auxiliary-functional organization of Furche and Ahlrichs,
    JCP 117, 7433 (2002), doi:10.1063/1.1508368. Real closed-shell pure LDA;
    The constructor builds pure-LDA CPKS; ``rhf`` builds dense RHF CPHF.
    Small LDA systems use dense orbital-space LU; larger systems use residual-
    controlled, gap-preconditioned GMRES without an orbital Hessian matrix.
    """

    def __init__(self, mf, eri, fxc, block_size=2048, dense_limit=512):
        self.mf, self.eri, self.fxc = mf, eri, fxc
        self.block_size = block_size
        self.occ, self.vir = np.flatnonzero(mf.mo_occ > 0), np.flatnonzero(mf.mo_occ == 0)
        self.co, self.cv = mf.mo_coeff[:, self.occ], mf.mo_coeff[:, self.vir]
        self.shape = (len(self.vir), len(self.occ))
        gap = mf.mo_energy[self.vir, None] - mf.mo_energy[self.occ]
        self.gap = gap
        self.matrix = None
        if gap.size > dense_limit:
            return
        self.matrix = np.diag(gap.ravel())
        self.matrix += 4 * eri.transform(self.cv, self.co, self.cv, self.co).reshape(gap.size, gap.size)
        for section, phi, weight in self.blocks():
            pairs = np.einsum('ga,gi->gai', phi @ self.cv, phi @ self.co).reshape(len(phi), -1)
            self.matrix += 4 * pairs.T @ ((weight * fxc[section])[:, None] * pairs)
        if np.linalg.cond(self.matrix) > 1e12:
            raise ValueError('Singular CPKS response: ground-state reference is unstable or degenerate.')
        self.factor = lu_factor(self.matrix)

    @classmethod
    def rhf(cls, mf):
        """Dense RHF orbital operator, reusing the analytic CPHF builder.

        Uses the RHFHessian sign convention (occupied minus virtual gaps).
        The shared solve method checks both ordinary and transpose residuals.
        """
        from .hf.hessian import RHFHessian
        obj = cls.__new__(cls)
        obj.mf = mf
        obj.occ, obj.vir = np.flatnonzero(mf.mo_occ > 0), np.flatnonzero(mf.mo_occ == 0)
        obj.co, obj.cv = mf.mo_coeff[:, obj.occ], mf.mo_coeff[:, obj.vir]
        obj.shape = (len(obj.vir), len(obj.occ))
        obj.gap = mf.mo_energy[obj.occ] - mf.mo_energy[obj.vir, None]
        driver = RHFHessian(mf)
        driver._require_scf()
        obj.matrix = driver._build_cphf_matrix(mf.mo_coeff, obj.co, obj.occ,
            obj.vir, mf.mo_energy[obj.occ], mf.mo_energy[obj.vir])
        if np.linalg.cond(obj.matrix) > 1e12:
            raise ValueError('Singular RHF orbital response.')
        obj.factor = lu_factor(obj.matrix)
        return obj

    def blocks(self):
        grid = self.mf.grid
        for start in range(0, len(grid.weights), self.block_size):
            section = slice(start, start+self.block_size)
            yield section, np.asarray(grid.ao[section]), np.asarray(grid.weights[section])

    def density_response(self, density):
        result = self.eri.j(density)
        for section, phi, weight in self.blocks():
            rho = np.einsum('gu,uv,gv->g', phi, density, phi, optimize=True)
            result += phi.T @ ((weight*self.fxc[section]*rho)[:, None]*phi)
        return result

    def apply(self, vectors):
        v = np.asarray(vectors)
        if self.matrix is None:
            return self._action(v)
        return (v.reshape(-1, self.matrix.shape[0]) @ self.matrix.T).reshape(v.shape)

    def apply_transpose(self, vectors):
        v = np.asarray(vectors)
        if self.matrix is None:
            # Real singlet Coulomb/LDA response is self-adjoint in this metric.
            return self._action(v)
        return (v.reshape(-1, self.matrix.shape[0]) @ self.matrix).reshape(v.shape)

    def _action(self, vectors):
        batch = vectors.reshape(-1, *self.shape)
        result = batch*self.gap
        for start in range(0, len(batch), 8):
            v = batch[start:start+8]
            density = 2*np.einsum('pa,oai,qi->opq', self.cv, v, self.co, optimize=True)
            density += density.transpose(0, 2, 1).copy()
            fock = np.array([self.eri.j(d) for d in density])
            for section, phi, weight in self.blocks():
                rho = np.einsum('gp,opq,gq->og', phi, density, phi, optimize=True)
                fock += np.einsum('gp,og,gq->opq', phi,
                    rho*(weight*self.fxc[section]), phi, optimize=True)
            result[start:start+8] += np.einsum('pa,opq,qi->oai', self.cv, fock, self.co, optimize=True)
        return result.reshape(vectors.shape)

    def solve(self, rhs, transpose=True):
        rhs = np.asarray(rhs)
        flat = rhs.reshape(-1, self.gap.size)
        self.iterations = []
        if self.matrix is not None:
            solution = lu_solve(self.factor, flat.T, trans=int(transpose)).T
        else:
            action = self.apply_transpose if transpose else self.apply
            operator = LinearOperator((self.gap.size, self.gap.size),
                matvec=lambda v: action(v.reshape(self.shape)).ravel(), dtype=float)
            diagonal = self.gap.ravel()
            if np.min(abs(diagonal)) < 1e-12:
                raise ValueError('Singular orbital-gap preconditioner.')
            preconditioner = LinearOperator(operator.shape, matvec=lambda v: v/diagonal, dtype=float)
            solution = []
            for b in flat:
                count = []
                value, status = gmres(operator, b, M=preconditioner, rtol=1e-11, atol=1e-13,
                    restart=min(80, self.gap.size), maxiter=100,
                    callback=lambda residual: count.append(residual), callback_type='pr_norm')
                if status:
                    raise RuntimeError('Orbital adjoint GMRES did not converge.')
                self.iterations.append(len(count))
                solution.append(value)
            solution = np.asarray(solution)
        action = self.apply_transpose if transpose else self.apply
        residual = action(solution.reshape(rhs.shape)).reshape(flat.shape)-flat
        self.residuals = np.linalg.norm(residual, axis=1)/np.maximum(np.linalg.norm(flat, axis=1), 1.)
        if not np.all(np.isfinite(solution)) or np.max(self.residuals, initial=0.) > 1e-10:
            raise RuntimeError('Orbital adjoint failed its residual check.')
        return solution.reshape(rhs.shape)
