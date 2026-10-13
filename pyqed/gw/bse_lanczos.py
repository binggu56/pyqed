"""Metric block Lanczos for stable, real molecular BSE eigenpairs."""
import time
import numpy as np
import scipy.linalg as la


def lanczos(action, diagonal, nroots, tol, max_cycle, max_space, memory, info):
    """Solve KM in the M inner product, M=A+B and K=A-B.

    Adaptation of Shao, da Jornada, Lin, Yang, Deslippe and Louie,
    SIAM J. Matrix Anal. Appl. 39, 683–711 (2018),
    https://doi.org/10.1137/16M1102641. Uses a block of nroots seeded random
    perturbations of low-gap unit vectors, full two-pass metric reorthogonalization and Rayleigh–Ritz
    eigenpair extraction. No optical quadrature, preconditioning or restart
    is used. Requires real symmetric positive M and K; projected positivity
    alone cannot certify global stability. Physical XY residuals govern
    convergence. max_space caps stored half-space vectors, not iterations.
    """
    dim = len(diagonal)
    space = min(dim, max(160, 20*nroots) if max_space is None else max_space)
    estimate = 8*(5*dim*space+20*dim*nroots+12*space*space)
    if estimate > memory:
        raise MemoryError('Lanczos workspace exceeds memory limit')
    if space < nroots:
        raise ValueError('Lanczos space is smaller than nroots')
    info.update(history=[], estimated_workspace_bytes=estimate, restarts=0)
    start = time.perf_counter()

    def apply(v, plus):
        out = action(np.vstack((v, np.zeros_like(v))))
        if np.iscomplexobj(out) and np.any(out.imag):
            raise ValueError('Lanczos requires a real operator')
        out = out.real
        return out[:dim]-out[dim:] if plus else out[:dim]+out[dim:]

    u = np.empty((dim, 0)); v = u.copy(); lu = u.copy()
    pending = np.random.default_rng(0).normal(size=(dim, nroots))
    pending /= np.maximum(abs(diagonal[:, None]), 1e-3)
    pending *= .01/np.linalg.norm(pending, axis=0)
    pending[np.argsort(diagonal)[:nroots], np.arange(nroots)] += 1.
    for cycle in range(max_cycle):
        for _ in range(2):
            coeff = v.T@pending
            pending -= u@coeff
        # Apply M after cancellation in the recurrence to avoid amplifying
        # roundoff in a separately updated metric partner.
        mp = apply(pending, True)
        gram = pending.T@mp
        if not np.allclose(gram, gram.T, atol=1e-9, rtol=1e-9):
            raise RuntimeError('Lanczos requires symmetric A+B')
        d, c = la.eigh((gram+gram.T)*.5)
        scale = max(np.max(abs(d)), np.finfo(float).tiny)
        if d[0] < -1e-10*scale:
            raise RuntimeError('Lanczos A+B is not positive definite')
        keep = d > 1e-12*scale
        if not np.any(keep):
            raise RuntimeError('Lanczos invariant subspace exhausted before convergence')
        transform = c[:, keep]/np.sqrt(d[keep])
        un, vn = pending@transform, mp@transform
        kn = apply(vn, False)
        u = np.column_stack((u, un)); v = np.column_stack((v, vn))
        lu = np.column_stack((lu, kn))
        projected = v.T@lu
        if not np.allclose(projected, projected.T, atol=1e-8, rtol=1e-9):
            raise RuntimeError('Lanczos requires symmetric A-B')
        values, z = la.eigh((projected+projected.T)*.5, subset_by_index=[0, nroots-1])
        if values[0] <= 0:
            raise RuntimeError('Lanczos A-B is not positive definite')
        roots = np.sqrt(values)
        p = (u@z)*np.sqrt(roots)
        q = (v@z)/np.sqrt(roots)
        residual = np.linalg.norm(lu@z-(u@z)*values, axis=0)/np.sqrt(2*roots)
        info.update(iterations=cycle+1, subspace=u.shape[1], residual_norms=residual)
        info['history'].append(dict(iteration=cycle+1, subspace=u.shape[1],
            seconds=time.perf_counter()-start, residual_norms=residual.tolist(),
            energies=roots.tolist()))
        if max(residual) <= .8*tol:
            return roots, np.vstack(((p+q)*.5, (p-q)*.5)), info
        if u.shape[1] >= space:
            raise RuntimeError('Lanczos max_space exhausted before convergence')
        # KM applied to the latest block; full metric reorthogonalization
        # realizes the block three-term recurrence with roundoff correction.
        pending = kn[:, :space-u.shape[1]].copy()
    raise RuntimeError('Lanczos did not converge within max_cycle')
