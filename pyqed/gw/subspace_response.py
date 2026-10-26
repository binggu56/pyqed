"""Invariant-block RHF/GW/BSE response at symmetry degeneracies.

Matrix-function response adapts Higham and Relton, SIMAX 35, 1019 (2014),
doi:10.1137/130945259. The quasiparticle block uses second-order Hermitian
partitioning (Löwdin, JCP 19, 1396 (1951), doi:10.1063/1.1748067) of the
energy-dependent spectral self-energy. This extends diagonal G0W0 within
each reference degenerate orbital block, not between distinct blocks.
"""

import numpy as np

from .response import (_Jet, _FirstJet, _contract, _sqrt,
                       _cached_first, _response_stage, reference_value, _second_array,
                       _second_entries, _second_shape)


def orbital_groups(energy, nocc, tolerance):
    """Group by energy spread, never bridging the occupied/virtual gap."""
    groups = []
    for start,stop in ((0,nocc),(nocc,len(energy))):
        for i in range(start,stop):
            if i == start or abs(energy[i]-energy[groups[-1][0]]) > tolerance:
                groups.append([i])
            else:
                groups[-1].append(i)
    return tuple(np.asarray(g,dtype=int) for g in groups)


def orbital_first(first, groups):
    """Parallel gauge inside blocks; canonical response between blocks."""
    e,s,f = first['mo_energy'],first['overlap_mo'],first['fock_total_mo']
    mask = np.zeros((len(e),len(e)),dtype=bool)
    for g in groups:
        mask[np.ix_(g,g)] = True
    u = -.5*s.copy()
    rows,cols = np.where(~mask)
    u[:,rows,cols] = (f[:,rows,cols]-s[:,rows,cols]*e[cols])/(e[cols]-e[rows])
    first['full_response'] = u
    # Full matrix energies retain first-order splittings inside every block.
    e1 = f + u*e[None,:,None] - (s+u)*e[None,None,:]
    return mask,e1


def orbital_second(driver, first, second, mask, e1):
    """Second block-gauge normalization and projected eigen-equations."""
    e,u,s,f = (first[k] for k in ('mo_energy','full_response','overlap_mo','fock_total_mo'))
    n,nmo = len(u),len(e)
    rows,cols = np.where(~mask)
    u2 = np.empty((*_second_shape(n),nmo,nmo))
    for a,b,x,y in _second_entries(n):
        s2 = second['overlap_mo'][a,b]
        norm = driver._second_normalization(u[x],u[y],s[x],s[y],s2)
        value = -.5*norm
        known = (second['fock_total_mo'][a,b]+f[x]@u[y]+f[y]@u[x]
                 -(s2+s[x]@u[y]+s[y]@u[x])*e[None,:]
                 -(s[x]+u[x])@e1[y]-(s[y]+u[y])@e1[x])
        value[rows,cols] = known[rows,cols]/(e[cols]-e[rows])
        u2[a,b] = value
    return u2


@_response_stage
def inverse(matrix):
    """First/second inverse response, without eigenvector derivatives."""
    diagonal = np.count_nonzero(matrix.v) == np.count_nonzero(np.diag(matrix.v))
    def prepare():
        if not diagonal:
            return np.linalg.inv(matrix.v)
        values = np.diag(matrix.v)
        if np.any(values == 0):
            raise np.linalg.LinAlgError('Singular matrix')
        return np.diag(1/values)
    def first_response():
        inv = reference_value('inverse', prepare)
        weights = np.diag(inv)
        first = (-matrix.d*weights[:,None]*weights[None,:] if diagonal else
                 np.array([-inv@a@inv for a in matrix.d]))
        return inv, first
    inv, first = _cached_first('inverse', first_response)
    if matrix.h is None:
        return _FirstJet(inv, first)
    weights = np.diag(inv)
    def second_response(a, b, x, y):
        rhs = (matrix.h[a,b]*weights[None,:] if diagonal else matrix.h[a,b]@inv)
        rhs = rhs + matrix.d[x]@first[y] + matrix.d[y]@first[x]
        return -weights[:,None]*rhs if diagonal else -inv@rhs
    second = _second_array(matrix.n, second_response)
    return _Jet(inv,first,second)


def gap_matrix(energy, no):
    nv = len(energy.v)-no
    return (_contract('ij,ab->iajb',np.eye(no),energy[no:,no:])
            -_contract('ji,ab->iajb',energy[:no,:no],np.eye(nv))).reshape(no*nv,no*nv)


@_response_stage
def screening(energy, integrals, no, factorized):
    """Square-root/RPA response in a fixed reference pole basis.

    Off-diagonal pole-block perturbations are retained. The reference-only
    diagonalization sets coordinates, so repeated pole eigenvalues are safe.
    """
    nmo,n = len(energy.v),energy.n
    d = no*(nmo-no)
    gap = gap_matrix(energy,no)
    root = _sqrt(gap)
    if factorized:
        ov = integrals[:,:no,no:].reshape(len(integrals.v),d)
        coulomb = _contract('Pt,Pu->tu',ov,ov)
    else:
        coulomb = integrals[:no,no:,:no,no:].reshape(d,d)
    casida = _contract('ik,kl->il',_contract('ij,jk->ik',root,gap+4*coulomb),root)
    w = _sqrt(casida)
    poles,basis = _cached_first('pole_basis', lambda: reference_value('pole_basis', lambda: np.linalg.eigh(w.v)))
    w = _contract('pi,pq,qj->ij',basis,w,basis)
    # Reference roundoff off diagonal must not enter diagonal resolvent solves.
    w.v = np.diag(poles)
    if factorized:
        weights = _contract('Pt,tu,uL->PL',ov,root,basis)
        bare = _contract('Ppq,PL->pqL',integrals,weights)
    else:
        pairs = integrals[:no,no:,:,:].reshape(d,nmo,nmo)
        bare = _contract('tpq,tu,uL->pqL',pairs,root,basis)
    couplings = _contract('pqt,tL->pqL',bare,inverse(_sqrt(w)))
    return w,couplings


@_response_stage
def resolvent_action(omega, energy, poles, rhs, sign):
    """Solve the internal-orbital/pole Kronecker resolvent through order two.

    No (n_internal*n_poles)^2 matrix is formed. Its reference is diagonal;
    orbital and screening subspace response act separately on the two axes.
    """
    denominator = omega-np.diag(energy.v)[:,None]+sign*np.diag(poles.v)[None,:]
    if np.min(abs(denominator)) < 1e-10:
        raise ValueError('QP root coincides with a self-energy pole.')
    def act(e,w,value):
        return (np.einsum('rs,sLq->rLq',e,value,optimize=True)
                -sign*np.einsum('LK,rKq->rLq',w,value,optimize=True))
    def first_response():
        value = reference_value('value', lambda: rhs.v/denominator[:,:,None])
        first = np.array([(rhs.d[x]+act(energy.d[x],poles.d[x],value))/denominator[:,:,None]
                          for x in range(rhs.n)])
        return value, first
    value, first = _cached_first('resolvent', first_response)
    if rhs.h is None:
        return _FirstJet(value, first)
    second = _second_array(rhs.n, lambda a,b,x,y:
        (rhs.h[a,b]+act(energy.h[a,b],poles.h[a,b],value)
         +act(energy.d[x],poles.d[x],first[y])
         +act(energy.d[y],poles.d[y],first[x]))/denominator[:,:,None])
    return _Jet(value,first,second)


@_response_stage
def self_energy(omega, energy, poles, couplings, no, group):
    """Block Sigma through requested order, Sigma_w through first, Sigma_ww.

    Nuclear derivatives of Sigma_w and Sigma_ww are deferred until order two;
    unused third total-order derivatives are never evaluated.
    """
    sigma = type(energy).constant(np.zeros((len(group),len(group))),energy.n)
    frequency = _FirstJet.constant(np.zeros_like(sigma.v),energy.n)
    frequency2 = np.zeros_like(sigma.v)
    for region,sign in ((slice(0,no),1),(slice(no,None),-1)):
        contribution, slope, curvature = _self_energy_region(
            omega, energy[region,region], poles, couplings[region][:,group,:], sign)
        sigma += contribution
        frequency.v += slope.v
        frequency.d += slope.d
        frequency2 += curvature
    return sigma,frequency,frequency2


@_response_stage
def _self_energy_region(omega, internal, poles, couplings, sign):
    m = _contract('rpL->rLp',couplings)
    y = resolvent_action(omega,internal,poles,m,sign)
    sigma = 2*_contract('rLp,rLq->pq',m,y)
    denominator = omega-np.diag(internal.v)[:,None]+sign*np.diag(poles.v)[None,:]
    slope = _cached_first('frequency_slope', lambda: reference_value('slope', lambda: -2*np.einsum(
        'rLp,rLq->pq',m.v,y.v/denominator[:,:,None],optimize=True)))
    if internal.h is None:
        return sigma, _FirstJet(slope, np.zeros_like(sigma.d)), np.zeros_like(slope)
    # These mixed nuclear/frequency terms are needed only for the Hessian.
    # A first-order jet avoids computing unused third total-order derivatives.
    y2 = resolvent_action(omega, _FirstJet(internal.v, internal.d),
                         _FirstJet(poles.v, poles.d), _FirstJet(y.v, y.d), sign)
    frequency = -2*_contract('rLp,rLq->pq',_FirstJet(m.v,m.d),y2)
    curvature = 4*np.einsum('rLp,rLq->pq',m.v,y2.v/denominator[:,:,None],optimize=True)
    return sigma, frequency, curvature


@_response_stage
def quasiparticles(bse, energy, poles, couplings, groups, tolerance):
    """Hermitian effective Dyson blocks at symmetry-protected scalar roots.

    Within a block retain Sigma_pq, its frequency response and all mixed
    nuclear derivatives. Couplings between distinct reference blocks are
    omitted, as in diagonal G0W0. A nonscalar reference root or residue
    requires a more general matrix Dyson solution and is rejected.
    """
    nmo,n = len(energy.v),energy.n
    result = type(energy).constant(np.diag(bse.e_qp),n)
    residuals = []
    for group in groups:
        omega = float(np.mean(np.asarray(bse.e_qp)[group]))
        if np.ptp(np.asarray(bse.e_qp)[group]) > tolerance:
            raise ValueError('Degenerate HF block has split QP roots; a matrix Dyson reference is required.')
        sigma,sw,sww = self_energy(omega,energy,poles,couplings,bse.nocc,group)
        g = np.ix_(group,group)
        identity = np.eye(len(group))
        residual = np.max(abs(omega*identity-energy.v[g]-sigma.v))
        residuals.append(float(residual))
        z = 1-np.trace(sw.v)/len(group)
        curvature = np.trace(sww)/len(group)
        slope_defect = np.max(abs(sw.v-(1-z)*identity))
        curvature_defect = np.max(abs(sww-curvature*identity))
        symmetry_tol = max(2e-7,2*tolerance)
        if (residual > max(2e-7,2*tolerance) or z <= 0
                or slope_defect > symmetry_tol*max(1,abs(1-z))
                or curvature_defect > symmetry_tol*max(1,abs(curvature))):
            raise ValueError('QP subspace requires a scalar symmetry-protected reference root and residue: '
                f'group={group.tolist()}, root_residual={residual:g}, '
                f'frequency_slope_defect={slope_defect:g}, frequency_curvature_defect={curvature_defect:g}.')
        f = _cached_first('qp_first', lambda: (energy.d[:,group][:,:,group]+sigma.d)/z)
        if energy.h is None:
            for p,i in enumerate(group):
                for q,j in enumerate(group):
                    result.d[:,i,j] = f[:,p,q]
            continue
        h = (energy.h[:,:,group][:,:,:,group]+sigma.h).copy()
        for a,b,x,y in _second_entries(n):
            h[a,b] += .5*(sw.d[x]@f[y]+f[y]@sw.d[x]+sw.d[y]@f[x]+f[x]@sw.d[y])
            h[a,b] += .5*curvature*(f[x]@f[y]+f[y]@f[x])
        for p,i in enumerate(group):
            for q,j in enumerate(group):
                result.d[:,i,j] = f[:,p,q]
                result.h[:,:,i,j] = h[:,:,p,q]/z
    return result,residuals


@_response_stage
def blocks(qp, integrals, poles, couplings, no, factorized):
    """Vectorized static BSE blocks with a matrix-valued quasiparticle gap."""
    nv = len(qp.v)-no
    d = no*nv
    if factorized:
        ov = integrals[:,:no,no:]
        direct = _contract('Pia,Pjb->iajb',ov,ov)
        exchange_a = _contract('Pij,Pab->iajb',integrals[:,:no,:no],integrals[:,no:,no:])
        exchange_b = _contract('Pib,Paj->iajb',ov,integrals[:,no:,:no])
    else:
        direct = integrals[:no,no:,:no,no:]
        exchange_a = _contract('ijab->iajb',integrals[:no,:no,no:,no:])
        exchange_b = _contract('ibaj->iajb',integrals[:no,no:,no:,:no])
    inv = inverse(poles)
    screened_oo = _contract('ijL,LM->ijM',couplings[:no,:no,:],inv)
    screened_ov = _contract('ibL,LM->ibM',couplings[:no,no:,:],inv)
    screening_a = _contract('ijM,abM->iajb',screened_oo,couplings[no:,no:,:])
    screening_b = _contract('ibM,ajM->iajb',screened_ov,couplings[no:,:no,:])
    return (gap_matrix(qp,no)+(2*direct-exchange_a+4*screening_a).reshape(d,d),
            (2*direct-exchange_b+4*screening_b).reshape(d,d))
