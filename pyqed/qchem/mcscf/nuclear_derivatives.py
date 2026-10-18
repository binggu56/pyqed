"""Streamed symmetric-frame CASSCF nuclear derivatives.

Uses the stationary symmetric-orthonormalization formulation of Helgaker
et al., JCP 84, 6266 (1986), doi:10.1063/1.450771. Algebraically equivalent
to the dense local-integral expansion; second ERI derivatives are contracted
inside shell quartets. First derivatives and AO density weights remain dense
one direction at a time. No screening, fitting, or numerical differentiation.
"""
import numpy as np


def orbital_gradient_bilinear(driver, mc, h, eri, left, right):
    """Orbital derivative between arbitrary CI vectors, including core norm.

    Uses active-block sigma derivatives; never forms a full derivative ERI
    for each orbital parameter. CI vectors need not be normalized.
    """
    from .orbopt import nonredundant_pairs
    core = mc.ncore
    result = orbital_ci_coupling(driver,mc,h,eri,left)@right
    if core:
        fcore = h+2*np.einsum('pqii->pq',eri[:,:,:core,:core])
        fcore -= np.einsum('piqi->pq',eri[:,:core,:,:core])
        core_gradient = np.asarray([4*((q<core)*fcore[p,q]-(p<core)*fcore[q,p])
            for p,q in nonredundant_pairs(core,mc.ncas,driver.nmo)])
        result += np.dot(left,right)*core_gradient
    return result


def orbital_ci_coupling(driver, mc, h, eri, c):
    """Build orbital/CI couplings using active blocks, not full derivative ERIs."""
    from .orbopt import nonredundant_pairs
    core, ncas, nmo = mc.ncore, mc.ncas, driver.nmo
    active = slice(core, core+ncas)
    pairs = nonredundant_pairs(core, ncas, nmo)
    fcore = h.copy()
    if core:
        fcore += 2*np.einsum('pqii->pq', eri[:, :, :core, :core])
        fcore -= np.einsum('piqi->pq', eri[:, :core, :, :core])
    dh = np.zeros((len(pairs), ncas, ncas))
    dg = np.zeros((len(pairs), ncas, ncas, ncas, ncas))
    for k, (p, q) in enumerate(pairs):
        for source, target, sign in ((p, q, 1.), (q, p, -1.)):
            if core <= target < core+ncas:
                a = target-core
                dh[k, a, :] += sign*fcore[source, active]
                dh[k, :, a] += sign*fcore[active, source]
                dg[k, a, :, :, :] += sign*eri[source, active, active, active]
                dg[k, :, a, :, :] += sign*eri[active, source, active, active]
                dg[k, :, :, a, :] += sign*eri[active, active, source, active]
                dg[k, :, :, :, a] += sign*eri[active, active, active, source]
            if target < core:
                dh[k] += sign*(4*eri[active, active, source, target]
                              - eri[active, source, active, target]
                              - eri[active, target, active, source])
    return driver._batched_derivative_sigma(mc, dh, dg, c)


def stationary_nuclear_terms(driver, mc, h, eri, dm1, dm2, c, modes):
    """Return gradient, explicit curvature, and coupled response RHS arrays."""
    from pyqed.qchem.basis_derivatives import (
        contracted_one_electron_curvature, contracted_eri_curvature,
        directional_one_electron_derivatives, directional_eri_derivatives)
    from pyqed.qchem.geometric import (
        _as_cartesian_mode_matrix, _one_electron_ao_to_mo, _eri_ao_to_mo,
        _one_electron_orbital_response, _eri_orbital_response,
        _active_integral_derivatives_from_full_mo, _nuclear_repulsion_hessian)
    from pyqed.qchem.mol import grad_nuc
    from pyqed.qchem.nac.sacasscf import FixedOrbitalCASCIDriver
    from .orbopt import generalized_fock, orbital_gradient

    mol, mo = driver.mol, driver.mo_coeff
    directions = (np.eye(3*mol.natom) if modes is None
                  else _as_cartesian_mode_matrix(modes, mol.natom))
    n = len(directions)
    # Dense normal modes lose atom-local integral sparsity. Build only the
    # small Cartesian gradient/curvature/response RHS arrays, then project
    # before solving electronic response. Derivative ERIs remain streamed.
    if (modes is not None and 2*n >= directions.shape[1]
            and np.count_nonzero(directions) > 3*directions.shape[1]):
        gradient, second, rhs = stationary_nuclear_terms(
            driver, mc, h, eri, dm1, dm2, c, None)
        return directions @ gradient, directions @ second @ directions.T, directions @ rhs
    fock = generalized_fock(h, eri, dm1, dm2)
    density = mo @ dm1 @ mo.T
    weighted = mo @ fock @ mo.T
    second = contracted_one_electron_curvature(mol, density, directions=directions)
    second -= contracted_one_electron_curvature(mol, weighted, 'overlap', directions)
    # Transform one density weight, not a (npert,npert,nao,nao,nao,nao) tensor.
    density2 = _eri_ao_to_mo(dm2, mo.T)
    second += contracted_eri_curvature(mol, .5*density2, directions)
    del density2
    second += directions @ _nuclear_repulsion_hessian(mol).real @ directions.T
    gradient = directions @ np.asarray(grad_nuc(mol)).reshape(-1)
    overlap1 = directional_one_electron_derivatives(mol, directions, 'overlap', order=1)
    kappa = -.5*_one_electron_ao_to_mo(overlap1, mo)
    adapter = FixedOrbitalCASCIDriver(driver.mf, mo, mc.ncore, mc.ncas)
    explicit_fock, transport_fock, rhs = [], [], []
    for a, direction in enumerate(directions):
        dh = _one_electron_ao_to_mo(directional_one_electron_derivatives(
            mol, direction[None], 'hcore', order=1)[0], mo)
        dv = _eri_ao_to_mo(directional_eri_derivatives(
            mol, direction[None], order=1, backend='native')[0], mo)
        explicit_fock.append(generalized_fock(dh, dv, dm1, dm2))
        th = _one_electron_orbital_response(h, kappa[a])
        tv = _eri_orbital_response(eri, kappa[a])
        transport_fock.append(generalized_fock(th, tv, dm1, dm2))
        dh += th
        dv += tv
        del tv
        gradient[a] += np.einsum('pq,qp', dh, dm1) + .5*np.einsum('pqrs,pqrs', dv, dm2)
        total_fock = explicit_fock[-1] + transport_fock[-1]
        orb = driver._pack_orbitals(orbital_gradient(total_fock), mc.ncore, mc.ncas, driver.nmo)
        ah, av = _active_integral_derivatives_from_full_mo(dh, dv, mc.ncore, mc.ncas)
        ci = adapter._make_active_sigma_casci(mc, ah, av).ci_sigma(c)
        ci = 2*(ci-c*np.dot(c, ci))
        rhs.append(np.concatenate((orb, ci)))
    explicit_fock, transport_fock = np.asarray(explicit_fock), np.asarray(transport_fock)
    cross = 2*np.einsum('apq,bpq->ab', explicit_fock, kappa)
    cross += np.einsum('apq,bpq->ab', transport_fock, kappa)
    second += cross + cross.T
    for a in range(n):
        for b in range(a+1):
            value = 2*np.einsum('pq,pq', fock, kappa[a]@kappa[b]+kappa[b]@kappa[a])
            second[a, b] += value
            if a != b:
                second[b, a] += value
    return gradient, second, np.asarray(rhs)
