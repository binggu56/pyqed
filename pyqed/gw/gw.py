# -*- coding: utf-8 -*-
"""
@Author: Timothy Berkelbach
         Bing Gu

Spin-orbital G0W0

Refs
[1] Lange and Berkelbach, 2018, JCTC

"""

import numpy as np
import scipy.linalg
import sys
from scipy.optimize import newton

from pyscf.lib import logger
import pyscf.ao2mo
import pyscf
from functools import reduce


_SUPPORTED_FREQ_INTEGRATION = {
    'exact': 'exact',
    'spectral': 'exact',
    'sum_over_poles': 'exact',
    'sum-over-poles': 'exact',
}

_FREQ_INTEGRATION_TODO = {
    'contour': 'contour_deformation',
    'contour_deformation': 'contour_deformation',
    'cd': 'contour_deformation',
    'analytic_continuation': 'analytic_continuation',
    'ac': 'analytic_continuation',
    'imaginary_axis': 'imaginary_axis',
}


def _canonical_freq_int(freq_int):
    key = str(freq_int).lower().replace('-', '_')
    if key in _SUPPORTED_FREQ_INTEGRATION:
        return _SUPPORTED_FREQ_INTEGRATION[key]
    if key in _FREQ_INTEGRATION_TODO:
        raise NotImplementedError(
            f"GW frequency integration {freq_int!r} is not implemented yet. "
            "Only the exact/sum-over-poles spectral representation is currently available."
        )
    allowed = sorted(set(_SUPPORTED_FREQ_INTEGRATION) | set(_FREQ_INTEGRATION_TODO))
    raise ValueError(f"Unknown GW frequency integration {freq_int!r}. Expected one of {allowed}.")


def _symmetrize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + mat.T.conjugate())


def _positive_matrix_power(mat, power, name, thresh=1e-10):
    evals, evecs = scipy.linalg.eigh(_symmetrize(mat))
    if evals[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"{name} is not positive semidefinite; lowest eigenvalue = {evals[0]:.6e}."
        )
    evals = np.clip(evals, 0.0, None)
    if power < 0 and np.any(evals <= thresh):
        raise np.linalg.LinAlgError(
            f"{name} is numerically singular; cannot form inverse power."
        )
    powered = evals ** power
    return (evecs * powered) @ evecs.T.conjugate(), evals


def _casida_eigh(A, B, thresh=1e-10):
    """Stable dense RPA/Casida diagonalization without scipy.sqrtm."""
    a_minus_b = _symmetrize(A - B)
    a_plus_b = _symmetrize(A + B)
    sqrt_a_minus_b, _ = _positive_matrix_power(a_minus_b, 0.5, "A-B", thresh=thresh)
    casida_h = _symmetrize(sqrt_a_minus_b @ a_plus_b @ sqrt_a_minus_b)
    omega2, t = scipy.linalg.eigh(casida_h)
    if omega2[0] < -thresh:
        raise np.linalg.LinAlgError(
            f"Casida matrix has negative eigenvalue = {omega2[0]:.6e}."
        )
    omega = np.sqrt(np.clip(omega2, 0.0, None))
    order = np.argsort(omega)
    return omega[order], t[:, order]


def _nelectron(mol):
    if hasattr(mol, 'nelectron'):
        return mol.nelectron
    if hasattr(mol, 'nelec'):
        return mol.nelec
    raise AttributeError("GW requires mol.nelectron or mol.nelec.")


def _get_k(mf):
    if hasattr(mf, 'get_k'):
        return mf.get_k()
    if hasattr(mf, 'get_jk'):
        return mf.get_jk()[1]
    raise AttributeError("GW requires an HF object with get_k() or get_jk().")


def _spin_orbital_eri_from_spatial(spatial_eri):
    spatial_eri = np.asarray(spatial_eri)
    nmo = spatial_eri.shape[0]
    nso = 2 * nmo
    eri = np.zeros((nso, nso, nso, nso), dtype=spatial_eri.dtype)
    eri[0::2, 0::2, 0::2, 0::2] = spatial_eri
    eri[0::2, 0::2, 1::2, 1::2] = spatial_eri
    eri[1::2, 1::2, 0::2, 0::2] = spatial_eri
    eri[1::2, 1::2, 1::2, 1::2] = spatial_eri
    return eri


def _spin_matrix_from_spatial(spatial_mat):
    spatial_mat = np.asarray(spatial_mat)
    nmo = spatial_mat.shape[0]
    spin_mat = np.zeros((2 * nmo, 2 * nmo), dtype=spatial_mat.dtype)
    spin_mat[0::2, 0::2] = spatial_mat
    spin_mat[1::2, 1::2] = spatial_mat
    return spin_mat


def _active_energy(gw):
    energy = getattr(gw, '_qp_energy_so', None)
    if energy is None:
        return gw.e_mf
    return energy


def _spatial_to_spin_energy(gw, spatial_energy):
    spatial_energy = np.asarray(spatial_energy, dtype=float)
    if spatial_energy.shape == (gw.nso,):
        return spatial_energy.copy()
    if spatial_energy.shape != (gw.nso // 2,):
        raise ValueError(
            f"Expected {gw.nso // 2} spatial energies or {gw.nso} spin-orbital energies; "
            f"got shape {spatial_energy.shape}."
        )
    spin_energy = np.zeros(gw.nso)
    spin_energy[0::2] = spin_energy[1::2] = spatial_energy
    return spin_energy


def _spin_to_spatial_energy(gw, spin_energy):
    spin_energy = np.asarray(spin_energy, dtype=float)
    if spin_energy.shape != (gw.nso,):
        raise ValueError(f"Expected {gw.nso} spin-orbital energies; got shape {spin_energy.shape}.")
    return 0.5 * (spin_energy[0::2] + spin_energy[1::2])


def _build_spatial_eri_mo(gw, mo_coeff):
    if hasattr(gw._scf, 'get_eri_mo'):
        return gw._scf.get_eri_mo(mo_coeff=mo_coeff, notation='chem')

    nmo = mo_coeff.shape[1]
    return gw._ao2mofn(
        gw._scf.mol,
        (mo_coeff, mo_coeff, mo_coeff, mo_coeff),
        compact=False,
    ).reshape(nmo, nmo, nmo, nmo)


def _get_ao_eri_factors(gw):
    eri_factors = getattr(gw._scf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(gw.mol, 'eri_factors', None)
    if eri_factors is None:
        return None
    return np.asarray(eri_factors, dtype=float)


def _build_spatial_pair_factors_mo(gw, mo_coeff):
    eri_factors = _get_ao_eri_factors(gw)
    if eri_factors is None:
        return None
    return np.einsum('Pmn,mp,nq->Ppq', eri_factors, mo_coeff, mo_coeff, optimize=True)


def _spin_pair_factors(gw):
    if gw._spin_pair_factors is not None:
        return gw._spin_pair_factors
    if gw._pair_factors is None:
        return None

    naux, nmo, _ = gw._pair_factors.shape
    factors = np.zeros((naux, 2 * nmo, 2 * nmo), dtype=gw._pair_factors.dtype)
    factors[:, 0::2, 0::2] = gw._pair_factors
    factors[:, 1::2, 1::2] = gw._pair_factors
    gw._spin_pair_factors = factors
    return factors


def _spin_pair_factor(gw, p, q):
    if gw._pair_factors is None or (p % 2) != (q % 2):
        return None
    return gw._pair_factors[:, p // 2, q // 2]


def _eri(gw, p, q, r, s):
    if gw.eri is not None:
        return gw.eri[p, q, r, s]

    pq = _spin_pair_factor(gw, p, q)
    rs = _spin_pair_factor(gw, r, s)
    if pq is None or rs is None:
        return 0.0
    return float(np.dot(pq, rs))


def _get_hcore_ao(gw):
    if hasattr(gw._scf, 'get_hcore'):
        return np.asarray(gw._scf.get_hcore(), dtype=float)
    return np.asarray(gw.mol.hcore, dtype=float)


def _get_overlap_ao(gw):
    if hasattr(gw._scf, 'get_ovlp'):
        return np.asarray(gw._scf.get_ovlp(), dtype=float)
    return np.asarray(gw.mol.overlap, dtype=float)


def _reference_total_energy(gw):
    if hasattr(gw._scf, 'e_tot') and gw._scf.e_tot is not None:
        return float(gw._scf.e_tot)
    if hasattr(gw._scf, 'energy_tot'):
        return float(gw._scf.energy_tot())
    raise AttributeError("GW total_energy requires an SCF object with e_tot or energy_tot().")


def _make_rhf_dm(mo_coeff, nocc):
    occ_coeff = np.asarray(mo_coeff)[:, :nocc]
    return 2.0 * occ_coeff @ occ_coeff.T


def _get_j_ao(gw, dm):
    if hasattr(gw._scf, 'get_j'):
        try:
            return np.asarray(gw._scf.get_j(dm=dm), dtype=float)
        except TypeError:
            pass

    from pyqed.qchem.hf.rhf import get_jk

    eri_factors = getattr(gw._scf, 'eri_factors', None)
    if eri_factors is None:
        eri_factors = getattr(gw.mol, 'eri_factors', None)
    return np.asarray(get_jk(gw.mol, dm, eri_factors=eri_factors)[0], dtype=float)


def _mo_operator_to_ao(gw, mo_coeff, op_mo):
    overlap = _get_overlap_ao(gw)
    return overlap @ mo_coeff @ op_mo @ mo_coeff.T @ overlap


def _set_rhf_orbitals(gw, mo_energy, mo_coeff, v_static_spatial=None):
    gw.mo_coeff = np.asarray(mo_coeff, dtype=float)
    gw.e_mf = _spatial_to_spin_energy(gw, mo_energy)

    if v_static_spatial is None:
        v_static_spatial = gw.v_mf[0::2, 0::2]
    gw.v_mf = _spin_matrix_from_spatial(v_static_spatial)

    gw._pair_factors = _build_spatial_pair_factors_mo(gw, gw.mo_coeff)
    gw._spin_pair_factors = None
    if gw._pair_factors is None:
        spatial_eri = _build_spatial_eri_mo(gw, gw.mo_coeff)
        gw.eri = _spin_orbital_eri_from_spatial(spatial_eri).real
    else:
        gw.eri = None
    gw._M = None
    gw._sigma_x_matrix = None
    gw._qp_energy_so = gw.e_mf.copy()


def g0(gw, omega):
    '''Return the 0th order GF matrix [G0]_{pq} in the basis of
    single-particle orbitals (MF eigenvectors).'''

    g0 = np.zeros((gw.nso,gw.nso), dtype=np.complex128)

    energy = _active_energy(gw)
    for p in range(gw.nso):
        if p < gw.nocc: sgn = -1
        else: sgn = +1
        g0[p,p] = 1.0/(omega - energy[p] + 1j*sgn*gw.eta)
    return g0


def rpa_AB_matrices(gw, method='TDH'):
    '''Compute the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    assert method in ('TDH','TDHF','TDDFT')
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa, dim_rpa))
    B = np.zeros((dim_rpa, dim_rpa))
    energy = _active_energy(gw)

    if method == 'TDH':
        occ_idx = np.repeat(np.arange(nocc), nvir)
        vir_idx = np.tile(np.arange(nocc, nso), nocc)
        delta = energy[vir_idx] - energy[occ_idx]
        if gw.eri is not None:
            A += gw.eri[
                vir_idx[:, None],
                occ_idx[:, None],
                occ_idx[None, :],
                vir_idx[None, :],
            ]
            B += gw.eri[
                vir_idx[:, None],
                occ_idx[:, None],
                vir_idx[None, :],
                occ_idx[None, :],
            ]
        else:
            pair_ai = _spin_pair_factors(gw)[:, vir_idx, occ_idx].T
            coulomb = pair_ai @ pair_ai.T
            A += coulomb
            B += coulomb
        A[np.diag_indices(dim_rpa)] += delta
        assert np.allclose(A, A.transpose())
        assert np.allclose(B, B.transpose())
        return A, B

    ai = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            A[ai,ai] = energy[a] - energy[i]
            bj = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ai,bj] += _eri(gw, a, i, j, b)
                    B[ai,bj] += _eri(gw, a, i, b, j)
                    if method == 'TDHF':
                        A[ai,bj] -= _eri(gw, a, b, j, i)
                        B[ai,bj] -= _eri(gw, a, j, b, i)
                    bj += 1
            ai += 1

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    return A, B

def rpa(gw, using_tda=False, using_casida=True, method='TDH'):
    r'''Get the RPA eigenvalues and eigenvectors.

    The RPA computation is required to construct the dielectric function, i.e. screened
    Coloumb interaction.

    Q^\dagger = \sum_{ia} X_{ia} a^+ i - Y_{ia} i^+ a
    Leads to the RPA eigenvalue equations:
      [ A  B ][X] = omega [ 1  0 ][X]
      [ B  A ][Y]         [ 0 -1 ][Y]
    which is equivalent to
      [ A  B ][X] = omega [ 1  0 ][X]
      [-B -A ][Y] =       [ 0  1 ][Y]

    See, e.g. Stratmann, Scuseria, and Frisch,
              J. Chem. Phys., 109, 8218 (1998)
    '''
    A, B = rpa_AB_matrices(gw, method=method)

    if using_tda:
        ham_rpa = A
        e, x = eig(ham_rpa)
        return e, x
    else:
        if not using_casida:
            ham_rpa = np.array(np.bmat([[A,B],[-B,-A]]))
            e, xy = eig_asymm(ham_rpa)
            return e, xy
        else:
            return _casida_eigh(A, B)


def rpa_correlation_energy(gw, mo_energy=None, method='direct', use_qp=False):
    '''Direct RPA correlation energy from the dense plasmon formula.

    The direct RPA expression is evaluated with the TDH response matrices,
    regardless of the GW object's screening setting:

        E_c^RPA = 1/2 [sum_p omega_p - Tr(A)]

    where omega_p are the positive Casida RPA roots.
    '''
    mode = str(method).lower()
    if mode not in {'direct', 'drpa', 'rpa'}:
        raise ValueError("Only direct RPA correlation energy is implemented.")

    saved_qp_energy = getattr(gw, '_qp_energy_so', None)
    try:
        if mo_energy is not None:
            gw._qp_energy_so = _spatial_to_spin_energy(gw, mo_energy)
        elif use_qp:
            if gw.e_qp is None:
                raise ValueError("use_qp=True requires GW quasiparticle energies in gw.e_qp.")
            gw._qp_energy_so = _spatial_to_spin_energy(gw, gw.e_qp)
        else:
            gw._qp_energy_so = gw.e_mf.copy()

        A, B = rpa_AB_matrices(gw, method='TDH')
        omega, _ = _casida_eigh(A, B)
        e_corr = 0.5 * (float(np.sum(omega)) - float(np.trace(A)))
    finally:
        gw._qp_energy_so = saved_qp_energy

    gw.e_corr = e_corr
    return e_corr


def get_m_rpa(gw, e_rpa, t_rpa):
    r'''Get the (intermediate) M_{pq,L} tensor needed to calculate the self-energy.

    M_{pq,L} = \sum_{ia} ( (eps_a-eps_i)/erpa_L )^{1/2} T_{ai,L} (ai|pq)
    '''
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc
    energy = _active_energy(gw)
    t_by_e = t_rpa.copy()
    for L in range(len(e_rpa)):
        t_by_e[:,L] /= np.sqrt(e_rpa[L])
    sqrt_eps = np.zeros(nocc*nvir)
    if gw._pair_factors is None:
        eri_product = np.zeros((nocc*nvir, nso, nso))
    else:
        pair_ai = np.zeros((nocc*nvir, gw._pair_factors.shape[0]))
    ai = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            sqrt_eps[ai] = np.sqrt(energy[a]-energy[i])
            if gw._pair_factors is None:
                eri_product[ai,:,:] = gw.eri[a,i,:,:]
            else:
                pair = _spin_pair_factor(gw, a, i)
                if pair is not None:
                    pair_ai[ai, :] = pair
            ai += 1
    if gw._pair_factors is None:
        M = np.einsum('a,al,apq->pql', sqrt_eps, t_by_e, eri_product, optimize=True)
    else:
        weighted_pairs = np.einsum('a,al,aP->Pl', sqrt_eps, t_by_e, pair_ai, optimize=True)
        M = np.einsum('Pl,Ppq->pql', weighted_pairs, _spin_pair_factors(gw), optimize=True)
    return M


def _sigma_x_matrix(gw):
    sigma_x = getattr(gw, '_sigma_x_matrix', None)
    if sigma_x is not None:
        return sigma_x

    nocc = gw.nocc
    if gw.eri is not None:
        sigma_x = -np.einsum('piiq->pq', gw.eri[:, :nocc, :nocc, :], optimize=True)
    else:
        factors = _spin_pair_factors(gw)
        sigma_x = -np.einsum('Ppi,Piq->pq', factors[:, :, :nocc], factors[:, :nocc, :], optimize=True)

    gw._sigma_x_matrix = sigma_x
    return sigma_x


def sigma(gw, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
    '''
    self energy sigma_{pq} = i [GW]_{pq}
    '''
    if not isinstance(omegas, (list,tuple,np.ndarray)):
        single_point = True
        omegas = [omegas]
    else:
        single_point = False

    if gw._M is None:
        gw._M = get_m_rpa(gw, e_rpa, t_rpa)

    nso = gw.nso
    nocc = gw.nocc
    energy = _active_energy(gw)
    omega = np.asarray(omegas, dtype=float)

    occ_weight = gw._M[:nocc, q, :] * gw._M[:nocc, p, :]
    vir_weight = gw._M[nocc:nso, q, :] * gw._M[nocc:nso, p, :]
    occ_denom = (
        omega[:, None, None]
        - energy[:nocc][None, :, None]
        + e_rpa[None, None, :]
        - 1j * gw.eta
    )
    vir_denom = (
        omega[:, None, None]
        - energy[nocc:nso][None, :, None]
        - e_rpa[None, None, :]
        + vir_sgn * 1j * gw.eta
    )
    sigma_c = np.sum(occ_weight[None, :, :] / occ_denom, axis=(1, 2))
    sigma_c += np.sum(vir_weight[None, :, :] / vir_denom, axis=(1, 2))
    sigma_x = np.full(omega.shape, _sigma_x_matrix(gw)[p, q], dtype=np.complex128)

    if single_point:
        return sigma_c[0], sigma_x[0]
    else:
        return list(sigma_c), list(sigma_x)

def _solve_qp_energies(gw, e_rpa, t_rpa, initial_so_energy):
    egw = np.zeros(int(gw.nso/2))

    for p in range(0,gw.nso,2):

        def quasiparticle(omega):
            
            sigma_c_ppw, sigma_x_ppw = sigma(gw, p, p, omega, e_rpa, t_rpa)
            
            sigma_ppw = sigma_c_ppw + sigma_x_ppw
            
            return omega - gw.e_mf[p] - (sigma_ppw.real - gw.v_mf[p,p])

        try:
            egw[int(p/2)] = newton(quasiparticle, initial_so_energy[p], tol=1e-6, maxiter=100)

        except RuntimeError:
            print("Newton-Raphson unconverged, setting GW eval to input eval.")
            egw[int(p/2)] = initial_so_energy[p]
        
        print(egw[int(p/2)])
    
    return egw


def kernel(gw, so_energy, so_coeff, verbose=logger.NOTE):
    '''Get the GW-corrected spatial orbital energies.

    Note: Works in spin-orbitals but returns energies for spatial orbitals.

    Args:
        gw : instance of :class:`GW`
        so_energy : (nso,) ndarray
        so_coeff : (nso,nso) ndarray

    Returns:
        egw : (nso/2,) ndarray
            The GW-corrected spatial orbital energies.
    '''
    print("# --- Performing RPA calculation ...")
    gw._qp_energy_so = _spatial_to_spin_energy(gw, so_energy)
    gw._M = None
    e_rpa, t_rpa = rpa(gw, method=gw.screening)

    print("RPA eigenvalues = ", e_rpa)

    print("done.")
    print("# --- Calculating GW QP corrections ...")

    egw = _solve_qp_energies(gw, e_rpa, t_rpa, gw._qp_energy_so)
    
    print("done.")

    return egw


def evgw_kernel(
    gw,
    so_energy,
    so_coeff,
    max_cycle=50,
    conv_tol=1e-7,
    damping=1.0,
    update_screening=True,
    verbose=logger.NOTE,
):
    '''Eigenvalue-only GW with fixed orbitals and fixed two-electron integrals.'''
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in the interval (0, 1].")

    current = np.asarray(so_energy, dtype=float).copy()
    e_rpa = None
    t_rpa = None
    fixed_M = None
    history = []

    print("# --- Performing eigenvalue-only GW iterations ...")
    for cycle in range(1, max_cycle + 1):
        gw._qp_energy_so = _spatial_to_spin_energy(gw, current)
        gw._M = None

        if update_screening or e_rpa is None:
            e_rpa, t_rpa = rpa(gw, method=gw.screening)
            fixed_M = None
            if not update_screening:
                fixed_M = get_m_rpa(gw, e_rpa, t_rpa)

        if fixed_M is not None:
            gw._M = fixed_M

        print(f"# --- evGW cycle {cycle} QP corrections ...")
        updated = _solve_qp_energies(gw, e_rpa, t_rpa, gw._qp_energy_so)
        mixed = current + damping * (updated - current)
        delta = float(np.max(np.abs(mixed - current)))
        history.append({
            "cycle": cycle,
            "delta": delta,
            "energy": mixed.copy(),
        })
        print(f"evGW cycle {cycle}: max |dE| = {delta:.6e} Ha")

        current = mixed
        if delta < conv_tol:
            gw.converged = True
            break
    else:
        gw.converged = False

    gw.evgw_history = history
    gw.e_rpa = e_rpa
    gw._qp_energy_so = _spatial_to_spin_energy(gw, current)
    print("# --- evGW iterations done. converged =", gw.converged)
    return current


def qs_static_potential(gw, e_rpa, t_rpa):
    '''Build the Hermitian static quasiparticle self-energy in the current MO basis.'''
    energy = _active_energy(gw)
    nso = gw.nso
    sigma_col_energy = np.zeros((nso, nso), dtype=np.complex128)

    for p in range(nso):
        for q in range(nso):
            sigma_c_pq, sigma_x_pq = sigma(gw, p, q, energy[q], e_rpa, t_rpa)
            sigma_col_energy[p, q] = sigma_c_pq + sigma_x_pq

    # MOLGW/Kotani hermitianization: build Sigma_pq(e_q), then symmetrize.
    v_qs = 0.5 * (sigma_col_energy + sigma_col_energy.T.conjugate()).real

    return 0.5 * (v_qs + v_qs.T)


def qsgw_kernel(
    gw,
    so_energy,
    so_coeff,
    max_cycle=50,
    conv_tol=1e-7,
    damping=0.5,
    verbose=logger.NOTE,
):
    '''Dense RHF quasiparticle self-consistent GW reference implementation.

    The updated static potential is
        V_qs[p,q] = 1/2 Re[Sigma[p,q](eps_p) + Sigma[q,p](eps_q)^*].
    Orbitals are rediagonalized in the current MO basis and two-electron
    integrals are rebuilt in the rotated MO basis each macroiteration.
    '''
    if not (0.0 < damping <= 1.0):
        raise ValueError("damping must be in the interval (0, 1].")

    current_energy = _spin_to_spatial_energy(gw, _spatial_to_spin_energy(gw, so_energy))
    current_coeff = np.asarray(so_coeff, dtype=float).copy()
    current_v = gw.v_mf[0::2, 0::2].copy()
    history = []

    print("# --- Performing quasiparticle self-consistent GW iterations ...")
    for cycle in range(1, max_cycle + 1):
        _set_rhf_orbitals(gw, current_energy, current_coeff, current_v)

        e_rpa, t_rpa = rpa(gw, method=gw.screening)
        gw.e_rpa = e_rpa

        v_qs_so = qs_static_potential(gw, e_rpa, t_rpa)
        v_qs = 0.5 * (v_qs_so[0::2, 0::2] + v_qs_so[1::2, 1::2])
        v_qs = 0.5 * (v_qs + v_qs.T)
        mixed_v = current_v + damping * (v_qs - current_v)

        dm = _make_rhf_dm(current_coeff, gw.nocc // 2)
        h_qs_ao = _get_hcore_ao(gw) + _get_j_ao(gw, dm)
        h_qs_ao = h_qs_ao + _mo_operator_to_ao(gw, current_coeff, mixed_v)
        h_qs_ao = 0.5 * (h_qs_ao + h_qs_ao.T)
        next_energy, next_coeff = scipy.linalg.eigh(h_qs_ao, _get_overlap_ao(gw))
        next_v = next_coeff.T @ _mo_operator_to_ao(gw, current_coeff, mixed_v) @ next_coeff
        next_v = 0.5 * (next_v + next_v.T)

        energy_delta = float(np.max(np.abs(next_energy - current_energy)))
        potential_delta = float(np.max(np.abs(mixed_v - current_v)))
        history.append({
            "cycle": cycle,
            "energy_delta": energy_delta,
            "potential_delta": potential_delta,
            "energy": next_energy.copy(),
        })
        print(
            f"qsGW cycle {cycle}: max |dE| = {energy_delta:.6e} Ha, "
            f"max |dV| = {potential_delta:.6e} Ha"
        )

        current_energy = next_energy
        current_coeff = next_coeff
        current_v = next_v
        if max(energy_delta, potential_delta) < conv_tol:
            gw.converged = True
            break
    else:
        gw.converged = False

    _set_rhf_orbitals(gw, current_energy, current_coeff, current_v)
    gw.qsgw_history = history
    gw.mo_coeff_qsgw = current_coeff
    gw.v_qsgw = current_v
    print("# --- qsGW iterations done. converged =", gw.converged)
    return current_energy



def eig(h, s=None):
    e, c = scipy.linalg.eigh(h,s)
    return e, c


def eig_asymm(h):
    '''Diagonalize a real, *asymmetrix* matrix and return sorted results.

    Return the eigenvalues and eigenvectors (column matrix)
    sorted from lowest to highest eigenvalue.
    '''
    e, c = np.linalg.eig(h)
    if np.allclose(e.imag, 0*e.imag):
        e = np.real(e)
    else:
        print("WARNING: Eigenvalues are complex, will be returned as such.")

    idx = e.argsort()
    e = e[idx]
    c = c[:,idx]

    return e, c


def is_positive_def(A):
    try:
        np.linalg.cholesky(A)
        return True
    except np.linalg.LinAlgError:
        return False


class GW(object):
    __array_priority__ = 1000

    def __init__(self, mf, ao2mofn=pyscf.ao2mo.outcore.general_iofree,
                 screening='TDH', eta=1e-2, freq_int='exact'):

        assert screening in ('TDH', 'TDHF', 'TDDFT')

        self.mol = mf.mol
        self._scf = mf
        self._ao2mofn = ao2mofn
        self.freq_int = _canonical_freq_int(freq_int)
        self.verbose = getattr(self.mol, 'verbose', getattr(mf, 'verbose', 0))
        self.stdout = getattr(self.mol, 'stdout', getattr(mf, 'stdout', sys.stdout))
        self.max_memory = getattr(mf, 'max_memory',
                                  getattr(self.mol, 'max_memory', 4000))

        self.nocc = _nelectron(self.mol)
        try:
            # DFT
            mf.xc = mf.xc
            v_mf = mf.get_veff() - mf.get_j()
        except AttributeError:
            # HF
            v_mf = -_get_k(mf)
        if mf.mo_occ[0] == 2:
            # RHF, convert to spin-orbitals
            nso = 2*len(mf.mo_energy)
            self.nso = nso
            self.mo_coeff = np.asarray(mf.mo_coeff, dtype=float)
            self.e_mf = np.zeros(nso)
            self.e_mf[0::2] = self.e_mf[1::2] = mf.mo_energy
            b = np.zeros((nso//2,nso))
            b[:,0::2] = b[:,1::2] = mf.mo_coeff
            self.v_mf = 0.5 * reduce(np.dot, (b.T, v_mf, b))
            self.v_mf[::2,1::2] = self.v_mf[1::2,::2] = 0
            self.eri = None
            self._pair_factors = None
            self._spin_pair_factors = None

            _set_rhf_orbitals(self, mf.mo_energy, mf.mo_coeff, self.v_mf[0::2, 0::2])
            if self.eri is not None:
                print("Imag part of ERIs =", np.linalg.norm(self.eri.imag))
            else:
                print("Using factorized MO ERIs with %d factors" % (self._pair_factors.shape[0],))
        else:
            # ROHF or UHF, these are already spin-orbitals
            print("\n*** Only supporting restricted calculations right now! ***\n")
            raise NotImplementedError
            nso = len(mf.mo_energy)
            self.nso = nso
            self.e_mf = mf.mo_energy
            b = mf.mo_coeff
            self.v_mf = reduce(np.dot, (b.T, v_mf, b))
            eri = ao2mofn(mf.mol, (b,b,b,b),
                          compact=False).reshape(nso,nso,nso,nso)
            self.eri = eri

        print("There are %d spin-orbitals"%(self.nso))

        self.screening = screening
        self.eta = eta
        self._M = None
        self._sigma_x_matrix = None

        self.e = None
        self._e_qp = None
        self.e_rpa = None
        self._qp_energy_so = None
        self.evgw_history = []
        self.qsgw_history = []
        self.scgw_result = None
        self.mo_coeff_qsgw = None
        self.v_qsgw = None
        self.converged = False
        self.e_corr = None
        self.e_tot = None
        self.info = None
        self.method = None

    @property
    def e_qp(self):
        """GW quasiparticle energies.

        ``egw`` is kept as a backward-compatible alias; prefer ``e_qp`` in new
        code.  ``e`` mirrors ``e_qp`` for the GW driver result.
        """
        return self._e_qp

    @e_qp.setter
    def e_qp(self, value):
        self._e_qp = value
        self.e = value

    @property
    def egw(self):
        return self.e_qp

    @egw.setter
    def egw(self, value):
        self.e_qp = value

    def _as_energy_array(self):
        if self.e_qp is None:
            raise ValueError("GW quasiparticle energies are not available. Run GW first.")
        return np.asarray(self.e_qp)

    def __array__(self, dtype=None):
        return np.asarray(self._as_energy_array(), dtype=dtype)

    def __len__(self):
        return len(self._as_energy_array())

    def __iter__(self):
        return iter(self._as_energy_array())

    def __getitem__(self, key):
        return self._as_energy_array()[key]

    def __mul__(self, other):
        return self._as_energy_array() * other

    def __rmul__(self, other):
        return other * self._as_energy_array()

    def __add__(self, other):
        return self._as_energy_array() + other

    def __radd__(self, other):
        return other + self._as_energy_array()

    def __sub__(self, other):
        return self._as_energy_array() - other

    def __rsub__(self, other):
        return other - self._as_energy_array()

    def __truediv__(self, other):
        return self._as_energy_array() / other

    def __neg__(self):
        return -self._as_energy_array()

    def run(self, mo_energy=None, mo_coeff=None, method='g0w0', **kwargs):
        
        if mo_coeff is None:
            mo_coeff = self._scf.mo_coeff
        if mo_energy is None:
            mo_energy = self._scf.mo_energy

        method = method.lower()
        if method in ('g0w0', 'gw', 'oneshot', 'one-shot'):
            self.e_qp = kernel(self, mo_energy, mo_coeff, verbose=self.verbose)
            self.converged = True
            self.method = 'g0w0'
        elif method in ('evgw', 'ev-gw', 'eigenvalue-only'):
            self.e_qp = evgw_kernel(self, mo_energy, mo_coeff, verbose=self.verbose, **kwargs)
            self.method = 'evgw'
        elif method in ('qsgw', 'qs-gw', 'quasiparticle-self-consistent'):
            self.e_qp = qsgw_kernel(self, mo_energy, mo_coeff, verbose=self.verbose, **kwargs)
            self.method = 'qsgw'
        elif method in ('scgw0', 'sc-gw0', 'self-consistent-gw0'):
            from pyqed.gw.scgw import SCGW

<<<<<<< HEAD
            init_keys = {
                "nfreq",
                "wmax",
                "beta",
                "adjust_mu",
                "target_nelec",
                "density_nfreq",
                "grid",
            }
=======
            init_keys = {"nfreq", "wmax", "beta", "adjust_mu", "target_nelec", "density_nfreq"}
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
            init_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in init_keys}
            kwargs.setdefault("update_screening", False)
            self.scgw_result = SCGW(
                self._scf,
                screening=self.screening,
                eta=self.eta,
                **init_kwargs,
            ).run(verbose=self.verbose, **kwargs)
            self.e_qp = self.scgw_result.e_qp
            self.converged = self.scgw_result.converged
            self.method = 'scgw0'
        elif method in ('scgw', 'sc-gw', 'self-consistent-gw'):
            from pyqed.gw.scgw import SCGW

<<<<<<< HEAD
            init_keys = {
                "nfreq",
                "wmax",
                "beta",
                "adjust_mu",
                "target_nelec",
                "density_nfreq",
                "grid",
            }
=======
            init_keys = {"nfreq", "wmax", "beta", "adjust_mu", "target_nelec", "density_nfreq"}
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
            init_kwargs = {key: kwargs.pop(key) for key in list(kwargs) if key in init_keys}
            kwargs.setdefault("update_screening", True)
            self.scgw_result = SCGW(
                self._scf,
                screening=self.screening,
                eta=self.eta,
                **init_kwargs,
            ).run(verbose=self.verbose, **kwargs)
            self.e_qp = self.scgw_result.e_qp
            self.converged = self.scgw_result.converged
            self.method = 'scgw'
        else:
            raise ValueError(
                f"Unknown GW method {method!r}. "
                "Use 'g0w0', 'evgw', 'qsgw', 'scgw0', or 'scgw'."
            )
        self.info = {
            "method": self.method,
            "frequency_integration": self.freq_int,
            "converged": self.converged,
            "uses_factorized_eris": self.eri is None and self._pair_factors is not None,
        }
        if self.scgw_result is not None:
            self.info["scgw"] = self.scgw_result.info
        logger.log(self, 'GW bandgap = %.15g', self.e_qp[self.nocc//2]-self.e_qp[self.nocc//2-1])
        return self

    def g0w0(self, mo_energy=None, mo_coeff=None, **kwargs):
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='g0w0', **kwargs)

    def evgw(self, mo_energy=None, mo_coeff=None, **kwargs):
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='evgw', **kwargs)

    def gnw0(self, mo_energy=None, mo_coeff=None, **kwargs):
        kwargs.setdefault('update_screening', False)
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='evgw', **kwargs)

    def qsgw(self, mo_energy=None, mo_coeff=None, **kwargs):
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='qsgw', **kwargs)

    def scgw0(self, mo_energy=None, mo_coeff=None, **kwargs):
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='scgw0', **kwargs)

    def scgw(self, mo_energy=None, mo_coeff=None, **kwargs):
        return self.run(mo_energy=mo_energy, mo_coeff=mo_coeff, method='scgw', **kwargs)

    def bse(self, **kwargs):
        """Construct a BSE driver from this GW reference."""
        from pyqed.gw.bse import BSE

        return BSE(self, **kwargs)

    def tda(self, **kwargs):
        """Construct a TDA-BSE driver from this GW reference."""
        from pyqed.gw.bse import TDA

        return TDA(self, **kwargs)

    def rpa_correlation_energy(self, mo_energy=None, method='direct', use_qp=False):
        return rpa_correlation_energy(self, mo_energy=mo_energy, method=method, use_qp=use_qp)

    def total_energy(self, method='rpa', **kwargs):
        mode = str(method).lower()
        if mode not in {'rpa', 'drpa', 'direct_rpa'}:
            raise ValueError("Only method='rpa' is implemented for GW total energies.")
        e_corr = self.rpa_correlation_energy(method='direct', **kwargs)
        self.e_tot = _reference_total_energy(self) + e_corr
        return self.e_tot

    def sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn=1):
        return sigma(self, p, q, omegas, e_rpa, t_rpa, vir_sgn)

    def g0(self, omega):
        return g0(self, omega)

    def get_m_rpa(self, e_rpa, t_rpa):
        return get_m_rpa(self, e_rpa, t_rpa)

    def rpa(self, using_tda=False, using_casida=True, method='TDH'):
        return rpa(self, using_tda, using_casida, method)

    def rpa_AB_matrices(self, method='TDH'):
        return rpa_AB_matrices(self, method)

if __name__ == '__main__':
    from pyscf import scf, gto
    mol = gto.Mole()
    mol.verbose = 1
    #mol.atom = [['Ne' , (0., 0., 0.)]]
    #mol.basis = {'Ne': '6-31G'}
    # This is from G2/97 i.e. MP2/6-31G*
    mol.atom = [['H' , (0.,      0., 0.)],
                ['F', (1.1, 0., 0.)]]
                # ['F' , (0.91, 0., 0.)]]
    mol.basis = '631g'
    mol.build()
    mf = scf.RHF(mol)
    #print(mf.scf())
    mf.kernel()

    gw = GW(mf, screening='TDHF')
    egw = gw.run()
    print('HF    vs.   GW ')
    for emf, eqp in zip(mf.mo_energy, egw):
        print("%0.6f %0.6f"%(emf, eqp))

    nocc = mol.nelectron//2
    ehomo = egw[nocc-1]
    print("GW -IP = GW HOMO =", ehomo, "au =", ehomo*27.211, "eV")
