"""Dense RPA algebra shared by spin-orbital GW and spatial singlet BSE.

The representation adapters retain their Coulomb multiplicities and exchange
index conventions. Casida is the unchanged full-matrix formulation of
Stratmann, Scuseria and Frisch, JCP 109, 8218 (1998).
No screening poles are truncated here; GW's optimized charge block remains
in the GW driver.
"""
import numpy as np
import scipy.linalg
from pyqed.phys import eig_asymm

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


def spin_matrices(gw, method='TDH'):
    '''Compute the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    from .gw import _active_energy, _spin_pair_factors, _eri
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

def singlet_matrices(gw, method='TDHF'):
    '''Get the RPA A and B matrices, using TDH, TDHF, or TDDFT.
    '''
    from .bse import _eri
    assert method in ('TDH','TDHF','TDDFT')
    nso = gw.nso
    nocc = gw.nocc
    nvir = nso - nocc

    dim_rpa = nocc*nvir
    A = np.zeros((dim_rpa,dim_rpa))
    B = np.zeros((dim_rpa,dim_rpa))

    ia = 0
    for i in range(nocc):
        for a in range(nocc,nso):
            A[ia, ia] = gw.e_mf[a] - gw.e_mf[i]
            jb = 0
            for j in range(nocc):
                for b in range(nocc,nso):
                    A[ia,jb] += 2.*_eri(gw, a, i, b, j)
                    B[ia,jb] += 2.*_eri(gw, i, a, j, b)

                    if method == 'TDHF':
                        A[ia,jb] -= _eri(gw, a, b, i, j)
                        B[ia,jb] -= _eri(gw, a, j, i, b)
                    jb += 1
            ia += 1

    assert np.allclose(A, A.transpose())
    assert np.allclose(B, B.transpose())

    return A, B

def diagonalize(A, B, *, using_tda=False, using_casida=True):
    """Solve the TDA, full nonsymmetric, or Hermitian Casida problem."""
    if using_tda:
        return scipy.linalg.eigh(A)
    if using_casida:
        return _casida_eigh(A, B)
    return eig_asymm(np.block([[A, B], [-B, -A]]))
