"""Spin-free one-electron X2C Hamiltonian."""

from functools import reduce

import numpy as np
from scipy import linalg

from pyqed import dag, fine_structure
from pyqed.qchem._libcint import CBasis1e

LIGHT_SPEED = 1.0 / fine_structure
LINEAR_DEP_THRESHOLD = 1.0e-14


def _symmetrize(mat):
    mat = np.asarray(mat)
    return 0.5 * (mat + dag(mat))


def _cbasis_integrals(mol):
    basis = getattr(mol, "_bas", None)
    if basis is None:
        raise ValueError("mol._bas is not available.")
    try:
        transform = getattr(mol, "_ao_cart2sph", None)
        if transform is not None and getattr(mol, "_bas_cart", None) is not None:
            basis = mol._bas_cart
            coord_type = "cartesian"
        else:
            coord_type = getattr(basis[0], "coord_type", "spherical")
        cbasis = CBasis1e(basis, mol.atom_symbols(), mol.atom_coords(), coord_type=coord_type)
        integrals = (
            cbasis.int1e("int1e_ovlp"),
            cbasis.int1e("int1e_kin"),
            cbasis.int1e("int1e_nuc"),
            cbasis.int1e("int1e_pnucp"),
        )
        if transform is not None:
            integrals = tuple(
                np.einsum("pi,pq,qj->ij", transform, mat, transform, optimize=True)
                for mat in integrals
            )
        if (
            getattr(mol, "overlap", None) is not None
            and integrals[0].shape == np.asarray(mol.overlap).shape
            and np.max(np.abs(_symmetrize(integrals[0]) - mol.overlap)) > 1e-7
        ):
            raise ValueError("native overlap integral is inconsistent with molecule overlap.")
        if (
            getattr(mol, "hcore", None) is not None
            and integrals[1].shape == np.asarray(mol.hcore).shape
            and np.max(np.abs(_symmetrize(integrals[1] + integrals[2]) - mol.hcore)) > 1e-6
        ):
            raise ValueError("native hcore integrals are inconsistent with molecule hcore.")
        return tuple(_symmetrize(mat) for mat in integrals)
    except Exception as exc:
        raise ValueError("native libcint one-electron integrals are unavailable.") from exc


def _pyscf_integrals(mol):
    if not hasattr(mol, "topyscf"):
        raise ValueError("A pyqed Molecule with native basis data or topyscf() is required.")
    pmol = mol.topyscf()
    if getattr(pmol, "has_ecp", lambda: False)():
        raise NotImplementedError("Scalar-X2C one-electron hcore is not implemented for ECPs.")
    return (
        pmol.intor_symmetric("int1e_ovlp"),
        pmol.intor_symmetric("int1e_kin"),
        pmol.intor_symmetric("int1e_nuc"),
        pmol.intor_symmetric("int1e_pnucp"),
    )


def _pyscf_integrals_in_mol_order(mol):
    """
    Return PySCF one-electron X2C integrals reordered to ``mol`` AO labels.

    The builtin spherical backend keeps pyqed's shell order, which can differ
    from PySCF/libcint's AO order for segmented bases.  This adapter is used
    only when native libcint integrals cannot be proven consistent with
    ``mol.overlap``/``mol.hcore``.
    """
    pmol = mol.topyscf()
    perm = mol.pyscf_ao_permutation(pmol)

    integrals = _pyscf_integrals(mol)
    reordered = tuple(mat[np.ix_(perm, perm)] for mat in integrals)
    if (
        getattr(mol, "overlap", None) is not None
        and np.max(np.abs(_symmetrize(reordered[0]) - mol.overlap)) > 1e-7
    ):
        raise ValueError("AO-label-reordered PySCF overlap does not match the pyqed overlap.")
    if (
        getattr(mol, "hcore", None) is not None
        and np.max(np.abs(_symmetrize(reordered[1] + reordered[2]) - mol.hcore)) > 1e-6
    ):
        raise ValueError("AO-label-reordered PySCF hcore does not match the pyqed hcore.")
    return tuple(_symmetrize(mat) for mat in reordered)


def x2c1e_integrals(mol):
    """
    Return ``S, T, V, W`` AO integrals needed by spin-free one-electron X2C.

    ``W`` is the scalar ``p V_nuc p`` integral.  The native libcint route is
    used when pyqed basis objects are available; PySCF is a compatibility
    fallback for molecules built through the PySCF driver.
    """
    try:
        return _cbasis_integrals(mol)
    except ValueError as exc:
        if getattr(mol, "_builtin_build_info", None) is not None:
            return _pyscf_integrals_in_mol_order(mol)
        return _pyscf_integrals(mol)


def _x2c1e_hcore_from_integrals(t, v, w, s, light_speed=LIGHT_SPEED):
    """
    Build the stable one-step spin-free X2C hcore from AO integrals.

    This follows the standard restricted-kinetic-balance generalized
    eigenproblem and evaluates the Foldy-Wouthuysen transformed Hamiltonian in
    the electronic positive-energy subspace.
    """
    c = float(light_speed)
    nao = s.shape[0]
    n2 = 2 * nao
    dtype = np.result_type(t, v, w, s)
    h = np.zeros((n2, n2), dtype=dtype)
    m = np.zeros((n2, n2), dtype=dtype)
    h[:nao, :nao] = v
    h[:nao, nao:] = t
    h[nao:, :nao] = t
    h[nao:, nao:] = w * (0.25 / c**2) - t
    m[:nao, :nao] = s
    m[nao:, nao:] = t * (0.5 / c**2)

    try:
        e, coeff = linalg.eigh(_symmetrize(h), _symmetrize(m))
        cl = coeff[:nao, nao:]
        e = e[nao:]
    except linalg.LinAlgError:
        metric_e, metric_u = np.linalg.eigh(_symmetrize(m))
        keep = metric_e > LINEAR_DEP_THRESHOLD
        metric_u = metric_u[:, keep] / np.sqrt(metric_e[keep])
        h_orth = reduce(np.dot, (metric_u.conj().T, h, metric_u))
        e, coeff_orth = np.linalg.eigh(_symmetrize(h_orth))
        coeff = np.dot(metric_u, coeff_orth)
        keep = e > -c**2
        cl = coeff[:nao, keep]
        e = e[keep]

    metric = reduce(np.dot, (cl.conj().T, s, cl))
    occ, u = np.linalg.eigh(_symmetrize(metric))
    keep = occ > LINEAR_DEP_THRESHOLD
    r = reduce(
        np.dot,
        (
            u[:, keep] / np.sqrt(occ[keep]),
            u[:, keep].conj().T,
            cl.conj().T,
            s,
        ),
    )
    hcore = reduce(np.dot, (r.conj().T * e, r))
    return _symmetrize(hcore)


def x2c1e_hcore(mol, light_speed=LIGHT_SPEED):
    """
    Spin-free one-electron X2C core Hamiltonian in the AO basis.

    The returned matrix replaces the nonrelativistic ``T + V_nuc`` hcore in a
    scalar RHF/ROHF calculation.  It does not include spin-orbit coupling.
    """
    s, t, v, w = x2c1e_integrals(mol)
    return _x2c1e_hcore_from_integrals(t, v, w, s, light_speed=light_speed)


scalar_x2c_hcore = x2c1e_hcore
