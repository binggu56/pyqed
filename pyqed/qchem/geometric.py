from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BOHamiltonianDerivatives:
    state_ids: tuple[int, ...]
    cartesian_labels: tuple[str, ...]
    h1_ao_cartesian: np.ndarray
    h2_ao_cartesian: np.ndarray
    vnn_gradient_cartesian: np.ndarray
    vnn_hessian_cartesian: np.ndarray
    F_cartesian: np.ndarray
    G_cartesian: np.ndarray
    mode_vectors: np.ndarray | None = None
    F_projected: np.ndarray | None = None
    G_projected: np.ndarray | None = None

    @property
    def first_order(self):
        return self.F_cartesian

    @property
    def second_order(self):
        return self.G_cartesian


def _infer_state_ids(state_model, state_ids):
    if state_ids is not None:
        return tuple(int(i) for i in state_ids)

    if hasattr(state_model, 'ci'):
        try:
            return tuple(range(len(state_model.ci)))
        except TypeError:
            pass

    e_tot = getattr(state_model, 'e_tot', None)
    if e_tot is not None:
        arr = np.asarray(e_tot)
        if arr.ndim > 0:
            return tuple(range(len(arr)))

    raise ValueError(
        "state_ids was not provided and the number of electronic states could not be inferred. "
        "Pass state_ids explicitly."
    )


def _as_cartesian_mode_matrix(mode_vectors, natom):
    vec = np.asarray(mode_vectors, dtype=float)
    ncart = 3 * natom

    if vec.ndim == 3:
        if vec.shape[1:] != (natom, 3):
            raise ValueError(
                f"mode_vectors shape {vec.shape} is incompatible with (nmodes, {natom}, 3)."
            )
        return vec.reshape(vec.shape[0], ncart)

    if vec.ndim == 2:
        if vec.shape == (ncart, 0):
            return vec.T
        if vec.shape[1] == ncart:
            return vec
        if vec.shape[0] == ncart:
            return vec.T

    raise ValueError(
        "mode_vectors must have shape (nmodes, natom, 3), (nmodes, 3*natom), "
        "or (3*natom, nmodes)."
    )


def _normalize_coord_type(coord_type):
    if coord_type in ('c', 'cartesian'):
        return 'cartesian'
    if coord_type in ('p', 'spherical'):
        return 'spherical'
    raise ValueError(f"Unsupported AO coord_type '{coord_type}'.")


def _move_component_axes_first(arr, ncomp_axes):
    arr = np.asarray(arr, dtype=np.float64)
    if ncomp_axes == 1:
        return np.moveaxis(arr, -1, 0)
    if ncomp_axes == 2:
        return np.moveaxis(arr, (-2, -1), (0, 1))
    raise ValueError("ncomp_axes must be 1 or 2.")


def _build_cbasis_from_reference(mol):
    try:
        from pyqed.qchem._libcint import CBasis1e
    except Exception as exc:
        raise NotImplementedError(
            "Analytic geometric F/G terms require the local libcint wrapper."
        ) from exc

    if getattr(mol, '_bas', None) is None:
        raise ValueError(
            "Analytic geometric F/G terms require mol._bas. "
            "Build the molecule with driver='builtin', 'gbasis', 'gbasis-pyscf', or 'pyscf'."
        )

    coord_type = getattr(mol._bas[0], 'coord_type', None)
    if coord_type is None:
        first_shell = mol._bas[0]
        if hasattr(first_shell, 'shell') and not hasattr(first_shell, 'angmom'):
            coord_type = 'cartesian'
        else:
            coord_type = 'cartesian' if getattr(mol, 'cart', False) else 'spherical'

    return CBasis1e(
        mol._bas,
        mol.atom_symbols(),
        mol.atom_coords(),
        coord_type=_normalize_coord_type(coord_type),
    )


def _nuclear_repulsion_hessian(mol):
    z = np.asarray(mol.atom_charges(), dtype=float)
    r = np.asarray(mol.atom_coords(), dtype=float)
    natom = len(z)
    h = np.zeros((natom, natom, 3, 3), dtype=float)

    for i in range(natom):
        r12 = r[i] - r
        s12 = np.sqrt(np.einsum('ki,ki->k', r12, r12))
        s12[i] = 1e60
        tmp1 = z[i] * z / s12**3
        tmp2 = np.einsum('k,ki,kj->kij', -3.0 * z[i] * z / s12**5, r12, r12)

        h[i, i, 0, 0] = h[i, i, 1, 1] = h[i, i, 2, 2] = -tmp1.sum()
        h[i, i] -= np.einsum('kij->ij', tmp2)

        h[i, :, 0, 0] += tmp1
        h[i, :, 1, 1] += tmp1
        h[i, :, 2, 2] += tmp1
        h[i, :] += tmp2

    return h.reshape(3 * natom, 3 * natom)


def _electron_nuclear_operator_derivatives(mol):
    cbas = _build_cbasis_from_reference(mol)
    natom = mol.natom
    nao = cbas.nbfn
    ncart = 3 * natom

    h1_cart = np.zeros((ncart, nao, nao), dtype=np.complex128)
    h2_cart = np.zeros((ncart, ncart, nao, nao), dtype=np.complex128)

    for atom_id in range(natom):
        charge = float(mol.atom_charge(atom_id))
        origin = np.asarray(mol.atom_coord(atom_id), dtype=float)
        sl = slice(3 * atom_id, 3 * atom_id + 3)

        ip_rinv = _move_component_axes_first(
            cbas.int1e(
                'int1e_iprinv',
                components=(3,),
                inv_origin=origin,
                hermi=False,
            ),
            1,
        )
        h1_atom = -charge * (ip_rinv + ip_rinv.transpose(0, 2, 1))
        h1_cart[sl] = 0.5 * (h1_atom + h1_atom.transpose(0, 2, 1).conj())

        ipip_rinv = _move_component_axes_first(
            cbas.int1e(
                'int1e_ipiprinv',
                components=(3, 3),
                inv_origin=origin,
                hermi=False,
            ),
            2,
        )
        ip_rinv_ip = _move_component_axes_first(
            cbas.int1e(
                'int1e_iprinvip',
                components=(3, 3),
                inv_origin=origin,
                hermi=False,
            ),
            2,
        )

        h2_atom = -charge * (
            ipip_rinv
            + ip_rinv_ip
            + ip_rinv_ip.swapaxes(0, 1)
            + ipip_rinv.swapaxes(0, 1).transpose(0, 1, 3, 2)
        )
        h2_atom = 0.5 * (h2_atom + h2_atom.swapaxes(0, 1))
        h2_atom = 0.5 * (h2_atom + h2_atom.transpose(0, 1, 3, 2).conj())
        h2_cart[sl, sl] = h2_atom

    return h1_cart, h2_cart


def _contract_ao_operator_with_state_model(state_model, bra_id, ket_id, h1e_ao):
    if hasattr(state_model, 'binary') and getattr(state_model, 'binary', None) is None:
        raise ValueError("Electronic-state determinant basis is not available.")
    if hasattr(state_model, 'binary') and (
        getattr(state_model, 'SC1', None) is None or getattr(state_model, 'SC2', None) is None
    ):
        from pyqed.qchem.ci.fci import SlaterCondon

        state_model.SC1, state_model.SC2 = SlaterCondon(state_model.binary)

    mo = np.asarray(state_model.mf.mo_coeff)
    h1e_mo = mo.T.conj() @ np.asarray(h1e_ao) @ mo

    ncore = state_model.ncore
    ncas = state_model.ncas
    h1e_cas = h1e_mo[ncore:ncore + ncas, ncore:ncore + ncas]

    if bra_id == ket_id:
        dm1 = state_model.make_rdm1(bra_id)
        value = np.einsum('pq,qp->', h1e_cas, dm1, optimize=True)
        if ncore > 0:
            value += 2.0 * np.trace(h1e_mo[:ncore, :ncore])
        return value

    if not hasattr(state_model, 'make_tdm1'):
        raise ValueError(
            f"{type(state_model).__name__} does not provide make_tdm1(bra_id, ket_id), "
            "which is required for off-diagonal BO Hamiltonian derivatives."
        )
    tdm1 = state_model.make_tdm1(bra_id, ket_id)
    return np.einsum('pq,qp->', h1e_cas, tdm1, optimize=True)


def bo_hamiltonian_derivatives(state_model, state_ids=None, mode_vectors=None, overlap_tol=1e-8):
    """
    Build the first- and second-order derivatives of the Born-Oppenheimer
    electronic Hamiltonian in an electronic-state basis.

    Parameters
    ----------
    state_model
        Electronic-structure object that provides:
        ``mf.mol``, ``ncore``, ``ncas``, ``make_rdm1(state_id)``, and
        ``make_tdm1(bra_id, ket_id)`` for off-diagonal state couplings.
        The 1-RDM / TDMs are assumed to be in the active MO basis.
    state_ids : sequence of int, optional
        Electronic states to include. If omitted, they are inferred from
        ``state_model.ci`` or ``state_model.e_tot`` when possible.
    mode_vectors : ndarray, optional
        Optional Cartesian-to-mode projection matrix with shape
        ``(nmodes, natom, 3)``, ``(nmodes, 3*natom)``, or ``(3*natom, nmodes)``.
    overlap_tol : float, optional
        Kept for backward compatibility. Currently unused.

    Notes
    -----
    The returned Cartesian tensors correspond to derivatives with respect to
    nuclear Cartesian coordinates. Optional ``mode_vectors`` project these
    Cartesian derivatives to a coarse-grained coordinate set such as normal
    coordinates.
    """
    del overlap_tol

    for attr in ('mf', 'ncore', 'ncas', 'make_rdm1'):
        if not hasattr(state_model, attr):
            raise ValueError(
                f"{type(state_model).__name__} is missing required attribute/method '{attr}' "
                "for BO Hamiltonian derivatives."
            )

    mol = state_model.mf.mol
    natom = mol.natom
    ncart = 3 * natom

    state_ids = _infer_state_ids(state_model, state_ids)

    labels = tuple(
        f"{mol.atom_symbol(a)}{a}:{axis}"
        for a in range(natom)
        for axis in ("x", "y", "z")
    )

    h1_cart, h2_cart = _electron_nuclear_operator_derivatives(mol)

    from pyqed.qchem.mol import grad_nuc

    nn_grad = np.asarray(grad_nuc(mol), dtype=np.complex128).reshape(ncart)
    nn_hess = np.asarray(_nuclear_repulsion_hessian(mol), dtype=np.complex128)

    nstates = len(state_ids)
    f_cart = np.zeros((ncart, nstates, nstates), dtype=np.complex128)
    g_cart = np.zeros((ncart, ncart, nstates, nstates), dtype=np.complex128)

    for ibra, bra in enumerate(state_ids):
        for iket, ket in enumerate(state_ids):
            for c in range(ncart):
                val = _contract_ao_operator_with_state_model(state_model, bra, ket, h1_cart[c])
                if bra == ket:
                    val += nn_grad[c]
                f_cart[c, ibra, iket] = val

            for c1 in range(ncart):
                for c2 in range(ncart):
                    val = _contract_ao_operator_with_state_model(
                        state_model,
                        bra,
                        ket,
                        h2_cart[c1, c2],
                    )
                    if bra == ket:
                        val += nn_hess[c1, c2]
                    g_cart[c1, c2, ibra, iket] = val

    mode_mat = None
    f_proj = None
    g_proj = None
    if mode_vectors is not None:
        mode_mat = _as_cartesian_mode_matrix(mode_vectors, natom)
        f_proj = np.einsum('ka,aij->kij', mode_mat, f_cart, optimize=True)
        g_proj = np.einsum('ka,lb,abij->klij', mode_mat, mode_mat, g_cart, optimize=True)

    return BOHamiltonianDerivatives(
        state_ids=state_ids,
        cartesian_labels=labels,
        h1_ao_cartesian=h1_cart,
        h2_ao_cartesian=h2_cart,
        vnn_gradient_cartesian=nn_grad,
        vnn_hessian_cartesian=nn_hess,
        F_cartesian=f_cart,
        G_cartesian=g_cart,
        mode_vectors=mode_mat,
        F_projected=f_proj,
        G_projected=g_proj,
    )


# Backward-compatible aliases for the earlier naming.
GeometricFGTerms = BOHamiltonianDerivatives
build_casci_bo_hamiltonian_derivatives = bo_hamiltonian_derivatives
build_casci_geometric_fg_terms = bo_hamiltonian_derivatives
