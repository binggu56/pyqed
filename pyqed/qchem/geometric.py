from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np


@dataclass
class BOHamiltonianDerivatives:
    state_ids: tuple[int, ...]
    cartesian_labels: tuple[str, ...]
    h1_ao_cartesian: np.ndarray
    h2_ao_cartesian: np.ndarray | None
    vnn_gradient_cartesian: np.ndarray
    vnn_hessian_cartesian: np.ndarray | None
    F_cartesian: np.ndarray | None
    G_cartesian: np.ndarray | None
    eri1_mo_cartesian: np.ndarray | None = None
    eri2_mo_cartesian: np.ndarray | None = None
    h1_mo_cartesian: np.ndarray | None = None
    h2_mo_cartesian: np.ndarray | None = None
    core_gradient_cartesian: np.ndarray | None = None
    core_hessian_cartesian: np.ndarray | None = None
    mode_vectors: np.ndarray | None = None
    F_projected: np.ndarray | None = None
    G_projected: np.ndarray | None = None
    moving_basis: str | bool = "symmetric"

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

    return h.transpose(0, 2, 1, 3).reshape(3 * natom, 3 * natom)


def _electron_nuclear_operator_derivatives(mol):
    pyscf_derivatives = _electron_nuclear_operator_derivatives_pyscf(mol)
    if pyscf_derivatives is not None:
        return pyscf_derivatives

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


def _electron_nuclear_operator_derivatives_pyscf(mol):
    pmol = None
    if hasattr(mol, "intor") and hasattr(mol, "with_rinv_as_nucleus"):
        pmol = mol
    elif all(hasattr(mol, name) for name in ("atom_symbols", "atom_coords")):
        try:
            from pyscf import gto

            atom = [
                [symbol, tuple(coord)]
                for symbol, coord in zip(mol.atom_symbols(), mol.atom_coords())
            ]
            pmol = gto.M(
                atom=atom,
                basis=getattr(mol, "basis", None),
                charge=int(getattr(mol, "charge", 0)),
                spin=int(getattr(mol, "spin", 0)),
                unit=getattr(mol, "unit", "bohr"),
                verbose=0,
            )
        except Exception:
            pmol = None
    if pmol is None and hasattr(mol, "topyscf"):
        try:
            pmol = mol.topyscf()
        except Exception:
            pmol = None
    if pmol is None:
        return None

    try:
        natom = int(pmol.natm if hasattr(pmol, "natm") else pmol.natom)
        nao = int(pmol.nao_nr() if hasattr(pmol, "nao_nr") else pmol.nao)
    except Exception:
        return None

    ncart = 3 * natom
    h1_cart = np.zeros((ncart, nao, nao), dtype=np.complex128)
    h2_cart = np.zeros((ncart, ncart, nao, nao), dtype=np.complex128)

    try:
        for atom_id in range(natom):
            charge = float(pmol.atom_charge(atom_id))
            sl = slice(3 * atom_id, 3 * atom_id + 3)
            with pmol.with_rinv_as_nucleus(atom_id):
                ip_rinv = np.asarray(
                    pmol.intor("int1e_iprinv", comp=3),
                    dtype=np.complex128,
                )
                ipip_rinv = np.asarray(
                    pmol.intor("int1e_ipiprinv", comp=9),
                    dtype=np.complex128,
                ).reshape(3, 3, nao, nao)
                ip_rinv_ip = np.asarray(
                    pmol.intor("int1e_iprinvip", comp=9),
                    dtype=np.complex128,
                ).reshape(3, 3, nao, nao)
            h1_atom = -charge * (ip_rinv + ip_rinv.transpose(0, 2, 1))
            h1_cart[sl] = 0.5 * (h1_atom + h1_atom.transpose(0, 2, 1).conj())

            h2_atom = -charge * (
                ipip_rinv
                + ip_rinv_ip
                + ip_rinv_ip.swapaxes(0, 1)
                + ipip_rinv.swapaxes(0, 1).transpose(0, 1, 3, 2)
            )
            h2_atom = 0.5 * (h2_atom + h2_atom.swapaxes(0, 1))
            h2_atom = 0.5 * (h2_atom + h2_atom.transpose(0, 1, 3, 2).conj())
            h2_cart[sl, sl] = h2_atom
    except Exception:
        return None

    return h1_cart, h2_cart


def _contract_ao_operator_with_state_model(state_model, bra_id, ket_id, h1e_ao):
    if hasattr(state_model, 'binary') and getattr(state_model, 'binary', None) is None:
        raise ValueError("Electronic-state determinant basis is not available.")
    if hasattr(state_model, 'binary') and (
        getattr(state_model, 'SC1', None) is None or getattr(state_model, 'SC2', None) is None
    ):
        from pyqed.qchem.ci.fci import SlaterCondon

        state_model.SC1, state_model.SC2 = SlaterCondon(state_model.binary)

    mo = np.asarray(
        getattr(state_model, "mo_coeff", state_model.mf.mo_coeff)
    )
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


def _state_model_mo_coeff(state_model):
    mo = getattr(state_model, "mo_coeff", None)
    if mo is None:
        mo = getattr(getattr(state_model, "mf", None), "mo_coeff", None)
    if mo is None:
        raise ValueError("state_model must provide mo_coeff or mf.mo_coeff.")
    if isinstance(mo, (tuple, list)):
        raise NotImplementedError("BO Hamiltonian derivative integrals currently support RHF-like orbitals.")
    return np.asarray(mo)


def _get_hcore_mo(state_model, mo):
    mf = state_model.mf
    if hasattr(mf, "get_hcore_mo"):
        return np.asarray(mf.get_hcore_mo(mo))
    if hasattr(mf, "get_hcore"):
        hcore = np.asarray(mf.get_hcore())
    else:
        hcore = np.asarray(mf.mol.hcore)
    return np.einsum("pi,pq,qj->ij", mo.conj(), hcore, mo, optimize=True)


def _get_eri_mo(state_model, mo):
    mf = state_model.mf
    if hasattr(mf, "get_eri_mo"):
        return np.asarray(mf.get_eri_mo(mo, notation="chem"))
    eri = getattr(mf, "eri", None)
    if eri is None:
        eri = getattr(mf.mol, "eri", None)
    if eri is None:
        raise ValueError("state_model.mf must provide get_eri_mo() or dense AO ERIs.")
    return np.einsum(
        "pqrs,pi,qj,rk,sl->ijkl",
        np.asarray(eri),
        mo.conj(),
        mo,
        mo.conj(),
        mo,
        optimize=True,
    )


def _one_electron_ao_to_mo(ao, mo):
    return np.einsum("pi,...pq,qj->...ij", mo.conj(), np.asarray(ao), mo, optimize=True)


def _eri_ao_to_mo(ao, mo):
    return np.einsum(
        "pi,qj,...pqrs,rk,sl->...ijkl",
        mo.conj(),
        mo,
        np.asarray(ao),
        mo.conj(),
        mo,
        optimize=True,
    )


def _one_electron_orbital_response(mat, kappa):
    mat = np.asarray(mat)
    kappa = np.asarray(kappa)
    return (
        np.einsum("ai,...aj->...ij", kappa, mat, optimize=True)
        + np.einsum("...ia,aj->...ij", mat, kappa, optimize=True)
    )


def _one_electron_second_orbital_response(mat, kappa_x, kappa_y, kappa_xy):
    out = _one_electron_orbital_response(mat, kappa_xy)
    out += np.einsum("ai,...ab,bj->...ij", kappa_x, mat, kappa_y, optimize=True)
    out += np.einsum("ai,...ab,bj->...ij", kappa_y, mat, kappa_x, optimize=True)
    return out


def _eri_apply_orbital_response_slot(eri, kappa, slot):
    eri = np.asarray(eri)
    kappa = np.asarray(kappa)
    if slot == 0:
        return np.einsum("pi,...pjkl->...ijkl", kappa, eri, optimize=True)
    if slot == 1:
        return np.einsum("qj,...iqkl->...ijkl", kappa, eri, optimize=True)
    if slot == 2:
        return np.einsum("rk,...ijrl->...ijkl", kappa, eri, optimize=True)
    if slot == 3:
        return np.einsum("sl,...ijks->...ijkl", kappa, eri, optimize=True)
    raise ValueError("slot must be 0, 1, 2, or 3.")


def _eri_orbital_response(eri, kappa):
    out = np.zeros_like(np.asarray(eri), dtype=np.result_type(eri, kappa))
    for slot in range(4):
        out = out + _eri_apply_orbital_response_slot(eri, kappa, slot)
    return out


def _eri_second_orbital_response(eri, kappa_x, kappa_y, kappa_xy):
    out = _eri_orbital_response(eri, kappa_xy)
    for slot_x in range(4):
        first = _eri_apply_orbital_response_slot(eri, kappa_x, slot_x)
        for slot_y in range(4):
            if slot_y == slot_x:
                continue
            out = out + _eri_apply_orbital_response_slot(first, kappa_y, slot_y)
    return out


def _normalize_moving_basis(moving_basis):
    if moving_basis is False or moving_basis is None:
        return False
    if moving_basis is True:
        return "symmetric"
    key = str(moving_basis).lower().replace("_", "-")
    if key in {"symmetric", "same-geometry", "lowdin", "loewdin"}:
        return "symmetric"
    if key in {"rhf-relaxed", "cphf", "relaxed"}:
        return "rhf-relaxed"
    if key in {
        "rhf-relaxed-pt",
        "relaxed-pt",
        "parallel-transport",
        "parallel",
    }:
        return "rhf-relaxed-pt"
    raise ValueError(
        "moving_basis must be False, True, 'symmetric', 'rhf-relaxed', or "
        "'rhf-relaxed-pt'. "
        "One-sided finite-difference AO-overlap transport is intentionally not used here."
    )


def _parallel_transport_orbital_response(
    kappa,
    overlap1,
    active,
    kappa2=None,
    overlap2=None,
):
    kappa = np.asarray(kappa)
    overlap1 = np.asarray(overlap1)
    active = np.arange(kappa.shape[-1])[active]
    cross1 = kappa + overlap1
    connection = 0.5 * (
        cross1 - cross1.swapaxes(-1, -2).conj()
    )
    gauge1 = np.zeros_like(kappa)
    gauge1[..., active[:, None], active] = -connection[
        ..., active[:, None], active
    ]
    transported1 = kappa + gauge1
    if kappa2 is None:
        return transported1, None
    if overlap2 is None:
        raise ValueError("Second-order parallel transport requires overlap2.")

    kappa2 = np.asarray(kappa2)
    overlap2 = np.asarray(overlap2)
    ncoord = len(kappa)
    cross2 = np.empty_like(kappa2)
    for x in range(ncoord):
        for y in range(ncoord):
            cross2[x, y] = (
                overlap2[x, y]
                + overlap1[x] @ kappa[y]
                + overlap1[y] @ kappa[x]
                + kappa2[x, y]
            )

    gauge2 = np.zeros_like(kappa2)
    for x in range(ncoord):
        qx = gauge1[x][np.ix_(active, active)]
        for y in range(ncoord):
            qy = gauge1[y][np.ix_(active, active)]
            aligned = (
                cross2[x, y]
                + cross1[x] @ gauge1[y]
                + cross1[y] @ gauge1[x]
            )[np.ix_(active, active)]
            gauge2[x, y][np.ix_(active, active)] = (
                -0.5 * (aligned - aligned.conj().T)
                + 0.5 * (qx @ qy + qy @ qx)
            )
    transported2 = np.empty_like(kappa2)
    for x in range(ncoord):
        for y in range(ncoord):
            transported2[x, y] = (
                kappa2[x, y]
                + kappa[x] @ gauge1[y]
                + kappa[y] @ gauge1[x]
                + gauge2[x, y]
            )
    transported2 = 0.5 * (
        transported2 + transported2.swapaxes(0, 1)
    )
    return transported1, transported2


def _basis_derivative_integrals_mo(
    state_model,
    *,
    moving_basis="symmetric",
    mode_vectors=None,
    order=2,
    backend="auto",
):
    from pyqed.qchem.basis_derivatives import (
        directional_eri_derivatives,
        directional_one_electron_derivatives,
        eri_derivatives,
        one_index_one_electron_derivatives,
        one_electron_derivatives,
    )

    order = int(order)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    moving_basis = _normalize_moving_basis(moving_basis)
    mol = state_model.mf.mol
    natom = int(mol.natom)
    nao = int(mol.nao)
    ncart = 3 * natom
    mo = _state_model_mo_coeff(state_model)

    if mode_vectors is None:
        ncoord = ncart
    else:
        mode_mat = _as_cartesian_mode_matrix(mode_vectors, natom)
        ncoord = mode_mat.shape[0]
    one_electron_backend = backend

    try:
        if mode_vectors is None:
            h1_ao = one_electron_derivatives(
                mol, "hcore", order=1, backend=one_electron_backend
            ).reshape(ncart, nao, nao)
            eri1_ao = eri_derivatives(mol, order=1).reshape(ncart, nao, nao, nao, nao)
            if order == 2:
                h2_ao = one_electron_derivatives(
                    mol, "hcore", order=2, backend=one_electron_backend
                ).reshape(
                    ncart,
                    ncart,
                    nao,
                    nao,
                )
                eri2_ao = eri_derivatives(mol, order=2).reshape(
                    ncart,
                    ncart,
                    nao,
                    nao,
                    nao,
                    nao,
                )
            else:
                h2_ao = None
                eri2_ao = None
        else:
            h1_ao = directional_one_electron_derivatives(
                mol,
                mode_vectors,
                "hcore",
                order=1,
                backend=one_electron_backend,
            ).reshape(ncoord, nao, nao)
            eri1_ao = directional_eri_derivatives(
                mol,
                mode_vectors,
                order=1,
                backend=backend,
            ).reshape(ncoord, nao, nao, nao, nao)
            if order == 2:
                h2_ao = directional_one_electron_derivatives(
                    mol,
                    mode_vectors,
                    "hcore",
                    order=2,
                    backend=one_electron_backend,
                ).reshape(ncoord, ncoord, nao, nao)
                eri2_ao = directional_eri_derivatives(
                    mol,
                    mode_vectors,
                    order=2,
                    backend=backend,
                ).reshape(ncoord, ncoord, nao, nao, nao, nao)
            else:
                h2_ao = None
                eri2_ao = None
    except Exception as exc:
        raise ValueError(
            "BO Hamiltonian derivatives require project-local derivative integrals. "
            "Build the molecule with the builtin Gaussian driver, e.g. "
            "mol.build(driver='builtin', eri='dense')."
        ) from exc

    h1_mo_explicit = _one_electron_ao_to_mo(h1_ao, mo)
    eri1_mo_explicit = _eri_ao_to_mo(eri1_ao, mo)
    h2_mo_explicit = (
        None if h2_ao is None else _one_electron_ao_to_mo(h2_ao, mo)
    )
    eri2_mo_explicit = (
        None if eri2_ao is None else _eri_ao_to_mo(eri2_ao, mo)
    )

    if moving_basis is False:
        return {
            "moving_basis": False,
            "h1_ao": h1_ao,
            "h2_ao": h2_ao,
            "h1_mo": h1_mo_explicit,
            "h2_mo": h2_mo_explicit,
            "eri1_mo": eri1_mo_explicit,
            "eri2_mo": eri2_mo_explicit,
            "s1_mo": None,
            "s2_mo": None,
        }

    if mode_vectors is None:
        s1_ao = one_electron_derivatives(mol, "overlap", order=1).reshape(ncart, nao, nao)
        s2_ao = (
            one_electron_derivatives(mol, "overlap", order=2).reshape(
                ncart,
                ncart,
                nao,
                nao,
            )
            if order == 2
            else None
        )
    else:
        s1_ao = directional_one_electron_derivatives(
            mol,
            mode_vectors,
            "overlap",
            order=1,
        ).reshape(ncoord, nao, nao)
        s2_ao = (
            directional_one_electron_derivatives(
                mol,
                mode_vectors,
                "overlap",
                order=2,
            ).reshape(ncoord, ncoord, nao, nao)
            if order == 2
            else None
        )
    s1_mo = _one_electron_ao_to_mo(s1_ao, mo)
    s2_mo = None if s2_ao is None else _one_electron_ao_to_mo(s2_ao, mo)
    if moving_basis in {"rhf-relaxed", "rhf-relaxed-pt"}:
        mf = state_model.mf
        canonical = np.asarray(mf.mo_coeff)
        overlap = np.asarray(mf.get_ovlp())
        rotation = canonical.conj().T @ overlap @ mo
        identity = np.eye(rotation.shape[0])
        if not np.allclose(
            rotation.conj().T @ rotation,
            identity,
            atol=1.0e-7,
            rtol=1.0e-7,
        ):
            raise ValueError(
                "CASCI orbitals must span the canonical RHF MO space for "
                "RHF-relaxed derivatives."
            )
        if order == 2:
            canonical_response, canonical_response2 = (
                mf.Hessian().orbital_second_response(
                    h1_ao,
                    eri1_ao,
                    s1_ao,
                    h2_ao,
                    eri2_ao,
                    s2_ao,
                )
            )
        else:
            canonical_response = mf.Hessian().orbital_response(
                h1_ao,
                eri1_ao,
                s1_ao,
            )
        kappa = np.einsum(
            "pi,xpq,qj->xij",
            rotation.conj(),
            canonical_response,
            rotation,
            optimize=True,
        )
        if order == 2:
            kappa2 = np.einsum(
                "pi,xypq,qj->xyij",
                rotation.conj(),
                canonical_response2,
                rotation,
                optimize=True,
            )
        if moving_basis == "rhf-relaxed-pt":
            active = slice(
                int(state_model.ncore),
                int(state_model.ncore + state_model.ncas),
            )
            cross1_cart = one_index_one_electron_derivatives(
                mol,
                "overlap",
                index="ket",
                order=1,
                backend=one_electron_backend,
            ).reshape(ncart, nao, nao)
            if mode_vectors is None:
                cross1_ao = cross1_cart
            else:
                cross1_ao = np.einsum(
                    "ka,apq->kpq",
                    mode_mat,
                    cross1_cart,
                    optimize=True,
                )
            cross1_mo = _one_electron_ao_to_mo(cross1_ao, mo)
            cross2_mo = None
            if order == 2:
                cross2_cart = one_index_one_electron_derivatives(
                    mol,
                    "overlap",
                    index="ket",
                    order=2,
                    backend=one_electron_backend,
                ).reshape(ncart, ncart, nao, nao)
                if mode_vectors is None:
                    cross2_ao = cross2_cart
                else:
                    cross2_ao = np.einsum(
                        "ka,lb,abpq->klpq",
                        mode_mat,
                        mode_mat,
                        cross2_cart,
                        optimize=True,
                    )
                cross2_mo = _one_electron_ao_to_mo(cross2_ao, mo)
            kappa, transported2 = _parallel_transport_orbital_response(
                kappa,
                cross1_mo,
                active,
                kappa2=kappa2 if order == 2 else None,
                overlap2=cross2_mo,
            )
            if order == 2:
                kappa2 = transported2
    else:
        kappa = -0.5 * s1_mo
    if order == 2 and moving_basis not in {"rhf-relaxed", "rhf-relaxed-pt"}:
        kappa2 = -0.5 * s2_mo + 0.375 * (
            np.einsum("xik,ykj->xyij", s1_mo, s1_mo, optimize=True)
            + np.einsum("yik,xkj->xyij", s1_mo, s1_mo, optimize=True)
        )

    h0_mo = _get_hcore_mo(state_model, mo)
    eri0_mo = _get_eri_mo(state_model, mo)

    h1_mo = h1_mo_explicit + np.asarray(
        [_one_electron_orbital_response(h0_mo, kappa[x]) for x in range(ncoord)]
    )
    eri1_mo = eri1_mo_explicit + np.asarray(
        [_eri_orbital_response(eri0_mo, kappa[x]) for x in range(ncoord)]
    )

    if order == 2:
        h2_mo = np.array(h2_mo_explicit, copy=True)
        eri2_mo = np.array(eri2_mo_explicit, copy=True)
        for x in range(ncoord):
            for y in range(ncoord):
                h2_mo[x, y] += _one_electron_orbital_response(h1_mo_explicit[x], kappa[y])
                h2_mo[x, y] += _one_electron_orbital_response(h1_mo_explicit[y], kappa[x])
                h2_mo[x, y] += _one_electron_second_orbital_response(
                    h0_mo,
                    kappa[x],
                    kappa[y],
                    kappa2[x, y],
                )
                eri2_mo[x, y] += _eri_orbital_response(eri1_mo_explicit[x], kappa[y])
                eri2_mo[x, y] += _eri_orbital_response(eri1_mo_explicit[y], kappa[x])
                eri2_mo[x, y] += _eri_second_orbital_response(
                    eri0_mo,
                    kappa[x],
                    kappa[y],
                    kappa2[x, y],
                )
    else:
        h2_mo = None
        eri2_mo = None

    h1_mo = 0.5 * (h1_mo + h1_mo.swapaxes(-1, -2).conj())
    if order == 2:
        h2_mo = 0.5 * (h2_mo + h2_mo.swapaxes(0, 1))
        h2_mo = 0.5 * (h2_mo + h2_mo.swapaxes(-1, -2).conj())

    return {
        "moving_basis": moving_basis,
        "h1_ao": h1_ao,
        "h2_ao": h2_ao,
        "h1_mo": h1_mo,
        "h2_mo": h2_mo,
        "eri1_mo": eri1_mo,
        "eri2_mo": eri2_mo,
        "s1_mo": s1_mo,
        "s2_mo": s2_mo,
    }


def _active_integral_derivatives_from_full_mo(dh_mo, deri_mo, ncore, ncas):
    ncore = int(ncore)
    ncas = int(ncas)
    active = slice(ncore, ncore + ncas)
    h_active = np.array(dh_mo[..., active, active], copy=True)
    if ncore > 0:
        core = slice(0, ncore)
        core_j = 2.0 * np.einsum(
            "...pqii->...pq",
            deri_mo[..., active, active, core, core],
            optimize=True,
        )
        core_k = np.einsum(
            "...piqi->...pq",
            deri_mo[..., active, core, active, core],
            optimize=True,
        )
        h_active = h_active + core_j - core_k
    eri_active = np.array(deri_mo[..., active, active, active, active], copy=True)
    return h_active, eri_active


def _core_energy_derivative_from_full_mo(dh_mo, deri_mo, ncore):
    ncore = int(ncore)
    out = np.zeros(dh_mo.shape[:-2], dtype=np.result_type(dh_mo, deri_mo))
    if ncore <= 0:
        return out
    core = slice(0, ncore)
    out += 2.0 * np.trace(dh_mo[..., core, core], axis1=-2, axis2=-1)
    out += 2.0 * np.einsum(
        "...iijj->...",
        deri_mo[..., core, core, core, core],
        optimize=True,
    )
    out -= np.einsum(
        "...ijji->...",
        deri_mo[..., core, core, core, core],
        optimize=True,
    )
    return out


def _state_overlap(state_model, bra_id, ket_id):
    if bra_id == ket_id:
        return 1.0
    ci = getattr(state_model, "ci", None)
    if ci is None:
        return 0.0
    return np.vdot(ci[bra_id], ci[ket_id])


def _transition_active_rdms(state_model, bra_id, ket_id):
    if bra_id == ket_id:
        dm1 = state_model.make_rdm1(bra_id)
        dm2 = state_model.make_rdm2(bra_id)
        return np.asarray(dm1), np.asarray(dm2)
    if not hasattr(state_model, "make_tdm1") or not hasattr(state_model, "make_tdm2"):
        raise ValueError(
            f"{type(state_model).__name__} must provide make_tdm1() and make_tdm2() "
            "for off-diagonal BO Hamiltonian derivatives."
        )
    return (
        np.asarray(state_model.make_tdm1(bra_id, ket_id)),
        np.asarray(state_model.make_tdm2(bra_id, ket_id)),
    )


def _axis_index(axis):
    if isinstance(axis, str):
        key = axis.lower()
        if key not in {"x", "y", "z"}:
            raise ValueError("axis must be 'x', 'y', 'z', or an integer 0, 1, 2.")
        return {"x": 0, "y": 1, "z": 2}[key]
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 'x', 'y', 'z', or an integer 0, 1, 2.")
    return axis


def _infer_nstates(state_model):
    ci = getattr(state_model, "ci", None)
    if ci is not None:
        try:
            return len(ci)
        except TypeError:
            pass
    e_tot = getattr(state_model, "e_tot", None)
    if e_tot is not None:
        arr = np.asarray(e_tot)
        if arr.ndim > 0:
            return len(arr)
    raise ValueError("Could not infer the number of electronic states.")


def _normalize_state_ids_for_overlap(state_model, state_ids):
    nstates = _infer_nstates(state_model)
    if state_ids is None:
        return tuple(range(nstates))
    ids = tuple(int(idx) for idx in state_ids)
    if len(ids) == 0:
        raise ValueError("state_ids must contain at least one state.")
    for idx in ids:
        if idx < 0 or idx >= nstates:
            raise ValueError(f"state id {idx} is outside the available range 0..{nstates - 1}.")
    return ids


def _electric_dipole_mo_component(state_model, axis, center=None, dipole_mo=None):
    axis = _axis_index(axis)
    if dipole_mo is None:
        if hasattr(state_model, "_electric_dipole_mo"):
            dipole_mo = state_model._electric_dipole_mo(center=center)
        else:
            mf = getattr(state_model, "mf", None)
            mo_coeff = getattr(mf, "mo_coeff", None)
            if mf is None or mo_coeff is None:
                raise ValueError("state_model must provide mf.mo_coeff or _electric_dipole_mo().")
            if isinstance(mo_coeff, (tuple, list)):
                raise NotImplementedError("UHF dipole exponential overlaps are not implemented.")
            if hasattr(mf, "dipole"):
                dipole_ao = mf.dipole(center=center, basis="ao")
            else:
                mol = getattr(state_model, "mol", getattr(mf, "mol", None))
                if mol is None or not hasattr(mol, "moment_integral"):
                    raise ValueError("Could not build the AO dipole operator.")
                if center is None:
                    center = mol.center_of_mass()
                dipole_ao = -np.asarray(
                    mol.moment_integral(center=np.asarray(center, dtype=float)),
                    dtype=float,
                )
            dipole_ao = np.asarray(dipole_ao)
            if dipole_ao.shape[0] != 3 and dipole_ao.shape[-1] == 3:
                dipole_ao = np.moveaxis(dipole_ao, -1, 0)
            dipole_mo = np.asarray(
                [mo_coeff.conj().T @ dipole_ao[xyz] @ mo_coeff for xyz in range(3)]
            )

    if isinstance(dipole_mo, (tuple, list)):
        raise NotImplementedError("Spin-dependent dipole exponential overlaps are not implemented.")

    dipole_mo = np.asarray(dipole_mo)
    if dipole_mo.ndim == 3:
        if dipole_mo.shape[0] != 3:
            if dipole_mo.shape[-1] == 3:
                dipole_mo = np.moveaxis(dipole_mo, -1, 0)
            else:
                raise ValueError("dipole_mo must have shape (3, nmo, nmo) or (nmo, nmo, 3).")
        mu = dipole_mo[axis]
    elif dipole_mo.ndim == 2:
        mu = dipole_mo
    else:
        raise ValueError("dipole_mo must be a rank-2 component or rank-3 Cartesian operator.")

    mu = np.asarray(mu, dtype=np.complex128)
    return 0.5 * (mu + mu.conj().T)


def dipole_orbital_rotation_unitary(
    state_model,
    eta_delta_q,
    *,
    axis="z",
    center=None,
    dipole_mo=None,
):
    """
    Build the one-particle orbital rotation for an exponential dipole link.

    The returned full-MO matrix is

    ``U = exp(1j * eta_delta_q * mu_axis)``

    where ``mu_axis`` is the electronic dipole operator in the MO basis. This
    is the one-particle representation of the many-electron operator used in
    geometric velocity-gauge links.
    """
    mu = _electric_dipole_mo_component(
        state_model,
        axis=axis,
        center=center,
        dipole_mo=dipole_mo,
    )
    evals, evecs = np.linalg.eigh(mu)
    phases = np.exp(1.0j * float(eta_delta_q) * evals)
    return (evecs * phases[np.newaxis, :]) @ evecs.conj().T


def orbital_rotation_ci_overlap(state_model, orbital_unitary, *, state_ids=None):
    """
    CI-root overlap after an exact one-particle orbital rotation.

    This evaluates

    ``<Psi_beta | Gamma(U) | Psi_alpha>``

    in the CASCI determinant model by passing the full one-particle MO overlap
    ``U`` to the generalized Slater-determinant overlap machinery. It therefore
    keeps core/active mixing and active/external leakage present in the full
    MO-space subblock, instead of exponentiating a truncated state-space dipole.
    """
    if getattr(state_model, "binary", None) is None or getattr(state_model, "ci", None) is None:
        raise ValueError("Run CASCI before requesting orbital-rotation overlaps.")
    if isinstance(getattr(getattr(state_model, "mf", None), "mo_coeff", None), (tuple, list)):
        raise NotImplementedError("UHF orbital-rotation CI overlaps are not implemented.")

    u = np.asarray(orbital_unitary, dtype=np.complex128)
    nmo = getattr(state_model, "nmo", None)
    if nmo is None:
        nmo = np.asarray(state_model.mf.mo_coeff).shape[1]
    if u.shape != (int(nmo), int(nmo)):
        raise ValueError(f"orbital_unitary has shape {u.shape}; expected {(int(nmo), int(nmo))}.")

    from pyqed.qchem.mcscf.casci import overlap as casci_overlap

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*encountered in det",
            category=RuntimeWarning,
        )
        overlap = casci_overlap(state_model, state_model, s=u)
    ids = _normalize_state_ids_for_overlap(state_model, state_ids)
    return np.asarray(overlap)[np.ix_(ids, ids)]


def dipole_exponential_ci_overlap(
    state_model,
    eta_delta_q,
    *,
    axis="z",
    center=None,
    state_ids=None,
    dipole_mo=None,
):
    """
    Many-electron CASCI overlap for ``exp(i * eta_delta_q * mu_axis)``.

    This is the orbital-rotation path for geometric velocity-gauge links:

    ``U_beta_alpha = <Psi_beta| exp(i * eta_delta_q * mu_hat) |Psi_alpha>``.

    The exponential is built in the full MO one-particle space and then lifted
    to determinant overlaps. This is distinct from, and generally more faithful
    than, ``expm(i * eta_delta_q * mu_state)`` in a truncated root basis.
    """
    u = dipole_orbital_rotation_unitary(
        state_model,
        eta_delta_q,
        axis=axis,
        center=center,
        dipole_mo=dipole_mo,
    )
    return orbital_rotation_ci_overlap(state_model, u, state_ids=state_ids)


def bo_hamiltonian_derivatives(
    state_model,
    state_ids=None,
    mode_vectors=None,
    overlap_tol=1e-8,
    *,
    moving_basis="symmetric",
    projected_only=False,
    derivative_order=2,
    backend="auto",
):
    """
    Build the first- and second-order derivatives of the Born-Oppenheimer
    electronic Hamiltonian in an electronic-state basis.

    Parameters
    ----------
    state_model
        Electronic-structure object that provides:
        ``mf.mol``, ``ncore``, ``ncas``, ``make_rdm1(state_id)``,
        ``make_rdm2(state_id)``, ``make_tdm1(bra_id, ket_id)``, and
        ``make_tdm2(bra_id, ket_id)`` for off-diagonal state couplings.
        The RDMs / TDMs are assumed to be in the active MO basis.
    state_ids : sequence of int, optional
        Electronic states to include. If omitted, they are inferred from
        ``state_model.ci`` or ``state_model.e_tot`` when possible.
    mode_vectors : ndarray, optional
        Optional Cartesian-to-mode projection matrix with shape
        ``(nmodes, natom, 3)``, ``(nmodes, 3*natom)``, or ``(3*natom, nmodes)``.
    overlap_tol : float, optional
        Kept for backward compatibility. Currently unused.
    moving_basis : {False, True, 'symmetric', 'rhf-relaxed', 'rhf-relaxed-pt'}, optional
        Include first- and second-order symmetric AO-overlap/Pulay transport in
        the local MO frame. ``'rhf-relaxed'`` additionally includes first-order
        CPHF orbital response. ``'rhf-relaxed-pt'`` removes the active-active
        orbital connection to express the relaxed derivative in the
        overlap-parallel CAS gauge. Both relaxed representations support first
        and second derivatives. ``True`` is an alias for ``'symmetric'``.
    projected_only : bool, optional
        If true and ``mode_vectors`` is supplied, build only the mode-projected
        derivative integrals. This avoids materializing full Cartesian second
        derivative ERIs for coarse-grained models.
    backend : {'auto', 'native', 'python', 'pyscf'}, optional
        Directional Gaussian derivative-integral backend. ``'native'`` uses the
        project-local C++ OS shell engine and ``'python'`` its reference
        implementation; ``'pyscf'`` uses libcint.

    Notes
    -----
    The returned Cartesian tensors correspond to derivatives with respect to
    nuclear Cartesian coordinates. Optional ``mode_vectors`` project these
    Cartesian derivatives to a coarse-grained coordinate set such as normal
    coordinates.
    """
    del overlap_tol
    derivative_order = int(derivative_order)
    if derivative_order not in (1, 2):
        raise ValueError("derivative_order must be 1 or 2")

    for attr in ('mf', 'ncore', 'ncas', 'make_rdm1', 'make_rdm2'):
        if not hasattr(state_model, attr):
            raise ValueError(
                f"{type(state_model).__name__} is missing required attribute/method '{attr}' "
                "for BO Hamiltonian derivatives."
            )

    mol = state_model.mf.mol
    natom = mol.natom
    ncart = 3 * natom

    state_ids = _infer_state_ids(state_model, state_ids)
    projected_only = bool(projected_only)
    if projected_only and mode_vectors is None:
        raise ValueError("projected_only=True requires mode_vectors.")
    mode_mat = None
    if mode_vectors is not None:
        mode_mat = _as_cartesian_mode_matrix(mode_vectors, natom)

    labels = tuple(
        f"{mol.atom_symbol(a)}{a}:{axis}"
        for a in range(natom)
        for axis in ("x", "y", "z")
    )

    derivative_integrals = _basis_derivative_integrals_mo(
        state_model,
        moving_basis=moving_basis,
        mode_vectors=mode_vectors if projected_only else None,
        order=derivative_order,
        backend=backend,
    )
    h1_cart = derivative_integrals["h1_ao"]
    h2_cart = derivative_integrals["h2_ao"]
    h1_mo = derivative_integrals["h1_mo"]
    h2_mo = derivative_integrals["h2_mo"]
    eri1_mo = derivative_integrals["eri1_mo"]
    eri2_mo = derivative_integrals["eri2_mo"]

    from pyqed.qchem.mol import grad_nuc

    nn_grad_cart = np.asarray(grad_nuc(mol), dtype=np.complex128).reshape(ncart)
    nn_hess_cart = (
        np.asarray(_nuclear_repulsion_hessian(mol), dtype=np.complex128)
        if derivative_order == 2
        else None
    )
    if projected_only:
        nn_grad = np.einsum("ka,a->k", mode_mat, nn_grad_cart, optimize=True)
        nn_hess = (
            np.einsum(
                "ka,lb,ab->kl", mode_mat, mode_mat, nn_hess_cart,
                optimize=True,
            )
            if derivative_order == 2
            else None
        )
    else:
        nn_grad = nn_grad_cart
        nn_hess = nn_hess_cart

    h1_active, eri1_active = _active_integral_derivatives_from_full_mo(
        h1_mo,
        eri1_mo,
        state_model.ncore,
        state_model.ncas,
    )
    core_grad = _core_energy_derivative_from_full_mo(
        h1_mo,
        eri1_mo,
        state_model.ncore,
    ) + nn_grad
    if derivative_order == 2:
        h2_active, eri2_active = _active_integral_derivatives_from_full_mo(
            h2_mo,
            eri2_mo,
            state_model.ncore,
            state_model.ncas,
        )
        core_hess = _core_energy_derivative_from_full_mo(
            h2_mo,
            eri2_mo,
            state_model.ncore,
        ) + nn_hess
    else:
        h2_active = None
        eri2_active = None
        core_hess = None

    nstates = len(state_ids)
    ncoord = h1_mo.shape[0]
    f_values = np.zeros((ncoord, nstates, nstates), dtype=np.complex128)
    g_values = (
        np.zeros((ncoord, ncoord, nstates, nstates), dtype=np.complex128)
        if derivative_order == 2
        else None
    )

    for ibra, bra in enumerate(state_ids):
        for iket, ket in enumerate(state_ids):
            state_overlap = _state_overlap(state_model, bra, ket)
            dm1, dm2 = _transition_active_rdms(state_model, bra, ket)
            f_values[:, ibra, iket] = (
                core_grad * state_overlap
                + np.einsum("xpq,qp->x", h1_active, dm1, optimize=True)
                + 0.5 * np.einsum("xpqrs,pqrs->x", eri1_active, dm2, optimize=True)
            )
            if derivative_order == 2:
                g_values[:, :, ibra, iket] = (
                    core_hess * state_overlap
                    + np.einsum("xypq,qp->xy", h2_active, dm1, optimize=True)
                    + 0.5 * np.einsum(
                        "xypqrs,pqrs->xy", eri2_active, dm2, optimize=True
                    )
                )

    if derivative_order == 2:
        g_values = 0.5 * (g_values + g_values.swapaxes(0, 1))

    f_proj = None
    g_proj = None
    if projected_only:
        f_cart = None
        g_cart = None
        f_proj = f_values
        g_proj = g_values
    else:
        f_cart = f_values
        g_cart = g_values
        if mode_vectors is not None:
            f_proj = np.einsum('ka,aij->kij', mode_mat, f_cart, optimize=True)
            if derivative_order == 2:
                g_proj = np.einsum(
                    'ka,lb,abij->klij',
                    mode_mat,
                    mode_mat,
                    g_cart,
                    optimize=True,
                )

    return BOHamiltonianDerivatives(
        state_ids=state_ids,
        cartesian_labels=labels,
        h1_ao_cartesian=h1_cart,
        h2_ao_cartesian=h2_cart,
        vnn_gradient_cartesian=nn_grad_cart,
        vnn_hessian_cartesian=nn_hess_cart,
        F_cartesian=f_cart,
        G_cartesian=g_cart,
        eri1_mo_cartesian=eri1_mo,
        eri2_mo_cartesian=eri2_mo,
        h1_mo_cartesian=h1_mo,
        h2_mo_cartesian=h2_mo,
        core_gradient_cartesian=core_grad,
        core_hessian_cartesian=core_hess,
        mode_vectors=mode_mat,
        F_projected=f_proj,
        G_projected=g_proj,
        moving_basis=derivative_integrals["moving_basis"],
    )


# Backward-compatible aliases for the earlier naming.
GeometricFGTerms = BOHamiltonianDerivatives
build_casci_bo_hamiltonian_derivatives = bo_hamiltonian_derivatives
build_casci_geometric_fg_terms = bo_hamiltonian_derivatives
