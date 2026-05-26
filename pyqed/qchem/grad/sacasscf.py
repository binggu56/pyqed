"""State-averaged CASSCF gradient-response contractions."""

from __future__ import annotations

import numpy as np

from pyqed.qchem.basis_derivatives import one_electron_derivatives, one_index_eri_derivatives
from pyqed.qchem.mcscf.casci import make_tdm1, make_tdm2
from pyqed.qchem.mcscf.orbopt import unpack_nonredundant


def _resolve_mo_coeff(backend, mo_coeff=None) -> np.ndarray:
    if mo_coeff is None:
        mo_coeff = getattr(backend.driver, "mo_coeff", None)
    if mo_coeff is None:
        mo_coeff = getattr(backend.mf, "mo_coeff", None)
    if mo_coeff is None:
        raise ValueError("mo_coeff must be supplied or available on backend.driver/backend.mf.")
    return np.asarray(mo_coeff, dtype=float)


def _ao_indices_by_atom(mol) -> list[np.ndarray]:
    labels = mol.ao_labels()
    groups: list[list[int]] = [[] for _ in range(int(mol.natom))]
    for idx, label in enumerate(labels):
        groups[int(str(label).split()[0])].append(idx)
    return [np.asarray(group, dtype=int) for group in groups]


def _pack_tril_diag_half(mats: np.ndarray) -> np.ndarray:
    mats = np.asarray(mats, dtype=float)
    nao = mats.shape[-1]
    pairs = [(p, q) for p in range(nao) for q in range(p + 1)]
    out = np.empty(mats.shape[:-2] + (len(pairs),), dtype=mats.dtype)
    for pair, (p, q) in enumerate(pairs):
        out[..., pair] = 0.5 * mats[..., p, q] if p == q else mats[..., p, q]
    return out


def _state_averaged_active_rdms(backend) -> tuple[np.ndarray, np.ndarray]:
    dm1 = np.zeros((backend.ncas, backend.ncas), dtype=float)
    dm2 = np.zeros((backend.ncas, backend.ncas, backend.ncas, backend.ncas), dtype=float)
    for root, weight in enumerate(backend.weights[: backend.nroots]):
        dm1 += float(weight) * backend.mc.make_rdm1(root, with_core=False)
        dm2 += float(weight) * backend.mc.make_rdm2(root, with_core=False)
    return dm1, dm2


def _transition_active_rdms_from_lci(backend, ci_z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lci = np.asarray(ci_z, dtype=float).reshape(backend.nroots, -1)
    ci = np.asarray(backend.roots[: backend.nroots], dtype=float)
    dm1 = np.zeros((backend.ncas, backend.ncas), dtype=float)
    dm2 = np.zeros((backend.ncas, backend.ncas, backend.ncas, backend.ncas), dtype=float)
    sc1 = getattr(backend.mc, "SC1", None)
    sc2 = getattr(backend.mc, "SC2", None)
    for root, weight in enumerate(backend.weights[: backend.nroots]):
        dm1 += float(weight) * make_tdm1(lci[root], ci[root], backend.mc.binary, sc1)
        dm2 += float(weight) * make_tdm2(lci[root], ci[root], backend.mc.binary, sc1, sc2)
    dm1 += dm1.T
    dm2 += dm2.transpose(1, 0, 3, 2)
    return dm1, dm2


def _native_first_index_get_jk(mol, dms) -> tuple[np.ndarray, np.ndarray]:
    dms = np.asarray(dms, dtype=float)
    if dms.ndim == 2:
        dms = dms[None, :, :]
    eri1 = one_index_eri_derivatives(mol, aosym="s1", convention="ip1")
    vj = -np.einsum("axijrs,nrs->naxij", eri1, dms, optimize=True)
    vk = -np.einsum("axijkl,njk->naxil", eri1, dms, optimize=True)
    return vj, vk


def _native_get_jk_many(mol, dms) -> tuple[np.ndarray, np.ndarray]:
    from pyqed.qchem.hf.rhf import get_jk as rhf_get_jk

    dms = np.asarray(dms, dtype=float)
    if dms.ndim == 2:
        dms = dms[None, :, :]
    vj = []
    vk = []
    for dm in dms:
        j, k = rhf_get_jk(mol, dm)
        vj.append(j)
        vk.append(k)
    return np.asarray(vj, dtype=float), np.asarray(vk, dtype=float)


def _active_pair_density_to_ao_pair(dm2_active: np.ndarray, mo_cas: np.ndarray) -> np.ndarray:
    dm2_ao = np.einsum("uvtw,pt,qw->uvpq", dm2_active, mo_cas, mo_cas, optimize=True)
    return _pack_tril_diag_half(dm2_ao).reshape(dm2_active.shape[0], dm2_active.shape[1], -1)


def _ao_lagrange_common(backend, mo_coeff=None):
    mo_coeff = _resolve_mo_coeff(backend, mo_coeff=mo_coeff)
    mol = backend.mf.mol
    ncore = backend.ncore
    ncas = backend.ncas
    nocc = ncore + ncas
    mo_core = mo_coeff[:, :ncore]
    mo_cas = mo_coeff[:, ncore:nocc]
    h1 = np.asarray(backend.mf.get_hcore(), dtype=float)
    _, eri_mo = backend.driver._get_integrals(mo_coeff)
    s0_inv = mo_coeff @ mo_coeff.T
    atom_aos = _ao_indices_by_atom(mol)
    hcore_deriv = one_electron_derivatives(mol, "hcore", order=1)
    s1 = one_electron_derivatives(mol, "overlap", order=1)
    eri1_s2kl = one_index_eri_derivatives(mol, aosym="s2kl", convention="ip1")
    return {
        "mo_coeff": mo_coeff,
        "mol": mol,
        "ncore": ncore,
        "ncas": ncas,
        "nocc": nocc,
        "mo_core": mo_core,
        "mo_cas": mo_cas,
        "h1": h1,
        "eri_mo": np.asarray(eri_mo, dtype=float),
        "s0_inv": s0_inv,
        "atom_aos": atom_aos,
        "hcore_deriv": hcore_deriv,
        "s1": s1,
        "eri1_s2kl": eri1_s2kl,
    }


def lorb_dot_dgorb_cartesian(backend, orbital_z: np.ndarray, *, mo_coeff=None) -> np.ndarray:
    """Return native AO ``Lorb · dGorb/dR`` for SA-CASSCF data."""

    orbital_z = np.asarray(orbital_z, dtype=float)
    if orbital_z.shape != (backend.orbital_size,):
        raise ValueError(f"orbital_z shape {orbital_z.shape} != {(backend.orbital_size,)}.")

    data = _ao_lagrange_common(backend, mo_coeff=mo_coeff)
    mo_coeff = data["mo_coeff"]
    mol = data["mol"]
    ncore = data["ncore"]
    ncas = data["ncas"]
    nocc = data["nocc"]
    mo_core = data["mo_core"]
    mo_cas = data["mo_cas"]
    h1 = data["h1"]
    eri_mo = data["eri_mo"]
    s0_inv = data["s0_inv"]
    hcore_deriv = data["hcore_deriv"]
    s1 = data["s1"]
    eri1_s2kl = data["eri1_s2kl"]
    atom_aos = data["atom_aos"]
    nao, nmo = mo_coeff.shape

    l_orb = unpack_nonredundant(orbital_z, backend.ncore, backend.ncas, backend.nmo)
    mo_l_coeff = mo_coeff @ l_orb
    mo_l_core = mo_l_coeff[:, :ncore]
    mo_l_cas = mo_l_coeff[:, ncore:nocc]

    casdm1, casdm2 = _state_averaged_active_rdms(backend)
    dm_core = mo_core @ mo_core.T * 2.0
    dm_cas = mo_cas @ casdm1 @ mo_cas.T
    dm_l_core = mo_l_core @ mo_core.T * 2.0
    dm_l_cas = mo_l_cas @ casdm1 @ mo_cas.T
    dm_l_core += dm_l_core.T
    dm_l_cas += dm_l_cas.T
    dm1 = dm_core + dm_cas
    dm1_l = dm_l_core + dm_l_cas

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=float)
    aapa_l = np.zeros_like(aapa)
    active = slice(ncore, nocc)
    for i in range(nmo):
        jbuf = eri_mo[i, :, active, active]
        kbuf = eri_mo[i, active, :, active]
        aapa[:, :, i, :] = jbuf[active, :, :].transpose(1, 2, 0)
        aapa_l[:, :, i, :] += np.tensordot(jbuf, l_orb[:, active], axes=(0, 0))
        ktmp = np.tensordot(kbuf, l_orb[:, active], axes=(1, 0)).transpose(1, 2, 0)
        aapa_l[:, :, i, :] += ktmp + ktmp.transpose(1, 0, 2)

    vj, vk = _native_get_jk_many(mol, (dm_core, dm_cas))
    vj_l, vk_l = _native_get_jk_many(mol, (dm_l_core, dm_l_cas))
    vhf_c = vj[0] - 0.5 * vk[0]
    vhf_a = vj[1] - 0.5 * vk[1]
    vhf_l_c = vj_l[0] - 0.5 * vk_l[0]
    vhf_l_a = vj_l[1] - 0.5 * vk_l[1]

    gfock = h1 @ dm1_l
    gfock += (vhf_c + vhf_a) @ dm_l_core
    gfock += (vhf_l_c + vhf_l_a) @ dm_core
    gfock += vhf_l_c @ dm_cas
    gfock += vhf_c @ dm_l_cas
    gfock = s0_inv @ gfock
    gfock += mo_coeff @ np.einsum("uviw,uvtw->it", aapa_l, casdm2, optimize=True) @ mo_cas.T
    gfock += mo_coeff @ np.einsum("uviw,vuwt->it", aapa, casdm2, optimize=True) @ mo_l_cas.T
    dme0 = 0.5 * (gfock + gfock.T)

    vj1, vk1 = _native_first_index_get_jk(mol, (dm_core, dm_cas, dm_l_core, dm_l_cas))
    vhf1c, vhf1a, vhf1c_l, vhf1a_l = vj1 - 0.5 * vk1

    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = _active_pair_density_to_ao_pair(casdm2_cc, mo_cas)

    dm2_lbuf = np.zeros((ncas * ncas, nmo, nmo), dtype=float)
    l_casdm2 = np.tensordot(l_orb[:, active], casdm2, axes=(1, 2)).transpose(1, 2, 0, 3)
    dm2_lbuf[:, :, active] = l_casdm2.reshape(ncas * ncas, nmo, ncas)
    l_casdm2 = np.tensordot(l_orb[:, active], casdm2, axes=(1, 3)).transpose(1, 2, 3, 0)
    dm2_lbuf[:, active, :] += l_casdm2.reshape(ncas * ncas, ncas, nmo)
    dm2_lbuf += dm2_lbuf.transpose(0, 2, 1)
    dm2_lbuf = dm2_lbuf.reshape(ncas, ncas, nmo, nmo)
    dm2_lbuf = np.einsum("uvij,pi,qj->uvpq", dm2_lbuf, mo_coeff, mo_coeff, optimize=True)
    dm2_lbuf = _pack_tril_diag_half(dm2_lbuf).reshape(ncas, ncas, -1)

    de = np.zeros((mol.natom, 3), dtype=float)
    all_q = np.arange(nao)
    for atom, pidx in enumerate(atom_aos):
        if pidx.size == 0:
            continue
        de[atom] += np.einsum("xpq,pq->x", hcore_deriv[atom], dm1_l, optimize=True)
        de[atom] -= np.einsum("xpq,pq->x", s1[atom], dme0, optimize=True)

        dm2_ao = np.einsum("ijw,pi,qj->pqw", dm2_lbuf, mo_cas[pidx], mo_cas, optimize=True)
        dm2_ao += np.einsum("ijw,pi,qj->pqw", dm2buf, mo_l_cas[pidx], mo_cas, optimize=True)
        dm2_ao += np.einsum("ijw,pi,qj->pqw", dm2buf, mo_cas[pidx], mo_l_cas, optimize=True)
        eri1 = eri1_s2kl[atom][:, pidx[:, None], all_q[None, :], :]
        de[atom] -= 2.0 * np.einsum("xpqw,pqw->x", eri1, dm2_ao, optimize=True)

        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1c[atom][:, pidx, :], dm1_l[pidx], optimize=True)
        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1c_l[atom][:, pidx, :], dm1[pidx], optimize=True)
        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1a[atom][:, pidx, :], dm_l_core[pidx], optimize=True)
        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1a_l[atom][:, pidx, :], dm_core[pidx], optimize=True)
    return de.reshape(-1)


def lci_dot_dgci_cartesian(backend, ci_z: np.ndarray, *, mo_coeff=None) -> np.ndarray:
    """Return native AO ``Lci · dGci/dR`` for SA-CASSCF data."""

    ci_z = np.asarray(ci_z, dtype=float)
    expected = int(backend.nroots) * int(backend.ndet)
    if ci_z.shape != (expected,):
        raise ValueError(f"ci_z shape {ci_z.shape} != {(expected,)}.")

    data = _ao_lagrange_common(backend, mo_coeff=mo_coeff)
    mo_coeff = data["mo_coeff"]
    mol = data["mol"]
    ncore = data["ncore"]
    ncas = data["ncas"]
    nocc = data["nocc"]
    mo_core = data["mo_core"]
    mo_cas = data["mo_cas"]
    h1 = data["h1"]
    eri_mo = data["eri_mo"]
    hcore_deriv = data["hcore_deriv"]
    s1 = data["s1"]
    eri1_s2kl = data["eri1_s2kl"]
    atom_aos = data["atom_aos"]
    nao, nmo = mo_coeff.shape

    casdm1, casdm2 = _transition_active_rdms_from_lci(backend, ci_z)
    dm_core = mo_core @ mo_core.T * 2.0
    dm_cas = mo_cas @ casdm1 @ mo_cas.T

    aapa = np.zeros((ncas, ncas, nmo, ncas), dtype=float)
    active = slice(ncore, nocc)
    for i in range(nmo):
        aapa[:, :, i, :] = eri_mo[i, :, active, active][active, :, :].transpose(1, 2, 0)

    vj, vk = _native_get_jk_many(mol, (dm_core, dm_cas))
    vhf_c = vj[0] - 0.5 * vk[0]
    vhf_a = vj[1] - 0.5 * vk[1]

    gfock = np.zeros((nmo, nmo), dtype=float)
    gfock[:, :nocc] = (mo_coeff.T @ vhf_a @ mo_coeff[:, :nocc]) * 2.0
    gfock[:, active] = mo_coeff.T @ (h1 + vhf_c) @ mo_cas @ casdm1
    gfock[:, active] += np.einsum("uvpw,vuwt->pt", aapa, casdm2, optimize=True)
    dme0 = mo_coeff @ (0.5 * (gfock + gfock.T)) @ mo_coeff.T

    vj1, vk1 = _native_first_index_get_jk(mol, (dm_core, dm_cas))
    vhf1c, vhf1a = vj1 - 0.5 * vk1

    casdm2_cc = casdm2 + casdm2.transpose(0, 1, 3, 2)
    dm2buf = _active_pair_density_to_ao_pair(casdm2_cc, mo_cas)

    de = np.zeros((mol.natom, 3), dtype=float)
    all_q = np.arange(nao)
    for atom, pidx in enumerate(atom_aos):
        if pidx.size == 0:
            continue
        de[atom] += np.einsum("xpq,pq->x", hcore_deriv[atom], dm_cas, optimize=True)
        de[atom] -= np.einsum("xpq,pq->x", s1[atom], dme0, optimize=True)

        dm2_ao = np.einsum("ijw,pi,qj->pqw", dm2buf, mo_cas[pidx], mo_cas, optimize=True)
        eri1 = eri1_s2kl[atom][:, pidx[:, None], all_q[None, :], :]
        de[atom] -= 2.0 * np.einsum("xpqw,pqw->x", eri1, dm2_ao, optimize=True)

        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1c[atom][:, pidx, :], dm_cas[pidx], optimize=True)
        de[atom] += 2.0 * np.einsum("xpq,pq->x", vhf1a[atom][:, pidx, :], dm_core[pidx], optimize=True)
    return de.reshape(-1)

