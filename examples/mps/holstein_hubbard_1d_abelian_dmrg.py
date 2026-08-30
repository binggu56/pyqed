#!/usr/bin/env python3
"""Abelian DMRG/MPS vs LETTA for the open 1D spinful Holstein-Hubbard model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.letta import LETTA
from pyqed.letta.abelian import Layout
from pyqed.mps import MPO, MPS, dense_to_symmetric_mpo, symmetric_to_dense
from pyqed.mps.dmrg import DMRG, _normalized_mps_mpo_expectation, dmrg_matvec_options
from pyqed.mps.mps import SpinHalfFermionOperators
from pyqed.mps.symmetry import AbelianSector
from pyqed.mps.abelian_storage import SymmetryManager
from pyqed.narg.holstein import (
    _displacement_overlap,
    boson_annihilation,
    spinful_holstein_hubbard_exact_hamiltonian,
)


ELECTRON_SECTORS = ((0, 0), (1, 0), (0, 1), (1, 1))
SECTOR_TO_ELECTRON = {sector: idx for idx, sector in enumerate(ELECTRON_SECTORS)}


def _as_complex_matrix(operator: np.ndarray) -> np.ndarray:
    return np.asarray(operator, dtype=complex)


def _kron_electron_phonon(electron: np.ndarray, phonon: np.ndarray) -> np.ndarray:
    return np.kron(_as_complex_matrix(electron), _as_complex_matrix(phonon))


def _spinful_fock_site_operators(nphonon: int, *, omega: float, coupling: float, hubbard_u: float):
    ops = SpinHalfFermionOperators()
    eye_e = np.eye(4, dtype=complex)
    eye_p = np.eye(int(nphonon), dtype=complex)
    b = boson_annihilation(int(nphonon), dtype=complex)
    bdag = b.T.conj()
    x = b + bdag
    num = bdag @ b
    local = (
        float(hubbard_u) * _kron_electron_phonon(ops["NuNd"], eye_p)
        + float(omega) * _kron_electron_phonon(eye_e, num)
        + float(coupling) * _kron_electron_phonon(ops["Ntot"], x)
    )
    out = {
        "I": np.eye(4 * int(nphonon), dtype=complex),
        "H": local,
        "JW": _kron_electron_phonon(ops["JW"], eye_p),
        "Ntot": _kron_electron_phonon(ops["Ntot"], eye_p),
        "Nu": _kron_electron_phonon(ops["Nu"], eye_p),
        "Nd": _kron_electron_phonon(ops["Nd"], eye_p),
        "NuNd": _kron_electron_phonon(ops["NuNd"], eye_p),
    }
    for name in ("Cu", "Cdu", "Cd", "Cdd"):
        out[name] = _kron_electron_phonon(ops[name], eye_p)
    out["PairAnn"] = out["Cd"] @ out["Cu"]
    out["PairCre"] = out["PairAnn"].T.conj()
    return out


def _put_block(matrix: np.ndarray, target: tuple[int, int], source: tuple[int, int], block: np.ndarray):
    active = block.shape[0]
    row = SECTOR_TO_ELECTRON[target] * active
    col = SECTOR_TO_ELECTRON[source] * active
    matrix[row : row + active, col : col + active] = block


def _spinful_polaron_site_operators(active_phonons: int, *, omega: float, coupling: float, hubbard_u: float):
    active = int(active_phonons)
    if active < 1:
        raise ValueError("active_phonons must be positive.")
    dim = 4 * active
    eye = np.eye(active, dtype=complex)
    levels = np.arange(active, dtype=float)
    charges = {
        (0, 0): (0, False),
        (1, 0): (1, False),
        (0, 1): (1, False),
        (1, 1): (2, True),
    }
    shifts = {}
    local = np.zeros((dim, dim), dtype=complex)
    ntot = np.zeros((dim, dim), dtype=complex)
    nu = np.zeros((dim, dim), dtype=complex)
    nd = np.zeros((dim, dim), dtype=complex)
    nund = np.zeros((dim, dim), dtype=complex)
    jw = np.zeros((dim, dim), dtype=complex)
    for sector, (charge, double) in charges.items():
        start = SECTOR_TO_ELECTRON[sector] * active
        shift = -float(coupling) * float(charge) / float(omega)
        energy = float(omega) * levels - (float(coupling) * float(charge)) ** 2 / float(omega)
        if double:
            energy = energy + float(hubbard_u)
        local[start : start + active, start : start + active] = np.diag(energy)
        ntot[start : start + active, start : start + active] = charge * eye
        nu[start : start + active, start : start + active] = sector[0] * eye
        nd[start : start + active, start : start + active] = sector[1] * eye
        nund[start : start + active, start : start + active] = (sector[0] * sector[1]) * eye
        jw[start : start + active, start : start + active] = ((-1) ** charge) * eye
        shifts[sector] = shift

    cu = np.zeros((dim, dim), dtype=complex)
    cd = np.zeros((dim, dim), dtype=complex)
    transitions = {
        "up": [((1, 0), (0, 0), 1.0), ((1, 1), (0, 1), 1.0)],
        "down": [((0, 1), (0, 0), 1.0), ((1, 1), (1, 0), -1.0)],
    }
    for source, target, sign in transitions["up"]:
        overlap = _displacement_overlap(shifts[source] - shifts[target], active)
        _put_block(cu, target, source, sign * overlap)
    for source, target, sign in transitions["down"]:
        overlap = _displacement_overlap(shifts[source] - shifts[target], active)
        _put_block(cd, target, source, sign * overlap)

    out = {
        "I": np.eye(dim, dtype=complex),
        "H": local,
        "JW": jw,
        "Cu": cu,
        "Cdu": cu.T.conj(),
        "Cd": cd,
        "Cdd": cd.T.conj(),
        "Ntot": ntot,
        "Nu": nu,
        "Nd": nd,
        "NuNd": nund,
    }
    out["PairAnn"] = out["Cd"] @ out["Cu"]
    out["PairCre"] = out["PairAnn"].T.conj()
    return out


def spinful_hh_site_operators(
    nphonon: int,
    *,
    hopping: float = 1.0,
    omega: float = 1.0,
    coupling: float = 1.0,
    hubbard_u: float = 4.0,
    phonon_basis: str = "polaron",
):
    """Return dense local operators for one electron-phonon supersite."""
    _ = hopping
    basis = str(phonon_basis).lower().replace("-", "_")
    if basis in {"polaron", "local_polaron", "natural_polaron"}:
        return _spinful_polaron_site_operators(
            int(nphonon),
            omega=float(omega),
            coupling=float(coupling),
            hubbard_u=float(hubbard_u),
        )
    if basis == "fock":
        return _spinful_fock_site_operators(
            int(nphonon),
            omega=float(omega),
            coupling=float(coupling),
            hubbard_u=float(hubbard_u),
        )
    raise ValueError("phonon_basis must be 'polaron' or 'fock'.")


def holstein_hubbard_mpo(
    nsites: int,
    nphonon: int,
    *,
    hopping: float = 1.0,
    omega: float = 1.0,
    coupling: float = 1.0,
    hubbard_u: float = 4.0,
    phonon_basis: str = "polaron",
) -> list[np.ndarray]:
    """Analytical open-chain spinful Holstein-Hubbard MPO.

    The convention matches ``pyqed.narg.holstein``:
    ``H_loc = U n_up n_down + omega b^dag b + g n (b + b^dag)``.
    In the polaron basis ``nphonon`` is the number of active oscillator states
    retained inside each local electronic charge sector.
    """
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be at least one.")
    ops = spinful_hh_site_operators(
        int(nphonon),
        hopping=float(hopping),
        omega=float(omega),
        coupling=float(coupling),
        hubbard_u=float(hubbard_u),
        phonon_basis=phonon_basis,
    )
    z = np.zeros_like(ops["I"])
    t = float(hopping)
    first = np.array(
        [[
            ops["H"],
            -t * ops["Cdu"] @ ops["JW"],
            -t * ops["Cdd"] @ ops["JW"],
            t * ops["Cu"] @ ops["JW"],
            t * ops["Cd"] @ ops["JW"],
            ops["I"],
        ]],
        dtype=complex,
    )
    bulk = np.array(
        [
            [ops["I"], z, z, z, z, z],
            [ops["Cu"], z, z, z, z, z],
            [ops["Cd"], z, z, z, z, z],
            [ops["Cdu"], z, z, z, z, z],
            [ops["Cdd"], z, z, z, z, z],
            [
                ops["H"],
                -t * ops["Cdu"] @ ops["JW"],
                -t * ops["Cdd"] @ ops["JW"],
                t * ops["Cu"] @ ops["JW"],
                t * ops["Cd"] @ ops["JW"],
                ops["I"],
            ],
        ],
        dtype=complex,
    )
    last = np.array(
        [[ops["I"]], [ops["Cu"]], [ops["Cd"]], [ops["Cdu"]], [ops["Cdd"]], [ops["H"]]],
        dtype=complex,
    )
    if nsites == 1:
        return [ops["H"].reshape(1, 1, ops["H"].shape[0], ops["H"].shape[1])]
    if nsites == 2:
        return [first, last]
    return [first] + [bulk.copy() for _ in range(nsites - 2)] + [last]


def transform_mpo_local_basis(mpo: list[np.ndarray], isometry: np.ndarray | list[np.ndarray]) -> list[np.ndarray]:
    """Project an MPO by local isometries with columns as the new basis."""
    if isinstance(isometry, (list, tuple)):
        isometries = [np.asarray(v, dtype=complex) for v in isometry]
        if len(isometries) != len(mpo):
            raise ValueError("one isometry per MPO site is required.")
    else:
        isometries = [np.asarray(isometry, dtype=complex) for _ in mpo]
    projected = []
    for site, (w, v) in enumerate(zip(mpo, isometries)):
        w = np.asarray(w, dtype=complex)
        if w.shape[2] != v.shape[0] or w.shape[3] != v.shape[0]:
            raise ValueError(f"isometry dimension mismatch at site {site}.")
        projected.append(np.einsum("ya,lryz,zb->lrab", v.conj(), w, v, optimize=True))
    return projected


def charge_resolved_phonon_isometry(nphonon: int, active_phonons: int) -> np.ndarray:
    """Block-diagonal electronic-sector isometry keeping low phonon levels."""
    nphonon = int(nphonon)
    active = int(active_phonons)
    if active < 1 or nphonon < active:
        raise ValueError("active_phonons must be between 1 and nphonon.")
    iso = np.zeros((4 * nphonon, 4 * active), dtype=complex)
    for electronic in range(4):
        row = electronic * nphonon
        col = electronic * active
        iso[row : row + active, col : col + active] = np.eye(active)
    return iso


def charge_resolved_natural_phonon_isometry(
    local_rdms: list[np.ndarray],
    nphonon: int,
    active_phonons: int,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Build a charge-resolved natural-phonon isometry from local RDMs."""
    nphonon = int(nphonon)
    active = int(active_phonons)
    if active < 1 or nphonon < active:
        raise ValueError("active_phonons must be between 1 and nphonon.")
    rho_by_e = [np.zeros((nphonon, nphonon), dtype=complex) for _ in range(4)]
    for rho in local_rdms:
        rho = np.asarray(rho, dtype=complex)
        if rho.shape != (4 * nphonon, 4 * nphonon):
            raise ValueError("each local RDM must have shape (4*nphonon, 4*nphonon).")
        for electronic in range(4):
            start = electronic * nphonon
            rho_by_e[electronic] += rho[start : start + nphonon, start : start + nphonon]

    iso = np.zeros((4 * nphonon, 4 * active), dtype=complex)
    discarded = np.empty(4, dtype=float)
    total_weight = np.empty(4, dtype=float)
    spectra = np.empty((4, nphonon), dtype=float)
    for electronic, rho in enumerate(rho_by_e):
        rho = 0.5 * (rho + rho.T.conj())
        evals, evecs = np.linalg.eigh(rho)
        order = np.argsort(evals.real)[::-1]
        evals = np.maximum(evals.real[order], 0.0)
        evecs = evecs[:, order]
        row = electronic * nphonon
        col = electronic * active
        iso[row : row + nphonon, col : col + active] = evecs[:, :active]
        spectra[electronic] = evals
        total_weight[electronic] = float(np.sum(evals))
        discarded[electronic] = float(np.sum(evals[active:]))

    info = {
        "discarded_weight": discarded,
        "total_weight": total_weight,
        "spectra": spectra,
    }
    return iso, info


def _standard_dense_factors(factors_or_mps) -> list[np.ndarray]:
    if isinstance(factors_or_mps, MPS):
        return [np.asarray(tensor) for tensor in factors_or_mps.to_order(["lv", "p", "rv"]).factors]
    if hasattr(factors_or_mps, "factors"):
        return _standard_dense_factors(MPS(factors_or_mps.factors, labels=getattr(factors_or_mps, "labels", ["lv", "p", "rv"])))
    return [np.asarray(tensor) for tensor in factors_or_mps]


def dense_mps_product_expectation(factors_or_mps, operators: list[np.ndarray]) -> complex:
    """Normalized product-operator expectation for dense MPS factors."""
    factors = _standard_dense_factors(factors_or_mps)
    if len(factors) != len(operators):
        raise ValueError("number of operators must match MPS length.")
    env = np.ones((1, 1), dtype=complex)
    norm_env = np.ones((1, 1), dtype=complex)
    for tensor, operator in zip(factors, operators):
        tensor = np.asarray(tensor, dtype=complex)
        operator = np.asarray(operator, dtype=complex)
        ident = np.eye(tensor.shape[1], dtype=complex)
        env = np.einsum("ab,atr,ts,bsu->ru", env, tensor.conj(), operator, tensor, optimize=True)
        norm_env = np.einsum("ab,atr,ts,bsu->ru", norm_env, tensor.conj(), ident, tensor, optimize=True)
    norm = norm_env.reshape(-1)[0]
    if abs(norm) < 1.0e-14:
        raise ValueError("MPS norm is numerically zero.")
    return env.reshape(-1)[0] / norm


def dense_mps_local_rdms(factors_or_mps) -> list[np.ndarray]:
    """Return normalized one-site RDMs for a dense MPS."""
    factors = _standard_dense_factors(factors_or_mps)
    nsites = len(factors)
    identities = [np.eye(factor.shape[1], dtype=complex) for factor in factors]
    rdms = []
    for site, factor in enumerate(factors):
        dim = factor.shape[1]
        rho = np.empty((dim, dim), dtype=complex)
        for out_s in range(dim):
            for in_s in range(dim):
                op = np.zeros((dim, dim), dtype=complex)
                op[out_s, in_s] = 1.0
                operators = list(identities)
                operators[site] = op
                rho[out_s, in_s] = dense_mps_product_expectation(factors, operators)
        rho = 0.5 * (rho + rho.T.conj())
        trace = np.trace(rho)
        if abs(trace) > 1.0e-14:
            rho = rho / trace
        rdms.append(rho)
    return rdms


def dense_mps_correlation_matrix(factors_or_mps, op_a: np.ndarray, op_b: np.ndarray | None = None) -> np.ndarray:
    factors = _standard_dense_factors(factors_or_mps)
    op_a = np.asarray(op_a, dtype=complex)
    op_b = op_a if op_b is None else np.asarray(op_b, dtype=complex)
    identities = [np.eye(factor.shape[1], dtype=complex) for factor in factors]
    nsites = len(factors)
    corr = np.empty((nsites, nsites), dtype=complex)
    for i in range(nsites):
        for j in range(nsites):
            operators = list(identities)
            if i == j:
                operators[i] = op_a @ op_b
            else:
                operators[i] = op_a
                operators[j] = op_b
            corr[i, j] = dense_mps_product_expectation(factors, operators)
    return corr


def distance_average(correlation: np.ndarray) -> list[float]:
    correlation = np.asarray(correlation)
    return [
        float(np.real(np.mean([correlation[i, i + r] for i in range(correlation.shape[0] - r)])))
        for r in range(correlation.shape[0])
    ]


def spinful_hh_site_qn_maps(nsites: int, active_phonons: int):
    q_empty = AbelianSector(("charge", "sz"), (0, 0))
    q_up = AbelianSector(("charge", "sz"), (1, 1))
    q_down = AbelianSector(("charge", "sz"), (1, -1))
    q_full = AbelianSector(("charge", "sz"), (2, 0))
    phys = [q_empty, q_up, q_down, q_full]
    local = {}
    for electronic, qn in enumerate(phys):
        for phonon in range(int(active_phonons)):
            local[electronic * int(active_phonons) + phonon] = qn
    return [dict(local) for _ in range(int(nsites))]


def _expanded_abelian_leg_labels(tensor, leg: int) -> list[tuple[int, ...]]:
    """Expand an Abelian tensor leg label once per dense degeneracy slot."""
    from collections import Counter

    qns = list(tensor.qns[int(leg)])
    dims = dict(Counter(qns))
    for key, block in tensor.data.items():
        q = key[int(leg)]
        dims[q] = max(int(dims.get(q, 0)), int(block.shape[int(leg)]))
    labels = []
    seen = set()
    for q in qns:
        if q in seen:
            continue
        seen.add(q)
        labels.extend([tuple(int(x) for x in q)] * int(dims[q]))
    return labels


def letta_layout_from_abelian_mps(sym_mps, site_qn_maps, target_qn) -> Layout:
    """Build a LETTA fixed-sector layout from Abelian MPS bond sectors."""
    factors = sym_mps.factors
    if len(factors) < 2:
        raise ValueError("LETTA layout needs at least two sites.")
    local_qns = [
        [tuple(int(x) for x in site_map[index]) for index in sorted(site_map)]
        for site_map in site_qn_maps
    ]
    bond_qns = [_expanded_abelian_leg_labels(factors[0], 0)]
    for site in range(len(factors) - 2):
        bond_qns.append(_expanded_abelian_leg_labels(factors[site], 1))
    target = tuple(int(x) for x in target_qn)
    return Layout(local_qns=local_qns, bond_qns=bond_qns, target=target)


def fixed_filling_product_mps(
    nsites: int,
    active_phonons: int,
    nup: int,
    ndown: int,
) -> list[np.ndarray]:
    """Distributed fixed-spin product state with phonon level zero."""
    nsites = int(nsites)
    active_phonons = int(active_phonons)
    nup = int(nup)
    ndown = int(ndown)
    if nsites < 1 or active_phonons < 1:
        raise ValueError("nsites and active_phonons must be positive.")
    if not 0 <= nup <= nsites or not 0 <= ndown <= nsites:
        raise ValueError("nup and ndown must lie between zero and nsites.")

    electronic_states = np.zeros(nsites, dtype=np.int64)
    nelectrons = nup + ndown
    if nelectrons <= nsites:
        occupied = np.floor(
            (np.arange(nelectrons, dtype=float) + 0.5) * nsites / max(nelectrons, 1)
        ).astype(np.int64)
        remaining = {1: nup, 2: ndown}
        previous = 2 if nup >= ndown else 1
        for site in occupied:
            preferred = 1 if previous == 2 else 2
            electronic = preferred if remaining[preferred] else 3 - preferred
            electronic_states[site] = electronic
            remaining[electronic] -= 1
            previous = electronic
    else:
        up_sites = np.floor(
            (np.arange(nup, dtype=float) + 0.25) * nsites / nup
        ).astype(np.int64)
        down_sites = np.floor(
            (np.arange(ndown, dtype=float) + 0.75) * nsites / ndown
        ).astype(np.int64)
        electronic_states[up_sites] += 1
        electronic_states[down_sites] += 2

    factors = []
    dim = 4 * active_phonons
    for electronic in electronic_states:
        physical = int(electronic) * active_phonons
        tensor = np.zeros((1, dim, 1), dtype=complex)
        tensor[0, physical, 0] = 1.0
        factors.append(tensor)
    return factors


def half_filled_product_mps(nsites: int, active_phonons: int) -> list[np.ndarray]:
    """Neel-like one-electron-per-site product state with phonon level zero."""
    if int(nsites) % 2:
        raise ValueError("balanced half filling requires an even number of sites.")
    return fixed_filling_product_mps(
        nsites,
        active_phonons,
        int(nsites) // 2,
        int(nsites) // 2,
    )


def expand_active_phonon_mps(
    factors,
    old_active_phonons: int,
    new_active_phonons: int,
) -> list[np.ndarray]:
    """Embed an MPS exactly into a larger charge-resolved phonon basis."""
    old_active = int(old_active_phonons)
    new_active = int(new_active_phonons)
    if old_active < 1 or new_active < old_active:
        raise ValueError("new_active_phonons must be at least old_active_phonons.")
    expanded = []
    for factor in factors:
        factor = np.asarray(factor)
        if factor.ndim != 3 or factor.shape[1] != 4 * old_active:
            raise ValueError("each MPS factor must have physical dimension 4*old_active_phonons.")
        out = np.zeros(
            (factor.shape[0], 4 * new_active, factor.shape[2]),
            dtype=factor.dtype,
        )
        for electronic in range(4):
            old = slice(electronic * old_active, (electronic + 1) * old_active)
            new = slice(electronic * new_active, electronic * new_active + old_active)
            out[:, new, :] = factor[:, old, :]
        expanded.append(out)
    return expanded


def _dense_mpo_to_fixed_sector_matrix(mpo: list[np.ndarray], nup: int, ndown: int) -> np.ndarray:
    """Small-system reference helper: project a dense MPO to fixed spin sector."""
    from pyqed.mps.mps import _mpo_to_dense_operator

    nsites = len(mpo)
    dim = mpo[0].shape[2]
    nphonon = dim // 4
    full = _mpo_to_dense_operator(type("MPOList", (), {"factors": mpo, "dims": (dim,) * nsites})())
    keep = []
    for flat in range(dim**nsites):
        value = flat
        up = 0
        down = 0
        for _site in range(nsites):
            local = value % dim
            value //= dim
            electronic = local // nphonon
            sector = ELECTRON_SECTORS[electronic]
            up += sector[0]
            down += sector[1]
        if up == int(nup) and down == int(ndown):
            keep.append(flat)
    return full[np.ix_(keep, keep)]


def _metrics_from_state(factors_or_letta, operators: dict[str, np.ndarray], *, is_letta: bool = False):
    ntot = operators["Ntot"]
    dcharge = ntot - np.eye(ntot.shape[0], dtype=complex)
    pair_cre = operators["PairCre"]
    pair_ann = operators["PairAnn"]
    if is_letta:
        state = factors_or_letta
        density = np.array(
            [
                state.expectation_product_operator(
                    [ntot if j == i else np.eye(ntot.shape[0], dtype=complex) for j in range(state.nsites)]
                )
                for i in range(state.nsites)
            ],
            dtype=complex,
        )
        cdw_corr = state.spatial_correlation(dcharge, dcharge, connected=False, average=False)
        pair_corr = state.spatial_correlation(pair_cre, pair_ann, connected=False, average=False)
    else:
        factors = _standard_dense_factors(factors_or_letta)
        identities = [np.eye(tensor.shape[1], dtype=complex) for tensor in factors]
        density = np.array(
            [
                dense_mps_product_expectation(
                    factors,
                    [ntot if j == i else identities[j] for j in range(len(factors))],
                )
                for i in range(len(factors))
            ],
            dtype=complex,
        )
        cdw_corr = dense_mps_correlation_matrix(factors, dcharge, dcharge)
        pair_corr = dense_mps_correlation_matrix(factors, pair_cre, pair_ann)
    nsites = int(density.shape[0])
    stagger = np.array([(-1) ** i for i in range(nsites)], dtype=float)
    cdw_order = abs(np.sum(stagger * (density.real - 1.0)) / nsites)
    cdw_structure = float(np.real(np.einsum("i,ij,j->", stagger, cdw_corr, stagger) / (nsites * nsites)))
    pair_by_distance = distance_average(pair_corr)
    return {
        "density": [float(np.real(value)) for value in density],
        "cdw_order": float(cdw_order),
        "cdw_structure_pi": cdw_structure,
        "pair_pair_by_distance": pair_by_distance,
        "pair_pair_edge": float(np.real(pair_corr[0, -1])),
        "pair_pair_mid_far": float(np.real(pair_corr[nsites // 4, nsites - 1 - nsites // 4])),
    }


def run_one_active(args, active: int, *, initial_factors=None) -> tuple[dict, list[np.ndarray]]:
    dense_mpo = holstein_hubbard_mpo(
        args.nsites,
        active,
        hopping=args.hopping,
        omega=args.omega,
        coupling=args.coupling,
        hubbard_u=args.hubbard_u,
        phonon_basis=args.basis,
    )
    site_qn_maps = spinful_hh_site_qn_maps(args.nsites, active)
    sym_mgr = SymmetryManager(["charge", "sz"])
    target_qn = sym_mgr.get_target_qn(
        int(args.nup) + int(args.ndown),
        int(args.nup) - int(args.ndown),
    )
    matvec_options = dmrg_matvec_options(args.dmrg_policy)
    symmetric_mpo = dense_to_symmetric_mpo(
        dense_mpo,
        site_qn_maps,
        native_site_storage=bool(matvec_options.get("native_site_storage", False)),
    )
    initial = (
        fixed_filling_product_mps(
            args.nsites,
            active,
            args.nup,
            args.ndown,
        )
        if initial_factors is None
        else [np.asarray(factor).copy() for factor in initial_factors]
    )

    start = perf_counter()
    dmrg = DMRG(
        MPO(symmetric_mpo),
        D=int(args.bond_dim),
        init_guess=initial,
        nsweeps=int(args.sweeps),
        opt="2site",
        symmetry=True,
        target_qn=target_qn,
        sym_mgr=sym_mgr,
        site_qn_maps=site_qn_maps,
        not_conv_err=False,
        verbose=int(args.verbose),
        sweep_tol=float(args.sweep_tol),
        davidson_tol=float(args.davidson_tol),
        davidson_max_iter=int(args.davidson_max_iter),
        noise=float(args.noise),
        noise_decay=float(args.noise_decay),
        performance=args.dmrg_policy,
        abelian_matvec_options=matvec_options,
    )
    dmrg.run()
    dmrg_time = perf_counter() - start
    letta_layout = letta_layout_from_abelian_mps(dmrg.state, site_qn_maps, target_qn)
    dense_mps = symmetric_to_dense(dmrg.state, site_qn_maps=site_qn_maps)
    dmrg_energy = _normalized_mps_mpo_expectation(dense_mps.factors, dense_mpo)

    operators = spinful_hh_site_operators(
        active,
        hopping=args.hopping,
        omega=args.omega,
        coupling=args.coupling,
        hubbard_u=args.hubbard_u,
        phonon_basis=args.basis,
    )
    dmrg_metrics = _metrics_from_state(dense_mps, operators)

    letta = LETTA.from_mps(
        dense_mps,
        hamiltonian=None,
        dims=[4 * active] * int(args.nsites),
        abelian_layout=letta_layout,
        seed=int(args.seed),
    )
    start = perf_counter()
    letta.run(
        dense_mpo,
        nsweeps=int(args.letta_sweeps),
        tol=float(args.letta_tol),
        verbose=int(args.verbose),
        local_solver=args.letta_solver,
        matrix_free_threshold=int(args.letta_matrix_free_threshold),
        matrix_free_tol=float(args.letta_matrix_free_tol),
        matrix_free_maxiter=int(args.letta_matrix_free_maxiter),
    )
    letta_time = perf_counter() - start
    letta_energy = float(letta.expectation_mpo(dense_mpo))
    letta_metrics = _metrics_from_state(letta, operators, is_letta=True)

    row = {
        "active_phonons": int(active),
        "local_dim": int(4 * active),
        "basis": args.basis,
        "dmrg_energy": float(dmrg_energy),
        "letta_energy": float(letta_energy),
        "letta_minus_dmrg": float(letta_energy - dmrg_energy),
        "dmrg_time_s": float(dmrg_time),
        "letta_time_s": float(letta_time),
        "dmrg_converged": bool(dmrg.converged),
        "letta_converged": bool(letta.converged),
        "dmrg_last_sweep_energy": None if not dmrg.sweep_history else dmrg.sweep_history[-1].get("energy"),
        "letta_history": [
            {
                "sweep": int(entry["sweep"]),
                "energy": float(np.real(entry["energy"])),
                "delta_energy": None if entry["delta_energy"] is None else float(entry["delta_energy"]),
            }
            for entry in letta.history
        ],
        "dmrg": dmrg_metrics,
        "letta": letta_metrics,
    }
    return row, [np.asarray(factor).copy() for factor in dense_mps.factors]


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=12)
    parser.add_argument("--nup", type=int, default=None)
    parser.add_argument("--ndown", type=int, default=None)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=4.0)
    parser.add_argument("--omega", type=float, default=0.5)
    parser.add_argument("-g", "--coupling", type=float, default=1.2)
    parser.add_argument("--basis", choices=("polaron", "fock"), default="polaron")
    parser.add_argument("--active-phonons", type=int, nargs="+", default=[2, 3, 4])
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=4)
    parser.add_argument("--letta-sweeps", type=int, default=2)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--letta-tol", type=float, default=1.0e-7)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-6)
    parser.add_argument("--davidson-max-iter", type=int, default=36)
    parser.add_argument("--noise", type=float, default=1.0e-6)
    parser.add_argument("--noise-decay", type=float, default=0.2)
    parser.add_argument("--dmrg-policy", default="packed-fast")
    parser.add_argument("--letta-solver", default="auto")
    parser.add_argument("--letta-matrix-free-threshold", type=int, default=4096)
    parser.add_argument("--letta-matrix-free-tol", type=float, default=1.0e-8)
    parser.add_argument("--letta-matrix-free-maxiter", type=int, default=80)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="/private/tmp/hh_l12_mps_vs_letta.json")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)
    if args.nup is None:
        if args.nsites % 2:
            raise ValueError("half-filled balanced default requires even L.")
        args.nup = args.nsites // 2
    if args.ndown is None:
        if args.nsites % 2:
            raise ValueError("half-filled balanced default requires even L.")
        args.ndown = args.nsites // 2
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "Spinful HH MPS vs LETTA: "
        f"L={args.nsites}, Nup={args.nup}, Ndown={args.ndown}, "
        f"t={args.hopping:g}, U={args.hubbard_u:g}, omega={args.omega:g}, g={args.coupling:g}, "
        f"basis={args.basis}, D={args.bond_dim}"
    )
    print("Convention: local coupling is g*n_i*(b_i + b_i^dag), matching pyqed.narg.holstein.")
    rows = []
    previous_active = None
    previous_mps = None
    for active in args.active_phonons:
        print(f"\nactive_phonons={active} local_dim={4 * int(active)}")
        active = int(active)
        initial_factors = None
        warm_start_active = None
        if previous_mps is not None and active >= previous_active:
            initial_factors = expand_active_phonon_mps(
                previous_mps,
                previous_active,
                active,
            )
            warm_start_active = int(previous_active)
            print(f"  warm start: embedded active_phonons={previous_active} MPS")
        row, previous_mps = run_one_active(
            args,
            active,
            initial_factors=initial_factors,
        )
        row["warm_start_active_phonons"] = warm_start_active
        previous_active = active
        rows.append(row)
        print(
            f"  E_MPS   = {row['dmrg_energy']: .12f}  "
            f"time={row['dmrg_time_s']:.2f}s  conv={row['dmrg_converged']}"
        )
        print(
            f"  E_LETTA = {row['letta_energy']: .12f}  "
            f"time={row['letta_time_s']:.2f}s  conv={row['letta_converged']}  "
            f"dE={row['letta_minus_dmrg']:+.3e}"
        )
        print(
            f"  CDW(MPS,LETTA)=({row['dmrg']['cdw_order']:.6f}, {row['letta']['cdw_order']:.6f})  "
            f"Spi=({row['dmrg']['cdw_structure_pi']:.6f}, {row['letta']['cdw_structure_pi']:.6f})"
        )
        print(
            f"  Pair edge(MPS,LETTA)=({row['dmrg']['pair_pair_edge']:.6e}, "
            f"{row['letta']['pair_pair_edge']:.6e})"
        )
        print(
            "  Pair C(r) MPS   = "
            + " ".join(f"{value:.3e}" for value in row["dmrg"]["pair_pair_by_distance"][1:])
        )
        print(
            "  Pair C(r) LETTA = "
            + " ".join(f"{value:.3e}" for value in row["letta"]["pair_pair_by_distance"][1:])
        )

    result = {
        "params": {
            "nsites": int(args.nsites),
            "nup": int(args.nup),
            "ndown": int(args.ndown),
            "hopping": float(args.hopping),
            "hubbard_u": float(args.hubbard_u),
            "omega": float(args.omega),
            "coupling": float(args.coupling),
            "basis": args.basis,
            "bond_dim": int(args.bond_dim),
            "sweeps": int(args.sweeps),
            "letta_sweeps": int(args.letta_sweeps),
            "dmrg_policy": args.dmrg_policy,
            "active_phonon_continuation": True,
        },
        "rows": rows,
    }
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nSaved JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
