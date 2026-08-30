#!/usr/bin/env python3
"""Open 2D Hubbard Abelian DMRG/MPS vs fixed-sector LETTA benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy.sparse.linalg import eigsh

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

try:
    from hubbard_2d_ed import hubbard_2d_hamiltonian, square_lattice_bonds
except ImportError:  # pragma: no cover - used when imported as examples.mps.*
    from examples.mps.hubbard_2d_ed import hubbard_2d_hamiltonian, square_lattice_bonds
from pyqed.letta import LETTA, NNNLETTA
from pyqed.letta.abelian import Layout
from pyqed.lattice import Site, SpinHalfFermionSite
from pyqed.mps import MPO, MPS, dense_to_symmetric_mpo, symmetric_to_dense
from pyqed.mps.abelian_storage import make_abelian_site_tensor
from pyqed.mps.dmrg import DMRG, _normalized_mps_mpo_expectation, dmrg_matvec_options
from pyqed.mps.mps import SpinHalfFermionOperators, _mpo_to_dense_operator
from pyqed.mps.symmetry import AbelianSector
from pyqed.mps.abelian_storage import SymmetryManager
from pyqed.qchem.dmrg.dmrg import (
    _build_spin_orbital_dense_hamiltonian_tensor_mpo,
    _group_spin_orbital_mpo_pairs,
)
from pyqed.tn import Hamiltonian


LOCAL_QNS = (
    (0, 0),   # empty
    (1, 1),   # up
    (1, -1),  # down
    (2, 0),   # double
)


def lattice_site_order(lx: int, ly: int, *, ordering: str = "snake") -> list[int]:
    """Return original lattice-site labels in the selected MPS order."""
    key = str(ordering).lower().replace("_", "-")
    if key in {"row", "row-major", "rowmajor"}:
        return [x + int(lx) * y for y in range(int(ly)) for x in range(int(lx))]
    if key in {"column", "column-major", "columnmajor", "rung", "rung-major"}:
        return [x + int(lx) * y for x in range(int(lx)) for y in range(int(ly))]
    if key == "snake":
        order = []
        for y in range(int(ly)):
            xs = range(int(lx)) if y % 2 == 0 else range(int(lx) - 1, -1, -1)
            order.extend(x + int(lx) * y for x in xs)
        return order
    raise ValueError("ordering must be 'row-major', 'column-major', or 'snake'.")


def _fuse_two_mpo_sites(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Fuse two neighboring dense MPO factors into one physical supersite."""
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.shape[1] != right.shape[0]:
        raise ValueError("MPO bond mismatch while fusing adjacent sites.")
    pair = np.tensordot(left, right, axes=([1], [0]))
    return pair.transpose(0, 3, 1, 4, 2, 5).reshape(
        left.shape[0],
        right.shape[1],
        left.shape[2] * right.shape[2],
        left.shape[3] * right.shape[3],
    )


def fuse_adjacent_mpo_sites(mpo: list[np.ndarray], *, group_size: int = 2) -> list[np.ndarray]:
    """Fuse adjacent dense MPO physical sites into fixed-width supersites."""
    group_size = int(group_size)
    if group_size < 1:
        raise ValueError("group_size must be positive.")
    if group_size == 1:
        return [np.asarray(factor, dtype=complex) for factor in mpo]
    if len(mpo) % group_size:
        raise ValueError("MPO length must be divisible by group_size.")
    fused = []
    for i in range(0, len(mpo), group_size):
        block = np.asarray(mpo[i], dtype=complex)
        for j in range(i + 1, i + group_size):
            block = _fuse_two_mpo_sites(block, mpo[j])
        fused.append(block)
    return fused


def ordered_lattice_bonds(
    lx: int,
    ly: int,
    *,
    ordering: str,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Nearest-neighbor bonds in the MPS orbital index convention."""
    order = lattice_site_order(lx, ly, ordering=ordering)
    lattice_to_orbital = {site: orbital for orbital, site in enumerate(order)}
    bonds = [
        (lattice_to_orbital[i], lattice_to_orbital[j])
        for i, j in square_lattice_bonds(lx, ly, periodic_x=periodic_x, periodic_y=periodic_y)
    ]
    return bonds, order


def hubbard_2d_dense_mpo(
    lx: int,
    ly: int,
    *,
    hopping: float,
    hubbard_u: float,
    mu: float = 0.0,
    ordering: str = "snake",
    periodic_x: bool = False,
    periodic_y: bool = False,
    site_grouping: str = "site",
) -> tuple[list[np.ndarray], dict]:
    """Build a spatial-site dense MPO for the open/periodic 2D Hubbard model."""
    nsites = int(lx) * int(ly)
    grouping = str(site_grouping).lower().replace("_", "-")
    if grouping not in {"site", "rung"}:
        raise ValueError("site_grouping must be 'site' or 'rung'.")
    if grouping == "rung":
        if int(ly) < 2:
            raise ValueError("rung supersite grouping requires ly >= 2.")
        ordering = "column-major"
    bonds, order = ordered_lattice_bonds(
        lx,
        ly,
        ordering=ordering,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
    )
    h1 = np.zeros((nsites, nsites), dtype=float)
    for i, j in bonds:
        h1[i, j] += -float(hopping)
        h1[j, i] += -float(hopping)
    if mu:
        h1 -= float(mu) * np.eye(nsites)

    eri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    for site in range(nsites):
        eri[site, site, site, site] = float(hubbard_u)
    h2 = np.stack((np.stack((eri, eri.copy())), np.stack((eri.copy(), eri.copy()))))

    spin_mpo, one_two_terms, spin_terms = _build_spin_orbital_dense_hamiltonian_tensor_mpo(
        [h1, h1],
        h2,
        nsites,
    )
    grouped = _group_spin_orbital_mpo_pairs(spin_mpo)
    dense_mpo = [np.asarray(factor, dtype=complex) for factor in grouped.factors]
    fused_blocks = None
    if grouping == "rung":
        fused_blocks = [order[i : i + int(ly)] for i in range(0, len(order), int(ly))]
        dense_mpo = fuse_adjacent_mpo_sites(dense_mpo, group_size=int(ly))
    info = {
        "nsites": len(dense_mpo),
        "spatial_sites": nsites,
        "site_grouping": grouping,
        "periodic_x": bool(periodic_x),
        "periodic_y": bool(periodic_y),
        "order": [int(site) for site in order],
        "fused_blocks": None if fused_blocks is None else [[int(site) for site in block] for block in fused_blocks],
        "bonds": [(int(i), int(j)) for i, j in bonds],
        "mpo_bond_dims": [int(factor.shape[1]) for factor in dense_mpo],
        "symbolic_terms": int(one_two_terms),
        "spin_purification_terms": int(spin_terms),
    }
    return dense_mpo, info


def hubbard_2d_local_hamiltonian(
    lx: int,
    ly: int,
    *,
    hopping: float,
    hubbard_u: float,
    mu: float = 0.0,
    ordering: str = "snake",
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> tuple[Hamiltonian, dict]:
    """Build the spinful Hubbard model as analytical Jordan--Wigner strings.

    The returned ``bonds`` are the physical hopping edges in Hamiltonian site
    order.  They are the appropriate graph ties; intermediate ``JW`` factors
    are algebraic string supports, not additional physical correlations.
    """
    nsites = int(lx) * int(ly)
    bonds, order = ordered_lattice_bonds(
        lx,
        ly,
        ordering=ordering,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
    )
    base = SpinHalfFermionSite()
    operators = dict(base.operators)
    operators.update(
        {
            "CduJW": operators["Cdu"] @ operators["JW"],
            "CddJW": operators["Cdd"] @ operators["JW"],
            "JWCu": operators["JW"] @ operators["Cu"],
            "JWCd": operators["JW"] @ operators["Cd"],
        }
    )
    site = Site(
        labels=base.labels,
        operators=operators,
        charges=base.charges,
        charge_labels=base.charge_labels,
        parities=base.parities,
        statistics=base.statistics,
        name=base.name,
    )
    hamiltonian = Hamiltonian((site,) * nsites)
    for orbital in range(nsites):
        if hubbard_u:
            hamiltonian.add_product(float(hubbard_u), (orbital, "NuNd"))
        if mu:
            hamiltonian.add_product(-float(mu), (orbital, "N"))

    for endpoint_a, endpoint_b in bonds:
        left, right = sorted((int(endpoint_a), int(endpoint_b)))
        string = tuple((middle, "JW") for middle in range(left + 1, right))
        hamiltonian.add_product(
            -float(hopping),
            (left, "CduJW"),
            *string,
            (right, "Cu"),
        )
        hamiltonian.add_product(
            -float(hopping),
            (left, "JWCu"),
            *string,
            (right, "Cdu"),
        )
        hamiltonian.add_product(
            -float(hopping),
            (left, "CddJW"),
            *string,
            (right, "Cd"),
        )
        hamiltonian.add_product(
            -float(hopping),
            (left, "JWCd"),
            *string,
            (right, "Cdd"),
        )
    return hamiltonian, {
        "order": [int(value) for value in order],
        "bonds": [tuple(sorted((int(left), int(right)))) for left, right in bonds],
    }


def site_qn_maps(nsites: int):
    sectors = [AbelianSector(("charge", "sz"), qn) for qn in LOCAL_QNS]
    return [{state: sectors[state] for state in range(4)} for _ in range(int(nsites))]


def rung_site_qn_maps(lx: int, ly: int):
    if int(ly) < 2:
        raise ValueError("rung_site_qn_maps requires ly >= 2.")
    base = site_qn_maps(1)[0]
    local = {}
    for state in range(4 ** int(ly)):
        digits = np.unravel_index(int(state), (4,) * int(ly))
        qn = AbelianSector(("charge", "sz"), (0, 0))
        for digit in digits:
            qn = qn + base[int(digit)]
        local[int(state)] = qn
    return [dict(local) for _ in range(int(lx))]


def _sector_counts_by_length(nsites: int, qn_maps):
    """Number of primitive configurations in each cumulative sector."""
    counts = [{AbelianSector(("charge", "sz"), (0, 0)): 1}]
    for site in range(int(nsites)):
        local_qns = [qn_maps[site][state] for state in sorted(qn_maps[site])]
        previous = counts[-1]
        current = {}
        for q_left, count in previous.items():
            for q_phys in local_qns:
                q_right = q_left + q_phys
                current[q_right] = current.get(q_right, 0) + int(count)
        counts.append(current)
    return counts


def random_fixed_sector_abelian_mps(
    nsites: int,
    nup: int,
    ndown: int,
    *,
    max_bond_dim: int,
    qn_maps,
    native_site_storage: bool = False,
    seed: int = 7,
    scale: float = 0.2,
):
    """Random Abelian MPS with reachable bond sectors in a fixed spin sector."""
    nsites = int(nsites)
    max_bond_dim = max(1, int(max_bond_dim))
    rng = np.random.default_rng(int(seed))
    target = AbelianSector(("charge", "sz"), (int(nup) + int(ndown), int(nup) - int(ndown)))
    zero = AbelianSector(("charge", "sz"), (0, 0))
    prefix_counts = _sector_counts_by_length(nsites, qn_maps)
    suffix_counts = _sector_counts_by_length(nsites, list(reversed(qn_maps)))

    bond_qns = [[zero]]
    for bond in range(1, nsites):
        left = prefix_counts[bond]
        right = suffix_counts[nsites - bond]
        weighted = []
        for qn, left_count in left.items():
            need = target - qn
            right_count = right.get(need, 0)
            if right_count:
                degeneracy = min(int(left_count), int(right_count))
                weight = int(left_count) * int(right_count)
                weighted.append((weight, tuple(int(x) for x in qn), qn, degeneracy))
        weighted.sort(key=lambda item: (-item[0], item[1]))
        labels = []
        for _weight, _key, qn, degeneracy in weighted:
            for _ in range(int(degeneracy)):
                if len(labels) >= max_bond_dim:
                    break
                labels.append(qn)
            if len(labels) >= max_bond_dim:
                break
        if not labels:
            raise ValueError(f"no fixed-sector support found at bond {bond}.")
        bond_qns.append(labels)
    bond_qns.append([target])

    phys_unique = sorted(set(qn_maps[0].values()))
    tensors = []
    for site in range(nsites):
        left_labels = bond_qns[site]
        right_labels = bond_qns[site + 1]
        states_by_qn = {}
        for state, qn in sorted(qn_maps[site].items()):
            states_by_qn.setdefault(qn, []).append(int(state))
        data = {}
        for q_left in sorted(set(left_labels)):
            left_deg = sum(q == q_left for q in left_labels)
            for q_phys, phys_states in sorted(states_by_qn.items()):
                q_right = q_left + q_phys
                right_deg = sum(q == q_right for q in right_labels)
                if right_deg == 0:
                    continue
                block = scale * (
                    rng.standard_normal((left_deg, right_deg, len(phys_states)))
                    + 1j * rng.standard_normal((left_deg, right_deg, len(phys_states)))
                )
                data[(q_left, q_right, q_phys)] = block.astype(complex)
        tensors.append(
            make_abelian_site_tensor(
                data,
                [left_labels, right_labels, phys_unique],
                [-1, 1, 1],
                native_site_storage=bool(native_site_storage),
                copy=False,
            )
        )
    return tensors


def fixed_sector_product_mps(nsites: int, nup: int, ndown: int) -> list[np.ndarray]:
    """Simple product state in the requested spin sector."""
    nsites = int(nsites)
    remaining_up = int(nup)
    remaining_down = int(ndown)
    factors = []
    for site in range(nsites):
        remaining_sites = nsites - site
        force_double = remaining_up + remaining_down > remaining_sites
        if force_double and remaining_up > 0 and remaining_down > 0:
            state = 3
            remaining_up -= 1
            remaining_down -= 1
        elif (site % 2 == 0 and remaining_up > 0) or remaining_down == 0:
            state = 1 if remaining_up > 0 else 0
            remaining_up -= int(state == 1)
        elif remaining_down > 0:
            state = 2
            remaining_down -= 1
        else:
            state = 0
        tensor = np.zeros((1, 4, 1), dtype=complex)
        tensor[0, state, 0] = 1.0
        factors.append(tensor)
    if remaining_up or remaining_down:
        raise ValueError("could not construct a product MPS in the requested sector.")
    return factors


def fixed_sector_product_mps_from_qn_maps(qn_maps, nup: int, ndown: int) -> list[np.ndarray]:
    """Greedy product state for arbitrary local maps, including rung supersites."""
    nsites = len(qn_maps)
    target = AbelianSector(("charge", "sz"), (int(nup) + int(ndown), int(nup) - int(ndown)))
    zero = AbelianSector(("charge", "sz"), (0, 0))
    suffix_counts = _sector_counts_by_length(nsites, list(reversed(qn_maps)))
    factors = []
    current = zero
    for site, qn_map in enumerate(qn_maps):
        remaining = suffix_counts[nsites - site - 1]
        chosen = None
        target_charge_left = int(target[0] - current[0])
        ideal_charge = target_charge_left / max(1, nsites - site)
        candidates = []
        for state, qn in sorted(qn_map.items()):
            need = target - (current + qn)
            if remaining.get(need, 0):
                candidates.append((abs(float(qn[0]) - ideal_charge), abs(float(qn[1])), int(state), qn))
        if candidates:
            _score_charge, _score_spin, chosen, chosen_qn = min(candidates)
        if chosen is None:
            raise ValueError("could not construct a product MPS in the requested sector.")
        dim = max(qn_map) + 1
        tensor = np.zeros((1, dim, 1), dtype=complex)
        tensor[0, int(chosen), 0] = 1.0
        factors.append(tensor)
        current = current + chosen_qn
    if current != target:
        raise ValueError("product MPS ended in the wrong symmetry sector.")
    return factors


def _fixed_particle_basis(nsites: int, nelec: int):
    basis = []
    for occ in combinations(range(int(nsites)), int(nelec)):
        bits = 0
        for site in occ:
            bits |= 1 << site
        basis.append(bits)
    return basis


def _flat_index_from_spin_bits(up_bits: int, down_bits: int, nsites: int) -> int:
    local = []
    for site in range(int(nsites)):
        up = (int(up_bits) >> site) & 1
        down = (int(down_bits) >> site) & 1
        local.append(3 if up and down else 1 if up else 2 if down else 0)
    return int(np.ravel_multi_index(tuple(local), (4,) * int(nsites)))


def fixed_sector_indices(nsites: int, nup: int, ndown: int) -> list[int]:
    return [
        _flat_index_from_spin_bits(up_bits, down_bits, nsites)
        for up_bits in _fixed_particle_basis(nsites, nup)
        for down_bits in _fixed_particle_basis(nsites, ndown)
    ]


def fixed_sector_indices_from_qn_maps(qn_maps, target_qn) -> list[int]:
    dims = [max(qn_map) + 1 for qn_map in qn_maps]
    zero = AbelianSector(("charge", "sz"), (0, 0))
    keep = []
    for multi in np.ndindex(*dims):
        qn = zero
        for site, state in enumerate(multi):
            qn = qn + qn_maps[site][int(state)]
        if qn == target_qn:
            keep.append(int(np.ravel_multi_index(multi, dims)))
    return keep


def projected_mpo_spectrum(
    dense_mpo: list[np.ndarray],
    *,
    nup: int,
    ndown: int,
    nroots: int = 4,
    qn_maps=None,
) -> np.ndarray:
    """Project a small dense MPO to fixed spin sector and return low eigenvalues."""
    nsites = len(dense_mpo)
    dims = tuple(int(factor.shape[2]) for factor in dense_mpo)
    full = _mpo_to_dense_operator(type("MPOList", (), {"factors": dense_mpo, "dims": dims})())
    if qn_maps is None and all(dim == 4 for dim in dims):
        keep = fixed_sector_indices(nsites, nup, ndown)
    else:
        if qn_maps is None:
            raise ValueError("qn_maps are required to project a non-spatial-site MPO.")
        target = AbelianSector(("charge", "sz"), (int(nup) + int(ndown), int(nup) - int(ndown)))
        keep = fixed_sector_indices_from_qn_maps(qn_maps, target)
    projected = full[np.ix_(keep, keep)]
    projected = 0.5 * (projected + projected.T.conj())
    evals = np.linalg.eigvalsh(projected)
    return np.asarray(evals[: max(1, int(nroots))], dtype=float)


def ed_ground_energy(
    lx: int,
    ly: int,
    *,
    nup: int,
    ndown: int,
    hopping: float,
    hubbard_u: float,
    mu: float,
    periodic_x: bool,
    periodic_y: bool,
    nroots: int = 4,
) -> tuple[np.ndarray, dict]:
    hamiltonian, info = hubbard_2d_hamiltonian(
        lx,
        ly,
        nup=nup,
        ndown=ndown,
        t=hopping,
        u=hubbard_u,
        mu=mu,
        periodic_x=periodic_x,
        periodic_y=periodic_y,
    )
    nroots = min(max(1, int(nroots)), int(hamiltonian.shape[0]))
    if nroots >= hamiltonian.shape[0]:
        evals = np.linalg.eigvalsh(hamiltonian.toarray())[:nroots]
    else:
        evals = eigsh(hamiltonian, k=nroots, which="SA", return_eigenvectors=False)
        evals.sort()
    return np.asarray(evals, dtype=float), info


def ed_phase_gaps(
    lx: int,
    ly: int,
    *,
    nup: int,
    ndown: int,
    hopping: float,
    hubbard_u: float,
    mu: float,
    periodic_x: bool,
    periodic_y: bool,
    ground_energy: float | None = None,
) -> dict[str, float | dict[str, int]]:
    """Return finite-cluster charge and spin-sector gaps from exact diagonalization."""
    nsites = int(lx) * int(ly)

    def sector_energy(up: int, down: int) -> float:
        values, _info = ed_ground_energy(
            lx,
            ly,
            nup=up,
            ndown=down,
            hopping=hopping,
            hubbard_u=hubbard_u,
            mu=mu,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            nroots=1,
        )
        return float(values[0])

    e0 = sector_energy(nup, ndown) if ground_energy is None else float(ground_energy)
    addition_sectors = [
        (up, down)
        for up, down in ((nup + 1, ndown), (nup, ndown + 1))
        if up <= nsites and down <= nsites
    ]
    removal_sectors = [
        (up, down)
        for up, down in ((nup - 1, ndown), (nup, ndown - 1))
        if up >= 0 and down >= 0
    ]
    spin_sectors = [
        (up, down)
        for up, down in ((nup + 1, ndown - 1), (nup - 1, ndown + 1))
        if 0 <= up <= nsites and 0 <= down <= nsites
    ]
    if not addition_sectors or not removal_sectors:
        raise ValueError("the charge gap requires both N+1 and N-1 sectors.")
    if not spin_sectors:
        raise ValueError("the spin gap requires a neighboring fixed-particle spin sector.")

    addition = min((sector_energy(*sector), sector) for sector in addition_sectors)
    removal = min((sector_energy(*sector), sector) for sector in removal_sectors)
    spin = min((sector_energy(*sector), sector) for sector in spin_sectors)
    return {
        "charge_gap": float(addition[0] + removal[0] - 2.0 * e0),
        "spin_gap": float(spin[0] - e0),
        "addition_energy": float(addition[0]),
        "removal_energy": float(removal[0]),
        "spin_sector_energy": float(spin[0]),
        "addition_sector": {"nup": int(addition[1][0]), "ndown": int(addition[1][1])},
        "removal_sector": {"nup": int(removal[1][0]), "ndown": int(removal[1][1])},
        "spin_sector": {"nup": int(spin[1][0]), "ndown": int(spin[1][1])},
    }


def _standard_dense_factors(factors_or_mps) -> list[np.ndarray]:
    if isinstance(factors_or_mps, MPS):
        return [np.asarray(tensor) for tensor in factors_or_mps.to_order(["lv", "p", "rv"]).factors]
    if hasattr(factors_or_mps, "factors"):
        return _standard_dense_factors(MPS(factors_or_mps.factors, labels=getattr(factors_or_mps, "labels", ["lv", "p", "rv"])))
    return [np.asarray(tensor) for tensor in factors_or_mps]


def dense_mps_product_expectation(factors_or_mps, operators: list[np.ndarray]) -> complex:
    factors = _standard_dense_factors(factors_or_mps)
    env = np.ones((1, 1), dtype=complex)
    norm_env = np.ones((1, 1), dtype=complex)
    for tensor, operator in zip(factors, operators):
        tensor = np.asarray(tensor, dtype=complex)
        operator = np.asarray(operator, dtype=complex)
        identity = np.eye(tensor.shape[1], dtype=complex)
        env = np.einsum("ab,atr,ts,bsu->ru", env, tensor.conj(), operator, tensor, optimize=True)
        norm_env = np.einsum("ab,atr,ts,bsu->ru", norm_env, tensor.conj(), identity, tensor, optimize=True)
    norm = norm_env.reshape(-1)[0]
    if abs(norm) < 1.0e-14:
        raise ValueError("MPS norm is numerically zero.")
    return env.reshape(-1)[0] / norm


def lattice_distance_average(
    correlation: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    lx: int,
    ly: int,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> list[float]:
    """Average a site correlation by Manhattan distance on the lattice."""
    correlation = np.asarray(correlation)
    groups: dict[int, list[complex]] = {}
    for i, (xi, yi) in enumerate(coords):
        for j, (xj, yj) in enumerate(coords):
            dx = abs(int(xi) - int(xj))
            dy = abs(int(yi) - int(yj))
            if periodic_x:
                dx = min(dx, int(lx) - dx)
            if periodic_y:
                dy = min(dy, int(ly) - dy)
            groups.setdefault(dx + dy, []).append(correlation[i, j])
    return [float(np.real(np.mean(groups[distance]))) for distance in sorted(groups)]


def _structure_factor_at(
    correlation: np.ndarray,
    coords: list[tuple[int, int]],
    qx: float,
    qy: float,
) -> float:
    """Return the conventional structure factor ``sum_ij C_ij/N``."""
    phase = np.asarray(
        [np.exp(-1.0j * (float(qx) * x + float(qy) * y)) for x, y in coords],
        dtype=complex,
    )
    value = np.vdot(phase, np.asarray(correlation) @ phase) / len(coords)
    return float(np.real_if_close(value).real)


def _structure_factor_grid(
    correlation: np.ndarray,
    coords: list[tuple[int, int]],
    *,
    lx: int,
    ly: int,
) -> tuple[list[dict[str, float]], dict[str, float]]:
    """Evaluate a correlation matrix on the rectangular reciprocal grid."""
    values = []
    for my in range(int(ly)):
        ky = my if my <= int(ly) // 2 else my - int(ly)
        qy = 2.0 * np.pi * ky / int(ly)
        for mx in range(int(lx)):
            kx = mx if mx <= int(lx) // 2 else mx - int(lx)
            qx = 2.0 * np.pi * kx / int(lx)
            values.append(
                {
                    "qx_over_pi": float(qx / np.pi),
                    "qy_over_pi": float(qy / np.pi),
                    "value": _structure_factor_at(correlation, coords, qx, qy),
                }
            )
    peak = max(values, key=lambda row: row["value"])
    return values, dict(peak)


def _expanded_abelian_leg_labels(tensor, leg: int) -> list[tuple[int, ...]]:
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


def letta_layout_from_abelian_mps(sym_mps, qn_maps, target_qn) -> Layout:
    factors = sym_mps.factors
    local_qns = [[tuple(int(x) for x in site_map[index]) for index in sorted(site_map)] for site_map in qn_maps]
    bond_qns = [_expanded_abelian_leg_labels(factors[0], 0)]
    for site in range(len(factors) - 2):
        bond_qns.append(_expanded_abelian_leg_labels(factors[site], 1))
    return Layout(local_qns=local_qns, bond_qns=bond_qns, target=tuple(int(x) for x in target_qn))


def _qn_tuple(qn) -> tuple[int, ...]:
    return tuple(int(x) for x in qn)


def _add_qns(*qns) -> tuple[int, ...]:
    if not qns:
        return ()
    total = [0] * len(qns[0])
    for qn in qns:
        for i, value in enumerate(qn):
            total[i] += int(value)
    return tuple(total)


def nnn_letta_sector_masks_from_abelian_mps(sym_mps, qn_maps, target_qn, dims) -> list[np.ndarray]:
    """Build fixed-sector masks for the NNN-LETTA MPS embedding convention."""
    nsites = len(qn_maps)
    if nsites < NNNLETTA.tensor_width:
        raise ValueError("NNN-LETTA sector masks need at least three sites.")
    factors = sym_mps.factors
    local_qns = [[_qn_tuple(site_map[index]) for index in sorted(site_map)] for site_map in qn_maps]
    bond_qns = [_expanded_abelian_leg_labels(factors[0], 0)]
    for site in range(nsites - NNNLETTA.tensor_width):
        bond_qns.append(_expanded_abelian_leg_labels(factors[site], 1))
    target = _qn_tuple(target_qn)
    bond_qns.append([target])

    masks = []
    nlocal = nsites - NNNLETTA.tensor_width + 1
    for tensor_index in range(nlocal):
        shape = (
            len(bond_qns[tensor_index]),
            int(dims[tensor_index]),
            int(dims[tensor_index + 1]),
            int(dims[tensor_index + 2]),
            len(bond_qns[tensor_index + 1]),
        )
        mask = np.zeros(shape, dtype=bool)
        for left_index, q_left in enumerate(bond_qns[tensor_index]):
            for right_index, q_right in enumerate(bond_qns[tensor_index + 1]):
                if tensor_index < nlocal - 1:
                    for s0, q0 in enumerate(local_qns[tensor_index]):
                        if _add_qns(q_left, q0) == q_right:
                            mask[left_index, s0, :, :, right_index] = True
                else:
                    for s0, q0 in enumerate(local_qns[tensor_index]):
                        for s1, q1 in enumerate(local_qns[tensor_index + 1]):
                            for s2, q2 in enumerate(local_qns[tensor_index + 2]):
                                if _add_qns(q_left, q0, q1, q2) == q_right:
                                    mask[left_index, s0, s1, s2, right_index] = True
        masks.append(mask)
    return masks


def hubbard_site_operators() -> dict[str, np.ndarray]:
    ops = {name: np.asarray(op, dtype=complex) for name, op in SpinHalfFermionOperators().items()}
    ops["PairAnn"] = ops["Cd"] @ ops["Cu"]
    ops["PairCre"] = ops["PairAnn"].T.conj()
    return ops


def _coords_for_order(lx: int, order: list[int]) -> list[tuple[int, int]]:
    return [(int(site) % int(lx), int(site) // int(lx)) for site in order]


def _rung_embeddings(mpo_info: dict) -> list[tuple[int, int | None]]:
    grouping = mpo_info.get("site_grouping", "site")
    if grouping == "site":
        return [(idx, None) for idx, _site in enumerate(mpo_info["order"])]
    if grouping != "rung":
        raise ValueError(f"unsupported site grouping {grouping!r}.")
    embeddings = {}
    for block_idx, block in enumerate(mpo_info["fused_blocks"]):
        for leg, lattice_site in enumerate(block):
            embeddings[int(lattice_site)] = (int(block_idx), int(leg))
    return [embeddings[int(site)] for site in mpo_info["order"]]


def _embed_rung_operator(operator: np.ndarray, leg: int | None, local_dim: int | None = None) -> np.ndarray:
    operator = np.asarray(operator, dtype=complex)
    if leg is None:
        return operator
    if local_dim is None:
        local_dim = 16
    width = 0
    remaining = int(local_dim)
    while remaining > 1 and remaining % 4 == 0:
        width += 1
        remaining //= 4
    if remaining != 1 or width < 1:
        raise ValueError("rung local_dim must be a power of the spatial Hubbard local dimension 4.")
    leg = int(leg)
    if leg < 0 or leg >= width:
        raise ValueError(f"rung operator leg {leg} is outside a width-{width} supersite.")
    factors = [np.eye(4, dtype=complex) for _ in range(width)]
    factors[leg] = operator
    embedded = factors[0]
    for factor in factors[1:]:
        embedded = np.kron(embedded, factor)
    return embedded


def _expect_product(factors_or_letta, operators: list[np.ndarray], *, is_letta: bool = False):
    if is_letta:
        return factors_or_letta.expectation_product_operator(operators)
    return dense_mps_product_expectation(factors_or_letta, operators)


def _site_expectations(factors_or_letta, base_operator, embeddings, dims, *, is_letta: bool = False):
    identities = [np.eye(dim, dtype=complex) for dim in dims]
    values = []
    for mps_site, leg in embeddings:
        operators = list(identities)
        operators[mps_site] = _embed_rung_operator(base_operator, leg, dims[mps_site])
        values.append(_expect_product(factors_or_letta, operators, is_letta=is_letta))
    return np.asarray(values, dtype=complex)


def _site_correlation(factors_or_letta, op_a, op_b, embeddings, dims, *, is_letta: bool = False):
    identities = [np.eye(dim, dtype=complex) for dim in dims]
    nsites = len(embeddings)
    corr = np.empty((nsites, nsites), dtype=complex)
    for i, (site_i, leg_i) in enumerate(embeddings):
        emb_a = _embed_rung_operator(op_a, leg_i, dims[site_i])
        for j, (site_j, leg_j) in enumerate(embeddings):
            emb_b = _embed_rung_operator(op_b, leg_j, dims[site_j])
            operators = list(identities)
            if site_i == site_j:
                operators[site_i] = emb_a @ emb_b
            else:
                operators[site_i] = emb_a
                operators[site_j] = emb_b
            corr[i, j] = _expect_product(factors_or_letta, operators, is_letta=is_letta)
    return corr


def _fermion_sequence_product_operators(
    sequence: list[tuple[int, np.ndarray]],
    *,
    nsites: int,
    parity: np.ndarray,
    embeddings: list[tuple[int, int | None]],
    dims: list[int],
) -> list[np.ndarray]:
    """Map an ordered fermion-operator sequence to Jordan-Wigner site factors."""
    primitive = [np.eye(4, dtype=complex) for _ in range(int(nsites))]
    parity = np.asarray(parity, dtype=complex)
    for site, local_operator in sequence:
        site = int(site)
        if not 0 <= site < int(nsites):
            raise IndexError(f"fermion operator site {site} is outside a {nsites}-site state.")
        for left in range(site):
            primitive[left] = primitive[left] @ parity
        primitive[site] = primitive[site] @ np.asarray(local_operator, dtype=complex)

    grouped = [np.eye(dim, dtype=complex) for dim in dims]
    for primitive_site, (mps_site, leg) in enumerate(embeddings):
        grouped[mps_site] = grouped[mps_site] @ _embed_rung_operator(
            primitive[primitive_site],
            leg,
            dims[mps_site],
        )
    return grouped


def _singlet_pair_terms(
    bond: tuple[int, int],
    operators: dict[str, np.ndarray],
    *,
    creation: bool,
) -> list[tuple[float, list[tuple[int, np.ndarray]]]]:
    i, j = sorted((int(bond[0]), int(bond[1])))
    scale = 1.0 / np.sqrt(2.0)
    annihilation = [
        (scale, [(i, operators["Cu"]), (j, operators["Cd"])]),
        (-scale, [(i, operators["Cd"]), (j, operators["Cu"])]),
    ]
    if not creation:
        return annihilation
    return [
        (
            coefficient,
            [(site, operator.T.conj()) for site, operator in reversed(sequence)],
        )
        for coefficient, sequence in annihilation
    ]


def _bond_singlet_pair_correlation(
    factors_or_letta,
    operators: dict[str, np.ndarray],
    bonds: list[tuple[int, int]],
    embeddings: list[tuple[int, int | None]],
    dims: list[int],
    *,
    is_letta: bool = False,
) -> np.ndarray:
    """Return ``<Delta_b^dagger Delta_b'>`` for nearest-neighbor singlet pairs."""
    creation_terms = [
        _singlet_pair_terms(bond, operators, creation=True) for bond in bonds
    ]
    annihilation_terms = [
        _singlet_pair_terms(bond, operators, creation=False) for bond in bonds
    ]
    corr = np.empty((len(bonds), len(bonds)), dtype=complex)
    for left, left_terms in enumerate(creation_terms):
        for right, right_terms in enumerate(annihilation_terms):
            value = 0.0j
            for left_coefficient, left_sequence in left_terms:
                for right_coefficient, right_sequence in right_terms:
                    product = _fermion_sequence_product_operators(
                        left_sequence + right_sequence,
                        nsites=len(embeddings),
                        parity=operators["JW"],
                        embeddings=embeddings,
                        dims=dims,
                    )
                    value += left_coefficient * right_coefficient * _expect_product(
                        factors_or_letta,
                        product,
                        is_letta=is_letta,
                    )
            corr[left, right] = value
    return corr


def _metrics_from_state(
    factors_or_letta,
    operators: dict[str, np.ndarray],
    *,
    lx: int,
    mpo_info: dict,
    is_letta: bool = False,
) -> dict:
    ntot = operators["Ntot"]
    sx = operators["Sx"]
    sy = operators["Sy"]
    sz = operators["Sz"]
    doublon = operators["NuNd"]
    pair_cre = operators["PairCre"]
    pair_ann = operators["PairAnn"]
    order = [int(site) for site in mpo_info["order"]]
    coords = _coords_for_order(lx, order)
    stagger = np.asarray([(-1) ** (x + y) for x, y in coords], dtype=float)
    embeddings = _rung_embeddings(mpo_info)

    if is_letta:
        state = factors_or_letta
        dims = list(state.dims)
        density = _site_expectations(state, ntot, embeddings, dims, is_letta=True)
        spin_z = _site_expectations(state, sz, embeddings, dims, is_letta=True)
        doublons = _site_expectations(state, doublon, embeddings, dims, is_letta=True)
        density_corr = _site_correlation(state, ntot, ntot, embeddings, dims, is_letta=True)
        spin_x_corr = _site_correlation(state, sx, sx, embeddings, dims, is_letta=True)
        spin_y_corr = _site_correlation(state, sy, sy, embeddings, dims, is_letta=True)
        spin_z_corr = _site_correlation(state, sz, sz, embeddings, dims, is_letta=True)
        pair_corr = _site_correlation(state, pair_cre, pair_ann, embeddings, dims, is_letta=True)
    else:
        factors = _standard_dense_factors(factors_or_letta)
        dims = [factor.shape[1] for factor in factors]
        density = _site_expectations(factors, ntot, embeddings, dims)
        spin_z = _site_expectations(factors, sz, embeddings, dims)
        doublons = _site_expectations(factors, doublon, embeddings, dims)
        density_corr = _site_correlation(factors, ntot, ntot, embeddings, dims)
        spin_x_corr = _site_correlation(factors, sx, sx, embeddings, dims)
        spin_y_corr = _site_correlation(factors, sy, sy, embeddings, dims)
        spin_z_corr = _site_correlation(factors, sz, sz, embeddings, dims)
        pair_corr = _site_correlation(factors, pair_cre, pair_ann, embeddings, dims)

    nsites = int(len(density))
    # Subtract measured one-point functions. This matters at open boundaries
    # and away from half filling; subtracting the requested filling is not a
    # connected charge correlation when the density is spatially nonuniform.
    charge_corr = density_corr - np.outer(density, density)
    spin_z_connected = spin_z_corr - np.outer(spin_z, spin_z)
    spin_dot_corr = spin_x_corr + spin_y_corr + spin_z_connected
    staggered_spin_density = float(abs(np.sum(stagger * spin_z.real) / nsites))
    charge_grid, charge_peak = _structure_factor_grid(
        charge_corr,
        coords,
        lx=int(lx),
        ly=int(mpo_info["spatial_sites"]) // int(lx),
    )
    spin_grid, spin_peak = _structure_factor_grid(
        spin_dot_corr,
        coords,
        lx=int(lx),
        ly=int(mpo_info["spatial_sites"]) // int(lx),
    )
    pair_grid, pair_peak = _structure_factor_grid(
        pair_corr,
        coords,
        lx=int(lx),
        ly=int(mpo_info["spatial_sites"]) // int(lx),
    )
    charge_pi_pi = _structure_factor_at(charge_corr, coords, np.pi, np.pi)
    spin_zz_pi_pi = _structure_factor_at(spin_z_connected, coords, np.pi, np.pi)
    spin_dot_pi_pi = _structure_factor_at(spin_dot_corr, coords, np.pi, np.pi)
    pair_q0 = _structure_factor_at(pair_corr, coords, 0.0, 0.0)
    bonds = [(int(i), int(j)) for i, j in mpo_info["bonds"]]
    nn_spin = float(np.real(np.mean([spin_dot_corr[i, j] for i, j in bonds])))
    bond_pair_corr = _bond_singlet_pair_correlation(
        factors_or_letta,
        operators,
        bonds,
        embeddings,
        dims,
        is_letta=is_letta,
    )
    bond_orientations = [
        "x" if coords[i][1] == coords[j][1] else "y" for i, j in bonds
    ]
    d_wave_weights = np.asarray(
        [1.0 if orientation == "x" else -1.0 for orientation in bond_orientations]
    )
    extended_s_weights = np.ones(len(bonds), dtype=float)
    bond_pair_offsite = np.asarray(bond_pair_corr).copy()
    np.fill_diagonal(bond_pair_offsite, 0.0)

    def pair_structure(weights, correlation) -> float:
        value = np.vdot(weights, np.asarray(correlation) @ weights) / len(bonds)
        return float(np.real_if_close(value).real)

    staggered_spin_corr = np.asarray(spin_dot_corr) * np.outer(stagger, stagger)
    distance_kwargs = {
        "lx": int(lx),
        "ly": int(mpo_info["spatial_sites"]) // int(lx),
        "periodic_x": bool(mpo_info.get("periodic_x", False)),
        "periodic_y": bool(mpo_info.get("periodic_y", False)),
    }
    return {
        "density": [float(np.real(value)) for value in density],
        "spin_z": [float(np.real(value)) for value in spin_z],
        "mean_density": float(np.real(np.mean(density))),
        "avg_doublon": float(np.real(np.mean(doublons))),
        "avg_local_moment": float(np.real(np.mean(density - 2.0 * doublons))),
        "avg_charge_fluctuation": float(np.real(np.trace(charge_corr) / nsites)),
        "staggered_charge_density": float(
            abs(np.sum(stagger * (density.real - np.mean(density.real))) / nsites)
        ),
        # This is reported only as a symmetry-breaking one-point diagnostic.
        # It is normally zero in a fixed-Sz finite state and is not AF order.
        "staggered_spin_density": staggered_spin_density,
        "charge_structure_factor_pi_pi": charge_pi_pi,
        "charge_order_parameter_sq_pi_pi": charge_pi_pi / nsites,
        "spin_structure_factor_pi_pi_zz": spin_zz_pi_pi,
        "spin_structure_factor_pi_pi_dot": spin_dot_pi_pi,
        "spin_order_parameter_sq_pi_pi": spin_dot_pi_pi / nsites,
        "nearest_neighbor_spin_dot": nn_spin,
        "charge_structure_grid": charge_grid,
        "charge_structure_peak": charge_peak,
        "spin_structure_grid": spin_grid,
        "spin_structure_peak": spin_peak,
        "onsite_pair_structure_grid": pair_grid,
        "onsite_pair_structure_peak": pair_peak,
        "onsite_pair_structure_q0": pair_q0,
        "bond_singlet_pairs": [
            {
                "sites": [int(i), int(j)],
                "orientation": orientation,
            }
            for (i, j), orientation in zip(bonds, bond_orientations)
        ],
        "d_wave_pair_structure_per_bond": pair_structure(
            d_wave_weights,
            bond_pair_corr,
        ),
        "d_wave_pair_offsite_per_bond": pair_structure(
            d_wave_weights,
            bond_pair_offsite,
        ),
        "extended_s_pair_structure_per_bond": pair_structure(
            extended_s_weights,
            bond_pair_corr,
        ),
        "extended_s_pair_offsite_per_bond": pair_structure(
            extended_s_weights,
            bond_pair_offsite,
        ),
        "bond_pair_hermiticity_error": float(
            np.max(np.abs(bond_pair_corr - bond_pair_corr.T.conj()))
        ),
        "staggered_spin_by_manhattan_distance": lattice_distance_average(
            staggered_spin_corr,
            coords,
            **distance_kwargs,
        ),
        "connected_charge_by_manhattan_distance": lattice_distance_average(
            charge_corr,
            coords,
            **distance_kwargs,
        ),
        "onsite_pair_by_manhattan_distance": lattice_distance_average(
            pair_corr,
            coords,
            **distance_kwargs,
        ),
    }


def run_case(args) -> dict:
    spatial_sites = int(args.lx) * int(args.ly)
    dense_mpo, mpo_info = hubbard_2d_dense_mpo(
        args.lx,
        args.ly,
        hopping=args.hopping,
        hubbard_u=args.hubbard_u,
        mu=args.mu,
        ordering=args.ordering,
        periodic_x=args.periodic_x,
        periodic_y=args.periodic_y,
        site_grouping=args.site_grouping,
    )
    if args.site_grouping == "rung":
        qn_maps = rung_site_qn_maps(args.lx, args.ly)
    else:
        qn_maps = site_qn_maps(spatial_sites)
    nsites = len(qn_maps)
    local_dims = [int(factor.shape[2]) for factor in dense_mpo]
    sym_mgr = SymmetryManager(["charge", "sz"])
    target_qn = sym_mgr.get_target_qn(int(args.nup) + int(args.ndown), int(args.nup) - int(args.ndown))
    matvec_options = dmrg_matvec_options(args.dmrg_policy)
    symmetric_mpo = dense_to_symmetric_mpo(
        dense_mpo,
        qn_maps,
        native_site_storage=bool(matvec_options.get("native_site_storage", False)),
    )

    ed_evals = None
    ed_info = None
    phase_gaps = None
    if not args.skip_ed:
        ed_evals, ed_info = ed_ground_energy(
            args.lx,
            args.ly,
            nup=args.nup,
            ndown=args.ndown,
            hopping=args.hopping,
            hubbard_u=args.hubbard_u,
            mu=args.mu,
            periodic_x=args.periodic_x,
            periodic_y=args.periodic_y,
            nroots=args.ed_roots,
        )
        if args.ed_phase_gaps:
            phase_gaps = ed_phase_gaps(
                args.lx,
                args.ly,
                nup=args.nup,
                ndown=args.ndown,
                hopping=args.hopping,
                hubbard_u=args.hubbard_u,
                mu=args.mu,
                periodic_x=args.periodic_x,
                periodic_y=args.periodic_y,
                ground_energy=float(ed_evals[0]),
            )

    mpo_projected_evals = None
    if (
        args.check_mpo
        and nsites <= args.check_mpo_max_sites
        and int(np.prod(local_dims)) <= int(args.check_mpo_max_dim)
    ):
        mpo_projected_evals = projected_mpo_spectrum(
            dense_mpo,
            nup=args.nup,
            ndown=args.ndown,
            nroots=args.ed_roots,
            qn_maps=qn_maps,
        )

    dmrg = None
    dmrg_time = 0.0
    dmrg_converged = None
    dmrg_history = []
    dmrg_checkpoint_payload = None
    if args.letta_from_dmrg_checkpoint:
        dmrg_checkpoint_payload = DMRG.load_checkpoint(args.letta_from_dmrg_checkpoint)
        if "mps" not in dmrg_checkpoint_payload:
            raise ValueError(f"DMRG checkpoint {args.letta_from_dmrg_checkpoint!r} does not contain an MPS.")
        if len(dmrg_checkpoint_payload["mps"]) != nsites:
            raise ValueError("DMRG checkpoint length does not match this benchmark geometry.")
        sym_mps = MPS(dmrg_checkpoint_payload["mps"], labels=["lv", "rv", "p"])
        dmrg_history = list(dmrg_checkpoint_payload.get("sweep_history", []))
        dmrg_converged = None
    else:
        if args.init == "product":
            if args.site_grouping == "site":
                initial = fixed_sector_product_mps(nsites, args.nup, args.ndown)
            else:
                initial = fixed_sector_product_mps_from_qn_maps(qn_maps, args.nup, args.ndown)
        else:
            initial = random_fixed_sector_abelian_mps(
                nsites,
                args.nup,
                args.ndown,
                max_bond_dim=args.bond_dim,
                qn_maps=qn_maps,
                native_site_storage=bool(matvec_options.get("native_site_storage", False)),
                seed=args.seed,
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
            site_qn_maps=qn_maps,
            not_conv_err=False,
            verbose=int(args.verbose),
            sweep_tol=float(args.sweep_tol),
            davidson_tol=float(args.davidson_tol),
            davidson_max_iter=int(args.davidson_max_iter),
            noise=float(args.noise),
            noise_decay=float(args.noise_decay),
            performance=args.dmrg_policy,
            abelian_matvec_options=matvec_options,
            checkpoint_path=args.dmrg_checkpoint,
            resume_from=args.resume_dmrg,
            checkpoint_interval=int(args.dmrg_checkpoint_interval),
            recenter_final=bool(args.dmrg_recenter),
        )
        dmrg.run()
        dmrg_time = perf_counter() - start
        sym_mps = dmrg.state
        dmrg_history = list(dmrg.sweep_history)
        dmrg_converged = bool(dmrg.converged)

    dense_mps = symmetric_to_dense(sym_mps, site_qn_maps=qn_maps)
    dmrg_energy = _normalized_mps_mpo_expectation(dense_mps.factors, dense_mpo)

    letta = None
    letta_time = 0.0
    letta_energy = None
    if not args.dmrg_only:
        if args.letta_load:
            if args.letta_variant == "standard":
                letta = LETTA.load(args.letta_load)
            elif args.letta_variant == "nnn":
                letta = NNNLETTA.load(args.letta_load)
            else:
                raise ValueError(f"unsupported LETTA variant {args.letta_variant!r}.")
            if tuple(int(dim) for dim in letta.dims) != tuple(local_dims):
                raise ValueError("loaded LETTA state dimensions do not match this benchmark geometry.")
        else:
            if args.letta_variant == "standard":
                letta_layout = letta_layout_from_abelian_mps(sym_mps, qn_maps, target_qn)
                letta = LETTA.from_mps(
                    dense_mps,
                    hamiltonian=None,
                    dims=local_dims,
                    abelian_layout=letta_layout,
                    seed=int(args.seed),
                )
            elif args.letta_variant == "nnn":
                nnn_masks = nnn_letta_sector_masks_from_abelian_mps(
                    sym_mps,
                    qn_maps,
                    target_qn,
                    local_dims,
                )
                letta = NNNLETTA.from_mps(
                    dense_mps,
                    dims=local_dims,
                    local_masks=nnn_masks,
                    seed=int(args.seed),
                )
            else:
                raise ValueError(f"unsupported LETTA variant {args.letta_variant!r}.")
        if args.letta_expand_bond_dim is not None:
            if args.letta_variant != "standard":
                raise ValueError("--letta-expand-bond-dim is currently supported only for standard LETTA.")
            letta.expand_bond_dim(
                int(args.letta_expand_bond_dim),
                noise=float(args.letta_expand_noise),
                seed=int(args.seed),
            )
        if int(args.letta_sweeps) > 0:
            start = perf_counter()
            letta.run(
                dense_mpo,
                nsweeps=int(args.letta_sweeps),
                tol=float(args.letta_tol),
                gauge=None if args.letta_gauge == "none" else args.letta_gauge,
                verbose=int(args.verbose),
                local_solver=args.letta_solver,
                matrix_free_threshold=int(args.letta_matrix_free_threshold),
                matrix_free_tol=float(args.letta_matrix_free_tol),
                matrix_free_maxiter=int(args.letta_matrix_free_maxiter),
            )
            letta_time = perf_counter() - start
        letta_energy = float(letta.expectation_mpo(dense_mpo))
        if args.letta_save:
            letta.save(
                args.letta_save,
                metadata={
                    "source": "examples/mps/hubbard_2d_mps_vs_letta.py",
                    "letta_variant": args.letta_variant,
                    "lx": int(args.lx),
                    "ly": int(args.ly),
                    "site_grouping": args.site_grouping,
                    "nup": int(args.nup),
                    "ndown": int(args.ndown),
                    "bond_dim": int(args.bond_dim),
                    "letta_bond_dim": int(letta.bond_dim),
                    "energy": float(letta_energy),
                },
            )

    dmrg_metrics = None
    letta_metrics = None
    if not args.skip_metrics:
        operators = hubbard_site_operators()
        dmrg_metrics = _metrics_from_state(
            dense_mps,
            operators,
            lx=args.lx,
            mpo_info=mpo_info,
        )
        if letta is not None:
            letta_metrics = _metrics_from_state(
                letta,
                operators,
                lx=args.lx,
                mpo_info=mpo_info,
                is_letta=True,
            )

    return {
        "params": {
            "lx": int(args.lx),
            "ly": int(args.ly),
            "nsites": spatial_sites,
            "mps_sites": nsites,
            "local_dims": local_dims,
            "site_grouping": args.site_grouping,
            "letta_variant": args.letta_variant,
            "nup": int(args.nup),
            "ndown": int(args.ndown),
            "hopping": float(args.hopping),
            "hubbard_u": float(args.hubbard_u),
            "mu": float(args.mu),
            "ordering": args.ordering,
            "bond_dim": int(args.bond_dim),
            "letta_bond_dim": None if letta is None else int(letta.bond_dim),
            "sweeps": int(args.sweeps),
            "letta_sweeps": int(args.letta_sweeps),
            "dmrg_policy": args.dmrg_policy,
            "dmrg_recenter": bool(args.dmrg_recenter),
            "init": args.init,
        },
        "mpo": mpo_info,
        "ed": None if ed_evals is None else {"energies": [float(x) for x in ed_evals], "info": ed_info},
        "ed_phase_gaps": phase_gaps,
        "mpo_projected_evals": None if mpo_projected_evals is None else [float(x) for x in mpo_projected_evals],
        "dmrg_energy": float(dmrg_energy),
        "letta_energy": None if letta_energy is None else float(letta_energy),
        "dmrg_minus_ed": None if ed_evals is None else float(dmrg_energy - ed_evals[0]),
        "letta_minus_ed": None if ed_evals is None or letta_energy is None else float(letta_energy - ed_evals[0]),
        "letta_minus_dmrg": None if letta_energy is None else float(letta_energy - dmrg_energy),
        "dmrg_time_s": float(dmrg_time),
        "letta_time_s": float(letta_time),
        "dmrg_converged": None if dmrg_converged is None else bool(dmrg_converged),
        "letta_converged": None if letta is None else bool(letta.converged),
        "dmrg_history": [
            {
                "sweep": int(row.get("sweep", idx)),
                "energy": None if row.get("energy") is None else float(np.real(row["energy"])),
                "local_energy": None
                if row.get("local_energy") is None
                else float(np.real(row["local_energy"])),
                "post_truncation_energy": None
                if row.get("post_truncation_energy") is None
                else float(np.real(row["post_truncation_energy"])),
                "truncation": None
                if row.get("truncation") is None
                else float(np.real(row["truncation"])),
                "states_kept": None if row.get("states_kept") is None else int(row["states_kept"]),
                "direction": row.get("direction"),
            }
            for idx, row in enumerate(dmrg_history)
        ],
        "letta_history": [
            {
                "sweep": int(row["sweep"]),
                "energy": float(np.real(row["energy"])),
                "delta_energy": None if row["delta_energy"] is None else float(row["delta_energy"]),
            }
            for row in ([] if letta is None else letta.history)
        ],
        "dmrg": dmrg_metrics,
        "letta": letta_metrics,
    }


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lx", type=int, default=4)
    parser.add_argument("--ly", type=int, default=2)
    parser.add_argument("--nup", type=int, default=None)
    parser.add_argument("--ndown", type=int, default=None)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=4.0)
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("--ordering", choices=("row-major", "column-major", "snake"), default="snake")
    parser.add_argument("--site-grouping", choices=("site", "rung"), default="site")
    parser.add_argument("--periodic-x", action="store_true")
    parser.add_argument("--periodic-y", action="store_true")
    parser.add_argument("--bond-dim", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--letta-sweeps", type=int, default=3)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-7)
    parser.add_argument("--letta-tol", type=float, default=1.0e-7)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-7)
    parser.add_argument("--davidson-max-iter", type=int, default=48)
    parser.add_argument("--noise", type=float, default=1.0e-6)
    parser.add_argument("--noise-decay", type=float, default=0.2)
    parser.add_argument("--init", choices=("sector-random", "product"), default="sector-random")
    parser.add_argument("--dmrg-policy", default="packed-compiled-fast")
    parser.add_argument("--dmrg-only", action="store_true")
    parser.add_argument("--dmrg-checkpoint", default=None)
    parser.add_argument("--resume-dmrg", default=None)
    parser.add_argument("--dmrg-checkpoint-interval", type=int, default=1)
    parser.add_argument(
        "--no-dmrg-recenter",
        dest="dmrg_recenter",
        action="store_false",
        default=True,
        help="return the converged boundary-centered MPS without a final optimizing recenter pass",
    )
    parser.add_argument("--letta-from-dmrg-checkpoint", default=None)
    parser.add_argument("--letta-variant", choices=("standard", "nnn"), default="standard")
    parser.add_argument("--letta-gauge", choices=("conditional", "none"), default="conditional")
    parser.add_argument("--letta-solver", default="auto")
    parser.add_argument("--letta-load", default=None)
    parser.add_argument("--letta-save", default=None)
    parser.add_argument("--letta-expand-bond-dim", type=int, default=None)
    parser.add_argument("--letta-expand-noise", type=float, default=0.0)
    parser.add_argument("--letta-matrix-free-threshold", type=int, default=4096)
    parser.add_argument("--letta-matrix-free-tol", type=float, default=1.0e-8)
    parser.add_argument("--letta-matrix-free-maxiter", type=int, default=80)
    parser.add_argument("--ed-roots", type=int, default=4)
    parser.add_argument(
        "--ed-phase-gaps",
        action="store_true",
        help="also diagonalize neighboring particle/spin sectors for finite-cluster gaps",
    )
    parser.add_argument("--skip-ed", action="store_true")
    parser.add_argument("--check-mpo", action="store_true")
    parser.add_argument("--check-mpo-max-sites", type=int, default=4)
    parser.add_argument("--check-mpo-max-dim", type=int, default=4096)
    parser.add_argument("--skip-metrics", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default="/private/tmp/hubbard_2d_mps_vs_letta.json")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)
    nsites = int(args.lx) * int(args.ly)
    if args.site_grouping == "rung" and int(args.ly) < 2:
        raise ValueError("--site-grouping rung requires --ly >= 2.")
    if args.dmrg_only and args.letta_from_dmrg_checkpoint:
        raise ValueError("--dmrg-only cannot be combined with --letta-from-dmrg-checkpoint.")
    if args.skip_ed and args.ed_phase_gaps:
        raise ValueError("--ed-phase-gaps cannot be combined with --skip-ed.")
    if args.letta_sweeps < 0:
        raise ValueError("--letta-sweeps must be nonnegative.")
    if args.letta_sweeps == 0 and not args.letta_load:
        raise ValueError("--letta-sweeps 0 requires --letta-load for metrics-only evaluation.")
    if args.nup is None:
        args.nup = nsites // 2
    if args.ndown is None:
        args.ndown = nsites // 2
    if args.nup < 0 or args.ndown < 0 or args.nup > nsites or args.ndown > nsites:
        raise ValueError("nup and ndown must lie between 0 and Lx*Ly.")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    print(
        "2D Hubbard MPS vs LETTA: "
        f"{args.lx}x{args.ly}, Nup={args.nup}, Ndown={args.ndown}, "
        f"t={args.hopping:g}, U={args.hubbard_u:g}, mu={args.mu:g}, "
        f"D={args.bond_dim}, ordering={args.ordering}, grouping={args.site_grouping}, "
        f"variant={args.letta_variant}"
    )
    result = run_case(args)
    ed0 = None if result["ed"] is None else result["ed"]["energies"][0]
    if result["ed"] is not None:
        print(
            f"  ED E0 = {ed0: .12f}  "
            f"dim={result['ed']['info']['dimension']} bonds={len(result['ed']['info']['bonds'])}"
        )
    if result["ed_phase_gaps"] is not None:
        print(
            f"  finite-cluster gaps: charge={result['ed_phase_gaps']['charge_gap']:.8f}  "
            f"spin={result['ed_phase_gaps']['spin_gap']:.8f}"
        )
    if result["mpo_projected_evals"] is not None and result["ed"] is not None:
        diff = max(
            abs(a - b)
            for a, b in zip(result["mpo_projected_evals"], result["ed"]["energies"])
        )
        print(f"  projected-MPO spectrum max |dE| = {diff:.3e}")
    print(
        f"  E_MPS   = {result['dmrg_energy']: .12f}  "
        f"time={result['dmrg_time_s']:.2f}s  conv={result['dmrg_converged']}"
    )
    if result["dmrg_history"]:
        last_dmrg = result["dmrg_history"][-1]
        local_energy = last_dmrg.get("local_energy")
        truncation = last_dmrg.get("truncation")
        if local_energy is not None and abs(float(local_energy) - result["dmrg_energy"]) > 1.0e-8:
            print(
                f"  last local two-site E = {float(local_energy): .12f}  "
                f"post-truncation gap={float(local_energy) - result['dmrg_energy']:+.3e}  "
                f"trunc={0.0 if truncation is None else float(truncation):.3e}"
            )
    if result["letta_energy"] is None:
        print("  LETTA skipped.")
    else:
        letta_label = "NNN-LETTA" if result["params"]["letta_variant"] == "nnn" else "LETTA"
        print(
            f"  E_{letta_label} = {result['letta_energy']: .12f}  "
            f"time={result['letta_time_s']:.2f}s  conv={result['letta_converged']}  "
            f"dE={result['letta_minus_dmrg']:+.3e}"
        )
    if ed0 is not None:
        line = f"  DMRG-ED={result['dmrg_minus_ed']:+.3e}"
        if result["letta_minus_ed"] is not None:
            line += f"  LETTA-ED={result['letta_minus_ed']:+.3e}"
        print(line)
    if result["dmrg"] is None:
        print("  Metrics skipped.")
    else:
        metric_states = [("MPS", result["dmrg"])]
        if result["letta"] is not None:
            metric_states.append(("LETTA", result["letta"]))
        for label, metrics in metric_states:
            spin_peak = metrics["spin_structure_peak"]
            charge_peak = metrics["charge_structure_peak"]
            print(
                f"  {label}: Sdot(pi,pi)={metrics['spin_structure_factor_pi_pi_dot']:.6f} "
                f"m_AF^2={metrics['spin_order_parameter_sq_pi_pi']:.6f} "
                f"N(pi,pi)={metrics['charge_structure_factor_pi_pi']:.6f}"
            )
            print(
                f"    local moment={metrics['avg_local_moment']:.6f} "
                f"charge fluctuation={metrics['avg_charge_fluctuation']:.6f} "
                f"doublon={metrics['avg_doublon']:.6f} "
                f"onsite P(0)={metrics['onsite_pair_structure_q0']:.6e}"
            )
            print(
                f"    bond singlet: d-wave={metrics['d_wave_pair_structure_per_bond']:.6e} "
                f"d-wave offsite={metrics['d_wave_pair_offsite_per_bond']:.6e} "
                f"extended-s={metrics['extended_s_pair_structure_per_bond']:.6e}"
            )
            print(
                "    peaks: spin q/pi="
                f"({spin_peak['qx_over_pi']:.3g},{spin_peak['qy_over_pi']:.3g}) "
                f"S={spin_peak['value']:.6f}; charge q/pi="
                f"({charge_peak['qx_over_pi']:.3g},{charge_peak['qy_over_pi']:.3g}) "
                f"N={charge_peak['value']:.6f}"
            )
            print(
                "    staggered spin C(r) = "
                + " ".join(
                    f"{value:.3e}"
                    for value in metrics["staggered_spin_by_manhattan_distance"]
                )
            )
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Saved JSON: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
