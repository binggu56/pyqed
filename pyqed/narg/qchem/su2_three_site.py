#!/usr/bin/env python3
"""Three-site SU(2)-NARG growth prototype.

This grows the validated two-site SU(2) block by one local spatial orbital.
The three-site Hamiltonian is still built by projection from the exact
determinant Hamiltonian; the point of this file is to validate the SU(2)
branch/truncation flow before replacing projection with assembled residual
operators.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache, wraps
import os
import time

import numpy as np
from scipy.linalg import eigh

from pyqed.symmetry import Irrep, Leg, IrrepTensor, OpIrrep, spin_label
from pyqed import SpinHalfFermionOperators
from .su2_core import (
    Multiplet,
    asarray,
    cg,
    full_jw_model,
    local_site_multiplets,
    local_su2_branches,
    qchem_integrals,
    scalar_hamiltonian_irrep_tensor,
    su2_branch_update,
    su2_irrep_tensor_roots,
    su2_product_symmetry,
)
from .su2_two_site import (
    BranchMultiplet,
    RenormalizedSU2Block,
    TruncatedSU2NARG,
    build_renormalized_two_site_block,
    retained_multiplets,
    truncate_to_D,
)
from .su2_reduced_tensor import (
    ReducedSU2Tensor,
    add_reduced_tensors,
    coupled_reduced_product,
    reconstruct_component_block,
    reduced_tensor_from_components,
    scale_reduced_tensor,
)
from .su2_backend import resolve_su2_narg_backend
from .su2_cython import (
    CYTHON_AVAILABLE,
    accumulate_bilinear,
    product_tensor_estimate_entries as cython_product_tensor_estimate_entries,
    product_tensor_group_indices as cython_product_tensor_group_indices,
    product_tensor_pair_entries as cython_product_tensor_pair_entries,
    scalar_product_pair_entries as cython_scalar_product_pair_entries,
)


OPS = SpinHalfFermionOperators()
CU = OPS["Cu"]
CD = OPS["Cd"]
CDU = OPS["Cdu"]
CDD = OPS["Cdd"]
JW = OPS["JW"]
NU = OPS["Nu"]
ND = OPS["Nd"]
NTOT = OPS["Ntot"]


SU2_PROFILE_ENABLED = os.environ.get("SU2_NARG_PROFILE", "0") == "1"
SU2_PACKED_BILINEAR = os.environ.get("SU2_NARG_PACKED_BILINEAR", "1") != "0"
SU2_PACKED_BILINEAR_MIN_TERMS = int(
    os.environ.get("SU2_NARG_PACKED_BILINEAR_MIN_TERMS", "1")
)
SU2_COALESCE_BILINEAR = os.environ.get("SU2_NARG_COALESCE_BILINEAR", "1") != "0"
SU2_COALESCE_BILINEAR_MIN_TERMS = int(
    os.environ.get("SU2_NARG_COALESCE_BILINEAR_MIN_TERMS", "512")
)
SU2_COALESCE_BILINEAR_ATOL = float(
    os.environ.get("SU2_NARG_COALESCE_BILINEAR_ATOL", "0.0")
)
SU2_COMPILED_ANGULAR = (
    (
        os.environ.get("SU2_NARG_DISABLE_CPP_ANGULAR", "0") != "1"
        or (CYTHON_AVAILABLE and cython_product_tensor_pair_entries is not None)
    )
    and os.environ.get("SU2_NARG_COMPILED_ANGULAR", "1") != "0"
)
SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS = int(
    os.environ.get("SU2_NARG_COMPILED_ANGULAR_MIN_STATE_PAIRS", "0")
)
_CPP_ANGULAR_CHECKED = False
_CPP_PRODUCT_TENSOR_PAIR_ENTRIES = None
_CPP_PRODUCT_TENSOR_GROUP_INDICES = None
_CPP_ACCUMULATE_BILINEAR = None
_SU2_PROFILE: dict[str, dict[str, float | int]] = {}


def _cpp_angular_requested() -> bool:
    return os.environ.get("SU2_NARG_DISABLE_CPP_ANGULAR", "0").strip().lower() not in {
        "1",
        "true",
        "yes",
        "on",
    }


def _cpp_angular_kernels():
    """Return optional native angular-entry kernels, compiling lazily if needed."""
    global _CPP_ANGULAR_CHECKED
    global _CPP_PRODUCT_TENSOR_PAIR_ENTRIES
    global _CPP_PRODUCT_TENSOR_GROUP_INDICES
    global _CPP_ACCUMULATE_BILINEAR

    if _CPP_ANGULAR_CHECKED:
        return _CPP_PRODUCT_TENSOR_PAIR_ENTRIES, _CPP_PRODUCT_TENSOR_GROUP_INDICES
    _CPP_ANGULAR_CHECKED = True
    if not _cpp_angular_requested():
        return None, None
    try:
        from . import su2_native
    except Exception:
        return None, None
    if getattr(su2_native, "CPP_ANGULAR_AVAILABLE", False):
        _CPP_PRODUCT_TENSOR_PAIR_ENTRIES = getattr(
            su2_native,
            "product_tensor_pair_entries",
            None,
        )
        _CPP_PRODUCT_TENSOR_GROUP_INDICES = getattr(
            su2_native,
            "product_tensor_group_indices",
            None,
        )
        _CPP_ACCUMULATE_BILINEAR = getattr(su2_native, "accumulate_bilinear", None)
    return _CPP_PRODUCT_TENSOR_PAIR_ENTRIES, _CPP_PRODUCT_TENSOR_GROUP_INDICES


def _compiled_product_tensor_pair_available() -> bool:
    cpp_pair, _ = _cpp_angular_kernels()
    return cpp_pair is not None or (
        CYTHON_AVAILABLE and cython_product_tensor_pair_entries is not None
    )


def _cpp_accumulate_bilinear_kernel():
    _cpp_angular_kernels()
    return _CPP_ACCUMULATE_BILINEAR


def reset_su2_profile() -> None:
    """Clear accumulated optional SU(2)-NARG profiling counters."""
    _SU2_PROFILE.clear()


def su2_profile_snapshot() -> dict[str, dict[str, float | int]]:
    """Return a copy of the optional SU(2)-NARG profiling counters."""
    return {
        name: {"calls": int(data["calls"]), "time": float(data["time"])}
        for name, data in sorted(_SU2_PROFILE.items())
    }


class profile_section:
    """Optional low-overhead wall-clock profiler for coarse SU(2)-NARG stages."""

    def __init__(self, name: str):
        self.name = name
        self.start = 0.0

    def __enter__(self):
        if SU2_PROFILE_ENABLED:
            self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        if SU2_PROFILE_ENABLED:
            data = _SU2_PROFILE.setdefault(self.name, {"calls": 0, "time": 0.0})
            data["calls"] += 1
            data["time"] += time.perf_counter() - self.start
        return False


def profile_function(name: str):
    """Decorator form of ``profile_section`` for whole-function timings."""

    def decorate(func):
        @wraps(func)
        def wrapped(*args, **kwargs):
            with profile_section(name):
                return func(*args, **kwargs)

        return wrapped

    return decorate


@dataclass
class ThreeSiteSU2NARG:
    """Three-site SU(2)-NARG data in the coupled branch basis."""

    source_block: TruncatedSU2NARG
    branch_states: list[BranchMultiplet]
    leg: Leg
    bases: dict[Irrep, np.ndarray]
    provenance: dict[Irrep, list[BranchMultiplet]]
    hamiltonian: IrrepTensor


@dataclass(frozen=True)
class ComponentState:
    irrep: Irrep
    root_index: int
    local_index: int
    m2: int
    energy: float
    vector: np.ndarray


@dataclass(frozen=True)
class CoupledProductState:
    """One retained-block x local-site multiplet coupled to total SU(2)."""

    branch: str
    block_irrep: Irrep
    block_local_index: int
    local_irrep: Irrep
    local_index: int
    total_irrep: Irrep
    total_j2: int


@dataclass(frozen=True)
class PackedBilinearGroup:
    """One prepacked block/local contraction group for batched contraction."""

    block_key: tuple[Irrep, Irrep]
    local_key: tuple[Irrep, Irrep]
    rows: np.ndarray
    cols: np.ndarray
    block_rows: np.ndarray
    block_cols: np.ndarray
    local_rows: np.ndarray
    local_cols: np.ndarray
    coeffs: np.ndarray


@dataclass(frozen=True)
class PackedBilinearEntries:
    """Angular entries plus reusable index arrays."""

    entries: tuple
    groups: tuple[PackedBilinearGroup, ...]


def grow_su2_block_by_one_site(block_multiplets: list[Multiplet]) -> list[BranchMultiplet]:
    """Couple retained block multiplets to one SU(2) local site."""
    grouped = su2_branch_update(block_multiplets, local_su2_branches())
    states: list[BranchMultiplet] = []
    for branch in local_su2_branches():
        states.extend(BranchMultiplet(branch.name, mp) for mp in grouped[branch.name])
    return states


def coupled_product_states(block: TruncatedSU2NARG) -> list[CoupledProductState]:
    """Grown-basis labels matching ``grow_su2_block_by_one_site`` ordering."""
    states: list[CoupledProductState] = []
    local_indices: dict[Irrep, int] = {}
    local_by_branch = {}
    for branch in local_su2_branches():
        local_irrep = Irrep((branch.nelec, branch.j2))
        local_by_branch[branch.name] = (local_irrep, local_indices.get(local_irrep, 0))
        local_indices[local_irrep] = local_indices.get(local_irrep, 0) + 1

    block_labels = [
        (root.irrep, root.local_index)
        for root in block.kept_roots
    ]

    for branch in local_su2_branches():
        local_irrep, local_index = local_by_branch[branch.name]
        local_nelec, local_j2 = local_irrep.charge
        for block_irrep, block_local_index in block_labels:
            block_nelec, block_j2 = block_irrep.charge
            for total_j2 in range(abs(block_j2 - local_j2), block_j2 + local_j2 + 1, 2):
                total_irrep = Irrep((block_nelec + local_nelec, total_j2))
                states.append(
                    CoupledProductState(
                        branch=branch.name,
                        block_irrep=block_irrep,
                        block_local_index=block_local_index,
                        local_irrep=local_irrep,
                        local_index=local_index,
                        total_irrep=total_irrep,
                        total_j2=total_j2,
                    )
                )
    return states


def product_states_by_irrep(
    block: TruncatedSU2NARG,
    *,
    allowed_nelec: set[int] | None = None,
) -> dict[Irrep, list[CoupledProductState]]:
    """Group coupled product labels by total ``(Ne, j2)`` sector."""
    grouped: dict[Irrep, list[CoupledProductState]] = {}
    for state in coupled_product_states(block):
        if allowed_nelec is not None and state.total_irrep.charge[0] not in allowed_nelec:
            continue
        grouped.setdefault(state.total_irrep, []).append(state)
    return grouped


def product_states_for_block(block: RenormalizedSU2Block) -> dict[Irrep, list[CoupledProductState]]:
    """Product states, optionally filtered for a requested final Ne sector."""
    allowed_nelec = getattr(block, "_su2_allowed_final_nelec", None)
    allowed_key = None if allowed_nelec is None else tuple(sorted(allowed_nelec))
    cache = getattr(block, "_su2_product_states_cache", None)
    if cache is None:
        cache = {}
        setattr(block, "_su2_product_states_cache", cache)
    grouped = cache.get(allowed_key)
    if grouped is None:
        grouped = product_states_by_irrep(block.truncated, allowed_nelec=allowed_nelec)
        cache[allowed_key] = grouped
    return grouped


def scalar_product_angular_cache(block: RenormalizedSU2Block) -> dict:
    """Mutable cache for scalar-product angular contractions."""
    cache = getattr(block, "_su2_scalar_product_cache", None)
    if cache is None:
        cache = {}
        setattr(block, "_su2_scalar_product_cache", cache)
    return cache


def branch_leg(
    states: list[BranchMultiplet], m2: int | None = None
) -> tuple[Leg, dict[Irrep, np.ndarray], dict[Irrep, list[BranchMultiplet]]]:
    """Build SU(2) sector bases from grown branch multiplets."""
    sectors: dict[Irrep, list[np.ndarray]] = {}
    provenance: dict[Irrep, list[BranchMultiplet]] = {}
    for state in states:
        selected_m2 = state.multiplet.j2 if m2 is None else m2
        vec = state.multiplet.states.get(selected_m2)
        if vec is None:
            continue
        irrep = Irrep((state.multiplet.nelec, state.multiplet.j2))
        sectors.setdefault(irrep, []).append(vec)
        provenance.setdefault(irrep, []).append(state)

    dims = {irrep: len(cols) for irrep, cols in sectors.items()}
    bases = {irrep: np.column_stack(cols) for irrep, cols in sectors.items()}
    return Leg(dims, symmetry=su2_product_symmetry()), bases, provenance


def expanded_component_states(block: TruncatedSU2NARG) -> list[ComponentState]:
    """Expand retained SU(2) multiplets into explicit magnetic components."""
    states: list[ComponentState] = []
    local_indices = {}
    for root_index, mp in enumerate(retained_multiplets(block)):
        irrep = Irrep((mp.nelec, mp.j2))
        local_index = local_indices.get(irrep, 0)
        local_indices[irrep] = local_index + 1
        energy = block.kept_roots[root_index].energy
        for m2 in range(-mp.j2, mp.j2 + 1, 2):
            vec = mp.states[m2]
            states.append(ComponentState(irrep, root_index, local_index, m2, energy, vec))
    return states


def expanded_component_basis(block: TruncatedSU2NARG) -> tuple[np.ndarray, list[ComponentState]]:
    states = expanded_component_states(block)
    basis = np.column_stack([state.vector for state in states])
    gram = basis.conj().T @ basis
    err = np.max(np.abs(gram - np.eye(gram.shape[0])))
    if err > 1e-10:
        raise ValueError(f"expanded component basis is not orthonormal; max Gram error {err:g}")
    return basis, states


def projected_component_operator(basis: np.ndarray, op) -> np.ndarray:
    dense = asarray(op)
    return basis.conj().T @ dense @ basis


def expanded_component_operators(block: TruncatedSU2NARG, h1e2, eri2):
    """Renormalized two-site operators in the explicit component basis."""
    basis, states = expanded_component_basis(block)
    model = full_jw_model(h1e2, eri2, nelec=2)
    operators = {
        "Cdu": [projected_component_operator(basis, op) for op in model.Cdu],
        "Cdd": [projected_component_operator(basis, op) for op in model.Cdd],
        "Cu": [projected_component_operator(basis, op) for op in model.Cu],
        "Cd": [projected_component_operator(basis, op) for op in model.Cd],
    }
    h_block = np.diag([state.energy for state in states]).astype(complex)
    return basis, states, h_block, operators


def expanded_operator_from_reduced(states: list[ComponentState], reduced: ReducedSU2Tensor, q2: int) -> np.ndarray:
    """Reconstruct one component matrix in the explicit retained-component basis."""
    dim = len(states)
    out = np.zeros((dim, dim), dtype=complex)
    if dim == 0 or not reduced.blocks:
        return out

    dnelec, _ = reduced.op.charge
    grouped: dict[tuple[Irrep, int], list[tuple[int, int]]] = {}
    for pos, state in enumerate(states):
        grouped.setdefault((state.irrep, int(state.m2)), []).append(
            (pos, int(state.local_index))
        )

    for (bra_irrep, ket_irrep), reduced_block in reduced.blocks.items():
        bra_nelec, bra_j2 = bra_irrep.charge
        ket_nelec, ket_j2 = ket_irrep.charge
        if bra_nelec != ket_nelec + dnelec:
            continue
        _, rank2 = reduced.op.charge
        norm = np.sqrt(bra_j2 + 1.0)
        for ket_m2 in range(-ket_j2, ket_j2 + 1, 2):
            bra_m2 = ket_m2 + q2
            if bra_m2 < -bra_j2 or bra_m2 > bra_j2:
                continue
            bra_group = grouped.get((bra_irrep, bra_m2))
            ket_group = grouped.get((ket_irrep, ket_m2))
            if not bra_group or not ket_group:
                continue
            coeff = cg(ket_j2, ket_m2, rank2, q2, bra_j2, bra_m2)
            if abs(coeff) <= 1.0e-14:
                continue
            bra_pos, bra_local = zip(*bra_group)
            ket_pos, ket_local = zip(*ket_group)
            component_block = (coeff / norm) * reduced_block
            out[np.ix_(bra_pos, ket_pos)] = component_block[
                np.ix_(bra_local, ket_local)
            ]
    return out


def expanded_reduced_operators(block: RenormalizedSU2Block):
    """Renormalized two-site operators reconstructed from reduced SU(2) tensors."""
    basis, states = expanded_component_basis(block.truncated)
    h_block = np.diag([state.energy for state in states]).astype(complex)
    operators = {"Cdu": [], "Cdd": [], "Cu": [], "Cd": []}
    for site_index in range(2):
        cdag = block.reduced_operators[("Cdag", site_index)]
        ctilde = block.reduced_operators[("Ctilde", site_index)]
        operators["Cdu"].append(expanded_operator_from_reduced(states, cdag, q2=1))
        operators["Cdd"].append(expanded_operator_from_reduced(states, cdag, q2=-1))
        operators["Cu"].append(expanded_operator_from_reduced(states, ctilde, q2=-1))
        operators["Cd"].append(-expanded_operator_from_reduced(states, ctilde, q2=1))
    return basis, states, h_block, operators


@lru_cache(maxsize=None)
def local_reduced_operator(name: str) -> ReducedSU2Tensor:
    """One-site reduced tensor in the local SU(2) basis."""
    multiplets = local_site_multiplets()
    if name == "I":
        return reduced_tensor_from_components(
            multiplets,
            {0: np.eye(4, dtype=complex)},
            OpIrrep((0, 0)),
        )
    if name == "JW":
        return reduced_tensor_from_components(
            multiplets,
            {0: JW},
            OpIrrep((0, 0)),
        )
    if name == "Ntot":
        return reduced_tensor_from_components(
            multiplets,
            {0: NTOT},
            OpIrrep((0, 0)),
        )
    if name == "Nu":
        return reduced_tensor_from_components(
            multiplets,
            {0: NU},
            OpIrrep((0, 0)),
        )
    if name == "Nd":
        return reduced_tensor_from_components(
            multiplets,
            {0: ND},
            OpIrrep((0, 0)),
        )
    if name == "Cdag":
        return reduced_tensor_from_components(
            multiplets,
            {1: CDU, -1: CDD},
            OpIrrep((1, 1)),
        )
    if name == "Ctilde":
        return reduced_tensor_from_components(
            multiplets,
            {-1: CU, 1: -CD},
            OpIrrep((-1, 1)),
        )
    if name == "JWCtilde":
        return reduced_tensor_from_components(
            multiplets,
            {-1: JW @ CU, 1: -(JW @ CD)},
            OpIrrep((-1, 1)),
        )
    if name == "JWCdag":
        return reduced_tensor_from_components(
            multiplets,
            {1: JW @ CDU, -1: JW @ CDD},
            OpIrrep((1, 1)),
        )
    if name == "PairCreate":
        return reduced_tensor_from_components(
            multiplets,
            {0: CDU @ CDD},
            OpIrrep((2, 0)),
        )
    if name == "PairAnnihilate":
        return reduced_tensor_from_components(
            multiplets,
            {0: CD @ CU},
            OpIrrep((-2, 0)),
        )
    if name == "JWDensityCtilde":
        return reduced_tensor_from_components(
            multiplets,
            {-1: JW @ ND @ CU, 1: -(JW @ NU @ CD)},
            OpIrrep((-1, 1)),
        )
    raise KeyError(f"unknown local reduced operator {name!r}")


def local_reduced_scalar_operator(component_op: np.ndarray) -> ReducedSU2Tensor:
    """One-site reduced rank-0 tensor from a local component matrix."""
    return reduced_tensor_from_components(
        local_site_multiplets(),
        {0: component_op},
        OpIrrep((0, 0)),
    )


@lru_cache(maxsize=None)
def local_spin_density_tensor() -> ReducedSU2Tensor:
    """One-site rank-1 spin-density tensor in the Cdag/Ctilde convention."""
    return reduced_tensor_from_components(
        local_site_multiplets(),
        {
            -2: CDD @ CU,
            0: (CDU @ CU - CDD @ CD) / np.sqrt(2.0),
            2: -(CDU @ CD),
        },
        OpIrrep((0, 2)),
    )


def reduced_tensor_component_element(
    tensor: ReducedSU2Tensor,
    bra_irrep: Irrep,
    ket_irrep: Irrep,
    bra_index: int,
    ket_index: int,
    ket_m2: int,
    q2: int,
) -> complex:
    """One magnetic component matrix element from a reduced tensor block."""
    block = reconstruct_component_block(tensor, bra_irrep, ket_irrep, ket_m2, q2)
    if block.size == 0:
        return 0.0
    return block[bra_index, ket_index]


def reduced_product_component_element(
    bra: CoupledProductState,
    ket: CoupledProductState,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    *,
    total_rank2: int = 0,
    total_q2: int = 0,
    ket_total_m2: int | None = None,
    atol: float = 1e-14,
) -> complex:
    """Component matrix element of ``[A_block x B_local]^K_Q``.

    This is the explicit Clebsch-Gordan contraction in the coupled basis.  It
    uses only reduced matrix element blocks plus angular coefficients; no
    expanded up/down block matrices are formed.
    """
    block_rank2 = block_tensor.op.charge[1]
    local_rank2 = local_tensor.op.charge[1]
    ket_total_m2 = ket.total_j2 if ket_total_m2 is None else ket_total_m2
    bra_total_m2 = ket_total_m2 + total_q2
    if abs(bra_total_m2) > bra.total_j2 or abs(ket_total_m2) > ket.total_j2:
        return 0.0

    value = 0.0j
    bra_block_j2 = bra.block_irrep.charge[1]
    ket_block_j2 = ket.block_irrep.charge[1]
    bra_local_j2 = bra.local_irrep.charge[1]
    ket_local_j2 = ket.local_irrep.charge[1]

    for ket_block_m2 in range(-ket_block_j2, ket_block_j2 + 1, 2):
        ket_local_m2 = ket_total_m2 - ket_block_m2
        if ket_local_m2 < -ket_local_j2 or ket_local_m2 > ket_local_j2:
            continue
        ket_cg = cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            ket.total_j2,
            ket_total_m2,
        )
        if abs(ket_cg) <= atol:
            continue

        for q_block2 in range(-block_rank2, block_rank2 + 1, 2):
            q_local2 = total_q2 - q_block2
            if q_local2 < -local_rank2 or q_local2 > local_rank2:
                continue
            tensor_cg = cg(
                block_rank2,
                q_block2,
                local_rank2,
                q_local2,
                total_rank2,
                total_q2,
            )
            if abs(tensor_cg) <= atol:
                continue

            bra_block_m2 = ket_block_m2 + q_block2
            bra_local_m2 = ket_local_m2 + q_local2
            if bra_block_m2 < -bra_block_j2 or bra_block_m2 > bra_block_j2:
                continue
            if bra_local_m2 < -bra_local_j2 or bra_local_m2 > bra_local_j2:
                continue
            bra_cg = cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                bra.total_j2,
                bra_total_m2,
            )
            if abs(bra_cg) <= atol:
                continue

            block_element = reduced_tensor_component_element(
                block_tensor,
                bra.block_irrep,
                ket.block_irrep,
                bra.block_local_index,
                ket.block_local_index,
                ket_block_m2,
                q_block2,
            )
            if abs(block_element) <= atol:
                continue
            local_element = reduced_tensor_component_element(
                local_tensor,
                bra.local_irrep,
                ket.local_irrep,
                bra.local_index,
                ket.local_index,
                ket_local_m2,
                q_local2,
            )
            if abs(local_element) <= atol:
                continue
            value += bra_cg * ket_cg * tensor_cg * block_element * local_element

    return value


@lru_cache(maxsize=None)
def scalar_product_pair_coeff(
    total_j2: int,
    bra_block_charge: tuple[int, int],
    bra_local_charge: tuple[int, int],
    ket_block_charge: tuple[int, int],
    ket_local_charge: tuple[int, int],
    block_op_charge: tuple[int, int],
    local_op_charge: tuple[int, int],
    *,
    atol: float = 1e-14,
) -> float:
    """Angular scalar-product coefficient for one coupled irrep tuple."""
    _, bra_block_j2 = bra_block_charge
    _, bra_local_j2 = bra_local_charge
    _, ket_block_j2 = ket_block_charge
    _, ket_local_j2 = ket_local_charge
    _, block_rank2 = block_op_charge
    _, local_rank2 = local_op_charge

    ket_total_m2 = int(total_j2)
    bra_total_m2 = int(total_j2)
    value = 0.0

    for ket_block_m2 in range(-ket_block_j2, ket_block_j2 + 1, 2):
        ket_local_m2 = ket_total_m2 - ket_block_m2
        if ket_local_m2 < -ket_local_j2 or ket_local_m2 > ket_local_j2:
            continue
        ket_cg = cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            total_j2,
            ket_total_m2,
        )
        if abs(ket_cg) <= atol:
            continue

        for q_block2 in range(-block_rank2, block_rank2 + 1, 2):
            q_local2 = -q_block2
            if q_local2 < -local_rank2 or q_local2 > local_rank2:
                continue
            tensor_cg = cg(block_rank2, q_block2, local_rank2, q_local2, 0, 0)
            if abs(tensor_cg) <= atol:
                continue

            bra_block_m2 = ket_block_m2 + q_block2
            bra_local_m2 = ket_local_m2 + q_local2
            if bra_block_m2 < -bra_block_j2 or bra_block_m2 > bra_block_j2:
                continue
            if bra_local_m2 < -bra_local_j2 or bra_local_m2 > bra_local_j2:
                continue

            bra_cg = cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                total_j2,
                bra_total_m2,
            )
            block_cg = cg(
                ket_block_j2,
                ket_block_m2,
                block_rank2,
                q_block2,
                bra_block_j2,
                bra_block_m2,
            )
            local_cg = cg(
                ket_local_j2,
                ket_local_m2,
                local_rank2,
                q_local2,
                bra_local_j2,
                bra_local_m2,
            )
            if abs(bra_cg) <= atol or abs(block_cg) <= atol or abs(local_cg) <= atol:
                continue

            value += (
                bra_cg
                * ket_cg
                * tensor_cg
                * block_cg
                * local_cg
                / np.sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            )
    return value


@profile_function("scalar_product_angular_terms")
def scalar_product_angular_terms(
    block: RenormalizedSU2Block,
    block_op: OpIrrep,
    local_op: OpIrrep,
    *,
    atol: float = 1e-14,
):
    """Precompute CG contractions for scalar products on one grown basis."""
    allowed = getattr(block, "_su2_allowed_final_nelec", None)
    allowed_key = None if allowed is None else tuple(sorted(allowed))
    key = (allowed_key, block_op.charge, local_op.charge)
    cache = scalar_product_angular_cache(block)
    cached = cache.get(key)
    if cached is not None:
        return cached

    grouped = product_states_for_block(block)
    dims = {irrep: len(states) for irrep, states in grouped.items()}
    site = Leg(dims, symmetry=su2_product_symmetry())
    block_dnelec, block_rank2 = block_op.charge
    local_dnelec, local_rank2 = local_op.charge
    terms_by_irrep = {}

    if block_rank2 == 0 and local_rank2 == 0:
        for irrep, states in grouped.items():
            use_compiled_pair = (
                SU2_COMPILED_ANGULAR
                and cython_scalar_product_pair_entries is not None
                and len(states) * len(states) >= SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS
            )
            if use_compiled_pair:
                state_table = product_state_integer_table(states)
                terms_by_irrep[irrep] = compiled_scalar_product_pair_entries(
                    states,
                    state_table,
                    total_j2=irrep.charge[1],
                    block_dnelec=block_dnelec,
                    block_rank2=block_rank2,
                    local_dnelec=local_dnelec,
                    local_rank2=local_rank2,
                    atol=atol,
                )
                continue
            entries = []
            for bra_pos, bra in enumerate(states):
                bra_block_nelec, bra_block_j2 = bra.block_irrep.charge
                bra_local_nelec, bra_local_j2 = bra.local_irrep.charge
                for ket_pos, ket in enumerate(states):
                    ket_block_nelec, ket_block_j2 = ket.block_irrep.charge
                    ket_local_nelec, ket_local_j2 = ket.local_irrep.charge
                    if bra_block_nelec != ket_block_nelec + block_dnelec:
                        continue
                    if bra_local_nelec != ket_local_nelec + local_dnelec:
                        continue
                    if bra_block_j2 != ket_block_j2 or bra_local_j2 != ket_local_j2:
                        continue
                    coeff = 1.0 / np.sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
                    entries.append(
                        (
                            bra_pos,
                            ket_pos,
                            coeff,
                            (bra.block_irrep, ket.block_irrep),
                            bra.block_local_index,
                            ket.block_local_index,
                            (bra.local_irrep, ket.local_irrep),
                            bra.local_index,
                            ket.local_index,
                        )
                    )
            terms_by_irrep[irrep] = finalize_bilinear_entries(entries)
        cached = (site, terms_by_irrep)
        cache[key] = cached
        return cached

    for irrep, states in grouped.items():
        total_j2 = irrep.charge[1]
        use_compiled_pair = (
            SU2_COMPILED_ANGULAR
            and cython_scalar_product_pair_entries is not None
            and len(states) * len(states) >= SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS
        )
        if use_compiled_pair:
            state_table = product_state_integer_table(states)
            terms_by_irrep[irrep] = compiled_scalar_product_pair_entries(
                states,
                state_table,
                total_j2=total_j2,
                block_dnelec=block_dnelec,
                block_rank2=block_rank2,
                local_dnelec=local_dnelec,
                local_rank2=local_rank2,
                atol=atol,
            )
            continue
        entries = []
        for bra_pos, bra in enumerate(states):
            bra_block_nelec, bra_block_j2 = bra.block_irrep.charge
            bra_local_nelec, bra_local_j2 = bra.local_irrep.charge
            for ket_pos, ket in enumerate(states):
                ket_block_nelec, ket_block_j2 = ket.block_irrep.charge
                ket_local_nelec, ket_local_j2 = ket.local_irrep.charge
                if bra_block_nelec != ket_block_nelec + block_dnelec:
                    continue
                if bra_local_nelec != ket_local_nelec + local_dnelec:
                    continue

                coeff = scalar_product_pair_coeff(
                    total_j2,
                    bra.block_irrep.charge,
                    bra.local_irrep.charge,
                    ket.block_irrep.charge,
                    ket.local_irrep.charge,
                    block_op.charge,
                    local_op.charge,
                )
                if abs(coeff) <= atol:
                    continue
                entries.append(
                    (
                        bra_pos,
                        ket_pos,
                        coeff,
                        (bra.block_irrep, ket.block_irrep),
                        bra.block_local_index,
                        ket.block_local_index,
                        (bra.local_irrep, ket.local_irrep),
                        bra.local_index,
                        ket.local_index,
                    )
                )
        terms_by_irrep[irrep] = finalize_bilinear_entries(entries)

    cached = (site, terms_by_irrep)
    cache[key] = cached
    return cached


@profile_function("pack_bilinear_entries")
def pack_bilinear_entries(entries) -> PackedBilinearEntries:
    """Convert angular entries into reusable arrays for batched contractions."""
    entries = tuple(entries)
    grouped = {}
    for (
        bra_pos,
        ket_pos,
        coeff,
        block_key,
        block_bra_index,
        block_ket_index,
        local_key,
        local_bra_index,
        local_ket_index,
    ) in entries:
        grouped.setdefault((block_key, local_key), []).append(
            (
                bra_pos,
                ket_pos,
                block_bra_index,
                block_ket_index,
                local_bra_index,
                local_ket_index,
                coeff,
            )
        )

    groups = []
    for (block_key, local_key), values in grouped.items():
        rows, cols, block_rows, block_cols, local_rows, local_cols, coeffs = zip(*values)
        groups.append(
            PackedBilinearGroup(
                block_key,
                local_key,
                np.asarray(rows, dtype=np.int64),
                np.asarray(cols, dtype=np.int64),
                np.asarray(block_rows, dtype=np.int64),
                np.asarray(block_cols, dtype=np.int64),
                np.asarray(local_rows, dtype=np.int64),
                np.asarray(local_cols, dtype=np.int64),
                np.asarray(coeffs, dtype=np.complex128),
            )
        )
    return PackedBilinearEntries(tuple(entries), tuple(groups))


@profile_function("pack_compiled_bilinear_arrays")
def pack_compiled_bilinear_arrays(
    bra_states: list[CoupledProductState],
    ket_states: list[CoupledProductState],
    bra_table: np.ndarray,
    ket_table: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    coeffs: np.ndarray,
    block_rows: np.ndarray,
    block_cols: np.ndarray,
    local_rows: np.ndarray,
    local_cols: np.ndarray,
) -> PackedBilinearEntries:
    """Pack compiled angular arrays directly without tuple rehydration."""
    if rows.size == 0:
        return PackedBilinearEntries((), ())

    _, cpp_group_indices = _cpp_angular_kernels()
    if cpp_group_indices is not None:
        group_keys, group_starts, order = cpp_group_indices(
            bra_table,
            ket_table,
            rows,
            cols,
        )
    elif CYTHON_AVAILABLE and cython_product_tensor_group_indices is not None:
        group_keys, group_starts, order = cython_product_tensor_group_indices(
            bra_table,
            ket_table,
            rows,
            cols,
        )
    else:
        group_keys = group_starts = order = None

    if group_keys is not None:
        groups = []
        for group_index, key_values in enumerate(group_keys):
            start = int(group_starts[group_index])
            stop = int(group_starts[group_index + 1])
            idx = order[start:stop]
            block_key = (
                Irrep((int(key_values[0]), int(key_values[1]))),
                Irrep((int(key_values[2]), int(key_values[3]))),
            )
            local_key = (
                Irrep((int(key_values[4]), int(key_values[5]))),
                Irrep((int(key_values[6]), int(key_values[7]))),
            )
            groups.append(
                PackedBilinearGroup(
                    block_key,
                    local_key,
                    np.asarray(rows[idx], dtype=np.int64),
                    np.asarray(cols[idx], dtype=np.int64),
                    np.asarray(block_rows[idx], dtype=np.int64),
                    np.asarray(block_cols[idx], dtype=np.int64),
                    np.asarray(local_rows[idx], dtype=np.int64),
                    np.asarray(local_cols[idx], dtype=np.int64),
                    np.asarray(coeffs[idx], dtype=np.complex128),
                )
            )
        return PackedBilinearEntries((), tuple(groups))

    grouped: dict[tuple[tuple[Irrep, Irrep], tuple[Irrep, Irrep]], list[int]] = {}
    for entry_index, (row, col) in enumerate(zip(rows, cols)):
        bra = bra_states[int(row)]
        ket = ket_states[int(col)]
        key = (
            (bra.block_irrep, ket.block_irrep),
            (bra.local_irrep, ket.local_irrep),
        )
        grouped.setdefault(key, []).append(entry_index)

    groups = []
    for (block_key, local_key), indices in grouped.items():
        idx = np.asarray(indices, dtype=np.int64)
        groups.append(
            PackedBilinearGroup(
                block_key,
                local_key,
                np.asarray(rows[idx], dtype=np.int64),
                np.asarray(cols[idx], dtype=np.int64),
                np.asarray(block_rows[idx], dtype=np.int64),
                np.asarray(block_cols[idx], dtype=np.int64),
                np.asarray(local_rows[idx], dtype=np.int64),
                np.asarray(local_cols[idx], dtype=np.int64),
                np.asarray(coeffs[idx], dtype=np.complex128),
            )
        )
    return PackedBilinearEntries((), tuple(groups))


@profile_function("coalesce_bilinear_entries")
def coalesce_bilinear_entries(entries):
    """Sum duplicate bilinear contraction addresses before batching."""
    entries = tuple(entries)
    if len(entries) <= 1:
        return entries

    coefficients = {}
    for (
        bra_pos,
        ket_pos,
        coeff,
        block_key,
        block_bra_index,
        block_ket_index,
        local_key,
        local_bra_index,
        local_ket_index,
    ) in entries:
        key = (
            bra_pos,
            ket_pos,
            block_key,
            block_bra_index,
            block_ket_index,
            local_key,
            local_bra_index,
            local_ket_index,
        )
        coefficients[key] = coefficients.get(key, 0.0) + coeff

    if len(coefficients) == len(entries):
        return entries

    coalesced = []
    for (
        bra_pos,
        ket_pos,
        block_key,
        block_bra_index,
        block_ket_index,
        local_key,
        local_bra_index,
        local_ket_index,
    ), coeff in coefficients.items():
        if abs(coeff) <= SU2_COALESCE_BILINEAR_ATOL:
            continue
        coalesced.append(
            (
                bra_pos,
                ket_pos,
                coeff,
                block_key,
                block_bra_index,
                block_ket_index,
                local_key,
                local_bra_index,
                local_ket_index,
            )
        )
    return tuple(coalesced)


def finalize_bilinear_entries(entries):
    """Use packed entries only when the batch is large enough to pay off."""
    entries = tuple(entries)
    if SU2_COALESCE_BILINEAR and len(entries) >= SU2_COALESCE_BILINEAR_MIN_TERMS:
        entries = coalesce_bilinear_entries(entries)
    if SU2_PACKED_BILINEAR and len(entries) >= SU2_PACKED_BILINEAR_MIN_TERMS:
        return pack_bilinear_entries(entries)
    return entries


@profile_function("accumulate_bilinear_entries")
def accumulate_bilinear_entries(
    mat: np.ndarray,
    entries,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    *,
    prefactor: complex = 1.0,
) -> np.ndarray:
    """Accumulate precomputed bilinear angular entries into ``mat``."""
    if isinstance(entries, PackedBilinearEntries):
        if not entries.groups:
            return mat
        packed_groups = entries.groups
    else:
        if not entries:
            return mat
        packed_groups = None

    block_mats = {}
    local_mats = {}

    if packed_groups is None:
        for (
            bra_pos,
            ket_pos,
            coeff,
            block_key,
            block_bra_index,
            block_ket_index,
            local_key,
            local_bra_index,
            local_ket_index,
        ) in entries:
            block_mat = block_mats.get(block_key)
            if block_mat is None:
                block_mat = block_tensor.block(*block_key)
                block_mats[block_key] = block_mat
            if block_mat.size == 0:
                continue
            local_mat = local_mats.get(local_key)
            if local_mat is None:
                local_mat = local_tensor.block(*local_key)
                local_mats[local_key] = local_mat
            if local_mat.size == 0:
                continue
            mat[bra_pos, ket_pos] += (
                prefactor
                * coeff
                * block_mat[block_bra_index, block_ket_index]
                * local_mat[local_bra_index, local_ket_index]
            )
        return mat

    cpp_accumulate = None if CYTHON_AVAILABLE else _cpp_accumulate_bilinear_kernel()
    compiled_accumulate = CYTHON_AVAILABLE or cpp_accumulate is not None

    if compiled_accumulate:
        mat = np.ascontiguousarray(mat, dtype=np.complex128)
    for group in packed_groups:
        block_key = group.block_key
        block_mat = block_mats.get(block_key)
        if block_mat is None:
            block_mat = block_tensor.block(*block_key)
            if compiled_accumulate:
                block_mat = np.ascontiguousarray(block_mat, dtype=np.complex128)
            block_mats[block_key] = block_mat
        if block_mat.size == 0:
            continue
        local_key = group.local_key
        local_mat = local_mats.get(local_key)
        if local_mat is None:
            local_mat = local_tensor.block(*local_key)
            if compiled_accumulate:
                local_mat = np.ascontiguousarray(local_mat, dtype=np.complex128)
            local_mats[local_key] = local_mat
        if local_mat.size == 0:
            continue
        if CYTHON_AVAILABLE:
            accumulate_bilinear(
                mat,
                group.rows,
                group.cols,
                group.block_rows,
                group.block_cols,
                group.local_rows,
                group.local_cols,
                group.coeffs,
                block_mat,
                local_mat,
                prefactor,
            )
        elif cpp_accumulate is not None:
            cpp_accumulate(
                mat,
                group.rows,
                group.cols,
                group.block_rows,
                group.block_cols,
                group.local_rows,
                group.local_cols,
                group.coeffs,
                block_mat,
                local_mat,
                prefactor,
            )
        else:
            values = (
                prefactor
                * group.coeffs
                * block_mat[group.block_rows, group.block_cols]
                * local_mat[group.local_rows, group.local_cols]
            )
            np.add.at(mat, (group.rows, group.cols), values)
    return mat


@profile_function("reduced_scalar_product_irrep_tensor")
def reduced_scalar_product_irrep_tensor(
    block: RenormalizedSU2Block,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    *,
    prefactor: complex = 1.0,
) -> IrrepTensor:
    """Scalar product of a retained-block tensor and a local-site tensor."""
    site, terms_by_irrep = scalar_product_angular_terms(block, block_tensor.op, local_tensor.op)
    blocks = {}
    for irrep, entries in terms_by_irrep.items():
        dim = site.sector_dim(irrep)
        mat = np.zeros((dim, dim), dtype=complex)
        accumulate_bilinear_entries(
            mat,
            entries,
            block_tensor,
            local_tensor,
            prefactor=prefactor,
        )
        if np.any(np.abs(mat) > 1e-14):
            blocks[(irrep, irrep)] = mat
    return IrrepTensor(site, site, OpIrrep((0, 0)), blocks)


def product_tensor_angular_cache(block: RenormalizedSU2Block) -> dict:
    """Mutable cache for non-scalar product angular contractions."""
    cache = getattr(block, "_su2_product_tensor_cache", None)
    if cache is None:
        cache = {}
        setattr(block, "_su2_product_tensor_cache", cache)
    return cache


@lru_cache(maxsize=None)
def product_tensor_pair_coeff(
    bra_total_j2: int,
    ket_total_j2: int,
    total_rank2: int,
    total_q2: int,
    ket_total_m2: int,
    bra_block_charge: tuple[int, int],
    bra_local_charge: tuple[int, int],
    ket_block_charge: tuple[int, int],
    ket_local_charge: tuple[int, int],
    block_op_charge: tuple[int, int],
    local_op_charge: tuple[int, int],
    *,
    atol: float = 1e-14,
) -> float:
    """Angular product coefficient for one reduced product estimate."""
    _, bra_block_j2 = bra_block_charge
    _, bra_local_j2 = bra_local_charge
    _, ket_block_j2 = ket_block_charge
    _, ket_local_j2 = ket_local_charge
    _, block_rank2 = block_op_charge
    _, local_rank2 = local_op_charge

    bra_total_m2 = int(ket_total_m2) + int(total_q2)
    if bra_total_m2 < -bra_total_j2 or bra_total_m2 > bra_total_j2:
        return 0.0

    value = 0.0
    for ket_block_m2 in range(-ket_block_j2, ket_block_j2 + 1, 2):
        ket_local_m2 = ket_total_m2 - ket_block_m2
        if ket_local_m2 < -ket_local_j2 or ket_local_m2 > ket_local_j2:
            continue
        ket_cg = cg(
            ket_block_j2,
            ket_block_m2,
            ket_local_j2,
            ket_local_m2,
            ket_total_j2,
            ket_total_m2,
        )
        if abs(ket_cg) <= atol:
            continue

        for q_block2 in range(-block_rank2, block_rank2 + 1, 2):
            q_local2 = total_q2 - q_block2
            if q_local2 < -local_rank2 or q_local2 > local_rank2:
                continue
            tensor_cg = cg(block_rank2, q_block2, local_rank2, q_local2, total_rank2, total_q2)
            if abs(tensor_cg) <= atol:
                continue

            bra_block_m2 = ket_block_m2 + q_block2
            bra_local_m2 = ket_local_m2 + q_local2
            if bra_block_m2 < -bra_block_j2 or bra_block_m2 > bra_block_j2:
                continue
            if bra_local_m2 < -bra_local_j2 or bra_local_m2 > bra_local_j2:
                continue

            bra_cg = cg(
                bra_block_j2,
                bra_block_m2,
                bra_local_j2,
                bra_local_m2,
                bra_total_j2,
                bra_total_m2,
            )
            block_cg = cg(
                ket_block_j2,
                ket_block_m2,
                block_rank2,
                q_block2,
                bra_block_j2,
                bra_block_m2,
            )
            local_cg = cg(
                ket_local_j2,
                ket_local_m2,
                local_rank2,
                q_local2,
                bra_local_j2,
                bra_local_m2,
            )
            if abs(bra_cg) <= atol or abs(block_cg) <= atol or abs(local_cg) <= atol:
                continue

            value += (
                bra_cg
                * ket_cg
                * tensor_cg
                * block_cg
                * local_cg
                / np.sqrt((bra_block_j2 + 1.0) * (bra_local_j2 + 1.0))
            )
    return value


def product_state_integer_table(states: list[CoupledProductState]) -> np.ndarray:
    """Integer table consumed by the optional compiled angular-entry builder."""
    table = np.empty((len(states), 6), dtype=np.int64)
    for pos, state in enumerate(states):
        block_nelec, block_j2 = state.block_irrep.charge
        local_nelec, local_j2 = state.local_irrep.charge
        table[pos, 0] = block_nelec
        table[pos, 1] = block_j2
        table[pos, 2] = state.block_local_index
        table[pos, 3] = local_nelec
        table[pos, 4] = local_j2
        table[pos, 5] = state.local_index
    return table


@profile_function("compiled_scalar_product_pair_entries")
def compiled_scalar_product_pair_entries(
    states: list[CoupledProductState],
    state_table: np.ndarray,
    *,
    total_j2: int,
    block_dnelec: int,
    block_rank2: int,
    local_dnelec: int,
    local_rank2: int,
    atol: float,
):
    """Build scalar-product angular entries with one optional Cython call."""
    (
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    ) = cython_scalar_product_pair_entries(
        state_table,
        int(total_j2),
        int(block_dnelec),
        int(block_rank2),
        int(local_dnelec),
        int(local_rank2),
        float(atol),
    )
    return pack_compiled_bilinear_arrays(
        states,
        states,
        state_table,
        state_table,
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    )


@profile_function("compiled_product_tensor_estimate_entries")
def compiled_product_tensor_estimate_entries(
    bra_states: list[CoupledProductState],
    ket_states: list[CoupledProductState],
    bra_table: np.ndarray,
    ket_table: np.ndarray,
    *,
    bra_total_j2: int,
    ket_total_j2: int,
    total_rank2: int,
    total_q2: int,
    ket_total_m2: int,
    block_dnelec: int,
    block_rank2: int,
    local_dnelec: int,
    local_rank2: int,
    scale: float,
    atol: float,
):
    """Build product angular entries using the optional Cython index kernel."""
    (
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    ) = cython_product_tensor_estimate_entries(
        bra_table,
        ket_table,
        int(bra_total_j2),
        int(ket_total_j2),
        int(total_rank2),
        int(total_q2),
        int(ket_total_m2),
        int(block_dnelec),
        int(block_rank2),
        int(local_dnelec),
        int(local_rank2),
        float(atol),
    )
    if rows.size == 0:
        return ()

    entries = []
    for row, col, coeff, block_row, block_col, local_row, local_col in zip(
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    ):
        bra = bra_states[int(row)]
        ket = ket_states[int(col)]
        entries.append(
            (
                int(row),
                int(col),
                float(coeff) * scale,
                (bra.block_irrep, ket.block_irrep),
                int(block_row),
                int(block_col),
                (bra.local_irrep, ket.local_irrep),
                int(local_row),
                int(local_col),
            )
        )
    return tuple(entries)


@profile_function("compiled_product_tensor_pair_entries")
def compiled_product_tensor_pair_entries(
    bra_states: list[CoupledProductState],
    ket_states: list[CoupledProductState],
    bra_table: np.ndarray,
    ket_table: np.ndarray,
    *,
    bra_total_j2: int,
    ket_total_j2: int,
    total_rank2: int,
    block_dnelec: int,
    block_rank2: int,
    local_dnelec: int,
    local_rank2: int,
    atol: float,
):
    """Build averaged product angular entries with one optional Cython call."""
    cpp_pair_entries, _ = _cpp_angular_kernels()
    pair_entries = (
        cpp_pair_entries
        if cpp_pair_entries is not None
        else cython_product_tensor_pair_entries
    )
    if pair_entries is None:
        raise RuntimeError("compiled product tensor pair entries are unavailable")
    (
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    ) = pair_entries(
        bra_table,
        ket_table,
        int(bra_total_j2),
        int(ket_total_j2),
        int(total_rank2),
        int(block_dnelec),
        int(block_rank2),
        int(local_dnelec),
        int(local_rank2),
        float(atol),
    )
    return pack_compiled_bilinear_arrays(
        bra_states,
        ket_states,
        bra_table,
        ket_table,
        rows,
        cols,
        coeffs,
        block_rows,
        block_cols,
        local_rows,
        local_cols,
    )


@profile_function("product_tensor_angular_terms")
def product_tensor_angular_terms(
    block: RenormalizedSU2Block,
    block_op: OpIrrep,
    local_op: OpIrrep,
    total_rank2: int,
    *,
    atol: float = 1e-14,
):
    """Precompute angular entries for ``[A_block x B_local]^K``."""
    allowed = getattr(block, "_su2_allowed_final_nelec", None)
    allowed_key = None if allowed is None else tuple(sorted(allowed))
    key = (allowed_key, block_op.charge, local_op.charge, int(total_rank2))
    cache = product_tensor_angular_cache(block)
    cached = cache.get(key)
    if cached is not None:
        return cached

    grouped = product_states_for_block(block)
    site = Leg(
        {irrep: len(states) for irrep, states in grouped.items()},
        symmetry=su2_product_symmetry(),
    )
    block_dnelec, block_rank2 = block_op.charge
    local_dnelec, local_rank2 = local_op.charge
    dnelec = block_dnelec + local_dnelec
    op = OpIrrep((dnelec, int(total_rank2)))
    terms_by_pair = {}

    for bra_irrep, bra_states in grouped.items():
        bra_nelec, bra_j2 = bra_irrep.charge
        bra_table = None
        for ket_irrep, ket_states in grouped.items():
            ket_nelec, ket_j2 = ket_irrep.charge
            if bra_nelec != ket_nelec + dnelec:
                continue
            if not site.symmetry.allows(bra_irrep.charge, op.charge, ket_irrep.charge):
                continue

            use_compiled_pair = (
                SU2_COMPILED_ANGULAR
                and len(bra_states) * len(ket_states)
                >= SU2_COMPILED_ANGULAR_MIN_STATE_PAIRS
                and _compiled_product_tensor_pair_available()
            )
            if use_compiled_pair:
                if bra_table is None:
                    bra_table = product_state_integer_table(bra_states)
                ket_table = product_state_integer_table(ket_states)
                merged_entries = compiled_product_tensor_pair_entries(
                    bra_states,
                    ket_states,
                    bra_table,
                    ket_table,
                    bra_total_j2=bra_j2,
                    ket_total_j2=ket_j2,
                    total_rank2=total_rank2,
                    block_dnelec=block_dnelec,
                    block_rank2=block_rank2,
                    local_dnelec=local_dnelec,
                    local_rank2=local_rank2,
                    atol=atol,
                )
                if merged_entries.groups:
                    terms_by_pair[(bra_irrep, ket_irrep)] = merged_entries
                continue

            estimates = []
            for total_q2 in range(-total_rank2, total_rank2 + 1, 2):
                for ket_total_m2 in range(-ket_j2, ket_j2 + 1, 2):
                    bra_total_m2 = ket_total_m2 + total_q2
                    if bra_total_m2 < -bra_j2 or bra_total_m2 > bra_j2:
                        continue
                    out_coeff = cg(ket_j2, ket_total_m2, total_rank2, total_q2, bra_j2, bra_total_m2)
                    if abs(out_coeff) <= atol:
                        continue

                    entries = []
                    for bra_pos, bra in enumerate(bra_states):
                        bra_block_nelec, bra_block_j2 = bra.block_irrep.charge
                        bra_local_nelec, bra_local_j2 = bra.local_irrep.charge
                        for ket_pos, ket in enumerate(ket_states):
                            ket_block_nelec, ket_block_j2 = ket.block_irrep.charge
                            ket_local_nelec, ket_local_j2 = ket.local_irrep.charge
                            if bra_block_nelec != ket_block_nelec + block_dnelec:
                                continue
                            if bra_local_nelec != ket_local_nelec + local_dnelec:
                                continue

                            coeff = product_tensor_pair_coeff(
                                bra_j2,
                                ket_j2,
                                total_rank2,
                                total_q2,
                                ket_total_m2,
                                bra.block_irrep.charge,
                                bra.local_irrep.charge,
                                ket.block_irrep.charge,
                                ket.local_irrep.charge,
                                block_op.charge,
                                local_op.charge,
                            )
                            if abs(coeff) <= atol:
                                continue
                            entries.append(
                                (
                                    bra_pos,
                                    ket_pos,
                                    coeff,
                                    (bra.block_irrep, ket.block_irrep),
                                    bra.block_local_index,
                                    ket.block_local_index,
                                    (bra.local_irrep, ket.local_irrep),
                                    bra.local_index,
                                    ket.local_index,
                                )
                            )
                    if entries:
                        estimates.append((np.sqrt(bra_j2 + 1.0) / out_coeff, entries))
            if estimates:
                weight = 1.0 / len(estimates)
                merged_entries = []
                for scale, entries in estimates:
                    scaled = scale * weight
                    for (
                        bra_pos,
                        ket_pos,
                        coeff,
                        block_key,
                        block_bra_index,
                        block_ket_index,
                        local_key,
                        local_bra_index,
                        local_ket_index,
                    ) in entries:
                        merged_entries.append(
                            (
                                bra_pos,
                                ket_pos,
                                coeff * scaled,
                                block_key,
                                block_bra_index,
                                block_ket_index,
                                local_key,
                                local_bra_index,
                                local_ket_index,
                            )
                        )
                terms_by_pair[(bra_irrep, ket_irrep)] = finalize_bilinear_entries(
                    merged_entries
                )

    cached = (site, op, terms_by_pair)
    cache[key] = cached
    return cached


@profile_function("reduced_product_tensor_irrep")
def reduced_product_tensor_irrep(
    block: RenormalizedSU2Block,
    block_tensor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
    *,
    total_rank2: int,
    atol: float = 1e-12,
) -> ReducedSU2Tensor:
    """Reduced tensor for ``[A_block x B_local]^K`` on the grown basis.

    This is the non-scalar companion to ``reduced_scalar_product_irrep_tensor``:
    it never expands determinant operators, only contracts reduced matrix
    elements and CG coefficients in the retained-block x local coupled basis.
    """
    site, op, terms_by_pair = product_tensor_angular_terms(
        block,
        block_tensor.op,
        local_tensor.op,
        total_rank2,
        atol=atol,
    )
    blocks = {}

    for (bra_irrep, ket_irrep), entries in terms_by_pair.items():
        dim = (site.sector_dim(bra_irrep), site.sector_dim(ket_irrep))
        reduced_block = np.zeros(dim, dtype=complex)
        accumulate_bilinear_entries(
            reduced_block,
            entries,
            block_tensor,
            local_tensor,
        )
        if np.any(np.abs(reduced_block) > atol):
            blocks[(bra_irrep, ket_irrep)] = reduced_block

    return ReducedSU2Tensor(IrrepTensor(site, site, op, blocks))


@profile_function("rotate_reduced_tensor_to_truncated")
def rotate_reduced_tensor_to_truncated(
    truncated: TruncatedSU2NARG,
    tensor: ReducedSU2Tensor,
    *,
    atol: float = 1e-12,
    backend=None,
) -> ReducedSU2Tensor:
    """Rotate a reduced tensor from the grown source basis into kept states."""
    backend = resolve_su2_narg_backend(backend)
    block_specs = []
    for (bra_irrep, ket_irrep), old_block in tensor.blocks.items():
        if bra_irrep not in truncated.leg.dims or ket_irrep not in truncated.leg.dims:
            continue
        if (
            truncated.source.leg.sector_dim(bra_irrep) == 0
            or truncated.source.leg.sector_dim(ket_irrep) == 0
        ):
            continue
        if not truncated.leg.symmetry.allows(
            bra_irrep.charge,
            tensor.op.charge,
            ket_irrep.charge,
        ):
            continue
        if old_block.size == 0:
            continue
        u_bra = truncated.transform.block(bra_irrep, bra_irrep)
        u_ket = truncated.transform.block(ket_irrep, ket_irrep)
        block_specs.append(((bra_irrep, ket_irrep), u_bra, old_block, u_ket))

    blocks = {}
    for (bra_irrep, ket_irrep), new_block in backend.rotate_operator_blocks(block_specs):
        if np.any(np.abs(new_block) > atol):
            blocks[(bra_irrep, ket_irrep)] = new_block
    return ReducedSU2Tensor(IrrepTensor(truncated.leg, truncated.leg, tensor.op, blocks))


@profile_function("rotate_reduced_tensors_to_truncated")
def rotate_reduced_tensors_to_truncated(
    truncated: TruncatedSU2NARG,
    tensors: dict,
    *,
    atol: float = 1e-12,
    backend=None,
) -> dict:
    """Rotate many reduced tensors into kept states with one backend batch.

    This is the projection boundary for the SU2-NARG growth step.  Keeping all
    tensor blocks in one request lets the backend group same-shaped rotations
    across spinors, densities, pairs, and weighted future packages.
    """
    backend = resolve_su2_narg_backend(backend)
    block_specs = []
    ops = {}
    for tensor_key, tensor in tensors.items():
        ops[tensor_key] = tensor.op
        for (bra_irrep, ket_irrep), old_block in tensor.blocks.items():
            if bra_irrep not in truncated.leg.dims or ket_irrep not in truncated.leg.dims:
                continue
            if (
                truncated.source.leg.sector_dim(bra_irrep) == 0
                or truncated.source.leg.sector_dim(ket_irrep) == 0
            ):
                continue
            if not truncated.leg.symmetry.allows(
                bra_irrep.charge,
                tensor.op.charge,
                ket_irrep.charge,
            ):
                continue
            if old_block.size == 0:
                continue
            u_bra = truncated.transform.block(bra_irrep, bra_irrep)
            u_ket = truncated.transform.block(ket_irrep, ket_irrep)
            block_specs.append(((tensor_key, bra_irrep, ket_irrep), u_bra, old_block, u_ket))

    rotated_blocks = {tensor_key: {} for tensor_key in tensors}
    for (tensor_key, bra_irrep, ket_irrep), new_block in backend.rotate_operator_blocks(block_specs):
        if np.any(np.abs(new_block) > atol):
            rotated_blocks[tensor_key][(bra_irrep, ket_irrep)] = new_block

    return {
        tensor_key: ReducedSU2Tensor(
            IrrepTensor(truncated.leg, truncated.leg, ops[tensor_key], blocks)
        )
        for tensor_key, blocks in rotated_blocks.items()
    }


def direct_reduced_hopping_tensor(block: RenormalizedSU2Block, h1e, site_index: int = 2) -> IrrepTensor:
    """Direct reduced-space one-electron hopping between block and new site."""
    local_annihilate = local_reduced_operator("JWCtilde")
    out = None
    for i in range(site_index):
        term = reduced_scalar_product_irrep_tensor(
            block,
            block.reduced_operators[("Cdag", i)],
            local_annihilate,
            prefactor=np.sqrt(2.0) * h1e[i, site_index],
        )
        out = term if out is None else add_irrep_tensors(out, term)
    if out is None:
        grouped = product_states_for_block(block)
        return IrrepTensor(
            Leg(
                {irrep: len(states) for irrep, states in grouped.items()},
                symmetry=su2_product_symmetry(),
            ),
            Leg(
                {irrep: len(states) for irrep, states in grouped.items()},
                symmetry=su2_product_symmetry(),
            ),
            OpIrrep((0, 0)),
            {},
        )
    return out


def direct_reduced_spinor_tensor_coupling(
    block: RenormalizedSU2Block,
    block_spinor: ReducedSU2Tensor,
    local_tensor: ReducedSU2Tensor,
) -> IrrepTensor:
    """Direct reduced scalar coupling of a reduced block spinor to a local spinor."""
    term = reduced_scalar_product_irrep_tensor(
        block,
        block_spinor,
        local_tensor,
        prefactor=np.sqrt(2.0),
    )
    return add_irrep_tensors(term, term.adjoint())


def block_reduced_scalar_operator(
    block: RenormalizedSU2Block,
    component_op: np.ndarray,
) -> ReducedSU2Tensor:
    """Extract a retained-block rank-0 reduced tensor from a component operator."""
    return reduced_tensor_from_components(
        retained_multiplets(block.truncated),
        {0: component_op},
        OpIrrep((0, 0)),
    )


def block_reduced_scalar_from_retained_components(
    block: RenormalizedSU2Block,
    component_op: np.ndarray,
) -> ReducedSU2Tensor:
    """Extract a scalar reduced tensor from a retained component-basis matrix."""
    primitive_basis, _ = expanded_component_basis(block.truncated)
    lifted_op = primitive_basis @ component_op @ primitive_basis.conj().T
    return block_reduced_scalar_operator(block, lifted_op)


def block_retained_scalar_tensor(
    block: RenormalizedSU2Block,
    component_blocks: dict[Irrep, np.ndarray],
) -> ReducedSU2Tensor:
    """Build a scalar reduced tensor directly in retained multiplicity space."""
    blocks = {}
    for irrep, component_block in component_blocks.items():
        _, j2 = irrep.charge
        reduced_block = np.sqrt(j2 + 1.0) * component_block
        if np.any(np.abs(reduced_block) > 1e-14):
            blocks[(irrep, irrep)] = reduced_block
    return ReducedSU2Tensor(
        IrrepTensor(block.truncated.leg, block.truncated.leg, OpIrrep((0, 0)), blocks)
    )


def block_zero_reduced_tensor(block: RenormalizedSU2Block, op_irrep: OpIrrep) -> ReducedSU2Tensor:
    """Zero reduced tensor on the retained block site."""
    return ReducedSU2Tensor(IrrepTensor(block.truncated.leg, block.truncated.leg, op_irrep, {}))


def block_primitive_data(block: RenormalizedSU2Block):
    """Primitive retained-block multiplets and operators, when available."""
    multiplets = getattr(block, "_su2_multiplets", None)
    ops = getattr(block, "_su2_primitive_ops", None)
    if multiplets is None or ops is None:
        return None
    return multiplets, ops


def block_composite_cache(block: RenormalizedSU2Block) -> dict:
    """Mutable cache for reduced composites associated with a retained block."""
    cache = getattr(block, "_su2_composite_cache", None)
    if cache is None:
        cache = {}
        setattr(block, "_su2_composite_cache", cache)
    return cache


def add_reduced_terms(terms: list[ReducedSU2Tensor], fallback: ReducedSU2Tensor) -> ReducedSU2Tensor:
    """Add a possibly empty list of reduced tensors."""
    return add_reduced_tensors(*terms) if terms else fallback


def block_density_tensor(block: RenormalizedSU2Block, i: int, j: int) -> ReducedSU2Tensor:
    """Reduced scalar ``sum_sigma c^dag_i,sigma c_j,sigma``."""
    direct = block.reduced_operators.get(("Density", i, j))
    if direct is not None:
        return direct
    cache = block_composite_cache(block)
    key = ("density", i, j)
    if key not in cache:
        cache[key] = scale_reduced_tensor(
            coupled_reduced_product(
                block.reduced_operators[("Cdag", i)],
                block.reduced_operators[("Ctilde", j)],
                rank2=0,
            ),
            np.sqrt(2.0),
        )
    return cache[key]


def block_spin_density_tensor(block: RenormalizedSU2Block, i: int, j: int) -> ReducedSU2Tensor:
    """Reduced rank-1 spin-density tensor from ``Cdag_i x Ctilde_j``."""
    direct = block.reduced_operators.get(("SpinDensity", i, j))
    if direct is not None:
        return direct
    cache = block_composite_cache(block)
    key = ("spin_density", i, j)
    if key not in cache:
        cache[key] = coupled_reduced_product(
            block.reduced_operators[("Cdag", i)],
            block.reduced_operators[("Ctilde", j)],
            rank2=2,
        )
    return cache[key]


def block_pair_annihilate_tensor(block: RenormalizedSU2Block, i: int, j: int) -> ReducedSU2Tensor:
    """Reduced singlet pair annihilation ``0.5*(c_d_i c_u_j - c_u_i c_d_j)``."""
    direct = block.reduced_operators.get(("PairAnnihilate", i, j))
    if direct is not None:
        return direct
    cache = block_composite_cache(block)
    key = ("pair_annihilate", i, j)
    if key not in cache:
        cache[key] = scale_reduced_tensor(
            coupled_reduced_product(
                block.reduced_operators[("Ctilde", i)],
                block.reduced_operators[("Ctilde", j)],
                rank2=0,
            ),
            -1.0 / np.sqrt(2.0),
        )
    return cache[key]


def block_weighted_density_tensor(
    block: RenormalizedSU2Block,
    weights,
    site_index: int,
) -> ReducedSU2Tensor:
    """Linear combination of block density tensors."""
    primitive = block_primitive_data(block)
    if primitive is not None:
        multiplets, ops = primitive
        component = None
        for i in range(site_index):
            for j in range(site_index):
                coeff = weights[i, j]
                if abs(coeff) > 0.0:
                    term = coeff * (ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j])
                    component = term if component is None else component + term
        if component is not None:
            return reduced_tensor_from_components(multiplets, {0: component}, OpIrrep((0, 0)))
    terms = []
    for i in range(site_index):
        for j in range(site_index):
            coeff = weights[i, j]
            if abs(coeff) > 0.0:
                terms.append(scale_reduced_tensor(block_density_tensor(block, i, j), coeff))
    return add_reduced_terms(terms, block_zero_reduced_tensor(block, OpIrrep((0, 0))))


def block_weighted_spin_density_tensor(
    block: RenormalizedSU2Block,
    weights,
    site_index: int,
) -> ReducedSU2Tensor:
    """Linear combination of block spin-density tensors."""
    primitive = block_primitive_data(block)
    if primitive is not None:
        multiplets, ops = primitive
        components = {-2: None, 0: None, 2: None}
        for i in range(site_index):
            for j in range(site_index):
                coeff = weights[i, j]
                if abs(coeff) > 0.0:
                    terms = {
                        -2: coeff * (ops["Cdd"][i] @ ops["Cu"][j]),
                        0: coeff
                        * (ops["Cdu"][i] @ ops["Cu"][j] - ops["Cdd"][i] @ ops["Cd"][j])
                        / np.sqrt(2.0),
                        2: -coeff * (ops["Cdu"][i] @ ops["Cd"][j]),
                    }
                    for q2, term in terms.items():
                        components[q2] = term if components[q2] is None else components[q2] + term
        nonzero = {q2: op for q2, op in components.items() if op is not None}
        if nonzero:
            return reduced_tensor_from_components(multiplets, nonzero, OpIrrep((0, 2)))
    terms = []
    for i in range(site_index):
        for j in range(site_index):
            coeff = weights[i, j]
            if abs(coeff) > 0.0:
                terms.append(scale_reduced_tensor(block_spin_density_tensor(block, i, j), coeff))
    return add_reduced_terms(terms, block_zero_reduced_tensor(block, OpIrrep((0, 2))))


def block_weighted_pair_annihilate_tensor(
    block: RenormalizedSU2Block,
    weights,
    site_index: int,
) -> ReducedSU2Tensor:
    """Linear combination of block singlet-pair annihilation tensors."""
    primitive = block_primitive_data(block)
    if primitive is not None:
        multiplets, ops = primitive
        component = None
        for i in range(site_index):
            for j in range(site_index):
                coeff = weights[i, j]
                if abs(coeff) > 0.0:
                    term = 0.5 * coeff * (ops["Cd"][i] @ ops["Cu"][j] - ops["Cu"][i] @ ops["Cd"][j])
                    component = term if component is None else component + term
        if component is not None:
            return reduced_tensor_from_components(multiplets, {0: component}, OpIrrep((-2, 0)))
    terms = []
    for i in range(site_index):
        for j in range(site_index):
            coeff = weights[i, j]
            if abs(coeff) > 0.0:
                terms.append(scale_reduced_tensor(block_pair_annihilate_tensor(block, i, j), coeff))
    return add_reduced_terms(terms, block_zero_reduced_tensor(block, OpIrrep((-2, 0))))


def block_weighted_cdag_tensor(block: RenormalizedSU2Block, weights, site_index: int) -> ReducedSU2Tensor:
    """Linear combination of block creation spinors."""
    primitive = block_primitive_data(block)
    if primitive is not None:
        multiplets, ops = primitive
        up = None
        down = None
        for i in range(site_index):
            coeff = weights[i]
            if abs(coeff) > 0.0:
                up_term = coeff * ops["Cdu"][i]
                down_term = coeff * ops["Cdd"][i]
                up = up_term if up is None else up + up_term
                down = down_term if down is None else down + down_term
        if up is not None and down is not None:
            return reduced_tensor_from_components(multiplets, {1: up, -1: down}, OpIrrep((1, 1)))
    terms = [
        scale_reduced_tensor(block.reduced_operators[("Cdag", i)], weights[i])
        for i in range(site_index)
        if abs(weights[i]) > 0.0
    ]
    return add_reduced_terms(terms, block_zero_reduced_tensor(block, OpIrrep((1, 1))))


def block_cdag_density_tensor(
    block: RenormalizedSU2Block, k: int, j: int, i: int
) -> ReducedSU2Tensor:
    """Reduced spinor ``c^dag_k sum_sigma c^dag_j,sigma c_i,sigma``."""
    direct = block.reduced_operators.get(("CdagDensity", k, j, i))
    if direct is not None:
        return direct
    return coupled_reduced_product(
        block.reduced_operators[("Cdag", k)],
        block_density_tensor(block, j, i),
        rank2=1,
    )


@profile_function("direct_reduced_density_tensor")
def direct_reduced_density_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space block-density times local-density term."""
    density = block.reduced_operators.get(("NextDensity", site_index))
    if density is None:
        density = block_weighted_density_tensor(
            block, eri[:site_index, :site_index, site_index, site_index], site_index
        )
    return reduced_scalar_product_irrep_tensor(
        block,
        density,
        local_reduced_operator("Ntot"),
    )


@profile_function("direct_reduced_exchange_tensor")
def direct_reduced_exchange_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space exchange plus spin-flip two-electron package."""
    weights = eri[:site_index, site_index, site_index, :site_index]
    density = block.reduced_operators.get(("NextExchangeDensity", site_index))
    if density is None:
        density = block_weighted_density_tensor(block, weights, site_index)
    spin_density = block.reduced_operators.get(("NextExchangeSpinDensity", site_index))
    if spin_density is None:
        spin_density = block_weighted_spin_density_tensor(block, weights, site_index)
    scalar_part = reduced_scalar_product_irrep_tensor(
        block,
        scale_reduced_tensor(density, -0.5),
        local_reduced_operator("Ntot"),
    )
    spin_part = reduced_scalar_product_irrep_tensor(
        block,
        spin_density,
        local_spin_density_tensor(),
        prefactor=np.sqrt(3.0),
    )
    return add_irrep_tensors(scalar_part, spin_part)


@profile_function("direct_reduced_pair_transfer_tensor")
def direct_reduced_pair_transfer_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space singlet pair transfer plus Hermitian partner."""
    pair_annihilate = block.reduced_operators.get(("NextPairAnnihilate", site_index))
    if pair_annihilate is None:
        pair_annihilate = block_weighted_pair_annihilate_tensor(
            block, eri[site_index, :site_index, site_index, :site_index], site_index
        )
    pair_term = reduced_scalar_product_irrep_tensor(
        block,
        pair_annihilate,
        local_reduced_operator("PairCreate"),
    )
    return add_irrep_tensors(pair_term, pair_term.adjoint())


@profile_function("direct_reduced_v1_tensor")
def direct_reduced_v1_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space one-electron plus residual spinor hopping package."""
    spinor = block.reduced_operators.get(("NextV1Spinor", site_index))
    if spinor is not None:
        return direct_reduced_spinor_tensor_coupling(
            block,
            spinor,
            local_reduced_operator("JWCtilde"),
        )
    primitive = block_primitive_data(block)
    if primitive is not None:
        multiplets, ops = primitive
        v1u = None
        v1d = None
        for i in range(site_index):
            if abs(h1e[i, site_index]) > 0.0:
                up_term = h1e[i, site_index] * ops["Cdu"][i]
                down_term = h1e[i, site_index] * ops["Cdd"][i]
                v1u = up_term if v1u is None else v1u + up_term
                v1d = down_term if v1d is None else v1d + down_term
        for i in range(site_index):
            for j in range(site_index):
                residual = ops["Cdu"][j] @ ops["Cu"][i] + ops["Cdd"][j] @ ops["Cd"][i]
                for k in range(site_index):
                    coeff = eri[k, site_index, j, i]
                    if abs(coeff) > 0.0:
                        up_term = coeff * (ops["Cdu"][k] @ residual)
                        down_term = coeff * (ops["Cdd"][k] @ residual)
                        v1u = up_term if v1u is None else v1u + up_term
                        v1d = down_term if v1d is None else v1d + down_term
        if v1u is not None and v1d is not None:
            spinor = reduced_tensor_from_components(
                multiplets,
                {1: v1u, -1: v1d},
                OpIrrep((1, 1)),
            )
            return direct_reduced_spinor_tensor_coupling(
                block,
                spinor,
                local_reduced_operator("JWCtilde"),
            )
    terms = [block_weighted_cdag_tensor(block, h1e[:site_index, site_index], site_index)]
    for i in range(site_index):
        for j in range(site_index):
            for k in range(site_index):
                coeff = eri[k, site_index, j, i]
                if abs(coeff) > 0.0:
                    terms.append(
                        scale_reduced_tensor(
                            block_cdag_density_tensor(block, k, j, i),
                            coeff,
                        )
                    )
    spinor = add_reduced_terms(terms, block_zero_reduced_tensor(block, OpIrrep((1, 1))))
    return direct_reduced_spinor_tensor_coupling(
        block,
        spinor,
        local_reduced_operator("JWCtilde"),
    )


@profile_function("direct_reduced_v3_tensor")
def direct_reduced_v3_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space local density-assisted spinor hopping package."""
    spinor = block.reduced_operators.get(("NextV3Cdag", site_index))
    if spinor is None:
        spinor = block_weighted_cdag_tensor(
            block, eri[:site_index, site_index, site_index, site_index], site_index
        )
    return direct_reduced_spinor_tensor_coupling(
        block,
        spinor,
        local_reduced_operator("JWDensityCtilde"),
    )


@profile_function("direct_reduced_base_tensor")
def direct_reduced_base_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced-space block Hamiltonian plus local-site Hamiltonian."""
    h_blocks = {
        irrep: block.truncated.hamiltonian.block(irrep, irrep)
        for irrep in block.truncated.leg.irreps
    }
    i_blocks = {
        irrep: np.eye(block.truncated.leg.sector_dim(irrep), dtype=complex)
        for irrep in block.truncated.leg.irreps
    }
    block_h_tensor = block_retained_scalar_tensor(block, h_blocks)
    block_i_tensor = block_retained_scalar_tensor(block, i_blocks)
    local_h_tensor = local_reduced_scalar_operator(single_site_hamiltonian(h1e, eri, site_index))
    block_term = reduced_scalar_product_irrep_tensor(
        block,
        block_h_tensor,
        local_reduced_operator("I"),
    )
    local_term = reduced_scalar_product_irrep_tensor(
        block,
        block_i_tensor,
        local_h_tensor,
    )
    return add_irrep_tensors(block_term, local_term)


def direct_reduced_base_hopping_density_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Partial direct reduced H3: base + one-electron hopping + density."""
    hopping = direct_reduced_hopping_tensor(block, h1e, site_index=site_index)
    out = direct_reduced_base_tensor(block, h1e, eri, site_index=site_index)
    out = add_irrep_tensors(out, hopping)
    out = add_irrep_tensors(out, hopping.adjoint())
    out = add_irrep_tensors(out, direct_reduced_density_tensor(block, h1e, eri, site_index=site_index))
    return out


def direct_reduced_base_hopping_density_exchange_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Partial direct reduced H3 including the exchange/spin-flip package."""
    out = direct_reduced_base_hopping_density_tensor(block, h1e, eri, site_index=site_index)
    return add_irrep_tensors(out, direct_reduced_exchange_tensor(block, h1e, eri, site_index=site_index))


def direct_reduced_base_hopping_density_exchange_pair_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Partial direct reduced H3 including pair transfer."""
    out = direct_reduced_base_hopping_density_exchange_tensor(
        block, h1e, eri, site_index=site_index
    )
    return add_irrep_tensors(
        out,
        direct_reduced_pair_transfer_tensor(block, h1e, eri, site_index=site_index),
    )


@profile_function("direct_reduced_full_hamiltonian_tensor")
def direct_reduced_full_hamiltonian_tensor(
    block: RenormalizedSU2Block,
    h1e,
    eri,
    site_index: int = 2,
) -> IrrepTensor:
    """Direct reduced H3 assembled term-by-term in SU(2) tensor form."""
    out = direct_reduced_base_tensor(block, h1e, eri, site_index=site_index)
    out = add_irrep_tensors(out, direct_reduced_density_tensor(block, h1e, eri, site_index=site_index))
    out = add_irrep_tensors(out, direct_reduced_exchange_tensor(block, h1e, eri, site_index=site_index))
    out = add_irrep_tensors(out, direct_reduced_pair_transfer_tensor(block, h1e, eri, site_index=site_index))
    out = add_irrep_tensors(out, direct_reduced_v1_tensor(block, h1e, eri, site_index=site_index))
    out = add_irrep_tensors(out, direct_reduced_v3_tensor(block, h1e, eri, site_index=site_index))
    return out


@profile_function("add_irrep_tensors")
def add_irrep_tensors(left: IrrepTensor, right: IrrepTensor) -> IrrepTensor:
    """Add matching IrrepTensor blocks."""
    keys = set(left.blocks) | set(right.blocks)
    blocks = {}
    for key in keys:
        block = left.block(*key) + right.block(*key)
        if np.any(np.abs(block) > 1e-14):
            blocks[key] = block
    return IrrepTensor(left.bra, left.ket, left.op, blocks)


def single_site_hamiltonian(h1e, eri, p: int) -> np.ndarray:
    return h1e[p, p] * (CDU @ CU + CDD @ CD) + eri[p, p, p, p] * NU @ ND


def assemble_component_hamiltonian_from_operators(block_basis, h_block, ops, h1e, eri) -> np.ndarray:
    """Assemble H(block + site 2) from retained block operators and local ops."""
    p = 2
    nb = h_block.shape[0]
    ib = np.eye(nb, dtype=complex)
    iloc = np.eye(4, dtype=complex)
    h_total = np.kron(h_block, iloc) + np.kron(ib, single_site_hamiltonian(h1e, eri, p))

    cdu = ops["Cdu"]
    cdd = ops["Cdd"]
    cu = ops["Cu"]
    cd = ops["Cd"]

    density = np.zeros((nb, nb), dtype=complex)
    exchange_u = np.zeros((nb, nb), dtype=complex)
    exchange_d = np.zeros((nb, nb), dtype=complex)
    v2a = np.zeros((nb, nb), dtype=complex)
    v2b = np.zeros((nb, nb), dtype=complex)
    for i in range(p):
        for j in range(p):
            density += eri[i, j, p, p] * (cdu[i] @ cu[j] + cdd[i] @ cd[j])
            exchange_u += eri[i, p, p, j] * cdu[i] @ cu[j]
            exchange_d += eri[i, p, p, j] * cdd[i] @ cd[j]
            v2a -= eri[i, p, p, j] * cdd[i] @ cu[j]
            v2b += 0.5 * eri[p, i, p, j] * (cd[i] @ cu[j] - cu[i] @ cd[j])

    h_total += np.kron(density, NTOT)
    h_total -= np.kron(exchange_u, NU)
    h_total -= np.kron(exchange_d, ND)
    h2a = np.kron(v2a, CDU @ CD)
    h2b = np.kron(v2b, CDU @ CDD)
    h_total += h2a + h2a.conj().T + h2b + h2b.conj().T

    v1u = np.zeros((nb, nb), dtype=complex)
    v1d = np.zeros((nb, nb), dtype=complex)
    for i in range(p):
        v1u += h1e[i, p] * cdu[i]
        v1d += h1e[i, p] * cdd[i]

    for i in range(p):
        for j in range(p):
            for k in range(p):
                residual = cdu[j] @ cu[i] + cdd[j] @ cd[i]
                v1u += eri[k, p, j, i] * cdu[k] @ residual
                v1d += eri[k, p, j, i] * cdd[k] @ residual

    v1 = np.kron(v1u, JW @ CU) + np.kron(v1d, JW @ CD)
    h_total += v1 + v1.conj().T

    v3u = np.zeros((nb, nb), dtype=complex)
    v3d = np.zeros((nb, nb), dtype=complex)
    for i in range(p):
        v3u += eri[i, p, p, p] * cdu[i]
        v3d += eri[i, p, p, p] * cdd[i]
    h3 = np.kron(v3u, JW @ ND @ CU) + np.kron(v3d, JW @ NU @ CD)
    h_total += h3 + h3.conj().T

    h_total = 0.5 * (h_total + h_total.conj().T)
    return h_total


def assemble_component_hamiltonian_for_site3(block: TruncatedSU2NARG, h1e, eri) -> tuple[np.ndarray, np.ndarray]:
    """Assemble H3 using component operators projected from primitive operators."""
    h1e2 = h1e[:2, :2]
    eri2 = eri[:2, :2, :2, :2]
    block_basis, _, h_block, ops = expanded_component_operators(block, h1e2, eri2)
    h_total = assemble_component_hamiltonian_from_operators(block_basis, h_block, ops, h1e, eri)
    return h_total, block_basis


def assemble_reduced_hamiltonian_for_site3(block: RenormalizedSU2Block, h1e, eri) -> tuple[np.ndarray, np.ndarray]:
    """Assemble H3 using operators reconstructed from reduced SU(2) tensors."""
    block_basis, _, h_block, ops = expanded_reduced_operators(block)
    h_total = assemble_component_hamiltonian_from_operators(block_basis, h_block, ops, h1e, eri)
    return h_total, block_basis


def product_basis_coordinates(block_basis: np.ndarray, primitive_basis: np.ndarray) -> np.ndarray:
    """Represent primitive three-site vectors in retained-block x local coordinates."""
    nb = block_basis.shape[1]
    out = np.zeros((nb * 4, primitive_basis.shape[1]), dtype=complex)
    for col in range(primitive_basis.shape[1]):
        tensor = primitive_basis[:, col].reshape((block_basis.shape[0], 4))
        coeff = block_basis.conj().T @ tensor
        out[:, col] = coeff.reshape(nb * 4)
    return out


def assembled_hamiltonian_irrep_tensor(
    h_component: np.ndarray,
    block_basis: np.ndarray,
    leg: Leg,
    bases: dict[Irrep, np.ndarray],
) -> IrrepTensor:
    """Project an assembled product-basis Hamiltonian into grown SU(2) sectors."""
    blocks = {}
    for irrep, primitive_basis in bases.items():
        product_basis = product_basis_coordinates(block_basis, primitive_basis)
        block = product_basis.conj().T @ h_component @ product_basis
        blocks[(irrep, irrep)] = 0.5 * (block + block.conj().T)
    return IrrepTensor(leg, leg, OpIrrep((0, 0)), blocks)


def product_operator_irrep_tensor(
    operator_component: np.ndarray,
    block_basis: np.ndarray,
    leg: Leg,
    bases: dict[Irrep, np.ndarray],
) -> IrrepTensor:
    """Project a product-basis scalar operator into grown SU(2) sectors."""
    blocks = {}
    for irrep, primitive_basis in bases.items():
        product_basis = product_basis_coordinates(block_basis, primitive_basis)
        block = product_basis.conj().T @ operator_component @ product_basis
        if np.any(np.abs(block) > 1e-14):
            blocks[(irrep, irrep)] = block
    return IrrepTensor(leg, leg, OpIrrep((0, 0)), blocks)


def validate_direct_reduced_hopping(block: RenormalizedSU2Block, h1e) -> dict[Irrep, float]:
    """Compare direct reduced tensor-product hopping with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    component = np.zeros((nb * 4, nb * 4), dtype=complex)
    for i in range(2):
        component += h1e[i, 2] * (
            np.kron(ops["Cdu"][i], JW @ CU) + np.kron(ops["Cdd"][i], JW @ CD)
        )
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_hopping_tensor(block, h1e)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_density(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced density coupling with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    density = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            density += eri[i, j, 2, 2] * (
                ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j]
            )
    component = np.kron(density, NTOT)
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_density_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_exchange(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced exchange/spin-flip package with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    exchange_u = np.zeros((nb, nb), dtype=complex)
    exchange_d = np.zeros((nb, nb), dtype=complex)
    v2a = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            exchange_u += eri[i, 2, 2, j] * ops["Cdu"][i] @ ops["Cu"][j]
            exchange_d += eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cd"][j]
            v2a -= eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cu"][j]
    spinflip = np.kron(v2a, CDU @ CD)
    component = (
        -np.kron(exchange_u, NU)
        - np.kron(exchange_d, ND)
        + spinflip
        + spinflip.conj().T
    )
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_exchange_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_pair_transfer(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced pair-transfer package with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    v2b = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            v2b += 0.5 * eri[2, i, 2, j] * (
                ops["Cd"][i] @ ops["Cu"][j] - ops["Cu"][i] @ ops["Cd"][j]
            )
    pair = np.kron(v2b, CDU @ CDD)
    component = pair + pair.conj().T
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_pair_transfer_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_v1(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced v1 spinor package with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    v1u = np.zeros((nb, nb), dtype=complex)
    v1d = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        v1u += h1e[i, 2] * ops["Cdu"][i]
        v1d += h1e[i, 2] * ops["Cdd"][i]
    for i in range(2):
        for j in range(2):
            for k in range(2):
                residual = ops["Cdu"][j] @ ops["Cu"][i] + ops["Cdd"][j] @ ops["Cd"][i]
                v1u += eri[k, 2, j, i] * ops["Cdu"][k] @ residual
                v1d += eri[k, 2, j, i] * ops["Cdd"][k] @ residual
    v1 = np.kron(v1u, JW @ CU) + np.kron(v1d, JW @ CD)
    component = v1 + v1.conj().T
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_v1_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_v3(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced v3 density-assisted spinor package with components."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, _, ops = expanded_reduced_operators(block)
    nb = block_basis.shape[1]
    v3u = np.zeros((nb, nb), dtype=complex)
    v3d = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        v3u += eri[i, 2, 2, 2] * ops["Cdu"][i]
        v3d += eri[i, 2, 2, 2] * ops["Cdd"][i]
    h3 = np.kron(v3u, JW @ ND @ CU) + np.kron(v3d, JW @ NU @ CD)
    component = h3 + h3.conj().T
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_v3_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_base(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare direct reduced base Hamiltonian with component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, states = expanded_component_basis(block.truncated)
    h_block = np.diag([state.energy for state in states]).astype(complex)
    component = np.kron(h_block, np.eye(4, dtype=complex)) + np.kron(
        np.eye(h_block.shape[0], dtype=complex),
        single_site_hamiltonian(h1e, eri, 2),
    )
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_base_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_partial(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare partial direct reduced H3 with the same component terms."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, states, _, ops = expanded_reduced_operators(block)
    h_block = np.diag([state.energy for state in states]).astype(complex)
    nb = h_block.shape[0]
    component = np.kron(h_block, np.eye(4, dtype=complex)) + np.kron(
        np.eye(nb, dtype=complex),
        single_site_hamiltonian(h1e, eri, 2),
    )

    hopping = np.zeros((nb * 4, nb * 4), dtype=complex)
    for i in range(2):
        hopping += h1e[i, 2] * (
            np.kron(ops["Cdu"][i], JW @ CU) + np.kron(ops["Cdd"][i], JW @ CD)
        )
    component += hopping + hopping.conj().T

    density = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            density += eri[i, j, 2, 2] * (
                ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j]
            )
    component += np.kron(density, NTOT)

    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_base_hopping_density_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_partial_exchange(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare partial direct reduced H3 through exchange with component terms."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, states, _, ops = expanded_reduced_operators(block)
    h_block = np.diag([state.energy for state in states]).astype(complex)
    nb = h_block.shape[0]
    component = np.kron(h_block, np.eye(4, dtype=complex)) + np.kron(
        np.eye(nb, dtype=complex),
        single_site_hamiltonian(h1e, eri, 2),
    )

    hopping = np.zeros((nb * 4, nb * 4), dtype=complex)
    for i in range(2):
        hopping += h1e[i, 2] * (
            np.kron(ops["Cdu"][i], JW @ CU) + np.kron(ops["Cdd"][i], JW @ CD)
        )
    component += hopping + hopping.conj().T

    density = np.zeros((nb, nb), dtype=complex)
    exchange_u = np.zeros((nb, nb), dtype=complex)
    exchange_d = np.zeros((nb, nb), dtype=complex)
    v2a = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            density += eri[i, j, 2, 2] * (
                ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j]
            )
            exchange_u += eri[i, 2, 2, j] * ops["Cdu"][i] @ ops["Cu"][j]
            exchange_d += eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cd"][j]
            v2a -= eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cu"][j]
    component += np.kron(density, NTOT)
    component -= np.kron(exchange_u, NU)
    component -= np.kron(exchange_d, ND)
    spinflip = np.kron(v2a, CDU @ CD)
    component += spinflip + spinflip.conj().T

    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_base_hopping_density_exchange_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_partial_exchange_pair(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare partial direct reduced H3 through pair transfer with component terms."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, states, _, ops = expanded_reduced_operators(block)
    h_block = np.diag([state.energy for state in states]).astype(complex)
    nb = h_block.shape[0]
    component = np.kron(h_block, np.eye(4, dtype=complex)) + np.kron(
        np.eye(nb, dtype=complex),
        single_site_hamiltonian(h1e, eri, 2),
    )

    hopping = np.zeros((nb * 4, nb * 4), dtype=complex)
    for i in range(2):
        hopping += h1e[i, 2] * (
            np.kron(ops["Cdu"][i], JW @ CU) + np.kron(ops["Cdd"][i], JW @ CD)
        )
    component += hopping + hopping.conj().T

    density = np.zeros((nb, nb), dtype=complex)
    exchange_u = np.zeros((nb, nb), dtype=complex)
    exchange_d = np.zeros((nb, nb), dtype=complex)
    v2a = np.zeros((nb, nb), dtype=complex)
    v2b = np.zeros((nb, nb), dtype=complex)
    for i in range(2):
        for j in range(2):
            density += eri[i, j, 2, 2] * (
                ops["Cdu"][i] @ ops["Cu"][j] + ops["Cdd"][i] @ ops["Cd"][j]
            )
            exchange_u += eri[i, 2, 2, j] * ops["Cdu"][i] @ ops["Cu"][j]
            exchange_d += eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cd"][j]
            v2a -= eri[i, 2, 2, j] * ops["Cdd"][i] @ ops["Cu"][j]
            v2b += 0.5 * eri[2, i, 2, j] * (
                ops["Cd"][i] @ ops["Cu"][j] - ops["Cu"][i] @ ops["Cd"][j]
            )
    component += np.kron(density, NTOT)
    component -= np.kron(exchange_u, NU)
    component -= np.kron(exchange_d, ND)
    spinflip = np.kron(v2a, CDU @ CD)
    pair = np.kron(v2b, CDU @ CDD)
    component += spinflip + spinflip.conj().T + pair + pair.conj().T

    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_base_hopping_density_exchange_pair_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def validate_direct_reduced_full(block: RenormalizedSU2Block, h1e, eri) -> dict[Irrep, float]:
    """Compare full direct reduced H3 with full component assembly."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(block.truncated))
    site, bases, _ = branch_leg(branch_states)
    block_basis, _, h_block, ops = expanded_reduced_operators(block)
    component = assemble_component_hamiltonian_from_operators(block_basis, h_block, ops, h1e, eri)
    component_tensor = product_operator_irrep_tensor(component, block_basis, site, bases)
    direct_tensor = direct_reduced_full_hamiltonian_tensor(block, h1e, eri)
    return compare_irrep_tensors(direct_tensor, component_tensor)


def build_three_site_su2_narg(h1e, eri, source_block: TruncatedSU2NARG) -> ThreeSiteSU2NARG:
    """Grow a retained two-site block by one site and project H3."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(source_block))
    site, bases, provenance = branch_leg(branch_states)
    model = full_jw_model(h1e, eri, nelec=3)
    hamiltonian = scalar_hamiltonian_irrep_tensor(model.H, site, bases)
    return ThreeSiteSU2NARG(source_block, branch_states, site, bases, provenance, hamiltonian)


def build_three_site_su2_narg_assembled(h1e, eri, source_block: TruncatedSU2NARG) -> ThreeSiteSU2NARG:
    """Grow by one site and assemble H3 from retained block operators."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(source_block))
    site, bases, provenance = branch_leg(branch_states)
    h_component, block_basis = assemble_component_hamiltonian_for_site3(source_block, h1e, eri)
    hamiltonian = assembled_hamiltonian_irrep_tensor(h_component, block_basis, site, bases)
    return ThreeSiteSU2NARG(source_block, branch_states, site, bases, provenance, hamiltonian)


def build_three_site_su2_narg_reduced(h1e, eri, source_block: RenormalizedSU2Block) -> ThreeSiteSU2NARG:
    """Grow by one site and assemble H3 from reduced SU(2) tensor operators."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(source_block.truncated))
    site, bases, provenance = branch_leg(branch_states)
    h_component, block_basis = assemble_reduced_hamiltonian_for_site3(source_block, h1e, eri)
    hamiltonian = assembled_hamiltonian_irrep_tensor(h_component, block_basis, site, bases)
    return ThreeSiteSU2NARG(source_block.truncated, branch_states, site, bases, provenance, hamiltonian)


def build_three_site_su2_narg_direct_reduced(h1e, eri, source_block: RenormalizedSU2Block) -> ThreeSiteSU2NARG:
    """Grow by one site and assemble H3 directly from reduced SU(2) tensors."""
    branch_states = grow_su2_block_by_one_site(retained_multiplets(source_block.truncated))
    site, bases, provenance = branch_leg(branch_states)
    hamiltonian = direct_reduced_full_hamiltonian_tensor(source_block, h1e, eri)
    return ThreeSiteSU2NARG(source_block.truncated, branch_states, site, bases, provenance, hamiltonian)


def sector_counts(states: list[BranchMultiplet]) -> dict[str, dict[tuple[int, int], int]]:
    out: dict[str, dict[tuple[int, int], int]] = {}
    for state in states:
        key = (state.multiplet.nelec, state.multiplet.j2)
        branch_counts = out.setdefault(state.branch, {})
        branch_counts[key] = branch_counts.get(key, 0) + 1
    return {branch: dict(sorted(counts.items())) for branch, counts in out.items()}


def diagonalize_sector(narg: ThreeSiteSU2NARG, nelec: int, j2: int, nroots: int = 8):
    irrep = Irrep((nelec, j2))
    block = narg.hamiltonian.block(irrep, irrep)
    evals, evecs = eigh(block)
    return evals[:nroots], evecs[:, :nroots], block


def print_branch_table(narg: ThreeSiteSU2NARG) -> None:
    print("Three-site SU2 branch update:")
    for branch, counts in sector_counts(narg.branch_states).items():
        pieces = [
            f"Ne={nelec} S={spin_label(j2)} dim={dim}"
            for (nelec, j2), dim in counts.items()
        ]
        print(f"  {branch}: " + "; ".join(pieces))


def compare_three_site_roots(h1e, eri, narg: ThreeSiteSU2NARG, nelec: int, j2: int, nroots: int):
    roots, _, block = diagonalize_sector(narg, nelec, j2, nroots=nroots)
    ref_roots, _, _ = su2_irrep_tensor_roots(h1e, eri, nelec, j2, nroots=nroots)
    ncompare = min(len(roots), len(ref_roots))
    diff = np.max(np.abs(roots[:ncompare] - ref_roots[:ncompare])) if ncompare else 0.0
    return roots, ref_roots, block, diff


def compare_two_narg_blocks(left: ThreeSiteSU2NARG, right: ThreeSiteSU2NARG) -> dict[Irrep, float]:
    """Blockwise Hamiltonian difference norms for matching grown bases."""
    errors = {}
    for irrep in left.leg.irreps:
        lblock = left.hamiltonian.block(irrep, irrep)
        rblock = right.hamiltonian.block(irrep, irrep)
        if lblock.shape == rblock.shape and lblock.size:
            errors[irrep] = float(np.linalg.norm(lblock - rblock))
    return errors


def compare_irrep_tensors(left: IrrepTensor, right: IrrepTensor) -> dict[Irrep, float]:
    """Blockwise norm differences for matching scalar IrrepTensors."""
    errors = {}
    for irrep in left.bra.irreps:
        lblock = left.block(irrep, irrep)
        rblock = right.block(irrep, irrep)
        if lblock.shape == rblock.shape and lblock.size:
            errors[irrep] = float(np.linalg.norm(lblock - rblock))
    return errors


def validate_three_site_growth() -> None:
    mol, mf, h1e, eri = qchem_integrals(3, span=1.5, basis="sto6g")
    enuc = mol.energy_nuc()

    h1e2 = h1e[:2, :2]
    eri2 = eri[:2, :2, :2, :2]

    exact_two_site_block = build_renormalized_two_site_block(h1e2, eri2, D=10)
    exact_growth = build_three_site_su2_narg(h1e, eri, exact_two_site_block.truncated)
    exact_assembled_growth = build_three_site_su2_narg_assembled(
        h1e, eri, exact_two_site_block.truncated
    )
    exact_reduced_growth = build_three_site_su2_narg_reduced(h1e, eri, exact_two_site_block)
    exact_direct_reduced_growth = build_three_site_su2_narg_direct_reduced(
        h1e, eri, exact_two_site_block
    )

    truncated_two_site_block = build_renormalized_two_site_block(
        h1e2, eri2, D=4, allowed_nelec={1, 2, 3}
    )
    truncated_growth = build_three_site_su2_narg(h1e, eri, truncated_two_site_block.truncated)
    truncated_assembled_growth = build_three_site_su2_narg_assembled(
        h1e, eri, truncated_two_site_block.truncated
    )
    truncated_reduced_growth = build_three_site_su2_narg_reduced(h1e, eri, truncated_two_site_block)
    truncated_direct_reduced_growth = build_three_site_su2_narg_direct_reduced(
        h1e, eri, truncated_two_site_block
    )

    print("Three-site SU2-NARG growth prototype")
    print("H3 span=1.5 spacing=1.5 Bohr")
    print(f"RHF total energy = {mf.e_tot:.12f}")
    print(f"nuclear repulsion = {enuc:.12f}")
    print()

    print("Exact two-site block grown to site 3:")
    print_branch_table(exact_growth)
    print("  direct reduced base-vs-component errors:")
    for irrep, err in validate_direct_reduced_base(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced hopping-vs-component errors:")
    for irrep, err in validate_direct_reduced_hopping(exact_two_site_block, h1e).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced density-vs-component errors:")
    for irrep, err in validate_direct_reduced_density(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced exchange-vs-component errors:")
    for irrep, err in validate_direct_reduced_exchange(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced pair-transfer-vs-component errors:")
    for irrep, err in validate_direct_reduced_pair_transfer(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced v1-vs-component errors:")
    for irrep, err in validate_direct_reduced_v1(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced v3-vs-component errors:")
    for irrep, err in validate_direct_reduced_v3(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H+exchange-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial_exchange(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H+exchange+pair-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial_exchange_pair(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced full-H-vs-component errors:")
    for irrep, err in validate_direct_reduced_full(exact_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  assembled-vs-projected block errors:")
    for irrep, err in compare_two_narg_blocks(exact_assembled_growth, exact_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  reduced-vs-component-assembled block errors:")
    for irrep, err in compare_two_narg_blocks(exact_reduced_growth, exact_assembled_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct-reduced-vs-component-assembled block errors:")
    for irrep, err in compare_two_narg_blocks(exact_direct_reduced_growth, exact_assembled_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    for j2, label in [(1, "doublet"), (3, "quartet")]:
        roots, ref_roots, block, diff = compare_three_site_roots(
            h1e, eri, exact_reduced_growth, nelec=3, j2=j2, nroots=6
        )
        print(f"  {label} block shape={block.shape}")
        print(f"  {label} roots = {np.array2string(roots + enuc, precision=10, separator=', ')}")
        print(f"  reduced-assembled max diff vs exact SU2 = {diff:.3e}")
    print()

    print("D=4 two-site block grown to site 3:")
    print_branch_table(truncated_growth)
    print("  direct reduced base-vs-component errors:")
    for irrep, err in validate_direct_reduced_base(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced hopping-vs-component errors:")
    for irrep, err in validate_direct_reduced_hopping(truncated_two_site_block, h1e).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced density-vs-component errors:")
    for irrep, err in validate_direct_reduced_density(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced exchange-vs-component errors:")
    for irrep, err in validate_direct_reduced_exchange(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced pair-transfer-vs-component errors:")
    for irrep, err in validate_direct_reduced_pair_transfer(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced v1-vs-component errors:")
    for irrep, err in validate_direct_reduced_v1(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced v3-vs-component errors:")
    for irrep, err in validate_direct_reduced_v3(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H+exchange-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial_exchange(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced partial-H+exchange+pair-vs-component errors:")
    for irrep, err in validate_direct_reduced_partial_exchange_pair(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct reduced full-H-vs-component errors:")
    for irrep, err in validate_direct_reduced_full(truncated_two_site_block, h1e, eri).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  assembled-vs-projected block errors:")
    for irrep, err in compare_two_narg_blocks(truncated_assembled_growth, truncated_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  reduced-vs-component-assembled block errors:")
    for irrep, err in compare_two_narg_blocks(truncated_reduced_growth, truncated_assembled_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    print("  direct-reduced-vs-component-assembled block errors:")
    for irrep, err in compare_two_narg_blocks(truncated_direct_reduced_growth, truncated_assembled_growth).items():
        nelec, j2 = irrep.charge
        print(f"    Ne={nelec} S={spin_label(j2)}: {err:.3e}")
    for j2, label in [(1, "doublet"), (3, "quartet")]:
        roots, ref_roots, block, diff = compare_three_site_roots(
            h1e, eri, truncated_reduced_growth, nelec=3, j2=j2, nroots=6
        )
        print(f"  {label} block shape={block.shape}")
        print(f"  {label} roots = {np.array2string(roots + enuc, precision=10, separator=', ')}")
        print(f"  reduced-assembled max diff vs exact SU2 = {diff:.3e}")
    print()

    print(f"Exact-growth IrrepTensor dense dim = {exact_growth.hamiltonian.to_dense().shape[0]}")
    print(f"D=4-growth IrrepTensor dense dim = {truncated_growth.hamiltonian.to_dense().shape[0]}")
    print(f"Primitive determinant dim = {asarray(full_jw_model(h1e, eri, nelec=3).H).shape[0]}")


def main() -> None:
    validate_three_site_growth()


if __name__ == "__main__":
    main()
