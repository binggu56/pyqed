"""Spatial-orbital transformations of fully reduced SU(2) MPSs.

The implementation factors a nonsingular one-particle map into diagonal
scalings and adjacent two-orbital rotations.  Their second-quantized actions
are applied directly to reduced charge x SU(2) channel blocks.  No determinant
vector or spin-component MPS is constructed.

This follows the nonunitary orbital-transformation strategy of Malmqvist,
Int. J. Quantum Chem. 30, 479 (1986), https://doi.org/10.1002/qua.560300404,
and its MPS use by Knecht et al., J. Chem. Theory Comput. 12, 5881 (2016),
https://doi.org/10.1021/acs.jctc.6b00889.  The circuit realization here is a
PyQED implementation, not a reproduction of either program.  It is exact up
to floating-point roundoff when ``cutoff=0`` and ``max_bond=None``.  An exact
general orbital map can nevertheless grow the MPS bond dimension
exponentially; sector preservation removes determinant-space storage, not that
fundamental worst case.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from pyqed.mps.su2 import SpatialOrbitalSite, fuse_charge_spin_sectors

from .contraction import merge_mps_sites
from .coupling import clebsch_gordan, ordered_two_m_values
from .decompose import svd_two_site
from .mps import MPS
from .tensor import NonabelianTensor


def is_fully_reduced_su2_mps(state):
    """Return whether every site uses PyQED's fully reduced SU(2) layout."""
    sites = list(getattr(state, "sites", state))
    return bool(sites) and all(
        isinstance(site, NonabelianTensor)
        and site.rank == 3
        and (site.metadata or {}).get("physical_basis") == "fully_reduced_su2"
        for site in sites
    )


def _as_reduced_mps(state):
    if not is_fully_reduced_su2_mps(state):
        raise TypeError("Expected a fully reduced charge x SU(2) spatial-orbital MPS.")
    if isinstance(state, MPS):
        return state.copy()
    return MPS.from_sites(
        list(getattr(state, "sites", state)),
        center=getattr(state, "center", None),
        target_sector=getattr(state, "target_sector", None),
    )


def _second_quantized_two_orbital_gate(one_particle_gate):
    """Second quantize a 2x2 spin-independent map in the local d=4 basis."""
    gate = np.asarray(one_particle_gate, dtype=complex)
    if gate.shape != (2, 2):
        raise ValueError(f"Two-orbital gate must have shape (2, 2), got {gate.shape}.")

    # Mode order is (alpha_i, beta_i, alpha_j, beta_j).  The spatial-site
    # component convention is |0>, |alpha>, |beta>, |alpha beta>.
    spin_gate = np.zeros((4, 4), dtype=complex)
    spin_gate[np.ix_((0, 2), (0, 2))] = gate
    spin_gate[np.ix_((1, 3), (1, 3))] = gate
    local_bits = (
        (0, 0),
        (1, 0),
        (0, 1),
        (1, 1),
    )
    occupations = []
    spin_counts = []
    for left in local_bits:
        for right in local_bits:
            bits = (left[0], left[1], right[0], right[1])
            occupations.append(np.flatnonzero(bits))
            spin_counts.append((bits[0] + bits[2], bits[1] + bits[3]))

    fock_gate = np.zeros((4, 4, 4, 4), dtype=complex)
    for out_index, out_occ in enumerate(occupations):
        out_left, out_right = divmod(out_index, 4)
        for in_index, in_occ in enumerate(occupations):
            if spin_counts[out_index] != spin_counts[in_index]:
                continue
            in_left, in_right = divmod(in_index, 4)
            if len(out_occ) == 0:
                value = 1.0
            else:
                # NumPy's batched LAPACK determinant path can report harmless
                # divide-by-zero warnings for exactly singular minors.
                with np.errstate(divide="ignore", invalid="ignore"):
                    value = np.linalg.det(spin_gate[np.ix_(out_occ, in_occ)])
            fock_gate[out_left, out_right, in_left, in_right] = value
    return fock_gate


@lru_cache(maxsize=None)
def _channel_component_tensor(q_left, q_phys1, q_mid, q_phys2, q_right):
    """CG expansion for one left-associated two-site reduced channel."""
    canonical_site = SpatialOrbitalSite()
    physical_indices = {
        sector: tuple(indices)
        for sector, indices in zip(canonical_site.qn, canonical_site.state_index)
    }
    if q_phys1 not in physical_indices or q_phys2 not in physical_indices:
        raise ValueError("Unsupported physical sector in reduced spatial-orbital channel.")

    left_ms = ordered_two_m_values(q_left.irrep)
    phys1_ms = ordered_two_m_values(q_phys1.irrep)
    mid_ms = ordered_two_m_values(q_mid.irrep)
    phys2_ms = ordered_two_m_values(q_phys2.irrep)
    right_ms = ordered_two_m_values(q_right.irrep)
    out = np.zeros((len(left_ms), 4, 4, len(right_ms)), dtype=float)
    for il, ml in enumerate(left_ms):
        for ip1, mp1 in enumerate(phys1_ms):
            c1_mid = ml + mp1
            if c1_mid not in mid_ms:
                continue
            c1 = clebsch_gordan(
                q_left.irrep, q_phys1.irrep, q_mid.irrep, ml, mp1, c1_mid
            )
            if c1 == 0.0:
                continue
            for ip2, mp2 in enumerate(phys2_ms):
                mr = c1_mid + mp2
                if mr not in right_ms:
                    continue
                c2 = clebsch_gordan(
                    q_mid.irrep, q_phys2.irrep, q_right.irrep, c1_mid, mp2, mr
                )
                if c2 == 0.0:
                    continue
                out[
                    il,
                    physical_indices[q_phys1][ip1],
                    physical_indices[q_phys2][ip2],
                    right_ms.index(mr),
                ] += c1 * c2
    return out


@lru_cache(maxsize=None)
def _allowed_output_channels(q_left, q_right):
    physical = tuple(SpatialOrbitalSite().qn)
    channels = []
    for q_phys1 in physical:
        for q_mid in fuse_charge_spin_sectors(q_left, q_phys1):
            for q_phys2 in physical:
                if q_right in fuse_charge_spin_sectors(q_mid, q_phys2):
                    channels.append((q_phys1, q_mid, q_phys2))
    return tuple(channels)


@lru_cache(maxsize=1)
def _compiled_channel_kernels():
    try:
        from . import _su2_kernel
    except ImportError:
        return None, None
    return (
        getattr(_su2_kernel, "apply_orbital_channel_gate", None),
        getattr(_su2_kernel, "mix_orbital_channel_blocks", None),
    )


@lru_cache(maxsize=None)
def _channel_projection_tensor(q_left, q_right, input_channels, output_channels):
    inputs = np.stack(
        [
            _channel_component_tensor(q_left, *channel, q_right)
            for channel in input_channels
        ]
    )
    outputs = np.stack(
        [
            _channel_component_tensor(q_left, *channel, q_right)
            for channel in output_channels
        ]
    )
    return np.einsum(
        "olxyr,ilabr->oixyab",
        outputs.conj(),
        inputs,
        optimize=True,
    ) / q_right.irrep.dim


def _mix_channel_blocks(q_left, q_right, candidates, inputs, fock_gate):
    input_channels = tuple(
        (key[1], key[2], key[3]) for key, _block in inputs
    )
    projection = _channel_projection_tensor(
        q_left,
        q_right,
        input_channels,
        tuple(candidates),
    )
    tolerance = 64.0 * np.finfo(float).eps
    source_blocks = np.stack(
        [np.asarray(block, dtype=complex) for _key, block in inputs]
    )
    gate_kernel, mixer = _compiled_channel_kernels()
    if gate_kernel is not None:
        output_blocks, coefficients = gate_kernel(
            fock_gate,
            projection,
            source_blocks,
            tolerance,
        )
        used_native = True
    else:
        coefficients = np.einsum(
            "oixyab,xyab->oi",
            projection,
            fock_gate,
            optimize=True,
        )
        coefficients[np.abs(coefficients) <= tolerance] = 0.0
        if mixer is None:
            output_blocks = np.tensordot(
                coefficients,
                source_blocks,
                axes=([1], [0]),
            )
            used_native = False
        else:
            output_blocks = mixer(coefficients, source_blocks)
            used_native = True
    active = np.any(np.abs(coefficients) > tolerance, axis=1)
    if not np.any(active):
        return (), output_blocks[:0], used_native

    active_candidates = tuple(
        candidate for candidate, keep in zip(candidates, active) if keep
    )
    return active_candidates, output_blocks[active], used_native


def _apply_adjacent_gate(
    state,
    bond,
    one_particle_gate,
    *,
    cutoff,
    max_bond,
    max_truncation_error=None,
):
    merged = merge_mps_sites(state.sites[bond], state.sites[bond + 1])
    input_channels = merged.metadata["contracted_channel_blocks"]
    fock_gate = _second_quantized_two_orbital_gate(one_particle_gate)
    output_channels = {}
    native_mix_calls = 0
    mix_batches = 0

    boundary_pairs = sorted({(key[0], key[4]) for key in input_channels})
    for q_left, q_right in boundary_pairs:
        candidates = _allowed_output_channels(q_left, q_right)
        inputs = [
            (key, block)
            for key, block in input_channels.items()
            if key[0] == q_left and key[4] == q_right
        ]
        active_candidates, output_blocks, used_native = _mix_channel_blocks(
            q_left,
            q_right,
            candidates,
            inputs,
            fock_gate,
        )
        mix_batches += 1
        native_mix_calls += int(used_native)
        for (q_phys1, q_mid, q_phys2), output_block in zip(
            active_candidates,
            output_blocks,
        ):
            if np.any(output_block != 0):
                output_channels[(q_left, q_phys1, q_mid, q_phys2, q_right)] = output_block

    data = {}
    for (q_left, q_phys1, _q_mid, q_phys2, q_right), block in output_channels.items():
        key = (q_left, q_phys1, q_phys2, q_right)
        data[key] = block.copy() if key not in data else data[key] + block
    transformed = NonabelianTensor(
        data,
        merged.qns,
        merged.dirs,
        fusion_legs=merged.fusion_legs,
        metadata={
            **merged.metadata,
            "contracted_channel_blocks": output_channels,
            "contracted_channel_blocks_current": True,
        },
    )
    left, right, _singular, _error, _kept = svd_two_site(
        transformed,
        max_bond=max_bond,
        cutoff=cutoff,
        max_truncation_error=max_truncation_error,
        absorb="right",
    )
    state.sites[bond] = left
    state.sites[bond + 1] = right
    state.center = bond + 1
    return float(_error), native_mix_calls, mix_batches


def _apply_diagonal(state, diagonal):
    for site_index, value in enumerate(np.asarray(diagonal, dtype=complex)):
        if value == 1.0:
            continue
        site = state.sites[site_index]
        state.sites[site_index] = NonabelianTensor(
            {
                key: np.asarray(block) * value ** key[1].charge
                for key, block in site.data.items()
            },
            site.qns,
            site.dirs,
            fusion_legs=site.fusion_legs,
            metadata=site.metadata,
        )


def _adjacent_unitary_circuit(unitary, *, tol=None):
    """Return diagonal/adjacent-row operations in state-application order."""
    work = np.asarray(unitary, dtype=complex).copy()
    n = work.shape[0]
    if tol is None:
        tol = 64.0 * np.finfo(float).eps * max(1, n)
    eliminations = []
    for column in range(n - 1):
        for row in range(n - 1, column, -1):
            a = work[row - 1, column]
            b = work[row, column]
            if abs(b) <= tol:
                continue
            norm = np.hypot(abs(a), abs(b))
            c = a / norm
            s = b / norm
            givens = np.array(
                [[np.conj(c), np.conj(s)], [-s, c]],
                dtype=complex,
            )
            work[[row - 1, row], :] = givens @ work[[row - 1, row], :]
            eliminations.append((row - 1, givens))
    diagonal = np.diag(work).copy()
    residue = work - np.diag(diagonal)
    if np.max(np.abs(residue), initial=0.0) > 100 * tol:
        raise np.linalg.LinAlgError("Failed to reduce unitary orbital map to adjacent rotations.")
    return [("diagonal", diagonal)] + [
        ("gate", bond, givens.conj().T)
        for bond, givens in reversed(eliminations)
    ]


def _orbital_circuit(one_particle_map, *, return_info=False):
    matrix = np.asarray(one_particle_map, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Orbital map must be square, got {matrix.shape}.")
    size = matrix.shape[0]
    identity = np.eye(size, dtype=complex)
    unitarity_residual = float(
        np.linalg.norm(matrix.conj().T @ matrix - identity, ord=2)
    )
    scale = max(1.0, float(np.linalg.norm(matrix, ord=2)) ** 2)
    unitarity_tolerance = 256.0 * np.finfo(float).eps * max(1, size) * scale
    if unitarity_residual <= unitarity_tolerance:
        circuit = _adjacent_unitary_circuit(matrix)
        factorization = "unitary_givens"
    else:
        u, singular, vh = np.linalg.svd(matrix, full_matrices=False)
        singular_tolerance = (
            np.finfo(float).eps * size * np.max(singular, initial=0.0)
        )
        if singular.size and np.min(singular) <= singular_tolerance:
            raise np.linalg.LinAlgError("Orbital map is numerically singular.")
        circuit = (
            _adjacent_unitary_circuit(vh)
            + [("diagonal", singular.astype(complex))]
            + _adjacent_unitary_circuit(u)
        )
        factorization = "svd_givens"
    if not return_info:
        return circuit
    return circuit, {
        "factorization": factorization,
        "unitarity_residual": unitarity_residual,
        "unitarity_tolerance": unitarity_tolerance,
    }


def apply_spatial_orbital_transform(
    state,
    orbital_transform,
    *,
    inverse=True,
    orbital_blocks=None,
    cutoff=1.0e-10,
    max_bond="auto",
    discarded_weight_budget=1.0e-6,
    adaptive_max_bond=8192,
    return_info=False,
):
    """Apply a spin-independent orbital map to a reduced SU(2) MPS.

    ``inverse=True`` matches the coefficient transformation used by
    Malmqvist biorthogonalization: the induced Fock-space action is that of
    ``orbital_transform^{-1}``.  The practical defaults use ``cutoff=1e-10``
    and ``max_bond="auto"``.  The automatic cap is 16 times the input reduced
    bond dimension, with a floor of 256, a soft ceiling of 8192, and never below
    the input dimension. ``max_bond="adaptive"`` keeps the smallest reduced
    bond satisfying a cumulative ``discarded_weight_budget``, subject to
    ``adaptive_max_bond``. Use ``cutoff=0`` and ``max_bond=None`` for an
    untruncated result exact up to floating-point roundoff. ``orbital_blocks``
    may partition a block-diagonal map into contiguous intervals; each block
    then has its own Gaussian circuit, avoiding gates across empty graph cuts.
    """
    transformed = _as_reduced_mps(state)
    input_bond = max(
        [1] + [len(site.qns[2]) for site in transformed.sites[:-1]]
    )
    requested_max_bond = max_bond
    adaptive = False
    if isinstance(max_bond, str):
        mode = max_bond.lower()
        if mode == "auto":
            max_bond = max(input_bond, min(8192, max(256, 16 * input_bond)))
        elif mode == "adaptive":
            adaptive = True
            adaptive_max_bond = int(adaptive_max_bond)
            if adaptive_max_bond < input_bond:
                raise ValueError(
                    "adaptive_max_bond cannot be smaller than the input bond dimension."
                )
            max_bond = adaptive_max_bond
        else:
            raise ValueError(
                "max_bond must be a positive integer, None, 'auto', or 'adaptive'."
            )
    elif max_bond is not None:
        max_bond = int(max_bond)
        if max_bond < 1:
            raise ValueError("max_bond must be positive when specified.")
    cutoff = float(cutoff)
    if cutoff < 0.0:
        raise ValueError("cutoff must be non-negative.")
    discarded_weight_budget = float(discarded_weight_budget)
    if not 0.0 <= discarded_weight_budget < 1.0:
        raise ValueError("discarded_weight_budget must lie in [0, 1).")
    transform = np.asarray(orbital_transform, dtype=complex)
    if transform.shape != (len(transformed), len(transformed)):
        raise ValueError(
            f"Orbital transform shape {transform.shape} does not match "
            f"the {len(transformed)}-site MPS."
        )
    one_particle_map = np.linalg.inv(transform) if inverse else transform
    if orbital_blocks is None:
        circuit, factor_info = _orbital_circuit(
            one_particle_map,
            return_info=True,
        )
        factor_infos = [factor_info]
        block_count = 1
    else:
        blocks = [tuple(int(index) for index in block) for block in orbital_blocks]
        flat = [index for block in blocks for index in block]
        if sorted(flat) != list(range(len(transformed))) or any(not block for block in blocks):
            raise ValueError("orbital_blocks must partition every orbital exactly once.")
        if any(
            block != tuple(range(block[0], block[-1] + 1))
            for block in blocks
        ):
            raise ValueError("orbital_blocks must be contiguous in the current MPS order.")
        block_labels = np.empty(len(transformed), dtype=int)
        for label, block in enumerate(blocks):
            block_labels[np.asarray(block, dtype=int)] = label
        off_block = block_labels[:, None] != block_labels[None, :]
        if np.max(np.abs(one_particle_map[off_block]), initial=0.0) > 1.0e-13:
            raise ValueError("orbital map contains nonzero couplings between orbital_blocks.")
        circuit = []
        factor_infos = []
        for block in blocks:
            local, factor_info = _orbital_circuit(
                one_particle_map[np.ix_(block, block)],
                return_info=True,
            )
            factor_infos.append(factor_info)
            for kind, *payload in local:
                if kind == "diagonal":
                    diagonal = np.ones(len(transformed), dtype=complex)
                    diagonal[np.asarray(block, dtype=int)] = payload[0]
                    circuit.append((kind, diagonal))
                else:
                    bond, gate = payload
                    circuit.append((kind, block[0] + int(bond), gate))
        block_count = len(blocks)
    factorizations = {info["factorization"] for info in factor_infos}
    factorization = (
        next(iter(factorizations)) if len(factorizations) == 1 else "mixed"
    )
    gate_count = 0
    native_mix_calls = 0
    mix_batches = 0
    peak_reduced_bond = 1
    truncation_errors = []
    gate_total = sum(kind == "gate" for kind, *_payload in circuit)
    gate_bonds = []
    gate_budgets = []
    gate_kept_bonds = []
    for kind, *payload in circuit:
        if kind == "diagonal":
            _apply_diagonal(transformed, payload[0])
        else:
            bond, gate = payload
            remaining_gates = gate_total - gate_count
            remaining_budget = max(
                0.0,
                discarded_weight_budget - sum(truncation_errors),
            )
            gate_budget = (
                remaining_budget / remaining_gates if adaptive else None
            )
            error, gate_native_mix_calls, gate_mix_batches = _apply_adjacent_gate(
                transformed,
                bond,
                gate,
                cutoff=cutoff,
                max_bond=max_bond,
                max_truncation_error=gate_budget,
            )
            truncation_errors.append(error)
            gate_bonds.append(int(bond))
            gate_budgets.append(gate_budget)
            gate_kept_bonds.append(len(transformed.sites[bond].qns[2]))
            native_mix_calls += gate_native_mix_calls
            mix_batches += gate_mix_batches
            gate_count += 1
            peak_reduced_bond = max(
                peak_reduced_bond,
                *(len(site.qns[2]) for site in transformed.sites[:-1]),
            )
    if not return_info:
        return transformed
    return transformed, {
        "method": "sector_preserving_adjacent_gaussian_circuit",
        "exact": cutoff == 0.0 and max_bond is None,
        "cutoff": cutoff,
        "max_bond": max_bond,
        "requested_max_bond": requested_max_bond,
        "adaptive": adaptive,
        "discarded_weight_budget": (
            discarded_weight_budget if adaptive else None
        ),
        "adaptive_max_bond": adaptive_max_bond if adaptive else None,
        "input_reduced_bond_dimension": input_bond,
        "adjacent_gate_count": gate_count,
        "orbital_block_count": block_count,
        "orbital_factorization": factorization,
        "channel_mix_backend": (
            "none"
            if mix_batches == 0
            else "compiled"
            if native_mix_calls == mix_batches
            else "numpy"
            if native_mix_calls == 0
            else "mixed"
        ),
        "channel_mix_batches": mix_batches,
        "compiled_channel_mix_batches": native_mix_calls,
        "orbital_block_factorizations": factor_infos,
        "unitarity_residual": max(
            info["unitarity_residual"] for info in factor_infos
        ),
        "peak_reduced_bond_dimension": peak_reduced_bond,
        "sum_gate_discarded_weight": float(sum(truncation_errors)),
        "max_gate_discarded_weight": float(max(truncation_errors, default=0.0)),
        "truncated_gate_count": sum(error > 0.0 for error in truncation_errors),
        "adaptive_budget_satisfied": (
            float(sum(truncation_errors)) <= discarded_weight_budget
            if adaptive
            else None
        ),
        "gate_bonds": gate_bonds,
        "gate_discarded_weight_budgets": gate_budgets,
        "gate_kept_reduced_bonds": gate_kept_bonds,
        "determinant_expansion": False,
        "component_expansion": False,
    }
