"""Exact finite-tier contractions for one-dimensional memory cLETTA states.

The memory construction is a structured cMPS on ``virtual x memory`` space,

``Q_c = Q x I - kappa I x N``

``R_c = R x I + sqrt(kappa) I x a + S x a^dagger``.

Expanding a double-layer environment in memory Fock blocks ``X[m, n]`` gives
the two-sided cLETTA hierarchy implemented here.  This differs from a standard
open-system HEOM: both ket and bra memory occupations are retained, and the
physical field insertion is the full matrix ``R_c``.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "apply_cletta_bra_insertion",
    "apply_cletta_ket_insertion",
    "apply_cletta_memory_hierarchy",
    "apply_cletta_multimode_bra_insertion",
    "apply_cletta_multimode_ket_insertion",
    "apply_cletta_multimode_memory_hierarchy",
    "apply_cletta_multimode_memory_hierarchy_adjoint",
    "cletta_bra_insertion_matrix",
    "cletta_ket_insertion_matrix",
    "cletta_memory_fock_keys",
    "cletta_memory_hierarchy_generator",
    "cletta_memory_matrices",
    "cletta_multimode_bra_insertion_matrix",
    "cletta_multimode_hierarchy_generator",
    "cletta_multimode_hierarchy_sparse_generator",
    "cletta_multimode_ket_insertion_matrix",
    "cletta_multimode_memory_matrices",
    "cletta_multifield_memory_matrices",
    "hierarchy_blocks_to_matrix",
    "matrix_to_hierarchy_blocks",
]


def _validated_parameters(q, r, s, kappa, memory_dim):
    q = np.asarray(q)
    r = np.asarray(r)
    s = np.asarray(s)
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if r.shape != q.shape or s.shape != q.shape:
        raise ValueError("R and S must have the same square shape as Q.")
    kappa = float(kappa)
    if not np.isfinite(kappa) or kappa <= 0.0:
        raise ValueError("kappa must be finite and positive.")
    memory_dim = int(memory_dim)
    if memory_dim < 1:
        raise ValueError("memory_dim must be positive.")
    dtype = np.result_type(q.dtype, r.dtype, s.dtype, np.complex128)
    return (
        np.asarray(q, dtype=dtype),
        np.asarray(r, dtype=dtype),
        np.asarray(s, dtype=dtype),
        kappa,
        memory_dim,
        dtype,
    )


def _memory_operators(memory_dim, dtype):
    annihilation = np.zeros((memory_dim, memory_dim), dtype=dtype)
    for occupation in range(1, memory_dim):
        annihilation[occupation - 1, occupation] = np.sqrt(float(occupation))
    number = np.diag(np.arange(memory_dim, dtype=float)).astype(dtype)
    return annihilation, number


def cletta_memory_matrices(q, r, s, kappa, *, memory_dim=2):
    r"""Return the explicit finite-memory cLETTA matrices ``(Q_c, R_c)``.

    ``memory_dim=2`` is the one-open-leg construction used by the existing
    Lieb-Liniger cLETTA example.  Larger values allow overlapping open ties.
    """
    q, r, s, kappa, memory_dim, dtype = _validated_parameters(q, r, s, kappa, memory_dim)
    dim = q.shape[0]
    eye_memory = np.eye(memory_dim, dtype=dtype)
    eye_virtual = np.eye(dim, dtype=dtype)
    annihilation, number = _memory_operators(memory_dim, dtype)
    q_memory = np.kron(eye_memory, q) - kappa * np.kron(number, eye_virtual)
    r_memory = (
        np.kron(eye_memory, r)
        + np.sqrt(kappa) * np.kron(annihilation, eye_virtual)
        + np.kron(annihilation.conj().T, s)
    )
    return q_memory, r_memory


def _fixed_total_occupations(total, num_modes):
    if num_modes == 1:
        yield (int(total),)
        return
    for occupation in range(int(total) + 1):
        for tail in _fixed_total_occupations(int(total) - occupation, int(num_modes) - 1):
            yield (occupation, *tail)


def cletta_memory_fock_keys(num_modes, depth):
    r"""Return total-occupation pseudomode keys through hierarchy depth ``L``.

    A key ``(n_1, ..., n_M)`` represents the number of simultaneously open
    ties in each memory channel.  ``depth=L`` retains all keys satisfying

    $$
    \sum_{\nu=1}^M n_\nu \leq L.
    $$
    """
    num_modes = int(num_modes)
    depth = int(depth)
    if num_modes < 1:
        raise ValueError("num_modes must be positive.")
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    keys = [
        key
        for total in range(depth + 1)
        for key in _fixed_total_occupations(total, num_modes)
    ]
    return np.asarray(keys, dtype=np.int64)


def _validated_multimode_parameters(q, r, tie_matrices, decay_rates, depth, frequencies=None):
    q = np.asarray(q)
    r = np.asarray(r)
    ties = np.asarray(tie_matrices)
    rates = np.atleast_1d(np.asarray(decay_rates, dtype=float))
    if q.ndim != 2 or q.shape[0] != q.shape[1]:
        raise ValueError("Q must be a square matrix.")
    if r.shape != q.shape:
        raise ValueError("R must have the same square shape as Q.")
    if ties.ndim != 3 or ties.shape[1:] != q.shape:
        raise ValueError("tie_matrices must have shape (num_modes, bond_dim, bond_dim).")
    if ties.shape[0] != rates.size:
        raise ValueError("tie_matrices and decay_rates must contain the same number of channels.")
    if rates.size < 1:
        raise ValueError("at least one memory channel is required.")
    if np.any(~np.isfinite(rates)) or np.any(rates <= 0.0):
        raise ValueError("all decay_rates must be finite and positive.")
    if frequencies is None:
        frequencies = np.zeros(rates.size, dtype=float)
    else:
        frequencies = np.atleast_1d(np.asarray(frequencies, dtype=float))
        if frequencies.shape != rates.shape:
            raise ValueError("frequencies and decay_rates must contain the same number of channels.")
        if np.any(~np.isfinite(frequencies)):
            raise ValueError("all frequencies must be finite.")
    depth = int(depth)
    if depth < 0:
        raise ValueError("depth must be non-negative.")
    dtype = np.result_type(q.dtype, r.dtype, ties.dtype, np.complex128)
    return (
        np.asarray(q, dtype=dtype),
        np.asarray(r, dtype=dtype),
        np.asarray(ties, dtype=dtype),
        rates,
        frequencies,
        depth,
        dtype,
    )


def _multimode_memory_operators(num_modes, depth, dtype):
    keys = cletta_memory_fock_keys(num_modes, depth)
    key_to_index = {tuple(key): index for index, key in enumerate(keys)}
    memory_dim = len(keys)
    annihilation = np.zeros((num_modes, memory_dim, memory_dim), dtype=dtype)
    number = np.zeros((num_modes, memory_dim, memory_dim), dtype=dtype)
    for column, key in enumerate(keys):
        for mode in range(num_modes):
            number[mode, column, column] = float(key[mode])
            if key[mode] == 0:
                continue
            lower = key.copy()
            lower[mode] -= 1
            row = key_to_index[tuple(lower)]
            annihilation[mode, row, column] = np.sqrt(float(key[mode]))
    return keys, key_to_index, annihilation, number


def _tensor_product_memory_operators(num_modes, depth, dtype):
    """Return commuting per-mode Fock operators with cutoff ``depth``."""
    local_dim = int(depth) + 1
    local_annihilation, local_number = _memory_operators(local_dim, dtype)
    local_identity = np.eye(local_dim, dtype=dtype)
    memory_dim = local_dim**int(num_modes)
    annihilation = np.zeros((num_modes, memory_dim, memory_dim), dtype=dtype)
    number = np.zeros_like(annihilation)
    for mode in range(num_modes):
        a_mode = np.array([[1.0]], dtype=dtype)
        n_mode = np.array([[1.0]], dtype=dtype)
        for factor in range(num_modes):
            a_mode = np.kron(
                a_mode,
                local_annihilation if factor == mode else local_identity,
            )
            n_mode = np.kron(
                n_mode,
                local_number if factor == mode else local_identity,
            )
        annihilation[mode] = a_mode
        number[mode] = n_mode
    return annihilation, number


def cletta_multimode_memory_matrices(
    q, r, tie_matrices, decay_rates, *, depth=1, frequencies=None
):
    r"""Return explicit finite-depth matrices for an ``M``-channel cLETTA.

    With total-occupation memory operators ``a_nu`` and ``N_nu``, the
    structured cMPS matrices are

    $$
    Q_c = I\otimes Q - \sum_\nu \gamma_\nu N_\nu\otimes I,
    $$

    $$
    R_c = I\otimes R + \sum_\nu\left(
      \sqrt{\gamma_\nu}a_\nu\otimes I +
      a_\nu^\dagger\otimes G_\nu
    \right).
    $$

    ``depth`` is the maximum total number of open ties in the ket auxiliary
    layer.  The double-layer hierarchy retains the corresponding ket and bra
    keys independently.
    """
    q, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    keys, _, annihilation, number = _multimode_memory_operators(len(rates), depth, dtype)
    memory_dim = len(keys)
    bond_dim = q.shape[0]
    eye_memory = np.eye(memory_dim, dtype=dtype)
    eye_virtual = np.eye(bond_dim, dtype=dtype)
    q_memory = np.kron(eye_memory, q)
    r_memory = np.kron(eye_memory, r)
    for mode, rate in enumerate(rates):
        q_memory -= (rate + 1.0j * frequencies[mode]) * np.kron(number[mode], eye_virtual)
        r_memory += np.sqrt(rate) * np.kron(annihilation[mode], eye_virtual)
        r_memory += np.kron(annihilation[mode].conj().T, ties[mode])
    return q_memory, r_memory


def cletta_multifield_memory_matrices(
    q,
    r_ops,
    tie_matrices,
    decay_rates,
    *,
    field=0,
    field_couplings=None,
    depth=1,
    frequencies=None,
):
    r"""Attach exponential-memory channels to a multifield cMPS.

    ``field_couplings[nu, i]`` controls how memory channel ``nu`` enters field
    ``i``.  Omitting it reproduces the one-hot coupling to ``field``.  A
    tensor-product Fock cutoff keeps distinct memory operators commuting.  The
    resulting multicomponent cMPS preserves bosonic regularity when the base
    fields and tie matrices commute.
    """
    ops = tuple(np.asarray(r) for r in r_ops)
    if not ops:
        raise ValueError("r_ops must contain at least one field matrix.")
    q, _r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        q,
        ops[0],
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    num_modes = rates.size
    if field_couplings is None:
        field = int(field)
        if field < 0 or field >= len(ops):
            raise ValueError("field index is out of range.")
        couplings = np.zeros((num_modes, len(ops)), dtype=float)
        couplings[:, field] = 1.0
    else:
        couplings = np.asarray(field_couplings, dtype=float)
        if couplings.shape != (num_modes, len(ops)):
            raise ValueError("field_couplings must have shape (num_modes, num_fields).")
        if np.any(~np.isfinite(couplings)):
            raise ValueError("field_couplings must be finite.")

    annihilation, number = _tensor_product_memory_operators(
        num_modes, depth, dtype
    )
    memory_dim = annihilation.shape[1]
    bond_dim = q.shape[0]
    eye_memory = np.eye(memory_dim, dtype=dtype)
    eye_virtual = np.eye(bond_dim, dtype=dtype)
    q_memory = np.kron(eye_memory, q)
    lifted = [np.kron(eye_memory, np.asarray(r, dtype=dtype)) for r in ops]
    for mode, rate in enumerate(rates):
        q_memory -= (rate + 1.0j * frequencies[mode]) * np.kron(
            number[mode], eye_virtual
        )
        memory_field = (
            np.sqrt(rate) * np.kron(annihilation[mode], eye_virtual)
            + np.kron(annihilation[mode].conj().T, ties[mode])
        )
        for field_index, coefficient in enumerate(couplings[mode]):
            lifted[field_index] += coefficient * memory_field
    return q_memory, tuple(lifted)


def _multimode_target_transitions(r, ties, rates, keys, key_to_index, dtype):
    eye = np.eye(r.shape[0], dtype=dtype)
    transitions = []
    for target_index, key in enumerate(keys):
        target = [(target_index, r)]
        for mode, rate in enumerate(rates):
            upper = key.copy()
            upper[mode] += 1
            source_index = key_to_index.get(tuple(upper))
            if source_index is not None:
                coefficient = np.sqrt(rate * float(key[mode] + 1))
                target.append((source_index, coefficient * eye))
            if key[mode] > 0:
                lower = key.copy()
                lower[mode] -= 1
                source_index = key_to_index[tuple(lower)]
                target.append((source_index, np.sqrt(float(key[mode])) * ties[mode]))
        transitions.append(target)
    return transitions


def _validated_multimode_blocks(
    blocks, q, r, tie_matrices, decay_rates, depth, frequencies=None
):
    q, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    keys, key_to_index, _, _ = _multimode_memory_operators(len(rates), depth, dtype)
    blocks = np.asarray(blocks, dtype=dtype)
    expected = (len(keys), len(keys), q.shape[0], q.shape[0])
    if blocks.shape != expected:
        raise ValueError(f"blocks must have shape {expected} for the requested depth.")
    transitions = _multimode_target_transitions(r, ties, rates, keys, key_to_index, dtype)
    return blocks, q, r, ties, rates, frequencies, keys, transitions, dtype


def apply_cletta_multimode_memory_hierarchy(
    blocks,
    q,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Apply the exact two-sided, multi-channel cLETTA hierarchy.

    This is a double-layer Fock hierarchy, not a standard one-sided
    open-system HEOM: both ket and bra open-tie occupations are retained.
    """
    blocks, q, _, _, rates, frequencies, keys, transitions, dtype = _validated_multimode_blocks(
        blocks,
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    out = np.zeros_like(blocks, dtype=dtype)
    decay_ket = np.asarray(keys, dtype=float) @ (
        np.asarray(rates, dtype=float) + 1.0j * np.asarray(frequencies, dtype=float)
    )
    decay_bra = np.asarray(keys, dtype=float) @ (
        np.asarray(rates, dtype=float) - 1.0j * np.asarray(frequencies, dtype=float)
    )
    for ket_target, ket_transitions in enumerate(transitions):
        for bra_target, bra_transitions in enumerate(transitions):
            value = q @ blocks[ket_target, bra_target]
            value += blocks[ket_target, bra_target] @ q.conj().T
            value -= (
                decay_ket[ket_target] + decay_bra[bra_target]
            ) * blocks[ket_target, bra_target]
            for ket_source, ket_operator in ket_transitions:
                for bra_source, bra_operator in bra_transitions:
                    value += (
                        ket_operator
                        @ blocks[ket_source, bra_source]
                        @ bra_operator.conj().T
                    )
            out[ket_target, bra_target] = value
    return out


def apply_cletta_multimode_memory_hierarchy_adjoint(
    blocks,
    q,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Apply the Frobenius adjoint of the exact two-sided hierarchy.

    This action is the ``rmatvec`` companion of
    :func:`apply_cletta_multimode_memory_hierarchy` and avoids materializing
    the dense double-layer generator for non-Hermitian Arnoldi iterations.
    """
    blocks, q, _, _, rates, frequencies, keys, transitions, dtype = _validated_multimode_blocks(
        blocks,
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    out = np.zeros_like(blocks, dtype=dtype)
    decay_ket = np.asarray(keys, dtype=float) @ (
        np.asarray(rates, dtype=float) + 1.0j * np.asarray(frequencies, dtype=float)
    )
    decay_bra = np.asarray(keys, dtype=float) @ (
        np.asarray(rates, dtype=float) - 1.0j * np.asarray(frequencies, dtype=float)
    )
    for ket_target, ket_transitions in enumerate(transitions):
        for bra_target, bra_transitions in enumerate(transitions):
            source = blocks[ket_target, bra_target]
            out[ket_target, bra_target] += q.conj().T @ source + source @ q
            out[ket_target, bra_target] -= np.conj(
                decay_ket[ket_target] + decay_bra[bra_target]
            ) * source
            for ket_source, ket_operator in ket_transitions:
                for bra_source, bra_operator in bra_transitions:
                    out[ket_source, bra_source] += (
                        ket_operator.conj().T @ source @ bra_operator
                    )
    return out


def apply_cletta_multimode_ket_insertion(
    blocks,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Apply ``R_c X`` to a multi-channel double-layer hierarchy."""
    zeros = np.zeros_like(np.asarray(r))
    blocks, _, _, _, _, _, _, transitions, dtype = _validated_multimode_blocks(
        blocks,
        zeros,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    out = np.zeros_like(blocks, dtype=dtype)
    for ket_target, ket_transitions in enumerate(transitions):
        for bra_index in range(blocks.shape[1]):
            for ket_source, ket_operator in ket_transitions:
                out[ket_target, bra_index] += ket_operator @ blocks[ket_source, bra_index]
    return out


def apply_cletta_multimode_bra_insertion(
    blocks,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Apply ``X R_c^dagger`` to a multi-channel double-layer hierarchy."""
    zeros = np.zeros_like(np.asarray(r))
    blocks, _, _, _, _, _, _, transitions, dtype = _validated_multimode_blocks(
        blocks,
        zeros,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    out = np.zeros_like(blocks, dtype=dtype)
    for bra_target, bra_transitions in enumerate(transitions):
        for ket_index in range(blocks.shape[0]):
            for bra_source, bra_operator in bra_transitions:
                out[ket_index, bra_target] += blocks[ket_index, bra_source] @ bra_operator.conj().T
    return out


def _validated_blocks(blocks, dim):
    blocks = np.asarray(blocks)
    if blocks.ndim != 4 or blocks.shape[0] != blocks.shape[1] or blocks.shape[2:] != (dim, dim):
        raise ValueError("blocks must have shape (memory_dim, memory_dim, bond_dim, bond_dim).")
    return blocks


def apply_cletta_memory_hierarchy(blocks, q, r, s, kappa):
    r"""Apply the exact two-sided cLETTA hierarchy to ``X[m, n]`` blocks."""
    memory_dim = int(np.asarray(blocks).shape[0])
    q, r, s, kappa, memory_dim, dtype = _validated_parameters(q, r, s, kappa, memory_dim)
    blocks = np.asarray(_validated_blocks(blocks, q.shape[0]), dtype=dtype)
    dim = q.shape[0]
    close = np.sqrt(kappa) * np.eye(dim, dtype=dtype)
    out = np.zeros_like(blocks, dtype=dtype)

    for m in range(memory_dim):
        for n in range(memory_dim):
            x = blocks[m, n]
            value = q @ x + x @ q.conj().T + r @ x @ r.conj().T
            value -= kappa * float(m + n) * x

            if m + 1 < memory_dim:
                root_m_up = np.sqrt(float(m + 1))
                value += root_m_up * close @ blocks[m + 1, n] @ r.conj().T
                if n + 1 < memory_dim:
                    value += (
                        root_m_up
                        * np.sqrt(float(n + 1))
                        * close
                        @ blocks[m + 1, n + 1]
                        @ close.conj().T
                    )
                if n > 0:
                    value += (
                        root_m_up
                        * np.sqrt(float(n))
                        * close
                        @ blocks[m + 1, n - 1]
                        @ s.conj().T
                    )

            if m > 0:
                root_m_down = np.sqrt(float(m))
                value += root_m_down * s @ blocks[m - 1, n] @ r.conj().T
                if n + 1 < memory_dim:
                    value += (
                        root_m_down
                        * np.sqrt(float(n + 1))
                        * s
                        @ blocks[m - 1, n + 1]
                        @ close.conj().T
                    )
                if n > 0:
                    value += (
                        root_m_down
                        * np.sqrt(float(n))
                        * s
                        @ blocks[m - 1, n - 1]
                        @ s.conj().T
                    )

            if n + 1 < memory_dim:
                value += np.sqrt(float(n + 1)) * r @ blocks[m, n + 1] @ close.conj().T
            if n > 0:
                value += np.sqrt(float(n)) * r @ blocks[m, n - 1] @ s.conj().T
            out[m, n] = value
    return out


def apply_cletta_ket_insertion(blocks, r, s, kappa):
    r"""Apply ``R_c X`` to a two-sided cLETTA hierarchy."""
    r = np.asarray(r)
    s = np.asarray(s)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or s.shape != r.shape:
        raise ValueError("R and S must be square matrices with the same shape.")
    memory_dim = int(np.asarray(blocks).shape[0])
    blocks = _validated_blocks(blocks, r.shape[0])
    dtype = np.result_type(blocks.dtype, r.dtype, s.dtype, np.complex128)
    blocks = np.asarray(blocks, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    s = np.asarray(s, dtype=dtype)
    close = np.sqrt(float(kappa)) * np.eye(r.shape[0], dtype=dtype)
    out = np.zeros_like(blocks, dtype=dtype)
    for m in range(memory_dim):
        for n in range(memory_dim):
            value = r @ blocks[m, n]
            if m + 1 < memory_dim:
                value += np.sqrt(float(m + 1)) * close @ blocks[m + 1, n]
            if m > 0:
                value += np.sqrt(float(m)) * s @ blocks[m - 1, n]
            out[m, n] = value
    return out


def apply_cletta_bra_insertion(blocks, r, s, kappa):
    r"""Apply ``X R_c^dagger`` to a two-sided cLETTA hierarchy."""
    r = np.asarray(r)
    s = np.asarray(s)
    if r.ndim != 2 or r.shape[0] != r.shape[1] or s.shape != r.shape:
        raise ValueError("R and S must be square matrices with the same shape.")
    memory_dim = int(np.asarray(blocks).shape[0])
    blocks = _validated_blocks(blocks, r.shape[0])
    dtype = np.result_type(blocks.dtype, r.dtype, s.dtype, np.complex128)
    blocks = np.asarray(blocks, dtype=dtype)
    r = np.asarray(r, dtype=dtype)
    s = np.asarray(s, dtype=dtype)
    close = np.sqrt(float(kappa)) * np.eye(r.shape[0], dtype=dtype)
    out = np.zeros_like(blocks, dtype=dtype)
    for m in range(memory_dim):
        for n in range(memory_dim):
            value = blocks[m, n] @ r.conj().T
            if n + 1 < memory_dim:
                value += np.sqrt(float(n + 1)) * blocks[m, n + 1] @ close.conj().T
            if n > 0:
                value += np.sqrt(float(n)) * blocks[m, n - 1] @ s.conj().T
            out[m, n] = value
    return out


def _linear_map_matrix(shape, action, dtype):
    size = int(np.prod(shape))
    matrix = np.zeros((size, size), dtype=dtype)
    for column in range(size):
        basis = np.zeros(shape, dtype=dtype)
        basis.reshape(-1)[column] = 1.0
        matrix[:, column] = np.asarray(action(basis), dtype=dtype).reshape(-1)
    return matrix


def cletta_memory_hierarchy_generator(q, r, s, kappa, *, memory_dim=2):
    r"""Return the dense generator for the exact ``(m, n)`` hierarchy."""
    q, r, s, kappa, memory_dim, dtype = _validated_parameters(q, r, s, kappa, memory_dim)
    shape = (memory_dim, memory_dim, q.shape[0], q.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_memory_hierarchy(blocks, q, r, s, kappa),
        dtype,
    )


def cletta_ket_insertion_matrix(r, s, kappa, *, memory_dim=2):
    r"""Return the hierarchy-space matrix representing ``R_c X``."""
    r = np.asarray(r)
    s = np.asarray(s)
    dtype = np.result_type(r.dtype, s.dtype, np.complex128)
    shape = (int(memory_dim), int(memory_dim), r.shape[0], r.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_ket_insertion(blocks, r, s, kappa),
        dtype,
    )


def cletta_bra_insertion_matrix(r, s, kappa, *, memory_dim=2):
    r"""Return the hierarchy-space matrix representing ``X R_c^dagger``."""
    r = np.asarray(r)
    s = np.asarray(s)
    dtype = np.result_type(r.dtype, s.dtype, np.complex128)
    shape = (int(memory_dim), int(memory_dim), r.shape[0], r.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_bra_insertion(blocks, r, s, kappa),
        dtype,
    )


def cletta_multimode_hierarchy_generator(
    q,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Return the dense generator of the multi-channel double hierarchy."""
    q, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    memory_dim = len(cletta_memory_fock_keys(len(rates), depth))
    shape = (memory_dim, memory_dim, q.shape[0], q.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_multimode_memory_hierarchy(
            blocks,
            q,
            r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        ),
        dtype,
    )


def cletta_multimode_hierarchy_sparse_generator(
    q,
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Return the exact double-hierarchy generator as a sparse block matrix."""
    from scipy.sparse import coo_matrix

    q, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        q,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    keys, key_to_index, _, _ = _multimode_memory_operators(len(rates), depth, dtype)
    transitions = _multimode_target_transitions(r, ties, rates, keys, key_to_index, dtype)
    memory_dim = len(keys)
    bond_dim = q.shape[0]
    block_size = bond_dim**2
    size = memory_dim**2 * block_size
    eye = np.eye(bond_dim, dtype=dtype)
    identity_block = np.eye(block_size, dtype=dtype)
    drift_block = np.kron(q, eye) + np.kron(eye, q.conj())
    decay_ket = np.asarray(keys, dtype=float) @ (rates + 1.0j * frequencies)
    decay_bra = np.asarray(keys, dtype=float) @ (rates - 1.0j * frequencies)
    rows = []
    columns = []
    entries = []

    def append_block(target, source, block):
        block = np.asarray(block)
        row, column = np.nonzero(block)
        rows.extend((target * block_size + row).tolist())
        columns.extend((source * block_size + column).tolist())
        entries.extend(block[row, column].tolist())

    for ket_target, ket_transitions in enumerate(transitions):
        for bra_target, bra_transitions in enumerate(transitions):
            target = ket_target * memory_dim + bra_target
            decay = decay_ket[ket_target] + decay_bra[bra_target]
            append_block(target, target, drift_block - decay * identity_block)
            for ket_source, ket_operator in ket_transitions:
                for bra_source, bra_operator in bra_transitions:
                    source = ket_source * memory_dim + bra_source
                    append_block(
                        target,
                        source,
                        np.kron(ket_operator, bra_operator.conj()),
                    )
    return coo_matrix(
        (np.asarray(entries, dtype=dtype), (rows, columns)),
        shape=(size, size),
        dtype=dtype,
    ).tocsr()


def cletta_multimode_ket_insertion_matrix(
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Return the hierarchy-space matrix representing ``R_c X``."""
    zeros = np.zeros_like(np.asarray(r))
    _, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        zeros,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    memory_dim = len(cletta_memory_fock_keys(len(rates), depth))
    shape = (memory_dim, memory_dim, r.shape[0], r.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_multimode_ket_insertion(
            blocks,
            r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        ),
        dtype,
    )


def cletta_multimode_bra_insertion_matrix(
    r,
    tie_matrices,
    decay_rates,
    *,
    depth=1,
    frequencies=None,
):
    r"""Return the hierarchy-space matrix representing ``X R_c^dagger``."""
    zeros = np.zeros_like(np.asarray(r))
    _, r, ties, rates, frequencies, depth, dtype = _validated_multimode_parameters(
        zeros,
        r,
        tie_matrices,
        decay_rates,
        depth,
        frequencies,
    )
    memory_dim = len(cletta_memory_fock_keys(len(rates), depth))
    shape = (memory_dim, memory_dim, r.shape[0], r.shape[0])
    return _linear_map_matrix(
        shape,
        lambda blocks: apply_cletta_multimode_bra_insertion(
            blocks,
            r,
            ties,
            rates,
            depth=depth,
            frequencies=frequencies,
        ),
        dtype,
    )


def hierarchy_blocks_to_matrix(blocks):
    """Convert ``X[m, n, i, j]`` blocks to the explicit memory matrix."""
    blocks = np.asarray(blocks)
    if blocks.ndim != 4 or blocks.shape[0] != blocks.shape[1] or blocks.shape[2] != blocks.shape[3]:
        raise ValueError("blocks must have shape (memory_dim, memory_dim, bond_dim, bond_dim).")
    memory_dim, _, dim, _ = blocks.shape
    return blocks.transpose(0, 2, 1, 3).reshape(memory_dim * dim, memory_dim * dim)


def matrix_to_hierarchy_blocks(matrix, *, bond_dim, memory_dim):
    """Convert an explicit memory matrix to ``X[m, n, i, j]`` blocks."""
    matrix = np.asarray(matrix)
    bond_dim = int(bond_dim)
    memory_dim = int(memory_dim)
    expected = memory_dim * bond_dim
    if matrix.shape != (expected, expected):
        raise ValueError("matrix shape does not match bond_dim * memory_dim.")
    return matrix.reshape(memory_dim, bond_dim, memory_dim, bond_dim).transpose(0, 2, 1, 3)
