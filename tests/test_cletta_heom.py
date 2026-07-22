import numpy as np
from scipy.linalg import expm

from pyqed.mps import (
    ContinuousMPS,
    apply_cletta_bra_insertion,
    apply_cletta_ket_insertion,
    apply_cletta_memory_hierarchy,
    apply_cletta_multimode_bra_insertion,
    apply_cletta_multimode_ket_insertion,
    apply_cletta_multimode_memory_hierarchy,
    apply_cletta_multimode_memory_hierarchy_adjoint,
    cletta_bra_insertion_matrix,
    cletta_ket_insertion_matrix,
    cletta_memory_fock_keys,
    cletta_memory_hierarchy_generator,
    cletta_memory_matrices,
    cletta_multimode_bra_insertion_matrix,
    cletta_multimode_hierarchy_generator,
    cletta_multimode_hierarchy_sparse_generator,
    cletta_multimode_ket_insertion_matrix,
    cletta_multimode_memory_matrices,
    hierarchy_blocks_to_matrix,
)


def _explicit_transfer(q, r):
    dim = q.shape[0]
    eye = np.eye(dim, dtype=np.result_type(q, r, np.complex128))
    return np.kron(q, eye) + np.kron(eye, q.conj()) + np.kron(r, r.conj())


def _hierarchy_to_explicit_permutation(bond_dim, memory_dim):
    size = (int(bond_dim) * int(memory_dim)) ** 2
    permutation = np.zeros((size, size), dtype=np.complex128)
    shape = (memory_dim, memory_dim, bond_dim, bond_dim)
    for column in range(size):
        blocks = np.zeros(shape, dtype=np.complex128)
        blocks.reshape(-1)[column] = 1.0
        permutation[:, column] = hierarchy_blocks_to_matrix(blocks).reshape(-1)
    return permutation


def _dominant_environment(generator):
    values, right_vectors = np.linalg.eig(generator)
    index = int(np.argmax(np.real(values)))
    value = values[index]
    right = right_vectors[:, index]

    left_values, left_vectors = np.linalg.eig(generator.conj().T)
    left_index = int(np.argmin(np.abs(left_values - value.conj())))
    left = left_vectors[:, left_index]
    overlap = np.vdot(left, right)
    if abs(overlap) < 1.0e-12:
        raise FloatingPointError("dominant cLETTA environments are nearly orthogonal.")
    return value, left, right / overlap


def _stationary_field_correlation(generator, ket_insertion, bra_insertion, distances):
    value, left, right = _dominant_environment(generator)
    shifted = generator - value * np.eye(generator.shape[0], dtype=generator.dtype)
    initial = ket_insertion @ right
    return np.asarray(
        [np.vdot(left, bra_insertion @ expm(float(distance) * shifted) @ initial) for distance in distances]
    )


def test_one_bit_cletta_matrices_match_existing_memory_construction():
    q = np.array([[-0.2, 0.1], [-0.05, -0.3]], dtype=np.complex128)
    r = np.array([[0.3, -0.04], [0.08, 0.2]], dtype=np.complex128)
    s = np.array([[0.02, 0.07], [-0.03, 0.05]], dtype=np.complex128)
    kappa = 1.4
    eye = np.eye(2, dtype=np.complex128)
    zeros = np.zeros_like(eye)

    q_memory, r_memory = cletta_memory_matrices(q, r, s, kappa, memory_dim=2)
    q_reference = np.block([[q, zeros], [zeros, q - kappa * eye]])
    r_reference = np.block([[r, np.sqrt(kappa) * eye], [s, r]])

    np.testing.assert_allclose(q_memory, q_reference, atol=1.0e-14)
    np.testing.assert_allclose(r_memory, r_reference, atol=1.0e-14)


def test_cletta_two_sided_hierarchy_matches_virtual_pseudomode_action():
    rng = np.random.default_rng(73)
    bond_dim = 2
    memory_dim = 3
    kappa = 1.3
    r = 0.2 * (rng.normal(size=(bond_dim, bond_dim)) + 1.0j * rng.normal(size=(bond_dim, bond_dim)))
    drift = 0.1 * (rng.normal(size=(bond_dim, bond_dim)) + 1.0j * rng.normal(size=(bond_dim, bond_dim)))
    drift = drift - drift.conj().T
    q = drift - 0.5 * (r.conj().T @ r)
    s = 0.08 * (rng.normal(size=(bond_dim, bond_dim)) + 1.0j * rng.normal(size=(bond_dim, bond_dim)))
    blocks = rng.normal(size=(memory_dim, memory_dim, bond_dim, bond_dim))
    blocks = blocks + 1.0j * rng.normal(size=blocks.shape)

    q_memory, r_memory = cletta_memory_matrices(q, r, s, kappa, memory_dim=memory_dim)
    matrix = hierarchy_blocks_to_matrix(blocks)
    explicit = q_memory @ matrix + matrix @ q_memory.conj().T + r_memory @ matrix @ r_memory.conj().T
    hierarchical = hierarchy_blocks_to_matrix(apply_cletta_memory_hierarchy(blocks, q, r, s, kappa))
    explicit_ket = r_memory @ matrix
    hierarchy_ket = hierarchy_blocks_to_matrix(apply_cletta_ket_insertion(blocks, r, s, kappa))
    explicit_bra = matrix @ r_memory.conj().T
    hierarchy_bra = hierarchy_blocks_to_matrix(apply_cletta_bra_insertion(blocks, r, s, kappa))

    np.testing.assert_allclose(hierarchical, explicit, atol=1.0e-12)
    np.testing.assert_allclose(hierarchy_ket, explicit_ket, atol=1.0e-12)
    np.testing.assert_allclose(hierarchy_bra, explicit_bra, atol=1.0e-12)


def test_cletta_hierarchy_matches_stationary_pseudomode_correlation():
    bond_dim = 2
    memory_dim = 3
    kappa = 1.1
    r = np.array([[0.22, 0.06], [-0.03, 0.17]], dtype=np.complex128)
    drift = np.array([[0.0, 0.09], [-0.09, 0.0]], dtype=np.complex128)
    q = drift - 0.5 * (r.conj().T @ r)
    s = np.array([[0.035, -0.018], [0.027, 0.041]], dtype=np.complex128)
    distances = np.array([0.0, 0.15, 0.5, 1.0])

    q_memory, r_memory = cletta_memory_matrices(q, r, s, kappa, memory_dim=memory_dim)
    explicit_generator = _explicit_transfer(q_memory, r_memory)
    explicit_dim = q_memory.shape[0]
    explicit_ket = np.kron(r_memory, np.eye(explicit_dim, dtype=np.complex128))
    explicit_bra = np.kron(np.eye(explicit_dim, dtype=np.complex128), r_memory.conj())
    explicit_values = _stationary_field_correlation(
        explicit_generator,
        explicit_ket,
        explicit_bra,
        distances,
    )

    hierarchy_generator = cletta_memory_hierarchy_generator(
        q,
        r,
        s,
        kappa,
        memory_dim=memory_dim,
    )
    hierarchy_ket = cletta_ket_insertion_matrix(r, s, kappa, memory_dim=memory_dim)
    hierarchy_bra = cletta_bra_insertion_matrix(r, s, kappa, memory_dim=memory_dim)
    hierarchy_values = _stationary_field_correlation(
        hierarchy_generator,
        hierarchy_ket,
        hierarchy_bra,
        distances,
    )

    permutation = _hierarchy_to_explicit_permutation(bond_dim, memory_dim)
    np.testing.assert_allclose(
        hierarchy_generator,
        permutation.conj().T @ explicit_generator @ permutation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        hierarchy_ket,
        permutation.conj().T @ explicit_ket @ permutation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        hierarchy_bra,
        permutation.conj().T @ explicit_bra @ permutation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(hierarchy_values, explicit_values, atol=2.0e-10)


def test_two_mode_cletta_hierarchy_matches_explicit_pseudomodes():
    rng = np.random.default_rng(103)
    bond_dim = 2
    depth = 2
    rates = np.array([0.7, 1.9])
    r = 0.2 * rng.normal(size=(bond_dim, bond_dim))
    drift = 0.1 * rng.normal(size=(bond_dim, bond_dim))
    drift = drift - drift.T
    q = drift - 0.5 * (r.T @ r)
    ties = 0.08 * rng.normal(size=(2, bond_dim, bond_dim))
    memory_dim = len(cletta_memory_fock_keys(2, depth))
    blocks = rng.normal(size=(memory_dim, memory_dim, bond_dim, bond_dim))
    blocks = blocks + 1.0j * rng.normal(size=blocks.shape)

    q_memory, r_memory = cletta_multimode_memory_matrices(
        q,
        r,
        ties,
        rates,
        depth=depth,
    )
    matrix = hierarchy_blocks_to_matrix(blocks)
    explicit = q_memory @ matrix + matrix @ q_memory.conj().T + r_memory @ matrix @ r_memory.conj().T
    hierarchical = hierarchy_blocks_to_matrix(
        apply_cletta_multimode_memory_hierarchy(
            blocks,
            q,
            r,
            ties,
            rates,
            depth=depth,
        )
    )
    explicit_ket = r_memory @ matrix
    hierarchy_ket = hierarchy_blocks_to_matrix(
        apply_cletta_multimode_ket_insertion(
            blocks,
            r,
            ties,
            rates,
            depth=depth,
        )
    )
    explicit_bra = matrix @ r_memory.conj().T
    hierarchy_bra = hierarchy_blocks_to_matrix(
        apply_cletta_multimode_bra_insertion(
            blocks,
            r,
            ties,
            rates,
            depth=depth,
        )
    )

    np.testing.assert_allclose(hierarchical, explicit, atol=1.0e-12)
    np.testing.assert_allclose(hierarchy_ket, explicit_ket, atol=1.0e-12)
    np.testing.assert_allclose(hierarchy_bra, explicit_bra, atol=1.0e-12)

    permutation = _hierarchy_to_explicit_permutation(bond_dim, memory_dim)
    explicit_generator = _explicit_transfer(q_memory, r_memory)
    hierarchy_generator = cletta_multimode_hierarchy_generator(
        q,
        r,
        ties,
        rates,
        depth=depth,
    )
    explicit_ket_matrix = np.kron(r_memory, np.eye(r_memory.shape[0], dtype=np.complex128))
    explicit_bra_matrix = np.kron(np.eye(r_memory.shape[0], dtype=np.complex128), r_memory.conj())
    np.testing.assert_allclose(
        hierarchy_generator,
        permutation.conj().T @ explicit_generator @ permutation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cletta_multimode_ket_insertion_matrix(r, ties, rates, depth=depth),
        permutation.conj().T @ explicit_ket_matrix @ permutation,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cletta_multimode_bra_insertion_matrix(r, ties, rates, depth=depth),
        permutation.conj().T @ explicit_bra_matrix @ permutation,
        atol=1.0e-12,
    )


def test_two_mode_cletta_zero_ties_recovers_cmps_observables():
    state = ContinuousMPS.random_left_canonical(2, seed=107, scale=0.2)
    ties = np.zeros((2, state.bond_dim, state.bond_dim))
    q_memory, r_memory = cletta_multimode_memory_matrices(
        state.q,
        state.r,
        ties,
        [0.5, 2.0],
        depth=2,
    )
    generator = _explicit_transfer(q_memory, r_memory)
    value, left, right = _dominant_environment(generator)

    def insertion(operator):
        matrix = np.kron(operator, operator.conj())
        return float(np.real_if_close(np.vdot(left, matrix @ right)))

    commutator = q_memory @ r_memory - r_memory @ q_memory
    recovered = {
        "density": insertion(r_memory),
        "kinetic": insertion(commutator),
        "contact": insertion(r_memory @ r_memory),
    }
    reference = state.lieb_liniger_observables(coupling=1.0, mu=0.0)

    np.testing.assert_allclose(value, 0.0, atol=1.0e-12)
    for name in recovered:
        np.testing.assert_allclose(recovered[name], reference[name], atol=2.0e-11)


def test_two_mode_complex_poles_match_explicit_pseudomodes():
    rng = np.random.default_rng(211)
    bond_dim = 2
    depth = 2
    rates = np.array([0.7, 1.2])
    frequencies = np.array([0.4, -0.3])
    r = 0.15 * rng.normal(size=(bond_dim, bond_dim))
    drift = 0.1 * rng.normal(size=(bond_dim, bond_dim))
    drift = drift - drift.T
    q = drift - 0.5 * (r.T @ r)
    ties = 0.08 * rng.normal(size=(2, bond_dim, bond_dim))
    memory_dim = len(cletta_memory_fock_keys(2, depth))

    q_memory, r_memory = cletta_multimode_memory_matrices(
        q,
        r,
        ties,
        rates,
        frequencies=frequencies,
        depth=depth,
    )
    explicit_generator = _explicit_transfer(q_memory, r_memory)
    hierarchy_generator = cletta_multimode_hierarchy_generator(
        q,
        r,
        ties,
        rates,
        frequencies=frequencies,
        depth=depth,
    )
    permutation = _hierarchy_to_explicit_permutation(bond_dim, memory_dim)
    np.testing.assert_allclose(
        hierarchy_generator,
        permutation.conj().T @ explicit_generator @ permutation,
        atol=1.0e-12,
    )


def test_multimode_hierarchy_adjoint_matches_dense_generator():
    rng = np.random.default_rng(911)
    bond_dim = 2
    depth = 2
    r = 0.2 * rng.normal(size=(bond_dim, bond_dim))
    q = -0.5 * (r.T @ r)
    ties = 0.05 * rng.normal(size=(2, bond_dim, bond_dim))
    rates = np.array([0.6, 1.3])
    frequencies = np.array([0.4, -0.2])
    memory_dim = len(cletta_memory_fock_keys(2, depth))
    blocks = rng.normal(size=(memory_dim, memory_dim, bond_dim, bond_dim))
    blocks = blocks + 1.0j * rng.normal(size=blocks.shape)

    generator = cletta_multimode_hierarchy_generator(
        q,
        r,
        ties,
        rates,
        depth=depth,
        frequencies=frequencies,
    )
    expected = (generator.conj().T @ blocks.reshape(-1)).reshape(blocks.shape)
    actual = apply_cletta_multimode_memory_hierarchy_adjoint(
        blocks,
        q,
        r,
        ties,
        rates,
        depth=depth,
        frequencies=frequencies,
    )
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)


def test_multimode_sparse_hierarchy_matches_dense_generator():
    rng = np.random.default_rng(919)
    bond_dim = 2
    r = 0.2 * rng.normal(size=(bond_dim, bond_dim))
    q = -0.5 * (r.T @ r)
    ties = 0.05 * rng.normal(size=(2, bond_dim, bond_dim))
    rates = np.array([0.6, 1.3])
    frequencies = np.array([0.4, -0.2])
    keywords = dict(depth=2, frequencies=frequencies)
    dense = cletta_multimode_hierarchy_generator(q, r, ties, rates, **keywords)
    sparse = cletta_multimode_hierarchy_sparse_generator(q, r, ties, rates, **keywords)
    np.testing.assert_allclose(sparse.toarray(), dense, atol=1.0e-12)
