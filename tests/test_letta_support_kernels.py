import numpy as np
import pytest

from pyqed.letta import _support_kernels


def _complex_random(rng, shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def _kernel_case(seed=7):
    rng = np.random.default_rng(seed)
    left = _complex_random(rng, (3, 3, 2, 2, 2))
    right = _complex_random(rng, (2, 2, 3, 2, 2))
    coords = np.asarray(
        [
            (a, si, sj, b)
            for a in range(3)
            for si in range(2)
            for sj in range(2)
            for b in range(2)
            if (a + si + 2 * sj + b) % 3 != 1
        ],
        dtype=np.int64,
    )

    bra_i = []
    ket_i = []
    bra_j = []
    ket_j = []
    entry_starts = [0]
    entry_m = []
    entry_n = []
    entry_values = []
    for pbra_i in range(2):
        for pket_i in range(2):
            for pbra_j in range(2):
                for pket_j in range(2):
                    bra_i.append(pbra_i)
                    ket_i.append(pket_i)
                    bra_j.append(pbra_j)
                    ket_j.append(pket_j)
                    nentries = 1 + (pbra_i + pket_i + pbra_j + pket_j) % 3
                    for entry in range(nentries):
                        entry_m.append((entry + pbra_i) % 2)
                        entry_n.append((2 * entry + pket_j) % 3)
                        entry_values.append(rng.normal() + 1j * rng.normal())
                    entry_starts.append(len(entry_m))

    return (
        coords,
        left,
        right,
        np.asarray(bra_i, dtype=np.int64),
        np.asarray(ket_i, dtype=np.int64),
        np.asarray(bra_j, dtype=np.int64),
        np.asarray(ket_j, dtype=np.int64),
        np.asarray(entry_starts, dtype=np.int64),
        np.asarray(entry_m, dtype=np.int64),
        np.asarray(entry_n, dtype=np.int64),
        np.asarray(entry_values, dtype=np.complex128),
    )


def test_python_support_action_matches_explicit_dense_matrix():
    inputs = _kernel_case()
    matrix = _support_kernels.assemble_support_hamiltonian(
        *inputs,
        backend="python",
    )
    rng = np.random.default_rng(19)
    vector = _complex_random(rng, (matrix.shape[0],))
    vectors = _complex_random(rng, (matrix.shape[0], 4))

    np.testing.assert_allclose(
        _support_kernels.apply_support_hamiltonian(
            *inputs,
            vector,
            backend="python",
        ),
        matrix @ vector,
        atol=2.0e-13,
    )
    np.testing.assert_allclose(
        _support_kernels.apply_support_hamiltonian(
            *inputs,
            vectors,
            backend="python",
        ),
        matrix @ vectors,
        atol=3.0e-13,
    )


@pytest.mark.skipif(
    not _support_kernels.native_available(),
    reason="optional C++ LETTA support kernels are not built",
)
def test_native_dense_support_hamiltonian_matches_python_complex_reference():
    inputs = _kernel_case(seed=11)
    reference = _support_kernels.assemble_support_hamiltonian(
        *inputs,
        backend="python",
    )
    native = _support_kernels.assemble_support_hamiltonian(
        *inputs,
        backend="native",
    )
    np.testing.assert_allclose(native, reference, atol=2.0e-13)


@pytest.mark.skipif(
    not _support_kernels.native_available(),
    reason="optional C++ LETTA support kernels are not built",
)
def test_native_batched_support_action_matches_dense_complex_reference():
    inputs = _kernel_case(seed=23)
    matrix = _support_kernels.assemble_support_hamiltonian(
        *inputs,
        backend="python",
    )
    rng = np.random.default_rng(29)
    vector = _complex_random(rng, (matrix.shape[0],))
    vectors = _complex_random(rng, (matrix.shape[0], 5))

    native_vector = _support_kernels.apply_support_hamiltonian(
        *inputs,
        vector,
        backend="native",
    )
    native_vectors = _support_kernels.apply_support_hamiltonian(
        *inputs,
        vectors,
        backend="native",
    )
    assert native_vector.shape == vector.shape
    assert native_vectors.shape == vectors.shape
    np.testing.assert_allclose(native_vector, matrix @ vector, atol=3.0e-13)
    np.testing.assert_allclose(native_vectors, matrix @ vectors, atol=4.0e-13)


def test_support_kernel_rejects_invalid_compact_transition_offsets():
    inputs = list(_kernel_case())
    inputs[7] = inputs[7].copy()
    inputs[7][-1] -= 1
    with pytest.raises(ValueError, match="entry_starts"):
        _support_kernels.assemble_support_hamiltonian(
            *inputs,
            backend="python",
        )
