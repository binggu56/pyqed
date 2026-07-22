import numpy as np

from pyqed.letta import LETTA


def test_letta_state_overlap_matches_dense_vector_overlap():
    dims = (2, 2, 2, 2)
    bra = LETTA(None, dims, bond_dim=3, seed=11)
    ket = LETTA(None, dims, bond_dim=2, seed=12)

    expected = np.vdot(bra.state_vector(), ket.state_vector())

    np.testing.assert_allclose(bra.state_overlap(ket), expected, atol=1.0e-12)
    np.testing.assert_allclose(bra.state_overlap(bra), np.vdot(bra.state_vector(), bra.state_vector()), atol=1.0e-12)
    np.testing.assert_allclose(bra.fidelity(bra), 1.0, atol=1.0e-12)


def test_letta_terminal_state_overlap_matches_dense_vector_overlap():
    rng = np.random.default_rng(7)
    dims = (2, 2, 2)
    bra_tensors = [
        rng.normal(size=(1, 2, 2, 3)),
        rng.normal(size=(3, 2, 2, 2)),
        rng.normal(size=(2, 2)),
    ]
    ket_tensors = [
        rng.normal(size=(1, 2, 2, 2)),
        rng.normal(size=(2, 2, 2, 4)),
        rng.normal(size=(2, 4)),
    ]
    bra = LETTA(None, dims, tensors=bra_tensors)
    ket = LETTA(None, dims, tensors=ket_tensors)

    expected = np.vdot(bra.state_vector(), ket.state_vector())

    np.testing.assert_allclose(bra.state_overlap(ket), expected, atol=1.0e-12)
    np.testing.assert_allclose(ket.state_overlap(bra), np.conj(expected), atol=1.0e-12)
    assert 0.0 <= bra.fidelity(ket) <= 1.0 + 1.0e-12
