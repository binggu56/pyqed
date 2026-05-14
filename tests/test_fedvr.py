import numpy as np

from pyqed.dvr import FEDVR


def test_fedvr_harmonic_oscillator_low_energies():
    dvr = FEDVR(-8.0, 8.0, n_elements=12, n_lobatto=6, mass=1.0)
    energies, _ = dvr.run(lambda x: 0.5 * x**2, num_eigs=6)

    exact = np.arange(6, dtype=float) + 0.5
    np.testing.assert_allclose(energies, exact, atol=2.0e-4)


def test_fedvr_kinetic_is_sparse_and_hermitian():
    dvr = FEDVR(-5.0, 5.0, n_elements=8, n_lobatto=5, mass=2.0)
    kinetic = dvr.kinetic_sparse()

    assert kinetic.nnz < dvr.npts * dvr.npts
    np.testing.assert_allclose(
        (kinetic - kinetic.conj().T).toarray(),
        0.0,
        atol=1.0e-12,
    )


def test_fedvr_boundary_none_keeps_global_endpoints():
    dvr = FEDVR(-1.0, 1.0, n_elements=2, n_lobatto=4, boundary="none")

    assert dvr.npts == 7
    np.testing.assert_allclose(dvr.x[[0, -1]], [-1.0, 1.0])
