import numpy as np

from pyqed.lgt import QuantumSchwingerDVR


def test_quantum_schwinger_basis_satisfies_charge_neutrality_and_gauss_law():
    model = QuantumSchwingerDVR(3, 8.0, flux_cutoff=2)

    for bits, flux in zip(model.basis_bits, model.basis_flux):
        assert int(bits).bit_count() == model.npts
        np.testing.assert_array_equal(model.gauss_law(int(bits), flux), 0)


def test_quantum_schwinger_hamiltonian_and_channels_are_hermitian():
    model = QuantumSchwingerDVR(3, 8.0, flux_cutoff=2)
    hamiltonian = model.build_hamiltonian()
    vector, scalar = model.build_channel_operators()

    np.testing.assert_allclose(hamiltonian.toarray(), hamiltonian.toarray().conj().T)
    np.testing.assert_allclose(vector.toarray(), vector.toarray().conj().T)
    np.testing.assert_allclose(scalar.toarray(), scalar.toarray().conj().T)


def test_quantum_schwinger_solver_resolves_nonvacuum_channels():
    model = QuantumSchwingerDVR(3, 10.0, flux_cutoff=2).run(nroots=24)

    assert model.vacuum_dimension >= 1
    assert model.vector_level >= model.vacuum_dimension
    assert model.scalar_level >= model.vacuum_dimension
    assert model.vector_excitation_energy > 0.0
    assert model.vector_gap >= 0.0
    assert model.scalar_gap > 0.0
