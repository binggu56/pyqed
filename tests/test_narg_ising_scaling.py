import numpy as np

from pyqed.narg import (
    TransverseFieldIsingNARG,
    finite_size_scaling_dimensions,
    narg_fixed_layer_scaling_dimensions,
    transverse_field_ising_hamiltonian,
)


def test_periodic_critical_ising_finite_size_dimensions_match_cft_targets():
    scaling = finite_size_scaling_dimensions(8, nlevels=4)

    np.testing.assert_allclose(scaling.dimensions[1], 0.125, atol=6e-3)
    np.testing.assert_allclose(scaling.dimensions[2], 1.0, atol=1.5e-2)
    assert scaling.velocity == 2.0


def test_transverse_field_ising_hamiltonian_is_hermitian():
    hamiltonian = transverse_field_ising_hamiltonian(4, sparse=False)

    assert hamiltonian.shape == (16, 16)
    np.testing.assert_allclose(hamiltonian, hamiltonian.T.conj(), atol=1e-12)


def test_sequential_ising_narg_runs_and_exposes_fixed_layer_scaling():
    result = TransverseFieldIsingNARG(
        6,
        bond_dim=4,
        nstart=2,
    ).run(nroots=4)
    scaling = narg_fixed_layer_scaling_dimensions(result.steps[-1].tensor)
    odd = narg_fixed_layer_scaling_dimensions(
        result.steps[-1].tensor,
        symmetry_operator=result.symmetry_operator,
        input_symmetry_operator=result.steps[-1].input_symmetry_operator,
        sector="odd",
    )
    even = narg_fixed_layer_scaling_dimensions(
        result.steps[-1].tensor,
        symmetry_operator=result.symmetry_operator,
        input_symmetry_operator=result.steps[-1].input_symmetry_operator,
        sector="even",
    )

    assert result.energies.shape == (4,)
    assert np.all(np.diff(result.energies) >= -1e-12)
    assert result.steps[-1].tensor.shape == (8, 4, 2)
    np.testing.assert_allclose(
        result.steps[-1].input_symmetry_operator @ result.steps[-1].input_symmetry_operator,
        np.eye(result.steps[-1].input_symmetry_operator.shape[0]),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        result.symmetry_operator @ result.symmetry_operator,
        np.eye(result.symmetry_operator.shape[0]),
        atol=1e-10,
    )
    assert scaling.superoperator.shape == (64, 64)
    assert odd.superoperator.shape[0] + even.superoperator.shape[0] == 64
    assert np.all(np.isfinite(scaling.dimensions))
    assert np.all(np.isfinite(odd.dimensions))
    assert np.all(np.isfinite(even.dimensions))
    assert scaling.dimensions[0] < 1e-10
