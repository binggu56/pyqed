import numpy as np

from examples.mps.unordered_trotter_boundary_mps import (
    compress_uniform_state,
    effective_uniform_tensor,
    evaluate,
)
from pyqed.mps.umps import UniformMPS


def test_layer_fusion_has_expected_bond_dimension():
    tensor = effective_uniform_tensor(
        np.array([0.1, -0.2, 0.3, 0.15, -0.1]),
        spacing=0.5,
        layers=3,
        local_cutoff=2,
    )
    assert tensor.shape == (3, 8, 8)


def test_full_boundary_dimension_preserves_exact_state():
    tensor = effective_uniform_tensor(
        np.array([0.1, -0.2, 0.3, 0.15, -0.1]),
        spacing=0.5,
        layers=2,
        local_cutoff=2,
    )
    state = UniformMPS(tensor).normalize_transfer()
    compressed, discarded = compress_uniform_state(state, state.bond_dim)
    np.testing.assert_allclose(discarded, 0.0)
    np.testing.assert_allclose(
        compressed.expectation_one_site(np.diag([0.0, 1.0, 2.0])),
        state.expectation_one_site(np.diag([0.0, 1.0, 2.0])),
    )


def test_operator_string_hierarchy_converges_to_exact_local_gate():
    parameters = np.array([0.1, -0.2, 0.3, 0.15, -0.1])
    exact = evaluate(
        parameters,
        spacing=0.5,
        coupling=1.0,
        layers=2,
        local_cutoff=2,
    )
    low = evaluate(
        parameters,
        spacing=0.5,
        coupling=1.0,
        layers=2,
        local_cutoff=2,
        string_order=2,
    )
    high = evaluate(
        parameters,
        spacing=0.5,
        coupling=1.0,
        layers=2,
        local_cutoff=2,
        string_order=8,
    )
    assert abs(high["energy"] - exact["energy"]) < abs(low["energy"] - exact["energy"])
