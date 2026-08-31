import numpy as np
import pytest
from scipy.linalg import expm
from types import SimpleNamespace

torch = pytest.importorskip("torch")

from pyqed.letta import (  # noqa: E402
    LETTAEvolution,
    TDVP,
    nearest_neighbor_hamiltonian,
    site_reduced_density_matrix,
    system_reduced_density_matrix,
    two_site_tdvp_step,
    window2_product_state,
    resolve_letta_backend,
)
from pyqed.letta.torch_tdvp import (  # noqa: E402
    TorchLETTATDVPEngine,
    TorchWindow2State,
    torch_backend_capabilities,
)


def _random_hermitian_bonds(dims, seed=31):
    rng = np.random.default_rng(seed)
    terms = []
    for left, right in zip(dims[:-1], dims[1:]):
        size = left * right
        matrix = rng.normal(size=(size, size)) + 1.0j * rng.normal(
            size=(size, size)
        )
        terms.append((matrix + matrix.conj().T) / (2.0 * size))
    return terms


def _phase_aligned_error(actual, reference):
    overlap = np.vdot(reference, actual)
    phase = 1.0 if abs(overlap) == 0.0 else overlap / abs(overlap)
    return np.max(np.abs(actual - phase * reference))


def test_torch_state_round_trip_preserves_raw_state_and_rdm():
    state = window2_product_state(
        ([1.0, 2.0j], [2.0, -1.0], [1.0j, 3.0]), max_bond=2
    )
    state.tensors[0] *= 0.37
    resident = TorchWindow2State.from_letta(state, dtype=torch.complex128)
    restored = resident.to_letta()
    np.testing.assert_allclose(
        restored.state_vector(), state.state_vector(), rtol=0.0, atol=0.0
    )
    np.testing.assert_allclose(
        system_reduced_density_matrix(resident),
        system_reduced_density_matrix(state),
        rtol=3.0e-14,
        atol=3.0e-14,
    )
    for site in range(len(state.dims)):
        np.testing.assert_allclose(
            site_reduced_density_matrix(resident, site),
            site_reduced_density_matrix(state, site),
            rtol=3.0e-14,
            atol=3.0e-14,
        )


def test_torch_two_site_step_is_exact_for_two_physical_sites():
    dims = (2, 3)
    term = _random_hermitian_bonds(dims)[0]
    state = window2_product_state(([1.0, 0.0], [0.0, 1.0, 0.0]), max_bond=3)
    engine = TorchLETTATDVPEngine(
        nearest_neighbor_hamiltonian([term], dims),
        max_bond=3,
        krylov_dim=32,
        krylov_tol=1.0e-14,
        num_threads=1,
    )
    actual, info = engine.step(state, 0.07)
    expected = expm(-0.07j * term) @ state.state_vector()
    np.testing.assert_allclose(
        actual.to_letta().state_vector(), expected, rtol=4.0e-13, atol=4.0e-13
    )
    assert info["integrator"] == "tdvp2"
    assert info["truncation_error"] < 1.0e-24


def test_torch_and_numpy_tdvp2_match_on_an_interior_center():
    dims = (2, 3, 2, 2)
    operator = nearest_neighbor_hamiltonian(
        _random_hermitian_bonds(dims, seed=37), dims
    )
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0], [0.0, 1.0]),
        max_bond=2,
    )
    numpy_state = two_site_tdvp_step(
        initial,
        operator,
        0.017,
        max_bond=2,
        krylov_dim=32,
        krylov_tol=1.0e-13,
    )
    torch_engine = TorchLETTATDVPEngine(
        operator,
        max_bond=2,
        krylov_dim=32,
        krylov_tol=1.0e-13,
        num_threads=1,
    )
    torch_state, _ = torch_engine.step(initial, 0.017)
    error = _phase_aligned_error(
        torch_state.to_letta().state_vector(), numpy_state.state_vector()
    )
    assert error < 2.0e-12


def test_torch_driver_checkpoint_matches_uninterrupted_run(tmp_path):
    dims = (2, 2, 2)
    operator = nearest_neighbor_hamiltonian(
        _random_hermitian_bonds(dims, seed=41), dims
    )
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0]), max_bond=2
    )
    options = dict(
        max_bond=2,
        saturation_steps=1,
        krylov_dim=32,
        krylov_tol=1.0e-13,
        backend="torch",
        torch_num_threads=1,
    )
    direct = LETTAEvolution(operator, **options)
    direct_state = direct.run(initial, 0.01, 3)

    first = LETTAEvolution(operator, **options)
    first.run(initial, 0.01, 1)
    path = first.save_checkpoint(tmp_path / "letta-torch.pkl")
    restarted = LETTAEvolution.load_checkpoint(path, operator)
    restarted_state = restarted.run(restarted.state, 0.01, 2)

    error = _phase_aligned_error(
        restarted_state.to_letta().state_vector(),
        direct_state.to_letta().state_vector(),
    )
    assert error < 3.0e-12
    assert restarted.backend == "torch"
    assert restarted.mode == direct.mode == "tdvp1"


def test_auto_backend_keeps_small_problems_on_numpy():
    dims = (2, 2, 2)
    operator = nearest_neighbor_hamiltonian(_random_hermitian_bonds(dims), dims)
    driver = LETTAEvolution(operator, max_bond=2, backend="auto")
    assert driver.backend == "numpy"


def test_public_tdvp_facade_dispatches_and_factorized_channels_match_dense():
    dims = (2, 3, 3, 2)
    operator = nearest_neighbor_hamiltonian(
        _random_hermitian_bonds(dims, seed=47), dims
    )
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0]),
        max_bond=2,
    )
    dense = TDVP(
        operator,
        backend="torch",
        channel_mode="dense",
        max_bond=2,
        krylov_dim=32,
        krylov_tol=1.0e-13,
        torch_num_threads=1,
    )
    factored = TDVP(
        operator,
        backend="torch",
        channel_mode="factorized",
        max_bond=2,
        krylov_dim=32,
        krylov_tol=1.0e-13,
        torch_num_threads=1,
    )
    dense_state, _ = dense.step(initial, 0.013)
    factored_state, _ = factored.step(initial, 0.013)
    error = _phase_aligned_error(
        factored_state.to_letta().state_vector(),
        dense_state.to_letta().state_vector(),
    )
    assert error < 3.0e-12
    assert factored.backend == "torch"
    assert factored.channel_mode == "factorized"

    numpy_dense = TDVP(
        operator, backend="numpy", channel_mode="dense", max_bond=2
    )
    numpy_factored = TDVP(
        operator, backend="numpy", channel_mode="factorized", max_bond=2
    )
    numpy_dense_state, _ = numpy_dense.step(initial, 0.013)
    numpy_factored_state, _ = numpy_factored.step(initial, 0.013)
    assert _phase_aligned_error(
        numpy_factored_state.state_vector(), numpy_dense_state.state_vector()
    ) < 3.0e-12


def test_unavailable_cuda_request_has_a_clear_error():
    if torch_backend_capabilities()["cuda"]:
        pytest.skip("CUDA is available on this runner.")
    dims = (2, 2)
    operator = nearest_neighbor_hamiltonian(_random_hermitian_bonds(dims), dims)
    with pytest.raises(RuntimeError, match="no available CUDA device"):
        TDVP(operator, backend="torch", device="cuda", max_bond=2)


def test_stateful_torch_tdvp_reuses_state_and_fixed_rank_plan():
    dims = (2, 3, 3)
    operator = nearest_neighbor_hamiltonian(
        _random_hermitian_bonds(dims, seed=53), dims
    )
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]), max_bond=2
    )
    engine = TDVP(
        operator,
        backend="torch",
        integrator="tdvp2",
        max_bond=2,
        torch_num_threads=1,
    )
    state, _ = engine.step(initial, 0.01)
    engine.set_integrator("tdvp1")
    same_state, _ = engine.step(state, 0.01)
    assert same_state is state
    same_state, _ = engine.step(same_state, 0.01)
    assert same_state is state
    assert engine.fixed_rank_plan_rebuilds == 1


def test_auto_backend_uses_measured_cpu_rank_crossover():
    operator = SimpleNamespace(nsites=20, dims=(2,) + (12,) * 20)
    assert resolve_letta_backend(operator, max_bond=2) == "numpy"
    assert resolve_letta_backend(operator, max_bond=8) == "torch"
