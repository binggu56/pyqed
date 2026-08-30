import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.lattice import Site, SpinHalfSite
from pyqed.peps import CTMRGEnvironment, PEPS, PEPSEvolution, U1PEPS
from pyqed.tn import Hamiltonian, LocalTerm


def spin_sites(shape):
    return tuple(SpinHalfSite() for _ in range(shape[0] * shape[1]))


def phase_aligned_error(reference, candidate):
    overlap = np.vdot(reference, candidate)
    phase = 1.0 if abs(overlap) == 0.0 else overlap / abs(overlap)
    return np.linalg.norm(candidate - phase * reference)


def test_finite_ctmrg_matches_exact_contraction_without_truncation():
    sites = spin_sites((2, 3))
    state = PEPS.random(
        sites,
        shape=(2, 3),
        D=2,
        seed=41,
        complex=True,
        normalize=False,
    )
    exact = state.norm_squared(method="exact")

    value, info = state.norm_squared(
        method="ctmrg",
        max_bond=64,
        rtol=0.0,
        return_info=True,
    )

    np.testing.assert_allclose(value, exact, atol=4.0e-13)
    assert info["method"] == "ctmrg"
    assert info["converged"]
    assert info["directional_spread"] < 1.0e-12
    assert set(info["corners"]) == {
        "northwest",
        "northeast",
        "southeast",
        "southwest",
    }
    assert all(corner.ndim == 2 for corner in info["corners"].values())
    assert len(info["history"]) >= 2
    environment = state.ctmrg(chi=64, rtol=0.0)
    assert isinstance(environment, CTMRGEnvironment)
    np.testing.assert_allclose(environment.value, exact, atol=4.0e-13)


def test_ctmrg_hamiltonian_expectation_matches_exact_at_sufficient_chi():
    sites = spin_sites((2, 2))
    state = PEPS.random(
        sites,
        shape=(2, 2),
        D=2,
        seed=43,
        complex=True,
        contraction="exact",
    )
    hamiltonian = Hamiltonian(sites)
    for first, second in ((0, 1), (0, 2), (1, 3), (2, 3)):
        hamiltonian.add_product(0.25, (first, "X"), (second, "X"))
        hamiltonian.add_product(0.25, (first, "Y"), (second, "Y"))
        hamiltonian.add_product(0.25, (first, "Z"), (second, "Z"))

    exact = state.expectation(hamiltonian, method="exact")
    ctmrg, info = state.expectation(
        hamiltonian,
        method="ctmrg",
        max_bond=64,
        rtol=0.0,
        workers=2,
        return_info=True,
    )

    np.testing.assert_allclose(ctmrg, exact, atol=3.0e-12)
    assert info["method"] == "ctmrg"
    assert info["norm"]["converged"]
    assert info["environment_reused"]
    assert info["environment_builds"] == 1
    assert info["workers"] == 2


def test_ctmrg_warm_start_skips_rank_growth_without_changing_value():
    sites = spin_sites((3, 3))
    state = PEPS.random(
        sites,
        shape=(3, 3),
        D=2,
        seed=45,
        normalize=False,
    )

    cold = state.ctmrg(chi=16)
    warm = state.ctmrg(chi=16)

    np.testing.assert_allclose(warm.value, cold.value, atol=2.0e-13)
    assert not cold.warm_started
    assert warm.warm_started
    assert len(warm.history) < len(cold.history)


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_peps_real_time_pair_evolution_matches_exact_result(shape):
    sites = spin_sites(shape)
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(1.0, (0, "X"), (1, "X"))
    state = PEPS.product_state(sites, [0, 0], shape=shape)
    initial = state.to_dense()
    final_time = 0.3

    evolution = state.evolve(
        hamiltonian,
        final_time,
        step=0.1,
        max_D=2,
        contraction="exact",
    )
    exact = expm(-1j * final_time * hamiltonian.to_dense()) @ initial

    assert isinstance(evolution, PEPSEvolution)
    assert evolution.time == final_time
    assert phase_aligned_error(exact, state.to_dense()) < 3.0e-13
    np.testing.assert_allclose(state.norm_squared(method="exact"), 1.0, atol=3.0e-13)
    assert evolution.history[-1]["max_bond"] == 2
    assert evolution.history[-1]["updates"][0]["backend"] == "dense"


def test_peps_imaginary_time_lowers_energy_and_matches_exact_pair_evolution():
    sites = spin_sites((1, 2))
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(-1.0, (0, "X"), (1, "X"))
    hamiltonian.add_product(-1.0, (0, "Z"), (1, "Z"))
    state = PEPS.product_state(sites, [0, 0], shape=(1, 2))
    initial = state.to_dense()
    initial_energy = hamiltonian.expectation(initial)
    beta = 0.4

    evolution = state.evolve(
        hamiltonian,
        beta,
        step=0.1,
        imaginary=True,
        max_D=2,
        contraction="exact",
    )
    exact = expm(-beta * hamiltonian.to_dense()) @ initial
    exact /= np.linalg.norm(exact)

    assert evolution.beta == beta
    assert evolution.energy < initial_energy
    assert phase_aligned_error(exact, state.to_dense()) < 3.0e-13


def test_second_order_peps_trotter_is_more_accurate_than_first_order():
    sites = spin_sites((1, 2))
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(0.7, (0, "Z"))
    hamiltonian.add_product(1.0, (0, "X"), (1, "X"))
    initial = np.array([1.0, 0.0, 0.0, 0.0])
    exact = expm(-0.1j * hamiltonian.to_dense()) @ initial
    errors = {}
    for order in (1, 2):
        state = PEPS.product_state(sites, [0, 0], shape=(1, 2))
        state.evolve(
            hamiltonian,
            0.1,
            step=0.1,
            order=order,
            max_D=2,
            contraction="exact",
        )
        errors[order] = phase_aligned_error(exact, state.to_dense())

    assert errors[2] < 0.15 * errors[1]


def test_peps_evolution_can_measure_energy_periodically_and_at_final_step():
    sites = spin_sites((1, 2))
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(-1.0, (0, "X"), (1, "X"))
    state = PEPS.product_state(sites, [0, 0], shape=(1, 2))

    evolution = state.evolve(
        hamiltonian,
        0.3,
        step=0.1,
        max_D=2,
        contraction="exact",
        measure_every=2,
        workers=2,
    )

    assert [record["measured"] for record in evolution.history] == [False, True, True]
    assert evolution.history[0]["energy"] is None
    assert evolution.history[-1]["energy"] == evolution.energy


def test_peps_evolution_rejects_non_nearest_neighbor_terms():
    sites = spin_sites((2, 2))
    operator = np.kron(sites[0].operator("X"), sites[3].operator("X"))
    hamiltonian = Hamiltonian(sites, terms=(LocalTerm((0, 3), operator),))
    state = PEPS.product_state(sites, [0, 0, 0, 0], shape=(2, 2))

    with pytest.raises(NotImplementedError, match="nearest-neighbor"):
        PEPSEvolution(state, hamiltonian)


def test_u1_peps_block_contraction_and_hamiltonian_match_dense_reference():
    sites = spin_sites((1, 2))
    bond_charges = {((0, 0), (0, 1)): (-1, 1)}
    state = U1PEPS.random(
        sites,
        shape=(1, 2),
        bond_charges=bond_charges,
        target_charges=(0, 0),
        seed=47,
        complex=True,
    )
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(0.5, (0, "X"), (1, "X"))
    hamiltonian.add_product(0.5, (0, "Y"), (1, "Y"))
    hamiltonian.add_product(0.2, (0, "Z"), (1, "Z"))
    dense = state.to_dense()

    norm, info = state.norm_squared(return_info=True)
    reference_norm, reference_info = state.norm_squared(
        method="enumerate",
        return_info=True,
    )
    energy, energy_info = state.expectation(
        hamiltonian,
        workers=2,
        return_info=True,
    )

    np.testing.assert_allclose(norm, np.vdot(dense, dense), atol=3.0e-13)
    np.testing.assert_allclose(norm, reference_norm, atol=3.0e-13)
    np.testing.assert_allclose(energy, hamiltonian.expectation(dense), atol=3.0e-13)
    assert info["method"] == "u1-block-frontier"
    assert info["exact"]
    assert info["max_active_frontiers"] <= 2
    assert reference_info["configurations"] == 2
    assert 0.0 < state.storage_fraction < 1.0
    assert energy_info["storage_fraction"] == state.storage_fraction
    assert energy_info["workers"] == 2


def test_u1_dense_roundtrip_preserves_tensor_blocks_and_state():
    sites = spin_sites((1, 2))
    charges = {((0, 0), (0, 1)): (-1, 1)}
    state = U1PEPS.random(
        sites,
        shape=(1, 2),
        bond_charges=charges,
        target_charges=(0, 0),
        seed=53,
    )

    restored = U1PEPS.from_dense(
        state.to_dense_peps(),
        bond_charges=charges,
        target_charges=(0, 0),
    )

    np.testing.assert_allclose(restored.to_dense(), state.to_dense(), atol=0.0)
    assert restored.block_count == state.block_count
    assert all(
        tensor.block_size < tensor.size
        for row in restored.tensors
        for tensor in row
    )


def test_u1_frontier_merges_sector_histories_and_supports_directional_ctmrg():
    shape = (3, 3)
    sites = spin_sites(shape)
    bond_charges = {}
    targets = []
    for row in range(shape[0]):
        for col in range(shape[1]):
            degree = (
                (row > 0)
                + (row + 1 < shape[0])
                + (col > 0)
                + (col + 1 < shape[1])
            )
            targets.append(1 if degree % 2 == 0 else 0)
            if col + 1 < shape[1]:
                bond_charges[((row, col), (row, col + 1))] = (-1, 1)
            if row + 1 < shape[0]:
                bond_charges[((row, col), (row + 1, col))] = (-1, 1)
    state = U1PEPS.random(
        sites,
        shape=shape,
        bond_charges=bond_charges,
        target_charges=targets,
        seed=59,
    )

    frontier, info = state.norm_squared(return_info=True)
    reference, reference_info = state.norm_squared(
        method="enumerate",
        return_info=True,
    )
    ctmrg, ctmrg_info = state.norm_squared(method="ctmrg", return_info=True)
    approximate, approximate_info = state.norm_squared(
        max_frontiers=4,
        return_info=True,
    )

    np.testing.assert_allclose(frontier, reference, atol=4.0e-13)
    np.testing.assert_allclose(ctmrg, reference, atol=4.0e-13)
    assert info["max_active_frontiers"] < reference_info["configurations"]
    assert ctmrg_info["method"] == "u1-block-ctmrg"
    assert ctmrg_info["exact"]
    assert np.isfinite(approximate)
    assert not approximate_info["exact"]
    assert approximate_info["max_active_frontiers"] <= 4
    assert approximate_info["discarded_weight"] >= 0.0


def test_u1_peps_supports_degenerate_charge_sectors_and_arbitrary_local_dim():
    site = Site(
        labels=("a", "b", "c"),
        charges=(0, 0, 1),
        charge_labels=("n",),
    )
    state = U1PEPS.product_state((site,), [1], shape=(1, 1))

    np.testing.assert_allclose(state.to_dense(), [0.0, 1.0, 0.0], atol=0.0)
    assert state.tensors[0][0].data[(0, 0, 0, 0, 0)].shape == (2, 1, 1, 1, 1)
    assert state.target_charge == 0

    dense = PEPS.product_state((site,), [2], shape=(1, 1))
    with pytest.raises(ValueError, match="symmetry-forbidden"):
        U1PEPS.from_dense(dense, target_charges=(0,))


def test_u1_peps_time_evolution_grows_charge_sectors_and_matches_exact():
    sites = spin_sites((1, 2))
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(0.5, (0, "X"), (1, "X"))
    hamiltonian.add_product(0.5, (0, "Y"), (1, "Y"))
    state = U1PEPS.product_state(sites, [0, 1], shape=(1, 2))
    initial = state.to_dense()
    final_time = 0.3

    evolution = state.evolve(
        hamiltonian,
        final_time,
        step=0.1,
        max_D=2,
    )
    exact = expm(-1j * final_time * hamiltonian.to_dense()) @ initial

    assert phase_aligned_error(exact, state.to_dense()) < 3.0e-13
    assert state.tensors[0][0].axis_charges[2] == (0, 2)
    assert evolution.history[-1]["updates"][0]["backend"] == "u1-block-svd"
    assert evolution.history[-1]["updates"][0]["sector_ranks"] == {0: 1, 2: 1}


def test_u1_peps_rejects_a_charge_breaking_local_gate():
    sites = spin_sites((1, 1))
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(1.0, (0, "X"))
    state = U1PEPS.product_state(sites, [0], shape=(1, 1))

    with pytest.raises(ValueError, match="symmetry-forbidden"):
        state.evolve(hamiltonian, 0.1, step=0.1)
