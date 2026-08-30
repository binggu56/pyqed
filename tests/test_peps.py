import numpy as np
import pytest

from pyqed import PEPS
from pyqed.lattice import SpinHalfSite
from pyqed.peps import BoundaryMPSContractor, PEPSOptimizer
from pyqed.tn import Hamiltonian


def spin_sites(shape):
    return tuple(SpinHalfSite() for _ in range(shape[0] * shape[1]))


def test_product_peps_uses_row_major_physical_order():
    sites = spin_sites((2, 2))
    state = PEPS.product_state(sites, [0, 1, 0, 1], shape=(2, 2))
    expected = np.zeros(16)
    expected[int("0101", 2)] = 1.0

    np.testing.assert_allclose(state.to_dense(), expected, atol=0.0)
    assert state.shape == (2, 2)
    assert state.bond_dims == {
        "horizontal": (1, 1),
        "vertical": (1, 1),
    }
    assert state.coordinate(3) == (1, 1)
    assert state.site_index((1, 0)) == 2


def test_exact_and_untruncated_boundary_contractions_match_dense_overlap():
    sites = spin_sites((2, 3))
    bra = PEPS.random(
        sites,
        shape=(2, 3),
        D=2,
        seed=3,
        complex=True,
        normalize=False,
    )
    ket = PEPS.random(
        sites,
        shape=(2, 3),
        D=2,
        seed=7,
        complex=True,
        normalize=False,
    )
    dense = np.vdot(bra.to_dense(), ket.to_dense())

    exact = bra.overlap(ket, method="exact")
    boundary, info = bra.overlap(
        ket,
        method="boundary",
        max_bond=None,
        rtol=0.0,
        return_info=True,
    )

    np.testing.assert_allclose(exact, dense, atol=2.0e-13)
    np.testing.assert_allclose(boundary, dense, atol=2.0e-13)
    assert info["method"] == "boundary"
    assert info["direction"] == "columns"
    assert info["max_relative_error"] == 0.0


def test_local_and_hamiltonian_expectations_match_dense_reference():
    sites = spin_sites((2, 2))
    state = PEPS.random(
        sites,
        shape=(2, 2),
        D=2,
        seed=11,
        complex=True,
        normalize=True,
        contraction="exact",
    )
    hamiltonian = Hamiltonian(sites, constant=0.17)
    horizontal_and_vertical = ((0, 1), (0, 2), (1, 3), (2, 3))
    for first, second in horizontal_and_vertical:
        hamiltonian.add_product(0.4, (first, "X"), (second, "X"))
        hamiltonian.add_product(0.3, (first, "Y"), (second, "Y"))
        hamiltonian.add_product(0.2, (first, "Z"), (second, "Z"))
    hamiltonian.add_product(-0.13, (3, "Z"))
    vector = state.to_dense()
    exact_energy = hamiltonian.expectation(vector)

    peps_energy = state.expectation(hamiltonian, method="exact")
    boundary_energy = state.expectation(
        hamiltonian,
        method="boundary",
        max_bond=None,
        rtol=0.0,
        workers=2,
        return_info=True,
    )
    local = state.local_expectation(
        {(0, 0): sites[0].operator("X"), 3: sites[3].operator("Z")},
        method="exact",
    )
    dense_operator = np.kron(
        np.kron(np.kron(sites[0].operator("X"), np.eye(2)), np.eye(2)),
        sites[3].operator("Z"),
    )

    np.testing.assert_allclose(peps_energy, exact_energy, atol=3.0e-12)
    np.testing.assert_allclose(boundary_energy[0], exact_energy, atol=3.0e-12)
    assert boundary_energy[1]["environment_reused"]
    assert boundary_energy[1]["environment_builds"] == 1
    assert boundary_energy[1]["workers"] == 2
    assert boundary_energy[1]["term_contractions"] == 13
    np.testing.assert_allclose(
        local,
        np.vdot(vector, dense_operator @ vector) / np.vdot(vector, vector),
        atol=3.0e-13,
    )


def test_boundary_contractor_reports_controlled_truncation():
    sites = spin_sites((3, 3))
    state = PEPS.random(
        sites,
        shape=(3, 3),
        D=2,
        seed=19,
        normalize=False,
    )

    value, info = state.norm_squared(
        method="boundary",
        max_bond=2,
        rtol=1.0e-12,
        return_info=True,
    )

    assert isinstance(BoundaryMPSContractor(max_bond=2), BoundaryMPSContractor)
    assert np.isfinite(value)
    assert info["max_bond"] == 2
    assert all(
        rank <= 2
        for row_ranks in info["row_bond_dims"]
        for rank in row_ranks
    )
    assert info["discarded_weight"] >= 0.0


def test_hamiltonian_frontier_batches_terms_and_reuses_double_layers():
    sites = spin_sites((3, 3))
    hamiltonian = Hamiltonian(sites)
    for first, second in ((0, 1), (1, 2), (0, 3), (1, 4), (4, 5), (4, 7)):
        for operator in ("X", "Y", "Z"):
            hamiltonian.add_product(
                0.25,
                (first, operator),
                (second, operator),
            )
    state = PEPS.random(
        sites,
        shape=(3, 3),
        D=2,
        seed=21,
        normalize=False,
    )

    first, first_info = state.expectation(
        hamiltonian,
        method="boundary",
        max_bond=8,
        return_info=True,
    )
    second, second_info = state.expectation(
        hamiltonian,
        method="boundary",
        max_bond=8,
        workers=2,
        return_info=True,
    )

    np.testing.assert_allclose(second, first, atol=2.0e-13)
    assert first_info["batched_frontier"]
    assert first_info["layer_cache_misses"] > 0
    assert second_info["layer_cache_misses"] == 0
    assert second_info["layer_cache_hits"] > 0


def test_exact_one_site_peps_optimizer_is_variational():
    sites = spin_sites((2, 2))
    hamiltonian = Hamiltonian(sites)
    for site in range(4):
        hamiltonian.add_product(-1.0, (site, "Z"))
    state = PEPS.random(
        sites,
        shape=(2, 2),
        D=2,
        seed=23,
        complex=True,
        normalize=True,
        contraction="exact",
    )
    initial_energy = state.expectation(hamiltonian, method="exact")

    optimizer = state.optimize(hamiltonian, sweeps=3, tol=1.0e-12)

    assert isinstance(optimizer, PEPSOptimizer)
    assert optimizer.success
    assert optimizer.energy <= initial_energy
    np.testing.assert_allclose(optimizer.energy, -4.0, atol=2.0e-12)
    np.testing.assert_allclose(
        state.expectation(hamiltonian, method="exact"),
        optimizer.energy,
        atol=2.0e-12,
    )
    assert all(
        update["energy_after"] <= update["energy_before"] + 1.0e-9
        for sweep in optimizer.history
        for update in sweep["updates"]
    )


def test_exact_network_effective_environment_matches_global_contractions():
    sites = spin_sites((2, 2))
    hamiltonian = Hamiltonian(sites, constant=0.07)
    hamiltonian.add_product(0.4, (0, "X"), (1, "X"))
    hamiltonian.add_product(0.2, (0, "Z"), (2, "Z"))
    hamiltonian.add_product(-0.3, (3, "Y"))
    state = PEPS.random(
        sites,
        shape=(2, 2),
        D=2,
        seed=31,
        complex=True,
        normalize=False,
    )
    coordinate = (0, 1)
    effective, metric, info = state.effective_environment(
        hamiltonian,
        coordinate,
    )
    tensor = state.tensors[coordinate[0]][coordinate[1]].reshape(-1)
    local_norm = np.vdot(tensor, metric @ tensor)
    local_energy = np.vdot(tensor, effective @ tensor) / local_norm

    np.testing.assert_allclose(
        local_norm,
        state.norm_squared(method="exact"),
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        local_energy,
        state.expectation(hamiltonian, method="exact"),
        atol=3.0e-13,
    )
    assert info["method"] == "exact-network"


@pytest.mark.parametrize("shape", [(1, 2), (2, 1)])
def test_two_site_optimizer_grows_bond_and_creates_entanglement(shape):
    sites = spin_sites(shape)
    hamiltonian = Hamiltonian(sites)
    hamiltonian.add_product(-1.0, (0, "X"), (1, "X"))
    hamiltonian.add_product(-1.0, (0, "Z"), (1, "Z"))
    state = PEPS.product_state(sites, [0, 0], shape=shape)

    optimizer = state.optimize(
        hamiltonian,
        update="two-site",
        max_D=2,
        sweeps=2,
        tol=1.0e-13,
    )

    np.testing.assert_allclose(optimizer.energy, -2.0, atol=2.0e-13)
    values = state.bond_dims["horizontal"] + state.bond_dims["vertical"]
    assert values == (2,)
    np.testing.assert_allclose(
        np.abs(state.to_dense()),
        [1 / np.sqrt(2), 0.0, 0.0, 1 / np.sqrt(2)],
        atol=2.0e-13,
    )
    assert optimizer.history[0]["updates"][0]["relative_split_error"] == 0.0


def test_peps_rejects_mismatched_virtual_bonds():
    tensors = [
        [np.zeros((2, 1, 2, 2, 1)), np.zeros((2, 1, 1, 2, 3))],
        [np.zeros((2, 2, 2, 1, 1)), np.zeros((2, 2, 1, 1, 2))],
    ]

    with pytest.raises(ValueError, match="horizontal bond mismatch"):
        PEPS(tensors)
