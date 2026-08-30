import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.lattice import Site, SpinHalfSite
from pyqed.mps import PurifiedMPS, TDMPS, infinite_temperature_mps, lift_physical_mpo
from pyqed.tn import Hamiltonian


def test_infinite_temperature_mps_is_a_product_of_local_bell_pairs():
    sites = (Site(2), Site(3))

    state = infinite_temperature_mps(sites)

    assert state.dims == [4, 9]
    assert state.get_bond_dimensions() == [1]
    np.testing.assert_allclose(state.norm_squared(), 1.0, atol=1.0e-15)
    for tensor, dim in zip(state.factors, (2, 3)):
        expected = np.eye(dim).reshape(-1) / np.sqrt(dim)
        np.testing.assert_allclose(tensor[0, :, 0], expected, atol=0.0)


def test_lifted_mpo_acts_only_on_the_physical_factor():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site,))
    hamiltonian.add_product(1.0, (0, "X"))

    lifted = lift_physical_mpo(hamiltonian.to_mpo())

    np.testing.assert_allclose(
        lifted.to_dense(),
        np.kron(site.operator("X"), np.eye(site.dim)),
        atol=0.0,
    )


def test_one_spin_purified_mps_matches_exact_thermal_energy_and_partition():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site,))
    hamiltonian.add_product(1.0, (0, "Z"))
    beta = 0.7

    thermal = PurifiedMPS(hamiltonian, D=2).run(beta, step=0.1)

    np.testing.assert_allclose(thermal.energy, -np.tanh(beta), atol=2.0e-14)
    assert isinstance(thermal.tdmps, TDMPS)
    np.testing.assert_allclose(
        thermal.log_partition_function,
        np.log(2.0 * np.cosh(beta)),
        atol=2.0e-14,
    )
    assert thermal.beta == beta
    assert thermal.success


def test_real_time_evolution_under_preparation_hamiltonian_is_stationary():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site,))
    hamiltonian.add_product(1.0, (0, "Z"))
    beta = 0.7
    thermal = PurifiedMPS(hamiltonian, D=2).run(beta, step=0.1)
    log_partition = thermal.log_partition_function
    thermal_energy = thermal.thermal_energy

    thermal.evolve(time=0.3, step=0.1)

    assert thermal.time == 0.3
    assert len(thermal.real_time_history) == 3
    np.testing.assert_allclose(thermal.energy, thermal_energy, atol=2.0e-14)
    np.testing.assert_allclose(thermal.thermal_energy, thermal_energy, atol=0.0)
    np.testing.assert_allclose(
        thermal.log_partition_function,
        log_partition,
        atol=0.0,
    )
    assert max(row["norm_error"] for row in thermal.real_time_history) < 1.0e-13


def test_real_time_quench_matches_exact_thermal_dynamics():
    site = SpinHalfSite()
    initial_hamiltonian = Hamiltonian((site,))
    initial_hamiltonian.add_product(1.0, (0, "Z"))
    quench_hamiltonian = Hamiltonian((site,))
    quench_hamiltonian.add_product(1.0, (0, "X"))
    beta = 0.7
    final_time = 0.4
    thermal = PurifiedMPS(initial_hamiltonian, D=2).run(beta, step=0.1)
    prepared_energy = thermal.thermal_energy

    thermal.evolve(
        time=final_time,
        step=0.05,
        hamiltonian=quench_hamiltonian,
        observables={"Z": initial_hamiltonian},
    )

    expected_z = -np.tanh(beta) * np.cos(2.0 * final_time)
    np.testing.assert_allclose(thermal.energy, 0.0, atol=2.0e-14)
    np.testing.assert_allclose(
        thermal.real_time_history[-1]["observables"]["Z"],
        expected_z,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(thermal.thermal_energy, prepared_energy, atol=0.0)
    with pytest.raises(ValueError, match="cannot continue after real-time"):
        thermal.run(beta + 0.1)


def test_two_spin_tdvp2_purification_matches_exact_interacting_result():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site, site))
    hamiltonian.add_product(-1.0, (0, "Z"), (1, "Z"))
    hamiltonian.add_product(-0.7, (0, "X"))
    hamiltonian.add_product(-0.7, (1, "X"))
    beta = 0.4

    thermal = PurifiedMPS(hamiltonian, D=4, cutoff=0.0).run(
        beta,
        step=0.05,
    )
    dense_h = hamiltonian.to_dense()
    density = expm(-beta * dense_h)
    partition = np.trace(density)
    exact_energy = np.trace(density @ dense_h) / partition

    np.testing.assert_allclose(thermal.energy, exact_energy, atol=3.0e-13)
    np.testing.assert_allclose(
        thermal.log_partition_function,
        np.log(partition),
        atol=3.0e-13,
    )
    assert thermal.bond_dims == (1, 4, 1)
    assert all(row["truncation_error"] == 0.0 for row in thermal.history)


def test_u1_purification_uses_block_two_site_tdvp_and_matches_exact_result():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site, site))
    hamiltonian.add_product(-1.0, (0, "Sp"), (1, "Sm"), add_hc=True)
    hamiltonian.add_product(0.3, (0, "Z"))
    hamiltonian.add_product(-0.2, (1, "Z"))
    beta = 0.4

    thermal = PurifiedMPS(
        hamiltonian,
        D=16,
        cutoff=0.0,
        symmetry="U1",
    ).run(beta, step=0.05)
    dense_h = hamiltonian.to_dense()
    density = expm(-beta * dense_h)
    partition = np.trace(density)

    assert hasattr(thermal.state.factors[0], "qns")
    assert thermal.symmetry == "U1"
    assert thermal.local_sectors == [0, 2, -2, 0]
    assert thermal.target_sector == 0
    assert len(thermal.history) == 8
    assert all(row["backend"] == "block-sparse" for row in thermal.history)
    assert thermal.bond_dims == (1, 4, 1)
    np.testing.assert_allclose(
        thermal.energy,
        np.trace(density @ dense_h) / partition,
        atol=3.0e-13,
    )
    np.testing.assert_allclose(
        thermal.log_partition_function,
        np.log(partition),
        atol=3.0e-13,
    )
    prepared_energy = thermal.energy
    thermal.evolve(time=0.1, step=0.05)
    assert all(
        row["backend"] == "block-sparse"
        for row in thermal.real_time_history
    )
    np.testing.assert_allclose(thermal.energy, prepared_energy, atol=3.0e-13)


def test_u1_purification_rejects_a_nonconserving_hamiltonian():
    site = SpinHalfSite()
    hamiltonian = Hamiltonian((site,))
    hamiltonian.add_product(1.0, (0, "X"))

    with pytest.raises(ValueError, match="charge-conserving Hamiltonian"):
        PurifiedMPS(hamiltonian, symmetry="U1")
