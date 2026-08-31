import numpy as np

from pyqed.lgt import (
    AlternatingWilsonDVRMPO,
    OpenSineWilsonDVRMPO,
    QuantumSchwingerDVR,
    WilsonDVRMPO,
)
from pyqed.mps import (
    DMRG,
    MPO,
    MPS,
    TDMPS,
    compress_symmetric_mps,
    compress_symmetric_mpo,
    dense_to_symmetric_mpo,
    symmetric_mpo_to_dense,
)
from pyqed.mps.mps import apply_mpo_symmetric


def test_wilson_dvr_mpo_matches_exact_physical_basis():
    exact = QuantumSchwingerDVR(3, 10.0, flux_cutoff=1)
    exact_hamiltonian = exact.build_hamiltonian().toarray()
    builder = WilsonDVRMPO(
        3,
        10.0,
        flux_cutoff=1,
        gauss_penalty=7.0,
    )
    dense = builder.build_mpo().to_dense()
    indices = builder.physical_product_indices(exact)
    projected = dense[np.ix_(indices, indices)]
    np.testing.assert_allclose(projected, exact_hamiltonian, atol=1.0e-12)
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-12)

    gauss = builder.build_gauss_mpo().to_dense()
    np.testing.assert_allclose(gauss[np.ix_(indices, indices)], 0.0, atol=1.0e-12)


def test_small_wilson_dvr_mpo_dmrg_matches_ed():
    exact = QuantumSchwingerDVR(3, 10.0, flux_cutoff=1).run(nroots=4)
    builder = WilsonDVRMPO(
        3,
        10.0,
        flux_cutoff=1,
        gauss_penalty=25.0,
    )
    hamiltonian = builder.build_mpo().compress(64)
    gauss = builder.build_gauss_mpo().compress(64)
    initial = builder.product_mps([1, 1, 1], [0, 0, 0])
    solver = DMRG(
        hamiltonian,
        D=48,
        init_guess=initial,
        nsweeps=8,
        not_conv_err=False,
        sweep_tol=1.0e-9,
        davidson_tol=1.0e-10,
        noise=1.0e-5,
        performance="dense",
    ).run()
    assert solver.converged
    np.testing.assert_allclose(
        solver.e_tot,
        exact.energies[0],
        atol=1.0e-10,
    )
    assert abs(solver.ground_state.expectation(gauss)) < 1.0e-11


def test_alternating_wilson_dvr_mpo_matches_exact_physical_basis():
    exact = QuantumSchwingerDVR(3, 10.0, flux_cutoff=1)
    exact_hamiltonian = exact.build_hamiltonian().toarray()
    builder = AlternatingWilsonDVRMPO(3, 10.0, flux_cutoff=1)
    dense = builder.build_mpo().to_dense()
    indices = builder.physical_product_indices(exact)
    np.testing.assert_allclose(
        dense[np.ix_(indices, indices)],
        exact_hamiltonian,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-12)

    exact.build_channel_operators()
    vector = builder.build_vector_mpo().to_dense()
    scalar = builder.build_scalar_mpo().to_dense()
    np.testing.assert_allclose(
        vector[np.ix_(indices, indices)],
        exact.vector_operator.toarray(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scalar[np.ix_(indices, indices)],
        exact.scalar_operator.toarray(),
        atol=1.0e-12,
    )


def test_gauss_symmetric_alternating_dmrg_matches_ed_without_penalty():
    exact = QuantumSchwingerDVR(3, 10.0, flux_cutoff=1).run(nroots=1)
    builder = AlternatingWilsonDVRMPO(3, 10.0, flux_cutoff=1)
    maps, target, manager = builder.gauss_symmetry()
    hamiltonian = MPO(
        dense_to_symmetric_mpo(
            builder.build_mpo().factors,
            maps,
            native_site_storage=True,
        )
    )
    initial = builder.gauss_seed_mps(
        bond_dim=16,
        seed=7,
        native_site_storage=True,
    )
    solver = DMRG(
        hamiltonian,
        D=16,
        init_guess=initial,
        nsweeps=6,
        symmetry=True,
        target_qn=target,
        sym_mgr=manager,
        site_qn_maps=maps,
        not_conv_err=False,
        sweep_tol=1.0e-10,
        davidson_tol=1.0e-11,
        noise=1.0e-6,
        performance="symmetric",
    ).run()
    assert solver.converged
    np.testing.assert_allclose(solver.e_tot, exact.energies[0], atol=1.0e-11)


def test_gauss_seed_mps_scales_without_determinant_enumeration():
    builder = AlternatingWilsonDVRMPO(13, 10.0, flux_cutoff=4)
    state = builder.gauss_seed_mps(
        bond_dim=128,
        seed=7,
        native_site_storage=True,
    )

    assert len(state.factors) == 26
    assert state.gauss_bond_dimensions[0] == 1
    assert state.gauss_bond_dimensions[-1] == 1
    assert max(state.gauss_bond_dimensions) <= 128
    assert state.norm_squared() > 0.0


def test_gauss_symmetric_mpo_compression_is_exact_and_charge_resolved():
    builder = AlternatingWilsonDVRMPO(3, 10.0, mass=0.2, flux_cutoff=1)
    raw = builder.build_mpo()
    maps, _target, _manager = builder.gauss_symmetry()
    symmetric = MPO(
        dense_to_symmetric_mpo(
            raw.factors,
            maps,
            native_site_storage=True,
        )
    )
    compressed = compress_symmetric_mpo(symmetric, rtol=1.0e-12)

    assert compressed.bond_orders() == [6, 10, 10, 10, 4, 1]
    dense_compressed = symmetric_mpo_to_dense(compressed, maps).to_dense()
    np.testing.assert_allclose(dense_compressed, raw.to_dense(), atol=2.0e-12)
    np.testing.assert_allclose(
        dense_compressed,
        dense_compressed.conj().T,
        atol=2.0e-12,
    )


def test_gauss_symmetric_mps_compression_preserves_channel_source():
    builder = AlternatingWilsonDVRMPO(3, 10.0, flux_cutoff=1)
    maps, _target, _manager = builder.gauss_symmetry()
    state = builder.gauss_seed_mps(
        bond_dim=16,
        seed=9,
        native_site_storage=True,
    ).normalize()
    operator = dense_to_symmetric_mpo(
        builder.build_scalar_mpo().factors,
        maps,
        native_site_storage=False,
    )
    source = MPS(
        apply_mpo_symmetric(operator, state.factors),
        labels=["lv", "rv", "p"],
    )
    compressed = compress_symmetric_mps(source)

    assert max(compressed.bond_orders()) <= max(source.bond_orders())
    np.testing.assert_allclose(
        compressed.norm_squared(),
        source.norm_squared(),
        atol=2.0e-12,
    )
    np.testing.assert_allclose(
        TDMPS.state_overlap(source, compressed),
        source.norm_squared(),
        atol=2.0e-12,
    )


def test_open_sine_cosine_dirac_spectrum_and_boundaries():
    builder = OpenSineWilsonDVRMPO(
        6,
        7.0,
        mass=0.3,
        flux_cutoff=1,
    )
    np.testing.assert_allclose(
        builder.cosine_transform.T @ builder.cosine_transform,
        np.eye(builder.npts),
        atol=2.0e-14,
    )
    np.testing.assert_allclose(
        builder.sine_transform.T @ builder.sine_transform,
        np.eye(builder.npts),
        atol=2.0e-14,
    )
    expected = np.sort(
        np.concatenate(
            (
                -np.sqrt(builder.momenta**2 + builder.mass**2),
                np.sqrt(builder.momenta**2 + builder.mass**2),
            )
        )
    )
    np.testing.assert_allclose(
        np.linalg.eigvalsh(builder.one_particle_matrix()),
        expected,
        atol=2.0e-14,
    )
    modes = np.arange(builder.npts) + 0.5
    np.testing.assert_allclose(np.sin(modes * 0.0), 0.0, atol=1.0e-15)
    np.testing.assert_allclose(np.cos(modes * np.pi), 0.0, atol=3.0e-15)


def test_open_sine_wilson_automaton_matches_reference_mpo():
    builder = OpenSineWilsonDVRMPO(
        3,
        5.0,
        mass=0.2,
        flux_cutoff=1,
    )
    automaton = builder.build_mpo().to_dense()
    reference = builder.build_reference_mpo().to_dense()
    np.testing.assert_allclose(automaton, reference, atol=2.0e-14)
    np.testing.assert_allclose(automaton, automaton.conj().T, atol=2.0e-14)
    assert max(builder.mpo.bond_orders()) == 4 * (builder.npts - 1) + 2
    assert max(builder.build_vector_mpo().bond_orders()) == 2
    assert max(builder.build_scalar_mpo().bond_orders()) == 2


def test_open_sine_wilson_exact_gauss_mpo_and_seed():
    builder = OpenSineWilsonDVRMPO(4, 8.0, flux_cutoff=2)
    maps, target, manager = builder.gauss_symmetry()
    symmetric = MPO(
        dense_to_symmetric_mpo(
            builder.build_mpo().factors,
            maps,
            native_site_storage=True,
        )
    )
    compressed = compress_symmetric_mpo(symmetric, rtol=1.0e-12)
    seed = builder.gauss_seed_mps(
        bond_dim=32,
        seed=11,
        native_site_storage=True,
    )
    assert len(seed.factors) == 2 * builder.npts - 1
    assert seed.gauss_bond_dimensions[0] == 1
    assert seed.gauss_bond_dimensions[-1] == 1
    assert max(seed.gauss_bond_dimensions) <= 32
    np.testing.assert_allclose(seed.norm_squared(), 1.0, atol=2.0e-12)
    assert max(compressed.bond_orders()) <= max(symmetric.bond_orders())
    assert len(target) == builder.npts
    assert manager is not None
