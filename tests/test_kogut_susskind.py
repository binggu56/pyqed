import numpy as np

from pyqed.lgt import KogutSusskindED, KogutSusskindMPO
from pyqed.mps import DMRG, MPO, dense_to_symmetric_mpo


def test_kogut_susskind_mpo_and_channels_match_physical_ed_basis():
    exact = KogutSusskindED(
        4,
        4.0,
        mass=0.2,
        flux_cutoff=1,
        background_field=0.1,
    )
    builder = KogutSusskindMPO(
        4,
        4.0,
        mass=0.2,
        flux_cutoff=1,
        background_field=0.1,
    )
    indices = builder.physical_product_indices(exact)
    dense = builder.build_mpo().to_dense()
    np.testing.assert_allclose(
        dense[np.ix_(indices, indices)],
        exact.build_hamiltonian().toarray(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(dense, dense.conj().T, atol=1.0e-12)

    vector = builder.build_vector_mpo().to_dense()
    scalar = builder.build_scalar_mpo().to_dense()
    np.testing.assert_allclose(
        vector[np.ix_(indices, indices)],
        exact.build_vector_operator().toarray(),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scalar[np.ix_(indices, indices)],
        exact.build_scalar_operator().toarray(),
        atol=1.0e-12,
    )


def test_kogut_susskind_gauss_operator_vanishes_for_nonzero_boundaries():
    exact = KogutSusskindED(
        4,
        4.0,
        flux_cutoff=2,
        left_flux=1,
        right_flux=0,
    )
    builder = KogutSusskindMPO(
        4,
        4.0,
        flux_cutoff=2,
        left_flux=1,
        right_flux=0,
    )
    indices = builder.physical_product_indices(exact)
    gauss = builder.build_gauss_mpo().to_dense()
    np.testing.assert_allclose(gauss[np.ix_(indices, indices)], 0.0, atol=1.0e-12)


def test_periodic_kogut_susskind_ed_keeps_loop_flux_and_gauss_law():
    exact = KogutSusskindED(
        6,
        6.0,
        mass=0.1,
        flux_cutoff=2,
        background_field=0.2,
        boundary="periodic",
    )
    hamiltonian = exact.build_hamiltonian()
    np.testing.assert_allclose(
        hamiltonian.toarray(),
        hamiltonian.toarray().conj().T,
        atol=1.0e-12,
    )
    for bits, flux in zip(exact.basis_bits, exact.basis_flux):
        np.testing.assert_array_equal(
            np.roll(flux, 1) - flux - exact.charges(bits),
            0,
        )
    assert exact.vector_momentum == 2.0 * np.pi / exact.length


def test_gauss_symmetric_kogut_susskind_dmrg_matches_ed():
    exact = KogutSusskindED(4, 4.0, mass=0.2, flux_cutoff=1).run(nroots=1)
    builder = KogutSusskindMPO(4, 4.0, mass=0.2, flux_cutoff=1)
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
        noise=1.0e-7,
        performance="symmetric",
    ).run()
    assert solver.converged
    np.testing.assert_allclose(solver.e_tot, exact.energies[0], atol=1.0e-11)
