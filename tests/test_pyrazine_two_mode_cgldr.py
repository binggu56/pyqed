import numpy as np

from examples.ldr.pyrazine_two_mode_cgldr import (
    build_cgldr,
    build_dvr,
    build_full_hamiltonian,
    electronic_hamiltonian,
    run_benchmark,
)


def test_pyrazine_cgldr_quadratic_reconstructs_exact_potential():
    dvr = build_dvr(npts=(8, 6), domains=((-4.0, 4.0),) * 2)
    dynamics = build_cgldr(dvr, max_rank=8)
    data = dynamics.electronic_data

    for sampled_index in (1, 5):
        tuning = dvr.x[0][sampled_index]
        for coupling in (-0.7, 0.4):
            reconstructed = np.diag(data.energies[sampled_index]).astype(
                complex
            )
            reconstructed += (
                coupling * data.hamiltonian_gradients[sampled_index, 0]
            )
            reconstructed += (
                0.5
                * coupling**2
                * data.hamiltonian_hessians[sampled_index, 0, 0]
            )
            np.testing.assert_allclose(
                reconstructed,
                electronic_hamiltonian(tuning, coupling),
                atol=1.0e-14,
            )


def test_pyrazine_full_hamiltonian_is_hermitian():
    dvr = build_dvr(npts=(8, 6), domains=((-4.0, 4.0),) * 2)
    hamiltonian = build_full_hamiltonian(dvr)

    np.testing.assert_allclose(
        hamiltonian.toarray(),
        hamiltonian.toarray().conj().T,
        atol=1.0e-14,
    )


def test_pyrazine_cgldr_matches_short_exact_dynamics():
    results = run_benchmark(
        npts=(8, 6),
        domains=((-4.0, 4.0),) * 2,
        time_step=0.1,
        steps=2,
        output_every=1,
        max_rank=8,
    )

    assert np.max(results["total_variation"]) < 1.0e-8
    np.testing.assert_allclose(
        results["cg_populations"],
        results["full_populations"],
        atol=1.0e-8,
    )
