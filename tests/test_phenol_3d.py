import numpy as np
import runpy

from pyqed.dvr import ExponentialDVR, SineDVR
from pyqed.models.phenol import Phenol, Phenol3D, dpes1


def _small_model_and_dvrs():
    probe = Phenol3D([Phenol3D.r_eq], [Phenol3D.theta_eq], [0.0])
    r_dvr = SineDVR(1.4, 2.4, 3, mass=probe.radial_mass)
    bend_dvr = SineDVR(
        Phenol3D.theta_eq - 0.3,
        Phenol3D.theta_eq + 0.3,
        3,
        mass=probe.bend_inertia,
    )
    torsion_dvr = ExponentialDVR(
        1, L=2.0 * np.pi, x0=np.pi / 3.0, mass=probe.torsional_inertia
    )
    model = Phenol3D(r_dvr.x, bend_dvr.x, torsion_dvr.x)
    return model, (r_dvr, bend_dvr, torsion_dvr)


def test_phenol_3d_matches_published_two_coordinate_cut_at_equilibrium_bend():
    r = np.array([1.7, 2.1])
    torsion = np.array([-0.4, 0.0, 0.7])
    model = Phenol3D(r, [Phenol3D.theta_eq], torsion)

    actual = model.buildV()[:, 0]
    expected = Phenol(r, torsion).buildV()

    np.testing.assert_allclose(actual, expected, atol=1.0e-14)


def test_phenol_diabatic_model_is_two_pi_periodic_in_torsion():
    for phi in (-1.2, -0.1, 0.8):
        np.testing.assert_allclose(dpes1(2.0, phi), dpes1(2.0, phi + 2.0 * np.pi), atol=1.0e-14)


def test_phenol_3d_overlap_ldr_matches_diabatic_representation():
    model, dvrs = _small_model_and_dvrs()
    adiabatic = model.hamiltonian(dvrs, representation="adiabatic")
    diabatic = model.hamiltonian(dvrs, representation="diabatic")
    frames = adiabatic["frames"].reshape(-1, model.nstates, model.nstates)

    rng = np.random.default_rng(11)
    psi_ad = rng.normal(size=(len(frames), model.nstates))
    psi_ad = psi_ad + 1j * rng.normal(size=psi_ad.shape)
    psi_diab = np.einsum("gda,ga->gd", frames, psi_ad, optimize=True)

    lhs = diabatic["hamiltonian"] @ psi_diab.reshape(-1)
    rhs_ad = adiabatic["hamiltonian"] @ psi_ad.reshape(-1)
    rhs = np.einsum(
        "gda,ga->gd",
        frames,
        rhs_ad.reshape(len(frames), model.nstates),
        optimize=True,
    ).reshape(-1)

    np.testing.assert_allclose(lhs, rhs, atol=2.0e-13)
    np.testing.assert_allclose(
        adiabatic["hamiltonian"].toarray(),
        adiabatic["hamiltonian"].getH().toarray(),
        atol=1.0e-14,
    )


def test_phenol_3d_rejects_wrong_dvr_shape():
    model, dvrs = _small_model_and_dvrs()
    wrong_r = SineDVR(1.4, 2.4, 4, mass=model.radial_mass)

    with np.testing.assert_raises_regex(ValueError, "does not match model shape"):
        model.hamiltonian((wrong_r, dvrs[1], dvrs[2]))


def test_phenol_stage1_geometry_uses_the_published_reactive_coordinates():
    namespace = runpy.run_path("examples/namd/phenol_staged_mace_ftt_ttldr.py")
    geometry = namespace["phenol_geometry"]
    planar = geometry((1.0, 0.0))
    twisted = geometry((1.0, np.pi / 2.0))
    assert planar.shape == (13, 3)
    np.testing.assert_allclose(planar[:, 2], 0.0, atol=1.0e-14)
    oxygen = planar[6]
    np.testing.assert_allclose(np.linalg.norm(planar[7] - oxygen), 1.0)
    np.testing.assert_allclose(np.linalg.norm(twisted[7] - twisted[6]), 1.0)
    coh = planar[0] - oxygen
    oh = planar[7] - oxygen
    angle = np.arccos(np.dot(coh, oh) / np.linalg.norm(coh) / np.linalg.norm(oh))
    np.testing.assert_allclose(np.rad2deg(angle), 108.8, atol=1.0e-12)
    assert twisted[7, 2] > 0.0


def test_phenol_stage1_sobol_sampling_is_nested_and_reference_is_symmetric():
    namespace = runpy.run_path("examples/namd/phenol_staged_mace_ftt_ttldr.py")
    bounds = ((0.8, 3.0), (-np.pi, np.pi))
    small = namespace["sobol_coordinates"](17, bounds, 11)
    large = namespace["sobol_coordinates"](33, bounds, 11)
    np.testing.assert_allclose(small, large[:17])
    plus = namespace["reference_dpem"]((1.2, 0.37))
    minus = namespace["reference_dpem"]((1.2, -0.37))
    np.testing.assert_allclose(np.diag(plus), np.diag(minus), atol=1.0e-14)
    np.testing.assert_allclose(plus[[0, 1], [1, 2]], -minus[[0, 1], [1, 2]], atol=1.0e-14)


def test_phenol_periodic_dvr_contains_the_planar_symmetry_cut():
    namespace = runpy.run_path("examples/namd/phenol_staged_mace_ftt_ttldr.py")
    axes, _dvrs = namespace["build_dvrs"](7, 9)
    assert np.any(np.isclose(axes[1], 0.0, atol=1.0e-14))


def test_phenol_reactive_design_is_exactly_reflection_paired():
    namespace = runpy.run_path("examples/namd/phenol_staged_mace_ftt_ttldr.py")
    values = namespace["reflection_paired_training_coordinates"](
        32, ((0.9, 3.2), (-np.pi, np.pi)), 13, planar_points=5
    )
    positive, negative = values[:16], values[16:32]
    np.testing.assert_allclose(positive[:, 0], negative[:, 0])
    np.testing.assert_allclose(positive[:, 1], -negative[:, 1])
    np.testing.assert_allclose(values[32:, 1], 0.0)


def test_phenol_parity_reduced_target_reconstructs_the_dpem():
    namespace = runpy.run_path("examples/namd/phenol_staged_mace_ftt_ttldr.py")
    coordinates = np.asarray([
        [1.0, -0.7], [1.2, 0.0], [1.8, 0.4], [2.5, np.pi - 0.2]
    ])
    reference = namespace["reference_dpem"](coordinates)
    coefficients = namespace["parity_reduce"](coordinates, reference)
    reconstructed = namespace["parity_expand"](coordinates, coefficients)
    np.testing.assert_allclose(reconstructed, reference, atol=1.0e-12)
    np.testing.assert_allclose(coefficients[:, 0, 1].imag, 0.0, atol=1.0e-14)
    np.testing.assert_allclose(coefficients[:, 1, 2].imag, 0.0, atol=1.0e-14)
    independent = namespace["independent_coefficients"](coefficients)
    reduced_reconstructed = np.einsum(
        "nk,kij->nij", independent, namespace["PHENOL_MATRIX_BASIS"]
    )
    np.testing.assert_allclose(reduced_reconstructed, coefficients, atol=1.0e-12)


def test_phenol_five_dimensional_chart_has_mass_orthonormal_modes_and_exact_cut():
    from pyqed.models.phenol import dpes1
    from pyqed.models.phenol_coordinates import (
        PHENOL_MASSES,
        PhenolReactiveChart,
        mode_reflection_parity,
    )
    from pyqed.units import au2angstrom

    chart = PhenolReactiveChart()
    assert chart.geometry(chart.equilibrium).shape == (13, 3)
    gram = np.einsum(
        "kia,lia,i->kl", chart.modes, chart.modes, PHENOL_MASSES
    )
    np.testing.assert_allclose(gram, np.eye(2), atol=1.0e-12)
    np.testing.assert_allclose(
        [mode_reflection_parity(mode) for mode in chart.modes], (-1.0, 1.0)
    )
    reference = dpes1(chart.equilibrium[0] / au2angstrom, 0.0)
    np.testing.assert_allclose(chart.model_dpem(chart.equilibrium), reference)

    coordinate = chart.equilibrium + np.asarray((0.2, 0.3, 0.04, 0.25, -0.08))
    reflected = coordinate * np.asarray((1.0, -1.0, 1.0, -1.0, 1.0))
    np.testing.assert_allclose(
        chart.geometry(reflected),
        chart.geometry(coordinate) @ np.diag((1.0, 1.0, -1.0)),
        atol=1.0e-12,
    )


def test_phenol_active_mode_selector_uses_wilson_symmetry_and_frequency():
    from pyqed.models.phenol_coordinates import (
        PHENOL_MASSES,
        phenol_template_modes,
        select_phenol_active_modes,
    )

    templates = phenol_template_modes()
    nuisance = np.roll(templates[0], 1, axis=0)
    frequencies = np.asarray((310.0, 250.0, 1690.0))
    modes = np.asarray((nuisance, templates[0], templates[1]))
    selected, diagnostics = select_phenol_active_modes(frequencies, modes)

    gram = np.einsum("kia,lia,i->kl", selected, selected, PHENOL_MASSES)
    np.testing.assert_allclose(gram, np.eye(2), atol=1.0e-12)
    assert [item["label"] for item in diagnostics] == ["16a", "8a"]
    assert [item["index"] for item in diagnostics] == [1, 2]
    reflection = np.diag((1.0, 1.0, -1.0))
    np.testing.assert_allclose(
        selected[0] @ reflection, -selected[0], atol=1.0e-14
    )
    np.testing.assert_allclose(
        selected[1] @ reflection, selected[1], atol=1.0e-14
    )


def test_phenol_overlap_tree_alignment_recovers_a_smooth_hamiltonian():
    namespace = runpy.run_path("examples/namd/phenol_abinitio_active.py")
    energies = np.asarray([[0.0, 1.0], [0.1, 1.1], [0.2, 1.2]])
    angle = 0.3
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    pairs = np.asarray([[0, 1], [1, 2]])
    overlaps = np.asarray([rotation, rotation])
    matrices, gauges, shift, residuals = namespace["align_hamiltonians"](
        energies, pairs, overlaps, np.ones(2)
    )
    assert matrices.shape == (3, 2, 2)
    assert gauges.shape == matrices.shape
    assert shift == 0.0
    assert np.max(residuals) < 1.0e-12
    np.testing.assert_allclose(matrices, matrices.swapaxes(-1, -2).conj())
