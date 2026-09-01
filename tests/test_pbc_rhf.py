import os

import numpy as np
import pytest

from pyqed.qchem.basis import ERI, S, T, point_charge
from pyqed.qchem.pbc.ewald import (
    ao_pair_ft_matrix_s,
    ewald_nuclear_gradient,
    ewald_nuclear_hessian,
    ewald_nuclear_repulsion,
    ewald_nuclear_repulsion_1d_inf_vacuum,
    gaussian_pair_ft,
    reciprocal_vectors,
    short_range_eri,
    short_range_eri_s,
    short_range_point_charge,
)
from pyqed.qchem.pbc import (
    Cell,
    Chain,
    EwaldRHF,
    KRHF,
    RHF,
    commensurate_gdf_q_derivative,
)
from pyqed.pbc.gw import (
    KPointTransitionSpace,
    commensurate_gdf_bare_tda_kernel_derivative,
    commensurate_gdf_screened_tda_kernel_derivative,
    diagonal_correlation_self_energy,
    diagonal_g0w0,
    gamma_gdf_bare_tda_kernel_derivative,
    gamma_gdf_diagonal_self_energy_derivative,
    gamma_gdf_g0w0_energy_derivative,
    gamma_gdf_screened_tda_kernel_derivative,
    gamma_tda_electron_phonon_coupling,
    periodic_bse_matrices,
    periodic_tda_operator,
)
from pyqed.units import amu_to_au


def test_reciprocal_vectors_accept_anisotropic_index_bounds():
    vectors = reciprocal_vectors(
        np.eye(3),
        (1, 2, 0),
        include_zero=True,
    )
    indices = np.asarray([entry[:3] for entry in vectors], dtype=int)

    assert indices.shape == (15, 3)
    np.testing.assert_array_equal(np.min(indices, axis=0), [-1, -2, 0])
    np.testing.assert_array_equal(np.max(indices, axis=0), [1, 2, 0])


def test_ewald_nuclear_gradient_matches_energy_finite_difference():
    charges = np.asarray([1.0, 2.0])
    coords = np.asarray([[0.2, 0.4, 0.3], [1.6, 1.1, 0.8]])
    lattice = np.diag([5.0, 5.5, 6.0])
    options = {"eta": 0.7, "real_cut": 2, "recip_cut": 3}
    analytic = ewald_nuclear_gradient(charges, coords, lattice, **options)
    numerical = np.zeros_like(coords)
    step = 1.0e-5
    for atom in range(len(charges)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            e_plus = ewald_nuclear_repulsion(
                charges,
                plus,
                lattice,
                neutralizing_background=True,
                **options,
            )
            e_minus = ewald_nuclear_repulsion(
                charges,
                minus,
                lattice,
                neutralizing_background=True,
                **options,
            )
            numerical[atom, axis] = (e_plus - e_minus) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, atol=2.0e-9, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=1.0e-12)


def test_ewald_nuclear_hessian_matches_gradient_finite_difference():
    charges = np.asarray([1.0, 2.0])
    coords = np.asarray([[0.2, 0.4, 0.3], [1.6, 1.1, 0.8]])
    lattice = np.diag([5.0, 5.5, 6.0])
    options = {"eta": 0.7, "real_cut": 2, "recip_cut": 3}
    analytic = ewald_nuclear_hessian(charges, coords, lattice, **options)
    numerical = np.zeros_like(analytic)
    step = 1.0e-5
    for atom in range(len(charges)):
        for axis in range(3):
            plus = coords.copy()
            minus = coords.copy()
            plus[atom, axis] += step
            minus[atom, axis] -= step
            gradient_plus = ewald_nuclear_gradient(
                charges,
                plus,
                lattice,
                **options,
            )
            gradient_minus = ewald_nuclear_gradient(
                charges,
                minus,
                lattice,
                **options,
            )
            numerical[:, :, atom, axis] = (
                gradient_plus - gradient_minus
            ) / (2.0 * step)

    np.testing.assert_allclose(analytic, numerical, atol=3.0e-9, rtol=0.0)
    np.testing.assert_allclose(
        analytic,
        analytic.transpose(2, 3, 0, 1),
        atol=1.0e-12,
    )
    np.testing.assert_allclose(np.sum(analytic, axis=2), 0.0, atol=2.0e-12)


def _all_electron_h2_ewald_krhf(coords, jk_builder="ewald", kpoint=None):
    atom = [("H", tuple(position)) for position in np.asarray(coords)]
    cell = Cell(
        atom=atom,
        a=np.eye(3) * 6.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    k_request = (
        {"nk": 1}
        if kpoint is None
        else {"kpts": np.asarray(kpoint, dtype=float).reshape(1, 3)}
    )
    return cell.KRHF(
        **k_request,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder=jk_builder,
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).run(max_cycle=60, conv_tol=1.0e-11, conv_tol_dm=1.0e-9)


def _gth_h2_reciprocal_krhf(coords):
    pseudo = {
        "H": [[1], 0.35, 2, [-3.2, 0.45], 1, [0.28, 1, [[1.7]]]],
    }
    cell = Cell(
        atom=[("H", tuple(position)) for position in np.asarray(coords)],
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        pseudo=pseudo,
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    return cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=2,
        recip_cut=2,
        pseudo_cut=0,
        one_body_nuclear_cut=1,
        jk_builder="reciprocal",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        pseudo_local_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _full_gdf_h2_krhf(coords):
    cell = Cell(
        atom=[("H", tuple(position)) for position in np.asarray(coords)],
        a=np.diag([5.0, 5.4, 5.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        reciprocal_kernel="full",
        recip_cut=2,
        pair_cut=0,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    )
    return mf.run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _full_gdf_two_k_h2_krhf():
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    return cell.KRHF(
        nk=(2, 1, 1),
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="sto-3g",
        reciprocal_kernel="full",
        recip_cut=2,
        pair_cut=0,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


def _full_gdf_lih_krhf(coords):
    cell = Cell(
        atom=[("Li", tuple(coords[0])), ("H", tuple(coords[1]))],
        a=np.diag([6.5, 6.8, 7.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        reciprocal_kernel="full",
        recip_cut=2,
        pair_cut=0,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    )
    return mf.run(max_cycle=80, conv_tol=1.0e-11, conv_tol_dm=1.0e-9)


def _range_separated_gdf_h2_krhf(
    coords,
    *,
    auxbasis="sto-3g",
    omega=0.7,
    recip_cut=3,
    image_cut=1,
    pair_partition="off",
    aux_partition="off",
    smooth_exponent_cutoff=None,
    reciprocal_only_pair_mask=None,
):
    cell = Cell(
        atom=[("H", tuple(position)) for position in np.asarray(coords)],
        a=np.diag([5.0, 5.4, 5.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=1,
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis=auxbasis,
        reciprocal_kernel="range_separated",
        omega=omega,
        recip_cut=recip_cut,
        pair_cut=0,
        pair_screen_tol=0.0,
        image_cut=image_cut,
        metric_tol=1.0e-12,
    )
    mf.gdf_rs_pair_partition = pair_partition
    mf.gdf_rs_aux_partition = aux_partition
    if reciprocal_only_pair_mask is not None:
        mf.gdf_reciprocal_only_pair_mask = np.asarray(
            reciprocal_only_pair_mask,
            dtype=np.bool_,
        )
    if smooth_exponent_cutoff is not None:
        mf.gdf_smooth_exponent_cutoff = float(smooth_exponent_cutoff)
    mf.gdf_short_range_screen_tol = 0.0
    return mf.run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)


@pytest.mark.parametrize("jk_builder", ["ewald", "reciprocal"])
def test_native_gamma_krhf_gradient_matches_total_energy_difference(jk_builder):
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder=jk_builder)
    assert mf.converged
    analytic = mf.nuc_grad_method().kernel()

    step = 2.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    mf_plus = _all_electron_h2_ewald_krhf(plus, jk_builder=jk_builder)
    mf_minus = _all_electron_h2_ewald_krhf(minus, jk_builder=jk_builder)
    assert mf_plus.converged and mf_minus.converged
    numerical = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-5, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-8)
    np.testing.assert_allclose(analytic[:, 1:], 0.0, atol=2.0e-8)


def test_one_k_twist_krhf_gradient_matches_total_energy_difference():
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    twist = np.asarray([0.19, 0.0, 0.0])
    mf = _all_electron_h2_ewald_krhf(
        coords,
        jk_builder="reciprocal",
        kpoint=twist,
    )
    analytic = mf.nuc_grad_method().kernel()

    step = 2.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    energy_plus = _all_electron_h2_ewald_krhf(
        plus,
        jk_builder="reciprocal",
        kpoint=twist,
    ).e_tot
    energy_minus = _all_electron_h2_ewald_krhf(
        minus,
        jk_builder="reciprocal",
        kpoint=twist,
    ).e_tot
    numerical = (energy_plus - energy_minus) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-12)


def test_cphf_relaxed_gamma_krhf_hessian_matches_gradient_difference():
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    hessian = mf.Hessian()
    step = 2.0e-4
    analytic = hessian.kernel(
        step=step,
        symmetrize=False,
        enforce_acoustic_sum_rule=False,
    )

    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    gradient_plus = _all_electron_h2_ewald_krhf(
        plus,
        jk_builder="reciprocal",
    ).nuc_grad_method().kernel()
    gradient_minus = _all_electron_h2_ewald_krhf(
        minus,
        jk_builder="reciprocal",
    ).nuc_grad_method().kernel()
    numerical_column = (gradient_plus - gradient_minus).reshape(-1) / (2.0 * step)

    np.testing.assert_allclose(analytic[:, 0], numerical_column, atol=3.0e-8)
    np.testing.assert_allclose(analytic, analytic.T, atol=2.0e-10)
    assert hessian.second_derivative_backend == "analytic"
    assert hessian.response.converged
    assert hessian.response.residual_norm < 1.0e-10
    assert hessian.first_order_density.shape == (2, 3, 2, 2)
    assert hessian.frequencies().shape == (6,)


def test_analytic_periodic_hessian_matches_derivative_difference_backend():
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    analytic_driver = mf.Hessian()
    analytic = analytic_driver.kernel(
        second_derivative_backend="analytic",
        symmetrize=False,
        enforce_acoustic_sum_rule=False,
    )
    reference_driver = mf.Hessian()
    reference = reference_driver.kernel(
        second_derivative_backend="finite_difference",
        step=2.0e-5,
        symmetrize=False,
        enforce_acoustic_sum_rule=False,
    )

    np.testing.assert_allclose(analytic, reference, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(
        analytic_driver.explicit_second,
        reference_driver.explicit_second,
        atol=2.0e-8,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        analytic_driver.nuclear_hessian,
        reference_driver.nuclear_hessian,
        atol=3.0e-9,
        rtol=0.0,
    )


def test_compiled_pair_ft_second_derivatives_match_python_path():
    from pyqed.qchem.fourier import has_periodic_pair_ft_backend

    if not has_periodic_pair_ft_backend():
        pytest.skip("compiled periodic AO-pair Fourier backend is unavailable")
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    gradient = mf.nuc_grad_method()
    gradient._validate()
    if mf._pair_ft_block_plan is None:
        pytest.skip("compiled periodic AO-pair Fourier plan is unavailable")
    gvecs = np.asarray(
        [[0.0, 0.0, 0.0], [0.7, -0.2, 0.4], [-0.3, 0.5, 0.8]]
    )

    compiled = gradient._pair_ft_second_derivatives_many(gvecs)
    plan = mf._pair_ft_block_plan
    mf._pair_ft_block_plan = None
    try:
        reference = gradient._pair_ft_second_derivatives_many(gvecs)
    finally:
        mf._pair_ft_block_plan = plan

    np.testing.assert_allclose(compiled, reference, atol=2.0e-13, rtol=0.0)


def test_compiled_one_body_second_derivatives_match_python_path():
    from pyqed.qchem import basis as basis_module

    if not hasattr(basis_module._basis_cy, "compute_periodic_one_electron"):
        pytest.skip("compiled periodic one-electron kernel is unavailable")
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    gradient = mf.nuc_grad_method()
    gradient._validate()

    compiled = gradient._real_space_one_body_second_derivatives()
    reference = gradient._real_space_one_body_second_derivatives_python()

    assert gradient.one_body_second_derivative_backend == "compiled"
    for actual, expected in zip(compiled, reference):
        np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=0.0)


def test_compiled_one_body_first_derivatives_match_python_path():
    from pyqed.qchem import basis as basis_module

    if not hasattr(basis_module._basis_cy, "compute_periodic_one_electron"):
        pytest.skip("compiled periodic one-electron kernel is unavailable")
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    gradient = mf.nuc_grad_method()
    gradient._validate()

    compiled = gradient._real_space_one_body_derivatives()
    reference = gradient._real_space_one_body_derivatives_python()

    assert gradient.one_body_derivative_backend == "compiled"
    for actual, expected in zip(compiled, reference):
        np.testing.assert_allclose(actual, expected, atol=2.0e-13, rtol=0.0)


def test_batched_reciprocal_veff_derivatives_match_density_loop():
    coords = np.asarray([[2.3, 3.0, 3.0], [3.7, 3.0, 3.0]])
    mf = _all_electron_h2_ewald_krhf(coords, jk_builder="reciprocal")
    gradient = mf.nuc_grad_method()
    gradient._validate()
    dm = np.asarray(mf.make_rdm1(), dtype=np.complex128)
    dms = np.asarray([dm, 0.25 * dm, dm.T.conj() - 0.4 * dm])

    batched = gradient.effective_potential_derivatives_many(dms)
    reference = np.asarray(
        [gradient.effective_potential_derivatives(value) for value in dms]
    )

    np.testing.assert_allclose(batched, reference, atol=2.0e-13, rtol=0.0)


def test_native_gth_krhf_gradient_matches_total_energy_difference():
    coords = np.asarray([[2.1, 3.0, 3.1], [3.65, 3.25, 3.3]])
    mf = _gth_h2_reciprocal_krhf(coords)
    assert mf.converged
    gradient = mf.nuc_grad_method()
    analytic = gradient.kernel()

    step = 1.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    mf_plus = _gth_h2_reciprocal_krhf(plus)
    mf_minus = _gth_h2_reciprocal_krhf(minus)
    numerical = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-12)
    assert gradient.timings["two_electron_seconds"] > 0.0


def test_native_full_gdf_krhf_gradient_matches_total_energy_difference():
    coords = np.asarray([[1.7, 2.4, 2.6], [3.15, 2.7, 2.85]])
    mf = _full_gdf_h2_krhf(coords)
    assert mf.converged
    gradient = mf.nuc_grad_method()
    analytic = gradient.kernel()

    step = 1.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    mf_plus = _full_gdf_h2_krhf(plus)
    mf_minus = _full_gdf_h2_krhf(minus)
    numerical = (mf_plus.e_tot - mf_minus.e_tot) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-10)
    assert gradient.gdf_response_info["kernel"] == "full"
    assert gradient.gdf_response_info["metric_rank"] > 0

    dm = np.asarray(mf.make_rdm1(), dtype=np.complex128)
    weights = np.asarray([[0.7, -0.2j, 0.1], [-0.4, 0.3, 0.2j]])
    full_s1, full_h1, full_veff1 = gradient.explicit_integral_derivatives(dm)
    directional = gradient.directional_integral_derivatives(weights, dm)
    for actual, tensor in zip(directional, (full_s1, full_h1, full_veff1)):
        expected = np.einsum("Ax,Axpq->pq", weights, tensor, optimize=True)
        np.testing.assert_allclose(actual, expected, atol=3.0e-12, rtol=0.0)
    assert gradient.directional_response_info["retained_peak_tensor_count"] == 2
    s1 = gradient.one_electron_derivatives()[0]
    analytic_veff1 = gradient.effective_potential_derivatives(dm, s1=s1)[0, 0]

    def fixed_density_veff(mean_field):
        vj, vk = mean_field.with_df.get_jk([dm])
        value = vj[0] - 0.5 * vk[0]
        if mean_field.madelung is not None:
            overlap = mean_field._overlap_k[0]
            value -= 0.5 * mean_field.madelung * (overlap @ dm @ overlap)
        return value

    numerical_veff1 = (
        fixed_density_veff(mf_plus) - fixed_density_veff(mf_minus)
    ) / (2.0 * step)
    np.testing.assert_allclose(
        analytic_veff1,
        numerical_veff1,
        atol=3.0e-9,
        rtol=0.0,
    )

    space = KPointTransitionSpace(mf, qpts="gamma")
    operator = periodic_tda_operator(
        space,
        q_index=0,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=0.0,
        coulomb_component="gdf",
    )
    mode = np.zeros((2, 3))
    mode[0, 0] = 1.0
    coupling = gamma_tda_electron_phonon_coupling(
        operator,
        mode,
        0.2,
        kernel_derivative="bare_gdf",
        cphf_tol=1.0e-11,
    )
    plus_energy = np.asarray(mf_plus.mo_energy, dtype=float).reshape(-1)
    minus_energy = np.asarray(mf_minus.mo_energy, dtype=float).reshape(-1)
    mass = mf.cell.unit_molecule.atom_mass_list()[0] * amu_to_au
    numerical_gap1 = (
        (plus_energy[1] - plus_energy[0])
        - (minus_energy[1] - minus_energy[0])
    ) / (2.0 * step * np.sqrt(mass))
    bare_kernel1 = gamma_gdf_bare_tda_kernel_derivative(
        operator,
        mode,
    )
    q_derivative = commensurate_gdf_q_derivative(
        mf,
        np.zeros(3),
        mode,
        cphf_tol=1.0e-11,
    )
    commensurate_kernel1 = commensurate_gdf_bare_tda_kernel_derivative(
        operator,
        q_derivative,
    )
    np.testing.assert_allclose(
        q_derivative.fock_derivative[0],
        coupling.fock_derivative_ao,
        atol=3.0e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        q_derivative.overlap_derivative[0],
        coupling.overlap_derivative_ao,
        atol=3.0e-12,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        commensurate_kernel1,
        bare_kernel1,
        atol=3.0e-11,
        rtol=0.0,
    )
    screened_operator = periodic_tda_operator(
        space,
        q_index=0,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    gamma_screened_kernel1 = gamma_gdf_screened_tda_kernel_derivative(
        screened_operator,
        mode,
        coupling.mo_couplings,
    )
    commensurate_screened_kernel1 = (
        commensurate_gdf_screened_tda_kernel_derivative(
            screened_operator,
            q_derivative,
        )
    )
    np.testing.assert_allclose(
        commensurate_screened_kernel1,
        gamma_screened_kernel1,
        atol=3.0e-10,
        rtol=0.0,
    )
    analytic_gap1 = (
        coupling.derivative.matvec(np.ones(1))[0] - bare_kernel1[0, 0]
    )
    np.testing.assert_allclose(
        analytic_gap1.real,
        numerical_gap1,
        atol=3.0e-9,
        rtol=0.0,
    )
    np.testing.assert_allclose(analytic_gap1.imag, 0.0, atol=2.0e-11)
    assert coupling.response.converged
    assert coupling.info["bse_bare_kernel_derivative"] == "frozen_orbital_gdf"

    reference_energy = np.asarray(mf.mo_energy, dtype=float).reshape(1, -1)
    reference_coeff = np.asarray(mf.mo_coeff, dtype=np.complex128).reshape(
        1,
        mf.cell.nao,
        -1,
    )
    reference_occ = np.asarray(mf.mo_occ, dtype=float).reshape(1, -1)

    def frozen_orbital_bse(mean_field):
        mean_field.mo_energy = [reference_energy[0].copy()]
        mean_field.mo_coeff = [reference_coeff[0].copy()]
        mean_field.mo_occ = [reference_occ[0].copy()]
        frozen_space = KPointTransitionSpace(mean_field, qpts="gamma")
        return periodic_bse_matrices(
            frozen_space,
            q_index=0,
            coulomb_component="gdf",
            direct_scale=2.0,
            exchange_scale=1.0,
            screened_exchange_scale=0.0,
        ).A

    numerical_kernel1 = (
        frozen_orbital_bse(mf_plus) - frozen_orbital_bse(mf_minus)
    ) / (2.0 * step * np.sqrt(mass))
    np.testing.assert_allclose(
        bare_kernel1,
        numerical_kernel1,
        atol=3.0e-9,
        rtol=0.0,
    )


def test_commensurate_finite_q_screened_kernel_obeys_star_adjoint():
    mf = _full_gdf_two_k_h2_krhf()
    space = KPointTransitionSpace(mf, qpts="mesh")
    zero_index = space.find_qpoint_index(np.zeros(3))
    phonon_index = next(
        index for index in range(space.nqpts) if index != zero_index
    )
    q_derivative = commensurate_gdf_q_derivative(
        mf,
        space.qpts[phonon_index],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )
    kernels = []
    for source_index in (zero_index, phonon_index):
        operator = periodic_tda_operator(
            space,
            q_index=source_index,
            direct_scale=2.0,
            exchange_scale=1.0,
            screened_exchange_scale=1.0,
            coulomb_component="gdf",
        )
        kernels.append(
            commensurate_gdf_screened_tda_kernel_derivative(
                operator,
                q_derivative,
            )
        )

    assert np.linalg.norm(kernels[0]) > 1.0e-5
    np.testing.assert_allclose(
        kernels[1],
        kernels[0].conj().T,
        atol=1.0e-9,
        rtol=0.0,
    )
    assert q_derivative.gdf_screened_kernel_derivative_info["star_projected"]
    assert (
        q_derivative.gdf_screened_kernel_derivative_info["raw_star_residual"]
        > 1.0e-10
    )
    response = q_derivative.gdf_screened_interaction_derivative
    assert set(response.rpa_matrices) == {zero_index, phonon_index}
    assert set(response.rpa_matrix_derivatives) == {
        (zero_index, phonon_index),
        (phonon_index, zero_index),
    }
    assert q_derivative.supercell_mean_field.with_df.mesh == (11, 5, 5)


def test_gamma_gdf_screening_and_gw_derivatives_match_displacements():
    coords = np.asarray([[1.5, 2.4, 2.6], [3.3, 2.8, 2.9]])
    mf = _full_gdf_lih_krhf(coords)
    space = KPointTransitionSpace(mf, qpts="gamma")
    operator = periodic_tda_operator(
        space,
        q_index=0,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    mode = np.zeros((2, 3))
    mode[1, 0] = 1.0
    coupling = gamma_tda_electron_phonon_coupling(
        operator,
        mode,
        0.2,
        kernel_derivative="screened_gdf",
        cphf_tol=1.0e-11,
    )
    response1 = coupling.gdf_screened_interaction_derivative
    assert response1 is not None
    assert coupling.info["bse_screening_derivative"] == (
        "direct_rpa_gdf_frozen_transition_orbitals"
    )
    reference_poles = space.screened_interaction(
        0,
        direct_scale=2.0,
        coulomb_component="gdf",
    )
    np.testing.assert_allclose(response1.omega, reference_poles.omega, atol=2.0e-12)
    np.testing.assert_allclose(
        response1.coupling @ response1.coupling.conj().T,
        reference_poles.residue_metric(),
        atol=2.0e-11,
    )

    mass = mf.cell.unit_molecule.atom_mass_list()[1] * amu_to_au
    step = 1.0e-4
    displacement = np.zeros_like(coords)
    displacement[1, 0] = step / np.sqrt(mass)
    mf_plus = _full_gdf_lih_krhf(coords + displacement)
    mf_minus = _full_gdf_lih_krhf(coords - displacement)
    reference_coeff = np.asarray(mf.mo_coeff, dtype=np.complex128).reshape(
        mf.cell.nao,
        -1,
    )
    reference_occ = np.asarray(mf.mo_occ, dtype=float).reshape(-1)

    def frozen_space(mean_field):
        mean_field.mo_coeff = [reference_coeff.copy()]
        mean_field.mo_occ = [reference_occ.copy()]
        return KPointTransitionSpace(mean_field, qpts="gamma")

    plus_space = frozen_space(mf_plus)
    minus_space = frozen_space(mf_minus)
    plus_poles = plus_space.screened_interaction(
        0,
        direct_scale=2.0,
        coulomb_component="gdf",
    )
    minus_poles = minus_space.screened_interaction(
        0,
        direct_scale=2.0,
        coulomb_component="gdf",
    )
    numerical_omega1 = (plus_poles.omega - minus_poles.omega) / (2.0 * step)
    assert np.linalg.norm(response1.omega1) > 1.0e-4
    np.testing.assert_allclose(
        response1.omega1,
        numerical_omega1,
        atol=3.0e-8,
        rtol=0.0,
    )

    def screened_exchange(test_space):
        return periodic_bse_matrices(
            test_space,
            q_index=0,
            coulomb_component="gdf",
            direct_scale=2.0,
            exchange_scale=1.0,
            screened_exchange_scale=1.0,
        ).screened_exchange

    numerical_screened1 = (
        screened_exchange(plus_space) - screened_exchange(minus_space)
    ) / (2.0 * step)
    assert np.linalg.norm(
        coupling.gdf_kernel_derivative_components["screened"]
    ) > 1.0e-5
    np.testing.assert_allclose(
        coupling.gdf_kernel_derivative_components["screened"],
        numerical_screened1,
        atol=5.0e-8,
        rtol=0.0,
    )

    omega = 0.3
    eta = 0.05
    analytic_sigma1 = gamma_gdf_diagonal_self_energy_derivative(
        response1,
        band_index=1,
        omega=omega,
        eta=eta,
    )

    def sigma(test_space):
        return diagonal_correlation_self_energy(
            test_space,
            k_index=0,
            band_index=1,
            omega=omega,
            q_indices=[0],
            eta=eta,
            direct_scale=2.0,
            coulomb_component="gdf",
        ).value()

    numerical_sigma1 = (sigma(plus_space) - sigma(minus_space)) / (2.0 * step)
    assert abs(analytic_sigma1) > 1.0e-5
    np.testing.assert_allclose(
        analytic_sigma1,
        numerical_sigma1,
        atol=8.0e-8,
        rtol=0.0,
    )

    analytic_qp1 = gamma_gdf_g0w0_energy_derivative(
        response1,
        band_index=1,
        eta=eta,
    )

    def qp_energy(test_space):
        result = diagonal_g0w0(
            test_space,
            q_indices=[0],
            eta=eta,
            direct_scale=2.0,
            coulomb_component="gdf",
            qp_bands=[1],
        )
        return result.e_qp[0, 1]

    numerical_qp1 = (qp_energy(plus_space) - qp_energy(minus_space)) / (
        2.0 * step
    )
    np.testing.assert_allclose(
        analytic_qp1,
        numerical_qp1,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_native_range_separated_gdf_gradient_matches_total_energy_difference():
    coords = np.asarray([[1.7, 2.4, 2.6], [3.15, 2.7, 2.85]])
    mf = _range_separated_gdf_h2_krhf(coords)
    assert mf.converged
    gradient = mf.nuc_grad_method()
    analytic = gradient.kernel()

    step = 1.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    numerical = (
        _range_separated_gdf_h2_krhf(plus).e_tot
        - _range_separated_gdf_h2_krhf(minus).e_tot
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-10)
    assert gradient.gdf_response_info["kernel"] == "range_separated"
    assert gradient.gdf_response_info["metric_rank"] > 0


def test_native_partitioned_range_separated_gdf_gradient_matches_difference():
    coords = np.asarray([[1.7, 2.4, 2.6], [3.15, 2.7, 2.85]])
    options = {
        "auxbasis": "def2-svp-jkfit",
        "omega": 1.5,
        "recip_cut": 2,
        "image_cut": 0,
        "pair_partition": "smooth",
        "aux_partition": "smooth",
        "smooth_exponent_cutoff": 1.0,
        "reciprocal_only_pair_mask": [[True, False], [False, False]],
    }
    mf = _range_separated_gdf_h2_krhf(coords, **options)
    assert mf.converged
    gradient = mf.nuc_grad_method()
    analytic = gradient.kernel()

    step = 1.0e-4
    plus = coords.copy()
    minus = coords.copy()
    plus[0, 0] += step
    minus[0, 0] -= step
    numerical = (
        _range_separated_gdf_h2_krhf(plus, **options).e_tot
        - _range_separated_gdf_h2_krhf(minus, **options).e_tot
    ) / (2.0 * step)

    np.testing.assert_allclose(analytic[0, 0], numerical, atol=2.0e-8, rtol=0.0)
    np.testing.assert_allclose(np.sum(analytic, axis=0), 0.0, atol=2.0e-10)
    assert gradient.gdf_response_info["rs_aux_partition_active"]
    assert gradient.gdf_response_info["rs_pair_partition_active"]
    assert 0 < gradient.gdf_response_info["rs_compact_aux"] < 36
    assert gradient.gdf_response_info["rs_compact_pairs"] == 3


def test_periodic_gdf_packing_rejects_nonhermitian_auxiliary_gauge():
    from pyqed.qchem.pbc.gdf import _cderi_is_hermitian

    factor = np.asarray([[[1.0, 0.2], [0.2, 0.5]]], dtype=np.complex128)
    phase_rotated = 1.0j * factor

    assert _cderi_is_hermitian(factor, 1.0e-12)
    assert not _cderi_is_hermitian(phase_rotated, 1.0e-12)
    np.testing.assert_allclose(
        factor.reshape(1, -1).T @ factor.reshape(1, -1).conj(),
        phase_rotated.reshape(1, -1).T @ phase_rotated.reshape(1, -1).conj(),
    )


def test_pbc_cell_builds_native_1d_and_makes_kpts():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    assert cell.built
    assert cell.nao == 1
    assert cell.nelectron == 2
    assert cell.low_dim_ft_type == "inf_vacuum"
    np.testing.assert_allclose(
        cell.lattice_vectors,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
    )

    kpts = cell.make_kpts(3)
    assert kpts.shape == (3, 3)
    np.testing.assert_allclose(kpts[1], np.zeros(3), atol=1e-12)


def test_pbc_chain_is_explicit_1d_api():
    chain = Chain(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        vacuum=20.0,
    ).build()

    assert isinstance(chain, Cell)
    assert chain.dimension == 1
    assert chain.lattice_constant == 3.0
    np.testing.assert_allclose(
        chain.lattice_vectors,
        np.asarray([[3.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 20.0]]),
    )


def test_pbc_chain_rhf_uses_native_1d_path():
    chain = Chain(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        spin=0,
        vacuum=20.0,
    ).build()

    mf = chain.RHF(nimages=1, nk=3).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert mf.nkpts == 3
    assert len(mf.dm) == 3


def test_native_pbc_rhf_runs_gamma_and_kpoint_1d():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf_gamma = RHF(cell, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_gamma.e_tot)
    assert mf_gamma.nkpts == 1
    assert mf_gamma.dm.shape == (1, 1)
    assert np.isfinite(mf_gamma.e_nuc)

    kpts = cell.make_kpts(3)
    mf_k = RHF(cell, kpts=kpts, nimages=1).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert np.isfinite(mf_k.e_tot)
    assert mf_k.nkpts == 3
    assert len(mf_k.dm) == 3
    assert all(d.shape == (1, 1) for d in mf_k.dm)
    assert np.isfinite(mf_k.e_nuc)


def test_native_pbc_rhf_reports_unconverged_gamma_runs():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = RHF(cell, nimages=1).run(max_cycle=1, conv_tol=1e-12, conv_tol_dm=1e-12)
    assert not mf.converged


def test_native_pbc_rhf_accepts_nk_convenience_mesh():
    cell = Cell(
        atom="He 0 0 0",
        a=3.0,
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(nimages=1, nk=3).run(max_cycle=30, conv_tol=1e-10, conv_tol_dm=1e-8)
    assert mf.nkpts == 3
    assert len(mf.dm) == 3


def test_pbc_cell_has_native_ewald_nuclear_repulsion():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    e1 = cell.ewald_nuclear_repulsion(real_cut=3, recip_cut=6)
    e2 = cell.ewald_nuclear_repulsion(real_cut=4, recip_cut=8)
    assert np.isfinite(e1)
    assert np.isfinite(e2)
    assert abs(e2 - e1) < 5e-2


def test_native_gth_cell_uses_valence_charge_and_explicit_data():
    carbon_gth_pade = [
        [2, 2],
        0.34883045,
        2,
        [-8.5137711, 1.22843203],
        2,
        [0.30455321, 1, [[9.52284179]]],
        [0.2326773, 0, []],
    ]
    cell = Cell(
        atom="C 0 0 0",
        a=np.eye(3) * 6.8,
        basis="sto-3g",
        pseudo={"C": carbon_gth_pade},
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()

    assert cell.has_pseudo
    assert cell.nelectron == 4
    np.testing.assert_allclose(cell.ionic_charges, [4.0])
    assert cell.KRHF(nk=1).jk_builder == "gdf"
    assert isinstance(cell.RHF(), EwaldRHF)
    with pytest.raises(ValueError, match="pair_cut >= 2"):
        cell.KRHF(nk=1, pair_cut=1)._validate()
    with pytest.raises(NotImplementedError, match="Pseudopotentials require"):
        cell.RHF(method="finite_image")


def test_native_gth_local_and_nonlocal_matrices_match_pyscf():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto
    from pyscf.pbc.gto.pseudo import pp, pp_int

    atom = "C 0 0 0"
    lattice = np.eye(3) * 6.8
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="gth-szv",
        pseudo="gth-pade",
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=1,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
        pseudo_cut=1,
        jk_builder="reciprocal",
    )
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()

    reference_cell = gto.Cell(
        atom=atom,
        a=lattice,
        basis="gth-szv",
        pseudo="gth-pade",
        unit="bohr",
        precision=1.0e-11,
        mesh=[81, 81, 81],
        cart=True,
        verbose=0,
    ).build()
    kvec = np.zeros(3)
    reference_nonlocal = pp_int.get_pp_nl(reference_cell, kvec)
    reference_local = pp.get_pp(reference_cell, kvec) - reference_nonlocal
    overlap = mf._fourier_sum(mf._s_r, kvec)
    native_local = (
        mf._fourier_sum(mf._vne_sr_r, kvec)
        + mf._reciprocal_nuclear_attraction(kvec)
        + mf._local_pseudopotential(kvec)
        + mf._nuclear_background_hcore(overlap)
    )

    np.testing.assert_allclose(
        native_local,
        reference_local,
        atol=1.0e-6,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        mf._nonlocal_pseudopotential(kvec),
        reference_nonlocal,
        atol=1.0e-7,
        rtol=0.0,
    )


def test_native_gth_gdf_rhf_total_energy_matches_pyscf():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto, scf

    atom = "C 0 0 0; C 1.7 1.7 1.7"
    lattice = np.eye(3) * 6.8
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="gth-szv",
        pseudo="gth-pade",
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=1,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        pseudo_cut=1,
        recip_cut=7,
    ).run(max_cycle=50, conv_tol=1.0e-7, conv_tol_dm=1.0e-5)

    reference_cell = gto.Cell(
        atom=atom,
        a=lattice,
        basis="gth-szv",
        pseudo="gth-pade",
        unit="bohr",
        precision=1.0e-9,
        mesh=[41, 41, 41],
        verbose=0,
    ).build()
    reference_mf = scf.RHF(reference_cell).density_fit()
    reference_mf.conv_tol = 1.0e-9
    reference_energy = reference_mf.kernel()

    assert mf.jk_builder == "gdf"
    assert mf.converged
    assert reference_mf.converged
    np.testing.assert_allclose(mf.e_nuc, reference_cell.energy_nuc(), atol=1.0e-10)
    np.testing.assert_allclose(mf.e_tot, reference_energy, atol=1.0e-4, rtol=0.0)


def test_s_gaussian_pair_fourier_matches_overlap_at_zero_g():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    pair_g0 = ao_pair_ft_matrix_s(cell.unit_molecule._bas, np.zeros(3))
    np.testing.assert_allclose(pair_g0, cell.unit_molecule.overlap, atol=1e-12)


def test_s_gaussian_pair_fourier_has_real_density_symmetry():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    gvec = np.asarray([0.3, -0.2, 0.1])
    pair_g = ao_pair_ft_matrix_s(cell.unit_molecule._bas, gvec)
    pair_minus_g = ao_pair_ft_matrix_s(cell.unit_molecule._bas, -gvec)
    np.testing.assert_allclose(pair_minus_g, pair_g.conj(), atol=1e-12)


def test_ewald_pair_fourier_keeps_finite_g_sp_terms_against_pyscf():
    pytest.importorskip("pyscf")
    from pyscf.pbc import gto
    from pyscf.pbc.df import ft_ao

    atom = "Li 0 0 0; H 3.0 0 0"
    lattice = np.diag([8.0, 8.0, 8.0])
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=2,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf._validate()
    mf._periodic_setup()

    g = np.pi / 4.0
    gvecs = np.asarray(
        [
            [0.0, 0.0, -g],
            [-g, 0.0, 0.0],
            [g, 0.0, 0.0],
            [0.0, -g, 0.0],
            [0.0, 0.0, g],
            [0.0, g, 0.0],
        ],
        dtype=float,
    )
    kpt = np.asarray(mf.kpts[0], dtype=float)
    native = mf._periodic_pair_ft_batch(gvecs, kpt)

    pyscf_cell = gto.Cell()
    pyscf_cell.atom = atom
    pyscf_cell.a = lattice
    pyscf_cell.basis = "sto-3g"
    pyscf_cell.unit = "B"
    pyscf_cell.spin = 0
    pyscf_cell.verbose = 0
    pyscf_cell.cart = True
    pyscf_cell.build()
    pyscf_ref = np.asarray(
        ft_ao.ft_aopair(
            pyscf_cell,
            gvecs,
            aosym="s1",
            kpti_kptj=np.asarray([kpt, kpt]),
        )
    )

    np.testing.assert_allclose(native[:, 0, 2:5], pyscf_ref[:, 0, 2:5], atol=5.0e-6)
    np.testing.assert_allclose(native, pyscf_ref, atol=5.0e-6)


def test_ewald_pair_fourier_compiled_block_matches_direct_shift_loop():
    from pyqed.qchem.fourier import has_compiled_ao_ft

    if not has_compiled_ao_ft():
        pytest.skip("compiled AO-pair Fourier backend is not available")

    cell = Cell(
        atom="Li 0 0 0; H 3.0 0 0",
        a=np.diag([8.0, 8.0, 8.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=1,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf._validate()
    mf._periodic_setup()

    assert mf._pair_ft_block_plan is not None
    gvecs = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 4.0, 0.0, 0.0],
            [0.0, -np.pi / 4.0, np.pi / 5.0],
        ],
        dtype=float,
    )
    kpt = np.asarray(mf.kpts[0], dtype=float)
    fast = mf._periodic_pair_ft_batch(gvecs, kpt)
    direct = mf._periodic_pair_ft_batch_direct(gvecs, kpt)

    np.testing.assert_allclose(fast, direct, atol=1.0e-12)


def test_ewald_pair_fourier_many_k_matches_single_phase_blocks():
    from pyqed.qchem.fourier import has_compiled_ao_ft

    if not has_compiled_ao_ft():
        pytest.skip("compiled AO-pair Fourier backend is not available")

    cell = Cell(
        atom="Li 0 0 0; H 3.0 0 0",
        a=np.diag([8.0, 8.0, 8.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=1,
        recip_cut=1,
        jk_builder="ewald",
    )
    mf._validate()
    mf._periodic_setup()

    gvecs = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [np.pi / 4.0, 0.0, 0.0],
            [0.0, -np.pi / 4.0, np.pi / 5.0],
        ],
        dtype=float,
    )
    fused = mf._periodic_pair_ft_batch_many(gvecs, mf.kpts)
    reference = np.stack(
        [mf._periodic_pair_ft_batch(gvecs, kvec) for kvec in mf.kpts],
        axis=0,
    )

    np.testing.assert_allclose(fused, reference, atol=1.0e-12, rtol=0.0)


def test_compiled_periodic_one_body_matches_python_sp_integrals():
    from pyqed.qchem import basis as basis_module

    if not hasattr(basis_module._basis_cy, "compute_periodic_one_electron"):
        pytest.skip("compiled periodic one-electron kernel unavailable")

    cell = Cell(
        atom="Li 0 0 0; H 1.6 0 0",
        a=np.diag([7.0, 7.0, 7.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=(1, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=1,
        jk_builder="gdf",
        one_body_nuclear_cut=0,
    )
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()

    key = (0, 0, 0)
    charges = np.asarray(cell.unit_molecule.atom_charges(), dtype=float)
    coords = np.asarray(cell._atom_coords, dtype=float)
    shifted = mf._shifted_basis[key]
    overlap = np.zeros_like(mf._s_r[key])
    kinetic = np.zeros_like(mf._t_r[key])
    vnuc = np.zeros_like(mf._vne_sr_r[key])
    for p, left in enumerate(mf._basis):
        for q, right in enumerate(shifted):
            overlap[p, q] = S(left, right)
            kinetic[p, q] = T(left, right)
            vnuc[p, q] = -sum(
                charge * short_range_point_charge(left, right, coord, mf.eta)
                for charge, coord in zip(charges, coords)
            )

    np.testing.assert_allclose(mf._s_r[key], overlap, atol=1.0e-12)
    np.testing.assert_allclose(mf._t_r[key], kinetic, atol=1.0e-12)
    np.testing.assert_allclose(mf._vne_sr_r[key], vnuc, atol=1.0e-11)


def test_periodic_one_body_lattice_centering_preserves_translation_hermiticity():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=(1, 1, 1),
        eta=0.5,
        real_cut=1,
        pair_cut=1,
        recip_cut=2,
        one_body_nuclear_cut=2,
        one_body_screen_tol=0.0,
    )
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()

    for key, block in mf._vne_sr_r.items():
        reverse = tuple(-value for value in key)
        np.testing.assert_allclose(block, mf._vne_sr_r[reverse].T, atol=1.0e-12)


def test_periodic_one_body_screen_does_not_truncate_eta_zero_reference():
    from pyqed.qchem import basis as basis_module

    compiled = getattr(basis_module._basis_cy, "compute_periodic_one_electron", None)
    if compiled is None:
        pytest.skip("compiled periodic one-electron kernel unavailable")
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    signatures = [
        basis_module._basis_signature(fn) for fn in cell.unit_molecule._bas
    ]
    shells, origins, exps, weights, nprim = (
        basis_module._pack_signatures_for_numba(signatures)
    )
    right_origins = np.ascontiguousarray(origins[None, :, :])
    coords = np.ascontiguousarray(cell._atom_coords, dtype=float)
    charges = np.ascontiguousarray(cell.ionic_charges, dtype=float)
    mask = np.ones((1, cell.nao, cell.nao), dtype=np.uint8)

    exact = compiled(
        shells, origins, right_origins, exps, weights, nprim,
        coords, charges, 0.0, mask, 0.0,
    )
    screened = compiled(
        shells, origins, right_origins, exps, weights, nprim,
        coords, charges, 0.0, mask, 1.0,
    )

    for actual, expected in zip(screened, exact):
        np.testing.assert_array_equal(actual, expected)


def test_reciprocal_pair_domain_cannot_truncate_one_body_domain():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()

    with pytest.raises(ValueError, match="pair_cut must be at least real_cut"):
        cell.KRHF(real_cut=2, pair_cut=1)._validate()


def test_automatic_periodic_image_domain_matches_screened_explicit_envelope():
    half = 7.72 / 2.0
    lattice = np.asarray(
        [[0.0, half, half], [half, 0.0, half], [half, half, 0.0]],
        dtype=float,
    )
    cell = Cell(
        atom=f"Li 0 0 0; H {half} {half} {half}",
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()

    automatic = cell.KRHF(real_cut="auto", pair_cut="auto")
    automatic._validate()
    automatic._periodic_setup()
    explicit = cell.KRHF(real_cut=8, pair_cut=8)
    explicit._validate()
    explicit._periodic_setup()

    assert automatic.real_cut == automatic.pair_cut == 8
    assert automatic._shift_keys == explicit._shift_keys
    assert automatic._pair_shift_keys == explicit._pair_shift_keys
    assert len(automatic._shift_keys) < len(cell.image_keys(8))
    assert set(automatic._shift_keys) <= set(automatic._pair_shift_keys)
    assert all(
        tuple(-value for value in key) in automatic._shift_vectors
        for key in automatic._shift_keys
    )


def test_pbc_cell_has_native_reciprocal_electronic_matrices():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    vne = cell.reciprocal_nuclear_attraction_matrix(recip_cut=8)
    dm = np.eye(cell.nao)
    jmat = cell.reciprocal_hartree_matrix(dm, recip_cut=8)

    assert vne.shape == (cell.nao, cell.nao)
    assert jmat.shape == (cell.nao, cell.nao)
    np.testing.assert_allclose(vne, vne.conj().T, atol=1e-12)
    np.testing.assert_allclose(jmat, jmat.conj().T, atol=1e-12)
    assert np.all(np.isfinite(vne.real))
    assert np.all(np.isfinite(jmat.real))


def test_native_reciprocal_electronic_matrices_are_cutoff_stable():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    eta = 0.5
    vne6 = cell.reciprocal_nuclear_attraction_matrix(recip_cut=6, eta=eta)
    vne8 = cell.reciprocal_nuclear_attraction_matrix(recip_cut=8, eta=eta)
    j6 = cell.reciprocal_hartree_matrix(np.eye(cell.nao), recip_cut=6, eta=eta)
    j8 = cell.reciprocal_hartree_matrix(np.eye(cell.nao), recip_cut=8, eta=eta)

    assert np.linalg.norm(vne8 - vne6) < 1e-2
    assert np.linalg.norm(j8 - j6) < 1e-2


def test_short_range_nuclear_attraction_matches_full_at_eta_zero():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    basis = cell.unit_molecule._bas
    kinetic = np.asarray([[T(a, b) for b in basis] for a in basis])
    full_vne = cell.unit_molecule.hcore - kinetic
    sr_vne = cell.short_range_nuclear_attraction_matrix(eta=0.0, real_cut=0)

    np.testing.assert_allclose(sr_vne, full_vne, atol=1e-10)


def test_short_range_nuclear_attraction_decays_with_eta():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_small_eta = cell.short_range_nuclear_attraction_matrix(eta=0.3, real_cut=1)
    sr_large_eta = cell.short_range_nuclear_attraction_matrix(eta=1.0, real_cut=1)
    assert np.linalg.norm(sr_large_eta) < np.linalg.norm(sr_small_eta)


def test_short_range_eri_matches_full_at_eta_zero():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_eri = cell.short_range_eri_tensor(eta=0.0)
    np.testing.assert_allclose(sr_eri, cell.unit_molecule.eri, atol=1e-10)


def test_short_range_eri_decays_with_eta():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    sr_small_eta = cell.short_range_eri_tensor(eta=0.3)
    sr_large_eta = cell.short_range_eri_tensor(eta=1.0)
    assert np.linalg.norm(sr_large_eta) < np.linalg.norm(sr_small_eta)


def test_reciprocal_eri_tensor_has_basic_permutation_symmetry():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    eri = cell.reciprocal_eri_tensor(recip_cut=6, eta=0.5)
    assert eri.shape == (cell.nao, cell.nao, cell.nao, cell.nao)
    np.testing.assert_allclose(eri, eri.transpose(1, 0, 2, 3), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(0, 1, 3, 2), atol=1e-12)
    np.testing.assert_allclose(eri, eri.transpose(2, 3, 0, 1), atol=1e-12)


def test_native_ewald_rhf_runs_s_gaussian_gamma():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(method="ewald", eta=0.5, real_cut=3, recip_cut=6, mesh=(9, 10, 10)).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    assert mf.converged
    assert np.isfinite(mf.e_tot)
    assert np.isfinite(mf.e_nuc)
    assert mf.dm.shape == (cell.nao, cell.nao)
    assert np.isfinite(mf.madelung)


def test_native_ewald_rhf_runs_s_gaussian_kpoints():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([4.0, 20.0, 20.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=1,
        spin=0,
        vacuum=20.0,
    ).build()

    mf = cell.RHF(method="ewald", nk=3, eta=0.5, real_cut=1, recip_cut=4, mesh=(7, 8, 8)).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    assert mf.converged
    assert mf.nkpts == 3
    assert len(mf.dm) == 3
    assert all(d.shape == (cell.nao, cell.nao) for d in mf.dm)
    assert all(np.allclose(f, f.conj().T, atol=1e-10) for f in mf.fock)


def test_pbc_exposes_krhf_alias_for_ewald_solver():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    assert KRHF is EwaldRHF
    assert isinstance(cell.KRHF(nk=(1, 1, 1), eta=0.5), EwaldRHF)
    assert isinstance(cell.RHF(method="krhf", nk=(1, 1, 1), eta=0.5), EwaldRHF)


def test_reciprocal_cut_auto_resolves_from_pair_ft_tail():
    cell = Cell(
        atom="H 2.3 3 3; H 3.7 3 3",
        a=np.eye(3) * 6.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    options = {
        "nk": 1,
        "eta": 0.5,
        "real_cut": 2,
        "pair_cut": 2,
        "one_body_nuclear_cut": 2,
        "jk_builder": "reciprocal",
    }
    automatic = cell.KRHF(
        **options,
        recip_cut="auto",
        recip_precision=1.0e-8,
    )
    automatic._validate()
    automatic._periodic_setup()

    assert automatic.recip_cut == 10
    assert automatic.recip_auto_info["mode"] == "auto"
    assert automatic.recip_auto_info["estimated_tail"] <= 1.0e-8

    explicit = cell.KRHF(**options, recip_cut=automatic.recip_cut)
    automatic._build_integrals()
    explicit._build_integrals()
    np.testing.assert_allclose(automatic.hcore, explicit.hcore, atol=0.0, rtol=0.0)
    assert automatic.integral_build_timings["recip_cut_auto"] is True
    assert automatic.integral_build_timings["recip_cut"] == 10


def test_reciprocal_cut_auto_is_precision_monotone_and_pads_k_transfers():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.eye(3) * 5.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    common = {
        "kpts": cell.make_kpts((2, 1, 1)),
        "eta": 0.5,
        "real_cut": 1,
        "pair_cut": 1,
        "jk_builder": "reciprocal",
    }
    loose = cell.KRHF(**common, recip_precision=1.0e-5)
    tight = cell.KRHF(**common, recip_precision=1.0e-8)
    loose._validate()
    tight._validate()
    loose._periodic_setup()
    tight._periodic_setup()

    assert tight.recip_cut >= loose.recip_cut
    assert loose.recip_auto_info["kpoint_pad"] == 1
    assert tight.recip_cut == (
        tight.recip_auto_info["base_cut"] + tight.recip_auto_info["kpoint_pad"]
    )

    direct_cut = tight.recip_cut
    tight.jk_builder = "ewald"
    assert tight._periodic_setup() is True
    assert tight.recip_auto_info["mode"] == "auto"
    assert tight.recip_cut <= direct_cut


def test_reciprocal_cut_explicit_bypasses_auto_estimator():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.eye(3) * 5.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mean_field = cell.KRHF(recip_cut=4)
    mean_field._validate()
    mean_field._periodic_setup()

    assert mean_field.recip_cut == 4
    assert mean_field.recip_auto_info == {
        "mode": "explicit",
        "requested": 4,
        "resolved_cut": 4,
    }


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"recip_precision": 0.0}, "recip_precision"),
        ({"recip_precision": np.inf}, "recip_precision"),
        ({"recip_max_cut": 1}, "recip_max_cut"),
    ],
)
def test_reciprocal_cut_auto_rejects_invalid_controls(options, message):
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.eye(3) * 5.0,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    with pytest.raises(ValueError, match=message):
        cell.KRHF(**options)._validate()


def test_periodic_setup_reuses_unchanged_pair_ft_plan_and_invalidates_cutoff(
    monkeypatch,
):
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        kpts=cell.make_kpts((2, 1, 1)),
        eta=0.5,
        real_cut=1,
        pair_cut=1,
        recip_cut=2,
    )
    mf._validate()

    assert mf._periodic_setup() is True
    pair_plan = mf._pair_ft_primitive_terms
    assert mf._periodic_setup() is False
    assert mf._pair_ft_primitive_terms is pair_plan

    mf.pair_cut = 2
    assert mf._periodic_setup() is True
    assert mf._pair_ft_primitive_terms is not pair_plan

    mf.real_cut = 0
    mf.pair_cut = 0
    mf.recip_cut = 1
    mf._build_integrals()
    overlap_k = tuple(mf._overlap_k)
    assert mf.integral_build_timings["one_body_reused"] is False
    assert mf.integral_build_timings["real_image_count"] == 1
    assert mf.integral_build_timings["pair_image_count"] == 1
    assert mf.integral_build_timings["total_seconds"] > 0.0

    def reject_reciprocal_rebuild(*_args, **_kwargs):
        raise AssertionError("unchanged one-electron blocks must be reused")

    monkeypatch.setattr(
        mf,
        "_reciprocal_nuclear_attraction_many",
        reject_reciprocal_rebuild,
    )
    mf._build_integrals()
    assert mf.integral_build_timings["one_body_reused"] is True
    assert all(actual is expected for actual, expected in zip(mf._overlap_k, overlap_k))


def test_periodic_one_body_image_workers_preserve_blocks_exactly():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        kpts=cell.make_kpts((2, 1, 1)),
        eta=0.5,
        real_cut=1,
        pair_cut=1,
        recip_cut=2,
        one_body_workers=1,
    )
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()
    serial = {
        name: {key: value.copy() for key, value in getattr(mf, name).items()}
        for name in ("_s_r", "_t_r", "_vne_sr_r")
    }

    mf.one_body_workers = 2
    mf._build_one_body_blocks()
    for name, blocks in serial.items():
        for key, expected in blocks.items():
            np.testing.assert_array_equal(getattr(mf, name)[key], expected)


def test_native_ewald_krhf_uses_global_kpoint_occupations():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf = cell.KRHF(kpts=cell.make_kpts((2, 1, 1)), eta=0.5)
    fock = [
        np.diag([-2.0, -1.0]),
        np.diag([0.0, 10.0]),
    ]
    overlap = [np.eye(cell.nao), np.eye(cell.nao)]

    _mo_energy, _mo_coeff, mo_occ, dm = mf._solve_fock(fock, overlap)

    np.testing.assert_allclose(mo_occ[0], [2.0, 2.0])
    np.testing.assert_allclose(mo_occ[1], [0.0, 0.0])
    electron_count = sum(np.trace(d).real for d in dm) / mf.nkpts
    np.testing.assert_allclose(electron_count, cell.nelectron, atol=1e-12)


def test_native_ewald_krhf_fractionally_occupies_degenerate_frontier_when_requested():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf = cell.KRHF(
        kpts=cell.make_kpts((2, 1, 1)),
        eta=0.5,
        occupation_mode="fractional",
    )
    fock = [
        np.diag([-2.0, 0.0]),
        np.diag([0.0, 10.0]),
    ]
    overlap = [np.eye(cell.nao), np.eye(cell.nao)]

    _mo_energy, mo_coeff, mo_occ, dm = mf._solve_fock(fock, overlap)
    rebuilt_dm = mf.make_rdm1(mo_coeff, mo_occ)

    np.testing.assert_allclose(mo_occ[0], [2.0, 1.0])
    np.testing.assert_allclose(mo_occ[1], [1.0, 0.0])
    for actual, rebuilt in zip(dm, rebuilt_dm):
        np.testing.assert_allclose(actual, rebuilt, atol=1e-12)
    electron_count = sum(np.trace(d).real for d in dm) / mf.nkpts
    np.testing.assert_allclose(electron_count, cell.nelectron, atol=1e-12)


def test_ewald_krhf_allows_odd_cell_fractional_even_kmesh():
    cell = Cell(
        atom="Li 0 0 0",
        a=np.asarray(
            [[-3.315, 3.315, 3.315], [3.315, -3.315, 3.315], [3.315, 3.315, -3.315]]
        ),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(
        kpts=cell.make_kpts((2, 1, 1)),
        eta=0.5,
        occupation_mode="fractional",
    )

    mf._validate()
    energies = [np.asarray([-2.0, 0.0]), np.asarray([-2.0, 0.0])]
    occupations = mf._occupations_from_energies(energies)

    np.testing.assert_allclose(occupations, [[2.0, 1.0], [2.0, 1.0]])


def test_ewald_krhf_rejects_odd_cell_without_fractional_even_kmesh():
    cell = Cell(
        atom="Li 0 0 0",
        a=np.diag([6.63, 6.63, 6.63]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    with pytest.raises(NotImplementedError, match="Odd-electron cells require"):
        cell.KRHF(kpts=cell.make_kpts((2, 1, 1)), eta=0.5)._validate()

    with pytest.raises(NotImplementedError, match="Odd-electron cells require"):
        cell.KRHF(
            kpts=cell.make_kpts((3, 1, 1)),
            eta=0.5,
            occupation_mode="fractional",
        )._validate()


def test_native_ewald_krhf_uses_integer_aufbau_for_degenerate_frontier_by_default():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.KRHF(kpts=cell.make_kpts((2, 1, 1)), eta=0.5)
    fock = [
        np.diag([-2.0, 0.0]),
        np.diag([0.0, 10.0]),
    ]

    _energy, _coeff, occupation, _density = mf._solve_fock(
        fock, [np.eye(cell.nao), np.eye(cell.nao)]
    )

    np.testing.assert_allclose(occupation[0], [2.0, 2.0])
    np.testing.assert_allclose(occupation[1], [0.0, 0.0])


def test_native_3d_hydrogen_cell_builds_and_makes_kpts():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    assert cell.built
    assert cell.dimension == 3
    assert cell.nao == 2
    kpts = cell.make_kpts((2, 2, 2))
    assert kpts.shape == (8, 3)
    assert np.all(np.isfinite(kpts))


def test_native_3d_ewald_rhf_runs_gamma_and_kpoints():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()

    mf_gamma = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    mf_nk1 = cell.RHF(method="ewald", nk=(1, 1, 1), eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    mf_k = cell.RHF(method="ewald", nk=(2, 2, 2), eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    assert mf_gamma.converged
    assert mf_nk1.converged
    assert mf_k.converged
    assert np.isfinite(mf_gamma.e_tot)
    assert np.isfinite(mf_k.e_tot)
    np.testing.assert_allclose(mf_gamma.e_tot, mf_nk1.e_tot, atol=1e-10)
    assert mf_k.nkpts == 8
    assert len(mf_k.dm) == 8
    assert all(d.shape == (cell.nao, cell.nao) for d in mf_k.dm)
    assert all(np.allclose(f, f.conj().T, atol=1e-10) for f in mf_k.fock)


def test_native_3d_hydrogen_band_structure_shapes():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=2).run(
        max_cycle=50,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )
    path = np.column_stack([np.linspace(-0.5, 0.5, 3), np.zeros(3), np.zeros(3)])

    bands = mf.band_structure(scaled_kpts=path, exchange="average")
    overlap_sorted = mf.band_structure(
        scaled_kpts=path,
        exchange="average",
        sort_bands="overlap",
    )

    assert bands["kpts"].shape == (3, 3)
    assert bands["mo_energy"].shape == (3, cell.nao)
    assert bands["mo_energy_reference"].shape == (3, cell.nao)
    assert overlap_sorted["mo_energy"].shape == (3, cell.nao)
    assert np.isfinite(bands["e_fermi"])
    assert np.all(np.isfinite(bands["mo_energy"]))
    assert np.all(np.isfinite(overlap_sorted["mo_energy"]))


def test_native_3d_hydrogen_mesh_interpolated_bands():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(
        method="ewald",
        nk=(2, 1, 1),
        eta=0.5,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        jk_builder="reciprocal",
    ).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    mesh_bands = mf.band_structure(kpts=mf.kpts, exchange="mesh")
    interp_at_mesh = mf.band_structure(kpts=mf.kpts, exchange="mesh_interpolate")
    path = np.column_stack([np.linspace(-0.5, 0.5, 5), np.zeros(5), np.zeros(5)])
    interp_path = mf.band_structure(scaled_kpts=path, exchange="mesh_interpolate")

    assert interp_at_mesh["interpolated"]
    np.testing.assert_allclose(
        interp_at_mesh["mo_energy"],
        mesh_bands["mo_energy"],
        atol=1e-10,
    )
    assert interp_path["mo_energy"].shape == (5, cell.nao)
    assert np.all(np.isfinite(interp_path["mo_energy"]))
    with pytest.raises(ValueError, match="self-consistent SCF k-points"):
        mf.band_structure(scaled_kpts=path, exchange="mesh")


def test_native_3d_reciprocal_jk_accepts_larger_pair_cut():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.RHF(
        method="ewald",
        kpts=kpts,
        eta=0.5,
        real_cut=0,
        pair_cut=1,
        recip_cut=2,
        jk_builder="reciprocal",
    ).run(max_cycle=80, conv_tol=1e-10, conv_tol_dm=1e-8)

    assert mf.converged
    assert mf.nkpts == 2
    assert all(np.allclose(fock, fock.conj().T, atol=1e-10) for fock in mf.fock)
    assert np.isfinite(mf.e_tot)


def test_native_3d_gdf_krhf_mesh_bands_and_off_mesh_guard(monkeypatch, tmp_path):
    import pyqed.pbc.gw as pbc_gw

    legacy_gdf_mo_jk = pbc_gw.gdf_mo_jk

    def reject_gw_transition_path(*args, **kwargs):
        raise AssertionError("KRHF GDF must contract persistent AO factors directly.")

    monkeypatch.setattr(pbc_gw, "gdf_mo_jk", reject_gw_transition_path)
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    assert cell.unit_molecule.eri is None
    assert cell.unit_molecule.eri_s4 is None
    assert cell.unit_molecule.eri_s8 is None
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        precision=1.0e-8,
        storage="disk",
        max_memory_mb=0.0,
        cache_dir=str(tmp_path),
        stream_pairs=True,
    )
    mf.with_df.build(workers=2)
    prebuilt_files = set(mf.with_df.cache_files)
    mf.run(max_cycle=40, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)

    assert mf.converged
    assert mf.niter < 40
    assert mf.with_df is not None
    q0 = mf.with_df._space.find_qpoint_index(np.zeros(3))
    packed_cderi = mf.with_df.packed_cderi(0)
    assert isinstance(packed_cderi.values, np.memmap)
    assert mf.with_df.memory_bytes == 0
    assert mf.with_df.disk_bytes > 0
    assert set(mf.with_df.cache_files) == prebuilt_files
    assert all(os.path.exists(path) for path in mf.with_df.cache_files)
    np.testing.assert_allclose(
        packed_cderi.to_dense(),
        mf.with_df.cderi(q0, 0, 0),
        atol=1.0e-12,
    )

    monkeypatch.setattr(pbc_gw, "gdf_mo_jk", legacy_gdf_mo_jk)
    from pyqed.pbc.gw.integrals import (
        _gdf_mf_cache,
        gdf_transition_factors,
    )
    from pyqed.pbc.gw.response import KPointTransitionSpace

    space = KPointTransitionSpace(mf, qpts="mesh")
    factors = gdf_transition_factors(space, q_index=0)
    assert factors.build_timings["persistent_backend_reuse"]
    assert all(
        not store.ao_blocks for store in _gdf_mf_cache(mf, "q_ao_store").values()
    )
    direct_j, direct_k = mf.with_df.get_jk(mf.dm)
    assert not mf.with_df._disk_maps
    legacy_j, legacy_k = legacy_gdf_mo_jk(space, dm=mf.dm)
    for k_index, coeff in enumerate(space.reference.mo_coeff):
        np.testing.assert_allclose(
            coeff.conj().T @ direct_j[k_index] @ coeff,
            legacy_j[k_index],
            atol=1.0e-10,
        )
        np.testing.assert_allclose(
            coeff.conj().T @ direct_k[k_index] @ coeff,
            legacy_k[k_index],
            atol=1.0e-10,
        )

    mesh_bands = mf.band_structure(kpts=mf.kpts, exchange="mesh", reference="none")
    np.testing.assert_allclose(mesh_bands["mo_energy"], mf.mo_energy, atol=1.0e-10)
    with pytest.raises(NotImplementedError, match="self-consistent k mesh"):
        mf.band_structure(kpts=mf.kpts, exchange="finite_q")
    cache_files = mf.with_df.cache_files
    mf.with_df.close()
    assert all(not os.path.exists(path) for path in cache_files)


def test_gdf_folded_mesh_handles_wrapped_boundary_and_rejects_shift():
    from pyqed.pbc.gw.integrals import _gdf_gamma_folded_mesh
    from pyqed.qchem.pbc.gdf import _KMeshSpace

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    gamma_mf = cell.KRHF(
        kpts=cell.make_kpts((4, 2, 2), gamma_centered=True),
    )
    mesh, indices = _gdf_gamma_folded_mesh(_KMeshSpace(gamma_mf).reference)
    assert mesh == (4, 2, 2)
    assert sorted(indices) == list(range(16))

    shifted_mf = cell.KRHF(kpts=cell.make_kpts((4, 2, 2)))
    assert _gdf_gamma_folded_mesh(_KMeshSpace(shifted_mf).reference) is None


def test_native_gdf_stream_pair_batching_preserves_factors():
    from pyqed.pbc.gw.integrals import _gdf_mf_cache

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    kpts = cell.make_kpts((2, 1, 1), gamma_centered=True)

    def build(batch_size, workers=1, batch_mb=128.0, storage="memory"):
        mf = cell.KRHF(
            kpts=kpts,
            eta=0.5,
            real_cut=2,
            pair_cut=2,
            recip_cut=5,
        ).density_fit(
            auxbasis="def2-svp-jkfit",
            precision=1.0e-8,
            storage=storage,
            stream_pairs=True,
            stream_pair_batch_size=batch_size,
            stream_pair_batch_mb=batch_mb,
            folded_batch_mb=0.001 if workers > 1 else 128.0,
        )
        if workers > 1:
            mf.gdf_folded_min_kpts = 1
        mf.with_df.build(workers=workers)
        return mf.with_df

    single_pair = build(1)
    bounded_q_local = build("auto", batch_mb=0.03)
    grouped = build("auto", workers=2, storage="disk")
    try:
        assert grouped.folded_batch_mb == 0.001
        assert grouped.mf.gdf_folded_batch_mb == 0.001
        assert grouped.disk_bytes > 0
        assert single_pair.multi_q_build_timings == []
        assert bounded_q_local.multi_q_build_timings == []
        assert sorted(
            timing["stream_pair_batch_pair_counts"]
            for timing in single_pair.build_timings.values()
        ) == [[1], [1, 1]]
        assert sorted(
            timing["stream_pair_batch_pair_counts"]
            for timing in grouped.build_timings.values()
        ) == [[1], [2]]
        assert sorted(
            timing["stream_pair_source_pair_count"]
            for timing in grouped.build_timings.values()
        ) == [1, 2]
        assert sorted(
            timing["stream_pair_self_opposite_pair_reuses"]
            for timing in grouped.build_timings.values()
        ) == [0, 1]
        assert all(
            timing["stream_pair_batches"] == 1
            for timing in grouped.build_timings.values()
        )
        assert len(grouped.multi_q_build_timings) == 1
        multi_q = grouped.multi_q_build_timings[0]
        assert multi_q["q_indices"] == [0, 1]
        assert multi_q["consumer"] == "periodic_gdf_bounded_j3c_cderi"
        assert multi_q["three_center_sr_folded"] is True
        assert multi_q["three_center_sr_folded_pipeline"] == (
            "aux_fft_consumer"
        )
        assert multi_q["three_center_sr_folded_storage_bytes"] == 0
        assert multi_q["three_center_sr_folded_batch_count"] > 1
        assert multi_q["bounded_j3c_storage_bytes"] > 0
        assert (
            0
            < multi_q["bounded_j3c_q_block_peak_bytes"]
            < multi_q["bounded_j3c_storage_bytes"]
        )
        assert multi_q["aux_metric_sr_grouped_seconds"] > 0.0
        assert multi_q["aux_metric_sr_grouped_batches"] >= 1
        assert multi_q["aux_metric_sr_grouped_batch_size"] >= 1
        assert (
            multi_q["aux_metric_sr_grouped_workspace_upper_bound_bytes"]
            <= int(grouped.stream_pair_batch_mb * 1.0e6)
        )
        assert multi_q["three_center_short_range_workers"] <= min(
            timing["inner_worker_cap"]
            for timing in grouped.build_timings.values()
        )
        assert all(
            timing["direct_cderi_stream"] is True
            and timing["direct_cderi_pipeline"]
            == "aux_fft_bounded_j3c_whiten"
            for timing in grouped.build_timings.values()
        )
        assert any(
            timing.get("direct_cderi_disk_shard", False)
            for timing in grouped.build_timings.values()
        )
        assert len(grouped.cache_files) < len(grouped._cderi_cache)
        assert all(
            timing["direct_cderi_global_budget_bytes"]
            == int(grouped.stream_pair_batch_mb * 1.0e6)
            for timing in grouped.build_timings.values()
        )
        assert not _gdf_mf_cache(grouped.mf, "three_center_ao_short_range")
        assert not _gdf_mf_cache(grouped.mf, "aux_metric_short_range")
        assert not _gdf_mf_cache(
            grouped.mf,
            "three_center_ao_short_range_folded",
        )
        assert all(
            timing["parallel_policy"] == "bounded_nested"
            and timing["prebuild_outer_workers"]
            * timing["inner_worker_cap"]
            <= (os.cpu_count() or 1)
            for timing in grouped.build_timings.values()
        )
        assert all(
            timing["aggregate_stream_pair_batch_mb"]
            == pytest.approx(grouped.stream_pair_batch_mb)
            and timing["stream_pair_budget_scope"] == "global"
            for timing in grouped.build_timings.values()
        )
        for q_index in range(len(grouped.qpts)):
            for k_index, kq_index in grouped.pair_keys(q_index):
                grouped_factor = grouped.cderi(q_index, k_index, kq_index)
                reference_factor = single_pair.cderi(
                    q_index, k_index, kq_index
                )
                grouped_flat = grouped_factor.reshape(
                    grouped_factor.shape[0], -1
                )
                reference_flat = reference_factor.reshape(
                    reference_factor.shape[0], -1
                )
                np.testing.assert_allclose(
                    grouped_flat.T @ grouped_flat.conj(),
                    reference_flat.T @ reference_flat.conj(),
                    atol=1.0e-12,
                )
                bounded_factor = bounded_q_local.cderi(
                    q_index, k_index, kq_index
                )
                bounded_flat = bounded_factor.reshape(
                    bounded_factor.shape[0], -1
                )
                np.testing.assert_allclose(
                    grouped_flat.T @ grouped_flat.conj(),
                    bounded_flat.T @ bounded_flat.conj(),
                    atol=1.0e-12,
                )
    finally:
        single_pair.close()
        bounded_q_local.close()
        grouped.close()


def _bounded_folded_gdf(cache_dir, *, storage="disk"):
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        kpts=cell.make_kpts((2, 1, 1), gamma_centered=True),
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        precision=1.0e-8,
        storage=storage,
        max_memory_mb=0.0,
        cache_dir=str(cache_dir),
        stream_pairs=True,
        stream_pair_batch_mb=0.03,
        folded_batch_mb=0.001,
    )
    mf.gdf_folded_min_kpts = 1
    return mf.with_df


def test_gdf_build_rolls_back_interruption_and_removes_spools(monkeypatch, tmp_path):
    import pyqed.pbc.gw.integrals as integrals

    backend = _bounded_folded_gdf(tmp_path)
    stream = integrals._gdf_stream_three_center_ao_short_range_folded

    def interrupt_stream(
        space,
        aux,
        omega,
        short_range_cut,
        consumer,
        **kwargs,
    ):
        def interrupt_after_first_batch(*args):
            consumer(*args)
            raise KeyboardInterrupt("test interruption")

        return stream(
            space,
            aux,
            omega,
            short_range_cut,
            interrupt_after_first_batch,
            **kwargs,
        )

    monkeypatch.setattr(
        integrals,
        "_gdf_stream_three_center_ao_short_range_folded",
        interrupt_stream,
    )
    with pytest.raises(KeyboardInterrupt, match="test interruption"):
        backend.build(workers=2)

    assert backend.memory_bytes == 0
    assert backend.disk_bytes == 0
    assert not backend.cache_files
    assert not backend._cderi_cache
    assert not backend._q_metadata
    assert not list(tmp_path.iterdir())


def test_gdf_disk_failure_is_atomic_and_rolls_back(monkeypatch, tmp_path):
    backend = _bounded_folded_gdf(tmp_path)
    open_memmap = np.lib.format.open_memmap

    def fail_factor_write(filename, *args, **kwargs):
        path = os.fspath(filename)
        parent = os.path.basename(os.path.dirname(path))
        if parent.startswith("pyqed-gdf-") and not parent.startswith(
            "pyqed-gdf-j3c-"
        ):
            with open(path, "wb") as stream:
                stream.write(b"partial")
            raise OSError("simulated disk full")
        return open_memmap(filename, *args, **kwargs)

    monkeypatch.setattr(np.lib.format, "open_memmap", fail_factor_write)
    with pytest.raises(OSError, match="simulated disk full"):
        backend.build(workers=2)

    assert backend.memory_bytes == 0
    assert backend.disk_bytes == 0
    assert not backend.cache_files
    assert not backend._cderi_cache
    assert not list(tmp_path.iterdir())


def test_gdf_concurrent_builds_share_one_transaction(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    backend = _bounded_folded_gdf(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as executor:
        completed = list(executor.map(lambda _index: backend.build(workers=2), range(2)))
    try:
        assert completed == [backend, backend]
        assert backend.memory_bytes == 0
        assert backend.disk_bytes > 0
        assert len(backend.cache_files) == len(set(backend.cache_files))
        assert not any(path.name.startswith(".") for path in tmp_path.rglob("*"))
        for q_index in range(len(backend.qpts)):
            for k_index, kq_index in backend.pair_keys(q_index):
                assert np.all(
                    np.isfinite(backend.cderi(q_index, k_index, kq_index))
                )
    finally:
        backend.close()
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize(
    "scaled_kpts",
    (
        [[-0.25, 0.0, 0.0], [0.25, 0.0, 0.0]],
        [
            [0.0, 0.0, 0.0],
            [1.0 / 3.0, 1.0 / 3.0, 0.0],
            [-1.0 / 3.0, -1.0 / 3.0, 0.0],
        ],
    ),
)
def test_gdf_direct_bloch_fallback_builds_shifted_and_noncommensurate_kpoints(
    scaled_kpts,
):
    from pyqed.pbc.gw.integrals import _gdf_gamma_folded_mesh
    from pyqed.qchem.pbc.gdf import _KMeshSpace

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    reciprocal = 2.0 * np.pi * np.linalg.inv(cell.lattice_vectors).T
    kpts = np.asarray(scaled_kpts, dtype=float) @ reciprocal
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
    ).density_fit(
        auxbasis="def2-svp-jkfit",
        precision=1.0e-8,
        storage="memory",
        stream_pairs=True,
    )
    mf.gdf_folded_min_kpts = 1
    assert _gdf_gamma_folded_mesh(_KMeshSpace(mf).reference) is None
    backend = mf.with_df.build(workers=2)
    try:
        assert all(
            not row.get("three_center_sr_folded", False)
            for row in backend.multi_q_build_timings
        )
        density = np.asarray([np.eye(cell.nao)] * len(kpts), dtype=complex)
        direct_j, direct_k = backend.get_jk(density)
        assert np.all(np.isfinite(direct_j))
        assert np.all(np.isfinite(direct_k))
        np.testing.assert_allclose(
            direct_j,
            direct_j.conj().transpose(0, 2, 1),
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            direct_k,
            direct_k.conj().transpose(0, 2, 1),
            atol=1.0e-12,
        )
    finally:
        backend.close()


def test_periodic_cell_rejects_unimplemented_two_dimensional_coulomb():
    with pytest.raises(NotImplementedError, match="dimension=1 or dimension=3"):
        Cell(
            atom="H 0 0 0; H 1.4 0 0",
            a=np.diag([5.0, 5.0, 15.0]),
            basis="sto-3g",
            unit="bohr",
            dimension=2,
        ).build()


def test_optional_pyscf_3d_hydrogen_gdf_krhf_energy():
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    from pyqed.pbc.gw.integrals import (
        _gdf_normalize_auxbasis_name,
        _pyscf_builtin_basis_dict,
        _pyscf_cell_from_reference,
    )
    from pyqed.pbc.gw.response import KPointTransitionSpace

    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.KRHF(
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
        jk_builder="gdf",
    )
    mf.gdf_auxbasis = "def2-svp-jkfit"
    mf.gdf_precision = 1.0e-8
    mf.run(max_cycle=40, conv_tol=1.0e-10, conv_tol_dm=1.0e-8)

    ref = KPointTransitionSpace(mf, qpts="mesh").reference
    pyscf_cell = _pyscf_cell_from_reference(ref)
    auxbasis = _pyscf_builtin_basis_dict(
        _gdf_normalize_auxbasis_name(mf.gdf_auxbasis),
        cell._atom_symbols,
    )
    ref_mf = pyscf_pbc_scf.KRHF(
        pyscf_cell,
        kpts=kpts,
        exxdiv="ewald",
    ).density_fit(auxbasis=auxbasis)
    ref_mf.conv_tol = 1.0e-10
    ref_mf.verbose = 0
    ref_mf.kernel()

    assert mf.converged
    assert ref_mf.converged
    np.testing.assert_allclose(
        mf.madelung,
        pytest.importorskip("pyscf.pbc.tools.pbc").madelung(pyscf_cell, kpts),
        atol=1.0e-10,
    )
    assert abs(mf.e_tot - ref_mf.e_tot) < 1.0e-7


def test_pyscf_basis_bridge_accepts_native_contraction_dictionary():
    from pyqed.pbc.gw.integrals import _pyscf_builtin_basis_dict

    basis = {
        "H": [
            (
                0,
                np.asarray([1.5, 0.4]),
                np.asarray([[0.8, 0.1], [0.2, 0.9]]),
            )
        ]
    }

    converted = _pyscf_builtin_basis_dict(basis, ("H", "H"))

    assert tuple(converted) == ("H",)
    assert converted["H"] == [
        [0, [1.5, 0.8], [0.4, 0.2]],
        [0, [1.5, 0.1], [0.4, 0.9]],
    ]


def test_optional_pyscf_3d_hydrogen_gamma_reference_scale():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = cell.RHF(method="ewald", eta=0.5, real_cut=0, recip_cut=3).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.RHF(ref_cell).run(conv_tol=1e-10)

    assert ref_mf.converged
    assert mf.converged
    assert np.isfinite(ref_mf.e_tot)
    assert np.isfinite(mf.e_tot)
    assert abs(mf.e_tot - ref_mf.e_tot) < 1.0


def test_optional_pyscf_3d_hydrogen_centered_krhf_reference_scale():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    kpts = cell.make_kpts((2, 1, 1))
    mf = cell.RHF(
        method="ewald",
        kpts=kpts,
        eta=0.5,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
        jk_builder="reciprocal",
    ).run(
        max_cycle=80,
        conv_tol=1e-10,
        conv_tol_dm=1e-8,
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.KRHF(ref_cell, kpts=kpts).run(conv_tol=1e-10)

    assert ref_mf.converged
    assert mf.converged
    assert abs(mf.e_tot - ref_mf.e_tot) < 5e-6

    scaled_path = np.column_stack(
        [np.linspace(-0.5, 0.5, 5), np.zeros(5), np.zeros(5)]
    )
    native_bands = mf.band_structure(scaled_kpts=scaled_path, exchange="finite_q")
    recip = 2.0 * np.pi * np.linalg.inv(lattice).T
    ref_bands, _ = ref_mf.get_bands(scaled_path @ recip)
    assert np.max(np.abs(native_bands["mo_energy"] - ref_bands)) < 1e-5


def test_optional_pyscf_3d_hydrogen_one_body_reference():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_scf = pytest.importorskip("pyscf.pbc.scf")

    from pyqed.qchem.pbc.hf.ewald_rhf import EwaldRHF

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = EwaldRHF(cell, eta=0.5, real_cut=1, recip_cut=5)
    mf._validate()
    mf._periodic_setup()
    mf._build_one_body_blocks()
    overlap = mf._fourier_sum(mf._s_r, np.zeros(3))
    hcore = (
        mf._fourier_sum(mf._t_r, np.zeros(3))
        + mf._fourier_sum(mf._vne_sr_r, np.zeros(3))
        + mf._reciprocal_nuclear_attraction(np.zeros(3))
        + mf._nuclear_background_hcore(overlap)
    )

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()
    ref_mf = pyscf_pbc_scf.RHF(ref_cell)

    assert np.linalg.norm(overlap - ref_mf.get_ovlp()) < 2e-3
    assert np.linalg.norm(hcore - ref_mf.get_hcore()) < 3e-3


def test_optional_pyscf_3d_hydrogen_madelung_reference():
    pyscf_pbc_gto = pytest.importorskip("pyscf.pbc.gto")
    pyscf_pbc_tools = pytest.importorskip("pyscf.pbc.tools.pbc")

    from pyqed.qchem.pbc.hf.ewald_rhf import EwaldRHF

    lattice = np.diag([5.0, 5.0, 5.0])
    atom = "H 0 0 0; H 1.4 0 0"
    cell = Cell(
        atom=atom,
        a=lattice,
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    mf = EwaldRHF(cell, eta=0.5, real_cut=1, recip_cut=5)
    mf._validate()

    ref_cell = pyscf_pbc_gto.Cell()
    ref_cell.atom = atom
    ref_cell.a = lattice
    ref_cell.basis = "sto-3g"
    ref_cell.unit = "B"
    ref_cell.charge = 0
    ref_cell.spin = 0
    ref_cell.verbose = 0
    ref_cell.build()

    np.testing.assert_allclose(
        mf._madelung(),
        pyscf_pbc_tools.madelung(ref_cell, np.zeros((1, 3))),
        atol=6e-4,
    )


def test_native_1d_inf_vacuum_probe_madelung_matches_reference_value():
    lattice = np.diag([4.0, 20.0, 20.0])
    probe_energy = ewald_nuclear_repulsion_1d_inf_vacuum(
        np.asarray([1.0]),
        np.zeros((1, 3)),
        lattice,
        eta=0.31622776601683794,
        real_cut=5,
        mesh=(15, 18, 18),
    )
    madelung = -2.0 * probe_energy
    np.testing.assert_allclose(madelung, -5.585196565523321, atol=1e-12)


def test_cartesian_p_d_pair_fourier_matches_overlap_at_zero_g():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)

    np.testing.assert_allclose(gaussian_pair_ft(p_fn, d_fn, np.zeros(3)), S(p_fn, d_fn), atol=1e-12)


def test_cartesian_p_d_short_range_point_charge_matches_full_at_eta_zero():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)
    center = np.asarray([0.2, -0.1, 0.3])

    sr = short_range_point_charge(p_fn, d_fn, center, eta=0.0)
    full = point_charge(p_fn, d_fn, center)
    np.testing.assert_allclose(sr, full, atol=1e-10)


def test_short_range_eri_s_shortcut_matches_generic_cartesian_integral():
    cell = Cell(
        atom="H 0 0 0; H 1.4 0 0",
        a=np.diag([5.0, 5.0, 5.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
    ).build()
    basis = cell.unit_molecule._bas

    expected = short_range_eri(basis[0], basis[1], basis[0], basis[1], eta=0.5)
    actual = short_range_eri_s(basis[0], basis[1], basis[0], basis[1], eta=0.5)
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_cartesian_p_d_short_range_eri_matches_full_at_eta_zero():
    chain = Chain(
        atom="C 0 0 0",
        a=8.0,
        basis="631g*",
        unit="bohr",
        spin=0,
        vacuum=20.0,
        integral_options={"coord_type": "cartesian"},
    ).build()
    basis = chain.unit_molecule._bas
    s_fn = next(fn for fn in basis if sum(fn.shell) == 0)
    p_fn = next(fn for fn in basis if sum(fn.shell) == 1)
    d_fn = next(fn for fn in basis if sum(fn.shell) == 2)

    sr = short_range_eri(d_fn, p_fn, s_fn, d_fn, eta=0.0)
    full = ERI(d_fn, p_fn, s_fn, d_fn)
    np.testing.assert_allclose(sr, full, atol=1e-10)
