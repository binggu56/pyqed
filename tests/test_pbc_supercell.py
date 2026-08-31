import numpy as np

from pyqed.qchem.pbc import (
    Cell,
    CommensurateSupercell,
    commensurate_gdf_q_derivative,
    gdf_q_derivative,
)
from pyqed.pbc.gw import (
    KPointTransitionSpace,
    commensurate_gdf_bare_tda_kernel_derivative,
    periodic_tda_operator,
    validate_commensurate_gdf_screened_tda_kernel_derivative,
)


def _cell():
    return Cell(
        atom="H 0 0 0; H 1.2 0.1 0",
        a=np.diag([4.0, 5.0, 6.0]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()


def test_commensurate_supercell_geometry_and_bloch_embeddings():
    cell = _cell()
    transform = CommensurateSupercell(cell, (3, 1, 1))
    kpoints = cell.make_kpts((3, 1, 1))

    assert transform.super_symbols == ("H", "H") * 3
    assert transform.super_positions.shape == (6, 3)
    np.testing.assert_allclose(transform.super_lattice, np.diag([12.0, 5.0, 6.0]))
    for left_index, left_k in enumerate(kpoints):
        left = transform.bloch_embedding(left_k)
        for right_index, right_k in enumerate(kpoints):
            right = transform.bloch_embedding(right_k)
            expected = np.eye(cell.nao) if left_index == right_index else 0.0
            np.testing.assert_allclose(left.conj().T @ right, expected, atol=2.0e-15)


def test_commensurate_supercell_preserves_shifted_mesh_twist():
    cell = _cell()
    transform = CommensurateSupercell(cell, (2, 1, 1))
    kpoints = cell.make_kpts((2, 1, 1))
    twist = transform.common_twist(kpoints)
    scaled_twist = twist @ np.linalg.inv(transform.super_reciprocal_vectors)

    np.testing.assert_allclose(scaled_twist, [-0.5, 0.0, 0.0], atol=2.0e-15)
    gamma_transform = CommensurateSupercell(cell, (3, 1, 1))
    np.testing.assert_allclose(
        gamma_transform.common_twist(cell.make_kpts((3, 1, 1))),
        np.zeros(3),
        atol=2.0e-15,
    )
    with np.testing.assert_raises_regex(ValueError, "must contain 2"):
        transform.common_twist(kpoints[:1])


def test_commensurate_supercell_operator_round_trip_at_general_q():
    rng = np.random.default_rng(8)
    cell = _cell()
    transform = CommensurateSupercell(cell, (3, 1, 1))
    kpoints = cell.make_kpts((3, 1, 1))
    qpoint = 2.0 * np.pi * np.asarray([1.0 / 12.0, 0.0, 0.0])
    blocks = tuple(
        rng.normal(size=(cell.nao, cell.nao))
        + 1.0j * rng.normal(size=(cell.nao, cell.nao))
        for _ in kpoints
    )

    matrix = transform.embed_operator(blocks, kpoints, qpoint)
    recovered = transform.fold_operator(matrix, kpoints, qpoint)

    for actual, expected in zip(recovered, blocks):
        np.testing.assert_allclose(actual, expected, atol=3.0e-15)


def test_commensurate_supercell_density_round_trip_and_q_validation():
    rng = np.random.default_rng(12)
    cell = _cell()
    transform = CommensurateSupercell(cell, (2, 2, 1))
    kpoints = cell.make_kpts((2, 2, 1))
    densities = []
    for _kpoint in kpoints:
        trial = rng.normal(size=(cell.nao, cell.nao))
        densities.append(trial + trial.T)

    super_density = transform.embed_density(densities, kpoints)
    recovered = transform.fold_operator(super_density, kpoints, np.zeros(3))

    for actual, expected in zip(recovered, densities):
        np.testing.assert_allclose(actual, expected, atol=3.0e-15)
    bad_q = 0.13 * transform.reciprocal_vectors[0]
    with np.testing.assert_raises_regex(ValueError, "not commensurate"):
        transform.validate_qpoint(bad_q)


def test_commensurate_gdf_q_derivative_builds_self_consistent_ao_blocks():
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=(2, 1, 1),
        eta=0.7,
        real_cut=2,
        pair_cut=2,
        recip_cut=8,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="sto-3g",
        reciprocal_kernel="full",
        recip_cut=8,
        pair_cut=2,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)
    qpoint = mf.with_df.qpts[1]
    derivative = commensurate_gdf_q_derivative(
        mf,
        qpoint,
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )

    assert derivative.success
    assert derivative.info["mesh"] == (2, 1, 1)
    assert derivative.info["temporary_supercell_nao"] == 4
    assert np.linalg.norm(derivative.info["supercell_twist"]) > 1.0e-8
    assert derivative.response.converged
    assert len(derivative.fock_derivative) == 2
    for name in (
        "explicit_fock_derivative",
        "induced_fock_derivative",
        "fock_derivative",
        "overlap_derivative",
    ):
        blocks = getattr(derivative, name)
        assert all(block.shape == (2, 2) for block in blocks)
        assert all(np.all(np.isfinite(block)) for block in blocks)
        np.testing.assert_allclose(
            blocks[1],
            blocks[0].conj().T,
            atol=2.0e-12,
            err_msg=name,
        )
    assert derivative.info["star_symmetry_residuals"]["induced_fock"] < 2.0e-12
    assert max(
        value
        for name, value in derivative.info["reference_residuals"].items()
        if name.endswith("_relative")
    ) < 1.0e-7
    space = KPointTransitionSpace(mf, qpts="mesh")
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=0.0,
        coulomb_component="gdf",
    )
    kernel1 = commensurate_gdf_bare_tda_kernel_derivative(
        operator,
        derivative,
    )
    target_q_index = space.find_qpoint_index(qpoint)
    assert kernel1.shape == (
        len(space.transitions(target_q_index)),
        len(space.transitions(zero_q_index)),
    )
    assert np.all(np.isfinite(kernel1))
    assert np.max(np.abs(kernel1)) > 1.0e-8
    assert derivative.gdf_bare_kernel_derivative_info[
        "screening_derivative"
    ] == "frozen"
    q_factors = derivative.gdf_q_derivative_factors
    assert q_factors.info["backend"] == (
        "primitive_cell_gdf_q_derivative_factors"
    )
    assert q_factors.info["producer"] == "primitive_cell_reciprocal_gdf"
    assert q_factors.info["temporary_supercell_factor_bytes"] == 0
    assert q_factors.info["released_supercell_factor_bytes"] > 0
    assert q_factors.info["engine_cached_bytes"] > 0
    assert q_factors.info["pair_factor_count"] > 0
    assert q_factors.info["retained_pair_bytes"] > 0

    screened_operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )
    validation = validate_commensurate_gdf_screened_tda_kernel_derivative(
        screened_operator,
        derivative,
        steps=(1.0e-3,),
    )
    assert derivative.gdf_q_derivative_factors is q_factors
    assert validation["directions"] == ("cosine",)
    assert validation["relative_error"][0] < 2.0e-7
    # Even meshes have one unmatched reciprocal Nyquist endpoint between the
    # primitive and finite-supercell representations.
    assert validation["component_errors"]["bare"]["relative"][0] < 2.0e-7
    assert (
        validation["component_errors"]["screened"]["relative"][0]
        < 2.0e-6
    )


def test_screened_gdf_derivative_matches_general_q_finite_difference():
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mf = cell.KRHF(
        nk=(3, 1, 1),
        eta=0.7,
        real_cut=2,
        pair_cut=2,
        recip_cut=5,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    ).density_fit(
        auxbasis="sto-3g",
        reciprocal_kernel="full",
        recip_cut=5,
        pair_cut=2,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    ).run(max_cycle=80, conv_tol=1.0e-12, conv_tol_dm=1.0e-10)
    space = KPointTransitionSpace(mf, qpts="mesh")
    zero_q_index = space.find_qpoint_index(np.zeros(3))
    phonon_q_index = next(
        index for index in range(space.nqpts) if index != zero_q_index
    )
    q_derivative = commensurate_gdf_q_derivative(
        mf,
        space.qpts[phonon_q_index],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )
    primitive_q_derivative = gdf_q_derivative(
        mf,
        space.qpts[phonon_q_index],
        [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]],
        cphf_tol=1.0e-10,
    )
    assert primitive_q_derivative.info["backend"] == (
        "primitive_cell_full_reciprocal_gdf"
    )
    assert primitive_q_derivative.info["temporary_supercell_nao"] == 0
    assert primitive_q_derivative.info["temporary_supercell_naux"] == 0
    assert primitive_q_derivative.primitive_engine is not None
    for name in (
        "overlap_derivative",
        "explicit_fock_derivative",
        "induced_fock_derivative",
        "fock_derivative",
    ):
        for actual, expected in zip(
            getattr(primitive_q_derivative, name),
            getattr(q_derivative, name),
        ):
            np.testing.assert_allclose(
                actual,
                expected,
                atol=3.0e-9,
                rtol=0.0,
                err_msg=name,
            )
    operator = periodic_tda_operator(
        space,
        q_index=zero_q_index,
        direct_scale=2.0,
        exchange_scale=1.0,
        screened_exchange_scale=1.0,
        coulomb_component="gdf",
    )

    validation = validate_commensurate_gdf_screened_tda_kernel_derivative(
        operator,
        q_derivative,
        steps=(1.0e-3,),
    )
    with np.testing.assert_raises_regex(ValueError, "representation_tol"):
        validate_commensurate_gdf_screened_tda_kernel_derivative(
            operator,
            q_derivative,
            steps=(1.0e-3,),
            representation_tol=0.0,
        )

    assert validation["directions"] == ("cosine", "sine")
    assert validation["relative_error"][0] < 2.0e-8
    assert validation["component_errors"]["bare"]["relative"][0] < 2.0e-8
    assert (
        validation["component_errors"]["screened"]["relative"][0]
        < 2.0e-8
    )
    assert np.isfinite(
        validation["step_details"][0]["one_body_leakage_norm"]
    )
    assert validation["largest_reference_residual"] < 2.0e-9
    assert validation["step_details"][0]["one_body_leakage_norm"] < 2.0e-8

    q_factors = q_derivative.gdf_q_derivative_factors
    assert q_factors.info["producer"] == "primitive_cell_reciprocal_gdf"
    engine = q_factors.primitive_engine
    zero_ao, plus_ao, _minus_ao, transfers = engine.pair_ao_factors(0, 0)
    left = np.asarray(mf.mo_coeff[0][:, 0])
    right = np.asarray(mf.mo_coeff[0][:, 1])
    primitive_zero = np.einsum(
        "Ppq,p,q->P", zero_ao, left.conj(), right, optimize=True
    )
    primitive_plus = np.einsum(
        "Ppq,p,q->P", plus_ao, left.conj(), right, optimize=True
    )

    commensurate = q_derivative.gradient.gdf_derivative_factors(
        require_scf=False
    )
    weights = q_derivative.transform.mode_weights(
        q_derivative.cartesian_mode,
        q_derivative.qpoint,
    ).reshape(q_derivative.transform.ncell * q_derivative.transform.natom, 3)
    super_three_center1 = np.einsum(
        "Ax,AxPpq->Ppq",
        weights,
        commensurate["three_center1"],
        optimize=True,
    )
    left_super = q_derivative.transform.bloch_embedding(mf.kpts[0]) @ left
    right_super = q_derivative.transform.bloch_embedding(mf.kpts[0]) @ right
    super_zero = np.einsum(
        "Ppq,p,q->P",
        commensurate["three_center"],
        left_super.conj(),
        right_super,
        optimize=True,
    )
    super_plus = np.einsum(
        "Ppq,p,q->P",
        super_three_center1,
        left_super.conj(),
        right_super,
        optimize=True,
    )
    ncell = q_derivative.transform.ncell
    naux = engine.aux.naux
    identity = np.eye(naux, dtype=np.complex128)

    def auxiliary_embedding(q_index):
        phases = np.exp(
            1.0j
            * (
                q_derivative.transform.translation_vectors
                @ np.asarray(space.qpts[q_index])
            )
        )
        return (
            (phases[:, None, None] * identity[None]).reshape(
                ncell * naux, naux
            )
            / np.sqrt(float(ncell))
        )

    zero_embedding = auxiliary_embedding(transfers[0])
    plus_embedding = auxiliary_embedding(transfers[1])
    np.testing.assert_allclose(
        primitive_zero,
        np.sqrt(float(ncell)) * (zero_embedding.conj().T @ super_zero),
        atol=2.0e-11,
    )
    np.testing.assert_allclose(
        primitive_plus,
        np.sqrt(float(ncell)) * (plus_embedding.conj().T @ super_plus),
        atol=2.0e-11,
    )

    super_inverse1 = np.einsum(
        "Ax,AxPQ->PQ",
        weights,
        commensurate["inverse_metric1"],
        optimize=True,
    )
    projected_inverse1 = (
        plus_embedding.conj().T @ super_inverse1 @ zero_embedding
    )
    np.testing.assert_allclose(
        engine.inverse_metric_derivative(transfers[0]),
        projected_inverse1,
        atol=2.0e-11,
    )
