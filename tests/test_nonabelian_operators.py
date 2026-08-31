import numpy as np
import pytest

from pyqed.mps.nonabelian import (
    Leg,
    ReducedTensorOperator,
    build_random_spatial_mps,
    compose_site_operators,
    coupled_reduced_tensor_product,
    physical_leg_from_spatial_orbital,
    reduced_spatial_fermion_annihilation,
    spatial_identity,
    spatial_number,
    spatial_number_up,
    spatial_number_down,
    spatial_double_occupancy,
    spatial_spin_square,
    spatial_projector,
    spatial_parity,
    spatial_annihilate_up,
    spatial_create_up,
    spatial_annihilate_down,
    spatial_create_down,
    time_reversed_reduced_operator,
)
from pyqed.mps.nonabelian.states import FullyReducedSpatialOrbitalSite
from pyqed.mps.su2 import SpatialOrbitalSite, SU2Irrep


def test_physical_leg_from_spatial_orbital_matches_site_structure():
    site = SpatialOrbitalSite()
    leg = physical_leg_from_spatial_orbital(site)

    assert isinstance(leg, Leg)
    assert leg.sectors == site.qn
    assert leg.sector_dims == {
        site.qn[0]: 1,
        site.qn[1]: 2,
        site.qn[2]: 1,
    }
    assert leg.total_dim == 4


def test_physical_leg_from_spatial_orbital_accepts_canonical_site_tensor():
    tensor_site = build_random_spatial_mps(2, seed=3)[0]
    leg = physical_leg_from_spatial_orbital(tensor_site)

    assert isinstance(leg, Leg)
    assert leg == physical_leg_from_spatial_orbital()


def test_spatial_identity_is_four_by_four_identity():
    op = spatial_identity()
    np.testing.assert_allclose(op.as_dense(), np.eye(4))


def test_reduced_spatial_fermion_annihilation_exposes_spinor_components():
    op = reduced_spatial_fermion_annihilation()

    assert isinstance(op, ReducedTensorOperator)
    assert op.components == (1, -1)

    np.testing.assert_allclose(
        op.component(-1).as_dense(),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        op.component(1).as_dense(),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )


def test_spatial_number_matches_expected_local_matrix():
    op = spatial_number()
    np.testing.assert_allclose(
        op.as_dense(),
        np.diag([0.0, 1.0, 1.0, 2.0]),
    )


def test_fully_reduced_spinfree_bilinear_recovers_local_number():
    leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    annihilation = reduced_spatial_fermion_annihilation(leg)
    scalar = coupled_reduced_tensor_product(
        annihilation.adjoint(),
        time_reversed_reduced_operator(annihilation),
        SU2Irrep(0),
    )

    np.testing.assert_allclose(
        -np.sqrt(2.0) * scalar.component(0).as_dense(),
        spatial_number(leg).as_dense(),
        atol=1.0e-12,
    )


def test_spatial_double_occupancy_matches_expected_local_matrix():
    op = spatial_double_occupancy()
    np.testing.assert_allclose(
        op.as_dense(),
        np.diag([0.0, 0.0, 0.0, 1.0]),
    )


def test_spatial_spin_square_matches_expected_local_matrix():
    op = spatial_spin_square()
    np.testing.assert_allclose(
        op.as_dense(),
        np.diag([0.0, 0.75, 0.75, 0.0]),
    )


@pytest.mark.parametrize(
    ("occupancy", "diag"),
    [
        ("empty", [1.0, 0.0, 0.0, 0.0]),
        ("single", [0.0, 1.0, 1.0, 0.0]),
        ("double", [0.0, 0.0, 0.0, 1.0]),
    ],
)
def test_spatial_projectors_match_expected_local_matrices(occupancy, diag):
    op = spatial_projector(occupancy)
    np.testing.assert_allclose(op.as_dense(), np.diag(diag))


def test_spatial_projector_rejects_unknown_occupancy():
    with pytest.raises(ValueError, match="expected empty/single/double"):
        spatial_projector("triply")


def test_spatial_parity_matches_expected_local_matrix():
    op = spatial_parity()
    np.testing.assert_allclose(op.as_dense(), np.diag([1.0, -1.0, -1.0, 1.0]))


def test_spin_resolved_number_operators_match_expected_local_matrices():
    np.testing.assert_allclose(
        spatial_number_up().as_dense(),
        np.diag([0.0, 1.0, 0.0, 1.0]),
    )
    np.testing.assert_allclose(
        spatial_number_down().as_dense(),
        np.diag([0.0, 0.0, 1.0, 1.0]),
    )


def test_spatial_fermionic_creation_annihilation_match_expected_matrices():
    np.testing.assert_allclose(
        spatial_annihilate_up().as_dense(),
        np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        spatial_annihilate_down().as_dense(),
        np.array(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, -1.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ),
    )
    np.testing.assert_allclose(
        spatial_create_up().as_dense(),
        spatial_annihilate_up().as_dense().T.conj(),
    )
    np.testing.assert_allclose(
        spatial_create_down().as_dense(),
        spatial_annihilate_down().as_dense().T.conj(),
    )


def test_spatial_fermion_operators_obey_local_anticommutation():
    identity = spatial_identity().as_dense()
    cu = spatial_annihilate_up().as_dense()
    cdu = spatial_create_up().as_dense()
    cd = spatial_annihilate_down().as_dense()
    cdd = spatial_create_down().as_dense()

    np.testing.assert_allclose(cu @ cdu + cdu @ cu, identity)
    np.testing.assert_allclose(cd @ cdd + cdd @ cd, identity)
    np.testing.assert_allclose(cu @ cdd + cdd @ cu, np.zeros_like(identity), atol=1.0e-12)
    np.testing.assert_allclose(cd @ cdu + cdu @ cd, np.zeros_like(identity), atol=1.0e-12)


def test_compose_site_operators_matches_dense_multiplication():
    composed = compose_site_operators(spatial_annihilate_up(), spatial_parity())
    np.testing.assert_allclose(
        composed.as_dense(),
        spatial_annihilate_up().as_dense() @ spatial_parity().as_dense(),
    )
