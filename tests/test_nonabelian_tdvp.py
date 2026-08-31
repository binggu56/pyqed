import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.mps.mps import MPO, _mpo_to_dense_operator
from pyqed.mps.nonabelian import (
    MPS,
    build_random_spatial_mps,
    build_spatial_one_body_reduced_mpo,
    build_spatial_hubbard_mpo,
    scale_mpo_chain,
    spatial_target_sector,
    sum_mpo_chains,
)
from pyqed.mps.nonabelian.tdvp import two_site_tdvp_step
from pyqed.mps.nonabelian.su2_kernel import cpp_available
from pyqed.qchem.dmrg.dmrg import (
    _dense_cores_from_nonabelian_mpo,
    _nonabelian_mps_to_dense_vector,
)


def _normalized_random_state(nsites, *, seed=3):
    target = spatial_target_sector(2, 0)
    state = MPS.from_tensors(
        build_random_spatial_mps(
            nsites,
            target_sector=target,
            bond_multiplicity=2,
            seed=seed,
        ),
        target_sector=target,
    )
    vector = _nonabelian_mps_to_dense_vector(state)
    scale = 1.0 / np.linalg.norm(vector)
    state.tensors[0].data = {
        key: np.asarray(block) * scale
        for key, block in state.tensors[0].data.items()
    }
    return state


def test_two_site_su2_tdvp_matches_dense_reference_without_truncation():
    state = _normalized_random_state(2)
    mpo = build_spatial_hubbard_mpo(
        state.tensors,
        hopping_t=0.3,
        onsite_u=1.0,
    )
    dense_h = _mpo_to_dense_operator(
        MPO(_dense_cores_from_nonabelian_mpo(mpo), homogeneous=False)
    )
    initial = _nonabelian_mps_to_dense_vector(state)
    dt = 0.013

    propagated, info = two_site_tdvp_step(
        state,
        mpo,
        dt,
        max_bond=16,
        krylov_dim=20,
        krylov_tol=1.0e-14,
    )

    reference = expm(-1j * dt * dense_h) @ initial
    actual = _nonabelian_mps_to_dense_vector(propagated)
    np.testing.assert_allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)
    assert info["native_reduced"] is True
    assert info["truncation_error"] < 1.0e-14
    if cpp_available():
        assert {
            update["local_objective"]["propagation_backend"]
            for half_sweep in info["half_sweeps"]
            for update in half_sweep["updates"]
        } == {"cpp"}


def test_three_site_su2_tdvp_is_time_reversible_without_truncation():
    state = _normalized_random_state(3)
    mpo = build_spatial_hubbard_mpo(
        state.tensors,
        hopping_t=0.3,
        onsite_u=1.0,
    )
    dt = 0.002

    forward, info = two_site_tdvp_step(state, mpo, dt, max_bond=16)
    backward, _ = two_site_tdvp_step(forward, mpo, -dt, max_bond=16)
    initial = _nonabelian_mps_to_dense_vector(state)
    returned = _nonabelian_mps_to_dense_vector(backward)
    phase = np.vdot(initial, returned)
    returned *= np.exp(-1j * np.angle(phase))

    np.testing.assert_allclose(returned, initial, atol=1.0e-11, rtol=1.0e-11)
    np.testing.assert_allclose(
        info["pre_normalization_norm2"],
        1.0,
        atol=1.0e-11,
    )


def test_su2_tdvp_propagates_a_reduced_affine_hamiltonian():
    state = _normalized_random_state(2)
    static = build_spatial_hubbard_mpo(
        state.tensors,
        hopping_t=0.3,
        onsite_u=1.0,
    )
    interaction = build_spatial_one_body_reduced_mpo(
        state.tensors,
        np.array([[0.2, -0.1], [-0.1, -0.3]]),
    )
    coefficient = -0.17
    dynamic = sum_mpo_chains(static, scale_mpo_chain(interaction, coefficient))
    dense_h = _mpo_to_dense_operator(
        MPO(_dense_cores_from_nonabelian_mpo(dynamic), homogeneous=False)
    )
    initial = _nonabelian_mps_to_dense_vector(state)
    dt = 0.009

    propagated, _ = two_site_tdvp_step(
        state,
        dynamic,
        dt,
        max_bond=16,
        krylov_dim=20,
        krylov_tol=1.0e-14,
    )

    reference = expm(-1j * dt * dense_h) @ initial
    actual = _nonabelian_mps_to_dense_vector(propagated)
    np.testing.assert_allclose(actual, reference, atol=1.0e-12, rtol=1.0e-12)


@pytest.mark.skipif(not cpp_available(), reason="SU(2) C++ kernel unavailable")
def test_su2_tdvp_reuses_cpp_dmrg_boundary_owner_for_complex_state():
    from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment

    nsites = 4
    state = _normalized_random_state(nsites, seed=7)
    mpo = build_spatial_hubbard_mpo(
        state.sites,
        hopping_t=0.2,
        onsite_u=0.5,
    )
    owner = SU2MovingEnvironment(
        np.zeros((nsites, nsites)),
        np.zeros((nsites,) * 4),
        2,
        two_s=0,
    )
    kwargs = dict(
        max_bond=32,
        krylov_dim=16,
        krylov_tol=1.0e-13,
    )

    reference, _ = two_site_tdvp_step(state, mpo, 0.001, **kwargs)
    propagated, info = two_site_tdvp_step(
        state,
        mpo,
        0.001,
        boundary_environment=owner,
        **kwargs,
    )

    expected = _nonabelian_mps_to_dense_vector(reference)
    actual = _nonabelian_mps_to_dense_vector(propagated)
    phase = np.vdot(expected, actual)
    actual *= np.exp(-1j * np.angle(phase))
    np.testing.assert_allclose(actual, expected, atol=1.0e-12, rtol=1.0e-12)
    assert info["cpp_moving_environment"] is True
    assert owner.stats["boundary_update_calls"] > 0
    assert owner.stats["owned_boundary_bytes"] > 0
