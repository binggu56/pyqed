import numpy as np
from scipy.linalg import expm

from pyqed.letta import (
    LETTA,
    LETTAEvolution,
    LETTATDVPEngine,
    nearest_neighbor_hamiltonian,
    one_site_tdvp_step,
    site_reduced_density_matrix,
    system_reduced_density_matrix,
    two_site_tdvp_step,
    window2_hamiltonian_from_mpo,
    window2_product_state,
)
from pyqed.letta import tdvp as letta_tdvp
from pyqed.mps.mpo import nearest_neighbor_mpo
from pyqed.models.impurity.spin_boson import (
    log_discretized_spin_boson_wilson_chain,
    spin_boson_bond_hamiltonians,
)
from pyqed.narg.spin_boson import (
    local_boson_operators,
    spin_boson_wilson_hamiltonian,
)


def _random_hermitian_bonds(dims, seed=7):
    rng = np.random.default_rng(seed)
    terms = []
    for left, right in zip(dims[:-1], dims[1:]):
        size = left * right
        matrix = rng.normal(size=(size, size)) + 1.0j * rng.normal(size=(size, size))
        terms.append((matrix + matrix.conj().T) / (2.0 * size))
    return terms


def _dense_nearest_neighbor(terms, dims):
    total = np.zeros((int(np.prod(dims)),) * 2, dtype=complex)
    for bond, term in enumerate(terms):
        left = int(np.prod(dims[:bond], dtype=int))
        right = int(np.prod(dims[bond + 2 :], dtype=int))
        total += np.kron(np.eye(left), np.kron(term, np.eye(right)))
    return total


def _phase_aligned_error(actual, reference):
    overlap = np.vdot(reference, actual)
    phase = 1.0 if abs(overlap) == 0.0 else overlap / abs(overlap)
    return np.max(np.abs(actual - phase * reference))


def _dense_site_rdm(vector, dims, site):
    tensor = np.asarray(vector).reshape(dims)
    matrix = np.moveaxis(tensor, site, 0).reshape(dims[site], -1)
    rho = matrix @ matrix.conj().T
    return rho / np.trace(rho)


def test_site_reduced_density_matrix_matches_dense_at_every_position():
    rng = np.random.default_rng(5)
    dims = (2, 3, 2, 2)
    tensors = [
        rng.normal(size=(1, 2, 3, 2)) + 1.0j * rng.normal(size=(1, 2, 3, 2)),
        rng.normal(size=(2, 3, 2, 3)) + 1.0j * rng.normal(size=(2, 3, 2, 3)),
        rng.normal(size=(3, 2, 2, 2)) + 1.0j * rng.normal(size=(3, 2, 2, 2)),
        rng.normal(size=(2, 2)) + 1.0j * rng.normal(size=(2, 2)),
    ]
    state = LETTA(None, dims, bond_dim=3, tensors=tensors)
    vector = state.state_vector()

    for site in range(len(dims)):
        rho, info = site_reduced_density_matrix(state, site, return_info=True)
        np.testing.assert_allclose(
            rho, _dense_site_rdm(vector, dims, site), rtol=3.0e-13, atol=3.0e-13
        )
        assert info["site"] == site
        assert info["trace_error"] < 3.0e-13
        assert info["hermiticity_error"] < 3.0e-13
    np.testing.assert_allclose(
        site_reduced_density_matrix(state, 0),
        system_reduced_density_matrix(state),
        rtol=3.0e-13,
        atol=3.0e-13,
    )


def test_window2_product_state_and_copy_preserve_raw_scale():
    factors = ([1.0, 2.0j], [2.0, -1.0], [1.0j, 3.0])
    state = window2_product_state(factors, max_bond=3)
    expected = np.kron(np.kron(factors[0], factors[1]), factors[2])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(state.state_vector(), expected, rtol=2.0e-14, atol=2.0e-14)

    state.tensors[0] *= 0.37
    copied = state.copy()
    np.testing.assert_allclose(copied.state_vector(), state.state_vector(), rtol=0.0, atol=0.0)

    rho = system_reduced_density_matrix(state)
    dense = state.state_vector().reshape(2, -1)
    expected_rho = dense @ dense.conj().T
    expected_rho /= np.trace(expected_rho)
    np.testing.assert_allclose(rho, expected_rho, rtol=2.0e-14, atol=2.0e-14)


def test_nearest_neighbor_mpo_matches_dense_bond_sum():
    dims = (2, 3, 2)
    terms = _random_hermitian_bonds(dims)
    mpo = nearest_neighbor_mpo(terms, dims)
    np.testing.assert_allclose(
        mpo.to_dense(), _dense_nearest_neighbor(terms, dims), rtol=2.0e-13, atol=2.0e-13
    )


def test_window2_mpo_lift_matches_native_nearest_neighbor_operator():
    dims = (2, 2, 2)
    terms = _random_hermitian_bonds(dims, seed=31)
    native = nearest_neighbor_hamiltonian(terms, dims)
    lifted = window2_hamiltonian_from_mpo(nearest_neighbor_mpo(terms, dims))
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0]), max_bond=2
    )
    native_state = two_site_tdvp_step(
        initial, native, 0.013, max_bond=2, krylov_dim=64, krylov_tol=1.0e-14
    )
    lifted_state = two_site_tdvp_step(
        initial, lifted, 0.013, max_bond=2, krylov_dim=64, krylov_tol=1.0e-14
    )
    assert _phase_aligned_error(
        lifted_state.state_vector(), native_state.state_vector()
    ) < 2.0e-12


def test_spin_boson_bond_terms_match_direct_finite_hamiltonian():
    chain = log_discretized_spin_boson_wilson_chain(
        2, alpha=0.04, Lambda=1.7, s=0.6, delta=0.1
    )
    operators = local_boson_operators(3, basis="fock")
    bonds, dims = spin_boson_bond_hamiltonians(chain, *operators)
    direct = spin_boson_wilson_hamiltonian(chain, 3, basis="fock")
    np.testing.assert_allclose(
        _dense_nearest_neighbor(bonds, dims), direct, rtol=2.0e-13, atol=2.0e-13
    )


def test_two_site_letta_tdvp_is_exact_for_two_physical_sites():
    dims = (2, 3)
    term = _random_hermitian_bonds(dims)[0]
    state = window2_product_state(([1.0, 0.0], [0.0, 1.0, 0.0]), max_bond=3)
    operator = nearest_neighbor_hamiltonian([term], dims)
    actual, info = two_site_tdvp_step(
        state,
        operator,
        0.07,
        max_bond=3,
        krylov_dim=32,
        krylov_tol=1.0e-14,
        return_info=True,
    )
    expected = expm(-0.07j * term) @ state.state_vector()
    np.testing.assert_allclose(actual.state_vector(), expected, rtol=3.0e-13, atol=3.0e-13)
    assert info["integrator"] == "tdvp2"
    assert info["truncation_error"] < 1.0e-24


def test_blas_two_site_kernel_matches_direct_einsum():
    rng = np.random.default_rng(29)
    physical, state_rank, operator_rank = 3, 2, 3
    shapes = (
        (physical, physical, state_rank, operator_rank, state_rank),
        (operator_rank, physical, physical, physical, physical, operator_rank),
        (operator_rank, physical, physical, physical, physical, operator_rank),
        (state_rank, physical, physical, physical, state_rank),
        (physical, physical, state_rank, operator_rank, state_rank),
    )
    operands = [
        rng.normal(size=shape) + 1.0j * rng.normal(size=shape)
        for shape in shapes
    ]
    actual = letta_tdvp._apply_two_site(
        operands[0], operands[4], operands[1], operands[2], operands[3]
    )
    expected = np.einsum(
        "pramc,mpqrsn,nqusxo,crsxf,uxbof->apqub", *operands, optimize=True
    )
    np.testing.assert_allclose(actual, expected, rtol=3.0e-13, atol=3.0e-13)


def test_two_site_letta_tdvp_is_time_reversible_without_truncation():
    dims = (2, 2, 2)
    terms = _random_hermitian_bonds(dims, seed=11)
    operator = nearest_neighbor_hamiltonian(terms, dims)
    initial = window2_product_state(([1.0, 0.0], [0.0, 1.0], [1.0, 0.0]), max_bond=2)
    forward = two_site_tdvp_step(
        initial,
        operator,
        0.025,
        max_bond=2,
        krylov_dim=64,
        krylov_tol=1.0e-14,
    )
    backward = two_site_tdvp_step(
        forward,
        operator,
        -0.025,
        max_bond=2,
        krylov_dim=64,
        krylov_tol=1.0e-14,
        canonicalize=False,
    )
    assert _phase_aligned_error(backward.state_vector(), initial.state_vector()) < 3.0e-13


def test_one_site_step_keeps_virtual_ranks_fixed():
    dims = (2, 2, 2)
    terms = _random_hermitian_bonds(dims, seed=13)
    operator = nearest_neighbor_hamiltonian(terms, dims)
    initial = window2_product_state(([1.0, 0.0], [0.0, 1.0], [1.0, 0.0]), max_bond=2)
    grown = two_site_tdvp_step(
        initial, operator, 0.01, max_bond=2, krylov_dim=64, krylov_tol=1.0e-14
    )
    before = tuple(tensor.shape[-1] for tensor in grown.tensors[:-1])
    actual, info = one_site_tdvp_step(
        grown, operator, 0.01, krylov_dim=64, krylov_tol=1.0e-14, return_info=True
    )
    after = tuple(tensor.shape[-1] for tensor in actual.tensors[:-1])
    assert after == before
    assert info["integrator"] == "tdvp1"


def test_one_site_step_is_time_reversible_with_inactive_physical_sectors():
    dims = (2, 2, 2, 2)
    operator = nearest_neighbor_hamiltonian(
        _random_hermitian_bonds(dims, seed=23), dims
    )
    initial = window2_product_state(
        ([1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]), max_bond=1
    )
    forward = one_site_tdvp_step(
        initial, operator, 0.01, krylov_dim=64, krylov_tol=1.0e-14
    )
    backward = one_site_tdvp_step(
        forward,
        operator,
        -0.01,
        krylov_dim=64,
        krylov_tol=1.0e-14,
        canonicalize=False,
    )
    assert _phase_aligned_error(backward.state_vector(), initial.state_vector()) < 3.0e-12


def test_engine_and_checkpoint_restart_match_uninterrupted_run(tmp_path):
    dims = (2, 2, 2)
    operator = nearest_neighbor_hamiltonian(_random_hermitian_bonds(dims, seed=17), dims)
    initial = window2_product_state(([1.0, 0.0], [0.0, 1.0], [1.0, 0.0]), max_bond=2)

    direct = LETTAEvolution(
        operator, max_bond=2, saturation_steps=1, krylov_dim=64, krylov_tol=1.0e-14
    )
    direct_state = direct.run(initial, 0.01, 3)

    first = LETTAEvolution(
        operator, max_bond=2, saturation_steps=1, krylov_dim=64, krylov_tol=1.0e-14
    )
    first.run(initial, 0.01, 1)
    path = first.save_checkpoint(tmp_path / "letta.pkl")
    restarted = LETTAEvolution.load_checkpoint(path, operator)
    restarted_state = restarted.run(restarted.state, 0.01, 2)

    np.testing.assert_allclose(
        restarted_state.state_vector(), direct_state.state_vector(), rtol=3.0e-13, atol=3.0e-13
    )
    assert restarted.mode == direct.mode == "tdvp1"
    assert restarted.step_index == direct.step_index == 3


def test_stateful_engine_only_canonicalizes_first_step():
    dims = (2, 2)
    operator = nearest_neighbor_hamiltonian(_random_hermitian_bonds(dims), dims)
    state = window2_product_state(([1.0, 0.0], [0.0, 1.0]), max_bond=2)
    engine = LETTATDVPEngine(
        operator, max_bond=2, krylov_dim=16, krylov_tol=1.0e-14
    )
    state, _ = engine.step(state, 0.01)
    state, _ = engine.step(state, 0.01)
    assert engine.prepared
    assert len(engine.history) == 2
