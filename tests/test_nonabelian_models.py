import numpy as np
import pytest

from pyqed.mps.nonabelian import (
    AutoMPO,
    NonabelianTensor,
    RankCoupledMPO,
    add_spatial_density_terms,
    build_hubbard_mpo,
    build_product_state,
    add_spatial_hubbard_terms,
    build_block_sparse_bond_operator,
    build_spatial_density_mpo,
    build_spatial_hubbard_mpo,
    contract_chain_expectation,
    merge_mps_sites,
    solve_local_two_site,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
    spatial_double_occupancy,
    spatial_number,
    spatial_parity,
)
from pyqed.mps.su2 import SpatialOrbitalSite


def _spatial_chain():
    site = SpatialOrbitalSite()
    q_empty, q_single, q_double = site.qn

    A = NonabelianTensor(
        data={
            (q_empty, q_empty, q_empty): np.array([[[1.0]]]),
            (q_single, q_single, q_single): np.array([[[1.0], [0.5]]]),
            (q_double, q_double, q_double): np.array([[[0.25]]]),
        },
        qns=[list(site.qn), list(site.qn), list(site.qn)],
        dirs=[-1, 1, 1],
    )
    B = NonabelianTensor(
        data={
            (q_empty, q_empty, q_empty): np.array([[[0.5]]]),
            (q_single, q_single, q_single): np.array([[[0.75], [1.25]]]),
            (q_double, q_double, q_double): np.array([[[1.0]]]),
        },
        qns=[list(site.qn), list(site.qn), list(site.qn)],
        dirs=[-1, 1, 1],
    )
    C = NonabelianTensor(
        data={
            (q_empty, q_empty, q_empty): np.array([[[1.5]]]),
            (q_single, q_single, q_single): np.array([[[0.2], [1.8]]]),
            (q_double, q_double, q_double): np.array([[[0.7]]]),
        },
        qns=[list(site.qn), list(site.qn), list(site.qn)],
        dirs=[-1, 1, 1],
    )
    return A, B, C


def _three_site_spatial_density_dense_mpo(mu, u, v):
    n = np.diag([0.0, 1.0, 1.0, 2.0])
    doublon = np.diag([0.0, 0.0, 0.0, 1.0])
    ident = np.eye(4)
    h = -mu * n + u * doublon

    first = np.zeros((1, 3, 4, 4))
    middle = np.zeros((3, 3, 4, 4))
    last = np.zeros((3, 1, 4, 4))

    first[0, 0] = ident
    first[0, 1] = v * n
    first[0, 2] = h

    middle[0, 0] = ident
    middle[0, 1] = v * n
    middle[0, 2] = h
    middle[1, 2] = n
    middle[2, 2] = ident

    last[0, 0] = h
    last[1, 0] = n
    last[2, 0] = ident

    return [first, middle, last]


def _assert_same_tensor(a, b):
    assert a.qns == b.qns
    assert a.dirs == b.dirs
    assert a.fusion_legs == b.fusion_legs
    assert set(a.data) == set(b.data)
    for key in a.data:
        np.testing.assert_allclose(a.data[key], b.data[key])


def _dense_matrix_from_mpo_list(mpo):
    states = {0: np.array([[1.0]], dtype=complex)}
    for core in mpo:
        dense_core = core.as_dense() if hasattr(core, "as_dense") else np.asarray(core)
        new_states = {}
        for left_index, accum in states.items():
            for right_index in range(dense_core.shape[1]):
                local = dense_core[left_index, right_index]
                if not np.any(local):
                    continue
                contrib = np.kron(accum, local)
                if right_index in new_states:
                    new_states[right_index] += contrib
                else:
                    new_states[right_index] = contrib
        states = new_states
    return states[0]


def _kron_all(operators):
    dense = np.asarray(operators[0], dtype=complex)
    for operator in operators[1:]:
        dense = np.kron(dense, np.asarray(operator, dtype=complex))
    return dense


def _jw_bilinear_dense(nsites, left_site, left_operator, right_site, right_operator, parity):
    if left_site >= right_site:
        raise ValueError("_jw_bilinear_dense requires left_site < right_site.")
    ident = np.eye(parity.shape[0], dtype=complex)
    ops = [ident.copy() for _ in range(nsites)]
    ops[left_site] = np.asarray(left_operator @ parity, dtype=complex)
    for site in range(left_site + 1, right_site):
        ops[site] = np.asarray(parity, dtype=complex)
    ops[right_site] = np.asarray(right_operator, dtype=complex)
    return _kron_all(ops)


def _dense_spatial_hubbard_hamiltonian(nsites, *, hopping_t, chemical_potential, onsite_u):
    ident = np.eye(4, dtype=complex)
    parity = spatial_parity().as_dense().astype(complex)
    number = spatial_number().as_dense().astype(complex)
    doublon = spatial_double_occupancy().as_dense().astype(complex)
    c_up = spatial_annihilate_up().as_dense().astype(complex)
    cd_up = spatial_create_up().as_dense().astype(complex)
    c_down = spatial_annihilate_down().as_dense().astype(complex)
    cd_down = spatial_create_down().as_dense().astype(complex)

    h = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for site in range(nsites):
        ops = [ident.copy() for _ in range(nsites)]
        ops[site] = -chemical_potential * number + onsite_u * doublon
        h += _kron_all(ops)

    for site in range(nsites - 1):
        h += -hopping_t * _jw_bilinear_dense(nsites, site, cd_up, site + 1, c_up, parity)
        h += +hopping_t * _jw_bilinear_dense(nsites, site, c_up, site + 1, cd_up, parity)
        h += -hopping_t * _jw_bilinear_dense(nsites, site, cd_down, site + 1, c_down, parity)
        h += +hopping_t * _jw_bilinear_dense(nsites, site, c_down, site + 1, cd_down, parity)
    return h


def test_build_spatial_density_mpo_matches_dense_reference():
    A, B, C = _spatial_chain()
    mu = 1.2
    u = 3.0
    v = 0.4
    mpo = build_spatial_density_mpo(
        [A, B, C],
        chemical_potential=mu,
        onsite_u=u,
        nearest_neighbor_v=v,
    )
    dense_mpo = _three_site_spatial_density_dense_mpo(mu, u, v)
    merged = merge_mps_sites(B, C)

    op_built = build_block_sparse_bond_operator([A, B, C], mpo, 1, merged)
    op_dense = build_block_sparse_bond_operator([A, B, C], dense_mpo, 1, merged)

    optimized_built, objective_built = solve_local_two_site(
        merged, op_built, tol=1e-10, itermax=50
    )
    optimized_dense, objective_dense = solve_local_two_site(
        merged, op_dense, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_built, optimized_dense)
    assert objective_built["energy"] == pytest.approx(objective_dense["energy"])


def test_add_spatial_density_terms_matches_direct_builder():
    A, B, C = _spatial_chain()
    mu = 0.7
    u = 2.5
    v = 0.2

    auto = AutoMPO.from_sites([A, B, C])
    add_spatial_density_terms(
        auto,
        chemical_potential=mu,
        onsite_u=u,
        nearest_neighbor_v=v,
    )
    mpo_manual = auto.build()
    mpo_direct = build_spatial_density_mpo(
        [A, B, C],
        chemical_potential=mu,
        onsite_u=u,
        nearest_neighbor_v=v,
    )
    merged = merge_mps_sites(B, C)

    op_manual = build_block_sparse_bond_operator([A, B, C], mpo_manual, 1, merged)
    op_direct = build_block_sparse_bond_operator([A, B, C], mpo_direct, 1, merged)

    optimized_manual, objective_manual = solve_local_two_site(
        merged, op_manual, tol=1e-10, itermax=50
    )
    optimized_direct, objective_direct = solve_local_two_site(
        merged, op_direct, tol=1e-10, itermax=50
    )

    _assert_same_tensor(optimized_manual, optimized_direct)
    assert objective_manual["energy"] == pytest.approx(objective_direct["energy"])


def test_dense_mpo_conversion_preserves_spatial_physical_ordering():
    sites = build_product_state(["double", "empty"])
    identity = np.eye(4)
    local = np.diag([0.5, 1.5, 2.5, 4.0])
    first = np.zeros((1, 2, 4, 4))
    last = np.zeros((2, 1, 4, 4))
    first[0, 0] = identity
    first[0, 1] = local
    last[0, 0] = local
    last[1, 0] = identity

    value = contract_chain_expectation(sites, [first, last])

    assert value == pytest.approx(4.5)


def test_autompo_add_fermionic_bilinear_matches_dense_jordan_wigner_string():
    A, B, C = _spatial_chain()
    parity = spatial_parity(dtype=float)
    builder = AutoMPO.from_sites([A, B, C])
    builder.add_fermionic_bilinear(
        0,
        spatial_create_up(dtype=float),
        2,
        spatial_annihilate_up(dtype=float),
        parity_operator=parity,
    )
    built_dense = _dense_matrix_from_mpo_list(builder.build())
    ref_dense = _jw_bilinear_dense(
        3,
        0,
        spatial_create_up().as_dense(),
        2,
        spatial_annihilate_up().as_dense(),
        spatial_parity().as_dense(),
    )
    np.testing.assert_allclose(built_dense, ref_dense)


def test_build_spatial_hubbard_mpo_matches_dense_reference():
    hopping_t = 0.9
    chemical_potential = 0.4
    onsite_u = 2.2

    built = build_spatial_hubbard_mpo(
        3,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    built_dense = _dense_matrix_from_mpo_list(built)
    ref_dense = _dense_spatial_hubbard_hamiltonian(
        3,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    np.testing.assert_allclose(built_dense, ref_dense)
    np.testing.assert_allclose(built_dense, built_dense.conj().T)
    assert any(isinstance(core, RankCoupledMPO) for core in built)


def test_build_hubbard_mpo_aliases_build_spatial_hubbard_mpo():
    built = build_hubbard_mpo(
        3,
        hopping_t=0.9,
        chemical_potential=0.4,
        onsite_u=2.2,
    )
    direct = build_spatial_hubbard_mpo(
        3,
        hopping_t=0.9,
        chemical_potential=0.4,
        onsite_u=2.2,
    )
    assert len(built) == len(direct)
    for a, b in zip(built, direct):
        if hasattr(a, "as_dense"):
            np.testing.assert_allclose(a.as_dense(), b.as_dense())
        else:
            np.testing.assert_allclose(np.asarray(a), np.asarray(b))


def test_add_spatial_hubbard_terms_matches_direct_builder():
    A, B, C = _spatial_chain()
    hopping_t = 1.1
    chemical_potential = 0.2
    onsite_u = 1.7

    auto = AutoMPO.from_sites([A, B, C])
    add_spatial_hubbard_terms(
        auto,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    built_dense = _dense_matrix_from_mpo_list(auto.build())
    ref_dense = _dense_spatial_hubbard_hamiltonian(
        3,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    np.testing.assert_allclose(built_dense, ref_dense)
    np.testing.assert_allclose(built_dense, built_dense.conj().T)
