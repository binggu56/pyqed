import numpy as np
import pytest
from itertools import permutations

from pyqed.mps.nonabelian import (
    AutoMPO,
    IrrepTensor,
    RankCoupledChannelTerm,
    RankCoupledMPO,
    SiteOperator,
    clebsch_gordan,
    coupled_reduced_tensor_product,
    add_spatial_one_body_terms,
    add_spatial_spinfree_eri_terms,
    add_spatial_density_terms,
    build_spatial_one_body_reduced_mpo,
    build_spatial_spinfree_eri_mpo,
    build_hubbard_mpo,
    build_product_state,
    add_spatial_hubbard_terms,
    build_block_sparse_bond_operator,
    build_spatial_density_mpo,
    build_spatial_hubbard_mpo,
    contract_chain_expectation,
    physical_leg_from_spatial_orbital,
    reduced_spatial_fermion_annihilation,
    merge_mps_sites,
    solve_local_two_site,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
    spatial_double_occupancy,
    spatial_number,
    spatial_pair_annihilation,
    spatial_pair_creation,
    spatial_parity,
    time_reversed_reduced_operator,
    FullyReducedSpatialOrbitalSite,
    spatial_target_sector,
    as_rank_coupled_mpo,
)
from pyqed.mps.su2 import SpatialOrbitalSite, SpinChargeSector, SU2Irrep
from pyqed.mps.nonabelian.coupling import ordered_two_m_values
from pyqed.mps.nonabelian.states import _fuse_spatial_sectors
from pyqed.mps.nonabelian import models as nonabelian_models


def _spatial_chain():
    site = SpatialOrbitalSite()
    q_empty, q_single, q_double = site.qn

    A = IrrepTensor(
        data={
            (q_empty, q_empty, q_empty): np.array([[[1.0]]]),
            (q_single, q_single, q_single): np.array([[[1.0], [0.5]]]),
            (q_double, q_double, q_double): np.array([[[0.25]]]),
        },
        qns=[list(site.qn), list(site.qn), list(site.qn)],
        dirs=[-1, 1, 1],
    )
    B = IrrepTensor(
        data={
            (q_empty, q_empty, q_empty): np.array([[[0.5]]]),
            (q_single, q_single, q_single): np.array([[[0.75], [1.25]]]),
            (q_double, q_double, q_double): np.array([[[1.0]]]),
        },
        qns=[list(site.qn), list(site.qn), list(site.qn)],
        dirs=[-1, 1, 1],
    )
    C = IrrepTensor(
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


def _dense_spatial_one_body_hamiltonian(h1e):
    h1e = np.asarray(h1e)
    nsites = h1e.shape[0]
    ident = np.eye(4, dtype=complex)
    parity = spatial_parity().as_dense().astype(complex)
    number = spatial_number().as_dense().astype(complex)
    c_up = spatial_annihilate_up().as_dense().astype(complex)
    cd_up = spatial_create_up().as_dense().astype(complex)
    c_down = spatial_annihilate_down().as_dense().astype(complex)
    cd_down = spatial_create_down().as_dense().astype(complex)

    h = np.zeros((4**nsites, 4**nsites), dtype=complex)
    for site, coeff in enumerate(np.diag(h1e)):
        ops = [ident.copy() for _ in range(nsites)]
        ops[site] = coeff * number
        h += _kron_all(ops)

    for left_site in range(nsites):
        for right_site in range(left_site + 1, nsites):
            h += h1e[left_site, right_site] * _jw_bilinear_dense(
                nsites, left_site, cd_up, right_site, c_up, parity
            )
            h += h1e[left_site, right_site] * _jw_bilinear_dense(
                nsites, left_site, cd_down, right_site, c_down, parity
            )
            h += -h1e[right_site, left_site] * _jw_bilinear_dense(
                nsites, left_site, c_up, right_site, cd_up, parity
            )
            h += -h1e[right_site, left_site] * _jw_bilinear_dense(
                nsites, left_site, c_down, right_site, cd_down, parity
            )
    return h


def _spatial_jw_product_dense(nsites, operators, sites):
    ident = np.eye(4, dtype=complex)
    parity = spatial_parity().as_dense().astype(complex)
    grouped = {}
    for operator, site in zip(operators, sites):
        site = int(site)
        for parity_site in range(site):
            grouped.setdefault(parity_site, []).append(parity)
        grouped.setdefault(site, []).append(np.asarray(operator, dtype=complex))
    local = [ident.copy() for _ in range(nsites)]
    for site, pieces in grouped.items():
        op = ident.copy()
        for piece in pieces:
            op = op @ piece
        local[site] = op
    return _kron_all(local)


def _dense_spatial_spinfree_eri_hamiltonian(eri_spatial):
    eri_spatial = np.asarray(eri_spatial)
    nsites = eri_spatial.shape[0]
    c_up = spatial_annihilate_up().as_dense().astype(complex)
    cd_up = spatial_create_up().as_dense().astype(complex)
    c_down = spatial_annihilate_down().as_dense().astype(complex)
    cd_down = spatial_create_down().as_dense().astype(complex)
    spin_terms = ((cd_up, c_up), (cd_down, c_down))
    h = np.zeros((4**nsites, 4**nsites), dtype=complex)
    values = 0.5 * eri_spatial
    for p, q, r, s in np.argwhere(np.abs(values) > 1.0e-14):
        val = values[p, q, r, s]
        for left_create, left_destroy in spin_terms:
            for right_create, right_destroy in spin_terms:
                h += val * _spatial_jw_product_dense(
                    nsites,
                    (left_create, left_destroy, right_create, right_destroy),
                    (p, q, r, s),
                )
        if q == r:
            for create, destroy in spin_terms:
                h -= val * _spatial_jw_product_dense(
                    nsites,
                    (create, destroy),
                    (p, s),
                )
    return h


def _dense_vector_from_reduced_spatial_mps(sites):
    site = SpatialOrbitalSite()
    state_sector = [None] * site.d
    state_two_m = [None] * site.d
    for sector_index, sector in enumerate(site.qn):
        for local_index, state_index in enumerate(site.state_index[sector_index]):
            state_sector[state_index] = sector
            state_two_m[state_index] = ordered_two_m_values(sector.irrep)[local_index]

    vector = np.zeros(site.d ** len(sites), dtype=complex)
    for basis_index in range(vector.size):
        encoded = basis_index
        physical_indices = [0] * len(sites)
        for site_index in range(len(sites) - 1, -1, -1):
            physical_indices[site_index] = encoded % site.d
            encoded //= site.d
        boundary = {(sites[0].qns[0][0], 0, 0): 1.0 + 0.0j}
        for tensor, physical_index in zip(sites, physical_indices):
            q_phys = state_sector[physical_index]
            two_m_phys = state_two_m[physical_index]
            updated = {}
            for (q_left, left_slot, two_m_left), amplitude in boundary.items():
                for (block_left, block_phys, block_right), block in tensor.data.items():
                    if block_left != q_left or block_phys != q_phys:
                        continue
                    arr = np.asarray(block)
                    for right_slot in range(arr.shape[2]):
                        for two_m_right in ordered_two_m_values(block_right.irrep):
                            coeff = clebsch_gordan(
                                block_left.irrep,
                                block_phys.irrep,
                                block_right.irrep,
                                two_m_left,
                                two_m_phys,
                                two_m_right,
                            )
                            if coeff:
                                key = (block_right, right_slot, two_m_right)
                                updated[key] = updated.get(key, 0.0) + (
                                    amplitude * arr[left_slot, 0, right_slot] * coeff
                                )
            boundary = updated
        target = sites[-1].qns[2][0]
        vector[basis_index] = boundary.get((target, 0, 0), 0.0)
    return vector


def _reduced_spatial_path_mps(labels, bonds):
    site = FullyReducedSpatialOrbitalSite()
    tensors = []
    for site_index, label in enumerate(labels):
        q_left = bonds[site_index]
        q_phys = site.qn[label]
        q_right = bonds[site_index + 1]
        tensors.append(
            IrrepTensor(
                data={(q_left, q_phys, q_right): np.ones((1, 1, 1))},
                qns=[[q_left], list(site.qn), [q_right]],
                dirs=[-1, 1, 1],
                metadata={"physical_basis": "fully_reduced_su2"},
            )
        )
    return tensors


def _reduced_spatial_path_basis(path_specs):
    basis_states = [
        _reduced_spatial_path_mps(labels, bonds)
        for labels, bonds in path_specs
    ]
    dense_vectors = [
        _dense_vector_from_reduced_spatial_mps(state)
        for state in basis_states
    ]
    return basis_states, dense_vectors


def _reduced_spatial_path_specs(nsites, target):
    site = FullyReducedSpatialOrbitalSite()
    vacuum = spatial_target_sector(0, 0)
    path_specs = []

    def walk(site_index, left, labels, bonds):
        if site_index == nsites:
            if left == target:
                path_specs.append((tuple(labels), tuple(bonds)))
            return
        for label, q_phys in enumerate(site.qn):
            for right in _fuse_spatial_sectors(left, q_phys):
                if right.charge <= target.charge:
                    walk(site_index + 1, right, labels + [label], bonds + [right])

    walk(0, vacuum, [], [vacuum])
    return tuple(path_specs)


def _contract_chain_transition(bra_sites, mpo_factors, ket_sites):
    from pyqed.mps.nonabelian.environment import (
        _contract_from_left_blocks,
        _contract_from_left_blocks_rank_coupled,
        _environment_map_expectation,
        _initial_left_env_blocks,
        _initial_left_env_blocks_rank_coupled,
        _is_rank_coupled_chain,
        _normalize_block_sparse_mpo_factors,
        _rank_coupled_channel_expectation,
        _tensor_dense_layout,
    )

    site_layouts = [_tensor_dense_layout(site) for site in ket_sites]
    sparse_mpo_factors = _normalize_block_sparse_mpo_factors(
        mpo_factors,
        site_layouts=site_layouts,
    )
    rank_coupled = _is_rank_coupled_chain(sparse_mpo_factors)
    if rank_coupled:
        env = _initial_left_env_blocks_rank_coupled(
            site_layouts[0],
            sparse_mpo_factors[0],
        )
        for idx in range(len(ket_sites)):
            env = _contract_from_left_blocks_rank_coupled(
                sparse_mpo_factors[idx],
                bra_sites[idx],
                env,
                ket_sites[idx],
            )
    else:
        phys_slice_maps = [layout["sector_slices"][1] for layout in site_layouts]
        env = _initial_left_env_blocks(site_layouts[0], sparse_mpo_factors[0])
        for idx in range(len(ket_sites)):
            env = _contract_from_left_blocks(
                sparse_mpo_factors[idx],
                bra_sites[idx],
                env,
                ket_sites[idx],
                phys_slice_maps[idx],
            )
    if (
        rank_coupled
        and getattr(
            sparse_mpo_factors[0],
            "normal_complementary_plan",
            None,
        )
        is not None
    ):
        return _rank_coupled_channel_expectation(env, 0)
    return _environment_map_expectation(env, rank_coupled=rank_coupled)


def test_fully_reduced_normal_complementary_matrix_matches_dense_reference():
    try:
        from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    except ImportError:
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    from pyqed.qchem.dmrg.backends.reduced import (
        build_su2_normal_complementary_mpo,
        refresh_su2_normal_complementary_mpo,
    )
    from pyqed.mps.nonabelian.environment import (
        LeftBlock,
        RightBlock,
        _initial_left_env_blocks_rank_coupled,
        _initial_right_env_blocks_rank_coupled,
        _rank_coupled_channel_expectation,
        _rank_coupled_cut_expectation,
        _tensor_dense_layout,
    )

    nsites = 4
    rng = np.random.default_rng(20260726)
    one_body = rng.normal(scale=0.03, size=(nsites, nsites))
    one_body = 0.5 * (one_body + one_body.T)
    cholesky = rng.normal(scale=0.02, size=(nsites, nsites, 5))
    cholesky = 0.5 * (cholesky + cholesky.swapaxes(0, 1))
    eri = np.einsum("pqL,rsL->pqrs", cholesky, cholesky)

    owner = SU2MovingEnvironment(
        np.zeros_like(one_body),
        eri,
        4,
        two_s=0,
        cutoff=1.0e-12,
        include_half=True,
    )
    revision_before = owner.system_stats["revision"]
    plans_before = tuple(
        owner.normal_complementary_plan(site) for site in range(nsites)
    )
    owner.update_h1(one_body)
    plans_after = tuple(
        owner.normal_complementary_plan(site) for site in range(nsites)
    )
    assert owner.system_stats["revision"] != revision_before
    topology_keys = (
        "source",
        "target",
        "operator",
        "first_index",
        "second_index",
        "family_mask",
    )
    for before, after in zip(plans_before, plans_after):
        for key in topology_keys:
            np.testing.assert_array_equal(after[key], before[key])
    assert any(
        not np.array_equal(after["coefficient"], before["coefficient"])
        for before, after in zip(plans_before, plans_after)
    )
    mpo = build_su2_normal_complementary_mpo(
        owner,
        fully_reduced=True,
    )
    for factor in mpo:
        object.__setattr__(
            factor,
            "normal_complementary_right_dual",
            True,
        )
    basis_states, dense_vectors = _reduced_spatial_path_basis(
        _reduced_spatial_path_specs(
            nsites,
            spatial_target_sector(4, 0),
        )
    )
    dense_hamiltonian = (
        _dense_spatial_one_body_hamiltonian(one_body)
        + _dense_spatial_spinfree_eri_hamiltonian(eri)
    )
    reference = np.asarray(
        [
            [
                _contract_chain_transition(bra, mpo, ket)
                for ket in basis_states
            ]
            for bra in basis_states
        ]
    )
    revision = 0

    def direct_transition(bra, ket):
        nonlocal revision
        revision += 1
        owner.clear_boundaries()
        env = LeftBlock(
            _initial_left_env_blocks_rank_coupled(
                _tensor_dense_layout(ket[0]),
                mpo[0],
            ),
            rank_coupled=True,
        )
        for site, (core, bra_site, ket_site) in enumerate(
            zip(mpo, bra, ket)
        ):
            env = env.advance(
                core,
                bra_site,
                ket_site,
                moving_environment=owner,
                parent_bond=site,
                child_bond=site + 1,
                numeric_revision=revision * nsites + site + 1,
            )
        return _rank_coupled_channel_expectation(env, 0)

    direct = np.asarray(
        [
            [direct_transition(bra, ket) for ket in basis_states]
            for bra in basis_states
        ]
    )

    def direct_right_transition(bra, ket):
        nonlocal revision
        revision += 1
        owner.clear_boundaries()
        env = RightBlock(
            _initial_right_env_blocks_rank_coupled(
                _tensor_dense_layout(ket[-1]),
                mpo[-1],
            ),
            rank_coupled=True,
        )
        for site in range(nsites - 1, -1, -1):
            env = env.advance(
                mpo[site],
                bra[site],
                ket[site],
                moving_environment=owner,
                parent_bond=site + 1,
                child_bond=site,
                numeric_revision=revision * nsites + nsites - site,
            )
        return _rank_coupled_channel_expectation(env, 1)

    direct_right = np.asarray(
        [
            [direct_right_transition(bra, ket) for ket in basis_states]
            for bra in basis_states
        ]
    )

    def interior_cut_transition(bra, ket, *, moving_environment):
        nonlocal revision
        revision += 1
        if moving_environment is not None:
            moving_environment.clear_boundaries()
        cut = 2
        left = LeftBlock(
            _initial_left_env_blocks_rank_coupled(
                _tensor_dense_layout(ket[0]),
                mpo[0],
            ),
            rank_coupled=True,
        )
        for site in range(cut):
            left = left.advance(
                mpo[site],
                bra[site],
                ket[site],
                moving_environment=moving_environment,
                parent_bond=site,
                child_bond=site + 1,
                numeric_revision=revision * nsites + site + 1,
            )
        right = RightBlock(
            _initial_right_env_blocks_rank_coupled(
                _tensor_dense_layout(ket[-1]),
                mpo[-1],
            ),
            rank_coupled=True,
        )
        for site in range(nsites - 1, cut - 1, -1):
            right = right.advance(
                mpo[site],
                bra[site],
                ket[site],
                moving_environment=moving_environment,
                parent_bond=site + 1,
                child_bond=site,
                numeric_revision=revision * nsites + nsites - site,
            )
        return _rank_coupled_cut_expectation(
            left,
            right,
            mpo[cut - 1].right_channel_irreps,
        )

    cut_cpp = np.asarray(
        [
            [
                interior_cut_transition(
                    bra,
                    ket,
                    moving_environment=owner,
                )
                for ket in basis_states
            ]
            for bra in basis_states
        ]
    )
    cut_python = np.asarray(
        [
            [
                interior_cut_transition(
                    bra,
                    ket,
                    moving_environment=None,
                )
                for ket in basis_states
            ]
            for bra in basis_states
        ]
    )

    expected = np.asarray(
        [
            [
                np.vdot(bra, dense_hamiltonian @ ket)
                for ket in dense_vectors
            ]
            for bra in dense_vectors
        ]
    )

    np.testing.assert_allclose(
        reference,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        direct,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        direct_right,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cut_cpp,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        cut_python,
        expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )

    factor_ids = tuple(id(factor) for factor in mpo)
    route_ids = tuple(
        id(term.visible_virtual_block.values)
        for factor in mpo
        for term in factor.reduced_terms
    )
    one_body_update = one_body + np.diag(np.linspace(-0.01, 0.01, nsites))
    assert owner.update_h1(one_body_update) is True
    refresh_su2_normal_complementary_mpo(owner, mpo)
    assert tuple(id(factor) for factor in mpo) == factor_ids
    assert tuple(
        id(term.visible_virtual_block.values)
        for factor in mpo
        for term in factor.reduced_terms
    ) == route_ids
    refreshed = np.asarray(
        [
            [
                _contract_chain_transition(bra, mpo, ket)
                for ket in basis_states
            ]
            for bra in basis_states
        ]
    )
    refreshed_dense_hamiltonian = (
        _dense_spatial_one_body_hamiltonian(one_body_update)
        + _dense_spatial_spinfree_eri_hamiltonian(eri)
    )
    refreshed_expected = np.asarray(
        [
            [
                np.vdot(bra, refreshed_dense_hamiltonian @ ket)
                for ket in dense_vectors
            ]
            for bra in dense_vectors
        ]
    )
    np.testing.assert_allclose(
        refreshed,
        refreshed_expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert owner.update_h1(one_body_update) is False

    eri_update = 1.1 * eri
    assert owner.update_integrals(one_body_update, eri_update, 0.17) is True
    refresh_su2_normal_complementary_mpo(owner, mpo)
    refreshed_all = np.asarray(
        [
            [
                _contract_chain_transition(bra, mpo, ket)
                for ket in basis_states
            ]
            for bra in basis_states
        ]
    )
    refreshed_all_dense = (
        _dense_spatial_one_body_hamiltonian(one_body_update)
        + _dense_spatial_spinfree_eri_hamiltonian(eri_update)
        + 0.17 * np.eye(4 ** nsites)
    )
    refreshed_all_expected = np.asarray(
        [
            [
                np.vdot(bra, refreshed_all_dense @ ket)
                for ket in dense_vectors
            ]
            for bra in dense_vectors
        ]
    )
    np.testing.assert_allclose(
        refreshed_all,
        refreshed_all_expected,
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    assert owner.update_integrals(one_body_update, eri_update, 0.17) is False


def test_normal_complementary_pair_adjoint_survives_rank_one_transport():
    try:
        from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    except ImportError:
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    from pyqed.mps.nonabelian.environment import (
        LeftBlock,
        _initial_left_env_blocks_rank_coupled,
        _rank_coupled_channel_expectation,
        _tensor_dense_layout,
    )
    from pyqed.qchem.dmrg.backends.reduced import (
        build_su2_normal_complementary_mpo,
    )

    nsites = 6
    eri = np.zeros((nsites, nsites, nsites, nsites))
    p, q, r, s = 4, 1, 5, 2
    for index in {
        (p, q, r, s),
        (q, p, r, s),
        (p, q, s, r),
        (q, p, s, r),
        (r, s, p, q),
        (s, r, p, q),
        (r, s, q, p),
        (s, r, q, p),
    }:
        eri[index] = 1.0

    owner = SU2MovingEnvironment(
        np.zeros((nsites, nsites)),
        eri,
        6,
        two_s=0,
        cutoff=1.0e-12,
        include_half=True,
    )
    mpo = build_su2_normal_complementary_mpo(owner, fully_reduced=True)
    for factor in mpo:
        object.__setattr__(
            factor,
            "normal_complementary_right_dual",
            True,
        )
    specs = _reduced_spatial_path_specs(
        nsites,
        spatial_target_sector(6, 0),
    )
    by_labels = {labels: bonds for labels, bonds in specs}
    bra = _reduced_spatial_path_mps(
        (2, 1, 1, 0, 1, 1),
        by_labels[(2, 1, 1, 0, 1, 1)],
    )
    ket = _reduced_spatial_path_mps(
        (2, 2, 2, 0, 0, 0),
        by_labels[(2, 2, 2, 0, 0, 0)],
    )
    revision = 0

    def transition(left_state, right_state):
        nonlocal revision
        revision += 1
        owner.clear_boundaries()
        env = LeftBlock(
            _initial_left_env_blocks_rank_coupled(
                _tensor_dense_layout(right_state[0]),
                mpo[0],
            ),
            rank_coupled=True,
        )
        for site, (core, bra_site, ket_site) in enumerate(
            zip(mpo, left_state, right_state)
        ):
            env = env.advance(
                core,
                bra_site,
                ket_site,
                moving_environment=owner,
                parent_bond=site,
                child_bond=site + 1,
                numeric_revision=revision * nsites + site + 1,
            )
        return _rank_coupled_channel_expectation(env, 0)

    expected = -np.sqrt(3.0)
    assert transition(bra, ket) == pytest.approx(expected, abs=1.0e-12)
    assert transition(ket, bra) == pytest.approx(expected, abs=1.0e-12)


def test_cpp_contextual_right_core_matches_reduced_reference_orientation():
    try:
        from pyqed.mps.nonabelian._su2_kernel import SU2MovingEnvironment
    except ImportError:
        pytest.skip("optional SU(2) C++ kernel is unavailable")
    from pyqed.mps.nonabelian.environment import (
        _component_basis_norm,
        _right_reduced_rank_coupled_block,
    )
    from pyqed.qchem.dmrg.backends.reduced import (
        build_su2_normal_complementary_mpo,
    )

    rng = np.random.default_rng(20260727)
    nsites = 4
    one_body = rng.normal(scale=0.03, size=(nsites, nsites))
    one_body = 0.5 * (one_body + one_body.T)
    cholesky = rng.normal(scale=0.02, size=(nsites, nsites, 4))
    cholesky = 0.5 * (cholesky + cholesky.swapaxes(0, 1))
    eri = np.einsum("pqL,rsL->pqrs", cholesky, cholesky)
    owner = SU2MovingEnvironment(
        one_body,
        eri,
        4,
        two_s=0,
        cutoff=1.0e-12,
        include_half=True,
    )
    mpo = build_su2_normal_complementary_mpo(owner, fully_reduced=True)
    for factor in mpo:
        object.__setattr__(
            factor,
            "normal_complementary_right_dual",
            True,
        )

    factor = mpo[-1]
    physical = {sector.charge: sector for sector in factor.phys_in_leg.sectors}
    inner_bra = SpinChargeSector(0, SU2Irrep(1))
    inner_ket = SpinChargeSector(0, SU2Irrep(1))
    outer_bra = SpinChargeSector(0, SU2Irrep(0))
    outer_ket = SpinChargeSector(0, SU2Irrep(0))
    sectors = (
        inner_bra,
        inner_ket,
        physical[1],
        physical[1],
        outer_bra,
        outer_ket,
    )
    reference = _right_reduced_rank_coupled_block(factor, *sectors)
    connected_reference = {}
    for (source, target), block in reference.items():
        source_irrep = factor.left_channel_irreps[source]
        weights = np.asarray(
            [
                _component_basis_norm(
                    inner_bra,
                    inner_ket,
                    source_irrep,
                    two_m,
                )
                for two_m in ordered_two_m_values(source_irrep)
            ]
        )
        connected_reference[(source, target)] = (
            np.asarray(block) * weights[:, None, None, None]
        )

    actual = owner.contextual_core(
        nsites - 1,
        1,
        1,
        outer_bra.irrep.two_j,
        outer_ket.irrep.two_j,
        physical[1].irrep.two_j,
        physical[1].irrep.two_j,
        inner_bra.irrep.two_j,
        inner_ket.irrep.two_j,
        False,
        True,
    )
    assert actual.keys() == connected_reference.keys()
    for key, expected in connected_reference.items():
        np.testing.assert_allclose(
            actual[key],
            expected,
            rtol=1.0e-12,
            atol=1.0e-12,
        )


def test_fully_reduced_two_site_exchange_eri_matrix_matches_exact_reduced_cg_reference():
    nsites = 2
    vacuum = spatial_target_sector(0, 0)
    target = spatial_target_sector(2, 0)
    path_specs = (
        ((0, 2), (vacuum, vacuum, target)),
        ((1, 1), (vacuum, spatial_target_sector(1, 1), target)),
        ((2, 0), (vacuum, target, target)),
    )
    basis_states, dense_vectors = _reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())

    for pattern in ((0, 1, 1, 0), (1, 0, 0, 1)):
        eri = np.zeros((nsites, nsites, nsites, nsites))
        eri[pattern] = 1.0
        autompo = AutoMPO([phys_leg] * nsites)
        add_spatial_spinfree_eri_terms(autompo, eri, cutoff=1.0e-12)
        mpo = autompo.build()
        expected_operator = _dense_spatial_spinfree_eri_hamiltonian(eri)

        for bra_index, bra_state in enumerate(basis_states):
            for ket_index, ket_state in enumerate(basis_states):
                expected = np.vdot(
                    dense_vectors[bra_index],
                    expected_operator @ dense_vectors[ket_index],
                )
                actual = _contract_chain_transition(bra_state, mpo, ket_state)
                assert actual == pytest.approx(expected, abs=1.0e-12)


def test_fully_reduced_scalar_pair_hopping_survives_rank_coupled_embedding():
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    vacuum = spatial_target_sector(0, 0)
    target = spatial_target_sector(2, 0)
    path_specs = (
        ((0, 2), (vacuum, vacuum, target)),
        ((2, 0), (vacuum, target, target)),
    )
    basis_states, _dense_vectors = _reduced_spatial_path_basis(path_specs)

    autompo = AutoMPO([phys_leg] * 2)
    autompo.add_term(
        (0, spatial_pair_creation(phys_leg)),
        (1, spatial_pair_annihilation(phys_leg)),
        coeff=1.0,
    )
    scalar_mpo = autompo.build()
    rank_coupled_mpo = [as_rank_coupled_mpo(core) for core in scalar_mpo]

    np.testing.assert_allclose(
        [
            [
                _contract_chain_transition(bra_state, scalar_mpo, ket_state)
                for ket_state in basis_states
            ]
            for bra_state in basis_states
        ],
        [
            [
                _contract_chain_transition(bra_state, rank_coupled_mpo, ket_state)
                for ket_state in basis_states
            ]
            for bra_state in basis_states
        ],
        atol=1.0e-12,
    )


def test_fully_reduced_exchange_eri_matrix_matches_exact_reduced_cg_reference():
    nsites = 3
    vacuum = spatial_target_sector(0, 0)
    single = spatial_target_sector(1, 1)
    target = spatial_target_sector(2, 0)
    path_specs = (
        ((0, 0, 2), (vacuum, vacuum, vacuum, target)),
        ((0, 1, 1), (vacuum, vacuum, single, target)),
        ((0, 2, 0), (vacuum, vacuum, target, target)),
        ((1, 0, 1), (vacuum, single, single, target)),
        ((1, 1, 0), (vacuum, single, target, target)),
        ((2, 0, 0), (vacuum, target, target, target)),
    )
    basis_states, dense_vectors = _reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    exchange_patterns = (
        (0, 1, 1, 2),
        (0, 1, 2, 0),
        (0, 2, 1, 0),
        (0, 2, 2, 1),
        (1, 0, 0, 2),
        (1, 0, 2, 1),
        (1, 2, 0, 1),
        (1, 2, 2, 0),
        (2, 0, 0, 1),
        (2, 0, 1, 2),
        (2, 1, 0, 2),
        (2, 1, 1, 0),
    )

    for pattern in exchange_patterns:
        eri = np.zeros((nsites, nsites, nsites, nsites))
        eri[pattern] = 1.0
        autompo = AutoMPO([phys_leg] * nsites)
        add_spatial_spinfree_eri_terms(autompo, eri, cutoff=1.0e-12)
        mpo = autompo.build()
        expected_operator = _dense_spatial_spinfree_eri_hamiltonian(eri)

        for bra_index, bra_state in enumerate(basis_states):
            for ket_index, ket_state in enumerate(basis_states):
                expected = np.vdot(
                    dense_vectors[bra_index],
                    expected_operator @ dense_vectors[ket_index],
                )
                actual = _contract_chain_transition(bra_state, mpo, ket_state)
                assert actual == pytest.approx(expected, abs=1.0e-12)


def test_fully_reduced_adjacent_one_body_matrix_matches_exact_reduced_cg_reference():
    nsites = 4
    path_specs = _reduced_spatial_path_specs(
        nsites,
        spatial_target_sector(4, 0),
    )
    basis_states, dense_vectors = _reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())

    for create_site, annihilate_site in ((0, 1), (1, 0)):
        h1e = np.zeros((nsites, nsites))
        h1e[create_site, annihilate_site] = 1.0
        mpo = build_spatial_one_body_reduced_mpo(
            [phys_leg] * nsites,
            h1e,
            cutoff=1.0e-12,
        )
        expected_operator = _dense_spatial_one_body_hamiltonian(h1e)

        for bra_index, bra_state in enumerate(basis_states):
            for ket_index, ket_state in enumerate(basis_states):
                expected = np.vdot(
                    dense_vectors[bra_index],
                    expected_operator @ dense_vectors[ket_index],
                )
                actual = _contract_chain_transition(bra_state, mpo, ket_state)
                assert actual == pytest.approx(expected, abs=1.0e-12)


def test_fully_reduced_one_body_embedded_matrix_matches_exact_reduced_cg_reference():
    nsites = 4
    path_specs = _reduced_spatial_path_specs(
        nsites,
        spatial_target_sector(4, 0),
    )
    basis_states, dense_vectors = _reduced_spatial_path_basis(path_specs)
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())

    for create_site, annihilate_site in (
        (0, 0),
        (2, 2),
        (0, 1),
        (0, 2),
        (1, 3),
        (3, 0),
    ):
        h1e = np.zeros((nsites, nsites))
        h1e[create_site, annihilate_site] = 1.0
        mpo = build_spatial_one_body_reduced_mpo(
            [phys_leg] * nsites,
            h1e,
            cutoff=1.0e-12,
        )
        expected_operator = _dense_spatial_one_body_hamiltonian(h1e)

        for bra_index, bra_state in enumerate(basis_states):
            for ket_index, ket_state in enumerate(basis_states):
                expected = np.vdot(
                    dense_vectors[bra_index],
                    expected_operator @ dense_vectors[ket_index],
                )
                actual = _contract_chain_transition(bra_state, mpo, ket_state)
                assert actual == pytest.approx(expected, abs=1.0e-12)


def test_fully_reduced_four_distinct_eri_refuses_inexact_recursive_growth():
    phys_leg = physical_leg_from_spatial_orbital(FullyReducedSpatialOrbitalSite())
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 2, 1, 3] = 0.07
    autompo = AutoMPO([phys_leg] * 4)
    with pytest.raises(NotImplementedError, match="four-site recoupling data"):
        add_spatial_spinfree_eri_terms(autompo, eri)


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


def test_build_spatial_one_body_reduced_mpo_matches_dense_reference():
    h1e = np.array(
        [
            [0.2, -0.03, 0.04],
            [-0.03, -0.1, 0.07],
            [0.04, 0.07, 0.5],
        ]
    )
    built = build_spatial_one_body_reduced_mpo(3, h1e)
    built_dense = _dense_matrix_from_mpo_list(built)
    ref_dense = _dense_spatial_one_body_hamiltonian(h1e)
    np.testing.assert_allclose(built_dense, ref_dense, atol=1e-12)
    assert any(isinstance(core, RankCoupledMPO) for core in built)


def test_build_spatial_spinfree_eri_mpo_matches_dense_reference_for_generic_terms():
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 2, 1, 0] = 0.07
    eri[2, 0, 0, 1] = -0.04
    eri[1, 1, 2, 2] = 0.11
    eri[0, 1, 1, 2] = 0.05

    built = build_spatial_spinfree_eri_mpo(3, eri)
    built_dense = _dense_matrix_from_mpo_list(built)
    ref_dense = _dense_spatial_spinfree_eri_hamiltonian(eri)
    np.testing.assert_allclose(built_dense, ref_dense, atol=1e-12)


def test_add_spatial_spinfree_eri_terms_matches_direct_builder():
    A, B, C = _spatial_chain()
    eri = np.zeros((3, 3, 3, 3))
    eri[0, 2, 1, 0] = 0.07
    eri[2, 0, 0, 1] = -0.04

    auto = AutoMPO.from_sites([A, B, C])
    count = add_spatial_spinfree_eri_terms(auto, eri)
    built_dense = _dense_matrix_from_mpo_list(auto.build())
    ref_dense = _dense_matrix_from_mpo_list(build_spatial_spinfree_eri_mpo(3, eri))

    assert count > 0
    np.testing.assert_allclose(built_dense, ref_dense, atol=1e-12)


def test_spatial_spinfree_eri_builder_uses_reduced_we_for_four_distinct_terms():
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 2, 1, 3] = 0.07

    auto = AutoMPO([physical_leg_from_spatial_orbital()] * 4)
    info = add_spatial_spinfree_eri_terms(auto, eri, return_info=True)
    built = auto.build()
    built_dense = _dense_matrix_from_mpo_list(built)
    ref_dense = _dense_spatial_spinfree_eri_hamiltonian(eri)

    assert info["we_product_terms"] > 0
    assert info["scalar_product_terms"] == 0
    assert any(getattr(core, "reduced_terms", ()) for core in built)
    np.testing.assert_allclose(built_dense, ref_dense, atol=1e-12)


def test_analytic_four_site_recoupling_matches_dense_reference_projection():
    for order in permutations(range(4)):
        analytic = nonabelian_models._spinfree_we_recoupling_coefficients(order)
        reference = nonabelian_models._reference_spinfree_we_recoupling_coefficients(order)
        np.testing.assert_allclose(analytic, reference, atol=1e-12)


def test_analytic_exchange_recoupling_matches_dense_reference_projection():
    for pattern in nonabelian_models._SPINFREE_EXCHANGE_CHANNELS:
        analytic = nonabelian_models._spinfree_exchange_recoupling_coefficients(pattern)
        reference = nonabelian_models._reference_spinfree_exchange_recoupling_coefficients(pattern)
        assert [ranks for _coeff, ranks in analytic] == [ranks for _coeff, ranks in reference]
        np.testing.assert_allclose(
            [coeff for coeff, _ranks in analytic],
            [coeff for coeff, _ranks in reference],
            atol=1e-12,
        )


def test_active_spinfree_builder_does_not_use_dense_recoupling_projection(monkeypatch):
    def fail(*_args, **_kwargs):
        raise AssertionError("dense recoupling reference entered the active builder")

    monkeypatch.setattr(nonabelian_models, "_spinfree_component_target_dense", fail)
    monkeypatch.setattr(nonabelian_models, "_spinfree_component_target_dense_for_sites", fail)
    eri = np.zeros((4, 4, 4, 4))
    eri[0, 2, 1, 3] = 0.07
    eri[0, 1, 1, 2] = -0.03
    built = nonabelian_models.build_spatial_spinfree_eri_mpo(4, eri)
    assert built


def test_reduced_channels_expand_through_trailing_scalar_only_cores():
    h1e = np.zeros((4, 4))
    h1e[0, 2] = 0.11
    built = build_spatial_one_body_reduced_mpo(4, h1e)
    dense_shapes = [core.as_dense().shape for core in built]

    for left, right in zip(dense_shapes, dense_shapes[1:]):
        assert left[1] == right[0]
    assert any(isinstance(core, RankCoupledMPO) for core in built)


def test_rank_coupled_mpo_can_use_clebsch_gordan_virtual_recoupling():
    phys_leg = physical_leg_from_spatial_orbital()
    annihilate = reduced_spatial_fermion_annihilation()
    left_irrep = SU2Irrep(1)
    right_irrep = SU2Irrep(0)
    core = RankCoupledMPO(
        dense_blocks={},
        left_channel_irreps=(left_irrep,),
        right_channel_irreps=(right_irrep,),
        reduced_terms=(
            RankCoupledChannelTerm(
                reduced_operator=annihilate,
                visible_virtual_block=np.ones((1, 1)),
                use_cg_coupling=True,
            ),
        ),
        phys_out_leg=phys_leg,
        phys_in_leg=phys_leg,
    )

    dense = core.as_dense()
    expected = np.zeros_like(dense)
    for row, two_m_left in enumerate((1, -1)):
        component = -two_m_left
        coeff = clebsch_gordan(
            left_irrep,
            annihilate.rank_irrep,
            right_irrep,
            two_m_left,
            component,
            0,
        )
        expected[row, 0] = coeff * annihilate.component(component).as_dense()

    np.testing.assert_allclose(dense, expected, atol=1e-12)


def test_autompo_reduced_string_uses_clebsch_gordan_recoupling():
    phys_leg = physical_leg_from_spatial_orbital()
    annihilate = reduced_spatial_fermion_annihilation()
    builder = AutoMPO([phys_leg] * 4)
    builder.add_reduced_string(
        (0, annihilate),
        (1, annihilate),
        (2, annihilate),
        (3, annihilate),
        intermediate_irreps=(SU2Irrep(1), SU2Irrep(0), SU2Irrep(1)),
        coeff=0.7,
    )
    built = _dense_matrix_from_mpo_list(builder.build())

    def component_dense(operator, component):
        blocks = {}
        for q_out in phys_leg.sectors:
            for q_in in phys_leg.sectors:
                block = operator.component_block(component, q_out, q_in)
                if block is not None:
                    blocks[(q_out, q_in)] = block
        return SiteOperator(
            blocks=blocks,
            phys_out_leg=phys_leg,
            phys_in_leg=phys_leg,
        ).as_dense()

    expected = np.zeros_like(built)
    j0 = SU2Irrep(0)
    j1 = SU2Irrep(1)
    j2 = SU2Irrep(0)
    j3 = SU2Irrep(1)
    for m1 in (1, -1):
        q0 = m1
        q1 = -m1
        c01 = clebsch_gordan(j0, annihilate.rank_irrep, j1, 0, q0, m1)
        c12 = clebsch_gordan(j1, annihilate.rank_irrep, j2, m1, q1, 0)
        for m3 in (1, -1):
            q2 = m3
            q3 = -m3
            c23 = clebsch_gordan(j2, annihilate.rank_irrep, j3, 0, q2, m3)
            c34 = clebsch_gordan(j3, annihilate.rank_irrep, j0, m3, q3, 0)
            expected += 0.7 * c01 * c12 * c23 * c34 * _kron_all(
                [
                    component_dense(annihilate, q0),
                    component_dense(annihilate, q1),
                    component_dense(annihilate, q2),
                    component_dense(annihilate, q3),
                ]
            )

    np.testing.assert_allclose(built, expected, atol=1e-12)


def test_coupled_reduced_tensor_product_matches_component_product():
    phys_leg = physical_leg_from_spatial_orbital()
    annihilate = reduced_spatial_fermion_annihilation()
    creation = annihilate.adjoint()
    dual_annihilate = time_reversed_reduced_operator(annihilate)

    for rank in (SU2Irrep(0), SU2Irrep(2)):
        coupled = coupled_reduced_tensor_product(creation, dual_annihilate, rank)
        for component in coupled.components:
            built = coupled.component(component).as_dense()
            expected = np.zeros_like(built)
            for left_component in creation.components:
                for right_component in dual_annihilate.components:
                    if int(left_component) + int(right_component) != int(component):
                        continue
                    coeff = clebsch_gordan(
                        creation.rank_irrep,
                        dual_annihilate.rank_irrep,
                        rank,
                        left_component,
                        right_component,
                        component,
                    )
                    expected += (
                        coeff
                        * creation.component(left_component).as_dense()
                        @ dual_annihilate.component(right_component).as_dense()
                    )
            np.testing.assert_allclose(built, expected, atol=1e-12)
        assert coupled.phys_out_leg == phys_leg
        assert coupled.phys_in_leg == phys_leg


def test_add_spatial_one_body_terms_matches_direct_builder():
    A, B, C = _spatial_chain()
    h1e = np.array(
        [
            [0.2, 0.05, -0.01],
            [0.05, -0.3, 0.02],
            [-0.01, 0.02, 0.4],
        ]
    )
    auto = AutoMPO.from_sites([A, B, C])
    add_spatial_one_body_terms(auto, h1e)
    built_dense = _dense_matrix_from_mpo_list(auto.build())
    ref_dense = _dense_spatial_one_body_hamiltonian(h1e)
    np.testing.assert_allclose(built_dense, ref_dense, atol=1e-12)


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
