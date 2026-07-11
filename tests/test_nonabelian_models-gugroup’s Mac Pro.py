import numpy as np
import pytest

from pyqed.mps.nonabelian import (
    AutoMPO,
    NonabelianTensor,
    RankCoupledMPO,
    add_spatial_density_terms,
    build_hubbard_mpo,
    add_spatial_hubbard_terms,
    build_block_sparse_bond_operator,
    build_product_spatial_mps,
    build_random_spatial_mps,
    build_random_reduced_spatial_mps,
    build_reduced_product_spatial_mps,
    build_spatial_density_mpo,
    build_spatial_hubbard_mpo,
    build_spatial_qchem_mpo,
    contract_chain_expectation,
    half_filled_singlet_sector,
    merge_mps_sites,
    run_sweeps,
    solve_local_two_site,
    spatial_annihilate_down,
    spatial_annihilate_up,
    spatial_create_down,
    spatial_create_up,
    spatial_double_occupancy,
    spatial_number,
    spatial_parity,
)
from pyqed.mps.nonabelian.coupling import clebsch_gordan, ordered_two_m_values
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


def _dense_half_filled_hubbard_energy(nsites, *, hopping_t, chemical_potential, onsite_u):
    hamiltonian = _dense_spatial_hubbard_hamiltonian(
        nsites,
        hopping_t=hopping_t,
        chemical_potential=chemical_potential,
        onsite_u=onsite_u,
    )
    local_occupations = np.array([0, 1, 1, 2])
    keep = []
    for state in range(4**nsites):
        encoded = state
        charge = 0
        for _ in range(nsites):
            charge += int(local_occupations[encoded % 4])
            encoded //= 4
        if charge == nsites:
            keep.append(state)
    projected = hamiltonian[np.ix_(keep, keep)]
    return float(np.linalg.eigvalsh(projected)[0].real)


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
        boundary = {(sites[0].qns[0][0], 0, 0): 1.0 + 0.0j}
        for tensor in sites:
            physical_index = encoded % site.d
            encoded //= site.d
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


def test_reduced_spatial_density_mpo_matches_legacy_scalar_expectation():
    labels = ["full", "empty", "full", "empty"]
    params = {
        "chemical_potential": 0.3,
        "onsite_u": 2.0,
        "nearest_neighbor_v": 0.5,
    }
    explicit_sites = build_product_spatial_mps(labels)
    reduced_sites = build_reduced_product_spatial_mps(labels)

    explicit_mpo = build_spatial_density_mpo(explicit_sites, **params)
    reduced_mpo = build_spatial_density_mpo(reduced_sites, **params)

    explicit_energy = contract_chain_expectation(explicit_sites, explicit_mpo)
    reduced_energy = contract_chain_expectation(reduced_sites, reduced_mpo)
    assert reduced_energy == pytest.approx(explicit_energy)


def test_reduced_spatial_density_sweep_matches_legacy_scalar_backend():
    labels = ["full", "empty", "full", "empty"]
    params = {
        "chemical_potential": 0.0,
        "onsite_u": 2.0,
        "nearest_neighbor_v": 0.5,
    }
    explicit_sites = build_product_spatial_mps(labels)
    reduced_sites = build_reduced_product_spatial_mps(labels)
    explicit_mpo = build_spatial_density_mpo(explicit_sites, **params)
    reduced_mpo = build_spatial_density_mpo(reduced_sites, **params)

    common_kwargs = {
        "nsweeps": 1,
        "max_bond": 8,
        "max_bond_mode": "reduced",
        "cutoff": 1e-12,
        "conv_tol": None,
        "local_solver_kwargs": {"itermax": 20},
        "mixer_zero_block_noise_scale": 0.0,
    }
    explicit = run_sweeps(explicit_sites, mpo_factors=explicit_mpo, **common_kwargs)
    reduced = run_sweeps(reduced_sites, mpo_factors=reduced_mpo, **common_kwargs)

    assert reduced["history"][-1]["energy"] == pytest.approx(explicit["history"][-1]["energy"])
    assert reduced["history"][-1]["local_problem_counts"] == {"standard": 3}


def test_two_site_reduced_spatial_hubbard_matches_dense_reference():
    params = {
        "hopping_t": 1.0,
        "chemical_potential": 0.0,
        "onsite_u": 4.0,
    }
    reduced_sites = build_reduced_product_spatial_mps(["full", "empty"])
    reduced_mpo = build_spatial_hubbard_mpo(reduced_sites, **params)
    merged = merge_mps_sites(reduced_sites[0], reduced_sites[1])
    operator = build_block_sparse_bond_operator(reduced_sites, reduced_mpo, 0, merged)
    _optimized, objective = solve_local_two_site(
        merged,
        operator,
        canonical_norm=True,
        itermax=80,
    )
    exact = _dense_half_filled_hubbard_energy(2, **params)
    assert objective["energy"] == pytest.approx(exact, abs=1.0e-12)


def test_two_site_reduced_spatial_hubbard_expectation_matches_full_cg_dense_reference():
    from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo

    params = {
        "hopping_t": 1.0,
        "chemical_potential": 0.0,
        "onsite_u": 4.0,
    }
    sites = build_random_reduced_spatial_mps(
        2,
        target_sector=half_filled_singlet_sector(2),
        bond_multiplicity=3,
        seed=3,
    )
    mpo = build_spatial_hubbard_mpo(sites, **params)
    dense_vector = _dense_vector_from_reduced_spatial_mps(sites)
    dense_hamiltonian = _dense_spatial_hubbard_hamiltonian(2, **params)

    expected_norm = np.vdot(dense_vector, dense_vector)
    expected_energy = np.vdot(dense_vector, dense_hamiltonian @ dense_vector)
    identity_mpo = _identity_mpo_factors_for_sites_and_mpo(sites, mpo)

    assert contract_chain_expectation(sites, identity_mpo) == pytest.approx(expected_norm)
    assert contract_chain_expectation(sites, mpo) == pytest.approx(expected_energy)


def test_two_site_reduced_spatial_hubbard_coupled_matrix_matches_dense_reference():
    from pyqed.mps.nonabelian.solver import _coupled_two_site_template, _pack_tensor_state

    params = {
        "hopping_t": 1.0,
        "chemical_potential": 0.0,
        "onsite_u": 4.0,
    }
    reduced_sites = build_reduced_product_spatial_mps(["full", "empty"])
    reduced_mpo = build_spatial_hubbard_mpo(reduced_sites, **params)
    merged = merge_mps_sites(reduced_sites[0], reduced_sites[1])
    operator = build_block_sparse_bond_operator(reduced_sites, reduced_mpo, 0, merged)
    coupled = _coupled_two_site_template(merged)
    _packed, coupled_layout = _pack_tensor_state(coupled)
    coupled_matrix = operator.coupled_matrix_factory(coupled, coupled_layout)

    exact = _dense_half_filled_hubbard_energy(2, **params)
    np.testing.assert_allclose(coupled_matrix, coupled_matrix.conj().T, atol=1.0e-12)
    assert float(np.linalg.eigvalsh(coupled_matrix)[0].real) == pytest.approx(
        exact,
        abs=1.0e-12,
    )


def test_four_site_legacy_spatial_hubbard_matches_dense_reference():
    params = {
        "hopping_t": 1.0,
        "chemical_potential": 0.0,
        "onsite_u": 4.0,
    }
    explicit_sites = build_random_spatial_mps(
        4,
        target_sector=half_filled_singlet_sector(4),
        bond_multiplicity=8,
        seed=14,
    )
    explicit_mpo = build_spatial_hubbard_mpo(explicit_sites, **params)
    result = run_sweeps(
        explicit_sites,
        mpo_factors=explicit_mpo,
        nsweeps=4,
        max_bond=16,
        max_bond_mode="reduced",
        cutoff=1e-12,
        conv_tol=None,
        local_solver_kwargs={"itermax": 80},
        mixer_zero_block_noise_scale=0.0,
    )
    exact = _dense_half_filled_hubbard_energy(4, **params)
    assert result["history"][-1]["energy"] == pytest.approx(exact, abs=3.0e-4)


@pytest.mark.xfail(
    reason=(
        "Degeneracy-only Hubbard still needs the optimized coupled two-site "
        "fusion channels preserved through the site split for L=4."
    ),
    strict=False,
)
def test_four_site_reduced_spatial_hubbard_matches_dense_reference():
    params = {
        "hopping_t": 1.0,
        "chemical_potential": 0.0,
        "onsite_u": 4.0,
    }
    reduced_sites = build_random_reduced_spatial_mps(
        4,
        target_sector=half_filled_singlet_sector(4),
        bond_multiplicity=8,
        seed=14,
    )
    reduced_mpo = build_spatial_hubbard_mpo(reduced_sites, **params)
    result = run_sweeps(
        reduced_sites,
        mpo_factors=reduced_mpo,
        nsweeps=4,
        max_bond=16,
        max_bond_mode="reduced",
        cutoff=1e-12,
        conv_tol=None,
        local_solver_kwargs={"itermax": 80},
        mixer_zero_block_noise_scale=0.0,
    )
    exact = _dense_half_filled_hubbard_energy(4, **params)
    assert result["history"][-1]["energy"] == pytest.approx(exact, abs=3.0e-4)


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


def test_build_spatial_qchem_mpo_matches_dense_reference():
    from pyqed.qchem.dmrg.dmrg import _build_spatial_active_hamiltonian_matrix

    h1 = np.array([[0.2, 0.03], [0.03, -0.1]])
    eri_aa = np.zeros((2, 2, 2, 2))
    eri_aa[0, 0, 0, 0] = 0.7
    eri_aa[1, 1, 1, 1] = 0.5
    eri_aa[0, 0, 1, 1] = 0.2
    eri_aa[1, 1, 0, 0] = 0.2
    eri_aa[0, 1, 1, 0] = 0.06
    h2 = np.stack((np.stack((eri_aa, eri_aa.copy())), np.stack((eri_aa.copy(), eri_aa.copy()))))

    dense_ref, _ = _build_spatial_active_hamiltonian_matrix([h1, h1], h2)
    built_dense = _dense_matrix_from_mpo_list(build_spatial_qchem_mpo(2, [h1, h1], h2))

    np.testing.assert_allclose(built_dense, dense_ref, atol=1e-12)
    np.testing.assert_allclose(built_dense, built_dense.conj().T, atol=1e-12)


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
