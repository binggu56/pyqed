import numpy as np
from scipy.sparse.linalg import eigsh

from pyqed.narg.qchem import NARG, NARGSCF
from pyqed.narg.qchem import abelian as abelian_narg
from pyqed.narg.qchem.active_space import prepare_active_space
from pyqed.narg.qchem.rdm import (
    spin_traced_rdm1_from_state,
    spin_traced_rdm2_from_state,
)
from pyqed.narg.core import narg_state_vector
from pyqed.qchem import CASCI, Molecule
from pyqed.mps.fermion import SpinHalfFermionChain


def _lih_sto3g_mf():
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    return mol.RHF().run()


def _hubbard_integrals(nsites, *, t, u):
    h1e = np.zeros((nsites, nsites))
    for i in range(nsites - 1):
        h1e[i, i + 1] = h1e[i + 1, i] = -float(t)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for i in range(nsites):
        eri[i, i, i, i] = float(u)
    return h1e, eri


def _assert_stored_state_energy(solver, h1e, eri, energy, *, atol=1.0e-10):
    psi = narg_state_vector(solver.tensors[:-1], solver.tensors[-1], root=0)
    hamiltonian = SpinHalfFermionChain(h1e, eri).jordan_wigner(
        forward=False
    ).toarray()
    expectation = np.vdot(psi, hamiltonian @ psi) / np.vdot(psi, psi)
    np.testing.assert_allclose(np.real(expectation), energy, atol=atol)


def _periodic_hubbard_integrals(nsites, *, t, u):
    h1e = np.zeros((nsites, nsites))
    for i in range(nsites):
        j = (i + 1) % nsites
        h1e[i, j] += -float(t)
        h1e[j, i] += -float(t)
    eri = np.zeros((nsites, nsites, nsites, nsites))
    for i in range(nsites):
        eri[i, i, i, i] = float(u)
    return h1e, eri


def _real_momentum_hubbard_integrals(nsites, *, t, u, order="energy", nelec=None):
    h1e, eri = _periodic_hubbard_integrals(nsites, t=t, u=u)
    sites = np.arange(nsites)
    cols = [np.ones(nsites) / np.sqrt(nsites)]
    eps = [-2.0 * float(t)]
    for m in range(1, nsites // 2):
        theta = 2.0 * np.pi * m * sites / nsites
        energy = -2.0 * float(t) * np.cos(2.0 * np.pi * m / nsites)
        cols.append(np.sqrt(2.0 / nsites) * np.cos(theta))
        cols.append(np.sqrt(2.0 / nsites) * np.sin(theta))
        eps.extend((energy, energy))
    cols.append((-1.0) ** sites / np.sqrt(nsites))
    eps.append(2.0 * float(t))
    transform = np.column_stack(cols)
    eps = np.asarray(eps)
    if order == "energy":
        orbital_order = np.argsort(eps, kind="stable")
    elif order == "fermi":
        sorted_eps = np.sort(eps)
        nocc = int(nelec[0] if isinstance(nelec, tuple) else nsites // 2)
        mu = 0.5 * (sorted_eps[nocc - 1] + sorted_eps[nocc])
        orbital_order = np.lexsort((np.arange(nsites), np.abs(eps - mu)))
    elif order in {"particle_hole", "ph"}:
        sorted_eps = np.sort(eps)
        nocc = int(nelec[0] if isinstance(nelec, tuple) else nsites // 2)
        mu = 0.5 * (sorted_eps[nocc - 1] + sorted_eps[nocc])
        shell_tol = 1e-12
        interleaved = [
            i
            for i in np.lexsort((np.arange(nsites), np.abs(eps - mu)))
            if abs(eps[i] - mu) <= shell_tol
        ]
        holes = [i for i in np.lexsort((np.arange(nsites), mu - eps)) if eps[i] < mu - shell_tol]
        particles = [i for i in np.lexsort((np.arange(nsites), eps - mu)) if eps[i] > mu + shell_tol]
        for hole, particle in zip(holes, particles):
            interleaved.extend((hole, particle))
        interleaved.extend(holes[len(particles):])
        interleaved.extend(particles[len(holes):])
        orbital_order = np.asarray(interleaved)
    else:
        raise ValueError("order must be 'energy', 'fermi', or 'particle_hole'.")
    transform = transform[:, orbital_order]
    h1e_mom = transform.T @ h1e @ transform
    eri_mom = np.einsum("ijkl,ip,jq,kr,ls->pqrs", eri, transform, transform, transform, transform, optimize=True)
    return h1e_mom, eri_mom


def _sector_ground_energy(h1e, eri, nelec):
    model = SpinHalfFermionChain(h1e, eri, nelec=nelec)
    hamiltonian = model.jordan_wigner()
    nu = np.rint(model.Nu_tot.diagonal()).astype(int)
    nd = np.rint(model.Nd_tot.diagonal()).astype(int)
    idx = np.flatnonzero((nu == nelec[0]) & (nd == nelec[1]))
    sector_hamiltonian = hamiltonian[np.ix_(idx, idx)]
    e, _ = eigsh(sector_hamiltonian, k=1, which="SA")
    return e


def _hchain_sto3g_mf(nsites, spacing):
    atom = "; ".join(f"H 0 0 {spacing * idx:.8f}" for idx in range(nsites))
    mol = Molecule(atom=atom, unit="angstrom", basis="sto-3g")
    mol.build()
    return mol.RHF().run()


def _h2_sto3g_mf():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(eri="dense")
    return mol.RHF().run(verbose=0)


def test_abelian_irrep_tensor_roundtrip_keeps_charge_shift_blocks():
    tensor = abelian_narg.labeled_irrep_tensor(
        abelian_narg.cdu,
        abelian_narg.LOCAL_QN,
        op=abelian_narg.OPERATOR_QN_SHIFT["Cdu"],
    )
    dense = abelian_narg.labeled_dense(tensor, abelian_narg.LOCAL_QN)

    np.testing.assert_allclose(dense, abelian_narg.cdu, atol=1e-12)
    assert tensor.op.charge == tuple(abelian_narg.OPERATOR_QN_SHIFT["Cdu"])


def test_abelian_irrep_product_rotation_matches_dense_rotate():
    block_qn = abelian_narg.LOCAL_QN.copy()
    n = len(block_qn)
    d = len(abelian_narg.LOCAL_QN)
    D = n
    U = np.zeros((n, D, d))
    for local_id in range(d):
        U[:, :, local_id] = np.eye(n)
    output_qn = abelian_narg.branch_qn(block_qn)

    cases = [
        (
            abelian_narg.cdu,
            abelian_narg.JW,
            abelian_narg.OPERATOR_QN_SHIFT["Cdu"],
        ),
        (
            abelian_narg.cdu,
            abelian_narg.JW @ abelian_narg.cu,
            abelian_narg.OPERATOR_QN_SHIFT["Cdu"] + abelian_narg.OPERATOR_QN_SHIFT["Cu"],
        ),
    ]
    for block_op, local_op, shift in cases:
        rotated, _ = abelian_narg.rotate_irrep_product(
            block_op,
            local_op,
            U,
            block_qn,
            output_qn,
            shift,
        )
        dense_irrep = abelian_narg.labeled_dense(rotated, output_qn)
        dense_ref = abelian_narg.rotate(block_op, local_op, U)
        np.testing.assert_allclose(dense_irrep, dense_ref, atol=1e-12)


def test_abelian_block_from_dense_carries_operator_table_as_irrep_tensors():
    h = np.diag(np.diag(abelian_narg.Ntot))
    table = abelian_narg.make_operator_table(
        {
            "Cdu": [abelian_narg.cdu],
            "Cdd": [abelian_narg.cdd],
            "Cu": [abelian_narg.cu],
            "Cd": [abelian_narg.cd],
        },
        abelian_narg.SINGLE_PATTERNS,
        1,
    )

    block = abelian_narg.abelian_block_from_dense(h, abelian_narg.LOCAL_QN, table)

    np.testing.assert_allclose(block.dense_h(), h, atol=1e-12)
    cdu_tensor = block.ops[((("Cdu",), (0,)))]
    np.testing.assert_allclose(
        abelian_narg.labeled_dense(cdu_tensor, abelian_narg.LOCAL_QN),
        abelian_narg.cdu,
        atol=1e-12,
    )
    assert cdu_tensor.op.charge == tuple(abelian_narg.OPERATOR_QN_SHIFT["Cdu"])


def test_extend_irrep_operator_table_matches_dense_operator_table():
    block_qn = abelian_narg.LOCAL_QN.copy()
    n = len(block_qn)
    d = len(abelian_narg.LOCAL_QN)
    D = n
    U = np.zeros((n, D, d))
    for local_id in range(d):
        U[:, :, local_id] = np.eye(n)
    output_qn = abelian_narg.branch_qn(block_qn)
    table = abelian_narg.make_operator_table(
        {
            "Cdu": [abelian_narg.cdu],
            "Cdd": [abelian_narg.cdd],
            "Cu": [abelian_narg.cu],
            "Cd": [abelian_narg.cd],
        },
        abelian_narg.OPERATOR_PATTERNS,
        1,
    )

    dense_ext = abelian_narg.extend_operator_table(
        table,
        abelian_narg.OPERATOR_PATTERNS,
        U,
        1,
        output_qn=output_qn,
        use_irrep_blocks=False,
    )
    irrep_table = abelian_narg.irrep_operator_table(table, block_qn)
    irrep_ext = abelian_narg.extend_irrep_operator_table(
        irrep_table,
        abelian_narg.OPERATOR_PATTERNS,
        U,
        block_qn,
        output_qn,
        1,
    )

    for pattern, entries in dense_ext.items():
        for indices, dense_op in entries.items():
            irrep_op = irrep_ext[(pattern, indices)]
            np.testing.assert_allclose(
                abelian_narg.labeled_dense(irrep_op, output_qn),
                dense_op,
                atol=1e-12,
            )


def test_abelian_kernel_irrep_operator_table_matches_dense_table_path():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    abelian_narg.mol = DummyMol()
    for nsites, D in [(4, 12), (6, 16)]:
        h1e, eri = _hubbard_integrals(nsites, t=0.7, u=2.0)
        e_dense = abelian_narg.kernel(
            h1e,
            eri,
            D=D,
            n0=1,
            nstates=1,
            growth_sites=1,
        )[0]
        e_irrep_table = abelian_narg.kernel(
            h1e,
            eri,
            D=D,
            n0=1,
            nstates=1,
            growth_sites=1,
            use_irrep_operator_table=True,
        )[0]

        np.testing.assert_allclose(e_irrep_table, e_dense, atol=1e-10)


def test_abelian_fast_energy_matches_returned_tensor_expectation_h6():
    mf = _hchain_sto3g_mf(6, 1.4)
    h1e, eri, active_mol, active_space = prepare_active_space(
        mf,
        mf.mol,
        ncas=6,
        nelecas=6,
        use_cholesky=True,
    )
    abelian_narg.mol = active_mol

    e, _x, tensors, _tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=64,
        n0=1,
        nstates=1,
        growth_sites=1,
        fast=True,
        return_tensors=True,
        return_tensor_qns=True,
    )

    hamiltonian = SpinHalfFermionChain(h1e, eri).jordan_wigner(forward=False).toarray()
    psi = narg_state_vector(tensors[:-1], tensors[-1], root=0)
    expectation = np.vdot(psi, hamiltonian @ psi) / np.vdot(psi, psi)

    np.testing.assert_allclose(e[0], np.real(expectation) + active_space.energy_core, atol=1e-10)


def test_abelian_irrep_pair_sums_match_dense_pair_sums():
    rng = np.random.default_rng(3)
    nblock = 2
    q = 2
    h1e = np.zeros((3, 3))
    eri = rng.normal(size=(3, 3, 3, 3))
    model = SpinHalfFermionChain(h1e[:nblock, :nblock], eri[:nblock, :nblock, :nblock, :nblock])
    model.jordan_wigner(forward=False)
    table = abelian_narg.make_operator_table(
        {
            "Cdu": model.Cdu,
            "Cdd": model.Cdd,
            "Cu": model.Cu,
            "Cd": model.Cd,
        },
        abelian_narg.OPERATOR_PATTERNS,
        nblock,
    )
    qn = abelian_narg.primitive_charge_labels(nblock)
    pair_terms, _ = abelian_narg.precompute_integral_terms(eri, cutoff=0.0, use_numba=False)

    dense = abelian_narg.build_pair_sums(table, pair_terms, q)
    irrep = abelian_narg.build_pair_sums_irrep(
        abelian_narg.irrep_operator_table(table, qn),
        pair_terms,
        q,
    )

    for name, dense_op in dense.items():
        np.testing.assert_allclose(
            abelian_narg.labeled_dense(irrep[name], qn),
            dense_op.toarray() if hasattr(dense_op, "toarray") else dense_op,
            atol=1e-12,
        )


def test_abelian_irrep_branch_hamiltonian_diagonalization_matches_dense():
    rng = np.random.default_rng(4)
    nblock = 2
    q = 2
    h1e = np.zeros((3, 3))
    eri = rng.normal(size=(3, 3, 3, 3))
    model = SpinHalfFermionChain(h1e[:nblock, :nblock], eri[:nblock, :nblock, :nblock, :nblock])
    H0 = model.jordan_wigner()
    table = abelian_narg.make_operator_table(
        {
            "Cdu": model.Cdu,
            "Cdd": model.Cdd,
            "Cu": model.Cu,
            "Cd": model.Cd,
        },
        abelian_narg.OPERATOR_PATTERNS,
        nblock,
    )
    qn = abelian_narg.primitive_charge_labels(nblock)
    pair_terms, _ = abelian_narg.precompute_integral_terms(eri, cutoff=0.0, use_numba=False)
    dense_pairs = abelian_narg.build_pair_sums(table, pair_terms, q)
    irrep_pairs = abelian_narg.build_pair_sums_irrep(abelian_narg.irrep_operator_table(table, qn), pair_terms, q)
    H0_irrep = abelian_narg.labeled_irrep_tensor(H0, qn, op=(0, 0))

    for nu, nd in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        dense_h = abelian_narg.branch_hamiltonian(H0, dense_pairs, nu, nd)
        irrep_h = abelian_narg.branch_hamiltonian_irrep(H0_irrep, irrep_pairs, nu, nd)
        np.testing.assert_allclose(abelian_narg.labeled_dense(irrep_h, qn), dense_h.toarray(), atol=1e-12)
        e_dense, _, qn_dense = abelian_narg.diagonalize_by_qn(dense_h, qn, 8)
        e_irrep, _, qn_irrep = abelian_narg.diagonalize_scalar_irrep_tensor(irrep_h, qn, 8)
        np.testing.assert_allclose(e_irrep, e_dense, atol=1e-12)
        np.testing.assert_array_equal(qn_irrep, qn_dense)


def test_abelian_block_sparse_branch_diagonalization_matches_dense():
    rng = np.random.default_rng(6)
    nblock = 3
    q = 3
    h1e = np.zeros((4, 4))
    eri = 0.1 * rng.normal(size=(4, 4, 4, 4))
    model = SpinHalfFermionChain(h1e[:nblock, :nblock], eri[:nblock, :nblock, :nblock, :nblock])
    H0 = model.jordan_wigner()
    table = abelian_narg.make_operator_table(
        {
            "Cdu": model.Cdu,
            "Cdd": model.Cdd,
            "Cu": model.Cu,
            "Cd": model.Cd,
        },
        abelian_narg.OPERATOR_PATTERNS,
        nblock,
    )
    qn = abelian_narg.primitive_charge_labels(nblock)
    pair_terms, _ = abelian_narg.precompute_integral_terms(eri, cutoff=0.0, use_numba=False)
    pair_sums = abelian_narg.build_pair_sums(table, pair_terms, q)
    allowed = {(2, 0), (3, 1), (3, -1)}

    for nu, nd in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        dense_h = abelian_narg.branch_hamiltonian(H0, pair_sums, nu, nd)
        e_dense, x_dense, qn_dense = abelian_narg.diagonalize_by_qn(dense_h, qn, 10, allowed_qn=allowed)
        e_block, x_block, qn_block = abelian_narg.branch_diagonalize_block_sparse(
            H0,
            pair_sums,
            nu,
            nd,
            qn,
            10,
            allowed_qn=allowed,
        )
        np.testing.assert_allclose(e_block, e_dense, atol=1e-12)
        assert sorted(map(tuple, qn_block)) == sorted(map(tuple, qn_dense))


def test_abelian_irrep_triple_residuals_match_dense_update():
    rng = np.random.default_rng(5)
    nblock = 2
    new_site = 2
    total_sites = 4
    h1e = np.zeros((total_sites, total_sites))
    eri = rng.normal(size=(total_sites, total_sites, total_sites, total_sites))
    model = SpinHalfFermionChain(h1e[:nblock, :nblock], eri[:nblock, :nblock, :nblock, :nblock])
    model.jordan_wigner(forward=False)
    single_ops = {
        "Cdu": model.Cdu,
        "Cdd": model.Cdd,
        "Cu": model.Cu,
        "Cd": model.Cd,
    }
    table = abelian_narg.make_operator_table(single_ops, abelian_narg.OPERATOR_PATTERNS, nblock)
    qn = abelian_narg.primitive_charge_labels(nblock)
    _, triple_terms = abelian_narg.precompute_integral_terms(eri, cutoff=0.0, use_numba=False)
    dense_res = abelian_narg.build_initial_triple_residuals(
        single_ops,
        nblock,
        range(nblock, total_sites),
        triple_terms,
    )
    irrep_table = abelian_narg.irrep_operator_table(table, qn)
    irrep_res = abelian_narg.build_initial_triple_residuals_irrep(
        irrep_table,
        nblock,
        range(nblock, total_sites),
        triple_terms,
    )

    for q, (dense_u, dense_d) in dense_res.items():
        irrep_u, irrep_d = irrep_res[q]
        np.testing.assert_allclose(abelian_narg.labeled_dense(irrep_u, qn), dense_u.toarray(), atol=1e-12)
        np.testing.assert_allclose(abelian_narg.labeled_dense(irrep_d, qn), dense_d.toarray(), atol=1e-12)

    n = len(qn)
    d = len(abelian_narg.LOCAL_QN)
    U = np.zeros((n, n, d))
    for local_id in range(d):
        U[:, :, local_id] = np.eye(n)
    output_qn = abelian_narg.branch_qn(qn)
    dense_next = abelian_narg.extend_triple_residuals(
        dense_res,
        table,
        U,
        new_site,
        total_sites,
        triple_terms,
        output_qn=output_qn,
        use_irrep_blocks=False,
    )
    irrep_next = abelian_narg.extend_triple_residuals_irrep(
        irrep_res,
        irrep_table,
        U,
        qn,
        new_site,
        total_sites,
        triple_terms,
        output_qn,
    )

    for q, (dense_u, dense_d) in dense_next.items():
        irrep_u, irrep_d = irrep_next[q]
        np.testing.assert_allclose(abelian_narg.labeled_dense(irrep_u, output_qn), dense_u, atol=1e-12)
        np.testing.assert_allclose(abelian_narg.labeled_dense(irrep_d, output_qn), dense_d, atol=1e-12)


def test_qchem_narg_prepares_frozen_core_cas_like_casci():
    mf = _lih_sto3g_mf()
    h1e, eri, active_mol, active_space = prepare_active_space(
        mf,
        mf.mol,
        ncas=4,
        nelecas=2,
    )
    mc = CASCI(mf, ncas=4, nelecas=2).run(nstates=1, use_cholesky=True)

    assert h1e.shape == (4, 4)
    assert eri.shape == (4, 4, 4, 4)
    assert active_mol.nelec == (1, 1)
    assert active_space.ncore == 1
    np.testing.assert_allclose(active_mol.energy_nuc(), mc.e_core, atol=1e-10)


def test_h2_public_active_space_api_matches_casci():
    mf = _h2_sto3g_mf()
    mc = CASCI(mf, ncas=2, nelecas=(1, 1), verbose=0).run(nstates=1, method="direct_ci")

    number = mf.NARG(
        symmetry="number",
        active=[0, 1],
        nelecas=(1, 1),
        D=16,
        nstates=1,
    )
    number.run()

    spin = mf.NARG(
        symmetry="spin",
        active=[0, 1],
        nelecas=(1, 1),
        D=16,
        nstates=1,
        target_j2=0,
        su2_backend="python",
    )
    spin.run()

    assert number.workflow["orbital_space"] == (0, 1)
    assert number.ncore == 0
    assert number.ncas == 2
    assert number.local_dims == (4, 4)
    assert spin.workflow["orbital_space"] == (0, 1)
    assert spin.target_irrep == (2, 0)
    np.testing.assert_allclose(number.e_tot[0], mc.e_tot[0], atol=1.0e-10)
    np.testing.assert_allclose(spin.e_tot[0], mc.e_tot[0], atol=1.0e-10)


def test_abelian_narg_accepts_cas_options():
    mf = _lih_sto3g_mf()
    narg = NARG(
        mf,
        symmetry="abelian",
        ncas=4,
        nelecas=2,
        D=8,
        nstates=2,
        store_tensors=False,
    )

    e, x = narg.run()

    assert narg.ncas == 4
    assert narg.nelecas == 2
    assert narg.ncore == 1
    assert narg.n0 == 3
    assert narg.local_dims == (4, 4, 4, 4)
    assert len(e) == 2
    assert x.shape[1] == 2
    assert np.all(np.isfinite(e))


def test_abelian_narg_exposes_e_tot_and_rdms_directly():
    mf = _lih_sto3g_mf()
    narg = NARG(
        mf,
        symmetry="abelian",
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
    )

    narg.run()
    dm1 = narg.make_rdm1()
    dm2 = narg.make_rdm2()
    e_active = (
        np.einsum("pq,pq", narg.h1e, dm1)
        + 0.5 * np.einsum("pqrs,pqrs", narg.eri, dm2)
    )

    assert narg.e_tot.shape == (1,)
    assert dm1.shape == (2, 2)
    assert dm2.shape == (2, 2, 2, 2)
    assert narg.rdm_backend == "tensor"
    np.testing.assert_allclose(np.trace(dm1), 2.0, atol=1.0e-10)
    np.testing.assert_allclose(e_active + narg.e_core, narg.e_tot[0], atol=1.0e-10)


def test_abelian_standard_narg_tensor_rdms_match_dense_reference():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        @staticmethod
        def energy_nuc():
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    solver = abelian_narg.NARG(
        object(),
        mol=DummyMol(),
        D=8,
        n0=1,
        nstates=1,
        growth_sites=1,
    )
    solver.run(h1e=h1e, eri=eri)

    psi = narg_state_vector(solver.tensors[:-1], solver.tensors[-1], root=0)
    dm1 = solver.make_rdm1()
    dm2 = solver.make_rdm2()

    assert solver.rdm_backend == "tensor"
    assert solver.rdm_implicit_factor_count == 4
    assert solver.rdm_factor_storage_bytes > 0
    assert solver.rdm_environment_storage_bytes > 0
    assert max(tensor.shape[0] for tensor in solver.tensors[1:-1]) >= 4 * 8
    np.testing.assert_allclose(
        dm1, spin_traced_rdm1_from_state(psi, 6), atol=1.0e-10
    )
    np.testing.assert_allclose(
        dm2, spin_traced_rdm2_from_state(psi, 6), atol=1.0e-10
    )


def test_abelian_narg_rdm_requires_stored_tensors():
    mf = _lih_sto3g_mf()
    narg = NARG(
        mf,
        symmetry="abelian",
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        store_tensors=False,
    )

    narg.run()
    try:
        narg.make_rdm1()
    except ValueError as exc:
        assert "store_tensors=True" in str(exc)
    else:
        raise AssertionError("make_rdm1 should require stored NARG tensors")


def test_su2_narg_exposes_e_tot_and_rdms_directly():
    mf = _lih_sto3g_mf()
    narg = NARG(
        mf,
        symmetry="su2",
        ncas=2,
        nelecas=2,
        D=8,
        nstates=1,
        su2_backend="python",
    )

    narg.run()
    dm1 = narg.make_rdm1()
    dm2 = narg.make_rdm2()

    assert narg.e_tot.shape == (1,)
    assert dm1.shape == (2, 2)
    assert dm2.shape == (2, 2, 2, 2)
    np.testing.assert_allclose(np.trace(dm1), 2.0, atol=1.0e-10)


def test_abelian_nargscf_runs_initial_cas_evaluation():
    mf = _lih_sto3g_mf()
    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        symmetry="abelian",
        D=8,
        nstates=1,
        max_cycle=0,
        store_tensors=False,
    ).run()

    assert mc.narg is not None
    assert mc.e_tot.shape == (1,)
    np.testing.assert_allclose(mc.e_tot, mc.narg.e_tot)


def test_su2_nargscf_runs_initial_cas_evaluation():
    mf = _lih_sto3g_mf()
    mc = NARGSCF(
        mf,
        ncas=2,
        nelecas=2,
        symmetry="su2",
        D=8,
        nstates=1,
        su2_backend="python",
        max_cycle=0,
    ).run()

    assert mc.narg is not None
    assert mc.e_tot.shape == (1,)
    np.testing.assert_allclose(mc.e_tot, mc.narg.e_tot)


def test_public_qchem_exports_nargscf():
    from pyqed.qchem import NARGSCF as PublicNARGSCF

    assert PublicNARGSCF is NARGSCF


def test_abelian_narg_two_site_growth_emits_two_site_tensor():
    class DummyMol:
        nelec = (1, 1)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e = np.diag(np.linspace(-0.4, 0.4, 5))
    eri = np.zeros((5, 5, 5, 5))
    abelian_narg.mol = DummyMol()

    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=4,
        n0=2,
        nstates=1,
        growth_sites=2,
        return_tensors=True,
        return_tensor_qns=True,
    )

    assert np.all(np.isfinite(e))
    assert x.shape[1] == 1
    assert any(tensor.ndim == 4 for tensor in tensors[:-1])
    assert any(factor.get("growth_sites") == 2 for factor in tensor_qns["factors"])
    assert any(factor.get("two_site_mode") == "supersite" for factor in tensor_qns["factors"])


def test_abelian_narg_auto_growth_uses_two_site_for_close_mos():
    class DummyMol:
        nelec = (1, 1)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e = np.diag(np.linspace(-0.4, 0.4, 5))
    eri = np.zeros((5, 5, 5, 5))
    abelian_narg.mol = DummyMol()

    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=4,
        n0=2,
        nstates=1,
        growth_sites="auto",
        two_site_max_dim=64,
        return_tensors=True,
        return_tensor_qns=True,
    )

    assert np.all(np.isfinite(e))
    assert x.shape[1] == 1
    assert any(tensor.ndim == 4 for tensor in tensors[:-1])
    assert any(factor.get("growth_sites") == 2 for factor in tensor_qns["factors"])


def test_abelian_narg_auto_growth_keeps_separated_mos_one_site():
    class DummyMol:
        nelec = (1, 1)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e = np.diag([0.0, 0.1, 0.2, 0.3, 10.0])
    eri = np.zeros((5, 5, 5, 5))
    abelian_narg.mol = DummyMol()

    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=4,
        n0=2,
        nstates=1,
        growth_sites="auto",
        return_tensors=True,
        return_tensor_qns=True,
    )

    assert np.all(np.isfinite(e))
    assert x.shape[1] == 1
    assert not any(tensor.ndim == 4 for tensor in tensors[:-1])
    assert not any(factor.get("growth_sites") == 2 for factor in tensor_qns["factors"])


def test_abelian_narg_fast_path_matches_default_energy():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    abelian_narg.mol = DummyMol()

    e_default = abelian_narg.kernel(
        h1e,
        eri,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0]
    e_fast = abelian_narg.kernel(
        h1e,
        eri,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
        fast=True,
    )[0]

    np.testing.assert_allclose(e_fast, e_default, atol=1e-10)


def test_sparse_operator_entries_reduce_hubbard_table():
    h1e, eri = _hubbard_integrals(8, t=0.7, u=2.0)
    pair_terms, triple_terms = abelian_narg.precompute_integral_terms(eri, cutoff=0.0, use_numba=False)

    required = abelian_narg.required_operator_entries(pair_terms, triple_terms, h1e.shape[0], nsites=4)

    full_count = sum(4 ** len(pattern) for pattern in abelian_narg.OPERATOR_PATTERNS)
    sparse_count = sum(len(required[pattern]) for pattern in abelian_narg.OPERATOR_PATTERNS)
    assert sparse_count < full_count // 4
    assert all(len(required[(name,)]) == 4 for name in ("Cu", "Cd", "Cdu", "Cdd"))


def test_mo_pair_factors_match_dense_narg_energy():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    factors = np.zeros((4, 4, 4))
    for site in range(4):
        factors[site, site, site] = np.sqrt(2.0)
    abelian_narg.mol = DummyMol()

    np.testing.assert_allclose(abelian_narg.eri_to_dense(factors), eri, atol=1.0e-12)
    dense_energy = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=2,
        nstates=1,
    )[0]
    factor_energy = abelian_narg.kernel(
        h1e,
        factors,
        D=16,
        n0=2,
        nstates=1,
    )[0]
    dense_cluster_energy = abelian_narg.reduced_supersite_kernel(
        h1e,
        eri,
        ((0, 1), (2, 3)),
        D=16,
        nstates=1,
        nelec=DummyMol.nelec,
    )[0]
    factor_cluster_energy = abelian_narg.reduced_supersite_kernel(
        h1e,
        factors,
        ((0, 1), (2, 3)),
        D=16,
        nstates=1,
        nelec=DummyMol.nelec,
    )[0]

    np.testing.assert_allclose(factor_energy, dense_energy, atol=1.0e-12)
    np.testing.assert_allclose(factor_cluster_energy, dense_cluster_energy, atol=1.0e-12)
    assert factors.nbytes < eri.nbytes


def test_cd_mean_field_stays_factorized_through_abelian_narg():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="cd",
        options={"low_rank_tol": 1.0e-12},
    )
    mf = mol.RHF().run(verbose=0)
    solver = abelian_narg.NARG(mf, mol=mol, D=4, n0=1, nstates=1)
    factor_energy = solver.run()[0]

    h1e = mf.get_hcore_mo()
    dense_eri = mf.get_eri_mo()
    abelian_narg.mol = mol
    dense_energy = abelian_narg.kernel(
        h1e,
        dense_eri,
        D=4,
        n0=1,
        nstates=1,
    )[0]

    assert solver.eri.ndim == 3
    assert solver.eri.shape[1:] == h1e.shape
    np.testing.assert_allclose(factor_energy, dense_energy, atol=1.0e-12)


def test_one_site_hamiltonian_action_matches_explicit_kronecker_matrix():
    rng = np.random.default_rng(9)
    block = rng.normal(size=(5, 5))
    block = 0.5 * (block + block.T)
    coupling = rng.normal(size=(5, 5))
    local = np.diag([0.0, 1.0, 2.0, 3.0])
    transition = abelian_narg.cdu @ abelian_narg.cd
    action = abelian_narg.OneSiteHamiltonianAction(5, 4)
    action.add(block, local, 0.7)
    action.add(coupling, transition, -0.2, hermitian=True)
    explicit = 0.7 * np.kron(block, local)
    term = -0.2 * np.kron(coupling, transition)
    explicit += term + term.T.conj()
    vectors = rng.normal(size=(20, 3))

    np.testing.assert_allclose(action.matmat(vectors), explicit @ vectors, atol=1.0e-12)
    for state in range(4):
        rows = np.arange(5) * 4 + state
        np.testing.assert_allclose(
            action.diagonal_block(state),
            explicit[np.ix_(rows, rows)],
            atol=1.0e-12,
        )


def test_abelian_narg_sparse_operator_table_matches_dense_hubbard():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    abelian_narg.mol = DummyMol()

    e_dense = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=1,
        sparse_operator_table=False,
    )[0]
    e_sparse = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=1,
        sparse_operator_table=True,
    )[0]
    e_fast = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=2,
        fast=True,
    )[0]
    e_two_dense = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=2,
        sparse_operator_table=False,
    )[0]

    np.testing.assert_allclose(e_sparse, e_dense, atol=1e-10)
    np.testing.assert_allclose(e_fast, e_two_dense, atol=1e-10)


def test_abelian_narg_sparse_operator_projection_matches_dense_hubbard():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    abelian_narg.mol = DummyMol()

    e_dense = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=2,
        sparse_operator_table=False,
    )[0]
    e_sparse_projected = abelian_narg.kernel(
        h1e,
        eri,
        D=16,
        n0=1,
        nstates=1,
        growth_sites=2,
        fast=True,
        use_sparse_operator_projection=True,
    )[0]

    np.testing.assert_allclose(e_sparse_projected, e_dense, atol=1e-10)


def test_abelian_narg_sparse_operator_table_matches_dense_qchem_like_integrals():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    rng = np.random.default_rng(42)
    nsites = 5
    h_raw = rng.normal(size=(nsites, nsites))
    h1e = 0.2 * (h_raw + h_raw.T)
    eri_raw = 0.05 * rng.normal(size=(nsites, nsites, nsites, nsites))
    eri = 0.25 * (
        eri_raw
        + eri_raw.transpose(1, 0, 3, 2)
        + eri_raw.transpose(2, 3, 0, 1)
        + eri_raw.transpose(3, 2, 1, 0)
    )
    abelian_narg.mol = DummyMol()

    e_dense = abelian_narg.kernel(
        h1e,
        eri,
        D=8,
        n0=2,
        nstates=1,
        growth_sites=1,
        sparse_operator_table=False,
    )[0]
    e_sparse = abelian_narg.kernel(
        h1e,
        eri,
        D=8,
        n0=2,
        nstates=1,
        growth_sites=1,
        sparse_operator_table=True,
    )[0]

    np.testing.assert_allclose(e_sparse, e_dense, atol=1e-10)


def test_abelian_narg_two_site_hubbard_matches_exact_without_truncation():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    exact = SpinHalfFermionChain(h1e, eri, nelec=DummyMol.nelec)
    hamiltonian = exact.jordan_wigner(forward=False).toarray()
    nu = np.rint(exact.Nu_tot.diagonal()).astype(int)
    nd = np.rint(exact.Nd_tot.diagonal()).astype(int)
    sector = np.flatnonzero((nu == DummyMol.nelec[0]) & (nd == DummyMol.nelec[1]))
    exact_e = eigsh(hamiltonian[np.ix_(sector, sector)], k=1, which="SA", return_eigenvectors=False)

    abelian_narg.mol = DummyMol()
    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=64,
        n0=1,
        nstates=1,
        growth_sites=2,
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact_e, atol=1e-10)
    assert x.shape[1] == 1
    assert any(tensor.ndim == 4 for tensor in tensors[:-1])
    assert any(factor.get("growth_sites") == 2 for factor in tensor_qns["factors"])


def test_abelian_narg_supersite_two_site_qchem_like_matches_exact_without_truncation():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    rng = np.random.default_rng(27)
    nsites = 5
    h_raw = rng.normal(size=(nsites, nsites))
    h1e = 0.2 * (h_raw + h_raw.T)
    eri_raw = 0.04 * rng.normal(size=(nsites, nsites, nsites, nsites))
    eri = 0.25 * (
        eri_raw
        + eri_raw.transpose(1, 0, 3, 2)
        + eri_raw.transpose(2, 3, 0, 1)
        + eri_raw.transpose(3, 2, 1, 0)
    )
    exact = SpinHalfFermionChain(h1e, eri, nelec=DummyMol.nelec)
    hamiltonian = exact.jordan_wigner(forward=False).toarray()
    nu = np.rint(exact.Nu_tot.diagonal()).astype(int)
    nd = np.rint(exact.Nd_tot.diagonal()).astype(int)
    sector = np.flatnonzero((nu == DummyMol.nelec[0]) & (nd == DummyMol.nelec[1]))
    exact_e = eigsh(hamiltonian[np.ix_(sector, sector)], k=1, which="SA", return_eigenvectors=False)

    abelian_narg.mol = DummyMol()
    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=256,
        n0=2,
        nstates=1,
        growth_sites=2,
        two_site_mode="supersite",
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact_e, atol=1e-10)
    assert x.shape[1] == 1
    assert any(tensor.ndim == 4 for tensor in tensors[:-1])
    assert any(factor.get("two_site_mode") == "supersite" for factor in tensor_qns["factors"])
    psi = narg_state_vector(tensors[:-1], tensors[-1], root=0)
    expectation = np.vdot(psi, hamiltonian @ psi) / np.vdot(psi, psi)
    np.testing.assert_allclose(e[0], np.real(expectation), atol=1e-10)


def test_abelian_narg_rolling_two_site_keeps_one_site_block_size():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    exact = _sector_ground_energy(h1e, eri, DummyMol.nelec)
    abelian_narg.mol = DummyMol()

    e, _x, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=64,
        n0=1,
        nstates=1,
        growth_sites=2,
        two_site_mode="two_site",
        return_tensor_qns=True,
    )

    rolling = [factor for factor in tensor_qns["factors"] if factor.get("two_site_mode") == "rolling"]
    assert rolling
    assert rolling[0]["local_dim"] == 4
    assert rolling[0]["temporary_local_dim"] == 16
    assert rolling[0]["block_dim"] == 64
    np.testing.assert_allclose(e, exact, atol=1e-10)


def test_abelian_recursive_conditional_cc_improves_truncated_hubbard_chain():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = float(_sector_ground_energy(h1e, eri, DummyMol.nelec)[0])
    abelian_narg.mol = DummyMol()

    common = dict(
        D=8,
        n0=1,
        nstates=1,
        growth_sites=2,
        two_site_mode="rolling",
    )
    plain = abelian_narg.kernel(h1e, eri, **common)[0][0]
    solver = abelian_narg.NARG(
        object(),
        mol=DummyMol(),
        **common,
        dressing="conditional_cc",
    )
    dressed, _vectors = solver.run(h1e=h1e, eri=eri)
    diagnostics = solver.dressing_history

    assert exact <= dressed[0] + 1.0e-10
    assert plain - exact > 0.3
    assert dressed[0] - exact < 0.1
    assert dressed[0] < plain - 0.2
    assert len(diagnostics) == 3
    assert diagnostics[0]["response_rank"] == 0
    assert all(item["response_rank"] > 0 for item in diagnostics[1:])
    assert all(item["discarded_residual_norm"] > 0.0 for item in diagnostics[1:])
    assert solver.tensors is None


def test_abelian_one_site_conditional_cc_projects_later_growth_operators():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = float(_sector_ground_energy(h1e, eri, DummyMol.nelec)[0])
    abelian_narg.mol = DummyMol()
    common = dict(D=8, n0=1, nstates=1, growth_sites=1)
    plain = abelian_narg.kernel(h1e, eri, **common)[0][0]

    solver = abelian_narg.NARG(
        object(),
        mol=DummyMol(),
        **common,
        dressing="conditional_cc",
    )
    dressed, _vectors = solver.run(h1e=h1e, eri=eri)

    assert exact <= dressed[0] + 1.0e-10
    assert plain - exact > 0.5
    assert dressed[0] - exact < 0.16
    assert dressed[0] < plain - 0.4
    assert len(solver.dressing_history) == 5
    assert solver.dressing_history[0]["response_rank"] == 0
    assert all(item["response_rank"] > 0 for item in solver.dressing_history[1:])
    assert solver.tensors is not None
    _assert_stored_state_energy(solver, h1e, eri, dressed[0])


def test_abelian_detached_frames_improve_same_dimension_hubbard_chain():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = float(_sector_ground_energy(h1e, eri, DummyMol.nelec)[0])
    abelian_narg.mol = DummyMol()

    plain = abelian_narg.kernel(
        h1e,
        eri,
        D=2,
        n0=2,
        nstates=1,
        growth_sites=1,
    )[0][0]
    solver = abelian_narg.NARG(
        object(),
        mol=DummyMol(),
        D=2,
        chi=12,
        n0=2,
        nstates=1,
        growth_sites=1,
        dressing="detached_frames",
    )
    detached = solver.run(h1e=h1e, eri=eri)[0][0]

    assert exact <= detached + 1.0e-10
    assert detached < plain - 0.2
    assert len(solver.detached_history) == 4
    assert solver.detached_history[0]["branch_ranks"] == (2, 2, 2, 2)
    assert all(
        item["anchor_rank"] == sum(item["branch_ranks"])
        for item in solver.detached_history
    )
    assert all(
        item["detached_dim"] == 4 * item["frame_rank"]
        for item in solver.detached_history
    )
    assert all(item["retained_dim"] <= 12 for item in solver.detached_history)
    assert all(item["orthogonality_error"] < 1.0e-12 for item in solver.detached_history)
    assert all(item["anchor_inclusion_error"] < 1.0e-12 for item in solver.detached_history)
    assert all(item["retained_anchor_error"] < 1.0e-12 for item in solver.detached_history)
    assert all(item["detached_improvement"] >= -1.0e-12 for item in solver.detached_history)
    assert solver.tensors is not None
    _assert_stored_state_energy(solver, h1e, eri, detached)


def test_abelian_detached_frames_adapt_rank_from_projected_residual():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    abelian_narg.mol = DummyMol()
    common = dict(
        mol=DummyMol(),
        D=2,
        chi=12,
        n0=2,
        nstates=1,
        growth_sites=1,
        dressing="detached_frames",
    )
    fixed = abelian_narg.NARG(object(), **common)
    fixed_energy = fixed.run(h1e=h1e, eri=eri)[0][0]
    adaptive = abelian_narg.NARG(
        object(),
        **common,
        frame_adapt_tol=0.1,
        frame_max_dim=12,
        frame_expand_dim=1,
    )
    adaptive_energy = adaptive.run(h1e=h1e, eri=eri)[0][0]

    assert adaptive_energy < fixed_energy - 0.01
    assert any(item["adapted_rank"] > 0 for item in adaptive.detached_history)
    assert all(
        item["frame_residual_norm"] <= 0.1 + 1.0e-12
        or item["frame_rank"] == 12
        for item in adaptive.detached_history
    )
    assert all(
        item["frame_residual_history"][-1]
        <= item["frame_residual_history"][0] + 1.0e-12
        for item in adaptive.detached_history
    )


def test_abelian_detached_cc_dresses_detached_projector_at_every_step():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = float(_sector_ground_energy(h1e, eri, DummyMol.nelec)[0])
    abelian_narg.mol = DummyMol()
    common = dict(
        mol=DummyMol(),
        D=2,
        chi=12,
        n0=2,
        nstates=1,
        growth_sites=1,
    )
    detached = abelian_narg.NARG(
        object(), **common, dressing="detached_frames"
    )
    detached_energy = detached.run(h1e=h1e, eri=eri)[0][0]
    combined = abelian_narg.NARG(
        object(), **common, dressing="detached+cc"
    )
    combined_energy = combined.run(h1e=h1e, eri=eri)[0][0]

    assert exact <= combined_energy + 1.0e-10
    assert combined_energy < detached_energy - 0.4
    assert len(combined.detached_history) == 4
    assert len(combined.dressing_history) == 4
    assert all(item["response_rank"] > 0 for item in combined.dressing_history)
    assert all(
        item["maximum_sector_leakage"] < 1.0e-8
        for item in combined.detached_history
    )
    assert all(
        item["sector_label_corrections"] >= 0
        for item in combined.detached_history
    )
    _assert_stored_state_energy(combined, h1e, eri, combined_energy)
    psi = narg_state_vector(combined.tensors[:-1], combined.tensors[-1], root=0)
    dm1 = combined.make_rdm1()
    dm2 = combined.make_rdm2()
    assert combined.rdm_backend == "tensor"
    np.testing.assert_allclose(
        dm1, spin_traced_rdm1_from_state(psi, 6), atol=1.0e-10
    )
    np.testing.assert_allclose(
        dm2, spin_traced_rdm2_from_state(psi, 6), atol=1.0e-10
    )
    np.testing.assert_allclose(np.trace(dm1), 6.0, atol=1.0e-10)
    np.testing.assert_allclose(
        np.einsum("pprr", dm2), 30.0, atol=1.0e-10
    )


def test_conditional_cc_uses_iterative_discarded_space_for_large_sector():
    rng = np.random.default_rng(91)
    old_dim = 20
    local_dim = 4
    full_dim = old_dim * local_dim
    matrix = rng.normal(size=(full_dim, full_dim))
    matrix = 0.5 * (matrix + matrix.T)
    h_lloo = matrix.reshape(
        old_dim, local_dim, old_dim, local_dim
    ).transpose(1, 3, 0, 2)
    trial = rng.normal(size=(full_dim, 4))
    projector, _upper = np.linalg.qr(trial, mode="reduced")
    projector = projector.reshape(old_dim, local_dim, 4)
    primitive_qn = np.zeros((full_dim, 2), dtype=int)
    output_qn = np.zeros((4, 2), dtype=int)
    plain = projector.reshape(full_dim, 4)
    plain_h = plain.conj().T @ matrix @ plain

    dressed_h, dressed, diagnostics = (
        abelian_narg.conditional_cc_transition_projector(
            h_lloo,
            primitive_qn,
            projector,
            output_qn,
            level_shift=20.0,
            response_tol=1.0e-9,
        )
    )

    dressed_flat = dressed.reshape(full_dim, 4)
    np.testing.assert_allclose(
        dressed_flat.conj().T @ dressed_flat,
        np.eye(4),
        atol=1.0e-9,
    )
    assert np.trace(dressed_h).real <= np.trace(plain_h).real + 1.0e-10
    assert diagnostics["iterative_solves"] == 4
    assert diagnostics["iterative_fallbacks"] == 0
    assert diagnostics["maximum_response_residual"] < 1.0e-8


def test_abelian_narg_supersite_hubbard_uses_literal_d16_sites():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    exact = _sector_ground_energy(h1e, eri, DummyMol.nelec)
    abelian_narg.mol = DummyMol()

    e, x, tensors, tensor_qns = abelian_narg.supersite_kernel(
        h1e,
        eri,
        groups=[(0, 1), (2, 3)],
        D=36,
        nstates=1,
        nelec=DummyMol.nelec,
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact, atol=1e-10)
    assert x.shape == (4 ** 4, 1)
    assert [tensor.shape[-1] for tensor in tensors] == [16, 16]
    assert [factor["local_dim"] for factor in tensor_qns["factors"]] == [16, 16]
    assert [factor["growth_sites"] for factor in tensor_qns["factors"]] == [2, 2]


def test_hierarchical_narg_hubbard_matches_exact_without_truncation():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(4, t=0.7, u=2.0)
    exact = _sector_ground_energy(h1e, eri, DummyMol.nelec)
    abelian_narg.mol = DummyMol()

    e, x, tree = abelian_narg.hierarchical_kernel(
        h1e,
        eri,
        D=256,
        leaf_size=2,
        nstates=1,
        nelec=DummyMol.nelec,
        return_tree=True,
    )

    np.testing.assert_allclose(e, exact, atol=1e-10)
    assert x.shape == (4 ** 4, 1)
    assert tree["levels"][0][0].orbitals == (0, 1)
    assert tree["levels"][0][1].orbitals == (2, 3)


def test_hierarchical_narg_uses_compressed_block_fusion():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    abelian_narg.mol = DummyMol()

    e, x, tree = abelian_narg.hierarchical_kernel(
        h1e,
        eri,
        D=16,
        leaf_size=2,
        nstates=1,
        nelec=DummyMol.nelec,
        max_state_orbitals=4,
        return_tree=True,
    )

    assert e.shape == (1,)
    assert x.shape == (tree["root"].h.shape[0], 1)
    assert tree["method"] == "hierarchical_narg"
    assert not tree["basis_is_primitive"]


def test_abelian_narg_two_site_longer_hubbard_matches_exact_without_truncation():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = _sector_ground_energy(h1e, eri, DummyMol.nelec)

    abelian_narg.mol = DummyMol()
    e, x, tensors, tensor_qns = abelian_narg.kernel(
        h1e,
        eri,
        D=400,
        n0=2,
        nstates=1,
        growth_sites=2,
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact, atol=1e-10)
    assert x.shape[1] == 1
    assert sum(tensor.ndim == 4 for tensor in tensors[:-1]) == 1
    assert [factor.get("growth_sites") for factor in tensor_qns["factors"]] == [1, 2, 1]


def test_abelian_narg_small_d_two_site_hubbard_improves_one_site():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h1e, eri = _hubbard_integrals(6, t=0.7, u=2.0)
    exact = _sector_ground_energy(h1e, eri, DummyMol.nelec)[0]
    abelian_narg.mol = DummyMol()

    e_one = abelian_narg.kernel(
        h1e,
        eri,
        D=12,
        n0=2,
        nstates=1,
        growth_sites=1,
    )[0][0]
    e_two = abelian_narg.kernel(
        h1e,
        eri,
        D=12,
        n0=2,
        nstates=1,
        growth_sites=2,
    )[0][0]

    assert exact <= e_two <= e_one - 1e-4


def test_momentum_space_hubbard_integrals_match_real_space_exact_energy():
    nelec = (3, 3)
    h_real, eri_real = _periodic_hubbard_integrals(6, t=0.7, u=0.5)
    h_mom, eri_mom = _real_momentum_hubbard_integrals(6, t=0.7, u=0.5)

    np.testing.assert_allclose(h_mom, np.diag(np.diag(h_mom)), atol=1e-12)
    np.testing.assert_allclose(
        _sector_ground_energy(h_mom, eri_mom, nelec),
        _sector_ground_energy(h_real, eri_real, nelec),
        atol=1e-10,
    )


def test_energy_groups_keep_nondegenerate_hubbard_band_edges_single():
    h_mom, _ = _real_momentum_hubbard_integrals(8, t=0.7, u=0.5, order="energy")
    eps = np.diag(h_mom)

    groups = abelian_narg.energy_groups(eps, tol=1e-10)

    assert groups == ((0,), (1, 2), (3, 4), (5, 6), (7,))


def test_supersite_kernel_accepts_energy_groups_with_d4_edges():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h_mom, eri_mom = _real_momentum_hubbard_integrals(4, t=0.7, u=1.0, order="energy")
    groups = abelian_narg.energy_groups(np.diag(h_mom), tol=1e-10)
    exact = _sector_ground_energy(h_mom, eri_mom, DummyMol.nelec)
    abelian_narg.mol = DummyMol()

    e, _, tensors, tensor_qns = abelian_narg.supersite_kernel(
        h_mom,
        eri_mom,
        groups=groups,
        D=64,
        nstates=1,
        nelec=DummyMol.nelec,
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact, atol=1e-10)
    assert groups == ((0,), (1, 2), (3,))
    assert [factor["local_dim"] for factor in tensor_qns["factors"]] == [4, 16, 4]
    assert [tensor.shape[-1] for tensor in tensors] == [4, 16, 4]


def test_reduced_supersite_kernel_keeps_mixed_energy_groups_exact():
    class DummyMol:
        nelec = (2, 2)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h_mom, eri_mom = _real_momentum_hubbard_integrals(4, t=0.7, u=1.0, order="energy")
    groups = abelian_narg.energy_groups(np.diag(h_mom), tol=1e-10)
    exact = _sector_ground_energy(h_mom, eri_mom, DummyMol.nelec)
    abelian_narg.mol = DummyMol()

    e, _, tensors, tensor_qns = abelian_narg.reduced_supersite_kernel(
        h_mom,
        eri_mom,
        groups=groups,
        D=16,
        nstates=1,
        nelec=DummyMol.nelec,
        return_tensors=True,
        return_tensor_qns=True,
    )

    np.testing.assert_allclose(e, exact, atol=1e-10)
    assert groups == ((0,), (1, 2), (3,))
    assert tensor_qns["path"] == "reduced"
    assert [factor["path"] for factor in tensor_qns["factors"]] == ["reduced", "reduced", "reduced"]
    assert [factor["local_dim"] for factor in tensor_qns["factors"]] == [4, 16, 4]
    assert [tuple(tensor.shape[2:]) for tensor in tensors[:-1]] == [(1,), (4, 4), (4,)]


def test_momentum_space_hubbard_two_site_narg_is_compact_at_weak_coupling():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h_real, eri_real = _periodic_hubbard_integrals(6, t=0.7, u=0.5)
    h_mom, eri_mom = _real_momentum_hubbard_integrals(6, t=0.7, u=0.5)
    abelian_narg.mol = DummyMol()

    e_real_two = abelian_narg.kernel(
        h_real,
        eri_real,
        D=10,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]
    e_mom_one = abelian_narg.kernel(
        h_mom,
        eri_mom,
        D=10,
        n0=1,
        nstates=1,
        growth_sites=1,
    )[0][0]
    e_mom_two = abelian_narg.kernel(
        h_mom,
        eri_mom,
        D=10,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]

    assert e_mom_two <= e_mom_one - 1e-3
    assert e_mom_two <= e_real_two - 0.02


def test_fermi_ordered_momentum_hubbard_narg_improves_energy_ordering():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h_energy, eri_energy = _real_momentum_hubbard_integrals(6, t=0.7, u=1.0, order="energy")
    h_fermi, eri_fermi = _real_momentum_hubbard_integrals(
        6,
        t=0.7,
        u=1.0,
        order="fermi",
        nelec=DummyMol.nelec,
    )
    abelian_narg.mol = DummyMol()

    e_energy = abelian_narg.kernel(
        h_energy,
        eri_energy,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]
    e_fermi = abelian_narg.kernel(
        h_fermi,
        eri_fermi,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]

    assert e_fermi <= e_energy - 0.02


def test_particle_hole_momentum_ordering_remains_close_to_fermi_ordering():
    class DummyMol:
        nelec = (3, 3)
        spin = 0

        def energy_nuc(self):
            return 0.0

    h_fermi, eri_fermi = _real_momentum_hubbard_integrals(
        6,
        t=0.7,
        u=0.5,
        order="fermi",
        nelec=DummyMol.nelec,
    )
    h_ph, eri_ph = _real_momentum_hubbard_integrals(
        6,
        t=0.7,
        u=0.5,
        order="particle_hole",
        nelec=DummyMol.nelec,
    )
    abelian_narg.mol = DummyMol()

    e_fermi = abelian_narg.kernel(
        h_fermi,
        eri_fermi,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]
    e_ph = abelian_narg.kernel(
        h_ph,
        eri_ph,
        D=12,
        n0=1,
        nstates=1,
        growth_sites="auto",
        two_site_energy_tol=1e-10,
    )[0][0]

    assert abs(e_ph - e_fermi) <= 5e-3
