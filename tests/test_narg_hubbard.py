import numpy as np
import pytest

from pyqed.narg import IntegralHamiltonian, MPOHamiltonian, NARG
from pyqed.narg.hubbard import Hubbard, chain_bonds, hubbard_integrals
from pyqed.narg.qchem.abelian import primitive_charge_labels
from pyqed.mps.fermion import SpinHalfFermionChain


def _exact_hubbard_sector_ground(h1e, eri, nelec):
    ham = SpinHalfFermionChain(h1e, eri).jordan_wigner()
    dense = ham.toarray() if hasattr(ham, "toarray") else np.asarray(ham)
    labels = primitive_charge_labels(h1e.shape[0])
    target = (int(sum(nelec)), int(nelec[0] - nelec[1]))
    rows = np.flatnonzero((labels[:, 0] == target[0]) & (labels[:, 1] == target[1]))
    values = np.linalg.eigvalsh(dense[np.ix_(rows, rows)])
    return float(values[0])


def test_hubbard_integrals_open_and_periodic_chain():
    assert chain_bonds(4) == ((0, 1), (1, 2), (2, 3))
    assert chain_bonds(4, periodic=True) == ((0, 1), (1, 2), (2, 3), (3, 0))

    h1e, eri = hubbard_integrals(4, t=0.7, U=2.0, bonds=chain_bonds(4))

    assert h1e.shape == (4, 4)
    assert eri.shape == (4, 4, 4, 4)
    np.testing.assert_allclose([eri[i, i, i, i] for i in range(4)], 2.0)
    assert eri[0, 0, 1, 1] == 0.0
    assert h1e[0, 1] == h1e[1, 0] == -0.7
    assert h1e[0, 3] == h1e[3, 0] == 0.0


def test_hubbard_instance_narg_runs_su2_dimer():
    model = Hubbard(nsites=2, nelec=(1, 1), t=0.7, U=2.0)

    solver = model.NARG(D=8, nstates=1, su2_backend="python")

    exact = 0.5 * model.U - 0.5 * np.sqrt(model.U**2 + 16.0 * model.t**2)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.target_irrep == (2, 0)
    np.testing.assert_allclose(solver.h1e, model.h1e)
    np.testing.assert_allclose(solver.eri, model.eri)


def test_hubbard_H_returns_integral_hamiltonian_for_narg_dispatch():
    hub = Hubbard(nsites=2, nelec=(1, 1), t=0.7, U=2.0)

    H = hub.H(symmetry="spin")
    solver = NARG(H, D=8, nstates=1, su2_backend="python").run()

    assert isinstance(H, IntegralHamiltonian)
    assert H.basis == "site"
    assert H.symmetry == "spin"
    assert solver.target_irrep == (2, 0)
    np.testing.assert_allclose(solver.e_tot[0], hub.NARG(D=8, nstates=1, su2_backend="python").e_tot[0])


def test_hubbard_H_accepts_public_symmetry_name():
    hub = Hubbard(nsites=2, nelec=(1, 1), t=0.7, U=2.0)

    assert hub.H(symmetry="number").symmetry == "number"
    assert hub.H(symmetry="spin").symmetry == "spin"
    assert hub.NARG(symmetry="number", D=8).__class__.__name__ == "HubbardMPONARG"

    with pytest.raises(TypeError, match="blocks"):
        hub.H(blocks="spin")

    with pytest.raises(TypeError, match="blocks"):
        hub.NARG(blocks="spin")


def test_hubbard_H_supports_momentum_integral_basis():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=1.0, U=0.0, periodic=True)

    H = hub.H(basis="momentum", symmetry="number", order="energy")

    assert isinstance(H, IntegralHamiltonian)
    assert H.basis == "momentum"
    assert H.symmetry == "number"
    np.testing.assert_allclose(np.diag(H.h1e), [-2.0, 0.0, 0.0, 2.0], atol=1.0e-12)
    np.testing.assert_allclose(H.h1e - np.diag(np.diag(H.h1e)), 0.0, atol=1.0e-12)


def test_hubbard_H_orbital_blocks_run_clustered_abelian_narg():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    H = hub.H(symmetry="number", orbital_blocks=[(0, 1), (2, 3)])
    solver = NARG(H, D=36, nstates=1)

    solver.run()

    exact = _exact_hubbard_sector_ground(H.h1e, H.eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.orbital_blocks == ((0, 1), (2, 3))
    assert solver.local_dims == (16, 16)
    assert [factor["growth_sites"] for factor in solver.tensor_qns["factors"]] == [2, 2]


def test_direct_mf_orbital_blocks_dispatch_to_clustered_abelian_narg():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    mf, mol, h1e, eri = hub.qchem_inputs()
    solver = NARG(
        mf,
        mol=mol,
        h1e=h1e,
        eri=eri,
        orbital_blocks=[(0, 1), (2, 3)],
        D=36,
        nstates=1,
    )

    solver.run()

    exact = _exact_hubbard_sector_ground(h1e, eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.orbital_blocks == ((0, 1), (2, 3))


def test_high_level_narg_active_clusters_hide_cas_plumbing():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    mf, mol, h1e, eri = hub.qchem_inputs()

    solver = NARG(
        mf,
        mol=mol,
        h1e=h1e,
        eri=eri,
        symmetry="number",
        active=[0, 1, 2, 3],
        nelecas=hub.nelec,
        clusters=[(0, 1), (2, 3)],
        D=36,
        nstates=1,
    )
    solver.run()

    exact = _exact_hubbard_sector_ground(h1e, eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.orbital_blocks == ((0, 1), (2, 3))
    assert solver.local_dims == (16, 16)
    assert solver.workflow["active"] == (0, 1, 2, 3)
    assert solver.workflow["orbital_space"] == (0, 1, 2, 3)
    assert solver.workflow["ncore"] == 0
    assert solver.workflow["clusters"] == ((0, 1), (2, 3))


def test_high_level_narg_auto_clusters_build_orbital_blocks():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    mf, mol, h1e, eri = hub.qchem_inputs()

    solver = NARG(
        mf,
        mol=mol,
        h1e=h1e,
        eri=eri,
        symmetry="number",
        active=[0, 1, 2, 3],
        nelecas=hub.nelec,
        clusters="auto",
        cluster_weights="integral",
        cluster_max_size=2,
        D=256,
        nstates=1,
    )
    solver.run()

    flat = sorted(i for block in solver.orbital_blocks for i in block)
    exact = _exact_hubbard_sector_ground(h1e, eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert flat == [0, 1, 2, 3]
    assert max(len(block) for block in solver.orbital_blocks) <= 2
    assert solver.workflow["cluster_info"]["method"] == "spectral"


def test_blocks_keyword_was_removed_from_public_narg_api():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    mf, mol, h1e, eri = hub.qchem_inputs()

    with pytest.raises(TypeError, match="blocks=.*removed"):
        NARG(
            mf,
            mol=mol,
            h1e=h1e,
            eri=eri,
            symmetry="number",
            blocks="number",
            D=16,
            nstates=1,
        )


def test_clustered_su2_narg_two_orbital_supersites_match_exact():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    H = hub.H(symmetry="spin", orbital_blocks=[(0, 1), (2, 3)])
    solver = NARG(H, D=64, nstates=1, su2_backend="python").run()

    exact = _exact_hubbard_sector_ground(H.h1e, H.eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.orbital_blocks == ((0, 1), (2, 3))
    assert solver.local_dims == (16, 16)
    assert solver.timings["cluster_boundaries"] == (2, 4)
    assert solver.timings["exact_internal_sizes"] == (3,)
    assert solver.timings["project_v1_packages"] is False


def test_public_spin_cluster_workflow_runs_two_orbital_su2_supersites():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    mf, mol, h1e, eri = hub.qchem_inputs()
    solver = NARG(
        mf,
        mol=mol,
        h1e=h1e,
        eri=eri,
        symmetry="spin",
        active=[0, 1, 2, 3],
        nelecas=hub.nelec,
        clusters=[(0, 1), (2, 3)],
        D=64,
        nstates=1,
        su2_backend="python",
    ).run()

    exact = _exact_hubbard_sector_ground(h1e, eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.workflow["clusters"] == ((0, 1), (2, 3))
    assert solver.orbital_blocks == ((0, 1), (2, 3))


def test_clustered_su2_narg_reorders_noncontiguous_pairs_and_rdms():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    H = hub.H(symmetry="spin", orbital_blocks=[(0, 2), (1, 3)])
    solver = NARG(H, D=64, nstates=1, su2_backend="python").run()

    exact = _exact_hubbard_sector_ground(H.h1e, H.eri, hub.nelec)
    dm1 = solver.make_rdm1()
    dm2 = solver.make_rdm2()
    electronic = (
        np.einsum("pq,pq", H.h1e, dm1)
        + 0.5 * np.einsum("pqrs,pqrs", H.eri, dm2)
    )
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    np.testing.assert_allclose(electronic, exact, atol=1.0e-9)
    assert solver.orbital_order == (0, 2, 1, 3)


@pytest.mark.parametrize(
    "blocks,boundaries,local_dims,internal_sizes",
    [
        ([(0, 1), (2,), (3,)], (2, 3, 4), (16, 4, 4), ()),
        ([(0, 1, 2), (3,)], (3, 4), (64, 4), (2,)),
        ([(0,), (1, 2), (3,)], (3, 4), (4, 16, 4), (2,)),
    ],
)
def test_clustered_su2_narg_supports_variable_supersites(
    blocks,
    boundaries,
    local_dims,
    internal_sizes,
):
    hub = Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0)
    H = hub.H(symmetry="spin", orbital_blocks=blocks)
    solver = NARG(H, D=64, nstates=1, su2_backend="python").run()

    exact = _exact_hubbard_sector_ground(H.h1e, H.eri, hub.nelec)
    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert solver.orbital_blocks == tuple(tuple(block) for block in blocks)
    assert solver.local_dims == local_dims
    assert solver.timings["cluster_boundaries"] == boundaries
    assert solver.timings["exact_internal_sizes"] == internal_sizes


def test_hubbard_mpo_narg_matches_integral_backend_and_keeps_chain_frontier():
    hub = Hubbard(nsites=4, nelec=(2, 2), t=1.0, U=4.0)

    H = hub.H(symmetry="number", form="mpo")
    mpo = NARG(H, D=16, n0=2, nstates=1)
    mpo.run()
    integral = hub.NARG(
        form="integrals",
        symmetry="number",
        D=16,
        n0=2,
        nstates=1,
    )

    assert isinstance(H, MPOHamiltonian)
    assert H.form == "mpo"
    np.testing.assert_allclose(mpo.e_tot, integral.e_tot, atol=1.0e-12)
    assert max(step["frontier_width"] for step in mpo.history) == 1
    assert max(step["environment_matrices"] for step in mpo.history) == 3


def test_hubbard_detached_mpo_narg_matches_integral_backend():
    hub = Hubbard(nsites=6, nelec=(3, 3), t=0.7, U=2.0)
    options = dict(
        symmetry="number",
        D=2,
        n0=2,
        nstates=1,
        dressing="detached_frames",
        chi=12,
    )

    mpo = hub.NARG(form="mpo", **options)
    integral = hub.NARG(form="integrals", **options)

    np.testing.assert_allclose(mpo.e_tot, integral.e_tot, atol=1.0e-12)
    assert len(mpo.detached_history) == 4


@pytest.mark.parametrize(
    "hub",
    [
        Hubbard(nsites=4, nelec=(2, 2), t=0.7, U=2.0, periodic=True),
        Hubbard(lx=2, ly=2, nelec=(2, 2), t=0.7, U=2.0),
    ],
)
def test_hubbard_mpo_frontier_is_exact_for_periodic_and_square_graphs(hub):
    solver = hub.NARG(
        form="mpo",
        symmetry="number",
        D=16,
        n0=2,
        nstates=1,
    )
    h1e, eri = hub.integrals()
    exact = _exact_hubbard_sector_ground(h1e, eri, hub.nelec)

    np.testing.assert_allclose(solver.e_tot[0], exact, atol=1.0e-10)
    assert max(step["frontier_width"] for step in solver.history) >= 1


def test_hubbard_narg_is_instance_api_only():
    with pytest.raises(TypeError):
        Hubbard.NARG(D=4)
