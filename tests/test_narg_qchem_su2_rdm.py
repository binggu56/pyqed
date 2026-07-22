import numpy as np
import pytest
from scipy.linalg import eigh
from types import SimpleNamespace

from pyqed.narg.qchem.su2 import NARG
from pyqed.narg.qchem.su2_chain import _direct_ci_binary, _sigma_compact_columns
from pyqed.qchem.mcscf.casci import CASCI, make_rdm1, make_rdm2


class DummyMol:
    def __init__(self, nelec, spin):
        self.nelec = tuple(nelec)
        self.spin = int(spin)

    def energy_nuc(self):
        return 0.0


class DummyMF:
    def __init__(self, h1e, eri, *, nelec: int, spin: int):
        self._h1e = np.asarray(h1e)
        self.eri = np.asarray(eri)
        self.nmo = int(self._h1e.shape[0])
        self.mo_coeff = np.eye(self.nmo)
        self.nelec = int(nelec)
        self.mol = DummyMol((nelec // 2, nelec // 2), spin=spin)

    def get_hcore(self):
        return self._h1e

    def energy_nuc(self):
        return 0.0


def _physical_random_integrals(nsites: int, *, seed: int):
    rng = np.random.default_rng(seed)
    h1e = rng.normal(scale=0.3, size=(nsites, nsites))
    h1e = 0.5 * (h1e + h1e.T)
    eri = rng.normal(scale=0.04, size=(nsites, nsites, nsites, nsites))
    eri = 0.25 * (
        eri
        + eri.swapaxes(0, 1)
        + eri.swapaxes(2, 3)
        + eri.swapaxes(0, 1).swapaxes(2, 3)
    )
    eri = 0.5 * (eri + eri.transpose(2, 3, 0, 1))
    return h1e, eri


def _casci_rdm_reference(h1e, eri, *, nelec: int, j2: int):
    """CASCI RDM reference using the same determinant basis as the SU2 sector."""
    binary = _direct_ci_binary(h1e.shape[0], nelec, j2)
    hamiltonian = _sigma_compact_columns(h1e, eri, binary, np.eye(binary.shape[0]))
    values, vectors = eigh(0.5 * (hamiltonian + hamiltonian.T.conj()), check_finite=False)
    ci = vectors[:, 0]
    return values[0], make_rdm1(ci, binary, None), make_rdm2(ci, binary, None, None)


def _assert_rdms_match_reference(nsites: int, *, seed: int, D: int):
    h1e, eri = _physical_random_integrals(nsites, seed=seed)
    nelec = nsites
    j2 = 0
    mol = DummyMol((nelec // 2, nelec // 2), spin=j2)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=D,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
    ).run()

    dm1, dm2 = solver.make_rdm12(0)
    ref_energy, ref_dm1, ref_dm2 = _casci_rdm_reference(h1e, eri, nelec=nelec, j2=j2)

    np.testing.assert_allclose(solver.e_tot[0], ref_energy, atol=1.0e-12)
    np.testing.assert_allclose(dm1, ref_dm1, atol=1.0e-12)
    np.testing.assert_allclose(dm2, ref_dm2, atol=1.0e-12)
    np.testing.assert_allclose(np.trace(dm1), nelec, atol=1.0e-12)
    np.testing.assert_allclose(np.einsum("pprr->", dm2), nelec * (nelec - 1), atol=1.0e-12)
    rdm_energy = np.einsum("pq,qp", h1e, dm1) + 0.5 * np.einsum("pqrs,pqrs", eri, dm2)
    np.testing.assert_allclose(rdm_energy, solver.e_tot[0], atol=1.0e-12)


def test_su2_narg_rdm12_matches_casci_rdm_for_two_site_seed():
    _assert_rdms_match_reference(2, seed=4, D=10)


def test_su2_narg_rdm12_matches_casci_rdm_after_reduced_growth():
    _assert_rdms_match_reference(4, seed=9, D=80)


def test_su2_narg_rdm12_matches_casci_driver_rdm_api():
    nsites = 4
    nelec = 4
    j2 = 0
    h1e, eri = _physical_random_integrals(nsites, seed=19)
    mf = DummyMF(h1e, eri, nelec=nelec, spin=j2)

    casci = CASCI(mf, ncas=nsites, nelecas=nelec, ncore=0, spin=j2).run(
        nstates=1,
        method="direct_ci",
        use_cholesky=False,
    )
    solver = NARG(
        mf,
        mol=mf.mol,
        h1e=h1e,
        eri=eri,
        D=80,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=j2,
        su2_backend="python",
    ).run()

    su2_dm1, su2_dm2 = solver.make_rdm12(0)
    casci_dm1, casci_dm2 = casci.make_rdm12(0)

    np.testing.assert_allclose(solver.e_tot[0], casci.e_tot[0], atol=1.0e-12)
    np.testing.assert_allclose(su2_dm1, casci_dm1, atol=1.0e-12)
    np.testing.assert_allclose(su2_dm2, casci_dm2, atol=1.0e-12)


def test_su2_narg_rdm_requires_density_operator_carry():
    h1e, eri = _physical_random_integrals(4, seed=12)
    mol = DummyMol((2, 2), spin=0)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=40,
        nstates=1,
        final_size=4,
        target_nelec=4,
        target_j2=0,
        su2_backend="python",
        carry_rdm_operators=False,
    ).run()

    with pytest.raises(ValueError, match="carry_rdm_operators=True"):
        solver.make_rdm1(0)


def test_spin_orbital_rdms_contract_to_spin_traced_su2_rdms():
    nsites = 4
    nelec = 4
    h1e, eri = _physical_random_integrals(nsites, seed=31)
    mol = DummyMol((2, 2), spin=0)
    solver = NARG(
        SimpleNamespace(mol=mol),
        mol=mol,
        h1e=h1e,
        eri=eri,
        D=80,
        nstates=1,
        final_size=nsites,
        target_nelec=nelec,
        target_j2=0,
        su2_backend="python",
        carry_spin_rdm_operators=True,
    ).run()

    spin_dm1, spin_dm2 = solver.make_spin_orbital_rdm12()
    dm1, dm2 = solver.make_rdm12()
    spatial_dm1 = spin_dm1[:nsites, :nsites] + spin_dm1[nsites:, nsites:]
    spatial_dm2 = np.zeros_like(dm2, dtype=spin_dm2.dtype)
    for sigma in range(2):
        for tau in range(2):
            a = slice(sigma * nsites, (sigma + 1) * nsites)
            b = slice(tau * nsites, (tau + 1) * nsites)
            spatial_dm2 += spin_dm2[a, b, a, b].transpose(0, 2, 1, 3)

    np.testing.assert_allclose(np.trace(spin_dm1), nelec, atol=1.0e-11)
    np.testing.assert_allclose(
        np.einsum("pqpq->", spin_dm2), nelec * (nelec - 1), atol=1.0e-11
    )
    np.testing.assert_allclose(spatial_dm1.T, dm1, atol=1.0e-11)
    np.testing.assert_allclose(spatial_dm2, dm2, atol=1.0e-11)

    graph = solver.orbital_mutual_correlation()
    clusters = solver.correlated_orbital_blocks()
    np.testing.assert_allclose(graph, graph.T, atol=1.0e-13)
    np.testing.assert_allclose(np.diag(graph), 0.0, atol=1.0e-13)
    assert sorted(i for cluster in clusters for i in cluster) == list(range(nsites))
    assert all(len(cluster) == 2 for cluster in clusters)

    variable = solver.correlated_orbital_blocks(
        method="spectral",
        n_clusters=2,
        max_size=3,
    )
    assert sorted(i for cluster in variable for i in cluster) == list(range(nsites))
    assert max(map(len, variable)) <= 3

    narg_order = solver.correlated_orbital_blocks(
        method="narg",
        trial_D=8,
        order_candidates=2,
    )
    trials = solver.cluster_order_trials
    assert len(trials) == 2
    assert tuple(narg_order) == min(
        trials,
        key=lambda item: (item["energy"], item["graph_rank"]),
    )["blocks"]
    selected_trial = min(trials, key=lambda item: (item["energy"], item["graph_rank"]))
    direct = NARG(
        solver.mf,
        mol=solver.mol,
        h1e=solver.h1e,
        eri=solver.eri,
        D=8,
        nstates=1,
        target_nelec=nelec,
        target_j2=0,
        orbital_blocks=narg_order,
        su2_backend="python",
    ).run()
    np.testing.assert_allclose(selected_trial["energy"], direct.e_tot[0], atol=1.0e-12)
