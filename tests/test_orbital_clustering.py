from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.hf import RHF, UHF
from pyqed.qchem.orbital_clustering import (
    cluster_orbitals,
    graph_cut_ratio,
    maximum_weight_pair_clusters,
    orbital_boundary_cut_cost,
    orbital_cluster_order_candidates,
    orbital_mutual_correlation_graph,
    orbital_interaction_graph,
    order_orbital_clusters,
    spectral_orbital_clusters,
    two_body_cumulant,
)


def _as_sets(clusters):
    return {frozenset(cluster) for cluster in clusters}


def test_spectral_orbital_clusters_groups_strong_graph_pairs():
    graph = np.full((4, 4), 0.05)
    np.fill_diagonal(graph, 0.0)
    graph[0, 1] = graph[1, 0] = 10.0
    graph[2, 3] = graph[3, 2] = 9.0

    clusters = spectral_orbital_clusters(graph, n_clusters=2, max_size=2)

    assert _as_sets(clusters) == {frozenset((0, 1)), frozenset((2, 3))}
    assert graph_cut_ratio(graph, clusters) < 0.02


def test_spectral_orbital_clusters_can_form_larger_supersites():
    graph = np.full((6, 6), 0.01)
    np.fill_diagonal(graph, 0.0)
    for cluster in ((0, 1, 2), (3, 4, 5)):
        for i in cluster:
            for j in cluster:
                if i != j:
                    graph[i, j] = 1.0

    clusters = spectral_orbital_clusters(graph, n_clusters=2, max_size=3)

    assert _as_sets(clusters) == {
        frozenset((0, 1, 2)),
        frozenset((3, 4, 5)),
    }


def test_order_orbital_clusters_minimizes_successive_boundary_cuts():
    graph = np.zeros((6, 6))
    clusters = [(0, 1), (2, 3), (4, 5)]
    graph[np.ix_(clusters[0], clusters[1])] = 2.0
    graph[np.ix_(clusters[1], clusters[0])] = 2.0
    graph[np.ix_(clusters[1], clusters[2])] = 1.5
    graph[np.ix_(clusters[2], clusters[1])] = 1.5
    graph[np.ix_(clusters[0], clusters[2])] = 0.1
    graph[np.ix_(clusters[2], clusters[0])] = 0.1

    scrambled = [clusters[1], clusters[0], clusters[2]]
    ordered = order_orbital_clusters(graph, scrambled)

    assert ordered[1] == clusters[1]
    assert orbital_boundary_cut_cost(graph, ordered) < orbital_boundary_cut_cost(
        graph, scrambled
    )

    candidates = orbital_cluster_order_candidates(
        graph,
        scrambled,
        max_candidates=6,
    )
    assert len(candidates) == 6
    assert ordered in candidates
    assert list(reversed(ordered)) in candidates


def test_mutual_correlation_vanishes_for_a_slater_determinant():
    occupations = np.array([1.0, 1.0, 0.0, 0.0])
    dm1 = np.diag(occupations)
    dm2 = (
        np.einsum("pr,qs->pqrs", dm1, dm1)
        - np.einsum("ps,qr->pqrs", dm1, dm1)
    )

    np.testing.assert_allclose(two_body_cumulant(dm1, dm2), 0.0, atol=1.0e-14)
    np.testing.assert_allclose(
        orbital_mutual_correlation_graph(dm1, dm2), 0.0, atol=1.0e-14
    )


def test_maximum_weight_pair_clusters_finds_global_pairing():
    graph = np.zeros((6, 6))
    graph[0, 4] = graph[4, 0] = 5.0
    graph[1, 3] = graph[3, 1] = 4.0
    graph[2, 5] = graph[5, 2] = 3.0
    graph[0, 1] = graph[1, 0] = 5.5

    clusters = maximum_weight_pair_clusters(graph)

    assert _as_sets(clusters) == {
        frozenset((0, 4)),
        frozenset((1, 3)),
        frozenset((2, 5)),
    }


def test_cluster_orbitals_preserves_active_orbital_indices():
    h1e = np.zeros((5, 5))
    h1e[1, 3] = h1e[3, 1] = 7.0
    h1e[2, 4] = h1e[4, 2] = 6.0
    eri = np.zeros((5, 5, 5, 5))

    clusters = cluster_orbitals(
        h1e=h1e,
        eri=eri,
        active=[1, 2, 3, 4],
        weights="integral",
        n_clusters=2,
        max_size=2,
    )

    assert _as_sets(clusters) == {frozenset((1, 3)), frozenset((2, 4))}


def test_orbital_interaction_graph_can_use_rdm_weights():
    dm = np.zeros((4, 4))
    dm[0, 1] = dm[1, 0] = 0.7
    dm[2, 3] = dm[3, 2] = 0.6

    graph = orbital_interaction_graph(dm=dm, weights="rdm")

    np.testing.assert_allclose(graph[0, 1], 0.7)
    np.testing.assert_allclose(graph[2, 3], 0.6)
    np.testing.assert_allclose(np.diag(graph), 0.0)


def test_rhf_cluster_uses_density_by_default():
    mol = SimpleNamespace(
        nao=4,
        nelec=4,
        hcore=np.zeros((4, 4)),
        eri=np.zeros((4, 4, 4, 4)),
        overlap=np.eye(4),
    )
    mf = RHF(mol)
    mf.mo_coeff = np.eye(4)
    mf.mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    calls = {"rdm": 0}

    def make_pair_density():
        calls["rdm"] += 1
        dm = np.zeros((4, 4))
        dm[0, 1] = dm[1, 0] = 1.0
        dm[2, 3] = dm[3, 2] = 0.9
        return dm

    mf.make_rdm1 = make_pair_density

    clusters, info = mf.cluster(n_clusters=2, max_size=2, return_info=True)

    assert calls["rdm"] == 1
    assert _as_sets(clusters) == {frozenset((0, 1)), frozenset((2, 3))}
    assert info["method"] == "spectral"
    assert info["weights"] == "integral+rdm"
    assert info["graph"].shape == (4, 4)


def test_rhf_cluster_rdm_weights_do_not_require_eri():
    mol = SimpleNamespace(
        nao=4,
        nelec=4,
        hcore=None,
        eri=None,
        overlap=np.eye(4),
    )
    mf = RHF(mol)
    mf.mo_coeff = np.eye(4)
    mf.mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    dm = np.zeros((4, 4))
    dm[0, 1] = dm[1, 0] = 1.0
    dm[2, 3] = dm[3, 2] = 0.9

    clusters = mf.cluster(weights="rdm", dm=dm, n_clusters=2, max_size=2)

    assert _as_sets(clusters) == {frozenset((0, 1)), frozenset((2, 3))}


def test_rhf_cluster_integral_weights_do_not_require_density():
    mol = SimpleNamespace(
        nao=4,
        nelec=4,
        hcore=np.zeros((4, 4)),
        eri=np.zeros((4, 4, 4, 4)),
        overlap=np.eye(4),
    )
    mol.hcore[0, 1] = mol.hcore[1, 0] = 2.0
    mol.hcore[2, 3] = mol.hcore[3, 2] = 1.5
    mf = RHF(mol)
    mf.mo_coeff = np.eye(4)
    mf.mo_occ = np.array([2.0, 2.0, 0.0, 0.0])
    mf.make_rdm1 = lambda: (_ for _ in ()).throw(AssertionError("density should not be used"))

    clusters = mf.cluster(weights="integral", n_clusters=2, max_size=2)

    assert _as_sets(clusters) == {frozenset((0, 1)), frozenset((2, 3))}


def test_rhf_cluster_can_localize_active_orbitals_before_clustering():
    mol = SimpleNamespace(
        nao=5,
        nelec=4,
        hcore=np.zeros((5, 5)),
        eri=np.zeros((5, 5, 5, 5)),
        overlap=np.eye(5),
    )
    mf = RHF(mol)
    mf.mo_coeff = np.eye(5)
    mf.mo_occ = np.array([2.0, 2.0, 0.0, 0.0, 0.0])
    calls = {}

    def fake_localize_orbitals(method, mo_coeff, **kwargs):
        calls["method"] = method
        calls["shape"] = mo_coeff.shape
        calls["kwargs"] = kwargs
        return mo_coeff

    mf.localize_orbitals = fake_localize_orbitals
    dm = np.zeros((5, 5))
    dm[1, 2] = dm[2, 1] = 1.0
    dm[3, 4] = dm[4, 3] = 0.9

    clusters, info = mf.cluster(
        orbitals="localized",
        localization="pm",
        active=[1, 2, 3, 4],
        weights="rdm",
        dm=dm,
        n_clusters=2,
        max_size=2,
        localize_kwargs={"max_cycle": 3},
        return_info=True,
    )

    assert calls == {"method": "pm", "shape": (5, 4), "kwargs": {"max_cycle": 3}}
    assert _as_sets(clusters) == {frozenset((1, 2)), frozenset((3, 4))}
    assert info["orbitals"] == "localized"
    assert info["localization"] == "pm"
    assert info["source_orbitals"] == (1, 2, 3, 4)
    np.testing.assert_allclose(info["mo_coeff"], np.eye(5))


def test_rhf_cluster_can_return_localized_orbital_coefficients():
    mol = SimpleNamespace(
        nao=3,
        nelec=2,
        hcore=np.zeros((3, 3)),
        eri=np.zeros((3, 3, 3, 3)),
        overlap=np.eye(3),
    )
    mf = RHF(mol)
    mf.mo_coeff = np.eye(3)
    mf.mo_occ = np.array([2.0, 0.0, 0.0])
    localized_block = np.eye(3)[:, [1, 0]]
    mf.localize_orbitals = lambda method, mo_coeff, **kwargs: localized_block

    clusters, coeff = mf.cluster(
        orbitals="localized",
        active=[0, 1],
        weights="integral",
        n_clusters=1,
        return_orbitals=True,
    )

    assert clusters == [(0, 1)]
    np.testing.assert_allclose(coeff[:, [0, 1]], localized_block)


def test_uhf_cluster_accepts_spin_resolved_density():
    mol = SimpleNamespace(
        nao=4,
        nelec=2,
        hcore=np.zeros((4, 4)),
        eri=np.zeros((4, 4, 4, 4)),
        overlap=np.eye(4),
    )
    mf = UHF(mol)
    mf.mo_coeff = (np.eye(4), np.eye(4))
    mf.mo_occ = (np.array([1.0, 0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0, 0.0]))

    dm = np.zeros((2, 4, 4))
    dm[0, 0, 1] = dm[0, 1, 0] = 0.8
    dm[1, 2, 3] = dm[1, 3, 2] = 0.7

    clusters = mf.cluster(dm=dm, n_clusters=2, max_size=2)

    assert _as_sets(clusters) == {frozenset((0, 1)), frozenset((2, 3))}


def test_uhf_cluster_rejects_implicit_localization():
    mol = SimpleNamespace(
        nao=2,
        nelec=2,
        hcore=np.zeros((2, 2)),
        eri=np.zeros((2, 2, 2, 2)),
        overlap=np.eye(2),
    )
    mf = UHF(mol)
    mf.mo_coeff = (np.eye(2), np.eye(2))
    mf.mo_occ = (np.array([1.0, 0.0]), np.array([1.0, 0.0]))

    with pytest.raises(NotImplementedError, match="localized"):
        mf.cluster(orbitals="localized", weights="integral")
