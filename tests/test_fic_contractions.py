import numpy as np
import pytest
from scipy import sparse
from pyqed.qchem.mcscf import caspt2


def test_fused_action_both_rank_orderings_and_workspace_reuse():
    kernel = caspt2._cpp_attr("apply_fic_graph")
    if kernel is None:
        pytest.skip("FIC kernel is unavailable")
    rng = np.random.default_rng(903)
    graph = sparse.csr_matrix(rng.normal(size=(5, 3)) * (rng.random((5, 3)) > 0.4))
    matrix = rng.normal(size=(2, 4))
    edges = []
    for s, t, g, m in [
        (0, 12, graph, matrix),
        (12, 0, graph.T.tocsr(), matrix.T.copy()),
    ]:
        edges.append(
            (
                s,
                t,
                g.shape[1],
                g.shape[0],
                m.shape[1],
                m.shape[0],
                g.indptr.astype(np.int64),
                g.indices.astype(np.int64),
                g.data,
                m,
            )
        )
    w = np.empty(10)
    z = np.empty(10)
    for _ in range(3):
        x = rng.normal(size=22)
        d = rng.normal(size=22)
        expected = d * x
        expected[:12] -= (2 * graph.T @ (x[12:].reshape(5, 2)) @ matrix).ravel()
        expected[12:] -= (2 * graph @ x[:12].reshape(3, 4) @ matrix.T).ravel()
        np.testing.assert_allclose(kernel(x, d, edges, w, z), expected, atol=1e-12)
    with pytest.raises(ValueError):
        kernel(x, d, edges, w[:1], z)


def test_fused_rdm_plan_matches_python_weighted_and_unweighted(monkeypatch):
    from pyqed.qchem.mcscf.rdm_contractions import RDMContractions
    from pyqed.qchem.mcscf.perturber_templates import _ActiveTemplate

    if caspt2._cpp_attr("contract_rdm_plan") is None:
        pytest.skip("RDM kernel is unavailable")
    binary = np.array([[[1, 1, 0], [1, 0, 1]], [[1, 0, 1], [1, 1, 0]]], dtype=np.int8)
    ci = np.array([0.6, 0.8])
    eps = np.array([-0.8, -0.1, 0.3])
    engine = RDMContractions(ci, binary, eps)
    for pattern in [
        ((0,), ()),
        ((), (0,)),
        ((0,), (0,)),
        ((0, 1), (0,)),
        ((0,), (0, 1)),
    ]:
        t = _ActiveTemplate(pattern, binary, ci, np.diag(eps), 1e-12, 1e-10)
        for weighted in [False, True]:
            actual = engine.matrix(t.operators, t.operators, t.ncore, weighted=weighted)
            with monkeypatch.context() as m:
                m.setattr(caspt2, "_cpp_attr", lambda name: None)
                expected = engine.matrix(
                    t.operators, t.operators, t.ncore, weighted=weighted
                )
            np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_topology_cache_refreshes_values_support_and_budget(monkeypatch):
    from pyqed.qchem.mcscf import perturber_templates as t

    groups = t._grouped_external_spaces(2, 6, 2, 0)
    assert groups is t._grouped_external_spaces(2, 6, 2, 0)
    with pytest.raises(TypeError):
        groups[()] = ()
    rng = np.random.default_rng(18)
    f = rng.normal(size=(6, 6))
    f = (f + f.T) / 2
    original = t._build_fock_topology
    calls = []

    def counted(*a, **kw):
        calls.append(1)
        return original(*a, **kw)

    monkeypatch.setattr(t, "_build_fock_topology", counted)
    t._fock_topology_cache = None
    first, n = t._fock_topology(groups, f, 2, 2)
    second, m = t._fock_topology(groups, 2 * f, 2, 2)
    assert len(calls) == 1 and m == n
    for k in first:
        np.testing.assert_array_equal(first[k][0], second[k][0])
        np.testing.assert_allclose(second[k][2], first[k][2] * (2 if k[2] == 2 else 1))
    with pytest.raises(MemoryError):
        t._fock_topology(groups, f, 2, 2, memory_bytes=1)
    f[4, 0] = f[0, 4] = 0
    t._fock_topology(groups, f, 2, 2)
    assert len(calls) == 2
    reduced = dict(list(groups.items())[1:])
    t._fock_topology(reduced, f, 2, 2)
    assert len(calls) == 3
