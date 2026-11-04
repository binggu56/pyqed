from itertools import combinations
import numpy as np
import pytest
from pyqed.qchem.mcscf import nevpt2


@pytest.mark.parametrize("compiled", [False, True])
def test_packed_pairs_preserve_ras_excitation_commutator(compiled, monkeypatch):
    if compiled and nevpt2._cpp_attr("spin_free_excitation_pairs") is None:
        pytest.skip("Excitation-link kernel is unavailable.")
    if not compiled:
        monkeypatch.setattr(nevpt2, "_cpp_attr", lambda name: None)
    strings = list(combinations(range(4), 2))
    binary = np.zeros((len(strings) ** 2, 2, 4), dtype=np.int8)
    for k, (a, b) in enumerate((a, b) for a in strings for b in strings):
        binary[k, 0, list(a)] = 1
        binary[k, 1, list(b)] = 1
    selected = binary[:, :, :2].sum(axis=(1, 2)) == 1
    ci = np.random.default_rng(805).normal(size=selected.sum())
    ci /= np.linalg.norm(ci)
    ci, binary = nevpt2._rdm_excitation_closure(ci, binary[selected])
    full = nevpt2._spin_free_excitation_pairs(ci, binary)
    singles, packed = nevpt2._spin_free_excitation_pairs(ci, binary, packed=True)
    a, b = np.triu_indices(16)
    np.testing.assert_allclose(packed, full.reshape(16, 16, -1)[a, b], atol=1e-14)
    np.testing.assert_allclose(singles, np.einsum("pprsd->rsd", full) / 4, atol=1e-14)
    for r, s, t, u in np.ndindex(4, 4, 4, 4):
        expected = (singles[r, u] if s == t else 0) - (singles[t, s] if r == u else 0)
        np.testing.assert_allclose(
            full[r, s, t, u] - full[t, u, r, s], expected, atol=1e-14
        )


def test_rdm_workspace_guard_and_empty_active_space():
    from pyqed.qchem.mcscf.rdm_contractions import RDMContractions
    from pyqed.qchem.mcscf.perturber_templates import solve_shared_fois

    binary = np.ones((1, 2, 2), dtype=np.int8)
    with pytest.raises(MemoryError):
        RDMContractions(np.ones(1), binary, np.zeros(2), max_memory_mb=0.0001)
    binary = np.zeros((1, 2, 0), dtype=np.int8)
    h = np.diag([-1.0, 0.5])
    eri = np.zeros((2,) * 4)
    eri[1, 0, 1, 0] = 0.2
    options = dict(
        binary=binary,
        ci=np.ones(1),
        ncore=1,
        ncas=0,
        h1=h,
        eri=eri,
        fock=h,
        reference_active_fock=0.0,
        norm_tol=1e-10,
    )
    a = solve_shared_fois(**options)
    b = solve_shared_fois(**options, fock_contraction="rdm")
    assert a["rank"] == b["rank"]
    np.testing.assert_allclose(a["energy"], b["energy"], atol=1e-12)


@pytest.mark.parametrize("n", [2, 3])
def test_rdm_only_metrics_and_weighted_fock_against_excitation_vectors(n):
    from pyqed.qchem.mcscf.perturber_templates import _ActiveTemplate
    from pyqed.qchem.mcscf.rdm_contractions import RDMContractions

    strings = list(combinations(range(n), n - 1))
    binary = np.zeros((len(strings) ** 2, 2, n), dtype=np.int8)
    for k, (a, b) in enumerate((a, b) for a in strings for b in strings):
        binary[k, 0, list(a)] = 1
        binary[k, 1, list(b)] = 1
    ci = np.random.default_rng(14).normal(size=len(binary))
    ci /= np.linalg.norm(ci)
    energies = np.linspace(-0.7, 0.3, n)
    engine = RDMContractions(ci, binary, energies)
    for pattern in [
        ((0,), ()),
        ((), (0,)),
        ((0,), (0,)),
        ((0, 1), (0,)),
        ((0,), (0, 1)),
        ((0, 0), (0, 0)),
    ]:
        t = _ActiveTemplate(pattern, binary, ci, np.diag(energies), 1e-12, 1e-10)
        gram = engine.matrix(t.operators, t.operators, t.ncore)
        ids = np.flatnonzero(t.nonzero)
        norm = t.norms[ids]
        reduced = gram[np.ix_(ids, ids)] / norm[:, None] / norm[None, :]
        np.testing.assert_allclose(reduced, (t.raw.T @ t.raw).toarray(), atol=1e-12)
        eps = np.r_[np.zeros(t.ncore), energies, np.zeros(t.nmo - t.ncore - n)]
        shifts = np.array([sum(eps[p] - eps[q] for p, q in op) for op in t.operators])
        moment = (
            engine.matrix(t.operators, t.operators, t.ncore, weighted=True)
            + 0.5 * (shifts[:, None] + shifts[None, :]) * gram
        )
        moment = moment[np.ix_(ids, ids)] / norm[:, None] / norm[None, :]
        np.testing.assert_allclose(
            t.transform.T @ moment @ t.transform, t.fock, atol=1e-11
        )


def test_transition_density_and_fock_moments_keep_open_intermediates():
    from scipy import sparse
    from pyqed.qchem.mcscf.transition_density import (
        annihilation_factor,
        contract_factors,
        one_body_moment,
    )
    from pyqed.qchem.mcscf.caspt2 import _one_body_sparse_matrix_in_determinant_space

    rows = np.array([3, 5, 6, 9, 10, 12], dtype=np.uint64)
    rng = np.random.default_rng(82)
    columns = sparse.csr_matrix(rng.normal(size=(6, 3)))
    fock = np.array([[0.3, 0.2], [0.2, -0.4]])
    operator, _ = _one_body_sparse_matrix_in_determinant_space(
        rows.tolist(), fock, fock, 2
    )
    expected = (columns.T @ operator @ columns).toarray()
    np.testing.assert_allclose(
        one_body_moment(rows, columns, fock), expected, atol=1e-12
    )
    # N-1 rows have no overlap with the supplied N-electron basis.
    factor = annihilation_factor(rows, columns, 0)
    assert not set(factor[0]) & set(rows)
    density = contract_factors(factor, factor)
    occupied = ((rows & 1) != 0).astype(float)
    np.testing.assert_allclose(
        density, (columns.T @ columns.multiply(occupied[:, None])).toarray(), atol=1e-12
    )


def test_restricted_density_contractions_keep_intermediate_states():
    n = 3
    strings = list(combinations(range(n), 2))
    binary = np.zeros((len(strings) ** 2, 2, n), dtype=np.int8)
    for k, (a, b) in enumerate((a, b) for a in strings for b in strings):
        binary[k, 0, list(a)] = 1
        binary[k, 1, list(b)] = 1
    selected = binary[:, :, 0].sum(axis=1) == 1
    ci = np.random.default_rng(791).normal(size=selected.sum())
    ci /= np.linalg.norm(ci)
    full = np.zeros(len(binary))
    full[selected] = ci
    expected = nevpt2._spin_free_rdms123_python(full, binary)
    actual = nevpt2._spin_free_rdms123(ci, binary[selected])
    truncated = nevpt2._spin_free_rdms123_python(ci, binary[selected])
    assert np.max(abs(truncated[2] - expected[2])) > 0.1
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    from pyqed.qchem.mcscf.rdm_contractions import RDMContractions, _normal_rdms

    eps = np.array([-0.8, 0.2, 0.5])
    engine = RDMContractions(ci, binary[selected], eps)
    expected_normal = _normal_rdms(full, binary)
    weighted = (binary.sum(axis=1) @ eps - engine.center) * full
    plus = _normal_rdms(full + weighted, binary)
    minus = _normal_rdms(full - weighted, binary)
    for actual, reference, p, m in zip(engine.rdms, expected_normal, plus, minus):
        np.testing.assert_allclose(actual, reference, atol=1e-12)
    for actual, p, m in zip(engine.weighted, plus, minus):
        np.testing.assert_allclose(actual, (p - m) / 4, atol=1e-12)
