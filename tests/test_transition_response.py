import numpy as np
import pytest

from pyqed.qchem import Molecule, RKS
from pyqed.qchem.tddft import TDA, TDDFT
from pyqed.qchem.transition_response import TransitionResponse


def test_streamed_factor_actions(monkeypatch):
    from types import SimpleNamespace
    from pyqed.qchem.basis import PackedRIFactors
    import pyqed.qchem.transition_response as module

    rng = np.random.default_rng(45)
    f = rng.normal(size=(137, 7, 7)) * 0.03
    f = (f + f.transpose(0, 2, 1)) / 2
    rows, cols = np.tril_indices(7)
    mol = SimpleNamespace(nao=7, eri_factors=PackedRIFactors(f[:, rows, cols], 7))
    energy = np.array([-1.0, -0.7, -0.5, 0.2, 0.4, 0.7, 1.0])
    mf = SimpleNamespace(
        mol=mol,
        mo_coeff=np.eye(7),
        mo_occ=np.r_[np.full(3, 2.0), np.zeros(4)],
        mo_energy=energy,
    )
    direct = 2 * np.einsum("Pia,Pjb->iajb", f[:, :3, 3:], f[:, :3, 3:])
    a = (direct - np.einsum("Pij,Pab->iajb", f[:, :3, :3], f[:, 3:, 3:])).reshape(
        12, 12
    )
    b = (direct - np.einsum("Pib,Pja->iajb", f[:, :3, 3:], f[:, :3, 3:])).reshape(
        12, 12
    )
    a += np.diag((energy[3:] - energy[:3, None]).ravel())
    calls = []
    transform = module.mo_pair_factors

    def bounded(factors, *coefficients):
        calls.append(len(factors))
        assert len(factors) <= 64
        return transform(factors, *coefficients)

    monkeypatch.setattr(module, "mo_pair_factors", bounded)
    monkeypatch.setattr(
        PackedRIFactors,
        "__array__",
        lambda *a, **k: pytest.fail("full AO factor expansion"),
    )
    response = TransitionResponse(SimpleNamespace(_scf=mf))
    assert not calls
    v = rng.normal(size=(12, 3)) + 1j * rng.normal(size=(12, 3))
    np.testing.assert_allclose(response.tda(v), a @ v, atol=1e-12)
    np.testing.assert_allclose(
        response.rpa(np.vstack((v, v * 0.2))),
        np.block([[a, b], [-b, -a]]) @ np.vstack((v, v * 0.2)),
        atol=1e-12,
    )
    assert max(calls) < len(f)


@pytest.fixture(params=["dense", "cd", "ri"])
def molecule(request):
    return Molecule(
        atom="O 0 0 0; H .2 1.4 1.1; H -.1 -1.4 1.2", unit="bohr", basis="sto-3g"
    ).build(
        eri=request.param,
        auxbasis="cc-pvdz-jkfit",
        options={"ri_cache": False, "low_rank_tol": 1e-10},
    )


@pytest.mark.parametrize("lda", [False, True])
def test_actions_and_roots(molecule, lda, monkeypatch):
    mf = RKS(molecule, xc="svwn") if lda else molecule.RHF()
    mf.run()
    td = TDDFT(mf)
    a, b = td.get_ab()
    n = td.nocc * td.nvir
    a, b = a.reshape(n, n), b.reshape(n, n)
    response = TransitionResponse(td, block_size=3)
    rng = np.random.default_rng(23)
    vectors = rng.normal(size=(n, 3)) + 1j * rng.normal(size=(n, 3))
    np.testing.assert_allclose(response.tda(vectors), a @ vectors, atol=1e-12)
    double = np.vstack((vectors, vectors * 0.31))
    np.testing.assert_allclose(
        response.rpa(double), np.block([[a, b], [-b, -a]]) @ double, atol=1e-12
    )
    if lda:
        root = np.sqrt(response.gap)
        np.testing.assert_allclose(
            response.casida(vectors),
            root[:, None] * ((a + b) @ (root[:, None] * vectors)),
            atol=1e-12,
        )
    for driver in (TDA, TDDFT):
        reference = driver(mf).run(nstates=2, solver="dense")
        with monkeypatch.context() as patch:
            patch.setattr(
                driver, "get_ab", lambda self: pytest.fail("dense response constructed")
            )
            actual = driver(mf).run(nstates=2)
        np.testing.assert_allclose(actual.e, reference.e, atol=1e-8)
        assert actual.a is None and actual.b is None
        assert actual.solver_info["converged"]
        assert max(actual.solver_info["response_residuals"]) < 1e-8


def test_packed_exact_actions(molecule):
    if getattr(molecule, "eri_factors", None) is not None:
        pytest.skip("exact storage test")
    mf = molecule.RHF().run()
    td = TDDFT(mf)
    response = TransitionResponse(td)
    vectors = np.random.default_rng(3).normal(size=(2 * td.nocc * td.nvir, 2))
    expected = response.rpa(vectors)
    from pyqed.qchem.tddft import _dense_eri

    tensor = _dense_eri(molecule)
    rows, cols = np.tril_indices(molecule.nao)
    s4 = tensor[rows[:, None], cols[:, None], rows, cols]
    molecule.eri = None
    molecule.eri_s4 = s4
    np.testing.assert_allclose(response.rpa(vectors), expected, atol=1e-12)
    molecule.eri_s8 = s4[np.tril_indices(len(rows))]
    molecule.eri_s4 = None
    np.testing.assert_allclose(response.rpa(vectors), expected, atol=1e-12)


def test_solver_controls(molecule):
    mf = RKS(molecule, xc="svwn").run()
    for roots in (0, -1, 1.5, True, 10000):
        with pytest.raises(ValueError):
            TDDFT(mf).run(nstates=roots)
    td = TDDFT(mf).run(nstates=2, using_tda=True)
    np.testing.assert_allclose(td.e, TDA(mf).run(nstates=2).e, atol=1e-10)
    assert td.response_method == "tda"
    with pytest.raises(RuntimeError):
        TDDFT(mf).run(nstates=1, iterations=1, tolerance=1e-14)


def test_pcm_action_and_packed_factor_guard(molecule, monkeypatch):
    from pyqed.qchem.basis import PackedRIFactors

    mf = RKS(molecule, xc="svwn").run()
    td = TDDFT(mf)

    class Solvent:
        def _B_dot_x(self, density):
            return -0.01 * (density + density.T)

    td.with_solvent = Solvent()
    a, b = td.get_ab()
    n = td.nocc * td.nvir
    x = np.random.default_rng(10).normal(size=(2 * n, 3))
    monkeypatch.setattr(
        PackedRIFactors,
        "__array__",
        lambda *args, **kwargs: pytest.fail("full AO factor expansion"),
    )
    response = TransitionResponse(td)
    expected = (
        np.block(
            [[a.reshape(n, n), b.reshape(n, n)], [-b.reshape(n, n), -a.reshape(n, n)]]
        )
        @ x
    )
    np.testing.assert_allclose(response.rpa(x), expected, atol=1e-12)


def test_unstable_gap_is_rejected(molecule):
    mf = RKS(molecule, xc="svwn").run()
    mf.mo_energy[mf.mo_occ == 0] = -100.0
    with pytest.raises(ValueError, match="positive orbital gaps"):
        TDDFT(mf).run(nstates=1)
