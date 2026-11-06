"""AO-backed CASSCF contractions versus explicit full-MO references."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem import Molecule, basis
from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
from pyqed.qchem.mcscf.casci import _get_mf_cholesky_factors
from pyqed.qchem.mcscf.direct_ci import transform_active_space_pair_factors
from pyqed.qchem.mcscf.orbopt import (
    CoreActiveRDM,
    generalized_fock_from_factors,
    core_active_factor_hessian_action,
)
from pyqed.qchem.mcscf.ri_orbitals import OrbitalRIFactors


@pytest.mark.parametrize("workers", [1, 4])
@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("ncore,naux", [(0, 0), (0, 19), (2, 19)])
def test_orbital_ri_exact_contractions(workers, cache, ncore, naux, monkeypatch):
    rng = np.random.default_rng(902)
    m, n, a = 11, 9, 3
    source = basis.PackedRIFactors(rng.normal(size=(naux, m * (m + 1) // 2)), m)
    coefficients = np.linalg.qr(rng.normal(size=(m, m)))[0][:, :n]
    h = rng.normal(size=(n, n))
    h += h.T.copy()
    ref = basis.mo_pair_factors(source, coefficients, packed=True)
    parent = SimpleNamespace(
        mol=None,
        mo_occ=np.r_[np.full(ncore + 1, 2.0), np.zeros(n - ncore - 1)],
        energy_nuc=lambda: 1.1,
    )
    frozen = SecondOrderCASSCF._FrozenFactorRHF(parent, h, ref, np.eye(n))
    expected = transform_active_space_pair_factors(frozen, np.eye(n), a, ncore)
    streamed = OrbitalRIFactors(
        source, coefficients, memory_mb=0.02, workers=workers, cache_occupied=cache
    )
    allocations = []
    from pyqed.qchem.mcscf import ri_orbitals

    allocate = ri_orbitals._ri_storage

    def checked(shape, **kwargs):
        allocations.append(shape)
        assert tuple(shape) not in [(naux, n, n), (naux, n * (n + 1) // 2)]
        return allocate(shape, **kwargs)

    monkeypatch.setattr(ri_orbitals, "_ri_storage", checked)
    actual = streamed.active_integrals(h, ncore, a, 1.1)
    for x, y in zip(actual, expected):
        np.testing.assert_allclose(x, y, atol=2e-12, rtol=0)
    o = ncore + a
    dm1a = rng.normal(size=(a, a))
    dm1a += dm1a.T.copy()
    dm1 = np.zeros((o, o))
    dm1[:ncore, :ncore] = 2 * np.eye(ncore)
    dm1[ncore:, ncore:] = dm1a
    dm2 = CoreActiveRDM(ncore, dm1a, rng.normal(size=(a,) * 4))
    np.testing.assert_allclose(
        streamed.generalized_fock(h, dm1, dm2),
        generalized_fock_from_factors(h, ref, dm1, dm2),
        atol=3e-11,
        rtol=0,
    )
    k = rng.normal(size=(3, n, n)) * 0.01
    k -= k.transpose(0, 2, 1).copy()
    expected_action = core_active_factor_hessian_action(h, ref, dm1, dm2, k)
    np.testing.assert_allclose(
        streamed.hessian_action(h, dm1, dm2, k), expected_action, atol=3e-11, rtol=0
    )
    np.testing.assert_allclose(
        streamed.hessian_action(h, dm1, dm2, k[0]),
        expected_action[0],
        atol=3e-11,
        rtol=0,
    )
    dm = rng.normal(size=(n, n)) + 1j * rng.normal(size=(n, n))
    np.testing.assert_allclose(
        streamed.veff(dm), frozen.get_veff(dm), atol=3e-11, rtol=0
    )
    info = streamed.storage_info()
    assert info["full_mo_factor_bytes"] == 0
    assert info["occupied_factor_bytes"] == (naux * n * o * 8 if cache else 0)
    assert (
        info["planned_workspace_bytes"] <= info["target_bytes"]
        or info["workers"] == info["auxiliary_block_size"] == 1
    )
    rotated = streamed.rotate(expm(k[0]))
    assert (
        rotated.source is source
        and rotated._occupied is None
        and not rotated.cache_occupied
    )
    np.testing.assert_allclose(
        rotated.mo_factors(np.eye(n)[:, :a]),
        basis.mo_pair_factors(source, coefficients @ expm(k[0])[:, :a]),
        atol=2e-12,
        rtol=0,
    )


def test_fused_trial_ci_uses_one_pass_and_no_full_potential(monkeypatch):
    rng = np.random.default_rng(38)
    n, p, c, a = 12, 23, 2, 3
    source = basis.PackedRIFactors(rng.normal(size=(p, n * (n + 1) // 2)), n)
    u = np.linalg.qr(rng.normal(size=(n, n)))[0]
    parent = SimpleNamespace(
        mol=None, mo_occ=np.r_[2, 2, 2, np.zeros(n - 3)], energy_nuc=lambda: 0.0
    )
    frozen = SecondOrderCASSCF._RotatedFactorRHF(
        parent, np.eye(n), source, u, np.eye(n), 0.01
    )

    def forbidden(*args, **kwargs):
        pytest.fail(
            "Trial CI requested a full potential or an extra active-factor pass"
        )

    monkeypatch.setattr(frozen, "get_veff", forbidden)
    monkeypatch.setattr(frozen, "mo_factors", forbidden)
    calls = []
    original = basis.PackedRIFactors.block

    def block(self, start, stop):
        calls.extend(range(start, stop))
        return original(self, start, stop)

    monkeypatch.setattr(basis.PackedRIFactors, "block", block)
    first = transform_active_space_pair_factors(frozen, np.eye(n), a, c)
    again = transform_active_space_pair_factors(frozen, np.eye(n), a, c)
    assert sorted(calls) == list(range(p))
    assert all(x is y for x, y in zip(first, again))
    assert frozen._materialized is None


def test_casscf_without_full_mo_reference(monkeypatch):
    import importlib
    import weakref

    module = importlib.import_module("pyqed.qchem.mcscf.casscf")
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = mol.RHF().run()
    dense = _get_mf_cholesky_factors(mf)
    packed = basis.mo_pair_factors(dense, np.eye(mf.nmo), packed=True)
    k = np.zeros((mf.nmo, mf.nmo))
    k[1, 3], k[3, 1] = 0.2, -0.2
    results = []
    original = module.transform_eri_factors_to_mo_pair
    caches = []
    active = OrbitalRIFactors.active_integrals

    def tracked(self, *args, **kwargs):
        building = self.cache_occupied and self._occupied is None
        if building:
            assert all(ref() is None for ref in caches)
        result = active(self, *args, **kwargs)
        if building:
            caches.append(weakref.ref(self._occupied))
        return result

    monkeypatch.setattr(OrbitalRIFactors, "active_integrals", tracked)
    for streaming in (False, True):
        mf.eri_factors = packed if streaming else dense
        solver = SecondOrderCASSCF(
            mf,
            ncas=2,
            nelecas=2,
            coupling="qn",
            use_cholesky=True,
            max_cycle=40,
            conv_tol=1e-10,
            conv_tol_grad=1e-6,
            conv_tol_grad_relaxed=1e-6,
            trial_memory_mb=0.01,
            verbose=0,
        )
        solver.state_average([0.5, 0.5])

        def transform(*args, **kwargs):
            assert not streaming, "Full MO-factor transformation in AO-backed CASSCF"
            return original(*args, **kwargs)

        monkeypatch.setattr(module, "transform_eri_factors_to_mo_pair", transform)
        solver.run(nstates=2, mo_coeff=mf.mo_coeff @ expm(k))
        assert solver.converged
        results.append(solver.e_tot)
        if streaming:
            assert len(caches) > 1
            assert all(
                row["ri_storage"]["full_mo_factor_bytes"] == 0 for row in solver.history
            )
    # The state-average objective is stationary; individual roots need not be.
    np.testing.assert_allclose(
        np.mean(results[0]), np.mean(results[1]), atol=5e-10, rtol=0
    )
    np.testing.assert_allclose(*results, atol=5e-7, rtol=0)


def test_parallel_budget_limits_blas_and_restores_caller():
    from threadpoolctl import threadpool_info, threadpool_limits

    source = basis.PackedRIFactors(np.ones((17, 6)), 3)
    factors = OrbitalRIFactors(source, np.eye(3), memory_mb=0.001, workers=4)

    def counts():
        return [
            library["num_threads"]
            for library in threadpool_info()
            if library["user_api"] == "blas"
        ]

    with threadpool_limits(limits=3, user_api="blas"):
        before = counts()
        rows = factors._run(100, lambda interval: counts(), fixed_bytes=100)
        assert rows and all(all(n == 1 for n in row) for row in rows)
        assert counts() == before
    assert factors.last_plan["workers"] == 4
    assert factors.last_plan["planned_workspace_bytes"] <= factors.memory_bytes
