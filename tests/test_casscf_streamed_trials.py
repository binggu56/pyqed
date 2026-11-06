"""Trial energies must not allocate a complete transformed RI tensor."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy.linalg import expm

from pyqed.qchem import Molecule, basis
from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
from pyqed.qchem.mcscf.orbopt import CoreActiveRDM, generalized_fock_from_factors


@pytest.mark.parametrize("occupied", [[], [0, 2], list(range(9))])
@pytest.mark.parametrize("complex_density", [False, True])
def test_streamed_trial_contractions_and_materialization(
    occupied, complex_density, monkeypatch
):
    rng = np.random.default_rng(812)
    n, p = 9, 23
    b = rng.normal(size=(p, n, n))
    b += b.transpose(0, 2, 1).copy()
    rows, cols = np.tril_indices(n)
    source = basis.PackedRIFactors(b[:, rows, cols], n)
    u = np.linalg.qr(rng.normal(size=(n, n)))[0]
    rotated = u.T @ b @ u
    parent = SimpleNamespace(
        mol=None, mo_occ=np.r_[2, 2, np.zeros(n - 2)], energy_nuc=lambda: 0.0
    )
    trial = SecondOrderCASSCF._RotatedFactorRHF(
        parent, np.eye(n), source, u, np.eye(n), 0.002
    )
    dm = np.zeros((n, n), dtype=complex if complex_density else float)
    d = rng.normal(size=(len(occupied), len(occupied)))
    if complex_density:
        d = d + 1j * rng.normal(size=d.shape)
    dm[np.ix_(occupied, occupied)] = d
    expected = np.einsum("Pkl,lk,Pij->ij", rotated, dm, rotated) - 0.5 * np.einsum(
        "Pil,lk,Pjk->ij", rotated, dm, rotated
    )
    blocks = []
    original = basis.PackedRIFactors.block

    def block(self, start, stop):
        blocks.append(stop - start)
        return original(self, start, stop)

    monkeypatch.setattr(basis.PackedRIFactors, "block", block)
    np.testing.assert_allclose(trial.get_veff(dm), expected, atol=3e-12)
    left, right = np.eye(n)[:, [2, 5]], np.eye(n)[:, [3, 6, 8]]
    np.testing.assert_allclose(
        trial.mo_factors(left, right), rotated[:, [2, 5]][:, :, [3, 6, 8]], atol=2e-12
    )
    assert trial._materialized is None
    assert max(blocks) <= 2
    dm1 = np.diag([2.0, 1.0, 1.0])
    dm2 = CoreActiveRDM(1, np.eye(2), rng.normal(size=(2,) * 4))
    expected_fock = generalized_fock_from_factors(
        np.eye(n), rotated[:, :, :3], dm1, dm2
    )
    np.testing.assert_allclose(
        trial.generalized_fock(dm1, dm2), expected_fock, atol=3e-12
    )
    assert trial._materialized is None
    materialized = trial.eri_factors
    assert trial.eri_factors is materialized
    assert trial._source is None
    assert not materialized.pair_factors.flags.writeable
    np.testing.assert_allclose(
        materialized.pair_factors, rotated[:, rows, cols], atol=2e-12
    )
    np.testing.assert_allclose(trial.get_veff(dm), expected, atol=3e-12)
    # The accepted geometry supplies the original derivative kernels unchanged.
    dm1 = np.diag([2.0, 1.0, 1.0])
    dm2 = CoreActiveRDM(1, np.eye(2), rng.normal(size=(2,) * 4))
    np.testing.assert_allclose(
        generalized_fock_from_factors(np.eye(n), materialized, dm1, dm2),
        generalized_fock_from_factors(np.eye(n), rotated[:, :, :3], dm1, dm2),
        atol=3e-12,
    )
    k = rng.normal(size=(n, n)) * 0.01
    k -= k.T.copy()
    solver = object.__new__(SecondOrderCASSCF)
    np.testing.assert_allclose(
        solver._orbital_hessian_action_from_factors(
            np.eye(n), materialized, dm1, dm2, k, {}
        ),
        solver._orbital_hessian_action_from_factors(
            np.eye(n), rotated, dm1, dm2, k, {}
        ),
        atol=3e-12,
    )


def test_streamed_casscf_matches_materialized_trials(monkeypatch):
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = mol.RHF().run()
    k = np.zeros((mf.nmo, mf.nmo))
    k[1, 3], k[3, 1] = 0.2, -0.2
    mo = mf.mo_coeff @ expm(k)
    runs = []
    trials = []
    for stream in (False, True):
        solver = SecondOrderCASSCF(
            mf,
            ncas=2,
            nelecas=2,
            max_cycle=40,
            conv_tol=1e-10,
            conv_tol_grad=1e-7,
            conv_tol_grad_relaxed=1e-7,
            coupling="qn",
            use_cholesky=True,
            trial_memory_mb=0.001,
            verbose=0,
        )
        solver.state_average([0.5, 0.5])
        cls = solver._RotatedFactorRHF

        def create(parent, h, source, u, coeff, budget):
            if stream:
                return cls(parent, h, source, u, coeff, budget)
            factors = basis.mo_pair_factors(source, u, packed=True)
            return solver._FrozenFactorRHF(parent, h, factors, coeff)

        monkeypatch.setattr(solver, "_RotatedFactorRHF", create)
        make = solver._make_factor_integral_casci

        def checked(*args, **kwargs):
            mc = make(*args, **kwargs)
            frozen = kwargs.get("frozen_mf")
            if stream and frozen is not None:
                assert frozen._materialized is None
                trials.append(frozen)
            return mc

        monkeypatch.setattr(solver, "_make_factor_integral_casci", checked)
        solver.run(nstates=2, mo_coeff=mo)
        assert solver.converged
        runs.append(solver)
    assert trials
    assert all(t._materialized is None for t in trials)
    np.testing.assert_allclose(runs[0].e_tot, runs[1].e_tot, atol=5e-9, rtol=0)
    assert all(s.history[-1]["gradient_norm"] < 1e-7 for s in runs)


@pytest.mark.parametrize("factor_output", [False, True])
def test_direct_ci_trial_integrals_do_not_request_full_factors(factor_output):
    from pyqed.qchem.mcscf.direct_ci import (
        transform_active_space_pair_factors,
        transform_active_space_spatial_integrals,
    )

    rng = np.random.default_rng(45)
    n, p = 9, 19
    source = basis.PackedRIFactors(rng.normal(size=(p, n * (n + 1) // 2)), n)
    u = np.linalg.qr(rng.normal(size=(n, n)))[0]
    h = rng.normal(size=(n, n))
    h += h.T.copy()
    parent = SimpleNamespace(
        mol=None, mo_occ=np.r_[2, 2, 2, np.zeros(n - 3)], energy_nuc=lambda: 1.2
    )
    trial = SecondOrderCASSCF._RotatedFactorRHF(parent, h, source, u, np.eye(n), 0.002)
    full = SecondOrderCASSCF._FrozenFactorRHF(
        parent, h, basis.mo_pair_factors(source, u, packed=True), np.eye(n)
    )
    if factor_output:
        transform = transform_active_space_pair_factors
        kwargs = {}
    else:
        transform = transform_active_space_spatial_integrals
        kwargs = {"use_cholesky": True}
    actual = transform(trial, np.eye(n), 3, 2, **kwargs)
    expected = transform(full, np.eye(n), 3, 2, **kwargs)
    for a, b in zip(actual, expected):
        np.testing.assert_allclose(a, b, atol=1e-12, rtol=0)
    assert trial._materialized is None


def test_macro_transition_releases_previous_reference(monkeypatch):
    import importlib
    import weakref

    module = importlib.import_module("pyqed.qchem.mcscf.casscf")
    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", unit="angstrom", basis="sto-3g")
    mol.build()
    mf = mol.RHF().run()
    solver = SecondOrderCASSCF(
        mf,
        ncas=2,
        nelecas=2,
        max_cycle=3,
        use_cholesky=True,
        coupling="qn",
        verbose=0,
        conv_tol_grad=1e-7,
        conv_tol_grad_relaxed=1e-7,
    )
    solver.state_average([0.5, 0.5])
    original = module.transform_eri_factors_to_mo_pair
    references = []

    def transform(source, coefficients, *args, **kwargs):
        macro_reference = coefficients is solver.mo_coeff_ref and kwargs.get("packed")
        if macro_reference:
            assert all(ref() is None for ref in references)
        result = original(source, coefficients, *args, **kwargs)
        if macro_reference:
            references.append(weakref.ref(result.pair_factors))
        return result

    monkeypatch.setattr(module, "transform_eri_factors_to_mo_pair", transform)

    # A callback that writes and returns must not pin the old factor tensor.
    class CheckpointReached(Exception):
        pass

    def checkpoint(state):
        if state["diagnostics"]["cycle"] == 2:
            raise CheckpointReached

    with pytest.raises(CheckpointReached):
        solver.run(nstates=2, macro_callback=checkpoint)
    assert len(references) == 2


def test_rejected_streamed_trials_keep_only_best_rotation(monkeypatch):
    solver = object.__new__(SecondOrderCASSCF)
    n, p = 10, 13
    solver.conv_tol, solver.trial_memory_mb = 1e-8, 0.002
    solver.ci_method = "direct_ci"
    solver.nstates, solver.state_id = 2, 0
    solver.mo_coeff_ref = np.eye(n)
    solver.mf = SimpleNamespace(
        mol=None, mo_occ=np.r_[2.0, np.zeros(n - 1)], energy_nuc=lambda: 0.0
    )
    solver._apply_orbital_update = lambda u, k: u + k
    source = basis.PackedRIFactors(
        np.arange(float(p * n * (n + 1) // 2)).reshape(p, -1), n
    )
    energies = iter([3.0, 4.0, 2.0, 5.0])
    saved = []

    def make(h, factors, mo, nstates, ci0=None, frozen_mf=None):
        assert factors is None
        frozen_mf.get_veff(np.diag(solver.mf.mo_occ))
        frozen_mf.mo_factors(np.eye(n)[:, 1:3])
        assert frozen_mf._materialized is None
        saved.append(frozen_mf)
        return SimpleNamespace(mf=frozen_mf, energy=next(energies))

    solver._make_factor_integral_casci = make
    solver._objective_energy = lambda mc, state: mc.energy
    accepted, u, energy, mc, scale = solver._factor_micro_line_search(
        np.eye(n), source, np.eye(n), 0.1 * np.eye(n), 0.0, None
    )
    assert not accepted and energy == 2.0 and scale == 0.25
    np.testing.assert_allclose(u, 1.025 * np.eye(n))
    assert len(saved) == 4 and all(t._materialized is None for t in saved)
    assert all(
        np.shares_memory(t._source.pair_factors, source.pair_factors) for t in saved
    )
    np.testing.assert_allclose(
        mc.mf.eri_factors.pair_factors, 1.025**2 * source.pair_factors, atol=1e-12
    )
