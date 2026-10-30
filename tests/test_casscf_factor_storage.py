from types import SimpleNamespace

import numpy as np
import pytest

from pyqed.qchem.mcscf.casscf import SecondOrderCASSCF
from pyqed.qchem import basis


@pytest.mark.parametrize('occupied', [[], [0, 2], list(range(7))])
@pytest.mark.parametrize('complex_values', [False, True])
def test_factor_veff_matches_full_contraction(occupied, complex_values):
    rng = np.random.default_rng(39)
    b = rng.normal(size=(13, 7, 7))
    if complex_values:
        b = b + 1j*rng.normal(size=b.shape)
    dm = np.zeros((7, 7), dtype=b.dtype)
    d = rng.normal(size=(len(occupied), len(occupied)))
    if complex_values:
        d = d + 1j*rng.normal(size=d.shape)
    dm[np.ix_(occupied, occupied)] = d
    parent = SimpleNamespace(mol=None, mo_occ=np.array([2, 2, 0, 0, 0, 0, 0]),
                             energy_nuc=lambda: 0.)
    frozen = SecondOrderCASSCF._FrozenFactorRHF(parent, np.eye(7), b, np.eye(7))
    assert np.shares_memory(frozen.eri_factors, b)
    assert not frozen.eri_factors.flags.writeable
    ref = (np.einsum('Pkl,lk,Pij->ij', b, dm, b, optimize=True)
           -.5*np.einsum('Pil,lk,Pkj->ij', b, dm, b, optimize=True))
    np.testing.assert_allclose(frozen.get_veff(dm), ref, atol=1e-11)
    solver = object.__new__(SecondOrderCASSCF)
    u = rng.normal(size=(7, 7))
    if complex_values:
        u = u + 1j*rng.normal(size=u.shape)
    _, rotated = solver._transform_frozen_factor_integrals(np.eye(7), b, u)
    np.testing.assert_allclose(rotated,
        np.einsum('pi,Ppq,qj->Pij', u.conj(), b, u, optimize=True), atol=1e-11)


def test_disk_factor_rotation_and_output_reuse(tmp_path, monkeypatch):
    allocations = []
    def allocate(shape, dtype=np.float64):
        mapped = np.memmap(tmp_path / str(len(allocations)), mode='w+', dtype=dtype, shape=shape)
        allocations.append(mapped)
        return mapped
    monkeypatch.setattr(basis, '_ri_storage', allocate)
    rng = np.random.default_rng(51)
    b = rng.normal(size=(11, 7, 7))
    b += b.transpose(0, 2, 1)
    rows, cols = np.tril_indices(7)
    packed = basis.PackedRIFactors(b[:, rows, cols], 7)
    u = np.linalg.qr(rng.normal(size=(7, 7)))[0]
    result = basis.mo_pair_factors(packed, u, packed=True)
    expected = np.einsum('pi,Ppq,qj->Pij', u, b, u)
    np.testing.assert_allclose(result.pair_factors, expected[:, rows, cols], atol=1e-12)
    assert np.shares_memory(result.pair_factors, allocations[0])
    again = basis.mo_pair_factors(packed, np.eye(7), packed=True, out=result)
    assert len(allocations) == 1
    assert np.shares_memory(again.pair_factors, result.pair_factors)
    np.testing.assert_allclose(again.pair_factors, packed.pair_factors, atol=1e-12)
    with pytest.raises(ValueError, match='disjoint'):
        basis.mo_pair_factors(packed, u, packed=True, out=packed)


def test_packed_occupied_contractions_do_not_unpack_spectators(monkeypatch):
    from pyqed.qchem.mcscf.orbopt import generalized_fock_from_factors
    rng = np.random.default_rng(52)
    b = rng.normal(size=(13, 7, 7))
    b += b.transpose(0, 2, 1)
    rows, cols = np.tril_indices(7)
    packed = basis.PackedRIFactors(b[:, rows, cols], 7)
    parent = SimpleNamespace(mol=None, mo_occ=np.array([2, 2, 0, 0, 0, 0, 0]), energy_nuc=lambda: 0.)
    frozen = SecondOrderCASSCF._FrozenFactorRHF(parent, np.eye(7), packed, np.eye(7))
    dm = np.diag([2., 2., .7, .3, 0., 0., 0.])
    expected = (np.einsum('Pkl,lk,Pij->ij', b, dm, b)
                -.5*np.einsum('Pil,lk,Pkj->ij', b, dm, b))
    monkeypatch.setattr(basis.PackedRIFactors, 'block', lambda *args: pytest.fail('Unpacked spectator pairs'))
    np.testing.assert_allclose(frozen.get_veff(dm), expected, atol=1e-11)
    d1 = dm[:4, :4]
    d2 = rng.normal(size=(4, 4, 4, 4))
    reference = generalized_fock_from_factors(np.eye(7), b[:, :, :4], d1, d2)
    actual = generalized_fock_from_factors(np.eye(7), packed, d1, d2)
    np.testing.assert_allclose(actual, reference, atol=1e-11)


def test_line_search_reuses_only_discarded_factor_storage(monkeypatch):
    solver = object.__new__(SecondOrderCASSCF)
    solver.conv_tol = 1e-8
    solver.mo_coeff_ref = np.eye(4)
    solver.nstates, solver.state_id = 2, 0
    solver._apply_orbital_update = lambda u, k: u + k
    energies = iter([3., 4., 2., 5.])
    allocated = []
    original = basis._ri_storage
    def allocate(shape, dtype=np.float64):
        result = original(shape, dtype)
        allocated.append(result)
        return result
    monkeypatch.setattr(basis, '_ri_storage', allocate)
    parent = SimpleNamespace(mol=None, mo_occ=np.array([2, 0, 0, 0]), energy_nuc=lambda: 0.)
    snapshots = {}
    def make(h, factors, mo, nstates, ci0=None):
        energy = next(energies)
        snapshots[energy] = factors.pair_factors.copy()
        return SimpleNamespace(mf=solver._FrozenFactorRHF(parent, h, factors, mo), energy=energy)
    solver._make_factor_integral_casci = make
    solver._objective_energy = lambda mc, state: mc.energy
    factors = basis.PackedRIFactors(np.arange(30.).reshape(3, 10), 4)
    accepted, _, energy, mc, _ = solver._factor_micro_line_search(
        np.eye(4), factors, np.eye(4), .1*np.eye(4), 0., None)
    assert not accepted and energy == 2.
    assert len(allocated) == 2
    np.testing.assert_array_equal(mc.mf.eri_factors.pair_factors, snapshots[2.])


def test_macro_checkpoint_preserves_physical_orbitals():
    from pyqed.qchem import Molecule
    mf = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g').build().RHF().run()
    solver = SecondOrderCASSCF(mf, 2, 2, max_cycle=2, verbose=0)
    solver.state_average([.5, .5])
    snapshots = []
    def checkpoint(data):
        snapshots.append(data)
        raise InterruptedError('checkpoint reached')
    with pytest.raises(InterruptedError, match='checkpoint reached'):
        solver.run(nstates=2, macro_callback=checkpoint)
    snapshot = snapshots[0]
    assert snapshot['diagnostics']['cycle'] == 1
    assert snapshot['casci'] is not solver.casci
    np.testing.assert_array_equal(snapshot['casci'].mo_coeff, snapshot['mo_coeff'])
    overlap = mf.get_ovlp()
    np.testing.assert_allclose(snapshot['mo_coeff'].T@overlap@snapshot['mo_coeff'],
                               np.eye(mf.nmo), atol=1e-12)
