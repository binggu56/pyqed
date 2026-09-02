import numpy as np
import pytest
from itertools import combinations
from types import SimpleNamespace

from pyqed.qchem import Molecule, bo_hamiltonian_derivatives
from pyqed.qchem.geometric import (
    BOHamiltonianDerivatives,
    GeometricFGTerms,
    dipole_exponential_ci_overlap,
    dipole_orbital_rotation_unitary,
    orbital_rotation_ci_overlap,
)
from pyqed.qchem.hf import RHF, UHF
from pyqed.qchem.mcscf.casci import (
    CASCI,
    _biorthogonal_ci_overlap_candidate,
    _factorized_ci_overlap,
    _occupation_lists,
    _overlap_slow_from_mo_overlap,
    _reconstruct_string_overlap_from_svd,
    _string_overlap_matrix,
    _string_singular_weights,
    _string_transform_matrix,
    mo_overlap,
    transform_spatial_eri_to_mo,
)
from pyqed.qchem.mcscf import direct_ci
from pyqed.qchem.mcscf.orbopt import (
    embed_rdm2,
    generalized_fock,
    orbital_gradient,
    pack_nonredundant,
)
from pyqed.qchem.mcscf.reduced_ci import (
    ReducedCISubspace,
    ci_rotation_gradient,
    ci_rotation_hessian,
    orbital_ci_coupling,
)


def test_casci_accepts_closed_shell_uhf_reference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    rhf = RHF(mol).run()
    uhf = UHF(mol).run()

    mc_rhf = CASCI(rhf, ncas=2, nelecas=2).run(nstates=1)
    mc_uhf = CASCI(uhf, ncas=2, nelecas=2).run(nstates=1)

    np.testing.assert_allclose(mc_uhf.e_tot[0], mc_rhf.e_tot[0], atol=1e-8)
    np.testing.assert_allclose(
        mc_uhf.make_rdm1(0),
        mc_rhf.make_rdm1(0),
        atol=1e-8,
    )


def test_casci_is_exported_from_pyqed_qchem():
    from pyqed.qchem import CASCI as ExportedCASCI

    assert ExportedCASCI is CASCI


def test_molecule_casci_owns_scan_configuration():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    electronic = mol.casci(
        ncas=2,
        nelecas=2,
        nstates=2,
        method='ci',
        run_options={'use_cholesky': False},
    )

    assert isinstance(electronic, CASCI)
    assert electronic.ncas == 2
    assert electronic.nelecas == 2
    assert electronic.nstates == 2
    scanner = electronic.as_scanner()
    assert scanner.nstates == 2
    assert scanner.method == 'ci'
    assert scanner.run_kwargs == {'use_cholesky': False}


def test_molecule_casci_forwards_singlet_multiplicity():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    electronic = mol.casci(
        ncas=2,
        nelecas=2,
        nstates=2,
        ms2=0,
        multiplicity=1,
    ).run(nstates=2)

    assert electronic.ms2 == 0
    assert electronic.multiplicity == 1
    assert electronic.solver_backend.startswith('direct_spin0')
    np.testing.assert_allclose(
        [electronic.spin_square(root) for root in range(2)],
        0.0,
        atol=1e-10,
    )


def test_casci_scanner_root_homing_tracks_and_phase_aligns_states():
    atom = '\n'.join(f'H 0 0 {1.8 * i:.10f}' for i in range(4))
    mol = Molecule(atom=atom, unit='bohr', basis='sto-6g')
    mol.build()
    mf = RHF(mol).run()

    template = CASCI(mf, ncas=4, nelecas=4)
    template.direct_ci_dense_fallback_ndets = 1
    template.run(nstates=2, method='direct_ci')
    scanner = template.as_scanner(
        nstates=2,
        method='direct_ci',
        reuse_ci=True,
        root_homing=True,
        root_homing_cushion=2,
    )

    coords = np.asarray(mol.atom_coords(), dtype=float)
    scanner(coords)
    displaced = coords.copy()
    displaced[-1, 2] += 0.02
    tracked = scanner(displaced)

    assert tracked.nstates == 2
    assert tracked.root_tracking_permutation.shape == (2,)
    assert len(np.unique(tracked.root_tracking_permutation)) == 2
    assert np.all(tracked.root_tracking_overlaps > 0.8)
    if tracked.direct_ci_native_diagnostics.get('used'):
        assert tracked.direct_ci_native_diagnostics['used_initial_guess'] is True


def test_casci_ms2_multiplicity_selects_triplet_m0_root():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc_singlet = CASCI(mf, ncas=2, nelecas=2, ms2=0, multiplicity=1).run(
        nstates=1, method='direct_ci'
    )
    mc_triplet_m0 = CASCI(mf, ncas=2, nelecas=2, ms2=0, multiplicity=3).run(
        nstates=1, method='direct_ci'
    )
    mc_triplet_p1 = CASCI(mf, ncas=2, nelecas=2, ms2=2, multiplicity=3).run(
        nstates=1, method='direct_ci'
    )

    np.testing.assert_allclose(mc_singlet.spin_square(0), 0.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_m0.spin_square(0), 2.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_p1.spin_square(0), 2.0, atol=1e-10)
    np.testing.assert_allclose(mc_triplet_m0.e_tot[0], mc_triplet_p1.e_tot[0], atol=1e-10)
    assert mc_triplet_m0.ms2 == 0
    assert mc_triplet_m0.spin == 0
    assert mc_triplet_m0.multiplicity == 3


def test_casci_spin_alias_and_multiplicity_validation():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()

    mc = CASCI(mf, ncas=2, nelecas=2, spin=2)
    assert mc.ms2 == 2
    assert mc.spin == 2
    assert mc.nelecas_spin == (2, 0)

    with pytest.raises(ValueError, match='spin and ms2'):
        CASCI(mf, ncas=2, nelecas=2, spin=2, ms2=0)

    with pytest.raises(ValueError, match='incompatible'):
        CASCI(mf, ncas=2, nelecas=2, ms2=2, multiplicity=1)


def test_casci_make_rdm1_ao_representation_matches_manual_transform():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=1)

    dm_mo = mc.make_rdm1(0, with_core=True, with_vir=True, representation='mo')
    dm_mo_no_vir = mc.make_rdm1(0, with_core=True, representation='mo')
    dm_ao = mc.make_rdm1(0, with_core=True, with_vir=True, representation='ao')
    dm_ao_no_vir = mc.make_rdm1(0, with_core=True, representation='ao')
    dm_ao_alias = mc.make_rdm1(0, with_core=True, with_vir=True, repr='ao')

    expected = mf.mo_coeff @ dm_mo @ mf.mo_coeff.conj().T
    expected_no_vir = mf.mo_coeff[:, :dm_mo_no_vir.shape[0]] @ dm_mo_no_vir @ mf.mo_coeff[:, :dm_mo_no_vir.shape[0]].conj().T
    np.testing.assert_allclose(dm_ao, expected, atol=1e-12)
    np.testing.assert_allclose(dm_ao_no_vir, expected_no_vir, atol=1e-12)
    np.testing.assert_allclose(dm_ao_no_vir, expected, atol=1e-12)
    np.testing.assert_allclose(dm_ao_alias, expected, atol=1e-12)


def test_casci_make_tdm1_ao_representation_matches_manual_transform():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    tdm_active = mc.make_tdm1(0, 1)
    tdm_ao = mc.make_tdm1(0, 1, representation='ao')
    tdm_ao_with_core = mc.make_tdm1(0, 1, with_core=True, representation='ao')
    tdm_ao_alias = mc.make_tdm1(0, 1, repr='ao')

    active_coeff = mf.mo_coeff[:, mc.ncore:mc.ncore + mc.ncas]
    expected = active_coeff @ tdm_active @ active_coeff.conj().T
    np.testing.assert_allclose(tdm_ao, expected, atol=1e-12)
    np.testing.assert_allclose(tdm_ao_with_core, expected, atol=1e-12)
    np.testing.assert_allclose(tdm_ao_alias, expected, atol=1e-12)


def test_casci_transition_dipole_moment_matches_ao_tdm_contraction():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    center = np.zeros(3)
    dipole_ao = mf.dipole(center=center, basis='ao')

    dense = CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='ci')
    direct = CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='direct_ci')
    raw_direct = direct_ci.CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='direct_ci')

    tdm_ao = dense.make_tdm1(2, 0, representation='ao')
    expected = np.einsum('xij,ij->x', dipole_ao, tdm_ao, optimize=True)

    np.testing.assert_allclose(
        dense.transition_dipole_moment(2, 0, center=center),
        expected,
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(direct.transition_dipole_moment(2, 0, center=center)),
        np.abs(expected),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        np.abs(raw_direct.transition_dipole_moment(2, 0, center=center)),
        np.abs(expected),
        atol=1e-10,
    )

    all_dipoles = dense.transition_dipole_moment(center=center)
    assert all_dipoles.shape == (3, 3)
    np.testing.assert_allclose(all_dipoles[1], expected, atol=1e-10)
    np.testing.assert_allclose(dense.transition_dipole(center=center), all_dipoles)


def test_dipole_exponential_ci_overlap_is_orbital_rotation_link():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=4, nelecas=2).run(nstates=4, method='ci')
    center = np.zeros(3)
    state_ids = [0, 2]

    zero = dipole_exponential_ci_overlap(
        mc,
        0.0,
        axis='z',
        center=center,
        state_ids=state_ids,
    )
    np.testing.assert_allclose(zero, np.eye(len(state_ids)), atol=1e-12)

    eps = 2.0e-6
    plus = dipole_exponential_ci_overlap(
        mc,
        eps,
        axis='z',
        center=center,
        state_ids=state_ids,
    )
    minus = dipole_exponential_ci_overlap(
        mc,
        -eps,
        axis='z',
        center=center,
        state_ids=state_ids,
    )
    derivative = (plus - minus) / (2.0j * eps)
    expected = np.asarray(
        [
            [
                mc.transition_dipole_moment(bra, ket, center=center)[2]
                for ket in state_ids
            ]
            for bra in state_ids
        ],
        dtype=complex,
    )
    np.testing.assert_allclose(derivative, expected, atol=2.0e-10)

    unitary = dipole_orbital_rotation_unitary(mc, eps, axis='z', center=center)
    via_unitary = orbital_rotation_ci_overlap(mc, unitary, state_ids=state_ids)
    np.testing.assert_allclose(via_unitary, plus, atol=1e-12)


def test_casci_accepts_open_shell_uhf_reference():
    mol = Molecule(atom='H 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build()

    uhf = UHF(mol).run()
    mc = CASCI(uhf, ncas=1, nelecas=(1, 0)).run(nstates=1)

    np.testing.assert_allclose(mc.e_tot[0], uhf.e_tot, atol=1e-10)
    np.testing.assert_allclose(mc.make_rdm1(0), np.array([[1.0]]), atol=1e-10)
    assert str(mc.solver_backend).startswith('direct_ci')


def test_casci_make_rdm1s_matches_pyscf_for_open_shell_li():
    pyscf = pytest.importorskip('pyscf')
    mcscf = pytest.importorskip('pyscf.mcscf')

    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build()

    mf = UHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=(2, 1)).run(nstates=1)

    pmf = pyscf.scf.UHF(mol.topyscf())
    pmf.conv_tol = 1e-10
    pmf.kernel()
    pmc = mcscf.UCASCI(pmf, 2, (2, 1))
    pmc.kernel()

    dm1a, dm1b = mc.make_rdm1s(0)
    pdm1a, pdm1b = pmc.fcisolver.make_rdm1s(pmc.ci, 2, (2, 1))

    np.testing.assert_allclose(dm1a, np.array(pdm1a), atol=1e-8)
    np.testing.assert_allclose(dm1b, np.array(pdm1b), atol=1e-8)
    np.testing.assert_allclose(dm1a + dm1b, mc.make_rdm1(0), atol=1e-8)


def test_direct_ci_casci_accepts_open_shell_uhf_reference():
    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build()

    mf = UHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=(2, 1))
    mc.direct_ci_dense_fallback_ndets = 1
    mc.run(nstates=1, method='direct_ci')

    dm1a, dm1b = mc.make_rdm1s(0)
    np.testing.assert_allclose(dm1a + dm1b, mc.make_rdm1(0), atol=1e-8)
    assert np.isfinite(mc.e_tot[0])
    assert str(mc.solver_backend).startswith('direct_ci')
    assert mc.direct_ci_native_diagnostics['used'] is False
    assert 'restricted spin strings' in mc.direct_ci_fallback_reason
    assert mc.direct_ci_diagnostics['backend'] == 'python_davidson'


def test_direct_ci_casci_uhf_supports_factor_backend():
    mol = Molecule(atom='Li 0 0 0', unit='bohr', basis='sto-3g', spin=1)
    mol.build()

    mf = UHF(mol).run()
    mc_dense = direct_ci.CASCI(mf, ncas=4, nelecas=(2, 1)).run(nstates=2, method='ci')
    mc_factor = direct_ci.CASCI(mf, ncas=4, nelecas=(2, 1)).run(
        nstates=2,
        method='direct_ci',
        use_cholesky=True,
    )

    np.testing.assert_allclose(mc_factor.e_tot[0], mc_dense.e_tot[0], atol=1e-8)
    assert mc_factor.solver_backend == 'direct_ci_spin_string_uhf'
    assert mc_factor.direct_connectivity is None
    sigma = mc_factor.ci_sigma(mc_factor.ci[0])
    assert mc_factor.direct_connectivity is None
    np.testing.assert_allclose(
        np.vdot(mc_factor.ci[0], sigma),
        mc_factor.e_tot[0] - mc_factor.e_core,
        atol=1.0e-8,
    )


def test_casci_make_tdm1s_sum_to_spin_traced_tdm():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    tdm1a, tdm1b = mc.make_tdm1s(1, 0)

    np.testing.assert_allclose(tdm1a + tdm1b, mc.make_tdm1(1, 0), atol=1e-10)
    np.testing.assert_allclose(sum(mc.make_rdm1s(0)), mc.make_rdm1(0), atol=1e-10)


def test_casci_make_tdm2_diagonal_matches_rdm2():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    mc_direct = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(
        nstates=2,
        method='direct_ci',
    )

    np.testing.assert_allclose(mc.make_tdm2(0, 0), mc.make_rdm2(0), atol=1e-10)
    np.testing.assert_allclose(mc.make_tdm2(1, 1), mc.make_rdm2(1), atol=1e-10)
    np.testing.assert_allclose(
        mc_direct.make_tdm2(0, 0),
        mc_direct.make_rdm2(0),
        atol=1e-10,
    )


def test_casci_rdm2_reconstructs_active_space_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    for cls, method in ((CASCI, 'ci'), (direct_ci.CASCI, 'direct_ci')):
        mc = cls(mf, ncas=4, nelecas=4).run(nstates=1, method=method)
        dm1, dm2 = mc.make_rdm12(0)
        hcore = np.asarray(mc.hcore)
        h1e = hcore[0] if hcore.ndim == 3 else hcore
        eri = getattr(mc, 'h2e_cas', None)
        if eri is None:
            ncore, ncas = mc.ncore, mc.ncas
            active = slice(ncore, ncore + ncas)
            eri = mf.get_eri_mo(mf.mo_coeff, notation='chem')[
                active, active, active, active
            ]
        eri = np.asarray(eri)
        eri = eri[0, 0] if eri.ndim == 6 else eri

        e_reconstructed = (
            mc.e_core
            + np.einsum('pq,pq', h1e, dm1)
            + 0.5 * np.einsum('pqrs,pqrs', eri, dm2)
        )

        np.testing.assert_allclose(e_reconstructed, mc.e_tot[0], atol=1e-10)


def test_reduced_ci_subspace_projects_casci_roots():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0, 1])
    energies, ci = subspace.diagonalize(nroots=2)
    residuals = subspace.residuals(mc, ci, energies)

    assert subspace.nvec == 2
    np.testing.assert_allclose(energies, mc.e_tot[:2], atol=1e-10)
    np.testing.assert_allclose(subspace.basis.T @ subspace.basis, np.eye(2), atol=1e-10)
    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)


def test_direct_ci_sigma_matches_ci_roots():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    for root, energy in zip(mc.ci, mc.e_tot):
        sigma = mc.ci_sigma(root)
        np.testing.assert_allclose(sigma, (energy - mc.e_core) * root, atol=1e-8)
    hdiag = mc.ci_diagonal()
    assert hdiag.shape == (len(mc.ci[0]),)
    assert np.all(np.isfinite(hdiag))


def test_factorized_direct_ci_sigma_matches_ci_root():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(
        nstates=1,
        method='direct_ci',
        use_cholesky=True,
    )

    sigma = mc.ci_sigma(mc.ci[0])
    np.testing.assert_allclose(sigma, (mc.e_tot[0] - mc.e_core) * mc.ci[0], atol=1e-8)


def test_factorized_connection_sigma_fast_matches_numba_for_blocks():
    rng = np.random.default_rng(19)
    ndet = 7
    H_diag = rng.normal(size=ndet)
    links = {
        'A': (np.array([1, 4, 6], dtype=np.int32), np.array([0, 3, 2], dtype=np.int32)),
        'B': (np.array([0, 3], dtype=np.int32), np.array([5, 1], dtype=np.int32)),
        'AA': (np.array([2, 5], dtype=np.int32), np.array([6, 0], dtype=np.int32)),
        'BB': (np.array([6], dtype=np.int32), np.array([4], dtype=np.int32)),
        'AB': (np.array([1, 3, 5, 0], dtype=np.int32), np.array([2, 6, 4, 1], dtype=np.int32)),
    }
    H_A = rng.normal(size=len(links['A'][0]))
    H_B = rng.normal(size=len(links['B'][0]))
    H_AA = rng.normal(size=len(links['AA'][0]))
    H_BB = rng.normal(size=len(links['BB'][0]))
    H_AB = rng.normal(size=len(links['AB'][0]))
    I_A, J_A = links['A']
    I_B, J_B = links['B']
    I_AA, J_AA = links['AA']
    I_BB, J_BB = links['BB']
    I_AB, J_AB = links['AB']

    c = rng.normal(size=ndet)
    sigma_ref = direct_ci._sigma_values_conn_numba(
        H_diag, H_A, H_B, H_AA, H_BB, H_AB, c,
        I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
    )
    sigma_fast = direct_ci._sigma_values_conn_fast(
        H_diag, H_A, H_B, H_AA, H_BB, H_AB, c,
        I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
    )
    np.testing.assert_allclose(sigma_fast, sigma_ref, atol=1e-14)

    block = rng.normal(size=(ndet, 4))
    block_ref = np.column_stack([
        direct_ci._sigma_values_conn_numba(
            H_diag, H_A, H_B, H_AA, H_BB, H_AB, block[:, root],
            I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
        )
        for root in range(block.shape[1])
    ])
    block_fast = direct_ci._sigma_values_conn_fast(
        H_diag, H_A, H_B, H_AA, H_BB, H_AB, block,
        I_A, J_A, I_B, J_B, I_AA, J_AA, I_BB, J_BB, I_AB, J_AB,
    )
    np.testing.assert_allclose(block_fast, block_ref, atol=1e-14)

    if direct_ci._casscf_cpp is not None:
        assert hasattr(direct_ci._casscf_cpp, "sigma_values_conn")


def test_spin_string_direct_connectivity_matches_slow_builder_records():
    mo_occ = np.zeros((2, 4), dtype=np.int8)
    mo_occ[0, :2] = 1
    mo_occ[1, :2] = 1
    binary = direct_ci.get_fci_combos(mo_occ=mo_occ)

    fast = direct_ci._build_direct_connectivity_from_spin_strings(binary)
    slow = direct_ci._build_direct_connectivity_slow(binary)
    groups = [
        ('A', ('I_A', 'J_A', 'p_A', 'q_A', 'phase_A')),
        ('B', ('I_B', 'J_B', 'p_B', 'q_B', 'phase_B')),
        ('AA', ('I_AA', 'J_AA', 'p_AA', 'q_AA', 'r_AA', 's_AA', 'phase_AA')),
        ('BB', ('I_BB', 'J_BB', 'p_BB', 'q_BB', 'r_BB', 's_BB', 'phase_BB')),
        ('AB', ('I_AB', 'J_AB', 'p_AB', 'q_AB', 'r_AB', 's_AB', 'phase_AB')),
    ]

    for _, fields in groups:
        fast_records = np.column_stack([getattr(fast, field) for field in fields])
        slow_records = np.column_stack([getattr(slow, field) for field in fields])
        fast_order = np.lexsort(tuple(fast_records[:, col] for col in range(fast_records.shape[1] - 1, -1, -1)))
        slow_order = np.lexsort(tuple(slow_records[:, col] for col in range(slow_records.shape[1] - 1, -1, -1)))
        np.testing.assert_array_equal(fast_records[fast_order], slow_records[slow_order])


def test_fci_string_basis_matches_expanded_determinant_ordering():
    mo_occ = np.zeros((2, 5), dtype=np.int8)
    mo_occ[0, :2] = 1
    mo_occ[1, :1] = 1
    basis = direct_ci.get_fci_string_basis(mo_occ)
    expanded = direct_ci.get_fci_combos(mo_occ=mo_occ)

    assert isinstance(basis, direct_ci.FCIStringBasis)
    assert basis.shape == expanded.shape
    assert basis.nbytes < expanded.nbytes
    np.testing.assert_array_equal(np.asarray(basis), expanded)
    np.testing.assert_array_equal(basis[7], expanded[7])
    np.testing.assert_array_equal(basis[[1, 5, 8], 1, :], expanded[[1, 5, 8], 1, :])

    rng = np.random.default_rng(73)
    h1 = rng.normal(size=(5, 5))
    eri_same = rng.normal(size=(5, 5, 5, 5))
    eri_cross = rng.normal(size=(5, 5, 5, 5))
    np.testing.assert_allclose(
        direct_ci._compute_diag_compact(h1, eri_same, eri_cross, basis),
        direct_ci._compute_diag_compact(h1, eri_same, eri_cross, expanded),
        atol=1.0e-13,
    )


def test_spin_orbital_diagonal_accepts_fci_string_basis():
    mo_occ = np.zeros((2, 5), dtype=np.int8)
    mo_occ[0, :2] = 1
    mo_occ[1, :1] = 1
    basis = direct_ci.get_fci_string_basis(mo_occ)
    expanded = basis.materialize()

    rng = np.random.default_rng(79)
    h1 = rng.normal(size=(2, 5, 5))
    h2 = rng.normal(size=(2, 2, 5, 5, 5, 5))

    np.testing.assert_allclose(
        direct_ci._compute_diag(h1, h2, basis),
        direct_ci._compute_diag(h1, h2, expanded),
        atol=1.0e-13,
    )


def test_large_direct_ci_retains_compact_string_basis_through_common_operations():
    atom = '\n'.join(f'H 0 0 {1.8 * i:.10f}' for i in range(6))
    mol = Molecule(atom=atom, unit='bohr', basis='sto-6g', spin=0)
    mol.build()
    mf = RHF(mol).run()

    mc = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    mc.direct_ci_dense_fallback_ndets = 1
    mc.run(nstates=1, method='direct_ci')
    basis = mc.binary

    assert isinstance(basis, direct_ci.FCIStringBasis)
    assert basis.nbytes < basis.shape[0] * basis.shape[1] * basis.shape[2]
    sigma = mc.ci_sigma(mc.ci[0])
    np.testing.assert_allclose(
        sigma,
        (mc.e_tot[0] - mc.e_core) * mc.ci[0],
        atol=2.0e-5,
    )
    assert abs(mc.spin_square(0)) < 1.0e-6
    dm1 = mc.make_rdm1(0)
    assert np.isclose(np.trace(dm1), 6.0)
    dm2 = mc.make_rdm2(0)
    assert dm2.shape == (6, 6, 6, 6)
    assert mc.binary is basis


def test_reduced_ci_full_subspace_reproduces_dense_casci():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=2, method='direct_ci')

    identity = np.eye(len(mc.ci[0]))
    subspace = ReducedCISubspace.from_casci(
        mc,
        root_ids=[],
        extra_vectors=identity,
    )
    energies, ci = subspace.diagonalize(nroots=2)
    residuals = subspace.residuals(mc, ci, energies)

    assert subspace.nvec == identity.shape[0]
    np.testing.assert_allclose(energies, mc.e_tot[:2], atol=1e-10)
    np.testing.assert_allclose(residuals, np.zeros_like(residuals), atol=1e-8)


def test_reduced_ci_rotation_block_for_root_plus_external_vector():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    extra = np.eye(len(mc.ci[0]))[:, 0]
    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0], extra_vectors=extra)
    grad, grad_pairs = subspace.rotation_gradient(nstates=1)
    hess, hess_pairs = subspace.rotation_hessian(nstates=1)
    grad_func, grad_pairs_func = ci_rotation_gradient(subspace.hamiltonian, nstates=1)
    hess_func, hess_pairs_func = ci_rotation_hessian(subspace.hamiltonian, nstates=1)

    assert grad_pairs == hess_pairs
    assert grad_pairs == grad_pairs_func
    assert hess_pairs == hess_pairs_func
    assert grad.shape == (subspace.nvec - 1,)
    assert hess.shape == (subspace.nvec - 1, subspace.nvec - 1)
    np.testing.assert_allclose(grad, grad_func, atol=1e-12)
    np.testing.assert_allclose(hess, hess_func, atol=1e-12)
    np.testing.assert_allclose(grad, np.zeros_like(grad), atol=1e-8)
    np.testing.assert_allclose(hess, hess.T, atol=1e-12)

    rotated = subspace.rotated_state_vectors(np.zeros(len(grad_pairs)), grad_pairs)
    np.testing.assert_allclose(rotated[:, 0], subspace.basis[:, 0], atol=1e-12)

    step = np.zeros(len(grad_pairs))
    step[0] = 0.1
    rotated = subspace.rotated_state_vectors(step, grad_pairs)
    assert rotated.shape == (subspace.ndet, 1)
    np.testing.assert_allclose(rotated.T @ rotated, np.eye(1), atol=1e-12)


def test_reduced_ci_expands_with_preconditioned_residual():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=1, method='direct_ci')

    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0])
    trial = mc.ci[0].copy()
    trial += 0.1 * np.eye(len(trial))[:, 0]
    trial /= np.linalg.norm(trial)
    energy = subspace.rayleigh_energies(mc, trial)[0]
    expanded, n_added = subspace.expand_with_residuals(
        mc,
        trial,
        np.array([energy]),
        max_vectors=1,
    )

    assert n_added == 1
    assert expanded.nvec == subspace.nvec + 1
    np.testing.assert_allclose(expanded.basis.T @ expanded.basis, np.eye(expanded.nvec), atol=1e-10)


def test_orbital_ci_coupling_matches_ci_finite_difference():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2).run(nstates=1, method='direct_ci')

    extra = np.eye(len(mc.ci[0]))[:, 0]
    subspace = ReducedCISubspace.from_casci(mc, root_ids=[0], extra_vectors=extra)
    coupling, pairs = subspace.orbital_coupling(
        mc,
        mf.get_hcore_mo(mf.mo_coeff),
        mf.get_eri_mo(mf.mo_coeff, notation='chem'),
        nstates=1,
        nmo=mf.nmo,
    )
    coupling_func, pairs_func = orbital_ci_coupling(
        mc,
        subspace,
        mf.get_hcore_mo(mf.mo_coeff),
        mf.get_eri_mo(mf.mo_coeff, notation='chem'),
        nstates=1,
        nmo=mf.nmo,
    )
    assert coupling.shape[1] == len(pairs)
    assert pairs == pairs_func
    np.testing.assert_allclose(coupling, coupling_func, atol=1e-12)

    p, m = pairs[0]
    eps = 1.0e-5
    c0 = subspace.basis[:, m]
    cp = subspace.basis[:, p]
    c_plus = c0 + eps * cp
    c_plus /= np.linalg.norm(c_plus)
    c_minus = c0 - eps * cp
    c_minus /= np.linalg.norm(c_minus)
    ci_saved = mc.ci
    try:
        mc.ci = [c_plus]
        dm1_plus = mc.make_rdm1(0, with_core=True, with_vir=True)
        dm2_plus = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
        grad_plus = orbital_gradient(
            generalized_fock(
                mf.get_hcore_mo(mf.mo_coeff),
                mf.get_eri_mo(mf.mo_coeff, notation='chem'),
                dm1_plus,
                dm2_plus,
            )
        )

        mc.ci = [c_minus]
        dm1_minus = mc.make_rdm1(0, with_core=True, with_vir=True)
        dm2_minus = embed_rdm2(mc.make_rdm2(0, with_core=True), mf.nmo)
        grad_minus = orbital_gradient(
            generalized_fock(
                mf.get_hcore_mo(mf.mo_coeff),
                mf.get_eri_mo(mf.mo_coeff, notation='chem'),
                dm1_minus,
                dm2_minus,
            )
        )
    finally:
        mc.ci = ci_saved

    fd_col = pack_nonredundant(
        (grad_plus - grad_minus) / (2.0 * eps),
        mc.ncore,
        mc.ncas,
        mf.nmo,
    )
    np.testing.assert_allclose(coupling[:, 0], fd_col, atol=1e-5, rtol=1e-5)


def test_bo_hamiltonian_derivatives_match_manual_tdm_contractions():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    cart = 2
    tdm1 = mc.make_tdm1(1, 0)
    tdm2 = mc.make_tdm2(1, 0)
    overlap = np.vdot(mc.ci[1], mc.ci[0])
    manual_offdiag_f = (
        terms.core_gradient_cartesian[cart] * overlap
        + np.einsum("pq,qp->", terms.h1_mo_cartesian[cart], tdm1, optimize=True)
        + 0.5 * np.einsum("pqrs,pqrs->", terms.eri1_mo_cartesian[cart], tdm2, optimize=True)
    )
    rdm1 = mc.make_rdm1(0)
    rdm2 = mc.make_rdm2(0)
    manual_diag_f = (
        terms.core_gradient_cartesian[cart]
        + np.einsum("pq,qp->", terms.h1_mo_cartesian[cart], rdm1, optimize=True)
        + 0.5 * np.einsum("pqrs,pqrs->", terms.eri1_mo_cartesian[cart], rdm2, optimize=True)
    )

    np.testing.assert_allclose(terms.F_cartesian[cart, 1, 0], manual_offdiag_f, atol=1e-10)
    np.testing.assert_allclose(terms.F_cartesian[cart, 0, 0], manual_diag_f, atol=1e-10)

    manual_offdiag_g = (
        terms.core_hessian_cartesian[cart, cart] * overlap
        + np.einsum("pq,qp->", terms.h2_mo_cartesian[cart, cart], tdm1, optimize=True)
        + 0.5 * np.einsum(
            "pqrs,pqrs->",
            terms.eri2_mo_cartesian[cart, cart],
            tdm2,
            optimize=True,
        )
    )
    manual_diag_g = (
        terms.core_hessian_cartesian[cart, cart]
        + np.einsum("pq,qp->", terms.h2_mo_cartesian[cart, cart], rdm1, optimize=True)
        + 0.5 * np.einsum(
            "pqrs,pqrs->",
            terms.eri2_mo_cartesian[cart, cart],
            rdm2,
            optimize=True,
        )
    )

    np.testing.assert_allclose(terms.G_cartesian[cart, cart, 1, 0], manual_offdiag_g, atol=1e-10)
    np.testing.assert_allclose(terms.G_cartesian[cart, cart, 0, 0], manual_diag_g, atol=1e-10)


def test_bo_hamiltonian_derivatives_projection_matches_cartesian_contraction():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1], mode_vectors=mode)
    mode_flat = mode.reshape(1, -1)

    expected_f = np.einsum('ka,aij->kij', mode_flat, terms.F_cartesian, optimize=True)
    expected_g = np.einsum(
        'ka,lb,abij->klij',
        mode_flat,
        mode_flat,
        terms.G_cartesian,
        optimize=True,
    )

    np.testing.assert_allclose(terms.F_projected, expected_f, atol=1e-10)
    np.testing.assert_allclose(terms.G_projected, expected_g, atol=1e-10)


def test_bo_hamiltonian_derivatives_projected_only_matches_cartesian_projection():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    full = bo_hamiltonian_derivatives(mc, state_ids=[0, 1], mode_vectors=mode)
    projected = bo_hamiltonian_derivatives(
        mc,
        state_ids=[0, 1],
        mode_vectors=mode,
        projected_only=True,
    )

    assert projected.F_cartesian is None
    assert projected.G_cartesian is None
    np.testing.assert_allclose(projected.F_projected, full.F_projected, atol=1.0e-8)
    np.testing.assert_allclose(projected.G_projected, full.G_projected, atol=1.0e-8)


def test_casci_vibronic_couplings_return_projected_f_and_g():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    f, g, terms = mc.vibronic_couplings(
        state_ids=[0, 1],
        modes=mode,
        return_terms=True,
    )

    assert f.shape == (2, 2, 1)
    assert g.shape == (2, 2, 1, 1)
    np.testing.assert_allclose(f, np.moveaxis(terms.F_projected, 0, -1), atol=1e-10)
    np.testing.assert_allclose(
        g,
        np.moveaxis(terms.G_projected, (0, 1), (-2, -1)),
        atol=1e-10,
    )


def test_casci_vibronic_gradients_match_full_first_order_terms():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    mode = np.zeros((1, mol.natom, 3))
    mode[0, 0, 2] = -1.0
    mode[0, 1, 2] = 1.0

    full = bo_hamiltonian_derivatives(
        mc,
        state_ids=[0, 1],
        mode_vectors=mode,
        projected_only=True,
    )
    gradients, first = mc.vibronic_gradients(
        state_ids=[0, 1],
        modes=mode,
        return_terms=True,
    )

    np.testing.assert_allclose(first.F_projected, full.F_projected, atol=1e-10)
    np.testing.assert_allclose(
        gradients,
        np.moveaxis(full.F_projected, 0, -1),
        atol=1e-10,
    )
    assert first.G_projected is None
    assert first.h2_ao_cartesian is None
    assert first.eri2_mo_cartesian is None


def test_casci_vibronic_hessian_is_symmetric_between_two_modes():
    mol = Molecule(
        atom="H 0 0 0; H 1.4 0 0; H 0.2 1.3 0",
        unit="bohr",
        basis="sto-3g",
        charge=1,
    )
    mol.build(eri="dense")

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=3, nelecas=2).run(nstates=3)
    modes = np.zeros((2, mol.natom, 3))
    modes[0, 0, 0] = -1.0
    modes[0, 1, 0] = 1.0
    modes[1, 0, 1] = -1.0
    modes[1, 2, 1] = 1.0

    _, g = mc.vibronic_couplings(state_ids=[1, 2], modes=modes)

    np.testing.assert_allclose(g, g.swapaxes(-1, -2), atol=1e-10)


def test_casci_vibronic_couplings_modes_are_cartesian_displacement_coefficients():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    mass_weighted = np.zeros((1, mol.natom, 3))
    mass_weighted[0, 0, 2] = -1.0
    mass_weighted[0, 1, 2] = 1.0
    masses = np.asarray(mol.atom_mass_list(), dtype=float)
    cartesian_modes = mass_weighted / np.sqrt(masses)[:, None]

    f, _ = mc.vibronic_couplings(state_ids=[0, 1], modes=cartesian_modes)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    expected = np.einsum(
        "mAx,Axab->abm",
        cartesian_modes,
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        optimize=True,
    )
    wrong_mass_weighted = np.einsum(
        "mAx,Axab->abm",
        mass_weighted,
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        optimize=True,
    )

    np.testing.assert_allclose(f, expected, rtol=1e-5, atol=1e-9)
    assert not np.allclose(f, wrong_mass_weighted, atol=1e-8)


def test_casci_vibronic_couplings_return_cartesian_f_and_g_without_modes():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    f, g, terms = mc.vibronic_couplings(state_ids=[0, 1], return_terms=True)

    assert f.shape == (2, 2, mol.natom, 3)
    assert g.shape == (2, 2, mol.natom, 3, mol.natom, 3)
    expected_f = np.moveaxis(
        terms.F_cartesian.reshape(mol.natom, 3, 2, 2),
        (0, 1),
        (-2, -1),
    )
    expected_g = np.moveaxis(
        terms.G_cartesian.reshape(mol.natom, 3, mol.natom, 3, 2, 2),
        (0, 1, 2, 3),
        (-4, -3, -2, -1),
    )
    np.testing.assert_allclose(f, expected_f, atol=1e-10)
    np.testing.assert_allclose(g, expected_g, atol=1e-10)


def test_bo_hamiltonian_derivatives_hcore_ao_terms_match_finite_difference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    coords = np.asarray(mol.atom_coords(), dtype=float)

    def hcore_at(test_coords):
        atom = [
            [mol.atom_symbol(atom_id), *test_coords[atom_id]]
            for atom_id in range(mol.natom)
        ]
        displaced = Molecule(atom=atom, unit="bohr", basis="sto-3g")
        displaced.build(eri="dense")
        return displaced.hcore

    atom_id = 1
    axis = 2
    cart = 3 * atom_id + axis
    delta = 1e-4

    coords_plus = coords.copy()
    coords_minus = coords.copy()
    coords_plus[atom_id, axis] += delta
    coords_minus[atom_id, axis] -= delta

    h0 = hcore_at(coords)
    h_plus = hcore_at(coords_plus)
    h_minus = hcore_at(coords_minus)

    fd_first = (h_plus - h_minus) / (2.0 * delta)
    fd_second = (h_plus - 2.0 * h0 + h_minus) / (delta ** 2)

    np.testing.assert_allclose(
        terms.h1_ao_cartesian[cart].real,
        fd_first,
        atol=1e-6,
        rtol=1e-5,
    )
    np.testing.assert_allclose(
        terms.h2_ao_cartesian[cart, cart].real,
        fd_second,
        atol=2e-4,
        rtol=2e-4,
    )
    assert np.isfinite(terms.F_cartesian).all()
    assert np.isfinite(terms.G_cartesian).all()


def test_bo_hamiltonian_derivatives_work_with_builtin_factor_only_reference():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(options={'eri_representation': 'factors'})

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)
    terms = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])

    assert terms.h1_ao_cartesian.shape == (3 * mol.natom, mol.nao, mol.nao)
    assert terms.h2_ao_cartesian.shape == (3 * mol.natom, 3 * mol.natom, mol.nao, mol.nao)
    assert np.isfinite(terms.F_cartesian).all()
    assert np.isfinite(terms.G_cartesian).all()


def test_bo_hamiltonian_derivatives_backward_aliases_remain_available():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(eri='dense')

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    terms_new = bo_hamiltonian_derivatives(mc, state_ids=[0, 1])
    from pyqed.qchem.geometric import build_casci_geometric_fg_terms
    terms_old = build_casci_geometric_fg_terms(mc, state_ids=[0, 1])

    assert isinstance(terms_new, BOHamiltonianDerivatives)
    assert isinstance(terms_old, GeometricFGTerms)
    np.testing.assert_allclose(terms_new.F_cartesian, terms_old.F_cartesian, atol=1e-10)
    np.testing.assert_allclose(terms_new.G_cartesian, terms_old.G_cartesian, atol=1e-10)


def test_public_casci_defaults_to_direct_ci():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)

    assert mc.solver_backend in {
        'ci_dense_fallback',
        'direct_ci_spin_string',
        'direct_ci_compact_conn',
    }


def test_direct_ci_falls_back_to_dense_solver_for_small_spaces():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')
    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    assert mc_direct.solver_backend == 'ci_dense_fallback'
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-10)


def test_direct_ci_on_the_fly_matches_dense_solver():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci')

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    assert mc_direct.solver_backend == 'direct_ci_spin_string'
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-10)
    workspace = mc_direct._direct_rhf_blas_matvec
    if workspace is not None:
        vector = np.random.default_rng(42).normal(size=mc_direct.binary.shape[0])
        fast_sigma = mc_direct.ci_sigma(vector)
        mc_direct._direct_rhf_blas_matvec = None
        try:
            reference_sigma = mc_direct.ci_sigma(vector)
        finally:
            mc_direct._direct_rhf_blas_matvec = workspace
        np.testing.assert_allclose(fast_sigma, reference_sigma, atol=1.0e-12)


def test_direct_ci_davidson_matches_eigsh():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_davidson = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_davidson.direct_ci_dense_fallback_ndets = 1
    mc_davidson.direct_ci_eigensolver = 'davidson'
    mc_davidson.run(nstates=2, method='direct_ci')

    mc_eigsh = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_eigsh.direct_ci_dense_fallback_ndets = 1
    mc_eigsh.direct_ci_eigensolver = 'eigsh'
    mc_eigsh.run(nstates=2, method='direct_ci')

    np.testing.assert_allclose(mc_davidson.e_tot, mc_eigsh.e_tot, atol=1e-10)


def test_python_davidson_supports_complex_hermitian_hamiltonians():
    rng = np.random.default_rng(2026)
    raw = rng.normal(size=(24, 24)) + 1j * rng.normal(size=(24, 24))
    hamiltonian = 0.03 * (raw + raw.conj().T)
    hamiltonian += np.diag(np.linspace(-2.0, 3.0, 24))
    reference, _ = np.linalg.eigh(hamiltonian)
    guess = rng.normal(size=(24, 4)) + 1j * rng.normal(size=(24, 4))

    energies, vectors, diagnostics = direct_ci.davidson_lowest(
        lambda vector: hamiltonian @ vector,
        np.real(np.diag(hamiltonian)),
        nroots=4,
        tol=1.0e-10,
        energy_tol=1.0e-11,
        max_cycle=100,
        max_subspace=24,
        guess=guess,
        return_info=True,
    )

    np.testing.assert_allclose(energies, reference[:4], atol=1.0e-10)
    np.testing.assert_allclose(
        vectors.conj().T @ vectors,
        np.eye(4),
        atol=1.0e-10,
    )
    assert diagnostics['converged'] is True
    assert np.max(diagnostics['residual_norms']) < 1.0e-10


def test_native_rhf_davidson_matches_python_multiroot_and_supports_restart():
    atom = '\n'.join(f'H 0 0 {1.8 * i:.10f}' for i in range(6))
    mol = Molecule(atom=atom, unit='bohr', basis='sto-6g', spin=0)
    mol.build()
    mf = RHF(mol).run()

    python_solver = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    python_solver.direct_ci_dense_fallback_ndets = 1
    python_solver.direct_ci_root_cushion = 0
    python_solver.direct_ci_native_davidson = False
    python_solver.run(nstates=1, method='direct_ci')
    assert python_solver.direct_ci_fallback_reason == 'native Davidson disabled by solver setting'
    assert python_solver.direct_ci_diagnostics['backend'] == 'python_davidson'

    native_solver = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    native_solver.direct_ci_dense_fallback_ndets = 1
    native_solver.direct_ci_root_cushion = 0
    native_solver.run(nstates=1, method='direct_ci')

    np.testing.assert_allclose(native_solver.e_tot, python_solver.e_tot, atol=2.0e-9)
    if native_solver._direct_ci_used_native_davidson:
        assert direct_ci._cpp_attr('davidson_rhf_workspace') is not None
        capabilities = direct_ci.direct_ci_capabilities()
        assert capabilities['native_extension'] is True
        assert capabilities['cblas'] is True
        assert capabilities['rhf_multiroot'] is True
        diagnostics = native_solver.direct_ci_native_diagnostics
        assert diagnostics['converged'] is True
        assert diagnostics['iterations'] >= 1
        assert diagnostics['fallback_reason'] is None
        assert len(diagnostics['residual_norms']) == 1

    python_multiroot = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    python_multiroot.direct_ci_dense_fallback_ndets = 1
    python_multiroot.direct_ci_root_cushion = 0
    python_multiroot.direct_ci_native_davidson = False
    python_multiroot.run(nstates=2, method='direct_ci')

    native_multiroot = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    native_multiroot.direct_ci_dense_fallback_ndets = 1
    native_multiroot.direct_ci_root_cushion = 0
    native_multiroot.run(nstates=2, method='direct_ci')

    np.testing.assert_allclose(
        native_multiroot.e_tot,
        python_multiroot.e_tot,
        atol=5.0e-9,
    )
    if native_solver._direct_ci_used_native_davidson:
        assert native_multiroot._direct_ci_used_native_davidson is True

    eigsh_four = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    eigsh_four.direct_ci_dense_fallback_ndets = 1
    eigsh_four.direct_ci_eigensolver = 'eigsh'
    eigsh_four.run(nstates=4, method='direct_ci')

    native_four = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    native_four.direct_ci_dense_fallback_ndets = 1
    native_four.run(nstates=4, method='direct_ci')

    np.testing.assert_allclose(native_four.e_tot, eigsh_four.e_tot, atol=2.0e-8)
    if native_solver._direct_ci_used_native_davidson:
        assert native_four._direct_ci_used_native_davidson is True

    restarted = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    restarted.direct_ci_dense_fallback_ndets = 1
    restarted.direct_ci_root_cushion = 0
    restarted.run(nstates=2, method='direct_ci', ci0=python_multiroot.ci)
    if native_solver._direct_ci_used_native_davidson:
        assert restarted._direct_ci_used_native_davidson is True
        assert restarted.direct_ci_native_diagnostics['used_initial_guess'] is True

    complex_restart = direct_ci.CASCI(mf, ncas=6, nelecas=6, tol=1.0e-8)
    complex_restart.direct_ci_dense_fallback_ndets = 1
    complex_restart.direct_ci_root_cushion = 0
    complex_restart.run(
        nstates=2,
        method='direct_ci',
        ci0=[1j * state for state in python_multiroot.ci],
    )
    assert complex_restart._direct_ci_used_native_davidson is False
    assert 'complex CI guess' in complex_restart.direct_ci_fallback_reason
    np.testing.assert_allclose(complex_restart.e_tot, python_multiroot.e_tot, atol=5.0e-9)


def test_direct_spin0_symm_davidson_matches_dense_spin0():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_dense = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_dense.run(nstates=3, method='direct_spin0_symm')

    mc_davidson = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_davidson.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc_davidson.run(nstates=3, method='direct_spin0_symm')

    assert mc_dense.solver_backend == 'direct_spin0_symm_dense'
    assert mc_davidson.solver_backend == 'direct_spin0_symm_davidson_spin0_pair'
    assert len(mc_davidson.spin0_pair_indices) == 21
    np.testing.assert_allclose(mc_davidson.e_tot, mc_dense.e_tot, atol=1e-10)

    mc_spin_string = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_spin_string.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc_spin_string.direct_spin0_native_pair = False
    mc_spin_string.run(nstates=3, method='direct_spin0_symm')

    assert mc_spin_string.solver_backend == 'direct_spin0_symm_davidson_spin_string'
    np.testing.assert_allclose(mc_spin_string.e_tot, mc_dense.e_tot, atol=1e-10)

    mc_native_davidson = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_native_davidson.direct_spin0_symm_dense_fallback_nconfigs = 0
    assert mc_native_davidson.direct_spin0_native_davidson is True
    mc_native_davidson.direct_ci_residual_tol = 1.0e-9
    mc_native_davidson.run(nstates=3, method='direct_spin0_symm')

    assert mc_native_davidson.solver_backend == 'direct_spin0_symm_davidson_spin0_pair'
    if direct_ci._cpp_attr("davidson_spin0_pair") is not None:
        capabilities = direct_ci.direct_ci_capabilities()
        assert capabilities['spin0_multiroot'] is True
        assert capabilities['spin0_initial_guess'] is True
        assert mc_native_davidson.direct_ci_native_diagnostics['used'] is True
        assert mc_native_davidson.direct_ci_native_diagnostics['nroots'] == 3
    np.testing.assert_allclose(mc_native_davidson.e_tot, mc_dense.e_tot, atol=1e-9)

    mc_native_restart = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_native_restart.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc_native_restart.direct_ci_residual_tol = 1.0e-9
    mc_native_restart.run(
        nstates=3,
        method='direct_spin0_symm',
        ci0=mc_native_davidson.ci,
    )
    if direct_ci._cpp_attr("davidson_spin0_pair") is not None:
        assert mc_native_restart.direct_ci_native_diagnostics['used'] is True
        assert mc_native_restart.direct_ci_native_diagnostics['used_initial_guess'] is True
    np.testing.assert_allclose(mc_native_restart.e_tot, mc_dense.e_tot, atol=1e-9)

    mc_python_davidson = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_python_davidson.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc_python_davidson.direct_spin0_native_davidson = False
    mc_python_davidson.direct_ci_workers = 4
    mc_python_davidson.run(nstates=2, method='direct_spin0_symm')

    np.testing.assert_allclose(
        mc_python_davidson.e_tot,
        mc_dense.e_tot[:2],
        atol=1e-10,
    )

    mc_public = CASCI(mf, ncas=4, nelecas=4)
    mc_public.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc_public.run(nstates=2, method='direct_spin0_symm')

    assert mc_public.solver_backend == 'direct_spin0_symm_davidson_spin0_pair'
    np.testing.assert_allclose(mc_public.e_tot, mc_dense.e_tot[:2], atol=1e-10)


def test_direct_spin0_symm_fix_spin_keeps_requested_root_count(monkeypatch):
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    mc.direct_spin0_symm_dense_fallback_nconfigs = 0
    mc.fix_spin(ss=0, shift=0.2)
    solve_calls = []

    def fake_solve(binary, nstates, **kwargs):
        solve_calls.append(nstates)
        mc.e_core = 0.0
        mc.solver_backend = 'fake_spin0'
        mc.direct_ci_diagnostics = {'spin_penalty_shift': mc.shift}
        vectors = np.eye(binary.shape[0], nstates)
        return np.arange(nstates, dtype=float), vectors

    monkeypatch.setattr(mc, '_direct_spin0_symm_solve', fake_solve)
    mc.run(nstates=3, method='direct_spin0_symm')

    assert solve_calls == [3]
    assert len(mc.e_tot) == 3
    assert mc.direct_ci_diagnostics['requested_nstates'] == 3
    assert mc.direct_ci_diagnostics['solved_nstates'] == 3
    assert mc.direct_ci_diagnostics['multiplicity_selected'] is False
    assert mc.direct_ci_diagnostics['spin_penalty_shift'] == 0.2


def test_direct_spin0_symm_fix_spin_native_is_spin_pure():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()
    reference = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    reference.direct_spin0_symm_dense_fallback_nconfigs = 0
    reference.run(nstates=6, method='direct_spin0_symm')

    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    mc.fix_spin(ss=0, shift=1.0)
    mc.run(nstates=6, method='direct_spin0_symm')

    assert mc.solver_backend == 'direct_spin0_symm_davidson_spin0_pair'
    assert mc.direct_ci_diagnostics['requested_nstates'] == 6
    assert mc.direct_ci_diagnostics['solved_nstates'] == 6
    assert mc.direct_ci_diagnostics['spin_penalty_shift'] == 1.0
    assert mc.direct_ci_diagnostics['spin_penalty_application'] == 'separate_native_operator'
    assert mc.direct_ci_native_diagnostics['used'] is True
    np.testing.assert_allclose(mc.hcore, reference.hcore, atol=1.0e-12)
    np.testing.assert_allclose(mc.h2e_cas, reference.h2e_cas, atol=1.0e-12)
    np.testing.assert_allclose(mc.e_tot, reference.e_tot, atol=1.0e-9)
    np.testing.assert_allclose(
        [mc.spin_square(root) for root in range(6)],
        0.0,
        atol=1.0e-9,
    )


def test_direct_spin0_symm_separate_penalty_removes_even_spin_contaminants():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()

    reference = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    reference.run(nstates=21, method='direct_spin0_symm')
    reference_singlets = [
        energy
        for root, energy in enumerate(reference.e_tot)
        if abs(reference.spin_square(root)) < 1.0e-9
    ]

    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    mc.fix_spin(ss=0, shift=1.0)
    mc.run(nstates=12, method='direct_spin0_symm')

    assert len(reference_singlets) >= 12
    assert mc.direct_ci_diagnostics['spin_penalty_application'] == 'separate_native_operator'
    np.testing.assert_allclose(mc.e_tot, reference_singlets[:12], atol=1.0e-9)
    np.testing.assert_allclose(
        [mc.spin_square(root) for root in range(12)],
        0.0,
        atol=1.0e-9,
    )


def test_direct_spin0_symm_separate_penalty_python_fallback_matches_native():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()

    native = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    native.fix_spin(ss=0, shift=1.0)
    native.run(nstates=6, method='direct_spin0_symm')

    fallback = direct_ci.CASCI(mf, ncas=4, nelecas=4, multiplicity=1)
    fallback.direct_spin0_native_pair = False
    fallback.direct_spin0_native_davidson = False
    fallback.fix_spin(ss=0, shift=1.0)
    fallback.run(nstates=6, method='direct_spin0_symm')

    assert fallback.direct_ci_diagnostics['spin_penalty_application'] == 'separate_operator'
    np.testing.assert_allclose(fallback.e_tot, native.e_tot, atol=1.0e-9)
    np.testing.assert_allclose(
        [fallback.spin_square(root) for root in range(6)],
        0.0,
        atol=1.0e-9,
    )


def test_direct_ci_defaults_to_davidson():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()
    mc = direct_ci.CASCI(mf, ncas=4, nelecas=4)

    assert mc.direct_ci_eigensolver == 'davidson'
    assert mc.direct_ci_root_cushion == 2


def test_direct_ci_residual_tolerance_defaults_to_sqrt_energy_tolerance():
    solver = SimpleNamespace(tol=1.0e-10, direct_ci_residual_tol=None)
    energy_tol, residual_tol = direct_ci._resolve_direct_ci_tolerances(solver)

    assert energy_tol == 1.0e-10
    assert residual_tol == 1.0e-5

    solver.direct_ci_residual_tol = 2.0e-7
    assert direct_ci._resolve_direct_ci_tolerances(solver) == (1.0e-10, 2.0e-7)


def test_direct_ci_auto_eigensolver_uses_eigsh_for_medium_spaces():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_auto = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_auto.direct_ci_dense_fallback_ndets = 1
    mc_auto.direct_ci_eigensolver = 'auto'
    mc_auto.direct_ci_auto_eigsh_ndets = 1000
    mc_auto.run(nstates=2, method='direct_ci')

    mc_eigsh = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_eigsh.direct_ci_dense_fallback_ndets = 1
    mc_eigsh.direct_ci_eigensolver = 'eigsh'
    mc_eigsh.run(nstates=2, method='direct_ci')

    assert mc_auto.direct_ci_eigensolver == 'auto'
    np.testing.assert_allclose(mc_auto.e_tot, mc_eigsh.e_tot, atol=1e-10)


def test_direct_ci_root_cushion_recovers_lowest_h8_roots():
    distance = 1.8
    atom = '\n'.join(f'H 0 0 {i * distance:.10f}' for i in range(8))

    mol = Molecule(atom=atom, unit='bohr', basis='sto-6g', spin=0)
    mol.build()

    mf = RHF(mol).run()

    mc_cushion = direct_ci.CASCI(mf, ncas=8, nelecas=8)
    mc_cushion.direct_ci_dense_fallback_ndets = 1
    mc_cushion.direct_ci_eigensolver = 'davidson'
    mc_cushion.direct_ci_root_cushion = 2
    mc_cushion.run(nstates=10, method='direct_ci')

    mc_reference = direct_ci.CASCI(mf, ncas=8, nelecas=8)
    mc_reference.direct_ci_dense_fallback_ndets = 1
    mc_reference.direct_ci_eigensolver = 'davidson'
    mc_reference.direct_ci_root_cushion = 0
    mc_reference.run(nstates=12, method='direct_ci')

    np.testing.assert_allclose(mc_cushion.e_tot, mc_reference.e_tot[:10], atol=1e-10)


def test_direct_ci_spin_square_matches_rdm_definition():
    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run()

    mc_direct = direct_ci.CASCI(mf, ncas=2, nelecas=2)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci')

    for root in range(2):
        dm1, dm2 = mc_direct.make_rdm12(root)
        np.testing.assert_allclose(
            mc_direct.spin_square(root),
            direct_ci.spin_square_from_rdm(dm1, dm2),
            atol=1e-10,
        )


def test_cholesky_active_space_eri_transform_matches_dense():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)
    mo_cas = mf.mo_coeff[:, 1:5]

    eri_dense = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=False)
    eri_cd = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=True)

    np.testing.assert_allclose(eri_cd, eri_dense, atol=1e-8)


def test_builtin_s8_dense_eri_transform_avoids_full_ao_unpack(monkeypatch):
    from pyqed.qchem import basis as basis_module

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol.build(aosym='s8', eri='dense')

    mf = mol.RHF().run()
    mo_cas = mf.mo_coeff[:, :2]
    eri_ao = basis_module.unpack_eri_s8(mf.eri_s8, mf.mol.nao)
    reference = np.einsum(
        'ijkl,ip,jq,kr,ls->pqrs',
        eri_ao,
        mo_cas.conj(),
        mo_cas,
        mo_cas.conj(),
        mo_cas,
        optimize=True,
    )

    def fail_unpack(*args, **kwargs):
        raise AssertionError("packed CASCI should not unpack the full AO ERI tensor")

    monkeypatch.setattr(basis_module, 'unpack_eri_s8', fail_unpack)

    assert mf.eri is None
    assert mf.eri_s8 is not None
    eri_active = transform_spatial_eri_to_mo(mf, mo_cas, use_cholesky=False)

    np.testing.assert_allclose(eri_active, reference, atol=1e-12)


def test_builtin_s8_transform_supports_distinct_mo_spaces():
    from pyqed.qchem.basis import unpack_eri_s8

    mol = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='6-31g')
    mol.build(aosym='s8', eri='dense')
    mf = mol.RHF().run()
    rng = np.random.default_rng(9281)
    coefficient_shapes = (2, 3, 2, 1)
    coefficients = tuple(rng.normal(size=(mol.nao, nmo)) for nmo in coefficient_shapes)
    eri_ao = unpack_eri_s8(mf.eri_s8, mol.nao)
    reference = np.einsum(
        'ijkl,ip,jq,kr,ls->pqrs',
        eri_ao,
        coefficients[0].conj(),
        coefficients[1],
        coefficients[2].conj(),
        coefficients[3],
        optimize=True,
    )

    transformed = transform_spatial_eri_to_mo(mf, *coefficients, use_cholesky=False)

    np.testing.assert_allclose(transformed, reference, atol=1e-11)


def test_casci_use_cholesky_matches_dense_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')
    mc_cd = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, use_cholesky=True)

    np.testing.assert_allclose(mc_cd.e_tot, mc_dense.e_tot, atol=1e-8)


def test_casci_auto_uses_cholesky_from_rhf_label():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_auto = CASCI(mf, ncas=4, nelecas=4).run(nstates=2)
    mc_cd = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, use_cholesky=True)

    assert mc_auto.use_cholesky
    np.testing.assert_allclose(mc_auto.e_tot, mc_cd.e_tot, atol=1e-8)


def test_direct_ci_use_cholesky_matches_dense_energy():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.direct_ci_dense_fallback_ndets = 1
    mc_direct.run(nstates=2, method='direct_ci', use_cholesky=True)

    mc_dense = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='ci')

    assert mc_direct.solver_backend == 'direct_ci_spin_string'
    assert mc_direct.direct_connectivity is None
    assert mc_direct._direct_factor_H_diag is None
    assert mc_direct.h2e_cas is not None
    sigma = mc_direct.ci_sigma(mc_direct.ci[0])
    assert mc_direct.direct_connectivity is None
    np.testing.assert_allclose(
        np.vdot(mc_direct.ci[0], sigma),
        mc_direct.e_tot[0] - mc_direct.e_core,
        atol=1.0e-8,
    )
    np.testing.assert_allclose(mc_direct.e_tot, mc_dense.e_tot, atol=1e-8)


def test_public_casci_retains_direct_solver_and_restart_state():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()

    mc = CASCI(mf, ncas=4, nelecas=4)
    mc.direct_ci_dense_fallback_ndets = 1
    mc.run(nstates=1)
    solver = mc._direct_solver
    connectivity = solver.spin_string_connectivity
    previous_ci = solver.ci[0].copy()

    mc.run(nstates=1)

    assert mc._direct_solver is solver
    assert solver.spin_string_connectivity is connectivity
    assert abs(np.vdot(previous_ci, solver.ci[0])) > 1.0 - 1.0e-10


def test_retained_direct_integrals_detect_in_place_mo_changes():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()
    mo = np.array(mf.mo_coeff, copy=True)

    mc = direct_ci.CASCI(mf, ncas=2, nelecas=2)
    mc.direct_ci_dense_fallback_ndets = 1
    mc.run(nstates=1, mo_coeff=mo)
    initial_hcore = np.array(mc.hcore, copy=True)

    mo[:, [mc.ncore, mc.ncore + mc.ncas]] = mo[:, [mc.ncore + mc.ncas, mc.ncore]]
    mc.run(nstates=1, mo_coeff=mo)

    assert not np.allclose(mc.hcore, initial_hcore)
    np.testing.assert_array_equal(mc._direct_integrals_mo_snapshot, mo)


def test_explicit_singlet_direct_ci_auto_uses_spin0_solver():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()
    mf = RHF(mol).run()

    auto = CASCI(mf, ncas=4, nelecas=4, multiplicity=1).run(nstates=1)
    explicit = direct_ci.CASCI(
        mf, ncas=4, nelecas=4, multiplicity=1
    ).run(nstates=1, method='direct_spin0_symm')

    assert auto.solver_backend.startswith('direct_spin0_symm')
    np.testing.assert_allclose(auto.e_tot, explicit.e_tot, atol=1.0e-10)


def test_public_casci_direct_ci_method_dispatches_to_direct_backend():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_public = CASCI(mf, ncas=4, nelecas=4).run(nstates=2, method='direct_ci')

    mc_direct = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_direct.run(nstates=2, method='direct_ci')

    assert mc_public.solver_backend == mc_direct.solver_backend
    np.testing.assert_allclose(mc_public.e_tot, mc_direct.e_tot, atol=1e-10)


def test_direct_ci_auto_uses_cholesky_from_rhf_label():
    mol = Molecule(atom='Li 0 0 0; H 0 0 1.6', unit='angstrom', basis='sto-3g')
    mol.build()

    mf = RHF(mol).run(cholesky_jk=True, cholesky_tol=1e-10)

    mc_auto = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_auto.direct_ci_dense_fallback_ndets = 1
    mc_auto.run(nstates=2, method='direct_ci')

    mc_cd = direct_ci.CASCI(mf, ncas=4, nelecas=4)
    mc_cd.direct_ci_dense_fallback_ndets = 1
    mc_cd.run(nstates=2, method='direct_ci', use_cholesky=True)

    assert mc_auto.use_cholesky
    np.testing.assert_allclose(mc_auto.e_tot, mc_cd.e_tot, atol=1e-8)


def _all_string_basis(norb, nelec):
    basis = np.zeros((0, norb), dtype=np.int8)
    if nelec < 0 or nelec > norb:
        return basis
    rows = []
    for occ in combinations(range(norb), nelec):
        row = np.zeros(norb, dtype=np.int8)
        row[list(occ)] = 1
        rows.append(row)
    return np.asarray(rows, dtype=np.int8)


def _dummy_ci_state(alpha_strings, beta_strings, ci):
    binary = []
    for alpha in alpha_strings:
        for beta in beta_strings:
            binary.append(np.stack((alpha, beta), axis=0))
    return SimpleNamespace(
        binary=np.asarray(binary, dtype=np.int8),
        ci=np.asarray(ci),
        ncore=1,
        ncas=alpha_strings.shape[1],
    )


def test_string_space_svd_reconstruction_matches_exact_minors():
    rng = np.random.default_rng(123)
    norb = 4
    nelec = 2

    q_left, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    q_right, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    sigma = np.array([1.7, 1.2, 0.8, 0.4], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )
    reconstructed = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.T,
        strings,
        s.dtype,
    )

    np.testing.assert_allclose(reconstructed, exact, atol=1e-10)


def test_string_space_right_factor_uses_vh_orientation():
    rng = np.random.default_rng(456)
    norb = 4
    nelec = 2

    q_left, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    q_right, _ = np.linalg.qr(rng.normal(size=(norb, norb)))
    sigma = np.array([1.9, 1.1, 0.7, 0.3], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )

    correct = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.T,
        strings,
        s.dtype,
    )

    wrong = (
        _string_transform_matrix(q_left, strings, s.dtype)
        @ np.diag(_string_singular_weights(sigma, strings))
        @ _string_transform_matrix(q_right, strings, s.dtype)
    )

    np.testing.assert_allclose(correct, exact, atol=1e-10)
    assert np.max(np.abs(wrong - exact)) > 1e-3


def test_string_space_svd_reconstruction_matches_exact_minors_complex():
    rng = np.random.default_rng(789)
    norb = 4
    nelec = 2

    left_real = rng.normal(size=(norb, norb))
    left_imag = rng.normal(size=(norb, norb))
    right_real = rng.normal(size=(norb, norb))
    right_imag = rng.normal(size=(norb, norb))
    q_left, _ = np.linalg.qr(left_real + 1j * left_imag)
    q_right, _ = np.linalg.qr(right_real + 1j * right_imag)
    sigma = np.array([1.6, 1.0, 0.75, 0.35], dtype=float)
    s = q_left @ np.diag(sigma) @ q_right.conj().T

    strings = _all_string_basis(norb, nelec)
    exact = _string_overlap_matrix(
        s,
        _occupation_lists(strings),
        _occupation_lists(strings),
        s.dtype,
    )
    reconstructed = _reconstruct_string_overlap_from_svd(
        q_left,
        sigma,
        q_right.conj().T,
        strings,
        s.dtype,
    )

    np.testing.assert_allclose(reconstructed, exact, atol=1e-10)


def test_factorized_overlap_ignores_external_mo_block():
    alpha = _all_string_basis(2, 1)
    beta = _all_string_basis(2, 1)
    ci = np.array([0.6, -0.2, 0.3, 0.7], dtype=float)
    cibra = _dummy_ci_state(alpha, beta, ci)
    ciket = _dummy_ci_state(alpha, beta, ci[::-1])

    base = np.array(
        [
            [1.2, 0.1, -0.2, 0.4, -0.1],
            [0.3, 0.8, 0.2, 0.5, 0.7],
            [-0.4, 0.1, 1.1, -0.3, 0.6],
            [0.2, -0.6, 0.9, 1.4, -0.8],
            [0.5, 0.3, -0.7, 0.2, 0.9],
        ],
        dtype=float,
    )
    modified = base.copy()
    modified[:, 3:] = np.array(
        [
            [5.0, -4.0],
            [-3.5, 2.5],
            [7.2, -6.1],
            [0.4, 1.7],
            [-2.2, 3.3],
        ]
    )
    modified[3:, :] = np.array(
        [
            [4.1, -5.2, 6.3, 0.4, 1.7],
            [-1.9, 2.8, -3.7, -2.2, 3.3],
        ]
    )

    exact_base = _factorized_ci_overlap(cibra, ciket, s=base)
    exact_modified = _factorized_ci_overlap(cibra, ciket, s=modified)

    np.testing.assert_allclose(exact_modified, exact_base, atol=1e-12)


def test_biorthogonal_ci_overlap_candidate_matches_exact_overlap():
    alpha = _all_string_basis(2, 1)
    beta = _all_string_basis(2, 1)
    cibra = _dummy_ci_state(alpha, beta, np.array([0.45, -0.3, 0.25, 0.8], dtype=float))
    ciket = _dummy_ci_state(alpha, beta, np.array([-0.1, 0.7, 0.55, -0.4], dtype=float))

    s = np.array(
        [
            [1.15, 0.12, -0.08, 0.5, -0.3],
            [0.18, 0.95, 0.14, -0.6, 0.2],
            [-0.11, 0.21, 1.08, 0.7, -0.5],
            [0.4, -0.3, 0.2, 1.0, 0.0],
            [-0.2, 0.1, -0.4, 0.0, 1.0],
        ],
        dtype=float,
    )

    exact = _factorized_ci_overlap(cibra, ciket, s=s)
    candidate = _biorthogonal_ci_overlap_candidate(cibra, ciket, s=s)

    np.testing.assert_allclose(candidate, exact, atol=1e-10)


def test_public_overlap_matches_slow_reference_for_displaced_casci_pair():
    mol1 = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol1.build()
    mf1 = RHF(mol1).run()
    mc1 = CASCI(mf1, ncas=2, nelecas=2).run(nstates=2)

    mol2 = Molecule(atom='H 0 0 0; H 0 0 1.5', unit='bohr', basis='sto-3g')
    mol2.build()
    mf2 = RHF(mol2).run()
    mc2 = CASCI(mf2, ncas=2, nelecas=2).run(nstates=2)

    exact = _overlap_slow_from_mo_overlap(mc1, mc2, s=None)
    fast = mc1.overlap(mc2)

    np.testing.assert_allclose(fast, exact, atol=1e-10)


def test_compact_casci_frames_preserve_displaced_overlap():
    mol1 = Molecule(atom='H 0 0 0; H 0 0 1.4', unit='bohr', basis='sto-3g')
    mol1.build(eri='dense')
    mc1 = CASCI(RHF(mol1).run(), ncas=2, nelecas=2).run(nstates=2)

    mol2 = Molecule(atom='H 0 0 0; H 0 0 1.5', unit='bohr', basis='sto-3g')
    mol2.build(eri='dense')
    mc2 = CASCI(RHF(mol2).run(), ncas=2, nelecas=2).run(nstates=2)

    np.testing.assert_allclose(
        mc1.frame().overlap(mc2.frame()),
        mc1.overlap(mc2),
        atol=1e-10,
    )


def test_builtin_casci_overlap_self_is_identity_with_p_orbitals():
    mol = Molecule(
        atom='O 0 0 0; H 0 -1.4 1.1; H 0 1.4 1.1',
        unit='bohr',
        basis='sto-3g',
    )
    mol.build(eri='dense')
    mf = RHF(mol).run()
    mc = CASCI(mf, ncas=2, nelecas=2).run(nstates=2)

    np.testing.assert_allclose(
        mo_overlap(mc, mc),
        np.eye(mc.mo_coeff.shape[1]),
        atol=1e-10,
    )
    np.testing.assert_allclose(mc.overlap(mc), np.eye(2), atol=1e-10)
