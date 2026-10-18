"""Reduced recursive Hubbard checks against explicit small-sector references."""
import numpy as np
import pytest

from examples.mps.hubbard_2d_dmrg import (
    build_2d_hubbard_mpo, calculate, normalized_energy,
)
from examples.mps.hubbard_2d_ed import hubbard_2d_hamiltonian
from pyqed.mps.nonabelian import build_random_reduced_spatial_mps
from pyqed.mps.nonabelian.basis import TwoSiteBasis
from pyqed.mps.nonabelian.canonical import mixed_canonicalize_sites
from pyqed.mps.nonabelian.contraction import merge_mps_sites
from pyqed.mps.nonabelian.environment import BlockSparseEnvironmentChain
from pyqed.mps.nonabelian.solver import (
    LocalOperator, pack_two_site_state, solve_local_two_site, two_site_state_basis,
)
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo


@pytest.fixture(autouse=True)
def require_reduced_kernel():
    pytest.importorskip("pyqed.mps.nonabelian._su2_kernel")


def test_reduced_hamiltonian_and_norm_share_channel_basis_and_are_hermitian():
    sites = build_random_reduced_spatial_mps(4, seed=7, cap_multiplicity=True)
    mpo = build_2d_hubbard_mpo(sites, 2, 2, hopping_t=1., onsite_u=4.)
    sites = mixed_canonicalize_sites(sites, 1, cutoff=0.)
    merged = merge_mps_sites(sites[0], sites[1])
    assert merged.metadata["physical_basis"] == "fully_reduced_su2"
    identity = _identity_mpo_factors_for_sites_and_mpo(sites, mpo)
    operators = [BlockSparseEnvironmentChain.build(sites, factors).bond_operator(0, merged)
                 for factors in (mpo, identity)]
    H, N = [o if isinstance(o, LocalOperator) else o.to_local_operator() for o in operators]
    assert H.basis.channel_resolved and N.basis.channel_resolved
    assert H.basis.compatible_with_layout(N.basis)
    eye = np.eye(H.basis.size)
    h, n = [np.column_stack([(o.packed_matvec or o.aux_packed_matvec)(v) for v in eye]) for o in (H, N)]
    np.testing.assert_allclose(h, h.conj().T, atol=1e-12)
    np.testing.assert_allclose(n, n.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(n).min() > 0
    x, _ = pack_two_site_state(merged, layout=H.basis)
    expected, _ = normalized_energy(sites, mpo)
    assert np.vdot(x, h @ x) / np.vdot(x, n @ x) == pytest.approx(expected, abs=1e-12)


def test_local_solver_rejects_mismatched_metric_basis():
    sites = build_random_reduced_spatial_mps(4, seed=5)
    merged = merge_mps_sites(sites[1], sites[2])
    channel = TwoSiteBasis.from_channel_tensor(merged)
    collapsed = two_site_state_basis(merged)
    H = LocalOperator(packed_matvec=lambda x: x, diag=np.ones(channel.size), basis=channel)
    N = LocalOperator(packed_matvec=lambda x: x, diag=np.ones(collapsed.size), basis=collapsed)
    with pytest.raises(ValueError, match="incompatible"):
        solve_local_two_site(merged, H, norm_operator=N)


@pytest.mark.parametrize("ly,passes,D", [(2, 1, 30), (3, 4, 64)])
def test_owned_reduced_sweeps_match_ed_and_export_current_state(ly, passes, D):
    from scipy.sparse.linalg import eigsh
    result = calculate(2, ly, D=D, nsweeps=passes, verbose=0)
    h, _ = hubbard_2d_hamiltonian(2, ly, nup=ly, ndown=ly, t=1., u=4.)
    exact = eigsh(h, k=1, which="SA", return_eigenvectors=False)[0]
    assert result["validated"]
    assert result["energy"] == pytest.approx(exact, abs=1e-8)
    assert result["energy_difference"] < 1e-10
    assert result["max_local_residual"] <= 1e-7
    assert result["max_kept"] <= D
    assert all(h["energy_source"] == "final_mpo_expectation" for h in result["history"])


@pytest.mark.parametrize("singular", [False, True])
def test_generalized_davidson_explores_metric_support_from_exact_excited_guess(singular):
    from pyqed.mps.nonabelian.solver import _solve_packed_generalized_davidson
    hdiag = np.array([0., 1., 2.])
    ndiag = np.array([1., 0. if singular else 1., 1.])
    energy, vector, info = _solve_packed_generalized_davidson(
        np.array([0., 0., 1.]), lambda x: hdiag*x, h_diag=hdiag,
        N=lambda x: ndiag*x, n_diag=ndiag, tol=1e-10,
        tol_residual=1e-8, itermax=30, max_space=3)
    assert info["davidson_converged"]
    assert info["residual"] < 1e-8
    assert energy == pytest.approx(0., abs=1e-10)
    assert abs(vector[0]) == pytest.approx(1.)
