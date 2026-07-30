import numpy as np
import pytest

from pyqed.qchem.dmrg import TDDMRG as QChemTDDMRG
from pyqed.qchem.gdvr.newton import CollocatedERIOp
from pyqed.qchem.gdvr.integrals import (
    PrimitiveLabel,
    V_en_sp_total_at_z,
    _eri_same_center_sp_analytic_tensor,
    _eri_prony_same_center_sp_tensor,
    _eri_prony_tensor,
    _eri_ssss_same_center_exact,
    eri_2d_cartesian_with_p,
)
from pyqed.qchem.gdvr.rhf import (
    AtomicChain,
    Molecule,
    STO6G_H_S_EXPS,
    STO6G_HE_S_EXPS,
    _default_transverse_basis,
    _resolve_transverse_basis,
    local_ecp_terms_from_pyscf,
    precompute_eri_method2_JK_psd,
    scf_rhf_method2,
    eri_JK_from_kernels_M1,
)
from pyqed.qchem.gdvr import TDDMRG as GDVRTDDMRG
from pyqed.qchem.gdvr.tddmrg import gdvr_z_operator


def test_atomic_chain_public_api_provides_default_h_basis():
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
    )

    assert mol.elements == ["H", "H"]
    np.testing.assert_allclose(mol.charges, [1.0, 1.0])
    assert mol.nelec == 2

    s_exps, p_exps, d_exps = _default_transverse_basis(mol.elements)
    np.testing.assert_allclose(s_exps, STO6G_H_S_EXPS)
    assert p_exps.size == 0
    assert d_exps.size == 0


def test_gdvr_molecule_dipole_operator_uses_electronic_sign_and_multiplicity():
    mol = AtomicChain(["H"], coords=[[0.0, 0.0, 0.0]])
    mol.build(Lz=2.0, Nz=3, M=2, verbose=False)

    z_diag = np.repeat(mol.z, 2)
    np.testing.assert_allclose(mol.position_operator("z"), np.diag(z_diag))
    np.testing.assert_allclose(mol.dipole_operator("z"), -np.diag(z_diag))
    np.testing.assert_allclose(mol.dipole_operator("z", electronic=False), np.diag(z_diag))
    np.testing.assert_allclose(gdvr_z_operator(mol, electronic=True), mol.dipole_operator("z"))


def test_gdvr_rhf_to_gto_returns_qchem_adapter():
    mol = AtomicChain(
        ["H", "H"],
        coords=[
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
    )
    mol.build(Lz=3.0, Nz=5, M=1, verbose=False)
    mf = mol.RHF().run(conv=1e-6, max_iter=40, verbose=False)
    assert mf.info["newton_cycles"] >= 1

    adapter = mf.to_gto()

    assert adapter.mol is mol
    assert callable(adapter.active_space_integrals)
    np.testing.assert_allclose(adapter.get_ovlp(), np.eye(mol.shapes["size"]))


def test_gdvr_rhf_tddmrg_dispatches_direct_and_active_space_paths():
    mol = AtomicChain(
        ["H", "H"],
        coords=[
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
    )
    mol.build(Lz=3.0, Nz=5, M=1, verbose=False)
    mf = mol.RHF().run(conv=1e-6, max_iter=40, verbose=False)

    direct = mf.TDDMRG()
    active = mf.TDDMRG(ncas=2, nelecas=(1, 1))

    assert type(direct) is GDVRTDDMRG
    assert type(active) is QChemTDDMRG


def test_sto6g_named_basis_is_element_aware_for_helium():
    s_exps, p_exps, d_exps, basis_name = _resolve_transverse_basis(
        charges=[2.0],
        transverse_basis="sto6g",
    )

    assert basis_name == "sto6g"
    np.testing.assert_allclose(s_exps, STO6G_HE_S_EXPS)
    assert p_exps.size == 0
    assert d_exps.size == 0


def test_pyscf_basis_fallback_reads_bfd_vdz_for_lithium():
    pytest.importorskip("pyscf")

    s_exps, p_exps, d_exps, basis_name = _resolve_transverse_basis(
        charges=[3.0],
        transverse_basis="bfd-vdz",
    )

    assert basis_name == "bfd-vdz"
    assert s_exps.size > 0
    assert p_exps.size > 0
    assert d_exps.size > 0


def test_lithium_bfd_ecp_terms_can_scalarize_nonlocal_channel():
    pytest.importorskip("pyscf")

    local_only = local_ecp_terms_from_pyscf("Li", "bfd")
    scalarized = local_ecp_terms_from_pyscf("Li", "bfd", scalarize_nonlocal=True)

    assert local_only["core_electrons"] == 2
    assert local_only["omitted_nonlocal_channels"] == (0,)
    assert len(local_only["semilocal_terms"]) == 1
    assert local_only["semilocal_terms"][0][:2] == (0, 0)
    np.testing.assert_allclose(local_only["semilocal_terms"][0][2:], [1.34319829, 7.09172172])
    assert scalarized["omitted_nonlocal_channels"] == ()
    assert scalarized["scalarized_nonlocal_channels"] == (0,)
    assert scalarized["semilocal_terms"] == ()
    assert len(scalarized["local_terms"]) > len(local_only["local_terms"])


def test_lithium_bfd_semilocal_projector_builds_fourier_gdvr_hcore():
    pytest.importorskip("pyscf")

    ecp = local_ecp_terms_from_pyscf("Li", "bfd")
    mol = Molecule(
        [1.0],
        coords=[(0.0, 0.0, 0.0)],
        nelec=2,
        spin=0,
        basis_charges=[3.0],
        local_ecp_terms=[ecp["local_terms"]],
        semilocal_ecp_terms=[ecp["semilocal_terms"]],
    )
    mol.build(Lz=4.0, Nz=8, M=1, transverse_basis="sto3g", dvr_method="exp", verbose=False)

    assert mol.hcore.shape == (8, 8)
    np.testing.assert_allclose(mol.hcore, mol.hcore.T, atol=1e-10)
    assert np.isfinite(mol.hcore).all()


def test_transverse_newton_uses_full_scf_default():
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
    )
    mol.build(Lz=3.0, Nz=7, M=1, verbose=False, dvr_method="sine")
    mf = mol.RHF().run(newton=False, conv=1e-6, max_iter=40, verbose=False)

    mf.newton(max_cycles=1, sweeps=1, scf_conv=1e-6, scf_max_iter=40, verbose=False)

    assert mf.info["newton_cycles"] == 1
    assert len(mf.info["newton_energy_history"]) == 2
    assert mf.info["newton_scf_mode"] == "full"


def test_gdvr_run_raises_when_initial_scf_does_not_converge():
    mol = AtomicChain(
        elements=["H", "H"],
        coords=[
            [0.0, 0.0, -0.7],
            [0.0, 0.0, 0.7],
        ],
    )
    mol.build(Lz=3.0, Nz=7, M=1, verbose=False, dvr_method="sine")

    mf = mol.RHF()
    with pytest.raises(RuntimeError, match="SCF did not converge"):
        mf.run(newton=False, conv=1e-12, max_iter=1, verbose=False)
    assert not mf.info["converged"]

    with pytest.raises(RuntimeError, match="transverse Newton optimization"):
        mol.RHF().run(
            conv=1e-6,
            max_iter=40,
            max_cycles=1,
            sweeps=1,
            verbose=False,
        )


def test_gdvr_rhf_rejects_odd_electron_count():
    with pytest.raises(ValueError, match="positive even"):
        scf_rhf_method2(
            np.eye(2),
            ERI_J=None,
            ERI_K=None,
            Nz=2,
            M=1,
            nelec=1,
            verbose=False,
        )


def test_m1_eri_rebuild_skips_omitted_offsets():
    C_list = [np.ones((1, 1)) for _ in range(3)]
    K_h = [np.array([[2.0]]), None, None]
    Kx_h = [np.array([[1.0]]), None, None]

    ERI_J, ERI_K = eri_JK_from_kernels_M1(C_list, K_h, Kx_h)

    np.testing.assert_allclose(np.diag(np.asarray(ERI_J, float)), [2.0, 2.0, 2.0])
    np.testing.assert_allclose(np.diag(np.asarray(ERI_K, float)), [1.0, 1.0, 1.0])
    assert ERI_J[0][1] == 0.0
    assert ERI_K[0][1] == 0.0


def test_collocated_eri_operator_fast_permutations_match_reference():
    rng = np.random.default_rng(7)
    n_ao = 3
    nz = 3
    kernels = []
    exchange_kernels = []
    for _ in range(nz):
        eri = rng.normal(size=(n_ao, n_ao, n_ao, n_ao))
        kernels.append(eri.reshape(n_ao * n_ao, n_ao * n_ao))
        exchange_kernels.append(eri.transpose(0, 2, 1, 3).reshape(n_ao * n_ao, n_ao * n_ao))

    op = CollocatedERIOp.from_kernels(n_ao, nz, 0.4, kernels, exchange_kernels)
    d = rng.normal(size=(nz, n_ao))

    for n in range(nz):
        for m in range(nz):
            h = abs(n - m)
            eri = kernels[h].reshape(n_ao, n_ao, n_ao, n_ao)
            np.testing.assert_allclose(
                op.block_nm__kl(n, n, m, m, d[m], d[m]),
                np.einsum("abcd,c,d->ab", eri, d[m], d[m], optimize=True),
            )
            np.testing.assert_allclose(
                op.block_nl__km(n, m, m, n, d[m], d[n]),
                np.einsum("adcb,c,d->ab", eri, d[m], d[n], optimize=True),
            )
            if n == m:
                np.testing.assert_allclose(
                    op.block_nk__ml(n, n, n, n, d[n], d[n]),
                    np.einsum("acbd,c,d->ab", eri, d[n], d[n], optimize=True),
                )
                np.testing.assert_allclose(
                    op.block_nl__mk(n, n, n, n, d[n], d[n]),
                    np.einsum("acdb,c,d->ab", eri, d[n], d[n], optimize=True),
                )


def test_gdvr_eri_precompute_includes_full_offset_range():
    alphas = np.array([0.8])
    centers = np.zeros((1, 2))
    labels = [PrimitiveLabel("2d-s", 2, (0, 0, 0))]
    z_grid = np.array([0.0, 1.0, 2.0])
    C_list = [np.ones((1, 1)) for _ in z_grid]

    ERI_J, ERI_K = precompute_eri_method2_JK_psd(
        alphas,
        centers,
        labels,
        z_grid,
        C_list,
        M=1,
        verbose=False,
    )

    assert ERI_J[0][2] > 0.0
    assert ERI_K[0][2] > 0.0


def test_same_center_sp_prony_eri_matches_generic_tensor():
    alphas = np.array([0.5, 1.3, 0.7, 0.9])
    centers = np.zeros((4, 2))
    labels = [
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-px", 2, (1, 0, 0)),
        PrimitiveLabel("2d-py", 2, (0, 1, 0)),
    ]

    ref = _eri_prony_tensor(alphas, centers, labels, 0.8)
    fast = _eri_prony_same_center_sp_tensor(alphas, centers, labels, 0.8)

    np.testing.assert_allclose(fast, ref, rtol=1e-12, atol=1e-12)


def test_same_center_sp_analytic_eri_matches_closed_form_s_block():
    alphas = np.array([0.5, 1.3, 0.7, 0.9])
    centers = np.zeros((4, 2))
    labels = [
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-px", 2, (1, 0, 0)),
        PrimitiveLabel("2d-py", 2, (0, 1, 0)),
    ]

    exact = _eri_same_center_sp_analytic_tensor(alphas, centers, labels, 0.8)

    for i in (0, 1):
        for j in (0, 1):
            for k in (0, 1):
                for l in (0, 1):
                    ref = _eri_ssss_same_center_exact(
                        alphas[i],
                        alphas[j],
                        alphas[k],
                        alphas[l],
                        0.8,
                    )
                    np.testing.assert_allclose(exact[i, j, k, l], ref, rtol=1e-12, atol=1e-12)


def test_same_center_sp_prony_agrees_with_analytic_fit_accuracy():
    alphas = np.array([0.5, 1.3, 0.7, 0.9])
    centers = np.zeros((4, 2))
    labels = [
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-px", 2, (1, 0, 0)),
        PrimitiveLabel("2d-py", 2, (0, 1, 0)),
    ]

    exact = _eri_same_center_sp_analytic_tensor(alphas, centers, labels, 2.4)
    prony = _eri_prony_same_center_sp_tensor(alphas, centers, labels, 2.4)
    meaningful = np.abs(exact) > 1.0e-10

    np.testing.assert_allclose(prony[meaningful], exact[meaningful], rtol=1e-5, atol=1e-10)
    np.testing.assert_allclose(
        eri_2d_cartesian_with_p(alphas, centers, labels, 2.4),
        exact,
        rtol=1e-12,
        atol=1e-12,
    )


def test_ven_matrix_cache_matches_uncached_result():
    alphas = np.array([0.8, 0.4])
    centers = np.zeros((2, 2))
    labels = [
        PrimitiveLabel("2d-s", 2, (0, 0, 0)),
        PrimitiveLabel("2d-px", 2, (1, 0, 0)),
    ]
    nuclei = [(1.0, 0.0, 0.0, -1.0), (1.0, 0.0, 0.0, 1.0)]

    uncached = V_en_sp_total_at_z(alphas, centers, labels, nuclei, 0.2)
    cache = {}
    cached = V_en_sp_total_at_z(alphas, centers, labels, nuclei, 0.2, matrix_cache=cache)

    np.testing.assert_allclose(cached, uncached, rtol=1e-12, atol=1e-12)
    assert cache
