import numpy as np
import pytest
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

pyscf = pytest.importorskip("pyscf")


def _use_source_tree_pyqed():
    for name in list(sys.modules):
        if name == "pyqed" or name.startswith("pyqed."):
            del sys.modules[name]
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))


def _pyscf_exact_gw_reference(mol):
    from pyscf import dft, gw, tddft

    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.kernel()
    td = tddft.dRPA(mf)
    nocc = mol.nelectron // 2
    td.nstates = nocc * (mf.mo_energy.size - nocc)
    td.kernel()
    return np.asarray(gw.GW(mf, freq_int="exact", tdmf=td).kernel())


@pytest.mark.parametrize(
    "label, atom, basis, cart",
    [
        ("h2_sto3g", "H 0 0 0; H 0 0 0.74", "sto-3g", False),
        ("lih_sto3g", "Li 0 0 0; H 0 0 1.6", "sto-3g", False),
        ("lih_ccpvdz_cart", "Li 0 0 0; H 0 0 1.6", "cc-pvdz", True),
        (
            "h2o_sto3g",
            "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            "sto-3g",
            False,
        ),
    ],
)
def test_spin_orbital_g0w0_matches_pyscf_exact_gw(label, atom, basis, cart):
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(atom=atom, basis=basis, unit="Angstrom", cart=cart, verbose=0)
    mf = scf.RHF(mol).run(verbose=0)

    egw = np.asarray(GW(mf, screening="TDH", eta=1e-6).run())
    ref = _pyscf_exact_gw_reference(mol)

    np.testing.assert_allclose(
        egw,
        ref,
        rtol=1e-8,
        atol=1e-8,
        err_msg=f"{label} PyQED G0W0 differs from PySCF exact GW",
    )


def test_spin_orbital_g0w0_accepts_pyscf_rhf_h2():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)

    egw = GW(mf, screening="TDH").run()

    np.testing.assert_allclose(
        egw,
        [-0.5969574466, 0.6895470787],
        rtol=1e-7,
        atol=1e-8,
    )


def test_gw_stores_result_metadata_and_validates_frequency_mode():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)

    gw = GW(mf, screening="TDH", eta=1e-6, freq_int="sum-over-poles")
    egw = gw.g0w0()

    assert gw.e_qp is gw.egw
    assert gw.e is gw.e_qp
    assert gw.info["method"] == "g0w0"
    assert gw.info["frequency_integration"] == "exact"
    np.testing.assert_allclose(gw.e_qp, egw, atol=0.0)

    with pytest.raises(NotImplementedError, match="not implemented"):
        GW(mf, screening="TDH", freq_int="contour_deformation")


def test_bse_accepts_chainable_gw_object():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE, TDA
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    gw = GW(mf, screening="TDH", eta=1e-3).run()
    bse = BSE(gw).run(nroots=1, low_rank=False, return_vectors=True)
    tda = TDA(gw).run(nroots=1, low_rank=False, return_vectors=True)
    bse_from_gw = gw.bse().run(nroots=1, low_rank=False)
    tda_from_gw = gw.tda().run(nroots=1, low_rank=False)

    assert bse.gw is gw
    assert tda.gw is gw
    assert bse.reference is gw
    assert bse.e_qp is gw.e_qp
    assert tda.e_qp is gw.e_qp
    assert bse.e.shape == (1,)
    assert tda.e.shape == (1,)
    assert bse.xy.shape[1] == 1
    assert tda.x.shape[1] == 1
    np.testing.assert_allclose(bse_from_gw.e, bse.e, atol=0.0)
    np.testing.assert_allclose(tda_from_gw.e, tda.e, atol=0.0)


def test_bse_and_tda_as_scanner_return_pes_arrays():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE, TDA
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-3).run()

    bse = BSE(gw).run(nroots=1, low_rank=False)
    scanner = bse.as_scanner(nroots=1, run_kwargs={"low_rank": False})
    e_pes = scanner(mol.atom_coords() + np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]))

    assert e_pes.shape == (2,)
    assert scanner.bse.e.shape == (1,)
    np.testing.assert_allclose(e_pes[0], scanner.mf.e_tot, atol=0.0)
    np.testing.assert_allclose(e_pes[1], scanner.mf.e_tot + scanner.bse.e[0], atol=0.0)

    tda = TDA(gw).run(nroots=1, low_rank=False)
    omega_scanner = tda.as_scanner(
        nroots=1,
        energy="excitation",
        run_kwargs={"low_rank": False},
    )
    omega = omega_scanner(mol.atom_coords())

    assert omega.shape == (1,)
    np.testing.assert_allclose(omega, omega_scanner.bse.e, atol=0.0)


def test_scgw_imaginary_axis_prototype_h2_shapes_are_finite():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import SCGW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    scgw = SCGW(mf, nfreq=7, wmax=8.0).run(max_cycle=2, damping=0.5)

    assert scgw.G.shape == (7, scgw.nso, scgw.nso)
    assert scgw.P.shape == (7, scgw.nso * scgw.nso, scgw.nso * scgw.nso)
    assert scgw.W.shape == scgw.P.shape
    assert scgw.Sigma_c.shape == scgw.G.shape
    assert scgw.density_matrix.shape == (scgw.nso, scgw.nso)
    assert scgw.e_qp.shape == (scgw.nso // 2,)
    assert len(scgw.history) == 2
    assert len(scgw.mu_history) == 3
    assert scgw.info["method"] == "scgw_imaginary_axis_prototype"
<<<<<<< HEAD
    assert scgw.info["grid"] == "tangent"
    assert scgw.info["adjust_mu"] is True
    assert scgw.info["total_energy"] == "galitskii_migdal"
    assert scgw.energy_components["method"] == "galitskii_migdal"
    np.testing.assert_allclose(scgw.e_tot, scgw.e_tot_gm, atol=0.0)
    np.testing.assert_allclose(scgw.e_tot_gm, scgw.e_tot_lw, atol=1e-12)
    np.testing.assert_allclose(scgw.nelec, scgw.target_nelec, atol=1e-8)
    np.testing.assert_allclose(np.trace(scgw.density_matrix), scgw.nelec, atol=1e-10)
    assert np.isfinite(scgw.e_tot)
=======
    assert scgw.info["adjust_mu"] is True
    assert scgw.info["total_energy"] == "not_implemented"
    np.testing.assert_allclose(scgw.nelec, scgw.target_nelec, atol=1e-8)
    np.testing.assert_allclose(np.trace(scgw.density_matrix), scgw.nelec, atol=1e-10)
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
    assert np.all(np.isfinite(scgw.e_qp))
    assert np.all(np.isfinite(scgw.G))


def test_scgw0_uses_fixed_screened_interaction():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import SCGW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    scgw0 = SCGW(mf, nfreq=7, wmax=8.0).scgw0(max_cycle=2, damping=0.5)

    assert scgw0.info["method"] == "scgw0_imaginary_axis_prototype"
    assert scgw0.info["update_screening"] is False
    assert scgw0.info["update_exchange"] is True
    assert scgw0.W0 is not None
    np.testing.assert_allclose(scgw0.W, scgw0.W0, atol=0.0)
    assert np.all(np.isfinite(scgw0.e_qp))


def test_gw_driver_exposes_scgw_and_scgw0():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    gw0 = GW(mf, screening="TDH", eta=1e-8).scgw0(
        nfreq=7,
        wmax=8.0,
<<<<<<< HEAD
        grid="linear",
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
        max_cycle=1,
        damping=0.5,
    )
    gw = GW(mf, screening="TDH", eta=1e-8).scgw(
        nfreq=7,
        wmax=8.0,
        max_cycle=1,
        damping=0.5,
    )

    assert gw0.method == "scgw0"
    assert gw.method == "scgw"
    assert gw0.scgw_result.info["update_screening"] is False
    assert gw.scgw_result.info["update_screening"] is True
<<<<<<< HEAD
    assert gw0.scgw_result.info["grid"] == "linear"
    assert gw.scgw_result.info["grid"] == "tangent"
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4
    assert gw0.e_qp.shape == (gw0.nso // 2,)
    assert gw.e_qp.shape == (gw.nso // 2,)
    assert np.all(np.isfinite(np.asarray(gw0)))
    assert np.all(np.isfinite(np.asarray(gw)))


def test_scgw_uses_factorized_backend_when_available():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import SCGW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="factors")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)

    scgw = SCGW(mf, nfreq=5, wmax=6.0).scgw0(max_cycle=1, damping=0.5)

    assert scgw.info["backend"] == "factorized"
    assert scgw.eri is None
    assert scgw.pair_factors is not None
    assert scgw.P.shape == (5, scgw.pair_factors.shape[0], scgw.pair_factors.shape[0])
    assert scgw.W.shape == scgw.P.shape
    assert np.all(np.isfinite(scgw.G))
    assert np.all(np.isfinite(scgw.Sigma_c))
<<<<<<< HEAD
    assert np.isfinite(scgw.e_tot)


def test_scgw_gm_and_lw_energy_components_are_consistent():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import SCGW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    scgw = SCGW(mf, nfreq=7, wmax=8.0).scgw0(max_cycle=1, damping=0.5)
    gm = scgw.galitskii_migdal_total_energy()
    lw = scgw.luttinger_ward_total_energy()

    expected = (
        gm["e_one"]
        + gm["e_hartree"]
        + gm["e_exchange"]
        + gm["e_correlation"]
        + gm["e_nuc"]
    )
    np.testing.assert_allclose(gm["e_tot"], expected, atol=1e-12)
    np.testing.assert_allclose(gm["e_tot"], lw["e_tot"], atol=1e-12)
    np.testing.assert_allclose(lw["phi_correlation"], gm["e_correlation"], atol=0.0)


def test_scgw_frequency_convergence_scan_reports_grid_deltas():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import frequency_convergence
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    rows = frequency_convergence(
        mf,
        nfreq_values=(5, 7),
        wmax=8.0,
        method="scgw0",
        grid="tangent",
        density_nfreq=65,
        run_kwargs={"max_cycle": 1, "damping": 0.5},
    )

    assert len(rows) == 2
    assert rows[0]["delta_e_tot"] is None
    assert rows[0]["delta_qp_max"] is None
    assert rows[0]["grid_converged"] is False
    assert rows[1]["delta_e_tot"] is not None
    assert rows[1]["delta_qp_max"] is not None
    assert isinstance(rows[1]["grid_converged"], bool)
    assert isinstance(rows[1]["reliable"], bool)
    assert rows[1]["energy_tol"] == 1e-5
    assert rows[1]["qp_tol"] == 1e-4
    assert rows[0]["method"] == "scgw0"
    assert rows[0]["grid"] == "tangent"
    assert rows[1]["nfreq"] == 7
    assert np.all(np.isfinite(rows[1]["e_qp"]))
    assert np.isfinite(rows[1]["e_tot"])
=======
>>>>>>> d6d6e73f3eb01265d5d7bf89f474427f6a1ea1d4


def test_scgw_matsubara_density_matches_static_limit():
    _use_source_tree_pyqed()
    from pyqed.gw.scgw import SCGW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    scgw = SCGW(mf, nfreq=7, wmax=8.0, beta=100.0, density_nfreq=1601)
    sigma = np.zeros_like(scgw.Sigma_c)
    mu = scgw._solve_mu(sigma)
    dm_green = scgw.make_density_matrix(sigma_c=sigma, mu=mu, method="green")
    dm_static = scgw.make_density_matrix(sigma_c=sigma, mu=mu, method="static")

    np.testing.assert_allclose(np.trace(dm_green), scgw.target_nelec, atol=1e-6)
    np.testing.assert_allclose(dm_green, dm_static, atol=5e-3)


def test_gw_gnw0_alias_matches_fixed_screening_evgw():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)

    ref = GW(mf, screening="TDH", eta=1e-3).evgw(
        max_cycle=5,
        conv_tol=1e-12,
        damping=0.7,
        update_screening=False,
    )
    gnw0 = GW(mf, screening="TDH", eta=1e-3).gnw0(
        max_cycle=5,
        conv_tol=1e-12,
        damping=0.7,
    )

    np.testing.assert_allclose(gnw0, ref, atol=0.0)


def test_dense_rpa_casida_matches_full_positive_roots_h2():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-6)

    e_casida, t_casida = gw.rpa(using_casida=True)
    e_full, _xy = gw.rpa(using_casida=False)
    positive = np.sort(np.asarray(e_full).real[np.asarray(e_full).real > 1e-8])

    assert np.all(np.isreal(e_casida))
    assert np.all(np.diff(e_casida) >= -1e-12)
    np.testing.assert_allclose(e_casida, positive, rtol=1e-10, atol=1e-10)
    assert t_casida.shape == (gw.nocc * (gw.nso - gw.nocc), e_casida.size)


def test_gw_rpa_correlation_energy_h2_matches_pyscf():
    from pyscf import gto, scf
    from pyscf.gw.rpa import RPA as PySCFRPA

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-6)

    e_corr = gw.rpa_correlation_energy()
    e_tot = gw.total_energy(method="rpa")
    ref = PySCFRPA(mf).run(verbose=0, nw=120)

    assert gw.e_corr == e_corr
    assert gw.e_tot == e_tot
    np.testing.assert_allclose(e_tot, mf.e_tot + e_corr, atol=0.0)
    np.testing.assert_allclose(e_corr, ref.e_corr, rtol=1e-5, atol=2e-5)


def test_gw_rpa_correlation_energy_factorized_matches_dense_lih():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    out = []
    for eri in ("dense", "factors"):
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 1.6",
            basis="sto-3g",
            unit="angstrom",
        )
        mol.build(driver="builtin", eri=eri)
        mf = RHF(mol).run(
            verbose=0,
            cholesky_jk=(eri == "factors"),
            cholesky_tol=1e-12,
        )
        gw = GW(mf, screening="TDH", eta=1e-6)
        out.append(gw.rpa_correlation_energy())
        if eri == "factors":
            assert gw.eri is None
            assert gw._pair_factors is not None

    np.testing.assert_allclose(out[1], out[0], rtol=1e-8, atol=1e-8)


def test_spin_orbital_g0w0_accepts_native_rhf_h2():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF
    from pyqed.gw.gw import GW

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)

    egw = GW(mf, screening="TDH").run()

    np.testing.assert_allclose(
        egw,
        [-0.5969574667, 0.6895471247],
        rtol=1e-7,
        atol=1e-8,
    )


def test_spin_orbital_g0w0_factorized_matches_dense_lih():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    out = []
    factorized_gw = None
    for eri in ("dense", "factors"):
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 1.6",
            basis="sto-3g",
            unit="angstrom",
        )
        mol.build(driver="builtin", eri=eri)
        mf = RHF(mol).run(
            verbose=0,
            cholesky_jk=(eri == "factors"),
            cholesky_tol=1e-12,
        )
        gw = GW(mf, screening="TDH", eta=1e-6)
        out.append(gw.run())
        if eri == "factors":
            factorized_gw = gw

    assert factorized_gw.eri is None
    assert factorized_gw._pair_factors is not None
    np.testing.assert_allclose(out[1], out[0], rtol=1e-8, atol=1e-8)


def test_native_ri_g0w0_h2_matches_molgw_cartesian_ri_reference():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    au2ev = 27.211386245988
    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)
    gw = GW(mf, screening="TDH", eta=1e-3)

    egw = np.asarray(gw.run())
    nocc = mol.nelec // 2

    assert gw.eri is None
    assert gw._pair_factors is not None
    assert gw._pair_factors.shape[0] == 30
    # MOLGW 3.4 RI-HF/G0W0 reference generated with:
    # basis='cc-pVDZ', auxil_basis='cc-pVDZ-RI', gaussian_type='cartesian'.
    np.testing.assert_allclose(
        egw[[nocc - 1, nocc]] * au2ev,
        [-16.258861, 5.212297],
        rtol=1e-7,
        atol=5e-5,
    )


@pytest.mark.parametrize(
    "label, atom, expected_nmo, expected_naux, molgw_homo_lumo_ev",
    [
        (
            "lih_ccpvdz",
            "Li 0 0 0; H 0 0 1.6",
            20,
            81,
            [-7.956506, -0.049093],
        ),
        (
            "h2o_ccpvdz",
            "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587",
            25,
            96,
            [-12.169763, 4.654951],
        ),
    ],
)
def test_native_ri_g0w0_larger_molecules_match_molgw_cartesian_ri_reference(
    label, atom, expected_nmo, expected_naux, molgw_homo_lumo_ev
):
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    au2ev = 27.211386245988
    mol = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)
    gw = GW(mf, screening="TDH", eta=1e-3)

    egw = np.asarray(gw.run())
    nocc = mol.nelec // 2

    assert len(mf.mo_energy) == expected_nmo
    assert gw.eri is None
    assert gw._pair_factors is not None
    assert gw._pair_factors.shape[0] == expected_naux
    # MOLGW 3.4 RI-HF/G0W0 references generated with:
    # basis='cc-pVDZ', auxil_basis='cc-pVDZ-RI', gaussian_type='cartesian'.
    np.testing.assert_allclose(
        egw[[nocc - 1, nocc]] * au2ev,
        molgw_homo_lumo_ev,
        rtol=1e-7,
        atol=1e-3,
        err_msg=f"{label} native RI-G0W0 differs from MOLGW",
    )


def test_spin_orbital_evgw_h2_converges():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-6)

    egw = gw.evgw(max_cycle=20, conv_tol=1e-8, damping=0.7)

    assert gw.converged
    assert len(gw.evgw_history) > 1
    np.testing.assert_allclose(
        egw,
        [-0.5967014248, 0.6892910569],
        rtol=1e-7,
        atol=1e-8,
    )


def test_spin_orbital_gnw0_h2_matches_molgw_reference():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-3)

    egw = gw.evgw(max_cycle=50, conv_tol=1e-10, damping=0.7, update_screening=False)

    assert gw.converged
    # MOLGW 3.4 no-RI GnW0 reference from ENERGY_QP.
    np.testing.assert_allclose(
        egw,
        [-0.5968406325, 0.6894302575],
        rtol=1e-7,
        atol=1e-7,
    )


def test_spin_orbital_qsgw_h2_matches_molgw_reference():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    au2ev = 27.211386245988
    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-2)

    egw = gw.qsgw(max_cycle=50, conv_tol=1e-8, damping=0.5)

    assert gw.converged
    assert len(gw.qsgw_history) > 1
    # MOLGW 3.4 no-RI QSGW reference from a forced SCF run with alpha_mixing=0.5.
    np.testing.assert_allclose(
        egw * au2ev,
        [-16.237066, 18.756558],
        rtol=1e-7,
        atol=5e-6,
    )


def test_spin_orbital_qsgw_lih_matches_molgw_reference():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    au2ev = 27.211386245988
    mol = Molecule(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    gw = GW(mf, screening="TDH", eta=1e-2)

    egw = gw.qsgw(max_cycle=60, conv_tol=1e-7, damping=0.5)

    assert gw.converged
    # MOLGW 3.4 no-RI QSGW reference from a forced SCF run with alpha_mixing=0.5.
    np.testing.assert_allclose(
        egw[[1, 2]] * au2ev,
        [-7.527326, 2.081734],
        rtol=1e-7,
        atol=5e-5,
    )


def test_spin_orbital_qsgw_factorized_matches_dense_lih():
    _use_source_tree_pyqed()
    from pyqed.gw.gw import GW
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    out = []
    for eri in ("dense", "factors"):
        mol = Molecule(
            atom="Li 0 0 0; H 0 0 1.6",
            basis="sto-3g",
            unit="angstrom",
        )
        mol.build(driver="builtin", eri=eri)
        mf = RHF(mol).run(
            verbose=0,
            cholesky_jk=(eri == "factors"),
            cholesky_tol=1e-12,
        )
        gw = GW(mf, screening="TDH", eta=1e-2)
        out.append(gw.qsgw(max_cycle=6, conv_tol=1e-12, damping=0.5))
        if eri == "factors":
            assert gw.eri is None
            assert gw._pair_factors is not None

    np.testing.assert_allclose(out[1], out[0], rtol=1e-9, atol=1e-9)


def test_pyscf_style_gw_wrapper_h2():
    from pyscf import dft, gto, tddft

    _use_source_tree_pyqed()
    from pyqed.gw.pyscf_gw import GW

    mol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    mf = dft.RKS(mol)
    mf.xc = "hf"
    mf.kernel()
    td = tddft.dRPA(mf)
    td.nstates = 1
    td.kernel()

    egw = GW(mf, td).kernel()

    np.testing.assert_allclose(
        egw,
        [-0.59695767, 0.68954730],
        rtol=1e-7,
        atol=1e-8,
    )


def test_bse_accepts_native_rhf_h2():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    def run_bse(mf):
        gw = BSE(mf, screening="TDH", eta=1e-6)
        egw = gw.kernel()
        e_tda = gw.bse(using_tda=True, using_casida=False)[0]
        e_full = gw.bse(using_tda=False, using_casida=True)[0]
        return np.asarray(egw), np.asarray(e_tda), np.asarray(e_full)

    pmol = gto.M(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="Angstrom",
        verbose=0,
    )
    pyscf_mf = scf.RHF(pmol).run(verbose=0)
    pyscf_out = run_bse(pyscf_mf)

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    native_mf = RHF(mol).run(verbose=0)
    native_out = run_bse(native_mf)

    for native, pyscf_ref in zip(native_out, pyscf_out):
        np.testing.assert_allclose(native, pyscf_ref, rtol=1e-7, atol=1e-7)


def test_bse_tda_wavefunction_overlap_same_geometry_identity():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import TDA
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    tda = TDA(mf, screening="TDH", eta=1e-3).run(
        use_qp=False,
        low_rank=False,
        nroots=2,
        return_vectors=True,
    )
    vectors = tda.x
    overlap = tda.wavefunction_overlap(tda, vectors, vectors)

    assert tda.e is tda.excitation_energies
    np.testing.assert_allclose(overlap, np.eye(2), atol=1e-8)
    np.testing.assert_allclose(tda.wavefunction_overlap(tda), overlap, atol=1e-12)
    assert tda.x is vectors
    assert tda.excitation_vectors is tda.x
    assert not hasattr(tda, "xy")
    assert not hasattr(tda, "XY")
    with pytest.raises(AttributeError, match=r"\.x, not \.xy"):
        _ = tda.xy
    assert tda.y is None
    assert tda.bse_metric == "tda"


def test_bse_tda_wavefunction_overlap_nearby_geometries():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import TDA
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    def run_h2_bse(distance):
        mol = Molecule(
            atom=f"H 0 0 0; H 0 0 {distance}",
            basis="sto-3g",
            unit="angstrom",
        )
        mol.build(driver="builtin", eri="dense")
        mf = RHF(mol).run(verbose=0)
        tda = TDA(mf, screening="TDH", eta=1e-3).run(
            use_qp=False,
            low_rank=False,
            nroots=1,
            return_vectors=True,
        )
        return tda, tda.x[:, 0]

    bse_bra, vec_bra = run_h2_bse(0.74)
    bse_ket, vec_ket = run_h2_bse(0.78)
    overlap = bse_bra.wavefunction_overlap(bse_ket, vec_bra, vec_ket)

    assert abs(overlap) > 0.8
    assert abs(overlap) <= 1.05


def test_bse_full_wavefunction_overlap_uses_stored_xy():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    def run_h2_bse(distance):
        mol = Molecule(
            atom=f"H 0 0 0; H 0 0 {distance}",
            basis="sto-3g",
            unit="angstrom",
        )
        mol.build(driver="builtin", eri="dense")
        mf = RHF(mol).run(verbose=0)
        bse = BSE(mf, screening="TDH", eta=1e-3)
        bse.e_qp = bse.e_mf.copy()
        assert bse.egw is bse.e_qp
        e_rpa, t_rpa = bse.rpa(method=bse.screening)
        bse._M = bse.get_m_rpa(e_rpa, t_rpa)
        bse.run(
            use_qp=False,
            low_rank=False,
            nroots=1,
            return_vectors=True,
        )
        xy = bse.xy
        dim = bse.nocc * (bse.nso - bse.nocc)
        assert xy.shape[0] == 2 * dim
        assert bse.xy is bse.XY
        assert bse.excitation_vectors is bse.xy
        assert bse.x.shape[0] == dim
        assert bse.y.shape[0] == dim
        assert bse.bse_metric == "full"
        return bse, xy

    bse_bra, xy_bra = run_h2_bse(0.74)
    bse_ket, xy_ket = run_h2_bse(0.78)

    explicit = bse_bra.wavefunction_overlap(
        bse_ket,
        xy_bra,
        xy_ket,
        metric="full",
    )
    stored = bse_bra.wavefunction_overlap(bse_ket, metric="full")

    np.testing.assert_allclose(stored, explicit, atol=1e-12)
    np.testing.assert_allclose(
        bse_bra.wavefunction_overlap(bse_bra, metric="full"),
        np.eye(1),
        atol=1e-10,
    )


def test_bse_full_same_geometry_overlap_is_identity_for_degenerate_roots():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="dense")
    mf = RHF(mol).run(verbose=0)
    bse = BSE(mf, screening="TDH", eta=1e-3).run(
        use_qp=False,
        low_rank=False,
        nroots=3,
    )

    np.testing.assert_allclose(bse.e[1], bse.e[2], atol=1e-12)
    np.testing.assert_allclose(
        bse.wavefunction_overlap(bse, metric="full"),
        np.eye(3),
        atol=1e-10,
    )


def test_dense_bse_matches_molgw_no_ri_lih_sto3g():
    from pyscf import gto, scf

    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE

    au2ev = 27.211386245988
    mol = gto.M(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="sto-3g",
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    mf = scf.RHF(mol).run(verbose=0)
    gw = BSE(mf, screening="TDH", eta=1e-3)
    gw.kernel()

    e_tda = gw.bse(using_tda=True, using_casida=False)[0]
    e_full = gw.bse(using_tda=False, using_casida=True)[0]
    gw.run(
        nroots=8,
        tol=1e-10,
        max_cycle=200,
        return_vectors=False,
        use_qp=False,
    )
    e_full_lr = gw.e

    # MOLGW 3.4 no-RI reference:
    # GW run without auxil_basis and print_w='yes', followed by BSE restart.
    np.testing.assert_allclose(e_tda[:4] * au2ev, [3.85871494, 5.54172082, 5.54172082, 17.68551222], atol=2e-4)
    np.testing.assert_allclose(e_full[:4] * au2ev, [3.78552655, 5.49055723, 5.49055723, 16.93616022], atol=2e-4)


def test_native_ri_direct_bse_matches_molgw_lih_ccpvdz():
    _use_source_tree_pyqed()
    from pyqed.gw.bse import BSE, TDA
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    au2ev = 27.211386245988
    mol = Molecule(
        atom="Li 0 0 0; H 0 0 1.6",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(driver="builtin", eri="ri", auxbasis="cc-pvdz-rifit")
    mf = RHF(mol).run(verbose=0, cholesky_jk=True, cholesky_tol=1e-12)
    gw = BSE(mf, screening="TDH", eta=1e-3)
    assert gw.eri is None
    assert gw._pair_factors is not None
    e_rpa, t_rpa = gw.rpa(method=gw.screening)
    gw._M = gw.get_m_rpa(e_rpa, t_rpa)
    gw.e_qp = gw.e_mf.copy()
    assert gw.egw is gw.e_qp

    e_tda = gw.bse(using_tda=True, using_casida=False)[0]
    tda = TDA(mf, screening="TDH", eta=1e-3).run(
        nroots=8,
        tol=1e-10,
        return_vectors=False,
        use_qp=False,
    )
    assert tda.eri is None
    assert tda._pair_factors is not None
    e_tda_lr = tda.e
    e_full = gw.bse(using_tda=False, using_casida=True)[0]
    gw.run(
        nroots=8,
        tol=1e-10,
        max_cycle=200,
        return_vectors=False,
        use_qp=False,
    )
    e_full_lr = gw.e

    # MOLGW 3.4 direct RI-BSE reference generated with:
    # basis='cc-pVDZ', auxil_basis='cc-pVDZ-RI', gaussian_type='cartesian',
    # postscf='BSE'.  MOLGW direct BSE uses HF/gKS orbital energies here,
    # not the G0W0 quasiparticle energies from a separate GW run.
    np.testing.assert_allclose(e_tda_lr, e_tda[:8], rtol=1e-9, atol=1e-10)
    np.testing.assert_allclose(
        e_tda_lr * au2ev,
        [
            4.07931719,
            5.14354776,
            5.14354776,
            7.19619242,
            7.88947421,
            7.96996555,
            7.96996555,
            10.94530367,
        ],
        rtol=1e-7,
        atol=5e-4,
    )
    np.testing.assert_allclose(
        e_full[:8] * au2ev,
        [
            4.02604781,
            5.10706435,
            5.10706435,
            7.10329968,
            7.88705679,
            7.95069957,
            7.95069957,
            10.78810262,
        ],
        rtol=1e-7,
        atol=5e-4,
    )
    np.testing.assert_allclose(e_full_lr, e_full[:8], rtol=1e-9, atol=1e-10)
