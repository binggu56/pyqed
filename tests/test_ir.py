import numpy as np
import pytest

import matplotlib
matplotlib.use("Agg", force=True)

from pyqed.qchem import IR, Molecule, RHF
from pyqed.qchem.dft import AOGrid, RKS


def test_ir_from_stick_data_computes_intensities_and_spectrum():
    ir = IR(
        frequencies=[1000.0, 1600.0],
        dipole_derivatives=[[1.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
    )

    returned = ir.run()
    x, signal = ir.spectrum(width=20.0)

    assert returned is ir
    np.testing.assert_allclose(ir.frequencies, [1000.0, 1600.0])
    np.testing.assert_allclose(ir.intensities, [5.0, 9.0])
    assert ir.dipole_derivatives.shape == (2, 3)
    assert ir.frequency_unit == "cm^-1"
    assert x.shape == signal.shape
    assert x.size == 1000
    assert np.all(np.isfinite(signal))
    assert np.max(signal) > 0.0


def test_ir_from_hessian_like_backend_and_mode_selection():
    class HessianLike:
        def vibrational_analysis(self):
            return {
                "freq_cm1": np.array([900.0, 1200.0, 1500.0]),
                "modes": np.ones((3, 2, 3)),
                "reduced_mass_amu": np.array([1.1, 1.2, 1.3]),
            }

    ir = IR.from_hessian(
        HessianLike(),
        dipole_derivatives=[
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
        ],
    )

    ir.run(mode_indices=[0, 2])

    np.testing.assert_allclose(ir.frequencies, [900.0, 1500.0])
    np.testing.assert_allclose(ir.intensities, [1.0, 9.0])
    np.testing.assert_allclose(ir.reduced_masses, [1.1, 1.3])
    assert ir.modes.shape == (2, 2, 3)


def test_ir_accepts_normal_modes_backend_and_precomputed_intensities():
    class NormalModeBackend:
        def normal_modes(self):
            modes = np.zeros((2, 1, 3))
            return np.array([500.0, 750.0]), modes, np.array([2.0, 3.0])

    ir = IR(
        NormalModeBackend(),
        intensities=[4.0, 8.0],
        intensity_unit="km/mol",
    ).run()

    np.testing.assert_allclose(ir.frequencies, [500.0, 750.0])
    np.testing.assert_allclose(ir.intensities, [4.0, 8.0])
    np.testing.assert_allclose(ir.reduced_masses, [2.0, 3.0])
    assert ir.intensity_unit == "km/mol"
    assert np.all(np.isnan(ir.dipole_derivatives))


def test_ir_accepts_harmonic_analysis_data():
    data = {
        "freq_cm1": np.array([1100.0, 1700.0]),
        "modes": np.zeros((2, 3, 3)),
        "reduced_mass_amu": np.array([1.5, 2.5]),
    }

    ir = IR.from_harmonic_analysis(
        data,
        dipole_derivatives=[[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]],
    ).run()

    np.testing.assert_allclose(ir.frequencies, [1100.0, 1700.0])
    np.testing.assert_allclose(ir.intensities, [2.0, 4.0])
    np.testing.assert_allclose(ir.reduced_masses, [1.5, 2.5])
    assert ir.modes.shape == (2, 3, 3)
    assert ir.frequency_unit == "cm^-1"


def test_finite_difference_dipole_derivatives():
    coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    modes = np.array([
        [[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[0.0, 0.0, 0.0], [0.0, 2.0, 0.0]],
    ])

    def dipole_fn(displaced):
        flat = displaced.reshape(-1)
        return np.array([
            2.0 * flat[0] - flat[3],
            flat[1] + 3.0 * flat[4],
            flat[2] - 4.0 * flat[5],
        ])

    derivatives = IR.finite_difference_dipole_derivatives(
        dipole_fn,
        coords,
        modes,
        step=1.0e-4,
    )

    np.testing.assert_allclose(
        derivatives,
        [[2.0, 0.0, 0.0], [0.0, 6.0, 0.0]],
        atol=1.0e-10,
    )


def test_ir_plot_mode_from_harmonic_data(tmp_path):
    data = {
        "freq_cm1": np.array([1200.0]),
        "modes": np.array([[[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]]]),
        "reduced_mass_amu": np.array([1.0]),
        "atom_coords": np.array([[0.0, 0.0, -0.7], [0.0, 0.0, 0.7]]),
        "atom_symbols": ["H", "H"],
    }
    ir = IR.from_harmonic_analysis(data, dipole_derivatives=[[1.0, 0.0, 0.0]]).run()

    ax = ir.plot_mode(0, view=(30.0, -45.0))
    out = tmp_path / "mode.png"
    ax.figure.savefig(out)

    assert ax.elev == pytest.approx(30.0)
    assert ax.azim == pytest.approx(-45.0)
    assert out.exists()
    assert out.stat().st_size > 0


def test_ir_plot_mode_requires_coordinates():
    ir = IR(
        frequencies=[1200.0],
        dipole_derivatives=[[1.0, 0.0, 0.0]],
        modes=np.zeros((1, 2, 3)),
    ).run()

    with pytest.raises(ValueError, match="Atomic coordinates are missing"):
        ir.plot_mode(0)


def test_ir_accepts_native_rhf_method_backend():
    mol = Molecule(atom="H 0 0 -0.7; H 0 0 0.7", unit="bohr", basis="sto-3g")
    mol.build(driver="builtin", eri="s8")
    mf = RHF(mol).run()

    ir = IR(mf, hessian_step=2.0e-3, dipole_step=2.0e-3).run()

    assert ir.frequencies.shape == (1,)
    assert ir.dipole_derivatives.shape == (1, 3)
    assert ir.intensities.shape == (1,)
    assert ir.modes.shape == (1, 2, 3)
    assert np.all(np.isfinite(ir.frequencies))
    assert np.all(np.isfinite(ir.intensities))


def test_ir_from_method_accepts_native_rks_method_backend():
    mol = Molecule(atom="H 0 0 -0.7; H 0 0 0.7", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    grid = AOGrid.atom_centered(mol, n_radial=4, n_angular=6, with_grad=False)
    mf = RKS(mol, grid=grid, xc="svwn")
    mf.max_cycle = 30
    mf.conv_tol = 1.0e-7
    mf.run()

    ir = IR.from_method(mf, hessian_step=3.0e-3, dipole_step=3.0e-3).run()

    assert ir.frequencies.shape == (1,)
    assert ir.dipole_derivatives.shape == (1, 3)
    assert ir.intensities.shape == (1,)
    assert ir.modes.shape == (1, 2, 3)
    assert np.all(np.isfinite(ir.frequencies))
    assert np.all(np.isfinite(ir.intensities))


def test_ir_rejects_casci_like_method_backend():
    class CASCI:
        ncas = 2
        nelecas = 2

    with pytest.raises(NotImplementedError, match="IR\\(CASCI\\)"):
        IR(CASCI()).run()


def test_cartesian_hessian_analysis_matches_pyscf_projection():
    pyscf = pytest.importorskip("pyscf")
    from pyscf.hessian import thermo
    from pyqed.qchem.dft.hessian import analyze_cartesian_hessian

    mol = pyscf.M(
        atom="O 0 0 0; H 0 -1.43233673 1.10715266; H 0 1.43233673 1.10715266",
        unit="Bohr",
        basis="sto-3g",
        verbose=0,
    )
    rng = np.random.default_rng(12)
    hess = rng.normal(size=(9, 9))
    hess = 0.5 * (hess + hess.T)
    hess4 = hess.reshape(3, 3, 3, 3).transpose(0, 2, 1, 3)

    data = analyze_cartesian_hessian(
        hess,
        mol.atom_coords(),
        mol.atom_mass_list(isotope_avg=True),
        negative_imaginary=False,
    )
    ref = thermo.harmonic_analysis(mol, hess4)

    np.testing.assert_allclose(data["freq_cm1"], ref["freq_wavenumber"], atol=2.0e-5)
    mode_sign = np.sign(np.einsum("ijk,ijk->i", data["modes"], ref["norm_mode"]))
    mode_sign[mode_sign == 0.0] = 1.0
    np.testing.assert_allclose(data["modes"] * mode_sign[:, None, None], ref["norm_mode"], atol=1.0e-10)
    np.testing.assert_allclose(data["reduced_mass_amu"], ref["reduced_mass"], atol=1.0e-10)
