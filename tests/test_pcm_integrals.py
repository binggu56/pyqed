import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem import pcm_integrals
from pyqed.qchem.solvent.pcm import PCM


def test_native_pcm_rys_surface_integrals_match_pyscf_fake_charge_backend():
    pytest.importorskip("pyscf")

    mol = Molecule(
        atom="O 0 0 0; H 0 1.0 0; H 0 0 1.0",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build()

    pcm_pyscf = PCM(mol)
    pcm_pyscf.lebedev_order = 3
    pcm_pyscf.integral_backend = "pyscf"
    pcm_pyscf.build()

    pcm_native = PCM(mol)
    pcm_native.lebedev_order = 3
    pcm_native.integral_backend = "native"
    pcm_native.build()

    np.testing.assert_allclose(
        pcm_native.v_grids_n,
        pcm_pyscf.v_grids_n,
        atol=1e-10,
        rtol=1e-10,
    )
    np.testing.assert_allclose(
        pcm_native._surface_coulomb_tensor(),
        pcm_pyscf._surface_coulomb_tensor(),
        atol=5e-8,
        rtol=5e-8,
    )


def test_pcm_default_integral_backend_is_native_without_pyscf_backend():
    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")

    pcm = PCM(mol)
    pcm.lebedev_order = 3
    pcm.build()

    assert pcm.integral_backend == "auto"
    assert pcm._intermediates["integral_backend"] == "native"
    assert pcm.v_grids_n is not None
    assert pcm._surface_coulomb_tensor().shape == (mol.nao, mol.nao, len(pcm.surface["grid_coords"]))


def test_pcm_auto_backend_uses_native_libcint_order_for_pyscf_general_basis():
    pytest.importorskip("pyscf")

    mol = Molecule(
        atom="O 0 0 0; H 0 1.0 0; H 0 0 1.0",
        unit="bohr",
        basis="6-31g",
    )
    mol.build()

    pcm_auto = PCM(mol)
    pcm_auto.verbose = 0
    pcm_auto.lebedev_order = 3
    pcm_auto.build()

    pcm_pyscf = PCM(mol)
    pcm_pyscf.verbose = 0
    pcm_pyscf.lebedev_order = 3
    pcm_pyscf.integral_backend = "pyscf"
    pcm_pyscf.build()

    assert pcm_auto._intermediates["integral_backend"] == "native"
    np.testing.assert_allclose(
        pcm_auto._surface_coulomb_tensor(),
        pcm_pyscf._surface_coulomb_tensor(),
        atol=5e-8,
        rtol=5e-8,
    )


def test_compiled_pcm_rys_surface_integrals_match_python_fallback(monkeypatch):
    if pcm_integrals._rys_cy is None or not hasattr(
        pcm_integrals._rys_cy,
        "compute_surface_charge_ao_coulomb_rys",
    ):
        pytest.skip("Compiled PCM Rys kernel is not available.")

    mol = Molecule(
        atom="H 0 0 0; H 0 0 1.4",
        unit="bohr",
        basis="sto-3g",
    )
    mol.build(eri="s8")

    pcm = PCM(mol)
    pcm.lebedev_order = 3
    pcm.integral_backend = "native"
    pcm.build()
    coords = pcm.surface["grid_coords"]
    exponents = pcm.surface["charge_exp"]**2

    compiled = pcm_integrals.surface_charge_ao_coulomb(mol, coords, exponents)
    monkeypatch.setattr(pcm_integrals, "_rys_cy", None)
    fallback = pcm_integrals.surface_charge_ao_coulomb(mol, coords, exponents)

    np.testing.assert_allclose(compiled, fallback, atol=1e-10, rtol=1e-10)
