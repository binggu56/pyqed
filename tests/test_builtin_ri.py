import sys
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _use_source_tree_pyqed():
    for name in list(sys.modules):
        if name == "pyqed" or name.startswith("pyqed."):
            del sys.modules[name]
    if str(ROOT) in sys.path:
        sys.path.remove(str(ROOT))
    sys.path.insert(0, str(ROOT))


def _pyqed_basis_as_pyscf(name, symbols):
    from pyqed.qchem.basis import _basis_path, parse_gbs

    basis_dict = parse_gbs(_basis_path(name))
    out = {}
    for sym in sorted(set(symbols)):
        shells = []
        for l, exps, coeffs in basis_dict[sym]:
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.ndim == 1:
                coeffs = coeffs[:, None]
            block = [int(l)]
            for ip, exp in enumerate(exps):
                block.append([float(exp), *[float(x) for x in coeffs[ip]]])
            shells.append(block)
        out[sym] = shells
    return out


def _pyscf_ri_eri(mol, auxbasis):
    from pyscf import df

    auxmol = df.addons.make_auxmol(mol, auxbasis)
    metric = auxmol.intor("int2c2e")
    j3 = df.incore.aux_e2(mol, auxmol, intor="int3c2e", aosym="s1")
    if j3.shape[0] == mol.nao_nr():
        j3 = np.moveaxis(j3, -1, 0)
    evals, evecs = np.linalg.eigh(metric)
    keep = evals > 1.0e-10
    invsqrt = (evecs[:, keep] / np.sqrt(evals[keep])) @ evecs[:, keep].T
    factors = np.einsum("PQ,Qij->Pij", invsqrt, j3, optimize=True)
    return np.einsum("Pij,Pkl->ijkl", factors, factors, optimize=True)


def test_builtin_native_ri_builds_factors_without_dense_eri():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri")

    assert mol.eri is None
    assert mol.eri_factors is not None
    assert mol.eri_factors.shape == (46, mol.nao, mol.nao)
    assert mol._builtin_build_info["factor_builder"] == "native-ri"
    assert mol._builtin_build_info["ri"]["auxbasis"] == "cc-pvdz-jkfit"
    assert mol._builtin_build_info["ri"]["metric_solver"] == "cholesky"
    assert mol._builtin_build_info["ri"]["storage"] == "full"
    assert mol._builtin_build_info["ri"]["tensor_builder"] in {
        "cpp-kernel-packed",
        "cython-kernel-packed",
        "cython-kernel-packed-parallel",
        "python",
        "python-parallel",
        "cpp-kernel-packed-spherical-pair-blocked",
        "cython-kernel-packed-spherical-pair-blocked",
        "cython-kernel-packed-parallel-spherical-pair-blocked",
    }


def test_builtin_native_ri_ignores_dense_aosym_keyword():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri", aosym="s8")

    assert mol.eri is None
    assert mol.eri_s8 is None
    assert mol.eri_factors is not None
    assert mol._builtin_build_info["representation"] == "ri"
    assert mol._builtin_build_info["aosym"] == "s1"


def test_builtin_native_ri_accepts_auxbasis_keyword():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri", auxbasis="cc-pvdz-rifit")

    assert mol.eri is None
    assert mol.eri_factors.shape == (28, mol.nao, mol.nao)
    assert mol._builtin_build_info["ri"]["auxbasis"] == "cc-pvdz-rifit"
    assert mol._builtin_build_info["ri"]["storage"] == "full"


def test_builtin_native_ri_defaults_pople_to_cc_pvdz_jkfit():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="6-31g",
        unit="angstrom",
    )
    mol.build(eri="ri")

    assert mol.eri is None
    assert mol.eri_factors is not None
    assert mol._builtin_build_info["ri"]["auxbasis"] == "cc-pvdz-jkfit"


def test_builtin_native_ri_falls_back_to_fe_capable_jkfit():
    _use_source_tree_pyqed()
    from pyqed.qchem.basis import _default_auxbasis_name

    auxbasis = _default_auxbasis_name(
        "sto-3g",
        purpose="jk",
        required_symbols=("Fe", "N", "C", "H"),
    )

    assert auxbasis == "def2-universal-jkfit"


def test_builtin_native_ri_purpose_can_prefer_rifit():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri", options={"ri_purpose": "ri"})

    assert mol._builtin_build_info["ri"]["auxbasis"] == "cc-pvdz-rifit"
    assert mol._builtin_build_info["ri"]["purpose"] == "ri"
    assert mol._builtin_build_info["ri"]["storage"] == "packed"


def test_builtin_native_ri_packed_storage_option():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    mol = Molecule(
        atom="H 0 0 0; H 0 0 0.74",
        basis="cc-pvdz",
        unit="angstrom",
    )
    mol.build(eri="ri", options={"ri_storage": "packed"})

    assert mol._builtin_build_info["ri"]["storage"] == "packed"
    assert mol.eri_factors.pair_shape == (46, mol.nao * (mol.nao + 1) // 2)


def test_builtin_native_ri_full_storage_option_matches_packed_jk():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule
    from pyqed.qchem.basis import contract_jk_ri, contract_jk_ri_mo

    atom = "H 0 0 0; H 0 0 0.74"
    packed = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    packed.build(eri="ri", auxbasis="cc-pvdz-rifit")

    full = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    full.build(eri="ri",
        auxbasis="cc-pvdz-rifit",
        options={"ri_storage": "full"},
    )

    rng = np.random.default_rng(123)
    dm = rng.normal(size=(packed.nao, packed.nao))
    dm = dm + dm.T
    vj_packed, vk_packed = contract_jk_ri(packed.eri_factors, dm, packed.nao)
    vj_full, vk_full = contract_jk_ri(full.eri_factors, dm, full.nao)

    assert full._builtin_build_info["ri"]["storage"] == "full"
    np.testing.assert_allclose(vj_packed, vj_full, atol=1e-11, rtol=1e-11)
    np.testing.assert_allclose(vk_packed, vk_full, atol=1e-11, rtol=1e-11)

    s = packed.overlap
    h = packed.hcore
    evals, evecs = np.linalg.eigh(s)
    x = evecs @ np.diag(evals ** -0.5) @ evecs.T
    _eps, c0 = np.linalg.eigh(x.T @ h @ x)
    mo_coeff = x @ c0
    mo_occ = np.zeros(packed.nao)
    mo_occ[: packed.nelec // 2] = 2.0
    dm_mo = mo_coeff @ np.diag(mo_occ) @ mo_coeff.T
    vj_dm, vk_dm = contract_jk_ri(packed.eri_factors, dm_mo, packed.nao)
    vj_mo, vk_mo = contract_jk_ri_mo(packed.eri_factors, mo_coeff, mo_occ, packed.nao)
    np.testing.assert_allclose(vj_mo, vj_dm, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(vk_mo, vk_dm, atol=1e-10, rtol=1e-10)


def test_builtin_native_ri_reconstructs_dense_eri_to_auxbasis_accuracy():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule

    atom = "H 0 0 0; H 0 0 0.74"
    dense = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    dense.build(eri="dense", aosym="s1")

    ri = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    ri.build(eri="ri")

    eri_ri = np.einsum("Pij,Pkl->ijkl", ri.eri_factors, ri.eri_factors, optimize=True)
    rel_error = np.linalg.norm(eri_ri - dense.eri) / np.linalg.norm(dense.eri)

    assert rel_error < 2.0e-3


def test_builtin_native_ri_rhf_runs_without_pyscf():
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    atom = "H 0 0 0; H 0 0 0.74"
    energies = []
    for eri in ("dense", "ri"):
        mol = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
        mol.build(eri=eri)
        mf = RHF(mol).run(verbose=0)
        energies.append(float(mf.e_tot))

    assert abs(energies[1] - energies[0]) < 5.0e-5


def test_builtin_native_ri_h2_matches_pyscf_df_integrals_and_energy():
    pyscf = pytest.importorskip("pyscf")
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule
    from pyqed.qchem.hf.rhf import RHF

    atom = "H 0 0 0; H 0 0 0.74"
    auxbasis = "cc-pvdz-rifit"
    mol = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    symbols = mol.atom_symbols()
    mol.build(eri="ri",
        auxbasis=auxbasis,
        options={"coord_type": "cartesian"},
    )
    mf = RHF(mol).run(verbose=0)

    pmol = pyscf.gto.M(
        atom=atom,
        basis=_pyqed_basis_as_pyscf("cc-pvdz", symbols),
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    pauxbasis = _pyqed_basis_as_pyscf(auxbasis, symbols)
    eri_ri = np.einsum("Pij,Pkl->ijkl", mol.eri_factors, mol.eri_factors, optimize=True)
    peri_ri = _pyscf_ri_eri(pmol, pauxbasis)
    pmf = pyscf.scf.RHF(pmol).density_fit(auxbasis=pauxbasis).run(verbose=0)

    np.testing.assert_allclose(eri_ri, peri_ri, rtol=1.0e-7, atol=1.0e-7)
    assert abs(float(mf.e_tot) - float(pmf.e_tot)) < 1.0e-8


def test_builtin_native_ri_casscf_matches_pyscf_cartesian_df():
    pyscf = pytest.importorskip("pyscf")
    pyscf_mcscf = pytest.importorskip("pyscf.mcscf")
    _use_source_tree_pyqed()
    from pyqed.qchem import Molecule, SecondOrderCASSCF
    from pyqed.qchem.hf.rhf import RHF

    atom = "O 0 0 0; H 0 -0.757 0.587; H 0 0.757 0.587"
    auxbasis = "cc-pvdz-jkfit"
    mol = Molecule(atom=atom, basis="cc-pvdz", unit="angstrom")
    symbols = mol.atom_symbols()
    mol.build(eri="ri",
        auxbasis=auxbasis,
        options={"coord_type": "cartesian"},
    )
    mf = RHF(mol).run(verbose=0)
    mc = SecondOrderCASSCF(
        mf,
        ncas=4,
        nelecas=4,
        max_cycle=30,
        max_micro_cycle=6,
        coupling="qn",
        ci_method="direct_ci",
        use_cholesky=True,
        conv_tol=1.0e-8,
        conv_tol_grad=1.0e-5,
        conv_tol_grad_relaxed=1.0e-4,
        conv_tol_step=1.0e-4,
        ah_max_cycle=6,
        ah_max_subspace=12,
        auto_active_restarts=False,
        verbose=0,
    ).run(nstates=1)

    pmol = pyscf.gto.M(
        atom=atom,
        basis=_pyqed_basis_as_pyscf("cc-pvdz", symbols),
        unit="Angstrom",
        cart=True,
        verbose=0,
    )
    pauxbasis = _pyqed_basis_as_pyscf(auxbasis, symbols)
    pmf = pyscf.scf.RHF(pmol).density_fit(auxbasis=pauxbasis)
    pmf.conv_tol = 1.0e-10
    pmf.verbose = 0
    pmf.kernel()
    pmc = pyscf_mcscf.CASSCF(pmf, 4, 4)
    pmc.conv_tol = 1.0e-8
    pmc.max_cycle_macro = 30
    pmc.max_cycle_micro = 6
    pmc.verbose = 0
    pmc.kernel(mo_coeff=pmf.mo_coeff)

    assert mc.converged
    assert pmc.converged
    assert abs(float(mf.e_tot) - float(pmf.e_tot)) < 1.0e-7
    assert abs(float(mc.e_tot[0]) - float(pmc.e_tot)) < 1.0e-6
