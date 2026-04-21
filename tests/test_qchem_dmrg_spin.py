import numpy as np
import pytest

from pyqed.qchem import Molecule
from pyqed.qchem.hf import RHF
from pyqed.qchem.dmrg.dmrg import (
    DMRG,
    _build_s2_term_map,
    _build_spin_purification_term_map,
)


def _kron_all(ops):
    out = ops[0]
    for op in ops[1:]:
        out = np.kron(out, op)
    return out


def _spin_orbital_annihilation(site, nsites):
    ident = np.eye(2)
    z = np.diag([1.0, -1.0])
    a = np.array([[0.0, 1.0], [0.0, 0.0]])
    ops = [z] * site + [a] + [ident] * (nsites - site - 1)
    return _kron_all(ops)


def _spin_orbital_creation(site, nsites):
    return _spin_orbital_annihilation(site, nsites).T


def _dense_from_symbolic_term_map(term_map, nsites):
    ident = np.eye(2)
    loc = {
        "a": np.array([[0.0, 1.0], [0.0, 0.0]]),
        r"a^\dagger": np.array([[0.0, 0.0], [1.0, 0.0]]),
        "n": np.diag([0.0, 1.0]),
        "sigma_z": np.diag([1.0, -1.0]),
    }
    out = np.zeros((2**nsites, 2**nsites), dtype=complex)
    for (symbol, dofs), factor in term_map.items():
        pieces = symbol.split()
        ops = [ident.copy() for _ in range(nsites)]
        for site, piece in zip(dofs, pieces):
            ops[site] = loc[piece]
        out += factor * _kron_all(ops)
    return out


def _dense_spin_square(ncas):
    nsites = 2 * ncas
    sp = np.zeros((2**nsites, 2**nsites), dtype=complex)
    sm = np.zeros_like(sp)
    sz = np.zeros_like(sp)
    for p in range(ncas):
        cup_dag = _spin_orbital_creation(2 * p, nsites)
        cup = _spin_orbital_annihilation(2 * p, nsites)
        cdn_dag = _spin_orbital_creation(2 * p + 1, nsites)
        cdn = _spin_orbital_annihilation(2 * p + 1, nsites)
        sp += cup_dag @ cdn
        sm += cdn_dag @ cup
        sz += 0.5 * (cup_dag @ cup - cdn_dag @ cdn)
    return sz @ sz + 0.5 * (sp @ sm + sm @ sp)


def test_dmrg_s2_term_map_matches_exact_dense_spin_square():
    ncas = 2
    nsites = 2 * ncas
    exact = _dense_spin_square(ncas)
    symbolic = _dense_from_symbolic_term_map(_build_s2_term_map(ncas), nsites)
    np.testing.assert_allclose(symbolic, exact, atol=1e-12)


def test_spin_purification_term_map_is_scaled_s2():
    ncas = 3
    shift = 0.37
    ref = _build_s2_term_map(ncas, scale=shift)
    got = _build_spin_purification_term_map(ncas, shift)
    assert set(ref) == set(got)
    for key in ref:
        assert np.allclose(got[key], ref[key], atol=1e-12)


def test_dmrg_fix_spin_accepts_non_singlet_targets_and_warns_for_linear_penalty():
    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build(driver="gbasis")
    mf = RHF(mol).run()
    dmrg = DMRG(mf, ncas=2, nelecas=2, D=4, init_guess="hf")

    with pytest.warns(RuntimeWarning, match="linear \\+shift\\*S\\^2 penalty"):
        dmrg.fix_spin(ss=2, shift=0.3)

    assert dmrg.spin_purification is True
    assert dmrg.ss == pytest.approx(2.0)
    assert dmrg.shift == pytest.approx(0.3)
