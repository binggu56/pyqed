from pyqed.qchem import Molecule


def test_molecule_build_returns_self(monkeypatch):
    monkeypatch.setattr("pyqed.qchem.mol.build_builtin", lambda molecule: None)
    molecule = Molecule("H 0 0 0", basis="sto-3g")

    assert molecule.build(eri="dense") is molecule
