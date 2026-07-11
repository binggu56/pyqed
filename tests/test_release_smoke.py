"""Release gates for the public installation and quickstart contract."""

from pathlib import Path
from importlib.util import find_spec
import runpy

import pytest

import pyqed
from pyqed.qchem.basis import _basis_path


def test_runtime_version_uses_the_release_source():
    assert pyqed.__version__ == "0.2.0"


def test_required_basis_data_is_packaged():
    basis_file = Path(_basis_path("sto-3g"))
    assert basis_file.is_file()
    assert basis_file.stat().st_size > 0


def test_public_quickstart_runs(capsys):
    quickstart = Path(__file__).parents[1] / "examples" / "quickstart.py"
    namespace = runpy.run_path(str(quickstart))

    assert namespace["mf"].converged
    assert namespace["mf"].e_tot == pytest.approx(-1.116759310293, abs=1.0e-10)
    assert "RHF energy:" in capsys.readouterr().out


def test_documented_package_paths_are_distributed():
    package_paths = (
        "pyqed.HEOM",
        "pyqed.HEOM.deom",
        "pyqed.floquet",
        "pyqed.gw",
        "pyqed.md",
        "pyqed.ml",
        "pyqed.namd",
        "pyqed.narg",
        "pyqed.mps.nonabelian",
        "pyqed.qchem.mp",
        "pyqed.qchem.dft",
    )
    assert all(find_spec(name) is not None for name in package_paths)

    from pyqed.gw import BSE, GW, TDA
    from pyqed.HEOM.heom import HEOMSolver
    from pyqed.namd import TDDFTDriver
    from pyqed.qchem import MP2, Molecule
    from pyqed.qchem.dft import RKS

    assert all(
        item is not None
        for item in (BSE, GW, TDA, HEOMSolver, MP2, Molecule, RKS, TDDFTDriver)
    )
