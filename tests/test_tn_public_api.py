import inspect

import pyqed.letta as letta
import pyqed.mps as mps
from pyqed.mps.nonabelian import AutoMPO
from pyqed.operator_mpo import ModelMPO
from pyqed.tn import Hamiltonian, LocalTerm, MPO, MPS, OperatorString


def test_canonical_tensor_network_surface_is_explicit():
    assert mps.MPS is MPS
    assert mps.MPO is MPO
    assert set(mps.__all__) == set(dir(mps))
    assert "AutoMPO" not in mps.__all__
    assert "Block" not in mps.__all__
    assert "Site" not in mps.__all__


def test_backend_specific_mpo_builders_have_distinct_namespaces():
    assert AutoMPO.__name__ == "AutoMPO"
    assert ModelMPO.__name__ == "ModelMPO"
    assert not hasattr(letta, "LocalHamiltonian")
    assert not hasattr(letta, "LocalTerm")


def test_canonical_constructor_and_attribute_names():
    assert tuple(inspect.signature(MPS).parameters)[:1] == ("tensors",)
    assert tuple(inspect.signature(MPO).parameters)[:1] == ("tensors",)
    assert "homogeneous" in inspect.signature(MPS).parameters
    assert "homogeneous" in inspect.signature(MPO).parameters
    assert MPO.STANDARD_LABELS == ("left", "right", "out", "in")


def test_hamiltonian_types_live_in_tn():
    assert Hamiltonian.__module__ == "pyqed.tn.hamiltonian"
    assert Hamiltonian.__name__ == "Hamiltonian"
    assert LocalTerm.__module__ == "pyqed.tn.hamiltonian"
    assert OperatorString.__module__ == "pyqed.tn.hamiltonian"
