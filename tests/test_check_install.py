import importlib
from types import SimpleNamespace

import pytest

from pyqed.qchem import check_install
from pyqed.qchem import basis as basis_module


def test_installation_check_reports_complete_accelerators(monkeypatch, capsys):
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(required=lambda: None, __file__="kernel.so"),
    )
    monkeypatch.setattr(
        check_install,
        "REQUIRED_ACCELERATORS",
        (("test kernel", "pyqed.qchem._test_kernel", ("required",)),),
    )

    assert check_install.main() == 0
    assert "production path is available" in capsys.readouterr().out


def test_installation_check_fails_for_missing_accelerator(monkeypatch, capsys):
    def missing(_name):
        raise ImportError("not built")

    monkeypatch.setattr(importlib, "import_module", missing)
    monkeypatch.setattr(
        check_install,
        "REQUIRED_ACCELERATORS",
        (("test kernel", "pyqed.qchem._test_kernel", ("required",)),),
    )

    assert check_install.main() == 1
    output = capsys.readouterr().out
    assert "MISSING" in output
    assert "production path is incomplete" in output


def test_auto_integrals_do_not_silently_use_reference_kernel(monkeypatch):
    signatures = ((((0, 0, 0)), (0.0, 0.0, 0.0), (1.0,), (1.0,)),)
    monkeypatch.setattr(basis_module, "_integrals_cpp", None)

    with pytest.raises(RuntimeError, match="explicit slow reference"):
        basis_module._resolve_eri_backend("auto", signatures)

    assert basis_module._resolve_eri_backend("python", signatures) == "python"
