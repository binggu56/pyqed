from types import SimpleNamespace

import numpy as np
import pytest

from examples.namd.generate_so2_cas88_somf import (
    augment_from_saved_orbitals,
    protocol,
    repair_v2_grouped_soc_operator,
    validate_spin_selection,
)


def test_so2_soc_protocol_records_exact_spin_selection():
    args = SimpleNamespace(
        basis="6-31g*",
        nelecas=8,
        ncas=8,
        singlet_roots=3,
        triplet_roots=3,
        singlet_candidates=6,
        triplet_candidates=6,
        orbital_backend="pyscf",
        symmetry_adapted=True,
    )
    value = protocol(args)
    assert value["schema"] == "pyqed-so2-cas88-somf-v6"
    assert value["orbitals"]["point_group_constrained"]
    assert value["state_interaction"]["exact_spin_selection"]
    assert value["state_interaction"]["spin_orbital_layout"].startswith("grouped")
    assert value["state_interaction"]["singlet_candidate_roots"] == 6
    assert value["state_interaction"]["root_selection"]["target_parities"] == [
        1,
        -1,
        -1,
    ]


def test_so2_soc_dataset_rejects_forbidden_singlet_block():
    h_scalar = np.diag([0.0, 0.2, 0.4, 0.5]).astype(complex)
    h_soc = np.asarray(
        [
            [0.0, 1.0e-4j, 0.0, 2.0e-4],
            [-1.0e-4j, 0.0, 3.0e-4, 0.0],
            [0.0, 3.0e-4, 0.0, 4.0e-4j],
            [2.0e-4, 0.0, -4.0e-4j, 0.0],
        ]
    )
    record = {
        "h_scalar": h_scalar,
        "h_soc": h_soc,
        "diagnostics": {},
    }
    with pytest.raises(RuntimeError, match="selection rule"):
        validate_spin_selection(record, 2)


def test_v2_soc_permutation_repair_recovers_grouped_matrix():
    grouped = np.arange(64).reshape(8, 8)
    interleaved_from_grouped = np.asarray((0, 4, 1, 5, 2, 6, 3, 7))
    grouped_from_interleaved = np.argsort(interleaved_from_grouped)
    legacy = grouped[np.ix_(grouped_from_interleaved, grouped_from_interleaved)]
    np.testing.assert_array_equal(
        repair_v2_grouped_soc_operator(legacy), grouped
    )


def test_v6_augmentation_reuses_orbitals_without_running_scf(monkeypatch):
    class FakeMol:
        overlap = np.eye(2)

        @staticmethod
        def energy_nuc():
            return 3.0

    class FakeRHF:
        def __init__(self, mol):
            self.mol = mol

        def run(self, **_kwargs):
            raise AssertionError("SCF must not run during v5-to-v6 augmentation")

    captured = {}

    def fake_build(coordinate, args, **kwargs):
        captured.update(kwargs)
        return {"diagnostics": {}}

    monkeypatch.setattr(
        "examples.namd.generate_so2_cas88_somf.molecule_at",
        lambda coordinate, basis: FakeMol(),
    )
    monkeypatch.setattr(
        "examples.namd.generate_so2_cas88_somf.RHF", FakeRHF
    )
    monkeypatch.setattr(
        "examples.namd.generate_so2_cas88_somf.build_state_interaction_record",
        fake_build,
    )
    source = {
        "coordinate": np.asarray((2.7, 2.7, 2.0)),
        "mo_coeff": np.eye(2),
        "rhf_energy": np.asarray(-5.0),
        "orbital_history": [],
    }
    result = augment_from_saved_orbitals(
        source, SimpleNamespace(basis="6-31g*")
    )
    assert captured["orbital_source"] == "reused-v5-casscf"
    np.testing.assert_array_equal(captured["common_mo"], np.eye(2))
    assert result["diagnostics"]["saved_orbital_orthonormality_defect"] == 0.0
