import numpy as np

from examples.mps.adaptive_physical_tying_annni import benchmark_point


def _by_method(rows):
    return {row["method"]: row for row in rows}


def test_annni_adaptive_parent_tracks_nearest_coupling_without_frustration():
    rows = _by_method(
        benchmark_point(
            nsites=6,
            nearest=1.0,
            next_nearest=0.0,
            field=0.8,
            relative_tolerance=1.0e-14,
        )
    )

    np.testing.assert_allclose(
        rows["A1"]["fidelity"],
        rows["R1"]["fidelity"],
        atol=1.0e-12,
    )
    assert rows["A1"]["distances"] == "1,1,1,1,1"


def test_annni_adaptive_parent_finds_nonlocal_frustrating_coupling():
    rows = _by_method(
        benchmark_point(
            nsites=6,
            nearest=1.0,
            next_nearest=1.0,
            field=0.8,
            relative_tolerance=1.0e-14,
        )
    )

    assert rows["A1"]["fidelity"] > rows["R1"]["fidelity"]
    assert rows["A1"]["energy_error"] < rows["R1"]["energy_error"]
    assert rows["A1"]["distances"].startswith("2,2")
    assert rows["A2"]["fidelity"] > rows["A1"]["fidelity"]


def test_annni_fixed_graph_variational_sweep_lowers_energy():
    rows = _by_method(
        benchmark_point(
            nsites=6,
            nearest=1.0,
            next_nearest=1.0,
            field=0.8,
            relative_tolerance=1.0e-14,
            variational_sweeps=4,
            variational_noise=1.0e-6,
            variational_seed=7,
        )
    )

    for method in ("R1", "R2", "A1", "A2"):
        assert rows[method]["variational_energy_error"] < rows[method]["energy_error"]


def test_annni_parent_rr_improves_energy_suboptimal_gram_graph():
    rows = _by_method(
        benchmark_point(
            nsites=4,
            nearest=1.0,
            next_nearest=0.6,
            field=0.8,
            relative_tolerance=0.0,
            variational_sweeps=8,
            rr_graph_sweeps=1,
            rr_tensor_sweeps=0,
        )
    )

    assert rows["A1"]["distances"] == "1,2,1"
    assert rows["RR1"]["distances"] == "1,1,1"
    assert rows["RR1"]["rr_graph_changes"] == 1
    assert (
        rows["RR1"]["variational_energy_error"]
        < rows["A1"]["variational_energy_error"]
    )


def test_annni_three_parent_budget_has_eight_physical_contexts():
    rows = _by_method(
        benchmark_point(
            nsites=4,
            nearest=1.0,
            next_nearest=0.8,
            field=0.8,
            relative_tolerance=0.0,
            tie_budgets=(3,),
        )
    )

    assert set(rows) == {"R0", "R3", "A3"}
    np.testing.assert_allclose(rows["R3"]["fidelity"], 1.0, atol=1.0e-13)
    np.testing.assert_allclose(rows["A3"]["fidelity"], 1.0, atol=1.0e-13)
    assert rows["R3"]["entries"] == 30
    assert rows["R3"]["distances"] == "1+2+3,1+2,1"
