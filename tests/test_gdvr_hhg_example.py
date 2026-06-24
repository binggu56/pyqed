import importlib.util
from pathlib import Path

import numpy as np


def _load_hhg_example():
    path = Path(__file__).resolve().parents[1] / "examples/qchem/gdvr_h2_hhg.py"
    spec = importlib.util.spec_from_file_location("gdvr_h2_hhg_example", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_half_wave_projection_suppresses_synthetic_even_harmonic():
    hhg = _load_hhg_example()
    omega0 = 0.2
    samples_per_cycle = 200
    cycles = 8
    dt = hhg.optical_period(omega0) / samples_per_cycle
    time = np.arange(cycles * samples_per_cycle) * dt
    signal = (
        np.sin(omega0 * time)
        + 0.3 * np.sin(2.0 * omega0 * time)
        + 0.2 * np.sin(3.0 * omega0 * time)
    )

    raw = hhg.hhg_spectrum(
        time,
        signal,
        omega0,
        acceleration=signal,
        zero_pad=8,
        harmonic_window=0.05,
        max_harmonic=6,
        symmetrize_half_wave=False,
    )
    projected = hhg.hhg_spectrum(
        time,
        signal,
        omega0,
        acceleration=signal,
        zero_pad=8,
        harmonic_window=0.05,
        max_harmonic=6,
        symmetrize_half_wave=True,
    )

    raw_ratio = hhg.symmetry_diagnostics(raw["harmonics"], value_col=4)[
        "max_even_over_max_odd"
    ]
    projected_ratio = hhg.symmetry_diagnostics(projected["harmonics"], value_col=4)[
        "max_even_over_max_odd"
    ]

    assert raw_ratio > 1.0
    assert projected_ratio < 1e-4


def test_flat_top_auto_analysis_window_uses_plateau():
    hhg = _load_hhg_example()
    omega0 = 0.057
    field = hhg.flat_top_pulse(0.08, omega0, cycles=8.0, ramp_cycles=1.0)

    start, stop, mode = hhg.analysis_bounds(field, "auto")

    assert mode == "flat-top"
    np.testing.assert_allclose(start / hhg.optical_period(omega0), 1.0)
    np.testing.assert_allclose((stop - start) / hhg.optical_period(omega0), 6.0)
