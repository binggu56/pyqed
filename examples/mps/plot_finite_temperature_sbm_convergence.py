#!/usr/bin/env python3
"""Combine cached finite-temperature SBM convergence cases into one figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from finite_temperature_sbm_thermofield_dynamics import _make_figure


ARRAY_FIELDS = (
    "sigma_z",
    "coherence",
    "norm",
    "energy",
    "max_bond",
    "fock_edge_population",
    "max_occupation",
    "truncation",
    "krylov_residual",
    "step_seconds",
)


def _selection(value):
    try:
        path, index = value.rsplit(":", 1)
        return Path(path), int(index)
    except (ValueError, TypeError) as error:
        raise argparse.ArgumentTypeError("selections must have the form directory:index") from error


def _load_case(directory, index):
    summary = json.loads((directory / "summary.json").read_text())
    record = summary["cases"][index]
    data = np.load(directory / "trajectories.npz")
    run = {
        field: np.asarray(data[f"case{index}_{field}"])
        for field in ARRAY_FIELDS
    }
    return {
        "nmodes": int(record["nmodes"]),
        "local_dim": int(record["local_dim"]),
        "max_bond": int(record["max_bond"]),
        "run": run,
        "record": record,
    }, summary["model"], np.asarray(data["time"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--select", type=_selection, action="append", required=True)
    parser.add_argument("--zero-source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    cases = []
    model = times = None
    for directory, index in args.select:
        case, case_model, case_times = _load_case(directory, index)
        if times is not None and not np.array_equal(times, case_times):
            raise ValueError("all selected cases must use the same time grid")
        if model is not None and float(model["temperature"]) != float(case_model["temperature"]):
            raise ValueError("all selected cases must use the same temperature")
        cases.append(case)
        model = case_model
        times = case_times

    zero_temperature = None
    if args.zero_source is not None:
        data = np.load(args.zero_source / "trajectories.npz")
        zero_temperature = {
            "run": {
                field: np.asarray(data[f"zero_temperature_{field}"])
                for field in ARRAY_FIELDS
            }
        }

    reference = cases[-1]["run"]
    output_arrays = {"time": times}
    output_records = []
    for index, case in enumerate(cases):
        run = case["run"]
        record = {
            "nmodes": case["nmodes"],
            "local_dim": case["local_dim"],
            "max_bond": case["max_bond"],
            "max_sigma_z_error_vs_reference": float(
                np.max(np.abs(run["sigma_z"] - reference["sigma_z"]))
            ),
            "max_fock_edge_population": float(
                np.max(run["fock_edge_population"])
            ),
            "max_norm_error": float(np.max(np.abs(run["norm"] - 1.0))),
            "max_energy_drift": float(
                np.max(np.abs(run["energy"] - run["energy"][0]))
            ),
        }
        output_records.append(record)
        for field, values in run.items():
            output_arrays[f"case{index}_{field}"] = values
    if zero_temperature is not None:
        for field, values in zero_temperature["run"].items():
            output_arrays[f"zero_temperature_{field}"] = values

    combined_model = dict(model)
    combined_model["nmodes_per_branch"] = [case["nmodes"] for case in cases]
    combined_model["purified_sites"] = [2 * case["nmodes"] + 1 for case in cases]
    combined = {"model": combined_model, "cases": output_records}
    (args.output / "summary.json").write_text(json.dumps(combined, indent=2))
    np.savez_compressed(args.output / "trajectories.npz", **output_arrays)
    _make_figure(
        times,
        cases,
        zero_temperature,
        args.output / "high_temperature_sbm_convergence.png",
        float(model["temperature"]),
    )


if __name__ == "__main__":
    main()
