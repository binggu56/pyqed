#!/usr/bin/env python3
"""Merge independently propagated phenol GP and NGP branch outputs."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np

from examples.namd.phenol_sa_casscf_5d_gp_control import (
    _save_results,
    _state_overlap,
)
from pyqed.mps import MPS


def _checkpoint_config(directory, mode):
    path = directory / f"{mode}_checkpoint.npz"
    with np.load(path, allow_pickle=False) as saved:
        return json.loads(str(saved["config_json"])), int(saved["step"])


def _load_result(directory, mode):
    summary = json.loads((directory / "summary.json").read_text())
    record = summary["dynamics"]["results"]
    if len(record) != 1 or record[0]["mode"] != mode:
        raise ValueError(f"{directory} is not a completed {mode.upper()} branch")
    with np.load(directory / "phenol_5d_gp_ngp.npz", allow_pickle=False) as saved:
        axes = tuple(np.asarray(saved[f"axis_{site}"]) for site in range(5))
        initial = tuple(
            np.asarray(saved[f"initial_marginal_{site}"]) for site in range(5)
        )
        marginals = tuple(
            np.asarray(saved[f"final_marginal_{site}_{mode}"]) for site in range(5)
        )
        factors = []
        site = 0
        while f"final_factor_{site}_{mode}" in saved:
            factors.append(np.asarray(saved[f"final_factor_{site}_{mode}"]))
            site += 1
        history = {
            key: np.asarray(saved[f"{key}_{mode}"])
            for key in (
                "times_fs",
                "norms",
                "cap_yield",
                "absorbed",
                "closure",
                "tdvp_truncation_error",
                "tdvp_norm_defect",
            )
        }
    result = {
        "mode": mode,
        **history,
        "marginals": marginals,
        "seconds": record[0]["seconds"],
        "final_ranks": record[0]["final_ranks"],
        "final_state": MPS(factors),
    }
    return summary, axes, initial, result


def merge(gp_directory, ngp_directory, output):
    gp_summary, gp_axes, gp_initial, gp = _load_result(gp_directory, "gp")
    ngp_summary, ngp_axes, ngp_initial, ngp = _load_result(ngp_directory, "ngp")
    for left, right in zip(gp_axes, ngp_axes):
        np.testing.assert_array_equal(left, right)
    for left, right in zip(gp_initial, ngp_initial):
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1.0e-13)

    gp_config, gp_step = _checkpoint_config(gp_directory, "gp")
    ngp_config, ngp_step = _checkpoint_config(ngp_directory, "ngp")
    gp_protocol = {key: value for key, value in gp_config.items() if key != "mode"}
    ngp_protocol = {key: value for key, value in ngp_config.items() if key != "mode"}
    if gp_protocol != ngp_protocol or gp_step != ngp_step:
        raise ValueError("GP and NGP branches do not share one propagation protocol")

    overlap = _state_overlap(gp["final_state"], ngp["final_state"])
    denominator = (
        gp["final_state"].norm_squared() * ngp["final_state"].norm_squared()
    )
    fidelity = float(abs(overlap) ** 2 / denominator)
    summary = copy.deepcopy(gp_summary)
    summary["operators"].update(ngp_summary["operators"])
    summary["dynamics"]["modes"] = ["gp", "ngp"]
    summary["dynamics"]["results"] = [
        gp_summary["dynamics"]["results"][0],
        ngp_summary["dynamics"]["results"][0],
    ]
    summary["dynamics"]["final_gp_ngp_fidelity"] = fidelity
    summary["dynamics"]["merged_from"] = [str(gp_directory), str(ngp_directory)]

    output.mkdir(parents=True, exist_ok=True)
    _save_results(
        output,
        gp_axes,
        gp_initial,
        [gp, ngp],
        gp_summary["fields"]["info"],
        summary,
    )
    for suffix in ("png", "pdf"):
        for stem in ("phenol_5d_gp_ngp_setup", "phenol_5d_gp_ngp_link_magnitudes"):
            source = gp_directory / f"{stem}.{suffix}"
            if source.is_file():
                shutil.copy2(source, output / source.name)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gp", type=Path, required=True)
    parser.add_argument("--ngp", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(merge(args.gp, args.ngp, args.output), indent=2))


if __name__ == "__main__":
    main()
