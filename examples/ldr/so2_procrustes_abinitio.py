#!/usr/bin/env python3
"""Fit SO2 Procrustes-aligned energy and link fields from on-demand CASCI."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from examples.ldr.so2_casci_cgldr import casci_overlap_active
from examples.ldr.so2_casci_full_ldr import STATE_IDS, _electronic_point
from examples.ldr.so2_procrustes_gauge import reference_index
from pyqed.ldr import AbInitioFit


@dataclass(frozen=True)
class SO2Builder:
    grids: tuple
    basis: str
    integral_workers: int = 1

    def __call__(self, index):
        return _electronic_point(
            (
                tuple(index),
                float(self.grids[0][index[0]]),
                float(self.grids[1][index[1]]),
                float(self.grids[2][index[2]]),
                self.basis,
                int(self.integral_workers),
            )
        )


def load_grids(filename):
    with np.load(filename, allow_pickle=False) as archive:
        if all(name in archive for name in ("qs", "theta", "qa")):
            return tuple(
                np.asarray(archive[name], dtype=float)
                for name in ("qs", "theta", "qa")
            )
        keys = [f"grid_{axis}" for axis in range(3)]
        if all(key in archive for key in keys):
            return tuple(np.asarray(archive[key], dtype=float) for key in keys)
    raise ValueError("grid archive must contain qs/theta/qa or grid_0/grid_1/grid_2")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("grids", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--point-cache", type=Path)
    parser.add_argument("--basis", default="6-31g*")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--integral-workers", type=int, default=1)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--energy-rank", type=int)
    parser.add_argument("--link-rank", type=int)
    parser.add_argument("--degree", type=int, default=8)
    parser.add_argument("--sweeps", type=int, default=8)
    parser.add_argument("--rtol", type=float, default=1.0e-8)
    parser.add_argument("--validation", type=int, default=128)
    parser.add_argument("--start-rank", type=int, default=1)
    parser.add_argument("--kick-rank", type=int, default=2)
    parser.add_argument(
        "--sampler", choices=("cross", "block-cross", "sparse"), default="cross"
    )
    parser.add_argument("--fit-sweeps", type=int, default=12)
    parser.add_argument("--initial", type=int, default=64)
    parser.add_argument("--rounds", type=int, default=6)
    parser.add_argument("--regularization", type=float, default=1.0e-10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    grids = load_grids(args.grids)
    shape = tuple(len(grid) for grid in grids)
    if any(size < 3 for size in shape):
        raise ValueError("each SO2 coordinate grid needs at least three points")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    point_cache = args.point_cache or args.output_dir / "point_cache"

    def report(_index, stats):
        count = stats["built"]
        if count == 1 or count % 10 == 0:
            print(
                f"[CASCI] built={count}, restored={stats['restored']}, "
                f"unique={stats['unique_requested']}",
                flush=True,
            )

    fit = AbInitioFit(
        grids,
        len(STATE_IDS),
        SO2Builder(grids, args.basis, args.integral_workers),
        anchor=reference_index(grids),
        frame=lambda record: record[1],
        energies=lambda record: record[2],
        overlap=lambda left, right: casci_overlap_active(left, right, STATE_IDS),
        energy_shift=None,
        cache=point_cache,
        workers=args.workers,
        progress=report,
    )
    try:
        fit.run(
            sampler=args.sampler,
            rank=args.rank,
            energy_rank=args.energy_rank,
            link_rank=args.link_rank,
            degrees=args.degree,
            sweeps=args.sweeps,
            rtol=args.rtol,
            validation=args.validation,
            seed=args.seed,
            start_rank=args.start_rank,
            kick_rank=args.kick_rank,
            fit_sweeps=args.fit_sweeps,
            initial=args.initial,
            rounds=args.rounds,
            regularization=args.regularization,
        )
    finally:
        fit.close()

    fit.save(
        args.output_dir,
        labels=("qs", "theta", "qa"),
        metadata={
            "method": "CASCI",
            "basis": args.basis,
            "point_cache": str(point_cache),
        },
    )
    print(
        f"wrote {fit.paths['summary']}; "
        f"QC calls={fit.info['quantum_chemistry_calls']}, "
        f"unique geometries={fit.info['unique_geometries']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
