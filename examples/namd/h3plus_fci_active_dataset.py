#!/usr/bin/env python3
"""Build an active-learning H3+ FCI iteration around MACE outer-shell errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from pyqed.ldr import AbInitioFit, Coord
from pyqed.ml import MACE
from pyqed.qchem import Molecule
from pyqed.units import au2ev

from h3plus_fci_expanded_dataset import (
    EXPANDED_BOUNDS,
    SPECIES,
    geometry,
    graph_pairs,
    link_diagnostics,
    plot_sampling,
    s3_sampling_symmetry,
    state_gap_diagnostics,
    uniform_outer_shell,
    validation_pairs,
)
from h3plus_fci_expanded_mace import mace_geometry


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-dir", type=Path,
        default=root / "data/h3plus_fci_augccpvdz/expanded_dataset_v2",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--worst-count", type=int, default=60)
    parser.add_argument("--neighbors-per-worst", type=int, default=3)
    parser.add_argument("--outer-count", type=int, default=120)
    parser.add_argument("--validation-pair-budget", type=int, default=600)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with np.load(args.source_dir / "sampled_fields.npz") as source:
        source = {name: np.asarray(source[name]) for name in source.files}
    model = MACE.load(args.checkpoint, mace_geometry, device="cpu", distill=False)
    validation_h = model.neural_energy.predict(source["validation_coordinates"])
    validation_errors = np.max(
        np.abs(
            np.linalg.eigvalsh(validation_h)
            - np.linalg.eigvalsh(source["validation_hamiltonians"])
        ),
        axis=1,
    )
    worst_indices = np.argsort(validation_errors)[-int(args.worst_count):]
    worst = source["validation_coordinates"][worst_indices]

    random = np.random.default_rng(401)
    directions = random.normal(
        size=(len(worst), int(args.neighbors_per_worst), 3)
    )
    directions /= np.linalg.norm(directions, axis=-1, keepdims=True)
    radii = random.uniform(0.045, 0.14, size=(*directions.shape[:2], 1))
    neighborhoods = (worst[:, None, :] + radii * directions).reshape(-1, 3)
    box = np.asarray(EXPANDED_BOUNDS, dtype=float)
    neighborhoods = np.clip(neighborhoods, box[:, 0], box[:, 1])

    coord = Coord(to_cartesian=geometry, bounds=EXPANDED_BOUNDS)
    molecule = Molecule(
        atom=list(zip(SPECIES, mace_geometry((0.0, 0.0, 0.0)))),
        charge=1, spin=0, unit="bohr", basis="aug-cc-pvdz",
    ).build(eri="dense")
    mean_field = molecule.RHF().run()
    electronic = molecule.casci(
        molecule.nao, 2, nstates=6, ms2=0, multiplicity=1, mf=mean_field
    ).run(nstates=6)
    sampler = AbInitioFit(
        electronic,
        coord=coord,
        states=(1, 2),
        nroots=6,
        database=args.output_dir / "electronic.sqlite",
        symmetry=s3_sampling_symmetry(),
        workers=int(args.workers),
        progress=False,
    )
    active = sampler.reduce_coordinates(
        np.vstack(
            (
                source["training_coordinates"],
                source["validation_coordinates"],
                neighborhoods,
                uniform_outer_shell(409, int(args.outer_count)),
            )
        )
    )
    active_pairs = graph_pairs(active)
    fresh_validation, fresh_pairs = validation_pairs(
        503, sampler.reduced_size(int(args.validation_pair_budget))
    )
    fresh_validation, fresh_pairs = sampler.reduce_pairs(
        fresh_validation, fresh_pairs
    )

    started = perf_counter()
    training_fields = sampler.continuous_fields(active, active_pairs)
    validation_fields = sampler.continuous_fields(fresh_validation, fresh_pairs)
    elapsed = perf_counter() - started
    training_roots, training_gaps = state_gap_diagnostics(sampler, active)
    validation_roots, validation_gaps = state_gap_diagnostics(
        sampler, fresh_validation
    )
    training_singular, training_links = link_diagnostics(training_fields["links"])
    _validation_singular, validation_links = link_diagnostics(
        validation_fields["links"]
    )

    np.savez_compressed(
        args.output_dir / "sampled_fields.npz",
        training_coordinates=active,
        training_pairs=active_pairs,
        training_hamiltonians=training_fields["hamiltonians"],
        training_links=training_fields["links"],
        training_root_energies=training_roots,
        validation_coordinates=fresh_validation,
        validation_pairs=fresh_pairs,
        validation_hamiltonians=validation_fields["hamiltonians"],
        validation_links=validation_fields["links"],
        validation_root_energies=validation_roots,
    )
    figure = args.output_dir / "sampling_coverage_and_links.png"
    plot_sampling(active, fresh_validation, active_pairs, training_singular, figure)
    report = {
        "strategy": "promote v2 validation + neighborhoods of 60 worst PES points",
        "source": str(args.source_dir),
        "source_checkpoint": str(args.checkpoint),
        "training_coordinates": int(len(active)),
        "training_pairs": int(len(active_pairs)),
        "validation_coordinates": int(len(fresh_validation)),
        "validation_pairs": int(len(fresh_pairs)),
        "new_neighborhood_candidates": int(len(neighborhoods)),
        "new_outer_candidates": int(args.outer_count),
        "source_worst_error_mev": float(np.max(validation_errors) * au2ev * 1e3),
        "training_link_diagnostics": training_links,
        "validation_link_diagnostics": validation_links,
        "training_state_gap_diagnostics": training_gaps,
        "validation_state_gap_diagnostics": validation_gaps,
        "database_stats": dict(sampler.database.stats),
        "sampling_seconds": elapsed,
        "sampling_figure": str(figure),
        "accepted_for_mace_training": bool(
            validation_links["minimum_link_singular_value"] >= 0.9
            and validation_gaps["minimum_excluded_root_gap_hartree"] > 0.0
        ),
    }
    (args.output_dir / "dataset_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    sampler.close()
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
