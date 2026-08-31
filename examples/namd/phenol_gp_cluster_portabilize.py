#!/usr/bin/env python3
"""Rewrite copied phenol GP/NGP cache identities for a new filesystem root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pyqed.cache import file_signature


def _atomic_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _rewrite_metadata(path, signatures):
    payload = json.loads(path.read_text())
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        raise ValueError(f"cache metadata has no spec mapping: {path}")
    for key, source in signatures.items():
        if key in spec:
            spec[key] = file_signature(source)
    _atomic_json(path, payload)


def _rewrite_checkpoint(path, signatures):
    if path is None or not path.is_file():
        return
    with np.load(path, allow_pickle=False) as saved:
        payload = {key: np.asarray(saved[key]) for key in saved.files}
    config = json.loads(str(payload["config_json"]))
    for key, source in signatures.items():
        if key in config:
            config[key] = file_signature(source)
    payload["config_json"] = np.asarray(json.dumps(config, sort_keys=True))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **payload)
    temporary.replace(path)


def portabilize(args):
    field_metadata = args.field_cache / "metadata.json"
    residual_metadata = args.residual_cache / "metadata.json"
    keo_metadata = args.keo_cache / "metadata.json"
    operator_metadata = [
        args.gp_operator_cache / "metadata.json",
        args.ngp_operator_cache / "metadata.json",
    ]
    required = (
        args.checkpoint,
        args.radial_correction,
        args.initial_state,
        field_metadata,
        residual_metadata,
        keo_metadata,
        *operator_metadata,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing staged inputs: " + ", ".join(missing))

    _rewrite_metadata(
        field_metadata,
        {
            "checkpoint": args.checkpoint,
            "radial_correction": args.radial_correction,
        },
    )
    _rewrite_metadata(
        residual_metadata,
        {
            "base_field": field_metadata,
            "checkpoint": args.checkpoint,
            "radial_correction": args.radial_correction,
            "initial_state": args.initial_state,
        },
    )
    for path in operator_metadata:
        _rewrite_metadata(
            path,
            {
                "field": field_metadata,
                "keo": keo_metadata,
                "potential_residual": residual_metadata,
            },
        )
    checkpoint_signatures = {
        "field": field_metadata,
        "potential_residual": residual_metadata,
        "keo": keo_metadata,
        "initial_state": args.initial_state,
    }
    for path in args.branch_checkpoint:
        _rewrite_checkpoint(path, checkpoint_signatures)

    manifest = {
        "checkpoint": file_signature(args.checkpoint),
        "radial_correction": file_signature(args.radial_correction),
        "initial_state": file_signature(args.initial_state),
        "field_metadata": file_signature(field_metadata),
        "residual_metadata": file_signature(residual_metadata),
        "keo_metadata": file_signature(keo_metadata),
        "operator_metadata": [file_signature(path) for path in operator_metadata],
        "branch_checkpoints": [
            file_signature(path) for path in args.branch_checkpoint if path.is_file()
        ],
    }
    output = args.manifest or field_metadata.parent.parent / "cluster_manifest.json"
    _atomic_json(output, manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--radial-correction", type=Path, required=True)
    parser.add_argument("--initial-state", type=Path, required=True)
    parser.add_argument("--keo-cache", type=Path, required=True)
    parser.add_argument("--field-cache", type=Path, required=True)
    parser.add_argument("--residual-cache", type=Path, required=True)
    parser.add_argument("--gp-operator-cache", type=Path, required=True)
    parser.add_argument("--ngp-operator-cache", type=Path, required=True)
    parser.add_argument("--branch-checkpoint", type=Path, action="append", default=[])
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(portabilize(args), indent=2))


if __name__ == "__main__":
    main()
