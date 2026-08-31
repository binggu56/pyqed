#!/usr/bin/env python3
"""Plot the strict phenol SA-CASSCF optimizer validation at one geometry."""

from pyqed.units import au2ev
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HARTREE_TO_EV = au2ev


def load(path):
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("keyframe", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reference = load(args.reference)
    keyframe = load(args.keyframe)
    states = np.arange(reference["energies"].size)
    relative_reference = (
        reference["energies"] - reference["energies"][0]
    ) * HARTREE_TO_EV
    relative_keyframe = (
        keyframe["energies"] - keyframe["energies"][0]
    ) * HARTREE_TO_EV
    error_neh = np.abs(keyframe["energies"] - reference["energies"]) * 1.0e9

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.8), constrained_layout=True)
    ax = axes[0, 0]
    ax.plot(states, relative_reference, "o-", label="Full-CI microiterations")
    ax.plot(states, relative_keyframe, "s", mfc="none", ms=7, label="Keyframe CI")
    ax.set(xlabel="Electronic state", ylabel=r"$E_i-E_{S_0}$ (eV)")
    ax.set_xticks(states, [f"S{i}" for i in states])
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.bar(states, error_neh, color="#2a9d8f")
    ax.set(xlabel="Electronic state", ylabel=r"$|E_i^{\rm key}-E_i^{\rm ref}|$ (n$E_h$)")
    ax.set_xticks(states, [f"S{i}" for i in states])

    ax = axes[1, 0]
    for data, label, marker in (
        (reference, "Full-CI optimization", "o"),
        (keyframe, "Keyframe confirmation", "s"),
    ):
        history = data["macro_history"]
        ax.semilogy(history[:, 0], history[:, 2], marker + "-", label=label)
    ax.axhline(1.0e-5, color="0.35", ls="--", lw=1.2, label="Threshold")
    ax.set(xlabel="Macrocycle", ylabel=r"$\|g_{\rm orb}\|$")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    for data, label in (
        (reference, "Full-CI optimization"),
        (keyframe, "Keyframe confirmation"),
    ):
        overlap = data["active_overlap_history"]
        ax.plot(np.arange(1, len(overlap) + 1), np.min(overlap, axis=1), label=label)
    ax.axhline(0.35, color="0.35", ls="--", lw=1.2, label="Safety floor")
    ax.set(
        xlabel="Microiteration",
        ylabel="Minimum active-projector singular value",
        ylim=(0.3, 1.02),
    )
    ax.legend(frameon=False)

    fig.suptitle("Phenol SA(6)-CASSCF(10e,10o)/6-31+G* optimizer gate")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    fig.savefig(args.output.with_suffix(".pdf"))
    print(args.output)


if __name__ == "__main__":
    main()
