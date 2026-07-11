"""Critical Ising scaling dimensions from a minimal NARG prototype."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import (
    TransverseFieldIsingNARG,
    finite_size_scaling_dimensions,
    narg_fixed_layer_scaling_dimensions,
)


def main():
    benchmark = finite_size_scaling_dimensions(12, nlevels=8)
    narg_result = TransverseFieldIsingNARG(
        12,
        bond_dim=8,
        nstart=4,
    ).run(nroots=6)
    narg_scaling = narg_fixed_layer_scaling_dimensions(narg_result.steps[-1].tensor)
    odd_scaling = narg_fixed_layer_scaling_dimensions(
        narg_result.steps[-1].tensor,
        symmetry_operator=narg_result.symmetry_operator,
        input_symmetry_operator=narg_result.steps[-1].input_symmetry_operator,
        sector="odd",
    )
    even_scaling = narg_fixed_layer_scaling_dimensions(
        narg_result.steps[-1].tensor,
        symmetry_operator=narg_result.symmetry_operator,
        input_symmetry_operator=narg_result.steps[-1].input_symmetry_operator,
        sector="even",
    )

    print("Critical transverse-field Ising chain")
    print("Finite-size benchmark dimensions, L=12:")
    for index, value in enumerate(benchmark.dimensions[:8]):
        print(f"  level {index}: {value:.6f}")

    print()
    print("Sequential NARG energies:")
    for index, value in enumerate(narg_result.energies[:6]):
        print(f"  level {index}: {value:.6f}")

    print()
    print("Raw dimensions from the final real NARG growth tensor:")
    for index, (eigenvalue, value) in enumerate(
        zip(narg_scaling.eigenvalues[:8], narg_scaling.dimensions[:8])
    ):
        print(f"  operator {index}: mu={eigenvalue:.6g}, Delta={value:.6f}")

    print()
    print("Raw candidates closest to sigma:")
    closest_sigma = sorted(
        zip(narg_scaling.eigenvalues, narg_scaling.dimensions),
        key=lambda item: abs(item[1] - 0.125),
    )
    for index, (eigenvalue, value) in enumerate(closest_sigma[:4]):
        print(f"  sigma {index}: mu={eigenvalue:.6g}, Delta={value:.6f}")

    print()
    print("Strict input/output Z2-odd sector:")
    for index, (eigenvalue, value) in enumerate(
        zip(odd_scaling.eigenvalues[:6], odd_scaling.dimensions[:6])
    ):
        print(f"  odd {index}: mu={eigenvalue:.6g}, Delta={value:.6f}")

    print()
    print("Strict input/output Z2-even sector:")
    non_identity = [
        (eigenvalue, value)
        for eigenvalue, value in zip(even_scaling.eigenvalues, even_scaling.dimensions)
        if value > 1e-6
    ]
    for index, (eigenvalue, value) in enumerate(non_identity[:6]):
        print(f"  even {index}: mu={eigenvalue:.6g}, Delta={value:.6f}")

    print()
    print("Z2-even candidates closest to epsilon:")
    closest_epsilon = sorted(
        non_identity,
        key=lambda item: abs(item[1] - 1.0),
    )
    for index, (eigenvalue, value) in enumerate(closest_epsilon[:4]):
        print(f"  epsilon {index}: mu={eigenvalue:.6g}, Delta={value:.6f}")

    print()
    print("Known Ising CFT targets: Delta_sigma=0.125, Delta_epsilon=1.0")


if __name__ == "__main__":
    main()
