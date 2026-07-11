"""Spin-boson Wilson-chain NARG benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import (
    SpinBosonWilsonNARG,
    log_discretized_spin_boson_wilson_chain,
    spin_boson_wilson_exact,
)


def main():
    chain = log_discretized_spin_boson_wilson_chain(
        4,
        alpha=0.05,
        Lambda=2.0,
        s=1.0,
        omegac=1.0,
        epsilon=0.0,
        delta=0.1,
    )
    nboson = 4
    basis = "dvr"

    exact, _ = spin_boson_wilson_exact(chain, nboson, nroots=4, basis=basis)
    narg_full = SpinBosonWilsonNARG(chain, nboson=nboson, bond_dim=512, basis=basis).run(nroots=4)
    narg_trunc = SpinBosonWilsonNARG(chain, nboson=nboson, bond_dim=12, basis=basis).run(nroots=4)

    print("Spin-boson Wilson chain")
    print(f"local boson basis: {basis}")
    print(f"star frequencies: {chain.star_frequencies}")
    print(f"star couplings:   {chain.star_couplings}")
    print(f"chain onsite:     {chain.onsite}")
    print(f"chain hopping:    {chain.hopping}")
    print(f"spin-chain t0:    {chain.impurity_coupling:.8f}")

    print()
    print("Lowest energies")
    for i, (e_exact, e_full, e_trunc) in enumerate(
        zip(exact, narg_full.energies, narg_trunc.energies)
    ):
        print(
            f"  {i}: exact={e_exact:.10f}  "
            f"NARG-full={e_full:.10f}  NARG-D12={e_trunc:.10f}"
        )

    print()
    print("Truncated NARG growth")
    for step in narg_trunc.steps:
        print(
            f"  site {step.site}: product_dim={step.product_dim}, "
            f"kept={step.kept}, E0={step.lowest_energy:.10f}"
        )


if __name__ == "__main__":
    main()
