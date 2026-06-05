#!/usr/bin/env python3
"""Compare spinful half-filled Holstein NARG with real-space TenPy DMRG."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.narg import SpinfulHolsteinAdiabaticElectronicNARG


try:
    from tenpy.algorithms import dmrg
    from tenpy.models.lattice import Lattice
    from tenpy.models.model import CouplingMPOModel
    from tenpy.networks.mps import MPS
    from tenpy.networks.site import BosonSite, SpinHalfFermionSite, set_common_charges
except ImportError as exc:  # pragma: no cover - optional dependency
    dmrg = None
    CouplingMPOModel = object
    TENPY_IMPORT_ERROR = exc
else:
    TENPY_IMPORT_ERROR = None


class SpinfulHolsteinFockModel(CouplingMPOModel):
    """Real-space spinful Holstein chain with sites ordered as e_i, ph_i."""

    def init_sites(self, model_params):
        fermion = SpinHalfFermionSite(
            cons_N=model_params.get("cons_N", "N"),
            cons_Sz=model_params.get("cons_Sz", "Sz"),
        )
        phonon = BosonSite(Nmax=model_params.get("Nmax", 8), conserve=None)
        phonon.add_op("X", phonon.B + phonon.Bd)
        set_common_charges([fermion, phonon], new_charges="same")
        return [fermion, phonon]

    def init_lattice(self, model_params):
        nsites = model_params.get("L", 4)
        sites = self.init_sites(model_params)
        return Lattice(
            [nsites],
            sites,
            positions=np.array([[0.0], [0.5]]),
            bc="open",
            bc_MPS="finite",
        )

    def init_terms(self, model_params):
        hopping = model_params.get("t", 1.0)
        omega = model_params.get("omega", 1.0)
        coupling = model_params.get("g", 1.0)
        hubbard_u = model_params.get("U", 0.0)

        self.add_onsite(hubbard_u, 0, "NuNd")
        self.add_onsite(omega, 1, "N")
        self.add_coupling(coupling, 0, "Ntot", 1, "X", np.array([0]))
        self.add_coupling(-hopping, 0, "Cdu", 0, "Cu", np.array([1]), plus_hc=True)
        self.add_coupling(-hopping, 0, "Cdd", 0, "Cd", np.array([1]), plus_hc=True)


def half_filled_product_state(nsites: int) -> list[str]:
    states = []
    for site in range(int(nsites)):
        states.append("up" if site % 2 == 0 else "down")
        states.append("vac")
    return states


def run_real_space_dmrg(args):
    if dmrg is None:
        raise RuntimeError(f"TenPy is not available: {TENPY_IMPORT_ERROR}")

    model = SpinfulHolsteinFockModel(
        dict(
            L=args.nsites,
            t=args.hopping,
            omega=args.omega,
            g=args.coupling,
            U=args.hubbard_u,
            Nmax=args.phonon_nmax,
        )
    )
    rows = []
    for bond_dim in args.dmrg_bond_dims:
        psi = MPS.from_product_state(
            model.lat.mps_sites(),
            half_filled_product_state(args.nsites),
            bc=model.lat.bc_MPS,
            unit_cell_width=2,
        )
        options = {
            "active_sites": 2,
            "mixer": True,
            "max_E_err": args.dmrg_energy_tol,
            "max_sweeps": args.dmrg_sweeps,
            "trunc_params": {
                "chi_max": int(bond_dim),
                "svd_min": 1e-10,
            },
        }
        start = perf_counter()
        info = dmrg.run(psi, model, options)
        seconds = perf_counter() - start
        rows.append((int(bond_dim), float(info["E"]), seconds, info))
    return rows


def response_transform_for_model(model, args):
    if args.narg_order == "local":
        return None, None
    report = model.density_response_mode_transform(
        nlow=args.nlow_electronic,
        center=args.narg_order == "response-centered",
    )
    return report.transform, report.strengths


def run_narg(args):
    seed = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=args.nsites,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        hubbard_u=args.hubbard_u,
        ngrid=args.ngrid,
        xmax=args.qmax,
    )
    transform, strengths = response_transform_for_model(seed, args)
    model = SpinfulHolsteinAdiabaticElectronicNARG(
        nsites=args.nsites,
        t=args.hopping,
        omega=args.omega,
        g=args.coupling,
        hubbard_u=args.hubbard_u,
        ngrid=args.ngrid,
        xmax=args.qmax,
        mode_transform=transform,
        mode_strengths=strengths,
    )

    rows = []
    for nstates in args.narg_states_per_point:
        start = perf_counter()
        result = model.run_sequential(
            nstates_per_point=int(nstates),
            bond_dim=args.narg_bond_dim,
            initial_electronic_states=args.initial_electronic_states,
            nroots=1,
        )
        seconds = perf_counter() - start
        step_dims = ",".join(
            f"{step.grid_dim}x{step.conditional_dim}->{step.kept}"
            for step in result.steps
        )
        rows.append((int(nstates), float(result.energies[0]), seconds, step_dims))
    return model, rows


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=8)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("--omega", type=float, default=1.0)
    parser.add_argument("-g", "--coupling", type=float, default=1.0)
    parser.add_argument("-U", "--hubbard-u", type=float, default=0.0)
    parser.add_argument("--ngrid", type=int, default=9)
    parser.add_argument("--qmax", type=float, default=6.0)
    parser.add_argument("--phonon-nmax", type=int, default=8)
    parser.add_argument(
        "--narg-order",
        choices=("response", "response-centered", "local"),
        default="response",
    )
    parser.add_argument("--nlow-electronic", type=int, default=64)
    parser.add_argument("--narg-bond-dim", type=int, default=64)
    parser.add_argument("--initial-electronic-states", type=int, default=64)
    parser.add_argument("--narg-states-per-point", type=int, nargs="+", default=[8, 16, 32])
    parser.add_argument("--dmrg-bond-dims", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--dmrg-sweeps", type=int, default=8)
    parser.add_argument("--dmrg-energy-tol", type=float, default=1e-8)
    parser.add_argument("--skip-narg", action="store_true")
    parser.add_argument("--skip-dmrg", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    print(
        f"Spinful half-filled Holstein: L={args.nsites}, t={args.hopping:g}, "
        f"omega={args.omega:g}, g={args.coupling:g}, U={args.hubbard_u:g}"
    )
    print(
        f"NARG basis=sine-DVR ngrid={args.ngrid} q_range=[-{args.qmax:g},{args.qmax:g}], "
        f"DMRG basis=Fock Nmax={args.phonon_nmax}"
    )

    if not args.skip_narg:
        model, narg_rows = run_narg(args)
        print(
            f"\nNARG sequential all {len(model._active_modes_tuple())} modes, "
            f"order={args.narg_order}, D={args.narg_bond_dim}"
        )
        print("nstates     energy          seconds   step_dims")
        for nstates, energy, seconds, step_dims in narg_rows:
            print(f"{nstates:7d} {energy: .12f} {seconds:9.3f}   {step_dims}")

    if not args.skip_dmrg:
        print(f"\nTenPy real-space DMRG, alternating e_i/ph_i sites, sweeps={args.dmrg_sweeps}")
        print("chi         energy          seconds")
        for bond_dim, energy, seconds, _info in run_real_space_dmrg(args):
            print(f"{bond_dim:7d} {energy: .12f} {seconds:9.3f}")


if __name__ == "__main__":
    main()
