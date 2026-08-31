#!/usr/bin/env python3
"""One-state H3+ APH TTLDR benchmark with explicit CASCI overlaps."""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import expm_multiply

from pyqed.dvr import ExponentialDVR, LegendreDVR, SineDVR
from pyqed.ldr import keo
from pyqed.ldr.oracle import Frames
from pyqed.ldr.overlap import dense as dense_overlap
from pyqed.ldr.overlap import nearest
from pyqed.mps import FunctionalTT
from pyqed.namd.ttldr import TTLDR
from pyqed.qchem import CASCI, Molecule
from pyqed.qchem.mcscf.casci import overlap as casci_overlap
from pyqed.units import au2ev, au2fs, proton_mass


@dataclass(frozen=True)
class CASCIFrameBuilder:
    axes: tuple
    basis: str

    def __call__(self, index):
        for name in (
            "OPENBLAS_NUM_THREADS",
            "OMP_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[name] = "1"
        aph = keo.APH(("H",) * 3, (proton_mass,) * 3)
        coordinates = tuple(self.axes[axis][index[axis]] for axis in range(3))
        molecule = Molecule(
            atom=aph.geometry(coordinates),
            basis=self.basis,
            charge=1,
            spin=0,
            unit="bohr",
        )
        molecule.build()
        mean_field = molecule.RHF(verbose=0).run(max_cycle=80)
        if not mean_field.converged:
            raise RuntimeError(f"RHF failed at APH index {index}")
        return CASCI(mean_field, ncas=3, nelecas=2, verbose=0).run(nstates=1)


class APHGrid:
    def __init__(self, potential, links):
        self.apes = np.asarray(potential)[..., None]
        self.nx = self.apes.shape[:-1]
        self.nstates = 1
        self.overlap_links = links
        self.overlap_matrix = None
        self.overlap_path_average = False


def electronic_energy(frame):
    return float(np.asarray(frame.e_tot).reshape(-1)[0])


def initial_packet(aph, dvrs, potential, *, collision_ev, rho_center):
    radial = np.asarray(dvrs[0].x)
    rho_index = int(np.argmin(np.abs(radial - float(rho_center))))
    angular = aph.angular_hamiltonian(radial[rho_index], dvrs[1:], potential[rho_index])
    _energies, vectors = np.linalg.eigh(angular)
    channel = vectors[:, 0]
    momentum = np.sqrt(2.0 * aph.mu * float(collision_ev) / au2ev)
    radial_state = np.exp(
        -0.5 * ((radial - radial[rho_index]) / 0.55) ** 2
        - 1j * momentum * (radial - radial[rho_index])
    )
    radial_state /= np.linalg.norm(radial_state)
    state = np.einsum("r,a->ra", radial_state, channel).reshape(
        *(len(dvr.x) for dvr in dvrs), 1
    )
    return state / np.linalg.norm(state)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/private/tmp/h3plus_functional_tt/h3plus_hybrid_functional_tt.npz"
        ),
    )
    parser.add_argument(
        "--outdir", type=Path, default=Path("/private/tmp/h3plus_aph_ttldr")
    )
    parser.add_argument("--n-rho", type=int, default=4)
    parser.add_argument("--n-theta", type=int, default=3)
    parser.add_argument("--n-phi", type=int, default=6)
    parser.add_argument("--rho-min", type=float, default=2.5)
    parser.add_argument("--rho-max", type=float, default=7.75)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--collision-ev", type=float, default=0.35)
    parser.add_argument("--rho-center", type=float, default=6.2)
    parser.add_argument("--dt-fs", type=float, default=0.02)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--state-rank", type=int, default=24)
    parser.add_argument("--operator-rank", type=int, default=64)
    parser.add_argument("--overlap-method", choices=("cross", "dense"), default="cross")
    parser.add_argument("--overlap-rank", type=int, default=12)
    parser.add_argument("--overlap-sweeps", type=int, default=8)
    args = parser.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    aph = keo.APH(("H",) * 3, (proton_mass,) * 3)
    dvrs = (
        SineDVR(args.rho_min, args.rho_max, args.n_rho, mass=aph.mu),
        LegendreDVR(0.0, 0.5 * np.pi, args.n_theta),
        ExponentialDVR(npts=args.n_phi, L=2.0 * np.pi, x0=np.pi),
    )
    axes = tuple(np.asarray(dvr.x) for dvr in dvrs)
    shape = tuple(len(axis) for axis in axes)

    builder = CASCIFrameBuilder(axes, args.basis)
    with Frames(
        shape,
        builder,
        cache_dir=args.outdir / "frames",
        workers=args.workers,
    ) as frames:
        records = frames.get_many(np.ndindex(shape))
        frame_map = dict(zip(np.ndindex(shape), records))
        qc_energies = np.asarray(
            [electronic_energy(frame_map[index]) for index in np.ndindex(shape)]
        ).reshape(shape)
        links = nearest(
            shape,
            lambda left, right: casci_overlap(frame_map[left], frame_map[right]),
            unitarize=False,
        )
        frame_stats = frames.stats

    model = FunctionalTT.load(args.model)
    transformed_phi = np.cos(6 * axes[2])
    pes_mpo = model.mpo((axes[0], axes[1], transformed_phi))
    mesh = np.meshgrid(axes[0], axes[1], transformed_phi, indexing="ij")
    coordinates = np.stack([item.reshape(-1) for item in mesh], axis=1)
    potential = model.predict(coordinates).reshape(shape)

    nuclear_keo = aph.mpo(
        dvrs,
        field_max_rank=12,
        field_rtol=1.0e-12,
        mpo_max_rank=32,
    )
    grid = APHGrid(potential, links)
    ttldr = TTLDR(
        grid,
        nuclear_keo=nuclear_keo,
        pes_mpo=pes_mpo,
        overlap_method=args.overlap_method,
        overlap_rank=args.overlap_rank,
        overlap_sweeps=args.overlap_sweeps,
        overlap_validation=64,
        operator_rank=args.operator_rank,
        potential_rank=24,
        gauge_sync=True,
    )

    packet = initial_packet(
        aph,
        dvrs,
        potential,
        collision_ev=args.collision_ev,
        rho_center=args.rho_center,
    )
    state = ttldr.state(packet, max_rank=args.state_rank)
    dt = args.dt_fs / au2fs
    ttldr.run(
        state,
        dt=dt,
        steps=args.steps,
        interval=1,
        max_bond=args.state_rank,
        integrator="tdvp2",
        progress=False,
    )

    overlap_matrix = dense_overlap(shape, links, nstates=1).reshape(
        np.prod(shape), np.prod(shape)
    )
    dense_hamiltonian = nuclear_keo.to_dense() * overlap_matrix
    dense_hamiltonian += np.diag(potential.reshape(-1))
    dense_final = expm_multiply(
        -1j * args.steps * dt * dense_hamiltonian, packet.reshape(-1)
    )
    tt_final = ttldr.dense(ttldr.final_state).reshape(-1)
    dense_final /= np.linalg.norm(dense_final)
    tt_final /= np.linalg.norm(tt_final)
    fidelity = float(abs(np.vdot(dense_final, tt_final)) ** 2)
    density_error = float(
        np.max(np.abs(np.abs(tt_final) ** 2 - np.abs(dense_final) ** 2))
    )

    link_values = np.asarray([np.asarray(value).item() for value in links.values()])
    identity_hamiltonian = nuclear_keo.to_dense() + np.diag(potential.reshape(-1))
    overlap_effect = float(
        np.linalg.norm(dense_hamiltonian - identity_hamiltonian)
        / np.linalg.norm(dense_hamiltonian)
    )
    np.savez(
        args.outdir / "h3plus_aph_ttldr.npz",
        rho=axes[0],
        theta=axes[1],
        phi=axes[2],
        potential=potential,
        qc_energies=qc_energies,
        link_magnitudes=np.abs(link_values),
        link_phases=np.angle(link_values),
        dense_final=dense_final,
        tt_final=tt_final,
        fidelity=fidelity,
        density_error=density_error,
        overlap_effect=overlap_effect,
    )
    print(f"grid:                       {shape} ({np.prod(shape)} points)")
    print(f"electronic frames:          {frame_stats}")
    print(f"overlap links:              {len(links)}")
    print(
        "link magnitude range:       "
        f"{np.min(np.abs(link_values)):.8f} .. {np.max(np.abs(link_values)):.8f}"
    )
    print(f"identity-overlap H error:   {overlap_effect:.6e}")
    print(f"overlap TT diagnostics:     {ttldr.overlap_info}")
    print(f"operator ranks:             {ttldr.operator_ranks}")
    print(f"final fidelity:             {fidelity:.12f}")
    print(f"maximum density error:      {density_error:.6e}")
    print(f"output:                     {args.outdir / 'h3plus_aph_ttldr.npz'}")


if __name__ == "__main__":
    main()
