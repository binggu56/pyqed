#!/usr/bin/env python3
"""Tiny real-CASCI/SA-CASSCF rovibronic H2O LDR example.

This script is meant to be read first and optimized later.  It scans CASCI or
state-averaged CASSCF energies and nearest-neighbor electronic overlap links on
a small ``(r1, r2, theta)`` grid, then propagates a fixed-``J, Jz`` rovibronic
wavepacket with linked-overlap LDR.

Example:
    MPLCONFIGDIR=/private/tmp /opt/anaconda3/bin/python examples/namd/h2o_casci_rovibronic.py

Larger active-space sketch:
    ... h2o_casci_rovibronic.py --basis 631g* --ncas 8 --nelecas 6
"""

from __future__ import annotations

import argparse
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pyqed.namd.triatomic import Triatom
from pyqed.qchem import CASSCF, Molecule
from pyqed.units import au2fs

FS_TO_AU = 1.0 / au2fs


def optional_int(value):
    text = str(value).strip().lower()
    return None if text in {"none", "full"} else int(value)


@contextmanager
def cd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def h2o_frame(r=1.80965, theta_deg=104.52):
    theta = np.deg2rad(theta_deg)
    return [
        ["H", (r, 0.0, 0.0)],
        ["O", (0.0, 0.0, 0.0)],
        ["H", (r * np.cos(theta), r * np.sin(theta), 0.0)],
    ]


def make_h2o_ldr(args):
    mol = Triatom(
        h2o_frame(),
        basis=args.basis,
        nstates=args.nstates,
        charge=0,
        spin=0,
        unit="bohr",
        J=args.J,
        Jz=args.Jz,
    )
    mol.set_dvr(
        domains=[
            [args.r_min, args.r_max],
            [args.r_min, args.r_max],
            [np.deg2rad(args.theta_min), np.deg2rad(args.theta_max)],
        ],
        npts=[args.n_r, args.n_r, args.n_theta],
        dvr_type=["sine", "sine", "legendre"],
    )
    mol.overlap_path_average = True
    return mol


class SACASSCFScanner:
    """Callable electronic driver used by ``Triatom.scan_pes(driver=...)``."""

    def __init__(self, args):
        self.symbols = ("H", "O", "H")
        self.basis = args.basis
        self.ncas = args.ncas
        self.nelecas = args.nelecas
        self.nstates = args.nstates
        self.max_cycle = args.casscf_max_cycle
        self.verbose = args.verbose
        self.optimizer = args.casscf_optimizer
        self.driver = args.qchem_driver
        self.eri = args.eri
        self.mol = Molecule(atom=h2o_frame(), basis=args.basis, charge=0, spin=0, unit="bohr")

    def as_scanner(self, nstates=None):
        if nstates is not None and int(nstates) != self.nstates:
            raise ValueError("SACASSCFScanner was constructed for a different nstates.")
        return self

    def __call__(self, xyz):
        atom = [[symbol, tuple(coord)] for symbol, coord in zip(self.symbols, xyz)]
        mol = Molecule(atom=atom, basis=self.basis, charge=0, spin=0, unit="bohr")
        mol.build(driver=self.driver, eri=self.eri)
        mf = mol.RHF(verbose=0).run(max_cycle=100)
        weights = np.ones(self.nstates) / self.nstates
        return (
            CASSCF(
                mf,
                ncas=self.ncas,
                nelecas=self.nelecas,
                max_cycle=self.max_cycle,
                verbose=self.verbose,
                optimizer=self.optimizer,
            )
            .state_average(weights)
            .run(nstates=self.nstates)
        )

    @staticmethod
    def overlap(left, right):
        return left.overlap(right)


def load_links(mol: Triatom, outdir: Path):
    apes = outdir / "apes.npz"
    links = outdir / "overlap_links.npz"
    if not (apes.exists() and links.exists()):
        return False
    mol.apes = np.load(apes, allow_pickle=True)["data"]
    raw = np.load(links, allow_pickle=True)
    mol.overlap_links = mol._unpack_overlap_links(raw["axes"], raw["indices"], raw["data"])
    mol.overlap_matrix = None
    return True


def scan_electronic(mol: Triatom, args):
    if args.reuse_cache and load_links(mol, args.outdir):
        print(f"[cache] loaded {args.outdir}/apes.npz and overlap_links.npz")
        return

    with cd(args.outdir):
        kwargs = dict(
            basis=args.basis,
            nstates=args.nstates,
            scan_roots=args.nstates,
            ncas=args.ncas,
            nelecas=args.nelecas,
            overlap_method="link-only",
            unitarize_overlap_links=args.unitarize_links,
            n_workers=args.n_workers,
            worker_threads=args.worker_threads,
        )
        if args.method == "casscf":
            kwargs["driver"] = SACASSCFScanner(args)
        else:
            kwargs.update(
                electronic_method="casci",
                scf_tol=1.0e-9,
                conv_tol_dm=1.0e-6,
                max_cycle=120,
                init_guess="hcore",
            )
        mol.scan_pes(**kwargs)


def nearest_grid_index(mol: Triatom, center):
    return tuple(int(np.argmin(np.abs(np.asarray(x) - q))) for x, q in zip(mol.x, center))


def initial_packet(
    mol: Triatom,
    state=1,
    center=(1.80965, 1.80965, np.deg2rad(104.52)),
    sigma_r=0.12,
    sigma_theta_deg=6.0,
):
    """Gaussian packet on one electronic state and the central K value."""
    center = np.asarray(center, dtype=float)
    sigmas = np.array([sigma_r, sigma_r, np.deg2rad(sigma_theta_deg)])
    ref = nearest_grid_index(mol, center)
    rot = mol.nrot // 2

    psi = np.zeros((*mol.nx, mol.nrot, mol.nstates), dtype=complex)
    for idx in mol._grid_indices():
        q = np.array([mol.x[a][idx[a]] for a in range(mol.ndim)])
        amp = np.exp(-0.5 * np.sum(((q - center) / sigmas) ** 2))

        # Phase-align the launched adiabatic state with linked overlap products.
        phase = 1.0 + 0.0j
        if mol.overlap_links is not None:
            block = mol._linked_overlap_between(idx, ref, mol.overlap_links, mol.nstates)
            if abs(block[state, state]) > 1.0e-14:
                phase = block[state, state] / abs(block[state, state])
        psi[idx + (rot, state)] = amp * phase

    psi = mol.to_quadrature_normalized(psi)
    return psi / mol.norm(psi)


def populations(psilist):
    electronic = np.array([np.sum(abs(psi) ** 2, axis=(0, 1, 2, 3)) for psi in psilist])
    rotational = np.array([np.sum(abs(psi) ** 2, axis=(0, 1, 2, 4)) for psi in psilist])
    return electronic, rotational


def plot_populations(times_fs, electronic, rotational, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.5), constrained_layout=True)
    for s in range(electronic.shape[1]):
        axes[0].plot(times_fs, electronic[:, s], marker="o", label=f"S{s}")
    axes[0].set(xlabel="time / fs", ylabel="population", title="electronic")
    axes[0].legend()

    for r in range(rotational.shape[1]):
        axes[1].plot(times_fs, rotational[:, r], marker="o", label=f"rot {r}")
    axes[1].set(xlabel="time / fs", ylabel="population", title="rotational sector")
    axes[1].legend(fontsize=8)
    fig.savefig(path, dpi=220)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=["casci", "casscf"], default="casci")
    p.add_argument("--basis", default="sto-3g")
    p.add_argument("--ncas", type=int, default=4)
    p.add_argument("--nelecas", type=int, default=4)
    p.add_argument("--nstates", type=int, default=2)
    p.add_argument("--J", type=int, default=1)
    p.add_argument("--Jz", type=optional_int, default=0, help="'none' gives the full M space")
    p.add_argument("--n-r", type=int, default=2)
    p.add_argument("--n-theta", type=int, default=2)
    p.add_argument("--r-min", type=float, default=1.70)
    p.add_argument("--r-max", type=float, default=1.95)
    p.add_argument("--theta-min", type=float, default=99.0)
    p.add_argument("--theta-max", type=float, default=110.0)
    p.add_argument("--center-r1", type=float, default=1.80965)
    p.add_argument("--center-r2", type=float, default=1.80965)
    p.add_argument("--center-theta", type=float, default=104.52)
    p.add_argument("--sigma-r", type=float, default=0.12)
    p.add_argument("--sigma-theta", type=float, default=6.0)
    p.add_argument("--dt-fs", type=float, default=0.1)
    p.add_argument("--tmax-fs", type=float, default=0.5)
    p.add_argument("--initial-state", type=int, default=1)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--worker-threads", type=int, default=1)
    p.add_argument("--casscf-max-cycle", type=int, default=20)
    p.add_argument("--casscf-optimizer", default="AH")
    p.add_argument("--qchem-driver", default="builtin")
    p.add_argument("--eri", default="dense")
    p.add_argument("--verbose", type=int, default=0)
    p.add_argument("--reuse-cache", action="store_true")
    p.add_argument("--unitarize-links", action="store_true")
    p.add_argument(
        "--kinetic-propagator",
        choices=["expm_multiply", "chebyshev"],
        default="expm_multiply",
    )
    p.add_argument(
        "--rovibronic-kinetic",
        choices=["none", "python", "compiled", "sparse"],
        default="none",
        help="Factorized J>0 rovibronic kinetic action; use 'sparse' for larger grids.",
    )
    p.add_argument("--outdir", type=Path, default=Path("/private/tmp/h2o_casci_rovibronic"))
    args = p.parse_args()
    rovibronic_kinetic = None if args.rovibronic_kinetic == "none" else args.rovibronic_kinetic

    args.outdir.mkdir(parents=True, exist_ok=True)
    mol = make_h2o_ldr(args)
    scan_electronic(mol, args)

    center = (args.center_r1, args.center_r2, np.deg2rad(args.center_theta))
    psi0 = initial_packet(
        mol,
        state=args.initial_state,
        center=center,
        sigma_r=args.sigma_r,
        sigma_theta_deg=args.sigma_theta,
    )
    nt = int(round(args.tmax_fs / args.dt_fs))
    result = mol.run(
        psi0,
        dt=args.dt_fs * FS_TO_AU,
        nt=nt,
        nout=1,
        kinetic_propagator=args.kinetic_propagator,
        kinetic_action="matrix-free",
        rovibronic_kinetic=rovibronic_kinetic,
    )

    times_fs = result["times"] * au2fs
    electronic, rotational = populations(result["psilist"])
    np.savez_compressed(
        args.outdir / "h2o_casci_rovibronic.npz",
        times_fs=times_fs,
        r1=mol.x[0],
        r2=mol.x[1],
        theta=mol.x[2],
        apes=mol.apes,
        psi_t=np.asarray(result["psilist"]),
        electronic_populations=electronic,
        rotational_populations=rotational,
        launch_center=np.asarray(center),
        method=args.method,
        kinetic_propagator=args.kinetic_propagator,
        rovibronic_kinetic=args.rovibronic_kinetic,
    )
    plot_populations(times_fs, electronic, rotational, args.outdir / "h2o_casci_populations.png")

    dim = int(np.prod(mol.nx) * mol.nrot * mol.nstates)
    print(f"grid={mol.nx}, J={mol.J}, Jz={mol.Jz}, nrot={mol.nrot}, nstates={mol.nstates}")
    label = "SA-CASSCF" if args.method == "casscf" else "CASCI"
    print(f"dimension={dim}, {label}=({args.nelecas}e,{args.ncas}o), basis={args.basis}")
    print(
        f"kinetic_propagator={args.kinetic_propagator}, "
        f"rovibronic_kinetic={args.rovibronic_kinetic}"
    )
    print(f"norm: {mol.norm(psi0):.12f} -> {mol.norm(result['psilist'][-1]):.12f}")
    print(f"outdir={args.outdir}")


if __name__ == "__main__":
    main()
