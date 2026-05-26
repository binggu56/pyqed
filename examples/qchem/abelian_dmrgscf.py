#!/usr/bin/env python3
"""
Abelian DMRGSCF constrained orbital optimization example.

This runs the qchem DMRGSCF driver with Abelian charge/Sz symmetry.  The small
default LiF/STO-3G CAS is intended as a quick API and timing check; increase
``--basis``, ``--ncas``, ``--nelecas``, ``--D``, and ``--macro-cycles`` for
production runs.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/xdg")

from pyqed.qchem import Molecule, RHF
from pyqed.qchem.dmrg import DMRGSCF


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--atom", default="Li 0 0 0; F 0 0 1.4")
    parser.add_argument("--unit", default="bohr")
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--driver", default="builtin")
    parser.add_argument("--eri", default="dense")
    parser.add_argument("--ncas", type=int, default=4)
    parser.add_argument("--nelecas", type=int, default=4)
    parser.add_argument("--D", type=int, default=12)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--macro-cycles", type=int, default=3)
    parser.add_argument("--macro-tol", type=float, default=1.0e-6)
    parser.add_argument("--site", choices=("spatial", "spin_orbital"), default="spatial")
    parser.add_argument("--symmetry", choices=("sz", "pg"), default="sz")
    parser.add_argument(
        "--spatial-abelian-mpo",
        choices=("spatial", "grouped", "direct"),
        default="spatial",
    )
    parser.add_argument("--orb-sym", default=None, help="Comma-separated AbelianPG orbital irreps, e.g. 0,1,0,1")
    parser.add_argument("--nstates", type=int, default=1)
    parser.add_argument("--conv-tol", type=float, default=None)
    parser.add_argument("--sweep-tol", type=float, default=None)
    parser.add_argument("--dmrg-conv-tol", type=float, default=1.0e-7)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-5)
    parser.add_argument("--davidson-max-iter", type=int, default=30)
    parser.add_argument("--local-dense-max-dim", default="0")
    parser.add_argument("--noise", type=float, default=1.0e-4)
    parser.add_argument("--noise-decay", type=float, default=0.1)
    parser.add_argument("--noise-cutoff", type=float, default=1.0e-9)
    parser.add_argument("--optimizer", default="RCG", choices=("RCG", "SD", "LBFGS", "NEWTON", "AH"))
    parser.add_argument("--orb-grad-tol", type=float, default=None)
    parser.add_argument("--optimizer-max-steps", type=int, default=200)
    parser.add_argument("--optimizer-max-step-norm", type=float, default=None)
    parser.add_argument("--no-macro-reject", action="store_true")
    parser.add_argument("--macro-rise-tol", type=float, default=1.0e-8)
    parser.add_argument("--macro-reject-max", type=int, default=8)
    parser.add_argument("--macro-trust-radius", type=float, default=0.25)
    parser.add_argument("--macro-trust-min", type=float, default=1.0e-4)
    parser.add_argument("--macro-trust-max", type=float, default=1.0)
    parser.add_argument("--macro-trust-shrink", type=float, default=0.5)
    parser.add_argument("--macro-trust-grow", type=float, default=1.5)
    parser.add_argument("--no-warm-start", action="store_true")
    parser.add_argument("--no-diis", action="store_true")
    parser.add_argument("--spin-penalty", type=float, default=0.0)
    parser.add_argument("--target-s2", type=float, default=0.0)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument("--allow-unconverged", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    t_total = time.perf_counter()

    print("[1/4] Molecule")
    print(f"      atom   : {args.atom}")
    print(f"      unit   : {args.unit}")
    print(f"      basis  : {args.basis}")
    mol = Molecule(atom=args.atom, unit=args.unit, basis=args.basis)
    if args.driver.lower() in {"builtin", "native"}:
        mol.build(driver=args.driver, eri=args.eri)
    else:
        mol.build(driver=args.driver)

    print("[2/4] RHF")
    t0 = time.perf_counter()
    mf = RHF(mol, verbose=args.verbose).run()
    t_hf = time.perf_counter() - t0

    print("[3/4] Abelian DMRGSCF constrained orbital optimization")
    print(f"      CAS        : ({args.nelecas}e, {args.ncas}o)")
    print(f"      symmetry   : {args.symmetry}")
    print(f"      site       : {args.site}")
    print(f"      MPO        : {args.spatial_abelian_mpo}")
    orb_sym = None if args.orb_sym is None else tuple(int(x) for x in args.orb_sym.split(",") if x.strip())
    if orb_sym is not None:
        print(f"      orb sym    : {orb_sym}")
    print(f"      DMRG       : D={args.D}, sweeps={args.sweeps}")
    sweep_tol = args.sweep_tol
    if sweep_tol is None:
        sweep_tol = args.conv_tol if args.conv_tol is not None else args.dmrg_conv_tol
    print(f"      sweep tol  : {sweep_tol:g}")
    print(f"      local tol  : {args.davidson_tol:g}")
    print(f"      local dense: {args.local_dense_max_dim}")
    print(f"      noise      : {args.noise:g} decay={args.noise_decay:g}")
    print(f"      macro tol  : {args.macro_tol:g}")
    print(f"      macro max  : {args.macro_cycles}")
    mc = DMRGSCF(
        mf,
        ncas=args.ncas,
        nelecas=args.nelecas,
        D=args.D,
        max_cycles=args.macro_cycles,
        macro_tol=args.macro_tol,
        init_guess="hf",
        site=args.site,
        spatial_abelian_mpo=args.spatial_abelian_mpo,
        orb_sym=orb_sym,
        symmetry=args.symmetry,
        spin=0,
        verbose=args.verbose,
        dmrg_conv_tol=args.dmrg_conv_tol,
    )
    if args.spin_penalty:
        mc.fix_spin(ss=args.target_s2, shift=args.spin_penalty)

    t0 = time.perf_counter()
    mc.run(
        nstates=args.nstates,
        nsweeps=args.sweeps,
        symmetry=args.symmetry,
        compute_s2=True,
        sweep_tol=sweep_tol,
        davidson_tol=args.davidson_tol,
        davidson_max_iter=args.davidson_max_iter,
        local_dense_max_dim=args.local_dense_max_dim,
        noise=args.noise,
        noise_decay=args.noise_decay,
        noise_cutoff=args.noise_cutoff,
        require_conv=not args.allow_unconverged,
        optimizer=args.optimizer,
        orb_grad_tol=args.orb_grad_tol,
        optimizer_max_steps=args.optimizer_max_steps,
        optimizer_max_step_norm=args.optimizer_max_step_norm,
        reject_macro_energy=not args.no_macro_reject,
        macro_energy_rise_tol=args.macro_rise_tol,
        macro_reject_max=args.macro_reject_max,
        macro_trust_radius=args.macro_trust_radius,
        macro_trust_min=args.macro_trust_min,
        macro_trust_max=args.macro_trust_max,
        macro_trust_shrink=args.macro_trust_shrink,
        macro_trust_grow=args.macro_trust_grow,
        warm_start_dmrg=not args.no_warm_start,
        diis=not args.no_diis,
    )
    t_dmrgscf = time.perf_counter() - t0
    t_total = time.perf_counter() - t_total

    print("\n[4/4] DMRGSCF summary")
    print(f"E(RHF)          = {mf.e_tot:.12f} Ha")
    print(f"E(DMRGSCF)      = {np.asarray(mc.e_tot)} Ha")
    print(f"macro converged = {mc.macro_converged}")
    print(f"solver converged= {mc.solver_converged}")
    print(f"macro cycles    = {mc.macro_iterations}")
    print(f"mo_coeff shape  = {None if mc.mo_coeff is None else mc.mo_coeff.shape}")
    print(f"t(RHF)          = {t_hf:.2f} s")
    print(f"t(DMRGSCF)      = {t_dmrgscf:.2f} s")
    print(f"t(total)        = {t_total:.2f} s")
    if getattr(mc, "e_history", None) is not None:
        print("energy history  =", np.asarray(mc.e_history))
    if getattr(mc, "macro_diagnostics", None):
        print("macro diagnostics:")
        for row in mc.macro_diagnostics:
            print(
                "  "
                f"i={row['macro']:02d} "
                f"E={row['energy']:.12f} "
                f"dE={row['dE']:.3e} "
                f"|g|={row['gn']:.3e} "
                f"tr={row['tr']} "
                f"rej={row['rej']} "
                f"solver={row.get('solver')} "
                f"nsw={row.get('nsw')} "
                f"sweep_dE={row.get('sweep_dE')} "
                f"trunc={row.get('trunc')}"
            )


if __name__ == "__main__":
    main()
