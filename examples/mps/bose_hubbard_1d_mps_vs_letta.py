#!/usr/bin/env python3
"""Compare Abelian DMRG/MPS and LETTA on the 1D Bose-Hubbard chain."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pyqed.letta import LETTA, Layout
from pyqed.mps import DMRG
from pyqed.mps.mps import dense_to_symmetric_mpo, symmetric_to_dense
from pyqed.mps.symmetry import AbelianSector, SymmetryManager
from pyqed.narg.bose_hubbard import (
    BoseHubbardObservables,
    bose_hubbard_observables,
    boson_annihilation,
    exact_bose_hubbard,
    fixed_number_basis,
)
from pyqed.narg.spin_boson import boson_dvr_operators


@dataclass
class BoseHubbardComparison:
    onsite_u: float
    letta_basis: str
    ed_energy: float | None
    dmrg_energy: float
    letta_initial: float
    letta_energy: float
    letta_number_weight: float
    ed_observables: BoseHubbardObservables | None
    letta_observables: BoseHubbardObservables
    dmrg_seconds: float
    letta_seconds: float
    dmrg_converged: bool


def bose_hubbard_mpo(
    nsites: int,
    nmax: int,
    *,
    hopping: float = 1.0,
    onsite_u: float = 1.0,
    mu: float = 0.0,
) -> list[np.ndarray]:
    """Return the analytical open-chain Bose-Hubbard MPO with bond dimension 4."""
    nsites = int(nsites)
    nmax = int(nmax)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if nmax < 0:
        raise ValueError("nmax must be non-negative.")
    dim = nmax + 1
    b = boson_annihilation(dim)
    bdag = b.T.conj()
    number = np.diag(np.arange(dim, dtype=float))
    identity = np.eye(dim)
    onsite = 0.5 * float(onsite_u) * number @ (number - identity) - float(mu) * number

    factors = []
    for site in range(nsites):
        core = np.zeros((4, 4, dim, dim), dtype=float)
        core[0, 0] = identity
        core[0, 1] = -float(hopping) * bdag
        core[0, 2] = -float(hopping) * b
        core[0, 3] = onsite
        core[1, 3] = b
        core[2, 3] = bdag
        core[3, 3] = identity
        if site == 0:
            core = core[0:1]
        if site == nsites - 1:
            core = core[:, 3:4]
        factors.append(core)
    return factors


def number_penalty_mpo(nsites: int, nmax: int, nbosons: int) -> list[np.ndarray]:
    r"""Return an exact MPO for $(\sum_i n_i-N)^2$."""
    dim = int(nmax) + 1
    identity = np.eye(dim)
    number = np.diag(np.arange(dim, dtype=float))
    onsite = (
        number @ number
        - 2.0 * int(nbosons) * number
        + (int(nbosons) ** 2 / int(nsites)) * identity
    )
    factors = []
    for site in range(int(nsites)):
        core = np.zeros((3, 3, dim, dim), dtype=float)
        core[0, 0] = identity
        core[0, 1] = number
        core[0, 2] = onsite
        core[1, 1] = identity
        core[1, 2] = 2.0 * number
        core[2, 2] = identity
        if site == 0:
            core = core[0:1]
        if site == int(nsites) - 1:
            core = core[:, 2:3]
        factors.append(core)
    return factors


def local_basis_transform(nmax: int, basis: str):
    """Return local basis labels and the Fock-to-local column transform."""
    basis = str(basis).lower()
    dim = int(nmax) + 1
    if basis == "fock":
        return np.arange(dim, dtype=float), np.eye(dim)
    if basis in {"gh-dvr", "dvr"}:
        *_, grid, transform = boson_dvr_operators(dim)
        return grid, transform
    raise ValueError("basis must be 'fock' or 'gh-dvr'.")


def transform_mpo_local_basis(mpo, transform):
    r"""Apply $W \mapsto U^\dagger W U$ to every MPO physical leg."""
    transform = np.asarray(transform)
    return [
        np.einsum(
            "ai,lrab,bj->lrij",
            transform.conj(),
            np.asarray(tensor),
            transform,
            optimize=True,
        )
        for tensor in mpo
    ]


def transform_product_state_local_basis(vector, nsites, transform, *, direction):
    """Transform a dense product-basis state between Fock and local bases."""
    transform = np.asarray(transform)
    local = transform.conj().T if direction == "fock-to-local" else transform
    if direction not in {"fock-to-local", "local-to-fock"}:
        raise ValueError("direction must be 'fock-to-local' or 'local-to-fock'.")
    tensor = np.asarray(vector).reshape((transform.shape[0],) * int(nsites))
    for axis in range(int(nsites)):
        tensor = np.tensordot(local, tensor, axes=(1, axis))
        tensor = np.moveaxis(tensor, 0, axis)
    return tensor.reshape(-1)


def bose_hubbard_site_qn_maps(nsites: int, nmax: int) -> list[dict[int, AbelianSector]]:
    """Return one U(1) particle-number sector map per boson site."""
    labels = ("charge",)
    local = {occ: AbelianSector(labels, (occ,)) for occ in range(int(nmax) + 1)}
    return [dict(local) for _ in range(int(nsites))]


def target_number_sector(nbosons: int) -> AbelianSector:
    return AbelianSector(("charge",), (int(nbosons),))


def product_mps_fixed_number(nsites: int, nbosons: int, nmax: int) -> list[np.ndarray]:
    """Return a product MPS whose local occupations sum to ``nbosons``."""
    nsites = int(nsites)
    nbosons = int(nbosons)
    nmax = int(nmax)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if nbosons < 0 or nbosons > nsites * nmax:
        raise ValueError("nbosons is incompatible with nsites and nmax.")
    base, remainder = divmod(nbosons, nsites)
    occupations = [base] * nsites
    for site in range(remainder):
        occupations[site] += 1
    if any(occ > nmax for occ in occupations):
        raise ValueError("balanced product state exceeds nmax.")

    factors = []
    dim = nmax + 1
    for occ in occupations:
        vector = np.zeros(dim)
        vector[occ] = 1.0
        factors.append(vector.reshape(1, dim, 1))
    return factors


def _sector_tuple(sector) -> tuple[int, ...]:
    if hasattr(sector, "components"):
        return tuple(int(x) for x in sector.components)
    if isinstance(sector, tuple):
        return tuple(int(x) for x in sector)
    return (int(sector),)


def _expanded_leg_labels(block_tensor, leg: int) -> list[tuple[int, ...]]:
    labels = []
    seen = set()
    for qn in block_tensor.qns[leg]:
        if qn in seen:
            continue
        seen.add(qn)
        dim = 0
        for qkey, block in block_tensor.data.items():
            if qkey[leg] == qn:
                dim = max(dim, int(block.shape[leg]))
        if dim == 0:
            dim = int(sum(1 for item in block_tensor.qns[leg] if item == qn))
        labels.extend([_sector_tuple(qn)] * dim)
    return labels


def letta_layout_from_symmetric_mps(
    mps,
    site_qn_maps: list[dict[int, AbelianSector]],
    target: tuple[int, ...],
) -> Layout:
    """Build a LETTA U(1) layout aligned to the actual Abelian DMRG MPS sectors."""
    factors = mps.to_order(["lv", "rv", "p"]).tensors
    nsites = len(factors)
    if nsites < 2:
        raise ValueError("LETTA needs at least two sites.")
    local_qns = [[_sector_tuple(site_map[state]) for state in sorted(site_map)] for site_map in site_qn_maps]
    bond_qns = [_expanded_leg_labels(factors[0], 0)]
    for factor in factors[: nsites - 2]:
        bond_qns.append(_expanded_leg_labels(factor, 1))
    return Layout(local_qns, bond_qns, tuple(int(x) for x in target))


def expand_letta_seed(
    letta: LETTA,
    layout: Layout,
    *,
    noise: float = 0.0,
    seed: int | None = None,
) -> LETTA:
    """Pad an MPS-embedded LETTA seed to the Abelian layout dimensions."""
    rng = np.random.default_rng(seed)
    masks = layout.local_masks()
    expanded = []
    for tensor, mask in zip(letta.tensors, masks):
        new_tensor = np.zeros(mask.shape, dtype=tensor.dtype)
        slices = tuple(slice(0, min(old, new)) for old, new in zip(tensor.shape, mask.shape))
        new_tensor[slices] = tensor[slices]
        if noise > 0.0:
            zero_allowed = mask & (np.abs(new_tensor) < 1.0e-14)
            new_tensor = new_tensor + float(noise) * rng.normal(size=mask.shape) * zero_allowed
        expanded.append(new_tensor)
    return LETTA(
        letta.hamiltonian,
        letta.dims,
        bond_dim=max(max(tensor.shape[0], tensor.shape[3]) for tensor in expanded),
        tensors=expanded,
        local_masks=masks,
        abelian_layout=layout,
    )


def project_product_state_to_fixed_basis(
    vector: np.ndarray,
    basis: list[tuple[int, ...]],
    *,
    nmax: int,
) -> np.ndarray:
    """Project a full product-basis vector onto a fixed-number occupation basis."""
    dim = int(nmax) + 1
    tensor = np.asarray(vector).reshape((dim,) * len(basis[0]))
    return np.asarray([tensor[state] for state in basis])


def fixed_basis_observables_from_product_state(
    vector: np.ndarray,
    nsites: int,
    nbosons: int,
    nmax: int,
) -> BoseHubbardObservables:
    basis = fixed_number_basis(nsites, nbosons, nmax)
    projected = project_product_state_to_fixed_basis(vector, basis, nmax=nmax)
    return bose_hubbard_observables(projected, basis)


def fixed_number_weight_from_product_state(vector, nsites, nbosons, nmax):
    """Return normalized probability in the selected total-number sector."""
    basis = fixed_number_basis(nsites, nbosons, nmax)
    projected = project_product_state_to_fixed_basis(vector, basis, nmax=nmax)
    norm = float(np.real(np.vdot(vector, vector)))
    if norm <= 0.0:
        raise ValueError("state vector has zero norm.")
    return float(np.real(np.vdot(projected, projected)) / norm)


def run_point(
    *,
    nsites: int,
    nbosons: int,
    nmax: int,
    hopping: float,
    onsite_u: float,
    mu: float,
    bond_dim: int,
    sweeps: int,
    letta_sweeps: int,
    letta_expand_noise: float,
    letta_seed: int | None,
    skip_ed: bool,
    verbose: int,
    davidson_tol: float,
    sweep_tol: float,
    letta_basis: str = "fock",
) -> BoseHubbardComparison:
    dense_mpo = bose_hubbard_mpo(nsites, nmax, hopping=hopping, onsite_u=onsite_u, mu=mu)
    site_qn_maps = bose_hubbard_site_qn_maps(nsites, nmax)
    target = target_number_sector(nbosons)
    sym_mgr = SymmetryManager(["charge"])
    sym_mpo = dense_to_symmetric_mpo(dense_mpo, site_qn_maps)
    init_mps = product_mps_fixed_number(nsites, nbosons, nmax)

    ed_energy = None
    ed_observables = None
    if not skip_ed:
        values, vectors, basis = exact_bose_hubbard(
            nsites,
            nbosons,
            t=hopping,
            U=onsite_u,
            nmax=nmax,
            mu=mu,
            nroots=1,
        )
        ed_energy = float(values[0])
        ed_observables = bose_hubbard_observables(vectors[:, 0], basis)

    start = perf_counter()
    dmrg = DMRG(
        sym_mpo,
        bond_dim,
        init_guess=init_mps,
        nsweeps=sweeps,
        opt="2site",
        symmetry=True,
        target_qn=target,
        sym_mgr=sym_mgr,
        not_conv_err=False,
        site_qn_maps=site_qn_maps,
        verbose=verbose,
        davidson_tol=davidson_tol,
        sweep_tol=sweep_tol,
    ).run()
    dmrg_seconds = perf_counter() - start
    dmrg_energy = float(np.real(dmrg.energy))

    start = perf_counter()
    dense_mps = symmetric_to_dense(
        dmrg.state, site_qn_maps=site_qn_maps
    ).to_order(["lv", "p", "rv"])
    letta = LETTA.from_mps(dense_mps, dims=(nmax + 1,) * nsites)
    layout = letta_layout_from_symmetric_mps(
        dmrg.state, site_qn_maps, (nbosons,)
    )
    letta = expand_letta_seed(
        letta,
        layout,
        noise=letta_expand_noise,
        seed=letta_seed,
    )
    letta_initial = letta.expectation(dense_mpo)
    if letta_sweeps > 0:
        letta.run(
            dense_mpo,
            nsweeps=letta_sweeps,
            start_direction="rl",
            local_solver="auto",
            matrix_free_tol=davidson_tol,
            verbose=verbose,
        )
    letta_energy = letta.expectation(dense_mpo)
    letta_seconds = perf_counter() - start
    state_vector = letta.state_vector()
    letta_observables = fixed_basis_observables_from_product_state(
        state_vector,
        nsites,
        nbosons,
        nmax,
    )

    return BoseHubbardComparison(
        onsite_u=float(onsite_u),
        letta_basis=str(letta_basis),
        ed_energy=ed_energy,
        dmrg_energy=dmrg_energy,
        letta_initial=float(letta_initial),
        letta_energy=float(letta_energy),
        letta_number_weight=fixed_number_weight_from_product_state(
            state_vector, nsites, nbosons, nmax
        ),
        ed_observables=ed_observables,
        letta_observables=letta_observables,
        dmrg_seconds=float(dmrg_seconds),
        letta_seconds=float(letta_seconds),
        dmrg_converged=bool(dmrg.converged),
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-L", "--nsites", type=int, default=6)
    parser.add_argument("-N", "--nbosons", type=int, default=None)
    parser.add_argument("--nmax", type=int, default=4)
    parser.add_argument("-t", "--hopping", type=float, default=1.0)
    parser.add_argument("-U", "--onsite-u", type=float, default=1.0)
    parser.add_argument("--sweep-u", type=float, nargs="+", default=[0.2, 1.0, 5.0])
    parser.add_argument("--mu", type=float, default=0.0)
    parser.add_argument("-D", "--bond-dim", type=int, default=16)
    parser.add_argument("--sweeps", type=int, default=6)
    parser.add_argument("--letta-sweeps", type=int, default=3)
    parser.add_argument("--letta-expand-noise", type=float, default=1.0e-6)
    parser.add_argument("--letta-seed", type=int, default=1)
    parser.add_argument("--skip-ed", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--davidson-tol", type=float, default=1.0e-9)
    parser.add_argument("--sweep-tol", type=float, default=1.0e-9)
    return parser.parse_args()


def _fmt(value, width=15, precision=9):
    if value is None:
        return " " * (width - 3) + "nan"
    return f"{value:{width}.{precision}f}"


def main():
    args = parse_args()
    nbosons = args.nsites if args.nbosons is None else int(args.nbosons)
    print(
        f"# 1D Bose-Hubbard MPS vs LETTA: L={args.nsites}, N={nbosons}, "
        f"nmax={args.nmax}, D={args.bond_dim}, t={args.hopping:g}, mu={args.mu:g}"
    )
    print(
        " U/t        E_ED          E_DMRG        E_LETTA      "
        "DMRG-ED    LETTA-ED   LETTA-DMRG  f0_ED  f0_LETTA  var_ED  var_LETTA  edge_ED  edge_LETTA  t_DMRG  t_LETTA"
    )
    for onsite_u in args.sweep_u:
        result = run_point(
            nsites=args.nsites,
            nbosons=nbosons,
            nmax=args.nmax,
            hopping=args.hopping,
            onsite_u=onsite_u,
            mu=args.mu,
            bond_dim=args.bond_dim,
            sweeps=args.sweeps,
            letta_sweeps=args.letta_sweeps,
            letta_expand_noise=args.letta_expand_noise,
            letta_seed=args.letta_seed,
            skip_ed=args.skip_ed,
            verbose=args.verbose,
            davidson_tol=args.davidson_tol,
            sweep_tol=args.sweep_tol,
        )
        ed = result.ed_energy
        ed_obs = result.ed_observables
        letta_obs = result.letta_observables
        print(
            f"{onsite_u:4.1f} "
            f"{_fmt(ed)} "
            f"{result.dmrg_energy:15.9f} "
            f"{result.letta_energy:15.9f} "
            f"{(np.nan if ed is None else result.dmrg_energy - ed):10.2e} "
            f"{(np.nan if ed is None else result.letta_energy - ed):10.2e} "
            f"{result.letta_energy - result.dmrg_energy:11.2e} "
            f"{(np.nan if ed_obs is None else ed_obs.condensate_fraction):6.3f} "
            f"{letta_obs.condensate_fraction:8.3f} "
            f"{(np.nan if ed_obs is None else ed_obs.average_number_variance):7.3f} "
            f"{letta_obs.average_number_variance:9.3f} "
            f"{(np.nan if ed_obs is None else ed_obs.edge_correlation):8.3f} "
            f"{letta_obs.edge_correlation:10.3f} "
            f"{result.dmrg_seconds:7.2f} "
            f"{result.letta_seconds:8.2f}"
        )


if __name__ == "__main__":
    main()
