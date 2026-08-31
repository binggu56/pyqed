"""Hubbard-model helpers for qchem NARG backends."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .hamiltonian import (
    IntegralHamiltonian,
    MPOHamiltonian,
    normalize_basis,
    normalize_form,
    normalize_symmetry,
)


def _default_nelec(nsites: int) -> tuple[int, int]:
    nsites = int(nsites)
    nup = (nsites + 1) // 2
    ndown = nsites // 2
    return nup, ndown


def _spin_tuple(nelec=None, *, nup=None, ndown=None, nsites: int | None = None):
    if nelec is not None and (nup is not None or ndown is not None):
        raise ValueError("Pass either nelec or nup/ndown, not both.")
    if nelec is None:
        if nup is None and ndown is None:
            if nsites is None:
                raise ValueError("nsites is required to infer half filling.")
            return _default_nelec(nsites)
        if nup is None or ndown is None:
            raise ValueError("Pass both nup and ndown.")
        return int(nup), int(ndown)
    if isinstance(nelec, (tuple, list)):
        if len(nelec) != 2:
            raise ValueError("nelec tuple must be (nup, ndown).")
        return int(nelec[0]), int(nelec[1])
    total = int(nelec)
    return (total + 1) // 2, total // 2


def chain_bonds(nsites: int, *, periodic: bool = False) -> tuple[tuple[int, int], ...]:
    """Return nearest-neighbor bonds for a one-dimensional chain."""
    nsites = int(nsites)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    bonds = [(i, i + 1) for i in range(nsites - 1)]
    if periodic and nsites > 2:
        bonds.append((nsites - 1, 0))
    return tuple(bonds)


def square_lattice_bonds(
    lx: int,
    ly: int,
    *,
    periodic_x: bool = False,
    periodic_y: bool = False,
) -> tuple[tuple[int, int], ...]:
    """Return nearest-neighbor bonds for a flattened square lattice."""
    lx = int(lx)
    ly = int(ly)
    if lx < 1 or ly < 1:
        raise ValueError("lx and ly must be positive.")
    bonds = []

    def site(x, y):
        return int(x) + lx * int(y)

    for y in range(ly):
        for x in range(lx):
            if x + 1 < lx:
                bonds.append((site(x, y), site(x + 1, y)))
            elif periodic_x and lx > 2:
                bonds.append((site(x, y), site(0, y)))
            if y + 1 < ly:
                bonds.append((site(x, y), site(x, y + 1)))
            elif periodic_y and ly > 2:
                bonds.append((site(x, y), site(x, 0)))
    return tuple(bonds)


def hubbard_integrals(
    nsites: int,
    *,
    t: float = 1.0,
    U: float = 4.0,
    mu: float = 0.0,
    bonds=None,
):
    """Return spatial-orbital qchem integrals for a spinful Hubbard model."""
    nsites = int(nsites)
    h1e = np.zeros((nsites, nsites), dtype=float)
    if bonds is None:
        bonds = chain_bonds(nsites)
    for i, j in bonds:
        i = int(i)
        j = int(j)
        h1e[i, j] += -float(t)
        h1e[j, i] += -float(t)
    if mu:
        h1e[np.diag_indices(nsites)] -= float(mu)

    eri = np.zeros((nsites, nsites, nsites, nsites), dtype=float)
    for i in range(nsites):
        eri[i, i, i, i] = float(U)
    return h1e, eri


def real_momentum_orbitals(nsites: int, *, order: str = "energy", nelec=None):
    """Return a real Fourier orbital transform for a periodic 1D Hubbard chain."""
    nsites = int(nsites)
    sites = np.arange(nsites)
    columns = [np.ones(nsites) / np.sqrt(nsites)]
    energies = [-2.0]
    for m in range(1, (nsites + 1) // 2):
        if 2 * m == nsites:
            continue
        theta = 2.0 * np.pi * m * sites / nsites
        energy = -2.0 * np.cos(2.0 * np.pi * m / nsites)
        columns.append(np.sqrt(2.0 / nsites) * np.cos(theta))
        columns.append(np.sqrt(2.0 / nsites) * np.sin(theta))
        energies.extend((energy, energy))
    if nsites % 2 == 0:
        columns.append((-1.0) ** sites / np.sqrt(nsites))
        energies.append(2.0)
    transform = np.column_stack(columns)
    energies = np.asarray(energies, dtype=float)

    key = str(order).lower().replace("-", "_")
    if key in {"chain", "site", "natural"}:
        orbital_order = np.arange(nsites)
    elif key == "energy":
        orbital_order = np.argsort(energies, kind="stable")
    elif key == "fermi":
        nocc = int(nelec[0] if isinstance(nelec, tuple) else nsites // 2)
        sorted_eps = np.sort(energies)
        if nocc <= 0:
            mu = sorted_eps[0] - 1.0
        elif nocc >= nsites:
            mu = sorted_eps[-1] + 1.0
        else:
            mu = 0.5 * (sorted_eps[nocc - 1] + sorted_eps[nocc])
        orbital_order = np.lexsort((np.arange(nsites), np.abs(energies - mu)))
    elif key in {"particle_hole", "ph"}:
        nocc = int(nelec[0] if isinstance(nelec, tuple) else nsites // 2)
        sorted_eps = np.sort(energies)
        if nocc <= 0:
            mu = sorted_eps[0] - 1.0
        elif nocc >= nsites:
            mu = sorted_eps[-1] + 1.0
        else:
            mu = 0.5 * (sorted_eps[nocc - 1] + sorted_eps[nocc])
        shell_tol = 1.0e-12
        orbital_order = [
            i
            for i in np.lexsort((np.arange(nsites), np.abs(energies - mu)))
            if abs(energies[i] - mu) <= shell_tol
        ]
        holes = [
            i
            for i in np.lexsort((np.arange(nsites), mu - energies))
            if energies[i] < mu - shell_tol
        ]
        particles = [
            i
            for i in np.lexsort((np.arange(nsites), energies - mu))
            if energies[i] > mu + shell_tol
        ]
        for hole, particle in zip(holes, particles):
            orbital_order.extend((hole, particle))
        orbital_order.extend(holes[len(particles):])
        orbital_order.extend(particles[len(holes):])
        orbital_order = np.asarray(orbital_order, dtype=int)
    else:
        raise ValueError("order must be 'chain', 'energy', 'fermi', or 'particle_hole'.")
    return transform[:, orbital_order], energies[orbital_order]


def transform_spatial_integrals(h1e, eri, coeff):
    """Transform spatial-orbital integrals by ``c_site = coeff @ c_new``."""
    coeff = np.asarray(coeff)
    h1e = np.asarray(h1e)
    eri = np.asarray(eri)
    h_new = coeff.conj().T @ h1e @ coeff
    g_new = np.einsum(
        "ip,jq,ijkl,kr,ls->pqrs",
        coeff.conj(),
        coeff,
        eri,
        coeff.conj(),
        coeff,
        optimize=True,
    )
    return np.real_if_close(h_new, tol=1000), np.real_if_close(g_new, tol=1000)


def _resolve_symmetry(symmetry, *, default: str) -> str:
    return normalize_symmetry(default if symmetry is None else symmetry)


class HubbardMol:
    """Minimal molecule-like object expected by qchem NARG."""

    def __init__(self, nelec):
        self.nelec = tuple(int(x) for x in nelec)
        self.spin = int(self.nelec[0] - self.nelec[1])
        self.nelectron = int(sum(self.nelec))

    def energy_nuc(self):
        return 0.0


class HubbardMF:
    """Restricted mean-field facade around Hubbard integrals."""

    def __init__(self, h1e, eri, mol: HubbardMol):
        self.h1e = np.asarray(h1e, dtype=float)
        self.eri = np.asarray(eri, dtype=float)
        self.mol = mol
        self.nelec = mol.nelec
        self.nmo = int(self.h1e.shape[0])
        self.mo_coeff = np.eye(self.nmo)

    def get_hcore(self):
        return np.array(self.h1e, copy=True)

    def get_hcore_mo(self, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)
        return mo_coeff.conj().T @ self.h1e @ mo_coeff

    def get_eri_mo(self, mo_coeff=None, notation="chem"):
        del notation
        if mo_coeff is None:
            mo_coeff = self.mo_coeff
        mo_coeff = np.asarray(mo_coeff)
        return np.einsum(
            "ip,jq,ijkl,kr,ls->pqrs",
            mo_coeff.conj(),
            mo_coeff,
            self.eri,
            mo_coeff.conj(),
            mo_coeff,
            optimize=True,
        )

    def energy_nuc(self):
        return self.mol.energy_nuc()


class HubbardMPONARG:
    """Number-symmetric Hubbard NARG with a graph-frontier MPO environment."""

    def __init__(
        self,
        hamiltonian: MPOHamiltonian,
        *,
        D=20,
        n0=4,
        nstates=1,
        dressing=None,
        chi=None,
        frame_adapt_tol=None,
        frame_max_dim=None,
        frame_expand_dim=1,
        **options,
    ):
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"unsupported Hubbard MPO-NARG options: {unknown}")
        if hamiltonian.symmetry != "number":
            raise NotImplementedError("Hubbard MPO-NARG currently requires symmetry='number'.")
        if hamiltonian.basis != "site":
            raise NotImplementedError("Hubbard MPO-NARG currently requires basis='site'.")
        if hamiltonian.model is None:
            raise ValueError("Hubbard MPO-NARG requires its originating Hubbard model.")
        self.hamiltonian = hamiltonian
        self.model = hamiltonian.model
        self.D = int(D)
        self.n0 = int(n0)
        self.nstates = int(nstates)
        self.dressing = dressing
        self.chi = chi
        self.frame_adapt_tol = frame_adapt_tol
        self.frame_max_dim = frame_max_dim
        self.frame_expand_dim = int(frame_expand_dim)
        self.energy = None
        self.e_tot = None
        self.vectors = None
        self.history = []
        self.detached_history = []
        self.success = False
        self.message = "not run"

    def run(self, **options):
        if options:
            unknown = ", ".join(sorted(options))
            raise TypeError(f"unsupported Hubbard MPO-NARG run options: {unknown}")
        from pyqed.mps.fermion import SpinHalfFermionChain
        from pyqed.narg.qchem.abelian import (
            JW,
            LOCAL_QN,
            add_local_kron_blocks_hc_lloo,
            add_local_kron_blocks_lloo,
            cdd,
            cdu,
            cd,
            charge_diagonalize,
            cu,
            detached_frame_transition_projector,
            diagonalize_by_qn,
            feasible_branch_qns,
            feasible_qns,
            primitive_charge_labels,
            project_two_site_operator,
        )

        model = self.model
        nsites = int(model.nsites)
        n0 = min(max(1, self.n0), nsites)

        def dense(operator):
            return operator.toarray() if hasattr(operator, "toarray") else np.asarray(operator)

        h1e, eri = model.integrals(basis="site")
        initial = SpinHalfFermionChain(
            h1e[:n0, :n0],
            eri[:n0, :n0, :n0, :n0],
            nelec=model.nelec,
        )
        initial.jordan_wigner(forward=False)
        block_h = np.asarray(dense(initial.H), dtype=complex)
        block_qn = primitive_charge_labels(n0)
        bonds = {tuple(sorted((int(i), int(j)))) for i, j in model.bonds}

        def has_future(site, prefix):
            return any(site in bond and max(bond) >= prefix for bond in bonds)

        frontier = {
            site: (dense(initial.Cdu[site]), dense(initial.Cdd[site]))
            for site in range(n0)
            if has_future(site, n0)
        }
        target_qn = (sum(model.nelec), model.spin)
        dressing = (
            "none"
            if self.dressing is None
            else str(self.dressing).lower().replace("-", "_")
        )
        if dressing in {"detached", "detached_frame"}:
            dressing = "detached_frames"
        if dressing not in {"none", "detached_frames"}:
            raise ValueError("Hubbard MPO-NARG dressing must be None or 'detached_frames'.")
        chi = self.chi
        if dressing == "detached_frames":
            frame_space = len(LOCAL_QN) * self.D
            chi = 2 * frame_space if chi is None else int(chi)
            if not frame_space < chi <= len(LOCAL_QN) * frame_space:
                raise ValueError(
                    f"detached_frames requires {frame_space} < chi <= "
                    f"{len(LOCAL_QN) * frame_space} for D={self.D}."
                )

        for site in range(n0, nsites):
            old_dim = block_h.shape[0]
            h_lloo = np.zeros(
                (4, 4, old_dim, old_dim),
                dtype=np.result_type(block_h, complex),
            )
            add_local_kron_blocks_lloo(h_lloo, block_h, np.eye(4))
            onsite = h1e[site, site] * (cdu @ cu + cdd @ cd)
            onsite += eri[site, site, site, site] * (cdu @ cu) @ (cdd @ cd)
            add_local_kron_blocks_lloo(h_lloo, np.eye(old_dim), onsite)
            for previous, (create_up, create_down) in frontier.items():
                coefficient = h1e[previous, site]
                if abs(coefficient) == 0:
                    continue
                add_local_kron_blocks_hc_lloo(
                    h_lloo, create_up, JW @ cu, coefficient
                )
                add_local_kron_blocks_hc_lloo(
                    h_lloo, create_down, JW @ cd, coefficient
                )
            primitive_qn = (
                block_qn[:, None, :] + LOCAL_QN[None, :, :]
            ).reshape((-1, LOCAL_QN.shape[1]))
            output_allowed = feasible_qns(target_qn, site + 1, nsites)

            diagnostics = None
            if dressing == "detached_frames":
                frame_allowed = {
                    tuple(np.asarray(output) - local)
                    for output in output_allowed
                    for local in LOCAL_QN
                }
                block_h, projector, output_qn, diagnostics = (
                    detached_frame_transition_projector(
                        h_lloo,
                        block_qn,
                        frame_dim=min(self.D, old_dim),
                        bond_dim=chi,
                        allowed_frame_qn=frame_allowed,
                        allowed_output_qn=output_allowed,
                        adapt_tol=self.frame_adapt_tol,
                        max_frame_rank=self.frame_max_dim,
                        expand_dim=self.frame_expand_dim,
                    )
                )
                self.detached_history.append(dict(diagnostics))
            else:
                branches = []
                labels = []
                for local_state, local_qn in enumerate(LOCAL_QN):
                    allowed = feasible_branch_qns(
                        target_qn, site, nsites, local_qn
                    )
                    _energies, vectors, qn = diagonalize_by_qn(
                        h_lloo[local_state, local_state],
                        block_qn,
                        min(self.D, old_dim),
                        allowed_qn=allowed,
                        allow_empty=True,
                    )
                    branches.append(vectors)
                    labels.append(qn + local_qn)
                output_dim = sum(branch.shape[1] for branch in branches)
                projector = np.zeros((old_dim, 4, output_dim), dtype=complex)
                output_qn = np.empty((output_dim, LOCAL_QN.shape[1]), dtype=int)
                offset = 0
                for local_state, (branch, qn) in enumerate(zip(branches, labels)):
                    width = branch.shape[1]
                    projector[:, local_state, offset : offset + width] = branch
                    output_qn[offset : offset + width] = qn
                    offset += width
                full_h = np.ascontiguousarray(
                    h_lloo.transpose(2, 0, 3, 1)
                ).reshape(old_dim * 4, old_dim * 4)
                matrix = projector.reshape(old_dim * 4, output_dim)
                block_h = matrix.conj().T @ (full_h @ matrix)
                block_h = 0.5 * (block_h + block_h.T.conj())

            next_frontier = {}
            for previous, (create_up, create_down) in frontier.items():
                if has_future(previous, site + 1):
                    next_frontier[previous] = (
                        project_two_site_operator(create_up, JW, projector),
                        project_two_site_operator(create_down, JW, projector),
                    )
            if has_future(site, site + 1):
                identity = np.eye(old_dim)
                next_frontier[site] = (
                    project_two_site_operator(identity, cdu, projector),
                    project_two_site_operator(identity, cdd, projector),
                )
            frontier = next_frontier
            block_qn = output_qn
            self.history.append(
                {
                    "site": site,
                    "retained_dim": int(block_h.shape[0]),
                    "frontier_width": len(frontier),
                    "environment_matrices": 1 + 2 * len(frontier),
                    "detached": diagnostics is not None,
                }
            )

        energies, vectors, final_qn = charge_diagonalize(
            block_h,
            block_qn,
            self.nstates,
            allowed_qn={target_qn},
        )
        self.energy = energies
        self.e_tot = energies
        self.vectors = vectors
        self.final_qn = final_qn
        self.success = True
        self.message = "converged"
        return self.e_tot, self.vectors


@dataclass
class Hubbard:
    """Spinful Hubbard model with a direct qchem-NARG launcher."""

    nsites: int = 4
    nelec: int | tuple[int, int] | None = None
    nup: int | None = None
    ndown: int | None = None
    t: float = 1.0
    U: float = 4.0
    mu: float = 0.0
    periodic: bool = False
    bonds: tuple[tuple[int, int], ...] | None = None
    lx: int | None = None
    ly: int | None = None
    periodic_x: bool = False
    periodic_y: bool = False

    def __post_init__(self):
        if self.lx is not None or self.ly is not None:
            if self.lx is None or self.ly is None:
                raise ValueError("Pass both lx and ly for a square lattice.")
            self.nsites = int(self.lx) * int(self.ly)
            self.bonds = square_lattice_bonds(
                int(self.lx),
                int(self.ly),
                periodic_x=self.periodic_x,
                periodic_y=self.periodic_y,
            )
        else:
            self.nsites = int(self.nsites)
            if self.bonds is None:
                self.bonds = chain_bonds(self.nsites, periodic=self.periodic)
            else:
                self.bonds = tuple((int(i), int(j)) for i, j in self.bonds)
        self.nelec = _spin_tuple(
            self.nelec,
            nup=self.nup,
            ndown=self.ndown,
            nsites=self.nsites,
        )
        self.nup, self.ndown = self.nelec

    @property
    def spin(self) -> int:
        return int(self.nup - self.ndown)

    @property
    def h1e(self):
        return self.integrals()[0]

    @property
    def eri(self):
        return self.integrals()[1]

    @property
    def mol(self) -> HubbardMol:
        return HubbardMol(self.nelec)

    def integrals(self, *, basis: str = "site", order: str = "energy"):
        h1e, eri = hubbard_integrals(
            self.nsites,
            t=self.t,
            U=self.U,
            mu=self.mu,
            bonds=self.bonds,
        )
        basis = normalize_basis(basis)
        if basis == "site":
            return h1e, eri
        if basis == "momentum":
            if not self.periodic and not (
                self.lx is None
                and tuple(self.bonds) == chain_bonds(self.nsites, periodic=True)
            ):
                raise ValueError(
                    "basis='momentum' currently requires a periodic 1D Hubbard chain."
                )
            coeff, _energies = real_momentum_orbitals(
                self.nsites,
                order=order,
                nelec=self.nelec,
            )
            return transform_spatial_integrals(h1e, eri, coeff)
        raise ValueError("basis must be 'site' or 'momentum'.")

    def mean_field(self, *, basis: str = "site", order: str = "energy") -> HubbardMF:
        h1e, eri = self.integrals(basis=basis, order=order)
        return HubbardMF(h1e, eri, self.mol)

    def H(
        self,
        *,
        basis: str = "site",
        symmetry: str | None = None,
        form: str = "auto",
        order: str = "energy",
        orbital_blocks=None,
    ):
        """Return a Hamiltonian object for generic NARG dispatch."""
        basis = normalize_basis(basis)
        symmetry = _resolve_symmetry(symmetry, default="number")
        form = normalize_form(form)
        if form == "auto":
            form = (
                "mpo"
                if basis == "site" and symmetry == "number" and orbital_blocks is None
                else "integrals"
            )
        if symmetry == "momentum":
            raise NotImplementedError(
                "symmetry='momentum' is reserved for total-K sectors and is not implemented yet."
            )
        metadata = {
            "t": float(self.t),
            "U": float(self.U),
            "mu": float(self.mu),
            "bonds": tuple(self.bonds),
            "order": str(order),
            "representation": "graph_frontier" if form == "mpo" else "integrals",
        }
        if form == "integrals":
            h1e, eri = self.integrals(basis=basis, order=order)
            mol = self.mol
            mf = HubbardMF(h1e, eri, mol)
            return IntegralHamiltonian(
                basis=basis,
                symmetry=symmetry,
                form=form,
                orbital_blocks=orbital_blocks,
                target={"nelec": self.nelec, "spin": self.spin},
                metadata=metadata,
                h1e=h1e,
                eri=eri,
                mol=mol,
                mf=mf,
                model=self,
            )
        if form == "mpo":
            return MPOHamiltonian(
                basis=basis,
                symmetry=symmetry,
                form=form,
                orbital_blocks=orbital_blocks,
                target={"nelec": self.nelec, "spin": self.spin},
                metadata=metadata,
                tensors=(),
                sites=tuple(range(self.nsites)),
                fermionic=True,
                model=self,
            )
        raise ValueError("form must be 'auto', 'integrals', or 'mpo'.")

    def hamiltonian(self, **options):
        """Alias for :meth:`H`."""
        return self.H(**options)

    def NARG(
        self,
        *,
        basis: str = "site",
        symmetry=None,
        form: str = "auto",
        order: str = "energy",
        orbital_blocks=None,
        run: bool = True,
        **options,
    ):
        """Build and optionally run a NARG solver for this Hubbard model."""
        from pyqed.narg import NARG as GenericNARG

        if "blocks" in options:
            raise TypeError("Hubbard.NARG(..., blocks=...) was removed; use symmetry=... instead.")
        form = normalize_form(form)
        symmetry = _resolve_symmetry(
            symmetry,
            default="number" if form == "mpo" else "spin",
        )
        if symmetry == "spin":
            options.setdefault("target_j2", abs(self.spin))
        hamiltonian = self.H(
            basis=basis,
            symmetry=symmetry,
            form=form,
            order=order,
            orbital_blocks=orbital_blocks,
        )
        solver = GenericNARG(hamiltonian, **options)
        if run:
            solver.run()
        return solver

    def qchem_inputs(self, *, basis: str = "site", order: str = "energy"):
        """Return ``(mf, mol, h1e, eri)`` for lower-level qchem drivers."""
        h1e, eri = self.integrals(basis=basis, order=order)
        mol = self.mol
        mf = HubbardMF(h1e, eri, mol)
        return mf, mol, h1e, eri

__all__ = [
    "Hubbard",
    "HubbardMPONARG",
    "HubbardMF",
    "HubbardMol",
    "chain_bonds",
    "hubbard_integrals",
    "real_momentum_orbitals",
    "square_lattice_bonds",
    "transform_spatial_integrals",
]
