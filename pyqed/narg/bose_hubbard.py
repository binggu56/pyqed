"""Bose-Hubbard exact diagonalization and NARG block growth."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.sparse import dok_matrix
from scipy.sparse.linalg import eigsh


def boson_annihilation(dim: int) -> np.ndarray:
    """Return the truncated boson annihilation operator."""
    dim = int(dim)
    if dim < 1:
        raise ValueError("dim must be positive.")
    op = np.zeros((dim, dim), dtype=float)
    for n in range(1, dim):
        op[n - 1, n] = np.sqrt(n)
    return op


def fixed_number_basis(nsites: int, nbosons: int, nmax: int) -> list[tuple[int, ...]]:
    """Enumerate occupation states with fixed total boson number."""
    nsites = int(nsites)
    nbosons = int(nbosons)
    nmax = int(nmax)
    if nsites < 1:
        raise ValueError("nsites must be positive.")
    if nbosons < 0:
        raise ValueError("nbosons must be non-negative.")
    if nmax < 0:
        raise ValueError("nmax must be non-negative.")
    basis: list[tuple[int, ...]] = []

    def rec(site: int, remaining: int, state: list[int]) -> None:
        if site == nsites:
            if remaining == 0:
                basis.append(tuple(state))
            return
        rest = nsites - site - 1
        low = max(0, remaining - rest * nmax)
        high = min(nmax, remaining)
        for occ in range(low, high + 1):
            state.append(occ)
            rec(site + 1, remaining - occ, state)
            state.pop()

    rec(0, nbosons, [])
    return basis


def bose_hubbard_hamiltonian(
    nsites: int,
    nbosons: int,
    *,
    t: float = 1.0,
    U: float = 1.0,
    nmax: int | None = None,
    mu: float = 0.0,
    periodic: bool = False,
):
    """Build the fixed-number Bose-Hubbard Hamiltonian.

    The Hamiltonian is

    ``-t sum_<ij> (b_i^dag b_j + h.c.) + U/2 sum_i n_i(n_i-1) - mu sum_i n_i``.
    """
    if nmax is None:
        nmax = nbosons
    basis = fixed_number_basis(nsites, nbosons, nmax)
    if not basis:
        raise ValueError("No fixed-number basis states exist for this cutoff.")
    index = {state: i for i, state in enumerate(basis)}
    H = dok_matrix((len(basis), len(basis)), dtype=float)
    bonds = [(i, i + 1) for i in range(nsites - 1)]
    if periodic and nsites > 2:
        bonds.append((nsites - 1, 0))

    for col, state in enumerate(basis):
        diag = sum(0.5 * U * n * (n - 1) - mu * n for n in state)
        H[col, col] = H[col, col] + diag

        for i, j in bonds:
            occ = list(state)
            if occ[j] > 0 and occ[i] < nmax:
                amp = -t * np.sqrt((occ[i] + 1) * occ[j])
                occ[i] += 1
                occ[j] -= 1
                H[index[tuple(occ)], col] = H[index[tuple(occ)], col] + amp

            occ = list(state)
            if occ[i] > 0 and occ[j] < nmax:
                amp = -t * np.sqrt((occ[j] + 1) * occ[i])
                occ[j] += 1
                occ[i] -= 1
                H[index[tuple(occ)], col] = H[index[tuple(occ)], col] + amp

    return H.tocsr(), basis


def exact_bose_hubbard(
    nsites: int,
    nbosons: int,
    *,
    t: float = 1.0,
    U: float = 1.0,
    nmax: int | None = None,
    mu: float = 0.0,
    periodic: bool = False,
    nroots: int = 1,
):
    """Return lowest fixed-number Bose-Hubbard eigenpairs."""
    H, basis = bose_hubbard_hamiltonian(
        nsites,
        nbosons,
        t=t,
        U=U,
        nmax=nmax,
        mu=mu,
        periodic=periodic,
    )
    nroots = min(int(nroots), H.shape[0])
    if H.shape[0] <= max(64, nroots + 2):
        values, vectors = np.linalg.eigh(H.toarray())
        return values[:nroots], vectors[:, :nroots], basis
    values, vectors = eigsh(H, k=nroots, which="SA")
    order = np.argsort(values)
    return values[order], vectors[:, order], basis


@dataclass
class BoseHubbardObservables:
    one_body_density_matrix: np.ndarray
    condensate_fraction: float
    site_occupations: np.ndarray
    number_variances: np.ndarray
    average_number_variance: float
    edge_correlation: float


@dataclass
class BoseHubbardNARGStep:
    site: int
    product_dim: int
    kept: int
    lowest_energy: float
    qn: np.ndarray


@dataclass
class BoseHubbardNARGResult:
    energies: np.ndarray
    vectors: np.ndarray
    steps: list[BoseHubbardNARGStep]
    observables: list[BoseHubbardObservables]


def _normalized_vector(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=complex).reshape(-1)
    norm = np.sqrt(np.real(np.vdot(vector, vector)))
    if norm == 0.0:
        raise ValueError("Cannot compute Bose-Hubbard observables from a zero vector.")
    return vector / norm


def one_body_density_matrix(vector, basis) -> np.ndarray:
    """Return rho_ij = <b_i^dag b_j> in a fixed-number occupation basis."""
    vector = _normalized_vector(vector)
    basis = [tuple(state) for state in basis]
    if len(vector) != len(basis):
        raise ValueError("vector length must match basis length.")
    if not basis:
        raise ValueError("basis must be non-empty.")
    nsites = len(basis[0])
    index = {state: i for i, state in enumerate(basis)}
    rho = np.zeros((nsites, nsites), dtype=complex)

    for ket, state in enumerate(basis):
        ket_coeff = vector[ket]
        if ket_coeff == 0:
            continue
        for j in range(nsites):
            if state[j] == 0:
                continue
            occ = list(state)
            amp_j = np.sqrt(occ[j])
            occ[j] -= 1
            for i in range(nsites):
                amp = amp_j * np.sqrt(occ[i] + 1)
                occ[i] += 1
                bra = index.get(tuple(occ))
                if bra is not None:
                    rho[i, j] += np.conjugate(vector[bra]) * amp * ket_coeff
                occ[i] -= 1
    return rho


def onsite_number_statistics(vector, basis):
    """Return <n_i> and var(n_i) in a fixed-number occupation basis."""
    vector = _normalized_vector(vector)
    basis = [tuple(state) for state in basis]
    if len(vector) != len(basis):
        raise ValueError("vector length must match basis length.")
    occupations = np.asarray(basis, dtype=float)
    weights = np.abs(vector) ** 2
    mean = weights @ occupations
    mean2 = weights @ (occupations**2)
    variance = mean2 - mean**2
    return mean, variance


def condensate_fraction(one_body_density, nbosons: float) -> float:
    """Largest natural-orbital occupation divided by total boson number."""
    nbosons = float(nbosons)
    if nbosons <= 0.0:
        return 0.0
    rho = np.asarray(one_body_density, dtype=complex)
    rho = 0.5 * (rho + rho.T.conj())
    return float(np.max(np.linalg.eigvalsh(rho)).real / nbosons)


def bose_hubbard_observables(vector, basis) -> BoseHubbardObservables:
    """Compute number-conserving finite-chain Bose-Hubbard diagnostics."""
    rho = one_body_density_matrix(vector, basis)
    occupations, variances = onsite_number_statistics(vector, basis)
    total_bosons = float(np.sum(occupations))
    edge = (
        float(np.real(rho[0, -1]))
        if rho.shape[0] > 1
        else float(np.real(rho[0, 0]))
    )
    return BoseHubbardObservables(
        one_body_density_matrix=rho,
        condensate_fraction=condensate_fraction(rho, total_bosons),
        site_occupations=occupations,
        number_variances=variances,
        average_number_variance=float(np.mean(variances)),
        edge_correlation=edge,
    )


def _observables_from_projected_ops(vectors, b_ops, n_ops, nbosons):
    observables = []
    for root in range(vectors.shape[1]):
        vector = _normalized_vector(vectors[:, root])
        b_vectors = [op @ vector for op in b_ops]
        rho = np.array(
            [[np.vdot(bi, bj) for bj in b_vectors] for bi in b_vectors],
            dtype=complex,
        )
        occupations = np.array(
            [np.real(np.vdot(vector, op @ vector)) for op in n_ops],
            dtype=float,
        )
        variances = np.array(
            [
                np.real(np.vdot(vector, (op @ op) @ vector)) - occupations[i] ** 2
                for i, op in enumerate(n_ops)
            ],
            dtype=float,
        )
        edge = (
            float(np.real(rho[0, -1]))
            if rho.shape[0] > 1
            else float(np.real(rho[0, 0]))
        )
        observables.append(
            BoseHubbardObservables(
                one_body_density_matrix=rho,
                condensate_fraction=condensate_fraction(rho, nbosons),
                site_occupations=occupations,
                number_variances=variances,
                average_number_variance=float(np.mean(variances)),
                edge_correlation=edge,
            )
        )
    return observables


class BoseHubbardNARG:
    """Sequential fixed-number NARG for an open Bose-Hubbard chain."""

    def __init__(
        self,
        nsites: int,
        nbosons: int,
        *,
        t: float = 1.0,
        U: float = 1.0,
        D: int = 20,
        nmax: int | None = None,
        mu: float = 0.0,
    ):
        self.nsites = int(nsites)
        self.nbosons = int(nbosons)
        self.t = float(t)
        self.U = float(U)
        self.D = int(D)
        self.nmax = int(self.nbosons if nmax is None else nmax)
        self.mu = float(mu)
        if self.nsites < 1:
            raise ValueError("nsites must be positive.")
        if self.nbosons < 0:
            raise ValueError("nbosons must be non-negative.")
        if self.D < 1:
            raise ValueError("D must be positive.")
        if self.nmax < 0:
            raise ValueError("nmax must be non-negative.")
        if self.nbosons > self.nsites * self.nmax:
            raise ValueError("nbosons exceeds the local cutoff capacity.")

    @property
    def local_dim(self) -> int:
        return self.nmax + 1

    def _local_hamiltonian(self):
        n = np.arange(self.local_dim, dtype=float)
        return np.diag(0.5 * self.U * n * (n - 1) - self.mu * n)

    def _allowed_counts(self, nsites: int) -> set[int]:
        remaining = self.nsites - int(nsites)
        low = max(0, self.nbosons - remaining * self.nmax)
        high = min(self.nbosons, int(nsites) * self.nmax)
        return set(range(low, high + 1))

    @staticmethod
    def _diagonalize_by_number(H, qn, nroots, allowed):
        qn = np.asarray(qn, dtype=int)
        allowed = set(int(x) for x in allowed)
        roots = []
        for number in sorted(set(qn.tolist()) & allowed):
            idx = np.flatnonzero(qn == number)
            block = H[np.ix_(idx, idx)]
            block = 0.5 * (block + block.T.conj())
            values, vectors = np.linalg.eigh(block)
            for col, value in enumerate(values):
                roots.append(
                    (float(np.real(value)), number, idx, vectors[:, col].copy())
                )
        if not roots:
            raise ValueError(
                "No Bose-Hubbard NARG states remain in allowed number sectors."
            )
        roots.sort(key=lambda item: item[0])
        nselect = min(int(nroots), len(roots))
        values = np.empty(nselect)
        vectors = np.zeros((len(qn), nselect), dtype=complex)
        numbers = np.empty(nselect, dtype=int)
        for col, (value, number, idx, vector) in enumerate(roots[:nselect]):
            values[col] = value
            numbers[col] = number
            vectors[idx, col] = vector
        return values, vectors, numbers

    def run(self, nroots: int = 1) -> BoseHubbardNARGResult:
        b = boson_annihilation(self.local_dim)
        bdag = b.T.conj()
        hloc = self._local_hamiltonian()
        nloc = np.arange(self.local_dim, dtype=int)
        nsite_op = np.diag(nloc.astype(float))

        H_block = np.zeros((1, 1), dtype=complex)
        qn_block = np.array([0], dtype=int)
        boundary_b = np.zeros((1, 1), dtype=complex)
        block_b_ops = []
        block_n_ops = []
        steps = []
        observables = []

        for site in range(self.nsites):
            block_dim = H_block.shape[0]
            eye_block = np.eye(block_dim, dtype=complex)
            eye_local = np.eye(self.local_dim, dtype=complex)
            H = np.kron(H_block, eye_local) + np.kron(eye_block, hloc)
            if site > 0:
                H -= self.t * (
                    np.kron(boundary_b.conj().T, b)
                    + np.kron(boundary_b, bdag)
                )
            qn = np.repeat(qn_block, self.local_dim) + np.tile(nloc, block_dim)
            keep = int(nroots) if site == self.nsites - 1 else self.D
            allowed = (
                {self.nbosons}
                if site == self.nsites - 1
                else self._allowed_counts(site + 1)
            )
            energies, vectors, qn_keep = self._diagonalize_by_number(
                H, qn, keep, allowed
            )
            product_b_ops = [np.kron(op, eye_local) for op in block_b_ops]
            product_b_ops.append(np.kron(eye_block, b))
            product_n_ops = [np.kron(op, eye_local) for op in block_n_ops]
            product_n_ops.append(np.kron(eye_block, nsite_op))
            if site == self.nsites - 1:
                observables = _observables_from_projected_ops(
                    vectors,
                    product_b_ops,
                    product_n_ops,
                    self.nbosons,
                )
            new_boundary_b = np.kron(eye_block, b)
            boundary_b = vectors.conj().T @ new_boundary_b @ vectors
            block_b_ops = [vectors.conj().T @ op @ vectors for op in product_b_ops]
            block_n_ops = [vectors.conj().T @ op @ vectors for op in product_n_ops]
            H_block = np.diag(energies).astype(complex)
            qn_block = qn_keep
            steps.append(
                BoseHubbardNARGStep(
                    site=site,
                    product_dim=H.shape[0],
                    kept=len(energies),
                    lowest_energy=float(energies[0]),
                    qn=qn_keep.copy(),
                )
            )

        return BoseHubbardNARGResult(
            energies=energies[:nroots],
            vectors=vectors[:, :nroots],
            steps=steps,
            observables=observables[:nroots],
        )


__all__ = [
    "BoseHubbardNARG",
    "BoseHubbardNARGResult",
    "BoseHubbardNARGStep",
    "BoseHubbardObservables",
    "bose_hubbard_hamiltonian",
    "bose_hubbard_observables",
    "boson_annihilation",
    "condensate_fraction",
    "exact_bose_hubbard",
    "fixed_number_basis",
    "one_body_density_matrix",
    "onsite_number_statistics",
]
