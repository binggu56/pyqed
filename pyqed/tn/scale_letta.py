"""Small scale-invariant LETTA reference contractions.

The implementation is deliberately dense and limited to eight spin-1/2
sites.  It is a validation helper for the multiscale construction rather than
an alternative to the scalable tree contraction code in :mod:`pyqed.tn`.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from pyqed.narg import transverse_field_ising_hamiltonian


def polar_isometry(matrix, *, return_residual=False):
    """Return the closest column isometry to a dense matrix."""
    matrix = np.asarray(matrix, dtype=complex)
    if matrix.ndim != 2:
        raise ValueError("matrix must have two axes.")
    if matrix.shape[0] < matrix.shape[1]:
        raise ValueError("an isometry requires at least as many rows as columns.")
    left, _, right = np.linalg.svd(matrix, full_matrices=False)
    isometry = left @ right
    residual = float(
        np.linalg.norm(
            isometry.conj().T @ isometry - np.eye(isometry.shape[1])
        )
    )
    return (isometry, residual) if return_residual else isometry


def operator_schmidt_factors(gate, *, local_dim=2, tol=1.0e-12):
    r"""Factor a two-site gate over its left/right operator spaces.

    The returned arrays obey

    .. math::

        U_{ab,cd}=\sum_\gamma L^\gamma_{ac}R^\gamma_{bd}.

    The index ``gamma`` is the same-scale LETTA tie.
    """
    gate = np.asarray(gate, dtype=complex)
    local_dim = int(local_dim)
    expected = local_dim * local_dim
    if gate.shape != (expected, expected):
        raise ValueError("gate shape is inconsistent with local_dim.")
    operator_matrix = (
        gate.reshape(local_dim, local_dim, local_dim, local_dim)
        .transpose(0, 2, 1, 3)
        .reshape(expected, expected)
    )
    left_vectors, singular_values, right_vectors = np.linalg.svd(
        operator_matrix, full_matrices=False
    )
    if singular_values[0] == 0.0:
        raise ValueError("gate cannot be the zero operator.")
    keep = singular_values > float(tol) * singular_values[0]
    roots = np.sqrt(singular_values[keep])
    left = (
        left_vectors[:, keep].T.reshape(-1, local_dim, local_dim)
        * roots[:, None, None]
    )
    right = (
        right_vectors[keep].reshape(-1, local_dim, local_dim)
        * roots[:, None, None]
    )
    return left, right


def contract_operator_schmidt(left, right):
    """Contract an operator-Schmidt tie back into its two-site gate."""
    left = np.asarray(left, dtype=complex)
    right = np.asarray(right, dtype=complex)
    if left.ndim != 3 or right.ndim != 3:
        raise ValueError("operator-Schmidt factors must have three axes.")
    if left.shape[0] != right.shape[0]:
        raise ValueError("operator-Schmidt tie dimensions do not match.")
    if left.shape[1] != left.shape[2] or right.shape[1] != right.shape[2]:
        raise ValueError("operator-Schmidt factors must contain square operators.")
    tensor = np.einsum("gac,gbd->abcd", left, right, optimize=True)
    return tensor.reshape(left.shape[1] * right.shape[1], -1)


def parity_isometry(even_angle, odd_angle):
    """Return a real binary isometry intertwining Ising spin-flip parity."""
    even_angle = float(even_angle)
    odd_angle = float(odd_angle)
    root2 = np.sqrt(2.0)
    even_1 = np.array([1.0, 0.0, 0.0, 1.0]) / root2
    even_2 = np.array([0.0, 1.0, 1.0, 0.0]) / root2
    odd_1 = np.array([1.0, 0.0, 0.0, -1.0]) / root2
    odd_2 = np.array([0.0, 1.0, -1.0, 0.0]) / root2
    plus = np.array([1.0, 1.0]) / root2
    minus = np.array([1.0, -1.0]) / root2
    even = np.cos(even_angle) * even_1 + np.sin(even_angle) * even_2
    odd = np.cos(odd_angle) * odd_1 + np.sin(odd_angle) * odd_2
    return np.outer(even, plus) + np.outer(odd, minus)


def ising_tie_gate(angle):
    r"""Return the parity-preserving rank-two gate ``exp(-i angle Z x Y)``."""
    identity = np.eye(2, dtype=complex)
    y = np.array([[0.0, -1.0j], [1.0j, 0.0]])
    z = np.diag([1.0, -1.0])
    generator = np.kron(z, y)
    return np.cos(float(angle)) * np.kron(identity, identity) - 1j * np.sin(
        float(angle)
    ) * generator


def _kron_power(matrix, count):
    out = np.array([[1.0]], dtype=complex)
    for _ in range(int(count)):
        out = np.kron(out, matrix)
    return out


def _apply_pair_gate(vector, gate, nsites, bonds):
    tensor = np.asarray(vector, dtype=complex).reshape((2,) * int(nsites))
    for left, right in bonds:
        if right != left + 1:
            raise ValueError("only adjacent pair gates are supported.")
        remaining = [site for site in range(nsites) if site not in (left, right)]
        permutation = [left, right] + remaining
        inverse = np.argsort(permutation)
        matrix = np.transpose(tensor, permutation).reshape(4, -1)
        tensor = np.transpose(
            (gate @ matrix).reshape((2,) * nsites),
            inverse,
        )
    return tensor.reshape(-1)


def _operator_parity_basis(parity, sector):
    parity = np.asarray(parity, dtype=complex)
    dim = parity.shape[0]
    conjugation = np.empty((dim * dim, dim * dim), dtype=complex)
    for column in range(dim * dim):
        operator = np.zeros((dim, dim), dtype=complex)
        operator.flat[column] = 1.0
        conjugation[:, column] = (parity @ operator @ parity).reshape(-1)
    values, vectors = np.linalg.eigh(
        0.5 * (conjugation + conjugation.conj().T)
    )
    target = {"even": 1.0, "odd": -1.0}.get(str(sector).lower())
    if target is None:
        raise ValueError("sector must be 'even' or 'odd'.")
    return vectors[:, np.abs(values - target) < 1.0e-10]


class EightSiteScaleLETTA:
    r"""Dense eight-site binary scale-LETTA at fixed local dimension two.

    A single isometry ``V_star: C^2 -> C^2 x C^2`` is reused at all
    three scales.  At the four- and eight-site levels, adjacent tree cells are
    connected by the same operator-Schmidt-factorized gate.  ``q=1`` replaces
    that gate by the identity and is the ordinary scale-tied TTN limit.
    """

    def __init__(self, *, q=1, parameters=None):
        q = int(q)
        if q not in (1, 2):
            raise ValueError("this prototype supports q=1 or q=2.")
        self.q = q
        if parameters is None:
            parameters = (
                (0.4, 0.0, 3.0 * np.pi / 4.0)
                if q == 1
                else (0.5, -0.08, np.pi / 4.0, -0.1)
            )
        self.parameters = np.asarray(parameters, dtype=float)
        expected = 3 if q == 1 else 4
        if self.parameters.shape != (expected,):
            raise ValueError(f"q={q} requires {expected} parameters.")
        self.energy = None
        self.exact_energy = None
        self.energy_error = None
        self.fidelity = None
        self.history = []
        self.success = False
        self.message = "not optimized"

    @property
    def fixed_isometry(self):
        """Return the shared upper-layer isometry ``V_star``."""
        return polar_isometry(parity_isometry(*self.parameters[:2]))

    @property
    def top_state(self):
        """Return the normalized one-site state at the top of the tree."""
        angle = self.parameters[2]
        return np.array([np.cos(angle), np.sin(angle)], dtype=complex)

    @property
    def tie_gate(self):
        """Return the exact gate reconstructed by contracting its tie index."""
        raw_gate = (
            np.eye(4, dtype=complex)
            if self.q == 1
            else ising_tie_gate(self.parameters[3])
        )
        left, right = operator_schmidt_factors(raw_gate)
        return contract_operator_schmidt(left, right)

    @property
    def tie_factors(self):
        """Return the left and right factors carrying the shared ``gamma``."""
        raw_gate = (
            np.eye(4, dtype=complex)
            if self.q == 1
            else ising_tie_gate(self.parameters[3])
        )
        return operator_schmidt_factors(raw_gate)

    @property
    def tie_dimension(self):
        return int(self.tie_factors[0].shape[0])

    def two_cell_layer(self, *, canonical=True):
        """Contract one tie and two copies of ``V_star`` into a scale layer."""
        identity = np.eye(2, dtype=complex)
        expanded = np.kron(self.fixed_isometry, self.fixed_isometry)
        tied = np.kron(identity, np.kron(self.tie_gate, identity)) @ expanded
        return polar_isometry(tied) if canonical else tied

    def state_vector(self):
        """Contract every tied layer exactly and return the normalized state."""
        isometry = self.fixed_isometry
        gate = self.tie_gate
        state = isometry @ self.top_state
        state = _kron_power(isometry, 2) @ state
        state = _apply_pair_gate(state, gate, 4, ((1, 2),))
        state = _kron_power(isometry, 4) @ state
        state = _apply_pair_gate(
            state,
            gate,
            8,
            ((1, 2), (3, 4), (5, 6)),
        )
        return state / np.linalg.norm(state)

    def ttn_state_vector(self):
        """Return an independent recursive TTN contraction for ``q=1``."""
        if self.q != 1:
            raise ValueError("the tie-free TTN reference requires q=1.")
        tensor = self.fixed_isometry.reshape(2, 2, 2)
        state = np.einsum(
            "ija,klb,mnc,opd,abe,cdf,efg,g->ijklmnop",
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            tensor,
            self.top_state,
            optimize=True,
        )
        vector = state.reshape(-1)
        return vector / np.linalg.norm(vector)

    def norm(self):
        return float(np.linalg.norm(self.state_vector()))

    def expectation(self, operator):
        operator = np.asarray(operator, dtype=complex)
        if operator.shape != (256, 256):
            raise ValueError("operator must act on eight spin-1/2 sites.")
        state = self.state_vector()
        return float(np.real(np.vdot(state, operator @ state)))

    def fit_critical_ising(self, *, periodic=True, maxiter=300):
        """Minimize the eight-site critical-Ising energy in this ansatz."""
        hamiltonian = transverse_field_ising_hamiltonian(
            8, periodic=bool(periodic), sparse=False
        )
        initial_energy = self.expectation(hamiltonian)
        self.history = [initial_energy]

        def objective(parameters):
            self.parameters = np.asarray(parameters, dtype=float)
            return self.expectation(hamiltonian)

        def record(parameters):
            self.parameters = np.asarray(parameters, dtype=float)
            self.history.append(self.expectation(hamiltonian))

        result = minimize(
            objective,
            self.parameters.copy(),
            method="BFGS",
            callback=record,
            options={"gtol": 1.0e-10, "maxiter": int(maxiter)},
        )
        self.parameters = np.asarray(result.x, dtype=float)
        self.energy = self.expectation(hamiltonian)
        exact_values, exact_vectors = np.linalg.eigh(hamiltonian)
        self.exact_energy = float(exact_values[0])
        self.energy_error = float(self.energy - self.exact_energy)
        state = self.state_vector()
        self.fidelity = float(abs(np.vdot(exact_vectors[:, 0], state)) ** 2)
        self.success = bool(
            np.isfinite(self.energy) and self.energy <= initial_energy + 1.0e-10
        )
        self.message = (
            "energy stationary" if self.success else str(result.message)
        )
        if not self.history or abs(self.history[-1] - self.energy) > 1.0e-13:
            self.history.append(self.energy)
        return self

    def scaling_superoperator(self):
        r"""Return the two-site ascending channel for the shared ``V_star``.

        The two neighboring coarse sites expand to four fine sites.  Their
        middle children are coupled by the LETTA tie, and a two-site operator
        on that causal cone is ascended back to the two coarse sites.
        """
        layer = self.two_cell_layer(canonical=True)
        identity = np.eye(2, dtype=complex)
        superoperator = np.empty((16, 16), dtype=complex)
        for column in range(16):
            operator = np.zeros((4, 4), dtype=complex)
            operator.flat[column] = 1.0
            fine_operator = np.kron(identity, np.kron(operator, identity))
            ascended = layer.conj().T @ fine_operator @ layer
            superoperator[:, column] = ascended.reshape(-1)
        return superoperator

    def scaling_dimensions(self, *, sector=None, tol=1.0e-12):
        """Diagonalize the shared scaling channel and return sorted dimensions."""
        superoperator = self.scaling_superoperator()
        if sector is not None:
            x = np.array([[0.0, 1.0], [1.0, 0.0]])
            basis = _operator_parity_basis(np.kron(x, x), sector)
            superoperator = basis.conj().T @ superoperator @ basis
        eigenvalues = np.linalg.eigvals(superoperator)
        keep = np.abs(eigenvalues) > float(tol)
        eigenvalues = eigenvalues[keep]
        dimensions = -np.log2(np.abs(eigenvalues))
        order = np.argsort(dimensions)
        return {
            "superoperator": superoperator,
            "eigenvalues": eigenvalues[order],
            "dimensions": np.real(dimensions[order]),
            "sector": sector,
        }


__all__ = [
    "EightSiteScaleLETTA",
    "contract_operator_schmidt",
    "ising_tie_gate",
    "operator_schmidt_factors",
    "parity_isometry",
    "polar_isometry",
]
