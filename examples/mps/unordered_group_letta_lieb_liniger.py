"""Finite-volume Lieb-Liniger test for momentum-transfer matrix LETTA.

The corrected unordered generator is

    X = Q + R a_0^dagger + sum_q K(q) rho_q,
    rho_q = sum_k a_{k+q}^dagger a_k.

The number-conserving tie ``rho_q`` scatters an existing physical leg.  A
virtual charge shift opposite to ``q`` makes every combined insertion carry
zero total momentum; tracing the virtual space then retains closed momentum
histories.  The exponential is evaluated exactly in a finite momentum and
particle-number regulator before projection onto a fixed particle number.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
from scipy.linalg import eigh
from scipy.optimize import minimize
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm_multiply


def boson_basis(num_modes, max_particles):
    """Return occupations with total particle number at most ``max_particles``."""
    states = [
        occupation
        for occupation in product(range(max_particles + 1), repeat=num_modes)
        if sum(occupation) <= max_particles
    ]
    states.sort(key=lambda occupation: (sum(occupation), occupation))
    return states


def creation_operators(basis):
    """Construct mode creation operators in a total-number-truncated basis."""
    index = {occupation: position for position, occupation in enumerate(basis)}
    num_modes = len(basis[0])
    operators = []
    for mode in range(num_modes):
        rows = []
        columns = []
        values = []
        for column, occupation in enumerate(basis):
            target = list(occupation)
            target[mode] += 1
            target = tuple(target)
            if target in index:
                rows.append(index[target])
                columns.append(column)
                values.append(np.sqrt(target[mode]))
        operators.append(
            csr_matrix(
                (values, (rows, columns)),
                shape=(len(basis), len(basis)),
                dtype=complex,
            )
        )
    return operators


def momentum_transfer_operators(creators, cutoff):
    r"""Return ``rho_q = sum_k a^dagger_{k+q} a_k`` for all allowed ``q``."""
    transfers = {}
    for transfer in range(-2 * cutoff, 2 * cutoff + 1):
        operator = csr_matrix(creators[0].shape, dtype=complex)
        for source in range(2 * cutoff + 1):
            target = source + transfer
            if 0 <= target < 2 * cutoff + 1:
                operator += creators[target] @ creators[source].getH()
        transfers[transfer] = operator
    return transfers


def virtual_charge_shift(virtual_cutoff, transfer):
    """Shift virtual charge by ``-transfer`` without cyclic aliasing."""
    charges = np.arange(-virtual_cutoff, virtual_cutoff + 1, dtype=int)
    index = {charge: position for position, charge in enumerate(charges)}
    matrix = np.zeros((charges.size, charges.size), dtype=complex)
    for column, charge in enumerate(charges):
        target = charge - int(transfer)
        if target in index:
            matrix[index[target], column] = 1.0
    return matrix


def lieb_liniger_hamiltonian(*, particles, cutoff, length, coupling):
    r"""Return the fixed-number momentum-cutoff Lieb-Liniger Hamiltonian."""
    mode_numbers = np.arange(-cutoff, cutoff + 1, dtype=int)
    basis = [
        occupation
        for occupation in product(range(particles + 1), repeat=mode_numbers.size)
        if sum(occupation) == particles
    ]
    basis.sort()
    index = {occupation: position for position, occupation in enumerate(basis)}
    momenta = 2.0 * np.pi * mode_numbers / float(length)
    hamiltonian = np.zeros((len(basis), len(basis)), dtype=float)

    for column, occupation in enumerate(basis):
        hamiltonian[column, column] += np.dot(momenta * momenta, occupation)
        for ir, nr in enumerate(mode_numbers):
            for is_, ns in enumerate(mode_numbers):
                after_annihilation = list(occupation)
                amplitude = 1.0
                if after_annihilation[is_] == 0:
                    continue
                amplitude *= np.sqrt(after_annihilation[is_])
                after_annihilation[is_] -= 1
                if after_annihilation[ir] == 0:
                    continue
                amplitude *= np.sqrt(after_annihilation[ir])
                after_annihilation[ir] -= 1
                total_momentum = nr + ns
                for ip, np_ in enumerate(mode_numbers):
                    nq = total_momentum - np_
                    matches = np.flatnonzero(mode_numbers == nq)
                    if matches.size == 0:
                        continue
                    iq = int(matches[0])
                    target = after_annihilation.copy()
                    created = amplitude * np.sqrt(target[iq] + 1)
                    target[iq] += 1
                    created *= np.sqrt(target[ip] + 1)
                    target[ip] += 1
                    hamiltonian[index[tuple(target)], column] += (
                        float(coupling) / float(length) * created
                    )
    return mode_numbers, basis, 0.5 * (hamiltonian + hamiltonian.T)


def _unpack_parameters(parameters, *, cutoff, virtual_cutoff):
    max_transfer = min(2 * int(cutoff), 2 * int(virtual_cutoff))
    parameters = np.asarray(parameters, dtype=float)
    if parameters.shape != (1 + 2 * max_transfer,):
        raise ValueError(f"expected {1 + 2 * max_transfer} parameters.")
    confinement = np.exp(parameters[0])
    strengths = parameters[1:].reshape(max_transfer, 2)
    return confinement, strengths


def unordered_projected_state(
    parameters,
    *,
    particles,
    cutoff,
    virtual_cutoff,
    basis=None,
    creators=None,
    transfers=None,
):
    r"""Evaluate the corrected ``Tr exp(Q + R a0^dagger + K rho)|0>``."""
    num_modes = 2 * cutoff + 1
    if basis is None:
        basis = boson_basis(num_modes, particles)
    if creators is None:
        creators = creation_operators(basis)
    if transfers is None:
        transfers = momentum_transfer_operators(creators, cutoff)
    confinement, strengths = _unpack_parameters(
        parameters,
        cutoff=cutoff,
        virtual_cutoff=virtual_cutoff,
    )

    charges = np.arange(-virtual_cutoff, virtual_cutoff + 1, dtype=float)
    bond_dim = charges.size
    q_matrix = -confinement * np.diag(charges * charges)
    physical_identity = eye(len(basis), dtype=complex, format="csr")
    virtual_identity = eye(bond_dim, dtype=complex, format="csr")
    generator = kron(csr_matrix(q_matrix), physical_identity, format="csr")
    generator += kron(virtual_identity, creators[cutoff], format="csr")

    for shell, shell_strengths in enumerate(strengths, start=1):
        for transfer, strength in zip((-shell, shell), shell_strengths):
            shift = virtual_charge_shift(virtual_cutoff, transfer)
            generator += float(strength) * kron(
                csr_matrix(shift),
                transfers[transfer],
                format="csr",
            )

    vacuum = basis.index((0,) * num_modes)
    initial = np.zeros((bond_dim * len(basis), bond_dim), dtype=complex)
    for auxiliary in range(bond_dim):
        initial[auxiliary * len(basis) + vacuum, auxiliary] = 1.0
    evolved = expm_multiply(generator, initial)

    physical_state = np.zeros(len(basis), dtype=complex)
    for auxiliary in range(bond_dim):
        column = evolved[:, auxiliary].reshape(bond_dim, len(basis))
        physical_state += column[auxiliary]
    fixed_indices = [
        position for position, occupation in enumerate(basis) if sum(occupation) == particles
    ]
    projected = physical_state[fixed_indices]
    norm = np.linalg.norm(projected)
    if not np.isfinite(norm) or norm < 1.0e-14:
        raise FloatingPointError("the projected momentum-transfer state has invalid norm.")
    return projected / norm


def noncommutativity_diagnostics(parameters, *, cutoff, virtual_cutoff):
    """Measure commutators among the virtual confinement and charge shifts."""
    confinement, strengths = _unpack_parameters(
        parameters,
        cutoff=cutoff,
        virtual_cutoff=virtual_cutoff,
    )
    charges = np.arange(-virtual_cutoff, virtual_cutoff + 1, dtype=float)
    matrices = [-confinement * np.diag(charges * charges)]
    for shell, shell_strengths in enumerate(strengths, start=1):
        matrices.extend(
            strength * virtual_charge_shift(virtual_cutoff, transfer)
            for transfer, strength in zip((-shell, shell), shell_strengths)
        )
    norms = []
    for left in range(len(matrices)):
        for right in range(left + 1, len(matrices)):
            commutator = matrices[left] @ matrices[right] - matrices[right] @ matrices[left]
            norms.append(float(np.linalg.norm(commutator)))
    return {
        "maximum_commutator_norm": max(norms, default=0.0),
        "rms_commutator_norm": float(np.sqrt(np.mean(np.square(norms)))) if norms else 0.0,
    }


def optimize_state(
    hamiltonian,
    *,
    particles,
    cutoff,
    virtual_cutoff,
    restarts,
    maxiter,
    seed,
):
    """Optimize one virtual-charge truncation and require convergence."""
    basis = boson_basis(2 * cutoff + 1, particles)
    creators = creation_operators(basis)
    transfers = momentum_transfer_operators(creators, cutoff)
    parameter_count = 1 + 2 * min(2 * cutoff, 2 * virtual_cutoff)

    def objective(parameters):
        try:
            state = unordered_projected_state(
                parameters,
                particles=particles,
                cutoff=cutoff,
                virtual_cutoff=virtual_cutoff,
                basis=basis,
                creators=creators,
                transfers=transfers,
            )
        except (FloatingPointError, ValueError, OverflowError):
            return 1.0e12
        return float(np.real(np.vdot(state, hamiltonian @ state)))

    rng = np.random.default_rng(seed)
    base = np.zeros(parameter_count)
    starts = [base]
    starts.extend(0.2 * rng.standard_normal(parameter_count) for _ in range(restarts - 1))
    attempts = []
    bounds = [(-8.0, 4.0)] + [(-8.0, 8.0)] * (parameter_count - 1)
    for start in starts:
        result = minimize(
            objective,
            start,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": 1.0e-12, "gtol": 1.0e-7},
        )
        attempts.append(result)
    converged = [
        result for result in attempts if result.success and np.isfinite(result.fun)
    ]
    if not converged:
        messages = "; ".join(str(result.message) for result in attempts)
        raise RuntimeError(
            f"D={2 * virtual_cutoff + 1} optimization did not converge: {messages}"
        )
    best = min(converged, key=lambda result: result.fun)
    return {
        "energy": float(best.fun),
        "parameters": np.asarray(best.x, dtype=float),
        "success": True,
        "nit": int(best.nit),
        "nfev": int(best.nfev),
        "converged_restarts": len(converged),
        "total_restarts": len(attempts),
        "message": str(best.message),
    }


def run(args):
    mode_numbers, fixed_basis, hamiltonian = lieb_liniger_hamiltonian(
        particles=args.particles,
        cutoff=args.cutoff,
        length=args.length,
        coupling=args.coupling,
    )
    exact_values, exact_vectors = eigh(hamiltonian, subset_by_index=[0, 0])
    exact_energy = float(exact_values[0])
    exact_state = exact_vectors[:, 0]
    condensate = np.zeros(len(fixed_basis), dtype=complex)
    condensate_occupation = tuple(
        args.particles if mode == 0 else 0 for mode in mode_numbers
    )
    condensate[fixed_basis.index(condensate_occupation)] = 1.0
    condensate_energy = float(np.real(np.vdot(condensate, hamiltonian @ condensate)))

    rows = []
    for virtual_cutoff in args.virtual_cutoffs:
        if virtual_cutoff == 0:
            row = {
                "virtual_cutoff": 0,
                "bond_dim": 1,
                "energy": condensate_energy,
                "error": condensate_energy - exact_energy,
                "exact_fidelity": float(abs(np.vdot(exact_state, condensate)) ** 2),
                "success": True,
                "parameters": [],
                "maximum_commutator_norm": 0.0,
                "rms_commutator_norm": 0.0,
            }
        else:
            result = optimize_state(
                hamiltonian,
                particles=args.particles,
                cutoff=args.cutoff,
                virtual_cutoff=virtual_cutoff,
                restarts=args.restarts,
                maxiter=args.maxiter,
                seed=args.seed + virtual_cutoff,
            )
            state = unordered_projected_state(
                result["parameters"],
                particles=args.particles,
                cutoff=args.cutoff,
                virtual_cutoff=virtual_cutoff,
            )
            diagnostics = noncommutativity_diagnostics(
                result["parameters"],
                cutoff=args.cutoff,
                virtual_cutoff=virtual_cutoff,
            )
            row = {
                **{key: value for key, value in result.items() if key != "parameters"},
                "virtual_cutoff": int(virtual_cutoff),
                "bond_dim": int(2 * virtual_cutoff + 1),
                "error": float(result["energy"] - exact_energy),
                "exact_fidelity": float(abs(np.vdot(exact_state, state)) ** 2),
                "parameters": result["parameters"].tolist(),
                **diagnostics,
            }
        rows.append(row)
        print(
            f"D={row['bond_dim']} energy={row['energy']:.10f} "
            f"error={row['error']:.3e} fidelity={row['exact_fidelity']:.8f}"
        )

    output = {
        "model": "periodic Lieb-Liniger momentum-cutoff benchmark",
        "ansatz": "Tr exp[Q + R a_0^dagger + sum_q K(q) rho_q] |0>",
        "tie_operator": "rho_q=sum_k a_{k+q}^dagger a_k",
        "particles": int(args.particles),
        "length": float(args.length),
        "density": float(args.particles / args.length),
        "coupling": float(args.coupling),
        "momentum_cutoff": int(args.cutoff),
        "fixed_number_dimension": len(fixed_basis),
        "exact_energy": exact_energy,
        "condensate_energy": condensate_energy,
        "rows": rows,
    }
    print(json.dumps(output, indent=2))
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, indent=2) + "\n")
        print(f"wrote {path}")
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--particles", type=int, default=4)
    parser.add_argument("--length", type=float, default=4.0)
    parser.add_argument("--coupling", type=float, default=1.0)
    parser.add_argument("--cutoff", type=int, default=2)
    parser.add_argument("--virtual-cutoffs", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--restarts", type=int, default=8)
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument(
        "--output",
        default="/private/tmp/unordered_group_letta_lieb_liniger.json",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
