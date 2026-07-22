#!/usr/bin/env python3
"""Benchmark differentiable JAX dominant eigensolver against SciPy ARPACK."""

from __future__ import annotations

import argparse
import time

import numpy as np


def transfer_pair(dim, seed):
    rng = np.random.default_rng(seed)
    q = -0.5 * np.eye(dim, dtype=np.complex128)
    q += 0.04 * (rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim)))
    r = 0.12 * (rng.normal(size=(dim, dim)) + 1.0j * rng.normal(size=(dim, dim)))
    return q, r


def make_actions(q, r):
    dim = q.shape[0]

    def action(vector):
        matrix = np.asarray(vector).reshape(dim, dim)
        return (q @ matrix + matrix @ q.conj().T + r @ matrix @ r.conj().T).reshape(-1)

    def adjoint(vector):
        matrix = np.asarray(vector).reshape(dim, dim)
        return (q.conj().T @ matrix + matrix @ q + r.conj().T @ matrix @ r).reshape(-1)

    return action, adjoint


def scipy_reference(action, adjoint, size, *, tol):
    from scipy.sparse.linalg import LinearOperator, eigs

    operator = LinearOperator((size, size), matvec=action, dtype=np.complex128)
    values, vectors = eigs(operator, k=1, which="LR", tol=tol, maxiter=5000, ncv=32)
    value = complex(values[0])
    vector = vectors[:, 0]
    residual = np.linalg.norm(action(vector) - value * vector) / np.linalg.norm(vector)
    return value, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dims", type=int, nargs="+", default=(8, 18, 32))
    parser.add_argument("--iterations", type=int, default=512)
    parser.add_argument("--shift", type=float, default=2.0)
    parser.add_argument("--tol", type=float, default=1.0e-10)
    args = parser.parse_args()

    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)
    from pyqed import dominant_eig

    print("dim transfer_dim scipy_s jax_compile_s jax_warm_s eig_error jax_residual")
    for index, dim in enumerate(args.dims):
        q, r = transfer_pair(int(dim), seed=71 + index)
        action, adjoint = make_actions(q, r)
        size = int(dim) * int(dim)

        start = time.perf_counter()
        scipy_value, scipy_residual = scipy_reference(action, adjoint, size, tol=args.tol)
        scipy_time = time.perf_counter() - start

        q_jax = jnp.asarray(q)
        r_jax = jnp.asarray(r)
        right0 = jnp.ones(size, dtype=jnp.complex128)
        left0 = jnp.ones(size, dtype=jnp.complex128)

        def solve():
            return dominant_eig(
                lambda vector: (q_jax @ vector.reshape(dim, dim)
                                + vector.reshape(dim, dim) @ q_jax.conj().T
                                + r_jax @ vector.reshape(dim, dim) @ r_jax.conj().T).reshape(-1),
                lambda vector: (q_jax.conj().T @ vector.reshape(dim, dim)
                                + vector.reshape(dim, dim) @ q_jax
                                + r_jax.conj().T @ vector.reshape(dim, dim) @ r_jax).reshape(-1),
                right0,
                left0,
                iterations=args.iterations,
                shift=args.shift,
            )

        compiled = jax.jit(solve)
        start = time.perf_counter()
        jax_value, _, jax_right = compiled()
        jax_value.block_until_ready()
        jax_compile_time = time.perf_counter() - start
        start = time.perf_counter()
        jax_value, _, jax_right = compiled()
        jax_value.block_until_ready()
        jax_warm_time = time.perf_counter() - start
        jax_value_np = complex(np.asarray(jax_value))
        jax_right_np = np.asarray(jax_right)
        jax_residual = np.linalg.norm(action(jax_right_np) - jax_value_np * jax_right_np)
        jax_residual /= np.linalg.norm(jax_right_np)
        print(
            f"{dim:3d} {size:11d} {scipy_time:8.3f} {jax_compile_time:13.3f} "
            f"{jax_warm_time:9.4f} {abs(jax_value_np - scipy_value):9.2e} "
            f"{jax_residual:11.2e}"
        )
        del scipy_residual


if __name__ == "__main__":
    main()
