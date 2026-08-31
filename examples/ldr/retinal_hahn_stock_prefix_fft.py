#!/usr/bin/env python3
"""Mixed prefix-FFT LDR test for the two-mode Hahn--Stock retinal model."""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from scipy.sparse.linalg import LinearOperator, expm_multiply

from pyqed.dvr import ExponentialDVR, HermiteDVR
from pyqed.ldr import kinetic, overlap
from pyqed.models.retinal import RetinalHahnStock
from pyqed.units import au2fs


def build_model(nphi, nq):
    model = RetinalHahnStock()
    phi_dvr = ExponentialDVR(
        npts=nphi,
        L=2.0 * np.pi,
        x0=0.5 * np.pi,
    )
    q_dvr = HermiteDVR(
        npts=nq,
        mass=1.0 / model.omega,
        omega=model.omega,
    )
    phi = phi_dvr.x
    q = q_dvr.x
    potential = model.diabatic_potential(phi[:, None], q[None, :])
    energies, frames = np.linalg.eigh(potential)
    shape = (nphi, nq)
    links = overlap.nearest(
        shape,
        lambda left, right: frames[left].conj().T @ frames[right],
    )
    phi_descriptor = phi_dvr.kinetic_toeplitz(
        mc2=1.0 / model.inverse_inertia
    )
    q_kinetic = q_dvr.t()
    operator = kinetic.PrefixFFTND(
        (phi_descriptor, q_kinetic), shape, links
    )
    return model, phi_dvr, q_dvr, potential, energies, frames, links, operator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--nq", type=int, default=16)
    parser.add_argument("--time-fs", type=float, default=0.2)
    args = parser.parse_args()
    if args.nphi < 3 or args.nq < 2 or args.time_fs < 0.0:
        raise ValueError("invalid grid or propagation time")

    (
        model,
        phi_dvr,
        q_dvr,
        potential,
        energies,
        frames,
        links,
        operator,
    ) = build_model(args.nphi, args.nq)
    shape = (args.nphi, args.nq)
    phi_kinetic = phi_dvr.t(mc2=1.0 / model.inverse_inertia)
    q_kinetic = q_dvr.t()
    nuclear = np.kron(phi_kinetic, np.eye(args.nq))
    nuclear += np.kron(np.eye(args.nphi), q_kinetic)
    reference_kinetic = kinetic.matrix(
        nuclear, shape, 2, links=links, symmetrize=False
    )
    energy_vector = energies.reshape(-1)
    reference = reference_kinetic + np.diag(energy_vector)

    def action(vector):
        vector = np.asarray(vector)
        result = operator.matvec(vector.reshape(-1)) + energy_vector * vector.reshape(-1)
        return result.reshape(vector.shape)

    def action_matrix(vectors):
        vectors = np.asarray(vectors)
        return operator.matmat(vectors) + energy_vector[:, None] * vectors

    hamiltonian = LinearOperator(
        reference.shape,
        matvec=action,
        rmatvec=action,
        matmat=action_matrix,
        rmatmat=action_matrix,
        dtype=complex,
    )
    rng = np.random.default_rng(23)
    probe = rng.normal(size=reference.shape[0]) + 1j * rng.normal(
        size=reference.shape[0]
    )
    reference_action = reference @ probe
    action_error = np.linalg.norm(action(probe) - reference_action) / np.linalg.norm(
        reference_action
    )

    v_phi = 0.5 * model.w0 * (1.0 - np.cos(phi_dvr.x))
    v_q = 0.5 * model.omega * q_dvr.x**2
    _, phi_states = np.linalg.eigh(phi_kinetic + np.diag(v_phi))
    _, q_states = np.linalg.eigh(q_kinetic + np.diag(v_q))
    diabatic = np.zeros((*shape, 2), dtype=complex)
    diabatic[..., 1] = np.outer(phi_states[:, 0], q_states[:, 0])
    initial = np.einsum(
        "...ia,...i->...a", frames.conj(), diabatic, optimize=True
    ).reshape(-1)
    propagation_time = args.time_fs / au2fs
    trace = np.trace(reference)
    fft_final = expm_multiply(
        -1j * propagation_time * hamiltonian,
        initial,
        traceA=-1j * propagation_time * trace,
    )
    dense_final = expm_multiply(-1j * propagation_time * reference, initial)
    propagation_error = np.linalg.norm(fft_final - dense_final)

    repetitions = 30
    action(probe)
    reference @ probe
    start = time.perf_counter()
    for _ in range(repetitions):
        action(probe)
    fft_time = (time.perf_counter() - start) / repetitions
    start = time.perf_counter()
    for _ in range(repetitions):
        reference @ probe
    dense_time = (time.perf_counter() - start) / repetitions

    print(
        json.dumps(
            {
                "model": "Hahn-Stock retinal",
                "shape": shape,
                "dimension": reference.shape[0],
                "axis_backends": [
                    info["backend"] for info in operator.info["axes"]
                ],
                "action_error": float(action_error),
                "propagation_time_fs": args.time_fs,
                "propagation_error": float(propagation_error),
                "norm_error": float(abs(np.vdot(fft_final, fft_final).real - 1.0)),
                "fft_action_seconds": fft_time,
                "dense_action_seconds": dense_time,
                "speedup": dense_time / fft_time,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
