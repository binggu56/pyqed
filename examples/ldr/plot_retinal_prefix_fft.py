#!/usr/bin/env python3
"""Plot mixed prefix-FFT dynamics for the Hahn--Stock retinal model."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
from matplotlib.ticker import LogFormatterMathtext
from scipy.sparse.linalg import LinearOperator, expm_multiply
import ultraplot as uplt

from pyqed.ldr import kinetic
from pyqed.units import au2fs

from examples.ldr.retinal_hahn_stock_prefix_fft import build_model


def observables(states, frames, phi, q, cis):
    shape = frames.shape[:2]
    aligned = states.reshape(len(states), *shape, 2)
    diabatic = np.einsum(
        "pqia,tpqa->tpqi", frames, aligned, optimize=True
    )
    density = np.abs(diabatic) ** 2
    nuclear = density.sum(axis=-1)
    return {
        "diabatic": density.sum(axis=(1, 2)),
        "cis": np.einsum("tpq,p->t", nuclear, cis),
        "trans": np.einsum("tpq,p->t", nuclear, ~cis),
        "product": np.einsum("tpq,p->t", density[..., 1], ~cis),
        "cos_phi": np.einsum("tpq,p->t", nuclear, np.cos(phi)),
        "q_mean": np.einsum("tpq,q->t", nuclear, q),
        "norm": density.sum(axis=(1, 2, 3)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nphi", type=int, default=32)
    parser.add_argument("--nq", type=int, default=16)
    parser.add_argument("--tmax-fs", type=float, default=200.0)
    parser.add_argument("--nt", type=int, default=201)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("/private/tmp/retinal_prefix_fft.png"),
    )
    args = parser.parse_args()

    model, phi_dvr, q_dvr, potential, energies, frames, links, operator = (
        build_model(args.nphi, args.nq)
    )
    shape = (args.nphi, args.nq)
    phi_kinetic = phi_dvr.t(mc2=1.0 / model.inverse_inertia)
    q_kinetic = q_dvr.t()
    nuclear = np.kron(phi_kinetic, np.eye(args.nq))
    nuclear += np.kron(np.eye(args.nphi), q_kinetic)
    reference = kinetic.matrix(
        nuclear, shape, 2, links=links, symmetrize=False
    )
    energy_vector = energies.reshape(-1)
    reference += np.diag(energy_vector)

    def matvec(vector):
        vector = np.asarray(vector)
        flat = vector.reshape(-1)
        result = operator.matvec(flat) + energy_vector * flat
        return result.reshape(vector.shape)

    def matmat(vectors):
        return operator.matmat(vectors) + energy_vector[:, None] * vectors

    hamiltonian = LinearOperator(
        reference.shape,
        matvec=matvec,
        rmatvec=matvec,
        matmat=matmat,
        rmatmat=matmat,
        dtype=complex,
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

    times_fs = np.linspace(0.0, args.tmax_fs, args.nt)
    stop = args.tmax_fs / au2fs
    trace = np.trace(reference)
    prefix_states = expm_multiply(
        -1j * hamiltonian,
        initial,
        start=0.0,
        stop=stop,
        num=args.nt,
        endpoint=True,
        traceA=-1j * trace,
    )
    dense_states = expm_multiply(
        -1j * reference,
        initial,
        start=0.0,
        stop=stop,
        num=args.nt,
        endpoint=True,
    )
    prefix = observables(
        prefix_states,
        frames,
        phi_dvr.x,
        q_dvr.x,
        model.cis_mask(phi_dvr.x),
    )
    dense = observables(
        dense_states,
        frames,
        phi_dvr.x,
        q_dvr.x,
        model.cis_mask(phi_dvr.x),
    )
    state_error = np.linalg.norm(prefix_states - dense_states, axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    data_path = args.out.with_suffix(".npz")
    np.savez_compressed(
        data_path,
        times_fs=times_fs,
        prefix_states=prefix_states,
        dense_states=dense_states,
        state_error=state_error,
        **{f"prefix_{key}": value for key, value in prefix.items()},
        **{f"dense_{key}": value for key, value in dense.items()},
    )

    colors = ("#0072B2", "#D55E00", "#009E73")
    fig, axes = uplt.subplots(
        nrows=2, ncols=2, figsize=(7.4, 5.4), share=False
    )
    marker_slice = slice(None, None, max(1, args.nt // 20))

    axes[0].plot(
        times_fs,
        prefix["diabatic"][:, 0],
        color=colors[0],
        label=r"prefix FFT",
    )
    axes[0].plot(
        times_fs[marker_slice],
        dense["diabatic"][marker_slice, 0],
        linestyle="none",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        markeredgecolor=colors[0],
        label="dense LDR",
    )

    for key, label, color in zip(
        ("trans", "product"),
        ("total trans", r"trans $S_1$"),
        (colors[1], colors[2]),
    ):
        axes[1].semilogy(times_fs, prefix[key], color=color, label=label)
        axes[1].semilogy(
            times_fs[marker_slice],
            dense[key][marker_slice],
            linestyle="none",
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            markeredgecolor=color,
        )

    axes[2].plot(
        times_fs, prefix["cos_phi"], color=colors[0], label=r"$\langle\cos\phi\rangle$"
    )
    axes[2].plot(
        times_fs, prefix["q_mean"], color=colors[1], label=r"$\langle q\rangle$"
    )
    axes[3].semilogy(
        times_fs,
        np.maximum(state_error, 1.0e-17),
        color="#CC79A7",
        label=r"$\|\chi_{\rm FFT}-\chi_{\rm dense}\|$",
    )
    axes[3].semilogy(
        times_fs,
        np.maximum(np.abs(prefix["norm"] - 1.0), 1.0e-17),
        color="#666666",
        linestyle="--",
        label="norm error",
    )

    for label, axis in zip("abcd", axes):
        axis.text(
            0.02,
            0.96,
            label,
            transform=axis.transAxes,
            va="top",
            fontweight="bold",
        )
    axes[0].format(ylabel=r"$S_0$ population", ylim=(-0.001, 0.022))
    axes[1].format(ylabel="Trans population", ylim=(1.0e-9, 2.0e-4))
    axes[2].format(xlabel="Time (fs)", ylabel="Coordinate moment")
    axes[3].format(xlabel="Time (fs)", ylabel="Numerical error", ylim=(1.0e-16, 1.0e-10))
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[3].yaxis.set_major_formatter(LogFormatterMathtext())
    for axis in axes:
        axis.format(
            xlim=(0.0, args.tmax_fs),
            grid=False,
            ticklabelsize=8,
            labelsize=9,
        )
        axis.legend(loc="best", frame=False, fontsize=7.5, ncols=1)
    fig.format(suptitle=rf"Hahn--Stock retinal, ${args.nphi}\times{args.nq}$ validation grid")
    fig.savefig(args.out, dpi=350, bbox_inches="tight")
    fig.savefig(args.out.with_suffix(".pdf"), bbox_inches="tight")
    print(f"saved {args.out}")
    print(f"saved {args.out.with_suffix('.pdf')}")
    print(f"saved {data_path}")
    print(f"maximum state error {state_error.max():.3e}")
    print(f"maximum norm error {np.max(np.abs(prefix['norm'] - 1.0)):.3e}")


if __name__ == "__main__":
    main()
