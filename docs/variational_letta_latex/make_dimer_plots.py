#!/usr/bin/env python3
"""Generate controlled dimer benchmark plots for variational LETTA."""

from __future__ import annotations

from pathlib import Path

from matplotlib.lines import Line2D
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
import ultraplot as uplt


HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)


def kron_all(operators):
    out = sp.csr_matrix([[1.0]])
    for operator in operators:
        out = sp.kron(out, operator, format="csr")
    return out


def bond_operator(nsites, site, left_op, right_op):
    identity = sp.eye(2, format="csr")
    terms = []
    for index in range(nsites):
        if index == site:
            terms.append(left_op)
        elif index == site + 1:
            terms.append(right_op)
        else:
            terms.append(identity)
    return kron_all(terms)


def alternating_heisenberg(nsites, weak_coupling):
    sx = 0.5 * sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
    sy = 0.5 * sp.csr_matrix(np.array([[0.0, -1.0j], [1.0j, 0.0]]))
    sz = 0.5 * sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))
    hamiltonian = sp.csr_matrix((2**nsites, 2**nsites), dtype=complex)
    for site in range(nsites - 1):
        coupling = 1.0 if site % 2 == 0 else weak_coupling
        for op in (sx, sy, sz):
            hamiltonian += coupling * bond_operator(nsites, site, op, op)
    return hamiltonian


def exact_ground_energy(nsites, weak_coupling):
    if abs(weak_coupling) < 1.0e-14:
        return -0.75 * (nsites // 2)
    hamiltonian = alternating_heisenberg(nsites, weak_coupling)
    return float(sla.eigsh(hamiltonian, k=1, which="SA", return_eigenvectors=False, tol=1.0e-11)[0].real)


def main() -> None:
    nsites = 12
    weak_couplings = np.array([0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    ndimers = nsites // 2
    exact = np.array([exact_ground_energy(nsites, value) for value in weak_couplings])
    letta_dimer = np.full_like(weak_couplings, -0.75 * ndimers, dtype=float)
    product = -0.25 * (ndimers + weak_couplings * (ndimers - 1))

    uplt.rc.update(
        {
            "font.size": 10.0,
            "axes.labelsize": 10.8,
            "axes.titlesize": 10.8,
            "legend.fontsize": 9.2,
            "xtick.labelsize": 9.4,
            "ytick.labelsize": 9.4,
            "lines.linewidth": 1.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 400,
            "savefig.facecolor": "white",
        }
    )

    fig, ax = uplt.subplots(refwidth=4.7, refheight=2.75)
    ax.plot(
        weak_couplings,
        letta_dimer - exact,
        marker="o",
        markersize=5.2,
        color="#2d9465",
        label=r"LETTA $D=1$ dimer state",
    )
    ax.plot(
        weak_couplings,
        product - exact,
        marker="s",
        markersize=5.0,
        markerfacecolor="white",
        markeredgewidth=1.2,
        linestyle="--",
        color="#c84d36",
        label=r"best product / MPS $D=1$",
    )
    ax.format(
        xlabel=r"weak-bond coupling $\lambda$",
        ylabel=r"energy error above exact",
        title=r"Controlled alternating-chain benchmark ($L=12$)",
        xlim=(-0.03, 1.03),
        ylim=(-0.03, 3.2),
        xticks=[0, 0.25, 0.5, 0.75, 1.0],
        grid=True,
    )
    fig.format(gridcolor="#d9d9d9", gridlinewidth=0.55, tickminor=False)
    fig.legend(
        handles=[
            Line2D([0], [0], color="#2d9465", marker="o", linewidth=1.6, markersize=5.2, label=r"LETTA $D=1$ dimer state"),
            Line2D(
                [0],
                [0],
                color="#c84d36",
                marker="s",
                markerfacecolor="white",
                markeredgewidth=1.2,
                linestyle="--",
                linewidth=1.6,
                markersize=5.0,
                label=r"best product / MPS $D=1$",
            ),
        ],
        loc="bottom",
        ncols=2,
        frame=False,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"dimer_letta_separation.{suffix}", bbox_inches="tight")
    uplt.close(fig)


if __name__ == "__main__":
    main()
