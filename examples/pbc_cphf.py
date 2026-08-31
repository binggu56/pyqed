#!/usr/bin/env python3
"""Validate q-resolved periodic CPHF against finite-field block KRHF."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eigh

from pyqed.qchem.pbc import Cell


def build_reference():
    cell = Cell(
        atom="H 2.3 3.0 3.0; H 3.7 3.0 3.0",
        a=np.diag([6.0, 6.4, 6.8]),
        basis="sto-3g",
        unit="bohr",
        dimension=3,
        spin=0,
        integral_options={"eri_representation": "direct"},
    ).build()
    mean_field = cell.KRHF(
        nk=(3, 1, 1),
        eta=0.7,
        real_cut=0,
        pair_cut=0,
        recip_cut=2,
        one_body_nuclear_cut=1,
        jk_builder="gdf",
        eri_screen_tol=0.0,
        pair_ft_screen_tol=0.0,
        one_body_screen_tol=0.0,
    )
    mean_field.density_fit(
        auxbasis="sto-3g",
        reciprocal_kernel="full",
        recip_cut=2,
        pair_cut=0,
        pair_screen_tol=0.0,
        metric_tol=1.0e-12,
    )
    return mean_field.run(
        max_cycle=80,
        conv_tol=1.0e-12,
        conv_tol_dm=1.0e-10,
    )


def finite_q_field_density(mean_field, perturbations, q_index, strength):
    nkpts = mean_field.nkpts
    nao = mean_field.cell.nao
    pair_by_k = {
        int(k): int(kq)
        for k, kq in mean_field.with_df.pair_keys(q_index)
    }
    diagonal = [np.array(density, copy=True) for density in mean_field.dm]
    density_q = [
        np.zeros((nao, nao), dtype=np.complex128) for _ in range(nkpts)
    ]
    overlap = np.zeros((nkpts * nao, nkpts * nao), dtype=np.complex128)
    for k_index, block in enumerate(mean_field._overlap_k):
        rows = slice(k_index * nao, (k_index + 1) * nao)
        overlap[rows, rows] = block

    for _cycle in range(300):
        diagonal_fock = mean_field._build_fock_k(diagonal)
        vj_q, vk_q = mean_field.with_df.get_jk_response(density_q, q_index)
        q_fock = []
        for k_index, kq_index in pair_by_k.items():
            block = (
                float(strength) * perturbations[k_index]
                + vj_q[k_index]
                - 0.5 * vk_q[k_index]
            )
            if mean_field.madelung is not None:
                block -= 0.5 * mean_field.madelung * (
                    mean_field._overlap_k[kq_index]
                    @ density_q[k_index]
                    @ mean_field._overlap_k[k_index]
                )
            q_fock.append(block)

        fock = np.zeros_like(overlap)
        for k_index, block in enumerate(diagonal_fock):
            rows = slice(k_index * nao, (k_index + 1) * nao)
            fock[rows, rows] = block
        for k_index, kq_index in pair_by_k.items():
            rows = slice(kq_index * nao, (kq_index + 1) * nao)
            columns = slice(k_index * nao, (k_index + 1) * nao)
            fock[rows, columns] = q_fock[k_index]
            fock[columns, rows] = q_fock[k_index].conj().T
        fock = 0.5 * (fock + fock.conj().T)

        _energy, coefficients = eigh(fock, overlap)
        electron_pairs = mean_field.cell.nelectron * nkpts // 2
        occupied = coefficients[:, :electron_pairs]
        density = 2.0 * occupied @ occupied.conj().T
        diagonal_new = []
        density_q_new = []
        for k_index, kq_index in pair_by_k.items():
            columns = slice(k_index * nao, (k_index + 1) * nao)
            rows_k = slice(k_index * nao, (k_index + 1) * nao)
            rows_kq = slice(kq_index * nao, (kq_index + 1) * nao)
            diagonal_new.append(density[rows_k, columns])
            density_q_new.append(density[rows_kq, columns])
        residual = max(
            max(
                np.linalg.norm(new - old)
                for new, old in zip(diagonal_new, diagonal)
            ),
            max(
                np.linalg.norm(new - old)
                for new, old in zip(density_q_new, density_q)
            ),
        )
        diagonal = [
            0.35 * old + 0.65 * new
            for old, new in zip(diagonal, diagonal_new)
        ]
        density_q = [
            0.35 * old + 0.65 * new
            for old, new in zip(density_q, density_q_new)
        ]
        if residual < 1.0e-12:
            return density_q
    raise RuntimeError("Finite-q block KRHF did not converge.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/private/tmp/pbc_cphf_validation.pdf"),
    )
    args = parser.parse_args()

    mean_field = build_reference()
    if not mean_field.converged:
        raise RuntimeError("Reference KRHF did not converge.")
    q_index = 1
    qpoint = mean_field.with_df.qpts[q_index]
    perturbations = [
        np.asarray(
            [[0.12, 0.03 + 0.01j], [-0.02 + 0.04j, -0.04]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[-0.05 + 0.02j, 0.07], [0.01 - 0.03j, 0.09]],
            dtype=np.complex128,
        ),
        np.asarray(
            [[0.03, -0.04 + 0.02j], [0.06 + 0.01j, -0.08 - 0.01j]],
            dtype=np.complex128,
        ),
    ]
    response = mean_field.response().kernel(
        perturbations,
        qpoint=qpoint,
        tol=1.0e-11,
    )
    analytic = np.asarray([block[0] for block in response.dm1])

    fields = np.asarray(
        [2.0e-2, 1.0e-2, 5.0e-3, 2.0e-3, 1.0e-3, 5.0e-4, 2.0e-4]
    )
    numerical = []
    errors = []
    for field in fields:
        plus = finite_q_field_density(
            mean_field, perturbations, q_index, field
        )
        minus = finite_q_field_density(
            mean_field, perturbations, q_index, -field
        )
        derivative = np.asarray(
            [
                (plus_block - minus_block) / (2.0 * field)
                for plus_block, minus_block in zip(plus, minus)
            ]
        )
        numerical.append(derivative)
        errors.append(float(np.max(np.abs(derivative - analytic))))
    numerical = np.asarray(numerical)
    errors = np.asarray(errors)

    fig, axes = plt.subplots(1, 4, figsize=(10.0, 2.65))
    color_limit = float(np.max(np.abs(analytic)))
    for k_index, axis in enumerate(axes[:3]):
        image = axis.imshow(
            np.abs(analytic[k_index]),
            cmap="viridis",
            aspect="equal",
            vmin=0.0,
            vmax=color_limit,
        )
        target = response.kq_indices[k_index]
        axis.set_title(rf"$|P_q(k_{target},k_{k_index})|$")
        axis.set_xlabel("AO column")
        if k_index == 0:
            axis.set_ylabel("AO row")
        axis.set_xticks(range(mean_field.cell.nao))
        axis.set_yticks(range(mean_field.cell.nao))
    fig.colorbar(image, ax=axes[:3], fraction=0.025, pad=0.02)

    axes[3].loglog(fields, errors, "o-", color="#28666E", linewidth=1.4)
    reference = errors[0] * (fields / fields[0]) ** 2
    axes[3].loglog(
        fields,
        reference,
        "--",
        color="#B24C63",
        label=r"$O(\lambda^2)$",
    )
    axes[3].set_xlabel(r"Field $\lambda$ (Ha)")
    axes[3].set_ylabel(r"$\max|P^{(1)}_{\rm FD}-P^{(1)}_{\rm CPHF}|$")
    axes[3].set_title(r"Coupled $\pm q$ validation")
    axes[3].legend(frameon=False)
    axes[3].grid(True, which="both", color="#E0E0E0", linewidth=0.6)
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.subplots_adjust(left=0.07, right=0.98, bottom=0.2, top=0.84, wspace=0.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output)
    png = args.output.with_suffix(".png")
    fig.savefig(png, dpi=300)
    data = args.output.with_suffix(".npz")
    np.savez(
        data,
        kpoints=mean_field.kpts,
        qpoint=qpoint,
        fields=fields,
        finite_field_density_response=numerical,
        cphf_density_response=analytic,
        max_abs_errors=errors,
        cphf_residual=response.residual_norm,
    )
    print(f"q point: {qpoint}")
    print(f"CPHF residual: {response.residual_norm:.3e}")
    print(f"smallest-field max error: {errors[-1]:.3e}")
    print(f"wrote {args.output}")
    print(f"wrote {png}")
    print(f"wrote {data}")


if __name__ == "__main__":
    main()
