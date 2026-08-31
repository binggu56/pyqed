#!/usr/bin/env python3
"""Fit the H3+ S1/S2 conical pair in an anchor-Procrustes gauge.

The electronic calculation is full CASCI(2e,3o)/STO-3G in the six-dimensional
singlet space.  Only the isolated S1/S2 E' pair is fitted.  Direct overlaps to
one anchor define a smooth rank-two frame, while nearest-neighbor overlaps are
kept nonunitary for subsequent LDR calculations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from examples.ldr.h3plus_rectilinear_cgldr import h3plus_geometry
from pyqed.units import au2ev
from pyqed.ldr.overlap import procrustes
from pyqed.mps.functional import FunctionalTT
from pyqed.qchem import Molecule
from pyqed.qchem.mcscf.casci import CASCI, overlap


HARTREE_TO_EV = au2ev
TARGET_STATES = (1, 2)


def electronic_point(qx, qy, *, basis="sto-3g", nroots=6):
    geometry = h3plus_geometry({"Qs": 0.0, "Qx": float(qx), "Qy": float(qy)})
    molecule = Molecule(
        atom=[["H", *xyz] for xyz in geometry],
        unit="bohr",
        basis=basis,
        charge=1,
        spin=0,
    )
    molecule.build()
    mean_field = molecule.RHF().run()
    return CASCI(mean_field, ncas=3, nelecas=2).run(nstates=int(nroots))


def generate_cache(
    qx,
    qy,
    *,
    basis="sto-3g",
    nroots=6,
    reference_coordinates=None,
):
    qx = np.asarray(qx, dtype=float)
    qy = np.asarray(qy, dtype=float)
    points = np.empty((len(qx), len(qy)), dtype=object)
    energies = np.empty((*points.shape, int(nroots)))
    total = points.size
    completed = 0
    for i, x in enumerate(qx):
        for j, y in enumerate(qy):
            point = electronic_point(x, y, basis=basis, nroots=nroots)
            points[i, j] = point
            energies[i, j] = np.asarray(point.e_tot, dtype=float)
            completed += 1
            if completed == 1 or completed % max(1, total // 10) == 0:
                print(f"[electronic] {completed}/{total}", flush=True)

    anchor = (int(np.argmin(np.abs(qx))), int(np.argmin(np.abs(qy))))
    if reference_coordinates is None:
        reference_point = points[anchor]
        anchor_coordinates = (qx[anchor[0]], qy[anchor[1]])
        stored_anchor = anchor
    else:
        anchor_coordinates = tuple(map(float, reference_coordinates))
        reference_point = electronic_point(
            *anchor_coordinates, basis=basis, nroots=nroots
        )
        stored_anchor = (-1, -1)
    reference = np.empty((*points.shape, nroots, nroots), dtype=complex)
    for index in np.ndindex(points.shape):
        reference[index] = np.asarray(
            overlap(points[index], reference_point), dtype=complex
        )

    links_x = np.empty((len(qx) - 1, len(qy), nroots, nroots), dtype=complex)
    links_y = np.empty((len(qx), len(qy) - 1, nroots, nroots), dtype=complex)
    for i in range(len(qx) - 1):
        for j in range(len(qy)):
            links_x[i, j] = overlap(points[i, j], points[i + 1, j])
    for i in range(len(qx)):
        for j in range(len(qy) - 1):
            links_y[i, j] = overlap(points[i, j], points[i, j + 1])
    return {
        "qx": qx,
        "qy": qy,
        "energies": energies,
        "reference_links": reference,
        "links_x": links_x,
        "links_y": links_y,
        "anchor": np.asarray(stored_anchor, dtype=int),
        "anchor_coordinates": np.asarray(anchor_coordinates, dtype=float),
        "basis": np.asarray(basis),
        "nroots": np.asarray(nroots),
        "target_states": np.asarray(TARGET_STATES),
        "coordinate_frame": np.asarray("H3+ rectilinear E-prime modes; Qs=0"),
    }


def save_cache(filename, data):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filename, **data)


def load_cache(filename):
    with np.load(filename) as archive:
        data = {key: archive[key] for key in archive.files}
    if tuple(np.asarray(data["target_states"], dtype=int)) != TARGET_STATES:
        raise ValueError("cache does not contain the H3+ S1/S2 target pair")
    return data


def selected_blocks(values, states=TARGET_STATES):
    states = np.asarray(states, dtype=int)
    return np.asarray(values)[..., states[:, None], states]


def anchor_procrustes_fields(data, *, energy_shift=None):
    """Return anchor-aligned Hamiltonians, links, gauges, and positive residuals."""
    energies = np.asarray(data["energies"], dtype=float)
    reference = selected_blocks(data["reference_links"])
    gauges, positive, reference_singular = procrustes(reference)
    if energy_shift is None:
        anchor = tuple(np.asarray(data["anchor"], dtype=int))
        if any(index < 0 for index in anchor):
            raise ValueError("an external-anchor cache requires an energy shift")
        shift = float(np.mean(energies[anchor][list(TARGET_STATES)]))
    else:
        shift = float(energy_shift)
    target_energy = energies[..., list(TARGET_STATES)] - shift
    hamiltonian = np.einsum(
        "...ia,...i,...ib->...ab",
        gauges.conj(),
        target_energy,
        gauges,
        optimize=True,
    )
    hamiltonian = 0.5 * (
        hamiltonian + hamiltonian.swapaxes(-1, -2).conj()
    )

    raw_x = selected_blocks(data["links_x"])
    raw_y = selected_blocks(data["links_y"])
    links_x = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauges[:-1].conj(), raw_x, gauges[1:], optimize=True,
    )
    links_y = np.einsum(
        "...ia,...ij,...jb->...ab",
        gauges[:, :-1].conj(), raw_y, gauges[:, 1:], optimize=True,
    )
    return {
        "hamiltonian": hamiltonian,
        "links_x": links_x,
        "links_y": links_y,
        "gauges": gauges,
        "reference_positive": positive,
        "reference_singular": reference_singular,
        "energy_shift": shift,
    }


def coordinate_tables(qx, qy):
    x, y = np.meshgrid(qx, qy, indexing="ij")
    points = np.stack((x.reshape(-1), y.reshape(-1)), axis=1)
    xmid = 0.5 * (qx[:-1] + qx[1:])
    xm, xy = np.meshgrid(xmid, qy, indexing="ij")
    links_x = np.stack((xm.reshape(-1), xy.reshape(-1)), axis=1)
    ymid = 0.5 * (qy[:-1] + qy[1:])
    yx, ym = np.meshgrid(qx, ymid, indexing="ij")
    links_y = np.stack((yx.reshape(-1), ym.reshape(-1)), axis=1)
    return points, links_x, links_y


def interior_holdout(shape, phase=0):
    indices = np.indices(shape)
    interior = np.ones(shape, dtype=bool)
    for axis, size in enumerate(shape):
        if size > 2:
            interior &= (indices[axis] > 0) & (indices[axis] < size - 1)
    return interior & ((indices.sum(axis=0) + int(phase)) % 4 == 0)


def fit_matrix_field(
    coordinates,
    values,
    heldout,
    *,
    bounds,
    degree,
    rank,
    seed,
    hermitian,
):
    values = np.asarray(values)
    heldout = np.asarray(heldout, dtype=bool).reshape(-1)
    train = ~heldout
    model = FunctionalTT(
        degrees=int(degree),
        rank=int(rank),
        bounds=tuple(tuple(map(float, bound)) for bound in bounds),
        normalization="frobenius",
        hermitian=bool(hermitian),
        regularization=1.0e-10,
        sweeps=40,
        rtol=1.0e-11,
        random_state=int(seed),
    ).fit(
        np.asarray(coordinates)[train],
        values.reshape(len(coordinates), *values.shape[-2:])[train],
        validation=(
            np.asarray(coordinates)[heldout],
            values.reshape(len(coordinates), *values.shape[-2:])[heldout],
        ),
    )
    return model, train, heldout


def error_metrics(predicted, reference, mask, *, scale=1.0):
    mask = np.asarray(mask, dtype=bool).reshape(-1)
    difference = (np.asarray(predicted) - np.asarray(reference)).reshape(
        len(mask), -1
    )[mask] * scale
    target = np.asarray(reference).reshape(len(mask), -1)[mask] * scale
    return {
        "rms": float(np.sqrt(np.mean(np.abs(difference) ** 2))),
        "max_abs": float(np.max(np.abs(difference))),
        "relative_frobenius": float(
            np.linalg.norm(difference)
            / max(np.linalg.norm(target), np.finfo(float).tiny)
        ),
    }


def plaquette_defects(links_x, links_y):
    ux = procrustes(links_x)[0]
    uy = procrustes(links_y)[0]
    defects = np.empty((links_x.shape[0], links_y.shape[1]))
    phases = np.empty((*defects.shape, links_x.shape[-1]))
    identity = np.eye(links_x.shape[-1])
    for i, j in np.ndindex(defects.shape):
        loop = ux[i, j] @ uy[i + 1, j] @ ux[i, j + 1].conj().T @ uy[i, j].conj().T
        defects[i, j] = np.linalg.norm(loop - identity)
        phases[i, j] = np.sort(np.angle(np.linalg.eigvals(loop)))
    return defects, phases


def plot_results(data, fields, fitted, output, metrics, *, validation=None):
    qx = np.asarray(data["qx"], dtype=float)
    qy = np.asarray(data["qy"], dtype=float)
    energies = np.asarray(data["energies"], dtype=float)
    diagnostic_data = data
    diagnostic_fields = fields
    diagnostic_fitted = fitted
    if validation is not None:
        diagnostic_data, diagnostic_fields, diagnostic_fitted = validation
    dx = np.asarray(diagnostic_data["qx"], dtype=float)
    dy = np.asarray(diagnostic_data["qy"], dtype=float)
    hamiltonian = diagnostic_fields["hamiltonian"]
    predicted_h = diagnostic_fitted["hamiltonian"]
    links_x = diagnostic_fields["links_x"]
    links_y = diagnostic_fields["links_y"]
    predicted_x = diagnostic_fitted["links_x"]
    predicted_y = diagnostic_fitted["links_y"]
    gap = (energies[..., 2] - energies[..., 1]) * HARTREE_TO_EV
    diagonal_error = np.max(
        np.abs(np.diagonal(predicted_h - hamiltonian, axis1=-2, axis2=-1)), axis=-1
    ) * HARTREE_TO_EV * 1.0e3
    offdiagonal_error = np.abs(predicted_h[..., 0, 1] - hamiltonian[..., 0, 1])
    offdiagonal_error *= HARTREE_TO_EV * 1.0e3
    link_error_x = np.linalg.norm(predicted_x - links_x, axis=(-2, -1))
    link_error_y = np.linalg.norm(predicted_y - links_y, axis=(-2, -1))
    xmid = 0.5 * (dx[:-1] + dx[1:])
    ymid = 0.5 * (dy[:-1] + dy[1:])
    defects, _ = plaquette_defects(links_x, links_y)

    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.1), constrained_layout=True)
    images = []
    images.append(axes[0, 0].pcolormesh(qx, qy, gap.T, shading="auto", cmap="magma"))
    axes[0, 0].contour(qx, qy, gap.T, levels=8, colors="white", linewidths=0.45, alpha=0.65)
    axes[0, 0].set_title(r"Adiabatic gap $E_{S_2}-E_{S_1}$ (eV)")
    images.append(axes[0, 1].pcolormesh(dx, dy, diagonal_error.T, shading="auto", cmap="viridis"))
    axes[0, 1].set_title(r"off-grid max diagonal $\bar E$ error (meV)")
    images.append(axes[0, 2].pcolormesh(dx, dy, offdiagonal_error.T, shading="auto", cmap="viridis"))
    axes[0, 2].set_title(r"off-grid off-diagonal $\bar E$ error (meV)")
    images.append(axes[1, 0].pcolormesh(xmid, dy, link_error_x.T, shading="auto", cmap="cividis"))
    axes[1, 0].set_title(r"$Q_x$-link fit error $\|\Delta L_x\|_F$")
    images.append(axes[1, 1].pcolormesh(dx, ymid, link_error_y.T, shading="auto", cmap="cividis"))
    axes[1, 1].set_title(r"$Q_y$-link fit error $\|\Delta L_y\|_F$")
    images.append(axes[1, 2].pcolormesh(xmid, ymid, defects.T, shading="auto", cmap="plasma"))
    axes[1, 2].set_title(r"plaquette unitary holonomy $\|W-I\|_F$")
    for axis, image in zip(axes.flat, images):
        axis.set_xlabel(r"$Q_x$ (bohr)")
        axis.set_ylabel(r"$Q_y$ (bohr)")
        axis.set_aspect("equal", adjustable="box")
        figure.colorbar(image, ax=axis, shrink=0.83)
    anchor = tuple(np.asarray(data["anchor"], dtype=int))
    axes[0, 0].scatter(qx[anchor[0]], qy[anchor[1]], marker="x", s=55, color="#56B4E9")
    held_e = metrics.get("energy_offgrid", metrics["energy_held"])
    held_l = max(
        metrics.get("link_x_offgrid", metrics["link_x_held"])["max_abs"],
        metrics.get("link_y_offgrid", metrics["link_y_held"])["max_abs"],
    )
    figure.suptitle(
        r"H$_3^+$ CASCI(2e,3o): anchor-Procrustes S$_1$/S$_2$ fit"
        + f"   off-grid max |dE|={held_e['max_abs'] * 1e3:.2f} meV, "
        + f"max |dL|={held_l:.2e}"
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=320)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def run_fit(data, *, degree=6, rank=8, seed=23):
    qx = np.asarray(data["qx"], dtype=float)
    qy = np.asarray(data["qy"], dtype=float)
    fields = anchor_procrustes_fields(data)
    point_coordinates, x_coordinates, y_coordinates = coordinate_tables(qx, qy)
    bounds = ((qx[0], qx[-1]), (qy[0], qy[-1]))
    held_h = interior_holdout(fields["hamiltonian"].shape[:2], phase=0)
    held_x = interior_holdout(fields["links_x"].shape[:2], phase=1)
    held_y = interior_holdout(fields["links_y"].shape[:2], phase=2)
    energy_model, train_h, test_h = fit_matrix_field(
        point_coordinates, fields["hamiltonian"], held_h,
        bounds=bounds, degree=degree, rank=rank, seed=seed, hermitian=True,
    )
    x_model, train_x, test_x = fit_matrix_field(
        x_coordinates, fields["links_x"], held_x,
        bounds=bounds, degree=degree, rank=rank, seed=seed + 1, hermitian=False,
    )
    y_model, train_y, test_y = fit_matrix_field(
        y_coordinates, fields["links_y"], held_y,
        bounds=bounds, degree=degree, rank=rank, seed=seed + 2, hermitian=False,
    )
    predicted_h = energy_model.predict(point_coordinates).reshape(fields["hamiltonian"].shape)
    predicted_x = x_model.predict(x_coordinates).reshape(fields["links_x"].shape)
    predicted_y = y_model.predict(y_coordinates).reshape(fields["links_y"].shape)

    energy_reference = fields["hamiltonian"].reshape(-1, 2, 2)
    link_x_reference = fields["links_x"].reshape(-1, 2, 2)
    link_y_reference = fields["links_y"].reshape(-1, 2, 2)
    target_singular = np.concatenate([
        np.linalg.svd(selected_blocks(data["links_x"]), compute_uv=False).reshape(-1),
        np.linalg.svd(selected_blocks(data["links_y"]), compute_uv=False).reshape(-1),
    ])
    full_singular = np.concatenate([
        np.linalg.svd(data["links_x"], compute_uv=False).reshape(-1),
        np.linalg.svd(data["links_y"], compute_uv=False).reshape(-1),
    ])
    energies = np.asarray(data["energies"], dtype=float)
    defects, phases = plaquette_defects(fields["links_x"], fields["links_y"])
    metrics = {
        "energy_train": error_metrics(
            predicted_h.reshape(-1, 2, 2), energy_reference, train_h,
            scale=HARTREE_TO_EV,
        ),
        "energy_held": error_metrics(
            predicted_h.reshape(-1, 2, 2), energy_reference, test_h,
            scale=HARTREE_TO_EV,
        ),
        "link_x_train": error_metrics(
            predicted_x.reshape(-1, 2, 2), link_x_reference, train_x,
        ),
        "link_x_held": error_metrics(
            predicted_x.reshape(-1, 2, 2), link_x_reference, test_x,
        ),
        "link_y_train": error_metrics(
            predicted_y.reshape(-1, 2, 2), link_y_reference, train_y,
        ),
        "link_y_held": error_metrics(
            predicted_y.reshape(-1, 2, 2), link_y_reference, test_y,
        ),
        "minimum_target_neighbor_singular_value": float(np.min(target_singular)),
        "minimum_full_neighbor_singular_value": float(np.min(full_singular)),
        "minimum_reference_singular_value": float(np.min(fields["reference_singular"])),
        "minimum_gap_to_S0_eV": float(np.min(energies[..., 1] - energies[..., 0]) * HARTREE_TO_EV),
        "minimum_gap_to_S3_eV": float(np.min(energies[..., 3] - energies[..., 2]) * HARTREE_TO_EV),
        "maximum_plaquette_holonomy_defect": float(np.max(defects)),
        "maximum_plaquette_eigenphase": float(np.max(np.abs(phases))),
    }
    return (
        fields,
        {"hamiltonian": predicted_h, "links_x": predicted_x, "links_y": predicted_y},
        metrics,
        (energy_model, x_model, y_model),
        {"energy": test_h, "links_x": test_x, "links_y": test_y},
    )


def evaluate_models(data, fields, models):
    qx = np.asarray(data["qx"], dtype=float)
    qy = np.asarray(data["qy"], dtype=float)
    point_coordinates, x_coordinates, y_coordinates = coordinate_tables(qx, qy)
    energy_model, x_model, y_model = models
    return {
        "hamiltonian": energy_model.predict(point_coordinates).reshape(
            fields["hamiltonian"].shape
        ),
        "links_x": x_model.predict(x_coordinates).reshape(fields["links_x"].shape),
        "links_y": y_model.predict(y_coordinates).reshape(fields["links_y"].shape),
    }


def append_offgrid_metrics(metrics, fields, fitted):
    all_energy = np.ones(fields["hamiltonian"].shape[:2], dtype=bool)
    all_x = np.ones(fields["links_x"].shape[:2], dtype=bool)
    all_y = np.ones(fields["links_y"].shape[:2], dtype=bool)
    metrics["energy_offgrid"] = error_metrics(
        fitted["hamiltonian"].reshape(-1, 2, 2),
        fields["hamiltonian"].reshape(-1, 2, 2),
        all_energy,
        scale=HARTREE_TO_EV,
    )
    metrics["link_x_offgrid"] = error_metrics(
        fitted["links_x"].reshape(-1, 2, 2),
        fields["links_x"].reshape(-1, 2, 2),
        all_x,
    )
    metrics["link_y_offgrid"] = error_metrics(
        fitted["links_y"].reshape(-1, 2, 2),
        fields["links_y"].reshape(-1, 2, 2),
        all_y,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--npts", type=int, default=9)
    parser.add_argument("--qmin", type=float, default=-0.12)
    parser.add_argument("--qmax", type=float, default=0.12)
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--degree", type=int, default=6)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--cache", type=Path,
        default=Path("/private/tmp/h3plus_casci_s1s2_qxqy_9x9.npz"),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--validation-cache", type=Path,
        default=Path("/private/tmp/h3plus_casci_s1s2_qxqy_offgrid_8x8.npz"),
    )
    parser.add_argument("--skip-offgrid", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/private/tmp/h3plus_casci_positive_link_fit"),
    )
    args = parser.parse_args()
    if args.npts < 5:
        raise ValueError("npts must be at least five for a held-out fit")
    requested = np.linspace(args.qmin, args.qmax, args.npts)
    if args.cache.exists() and not args.force:
        data = load_cache(args.cache)
        np.testing.assert_allclose(data["qx"], requested)
        np.testing.assert_allclose(data["qy"], requested)
        print(f"[cache] loaded {args.cache}", flush=True)
    else:
        data = generate_cache(requested, requested, basis=args.basis, nroots=6)
        save_cache(args.cache, data)
        print(f"[cache] saved {args.cache}", flush=True)

    fields, fitted, metrics, models, heldout = run_fit(
        data, degree=args.degree, rank=args.rank, seed=args.seed
    )
    validation = None
    validation_fields = None
    validation_fitted = None
    if not args.skip_offgrid:
        centers = 0.5 * (requested[:-1] + requested[1:])
        anchor_index = tuple(np.asarray(data["anchor"], dtype=int))
        anchor_coordinates = (
            float(data["qx"][anchor_index[0]]),
            float(data["qy"][anchor_index[1]]),
        )
        if args.validation_cache.exists() and not args.force:
            validation = load_cache(args.validation_cache)
            np.testing.assert_allclose(validation["qx"], centers)
            np.testing.assert_allclose(validation["qy"], centers)
            np.testing.assert_allclose(
                validation["anchor_coordinates"], anchor_coordinates
            )
            print(f"[cache] loaded {args.validation_cache}", flush=True)
        else:
            validation = generate_cache(
                centers,
                centers,
                basis=args.basis,
                nroots=6,
                reference_coordinates=anchor_coordinates,
            )
            save_cache(args.validation_cache, validation)
            print(f"[cache] saved {args.validation_cache}", flush=True)
        validation_fields = anchor_procrustes_fields(
            validation, energy_shift=fields["energy_shift"]
        )
        validation_fitted = evaluate_models(validation, validation_fields, models)
        append_offgrid_metrics(metrics, validation_fields, validation_fitted)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "h3plus_casci_positive_link_fit"
    for model, suffix in zip(models, ("energy", "links_x", "links_y")):
        model.save(stem.with_name(stem.name + f"_{suffix}.npz"))
    np.savez(
        stem.with_name(stem.name + "_data.npz"),
        qx=data["qx"],
        qy=data["qy"],
        hamiltonian=fields["hamiltonian"],
        links_x=fields["links_x"],
        links_y=fields["links_y"],
        predicted_hamiltonian=fitted["hamiltonian"],
        predicted_links_x=fitted["links_x"],
        predicted_links_y=fitted["links_y"],
        gauges=fields["gauges"],
        reference_positive=fields["reference_positive"],
        heldout_energy=heldout["energy"],
        heldout_links_x=heldout["links_x"],
        heldout_links_y=heldout["links_y"],
        **({} if validation is None else {
            "offgrid_qx": validation["qx"],
            "offgrid_qy": validation["qy"],
            "offgrid_hamiltonian": validation_fields["hamiltonian"],
            "offgrid_links_x": validation_fields["links_x"],
            "offgrid_links_y": validation_fields["links_y"],
            "predicted_offgrid_hamiltonian": validation_fitted["hamiltonian"],
            "predicted_offgrid_links_x": validation_fitted["links_x"],
            "predicted_offgrid_links_y": validation_fitted["links_y"],
        }),
    )
    metrics_path = stem.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    figure_path = stem.with_suffix(".png")
    plot_results(
        data,
        fields,
        fitted,
        figure_path,
        metrics,
        validation=None if validation is None else (
            validation, validation_fields, validation_fitted
        ),
    )
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"figure: {figure_path}", flush=True)


if __name__ == "__main__":
    main()
