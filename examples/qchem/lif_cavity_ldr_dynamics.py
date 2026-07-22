#!/usr/bin/env python3
"""LiF polariton dynamics in the nuclear local diabatic representation.

The electronic stage builds local cavity-polariton eigenstates on a uniform
Li--F grid and their nearest-neighbor many-electron overlaps.  The dynamics
stage forms the linked LDR Hamiltonian

    H[i,a,j,b] = T_R[i,j] <Phi_a(R_i)|Phi_b(R_j)>
                   + delta_ij delta_ab E_a(R_i)

and propagates a Gaussian nuclear packet prepared in the first dipole-bright
polariton at the initial geometry.  The expensive electronic data are cached.

This script imports the two LiF gauge drivers used for the manuscript.  Keep
``make_lif_casscf_cavity_demo.py`` and ``make_lif_casci_glg.py`` together and
pass their directory with ``--driver-dir``.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-pyqed")

REPO_ROOT = Path(os.environ.get("PYQED_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import scipy.linalg

from pyqed.dvr.dvr_1d import SineDVR
from pyqed.qchem.geometric import dipole_orbital_rotation_unitary
from pyqed.qchem.mcscf.casci import (
    _compute_ci_mo_overlap,
    overlap as casci_overlap,
)


ANGSTROM_TO_BOHR = 1.8897261254578281
AU_TIME_FS = 0.024188843265857
EV_TO_HARTREE = 1.0 / 27.211386245988
AMU_TO_ELECTRON_MASS = 1822.888486209
LI7_MASS_AMU = 7.0160034366
F19_MASS_AMU = 18.9984031627
DEFAULT_REDUCED_MASS_AMU = LI7_MASS_AMU * F19_MASS_AMU / (
    LI7_MASS_AMU + F19_MASS_AMU
)
GAUGE_ORDER = ("lg", "vg", "glg", "gvg")
GAUGE_LABELS = {"lg": "LG", "vg": "VG", "glg": "GLG", "gvg": "GVG"}
GAUGE_COLORS = {
    "lg": "#D55E00",
    "vg": "#009E73",
    "glg": "#CC79A7",
    "gvg": "#0072B2",
}


def load_module(name: str, path: Path):
    if not path.exists():
        raise FileNotFoundError(path)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hermitize(matrix):
    matrix = np.asarray(matrix)
    return 0.5 * (matrix + matrix.conj().T)


def low_eigenpairs(hamiltonian, nroots):
    nroots = min(int(nroots), hamiltonian.shape[0])
    return scipy.linalg.eigh(
        hermitize(hamiltonian),
        subset_by_index=(0, nroots - 1),
    )


def polariton_brightness(roots, vectors, dipole_blocks):
    """Dipole oscillator-strength proxy from the polariton ground state."""
    nq, nstates, _ = dipole_blocks.shape
    coeff = vectors.reshape(nq, nstates, len(roots))
    transition = np.einsum(
        "qa,qab,qbr->r",
        coeff[:, :, 0].conj(),
        dipole_blocks,
        coeff,
        optimize=True,
    )
    return np.maximum(roots - roots[0], 0.0) * np.abs(transition) ** 2


def exact_singlet_glg_point(
    casci,
    photon,
    *,
    omega,
    coupling,
    axis_index,
    nlocal,
    npolariton,
    base,
    glg,
):
    """Return exact-singlet GLG roots, vectors, and local CI frames."""
    h0_full = glg.determinant_hamiltonian(casci, base)
    mu_full, mu2_full = glg.determinant_dipole_moments(
        casci,
        base,
        axis_index=axis_index,
        center=np.zeros(3),
    )
    projector, _ = glg.exact_singlet_projector(casci, base)
    h0 = hermitize(projector.conj().T @ h0_full @ projector)
    mu = hermitize(projector.conj().T @ mu_full @ projector)
    mu2 = hermitize(projector.conj().T @ mu2_full @ projector)

    local_energies = []
    local_frames = []
    local_ci = []
    local_mu = []
    for field in photon["q"]:
        hloc = h0 + coupling * field * mu + 0.5 * coupling**2 / omega * mu2
        values, vectors = scipy.linalg.eigh(
            hermitize(hloc),
            subset_by_index=(0, nlocal - 1),
        )
        local_energies.append(values)
        local_frames.append(vectors)
        local_ci.append((projector @ vectors).T)
        local_mu.append(hermitize(vectors.conj().T @ mu @ vectors))

    local_energies = np.asarray(local_energies)
    local_frames = np.asarray(local_frames)
    local_ci = np.asarray(local_ci)
    local_mu = np.asarray(local_mu)
    overlaps = np.einsum(
        "qpa,rpb->qarb",
        local_frames.conj(),
        local_frames,
        optimize=True,
    )
    nq = len(photon["q"])
    hamiltonian = np.einsum(
        "qr,qarb->qarb",
        photon["tq"],
        overlaps,
        optimize=True,
    )
    for q, photon_potential in enumerate(photon["vq"]):
        hamiltonian[q, :, q, :] += np.diag(local_energies[q] + photon_potential)
    hamiltonian = hamiltonian.reshape(nq * nlocal, nq * nlocal)
    roots, vectors = low_eigenpairs(hamiltonian, npolariton)
    brightness = polariton_brightness(roots, vectors, local_mu)
    return roots, vectors, brightness, local_ci


def ordinary_point(
    gauge,
    casci,
    energies,
    mu,
    mu2,
    photon,
    *,
    omega,
    coupling,
    state_ids,
    axis_index,
    npolariton,
    base,
):
    active_energy = energies[list(state_ids)]
    active_mu = mu[np.ix_(state_ids, state_ids)]
    active_mu2 = mu2[np.ix_(state_ids, state_ids)]
    if gauge == "lg":
        hamiltonian = base.length_truncated_dvr_hamiltonian(
            active_energy,
            active_mu,
            photon,
            omega=omega,
            coupling=coupling,
            dipole2=active_mu2,
        )
    elif gauge == "vg":
        hamiltonian = base.velocity_truncated_dvr_hamiltonian(
            active_energy,
            active_mu,
            photon,
            omega=omega,
            coupling=coupling,
        )
    elif gauge == "gvg":
        hamiltonian = base.geometric_velocity_dvr_hamiltonian(
            energies,
            mu,
            photon,
            omega=omega,
            coupling=coupling,
            state_ids=state_ids,
            casci=casci,
            axis_index=axis_index,
            center=np.zeros(3),
        )
    else:
        raise ValueError(f"Unsupported ordinary gauge {gauge!r}.")

    roots, vectors = low_eigenpairs(hamiltonian, npolariton)
    dipole_blocks = np.repeat(active_mu[None, :, :], len(photon["q"]), axis=0)
    brightness = polariton_brightness(roots, vectors, dipole_blocks)
    return roots, vectors, brightness


def build_point(distance, args, photon, base, glg):
    _, _, mc, casci = base.build_lif_casscf(
        distance,
        basis=args.basis,
        ncas=args.ncas,
        nelecas=args.nelecas,
        nstates=args.parent_states,
        max_cycle=args.max_cycle,
        driver=args.driver,
        fix_spin=False,
        spin_shift=0.0,
        ci_method="direct_spin0_symm",
    )
    if not mc.converged:
        raise RuntimeError(f"CASSCF did not converge at R={distance:.8f} Angstrom.")
    spin_square = np.asarray(
        [casci.spin_square(index) for index in range(args.parent_states)],
        dtype=float,
    )
    if np.max(np.abs(spin_square)) > args.spin_tolerance:
        raise RuntimeError(
            f"Non-singlet parent root at R={distance:.8f}: S2={spin_square}."
        )

    energies = np.asarray(casci.e_tot, dtype=float)
    dipoles = base.dipole_matrix(casci, center=np.zeros(3))
    mu = dipoles[:, :, args.axis_index]
    mu2 = base.projected_dipole_square_from_links(
        casci,
        axis_index=args.axis_index,
        center=np.zeros(3),
        state_ids=tuple(range(args.parent_states)),
    )
    point = {
        "casci": casci,
        "roots": {},
        "vectors": {},
        "brightness": {},
        "spin_square": spin_square,
    }
    for gauge in args.gauges:
        if gauge == "glg":
            roots, vectors, brightness, local_ci = exact_singlet_glg_point(
                casci,
                photon,
                omega=args.omega,
                coupling=args.coupling,
                axis_index=args.axis_index,
                nlocal=args.local_states,
                npolariton=args.polariton_states,
                base=base,
                glg=glg,
            )
            point["local_ci"] = local_ci
        else:
            roots, vectors, brightness = ordinary_point(
                gauge,
                casci,
                energies,
                mu,
                mu2,
                photon,
                omega=args.omega,
                coupling=args.coupling,
                state_ids=args.state_ids,
                axis_index=args.axis_index,
                npolariton=args.polariton_states,
                base=base,
            )
        point["roots"][gauge] = roots
        point["vectors"][gauge] = vectors
        point["brightness"][gauge] = brightness

    if "gvg" in args.gauges:
        eta = args.coupling / args.omega
        point["gvg_orbital_u"] = np.asarray(
            [
                dipole_orbital_rotation_unitary(
                    casci,
                    eta * q,
                    axis=args.axis_index,
                    center=np.zeros(3),
                )
                for q in photon["q"]
            ]
        )
    return point


def contract_polariton_link(left_vectors, overlap_blocks, right_vectors):
    nq, nleft, nright = overlap_blocks.shape
    npolariton = left_vectors.shape[1]
    left = left_vectors.reshape(nq, nleft, npolariton)
    right = right_vectors.reshape(nq, nright, npolariton)
    return np.einsum(
        "qar,qab,qbs->rs",
        left.conj(),
        overlap_blocks,
        right,
        optimize=True,
    )


def local_ci_overlap_blocks(left, right, nq):
    blocks = []
    left_model = copy.copy(left["casci"])
    right_model = copy.copy(right["casci"])
    for q in range(nq):
        left_model.ci = left["local_ci"][q]
        right_model.ci = right["local_ci"][q]
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            value = casci_overlap(left_model, right_model)
        blocks.append(np.asarray(value, dtype=complex))
    return np.asarray(blocks)


def gvg_overlap_blocks(left, right, args, nq):
    mo_overlap = _compute_ci_mo_overlap(left["casci"], right["casci"])
    blocks = []
    for q in range(nq):
        transformed_overlap = (
            left["gvg_orbital_u"][q].conj().T
            @ mo_overlap
            @ right["gvg_orbital_u"][q]
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            parent_overlap = casci_overlap(
                left["casci"],
                right["casci"],
                s=transformed_overlap,
            )
        blocks.append(parent_overlap[np.ix_(args.state_ids, args.state_ids)])
    return np.asarray(blocks)


def build_neighbor_links(left, right, args, photon):
    nq = len(photon["q"])
    links = {}
    ordinary_overlap = None
    if any(gauge in args.gauges for gauge in ("lg", "vg")):
        parent_overlap = casci_overlap(left["casci"], right["casci"])
        ordinary_overlap = np.asarray(
            parent_overlap[np.ix_(args.state_ids, args.state_ids)],
            dtype=complex,
        )

    for gauge in args.gauges:
        if gauge in ("lg", "vg"):
            blocks = np.repeat(ordinary_overlap[None, :, :], nq, axis=0)
        elif gauge == "gvg":
            blocks = gvg_overlap_blocks(left, right, args, nq)
        elif gauge == "glg":
            blocks = local_ci_overlap_blocks(left, right, nq)
        else:
            raise ValueError(gauge)
        links[gauge] = contract_polariton_link(
            left["vectors"][gauge],
            blocks,
            right["vectors"][gauge],
        )
    return links


def scan(args, base, glg):
    mass = args.reduced_mass_amu * AMU_TO_ELECTRON_MASS
    nuclear_dvr = SineDVR(
        args.r_min * ANGSTROM_TO_BOHR,
        args.r_max * ANGSTROM_TO_BOHR,
        args.nuclear_points,
        mass=mass,
    )
    distances = nuclear_dvr.x / ANGSTROM_TO_BOHR
    dx_bohr = nuclear_dvr.dx
    kinetic = np.diag(np.full(args.nuclear_points, 1.0 / (mass * dx_bohr**2)))
    kinetic += np.diag(
        np.full(args.nuclear_points - 1, -0.5 / (mass * dx_bohr**2)),
        k=1,
    )
    kinetic += np.diag(
        np.full(args.nuclear_points - 1, -0.5 / (mass * dx_bohr**2)),
        k=-1,
    )
    photon = base.build_photon_dvr(
        args.q_domain,
        q_level=args.q_level,
        omega=args.omega,
    )
    roots = {gauge: [] for gauge in args.gauges}
    brightness = {gauge: [] for gauge in args.gauges}
    links = {gauge: [] for gauge in args.gauges}
    spin_square = []
    previous = None

    for index, distance in enumerate(distances):
        print(
            f"[scan] {index + 1:3d}/{len(distances)} R={distance:.8f} Angstrom",
            flush=True,
        )
        point = build_point(distance, args, photon, base, glg)
        spin_square.append(point["spin_square"])
        for gauge in args.gauges:
            roots[gauge].append(point["roots"][gauge])
            brightness[gauge].append(point["brightness"][gauge])
        if previous is not None:
            point_links = build_neighbor_links(previous, point, args, photon)
            for gauge in args.gauges:
                links[gauge].append(point_links[gauge])
                singular_values = scipy.linalg.svdvals(point_links[gauge])
                print(
                    f"       {GAUGE_LABELS[gauge]} link singular values "
                    f"[{singular_values.min():.6f}, {singular_values.max():.6f}]",
                    flush=True,
                )
        previous = point

    config = scan_config(args)
    payload = {
        "distances_angstrom": distances,
        "nuclear_kinetic_hartree": kinetic,
        "photon_q": photon["q"],
        "spin_square": np.asarray(spin_square),
        "gauges": np.asarray(args.gauges),
        "config_json": np.asarray(json.dumps(config, sort_keys=True)),
        "overlap_method": np.asarray("nearest_neighbor_finite_difference_LDR"),
    }
    for gauge in args.gauges:
        payload[f"energies_{gauge}"] = np.asarray(roots[gauge])
        payload[f"brightness_{gauge}"] = np.asarray(brightness[gauge])
        payload[f"links_{gauge}"] = np.asarray(links[gauge])
    args.cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.cache, **payload)
    print(f"[cache] Saved {args.cache}", flush=True)
    return payload


def scan_config(args):
    return {
        "basis": args.basis,
        "driver": args.driver,
        "ncas": args.ncas,
        "nelecas": args.nelecas,
        "parent_states": args.parent_states,
        "active_states": args.active_states,
        "local_states": args.local_states,
        "polariton_states": args.polariton_states,
        "state_ids": list(args.state_ids),
        "nuclear_points": args.nuclear_points,
        "r_domain_angstrom": [args.r_min, args.r_max],
        "reduced_mass_amu": args.reduced_mass_amu,
        "nuclear_keo": "second_order_finite_difference_Dirichlet",
        "q_domain": list(args.q_domain),
        "q_level": args.q_level,
        "omega_ev": args.omega_ev,
        "coupling_ev": args.coupling_ev,
        "axis": args.axis,
        "gauges": list(args.gauges),
        "ci_method": "direct_spin0_symm",
        "spin_projection": "exact_S2_zero_for_GLG",
    }


def load_cache(args):
    with np.load(args.cache) as data:
        payload = {key: np.asarray(data[key]) for key in data.files}
    missing = []
    for gauge in args.gauges:
        for prefix in ("energies", "brightness", "links"):
            key = f"{prefix}_{gauge}"
            if key not in payload:
                missing.append(key)
    if missing:
        raise KeyError(f"Cache is missing {missing}; rerun with --force-scan.")
    cached_config = json.loads(str(payload["config_json"]))
    expected_config = scan_config(args)
    cached_config.pop("gauges", None)
    expected_config.pop("gauges", None)
    if cached_config != expected_config:
        raise ValueError(
            "The cache configuration differs from the requested scan. "
            "Use a different --cache path or rerun with --force-scan."
        )
    print(f"[cache] Loaded {args.cache}", flush=True)
    print(f"[cache] Configuration: {str(payload['config_json'])}", flush=True)
    return payload


def linked_overlap_matrix(links):
    links = np.asarray(links, dtype=complex)
    ngrid = links.shape[0] + 1
    nstates = links.shape[1]
    overlap = np.zeros((ngrid, nstates, ngrid, nstates), dtype=complex)
    for i in range(ngrid):
        overlap[i, :, i, :] = np.eye(nstates)
        product = np.eye(nstates, dtype=complex)
        for j in range(i + 1, ngrid):
            product = product @ links[j - 1]
            overlap[i, :, j, :] = product
            overlap[j, :, i, :] = product.conj().T
    return overlap


def select_first_bright(strength, threshold):
    strength = np.asarray(strength, dtype=float)
    excited = strength[1:]
    maximum = float(np.max(excited))
    if maximum <= 0.0:
        raise RuntimeError("No dipole-bright polariton was found.")
    candidates = np.flatnonzero(excited >= float(threshold) * maximum) + 1
    return int(candidates[0])


def propagate_one_gauge(gauge, payload, args):
    distances = np.asarray(payload["distances_angstrom"], dtype=float)
    kinetic = np.asarray(payload["nuclear_kinetic_hartree"], dtype=complex)
    energies = np.asarray(payload[f"energies_{gauge}"], dtype=float)
    brightness = np.asarray(payload[f"brightness_{gauge}"], dtype=float)
    links = np.asarray(payload[f"links_{gauge}"], dtype=complex)
    reference_overlap = linked_overlap_matrix(links)
    ngrid, nstates = energies.shape
    reference_index = int(np.argmin(np.abs(distances - args.initial_r)))
    if args.initial_root is None:
        bright_root = select_first_bright(
            brightness[reference_index],
            args.bright_threshold,
        )
    else:
        bright_root = int(args.initial_root)
        if not 0 <= bright_root < nstates:
            raise ValueError("--initial-root is outside the cached polariton space.")

    if brightness[reference_index, bright_root] < args.minimum_brightness:
        raise RuntimeError(
            f"The selected {gauge} root has brightness "
            f"{brightness[reference_index, bright_root]:.3e}, below "
            f"--minimum-brightness={args.minimum_brightness:.3e}. "
            "Increase the photon/local-state truncation."
        )

    kinetic_overlap = np.zeros(
        (ngrid, nstates, ngrid, nstates),
        dtype=complex,
    )
    for index in range(ngrid):
        kinetic_overlap[index, :, index, :] = np.eye(nstates)
    for index, link in enumerate(links):
        kinetic_overlap[index, :, index + 1, :] = link
        kinetic_overlap[index + 1, :, index, :] = link.conj().T
    hamiltonian = np.einsum(
        "ij,iajb->iajb",
        kinetic,
        kinetic_overlap,
        optimize=True,
    ).reshape(ngrid * nstates, ngrid * nstates)
    energy_zero = float(np.min(energies[:, 0]))
    hamiltonian += np.diag((energies - energy_zero).reshape(-1))
    hamiltonian = hermitize(hamiltonian)

    gaussian = np.exp(
        -((distances - args.initial_r) ** 2) / (4.0 * args.sigma_angstrom**2)
    )
    gaussian /= np.linalg.norm(gaussian)
    psi0 = gaussian[:, None] * reference_overlap[:, :, reference_index, bright_root]
    projected_weight = float(np.vdot(psi0, psi0).real)
    if projected_weight <= 1.0e-14:
        raise RuntimeError(f"Initial {gauge} reference projection has zero norm.")
    psi0 /= np.sqrt(projected_weight)

    eigenvalues, eigenvectors = scipy.linalg.eigh(hamiltonian)
    initial_coefficients = eigenvectors.conj().T @ psi0.reshape(-1)
    times_fs = np.arange(
        0.0,
        args.time_fs + 0.5 * args.output_every_fs,
        args.output_every_fs,
    )
    phases = np.exp(
        -1.0j * eigenvalues[:, None] * times_fs[None, :] / AU_TIME_FS
    )
    psi_t = (
        eigenvectors @ (initial_coefficients[:, None] * phases)
    ).T.reshape(len(times_fs), ngrid, nstates)
    probability = np.abs(psi_t) ** 2
    norms = np.sum(probability, axis=(1, 2))
    density = np.sum(probability, axis=2)
    mean_r = np.sum(density * distances[None, :], axis=1) / norms
    populations = np.sum(probability, axis=1)
    autocorrelation = np.abs(
        np.einsum("ia,tia->t", psi0.conj(), psi_t, optimize=True)
    ) ** 2
    singular_values = np.asarray([scipy.linalg.svdvals(link) for link in links])
    print(
        f"[initial] {GAUGE_LABELS[gauge]} at R={distances[reference_index]:.8f} "
        f"uses P{bright_root}; brightness={brightness[reference_index, bright_root]:.6e}",
        flush=True,
    )
    return {
        "times_fs": times_fs,
        "bright_root": bright_root,
        "reference_index": reference_index,
        "reference_distance_angstrom": distances[reference_index],
        "projected_weight": projected_weight,
        "energies_hartree": energies,
        "brightness": brightness,
        "norm": norms,
        "mean_r_angstrom": mean_r,
        "populations": populations,
        "autocorrelation": autocorrelation,
        "density_dvr": density,
        "psi": psi_t,
        "link_singular_values": singular_values,
        "ldr_eigenvalues_hartree": eigenvalues,
    }


def plot_results(results, payload, args, output):
    import matplotlib.pyplot as plt

    distances = np.asarray(payload["distances_angstrom"], dtype=float)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.2))
    axes = axes.ravel()
    for gauge in args.gauges:
        result = results[gauge]
        root = result["bright_root"]
        color = GAUGE_COLORS[gauge]
        label = f"{GAUGE_LABELS[gauge]} (P{root})"
        energies = result["energies_hartree"]
        axes[0].plot(
            distances,
            (energies[:, root] - np.min(energies[:, 0])) / EV_TO_HARTREE,
            color=color,
            lw=1.35,
            label=label,
        )
        axes[1].plot(
            result["times_fs"],
            result["mean_r_angstrom"],
            color=color,
            lw=1.35,
        )
        axes[2].plot(
            result["times_fs"],
            result["populations"][:, root],
            color=color,
            lw=1.35,
        )
        axes[3].plot(
            result["times_fs"],
            result["autocorrelation"],
            color=color,
            lw=1.35,
        )
    axes[0].set(xlabel=r"Li--F distance $R$ (Angstrom)", ylabel="bright PES (eV)")
    axes[1].set(xlabel="time (fs)", ylabel=r"$\langle R\rangle$ (Angstrom)")
    axes[2].set(xlabel="time (fs)", ylabel="local bright-root population", ylim=(-0.02, 1.02))
    axes[3].set(xlabel="time (fs)", ylabel="return probability", ylim=(-0.02, 1.02))
    axes[0].legend(frameon=False, ncol=2)
    for panel, axis in zip("abcd", axes):
        axis.text(-0.14, 1.03, panel, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"))
    fig.savefig(output.with_suffix(".png"), dpi=350)
    plt.close(fig)


def save_results(results, payload, args):
    output = args.output.with_suffix("")
    output.parent.mkdir(parents=True, exist_ok=True)
    npz_payload = {
        "distances_angstrom": payload["distances_angstrom"],
        "gauges": np.asarray(args.gauges),
        "initial_r_angstrom": args.initial_r,
        "sigma_angstrom": args.sigma_angstrom,
        "overlap_method": payload["overlap_method"],
    }
    summary = {
        "method": "finite-difference linked local diabatic representation",
        "cache": str(args.cache.resolve()),
        "initial_condition": "reference-projected first dipole-bright polariton",
        "initial_r_angstrom": args.initial_r,
        "sigma_angstrom": args.sigma_angstrom,
        "bright_threshold_relative": args.bright_threshold,
        "gauges": {},
    }
    for gauge, result in results.items():
        for key, value in result.items():
            npz_payload[f"{gauge}_{key}"] = value
        summary["gauges"][gauge] = {
            "bright_root": int(result["bright_root"]),
            "reference_distance_angstrom": float(result["reference_distance_angstrom"]),
            "reference_brightness": float(
                result["brightness"][result["reference_index"], result["bright_root"]]
            ),
            "reference_projection_weight": float(result["projected_weight"]),
            "maximum_norm_error": float(np.max(np.abs(result["norm"] - 1.0))),
            "final_mean_r_angstrom": float(result["mean_r_angstrom"][-1]),
            "minimum_return_probability": float(np.min(result["autocorrelation"])),
            "minimum_link_singular_value": float(
                np.min(result["link_singular_values"])
            ),
        }
    np.savez_compressed(output.with_suffix(".npz"), **npz_payload)
    output.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    if not args.no_plot:
        plot_results(results, payload, args, output)
    print(json.dumps(summary, indent=2), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--driver-dir",
        type=Path,
        default=Path(os.environ.get("LIF_GAUGE_DRIVER_DIR", Path(__file__).resolve().parent)),
    )
    parser.add_argument("--cache", type=Path, default=Path("lif_cavity_ldr_cache.npz"))
    parser.add_argument("--output", type=Path, default=Path("lif_cavity_ldr_dynamics"))
    parser.add_argument("--force-scan", action="store_true")
    parser.add_argument("--scan-only", action="store_true")
    parser.add_argument("--gauges", nargs="+", choices=GAUGE_ORDER, default=list(GAUGE_ORDER))
    parser.add_argument("--basis", default="sto-3g")
    parser.add_argument("--driver", default="gbasis-pyscf")
    parser.add_argument("--ncas", type=int, default=6)
    parser.add_argument("--nelecas", type=int, default=6)
    parser.add_argument("--parent-states", type=int, default=5)
    parser.add_argument("--active-states", type=int, default=4)
    parser.add_argument("--local-states", type=int, default=12)
    parser.add_argument("--polariton-states", type=int, default=12)
    parser.add_argument("--max-cycle", type=int, default=60)
    parser.add_argument("--spin-tolerance", type=float, default=1.0e-7)
    parser.add_argument("--r-min", type=float, default=1.15)
    parser.add_argument("--r-max", type=float, default=3.05)
    parser.add_argument("--nuclear-points", type=int, default=63)
    parser.add_argument("--reduced-mass-amu", type=float, default=DEFAULT_REDUCED_MASS_AMU)
    parser.add_argument("--q-domain", type=float, nargs=2, default=(-8.0, 8.0))
    parser.add_argument("--q-level", type=int, default=5)
    parser.add_argument("--omega-ev", type=float, default=1.7330247707268651)
    parser.add_argument("--coupling-ev", type=float, default=2.0)
    parser.add_argument("--axis", choices=("x", "y", "z"), default="y")
    parser.add_argument("--initial-r", type=float, default=1.45)
    parser.add_argument("--sigma-angstrom", type=float, default=0.06)
    parser.add_argument("--initial-root", type=int, default=None)
    parser.add_argument("--bright-threshold", type=float, default=0.05)
    parser.add_argument("--minimum-brightness", type=float, default=1.0e-10)
    parser.add_argument("--time-fs", type=float, default=200.0)
    parser.add_argument("--output-every-fs", type=float, default=0.5)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if len(set(args.gauges)) != len(args.gauges):
        parser.error("--gauges contains duplicates.")
    if not 1 <= args.active_states <= args.parent_states:
        parser.error("--active-states must be between 1 and --parent-states.")
    if args.local_states < 1 or args.polariton_states < 2:
        parser.error("Use at least one local GLG state and two polariton states.")
    if args.nuclear_points < 3 or args.r_min >= args.r_max:
        parser.error("The nuclear sine DVR requires at least three points and r-min < r-max.")
    if not args.r_min < args.initial_r < args.r_max:
        parser.error("--initial-r must lie inside the nuclear domain.")
    if args.sigma_angstrom <= 0.0 or args.reduced_mass_amu <= 0.0:
        parser.error("The packet width and reduced mass must be positive.")
    if not 0.0 < args.bright_threshold <= 1.0:
        parser.error("--bright-threshold must lie in (0, 1].")
    if args.minimum_brightness <= 0.0:
        parser.error("--minimum-brightness must be positive.")
    args.state_ids = tuple(range(args.active_states))
    args.axis_index = {"x": 0, "y": 1, "z": 2}[args.axis]
    args.omega = args.omega_ev * EV_TO_HARTREE
    args.coupling = args.coupling_ev * EV_TO_HARTREE
    return args


def main():
    args = parse_args()
    base = load_module(
        "lif_cavity_base_driver",
        args.driver_dir / "make_lif_casscf_cavity_demo.py",
    )
    glg = load_module(
        "lif_cavity_glg_driver",
        args.driver_dir / "make_lif_casci_glg.py",
    )
    if args.force_scan or not args.cache.exists():
        payload = scan(args, base, glg)
    else:
        payload = load_cache(args)
    if args.scan_only:
        return
    results = {
        gauge: propagate_one_gauge(gauge, payload, args)
        for gauge in args.gauges
    }
    save_results(results, payload, args)


if __name__ == "__main__":
    main()
