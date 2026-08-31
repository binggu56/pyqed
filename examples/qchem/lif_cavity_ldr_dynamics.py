#!/usr/bin/env python3
"""LiF cavity dynamics with finite-difference linked LDR.

The scan computes polariton energies and exact neighboring-geometry overlaps.
The dynamics starts from a Gaussian packet in the first dipole-bright
polariton at ``R0``.  Expensive scan data are cached in one NPZ file.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import sys
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib-pyqed")

ROOT = Path(os.environ.get("PYQED_REPO_ROOT", Path(__file__).resolve().parents[2]))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import scipy.linalg

from pyqed.qchem.geometric import dipole_orbital_rotation_unitary
from pyqed.qchem.mcscf.casci import mo_overlap
from pyqed.qchem.mcscf.casci import overlap as casci_overlap
from pyqed.units import amu2au, au2angstrom, au2ev, au2fs


# LiF reduced mass in unified atomic mass units.
MASS_LIF = 7.0160034366 * 18.9984031627 / (7.0160034366 + 18.9984031627)

# Electronic and cavity model used in the manuscript.
BASIS = "sto-3g"
NCAS, NELECAS = 6, 6  # CAS(6e,6o)
NPARENT, ACTIVE = 5, (0, 1, 2, 3)  # Compute 5 roots; retain S0-S3 in LG/VG/GVG.
AXIS = 1  # y polarization
RMIN, RMAX = 1.15, 3.05  # Li-F distance range in Angstrom.
QDOMAIN = (-8.0, 8.0)  # Photon displacement-coordinate interval.
GAUGES = ("lg", "vg", "glg", "gvg")
LABEL = {"lg": "LG", "vg": "VG", "glg": "GLG", "gvg": "GVG"}
COLOR = {"lg": "#D55E00", "vg": "#009E73", "glg": "#CC79A7", "gvg": "#0072B2"}

# Calculation settings. Edit these values, then run this file directly.
NR = 63  # Number of interior nuclear grid points.
Q_LEVEL = 5  # Nested photon DVR level; level 5 contains 31 q points.
NLOCAL = 12  # GLG electronic states retained at each photon coordinate.
NPOL = 12  # Polariton states retained at each nuclear geometry.
OMEGA_EV = 1.7330247707268651  # Cavity frequency in eV.
G_EV = 2.0  # Light-matter coupling in eV.
R0, SIGMA = 1.45, 0.06  # Initial Gaussian center and width in Angstrom.
TMAX, DTOUT = 200.0, 0.5  # Final time and output interval in fs.
REBUILD = False  # Set True to recompute the electronic/polaritonic scan.

# Convert user-facing units to atomic units once.
OMEGA, G = OMEGA_EV / au2ev, G_EV / au2ev
HERE = Path(__file__).resolve().parent
DRIVERS = HERE  # Directory containing the two LiF electronic-structure drivers.
CACHE = HERE / f"lif_ldr_g{G_EV:g}_nr{NR}_cache.npz"  # Expensive scan data.
OUTPUT = HERE / f"lif_ldr_g{G_EV:g}_nr{NR}_dynamics"  # Result filename stem.


# -----------------------------------------------------------------------------
# Small numerical and driver-loading helpers
# -----------------------------------------------------------------------------


def load_driver(name, path):
    """Load one of the companion LiF calculation scripts as a Python module."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hermitian(a):
    """Remove insignificant numerical non-Hermiticity before diagonalization."""
    return 0.5 * (a + a.conj().T)


def lowest(h, n):
    """Return the lowest ``n`` eigenpairs of a dense Hermitian matrix."""
    return scipy.linalg.eigh(hermitian(h), subset_by_index=(0, min(n, len(h)) - 1))


def safe_overlap(left, right, s=None):
    """Evaluate a many-electron CASCI overlap while suppressing benign warnings."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        return np.asarray(casci_overlap(left, right, s=s), dtype=complex)


def brightness(energy, vectors, mu_q):
    """Return $(E_n-E_0)|<P_0|mu|P_n>|^2$."""
    nq, ne, _ = mu_q.shape
    c = vectors.reshape(nq, ne, len(energy))
    transition = np.einsum("qa,qab,qbr->r", c[:, :, 0].conj(), mu_q, c, optimize=True)
    return np.maximum(energy - energy[0], 0.0) * abs(transition) ** 2


# -----------------------------------------------------------------------------
# Polariton states at one nuclear geometry
# -----------------------------------------------------------------------------


def glg_state(casci, photon, base, glg):
    """Build GLG by rediagonalizing the exact-singlet electronic Hamiltonian at each q."""
    # Transform determinant-space operators into the exact spin-singlet sector.
    h0f = glg.determinant_hamiltonian(casci, base)
    muf, mu2f = glg.determinant_dipole_moments(
        casci, base, axis_index=AXIS, center=np.zeros(3)
    )
    projector, _ = glg.exact_singlet_projector(casci, base)
    h0 = hermitian(projector.conj().T @ h0f @ projector)
    mu = hermitian(projector.conj().T @ muf @ projector)
    mu2 = hermitian(projector.conj().T @ mu2f @ projector)

    # At every photon coordinate, retain the lowest field-dressed electronic states.
    local_e, frame, local_ci, local_mu = [], [], [], []
    for q in photon["q"]:
        h = h0 + G * q * mu + 0.5 * G**2 / OMEGA * mu2
        e, u = lowest(h, NLOCAL)
        local_e.append(e)
        frame.append(u)
        local_ci.append((projector @ u).T)
        local_mu.append(hermitian(u.conj().T @ mu @ u))

    # The photon kinetic operator connects q-dependent electronic frames through
    # their overlaps. Diagonalizing the assembled matrix gives GLG polaritons.
    local_e, frame = np.asarray(local_e), np.asarray(frame)
    overlap = np.einsum("qpa,rpb->qarb", frame.conj(), frame, optimize=True)
    h = np.einsum("qr,qarb->qarb", photon["tq"], overlap, optimize=True)
    for q, vq in enumerate(photon["vq"]):
        h[q, :, q, :] += np.diag(local_e[q] + vq)
    e, u = lowest(h.reshape(len(h) * NLOCAL, -1), NPOL)
    return e, u, brightness(e, u, np.asarray(local_mu)), np.asarray(local_ci)


def ordinary_state(gauge, casci, energy, mu, mu2, photon, base):
    """Build and diagonalize the truncated LG, VG, or GVG Hamiltonian."""
    ids = list(ACTIVE)
    e, m = energy[ids], mu[np.ix_(ids, ids)]
    if gauge == "lg":
        h = base.length_truncated_dvr_hamiltonian(
            e, m, photon, omega=OMEGA, coupling=G,
            dipole2=mu2[np.ix_(ids, ids)],
        )
    elif gauge == "vg":
        h = base.velocity_truncated_dvr_hamiltonian(
            e, m, photon, omega=OMEGA, coupling=G
        )
    else:
        h = base.geometric_velocity_dvr_hamiltonian(
            energy, mu, photon, omega=OMEGA, coupling=G,
            state_ids=ACTIVE, casci=casci, axis_index=AXIS, center=np.zeros(3),
        )
    roots, vectors = lowest(h, NPOL)
    mu_q = np.repeat(m[None], len(photon["q"]), axis=0)
    return roots, vectors, brightness(roots, vectors, mu_q)


def build_point(r, photon, base, glg):
    """Compute all electronic and polaritonic data at one Li-F distance."""
    # State-averaged CASSCF orbitals are followed by rigorous singlet-only CASCI.
    _, _, mc, casci = base.build_lif_casscf(
        r, basis=BASIS, ncas=NCAS, nelecas=NELECAS, nstates=NPARENT,
        max_cycle=60, fix_spin=False, spin_shift=0.0,
        ci_method="direct_spin0_symm",
    )
    if not mc.converged:
        raise RuntimeError(f"CASSCF failed at R={r:.8f} Angstrom")
    s2 = np.asarray([casci.spin_square(i) for i in range(NPARENT)])
    if max(abs(s2)) > 1e-7:
        raise RuntimeError(f"Non-singlet parent root at R={r:.8f}: S2={s2}")

    # Project the dipole and dipole-square operators into the parent-state basis.
    energy = np.asarray(casci.e_tot)
    mu = base.dipole_matrix(casci, center=np.zeros(3))[:, :, AXIS]
    mu2 = base.projected_dipole_square_from_links(
        casci, axis_index=AXIS, center=np.zeros(3), state_ids=range(NPARENT)
    )
    point = {"casci": casci, "energy": {}, "vector": {}, "bright": {}}
    for gauge in GAUGES:
        if gauge == "glg":
            e, u, b, point["local_ci"] = glg_state(casci, photon, base, glg)
        else:
            e, u, b = ordinary_state(gauge, casci, energy, mu, mu2, photon, base)
        point["energy"][gauge], point["vector"][gauge], point["bright"][gauge] = e, u, b

    # GVG links require the exact one-electron orbital rotation at every q.
    if "gvg" in GAUGES:
        eta = G / OMEGA
        point["gvg_u"] = np.asarray([
            dipole_orbital_rotation_unitary(
                casci, eta * q, axis=AXIS, center=np.zeros(3)
            ) for q in photon["q"]
        ])
    return point


# -----------------------------------------------------------------------------
# Exact neighboring-geometry overlaps for linked LDR
# -----------------------------------------------------------------------------


def polariton_link(left, blocks, right):
    """Contract electronic overlaps with polariton coefficients to form an LDR link."""
    nq, nl, nr = blocks.shape
    npol = left.shape[1]
    cl = left.reshape(nq, nl, npol)
    cr = right.reshape(nq, nr, npol)
    return np.einsum("qar,qab,qbs->rs", cl.conj(), blocks, cr, optimize=True)


def build_links(left, right, photon):
    """Build gauge-specific polariton overlaps between neighboring geometries."""
    nq = len(photon["q"])
    links = {}
    if {"lg", "vg"} & set(GAUGES):
        s = safe_overlap(left["casci"], right["casci"])[np.ix_(ACTIVE, ACTIVE)]
        ordinary = np.repeat(s[None], nq, axis=0)

    for gauge in GAUGES:
        if gauge in ("lg", "vg"):
            blocks = ordinary
        elif gauge == "glg":
            # GLG uses q-dependent field-dressed CI states at both geometries.
            a, b = copy.copy(left["casci"]), copy.copy(right["casci"])
            blocks = []
            for q in range(nq):
                a.ci, b.ci = left["local_ci"][q], right["local_ci"][q]
                blocks.append(safe_overlap(a, b))
            blocks = np.asarray(blocks)
        else:
            # GVG first rotates the cross-geometry MO overlap, then evaluates the
            # resulting many-electron overlap in the retained CASCI state basis.
            smo = mo_overlap(left["casci"], right["casci"])
            blocks = []
            for q in range(nq):
                s = left["gvg_u"][q].conj().T @ smo @ right["gvg_u"][q]
                blocks.append(safe_overlap(left["casci"], right["casci"], s)[np.ix_(ACTIVE, ACTIVE)])
            blocks = np.asarray(blocks)
        links[gauge] = polariton_link(
            left["vector"][gauge], blocks, right["vector"][gauge]
        )
    return links


# -----------------------------------------------------------------------------
# Electronic scan and cache
# -----------------------------------------------------------------------------


def nuclear_grid(nr):
    """Return the Li-F grid and its second-order finite-difference kinetic matrix."""
    r = np.linspace(RMIN, RMAX, nr + 2)[1:-1]
    dx = (r[1] - r[0]) / au2angstrom
    mass = MASS_LIF * amu2au
    t = np.diag(np.full(nr, 1.0 / (mass * dx**2)))
    off = np.full(nr - 1, -0.5 / (mass * dx**2))
    return r, t + np.diag(off, 1) + np.diag(off, -1)


def scan(base, glg):
    """Scan all geometries and cache PESs, brightnesses, and adjacent LDR links."""
    r, kinetic = nuclear_grid(NR)
    photon = base.build_photon_dvr(QDOMAIN, q_level=Q_LEVEL, omega=OMEGA)
    energy = {g: [] for g in GAUGES}
    bright = {g: [] for g in GAUGES}
    links = {g: [] for g in GAUGES}
    previous = None

    for i, ri in enumerate(r):
        print(f"[scan] {i + 1:3d}/{NR}: R={ri:.8f} Angstrom", flush=True)
        point = build_point(ri, photon, base, glg)
        for gauge in GAUGES:
            energy[gauge].append(point["energy"][gauge])
            bright[gauge].append(point["bright"][gauge])
        if previous is not None:
            new = build_links(previous, point, photon)
            status = []
            for gauge in GAUGES:
                links[gauge].append(new[gauge])
                sv = scipy.linalg.svdvals(new[gauge])
                status.append(f"{LABEL[gauge]}={sv.min():.3f}:{sv.max():.3f}")
            print("       links " + "  ".join(status), flush=True)
        previous = point

    data = {"r": r, "kinetic": kinetic, "gauges": np.asarray(GAUGES)}
    for gauge in GAUGES:
        data[f"energy_{gauge}"] = np.asarray(energy[gauge])
        data[f"bright_{gauge}"] = np.asarray(bright[gauge])
        data[f"link_{gauge}"] = np.asarray(links[gauge])
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, **data)
    print(f"[cache] wrote {CACHE}")
    return data


def load_cache(path):
    """Load a previously completed scan without repeating CASSCF/CASCI."""
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


# -----------------------------------------------------------------------------
# Linked-LDR nuclear dynamics
# -----------------------------------------------------------------------------


def transported_state(links, reference, root):
    """Represent one reference polariton in every local LDR frame."""
    nr, npol = len(links) + 1, links.shape[1]
    out = np.empty((nr, npol), dtype=complex)
    ket = np.eye(npol)[:, root]
    out[reference] = ket
    product = np.eye(npol, dtype=complex)
    for i in range(reference - 1, -1, -1):
        product = links[i] @ product
        out[i] = product @ ket
    product = np.eye(npol, dtype=complex)
    for i in range(reference + 1, nr):
        product = product @ links[i - 1]
        out[i] = product.conj().T @ ket
    return out


def propagate(gauge, data):
    """Propagate a Gaussian packet on one gauge's linked polariton surfaces."""
    r, t = data["r"], data["kinetic"]
    energy, bright = data[f"energy_{gauge}"], data[f"bright_{gauge}"]
    links = data[f"link_{gauge}"]
    nr, npol = energy.shape
    # Select the first state carrying at least 5% of the largest local brightness.
    i0 = int(np.argmin(abs(r - R0)))
    cutoff = 0.05 * max(bright[i0, 1:])
    root = int(np.flatnonzero(bright[i0] >= cutoff)[0])
    if root == 0 or bright[i0, root] < 1e-10:
        raise RuntimeError(f"No resolved bright {LABEL[gauge]} polariton at R={r[i0]:.5f}")

    # Assemble the LDR Hamiltonian. Off-diagonal nuclear kinetic blocks are
    # multiplied by exact neighboring-geometry polariton overlap matrices.
    h = np.diag((energy - energy[:, 0].min()).ravel()).astype(complex)
    eye = np.eye(npol)
    for i in range(nr):
        sl = slice(i * npol, (i + 1) * npol)
        h[sl, sl] += t[i, i] * eye
    for i, link in enumerate(links):
        a, b = slice(i * npol, (i + 1) * npol), slice((i + 1) * npol, (i + 2) * npol)
        h[a, b] = t[i, i + 1] * link
        h[b, a] = h[a, b].conj().T

    # Prepare the Gaussian in one reference bright polariton and parallel-
    # transport that electronic state into every local LDR frame.
    chi = np.exp(-(r - R0) ** 2 / (4 * SIGMA**2))
    chi /= np.linalg.norm(chi)
    psi0 = chi[:, None] * transported_state(links, i0, root)
    weight = np.vdot(psi0, psi0).real
    psi0 = (psi0 / np.sqrt(weight)).ravel()

    # Exact time propagation within the finite LDR Hamiltonian eigenbasis.
    eig, vec = scipy.linalg.eigh(hermitian(h))
    times = np.arange(0.0, TMAX + 0.5 * DTOUT, DTOUT)
    c0 = vec.conj().T @ psi0
    psi = (vec @ (c0[:, None] * np.exp(-1j * eig[:, None] * times / au2fs))).T
    psi = psi.reshape(len(times), nr, npol)
    prob = abs(psi) ** 2
    density = prob.sum(axis=2)
    norm = density.sum(axis=1)
    return {
        "root": root,
        "times": times,
        "energy": energy,
        "bright": bright,
        "mean_r": (density * r).sum(axis=1) / norm,
        "population": prob.sum(axis=1),
        "return_probability": abs(psi.reshape(len(times), -1) @ psi0.conj()) ** 2,
        "density": density,
        "norm": norm,
        "projection_weight": weight,
        "min_link_sv": min(scipy.linalg.svdvals(link).min() for link in links),
    }


# -----------------------------------------------------------------------------
# Save numerical results and diagnostic figures
# -----------------------------------------------------------------------------


def plot(results, data):
    """Plot the bright PES and the principal dynamical observables."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.2))
    ax = ax.ravel()
    for gauge, result in results.items():
        root, color = result["root"], COLOR[gauge]
        label = f"{LABEL[gauge]} ($P_{root}$)"
        e = result["energy"]
        ax[0].plot(data["r"], (e[:, root] - e[:, 0].min()) * au2ev, color=color, label=label)
        ax[1].plot(result["times"], result["mean_r"], color=color)
        ax[2].plot(result["times"], result["population"][:, root], color=color)
        ax[3].plot(result["times"], result["return_probability"], color=color)
    ax[0].set(xlabel=r"Li--F distance $R$ (Angstrom)", ylabel="bright PES (eV)")
    ax[1].set(xlabel="time (fs)", ylabel=r"$\langle R\rangle$ (Angstrom)")
    ax[2].set(xlabel="time (fs)", ylabel="bright-root population", ylim=(-0.02, 1.02))
    ax[3].set(xlabel="time (fs)", ylabel="return probability", ylim=(-0.02, 1.02))
    ax[0].legend(frameon=False, ncol=2)
    for letter, axis in zip("abcd", ax):
        axis.text(-0.14, 1.03, letter, transform=axis.transAxes, fontweight="bold")
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTPUT.with_suffix(".pdf"))
    fig.savefig(OUTPUT.with_suffix(".png"), dpi=350)
    plt.close(fig)


def save(results, data):
    """Write full arrays, a compact JSON diagnostic, and PDF/PNG figures."""
    out = {"r": data["r"], "gauges": np.asarray(GAUGES)}
    summary = {}
    for gauge, result in results.items():
        for key, value in result.items():
            out[f"{gauge}_{key}"] = value
        summary[gauge] = {
            "bright_root": int(result["root"]),
            "projection_weight": float(result["projection_weight"]),
            "max_norm_error": float(max(abs(result["norm"] - 1))),
            "minimum_link_singular_value": float(result["min_link_sv"]),
        }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT.with_suffix(".npz"), **out)
    OUTPUT.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    plot(results, data)
    print(json.dumps(summary, indent=2))


def main():
    # The scan dominates runtime, so reuse it unless REBUILD is requested.
    if REBUILD or not CACHE.exists():
        base = load_driver("lif_base", DRIVERS / "make_lif_casscf_cavity_demo.py")
        glg = load_driver("lif_glg", DRIVERS / "make_lif_casci_glg.py")
        data = scan(base, glg)
    else:
        data = load_cache(CACHE)
        print(f"[cache] loaded {CACHE}")
    results = {gauge: propagate(gauge, data) for gauge in GAUGES}
    save(results, data)


if __name__ == "__main__":
    main()
