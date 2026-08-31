#!/usr/bin/env python3

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pyscf import gto, scf, tdscf

from pyqed.qchem import Molecule, RTTDHF
from pyqed.qchem.hf import RHF


def _gaussian_smooth(y, sigma_bins=3.0):
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")


def main():
    atom = "H 0 0 0; H 0 0 1.4"

    mol = Molecule(atom=atom, unit="bohr", basis="sto-3g")
    mol.build()
    mf = RHF(mol).run()

    pyscf_mol = gto.M(atom=atom, unit="Bohr", basis="sto-3g")
    pyscf_mf = scf.RHF(pyscf_mol).run()
    ref_root = tdscf.TDHF(pyscf_mf).kernel(nstates=1)[0][0]

    dt = 0.02
    nsteps = 1000
    rt = RTTDHF(mf).run(
        dt=dt,
        nsteps=nsteps,
        store_dm=False,
        kick={"strength": 1e-4, "axis": "z"},
    )

    signal = rt.dipoles[:, 2] - rt.dipoles[0, 2]
    window = np.hanning(signal.size)
    nfft = 8 * signal.size

    spectrum = np.abs(np.fft.rfft(signal * window, n=nfft))
    freq = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)
    spectrum_smooth = _gaussian_smooth(spectrum, sigma_bins=4.0)

    mask = freq <= 2.0
    freq = freq[mask]
    spectrum = spectrum[mask]
    spectrum_smooth = spectrum_smooth[mask]

    fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
    ax.plot(freq, spectrum, lw=0.9, alpha=0.25, color="C0", label="Raw FFT magnitude")
    ax.plot(freq, spectrum_smooth, lw=2.0, color="C0", label="Smoothed spectrum")
    ax.axvline(ref_root, color="k", ls="--", lw=1.2, label=f"PySCF TDHF root = {ref_root:.4f} Ha")
    ax.set_xlim(0.0, 2.0)
    ax.set_xlabel("Angular frequency (Ha)")
    ax.set_ylabel(r"$|\mathcal{F}[\Delta\mu_z(t)]|$")
    ax.set_title(r"H$_2$ / STO-3G RT-TDHF small-kick spectrum")
    ax.legend(frameon=False)
    fig.tight_layout()

    out = "examples/qchem/rttdhf_h2_kick_spectrum.png"
    fig.savefig(out, bbox_inches="tight")
    print(out)
    print(f"PySCF TDHF root = {ref_root:.8f} Ha")


if __name__ == "__main__":
    main()
