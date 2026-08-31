#!/usr/bin/env python3

import numpy as np

import ultraplot as plt 

from pyqed.qchem import Molecule, RTTDHF
from pyqed.qchem.hf import RHF


def _gaussian_smooth(y, sigma_bins=3.0):
    radius = max(1, int(np.ceil(4.0 * sigma_bins)))
    x = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (x / sigma_bins) ** 2)
    kernel /= kernel.sum()
    return np.convolve(y, kernel, mode="same")




    # window = np.hanning(signal.size)
    # nfft = 8 * signal.size

    # spectrum = np.abs(np.fft.rfft(signal * window, n=nfft))
    # freq = 2.0 * np.pi * np.fft.rfftfreq(nfft, d=dt)
    # spectrum_smooth = _gaussian_smooth(spectrum, sigma_bins=4.0)

    # mask = freq <= 2.0
    # freq = freq[mask]
    # spectrum = spectrum[mask]
    # spectrum_smooth = spectrum_smooth[mask]

    # fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    # ax.plot(freq, spectrum, lw=0.8, alpha=0.20, color="C0", label="Raw FFT magnitude")
    # ax.plot(freq, spectrum_smooth, lw=2.0, color="C0", label="Smoothed spectrum")

    # for i, root in enumerate(roots[:4]):
    #     if root <= 2.0:
    #         ax.axvline(
    #             root,
    #             color=f"C{i+1}",
    #             ls="--",
    #             lw=1.1,
    #             label=f"PySCF TDHF root {i+1} = {root:.4f} Ha",
    #         )

    # ax.set_xlim(0.0, 2.0)
    # ax.set_xlabel("Angular frequency (Ha)")
    # ax.set_ylabel(r"$|\mathcal{F}[\Delta\mu_z(t)]|$")
    # ax.set_title(r"HF / STO-3G RT-TDHF small-kick spectrum")
    # ax.legend(frameon=False, fontsize=8)
    # fig.tight_layout()

    # out = "examples/qchem/rttdhf_hf_kick_spectrum.png"
    # fig.savefig(out, bbox_inches="tight")
    # print(out)
    # print("PySCF TDHF roots (Ha):", roots[:5])


atom = "H 0 0 0; F 0 0 1.75"

mol = Molecule(atom=atom, unit="bohr", basis="6311g")
mol.build()
mf = RHF(mol).run()


dt = 0.02
nsteps = 4000
rt = RTTDHF(mf).run(
    dt=dt,
    nsteps=nsteps,
    store_dm=False,
    kick={"strength": 1e-4, "axis": "z"},
)

signal = rt.dipoles[:, 2] - rt.dipoles[0, 2]


fig, ax = plt.subplots()
ax.plot(rt.times, signal)

