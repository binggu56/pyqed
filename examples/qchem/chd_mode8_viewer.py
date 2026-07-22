"""Create a PyQED Viewer animation for CHD normal mode 8."""

import os
from pathlib import Path

import numpy as np

from pyqed.visualization import view


MODE_FILE = Path("chd_c2_b3lyp_augccpvdz_normal_modes.npz")
OUTPUT = Path("chd_c2_mode8_pyqed_viewer.html")
MODE_LABEL = 8
VIEWER_URL = os.environ.get("PYQED_VIEWER_URL", "https://pyqed.org/viewer")


class SavedMolecule:
    def __init__(self, symbols, coordinates_angstrom):
        self._symbols = list(symbols)
        self._coordinates = np.asarray(coordinates_angstrom, dtype=float)

    def atom_symbols(self):
        return self._symbols

    def atom_coords(self):
        return self._coordinates


class SavedVibrations:
    def __init__(self, data):
        self.mol = SavedMolecule(data["symbols"], data["coordinates_angstrom"])
        self._analysis = {
            "modes": data["normal_modes"],
            "freq_cm1": data["frequencies_cm1"],
        }

    def vibrational_analysis(self):
        return self._analysis


def main():
    with np.load(MODE_FILE) as data:
        vibrations = SavedVibrations(data)

    # PyQED uses zero-based array indices, whereas our chemistry mode labels
    # are one-based.
    scene = view(
        vibrations,
        mode=MODE_LABEL - 1,
        amplitude=0.45,
        frames=36,
        interval=40,
        coordinates_unit="angstrom",
        title="CHD mode 8: 730.8 cm^-1 (C2 symmetry A)",
        labels=True,
        open_browser=False,
    )
    launcher = scene._launcher_html(full_page=True)
    if VIEWER_URL != scene.viewer_url:
        # The checked-out frontend may be newer than the deployed viewer. Keep
        # PyQED's validated scene/launcher generation, but point the iframe and
        # postMessage origin at a locally served copy when requested.
        local_origin = VIEWER_URL.split("/viewer", 1)[0]
        launcher = launcher.replace(scene.viewer_url, VIEWER_URL)
        launcher = launcher.replace("https://pyqed.org", local_origin)
    OUTPUT.write_text(launcher, encoding="utf-8")
    print(OUTPUT.resolve())


if __name__ == "__main__":
    main()
