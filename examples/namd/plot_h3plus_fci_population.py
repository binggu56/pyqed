"""Plot the first 5 fs of the stored H3+ FCI TNLDR benchmark."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pyqed.units import au2fs


stem = "h3plus_fci_augccpvdz_3d_s3_mace_ftt_vs_direct_7x7x7_20fs"
output = Path("/private/tmp") / stem
data = np.load(output / f"{stem}.npz")
time = data["times"] * au2fs
keep = time <= 5.0 + 1.0e-12

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.6,
        "savefig.bbox": "tight",
    }
)
figure, panel = plt.subplots(figsize=(5.0, 3.4), constrained_layout=True)
for state, color in enumerate(("#0072B2", "#D55E00")):
    physical = state + 1
    panel.plot(
        time[keep], data["direct_populations"][keep, state],
        color=color, linewidth=2.5, alpha=0.55,
        label=fr"Direct $S_{physical}$",
    )
    panel.plot(
        time[keep], data["tnldr_populations"][keep, state], "--",
        color=color, linewidth=1.35, label=fr"TNLDR $S_{physical}$",
    )
panel.set(
    xlim=(0.0, 5.0), ylim=(-0.025, 1.025),
    xlabel="Time (fs)", ylabel="Adiabatic population",
)
panel.grid(alpha=0.2, linewidth=0.6)
panel.legend(ncol=2, frameon=False)

figure_path = output / f"{stem}_population_0_5fs"
figure.savefig(figure_path.with_suffix(".pdf"))
figure.savefig(figure_path.with_suffix(".png"), dpi=350)
print(figure_path.with_suffix(".pdf"))
print(figure_path.with_suffix(".png"))
