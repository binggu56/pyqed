#!/usr/bin/env python3
"""Plot spin-pure singlet direct-product grid convergence for H3+ dynamics."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


output = Path("/private/tmp/h3plus_fci_augccpvdz_physical_singlets")
files = {
    13: output / "h3plus_fci_physical_direct_13.npz",
    15: output / "h3plus_fci_physical_direct_15.npz",
}


def main():
    data = {size: np.load(path) for size, path in files.items()}
    colors = {13: "#E69F00", 15: "#009E73"}
    metrics = {}
    for size, values in data.items():
        populations = values["populations"]
        survival = values["norms"]
        conditional = populations / survival[:, None]
        time = values["time_fs"]
        peak = int(np.argmax(populations[:, 0]))
        metrics[str(size)] = {
            "peak_S1_population": float(populations[peak, 0]),
            "peak_S1_time_fs": float(time[peak]),
            "final_survival": float(survival[-1]),
            "final_conditional_populations": conditional[-1].tolist(),
            "maximum_edge_probability": float(
                np.max(values["edge_probability"])
            ),
        }
    for coarse, fine in ((13, 15),):
        left, right = data[coarse], data[fine]
        left_conditional = left["populations"] / left["norms"][:, None]
        right_conditional = right["populations"] / right["norms"][:, None]
        metrics[f"{coarse}_to_{fine}"] = {
            "maximum_absolute_population_change": float(
                np.max(np.abs(left["populations"] - right["populations"]))
            ),
            "maximum_conditional_population_change": float(
                np.max(np.abs(left_conditional - right_conditional))
            ),
            "maximum_survival_change": float(
                np.max(np.abs(left["norms"] - right["norms"]))
            ),
        }
    (output / "h3plus_fci_physical_grid_convergence.json").write_text(
        json.dumps(metrics, indent=2) + "\n"
    )

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure, panels = plt.subplots(1, 3, figsize=(9.3, 2.8), constrained_layout=True)
    for size, values in data.items():
        time = values["time_fs"]
        populations = values["populations"]
        survival = values["norms"]
        color = colors[size]
        label = fr"${size}^3$"
        panels[0].plot(time, populations[:, 0], color=color, label=label)
        panels[1].plot(
            time,
            populations[:, 0] / survival,
            color=color,
            label=label,
        )
        panels[2].plot(time, survival, color=color, label=label)
    panels[0].set(
        xlabel="time (fs)", ylabel=r"absolute $S_1$ population",
        title=r"(a) $S_2\rightarrow S_1$ transfer", ylim=(-0.02, 0.36),
    )
    panels[1].set(
        xlabel="time (fs)", ylabel=r"$P(S_1\mid\mathrm{survival})$",
        title="(b) Conditional branching", ylim=(-0.02, 1.02),
    )
    panels[2].set(
        xlabel="time (fs)", ylabel="survival probability",
        title="(c) Outgoing CAP flux", ylim=(-0.02, 1.02),
    )
    for panel in panels:
        panel.legend(frameon=False)
        panel.spines[["top", "right"]].set_visible(False)
        panel.tick_params(direction="out")
    path = output / "h3plus_fci_physical_grid_convergence"
    figure.savefig(path.with_suffix(".pdf"))
    figure.savefig(path.with_suffix(".png"), dpi=360)
    plt.close(figure)
    print(json.dumps(metrics, indent=2), flush=True)
    print(path.with_suffix(".png"), flush=True)


if __name__ == "__main__":
    main()
