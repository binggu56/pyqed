"""Simple geometry minimizers for MD setup."""

import numpy as np


def steepest_descent(
    atoms,
    steps=100,
    max_step=0.01,
    fmax=1e-4,
    max_line_search=12,
):
    """Minimize an ``Atoms`` object with backtracking steepest descent.

    Parameters are in atomic units.  ``max_step`` is the largest per-iteration
    atomic displacement, and ``fmax`` is the convergence threshold for the
    largest atomic force norm.
    """
    steps = int(steps)
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    max_step = float(max_step)
    if max_step <= 0.0:
        raise ValueError("max_step must be positive.")
    fmax = float(fmax)
    energy = atoms.get_potential_energy()
    history = []

    converged = False
    for step in range(steps):
        forces = atoms.get_forces()
        force_norms = np.linalg.norm(forces, axis=1)
        max_force = float(np.max(force_norms)) if len(force_norms) else 0.0
        history.append((step, energy, max_force))
        if max_force <= fmax:
            converged = True
            break

        old_positions = atoms.get_positions()
        direction = forces / max_force
        trial_step = max_step
        accepted = False
        for _ in range(max_line_search):
            atoms.set_positions(old_positions + trial_step * direction)
            trial_energy = atoms.get_potential_energy()
            if np.isfinite(trial_energy) and trial_energy <= energy:
                energy = trial_energy
                accepted = True
                break
            trial_step *= 0.5

        if not accepted:
            atoms.set_positions(old_positions)
            break

    forces = atoms.get_forces()
    force_norms = np.linalg.norm(forces, axis=1)
    max_force = float(np.max(force_norms)) if len(force_norms) else 0.0
    converged = converged or max_force <= fmax
    return {
        "steps": len(history),
        "energy": atoms.get_potential_energy(),
        "fmax": max_force,
        "converged": converged,
        "history": history,
    }


def soft_relaxation(
    atoms,
    stages=((0.1, 0.1, 20), (0.5, 0.5, 20), (1.0, 1.0, 20)),
    max_step=0.01,
    fmax=1e-4,
):
    """Relax rough solvent packings by gradually turning on nonbonded terms.

    Each stage is ``(charge_scale, lj_scale, steps)``.  Scaling is applied to
    the attached molecular-mechanics calculator and restored before return.
    """
    calc = getattr(atoms, "calc", None)
    if calc is None:
        raise RuntimeError("soft_relaxation requires an attached calculator.")

    original_charges = None if getattr(calc, "charges", None) is None else calc.charges.copy()
    original_lj = None if getattr(calc, "lj_epsilon", None) is None else calc.lj_epsilon.copy()
    results = []
    try:
        for stage_index, (charge_scale, lj_scale, steps) in enumerate(stages):
            if original_charges is not None:
                calc.charges = original_charges * float(charge_scale)
            if original_lj is not None:
                calc.lj_epsilon = original_lj * float(lj_scale)
            result = steepest_descent(atoms, steps=steps, max_step=max_step, fmax=fmax)
            result.update(
                {
                    "stage": stage_index,
                    "charge_scale": float(charge_scale),
                    "lj_scale": float(lj_scale),
                }
            )
            results.append(result)
    finally:
        if original_charges is not None:
            calc.charges = original_charges
        if original_lj is not None:
            calc.lj_epsilon = original_lj
    return results


def write_minimization_log(filename, histories):
    """Write minimization/relaxation histories to a text file."""
    with open(filename, "w") as handle:
        handle.write("stage step energy fmax charge_scale lj_scale\n")
        for stage in _as_stage_list(histories):
            charge_scale = stage.get("charge_scale", 1.0)
            lj_scale = stage.get("lj_scale", 1.0)
            for step, energy, fmax_value in stage.get("history", []):
                handle.write(
                    f"{stage.get('stage', 0)} {step} {energy:.12e} "
                    f"{fmax_value:.12e} {charge_scale:.8f} {lj_scale:.8f}\n"
                )


def _as_stage_list(histories):
    if isinstance(histories, dict):
        return [histories]
    return list(histories)
