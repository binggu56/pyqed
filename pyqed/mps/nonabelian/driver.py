#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Small driver objects for fixed-layout non-Abelian sweeps.
"""

from __future__ import annotations

from .mps import MPS
from .sweep import run_sweeps
from .tensor import NonabelianTensor


def _as_mps(state):
    return state.copy() if isinstance(state, MPS) else MPS.from_sites(state)


class SweepDriver:
    """
    Minimal stateful driver around :func:`run_sweeps`.

    This intentionally mirrors just the bookkeeping shape of a solver/driver:
    it stores the current site tensors, sweep history, and convergence status,
    while delegating the actual update logic to the functional sweep helpers.
    When solver callbacks report local objectives, the latest sweep-averaged
    values are exposed as ``last_energy`` and ``last_objective_metric``. MPO-
    driven runs also keep the sweep-local trace separately as
    ``last_objective_energy`` while ``last_energy`` reports the true MPO
    expectation value. The same interface also works with a ``local_operator``
    callback/specification that uses the built-in Davidson local solver.
    """

    def __init__(
        self,
        sites,
        *,
        nsweeps=1,
        start_direction="lr",
        alternate=True,
        solver=None,
        local_operator=None,
        mpo_factors=None,
        local_solver_kwargs=None,
        local_solver_schedule=None,
        bond_coupling="left",
        max_bond=None,
        max_bond_mode=None,
        cutoff=1e-10,
        conv_tol=None,
        measure=None,
        prefer_reduced_local_operator=False,
        warm_start_bonds=False,
        mixer_zero_block_noise_scale=0.0,
        mixer_zero_block_noise_seed=None,
        mixer_nsweeps=1,
        record_post_update_energy=False,
        verbose=0,
    ):
        self.initial_mps = _as_mps(sites)
        self.mps = self.initial_mps.copy()
        self.nsweeps = nsweeps
        self.start_direction = start_direction
        self.alternate = alternate
        self.solver = solver
        self.local_operator = local_operator
        self.mpo_factors = mpo_factors
        self.local_solver_kwargs = local_solver_kwargs
        self.local_solver_schedule = local_solver_schedule
        self.bond_coupling = bond_coupling
        self.max_bond = max_bond
        self.max_bond_mode = max_bond_mode
        self.cutoff = cutoff
        self.conv_tol = conv_tol
        self.measure = measure
        self.prefer_reduced_local_operator = prefer_reduced_local_operator
        self.warm_start_bonds = warm_start_bonds
        self.mixer_zero_block_noise_scale = mixer_zero_block_noise_scale
        self.mixer_zero_block_noise_seed = mixer_zero_block_noise_seed
        self.mixer_nsweeps = mixer_nsweeps
        self.record_post_update_energy = record_post_update_energy
        self.verbose = int(verbose)

        self.history = []
        self.converged = False
        self.last_direction = None
        self.ncompleted = 0
        self.last_energy = None
        self.last_objective_energy = None
        self.last_objective_metric = None

    @property
    def sites(self):
        return self.mps.sites

    @sites.setter
    def sites(self, value):
        self.mps = _as_mps(value)

    def reset(self, *, sites=None):
        """
        Reset the stored state before another run.
        """
        if sites is not None:
            self.initial_mps = _as_mps(sites)
        self.mps = self.initial_mps.copy()
        self.history = []
        self.converged = False
        self.last_direction = None
        self.ncompleted = 0
        self.last_energy = None
        self.last_objective_energy = None
        self.last_objective_metric = None
        return self

    def run(self, *, sites=None, **overrides):
        """
        Execute repeated non-Abelian sweeps and store the result on the driver.
        """
        if sites is not None:
            self.reset(sites=sites)
        else:
            self.reset()

        resolved_start_direction = overrides.pop("start_direction", self.start_direction)
        resolved_alternate = overrides.pop("alternate", self.alternate)

        result = run_sweeps(
            self.mps,
            nsweeps=overrides.pop("nsweeps", self.nsweeps),
            start_direction=resolved_start_direction,
            alternate=resolved_alternate,
            solver=overrides.pop("solver", self.solver),
            local_operator=overrides.pop("local_operator", self.local_operator),
            mpo_factors=overrides.pop("mpo_factors", self.mpo_factors),
            local_solver_kwargs=overrides.pop("local_solver_kwargs", self.local_solver_kwargs),
            local_solver_schedule=overrides.pop("local_solver_schedule", self.local_solver_schedule),
            bond_coupling=overrides.pop("bond_coupling", self.bond_coupling),
            max_bond=overrides.pop("max_bond", self.max_bond),
            max_bond_mode=overrides.pop("max_bond_mode", self.max_bond_mode),
            cutoff=overrides.pop("cutoff", self.cutoff),
            conv_tol=overrides.pop("conv_tol", self.conv_tol),
            measure=overrides.pop("measure", self.measure),
            prefer_reduced_local_operator=overrides.pop(
                "prefer_reduced_local_operator",
                self.prefer_reduced_local_operator,
            ),
            warm_start_bonds=overrides.pop(
                "warm_start_bonds",
                self.warm_start_bonds,
            ),
            mixer_zero_block_noise_scale=overrides.pop(
                "mixer_zero_block_noise_scale",
                self.mixer_zero_block_noise_scale,
            ),
            mixer_zero_block_noise_seed=overrides.pop(
                "mixer_zero_block_noise_seed",
                self.mixer_zero_block_noise_seed,
            ),
            mixer_nsweeps=overrides.pop(
                "mixer_nsweeps",
                self.mixer_nsweeps,
            ),
            record_post_update_energy=overrides.pop(
                "record_post_update_energy",
                self.record_post_update_energy,
            ),
            verbose=overrides.pop(
                "verbose",
                self.verbose,
            ),
        )
        if overrides:
            unknown = ", ".join(sorted(overrides))
            raise TypeError(f"Unknown run() override(s): {unknown}")

        self.mps = result.get("mps", MPS.from_sites(result["sites"]))
        self.history = result["history"]
        self.converged = result["converged"]
        self.last_direction = result["last_direction"]
        self.ncompleted = result["ncompleted"]
        if self.history:
            best_energy = result.get("best_energy")
            if best_energy is not None:
                self.last_energy = best_energy
            else:
                self.last_energy = self.history[-1].get("energy")
            self.last_objective_energy = self.history[-1].get("objective_energy")
            if self.last_objective_energy is None:
                self.last_objective_energy = self.history[-1].get("energy")
            self.last_objective_metric = self.history[-1].get("objective_metric")
        else:
            self.last_energy = None
            self.last_objective_energy = None
            self.last_objective_metric = None
        return self

Driver = SweepDriver
