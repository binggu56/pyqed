"""Shared scheduling and stopping policy for reduced-symmetry DMRG.

Finite-system sweep control follows Schollwoeck, Ann. Phys. 326, 96 (2011),
doi:10.1016/j.aop.2010.09.012. This is an adaptation: energy/metric stopping
is a numerical criterion, not a guarantee of the global variational minimum.
Symmetry-specific compiled executors retain ownership of bond operations.
"""
from numbers import Integral
import math
from .linalg import normalize_max_bond_mode


class SweepControl:
    """Common control for full-cycle and half-sweep executors.

    ``steps`` counts executor transactions. A spatial transaction is a full
    LR/RL cycle; a spin-only transaction is a half sweep. The caller supplies
    the convergence boundary and the energy difference between like gauges.
    """
    def __init__(self, steps, *, algorithm='two_site', tolerance=None,
                 min_steps=1, max_bond_mode=None, default_bond_mode='reduced'):
        if not isinstance(steps, Integral) or isinstance(steps, bool) or steps < 1:
            raise ValueError('nsweeps must be a positive integer.')
        if not isinstance(min_steps, Integral) or isinstance(min_steps, bool) or not 1 <= min_steps <= steps:
            raise ValueError('min_sweeps must lie between one and nsweeps.')
        if algorithm not in ('two_site', 'one_site', 'one_site_cbe'):
            raise ValueError('algorithm must be two_site, one_site, or one_site_cbe.')
        if tolerance is not None and (not math.isfinite(tolerance) or tolerance < 0):
            raise ValueError('Convergence tolerance must be finite and nonnegative.')
        self.steps = range(steps)
        self.algorithm = algorithm
        self.tolerance = None if tolerance == 0 else tolerance
        self.min_steps = min_steps
        self.max_bond_mode = normalize_max_bond_mode(max_bond_mode, default=default_bond_mode)

    def finish(self, record, *, step, energy_delta=None, metric=None,
               boundary=True, local_converged=True):
        """Record a stopping decision; every supplied criterion must pass."""
        record.update(algorithm=self.algorithm, max_bond_mode=self.max_bond_mode)
        record.setdefault("energy_delta", energy_delta)
        criteria = [('energy_delta', energy_delta), ('metric', metric)]
        criteria = [(name, value) for name, value in criteria if value is not None]
        passed = bool(boundary and step >= self.min_steps and local_converged
                      and self.tolerance is not None and criteria
                      and all(math.isfinite(value) and abs(value) <= self.tolerance
                              for _, value in criteria))
        record.pop('converged', None)
        record.pop('convergence_metric', None)
        if passed:
            record['converged'] = True
            record['convergence_metric'] = '+'.join(name for name, _ in criteria)
        return passed

    def diagnostics(self):
        return dict(algorithm=self.algorithm, max_bond_mode=self.max_bond_mode,
                    sweep_control='shared', convergence_tolerance=self.tolerance)
