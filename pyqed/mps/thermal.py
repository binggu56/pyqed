"""Purification-based finite-temperature matrix-product states."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral

import numpy as np

from pyqed.lattice import CompositeSite, Site
from pyqed.mps.mps import MPS, dense_to_symmetric_mpo, expect_mps
from pyqed.mps.tdmps import TDMPS
from pyqed.mps.tdvp import _block_sparse_site_qn_maps
from pyqed.tn import Hamiltonian, MPO


def _physical_mpo(operator) -> MPO:
    if isinstance(operator, Hamiltonian):
        return operator.to_mpo()
    if isinstance(operator, MPO):
        return operator
    raise TypeError("operator must be a Hamiltonian or MPO.")


def _purification_sites(sites, *, conjugate_charges=False):
    physical_sites = tuple(sites)
    if not physical_sites:
        raise ValueError("a purified MPS requires at least one physical site.")
    if any(not isinstance(site, Site) for site in physical_sites):
        raise TypeError("sites must contain canonical Site objects.")

    auxiliary_sites = []
    for site in physical_sites:
        charges = None
        charge_labels = None
        if conjugate_charges:
            if site.charges is None:
                raise ValueError(
                    "U(1) purification requires charges on every physical site."
                )
            charges = tuple(
                tuple(-component for component in charge)
                for charge in site.charges
            )
            charge_labels = site.charge_labels
        auxiliary_sites.append(
            Site(
                labels=tuple(f"aux:{label}" for label in site.labels),
                charges=charges,
                charge_labels=charge_labels,
                parities=site.parities,
                statistics=site.statistics,
                name=f"aux({site.name})",
            )
        )
    auxiliary_sites = tuple(auxiliary_sites)
    composite_sites = tuple(
        CompositeSite((physical, auxiliary), name=f"{physical.name} x aux")
        for physical, auxiliary in zip(physical_sites, auxiliary_sites)
    )
    return physical_sites, auxiliary_sites, composite_sites


def infinite_temperature_mps(sites, *, conjugate_charges=False) -> MPS:
    r"""Return the normalized local thermofield-double state at :math:`\beta=0`.

    Physical and auxiliary indices are fused locally in ``(physical, aux)``
    order, so this state has composite dimension ``d**2`` and virtual bond
    dimension one at every site.
    """

    _physical, _auxiliary, composite_sites = _purification_sites(
        sites,
        conjugate_charges=conjugate_charges,
    )
    tensors = []
    for composite in composite_sites:
        physical_dim, auxiliary_dim = composite.factor_dims
        if physical_dim != auxiliary_dim:
            raise ValueError("physical and auxiliary dimensions must match.")
        bell = np.zeros(physical_dim * auxiliary_dim, dtype=complex)
        for state in range(physical_dim):
            bell[composite.flatten((state, state))] = 1.0 / np.sqrt(physical_dim)
        tensors.append(bell.reshape(1, composite.dim, 1))
    return MPS(tensors, sites=composite_sites)


def lift_physical_mpo(mpo, *, auxiliary_sites=None) -> MPO:
    r"""Lift a physical MPO to purification space as :math:`W\otimes I_a`."""

    mpo = _physical_mpo(mpo)
    physical_sites = tuple(mpo.sites)
    if auxiliary_sites is None:
        _physical, auxiliary_sites, composite_sites = _purification_sites(
            physical_sites
        )
    else:
        auxiliary_sites = tuple(auxiliary_sites)
        if len(auxiliary_sites) != len(physical_sites):
            raise ValueError("auxiliary_sites must match the MPO length.")
        if any(not isinstance(site, Site) for site in auxiliary_sites):
            raise TypeError("auxiliary_sites must contain canonical Site objects.")
        if any(
            physical.dim != auxiliary.dim
            for physical, auxiliary in zip(physical_sites, auxiliary_sites)
        ):
            raise ValueError("every auxiliary dimension must match its physical site.")
        composite_sites = tuple(
            CompositeSite((physical, auxiliary), name=f"{physical.name} x aux")
            for physical, auxiliary in zip(physical_sites, auxiliary_sites)
        )

    tensors = []
    for tensor, physical, auxiliary in zip(
        mpo.tensors, physical_sites, auxiliary_sites
    ):
        tensor = np.asarray(tensor)
        identity = np.eye(auxiliary.dim, dtype=np.result_type(tensor, float))
        lifted = np.einsum(
            "lrpq,ab->lrpaqb",
            tensor,
            identity,
            optimize=True,
        ).reshape(
            tensor.shape[0],
            tensor.shape[1],
            physical.dim * auxiliary.dim,
            physical.dim * auxiliary.dim,
        )
        tensors.append(lifted)
    return MPO(tensors, sites=composite_sites)


def _u1_local_sectors(composite_sites):
    tables = []
    labels = None
    for site in composite_sites:
        if site.charges is None:
            raise ValueError("U(1) purification requires charged composite sites.")
        if labels is None:
            labels = site.charge_labels
        elif labels != site.charge_labels:
            raise ValueError("all sites must use the same charge labels.")
        tables.append(tuple(tuple(charge) for charge in site.charges))
    if any(table != tables[0] for table in tables[1:]):
        raise NotImplementedError(
            "block-sparse thermal TDVP currently requires identical local "
            "charge tables on every composite site."
        )
    rank = len(tables[0][0])
    if rank == 1:
        return [charge[0] for charge in tables[0]], 0
    return list(tables[0]), tuple(0 for _ in range(rank))


def _require_charge_conserving_mpo(mpo, physical_sites, *, atol=1.0e-13):
    """Reject MPO paths carrying nonzero net additive charge."""

    charge_tables = []
    rank = None
    for site in physical_sites:
        if site.charges is None:
            raise ValueError("U(1) purification requires charges on every site.")
        table = tuple(tuple(charge) for charge in site.charges)
        if rank is None:
            rank = len(table[0])
        if any(len(charge) != rank for charge in table):
            raise ValueError("all physical charges must have the same rank.")
        charge_tables.append(table)

    zero = (0,) * rank
    reachable = {(0, zero)}
    for tensor, charges in zip(mpo.tensors, charge_tables):
        tensor = np.asarray(tensor)
        following = set()
        nonzero = np.argwhere(np.abs(tensor) > float(atol))
        by_left = {}
        for left, right, out_state, in_state in nonzero:
            by_left.setdefault(int(left), []).append(
                (int(right), int(out_state), int(in_state))
            )
        for left, flux in reachable:
            for right, out_state, in_state in by_left.get(left, ()):
                delta = tuple(
                    charges[out_state][component] - charges[in_state][component]
                    for component in range(rank)
                )
                following.add(
                    (right, tuple(a + b for a, b in zip(flux, delta)))
                )
        reachable = following
    nonconserving = sorted(flux for right, flux in reachable if right == 0 and flux != zero)
    if nonconserving:
        raise ValueError(
            "symmetry='U1' requires a charge-conserving Hamiltonian; "
            f"found net MPO charge flux {nonconserving[0]}."
        )


class PurifiedMPS:
    r"""Finite-temperature MPS obtained from thermofield purification.

    The normalized infinite-temperature state is evolved according to

    .. math::

        |\Psi(\beta)\rangle =
        (e^{-\beta H/2}\otimes I_a)|\Psi(0)\rangle.

    Two-site TDVP is the default because it can grow the initially unit virtual
    bonds.  ``state`` stores the normalized purified MPS; normalization factors
    accumulated during evolution provide ``log_partition_function``.
    """

    def __init__(
        self,
        hamiltonian,
        *,
        D=64,
        cutoff=1.0e-12,
        integrator="tdvp2",
        symmetry=None,
        krylov_dim=12,
        krylov_tol=1.0e-13,
        krylov_method="lanczos",
    ):
        self.physical_mpo = _physical_mpo(hamiltonian)
        if isinstance(D, bool) or not isinstance(D, Integral) or int(D) < 1:
            raise ValueError("D must be a positive integer.")
        self.D = int(D)
        self.cutoff = float(cutoff)
        if not np.isfinite(self.cutoff) or self.cutoff < 0.0:
            raise ValueError("cutoff must be finite and nonnegative.")
        self.integrator = str(integrator)
        self.krylov_dim = int(krylov_dim)
        self.krylov_tol = float(krylov_tol)
        self.krylov_method = str(krylov_method)

        symmetry_key = "none" if symmetry is None else str(symmetry).lower()
        if symmetry_key in {"none", "dense", "false"}:
            self.symmetry = None
        elif symmetry_key in {"u1", "u(1)", "abelian"}:
            self.symmetry = "U1"
        else:
            raise ValueError("symmetry must be None or 'U1'.")

        (
            self.physical_sites,
            self.auxiliary_sites,
            self.composite_sites,
        ) = _purification_sites(
            self.physical_mpo.sites,
            conjugate_charges=self.symmetry == "U1",
        )
        self.purified_mpo = lift_physical_mpo(
            self.physical_mpo,
            auxiliary_sites=self.auxiliary_sites,
        )
        self.state = infinite_temperature_mps(
            self.physical_sites,
            conjugate_charges=self.symmetry == "U1",
        )
        self.state = MPS(
            [tensor.copy() for tensor in self.state.factors],
            sites=self.composite_sites,
        )
        self.local_sectors = None
        self.target_sector = None
        if self.symmetry == "U1":
            _require_charge_conserving_mpo(
                self.physical_mpo,
                self.physical_sites,
            )
            self.local_sectors, self.target_sector = _u1_local_sectors(
                self.composite_sites
            )
        self._lifted_observable_cache = {
            id(self.physical_mpo): self.purified_mpo,
        }
        self._block_observable_cache = {}
        self.beta = 0.0
        self.log_norm_squared = 0.0
        self.history = []
        self.energy = float(np.real(self.expectation(self.physical_mpo)))
        self.thermal_energy = self.energy
        self.time = 0.0
        self.real_time_history = []
        self.success = True
        self.message = "initialized at infinite temperature"
        self.tdmps = self._make_tdmps(self.purified_mpo)
        self.evolution_mpo = self.physical_mpo
        self.evolution_tdmps = self.tdmps

    def _make_tdmps(self, lifted_mpo):
        return TDMPS(
            lifted_mpo,
            D=self.D,
            cutoff=self.cutoff,
            local_sectors=self.local_sectors,
            target_sector=self.target_sector,
            tdvp_projection_backend=(
                "block-sparse" if self.symmetry == "U1" else None
            ),
        )

    def _adopt_state(self, state):
        if state.factors and hasattr(state.factors[0], "qns"):
            self.state = MPS(
                [tensor.copy() for tensor in state.factors],
                labels=["lv", "rv", "p"],
                sites=self.composite_sites,
            )
        else:
            self.state = MPS(
                [np.asarray(tensor).copy() for tensor in state.factors],
                sites=self.composite_sites,
            )

    @property
    def log_partition_function(self):
        r"""Return :math:`\log Z(\beta)`, including the infinite-T dimension."""

        log_dimension = sum(np.log(site.dim) for site in self.physical_sites)
        return float(log_dimension + self.log_norm_squared)

    @property
    def free_energy(self):
        """Return the Helmholtz free energy at the current inverse temperature."""

        if self.beta == 0.0:
            return -np.inf
        return float(-self.log_partition_function / self.beta)

    @property
    def bond_dims(self):
        """Return all purified-MPS bond dimensions, including boundaries."""

        return (1,) + tuple(self.state.get_bond_dimensions()) + (1,)

    def expectation(self, operator):
        """Return a normalized thermal expectation of a physical operator."""

        physical = _physical_mpo(operator)
        if physical.dims != self.physical_mpo.dims:
            raise ValueError("operator physical dimensions do not match the state.")
        cache_key = id(physical)
        lifted = self._lifted_observable_cache.get(cache_key)
        if lifted is None:
            lifted = lift_physical_mpo(
                physical,
                auxiliary_sites=self.auxiliary_sites,
            )
            self._lifted_observable_cache[cache_key] = lifted
        factors = lifted.factors
        if self.state.factors and hasattr(self.state.factors[0], "qns"):
            factors = self._block_observable_cache.get(cache_key)
            if factors is None:
                site_qn_maps, _target_qn = _block_sparse_site_qn_maps(
                    self.local_sectors,
                    self.state.L,
                    tuple(site.dim for site in self.composite_sites),
                    self.target_sector,
                )
                factors = dense_to_symmetric_mpo(
                    lifted.factors,
                    site_qn_maps,
                    native_site_storage=True,
                )
                self._block_observable_cache[cache_key] = factors
        numerator = expect_mps(
            self.state.factors,
            factors,
        )
        denominator = self.state.norm_squared()
        if abs(denominator) <= np.finfo(float).tiny:
            raise ValueError("cannot evaluate an expectation for a zero state.")
        return np.real_if_close(numerator / denominator)

    def run(self, beta, *, step=0.05, verbose=False):
        r"""Evolve from the current inverse temperature to ``beta``.

        Each increment applies :math:`e^{-\Delta\beta H/2}` to the physical
        legs. Evolution toward a
        smaller beta is intentionally rejected; construct a new object to
        restart from infinite temperature.
        """

        if self.time != 0.0:
            raise ValueError(
                "imaginary-time preparation cannot continue after real-time "
                "evolution; construct a new PurifiedMPS instead."
            )
        beta = float(beta)
        step = float(step)
        if not np.isfinite(beta) or beta < self.beta:
            raise ValueError("beta must be finite and not smaller than current beta.")
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive.")

        index = len(self.history)
        beta_tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(beta))
        while beta - self.beta > beta_tolerance:
            increment = min(step, beta - self.beta)
            # TDVP uses exp(-i H dt); dt=-i*increment/2 gives exp(-increment*H/2).
            evolved = self.tdmps.step(
                self.state,
                dt=-0.5j * increment,
                integrator=self.integrator,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
            )
            info = dict(self.tdmps.last_step_info or {})
            norm2 = float(info["pre_normalization_norm2"])
            if not np.isfinite(norm2) or norm2 <= 0.0:
                self.success = False
                self.message = "imaginary-time step produced a nonpositive norm"
                raise FloatingPointError(self.message)
            self.log_norm_squared += float(np.log(norm2))
            self._adopt_state(evolved)
            self.beta = float(self.beta + increment)
            self.energy = float(np.real(self.expectation(self.physical_mpo)))
            self.thermal_energy = self.energy
            row = {
                "step": index,
                "beta": self.beta,
                "energy": self.energy,
                "logZ": self.log_partition_function,
                "max_bond": max(self.bond_dims),
                "truncation_error": float(info.get("truncation_error", 0.0)),
                "backend": info.get("backend", "dense"),
            }
            self.history.append(row)
            if verbose:
                print(
                    f"thermal-MPS step={index:4d} beta={self.beta:.8f} "
                    f"E={self.energy:.12f} D={row['max_bond']} "
                    f"trunc={row['truncation_error']:.3e}"
                )
            index += 1

        if abs(beta - self.beta) <= beta_tolerance:
            self.beta = beta
        self.success = True
        self.message = f"reached beta={self.beta:g}"
        return self

    def evolve(
        self,
        time,
        *,
        step=0.05,
        hamiltonian=None,
        observables=None,
        verbose=False,
    ):
        r"""Evolve the prepared thermal purification in real time.

        The physical part of the purification evolves as

        .. math::

            |\Psi_\beta(t)\rangle =
            (e^{-i H_{\rm rt}t}\otimes I_a)|\Psi_\beta(0)\rangle.

        ``time`` is the target absolute time and ``step`` is the largest time
        increment. Supplying ``hamiltonian`` switches to that physical
        Hamiltonian at the current time, implementing a sudden quench.
        Omitting it continues with the currently active Hamiltonian.

        Real-time evolution leaves ``beta``, ``thermal_energy``, and the
        preparation partition function unchanged. ``energy`` is the
        expectation value of the active real-time Hamiltonian.
        """

        time = float(time)
        step = float(step)
        if not np.isfinite(time) or time < self.time:
            raise ValueError(
                "time must be finite and not smaller than the current time."
            )
        if not np.isfinite(step) or step <= 0.0:
            raise ValueError("step must be finite and positive.")
        if observables is None:
            observables = {}
        elif not isinstance(observables, Mapping):
            raise TypeError("observables must map names to Hamiltonians or MPOs.")
        else:
            observables = dict(observables)

        if hamiltonian is not None:
            physical = _physical_mpo(hamiltonian)
            if physical.dims != self.physical_mpo.dims:
                raise ValueError(
                    "real-time Hamiltonian dimensions do not match the state."
                )
            if self.symmetry == "U1":
                _require_charge_conserving_mpo(physical, self.physical_sites)
            if physical is self.physical_mpo:
                self.evolution_mpo = self.physical_mpo
                self.evolution_tdmps = self.tdmps
            else:
                lifted = lift_physical_mpo(
                    physical,
                    auxiliary_sites=self.auxiliary_sites,
                )
                self.evolution_mpo = physical
                self.evolution_tdmps = self._make_tdmps(lifted)
            self.energy = float(np.real(self.expectation(self.evolution_mpo)))

        index = len(self.real_time_history)
        time_tolerance = 16.0 * np.finfo(float).eps * max(1.0, abs(time))
        while time - self.time > time_tolerance:
            increment = min(step, time - self.time)
            evolved = self.evolution_tdmps.step(
                self.state,
                time=self.time,
                dt=increment,
                integrator=self.integrator,
                krylov_dim=self.krylov_dim,
                krylov_tol=self.krylov_tol,
                krylov_method=self.krylov_method,
            )
            info = dict(self.evolution_tdmps.last_step_info or {})
            norm2 = float(info.get("pre_normalization_norm2", np.nan))
            if not np.isfinite(norm2) or norm2 <= 0.0:
                self.success = False
                self.message = "real-time step produced a nonpositive norm"
                raise FloatingPointError(self.message)
            self._adopt_state(evolved)
            self.time = float(self.time + increment)
            self.energy = float(np.real(self.expectation(self.evolution_mpo)))
            reference_energy = float(
                np.real(self.expectation(self.physical_mpo))
            )
            values = {
                name: np.real_if_close(self.expectation(operator)).item()
                for name, operator in observables.items()
            }
            row = {
                "step": index,
                "time": self.time,
                "energy": self.energy,
                "reference_energy": reference_energy,
                "norm_error": abs(norm2 - 1.0),
                "max_bond": max(self.bond_dims),
                "truncation_error": float(info.get("truncation_error", 0.0)),
                "backend": info.get("backend", "dense"),
                "observables": values,
            }
            self.real_time_history.append(row)
            if verbose:
                print(
                    f"thermal-MPS real-time step={index:4d} "
                    f"t={self.time:.8f} E={self.energy:.12f} "
                    f"D={row['max_bond']} trunc={row['truncation_error']:.3e}"
                )
            index += 1

        if abs(time - self.time) <= time_tolerance:
            self.time = time
        self.success = True
        self.message = f"reached time={self.time:g}"
        return self


__all__ = [
    "PurifiedMPS",
    "infinite_temperature_mps",
    "lift_physical_mpo",
]
