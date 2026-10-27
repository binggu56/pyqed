"""Finite-group intertwiners on fully spin-reduced LETTA/MPS chains.

The spatial virtual leg is the finite group's regular representation. Its
Fourier boundary selects a complete irrep, including every E partner. Free
parameters live in the seed reduced tensors; the regular leg is structural
and is never truncated or optimized independently. All local objectives and
norms include this leg throughout optimization, not as final state repair.

This is a finite-group regular-representation realization of symmetric tensor
networks (Singh, Pfeifer and Vidal, Phys. Rev. A 82, 050301, 2010,
https://doi.org/10.1103/PhysRevA.82.050301). It is a PyQED adaptation, not their
implementation. Spin remains Wigner--Eckart reduced. Spatial rows are carried
explicitly on a structural group leg, rather than optimally compressed with
finite-group F symbols. Orbital gates reuse the reduced Malmqvist/Knecht transforms
in pyqed.mps.nonabelian.orbital_transform. No global determinant basis is constructed.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, null_space
from threadpoolctl import threadpool_limits

from pyqed.qchem.spatial_symmetry import SpatialGroupAction
from pyqed.mps.nonabelian.environment import contract_chain_transition, LocalTransitionPlan, one_site_reduced_adjoint
from pyqed.mps.nonabelian.sweep import _identity_mpo_factors_for_sites_and_mpo
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
from pyqed.mps.nonabelian.mps import MPS
from pyqed.symmetry import IrrepTensor


def sum_reduced_mps(states, coefficients):
    """Exact direct sum on reduced virtual multiplicities, without truncation.

    MPS.__add__ assembles dense site arrays; it cannot preserve the reduced
    sector legs required here. This helper sums each spin sector separately.
    """
    selected = [(list(getattr(state, 'tensors', state)), weight)
                for state, weight in zip(states, coefficients) if abs(weight) > 1e-14]
    if not selected:
        raise ValueError('The group boundary has no nonzero coefficients.')
    length = len(selected[0][0])
    if any(len(chain) != length for chain, _ in selected):
        raise ValueError('MPS lengths differ.')
    bond_dims, offsets = [], []
    for bond in range(length + 1):
        dimensions, starts = {}, []
        for chain, _ in selected:
            leg = chain[0].legs[0] if bond == 0 else chain[bond-1].legs[2]
            starts.append({sector: (0 if bond in (0, length) else dimensions.get(sector, 0))
                           for sector in leg.dims})
            for sector, dim in leg.dims.items():
                dimensions[sector] = max(dim, dimensions.get(sector, 0)) if bond in (0, length) else dimensions.get(sector, 0) + dim
        bond_dims.append(dimensions)
        offsets.append(starts)
    result = []
    for site in range(length):
        data = {}
        for component, (chain, weight) in enumerate(selected):
            for key, block in chain[site].data.items():
                left, physical, right = key
                shape = (bond_dims[site][left], block.shape[1], bond_dims[site+1][right])
                if key not in data:
                    data[key] = np.zeros(shape, dtype=complex)
                a, b = offsets[site][component][left], offsets[site+1][component][right]
                data[key][a:a+block.shape[0], :, b:b+block.shape[2]] += block * (weight if site == 0 else 1)
        qns = [[sector for sector, dim in bond_dims[site].items() for _ in range(dim)],
               selected[0][0][site].qns[1],
               [sector for sector, dim in bond_dims[site+1].items() for _ in range(dim)]]
        result.append(IrrepTensor(data=data, qns=qns, dirs=selected[0][0][site].dirs,
                                  metadata={'physical_basis': 'fully_reduced_su2'}))
    return MPS.from_tensors(result, target_sector=result[-1].qns[2][0])


class SpatialMultipletSweep:
    """Optimize a full spatial multiplet with reduced SU(2) contractions.

    A fixed regular-representation virtual leg carries every group operation
    and a matrix-element boundary carries all target rows. ``D`` bounds seed
    spin multiplicities; the uncompressed structural spatial leg has |G|
    channels. Gates are applied without Schmidt truncation. This preserves
    spatial covariance even at small D, including E partners, at the cost of
    more contraction work than a spatial-CG-compressed implementation.

    The one-site alternating Rayleigh--Ritz optimizer uses exact reduced MPO
    environments for its local generalized eigenproblems. Local parameter
    matrices may be dense; no global/determinant Hamiltonian, final dense
    projection, or primitive branch growth is used. Graph ties remain those
    of the seed ansatz. MPS and LETTA supply distinct local tensor owners.

    References and fidelity: Singh et al., Phys. Rev. A 82, 050301 (2010),
    https://doi.org/10.1103/PhysRevA.82.050301; Malmqvist, Int. J. Quantum
    Chem. 30, 479 (1986), https://doi.org/10.1002/qua.560300404; Knecht et al.,
    JCTC 12, 5881 (2016), https://doi.org/10.1021/acs.jctc.6b00889.
    This regular-leg, graph-tied optimizer is an adaptation, not an exact
    reproduction of their algorithms. It has no global convergence guarantee;
    local dense eigensolves and untruncated orbital gates limit scalability.
    Open boundaries, restricted real invariant Hamiltonians, and a fixed seed
    bond dimension are supported. nstates counts full spatial multiplets.
    """

    def __init__(self, seed, orbital_representations, wfnsym, *, point_group='D3h',
                 h1e=None, eri=None, orthogonal_to=()):
        if seed.virtual_boundary != 'open':
            raise ValueError('Finite spatial multiplets require open reduced boundaries.')
        self.seed = seed
        self.spatial_symmetry = SpatialGroupAction(orbital_representations, wfnsym, point_group=point_group)
        if h1e is None:
            specification = getattr(seed, '_hamiltonian_spec', None)
            if specification is None:
                raise ValueError('Provide Hamiltonian integrals to validate spatial invariance of an existing seed.')
            h1e, eri = specification['h1e'], specification['eri']
        self.spatial_symmetry.validate_hamiltonian(h1e, eri)
        self.wfnsym = self.spatial_symmetry.wfnsym
        self.partner_dimension = self.spatial_symmetry.dimension
        self.nsites, self.D, self.ecore = seed.nsites, seed.D, seed.ecore
        self.n_threads, self.workers = seed.n_threads, 1
        self.mpo = seed.mpo
        self.identity = _identity_mpo_factors_for_sites_and_mpo(seed.materialize(), self.mpo)
        self.orthogonal_to = tuple(orthogonal_to)
        self.history = []
        self.energy = None
        self.e_tot = None
        self.success = self.converged = False
        self.message = 'not run'
        self.states = ()
        self._seed_factory = None
        self.spatial_multiplet_count = 1
        self.nstates = self.partner_dimension
        self.spatial_partner_rows = tuple(range(self.partner_dimension))
        self.diagnostics = {'point_group': self.spatial_symmetry.group.name,
                            'spatial_irrep': self.wfnsym, 'spin_reduced': True,
                            'structural_group_channels': len(self.spatial_symmetry.names)}
        boundary = seed.materialize()
        left_sectors = boundary[0].legs[0].sectors
        right_sectors = boundary[-1].legs[-1].sectors
        charge = (right_sectors[0].charge - left_sectors[0].charge
                  if len(left_sectors) == len(right_sectors) == 1 else None)
        matrices, weights = [], []
        for matrix, weight in zip(self.spatial_symmetry.matrices,
                                  self.spatial_symmetry.weights.transpose(2, 0, 1)):
            match, factor = None, 1
            for i, old in enumerate(matrices):
                pivot = np.unravel_index(np.argmax(abs(old)), old.shape)
                phase = matrix[pivot] / old[pivot] if charge is not None else 1
                if (abs(abs(phase)-1) < 1e-13
                        and np.allclose(matrix, phase*old, atol=1e-13, rtol=0)):
                    match, factor = i, phase**charge if charge is not None else 1
                    break
            if match is None:
                matrices.append(matrix)
                weights.append(weight.astype(complex, copy=True))
            else:
                # On fixed charge N, Gamma(z U) = z**N Gamma(U). Combine
                # equivalent many-body actions, retaining every Fourier row.
                weights[match] += factor * weight
        selected = [i for i, weight in enumerate(weights) if np.max(abs(weight)) > 1e-14]
        self._matrices = tuple(matrices[i] for i in selected)
        self._weights = np.stack([weights[i] for i in selected], axis=-1) if selected else np.zeros(
            (self.partner_dimension, self.partner_dimension, 0), complex)
        self.diagnostics['distinct_group_actions'] = len(self._matrices)
        self._block_actions = {}
        for block in self.spatial_symmetry.orbital_blocks:
            representatives, labels = [], []
            for matrix in self._matrices:
                local = matrix[np.ix_(block, block)]
                label = next((i for i, old in enumerate(representatives)
                              if np.allclose(local, old, atol=1e-13, rtol=0)), None)
                if label is None:
                    label = len(representatives)
                    representatives.append(local)
                labels.append(label)
            self._block_actions[tuple(block)] = labels
        self._refresh()

    @property
    def state(self):
        return self.states[0] if self.partner_dimension == 1 and len(self.states) == 1 else self.states

    def _orbit(self, chain, *, site=None, cache=None):
        blocks = self.spatial_symmetry.orbital_blocks
        active = next((block for block in blocks if site in block), None)
        orbit = []
        local_orbits = {}
        for index, matrix in enumerate(self._matrices):
            if np.allclose(matrix, np.eye(len(matrix)), atol=1e-13, rtol=0):
                orbit.append(chain)
                continue
            previous = None if cache is None else cache.get(index)
            transform = matrix
            if previous is not None and active is not None:
                # Gates never cross invariant-block boundaries. Only the block
                # containing the varied site needs to be transformed again.
                transform = np.eye(len(matrix), dtype=complex)
                transform[np.ix_(active, active)] = matrix[np.ix_(active, active)]
            label = (self._block_actions[tuple(active)][index]
                     if previous is not None and active is not None else None)
            tensors = local_orbits.get(label) if label is not None else None
            if tensors is None:
                tensors = (list(chain) if np.max(np.abs(transform - np.eye(len(transform)))) < 1e-13
                           else list(apply_spatial_orbital_transform(
                               chain, transform, inverse=False, cutoff=0, max_bond=None,
                               orbital_blocks=blocks).tensors))
                if label is not None:
                    local_orbits[label] = tensors
            tensors = list(tensors)
            if previous is not None and active is not None:
                for position in range(len(tensors)):
                    if position not in active:
                        tensors[position] = previous[position]
            elif cache is not None:
                cache[index] = tensors
            orbit.append(tensors)
        return tuple(orbit)

    def _bilinear(self, bra, orbit, factors):
        return sum(weight * contract_chain_transition(bra, factors, ket)
                   for weight, ket in zip(self._weights[0, 0], orbit)
                   if abs(weight) > 1e-14)

    def _refresh(self):
        chain = self.seed.materialize()
        orbit = self._orbit(chain)
        norm = float(self._bilinear(chain, orbit, self.identity).real)
        if norm <= 1e-14:
            raise ValueError('The initial seed has no support in the target spatial irrep.')
        self.energy = float(self._bilinear(chain, orbit, self.mpo).real / norm + self.ecore)
        self.e_tot = np.repeat(self.energy, self.partner_dimension)
        self.states = tuple(sum_reduced_mps(orbit, self._weights[row, 0] / np.sqrt(norm))
                            for row in range(self.partner_dimension))
        for row, state in enumerate(self.states):
            state.spatial_irrep = self.wfnsym
            state.spatial_partner_row = row
            state.spatial_group = self.spatial_symmetry.group.name
        return norm

    def norm(self):
        return float(contract_chain_transition(self.states[0].tensors, self.identity, self.states[0].tensors).real)

    def expectation(self, mpo=None):
        factors = self.mpo if mpo is None else getattr(mpo, 'factors', mpo)
        value = contract_chain_transition(self.states[0].tensors, factors, self.states[0].tensors)
        return float(value.real + (self.ecore if mpo is None else 0))

    def optimize_site(self, site, *, metric_rtol=1e-10, accept_tol=1e-9):
        """Local Rayleigh--Ritz update with the spatial structural leg retained.

        The norm is assembled first; Hamiltonian contractions act only on its
        retained orthonormal directions (relative cutoff ``metric_rtol``).
        This reorders the same local generalized eigensolve, with no change
        to the regular-representation ansatz described by the class.
        """
        current = self.seed._pack_site(site)
        count = len(current)
        chains, indices = [], []
        baseline = tuple(self.seed.materialize())
        norm_plan = LocalTransitionPlan.build(baseline, self.identity, site, direction='lr')
        norm_right = (LocalTransitionPlan.build(baseline, self.identity, site, direction='rl')
                      if norm_plan.rank_coupled else None)
        try:
            for index in range(count):
                unit = np.zeros(count, dtype=complex)
                unit[index] = 1
                self.seed._set_site_vector(site, unit)
                chain = self.seed.materialize()
                if norm_right is None:
                    norm = norm_plan.contract(chain[site], chain[site]).real
                else:
                    action = one_site_reduced_adjoint(chain[site], chain[site],
                        norm_plan.mpo_factors[site], norm_plan.anchor.data, norm_right.anchor.data)
                    norm = sum(np.vdot(block, action[key]).real
                               for key, block in chain[site].data.items())
                if norm > 1e-20:
                    snapshot = list(baseline)
                    snapshot[site] = chain[site].copy()
                    chains.append(snapshot)
                    indices.append(index)
        finally:
            self.seed._set_site_vector(site, current)
        orbit_cache, boundary_cache = {}, {}
        block_keys = tuple(baseline[site].data)
        bra_basis = np.stack([
            np.concatenate([chain[site].data[key].ravel() for key in block_keys]).conj()
            for chain in chains
        ]) if chains else np.empty((0, count), complex)
        active = next(block for block in self.spatial_symmetry.orbital_blocks if site in block)

        def assemble(directions, factors, coefficients=None, with_overlaps=False):
            dimension = len(directions)
            matrix = np.zeros((dimension, dimension), complex)
            overlaps = np.zeros((len(self.orthogonal_to), dimension), complex)
            environment_cache = {}
            for j, ket in enumerate(directions):
                orbit = self._orbit(ket, site=site, cache=orbit_cache)
                for op, (weight, transformed) in enumerate(zip(self._weights[0, 0], orbit)):
                    if abs(weight) < 1e-14:
                        continue
                    plans = []
                    identity_action = transformed is ket
                    for direction, fixed in (('lr', identity_action or site == active[0]),
                                             ('rl', identity_action or site == active[-1])):
                        key = (op, direction)
                        plan = environment_cache.get(key)
                        if plan is None:
                            plan = LocalTransitionPlan.build(baseline, factors, site,
                                direction=direction, ket_sites=transformed, boundary_cache=boundary_cache)
                            if fixed and plan.rank_coupled:
                                environment_cache[key] = plan
                        plans.append(plan)
                    left, right = plans
                    if left.rank_coupled:
                        action = one_site_reduced_adjoint(baseline[site], transformed[site],
                            left.mpo_factors[site], left.anchor.data, right.anchor.data)
                        vector = np.concatenate([action[key].ravel() for key in block_keys])
                        entries = (bra_basis[j:] @ vector if coefficients is None
                                   else coefficients[:, j:].conj().T @ (bra_basis @ vector))
                        matrix[j:, j] += weight * entries
                    else:
                        for i in range(j, dimension):
                            matrix[i, j] += weight * left.contract(directions[i][site], transformed[site])
                matrix[j, j+1:] = matrix[j+1:, j].conj()
                if with_overlaps:
                    for i, previous in enumerate(self.orthogonal_to):
                        overlaps[i, j] = self._bilinear(previous.tensors, orbit, self.identity)
            return (matrix + matrix.conj().T)/2, overlaps

        metric, overlaps = assemble(chains, self.identity, with_overlaps=True)
        values, vectors = eigh(metric)
        keep = values > metric_rtol * max(float(values.max(initial=0)), 1e-30)
        if not np.any(keep):
            return {'site': site, 'accepted': False, 'metric_rank': 0}
        whitener = vectors[:, keep] / np.sqrt(values[keep])
        if len(self.orthogonal_to):
            whitener = whitener @ null_space(overlaps @ whitener, rcond=metric_rtol)
        if whitener.shape[1] == 0:
            return {'site': site, 'accepted': False, 'metric_rank': 0}
        # Remove null/gauge directions before Hamiltonian contractions. These
        # are local reduced tensor directions, not a determinant-space basis.
        independent = []
        try:
            for column in whitener.T:
                vector = np.zeros(count, complex)
                vector[indices] = column
                self.seed._set_site_vector(site, vector)
                chain = list(baseline)
                chain[site] = self.seed.materialize()[site].copy()
                independent.append(chain)
        finally:
            self.seed._set_site_vector(site, current)
        h, _ = assemble(independent, self.mpo, coefficients=whitener)
        energies, local = eigh(h)
        candidate = np.zeros(count, dtype=complex)
        candidate[indices] = whitener @ local[:, 0]
        before = self.energy
        # An excited-state seed can initially overlap lower roots; its first
        # constrained update establishes the variational orthogonality domain.
        accepted = (len(self.orthogonal_to) > 0 and not self.history) or energies[0] + self.ecore <= before + accept_tol
        if accepted:
            self.seed._set_site_vector(site, candidate)
            self.energy = float(energies[0] + self.ecore)
        return {'site': site, 'accepted': bool(accepted), 'energy_before': before,
                'energy_after': self.energy, 'metric_rank': whitener.shape[1],
                'parameters': count, 'spatial_irrep': self.wfnsym,
                'partner_dimension': self.partner_dimension}

    def run(self, *, nsweeps=2, tol=1e-9, nstates=1, verbose=0, **kwargs):
        """Run complete left/right one-site cycles; return every target partner."""
        allowed = {'metric_rtol', 'accept_tol'}
        unknown = set(kwargs) - allowed
        if unknown:
            raise TypeError(f'Unknown spatial-multiplet sweep options: {sorted(unknown)}')
        if int(nsweeps) < 1 or int(nstates) < 1:
            raise ValueError('nsweeps and nstates must be positive.')
        with threadpool_limits(limits=1):
            for cycle in range(int(nsweeps)):
                before = self.energy
                updates = []
                for direction in (range(self.nsites), range(self.nsites-1, -1, -1)):
                    updates.extend(self.optimize_site(site, **kwargs) for site in direction)
                self._refresh()
                change = abs(self.energy - before)
                self.history.append({'cycle': cycle+1, 'energy': self.energy, 'energy_change': change,
                                     'updates': updates, 'partner_dimension': self.partner_dimension})
                if verbose:
                    print(f'{self.wfnsym} cycle {cycle+1}: {self.energy:.12f}')
                if change < tol:
                    self.converged = True
                    break
        self.success = self.converged
        self.message = 'converged' if self.converged else 'maximum spatial-multiplet sweeps reached'
        if int(nstates) > 1:
            if self._seed_factory is None:
                raise ValueError('Multiple multiplets require construction with from_integrals().')
            roots = [self]
            previous = [self.states[0]]
            energies, states = list(self.e_tot), list(self.states)
            for root in range(1, int(nstates)):
                next_solver = self._seed_factory(root, previous)
                next_solver.run(nsweeps=nsweeps, tol=tol, verbose=verbose, **kwargs)
                roots.append(next_solver)
                previous.append(next_solver.states[0])
                energies.extend(next_solver.e_tot)
                states.extend(next_solver.states)
            self.multiplet_solvers = tuple(roots)
            self.states, self.e_tot = tuple(states), np.array(energies)
            self.spatial_multiplet_count = int(nstates)
            self.nstates = len(self.states)
            self.spatial_partner_rows = tuple(range(self.partner_dimension)) * int(nstates)
            self.converged = self.success = all(root.success for root in roots)
        gram = np.array([[contract_chain_transition(a.tensors, self.identity, b.tensors)
                          for b in self.states] for a in self.states])
        orthogonality_error = float(np.max(np.abs(gram - np.eye(len(self.states)))))
        self.diagnostics['state_orthogonality_error'] = orthogonality_error
        if orthogonality_error > max(1e-8, 10 * kwargs.get('metric_rtol', 1e-10)):
            self.converged = self.success = False
            self.message = 'Spatial multiplet states failed the orthogonality check.'
        self.state_irreps = (self.wfnsym,) * len(self.e_tot)
        if verbose:
            from pyqed.qchem.symmetry import print_irrep_table
            print_irrep_table("State", self.e_tot, self.state_irreps,
                              partners=self.spatial_partner_rows)
        return self

    def close(self):
        self._seed_factory = None
        self.seed.close()
        for solver in getattr(self, 'multiplet_solvers', ()):
            if solver is not self:
                solver.close()


class _MPSTensorOwner:
    """Local reduced MPS parameters; independent of LETTA graphs and ties."""

    def __init__(self, hamiltonian, *, nelec, spin, D, seed, n_threads):
        from numbers import Integral
        from pyqed import num_workers
        from .states import build_random_reduced_spatial_mps, spatial_target_sector
        from .mpo import prepare_hamiltonian_factors
        if not isinstance(D, Integral) or isinstance(D, bool) or D < 1:
            raise ValueError('D must be a positive integer')
        self.n_threads = num_workers() if n_threads is None else n_threads
        if not isinstance(self.n_threads, Integral) or isinstance(self.n_threads, bool) or self.n_threads < 1:
            raise ValueError('n_threads must be a positive integer')
        self.virtual_boundary = 'open'
        self.nsites, self.D = hamiltonian.n_sites, int(D)
        self.ecore = 0.0 if hamiltonian.info.get('includes_core_energy', False) else hamiltonian.ecore
        self.mpo = prepare_hamiltonian_factors(hamiltonian.factors, materialize=True)
        self.mps = MPS.from_tensors(build_random_reduced_spatial_mps(
            self.nsites, target_sector=spatial_target_sector(nelec, spin),
            bond_multiplicity=self.D, seed=seed, cap_multiplicity=True))
        self._runtime = hamiltonian.moving_environment
        if self._runtime is not None and hasattr(self._runtime, 'set_num_threads'):
            self._runtime.set_num_threads(self.n_threads)

    def materialize(self):
        return self.mps.tensors

    def _pack_site(self, site):
        return np.concatenate([block.ravel() for block in self.mps.tensors[site].data.values()])

    def _set_site_vector(self, site, vector):
        tensor = self.mps.tensors[site]
        offset, data = 0, {}
        for key, block in tensor.data.items():
            data[key] = np.asarray(vector[offset:offset+block.size]).reshape(block.shape).copy()
            offset += block.size
        self.mps.tensors[site] = IrrepTensor(data=data, qns=tensor.qns,
                                           dirs=tensor.dirs, metadata=tensor.metadata)

    def close(self):
        self._runtime = None


class SpatialSymmetryDMRG(SpatialMultipletSweep):
    """Spatial DMRG factory, defaulting to direct compiled NC two-site sweeps.

    Explicit ``contractions="group_action"`` selects the independent legacy
    one-site MPO reference described below.

    Uses the MPS module's reduced tensors, orbital gates, and cached MPO
    environments. No LETTA object, graph, or parameterization is involved.
    This fixed-D variational solver shares the exact regular-group objectives
    and limitations documented by SpatialMultipletSweep (Singh et al., PRA
    82, 050301, 2010, doi:10.1103/PhysRevA.82.050301). Spatial CG compression
    and two-site rank growth are not implemented here.
    """

    @classmethod
    def from_integrals(cls, h1e, eri, *, nelec, spin=0, D=2, ecore=0.,
                       orbital_representations, wfnsym, point_group='D3h',
                       seed=11, n_threads=None, cutoff=1e-10, contractions='nc'):
        if contractions == 'nc':
            from .nc_multiplet import NCMultipletDMRG
            return NCMultipletDMRG.from_integrals(h1e,eri,nelec=nelec,spin=spin,D=D,
                ecore=ecore,orbital_representations=orbital_representations,wfnsym=wfnsym,
                point_group=point_group,seed=seed,n_threads=n_threads,cutoff=cutoff)
        if contractions != 'group_action':
            raise ValueError('contractions must be nc or group_action.')
        from pyqed.qchem.dmrg.backends.reduced import build_spatial_reduced_hamiltonian_mpo
        SpatialGroupAction(orbital_representations, wfnsym,
                           point_group=point_group).validate_hamiltonian(h1e, eri)
        eri_builder = np.asarray(eri)[None, None, ...] if np.asarray(eri).ndim == 4 else eri
        hamiltonian = build_spatial_reduced_hamiltonian_mpo(
            h1e, eri_builder, fully_reduced=True, n_elec=nelec, spin=spin,
            ecore=ecore, cutoff=cutoff)
        def create(root=0, previous=()):
            owner = _MPSTensorOwner(hamiltonian, nelec=nelec, spin=spin,
                                   D=D, seed=seed+root, n_threads=n_threads)
            return cls(owner, orbital_representations, wfnsym, point_group=point_group,
                       h1e=h1e, eri=eri, orthogonal_to=previous)
        solver = create()
        solver._seed_factory = create
        return solver
