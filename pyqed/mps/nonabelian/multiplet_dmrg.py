"""DMRG on complete orbital multiplets with charge x SU(2) x point-group Legs.

A fixed-multiplicity, one-site variational MPS algorithm adapted from the
symmetry-factorized tensors of Singh, Pfeifer and Vidal, PRA 82, 050301
(2010), doi:10.1103/PhysRevA.82.050301. Local generalized Rayleigh--Ritz
problems use reduced Hamiltonian actions. CG maps fix all spin/spatial
partners; only reduced multiplicity tensors are optimized. Finite and continuous molecular groups default to direct compiled NC two-site sweeps.
Explicit ``contractions="reduced"`` selects the independent reduced MPO
reference.
``contractions="components"`` retains explicit irrep rows for comparisons.
Small reduced local matrices are assembled up to 256 parameters; larger
Hamiltonian actions are matrix-free.
It constructs neither a molecular determinant basis nor a projected-growth
Hamiltonian. D caps whole reduced multiplicities. The reduced subclass also
supports one-site CBE; see cbe.py for its reference and fidelity limitations.
"""
from functools import lru_cache
from .sweep_control import SweepControl
from numbers import Integral
import numpy as np
from scipy.linalg import eigh, null_space
from scipy.sparse.linalg import LinearOperator, eigsh
from opt_einsum import contract_expression
from threadpoolctl import threadpool_limits
from pyqed import num_workers
from pyqed.symmetry import Irrep, Leg
from pyqed.qchem.spatial_symmetry import SpatialGroupAction
from .mps import MPS
from .fusion import TensorFusionMap, move_multiplet_center
from .multiplet_sites import OrbitalMultipletSite, orbital_multiplets, molecular_symmetry


@lru_cache(maxsize=1024)
def _contraction(expression, shapes):
    return contract_expression(expression, *shapes, optimize='greedy')


def _contract(expression, *arrays):
    return _contraction(expression, tuple(a.shape for a in arrays))(*arrays)


def _overlap(bra, ket):
    if len(bra) != len(ket):
        raise ValueError('MPS lengths differ.')
    env = np.ones((1,1), complex)
    for a,b in zip(bra,ket):
        env = _contract('ab,apc,bpd->cd',env,a.conj(),b)
    return np.trace(env)


class MultipletMPS(MPS):
    """Reduced MPS with a chosen row of its final spin/spatial multiplet.

    ``tensors`` remain reduced IrrepTensors; ``component_tensors`` expands
    only local structural CG maps for contractions or interoperability.
    """
    def component_tensors(self, *, boundary_row=None):
        tensors = [tensor.metadata['fusion_map'].expand(
            tensor.metadata['fusion_map'].pack(tensor)) for tensor in self.tensors]
        row = self.tensors[-1].metadata.get('boundary_row',0) if boundary_row is None else boundary_row
        if not isinstance(row,Integral) or isinstance(row,bool) or not 0 <= row < tensors[-1].shape[-1]:
            raise IndexError('Invalid multiplet boundary row.')
        tensors[-1] = tensors[-1][:,:,row:row+1]
        return tensors

    def norm_squared(self):
        tensors = self.component_tensors()
        return float(_overlap(tensors,tensors).real)

    def overlap(self, other):
        return _overlap(self.component_tensors(),other.component_tensors())

    def expectation(self, mpo):
        if hasattr(mpo,'expectation'):
            return mpo.expectation(self)
        cores = getattr(mpo,'factors',mpo)
        if len(cores) != len(self.tensors):
            raise ValueError('MPO and MPS lengths differ.')
        env = np.ones((1,1,1),complex)
        for tensor,core in zip(self.component_tensors(),cores):
            env = _contract('axb,apc,xypq,bqd->cyd',env,tensor.conj(),np.asarray(core),tensor)
        return env.item()/self.norm_squared()

    def canonicalize(self, center, *, cutoff=0., max_bond=None, **kwargs):
        if cutoff or max_bond is not None or kwargs:
            raise NotImplementedError('Multiplet gauge moves are exact; rank truncation is not implemented here.')
        if not 0 <= center < len(self.tensors):
            raise IndexError('Invalid orthogonality center.')
        tensors = [t.copy() for t in self.tensors]
        for i in range(center):
            move_multiplet_center(tensors,i,'lr')
        for i in range(len(tensors)-1,center,-1):
            move_multiplet_center(tensors,i,'rl')
        self.factors[:] = tensors
        self.center,self.gauge = center,'mixed'
        return self


def _bond_legs(sites, target, D):
    symmetry = sites[0].symmetry
    spatial = symmetry.factors[-1]
    identity = spatial.identity if hasattr(spatial,'identity') else spatial.group.irrep_names[0]
    vacuum = Irrep((0,0,identity))
    counts = [{vacuum:1}]
    for site in sites:
        following = {}
        for left, count in counts[-1].items():
            for physical, multiplicity in site.leg.dims.items():
                for charge in symmetry.fuse(left.charge,physical.charge):
                    if charge[0] <= target.charge[0]:
                        sector = Irrep(charge)
                        following[sector] = following.get(sector,0)+count*multiplicity
        counts.append(following)
    if target not in counts[-1]:
        raise ValueError(f'Target sector {target} is absent from the orbital space.')
    completions = [None]*(len(sites)+1)
    completions[-1] = {target:1}
    for i in range(len(sites)-1,-1,-1):
        completions[i] = {}
        for left in counts[i]:
            count = sum(multiplicity*completions[i+1].get(Irrep(charge),0)
                        for physical,multiplicity in sites[i].leg.dims.items()
                        for charge in symmetry.fuse(left.charge,physical.charge))
            if count:
                completions[i][left] = count
    legs = [Leg({q:min(D,counts[i][q],n) for q,n in completion.items()},symmetry=symmetry)
            for i,completion in enumerate(completions)]
    return legs, counts[-1][target]


def _group_mpo(h1e, eri, sites, blocks, cutoff):
    if not np.any(h1e) and not np.any(eri):
        return tuple(np.eye(site.leg.full_dim)[None,None,:,:]*(0 if i==0 else 1)
                     for i,site in enumerate(sites))
    from pyqed.qchem.dmrg.dmrg import _build_spatial_hamiltonian_tensor_mpo
    mpo, _, _ = _build_spatial_hamiltonian_tensor_mpo(
        h1e[None,...],eri[None,None,...],cutoff=cutoff)
    cores = [np.asarray(core) for core in mpo.factors]
    grouped = []
    for site,block in zip(sites,blocks):
        core = cores[block[0]]
        for orbital in block[1:]:
            other = cores[orbital]
            core = np.einsum('axpq,xbuv->abpuqv',core,other,optimize=True).reshape(
                core.shape[0],other.shape[1],core.shape[2]*other.shape[2],core.shape[3]*other.shape[3])
        grouped.append(np.real_if_close(_contract('pi,abpq,qj->abij',site.basis.conj(),core,site.basis)))
    return tuple(grouped)


class MultipletDMRG:
    """Fixed-D reduced MPS sweeps on complete finite-group orbital multiplets.

    The reduced and component contraction implementations are described in
    the module docstring. E partners share the same reduced
    tensors and are never independently truncated. ``D`` caps multiplicity
    per (N,2S,Gamma) sector, not the total component bond dimension.

    This is an adaptation of Singh, Pfeifer and Vidal, Phys. Rev. A 82,
    050301 (2010), https://doi.org/10.1103/PhysRevA.82.050301, with one-site
    sweeps and exact sector QR gauges as reviewed by Schollwoeck, Ann. Phys.
    326, 96--192 (2011), https://doi.org/10.1016/j.aop.2010.09.012.
    The reduced subclass supports controlled bond expansion (see cbe.py).
    The factory defaults to fully reduced
    MPO/environment contractions; ``contractions="components"`` selects the
    previous component implementation for comparisons.
    """
    @classmethod
    def from_integrals(cls, h1e, eri, *, nelec, spin=0, D=32, ecore=0.,
                       orbital_representations, wfnsym, point_group='D3h',
                       seed=11, n_threads=None, cutoff=1e-12, contractions=None):
        """Build reduced charge × spin × orbital multiplet DMRG.

        ``orbital_representations`` contains unitary operation matrices for
        finite groups. For Coov/Dooh/SO3 it contains Hermitian L generators
        and unitary reflection/inversion matrices, as returned by
        ``mo_symmetry_representations(..., generators=True)``. Every site
        contains a complete orbital multiplet. Fusion follows the
        Wigner–Eckart construction of Singh and Vidal, PRB 86, 195114 (2012),
        https://doi.org/10.1103/PhysRevB.86.195114, adapted to a direct product
        of charge, spin and spatial symmetry. No spin-orbit coupling is included.
        All supported finite and continuous groups default to direct NC; request ``contractions="reduced"``
        explicitly for the independent MPO reference.
        Local shell Fock construction has a 512 MiB estimated workspace limit;
        it does not expand the molecular determinant space.
        """
        if contractions is None:
            contractions = 'nc'
        if contractions not in ('components','reduced','nc'):
            raise ValueError("contractions must be components, reduced or nc.")
        if contractions == 'nc':
            from .nc_multiplet import NCMultipletDMRG
            cls = NCMultipletDMRG
        if contractions == 'reduced':
            from .multiplet_sweep import ReducedMultipletDMRG
            cls = ReducedMultipletDMRG
        elif contractions == 'components':
            cls = MultipletDMRG
        if not isinstance(D,Integral) or isinstance(D,bool) or D<1:
            raise ValueError('D must be a positive integer.')
        obj = cls()
        obj.contractions = contractions
        obj.D, obj.ecore = int(D), float(ecore)
        if any(not isinstance(value,Integral) or isinstance(value,bool) or value<0
               for value in (nelec,spin)):
            raise ValueError('nelec and spin (2S) must be nonnegative integers.')
        obj.n_threads = num_workers() if n_threads is None else n_threads
        if not isinstance(obj.n_threads,Integral) or isinstance(obj.n_threads,bool) or obj.n_threads<1:
            raise ValueError('n_threads must be positive.')
        obj.workers = 1
        if point_group in ('Coov','Dooh','SO3'):
            from pyqed.symmetry.continuous import OrbitalGeneratorAction
            obj.spatial_symmetry = OrbitalGeneratorAction(orbital_representations,wfnsym,point_group=point_group)
        else:
            obj.spatial_symmetry = SpatialGroupAction(orbital_representations,wfnsym,point_group=point_group)
        obj.spatial_symmetry.validate_hamiltonian(h1e,eri)
        obj.wfnsym = obj.spatial_symmetry.wfnsym
        obj.partner_dimension = obj.spatial_symmetry.dimension
        transform, labels, blocks = orbital_multiplets(point_group,orbital_representations,
            subspaces=getattr(obj.spatial_symmetry, "subspaces", None))
        obj.orbital_transform, obj.orbital_blocks = transform,blocks
        obj.sites = tuple(OrbitalMultipletSite(point_group,label) for label in labels)
        obj.nsites = len(obj.sites)
        obj.target_sector = Irrep((int(nelec),int(spin),obj.wfnsym))
        obj.bond_legs,obj.available_multiplets = _bond_legs(obj.sites,obj.target_sector,obj.D)
        obj.fusion_maps = tuple(TensorFusionMap(obj.bond_legs[i],site.leg,obj.bond_legs[i+1])
                               for i,site in enumerate(obj.sites))
        obj._initial_fusion_maps = obj.fusion_maps
        obj._previous = []
        h = transform.conj().T @ np.asarray(h1e) @ transform
        v = np.einsum('pa,qb,rc,sd,pqrs->abcd',transform.conj(),transform,
                      transform.conj(),transform,np.asarray(eri),optimize=True)
        obj._build_hamiltonian(h,v,obj.sites,blocks,cutoff)
        obj._seed = int(seed)
        obj._initialize(seed)
        obj.states, obj.history = (), []
        obj.success = obj.converged = False
        obj.message = 'not run'
        obj.diagnostics = {'point_group':point_group,'spin_reduced':True,'spatial_reduced':True,
                           'orbital_multiplets':labels,'physical_dimensions':tuple(s.leg.full_dim for s in obj.sites),
                           'reduced_physical_dimensions':tuple(s.leg.dim for s in obj.sites),
                           'contraction_representation':contractions,
                           'mpo_reduced':contractions=='reduced','environments_reduced':contractions in ('reduced','nc')}
        obj.diagnostics.update(obj._hamiltonian_diagnostics())
        return obj

    _make_mpo = staticmethod(_group_mpo)

    def _build_hamiltonian(self,h,v,sites,blocks,cutoff):
        self.mpo = self._make_mpo(h,v,sites,blocks,cutoff)

    def _hamiltonian_diagnostics(self):
        return {'hamiltonian_representation':'mpo','hamiltonian_mpo':True}

    def _state_energy(self,state):
        return float(state.expectation(self.mpo).real+self.ecore)

    def _edge(self, right=False):
        if not right:
            return np.ones((1,1,1)),np.ones((1,1))
        dim = self.components[-1].shape[-1]
        return np.eye(dim)[:,None,:]/dim,np.eye(dim)/dim

    def _advance(self, i, h, norm, direction):
        a,w = self.components[i],self.mpo[i]
        if direction == 'lr':
            return (_contract('axb,apc,xypq,bqd->cyd',h,a.conj(),w,a),
                    _contract('ab,apc,bpd->cd',norm,a.conj(),a))
        return (_contract('apc,xypq,bqd,cyd->axb',a.conj(),w,a,h),
                _contract('apc,bpd,cd->ab',a.conj(),a,norm))

    def _snapshot(self):
        return tuple(a.copy() for a in self.components)

    def _previous_overlap(self, previous):
        return _overlap(previous,self.components)/self.components[-1].shape[-1]

    def _initialize(self, seed):
        rng = np.random.default_rng(seed)
        self.vectors = [rng.normal(size=f.matrix.shape[1]) for f in self.fusion_maps]
        self.components = [f.expand(v) for f,v in zip(self.fusion_maps,self.vectors)]
        for i in range(self.nsites-1,0,-1):
            self._move_center(i,'rl')
        self._refresh()

    def _move_center(self, site, direction):
        tensors = [f.tensor(v) for f,v in zip(self.fusion_maps,self.vectors)]
        move_multiplet_center(tensors,site,direction)
        maps = list(self.fusion_maps)
        for i in (site,site+1 if direction=='lr' else site-1):
            maps[i] = tensors[i].metadata['fusion_map']
            self.vectors[i] = maps[i].pack(tensors[i])
            self.components[i] = maps[i].expand(self.vectors[i])
        self.fusion_maps = tuple(maps)

    def _refresh(self):
        dim = self.components[-1].shape[-1]
        norm = _overlap(self.components,self.components).real/dim
        if norm <= 0:
            raise ArithmeticError('The MPS has zero norm.')
        self.vectors[0] /= np.sqrt(norm)
        self.components[0] /= np.sqrt(norm)
        env = np.ones((1,1,1),complex)
        for a,w in zip(self.components,self.mpo):
            env = _contract('axb,apc,xypq,bqd->cyd',env,a.conj(),w,a)
        self.energy = float(np.trace(env[:,0,:]).real/dim+self.ecore)

    def _boundaries(self, direction):
        size = self.nsites
        h, norm = [None]*(size+1),[None]*(size+1)
        if direction == 'lr':
            dim = self.components[-1].shape[-1]
            h[-1], norm[-1] = np.eye(dim)[:,None,:]/dim, np.eye(dim)/dim
            for i in reversed(range(size)):
                a,w = self.components[i],self.mpo[i]
                h[i] = _contract('apc,xypq,bqd,cyd->axb',a.conj(),w,a,h[i+1])
                norm[i] = _contract('apc,bpd,cd->ab',a.conj(),a,norm[i+1])
        else:
            h[0], norm[0] = np.ones((1,1,1)),np.ones((1,1))
            for i in range(size):
                a,w = self.components[i],self.mpo[i]
                h[i+1] = _contract('axb,apc,xypq,bqd->cyd',h[i],a.conj(),w,a)
                norm[i+1] = _contract('ab,apc,bpd->cd',norm[i],a.conj(),a)
        return h,norm

    def _optimize_site(self, site, left, right, norm_left, norm_right, *, metric_rtol, local_tol):
        fusion = self.fusion_maps[site]
        count = fusion.matrix.shape[1]
        # Bound local workspace explicitly; never replace this with a molecular
        # determinant-space Hamiltonian or an exponentially large fallback.
        if 16*(2*int(np.prod(fusion.shape))*count + 3*count**2) > 256*1024**2:
            raise MemoryError('Local fusion workspace exceeds 256 MiB; reduce D.')
        basis = fusion.matrix.toarray().reshape(*fusion.shape,count)
        norm_basis = _contract('ab,bqck,dc->aqdk',norm_left,basis,norm_right)
        metric = fusion.matrix.conj().T @ norm_basis.reshape(-1,count)
        values,vectors = eigh((metric+metric.conj().T)/2)
        keep = values > metric_rtol*max(values.max(initial=0),1e-30)
        if not np.any(keep):
            raise ArithmeticError('No non-null local fusion directions.')
        whitening = vectors[:,keep]/np.sqrt(values[keep])
        if self._previous:
            overlaps = []
            for previous in self._previous:
                l = np.ones((1,1),complex)
                dim = previous[-1].shape[-1]
                r = np.eye(dim,dtype=complex)/dim
                for i in range(site):
                    l = _contract('ab,apc,bpd->cd',l,previous[i].conj(),self.components[i])
                for i in range(self.nsites-1,site,-1):
                    r = _contract('apc,bpd,cd->ab',previous[i].conj(),self.components[i],r)
                row = _contract('ab,apc,cd->bpd',l,previous[site].conj(),r)
                overlaps.append(row.ravel() @ fusion.matrix)
            constraints = np.asarray(overlaps) @ whitening
            scale = max(np.linalg.norm(constraints,2),1e-30)
            whitening = whitening @ null_space(constraints,rcond=metric_rtol/scale)
            if not whitening.shape[1]:
                return {'site':site,'parameters':count,'metric_rank':0,'accepted':False}
        def apply(vector):
            tensor = fusion.expand(whitening @ vector)
            result = _contract('axb,xypq,bqd,cyd->apc',left,self.mpo[site],tensor,right)
            return whitening.conj().T @ (fusion.matrix.conj().T @ result.ravel())
        rank = whitening.shape[1]
        if rank <= 2:
            local = np.column_stack([apply(v) for v in np.eye(rank)])
            energies,coefficients = eigh((local+local.conj().T)/2)
        else:
            operator = LinearOperator((rank,rank),matvec=apply,dtype=np.complex128)
            guess = whitening.conj().T @ metric @ self.vectors[site]
            energies,coefficients = eigsh(operator,k=1,which='SA',tol=local_tol,
                v0=guess if np.linalg.norm(guess)>1e-14 else np.ones(rank),maxiter=max(300,10*rank))
        self.vectors[site] = whitening @ coefficients[:,0]
        self.components[site] = fusion.expand(self.vectors[site])
        return {'site':site,'parameters':count,'metric_rank':rank,'energy':float(energies[0]+self.ecore)}

    def _finish_cycle(self, h, norm):
        self._refresh()

    def _configure_sweeps(self,nsweeps,algorithm,tol,min_sweeps,max_bond_mode):
        self._sweep_control = SweepControl(nsweeps, algorithm=algorithm, tolerance=tol,
            min_steps=min_sweeps, max_bond_mode=max_bond_mode, default_bond_mode="per_sector")
        if self._sweep_control.max_bond_mode != "per_sector":
            raise ValueError("Spatial multiplet D counts multiplicities per sector; use max_bond_mode=per_sector.")
        self.diagnostics.update(self._sweep_control.diagnostics())

    def _complete_cycle(self,cycle,before,records):
        record = dict(cycle=cycle+1, energy=self.energy, sites=records)
        self.history.append(record)
        self.success = self.converged = self._sweep_control.finish(record,
            step=cycle+1, energy_delta=abs(self.energy-before),
            local_converged=all(r.get('converged',True) for r in records))
        return self.success

    def _run_multiplet(self, *, nsweeps=12, tol=1e-9, metric_rtol=1e-10,
            local_tol=1e-11, verbose=0, algorithm="one_site", cbe_expansion=8,
            cbe_selection="shrewd", cbe_preselection=None, min_sweeps=1, max_bond_mode="per_sector"):
        self._configure_sweeps(nsweeps,algorithm,tol,min_sweeps,max_bond_mode)
        if algorithm not in {"one_site", "one_site_cbe"}:
            raise ValueError("Multiplet algorithm must be one_site or one_site_cbe")
        self._cbe_expansion = 0
        if algorithm == "one_site_cbe":
            if not hasattr(self, "_expand_bond"):
                raise NotImplementedError("CBE requires reduced spatial contractions")
            if int(cbe_expansion) != cbe_expansion or cbe_expansion < 1:
                raise ValueError("cbe_expansion must be positive")
            self._cbe_expansion = int(cbe_expansion)
            self._cbe_selection = cbe_selection
            self._cbe_preselection = cbe_preselection
        if any(not np.isfinite(value) or value <= 0 for value in (tol,local_tol,metric_rtol)) or metric_rtol >= 1:
            raise ValueError('Tolerances must be finite and positive, with metric_rtol < 1.')
        fixed = None
        with threadpool_limits(limits=1):
            for cycle in self._sweep_control.steps:
                before = self.energy
                records = []
                for direction in ('lr','rl'):
                    fixed_h,fixed_n = self._boundaries(direction) if fixed is None else fixed
                    next_h,next_n = [None]*(self.nsites+1),[None]*(self.nsites+1)
                    if direction == 'lr':
                        moving_h,moving_n = self._edge()
                        next_h[0],next_n[0] = moving_h,moving_n
                        order = range(self.nsites)
                    else:
                        moving_h,moving_n = self._edge(True)
                        next_h[-1],next_n[-1] = moving_h,moving_n
                        order = reversed(range(self.nsites))
                    for i in order:
                        if direction == 'lr':
                            if self._cbe_expansion and i+1 < self.nsites:
                                self._expand_bond(i,direction,moving_h,fixed_h[i+2],moving_n,fixed_n[i+2])
                                fixed_h[i+1],fixed_n[i+1]=self._advance(i+1,fixed_h[i+2],fixed_n[i+2],'rl')
                            records.append(self._optimize_site(i,moving_h,fixed_h[i+1],moving_n,fixed_n[i+1],
                                metric_rtol=metric_rtol,local_tol=local_tol))
                            if i+1 < self.nsites:
                                self._move_center(i,'lr')
                            moving_h,moving_n = self._advance(i,moving_h,moving_n,direction)
                            next_h[i+1],next_n[i+1] = moving_h,moving_n
                        else:
                            if self._cbe_expansion and i:
                                self._expand_bond(i-1,direction,fixed_h[i-1],moving_h,fixed_n[i-1],moving_n)
                                fixed_h[i],fixed_n[i]=self._advance(i-1,fixed_h[i-1],fixed_n[i-1],'lr')
                            records.append(self._optimize_site(i,fixed_h[i],moving_h,fixed_n[i],moving_n,
                                metric_rtol=metric_rtol,local_tol=local_tol))
                            if i:
                                self._move_center(i,'rl')
                            moving_h,moving_n = self._advance(i,moving_h,moving_n,direction)
                            next_h[i],next_n[i] = moving_h,moving_n
                    # Completed prefixes/suffixes remain valid for the reverse
                    # pass. Refresh scales site zero only, outside lr suffixes.
                    fixed = next_h,next_n
                self._finish_cycle(moving_h,moving_n)
                done = self._complete_cycle(cycle,before,records)
                if verbose:
                    print(f'Multiplet DMRG {cycle+1}: {self.energy:.12f}',flush=True)
                if done:
                    break
        self.message = 'converged' if self.success else 'sweep limit reached'
        return self._store_multiplet_states()

    def _store_multiplet_states(self):
        tensors = [f.tensor(v) for f,v in zip(self.fusion_maps,self.vectors)]
        states = []
        for row in range(self.partner_dimension):
            copy = [t.copy() for t in tensors]
            copy[-1].metadata['boundary_row'] = row
            states.append(MultipletMPS.from_tensors(copy,target_sector=self.target_sector,
                          sites=[site.leg for site in self.sites]))
        self.states = tuple(states)
        self.state = self.states[0] if len(states)==1 else self.states
        self.e_tot = np.repeat(self.energy,self.partner_dimension)
        self.state_irreps = (self.wfnsym,)*self.partner_dimension
        self.spatial_partner_rows = tuple(range(self.partner_dimension))
        self.nstates,self.spatial_multiplet_count = self.partner_dimension,1
        return self

    def run(self, *, nstates=1, **kwargs):
        if kwargs.get("algorithm") == "one_site_cbe" and nstates != 1:
            raise ValueError("CBE currently targets one complete spatial multiplet")
        """Optimize ``nstates`` multiplets, returning every E partner of each."""
        if not isinstance(nstates,Integral) or isinstance(nstates,bool) or nstates<1:
            raise ValueError('nstates must count a positive number of multiplets.')
        if nstates > self.available_multiplets:
            raise ValueError(f'Only {self.available_multiplets} target multiplets exist.')
        states, energies, history, converged = [],[],[],[]
        self._previous = []
        self._rdm_cache = {}
        for root in range(nstates):
            if root:
                self.fusion_maps = self._initial_fusion_maps
                self._initialize(self._seed+root)
            self.history = []
            self.success = self.converged = False
            self._run_multiplet(**kwargs)
            if any(abs(self._previous_overlap(previous))>1e-7 for previous in self._previous):
                self.success = self.converged = False
            states.extend(self.states); energies.extend(self.e_tot)
            history.extend(dict(record,multiplet=root) for record in self.history)
            converged.append(self.success)
            self._previous.append(self._snapshot())
        self.states,self.e_tot,self.history = tuple(states),np.asarray(energies),history
        self.energy = float(energies[0])
        self.state = self.states[0] if len(states)==1 else self.states
        self.success = self.converged = all(converged)
        self.message = 'converged' if self.success else 'sweep limit reached'
        self.nstates,self.spatial_multiplet_count = len(states),nstates
        self.state_irreps = (self.wfnsym,)*len(states)
        self.spatial_partner_rows = tuple(range(self.partner_dimension))*nstates
        return self

    def close(self):
        self._previous = []
        self._rdm_cache = {}

    def make_rdm12(self, state_id=0):
        """Spin-traced active RDMs in the input orbital basis, using MPS NPDMs.

        Local multiplet components are exactly split into d=4 sites for the
        existing environment-cached fermion-string contractions. No global
        wavefunction or determinant operator is materialized.
        """
        from pyqed.qchem.dmrg.dmrg import _SpatialNPDMContractions
        if state_id in self._rdm_cache:
            return self._rdm_cache[state_id]
        state = self.states[state_id]
        factors = []
        for site,tensor in zip(self.sites,state.component_tensors()):
            tensor = np.einsum('pq,aqb->apb',site.basis,tensor)
            if site.norb==1:
                factors.append(tensor)
            else:
                left,_,right = tensor.shape
                u,s,v = np.linalg.svd(tensor.reshape(left*4,4*right),full_matrices=False)
                factors.extend((u.reshape(left,4,-1),(s[:,None]*v).reshape(len(s),4,right)))
        npdm = _SpatialNPDMContractions(MPS.from_tensors(factors))
        norb = len(self.orbital_transform)
        dm1,dm2 = np.zeros((norb,)*2,complex),np.zeros((norb,)*4,complex)
        for p,q in np.ndindex(dm1.shape):
            dm1[p,q] = sum(npdm.expect_string([('cre',sigma,q),('ann',sigma,p)]) for sigma in range(2))
        for p,q,r,s in np.ndindex(dm2.shape):
            dm2[p,q,r,s] = sum(npdm.expect_string([('cre',sigma,p),('cre',tau,r),
                ('ann',tau,s),('ann',sigma,q)]) for sigma in range(2) for tau in range(2))
        u = self.orbital_transform
        dm1 = u @ dm1 @ u.conj().T
        dm2 = np.einsum('pa,qb,rc,sd,abcd->pqrs',u.conj(),u,u.conj(),u,dm2,optimize=True)
        self._rdm_cache[state_id] = np.real_if_close(dm1),np.real_if_close(dm2)
        return self._rdm_cache[state_id]

    def symmetry_diagnostics(self):
        """Check every returned partner against every finite-group operation."""
        from pyqed.qchem.symmetry import irrep_representation_matrices
        from pyqed.symmetry.continuous import OrbitalRotationSymmetry
        group = self.spatial_symmetry.group
        continuous = isinstance(group,OrbitalRotationSymmetry)
        matrices = group.generators(self.wfnsym) if continuous else irrep_representation_matrices(group,self.wfnsym)
        chains = [state.component_tensors() for state in self.states]
        gram = np.array([[_overlap(a,b) for b in chains] for a in chains])
        error = 0.
        for op,matrix in matrices.items():
            local = [site.basis.conj().T @ site.representations[op] @ site.basis for site in self.sites]
            if continuous and op.startswith('L'):
                actual = np.zeros((len(chains),len(chains)),complex)
                for site,u in enumerate(local):
                    transformed = [chain[:site]+[np.einsum('pq,aqb->apb',u,chain[site])]+chain[site+1:] for chain in chains]
                    actual += np.array([[_overlap(a,b) for b in transformed] for a in chains])
            else:
                transformed = [[np.einsum('pq,aqb->apb',u,a) for u,a in zip(local,chain)] for chain in chains]
                actual = np.array([[_overlap(a,b) for b in transformed] for a in chains])
            expected = np.kron(np.eye(self.spatial_multiplet_count),matrix)
            error = max(error,float(np.max(abs(actual-expected))))
        return {'orthogonality_error':float(np.max(abs(gram-np.eye(len(chains))))),
                'covariance_error':error,
                'partner_energies_hartree':[self._state_energy(state) for state in self.states]}
