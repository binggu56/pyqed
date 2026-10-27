"""Complete-multiplet DMRG using direct compiled normal/complementary routes."""
import numpy as np
from numbers import Integral
from threadpoolctl import threadpool_limits
from pyqed.symmetry import Leg
from .multiplet_sweep import ReducedMultipletDMRG
from .multiplet_operator import boundary
from . import _su2_kernel as kernels
from .nc_spatial_environment import SpatialNCEnvironment


class NCMultipletDMRG(ReducedMultipletDMRG):
    """Direct compiled charge/spin/orbital-rank normal/complementary DMRG.

    The integral-owned normal/complementary partition adapts Chan and
    Head-Gordon, JCP 116, 4462 (2002), doi:10.1063/1.1449459, to complete
    orbital shells. Exact CG reduction follows Singh and Vidal, PRB 86,
    195114 (2012), doi:10.1103/PhysRevB.86.195114. Channels retain at most
    two normal fermion indices and one complementary index. All spin and
    orbital partners share multiplicity coefficients through the
    one-site sweeps. Sweep control, norm/overlap contractions, sector QR and
    local metric reduction execute in the compiled extension. Local Davidson
    iterations and Hamiltonian actions execute in the C++ moving environment.
    One-site gauges and optimization
    use the standard MPS formulation reviewed by Schollwoeck, Ann. Phys. 326,
    96 (2011), doi:10.1016/j.aop.2010.09.012, and the shared packed Davidson
    adaptation (J. Comput. Phys. 17, 87, 1975, doi:10.1016/0021-9991(75)90065-0).
    There is no Hamiltonian MPO, numerical operator-train fit, or molecular
    determinant-space projection.

    Supports real screened molecular integrals, complex state tensors, and
    Coov/Dooh/SO3 and finite C1, Cs, Ci, C2, C2v, C2h, D2, D2h, D3h, D6h
    symmetry. Finite-group CG maps use the exact matrix-unit construction
    (Cotton, Chemical Applications of Group Theory, 3rd ed., Wiley, 1990).
    Supported finite groups have real irreps and multiplicity-free binary
    products; complex irreps and outer product multiplicities are unsupported. ``run(algorithm="one_site_cbe")`` uses compiled
    shrewd controlled bond expansion, adapting Gleis, Li and von Delft,
    PRL 130, 246402 (2023), doi:10.1103/PhysRevLett.130.246402. Selection
    preserves whole multiplets with a global expansion budget; truncation
    caps each sector at D. Reduced parent layouts and bounded operator-half
    workspaces remain, so the paper's single-site scaling is not guaranteed.
    ``run()`` defaults to ``algorithm="two_site"`` and performs finite-system two-site sweeps,
    following Schollwoeck (2011), with direct NC channel halves and the same
    compiled Davidson solver. Pair SVD truncates whole product-group sectors
    to D multiplicities each. Pair layouts and channel halves have bounded
    explicit storage; no full two-site Hamiltonian is assembled. This is an
    adaptation with a per-sector cap, not a global discarded-weight guarantee.
    CBE and two-site currently target one spatial multiplet. Spin-orbit coupling is unsupported. Local Hamiltonians remain matrix-free except for
    rank-one/two Ritz problems and Davidson subspaces. Bounded local metric
    whitening and constraint SVD retain non-null reduced variational directions.
    Python supplies symmetry topology and state export. Fixed-D one-site
    convergence is variational within the retained multiplicity spaces, with
    no global-convergence guarantee.
    """
    def run(self, *, nstates=1, **kwargs):
        kwargs.setdefault('algorithm', 'two_site')
        if kwargs['algorithm']=='two_site' and nstates!=1:
            raise ValueError('NC two-site currently targets one complete spatial multiplet.')
        before=getattr(self,'completed_sweeps',0)
        self._checkpoint_ready=False
        result=super().run(nstates=nstates, **kwargs)
        self.completed_sweeps=before+len(self.history)
        self._checkpoint_ready=True
        return result

    def save_checkpoint(self,path):
        """Atomically save reduced coefficients after a complete sweep run."""
        from .nc_checkpoint import save_checkpoint
        return save_checkpoint(self,path)

    def load_checkpoint(self,path):
        """Restore into the identical integral/basis/target problem.

        Thread count may change. The saved D and completed sweep count are
        restored; the next run performs additional sweeps. Compiled boundaries
        are rebuilt, never deserialized. Only one complete multiplet is saved.
        """
        from .nc_checkpoint import load_checkpoint
        return load_checkpoint(self,path)

    def _build_hamiltonian(self,h,v,sites,blocks,cutoff):
        from .nc_checkpoint import problem_signature
        self._problem_signature=problem_signature(self,h,v,cutoff)
        self.nc = SpatialNCEnvironment(h,v,sites,blocks,
            [leg for fusion in self.fusion_maps for leg in fusion.legs],
            nelec=self.target_sector.charge[0],spin=self.target_sector.charge[1],
            cutoff=cutoff,n_threads=self.n_threads)

    def _hamiltonian_diagnostics(self):
        return {'hamiltonian_representation':'normal_complementary',
                'hamiltonian_mpo':False,'operator_train_fit':False,
                'sweep_driver':'compiled','local_solver':'davidson',
                'normal_complementary_channel_counts':tuple(leg.full_dim for leg in self.nc.legs),
                'normal_complementary_rank_transition_counts':tuple(len(x) for x in self.nc.records)}

    def _initialize(self,seed):
        rng = np.random.default_rng(seed)
        self.vectors = [rng.normal(size=f.parameter_count) for f in self.fusion_maps]
        sym = self.sites[0].symmetry
        self._identity_leg = Leg({self.nc.zero:1},symmetry=sym)
        self.identity = tuple({(self.nc.zero,self.nc.zero,p,p,self.nc.zero):
            np.eye(site.leg.dims[p])[None,None,:,:] for p in site.leg.sectors} for site in self.sites)
        for i in range(self.nsites-1,0,-1):
            self._move_center(i,'rl')
        self._refresh()

    def _edge(self,right=False):
        leg = self.fusion_maps[-1].legs[-1] if right else self.fusion_maps[0].legs[0]
        return self.nc.edge(leg,right=right),boundary(leg,self._identity_leg,right)

    def _advance(self,i,h,norm,direction):
        data = self.fusion_maps[i].unpack(self.vectors[i])
        return (self.nc.advance(i,h,data,data,direction),
                kernels.multiplet_norm_advance(self.sites[0].symmetry,norm,self.identity[i],data,data,direction))

    def _move_center(self,site,direction):
        kernels.move_spatial_nc_center(self,site,direction)

    def _overlap_rows(self,site):
        return kernels.spatial_nc_overlap_rows(self,site)

    def _previous_overlap(self,previous):
        _,env=self._edge()
        for i in range(self.nsites):
            env=kernels.multiplet_norm_advance(self.sites[0].symmetry,env,self.identity[i],previous[i].data,self._tensor(i).data,'lr')
        return next(iter(env.values())).item()

    def _optimize_site(self,site,left,right,norm_left,norm_right,*,metric_rtol,local_tol):
        return kernels.optimize_spatial_nc_site(self,site,left,right,norm_left,norm_right,metric_rtol,local_tol)

    def _run_multiplet(self,*,nsweeps=12,tol=1e-9,metric_rtol=1e-10,
                      local_tol=1e-11,verbose=0,algorithm='two_site',
                      cbe_expansion=8,cbe_selection='shrewd',cbe_preselection=None,
                      min_sweeps=1,max_bond_mode="per_sector"):
        self._configure_sweeps(nsweeps,algorithm,tol,min_sweeps,max_bond_mode)
        if not isinstance(self.D,Integral) or isinstance(self.D,bool) or self.D<1:
            raise ValueError('D must be a positive integer.')
        if cbe_selection!='shrewd':
            raise ValueError('Direct spatial NC CBE uses shrewd selection.')
        if not isinstance(cbe_expansion,Integral) or isinstance(cbe_expansion,bool) or cbe_expansion<1:
            raise ValueError('cbe_expansion must be a positive integer.')
        if cbe_preselection is not None and (not isinstance(cbe_preselection,Integral)
                or isinstance(cbe_preselection,bool) or cbe_preselection<1):
            raise ValueError('cbe_preselection must be a positive integer.')
        if algorithm=='two_site' and self._previous:
            raise NotImplementedError('NC two-site currently targets one complete multiplet.')
        self._cbe_expansion=int(cbe_expansion) if algorithm=='one_site_cbe' else 0
        self._cbe_preselection=cbe_preselection
        self._min_sweeps=int(min_sweeps)
        if any(not np.isfinite(x) or x<=0 for x in (tol,metric_rtol,local_tol)) or metric_rtol>=1:
            raise ValueError('Invalid NC convergence tolerances.')
        with threadpool_limits(limits=1):
            if algorithm=='two_site' and self.nsites>1:
                kernels.run_spatial_nc_two_site(self,nsweeps,tol,local_tol,verbose)
            else:
                kernels.run_spatial_nc_sweeps(self,nsweeps,tol,metric_rtol,local_tol,verbose)
        self.message='converged' if self.success else 'sweep limit reached'
        return self._store_multiplet_states()

    def _record_energy(self,h,norm,copy):
        kernels.record_spatial_nc_energy(self,h,norm,copy)

    def _boundaries(self,direction):
        return kernels.spatial_nc_boundaries(self,direction)

    def _refresh(self):
        h,n = self._boundaries('rl')
        self._record_energy(h[-1],n[-1],0)

    def _finish_cycle(self,h,n):
        self._record_energy(h,n,1)

    def _hamiltonian_action(self,site,left,right,*,assemble=False):
        action = self.nc.local_action(site,self.fusion_maps[site],left,right)
        return action.assemble() if assemble else action

    def _expand_bond(self,bond,direction,left,right,norm_left,norm_right):
        return kernels.expand_spatial_nc_bond(self,bond,direction,left,right,norm_left,norm_right)

    def _state_energy(self,state):
        tensors = state.tensors
        h = self.nc.edge(tensors[0].legs[0])
        norm = boundary(tensors[0].legs[0],self._identity_leg,False)
        for i,tensor in enumerate(tensors):
            h = self.nc.advance(i,h,tensor.data,tensor.data,'lr')
            norm = kernels.multiplet_norm_advance(self.sites[0].symmetry,norm,self.identity[i],tensor.data,tensor.data,'lr')
        return float((next(iter(h.values()))[0,0,0]/next(iter(norm.values())).item()).real+self.ecore)

    def close(self):
        super().close()
        self.nc.owner.clear_boundaries()
        self.nc._installed.clear()
        self.nc._route_cache.clear()
        self.nc._channel_connections.clear()
        self.nc._route_cache_size = 0
        self._cbe_decomposition = None
