"""Moving reduced boundaries and local actions for the compiled spatial NC owner.

The Hamiltonian is the integral-owned normal/complementary schedule in
``System``. Python prepares only reduced block topology and cached structural
CG weights. Multiplicity contractions, sparse Hamiltonian actions and their
exact diagonals execute in the compiled moving environment. No Hamiltonian
MPO or global Hilbert-space projection is constructed.
"""
from collections import defaultdict, OrderedDict
import numpy as np
from pyqed.symmetry import Irrep
from .multiplet_operator import structural_weight,dimension
from .nc_spatial import prepare_rank_schedule


class SpatialNCEnvironment:
    """Direct U(1) x SU(2) x orbital-rank NC environment owner.

    Adapts Chan and Head-Gordon, JCP 116, 4462 (2002),
    doi:10.1063/1.1449459, to complete local orbital multiplets using the
    product-irrep CG construction of Singh and Vidal, PRB 86, 195114 (2012),
    doi:10.1103/PhysRevB.86.195114. Screened real integrals are supported;
    state tensors and environments may be complex. All structural partners
    are retained. Only reduced multiplicity axes enter numerical sweeps.
    """
    def __init__(self,h,v,sites,blocks,state_legs,*,nelec,spin,cutoff,n_threads):
        from ._su2_kernel import SU2MovingEnvironment
        if np.iscomplexobj(np.real_if_close(h)) or np.iscomplexobj(np.real_if_close(v)):
            raise ValueError('The direct spatial NC integral owner requires real integrals.')
        self.symmetry = sites[0].symmetry
        self.owner = SU2MovingEnvironment(np.ascontiguousarray(np.real(h)),
            np.ascontiguousarray(np.real(v)),nelec,two_s=spin,cutoff=cutoff)
        self.owner.set_num_threads(n_threads)
        offsets = [0]+[block[-1]+1 for block in blocks]
        self.owner.configure_spatial_shells(offsets)
        self.ranks,self.legs = prepare_rank_schedule(self.owner,sites,offsets,state_legs=state_legs)
        self.ids = {q:i for i,q in enumerate(self.ranks)}
        self.zero = Irrep((0,0,self.symmetry.factors[-1].identity))
        self.records = []
        self.physical_routes = []
        for site in range(len(sites)):
            labels,_,_ = self.owner.spatial_nc_rank_plan(site)
            self.records.append(labels)
            groups = defaultdict(list)
            for index,row in enumerate(labels):
                l,r,p,q,k = (self.ranks[i] for i in row[:5])
                groups[p,q,l,r,k].append(index)
            by_physical = defaultdict(list)
            for (p,q,l,r,k),indices in groups.items():
                by_physical[p,q].append((l,r,k,indices))
            self.physical_routes.append(by_physical)
        self._installed = {}
        self._route_cache = OrderedDict()
        self._route_cache_size = 0
        self._route_cache_limit = 200_000
        self._channel_connections = {}

    def pack(self,blocks):
        arrays = tuple(blocks.values())
        values = np.concatenate([np.asarray(x,dtype=complex).ravel() for x in arrays]) if arrays else np.empty(0,complex)
        offsets = np.asarray([0]+list(np.cumsum([x.size for x in arrays])),dtype=np.int64)
        shapes = np.asarray([x.shape for x in arrays],dtype=np.int64).reshape(-1,3)
        labels = np.asarray([[self.ids[q] for q in key] for key in blocks],dtype=np.int64).reshape(-1,3)
        return values,offsets,shapes,labels

    @staticmethod
    def _views(values,keys,offsets,shapes):
        return {key:values[offsets[i]:offsets[i+1]].reshape(shapes[i]) for i,key in enumerate(keys)}

    def install(self,side,bond,blocks):
        packed = self.pack(blocks)
        values = self.owner.install_spatial_nc_boundary(side,bond,packed)
        result = self._views(values,tuple(blocks),packed[1],packed[2])
        self._installed[side,bond] = result
        return result

    def edge(self,state_leg,*,right=False):
        cut = len(self.legs)-1 if right else 0
        q = state_leg.sectors[0]
        data = np.zeros((1,self.legs[cut].dims[self.zero],1),complex)
        data[0,0 if right else 1,0] = 1/dimension(self.symmetry,q) if right else 1
        return self.install('right' if right else 'left',cut,{(q,self.zero,q):data})

    def _structural_routes(self,site,bra_keys,ket_keys):
        # Routes depend on irrep keys, never on state values or multiplicities.
        key = site, tuple(bra_keys), tuple(ket_keys)
        cached = self._route_cache.get(key)
        if cached is not None:
            self._route_cache.move_to_end(key)
            yield from cached
            return
        records = []
        for record in self._iter_structural_routes(*key):
            if records is not None:
                records.append(record)
                if len(records) > self._route_cache_limit:
                    records = None
            yield record
        if records is not None:
            while self._route_cache and (self._route_cache_size + len(records) > self._route_cache_limit
                                        or len(self._route_cache) >= 64):
                _, old = self._route_cache.popitem(last=False)
                self._route_cache_size -= len(old)
            self._route_cache[key] = tuple(records)
            self._route_cache_size += len(records)

    def _channel_connections_for(self,bra,ket):
        key=bra,ket
        cached=self._channel_connections.get(key)
        if cached is not None:
            return cached
        an,aj,ao=bra.charge;bn,bj,bo=ket.charge
        # NC cut channels contain at most two open fermion indices. All
        # supported orbital irreps are real and self-dual, so reciprocity
        # gives l in a x b iff a is in b x l.
        if abs(an-bn)>2 or abs(aj-bj)>2:
            result=frozenset()
        else:
            orbitals=self.symmetry.factors[-1].fuse(ao,bo)
            result=frozenset((an-bn,j,o) for j in range(abs(aj-bj),min(2,aj+bj)+1,2)
                             for o in orbitals)
        if len(self._channel_connections)<32768:
            self._channel_connections[key]=result
        return result

    def _iter_structural_routes(self,site,bra_keys,ket_keys):
        for ai,(a,p,c) in enumerate(bra_keys):
            for bi,(b,q,d) in enumerate(ket_keys):
                left=self._channel_connections_for(a,b)
                if not left:continue
                right=self._channel_connections_for(c,d)
                if not right:continue
                for l,r,k,entries in self.physical_routes[site].get((p,q),()):
                    if l.charge not in left or r.charge not in right:continue
                    weight = structural_weight(self.symmetry,a.charge,p.charge,c.charge,
                        b.charge,q.charge,d.charge,l.charge,k.charge,r.charge)
                    if abs(weight)>1e-13:
                        yield ai,bi,(a,l,b),(c,r,d),float(np.real(weight)),entries

    def advance(self,site,parent,bra,ket,direction):
        if direction not in ('lr','rl'):
            raise ValueError('direction must be lr or rl')
        left = direction == 'lr'
        side = 'left' if left else 'right'
        parent_bond,child_bond = (site,site+1) if left else (site+1,site)
        if self._installed.get((side,parent_bond)) is not parent:
            parent = self.install(side,parent_bond,parent)
        positions = {key:i for i,key in enumerate(parent)}
        targets,shapes,routes,weights = {},[],[],[]
        a_keys,b_keys = tuple(bra),tuple(ket)
        a_values,b_values = tuple(bra.values()),tuple(ket.values())
        for ai,bi,lk,rk,weight,entries in self._structural_routes(site,a_keys,b_keys):
            source,target = (lk,rk) if left else (rk,lk)
            if source not in positions:
                continue
            if target not in targets:
                targets[target] = len(targets)
                axis = 2 if left else 0
                shapes.append((a_values[ai].shape[axis],self.legs[child_bond].dims[target[1]],b_values[bi].shape[axis]))
            weight /= dimension(self.symmetry,a_keys[ai][2 if left else 0])
            for entry in entries:
                routes.append((positions[source],ai,bi,entry,targets[target]))
                weights.append(weight)
        offsets = np.asarray([0]+list(np.cumsum([np.prod(shape,dtype=np.int64) for shape in shapes])),dtype=np.int64)
        labels = np.asarray([[self.ids[q] for q in key] for key in targets],dtype=np.int64).reshape(-1,3)
        values = self.owner.advance_spatial_nc_boundary(side,parent_bond,child_bond,site,
            routes,weights,self.pack(bra)[:3],self.pack(ket)[:3],(offsets,np.asarray(shapes,dtype=np.int64).reshape(-1,3),labels))
        result = self._views(values,tuple(targets),offsets,shapes)
        self._installed[side,child_bond] = result
        return result

    def local_action(self,site,fusion,left,right):
        if self._installed.get(('left',site)) is not left:
            self.install('left',site,left)
        if self._installed.get(('right',site+1)) is not right:
            self.install('right',site+1,right)
        lpos,rpos = ({key:i for i,key in enumerate(blocks)} for blocks in (left,right))
        keys = tuple(fusion.blocks)
        routes,weights = [],[]
        for ai,bi,lk,rk,weight,entries in self._structural_routes(site,keys,keys):
            if lk not in lpos or rk not in rpos:
                continue
            for entry in entries:
                routes.append((ai,bi,lpos[lk],rpos[rk],entry))
                weights.append(weight)
        offsets = [span.start for span,_ in fusion.blocks.values()]+[fusion.parameter_count]
        shapes = [shape for _,shape in fusion.blocks.values()]
        return self.owner.install_spatial_nc_local(site,routes,weights,offsets,shapes)
