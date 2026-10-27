"""Structural maps for direct NC charge/spin/orbital operator channels.

Uses exact product-irrep CG maps (Singh and Vidal, PRB 86, 195114, 2012,
doi:10.1103/PhysRevB.86.195114) on the normal/complementary partition of
Chan and Head-Gordon, JCP 116, 4462 (2002), doi:10.1063/1.1449459.
Only bounded local Fock operators are reduced. No molecular Hilbert basis,
operator-train fit, or Hamiltonian MPO conversion is performed.
"""
from collections import defaultdict
import numpy as np
from pyqed.symmetry import Irrep, Leg, SU2Symmetry
from .fusion import irrep_coupling


def channel_basis(symmetry, descriptors, sites, offsets):
    """Exact rank basis for canonical normal and complementary channels.

    Each normal word owns its ordered operator product and the parity of
    the unconsumed word. Complementary channels transform contragrediently
    to their outstanding operator word. Spin dualization uses the shared
    Condon--Shortley metric; orbital rows are in real, self-dual bases.
    Returns a multiplicity Leg and a sparse expansion for each raw channel.
    All channels in an orbital component group must be present.
    """
    spatial = symmetry.factors[-1]
    spin = SU2Symmetry()
    zero = Irrep((0,0,spatial.identity))
    groups = defaultdict(list)
    for index,(kind,first,second) in enumerate(descriptors):
        modes = [mode for mode in (first,second) if mode]
        labels = []
        components = []
        for mode in modes:
            orbital,ms = divmod(abs(mode)-1,2)
            site = int(np.searchsorted(offsets,orbital,side='right')-1)
            labels.append((1 if mode>0 else -1,site))
            components.append((orbital-offsets[site],ms))
        groups[(kind,tuple(labels))].append((index,tuple(components)))
    dims = defaultdict(int)
    expansions = [[] for _ in descriptors]
    dual = np.array([[0.,-1.],[1.,0.]])
    for (kind,labels),entries in groups.items():
        if not labels:
            copy = dims[zero]
            dims[zero] += 1
            if len(entries) != 1:
                raise ValueError('Scalar NC spine has duplicate channels.')
            expansions[entries[0][0]] = [(zero,copy,0,1.)]
            continue
        orbitals = [sites[site].orbital_irrep for _,site in labels]
        matrices = [np.eye(2) if charge>0 else dual for charge,_ in labels]
        charge = sum(q for q,_ in labels)*(-1 if kind == 3 else 1)
        if len(labels) == 1:
            sectors = [(1,orbitals[0])]
        else:
            sectors = [(j,l) for j in (0,2) for l in spatial.fuse(*orbitals)]
        expected = np.prod([2*spatial.dimension(l) for l in orbitals])
        if len(entries) != expected:
            raise ValueError('NC operator channel is missing orbital/spin partners.')
        columns = []
        for j,l in sectors:
            sector = Irrep((charge,j,l))
            copy = dims[sector]
            dims[sector] += 1
            dimension = (j+1)*spatial.dimension(l)
            coefficient = np.zeros((len(entries),dimension))
            metric = (irrep_coupling(spin,j,j,0).reshape(j+1,j+1)*np.sqrt(j+1)
                      if kind == 3 else np.eye(j+1))
            if len(labels) == 2:
                cg_spin = irrep_coupling(spin,1,1,j)
                cg_space = irrep_coupling(spatial,*orbitals,l)
            for row,(index,components) in enumerate(entries):
                if len(labels) == 1:
                    orbital,ms = components[0]
                    value = np.zeros((2,spatial.dimension(l)))
                    value[:,orbital] = matrices[0][ms]
                else:
                    (a,sa),(b,sb) = components
                    value = np.einsum('u,v,uvj,r->jr',matrices[0][sa],matrices[1][sb],
                                      cg_spin,cg_space[a,b])
                coefficient[row] = (metric.T@value).ravel()
                for partner in np.flatnonzero(abs(coefficient[row])>1e-14):
                    expansions[index].append((sector,copy,int(partner),float(coefficient[row,partner])))
            columns.append(coefficient)
        # This is a small structural CG isometry, not numerical rank fitting.
        unitary = np.column_stack(columns)
        if not np.allclose(unitary.T@unitary,np.eye(len(entries)),atol=1e-12,rtol=0):
            raise ArithmeticError('NC channel CG map is not complete and isometric.')
    return Leg(dict(dims),symmetry=symmetry),expansions


def prepare_rank_schedule(owner, sites, offsets, *, state_legs=()):
    """Install exact structural maps into the compiled molecular NC owner.

    Local primitives are projected onto an orthonormal Wigner--Eckart
    operator basis. Virtual normal/complementary channels are transformed
    with their analytic one-/two-fermion CG maps, without numerical fitting
    or removal of any partner. The integral-weighted summation runs in C++.
    """
    from .multiplet_operator import OperatorBasis, dimension
    from .fusion import full_slices
    sym = sites[0].symmetry
    spatial = sym.factors[-1]
    raw = owner.spatial_normal_complementary_schedule()
    ranks = {}
    def rank_id(q):
        if q not in ranks:
            ranks[q] = len(ranks)
        return ranks[q]
    for leg in state_legs:
        for q in leg.sectors:
            rank_id(q)
    legs,channels = [],[]
    for descriptors in raw['channels']:
        leg,expansion = channel_basis(sym,descriptors,sites,offsets)
        legs.append(leg)
        channels.append([[(rank_id(q),copy,row,value) for q,copy,row,value in entries]
                         for entries in expansion])
    physical = []
    operator_ranks = []
    for site,plan in zip(sites,raw['sites']):
        structural_size = sum(multiplicity*dimension(sym,q)**2 for q,multiplicity in site.leg.dims.items())
        if 16*structural_size**2 > 128*1024**2:
            raise MemoryError('Direct NC local operator basis exceeds its 128 MiB workspace budget.')
        basis = OperatorBasis(site.leg)
        spans = full_slices(basis.leg)
        maps = []
        used = set()
        for primitive in plan['primitives']:
            vector = basis.transform @ (site.basis.conj().T@primitive@site.basis).ravel()
            entries = []
            for k,span in spans.items():
                block = vector[span].reshape(basis.leg.dims[k],dimension(sym,k))
                for copy,row in zip(*np.nonzero(abs(block)>1e-13)):
                    p,q,i,j = basis.entries[k][copy]
                    value = block[copy,row]*np.sqrt(dimension(sym,k)/dimension(sym,p))
                    entries.append((rank_id(p),rank_id(q),rank_id(k),i,j,int(row),value))
                    used.add(k)
            maps.append(entries)
        physical.append(maps)
        operator_ranks.append(used)
    couplings = {}
    for site,used in enumerate(operator_ranks):
        for a in legs[site].sectors:
            for k in used:
                for charge in sym.fuse(a.charge,k.charge):
                    b = Irrep(charge)
                    if b not in legs[site+1].dims:
                        continue
                    key = rank_id(a),rank_id(k),rank_id(b)
                    if key not in couplings:
                        cg = np.real_if_close(irrep_coupling(sym,a.charge,k.charge,b.charge))
                        if np.iscomplexobj(cg):
                            raise ValueError('The NC orbital basis must use real structural CG maps.')
                        couplings[key] = np.asarray(cg,dtype=float)
    group_code = {'Coov':1,'Dooh':2,'SO3':3}.get(spatial.name,4)
    dimensions, products = [], []
    if group_code == 4:
        labels = tuple(spatial.group.irrep_names)
        ids = {q:i for i,q in enumerate(labels)}
        dimensions = [spatial.dimension(q) for q in labels]
        products = [spatial.fuse(a,b).count(c) for a in labels for b in labels for c in labels]
        quantum_numbers = [(q.charge[0],q.charge[1],ids[q.charge[2]],1,1) for q in ranks]
    else:
        quantum_numbers = [(q.charge[0],q.charge[1],*spatial.quantum_numbers(q.charge[2]))
                           for q in ranks]
    owner.reduce_spatial_normal_complementary(
        group_code,quantum_numbers,channels,physical,
        [(*key,value) for key,value in couplings.items()],dimensions,products)
    return tuple(ranks),tuple(legs)
