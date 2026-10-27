"""Atomic, reduced-state checkpoints for complete-multiplet NC sweeps.

Only reduced coefficients and whole-irrep bond multiplicities are persisted.
Structural maps and integral-owned environments are rebuilt by the solver;
no compiled pointer, component wavefunction, or Hamiltonian MPO is serialized.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
from pyqed.symmetry import Irrep, Leg
from .fusion import TensorFusionMap


def problem_signature(solver,h,v,cutoff):
    digest=hashlib.sha256()
    digest.update(json.dumps([solver.target_sector.charge,
        [s.orbital_irrep for s in solver.sites],solver.sites[0].group,
        float(solver.ecore),float(cutoff)]).encode())
    for array in (h,v,solver.orbital_transform):
        array=np.ascontiguousarray(array,dtype=np.complex128)
        digest.update(str(array.shape).encode());digest.update(array.tobytes())
    return digest.hexdigest()


def save_checkpoint(solver,path):
    if not getattr(solver,'_checkpoint_ready',False) or not solver.history or len(solver.states)!=solver.partner_dimension:
        raise ValueError('Save NC checkpoints after a completed sweep run of one multiplet.')
    metadata=dict(version=1,problem=solver._problem_signature,D=int(solver.D),
        completed_sweeps=int(solver.completed_sweeps),energy=float(solver.energy),
        history=solver.history,legs=[[[list(q.charge),int(leg.dims[q])]
            for q in leg.sectors] for leg in solver.bond_legs])
    arrays={f'site_{i}':np.asarray(v) for i,v in enumerate(solver.vectors)}
    arrays['metadata']=np.array(json.dumps(metadata,default=lambda x:x.item()))
    path=Path(path).expanduser();path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            np.savez(stream,**arrays);stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)
    return path


def load_checkpoint(solver,path):
    with np.load(path,allow_pickle=False) as data:
        metadata=json.loads(str(data['metadata']))
        if metadata.get('version')!=1 or metadata.get('problem')!=solver._problem_signature:
            raise ValueError('NC checkpoint belongs to a different Hamiltonian, basis, target, or format.')
        entries=metadata['legs']
        if len(entries)!=solver.nsites+1 or metadata['D']<1 or metadata['completed_sweeps']<0:
            raise ValueError('Invalid checkpoint bond layout.')
        legs=[]
        for entries_on_bond in entries:
            dims={Irrep(tuple(charge)):int(dim) for charge,dim in entries_on_bond}
            if len(dims)!=len(entries_on_bond) or any(d<1 or d>metadata['D'] or q not in solver.nc.ids for q,d in dims.items()):
                raise ValueError('Invalid checkpoint sector multiplicities.')
            legs.append(Leg(dims,symmetry=solver.sites[0].symmetry))
        if legs[0].dims!=solver.bond_legs[0].dims or legs[-1].dims!=solver.bond_legs[-1].dims:
            raise ValueError('Checkpoint boundary target differs.')
        maps=tuple(TensorFusionMap(legs[i],site.leg,legs[i+1]) for i,site in enumerate(solver.sites))
        vectors=[np.asarray(data[f'site_{i}']).copy() for i in range(solver.nsites)]
        if any(v.shape!=(f.parameter_count,) or not np.isfinite(v).all() for v,f in zip(vectors,maps)):
            raise ValueError('Invalid checkpoint reduced coefficients.')
    # Validate everything above before replacing the current state. The fresh
    # integral owner supplies the structural ranks; boundaries are recomputed.
    previous=(solver.bond_legs,solver.fusion_maps,solver.vectors)
    solver.nc.owner.clear_boundaries();solver.nc._installed.clear()
    solver.bond_legs=list(legs);solver.fusion_maps=maps;solver.vectors=vectors
    try:
        solver._refresh()
        if not np.isfinite(solver.energy) or not np.isclose(solver.energy,metadata['energy'],atol=1e-10,rtol=0):
            raise ValueError('Checkpoint energy does not match its reduced state.')
    except Exception:
        solver.bond_legs,solver.fusion_maps,solver.vectors=previous
        solver.nc.owner.clear_boundaries();solver.nc._installed.clear()
        solver._refresh()
        raise
    solver.D=int(metadata['D']);solver.completed_sweeps=int(metadata['completed_sweeps'])
    solver._previous=[];solver._rdm_cache={}
    solver.history=metadata['history']
    solver.success=solver.converged=False
    solver.message='checkpoint restored; convergence requires resumed sweeps'
    solver._store_multiplet_states()
    solver._checkpoint_ready=True
    return solver
