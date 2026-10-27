"""Direct reduced operator growth for spatial complementary families.

The local recursion is a Wigner--Eckart product-group 9j contraction, adapting
Singh and Vidal, PRB 86, 195114 (2012), doi:10.1103/PhysRevB.86.195114.
It carries multiplicities through truncation and uses the shared structural
CG cache. It constructs neither MPO cores nor component-space block operators.
These NumPy primitives are independent reference contractions for coupled
families. Molecular production sweeps use the compiled System and
MovingEnvironment through nc_multiplet, rather than a Python family owner.
"""
import numpy as np
from pyqed.symmetry import Irrep, Leg
from .multiplet_operator import structural_weight
from .multiplet_dmrg import _contract


def grow_operator(symmetry, left, local, bra, ket, *, left_rank, local_rank, rank, direction="lr"):
    """Grow a coupled reduced operator into the truncated outgoing bond.

    Operator dictionaries map (bra irrep, ket irrep) to multiplicity matrices.
    MPS dictionaries map (left, physical, right) irreps to reduced tensors.
    In ``rl`` growth, ``left`` holds the incoming right environment; the
    three rank arguments retain their left-to-right meaning.
    Operator ranks are charge tuples for product symmetry, or scalar labels
    for a single symmetry. Fermionic parity strings belong in the supplied
    operators; this contraction does not infer or insert their signs.
    """
    if direction not in ("lr", "rl"):
        raise ValueError("direction must be lr or rl")
    if rank not in symmetry.fuse(left_rank,local_rank):
        raise ValueError('Forbidden complementary operator fusion channel.')
    output = {}
    for (a,p,c),av in bra.items():
        for (b,q,d),bv in ket.items():
            source = (a,b) if direction == "lr" else (c,d)
            target = (c,d) if direction == "lr" else (a,b)
            if source not in left or (p,q) not in local:
                continue
            if c.charge not in symmetry.fuse(d.charge,rank):
                continue
            weight = structural_weight(symmetry,a.charge,p.charge,c.charge,
                b.charge,q.charge,d.charge,left_rank,local_rank,rank)
            if abs(weight)<1e-14:
                continue
            weight /= Leg._irrep_dimension(symmetry,(c if direction == "lr" else a).charge)
            expression = ('apc,ab,pq,bqd->cd' if direction == "lr"
                          else 'apc,cd,pq,bqd->ab')
            value = weight*_contract(expression,np.asarray(av).conj(),
                                     left[source],local[p,q],bv)
            if target in output:
                output[target] += value
            else:
                output[target] = value
    return output


def update_families(symmetry, environment, local, bra, ket, transitions, *, direction="lr"):
    """Update named reduced families through coupled, coefficient-weighted routes.

    Each transition is ``(left_name, local_name, right_name, left_rank,
    local_rank, right_rank, coefficient)``. Names include orbital indices and
    coupling channels where required; repeated routes accumulate coherently.
    Identity/spectator and fermion parity operators must be supplied explicitly.
    No partner rows are truncated or selected by this recursion.
    Adapts Singh and Vidal, Phys. Rev. B 86, 195114 (2012),
    doi:10.1103/PhysRevB.86.195114, using closed product-group CG networks.
    """
    if direction not in ("lr", "rl"):
        raise ValueError("direction must be lr or rl")
    output = {}
    for l,k,r,lr,kr,rr,coefficient in transitions:
        source,target = (l,r) if direction == "lr" else (r,l)
        if source not in environment or k not in local:
            continue
        blocks = grow_operator(symmetry,environment[source],local[k],bra,ket,
            left_rank=lr,local_rank=kr,rank=rr,direction=direction)
        destination = output.setdefault(target,{})
        for key,value in blocks.items():
            if key in destination:
                destination[key] += coefficient*value
            else:
                destination[key] = coefficient*value
    return output


def local_action(symmetry, left, local, right, ket, bra_keys, transitions):
    """Apply a family Hamiltonian to local reduced tensors, in the CG metric.

    Uses the same transition convention as :func:`update_families`. The right
    boundary includes the inverse dimension of the terminal target irrep.
    The returned action is covariant (metric weighted); callers must solve
    the generalized local eigenproblem or whiten the norm separately.
    This is a local contraction, never a projection of a global Hilbert basis.
    Adapts Singh and Vidal, Phys. Rev. B 86, 195114 (2012),
    doi:10.1103/PhysRevB.86.195114; fermion signs and Hamiltonian coefficients
    are supplied by the caller, rather than inferred here.
    """
    output = {}
    bra_keys = tuple(bra_keys)
    for l,k,r,lr,kr,rr,coefficient in transitions:
        if rr not in symmetry.fuse(lr,kr):
            raise ValueError("Forbidden complementary operator fusion channel.")
        if l not in left or k not in local or r not in right:
            continue
        for a,p,c in bra_keys:
            for (b,q,d),value in ket.items():
                if (a,b) not in left[l] or (p,q) not in local[k] or (c,d) not in right[r]:
                    continue
                weight = structural_weight(symmetry,a.charge,p.charge,c.charge,
                    b.charge,q.charge,d.charge,lr,kr,rr)
                if abs(weight)<1e-14:
                    continue
                block = coefficient*weight*_contract('ab,pq,cd,bqd->apc',
                    left[l][a,b],local[k][p,q],right[r][c,d],value)
                key = (a,p,c)
                if key in output:
                    output[key] += block
                else:
                    output[key] = block
    return output
