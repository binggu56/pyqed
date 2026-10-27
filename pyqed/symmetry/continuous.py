"""Orbital SO(3), C(infinity)v and D(infinity)h irreps and exact fusion.

Angular coupling uses the Condon--Shortley convention (NIST DLMF 34.1,
https://dlmf.nist.gov/34.1) and the shared SU(2) CG implementation. These are
orbital symmetries, independent of electron spin. Cylindrical irreps retain
both +/-Lambda partners and reflection parity at Lambda=0; Dooh also carries
inversion parity. No finite-subgroup approximation or angular cutoff is used.
"""
from dataclasses import dataclass
from functools import lru_cache
import re
import numpy as np
from scipy.linalg import null_space


@dataclass(frozen=True)
class OrbitalRotationSymmetry:
    """Complete integer-angular-momentum irreps for shared Legs.

    Labels are L0, L1, ... for SO3; A1, A2, E1, ... for Coov; and A1g,
    A2g, E1g, A1u, A2u, E1u, ... for Dooh. E labels denote whole doublets.
    SO3 rotations alone do not encode inversion parity. Couplings adapt the
    Condon--Shortley CG convention to real orbital rows (NIST DLMF 34.1).
    """
    name: str

    def __post_init__(self):
        names = {'coov':'Coov', 'dooh':'Dooh', 'so3':'SO3'}
        if str(self.name).lower() not in names:
            raise ValueError('Expected Coov, Dooh or SO3.')
        object.__setattr__(self, 'name', names[self.name.lower()])

    def quantum_numbers(self, label):
        label = str(label)
        if self.name == 'SO3':
            match = re.fullmatch(r'L(0|[1-9]\d*)', label)
            if not match:
                raise ValueError('SO3 irreps use L0, L1, ... (integer orbital L).')
            return int(match[1]), 1, 1
        parity = 1
        if self.name == 'Dooh':
            if not label.endswith(('g', 'u')):
                raise ValueError('Dooh requires g/u inversion parity.')
            parity = 1 if label[-1] == 'g' else -1
            label = label[:-1]
        if label in ('A1', 'A2'):
            return 0, 1 if label == 'A1' else -1, parity
        match = re.fullmatch(r'E([1-9]\d*)', label)
        if not match:
            raise ValueError('Use a whole cylindrical irrep A1, A2 or E<n>, not x/y partners.')
        return int(match[1]), 1, parity

    def _label(self, order, reflection=1, parity=1):
        if self.name == 'SO3':
            return f'L{order}'
        label = f'E{order}' if order else ('A1' if reflection == 1 else 'A2')
        return label + (('g' if parity == 1 else 'u') if self.name == 'Dooh' else '')

    @property
    def identity(self):
        return self._label(0)

    def dual(self, irrep):
        return self._label(*self.quantum_numbers(irrep))

    def dimension(self, irrep):
        order, _, _ = self.quantum_numbers(irrep)
        return 2*order+1 if self.name == 'SO3' else (2 if order else 1)

    def irrep_id(self, label):
        order,reflection,parity = self.quantum_numbers(label)
        code = order if self.name == 'SO3' else (order+1 if order else int(reflection<0))
        return 2*code+int(parity<0) if self.name == 'Dooh' else code

    def irrep_name(self, code):
        if int(code) != code or code<0:
            raise ValueError('Orbital irrep ids must be nonnegative integers.')
        code = int(code)
        if self.name == 'SO3':
            return self._label(code)
        parity = 1
        if self.name == 'Dooh':
            code,odd = divmod(code,2)
            parity = -1 if odd else 1
        return self._label(max(0,code-1),-1 if code==1 else 1,parity)

    def fuse(self, left, right):
        a, ra, pa = self.quantum_numbers(left)
        b, rb, pb = self.quantum_numbers(right)
        if self.name == 'SO3':
            return tuple(self._label(c) for c in range(abs(a-b), a+b+1))
        parity = pa*pb
        if not a or not b:
            return (self._label(a+b, ra*rb, parity),)
        labels = ((self._label(0, 1, parity), self._label(0, -1, parity))
                  if a == b else (self._label(abs(a-b), 1, parity),))
        return labels + (self._label(a+b, 1, parity),)

    def allows(self, bra, op, ket):
        return bra in self.fuse(ket, op)

    def coupling(self, left, right, fused):
        return orbital_coupling(self, left, right, fused)

    def generators(self, irrep):
        """Hermitian angular generators and unitary discrete symmetries."""
        order, reflection, parity = self.quantum_numbers(irrep)
        if self.name == 'SO3':
            m = np.arange(order, -order-1, -1)
            jp = np.zeros((len(m), len(m)))
            for i in range(1, len(m)):
                jp[i-1,i] = np.sqrt((order-m[i])*(order+m[i]+1))
            u = real_spherical_transform(order)
            return {name: u.conj().T @ value @ u for name,value in
                    [('Lx', (jp+jp.T)/2), ('Ly', (jp-jp.T)/(2j)), ('Lz', np.diag(m))]}
        if order:
            result = {'Lz':np.array([[0, 1j*order], [-1j*order, 0]]),
                      'reflection':np.diag([1., -1.])}
        else:
            result = {'Lz':np.zeros((1,1)), 'reflection':np.array([[float(reflection)]])}
        if self.name == 'Dooh':
            result['inversion'] = parity*np.eye(self.dimension(irrep))
        return result


@lru_cache(maxsize=128)
def real_spherical_transform(l):
    """Columns: m=0, cosine/sine m=1,...,l in complex descending-m rows."""
    u = np.zeros((2*l+1,2*l+1), complex)
    u[l,0] = 1
    for m in range(1,l+1):
        u[l-m,2*m-1] = 1/np.sqrt(2)
        u[l+m,2*m-1] = (-1)**m/np.sqrt(2)
        u[l-m,2*m] = -1j/np.sqrt(2)
        u[l+m,2*m] = 1j*(-1)**m/np.sqrt(2)
    u.setflags(write=False)
    return u


@lru_cache(maxsize=1024)
def orbital_coupling(symmetry, left, right, fused):
    if fused not in symmetry.fuse(left, right):
        raise ValueError('Forbidden orbital fusion channel.')
    da, db, dc = map(symmetry.dimension, (left, right, fused))
    if symmetry.name == 'SO3':
        from pyqed.mps.nonabelian.fusion import irrep_coupling
        from .irrep import SU2Symmetry
        a,b,c = [symmetry.quantum_numbers(q)[0] for q in (left,right,fused)]
        cg = irrep_coupling(SU2Symmetry(), 2*a, 2*b, 2*c)
        out = np.einsum('ia,jb,ijk,kc->abc', real_spherical_transform(a).conj(),
                        real_spherical_transform(b).conj(), cg, real_spherical_transform(c))
        out *= 1j**(a+b-c)
        out = np.real_if_close(out, tol=1000)
    else:
        ga,gb,gc = [symmetry.generators(q) for q in (left,right,fused)]
        constraints = []
        for name in ga:
            total = (np.kron(ga[name],np.eye(db)) + np.kron(np.eye(da),gb[name])
                     if name == 'Lz' else np.kron(ga[name],gb[name]))
            constraints.append(np.kron(total,np.eye(dc))-np.kron(np.eye(da*db),gc[name].T))
        basis = null_space(np.vstack(constraints), rcond=1e-12)
        if basis.shape[1] != 1:
            raise ArithmeticError('Orbital coupling must have binary multiplicity one.')
        out = basis[:,0].reshape(da*db,dc)*np.sqrt(dc)
        pivot = out.flat[np.argmax(abs(out))]
        out *= np.conj(pivot)/abs(pivot)
        out = np.real_if_close(out.reshape(da,db,dc), tol=1000)
    if not np.allclose(out.reshape(-1,dc).conj().T @ out.reshape(-1,dc), np.eye(dc), atol=1e-12):
        raise ArithmeticError('Non-isometric orbital coupling.')
    out.setflags(write=False)
    return out


def orbital_subspaces(symmetry, generators, *, tol=1e-9):
    """Decompose a unitary orbital representation using its Lie generators.

    Returns (component, copy, row) intertwiners. Casimir eigenspaces and exact
    ladder operations replace finite-group projectors. All generators must be
    supplied in an orthonormal basis. No angular-momentum truncation is made.
    """
    from scipy.linalg import eigh
    symmetry = OrbitalRotationSymmetry(symmetry) if isinstance(symmetry,str) else symmetry
    g = {name: np.asarray(value,complex) for name,value in generators.items()}
    n = len(g['Lz'])
    eye = np.eye(n)
    required = ('Lx','Ly','Lz') if symmetry.name == 'SO3' else (
        ('Lz','reflection','inversion') if symmetry.name == 'Dooh' else ('Lz','reflection'))
    if any(name not in g for name in required):
        raise ValueError('Missing continuous-symmetry generator.')
    if any(a.shape != (n,n) or not np.allclose(a,a.conj().T,atol=tol,rtol=0) for a in g.values()):
        raise ValueError('Orbital generators must be Hermitian matrices of a common shape.')
    z = g['Lz']
    if symmetry.name == 'SO3':
        x,y = g['Lx'],g['Ly']
        for a,b,c in ((x,y,z),(y,z,x),(z,x,y)):
            if not np.allclose(a@b-b@a,1j*c,atol=tol,rtol=tol):
                raise ValueError('Orbital generators violate the SO3 Lie algebra.')
        casimir = x@x+y@y+z@z
    else:
        reflection = g['reflection']
        if not np.allclose(reflection@reflection,eye,atol=tol) or not np.allclose(reflection@z@reflection,-z,atol=tol):
            raise ValueError('Invalid cylindrical reflection.')
        casimir = z@z
        if 'inversion' in g:
            inversion = g['inversion']
            if not np.allclose(inversion@inversion,eye,atol=tol) or any(
                    not np.allclose(inversion@a,a@inversion,atol=tol) for a in g.values()):
                raise ValueError('Invalid inversion generator.')
    values,vectors = eigh(casimir)
    values[abs(values)<tol] = 0
    raw = (np.sqrt(np.maximum(0,1+4*values))-1)/2 if symmetry.name == 'SO3' else np.sqrt(np.maximum(0,values))
    orders = np.rint(raw).astype(int)
    if np.max(abs(raw-orders),initial=0) > tol*10:
        raise ValueError('The representation does not have integer orbital angular momentum.')
    result = {}
    for order in sorted(set(orders)):
        space = vectors[:,orders == order]
        if symmetry.name == 'SO3':
            restricted = z@space
            scale = np.linalg.norm(restricted,2)
            seeds = space if scale<tol else space@null_space(restricted,rcond=tol/scale)
            if seeds.shape[1]*(2*order+1) != space.shape[1]:
                raise ValueError('Incomplete SO3 multiplet.')
            # Choose real m=0 seeds whenever the input is a real orbital representation.
            projector = seeds@seeds.conj().T
            if np.max(abs(projector.imag),initial=0)<tol:
                val,vec = eigh(projector.real)
                seeds = vec[:,val>1-tol]
            highest = seeds.copy()
            for m in range(order):
                highest = (x+1j*y)@highest/np.sqrt((order-m)*(order+m+1))
            rows = [highest]
            for m in range(order,-order,-1):
                rows.append((x-1j*y)@rows[-1]/np.sqrt((order+m)*(order-m+1)))
            result[f'L{order}'] = np.einsum('ncm,mr->ncr',np.stack(rows,axis=-1),real_spherical_transform(order))
        else:
            for parity in ((1,-1) if symmetry.name == 'Dooh' else (1,)):
                parity_space = space
                if 'inversion' in g:
                    val,vec = eigh(space.conj().T@g['inversion']@space)
                    parity_space = space@vec[:,abs(val-parity)<tol]
                if not parity_space.shape[1]:
                    continue
                for reflection_sign in ((1,-1) if order == 0 else (1,)):
                    val,vec = eigh(parity_space.conj().T@g['reflection']@parity_space)
                    seeds = parity_space@vec[:,abs(val-reflection_sign)<tol]
                    if not seeds.shape[1]:
                        continue
                    projector = seeds@seeds.conj().T
                    if np.max(abs(projector.imag),initial=0)<tol:
                        val,vec = eigh(projector.real)
                        seeds = vec[:,val>1-tol]
                    rows = np.stack([seeds,1j*z@seeds/order],axis=-1) if order else seeds[:,:,None]
                    result[symmetry._label(order,reflection_sign,parity)] = rows
    basis = np.column_stack([a.reshape(n,-1) for a in result.values()]) if result else np.empty((n,0))
    if basis.shape != (n,n) or not np.allclose(basis@basis.conj().T,eye,atol=tol*10):
        raise ValueError('Incomplete continuous orbital multiplets.')
    return {name:np.real_if_close(value,tol=1000) for name,value in result.items()}


class OrbitalGeneratorAction:
    """Validate a continuous orbital representation and invariant integrals.

    Angular commutators enforce the full connected group; reflection and
    inversion generators enforce its disconnected components. This does not
    infer continuous invariance from a finite subgroup.
    """
    def __init__(self, generators, wfnsym, *, point_group, tol=1e-8):
        self.group = OrbitalRotationSymmetry(point_group)
        self.wfnsym = self.group.dual(wfnsym)
        self.dimension = self.group.dimension(self.wfnsym)
        self.generators = {k:np.asarray(v,complex) for k,v in generators.items()}
        self.names = tuple(self.generators)
        self.matrices = tuple(self.generators.values())
        self.subspaces = orbital_subspaces(self.group,self.generators,tol=tol)
        self.norb = len(self.generators['Lz'])
        self.tol = tol

    def validate_hamiltonian(self, h1e, eri=None):
        h = np.asarray(h1e)
        if h.shape != (self.norb,self.norb) or not np.allclose(h,h.conj().T,atol=self.tol):
            raise ValueError('Expected a Hermitian Hamiltonian in the generator basis.')
        for name,g in self.generators.items():
            if name.startswith('L'):
                error = g@h-h@g
            else:
                error = g.conj().T@h@g-h
            if np.max(abs(error),initial=0)>self.tol*max(1,np.max(abs(h),initial=0)):
                raise ValueError('One-electron Hamiltonian breaks continuous orbital symmetry.')
            if eri is None:
                continue
            v = np.asarray(eri)
            if v.shape != (self.norb,)*4:
                raise ValueError('ERI and orbital generators have different dimensions.')
            if name.startswith('L'):
                error = (np.einsum('pa,pqrs->aqrs',g.conj(),v)
                         - np.einsum('qb,pqrs->pbrs',g,v)
                         + np.einsum('rc,pqrs->pqcs',g.conj(),v)
                         - np.einsum('sd,pqrs->pqrd',g,v))
            else:
                error = np.einsum('pa,qb,rc,sd,pqrs->abcd',g.conj(),g,g.conj(),g,v,optimize=True)-v
            if np.max(abs(error),initial=0)>self.tol*max(1,np.max(abs(v),initial=0)):
                raise ValueError('Two-electron Hamiltonian breaks continuous orbital symmetry.')
