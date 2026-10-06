"""Explicit orbital counter-transformations of reduced chain and NN/NNN LETTA states.

Adaptation of Malmqvist, IJQC 30, 479 (1986), doi:10.1002/qua.560300404,
and Knecht et al., JCTC 12, 5881 (2016), doi:10.1021/acs.jctc.6b00889.
Pivoted, balanced LU replaces their particular orbital factorization. Local
exterior-power gates and Wigner 6j recoupling act on reduced state tensors;
no MPS owner, magnetic components, or many-body overlap operator is built.
NN/NNN physical ties use conditional three/four-site updates. NNN fusion
selectors use three-site updates. Ties beyond range two are unsupported.
"""

import numpy as np
from scipy.linalg import lu, solve_triangular

from pyqed.mps.nonabelian.reduced_overlap import (
    _SPIN, _fuse, _recouple, _two_orbital_gate, _size_guard,
    _left_gauge, _right_gauge, _split, _bond_dim, contract_overlap,
)


class OverlapConvergenceError(RuntimeError):
    """The computed truncation bound exceeds the requested absolute tolerance."""

    def __init__(self, tol, info):
        self.tol, self.info = tol, info
        super().__init__(
            f"Reduced overlap truncation bound {info['overlap_error_bound']:.6g} "
            f"exceeds tol={tol:.6g} with max_bond={info['max_bond']}. "
            "Increase max_bond, remove the cap, or relax tol.")


def _target(state):
    target = getattr(state, 'target_sector', None)
    if target is None:
        sectors = state.tensors[-1].legs[getattr(state, "rv_idx", 2)].sectors
        if len(sectors) != 1:
            raise NotImplementedError('Reduced orbital overlap requires one terminal charge/spin sector.')
        target = sectors[0]
    return target


def _check_tol(tol):
    if tol is not None and (isinstance(tol, (bool, np.bool_)) or not np.isfinite(tol) or tol <= 0):
        raise ValueError('tol must be finite and positive, or None for cap-only compression.')


def _split_budgets(circuit, state_tol, target_dim, splits_per_gate=2):
    if state_tol is None:
        return [None]*len(circuit)
    # Budget each conditional split's final-state loss
    # equally, undoing all subsequent gate amplification before the local SVD.
    allowance = state_tol / max(1, splits_per_gate*len(circuit))
    budgets = []
    for kind, *payload in reversed(circuit):
        budgets.append(target_dim*allowance**2)
        if kind == 'gate':
            allowance /= np.prod(np.maximum(1.0, np.linalg.svd(payload[1], compute_uv=False)))**2
        elif kind == 'diagonal':
            for scale in payload[0]:
                allowance /= max(1.0, abs(scale))**2
    return budgets[::-1]


def _transformed_norm_bound(state, matrix):
    if getattr(state, 'graph', ()) and (state.tie == 'physical' or any(b-a > 1 for a, b in state.graph)):
        from pyqed.letta.conditional_orbitals import _read, _norm as norm
        value = norm(_read(state), _target(state), state.tie)
    else:
        value = _norm(_read_sites(state), _target(state))
    value *= np.prod(np.maximum(1.0, np.linalg.svd(matrix, compute_uv=False)))**2
    if not np.isfinite(value):
        raise FloatingPointError('Orbital transformation norm bound is nonfinite.')
    return value


def _check_controls(max_bond, cutoff, memory_limit):
    if max_bond is not None and (isinstance(max_bond, (bool, np.bool_)) or int(max_bond) != max_bond or max_bond < 1):
        raise ValueError('max_bond must be a positive integer or None.')
    if not np.isfinite(cutoff) or cutoff < 0:
        raise ValueError('cutoff must be finite and nonnegative.')
    if memory_limit is not None and (isinstance(memory_limit, (bool, np.bool_)) or int(memory_limit) != memory_limit or memory_limit < 1):
        raise ValueError('memory_limit must be a positive integer or None.')


def _check_graph(state, name='state'):
    unsupported = tuple(edge for edge in getattr(state, "graph", ()) if not 1 <= edge[1]-edge[0] <= 2)
    if unsupported or (getattr(state, "graph", ()) and getattr(state, "tie", None) not in {'physical', 'fusion'}):
        raise NotImplementedError(
            f"Unsupported {name} graph ties for orbital transformation: "
            f"tie={getattr(state, "tie", None)!r}, edges={unsupported or getattr(state, "graph", ())!r}. "
            "Only untied states and NN/NNN physical or fusion ties are supported."
        )


def _matrix(matrix, n):
    matrix = np.asarray(matrix, complex)
    if matrix.shape != (n, n) or not np.all(np.isfinite(matrix)):
        raise ValueError('s must be a finite square matrix matching the site count.')
    return matrix


def orbital_factors(metric):
    """Return orbital maps XL, XR and inverse coefficient maps A, B.

    More precisely XL=A^-1, XR=B^-1, A† B=s. No pseudoinverse is used:
    numerically singular cross metrics cannot define a biorthonormal basis.
    """
    n = len(metric)
    singular = np.linalg.svd(metric, compute_uv=False)
    if singular[-1] <= np.finfo(float).eps * n * singular[0]:
        raise np.linalg.LinAlgError('A singular orbital metric has no biorthonormal basis.')
    p, lower, upper = lu(metric)
    scale = np.sqrt(np.abs(np.diag(upper)))
    a = scale[:, None] * lower.conj().T @ p.conj().T
    b = upper / scale[:, None]
    xl = p @ solve_triangular(lower.conj().T, np.diag(1/scale), lower=False, unit_diagonal=True)
    xr = solve_triangular(upper, np.diag(scale), lower=False)
    return xl, xr, a, b


def _circuit(matrix):
    """Exact diagonal/shear circuit with swap cancellation and adjacent gate fusion."""
    work = matrix.copy()
    n = len(work)
    inverse = []
    swap = np.array([[0, 1], [1, 0]], complex)
    for col in range(n):
        row = col + int(np.argmax(np.abs(work[col:, col])))
        if row != col:
            work[[col, row]] = work[[row, col]]
            inverse.append((col, row, swap))
        if work[col, col] == 0:
            raise np.linalg.LinAlgError('Orbital counter-transform is singular.')
        for row in range(n):
            if row == col or work[row, col] == 0:
                continue
            factor = work[row, col] / work[col, col]
            work[row] -= factor * work[col]
            work[row, col] = 0
            if row < col:
                inverse.append((row, col, np.array([[1, factor], [0, 1]], complex)))
            else:
                inverse.append((col, row, np.array([[1, 0], [factor, 1]], complex)))
    circuit = [('diagonal', np.diag(work).copy())]
    def append_swap(bond):
        if circuit[-1][0] == 'gate' and circuit[-1][1] == bond and circuit[-1][2] is swap:
            circuit.pop()
        else:
            circuit.append(('gate', bond, swap))

    for first, last, gate in reversed(inverse):
        for bond in range(last-1, first, -1):
            append_swap(bond)
        if gate is swap:
            append_swap(first)
        else:
            circuit.append(('gate', first, gate))
        for bond in range(first+1, last):
            append_swap(bond)
    fused = [circuit[0]]
    for _, bond, gate in circuit[1:]:
        if fused[-1][0] == 'gate' and fused[-1][1] == bond:
            gate = gate @ fused.pop()[2]
        if not np.array_equal(gate, np.eye(2)):
            fused.append(('gate', bond, gate))
    return fused


def _state_gate(sites, i, gate, cutoff, max_bond, memory_limit, discarded_budget=None):
    action = _two_orbital_gate(gate)
    channels = {}
    for (left, p, middle), a in sites[i].items():
        for (other, q, right), b in sites[i+1].items():
            if middle != other:
                continue
            _size_guard(a.shape[0] * b.shape[2], memory_limit)
            block = a[:, 0, :] @ b[:, 0, :]
            for spin in _fuse(p[1], q[1]):
                incoming = _recouple(left[1], p[1], middle[1], q[1], right[1], spin)
                if not incoming:
                    continue
                for outp in range(3):
                    outq = p[0] + q[0] - outp
                    if not 0 <= outq <= 2:
                        continue
                    value = action.get((spin, (outp, outq), (p[0], q[0])), 0)
                    if not value:
                        continue
                    pp, qq = (outp, _SPIN[outp]), (outq, _SPIN[outq])
                    for j in _fuse(left[1], pp[1]):
                        if right[1] not in _fuse(j, qq[1]):
                            continue
                        mid = (left[0]+outp, j)
                        factor = incoming * value * _recouple(left[1], pp[1], j, qq[1], right[1], spin)
                        key = (left, pp, mid, qq, right)
                        if factor:
                            contribution = factor * block[:, None, None, :]
                            if key in channels:
                                channels[key] += contribution
                            else:
                                channels[key] = contribution
    sites[i], sites[i+1], discarded, _ = _split(channels, cutoff, max_bond, memory_limit, discarded_budget)
    return discarded


def _read_sites(state):
    from pyqed.mps.mps import MPS
    if isinstance(state, MPS):
        from .orbital_transform import is_fully_reduced_su2_mps
        if not is_fully_reduced_su2_mps(state):
            raise NotImplementedError('Orbital overlap requires fully reduced SU(2) spatial-orbital MPS tensors.')
        if state.labels != ("lv", "p", "rv"):
            state = state.to_order(["lv", "p", "rv"])
        return [{tuple((q.charge, q.irrep.two_j) for q in key): np.array(block, complex, copy=True)
                 for key, block in site.data.items()} for site in state.tensors]
    nn_fusion = getattr(state, "tie", None) == 'fusion' and all(b == a+1 for a, b in getattr(state, "graph", ()))
    if getattr(state, "graph", ()) and not nn_fusion:
        raise NotImplementedError('The chain reader accepts only untied states or NN fusion selectors; other supported ties use the conditional engine.')
    def sector(q):
        return (q.charge, q.irrep.two_j)
    sites = []
    for i, tensor in enumerate(state.tensors):
        local = {}
        for key, block in tensor.items():
            # An NN fusion endpoint is exactly this site's existing right
            # sector. Other conditional entries are unreachable, not bond states.
            if state.parent_sets[i] and state.tie_domains[i+1][key[3][0]] != key[2]:
                continue
            local[tuple(sector(q) for q in key[:3])] = np.array(block, dtype=complex, copy=True)
        sites.append(local)
    return sites


def _install(state, sites):
    from pyqed.symmetry import IrrepTensor
    from pyqed.mps.nonabelian.states import spatial_target_sector

    reachable = {(0, 0)}
    for i, site in enumerate(sites):
        sites[i] = {key: block for key, block in site.items() if key[0] in reachable}
        reachable = {key[2] for key in sites[i]}
    reachable = {(_target(state).charge, _target(state).irrep.two_j)}
    for i in range(len(sites)-1, -1, -1):
        sites[i] = {key: block for key, block in sites[i].items() if key[2] in reachable}
        reachable = {key[0] for key in sites[i]}
    bases = []
    for site in sites:
        data = {tuple(spatial_target_sector(*q) for q in key): block for key, block in site.items()}
        qns = [sorted({key[axis] for key in data}, key=lambda q: (q.charge, q.irrep.two_j)) for axis in range(3)]
        bases.append(IrrepTensor(data=data, qns=qns, dirs=[-1, 1, 1], metadata={'physical_basis': 'fully_reduced_su2'}))
    from pyqed.mps.mps import MPS
    if isinstance(state, MPS):
        transformed = MPS.from_tensors(bases, target_sector=_target(state))
        if hasattr(state, 'root_ids'):
            transformed.root_ids = state.root_ids
        return transformed
    from pyqed.letta.su2_qchem import SU2LETTA
    transformed = SU2LETTA(None, base_sites=bases, target_sector=_target(state), graph=getattr(state, "graph", ()), tie=getattr(state, "tie", None),
                           D=max((_bond_dim(site) for site in sites[:-1]), default=1))
    if hasattr(state, 'root_ids'):
        transformed.root_ids = state.root_ids
    return transformed


def _norm(sites, target):
    env = {(0, 0): np.ones((1, 1), complex)}
    for site in sites:
        out = {}
        for (left, physical, right), block in site.items():
            if left in env:
                value = block[:, 0, :].conj().T @ env[left] @ block[:, 0, :]
                out[right] = out.get(right, 0) + value
        env = out
    gram = env.get((target.charge, target.irrep.two_j))
    return np.sqrt(max(0.0, float(np.trace(gram).real)))


def transform_orbitals(state, matrix, *, max_bond=128, cutoff=1e-12,
                       memory_limit=2**24, return_info=False, _state_tol=None):
    """Apply Γ(matrix) directly to a reduced chain or NN/NNN LETTA root batch.

    This is a coefficient counter-transformation: the corresponding orbital
    basis change is matrix^-1. Returned states are not renormalized. Whole
    reduced multiplets are truncated with a shared root-batch Frobenius
    metric; max_bond counts total multiplets per bond, excluding the explicit
    look-ahead labels. NN/NNN physical ties use conditional three/four-site
    patches. NNN fusion ties keep the future selector in a three-site patch;
    NN fusion selectors equal existing right sectors. Ties beyond range two
    are unsupported. See this module's Malmqvist/Knecht adaptation references.
    The accumulated state error bound excludes floating-point roundoff.
    """
    _check_controls(max_bond, cutoff, memory_limit)
    _check_graph(state)
    matrix = _matrix(matrix, len(state.tensors))
    if getattr(state, 'graph', ()) and (state.tie == 'physical' or any(b-a > 1 for a, b in state.graph)):
        from pyqed.letta.conditional_orbitals import transform_orbitals as transform_conditional
        return transform_conditional(state, matrix, max_bond=max_bond, cutoff=cutoff,
                            memory_limit=memory_limit, return_info=return_info, _state_tol=_state_tol)
    sites = _read_sites(state)
    for site in sites:
        _size_guard(sum(block.size for block in site.values()), memory_limit)
    error = 0.0
    center = None
    discarded_norms = []
    circuit = _circuit(matrix)
    if max_bond is not None:
        circuit += [("compress", i) for i in range(len(sites)-1)]
    budgets = _split_budgets(circuit, _state_tol, _target(state).irrep.dim)
    for (kind, *payload), budget in zip(circuit, budgets):
        if kind == 'diagonal':
            for i, scale in enumerate(payload[0]):
                sites[i] = {key: block * scale**key[1][0] for key, block in sites[i].items()}
                error *= max(1.0, abs(scale))**2
            center = None
            continue
        if kind == "compress":
            i = payload[0]
            if _bond_dim(sites[i]) <= max_bond:
                continue
            gate = np.eye(2)
        else:
            i, gate = payload
        if center is None:
            for j in range(len(sites)-1, 0, -1):
                _right_gauge(sites, j, memory_limit)
            center = 0
        while center < i:
            _left_gauge(sites, center, memory_limit)
            center += 1
        while center > i+1:
            _right_gauge(sites, center, memory_limit)
            center -= 1
        discarded = _state_gate(sites, i, gate, cutoff, max_bond, memory_limit, budget)
        center = i+1
        for site in sites[i:i+2]:
            _size_guard(sum(block.size for block in site.values()), memory_limit)
        loss = np.sqrt(discarded / _target(state).irrep.dim)
        amplification = np.prod(np.maximum(1.0, np.linalg.svd(gate, compute_uv=False)))**2
        error = amplification * error + loss
        discarded_norms.append(float(loss))
    transformed = _install(state, sites)
    info = dict(state_error_bound=float(error), transformed_batch_norm=_norm(sites, _target(state)),
                state_bond_dimensions=[_bond_dim(site) for site in sites[:-1]],
                adjacent_gate_count=len(discarded_norms), discarded_norms=discarded_norms,
                conditional_ties=bool(getattr(state, "graph", ())), tie_labels_folded=False,
                max_update_sites=2 if discarded_norms else 0)
    return (transformed, info) if return_info else transformed


def biorthogonalize(bra, ket, s, *, tol=1e-8, max_bond=None,
                    memory_limit=2**24, return_info=False):
    """Return two explicitly counter-transformed reduced LETTA states.

    ``s`` is the spatial-orbital cross metric; uppercase S denotes state overlap.
    XL† s XR=I, with new orbital coefficients C_L XL and C_R XR. State
    coefficients change by Γ(XL^-1) and Γ(XR^-1). Balanced pivoted LU is an
    adaptation of Malmqvist (1986)/Knecht et al. (2016), not their specific
    factorization or MPS implementation. Singular metrics and graph ties
    beyond range two are rejected. NN/NNN physical/fusion ties are supported. States
    remain unnormalized; input states are intact.

    tol (default 1e-8) bounds absolute overlap-matrix Frobenius truncation
    error, excluding roundoff and orbital-factorization error. Local weighted
    singular tails share a budget accounting for subsequent gate amplification
    and upper bounds on transformed norms. max_bond optionally caps reduced
    ranks; failure to certify tol raises OverlapConvergenceError. tol=None
    selects cap-only compression (or no truncation when max_bond=None).
    This conservative error allocation is a LETTA-specific adaptation.
    """
    if len(bra.tensors) != len(ket.tensors):
        raise ValueError('LETTA site counts must match.')
    _check_tol(tol)
    cutoff = 0.0
    _check_controls(max_bond, cutoff, memory_limit)
    _check_graph(bra, "bra")
    _check_graph(ket, "ket")
    metric = _matrix(s, len(bra.tensors))
    xl, xr, a, b = orbital_factors(metric)
    lt = rt = None
    if tol is not None:
        na, nb = _transformed_norm_bound(bra, a), _transformed_norm_bound(ket, b)
        lt = min(np.sqrt(tol)/4, tol/(4*max(nb, 1.0)))
        rt = min(np.sqrt(tol)/4, tol/(4*max(na, 1.0)))
    left, li = transform_orbitals(bra, a, _state_tol=lt, max_bond=max_bond, cutoff=cutoff, memory_limit=memory_limit, return_info=True)
    right, ri = transform_orbitals(ket, b, _state_tol=rt, max_bond=max_bond, cutoff=cutoff, memory_limit=memory_limit, return_info=True)
    el, er = li['state_error_bound'], ri['state_error_bound']
    bound = el*ri['transformed_batch_norm'] + er*li['transformed_batch_norm'] + el*er
    info = dict(backend='letta_biorthogonal', fully_reduced=True, mps_conversion=False,
                determinant_expansion=False, local_magnetic_expansion=False, tie_labels_folded=False,
                orbital_transforms={'bra': xl, 'ket': xr}, coefficient_transforms={'bra': a, 'ket': b},
                biorthogonality_residual=float(np.linalg.norm(xl.conj().T @ metric @ xr - np.eye(len(metric)))),
                bra=li, ket=ri, overlap_error_bound=float(bound),
                adjacent_gate_count=li['adjacent_gate_count'] + ri['adjacent_gate_count'],
                max_bond=max_bond, tol=tol, tolerance_met=None if tol is None else bool(bound <= tol),
                exact=bool(el == 0 and er == 0))
    if tol is not None and (not np.isfinite(bound) or bound > tol):
        raise OverlapConvergenceError(tol, info)
    return (left, right, info) if return_info else (left, right)


def _chain_overlap(bra, ket, memory_limit):
    """Identity contraction for untied reduced tensors, without operator/frontier axes."""
    left, right = _read_sites(bra), _read_sites(ket)
    nb = next(iter(left[-1].values())).shape[2]
    nk = next(iter(right[-1].values())).shape[2]
    if _target(bra) != _target(ket):
        return np.zeros((nb, nk), complex), 1
    env = {(0, 0): np.ones((1, 1), complex)}
    peak = 1
    for a, b in zip(left, right):
        out = {}
        for (ql, p, qr), block in a.items():
            other = b.get((ql, p, qr))
            if ql not in env or other is None:
                continue
            _size_guard(block.shape[2]*other.shape[2], memory_limit)
            value = block[:, 0, :].conj().T @ env[ql] @ other[:, 0, :]
            if qr in out:
                out[qr] += value
            else:
                out[qr] = value
        env = out
        size = sum(block.size for block in env.values())
        _size_guard(size, memory_limit)
        peak = max(peak, size)
    target = (_target(bra).charge, _target(bra).irrep.two_j)
    return env.get(target, np.zeros((nb, nk), complex)), peak


def overlap(bra, ket, *, s=None, tol=1e-8, max_bond=None,
                  memory_limit=2**24, return_info=False):
    """Biorthogonal counter-transformations followed by an identity-metric overlap."""
    _check_tol(tol)
    _check_controls(max_bond, 0.0, memory_limit)
    if s is None:
        left, right = bra, ket
        info = dict(backend='letta_reduced', fully_reduced=True, mps_conversion=False,
                    determinant_expansion=False, local_magnetic_expansion=False,
                    adjacent_gate_count=0, overlap_error_bound=0.0, exact=True)
    else:
        left, right, info = biorthogonalize(bra, ket, s, max_bond=max_bond,
                                            tol=tol, memory_limit=memory_limit, return_info=True)
    if not getattr(left, 'graph', ()) and not getattr(right, 'graph', ()):
        value, peak = _chain_overlap(left, right, memory_limit)
    elif (getattr(left, 'graph', ()) and getattr(right, 'graph', ())
          and left.tie == right.tie
          and max(b-a for a, b in left.graph) == max(b-a for a, b in right.graph)
          and max(b-a for a, b in left.graph) <= 2
          and (left.tie == 'physical' or max(b-a for a, b in left.graph) == 2)):
        from pyqed.letta.conditional_orbitals import overlap as conditional_overlap
        value, peak = conditional_overlap(left, right, memory_limit)
    else:
        identity = [{((2*i, 0), (2, 0), (2*i+2, 0)): np.array([1, np.sqrt(2), 1])[None, :, None]}
                    for i in range(len(bra.tensors))]
        value, peak = contract_overlap(left, right, identity, memory_limit=memory_limit)
    info.update(peak_frontier_elements=peak, memory_limit=memory_limit, tol=tol,
                tolerance_met=None if tol is None else True)
    if value.shape == (1, 1) and not hasattr(bra, 'root_ids') and not hasattr(ket, 'root_ids'):
        value = value.item()
    return (value, info) if return_info else value
