"""Factorized Coulomb contractions and analytic CD/RI nuclear response.

CD uses a fixed AO-product auxiliary basis (the selected pivots), following
the CD/DF equivalence of Aquilante, Lindh and Pedersen, JCP 129, 034106
(2008), https://doi.org/10.1063/1.2955755. RI differentiates the Coulomb fit
including its moving auxiliary basis and metric; see Dunlap, PCCP 2,
2113–2116 (2000), https://doi.org/10.1039/B000027M.

This is an adaptation: both directional factor response and contracted factor
adjoints are available. No four-index ERI is reconstructed on the factorized path.
CD columns use selected shell-quartet batching and sparse AO-pair projection,
followed by a block triangular-solve derivative. This is an exact
reorganization of the fixed-pivot recurrence, not a new approximation.
Contracted CD response reverses that recurrence and accumulates all nuclear
directions directly in the shell kernel. Its factor adjoints are formed in
rank blocks and stored as packed AO pairs, without expanding the whole factor
tensor. Coulomb-only adjoints use packed rank-two outer products and solve
their charge vectors before expansion; exchange retains the rank-block path.
This changes contraction order only. Cartesian-source densities are transformed before contraction.
RI reverses the metric/factor map and contracts shell-blocked three-center
derivatives directly into all atomic gradients. Two-center metric derivatives
are contracted independently; auxiliary-center motion is retained exactly.
Triplet HRR expansions are reused across primitive products and radial shells,
keyed by angular orders and exact AO-center displacement bits within each call.
Cholesky response reuses the molecule's screened factors and rebuilds only the
auxiliary metric. Spectral response retains full three-center tensors for
discarded-eigenspace derivatives.
Derivatives are local to fixed CD pivots/rank and RI metric rank. Changes of
these discrete choices are not differentiable. RI eigenvalue truncation
includes retained/discarded eigenspace response. Integral screening switches
are not differentiated; converge screening along with CD/metric thresholds.
"""

import numpy as np
from scipy.linalg import solve_triangular

from . import basis as gaussian
from .basis_derivatives import (
    _basis_and_transform, _basis_signature, _atom_ids_for_basis,
    _derivative_signatures_from_signature,
    directional_eri_derivatives,
)


def _cholesky_data(factors, nao, transform=None):
    """Recover pivot columns from ordered, unrotated Cholesky vectors.

    The largest absolute element of each residual Cholesky row is its pivot
    (Cauchy–Schwarz). Check the resulting triangular block before using it.
    This also handles builders that do not return their pivot indices.
    """
    pair = gaussian._as_ri_pair_factors(factors, nao)
    pivots = np.argmax(abs(pair), axis=1) if len(pair) else np.empty(0, dtype=int)
    triangle = pair[:, pivots]
    if (len(np.unique(pivots)) != len(pivots)
            or not np.allclose(np.tril(triangle, -1), 0., atol=1e-8, rtol=0)):
        raise ValueError('CD response requires ordered, unrotated Cholesky vectors.')
    return dict(factors=factors, nao=nao, pivots=pivots, transform=transform)


def _directional_signatures(signatures, owners, direction):
    result = []
    for sig, owner in zip(signatures, owners):
        terms = []
        for axis, scale in enumerate(direction[owner]):
            if scale == 0:
                continue
            order = tuple(int(i == axis) for i in range(3))
            for shell, origin, exps, weights in _derivative_signatures_from_signature(sig, order):
                terms.append((shell, origin, exps, tuple(scale*w for w in weights)))
        result.append(terms)
    return result


def _augment(signatures, derivatives):
    expanded = list(signatures)
    mapping = []
    for terms in derivatives:
        mapping.append(list(range(len(expanded), len(expanded)+len(terms))))
        expanded.extend(terms)
    return tuple(expanded), mapping


def _ri_tensors(primary, auxiliary, pair_support=None):
    tensors = gaussian._compute_native_ri_pair_tensors_cpp(
        primary, auxiliary, np.ones((len(primary), len(primary))), 0., pair_support=pair_support)
    if tensors is None:
        raise RuntimeError('RI response requires the compiled Gaussian RI tensor kernel.')
    metric, packed = tensors[:2]
    return metric, gaussian._pair_factors_to_full(packed, len(primary))


class CoulombIntegrals:
    """Contract the ERI representation actually used by a molecular calculation.

    ``derivative(direction)`` returns analytic directional Coulomb response.
    CD/RI use forward factor differentiation, adapting Aquilante et al.,
    JCP 129, 034106 (2008), doi:10.1063/1.2955755, and Coulomb-metric fitting
    (Dunlap, PCCP 2, 2113 (2000), doi:10.1039/B000027M). See module-level
    fidelity/threshold restrictions. Real AO bases only; no dense fallback
    for unsupported factor provenance.
    """

    def __init__(self, mol=None, *, tensor=None, terms=()):
        self.mol, self.tensor, self._terms = mol, tensor, terms
        self._ri = None
        self._cd_arrays = None
        self._factor_source = None if mol is None else getattr(mol, 'eri_factors', None)
        self._expanded_factors = None

    @property
    def factors(self):
        if self._factor_source is None:
            return None
        if self._expanded_factors is None:
            self._expanded_factors = np.asarray(self._factor_source)
        return self._expanded_factors

    @property
    def terms(self):
        if self._factor_source is not None:
            factors = self.factors
            return ((factors, factors),)
        return self._terms

    def _dense(self):
        if self.tensor is None:
            from .tddft import _dense_eri
            self.tensor = _dense_eri(self.mol)
        return self.tensor

    def j(self, density):
        if not self.terms:
            return np.einsum('pqrs,rs->pq', self._dense(), density, optimize=True)
        return sum(np.einsum('Ppq,P->pq', left,
                            np.einsum('Prs,rs->P', right, density, optimize=True),
                            optimize=True) for left, right in self.terms)

    def transform(self, left, right, left2, right2):
        if not self.terms:
            return np.einsum('pqrs,pi,qa,rj,sb->iajb', self._dense(),
                             left, right, left2, right2, optimize=True)
        return sum(np.einsum('Pia,Pjb->iajb',
                            gaussian.mo_pair_factors(a, left, right),
                            gaussian.mo_pair_factors(b, left2, right2), optimize=True)
                   for a, b in self.terms)

    def derivative(self, direction):
        direction = np.asarray(direction, dtype=float).reshape(self.mol.natom, 3)
        if not self.terms:
            return CoulombIntegrals(tensor=directional_eri_derivatives(
                self.mol, direction[None], backend='native')[0])
        info = getattr(self.mol, '_builtin_build_info', {})
        if info.get('ri') is not None:
            factors, derivative = self._ri_derivative(direction)
        else:
            factors, derivative = self._cd_derivative(direction)
        return CoulombIntegrals(terms=((derivative, factors), (factors, derivative)))

    def contract_derivatives(self, observables, *, exchange_fraction=0.,
                             workers=None, screen_tol=None, kernel=None):
        """Contract ``scale * <left,(J-exchange_fraction*K)(right)>`` derivatives.

        Densities are real and symmetrized for exchange contractions. Restricted
        global hybrids use scale=1/2 and exchange_fraction=hybrid_coeff/2.
        Exchange differentiates the same exact/CD/RI ERIs as Coulomb, using
        a factor adjoint rather than constructing four-index derivative tensors.

        CD reverses the fixed-pivot factor recurrence (Aquilante et al.,
        JCP 129, 034106, doi:10.1063/1.2955755). Coordinate columns are streamed;
        discrete pivot/rank and screening switches are not differentiated.
        RI reverses its metric solve, including spectral subspace response.
        RI three-center derivatives are contracted in complete shell blocks;
        moving auxiliary-center and metric terms are included (Dunlap, PCCP 2,
        2113, doi:10.1039/B000027M). This is an exact reorganization at fixed
        screening/rank, not a new fit. The contracted kernel is serial and
        supports primary shell pairs with total angular momentum <= 6 and
        auxiliary shells through l=6; unsupported shells raise explicitly.
        Directional factor derivatives retain the independent reference path.
        HRR geometry recipes are reused across shell triplets with matching
        angular orders and exact AO-center displacements, using a per-call
        16 MiB vector/key payload budget (map overhead is additional). Cache
        eviction recomputes recipes; no density, exponent, or final derivative
        is cached. Cholesky RI response reuses existing
        factors rather than rebuilding three-center tensors; truncated spectral
        fits retain those tensors to preserve discarded-space response.
        The ERI recurrence shares the derivative angular-order budget between
        both shell pairs; removing unreachable higher orders is exact and
        introduces no additional screening or approximation. For contracted
        first derivatives with equal component radial weights, a quartet-local
        recurrence plan evaluates only dependencies of the contracted HRR terms;
        its indices are reused across primitive quartets. Contractions with at
        least 32 primitive quartets traverse that plan in batches of 32 along
        a contiguous primitive axis, using worker-local scratch; final partial
        batches are included. Workers reuse index-only plans keyed by angular
        bounds and requested VRR nodes, within this derivative call (at most
        256 plans and an 8 MiB payload budget per worker). Geometry and density
        values are never cached by those index plans; ``derivative_info``
        reports their hits/misses. A separate worker-local HRR cache reuses
        geometric expansion coefficients across radial shells, keyed by exact
        center displacements and angular/derivative structure (16 MiB payload
        budget). It is discarded after the derivative call; exponents,
        densities and final derivative values are not cached. Serial tasks
        group equal centers to improve reuse; parallel cost ordering is retained.
        Proportional component weights (including Cartesian d-shell
        normalizations) also use this contraction for batches of at least 32
        primitive quartets. Ratios are checked once per shell within 16 machine
        epsilons; nonproportional weights retain the general path. Workers also
        reuse HRR polynomials (4096 entries/8 MiB payload) and complete first-
        derivative recipe sets (256 entries/16 MiB payload). Their keys include
        angular dimensions, complete shell classes, active derivative slots and
        exact-zero displacement patterns. Geometry-dependent coefficients are
        recomputed for every quartet; no small nonzero displacement is dropped.
        This only reorders arithmetic. Other cases retain
        the full recurrence, with identical integral and screening conventions.
        For CD, ``workers`` controls quartet threads. ``screen_tol`` is an
        opt-in absolute error budget per contracted gradient component, using
        derivative-pair Coulomb norms and primitive triangle bounds. It does
        not bound CD truncation or floating-point error. ``derivative_info``
        reports skipped quartets and the allocated bound. ``kernel='rys'``
        uses raised/lowered s/p Rys moments (higher shells retain recurrence).
        This adapts Dupuis, Rys and King, JCP 65, 111 (1976),
        doi:10.1063/1.432807, using analytic Gaussian center derivatives,
        not finite differences; it is not a reproduction of HONDO.
        ``auto`` currently selects the measured faster recurrence path.
        """
        exchange_fraction = float(exchange_fraction)
        if not np.isfinite(exchange_fraction):
            raise ValueError('exchange_fraction must be finite.')
        screen_tol = self.mol.derivative_screen_tol if screen_tol is None else screen_tol
        kernel = self.mol.derivative_kernel if kernel is None else kernel
        output = np.zeros((len(observables), self.mol.natom, 3))
        has_factors = self._factor_source is not None or bool(self._terms)
        is_ri = has_factors and getattr(self.mol, '_builtin_build_info', {}).get('ri') is not None
        is_cd = has_factors and not is_ri
        if is_cd:
            return self._contract_cd_derivatives(observables, exchange_fraction,
                workers=workers, screen_tol=screen_tol, kernel=kernel)
        if not is_cd and (workers is not None or screen_tol != 0. or kernel != 'auto'):
            raise NotImplementedError('These derivative controls currently apply only to CD.')
        if not self.terms:
            from .basis_derivatives import _directional_eri_derivative_scalar_cpp
            directions = np.eye(self.mol.natom*3).reshape(-1, self.mol.natom, 3)
            lefts, rights, indices = [], [], []
            for index, terms in enumerate(observables):
                for scale, left, right in terms:
                    if scale:
                        lefts.append(scale*(left+left.T)*.5)
                        rights.append((right+right.T)*.5)
                        indices.append(index)
            if indices:
                values = _directional_eri_derivative_scalar_cpp(self.mol, directions,
                    np.asarray(lefts), np.asarray(rights), order=1, exchange_fraction=exchange_fraction)
                for term, index in enumerate(indices):
                    output[index] += values[:, term].reshape(self.mol.natom, 3)
            return output
        if is_ri:
            if self._ri is None:
                self._prepare_ri()
            factors = self._ri[6]
        if self.terms:
            sensitivity = np.zeros((len(observables),)+factors.shape)
            for index, terms in enumerate(observables):
                for scale, left, right in terms:
                    sensitivity[index] += scale*(
                        np.einsum('Ppq,pq->P', factors, right)[:, None, None]*left
                        + np.einsum('Ppq,pq->P', factors, left)[:, None, None]*right)
                    if exchange_fraction and scale:
                        left_sym = .5*(left+left.T)
                        right_sym = .5*(right+right.T)
                        exchange = left_sym @ factors @ right_sym
                        sensitivity[index] -= scale*exchange_fraction*(exchange+exchange.transpose(0, 2, 1))
        if is_ri:
            three, root, factors, spectral = self._ri[4:8]
            flat = factors.reshape(len(factors), -1)
            three_weights, metric_weights = [], []
            for bar in sensitivity.reshape(len(observables), len(factors), -1):
                if spectral is None:
                    three_weights.append(solve_triangular(root.T, bar, lower=False))
                    connection = -bar@flat.T
                    connection = np.tril(connection, -1)+.5*np.diag(np.diag(connection))
                    metric = solve_triangular(root.T, connection, lower=False)
                    metric_weights.append(solve_triangular(root.T, metric.T, lower=False).T)
                else:
                    _, vectors, _, _ = spectral
                    rootbar = bar@three.reshape(len(root), -1).T
                    metric_weights.append(vectors@(_metric_divided_difference(spectral)
                        *(vectors.T@rootbar@vectors))@vectors.T)
                    three_weights.append(root.T@bar)
            three_weights = np.asarray(three_weights).reshape(sensitivity.shape)
            metric_weights = np.asarray(metric_weights)
            return self._contract_ri_integral_derivatives(three_weights, metric_weights)
        return output

    def _contract_ri_integral_derivatives(self, three_weights, metric_weights):
        primary, auxiliary, transform, aux_transform = self._ri[:4]
        if transform is not None:
            three_weights = np.einsum('AP,oPij,mi,nj->oAmn', aux_transform,
                                     three_weights, transform, transform, optimize=True)
            metric_weights = np.einsum('AP,oPQ,BQ->oAB', aux_transform,
                                      metric_weights, aux_transform, optimize=True)
        three_weights = three_weights*self._ri[8]
        p, q = np.tril_indices(len(primary))
        packed = three_weights[:, :, p, q] + three_weights[:, :, q, p]
        packed[:, :, p == q] *= .5
        coords = self.mol.atom_coords()
        def owners(signatures):
            delta = np.linalg.norm(np.asarray([s[1] for s in signatures])[:, None]
                                   - coords[None], axis=2)
            if np.any(delta.min(axis=1) > 1e-10):
                raise ValueError('RI basis center does not belong to an atom.')
            return np.asarray(delta.argmin(axis=1), dtype=np.int64)
        return gaussian._integrals_cpp.contract_ri_derivatives(
            tuple(gaussian._pack_signatures_for_numba(primary)),
            tuple(gaussian._pack_signatures_for_numba(auxiliary)),
            owners(primary), owners(auxiliary), packed, metric_weights, self.mol.natom)

    def _contract_cd_derivatives(self, observables, exchange_fraction, **controls):
        """Build packed factor adjoints in rank blocks, in the source AO basis."""
        data = self.mol._cd_factor_data
        pair = gaussian._as_ri_pair_factors(data['factors'], data['nao'])
        p, q = np.tril_indices(data['nao'])
        diagonal = p == q
        _, transform = _basis_and_transform(self.mol)
        transform = transform if data['transform'] is None else None
        root = pair[:, data['pivots']].T
        weights = np.empty((len(observables), *pair.shape))
        for index, terms in enumerate(observables):
            densities = []
            for scale, left, right in terms:
                if transform is not None:
                    left, right = transform@left@transform.T, transform@right@transform.T
                densities.append((scale, .5*(left+left.T), .5*(right+right.T)))
            bar = weights[index]
            if not exchange_fraction:
                # Coulomb factor adjoints are sums of rank-two outer products.
                # Apply the triangular solve before expanding into pair space.
                bar.fill(0.)
                connection = np.zeros((len(pair), len(pair)))
                for scale, left, right in densities:
                    left_pair = left[p, q]+left[q, p]
                    right_pair = right[p, q]+right[q, p]
                    left_pair[diagonal] *= .5
                    right_pair[diagonal] *= .5
                    charges = pair @ np.column_stack((left_pair, right_pair))
                    solved = solve_triangular(root.T, charges, lower=False)
                    bar += scale*(solved[:, 1, None]*left_pair
                                  + solved[:, 0, None]*right_pair)
                    connection -= scale*(charges[:, 1, None]*charges[:, 0]
                                         + charges[:, 0, None]*charges[:, 1])
            for start in range(0, len(pair) if exchange_fraction else 0, 32):
                stop = min(start+32, len(pair))
                factors = gaussian._pair_factors_to_full(pair[start:stop], data['nao'])
                sensitivity = np.zeros_like(factors)
                for scale, left, right in densities:
                    sensitivity += scale*(
                        np.einsum('Ppq,pq->P', factors, right)[:, None, None]*left
                        +np.einsum('Ppq,pq->P', factors, left)[:, None, None]*right)
                    if exchange_fraction and scale:
                        exchange = left@factors@right
                        sensitivity -= scale*exchange_fraction*(exchange+exchange.transpose(0, 2, 1))
                bar[start:stop] = sensitivity[:, p, q]+sensitivity[:, q, p]
                bar[start:stop, diagonal] *= .5
            if exchange_fraction:
                connection = -bar@pair.T
                weights[index] = solve_triangular(root.T, bar, lower=False)
            connection = np.tril(connection, -1)+.5*np.diag(np.diag(connection))
            metric = solve_triangular(root.T, connection, lower=False)
            metric = solve_triangular(root.T, metric.T, lower=False).T
            weights[index][:, data['pivots']] += metric
        directions = np.eye(self.mol.natom*3).reshape(-1, self.mol.natom, 3)
        return self._cd_columns(directions, sensitivities=weights, **controls).T.reshape(
            len(observables), self.mol.natom, 3)

    def _cd_columns(self, directions, workers=None, sensitivities=None, *,
                    early_contraction=True, screen_tol=0., kernel='auto'):
        if kernel not in ('auto', 'recurrence', 'rys'):
            raise ValueError('Derivative kernel must be auto, recurrence, or rys.')
        if not np.isfinite(screen_tol) or screen_tol < 0:
            raise ValueError('screen_tol must be finite and nonnegative.')
        data = getattr(self.mol, '_cd_factor_data', None)
        if data is None:
            raise NotImplementedError('CD derivatives require build-time factor provenance.')
        if not hasattr(gaussian._integrals_cpp, 'compute_eri_derivative_columns'):
            raise RuntimeError('Rebuild the Gaussian integral extension for CD derivative columns.')
        if self._cd_arrays is None:
            basis, _ = _basis_and_transform(self.mol)
            packed = gaussian._pack_signatures_for_numba(tuple(_basis_signature(fn) for fn in basis))
            arrays = tuple(np.ascontiguousarray(a, dtype=dtype) for a, dtype in
                           zip(packed, (np.int64, float, float, float, np.int64)))
            owners = _atom_ids_for_basis(basis, self.mol.atom_coords())
            source = data['transform']
            source = np.eye(len(basis)) if source is None else source
            self._cd_arrays = arrays + (np.ascontiguousarray(owners, dtype=np.int64),
                                        np.ascontiguousarray(source, dtype=float),
                                        np.ascontiguousarray(data['pivots'], dtype=np.int64))
        shells, origins, exps, weights, nprim, owners, source, pivots = self._cd_arrays
        if workers is None:
            workers = gaussian._builtin_worker_count(self.mol, len(shells))
        directions = np.ascontiguousarray(directions, dtype=float).reshape(-1, self.mol.natom, 3)
        result, self.derivative_info = gaussian._integrals_cpp.compute_eri_derivative_columns(
            shells, origins, exps, weights, nprim, owners, directions, source, pivots, workers, sensitivities,
            early_contraction, int(kernel == 'rys'), screen_tol, True)
        return result

    def _cd_derivative(self, direction):
        columns = self._cd_columns(direction[None])[0]
        data = self.mol._cd_factor_data
        _, ao_transform = _basis_and_transform(self.mol)
        source_transform = data['transform']
        pair = gaussian._as_ri_pair_factors(data['factors'], data['nao'])
        # The pivot submatrix is a Cholesky root. Differentiate its triangular
        # solve in blocks rather than updating every rank in a Python loop.
        root = pair[:, data['pivots']].T
        connection = solve_triangular(root, columns[:, data['pivots']], lower=True)
        connection = solve_triangular(root, connection.T, lower=True).T
        connection = np.tril(connection, -1) + .5*np.diag(np.diag(connection))
        dpair = solve_triangular(root, columns, lower=True) - connection @ pair
        factors = gaussian._pair_factors_to_full(pair, data['nao'])
        derivative = gaussian._pair_factors_to_full(dpair, data['nao'])
        if source_transform is None and ao_transform is not None:
            factors = gaussian.mo_pair_factors(factors, ao_transform, ao_transform)
            derivative = gaussian.mo_pair_factors(derivative, ao_transform, ao_transform)
        return factors, derivative

    def _prepare_ri(self):
        mol = self.mol
        info = mol._builtin_build_info['ri']
        basis, transform = _basis_and_transform(mol)
        primary = tuple(_basis_signature(fn) for fn in basis)
        auxiliary = gaussian._make_contraction_signatures(
            gaussian.load_basis_dict(info['auxbasis'], mol.atom_symbols()),
            mol.atom_symbols(), mol.atom_coords(), coord_types='c')
        aux_transform = None
        if transform is not None:
            aux_transform, _ = gaussian._global_cartesian_to_spherical_transform(auxiliary)
        solver = info['metric_solver']
        # Cholesky factors already contain the screened three-center integrals.
        # The spectral branch needs their discarded-space components as well.
        metric, three = _ri_tensors(() if solver == 'cholesky' else primary, auxiliary)
        retained = True
        if info['screen_tol'] > 0:
            bounds = gaussian._compute_pair_bounds(primary)
            retained = (np.sqrt(abs(np.diag(metric)))[:, None, None]
                        * bounds[None] >= info['screen_tol'])
        if solver == 'cholesky':
            three = None
        else:
            three *= retained
        if transform is not None:
            metric = aux_transform.T @ metric @ aux_transform
            if three is not None:
                three = np.einsum('AP,Amn,mi,nj->Pij', aux_transform, three,
                                  transform, transform, optimize=True)
        if solver == 'cholesky':
            root = np.linalg.cholesky(metric)
            factors = self.factors
            spectral = None
        else:
            vals, vectors = np.linalg.eigh(metric)
            tol = float(getattr(mol, 'builtin_ri_metric_tol', 1e-10))
            keep = vals > tol
            if np.any(abs(vals-tol) < 1e-8 * max(tol, np.max(abs(vals)))):
                raise ValueError('RI metric eigenvalue is too close to the rank threshold.')
            f = np.zeros_like(vals)
            f[keep] = 1 / np.sqrt(vals[keep])
            root = (vectors * f) @ vectors.T
            factors = root @ three.reshape(len(root), -1)
            spectral = (vals, vectors, f, keep)
        self._ri = (primary, auxiliary, transform, aux_transform, three,
                    root, factors.reshape(-1, mol.nao, mol.nao), spectral, retained)

    def _ri_primitive_derivative(self, direction):
        if self._ri is None:
            self._prepare_ri()
        primary, auxiliary, transform, aux_transform, three, root, factors, spectral, retained = self._ri
        coords = np.asarray(self.mol.atom_coords())
        def owners(signatures):
            return np.argmin(np.linalg.norm(
                np.array([sig[1] for sig in signatures])[:, None] - coords[None], axis=2), axis=1)
        p_exp, p_map = _augment(primary, _directional_signatures(primary, owners(primary), direction))
        a_exp, a_map = _augment(auxiliary, _directional_signatures(auxiliary, owners(auxiliary), direction))
        metric, j3 = _ri_tensors(p_exp, a_exp)
        np_, na = len(primary), len(auxiliary)
        dmetric = np.array([metric[indices, :na].sum(axis=0) for indices in a_map])
        dmetric += dmetric.T.copy()
        dj3 = np.array([j3[indices, :np_, :np_].sum(axis=0) for indices in a_map])
        primary_term = np.stack([j3[:na, indices, :np_].sum(axis=1) for indices in p_map], axis=1)
        dj3 += primary_term + primary_term.transpose(0, 2, 1)
        dj3 *= retained
        if transform is not None:
            dmetric = aux_transform.T @ dmetric @ aux_transform
            dj3 = np.einsum('AP,Amn,mi,nj->Pij', aux_transform, dj3,
                            transform, transform, optimize=True)
        return dj3, dmetric

    def _ri_derivative(self, direction):
        dj3, dmetric = self._ri_primitive_derivative(direction)
        three, root, factors, spectral = self._ri[4:8]
        flat = factors.reshape(len(factors), -1)
        if spectral is None:
            connection = solve_triangular(root, dmetric, lower=True)
            connection = solve_triangular(root, connection.T, lower=True).T
            connection = np.tril(connection, -1) + .5*np.diag(np.diag(connection))
            derivative = solve_triangular(root, dj3.reshape(len(root), -1), lower=True) - connection @ flat
        else:
            vals, vectors, f, keep = spectral
            divided = _metric_divided_difference(spectral)
            droot = vectors @ (divided * (vectors.T @ dmetric @ vectors)) @ vectors.T
            derivative = droot @ three.reshape(len(root), -1) + root @ dj3.reshape(len(root), -1)
        return factors, derivative.reshape(factors.shape)


def _metric_divided_difference(spectral):
    vals, vectors, f, keep = spectral
    delta = vals[:, None]-vals[None, :]
    divided = np.zeros_like(delta)
    both = keep[:, None] & keep[None, :]
    roots = np.sqrt(np.maximum(vals, 0))
    denom = roots[:, None]*roots[None, :]*(roots[:, None]+roots[None, :])
    np.divide(-1., denom, out=divided, where=both)
    mixed = keep[:, None] != keep[None, :]
    np.divide(f[:, None]-f[None, :], delta, out=divided, where=mixed)
    return divided
