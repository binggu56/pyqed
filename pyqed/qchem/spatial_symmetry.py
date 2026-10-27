"""Finite-group spatial actions and symmetry-restricted CI.

Uses exact matrix-element projection, Cotton, Chemical Applications of Group
Theory, 3rd ed., Wiley (1990), ISBN 9780471510949. CI actions reuse PyQED's
Slater string transformations; no scalar irrep is assigned to an E determinant.
Tensor-network algorithms do not use this determinant-space implementation.
"""
from __future__ import annotations

from collections import OrderedDict
from time import perf_counter

import numpy as np

from .symmetry import get_point_group, irrep_representation_matrices, mo_symmetry_representations


def resolve_spatial_target(group, wfnsym=None):
    if wfnsym is None:
        return None
    if getattr(group,'continuous',False):
        from pyqed.symmetry.continuous import OrbitalRotationSymmetry
        if not isinstance(wfnsym,str):
            raise ValueError('Continuous orbital targets require a whole-irrep label.')
        return OrbitalRotationSymmetry(group.name).dual(wfnsym)
    index = group.irrep_id(wfnsym) if isinstance(wfnsym, str) else int(wfnsym)
    if index < 0 or index >= len(group.irrep_names):
        raise ValueError('Invalid spatial irrep id.')
    return group.irrep_names[index]


class SpatialGroupAction:
    """Closed one-particle representation and complete target multiplet.

    Operation matrices are retained, including partner mixing and repeated
    copies. ``weights[row,column]`` are Fourier boundary intertwiners, rather
    than scalar orbital quantum numbers. Only finite real point groups are
    currently supported. Matrices must already be in the calculation's MO order.
    """

    def __init__(self, orbital_representations, wfnsym, *, point_group='D3h', tol=1e-8):
        self.group = get_point_group(point_group) if isinstance(point_group, str) else point_group
        if self.group.continuous:
            raise ValueError('Use angular generators and the reduced multiplet solver for continuous groups.')
        self.wfnsym = resolve_spatial_target(self.group, wfnsym)
        if self.wfnsym is None:
            raise ValueError('A target spatial irrep is required.')
        self.names = self.group.operation_names
        self.matrices = tuple(np.asarray(orbital_representations[name], dtype=complex) for name in self.names)
        self.norb = self.matrices[0].shape[0]
        eye = np.eye(self.norb)
        if any(matrix.shape != eye.shape or not np.all(np.isfinite(matrix)) or
               not np.allclose(matrix.conj().T @ matrix, eye, atol=tol, rtol=0)
               for matrix in self.matrices):
            raise ValueError('Orbital group matrices must be finite, square and unitary.')
        physical = np.asarray(self.group.operation_matrices)
        products = physical[:, None] @ physical[None, :]
        self.table = np.argmin(np.linalg.norm(
            products[:, :, None] - physical[None, None], axis=(-2, -1)), axis=-1)
        matrices = np.asarray(self.matrices)
        # Batch one multiplication-table row at a time, bounding temporary
        # storage by |G|*norb² while retaining every closure check.
        for i, matrix in enumerate(matrices):
            if not np.allclose(matrix @ matrices, matrices[self.table[i]], atol=tol, rtol=0):
                raise ValueError('Orbital matrices do not form a closed group representation.')
        identity = int(np.argmin(np.linalg.norm(physical - np.eye(3), axis=(1, 2))))
        generated, generators = {identity}, []
        for index in range(len(self.names)):
            if index in generated:
                continue
            generators.append(index)
            while True:
                expanded = generated | {int(self.table[g, h])
                                        for g in generators for h in generated}
                if expanded == generated:
                    break
                generated = expanded
        self.generators = tuple(generators)
        self.irrep_matrices = irrep_representation_matrices(self.group, self.wfnsym)
        self.dimension = self.group.irrep_dimensions[self.wfnsym]
        self.weights = self.dimension / len(self.names) * np.stack(
            [self.irrep_matrices[name].conj() for name in self.names], axis=-1)
        cuts = [0] + [cut for cut in range(1, self.norb)
                      if all(np.max(np.abs(matrix[:cut, cut:]), initial=0) < 1e-13 and
                             np.max(np.abs(matrix[cut:, :cut]), initial=0) < 1e-13
                             for matrix in self.matrices)] + [self.norb]
        self.orbital_blocks = tuple(tuple(range(a, b)) for a, b in zip(cuts[:-1], cuts[1:]))
        self.tol = tol

    def validate_hamiltonian(self, h1e, eri=None):
        """Reject a Hamiltonian that breaks the requested group, including ERIs."""
        h1e = np.asarray(h1e)
        if np.max(np.abs(h1e.imag), initial=0) > 1e-12 or (
                eri is not None and np.max(np.abs(np.asarray(eri).imag), initial=0) > 1e-12):
            raise ValueError('Spatial targeting currently requires a real restricted Hamiltonian.')
        if h1e.shape != (self.norb, self.norb):
            raise ValueError('Hamiltonian and group representations use different orbital spaces.')
        # Invariance under generators implies invariance under every product.
        for index in self.generators:
            matrix = self.matrices[index]
            if not np.allclose(matrix.conj().T @ h1e @ matrix, h1e, atol=self.tol, rtol=self.tol):
                raise ValueError('One-electron Hamiltonian is not invariant under the spatial group.')
            if eri is not None:
                transformed = np.einsum('pa,qb,rc,sd,pqrs->abcd', matrix.conj(), matrix,
                                        matrix.conj(), matrix, np.asarray(eri), optimize=True)
                if not np.allclose(transformed, eri, atol=self.tol, rtol=self.tol):
                    raise ValueError('Two-electron Hamiltonian is not invariant under the spatial group.')


_STRING_TRANSFORM_CACHE = OrderedDict()
_MAX_STRING_CACHE_BYTES = 64 * 1024**2
_MAX_STRING_ACTION_BYTES = 256 * 1024**2


def _cached_string_transform(norb, matrix_bytes, occupation_shape, occupation_bytes):
    """Cache validated dense and sparse forms of the same string transform."""
    key = (norb, matrix_bytes, occupation_shape, occupation_bytes)
    if key in _STRING_TRANSFORM_CACHE:
        _STRING_TRANSFORM_CACHE.move_to_end(key)
        return _STRING_TRANSFORM_CACHE[key][0]
    from .mcscf.casci import _string_transform_matrix
    matrix = np.frombuffer(matrix_bytes, dtype=complex).reshape(norb, norb)
    occupations = np.frombuffer(occupation_bytes, dtype=np.int8).reshape(occupation_shape)
    if not np.any(matrix.imag):
        matrix = matrix.real
    with np.errstate(divide='ignore', invalid='ignore'):
        transform = _string_transform_matrix(matrix, occupations, matrix.dtype)
    from scipy import sparse
    mask = abs(transform) > 1e-13
    compact = sparse.csr_matrix(np.real_if_close(np.where(mask, transform, 0)))
    if np.count_nonzero(mask) < transform.size // 4:
        error = compact.conj().T @ compact - sparse.eye(len(transform))
        dropped = np.linalg.norm(transform[~mask])
        # Bound the omitted Gram terms instead of weakening the unitary check.
        norm_bound = np.sqrt(np.max(np.asarray(abs(compact).sum(axis=0)), initial=0)
                             * np.max(np.asarray(abs(compact).sum(axis=1)), initial=0))
        unitary_error = np.max(abs(error.data), initial=0) + 2*norm_bound*dropped + dropped*dropped
    else:
        unitary_error = np.max(abs(transform.conj().T @ transform - np.eye(len(transform))), initial=0)
    if unitary_error > 1e-8:
        raise ValueError('CI strings omit part of a spatial multiplet.')
    transform.setflags(write=False)
    result = transform, compact
    size = (transform.nbytes + compact.data.nbytes + compact.indices.nbytes
            + compact.indptr.nbytes + len(matrix_bytes) + len(occupation_bytes))
    if size <= _MAX_STRING_CACHE_BYTES:
        while _STRING_TRANSFORM_CACHE and (len(_STRING_TRANSFORM_CACHE) >= 32 or
                size + sum(v[1] for v in _STRING_TRANSFORM_CACHE.values()) > _MAX_STRING_CACHE_BYTES):
            _STRING_TRANSFORM_CACHE.popitem(last=False)
        _STRING_TRANSFORM_CACHE[key] = result, size
    return result


_cached_string_transform.cache_clear = _STRING_TRANSFORM_CACHE.clear


class CISpatialGroupAction(SpatialGroupAction):
    """Matrix-free finite-group action on rectangular alpha/beta CI strings.

    Stores two string transformation matrices per operation, never an ndet²
    projector. Matrix units preserve both E partners. This is an ordinary CI
    implementation and is not used by SU2-LETTA, NARG, or DMRG.
    """

    def __init__(self, model, wfnsym):
        from .mcscf.casci import _basis_spin_strings, _cached_spin_string_basis
        coeff = np.asarray(model.mo_coeff)
        mol = model.mol
        ncore, ncas = model.ncore, model.ncas
        # Build current-orbital metadata once for both the solver and reporting.
        # Rebuild on each invocation, since MCSCF may have rotated the orbitals.
        info = model._update_symmetry_metadata(model.binary)
        self.active_symmetry = info
        reps = info.orbital_representations
        if reps is None:
            from .symmetry import _symmetry_ao_coeff
            coeff = _symmetry_ao_coeff(model.mf, coeff)
            mo_symmetry_representations(mol, coeff[:, :ncore])
            reps = mo_symmetry_representations(mol, coeff[:, ncore:ncore+ncas])
        super().__init__(reps, wfnsym, point_group=mol.symmetry_info.group)
        binary = model.binary
        self.alpha, self.beta = _basis_spin_strings(binary)
        if hasattr(binary, 'alpha_occ'):
            self.ia, self.ib = np.indices((len(self.alpha), len(self.beta)))
            self.ia, self.ib = self.ia.ravel(), self.ib.ravel()
        else:
            binary = np.ascontiguousarray(binary, dtype=np.int8)
            _, _, self.ia, self.ib = _cached_spin_string_basis(binary.shape, binary.tobytes())
            # Explicit determinant lists may omit or duplicate product entries.
            pairs = self.ia * len(self.beta) + self.ib
            if (len(pairs) != len(self.alpha) * len(self.beta)
                    or np.unique(pairs).size != len(pairs)):
                raise ValueError('Non-Abelian CI targeting requires a complete rectangular spin-string basis.')
        self.ndet = len(self.ia)
        unique, transforms, sparse_transforms, weights, indices = [], [], [], [], []
        same_strings = np.array_equal(self.alpha, self.beta)
        transform_bytes = 0
        with np.errstate(divide='ignore', invalid='ignore'):
            for op, rep in enumerate(self.matrices):
                index = next((i for i, previous in enumerate(unique)
                              if np.allclose(rep, previous, atol=1e-13, rtol=0)), None)
                if index is None:
                    # Dense string actions are retained even by the projected
                    # fallback; reject excess storage before constructing them.
                    itemsize = 16 if np.any(rep.imag) else 8
                    transform_bytes += itemsize * (len(self.alpha)**2 +
                        (0 if same_strings else len(self.beta)**2))
                    if transform_bytes > _MAX_STRING_ACTION_BYTES:
                        raise MemoryError('Spatial CI string transforms exceed the 256 MiB '
                                          'dense-action budget; reduce the active space.')
                    index = len(unique)
                    unique.append(rep)
                    wa, sa = _cached_string_transform(self.norb, rep.tobytes(), self.alpha.shape,
                                                  np.asarray(self.alpha, dtype=np.int8).tobytes())
                    wb, sb = (wa, sa) if same_strings else _cached_string_transform(
                        self.norb, rep.tobytes(), self.beta.shape,
                        np.asarray(self.beta, dtype=np.int8).tobytes())
                    transforms.append((wa, wb))
                    sparse_transforms.append((sa, sb))
                    weights.append(np.zeros_like(self.weights[:, :, op]))
                indices.append(index)
                weights[index] += self.weights[:, :, op]
        self.string_matrices = tuple(transforms[i] for i in indices)
        self._projector_terms = tuple(zip(weights, transforms))
        self._sparse_projector_terms = tuple(zip(weights, sparse_transforms))
        characters = np.array([np.trace(a)*np.trace(b) for a, b in self.string_matrices])
        self.multiplicity = int(round(np.dot(self.weights[0, 0], characters).real))

    def apply(self, vector, row=0, column=0):
        vector = np.asarray(vector)
        if vector.ndim not in (1, 2) or vector.shape[0] != self.ndet:
            raise ValueError('CI vectors must have shape (ndet,) or (ndet, nroots).')
        batch = vector[:, None] if vector.ndim == 1 else vector
        coeff = np.zeros((batch.shape[1], len(self.alpha), len(self.beta)), dtype=complex)
        coeff[:, self.ia, self.ib] = batch.T
        out = np.zeros_like(coeff)
        for weights, (wa, wb) in self._projector_terms:
            weight = weights[row, column]
            if abs(weight) > 1e-14:
                out += weight * (wa @ coeff @ wb.T)
        result = np.real_if_close(out[:, self.ia, self.ib].T)
        return result[:, 0] if vector.ndim == 1 else result



def run_spatial_ci(model, *, nstates, mo_coeff=None, ci0=None, use_cholesky=None,
                   wfnsym=None, spin_root_cushion=None,
                   spin_selection_tol=None, method="direct_ci", **kwargs):
    """Solve lowest CI multiplets in one irrep row, then retain every partner.

    ``nstates`` counts multiplets: an E target returns ``2*nstates`` roots.
    Davidson uses an exact representative-row Hamiltonian action within a
    joint spin/irrep basis; no reduced Hamiltonian is assembled. Ill-conditioned
    row selection uses sparse assembly, and excessive basis/block fill uses
    projected direct CI. Both full output-partner residuals are retained.
    A failed full residual check raises and leaves ``converged=False``; its
    tolerance is ten times the requested Davidson residual tolerance.
    This adapts Cotton's finite-group matrix units (Chemical Applications of
    Group Theory, 3rd ed., Wiley, 1990, ISBN 9780471510949) with Slater–Condon
    contraction and Davidson iteration. Projection is exact up to numerical rank
    tolerance; see ``pyqed.qchem.ci.spatial_blocks`` for storage limits.
    """
    from .mcscf.casci import _slice_active_orbitals, _reference_active_occupations, _resolve_use_cholesky
    from .ci.fci import get_fci_string_basis
    from .mcscf.direct_ci import CASCI as DirectCASCI, davidson_lowest, build_spin_string_connectivity

    model.converged = False
    model.direct_ci_diagnostics = {}
    if kwargs.get('solvent_response') is not None:
        raise NotImplementedError('Spatial targeting with solvent response needs invariant response operators.')
    group = model.mol.symmetry_info.group
    method_key = str(method).lower().replace('-', '_')
    if method_key in {'direct_spin0', 'direct_spin0_symm', 'spin0', 'spin0_symm'}:
        if model.nelecas_spin[0] != model.nelecas_spin[1] or model.multiplicity not in (None, 1):
            raise ValueError('Spin-zero spatial CI requires balanced alpha/beta strings and singlet spin.')
        model.multiplicity = 1
        model.target_s2 = 0.0
    elif method_key not in {'ci', 'direct_ci'}:
        raise ValueError('Spatial CI targeting supports ci, direct_ci, and direct_spin0_symm.')
    target = resolve_spatial_target(group, wfnsym)
    requested = int(nstates)
    if requested < 1:
        raise ValueError('nstates must be positive.')
    model.mo_coeff = model.mf.mo_coeff if mo_coeff is None else mo_coeff
    if isinstance(model.mo_coeff, (list, tuple)):
        raise NotImplementedError('Spatial CI targeting currently requires restricted orbitals.')
    if np.iscomplexobj(model.mo_coeff) and np.any(np.asarray(model.mo_coeff).imag):
        raise NotImplementedError('Spatial CI targeting currently requires real molecular orbitals.')
    model.mo_core, model.mo_cas = _slice_active_orbitals(model.mo_coeff, model.ncore, model.ncas)
    model.use_cholesky = _resolve_use_cholesky(model.mf, use_cholesky)
    if model.binary is None:
        model.binary = get_fci_string_basis(mo_occ=_reference_active_occupations(model.nelecas_spin, model.ncas))
    action = getattr(model, '_spatial_ci_action', None)
    if action is None:
        if group.continuous:
            from .orbital_symmetry import CIOrbitalAction
            action = CIOrbitalAction(model,target)
        else:
            action = CISpatialGroupAction(model, target)
    else:
        if action.wfnsym != target or action.norb != model.ncas or action.ndet != len(model.binary):
            raise ValueError('Cached spatial CI action does not match this solve.')
        model._set_symmetry_metadata(action.active_symmetry)
    if requested > action.multiplicity:
        raise ValueError(f'Only {action.multiplicity} {target} multiplets exist in this CI space.')
    direct = model if isinstance(model, DirectCASCI) else DirectCASCI(
        model.mf, ncas=model.ncas, nelecas=model.nelecas, ncore=model.ncore,
        ms2=model.ms2, multiplicity=model.multiplicity, verbose=0)
    direct.mo_coeff, direct.mo_core, direct.mo_cas = model.mo_coeff, model.mo_core, model.mo_cas
    direct.binary = model.binary
    direct.use_cholesky = model.use_cholesky
    h1, same, eri, energy_core = direct.get_direct_compact_integrals(use_cholesky=model.use_cholesky)
    action.validate_hamiltonian(h1, eri)
    direct.hcore = np.asarray([h1, h1])
    direct.h2e_cas = eri
    direct._direct_same_spin_eri, direct._direct_cross_spin_eri = same, eri
    direct.spin_string_connectivity = build_spin_string_connectivity(model.binary)
    cushion = model.spin_root_cushion if spin_root_cushion is None else int(spin_root_cushion)
    roots_to_solve = min(action.multiplicity, requested + (max(0, cushion) if model.multiplicity is not None else 0))
    # Reuse the determinant diagonal in tile assembly and the fallback preconditioner.
    from .mcscf.direct_ci import _compute_diag_compact
    diagonal = _compute_diag_compact(h1, same, eri, model.binary)
    direct._direct_H_diag = diagonal
    from .ci.spatial_blocks import row_basis, assemble_row_hamiltonian, matrix_free_row_action
    start = perf_counter()
    basis, cache_hit = row_basis(action,
        spin=None if model.multiplicity is None else (model.multiplicity-1)/2,
        conn=direct.spin_string_connectivity, binary=model.binary)
    plan_seconds = perf_counter() - start
    start = perf_counter()
    prepared = None if basis is None else matrix_free_row_action(
        basis, direct.spin_string_connectivity, h1, same, eri, diagonal)
    block = None
    contraction_info = {}
    if prepared is not None:
        operator, reduced_diagonal, contraction_info = prepared
    else:
        block = None if basis is None else assemble_row_hamiltonian(
            basis, direct.spin_string_connectivity, h1, same, eri, diagonal)
        operator = None if block is None else block.dot
        reduced_diagonal = None if block is None else block.diagonal().real
    assembly_seconds = perf_counter() - start
    options = dict(nroots=roots_to_solve,
        tol=getattr(model, 'direct_ci_residual_tol', None) or 1e-9,
        max_cycle=getattr(model, 'direct_ci_max_cycle', 100), return_info=True)
    start = perf_counter()
    if operator is None:
        values, vectors, info = davidson_lowest(
            lambda vector: action.apply(direct.ci_sigma(vector)), diagonal,
            guess=ci0, projector=action.apply, **options)
    else:
        if basis.shape[1] < requested:
            raise ValueError('Requested more roots than the joint spin/spatial sector contains.')
        options['nroots'] = requested
        if isinstance(ci0, (list, tuple)):
            ci0 = np.column_stack(ci0)
        guess = None if ci0 is None else basis.conj().T @ np.asarray(ci0)
        values, reduced, info = davidson_lowest(operator, reduced_diagonal,
            guess=guess, matvec_block=operator, **options)
        vectors = basis @ reduced
    info.update(symmetry_blocked=operator is not None, matrix_free=prepared is not None,
                transformation_cache_hit=cache_hit, **contraction_info,
                determinant_dimension=action.ndet, row_dimension=action.multiplicity if basis is None else basis.shape[1],
                block_nnz=0 if block is None else block.nnz,
                transformation_seconds=plan_seconds, assembly_seconds=assembly_seconds,
                davidson_seconds=perf_counter()-start)
    model.e_core = energy_core
    model.SC2 = getattr(model, "SC2", None)
    if operator is None:
        values, vectors = model._apply_multiplicity_selection(
            values, vectors, requested, spin_selection_tol=spin_selection_tol)
    info['spin_selection'] = 'joint_basis' if operator is not None else 'root_filter'

    partners = [vectors[:, root].copy() if operator is not None and row == 0
                else action.apply(vectors[:, root], row, 0)
                for root in range(requested) for row in range(action.dimension)]
    model.ci = partners
    model.e_core = energy_core
    model.e_tot = np.repeat(values + energy_core, action.dimension)
    model.nstates = len(partners)
    model.spatial_multiplet_count = requested
    model.spatial_partner_rows = tuple(range(action.dimension)) * requested
    model.spatial_irrep = model.wfnsym = target
    model.spatial_symmetry = action
    model.hcore, model.h2e_cas = direct.hcore, direct.h2e_cas
    model.eri_so = None
    model.solver_backend = 'spatial_symmetry_direct_ci'
    # The variational equation is solved within the selected row. Small
    # integral/factorization errors can leave an out-of-sector residual even
    # after that projected equation converges; retain it explicitly.
    if prepared is not None:
        from .ci.spatial_blocks import full_space_validation_action
        from .mcscf.direct_ci import _resolve_direct_ci_workers
        direct._direct_rhf_blas_matvec = full_space_validation_action(
            h1, same, eri, diagonal, direct.spin_string_connectivity, model.binary,
            workers=_resolve_direct_ci_workers(direct, action.ndet))
    info['unprojected_residual_norms'] = np.array([
        np.linalg.norm(direct.ci_sigma(vector) - value*vector)
        for value, vector in zip(np.repeat(values, action.dimension), partners)])
    info['residual_space'] = 'target_irrep_row'
    model.direct_ci_diagnostics = info
    info['unprojected_residual_tolerance'] = 10 * options['tol']
    residuals = info['unprojected_residual_norms']
    model.converged = bool(np.all(np.isfinite(residuals)) and
        np.all(residuals <= info['unprojected_residual_tolerance']))
    info['converged'] = model.converged
    if not model.converged:
        raise RuntimeError('Spatial CI failed the full Hamiltonian residual check; '
                           'inspect direct_ci_diagnostics before using these states.')
    model.wfnsym = target
    from .symmetry import report_ci_symmetry
    report_ci_symmetry(model)
    return model
