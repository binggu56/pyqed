from itertools import combinations

import numpy as np
import pytest

from pyqed.qchem.mcscf import caspt2 as pt
from pyqed.qchem.mcscf import perturber_templates as shared


def _reference(ncas=2, nele=1):
    strings = list(combinations(range(ncas), nele))
    binary = np.zeros((len(strings) ** 2, 2, ncas), dtype=np.int8)
    for k, (a, b) in enumerate((a, b) for a in strings for b in strings):
        binary[k, 0, list(a)] = 1
        binary[k, 1, list(b)] = 1
    rng = np.random.default_rng(19)
    ci = rng.normal(size=len(binary))
    return binary, ci / np.linalg.norm(ci)


@pytest.mark.parametrize("nmo", [3, 32, 35])
def test_sparse_excitation_arrays_match_fermion_reference(nmo):
    rng = np.random.default_rng(73)
    rows = sorted(
        {
            sum(int(b) << i for i, b in enumerate(rng.integers(0, 2, 2 * nmo)))
            for _ in range(40)
        }
    )
    rows = np.array(rows, dtype=np.uint64 if 2 * nmo <= 64 else object)
    values = rng.normal(size=len(rows))
    original = rows.copy(), values.copy()
    for p, q in [(0, nmo - 1), (nmo - 1, 0), (nmo - 1, nmo - 1)]:
        bits, amplitudes = shared._excitation_action((rows, values), p, q, nmo)
        actual = dict(zip(map(int, bits), amplitudes))
        expected = pt._apply_spatial_one_body(
            dict(zip(map(int, rows), values)), p, q, nmo
        )
        for det in actual.keys() | expected.keys():
            np.testing.assert_allclose(
                actual.get(det, 0), expected.get(det, 0), atol=1e-14
            )
    np.testing.assert_array_equal(rows, original[0])
    np.testing.assert_array_equal(values, original[1])


@pytest.mark.parametrize("ncore, nmo", [(3, 8), (33, 38)])
def test_template_embedding_preserves_external_fermion_phases(ncore, nmo):
    binary, ci = _reference()
    template = shared._ActiveTemplate(((0,), (0,)), binary, ci, np.eye(2), 1e-12, 1e-10)
    group = (1,), (nmo - 1,)
    rows, phases = shared._mapped_rows(template, *group, ncore, nmo)
    mapping = template.orbital_map(*group, ncore)
    reference = dict(zip(pt._embed_active_determinants(binary, ncore, 2, nmo), ci))
    for j, k in enumerate(np.flatnonzero(template.nonzero)):
        state = reference
        for p, q in reversed(template.operators[k]):
            state = pt._apply_spatial_one_body(
                state, int(mapping[p]), int(mapping[q]), nmo
            )
        actual = phases * template.raw[:, j].toarray().ravel() * template.norms[k]
        np.testing.assert_allclose(
            actual, [state.get(det, 0.0) for det in rows], atol=1e-13
        )


def test_diagonal_active_fock_moment_matches_explicit_operator():
    binary, ci = _reference(ncas=3)
    fock = np.diag([-1.2, -0.4, 0.7])
    template = shared._ActiveTemplate(((0,), (0,)), binary, ci, fock, 1e-12, 1e-10)
    full = np.zeros((template.nmo, template.nmo))
    full[1:4, 1:4] = fock
    q = template.raw @ template.transform
    explicit = pt._direct_local_fock_matrix(template.rows, full, 1, 3, template.nmo)
    np.testing.assert_allclose(template.fock, q.T @ explicit @ q, atol=1e-12)


@pytest.mark.parametrize("frozen", [0, 1])
def test_shared_template_rhs_matches_slater_condon(frozen):
    ncore, ncas, nmo = 2, 2, 6
    binary, ci = _reference()
    refs = pt._embed_active_determinants(binary, ncore, ncas, nmo)
    rng = np.random.default_rng(13)
    h = rng.normal(size=(nmo, nmo))
    h = (h + h.T) / 2
    factors = rng.normal(size=(4, nmo, nmo))
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    single, pair, ids = shared._hamiltonian_coefficients(h, eri, ncore, ncas, frozen)
    for pattern, groups in shared._grouped_external_spaces(
        ncore, nmo, ncas, frozen
    ).items():
        template = shared._ActiveTemplate(
            pattern, binary, ci, h[2:4, 2:4], 1e-12, 1e-10
        )
        rhs = template.rhs(groups, ncore, single, pair, ids)
        for actual, group in zip(rhs, groups):
            rows, phases = shared._mapped_rows(template, *group, ncore, nmo)
            coupling = np.array(
                [
                    sum(
                        c * pt._hamiltonian_element_bits(row, ref, h, eri, nmo)
                        for c, ref in zip(ci, refs)
                    )
                    for row in rows
                ]
            )
            expected = template.transform.T @ (template.raw.T @ (phases * coupling))
            np.testing.assert_allclose(actual, expected, atol=1e-11)
    sf, pf, _ = shared._hamiltonian_coefficients(h, factors, ncore, ncas, frozen)
    np.testing.assert_allclose(sf, single, atol=1e-12)
    np.testing.assert_allclose(pf, pair, atol=1e-12)


@pytest.mark.parametrize("ncas", [0, 2])
@pytest.mark.parametrize("frozen", [0, 1, 3])
@pytest.mark.parametrize("prune", [False, True])
def test_batched_fock_topology_matches_external_insertions(ncas, frozen, prune):
    ncore = 3
    nmo = ncore + ncas + 3
    groups = shared._grouped_external_spaces(ncore, nmo, ncas, frozen)
    if prune:
        groups = {p: g for i, (p, g) in enumerate(groups.items()) if i % 3}
    fock = np.random.default_rng(274).normal(size=(nmo, nmo))
    fock[ncore : ncore + ncas, 2] = 0
    fock[-1, ncore : ncore + ncas] = 0
    fock[ncore + ncas, 1] = 1e-15
    lookup = {g: (p, i) for p, rows in groups.items() for i, g in enumerate(rows)}
    expected = {}
    for s, rows in groups.items():
        for col, (h, v) in enumerate(rows):
            changes = []
            if ncas and len(h) < 2:
                changes.extend(
                    (0, ncore, i, (tuple(sorted(h + (i,))), v))
                    for i in range(frozen, ncore)
                    if np.any(abs(fock[ncore : ncore + ncas, i]) >= 1e-14)
                )
            if ncas and len(v) < 2:
                changes.extend(
                    (1, r, ncore, (h, tuple(sorted(v + (r,)))))
                    for r in range(ncore + ncas, nmo)
                    if np.any(abs(fock[r, ncore : ncore + ncas]) >= 1e-14)
                )
            if len(h) < 2 and len(v) < 2:
                changes.extend(
                    (2, r, i, (tuple(sorted(h + (i,))), tuple(sorted(v + (r,)))))
                    for i in range(frozen, ncore)
                    for r in range(ncore + ncas, nmo)
                    if abs(fock[r, i]) >= 1e-14
                )
            for kind, p, q, target in changes:
                if target not in lookup:
                    continue
                t, row = lookup[target]
                ho = int(np.sign(q - h[0])) if kind != 1 and h else 0
                vo = int(np.sign(p - v[0])) if kind != 0 and v else 0
                key = (s, t, kind, ho, vo, row, col)
                expected[key] = (
                    q if kind == 0 else p if kind == 1 else fock[p, q],
                    shared._transition_key((h, v), target, p, q, ncore, ncas),
                )
    graph, count = shared._fock_topology(groups, fock, ncore, ncas)
    actual = {
        (*key, int(r), int(c)): (value, base)
        for key, (rr, cc, values, base) in graph.items()
        for r, c, value in zip(rr, cc, values)
    }
    assert count == len(expected)
    assert actual == expected
    if groups:
        with pytest.raises(MemoryError, match="during assembly"):
            shared._fock_topology(groups, fock, ncore, ncas, memory_bytes=1)


@pytest.mark.parametrize("shift", [0.0, 0.25])
@pytest.mark.parametrize("real_shift", [0.0, 0.1])
@pytest.mark.parametrize("frozen", [0, 1, 2])
@pytest.mark.parametrize("ncore", [2, 3])
@pytest.mark.parametrize("fock_contraction", ["operator", "density", "rdm"])
def test_shared_fois_matches_full_coupled_direct_equation(
    shift, real_shift, frozen, ncore, fock_contraction
):
    ncas, nmo = 2, ncore + 4
    binary, ci = _reference()
    refs = pt._embed_active_determinants(binary, ncore, ncas, nmo)
    rng = np.random.default_rng(23)
    h = rng.normal(size=(nmo, nmo))
    h = (h + h.T) / 2
    factors = rng.normal(size=(4, nmo, nmo)) * 0.1
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    fock = rng.normal(size=(nmo, nmo)) * 0.03
    fock = (fock + fock.T) / 2
    fock[:ncore, :ncore] = np.diag(np.arange(ncore) - 5.0)
    fock[ncore + 2 :, ncore + 2 :] = np.diag([3.0, 4.0])
    active = slice(ncore, ncore + 2)
    fock[active, active] += np.diag([-1.0, -0.5])
    if fock_contraction == "rdm":
        fock[active, active] = np.diag(np.diag(fock[active, active]))
    da, db = pt._active_spin_rdms_from_determinants(binary, ci)
    active_fock = float(np.sum(fock[active, active] * (da + db)))
    blocks, _ = pt._build_direct_tensor_blocks(
        refs, ci, ncore, ncas, nmo, frozen_core=frozen
    )
    expected = pt._solve_direct_signature_blocks(
        blocks,
        refs,
        ci,
        h,
        eri,
        fock,
        2 * np.trace(fock[:ncore, :ncore]) + active_fock,
        ncore,
        ncas,
        nmo,
        real_shift=real_shift,
        imaginary_shift=shift,
        denominator_tol=1e-12,
        workers=1,
    )
    result = shared.solve_shared_fois(
        binary,
        ci,
        ncore,
        ncas,
        h,
        eri,
        fock,
        active_fock,
        frozen_core=frozen,
        imaginary_shift=shift,
        real_shift=real_shift,
        fock_contraction=fock_contraction,
        norm_tol=1e-10 if fock_contraction == "rdm" else 0.0,
    )
    assert result["rank"] == expected["rank"]
    np.testing.assert_allclose(
        result["energy"], expected["variational_energy"], atol=1e-10
    )
    np.testing.assert_allclose(result["norm"], expected["first_order_norm"], atol=1e-10)
    assert result["equation_residual"] < 1e-9


def test_shared_caspt2_driver_matches_direct_and_mp2_limits():
    from pyqed.qchem import CASCI, Molecule

    mol = Molecule(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", unit="angstrom").build(
        eri="s8"
    )
    mf = mol.RHF().run()
    for ncas, nelecas in ((0, 0), (2, 2)):
        mc = CASCI(mf, ncas=ncas, nelecas=nelecas).run(nstates=1, method="ci")
        for shift in (0.0, 0.25):
            options = dict(ic_basis_backend="direct", imaginary_shift=shift)
            direct = pt.CASPT2(mc, **options)
            direct.run()
            factored = pt.CASPT2(mc, direct_build_backend="shared", **options)
            factored.run()
            np.testing.assert_allclose(factored.e_corr, direct.e_corr, atol=1e-10)
            np.testing.assert_allclose(
                factored.reference_weight, direct.reference_weight, atol=1e-10
            )
            assert factored.success
            assert factored.ic_metric_backend == "shared_active_templates"


def test_shared_template_memory_guard_precedes_metric_allocation():
    binary, ci = _reference()
    with pytest.raises(MemoryError, match="during assembly"):
        shared._ActiveTemplate(
            ((0,), (0,)), binary, ci, np.eye(2), 1e-12, 1e-10, memory_bytes=1
        )


def test_uncoupled_zero_denominators_have_zero_amplitude():
    binary, ci = _reference()
    result = shared.solve_shared_fois(
        binary, ci, 0, 2, np.zeros((3, 3)), np.zeros((3,) * 4), np.zeros((3, 3)), 0.0
    )
    assert result["energy"] == 0.0
    assert result["norm"] == 0.0


def test_molcas_metric_screens_squared_norms_and_folds_singles():
    binary = np.array([[[1, 0], [1, 0]], [[0, 1], [0, 1]]])
    ci = np.array([np.sqrt(1 - 1e-12), 1e-6])
    t = shared._ActiveTemplate(
        ((0,), ()),
        binary,
        ci,
        np.eye(2),
        1e-14,
        1e-8,
        ic_convention="molcas",
        norm_tol=1e-10,
    )
    # E_ai E_bb on the small closed-pair component has squared norm 8e-12,
    # so Molcas THRSHN removes it. The large component leaves three columns.
    assert t.raw.shape[1] == 3
    assert t.rank == 2
    q = t.raw @ t.transform
    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)


@pytest.mark.parametrize(
    "shift, expected", [(0.0, -0.0717435142), (0.25, -0.0717229841)]
)
def test_nonstationary_reference_matches_independent_molcas(shift, expected):
    """OpenMolcas 26.06, fixed H6 STO-3G CAS(2,2); decks in benchmarks/data.

    Mixing inactive/virtual RHF orbitals exposes the A/E normalization. A
    stationary small-molecule reference does not reliably exercise that term.
    """
    from pyqed.qchem import CASCI, Molecule

    atom = "H 0 0 0; H 0 0 .7; H 1.1 0 .1; H 1.4 .2 .8; " "H 2 -.3 .2; H 2.5 .4 .7"
    mol = Molecule(atom=atom, basis="sto-3g", unit="angstrom").build(eri="s8")
    mf = mol.RHF().run()
    c = mf.mo_coeff.copy()
    c *= np.sign(c[np.argmax(abs(c), axis=0), np.arange(len(c))])
    angle = 0.1
    c[:, [0, 5]] = c[:, [0, 5]] @ np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    mc = CASCI(mf, 2, 2).run(mo_coeff=c, nstates=1, method="direct_spin0")
    saved_ci, saved_mo = mc.ci.copy(), mc.mo_coeff.copy()
    solver = pt.CASPT2(
        mc,
        ic_basis_backend="direct",
        direct_build_backend="shared",
        ic_convention="molcas",
        imaginary_shift=shift,
    )
    solver.run()
    np.testing.assert_allclose(mc.e_tot[0], -2.7600130349, atol=1e-9, rtol=0)
    np.testing.assert_allclose(solver.e_corr, expected, atol=1e-9, rtol=0)
    assert solver.contracted_basis_rank == 72
    assert solver.lindep_tol == 1e-8 and solver.metric_norm_tol == 1e-10
    np.testing.assert_array_equal(mc.ci, saved_ci)
    np.testing.assert_array_equal(mc.mo_coeff, saved_mo)


def test_ordered_transition_rdms_match_ras_external_actions():
    from pyqed.qchem.mcscf.rdm_contractions import RDMContractions

    binary, ci = _reference(3, 1)
    binary, ci = binary[:5], ci[:5]
    bra = np.arange(1.0, 6.0)
    bra /= np.linalg.norm(bra)
    density = RDMContractions.transition(bra, ci, binary)
    nc, nmo = 1, 5
    bits = np.array(pt._embed_active_determinants(binary, nc, 3, nmo), dtype=np.uint64)
    ops = [
        op
        for op in shared._operators(nc, 3, nmo)
        if shared._external_key(op, nc, 3) == ((0,), (4,))
    ]

    def columns(vector):
        result = []
        for op in ops:
            state = (bits, vector)
            for p, q in reversed(op):
                state = shared._excitation_action(state, p, q, nmo)
            result.append(dict(zip(map(int, state[0]), state[1])))
        return result

    left, right = columns(bra), columns(ci)
    expected = np.array(
        [[sum(v * b.get(k, 0.0) for k, v in a.items()) for b in right] for a in left]
    )
    np.testing.assert_allclose(density.matrix(ops, ops, nc), expected, atol=1e-12)
