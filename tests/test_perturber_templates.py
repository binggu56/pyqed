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
def test_batched_fock_topology_matches_external_insertions(
    ncas, frozen, prune, monkeypatch
):
    monkeypatch.setattr(shared, "_FOCK_EDGE_BATCH", 3)
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
    streamed = {}
    for key, rr, cc, labels, base in shared._iter_fock_edges(
        groups, fock, ncore, ncas, 3
    ):
        assert len(rr) <= 3
        values = fock[labels[0], labels[1]] if key[2] == 2 else labels
        streamed.update(
            {
                (*key, int(r), int(c)): (value, base)
                for r, c, value in zip(rr, cc, values)
            }
        )
    assert streamed == expected
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


def test_fock_graph_skips_absent_destination_before_allocating_candidates():
    ncore, ncas, nvirt = 12, 2, 16
    nmo = ncore + ncas + nvirt
    patterns = (((0,), (0,)), ((0, 1), (0,)), ((0,), (0, 1)))
    groups = shared._grouped_external_spaces(ncore, nmo, ncas, 0, patterns)
    fock = np.ones((nmo, nmo))
    # Fit both surviving single insertions. The discarded double-insertion
    # batch from the (1, 1) class alone would exceed the budget.
    nsource = len(groups[patterns[0]])
    budget = 128 * (sum(map(len, groups.values())) + nsource * (ncore + nvirt))
    graph, entries, peak = shared._build_fock_topology(
        groups, fock, ncore, ncas, memory_bytes=budget
    )
    assert peak <= budget
    assert entries == nsource * (ncore + nvirt - 2)
    assert {(source, target, kind) for source, target, kind, _, _ in graph} == {
        (patterns[0], patterns[1], 0),
        (patterns[0], patterns[2], 1),
    }


def test_external_group_labels_store_only_the_axes():
    pattern = (0,), (0, 1)
    keys = shared._grouped_external_spaces(267, 798, 8, 82, (pattern,))[pattern]
    assert len(keys) == 25_253_055
    assert keys.holes.nbytes + keys.virtuals.nbytes < 1_100_000
    assert keys[0] == ((82,), (275, 276))
    assert keys[-1] == ((266,), (796, 797))
    holes, particles = keys.orbital_batch(136_502, 136_504)
    np.testing.assert_array_equal(holes, [[82], [83]])
    np.testing.assert_array_equal(particles, [[796, 797], [275, 276]])


def test_shared_minres_solves_symmetric_indefinite_equation():
    from scipy.sparse.linalg import aslinearoperator

    matrix = np.array([[-2.0, 0.4, -0.2], [0.4, 1.3, 0.5], [-0.2, 0.5, -0.6]])
    rhs = np.array([1.0, -3.0, 2.0])
    diagonal = np.diag(matrix).copy()
    solution, iterations, history = shared._solve_shared_minres(
        aslinearoperator(matrix), rhs, diagonal, rhs / diagonal, 1e-12, 10
    )
    np.testing.assert_allclose(solution, np.linalg.solve(matrix, rhs), atol=1e-12)
    assert 0 < iterations <= 10
    assert history[-1] < 1e-12


@pytest.mark.parametrize(
    "shift,real_shift,frozen",
    [(0.0, 0.0, 0), (0.25, 0.0, 1), (0.0, 0.1, 0), (0.25, 0.1, 0)],
)
@pytest.mark.parametrize("cache_limit", [0, 1, 16 << 20])
def test_eliminated_external_doubles_match_coupled_shared_solver(
    shift, real_shift, frozen, cache_limit, monkeypatch
):
    monkeypatch.setattr(shared, "_FOCK_CACHE_BYTES", cache_limit)
    monkeypatch.setattr(shared, "_DOUBLE_TILE_PAIRS", 6)
    monkeypatch.setattr(shared, "_DIAGONAL_BLOCK_SIZE", 3)
    rhs_buffers = []
    import weakref

    pair_blocks = []
    prepare_pairs = shared._ExcitationCoefficients.prepare_active_pair_integrals

    def record_pairs(self, *args):
        result = prepare_pairs(self, *args)
        if self.active_pair_integrals is not None:
            pair_blocks.append(weakref.ref(self.active_pair_integrals.values))
        return result

    monkeypatch.setattr(
        shared._ExcitationCoefficients, "prepare_active_pair_integrals", record_pairs
    )
    build_rhs = shared._ActiveTemplate.tiled_rhs

    def record_rhs(self, groups, coefficients, out=None):
        assert out is not None
        result = build_rhs(self, groups, coefficients, out=out)
        assert result is out
        rhs_buffers.append(out)
        return result

    solve = shared._solve_shared_minres

    def check_rhs_alias(operator, rhs, *args, **kwargs):
        assert all(np.shares_memory(rhs, block) for block in rhs_buffers)
        assert pair_blocks and all(block() is None for block in pair_blocks)
        return solve(operator, rhs, *args, **kwargs)

    monkeypatch.setattr(shared._ActiveTemplate, "tiled_rhs", record_rhs)
    monkeypatch.setattr(shared, "_solve_shared_minres", check_rhs_alias)
    ncore, ncas, nmo = 5, 2, 11
    binary, ci = _reference()
    rng = np.random.default_rng(18)
    h = rng.normal(size=(nmo, nmo))
    h = (h + h.T) / 2
    factors = rng.normal(size=(5, nmo, nmo)) * 0.1
    factors = (factors + factors.transpose(0, 2, 1)) / 2
    eri = np.einsum("Lpq,Lrs->pqrs", factors, factors)
    core_potential = sum(2 * eri[:, :, i, i] - eri[:, i, i, :] for i in range(ncore))
    reference_factors = shared.ReferencePairFactors(
        factors[:, : ncore + ncas], core_potential
    )
    fock = np.diag(np.linspace(-6.0, 5.0, nmo))
    partitions = np.repeat([0, 1, 2], [ncore, ncas, nmo - ncore - ncas])
    cross = rng.normal(size=(nmo, nmo)) * 0.1
    fock += (cross + cross.T) * (partitions[:, None] != partitions)
    da, db = pt._active_spin_rdms_from_determinants(binary, ci)
    active = slice(ncore, ncore + ncas)
    active_fock = np.sum(fock[active, active] * (da + db))
    other = rng.normal(size=len(ci))
    other -= (other @ ci) * ci
    options = dict(
        frozen_core=frozen,
        imaginary_shift=shift,
        real_shift=real_shift,
        fock_contraction="rdm",
        ic_convention="molcas",
        norm_tol=1e-10,
        transition_ci=[ci, other / np.linalg.norm(other)],
    )
    full = shared.solve_shared_fois(
        binary, ci, ncore, ncas, h, eri, fock, active_fock, **options
    )

    def stored_topology(*args, **kwargs):
        pytest.fail("The streamed solve must not construct stored topology.")

    monkeypatch.setattr(shared, "_fock_topology_indices", stored_topology)
    result = shared.solve_shared_fois(
        binary,
        ci,
        ncore,
        ncas,
        h,
        reference_factors,
        fock,
        active_fock,
        eliminate_external_doubles=True,
        direct_workers=2,
        **options,
    )
    assert (result["fock_topology_bytes"] > 0) == (cache_limit > 1)
    assert shared._fock_topology_cache is None
    assert len(pair_blocks) == 2
    midpoint = len(rhs_buffers) // 2
    assert midpoint > 0
    for first, restored in zip(rhs_buffers[:midpoint], rhs_buffers[midpoint:]):
        assert first is restored
    assert result["rank"] == full["rank"]
    assert result["rank"] > result["iterative_rank"]
    assert result["eliminated_groups"] == full["groups"] - result["groups"]
    assert result["external_double_workers"] == 2
    assert result["solver"] == "coupled_template_minres"
    assert result["work_vectors"] < full["work_vectors"]
    assert result["iterations"] > 0
    assert result["equation_residual"] < 1e-9
    for key in ("energy", "nonvariational", "norm", "transition_couplings", "residual"):
        np.testing.assert_allclose(result[key], full[key], atol=1e-9, rtol=0.0)
    for name, component in full["components"].items():
        assert result["components"][name].count == component.count
        np.testing.assert_allclose(
            result["components"][name].energy, component.energy, atol=1e-9, rtol=0.0
        )


def test_eliminated_doubles_preflight_does_not_allocate_external_labels(monkeypatch):
    counts = shared._external_pattern_counts(185, 523)
    assert (
        sum(n for (h, v), n in counts.items() if len(h) == len(v) == 2) == 2_357_532_330
    )
    ncore, ncas, nmo = 185, 2, 710
    binary, ci = _reference()
    one_body = np.diag(np.linspace(-6.0, 5.0, nmo))
    integrals = shared.ReferencePairFactors(np.empty((0, ncore + ncas, nmo)), one_body)

    def unexpected(*args):
        pytest.fail("External orbital labels were allocated before the memory check.")

    monkeypatch.setattr(shared, "_grouped_external_spaces", unexpected)
    with pytest.raises(MemoryError, match="Shared FIC workspace requires"):
        shared.solve_shared_fois(
            binary,
            ci,
            ncore,
            ncas,
            one_body,
            integrals,
            one_body,
            0.0,
            fock_contraction="rdm",
            norm_tol=1e-10,
            max_memory_mb=8.0,
            eliminate_external_doubles=True,
        )


def test_ri_active_pair_integrals_match_direct_dots_and_respect_budget():
    rng = np.random.default_rng(39)
    ncore, ncas, nmo, frozen = 3, 2, 9, 1
    factors = rng.normal(size=(7, nmo, nmo))
    factors += factors.transpose(0, 2, 1).copy()
    h = np.zeros((nmo, nmo))
    eri = shared.ReferencePairFactors(factors[:, frozen : ncore + ncas], h, frozen)
    coeff = shared._ExcitationCoefficients(h, eri, ncore, ncas, frozen)
    cache = {}
    assert coeff.prepare_active_pair_integrals(1, cache) == (0, False)
    size, reused = coeff.prepare_active_pair_integrals(1 << 20, cache)
    assert size > 0 and not reused
    p, q, r, s = np.array([[6, 7, 3], [3, 1, 1], [5, 3, 7], [2, 1, 4]])
    np.testing.assert_allclose(
        coeff.retained_pair(p, q, r, s), coeff.pair(p, q, r, s), atol=1e-12
    )
    other = shared._ExcitationCoefficients(h, eri, ncore, ncas, frozen)
    assert other.prepare_active_pair_integrals(1 << 20, cache) == (size, True)
    assert other.prepare_active_pair_integrals(1, cache) == (0, False)
    assert not cache


@pytest.mark.parametrize("singular", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_shared_minres_scaled_indefinite_reference(singular, in_place):
    from scipy.sparse.linalg import aslinearoperator

    rng = np.random.default_rng(824)
    q, _ = np.linalg.qr(rng.normal(size=(24, 24)))
    spectrum = np.linspace(-2.0, 3.0, 24)
    if singular:
        spectrum[8] = 0.0
    scale = np.geomspace(0.1, 10.0, 24)
    matrix = scale[:, None] * ((q * spectrum) @ q.T) * scale[None, :]
    rhs = matrix @ rng.normal(size=24)

    def accumulate(vector, out):
        out += matrix @ vector

    solution, _, history = shared._solve_shared_minres(
        aslinearoperator(matrix),
        rhs,
        scale**2,
        np.zeros(24),
        1e-11,
        300,
        accumulate=accumulate if in_place else None,
    )
    assert np.linalg.norm(matrix @ solution - rhs) / np.linalg.norm(rhs) < 1e-11
    assert history[-1] < 1e-11


def test_shared_minres_chunk_boundary_and_exact_initial():
    from scipy.sparse.linalg import LinearOperator

    diagonal = np.resize(np.array([-2.0, 3.0, 5.0]), (1 << 16) + 7)
    operator = LinearOperator((len(diagonal),) * 2, matvec=lambda x: diagonal * x)
    rhs = np.ones_like(diagonal)
    solution, iterations, _ = shared._solve_shared_minres(
        operator, rhs, abs(diagonal), np.zeros_like(rhs), 1e-12, 10
    )
    np.testing.assert_allclose(solution, 1.0 / diagonal, atol=1e-12)
    assert iterations <= 3
    solution, iterations, _ = shared._solve_shared_minres(
        operator, rhs, abs(diagonal), 1.0 / diagonal, 1e-12, 10
    )
    assert iterations == 0


def test_shared_minres_rejects_inconsistent_singular_equation():
    from scipy.sparse.linalg import aslinearoperator

    with pytest.raises(RuntimeError, match="MINRES residual"):
        shared._solve_shared_minres(
            aslinearoperator(np.diag([1.0, 0.0])),
            np.ones(2),
            np.ones(2),
            np.zeros(2),
            1e-12,
            10,
        )


@pytest.mark.parametrize("block_size", [1, 7, 64])
def test_shared_minres_generated_preconditioner(block_size):
    from scipy.sparse.linalg import aslinearoperator

    rng = np.random.default_rng(192)
    matrix = rng.normal(size=(19, 19))
    matrix += matrix.T
    rhs = rng.normal(size=19)
    diagonal = np.maximum(abs(np.diag(matrix)), 0.1)
    calls = []

    def blocks():
        calls.append(True)
        for start in range(0, len(rhs), block_size):
            rows = slice(start, start + block_size)
            yield rows, diagonal[rows].copy()

    args = (aslinearoperator(matrix), rhs)
    dense, _, _ = shared._solve_shared_minres(*args, diagonal.copy(), None, 1e-11, 100)
    blocked, _, history = shared._solve_shared_minres(*args, blocks, None, 1e-11, 100)
    np.testing.assert_allclose(blocked, dense, atol=1e-9, rtol=1e-10)
    assert history[-1] < 1e-11
    assert len(calls) > 1


@pytest.mark.parametrize("shift", [0.0, 0.3])
def test_shared_minres_accumulates_into_two_residual_buffers(shift):
    from scipy.sparse.linalg import LinearOperator

    rng = np.random.default_rng(293)
    matrix = rng.normal(size=(31, 31))
    matrix += matrix.T
    matrix += shift * np.eye(31)
    rhs = rng.normal(size=31)
    diagonal = np.maximum(abs(np.diag(matrix)), 0.2)
    products, buffers, input_buffers = [], [], []

    def matvec(vector):
        products.append(True)
        return matrix @ vector

    def accumulate(vector, out):
        assert not np.shares_memory(vector, out)
        buffers.append(out.__array_interface__["data"][0])
        input_buffers.append(vector.__array_interface__["data"][0])
        out += matrix @ vector

    operator = LinearOperator(matrix.shape, matvec=matvec, dtype=float)
    solution, iterations, history = shared._solve_shared_minres(
        operator, rhs, diagonal, None, 1e-10, 100, accumulate=accumulate
    )
    np.testing.assert_allclose(
        solution, np.linalg.solve(matrix, rhs), atol=1e-8, rtol=1e-9
    )
    assert history[-1] < 1e-10
    assert len(buffers) == iterations
    assert len(products) == len(history) + 1
    # Each uninterrupted Lanczos recurrence rotates just two residual buffers.
    assert len(set(buffers) | set(input_buffers)) <= 2 * len(history)


@pytest.mark.parametrize("ranks", [(2, 5), (5, 2), (3, 3)])
@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("panel_rows", [3, 128])
def test_fock_edge_batches_match_dense_with_repeated_indices(ranks, cross, panel_rows):
    from pyqed.qchem.mcscf.fic_fock_action import apply_compact_fock_graph

    rng = np.random.default_rng(408)
    rs, rt = ranks
    xs, xt = rng.normal(size=(7, rs)), rng.normal(size=(9, rt))
    matrix = rng.normal(size=(rt, rs))
    rows, columns = rng.integers(9, size=31), rng.integers(7, size=31)
    labels = rng.integers(4, size=(2 if cross else 1, 31))
    weights = rng.normal(size=(4, 4 if cross else 1))
    weights[0, 0] = 1e-16
    graph = np.zeros((9, 7))
    edge_weights = weights[labels[0], labels[1] if cross else 0]
    edge_weights[abs(edge_weights) < 1e-14] = 0.0
    np.add.at(graph, (rows, columns), edge_weights)
    ys, yt = rng.normal(size=xs.shape), rng.normal(size=xt.shape)
    expected_s = ys - (graph.T @ xt) @ matrix
    expected_t = yt - (graph @ xs) @ matrix.T
    terms = [(0, 1, rows, columns, labels, weights, matrix)]
    scratch = panel_rows * max(rs, rt)
    apply_compact_fock_graph(
        {0: xs, 1: xt}, {0: ys, 1: yt}, terms, np.empty(scratch), np.empty(scratch)
    )
    np.testing.assert_allclose(ys, expected_s, atol=1e-12)
    np.testing.assert_allclose(yt, expected_t, atol=1e-12)
