"""Local reduced-space assembly versus the unreduced group-action reference."""
import numpy as np
import pytest
from scipy.linalg import eigh

from pyqed.mps.nonabelian.environment import contract_chain_transition
from pyqed.mps.nonabelian.orbital_transform import apply_spatial_orbital_transform
from pyqed.mps.nonabelian.spatial_symmetry import SpatialSymmetryDMRG
from pyqed.qchem.symmetry import irrep_representation_matrices
from pyqed.symmetry import Leg


@pytest.mark.parametrize('nelec,spin,target', [(1, 1, 'E1u'), (2, 0, 'A1g'), (2, 0, 'E2g')])
def test_local_metric_compression_matches_full_group(nelec, spin, target):
    reps = {op: np.kron(np.eye(2), matrix)
            for op, matrix in irrep_representation_matrices('D6h', 'E1u').items()}
    h = np.kron([[.1, .3], [.3, .7]], np.eye(2))
    solver = SpatialSymmetryDMRG.from_integrals(h, np.zeros((4,)*4), contractions='group_action', nelec=nelec,
        spin=spin, D=1, seed=3, orbital_representations=reps, wfnsym=target, point_group='D6h')
    try:
        site = 1
        rng = np.random.default_rng(26)
        for position in range(4):
            vector = solver.seed._pack_site(position)
            solver.seed._set_site_vector(position, vector + .2j*rng.normal(size=vector.size))
        solver._refresh()
        baseline = list(solver.seed.materialize())
        assert all(isinstance(leg, Leg) for tensor in baseline for leg in tensor.legs)
        current = solver.seed._pack_site(site)
        basis = []
        for unit in np.eye(len(current)):
            solver.seed._set_site_vector(site, unit)
            chain = list(baseline)
            chain[site] = solver.seed.materialize()[site].copy()
            basis.append(chain)
        solver.seed._set_site_vector(site, current)
        metric = np.zeros((len(basis),)*2, complex)
        energy = np.zeros_like(metric)
        # Deliberately use all 24 original operations, including negative
        # orbital phases. This checks even/odd charge and every target row.
        action = solver.spatial_symmetry
        for j, ket in enumerate(basis):
            orbit = [apply_spatial_orbital_transform(ket, matrix, inverse=False,
                cutoff=0, max_bond=None, orbital_blocks=action.orbital_blocks).tensors
                for matrix in action.matrices]
            for i, bra in enumerate(basis):
                for factors, output in ((solver.identity, metric), (solver.mpo, energy)):
                    output[i,j] = sum(weight*contract_chain_transition(bra, factors, transformed)
                        for weight, transformed in zip(action.weights[0,0], orbit))
        values, vectors = eigh((metric+metric.conj().T)/2)
        keep = values > 1e-10*values.max()
        whitening = vectors[:,keep]/np.sqrt(values[keep])
        expected = np.linalg.eigvalsh(whitening.conj().T @ energy @ whitening)[0]
        record = solver.optimize_site(site)
        assert record['accepted']
        assert record['metric_rank'] == sum(keep)
        np.testing.assert_allclose(solver.energy, expected, atol=1e-10, rtol=0)
        assert solver.diagnostics['distinct_group_actions'] < 12
        solver._refresh()
        np.testing.assert_allclose(solver.e_tot, expected, atol=1e-10, rtol=0)
    finally:
        solver.close()


def test_shared_block_actions_preserve_complex_orbits():
    from scipy.linalg import block_diag
    components = [irrep_representation_matrices('D6h', name)
                  for name in ('A1g', 'E1g', 'A2u')]
    reps = {op: block_diag(*(component[op] for component in components))
            for op in components[0]}
    solver = SpatialSymmetryDMRG.from_integrals(np.diag([.2,.4,.4,.7]),
        np.zeros((4,)*4), contractions='group_action', nelec=2, D=1, orbital_representations=reps,
        wfnsym='A1g', point_group='D6h')
    try:
        assert any(len(set(labels)) < len(solver._matrices)
                   for labels in solver._block_actions.values())
        baseline = solver.seed.materialize()
        rng = np.random.default_rng(12)
        for site in range(4):
            cache = {}
            solver._orbit(baseline, site=site, cache=cache)
            varied = [tensor.copy() for tensor in baseline]
            for key, block in varied[site].data.items():
                varied[site].data[key] = rng.normal(size=block.shape) + 1j*rng.normal(size=block.shape)
            full = solver._orbit(varied)
            cached = solver._orbit(varied, site=site, cache=cache)
            for left, right in zip(full, cached):
                a = contract_chain_transition(left, solver.identity, left)
                b = contract_chain_transition(right, solver.identity, right)
                c = contract_chain_transition(left, solver.identity, right)
                np.testing.assert_allclose(a+b-2*c.real, 0, atol=1e-10)
    finally:
        solver.close()
