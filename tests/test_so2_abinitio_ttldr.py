import os
from pathlib import Path

import numpy as np
import pytest

from examples.ldr.so2_casci_cgldr import (
    REFERENCE_BOND,
    REFERENCE_THETA_DEG,
    casci_reference_point,
    casci_overlap_active,
    require_smooth_active_space,
    so2_body_frame,
)
from examples.ldr.so2_casci_full_ldr import (
    STATE_IDS,
    full_hamiltonian,
    path_overlap,
)
from examples.ldr.so2_procrustes_gauge import stitch_upper
from examples.namd.so2_abinitio_ftt_ttldr import propagate
from pyqed.ldr import AbInitioFit, keo
from pyqed.ldr.overlap import procrustes
from pyqed.ldr.ttfit import LinkPath
from pyqed.namd.ttldr import TTLDR
from pyqed.namd.triatomic import Triatom


FIXTURE = Path(__file__).parent / "data" / "so2_am1_meci_3x3x3.npz"


def test_stitch_upper_keeps_lower_atlas_and_aligns_interface():
    angle = 0.37
    rotation = np.asarray(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]],
        dtype=complex,
    )
    lower = np.repeat(np.eye(2, dtype=complex)[None, :, :], 3, axis=0)
    upper = np.repeat(rotation[None, :, :], 3, axis=0)
    positive = np.diag([0.8, 0.5])
    links = {(0, (0,)): np.eye(2), (0, (1,)): positive}

    combined, _transition = stitch_upper(
        (3,), links, lower, upper, axis=0, boundary=1
    )

    np.testing.assert_allclose(combined[:2], lower[:2], atol=1.0e-14)
    interface = combined[1].conj().T @ links[(0, (1,))] @ combined[2]
    np.testing.assert_allclose(procrustes(interface)[0], np.eye(2), atol=1.0e-14)


def test_dense_reference_propagates_complex_hermitian_state_from_t0():
    hamiltonian = np.asarray([[0.2, 0.3j], [-0.3j, -0.1]])
    initial = np.asarray([1.0 + 0.2j, -0.4j])
    initial /= np.linalg.norm(initial)
    states = propagate(hamiltonian, initial, np.asarray([0.0, 0.3]))

    np.testing.assert_allclose(states[0], initial, atol=1.0e-14)
    np.testing.assert_allclose(np.linalg.norm(states, axis=1), 1.0, atol=1.0e-14)


def _kinetic_terms(grids):
    identities = tuple(np.eye(len(grid)) for grid in grids)
    terms = []
    for axis, grid in enumerate(grids):
        spacing = float(np.mean(np.diff(grid)))
        laplacian = np.diag(np.full(len(grid), 2.0))
        laplacian += np.diag(np.full(len(grid) - 1, -1.0), 1)
        laplacian += np.diag(np.full(len(grid) - 1, -1.0), -1)
        factors = list(identities)
        factors[axis] = laplacian / (2.0 * spacing**2)
        terms.append((1.0, tuple(factors)))
    return tuple(terms)


def _jacobi_keo(triatom):
    atom_a, atom_b, atom_c = triatom._jacobi_atoms
    mass_a, mass_b, mass_c = (
        float(triatom.mass[atom]) for atom in (atom_a, atom_b, atom_c)
    )
    masses = (
        mass_b * mass_c / (mass_b + mass_c),
        mass_a * (mass_b + mass_c) / (mass_a + mass_b + mass_c),
    )
    return keo.jacobi(triatom.dvrs, mass=masses, inertia=None)


def _fixed_overlap(shape, nstates, links):
    transport = LinkPath(shape, nstates, links)
    indices = tuple(np.ndindex(shape))
    overlap = np.empty(
        (len(indices), nstates, len(indices), nstates),
        dtype=complex,
    )
    for left_flat, left in enumerate(indices):
        for right_flat, right in enumerate(indices):
            overlap[left_flat, :, right_flat, :] = transport.between(left, right)
    return overlap


def _align_hamiltonian(hamiltonian, gauges):
    ngrid, nstates = gauges.shape[:2]
    blocks = np.asarray(hamiltonian).reshape(ngrid, nstates, ngrid, nstates)
    aligned = np.einsum(
        "ipa,ipjq,jqb->iajb",
        gauges.conj(),
        blocks,
        gauges,
        optimize=True,
    )
    return aligned.reshape(ngrid * nstates, ngrid * nstates)


def test_so2_abinitio_fit_to_ttldr_propagation(tmp_path):
    with np.load(FIXTURE, allow_pickle=False) as archive:
        grids = tuple(np.asarray(archive[name]) for name in ("r1", "r2", "theta"))
        energies = np.asarray(archive["energies"])
        link_arrays = tuple(np.asarray(archive[f"links_{axis}"]) for axis in range(3))

    shape = energies.shape[:-1]
    nstates = energies.shape[-1]
    links = {
        (axis, index): values[index]
        for axis, values in enumerate(link_arrays)
        for index in np.ndindex(values.shape[:-2])
    }
    transport = LinkPath(shape, nstates, links)
    calls = []

    def builder(index):
        index = tuple(index)
        calls.append(index)
        return index, energies[index]

    output = tmp_path / "fields"
    with AbInitioFit(
        grids,
        nstates,
        builder,
        anchor=(1, 1, 1),
        frame=lambda record: record[0],
        energies=lambda record: record[1],
        overlap=transport.between,
        energy_shift=None,
        cache=tmp_path / "points",
    ) as fit:
        fit.run(
            rank=9,
            degrees=2,
            sweeps=3,
            rtol=1.0e-10,
            validation=32,
            start_rank=9,
            kick_rank=1,
            seed=7,
        )
        fit.save(
            output,
            labels=("r1", "r2", "theta"),
            metadata={"molecule": "SO2", "method": "AM1/MECI"},
        )

    assert 0 < len(set(calls)) <= int(np.prod(shape))
    fitted = AbInitioFit.load(output)
    try:
        driver = TTLDR.from_fit(
            fitted,
            keo=_kinetic_terms(grids),
            overlap_rank=16,
            overlap_sweeps=3,
            overlap_rtol=1.0e-10,
            overlap_validation=32,
            cross_start=16,
            cross_kick=1,
            operator_rank=None,
            potential_rank=None,
            seed=11,
        )
        assert driver._hamiltonian is None
        hamiltonian = driver.hamiltonian.to_dense()
        np.testing.assert_allclose(
            hamiltonian,
            hamiltonian.conj().T,
            atol=1.0e-10,
        )

        values = np.zeros(driver.dims, dtype=complex)
        values[1, 1, 1, 2] = 1.0
        state = driver.state(values, physical=False)
        driver.run(
            state,
            dt=1.0e-3,
            steps=1,
            max_bond=12,
            progress=False,
            e_ops=driver.projectors(),
        )
        np.testing.assert_allclose(driver.norms, 1.0, atol=1.0e-10)
        np.testing.assert_allclose(
            np.sum(driver.populations, axis=1),
            1.0,
            atol=1.0e-10,
        )
    finally:
        fitted.close()


@pytest.mark.qchem
@pytest.mark.skipif(
    os.environ.get("PYQED_RUN_QCHEM_TESTS") != "1",
    reason="set PYQED_RUN_QCHEM_TESTS=1 to run live SO2 CASCI",
)
def test_live_so2_casci_jacobi_abinitio_fit_to_ttldr(tmp_path):
    triatom = Triatom(
        so2_body_frame(),
        basis="sto-3g",
        nstates=len(STATE_IDS),
        charge=0,
        spin=0,
        unit="bohr",
        coordinates="jacobi",
        dvr_type=("sine", "sine", "legendre"),
    )
    equilibrium = triatom.valence_to_jacobi(
        REFERENCE_BOND,
        REFERENCE_BOND,
        np.deg2rad(REFERENCE_THETA_DEG),
    )
    widths = (0.05, 0.05, np.deg2rad(1.0))
    triatom.set_dvr(
        domains=[
            [center - width, center + width]
            for center, width in zip(equilibrium, widths)
        ],
        npts=(3, 3, 3),
        dvr_type=("sine", "sine", "legendre"),
    )
    grids = tuple(np.asarray(dvr.x) for dvr in triatom.dvrs)
    calls = []

    def builder(index):
        index = tuple(index)
        calls.append(index)
        coordinates = tuple(
            grid[position] for grid, position in zip(grids, index)
        )
        xyz = triatom._jacobi_to_xyz(
            coordinates,
            triatom.mass,
            triatom._jacobi_atoms,
        )
        point = casci_reference_point(
            [
                [symbol, tuple(position)]
                for symbol, position in zip(("O", "S", "O"), xyz)
            ],
            basis="sto-3g",
            charge=0,
            spin=0,
            unit="bohr",
            ncas=6,
            nelecas=6,
            nstates=len(STATE_IDS),
            scf_tol=1.0e-8,
            scf_max_cycle=80,
            multiplicity=1,
            eri_workers=1,
        )
        require_smooth_active_space(point)
        return index, point.frame(), np.asarray(point.e_tot, dtype=float)

    output = tmp_path / "casci_fields"
    with AbInitioFit(
        grids,
        len(STATE_IDS),
        builder,
        anchor=(1, 1, 1),
        frame=lambda record: record[1],
        energies=lambda record: record[2],
        overlap=lambda left, right: casci_overlap_active(left, right, STATE_IDS),
        energy_shift=None,
        cache=tmp_path / "casci_points",
    ) as fit:
        fit.run(
            rank=9,
            degrees=2,
            sweeps=2,
            rtol=1.0e-10,
            validation=16,
            start_rank=9,
            kick_rank=1,
            seed=17,
        )
        fit.save(
            output,
            labels=("r", "R", "gamma"),
            metadata={"molecule": "SO2", "method": "CASCI(6,6)/STO-3G"},
        )
        indices = tuple(np.ndindex(3, 3, 3))
        records = fit.frames.get_many(indices)
        frames = {index: record[1] for index, record in zip(indices, records)}
        sampled_energies = np.asarray(
            [record[2] for record in records]
        ).reshape(3, 3, 3, len(STATE_IDS))
        gauges = np.asarray(fit.oracle.gauges(indices))
        energy_shift = float(fit.energy_shift)

    assert set(calls) == set(np.ndindex(3, 3, 3))
    raw_links = {}
    for index in np.ndindex(3, 3, 3):
        for axis in range(3):
            if index[axis] + 1 >= 3:
                continue
            right = list(index)
            right[axis] += 1
            raw_links[(axis, index)] = casci_overlap_active(
                frames[index],
                frames[tuple(right)],
                STATE_IDS,
            )

    fitted = AbInitioFit.load(output)
    try:
        driver = TTLDR.from_fit(
            fitted,
            solver=triatom,
            overlap_rank=128,
            overlap_sweeps=3,
            overlap_rtol=1.0e-10,
            overlap_validation=128,
            cross_start=128,
            cross_kick=1,
            operator_rank=None,
            potential_rank=None,
            seed=19,
        )
        assert driver.overlap_info["groups"] >= 3
        kinetic = _jacobi_keo(triatom).to_dense()
        overlap_fixed = _fixed_overlap((3, 3, 3), len(STATE_IDS), raw_links)
        overlap_average = path_overlap((3, 3, 3), raw_links)

        def reference(overlap):
            hamiltonian = full_hamiltonian(
                kinetic,
                overlap,
                sampled_energies,
            )
            shift_correction = float(np.min(sampled_energies)) - energy_shift
            hamiltonian += shift_correction * np.eye(hamiltonian.shape[0])
            return _align_hamiltonian(hamiltonian, gauges)

        reference_fixed = reference(overlap_fixed)
        reference_average = reference(overlap_average)
        fitted_hamiltonian = driver.hamiltonian.to_dense()
        fixed_error = np.linalg.norm(fitted_hamiltonian - reference_fixed)
        fixed_error /= np.linalg.norm(reference_fixed)
        average_error = np.linalg.norm(fitted_hamiltonian - reference_average)
        average_error /= np.linalg.norm(reference_average)
        local = np.asarray(
            [
                gauge.conj().T
                @ np.diag(energy - energy_shift)
                @ gauge
                for gauge, energy in zip(
                    gauges,
                    sampled_energies.reshape(-1, len(STATE_IDS)),
                )
            ]
        )
        exact_potential = np.zeros_like(fitted_hamiltonian)
        for point, block in enumerate(local):
            begin = point * len(STATE_IDS)
            exact_potential[
                begin : begin + len(STATE_IDS),
                begin : begin + len(STATE_IDS),
            ] = block
        fitted_potential = driver.potential.to_dense()
        potential_error = np.linalg.norm(fitted_potential - exact_potential)
        potential_error /= np.linalg.norm(exact_potential)
        link_errors = []
        for (axis, left), raw in raw_links.items():
            right = list(left)
            right[axis] += 1
            left_flat = np.ravel_multi_index(left, (3, 3, 3))
            right_flat = np.ravel_multi_index(tuple(right), (3, 3, 3))
            exact_link = (
                gauges[left_flat].conj().T @ raw @ gauges[right_flat]
            )
            edge = np.asarray(
                [
                    0.5 * (driver.grids[coordinate][left[coordinate]]
                           + driver.grids[coordinate][right[coordinate]])
                    if coordinate == axis
                    else driver.grids[coordinate][left[coordinate]]
                    for coordinate in range(3)
                ]
            )
            fitted_link = np.asarray(driver.links[axis].predict(edge[None, :]))[0]
            link_errors.append(
                np.linalg.norm(fitted_link - exact_link)
                / np.linalg.norm(exact_link)
            )
        link_error = float(np.max(link_errors))
        fitted_kinetic = fitted_hamiltonian - fitted_potential
        exact_kinetic = reference_fixed - exact_potential
        kinetic_error = np.linalg.norm(fitted_kinetic - exact_kinetic)
        kinetic_error /= np.linalg.norm(exact_kinetic)
        print(
            "SO2 Jacobi operator comparison: "
            f"potential={potential_error:.3e}, links={link_error:.3e}, "
            f"kinetic={kinetic_error:.3e}, "
            f"fixed={fixed_error:.3e}, average={average_error:.3e}, "
            f"cross_validation={driver.overlap_info.get('max_validation_error')}"
        )
        assert potential_error < 1.0e-8
        assert link_error < 1.0e-3
        assert kinetic_error < 2.0e-4
        assert fixed_error < 1.0e-4
        assert fixed_error <= average_error + 1.0e-12

        values = np.zeros(driver.dims, dtype=complex)
        values[1, 1, 1, 2] = 1.0
        state = driver.state(values, physical=False)
        dt = 0.1
        steps = 10
        driver.run(
            state,
            dt=dt,
            steps=steps,
            max_bond=9,
            progress=False,
            e_ops=driver.projectors(),
        )
        from scipy.linalg import expm

        initial = values.reshape(-1)
        exact_fixed = expm(-1j * dt * steps * reference_fixed) @ initial
        exact_average = expm(-1j * dt * steps * reference_average) @ initial
        propagated = driver.dense(driver.final_state, physical=False).reshape(-1)
        state_error_fixed = float(np.linalg.norm(propagated - exact_fixed))
        state_error_average = float(np.linalg.norm(propagated - exact_average))
        propagated_raw = np.einsum(
            "ipa,ia->ip",
            gauges,
            propagated.reshape(-1, len(STATE_IDS)),
            optimize=True,
        )
        exact_raw = np.einsum(
            "ipa,ia->ip",
            gauges,
            exact_average.reshape(-1, len(STATE_IDS)),
            optimize=True,
        )
        populations = np.sum(np.abs(propagated_raw) ** 2, axis=0)
        exact_populations = np.sum(np.abs(exact_raw) ** 2, axis=0)
        population_error = float(np.max(np.abs(populations - exact_populations)))
        autocorrelation_error = float(
            abs(np.vdot(initial, propagated) - np.vdot(initial, exact_average))
        )
        print(
            "SO2 Jacobi full-LDR comparison: "
            f"H_fixed={fixed_error:.3e}, H_path_average={average_error:.3e}, "
            f"psi_fixed={state_error_fixed:.3e}, "
            f"psi_path_average={state_error_average:.3e}, "
            f"population={population_error:.3e}, "
            f"autocorrelation={autocorrelation_error:.3e}"
        )
        assert state_error_fixed < 5.0e-5
        assert state_error_fixed <= state_error_average + 1.0e-12
        assert population_error < 1.0e-6
        assert autocorrelation_error < 1.0e-5
        np.testing.assert_allclose(driver.norms, 1.0, atol=1.0e-10)
        np.testing.assert_allclose(
            np.sum(driver.populations, axis=1),
            1.0,
            atol=1.0e-10,
        )
    finally:
        fitted.close()
