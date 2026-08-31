import numpy as np
import pytest

from pyqed.dvr import DVR, SineDVR
from pyqed.ldr import AbInitioFit, Coord, LDR
from pyqed.ldr import keo as keo_tools
from pyqed.ldr.kinetic import linked
from pyqed.namd import TNLDR


class _Frame:
    def __init__(self, vectors):
        self.vectors = np.asarray(vectors, dtype=complex)

    def overlap(self, other):
        return self.vectors.conj().T @ other.vectors


class _Result:
    def __init__(self, q):
        angle = 0.12 * float(q)
        self.e_tot = np.asarray((0.03 * q**2, 0.4 + 0.05 * q))
        self._frame = _Frame(
            ((np.cos(angle), -np.sin(angle)),
             (np.sin(angle), np.cos(angle)))
        )

    def frame(self):
        return self._frame


class _Scanner:
    def __init__(self, calls):
        self.calls = calls

    def __call__(self, geometry):
        q = float(np.asarray(geometry).reshape(-1)[0])
        self.calls.append(q)
        return _Result(q)


class _Electronic(_Result):
    def __init__(self):
        super().__init__(0.0)
        self.nstates = 2
        self.calls = []

    def as_scanner(self, nstates=None):
        assert nstates == 2
        return _Scanner(self.calls)


def _grid_and_links(nstates=2):
    axes = (SineDVR(-1.0, 1.0, 3), SineDVR(-0.8, 1.2, 3))
    grid = DVR.from_axes(axes)
    links = {}
    for axis in range(2):
        edge_shape = list(grid.shape)
        edge_shape[axis] -= 1
        for index in np.ndindex(*edge_shape):
            links[axis, index] = np.array(
                [[0.94, 0.03], [-0.02, 0.91]], dtype=complex
            )[:nstates, :nstates]
    return axes, grid, links


def test_tnldr_from_ldr_uses_raw_links_without_polar_projection(monkeypatch):
    axes, grid, links = _grid_and_links()
    energies = np.zeros((*grid.shape, 2))
    energies[..., 0] = 0.03
    energies[..., 1] = 0.08
    ldr = LDR(grid, 2, energies=energies, links=links)

    def reject_polar(_value):
        raise AssertionError("raw-link TTLDR must not polar-project links")

    monkeypatch.setattr("pyqed.namd.ttldr.polar_unitary", reject_polar)
    driver = TNLDR.from_ldr(ldr)

    kinetic = sum(
        np.kron(
            axes[axis].t(), np.eye(grid.shape[1 - axis])
        )
        if axis == 0
        else np.kron(np.eye(grid.shape[0]), axes[axis].t())
        for axis in range(2)
    )
    expected = linked(
        kinetic,
        grid.shape,
        links,
        nstates=2,
        symmetrize=True,
    ).toarray()
    expected += np.diag(energies.reshape(-1))

    np.testing.assert_allclose(driver.hamiltonian.to_dense(), expected, atol=1.0e-11)
    assert driver.overlap_info["raw_links"]
    assert not driver.overlap_info["polar_link_projection"]
    assert driver.overlap_info["selection"] == {
        "requested": "auto",
        "resolved": "dense",
        "largest_fiber_elements": 108,
        "dense_max_elements": 250_000,
    }
    with pytest.raises(ValueError, match="preserves raw links"):
        TNLDR.from_ldr(ldr, gauge_sync=True)


def test_tnldr_from_ldr_accepts_curvilinear_metric_and_pseudopotential():
    axes, grid, links = _grid_and_links()
    identity_links = {key: np.eye(2) for key in links}
    energies = np.zeros((*grid.shape, 2))
    energies[..., 0] = 0.02
    energies[..., 1] = 0.07
    metric = np.empty((*grid.shape, 2, 2))
    metric[...] = np.array([[1.2, 0.25], [0.25, 0.8]])
    q0, q1 = np.meshgrid(axes[0].x, axes[1].x, indexing="ij")
    pseudopotential = 0.01 * (q0**2 + 0.5 * q1**2)
    curvilinear_keo = keo_tools.podolsky(
        metric, pseudopotential
    )
    ldr = LDR(
        grid,
        2,
        energies=energies,
        links=identity_links,
        keo=curvilinear_keo,
    )
    with pytest.raises(NotImplementedError, match="TNLDR.from_ldr"):
        ldr.kinetic_operator()
    driver = TNLDR.from_ldr(ldr)

    nuclear = ldr.keo.to_dense()
    expected = np.kron(nuclear, np.eye(2))
    expected += np.diag(energies.reshape(-1))

    np.testing.assert_allclose(driver.hamiltonian.to_dense(), expected, atol=1.0e-11)
    assert driver.overlap_info["backend"] == "raw-link-labelled-mpo"
    assert driver.overlap_info["fields"][0]["active"] == (0, 1)
    assert driver.overlap_info["selection"]["resolved"] == "dense"

    packet = np.zeros(driver.dims, dtype=complex)
    packet[1, 1, 0] = 1.0
    driver.run(
        driver.state(packet, max_rank=16),
        dt=1.0e-3,
        steps=1,
        max_bond=32,
        progress=False,
    )
    np.testing.assert_allclose(driver.norms, 1.0, atol=1.0e-11)


def test_tnldr_builds_mpos_directly_from_an_independent_sampling_grid(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("PYQED_ELECTRONIC_CACHE_DIR", str(tmp_path / "electronic"))
    dynamics = DVR.from_axes((SineDVR(-0.9, 0.9, 3),))
    electronic = _Electronic()
    coord = Coord(to_cartesian=lambda q: q, bounds=((-0.9, 0.9),))
    fit = AbInitioFit(
        electronic,
        coord=coord,
        states=(0, 1),
        fit_options={"rank": 4, "sweeps": 3, "validation": 8, "seed": 3},
    ).build()

    assert fit.config["gauge"] == "anchor-procrustes"
    assert fit.config["unitarize_links"] is False

    driver = TNLDR(
        fit,
        grid=dynamics,
        coord=coord,
        keo=keo_tools.product(dynamics.axes),
        overlap_rank=8,
        operator_rank=None,
    ).build()

    assert driver.electronic.success
    assert driver.electronic.feature is not None
    assert driver.electronic.links is None
    assert driver.dims == (*dynamics.shape, 2)
    assert driver.potential_info["backend"] == "functional-tt"
    assert not driver.overlap_info["materialized_link_grid"]
    assert not driver.overlap_info["materialized_overlap_fiber"]
    assert driver.overlap_info["action"] == "linked-product-approximation"
    assert not driver.overlap_info["unitarized"]
    expected_sampling = {
        "candidate_shape": (9,),
        "dynamics_shape": (3,),
        "representation": "adaptive-sync",
        "initial_geometries": 7,
        "sampled_geometries": 9,
        "maximum_geometries": 9,
        "direct_mpo": True,
    }
    assert all(
        driver.overlap_info["electronic_sampling"][key] == value
        for key, value in expected_sampling.items()
    )
    assert driver.database_info["writes"] == 9
    assert electronic.calls
    assert all(
        np.min(np.abs(driver.sampling_grid[0] - value)) < 1.0e-13
        for value in electronic.calls
    )
    assert any(
        np.min(np.abs(dynamics.x[0] - value)) > 1.0e-8
        for value in electronic.calls
    )

    reference_grid = DVR.from_axes((SineDVR(-0.9, 0.9, 5),))
    reference = fit.direct_product(
        reference_grid,
        keo=keo_tools.product(reference_grid.axes),
    )
    assert reference.shape == (5,)
    assert reference.energies.shape == (5, 2)
    assert reference.overlaps is None
    assert len(reference.links) == 4
    assert reference.direct_product_info["overlap_representation"] == (
        "nearest-link-lpa"
    )
    assert reference.direct_product_info["action"] == (
        "linked-product-approximation"
    )
    assert reference.direct_product_info["geometries"] == 5
    assert reference.direct_product_info["gauge"] == "anchor-procrustes"
    assert reference.procrustes_gauges.shape == (5, 2, 2)
    np.testing.assert_allclose(
        reference.procrustes_gauges.conj().swapaxes(-1, -2)
        @ reference.procrustes_gauges,
        np.broadcast_to(np.eye(2), (5, 2, 2)),
        atol=1.0e-12,
    )
    assert reference.database_path == fit.database_path

    cached_electronic = _Electronic()
    cached_fit = AbInitioFit(
        cached_electronic,
        coord=coord,
        states=(0, 1),
        fit_options={"rank": 4, "sweeps": 3, "validation": 8, "seed": 3},
    ).build()
    cached = TNLDR(
        cached_fit,
        grid=dynamics,
        coord=coord,
        keo=keo_tools.product(dynamics.axes),
        overlap_rank=8,
        operator_rank=None,
    ).build()

    assert cached_electronic.calls == []
    assert cached.database_path == driver.database_path
    assert cached.database_info["hits"] > 0
    assert cached.database_info["writes"] == 0


def test_native_database_stores_all_roots_and_reuses_state_views(tmp_path, monkeypatch):
    monkeypatch.setenv("PYQED_ELECTRONIC_CACHE_DIR", str(tmp_path / "electronic"))

    class Result:
        def __init__(self, q):
            self.e_tot = np.asarray((q, 1.0 + q, 2.0 + q))
            self._frame = _Frame(np.eye(3))

        def frame(self):
            return self._frame

    class Scanner:
        def __init__(self, calls):
            self.calls = calls

        def __call__(self, geometry):
            q = float(np.asarray(geometry).reshape(-1)[0])
            self.calls.append(q)
            return Result(q)

    class Electronic(Result):
        def __init__(self):
            super().__init__(0.0)
            self.nstates = 3
            self.calls = []

        def as_scanner(self, nstates=None):
            assert nstates == 3
            return Scanner(self.calls)

    coord = Coord(to_cartesian=lambda q: q, bounds=((-1.0, 1.0),))
    first_electronic = Electronic()
    first = AbInitioFit(
        first_electronic, coord=coord, states=(0, 1), nroots=3
    )
    with first:
        first_record = first.frames.get_many(((4,),))[0]
    np.testing.assert_allclose(first.energies_of(first_record), (0.0, 1.0))
    assert first_electronic.calls == [0.0]
    assert "selected_states" not in first.protocol

    second_electronic = Electronic()
    second = AbInitioFit(
        second_electronic, coord=coord, states=(1, 2), nroots=3
    )
    with second:
        second_record = second.frames.get_many(((4,),))[0]
    np.testing.assert_allclose(second.energies_of(second_record), (1.0, 2.0))
    assert second.protocol == first.protocol
    assert second_electronic.calls == []
    assert second.frames.stats["database_hits"] == 1
