import base64
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest

import pyqed
from pyqed.qchem import Molecule
from pyqed.units import au2angstrom
from pyqed.visualization import (
    MoleculeView,
    ScalarField3D,
    SceneView,
    VolumeView,
    view,
)


class TinyMolecule:
    def atom_symbols(self):
        return ["H", "F"]

    def atom_coords(self):
        return np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])


class TinyHessian:
    mol = TinyMolecule()

    def vibrational_analysis(self):
        return {
            "freq_cm1": np.array([1234.5]),
            "modes": np.array([[[0.0, 0.0, -2.0], [0.0, 0.0, 1.0]]]),
            "reduced_mass_amu": np.array([1.2]),
        }


def fragment_parameters(result):
    return parse_qs(urlsplit(result.url).fragment)


def test_view_encodes_bohr_geometry_for_the_web_viewer():
    result = view(TinyMolecule(), labels=True, open_browser=False)

    assert isinstance(result, MoleculeView)
    parameters = fragment_parameters(result)
    assert parameters["representation"] == ["ball-stick"]
    assert parameters["labels"] == ["1"]
    assert f"{2.0 * au2angstrom:.10f}" in parameters["xyz"][0]
    assert result.url.startswith("https://pyqed.org/viewer#")


def test_view_opens_a_browser_when_requested(monkeypatch):
    opened = []
    monkeypatch.setattr("webbrowser.open_new_tab", lambda url: opened.append(url) or True)

    result = view(TinyMolecule(), representation="wireframe", open_browser=True)

    assert opened == [result.url]
    assert fragment_parameters(result)["representation"] == ["wireframe"]


def test_view_provides_an_inline_notebook_representation():
    result = view(
        TinyMolecule(),
        coordinates_unit="angstrom",
        width=720,
        height=480,
        open_browser=False,
    )

    html = result._repr_html_()
    assert "<iframe" in html
    assert 'width="720"' in html
    assert 'height="480"' in html
    assert "PyQED molecular viewer" in html
    assert "allow-scripts" in html


def test_view_hessian_builds_browser_normal_mode_animation():
    default_result = view(TinyHessian(), mode=0, open_browser=False)
    assert default_result.payload()["vibration"]["interval"] == 30

    result = view(
        TinyHessian(),
        mode=0,
        amplitude=0.25,
        frames=32,
        interval=45,
        open_browser=False,
    )

    payload = result.payload()
    vibration = payload["vibration"]
    assert result.url == "https://pyqed.org/viewer"
    assert result.title == "Mode 0: 1234.5 cm^-1"
    assert vibration["mode_index"] == 0
    assert vibration["frequency_cm1"] == pytest.approx(1234.5)
    assert vibration["frames"] == 32
    assert vibration["interval"] == 45
    assert vibration["amplitude_angstrom"] == pytest.approx(0.25)
    displacement = np.asarray(vibration["displacements"])
    assert displacement.shape == (2, 3)
    assert np.max(np.linalg.norm(displacement, axis=1)) == pytest.approx(0.25)
    assert "postMessage" in result._repr_html_()


def test_view_hessian_rejects_invalid_or_unavailable_modes():
    with pytest.raises(IndexError, match="outside"):
        view(TinyHessian(), mode=1, open_browser=False)
    with pytest.raises(TypeError, match="integer"):
        view(TinyHessian(), mode=True, open_browser=False)
    with pytest.raises(TypeError, match="completed Hessian"):
        view(TinyMolecule(), mode=0, open_browser=False)

    class PendingHessian(TinyHessian):
        def vibrational_analysis(self):
            raise ValueError("run first")

    with pytest.raises(ValueError, match="run the Hessian"):
        view(PendingHessian(), mode=0, open_browser=False)


def test_top_level_and_molecule_convenience_apis():
    mol = Molecule(atom="H 0 0 0; H 0 0 0.74", unit="angstrom")

    direct = pyqed.view(mol, open_browser=False)
    method = mol.view(open_browser=False)

    assert fragment_parameters(direct)["xyz"] == fragment_parameters(method)["xyz"]
    assert "0.7400000000" in fragment_parameters(method)["xyz"][0]
    assert pyqed.ScalarField3D is ScalarField3D
    assert pyqed.SceneView is SceneView
    assert pyqed.VolumeView is VolumeView


@pytest.mark.parametrize("representation", ["sticks", "cartoon", "surface"])
def test_view_rejects_unknown_representations(representation):
    with pytest.raises(ValueError, match="representation"):
        view(TinyMolecule(), representation=representation, open_browser=False)


def test_view_validates_molecule_shape_and_dimensions():
    class BadMolecule(TinyMolecule):
        def atom_coords(self):
            return np.zeros((2, 2))

    with pytest.raises(ValueError, match="shape"):
        view(BadMolecule(), open_browser=False)
    with pytest.raises(ValueError, match="at least 240"):
        view(TinyMolecule(), width=100, open_browser=False)


@pytest.fixture(scope="module")
def h2_reference():
    from pyqed.qchem.hf.rhf import RHF

    mol = Molecule(atom="H 0 0 0; H 0 0 1.4", unit="bohr", basis="sto-3g")
    mol.build()
    return RHF(mol).run()


def test_orbital_all_and_frontier_selectors_build_one_scene(h2_reference):
    all_orbitals = view(h2_reference, orbital="all", nx=4, open_browser=False)
    frontier = view(
        h2_reference,
        orbital=["homo", "lumo"],
        nx=4,
        open_browser=False,
    )

    assert isinstance(all_orbitals, VolumeView)
    assert isinstance(all_orbitals, SceneView)
    assert [field.name for field in all_orbitals.fields] == ["mo-1", "mo-2"]
    assert [field.name for field in frontier.fields] == ["mo-1", "mo-2"]
    assert all(field.kind == "orbital" for field in all_orbitals.fields)
    assert all(field.values.shape == (4, 4, 4) for field in all_orbitals.fields)
    assert all_orbitals.url == "https://pyqed.org/viewer"
    assert urlsplit(all_orbitals.url).fragment == ""


def test_custom_orbital_coefficients_and_density_matrix(h2_reference):
    coeff = np.asarray(h2_reference.mo_coeff[:, 0])
    orbital = view(h2_reference, coeff=coeff, nx=3, open_browser=False)
    density = view(
        h2_reference,
        density=h2_reference.make_rdm1(),
        nx=3,
        open_browser=False,
    )

    assert [field.name for field in orbital.fields] == ["custom-orbital"]
    assert orbital.fields[0].metadata["source"] == "custom-coefficients"
    assert density.fields[0].name == "custom-density"
    assert density.fields[0].kind == "electron-density"
    assert np.min(density.fields[0].values) >= 0.0


def test_named_state_density_matrices_preserve_transition_and_difference_kinds(
    h2_reference,
):
    dm = h2_reference.make_rdm1()
    scene = view(
        h2_reference,
        density={
            "S1 transition density": dm,
            "S1 - S0 difference density": -dm,
        },
        nx=3,
        open_browser=False,
    )

    assert [field.kind for field in scene.fields] == [
        "transition-density",
        "difference-density",
    ]


def test_uhf_density_all_exposes_total_alpha_beta_and_spin(h2_reference):
    class FakeUHF:
        mol = h2_reference.mol
        mo_coeff = (h2_reference.mo_coeff, h2_reference.mo_coeff)
        mo_occ = (np.array([1.0, 0.0]), np.array([0.0, 1.0]))
        mo_energy = (h2_reference.mo_energy, h2_reference.mo_energy)

        def make_rdm1(self):
            ca, cb = self.mo_coeff
            return np.array(
                [
                    ca @ np.diag(self.mo_occ[0]) @ ca.T,
                    cb @ np.diag(self.mo_occ[1]) @ cb.T,
                ]
            )

    scene = view(FakeUHF(), density="all", nx=3, open_browser=False)

    assert [field.name for field in scene.fields] == [
        "electron-density",
        "alpha-density",
        "beta-density",
        "spin-density",
    ]
    np.testing.assert_allclose(
        scene.fields[0].values,
        scene.fields[1].values + scene.fields[2].values,
        rtol=1.0e-6,
        atol=1.0e-12,
    )
    np.testing.assert_allclose(
        scene.fields[3].values,
        scene.fields[1].values - scene.fields[2].values,
        rtol=1.0e-6,
        atol=1.0e-12,
    )


def test_esp_uses_matching_density_surface_and_labels_fft_fallback(h2_reference):
    scene = view(
        h2_reference,
        esp=True,
        esp_method="fft",
        nx=4,
        open_browser=False,
    )

    density, esp = scene.fields
    assert density.name == "electron-density"
    assert esp.kind == "esp"
    assert esp.surface_field == density.name
    assert esp.shape == density.shape
    np.testing.assert_allclose(esp.origin, density.origin)
    np.testing.assert_allclose(esp.axes, density.axes)
    assert esp.metadata == {
        "method": "fft-convolution",
        "approximate": True,
        "units": "hartree/e",
    }
    assert np.isfinite(esp.values).all()


def test_custom_esp_density_is_not_replaced_by_requested_default_density(h2_reference):
    zero_dm = np.zeros_like(h2_reference.make_rdm1())
    scene = view(
        h2_reference,
        density=True,
        esp=zero_dm,
        esp_method="fft",
        nx=4,
        open_browser=False,
    )
    default_scene = view(
        h2_reference,
        density=True,
        esp=True,
        esp_method="fft",
        nx=4,
        open_browser=False,
    )

    assert [field.name for field in scene.fields] == [
        "electron-density",
        "esp-surface-density",
        "esp",
    ]
    assert scene.fields[-1].surface_field == "esp-surface-density"
    assert not np.allclose(scene.fields[-1].values, default_scene.fields[-1].values)


def test_nto_all_builds_hole_and_particle_fields(h2_reference):
    class TinyTD:
        _scf = h2_reference
        e = np.array([0.4, 0.7])
        xy = [
            (np.array([[1.0]]), 0),
            (np.array([[-1.0]]), np.zeros((1, 1))),
        ]

    scene = view(TinyTD(), nto="all", nx=3, open_browser=False)

    assert [field.name for field in scene.fields] == [
        "state-1-nto-1-hole",
        "state-1-nto-1-particle",
        "state-2-nto-1-hole",
        "state-2-nto-1-particle",
    ]
    assert scene.fields[0].metadata["role"] == "hole"
    assert scene.fields[1].metadata["role"] == "particle"
    assert scene.fields[0].metadata["weight"] == pytest.approx(1.0)


def test_nto_rejects_legacy_casida_vectors(h2_reference):
    class LegacyCasidaTD:
        __module__ = "pyqed.qchem.tdscf.tdhf"
        _scf = h2_reference
        e = np.array([0.4])
        xy = np.ones((1, 1))

    with pytest.raises(NotImplementedError, match="Casida eigenvectors"):
        view(LegacyCasidaTD(), nto="all", nx=3, open_browser=False)


def test_scalar_field_validation_payload_and_postmessage_transport():
    field = ScalarField3D(
        "test-field",
        np.arange(8.0).reshape(2, 2, 2),
        origin=(0.0, 0.0, 0.0),
        axes=np.eye(3),
        kind="generic",
    )
    scene = VolumeView(xyz=None, fields=(field,), active_field="test-field")
    payload = scene.payload()
    html = scene._repr_html_()

    assert payload["kind"] == "pyqed-scene"
    assert payload["fields"][0]["value_encoding"] == "float32-le-base64"
    decoded = np.frombuffer(
        base64.b64decode(payload["fields"][0]["values"]),
        dtype="<f4",
    )
    np.testing.assert_array_equal(decoded, np.arange(8.0))
    assert payload["fields"][0]["shape"] == [2, 2, 2]
    assert "postMessage" in html
    assert "pyqed:viewer-ready" in html
    assert "https://pyqed.org" in html
    assert "[0.0,1.0,2.0" not in html
    assert "#" not in scene.url

    with pytest.raises(ValueError, match="shape"):
        ScalarField3D("bad", np.zeros((2, 2)), (0, 0, 0), np.eye(3))
    with pytest.raises(ValueError, match="linearly independent"):
        ScalarField3D("bad", np.zeros((2, 2, 2)), (0, 0, 0), np.zeros((3, 3)))


def test_scalar_field_owns_data_and_enforces_browser_contract():
    values = np.arange(8.0).reshape(2, 2, 2)
    metadata = {"states": [0, 1]}
    field = ScalarField3D(
        "owned-field",
        values,
        origin=(0, 0, 0),
        axes=np.eye(3),
        metadata=metadata,
    )
    values[:] = -1
    metadata["states"].append(2)

    np.testing.assert_array_equal(field.values, np.arange(8.0).reshape(2, 2, 2))
    assert field.metadata["states"] == (0, 1)
    with pytest.raises(ValueError, match="at most 80"):
        ScalarField3D("x" * 81, np.zeros((2, 2, 2)), (0, 0, 0), np.eye(3))
    with pytest.raises(ValueError, match="one or two"):
        ScalarField3D(
            "bad-colors",
            np.zeros((2, 2, 2)),
            (0, 0, 0),
            np.eye(3),
            colors=("red", "blue", "green"),
        )


def test_cube_path_round_trip_uses_c_order_and_no_ase_dependency(tmp_path, h2_reference):
    from pyqed.qchem.tools.cubegen import Cube

    cube = Cube(h2_reference.mol, nx=3, ny=4, nz=5, margin=1.0)
    values = np.arange(60.0).reshape(3, 4, 5)
    path = tmp_path / "ordered.cube"
    cube.write(values, path, comment="ordered electron density")

    scene = view(path, open_browser=False)
    field = scene.fields[0]

    assert field.shape == (3, 4, 5)
    np.testing.assert_allclose(field.values, values)
    np.testing.assert_allclose(field.origin, cube.origin * au2angstrom, atol=1e-6)
    np.testing.assert_allclose(
        field.axes,
        np.diag(cube.spacing) * au2angstrom,
        atol=1e-6,
    )
    assert field.kind == "electron-density"


def test_cube_uses_both_header_lines_and_can_load_all_datasets(tmp_path):
    path = tmp_path / "orbitals.cube"
    path.write_text(
        "Water field\n"
        "MO test\n"
        "-1 0 0 0\n"
        "2 1 0 0\n"
        "2 0 1 0\n"
        "2 0 0 1\n"
        "1 1 0 0 0\n"
        "2 5 8\n"
        "0 10 1 11 2 12 3 13 4 14 5 15 6 16 7 17\n",
        encoding="utf-8",
    )

    scene = view(path, dataset="all", open_browser=False)

    assert [field.label for field in scene.fields] == ["Orbital 5", "Orbital 8"]
    assert all(field.kind == "orbital" for field in scene.fields)
    np.testing.assert_array_equal(scene.fields[0].values.ravel(), np.arange(8.0))
    np.testing.assert_array_equal(scene.fields[1].values.ravel(), np.arange(10.0, 18.0))
