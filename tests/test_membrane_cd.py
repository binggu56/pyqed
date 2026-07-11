import numpy as np

from pyqed.md import Atoms, MembraneEmbeddingSnapshot, membrane_embedding_snapshot
from pyqed.qchem import MembraneCD


def _embedded_h2o2_snapshot(charge_shift=0.0):
    atoms = Atoms(
        [
            ["O", (0.000, 0.000, 0.000)],
            ["O", (2.740, 0.000, 0.000)],
            ["H", (-0.850, 1.436, 0.000)],
            ["H", (3.590, 1.436, 1.134)],
            ["He", (7.0 + charge_shift, 0.0, 0.0)],
            ["He", (-5.0, 2.0, 1.0)],
        ],
        cell=[20.0, 20.0, 20.0],
        pbc=True,
    )
    atoms.set_array("charges", [0.0, 0.0, 0.0, 0.0, -0.2, 0.1], float, ())
    atoms.set_array("leaflets", [0, 0, 0, 0, 1, -1], int, ())
    return atoms


def test_membrane_cd_tda_workflow_runs_and_averages_spectrum():
    snapshots = [_embedded_h2o2_snapshot(0.0), _embedded_h2o2_snapshot(0.2)]

    workflow = MembraneCD(
        snapshots,
        qm_indices=[0, 1, 2, 3],
        method="tda",
        nstates=1,
        basis="sto3g",
        cutoff=9.0,
        embedding_pbc="nearest",
        cap_charge_distance=1.0,
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()
    x, signal = result.spectrum(width=0.4, units="ev")

    assert len(result.frames) == 2
    assert result.method == "tda"
    assert result.depths.shape == (2,)
    for frame in result.frames:
        assert frame.cd_result.excitation_energies.shape == (1,)
        assert frame.cd_result.rotatory_strengths.shape == (1,)
        assert frame.snapshot.charge_coords.shape[1] == 3
        assert np.all(np.isfinite(frame.cd_result.rotatory_strengths))
    assert x.shape == signal.shape
    assert x.size == 1000
    assert np.all(np.isfinite(signal))


def test_membrane_cd_accepts_preextracted_embedding_snapshot():
    atoms = _embedded_h2o2_snapshot()
    snapshot = membrane_embedding_snapshot(
        atoms,
        qm_indices=[0, 1, 2, 3],
        cutoff=9.0,
        embedding_pbc="nearest",
    )

    workflow = MembraneCD(
        [snapshot],
        qm_indices=[0, 1, 2, 3],
        atom_symbols=["O", "O", "H", "H"],
        method="tda",
        nstates=1,
        basis="sto3g",
        mf_run_kwargs={"verbose": 0, "max_cycle": 100},
    )
    result = workflow.run()

    assert isinstance(result.frames[0].snapshot, MembraneEmbeddingSnapshot)
    assert result.frames[0].atom_symbols == ("O", "O", "H", "H")
    assert result.frames[0].cd_result.excitation_energies.shape == (1,)
