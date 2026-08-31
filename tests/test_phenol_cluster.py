import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from examples.namd.phenol_gp_cluster_portabilize import portabilize


def _metadata(path, spec):
    path.mkdir(parents=True)
    (path / "metadata.json").write_text(
        json.dumps({"version": 1, "spec": spec, "files": {}}) + "\n"
    )


def test_portabilize_rewrites_dependency_chain_and_checkpoint(tmp_path):
    model = tmp_path / "model.pt"
    radial = tmp_path / "radial.npz"
    initial = tmp_path / "initial.npz"
    for path, value in ((model, b"m"), (radial, b"r"), (initial, b"i")):
        path.write_bytes(value)
    keo = tmp_path / "keo"
    field = tmp_path / "field"
    residual = tmp_path / "residual"
    gp = tmp_path / "gp"
    ngp = tmp_path / "ngp"
    _metadata(keo, {})
    _metadata(field, {"checkpoint": {}, "radial_correction": {}})
    _metadata(
        residual,
        {
            "base_field": {},
            "checkpoint": {},
            "radial_correction": {},
            "initial_state": {},
        },
    )
    _metadata(gp, {"field": {}, "keo": {}})
    _metadata(ngp, {"field": {}, "keo": {}})
    branch = tmp_path / "gp_checkpoint.npz"
    np.savez_compressed(
        branch,
        config_json=np.asarray(
            json.dumps(
                {
                    "field": {},
                    "potential_residual": {},
                    "keo": {},
                    "initial_state": {},
                }
            )
        ),
        step=np.asarray(20),
    )
    manifest = tmp_path / "manifest.json"

    portabilize(
        SimpleNamespace(
            checkpoint=model,
            radial_correction=radial,
            initial_state=initial,
            keo_cache=keo,
            field_cache=field,
            residual_cache=residual,
            gp_operator_cache=gp,
            ngp_operator_cache=ngp,
            branch_checkpoint=[branch],
            manifest=manifest,
        )
    )

    field_spec = json.loads((field / "metadata.json").read_text())["spec"]
    residual_spec = json.loads((residual / "metadata.json").read_text())["spec"]
    gp_spec = json.loads((gp / "metadata.json").read_text())["spec"]
    assert field_spec["checkpoint"]["sha256"]
    assert residual_spec["base_field"]["path"] == str(
        (field / "metadata.json").resolve()
    )
    assert gp_spec["field"]["sha256"]
    with np.load(branch, allow_pickle=False) as saved:
        config = json.loads(str(saved["config_json"]))
    assert config["potential_residual"]["sha256"]
    assert manifest.is_file()


def test_phenol_cluster_batch_is_a_two_branch_fast_native_array():
    text = Path("cluster/slurm/phenol_gp_ngp_5d_50fs.sbatch").read_text()
    assert "#SBATCH --array=0-1%2" in text
    assert "--mode \"$MODE\"" in text
    assert "CPP_TDVP_HAS_BLAS" in text
    assert "--time-fs 50" in text
    assert "--state-rank 32" in text
