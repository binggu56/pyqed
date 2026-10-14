import ast
import io
import os
from types import SimpleNamespace
from pathlib import Path
import shutil
import stat
import subprocess
import textwrap

import pytest


ROOT = Path(__file__).resolve().parents[1]
FEAST_SOURCE = ROOT / "pyqed" / "dvr" / "NuSol" / "lib" / "NuSol_FEAST.c"
NUSOL_SOURCE = ROOT / "pyqed" / "dvr" / "NuSol" / "NuSol.py"


@pytest.fixture(scope="module", params=[False, True], ids=["standard", "sanitized"])
def nusol_feast(tmp_path_factory, request):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("a C compiler is required for the NuSol FEAST security test")

    build_dir = tmp_path_factory.mktemp("nusol-feast")
    (build_dir / "feast.h").write_text(
        "void feastinit(int *parameters);\n", encoding="utf-8"
    )
    (build_dir / "feast_sparse.h").write_text(
        textwrap.dedent(
            """
            void dfeast_scsrgv(
                char *, int *, double *, int *, int *, double *, int *, int *,
                int *, double *, int *, double *, double *, int *, double *,
                double *, int *, double *, int *);
            """
        ),
        encoding="utf-8",
    )
    stub = build_dir / "feast_stub.c"
    stub.write_text(
        textwrap.dedent(
            """
            void feastinit(int *parameters) {
              for (int i = 0; i < 64; ++i) parameters[i] = 0;
            }

            void dfeast_scsrgv(
                char *uplo, int *n, double *sa, int *isa, int *jsa,
                double *sb, int *isb, int *jsb, int *parameters,
                double *epsout, int *loop, double *emin, double *emax,
                int *m0, double *eigenvalues, double *vectors, int *m,
                double *residuals, int *info) {
              (void)uplo; (void)sa; (void)isa; (void)jsa; (void)sb;
              (void)isb; (void)jsb; (void)parameters; (void)emin; (void)emax;
              (void)m0;
              *epsout = 0.0; *loop = 1; *m = 1; *info = 0;
              eigenvalues[0] = 2.0; residuals[0] = 0.0;
              for (int i = 0; i < *n; ++i) vectors[i] = 1.0;
            }
            """
        ),
        encoding="utf-8",
    )
    executable = build_dir / "NuSol_FEAST"
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-Wall",
            "-Wextra",
            "-Werror",
            *(["-fsanitize=address,undefined", "-fno-omit-frame-pointer"] if request.param else []),
            "-I",
            str(build_dir),
            str(FEAST_SOURCE),
            str(stub),
            "-o",
            str(executable),
        ],
        check=True,
    )
    return executable


def _run(executable, matrix, values, vectors, m0="1"):
    return subprocess.run(
        [str(executable), "-1", "4", m0, str(matrix), str(values), str(vectors)],
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_feast_rejects_unbounded_or_malformed_input(nusol_feast, tmp_path):
    matrix = tmp_path / "matrix.dat"
    matrix.write_text("100000001 1 1 0 0 0 1 0 0 0 0 1 0 0\n", encoding="ascii")
    values = tmp_path / "values.dat"
    vectors = tmp_path / "vectors.dat"

    result = _run(nusol_feast, matrix, values, vectors)

    assert result.returncode != 0
    assert "outside safe limits" in result.stderr
    assert not values.exists()
    assert not vectors.exists()


def test_feast_creates_private_outputs_and_rejects_symlinks(nusol_feast, tmp_path):
    matrix = tmp_path / "matrix.dat"
    matrix.write_text("1 1 1 0 0 0 1 0 0 0 0 1 0 0\n1 1 2.0 1.0\n", encoding="ascii")
    values = tmp_path / "values.dat"
    vectors = tmp_path / "vectors.dat"

    result = _run(nusol_feast, matrix, values, vectors)

    assert result.returncode == 0, result.stderr
    assert stat.S_IMODE(values.stat().st_mode) == 0o600
    assert stat.S_IMODE(vectors.stat().st_mode) == 0o600

    symlink = tmp_path / "values-link.dat"
    symlink.symlink_to(values)
    rejected = _run(nusol_feast, matrix, symlink, vectors)
    assert rejected.returncode != 0
    assert values.read_text(encoding="ascii") == "2.000000000000000e+00 0.000000000000000e+00\n"


def test_nusol_launcher_never_interpolates_shell_commands():
    source = NUSOL_SOURCE.read_text(encoding="utf-8")

    assert "shell=True" not in source
    assert 'subprocess.Popen(["ldd", feast_binary]' in source


VALID_MATRIX = "1 1 1 0 0 0 1 0 0 0 0 1 0 0\n1 1 2.0 1.0\n"


@pytest.mark.parametrize("content", [
    "", "1 1 invalid", VALID_MATRIX.replace("1 1 1", "0 1 1", 1),
    VALID_MATRIX.replace("1 1 1", "9" * 100 + " 1 1", 1),
    VALID_MATRIX.replace("1 1 1", "9223372036854775808 1 1", 1),
    VALID_MATRIX.replace("1 1 1", "1 100000001 1", 1),
    VALID_MATRIX.replace("0 1 0", "nan 1 0", 1),
    VALID_MATRIX.replace("2.0", "nan"), VALID_MATRIX.replace("1.0", "inf"),
    VALID_MATRIX.replace("1 1 2.0", "0 1 2.0"),
    VALID_MATRIX.replace("1 1 2.0", "1 2 2.0"),
    VALID_MATRIX.rsplit("1.0", 1)[0], VALID_MATRIX + "1 1 4.0 1.0\n",
    "2 2 2 0 0 0 1 0 0 0 0 1 0 0\n2 2 2.0 1.0\n1 1 2.0 1.0\n",
    "1 2 1 0 0 0 1 0 0 0 0 1 0 0\n1 1 2.0 1.0\n1 1 2.0 1.0\n",
])
def test_invalid_matrix_does_not_touch_outputs(nusol_feast, tmp_path, content):
    matrix, values, vectors = [tmp_path / name for name in ("matrix", "values", "vectors")]
    matrix.write_text(content)
    values.write_text("keep values")
    vectors.write_text("keep vectors")
    result = _run(nusol_feast, matrix, values, vectors)
    assert result.returncode > 0, result.stderr
    assert "Sanitizer" not in result.stderr
    assert values.read_text() == "keep values"
    assert vectors.read_text() == "keep vectors"


@pytest.mark.parametrize("arguments", [[], ["nan", "4", "1"], ["-1", "inf", "1"],
    ["4", "-1", "1"], ["-1", "4", "0"], ["-1", "4", "2"],
    ["-1", "4", "999999999999999999999999999999999999999999"]])
def test_invalid_cli_arguments(nusol_feast, tmp_path, arguments):
    matrix, values, vectors = [tmp_path / name for name in ("matrix", "values", "vectors")]
    matrix.write_text(VALID_MATRIX)
    command = [str(nusol_feast), *arguments]
    if arguments: command.extend(map(str, (matrix, values, vectors)))
    result = subprocess.run(command, capture_output=True, text=True, timeout=10)
    assert result.returncode > 0, result.stderr
    assert "Sanitizer" not in result.stderr
    assert not values.exists() and not vectors.exists()


def test_existing_outputs_become_private(nusol_feast, tmp_path):
    matrix, values, vectors = [tmp_path / name for name in ("matrix", "values", "vectors")]
    matrix.write_text(VALID_MATRIX)
    for path in (values, vectors):
        path.write_text("old output that must be truncated")
        path.chmod(0o644)
    result = _run(nusol_feast, matrix, values, vectors)
    assert result.returncode == 0, result.stderr
    assert values.read_text() == "2.000000000000000e+00 0.000000000000000e+00\n"
    for path in (values, vectors):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert "old output" not in path.read_text()


@pytest.mark.parametrize("target", ["input", "other_output", "hardlink", "fifo"])
def test_unsafe_output_does_not_truncate_existing_files(nusol_feast, tmp_path, target):
    matrix, values, vectors = [tmp_path / name for name in ("matrix", "values", "vectors")]
    matrix.write_text(VALID_MATRIX)
    values.write_text("keep values")
    if target == "input": vectors = matrix
    elif target == "other_output": vectors = values
    elif target == "hardlink": os.link(values, vectors)
    else: os.mkfifo(vectors)
    result = _run(nusol_feast, matrix, values, vectors)
    assert result.returncode > 0, result.stderr
    assert matrix.read_text() == VALID_MATRIX
    assert values.read_text() == "keep values"


def test_fifo_input_does_not_block(nusol_feast, tmp_path):
    matrix = tmp_path / "matrix"
    os.mkfifo(matrix)
    result = _run(nusol_feast, matrix, tmp_path / "values", tmp_path / "vectors")
    assert result.returncode > 0, result.stderr


def test_launcher_passes_metacharacters_as_literal_arguments(tmp_path):
    calls = []
    executable = tmp_path / "feast;touch INJECTED" / "NuSol_FEAST"
    executable.parent.mkdir()
    executable.touch(mode=0o700)
    cfg = SimpleNamespace(METHOD="numerov", NDIM=1, NGRIDX=1, h=1,
        USE_FEAST="true", FEAST_PATH=str(executable.parent), FEAST_E_MIN=-1,
        FEAST_E_MAX=4, FEAST_M=1, FEAST_MATRIX_OUT_PATH="matrix;touch INJECTED",
        EIGENVALUES_OUT="values with spaces", EIGENVECTORS_OUT="vectors$(echo bad)",
        read=lambda _: None)
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(communicate=lambda: (b"libraries loaded", None),
            returncode=0, stdout=io.BytesIO(b"done\n"), wait=lambda: 0)
    node = next(node for node in ast.parse(NUSOL_SOURCE.read_text()).body
                if isinstance(node, ast.ClassDef) and node.name == "numerov")
    namespace = {"os": os, "subprocess": SimpleNamespace(Popen=popen, PIPE=-1, STDOUT=-2),
        "SafeConfigParser": lambda: cfg, "NuSol_cfg_obj": lambda value: value,
        "NuSol_matrices": lambda _: SimpleNamespace(Numerov_Matrix_1D=lambda: (None, None))}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(NUSOL_SOURCE), "exec"), namespace)
    namespace["numerov"]("unused.cfg")
    assert calls[0][0] == ["ldd", str(executable)]
    assert calls[1][0] == [str(executable), "-1", "4", "1", cfg.FEAST_MATRIX_OUT_PATH,
                           cfg.EIGENVALUES_OUT, cfg.EIGENVECTORS_OUT]
    assert all(not kwargs.get("shell", False) for _, kwargs in calls)
