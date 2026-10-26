"""Keep unfinished drafts and unresolved merges out of the GW package."""
import ast
from pathlib import Path


def test_gw_modules_are_valid_python():
    package = Path(__file__).resolve().parents[1] / "pyqed" / "gw"
    for path in package.rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
