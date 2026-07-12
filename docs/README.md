# PyQED documentation

The public user guide is built from the reStructuredText sources in
`docs/source/` with Sphinx. Read the Docs uses `.readthedocs.yaml` and
`docs/requirements.txt` from the repository root.

Build the same strict HTML documentation locally from the repository root:

```bash
python -m pip install -r docs/requirements.txt
python -m sphinx -W --keep-going -b html docs/source /tmp/pyqed-docs
```

Open `/tmp/pyqed-docs/index.html` after the build succeeds.

Documentation principles:

- Keep one canonical page for each method and link to it from the guide hub.
- Put a minimal runnable example and expected result near the top of a method
  page.
- State units, convergence controls, limitations, maturity, and optional
  dependencies explicitly.
- Use `literalinclude` for tracked examples when practical so code on the web
  stays synchronized with executable files.
- Run the strict Sphinx command above before submitting documentation changes.

The project overview at `pyqed.org` should remain concise; detailed scientific
guidance belongs here and is published at `docs.pyqed.org`.
