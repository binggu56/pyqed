"""Optional backend discovery hooks for MD workflows."""


def backend_status(name="python"):
    """Return availability metadata for an MD backend name."""
    name = name.lower()
    if name == "python":
        return {"name": "python", "available": True, "reason": "native PyQED MD backend"}
    if name == "openmm":
        try:
            import openmm  # noqa: F401
        except ModuleNotFoundError:
            return {"name": "openmm", "available": False, "reason": "OpenMM is not installed"}
        return {
            "name": "openmm",
            "available": False,
            "reason": "OpenMM is installed, but the PyQED OpenMM adapter is not wired yet",
        }
    raise ValueError("backend must be 'python' or 'openmm'.")
