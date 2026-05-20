"""Small trajectory and energy-log helpers for :mod:`pyqed.md`."""


def write_xyz(atoms, fileobj, comment=""):
    """Write one XYZ frame to a file-like object."""
    positions = atoms.get_positions()
    fileobj.write(f"{len(atoms)}\n")
    fileobj.write(f"{comment}\n")
    for symbol, xyz in zip(atoms.atom_symbols(), positions):
        fileobj.write(f"{symbol} {xyz[0]: .10f} {xyz[1]: .10f} {xyz[2]: .10f}\n")


class XYZTrajectoryWriter:
    """Observer-compatible XYZ trajectory writer."""

    def __init__(self, atoms, filename, dynamics=None):
        self.atoms = atoms
        self.dynamics = dynamics
        self.fileobj = open(filename, "w")

    def __call__(self):
        time = "" if self.dynamics is None else f"time={self.dynamics.get_time():.8f}"
        write_xyz(self.atoms, self.fileobj, comment=time)
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()


class EnergyLogger:
    """Observer-compatible energy logger."""

    def __init__(self, atoms, filename, dynamics=None):
        self.atoms = atoms
        self.dynamics = dynamics
        self.fileobj = open(filename, "w")
        self.fileobj.write("step time potential kinetic total temperature_K\n")

    def __call__(self):
        step = 0 if self.dynamics is None else self.dynamics.get_number_of_steps()
        time = 0.0 if self.dynamics is None else self.dynamics.get_time()
        potential = self.atoms.get_potential_energy()
        kinetic = self.atoms.get_kinetic_energy()
        self.fileobj.write(
            f"{step:d} {time:.10f} {potential:.12e} {kinetic:.12e} "
            f"{potential + kinetic:.12e} {self.atoms.get_temperature():.8f}\n"
        )
        self.fileobj.flush()

    def close(self):
        self.fileobj.close()
