#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from .cell import Cell


class Chain(Cell):
    """
    Explicit 1D periodic chain.

    ``Chain`` is the preferred public API for the native 1D PBC code. The older
    ``Cell(..., dimension=1)`` path remains for compatibility.
    """

    def __init__(
        self,
        atom,
        a,
        basis,
        unit="bohr",
        charge=0,
        spin=0,
        vacuum=20.0,
        low_dim_ft_type="inf_vacuum",
        integral_driver="builtin",
        integral_options=None,
    ):
        super().__init__(
            atom=atom,
            a=a,
            basis=basis,
            unit=unit,
            charge=charge,
            spin=spin,
            dimension=1,
            vacuum=vacuum,
            low_dim_ft_type=low_dim_ft_type,
            integral_driver=integral_driver,
            integral_options=integral_options,
        )

    @property
    def lattice_constant(self):
        if not self.built:
            self.build()
        return float((self.lattice_vectors[0] @ self.lattice_vectors[0]) ** 0.5)
