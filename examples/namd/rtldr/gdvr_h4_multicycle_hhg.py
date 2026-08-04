"""Preset launcher for a longer H4 GDVR RT-LDR HHG run."""

from __future__ import annotations

import argparse

from . import gdvr_h4_modes_hhg as h4_hhg


_ACTIVE_MODE_ALIASES = {
    "all": None,
    "three": None,
    "breathing": (0,),
    "symmetric": (0,),
    "antisymmetric": (1,),
    "outer": (2,),
}


def _active_modes(value):
    key = str(value).lower().replace("_", "-")
    if key in _ACTIVE_MODE_ALIASES:
        return _ACTIVE_MODE_ALIASES[key]
    return tuple(int(item) for item in key.split(",") if item)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycles", type=float, default=10.0)
    parser.add_argument("--ramp-cycles", type=float, default=1.5)
    parser.add_argument("--initial-state", default="ground")
    parser.add_argument("--active-modes", default="all")
    parser.add_argument("--nmode", type=int, default=3)
    parser.add_argument("--tag", default="h4_three_mode_multicycle_hhg")
    parser.add_argument("--propagation-workers", type=int, default=1)
    parser.add_argument("--electronic-substeps", type=int, default=1)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.active_mode_indices = _active_modes(args.active_modes)
    return h4_hhg.run(args)


if __name__ == "__main__":
    main()
