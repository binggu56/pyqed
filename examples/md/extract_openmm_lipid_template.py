"""Inspect lipid residue templates from an installed OpenMM force-field XML."""

from __future__ import annotations

import argparse
import json

from pyqed.md import available_openmm_lipid_templates, openmm_lipid_template


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--residue", default="DPPC", help="Residue name to extract, e.g. DPPC or POPC.")
    parser.add_argument("--source", default=None, help="Optional OpenMM force-field XML path.")
    parser.add_argument("--list", action="store_true", help="List available residue templates.")
    args = parser.parse_args()

    if args.list:
        for name in available_openmm_lipid_templates(args.source):
            print(name)
        return

    template = openmm_lipid_template(args.residue, source=args.source)
    summary = {
        "residue_name": template.residue_name,
        "source": template.source,
        "validated": template.validated,
        "natoms": template.natoms,
        "net_charge": template.net_charge,
        "bonds": len(template.bonds),
        "angles": len(template.angles),
        "torsions": len(template.torsions),
        "coulomb14scale": template.coulomb14scale,
        "lj14scale": template.lj14scale,
        "first_atoms": [
            {
                "name": name,
                "type": atom_type,
                "element": element,
                "charge": charge,
            }
            for name, atom_type, element, charge in zip(
                template.atom_names[:8],
                template.atom_types[:8],
                template.elements[:8],
                template.charges[:8],
            )
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
