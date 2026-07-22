"""System-composition summaries for MD structures."""

from __future__ import annotations

from collections import Counter


PROTEIN_RESIDUES = frozenset(
    {
        "ALA",
        "ARG",
        "ASN",
        "ASP",
        "CYS",
        "GLN",
        "GLU",
        "GLY",
        "HIS",
        "HID",
        "HIE",
        "HIP",
        "ILE",
        "LEU",
        "LYS",
        "MET",
        "PHE",
        "PRO",
        "SER",
        "THR",
        "TRP",
        "TYR",
        "VAL",
    }
)
LIPID_RESIDUES = frozenset({"DPP", "DPPC", "POPC", "POPE", "DOPC", "DLPC", "DMPC", "DSPC", "POPG", "CHL1"})
WATER_RESIDUES = frozenset({"HOH", "WAT", "TIP3", "SOL"})
ION_RESIDUES = frozenset({"NA", "CLA", "CL", "K", "POT", "MG", "CAL", "CA", "ZN"})


def residue_composition(atoms):
    """Summarize residue classes from atom metadata arrays."""
    summary = {"atoms": int(len(atoms))}
    if not hasattr(atoms, "has") or not atoms.has("residue_names"):
        return summary

    residue_names = [str(name).strip() for name in atoms.get_array("residue_names")]
    residue_ids = (
        [str(value).strip() for value in atoms.get_array("residue_ids")]
        if atoms.has("residue_ids")
        else [str(index) for index in range(len(atoms))]
    )
    chain_ids = (
        [str(value).strip() for value in atoms.get_array("chain_ids")]
        if atoms.has("chain_ids")
        else [""] * len(atoms)
    )
    classes = {
        "protein": PROTEIN_RESIDUES,
        "lipid": LIPID_RESIDUES,
        "water": WATER_RESIDUES,
        "ion": ION_RESIDUES,
    }
    residues_by_class = {name: set() for name in (*classes, "other")}
    atoms_by_class = {name: 0 for name in (*classes, "other")}
    residue_counts = Counter(residue_names)

    for residue_name, residue_id, chain_id in zip(residue_names, residue_ids, chain_ids):
        key = (chain_id, residue_id, residue_name)
        category = "other"
        for name, names in classes.items():
            if residue_name in names:
                category = name
                break
        atoms_by_class[category] += 1
        residues_by_class[category].add(key)

    for category in (*classes, "other"):
        summary[f"{category}_atoms"] = int(atoms_by_class[category])
        summary[f"{category}_residues"] = int(len(residues_by_class[category]))
    summary["protein_chains"] = int(len({chain for chain, _resid, _name in residues_by_class["protein"]}))
    summary["residue_counts"] = {name: int(count) for name, count in sorted(residue_counts.items())}
    summary["residues"] = int(sum(len(values) for values in residues_by_class.values()))
    return summary
