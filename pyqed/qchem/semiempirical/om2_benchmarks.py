"""Published OM2 benchmark targets.

These records collect both aggregate benchmark statistics and selected
molecule-level supporting-information values.  They are useful until an
executable MNDO reference is available locally.
"""

from __future__ import annotations

from dataclasses import dataclass


GROUND_STATE_BENCHMARK_DOI = "10.1021/acs.jctc.5b01047"
GROUND_STATE_BENCHMARK_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4785506/"
PARAMETER_BENCHMARK_DOI = "10.1021/acs.jctc.5b01046"
PARAMETER_BENCHMARK_URL = "https://pmc.ncbi.nlm.nih.gov/articles/PMC4785507/"


@dataclass(frozen=True)
class PublishedOM2Benchmark:
    """One published aggregate OM2 benchmark statistic."""

    group: str
    subset: str
    n: int
    property: str
    unit: str
    mae: float | None = None
    mae_min: float | None = None
    mae_max: float | None = None
    method: str = "OM2"
    source_table: str = ""
    doi: str = GROUND_STATE_BENCHMARK_DOI
    url: str = GROUND_STATE_BENCHMARK_URL
    note: str = ""

    @property
    def value_label(self):
        if self.mae is not None:
            return f"{self.mae:g} {self.unit}"
        return f"{self.mae_min:g}-{self.mae_max:g} {self.unit}"


@dataclass(frozen=True)
class PublishedOM2MoleculeBenchmark:
    """One molecule/complex-level published OM2 benchmark value."""

    dataset: str
    name: str
    property: str
    unit: str
    reference: float
    om2: float
    method: str = "OM2"
    source_table: str = ""
    doi: str = PARAMETER_BENCHMARK_DOI
    url: str = PARAMETER_BENCHMARK_URL
    note: str = ""

    @property
    def error(self):
        return self.om2 - self.reference


PUBLISHED_OM2_S22_SINGLE_POINT_INTERACTIONS = (
    PublishedOM2MoleculeBenchmark("S22", "ammonia dimer", "interaction energy", "kcal/mol", -3.17, -2.0, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "water dimer", "interaction energy", "kcal/mol", -5.02, -7.0, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "formic acid dimer", "interaction energy", "kcal/mol", -18.8, -13.6, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "formamide dimer", "interaction energy", "kcal/mol", -16.1, -13.0, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "uracil dimer", "interaction energy", "kcal/mol", -20.47, -17.5, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "2-pyridoxine 2-aminopyridine", "interaction energy", "kcal/mol", -17.0, -11.0, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "adenine thymine", "interaction energy", "kcal/mol", -16.7, -11.1, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "methane dimer", "interaction energy", "kcal/mol", -0.53, 0.1, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "ethene dimer", "interaction energy", "kcal/mol", -1.51, -0.1, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene methane", "interaction energy", "kcal/mol", -1.45, -0.2, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene dimer parallel displaced", "interaction energy", "kcal/mol", -2.73, 1.3, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "pyrazine dimer", "interaction energy", "kcal/mol", -4.42, -0.9, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "uracil dimer stack", "interaction energy", "kcal/mol", -9.88, 4.3, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "indole benzene stack", "interaction energy", "kcal/mol", -5.22, -1.5, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "adenine thymine stack", "interaction energy", "kcal/mol", -11.7, -4.0, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "ethene ethyne", "interaction energy", "kcal/mol", -1.53, -1.1, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene water", "interaction energy", "kcal/mol", -3.28, -2.5, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene benzene T-shaped", "interaction energy", "kcal/mol", -2.3, -1.2, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene HCN", "interaction energy", "kcal/mol", -4.46, -3.1, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "benzene dimer T-shaped", "interaction energy", "kcal/mol", -2.74, -0.7, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "indole benzene T-shaped", "interaction energy", "kcal/mol", -5.73, -2.5, source_table="Table S9"),
    PublishedOM2MoleculeBenchmark("S22", "phenol dimer", "interaction energy", "kcal/mol", -7.05, -4.8, source_table="Table S9"),
)


PUBLISHED_OM2_G2_HEATS_OF_FORMATION_SAMPLE = (
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "OH radical, doublet",
        "heat of formation",
        "kcal/mol",
        142.5,
        140.0,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "triplet methylene (CH2)",
        "heat of formation",
        "kcal/mol",
        93.7,
        91.9,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "singlet methylene (CH2)",
        "heat of formation",
        "kcal/mol",
        102.8,
        103.9,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "methyl radical (CH3)",
        "heat of formation",
        "kcal/mol",
        35.0,
        33.7,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "methane (CH4)",
        "heat of formation",
        "kcal/mol",
        -17.9,
        -19.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "NH, triplet",
        "heat of formation",
        "kcal/mol",
        85.2,
        85.4,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "NH2 radical",
        "heat of formation",
        "kcal/mol",
        45.1,
        47.8,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "ammonia (NH3)",
        "heat of formation",
        "kcal/mol",
        -11.0,
        -5.9,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "water (H2O)",
        "heat of formation",
        "kcal/mol",
        -57.8,
        -56.5,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "hydrogen fluoride (HF)",
        "heat of formation",
        "kcal/mol",
        -65.1,
        -63.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "acetylene (C2H2)",
        "heat of formation",
        "kcal/mol",
        54.2,
        54.2,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "ethylene (C2H4)",
        "heat of formation",
        "kcal/mol",
        12.5,
        12.9,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "ethane (C2H6)",
        "heat of formation",
        "kcal/mol",
        -20.1,
        -21.2,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "hydrogen cyanide (HCN)",
        "heat of formation",
        "kcal/mol",
        31.5,
        26.1,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "carbon monoxide (CO)",
        "heat of formation",
        "kcal/mol",
        -26.4,
        -20.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "formaldehyde (H2C=O)",
        "heat of formation",
        "kcal/mol",
        -26.0,
        -30.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "methanol (CH3-OH)",
        "heat of formation",
        "kcal/mol",
        -48.0,
        -49.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "N2 molecule",
        "heat of formation",
        "kcal/mol",
        0.0,
        2.8,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "hydrogen peroxide (HO-OH)",
        "heat of formation",
        "kcal/mol",
        -32.5,
        -35.3,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "F2 molecule",
        "heat of formation",
        "kcal/mol",
        0.0,
        0.6,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "carbon dioxide (CO2)",
        "heat of formation",
        "kcal/mol",
        -94.1,
        -80.5,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "CF4",
        "heat of formation",
        "kcal/mol",
        -223.0,
        -220.5,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "COF2",
        "heat of formation",
        "kcal/mol",
        -152.7,
        -136.5,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "N2O",
        "heat of formation",
        "kcal/mol",
        19.6,
        21.2,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "NF3",
        "heat of formation",
        "kcal/mol",
        -31.6,
        -30.1,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
    PublishedOM2MoleculeBenchmark(
        "G2-CHNOF",
        "O3 (ozone)",
        "heat of formation",
        "kcal/mol",
        34.1,
        37.9,
        source_table="Table S16",
        doi=GROUND_STATE_BENCHMARK_DOI,
        url=GROUND_STATE_BENCHMARK_URL,
    ),
)


PUBLISHED_OM2_GROUND_STATE_MAES = (
    PublishedOM2Benchmark(
        group="G2G3-CHNOF",
        subset="G2-CHNOF",
        n=93,
        property="heats of formation",
        unit="kcal/mol",
        mae=3.37,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="G2G3-CHNOF",
        subset="G3-CHNOF",
        n=52,
        property="heats of formation",
        unit="kcal/mol",
        mae=3.18,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="G2G3-CHNOF",
        subset="alkanes28",
        n=6,
        property="relative energies",
        unit="kcal/mol",
        mae=0.61,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="OVS7-CHNOF",
        subset="radicals71",
        n=42,
        property="heats of formation",
        unit="kcal/mol",
        mae=4.98,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="OVS7-CHNOF",
        subset="radicals71",
        n=4,
        property="relative energies",
        unit="kcal/mol",
        mae=3.95,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="OVS7-CHNOF",
        subset="radicals71",
        n=25,
        property="ionization potentials",
        unit="eV",
        mae_min=0.37,
        mae_max=0.38,
        source_table="Table 4",
        note="Table reports OM1 and OM2 together as 0.37-0.38 eV.",
    ),
    PublishedOM2Benchmark(
        group="OVS7-CHNOF",
        subset="BIGMOL20",
        n=20,
        property="heats of formation",
        unit="kcal/mol",
        mae=4.85,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="OVS7-CHNOF",
        subset="isomers44",
        n=27,
        property="heats of formation",
        unit="kcal/mol",
        mae=1.05,
        source_table="Table 4",
    ),
    PublishedOM2Benchmark(
        group="fluorine91",
        subset="fluorine91",
        n=91,
        property="heats of formation",
        unit="kcal/mol",
        mae_min=7.15,
        mae_max=7.17,
        source_table="Table 4",
        note="Table reports OM1 and OM2 together as 7.15-7.17 kcal/mol.",
    ),
    PublishedOM2Benchmark(
        group="fluorine91",
        subset="fluorine91",
        n=455,
        property="bond lengths",
        unit="angstrom",
        mae_min=0.015,
        mae_max=0.016,
        source_table="Table 4",
        note="Table reports OM1, OM2, and PM3 together as 0.015-0.016 angstrom.",
    ),
    PublishedOM2Benchmark(
        group="fluorine91",
        subset="fluorine91",
        n=355,
        property="bond angles",
        unit="degree",
        mae_min=1.78,
        mae_max=2.04,
        source_table="Table 4",
        note="Table reports the OMx family together as 1.78-2.04 degrees.",
    ),
)


PUBLISHED_OM2_D_INTERACTION_MAES = (
    PublishedOM2Benchmark(
        group="S22",
        subset="overall",
        n=22,
        property="interaction energies",
        unit="kcal/mol",
        mae=0.91,
        method="OM2-D3",
        source_table="Table 7",
        doi=PARAMETER_BENCHMARK_DOI,
        url=PARAMETER_BENCHMARK_URL,
    ),
    PublishedOM2Benchmark(
        group="S22",
        subset="mixed",
        n=7,
        property="interaction energies",
        unit="kcal/mol",
        mae=0.27,
        method="OM2-D3",
        source_table="Table 7",
        doi=PARAMETER_BENCHMARK_DOI,
        url=PARAMETER_BENCHMARK_URL,
    ),
    PublishedOM2Benchmark(
        group="S22",
        subset="dispersion",
        n=8,
        property="interaction energies",
        unit="kcal/mol",
        mae=0.36,
        method="OM2-D3",
        source_table="Table 7",
        doi=PARAMETER_BENCHMARK_DOI,
        url=PARAMETER_BENCHMARK_URL,
    ),
    PublishedOM2Benchmark(
        group="S66x8",
        subset="overall",
        n=528,
        property="interaction energies",
        unit="kcal/mol",
        mae=0.78,
        method="OM2-D3",
        source_table="Table 7",
        doi=PARAMETER_BENCHMARK_DOI,
        url=PARAMETER_BENCHMARK_URL,
    ),
)


def published_om2_benchmarks(include_dispersion=True):
    """Return published aggregate OM2 benchmark records."""
    records = list(PUBLISHED_OM2_GROUND_STATE_MAES)
    if include_dispersion:
        records.extend(PUBLISHED_OM2_D_INTERACTION_MAES)
    return tuple(records)


def published_om2_molecule_benchmarks(dataset="S22"):
    """Return molecule-level published OM2 benchmark records."""
    dataset = dataset.lower()
    if dataset in {"all", "*"}:
        return PUBLISHED_OM2_S22_SINGLE_POINT_INTERACTIONS + PUBLISHED_OM2_G2_HEATS_OF_FORMATION_SAMPLE
    if dataset == "s22":
        return PUBLISHED_OM2_S22_SINGLE_POINT_INTERACTIONS
    if dataset in {"g2", "g2-chnof"}:
        return PUBLISHED_OM2_G2_HEATS_OF_FORMATION_SAMPLE
    raise ValueError(f"Unknown OM2 molecule benchmark dataset {dataset!r}.")


def format_published_om2_benchmarks(records=None):
    """Format benchmark records as a compact text table."""
    if records is None:
        records = published_om2_benchmarks()
    header = f"{'method':<8} {'group':<14} {'subset':<14} {'N':>5} {'property':<22} {'MAE':>16} {'source'}"
    lines = [header, "-" * len(header)]
    for rec in records:
        lines.append(
            f"{rec.method:<8} {rec.group:<14} {rec.subset:<14} {rec.n:>5} "
            f"{rec.property:<22} {rec.value_label:>16} {rec.source_table}"
        )
    return "\n".join(lines)


def format_published_om2_molecule_benchmarks(records=None):
    """Format molecule-level benchmark records as a compact text table."""
    if records is None:
        records = published_om2_molecule_benchmarks()
    header = f"{'dataset':<8} {'name':<36} {'ref':>10} {'OM2':>10} {'err':>10} {'unit'}"
    lines = [header, "-" * len(header)]
    for rec in records:
        lines.append(
            f"{rec.dataset:<8} {rec.name:<36} {rec.reference:>10.2f} "
            f"{rec.om2:>10.2f} {rec.error:>10.2f} {rec.unit}"
        )
    return "\n".join(lines)
