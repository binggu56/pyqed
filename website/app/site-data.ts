export const release = {
  version: "0.2.0",
  python: "3.10–3.13",
  installCommand: "python -m pip install pyqed==0.2.0",
} as const;

export const paper = {
  title: "PyQED: A Python Framework for Ab Initio Geometric Quantum Dynamics",
  authors: ["Yujuan Xie", "Xiaotong Zhu", "Bing Gu"],
  journal: "Chinese Journal of Chemical Physics",
  year: 2026,
  doi: "10.1063/1674-0068/cjcp2510161",
  url: "https://doi.org/10.1063/1674-0068/cjcp2510161",
} as const;

export const softwareArchive = {
  conceptDoi: "10.5281/zenodo.21316543",
  conceptUrl: "https://doi.org/10.5281/zenodo.21316543",
  versionDoi: "10.5281/zenodo.21316544",
  versionUrl: "https://doi.org/10.5281/zenodo.21316544",
} as const;

export const links = {
  docs: "https://docs.pyqed.org/en/stable/",
  userGuide: "https://docs.pyqed.org/en/latest/guide/guide.html",
  installation: "https://docs.pyqed.org/en/stable/installation.html",
  quickstart: "https://docs.pyqed.org/en/stable/quickstart.html",
  qchem: "https://docs.pyqed.org/en/stable/qchem.html",
  namd: "https://docs.pyqed.org/en/stable/pyqed.namd.html",
  heom: "https://docs.pyqed.org/en/stable/heom.html",
  mps: "https://docs.pyqed.org/en/stable/mps.html",
  geometricDynamics:
    "https://docs.pyqed.org/en/stable/geometric_quantum_dynamics.html",
  openDynamicsGuide:
    "https://docs.pyqed.org/en/stable/guide/guide_open_dynamics.html",
  spectroscopyGuide:
    "https://docs.pyqed.org/en/stable/guide/guide_spectroscopy.html",
  developmentDocs: "https://docs.pyqed.org/en/latest/",
  benchmarks: "https://github.com/binggu56/pyqed/tree/main/benchmarks",
  benchmarkCatalog:
    "https://github.com/binggu56/pyqed/blob/main/benchmarks/catalog.json",
  h2Validation:
    "https://github.com/binggu56/pyqed/tree/main/benchmarks/h2-sto3g-rhf-pyscf",
  github: "https://github.com/binggu56/pyqed",
  examples: "https://github.com/binggu56/pyqed/tree/main/examples",
  releaseExamples:
    "https://github.com/binggu56/pyqed/tree/v0.2.0/examples",
  quickstartSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/quickstart.py",
  quickstartReleaseSource:
    "https://github.com/binggu56/pyqed/blob/v0.2.0/examples/quickstart.py",
  sineDvrReleaseSource:
    "https://github.com/binggu56/pyqed/blob/v0.2.0/examples/dvr/fedvr_vs_sine_quartic.py",
  heomReleaseSource:
    "https://github.com/binggu56/pyqed/blob/v0.2.0/examples/heom.py",
  ehrenfestHistoriesReleaseSource:
    "https://github.com/binggu56/pyqed/blob/v0.2.0/examples/namd/ehrenfest_histories.py",
  dvrGuide: "https://docs.pyqed.org/en/stable/dvr.html",
  tutorials: "https://docs.pyqed.org/en/stable/tutorials.html",
  avoidedCrossingSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/namd/ldrfg_avoided_crossing.py",
  heomSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/heom.py",
  absorptionSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/signals/absorption.py",
  autompoSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/mps/autompo.py",
  dmrgscfSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/qchem/dmrgscf.py",
  h3GapSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/namd/h3plus_am1_meci_s1_s2_scan.py",
  h3LinkSource:
    "https://github.com/binggu56/pyqed/blob/main/examples/namd/h3plus_am1_meci_r1theta_link_scan.py",
  so2Source:
    "https://github.com/binggu56/pyqed/blob/main/examples/namd/so2_3d_sine_legendre_linked_ldr.py",
  pyrazine24Source:
    "https://github.com/binggu56/pyqed/blob/main/examples/namd/pyrazine_24mode_ldrfg.py",
  pypi: "https://pypi.org/project/pyqed/0.2.0/",
  release: "https://github.com/binggu56/pyqed/releases/tag/v0.2.0",
  citation: "https://github.com/binggu56/pyqed/blob/main/CITATION.cff",
  contributing:
    "https://github.com/binggu56/pyqed/blob/main/CONTRIBUTING.md",
  contributors: "https://github.com/binggu56/pyqed/graphs/contributors",
  discussions: "https://github.com/binggu56/pyqed/discussions",
  issues: "https://github.com/binggu56/pyqed/issues",
  license: "https://github.com/binggu56/pyqed/blob/main/LICENSE",
  maintainer: "https://github.com/binggu56",
} as const;
