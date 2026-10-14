import Image from "next/image";
import Link from "next/link";
import { CopyButton } from "./copy-button";
import { links, paper, release, softwareArchive } from "./site-data";

const workflows = [
  {
    slug: "electronic-structure",
    index: "01",
    title: "Electronic structure",
    status: "Core path",
    duration: "≈5 min",
    task: "Build H₂, run native RHF, and print a converged total energy.",
    description:
      "Start with molecular construction and native integrals, then continue into MP2, CI, CASCI, and CASSCF research workflows.",
    prerequisites: "Core PyQED install",
    expected: "RHF energy: −1.11675931 Ha",
    exampleHref: links.quickstartSource,
    exampleLabel: "Run the H₂ example",
    docsHref: links.qchem,
  },
  {
    slug: "nonadiabatic-dynamics",
    index: "02",
    title: "Nonadiabatic dynamics",
    status: "Experimental",
    duration: "Starter model",
    task: "Propagate a two-state avoided crossing with LDRFG.",
    description:
      "Learn the locally diabatic representation on a small model before moving to molecular state surfaces, linked overlaps, and sparse grids.",
    prerequisites: "Core install · NumPy/SciPy",
    expected: "State populations against an exact grid reference",
    exampleHref: links.avoidedCrossingSource,
    exampleLabel: "Run the avoided crossing",
    docsHref: links.geometricDynamics,
  },
  {
    slug: "open-systems",
    index: "03",
    title: "Open systems & spectroscopy",
    status: "Experimental",
    duration: "Starter model",
    task: "Propagate a spin–boson HEOM model and inspect spectra.",
    description:
      "Use hierarchy and master-equation methods for dissipative dynamics, correlation functions, and nonlinear optical response.",
    prerequisites: "Core install · model parameters",
    expected: "Population dynamics and response functions",
    exampleHref: links.heomSource,
    exampleLabel: "Run the HEOM example",
    docsHref: links.heom,
  },
  {
    slug: "tensor-networks",
    index: "04",
    title: "MPS & DMRG",
    status: "Experimental",
    duration: "Starter MPO",
    task: "Construct a fermionic MPO with automatic Jordan–Wigner strings.",
    description:
      "Continue from operator construction to MPS, DMRG, TDVP, symmetry-aware algorithms, and optional active-space workflows.",
    prerequisites: "Core install · PySCF optional for DMRG-SCF",
    expected: "An inspectable many-body operator and bond structure",
    exampleHref: links.autompoSource,
    exampleLabel: "Build the MPO",
    docsHref: links.mps,
  },
] as const;

const studies = [
  {
    src: "/research/h3plus-crossing-topology.png",
    label: "Electronic-state topology",
    title: "H₃⁺ near an S₁/S₂ crossing",
    description:
      "Energy-gap topology, overlap links, and adiabatic potential-energy surfaces along coupled nuclear coordinates.",
    provenance: "AM1/MECI · S₁/S₂ topology · linked-overlap scans",
    availability:
      "Related public scans are linked; the exact plotted output archive is not yet published.",
    sourceHref: links.h3GapSource,
    sourceLabel: "View related gap scan",
    secondaryHref: links.h3LinkSource,
    secondaryLabel: "Inspect link scan",
    methodHref: links.geometricDynamics,
    alt: "Three-panel H3-plus plot showing an S1–S2 energy-gap contour, a colored overlap-link grid, and two adiabatic surfaces.",
    className: "study-wide",
    sizes: "(max-width: 760px) calc(100vw - 36px), (max-width: 1200px) 55vw, 700px",
    width: 2505,
    height: 735,
  },
  {
    src: "/research/so2-state-wavepackets.png",
    label: "Nonadiabatic dynamics",
    title: "SO₂ state-resolved wavepackets",
    description:
      "Nuclear probability moves across three coupled electronic states over a 100-femtosecond trajectory.",
    provenance: "Three states · 15×15×9 grid · 100 fs propagation",
    availability:
      "The calculation source is public; the exact output bundle is not yet archived.",
    sourceHref: links.so2Source,
    sourceLabel: "View calculation source",
    methodHref: links.namd,
    alt: "Eighteen heat maps tracking SO2 nuclear probability in r1–r2 coordinates on three electronic states from 0 to 100 femtoseconds.",
    className: "study-square",
    sizes: "(max-width: 760px) calc(100vw - 36px), (max-width: 1200px) 38vw, 470px",
    width: 2760,
    height: 1350,
  },
  {
    src: "/research/pyrazine-spectrum.png",
    label: "Spectroscopy",
    title: "24-mode pyrazine spectrum",
    description:
      "A QDE/LDRFG spectrum shown alongside a shifted LDR reference and experimental line shape.",
    provenance: "24-mode LVC model · 80 fs spectrum · source-controlled simulation",
    availability:
      "The simulation source is public; the exact plotted NPZ archive is not yet published.",
    sourceHref: links.pyrazine24Source,
    sourceLabel: "View simulation source",
    methodHref: links.geometricDynamics,
    alt: "Normalized pyrazine spectral curves comparing QDE/LDRFG, a shifted LDR reference, and experiment between about 4.65 and 5.25 electronvolts.",
    className: "study-wide",
    sizes: "(max-width: 760px) calc(100vw - 36px), (max-width: 1200px) 60vw, 760px",
    width: 1980,
    height: 1110,
  },
] as const;

const installCommand = "pip install pyqed";

const quickstart = `from pyqed.qchem import Molecule

mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    unit="angstrom",
    basis="sto-3g",
)
mol.build(eri="auto")

mf = mol.RHF().run()
print(f"RHF energy: {mf.e_tot:.8f} Ha")`;

const validationCards = [
  {
    label: "Reviewed external validation",
    metric: "1.13 × 10⁻¹⁰ Ha",
    title: "H₂ RHF difference from PySCF",
    description:
      "A scoped STO-3G comparison against PySCF 2.12.1 passed its 10⁻⁹-hartree tolerance. Inputs, raw output, hashes, and limitations are recorded.",
    href: links.h2Validation,
    action: "Inspect the validation record",
  },
  {
    label: "Published runtime",
    metric: `Python ${release.python}`,
    title: "One installable research release",
    description:
      "Platform wheels and a source archive are published on PyPI. Optional scientific backends are documented separately from the native core path.",
    href: links.installation,
    action: "Review installation options",
  },
  {
    label: "Release provenance",
    metric: `v${release.version}`,
    title: "Tagged source and verified publishing",
    description:
      "Release artifacts are connected to tagged source, checksums, workflow records, and PyPI provenance attestations.",
    href: links.release,
    action: `Read the ${release.version} release`,
  },
] as const;

const projectLinks = [
  { label: `Release v${release.version}`, href: links.release },
  { label: "Benchmark records", href: links.benchmarkCatalog },
  { label: "Contribute", href: links.contributing },
  { label: "Discussions", href: links.discussions },
  { label: "Contributors", href: links.contributors },
  { label: "Software archive", href: softwareArchive.conceptUrl },
] as const;

const softwareJsonLd = {
  "@context": "https://schema.org",
  "@type": "SoftwareApplication",
  name: "PyQED",
  url: "https://pyqed.org",
  applicationCategory: "ScienceApplication",
  operatingSystem: "Cross-platform",
  programmingLanguage: "Python",
  runtimePlatform: `Python ${release.python}`,
  softwareVersion: release.version,
  description:
    "Open-source research software for electronic structure and quantum dynamics.",
  codeRepository: links.github,
  downloadUrl: links.pypi,
  releaseNotes: links.release,
  softwareHelp: links.userGuide,
  license: links.license,
  identifier: {
    "@type": "PropertyValue",
    propertyID: "GitHub release",
    value: `v${release.version}`,
    url: links.release,
  },
  isPartOf: {
    "@type": "CreativeWorkSeries",
    name: "PyQED software archive (all versions)",
    identifier: {
      "@type": "PropertyValue",
      propertyID: "DOI",
      value: softwareArchive.conceptDoi,
    },
    url: softwareArchive.conceptUrl,
  },
  citation: {
    "@type": "ScholarlyArticle",
    name: paper.title,
    author: paper.authors.map((name) => ({ "@type": "Person", name })),
    isPartOf: { "@type": "Periodical", name: paper.journal },
    datePublished: String(paper.year),
    identifier: {
      "@type": "PropertyValue",
      propertyID: "DOI",
      value: paper.doi,
    },
    url: paper.url,
  },
  isAccessibleForFree: true,
  author: [{ "@type": "Person", name: "Bing Gu", url: links.maintainer }],
  maintainer: {
    "@type": "Person",
    name: "Bing Gu",
    url: links.maintainer,
  },
  sameAs: [links.docs, links.github, links.pypi, softwareArchive.conceptUrl],
};

const navigation = [
  ["Quickstart", links.quickstart],
  ["User Guide", links.userGuide],
  ["Examples", "/examples"],
  ["Viewer", "/viewer"],
] as const;

export default function Home() {
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(softwareJsonLd) }}
      />

      <header className="site-header home-header">
        <nav className="site-nav shell" aria-label="Primary navigation">
          <Link className="wordmark" href="/" aria-label="PyQED home">
            <span className="wordmark-mark">P</span>
            <span>PyQED</span>
          </Link>

          <div className="nav-links">
            {navigation.map(([label, href]) => (
              <a href={href} key={href}>
                {label}
              </a>
            ))}
          </div>

          <div className="nav-actions">
            <a className="nav-github" href={links.github}>
              GitHub <span aria-hidden="true">↗</span>
            </a>
            <details className="mobile-menu">
              <summary>Menu</summary>
              <nav aria-label="Mobile navigation">
                {navigation.map(([label, href]) => (
                  <a href={href} key={href}>
                    {label}
                  </a>
                ))}
                <a href={links.docs}>Stable documentation</a>
              </nav>
            </details>
          </div>
        </nav>
      </header>

      <main className="home" id="main-content">
        <section className="hero" id="top">
          <div className="hero-grid" aria-hidden="true" />

          <div className="hero-content shell">
            <div className="hero-copy">
              <p className="eyebrow">
                <span /> Research software for chemical physics
              </p>
              <h1>
                Electronic structure and{" "}
                <em>quantum dynamics in Python.</em>
              </h1>
              <p className="hero-lede">
                Build molecular models, follow quantum dynamics, and explore
                spectra with an open Python framework for chemical physics.
              </p>

              <ul className="release-chips" aria-label="Current release status">
                <li>PyQED {release.version}</li>
                <li>Python {release.python}</li>
                <li>MIT</li>
              </ul>

              <div className="hero-install" aria-label="Install PyQED">
                <code>{installCommand}</code>
                <CopyButton
                  analyticsEvent="install_copy"
                  text={installCommand}
                />
              </div>

              <div className="hero-actions">
                <a className="button button-primary" href={links.quickstart}>
                  Run the 5-minute quickstart <span aria-hidden="true">→</span>
                </a>
                <a className="button button-quiet" href={links.userGuide}>
                  Open the User Guide
                </a>
              </div>
            </div>

            <div className="code-window" id="quickstart" aria-label="Python quickstart example">
              <div className="code-titlebar">
                <span className="window-dots" aria-hidden="true">
                  <i />
                  <i />
                  <i />
                </span>
                <span>quickstart.py</span>
                <CopyButton label="Copy code" text={quickstart} />
              </div>
              <pre>
                <code>{quickstart}</code>
              </pre>
              <p className="code-result">
                <span>Expected</span>
                <code>RHF energy: -1.11675931 Ha</code>
              </p>
            </div>
          </div>
        </section>

        <section
          className="capabilities"
          id="workflows"
          aria-labelledby="workflows-heading"
        >
          <div className="section-heading shell">
            <div>
              <p className="kicker">Choose a starting point</p>
              <h2 id="workflows-heading">What you can do with PyQED</h2>
            </div>
            <p>
              Choose a field and start with a small, runnable example.
            </p>
          </div>

          <div className="capability-grid shell">
            {workflows.map((workflow) => (
              <article
                className="capability-card"
                data-workflow={workflow.slug}
                key={workflow.slug}
              >
                <div className="capability-topline">
                  <span>{workflow.index}</span>
                  <span>{workflow.status}</span>
                </div>
                <h3>{workflow.title}</h3>
                <p className="workflow-task">{workflow.task}</p>
                <details className="workflow-more">
                  <summary>Requirements and details</summary>
                  <p>{workflow.description}</p>
                  <p className="workflow-duration">{workflow.duration}</p>
                  <dl className="workflow-details">
                    <div>
                      <dt>Prerequisites</dt>
                      <dd>{workflow.prerequisites}</dd>
                    </div>
                    <div>
                      <dt>Expected result</dt>
                      <dd>{workflow.expected}</dd>
                    </div>
                  </dl>
                </details>
                <div className="card-actions">
                  <a className="capability-link" href={workflow.exampleHref}>
                    {workflow.exampleLabel} <span aria-hidden="true">↗</span>
                  </a>
                  <a className="secondary-link" href={workflow.docsHref}>
                    Read the guide
                  </a>
                </div>
              </article>
            ))}
          </div>

          <p className="scope-note shell">
            Research APIs and method coverage are evolving.
            <a href={links.benchmarkCatalog}>Review method and benchmark scope →</a>
          </p>
        </section>

        <section className="research" id="research">
          <div className="section-heading section-heading-light shell">
            <div>
              <p className="kicker">Selected research</p>
              <h2>Computed with PyQED.</h2>
            </div>
            <p>
              Explore the calculations, source code, and methods behind
              these research figures.
            </p>
          </div>

          <div className="study-grid shell">
            {studies.map((study) => (
              <figure className={`study-card ${study.className}`} key={study.title}>
                <a
                  className="study-image-wrap"
                  href={study.src}
                  aria-label={`View full-size figure: ${study.title}`}
                >
                  <Image
                    src={study.src}
                    alt={study.alt}
                    width={study.width}
                    height={study.height}
                    loading="lazy"
                    sizes={study.sizes}
                    unoptimized
                  />
                </a>
                <figcaption>
                  <p className="study-label">{study.label}</p>
                  <h3>{study.title}</h3>
                  <p>{study.description}</p>
                  <details className="research-details">
                    <summary>Calculation details</summary>
                    <dl className="study-provenance">
                      <div>
                        <dt>Parameters</dt>
                        <dd>{study.provenance}</dd>
                      </div>
                      <div>
                        <dt>Availability</dt>
                        <dd>{study.availability}</dd>
                      </div>
                    </dl>
                  </details>
                  <div className="study-actions">
                    <a data-research-source href={study.sourceHref}>
                      {study.sourceLabel} <span aria-hidden="true">↗</span>
                    </a>
                    {"secondaryHref" in study ? (
                      <a href={study.secondaryHref}>{study.secondaryLabel}</a>
                    ) : null}
                    <a href={study.methodHref}>Method guide</a>
                  </div>
                </figcaption>
              </figure>
            ))}
          </div>
        </section>

        <section className="project" id="project">
          <div className="project-overview shell">
            <div className="project-citation">
              <p className="kicker">About the project</p>
              <h2>Open source for chemical physics.</h2>
              <p className="project-maintainer">
                Maintained by <a href={links.maintainer}>Bing Gu</a> · MIT license
              </p>
              <div className="citation-note">
                <h3>Cite PyQED</h3>
                <p>
                  <a href={paper.url}>{paper.title}</a>
                  <br />
                  {paper.authors.join(", ")} · <em>{paper.journal}</em> ({paper.year})
                </p>
                <a className="text-link" href={paper.url}>DOI: {paper.doi} ↗</a>
              </div>
            </div>
            <div className="project-resources">
              <nav aria-label="Project resources">
                {projectLinks.map((link) => (
                  <a href={link.href} key={link.href}>
                    {link.label} <span aria-hidden="true">↗</span>
                  </a>
                ))}
              </nav>
              <details className="release-details" id="evidence">
                <summary>Validation and release information</summary>
                {validationCards.map((card) => (
                  <article key={card.label}>
                    <h3>{card.title}</h3>
                    <p><strong>{card.metric}</strong> · {card.description}</p>
                    <a href={card.href}>{card.action} ↗</a>
                  </article>
                ))}
              </details>
            </div>
          </div>
        </section>
      </main>

      <footer>
        <div className="footer-inner shell">
          <div>
            <Link className="wordmark wordmark-footer" href="/">
              <span className="wordmark-mark">P</span>
              <span>PyQED</span>
            </Link>
            <p>Electronic structure and quantum dynamics in Python.</p>
          </div>
          <div className="footer-links">
            <Link href="/examples">Examples</Link>
            <Link href="/viewer">Viewer</Link>
            <a href={links.userGuide}>User Guide</a>
            <a href={links.docs}>Stable docs</a>
            <a href={links.developmentDocs}>Development docs</a>
            <a href={links.github}>GitHub</a>
            <a href={links.pypi}>PyPI</a>
            <a href={links.citation}>Cite</a>
            <Link href="/privacy">Privacy</Link>
          </div>
          <p className="footer-meta">
            PyQED {release.version} · MIT
            <br />
            pyqed.org
          </p>
        </div>
      </footer>
    </>
  );
}
