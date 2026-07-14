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

const quickstart = `from pyqed.qchem import Molecule

mol = Molecule(
    atom="H 0 0 0; H 0 0 0.74",
    unit="angstrom",
    basis="sto-3g",
)
mol.build(driver="builtin", eri="auto")

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
      "The universal wheel and source archive are published on PyPI. Optional scientific backends are documented separately from the native core path.",
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

const projectPaths = [
  {
    label: "Release",
    title: "Installable artifacts, traceable source.",
    description:
      "Use the tagged release for reproducible work and record the exact package version with your calculation.",
    href: links.release,
    action: `Open release ${release.version}`,
  },
  {
    label: "Cite",
    title: paper.title,
    description: `${paper.authors.join(", ")} · ${paper.journal} (${paper.year}). Cite the article for the project and the Zenodo archive for the exact software release.`,
    href: paper.url,
    action: `Open DOI ${paper.doi}`,
  },
  {
    label: "Benchmarks",
    title: "Read claims at their stated scope.",
    description:
      "Reviewed and candidate records are labeled separately, with runnable inputs and explicit claim boundaries.",
    href: links.benchmarkCatalog,
    action: "Review the benchmark catalog",
  },
  {
    label: "Contribute",
    title: "Improve PyQED in the open.",
    description:
      "Follow the contribution guide for development setup, focused tests, review expectations, and responsible changes.",
    href: links.contributing,
    action: "Read the contribution guide",
  },
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
    propertyID: "DOI",
    value: softwareArchive.versionDoi,
    url: softwareArchive.versionUrl,
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
  ["Workflows", "#workflows"],
  ["Quickstart", "#quickstart"],
  ["Examples", "/examples"],
  ["Evidence", "#evidence"],
  ["Research", "#research"],
  ["Project", "#project"],
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

      <header className="site-header">
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
            <Link href="/viewer">Viewer</Link>
            <a href={links.userGuide}>User Guide</a>
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
                <Link href="/viewer">Viewer</Link>
                <a href={links.userGuide}>User Guide</a>
                <a href={links.docs}>Stable documentation</a>
              </nav>
            </details>
          </div>
        </nav>
      </header>

      <main id="main-content">
        <section className="hero" id="top">
          <div className="hero-grid" aria-hidden="true" />

          <div className="hero-content shell">
            <div className="hero-copy">
              <p className="eyebrow">
                <span /> Research software for chemical physics
              </p>
              <h1>
                Electronic structure and
                <br />
                <em>quantum dynamics in Python.</em>
              </h1>
              <p className="hero-lede">
                PyQED connects molecular electronic structure, nonadiabatic
                motion, open-system dynamics, spectroscopy, and tensor
                networks in one inspectable Python codebase.
              </p>

              <ul className="release-chips" aria-label="Current release status">
                <li>PyQED {release.version}</li>
                <li>Python {release.python}</li>
                <li>MIT</li>
                <li>Research APIs evolving</li>
              </ul>

              <div className="hero-install" aria-label="Install PyQED">
                <code>{release.installCommand}</code>
                <CopyButton
                  analyticsEvent="install_copy"
                  text={release.installCommand}
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

            <div className="orbital-stage" aria-hidden="true">
              <div className="orbital-glow" />
              <div className="orbit orbit-one">
                <span className="particle particle-one" />
              </div>
              <div className="orbit orbit-two">
                <span className="particle particle-two" />
              </div>
              <div className="orbit orbit-three">
                <span className="particle particle-three" />
              </div>
              <div className="orbital-core">
                <span>iℏ</span>
                <small>∂ₜ|ψ⟩ = Ĥ|ψ⟩</small>
              </div>
              <p className="orbit-note orbit-note-one">electrons</p>
              <p className="orbit-note orbit-note-two">nuclei</p>
              <p className="orbit-note orbit-note-three">photons</p>
            </div>
          </div>

          <div className="hero-foot shell">
            <p>For computational chemists and chemical physicists</p>
            <div className="hero-metrics" aria-label="Project scope">
              <span>Native core path</span>
              <span>Scoped validation</span>
              <span>Open methods</span>
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
              <h2 id="workflows-heading">Four paths into the code.</h2>
            </div>
            <p>
              Begin with a small, inspectable calculation. Each path states
              what it needs, what it produces, and where to go deeper.
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
                <p className="workflow-duration">{workflow.duration}</p>
                <h3>{workflow.title}</h3>
                <p className="workflow-task">{workflow.task}</p>
                <p>{workflow.description}</p>
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

          <div className="scope-note shell">
            <strong>Know the scope.</strong>
            <p>
              PyQED is research software: interfaces and method coverage are
              evolving. Treat benchmark claims as workload-specific, and read
              each example&apos;s prerequisites before scaling it up.
            </p>
            <a href={links.benchmarkCatalog}>Read benchmark status definitions →</a>
          </div>
        </section>

        <section className="quickstart" id="quickstart">
          <div className="quickstart-inner shell">
            <div className="quickstart-copy">
              <p className="kicker">Five-minute quickstart</p>
              <h2>From geometry to a checked energy.</h2>
              <p>
                Install the current release, construct H₂, and run a native
                restricted Hartree–Fock calculation. The built-in driver keeps
                the first example independent of optional PySCF integrations.
              </p>
              <div className="install-command" aria-label="Install command">
                <code>{release.installCommand}</code>
                <CopyButton
                  analyticsEvent="install_copy"
                  text={release.installCommand}
                />
              </div>
              <p className="quickstart-note">
                Expected on the documented 0.74 Å geometry: −1.11675931 Ha.
                Automatic integral storage selects a native representation.
              </p>
              <div className="quickstart-links">
                <a className="text-link" href={links.quickstart}>
                  Follow the stable guide <span aria-hidden="true">→</span>
                </a>
                <a className="secondary-link secondary-link-dark" href={links.installation}>
                  Installation and optional backends
                </a>
                <Link className="secondary-link secondary-link-dark" href="/examples">
                  Browse runnable examples
                </Link>
              </div>
            </div>

            <div className="code-window" aria-label="Python quickstart example">
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

        <section className="evidence" id="evidence">
          <div className="section-heading shell">
            <div>
              <p className="kicker">Validated and maintained</p>
              <h2>Evidence before adjectives.</h2>
            </div>
            <p>
              PyQED publishes scoped numerical records and release provenance
              so that a claim can be traced back to its workload and artifacts.
            </p>
          </div>

          <div className="validation-grid shell">
            {validationCards.map((card) => (
              <article className="validation-card" key={card.label}>
                <p>{card.label}</p>
                <strong>{card.metric}</strong>
                <h3>{card.title}</h3>
                <p>{card.description}</p>
                <a href={card.href}>
                  {card.action} <span aria-hidden="true">↗</span>
                </a>
              </article>
            ))}
          </div>
        </section>

        <section className="research" id="research">
          <div className="section-heading section-heading-light shell">
            <div>
              <p className="kicker">Computed with PyQED</p>
              <h2>Methods with provenance.</h2>
            </div>
            <p>
              These studies expose their public calculation sources and method
              guides. Where an exact output archive is missing, the limitation
              is stated instead of implying full reproduction.
            </p>
          </div>

          <div className="study-grid shell">
            {studies.map((study) => (
              <figure className={`study-card ${study.className}`} key={study.title}>
                <div className="study-image-wrap">
                  <Image
                    src={study.src}
                    alt={study.alt}
                    width={study.width}
                    height={study.height}
                    loading="lazy"
                    sizes={study.sizes}
                    unoptimized
                  />
                </div>
                <figcaption>
                  <p className="study-label">{study.label}</p>
                  <h3>{study.title}</h3>
                  <p>{study.description}</p>
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
          <div className="section-heading section-heading-light shell">
            <div>
              <p className="kicker">The public record</p>
              <h2>Release, cite, inspect, contribute.</h2>
            </div>
            <p>
              Scientific software is more than source code. Follow the records
              for releases, evidence, attribution, and project stewardship.
            </p>
          </div>

          <div className="project-grid shell">
            {projectPaths.map((path) => (
              <article className="project-card" key={path.label}>
                <p>{path.label}</p>
                <h3>{path.title}</h3>
                <p>{path.description}</p>
                <a href={path.href}>
                  {path.action} <span aria-hidden="true">↗</span>
                </a>
              </article>
            ))}
          </div>
        </section>

        <section className="community" id="community">
          <div className="community-inner shell">
            <div className="community-copy">
              <p className="kicker">People and stewardship</p>
              <h2>Built in the open, credited precisely.</h2>
              <p>
                PyQED&apos;s citation metadata names Bing Gu as the software
                author. The repository records the broader history of
                contributions, discussion, and review.
              </p>
              <div className="source-actions">
                <a className="button button-ink" href={links.contributors}>
                  View contributors <span aria-hidden="true">↗</span>
                </a>
                <a className="button button-outline-ink" href={links.discussions}>
                  Join discussions
                </a>
              </div>
            </div>

            <div className="team-grid" aria-label="Cited software author">
              <article className="team-card">
                <p>Maintainer · cited author</p>
                <h3>Bing Gu</h3>
                <a href={links.maintainer}>GitHub profile ↗</a>
              </article>
              <div className="community-note">
                <strong>Using PyQED in published work?</strong>
                <p>
                  Cite the <a href={paper.url}>PyQED project paper</a> for the
                  framework and the <a href={softwareArchive.versionUrl}>
                    archived v{release.version} release (DOI {softwareArchive.versionDoi})
                  </a>{" "}
                  for the software used. The all-versions DOI{" "}
                  <a href={softwareArchive.conceptUrl}>
                    {softwareArchive.conceptDoi}
                  </a>{" "}
                  resolves all archived versions; also record the exact commit
                  and cite the algorithms used.
                </p>
              </div>
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
