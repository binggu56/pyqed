import type { Metadata } from "next";
import Link from "next/link";
import { CopyButton } from "../copy-button";
import { links, release } from "../site-data";
import { examples } from "./examples-data";
import { PythonCode } from "./python-code";

export const metadata: Metadata = {
  title: "Runnable Python Examples",
  description:
    "Copy, run, and inspect curated PyQED examples for electronic structure, Sine DVR, HEOM, and nonadiabatic dynamics.",
  alternates: { canonical: "/examples" },
  openGraph: {
    url: "https://pyqed.org/examples",
    title: "Runnable Python Examples | PyQED",
    description:
      "Four release-pinned programs with prerequisites, expected results, full source, and learning guides.",
  },
};

const examplesJsonLd = {
  "@context": "https://schema.org",
  "@type": "CollectionPage",
  name: "Runnable PyQED examples",
  url: "https://pyqed.org/examples",
  description:
    "Curated Python examples for electronic structure and quantum dynamics.",
  mainEntity: {
    "@type": "ItemList",
    itemListElement: examples.map((example, index) => ({
      "@type": "ListItem",
      position: index + 1,
      url: `https://pyqed.org/examples#${example.id}`,
      name: example.title,
    })),
  },
};

const navigation = [
  ["Home", "/"],
  ["Examples", "#examples"],
  ["Run notes", "#run-notes"],
] as const;

export default function ExamplesPage() {
  return (
    <>
      <a className="skip-link" href="#main-content">
        Skip to content
      </a>

      <script
        type="application/ld+json"
        dangerouslySetInnerHTML={{ __html: JSON.stringify(examplesJsonLd) }}
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
                <a href={links.userGuide}>User Guide</a>
                <a href={links.docs}>Stable documentation</a>
              </nav>
            </details>
          </div>
        </nav>
      </header>

      <main id="main-content" className="examples-page">
        <section className="examples-hero" aria-labelledby="examples-title">
          <div className="hero-grid" aria-hidden="true" />
          <div className="examples-hero-inner shell">
            <div>
              <p className="eyebrow">
                <span /> Learn from working programs
              </p>
              <h1 id="examples-title">
                Read it. Run it.
                <br />
                <em>Follow the source.</em>
              </h1>
              <p className="examples-hero-lede">
                Start with four inspectable calculations. Every example names
                its prerequisites, expected result, runtime class, exact
                release source, and next guide.
              </p>
              <div className="hero-actions">
                <a className="button button-primary" href="#examples">
                  Explore the examples <span aria-hidden="true">↓</span>
                </a>
                <a className="button button-quiet" href={links.releaseExamples}>
                  {`Browse all v${release.version} source`}
                </a>
              </div>
            </div>

            <aside className="examples-release-note" aria-label="Example provenance">
              <p>Curated release</p>
              <strong>{`PyQED ${release.version}`}</strong>
              <span>{`Python ${release.python}`}</span>
              <span>{`Source links pinned to tag v${release.version}`}</span>
              <span>Development branch may differ</span>
            </aside>
          </div>
        </section>

        <nav className="example-jump shell" aria-label="Examples on this page">
          {examples.map((example) => (
            <a href={`#${example.id}`} key={example.id}>
              <span>{example.index}</span>
              <strong>{example.track}</strong>
              <small>{example.title}</small>
            </a>
          ))}
        </nav>

        <section
          className="example-library"
          id="examples"
          aria-labelledby="library-heading"
        >
          <div className="section-heading shell">
            <div>
              <p className="kicker">Runnable, then readable</p>
              <h2 id="library-heading">Small programs, clear outcomes.</h2>
            </div>
            <p>
              The code below is taken from the tagged {release.version} source.
              Copy it for a quick inspection, or open the full file when you
              need its repository context.
            </p>
          </div>

          <div className="example-list shell">
            {examples.map((example) => (
              <article
                className="example-entry"
                id={example.id}
                data-example={example.id}
                key={example.id}
              >
                <div className="example-copy">
                  <div className="example-topline">
                    <span>{example.index}</span>
                    <span>{example.track}</span>
                  </div>
                  <h2>{example.title}</h2>
                  <p>{example.summary}</p>

                  <dl className="example-facts">
                    <div>
                      <dt>Prerequisites</dt>
                      <dd>{example.prerequisites}</dd>
                    </div>
                    <div>
                      <dt>Typical runtime</dt>
                      <dd>{example.runtime}</dd>
                    </div>
                    <div>
                      <dt>Run from repository root</dt>
                      <dd>
                        <code>{example.runCommand}</code>
                        <CopyButton label="Copy run command" text={example.runCommand} />
                      </dd>
                    </div>
                  </dl>

                  <div className="example-actions">
                    <a href={example.sourceHref}>
                      {`Full source · v${release.version}`} <span aria-hidden="true">↗</span>
                    </a>
                    <a href={example.guideHref}>
                      {example.guideLabel} <span aria-hidden="true">→</span>
                    </a>
                  </div>
                </div>

                <div className="code-window example-code-window">
                  <div className="code-titlebar">
                    <span className="window-dots" aria-hidden="true">
                      <i />
                      <i />
                      <i />
                    </span>
                    <span>{example.fileName}</span>
                    <CopyButton label="Copy code" text={example.code} />
                  </div>
                  <pre tabIndex={0} aria-label={`${example.title} Python source`}>
                    <PythonCode code={example.code} />
                  </pre>
                  <div className="example-result">
                    <div>
                      <span>Expected</span>
                      <small>{example.expectedNote}</small>
                    </div>
                    <pre>
                      <code>{example.expected}</code>
                    </pre>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <section className="run-notes" id="run-notes" aria-labelledby="run-heading">
          <div className="run-notes-inner shell">
            <div>
              <p className="kicker">Before you scale up</p>
              <h2 id="run-heading">Turn an example into your calculation.</h2>
              <p>
                Examples establish a known starting point. Keep the release or
                commit fixed while you change one physical or numerical choice
                at a time.
              </p>
            </div>
            <ol>
              <li>
                <span>01</span>
                <div>
                  <strong>Pin the software.</strong>
                  <p>
                    Install <code>{release.installCommand}</code> or work from
                    the v{release.version} source tag.
                  </p>
                </div>
              </li>
              <li>
                <span>02</span>
                <div>
                  <strong>Reproduce the stated result.</strong>
                  <p>
                    Run from the repository root so relative output paths and
                    local imports resolve as documented.
                  </p>
                </div>
              </li>
              <li>
                <span>03</span>
                <div>
                  <strong>Record what changes.</strong>
                  <p>
                    Preserve units, grids, basis sets, tolerances, optional
                    backends, and convergence checks with your output.
                  </p>
                </div>
              </li>
            </ol>
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
            <a href={links.userGuide}>User Guide</a>
            <a href={links.docs}>Stable docs</a>
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
