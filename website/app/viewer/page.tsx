import type { Metadata } from "next";
import Link from "next/link";
import { MoleculeViewer } from "./molecule-viewer";

export const metadata: Metadata = {
  title: "Molecular Orbitals and Density Viewer",
  description:
    "Explore molecular geometries, orbitals, electron and spin densities, difference fields, and electrostatic potential in the browser.",
  alternates: { canonical: "/viewer" },
  openGraph: {
    url: "https://pyqed.org/viewer",
    title: "Molecular Orbitals and Density Viewer | PyQED",
    description:
      "Rotate molecular structures and inspect every orbital or scalar-field state produced by PyQED.",
  },
};

export default function ViewerPage() {
  return (
    <>
      <a className="skip-link" href="#viewer-main">
        Skip to viewer
      </a>

      <header className="site-header">
        <nav className="site-nav shell viewer-nav" aria-label="Primary navigation">
          <Link className="wordmark" href="/" aria-label="PyQED home">
            <span className="wordmark-mark">P</span>
            <span>PyQED</span>
          </Link>
          <p>Molecular field viewer</p>
          <div className="nav-actions">
            <Link href="/examples">Examples</Link>
            <Link className="nav-github" href="/">
              Project home
            </Link>
          </div>
        </nav>
      </header>

      <main className="viewer-page" id="viewer-main">
        <section className="viewer-intro shell" aria-labelledby="viewer-heading">
          <div>
            <p className="kicker">PyQED laboratory</p>
            <h1 id="viewer-heading">See every state, not just the geometry.</h1>
          </div>
          <p>
            Open XYZ and Gaussian cube files, or send a scene directly from
            <code> view(...)</code>. Rotate the structure, compare every orbital or
            density state, and map electrostatic potential without uploading the data.
            Everything runs locally in your browser.
          </p>
        </section>

        <MoleculeViewer />
      </main>
    </>
  );
}
