import type { Metadata } from "next";
import Link from "next/link";
import { MoleculeViewer } from "./molecule-viewer";

export const metadata: Metadata = {
  title: "Interactive Molecular Viewer",
  description:
    "Explore molecular geometries in the browser and export coordinates for PyQED calculations.",
  alternates: { canonical: "/viewer" },
  openGraph: {
    url: "https://pyqed.org/viewer",
    title: "Interactive Molecular Viewer | PyQED",
    description:
      "Rotate, inspect, and export molecular geometries for PyQED calculations.",
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
          <p>Molecular viewer</p>
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
            <h1 id="viewer-heading">See the geometry before you calculate.</h1>
          </div>
          <p>
            Paste PyQED coordinates or open an XYZ file. Rotate, zoom, inspect
            inferred bonds, and export a clean geometry for your next calculation.
            Everything runs locally in your browser.
          </p>
        </section>

        <MoleculeViewer />
      </main>
    </>
  );
}
