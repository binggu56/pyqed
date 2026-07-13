import type { Metadata } from "next";
import Link from "next/link";

export const metadata: Metadata = {
  title: "Privacy",
  description: "Privacy information for the static PyQED project website.",
  alternates: { canonical: "/privacy" },
};

export default function PrivacyPage() {
  return (
    <main className="policy-page">
      <article className="policy-card">
        <p className="kicker">Privacy</p>
        <h1>A static website with no visitor tracking.</h1>
        <p>
          The PyQED project homepage does not run project analytics, create
          visitor profiles, or store information about individual visitors.
        </p>
        <h2>What the project does not collect</h2>
        <p>
          PyQED does not set project cookies or collect IP addresses, browser
          details, referrers, query strings, session identifiers, or personal
          data through this website.
        </p>
        <h2>Infrastructure logs</h2>
        <p>
          GitHub Pages and network providers may process ordinary technical
          request logs under their own policies to operate and secure the
          service. Those logs are not available to the PyQED website code.
        </p>
        <Link className="button button-ink" href="/">
          Return to PyQED
        </Link>
      </article>
    </main>
  );
}
