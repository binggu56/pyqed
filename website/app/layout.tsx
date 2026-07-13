import type { Metadata } from "next";
import "./globals.css";
import { paper, release } from "./site-data";

export const metadata: Metadata = {
  metadataBase: new URL("https://pyqed.org"),
  title: {
    default: "PyQED — Electronic Structure and Quantum Dynamics in Python",
    template: "%s | PyQED",
  },
  description:
    "Research software for electronic structure, nonadiabatic dynamics, open quantum systems, spectroscopy, and tensor-network methods.",
  alternates: {
    canonical: "/",
  },
  icons: {
    icon: [{ url: "/icon.png", type: "image/png", sizes: "128x128" }],
  },
  openGraph: {
    type: "website",
    url: "https://pyqed.org",
    siteName: "PyQED",
    title: "PyQED — Electronic Structure and Quantum Dynamics in Python",
    description:
      `PyQED ${release.version}: electronic structure, quantum dynamics, spectroscopy, and many-body methods in an open-source Python framework.`,
    images: [
      {
        url: "/og-v2.png",
        width: 1200,
        height: 630,
        alt: "PyQED — Electronic structure and quantum dynamics in Python.",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    title: "PyQED — Electronic Structure and Quantum Dynamics in Python",
    description:
      `PyQED ${release.version}: research software for electronic structure and quantum dynamics.`,
    images: ["/og-v2.png"],
  },
  keywords: [
    "quantum dynamics",
    "electronic structure",
    "nonadiabatic dynamics",
    "open quantum systems",
    "tensor networks",
    "computational chemistry",
    "Python",
  ],
  other: {
    citation_title: paper.title,
    citation_author: [...paper.authors],
    citation_journal_title: paper.journal,
    citation_publication_date: String(paper.year),
    citation_doi: paper.doi,
    citation_public_url: paper.url,
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
