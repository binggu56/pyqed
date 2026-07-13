import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

const projectRoot = new URL("../", import.meta.url);
const outputRoot = new URL("../out/", import.meta.url);

async function page(pathname = "/") {
  const route = pathname.replace(/^\/+|\/+$/g, "");
  return readFile(new URL(route ? `${route}/index.html` : "index.html", outputRoot), "utf8");
}

function section(html, id) {
  const match = html.match(
    new RegExp(`<section[^>]*id="${id}"[^>]*>[\\s\\S]*?<\\/section>`, "i"),
  );
  assert.ok(match, `section #${id} should be rendered`);
  return match[0];
}

test("exports the research-first PyQED homepage", async () => {
  const html = await page();

  assert.match(
    html,
    /<title>PyQED[^<]*Electronic Structure and Quantum Dynamics in Python<\/title>/i,
  );
  assert.match(html, /<main[^>]*id="main-content"/i);
  assert.match(html, /<nav[^>]*aria-label="Primary navigation"/i);
  assert.match(
    html,
    /<h1[^>]*>[\s\S]*Electronic structure and[\s\S]*quantum dynamics in Python\./i,
  );
  assert.match(html, /python -m pip install pyqed==0\.2\.0/i);
  assert.match(html, /Open the User Guide/i);
  assert.match(
    html,
    /href="https:\/\/docs\.pyqed\.org\/en\/latest\/guide\/guide\.html"[^>]*>User Guide/i,
  );
  assert.match(html, /href="\/examples\/?"[^>]*>Examples/i);

  const workflows = section(html, "workflows");
  assert.equal((workflows.match(/data-workflow=/gi) ?? []).length, 4);
  for (const slug of [
    "electronic-structure",
    "nonadiabatic-dynamics",
    "open-systems",
    "tensor-networks",
  ]) {
    assert.match(workflows, new RegExp(`data-workflow="${slug}"`, "i"));
  }

  const research = section(html, "research");
  assert.equal((research.match(/<figure\b/gi) ?? []).length, 3);
  assert.equal((research.match(/data-research-source/gi) ?? []).length, 3);
  assert.equal((research.match(/loading="lazy"/gi) ?? []).length, 3);
  assert.doesNotMatch(research, /alt=""/i);

  assert.match(html, /Bing Gu/i);
  assert.doesNotMatch(html, /Zihao Chen/i);
  assert.match(html, /RHF energy: -1\.11675931 Ha/i);
  assert.match(html, /10\.1063\/1674-0068\/cjcp2510161/i);
  assert.match(html, /10\.5281\/zenodo\.21316543/i);
  assert.match(html, /10\.5281\/zenodo\.21316544/i);
  assert.doesNotMatch(html, /chatgpt\.site|codex-preview|Your site is taking shape/i);

  const jsonLdMatch = html.match(
    /<script[^>]*type="application\/ld\+json"[^>]*>([^<]+)<\/script>/i,
  );
  assert.ok(jsonLdMatch, "software JSON-LD should be rendered");
  const jsonLd = JSON.parse(jsonLdMatch[1]);
  assert.equal(jsonLd.softwareVersion, "0.2.0");
  assert.equal(jsonLd.maintainer.name, "Bing Gu");
  assert.equal(jsonLd.identifier.value, "10.5281/zenodo.21316544");
  assert.equal(jsonLd.isPartOf.identifier.value, "10.5281/zenodo.21316543");
});

test("exports the release-pinned examples library", async () => {
  const html = await page("/examples");

  assert.match(html, /<title>Runnable Python Examples \| PyQED<\/title>/i);
  assert.match(html, /rel="canonical" href="https:\/\/pyqed\.org\/examples\/?"/i);
  assert.match(html, /Read it\. Run it\./i);
  assert.match(html, /Follow the source\./i);
  assert.equal((html.match(/data-example=/gi) ?? []).length, 4);
  assert.equal((html.match(/class="python-code"/gi) ?? []).length, 4);
  assert.ok(
    (html.match(/class="code-line"/gi) ?? []).length > 60,
    "examples should render readable source lines",
  );

  for (const [id, title] of [
    ["h2-rhf", "H₂ in one native RHF calculation"],
    ["sine-dvr-oscillator", "A harmonic oscillator with Sine DVR"],
    ["heom-spin-boson", "Spin–boson dynamics with HEOM"],
    ["shin-metiu-ehrenfest", "Shin–Metiu histories with Ehrenfest dynamics"],
  ]) {
    assert.match(html, new RegExp(`data-example="${id}"`, "i"));
    assert.match(html, new RegExp(title, "i"));
  }

  assert.match(html, /Final &lt;sigma_z&gt;: -0\.96907844/i);
  assert.match(html, /Copy code/i);
  assert.match(html, /Copy run command/i);
  assert.match(html, /Full source · v0\.2\.0/i);
  assert.doesNotMatch(html, /chatgpt\.site/i);
});

test("uses a static, repository-owned deployment", async () => {
  const [layout, copyButton, privacy, nextConfig, packageText, workflow] =
    await Promise.all([
      readFile(new URL("app/layout.tsx", projectRoot), "utf8"),
      readFile(new URL("app/copy-button.tsx", projectRoot), "utf8"),
      readFile(new URL("app/privacy/page.tsx", projectRoot), "utf8"),
      readFile(new URL("next.config.ts", projectRoot), "utf8"),
      readFile(new URL("package.json", projectRoot), "utf8"),
      readFile(new URL("../.github/workflows/pages.yml", projectRoot), "utf8"),
    ]);
  const packageJson = JSON.parse(packageText);

  assert.match(nextConfig, /output:\s*"export"/);
  assert.match(nextConfig, /trailingSlash:\s*true/);
  assert.match(nextConfig, /unoptimized:\s*true/);
  assert.match(packageJson.scripts.build, /^next build$/);
  assert.doesNotMatch(packageText, /vinext|wrangler|drizzle|cloudflare/i);
  assert.doesNotMatch(layout, /Analytics/);
  assert.doesNotMatch(copyButton, /sendAnalyticsEvent|sendBeacon|fetch\(/);
  assert.match(privacy, /no visitor tracking/i);
  assert.match(workflow, /actions\/deploy-pages@v4/);
  assert.match(workflow, /path:\s*website\/out/);

  for (const artifact of [
    "index.html",
    "examples/index.html",
    "privacy/index.html",
    "robots.txt",
    "sitemap.xml",
    "icon.png",
    "og-v2.png",
    "research/so2-state-wavepackets.png",
    "schemas/benchmark-manifest-1.0.json",
    "34951641-C871-4E69-B139-BB78ADE2FE0F.txt",
  ]) {
    await access(new URL(artifact, outputRoot));
  }

  for (const removedRuntime of [
    "app/api",
    "app/analytics.tsx",
    "db",
    "drizzle",
    "worker",
    ".openai/hosting.json",
    "vite.config.ts",
  ]) {
    await assert.rejects(access(new URL(removedRuntime, projectRoot)));
  }
});

test("exports an accurate privacy page", async () => {
  const html = await page("/privacy");

  assert.match(html, /A static website with no visitor tracking/i);
  assert.match(html, /does not set project cookies/i);
  assert.match(html, /GitHub Pages and network providers/i);
  assert.match(html, /rel="canonical" href="https:\/\/pyqed\.org\/privacy\/?"/i);
  assert.doesNotMatch(html, /api\/analytics|400 days|unique people/i);
});
