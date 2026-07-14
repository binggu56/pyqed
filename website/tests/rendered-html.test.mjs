import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";
import {
  BOHR_TO_ANGSTROM,
  MAX_GRID_POINTS,
  fieldToCube,
  parseCube,
  parseGeometry,
  validateSceneMessage,
} from "../app/viewer/volume-data.mjs";

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

function float32Base64(values) {
  const buffer = new ArrayBuffer(values.length * Float32Array.BYTES_PER_ELEMENT);
  const view = new DataView(buffer);
  values.forEach((value, index) => {
    view.setFloat32(index * Float32Array.BYTES_PER_ELEMENT, value, true);
  });
  return Buffer.from(buffer).toString("base64");
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
  assert.match(html, /href="\/viewer\/?"[^>]*>Viewer/i);

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

test("exports the interactive molecular viewer workspace", async () => {
  const html = await page("/viewer");

  assert.match(html, /<title>Molecular Orbitals and Density Viewer \| PyQED<\/title>/i);
  assert.match(
    html,
    /rel="canonical" href="https:\/\/pyqed\.org\/viewer\/?"/i,
  );
  assert.match(html, /See every state, not just the geometry/i);
  assert.match(html, /Molecular geometry workspace/i);
  assert.match(html, /XYZ or PyQED atom string/i);
  assert.match(html, /Open XYZ \/ cube/i);
  assert.match(html, /Scalar fields/i);
  assert.match(html, /Field \/ state/i);
  assert.match(html, /Molecule only/i);
  assert.match(html, /Multi-state files remain selectable/i);
  assert.match(html, /never enter the page URL/i);
  assert.match(html, /Ball &amp; stick/i);
  assert.match(html, /Auto-rotate/i);
  assert.match(html, /Export XYZ/i);
  assert.match(html, /Everything runs locally in your browser/i);
});

test("parses Python XYZ comments without treating their semicolons as atom separators", () => {
  const atoms = parseGeometry(`2
Generated by PyQED; coordinates in angstrom
H    0.0000000000  0.0000000000  0.0000000000
H    0.0000000000  0.0000000000  0.7400000000`);

  assert.equal(atoms.length, 2);
  assert.equal(atoms[1].element, "H");
  assert.equal(atoms[1].z, 0.74);
  assert.equal(parseGeometry("H 0 0 0; F 0 0 0.917").length, 2);
  assert.throws(() => parseGeometry("Qq 0 0 0"), /unsupported element/i);
  assert.throws(() => parseGeometry("H 4e38 0 0"), /32-bit floating-point/i);
});

test("validates encoded scalar fields and ESP density mappings", () => {
  const density = {
    name: "rho-s0",
    label: "S0 electron density",
    kind: "electron-density",
    shape: [2, 2, 2],
    origin: [-1, -1, -1],
    axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    values: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    isovalues: [0.05],
    colors: ["#2563eb"],
    units: "bohr^-3",
    metadata: { state: 0, source: "one-particle density matrix" },
  };
  const esp = {
    name: "esp-s0",
    label: "S0 electrostatic potential",
    kind: "esp",
    shape: [2, 2, 2],
    origin: [-1, -1, -1],
    axes: [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
    values: float32Base64([-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4]),
    value_encoding: "float32-le-base64",
    surface_field: "rho-s0",
    metadata: { source: "cube", dataset: 0 },
  };
  const scene = validateSceneMessage({
    type: "pyqed:scene",
    scene: {
      version: 1,
      kind: "pyqed-scene",
      title: "H field set",
      molecule: { xyz: "H 0 0 0", representation: "ball-stick", labels: true },
      fields: [density, esp],
      active_field: "esp-s0",
    },
  });

  assert.equal(scene.activeFieldIndex, 1);
  assert.equal(scene.fields.length, 2);
  assert.ok(scene.fields[0].values instanceof Float32Array);
  assert.ok(scene.fields[1].values instanceof Float32Array);
  assert.ok(Math.abs(scene.fields[1].values[0] + 0.4) < 1e-6);
  assert.equal(scene.fields[1].metadata.source, "cube");
  assert.equal(scene.fields[1].surface_field, "rho-s0");

  const exactEsp = validateSceneMessage({
    type: "pyqed:scene",
    scene: {
      version: 1,
      kind: "pyqed-scene",
      molecule: null,
      fields: [{
        ...esp,
        values: [-0.4, -0.3, -0.2, -0.1, 0.1, 0.2, 0.3, 0.4],
        value_encoding: undefined,
        surface_field: undefined,
        metadata: { method: "pyscf-exact", approximate: false, units: "hartree/e" },
      }],
    },
  });
  assert.equal(exactEsp.fields[0].metadata.method, "pyscf-exact");

  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: { xyz: "H 0 0 0" },
        fields: [],
        script: "alert(1)",
      },
    }),
    /unsupported property/i,
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: { xyz: "H 0 0 0" },
        fields: [{ ...density, values: [0, 1] }],
      },
    }),
    /C-order shape.*requires 8/i,
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: { xyz: "H 0 0 0" },
        fields: [{ ...density, shape: [200, 200, 51], values: [] }],
      },
    }),
    new RegExp(`limit is ${MAX_GRID_POINTS.toLocaleString()}`),
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: { xyz: "H 0 0 0" },
        fields: [density, { ...esp, origin: [0, 0, 0] }],
      },
    }),
    /same grid/i,
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: null,
        fields: [{ ...esp, values: "AAAA" }],
      },
    }),
    /base64 characters.*require/i,
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: null,
        fields: [{
          ...esp,
          values: float32Base64([NaN, 0, 0, 0, 0, 0, 0, 0]),
          surface_field: undefined,
        }],
      },
    }),
    /finite 32-bit floating-point/i,
  );
  assert.throws(
    () => validateSceneMessage({
      type: "pyqed:scene",
      scene: {
        version: 1,
        kind: "pyqed-scene",
        molecule: null,
        fields: [{
          ...esp,
          surface_field: undefined,
          metadata: { source: "cube", dataset: -1 },
        }],
      },
    }),
    /non-negative integer/i,
  );
});

test("parses single and multi-state Gaussian cube files in C order", () => {
  const singleCube = `Water field
MO test
    1 0.0 0.0 0.0
    2 1.0 0.0 0.0
    2 0.0 1.0 0.0
    2 0.0 0.0 1.0
    1 1.0 0.0 0.0 0.0
0 1 2 3 4 5 6 7
`;
  const single = parseCube(singleCube, { fileName: "homo.cube" });
  assert.equal(single.fields.length, 1);
  assert.equal(single.fields[0].kind, "orbital");
  assert.deepEqual([...single.fields[0].values], [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.ok(Math.abs(single.fields[0].axes[0][0] - BOHR_TO_ANGSTROM) < 1e-12);

  const multiCube = `Two orbital field
Molecular orbitals
   -1 0.0 0.0 0.0
    2 1.0 0.0 0.0
    2 0.0 1.0 0.0
    2 0.0 0.0 1.0
    1 1.0 0.0 0.0 0.0
2 5 8
0 10 1 11 2 12 3 13 4 14 5 15 6 16 7 17
`;
  const multi = parseCube(multiCube, { fileName: "all-mo.cube" });
  assert.equal(multi.fields.length, 2);
  assert.equal(multi.fields[0].label, "Orbital 5");
  assert.equal(multi.fields[1].label, "Orbital 8");
  assert.deepEqual([...multi.fields[0].values], [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual([...multi.fields[1].values], [10, 11, 12, 13, 14, 15, 16, 17]);

  const roundTrip = parseCube(fieldToCube(single.fields[0], single.atoms));
  assert.deepEqual([...roundTrip.fields[0].shape], [2, 2, 2]);
  assert.deepEqual([...roundTrip.fields[0].values], [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.throws(() => parseCube(singleCube.replace("0 1 2 3 4 5 6 7", "0 1")), /requires 8/i);
});

test("uses a static, repository-owned deployment", async () => {
  const [
    layout,
    copyButton,
    privacy,
    nextConfig,
    packageText,
    workflow,
    moleculeViewer,
    sitemap,
    css,
  ] =
    await Promise.all([
      readFile(new URL("app/layout.tsx", projectRoot), "utf8"),
      readFile(new URL("app/copy-button.tsx", projectRoot), "utf8"),
      readFile(new URL("app/privacy/page.tsx", projectRoot), "utf8"),
      readFile(new URL("next.config.ts", projectRoot), "utf8"),
      readFile(new URL("package.json", projectRoot), "utf8"),
      readFile(new URL("../.github/workflows/pages.yml", projectRoot), "utf8"),
      readFile(new URL("app/viewer/molecule-viewer.tsx", projectRoot), "utf8"),
      readFile(new URL("app/sitemap.ts", projectRoot), "utf8"),
      readFile(new URL("app/globals.css", projectRoot), "utf8"),
    ]);
  const packageJson = JSON.parse(packageText);

  assert.match(nextConfig, /output:\s*"export"/);
  assert.match(nextConfig, /trailingSlash:\s*true/);
  assert.match(nextConfig, /unoptimized:\s*true/);
  assert.match(packageJson.scripts.build, /^next build$/);
  assert.equal(packageJson.dependencies.next, "16.2.10");
  assert.equal(packageJson.dependencies["3dmol"], "2.5.5");
  assert.doesNotMatch(packageText, /vinext|wrangler|drizzle|cloudflare/i);
  assert.doesNotMatch(layout, /Analytics/);
  assert.doesNotMatch(copyButton, /sendAnalyticsEvent|sendBeacon|fetch\(/);
  assert.match(privacy, /no visitor tracking/i);
  assert.match(workflow, /actions\/deploy-pages@v4/);
  assert.match(workflow, /path:\s*website\/out/);
  assert.match(sitemap, /https:\/\/pyqed\.org\/viewer/);
  assert.match(moleculeViewer, /window\.location\.hash/);
  assert.match(moleculeViewer, /event\.source === window\.parent/);
  assert.match(moleculeViewer, /validateSceneMessage\(event\.data\)/);
  assert.match(moleculeViewer, /import\("3dmol"\)/);
  assert.match(moleculeViewer, /pyqed:viewer-ready/);
  assert.ok(
    moleculeViewer.indexOf('window.addEventListener("message", receiveScene)') <
      moleculeViewer.indexOf('type: "pyqed:viewer-ready"'),
    "the scene listener should be installed before the ready handshake",
  );
  assert.match(moleculeViewer, /WeakMap<VolumeField, unknown>/);
  assert.equal((moleculeViewer.match(/new library\.VolumeData/g) ?? []).length, 1);
  assert.match(
    moleculeViewer,
    /\}, \[\s*activeFieldIndex,\s*atoms,\s*fields,\s*isovalue,\s*rendererStatus,\s*surfaceMode,\s*\]\);/,
  );
  assert.match(moleculeViewer, /useDeferredValue\(isovalue\)/);
  assert.match(moleculeViewer, /prefers-reduced-motion: reduce/);
  assert.doesNotMatch(moleculeViewer, /https?:\/\/.*3dmol/i);
  assert.match(css, /\.file-button:focus-within/);
  assert.match(css, /animation-duration:\s*0\.01ms\s*!important/);

  for (const artifact of [
    "index.html",
    "examples/index.html",
    "viewer/index.html",
    "privacy/index.html",
    "robots.txt",
    "sitemap.xml",
    "icon.png",
    "og-v2.png",
    "research/so2-state-wavepackets.png",
    "schemas/benchmark-manifest-1.0.json",
    "licenses/3dmol-2.5.5-LICENSE.txt",
    "34951641-C871-4E69-B139-BB78ADE2FE0F.txt",
  ]) {
    await access(new URL(artifact, outputRoot));
  }

  const viewerLicense = await readFile(
    new URL("licenses/3dmol-2.5.5-LICENSE.txt", outputRoot),
    "utf8",
  );
  assert.match(viewerLicense, /Copyright \(c\) 2014, University of Pittsburgh/);
  assert.match(viewerLicense, /\* GLmol/);
  assert.match(viewerLicense, /\* Three\.js/);
  assert.match(viewerLicense, /\* jQuery/);
  assert.match(viewerLicense, /THIS SOFTWARE IS PROVIDED[\s\S]+AS IS/);

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
