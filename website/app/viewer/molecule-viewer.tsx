"use client";

import type {
  ChangeEvent,
  PointerEvent as ReactPointerEvent,
  WheelEvent as ReactWheelEvent,
} from "react";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

type Atom = {
  element: string;
  x: number;
  y: number;
  z: number;
};

type Bond = [number, number];
type RenderMode = "ball-stick" | "space-fill" | "wireframe";

const BOHR_TO_ANGSTROM = 0.529177210903;

const ELEMENTS: Record<
  string,
  { color: string; radius: number; covalent: number; name: string }
> = {
  H: { color: "#f4f5f6", radius: 0.31, covalent: 0.31, name: "Hydrogen" },
  B: { color: "#f4a67c", radius: 0.84, covalent: 0.84, name: "Boron" },
  C: { color: "#475965", radius: 0.76, covalent: 0.76, name: "Carbon" },
  N: { color: "#5085e8", radius: 0.71, covalent: 0.71, name: "Nitrogen" },
  O: { color: "#ef5350", radius: 0.66, covalent: 0.66, name: "Oxygen" },
  F: { color: "#74c985", radius: 0.57, covalent: 0.57, name: "Fluorine" },
  P: { color: "#e29a45", radius: 1.07, covalent: 1.07, name: "Phosphorus" },
  S: { color: "#e8cb4b", radius: 1.05, covalent: 1.05, name: "Sulfur" },
  Cl: { color: "#62bd6b", radius: 1.02, covalent: 1.02, name: "Chlorine" },
  Fe: { color: "#bf6a43", radius: 1.32, covalent: 1.32, name: "Iron" },
  Br: { color: "#a34f45", radius: 1.20, covalent: 1.20, name: "Bromine" },
  I: { color: "#8b63b8", radius: 1.39, covalent: 1.39, name: "Iodine" },
};

const FALLBACK_ELEMENT = {
  color: "#8fa6b0",
  radius: 0.9,
  covalent: 0.9,
  name: "Element",
};

const PRESETS = {
  Water: `3
Water · H2O
O  0.000000  0.000000  0.000000
H  0.000000 -0.757000  0.587000
H  0.000000  0.757000  0.587000`,
  Methane: `5
Methane · CH4
C  0.000000  0.000000  0.000000
H  0.629118  0.629118  0.629118
H -0.629118 -0.629118  0.629118
H -0.629118  0.629118 -0.629118
H  0.629118 -0.629118 -0.629118`,
  Benzene: `12
Benzene · C6H6
C  1.396000  0.000000  0.000000
C  0.698000  1.209000  0.000000
C -0.698000  1.209000  0.000000
C -1.396000  0.000000  0.000000
C -0.698000 -1.209000  0.000000
C  0.698000 -1.209000  0.000000
H  2.479000  0.000000  0.000000
H  1.240000  2.147000  0.000000
H -1.240000  2.147000  0.000000
H -2.479000  0.000000  0.000000
H -1.240000 -2.147000  0.000000
H  1.240000 -2.147000  0.000000`,
  "Hydrogen fluoride": `2
Hydrogen fluoride · HF
H 0.000000 0.000000 0.000000
F 0.000000 0.000000 0.917000`,
} as const;

function canonicalElement(value: string): string {
  const trimmed = value.trim();
  if (!/^[A-Za-z]{1,2}$/.test(trimmed)) {
    throw new Error(`Invalid element symbol “${value}”.`);
  }
  return trimmed[0].toUpperCase() + trimmed.slice(1).toLowerCase();
}

export function parseGeometry(text: string, unit: "angstrom" | "bohr"): Atom[] {
  const normalized = text.replaceAll(";", "\n").trim();
  if (!normalized) throw new Error("Enter at least one atom.");

  const rawLines = normalized.split(/\r?\n/).map((line) => line.trim());
  const firstIsCount = /^\d+$/.test(rawLines[0]);
  const expectedCount = firstIsCount ? Number(rawLines[0]) : null;
  const coordinateLines = firstIsCount ? rawLines.slice(2) : rawLines;
  const scale = unit === "bohr" ? BOHR_TO_ANGSTROM : 1;

  const atoms = coordinateLines
    .filter(Boolean)
    .map((line, index) => {
      const fields = line.split(/[\s,]+/);
      if (fields.length !== 4) {
        throw new Error(
          `Line ${index + (firstIsCount ? 3 : 1)} must contain an element and three coordinates.`,
        );
      }
      const [symbol, ...coordinateText] = fields;
      const coordinates = coordinateText.map(Number);
      if (coordinates.some((value) => !Number.isFinite(value))) {
        throw new Error(`Line ${index + (firstIsCount ? 3 : 1)} contains an invalid number.`);
      }
      return {
        element: canonicalElement(symbol),
        x: coordinates[0] * scale,
        y: coordinates[1] * scale,
        z: coordinates[2] * scale,
      };
    });

  if (atoms.length === 0) throw new Error("No coordinate rows were found.");
  if (atoms.length > 500) throw new Error("The viewer currently supports up to 500 atoms.");
  if (expectedCount !== null && expectedCount !== atoms.length) {
    throw new Error(`XYZ header declares ${expectedCount} atoms, but ${atoms.length} were found.`);
  }
  return atoms;
}

function inferBonds(atoms: Atom[]): Bond[] {
  const bonds: Bond[] = [];
  for (let i = 0; i < atoms.length; i += 1) {
    for (let j = i + 1; j < atoms.length; j += 1) {
      const a = atoms[i];
      const b = atoms[j];
      const distance = Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);
      const ra = (ELEMENTS[a.element] ?? FALLBACK_ELEMENT).covalent;
      const rb = (ELEMENTS[b.element] ?? FALLBACK_ELEMENT).covalent;
      if (distance > 0.1 && distance <= (ra + rb) * 1.22) bonds.push([i, j]);
    }
  }
  return bonds;
}

function formulaFor(atoms: Atom[]): string {
  const counts = new Map<string, number>();
  for (const atom of atoms) counts.set(atom.element, (counts.get(atom.element) ?? 0) + 1);
  const symbols = [...counts.keys()].sort((a, b) => {
    const order = (symbol: string) => (symbol === "C" ? 0 : symbol === "H" ? 1 : 2);
    return order(a) - order(b) || a.localeCompare(b);
  });
  return symbols.map((symbol) => `${symbol}${counts.get(symbol) === 1 ? "" : counts.get(symbol)}`).join("");
}

function xyzFor(atoms: Atom[]): string {
  const rows = atoms.map(
    ({ element, x, y, z }) =>
      `${element.padEnd(2)} ${x.toFixed(8).padStart(13)} ${y.toFixed(8).padStart(13)} ${z.toFixed(8).padStart(13)}`,
  );
  return `${atoms.length}\nExported from the PyQED molecular viewer · coordinates in angstrom\n${rows.join("\n")}\n`;
}

function MoleculeCanvas({
  atoms,
  bonds,
  mode,
  labels,
  autoRotate,
  resetSignal,
}: {
  atoms: Atom[];
  bonds: Bond[];
  mode: RenderMode;
  labels: boolean;
  autoRotate: boolean;
  resetSignal: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const rotation = useRef({ x: -0.28, y: 0.55 });
  const zoom = useRef(1);
  const drag = useRef<{ x: number; y: number; pointerId: number } | null>(null);
  const animationFrame = useRef<number | null>(null);
  const autoRotateRef = useRef(autoRotate);

  useEffect(() => {
    autoRotateRef.current = autoRotate;
  }, [autoRotate]);

  useEffect(() => {
    rotation.current = { x: -0.28, y: 0.55 };
    zoom.current = 1;
  }, [resetSignal, atoms]);

  const draw = useCallback(function drawFrame() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    const pixelRatio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * pixelRatio));
    const height = Math.max(1, Math.round(rect.height * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);

    const cssWidth = rect.width;
    const cssHeight = rect.height;
    const gradient = context.createRadialGradient(
      cssWidth * 0.52,
      cssHeight * 0.46,
      0,
      cssWidth * 0.5,
      cssHeight * 0.5,
      Math.max(cssWidth, cssHeight) * 0.72,
    );
    gradient.addColorStop(0, "#132d3b");
    gradient.addColorStop(0.62, "#091923");
    gradient.addColorStop(1, "#06131c");
    context.fillStyle = gradient;
    context.fillRect(0, 0, cssWidth, cssHeight);

    if (autoRotateRef.current && !drag.current) rotation.current.y += 0.0035;

    const center = atoms.reduce(
      (sum, atom) => ({ x: sum.x + atom.x, y: sum.y + atom.y, z: sum.z + atom.z }),
      { x: 0, y: 0, z: 0 },
    );
    center.x /= atoms.length;
    center.y /= atoms.length;
    center.z /= atoms.length;

    const spread = Math.max(
      1.2,
      ...atoms.map((atom) => Math.hypot(atom.x - center.x, atom.y - center.y, atom.z - center.z)),
    );
    const scale = (Math.min(cssWidth, cssHeight) * 0.34 * zoom.current) / spread;
    const sinX = Math.sin(rotation.current.x);
    const cosX = Math.cos(rotation.current.x);
    const sinY = Math.sin(rotation.current.y);
    const cosY = Math.cos(rotation.current.y);

    const projected = atoms.map((atom, index) => {
      const x0 = atom.x - center.x;
      const y0 = atom.y - center.y;
      const z0 = atom.z - center.z;
      const x1 = x0 * cosY + z0 * sinY;
      const z1 = -x0 * sinY + z0 * cosY;
      const y1 = y0 * cosX - z1 * sinX;
      const z2 = y0 * sinX + z1 * cosX;
      return {
        index,
        x: cssWidth / 2 + x1 * scale,
        y: cssHeight / 2 - y1 * scale,
        z: z2,
      };
    });

    const bondWidth = mode === "wireframe" ? 2 : Math.max(4, Math.min(10, scale * 0.055));
    const sortedBonds = bonds
      .map(([a, b]) => ({ a: projected[a], b: projected[b], depth: (projected[a].z + projected[b].z) / 2 }))
      .sort((left, right) => left.depth - right.depth);

    for (const bond of sortedBonds) {
      const midpointX = (bond.a.x + bond.b.x) / 2;
      const midpointY = (bond.a.y + bond.b.y) / 2;
      context.lineCap = "round";
      context.lineWidth = bondWidth;
      context.strokeStyle = mode === "wireframe" ? "rgba(169, 225, 231, 0.72)" : "#80939c";
      context.beginPath();
      context.moveTo(bond.a.x, bond.a.y);
      context.lineTo(midpointX, midpointY);
      context.stroke();
      context.strokeStyle = mode === "wireframe" ? "rgba(169, 225, 231, 0.72)" : "#b4c1c5";
      context.beginPath();
      context.moveTo(midpointX, midpointY);
      context.lineTo(bond.b.x, bond.b.y);
      context.stroke();
    }

    if (mode !== "wireframe") {
      const sortedAtoms = [...projected].sort((a, b) => a.z - b.z);
      for (const point of sortedAtoms) {
        const atom = atoms[point.index];
        const element = ELEMENTS[atom.element] ?? FALLBACK_ELEMENT;
        const radiusScale = mode === "space-fill" ? 0.42 + element.radius * 0.5 : 0.34;
        const radius = Math.max(8, Math.min(42, scale * radiusScale));
        const sphere = context.createRadialGradient(
          point.x - radius * 0.32,
          point.y - radius * 0.38,
          radius * 0.08,
          point.x,
          point.y,
          radius,
        );
        sphere.addColorStop(0, "#ffffff");
        sphere.addColorStop(0.18, element.color);
        sphere.addColorStop(1, "#101a20");
        context.fillStyle = sphere;
        context.beginPath();
        context.arc(point.x, point.y, radius, 0, Math.PI * 2);
        context.fill();
        context.strokeStyle = "rgba(255, 255, 255, 0.22)";
        context.lineWidth = 1;
        context.stroke();
      }
    }

    if (labels) {
      context.font = "700 12px ui-sans-serif, system-ui, sans-serif";
      context.textAlign = "center";
      context.textBaseline = "middle";
      for (const point of projected) {
        const label = `${atoms[point.index].element}${point.index + 1}`;
        const metrics = context.measureText(label);
        const labelWidth = metrics.width + 10;
        context.fillStyle = "rgba(3, 15, 22, 0.78)";
        context.fillRect(point.x - labelWidth / 2, point.y - 10, labelWidth, 20);
        context.fillStyle = "#e7f7f8";
        context.fillText(label, point.x, point.y + 0.5);
      }
    }

    context.fillStyle = "rgba(168, 238, 242, 0.62)";
    context.font = "600 11px ui-monospace, monospace";
    context.textAlign = "left";
    context.fillText("DRAG TO ROTATE  ·  SCROLL TO ZOOM", 18, cssHeight - 18);
    animationFrame.current = window.requestAnimationFrame(drawFrame);
  }, [atoms, bonds, labels, mode]);

  useEffect(() => {
    animationFrame.current = window.requestAnimationFrame(draw);
    return () => {
      if (animationFrame.current !== null) window.cancelAnimationFrame(animationFrame.current);
    };
  }, [draw]);

  function handlePointerDown(event: ReactPointerEvent<HTMLCanvasElement>) {
    event.currentTarget.setPointerCapture(event.pointerId);
    drag.current = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
  }

  function handlePointerMove(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (!drag.current || drag.current.pointerId !== event.pointerId) return;
    rotation.current.y += (event.clientX - drag.current.x) * 0.009;
    rotation.current.x = Math.max(
      -Math.PI / 2,
      Math.min(Math.PI / 2, rotation.current.x + (event.clientY - drag.current.y) * 0.009),
    );
    drag.current = { x: event.clientX, y: event.clientY, pointerId: event.pointerId };
  }

  function handlePointerUp(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (drag.current?.pointerId === event.pointerId) drag.current = null;
  }

  function handleWheel(event: ReactWheelEvent<HTMLCanvasElement>) {
    event.preventDefault();
    zoom.current = Math.max(0.45, Math.min(3.5, zoom.current * Math.exp(-event.deltaY * 0.001)));
  }

  return (
    <canvas
      ref={canvasRef}
      className="molecule-canvas"
      aria-label={`Interactive three-dimensional molecular view with ${atoms.length} atoms and ${bonds.length} inferred bonds`}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerUp={handlePointerUp}
      onPointerCancel={handlePointerUp}
      onWheel={handleWheel}
    />
  );
}

export function MoleculeViewer() {
  const [source, setSource] = useState<string>(PRESETS.Water);
  const [unit, setUnit] = useState<"angstrom" | "bohr">("angstrom");
  const [atoms, setAtoms] = useState<Atom[]>(() => parseGeometry(PRESETS.Water, "angstrom"));
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<RenderMode>("ball-stick");
  const [labels, setLabels] = useState(false);
  const [autoRotate, setAutoRotate] = useState(true);
  const [resetSignal, setResetSignal] = useState(0);
  const bonds = useMemo(() => inferBonds(atoms), [atoms]);
  const formula = useMemo(() => formulaFor(atoms), [atoms]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const parameters = new URLSearchParams(window.location.hash.slice(1));
      const geometry = parameters.get("xyz");
      if (!geometry) return;

      try {
        setAtoms(parseGeometry(geometry, "angstrom"));
        setSource(geometry);
        setUnit("angstrom");
        setError(null);
        const requestedMode = parameters.get("representation");
        if (
          requestedMode === "ball-stick" ||
          requestedMode === "space-fill" ||
          requestedMode === "wireframe"
        ) {
          setMode(requestedMode);
        }
        setLabels(parameters.get("labels") === "1");
        setResetSignal((value) => value + 1);
      } catch (parseError) {
        setError(
          parseError instanceof Error
            ? `Could not load the linked geometry: ${parseError.message}`
            : "Could not load the linked geometry.",
        );
      }
    });

    return () => window.cancelAnimationFrame(frame);
  }, []);

  function updateGeometry(nextSource = source, nextUnit = unit) {
    try {
      setAtoms(parseGeometry(nextSource, nextUnit));
      setError(null);
      setResetSignal((value) => value + 1);
    } catch (parseError) {
      setError(parseError instanceof Error ? parseError.message : "Could not read this geometry.");
    }
  }

  function selectPreset(name: keyof typeof PRESETS) {
    const nextSource = PRESETS[name];
    setSource(nextSource);
    setUnit("angstrom");
    updateGeometry(nextSource, "angstrom");
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size > 1_000_000) {
      setError("Choose an XYZ file smaller than 1 MB.");
      return;
    }
    const nextSource = await file.text();
    setSource(nextSource);
    setUnit("angstrom");
    updateGeometry(nextSource, "angstrom");
    event.target.value = "";
  }

  function downloadXyz() {
    const url = URL.createObjectURL(new Blob([xyzFor(atoms)], { type: "chemical/x-xyz" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${formula.toLowerCase() || "molecule"}.xyz`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  return (
    <section className="viewer-workspace shell" aria-label="Molecular geometry workspace">
      <aside className="viewer-input-panel">
        <div className="panel-heading">
          <span>01</span>
          <div>
            <p>Geometry input</p>
            <h2>Coordinates</h2>
          </div>
        </div>

        <div className="preset-row" aria-label="Example molecules">
          {(Object.keys(PRESETS) as (keyof typeof PRESETS)[]).map((name) => (
            <button key={name} type="button" onClick={() => selectPreset(name)}>
              {name}
            </button>
          ))}
        </div>

        <label className="field-label" htmlFor="geometry-source">
          XYZ or PyQED atom string
        </label>
        <textarea
          id="geometry-source"
          value={source}
          onChange={(event) => setSource(event.target.value)}
          spellCheck={false}
          aria-describedby="geometry-help geometry-error"
        />
        <p className="field-help" id="geometry-help">
          One atom per line, or use semicolons: <code>H 0 0 0; H 0 0 0.74</code>
        </p>
        {error ? (
          <p className="viewer-error" id="geometry-error" role="alert">
            {error}
          </p>
        ) : null}

        <div className="input-actions">
          <label className="file-button">
            Open XYZ
            <input accept=".xyz,text/plain,chemical/x-xyz" onChange={handleFile} type="file" />
          </label>
          <label className="unit-control">
            Input unit
            <select
              value={unit}
              onChange={(event) => setUnit(event.target.value as "angstrom" | "bohr")}
            >
              <option value="angstrom">Ångström</option>
              <option value="bohr">Bohr</option>
            </select>
          </label>
          <button className="button button-primary viewer-update" type="button" onClick={() => updateGeometry()}>
            Update view
          </button>
        </div>
      </aside>

      <div className="viewer-display-panel">
        <div className="viewer-toolbar">
          <div className="molecule-identity">
            <span>Live structure</span>
            <strong>{formula}</strong>
            <small>{atoms.length} atoms · {bonds.length} inferred bonds</small>
          </div>
          <div className="viewer-controls" aria-label="Viewer controls">
            <label>
              Representation
              <select value={mode} onChange={(event) => setMode(event.target.value as RenderMode)}>
                <option value="ball-stick">Ball &amp; stick</option>
                <option value="space-fill">Space filling</option>
                <option value="wireframe">Wireframe</option>
              </select>
            </label>
            <button
              type="button"
              className={labels ? "is-active" : ""}
              aria-pressed={labels}
              onClick={() => setLabels((value) => !value)}
            >
              Labels
            </button>
            <button
              type="button"
              className={autoRotate ? "is-active" : ""}
              aria-pressed={autoRotate}
              onClick={() => setAutoRotate((value) => !value)}
            >
              Auto-rotate
            </button>
            <button type="button" onClick={() => setResetSignal((value) => value + 1)}>
              Reset view
            </button>
          </div>
        </div>

        <MoleculeCanvas
          atoms={atoms}
          bonds={bonds}
          mode={mode}
          labels={labels}
          autoRotate={autoRotate}
          resetSignal={resetSignal}
        />

        <div className="viewer-footer-bar">
          <div className="element-legend" aria-label="Elements in molecule">
            {[...new Set(atoms.map((atom) => atom.element))].map((symbol) => {
              const element = ELEMENTS[symbol] ?? FALLBACK_ELEMENT;
              return (
                <span key={symbol}>
                  <i style={{ background: element.color }} />
                  {symbol} · {element.name}
                </span>
              );
            })}
          </div>
          <button type="button" className="export-button" onClick={downloadXyz}>
            Export XYZ <span aria-hidden="true">↓</span>
          </button>
        </div>
      </div>
    </section>
  );
}
