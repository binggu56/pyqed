"use client";

import type {
  ChangeEvent,
  DragEvent as ReactDragEvent,
} from "react";
import {
  useCallback,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import type {
  Atom,
  ValidatedScene,
  VolumeField,
} from "./volume-data.mjs";
import {
  MAX_CUBE_FILE_BYTES,
  fieldToCube,
  gridsMatch,
  parseCube,
  parseGeometry,
  validateSceneMessage,
  xyzFor,
} from "./volume-data.mjs";

type Bond = [number, number];
type RenderMode = "ball-stick" | "space-fill" | "wireframe";
type SurfaceMode = "both" | "positive" | "negative" | "esp-map" | "hidden";
type Vibration = NonNullable<ValidatedScene["vibration"]>;

type ViewerApi = {
  animate(options: Record<string, unknown>): unknown;
  addIsosurface(data: unknown, options: Record<string, unknown>): unknown;
  addLabel(text: string, options: Record<string, unknown>): unknown;
  addModel(data: string, format: string): unknown;
  addSurface(
    type: unknown,
    options: Record<string, unknown>,
    selection: Record<string, never>,
  ): unknown;
  getView(): number[];
  removeAllLabels(): unknown;
  removeAllModels(): unknown;
  removeAllShapes(): unknown;
  removeAllSurfaces(): unknown;
  render(callback?: () => void): unknown;
  resize(): unknown;
  setStyle(selection: Record<string, never>, style: Record<string, unknown>): unknown;
  setView(view: number[]): unknown;
  spin(axis: string | false, speed?: number): unknown;
  stopAnimate(): unknown;
  vibrate(
    frames: number,
    amplitude: number,
    bothWays: boolean,
    arrowSpec?: Record<string, unknown>,
  ): unknown;
  zoomTo(): unknown;
};

type ThreeDmolApi = {
  createViewer(element: HTMLElement, options: Record<string, unknown>): ViewerApi;
  Gradient: { RWB: new (minimum: number, maximum: number) => unknown };
  SurfaceType: { VDW: unknown };
  VolumeData: new (data: string, format: string) => unknown;
};

const ELEMENTS: Record<
  string,
  { color: string; covalent: number; name: string }
> = {
  H: { color: "#f4f5f6", covalent: 0.31, name: "Hydrogen" },
  B: { color: "#f4a67c", covalent: 0.84, name: "Boron" },
  C: { color: "#475965", covalent: 0.76, name: "Carbon" },
  N: { color: "#5085e8", covalent: 0.71, name: "Nitrogen" },
  O: { color: "#ef5350", covalent: 0.66, name: "Oxygen" },
  F: { color: "#74c985", covalent: 0.57, name: "Fluorine" },
  P: { color: "#e29a45", covalent: 1.07, name: "Phosphorus" },
  S: { color: "#e8cb4b", covalent: 1.05, name: "Sulfur" },
  Cl: { color: "#62bd6b", covalent: 1.02, name: "Chlorine" },
  Fe: { color: "#bf6a43", covalent: 1.32, name: "Iron" },
  Br: { color: "#a34f45", covalent: 1.20, name: "Bromine" },
  I: { color: "#8b63b8", covalent: 1.39, name: "Iodine" },
};

const FALLBACK_ELEMENT = {
  color: "#8fa6b0",
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

const FIELD_KIND_LABELS: Record<VolumeField["kind"], string> = {
  orbital: "Molecular orbital",
  "electron-density": "Electron density",
  "spin-density": "Spin density",
  "difference-density": "Difference density",
  "transition-density": "Transition density",
  esp: "Electrostatic potential",
  generic: "Scalar field",
};

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
  return symbols
    .map((symbol) => `${symbol}${counts.get(symbol) === 1 ? "" : counts.get(symbol)}`)
    .join("");
}

function isSignedField(field: VolumeField): boolean {
  return (
    field.kind === "orbital" ||
    field.kind === "spin-density" ||
    field.kind === "difference-density" ||
    field.kind === "transition-density" ||
    (field.kind === "generic" && field.range[0] < 0)
  );
}

function espForDensity(field: VolumeField, fields: VolumeField[]): VolumeField | undefined {
  return fields.find(
    (candidate) =>
      candidate.kind === "esp" &&
      candidate.surface_field === field.name &&
      gridsMatch(candidate, field),
  );
}

function densityForEsp(field: VolumeField, fields: VolumeField[]): VolumeField | undefined {
  if (field.kind !== "esp" || !field.surface_field) return undefined;
  return fields.find(
    (candidate) =>
      candidate.name === field.surface_field &&
      candidate.kind === "electron-density" &&
      gridsMatch(candidate, field),
  );
}

function defaultSurface(field: VolumeField | undefined): SurfaceMode {
  if (!field) return "hidden";
  if (field.kind === "esp") return "esp-map";
  return isSignedField(field) ? "both" : "positive";
}

function initialIsovalue(field: VolumeField | undefined, fields: VolumeField[]): number {
  if (!field) return 0.02;
  const surfaceField = field.kind === "esp" ? densityForEsp(field, fields) : field;
  return surfaceField?.isovalues[0] ?? field.isovalues[0] ?? 0.02;
}

function modelStyle(mode: RenderMode): Record<string, unknown> {
  if (mode === "space-fill") return { sphere: { scale: 1 } };
  if (mode === "wireframe") return { line: { linewidth: 1.5 } };
  return {
    stick: { radius: 0.16 },
    sphere: { scale: 0.3 },
  };
}

function vibrationalXyz(atoms: Atom[], vibration: Vibration): string {
  const rows = atoms.map((atom, index) => {
    const [dx, dy, dz] = vibration.displacements[index];
    return `${atom.element.padEnd(2)} ${atom.x.toFixed(10)} ${atom.y.toFixed(10)} ${atom.z.toFixed(10)} ${dx.toFixed(10)} ${dy.toFixed(10)} ${dz.toFixed(10)}`;
  });
  return `${atoms.length}\nMode ${vibration.modeIndex}: ${vibration.frequencyCm1.toFixed(3)} cm^-1\n${rows.join("\n")}\n`;
}

function MolecularFieldCanvas({
  atoms,
  fields,
  activeFieldIndex,
  mode,
  labels,
  autoRotate,
  surfaceMode,
  isovalue,
  resetSignal,
  vibration,
  vibrationPlaying,
}: {
  atoms: Atom[];
  fields: VolumeField[];
  activeFieldIndex: number;
  mode: RenderMode;
  labels: boolean;
  autoRotate: boolean;
  surfaceMode: SurfaceMode;
  isovalue: number;
  resetSignal: number;
  vibration: Vibration | null;
  vibrationPlaying: boolean;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const viewerRef = useRef<ViewerApi | null>(null);
  const libraryRef = useRef<ThreeDmolApi | null>(null);
  const hasRendered = useRef(false);
  const renderRevision = useRef(0);
  const volumeCacheRef = useRef<WeakMap<VolumeField, unknown>>(new WeakMap());
  const [rendererStatus, setRendererStatus] = useState<"loading" | "ready" | "error">("loading");

  useEffect(() => {
    let cancelled = false;
    const container = containerRef.current;
    if (!container) return;

    void import("3dmol")
      .then((imported) => {
        if (cancelled) return;
        const candidate = ("default" in imported && imported.default
          ? imported.default
          : imported) as unknown as ThreeDmolApi;
        const viewer = candidate.createViewer(container, {
          antialias: true,
          backgroundColor: "#06131c",
          id: "pyqed-field-viewer",
        });
        libraryRef.current = candidate;
        viewerRef.current = viewer;
        setRendererStatus("ready");
      })
      .catch(() => {
        if (!cancelled) setRendererStatus("error");
      });

    return () => {
      cancelled = true;
      renderRevision.current += 1;
      viewerRef.current?.stopAnimate();
      viewerRef.current?.spin(false);
      viewerRef.current = null;
      libraryRef.current = null;
      volumeCacheRef.current = new WeakMap();
      container.replaceChildren();
    };
  }, []);

  useEffect(() => {
    if (rendererStatus !== "ready") return;
    const viewer = viewerRef.current;
    if (!viewer) return;
    const previousView = hasRendered.current ? viewer.getView() : null;

    viewer.stopAnimate();
    viewer.removeAllLabels();
    viewer.removeAllModels();
    if (atoms.length > 0) {
      viewer.addModel(
        vibration ? vibrationalXyz(atoms, vibration) : xyzFor(atoms),
        "xyz",
      );
      viewer.setStyle({}, modelStyle(mode));
      if (vibration) {
        viewer.vibrate(vibration.frames, 1, true, {
          color: "#35d6e3",
          radius: 0.045,
          radiusRatio: 1.7,
        });
        if (vibrationPlaying) {
          viewer.animate({
            interval: vibration.interval,
            loop: "backAndForth",
            reps: 0,
          });
        }
      }
    }

    if (labels && atoms.length > 0 && !vibration) {
      atoms.forEach((atom, index) => {
        viewer.addLabel(`${atom.element}${index + 1}`, {
          position: { x: atom.x, y: atom.y, z: atom.z },
          fontColor: "#e7f7f8",
          backgroundColor: "#031016",
          backgroundOpacity: 0.78,
          borderColor: "#35d6e3",
          borderThickness: 1,
          fontSize: 12,
          inFront: true,
        });
      });
    }

    if (!hasRendered.current) viewer.zoomTo();
    else if (previousView) viewer.setView(previousView);
    viewer.render();
    hasRendered.current = true;
  }, [atoms, labels, mode, rendererStatus, vibration, vibrationPlaying]);

  useEffect(() => {
    if (rendererStatus !== "ready") return;
    const viewer = viewerRef.current;
    const library = libraryRef.current;
    if (!viewer || !library) return;
    const revision = renderRevision.current + 1;
    renderRevision.current = revision;

    viewer.removeAllShapes();
    viewer.removeAllSurfaces();

    const field = fields[activeFieldIndex];
    if (field && surfaceMode !== "hidden" && Number.isFinite(isovalue) && isovalue > 0) {
      const volumeFor = (volumeField: VolumeField) => {
        const cached = volumeCacheRef.current.get(volumeField);
        if (cached) return cached;
        const volume = new library.VolumeData(fieldToCube(volumeField, atoms), "cube");
        volumeCacheRef.current.set(volumeField, volume);
        return volume;
      };
      const fieldVolume = volumeFor(field);
      const positiveOptions: Record<string, unknown> = {
        isoval: Math.abs(isovalue),
        color: field.colors.positive,
        opacity: field.opacity,
        smoothness: 4,
      };

      if (surfaceMode === "esp-map") {
        const density = field.kind === "esp" ? densityForEsp(field, fields) : field;
        const esp = field.kind === "esp" ? field : espForDensity(field, fields);
        if (density && esp) {
          const densityVolume = density === field
            ? fieldVolume
            : volumeFor(density);
          const espVolume = esp === field
            ? fieldVolume
            : volumeFor(esp);
          const range = Math.max(Math.abs(esp.range[0]), Math.abs(esp.range[1]), 1e-12);
          viewer.addIsosurface(densityVolume, {
            ...positiveOptions,
            isoval: Math.abs(isovalue),
            opacity: density.opacity,
            voldata: espVolume,
            volscheme: new library.Gradient.RWB(-range, range),
          });
        } else if (field.kind === "esp" && atoms.length > 0) {
          const range = Math.max(Math.abs(field.range[0]), Math.abs(field.range[1]), 1e-12);
          const surfaceResult = viewer.addSurface(
            library.SurfaceType.VDW,
            {
              opacity: field.opacity,
              voldata: fieldVolume,
              volscheme: new library.Gradient.RWB(-range, range),
            },
            {},
          );
          if (surfaceResult && typeof (surfaceResult as Promise<unknown>).then === "function") {
            void Promise.resolve(surfaceResult).then(
              () => {
                if (renderRevision.current === revision) viewer.render();
              },
              () => undefined,
            );
          }
        } else if (field.kind === "esp") {
          viewer.addIsosurface(fieldVolume, positiveOptions);
          viewer.addIsosurface(fieldVolume, {
            isoval: -Math.abs(isovalue),
            color: field.colors.negative,
            opacity: field.opacity,
            smoothness: 4,
          });
        }
      } else {
        if (surfaceMode === "positive" || surfaceMode === "both") {
          viewer.addIsosurface(fieldVolume, positiveOptions);
        }
        if ((surfaceMode === "negative" || surfaceMode === "both") && isSignedField(field)) {
          viewer.addIsosurface(fieldVolume, {
            isoval: -Math.abs(isovalue),
            color: field.colors.negative,
            opacity: field.opacity,
            smoothness: 4,
          });
        }
      }
    }

    viewer.render();
  }, [
    activeFieldIndex,
    atoms,
    fields,
    isovalue,
    rendererStatus,
    surfaceMode,
  ]);

  useEffect(() => {
    if (rendererStatus !== "ready") return;
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.spin(autoRotate ? "y" : false, 0.45);
    viewer.render();
  }, [autoRotate, rendererStatus]);

  useEffect(() => {
    if (rendererStatus !== "ready") return;
    const viewer = viewerRef.current;
    if (!viewer) return;
    viewer.zoomTo();
    viewer.render();
  }, [rendererStatus, resetSignal]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || typeof ResizeObserver === "undefined") return;
    const observer = new ResizeObserver(() => {
      viewerRef.current?.resize();
      viewerRef.current?.render();
    });
    observer.observe(container);
    return () => observer.disconnect();
  }, []);

  const activeField = fields[activeFieldIndex];
  const description = activeField
    ? `${FIELD_KIND_LABELS[activeField.kind]} “${activeField.label}” on ${activeField.shape.join(" by ")} grid points`
    : atoms.length > 0 ? "molecular geometry" : "an empty scene";

  return (
    <div className="molecule-canvas-wrap">
      <div
        ref={containerRef}
        className="molecule-canvas"
        role="img"
        aria-label={`Interactive three-dimensional view of ${description}`}
      />
      {rendererStatus === "loading" ? (
        <p className="viewer-renderer-status" role="status">Loading local 3D renderer…</p>
      ) : null}
      {rendererStatus === "error" ? (
        <p className="viewer-renderer-status viewer-renderer-error" role="alert">
          WebGL could not start in this browser. The geometry and field data remain local.
        </p>
      ) : null}
      <p className="viewer-canvas-hint" aria-hidden="true">
        Drag to rotate · scroll to zoom · right-drag to pan
      </p>
    </div>
  );
}

export function MoleculeViewer() {
  const [source, setSource] = useState<string>(PRESETS.Water);
  const [unit, setUnit] = useState<"angstrom" | "bohr">("angstrom");
  const [atoms, setAtoms] = useState<Atom[]>(() => parseGeometry(PRESETS.Water, "angstrom"));
  const [fields, setFields] = useState<VolumeField[]>([]);
  const [vibration, setVibration] = useState<Vibration | null>(null);
  const [vibrationPlaying, setVibrationPlaying] = useState(false);
  const [activeFieldIndex, setActiveFieldIndex] = useState(-1);
  const [surfaceMode, setSurfaceMode] = useState<SurfaceMode>("hidden");
  const [isovalue, setIsovalue] = useState(0.02);
  const [sceneTitle, setSceneTitle] = useState("Water");
  const [status, setStatus] = useState("Geometry loaded");
  const [error, setError] = useState<string | null>(null);
  const [mode, setMode] = useState<RenderMode>("ball-stick");
  const [labels, setLabels] = useState(false);
  const [autoRotate, setAutoRotate] = useState(false);
  const [isDraggingFile, setIsDraggingFile] = useState(false);
  const [resetSignal, setResetSignal] = useState(0);
  const bonds = useMemo(() => inferBonds(atoms), [atoms]);
  const formula = useMemo(() => formulaFor(atoms), [atoms]);
  const activeField = fields[activeFieldIndex];
  const activeEsp = activeField?.kind === "electron-density"
    ? espForDensity(activeField, fields)
    : undefined;
  const activeDensity = activeField?.kind === "esp"
    ? densityForEsp(activeField, fields)
    : undefined;
  const renderedIsovalue = useDeferredValue(isovalue);

  useEffect(() => {
    const reducedMotion = typeof window.matchMedia === "function"
      ? window.matchMedia("(prefers-reduced-motion: reduce)")
      : null;
    const frame = window.requestAnimationFrame(() => {
      if (!reducedMotion?.matches) setAutoRotate(true);
    });
    if (!reducedMotion) return () => window.cancelAnimationFrame(frame);
    const disableAnimation = (event: MediaQueryListEvent) => {
      if (event.matches) setAutoRotate(false);
    };
    reducedMotion.addEventListener("change", disableAnimation);
    return () => {
      window.cancelAnimationFrame(frame);
      reducedMotion.removeEventListener("change", disableAnimation);
    };
  }, []);

  const chooseField = useCallback((nextIndex: number, nextFields = fields) => {
    const nextField = nextFields[nextIndex];
    setActiveFieldIndex(nextField ? nextIndex : -1);
    setSurfaceMode(defaultSurface(nextField));
    setIsovalue(initialIsovalue(nextField, nextFields));
  }, [fields]);

  const applyScene = useCallback((scene: ValidatedScene, nextStatus: string) => {
    setAtoms(scene.atoms);
    setSource(scene.molecule?.xyz ?? "");
    setUnit("angstrom");
    setFields(scene.fields);
    setVibration(scene.vibration);
    setVibrationPlaying(scene.vibration !== null);
    setSceneTitle(scene.title);
    setMode(scene.molecule?.representation ?? "ball-stick");
    setLabels(scene.molecule?.labels ?? false);
    if (scene.vibration) setAutoRotate(false);
    setError(null);
    setStatus(nextStatus);
    const nextField = scene.fields[scene.activeFieldIndex];
    setActiveFieldIndex(nextField ? scene.activeFieldIndex : -1);
    setSurfaceMode(defaultSurface(nextField));
    setIsovalue(initialIsovalue(nextField, scene.fields));
    setResetSignal((value) => value + 1);
  }, []);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      const parameters = new URLSearchParams(window.location.hash.slice(1));
      const geometry = parameters.get("xyz");
      if (!geometry) return;

      try {
        const linkedAtoms = parseGeometry(geometry, "angstrom");
        setAtoms(linkedAtoms);
        setSource(geometry);
        setUnit("angstrom");
        setFields([]);
        setVibration(null);
        setVibrationPlaying(false);
        setActiveFieldIndex(-1);
        setSurfaceMode("hidden");
        setSceneTitle("Linked geometry");
        setStatus("Geometry loaded from link");
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

  useEffect(() => {
    function receiveScene(event: MessageEvent<unknown>) {
      const sourceIsAllowed =
        event.source === window ||
        event.source === window.parent ||
        (window.opener !== null && event.source === window.opener);
      if (!sourceIsAllowed) return;
      if (
        event.data === null ||
        typeof event.data !== "object" ||
        !("type" in event.data) ||
        event.data.type !== "pyqed:scene"
      ) {
        return;
      }
      try {
        const scene = validateSceneMessage(event.data);
        applyScene(
          scene,
          scene.vibration
            ? `Normal mode ${scene.vibration.modeIndex} received from PyQED`
            : `${scene.fields.length} ${scene.fields.length === 1 ? "surface" : "states / surfaces"} received from PyQED`,
        );
      } catch (sceneError) {
        setError(
          sceneError instanceof Error
            ? `Could not load the PyQED scene: ${sceneError.message}`
            : "Could not load the PyQED scene.",
        );
      }
    }

    window.addEventListener("message", receiveScene);
    const readyMessage = { type: "pyqed:viewer-ready", version: 1 };
    const recipients: Window[] = [];
    if (window.parent !== window) recipients.push(window.parent);
    if (window.opener !== null) recipients.push(window.opener as Window);
    for (const recipient of recipients) {
      try {
        // The launcher may have a file: or loopback origin; this message carries no scene data.
        recipient.postMessage(readyMessage, "*");
      } catch {
        // A parent or opener can close between discovery and notification.
      }
    }
    return () => window.removeEventListener("message", receiveScene);
  }, [applyScene]);

  function updateGeometry(
    nextSource = source,
    nextUnit = unit,
    nextTitle = "Edited geometry",
  ) {
    try {
      setAtoms(parseGeometry(nextSource, nextUnit));
      setFields([]);
      setVibration(null);
      setVibrationPlaying(false);
      setActiveFieldIndex(-1);
      setSurfaceMode("hidden");
      setSceneTitle(nextTitle);
      setStatus("Geometry updated");
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
    updateGeometry(nextSource, "angstrom", name);
  }

  async function loadFile(file: File) {
    const name = file.name.toLowerCase();
    const isCube = name.endsWith(".cube") || name.endsWith(".cub");
    const isXyz = name.endsWith(".xyz");
    if (!isCube && !isXyz) {
      setError("Choose an .xyz, .cube, or .cub file.");
      return;
    }
    const limit = isCube ? MAX_CUBE_FILE_BYTES : 1_000_000;
    if (file.size > limit) {
      setError(isCube ? "Choose a cube file smaller than 80 MB." : "Choose an XYZ file smaller than 1 MB.");
      return;
    }
    try {
      const text = await file.text();
      if (isCube) {
        const scene = parseCube(text, { fileName: file.name });
        applyScene(
          scene,
          `${scene.fields.length} ${scene.fields.length === 1 ? "surface" : "states / surfaces"} loaded from ${file.name}`,
        );
      } else {
        const nextAtoms = parseGeometry(text, "angstrom");
        setSource(text);
        setUnit("angstrom");
        setAtoms(nextAtoms);
        setFields([]);
        setVibration(null);
        setVibrationPlaying(false);
        setActiveFieldIndex(-1);
        setSurfaceMode("hidden");
        setSceneTitle(file.name.replace(/\.xyz$/i, ""));
        setStatus(`Geometry loaded from ${file.name}`);
        setError(null);
        setResetSignal((value) => value + 1);
      }
    } catch (fileError) {
      setError(fileError instanceof Error ? fileError.message : "Could not read this file.");
    }
  }

  async function handleFile(event: ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0];
    if (file) await loadFile(file);
    event.target.value = "";
  }

  function handleDragOver(event: ReactDragEvent<HTMLDivElement>) {
    event.preventDefault();
    if (event.dataTransfer.types.includes("Files")) setIsDraggingFile(true);
  }

  function handleDragLeave(event: ReactDragEvent<HTMLDivElement>) {
    if (!event.currentTarget.contains(event.relatedTarget as Node | null)) setIsDraggingFile(false);
  }

  async function handleDrop(event: ReactDragEvent<HTMLDivElement>) {
    event.preventDefault();
    setIsDraggingFile(false);
    const file = event.dataTransfer.files[0];
    if (file) await loadFile(file);
  }

  function downloadXyz() {
    if (atoms.length === 0) return;
    const url = URL.createObjectURL(new Blob([xyzFor(atoms)], { type: "chemical/x-xyz" }));
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${formula.toLowerCase() || "molecule"}.xyz`;
    anchor.click();
    URL.revokeObjectURL(url);
  }

  const isovalueMaximum = activeField
    ? Math.max(
        Math.abs((activeField.kind === "esp" ? activeDensity : activeField)?.range[0] ?? 0),
        Math.abs((activeField.kind === "esp" ? activeDensity : activeField)?.range[1] ?? 1),
        isovalue * 10,
        1e-6,
      )
    : 1;
  const isovalueStep = Math.max(isovalueMaximum / 500, 1e-8);

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
            Open XYZ / cube
            <input
              accept=".xyz,.cube,.cub,text/plain,chemical/x-xyz,chemical/x-gaussian-cube"
              onChange={handleFile}
              type="file"
            />
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

        <div className="volume-input-note">
          <span>02</span>
          <div>
            <strong>Scalar fields</strong>
            <p>
              Drop a Gaussian cube file onto the viewer, or call <code>view(...)</code> in
              PyQED. Multi-state files remain selectable and never enter the page URL.
            </p>
          </div>
        </div>
      </aside>

      <div
        className={`viewer-display-panel${isDraggingFile ? " is-file-dragging" : ""}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        <div className="viewer-toolbar">
          <div className="molecule-identity">
            <span>
              {vibration ? "Normal mode" : activeField ? FIELD_KIND_LABELS[activeField.kind] : "Live structure"}
            </span>
            <strong>
              {vibration
                ? `Mode ${vibration.modeIndex} · ${vibration.frequencyCm1.toFixed(1)} cm⁻¹`
                : activeField?.label ?? formula}
            </strong>
            <small>
              {sceneTitle} · {atoms.length} atoms · {bonds.length} inferred bonds
              {fields.length > 0 ? ` · ${fields.length} ${fields.length === 1 ? "surface" : "states / surfaces"}` : ""}
            </small>
          </div>
          <div className="viewer-controls" aria-label="Viewer controls">
            <label>
              Field / state
              <select
                value={activeFieldIndex}
                onChange={(event) => chooseField(Number(event.target.value))}
              >
                <option value={-1}>Molecule only</option>
                {fields.map((field, index) => (
                  <option key={field.name} value={index}>
                    {index + 1}. {field.label}
                  </option>
                ))}
              </select>
            </label>
            <label>
              Surface
              <select
                value={surfaceMode}
                disabled={!activeField}
                onChange={(event) => setSurfaceMode(event.target.value as SurfaceMode)}
              >
                {!activeField ? <option value="hidden">No scalar field</option> : null}
                {activeField && isSignedField(activeField) ? (
                  <>
                    <option value="both">Positive + negative</option>
                    <option value="positive">Positive phase</option>
                    <option value="negative">Negative phase</option>
                  </>
                ) : null}
                {activeField && activeField.kind !== "esp" && !isSignedField(activeField) ? (
                  <option value="positive">Positive isosurface</option>
                ) : null}
                {activeField && (activeField.kind === "esp" || activeEsp) ? (
                  <option value="esp-map">
                    {activeDensity || activeEsp
                      ? "Density colored by ESP"
                      : atoms.length > 0 ? "ESP molecular surface" : "ESP isosurfaces"}
                  </option>
                ) : null}
                {activeField ? <option value="hidden">Hide surface</option> : null}
              </select>
            </label>
            <label className="isovalue-control">
              {activeField?.kind === "esp" && activeDensity ? "Density isovalue" : "Isovalue"}
              <input
                type="number"
                min={isovalueStep}
                max={isovalueMaximum}
                step={isovalueStep}
                value={Number(isovalue.toPrecision(5))}
                disabled={!activeField || (activeField.kind === "esp" && !activeDensity && atoms.length > 0)}
                onChange={(event) => {
                  const value = Number(event.target.value);
                  if (Number.isFinite(value) && value > 0) setIsovalue(value);
                }}
              />
            </label>
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
            {vibration ? (
              <button
                type="button"
                className={vibrationPlaying ? "is-active" : ""}
                aria-pressed={vibrationPlaying}
                onClick={() => setVibrationPlaying((value) => !value)}
              >
                {vibrationPlaying ? "Pause mode" : "Play mode"}
              </button>
            ) : null}
            <button type="button" onClick={() => setResetSignal((value) => value + 1)}>
              Reset view
            </button>
          </div>
        </div>

        <MolecularFieldCanvas
          atoms={atoms}
          fields={fields}
          activeFieldIndex={activeFieldIndex}
          mode={mode}
          labels={labels}
          autoRotate={autoRotate}
          surfaceMode={surfaceMode}
          isovalue={renderedIsovalue}
          resetSignal={resetSignal}
          vibration={vibration}
          vibrationPlaying={vibrationPlaying}
        />

        {isDraggingFile ? (
          <div className="viewer-drop-overlay" role="status">
            <strong>Drop XYZ or Gaussian cube</strong>
            <span>Geometry and every scalar state will load locally.</span>
          </div>
        ) : null}

        <div className="viewer-footer-bar">
          <div>
            <div className="element-legend" aria-label="Elements and surfaces in view">
              {[...new Set(atoms.map((atom) => atom.element))].map((symbol) => {
                const element = ELEMENTS[symbol] ?? FALLBACK_ELEMENT;
                return (
                  <span key={symbol}>
                    <i style={{ background: element.color }} />
                    {symbol} · {element.name}
                  </span>
                );
              })}
              {activeField && surfaceMode !== "hidden" && surfaceMode !== "esp-map" ? (
                <>
                  <span><i className="surface-positive" />+{activeField.label}</span>
                  {isSignedField(activeField) && surfaceMode !== "positive" ? (
                    <span><i className="surface-negative" />−{activeField.label}</span>
                  ) : null}
                </>
              ) : null}
              {activeField && surfaceMode === "esp-map" ? (
                <span className="esp-gradient-key"><i />negative ESP · neutral · positive ESP</span>
              ) : null}
            </div>
            <p className="viewer-file-status" aria-live="polite">{status}</p>
          </div>
          <button type="button" className="export-button" onClick={downloadXyz} disabled={atoms.length === 0}>
            Export XYZ <span aria-hidden="true">↓</span>
          </button>
        </div>
      </div>
    </section>
  );
}
