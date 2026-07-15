export type Atom = {
  element: string;
  x: number;
  y: number;
  z: number;
};

export type FieldKind =
  | "orbital"
  | "electron-density"
  | "spin-density"
  | "difference-density"
  | "transition-density"
  | "esp"
  | "generic";

export type VolumeField = {
  name: string;
  label: string;
  kind: FieldKind;
  shape: [number, number, number];
  origin: [number, number, number];
  axes: [[number, number, number], [number, number, number], [number, number, number]];
  values: Float32Array;
  isovalues: number[];
  colors: { positive: string; negative: string };
  opacity: number;
  units?: string;
  surface_field?: string;
  metadata?: Record<string, JsonValue>;
  range: [number, number];
};

export type JsonValue = string | number | boolean | null | JsonValue[] | { [key: string]: JsonValue };

export type ValidatedScene = {
  title: string;
  atoms: Atom[];
  molecule: {
    xyz: string;
    representation: "ball-stick" | "space-fill" | "wireframe";
    labels: boolean;
  } | null;
  vibration: {
    modeIndex: number;
    frequencyCm1: number;
    displacements: [number, number, number][];
    amplitudeAngstrom: number;
    frames: number;
    interval: number;
  } | null;
  fields: VolumeField[];
  activeFieldIndex: number;
};

export const BOHR_TO_ANGSTROM: number;
export const MAX_ATOMS: number;
export const MAX_FIELDS: number;
export const MAX_GRID_POINTS: number;
export const MAX_TOTAL_GRID_POINTS: number;
export const MAX_CUBE_FILE_BYTES: number;

export function parseGeometry(text: string, unit?: "angstrom" | "bohr"): Atom[];
export function xyzFor(atoms: Atom[], comment?: string): string;
export function validateSceneMessage(message: unknown): ValidatedScene;
export function parseCube(text: string, options?: { fileName?: string }): ValidatedScene;
export function suggestIsovalue(values: Iterable<number>, kind?: FieldKind): number;
export function fieldToCube(field: VolumeField, atoms: Atom[]): string;
export function gridsMatch(left: VolumeField, right: VolumeField): boolean;
