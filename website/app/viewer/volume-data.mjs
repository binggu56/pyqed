export const BOHR_TO_ANGSTROM = 0.529177210903;
export const MAX_ATOMS = 500;
export const MAX_FIELDS = 512;
export const MAX_GRID_POINTS = 2_000_000;
export const MAX_TOTAL_GRID_POINTS = 8_000_000;
export const MAX_CUBE_FILE_BYTES = 80_000_000;

const MAX_GEOMETRY_CHARS = 1_000_000;
const MAX_FLOAT32 = 3.4028234663852886e38;
const FIELD_KINDS = new Set([
  "orbital",
  "electron-density",
  "spin-density",
  "difference-density",
  "transition-density",
  "esp",
  "generic",
]);
const REPRESENTATIONS = new Set(["ball-stick", "space-fill", "wireframe"]);
const FIELD_ALIASES = new Map([
  ["mo", "orbital"],
  ["density", "electron-density"],
  ["electron_density", "electron-density"],
  ["spin", "spin-density"],
  ["spin_density", "spin-density"],
  ["difference", "difference-density"],
  ["difference_density", "difference-density"],
  ["transition", "transition-density"],
  ["transition_density", "transition-density"],
  ["electrostatic-potential", "esp"],
  ["electrostatic_potential", "esp"],
]);

const ELEMENT_SYMBOLS = Object.freeze([
  "X",
  "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
  "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
  "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
  "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
  "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
  "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
  "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
  "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
  "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
  "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
  "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds",
  "Rg", "Cn", "Nh", "Fl", "Mc", "Lv", "Ts", "Og",
]);
const ATOMIC_NUMBERS = new Map(ELEMENT_SYMBOLS.map((symbol, index) => [symbol, index]));

function fail(message) {
  throw new Error(message);
}

function isRecord(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function assertAllowedKeys(record, allowed, context) {
  for (const key of Object.keys(record)) {
    if (!allowed.has(key)) fail(`${context} contains unsupported property “${key}”.`);
  }
}

function cleanText(value, context, maximum, fallback) {
  if (value === undefined && fallback !== undefined) return fallback;
  if (typeof value !== "string") fail(`${context} must be text.`);
  const cleaned = value.trim();
  if (!cleaned) fail(`${context} cannot be empty.`);
  if (cleaned.length > maximum) fail(`${context} is longer than ${maximum} characters.`);
  return cleaned;
}

function cleanLine(value, context, maximum, fallback) {
  const cleaned = cleanText(value, context, maximum, fallback);
  if (/[\u0000-\u001f\u007f]/.test(cleaned)) fail(`${context} must fit on one line.`);
  return cleaned;
}

function finiteNumber(value, context) {
  if (typeof value !== "number" || !Number.isFinite(value) || Math.abs(value) > MAX_FLOAT32) {
    fail(`${context} must be a finite 32-bit floating-point value.`);
  }
  return value;
}

function finiteVector(value, context) {
  if (!Array.isArray(value) || value.length !== 3) fail(`${context} must contain three numbers.`);
  return value.map((entry, index) => finiteNumber(entry, `${context}[${index}]`));
}

function sanitizeMetadata(value, context, depth = 0, budget = { entries: 0 }) {
  budget.entries += 1;
  if (budget.entries > 512) fail(`${context} contains too many entries.`);
  if (depth > 5) fail(`${context} is nested too deeply.`);
  if (value === null || typeof value === "boolean") return value;
  if (typeof value === "number") {
    if (!Number.isFinite(value)) fail(`${context} contains a non-finite number.`);
    return value;
  }
  if (typeof value === "string") {
    if (value.length > 1_000) fail(`${context} contains text longer than 1,000 characters.`);
    return value;
  }
  if (Array.isArray(value)) {
    if (value.length > 128) fail(`${context} contains an array longer than 128 entries.`);
    return value.map((entry, index) => sanitizeMetadata(entry, `${context}[${index}]`, depth + 1, budget));
  }
  if (!isRecord(value)) fail(`${context} must contain JSON-compatible values only.`);
  const result = Object.create(null);
  for (const [key, entry] of Object.entries(value)) {
    if (key.length === 0 || key.length > 80 || key === "__proto__" || key === "constructor" || key === "prototype") {
      fail(`${context} contains an unsafe or overlong key.`);
    }
    result[key] = sanitizeMetadata(entry, `${context}.${key}`, depth + 1, budget);
  }
  return result;
}

function determinant(axes) {
  const [a, b, c] = axes;
  return (
    a[0] * (b[1] * c[2] - b[2] * c[1]) -
    a[1] * (b[0] * c[2] - b[2] * c[0]) +
    a[2] * (b[0] * c[1] - b[1] * c[0])
  );
}

function sameGrid(left, right) {
  if (left.shape.some((value, index) => value !== right.shape[index])) return false;
  const leftNumbers = [...left.origin, ...left.axes.flat()];
  const rightNumbers = [...right.origin, ...right.axes.flat()];
  return leftNumbers.every(
    (value, index) => Math.abs(value - rightNumbers[index]) <= 1e-9 * Math.max(1, Math.abs(value)),
  );
}

function normalizeKind(value, context) {
  if (typeof value !== "string") fail(`${context} must be text.`);
  const normalized = FIELD_ALIASES.get(value) ?? value;
  if (!FIELD_KINDS.has(normalized)) {
    fail(`${context} must identify an orbital, density, ESP, or generic scalar field.`);
  }
  return normalized;
}

function normalizeColor(value, context, fallback) {
  if (value === undefined) return fallback;
  if (
    typeof value !== "string" ||
    !(/^(?:#[0-9a-f]{3}|#[0-9a-f]{6}|#[0-9a-f]{8}|[a-z][a-z0-9_-]{0,31})$/i.test(value))
  ) {
    fail(`${context} must be a hexadecimal or named color.`);
  }
  return value.toLowerCase();
}

function canonicalElement(value) {
  if (typeof value !== "string") fail("Element symbol must be text.");
  const trimmed = value.trim();
  if (!/^[A-Za-z]{1,2}$/.test(trimmed)) fail(`Invalid element symbol “${value}”.`);
  const symbol = trimmed[0].toUpperCase() + trimmed.slice(1).toLowerCase();
  if (!ATOMIC_NUMBERS.has(symbol)) fail(`Unsupported element symbol “${value}”.`);
  return symbol;
}

export function parseGeometry(text, unit = "angstrom") {
  if (typeof text !== "string" || text.length > MAX_GEOMETRY_CHARS) {
    fail("The geometry must be text smaller than 1 MB.");
  }
  if (unit !== "angstrom" && unit !== "bohr") fail("Geometry unit must be angstrom or bohr.");
  const trimmed = text.trim();
  if (!trimmed) fail("Enter at least one atom.");

  // Semicolons are compact row separators only for headerless atom strings.
  // A standard XYZ comment is free text and may legitimately contain them.
  const firstLine = trimmed.split(/\r?\n/, 1)[0].trim();
  const firstIsCount = /^\d+$/.test(firstLine);
  const normalized = firstIsCount ? trimmed : trimmed.replaceAll(";", "\n");

  const rawLines = normalized.split(/\r?\n/).map((line) => line.trim());
  const expectedCount = firstIsCount ? Number(rawLines[0]) : null;
  const coordinateLines = firstIsCount ? rawLines.slice(2) : rawLines;
  const scale = unit === "bohr" ? BOHR_TO_ANGSTROM : 1;
  const atoms = coordinateLines
    .filter(Boolean)
    .map((line, index) => {
      const fields = line.split(/[\s,]+/);
      if (fields.length !== 4) {
        fail(`Line ${index + (firstIsCount ? 3 : 1)} must contain an element and three coordinates.`);
      }
      const [symbol, ...coordinateText] = fields;
      const coordinates = coordinateText.map((entry) => Number(entry.replace(/[dD]/, "e")));
      if (coordinates.some((value) => !Number.isFinite(value))) {
        fail(`Line ${index + (firstIsCount ? 3 : 1)} contains an invalid number.`);
      }
      const lineNumber = index + (firstIsCount ? 3 : 1);
      const scaledCoordinates = coordinates.map((value, axis) =>
        finiteNumber(value * scale, `Line ${lineNumber} coordinate ${axis + 1}`),
      );
      return {
        element: canonicalElement(symbol),
        x: scaledCoordinates[0],
        y: scaledCoordinates[1],
        z: scaledCoordinates[2],
      };
    });

  if (atoms.length === 0) fail("No coordinate rows were found.");
  if (atoms.length > MAX_ATOMS) fail(`The viewer supports up to ${MAX_ATOMS} atoms.`);
  if (expectedCount !== null && expectedCount !== atoms.length) {
    fail(`XYZ header declares ${expectedCount} atoms, but ${atoms.length} were found.`);
  }
  return atoms;
}

export function xyzFor(atoms, comment = "Exported from the PyQED molecular viewer") {
  if (!Array.isArray(atoms) || atoms.length === 0 || atoms.length > MAX_ATOMS) {
    fail(`A geometry must contain between 1 and ${MAX_ATOMS} atoms.`);
  }
  const rows = atoms.map(({ element, x, y, z }, index) => {
    const symbol = canonicalElement(element);
    const coordinates = [x, y, z].map((value, axis) =>
      finiteNumber(value, `Atom ${index + 1} coordinate ${axis + 1}`),
    );
    return `${symbol.padEnd(2)} ${coordinates[0].toFixed(8).padStart(13)} ${coordinates[1].toFixed(8).padStart(13)} ${coordinates[2].toFixed(8).padStart(13)}`;
  });
  return `${atoms.length}\n${comment} · coordinates in angstrom\n${rows.join("\n")}\n`;
}

function decodeFloat32Base64(value, pointCount, context) {
  if (typeof value !== "string") {
    fail(`${context} must be base64 text when value_encoding is float32-le-base64.`);
  }
  const byteLength = pointCount * Float32Array.BYTES_PER_ELEMENT;
  const expectedLength = Math.ceil(byteLength / 3) * 4;
  if (value.length !== expectedLength) {
    fail(`${context} has ${value.length} base64 characters, but ${pointCount} Float32 values require ${expectedLength}.`);
  }
  const paddingLength = (3 - (byteLength % 3)) % 3;
  const paddingIndex = value.indexOf("=");
  const expectedPaddingIndex = paddingLength === 0 ? -1 : value.length - paddingLength;
  if (
    /[^A-Za-z0-9+/=]/.test(value) ||
    paddingIndex !== expectedPaddingIndex ||
    (paddingLength > 0 && value.slice(-paddingLength) !== "=".repeat(paddingLength))
  ) {
    fail(`${context} must use canonical base64 characters and padding.`);
  }
  if (paddingLength > 0) {
    const alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    const finalDigit = alphabet.indexOf(value[value.length - paddingLength - 1]);
    const unusedBitFactor = paddingLength === 2 ? 16 : 4;
    if (finalDigit < 0 || finalDigit % unusedBitFactor !== 0) {
      fail(`${context} must use canonical base64 padding bits.`);
    }
  }

  let decoded;
  try {
    decoded = atob(value);
  } catch {
    fail(`${context} is not valid base64.`);
  }
  if (decoded.length !== byteLength) {
    fail(`${context} decodes to ${decoded.length} bytes, but ${byteLength} are required.`);
  }
  const buffer = new ArrayBuffer(byteLength);
  const bytes = new Uint8Array(buffer);
  for (let index = 0; index < decoded.length; index += 1) bytes[index] = decoded.charCodeAt(index);
  const data = new DataView(buffer);
  const values = new Float32Array(pointCount);
  for (let index = 0; index < pointCount; index += 1) {
    values[index] = data.getFloat32(index * Float32Array.BYTES_PER_ELEMENT, true);
  }
  return values;
}

function validateField(rawField, index) {
  const context = `scene.fields[${index}]`;
  if (!isRecord(rawField)) fail(`${context} must be an object.`);
  assertAllowedKeys(
    rawField,
    new Set([
      "name", "label", "kind", "shape", "origin", "axes", "values", "isovalues",
      "colors", "opacity", "units", "surface_field", "metadata", "value_encoding",
    ]),
    context,
  );

  const name = cleanLine(rawField.name, `${context}.name`, 80);
  const label = cleanLine(rawField.label, `${context}.label`, 120, name);
  const kind = normalizeKind(rawField.kind, `${context}.kind`);
  if (!Array.isArray(rawField.shape) || rawField.shape.length !== 3) {
    fail(`${context}.shape must contain three grid dimensions.`);
  }
  const shape = rawField.shape.map((value, axis) => {
    if (!Number.isSafeInteger(value) || value < 2 || value > 2048) {
      fail(`${context}.shape[${axis}] must be an integer between 2 and 2048.`);
    }
    return value;
  });
  const pointCount = shape[0] * shape[1] * shape[2];
  if (pointCount > MAX_GRID_POINTS) {
    fail(`${context} contains ${pointCount.toLocaleString()} points; the per-field limit is ${MAX_GRID_POINTS.toLocaleString()}.`);
  }

  const origin = finiteVector(rawField.origin, `${context}.origin`);
  if (!Array.isArray(rawField.axes) || rawField.axes.length !== 3) {
    fail(`${context}.axes must contain three step vectors in angstrom.`);
  }
  const axes = rawField.axes.map((axis, axisIndex) =>
    finiteVector(axis, `${context}.axes[${axisIndex}]`),
  );
  if (Math.abs(determinant(axes)) < 1e-15) fail(`${context}.axes must span a three-dimensional grid.`);

  const valueEncoding = rawField.value_encoding;
  if (valueEncoding !== undefined && valueEncoding !== "float32-le-base64") {
    fail(`${context}.value_encoding must be float32-le-base64 when supplied.`);
  }
  const rawValues = rawField.values;
  let values;
  let valueSource;
  if (valueEncoding === "float32-le-base64") {
    values = decodeFloat32Base64(rawValues, pointCount, `${context}.values`);
    valueSource = values;
  } else {
    const valuesAreArray = Array.isArray(rawValues);
    const valuesAreTypedArray = ArrayBuffer.isView(rawValues) && !(rawValues instanceof DataView);
    if (!valuesAreArray && !valuesAreTypedArray) fail(`${context}.values must be a numeric array.`);
    if (rawValues.length !== pointCount) {
      fail(`${context}.values has ${rawValues.length} entries, but C-order shape ${shape.join(" × ")} requires ${pointCount}.`);
    }
    values = new Float32Array(pointCount);
    valueSource = rawValues;
  }
  let minimum = Number.POSITIVE_INFINITY;
  let maximum = Number.NEGATIVE_INFINITY;
  for (let valueIndex = 0; valueIndex < pointCount; valueIndex += 1) {
    const value = finiteNumber(valueSource[valueIndex], `${context}.values[${valueIndex}]`);
    values[valueIndex] = value;
    minimum = Math.min(minimum, value);
    maximum = Math.max(maximum, value);
  }
  let isovalues;
  if (rawField.isovalues === undefined) {
    isovalues = [suggestIsovalue(values, kind)];
  } else {
    if (!Array.isArray(rawField.isovalues) || rawField.isovalues.length === 0 || rawField.isovalues.length > 8) {
      fail(`${context}.isovalues must contain between one and eight numbers.`);
    }
    isovalues = [...new Set(rawField.isovalues.map((value, isoIndex) => {
      const normalized = Math.abs(finiteNumber(value, `${context}.isovalues[${isoIndex}]`));
      if (normalized === 0) fail(`${context}.isovalues cannot contain zero.`);
      return normalized;
    }))];
  }

  let positive = "#35d6e3";
  let negative = "#f38b6b";
  if (Array.isArray(rawField.colors)) {
    if (rawField.colors.length < 1 || rawField.colors.length > 2) {
      fail(`${context}.colors must contain one or two colors.`);
    }
    positive = normalizeColor(rawField.colors[0], `${context}.colors[0]`, positive);
    negative = normalizeColor(rawField.colors[1], `${context}.colors[1]`, negative);
  } else if (rawField.colors !== undefined) {
    if (!isRecord(rawField.colors)) fail(`${context}.colors must be an object or a two-color array.`);
    assertAllowedKeys(rawField.colors, new Set(["positive", "negative"]), `${context}.colors`);
    positive = normalizeColor(rawField.colors.positive, `${context}.colors.positive`, positive);
    negative = normalizeColor(rawField.colors.negative, `${context}.colors.negative`, negative);
  }

  const opacity = rawField.opacity === undefined
    ? 0.72
    : finiteNumber(rawField.opacity, `${context}.opacity`);
  if (opacity <= 0 || opacity > 1) fail(`${context}.opacity must be greater than zero and at most 1.`);

  let units;
  if (rawField.units !== undefined && rawField.units !== null) {
    units = cleanLine(rawField.units, `${context}.units`, 40);
  }

  let surfaceField;
  if (rawField.surface_field !== undefined) {
    surfaceField = cleanLine(rawField.surface_field, `${context}.surface_field`, 80);
  }

  let metadata;
  if (rawField.metadata !== undefined) {
    if (!isRecord(rawField.metadata)) fail(`${context}.metadata must be an object.`);
    metadata = sanitizeMetadata(rawField.metadata, `${context}.metadata`);
    if (kind === "esp") {
      const method = rawField.metadata.method;
      if (method === "pyscf-exact" || method === "fft-convolution") {
        const expectedApproximate = method === "fft-convolution";
        if (rawField.metadata.approximate !== expectedApproximate) {
          fail(`${context}.metadata.approximate is inconsistent with ${method}.`);
        }
        if (rawField.metadata.units !== "hartree/e") {
          fail(`${context}.metadata.units must be hartree/e for ${method}.`);
        }
      }
      if (rawField.metadata.source === "cube") {
        if (!Number.isSafeInteger(rawField.metadata.dataset) || rawField.metadata.dataset < 0) {
          fail(`${context}.metadata.dataset must be a non-negative integer for cube data.`);
        }
      }
    }
  }

  return {
    name,
    label,
    kind,
    shape,
    origin,
    axes,
    values,
    isovalues,
    colors: { positive, negative },
    opacity,
    units,
    surface_field: surfaceField,
    metadata,
    range: [minimum, maximum],
  };
}

export function validateSceneMessage(message) {
  if (!isRecord(message)) fail("The PyQED message must be an object.");
  assertAllowedKeys(message, new Set(["type", "scene"]), "message");
  if (message.type !== "pyqed:scene") fail("Unsupported message type.");
  if (!isRecord(message.scene)) fail("message.scene must be an object.");
  const scene = message.scene;
  assertAllowedKeys(
    scene,
    new Set(["version", "kind", "title", "molecule", "fields", "active_field"]),
    "message.scene",
  );
  if (scene.version !== 1 || scene.kind !== "pyqed-scene") {
    fail("Only pyqed-scene version 1 is supported.");
  }
  const title = scene.title === undefined ? "PyQED scene" : cleanLine(scene.title, "message.scene.title", 160);
  let molecule = null;
  let atoms = [];
  if (scene.molecule !== null) {
    if (!isRecord(scene.molecule)) fail("message.scene.molecule must be an object or null.");
    assertAllowedKeys(
      scene.molecule,
      new Set(["xyz", "representation", "labels"]),
      "message.scene.molecule",
    );
    const xyz = cleanText(scene.molecule.xyz, "message.scene.molecule.xyz", MAX_GEOMETRY_CHARS);
    atoms = parseGeometry(xyz, "angstrom");
    const representation = scene.molecule.representation ?? "ball-stick";
    if (!REPRESENTATIONS.has(representation)) fail("message.scene.molecule.representation is invalid.");
    const labels = scene.molecule.labels ?? false;
    if (typeof labels !== "boolean") fail("message.scene.molecule.labels must be true or false.");
    molecule = { xyz, representation, labels };
  }

  if (!Array.isArray(scene.fields) || scene.fields.length > MAX_FIELDS) {
    fail(`message.scene.fields must be an array with at most ${MAX_FIELDS} entries.`);
  }
  const fields = [];
  let totalPoints = 0;
  for (let index = 0; index < scene.fields.length; index += 1) {
    const field = validateField(scene.fields[index], index);
    totalPoints += field.values.length;
    if (totalPoints > MAX_TOTAL_GRID_POINTS) {
      fail(`The scene contains more than ${MAX_TOTAL_GRID_POINTS.toLocaleString()} scalar values.`);
    }
    fields.push(field);
  }
  if (!molecule && fields.length === 0) fail("A scene must contain a molecule or at least one scalar field.");
  const names = new Set();
  for (const field of fields) {
    if (names.has(field.name)) fail(`Field name “${field.name}” is duplicated.`);
    names.add(field.name);
  }
  for (const field of fields) {
    if (!field.surface_field) continue;
    if (field.kind !== "esp") fail(`Only an ESP field may reference surface_field.`);
    const target = fields.find((candidate) => candidate.name === field.surface_field);
    if (!target) fail(`ESP field “${field.name}” references missing field “${field.surface_field}”.`);
    if (target.kind !== "electron-density") {
      fail(`ESP field “${field.name}” must reference an electron-density field.`);
    }
    if (!sameGrid(field, target)) fail(`ESP field “${field.name}” and its surface field must use the same grid.`);
  }

  let activeFieldIndex = fields.length > 0 ? 0 : -1;
  if (scene.active_field !== undefined && scene.active_field !== null) {
    if (Number.isInteger(scene.active_field)) {
      activeFieldIndex = scene.active_field;
    } else if (typeof scene.active_field === "string") {
      activeFieldIndex = fields.findIndex((field) => field.name === scene.active_field);
    } else {
      fail("message.scene.active_field must be a field name or integer index.");
    }
    if (activeFieldIndex < 0 || activeFieldIndex >= fields.length) {
      fail("message.scene.active_field does not identify a supplied field.");
    }
  }

  return {
    title,
    atoms,
    molecule,
    fields,
    activeFieldIndex,
  };
}

function parseCubeNumber(value, context) {
  const number = Number(value.replace(/[dD]/, "e"));
  if (!Number.isFinite(number) || Math.abs(number) > MAX_FLOAT32) fail(`${context} is not a finite number.`);
  return number;
}

function cubeLine(lines, index, minimumFields, context) {
  const fields = (lines[index] ?? "").trim().split(/\s+/).filter(Boolean);
  if (fields.length < minimumFields) fail(`${context} is incomplete.`);
  return fields;
}

function inferFieldKind(text, multipleOrbitals) {
  if (multipleOrbitals || /\b(orbital|homo|lumo|mo\s*\d*)\b/i.test(text)) return "orbital";
  if (/\b(electrostatic|esp|potential)\b/i.test(text)) return "esp";
  if (/\bspin[\s_-]*density\b/i.test(text)) return "spin-density";
  if (/\b(difference|difference-density|delta-density)\b/i.test(text)) return "difference-density";
  if (/\btransition[\s_-]*density\b/i.test(text)) return "transition-density";
  if (/\b(density|charge-density|electron-density)\b/i.test(text)) return "electron-density";
  return "generic";
}

function safeName(value, fallback) {
  const normalized = value.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "");
  return (normalized || fallback).slice(0, 80);
}

export function suggestIsovalue(values, kind = "generic") {
  let maximumAbsolute = 0;
  let maximumPositive = 0;
  for (const value of values) {
    maximumAbsolute = Math.max(maximumAbsolute, Math.abs(value));
    maximumPositive = Math.max(maximumPositive, value);
  }
  if (maximumAbsolute === 0) return 1e-6;
  const fraction = kind === "electron-density" ? 0.02 : kind === "esp" ? 0.15 : 0.08;
  return Math.max(Number.EPSILON, (kind === "electron-density" ? maximumPositive : maximumAbsolute) * fraction);
}

export function parseCube(text, options = {}) {
  if (typeof text !== "string") fail("Cube data must be text.");
  if (text.length > MAX_CUBE_FILE_BYTES) fail("Choose a cube file smaller than 80 MB.");
  const lines = text.split(/\r?\n/);
  if (lines.length < 7) fail("Cube file is missing its header or scalar data.");
  const comment1 = (lines[0] || "Cube scalar field").trim() || "Cube scalar field";
  const comment2 = (lines[1] || "Volumetric data").trim() || "Volumetric data";
  const header = cubeLine(lines, 2, 4, "Cube atom/origin line");
  const atomCountRaw = parseCubeNumber(header[0], "Cube atom count");
  if (!Number.isInteger(atomCountRaw) || atomCountRaw === 0) fail("Cube atom count must be a non-zero integer.");
  const atomCount = Math.abs(atomCountRaw);
  if (atomCount > MAX_ATOMS) fail(`Cube files may contain at most ${MAX_ATOMS} atoms.`);
  const rawOrigin = header.slice(1, 4).map((value, index) => parseCubeNumber(value, `Cube origin ${index + 1}`));
  const headerValueCount = header[4] === undefined ? 1 : parseCubeNumber(header[4], "Cube NVAL");
  if (!Number.isInteger(headerValueCount) || headerValueCount < 1 || headerValueCount > MAX_FIELDS) {
    fail(`Cube NVAL must be an integer between 1 and ${MAX_FIELDS}.`);
  }
  if (atomCountRaw < 0 && headerValueCount !== 1) fail("A multi-orbital cube must omit NVAL or set it to one.");

  const rawAxes = [];
  const signedCounts = [];
  for (let axis = 0; axis < 3; axis += 1) {
    const axisLine = cubeLine(lines, 3 + axis, 4, `Cube axis ${axis + 1}`);
    const signedCount = parseCubeNumber(axisLine[0], `Cube axis ${axis + 1} count`);
    if (!Number.isInteger(signedCount) || signedCount === 0) fail(`Cube axis ${axis + 1} count must be a non-zero integer.`);
    signedCounts.push(signedCount);
    rawAxes.push(axisLine.slice(1, 4).map((value, component) =>
      parseCubeNumber(value, `Cube axis ${axis + 1} component ${component + 1}`),
    ));
  }
  const allPositive = signedCounts.every((count) => count > 0);
  const allNegative = signedCounts.every((count) => count < 0);
  if (!allPositive && !allNegative) fail("Cube grid counts must use one consistent unit sign.");
  const unitScale = allPositive ? BOHR_TO_ANGSTROM : 1;
  const shape = signedCounts.map(Math.abs);
  if (shape.some((count) => count < 2 || count > 2048)) fail("Each cube grid dimension must be between 2 and 2048.");
  const pointCount = shape[0] * shape[1] * shape[2];
  if (pointCount > MAX_GRID_POINTS) {
    fail(`Cube grid contains ${pointCount.toLocaleString()} points; the limit is ${MAX_GRID_POINTS.toLocaleString()}.`);
  }
  const origin = rawOrigin.map((value) => value * unitScale);
  const axes = rawAxes.map((axis) => axis.map((value) => value * unitScale));
  if (Math.abs(determinant(axes)) < 1e-15) fail("Cube axes must span a three-dimensional grid.");

  const atoms = [];
  for (let atomIndex = 0; atomIndex < atomCount; atomIndex += 1) {
    const atomLine = cubeLine(lines, 6 + atomIndex, 5, `Cube atom ${atomIndex + 1}`);
    const atomicNumber = parseCubeNumber(atomLine[0], `Cube atom ${atomIndex + 1} atomic number`);
    if (!Number.isInteger(atomicNumber) || atomicNumber < 0 || atomicNumber >= ELEMENT_SYMBOLS.length) {
      fail(`Cube atom ${atomIndex + 1} has an unsupported atomic number.`);
    }
    atoms.push({
      element: ELEMENT_SYMBOLS[atomicNumber],
      x: parseCubeNumber(atomLine[2], `Cube atom ${atomIndex + 1} x`) * unitScale,
      y: parseCubeNumber(atomLine[3], `Cube atom ${atomIndex + 1} y`) * unitScale,
      z: parseCubeNumber(atomLine[4], `Cube atom ${atomIndex + 1} z`) * unitScale,
    });
  }

  function* cubeDataTokens() {
    for (let lineIndex = 6 + atomCount; lineIndex < lines.length; lineIndex += 1) {
      const tokens = lines[lineIndex].match(/\S+/g) ?? [];
      yield* tokens;
    }
  }
  const dataTokens = cubeDataTokens();
  function nextCubeToken(context) {
    const next = dataTokens.next();
    if (next.done) fail(`${context} is missing.`);
    return next.value;
  }
  let datasetIds = [];
  let valueCount = headerValueCount;
  if (atomCountRaw < 0) {
    valueCount = parseCubeNumber(nextCubeToken("Cube dataset count"), "Cube dataset count");
    if (!Number.isInteger(valueCount) || valueCount < 1 || valueCount > MAX_FIELDS) {
      fail(`Cube dataset count must be between 1 and ${MAX_FIELDS}.`);
    }
    if (pointCount * valueCount > MAX_TOTAL_GRID_POINTS) fail("Multi-orbital cube exceeds the total scalar-value limit.");
    datasetIds = Array.from({ length: valueCount }, (_, index) => {
      const identifier = parseCubeNumber(
        nextCubeToken(`Cube dataset identifier ${index + 1}`),
        `Cube dataset identifier ${index + 1}`,
      );
      if (!Number.isInteger(identifier)) fail("Cube dataset identifiers must be integers.");
      return identifier;
    });
  } else {
    datasetIds = Array.from({ length: valueCount }, (_, index) => index + 1);
  }
  if (new Set(datasetIds).size !== datasetIds.length) fail("Cube dataset identifiers must be unique.");
  if (pointCount * valueCount > MAX_TOTAL_GRID_POINTS) fail("Cube file exceeds the total scalar-value limit.");
  const expectedValues = pointCount * valueCount;
  const valuesByDataset = Array.from({ length: valueCount }, () => new Float32Array(pointCount));
  let actualValues = 0;
  for (; actualValues < expectedValues; actualValues += 1) {
    const next = dataTokens.next();
    if (next.done) {
      fail(`Cube data has ${actualValues} values, but C-order grid ${shape.join(" × ")} × ${valueCount} dataset(s) requires ${expectedValues}.`);
    }
    const datasetIndex = actualValues % valueCount;
    const pointIndex = Math.floor(actualValues / valueCount);
    valuesByDataset[datasetIndex][pointIndex] = parseCubeNumber(
      next.value,
      `Cube scalar value ${actualValues + 1}`,
    );
  }
  if (!dataTokens.next().done) {
    fail(`Cube data contains more than the required ${expectedValues} scalar values.`);
  }

  const fileName = typeof options.fileName === "string" ? options.fileName : "";
  const descriptor = `${comment1} ${comment2} ${fileName}`;
  const kind = inferFieldKind(descriptor, atomCountRaw < 0);
  const fields = valuesByDataset.map((values, index) => {
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    for (const value of values) {
      minimum = Math.min(minimum, value);
      maximum = Math.max(maximum, value);
    }
    const datasetLabel = valueCount === 1
      ? comment2.slice(0, 120)
      : `${kind === "orbital" ? "Orbital" : "Dataset"} ${datasetIds[index]}`;
    return {
      name: valueCount === 1
        ? safeName(datasetLabel, `dataset-${datasetIds[index]}`)
        : `${kind === "orbital" ? "orbital" : "dataset"}-${index + 1}-${datasetIds[index]}`,
      label: datasetLabel,
      kind,
      shape: [...shape],
      origin: [...origin],
      axes: axes.map((axis) => [...axis]),
      values,
      isovalues: [suggestIsovalue(values, kind)],
      colors: { positive: "#35d6e3", negative: "#f38b6b" },
      opacity: 0.72,
      units: undefined,
      surface_field: undefined,
      metadata: undefined,
      range: [minimum, maximum],
    };
  });

  return {
    title: comment1.slice(0, 160),
    atoms,
    molecule: {
      xyz: xyzFor(atoms, comment1),
      representation: "ball-stick",
      labels: false,
    },
    fields,
    activeFieldIndex: fields.length > 0 ? 0 : -1,
  };
}

export function fieldToCube(field, atoms) {
  const atomicRows = atoms.map((atom) => {
    const atomicNumber = ATOMIC_NUMBERS.get(canonicalElement(atom.element));
    if (atomicNumber === undefined) fail(`Cannot encode element ${atom.element} in a cube file.`);
    return `${String(atomicNumber).padStart(5)} ${atomicNumber.toFixed(6).padStart(13)} ${(atom.x / BOHR_TO_ANGSTROM).toExponential(6)} ${(atom.y / BOHR_TO_ANGSTROM).toExponential(6)} ${(atom.z / BOHR_TO_ANGSTROM).toExponential(6)}`;
  });
  const origin = field.origin.map((value) => value / BOHR_TO_ANGSTROM);
  const axisRows = field.axes.map((axis, index) =>
    `${String(field.shape[index]).padStart(5)} ${axis.map((value) => (value / BOHR_TO_ANGSTROM).toExponential(6)).join(" ")}`,
  );
  const values = [];
  for (let offset = 0; offset < field.values.length; offset += 6) {
    values.push(Array.from(field.values.subarray(offset, offset + 6), (value) => value.toExponential(6)).join(" "));
  }
  return [
    "PyQED browser scalar field",
    field.label,
    `${String(atoms.length).padStart(5)} ${origin.map((value) => value.toExponential(6)).join(" ")}`,
    ...axisRows,
    ...atomicRows,
    ...values,
    "",
  ].join("\n");
}

export function gridsMatch(left, right) {
  return sameGrid(left, right);
}
