// In-browser prediction for the static site. Same pipeline as web/app.py:
// decode -> RGB -> Pillow-identical bicubic resize to 128x128 -> model (which does its
// own 0-255 -> 0-1 rescaling) -> class probabilities.
import { resizeRGB, rgbaToRgb } from "./preprocess.js";

const config = JSON.parse(document.getElementById("config").textContent);
const MAX_PIXELS = 89_478_485; // Pillow's decompression-bomb threshold
const $ = (id) => document.getElementById(id);

ort.env.wasm.wasmPaths = config.wasmPath;
ort.env.wasm.numThreads = 1; // multi-threading needs cross-origin isolation; not worth it here

let session = null;

function showError(message) {
  $("error").textContent = message;
  $("error").hidden = !message;
}

async function loadModel() {
  try {
    session = await ort.InferenceSession.create(config.modelUrl, {
      executionProviders: ["wasm"],
    });
    $("status").textContent = "Choose an MRI slice to classify.";
    $("submit").disabled = false;
  } catch (err) {
    console.error(err);
    $("status").textContent = "The model failed to load.";
    showError("Couldn't load the model. Check your connection and reload the page.");
  }
}

// Preferred decoder: WebCodecs ImageDecoder gives exactly Pillow's pixels (verified on
// JPEG and PNG). Canvas readback is the fallback for browsers without ImageDecoder; it
// can be off by 1 on a few pixels (browser anti-fingerprinting noise), which in testing
// never changed a prediction.
async function decodeWithImageDecoder(file) {
  const decoder = new ImageDecoder({
    data: await file.arrayBuffer(),
    type: file.type,
    colorSpaceConversion: "none",
    preferAnimation: false,
  });
  try {
    const { image } = await decoder.decode();
    try {
      const width = image.displayWidth;
      const height = image.displayHeight;
      if (width * height > MAX_PIXELS) throw new RangeError("too many pixels");
      const rgba = new Uint8Array(image.allocationSize({ format: "RGBA" }));
      await image.copyTo(rgba, { format: "RGBA" });
      return { rgb: rgbaToRgb(rgba), width, height };
    } finally {
      image.close();
    }
  } finally {
    decoder.close();
  }
}

async function decode(file) {
  if (
    "ImageDecoder" in window &&
    file.type &&
    (await ImageDecoder.isTypeSupported(file.type))
  ) {
    return decodeWithImageDecoder(file);
  }
  return decodeWithCanvas(file);
}

async function decodeWithCanvas(file) {
  // colorSpaceConversion "none": use the file's raw pixel values, like Pillow does.
  const bitmap = await createImageBitmap(file, {
    colorSpaceConversion: "none",
    premultiplyAlpha: "none",
  });
  const { width, height } = bitmap;
  if (width * height > MAX_PIXELS) throw new RangeError("too many pixels");
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0);
  bitmap.close();
  const rgba = ctx.getImageData(0, 0, width, height).data;
  return { rgb: rgbaToRgb(rgba), width, height };
}

class BadImageError extends Error {}

async function classify(file) {
  let decoded;
  try {
    decoded = await decode(file);
  } catch (err) {
    throw new BadImageError(err.message);
  }
  const { rgb, width, height } = decoded;
  const size = config.imgSize;
  const pixels = resizeRGB(rgb, width, height, size, size);
  const input = new ort.Tensor("float32", Float32Array.from(pixels), [1, size, size, 3]);
  const output = await session.run({ [session.inputNames[0]]: input });
  const probs = Array.from((output.probabilities ?? output[session.outputNames[0]]).data);
  const ranked = config.classNames
    .map((name, i) => ({ name, index: i, pct: Math.round(probs[i] * 10000) / 100 }))
    .sort((a, b) => b.pct - a.pct);
  // Grad-CAM grid for the top class, if the model provides one (see export_onnx.py).
  const cam = output.cam ? camForClass(output.cam, ranked[0].index) : null;
  return { ranked, cam };
}

// cam tensor: [1, h, w, classes] -> {h, w, values} for one class, ReLU'd, scaled to 0-1.
function camForClass(tensor, classIndex) {
  const [, h, w, classes] = tensor.dims;
  const values = new Float32Array(h * w);
  let peak = 0;
  for (let i = 0; i < h * w; i++) {
    const v = Math.max(tensor.data[i * classes + classIndex], 0);
    values[i] = v;
    if (v > peak) peak = v;
  }
  if (peak > 0) for (let i = 0; i < values.length; i++) values[i] /= peak;
  return { h, w, values };
}

// Same colour stops as HEATMAP_STOPS in web/app.py.
const STOPS = [
  [0, 0, 128],
  [0, 160, 255],
  [60, 220, 120],
  [255, 220, 0],
  [230, 30, 30],
];
const PREVIEW_SIZE = 512;

function sampleBilinear({ h, w, values }, x, y) {
  // Pixel-centre aligned, clamped at the edges.
  const fx = Math.min(Math.max(x - 0.5, 0), w - 1);
  const fy = Math.min(Math.max(y - 0.5, 0), h - 1);
  const x0 = Math.floor(fx), y0 = Math.floor(fy);
  const x1 = Math.min(x0 + 1, w - 1), y1 = Math.min(y0 + 1, h - 1);
  const ax = fx - x0, ay = fy - y0;
  const top = values[y0 * w + x0] * (1 - ax) + values[y0 * w + x1] * ax;
  const bottom = values[y1 * w + x0] * (1 - ax) + values[y1 * w + x1] * ax;
  return top * (1 - ay) + bottom * ay;
}

async function renderHeatmap(file, cam, label) {
  const bitmap = await createImageBitmap(file);
  const scale = Math.min(1, PREVIEW_SIZE / Math.max(bitmap.width, bitmap.height));
  const width = Math.max(1, Math.round(bitmap.width * scale));
  const height = Math.max(1, Math.round(bitmap.height * scale));
  const canvas = $("heatmap");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(bitmap, 0, 0, width, height);
  bitmap.close();

  // The model saw the whole image squashed to a square, so the grid maps onto the full
  // image by scaling each axis independently.
  const image = ctx.getImageData(0, 0, width, height);
  const px = image.data;
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const v = sampleBilinear(cam, ((x + 0.5) * cam.w) / width, ((y + 0.5) * cam.h) / height);
      const t = v * (STOPS.length - 1);
      const lo = Math.min(Math.floor(t), STOPS.length - 2);
      const f = t - lo;
      const alpha = 0.15 + 0.5 * v; // low attention stays mostly see-through
      const o = (y * width + x) * 4;
      for (let c = 0; c < 3; c++) {
        const colour = STOPS[lo][c] * (1 - f) + STOPS[lo + 1][c] * f;
        px[o + c] = px[o + c] * (1 - alpha) + colour * alpha;
      }
    }
  }
  ctx.putImageData(image, 0, 0);
  $("heatmap-label").textContent = label;
  $("heatmap-box").hidden = false;
}

function render(file, ranked) {
  const preview = $("preview");
  if (preview.src) URL.revokeObjectURL(preview.src);
  preview.src = URL.createObjectURL(file);

  const top = ranked[0];
  $("label").textContent = top.name;
  $("label").className = "predicted-label" + (top.name === "notumor" ? " notumor" : "");
  $("confidence").textContent = `${top.pct}% confidence`;

  const rows = ranked.map(({ name, pct }) => {
    const row = document.createElement("div");
    row.className = "bar-row";
    const label = document.createElement("span");
    label.textContent = name;
    const track = document.createElement("span");
    track.className = "bar-track";
    const fill = document.createElement("span");
    fill.className = "bar-fill";
    fill.style.width = `${pct}%`;
    track.append(fill);
    const value = document.createElement("span");
    value.textContent = `${pct}%`;
    row.append(label, track, value);
    return row;
  });
  $("breakdown").replaceChildren(...rows);
  $("result").hidden = false;
}

$("form").addEventListener("submit", async (event) => {
  event.preventDefault();
  showError("");
  const file = $("file").files[0];
  if (!file) {
    showError("Please choose an MRI image to upload.");
    return;
  }
  if (file.size > config.maxUploadMb * 1024 * 1024) {
    showError(`That file is too large. The upload limit is ${config.maxUploadMb} MB.`);
    return;
  }
  $("submit").disabled = true;
  $("heatmap-box").hidden = true;
  try {
    const { ranked, cam } = await classify(file);
    render(file, ranked);
    if (cam) {
      // The heatmap is extra: if drawing it fails, keep the prediction on screen.
      await renderHeatmap(file, cam, ranked[0].name).catch((err) => console.error(err));
    }
  } catch (err) {
    // A bad upload is the visitor's mistake, not a site error: warn, don't error, so
    // console errors (which monitoring treats as failures) only mean real breakage.
    if (err instanceof BadImageError) console.warn("Unreadable image:", err.message);
    else console.error(err);
    $("result").hidden = true;
    showError(
      err instanceof BadImageError
        ? "Couldn't read that file as an image. Please upload a JPG or PNG."
        : "Something went wrong running the model. Please reload the page and try again.",
    );
  } finally {
    $("submit").disabled = false;
  }
});

loadModel();
