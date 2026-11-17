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
  const probs = Array.from(output[session.outputNames[0]].data);
  return config.classNames
    .map((name, i) => ({ name, pct: Math.round(probs[i] * 10000) / 100 }))
    .sort((a, b) => b.pct - a.pct);
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
  try {
    render(file, await classify(file));
  } catch (err) {
    console.error(err);
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
