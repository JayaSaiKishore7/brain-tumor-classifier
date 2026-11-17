// Image preprocessing for in-browser inference, matching the training pipeline exactly.
//
// Training (src/train.py) and the Flask app do: PIL Image.convert("RGB").resize((128, 128)),
// i.e. Pillow's default BICUBIC resample with antialiasing. Browser canvas scaling uses a
// different filter, which would feed the model slightly different pixels than it was
// trained on. This is a port of Pillow's resampler (src/libImaging/Resample.c, 8-bit
// path: fixed-point coefficients, horizontal pass then vertical, uint8 rounding between
// passes), so the model gets the same pixels in the browser as in Python.
// tests/test_preprocess_js.py checks this against Pillow, pixel for pixel.

const PRECISION_BITS = 32 - 8 - 2;
const SCALE = 2 ** PRECISION_BITS;
const HALF = 2 ** (PRECISION_BITS - 1);
const A = -0.5; // Pillow's bicubic "a"

function bicubic(x) {
  x = Math.abs(x);
  if (x < 1) return ((A + 2) * x - (A + 3)) * x * x + 1;
  if (x < 2) return (((x - 5) * x + 8) * x - 4) * A;
  return 0;
}

function precomputeCoeffs(inSize, outSize) {
  const scale = inSize / outSize;
  const filterscale = Math.max(scale, 1);
  const support = 2 * filterscale; // bicubic support = 2
  const ss = 1 / filterscale;
  const out = [];
  for (let xx = 0; xx < outSize; xx++) {
    const center = (xx + 0.5) * scale;
    let xmin = Math.trunc(center - support + 0.5);
    if (xmin < 0) xmin = 0;
    let xmax = Math.trunc(center + support + 0.5);
    if (xmax > inSize) xmax = inSize;
    const n = xmax - xmin;
    const k = new Float64Array(n);
    let ww = 0;
    for (let x = 0; x < n; x++) {
      const w = bicubic((x + xmin - center + 0.5) * ss);
      k[x] = w;
      ww += w;
    }
    const kk = new Int32Array(n);
    for (let x = 0; x < n; x++) {
      const v = ww !== 0 ? k[x] / ww : k[x];
      kk[x] = v < 0 ? Math.trunc(-0.5 + v * SCALE) : Math.trunc(0.5 + v * SCALE);
    }
    out.push({ xmin, k: kk });
  }
  return out;
}

function clip8(ss) {
  const v = Math.floor(ss / SCALE);
  return v < 0 ? 0 : v > 255 ? 255 : v;
}

// rgb: Uint8Array of width*height*3 (RGB, row-major). Returns Uint8Array outW*outH*3.
export function resizeRGB(rgb, width, height, outW, outH) {
  let src = rgb;
  let w = width;
  if (outW !== width) {
    const coeffs = precomputeCoeffs(width, outW);
    const tmp = new Uint8Array(outW * height * 3);
    for (let y = 0; y < height; y++) {
      const row = y * width * 3;
      for (let xx = 0; xx < outW; xx++) {
        const { xmin, k } = coeffs[xx];
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let x = 0; x < k.length; x++) {
          const p = row + (xmin + x) * 3;
          s0 += src[p] * k[x];
          s1 += src[p + 1] * k[x];
          s2 += src[p + 2] * k[x];
        }
        const o = (y * outW + xx) * 3;
        tmp[o] = clip8(s0);
        tmp[o + 1] = clip8(s1);
        tmp[o + 2] = clip8(s2);
      }
    }
    src = tmp;
    w = outW;
  }
  if (outH !== height) {
    const coeffs = precomputeCoeffs(height, outH);
    const tmp = new Uint8Array(w * outH * 3);
    for (let yy = 0; yy < outH; yy++) {
      const { xmin: ymin, k } = coeffs[yy];
      for (let x = 0; x < w; x++) {
        let s0 = HALF, s1 = HALF, s2 = HALF;
        for (let y = 0; y < k.length; y++) {
          const p = ((ymin + y) * w + x) * 3;
          s0 += src[p] * k[y];
          s1 += src[p + 1] * k[y];
          s2 += src[p + 2] * k[y];
        }
        const o = (yy * w + x) * 3;
        tmp[o] = clip8(s0);
        tmp[o + 1] = clip8(s1);
        tmp[o + 2] = clip8(s2);
      }
    }
    src = tmp;
  }
  return src;
}

// RGBA (e.g. canvas ImageData.data) -> RGB, dropping alpha like PIL's convert("RGB").
export function rgbaToRgb(rgba) {
  const n = rgba.length / 4;
  const rgb = new Uint8Array(n * 3);
  for (let i = 0; i < n; i++) {
    rgb[i * 3] = rgba[i * 4];
    rgb[i * 3 + 1] = rgba[i * 4 + 1];
    rgb[i * 3 + 2] = rgba[i * 4 + 2];
  }
  return rgb;
}
