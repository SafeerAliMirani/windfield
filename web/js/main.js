// Windfield: real NOAA GFS wind, one forecast hour, pushed through a million
// particles on the GPU. Advection runs in a WGSL compute shader; the trails
// pile up in a float screen buffer that fades a little each frame. Drag to pan,
// scroll to zoom. Coastlines and city dots come from public-domain map data.

const canvas = document.getElementById("gpu");
const $ = (id) => document.getElementById(id);
const setLoading = (t) => { const e = $("loading-text"); if (e) e.textContent = t; };
const hideLoading = () => $("loading")?.classList.add("hidden");
const noGPU = (m) => { $("nogpu")?.classList.remove("hidden"); const e = $("nogpu-msg"); if (e && m) e.innerHTML = m; hideLoading(); };
const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
function formatValid(s) {   // "20260707 06Z" -> "07 Jul 2026 · 06:00 UTC"
  const m = /^(\d{4})(\d{2})(\d{2})\s+(\d{2})Z$/.exec(s || "");
  return m ? `${m[3]} ${MONTHS[+m[2] - 1]} ${m[1]} · ${m[4]}:00 UTC` : (s || "");
}

const N = 1 << 22;          // buffer holds up to ~4.2M; the slider picks how many actually run
const SPEED = 0.00018;      // how far a particle steps per frame, per m/s
const DROP = 0.003;         // baseline chance a particle respawns
const DROP_BUMP = 0.010;    // fast particles respawn a bit more, keeps density even
const trailFade = (p) => 0.90 + p / 100 * 0.09;   // trails slider percent, mapped to how much of the last frame survives
const INTENSITY = 0.55;     // brightness each particle adds
const BG_DIM = 0.16;        // how dim the background speed map sits
const COLOR_REF = 20.0;     // wind speed (m/s) that hits the bright end of the ramp

// a few well-known cities so you can place yourself on the map [name, lon, lat]
const CITIES = [
  ["Tokyo", 139.7, 35.7], ["Beijing", 116.4, 39.9], ["Delhi", 77.2, 28.6], ["Singapore", 103.8, 1.3],
  ["Dubai", 55.3, 25.2], ["Mumbai", 72.9, 19.1], ["Jakarta", 106.8, -6.2], ["Moscow", 37.6, 55.8],
  ["London", -0.1, 51.5], ["Paris", 2.3, 48.9], ["Cairo", 31.2, 30.0], ["Lagos", 3.4, 6.5],
  ["Nairobi", 36.8, -1.3], ["Cape Town", 18.4, -33.9], ["New York", -74.0, 40.7], ["Los Angeles", -118.2, 34.1],
  ["Chicago", -87.6, 41.9], ["Mexico City", -99.1, 19.4], ["Bogota", -74.1, 4.7], ["Lima", -77.0, -12.0],
  ["Sao Paulo", -46.6, -23.5], ["Buenos Aires", -58.4, -34.6], ["Sydney", 151.2, -33.9], ["Reykjavik", -21.9, 64.1],
];

const RAMP = `
fn ramp(t : f32) -> vec3<f32> {
  let a = vec3<f32>(0.03, 0.06, 0.16);
  let b = vec3<f32>(0.00, 0.55, 0.66);
  let c = vec3<f32>(0.98, 0.80, 0.32);
  let d = vec3<f32>(1.00, 1.00, 1.00);
  if (t < 0.5) { return mix(a, b, t / 0.5); }
  if (t < 0.8) { return mix(b, c, (t - 0.5) / 0.3); }
  return mix(c, d, (t - 0.8) / 0.2);
}`;

// cx/cy = map point at screen center, zx/zy = clip units per uv
const VIEW = `struct V { cx:f32, cy:f32, zx:f32, zy:f32 };`;

const ADVECT = `
struct P {
  uMin:f32, uMax:f32, vMin:f32, vMax:f32,
  maxSpeed:f32, speed:f32, dropRate:f32, dropBump:f32,
  frame:u32, n:u32, p0:u32, p1:u32,
};
@group(0) @binding(0) var<storage, read_write> pos : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> P0 : P;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var wind : texture_2d<f32>;

fn hash(n : u32) -> f32 {
  var x = n;
  x = (x ^ 61u) ^ (x >> 16u);
  x = x + (x << 3u);
  x = x ^ (x >> 4u);
  x = x * 0x27d4eb2du;
  x = x ^ (x >> 15u);
  return f32(x) / 4294967295.0;
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= P0.n) { return; }
  var p = pos[i];
  let w = textureSampleLevel(wind, samp, p, 0.0);
  let u = mix(P0.uMin, P0.uMax, w.r);
  let v = mix(P0.vMin, P0.vMax, w.g);
  var np = p + vec2<f32>(u, -v) * P0.speed;   // north wind lifts the particle up the image
  np.x = fract(np.x + 1.0);
  let s01 = clamp(length(vec2<f32>(u, v)) / max(P0.maxSpeed, 1e-3), 0.0, 1.0);
  let drop = P0.dropRate + s01 * P0.dropBump;
  let r = hash(i * 1664525u + P0.frame * 1013904223u);
  if (r < drop || np.y < 0.0 || np.y > 1.0) {
    np = vec2<f32>(hash(i * 7u + P0.frame * 92821u), hash(i * 13u + P0.frame * 53987u));
  }
  pos[i] = np;
}`;

const POINTS = `
struct R { uMin:f32, uMax:f32, vMin:f32, vMax:f32, colorRef:f32, intensity:f32, p0:f32, p1:f32 };
${VIEW}
@group(0) @binding(0) var<storage, read> pos : array<vec2<f32>>;
@group(0) @binding(1) var<uniform> R0 : R;
@group(0) @binding(2) var samp : sampler;
@group(0) @binding(3) var wind : texture_2d<f32>;
@group(0) @binding(4) var<uniform> VP : V;
${RAMP}
struct VO { @builtin(position) pos : vec4<f32>, @location(0) col : vec3<f32> };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VO {
  let p = pos[i];
  var o : VO;
  var dx = p.x - VP.cx; dx = dx - round(dx);   // wrap to the nearest copy of this longitude
  o.pos = vec4<f32>(dx * 2.0 * VP.zx, -(p.y - VP.cy) * 2.0 * VP.zy, 0.0, 1.0);
  let w = textureSampleLevel(wind, samp, p, 0.0);
  let u = mix(R0.uMin, R0.uMax, w.r);
  let v = mix(R0.vMin, R0.vMax, w.g);
  let s = clamp(length(vec2<f32>(u, v)) / max(R0.colorRef, 1e-3), 0.0, 1.0);
  o.col = ramp(s) * R0.intensity;
  return o;
}
@fragment fn fs(in : VO) -> @location(0) vec4<f32> { return vec4<f32>(in.col, 1.0); }`;

const FULLSCREEN_VS = `
struct VO { @builtin(position) pos : vec4<f32>, @location(0) uv : vec2<f32> };
@vertex fn vs(@builtin(vertex_index) i : u32) -> VO {
  var p = array<vec2<f32>, 3>(vec2<f32>(-1.0, -3.0), vec2<f32>(-1.0, 1.0), vec2<f32>(3.0, 1.0));
  var o : VO; let xy = p[i];
  o.pos = vec4<f32>(xy, 0.0, 1.0);
  o.uv = vec2<f32>((xy.x + 1.0) * 0.5, 1.0 - (xy.y + 1.0) * 0.5);
  return o;
}`;

const FADE_SHADER = `
${FULLSCREEN_VS}
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var prev : texture_2d<f32>;
@group(0) @binding(2) var<uniform> F : vec4<f32>;
@fragment fn fs(in : VO) -> @location(0) vec4<f32> {
  return vec4<f32>(textureSample(prev, samp, in.uv).rgb * F.x, 1.0);
}`;

const COMPOSITE = `
${FULLSCREEN_VS}
struct C { uMin:f32, uMax:f32, vMin:f32, vMax:f32, colorRef:f32, bgDim:f32, p0:f32, p1:f32 };
${VIEW}
@group(0) @binding(0) var samp : sampler;
@group(0) @binding(1) var wind : texture_2d<f32>;
@group(0) @binding(2) var trails : texture_2d<f32>;
@group(0) @binding(3) var<uniform> C0 : C;
@group(0) @binding(4) var<uniform> VP : V;
${RAMP}
@fragment fn fs(in : VO) -> @location(0) vec4<f32> {
  // turn this screen pixel back into a map point so the speed map pans with the trails
  let clip = vec2<f32>(in.uv.x * 2.0 - 1.0, 1.0 - in.uv.y * 2.0);
  let du = VP.cx + clip.x / (2.0 * VP.zx);
  let dv = VP.cy - clip.y / (2.0 * VP.zy);
  let inMap = f32(dv >= 0.0 && dv <= 1.0);
  let w = textureSample(wind, samp, vec2<f32>(fract(du), clamp(dv, 0.0, 1.0)));
  let u = mix(C0.uMin, C0.uMax, w.r);
  let v = mix(C0.vMin, C0.vMax, w.g);
  let s = clamp(length(vec2<f32>(u, v)) / max(C0.colorRef, 1e-3), 0.0, 1.0);
  let bg = ramp(s) * C0.bgDim * inMap;
  let tr = textureSample(trails, samp, in.uv).rgb;
  let mapped = vec3<f32>(1.0) - exp(-(bg + tr) * 1.25);   // soft rolloff so bright trails don't flat-clip
  return vec4<f32>(mapped, 1.0);
}`;

// thin coastline strokes, drawn three times (lon-1, lon, lon+1) to survive the wrap
const COAST = `
${VIEW}
@group(0) @binding(0) var<uniform> VP : V;
@vertex fn vs(@location(0) uv : vec2<f32>, @builtin(instance_index) inst : u32) -> @builtin(position) vec4<f32> {
  let off = f32(inst) - 1.0;
  let dx = (uv.x + off) - VP.cx;
  return vec4<f32>(dx * 2.0 * VP.zx, -(uv.y - VP.cy) * 2.0 * VP.zy, 0.0, 1.0);
}
@fragment fn fs() -> @location(0) vec4<f32> { return vec4<f32>(0.82, 0.90, 1.0, 0.5); }`;

async function main() {
  if (!navigator.gpu) { noGPU("Open it in desktop Chrome or Edge."); return; }

  setLoading("loading real wind data");
  let meta, bmp;
  try {
    meta = await fetch("data/wind.json").then((r) => { if (!r.ok) throw 0; return r.json(); });
    const blob = await fetch("data/wind.png").then((r) => { if (!r.ok) throw 0; return r.blob(); });
    bmp = await createImageBitmap(blob);
  } catch (e) {
    noGPU("No wind data yet. Run <code>python convert.py</code> first, then reload.");
    return;
  }
  // coastline is optional; the map still works if it's missing
  let coast = null;
  try { coast = await fetch("data/coastline.json").then((r) => (r.ok ? r.json() : null)); } catch (e) { coast = null; }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) { noGPU("No compatible GPU adapter."); return; }
  const device = await adapter.requestDevice();
  const ctx = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  ctx.configure({ device, format, alphaMode: "opaque" });
  const SCREEN_FORMAT = "rgba16float";

  const wind = device.createTexture({
    size: [bmp.width, bmp.height], format: "rgba8unorm",
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bmp }, { texture: wind }, [bmp.width, bmp.height]);
  const windView = wind.createView();
  const samp = device.createSampler({ magFilter: "linear", minFilter: "linear", addressModeU: "repeat", addressModeV: "clamp-to-edge" });

  const maxSpeed = Math.hypot(
    Math.max(Math.abs(meta.uMin), Math.abs(meta.uMax)),
    Math.max(Math.abs(meta.vMin), Math.abs(meta.vMax)));

  // scatter the particles at random and hand them to the GPU
  const seed = new Float32Array(N * 2);
  for (let i = 0; i < seed.length; i++) seed[i] = Math.random();
  const posBuf = device.createBuffer({ size: seed.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(posBuf, 0, seed);

  const advectU = device.createBuffer({ size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const pointU = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const fadeU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const compU = device.createBuffer({ size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  const viewU = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(pointU, 0, new Float32Array([meta.uMin, meta.uMax, meta.vMin, meta.vMax, COLOR_REF, INTENSITY, 0, 0]));
  device.queue.writeBuffer(fadeU, 0, new Float32Array([trailFade(60), 0, 0, 0]));
  device.queue.writeBuffer(compU, 0, new Float32Array([meta.uMin, meta.uMax, meta.vMin, meta.vMax, COLOR_REF, BG_DIM, 0, 0]));

  // pack the coastline polylines into line segments in uv space
  let coastBuf = null, coastCount = 0;
  if (coast) {
    const segs = [];
    for (const line of coast) {
      for (let i = 0; i + 1 < line.length; i++) {
        const a = line[i], b = line[i + 1];
        const ax = (((a[0] % 360) + 360) % 360) / 360, ay = (90 - a[1]) / 180;
        const bx = (((b[0] % 360) + 360) % 360) / 360, by = (90 - b[1]) / 180;
        if (Math.abs(ax - bx) > 0.5) continue;   // don't draw the segment that jumps the dateline
        segs.push(ax, ay, bx, by);
      }
    }
    const arr = new Float32Array(segs);
    coastBuf = device.createBuffer({ size: arr.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    device.queue.writeBuffer(coastBuf, 0, arr);
    coastCount = arr.length / 2;
  }

  const advectMod = device.createShaderModule({ code: ADVECT });
  const advectPipe = device.createComputePipeline({ layout: "auto", compute: { module: advectMod, entryPoint: "main" } });
  const advectBind = device.createBindGroup({
    layout: advectPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: advectU } }, { binding: 2, resource: samp }, { binding: 3, resource: windView }],
  });

  const pointMod = device.createShaderModule({ code: POINTS });
  const pointPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: pointMod, entryPoint: "vs" },
    fragment: { module: pointMod, entryPoint: "fs", targets: [{ format: SCREEN_FORMAT, blend: { color: { srcFactor: "one", dstFactor: "one", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one", operation: "add" } } }] },
    primitive: { topology: "point-list" },
  });
  const pointBind = device.createBindGroup({
    layout: pointPipe.getBindGroupLayout(0),
    entries: [{ binding: 0, resource: { buffer: posBuf } }, { binding: 1, resource: { buffer: pointU } }, { binding: 2, resource: samp }, { binding: 3, resource: windView }, { binding: 4, resource: { buffer: viewU } }],
  });

  const fadeMod = device.createShaderModule({ code: FADE_SHADER });
  const fadePipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: fadeMod, entryPoint: "vs" },
    fragment: { module: fadeMod, entryPoint: "fs", targets: [{ format: SCREEN_FORMAT }] },
    primitive: { topology: "triangle-list" },
  });

  const compMod = device.createShaderModule({ code: COMPOSITE });
  const compPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: compMod, entryPoint: "vs" },
    fragment: { module: compMod, entryPoint: "fs", targets: [{ format }] },
    primitive: { topology: "triangle-list" },
  });

  const coastMod = device.createShaderModule({ code: COAST });
  const coastPipe = device.createRenderPipeline({
    layout: "auto",
    vertex: { module: coastMod, entryPoint: "vs", buffers: [{ arrayStride: 8, attributes: [{ shaderLocation: 0, offset: 0, format: "float32x2" }] }] },
    fragment: { module: coastMod, entryPoint: "fs", targets: [{ format, blend: { color: { srcFactor: "src-alpha", dstFactor: "one-minus-src-alpha", operation: "add" }, alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" } } }] },
    primitive: { topology: "line-list" },
  });
  const coastBind = device.createBindGroup({ layout: coastPipe.getBindGroupLayout(0), entries: [{ binding: 0, resource: { buffer: viewU } }] });

  // two screen buffers we ping-pong; rebuilt on resize
  let screens, fadeBinds, compBinds;
  function build() {
    const d = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(2, Math.round(window.innerWidth * d));
    canvas.height = Math.max(2, Math.round(window.innerHeight * d));
    screens = [0, 1].map(() => device.createTexture({
      size: [canvas.width, canvas.height], format: SCREEN_FORMAT,
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    }));
    fadeBinds = screens.map((t) => device.createBindGroup({
      layout: fadePipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: samp }, { binding: 1, resource: t.createView() }, { binding: 2, resource: { buffer: fadeU } }],
    }));
    compBinds = screens.map((t) => device.createBindGroup({
      layout: compPipe.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: samp }, { binding: 1, resource: windView }, { binding: 2, resource: t.createView() }, { binding: 3, resource: { buffer: compU } }, { binding: 4, resource: { buffer: viewU } }],
    }));
  }
  window.addEventListener("resize", build);
  build();

  const view = { cx: 0.5, cy: 0.5, zoom: 1 };
  let activeN = 1 << 20, speedMul = 1;
  const zoomXY = () => { const A = canvas.width / canvas.height, fit = Math.min(1, 2 / A); const zx = fit * view.zoom; return [zx, zx * A / 2]; };
  const clampCy = () => { const zy = zoomXY()[1]; const h = 0.5 / zy; view.cy = h >= 0.5 ? 0.5 : Math.max(h, Math.min(1 - h, view.cy)); };

  // one finger or mouse drags to pan; two fingers pinch to zoom; wheel zooms on desktop
  const panBy = (dx, dy) => {
    const [zx, zy] = zoomXY();
    view.cx = ((view.cx - dx / (zx * window.innerWidth)) % 1 + 1) % 1;
    view.cy = view.cy - dy / (zy * window.innerHeight);
    clampCy();
  };
  const zoomAt = (sx, sy, factor) => {
    let [zx, zy] = zoomXY();
    const clipX = 2 * (sx / window.innerWidth) - 1, clipY = 1 - 2 * (sy / window.innerHeight);
    const du = view.cx + clipX / (2 * zx), dv = view.cy - clipY / (2 * zy);   // point under the cursor
    view.zoom = Math.min(40, Math.max(1, view.zoom * factor));
    [zx, zy] = zoomXY();
    view.cx = ((du - clipX / (2 * zx)) % 1 + 1) % 1;                          // keep it under the cursor
    view.cy = dv + clipY / (2 * zy);
    clampCy();
  };
  const ptrs = new Map();
  let pinch = 0;
  canvas.style.cursor = "grab";
  canvas.style.touchAction = "none";
  canvas.addEventListener("pointerdown", (e) => { ptrs.set(e.pointerId, { x: e.clientX, y: e.clientY }); canvas.setPointerCapture(e.pointerId); canvas.style.cursor = "grabbing"; });
  const endPtr = (e) => { ptrs.delete(e.pointerId); if (ptrs.size < 2) pinch = 0; if (ptrs.size === 0) canvas.style.cursor = "grab"; };
  canvas.addEventListener("pointerup", endPtr);
  canvas.addEventListener("pointercancel", endPtr);
  canvas.addEventListener("pointermove", (e) => {
    const p = ptrs.get(e.pointerId); if (!p) return;
    const px = p.x, py = p.y; p.x = e.clientX; p.y = e.clientY;
    if (ptrs.size === 1) {
      panBy(e.clientX - px, e.clientY - py);
    } else if (ptrs.size === 2) {
      const a = [...ptrs.values()];
      const d = Math.hypot(a[0].x - a[1].x, a[0].y - a[1].y);
      if (pinch > 0 && d > 0) zoomAt((a[0].x + a[1].x) / 2, (a[0].y + a[1].y) / 2, d / pinch);
      pinch = d;
    }
  });
  canvas.addEventListener("wheel", (e) => { e.preventDefault(); zoomAt(e.clientX, e.clientY, Math.exp(-e.deltaY * 0.0012)); }, { passive: false });

  const fmtCount = (n) => (n >= 1e6 ? (n / 1e6).toFixed(1) + "M" : Math.round(n / 1e3) + "K");
  const setFill = (el) => el && el.style.setProperty("--fill", 100 * (el.value - el.min) / (el.max - el.min) + "%");
  const countEl = $("count");
  const cN = $("c-n"), cF = $("c-f"), cT = $("c-t");
  if (cN) cN.oninput = () => { activeN = 2 ** (+cN.value); const o = $("o-n"); if (o) o.textContent = fmtCount(activeN); if (countEl) countEl.textContent = activeN.toLocaleString(); setFill(cN); };
  if (cF) cF.oninput = () => { speedMul = +cF.value; const o = $("o-f"); if (o) o.textContent = (+cF.value).toFixed(2).replace(/0$/, "") + "×"; setFill(cF); };
  if (cT) cT.oninput = () => { device.queue.writeBuffer(fadeU, 0, new Float32Array([trailFade(+cT.value), 0, 0, 0])); const o = $("o-t"); if (o) o.textContent = cT.value + "%"; setFill(cT); };
  [cN, cF, cT].forEach(setFill);

  const hint = $("hint");
  const dropHint = () => { if (hint) hint.style.opacity = "0"; };
  window.addEventListener("pointerdown", dropHint, { once: true });
  window.addEventListener("wheel", dropHint, { once: true });

  const aboutEl = $("about"), aboutBtn = $("aboutBtn"), aboutClose = $("aboutClose");
  if (aboutBtn) aboutBtn.onclick = () => aboutEl && aboutEl.classList.add("open");
  if (aboutClose) aboutClose.onclick = () => aboutEl && aboutEl.classList.remove("open");
  if (aboutEl) aboutEl.addEventListener("click", (e) => { if (e.target === aboutEl) aboutEl.classList.remove("open"); });

  // city labels: one span each, moved to the right screen spot every frame
  const labelWrap = $("labels");
  const labels = CITIES.map(([name, lon, lat]) => {
    const el = document.createElement("span");
    el.textContent = name;
    labelWrap && labelWrap.appendChild(el);
    return { el, ux: (((lon % 360) + 360) % 360) / 360, uy: (90 - lat) / 180 };
  });

  let frame = 0, cur = 0, last = performance.now(), acc = 0, fc = 0;
  const validEl = $("valid"), fpsEl = $("fps");
  const au = new ArrayBuffer(48), auf = new Float32Array(au), aui = new Uint32Array(au);
  auf[0] = meta.uMin; auf[1] = meta.uMax; auf[2] = meta.vMin; auf[3] = meta.vMax;
  auf[4] = maxSpeed; auf[6] = DROP; auf[7] = DROP_BUMP;

  function tick() {
    aui[8] = frame >>> 0; aui[9] = activeN; auf[5] = SPEED * speedMul;
    device.queue.writeBuffer(advectU, 0, au);
    const [zx, zy] = zoomXY();
    device.queue.writeBuffer(viewU, 0, new Float32Array([view.cx, view.cy, zx, zy]));
    const nxt = cur ^ 1;
    const enc = device.createCommandEncoder();

    const cp = enc.beginComputePass();
    cp.setPipeline(advectPipe); cp.setBindGroup(0, advectBind);
    cp.dispatchWorkgroups(Math.ceil(activeN / 256)); cp.end();

    // fade the old trails, add this frame's particles on top
    const rp = enc.beginRenderPass({ colorAttachments: [{ view: screens[nxt].createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp.setPipeline(fadePipe); rp.setBindGroup(0, fadeBinds[cur]); rp.draw(3);
    rp.setPipeline(pointPipe); rp.setBindGroup(0, pointBind); rp.draw(activeN);
    rp.end();

    // speed map + trails to the screen, then coastlines over the top
    const rp2 = enc.beginRenderPass({ colorAttachments: [{ view: ctx.getCurrentTexture().createView(), clearValue: { r: 0, g: 0, b: 0, a: 1 }, loadOp: "clear", storeOp: "store" }] });
    rp2.setPipeline(compPipe); rp2.setBindGroup(0, compBinds[nxt]); rp2.draw(3);
    if (coastBuf) { rp2.setPipeline(coastPipe); rp2.setVertexBuffer(0, coastBuf); rp2.setBindGroup(0, coastBind); rp2.draw(coastCount, 3); }
    rp2.end();

    device.queue.submit([enc.finish()]);

    // move the city labels to match the view
    for (const L of labels) {
      let dx = L.ux - view.cx; dx -= Math.round(dx);
      const clx = dx * 2 * zx, cly = -(L.uy - view.cy) * 2 * zy;
      if (Math.abs(clx) <= 1 && Math.abs(cly) <= 1) {
        L.el.style.display = "";
        L.el.style.left = (clx * 0.5 + 0.5) * 100 + "%";
        L.el.style.top = (0.5 - cly * 0.5) * 100 + "%";
      } else {
        L.el.style.display = "none";
      }
    }

    const now = performance.now(); acc += now - last; last = now; fc++;
    if (fc >= 30) { const fps = Math.round(1000 / (acc / fc)); acc = 0; fc = 0; if (fpsEl) fpsEl.textContent = String(fps); }
    cur = nxt; frame++;
    requestAnimationFrame(tick);
  }

  hideLoading();
  if (validEl) validEl.textContent = formatValid(meta.forecastTime);
  if (countEl) countEl.textContent = activeN.toLocaleString();
  requestAnimationFrame(tick);
  console.log(`Windfield running: ${activeN.toLocaleString()} of ${N.toLocaleString()} particles, GFS ${meta.forecastTime}`);
}

main();
