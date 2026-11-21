// In-browser engine: runs the patched Qwen3.5-2B on the visitor's GPU (WebGPU) and returns the same shape as
// lab.py's run(), so index.html draws either one. Mirrors lab.py; keep the two in sync.
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/ort.webgpu.min.mjs";
import { Tokenizer } from "https://cdn.jsdelivr.net/npm/@huggingface/tokenizers@0.2/+esm";

const MODEL = "onnx-community/Qwen3.5-2B-ONNX-OPT";
// ?hf=<base url> loads the model files from elsewhere (dev: a local copy, so you don't re-download 1.4 GB)
const HF = new URLSearchParams(location.search).get("hf") ?? `https://huggingface.co/${MODEL}/resolve/main/`;
const HERE = new URL("./model/", import.meta.url);
const TOP_K = 8;
const TEMPLATES = ["Tell me about {}", "I keep thinking about {}", "Here is a fact about {}", "The topic is {}"];
const STOP = new Set([248044, 248046]); // eos, <|im_end|>
const chat = (text) => `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`;

ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.31.1/dist/";

let tok, dec, emb, meta, baseline, mlp; // mlp[L] = {up, down}: 4-bit weight slices for the edges
const conceptCache = new Map();

// fetch once, then serve from the Cache API on later visits
async function download(url, onProgress) {
  const cache = await caches.open("neuron-brain-v1");
  const hit = await cache.match(url);
  if (hit) return new Uint8Array(await hit.arrayBuffer());
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed (${res.status}): ${url}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    onProgress?.(got, total);
  }
  const bytes = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { bytes.set(c, o); o += c.length; }
  await cache.put(url, new Response(bytes)).catch(() => {}); // quota errors just mean no caching
  return bytes;
}

export async function load(status) {
  if (!navigator.gpu) throw new Error("This browser has no WebGPU. Try a recent Chrome, Edge or Safari.");
  const mb = (n) => (n / 1e6).toFixed(0);
  status("loading tokenizer…");
  const [tj, tc] = await Promise.all(["tokenizer.json", "tokenizer_config.json"].map((f) => fetch(HF + f).then((r) => r.json())));
  tok = new Tokenizer(tj, tc);
  meta = await fetch(new URL("decoder.weights.json", HERE)).then((r) => r.json());
  baseline = new Float32Array(await fetch(new URL("baseline.bin", HERE)).then((r) => r.arrayBuffer()));

  const embData = await download(HF + "onnx/embed_tokens_q4f16.onnx_data", (g, t) => status(`downloading embeddings ${mb(g)}/${mb(t)} MB`));
  const decData = await download(HF + "onnx/" + meta.file, (g, t) => status(`downloading model ${mb(g)}/${mb(t)} MB (once, then cached)`));
  // keep copies of the MLP weights so we can draw edges; the big buffer is released after the session loads
  mlp = meta.layers.map((l) => Object.fromEntries(["up_proj", "down_proj"].map((p) => {
    const w = l[p], slice = ([off, len]) => decData.slice(off, off + len);
    return [p, { ...w, q: slice(w.quant), s: new Uint16Array(slice(w.scales).buffer), z: slice(w.zp), cache: new Map() }];
  })));

  status("starting the model on your GPU…");
  const opts = { executionProviders: ["webgpu"], graphOptimizationLevel: "all" };
  emb = await ort.InferenceSession.create(HF + "onnx/embed_tokens_q4f16.onnx",
    { ...opts, externalData: [{ path: "embed_tokens_q4f16.onnx_data", data: embData }] });
  dec = await ort.InferenceSession.create(new URL("decoder.onnx", HERE).href, {
    ...opts,
    externalData: [{ path: meta.file, data: decData }],
    // cache states stay on the GPU between steps instead of round-tripping through JS
    preferredOutputLocation: Object.fromEntries(meta.states.map((n) => [n, "gpu-buffer"])),
  });
  status("");
  return { model: MODEL, n_layers: meta.n_layers };
}

// ---- model calls -------------------------------------------------------------------------------------------

const encode = (text) => tok.encode(text).ids;

function emptyState() {
  const s = {};
  for (const name of dec.inputNames.filter((n) => n.startsWith("past"))) {
    const shape = name.startsWith("past_conv") ? [1, 6144, 3] : name.startsWith("past_recurrent") ? [1, 16, 128, 128] : [1, 2, 0, 256];
    s[name] = new ort.Tensor("float16", new Uint16Array(shape.reduce((a, b) => a * b, 1)), shape);
  }
  return s;
}

async function step(ids, inject, state, past) {
  const n = ids.length;
  const { inputs_embeds } = await emb.run({ input_ids: new ort.Tensor("int64", BigInt64Array.from(ids, BigInt), [1, n]) });
  const pos = new BigInt64Array(3 * n);
  for (let r = 0; r < 3; r++) for (let i = 0; i < n; i++) pos[r * n + i] = BigInt(past + i);
  const out = await dec.run({
    ...state,
    inputs_embeds,
    attention_mask: new ort.Tensor("int64", new BigInt64Array(past + n).fill(1n), [1, past + n]),
    position_ids: new ort.Tensor("int64", pos, [3, 1, n]),
    num_logits_to_keep: new ort.Tensor("int64", BigInt64Array.of(1n), []),
    inject: new ort.Tensor("float32", inject, [meta.n_layers, meta.hidden]),
  });
  inputs_embeds.dispose();
  for (const t of Object.values(state)) if (t.location === "gpu-buffer") t.dispose();
  const next = {};
  for (const name of meta.states) next[name.replace("present_", "past_").replace("present.", "past_key_values.")] = out[name];
  return { logits: out.logits.data, neurons: out.neurons, resid: out.resid, state: next };
}

const half = (h) => {
  const e = (h >> 10) & 31, f = h & 1023, s = h & 0x8000 ? -1 : 1;
  return e === 0 ? s * f * 2 ** -24 : e === 31 ? (f ? NaN : s * Infinity) : s * (1 + f / 1024) * 2 ** (e - 15);
};

function argmax(logits) { // fp16 bits
  let best = 0, bv = -Infinity;
  for (let i = 0; i < logits.length; i++) { const v = half(logits[i]); if (v > bv) { bv = v; best = i; } }
  return best;
}

// prompt pass (returns its neurons + resid) then greedy decoding
async function generate(ids, inject, n) {
  let r = await step(ids, inject, emptyState(), 0);
  const first = { neurons: r.neurons, resid: r.resid };
  const out = [];
  for (let past = ids.length; out.length < n; past++) {
    const t = argmax(r.logits);
    if (STOP.has(t)) break;
    out.push(t);
    if (past > ids.length) { r.neurons.dispose(); r.resid.dispose(); } // keep the prompt pass's tensors
    r = await step([t], inject, r.state, past);
  }
  for (const t of Object.values(r.state)) t.dispose();
  return { text: tok.decode(out, { skip_special_tokens: true }), ...first };
}

// ---- concept vectors ---------------------------------------------------------------------------------------

async function lastResid(text) {
  const { resid, state } = await step(encode(text), new Float32Array(meta.n_layers * meta.hidden), emptyState(), 0);
  for (const t of Object.values(state)) t.dispose();
  const [L, seq, H] = resid.dims, out = new Float32Array(L * H);
  for (let l = 0; l < L; l++) out.set(resid.data.subarray((l * seq + seq - 1) * H, (l * seq + seq) * H), l * H);
  return out;
}

async function conceptVector(concept) {
  if (conceptCache.has(concept)) return conceptCache.get(concept);
  const v = new Float32Array(meta.n_layers * meta.hidden);
  for (const t of TEMPLATES) { const r = await lastResid(t.replace("{}", concept)); for (let i = 0; i < v.length; i++) v[i] += r[i] / TEMPLATES.length; }
  for (let i = 0; i < v.length; i++) v[i] -= baseline[i];
  conceptCache.set(concept, v);
  return v;
}

// ---- graph: top neurons per token per layer, edges on demand -----------------------------------------------

function topNodes(data, L, seq, N) { // data: [L, seq, N] -> nodes[t][l] = [[idx, val] x K], strongest first
  const nodes = Array.from({ length: seq }, () => []);
  for (let l = 0; l < L; l++) for (let t = 0; t < seq; t++) {
    const base = (l * seq + t) * N, top = [];
    for (let i = 0; i < N; i++) {
      const v = data[base + i], a = Math.abs(v);
      if (top.length < TOP_K || a > Math.abs(top[top.length - 1][1])) {
        top.push([i, v]);
        top.sort((x, y) => Math.abs(y[1]) - Math.abs(x[1]));
        if (top.length > TOP_K) top.pop();
      }
    }
    nodes[t].push(top);
  }
  return nodes;
}

// one weight vector out of the 4-bit MatMulNBits layout (same math as validate.py's dequant)
function weightVec(w, kind, idx) { // kind "row": W[idx, :] (length K); "col": W[:, idx] (length N)
  const key = kind + idx;
  if (w.cache.has(key)) return w.cache.get(key);
  const blocks = Math.ceil(w.K / w.block), zpRow = Math.ceil(blocks / 2);
  const at = (n, k) => {
    const b = Math.floor(k / w.block);
    const q = w.q[n * blocks * (w.block / 2) + (k >> 1)], z = w.z[n * zpRow + (b >> 1)];
    return (((k & 1) ? q >> 4 : q & 15) - ((b & 1) ? z >> 4 : z & 15)) * half(w.s[n * blocks + b]);
  };
  const v = kind === "row" ? Float32Array.from({ length: w.K }, (_, k) => at(idx, k)) : Float32Array.from({ length: w.N }, (_, n) => at(n, idx));
  w.cache.set(key, v);
  return v;
}

// edge a->b = act_a * (layer L down_proj column a . layer L+1 up_proj row b), as in lab.graph()
function edges(nodes) {
  return nodes.slice(0, -1).map((col, l) => col.map(([a, va]) => {
    const write = weightVec(mlp[l].down_proj, "col", a);
    return nodes[l + 1].map(([b]) => {
      const read = weightVec(mlp[l + 1].up_proj, "row", b);
      let d = 0;
      for (let i = 0; i < write.length; i++) d += write[i] * read[i];
      return va * d;
    });
  }));
}

function graph(data, dims) {
  const [L, seq, N] = dims, nodes = topNodes(data, L, seq, N), cache = new Map();
  return { nodes, edgesFor: (t) => cache.get(t) ?? cache.set(t, edges(nodes[t])).get(t) };
}

// ---- same contract as lab.py run() -------------------------------------------------------------------------

export async function run({ prompt, concept = "", layer = 8, strength = 0.5, max_new_tokens = 60 }) {
  concept = concept.trim();
  if (!(layer >= 0 && layer < meta.n_layers)) throw new Error(`layer must be 0..${meta.n_layers - 1}`);
  const ids = encode(chat(prompt)), H = meta.hidden, zero = new Float32Array(meta.n_layers * H);
  const clean = await generate(ids, zero, max_new_tokens);
  const out = {
    model: `${MODEL} · running in your browser`, tokens: ids.map((i) => tok.decode([i])), n_layers: meta.n_layers, layer,
    clean: graph(clean.neurons.data, clean.neurons.dims), clean_text: clean.text,
  };
  if (concept && strength) {
    // scale: 1.0 = typical residual size at this layer for this prompt (token 0 skipped, it's an outlier)
    const seq = ids.length, r = clean.resid.data, norms = [];
    for (let t = 1; t < seq; t++) { let s = 0; for (let i = 0; i < H; i++) s += r[(layer * seq + t) * H + i] ** 2; norms.push(Math.sqrt(s)); }
    norms.sort((a, b) => a - b);
    const typical = norms[Math.floor((norms.length - 1) / 2)];
    const v = (await conceptVector(concept)).subarray(layer * H, (layer + 1) * H);
    const vn = Math.hypot(...v), inject = new Float32Array(meta.n_layers * H);
    for (let i = 0; i < H; i++) inject[layer * H + i] = (strength * typical * v[i]) / vn;
    const inj = await generate(ids, inject, max_new_tokens);
    const diff = inj.neurons.data.map((x, i) => x - clean.neurons.data[i]);
    Object.assign(out, {
      injected: graph(inj.neurons.data, inj.neurons.dims), injected_text: inj.text,
      diff: graph(diff, inj.neurons.dims),
    });
    inj.neurons.dispose(); inj.resid.dispose();
  }
  clean.neurons.dispose(); clean.resid.dispose();
  return out;
}
