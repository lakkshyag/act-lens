"""Neuron Brain: watch MLP neurons fire layer by layer, inject a concept, see what it disturbs.

    uv run python lab.py          # UI on http://127.0.0.1:8000
    uv run python lab.py check    # self-check the hooks
    MODEL=Qwen/Qwen3.5-4B uv run python lab.py   # any Llama/Qwen-style chat model
"""
import json
import os
import sys
from contextlib import contextmanager
from functools import cache
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = os.environ.get("MODEL", "Qwen/Qwen3.5-2B")
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
TOP_K = 8  # neurons shown per layer
# Neutral words. Their mean activation is subtracted so a concept vector keeps only what is specific to the concept.
BASELINE = (
    "table cloud river window music paper horse garden engine bottle mountain pencil coffee doctor train "
    "shadow letter island candle mirror forest camera kitchen planet button ladder violin desert blanket clock"
).split()

tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).to(DEVICE).eval()
layers = model.model.layers


def encode(text):
    chat = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    return tok(chat, return_tensors="pt").to(DEVICE)


@torch.no_grad()
def forward(ids):
    """One pass. Returns MLP neuron activations per layer [seq, n_neurons] and the residual after each layer [seq, d]."""
    acts = []
    hooks = [l.mlp.down_proj.register_forward_pre_hook(lambda m, a: acts.append(a[0][0].float())) for l in layers]
    try:
        hidden = model(**ids, output_hidden_states=True).hidden_states[1:]
    finally:
        for h in hooks:
            h.remove()
    return acts, [h[0].float() for h in hidden]


@torch.no_grad()
def generate(ids, n):
    out = model.generate(**ids, max_new_tokens=n, do_sample=False, temperature=None, top_p=None, top_k=None)
    return tok.decode(out[0, ids.input_ids.shape[1]:], skip_special_tokens=True)


def last_resid(text):
    """Residual at the final token of raw text (no chat template, so the final token is the concept word itself)."""
    return torch.stack([r[-1] for r in forward(tok(text, return_tensors="pt").to(DEVICE))[1]])  # [n_layers, d]


# Several phrasings, averaged, so the vector is the concept and not one sentence's quirks.
TEMPLATES = ["Tell me about {}", "I keep thinking about {}", "Here is a fact about {}", "The topic is {}"]


def mean_resid(word):
    return torch.stack([last_resid(t.format(word)) for t in TEMPLATES]).mean(0)


@cache
def baseline():
    return torch.stack([mean_resid(w) for w in BASELINE]).mean(0)


@cache
def concept_vector(concept):
    """Direction for `concept` at every layer, [n_layers, d]. Must be computed outside inject()."""
    return mean_resid(concept) - baseline()


@contextmanager
def inject(vec, layer, strength):
    """Add strength * vec to the residual stream coming out of `layer`, at every position."""

    def hook(mod, args, out):
        if isinstance(out, tuple):
            return (out[0] + strength * vec.to(out[0].dtype),) + out[1:]
        return out + strength * vec.to(out.dtype)

    h = layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


@torch.no_grad()
def graph(acts):
    """Top-K neurons per token per layer, and edges between consecutive layers.

    Edge a->b = act_a * (layer L down_proj column a . layer L+1 up_proj row b): how much neuron a writes
    into what neuron b reads. ponytail: direct path only, ignores attention, gating and norms; switch to
    gradient x activation attribution if the arrows need to be exact.
    """
    top = [a.abs().topk(TOP_K, dim=-1).indices for a in acts]  # per layer [seq, K], strongest first
    vals = [a.gather(-1, i) for a, i in zip(acts, top)]
    edges = []
    for L in range(len(acts) - 1):
        write = layers[L].mlp.down_proj.weight.T[top[L]].float()  # [seq, K, d]
        read = layers[L + 1].mlp.up_proj.weight[top[L + 1]].float()  # [seq, K, d]
        edges.append((torch.einsum("skd,sjd->skj", write, read) * vals[L][..., None]).tolist())
    top, vals = [t.tolist() for t in top], [v.tolist() for v in vals]
    seq = len(top[0])
    return {
        "nodes": [[list(zip(top[L][t], vals[L][t])) for L in range(len(acts))] for t in range(seq)],
        "edges": [[e[t] for e in edges] for t in range(seq)],
    }


def run(prompt, concept="", layer=8, strength=0.5, max_new_tokens=60):
    layer, strength, concept = int(layer), float(strength), concept.strip()
    if not 0 <= layer < len(layers):
        raise ValueError(f"layer must be 0..{len(layers) - 1}")
    n = max(1, min(int(max_new_tokens), 200))
    ids = encode(prompt)
    clean, resid = forward(ids)
    out = {
        "model": MODEL,
        "tokens": [tok.decode(t) for t in ids.input_ids[0]],
        "n_layers": len(layers),
        "layer": layer,
        "clean": graph(clean),
        "clean_text": generate(ids, n),
    }
    if concept and strength:
        # strength is relative: 1.0 = as big as a typical residual at this layer (token 0 skipped, it's an outlier)
        vec = concept_vector(concept)[layer]
        vec = vec / vec.norm() * resid[layer][1:].norm(dim=-1).median()
        with inject(vec, layer, strength):
            injected, _ = forward(ids)
            out["injected_text"] = generate(ids, n)
        out["injected"] = graph(injected)
        out["diff"] = graph([i - c for i, c in zip(injected, clean)])
    return out


class Handler(SimpleHTTPRequestHandler):
    """Serves the project folder (index.html, web/) plus GET /info and POST /run."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(Path(__file__).parent), **kwargs)

    def reply(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/info":
            return self.reply(200, json.dumps({"model": MODEL}).encode(), "application/json")
        super().do_GET()

    def do_POST(self):
        try:
            req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            self.reply(200, json.dumps(run(**req)).encode(), "application/json")
        except Exception as e:
            self.reply(400, json.dumps({"error": f"{type(e).__name__}: {e}"}).encode(), "application/json")


def check():
    ids = encode("The cat sat on the mat.")
    clean, _ = forward(ids)
    L = len(layers) // 2
    vec = concept_vector("volcanoes")[L]
    with inject(vec, L, 0.0):
        same, _ = forward(ids)
    assert all(torch.allclose(a, b) for a, b in zip(clean, same)), "zero-strength injection changed activations"
    n_hooks = len(layers[L]._forward_hooks)  # transformers registers its own hooks, so compare counts
    with inject(vec, L, 4.0):
        hit, _ = forward(ids)
    assert len(layers[L]._forward_hooks) == n_hooks, "injection hook not removed"
    assert all(torch.allclose(a, b) for a, b in zip(clean[: L + 1], hit[: L + 1])), "injection leaked upstream"
    assert not torch.allclose(clean[L + 1], hit[L + 1]), "injection had no effect downstream"
    res = json.loads(json.dumps(run("Hi", "volcanoes", L, 4.0, 5)))
    assert len(res["diff"]["nodes"]) == len(res["tokens"]) and len(res["diff"]["edges"][0]) == len(layers) - 1
    print("ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["check"]:
        check()
    else:
        print(f"{MODEL} on {DEVICE}: http://127.0.0.1:8000")
        HTTPServer(("127.0.0.1", 8000), Handler).serve_forever()  # single-threaded on purpose: one model, one request
