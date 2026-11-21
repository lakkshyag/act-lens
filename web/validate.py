"""Check the patched ONNX graph against the PyTorch lab: same neurons, same injection effect.

    uv run --with onnx --with onnxruntime python web/validate.py <patched.onnx> <embed_tokens.onnx>
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
import lab  # noqa: E402  (loads the PyTorch model)

decoder_path, embed_path = sys.argv[1], sys.argv[2]
dec = ort.InferenceSession(decoder_path)
emb = ort.InferenceSession(embed_path)
F16 = np.float16


def ort_step(ids, inject, state=None, past_len=0):
    """One decoder call. Returns logits, neurons, resid, new state."""
    n = ids.shape[1]
    feeds = {
        "inputs_embeds": emb.run(None, {"input_ids": ids})[0],
        "attention_mask": np.ones((1, past_len + n), np.int64),
        "position_ids": np.broadcast_to(np.arange(past_len, past_len + n, dtype=np.int64), (3, 1, n)).copy(),
        "num_logits_to_keep": np.array(1, np.int64),
        "inject": inject,
    }
    for i in dec.get_inputs():
        if i.name.startswith("past"):
            shape = [1 if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape]
            feeds[i.name] = state[i.name] if state else np.zeros(shape, F16)
    outs = dict(zip([o.name for o in dec.get_outputs()], dec.run(None, feeds)))
    new_state = {
        k.replace("present_", "past_").replace("present.", "past_key_values."): v
        for k, v in outs.items() if k.startswith("present")
    }
    return outs["logits"][0, -1], outs["neurons"], outs["resid"], new_state


def ort_generate(ids, inject, n=30):
    logits, neurons, _, state = ort_step(ids, inject)
    out, past = [], ids.shape[1]
    for _ in range(n):
        t = int(np.argmax(logits))
        if t == lab.tok.eos_token_id or t == lab.tok.convert_tokens_to_ids("<|im_end|>"):
            break
        out.append(t)
        logits, _, _, state = ort_step(np.array([[t]], np.int64), inject, state, past)
        past += 1
    return lab.tok.decode(out, skip_special_tokens=True), neurons


def cos(a, b):
    a, b = a.ravel(), b.ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


prompt, concept, L, strength = "Write one sentence about breakfast.", "thunderstorms", 8, 0.5
enc = lab.encode(prompt)
ids = enc.input_ids.cpu().numpy()
n_layers, hidden = len(lab.layers), lab.model.config.hidden_size

# clean: neurons should match PyTorch closely (ONNX weights are 4-bit, so not exactly)
pt_clean, pt_resid = lab.forward(enc)
ort_text, ort_neurons = ort_generate(ids, np.zeros((n_layers, hidden), np.float32))
sims = [cos(ort_neurons[l], pt_clean[l].cpu().numpy()) for l in range(n_layers)]
print("clean neuron cosine per layer:", " ".join(f"{s:.3f}" for s in sims))
print("ORT clean text:", ort_text)

# injected: same vector, same scale as lab.run
vec = lab.concept_vector(concept)[L]
vec = (vec / vec.norm() * pt_resid[L][1:].norm(dim=-1).median()).cpu().numpy()
inject = np.zeros((n_layers, hidden), np.float32)
inject[L] = strength * vec
with lab.inject(torch.tensor(strength * vec, device=lab.DEVICE), L, 1.0):
    pt_inj, _ = lab.forward(enc)
ort_text, ort_neurons = ort_generate(ids, inject)
sims = [cos(ort_neurons[l], pt_inj[l].cpu().numpy()) for l in range(n_layers)]
print("injected neuron cosine per layer:", " ".join(f"{s:.3f}" for s in sims))
print("ORT injected text:", ort_text)
untouched = np.abs(ort_neurons[: L + 1] - ort_step(ids, np.zeros_like(inject))[1][: L + 1]).max()
assert untouched == 0, f"injection leaked into layers <= {L}: {untouched}"
assert min(sims) > 0.9, "patched graph disagrees with PyTorch"

# 4-bit weight reader (the browser does the same in JS): compare against the PyTorch weights
import json  # noqa: E402

meta = json.loads(Path(decoder_path).with_suffix(".weights.json").read_text())
blob = np.memmap(Path(decoder_path).parent / meta["file"], dtype=np.uint8, mode="r")


def dequant(w):
    q = np.asarray(blob[w["quant"][0]: sum(w["quant"])]).reshape(w["N"], -1)
    sc = np.asarray(blob[w["scales"][0]: sum(w["scales"])]).view(np.float16).reshape(w["N"], -1).astype(np.float32)
    zp = np.asarray(blob[w["zp"][0]: sum(w["zp"])]).reshape(w["N"], -1)
    vals = np.stack([q & 15, q >> 4], -1).reshape(w["N"], -1)[:, : w["K"]].astype(np.float32)
    zps = np.stack([zp & 15, zp >> 4], -1).reshape(w["N"], -1).astype(np.float32)
    blocks = np.arange(w["K"]) // w["block"]
    return (vals - zps[:, blocks]) * sc[:, blocks]  # [N, K], same as torch Linear.weight


for proj in ("up_proj", "down_proj"):
    mine = dequant(meta["layers"][5][proj])
    ref = getattr(lab.layers[5].mlp, proj).weight.detach().float().cpu().numpy()
    print(f"layer 5 {proj} dequant cosine: {cos(mine, ref):.4f}")
    assert cos(mine, ref) > 0.98, f"{proj} weight reader is wrong"
print("ok")
