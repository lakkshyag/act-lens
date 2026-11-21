"""Precompute the concept-vector baseline with the browser (ONNX) model, so the browser skips ~120 passes.

Mirrors lab.baseline(): mean residual at the final token over BASELINE words x TEMPLATES, raw text.

    uv run --with onnxruntime python web/baseline.py <patched.onnx> <embed_tokens.onnx> web/model/baseline.bin
"""
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from transformers import AutoTokenizer

BASELINE = "table cloud river window music paper horse garden engine bottle mountain pencil coffee doctor train shadow letter island candle mirror forest camera kitchen planet button ladder violin desert blanket clock".split()
TEMPLATES = ["Tell me about {}", "I keep thinking about {}", "Here is a fact about {}", "The topic is {}"]
# kept in sync by hand with lab.py and web/engine.js
dec_path, emb_path, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-2B")
dec, emb = ort.InferenceSession(dec_path), ort.InferenceSession(emb_path)


def last_resid(text):
    ids = np.array([tok(text).input_ids], np.int64)
    n = ids.shape[1]
    feeds = {
        "inputs_embeds": emb.run(None, {"input_ids": ids})[0],
        "attention_mask": np.ones((1, n), np.int64),
        "position_ids": np.broadcast_to(np.arange(n, dtype=np.int64), (3, 1, n)).copy(),
        "num_logits_to_keep": np.array(1, np.int64),
        "inject": np.zeros((24, 2048), np.float32),
    }
    for i in dec.get_inputs():
        if i.name.startswith("past"):
            feeds[i.name] = np.zeros([1 if d == "batch_size" else 0 if isinstance(d, str) else d for d in i.shape], np.float16)
    resid = dec.run(["resid"], feeds)[0]  # [layers, seq, hidden]
    return resid[:, -1]


base = np.mean([last_resid(t.format(w)) for w in BASELINE for t in TEMPLATES], axis=0).astype(np.float32)
base.tofile(out)
print("baseline", base.shape, "->", out)
