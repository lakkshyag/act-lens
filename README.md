# actlens

Watch a language model's neurons fire layer by layer, inject a concept into its activations, and see exactly what the injection disturbs.

Type a prompt. For every token, actlens shows the 8 strongest MLP neurons in each of Qwen3.5-2B's 24 layers, the connections between them, and the strongest path through the network. Then you inject a concept ("thunderstorms", "the ocean", anything) into the residual stream after a chosen layer. You get the clean and injected outputs side by side, plus a diff view of which neurons the injection pushed up or down.

Inspired by Anthropic's work on concept injection and introspection (the "injected thoughts" experiments), built to experiment with your own concepts on an open model.

```
prompt:    Write one sentence about breakfast.
clean:     Breakfast is the perfect time to start your day with a warm, energizing meal.
+ "the ocean" after layer 8, strength 0.5:
injected:  The sun rises over the vast, salty, and sweet embrace of the Pacific, where the first
           wave of waves gently laps against the shore...
```

## Two engines, one UI

**Local (PyTorch).** Full-precision `Qwen/Qwen3.5-2B` on your machine (Apple GPU via MPS, CUDA, or CPU).

```bash
uv run python lab.py          # http://127.0.0.1:8000
uv run python lab.py check    # self-check: zero injection is a no-op, nothing leaks upstream, hooks clean up
```

**In the browser (WebGPU).** The whole model runs on the visitor's GPU. No server: the page is static, and the 4-bit weights (~1.4 GB, cached after the first visit) stream from the Hugging Face Hub. Open `index.html` from any static host; it falls back to the browser engine when there is no local `lab.py`.

## How it works

- **Neurons** are the MLP activations `act(gate(x)) * up(x)`: 6,144 per layer.
- **Concept vectors**: the residual stream at the concept word, averaged over four phrasings, minus the mean over 30 neutral words, at every layer.
- **Injection** adds `strength × concept vector` to the residual stream after one layer, at every position, during both the prompt pass and generation. Strength is relative: 1.0 is as large as a typical residual at that layer.
- **Edges** between neurons in consecutive layers are `activation_a × (down_proj[:, a] · up_proj[b, :])`: how much neuron *a* writes into what neuron *b* reads. This is a direct-path approximation; it ignores attention and normalization.

### Making the browser model steerable

Off-the-shelf ONNX builds of Qwen3.5 are sealed: text in, text out. Re-exporting is not an option, because Qwen3.5's Gated DeltaNet linear-attention layers break the standard export tools. Instead, [`web/patch_onnx.py`](web/patch_onnx.py) performs surgery on [onnx-community's q4f16 graph](https://huggingface.co/onnx-community/Qwen3.5-2B-ONNX-OPT) without touching the weights:

- a new `inject` input `[24, 2048]`, added to each layer's MLP output (all zeros = clean run)
- new outputs `neurons [24, seq, 6144]` and `resid [24, seq, 2048]`
- a manifest locating each layer's 4-bit MLP weights in the data file, so the browser can dequantize single rows and columns for the edges

Only the 0.7 MB graph file changes. [`web/validate.py`](web/validate.py) checks the patched graph against PyTorch:

| check | result |
|---|---|
| neuron activations, clean, cosine vs PyTorch, per layer | 0.93 to 0.997 |
| neuron activations, injected | 0.93 to 0.997 |
| layers before the injection point | bit-identical to clean |
| 4-bit weight reader vs PyTorch weights | 0.997 |

The gap from 1.0 is 4-bit quantization.

## Files

| file | what |
|---|---|
| `lab.py` | local engine and server |
| `index.html` | the UI, shared by both engines |
| `web/engine.js` | in-browser engine (onnxruntime-web + WebGPU) |
| `web/patch_onnx.py` | ONNX graph surgery |
| `web/validate.py` | checks the patched graph against PyTorch |
| `web/baseline.py` | precomputes the concept-vector baseline for the browser |
| `web/model/` | patched graph, weight manifest, baseline |

Rebuilding the browser model:

```bash
uv run --with onnx python web/patch_onnx.py decoder_model_merged_q4f16.onnx web/model/decoder.onnx
uv run --with onnxruntime python web/baseline.py web/model/decoder.onnx embed_tokens_q4f16.onnx web/model/baseline.bin
uv run --with onnx --with onnxruntime python web/validate.py web/model/decoder.onnx embed_tokens_q4f16.onnx
```

The `.onnx_data` weight files must sit next to the graph you pass in.
