"""Patch onnx-community's Qwen3.5 decoder graph so the browser can see and steer neurons.

Adds to the graph (weights untouched, only the small .onnx graph file changes):
  input  inject   float32 [n_layers, hidden]   added to each layer's MLP output, i.e. into the residual
                                                 stream after that layer. All zeros = clean run.
  output neurons  float32 [n_layers, seq, n_neurons]   MLP neuron activations (act(gate) * up)
  output resid    float32 [n_layers, seq, hidden]      residual stream after each layer

Also writes <out>.weights.json: where each layer's 4-bit MLP weights sit inside the .onnx_data file, so the
browser can read them for the edges (MatMulNBits layout: quant [N, blocks, 16] uint8 low-nibble-first,
scales [N, blocks] fp16, zero points [N, blocks/2] uint8 low-nibble-first).

    uv run --with onnx python web/patch_onnx.py <decoder.onnx> <out.onnx>
"""
import json
import sys
from pathlib import Path

import onnx
from onnx import TensorProto, helper, numpy_helper
import numpy as np

src, dst = sys.argv[1], sys.argv[2]
m = onnx.load(src, load_external_data=False)  # weights stay in the original .onnx_data files
g = m.graph
cfg_layers = sorted({int(n.name.split("/layers.")[1].split("/")[0]) for n in g.node if "/mlp/down_proj/" in n.name})
n_layers = len(cfg_layers)
hidden = next(i for i in g.input if i.name == "inputs_embeds").type.tensor_type.shape.dim[2].dim_value
consumers = {}
for n in g.node:
    for k, name in enumerate(n.input):
        consumers.setdefault(name, []).append((n, k))

new_nodes = [helper.make_node("Cast", ["inject"], ["inject_f16"], to=TensorProto.FLOAT16, name="/patch/inject_cast")]
neuron_parts, resid_parts = [], []
for L in range(n_layers):
    p = f"/patch/layers.{L}"
    # injection: down_proj output + inject[L], rewired into whatever consumed down_proj output
    down = f"/model/layers.{L}/mlp/down_proj/MatMul/output_0"
    users = consumers[down]
    g.initializer.append(numpy_helper.from_array(np.array(L, dtype=np.int64), f"{p}/idx"))
    new_nodes += [
        helper.make_node("Gather", ["inject_f16", f"{p}/idx"], [f"{p}/vec"], axis=0, name=f"{p}/gather"),
        helper.make_node("Add", [down, f"{p}/vec"], [f"{p}/down_injected"], name=f"{p}/inject"),
    ]
    for node, k in users:
        node.input[k] = f"{p}/down_injected"
    # neurons out
    neuron_parts.append(f"/model/layers.{L}/mlp/Mul/output_0")
    # residual after layer L = the skip-sum (output 3) of the next norm, which already includes the injection
    if L + 1 < n_layers:
        resid_parts.append(f"/model/layers.{L + 1}/input_layernorm/output_3")
    else:  # the final norm does not emit its skip-sum, so add its two inputs ourselves
        fn = f"/model/layers.{n_layers}/final_norm_layernorm/SkipLayerNorm"
        new_nodes.append(helper.make_node(
            "Add", [f"{fn}/root_input/Cast/output_0", f"{fn}/skip_input/Cast/output_0"], [f"{p}/resid"], name=f"{p}/resid"))
        resid_parts.append(f"{p}/resid")

for name, parts in (("neurons", neuron_parts), ("resid", resid_parts)):
    casts = [f"/patch/{name}.{L}" for L in range(n_layers)]
    new_nodes += [helper.make_node("Cast", [t], [c], to=TensorProto.FLOAT, name=c) for t, c in zip(parts, casts)]
    new_nodes.append(helper.make_node("Concat", casts, [name], axis=0, name=f"/patch/{name}_concat"))  # batch=1

# ONNX wants nodes in topological order; re-sort with the new ones mixed in (Kahn's algorithm)
nodes = list(g.node) + new_nodes
produced_by = {o: i for i, n in enumerate(nodes) for o in n.output if o}
deps = [{produced_by[x] for x in n.input if x in produced_by} for n in nodes]
users = [[] for _ in nodes]
for i, d in enumerate(deps):
    for j in d:
        users[j].append(i)
ready, order, left = [i for i, d in enumerate(deps) if not d], [], [len(d) for d in deps]
while ready:
    i = ready.pop()
    order.append(nodes[i])
    for u in users[i]:
        left[u] -= 1
        if not left[u]:
            ready.append(u)
assert len(order) == len(nodes), "graph has a cycle"
del g.node[:]
g.node.extend(order)
g.input.append(helper.make_tensor_value_info("inject", TensorProto.FLOAT, [n_layers, hidden]))
g.output.append(helper.make_tensor_value_info("neurons", TensorProto.FLOAT, [n_layers, "sequence_length", None]))
g.output.append(helper.make_tensor_value_info("resid", TensorProto.FLOAT, [n_layers, "sequence_length", hidden]))
onnx.save(m, dst)

init = {i.name: {e.key: e.value for e in i.external_data} for i in g.initializer}
weights = {"file": None, "layers": [], "n_layers": n_layers, "hidden": hidden,
           "states": [o.name for o in g.output if o.name.startswith("present")]}
for L in range(n_layers):
    entry = {}
    for proj in ("up_proj", "down_proj"):
        node = next(n for n in g.node if n.name == f"/model/layers.{L}/mlp/{proj}/MatMul_Quant")
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        q, sc, zp = (init[x] for x in node.input[1:4])
        weights["file"] = q["location"]
        entry[proj] = {"N": attrs["N"], "K": attrs["K"], "block": attrs["block_size"],
                       **{k: [int(t["offset"]), int(t["length"])] for k, t in (("quant", q), ("scales", sc), ("zp", zp))}}
    weights["layers"].append(entry)
Path(dst).with_suffix(".weights.json").write_text(json.dumps(weights))
print(f"patched {n_layers} layers, hidden {hidden} -> {dst}")
