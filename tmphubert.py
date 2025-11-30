import onnx
h = onnx.load("models/hubert.onnx")
print("=== HUBERT MODEL OUTPUTS ===")
for out in h.graph.output:
    t = out.type.tensor_type
    print(out.name, [d.dim_value for d in t.shape.dim], t.elem_type)

