import onnx
m = onnx.load("models/voice.onnx")
print("=== VOICE MODEL INPUTS ===")
for inp in m.graph.input:
    t = inp.type.tensor_type
    print(inp.name, [d.dim_value for d in t.shape.dim], t.elem_type)

