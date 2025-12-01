import torch.onnx
from tmp_s.SynthesizerTrn import SynthesizerTrn

# -------------------------------
# CONFIGURATION
# -------------------------------
CHECKPOINT_PATH = "/pth-2-onnx-Bullshite/calli5_220e_29920s.pth"
EXPORT_PATH = "/models/z-not_in_use/CalliopebyBoston/normed/calli5.onnx"

# -------------------------------
# LOAD CHECKPOINT
# -------------------------------
print(f"Loading checkpoint from: {CHECKPOINT_PATH}")
ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")

if "config" in ckpt:
    cfg_list = ckpt["config"]
elif "model_config" in ckpt:
    cfg_list = ckpt["model_config"]
else:
    raise ValueError("Checkpoint does not contain config information.")

print(f"Config loaded from checkpoint: {cfg_list}")

# -------------------------------
# FIX UPSAMPLE RATES IF NEEDED
# -------------------------------
upsample_rates = cfg_list[13]
if isinstance(upsample_rates, int):
    upsample_rates = [upsample_rates]

# -------------------------------
# INSTANTIATE MODEL
# -------------------------------
print("Instantiating SynthesizerTrn model...")
net = SynthesizerTrn(
    spec_channels=cfg_list[0],
    hidden_channels=cfg_list[2],
    n_layers=cfg_list[6],
    n_heads=cfg_list[5],
    upsample_rates=upsample_rates,
    resblock_channels=cfg_list[15],
    n_resblocks=cfg_list[16]
)

# -------------------------------
# LOAD WEIGHTS
# -------------------------------
print("Loading weights into the model...")
try:
    net.load_state_dict(ckpt["weight"])
except RuntimeError as e:
    print("Warning: state_dict mismatch. Attempting strict=False load.")
    net.load_state_dict(ckpt["weight"], strict=False)

net.eval()

# -------------------------------
# CREATE DUMMY INPUTS
# -------------------------------
dummy_feats = torch.randn(1, 513, 250)        # [batch, spec_channels, time]
dummy_p_len = torch.tensor([250], dtype=torch.float32)
dummy_pitch = torch.randn(1, 20)             # pitch embedding
dummy_pitchf = torch.randn(1, 250)           # pitch frame
dummy_sid = torch.tensor([0], dtype=torch.float32)  # sid as float if ONNX expects float

# -------------------------------
# TEST FORWARD PASS
# -------------------------------
print("Testing model forward pass...")
try:
    out = net(dummy_feats, dummy_p_len, dummy_pitch, dummy_pitchf, dummy_sid)
    print("Forward pass successful. Output shape:", out.shape)
except Exception as e:
    print("Forward pass failed:", e)
    exit(1)

# -------------------------------
# EXPORT TO ONNX
# -------------------------------
print(f"Exporting ONNX model to {EXPORT_PATH}...")
torch.onnx.export(
    net,
    (dummy_feats, dummy_p_len, dummy_pitch, dummy_pitchf, dummy_sid),
    EXPORT_PATH,
    opset_version=17,
    input_names=["feats", "p_len", "pitch", "pitchf", "sid"],
    output_names=["audio"],
    dynamic_axes={
        "feats": {2: "time"},
        "pitchf": {1: "time"},
        "audio": {2: "time"},
        "p_len": {0: "batch"}
    }
)
print("ONNX export completed successfully!")
