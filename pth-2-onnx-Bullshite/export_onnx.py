import torch
import torch.nn as nn
import torch.onnx
import numpy as np
import sys

# --- CONFIGURATION (MUST MATCH YOUR MODEL) ---
# Check your model's training parameters to confirm the following settings:
MODEL_NAME = "amitaro_v2_40k_e100_simple"  # Name of your model
PTH_FILE_PATH = f"models/{MODEL_NAME}.pth"
ONNX_OUTPUT_PATH = f"models/{MODEL_NAME}.onnx"
VERSION = 'v2'  # 'v1' or 'v2'
SR = 40000      # 40000 or 48000
DIMENSION = 768 # 768 for v2, 256 for v1 (HuBERT feature size)
HOP_SIZE = 160  # Standard RVC hop size for inference

# --- CRITICAL: MODEL ARCHITECTURE DEFINITION ---
# You must import or paste the exact Generator class definition (e.g., Generator, or RVC_Model)
# that matches your saved .pth file. Since this architecture is hundreds of lines long,
# we rely on the user having access to the source code definitions (usually in your RVC fork).
# For demonstration, we assume you have a Generator class:
try:
    # NOTE: Replace this with the correct import path for your RVC Generator class
    # Example: from infer_pack.models import SynthesizerTrnMs256NSFsid as Generator
    # If using RVC WebUI, check the /models folder of the repository.
    class Generator(nn.Module):
        # Placeholder class: Replace this entire class definition with the REAL Generator/SynthesizerTrn class
        # from your model's source code.
        def __init__(self, *args, **kwargs):
            super().__init__()
            print("ERROR: Generator architecture must be imported/defined here.")
            sys.exit(1)
except ImportError:
    # If using a local Generator definition:
    pass
# --- END ARCHITECTURE DEFINITION ---

def convert_pth_to_onnx():
    # 1. Initialize the Model Architecture
    print(f"Initializing Generator architecture for RVC {VERSION} at {SR}Hz.")

    # *** REPLACE THE LINE BELOW WITH YOUR MODEL INITIALIZATION ***
    # This must match the version (e.g., if it's v1, the parameters are different)
    # The default RVC v2 initialization often involves many parameters that must be correct.
    # Placeholder: Assuming a simplified v2 load structure (replace as needed)
    # model = Generator(parameters_matching_v2_architecture).to('cpu')
    model = Generator(
        # Pass required architecture parameters here
    ).to('cpu')

    # 2. Load Weights
    print(f"Loading state dictionary from {PTH_FILE_PATH}...")
    # NOTE: RVC models often save only the 'generator' state, not the whole dict.
    ckpt = torch.load(PTH_FILE_PATH, map_location='cpu')
    model.load_state_dict(ckpt['net_g'] if 'net_g' in ckpt else ckpt, strict=False)
    model.eval()
    print("Weights loaded successfully.")

    # 3. Define Dynamic Inputs
    # RVC inputs: feats (embeddings), p_len (length), pitch (coarse), pitchf (fine), sid (speaker ID)
    # The '?'' marks dynamic axes (variable length/time)
    dynamic_axes = {
        'feats': {1: 'time'},
        'pitch': {1: 'time'},
        'pitchf': {1: 'time'},
        'audio': {1: 'audio_time'}
    }

    # 4. Create Dummy Inputs (Critical: MUST have correct shapes and types)
    # The model expects a batch size of 1.
    T = 200 # Dummy time steps (200 frames ≈ 2 seconds)

    # feats: [1, T, DIMENSION] (Content features, 768 for v2)
    dummy_feats = torch.randn(1, T, DIMENSION).to(torch.float32)

    # p_len: [1] (Length of feature vector)
    dummy_p_len = torch.tensor([T]).to(torch.long)

    # pitch: [1, T] (Coarse pitch index 1-255)
    dummy_pitch = torch.randint(1, 256, (1, T)).to(torch.long)

    # pitchf: [1, T] (Fine pitch frequency Hz)
    dummy_pitchf = torch.randn(1, T).to(torch.float32)

    # sid: [1] (Speaker ID, usually 0)
    dummy_sid = torch.tensor([0]).to(torch.long)

    # Arguments for torch.onnx.export
    example_inputs = (dummy_feats, dummy_p_len, dummy_pitch, dummy_pitchf, dummy_sid)
    input_names = ['feats', 'p_len', 'pitch', 'pitchf', 'sid']
    output_names = ['audio']

    # 5. Export to ONNX
    print(f"Exporting model to {ONNX_OUTPUT_PATH}...")
    torch.onnx.export(
        model,
        example_inputs,
        ONNX_OUTPUT_PATH,
        export_params=True,
        opset_version=17, # Use a modern Opset version
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        verbose=False
    )
    print("ONNX export successful!")

if __name__ == '__main__':
    # You must manually define or import the Generator architecture before running this script.
    convert_pth_to_onnx()