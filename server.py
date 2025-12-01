# server.py

import socket
import struct
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import sys
import traceback
import time
import faiss
import torchcrepe

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 51235
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"
INDEX_PATH = "models/voice.index"
INDEX_RATE = 0.75

# CHANGED: Switch from 40000 to 48000
MODEL_SAMPLE_RATE = 48000  # <--- THIS IS THE FIX
TARGET_SAMPLE_RATE = 48000
INPUT_SR = 48000

print(f"Python Version: {sys.version}")
print("Loading Models...")

# --- LOAD FAISS INDEX ---
index = None
try:
    print(f"Attempting to load index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()
    print(f"Index Loaded Successfully. Total Vectors: {index.ntotal}")
except Exception as e:
    print(f"[WARN] Could not load Index file: {e}")

# --- LOAD AI MODELS ---
try:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3

    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)

    # --- MODEL INTROSPECTION (DEBUG) ---
    print("\n[DEBUG] Voice Model Expected Inputs:")
    model_inputs = voice_sess.get_inputs()
    for i in model_inputs:
        print(f"  - Name: {i.name}, Shape: {i.shape}, Type: {i.type}")
    print("----------------------------------\n")

    print("AI Loaded. Ready.")
except Exception as e:
    print("CRITICAL MODEL LOAD ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

# --- INIT RESAMPLERS ---
try:
    # 1. Voice Model Input (Not used for features, but used for volume envelope if needed)
    input_resampler = torchaudio.transforms.Resample(INPUT_SR, MODEL_SAMPLE_RATE)

    # 2. Output (40k -> 48k)
    output_resampler = torchaudio.transforms.Resample(MODEL_SAMPLE_RATE, TARGET_SAMPLE_RATE)

    # 3. Analysis (48k -> 16k) - CRITICAL for HuBERT and Crepe
    analysis_resampler = torchaudio.transforms.Resample(INPUT_SR, 16000)
    print("Torchaudio Resamplers Initialized.")
except Exception as e:
    print("Error initializing torchaudio:", e)
    sys.exit(1)

# --- HELPER: CONVERT HZ TO COARSE INTEGER PITCH ---
def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)

    # Normalize to 1-255 range
    f0_coarse = (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_coarse[f0_mel <= f0_mel_min] = 1
    f0_coarse[f0_mel >= f0_mel_max] = 255

    return np.clip(np.round(f0_coarse), 1, 255).astype(np.int64)

# --- AUDIO PROCESSING ---
def process_audio(data, pitch_shift):
    # DEBUG PRINT
    print(f"  >>> [DEBUG] Python Received Pitch Shift: {pitch_shift}")

    try:
        t_start = time.time()
        audio_raw = np.frombuffer(data, dtype=np.float32).copy()

        # Amplitude Safety Check
        if np.max(np.abs(audio_raw)) < 0.01:
            print("[WARNING] Audio is near SILENCE.")

        if len(audio_raw) < 1600: return b''

        # PREPARE TENSORS
        audio_tensor = torch.from_numpy(audio_raw)

        # 1. Analysis Audio (16k) for Pitch & Content
        audio_16k_tensor = analysis_resampler(audio_tensor)
        audio_16k = audio_16k_tensor.numpy()

        # 2. HUBERT (CONTENT)
        # Using 16k audio (Correct)
        inp = audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        # Transpose logic: (1, 768, T) -> (1, T, 768)
        # This matches Standard RVC V2
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        # 3. INDEX MIXING
        if index is not None and INDEX_RATE > 0:
            try:
                query = embed.squeeze(0).astype(np.float32)
                D, I = index.search(query, 1)
                retrieved = index.reconstruct_batch(I.flatten().astype(np.int64))
                weight = INDEX_RATE
                embed_mixed = (query * (1 - weight)) + (retrieved * weight)
                embed = embed_mixed[np.newaxis, :, :]
            except Exception as e:
                pass

        T = embed.shape[1]

        # 4. PITCH DETECTION (CREPE)
        f0 = None
        try:
            # Using 'tiny' for speed, 'full' for accuracy.
            # If tiny is too bad, switch back to full.
            f0 = torchcrepe.predict(
                audio_16k_tensor.unsqueeze(0),
                16000,
                hop_length=160,
                fmin=50,
                fmax=1000,
                model='tiny',
                batch_size=256,
                device='cpu',
                decoder=torchcrepe.decode.weighted_argmax
            )
            f0 = f0.squeeze(0).numpy()

            # Interpolate to match Hubert Time Steps
            if len(f0) != T:
                f0 = np.interp(np.linspace(0, len(f0), T), np.arange(len(f0)), f0).astype(np.float32)

            # DEBUG: Print Pitch
            avg_p = np.mean(f0[f0 > 0]) if np.any(f0 > 0) else 0
            print(f"[DEBUG] Original Pitch: {avg_p:.1f} Hz")

        except Exception as e:
            print(f"[ERROR] Crepe Crashed: {e}")
            f0 = np.zeros(T, dtype=np.float32)

        # 5. PITCH SHIFTING
        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        # 6. COARSE PITCH
        pitch_coarse = f0_to_coarse(f0)

        inputs = {
            "feats": embed,
            "p_len": np.array([T], dtype=np.int64),
            "pitch": pitch_coarse[None, :], # Correct Coarse Pitch
            "pitchf": f0[None, :],          # Correct Fine Pitch
            "sid": np.array([0], dtype=np.int64)
        }

        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            out_tensor = torch.from_numpy(out)
            out_tensor = output_resampler(out_tensor)
            out = out_tensor.numpy()

        process_time = time.time() - t_start
        print(f"  -> Processed in {process_time:.2f}s")
        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] process_audio:", e)
        traceback.print_exc()
        return b''

def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None
        data += chunk
    return data

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                while True:
                    header = recvall(conn, 8)
                    if not header: break
                    size, pitch = struct.unpack('ii', header)
                    data = recvall(conn, size * 4)
                    if not data: break
                    if len(data) == size * 4:
                        try:
                            result = process_audio(data, pitch)
                            if len(result) > 0:
                                conn.sendall(struct.pack('i', len(result)//4))
                                conn.sendall(result)
                            else:
                                conn.sendall(struct.pack('i', 0))
                        except Exception as e:
                            print(f"Inference Error: {e}")
                            conn.sendall(struct.pack('i', 0))
    except Exception as e:
        print("Socket Error:", e)
        traceback.print_exc()