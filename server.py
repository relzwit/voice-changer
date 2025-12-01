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
import configparser # <-- NEW: For reading config.ini

# --- CONFIG LOADING ---
config = configparser.ConfigParser()
config.read('config.ini')

# Safely cast all config values
HOST = config['NETWORK'].get('HOST', '127.0.0.1')
PORT = config['NETWORK'].getint('PORT', 51235)
MODEL_PATH = config['PATHS'].get('MODEL_PATH', 'models/voice.onnx')
HUBERT_PATH = config['PATHS'].get('HUBERT_PATH', 'models/hubert.onnx')
INDEX_PATH = config['PATHS'].get('INDEX_PATH', 'models/voice.index')
INDEX_RATE = config['MODEL_TUNING'].getfloat('INDEX_RATE', 0.75)
MODEL_SAMPLE_RATE = config['MODEL_TUNING'].getint('MODEL_SAMPLE_RATE', 40000)

# Fixed settings for the client/analysis loop
TARGET_SAMPLE_RATE = 48000
INPUT_SR = 48000
# ----------------------

print("="*60)
print("VOICE CHANGER SERVER - DEBUG MODE")
print("="*60)
print(f"Python Version: {sys.version}")
print(f"MODEL_SAMPLE_RATE: {MODEL_SAMPLE_RATE}")
print(f"TARGET_SAMPLE_RATE: {TARGET_SAMPLE_RATE}")
print(f"INDEX_RATE: {INDEX_RATE}")
print("="*60)
print("Loading Models...")

# --- LOAD FAISS INDEX ---
index = None
try:
    print(f"[DEBUG] Attempting to load index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()
    print(f"[DEBUG] Index Loaded Successfully. Total Vectors: {index.ntotal}")
except Exception as e:
    print(f"[WARN] Could not load Index file: {e}")

# --- LOAD AI MODELS ---
try:
    print("[DEBUG] Creating ONNX Runtime Session Options...")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.log_severity_level = 3
    print(f"[DEBUG] Threads: {opts.intra_op_num_threads}")

    print(f"[DEBUG] Loading HuBERT from: {HUBERT_PATH}")
    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    print("[DEBUG] HuBERT loaded successfully")

    print(f"[DEBUG] Loading Voice Model from: {MODEL_PATH}")
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
    print("[DEBUG] Voice Model loaded successfully")

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
    print("[DEBUG] Initializing Resamplers...")
    # Analysis resampler: 48k -> 16k (for HuBERT and Crepe)
    analysis_resampler = torchaudio.transforms.Resample(INPUT_SR, 16000)
    print(f"[DEBUG] Analysis Resampler: {INPUT_SR} Hz -> 16000 Hz")

    # Output resampler: MODEL_SR -> 48k (for C++ playback)
    output_resampler = torchaudio.transforms.Resample(MODEL_SAMPLE_RATE, TARGET_SAMPLE_RATE)
    print(f"[DEBUG] Output Resampler: {MODEL_SAMPLE_RATE} Hz -> {TARGET_SAMPLE_RATE} Hz")

    print("[DEBUG] Torchaudio Resamplers Initialized Successfully.")
except Exception as e:
    print("Error initializing torchaudio:", e)
    sys.exit(1)

# --- HELPER: CONVERT HZ TO COARSE INTEGER PITCH ---
def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)

    f0_coarse = (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_coarse[f0_mel <= f0_mel_min] = 1
    f0_coarse[f0_mel >= f0_mel_max] = 255

    return np.clip(np.round(f0_coarse), 1, 255).astype(np.int64)

# --- AUDIO PROCESSING ---
def process_audio(data, pitch_shift):
    print("\n" + "="*60)
    print("STARTING AUDIO PROCESSING")
    print("="*60)
    print(f"[DEBUG] Pitch Shift: {pitch_shift} semitones")
    print(f"[DEBUG] Received {len(data)//4} samples ({len(data)} bytes)")

    try:
        t_start = time.time()
        audio_raw = np.frombuffer(data, dtype=np.float32).copy()
        print(f"[DEBUG] Input audio: {audio_raw.shape}, range [{np.min(audio_raw):.4f}, {np.max(audio_raw):.4f}]")

        if np.max(np.abs(audio_raw)) < 0.01:
            print("[WARNING] Audio is near SILENCE.")

        if len(audio_raw) < 1600:
            print("[ERROR] Audio too short")
            return b''

        audio_tensor = torch.from_numpy(audio_raw)

        # Downsample to 16k for analysis (HuBERT and Crepe)
        print(f"[DEBUG] Resampling to 16kHz for analysis...")
        audio_16k_tensor = analysis_resampler(audio_tensor)
        audio_16k = audio_16k_tensor.numpy()
        print(f"[DEBUG] 16kHz audio: {audio_16k.shape} samples")

        # HUBERT (CONTENT)
        print("[DEBUG] Running HuBERT...")
        inp = audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]
        print(f"[DEBUG] HuBERT output: {embed.shape}")

        # Transpose: (1, 768, T) -> (1, T, 768)
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))
            print(f"[DEBUG] Transposed to: {embed.shape}")

        # INDEX MIXING
        if index is not None and INDEX_RATE > 0:
            print(f"[DEBUG] Applying Index (rate={INDEX_RATE})...")
            try:
                query = embed.squeeze(0).astype(np.float32)
                D, I = index.search(query, 1)
                retrieved = index.reconstruct_batch(I.flatten().astype(np.int64))
                embed_mixed = (query * (1 - INDEX_RATE)) + (retrieved * INDEX_RATE)
                embed = embed_mixed[np.newaxis, :, :]
                print(f"[DEBUG] Index mixed: {embed.shape}")
            except Exception as e:
                print(f"[WARN] Index mixing failed: {e}")

        T = embed.shape[1]
        print(f"[DEBUG] Time steps: {T}")

        # PITCH DETECTION (CREPE)
        try:
            print("[DEBUG] Running CREPE...")
            f0 = torchcrepe.predict(
                audio_16k_tensor.unsqueeze(0),
                16000,
                hop_length=160,
                fmin=50,
                fmax=1000,
                model='tiny',
                batch_size=256,
                device='cpu',
                decoder=torchcrepe.decode.viterbi
            )
            f0 = f0.squeeze(0).numpy()
            print(f"[DEBUG] CREPE output: {f0.shape}")

            # Interpolate to match HuBERT frames
            if len(f0) != T:
                f0 = np.interp(np.linspace(0, len(f0), T), np.arange(len(f0)), f0).astype(np.float32)

            voiced = f0 > 0
            if np.any(voiced):
                avg_pitch = np.mean(f0[voiced])
                print(f"[DEBUG] Original pitch: {avg_pitch:.1f} Hz ({np.sum(voiced)}/{len(f0)} voiced)")
            else:
                print("[WARN] No voiced frames detected!")

        except Exception as e:
            print(f"[ERROR] CREPE failed: {e}")
            f0 = np.zeros(T, dtype=np.float32)

        # PITCH SHIFTING
        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)

        voiced = f0 > 0

        # Apply shift only to voiced segments
        f0[voiced] *= factor

        if np.any(voiced):
            print(f"[DEBUG] Shifted pitch: {np.mean(f0[voiced]):.1f} Hz (factor={factor:.3f})")

        # COARSE PITCH
        pitch_coarse = f0_to_coarse(f0)

        # Ensure unvoiced segments are mapped to the 0/silence bucket
        pitch_coarse[~voiced] = 0

        inputs = {
            "feats": embed,
            "p_len": np.array([T], dtype=np.int64),
            "pitch": pitch_coarse[None, :],
            "pitchf": f0[None, :],
            "sid": np.array([0], dtype=np.int64)
        }

        # RUN VOICE MODEL (outputs at MODEL_SR)
        print("[DEBUG] Running Voice Model...")
        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        print(f"[DEBUG] Model output: {len(out)} samples at {MODEL_SAMPLE_RATE} Hz")

        # RESAMPLE MODEL_SR -> 48k for playback
        print(f"[DEBUG] Resampling to 48kHz...")
        out_tensor = torch.from_numpy(out)
        out_tensor = output_resampler(out_tensor)
        out = out_tensor.numpy()

        print(f"[DEBUG] Final output: {len(out)} samples at 48kHz = {len(out)/TARGET_SAMPLE_RATE:.2f}s")
        print(f"[DEBUG] Processing time: {time.time() - t_start:.2f}s")
        print("="*60 + "\n")

        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] Processing failed:", e)
        traceback.print_exc()
        return b''

def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None
        data += chunk
    return data

print("\n" + "="*60)
print("SERVER READY - Waiting for C++ Client...")
print("="*60 + "\n")

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"[DEBUG] Connected: {addr}")
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
                            print(f"[ERROR] Inference failed: {e}")
                            conn.sendall(struct.pack('i', 0))
    except Exception as e:
        print("[ERROR] Socket error:", e)
        traceback.print_exc()