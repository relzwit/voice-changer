# server.py

import socket
import struct
import numpy as np
import onnxruntime as ort
import torch
import torchaudio
import torchcrepe
import sys
import traceback
import time
import faiss

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 51235
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"
INDEX_PATH = "models/voice.index"
INDEX_RATE = 0.75

MODEL_SAMPLE_RATE = 40000
TARGET_SAMPLE_RATE = 48000
INPUT_SR = 48000

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("Loading Models...")

# --- LOAD FAISS INDEX ---
index = None
try:
    print(f"Attempting to load index: {INDEX_PATH}")
    index = faiss.read_index(INDEX_PATH)

    # --- FIXED: INITIALIZE DIRECT MAP ---
    # This is required for .reconstruct_batch() to work on IVF indices.
    # It builds a lookup table in RAM so we can retrieve original vectors.
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()

    print(f"Index Loaded Successfully. Total Vectors: {index.ntotal}")
except Exception as e:
    print(f"[WARN] Could not load Index file: {e}")
    print("Continuing in 'Index-less' mode (Lower Quality).")

try:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3

    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
    print("AI Loaded. Ready.")
except Exception as e:
    print("CRITICAL MODEL LOAD ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

# --- INIT RESAMPLERS ---
try:
    input_resampler = torchaudio.transforms.Resample(
        orig_freq=INPUT_SR,
        new_freq=MODEL_SAMPLE_RATE,
        dtype=torch.float32
    )

    output_resampler = torchaudio.transforms.Resample(
        orig_freq=MODEL_SAMPLE_RATE,
        new_freq=TARGET_SAMPLE_RATE,
        dtype=torch.float32
    )
    print("Torchaudio Resamplers Initialized.")
except Exception as e:
    print("Error initializing torchaudio:", e)
    sys.exit(1)

# --- PITCH DETECTION ---
def get_f0(audio, sr=MODEL_SAMPLE_RATE):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    f0 = torchcrepe.predict(
        x, sr,
        hop_length=160,
        fmin=50,
        fmax=1000,
        model='tiny',
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi
    )

    return f0.squeeze(0).cpu().numpy()

# --- AUDIO PROCESSING ---
def process_audio(data, pitch_shift):
    try:
        t_start = time.time()

        # 1. PREPARE AUDIO
        # --- FIXED: ADD .copy() ---
        # np.frombuffer creates a read-only view of memory.
        # PyTorch complains about this. .copy() makes it a writable array.
        audio_raw = np.frombuffer(data, dtype=np.float32).copy()

        if len(audio_raw) < 1600: return b''

        # Resample Input (48k -> 40k)
        audio_tensor = torch.from_numpy(audio_raw)
        audio_tensor = input_resampler(audio_tensor)
        audio = audio_tensor.numpy()

        # 2. HUBERT (CONTENT)
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        # --- INDEX MIXING ---
        if index is not None and INDEX_RATE > 0:
            try:
                query = embed.squeeze(0).astype(np.float32)

                D, I = index.search(query, 1)

                # This line crashed before. It should work now.
                retrieved = index.reconstruct_batch(I.flatten().astype(np.int64))

                weight = INDEX_RATE
                embed_mixed = (query * (1 - weight)) + (retrieved * weight)

                embed = embed_mixed[np.newaxis, :, :]
            except Exception as e:
                # If it still fails, we print detailed info to debug
                print(f"[Index Error] Skipping index: {e}")

        T = embed.shape[1]

        # 3. PITCH
        try:
            f0 = get_f0(audio, MODEL_SAMPLE_RATE)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        if len(f0) != T:
            xp = np.arange(len(f0))
            x  = np.linspace(0, len(f0) - 1, T)
            f0 = np.interp(x, xp, f0).astype(np.float32)

        # 4. INFERENCE
        pitchf = f0[None, :]
        pitch  = np.zeros((1, T), dtype=np.int64)
        p_len  = np.array([T], dtype=np.int64)
        sid    = np.array([0], dtype=np.int64)

        inputs = {
            "feats": embed,
            "p_len": p_len,
            "pitch": pitch,
            "pitchf": pitchf,
            "sid": sid
        }

        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        # 5. RESAMPLE OUTPUT
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            try:
                out_tensor = torch.from_numpy(out)
                out_tensor = output_resampler(out_tensor)
                out = out_tensor.numpy()
            except Exception as e:
                print("[WARN] output resample failed:", e)

        process_time = time.time() - t_start
        print(f"  -> AI processed {len(audio_raw)/INPUT_SR:.2f}s audio in {process_time:.2f}s")

        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] process_audio:", e)
        traceback.print_exc()
        return b''

# --- NETWORK LOOP ---
def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None
        data += chunk
    return data

print("Waiting for C++ Client...")

while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()

            conn, addr = s.accept()
            with conn:
                print("Connected:", addr)
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