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
import configparser

# ---------------- COLORS ----------------
GREEN = "\033[32m"
RESET = "\033[0m"

def green_print(msg):
    print(f"{GREEN}{msg}{RESET}")

# ---------------- CONFIG LOADING ----------------
config = configparser.ConfigParser()
config.read('config.ini')

HOST = config['NETWORK'].get('HOST', '127.0.0.1')
PORT = config['NETWORK'].getint('PORT', 51235)
MODEL_PATH = config['PATHS'].get('MODEL_PATH', 'models/voice.onnx')
HUBERT_PATH = config['PATHS'].get('HUBERT_PATH', 'models/hubert.onnx')
INDEX_PATH = config['PATHS'].get('INDEX_PATH', 'models/voice.index')
INDEX_RATE = config['MODEL_TUNING'].getfloat('INDEX_RATE', 0.75)
MODEL_SAMPLE_RATE = config['MODEL_TUNING'].getint('MODEL_SAMPLE_RATE', 40000)

TARGET_SAMPLE_RATE = 48000
INPUT_SR = 48000

# ---------------- PRINT SELECTED STATUS ----------------
green_print(f"[Server] Model sample rate: {MODEL_SAMPLE_RATE} Hz")
green_print(f"[Server] Index rate: {INDEX_RATE}")

# ---------------- LOAD FAISS INDEX ----------------
index = None
try:
    index = faiss.read_index(INDEX_PATH)
    if hasattr(index, 'make_direct_map'):
        index.make_direct_map()
except Exception:
    index = None  # silently fail

# ---------------- LOAD AI MODELS ----------------
opts = ort.SessionOptions()
opts.intra_op_num_threads = 4
opts.log_severity_level = 3

hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
green_print(f"[Server] HuBERT loaded from: {HUBERT_PATH}")

voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
green_print(f"[Server] Voice model loaded from: {MODEL_PATH}")

# ---------------- RESAMPLERS ----------------
analysis_resampler = torchaudio.transforms.Resample(INPUT_SR, 16000)
output_resampler = torchaudio.transforms.Resample(MODEL_SAMPLE_RATE, TARGET_SAMPLE_RATE)

# ---------------- UTILS ----------------
def f0_to_coarse(f0):
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel_min = 1127 * np.log(1 + 50 / 700)
    f0_mel_max = 1127 * np.log(1 + 1100 / 700)
    f0_coarse = (f0_mel - f0_mel_min) * 254 / (f0_mel_max - f0_mel_min) + 1
    f0_coarse[f0_mel <= f0_mel_min] = 1
    f0_coarse[f0_mel >= f0_mel_max] = 255
    return np.clip(np.round(f0_coarse), 1, 255).astype(np.int64)

def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
        data += chunk
    return data

# ---------------- AUDIO PROCESSING ----------------
def process_audio(data, pitch_shift):
    try:
        audio_raw = np.frombuffer(data, dtype=np.float32).copy()
        green_print(f"[Server] Received audio: {len(audio_raw)} samples")
        if len(audio_raw) < 1600:
            return b''

        audio_tensor = torch.from_numpy(audio_raw)
        audio_16k_tensor = analysis_resampler(audio_tensor)
        audio_16k = audio_16k_tensor.numpy()

        # HuBERT embedding
        inp = audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        # INDEX MIXING
        if index is not None and INDEX_RATE > 0:
            try:
                query = embed.squeeze(0).astype(np.float32)
                D, I = index.search(query, 1)
                retrieved = index.reconstruct_batch(I.flatten().astype(np.int64))
                embed_mixed = (query * (1 - INDEX_RATE)) + (retrieved * INDEX_RATE)
                embed = embed_mixed[np.newaxis, :, :]
            except Exception:
                pass

        T = embed.shape[1]

        # Pitch detection (CREPE)
        try:
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
            ).squeeze(0).numpy()
            if len(f0) != T:
                f0 = np.interp(np.linspace(0, len(f0), T), np.arange(len(f0)), f0).astype(np.float32)
        except Exception:
            f0 = np.zeros(T, dtype=np.float32)

        factor = 2 ** (pitch_shift / 12.0)
        voiced = f0 > 0
        f0[voiced] *= factor
        pitch_coarse = f0_to_coarse(f0)
        pitch_coarse[~voiced] = 0

        inputs = {
            "feats": embed,
            "p_len": np.array([T], dtype=np.int64),
            "pitch": pitch_coarse[None, :],
            "pitchf": f0[None, :],
            "sid": np.array([0], dtype=np.int64)
        }

        out = voice_sess.run(["audio"], inputs)[0].squeeze().astype(np.float32)
        if out.ndim > 1:
            out = out.reshape(-1)

        out_tensor = torch.from_numpy(out)
        out_tensor = output_resampler(out_tensor)
        out = out_tensor.numpy()

        return out.astype(np.float32).tobytes()

    except Exception:
        return b''

# ---------------- MAIN SERVER LOOP ----------------
green_print(f"[Server] Server ready. Listening on {HOST}:{PORT}")

while True:
    server_socket = None
    try:
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server_socket.bind((HOST, PORT))
        server_socket.listen()

        conn, addr = server_socket.accept()
        green_print(f"[Server] Client connected: {addr}")

        with conn:
            while True:
                header = recvall(conn, 8)
                if not header:
                    green_print("[Server] Client disconnected")
                    break

                size, pitch = struct.unpack('ii', header)
                data = recvall(conn, size * 4)
                if not data:
                    green_print("[Server] Client disconnected")
                    break

                try:
                    result = process_audio(data, pitch)
                    if len(result) > 0:
                        conn.sendall(struct.pack('i', len(result)//4))
                        conn.sendall(result)
                    else:
                        conn.sendall(struct.pack('i', 0))
                except Exception:
                    conn.sendall(struct.pack('i', 0))

    except KeyboardInterrupt:
        break

    except Exception:
        time.sleep(1)

    finally:
        if server_socket:
            server_socket.close()
            server_socket = None
            green_print("[Server] Socket closed")

green_print("[Server] Server terminated")