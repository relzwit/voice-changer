import socket
import struct
import numpy as np
import onnxruntime as ort
import torch
import torchcrepe
import sys
import librosa
import traceback

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 51235
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"

MODEL_SAMPLE_RATE = 40000   # // RVC Native Rate
TARGET_SAMPLE_RATE = 48000  # // Output Rate (Back to C++)
INPUT_SR = 48000            # // Input Rate (From C++)
AI_INTERNAL_SR = 16000      # // What Hubert/Crepe need

print(f"Python Version: {sys.version}")
print("Loading Models...")

try:
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
    print("AI Loaded. Ready.")
except Exception as e:
    print("CRITICAL MODEL LOAD ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

def get_f0(audio, sr=16000):
    audio_torch = torch.from_numpy(audio).unsqueeze(0)
    f0 = torchcrepe.predict(
        audio_torch, sr,
        hop_length=128, fmin=50, fmax=1000,
        model='full', batch_size=256, device='cpu', decoder=torchcrepe.decode.viterbi
    )
    return f0.squeeze(0).numpy()

def get_f0_coarse(f0):
    f0_min = 50
    f0_max = 1100
    f0_bin = 256
    f0_mel_min = 1127 * np.log(1 + f0_min / 700)
    f0_mel_max = 1127 * np.log(1 + f0_max / 700)
    f0_mel = 1127 * np.log(1 + f0 / 700)
    f0_mel[f0_mel > 0] = (f0_mel[f0_mel > 0] - f0_mel_min) * (f0_bin - 1) / (f0_mel_max - f0_mel_min) + 1
    f0_coarse = np.rint(f0_mel).astype(np.int64)
    f0_coarse[f0_coarse <= 1] = 1
    f0_coarse[f0_coarse > 255] = 255
    return f0_coarse

def process_audio(data, pitch_shift):
    try:
        # // 1. Receive 48k Audio
        audio_48k = np.frombuffer(data, dtype=np.float32).copy()

        if len(audio_48k) < 2400: return b''

        # // 2. Downsample to 16k for AI (Hubert/Crepe)
        # // Librosa does this with high quality (Anti-aliasing)
        audio_16k = librosa.resample(audio_48k, orig_sr=INPUT_SR, target_sr=AI_INTERNAL_SR)

        # // 3. Hubert Encoding
        inp = audio_16k[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))
        T = embed.shape[1]

        # // 4. Pitch Detection (on 16k audio)
        try:
            f0 = get_f0(audio_16k, AI_INTERNAL_SR)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        # // 5. Shift & Floor
        f0 = np.clip(f0, 100.0, 1000.0)
        f0_shifted = f0 * (2.0 ** (pitch_shift / 12.0))
        f0_shifted = np.clip(f0_shifted, 250.0, 1500.0)

        if len(f0_shifted) != T:
            f0_shifted = np.interp(np.linspace(0, len(f0_shifted), T), np.arange(len(f0_shifted)), f0_shifted)

        length = np.array([T], dtype=np.int64)
        sid = np.array([0], dtype=np.int64)
        f0_coarse = get_f0_coarse(f0_shifted)

        inputs = {
            "feats": embed, "p_len": length, "pitch": f0_coarse[None, :],
            "pitchf": f0_shifted.astype(np.float32)[None, :], "sid": sid
        }

        # // 6. Run Model (Outputs 40k)
        audio_out = voice_sess.run(["audio"], inputs)[0]
        audio_out = np.squeeze(audio_out)

        # // 7. Resample Output 40k -> 48k
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            audio_out = librosa.resample(audio_out, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)

        return audio_out.astype(np.float32).tobytes()

    except Exception as e:
        print(f"[ERROR] {e}")
        traceback.print_exc()
        return b''

# SERVER LOOP
def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None
        data += chunk
    return data

print("Waiting for C++ Client...")
with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen()
    conn, addr = s.accept()
    with conn:
        print(f"Connected: {addr}")
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
                        print(f"Processed: {size} -> {len(result)//4}")
                    else:
                        conn.sendall(struct.pack('i', 0))
                except:
                    conn.sendall(struct.pack('i', 0))