import socket             # // Networking.
import struct             # // Binary packing.
import numpy as np        # // Math.
import onnxruntime as ort # // AI Runtime.
import torch              # // PyTorch.
import torchcrepe         # // Pitch Detection.
import sys
import librosa            # // Audio Resampling.
import traceback          # // Error reporting.

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 51235
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"

MODEL_SAMPLE_RATE = 40000   # // RVC Native Rate.
TARGET_SAMPLE_RATE = 48000  # // Playback Rate.
INPUT_SR = 40000            # // Input Rate (C++ sends 40k).

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

# --- PITCH DETECTION ---
def get_f0(audio, sr=INPUT_SR):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # // Convert to Tensor for TorchCrepe.
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    # // Run High-Quality Pitch Detection.
    f0 = torchcrepe.predict(
        x, sr,
        hop_length=128,
        fmin=50,
        fmax=1000,
        model='full', # // High accuracy model.
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi
    )

    return f0.squeeze(0).cpu().numpy()

# --- AUDIO PROCESSING ---
def process_audio(data, pitch_shift):
    try:
        # // Convert bytes to float array.
        audio = np.frombuffer(data, dtype=np.float32)

        # // Ignore empty/tiny buffers.
        if len(audio) < 1600:
            return b''

        # 1. HUBERT ENCODING
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        # --- CRITICAL FIX: TRANSPOSE ---
        # // Check if dimensions are flipped (Channels, Time) -> (Time, Channels).
        # // If we don't do this, the model hears garbage.
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        T = embed.shape[1] # // Time steps.

        # 2. PITCH DETECTION
        try:
            f0 = get_f0(audio, INPUT_SR)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        # 3. PITCH SHIFTING
        f0 = np.asarray(f0, dtype=np.float32)
        # // Math: frequency * 2^(semitones/12).
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        # // Interpolate pitch curve to match embedding length (T).
        if len(f0) != T:
            xp = np.arange(len(f0))
            x  = np.linspace(0, len(f0) - 1, T)
            f0 = np.interp(x, xp, f0).astype(np.float32)

        # 4. PREPARE TENSORS
        pitchf = f0[None, :]
        # // Coarse pitch is usually derived from pitchf, zero is a safe fallback for some models.
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

        # 5. RUN VOICE MODEL
        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        # 6. RESAMPLE OUTPUT
        # // Convert model rate (40k) to playback rate (48k).
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            try:
                out = librosa.resample(out, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)
            except Exception as e:
                print("[WARN] resample failed:", e)

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
                    # // Read 8-byte Header.
                    header = recvall(conn, 8)
                    if not header: break

                    size, pitch = struct.unpack('ii', header)

                    # // Read Audio Payload.
                    data = recvall(conn, size * 4)
                    if not data: break

                    result = process_audio(data, pitch)

                    # // Send Response.
                    if len(result) > 0:
                        conn.sendall(struct.pack('i', len(result)//4))
                        conn.sendall(result)
                        print(f"Processed: {size} -> {len(result)//4}")
                    else:
                        conn.sendall(struct.pack('i', 0))

    except Exception as e:
        print("Socket Error:", e)