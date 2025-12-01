import socket             # // Networking library.
import struct             # // Binary data packing.
import numpy as np        # // Math library.
import onnxruntime as ort # // AI Runtime.
import torch              # // PyTorch.
import torchcrepe         # // Pitch Detection.
import sys
import librosa            # // Audio processing.
import traceback          # // Error logging.

# --- CONFIG ---
HOST = '127.0.0.1'
PORT = 51235
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"

MODEL_SAMPLE_RATE = 40000
TARGET_SAMPLE_RATE = 48000
INPUT_SR = 40000

print(f"Python Version: {sys.version}")
print("Loading Models...")

try:
    # // Configure ONNX to use all available CPU power.
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4 # // Allow parallel math.
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3

    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
    print("AI Loaded. Ready.")
except Exception as e:
    print("CRITICAL MODEL LOAD ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

# --- PITCH DETECTION (OPTIMIZED) ---
def get_f0(audio, sr=INPUT_SR):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # // Convert to PyTorch Tensor.
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    # // RUN CREPE (SPEED OPTIMIZED)
    # // We changed model='full' -> 'tiny'. This is much faster.
    f0 = torchcrepe.predict(
        x, sr,
        hop_length=160,       # // Increased hop (160) = Faster scan.
        fmin=50,
        fmax=1000,
        model='tiny',         # // 'tiny' is lightweight and fast on CPU.
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi
    )

    return f0.squeeze(0).cpu().numpy()

# --- AUDIO PROCESSING ---
def process_audio(data, pitch_shift):
    try:
        t_start = time.time() # // Start timer.

        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) < 1600: return b''

        # 1. Hubert Encoding
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        T = embed.shape[1]

        # 2. Pitch Detection
        try:
            f0 = get_f0(audio, INPUT_SR)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        # 3. Pitch Shifting
        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        if len(f0) != T:
            xp = np.arange(len(f0))
            x  = np.linspace(0, len(f0) - 1, T)
            f0 = np.interp(x, xp, f0).astype(np.float32)

        # 4. Prepare Inputs
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

        # 5. Run Model
        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        # 6. Resample
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            try:
                out = librosa.resample(out, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)
            except Exception as e:
                print("[WARN] resample failed:", e)

        # // Print timing debug
        process_time = time.time() - t_start
        print(f"  -> AI processed {len(audio)/INPUT_SR:.2f}s audio in {process_time:.2f}s")

        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] process_audio:", e)
        traceback.print_exc()
        return b''

# --- NETWORK LOOP ---
import time # // Re-import time for stats

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