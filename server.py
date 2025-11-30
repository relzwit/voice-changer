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
PORT = 5555
MODEL_PATH = "models/voice.onnx"
HUBERT_PATH = "models/hubert.onnx"

MODEL_SAMPLE_RATE = 40000   # RVC trained sample rate (40k)
TARGET_SAMPLE_RATE = 48000  # We return 48k audio so C++ can playback natively
INPUT_SR = 40000            # The C++ side will send 40k audio now

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
    exit(1)


def get_f0(audio, sr=INPUT_SR):
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
    f0 = torchcrepe.predict(
        x, sr,
        hop_length=128,
        fmin=50,
        fmax=1000,
        model='full',
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi
    )
    return f0.squeeze(0).cpu().numpy()


def process_audio(data, pitch_shift):
    try:
        audio = np.frombuffer(data, dtype=np.float32)
        if len(audio) < 1600:
            return b''

        # Hubert expects [1,1,time]
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        # If hubert returned (1, 768, T) transpose to (1, T, 768)
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        if embed.ndim != 3:
            print("Unexpected embed shape:", embed.shape)
            return b''

        T = embed.shape[1]

        # Pitch detection at 40k
        try:
            f0 = get_f0(audio, INPUT_SR)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        # Resize pitch curve to match T
        if len(f0) != T:
            xp = np.arange(len(f0))
            x  = np.linspace(0, len(f0) - 1, T)
            f0 = np.interp(x, xp, f0).astype(np.float32)

        pitchf = f0[None, :]
        pitch  = np.zeros(T, dtype=np.int64)[None, :]
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

        # Resample model output (40k) -> 48k for playback
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            try:
                out = librosa.resample(out, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)
            except Exception as e:
                print("[WARN] resample failed:", e)
                src_len = out.shape[0]
                tgt_len = int(src_len * (TARGET_SAMPLE_RATE / float(MODEL_SAMPLE_RATE)))
                if src_len > 1 and tgt_len > 0:
                    xp = np.linspace(0, src_len - 1, src_len)
                    x = np.linspace(0, src_len - 1, tgt_len)
                    out = np.interp(x, xp, out).astype(np.float32)
                else:
                    out = np.array([], dtype=np.float32)

        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] process_audio:", e)
        traceback.print_exc()
        return b''


def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            return None
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
                    if not header:
                        break

                    size, pitch = struct.unpack('ii', header)
                    data = recvall(conn, size * 4)
                    if not data:
                        break

                    result = process_audio(data, pitch)

                    if len(result) > 0:
                        conn.sendall(struct.pack('i', len(result)//4))
                        conn.sendall(result)
                    else:
                        conn.sendall(struct.pack('i', 0))

    except Exception as e:
        print("Socket Error:", e)
        traceback.print_exc()
