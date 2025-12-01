# server.py

import socket             # Standard networking library (TCP/IP).
import struct             # Binary data packing (to read C++ structs).
import numpy as np        # Heavy math library for matrix operations.
import onnxruntime as ort # The engine that runs the AI models (faster than raw PyTorch).
import torch              # PyTorch: Used here for Tensor manipulation.
import torchaudio         # <--- CHANGED: Adobe-quality resampling, optimized for CPU speed.
import torchcrepe         # The library that detects "Pitch" (how high/low the voice is).
import sys
import traceback          # detailed error printing.
import time               # For measuring how fast the AI runs.

# --- CONFIG ---
HOST = '127.0.0.1'       # Localhost (we only accept connections from this computer).
PORT = 51235             # The specific port C++ will connect to.
MODEL_PATH = "models/voice.onnx"   # The Voice Model (The "Speaker Identity").
HUBERT_PATH = "models/hubert.onnx" # The Content Model (Recognizes "What" is being said).

# RVC models are usually trained at 40kHz or 48kHz.
# Most v2 models are 40k. If we don't match this, the voice sounds fast/slow (chipmunk effect).
MODEL_SAMPLE_RATE = 40000
TARGET_SAMPLE_RATE = 48000 # The C++ client expects DVD quality (48k).
INPUT_SR = 40000           # We assume C++ has already downsampled the input to 40k.

print(f"Python Version: {sys.version}")
print(f"PyTorch Version: {torch.__version__}")
print("Loading Models...")

try:
    # --- ONNX RUNTIME OPTIMIZATION ---
    # Since we are on a CPU (ThinkPad T480), we need every ounce of speed.
    opts = ort.SessionOptions()

    # "Intra Op Threads": How many CPU cores to use for ONE matrix math operation.
    # 4 is usually the sweet spot for quad-core laptops.
    opts.intra_op_num_threads = 4

    # Enable all internal optimizations (graph fusion, constant folding, etc.)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.log_severity_level = 3 # Only show errors, hide warnings.

    # Load the "Brains".
    # Hubert: Reads the audio and extracts the "Lyrics/Phonemes".
    hubert_sess = ort.InferenceSession(HUBERT_PATH, opts)
    # Voice: Takes the Lyrics + Pitch and generates the new voice.
    voice_sess  = ort.InferenceSession(MODEL_PATH, opts)
    print("AI Loaded. Ready.")
except Exception as e:
    print("CRITICAL MODEL LOAD ERROR:", e)
    traceback.print_exc()
    sys.exit(1)

# --- INIT RESAMPLER GLOBALLY ---
# OPTIMIZATION: We create the resampler object ONCE here.
# Creating it inside the loop would waste CPU time rebuilding the math filters every 5 seconds.
# 'torchaudio' provides near-perfect quality (Kaiser window) with C++ speed.
try:
    resampler = torchaudio.transforms.Resample(
        orig_freq=MODEL_SAMPLE_RATE,
        new_freq=TARGET_SAMPLE_RATE,
        dtype=torch.float32
    )
    print("Torchaudio Resampler Initialized.")
except Exception as e:
    print("Error initializing torchaudio:", e)
    sys.exit(1)

# --- PITCH DETECTION (CREPE) ---
# "F0" stands for Fundamental Frequency (The pitch of the voice).
def get_f0(audio, sr=INPUT_SR):
    # Ensure input is a numpy array.
    if not isinstance(audio, np.ndarray):
        audio = np.array(audio, dtype=np.float32)

    # Convert to PyTorch Tensor (required for Crepe).
    # .unsqueeze(0) adds a "Batch Dimension" (Models expect a batch of files, even if it's just 1).
    x = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)

    # RUN CREPE (SPEED OPTIMIZED)
    # We use 'torchcrepe' to find the pitch curve.
    f0 = torchcrepe.predict(
        x, sr,
        hop_length=160,       # Optimization: Larger hop = Faster scan, slightly less precision.
        fmin=50,              # Ignore deep rumbles below 50Hz.
        fmax=1000,            # Ignore high squeaks above 1000Hz.
        model='tiny',         # Optimization: Use the smallest neural net (fastest on CPU).
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi # 'Viterbi' algorithm smooths the jittery pitch line.
    )

    # Convert back to Numpy and remove the batch dimension.
    return f0.squeeze(0).cpu().numpy()

# --- AUDIO PROCESSING (THE MAIN PIPELINE) ---

def process_audio(data, pitch_shift):
    try:
        t_start = time.time() # Start stopwatch.

        # Convert raw C++ bytes into a float array.
        audio = np.frombuffer(data, dtype=np.float32)

        # Safety: If audio is too short (milliseconds), the AI can't detect pitch.
        if len(audio) < 1600: return b''

        # 1. HUBERT ENCODING (Extract Content)
        # We run the Hubert model to get the "Features" (What is being said).
        inp = audio[np.newaxis, np.newaxis, :].astype(np.float32)
        embed = hubert_sess.run(["embed"], {"source": inp})[0]

        # Transpose matrix dimensions to match what the Voice Model expects.
        if embed.ndim == 3 and embed.shape[1] == 768:
            embed = np.transpose(embed, (0, 2, 1))

        T = embed.shape[1] # The time dimension (number of frames).

        # 2. PITCH DETECTION
        try:
            f0 = get_f0(audio, INPUT_SR)
        except Exception as e:
            print("[WARN] F0 failed:", e)
            f0 = np.zeros(T, dtype=np.float32)

        # 3. PITCH SHIFTING
        # We multiply the detected frequency by a factor.
        # Formula: NewFreq = OldFreq * 2^(semitones / 12)
        f0 = np.asarray(f0, dtype=np.float32)
        factor = 2 ** (pitch_shift / 12.0)
        f0 *= factor

        # INTERPOLATION
        # The Pitch array (f0) and Content array (embed) might have slightly different lengths
        # due to rounding. We stretch 'f0' to match 'embed' perfectly.
        if len(f0) != T:
            xp = np.arange(len(f0))
            x  = np.linspace(0, len(f0) - 1, T)
            f0 = np.interp(x, xp, f0).astype(np.float32)

        # 4. PREPARE INPUTS FOR VOICE MODEL
        pitchf = f0[None, :]
        pitch  = np.zeros((1, T), dtype=np.int64) # Coarse pitch (optional, we use zero).
        p_len  = np.array([T], dtype=np.int64)    # Length of the audio.
        sid    = np.array([0], dtype=np.int64)    # Speaker ID (usually 0 for single-speaker models).

        inputs = {
            "feats": embed,
            "p_len": p_len,
            "pitch": pitch,
            "pitchf": pitchf,
            "sid": sid
        }

        # 5. RUN VOICE MODEL (INFERENCE)
        # This is the heavy math step. Takes features + pitch -> New Audio.
        out = voice_sess.run(["audio"], inputs)[0]
        out = np.squeeze(out).astype(np.float32)

        if out.ndim > 1:
            out = out.reshape(-1)

        # 6. RESAMPLE (40k -> 48k)
        if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
            try:
                # Torchaudio requires a PyTorch Tensor.
                out_tensor = torch.from_numpy(out)

                # Run the globally initialized resampler.
                out_tensor = resampler(out_tensor)

                # Convert back to Numpy so we can send bytes.
                out = out_tensor.numpy()
            except Exception as e:
                print("[WARN] resample failed:", e)

        # Print debug info so we know how fast it's running.
        process_time = time.time() - t_start
        print(f"  -> AI processed {len(audio)/INPUT_SR:.2f}s audio in {process_time:.2f}s")

        # Convert array back to raw bytes for C++.
        return out.astype(np.float32).tobytes()

    except Exception as e:
        print("[ERROR] process_audio:", e)
        traceback.print_exc()
        return b''

# --- NETWORK LOOP ---
# Helper function to solve TCP Fragmentation.
# "Recv(n)" might only return n/2 bytes if the network is busy.
# We loop until we get exactly 'n' bytes.
def recvall(sock, n):
    data = b''
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk: return None # Connection closed.
        data += chunk
    return data

print("Waiting for C++ Client...")

# Infinite Loop: Keep the server running forever.
while True:
    try:
        # Create the socket.
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # Allow us to restart the server immediately without waiting for the OS to timeout the port.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()

            # Wait for C++ to connect.
            conn, addr = s.accept()
            with conn:
                print("Connected:", addr)

                # Inner Loop: Handle one specific connection.
                while True:
                    # 1. Read Header (8 bytes: 4 for size, 4 for pitch).
                    header = recvall(conn, 8)
                    if not header: break

                    # 'struct.unpack' converts binary C-structs into Python numbers.
                    size, pitch = struct.unpack('ii', header)

                    # 2. Read Audio Data based on the size we just read.
                    data = recvall(conn, size * 4) # 4 bytes per float.
                    if not data: break

                    if len(data) == size * 4:
                        try:
                            # 3. Process the Audio.
                            result = process_audio(data, pitch)

                            # 4. Send Response.
                            if len(result) > 0:
                                # Send size first.
                                conn.sendall(struct.pack('i', len(result)//4))
                                # Send audio data.
                                conn.sendall(result)
                            else:
                                # Send size 0 if error.
                                conn.sendall(struct.pack('i', 0))
                        except Exception as e:
                            print(f"Inference Error: {e}")
                            conn.sendall(struct.pack('i', 0))
    except Exception as e:
        print("Socket Error:", e)
        traceback.print_exc()