import socket             # // Imports the socket library for network communication.
import struct             # // Imports struct to pack/unpack C data types (like integers) to/from bytes.
import numpy as np        # // Imports NumPy for efficient array manipulation (the language of AI data).
import onnxruntime as ort # // Imports the ONNX Runtime library to execute the neural network model.
import torch              # // Imports PyTorch (required by the pitch detection library).
import torchcrepe         # // Imports the AI-based pitch detection library (Crepe).
import sys                # // Imports sys for system-level functions.
import librosa            # // Imports librosa for high-quality audio resampling.

# --- CONFIGURATION ---
HOST = '127.0.0.1'       # // The IP address the server listens on (Localhost: same machine).
PORT = 5555              # // The port number for communication with the C++ client.
MODEL_PATH = "models/voice.onnx"   # // Path to the final Voice Model (The 'Mouth').
HUBERT_PATH = "models/hubert.onnx" # // Path to the Content Encoder (The 'Ear').
MODEL_SAMPLE_RATE = 40000          # // The sample rate the RVC model was trained on (40kHz).
TARGET_SAMPLE_RATE = 48000         # // The sample rate of the user's sound card (48kHz).

print(f"Python Version: {sys.version}") # // Prints Python version for debugging.
print("Loading Models...")

try:
    sess_options = ort.SessionOptions()
    sess_options.log_severity_level = 3 # // Suppress low-level logging/warnings from ONNX.

    # Load the Hubert model (inference session)
    hubert_sess = ort.InferenceSession(HUBERT_PATH, sess_options)
    # Load the Voice Model (inference session)
    voice_sess = ort.InferenceSession(MODEL_PATH, sess_options)
    print("AI Loaded. Ready.")
except Exception as e:
    print(f"CRITICAL ERROR LOADING MODELS: {e}")
    exit(1)

# --- PITCH DETECTION ---
def get_f0(audio, sr=16000):
    # // Convert the NumPy array (audio) into a PyTorch tensor, adding a batch dimension (unsqueeze).
    audio_torch = torch.from_numpy(audio).unsqueeze(0)

    # // Run the Crepe pitch detection AI model.
    f0 = torchcrepe.predict(
        audio_torch,
        sr,
        hop_length=128,          # // Scans pitch every 8ms (high resolution).
        fmin=50, fmax=1000,      # // Limits pitch search to human vocal range.
        model='full',            # // Use the highest quality model for best results.
        batch_size=256,
        device='cpu',
        decoder=torchcrepe.decode.viterbi
    )
    # // Convert the result back to a NumPy array for use with ONNX.
    return f0.squeeze(0).numpy()

def process_audio(audio_bytes, pitch_shift):
    # // Convert raw float bytes received from C++ into a NumPy array.
    audio = np.frombuffer(audio_bytes, dtype=np.float32)

    # // Safety Check: If the audio is too short (< 0.1s), return empty and skip processing.
    if len(audio) < 1600: return b''

    # 1. RUN HUBERT (Content Encoding)
    # // Reshape input for Hubert: [1, 1, Time] (Batch, Channel, Time)
    hubert_input = audio[np.newaxis, np.newaxis, :]
    # // Run Hubert: input 'source' -> output 'embed' (the voice content features).
    embed = hubert_sess.run(["embed"], {"source": hubert_input})[0]

    # 2. DETECT PITCH (F0)
    try:
        f0 = get_f0(audio)
    except:
        # // Fallback to zero pitch if pitch detection fails.
        f0 = np.zeros(embed.shape[1], dtype=np.float32)

    # 3. APPLY SHIFT (F0)
    f0_shifted = f0 * (2.0 ** (pitch_shift / 12.0)) # // Calculate the frequency multiplier (e.g., *2.0 for +12 semitones).

    # // Ensure the pitch vector is the same length as the embedding vector (essential step).
    target_len = embed.shape[1]
    if len(f0_shifted) != target_len:
        # // Interpolate the pitch curve to match the embedding length.
        f0_shifted = np.interp(np.linspace(0, len(f0_shifted), target_len), np.arange(len(f0_shifted)), f0_shifted)

    # 4. PREPARE VOICE MODEL INPUTS
    length = np.array([target_len], dtype=np.int64)
    sid = np.array([0], dtype=np.int64)
    f0_coarse = np.zeros_like(f0_shifted, dtype=np.int64) # // Pitch ID index (set to 0 here as we rely on pitchf).

    # // Create the final inputs dictionary for the RVC Voice Model.
    inputs = {
        "feats": embed,
        "p_len": length,
        "pitch": f0_coarse[None, :],                # // Add Batch Dimension [1, Time].
        "pitchf": f0_shifted.astype(np.float32)[None, :], # // Add Batch Dimension [1, Time].
        "sid": sid
    }

    # // Run the Voice Model (Timbre Swap). Result is raw audio (e.g., 40kHz).
    audio_out = voice_sess.run(["audio"], inputs)[0]

    # 5. RESAMPLE OUTPUT TO 48K
    if MODEL_SAMPLE_RATE != TARGET_SAMPLE_RATE:
        # // Librosa handles the high-quality 40kHz -> 48kHz resampling.
        audio_out = librosa.resample(audio_out, orig_sr=MODEL_SAMPLE_RATE, target_sr=TARGET_SAMPLE_RATE)

    # // Return the raw audio bytes.
    return audio_out.tobytes()

# --- HELPER: Guarantee we read N bytes ---
def recvall(sock, n):
    data = b''
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet: return None # // Connection closed (EOF).
        data += packet
    return data

# --- SERVER LOOP ---
print("Waiting for C++ Client...")
while True:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((HOST, PORT))
            s.listen()
            conn, addr = s.accept()
            with conn:
                print(f"Connected: {addr}")
                while True:
                    # // 1. Read Header (8 bytes) using the safe recvall function.
                    header = recvall(conn, 8)
                    if not header: break
                    size, pitch = struct.unpack('ii', header) # // Unpack [Size] and [Pitch].

                    # // 2. Read Audio Data (Size * 4 bytes)
                    data = recvall(conn, size * 4)
                    if not data: break

                    # // 3. Process and Send
                    if len(data) == size * 4:
                        try:
                            result = process_audio(data, pitch)
                            if len(result) > 0:
                                # // Send back the size of the result array.
                                conn.sendall(struct.pack('i', len(result)//4))
                                # // Send back the processed audio data.
                                conn.sendall(result)
                                print(f"Processed: {size} samples")
                            else:
                                conn.sendall(struct.pack('i', 0))
                        except Exception as e:
                            print(f"Inference Error: {e}")
                            conn.sendall(struct.pack('i', 0))
    except Exception as e:
        print(f"Socket Error (Restarting Server): {e}")