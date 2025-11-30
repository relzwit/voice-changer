🚀 RVC Voice Changer: Adaptive Python Bridge (CLI)

This is a high-performance, cross-platform voice changer application designed to run Retrieval-based Voice Conversion (RVC) models with minimal latency on systems of varying capabilities.

The project uses a Client-Server architecture to maximize stability:

    C++ Client (The Body): Handles low-latency audio hardware access and user input.

    Python Server (The Brain): Executes the heavy AI computation using PyTorch, ONNX, and specialized pitch detection.

🛠️ Prerequisites

Before compiling and running the application, you must have the following dependencies installed on your system (Fedora):

1. System Dependencies (C++ Toolchain)

You've already installed these, but for a new user:
Bash

# Compiler, CMake, Ninja, Git, Libs for RNNoise/ALSA
sudo dnf install cmake ninja-build git build-essential pkg-config alsa-lib-devel
sudo dnf install rnnoise-devel # AI Noise Reduction Library

2. Python Environment (The AI Brain)

The Python server requires a stable environment (Python 3.10 is recommended due to numba/librosa dependencies). You must install the required libraries within a virtual environment (venv):
Bash

# 1. Create and Activate the environment
python3.10 -m venv venv
source venv/bin/activate

# 2. Install Core AI Dependencies
pip install numpy==1.24.3 torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install onnxruntime faiss-cpu scipy librosa torchcrepe soundfile

# 3. Deactivate the environment when done
deactivate

3. Model Files (The Assets)

Create a models directory in your project root and place the following two files inside:
File Name	Role	Requirement
hubert.onnx	Content Encoder (The Ear)	vec-768-layer-12.onnx (300-400MB)
voice.onnx	Voice Model (The Mouth)	amitaro_v2_40k_e100_float.onnx (110MB)

▶️ How to Run the Program

You must run the Python Server before starting the C++ Client.

Step 1: Start the AI Server (Terminal 1)

Open your first terminal, navigate to the project directory, and start the Python environment:
Bash

# Activate your safe environment
source venv/bin/activate

# Start the AI Server
python server.py

(Wait until the server prints: "AI Loaded. Ready. Waiting for C++ Client...")

Step 2: Configure, Build, and Run the C++ Client (Terminal 2 / CLion)

    Open CLion and load the voice-changer project.

    Ensure CMake is configured to use vcpkg and the C++ standard library.

    Build the Project: Click the Hammer icon (or press Ctrl+F9).

    Run the Executable: Click the Play icon (or press Shift+F10).

The program will immediately prompt you for the configuration:

>>> VOICE CHANGER (PYTHON BRIDGE) <<<
...
CALIBRATING... PLEASE STAY SILENT.

(Wait for the calibration dots to finish.)

Step 3: Use the CLI Interface

Once calibration is complete, the application will enter the command loop.
Command	Action	Notes
5 (or any positive number)	Start Recording for 5 seconds.	Recommended for stable performance. Speak a full sentence during this time.
-1	Replay the last converted audio instantly.	Uses cached audio; skips processing time.
+18 (or any number)	Change Pitch to +18 semitones.	Default is +12. You can tune this to achieve the most natural feminine voice.
0	Quit the application.	Exits the C++ client.
