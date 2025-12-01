Well, if you are wondering what this whole thing is. It is a project I made...
by prompting AI for the last idk....15ish hours? Split up by sleep and other activities.
I do not know C++ or how to use AI with Python.
This program is almost completely AI-written code.

I do want to go over it eventually and slowly understand everything that has been done.
This is why I made the AI comment so heavily.

Here is the AI explanation of the project:

# System Architecture

## The Concept: "The Factory & The Lab"

Think of this program as two distinct buildings connected by a pneumatic tube (the TCP Network Socket).

* 🏭 **The Factory (C++):** This is where the heavy lifting happens. It handles the raw materials (audio bytes) coming in and the finished product going out. It is built for **speed and safety**. It never stops moving.
* 🧪 **The Lab (Python):** This is where the magic happens. It is slower, but it is "smart." It takes a chunk of audio, studies it, reconstructs it, and sends it back.

---

## Phase 1: Initialization (The Setup)

Before you say a word, the system prepares the pipeline.

**Python Starts First (`server.py`):**
* It loads the **Hubert Model** (The "Ear"): This AI knows how to listen to speech and extract *what* is being said (phonemes), ignoring *who* is saying it.
* It loads the **RVC Model** (The "Voice"): This AI knows exactly what the target character sounds like.
* It opens a **TCP Socket** on Port `51235` and sits there, waiting.

**C++ Starts Second (`main.cpp`):**
* It initializes `miniaudio`. This library talks to your OS (Windows/Linux) and grabs exclusive access to the microphone and speakers.
* It creates two **Ring Buffers** (circular queues for thread-safe data transfer).
* It spawns a **Worker Thread** (`processing_thread_func`). This is a background worker who will do all the data moving so the main window doesn't freeze.
* It dials `127.0.0.1:51235`. If Python answers, the connection is established.

---

## Phase 2: The Capture (The Factory Floor)

**Action:** You press `5` and hit Enter. `g_is_recording` becomes `true`.

### A. The Hardware Interrupt (`data_callback`)
Every few milliseconds (specifically every 4096 samples), your sound card interrupts the CPU and says, *"I have audio data!"*

* The `data_callback` function runs immediately. It must be fast.
* It takes the raw floats from the mic and **writes** them into the `g_rb_input` (Input Ring Buffer).
* It looks at `g_rb_output` (Output Ring Buffer) to see if there is anything to play. If yes, it sends it to the speakers.

### B. The Worker Thread (`processing_thread_func`)
While the callback is dumping data into the buffer, the Worker Thread is waking up constantly to check the buffer.

1.  **Read:** It pulls 1024 samples out of the Input Ring Buffer.
2.  **Denoise (RNNoise):**
    * It multiplies the samples by `32768.0` (converting them to a range the AI understands).
    * The `DenoiseEngine` looks at the audio. If it hears steady static (fans) or sudden clicks, it zeros them out.
    * It divides by `32768.0` to return to normal volume.
3.  **Accumulate:** It takes this clean chunk and adds it to a growing list called `burst_buffer`.
4.  **Wait:** It repeats this until `burst_buffer` holds exactly 5 seconds of audio.

---

## Phase 3: The Transport (Crossing the Bridge)

Once the buffer is full (5 seconds), the Worker Thread triggers the "Process" phase.

1.  **Resample Down:**
    * Your mic is recording at **48,000Hz** (DVD Quality).
    * The AI models were trained at **40,000Hz**.
    * C++ runs `Resample48To40`. It calculates the ratio (0.833) and shrinks the array.

2.  **Serialization:**
    * The `PythonBridge` takes this vector of float numbers.
    * It creates a Header: `[Size of Audio] [Pitch Shift Amount]`.
    * It sends the Header first, then the huge chunk of Audio bytes over the TCP socket.

3.  **The Wait:**
    * The C++ thread effectively pauses here (technically inside `recv`). It prints the spinning `| / - \` animation while it waits for the Lab to finish.

---

## Phase 4: The Transformation (Inside the Lab)

Python receives the bytes. This is where the "Voice Change" actually occurs. This is the **RVC (Retrieval-based Voice Conversion)** pipeline:

### Step A: Feature Extraction (Hubert)
The Hubert model scans the audio. It is a "Content Encoder."
* **Input:** Your voice saying "Hello."
* **Output:** A mathematical representation (Embedding) of the *sound* "Hello," stripped of your accent, gender, and tone. It extracts the **lyrics**.

### Step B: Pitch Detection (Crepe)
We use `TorchCrepe` to scan the audio for frequency.
* It creates a curve representing how high or low your voice is at every millisecond.
* **Pitch Shifting:** If you requested `+12` semitones (1 octave), we simply multiply this frequency curve by `2.0`.
* **Result:** We now have the *melody* of your speech, but shifted up.

### Step C: Inference (The Voice Model)
This is the core AI. The `voice.onnx` model takes two inputs:
1.  The **Content** (from Hubert).
2.  The **New Pitch** (from Crepe).

It acts like a synthesizer. It "plays" the Content using the New Pitch, but using the specific vocal cords learned during its training (e.g., an Anime character or a celebrity).

### Step D: Resampling (The Return)
The AI generates audio at 40kHz. We need 48kHz to match the C++ system.
* We use `torchaudio` to stretch the audio back to 48kHz using high-quality math (Sinc Interpolation) to prevent metallic artifacts.

---

## Phase 5: The Playback (Shipping Out)

### 1. Receipt
Python sends the new audio bytes back over the socket. The C++ `PythonBridge` receives them and converts them back into a `std::vector<float>`.

### 2. Elastic Resampling (Crucial Step)
* **The Problem:** Audio clocks are imperfect. You sent exactly 240,000 samples. Python might send back 240,005 samples due to rounding errors. If we just played that, the audio would drift over time.
* **The Solution:** `ResampleToCount`. We force the received audio to be **exactly** the same length as the input buffer. If it's too long, we squash it slightly. If it's too short, we stretch it.

### 3. To the Speakers
The Worker Thread converts the Mono AI audio into Stereo (copying Left to Right). It writes this data into `g_rb_output` (The Output Ring Buffer).

### 4. The Final Callback
The next time the hardware interrupts (`data_callback`), it sees data in the Output Ring Buffer. It grabs it and pushes it to your speakers/headphones.

---

## Summary of Data Flow

```mermaid
graph TD
    A[Mic 48kHz] --> B[Callback]
    B --> C[Input RingBuffer]
    C --> D[Worker Thread]
    D --> E[Denoise]
    E --> F[Accumulate 5s]
    F --> G[Resample 40kHz]
    G --> H[TCP Send]
    H --> I[Python Lab]
    
    subgraph Python
    I --> J[Hubert: Extract Words]
    I --> K[Crepe: Extract Pitch]
    K --> L[Pitch Shift]
    J --> M[RVC Model]
    L --> M
    M --> N[New Voice]
    N --> O[Resample 48kHz]
    end
    
    O --> P[TCP Return]
    P --> Q[C++ Worker]
    Q --> R[Elastic Fit]
    R --> S[Output RingBuffer]
    S --> T[Callback]
    T --> U[Speakers]
