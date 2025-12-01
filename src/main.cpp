// /src/main.cpp

#include <iostream>
#include <vector>
#include <thread>   // For std::thread (background worker)
#include <atomic>   // For thread-safe flags (g_running, g_is_recording) without using heavy mutex locks.
#include <cstring>  // For memcpy (fast memory copying)
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <limits>
#include <future>   // For std::async (running the Python task while we animate the spinner)
#include <fstream>  // <--- ADDED: Required for reading files (Todo list)
#include <string>   // <--- ADDED: Required for string manipulation

// Audio Driver Implementation.
// Miniaudio is a "single header library". Defining implementation here compels the compiler
// to build the actual library code inside this file.
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "Denoise.h"
#include "PythonBridge.h"

// --- SETTINGS ---
// Simple struct to hold user configurations.
struct AppConfig {
    float pitch_shift_semitones = 12.0f; // The AI will shift the voice up by 1 octave.
    float recording_duration = 5.0f;     // Default recording length.
} g_config;

// --- STATE ---
// Ring Buffers are circular queues. They allow the audio hardware to "Write" data
// and our processing thread to "Read" data simultaneously without crashing.
AudioRingBuffer g_rb_input;  // Microphone -> RingBuffer
AudioRingBuffer g_rb_output; // RingBuffer -> Speakers

// Atomic flags allow different threads to communicate status instantly.
std::atomic<bool> g_running(true);        // specific signal to kill the worker thread on quit.
std::atomic<bool> g_is_recording(false);  // UI tells Worker: "Start saving audio now".
std::atomic<bool> g_is_processing(false); // Worker tells UI: "I am busy talking to Python".
std::atomic<bool> g_is_replaying(false);  // UI tells Worker: "Don't record, just play the last clip".

// We store the last processed audio here so the user can hit "-1" to hear it again without re-processing.
std::vector<float> g_last_recording;

// The class that handles the TCP socket connection to server.py.
PythonBridge g_bridge;

// Audio Configuration
#define INTERNAL_CHANNELS 2      // We work in Stereo internally.
#define INTERNAL_SAMPLE_RATE 48000 // Standard DVD quality audio.

// --- AUDIO HELPERS ---
// This function converts the hardware audio format (which might be mono or stereo)
// into our internal format. If input is mono, we duplicate it to stereo (L=Input, R=Input).
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pInternalBuffer[i*2]=pInput[i]; pInternalBuffer[i*2+1]=pInput[i]; }
}

// This function converts our internal stereo format back to whatever the hardware speakers expect.
// If speakers are mono, we mix L+R together.
void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pOutput[i]=(pInternalBuffer[i*2]+pInternalBuffer[i*2+1])*0.5f; }
}

// --- AUDIO CALLBACK ---
// CRITICAL: This function is called by the Operating System / Audio Driver.
// It runs at a very high priority. You CANNOT do heavy math or networking here.
// Its only job is to move bytes: Mic -> RingBuffer, and RingBuffer -> Speaker.
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // 1. Capture Microphone Input
    float tempInput[4096 * INTERNAL_CHANNELS];
    convert_to_internal((const float*)pInput, tempInput, frameCount, pDevice->capture.channels);
    g_rb_input.Write(tempInput, frameCount * INTERNAL_CHANNELS);

    // 2. Prepare Speaker Output
    float tempOutput[4096 * INTERNAL_CHANNELS];
    size_t required  = frameCount * INTERNAL_CHANNELS;
    size_t available = g_rb_output.AvailableRead();

    // We can only play what we have available.
    size_t toRead    = (available >= required) ? required : available;

    g_rb_output.Read(tempOutput, toRead);

    ma_uint32 framesRead = toRead / INTERNAL_CHANNELS;
    if (framesRead > 0) convert_from_internal(tempOutput, (float*)pOutput, framesRead, pDevice->playback.channels);

    // If we don't have enough audio to fill the speakers, fill the rest with Silence (0.0f).
    // If we don't do this, the speakers will play "garbage memory" (static noise).
    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i=0;i<remaining;++i) outRaw[offset+i] = 0.0f;
    }
}

// --- WORKER THREAD ---
// This acts as the "Engine Room". It runs in the background, separate from the UI and Audio Driver.
void processing_thread_func() {
    DenoiseEngine denoiser; // Loads the RNNoise library logic.
    const float INPUT_GAIN = 4.0f; // Multiplier to make the mic louder.

    // OPTIMIZATION: We declare these vectors OUTSIDE the while loop.
    // If we put them inside, the computer would have to ask for RAM and delete it 60 times a second.
    // By keeping them here, we reuse the same memory block forever.
    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60); // Pre-allocate enough space for 60 seconds of audio.

    const size_t WORK_SIZE = 1024 * 2; // We process audio in small chunks (frames).
    std::vector<float> input_chunk(WORK_SIZE);

    std::vector<float> mono;
    mono.reserve(WORK_SIZE/2); // Mono is half the size of Stereo.

    while (g_running) {
        // Only process if we have enough data waiting in the ring buffer.
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // 1. REPLAY MODE
            // If user hit "-1", we just dump the saved audio back into the output buffer.
            if (g_is_replaying) {
                if (!g_last_recording.empty()) {
                    std::cout << "\r[REPLAYING]..." << std::flush;
                    g_rb_output.Write(g_last_recording.data(), g_last_recording.size());
                } else {
                    std::cout << "\r[ERROR] No recording." << std::endl;
                }
                g_is_replaying = false;
                g_is_processing = false;
                continue; // Skip the rest of the loop.
            }

            // 2. IDLE DRAIN
            // If we aren't recording, we must still "read" the microphone data and throw it away.
            // If we don't, the ring buffer fills up and old audio gets stuck there.
            if (!g_is_recording) {
                if (!burst_buffer.empty()) burst_buffer.clear();
                continue;
            }

            // 3. RECORDING PROCESSING
            // Convert Stereo (L,R) to Mono (L+R/2).
            // AI models and Denoise algorithms usually expect Mono.
            mono.clear(); // Reset "size" to 0, but keep capacity (fast).
            for (size_t i=0; i < WORK_SIZE/2; ++i) {
                mono.push_back((input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f);
            }

            // Run AI Denoising (RNNoise) to remove background fans/clicks.
            denoiser.Process(mono);

            // Apply Gain and "Soft Clipping"
            // std::tanh is a math function that rounds off loud spikes so they don't crackle.
            for (float &s : mono) { s *= INPUT_GAIN; s = std::tanh(s); }

            // Store this chunk into our big recording buffer.
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            // Update the console progress bar every ~4000 samples.
            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // 4. TRIGGER AI PROCESSING
            // Check if we have recorded enough seconds.
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;
                std::cout << "\n[PROCESSING] Sending to AI..." << std::flush;

                // A. Resample 48k -> 40k
                // The AI model was trained on 40k audio. We must match it.
                auto audio_40k = Resample48To40(burst_buffer);

                // B. Async Python Call
                // We launch a new "async" task to talk to Python.
                // This allows the while-loop below to run and animate the spinner
                // while we wait for the network data.
                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(audio_40k, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // C. Spinner Animation
                // This loops until the network data arrives.
                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx + 1) % 4;
                }
                auto processed = future_result.get(); // Retrieve the data.

                if (!processed.empty()) {
                    // D. Elastic Resample
                    // Sometimes resampling math isn't perfect. We force the output size
                    // to match exactly what we expect so the audio doesn't drift.
                    auto output = ResampleToCount(processed, burst_buffer.size());

                    // Convert Mono back to Stereo for playback.
                    std::vector<float> stereo;
                    stereo.reserve(output.size()*2);
                    for (float s : output) { stereo.push_back(s); stereo.push_back(s); }

                    // Save it to global memory (for Replay).
                    g_last_recording = stereo;
                    // Push it to the speakers.
                    g_rb_output.Write(stereo.data(), stereo.size());
                    std::cout << "\r[PLAYING]           " << std::endl;
                } else {
                    std::cout << "\r[PYTHON ERROR]     " << std::endl;
                }

                burst_buffer.clear(); // Reset buffer for next time.
                g_is_processing = false;
                g_is_recording = false; // Stop recording.
            }
        } else {
            // If ring buffer is empty, sleep briefly to save CPU power.
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    \n";
    std::cout << "========================================\n";

    // 1. Connection Loop
    // We try to connect to server.py. If it fails, we wait 2 seconds and try again.
    while (!g_bridge.Connect()) {
        std::cout << "Waiting for 'server.py'..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // 2. Initialize Ring Buffers
    // We allocate 30 seconds worth of buffer space to be safe.
    g_rb_input.Init(48000 * 30 * INTERNAL_CHANNELS);
    g_rb_output.Init(48000 * 30 * INTERNAL_CHANNELS);

    // 3. Configure Miniaudio Device
    ma_device_config config = ma_device_config_init(ma_device_type_duplex); // Duplex = Mic AND Speakers
    config.sampleRate = INTERNAL_SAMPLE_RATE;
    config.capture.format = ma_format_f32; // Floating point audio is standard for DSP
    config.playback.format = ma_format_f32;
    config.dataCallback = data_callback;   // Tell it which function to call when it has data
    config.periodSizeInFrames = 4096;      // Latency setting (higher = safer, lower = faster)
    config.periods = 3;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return -1;

    // 4. Start the Worker Thread
    std::thread ai_thread(processing_thread_func);
    // 5. Start the Audio Hardware
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    // UI LOOP
    // This loop runs on the main thread and handles User Input only.
    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### enter recording length (seconds):" << std::endl;
        std::cout << "### (-1 to replay previous)" << std::endl;
        std::cout << "### (0 to Quit)" << std::endl;
        std::cout << "> ";

        float input;
        // Check for invalid input (letters instead of numbers)
        if (!(std::cin >> input)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            continue;
        }

        // --- QUIT & TODO LIST FEATURE ---
        if (input == 0) {
            std::cout << "\n----------------------------------------\n";
            std::cout << "               TODO LIST                \n";
            std::cout << "----------------------------------------\n";

            // Try to open 'todo.txt' in current directory
            std::ifstream file("todo.txt");
            if (!file.is_open()) {
                // Fallback: If failed, try one directory up (useful for cmake-build-debug folder structures)
                file.open("../todo.txt");
            }

            if (file.is_open()) {
                // Efficient way to print entire file to console
                std::cout << file.rdbuf() << std::endl;
            } else {
                std::cout << "[ERROR] Could not find 'todo.txt' in current or parent directory." << std::endl;
                std::cout << "Make sure you created the file!" << std::endl;
            }
            std::cout << "----------------------------------------\n";
            std::cout << "Exiting..." << std::endl;
            break; // Break the while(true) loop to exit program
        }

        // --- REPLAY HANDLING ---
        if (input == -1) {
            g_is_replaying = true;
            g_is_processing = true; // Block UI while replaying
        } else {
            // --- NEW RECORDING ---
            g_config.recording_duration = input;
            g_is_recording = true;
        }

        // Wait here while the worker thread does its job.
        // We sleep to keep the UI thread from eating 100% CPU while waiting.
        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Playback delay logic
        // If we just finished a recording, calculate how long the playback is
        // and sleep the UI thread so the prompt doesn't appear until audio finishes playing.
        if (input > 0 && g_last_recording.size() > 0) {
            float duration = static_cast<float>(g_last_recording.size()) / (48000.0f * 2.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration*1000)+500));
        }
    }

    // Cleanup
    g_running = false;            // Tell worker thread to stop
    ai_thread.join();             // Wait for worker thread to finish safely
    ma_device_uninit(&device);    // Shutdown audio hardware
    g_bridge.Disconnect();        // Close network socket
    return 0;
}