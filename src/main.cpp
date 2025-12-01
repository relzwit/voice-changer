// /src/main.cpp
//
// This file is the main C++ client application for a Real-time Voice Changer (RVC)
// system, which uses an AI model (likely accessed via Python) to process audio.
// It manages audio I/O using miniaudio, threading, configuration, and user interaction.

#include "indicators/indeterminate_progress_bar.hpp" // Header for a library to display terminal progress bars (for server connection).
#include "indicators/cursor_control.hpp"            // Utility for hiding/showing the console cursor during progress updates.
#include "indicators/progress_bar.hpp"              // Standard progress bar definition (not explicitly used in final UI, but included).
#include "indicators/block_progress_bar.hpp"        // Block-style progress bar (used for the recording bar, though commented out later).

#include <iostream>     // Standard I/O operations (cin, cout).
#include <cstdlib>      // General utilities (e.g., exit, though not used heavily).
#include <vector>       // Standard dynamic array (used heavily for audio buffers).
#include <thread>       // Standard threading for parallel processing.
#include <atomic>       // Thread-safe boolean flags for application state.
#include <cstring>      // For memory manipulation functions like memcpy.
#include <cmath>        // For mathematical functions (e.g., std::abs for audio normalization).
#include <algorithm>    // For min/max/abs functions (used in audio processing and normalization).
#include <iomanip>      // For I/O manipulators (e.g., std::setprecision for displaying time).
#include <chrono>       // For time-related operations and thread sleep.
#include <limits>       // Standard numeric limits (not explicitly used, but included).
#include <future>       // For std::async and std::future to run the AI processing asynchronously.
#include <fstream>      // For file stream operations (config reading, WAV saving).
#include <string>       // Standard string operations.
#include <sstream>      // String stream operations (used for timestamped filename generation).
#include <ctime>        // Time utilities (used for getting the current time for saving files).

// --- MINIAUDIO INTEGRATION ---
#define MINIAUDIO_IMPLEMENTATION // Must be defined in one compilation unit to include implementation.
#include <miniaudio.h>           // The cross-platform audio I/O library.

#include "Utils.h"       // Likely contains the AudioRingBuffer definition and other utility functions.
#include "PythonBridge.h" // Class responsible for communicating with the Python AI server.

// --- CONFIGURATION MANAGEMENT ---
// Function to load settings (pitch shift and recording duration) from a config.ini file.
void load_config(float& pitch_shift, float& duration) {
    std::ifstream file("config.ini"); // Open the configuration file.
    if (!file.is_open()) return;      // If the file cannot be opened, use default settings and exit.

    std::string line; // Variable to hold the current line read from the file.
    while (std::getline(file, line)) { // Read the file line by line.
        if (line.empty() || line[0] == '[' || line[0] == ';') continue; // Skip empty lines, section headers, and comments.

        std::size_t equalPos = line.find('='); // Find the position of the '=' sign.
        if (equalPos == std::string::npos) continue; // Skip lines without an '=' sign.

        std::string key = line.substr(0, equalPos);      // Extract the key (before '=').
        std::string value = line.substr(equalPos + 1);   // Extract the value (after '=').

        // Trim leading and trailing whitespace from the key.
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        // Trim leading and trailing whitespace from the value.
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "PITCH_SHIFT_SEMITONES") {
            try { pitch_shift = std::stof(value); } catch (...) {} // Convert value to float for pitch shift. Handle conversion errors gracefully.
        } else if (key == "RECORDING_DURATION_SECONDS") {
            try { duration = std::stof(value); } catch (...) {} // Convert value to float for recording duration. Handle conversion errors gracefully.
        }
    }
}

// --- SETTINGS ---
// Structure to hold the global application configuration.
struct AppConfig {
    float pitch_shift_semitones = 12.0f; // Default pitch shift to one octave (12 semitones).
    float recording_duration = 5.0f;     // Default recording duration of 5 seconds.
} g_config; // Global instance of the configuration.

// --- STATE ---
// Global audio ring buffer for incoming audio data (microphone -> processing thread).
AudioRingBuffer g_rb_input;
// Global audio ring buffer for outgoing audio data (processing thread -> speakers).
AudioRingBuffer g_rb_output;

std::atomic<bool> g_running(true);        // Global flag to keep the main loops/threads running. Set to false on quit ('q').
std::atomic<bool> g_is_recording(false);  // Flag indicating if the application is currently accumulating audio for recording.
std::atomic<bool> g_is_processing(false); // Flag indicating if the audio is currently being sent to/processed by the AI.
std::atomic<bool> g_is_replaying(false);  // Flag indicating a request to replay the last processed audio.

std::vector<float> g_last_recording; // Stores the last processed stereo audio clip for playback or saving.

PythonBridge g_bridge; // Global instance of the Python bridge for AI communication.

#define INTERNAL_CHANNELS 2      // Internal processing is done in stereo (2 channels).
#define INTERNAL_SAMPLE_RATE 48000 // Internal standard sample rate (48 kHz).

// --- WAV FILE UTILITY ---
// Function to save a raw audio buffer (vector of floats) to a standard 16-bit WAV file.
bool SaveWavFile(const std::vector<float>& buffer, const std::string& filename, int channels, int sampleRate) {
    if (buffer.empty()) return false; // Fail if the buffer is empty.
    std::ofstream file(filename, std::ios::binary); // Open file in binary mode.
    if (!file.is_open()) return false; // Fail if file couldn't be opened.

    // WAV File Header Calculations (Standard 16-bit PCM format)
    int bitsPerSample = 16;
    long byteRate = sampleRate * channels * (bitsPerSample / 8);
    long blockAlign = channels * (bitsPerSample / 8);
    long dataSize = buffer.size() * (bitsPerSample / 8); // Size of the raw audio data in bytes.
    long chunkSize = 36 + dataSize; // Total file size minus "RIFF" and "chunkSize".

    // RIFF Chunk
    file.write("RIFF", 4);
    file.write((char*)&chunkSize, 4);
    file.write("WAVE", 4);

    // fmt Sub-chunk
    file.write("fmt ", 4);
    int subChunk1Size = 16;
    file.write((char*)&subChunk1Size, 4);
    short audioFormat = 1; // 1 for PCM.
    file.write((char*)&audioFormat, 2);
    short numChannels = (short)channels;
    file.write((char*)&numChannels, 2);
    file.write((char*)&sampleRate, 4);
    file.write((char*)&byteRate, 4);
    file.write((char*)&blockAlign, 2);
    short bps = (short)bitsPerSample;
    file.write((char*)&bps, 2);

    // data Sub-chunk
    file.write("data", 4);
    file.write((char*)&dataSize, 4);

    // Convert float samples (-1.0 to 1.0) to 16-bit short samples (-32768 to 32767).
    std::vector<short> shortBuffer(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, buffer[i])); // Clip to [-1.0, 1.0].
        shortBuffer[i] = (short)(sample * 32767.0f); // Scale and cast to short.
    }

    // Write the actual audio data.
    file.write((char*)shortBuffer.data(), dataSize);
    return true; // Return success.
}

// --- AUDIO HELPERS ---
// Function to convert hardware-specific input format to internal stereo float format.
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float)); // If input is stereo, just copy the raw data.
    // If input is mono, copy the sample to both L/R channels to create stereo.
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pInternalBuffer[i*2]=pInput[i]; pInternalBuffer[i*2+1]=pInput[i]; }
}

// Function to convert internal stereo float format to hardware-specific output format.
void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float)); // If output is stereo, just copy the raw data.
    // If output is mono, average the internal L/R channels before copying.
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pOutput[i]=(pInternalBuffer[i*2]+pInternalBuffer[i*2+1])*0.5f; }
}

// --- AUDIO CALLBACK ---
// This is the critical, time-sensitive function called by the miniaudio device thread.
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // CAPTURE (Microphone Input)
    float tempInput[4096 * INTERNAL_CHANNELS]; // Temporary buffer for converting input.
    // Convert incoming hardware audio to the internal stereo format.
    convert_to_internal((const float*)pInput, tempInput, frameCount, pDevice->capture.channels);
    // Write the captured data to the input ring buffer (g_rb_input).
    g_rb_input.Write(tempInput, frameCount * INTERNAL_CHANNELS);

    // PLAYBACK (Speaker Output)
    float tempOutput[4096 * INTERNAL_CHANNELS]; // Temporary buffer for holding output data read from ring buffer.
    size_t required  = frameCount * INTERNAL_CHANNELS; // Total samples needed for this frame.
    size_t available = g_rb_output.AvailableRead();    // Total samples available in the output ring buffer.
    size_t toRead    = (available >= required) ? required : available; // Read only what's available, up to what's required.

    // Read the processed data from the output ring buffer (g_rb_output).
    g_rb_output.Read(tempOutput, toRead);

    ma_uint32 framesRead = toRead / INTERNAL_CHANNELS; // Convert sample count back to frame count.
    // Convert the data read from the internal format to the hardware's playback format.
    if (framesRead > 0) convert_from_internal(tempOutput, (float*)pOutput, framesRead, pDevice->playback.channels);

    // Handle underflow (not enough data in ring buffer for playback).
    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels; // Remaining samples to fill with silence.
        ma_uint32 offset    = framesRead * pDevice->playback.channels;             // Starting position for silence.
        float* outRaw = (float*)pOutput; // Pointer to the raw output buffer.
        // Fill the remainder of the output buffer with silence (0.0f).
        for (ma_uint32 i=0;i<remaining;++i) outRaw[offset+i] = 0.0f;
    }
}

// --- WORKER THREAD ---
// The main processing loop, executed in a separate thread.
void processing_thread_func() {
    std::vector<float> burst_buffer;    // Accumulates mono audio data for the AI model's input.
    burst_buffer.reserve(48000 * 60); // Reserve space for 60 seconds of mono audio.
    const size_t WORK_SIZE = 1024 * 2; // The size of chunks (in samples) to read from the ring buffer.
    std::vector<float> input_chunk(WORK_SIZE); // Temporary buffer for reading stereo data from g_rb_input.
    std::vector<float> mono;                   // Temporary buffer to hold the stereo input converted to mono.
    mono.reserve(WORK_SIZE/2);                 // Reserve space for half of WORK_SIZE (since it's mono).

    while (g_running) { // Main loop runs as long as the application is active.
        // Check if there's enough data in the input buffer to process a chunk.
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            // Read a chunk of stereo data from the input ring buffer.
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // Special case: Handle the replay command (-1).
            if (g_is_replaying && !g_last_recording.empty()) {
                std::cout << "\r[REPLAYING]..." << std::flush;
                g_rb_output.Write(g_last_recording.data(), g_last_recording.size()); // Write the last processed clip to output buffer.
                g_is_replaying = false; // Reset replay flag.
                g_is_processing = false; // Reset processing flag (playback is now handled by callback).
                continue; // Skip the rest of the loop and start the next iteration.
            }

            // If not recording, clear the accumulation buffer and discard the new chunk.
            if (!g_is_recording) {
                burst_buffer.clear();
                continue;
            }

            // Convert the stereo input chunk into mono for the AI model.
            mono.clear();
            for (size_t i=0; i<WORK_SIZE/2; ++i) mono.push_back((input_chunk[i*2]+input_chunk[i*2+1])*0.5f);
            // Append the new mono chunk to the main accumulation buffer.
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            // Provide real-time feedback on recording progress (every 4096 samples/0.085s).
            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size()/INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // Check if the recording duration target has been reached.
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true; // Set processing flag to block new recordings/replays.
                std::cout << "\n[PROCESSING] Sending audio to AI..." << std::flush;

                // --- AUDIO NORMALIZATION ---
                float max_val = 0.0f; // Find the peak amplitude.
                for (float s : burst_buffer) max_val = std::max(max_val, std::abs(s));
                if (max_val > 0.001f) {
                    float factor = std::min(0.9f/max_val, 5.0f); // Calculate scaling factor to bring peak to 0.9, capped at 5x gain.
                    for (float &s : burst_buffer) s *= factor; // Apply the gain.
                }
                // --- END NORMALIZATION ---

                // Launch the AI processing in a separate asynchronous task.
                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(burst_buffer, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // Display a spinning cursor while waiting for the AI result (non-blocking wait).
                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] Sending audio to AI... " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx+1)%4;
                }

                // Retrieve the processed result once the AI task is complete.
                auto processed = future_result.get();
                if (!processed.empty()) {
                    // Convert the mono processed audio back to stereo for playback.
                    std::vector<float> stereo;
                    stereo.reserve(processed.size()*2);
                    for (float s : processed) { stereo.push_back(s); stereo.push_back(s); }

                    g_last_recording = stereo; // Store the stereo processed result.
                    g_rb_output.Write(stereo.data(), stereo.size()); // Write to the output ring buffer for playback.
                    std::cout << "\r[PROCESSING] Sending audio to AI... [DONE]" << std::endl;
                    std::cout << "[PLAYING AI RESULT]" << std::endl;
                } else {
                    std::cout << "\r[PROCESSING] Sending audio to AI... [ERROR]" << std::endl;
                }

                burst_buffer.clear();      // Clear the accumulation buffer for the next recording.
                g_is_processing = false;   // Processing is complete.
                g_is_recording = false;    // Recording is complete.
            }
        } else std::this_thread::sleep_for(std::chrono::milliseconds(1)); // Wait briefly if no data is available to prevent busy-waiting.
    }
}

// --- MAIN FUNCTION ---
int main() {
    load_config(g_config.pitch_shift_semitones, g_config.recording_duration); // Load pitch shift and duration from config.ini.

    // Print initial header and configuration settings.
    std::cout << "========================================\n";
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    \n";
    std::cout << "========================================\n";
    std::cout << "[Client] Pitch shift: " << g_config.pitch_shift_semitones << " semitones." << std::endl;


    // --- PROGRESS BAR INITIALIZATION ---
    using namespace indicators; // Use the indicators library namespace.
    // Progress bar for the server connection step.
    IndeterminateProgressBar server_bar{
        option::BarWidth{30},
        option::Start{"["},
        option::Fill{"·"},
        option::Lead{"<==>"},
        option::End{"]"},
        option::PostfixText{},
        option::PrefixText(),
        option::ForegroundColor{indicators::Color::yellow},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };

    // Block progress bar (initialized but not actively used in the main loop UI).
    BlockProgressBar recording_bar{
        option::BarWidth{80},
        option::Start{"["},
        option::End{"]"},
        option::ForegroundColor{Color::white}  ,
        option::ShowPercentage{true},
        option::FontStyles{std::vector<FontStyle>{FontStyle::bold}}
    };
    // --- WAIT FOR PYTHON SERVER ---

    show_console_cursor(false); // Hide the cursor while the progress bar is active.
    server_bar.set_option(option::PrefixText("Waiting for Server"));

    // Loop until the Python server connection is successful.
    while (!g_bridge.Connect()) {
        server_bar.tick(); // Advance the indeterminate progress bar.
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Wait 100ms.
    }
    server_bar.mark_as_completed(); // Mark the bar as complete.
    // Print success message in bold green.
    std::cout << termcolor::bold << termcolor::green
            << "Server connected!\n" << termcolor::reset;

    show_console_cursor(true); // Restore the cursor for user input.

// --------------------------------------

    // Initialize the input ring buffer (30 seconds capacity).
    g_rb_input.Init(48000*30*INTERNAL_CHANNELS);
    // Initialize the output ring buffer (30 seconds capacity).
    g_rb_output.Init(48000*30*INTERNAL_CHANNELS);

    // Initialize miniaudio device configuration for duplex (simultaneous capture and playback).
    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = INTERNAL_SAMPLE_RATE;
    config.capture.format = ma_format_f32; // Set capture format to 32-bit float.
    config.playback.format = ma_format_f32; // Set playback format to 32-bit float.
    config.dataCallback = data_callback; // Assign the audio processing callback function.
    config.periodSizeInFrames = 4096; // Set the chunk size for the callback.
    config.periods = 3; // Number of periods (buffering factor).

    ma_device device;
    // Initialize the audio device with the configured settings.
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return -1;

    // Start the worker thread responsible for AI processing and recording logic.
    std::thread ai_thread(processing_thread_func);
    // Start the miniaudio device (this begins calling data_callback).
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    std::string input_line;
    // Main command loop for user interaction.
    while (true) {
        // Display user prompt and commands.
        std::cout << "\n----------------------------------------\n";
        std::cout << "### Enter command or seconds to record:\n";
        std::cout << "### (-1: Replay, 's': Save Last Clip, 'q': Quit)\n> ";

        // Read user input. Continue if input is empty.
        if (!std::getline(std::cin, input_line) || input_line.empty()) continue;

        if (input_line=="q") break; // Exit the main loop if 'q' is entered.

        if (input_line=="s") {
            // Logic for saving the last processed clip.
            if (g_last_recording.empty()) { std::cout << "[ERROR] No audio clip to save." << std::endl; continue; }

            std::string user_filename;
            std::cout << "Enter filename (without extension): ";
            // Get user-specified filename or generate a timestamped one.
            if (!std::getline(std::cin, user_filename) || user_filename.empty()) {
                std::time_t t = std::time(nullptr);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
                user_filename = oss.str();
            }

            // Construct the full save path.
            std::string dir = "/home/relz/code/voice-changer/saved_outputs"; // Hardcoded save directory.
            std::string full_path = dir + "/" + user_filename + ".wav";

            // Attempt to save the WAV file and report the result.
            if (SaveWavFile(g_last_recording, full_path, 2, INTERNAL_SAMPLE_RATE))
                std::cout << "[SUCCESS] Saved to " << full_path << std::endl;
            else
                std::cout << "[ERROR] Failed to save file." << std::endl;
            continue;
        }

        // Try to convert the input to a float (seconds to record or -1 for replay).
        float input_float = 0.0f;
        try { input_float = std::stof(input_line); } catch (...) {
            std::cout << "[ERROR] Invalid input. Enter seconds, 's', or 'q'." << std::endl;
            continue;
        }

        // Handle the replay command.
        if (input_float == -1) { g_is_replaying = true; g_is_processing = true; } // Set replay flag. g_is_processing blocks new recordings.
        // Handle the record command.
        else { g_config.recording_duration = input_float; g_is_recording = true; } // Set new duration and start recording.

        // Block the main thread, waiting for recording and processing to complete in the worker thread.
        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // After processing, wait for the duration of the processed audio plus a small buffer to finish playback.
        if (input_float>0 && !g_last_recording.empty()) {
            float duration = static_cast<float>(g_last_recording.size()) / (48000.0f*2.0f); // Calculate audio duration (samples / (SR * channels)).
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration*1000)+500)); // Wait for duration + 0.5s.
        }
    }

    // --- CLEANUP AND SHUTDOWN ---
    g_running = false; // Signal the worker thread to stop.
    ai_thread.join(); // Wait for the worker thread to safely exit.
    g_bridge.Disconnect(); // Close the connection to the Python AI server.
    ma_device_uninit(&device); // Uninitialize the miniaudio device.

    std::cout << "[Client] Shutdown complete. Exiting." << std::endl;
    return 0;
}