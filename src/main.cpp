#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <limits>
#include <future>

// // Include the miniaudio implementation here (only once in the project).
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "Denoise.h"
#include "PythonBridge.h"

// --- GLOBAL SETTINGS ---
struct AppConfig {
    float pitch_shift_semitones = 12.0f; // // Pitch shift (+12 = female).
    float recording_duration = 5.0f;     // // How long to record.
} g_config;

// --- GLOBAL STATE ---
AudioRingBuffer g_rb_input;
AudioRingBuffer g_rb_output;
std::atomic<bool> g_running(true);
std::atomic<bool> g_is_recording(false);
std::atomic<bool> g_is_processing(false);
std::atomic<bool> g_is_replaying(false);

std::atomic<bool> g_do_calibration(false);
std::atomic<bool> g_calibration_done(false);

std::vector<float> g_last_recording; // // Caches audio for replay (-1).

PythonBridge g_bridge;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000

// --- AUDIO HELPERS ---
// // Converts whatever the mic gives us to our internal 2-channel float format.
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pInternalBuffer[i*2]=pInput[i]; pInternalBuffer[i*2+1]=pInput[i]; }
}

// // Converts our internal format to whatever the speakers want.
void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pOutput[i]=(pInternalBuffer[i*2]+pInternalBuffer[i*2+1])*0.5f; }
}

// --- AUDIO CALLBACK ---
// // This runs on the high-priority audio thread. Keep it fast!
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    float tempInput[4096 * INTERNAL_CHANNELS];
    convert_to_internal((const float*)pInput, tempInput, frameCount, pDevice->capture.channels);
    g_rb_input.Write(tempInput, frameCount * INTERNAL_CHANNELS);

    float tempOutput[4096 * INTERNAL_CHANNELS];
    size_t required  = frameCount * INTERNAL_CHANNELS;
    size_t available = g_rb_output.AvailableRead();
    size_t toRead    = (available >= required) ? required : available;

    g_rb_output.Read(tempOutput, toRead);

    ma_uint32 framesRead = toRead / INTERNAL_CHANNELS;
    if (framesRead > 0) convert_from_internal(tempOutput, (float*)pOutput, framesRead, pDevice->playback.channels);

    // // Fill with silence if we run out of data (prevents buzzing).
    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i=0;i<remaining;++i) outRaw[offset+i] = 0.0f;
    }
}

// --- PROCESSING THREAD ---
// // Handles the heavy logic: Denoising, Buffering, sending to Python.
void processing_thread_func() {
    DenoiseEngine denoiser;
    const float INPUT_GAIN = 3.0f; // // Boost mic volume.

    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60);

    const size_t WORK_SIZE = 1024 * 2;
    std::vector<float> input_chunk(WORK_SIZE);

    bool local_calibrating = false;
    int calibration_frames = 0;
    float max_noise_rms = 0.0f;

    while (g_running) {
        // // Check for calibration trigger.
        if (g_do_calibration.load() && !local_calibrating) {
            local_calibrating = true;
            calibration_frames = 0;
            max_noise_rms = 0.0f;
            std::cout << "[CALIBRATION] Starting (~5s: 2s mute + 3s measure)..." << std::endl;
        }

        // // Consume audio from the ring buffer.
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // // Mix to Mono for RNNoise.
            std::vector<float> mono(WORK_SIZE/2);
            for (size_t i=0;i<mono.size();++i) mono[i] = (input_chunk[i*2]+input_chunk[i*2+1])*0.5f;

            // // Run Denoise.
            denoiser.Process(mono);

            // // Apply Gain and Limiter.
            for (float &s : mono) { s *= INPUT_GAIN; s = std::tanh(s); }

            // // Calculate RMS volume.
            float sum = 0.0f;
            for (float s : mono) sum += s*s;
            float rms = std::sqrt(sum / mono.size());

            // // --- CALIBRATION LOGIC ---
            if (local_calibrating) {
                calibration_frames++;
                // // Ignore first 2 seconds to let hardware settle.
                if (calibration_frames > 100) {
                    if (rms > max_noise_rms) max_noise_rms = rms;
                }
                if (calibration_frames % 10 == 0) std::cout << "." << std::flush;
                if (calibration_frames > 250) { // // ~5 seconds total.
                    local_calibrating = false;
                    g_do_calibration.store(false);
                    g_calibration_done.store(true);
                    std::cout << "\n[CALIBRATION DONE] Noise floor: " << max_noise_rms << std::endl;
                }
                continue; // // Skip recording during calibration.
            }

            // // --- REPLAY LOGIC ---
            if (g_is_replaying) {
                if (!g_last_recording.empty()) {
                    g_rb_output.Write(g_last_recording.data(), g_last_recording.size());
                } else {
                    std::cout << "[ERROR] No previous recording." << std::endl;
                }
                g_is_replaying = false;
                g_is_processing = false;
                continue;
            }

            if (!g_is_recording) {
                if (!burst_buffer.empty()) burst_buffer.clear();
                continue;
            }

            // // --- RECORDING LOGIC ---
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s" << std::flush;
            }

            // // --- SEND TO PYTHON ---
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;
                std::cout << "\n[PROCESSING] Sending to AI..." << std::endl;

                // // 1. Resample 48k -> 40k (Correct for the RVC model).
                auto audio_40k = Resample48To40(burst_buffer);

                // // 2. Launch Async task to keep UI responsive.
                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(audio_40k, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // // Spinner Animation.
                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx + 1) % 4;
                }
                auto processed = future_result.get();

                if (!processed.empty()) {
                    // // 3. Elastic Resample: Force output length to match input length exactly.
                    auto output = ResampleToCount(processed, burst_buffer.size());

                    std::vector<float> stereo;
                    stereo.reserve(output.size()*2);
                    for (float s : output) { stereo.push_back(s); stereo.push_back(s); }

                    g_last_recording = stereo;
                    g_rb_output.Write(stereo.data(), stereo.size());
                    std::cout << "\r[PLAYING]           " << std::endl;
                } else {
                    std::cout << "\r[PYTHON ERROR]     " << std::endl;
                }

                burst_buffer.clear();
                g_is_processing = false;
                g_is_recording = false;
            }
        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

int main() {
    std::cout << "========================================\n";
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    \n";
    std::cout << "========================================\n";

    // // Connect to Python first.
    while (!g_bridge.Connect()) {
        std::cout << "Waiting for 'server.py'..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // // Initialize memory.
    g_rb_input.Init(48000 * 30 * INTERNAL_CHANNELS);
    g_rb_output.Init(48000 * 30 * INTERNAL_CHANNELS);

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = INTERNAL_SAMPLE_RATE;
    config.capture.format = ma_format_f32;
    config.playback.format = ma_format_f32;
    config.dataCallback = data_callback;
    config.periodSizeInFrames = 4096;
    config.periods = 3;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return -1;

    std::thread ai_thread(processing_thread_func);
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    std::cout << "\nEnsure room is quiet." << std::endl;
    std::cout << "Press ENTER to start noise-floor calibration..." << std::endl;
    std::cin.ignore(10000, '\n');
    std::string dummy; std::getline(std::cin, dummy);

    // // Trigger calibration.
    g_do_calibration.store(true);
    while (!g_calibration_done.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    // // Main Input Loop.
    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### enter recording length (seconds):" << std::endl;
        std::cout << "### (-1 to replay previous)" << std::endl;
        std::cout << "### (0 to Quit)" << std::endl;
        std::cout << "> ";

        float input;
        if (!(std::cin >> input)) {
            std::cin.clear();
            std::cin.ignore(10000, '\n');
            continue;
        }

        if (input == 0) break;

        if (input == -1) {
            g_is_replaying = true;
            g_is_processing = true;
        } else {
            g_config.recording_duration = input;
            g_is_recording = true;
        }

        // // Wait for work to finish before asking again.
        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // // Wait for playback to finish.
        if (input > 0 && g_last_recording.size() > 0) {
            float duration = static_cast<float>(g_last_recording.size()) / (48000.0f * 2.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration*1000)+500));
        }
    }

    g_running = false;
    ai_thread.join();
    ma_device_uninit(&device);
    g_bridge.Disconnect();
    return 0;
}