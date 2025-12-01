// /src/main.cpp

#include <iostream>
#include <vector>
#include <thread>   // For std::thread (background worker)
#include <atomic>   // For thread-safe flags
#include <cstring>  // For memcpy
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>
#include <limits>
#include <future>   // For std::async
#include <fstream>
#include <string>

// Audio Driver Implementation.
#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "Denoise.h"
#include "PythonBridge.h"

// --- SETTINGS ---
struct AppConfig {
    float pitch_shift_semitones = 12.0f; // +1 Octave
    float recording_duration = 5.0f;
} g_config;

// --- STATE ---
AudioRingBuffer g_rb_input;  // Microphone -> RingBuffer
AudioRingBuffer g_rb_output; // RingBuffer -> Speakers

std::atomic<bool> g_running(true);
std::atomic<bool> g_is_recording(false);
std::atomic<bool> g_is_processing(false);
std::atomic<bool> g_is_replaying(false);

std::vector<float> g_last_recording;

PythonBridge g_bridge;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000

// --- AUDIO HELPERS ---
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pInternalBuffer[i*2]=pInput[i]; pInternalBuffer[i*2+1]=pInput[i]; }
}

void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0;i<frameCount;++i){ pOutput[i]=(pInternalBuffer[i*2]+pInternalBuffer[i*2+1])*0.5f; }
}

// --- AUDIO CALLBACK ---
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    // 1. Capture Microphone Input
    float tempInput[4096 * INTERNAL_CHANNELS];
    convert_to_internal((const float*)pInput, tempInput, frameCount, pDevice->capture.channels);
    g_rb_input.Write(tempInput, frameCount * INTERNAL_CHANNELS);

    // 2. Prepare Speaker Output
    float tempOutput[4096 * INTERNAL_CHANNELS];
    size_t required  = frameCount * INTERNAL_CHANNELS;
    size_t available = g_rb_output.AvailableRead();

    size_t toRead    = (available >= required) ? required : available;

    g_rb_output.Read(tempOutput, toRead);

    ma_uint32 framesRead = toRead / INTERNAL_CHANNELS;
    if (framesRead > 0) convert_from_internal(tempOutput, (float*)pOutput, framesRead, pDevice->playback.channels);

    // Fill silence if needed
    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i=0;i<remaining;++i) outRaw[offset+i] = 0.0f;
    }
}

// --- WORKER THREAD ---
void processing_thread_func() {
    DenoiseEngine denoiser;

    // NOTE: Removed hardcoded INPUT_GAIN and tanh here.

    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60);

    const size_t WORK_SIZE = 1024 * 2;
    std::vector<float> input_chunk(WORK_SIZE);

    std::vector<float> mono;
    mono.reserve(WORK_SIZE/2);

    while (g_running) {
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // 1. REPLAY MODE
            if (g_is_replaying) {
                if (!g_last_recording.empty()) {
                    std::cout << "\r[REPLAYING]..." << std::flush;
                    g_rb_output.Write(g_last_recording.data(), g_last_recording.size());
                } else {
                    std::cout << "\r[ERROR] No recording." << std::endl;
                }
                g_is_replaying = false;
                g_is_processing = false;
                continue;
            }

            // 2. IDLE DRAIN
            if (!g_is_recording) {
                if (!burst_buffer.empty()) burst_buffer.clear();
                continue;
            }

            // 3. RECORDING PROCESSING
            mono.clear();
            for (size_t i=0; i < WORK_SIZE/2; ++i) {
                mono.push_back((input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f);
            }

            // Run AI Denoising (RNNoise)
            denoiser.Process(mono);

            // Store raw, clean audio.
            // We do NOT apply gain or distortion here anymore.
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            // Update progress bar
            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // 4. TRIGGER AI PROCESSING
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;
                std::cout << "\n[PROCESSING] Preparing audio..." << std::flush;

                // --- NEW STEP: PEAK NORMALIZATION ---
                // Scan for the loudest peak
                float max_val = 0.0f;
                for (float s : burst_buffer) {
                    float abs_s = std::abs(s);
                    if (abs_s > max_val) max_val = abs_s;
                }

                // If peak is detected, normalize entire buffer to 0.9 (90% volume)
                // This maximizes volume without 'tanh' distortion.
                if (max_val > 0.001f) {
                    float normalization_factor = 0.9f / max_val;
                    // Cap boost to 5x to prevent amplifying silence to static
                    if (normalization_factor > 5.0f) normalization_factor = 5.0f;

                    for (float &s : burst_buffer) s *= normalization_factor;
                }

                std::cout << "\r[PROCESSING] Sending to AI..." << std::flush;

                // --- CHANGED: SEND RAW 48k AUDIO ---
                // We no longer call Resample48To40 here. We let Python do it.
                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(burst_buffer, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // C. Spinner Animation
                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx + 1) % 4;
                }
                auto processed = future_result.get();

                if (!processed.empty()) {
                    // D. Elastic Resample
                    // We still use this to ensure the playback buffer matches exactly what the speakers expect.
                    auto output = ResampleToCount(processed, burst_buffer.size());

                    // Convert Mono back to Stereo
                    std::vector<float> stereo;
                    stereo.reserve(output.size()*2);
                    for (float s : output) { stereo.push_back(s); stereo.push_back(s); }

                    // Save and Play
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

    while (!g_bridge.Connect()) {
        std::cout << "Waiting for 'server.py'..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

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

    // UI LOOP
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

        if (input == 0) {
            std::cout << "\n----------------------------------------\n";
            std::cout << "               TODO LIST                \n";
            std::cout << "----------------------------------------\n";

            std::ifstream file("todo.txt");
            if (!file.is_open()) file.open("../todo.txt");

            if (file.is_open()) {
                std::cout << file.rdbuf() << std::endl;
            } else {
                std::cout << "[ERROR] Could not find 'todo.txt'." << std::endl;
            }
            std::cout << "----------------------------------------\n";
            std::cout << "Exiting..." << std::endl;
            break;
        }

        if (input == -1) {
            g_is_replaying = true;
            g_is_processing = true;
        } else {
            g_config.recording_duration = input;
            g_is_recording = true;
        }

        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

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