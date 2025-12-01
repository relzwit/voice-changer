// /src/main.cpp

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
#include <fstream>
#include <string>
#include <sstream> // For parsing strings

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "PythonBridge.h"

// --- CONFIGURATION MANAGEMENT ---
// This simple function reads key/value pairs from the INI file
void load_config(float& pitch_shift, float& duration) {
    std::ifstream file("config.ini");
    if (!file.is_open()) {
        std::cerr << "[WARN] config.ini not found. Using defaults." << std::endl;
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '[' || line[0] == ';') continue;

        std::size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) continue;

        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);

        // Simple trim (basic implementation)
        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "PITCH_SHIFT_SEMITONES") {
            try {
                pitch_shift = std::stof(value);
            } catch (...) { std::cerr << "[WARN] Invalid PITCH_SHIFT value in config." << std::endl; }
        } else if (key == "RECORDING_DURATION_SECONDS") {
            try {
                duration = std::stof(value);
            } catch (...) { std::cerr << "[WARN] Invalid DURATION value in config." << std::endl; }
        }
    }
}
// --- END CONFIGURATION MANAGEMENT ---

// --- SETTINGS ---
struct AppConfig {
    float pitch_shift_semitones = 12.0f; // Default if config fails
    float recording_duration = 5.0f;     // Default if config fails
} g_config;

// --- STATE ---
AudioRingBuffer g_rb_input;
AudioRingBuffer g_rb_output;

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

    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i=0;i<remaining;++i) outRaw[offset+i] = 0.0f;
    }
}

// --- WORKER THREAD ---
void processing_thread_func() {
    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60);

    const size_t WORK_SIZE = 1024 * 2;
    std::vector<float> input_chunk(WORK_SIZE);
    std::vector<float> mono;
    mono.reserve(WORK_SIZE/2);

    while (g_running) {
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

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

            if (!g_is_recording) {
                if (!burst_buffer.empty()) burst_buffer.clear();
                continue;
            }

            // Convert Stereo to Mono
            mono.clear();
            for (size_t i=0; i < WORK_SIZE/2; ++i) {
                mono.push_back((input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f);
            }

            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // TRIGGER
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;
                std::cout << "\n[PROCESSING] Preparing audio..." << std::flush;

                // --- PEAK NORMALIZATION ---
                float max_val = 0.0f;
                for (float s : burst_buffer) {
                    float abs_s = std::abs(s);
                    if (abs_s > max_val) max_val = abs_s;
                }

                if (max_val > 0.001f) {
                    float normalization_factor = 0.9f / max_val;
                    if (normalization_factor > 5.0f) normalization_factor = 5.0f;
                    for (float &s : burst_buffer) s *= normalization_factor;
                }

                // --- DEBUG MONITOR: PLAY RAW AUDIO BEFORE SENDING (Removed for simplicity, keep just log) ---
                std::cout << "\n[PROCESSING] Sending " << burst_buffer.size() << " samples to AI..." << std::flush;

                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(burst_buffer, static_cast<int>(g_config.pitch_shift_semitones));
                });

                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx + 1) % 4;
                }
                auto processed = future_result.get();

                if (!processed.empty()) {
                    // Python already resampled to 48kHz, so just convert mono->stereo
                    std::vector<float> stereo;
                    stereo.reserve(processed.size()*2);
                    for (float s : processed) {
                        stereo.push_back(s);
                        stereo.push_back(s);
                    }

                    g_last_recording = stereo;
                    g_rb_output.Write(stereo.data(), stereo.size());
                    std::cout << "\r[PLAYING AI RESULT]     " << std::endl;
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
    // Load configuration from config.ini BEFORE initializing anything
    load_config(g_config.pitch_shift_semitones, g_config.recording_duration);

    std::cout << "========================================\n";
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    \n";
    std::cout << "========================================\n";
    std::cout << "[INFO] Shift loaded: " << g_config.pitch_shift_semitones << " semitones." << std::endl;

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

    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### enter recording length (s) (-1 to replay, 0 to Quit): ";
        float input;
        if (!(std::cin >> input)) { std::cin.clear(); std::cin.ignore(10000, '\n'); continue; }

        if (input == 0) break;
        if (input == -1) { g_is_replaying = true; g_is_processing = true; }
        else {
            // In a real app, you would ask for duration or check config here.
            // For now, we take user input, but rely on loaded duration if no input is given.
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