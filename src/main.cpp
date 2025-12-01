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

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "Denoise.h"
#include "PythonBridge.h"

// --- GLOBAL SETTINGS ---
struct AppConfig {
    // // Default is +12 (1 Octave), which is standard for Male -> Female.
    float pitch_shift_semitones = 12.0f;
    float recording_duration = 5.0f;
} g_config;

// --- STATE ---
AudioRingBuffer g_rb_input;
AudioRingBuffer g_rb_output;
std::atomic<bool> g_running(true);
std::atomic<bool> g_is_recording(false);
std::atomic<bool> g_is_processing(false);
std::atomic<bool> g_is_replaying(false);

// // Cache for the "-1" replay feature
std::vector<float> g_last_recording;

PythonBridge g_bridge;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000

// --- AUDIO HELPERS ---
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pInternalBuffer[i*2] = pInput[i]; pInternalBuffer[i*2+1] = pInput[i]; }
}

void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pOutput[i] = (pInternalBuffer[i*2] + pInternalBuffer[i*2+1]) * 0.5f; }
}

// --- AUDIO CALLBACK ---
void data_callback(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    float tempInput[4096 * INTERNAL_CHANNELS];
    convert_to_internal((const float*)pInput, tempInput, frameCount, pDevice->capture.channels);
    g_rb_input.Write(tempInput, frameCount * INTERNAL_CHANNELS);

    float tempOutput[4096 * INTERNAL_CHANNELS];
    size_t required = frameCount * INTERNAL_CHANNELS;
    size_t available = g_rb_output.AvailableRead();
    size_t toRead = (available >= required) ? required : available;

    g_rb_output.Read(tempOutput, toRead);

    ma_uint32 framesRead = toRead / INTERNAL_CHANNELS;
    if (framesRead > 0) convert_from_internal(tempOutput, (float*)pOutput, framesRead, pDevice->playback.channels);

    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i=0; i<remaining; ++i) outRaw[offset+i] = 0.0f;
    }
}

// --- PROCESSING THREAD ---
void processing_thread_func() {
    DenoiseEngine denoiser;
    const float INPUT_GAIN = 4.0f;

    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60);

    const size_t WORK_SIZE = 1024 * 2;
    std::vector<float> input_chunk(WORK_SIZE);

    while (g_running) {
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // --- REPLAY LOGIC ---
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

            // --- RECORDING LOGIC ---
            std::vector<float> mono(WORK_SIZE / 2);
            for(size_t i=0; i<mono.size(); ++i) mono[i] = (input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f;

            denoiser.Process(mono);
            for(float& s : mono) { s *= INPUT_GAIN; s = std::tanh(s); }

            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            // Progress UI
            if (burst_buffer.size() % 8192 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // --- TRIGGER PROCESSING ---
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;

                std::cout << "\n[PROCESSING] Sending..." << std::flush;

                // 1. Resample 48k -> 40k (Match Model)
                auto audio_40k = Resample48To40(burst_buffer);

                // 2. Async Python Call
                auto future = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(audio_40k, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // Spinner
                const char spinner[] = {'|', '/', '-', '\\'};
                int idx = 0;
                while (future.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[idx] << std::flush;
                    idx = (idx + 1) % 4;
                }

                auto processed = future.get();

                if (!processed.empty()) {
                    // 3. Elastic Resample (40k back to original length)
                    auto output = ResampleToCount(processed, burst_buffer.size());

                    std::vector<float> stereo;
                    stereo.reserve(output.size()*2);
                    for(float s : output) { stereo.push_back(s); stereo.push_back(s); }

                    g_last_recording = stereo;
                    g_rb_output.Write(stereo.data(), stereo.size());
                    std::cout << "\r[PLAYING]           " << std::endl;
                } else {
                    std::cout << "\r[PYTHON ERROR]     " << std::endl;
                }

                burst_buffer.clear();
                g_is_recording = false;
                g_is_processing = false;
            }
        }
        else {
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

    g_rb_input.Init(48000 * 60 * INTERNAL_CHANNELS);
    g_rb_output.Init(48000 * 60 * INTERNAL_CHANNELS);

    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = INTERNAL_SAMPLE_RATE;
    config.capture.format = ma_format_f32;
    config.playback.format = ma_format_f32;
    config.dataCallback = data_callback;
    config.periodSizeInFrames = 4096;
    config.periods = 3;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) return -1;
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    std::thread ai_thread(processing_thread_func);

    // --- UI LOOP ---
    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### enter recording length (seconds):" << std::endl;
        std::cout << "### (-1 to replay previous)" << std::endl;
        std::cout << "### (0 to Quit)" << std::endl;
        std::cout << "> ";

        float input;
        // Safe Input Reading
        if (!(std::cin >> input)) {
            std::cin.clear();
            std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            continue;
        }

        if (input == 0) break;

        if (input == -1) {
            g_is_replaying = true;
            g_is_processing = true;
        }
        else {
            // Reverted default check to standard logic
            if (input < 0 && input > -99) {
                g_config.pitch_shift_semitones = input;
                std::cout << "Pitch shift set to " << input << " semitones." << std::endl;
                continue;
            }

            g_config.recording_duration = input;
            g_is_recording = true;
        }

        while (g_is_recording || g_is_processing) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (input > 0 && g_last_recording.size() > 0) {
             float duration = (float)g_last_recording.size() / (48000.0f * 2.0f);
             std::this_thread::sleep_for(std::chrono::milliseconds((int)(duration * 1000) + 500));
        }
    }

    g_running = false;
    ai_thread.join();
    ma_device_uninit(&device);
    g_bridge.Disconnect();
    return 0;
}