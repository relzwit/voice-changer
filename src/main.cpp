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
    float pitch_shift_semitones = 12.0f; // +12 semitones = 1 octave
    float recording_duration = 5.0f;
} g_config;

// --- STATE FLAGS AND BUFFERS ---
AudioRingBuffer g_rb_input;
AudioRingBuffer g_rb_output;
std::atomic<bool> g_running(true);
std::atomic<bool> g_is_recording(false);
std::atomic<bool> g_is_processing(false);
std::atomic<bool> g_is_replaying(false);

std::vector<float> g_last_recording;

PythonBridge g_bridge;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000 // keep system at 48k (safer for RNNoise/hardware)

// --- AUDIO HELPERS (Required by data_callback) ---
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pInternalBuffer[i*2] = pInput[i]; pInternalBuffer[i*2+1] = pInput[i]; }
}

void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pOutput[i] = (pInternalBuffer[i*2] + pInternalBuffer[i*2+1]) * 0.5f; }
}

// --- AUDIO DRIVER CALLBACK (The Fast Thread) ---
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

    // Silence Fill
    if (framesRead < frameCount) {
        ma_uint32 remaining = (frameCount - framesRead) * pDevice->playback.channels;
        ma_uint32 offset    = framesRead * pDevice->playback.channels;
        float* outRaw = (float*)pOutput;
        for (ma_uint32 i = 0; i < remaining; ++i) outRaw[offset + i] = 0.0f;
    }
}

// --- PROCESSING THREAD (Slow Thread) ---
void processing_thread_func() {
    DenoiseEngine denoiser;
    const float INPUT_GAIN = 4.0f;

    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 60);

    const size_t WORK_SIZE = 1024 * 2;
    std::vector<float> input_chunk(WORK_SIZE);

    // Auto-Calibration Vars
    bool is_calibrating = true;
    int calibration_frames = 0;
    float max_noise_rms = 0.0f;

    while (g_running) {
        if (g_rb_input.AvailableRead() >= WORK_SIZE) {
            g_rb_input.Read(input_chunk.data(), WORK_SIZE);

            // 1. Mono Mix + Denoise (denoiser runs at 48k in this pipeline)
            std::vector<float> mono(WORK_SIZE / 2);
            for(size_t i=0; i<mono.size(); ++i) mono[i] = (input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f;
            denoiser.Process(mono);

            // 2. Gain & Soft Limit
            for(float& s : mono) { s *= INPUT_GAIN; s = std::tanh(s); }

            // 3. RMS Calculation (Calibration/Gate)
            float sum = 0.0f;
            for(float s : mono) sum += s*s;
            float rms = std::sqrt(sum / mono.size());

            // --- CALIBRATION ---
            if (is_calibrating) {
                calibration_frames++;
                if (calibration_frames > 50) {
                    if (rms > max_noise_rms) max_noise_rms = rms;
                    if (calibration_frames % 10 == 0) std::cout << "." << std::flush;
                }
                if (calibration_frames > 150) {
                    is_calibrating = false;
                    float GATE_THRESH = max_noise_rms + 0.015f;
                    std::cout << "\n[CALIBRATION DONE] Gate: " << GATE_THRESH << std::endl;
                    std::cout << "SYSTEM ONLINE. Speak now." << std::endl;
                }
                continue;
            }

            // --- REPLAY LOGIC ---
            if (g_is_replaying) {
                if (!g_last_recording.empty()) {
                    std::cout << "\r[REPLAYING CACHED AUDIO]" << std::flush;
                    g_rb_output.Write(g_last_recording.data(), g_last_recording.size());
                } else {
                    std::cout << "\r[ERROR] No previous recording." << std::endl;
                }
                g_is_replaying = false;
                g_is_processing = false;
                continue;
            }

            // If not recording, discard
            if (!g_is_recording) {
                if (!burst_buffer.empty()) burst_buffer.clear();
                continue;
            }

            // --- RECORDING LOGIC ---
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            // UI Update
            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size() / INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            // Check Duration & Trigger Processing
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;

                // ASYNC TASK: Resample 48k -> 40k then call Python
                auto future_result = std::async(std::launch::async, [&]() {
                    auto audio_40k = Resample48To40(burst_buffer); // <-- important: 40k for the RVC model
                    return g_bridge.ProcessAudio(audio_40k, static_cast<int>(g_config.pitch_shift_semitones));
                });

                // Spinner UI
                const char spinner[] = {'|', '/', '-', '\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] " << spinner[spin_idx] << "   " << std::flush;
                    spin_idx = (spin_idx + 1) % 4;
                }

                auto processed = future_result.get(); // processed is expected to be 48k float samples (see server.py)

                if (!processed.empty()) {
                    // Final fix: Elastic resample to match original sample count exactly.
                    auto output = ResampleToCount(processed, burst_buffer.size());

                    std::vector<float> stereo;
                    stereo.reserve(output.size()*2);
                    for(float s : output) { stereo.push_back(s); stereo.push_back(s); }

                    g_last_recording = stereo;
                    g_rb_output.Write(stereo.data(), stereo.size());

                    std::cout << "\r[PLAYING]           " << std::endl;
                } else {
                    std::cout << "\r[PYTHON ERROR]" << std::endl;
                }

                burst_buffer.clear();
                g_is_processing = false;
                g_is_recording = false;
            }
        }
        else {
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    " << std::endl;
    std::cout << "========================================" << std::endl;

    // 1. CONNECT TO PYTHON
    std::cout << "Launching Python Bridge... Ensure 'server.py' is running!" << std::endl;
    while (!g_bridge.Connect()) {
        std::cout << "Retrying connection in 2s..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
    }

    // 2. INITIALIZE BUFFERS
    g_rb_input.Init(48000 * 30 * INTERNAL_CHANNELS);
    g_rb_output.Init(48000 * 30 * INTERNAL_CHANNELS);

    // 3. CONFIGURE MINIAUDIO DEVICE (keep device at 48k)
    ma_device_config config = ma_device_config_init(ma_device_type_duplex);
    config.sampleRate = INTERNAL_SAMPLE_RATE;
    config.capture.format = ma_format_f32;
    config.playback.format = ma_format_f32;
    config.dataCallback = data_callback;
    config.periodSizeInFrames = 4096;
    config.periods = 3;

    ma_device device;
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        std::cerr << "Device Init Failed" << std::endl;
        return -1;
    }

    // 4. START THREADS
    std::thread ai_thread(processing_thread_func);
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    // 5. UI LOOP
    while (true) {
        std::cout << "\n----------------------------------------" << std::endl;
        std::cout << "### enter recording length (seconds):" << std::endl;
        std::cout << "### (-1 to replay previous)" << std::endl;
        std::cout << "### (0 to Quit)" << std::endl;
        std::cout << "> ";

        float input;
        if (!(std::cin >> input)) {
            std::cin.clear(); std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
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

        while (g_is_recording || g_is_processing) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        if (input > 0 && g_last_recording.size() > 0) {
             float duration = static_cast<float>(g_last_recording.size()) / (static_cast<float>(INTERNAL_SAMPLE_RATE) * 2.0f);
             std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration * 1000) + 500));
        }
    }

    g_running = false;
    ai_thread.join();
    ma_device_uninit(&device);
    return 0;
}
