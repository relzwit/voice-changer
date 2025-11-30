#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <chrono>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "RVCEngine.h"
#include "Denoise.h"

// --- GLOBAL SETTINGS ---
struct AppConfig {
    bool use_gpu = false;
    bool walkie_talkie_mode = true;
    float pitch_shift_semitones = 12.0f;
} g_config;

AudioRingBuffer g_rb_input;
AudioRingBuffer g_rb_output;
std::atomic<bool> g_running(true);
RVCEngine g_ai_engine;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000

// --- HELPERS ---
void convert_to_internal(const float* pInput, float* pInternalBuffer, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pInternalBuffer, pInput, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pInternalBuffer[i*2] = pInput[i]; pInternalBuffer[i*2+1] = pInput[i]; }
}

void convert_from_internal(const float* pInternalBuffer, float* pOutput, ma_uint32 frameCount, ma_uint32 hwChannels) {
    if (hwChannels == 2) memcpy(pOutput, pInternalBuffer, frameCount * 2 * sizeof(float));
    else if (hwChannels == 1) for (ma_uint32 i=0; i<frameCount; ++i) { pOutput[i] = (pInternalBuffer[i*2] + pInternalBuffer[i*2+1]) * 0.5f; }
}

// --- AUDIO DRIVER ---
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
        for (ma_uint32 i = 0; i < remaining; ++i) outRaw[offset + i] = 0.0f;
    }
}

// --- PROCESSING THREAD ---
void processing_thread_func() {
    std::cout << "[THREAD] Processor Started. Mode: "
              << (g_config.walkie_talkie_mode ? "WALKIE-TALKIE" : "REAL-TIME")
              << std::endl;

    // Pitch Config
    float multiplier = std::pow(2.0f, g_config.pitch_shift_semitones / 12.0f);
    g_ai_engine.SetPitchShift(multiplier);

    // Tools
    DenoiseEngine denoiser;
    const float INPUT_GAIN = 4.0f;
    const float GATE_THRESH = 0.03f;

    // Walkie Talkie Vars
    std::vector<float> burst_buffer;
    burst_buffer.reserve(48000 * 10); // 10 seconds capacity
    bool is_recording_burst = false;
    int silence_counter = 0;

    const size_t WORK_SIZE = 1024 * INTERNAL_CHANNELS;
    std::vector<float> input_chunk(WORK_SIZE);

    while (g_running) {
        if (!g_config.walkie_talkie_mode) {
            // Real Time Logic (Placeholder for gaming PC)
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
        else {
            if (g_rb_input.AvailableRead() >= WORK_SIZE) {
                g_rb_input.Read(input_chunk.data(), WORK_SIZE);

                // 1. Mono Mix
                std::vector<float> mono(WORK_SIZE / 2);
                for(size_t i=0; i<mono.size(); ++i) mono[i] = (input_chunk[i*2] + input_chunk[i*2+1]) * 0.5f;

                // 2. AI DENOISE
                denoiser.Process(mono);

                // 3. Gain & Metering
                float sum = 0.0f;
                for(float& s : mono) {
                    s *= INPUT_GAIN;
                    if(s > 1.0f) s = 1.0f;
                    if(s < -1.0f) s = -1.0f;
                    sum += s*s;
                }
                float rms = std::sqrt(sum / mono.size());

                // 4. Logic
                if (rms > GATE_THRESH) {
                    is_recording_burst = true;
                    silence_counter = 0;
                    burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());
                    std::cout << "\r[REC] Vol:" << std::fixed << std::setprecision(2) << rms << " Size:" << burst_buffer.size()/48000.0f << "s   " << std::flush;
                }
                else {
                    if (is_recording_burst) {
                        silence_counter++;

                        // FIX: Wait ~1.5s (60 ticks) before cutting off
                        if (silence_counter > 60) {
                            std::cout << " [PROCESSING...] " << std::flush;

                            auto resampled = Resample48To16(burst_buffer);
                            auto processed = g_ai_engine.ProcessChunk(resampled);
                            auto output = Resample16To48(processed);

                            std::vector<float> stereo;
                            stereo.reserve(output.size()*2);
                            for(float s : output) { stereo.push_back(s); stereo.push_back(s); }

                            g_rb_output.Write(stereo.data(), stereo.size());

                            std::cout << "[PLAYING]" << std::endl;

                            burst_buffer.clear();
                            is_recording_burst = false;
                            silence_counter = 0;
                        } else {
                            // Keep buffer alive during pauses
                            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());
                        }
                    } else {
                        // Idle
                        std::vector<float> silence(WORK_SIZE, 0.0f);
                        g_rb_output.Write(silence.data(), silence.size());
                    }
                }
            } else {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "   RVC VOICE CHANGER - CONFIGURATION    " << std::endl;
    std::cout << "========================================" << std::endl;

    // Default to Walkie-Talkie on laptop
    g_config.walkie_talkie_mode = true;

    std::cout << "Pitch Shift (Semitones):" << std::endl;
    std::cout << "  0  = No Change" << std::endl;
    std::cout << " +12 = Female (High)" << std::endl;
    std::cout << "> ";
    float pitch;
    if (!(std::cin >> pitch)) pitch = 12.0f;
    g_config.pitch_shift_semitones = pitch;

    std::cout << "\n----------------------------------------" << std::endl;

    if (!g_ai_engine.LoadModels("../models/hubert.onnx", "../models/voice.onnx", false)) {
        return -1;
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
    if (ma_device_init(NULL, &config, &device) != MA_SUCCESS) {
        std::cerr << "Device Init Failed" << std::endl;
        return -1;
    }

    std::thread ai_thread(processing_thread_func);
    if (ma_device_start(&device) != MA_SUCCESS) return -1;

    std::string dummy;
    std::getline(std::cin, dummy);

    std::cout << "SYSTEM ONLINE. AI Denoising Active." << std::endl;
    getchar();

    g_running = false;
    ai_thread.join();
    ma_device_uninit(&device);
    return 0;
}