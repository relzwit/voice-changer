// /src/main.cpp

#include <iostream>
#include <cstdlib>
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
#include <sstream>
#include <ctime>

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "PythonBridge.h"

// --- CONFIGURATION MANAGEMENT ---
void load_config(float& pitch_shift, float& duration) {
    std::ifstream file("config.ini");
    if (!file.is_open()) return;

    std::string line;
    while (std::getline(file, line)) {
        if (line.empty() || line[0] == '[' || line[0] == ';') continue;

        std::size_t equalPos = line.find('=');
        if (equalPos == std::string::npos) continue;

        std::string key = line.substr(0, equalPos);
        std::string value = line.substr(equalPos + 1);

        key.erase(0, key.find_first_not_of(" \t"));
        key.erase(key.find_last_not_of(" \t") + 1);
        value.erase(0, value.find_first_not_of(" \t"));
        value.erase(value.find_last_not_of(" \t") + 1);

        if (key == "PITCH_SHIFT_SEMITONES") {
            try { pitch_shift = std::stof(value); } catch (...) {}
        } else if (key == "RECORDING_DURATION_SECONDS") {
            try { duration = std::stof(value); } catch (...) {}
        }
    }
}

// --- SETTINGS ---
struct AppConfig {
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

std::vector<float> g_last_recording;

PythonBridge g_bridge;

#define INTERNAL_CHANNELS 2
#define INTERNAL_SAMPLE_RATE 48000

// --- WAV FILE UTILITY ---
bool SaveWavFile(const std::vector<float>& buffer, const std::string& filename, int channels, int sampleRate) {
    if (buffer.empty()) return false;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    int bitsPerSample = 16;
    long byteRate = sampleRate * channels * (bitsPerSample / 8);
    long blockAlign = channels * (bitsPerSample / 8);
    long dataSize = buffer.size() * (bitsPerSample / 8);
    long chunkSize = 36 + dataSize;

    file.write("RIFF", 4);
    file.write((char*)&chunkSize, 4);
    file.write("WAVE", 4);

    file.write("fmt ", 4);
    int subChunk1Size = 16;
    file.write((char*)&subChunk1Size, 4);
    short audioFormat = 1;
    file.write((char*)&audioFormat, 2);
    short numChannels = (short)channels;
    file.write((char*)&numChannels, 2);
    file.write((char*)&sampleRate, 4);
    file.write((char*)&byteRate, 4);
    file.write((char*)&blockAlign, 2);
    short bps = (short)bitsPerSample;
    file.write((char*)&bps, 2);

    file.write("data", 4);
    file.write((char*)&dataSize, 4);

    std::vector<short> shortBuffer(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        float sample = std::max(-1.0f, std::min(1.0f, buffer[i]));
        shortBuffer[i] = (short)(sample * 32767.0f);
    }

    file.write((char*)shortBuffer.data(), dataSize);
    return true;
}

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

            if (g_is_replaying && !g_last_recording.empty()) {
                std::cout << "\r[REPLAYING]..." << std::flush;
                g_rb_output.Write(g_last_recording.data(), g_last_recording.size());
                g_is_replaying = false;
                g_is_processing = false;
                continue;
            }

            if (!g_is_recording) {
                burst_buffer.clear();
                continue;
            }

            mono.clear();
            for (size_t i=0; i<WORK_SIZE/2; ++i) mono.push_back((input_chunk[i*2]+input_chunk[i*2+1])*0.5f);
            burst_buffer.insert(burst_buffer.end(), mono.begin(), mono.end());

            if (burst_buffer.size() % 4096 == 0) {
                std::cout << "\r[REC] " << std::fixed << std::setprecision(1)
                          << (float)burst_buffer.size()/INTERNAL_SAMPLE_RATE << "s / "
                          << g_config.recording_duration << "s   " << std::flush;
            }

            size_t target_samples = static_cast<size_t>(g_config.recording_duration * INTERNAL_SAMPLE_RATE);
            if (burst_buffer.size() >= target_samples) {
                g_is_processing = true;
                std::cout << "\n[PROCESSING] Sending audio to AI..." << std::flush;

                float max_val = 0.0f;
                for (float s : burst_buffer) max_val = std::max(max_val, std::abs(s));
                if (max_val > 0.001f) {
                    float factor = std::min(0.9f/max_val, 5.0f);
                    for (float &s : burst_buffer) s *= factor;
                }

                auto future_result = std::async(std::launch::async, [&]() {
                    return g_bridge.ProcessAudio(burst_buffer, static_cast<int>(g_config.pitch_shift_semitones));
                });

                const char spinner[] = {'|','/','-','\\'};
                int spin_idx = 0;
                while (future_result.wait_for(std::chrono::milliseconds(100)) != std::future_status::ready) {
                    std::cout << "\r[PROCESSING] Sending audio to AI... " << spinner[spin_idx] << std::flush;
                    spin_idx = (spin_idx+1)%4;
                }

                auto processed = future_result.get();
                if (!processed.empty()) {
                    std::vector<float> stereo;
                    stereo.reserve(processed.size()*2);
                    for (float s : processed) { stereo.push_back(s); stereo.push_back(s); }

                    g_last_recording = stereo;
                    g_rb_output.Write(stereo.data(), stereo.size());
                    std::cout << "\r[PROCESSING] Sending audio to AI... [DONE]" << std::endl;
                    std::cout << "[PLAYING AI RESULT]" << std::endl;
                } else {
                    std::cout << "\r[PROCESSING] Sending audio to AI... [ERROR]" << std::endl;
                }

                burst_buffer.clear();
                g_is_processing = false;
                g_is_recording = false;
            }
        } else std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

int main() {
    load_config(g_config.pitch_shift_semitones, g_config.recording_duration);

    std::cout << "========================================\n";
    std::cout << "      RVC VOICE CHANGER - C++ CLIENT    \n";
    std::cout << "========================================\n";
    std::cout << "[Client] Pitch shift: " << g_config.pitch_shift_semitones << " semitones." << std::endl;

    // --- WAIT FOR PYTHON SERVER ---
    const char spinner[] = {'|','/','-','\\'};
    int spin_idx = 0;
    const std::string connect_msg = "Waiting for Python server...";
    while (!g_bridge.Connect()) {
        std::cout << "\r" << connect_msg << " " << spinner[spin_idx] << std::flush;
        spin_idx = (spin_idx + 1) % 4;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
    std::cout << "\r" << std::string(connect_msg.length() + 5, ' ') << "\r"; // Clear line
    std::cout << "[SUCCESS] Python Bridge Connected.\n";
    // -------------------------------

    g_rb_input.Init(48000*30*INTERNAL_CHANNELS);
    g_rb_output.Init(48000*30*INTERNAL_CHANNELS);

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

    std::string input_line;
    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### Enter command or seconds to record:\n";
        std::cout << "### (-1: Replay, 's': Save Last Clip, 'q': Quit)\n> ";

        if (!std::getline(std::cin, input_line) || input_line.empty()) continue;

        if (input_line=="q") break;

        if (input_line=="s") {
            if (g_last_recording.empty()) { std::cout << "[ERROR] No audio clip to save." << std::endl; continue; }

            std::string user_filename;
            std::cout << "Enter filename (without extension): ";
            if (!std::getline(std::cin, user_filename) || user_filename.empty()) {
                std::time_t t = std::time(nullptr);
                std::tm tm = *std::localtime(&t);
                std::ostringstream oss;
                oss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S");
                user_filename = oss.str();
            }

            std::string dir = "/home/relz/code/voice-changer/saved_outputs";
            std::string full_path = dir + "/" + user_filename + ".wav";

            if (SaveWavFile(g_last_recording, full_path, 2, INTERNAL_SAMPLE_RATE))
                std::cout << "[SUCCESS] Saved to " << full_path << std::endl;
            else
                std::cout << "[ERROR] Failed to save file." << std::endl;
            continue;
        }

        float input_float = 0.0f;
        try { input_float = std::stof(input_line); } catch (...) {
            std::cout << "[ERROR] Invalid input. Enter seconds, 's', or 'q'." << std::endl;
            continue;
        }

        if (input_float == -1) { g_is_replaying = true; g_is_processing = true; }
        else { g_config.recording_duration = input_float; g_is_recording = true; }

        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

        if (input_float>0 && !g_last_recording.empty()) {
            float duration = static_cast<float>(g_last_recording.size()) / (48000.0f*2.0f);
            std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(duration*1000)+500));
        }
    }

    g_running = false;
    ai_thread.join();
    g_bridge.Disconnect();
    ma_device_uninit(&device);

    std::cout << "[Client] Shutdown complete. Exiting." << std::endl;
    return 0;
}
