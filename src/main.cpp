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
#include <ctime> // For generating unique filenames

#define MINIAUDIO_IMPLEMENTATION
#include <miniaudio.h>

#include "Utils.h"
#include "PythonBridge.h"

// --- CONFIGURATION MANAGEMENT ---
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


// --- WAV FILE UTILITY ---
// This function manually constructs and writes a WAV header.
bool SaveWavFile(const std::vector<float>& buffer, const std::string& filename, int channels, int sampleRate) {
    if (buffer.empty()) return false;
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) return false;

    // We save as 16-bit PCM (signed short), the most compatible format.
    int bitsPerSample = 16;
    long byteRate = sampleRate * channels * (bitsPerSample / 8);
    long blockAlign = channels * (bitsPerSample / 8);
    long dataSize = buffer.size() * (bitsPerSample / 8);
    long chunkSize = 36 + dataSize;

    // --- RIFF CHUNK ---
    file.write("RIFF", 4);                                  // Chunk ID
    file.write((char*)&chunkSize, 4);                       // Chunk Size (File size - 8)
    file.write("WAVE", 4);                                  // Format

    // --- FMT SUB-CHUNK ---
    file.write("fmt ", 4);                                  // Sub-chunk 1 ID
    int subChunk1Size = 16;
    file.write((char*)&subChunk1Size, 4);                   // Sub-chunk 1 Size (16 for PCM)
    short audioFormat = 1; // 1 = PCM (uncompressed)
    file.write((char*)&audioFormat, 2);                     // Audio Format
    short numChannels = (short)channels;
    file.write((char*)&numChannels, 2);                     // Number of Channels
    file.write((char*)&sampleRate, 4);                      // Sample Rate
    file.write((char*)&byteRate, 4);                        // Byte Rate
    file.write((char*)&blockAlign, 2);                      // Block Align
    short bps = (short)bitsPerSample;
    file.write((char*)&bps, 2);                             // Bits per Sample

    // --- DATA SUB-CHUNK ---
    file.write("data", 4);                                  // Sub-chunk 2 ID
    file.write((char*)&dataSize, 4);                        // Data Size

    // --- CONVERT AND WRITE DATA (float to 16-bit PCM) ---
    std::vector<short> shortBuffer(buffer.size());
    for (size_t i = 0; i < buffer.size(); ++i) {
        // Clamp float [-1.0, 1.0] and scale to short [-32768, 32767]
        float sample = std::max(-1.0f, std::min(1.0f, buffer[i]));
        shortBuffer[i] = (short)(sample * 32767.0f);
    }

    file.write((char*)shortBuffer.data(), dataSize);
    return true;
}
// --- END WAV FILE UTILITY ---


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

                // --- DEBUG MONITOR ---
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

    int result = std::system("kitty --detach bash /home/relz/code/voice-changer/src/server/start_server.sh &");  // '&' runs in background

    if(result != 0) {
        std::cerr << "Failed to start the server script!" << std::endl;
        return 1;
    }

    std::cout << "Server script started successfully!" << std::endl;
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

    std::string input_line;
    while (true) {
        std::cout << "\n----------------------------------------\n";
        std::cout << "### Enter command or seconds to record:\n";
        std::cout << "### (-1: Replay, 's': Save Last Clip, 'q': Quit)\n";
        std::cout << "> ";

        // Read the entire line of input
        if (!std::getline(std::cin, input_line) || input_line.empty()) continue;

        // --- COMMAND HANDLING ---
        if (input_line == "q") break;

        if (input_line == "s") {
            if (g_last_recording.empty()) {
                std::cout << "[ERROR] No audio clip to save. Record something first." << std::endl;
                continue;
            }
            // Generate unique filename based on timestamp
            std::time_t t = std::time(nullptr);
            std::tm tm = *std::localtime(&t);
            std::ostringstream oss;
            oss << "output_" << std::put_time(&tm, "%Y%m%d_%H%M%S") << ".wav";
            std::string filename = oss.str();

            // g_last_recording is Stereo (2 channels)
            if (SaveWavFile(g_last_recording, filename, 2, INTERNAL_SAMPLE_RATE)) {
                std::cout << "[SUCCESS] Saved to " << filename << std::endl;
            } else {
                std::cout << "[ERROR] Failed to save file." << std::endl;
            }
            continue;
        }

        // --- NUMERIC INPUT HANDLING (Record/Replay) ---
        float input_float = 0.0f;
        try {
            input_float = std::stof(input_line);
        } catch (...) {
            std::cout << "[ERROR] Invalid input. Enter seconds, 's', or 'q'." << std::endl;
            continue;
        }

        if (input_float == -1) {
            g_is_replaying = true;
            g_is_processing = true;
        } else {
            g_config.recording_duration = input_float;
            g_is_recording = true;
        }

        while (g_is_recording || g_is_processing) std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Sleep to allow audio playback to finish
        if (input_float > 0 && g_last_recording.size() > 0) {
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