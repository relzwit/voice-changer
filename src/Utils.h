#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// --- LOCK-FREE RING BUFFER (SAFE) ---
class AudioRingBuffer {
private:
    std::vector<float> m_buffer;
    std::atomic<size_t> m_writeIndex{0};
    std::atomic<size_t> m_readIndex{0};
    size_t m_size = 0;

public:
    void Init(size_t capacity) {
        m_buffer.resize(capacity, 0.0f);
        m_size = capacity;
        m_writeIndex = 0;
        m_readIndex = 0;
    }

    size_t AvailableRead() const {
        if (m_size == 0) return 0;
        return m_writeIndex.load() - m_readIndex.load();
    }

    void Write(const float* data, size_t count) {
        if (m_size == 0) return;
        size_t currentWrite = m_writeIndex.load();
        for (size_t i = 0; i < count; ++i) {
            m_buffer[(currentWrite + i) % m_size] = data[i];
        }
        m_writeIndex.store(currentWrite + count);
    }

    void Read(float* output, size_t count) {
        if (m_size == 0) return;
        size_t currentRead = m_readIndex.load();
        for (size_t i = 0; i < count; ++i) {
            output[i] = m_buffer[(currentRead + i) % m_size];
        }
        m_readIndex.store(currentRead + count);
    }
};

// --- SAFE RESAMPLER ---
inline std::vector<float> Resample48To16(const std::vector<float>& input) {
    if (input.empty()) return {};
    std::vector<float> output;
    size_t new_size = input.size() / 3;
    output.reserve(new_size);

    for (size_t i = 0; i < new_size; ++i) {
        size_t base = i * 3;
        if (base + 2 < input.size()) {
            float avg = (input[base] + input[base+1] + input[base+2]) / 3.0f;
            output.push_back(avg);
        } else {
            output.push_back(input[base]);
        }
    }
    return output;
}

inline std::vector<float> Resample16To48(const std::vector<float>& input) {
    if (input.empty()) return {};
    std::vector<float> output;
    output.reserve(input.size() * 3);

    for (size_t i = 0; i < input.size(); ++i) {
        float current = input[i];
        float next = (i + 1 < input.size()) ? input[i+1] : current;

        output.push_back(current);
        output.push_back(current * 0.666f + next * 0.333f);
        output.push_back(current * 0.333f + next * 0.666f);
    }
    return output;
}

// --- NORMALIZED PITCH DETECTOR ---
inline float DetectPitch(const std::vector<float>& audio, int sampleRate) {
    size_t n = audio.size();
    if (n < 512) return 0.0f;

    // UPDATED: Ignore < 100Hz (Prevents "Octave Errors")
    int minLag = sampleRate / 800; // ~800Hz Max
    int maxLag = sampleRate / 100; // ~100Hz Min

    float bestCorr = -1.0f;
    int bestLag = 0;

    float energy = 0.0f;
    for (float s : audio) energy += s * s;
    if (energy < 0.0001f) return 0.0f;

    for (int lag = minLag; lag <= maxLag; lag += 2) {
        float sum = 0.0f;
        for (int i = 0; i < n - lag; i += 4) {
            sum += audio[i] * audio[i + lag];
        }
        float norm = sum / energy;
        if (norm > bestCorr) {
            bestCorr = norm;
            bestLag = lag;
        }
    }

    if (bestLag == 0 || bestCorr < 0.05f) return 0.0f;
    return (float)sampleRate / bestLag;
}

// --- RVC PITCH MATH ---
inline int64_t FreqToCoarsePitch(float f0) {
    if (f0 <= 0.0f) return 0;

    const float f0_min = 50.0f;
    const float f0_max = 1100.0f;
    const float f0_bin = 256.0f;

    float f0_mel_min = 1127.0f * std::log(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * std::log(1.0f + f0_max / 700.0f);

    float f0_mel = 1127.0f * std::log(1.0f + f0 / 700.0f);
    f0_mel = std::max(f0_mel_min, std::min(f0_mel_max, f0_mel));

    int64_t idx = static_cast<int64_t>((f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * (f0_bin - 1) + 1);

    // Safety Clamps
    if (idx > 255) idx = 255;
    if (idx < 1) idx = 1;
    return idx;
}

#endif //VOICE_CHANGER_UTILS_H