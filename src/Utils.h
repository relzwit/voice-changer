#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// --- LOCK-FREE RING BUFFER ---
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

// --- GENERIC LINEAR RESAMPLER ---
// Simple, robust linear resampling. Good quality for down/up resampling at moderate ratios.
// Not as high quality as libsamplerate, but avoids external deps and will fix sample-rate mismatches.
inline std::vector<float> ResampleLinear(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (input.empty()) return {};
    if (src_rate == dst_rate) return input; // no-op

    // Compute output length with rounding
    double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    size_t out_len = std::max<size_t>(1, static_cast<size_t>(std::lrint(input.size() * ratio)));

    std::vector<float> out;
    out.resize(out_len);

    double src_len = static_cast<double>(input.size());
    for (size_t i = 0; i < out_len; ++i) {
        // map output index i to fractional input position
        double pos = (static_cast<double>(i) / static_cast<double>(out_len - 1)) * (src_len - 1.0);
        if (pos <= 0.0) {
            out[i] = input[0];
            continue;
        }
        if (pos >= src_len - 1.0) {
            out[i] = input[input.size() - 1];
            continue;
        }
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        float s0 = input[idx];
        float s1 = input[idx + 1];
        out[i] = static_cast<float>(s0 * (1.0 - frac) + s1 * frac);
    }

    return out;
}

// Convenience wrapper: 48k -> 40k (used to prepare audio for the 40k RVC model).
inline std::vector<float> Resample48To40(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 40000);
}

// Convenience wrapper: Resample arbitrary input to a target count (keeps duration).
inline std::vector<float> ResampleToCount(const std::vector<float>& input, size_t target_count) {
    if (input.empty() || target_count == 0) return {};
    std::vector<float> output;
    output.reserve(target_count);

    double ratio = static_cast<double>(input.size()) / static_cast<double>(target_count);
    double pos = 0.0;

    for (size_t i = 0; i < target_count; ++i) {
        size_t index = static_cast<size_t>(pos);
        if (index >= input.size() - 1) {
            output.push_back(input.back());
        } else {
            float frac = static_cast<float>(pos - index);
            float val = input[index] * (1.0f - frac) + input[index + 1] * frac;
            output.push_back(val);
        }
        pos += ratio;
    }
    return output;
}

#endif //VOICE_CHANGER_UTILS_H
