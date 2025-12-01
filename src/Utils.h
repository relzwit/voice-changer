#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>     // // Imports the dynamic array container.
#include <atomic>     // // Imports thread-safe variable types.
#include <cstring>    // // Imports memory copy functions.
#include <cmath>      // // Imports math functions (rounding, etc).
#include <algorithm>  // // Imports algorithms like max/min.
#include <iostream>   // // Imports input/output stream.

// --- LOCK-FREE RING BUFFER ---
// // A circular buffer that allows one thread to write audio and another to read it simultaneously without crashing.
class AudioRingBuffer {
private:
    std::vector<float> m_buffer;           // // The actual container for audio samples.
    std::atomic<size_t> m_writeIndex{0};   // // Tracks where data is being written (Thread-Safe).
    std::atomic<size_t> m_readIndex{0};    // // Tracks where data is being read (Thread-Safe).
    size_t m_size = 0;                     // // The total capacity of the buffer.

public:
    // // Initializes the buffer with a specific size (fill with zeros).
    void Init(size_t capacity) {
        m_buffer.resize(capacity, 0.0f);
        m_size = capacity;
        m_writeIndex = 0;
        m_readIndex = 0;
    }

    // // Calculates how many samples are currently waiting in the buffer.
    size_t AvailableRead() const {
        if (m_size == 0) return 0;
        return m_writeIndex.load() - m_readIndex.load();
    }

    // // Writes new audio data into the buffer.
    void Write(const float* data, size_t count) {
        if (m_size == 0) return;
        size_t currentWrite = m_writeIndex.load();
        for (size_t i = 0; i < count; ++i) {
            // // The modulo operator (%) wraps the index back to 0 if we hit the end.
            m_buffer[(currentWrite + i) % m_size] = data[i];
        }
        m_writeIndex.store(currentWrite + count); // // Update the index atomically.
    }

    // // Reads audio data out of the buffer.
    void Read(float* output, size_t count) {
        if (m_size == 0) return;
        size_t currentRead = m_readIndex.load();
        for (size_t i = 0; i < count; ++i) {
            // // Copy data from the buffer to the output array.
            output[i] = m_buffer[(currentRead + i) % m_size];
        }
        m_readIndex.store(currentRead + count); // // Update the index atomically.
    }
};

// --- RESAMPLERS ---

// // Generic Linear Resampler: Converts audio from one sample rate to another.
inline std::vector<float> ResampleLinear(const std::vector<float>& input, int src_rate, int dst_rate) {
    if (input.empty()) return {}; // // Return empty if input is empty.
    if (src_rate == dst_rate) return input; // // Do nothing if rates match.

    // // Calculate the ratio between target and source rates.
    double ratio = static_cast<double>(dst_rate) / static_cast<double>(src_rate);
    // // Calculate new size.
    size_t out_len = std::max<size_t>(1, static_cast<size_t>(std::lrint(input.size() * ratio)));

    std::vector<float> out;
    out.resize(out_len); // // Allocate memory for output.

    double src_len = static_cast<double>(input.size());

    // // Loop through every sample in the *output* buffer.
    for (size_t i = 0; i < out_len; ++i) {
        // // Calculate which position in the *input* this corresponds to.
        double pos = (static_cast<double>(i) / static_cast<double>(out_len - 1)) * (src_len - 1.0);

        // // Safety checks for start/end of buffer.
        if (pos <= 0.0) {
            out[i] = input[0];
            continue;
        }
        if (pos >= src_len - 1.0) {
            out[i] = input[input.size() - 1];
            continue;
        }

        // // Linear Interpolation Math:
        // // Finds the value "between" two existing samples (s0 and s1).
        size_t idx = static_cast<size_t>(pos);
        double frac = pos - static_cast<double>(idx);
        float s0 = input[idx];
        float s1 = input[idx + 1];
        out[i] = static_cast<float>(s0 * (1.0 - frac) + s1 * frac);
    }

    return out;
}

// // Helper: Converts 48k (Mic) -> 40k (AI Model Input).
inline std::vector<float> Resample48To40(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 40000);
}

// // Helper: Elastic Resampler.
// // Forces the 'input' audio to be exactly 'target_count' samples long.
// // This is crucial for preventing the "Deep Voice" effect caused by speed mismatches.
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