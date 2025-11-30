#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>     // // Provides std::vector (dynamic array).
#include <atomic>     // // Provides std::atomic (thread-safe variables).
#include <cstring>    // // Provides memcpy (fast memory copy).
#include <cmath>      // // Provides math functions like std::log, std::pow.
#include <algorithm>  // // Provides std::min/max.
#include <iostream>   // // Standard I/O.

// --- LOCK-FREE RING BUFFER ---
// // This structure is designed to pass data between the high-speed audio thread and the slower logic thread safely.
class AudioRingBuffer {
private:
    std::vector<float> m_buffer;           // // The actual memory buffer holding the audio data.
    std::atomic<size_t> m_writeIndex{0};   // // Atomic counter for the writing position (Producer).
    std::atomic<size_t> m_readIndex{0};    // // Atomic counter for the reading position (Consumer).
    size_t m_size = 0;                     // // Total size of the buffer.

public:
    void Init(size_t capacity) {
        m_buffer.resize(capacity, 0.0f); // // Allocates the memory.
        m_size = capacity;
        m_writeIndex = 0;
        m_readIndex = 0;
    }

    size_t AvailableRead() const {
        if (m_size == 0) return 0;
        return m_writeIndex.load() - m_readIndex.load(); // // Calculates available data by index difference.
    }

    void Write(const float* data, size_t count) {
        if (m_size == 0) return;
        size_t currentWrite = m_writeIndex.load();
        for (size_t i = 0; i < count; ++i) {
            // // Modulo arithmetic (%) ensures the index wraps around (circular behavior).
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

// --- RESAMPLERS ---

// // Elastic Resampler: Stretches/Squashes any input vector to match a precise target length.
inline std::vector<float> ResampleToCount(const std::vector<float>& input, size_t target_count) {
    if (input.empty() || target_count == 0) return {};
    std::vector<float> output;
    output.reserve(target_count);

    // // Ratio of source size to target size (determines the stretch factor).
    double ratio = (double)input.size() / (double)target_count;
    double pos = 0.0;

    for (size_t i = 0; i < target_count; ++i) {
        size_t index = (size_t)pos; // // Integer part of the position.

        if (index >= input.size() - 1) {
            output.push_back(input.back()); // // Safety: Copy last sample if we reach the end.
        } else {
            // // Linear Interpolation: Calculates a point between input[index] and input[index+1].
            float frac = (float)(pos - index);
            float val = input[index] * (1.0f - frac) + input[index + 1] * frac;
            output.push_back(val);
        }
        pos += ratio; // // Advance position by the ratio.
    }
    return output;
}

// // Downsampling: 48k -> 16k (Required for RVC/Hubert input). Divides sample count by 3.
inline std::vector<float> Resample48To16(const std::vector<float>& input) {
    if (input.empty()) return {};
    std::vector<float> output;
    size_t new_size = input.size() / 3;
    output.reserve(new_size);
    for (size_t i = 0; i < new_size; ++i) {
        size_t base = i * 3;
        float sum = 0.0f;
        int count = 0;
        // // Box filter: Averages 3 samples to get one output sample (better than just skipping).
        for(int k=0; k<3; ++k) {
            if (base + k < input.size()) { sum += input[base + k]; count++; }
        }
        output.push_back(count > 0 ? sum / count : 0.0f);
    }
    return output;
}

// --- PITCH DETECTION HELPERS ---

// // Internal function to detect a single pitch value (Hz) in a chunk of audio using Autocorrelation.
inline float DetectPitchSingle(const float* audio, size_t size, int sampleRate) {
    if (size < 512) return 0.0f;

    int minLag = sampleRate / 1000; // // Minimum lag (Max freq 1000Hz).
    int maxLag = sampleRate / 100;  // // Maximum lag (Min freq 100Hz).

    float bestCorr = -1.0f;
    int bestLag = 0;

    // // Calculate Energy (RMS) for normalization (to make detection volume-independent).
    float energy = 0.0f;
    for(size_t i=0; i<size; ++i) energy += audio[i] * audio[i];
    if (energy < 0.0001f) return 0.0f;

    // // Autocorrelation Loop: Compares signal against itself with a delay (lag).
    for (int lag = minLag; lag <= maxLag; lag += 2) {
        float sum = 0.0f;
        for (size_t i = 0; i < size - lag; i += 4) {
            sum += audio[i] * audio[i + lag]; // // Signal * Delayed Signal
        }

        float norm = sum / energy; // // Normalized correlation coefficient (range -1.0 to 1.0).

        // // OCTAVE FIX: Penalize deep notes (large lag) slightly to prefer the fundamental frequency.
        norm -= (float)lag * 0.0001f;

        if (norm > bestCorr) { bestCorr = norm; bestLag = lag; }
    }

    if (bestLag == 0 || bestCorr < 0.05f) return 0.0f; // // If score is too low, it's unvoiced/noise.
    return (float)sampleRate / bestLag; // // Returns Frequency (Hz).
}

// // Generates a pitch curve for the entire recorded phrase.
inline std::vector<float> DetectPitchCurve(const std::vector<float>& audio, int sampleRate, int64_t target_frames) {
    std::vector<float> f0_curve(target_frames, 0.0f);

    // // Calculate hop size (how many audio samples correspond to one AI embedding frame).
    size_t hop_size = audio.size() / target_frames;
    if (hop_size == 0) hop_size = 1;

    float last_valid_pitch = 100.0f;

    for (int64_t i = 0; i < target_frames; ++i) {
        size_t start = i * hop_size;
        size_t window = 640;
        if (start + window > audio.size()) window = audio.size() - start;

        // // Get pitch for the current window slice.
        float pitch = DetectPitchSingle(audio.data() + start, window, sampleRate);

        // // Smoothing: If silent, hold the last pitch to prevent pitch breaks.
        if (pitch == 0.0f) pitch = last_valid_pitch;
        else last_valid_pitch = pitch;

        f0_curve[i] = pitch;
    }
    return f0_curve;
}

// // Converts Frequency (Hz) into the specific integer index the RVC model expects (1-255).
inline int64_t FreqToCoarsePitch(float f0) {
    if (f0 <= 0.0f) return 0;

    // // RVC Mel Scale constants (Used by the AI for its internal pitch mapping).
    const float f0_min = 50.0f;
    const float f0_max = 1100.0f;
    const float f0_bin = 256.0f;

    float f0_mel_min = 1127.0f * std::log(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * std::log(1.0f + f0_max / 700.0f);

    float f0_mel = 1127.0f * std::log(1.0f + f0 / 700.0f);
    f0_mel = std::max(f0_mel_min, std::min(f0_mel_max, f0_mel));

    int64_t idx = static_cast<int64_t>((f0_mel - f0_mel_min) / (f0_mel_max - f0_mel_min) * (f0_bin - 1) + 1);

    // // Safety Clamps
    if (idx > 255) idx = 255;
    if (idx < 1) idx = 1;
    return idx;
}

#endif //VOICE_CHANGER_UTILS_H