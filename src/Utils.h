#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>
#include <atomic>   // Required for thread-safety (std::atomic)
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// --- RING BUFFER ---
// This class solves the "Producer-Consumer" problem.
// The Microphone (Producer) writes data at high speed.
// The AI (Consumer) reads data at a slower, varying speed.
// The Ring Buffer acts as a circular waiting room so they never have to stop and wait for each other.
class AudioRingBuffer {
private:
    std::vector<float> m_buffer; // The actual memory storage for the audio samples.

    // We use std::atomic for indices.
    // Why? In a multi-threaded program, the CPU might "cache" a variable.
    // Thread A might update 'm_writeIndex', but Thread B might still see the old value.
    // 'atomic' forces the CPU to share the true value instantly between threads.
    std::atomic<size_t> m_writeIndex{0}; // Total samples ever written (monotonically increasing).
    std::atomic<size_t> m_readIndex{0};  // Total samples ever read.

    size_t m_size = 0; // The capacity of the buffer.

public:
    // Allocates the memory. Called once at startup.
    void Init(size_t cap) { m_buffer.resize(cap, 0.0f); m_size = cap; }

    // Calculates how many samples are currently waiting to be processed.
    // Logic: (Total Written) - (Total Read).
    // Note: Even if these numbers get huge and overflow, unsigned subtraction handles it correctly.
    size_t AvailableRead() const { if(m_size==0)return 0; return m_writeIndex.load()-m_readIndex.load(); }

    // Pushes data INTO the buffer (Called by Audio Driver).
    void Write(const float* d, size_t c) {
        if(m_size==0)return;

        // Load the current write position.
        size_t w=m_writeIndex.load();

        for(size_t i=0;i<c;++i) {
            // THE MODULO OPERATOR (%):
            // This creates the "Ring" effect.
            // If the buffer size is 100, writing to index 100 wraps around to index 0.
            m_buffer[(w+i)%m_size]=d[i];
        }

        // Update the counter so the Reader knows new data is available.
        m_writeIndex.store(w+c);
    }

    // Pulls data OUT of the buffer (Called by Worker Thread).
    void Read(float* o, size_t c) {
        if(m_size==0)return;

        size_t r=m_readIndex.load();

        for(size_t i=0;i<c;++i) {
            // Same circular logic: Read from (Index % Size).
            o[i]=m_buffer[(r+i)%m_size];
        }

        // Mark these samples as "read" so the Writer can overwrite them later.
        m_readIndex.store(r+c);
    }
};

// --- RESAMPLERS ---
// These functions stretch or squash audio arrays to match sample rates (e.g., 48k -> 40k).

// Core Implementation: Linear Interpolation (Lerp)
// This takes a vector and forces it to be exactly 'target_count' samples long.
inline std::vector<float> ResampleToCount(const std::vector<float>& input, size_t target_count) {
    if (input.empty() || target_count == 0) return {};

    std::vector<float> output;
    output.reserve(target_count); // Pre-allocate memory for performance.

    // Calculate the 'stride' or step size.
    // If shrinking (downsampling), ratio > 1.0 (step over input samples).
    // If stretching (upsampling), ratio < 1.0 (duplicate/blend input samples).
    double ratio = (double)input.size() / (double)target_count;
    double pos = 0.0;

    for (size_t i=0; i<target_count; ++i) {
        // 'pos' is a float (e.g., index 5.75). We need to blend index 5 and index 6.
        size_t idx = (size_t)pos;

        // Boundary check: If we are at the very end, just take the last sample.
        if (idx >= input.size()-1) output.push_back(input.back());
        else {
            // The fractional part (0.75).
            float f = (float)(pos - idx);

            // The Math: (ValueAt5 * 0.25) + (ValueAt6 * 0.75)
            // This creates a smooth line between the two points.
            output.push_back(input[idx]*(1.0f-f) + input[idx+1]*f);
        }
        pos += ratio; // Advance the cursor.
    }
    return output;
}

// Wrapper: Calculates target count based on source and destination Sample Rates.
inline std::vector<float> ResampleLinear(const std::vector<float>& input, int src, int dst) {
    if (input.empty()) return {};
    // New Size = Old Size * (TargetRate / SourceRate)
    return ResampleToCount(input, (size_t)(input.size() * ((double)dst/src)));
}

// Helper for Downsampling to 16kHz (Standard for speech recognition, though not used by RVC).
inline std::vector<float> Resample48To16(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 16000);
}

// Helper for Downsampling to 40kHz (Required for the standard RVC AI models).
inline std::vector<float> Resample48To40(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 40000);
}

#endif