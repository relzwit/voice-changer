#ifndef VOICE_CHANGER_DENOISE_H
#define VOICE_CHANGER_DENOISE_H

#include <vector>
// This is the external C library "RNNoise" (Recurrent Neural Network Noise suppression).
// It is a lightweight AI model trained to recognize and remove noise like fans,
// keyboard clicks, and static.
#include <rnnoise.h>

// RNNoise has a strict requirement: it MUST process audio in chunks of exactly 480 samples.
// At 48,000 Hz sample rate, 480 samples = 10 milliseconds of audio.
#define RNNOISE_FRAME_SIZE 480

// This class is a "Wrapper". It wraps the messy C-style functions of the library
// into a clean C++ object that manages its own memory (RAII pattern).
class DenoiseEngine {
private:
    // This pointer holds the "brain" of the AI (the memory of previous sounds).
    // In C, we have to manually create and destroy this.
    DenoiseState* st = nullptr;

public:
    // Constructor: Called when we write "DenoiseEngine denoiser;"
    // It allocates memory for the neural network state.
    DenoiseEngine() {
        st = rnnoise_create(NULL);
    }

    // Destructor: Called automatically when the engine goes out of scope.
    // We MUST free the memory here, or we will leak RAM every time we run a thread.
    ~DenoiseEngine() {
        if(st) rnnoise_destroy(st);
    }

    // The main function. It takes a buffer of audio, cleans it, and overwrites it.
    // "std::vector<float>&" means we are modifying the original data directly, not a copy.
    void Process(std::vector<float>& buffer) {
        // Safety check: if the AI failed to load, don't crash.
        if(!st) return;

        size_t processed = 0;

        // --- AUDIO SCALING MAGIC NUMBER ---
        // Floating point audio (what we use) is usually between -1.0 and 1.0.
        // 16-bit Integer audio (what RNNoise expects) is between -32768 and 32767.
        // We define this scale factor to convert between the two formats.
        const float SCALE = 32768.0f;

        // We process the buffer in chunks of 480 samples (10ms).
        // If the buffer has 4800 samples, this loop runs 10 times.
        while(processed + RNNOISE_FRAME_SIZE <= buffer.size()) {

            // Temporary array for the current 10ms chunk.
            float tmp[RNNOISE_FRAME_SIZE];

            // 1. PREPARE INPUT (Scale Up)
            // We multiply by 32768 to make the tiny float numbers (-1.0 to 1.0)
            // look like big integer numbers (-32768 to 32767) that the AI understands.
            for(int i=0; i<480; ++i) {
                tmp[i] = buffer[processed + i] * SCALE;
            }

            // 2. RUN AI
            // The library reads from 'tmp', removes noise, and writes the clean audio back to 'tmp'.
            rnnoise_process_frame(st, tmp, tmp);

            // 3. FINALIZE OUTPUT (Scale Down)
            // We divide by 32768 to shrink the numbers back down to the -1.0 to 1.0 range
            // that the rest of our C++ program expects.
            for(int i=0; i<480; ++i) {
                buffer[processed + i] = tmp[i] / SCALE;
            }

            // Move the index forward to the next chunk.
            processed += 480;
        }
    }
};

#endif