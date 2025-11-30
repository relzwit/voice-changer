#ifndef VOICE_CHANGER_DENOISE_H
#define VOICE_CHANGER_DENOISE_H

#include <vector>
#include <rnnoise.h> // // RNNoise C library header (The AI-based noise suppressor).

// // RNNoise processes exactly 480 samples (10ms @ 48kHz) at a time.
#define RNNOISE_FRAME_SIZE 480

// // Wrapper class to manage the RNNoise state machine.
class DenoiseEngine {
private:
    DenoiseState* st = nullptr; // // Pointer to the persistent state of the denoising AI.

public:
    DenoiseEngine() { st = rnnoise_create(NULL); }
    ~DenoiseEngine() { if (st) rnnoise_destroy(st); }

    // // Main processing function: cleans the audio buffer in 10ms chunks.
    void Process(std::vector<float>& buffer) {
        if (!st) return;

        size_t total_samples = buffer.size();
        size_t processed = 0;

        // // RNNoise expects audio scaled to the size of a 16-bit integer (±32768).
        const float SCALE = 32768.0f;

        while (processed + RNNOISE_FRAME_SIZE <= total_samples) {
            float temp_frame[RNNOISE_FRAME_SIZE];

            // // 1. Scale Up: Convert [±1.0] to [±32768].
            for (int i = 0; i < RNNOISE_FRAME_SIZE; ++i) {
                temp_frame[i] = buffer[processed + i] * SCALE;
            }

            // // 2. Run AI Denoising: The core function call.
            rnnoise_process_frame(st, temp_frame, temp_frame);

            // // 3. Scale Down: Convert the cleaned audio back to the [±1.0] range.
            for (int i = 0; i < RNNOISE_FRAME_SIZE; ++i) {
                buffer[processed + i] = temp_frame[i] / SCALE;
            }

            processed += RNNOISE_FRAME_SIZE;
        }
    }
};

#endif //VOICE_CHANGER_DENOISE_H