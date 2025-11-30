#ifndef VOICE_CHANGER_DENOISE_H
#define VOICE_CHANGER_DENOISE_H

#include <vector>
#include <rnnoise.h>

// RNNoise processes 10ms chunks (480 samples @ 48kHz)
#define RNNOISE_FRAME_SIZE 480

class DenoiseEngine {
private:
    DenoiseState* st = nullptr;

public:
    DenoiseEngine() {
        st = rnnoise_create(NULL);
    }

    ~DenoiseEngine() {
        if (st) rnnoise_destroy(st);
    }

    // Process a large buffer in 480-sample steps
    void Process(std::vector<float>& buffer) {
        if (!st) return;

        size_t total_samples = buffer.size();
        size_t processed = 0;

        // RNNoise expects floats in the range of short ints (-32768 to 32767)
        const float SCALE = 32768.0f;

        while (processed + RNNOISE_FRAME_SIZE <= total_samples) {
            float temp_frame[RNNOISE_FRAME_SIZE];

            // 1. Scale Up (0.5 -> 16384)
            for (int i = 0; i < RNNOISE_FRAME_SIZE; ++i) {
                temp_frame[i] = buffer[processed + i] * SCALE;
            }

            // 2. Run AI Cleaning
            rnnoise_process_frame(st, temp_frame, temp_frame);

            // 3. Scale Down (16384 -> 0.5)
            for (int i = 0; i < RNNOISE_FRAME_SIZE; ++i) {
                buffer[processed + i] = temp_frame[i] / SCALE;
            }

            processed += RNNOISE_FRAME_SIZE;
        }
    }
};

#endif //VOICE_CHANGER_DENOISE_H