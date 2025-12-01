#ifndef VOICE_CHANGER_DENOISE_H
#define VOICE_CHANGER_DENOISE_H
#include <vector>
#include <rnnoise.h>
#define RNNOISE_FRAME_SIZE 480
class DenoiseEngine {
private: DenoiseState* st = nullptr;
public:
    DenoiseEngine() { st = rnnoise_create(NULL); }
    ~DenoiseEngine() { if(st) rnnoise_destroy(st); }
    void Process(std::vector<float>& buffer) {
        if(!st) return;
        size_t processed = 0;
        const float SCALE = 32768.0f;
        while(processed + RNNOISE_FRAME_SIZE <= buffer.size()) {
            float tmp[RNNOISE_FRAME_SIZE];
            for(int i=0;i<480;++i) tmp[i] = buffer[processed+i]*SCALE;
            rnnoise_process_frame(st, tmp, tmp);
            for(int i=0;i<480;++i) buffer[processed+i] = tmp[i]/SCALE;
            processed += 480;
        }
    }
};
#endif