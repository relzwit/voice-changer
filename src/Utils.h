#ifndef VOICE_CHANGER_UTILS_H
#define VOICE_CHANGER_UTILS_H

#include <vector>
#include <atomic>
#include <cstring>
#include <cmath>
#include <algorithm>
#include <iostream>

// --- RING BUFFER ---
class AudioRingBuffer {
private:
    std::vector<float> m_buffer;
    std::atomic<size_t> m_writeIndex{0};
    std::atomic<size_t> m_readIndex{0};
    size_t m_size = 0;
public:
    void Init(size_t cap) { m_buffer.resize(cap, 0.0f); m_size = cap; }
    size_t AvailableRead() const { if(m_size==0)return 0; return m_writeIndex.load()-m_readIndex.load(); }
    void Write(const float* d, size_t c) { if(m_size==0)return; size_t w=m_writeIndex.load(); for(size_t i=0;i<c;++i) m_buffer[(w+i)%m_size]=d[i]; m_writeIndex.store(w+c); }
    void Read(float* o, size_t c) { if(m_size==0)return; size_t r=m_readIndex.load(); for(size_t i=0;i<c;++i) o[i]=m_buffer[(r+i)%m_size]; m_readIndex.store(r+c); }
};

// --- RESAMPLERS ---
inline std::vector<float> ResampleToCount(const std::vector<float>& input, size_t target_count) {
    if (input.empty() || target_count == 0) return {};
    std::vector<float> output; output.reserve(target_count);
    double ratio = (double)input.size() / (double)target_count;
    double pos = 0.0;
    for (size_t i=0; i<target_count; ++i) {
        size_t idx = (size_t)pos;
        if (idx >= input.size()-1) output.push_back(input.back());
        else {
            float f = (float)(pos - idx);
            output.push_back(input[idx]*(1.0f-f) + input[idx+1]*f);
        }
        pos += ratio;
    }
    return output;
}

inline std::vector<float> ResampleLinear(const std::vector<float>& input, int src, int dst) {
    if (input.empty()) return {};
    return ResampleToCount(input, (size_t)(input.size() * ((double)dst/src)));
}

inline std::vector<float> Resample48To16(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 16000);
}

inline std::vector<float> Resample48To40(const std::vector<float>& input) {
    return ResampleLinear(input, 48000, 40000);
}

#endif