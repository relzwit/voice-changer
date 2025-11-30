#ifndef VOICE_CHANGER_RVCENGINE_H
#define VOICE_CHANGER_RVCENGINE_H

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <onnxruntime_cxx_api.h>

class RVCEngine {
private:
    Ort::Env env;
    Ort::SessionOptions session_opts;
    std::unique_ptr<Ort::Session> hubert_session;
    std::unique_ptr<Ort::Session> voice_session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::map<std::string, ONNXTensorElementDataType> input_type_map;
    float last_detected_pitch = 100.0f;
    float pitch_multiplier = 2.0f;

    Ort::Value CreateDynamicTensor(
            Ort::MemoryInfo& mem_info,
            const std::string& name,
            std::vector<float>& source_data,
            std::vector<Ort::Float16_t>& fp16_storage,
            const std::vector<int64_t>& shape
    );

public:
    RVCEngine();

    // Updated to accept GPU flag
    bool LoadModels(const std::string& hubert_path, const std::string& voice_path, bool use_gpu);

    std::vector<float> ProcessChunk(std::vector<float>& audio_16k);
    double Benchmark(int test_samples);
    void SetPitchShift(float multiplier) { pitch_multiplier = multiplier; }
};

#endif //VOICE_CHANGER_RVCENGINE_H