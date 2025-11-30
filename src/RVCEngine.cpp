#include "RVCEngine.h"
#include "Utils.h"
#include <iostream>
#include <cmath>
#include <chrono>

RVCEngine::RVCEngine() : env(ORT_LOGGING_LEVEL_WARNING, "RVC_Engine") {
    session_opts.SetIntraOpNumThreads(2);
    session_opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
}

bool RVCEngine::LoadModels(const std::string& hubert_path, const std::string& voice_path, bool use_gpu) {
    try {
        // --- GPU SELECTION LOGIC ---
        if (use_gpu) {
            std::cout << "[AI] Attempting to enable NVIDIA GPU (CUDA)..." << std::endl;
            try {
                OrtCUDAProviderOptions cuda_options;
                cuda_options.device_id = 0; // Default GPU
                session_opts.AppendExecutionProvider_CUDA(cuda_options);
                std::cout << "[AI] CUDA Provider Added!" << std::endl;
            } catch (...) {
                std::cerr << "[AI] WARNING: Could not load CUDA provider. Falling back to CPU." << std::endl;
            }
        }

        hubert_session = std::make_unique<Ort::Session>(env, hubert_path.c_str(), session_opts);
        voice_session = std::make_unique<Ort::Session>(env, voice_path.c_str(), session_opts);

        size_t num_inputs = voice_session->GetInputCount();
        for(size_t i=0; i<num_inputs; i++) {
            auto name_ptr = voice_session->GetInputNameAllocated(i, allocator);
            std::string name = name_ptr.get();
            auto type_info = voice_session->GetInputTypeInfo(i);
            auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
            input_type_map[name] = tensor_info.GetElementType();
        }

        std::cout << "[AI] Models Loaded. Type: "
                  << (input_type_map["feats"] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? "FP16" : "FP32")
                  << std::endl;

        return true;
    } catch (const std::exception& e) {
        std::cerr << "[AI] ERROR LOADING MODELS: " << e.what() << std::endl;
        return false;
    }
}

double RVCEngine::Benchmark(int test_samples) {
    std::vector<float> dummy_audio(test_samples, 0.0f);
    auto start = std::chrono::high_resolution_clock::now();
    ProcessChunk(dummy_audio);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms = end - start;
    return ms.count();
}

Ort::Value RVCEngine::CreateDynamicTensor(
        Ort::MemoryInfo& mem_info,
        const std::string& name,
        std::vector<float>& source_data,
        std::vector<Ort::Float16_t>& fp16_storage,
        const std::vector<int64_t>& shape
) {
    if (input_type_map[name] == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        fp16_storage.clear();
        fp16_storage.reserve(source_data.size());
        for (float val : source_data) fp16_storage.push_back(Ort::Float16_t(val));

        return Ort::Value::CreateTensor<Ort::Float16_t>(
            mem_info, fp16_storage.data(), fp16_storage.size(), shape.data(), shape.size()
        );
    } else {
        return Ort::Value::CreateTensor<float>(
            mem_info, source_data.data(), source_data.size(), shape.data(), shape.size()
        );
    }
}

std::vector<float> RVCEngine::ProcessChunk(std::vector<float>& audio_16k) {
    Ort::MemoryInfo mem_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 1. PITCH
    float current_pitch = DetectPitch(audio_16k, 16000);
    float target_pitch = 0.0f;
    if (current_pitch > 0.0f) {
        target_pitch = current_pitch * pitch_multiplier;
        if (target_pitch < 220.0f) target_pitch = 220.0f;
    }
    if (target_pitch == 0.0f) target_pitch = last_detected_pitch;

    // 2. HUBERT
    std::vector<int64_t> hubert_shape = {1, 1, (int64_t)audio_16k.size()};
    Ort::Value hubert_input = Ort::Value::CreateTensor<float>(
            mem_info, audio_16k.data(), audio_16k.size(), hubert_shape.data(), hubert_shape.size()
    );
    const char* hubert_io[] = {"source", "embed"};
    auto hubert_result = hubert_session->Run(Ort::RunOptions{nullptr}, &hubert_io[0], &hubert_input, 1, &hubert_io[1], 1);

    float* embed_data = hubert_result[0].GetTensorMutableData<float>();
    auto embed_info = hubert_result[0].GetTensorTypeAndShapeInfo();
    std::vector<int64_t> embed_shape = embed_info.GetShape();
    size_t embed_count = embed_info.GetElementCount();
    std::vector<float> feats_vector(embed_data, embed_data + embed_count);

    // 3. PITCH VECTORS
    int64_t num_frames = embed_shape[1];
    std::vector<float> pitch_float(num_frames);
    std::vector<int64_t> pitch_int(num_frames);

    for (int i = 0; i < num_frames; ++i) {
        float t = (float)i / (float)num_frames;
        float smooth_freq = last_detected_pitch * (1.0f - t) + target_pitch * t;
        if (smooth_freq < 50.0f) smooth_freq = 220.0f;
        pitch_float[i] = smooth_freq;
        pitch_int[i] = FreqToCoarsePitch(smooth_freq);
    }
    last_detected_pitch = target_pitch;

    // 4. INFERENCE
    std::vector<int64_t> sid(1, 0);
    std::vector<int64_t> p_len(1, num_frames);
    std::vector<Ort::Float16_t> feats_fp16, pitchf_fp16;

    Ort::Value feats_t = CreateDynamicTensor(mem_info, "feats", feats_vector, feats_fp16, embed_shape);
    Ort::Value pitchf_t = CreateDynamicTensor(mem_info, "pitchf", pitch_float, pitchf_fp16, {1, num_frames});

    std::vector<int64_t> shape_1D = {1};
    std::vector<int64_t> shape_2D = {1, num_frames};

    Ort::Value plen_t = Ort::Value::CreateTensor<int64_t>(mem_info, p_len.data(), p_len.size(), shape_1D.data(), shape_1D.size());
    Ort::Value sid_t = Ort::Value::CreateTensor<int64_t>(mem_info, sid.data(), sid.size(), shape_1D.data(), shape_1D.size());
    Ort::Value pitch_t = Ort::Value::CreateTensor<int64_t>(mem_info, pitch_int.data(), pitch_int.size(), shape_2D.data(), shape_2D.size());

    const char* input_names[] = {"feats", "p_len", "pitch", "pitchf", "sid"};
    const char* output_names[] = {"audio"};
    Ort::Value inputs[] = { std::move(feats_t), std::move(plen_t), std::move(pitch_t), std::move(pitchf_t), std::move(sid_t) };

    auto voice_result = voice_session->Run(Ort::RunOptions{nullptr}, input_names, inputs, 5, output_names, 1);

    auto output_info = voice_result[0].GetTensorTypeAndShapeInfo();
    size_t audio_count = output_info.GetElementCount();

    if (output_info.GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16) {
        const Ort::Float16_t* raw_out = voice_result[0].GetTensorMutableData<Ort::Float16_t>();
        std::vector<float> final_audio;
        final_audio.reserve(audio_count);
        for(size_t i=0; i<audio_count; ++i) final_audio.push_back(static_cast<float>(raw_out[i]));
        return final_audio;
    } else {
        float* audio_out = voice_result[0].GetTensorMutableData<float>();
        return std::vector<float>(audio_out, audio_out + audio_count);
    }
}