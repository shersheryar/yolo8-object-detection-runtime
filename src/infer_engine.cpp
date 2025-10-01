#include "infer_engine.h"
#include <filesystem>
#include <stdexcept>

InferEngine::InferEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "InferEngine") {}

InferEngine::InferEngine(const std::string& model_path) : env_(ORT_LOGGING_LEVEL_WARNING, "InferEngine") {
    if (!loadModel(model_path)) {
        throw std::runtime_error("Failed to load model: " + model_path);
    }
}

InferEngine::~InferEngine() = default;

bool InferEngine::loadModel(const std::string& model_path) {
    try {
        if (!std::filesystem::exists(model_path)) {
            std::cerr << "Model file not found: " << model_path << std::endl;
            return false;
        }

        Ort::SessionOptions session_options;
        session_options.SetIntraOpNumThreads(1);
        session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);

        Ort::AllocatorWithDefaultOptions allocator;
        size_t num_input_nodes = session_->GetInputCount();
        if (num_input_nodes != 1) {
            std::cerr << "Expected 1 input node, got " << num_input_nodes << std::endl;
            return false;
        }

        auto input_name = session_->GetInputNameAllocated(0, allocator);
        auto input_type_info = session_->GetInputTypeInfo(0);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();

        if (input_dims.size() == 4) {
            input_height_ = static_cast<int>(input_dims[2]);
            input_width_ = static_cast<int>(input_dims[3]);
        }

        model_path_ = model_path;
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        std::cout << "Input dimensions: " << input_width_ << "x" << input_height_ << std::endl;
        
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        return false;
    }
}

cv::Mat InferEngine::infer(const cv::Mat& input_blob) {
    if (!session_) {
        std::cerr << "Model not loaded" << std::endl;
        return cv::Mat();
    }

    if (input_blob.empty()) {
        return cv::Mat();
    }

    try {
        Ort::AllocatorWithDefaultOptions allocator;
        auto input_name = session_->GetInputNameAllocated(0, allocator);
        auto output_name = session_->GetOutputNameAllocated(0, allocator);

        std::vector<int64_t> input_shape = {1, 3, input_height_, input_width_};
        size_t input_tensor_size = input_blob.total() * input_blob.elemSize();
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, 
            const_cast<float*>(input_blob.ptr<float>()), 
            input_tensor_size / sizeof(float),
            input_shape.data(), 
            input_shape.size()
        );

        const char* input_names[] = {input_name.get()};
        const char* output_names[] = {output_name.get()};
        
        auto output_tensors = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            &input_tensor,
            1,
            output_names,
            1
        );

        auto& output_tensor = output_tensors[0];
        float* output_data = output_tensor.GetTensorMutableData<float>();
        auto output_shape = output_tensor.GetTensorTypeAndShapeInfo().GetShape();

        int rows = static_cast<int>(output_shape[1]);
        int cols = static_cast<int>(output_shape[2]);
        
        cv::Mat result(rows, cols, CV_32F, output_data);
        return result.clone();

    } catch (const std::exception& e) {
        std::cerr << "Inference error: " << e.what() << std::endl;
        return cv::Mat();
    }
}
