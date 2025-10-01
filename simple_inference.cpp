#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <vector>
#include <chrono>
#include <onnxruntime_cxx_api.h>

class SimpleMat {
public:
    int rows, cols, channels;
    float* data;
    
    SimpleMat(int r, int c, int ch) : rows(r), cols(c), channels(ch) {
        data = new float[r * c * ch];
    }
    
    ~SimpleMat() {
        delete[] data;
    }
    
    bool empty() const { return data == nullptr; }
    float* ptr() const { return data; }
    int total() const { return rows * cols * channels; }
    size_t elemSize() const { return sizeof(float); }
    
    SimpleMat clone() const {
        SimpleMat result(rows, cols, channels);
        memcpy(result.data, data, total() * elemSize());
        return result;
    }
    SimpleMat(const SimpleMat& other) : rows(other.rows), cols(other.cols), channels(other.channels) {
        data = new float[total()];
        memcpy(data, other.data, total() * elemSize());
    }
    
    SimpleMat& operator=(const SimpleMat& other) {
        if (this != &other) {
            delete[] data;
            rows = other.rows;
            cols = other.cols;
            channels = other.channels;
            data = new float[total()];
            memcpy(data, other.data, total() * elemSize());
        }
        return *this;
    }
};

class SimpleInferEngine {
public:
    SimpleInferEngine() : env_(ORT_LOGGING_LEVEL_WARNING, "SimpleInferEngine") {}
    
    bool loadModel(const std::string& model_path) {
        try {
            if (!std::ifstream(model_path).good()) {
                std::cerr << "Model file not found: " << model_path << std::endl;
                return false;
            }
            
            Ort::SessionOptions session_options;
            session_options.SetIntraOpNumThreads(1);
            session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
            
            session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options);
            
            std::cout << "Model loaded successfully: " << model_path << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading model: " << e.what() << std::endl;
            return false;
        }
    }
    
    SimpleMat infer(const SimpleMat& input_blob) {
        if (!session_) {
            std::cerr << "Model not loaded" << std::endl;
            return SimpleMat(0, 0, 0);
        }
        
        if (input_blob.empty()) {
            return SimpleMat(0, 0, 0);
        }
        
        try {
            Ort::AllocatorWithDefaultOptions allocator;
            auto input_name = session_->GetInputNameAllocated(0, allocator);
            auto output_name = session_->GetOutputNameAllocated(0, allocator);
            
            std::vector<int64_t> input_shape = {1, 3, 640, 640};
            size_t input_tensor_size = input_blob.total() * input_blob.elemSize();
            
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            auto input_tensor = Ort::Value::CreateTensor<float>(
                memory_info, 
                const_cast<float*>(input_blob.ptr()),
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
            
            int rows = static_cast<int>(output_shape[1]);  // 84
            int cols = static_cast<int>(output_shape[2]);  // 8400
            
            SimpleMat result(rows, cols, 1);
            memcpy(result.data, output_data, rows * cols * sizeof(float));
            
            return result;
            
        } catch (const std::exception& e) {
            std::cerr << "Inference error: " << e.what() << std::endl;
            return SimpleMat(0, 0, 0);
        }
    }
    
private:
    std::unique_ptr<Ort::Session> session_;
    Ort::Env env_;
};

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }
    
    std::string model_path = argv[1];
    
    std::cout << "Simple YOLOv8 Inference Test" << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    
    SimpleInferEngine engine;
    if (!engine.loadModel(model_path)) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    SimpleMat input(640, 640, 3);
    
    for (int i = 0; i < input.total(); i++) {
        input.data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    
    std::cout << "Running inference..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    SimpleMat output = engine.infer(input);
    auto end = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (output.empty()) {
        std::cerr << "Inference failed" << std::endl;
        return 1;
    }
    
    std::cout << "Inference successful!" << std::endl;
    std::cout << "Output shape: " << output.rows << "x" << output.cols << std::endl;
    std::cout << "Inference time: " << duration.count() << " ms" << std::endl;
    
    float max_conf = 0.0f;
    for (int i = 0; i < output.total(); i++) {
        if (output.data[i] > max_conf) {
            max_conf = output.data[i];
        }
    }
    std::cout << "Max confidence: " << max_conf << std::endl;
    
    return 0;
}
