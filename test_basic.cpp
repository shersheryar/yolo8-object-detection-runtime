#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <onnxruntime_cxx_api.h>

int main() {
    std::cout << "Testing basic ONNX Runtime functionality..." << std::endl;
    
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestApp");
        std::cout << "ONNX Runtime environment created successfully" << std::endl;
        
        std::string model_path = "yolov8n.onnx";
        std::ifstream file(model_path);
        if (file.good()) {
            std::cout << "Model file found: " << model_path << std::endl;
            
            Ort::SessionOptions session_options;
            auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            std::cout << "Model loaded successfully!" << std::endl;
            
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = session->GetInputCount();
            std::cout << "Number of input nodes: " << num_input_nodes << std::endl;
            
            if (num_input_nodes > 0) {
                auto input_name = session->GetInputNameAllocated(0, allocator);
                std::cout << "Input name: " << input_name.get() << std::endl;
                
                auto input_type_info = session->GetInputTypeInfo(0);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                auto input_dims = input_tensor_info.GetShape();
                
                std::cout << "Input dimensions: ";
                for (size_t i = 0; i < input_dims.size(); i++) {
                    std::cout << input_dims[i];
                    if (i < input_dims.size() - 1) std::cout << "x";
                }
                std::cout << std::endl;
            }
            
            size_t num_output_nodes = session->GetOutputCount();
            std::cout << "Number of output nodes: " << num_output_nodes << std::endl;
            
            if (num_output_nodes > 0) {
                auto output_name = session->GetOutputNameAllocated(0, allocator);
                std::cout << "Output name: " << output_name.get() << std::endl;
                
                auto output_type_info = session->GetOutputTypeInfo(0);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_dims = output_tensor_info.GetShape();
                
                std::cout << "Output dimensions: ";
                for (size_t i = 0; i < output_dims.size(); i++) {
                    std::cout << output_dims[i];
                    if (i < output_dims.size() - 1) std::cout << "x";
                }
                std::cout << std::endl;
            }
            
        } else {
            std::cout << "Model file not found: " << model_path << std::endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    std::cout << "Basic ONNX Runtime test completed successfully!" << std::endl;
    return 0;
}
