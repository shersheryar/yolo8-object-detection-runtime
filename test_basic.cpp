#include <iostream>
#include <string>
#include <memory>
#include <fstream>
#include <onnxruntime_cxx_api.h>
using namespace std;

int main() {
    cout << "Testing basic ONNX Runtime functionality..." << endl;
    
    try {
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "TestApp");
        cout << "ONNX Runtime environment created successfully" << endl;
        
        string model_path = "yolov8n.onnx";
        ifstream file(model_path);
        if (file.good()) {
            cout << "Model file found: " << model_path << endl;
            
            Ort::SessionOptions session_options;
            auto session = std::make_unique<Ort::Session>(env, model_path.c_str(), session_options);
            cout << "Model loaded successfully!" << endl;
            
            Ort::AllocatorWithDefaultOptions allocator;
            size_t num_input_nodes = session->GetInputCount();
            cout << "Number of input nodes: " << num_input_nodes << endl;
            
            if (num_input_nodes > 0) {
                auto input_name = session->GetInputNameAllocated(0, allocator);
                cout << "Input name: " << input_name.get() << endl;
                
                auto input_type_info = session->GetInputTypeInfo(0);
                auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
                auto input_dims = input_tensor_info.GetShape();
                
                cout << "Input dimensions: ";
                for (size_t i = 0; i < input_dims.size(); i++) {
                    cout << input_dims[i];
                    if (i < input_dims.size() - 1) cout << "x";
                }
                cout << endl;
            }
            
            size_t num_output_nodes = session->GetOutputCount();
            cout << "Number of output nodes: " << num_output_nodes << endl;
            
            if (num_output_nodes > 0) {
                auto output_name = session->GetOutputNameAllocated(0, allocator);
                cout << "Output name: " << output_name.get() << endl;
                
                auto output_type_info = session->GetOutputTypeInfo(0);
                auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
                auto output_dims = output_tensor_info.GetShape();
                
                cout << "Output dimensions: ";
                for (size_t i = 0; i < output_dims.size(); i++) {
                    cout << output_dims[i];
                    if (i < output_dims.size() - 1) cout << "x";
                }
                cout << endl;
            }
            
        } else {
            cout << "Model file not found: " << model_path << endl;
            return 1;
        }
        
    } catch (const std::exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
    
    cout << "Basic ONNX Runtime test completed successfully!" << endl;
    return 0;
}
