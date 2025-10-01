#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <iomanip>
#include <chrono>

// Include all the corrected and verified headers
#include "../headers/infer_engine.h"
#include "../headers/preprocess.h"
#include "../headers/nms.h"
#include "../headers/frame_queue.h"

// The producer function reads frames from a video source and pushes them into a queue.
void producer(FrameQueue& fq, const std::string& video_path, std::atomic<bool>& running) {
    cv::VideoCapture cap;
    
    // Open video source
    if (video_path == "0") {
        cap.open(0); // Webcam
    } else {
        cap.open(video_path);
    }
    
    if (!cap.isOpened()) {
        std::cerr << "Error: Could not open video source: " << video_path << std::endl;
        return;
    }
    
    std::cout << "Producer started. Reading from: " << video_path << std::endl;
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (running.load() && cap.read(frame)) {
        if (frame.empty()) {
            std::cout << "End of video stream reached." << std::endl;
            break;
        }
        
        // Push frame to queue
        if (!fq.push(frame)) {
            std::cout << "Queue closed, producer stopping." << std::endl;
            break;
        }
        
        frame_count++;
        if (frame_count % 100 == 0) {
            std::cout << "Producer: Processed " << frame_count << " frames" << std::endl;
        }
        
        // Small delay to prevent overwhelming the system
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    
    cap.release();
    std::cout << "Producer finished. Total frames processed: " << frame_count << std::endl;
}

// The consumer function takes frames from the queue and performs the full inference pipeline.
void consumer(FrameQueue& fq, InferEngine& engine, std::atomic<bool>& running,
              float conf_threshold, float nms_threshold)
{
    std::cout << "Consumer started. Confidence threshold: " << conf_threshold 
              << ", NMS threshold: " << nms_threshold << std::endl;
    
    Preprocessor preprocessor(engine.getInputWidth(), engine.getInputHeight());
    cv::Mat frame;
    int processed_count = 0;
    
    // COCO class names for display
    std::vector<std::string> class_names = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
        "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
        "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
        "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
        "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
        "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop",
        "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
        "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    
    while (running.load()) {
        if (!fq.pop(frame)) {
            if (fq.isClosed()) {
                std::cout << "Queue closed, consumer stopping." << std::endl;
                break;
            }
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }
        
        // Preprocess frame
        cv::Mat blob = preprocessor.process(frame);
        if (blob.empty()) {
            continue;
        }
        
        // Run inference
        cv::Mat predictions = engine.infer(blob);
        if (predictions.empty()) {
            continue;
        }
        
        // Postprocess results
        std::vector<Detection> detections = postprocess(
            predictions, 
            frame.size(), 
            conf_threshold, 
            nms_threshold
        );
        
        // Draw detections on frame
        cv::Mat display_frame = frame.clone();
        for (const auto& det : detections) {
            // Draw bounding box
            cv::Rect rect(static_cast<int>(det.box.x), static_cast<int>(det.box.y),
                         static_cast<int>(det.box.width), static_cast<int>(det.box.height));
            cv::rectangle(display_frame, rect, cv::Scalar(0, 255, 0), 2);
            
            // Draw label
            std::string label;
            if (det.cls < class_names.size()) {
                label = class_names[det.cls] + " " + std::to_string(det.conf).substr(0, 4);
            } else {
                label = "class_" + std::to_string(det.cls) + " " + std::to_string(det.conf).substr(0, 4);
            }
            
            int baseline = 0;
            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point text_org(rect.x, rect.y - 5);
            if (text_org.y < text_size.height) {
                text_org.y = rect.y + text_size.height + 5;
            }
            
            cv::rectangle(display_frame, 
                         cv::Point(text_org.x, text_org.y - text_size.height - 5),
                         cv::Point(text_org.x + text_size.width, text_org.y + baseline),
                         cv::Scalar(0, 255, 0), -1);
            cv::putText(display_frame, label, text_org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }
        
        // Display frame
        cv::imshow("YOLOv8 Object Detection", display_frame);
        
        // Check for ESC key
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) { // ESC key
            std::cout << "ESC pressed, stopping..." << std::endl;
            running = false;
            break;
        }
        
        processed_count++;
        if (processed_count % 50 == 0) {
            std::cout << "Consumer: Processed " << processed_count << " frames, "
                      << "detections: " << detections.size() << std::endl;
        }
    }
    
    cv::destroyAllWindows();
    std::cout << "Consumer finished. Total frames processed: " << processed_count << std::endl;
}
