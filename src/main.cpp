#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <csignal>
#ifdef _WIN32
#include <windows.h>
#endif
#include "infer_engine.h"
#include "frame_queue.h"

std::atomic<bool> running(true);

void signalHandler(int signum) {
    std::cout << "\n[INFO] Received signal " << signum << ". Shutting down gracefully..." << std::endl;
    running = false;
}

#ifdef _WIN32
static BOOL WINAPI ConsoleCtrlHandler(DWORD ctrl_type) {
    switch (ctrl_type) {
        case CTRL_C_EVENT:
        case CTRL_BREAK_EVENT:
        case CTRL_CLOSE_EVENT:
        case CTRL_LOGOFF_EVENT:
        case CTRL_SHUTDOWN_EVENT:
            signalHandler(SIGINT);
            return TRUE;
        default:
            return FALSE;
    }
}
#endif

extern void producer(FrameQueue& fq, const std::string& video_path, std::atomic<bool>& running);

extern void consumer(FrameQueue& fq, InferEngine& engine, std::atomic<bool>& running,
                    float conf_threshold, float nms_threshold);

// --- Argument Parser and Main Execution Logic ---
void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " --model <path> [options]\n\n"
              << "A multi-threaded YOLOv8 object detection application.\n\n"
              << "Required Arguments:\n"
              << "  --model <path>     Path to the ONNX model file.\n\n"
              << "Optional Arguments:\n"
              << "  --video <path>     Path to video file or '0' for webcam. (Default: 0)\n"
              << "  --conf <float>     Confidence threshold for detections. (Default: 0.25)\n"
              << "  --nms <float>      NMS IoU threshold for filtering boxes. (Default: 0.45)\n"
              << "  --queue-size <int> Max number of frames to buffer. (Default: 24)\n"
              << "  --help             Show this help message.\n";
}

int main(int argc, char** argv) {
    // Set up signal handling for graceful shutdown (Ctrl+C).
#ifdef _WIN32
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
    signal(SIGINT, signalHandler);
#else
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
#endif

    // Default parameters
    std::string model_path, video_path = "0";
    float conf_threshold = 0.25f, nms_threshold = 0.45f;
    size_t queue_size = 24;

    // Parse command-line arguments
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf_threshold = std::stof(argv[++i]);
        else if (arg == "--nms" && i + 1 < argc) nms_threshold = std::stof(argv[++i]);
        else if (arg == "--queue-size" && i + 1 < argc) queue_size = std::stoul(argv[++i]);
        else if (arg == "--help") { printUsage(argv[0]); return 0; }
    }

    // Validate required arguments
    if (model_path.empty()) {
        std::cerr << "Error: --model argument is required." << std::endl;
        printUsage(argv[0]);
        return 1;
    }

    // Initialize inference engine
    InferEngine engine;
    if (!engine.loadModel(model_path)) {
        std::cerr << "Failed to load model: " << model_path << std::endl;
        return 1;
    }

    // Create frame queue
    FrameQueue frame_queue(queue_size);
    
    std::cout << "Starting YOLOv8 Object Detection Pipeline..." << std::endl;
    std::cout << "Model: " << model_path << std::endl;
    std::cout << "Video: " << video_path << std::endl;
    std::cout << "Confidence threshold: " << conf_threshold << std::endl;
    std::cout << "NMS threshold: " << nms_threshold << std::endl;
    std::cout << "Queue size: " << queue_size << std::endl;
    std::cout << "Press ESC to stop..." << std::endl;

    // Start producer and consumer threads
    std::thread producer_thread(producer, std::ref(frame_queue), video_path, std::ref(running));
    std::thread consumer_thread(consumer, std::ref(frame_queue), std::ref(engine), 
                               std::ref(running), conf_threshold, nms_threshold);

    // Wait for threads to complete
    producer_thread.join();
    consumer_thread.join();

    // Close frame queue
    frame_queue.close();

    std::cout << "Pipeline completed successfully." << std::endl;
    return 0;
}