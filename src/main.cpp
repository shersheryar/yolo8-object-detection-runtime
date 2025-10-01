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
using namespace std;

std::atomic<bool> running(true);

void signalHandler(int signum) {
    cout << "\n[INFO] Received signal " << signum << ". Shutting down gracefully..." << endl;
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

void printUsage(const char* prog) {
    cout << "Usage: " << prog << " --model <path> [options]\n\n"
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
#ifdef _WIN32
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);
    signal(SIGINT, signalHandler);
#else
    signal(SIGINT, signalHandler);
    signal(SIGTERM, signalHandler);
#endif

    string model_path, video_path = "0";
    float conf_threshold = 0.25f, nms_threshold = 0.45f;
    size_t queue_size = 24;

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--video" && i + 1 < argc) video_path = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) conf_threshold = std::stof(argv[++i]);
        else if (arg == "--nms" && i + 1 < argc) nms_threshold = std::stof(argv[++i]);
        else if (arg == "--queue-size" && i + 1 < argc) queue_size = std::stoul(argv[++i]);
        else if (arg == "--help") { printUsage(argv[0]); return 0; }
    }

    if (model_path.empty()) {
        cerr << "Error: --model argument is required." << endl;
        printUsage(argv[0]);
        return 1;
    }

    InferEngine engine;
    if (!engine.loadModel(model_path)) {
        cerr << "Failed to load model: " << model_path << endl;
        return 1;
    }

    FrameQueue frame_queue(queue_size);
    
    cout << "Starting YOLOv8 Object Detection Pipeline..." << endl;
    cout << "Model: " << model_path << endl;
    cout << "Video: " << video_path << endl;
    cout << "Confidence threshold: " << conf_threshold << endl;
    cout << "NMS threshold: " << nms_threshold << endl;
    cout << "Queue size: " << queue_size << endl;
    cout << "Press ESC to stop..." << endl;

    std::thread producer_thread(producer, std::ref(frame_queue), video_path, std::ref(running));
    std::thread consumer_thread(consumer, std::ref(frame_queue), std::ref(engine), 
                               std::ref(running), conf_threshold, nms_threshold);

    producer_thread.join();
    consumer_thread.join();

    frame_queue.close();

    cout << "Pipeline completed successfully." << endl;
    return 0;
}