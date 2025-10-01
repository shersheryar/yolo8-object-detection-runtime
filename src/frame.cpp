#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <vector>
#include <iomanip>
#include <chrono>
using namespace std;

// Include all the corrected and verified headers
#include "../headers/infer_engine.h"
#include "../headers/preprocess.h"
#include "../headers/nms.h"
#include "../headers/frame_queue.h"

// The producer function reads frames from a video source and pushes them into a queue.
void producer(FrameQueue& fq, const string& video_path, atomic<bool>& running) {
    cv::VideoCapture cap;
    
    // Open video source
    if (video_path == "0") {
        cap.open(0); // Webcam
    } else {
        cap.open(video_path);
    }
    
    if (!cap.isOpened()) {
        cerr << "Error: Could not open video source: " << video_path << endl;
        return;
    }
    
    cout << "Producer started. Reading from: " << video_path << endl;
    
    cv::Mat frame;
    int frame_count = 0;
    
    while (running.load() && cap.read(frame)) {
        if (frame.empty()) {
            cout << "End of video stream reached." << endl;
            break;
        }
        
        if (!fq.push(frame)) {
            cout << "Queue closed, producer stopping." << endl;
            break;
        }
        
        frame_count++;
        if (frame_count % 100 == 0) {
            cout << "Producer: Processed " << frame_count << " frames" << endl;
        }
        
        this_thread::sleep_for(chrono::milliseconds(1));
    }
    
    cap.release();
    cout << "Producer finished. Total frames processed: " << frame_count << endl;
}

// The consumer function takes frames from the queue and performs the full inference pipeline.
namespace {
struct Track {
    int id;
    cv::Rect2f box;
    float conf;
    int cls;
    int age;
    int lost;
    cv::Rect2f smooth;
};

float iouRect(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = max(a.x, b.x);
    float y1 = max(a.y, b.y);
    float x2 = min(a.x + a.width, b.x + b.width);
    float y2 = min(a.y + a.height, b.y + b.height);
    if (x2 <= x1 || y2 <= y1) return 0.0f;
    float inter = (x2 - x1) * (y2 - y1);
    float uni = a.width * a.height + b.width * b.height - inter;
    return uni > 0 ? inter / uni : 0.0f;
}
}

void consumer(FrameQueue& fq, InferEngine& engine, atomic<bool>& running,
              float conf_threshold, float nms_threshold)
{
    cout << "Consumer started. Confidence threshold: " << conf_threshold 
         << ", NMS threshold: " << nms_threshold << endl;
    
    Preprocessor preprocessor(engine.getInputWidth(), engine.getInputHeight());
    cv::Mat frame;
    int processed_count = 0;
    
    vector<string> class_names = {
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

    vector<int> allowed = {2, 3, 5, 7};
    vector<Track> tracks;
    int next_id = 1;
    const float alpha = 0.7f;
    const float match_iou = 0.4f;
    const float enter_conf = 0.5f;
    const float keep_conf = 0.3f;
    const int min_age_draw = 2;
    const int grace_lost = 3;
    cv::VideoWriter writer;
    bool writer_opened = false;
    
    while (running.load()) {
        if (!fq.pop(frame)) {
            if (fq.isClosed()) {
                cout << "Queue closed, consumer stopping." << endl;
                break;
            }
            continue;
        }
        
        if (frame.empty()) {
            continue;
        }
        
        cv::Mat blob = preprocessor.process(frame);
        if (blob.empty()) {
            continue;
        }
        
        cv::Mat predictions = engine.infer(blob);
        if (predictions.empty()) {
            continue;
        }
        
        vector<Detection> detections = postprocess(
            predictions, 
            frame.size(), 
            conf_threshold, 
            nms_threshold
        );

        vector<Detection> filtered;
        for (auto& d : detections) {
            if (find(allowed.begin(), allowed.end(), d.cls) != allowed.end()) {
                filtered.push_back(d);
            }
        }

        vector<int> det_assigned(filtered.size(), -1);
        vector<int> track_assigned(tracks.size(), -1);

        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            float best_iou = 0.0f; int best_j = -1;
            for (size_t j = 0; j < filtered.size(); ++j) {
                if (det_assigned[j] != -1) continue;
                float conf_gate = (tracks[ti].age > 0) ? keep_conf : enter_conf;
                if (filtered[j].conf < conf_gate) continue;
                if (filtered[j].cls != tracks[ti].cls) continue;
                float iou = iouRect(tracks[ti].box, filtered[j].box);
                if (iou > best_iou) { best_iou = iou; best_j = (int)j; }
            }
            if (best_j != -1 && best_iou >= match_iou) {
                det_assigned[best_j] = (int)ti;
                track_assigned[ti] = best_j;
            }
        }

        for (size_t j = 0; j < filtered.size(); ++j) {
            if (det_assigned[j] != -1) continue;
            if (filtered[j].conf < enter_conf) continue;
            Track t;
            t.id = next_id++;
            t.box = filtered[j].box;
            t.smooth = filtered[j].box;
            t.conf = filtered[j].conf;
            t.cls = filtered[j].cls;
            t.age = 0; t.lost = 0;
            tracks.push_back(t);
        }

        for (size_t ti = 0; ti < tracks.size(); ++ti) {
            int dj = track_assigned[ti];
            if (dj != -1) {
                auto& det = filtered[dj];
                cv::Rect2f b = tracks[ti].smooth;
                b.x = alpha * det.box.x + (1 - alpha) * b.x;
                b.y = alpha * det.box.y + (1 - alpha) * b.y;
                b.width = alpha * det.box.width + (1 - alpha) * b.width;
                b.height = alpha * det.box.height + (1 - alpha) * b.height;
                tracks[ti].box = det.box;
                tracks[ti].smooth = b;
                tracks[ti].conf = det.conf;
                tracks[ti].cls = det.cls;
                tracks[ti].age++;
                tracks[ti].lost = 0;
            } else {
                tracks[ti].lost++;
            }
        }

        tracks.erase(remove_if(tracks.begin(), tracks.end(), [&](const Track& t){ return t.lost > grace_lost; }), tracks.end());
        
        cv::Mat display_frame = frame.clone();
        for (const auto& t : tracks) {
            if (t.age < min_age_draw) continue;
            cv::Rect rect((int)t.smooth.x, (int)t.smooth.y, (int)t.smooth.width, (int)t.smooth.height);
            cv::rectangle(display_frame, rect, cv::Scalar(0, 255, 0), 2);
            string clsname = (t.cls >= 0 && t.cls < (int)class_names.size()) ? class_names[t.cls] : ("class_" + to_string(t.cls));
            string label = clsname + " " + to_string(t.conf).substr(0, 4) + " id=" + to_string(t.id);
            int baseline = 0;
            cv::Size ts = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
            cv::Point org(rect.x, max(0, rect.y - 5));
            if (org.y < ts.height) org.y = rect.y + ts.height + 5;
            cv::rectangle(display_frame, cv::Point(org.x, org.y - ts.height - 5), cv::Point(org.x + ts.width, org.y + baseline), cv::Scalar(0, 255, 0), -1);
            cv::putText(display_frame, label, org, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
        }

        if (!writer_opened) {
            int fourcc = cv::VideoWriter::fourcc('M','J','P','G');
            double fps = 30.0;
            writer.open("output.mp4", fourcc, fps, display_frame.size());
            writer_opened = writer.isOpened();
        }
        if (writer_opened) writer.write(display_frame);
        
        cv::imshow("YOLOv8 Object Detection", display_frame);
        
        char key = cv::waitKey(1) & 0xFF;
        if (key == 27) {
            cout << "ESC pressed, stopping..." << endl;
            running = false;
            break;
        }
        
        processed_count++;
        if (processed_count % 50 == 0) {
            cout << "Consumer: Processed " << processed_count << " frames" << endl;
        }
    }
    
    cv::destroyAllWindows();
    if (writer_opened) writer.release();
    cout << "Consumer finished. Total frames processed: " << processed_count << endl;
}
