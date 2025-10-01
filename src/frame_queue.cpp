#include "../headers/frame_queue.h"
using namespace std;

FrameQueue::FrameQueue(size_t max_size)
    : max_size(max_size), closed(false) {}

FrameQueue::~FrameQueue() {
    close();
}

bool FrameQueue::push(const cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mtx);
    
    if (closed || max_size == 0) {
        return false;
    }
    
    cv_push.wait(lock, [this] { return q.size() < max_size || closed; });
    
    if (closed) {
        return false;
    }
    
    q.push(frame.clone());
    cv_pop.notify_one();
    return true;
}

bool FrameQueue::pop(cv::Mat& frame) {
    std::unique_lock<std::mutex> lock(mtx);
    
    cv_pop.wait(lock, [this] { return !q.empty() || closed; });
    
    if (q.empty() && closed) {
        return false;
    }
    
    frame = q.front();
    q.pop();
    cv_push.notify_one();
    return true;
}

bool FrameQueue::empty() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.empty();
}

size_t FrameQueue::size() const {
    std::lock_guard<std::mutex> lock(mtx);
    return q.size();
}

void FrameQueue::close() {
    std::lock_guard<std::mutex> lock(mtx);
    closed = true;
    cv_push.notify_all();
    cv_pop.notify_all();
}

bool FrameQueue::isClosed() const {
    std::lock_guard<std::mutex> lock(mtx);
    return closed;
}
