#pragma once
#include <queue>
#include <mutex>
#include <condition_variable>
#include "opencv_minimal.h"

class FrameQueue {
public:
    explicit FrameQueue(size_t max_size = 10);
    ~FrameQueue();

    bool push(const cv::Mat& frame);

    bool pop(cv::Mat& frame);

    bool empty() const;
    size_t size() const;

    void close();

    bool isClosed() const;

private:
    mutable std::mutex mtx;
    std::condition_variable cv_push;
    std::condition_variable cv_pop;

    std::queue<cv::Mat> q;
    size_t max_size;
    bool closed;
};
