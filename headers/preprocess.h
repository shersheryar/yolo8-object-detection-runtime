#pragma once
#include "opencv_minimal.h"
#include <string>
#include <vector>

class Preprocessor {
public:
    Preprocessor(int input_width = 640, int input_height = 640);
    cv::Mat process(const cv::Mat& image);
    std::pair<float, cv::Point> getScaleAndPadding() const;

private:
    int input_width_;
    int input_height_;

    float scale_;
    cv::Point padding_; 
};
