#include "preprocess.h"

Preprocessor::Preprocessor(int input_width, int input_height)
    : input_width_(input_width), input_height_(input_height) {}

cv::Mat Preprocessor::process(const cv::Mat& image) {
    if (image.empty()) {
        return cv::Mat();
    }

    float scale = std::min(static_cast<float>(input_width_) / image.cols, 
                          static_cast<float>(input_height_) / image.rows);
    
    int new_width = static_cast<int>(image.cols * scale);
    int new_height = static_cast<int>(image.rows * scale);
    
    int pad_x = (input_width_ - new_width) / 2;
    int pad_y = (input_height_ - new_height) / 2;
    
    scale_ = scale;
    padding_ = cv::Point(pad_x, pad_y);

    cv::Mat resized;
    cv::resize(image, resized, cv::Size(new_width, new_height));
    
    cv::Mat padded = cv::Mat::zeros(input_height_, input_width_, CV_8UC3);
    cv::Rect roi(pad_x, pad_y, new_width, new_height);
    resized.copyTo(padded(roi));
    
    cv::Mat normalized;
    padded.convertTo(normalized, CV_32F, 1.0 / 255.0);
    
    cv::Mat rgb;
    cv::cvtColor(normalized, rgb, cv::COLOR_BGR2RGB);
    
    std::vector<cv::Mat> channels(3);
    cv::split(rgb, channels);
    
    std::vector<float> blob_data;
    blob_data.reserve(3 * input_height_ * input_width_);
    
    for (int c = 0; c < 3; c++) {
        blob_data.insert(blob_data.end(), 
                        channels[c].ptr<float>(), 
                        channels[c].ptr<float>() + channels[c].total());
    }
    
    cv::Mat blob(blob_data.size(), 1, CV_32F, blob_data.data());
    return blob.clone();
}

std::pair<float, cv::Point> Preprocessor::getScaleAndPadding() const {
    return std::make_pair(scale_, padding_);
}
