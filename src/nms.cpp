#include "nms.h"
#include <vector>
#include <algorithm>
#include <cstdint>



float computeIoU(const cv::Rect2f& a, const cv::Rect2f& b) {
    float x1 = std::max(a.x, b.x);
    float y1 = std::max(a.y, b.y);
    float x2 = std::min(a.x + a.width, b.x + b.width);
    float y2 = std::min(a.y + a.height, b.y + b.height);
    
    if (x2 <= x1 || y2 <= y1) {
        return 0.0f;
    }
    
    float intersection_area = (x2 - x1) * (y2 - y1);
    float area_a = a.width * a.height;
    float area_b = b.width * b.height;
    float union_area = area_a + area_b - intersection_area;
    
    if (union_area <= 0.0f) {
        return 0.0f;
    }
    
    return intersection_area / union_area;
}

std::vector<Detection> postprocess(
    const cv::Mat& predictions,
    cv::Size original_image_size,
    float conf_threshold,
    float iou_threshold
) 
{
    std::vector<Detection> detections;
    
    if (predictions.empty()) {
        return detections;
    }
    
    int num_classes = predictions.rows - 4;
    int num_anchors = predictions.cols;
    
    for (int i = 0; i < num_anchors; i++) {
        float max_conf = 0.0f;
        int max_class = -1;
        
        for (int j = 4; j < predictions.rows; j++) {
            float conf = predictions.at<float>(j, i);
            if (conf > max_conf) {
                max_conf = conf;
                max_class = j - 4;
            }
        }
        
        if (max_conf < conf_threshold) {
            continue;
        }
        
        float center_x = predictions.at<float>(0, i);
        float center_y = predictions.at<float>(1, i);
        float width = predictions.at<float>(2, i);
        float height = predictions.at<float>(3, i);
        
        float x1 = center_x - width / 2.0f;
        float y1 = center_y - height / 2.0f;
        
        float scale_x = static_cast<float>(original_image_size.width) / 640.0f;
        float scale_y = static_cast<float>(original_image_size.height) / 640.0f;
        
        x1 *= scale_x;
        y1 *= scale_y;
        width *= scale_x;
        height *= scale_y;
        
        x1 = std::max(0.0f, std::min(x1, static_cast<float>(original_image_size.width)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(original_image_size.height)));
        width = std::min(width, static_cast<float>(original_image_size.width) - x1);
        height = std::min(height, static_cast<float>(original_image_size.height) - y1);
        
        if (width > 0 && height > 0) {
            Detection det;
            det.box = cv::Rect2f(x1, y1, width, height);
            det.conf = max_conf;
            det.cls = max_class;
            detections.push_back(det);
        }
    }
    
    if (detections.empty()) {
        return detections;
    }
    
    std::sort(detections.begin(), detections.end(), 
              [](const Detection& a, const Detection& b) {
                  return a.conf > b.conf;
              });
    
    std::vector<Detection> nms_detections;
    std::vector<bool> suppressed(detections.size(), false);
    
    for (size_t i = 0; i < detections.size(); i++) {
        if (suppressed[i]) {
            continue;
        }
        
        nms_detections.push_back(detections[i]);
        
        for (size_t j = i + 1; j < detections.size(); j++) {
            if (suppressed[j] || detections[i].cls != detections[j].cls) {
                continue;
            }
            
            float iou = computeIoU(detections[i].box, detections[j].box);
            if (iou > iou_threshold) {
                suppressed[j] = true;
            }
        }
    }
    
    return nms_detections;
}