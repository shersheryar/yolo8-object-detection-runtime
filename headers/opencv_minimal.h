#ifndef OPENCV_MINIMAL_H
#define OPENCV_MINIMAL_H

#include <vector>
#include <memory>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

// Basic Mat class
class Mat {
public:
    int rows, cols, channels;
    int depth;
    int dims;
    int* size;
    void* data;
    
    Mat() : rows(0), cols(0), channels(0), depth(0), dims(0), size(nullptr), data(nullptr) {}
    Mat(int rows, int cols, int type) : rows(rows), cols(cols), channels(3), depth(0), dims(2), data(nullptr) {
        size = new int[2];
        size[0] = rows;
        size[1] = cols;
        int total = rows * cols * channels;
        data = new float[total];
    }
    
    ~Mat() {
        if (size) delete[] size;
        if (data) delete[] static_cast<float*>(data);
    }
    
    bool empty() const { return data == nullptr; }
    
    float* ptr(int row = 0) const {
        return static_cast<float*>(data) + row * cols * channels;
    }
    
    template<typename T>
    T* ptr(int row = 0) const {
        return static_cast<T*>(data) + row * cols * channels;
    }
    
    int total() const { return rows * cols * channels; }
    
    Mat clone() const {
        Mat result(rows, cols, 0);
        if (data) {
            int total_size = total() * sizeof(float);
            memcpy(result.data, data, total_size);
        }
        return result;
    }
    
    static Mat zeros(int rows, int cols, int type) {
        return Mat(rows, cols, type);
    }
    
    static Mat zeros(const Size& size, int type) {
        return Mat(size.height, size.width, type);
    }
    
    size_t elemSize() const { return sizeof(float); }
    
    Mat(int rows, int cols, int type, void* data) : rows(rows), cols(cols), channels(3), depth(0), dims(2), data(data) {
        size = new int[2];
        size[0] = rows;
        size[1] = cols;
    }
};

// Basic Point class
class Point {
public:
    int x, y;
    Point(int x = 0, int y = 0) : x(x), y(y) {}
};

// Basic Size class
class Size {
public:
    int width, height;
    Size(int w = 0, int h = 0) : width(w), height(h) {}
};

// Basic Rect class
class Rect {
public:
    int x, y, width, height;
    Rect(int x = 0, int y = 0, int w = 0, int h = 0) : x(x), y(y), width(w), height(h) {}
};

// Basic Rect2f class
class Rect2f {
public:
    float x, y, width, height;
    Rect2f(float x = 0, float y = 0, float w = 0, float h = 0) : x(x), y(y), width(w), height(h) {}
};

// Basic Scalar class
class Scalar {
public:
    float val[4];
    Scalar(float v0 = 0, float v1 = 0, float v2 = 0, float v3 = 0) {
        val[0] = v0; val[1] = v1; val[2] = v2; val[3] = v3;
    }
};

// Constants
const int CV_8UC3 = 0;
const int CV_32F = 1;
const int CV_FONT_HERSHEY_SIMPLEX = 0;

// Basic VideoCapture class
class VideoCapture {
public:
    bool isOpened() const { return false; }
    bool read(Mat& frame) { return false; }
    void open(int device) {}
    void open(const std::string& filename) {}
    void release() {}
};

// Basic functions
void imshow(const std::string& name, const Mat& mat) {
    std::cout << "Displaying image: " << name << " (" << mat.rows << "x" << mat.cols << ")" << std::endl;
}

int waitKey(int delay) { return -1; }
void destroyAllWindows() {}

void rectangle(Mat& img, const Rect& rect, const Scalar& color, int thickness) {
    std::cout << "Drawing rectangle at (" << rect.x << "," << rect.y << ") size " << rect.width << "x" << rect.height << std::endl;
}

void putText(Mat& img, const std::string& text, const Point& org, int fontFace, double fontScale, const Scalar& color, int thickness) {
    std::cout << "Drawing text: " << text << " at (" << org.x << "," << org.y << ")" << std::endl;
}

Size getTextSize(const std::string& text, int fontFace, double fontScale, int thickness, int* baseline) {
    if (baseline) *baseline = 0;
    return Size(text.length() * 10, 20); // Rough estimate
}

void resize(const Mat& src, Mat& dst, const Size& size) {
    dst = Mat(size.height, size.width, 0);
    std::cout << "Resizing from " << src.rows << "x" << src.cols << " to " << size.width << "x" << size.height << std::endl;
}

void cvtColor(const Mat& src, Mat& dst, int code) {
    dst = src.clone();
    std::cout << "Converting color space" << std::endl;
}

void split(const Mat& src, std::vector<Mat>& channels) {
    channels.resize(3);
    for (int i = 0; i < 3; i++) {
        channels[i] = Mat(src.rows, src.cols, 0);
    }
    std::cout << "Splitting channels" << std::endl;
}


// Color conversion constants
const int COLOR_BGR2RGB = 0;

// cv namespace
namespace cv {
    using ::Mat;
    using ::Point;
    using ::Size;
    using ::Rect;
    using ::Rect2f;
    using ::Scalar;
    using ::VideoCapture;
    
    // Constants
    const int CV_8UC3 = ::CV_8UC3;
    const int CV_32F = ::CV_32F;
    const int CV_FONT_HERSHEY_SIMPLEX = ::CV_FONT_HERSHEY_SIMPLEX;
    const int COLOR_BGR2RGB = ::COLOR_BGR2RGB;
    
    // Functions
    using ::imshow;
    using ::waitKey;
    using ::destroyAllWindows;
    using ::rectangle;
    using ::putText;
    using ::getTextSize;
    using ::resize;
    using ::cvtColor;
    using ::split;
}

#endif // OPENCV_MINIMAL_H
