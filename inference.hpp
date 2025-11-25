#pragma once

#include <opencv2/opencv.hpp>
#include <onnxruntime_cxx_api.h>
#include <vector>
#include <string>

struct Detection {
    int class_id;
    float confidence;
    cv::Rect box;
};

class InferenceEngine {
public:
    InferenceEngine(const std::string& modelPath);
    std::vector<Detection> runInference(const cv::Mat& inputImage);

private:
    // ONNX Runtime Resources
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    // Model specific parameters (YOLOv8n default)
    const int inputWidth = 640;
    const int inputHeight = 640;
    std::vector<const char*> inputNodeNames;
    std::vector<const char*> outputNodeNames;

    // Helper to preprocess image for YOLO (Resize -> Normalize -> HWC to CHW)
    cv::Mat preprocess(const cv::Mat& image, float& scale);
};