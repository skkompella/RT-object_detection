#include "inference.hpp"
#include <algorithm>

InferenceEngine::InferenceEngine(const std::string& modelPath) 
    : env(ORT_LOGGING_LEVEL_WARNING, "YoloInference"), 
      session(env, modelPath.c_str(), Ort::SessionOptions()) {
    
    // Define input/output names for YOLOv8
    // Note: These names must match the ONNX file structure. 
    // Usually "images" for input and "output0" for output in YOLOv8.
    // We allocate strings dynamically to ensure lifetime validity if needed, 
    // but literals work for standard exports.
    inputNodeNames = {"images"};
    outputNodeNames = {"output0"};
}

cv::Mat InferenceEngine::preprocess(const cv::Mat& image, float& scale) {
    cv::Mat resized;
    // Calculate scale factor to fit into 640x640 while maintaining aspect ratio
    float scaleX = (float)inputWidth / image.cols;
    float scaleY = (float)inputHeight / image.rows;
    scale = std::min(scaleX, scaleY);

    // Letterbox pad could be done here, but simple resize is faster for this demo
    // We strictly resize for speed, ignoring aspect ratio distortion slightly 
    // (Simpler for basic object tracking)
    cv::resize(image, resized, cv::Size(inputWidth, inputHeight));
    
    return resized;
}

std::vector<Detection> InferenceEngine::runInference(const cv::Mat& inputImage) {
    float scaleX = (float)inputImage.cols / inputWidth;
    float scaleY = (float)inputImage.rows / inputHeight;

    cv::Mat blob;
    // cv::dnn::blobFromImage handles the Math: (Image - Mean) / Std and HWC -> CHW swap
    // YOLO expects RGB, 0-1 normalized float data
    cv::dnn::blobFromImage(inputImage, blob, 1.0/255.0, cv::Size(inputWidth, inputHeight), cv::Scalar(), true, false);

    // Create Input Tensor
    // Blob is already in [1, 3, 640, 640] format continuous memory
    std::vector<int64_t> inputShape = {1, 3, inputHeight, inputWidth};
    size_t inputTensorSize = 1 * 3 * inputHeight * inputWidth;
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    
    Ort::Value inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo, (float*)blob.data, inputTensorSize, inputShape.data(), inputShape.size()
    );

    // Run Inference
    auto outputTensors = session.Run(
        Ort::RunOptions{nullptr}, 
        inputNodeNames.data(), 
        &inputTensor, 1, 
        outputNodeNames.data(), 1
    );

    // Post-Process Output
    // YOLOv8 Output shape is [1, 84, 8400] -> [Batch, (4 box + 80 classes), Proposals]
    float* rawOutput = outputTensors[0].GetTensorMutableData<float>();
    
    // We need to parse this 1D array. 
    // Stride is 8400 columns.
    // Rows 0-3 are Box (cx, cy, w, h)
    // Rows 4-83 are Class Probabilities
    
    std::vector<int> classIds;
    std::vector<float> confidences;
    std::vector<cv::Rect> boxes;
    
    int dimensions = 84; // 4 + 80
    int rows = 8400;     // Anchors

    // Iterating through the 8400 proposals
    for (int i = 0; i < rows; ++i) {
        float maxClassScore = 0.0f;
        int classId = -1;

        // Find best class for this proposal
        // The class scores start at index 4 of the column
        for (int c = 0; c < 80; ++c) {
             // Accessing matrix in column-major order equivalent logic
             // logic: rawOutput[row * rows + i] 
             // Note: YOLOv8 export usually is [1, 84, 8400]. 
             // Access logic: data[channel_index * 8400 + proposal_index]
             float score = rawOutput[(4 + c) * rows + i];
             if (score > maxClassScore) {
                 maxClassScore = score;
                 classId = c;
             }
        }

        if (maxClassScore > 0.45) { // Confidence Threshold
            // Extract Box
            float cx = rawOutput[0 * rows + i];
            float cy = rawOutput[1 * rows + i];
            float w  = rawOutput[2 * rows + i];
            float h  = rawOutput[3 * rows + i];

            // Convert center-width-height to top-left format
            int x = int((cx - w/2) * scaleX);
            int y = int((cy - h/2) * scaleY);
            int width = int(w * scaleX);
            int height = int(h * scaleY);

            boxes.push_back(cv::Rect(x, y, width, height));
            confidences.push_back(maxClassScore);
            classIds.push_back(classId);
        }
    }

    // Non-Maximum Suppression (NMS) to remove overlapping boxes
    std::vector<int> indices;
    cv::dnn::NMSBoxes(boxes, confidences, 0.45, 0.45, indices);

    std::vector<Detection> finalDetections;
    for (int idx : indices) {
        finalDetections.push_back({classIds[idx], confidences[idx], boxes[idx]});
    }

    return finalDetections;
}