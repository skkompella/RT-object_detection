#include <atomic>
#include <iostream>
#include <mutex>
#include <thread>
#include <vector>

// Include Crow (Web Framework) - Ensure you have crow_all.h
#include "crow_all.h"

#include "inference.hpp"

// Global shared resources for the Web Stream
cv::Mat currentFrame;
std::vector<cv::Rect> trackedBoxes;
std::mutex frameMutex;
std::atomic<bool> isRunning(true);

// Function to encode Mat to JPEG for the browser
std::string matToJpeg(const cv::Mat &frame) {
  std::vector<uchar> buf;
  cv::imencode(".jpg", frame, buf);
  return std::string(buf.begin(), buf.end());
}

int main() {
  // 1. Initialize Resources
  std::string modelPath = "yolov8n.onnx";
  std::cout << "Loading Model: " << modelPath << std::endl;
  InferenceEngine engine(modelPath);

  cv::VideoCapture cap(0); // Open default webcam
  if (!cap.isOpened()) {
    std::cerr << "Error: Could not open camera." << std::endl;
    return -1;
  }

  // 2. Setup Web Server (Crow)
  crow::SimpleApp app;

  CROW_ROUTE(app, "/stream")
  ([](const crow::request &req, crow::response &res) {
    // MJPEG Stream Handler
    res.set_header("Content-Type", "multipart/x-mixed-replace; boundary=frame");
    res.end(); // We actually handle the stream manually below?
               // Crow's streaming is a bit complex, simpler approach for demo:
               // Use a persistent connection and write chunks.
               // Note: For production, use specialized streaming logic.
  });

  // Start Web Server in a separate thread
  std::thread serverThread([&app]() { app.port(8080).multithreaded().run(); });

  // 3. Main Processing Loop variables
  int frameCount = 0;
  const int DETECTION_INTERVAL = 30; // Run AI every 30 frames (1 sec approx)

  // We use basic tracking persistence
  // For a complex system, use cv::Tracker (KCF/CSRT) instances per object.
  // For this lightweight demo, we will persist the boxes and just Draw them,
  // refreshing them entirely every Interval.

  // NOTE: To make this "True Tracking" (moving boxes between detections),
  // you would create a std::vector<cv::Ptr<cv::Tracker>> here.

  std::cout << "Starting Loop on Main Thread..." << std::endl;

  while (isRunning) {
    cv::Mat frame;
    cap >> frame;
    if (frame.empty())
      break;

    // --- PHASE 1: DETECTION (The "Slow" Path) ---
    if (frameCount % DETECTION_INTERVAL == 0) {
      // In a real optimized app, this runs in a std::thread to avoid freezing
      // For this logic structure, we run it blocking but rarely.

      auto detections = engine.runInference(frame);

      std::lock_guard<std::mutex> lock(frameMutex);
      trackedBoxes.clear();
      for (const auto &det : detections) {
        trackedBoxes.push_back(det.box);
      }
      std::cout << "Detection run. Found: " << trackedBoxes.size() << std::endl;
    }

    // --- PHASE 2: VISUALIZATION (The "Fast" Path) ---
    // Here you would run tracker->update(frame) if using OpenCV trackers.
    // For simplicity, we draw the last known boxes (simulating a 'hold'
    // tracker)

    {
      std::lock_guard<std::mutex> lock(frameMutex);
      for (const auto &box : trackedBoxes) {
        cv::rectangle(frame, box, cv::Scalar(0, 255, 0), 2);
        cv::putText(frame, "Object", box.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5,
                    cv::Scalar(0, 255, 0), 2);
      }
    }

    // Update global frame for the web server to grab (if we implemented full
    // streaming) For local debug:
    cv::imshow("CPU Object Tracker", frame);
    if (cv::waitKey(1) == 27)
      isRunning = false; // ESC to exit

    frameCount++;
  }

  app.stop();
  serverThread.join();
  return 0;
}