# CPU-Optimized Real-Time Object Tracking System (YOLOv8 & C++ ONNX Runtime)

This project demonstrates a high-performance, real-time computer vision pipeline optimized specifically for execution on a CPU-only environment. It achieves high frame rates (30+ FPS) by employing a Hybrid Tracking Strategy, separating the slow object Detection (using YOLOv8n) from the fast frame-to-frame Tracking (using simplified persistence).

# Architecture: The Asynchronous Handshake

The core innovation is running two processes at different frequencies to maximize CPU efficiency:

Detection (Slow Path, ~5 FPS): The lightweight YOLOv8n model runs periodically (e.g., every 30 frames) to perform accurate object localization and classification. This happens via the C++ ONNX Runtime.

Tracking (Fast Path, 30+ FPS): In between detections, the system holds and draws the last known bounding boxes, providing a smooth, responsive visual output without burning CPU cycles on continuous deep learning inference. (Note: For production, this would be replaced with a traditional C++ tracker like OpenCV's KCF.)

## Setup and Installation Guide

You will need a C++ compiler, CMake, and the OpenCV and ONNX Runtime libraries.

## Step 1: Python Model Preparation

The Python script is a one-time step to create the necessary model asset.

Install Python Dependencies:

pip install ultralytics onnx onnxruntime


Run the Exporter:
Execute the provided export_model.py script.

python export_model.py


This generates the file yolov8n.onnx, which is required by the C++ application. Place this file in your main project directory.

## Step 2: C++ Library Installation

Choose your operating system:

### A. Linux (Ubuntu / Debian)

Install Prerequisites:

sudo apt update
sudo apt install libopencv-dev cmake g++ curl


Install ONNX Runtime:

Download the CPU-only release package (e.g., onnxruntime-linux-x64-1.x.x.tgz) from the ONNX Runtime Releases page.

Extract it to a known location, for example, your home directory:

tar -xvzf onnxruntime-linux-x64-1.16.3.tgz -C ~/


Crucial: Note the extraction path (e.g., /home/user/onnxruntime-linux-x64-1.16.3). You will update the CMakeLists.txt with this path later.

### B. macOS (using Homebrew)

Install Prerequisites & OpenCV:

brew install cmake opencv


Install ONNX Runtime:

Download the Universal binary package (e.g., osx-universal2-1.x.x.tgz) from the ONNX Runtime Releases page.

Extract it to a known location, for example, your home directory:

tar -xvzf onnxruntime-osx-universal2-1.16.3.tgz -C ~/


Crucial: Note the extraction path. You will update the CMakeLists.txt with this path later.

## Step 3: Crow Web Server Header

The Crow web framework is header-only.

Create an include directory in your project root:

mkdir include


Download the single-file header into that folder:

curl -L -o include/crow_all.h [https://github.com/CrowCpp/Crow/releases/latest/download/crow_all.h](https://github.com/CrowCpp/Crow/releases/latest/download/crow_all.h)


## Building and Running the C++ Application

### 1. Update CMakeLists.txt

Before building, you MUST edit the CMakeLists.txt file and update the path for the ONNX Runtime library to match your installation directory.

Look for this line and modify it:

# UPDATE THIS PATH to where you extracted onnxruntime-linux-x64-1.x.x
set(ONNXRUNTIME_DIR "/usr/local/include/onnxruntime") 
# Example change for a user on Linux:
# set(ONNXRUNTIME_DIR "/home/user/onnxruntime-linux-x64-1.16.3") 


### 2. Compile

Run these commands in your project root directory:

# Create and enter the build directory
mkdir build && cd build

# Configure the project
cmake ..

# Compile the application (adjust -j for your core count)
make -j4


### 3. Execution

The final executable will be located in the build folder.

./CpuTracker


### Stopping the Application:
Since this program uses an OpenCV window, standard Ctrl+C often doesn't work reliably. To terminate the program gracefully:

Click on the "CPU Object Tracker" window to give it focus.

Press the Esc key.

The application will exit, and the web server thread (running on port 8080) will be shut down.

## Key Implementation Details

### YOLOv8 Post-Processing (inference.cpp)

The C++ code manually implements the decoding of the YOLOv8 output tensor (which is in the shape [1, 84, 8400]). This requires transforming the normalized (cx, cy, w, h) coordinates back to pixel values and performing Non-Maximum Suppression (NMS), which is crucial for turning thousands of raw predictions into a few clean bounding boxes.

### CPU Efficiency

Fixed Input Shape (640x640): Using a static input size in the ONNX export allows the C++ runtime to pre-allocate memory, drastically speeding up the tensor preparation step.

Decoupled Pipeline: The high FPS (visualization/tracking) is entirely decoupled from the low FPS (detection), ensuring the camera feed never feels laggy, even if the AI is slow.

OpenCV dnn module: We utilize cv::dnn::blobFromImage to leverage OpenCV's highly optimized, vectorized functions for image preprocessing (normalization and channel rearrangement), which is much faster than doing it manually.