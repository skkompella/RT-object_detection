cmake_minimum_required(VERSION 3.17)
project(CpuObjectTracker)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 1. Setup OpenCV
set(OpenCV_DIR "/opt/homebrew/opt/opencv/lib/cmake/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} "/opt/homebrew/include")

# 2. Setup ONNX Runtime
# UPDATE THIS PATH to where you extracted onnxruntime-linux-x64-1.x.x
set(ONNXRUNTIME_DIR "your_path")
set(ONNXRUNTIME_LIB_DIR "your_path")

include_directories(${ONNXRUNTIME_DIR}/include)
link_directories(${ONNXRUNTIME_LIB_DIR})

# 3. Setup Crow (Web Server)
# Assumes crow_all.h is in the ./include folder or installed globally
include_directories(include)

# 4. Define Sources
set(SOURCES
    main.cpp
    inference.cpp
)

# 5. Create Executable
add_executable(CpuTracker ${SOURCES})

# 6. Link Libraries
target_link_libraries(CpuTracker 
    ${OpenCV_LIBS} 
    onnxruntime
    pthread
)

# Copy model file to build directory
configure_file(${CMAKE_SOURCE_DIR}/yolov8n.onnx ${CMAKE_BINARY_DIR}/yolov8n.onnx COPYONLY)