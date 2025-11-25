import os
from ultralytics import YOLO

def export_yolo_to_onnx(model_name='yolov8n.pt', output_name='yolov8n.onnx'):
    """
    Downloads a YOLOv8 Nano model and exports it to ONNX format 
    optimized for CPU deployment.
    """
    print(f"Starting export for {model_name}...")
    
    # 1. Load the YOLOv8 Nano model
    # This will download the weights from GitHub if they don't exist locally.
    try:
        model = YOLO(model_name)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Export to ONNX
    # Arguments explained:
    # - format='onnx': The target format compatible with C++.
    # - opset=12: A stable ONNX version widely supported by C++ runtimes.
    # - simplify=True: Removes redundant graph operations to speed up inference.
    # - imgsz=640: Fixes the input layer to 640x640 pixels. Static shapes 
    #   are significantly faster on CPUs than dynamic shapes.
    try:
        print("Exporting model... this may take a minute.")
        success = model.export(format='onnx', opset=12, simplify=True, imgsz=640)
        
        if success:
            print(f"\nSUCCESS: Model exported to {output_name}")
            print("You can now use this .onnx file in your C++ application.")
        else:
            print("Export failed.")
            
    except Exception as e:
        print(f"Error during export: {e}")

if __name__ == "__main__":
    # Ensure we are in the correct directory or handle paths accordingly
    current_dir = os.getcwd()
    print(f"Working directory: {current_dir}")
    
    export_yolo_to_onnx()