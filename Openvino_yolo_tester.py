import cv2
import numpy as np
from pathlib import Path
import time
import psutil
import GPUtil
from ultralytics import YOLO
import openvino as ov

# Function to get GPU load
def get_gpu_load():
    gpus = GPUtil.getGPUs()
    if gpus:
        return gpus[0].load * 100  # It is not working properly for now
    return 0

# Function to get CPU and RAM usage
def get_system_usage():
    cpu_usage = psutil.cpu_percent(interval=0.1)
    ram_usage = psutil.virtual_memory().percent
    return cpu_usage, ram_usage

# Initialize variables for tracking performance
start_time = time.time()
frame_count = 0
total_inference_time = 0
fps_list = []
cpu_usage_list = []
ram_usage_list = []
gpu_load_list = []

# Load the YOLO model. You can change YoloV5 model to YoloV5s, YoloV5m, YoloV5l, YoloV5x, YoloV5nu or other versions
model_name = "yolov5nu"
DET_MODEL_NAME = model_name
det_model = YOLO(f"{DET_MODEL_NAME}.pt")
det_model.to("cpu")


det_model_path = Path(f"{DET_MODEL_NAME}_openvino_model/{DET_MODEL_NAME}.xml")
if not det_model_path.exists():
    det_model.export(format="openvino", dynamic=True, half=True)

device = "GPU"
core = ov.Core()
det_ov_model = core.read_model(det_model_path)

ov_config = {}
if device != "CPU":
    det_ov_model.reshape({0: [1, 3, 640, 640]})
if "GPU" in device or ("AUTO" in device and "GPU" in core.available_devices):
    ov_config = {"GPU_DISABLE_WINOGRAD_CONVOLUTION": "YES"}
det_compiled_model = core.compile_model(det_ov_model, device, ov_config)

det_model = YOLO(det_model_path.parent, task="detect")
if det_model.predictor is None:
    custom = {"conf": 0.25, "batch": 1, "save": False, "mode": "predict"}
    args = {**det_model.overrides, **custom}
    det_model.predictor = det_model._smart_load("predictor")(overrides=args, _callbacks=det_model.callbacks)
    det_model.predictor.setup_model(model=det_model.model)
det_model.predictor.model.ov_compiled_model = det_compiled_model

# Open the video file
video_name = "test.mp4" # change it to your video file
cap = cv2.VideoCapture(video_name)
if not cap.isOpened():
    raise Exception(f"Error: Could not open video file {video_name}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))


max_frames = int(fps * 15)  # I used 15 seconds only, you can change it as needed
print(f"Processing first {max_frames} frames (1 minute) of the video...")

try:
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break  

        # Resize the frame to the model's expected input shape (640x640)
        resized_frame = cv2.resize(frame, (640, 640))

        inference_start_time = time.time()
        results = det_model.track(resized_frame, show=False)
        inference_time = time.time() - inference_start_time
        total_inference_time += inference_time
        current_fps = 1.0 / inference_time
        fps_list.append(current_fps)

        cpu_usage, ram_usage = get_system_usage()
        cpu_usage_list.append(cpu_usage)
        ram_usage_list.append(ram_usage)

        gpu_load = get_gpu_load()
        gpu_load_list.append(gpu_load)

        frame_count += 1

        print(f"Processed frame {frame_count}/{max_frames} | FPS: {current_fps:.2f} | Inference: {inference_time * 1000:.2f}ms")

except KeyboardInterrupt:
    print("Process interrupted by user.")

finally:
    cap.release()
    if frame_count > 0:
        average_fps = sum(fps_list) / len(fps_list)
        average_inference_time = total_inference_time / frame_count
        average_cpu_usage = sum(cpu_usage_list) / len(cpu_usage_list)
        average_ram_usage = sum(ram_usage_list) / len(ram_usage_list)
        average_gpu_load = sum(gpu_load_list) / len(gpu_load_list)

        # Print the results
        print("\nPerformance Metrics:")
        print(f"Average FPS: {average_fps:.2f}")
        print(f"Average Inference Time per Frame: {average_inference_time * 1000:.2f}ms")
        print(f"Average CPU Usage: {average_cpu_usage:.2f}%")
        print(f"Average RAM Usage: {average_ram_usage:.2f}%")
        print(f"Average GPU Load: {average_gpu_load:.2f}%")
    else:
        print("No frames processed.")
