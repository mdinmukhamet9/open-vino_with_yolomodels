# YOLO Object Detection with OpenVINO Optimization

This repository contains a Python script for performing real-time object detection using YOLO (You Only Look Once) models optimized with OpenVINO. The script processes video files, performs object detection and tracking, and provides performance metrics such as FPS, inference time, CPU/GPU usage, and RAM usage.

## Features

- **YOLO Models**: Supports YOLOv5, YOLOv8, and other YOLO variants.
- **OpenVINO Optimization**: Converts YOLO models to OpenVINO Intermediate Representation (IR) for improved performance on Intel hardware.
- **Performance Metrics**: Tracks and displays FPS, inference time, CPU usage, RAM usage, and GPU load.
- **Video Processing**: Processes video files and performs object detection and tracking.
- **Customizable**: Easily switch between different YOLO models, input resolutions, and devices (CPU/GPU).

## Requirements

- Python 3.8 or higher
- OpenVINO 2024.6 or higher
- Ultralytics YOLO library
- OpenCV
- psutil
- GPUtil
