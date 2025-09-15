# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a ROS 2 package that wraps YOLO models from Ultralytics for object detection, tracking, instance segmentation, human pose estimation, and 3D object detection. It supports multiple YOLO versions (YOLOv3-YOLOv12, YOLO-World, YOLOE) and provides both 2D and 3D detection capabilities using RGB and depth camera inputs.

## Package Structure

The repository contains three main ROS 2 packages:

- **yolo_msgs**: Message definitions for YOLO detection results
- **yolo_ros**: Core Python nodes (yolo_node, debug_node, tracking_node, detect_3d_node)
- **yolo_bringup**: Launch files and configuration for different YOLO models

## Build Commands

```bash
# Install Python dependencies
pip3 install -r requirements.txt

# Install ROS dependencies
rosdep install --from-paths src --ignore-src -r -y

# Build workspace
colcon build

# Build specific packages
colcon build --packages-select yolo_msgs
colcon build --packages-select yolo_ros yolo_bringup
```

## Launch Commands

### Standard Object Detection
```bash
# Generic launcher (uses yolov8m.pt by default)
ros2 launch yolo_bringup yolo.launch.py

# Specific model versions
ros2 launch yolo_bringup yolov8.launch.py
ros2 launch yolo_bringup yolov11.launch.py
ros2 launch yolo_bringup yolo-world.launch.py
ros2 launch yolo_bringup yoloe.launch.py
```

### 3D Detection
```bash
# Enable 3D detection with depth camera
ros2 launch yolo_bringup yolo.launch.py use_3d:=True

# 3D with segmentation masks
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-seg.pt use_3d:=True

# 3D human pose estimation
ros2 launch yolo_bringup yolo.launch.py model:=yolov8m-pose.pt use_3d:=True
```

## Key Architecture Components

### Core Nodes
- **yolo_node.py**: Main YOLO inference node, subscribes to camera images and publishes detections
- **tracking_node.py**: Object tracking using ByteTrack, assigns IDs to detected objects
- **debug_node.py**: Visualization node, creates debug images with bounding boxes and labels
- **detect_3d_node.py**: 3D detection node, uses depth images to create 3D bounding boxes

### Node Communication
- RGB images → yolo_node → /yolo/detections
- Detections → tracking_node → /yolo/tracking
- Tracking + RGB → debug_node → /yolo/debug_image
- Detections + Depth → detect_3d_node → /yolo/detections_3d

### Launch File Parameters
The main launch file `yolo_bringup/launch/yolo.launch.py` contains comprehensive parameter configuration for:
- Model selection (model, model_type, device)
- Detection thresholds (threshold, iou, max_det)
- Image processing (imgsz_height, imgsz_width, yolo_encoding)
- Topic configuration (input_image_topic, input_depth_topic, target_frame)
- Feature flags (use_tracking, use_3d, use_debug, enable)

## Docker Support

```bash
# Build Docker image
docker build -t yolo_ros .

# Run with GPU support (requires NVIDIA Container Toolkit)
docker run -it --rm --gpus all yolo_ros
```

## Development Notes

- Python package uses setuptools with entry points for each node
- Supports Lifecycle Nodes for resource management
- All models from Ultralytics are compatible (YOLOv3-v12, YOLO-World, YOLOE)
- 3D detection requires RGB-D camera setup
- Depth processing uses configurable units divisor (default: 1000 for millimeters)
- Tracking uses configurable tracker files (default: bytetrack.yaml)