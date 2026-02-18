# ROS 2 Interface Spec for YOLO Processing and Outputs

This document is written for another AI agent that will build a ROS 2 node consuming YOLO results.

## Scope

- control YOLO processing on/off at runtime
- consume 2D/3D detections
- consume generated 3D pointcloud

Default namespace in launch is `yolo`. Replace `/yolo/...` if you launch with another namespace.

## Runtime Control Contract

### Service: enable / disable YOLO

- Name: `/yolo/enable`
- Type: `std_srvs/srv/SetBool`
- Request:
  - `data=true`: enable YOLO processing
  - `data=false`: disable YOLO processing

### State Topic: current enable state

- Name: `/yolo/enable_state`
- Type: `std_msgs/msg/Bool`
- Semantics:
  - `data=true`: YOLO processing enabled
  - `data=false`: YOLO processing disabled
- This topic is published as transient local (latched behavior), so late subscribers can get the last state.

### CLI commands

Enable:

```bash
ros2 service call /yolo/enable std_srvs/srv/SetBool "{data: true}"
```

Disable:

```bash
ros2 service call /yolo/enable std_srvs/srv/SetBool "{data: false}"
```

Read state once:

```bash
ros2 topic echo /yolo/enable_state --once
```

## Output Topics Contract

### 2D detections

- Name: `/yolo/detections`
- Type: `yolo_msgs/msg/DetectionArray`
- Message:
  - `header`: source image header
  - `detections[]`: list of `yolo_msgs/msg/Detection`

### 3D detections

- Name: `/yolo/detections_3d`
- Type: `yolo_msgs/msg/DetectionArray`
- Message:
  - `header`: forwarded from detections stream
  - `detections[]`: same base structure, with `bbox3d`/`keypoints3d` filled when available

### 3D points of detections

- Name: `/yolo/detections_pointcloud`
- Type: `sensor_msgs/msg/PointCloud2`
- Frame:
  - `header.frame_id` is `target_frame` parameter of `detect_3d_node` (default `base_link`)

## Detection Message Fields You Should Use

Type: `yolo_msgs/msg/Detection`

- `class_id` (`int32`)
- `class_name` (`string`)
- `score` (`float64`)
- `id` (`string`, tracking id when tracking is enabled)
- `bbox` (`yolo_msgs/BoundingBox2D`) in pixels
- `bbox3d` (`yolo_msgs/BoundingBox3D`) in meters
- `mask` (`yolo_msgs/Mask`) optional boundary points
- `keypoints` (`yolo_msgs/KeyPoint2DArray`) optional
- `keypoints3d` (`yolo_msgs/KeyPoint3DArray`) optional

Consumer rule: treat `mask`, `keypoints`, `bbox3d`, `keypoints3d` as optional per detection.

## Processing/Rate Behavior

- `yolo_node` has parameter `max_process_rate_hz`:
  - `0.0` => no internal rate limit
  - `>0.0` => YOLO callback processing is throttled to max Hz
- Effective 2D detection frequency is bounded by:
  - image topic rate
  - model throughput
  - `max_process_rate_hz`

## 3D Node Follow Behavior

- `detect_3d_node` parameters:
  - `enable` (initial state)
  - `auto_follow_yolo_enable` (default `True`)
- With `auto_follow_yolo_enable=True`, `detect_3d_node` follows `/yolo/enable_state`.
- When disabled, it stops heavy synchronized subscriptions (depth, camera info, pointcloud, detections), reducing CPU load.

## Launch Example

```bash
ros2 launch yolo_bringup yolo.launch.py \
  use_3d:=True \
  enable:=False \
  auto_follow_yolo_enable:=True \
  max_process_rate_hz:=5.0
```

## Recommended Consumer Node Logic

1. Subscribe to `/yolo/enable_state` and keep a local `enabled` flag.
2. Subscribe to `/yolo/detections` and/or `/yolo/detections_3d`.
3. Optionally subscribe to `/yolo/detections_pointcloud` for 3D point processing.
4. If `enabled` is false, skip downstream heavy compute in your node.
5. Handle empty detection arrays gracefully (`len(msg.detections) == 0`).

## Quick Verification Commands

```bash
ros2 topic info /yolo/detections
ros2 topic info /yolo/detections_3d
ros2 topic info /yolo/detections_pointcloud
ros2 topic hz /yolo/detections
ros2 topic hz /yolo/detections_3d
ros2 topic hz /yolo/detections_pointcloud
```
