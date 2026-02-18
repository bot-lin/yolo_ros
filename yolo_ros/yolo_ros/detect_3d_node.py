# Copyright (C) 2023 Miguel Ángel González Santamarta

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.


import cv2
import numpy as np
import threading
from typing import List, Tuple, Optional

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import message_filters
from cv_bridge import CvBridge
from tf2_ros.buffer import Buffer
from tf2_ros import TransformException
from tf2_ros.transform_listener import TransformListener

from sensor_msgs.msg import CameraInfo, Image, PointCloud2
from geometry_msgs.msg import TransformStamped
from std_msgs.msg import Bool, Header
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.msg import KeyPoint3D
from yolo_msgs.msg import KeyPoint3DArray
from yolo_msgs.msg import BoundingBox3D

from sensor_msgs_py import point_cloud2


class Detect3DNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("bbox3d_node")

        # parameters
        self.declare_parameter("target_frame", "base_link")
        self.declare_parameter("maximum_detection_threshold", 0.3)
        self.declare_parameter("depth_image_units_divisor", 1000)
        self.declare_parameter("enable", True)
        self.declare_parameter("auto_follow_yolo_enable", True)
        self.declare_parameter(
            "depth_image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )
        self.declare_parameter("depth_info_reliability", QoSReliabilityPolicy.BEST_EFFORT)

        # aux
        self.tf_buffer = Buffer()
        self.cv_bridge = CvBridge()
        self._warned_pointcloud_mismatch = False
        self._sync_lock = threading.Lock()
        self._processing_subscriptions_active = False

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        self.target_frame = (
            self.get_parameter("target_frame").get_parameter_value().string_value
        )
        self.maximum_detection_threshold = (
            self.get_parameter("maximum_detection_threshold")
            .get_parameter_value()
            .double_value
        )
        self.depth_image_units_divisor = (
            self.get_parameter("depth_image_units_divisor")
            .get_parameter_value()
            .integer_value
        )
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.auto_follow_yolo_enable = (
            self.get_parameter("auto_follow_yolo_enable")
            .get_parameter_value()
            .bool_value
        )
        dimg_reliability = (
            self.get_parameter("depth_image_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_image_qos_profile = QoSProfile(
            reliability=dimg_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        dinfo_reliability = (
            self.get_parameter("depth_info_reliability")
            .get_parameter_value()
            .integer_value
        )

        self.depth_info_qos_profile = QoSProfile(
            reliability=dinfo_reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # pubs
        self._pub = self.create_publisher(DetectionArray, "detections_3d", 10)
        self._pointcloud_pub = self.create_publisher(PointCloud2, "detections_pointcloud", 10)

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        self._enable_state_sub = None
        if self.auto_follow_yolo_enable:
            self._enable_state_sub = self.create_subscription(
                Bool,
                "enable_state",
                self._enable_state_cb,
                QoSProfile(
                    reliability=QoSReliabilityPolicy.RELIABLE,
                    history=QoSHistoryPolicy.KEEP_LAST,
                    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                    depth=1,
                ),
            )
        self._set_processing_enabled(self.enable)

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        if self._enable_state_sub is not None:
            self.destroy_subscription(self._enable_state_sub)
            self._enable_state_sub = None
        self._stop_processing_subscriptions()

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        del self.tf_listener

        self.destroy_publisher(self._pub)
        self.destroy_publisher(self._pointcloud_pub)

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def on_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        pointcloud_msg: PointCloud2,
        detections_msg: DetectionArray,
    ) -> None:
        if not self.enable:
            return

        new_detections_msg = DetectionArray()
        new_detections_msg.header = detections_msg.header
        new_detections_msg.detections, points = self.process_detections(
            depth_msg, depth_info_msg, pointcloud_msg, detections_msg
        )
        self._pub.publish(new_detections_msg)
        pointcloud_msg = self.create_pointcloud_msg(
            points, pointcloud_msg.header.stamp, self.target_frame
        )
        self._pointcloud_pub.publish(pointcloud_msg)

    def _enable_state_cb(self, msg: Bool) -> None:
        self._set_processing_enabled(msg.data)

    def _set_processing_enabled(self, enabled: bool) -> None:
        with self._sync_lock:
            if self.enable == enabled:
                if self.enable and not self._processing_subscriptions_active:
                    self._start_processing_subscriptions()
                elif not self.enable and self._processing_subscriptions_active:
                    self._stop_processing_subscriptions()
                return
            self.enable = enabled
            if self.enable:
                self.get_logger().info("3D processing enabled")
                self._start_processing_subscriptions()
            else:
                self.get_logger().info("3D processing disabled")
                self._stop_processing_subscriptions()

    def _start_processing_subscriptions(self) -> None:
        if self._processing_subscriptions_active:
            return

        self.depth_sub = message_filters.Subscriber(
            self, Image, "depth_image", qos_profile=self.depth_image_qos_profile
        )
        self.depth_info_sub = message_filters.Subscriber(
            self, CameraInfo, "depth_info", qos_profile=self.depth_info_qos_profile
        )
        self.pointcloud_sub = message_filters.Subscriber(self, PointCloud2, "pointcloud")
        self.detections_sub = message_filters.Subscriber(self, DetectionArray, "detections")

        self._synchronizer = message_filters.ApproximateTimeSynchronizer(
            (self.depth_sub, self.depth_info_sub, self.pointcloud_sub, self.detections_sub),
            10,
            0.5,
        )
        self._synchronizer.registerCallback(self.on_detections)
        self._processing_subscriptions_active = True

    def _stop_processing_subscriptions(self) -> None:
        if not self._processing_subscriptions_active:
            return

        self.destroy_subscription(self.depth_sub.sub)
        self.destroy_subscription(self.depth_info_sub.sub)
        self.destroy_subscription(self.pointcloud_sub.sub)
        self.destroy_subscription(self.detections_sub.sub)
        del self._synchronizer
        self._processing_subscriptions_active = False

    def process_detections(
        self,
        depth_msg: Image,
        depth_info_msg: CameraInfo,
        pointcloud_msg: PointCloud2,
        detections_msg: DetectionArray,
    ) -> Tuple[List[Detection], np.ndarray]:

        # check if there are detections
        if not detections_msg.detections:
            return [], np.empty((0, 3), dtype=np.float32)

        transform = self.get_transform(depth_info_msg.header.frame_id)

        if transform is None:
            return [], np.empty((0, 3), dtype=np.float32)

        new_detections = []
        aggregated_points: List[np.ndarray] = []
        depth_image = self.cv_bridge.imgmsg_to_cv2(
            depth_msg, desired_encoding="passthrough"
        )

        for detection in detections_msg.detections:
            conversion = self.convert_bb_to_3d(depth_image, depth_info_msg, detection)

            if conversion is None:
                continue

            bbox3d, z_reference = conversion

            new_detections.append(detection)

            bbox3d = Detect3DNode.transform_3d_box(bbox3d, transform[0], transform[1])
            bbox3d.frame_id = self.target_frame
            new_detections[-1].bbox3d = bbox3d

            if detection.keypoints.data:
                keypoints3d = self.convert_keypoints_to_3d(
                    depth_image, depth_info_msg, detection
                )
                keypoints3d = Detect3DNode.transform_3d_keypoints(
                    keypoints3d, transform[0], transform[1]
                )
                keypoints3d.frame_id = self.target_frame
                new_detections[-1].keypoints3d = keypoints3d

            detection_points = self.collect_detection_points(
                depth_image, depth_info_msg, pointcloud_msg, detection, z_reference
            )

            if detection_points.size:
                detection_points = Detect3DNode.transform_points(
                    detection_points, transform[0], transform[1]
                )
                aggregated_points.append(detection_points)

        if aggregated_points:
            stacked_points = np.vstack(aggregated_points).astype(np.float32)
        else:
            stacked_points = np.empty((0, 3), dtype=np.float32)

        return new_detections, stacked_points

    def convert_bb_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> Optional[Tuple[BoundingBox3D, float]]:

        center_x = int(detection.bbox.center.position.x)
        center_y = int(detection.bbox.center.position.y)
        size_x = int(detection.bbox.size.x)
        size_y = int(detection.bbox.size.y)

        if detection.mask.data:
            # crop depth image by mask
            mask_array = np.array(
                [[int(ele.x), int(ele.y)] for ele in detection.mask.data]
            )
            mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
            roi = cv2.bitwise_and(depth_image, depth_image, mask=mask)

        else:
            # crop depth image by the 2d BB
            u_min = max(center_x - size_x // 2, 0)
            u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
            v_min = max(center_y - size_y // 2, 0)
            v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

            roi = depth_image[v_min:v_max, u_min:u_max]

        roi = roi / self.depth_image_units_divisor  # convert to meters
        if not np.any(roi):
            return None

        # find the z coordinate on the 3D BB
        if detection.mask.data:
            roi = roi[roi > 0]
            bb_center_z_coord = np.median(roi)

        else:
            bb_center_z_coord = (
                depth_image[int(center_y)][int(center_x)] / self.depth_image_units_divisor
            )

        z_diff = np.abs(roi - bb_center_z_coord)
        mask_z = z_diff <= self.maximum_detection_threshold
        if not np.any(mask_z):
            return None

        roi = roi[mask_z]
        z_min, z_max = np.min(roi), np.max(roi)
        z = (z_max + z_min) / 2

        if z == 0:
            return None

        # project from image to world space
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (center_x - px) / fx
        y = z * (center_y - py) / fy
        w = z * (size_x / fx)
        h = z * (size_y / fy)

        # create 3D BB
        msg = BoundingBox3D()
        msg.center.position.x = x
        msg.center.position.y = y
        msg.center.position.z = z
        msg.size.x = w
        msg.size.y = h
        msg.size.z = float(z_max - z_min)

        return msg, float(bb_center_z_coord)

    def collect_detection_points(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        pointcloud_msg: PointCloud2,
        detection: Detection,
        z_reference: float,
    ) -> np.ndarray:

        height = pointcloud_msg.height
        width = pointcloud_msg.width

        if height <= 1 or width == 0:
            return self._collect_points_from_depth(
                depth_image, depth_info, detection, z_reference
            )

        if (
            depth_image.shape[0] != height
            or depth_image.shape[1] != width
        ):
            if not self._warned_pointcloud_mismatch:
                self.get_logger().warn(
                    "Point cloud dimensions do not match depth image, falling back to depth sampling."
                )
                self._warned_pointcloud_mismatch = True
            return self._collect_points_from_depth(
                depth_image, depth_info, detection, z_reference
            )

        if detection.mask.data:
            mask_array = np.array(
                [[int(ele.x), int(ele.y)] for ele in detection.mask.data]
            )
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
            rows, cols = np.nonzero(mask)
        else:
            center_x = int(detection.bbox.center.position.x)
            center_y = int(detection.bbox.center.position.y)
            size_x = int(detection.bbox.size.x)
            size_y = int(detection.bbox.size.y)

            u_min = max(center_x - size_x // 2, 0)
            u_max = min(center_x + size_x // 2, width - 1)
            v_min = max(center_y - size_y // 2, 0)
            v_max = min(center_y + size_y // 2, height - 1)

            if u_min >= u_max or v_min >= v_max:
                return np.empty((0, 3), dtype=np.float32)

            region_rows, region_cols = np.indices(
                (v_max - v_min, u_max - u_min), dtype=np.int32
            )
            rows = (region_rows + v_min).reshape(-1)
            cols = (region_cols + u_min).reshape(-1)

        if rows.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        uvs = list(zip(cols.tolist(), rows.tolist()))

        points_iter = point_cloud2.read_points(
            pointcloud_msg, field_names=("x", "y", "z"), skip_nans=False, uvs=uvs
        )
        points = np.array(list(points_iter), dtype=np.float32)

        if points.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        finite_mask = np.isfinite(points).all(axis=1)
        if not np.any(finite_mask):
            return np.empty((0, 3), dtype=np.float32)

        points = points[finite_mask]

        z_diff = np.abs(points[:, 2] - z_reference)
        valid_mask = z_diff <= self.maximum_detection_threshold
        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32)

        return points[valid_mask]

    def _collect_points_from_depth(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
        z_reference: float,
    ) -> np.ndarray:

        depth_scale = self.depth_image_units_divisor
        threshold = self.maximum_detection_threshold

        if detection.mask.data:
            mask_array = np.array(
                [[int(ele.x), int(ele.y)] for ele in detection.mask.data]
            )
            mask = np.zeros(depth_image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(mask_array, dtype=np.int32)], 255)
            rows, cols = np.nonzero(mask)
        else:
            center_x = int(detection.bbox.center.position.x)
            center_y = int(detection.bbox.center.position.y)
            size_x = int(detection.bbox.size.x)
            size_y = int(detection.bbox.size.y)

            u_min = max(center_x - size_x // 2, 0)
            u_max = min(center_x + size_x // 2, depth_image.shape[1] - 1)
            v_min = max(center_y - size_y // 2, 0)
            v_max = min(center_y + size_y // 2, depth_image.shape[0] - 1)

            if u_min >= u_max or v_min >= v_max:
                return np.empty((0, 3), dtype=np.float32)

            region_rows, region_cols = np.indices(
                (v_max - v_min, u_max - u_min), dtype=np.int32
            )
            rows = (region_rows + v_min).reshape(-1)
            cols = (region_cols + u_min).reshape(-1)

        depths = depth_image[rows, cols] / depth_scale
        positive_mask = depths > 0
        if not np.any(positive_mask):
            return np.empty((0, 3), dtype=np.float32)

        depths = depths[positive_mask]
        rows = rows[positive_mask]
        cols = cols[positive_mask]

        z_diff = np.abs(depths - z_reference)
        valid_mask = z_diff <= threshold
        if not np.any(valid_mask):
            return np.empty((0, 3), dtype=np.float32)

        depths = depths[valid_mask]
        rows = rows[valid_mask]
        cols = cols[valid_mask]

        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]

        x = depths * (cols - px) / fx
        y = depths * (rows - py) / fy

        points = np.stack((x, y, depths), axis=-1)
        return points.astype(np.float32)

    def convert_keypoints_to_3d(
        self,
        depth_image: np.ndarray,
        depth_info: CameraInfo,
        detection: Detection,
    ) -> KeyPoint3DArray:

        # build an array of 2d keypoints
        keypoints_2d = np.array(
            [[p.point.x, p.point.y] for p in detection.keypoints.data], dtype=np.int16
        )
        u = np.array(keypoints_2d[:, 1]).clip(0, depth_info.height - 1)
        v = np.array(keypoints_2d[:, 0]).clip(0, depth_info.width - 1)

        # sample depth image and project to 3D
        z = depth_image[u, v]
        k = depth_info.k
        px, py, fx, fy = k[2], k[5], k[0], k[4]
        x = z * (v - px) / fx
        y = z * (u - py) / fy
        points_3d = (
            np.dstack([x, y, z]).reshape(-1, 3) / self.depth_image_units_divisor
        )  # convert to meters

        # generate message
        msg_array = KeyPoint3DArray()
        for p, d in zip(points_3d, detection.keypoints.data):
            if not np.isnan(p).any():
                msg = KeyPoint3D()
                msg.point.x = p[0]
                msg.point.y = p[1]
                msg.point.z = p[2]
                msg.id = d.id
                msg.score = d.score
                msg_array.data.append(msg)

        return msg_array

    def get_transform(self, frame_id: str) -> Tuple[np.ndarray]:
        # transform position from image frame to target_frame
        rotation = None
        translation = None

        try:
            transform: TransformStamped = self.tf_buffer.lookup_transform(
                self.target_frame, frame_id, rclpy.time.Time()
            )

            translation = np.array(
                [
                    transform.transform.translation.x,
                    transform.transform.translation.y,
                    transform.transform.translation.z,
                ]
            )

            rotation = np.array(
                [
                    transform.transform.rotation.w,
                    transform.transform.rotation.x,
                    transform.transform.rotation.y,
                    transform.transform.rotation.z,
                ]
            )

            return translation, rotation

        except TransformException as ex:
            self.get_logger().error(f"Could not transform: {ex}")
            return None

    @staticmethod
    def transform_3d_box(
        bbox: BoundingBox3D,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> BoundingBox3D:

        # position
        position = (
            Detect3DNode.qv_mult(
                rotation,
                np.array(
                    [
                        bbox.center.position.x,
                        bbox.center.position.y,
                        bbox.center.position.z,
                    ]
                ),
            )
            + translation
        )

        bbox.center.position.x = position[0]
        bbox.center.position.y = position[1]
        bbox.center.position.z = position[2]

        # size
        size = Detect3DNode.qv_mult(
            rotation, np.array([bbox.size.x, bbox.size.y, bbox.size.z])
        )

        bbox.size.x = abs(size[0])
        bbox.size.y = abs(size[1])
        bbox.size.z = abs(size[2])

        return bbox

    @staticmethod
    def transform_3d_keypoints(
        keypoints: KeyPoint3DArray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> KeyPoint3DArray:

        for point in keypoints.data:
            position = (
                Detect3DNode.qv_mult(
                    rotation, np.array([point.point.x, point.point.y, point.point.z])
                )
                + translation
            )

            point.point.x = position[0]
            point.point.y = position[1]
            point.point.z = position[2]

        return keypoints

    @staticmethod
    def transform_points(
        points: np.ndarray,
        translation: np.ndarray,
        rotation: np.ndarray,
    ) -> np.ndarray:

        if points.size == 0:
            return points

        transformed = [
            Detect3DNode.qv_mult(rotation, point.astype(np.float64)) + translation
            for point in points
        ]

        return np.asarray(transformed, dtype=np.float32)

    @staticmethod
    def qv_mult(q: np.ndarray, v: np.ndarray) -> np.ndarray:
        q = np.array(q, dtype=np.float64)
        v = np.array(v, dtype=np.float64)
        qvec = q[1:]
        uv = np.cross(qvec, v)
        uuv = np.cross(qvec, uv)
        return v + 2 * (uv * q[0] + uuv)

    @staticmethod
    def create_pointcloud_msg(
        points: np.ndarray, stamp, frame_id: str
    ) -> PointCloud2:

        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id

        return point_cloud2.create_cloud_xyz32(header, points.tolist())


def main():
    rclpy.init()
    node = Detect3DNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
