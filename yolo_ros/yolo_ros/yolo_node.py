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
from typing import List, Dict, Optional
from cv_bridge import CvBridge

import rclpy
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from rclpy.lifecycle import LifecycleNode
from rclpy.lifecycle import TransitionCallbackReturn
from rclpy.lifecycle import LifecycleState

import torch
from ultralytics import YOLO, YOLOWorld, YOLOE
from ultralytics.engine.results import Results
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Masks
from ultralytics.engine.results import Keypoints

from std_srvs.srv import SetBool
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from yolo_msgs.msg import Point2D
from yolo_msgs.msg import BoundingBox2D
from yolo_msgs.msg import Mask
from yolo_msgs.msg import KeyPoint2D
from yolo_msgs.msg import KeyPoint2DArray
from yolo_msgs.msg import Detection
from yolo_msgs.msg import DetectionArray
from yolo_msgs.srv import SetClasses

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    GST_AVAILABLE = True
except (ImportError, ValueError):
    Gst = None
    GST_AVAILABLE = False


class RockchipMppJpegDecoder:
    def __init__(self, timeout_ms: int = 40) -> None:
        if not GST_AVAILABLE:
            raise RuntimeError("python3-gi / GStreamer bindings are not available")

        Gst.init(None)

        pipeline_description = (
            "appsrc name=src is-live=true block=true format=time do-timestamp=true "
            "! image/jpeg "
            "! jpegparse "
            "! mppjpegdec "
            "! videoconvert "
            "! video/x-raw,format=BGR "
            "! appsink name=sink sync=false max-buffers=1 drop=true"
        )

        self.pipeline = Gst.parse_launch(pipeline_description)
        self.src = self.pipeline.get_by_name("src")
        self.sink = self.pipeline.get_by_name("sink")

        if self.src is None or self.sink is None:
            raise RuntimeError("failed to initialize mppjpegdec pipeline")

        self.src.set_property("caps", Gst.Caps.from_string("image/jpeg"))
        self.timeout_ns = int(timeout_ms * 1e6)

        state_change = self.pipeline.set_state(Gst.State.PLAYING)
        if state_change == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("failed to set mppjpegdec pipeline to PLAYING state")

    def decode(self, jpeg_data: bytes) -> Optional[np.ndarray]:
        gst_buffer = Gst.Buffer.new_allocate(None, len(jpeg_data), None)
        gst_buffer.fill(0, jpeg_data)

        flow_ret = self.src.emit("push-buffer", gst_buffer)
        if flow_ret != Gst.FlowReturn.OK:
            raise RuntimeError(f"push-buffer failed: {flow_ret}")

        sample = self.sink.emit("try-pull-sample", self.timeout_ns)
        if sample is None:
            return None

        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))

        buffer = sample.get_buffer()
        success, map_info = buffer.map(Gst.MapFlags.READ)
        if not success:
            return None

        try:
            image = np.frombuffer(map_info.data, dtype=np.uint8)
            image = image.reshape((height, width, 3)).copy()
        finally:
            buffer.unmap(map_info)

        return image

    def close(self) -> None:
        if hasattr(self, "pipeline") and self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)


class YoloNode(LifecycleNode):

    def __init__(self) -> None:
        super().__init__("yolo_node")

        # params
        self.declare_parameter("model_type", "YOLO")
        self.declare_parameter("model", "yolov8m.pt")
        self.declare_parameter("device", "cuda:0")
        self.declare_parameter("yolo_encoding", "bgr8")
        self.declare_parameter("enable", True)
        self.declare_parameter("max_process_rate_hz", 0.0)
        self.declare_parameter("image_reliability", QoSReliabilityPolicy.BEST_EFFORT)
        self.declare_parameter("image_is_compressed", False)
        self.declare_parameter("compressed_decode_backend", "auto")
        self.declare_parameter("mpp_decode_timeout_ms", 40)

        self.declare_parameter("threshold", 0.5)
        self.declare_parameter("confidence", 0.5)  # legacy
        self.declare_parameter("iou", 0.5)
        self.declare_parameter("imgsz_height", 640)
        self.declare_parameter("imgsz_width", 640)
        self.declare_parameter("half", False)
        self.declare_parameter("max_det", 300)
        self.declare_parameter("augment", False)
        self.declare_parameter("agnostic_nms", False)
        self.declare_parameter("retina_masks", False)        
        self.declare_parameter("wanted_classes", [0])


        self.type_to_model = {"YOLO": YOLO, "World": YOLOWorld, "YOLOE": YOLOE}
        self._compressed_cb_lock = threading.Lock()
        self._compressed_drop_counter = 0
        self._no_client_skip_counter = 0
        self._rate_limit_skip_counter = 0

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Configuring...")

        # model params
        self.model_type = (
            self.get_parameter("model_type").get_parameter_value().string_value
        )
        self.model = self.get_parameter("model").get_parameter_value().string_value
        self.device = self.get_parameter("device").get_parameter_value().string_value
        self.yolo_encoding = (
            self.get_parameter("yolo_encoding").get_parameter_value().string_value
        )

        # inference params
        self.threshold = (
            self.get_parameter("threshold").get_parameter_value().double_value
        )
        self.confidence = (
            self.get_parameter("confidence").get_parameter_value().double_value
        )  # legacy
        self.iou = self.get_parameter("iou").get_parameter_value().double_value
        self.imgsz_height = (
            self.get_parameter("imgsz_height").get_parameter_value().integer_value
        )
        self.imgsz_width = (
            self.get_parameter("imgsz_width").get_parameter_value().integer_value
        )
        self.half = self.get_parameter("half").get_parameter_value().bool_value
        self.max_det = self.get_parameter("max_det").get_parameter_value().integer_value
        self.augment = self.get_parameter("augment").get_parameter_value().bool_value
        self.agnostic_nms = (
            self.get_parameter("agnostic_nms").get_parameter_value().bool_value
        )
        self.retina_masks = (
            self.get_parameter("retina_masks").get_parameter_value().bool_value
        )

        # ros params
        self.enable = self.get_parameter("enable").get_parameter_value().bool_value
        self.max_process_rate_hz = (
            self.get_parameter("max_process_rate_hz").get_parameter_value().double_value
        )
        self.reliability = (
            self.get_parameter("image_reliability").get_parameter_value().integer_value
        )
        self.image_is_compressed = (
            self.get_parameter("image_is_compressed").get_parameter_value().bool_value
        )
        self.compressed_decode_backend = (
            self.get_parameter("compressed_decode_backend")
            .get_parameter_value()
            .string_value
            .lower()
        )
        self.mpp_decode_timeout_ms = (
            self.get_parameter("mpp_decode_timeout_ms")
            .get_parameter_value()
            .integer_value
        )
        self._mpp_decoder: Optional[RockchipMppJpegDecoder] = None
        self._warned_unsupported_compressed_encoding = False
        self._last_process_time_ns: Optional[int] = None
        if self.max_process_rate_hz > 0.0:
            self._min_process_period_ns = max(
                1, int(1e9 / self.max_process_rate_hz)
            )
        else:
            self._min_process_period_ns = 0

        self.wanted_classes = (
            self.get_parameter("wanted_classes").get_parameter_value().integer_array_value
        )

        # detection pub
        self.image_qos_profile = QoSProfile(
            reliability=self.reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._pub = self.create_lifecycle_publisher(DetectionArray, "detections", 10)
        self.cv_bridge = CvBridge()

        super().on_configure(state)
        self.get_logger().info(f"[{self.get_name()}] Configured")

        return TransitionCallbackReturn.SUCCESS

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Activating...")

        try:
            self.yolo = self.type_to_model[self.model_type](self.model)
        except FileNotFoundError:
            self.get_logger().error(f"Model file '{self.model}' does not exists")
            return TransitionCallbackReturn.ERROR



        self._enable_srv = self.create_service(SetBool, "enable", self.enable_cb)
        self._enable_state_pub = self.create_publisher(
            Bool,
            "enable_state",
            QoSProfile(
                reliability=QoSReliabilityPolicy.RELIABLE,
                history=QoSHistoryPolicy.KEEP_LAST,
                durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
                depth=1,
            ),
        )
        self._publish_enable_state()

        if isinstance(self.yolo, YOLOWorld):
            self._set_classes_srv = self.create_service(
                SetClasses, "set_classes", self.set_classes_cb
            )

        if self.image_is_compressed:
            self._setup_compressed_decoder()
            self._sub = self.create_subscription(
                CompressedImage,
                "image_raw",
                self.compressed_image_cb,
                self.image_qos_profile,
            )
        else:
            self._sub = self.create_subscription(
                Image, "image_raw", self.image_cb, self.image_qos_profile
            )

        super().on_activate(state)
        self.get_logger().info(f"[{self.get_name()}] Activated")

        return TransitionCallbackReturn.SUCCESS

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Deactivating...")

        is_world_model = isinstance(self.yolo, YOLOWorld)

        del self.yolo
        if "cuda" in self.device:
            self.get_logger().info("Clearing CUDA cache")
            torch.cuda.empty_cache()

        self.destroy_service(self._enable_srv)
        self._enable_srv = None

        self.destroy_publisher(self._enable_state_pub)
        self._enable_state_pub = None

        if is_world_model:
            self.destroy_service(self._set_classes_srv)
            self._set_classes_srv = None

        self.destroy_subscription(self._sub)
        self._sub = None

        if self._mpp_decoder is not None:
            self._mpp_decoder.close()
            self._mpp_decoder = None

        super().on_deactivate(state)
        self.get_logger().info(f"[{self.get_name()}] Deactivated")

        return TransitionCallbackReturn.SUCCESS

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Cleaning up...")

        self.destroy_publisher(self._pub)

        del self.image_qos_profile

        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Cleaned up")

        return TransitionCallbackReturn.SUCCESS

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.get_logger().info(f"[{self.get_name()}] Shutting down...")
        super().on_cleanup(state)
        self.get_logger().info(f"[{self.get_name()}] Shutted down")
        return TransitionCallbackReturn.SUCCESS

    def enable_cb(
        self,
        request: SetBool.Request,
        response: SetBool.Response,
    ) -> SetBool.Response:
        self.enable = request.data
        self._publish_enable_state()
        response.success = True
        return response

    def _publish_enable_state(self) -> None:
        msg = Bool()
        msg.data = self.enable
        self._enable_state_pub.publish(msg)

    def _setup_compressed_decoder(self) -> None:
        if self.compressed_decode_backend not in {"auto", "cpu", "vpu_mpp"}:
            self.get_logger().warn(
                "Invalid compressed_decode_backend. Valid values: auto, cpu, vpu_mpp. Falling back to auto."
            )
            self.compressed_decode_backend = "auto"

        if self.compressed_decode_backend in {"auto", "vpu_mpp"}:
            try:
                self._mpp_decoder = RockchipMppJpegDecoder(self.mpp_decode_timeout_ms)
                self.get_logger().info(
                    "Using RK3588 VPU JPEG decoder backend: mppjpegdec"
                )
                return
            except Exception as exc:
                self.get_logger().warn(
                    f"Unable to initialize VPU JPEG decoder ({exc}). Falling back to CPU decoder."
                )

        self._mpp_decoder = None
        self.get_logger().info("Using CPU JPEG decoder backend: cv2.imdecode")

    def _decode_compressed_image(self, msg: CompressedImage) -> Optional[np.ndarray]:
        image = None
        jpeg_data = bytes(msg.data)

        if self._mpp_decoder is not None:
            try:
                image = self._mpp_decoder.decode(jpeg_data)
            except Exception as exc:
                self.get_logger().warn(
                    f"VPU JPEG decode failed ({exc}). Switching to CPU decoder."
                )
                self._mpp_decoder.close()
                self._mpp_decoder = None

        if image is None:
            np_data = np.frombuffer(jpeg_data, dtype=np.uint8)
            image = cv2.imdecode(np_data, cv2.IMREAD_COLOR)

        if image is None:
            self.get_logger().warn("Failed to decode compressed image")
            return None

        if self.yolo_encoding == "bgr8":
            return image
        if self.yolo_encoding == "rgb8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.yolo_encoding == "mono8":
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Fall back to cv_bridge if a custom encoding is requested.
        if not self._warned_unsupported_compressed_encoding:
            self.get_logger().warn(
                f"Compressed decoding currently supports bgr8/rgb8/mono8 only. "
                f"Falling back to cv_bridge for encoding '{self.yolo_encoding}'."
            )
            self._warned_unsupported_compressed_encoding = True
        return self.cv_bridge.compressed_imgmsg_to_cv2(
            msg, desired_encoding=self.yolo_encoding
        )

    def _has_result_clients(self) -> bool:
        return self._pub.get_subscription_count() > 0

    def _skip_if_no_clients(self) -> bool:
        if self._has_result_clients():
            self._no_client_skip_counter = 0
            return False

        self._no_client_skip_counter += 1
        if self._no_client_skip_counter % 120 == 0:
            self.get_logger().info(
                "No subscribers on 'detections'; skipping image processing"
            )
        return True

    def _skip_if_rate_limited(self) -> bool:
        if self._min_process_period_ns <= 0:
            return False

        now_ns = self.get_clock().now().nanoseconds
        if self._last_process_time_ns is None:
            self._last_process_time_ns = now_ns
            self._rate_limit_skip_counter = 0
            return False

        elapsed_ns = now_ns - self._last_process_time_ns
        if elapsed_ns < 0:
            self._last_process_time_ns = now_ns
            self._rate_limit_skip_counter = 0
            return False

        if elapsed_ns < self._min_process_period_ns:
            self._rate_limit_skip_counter += 1
            if self._rate_limit_skip_counter % 120 == 0:
                self.get_logger().info(
                    f"Rate-limiting image processing to {self.max_process_rate_hz:.2f} Hz"
                )
            return True

        self._last_process_time_ns = now_ns
        self._rate_limit_skip_counter = 0
        return False

    def parse_hypothesis(self, results: Results) -> List[Dict]:

        hypothesis_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:
                hypothesis = {
                    "class_id": int(box_data.cls),
                    "class_name": self.yolo.names[int(box_data.cls)],
                    "score": float(box_data.conf),
                }
                hypothesis_list.append(hypothesis)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                hypothesis = {
                    "class_id": int(results.obb.cls[i]),
                    "class_name": self.yolo.names[int(results.obb.cls[i])],
                    "score": float(results.obb.conf[i]),
                }
                hypothesis_list.append(hypothesis)

        return hypothesis_list

    def parse_boxes(self, results: Results) -> List[BoundingBox2D]:

        boxes_list = []

        if results.boxes:
            box_data: Boxes
            for box_data in results.boxes:

                msg = BoundingBox2D()

                # get boxes values
                box = box_data.xywh[0]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        elif results.obb:
            for i in range(results.obb.cls.shape[0]):
                msg = BoundingBox2D()

                # get boxes values
                box = results.obb.xywhr[i]
                msg.center.position.x = float(box[0])
                msg.center.position.y = float(box[1])
                msg.center.theta = float(box[4])
                msg.size.x = float(box[2])
                msg.size.y = float(box[3])

                # append msg
                boxes_list.append(msg)

        return boxes_list

    def parse_masks(self, results: Results) -> List[Mask]:

        masks_list = []

        def create_point2d(x: float, y: float) -> Point2D:
            p = Point2D()
            p.x = x
            p.y = y
            return p

        mask: Masks
        for mask in results.masks:

            msg = Mask()

            msg.data = [
                create_point2d(float(ele[0]), float(ele[1]))
                for ele in mask.xy[0].tolist()
            ]
            msg.height = results.orig_img.shape[0]
            msg.width = results.orig_img.shape[1]

            masks_list.append(msg)

        return masks_list

    def parse_keypoints(self, results: Results) -> List[KeyPoint2DArray]:

        keypoints_list = []

        points: Keypoints
        for points in results.keypoints:

            msg_array = KeyPoint2DArray()

            if points.conf is None:
                continue

            for kp_id, (p, conf) in enumerate(zip(points.xy[0], points.conf[0])):

                if conf >= self.threshold:
                    msg = KeyPoint2D()

                    msg.id = kp_id + 1
                    msg.point.x = float(p[0])
                    msg.point.y = float(p[1])
                    msg.score = float(conf)

                    msg_array.data.append(msg)

            keypoints_list.append(msg_array)

        return keypoints_list

    def image_cb(self, msg: Image) -> None:
        if not self.enable:
            return
        if self._skip_if_no_clients():
            return
        if self._skip_if_rate_limited():
            return

        cv_image = self.cv_bridge.imgmsg_to_cv2(msg, desired_encoding=self.yolo_encoding)
        self.process_image(cv_image, msg.header)

    def compressed_image_cb(self, msg: CompressedImage) -> None:
        if not self.enable:
            return
        if self._skip_if_no_clients():
            return
        if self._skip_if_rate_limited():
            return

        # Drop incoming compressed frames while the previous one is still running.
        if not self._compressed_cb_lock.acquire(blocking=False):
            self._compressed_drop_counter += 1
            if self._compressed_drop_counter % 60 == 0:
                self.get_logger().warn(
                    "Dropping compressed frame because previous callback is still processing"
                )
            return

        try:
            cv_image = self._decode_compressed_image(msg)
            if cv_image is None:
                return

            self.process_image(cv_image, msg.header)
        finally:
            self._compressed_cb_lock.release()

    def process_image(self, cv_image, header) -> None:
        results = self.yolo(cv_image)
        results: Results = results[0].cpu()

        filtered_indices = None
        hypothesis = []
        boxes = []
        masks = []
        keypoints = []

        # Early exit if no wanted classes detected
        if self.wanted_classes and len(self.wanted_classes) > 0:
            if results.boxes or results.obb:

                has_wanted_class = False
                for box_data in (results.boxes if results.boxes else []):
                    if int(box_data.cls) in self.wanted_classes:
                        has_wanted_class = True
                        break

                if not has_wanted_class:
                    detections_msg = DetectionArray()
                    detections_msg.header = header
                    self._pub.publish(detections_msg)
                    del results
                    return
            else:
                detections_msg = DetectionArray()
                detections_msg.header = header
                self._pub.publish(detections_msg)
                del results
                return

        if results.boxes or results.obb:
            hypothesis = self.parse_hypothesis(results)
            boxes = self.parse_boxes(results)

            # Filter detections by wanted classes
            if self.wanted_classes and len(self.wanted_classes) > 0:
                filtered_indices = []
                for i, hyp in enumerate(hypothesis):
                    if (
                        hyp["class_id"] in self.wanted_classes
                        and hyp["score"] >= self.confidence
                    ):
                        filtered_indices.append(i)

                hypothesis = [hypothesis[i] for i in filtered_indices]
                boxes = [boxes[i] for i in filtered_indices]

        if results.masks:
            masks = self.parse_masks(results)
            if (
                self.wanted_classes
                and len(self.wanted_classes) > 0
                and filtered_indices is not None
            ):
                masks = [masks[i] for i in filtered_indices]

        if results.keypoints:
            keypoints = self.parse_keypoints(results)
            if (
                self.wanted_classes
                and len(self.wanted_classes) > 0
                and filtered_indices is not None
            ):
                keypoints = [keypoints[i] for i in filtered_indices]

        # create detection msgs
        detections_msg = DetectionArray()

        num_detections = 0
        if (results.boxes or results.obb) and hypothesis and boxes:
            num_detections = len(hypothesis)
        elif results.masks and masks:
            num_detections = len(masks)
        elif results.keypoints and keypoints:
            num_detections = len(keypoints)

        for i in range(num_detections):
            aux_msg = Detection()

            if (results.boxes or results.obb) and hypothesis and boxes:
                aux_msg.class_id = hypothesis[i]["class_id"]
                aux_msg.class_name = hypothesis[i]["class_name"]
                aux_msg.score = hypothesis[i]["score"]
                aux_msg.bbox = boxes[i]

            if results.masks and masks:
                aux_msg.mask = masks[i]

            if results.keypoints and keypoints:
                aux_msg.keypoints = keypoints[i]

            detections_msg.detections.append(aux_msg)

        detections_msg.header = header
        self._pub.publish(detections_msg)

        del results

    def set_classes_cb(
        self,
        req: SetClasses.Request,
        res: SetClasses.Response,
    ) -> SetClasses.Response:
        self.get_logger().info(f"Setting classes: {req.classes}")
        self.yolo.set_classes(req.classes)
        self.get_logger().info(f"New classes: {self.yolo.names}")
        return res


def main():
    rclpy.init()
    node = YoloNode()
    node.trigger_configure()
    node.trigger_activate()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
