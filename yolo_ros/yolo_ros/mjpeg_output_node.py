import cv2
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.qos import QoSHistoryPolicy
from rclpy.qos import QoSDurabilityPolicy
from rclpy.qos import QoSReliabilityPolicy
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class _ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True
    allow_reuse_address = True


class MjpegOutputNode(Node):
    def __init__(self) -> None:
        super().__init__("mjpeg_output_node")

        self.declare_parameter("input_topic", "yolo/dbg_image")
        self.declare_parameter("host", "0.0.0.0")
        self.declare_parameter("port", 8083)
        self.declare_parameter("jpeg_quality", 80)
        self.declare_parameter("max_rate_hz", 10.0)
        self.declare_parameter(
            "image_reliability", QoSReliabilityPolicy.BEST_EFFORT
        )

        self.input_topic = (
            self.get_parameter("input_topic").get_parameter_value().string_value
        )
        self.host = (
            self.get_parameter("host").get_parameter_value().string_value
        )
        self.port = (
            self.get_parameter("port").get_parameter_value().integer_value
        )
        self.jpeg_quality = (
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )
        self.max_rate_hz = (
            self.get_parameter("max_rate_hz").get_parameter_value().double_value
        )
        reliability = (
            self.get_parameter("image_reliability")
            .get_parameter_value()
            .integer_value
        )

        self._cv_bridge = CvBridge()
        self._latest_jpeg = None
        self._lock = threading.Lock()
        self._client_count = 0
        self._last_encode_time = 0.0
        self._min_interval = 1.0 / max(self.max_rate_hz, 0.1)

        image_qos = QoSProfile(
            reliability=reliability,
            history=QoSHistoryPolicy.KEEP_LAST,
            durability=QoSDurabilityPolicy.VOLATILE,
            depth=1,
        )

        self._sub = self.create_subscription(
            Image, self.input_topic, self._image_cb, image_qos
        )

        # Start HTTP server
        self._httpd = None
        self._http_thread = None
        self._start_http_server()

    def _start_http_server(self) -> None:
        node_ref = self

        class StreamHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path not in ("/", "/stream", "/stream.mjpg"):
                    self.send_response(404)
                    self.end_headers()
                    return

                with node_ref._lock:
                    node_ref._client_count += 1

                self.send_response(200)
                self.send_header(
                    "Cache-Control", "no-cache, no-store, must-revalidate"
                )
                self.send_header("Pragma", "no-cache")
                self.send_header("Expires", "0")
                self.send_header(
                    "Content-Type",
                    "multipart/x-mixed-replace; boundary=frame",
                )
                self.end_headers()

                try:
                    while True:
                        with node_ref._lock:
                            frame = node_ref._latest_jpeg
                        if frame is None:
                            time.sleep(0.02)
                            continue
                        try:
                            self.wfile.write(b"--frame\r\n")
                            self.wfile.write(b"Content-Type: image/jpeg\r\n")
                            self.wfile.write(
                                f"Content-Length: {len(frame)}\r\n\r\n".encode(
                                    "ascii"
                                )
                            )
                            self.wfile.write(frame)
                            self.wfile.write(b"\r\n")
                        except (BrokenPipeError, ConnectionResetError):
                            break
                        except Exception:
                            break
                        time.sleep(node_ref._min_interval)
                finally:
                    with node_ref._lock:
                        node_ref._client_count = max(
                            0, node_ref._client_count - 1
                        )

            def log_message(self, format, *args):
                return

        try:
            self._httpd = _ThreadedHTTPServer(
                (self.host, self.port), StreamHandler
            )
        except OSError as exc:
            self.get_logger().error(
                f"Failed to start MJPEG server on {self.host}:{self.port}: {exc}"
            )
            return

        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, daemon=True
        )
        self._http_thread.start()
        self.get_logger().info(
            f"MJPEG output stream: http://{self.host}:{self.port}/stream"
        )

    def _image_cb(self, msg: Image) -> None:
        # Skip encoding when no clients are connected
        with self._lock:
            if self._client_count <= 0:
                return

        # Rate limiting
        now = time.monotonic()
        if now - self._last_encode_time < self._min_interval:
            return

        try:
            cv_image = self._cv_bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().warning(f"cv_bridge conversion failed: {exc}")
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
        ok, encoded = cv2.imencode(".jpg", cv_image, encode_params)
        if not ok:
            return

        with self._lock:
            self._latest_jpeg = encoded.tobytes()
        self._last_encode_time = now

    def destroy_node(self):
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
        super().destroy_node()


def main():
    rclpy.init()
    node = MjpegOutputNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
