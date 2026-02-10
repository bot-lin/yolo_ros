import cv2

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage


class MjpegStreamNode(Node):
    def __init__(self) -> None:
        super().__init__("mjpeg_stream_node")

        self.declare_parameter("stream_url", "http://127.0.0.1:8081/")
        self.declare_parameter("output_topic", "image_compressed")
        self.declare_parameter("frame_id", "camera_link")
        self.declare_parameter("jpeg_quality", 90)
        self.declare_parameter("reconnect_period_sec", 1.0)

        self.stream_url = self.get_parameter("stream_url").get_parameter_value().string_value
        self.output_topic = (
            self.get_parameter("output_topic").get_parameter_value().string_value
        )
        self.frame_id = self.get_parameter("frame_id").get_parameter_value().string_value
        self.jpeg_quality = (
            self.get_parameter("jpeg_quality").get_parameter_value().integer_value
        )
        self.reconnect_period_sec = (
            self.get_parameter("reconnect_period_sec").get_parameter_value().double_value
        )

        self.publisher = self.create_publisher(CompressedImage, self.output_topic, 10)
        self.cap = None
        self.last_reconnect_try = self.get_clock().now()

        self.connect_stream()
        self.timer = self.create_timer(0.001, self.read_and_publish)

    def connect_stream(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        self.get_logger().info(f"Connecting MJPEG stream: {self.stream_url}")
        self.cap = cv2.VideoCapture(self.stream_url)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open MJPEG stream")
            self.cap = None
        else:
            self.get_logger().info("MJPEG stream connected")

    def read_and_publish(self) -> None:
        if self.cap is None:
            now = self.get_clock().now()
            if (now - self.last_reconnect_try).nanoseconds >= int(
                self.reconnect_period_sec * 1e9
            ):
                self.last_reconnect_try = now
                self.connect_stream()
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.get_logger().warning("Read frame failed, reconnecting stream")
            self.connect_stream()
            return

        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            self.get_logger().warning("JPEG encode failed for current frame")
            return

        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self.frame_id
        msg.format = "jpeg"
        msg.data = encoded.tobytes()
        self.publisher.publish(msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    node = MjpegStreamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

