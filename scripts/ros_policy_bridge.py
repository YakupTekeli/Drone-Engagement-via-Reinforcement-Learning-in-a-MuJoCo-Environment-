#!/usr/bin/env python3
"""ROS 2 köprü düğümü: Unreal Engine kamerası + uçuş durumu -> PPO politikası -> hız.

Bu düğüm, eğitilmiş Ebabil politikasını Unreal Engine (veya başka bir ROS 2
simülasyonu) içinde çalıştırır. İş akışı her politika adımında (varsayılan
`20 Hz`):

1. En son kamera karesini al (`sensor_msgs/Image`).
2. Kullanıcının YOLO modelini çalıştır (ağaç, tank); kutuları `obstacle`/`target`
   sınıflarına eşle.
3. `ebabil.deploy.PolicyRunner` ile `vision-v4` gözlemini kur ve politikayı çağır.
4. Gövde-çerçevesi (FLU, SI) hız komutunu `geometry_msgs/Twist` olarak yayınla.

Davranış (kullanıcı kararı): ağaçlardan kaçınarak tanka yönel, güvenli bir
**temassız** yakın mesafede dur. Hedefe temas/çarpma yoktur; standoff tabanı
`DeploymentConfig` içinde zorlanır.

Bu dosya ROS 2 (rclpy), cv_bridge ve ultralytics kurulu bir ortamda çalışır; bu
depoda bu bağımlılıklar yoktur. Topic adları ve kamera/standoff parametreleri
ROS parametreleriyle ayarlanır. Entegrasyon adımları:
`docs/UNREAL_ROS_ENTEGRASYON.md`.
"""

from __future__ import annotations

import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO

from ebabil.deploy.policy_runner import (
    DeploymentConfig,
    DronePose,
    PolicyRunner,
    camera_geometry_from_intrinsics,
    load_policy_predict_fn,
    quaternion_wxyz_to_matrix,
)
from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, Detection


class PolicyBridge(Node):
    def __init__(self) -> None:
        super().__init__("ebabil_policy_bridge")

        # --- Parametreler (ros2 param ile veya launch'ta ayarlanır) ---
        self.declare_parameter("policy_path", "")
        self.declare_parameter("yolo_weights", "")
        self.declare_parameter("image_topic", "/drone/camera/image_raw")
        self.declare_parameter("odometry_topic", "/drone/odom")
        self.declare_parameter("command_topic", "/drone/cmd_vel")
        self.declare_parameter("control_hz", 20.0)
        # Kamera: Unreal kamerasını eğitimdekiyle eşleyin (FOV 75, pitch 40).
        self.declare_parameter("horizontal_fov_deg", 75.0)
        self.declare_parameter("pitch_down_deg", 40.0)
        self.declare_parameter("camera_position_body_m", [0.10, 0.0, -0.01])
        # YOLO sınıf adları -> rol. Kendi modelinizin sınıf adlarını yazın.
        self.declare_parameter("obstacle_class_names", ["tree"])
        self.declare_parameter("target_class_names", ["tank"])
        self.declare_parameter("yolo_conf", 0.25)
        # Yaklaşık tank konumu (dünya, metre). ADR-019 waypoint girdisi.
        self.declare_parameter("believed_target_world_m", [10.0, 0.0, 0.0])
        # Durma mesafesi tankın YÜZEYİNE göre: target_radius_m tankın yaklaşık
        # yarı-boyutu, surface_clearance_m yüzeyden bırakılacak pay (temassız).
        self.declare_parameter("target_radius_m", 3.5)
        self.declare_parameter("surface_clearance_m", 0.5)

        policy_path = self.get_parameter("policy_path").value
        yolo_weights = self.get_parameter("yolo_weights").value
        if not policy_path or not yolo_weights:
            raise RuntimeError("policy_path ve yolo_weights parametreleri zorunludur")

        self._hfov = float(self.get_parameter("horizontal_fov_deg").value)
        self._pitch = float(self.get_parameter("pitch_down_deg").value)
        self._camera_position = tuple(
            float(v) for v in self.get_parameter("camera_position_body_m").value
        )
        self._obstacle_names = {
            str(n).lower() for n in self.get_parameter("obstacle_class_names").value
        }
        self._target_names = {
            str(n).lower() for n in self.get_parameter("target_class_names").value
        }
        self._yolo_conf = float(self.get_parameter("yolo_conf").value)
        self._believed_target = tuple(
            float(v) for v in self.get_parameter("believed_target_world_m").value
        )

        # Yüzey payı tabanı DeploymentConfig içinde ayrıca zorlanır (temassızlık).
        self._target_radius_m = float(self.get_parameter("target_radius_m").value)
        self._surface_clearance_m = float(
            self.get_parameter("surface_clearance_m").value
        )
        config = DeploymentConfig(
            target_radius_m=self._target_radius_m,
            surface_clearance_m=self._surface_clearance_m,
        )
        self._runner = PolicyRunner(load_policy_predict_fn(policy_path), config)
        self._detector = YOLO(yolo_weights)
        self._bridge = CvBridge()

        self._latest_image: np.ndarray | None = None
        self._latest_pose: DronePose | None = None
        self._camera_size: tuple[int, int] | None = None

        self.create_subscription(
            Image, self.get_parameter("image_topic").value, self._on_image, 1
        )
        self.create_subscription(
            Odometry, self.get_parameter("odometry_topic").value, self._on_odometry, 10
        )
        self._publisher = self.create_publisher(
            Twist, self.get_parameter("command_topic").value, 10
        )
        period = 1.0 / float(self.get_parameter("control_hz").value)
        self.create_timer(period, self._on_tick)
        self.get_logger().info("Ebabil politika köprüsü hazır")

    def _on_image(self, message: Image) -> None:
        # Politikanın beklediği RGB; cv_bridge ile dönüştür.
        self._latest_image = self._bridge.imgmsg_to_cv2(message, "rgb8")

    def _on_odometry(self, message: Odometry) -> None:
        position = message.pose.pose.position
        orientation = message.pose.pose.orientation  # ROS: x, y, z, w
        rotation = quaternion_wxyz_to_matrix(
            (orientation.w, orientation.x, orientation.y, orientation.z)
        )
        twist = message.twist.twist  # child_frame_id (base_link = gövde) varsayılır
        self._latest_pose = DronePose(
            position_world_m=(position.x, position.y, position.z),
            rotation_body_to_world=rotation,
            linear_velocity_body_m_s=(twist.linear.x, twist.linear.y, twist.linear.z),
            angular_velocity_body_rad_s=(
                twist.angular.x,
                twist.angular.y,
                twist.angular.z,
            ),
        )

    def _ensure_geometry(self, width: int, height: int) -> None:
        # Kamera geometrisini gerçek kare boyutuna göre bir kez kur (kutular bu
        # piksel uzayında). FOV ve pitch Unreal kamerasıyla eşleşmeli.
        if self._camera_size == (width, height):
            return
        self._runner.config = DeploymentConfig(
            target_radius_m=self._target_radius_m,
            surface_clearance_m=self._surface_clearance_m,
            camera=camera_geometry_from_intrinsics(
                output_width_px=width,
                output_height_px=height,
                horizontal_fov_deg=self._hfov,
                pitch_down_deg=self._pitch,
                position_body_m=self._camera_position,
            ),
        )
        self._camera_size = (width, height)

    def _detections_from_image(self, image: np.ndarray) -> list[Detection]:
        height, width = image.shape[:2]
        self._ensure_geometry(width, height)
        result = self._detector.predict(
            image[:, :, ::-1], conf=self._yolo_conf, verbose=False
        )[0]
        detections: list[Detection] = []
        names = result.names
        for box in result.boxes:
            name = str(names[int(box.cls)]).lower()
            if name in self._obstacle_names:
                class_id = OBSTACLE_CLASS_ID
            elif name in self._target_names:
                class_id = TARGET_CLASS_ID
            else:
                continue
            x_min, y_min, x_max, y_max = (float(v) for v in box.xyxy[0])
            detections.append(Detection(class_id, x_min, y_min, x_max, y_max))
        return detections

    def _on_tick(self) -> None:
        if self._latest_image is None or self._latest_pose is None:
            return
        detections = self._detections_from_image(self._latest_image)
        command = self._runner.step(
            pose=self._latest_pose,
            detections=detections,
            believed_target_world_m=self._believed_target,
        )
        message = Twist()
        message.linear.x = command.forward_m_s
        message.linear.y = command.left_m_s
        message.linear.z = command.up_m_s
        message.angular.z = command.yaw_rate_rad_s
        self._publisher.publish(message)


def main() -> None:
    rclpy.init()
    node = PolicyBridge()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
