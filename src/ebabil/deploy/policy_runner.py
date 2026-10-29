"""Çatı-bağımsız politika köprüsü çekirdeği (Unreal/ROS dağıtımı).

İki sorumluluk:

1. `build_observation`: dedektör kutuları + uçuş durumu + yaklaşık hedef
   konumundan `vision-v4` (88 sayı) gözlemini kurar. Kurulum, eğitim
   ortamındaki (`src/ebabil/envs/vision.py` ve `detection_vision.py`) ile
   BİREBİR aynı fonksiyonları ve ölçekleri kullanır.
2. `command`: politikanın `[-1, 1]^4` action'ını gövde-çerçevesi hız komutuna
   çevirir ve eğitimdeki deploy edilebilir kısıtları (ADR-024 kamera konisi +
   koridor hız sınırlayıcı) uygular. Hedefe yaklaşıldığında temassız bir
   "yaklaş ve dur" davranışı uygular (bkz. modül başlığı).

Çerçeveler. Politika gövde çerçevesi FLU'dur (`+x` ileri, `+y` sol, `+z`
yukarı); dünya sağ-elli, `+z` yukarı. Bu, ROS REP-103 gövde (base_link) ve ENU
dünya kuralıyla aynıdır, bu yüzden ROS tarafında ek işaret çevirisi gerekmez.
Unreal'in sol-elli, santimetre çerçevesine dönüşüm, mevcut Pixhawk-Unreal
köprüsünün (ROS <-> UE) işidir; bu çekirdek yalnız SI + FLU üretir.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from ebabil.envs.motion_constraint import (
    MotionConstraintConfig,
    SpeedGovernorConfig,
    apply_speed_limit,
    constrain_horizontal_velocity,
    governor_speed_limit,
)
from ebabil.perception.detection_slots import (
    CameraGeometry,
    DetectionSlotConfig,
    SlotEncoding,
    encode_detection_slots,
)
from ebabil.perception.detections import Detection

FloatArray = npt.NDArray[np.float64]

#: Politikanın `model.predict` imzasıyla uyumlu tahmin fonksiyonu.
PredictFn = Callable[[dict[str, npt.NDArray[np.float32]]], npt.NDArray[np.float64]]


def quaternion_wxyz_to_matrix(quaternion_wxyz: Sequence[float]) -> FloatArray:
    """`[w, x, y, z]` birim kuaterniyonundan gövde->dünya rotasyon matrisi.

    ROS'tan gelen kuaterniyon genelde `[x, y, z, w]` sırasındadır; bu
    fonksiyona vermeden önce `[w, x, y, z]` sırasına getirin.
    """

    w, x, y, z = (float(v) for v in quaternion_wxyz)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1e-12:
        raise ValueError("Sıfır normlu kuaterniyon")
    w, x, y, z = (v / norm for v in (w, x, y, z))
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def camera_geometry_from_intrinsics(
    *,
    output_width_px: int,
    output_height_px: int,
    horizontal_fov_deg: float,
    pitch_down_deg: float,
    position_body_m: tuple[float, float, float],
) -> CameraGeometry:
    """Kamera içsel parametrelerinden menzil geometrisi.

    Kutular bu çözünürlüğün piksel uzayında verilmelidir. Eğitimdeki değerler:
    `320 x 240`, yatay FOV `75`, pitch `40`, konum `(0.10, 0.0, -0.01)`.
    Unreal kamerasını mümkünse bu FOV ve pitch'e ayarlayın; aksi hâlde menzil
    tahmini eğitim dağılımının dışına çıkar.
    """

    focal_px = (output_width_px / 2.0) / math.tan(
        math.radians(horizontal_fov_deg) / 2.0
    )
    return CameraGeometry(
        width_px=int(output_width_px),
        height_px=int(output_height_px),
        focal_px=focal_px,
        principal_point_px=(output_width_px / 2.0, output_height_px / 2.0),
        pitch_down_rad=math.radians(pitch_down_deg),
        position_body_m=position_body_m,
    )


@dataclass(frozen=True)
class DronePose:
    """Bir politika adımındaki uçuş durumu (dünya + gövde FLU, SI birimleri)."""

    position_world_m: tuple[float, float, float]
    #: Gövde->dünya rotasyonu (3x3). `quaternion_wxyz_to_matrix` ile kurulabilir.
    rotation_body_to_world: FloatArray
    linear_velocity_body_m_s: tuple[float, float, float]
    angular_velocity_body_rad_s: tuple[float, float, float]


@dataclass(frozen=True)
class BodyVelocityCommand:
    """Gövde-çerçevesi hız komutu (FLU, SI). `yaw_rate` sağ-el, `+z` etrafında."""

    forward_m_s: float
    left_m_s: float
    up_m_s: float
    yaw_rate_rad_s: float
    phase: str  # "transit" | "hold"

    def as_array(self) -> FloatArray:
        return np.array(
            [self.forward_m_s, self.left_m_s, self.up_m_s, self.yaw_rate_rad_s],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class DeploymentConfig:
    """Köprü sabitleri. Varsayılanlar `vision_navigation_v4.toml` değerleridir."""

    # Action ölçekleri (vision-action-v3).
    max_horizontal_speed_m_s: float = 1.5
    max_vertical_speed_m_s: float = 1.0
    max_yaw_rate_rad_s: float = 1.5
    # fc_state açısal ölçek (controller max_body_rate_rad_s).
    max_body_rate_rad_s: tuple[float, float, float] = (3.0, 3.0, 2.0)
    # Gözlem ölçekleri.
    workspace_min_z_m: float = 0.0
    workspace_max_z_m: float = 3.0
    waypoint_distance_scale_m: float = 30.0
    maximum_frame_age_s: float = 0.10
    # Geometri ve yuvalar.
    camera: CameraGeometry = field(
        default_factory=lambda: camera_geometry_from_intrinsics(
            output_width_px=320,
            output_height_px=240,
            horizontal_fov_deg=75.0,
            pitch_down_deg=40.0,
            position_body_m=(0.10, 0.0, -0.01),
        )
    )
    slot_config: DetectionSlotConfig = field(default_factory=DetectionSlotConfig)
    motion_constraint: MotionConstraintConfig = field(
        default_factory=MotionConstraintConfig
    )
    speed_governor: SpeedGovernorConfig = field(default_factory=SpeedGovernorConfig)
    # Terminal "yaklaş ve dur" (temassız).
    #
    # Durma mesafesi hedefin MERKEZİNE değil YÜZEYİNE göre tanımlıdır. `target_radius_m`
    # tankın yaklaşık yarı-boyutudur (ör. ~3.5 m); `surface_clearance_m` tankın
    # kenarından bırakılacak paydır. Drone merkeze `target_radius_m + clearance`
    # kadar yaklaşınca durur, yani yüzeyden `clearance` kadar uzakta. `clearance`
    # pozitif bir tabana (`surface_clearance_floor_m`) zorlanır: hedefe fiziksel
    # temas bu köprünün kapsamı dışındadır (güvenlik sınırı).
    target_radius_m: float = 3.5
    surface_clearance_m: float = 0.5
    surface_clearance_floor_m: float = 0.3
    station_keep_yaw_gain: float = 0.8

    def effective_clearance_m(self) -> float:
        return max(self.surface_clearance_m, self.surface_clearance_floor_m)

    def effective_standoff_m(self) -> float:
        # Hedef merkezine ölçülen durma mesafesi = yarıçap + yüzey payı.
        return max(self.target_radius_m, 0.0) + self.effective_clearance_m()


class PolicyRunner:
    """Gözlem kurar, politikayı çağırır ve kısıtlı hız komutu üretir."""

    def __init__(self, predict_fn: PredictFn, config: DeploymentConfig | None = None):
        self._predict = predict_fn
        self.config = config or DeploymentConfig()
        self.config.slot_config.validate()
        self.config.motion_constraint.validate()

    def build_observation(
        self,
        *,
        pose: DronePose,
        detections: Iterable[Detection],
        believed_target_world_m: tuple[float, float, float],
        remaining_fraction: float,
        frame_age_s: float,
    ) -> tuple[dict[str, npt.NDArray[np.float32]], SlotEncoding]:
        """`vision-v4` gözlemini ve dedektör yuva kodlamasını üret."""

        cfg = self.config
        rotation = np.asarray(pose.rotation_body_to_world, dtype=np.float64)
        position = np.asarray(pose.position_world_m, dtype=np.float64)

        # waypoint: inanılan (yaklaşık) hedeften, gövde çerçevesinde (ADR-019).
        waypoint_world = (
            np.asarray(believed_target_world_m, dtype=np.float64) - position
        )
        waypoint_body = rotation.T @ waypoint_world
        distance_m = float(np.linalg.norm(waypoint_body))
        direction_body = (
            waypoint_body / distance_m if distance_m > 1e-9 else np.zeros(3)
        )
        distance_fraction = min(
            math.log1p(distance_m) / math.log1p(cfg.waypoint_distance_scale_m), 1.0
        )
        waypoint = np.concatenate((direction_body, [2.0 * distance_fraction - 1.0]))

        # fc_state: hız, açısal hız, yerçekimi yönü (attitude), irtifa.
        velocity_scale = np.array(
            [
                cfg.max_horizontal_speed_m_s,
                cfg.max_horizontal_speed_m_s,
                cfg.max_vertical_speed_m_s,
            ],
            dtype=np.float64,
        )
        angular_scale = np.asarray(cfg.max_body_rate_rad_s, dtype=np.float64)
        gravity_direction_body = rotation.T @ np.array([0.0, 0.0, -1.0])
        altitude_m = float(position[2])
        altitude_fraction = (altitude_m - cfg.workspace_min_z_m) / (
            cfg.workspace_max_z_m - cfg.workspace_min_z_m
        )
        fc_state = np.concatenate(
            (
                np.asarray(pose.linear_velocity_body_m_s, dtype=np.float64)
                / velocity_scale,
                np.asarray(pose.angular_velocity_body_rad_s, dtype=np.float64)
                / angular_scale,
                gravity_direction_body,
                [2.0 * altitude_fraction - 1.0],
            )
        )

        encoding = encode_detection_slots(
            detections,
            config=cfg.slot_config,
            geometry=cfg.camera,
            gravity_direction_body=gravity_direction_body,
            altitude_m=altitude_m,
        )

        timing = np.array(
            [
                float(np.clip(remaining_fraction, 0.0, 1.0)),
                min(max(frame_age_s, 0.0) / cfg.maximum_frame_age_s, 1.0),
            ],
            dtype=np.float32,
        )

        observation = {
            "detections": encoding.vector.astype(np.float32),
            "fc_state": np.clip(fc_state, -1.0, 1.0).astype(np.float32),
            "timing": timing,
            "waypoint": np.clip(waypoint, -1.0, 1.0).astype(np.float32),
        }
        return observation, encoding

    def command(
        self,
        action: npt.ArrayLike,
        encoding: SlotEncoding,
        *,
        target_horizontal_distance_m: float,
    ) -> BodyVelocityCommand:
        """Action'ı kısıtlı gövde hız komutuna çevir.

        `target_horizontal_distance_m` inanılan (yaklaşık) hedefe yatay
        mesafedir; standoff tetikleyicisi budur (dedektörün tank menzili düz-kare
        varsayımına dayandığı için güvenilir değildir).
        """

        cfg = self.config
        raw = np.asarray(action, dtype=np.float64).reshape(-1)
        horizontal = raw[:2] * cfg.max_horizontal_speed_m_s
        speed = float(np.linalg.norm(horizontal))
        if speed > cfg.max_horizontal_speed_m_s:
            horizontal = horizontal * (cfg.max_horizontal_speed_m_s / speed)

        if target_horizontal_distance_m <= cfg.effective_standoff_m():
            # Temassız "yaklaş ve dur": yatay dur, irtifa tut, hedefi ortala.
            yaw_rate = float(
                np.clip(
                    -cfg.station_keep_yaw_gain * encoding.target_center_x,
                    -cfg.max_yaw_rate_rad_s,
                    cfg.max_yaw_rate_rad_s,
                )
            )
            return BodyVelocityCommand(0.0, 0.0, 0.0, yaw_rate, phase="hold")

        # Seyir: ADR-024 kamera konisi + koridor hız sınırlayıcı.
        horizontal = constrain_horizontal_velocity(
            horizontal, cfg.motion_constraint, relaxed=False
        )
        limit = governor_speed_limit(
            encoding.vector,
            cfg.speed_governor,
            obstacle_slots=cfg.slot_config.obstacle_slots,
        )
        horizontal = apply_speed_limit(horizontal, limit)
        vertical = float(raw[2] * cfg.max_vertical_speed_m_s)
        yaw_rate = float(raw[3] * cfg.max_yaw_rate_rad_s)
        return BodyVelocityCommand(
            float(horizontal[0]),
            float(horizontal[1]),
            vertical,
            yaw_rate,
            phase="transit",
        )

    def step(
        self,
        *,
        pose: DronePose,
        detections: Iterable[Detection],
        believed_target_world_m: tuple[float, float, float],
        remaining_fraction: float = 1.0,
        frame_age_s: float = 0.0,
    ) -> BodyVelocityCommand:
        """Tek politika adımı: gözlem -> tahmin -> kısıtlı komut."""

        observation, encoding = self.build_observation(
            pose=pose,
            detections=detections,
            believed_target_world_m=believed_target_world_m,
            remaining_fraction=remaining_fraction,
            frame_age_s=frame_age_s,
        )
        action = self._predict(observation)
        position = np.asarray(pose.position_world_m, dtype=np.float64)
        target = np.asarray(believed_target_world_m, dtype=np.float64)
        horizontal_distance = float(np.linalg.norm((target - position)[:2]))
        return self.command(
            action, encoding, target_horizontal_distance_m=horizontal_distance
        )


def load_policy_predict_fn(model_path: str, *, deterministic: bool = True) -> PredictFn:
    """SB3 PPO kontrol noktasından `predict` fonksiyonu üret (tembel içe aktarma).

    `vecnormalize.pkl` gerekmez: `ppo_detector_d1a.toml` gözlem normalizasyonunu
    kapalı tutar (`[normalization] observations = false`), yalnız ödül
    normalize edilir; çıkarımda etkisizdir.
    """

    from stable_baselines3 import PPO  # tembel: SB3 yalnız gerçek dağıtımda gerekir

    model = PPO.load(model_path, device="cpu")

    def predict(
        observation: dict[str, npt.NDArray[np.float32]],
    ) -> npt.NDArray[np.float64]:
        action, _state = model.predict(observation, deterministic=deterministic)
        return np.asarray(action, dtype=np.float64).reshape(-1)

    return predict
