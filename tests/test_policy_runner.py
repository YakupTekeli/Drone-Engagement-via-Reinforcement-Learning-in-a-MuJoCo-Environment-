"""`ebabil.deploy.policy_runner` köprü çekirdeği testleri (SB3 gerektirmez)."""

from __future__ import annotations

import math

import numpy as np

from ebabil.deploy.policy_runner import (
    BodyVelocityCommand,
    DeploymentConfig,
    DronePose,
    PolicyRunner,
    camera_geometry_from_intrinsics,
    quaternion_wxyz_to_matrix,
)
from ebabil.perception.detections import OBSTACLE_CLASS_ID, TARGET_CLASS_ID, Detection


def _level_pose(position=(0.0, 0.0, 1.5), velocity=(0.0, 0.0, 0.0)) -> DronePose:
    return DronePose(
        position_world_m=position,
        rotation_body_to_world=np.eye(3),
        linear_velocity_body_m_s=velocity,
        angular_velocity_body_rad_s=(0.0, 0.0, 0.0),
    )


def _constant_runner(action) -> PolicyRunner:
    vector = np.asarray(action, dtype=np.float64)

    def predict(_observation):
        return vector

    return PolicyRunner(predict)


def test_quaternion_identity_is_identity_matrix():
    matrix = quaternion_wxyz_to_matrix((1.0, 0.0, 0.0, 0.0))
    assert np.allclose(matrix, np.eye(3))


def test_camera_geometry_matches_training_focal():
    geometry = camera_geometry_from_intrinsics(
        output_width_px=320,
        output_height_px=240,
        horizontal_fov_deg=75.0,
        pitch_down_deg=40.0,
        position_body_m=(0.10, 0.0, -0.01),
    )
    # 160 / tan(37.5 derece) ~ 208.5 px
    assert math.isclose(geometry.focal_px, 160.0 / math.tan(math.radians(37.5)))
    assert geometry.principal_point_px == (160.0, 120.0)


def test_observation_has_correct_keys_shapes_and_ranges():
    runner = _constant_runner((0.0, 0.0, 0.0, 0.0))
    observation, encoding = runner.build_observation(
        pose=_level_pose(),
        detections=[],
        believed_target_world_m=(5.0, 0.0, 0.0),
        remaining_fraction=1.0,
        frame_age_s=0.0,
    )
    assert tuple(sorted(observation)) == (
        "detections",
        "fc_state",
        "timing",
        "waypoint",
    )
    assert observation["detections"].shape == (72,)
    assert observation["fc_state"].shape == (10,)
    assert observation["timing"].shape == (2,)
    assert observation["waypoint"].shape == (4,)
    for key in ("detections", "fc_state", "waypoint"):
        assert np.all(observation[key] >= -1.0) and np.all(observation[key] <= 1.0)
    # Boş dedektör: bütün yuvalar boş (range_norm = 1.0 alanları).
    assert encoding.obstacles_encoded == 0
    assert not encoding.target_present


def test_waypoint_points_forward_when_target_ahead():
    runner = _constant_runner((0.0, 0.0, 0.0, 0.0))
    observation, _ = runner.build_observation(
        pose=_level_pose(),
        detections=[],
        believed_target_world_m=(8.0, 0.0, 1.5),
        remaining_fraction=1.0,
        frame_age_s=0.0,
    )
    # Gövde FLU: hedef önde -> yön vektörünün ileri (x) bileşeni baskın ve pozitif.
    direction = observation["waypoint"][:3]
    assert direction[0] > 0.9
    assert abs(direction[1]) < 1e-6


def test_transit_applies_cone_constraint_to_backward_command():
    # Geriye doğru yatay komut koni dışıdır ve seyirde sıfırlanır (ADR-024).
    runner = _constant_runner((-1.0, 0.0, 0.0, 0.0))
    command = runner.step(
        pose=_level_pose(),
        detections=[],
        believed_target_world_m=(10.0, 0.0, 1.5),
    )
    assert command.phase == "transit"
    assert command.forward_m_s >= -0.05
    assert abs(command.forward_m_s) < 1e-6


def test_governor_limits_forward_speed_with_obstacle_in_corridor():
    runner = _constant_runner((1.0, 0.0, 0.0, 0.0))
    # Orta koridorda yakın bir ağaç (engel) kutusu.
    obstacle = Detection(OBSTACLE_CLASS_ID, 150.0, 120.0, 170.0, 230.0)
    free = runner.step(
        pose=_level_pose(),
        detections=[],
        believed_target_world_m=(10.0, 0.0, 1.5),
    )
    blocked = runner.step(
        pose=_level_pose(),
        detections=[obstacle],
        believed_target_world_m=(10.0, 0.0, 1.5),
    )
    assert blocked.forward_m_s < free.forward_m_s


def test_station_keep_within_standoff_stops_and_centers_target():
    runner = _constant_runner((1.0, 0.0, 0.0, 0.0))
    config = runner.config
    # Hedef standoff içinde (yatay mesafe < taban).
    distance = config.effective_standoff_m() - 0.5
    command = runner.step(
        pose=_level_pose(position=(0.0, 0.0, 1.5)),
        detections=[],
        believed_target_world_m=(distance, 0.0, 0.0),
    )
    assert command.phase == "hold"
    assert command.forward_m_s == 0.0
    assert command.left_m_s == 0.0
    assert command.up_m_s == 0.0


def test_surface_clearance_floor_prevents_contact():
    # surface_clearance_m 0 verilse bile pozitif yüzey payı tabanı kalır; durma
    # mesafesi tank yarıçapının dışında kalır (temassızlık).
    config = DeploymentConfig(
        target_radius_m=3.5, surface_clearance_m=0.0, surface_clearance_floor_m=0.3
    )
    assert config.effective_clearance_m() >= 0.3
    assert config.effective_standoff_m() > config.target_radius_m


def test_surface_clearance_sets_stop_distance_beyond_radius():
    # Kullanıcı 0.5 m yüzey payı -> merkeze durma mesafesi = yarıçap + 0.5.
    config = DeploymentConfig(target_radius_m=3.5, surface_clearance_m=0.5)
    assert math.isclose(config.effective_standoff_m(), 4.0)


def test_yaw_centering_turns_toward_off_center_target():
    runner = _constant_runner((0.0, 0.0, 0.0, 0.0))
    # Hedef kutusu kadrajın sağında -> target_center_x > 0 -> yaw_rate < 0 (sağa döner).
    target_box = Detection(TARGET_CLASS_ID, 250.0, 120.0, 300.0, 200.0)
    distance = runner.config.effective_standoff_m() - 0.5
    command = runner.step(
        pose=_level_pose(),
        detections=[target_box],
        believed_target_world_m=(distance, 0.0, 0.0),
    )
    assert command.phase == "hold"
    assert isinstance(command, BodyVelocityCommand)
    assert command.yaw_rate_rad_s < 0.0
