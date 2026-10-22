"""Araç konfigürasyonu, mixer ve rotor dinamiği testleri."""

from __future__ import annotations

import unittest

import mujoco
import numpy as np

from ebabil.config import load_vehicle_config
from ebabil.dynamics import QuadrotorPropulsion, RotorDynamics, XQuadrotorMixer
from ebabil.models import load_quadrotor


class VehicleConfigurationTests(unittest.TestCase):
    def test_configuration_is_physically_consistent(self) -> None:
        config = load_vehicle_config()
        self.assertAlmostEqual(config.hover_thrust_per_rotor_n, 2.4516625)
        self.assertGreater(config.thrust_to_weight_ratio, 2.0)
        self.assertLess(config.thrust_to_weight_ratio, 3.0)
        self.assertAlmostEqual(
            config.rotor.thrust_coefficient * config.rotor.max_speed_rad_s**2,
            config.rotor.max_thrust_n,
        )

    def test_configuration_matches_mjcf_nominal_values(self) -> None:
        config = load_vehicle_config()
        model = load_quadrotor()
        body_id = model.body("quadrotor").id
        self.assertAlmostEqual(float(model.body_mass[body_id]), config.mass_kg)
        np.testing.assert_allclose(
            model.body_inertia[body_id],
            config.inertia_diagonal_kg_m2,
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.actuator_ctrlrange[:, 1],
            np.full(4, config.rotor.max_thrust_n),
            rtol=0.0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            model.actuator_gear[:, 5],
            config.rotor.yaw_moment_ratio_m
            * np.asarray(config.rotor.reaction_torque_signs),
            rtol=0.0,
            atol=1e-12,
        )


class MixerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_vehicle_config()
        self.mixer = XQuadrotorMixer.from_config(self.config)

    def test_unsaturated_allocation_recovers_requested_wrench(self) -> None:
        requested_torque = np.array([0.1, -0.08, 0.02])
        result = self.mixer.allocate(9.80665, requested_torque)
        self.assertFalse(result.saturated)
        np.testing.assert_allclose(
            result.achieved_wrench,
            np.concatenate(([9.80665], requested_torque)),
            rtol=0.0,
            atol=1e-12,
        )

    def test_hover_allocation_is_equal(self) -> None:
        result = self.mixer.allocate(9.80665, np.zeros(3))
        np.testing.assert_allclose(
            result.applied_thrusts,
            np.full(4, self.config.hover_thrust_per_rotor_n),
            rtol=0.0,
            atol=1e-12,
        )

    def test_moment_signs_follow_flu_convention(self) -> None:
        hover_collective = self.config.mass_kg * self.config.gravity_m_s2
        roll = self.mixer.allocate(hover_collective, [0.1, 0.0, 0.0])
        self.assertGreater(roll.applied_thrusts[0], roll.applied_thrusts[1])
        self.assertGreater(roll.applied_thrusts[3], roll.applied_thrusts[2])

        pitch = self.mixer.allocate(hover_collective, [0.0, 0.1, 0.0])
        self.assertGreater(pitch.applied_thrusts[2], pitch.applied_thrusts[1])
        self.assertGreater(pitch.applied_thrusts[3], pitch.applied_thrusts[0])

        yaw = self.mixer.allocate(hover_collective, [0.0, 0.0, 0.02])
        self.assertGreater(yaw.applied_thrusts[0], yaw.applied_thrusts[1])
        self.assertGreater(yaw.applied_thrusts[2], yaw.applied_thrusts[3])

    def test_saturation_is_reported_and_clipped(self) -> None:
        result = self.mixer.allocate(30.0, np.zeros(3))
        self.assertTrue(result.saturated)
        np.testing.assert_allclose(
            result.applied_thrusts,
            np.full(4, self.config.rotor.max_thrust_n),
        )
        self.assertAlmostEqual(
            float(result.achieved_wrench[0]),
            4.0 * self.config.rotor.max_thrust_n,
        )


class RotorDynamicsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_vehicle_config()
        self.rotors = RotorDynamics(self.config.rotor)

    def test_step_response_is_monotonic_and_converges(self) -> None:
        desired = np.full(4, 3.0)
        previous = 0.0
        output = None
        for _ in range(1_000):
            output = self.rotors.step(desired, dt_s=0.002)
            self.assertGreaterEqual(float(output.actual_thrusts_n[0]), previous)
            previous = float(output.actual_thrusts_n[0])
        assert output is not None
        np.testing.assert_allclose(output.actual_thrusts_n, desired, atol=1e-10)

    def test_exact_discretization_is_step_size_consistent(self) -> None:
        desired = np.full(4, 4.0)
        one_step = RotorDynamics(self.config.rotor)
        ten_steps = RotorDynamics(self.config.rotor)
        one_output = one_step.step(desired, dt_s=0.1)
        for _ in range(10):
            ten_output = ten_steps.step(desired, dt_s=0.01)
        np.testing.assert_allclose(
            one_output.rotor_speeds_rad_s,
            ten_output.rotor_speeds_rad_s,
            rtol=0.0,
            atol=1e-12,
        )

    def test_thrust_commands_are_clipped_to_physical_range(self) -> None:
        output = self.rotors.step([-1.0, 0.0, 6.0, 10.0], dt_s=0.5)
        np.testing.assert_allclose(output.clipped_thrusts_n, [0.0, 0.0, 6.0, 6.0])
        self.assertTrue(np.all(output.actual_thrusts_n >= 0.0))
        self.assertTrue(
            np.all(output.actual_thrusts_n <= self.config.rotor.max_thrust_n)
        )

    def test_reset_to_thrust_sets_steady_state(self) -> None:
        desired = np.full(4, self.config.hover_thrust_per_rotor_n)
        self.rotors.reset(desired)
        output = self.rotors.step(desired, dt_s=0.002)
        np.testing.assert_allclose(output.actual_thrusts_n, desired, atol=1e-12)


class PropulsionIntegrationTests(unittest.TestCase):
    def test_hover_initialized_propulsion_balances_mujoco_model(self) -> None:
        config = load_vehicle_config()
        model = load_quadrotor()
        data = mujoco.MjData(model)
        propulsion = QuadrotorPropulsion(config)
        propulsion.reset_to_hover()

        for _ in range(250):
            propulsion.step(
                data=data,
                collective_thrust_n=config.mass_kg * config.gravity_m_s2,
                body_torque_nm=np.zeros(3),
                dt_s=model.opt.timestep,
            )
            mujoco.mj_step(model, data)

        self.assertAlmostEqual(float(data.qpos[2]), 1.0, places=9)
        np.testing.assert_allclose(data.qvel, np.zeros(6), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
