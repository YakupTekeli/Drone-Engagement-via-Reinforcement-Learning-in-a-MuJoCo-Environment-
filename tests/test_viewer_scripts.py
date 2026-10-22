"""Viewer giriş noktalarının GUI açmadan hazırlanabildiğini doğrula."""

from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from ebabil.evaluation import EpisodeReplay, save_replay

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ViewerScriptTests(unittest.TestCase):
    def run_dry(
        self, script_name: str, *arguments: str
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / script_name),
                *arguments,
                "--dry-run",
            ],
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            timeout=15.0,
        )

    def test_hover_replay_preparation(self) -> None:
        result = self.run_dry("view_hover_demo.py", "--duration", "0.01")

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("replay_frames=", result.stdout)

    def test_static_scene_preparation(self) -> None:
        result = self.run_dry(
            "view_static_scene.py",
            "--seed",
            "20260910",
            "--mode",
            "single_blocker",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("mode=single_blocker", result.stdout)
        self.assertIn("obstacle_count=1", result.stdout)

    def _d1_replay(self, position_xy: tuple[float, float]) -> EpisodeReplay:
        """D1 ölçeğinde, legacy workspace'e sığmayan bir yörünge."""

        x, y = position_xy
        return EpisodeReplay(
            time_s=np.array([0.0, 0.05]),
            qpos=np.array(
                [
                    [x, y, 1.5, 1.0, 0.0, 0.0, 0.0],
                    [x, y, 1.5, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            qvel=np.zeros((2, 6)),
            policy_time_s=np.array([0.05]),
            actions=np.zeros((1, 4)),
            rewards=np.zeros(1),
            distance_to_goal_m=np.array([4.0, 3.99]),
            minimum_range_m=np.ones(1),
            reward_components=np.zeros((1, 8)),
            rotor_thrusts_n=np.zeros((1, 4)),
            velocity_command_world_m_s=np.zeros((1, 3)),
            yaw_rate_command_rad_s=np.zeros(1),
            mixer_saturated=np.zeros(1, dtype=np.bool_),
            torque_saturated=np.zeros(1, dtype=np.bool_),
        )

    def test_policy_replay_uses_the_recorded_scene_config(self) -> None:
        """Replay kaydedildiği sahne config'iyle yeniden üretilmeli.

        Aksi halde varsayılan legacy `6 x 5 m` workspace çizilir, araç kadraj
        dışında kalır ve engel/hedef yerleşimi yanlış görünür. Bu hata bir
        demoda gözle yakalandı: kullanıcı D1 koşusunun replay'ini açtığında
        legacy sahneyi ve aracın olmadığı bir kadraj gördü.
        """

        replay = self._d1_replay((-7.5, 8.0))
        metadata = {
            "schema_version": 1,
            "model_variant": "vision",
            "scene_mode": "single_blocker",
            "endpoint_mode": "random_both",
            "scene_seed": 1941072691,
            "termination_reason": "hover_success",
            "scene_config": "src/ebabil/configs/environment/static_scene_d1.toml",
        }
        with tempfile.TemporaryDirectory() as directory:
            replay_path, _ = save_replay(
                Path(directory) / "episode.npz", replay, metadata
            )
            ok = self.run_dry("view_policy_replay.py", str(replay_path))

            without = dict(metadata)
            without.pop("scene_config")
            fallback_path, _ = save_replay(
                Path(directory) / "legacy.npz", replay, without
            )
            missing = self.run_dry("view_policy_replay.py", str(fallback_path))

        self.assertEqual(ok.returncode, 0, msg=ok.stderr)
        self.assertIn("replay_frames=2", ok.stdout)

        # scene_config olmadan legacy workspace kullanılır ve yörünge dışarı
        # düşer; sessizce yanlış sahne çizmek yerine hata vermeli.
        self.assertNotEqual(missing.returncode, 0)
        self.assertIn("workspace", missing.stderr)

    def test_policy_replay_preparation(self) -> None:
        replay = EpisodeReplay(
            time_s=np.array([0.0, 0.05]),
            qpos=np.array(
                [
                    [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                    [0.01, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0],
                ]
            ),
            qvel=np.zeros((2, 6)),
            policy_time_s=np.array([0.05]),
            actions=np.zeros((1, 4)),
            rewards=np.zeros(1),
            distance_to_goal_m=np.array([4.0, 3.99]),
            minimum_range_m=np.ones(1),
            reward_components=np.zeros((1, 8)),
            rotor_thrusts_n=np.zeros((1, 4)),
            velocity_command_world_m_s=np.zeros((1, 3)),
            yaw_rate_command_rad_s=np.zeros(1),
            mixer_saturated=np.zeros(1, dtype=np.bool_),
            torque_saturated=np.zeros(1, dtype=np.bool_),
        )
        with tempfile.TemporaryDirectory() as directory:
            replay_path, _ = save_replay(
                Path(directory) / "episode.npz",
                replay,
                {
                    "schema_version": 1,
                    "scene_mode": "clear",
                    "scene_seed": 123,
                    "termination_reason": "time_limit",
                },
            )

            result = self.run_dry("view_policy_replay.py", str(replay_path))

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("replay_frames=2", result.stdout)
        self.assertIn("mode=clear", result.stdout)
        self.assertIn("endpoint_mode=fixed", result.stdout)

    def test_target_camera_preparation(self) -> None:
        result = self.run_dry(
            "view_target_camera.py",
            "--pose",
            "hover",
            "--camera",
            "onboard",
        )

        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertIn("pose=hover", result.stdout)
        self.assertIn("camera=onboard", result.stdout)
        self.assertIn("visible_fraction=0.50", result.stdout)


if __name__ == "__main__":
    unittest.main()
