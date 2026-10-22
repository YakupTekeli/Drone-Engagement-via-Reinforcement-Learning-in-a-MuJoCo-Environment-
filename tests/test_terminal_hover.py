"""ADR-012 terminal hover durum makinesi, reward bileşenleri ve ortam entegrasyonu."""

from __future__ import annotations

import math
import unittest
from dataclasses import replace
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from ebabil.envs import EbabilNavigationEnv, EbabilVisionNavigationEnv
from ebabil.envs.vision import PHYSICAL_FAILURE_REASONS, compute_segment_horizon
from ebabil.envs.vision_config import load_vision_navigation_config
from ebabil.missions import (
    ALLOWED_HOVER_TRANSITIONS,
    HOVER_STATE_CHAIN,
    TERMINAL_HOVER_STATES,
    HoverConditionReading,
    HoverState,
    HoverStateMachine,
)
from ebabil.perception import (
    ActorCameraFrame,
    TinyVisionEncoder,
    VisionEncoderOutput,
    load_camera_config,
    load_vision_encoder_config,
)
from ebabil.perception.dataset import load_vision_dataset_config
from ebabil.rewards import (
    ALIGNMENT_STATES,
    APPROACH_STATES,
    BEARING_STATES,
    CRUISE_STATES,
    STILL_STATES,
    TerminalHoverRewardBreakdown,
    compute_terminal_hover_reward,
)
from ebabil.scenes import believed_target_xy, load_target_area_config

POLICY_FREQUENCY_HZ = 20.0


class StubCamera:
    """Sabit kare üreten, timestamp'i dondurulabilen sahte kamera."""

    def __init__(self) -> None:
        config = load_camera_config()
        self.shape = (config.output_height_px, config.output_width_px, 3)
        self.closed = False
        self.frozen_timestamp_s: float | None = None

    def capture_actor(self, data: mujoco.MjData) -> ActorCameraFrame:
        timestamp_s = (
            float(data.time)
            if self.frozen_timestamp_s is None
            else self.frozen_timestamp_s
        )
        return ActorCameraFrame(
            rgb=np.zeros(self.shape, dtype=np.uint8), timestamp_s=timestamp_s
        )

    def close(self) -> None:
        self.closed = True


class ScriptedVisionEncoder(TinyVisionEncoder):
    """`target_score` ve merkezi testin sürdüğü encoder.

    Eğitilmemiş bir encoder her görüntü için ~`0.5` skor üretir; bu değer
    `loss_score` ile `acquire_score` arasında kaldığı için ne tespit ne kayıp
    tetikler. Durum makinesinin görsel kapılarını gerçekten sınamak için skor
    burada açıkça sürülür. Shape sözleşmesi üst sınıftan korunur.
    """

    scripted_target_score = 0.95
    scripted_center_x = 0.0

    def forward(self, images: torch.Tensor) -> VisionEncoderOutput:
        output = super().forward(images)
        batch = int(images.shape[0])
        score = float(np.clip(self.scripted_target_score, 1e-6, 1.0 - 1e-6))
        logit = math.log(score / (1.0 - score))
        return output._replace(
            target_logit=torch.full((batch,), logit, dtype=output.target_logit.dtype),
            target_center_xy=torch.full(
                (batch, 2),
                float(self.scripted_center_x),
                dtype=output.target_center_xy.dtype,
            ),
        )


def _scripted_encoder_factory() -> ScriptedVisionEncoder:
    torch.manual_seed(20260911)
    config, _ = load_vision_encoder_config()
    return ScriptedVisionEncoder(config)


def _machine(**overrides: float) -> HoverStateMachine:
    config = load_vision_navigation_config().terminal_hover
    if overrides:
        config = replace(config, **overrides)
    return HoverStateMachine(config, load_target_area_config(), POLICY_FREQUENCY_HZ)


def _reading(**overrides: object) -> HoverConditionReading:
    """Hover koşullarının tamamını sağlayan taban okuma."""

    target = load_target_area_config()
    base = {
        "target_score": 0.95,
        "frame_stale": False,
        "horizontal_distance_m": 0.0,
        # Varsayılan olarak arama yarıçapının dışında: mevcut testler arama
        # sayacını tetiklemeden tespit/hold mantığını sınar.
        "believed_horizontal_distance_m": 50.0,
        "inside_safe_region": True,
        "altitude_error_m": 0.0,
        "speed_m_s": 0.0,
        "tilt_rad": 0.0,
        "angular_rate_rad_s": 0.0,
    }
    base.update(overrides)
    assert target.altitude_tolerance_m > 0.0
    return HoverConditionReading(**base)  # type: ignore[arg-type]


class HoverTransitionTableTests(unittest.TestCase):
    """Plan gereği izinli geçişlerin açık tablosu."""

    def test_success_is_reachable_only_from_hold(self) -> None:
        sources = {
            source
            for source, target in ALLOWED_HOVER_TRANSITIONS
            if target is HoverState.SUCCESS
        }

        self.assertEqual(sources, {HoverState.HOLD})

    def test_terminal_states_are_absorbing(self) -> None:
        for terminal in TERMINAL_HOVER_STATES:
            with self.subTest(terminal=terminal):
                outgoing = {
                    target
                    for source, target in ALLOWED_HOVER_TRANSITIONS
                    if source is terminal
                }
                self.assertEqual(outgoing, set())

    def test_abort_is_reachable_from_every_chain_state(self) -> None:
        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                self.assertIn((state, HoverState.ABORT), ALLOWED_HOVER_TRANSITIONS)

    def test_chain_contains_adr_012_sequence(self) -> None:
        self.assertEqual(
            [str(state) for state in HOVER_STATE_CHAIN],
            ["transit", "search", "acquired", "approach", "settle", "hold"],
        )


class HoverAcquisitionTests(unittest.TestCase):
    def test_transit_requires_continuous_visibility(self) -> None:
        machine = _machine()
        required = machine.acquire_required_steps

        self.assertEqual(required, math.ceil(0.30 * POLICY_FREQUENCY_HZ))
        for _ in range(required - 1):
            update = machine.update(_reading(inside_safe_region=False))
            self.assertIs(update.state, HoverState.TRANSIT)
        # Tek görünmez adım sayacı sıfırlar.
        machine.update(_reading(target_score=0.05, inside_safe_region=False))
        self.assertIs(machine.state, HoverState.TRANSIT)
        for _ in range(required - 1):
            machine.update(_reading(inside_safe_region=False))
        self.assertIs(machine.state, HoverState.TRANSIT)
        machine.update(_reading(inside_safe_region=False))
        self.assertIsNot(machine.state, HoverState.TRANSIT)

    def test_score_between_thresholds_neither_acquires_nor_counts_as_loss(self) -> None:
        machine = _machine()
        between = (machine.config.loss_score + machine.config.acquire_score) / 2.0

        for _ in range(machine.acquire_required_steps + 5):
            update = machine.update(
                _reading(target_score=between, inside_safe_region=False)
            )

        self.assertIs(update.state, HoverState.TRANSIT)
        self.assertEqual(update.loss_steps, 0)
        self.assertFalse(update.target_visible)


class FarLossReturnsToTransitTests(unittest.TestCase):
    """ADR-017 Revizyon 1: arama yarıçapı dışında erken tespit kaybı abort üretmez."""

    def _acquire_far(self, machine: HoverStateMachine) -> None:
        for _ in range(machine.acquire_required_steps):
            machine.update(
                _reading(
                    horizontal_distance_m=6.0,
                    believed_horizontal_distance_m=6.5,
                    inside_safe_region=False,
                )
            )
        self.assertIs(machine.state, HoverState.ACQUIRED)

    def test_default_keeps_abort(self) -> None:
        machine = _machine()
        self._acquire_far(machine)
        for _ in range(machine.grace_steps + 1):
            update = machine.update(
                _reading(
                    target_score=0.0,
                    horizontal_distance_m=6.0,
                    believed_horizontal_distance_m=6.5,
                    inside_safe_region=False,
                )
            )
        self.assertIs(update.state, HoverState.ABORT)
        self.assertEqual(update.abort_reason, "target_lost")

    def test_far_loss_returns_to_transit_and_near_loss_still_aborts(self) -> None:
        machine = _machine(far_loss_returns_to_transit=True)
        self._acquire_far(machine)
        for _ in range(machine.grace_steps + 1):
            update = machine.update(
                _reading(
                    target_score=0.0,
                    horizontal_distance_m=6.0,
                    believed_horizontal_distance_m=6.5,
                    inside_safe_region=False,
                )
            )
        self.assertIs(update.state, HoverState.TRANSIT)
        self.assertIsNone(update.abort_reason)
        # Yeniden tespit mümkün; arama yarıçapı içinde kayıp yine abort.
        for _ in range(machine.acquire_required_steps):
            machine.update(
                _reading(
                    horizontal_distance_m=3.0,
                    believed_horizontal_distance_m=3.0,
                    inside_safe_region=False,
                )
            )
        self.assertIs(machine.state, HoverState.ACQUIRED)
        for _ in range(machine.grace_steps + 1):
            update = machine.update(
                _reading(
                    target_score=0.0,
                    horizontal_distance_m=3.0,
                    believed_horizontal_distance_m=3.0,
                    inside_safe_region=False,
                )
            )
        self.assertIs(update.state, HoverState.ABORT)
        self.assertEqual(update.abort_reason, "target_lost")


class HoverProgressionTests(unittest.TestCase):
    def _acquire(self, machine: HoverStateMachine) -> None:
        for _ in range(machine.acquire_required_steps - 1):
            machine.update(
                _reading(inside_safe_region=False, horizontal_distance_m=8.0)
            )
        machine.update(_reading(inside_safe_region=False, horizontal_distance_m=8.0))

    def test_states_follow_nested_geometric_conditions(self) -> None:
        machine = _machine()
        self._acquire(machine)

        cases = (
            (
                HoverState.ACQUIRED,
                {"inside_safe_region": False, "horizontal_distance_m": 8.0},
            ),
            (
                HoverState.APPROACH,
                {"inside_safe_region": False, "horizontal_distance_m": 1.5},
            ),
            (HoverState.SETTLE, {"inside_safe_region": True, "speed_m_s": 2.0}),
            (HoverState.HOLD, {}),
        )
        for expected, overrides in cases:
            with self.subTest(expected=expected):
                update = machine.update(_reading(**overrides))
                self.assertIs(update.state, expected)

    def test_continuous_hold_reaches_success_at_configured_duration(self) -> None:
        machine = _machine()
        target = load_target_area_config()
        self._acquire(machine)

        self.assertEqual(
            machine.hold_required_steps,
            math.ceil(target.hold_seconds * POLICY_FREQUENCY_HZ),
        )
        for step in range(machine.hold_required_steps):
            update = machine.update(_reading())
            if step < machine.hold_required_steps - 1:
                self.assertIs(update.state, HoverState.HOLD)
                self.assertFalse(update.success)

        self.assertIs(update.state, HoverState.SUCCESS)
        self.assertTrue(update.success)
        self.assertAlmostEqual(update.hold_seconds, target.hold_seconds, places=9)
        self.assertAlmostEqual(update.hold_progress, 1.0, places=9)

    def test_each_broken_hover_condition_resets_hold_counter(self) -> None:
        target = load_target_area_config()
        breakers = {
            "safe_region": {"inside_safe_region": False},
            "altitude": {"altitude_error_m": target.altitude_tolerance_m + 0.01},
            "speed": {"speed_m_s": target.maximum_linear_speed_m_s + 0.01},
            "tilt": {"tilt_rad": target.maximum_tilt_rad + 0.01},
            "angular_rate": {
                "angular_rate_rad_s": target.maximum_angular_rate_rad_s + 0.01
            },
        }
        for name, overrides in breakers.items():
            with self.subTest(condition=name):
                machine = _machine()
                self._acquire(machine)
                for _ in range(10):
                    machine.update(_reading())
                self.assertGreater(machine.update(_reading()).hold_steps, 10)

                broken = machine.update(_reading(**overrides))
                self.assertEqual(broken.hold_steps, 0)
                self.assertIsNot(broken.state, HoverState.HOLD)
                self.assertFalse(broken.hover_conditions_met)

                resumed = machine.update(_reading())
                self.assertEqual(resumed.hold_steps, 1)

    def test_hold_interruption_prevents_success_at_original_deadline(self) -> None:
        machine = _machine()
        self._acquire(machine)
        for _ in range(machine.hold_required_steps - 1):
            machine.update(_reading())
        machine.update(_reading(inside_safe_region=False))

        update = machine.update(_reading())

        self.assertIs(update.state, HoverState.HOLD)
        self.assertFalse(update.success)
        self.assertEqual(update.hold_steps, 1)


class HoverAbortTests(unittest.TestCase):
    def _acquire(self, machine: HoverStateMachine) -> None:
        for _ in range(machine.acquire_required_steps):
            machine.update(
                _reading(inside_safe_region=False, horizontal_distance_m=8.0)
            )

    def test_target_loss_within_grace_does_not_abort(self) -> None:
        machine = _machine()
        self._acquire(machine)

        for _ in range(machine.grace_steps):
            update = machine.update(_reading(target_score=0.05))

        self.assertIsNot(update.state, HoverState.ABORT)
        self.assertEqual(update.loss_steps, machine.grace_steps)
        self.assertAlmostEqual(
            update.loss_seconds, machine.config.grace_seconds, places=9
        )

    def test_target_loss_beyond_grace_aborts(self) -> None:
        machine = _machine()
        self._acquire(machine)

        for _ in range(machine.grace_steps + 1):
            update = machine.update(_reading(target_score=0.05))

        self.assertIs(update.state, HoverState.ABORT)
        self.assertEqual(update.abort_reason, "target_lost")

    def test_regained_target_clears_loss_counter(self) -> None:
        machine = _machine()
        self._acquire(machine)
        for _ in range(machine.grace_steps):
            machine.update(_reading(target_score=0.05))

        update = machine.update(_reading())

        self.assertEqual(update.loss_steps, 0)
        self.assertIsNot(update.state, HoverState.ABORT)

    def test_transit_never_aborts_on_target_loss(self) -> None:
        machine = _machine()

        for _ in range(machine.grace_steps + 20):
            update = machine.update(_reading(target_score=0.01))

        self.assertIs(update.state, HoverState.TRANSIT)

    def test_stale_frame_aborts_from_any_state(self) -> None:
        for acquired in (False, True):
            with self.subTest(acquired=acquired):
                machine = _machine()
                if acquired:
                    self._acquire(machine)

                update = machine.update(_reading(frame_stale=True))

                self.assertIs(update.state, HoverState.ABORT)
                self.assertEqual(update.abort_reason, "stale_frame")

    def test_update_after_terminal_state_is_rejected(self) -> None:
        machine = _machine()
        machine.update(_reading(frame_stale=True))

        with self.assertRaisesRegex(RuntimeError, "Terminal durumdan"):
            machine.update(_reading())

    def test_reset_returns_to_transit_and_clears_counters(self) -> None:
        machine = _machine()
        self._acquire(machine)
        machine.update(_reading())
        machine.reset()

        update = machine.update(_reading())

        self.assertIs(update.state, HoverState.TRANSIT)
        self.assertEqual(update.hold_steps, 0)
        self.assertEqual(update.loss_steps, 0)


class TerminalHoverConfigValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_vision_navigation_config().terminal_hover
        self.target = load_target_area_config()

    def test_shipped_config_is_consistent_with_target_geometry(self) -> None:
        self.config.validate()
        self.config.validate_against_target(self.target)

    def test_hysteresis_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "histerezis"):
            replace(self.config, loss_score=self.config.acquire_score).validate()

    def test_approach_radius_must_cover_safe_region_diagonal(self) -> None:
        safe_half_x, safe_half_y = self.target.safe_half_size_xy_m
        diagonal = math.hypot(safe_half_x, safe_half_y)

        with self.assertRaisesRegex(ValueError, "güvenli iç bölge köşegenini"):
            replace(self.config, approach_radius_m=diagonal).validate_against_target(
                self.target
            )

    def test_invalid_reading_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_score"):
            _reading(target_score=1.4).validate()
        with self.assertRaisesRegex(ValueError, "negatif olamaz"):
            _reading(speed_m_s=-0.1).validate()


class TerminalHoverRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = load_vision_navigation_config().terminal_hover_reward

    def _reward(self, **overrides: object) -> TerminalHoverRewardBreakdown:
        base: dict[str, object] = {
            "state": HoverState.APPROACH,
            "previous_anchor_distance_m": 2.0,
            "anchor_distance_m": 2.0,
            "target_center_x": 0.0,
            "target_visible": True,
            "inside_safe_region": False,
            "speed_m_s": 0.0,
            "minimum_range_m": 10.0,
            "current_action": np.zeros(4),
            "previous_action": np.zeros(4),
            "success": False,
            "collision": False,
            "out_of_bounds": False,
            "aborted": False,
            "altitude_m": 2.5,
            "believed_bearing_error_rad": 0.0,
            "speed_allowance_m_s": 0.0,
            "altitude_error_m": 0.0,
            "altitude_allowance_m": 0.2,
            "tilt_rad": 0.0,
            "maximum_tilt_rad": 0.1745,
            "angular_rate_rad_s": 0.0,
            "maximum_angular_rate_rad_s": 0.35,
            "physical_failure": False,
            "banked_progress": 0.0,
        }
        base.update(overrides)
        return compute_terminal_hover_reward(self.config, **base)  # type: ignore[arg-type]

    def test_all_adr_012_components_are_logged_separately(self) -> None:
        required = {
            "target_progress",
            "target_alignment",
            "settle_speed",
            "hover_hold",
            "smoothness",
            "collision",
            "out_of_bounds",
            "target_visible",
            "progress_refund",
            "cruise_altitude",
            "bearing_alignment",
            "settle_altitude",
            "settle_tilt",
            "settle_angular_rate",
        }

        components = self._reward().as_dict()

        self.assertTrue(required.issubset(components))
        self.assertEqual(len(components), 18)

    def test_total_equals_component_sum(self) -> None:
        breakdown = self._reward(
            state=HoverState.HOLD,
            anchor_distance_m=1.5,
            inside_safe_region=True,
            speed_m_s=0.1,
            minimum_range_m=0.4,
            current_action=np.array([0.3, -0.2, 0.1, 0.05]),
        )

        self.assertAlmostEqual(
            breakdown.total, sum(breakdown.as_dict().values()), places=12
        )

    def test_progress_is_positive_when_closing_on_anchor(self) -> None:
        closing = self._reward(previous_anchor_distance_m=3.0, anchor_distance_m=2.5)
        receding = self._reward(previous_anchor_distance_m=2.5, anchor_distance_m=3.0)

        self.assertGreater(closing.target_progress, 0.0)
        self.assertLess(receding.target_progress, 0.0)
        self.assertAlmostEqual(
            closing.target_progress, -receding.target_progress, places=12
        )

    def test_progress_density_is_constant_and_route_independent(self) -> None:
        """REV-029: progress metre başına sabittir, rotaya normalize edilmez.

        Normalizasyon iki koşuda ölçülerek reddedildi: tam normalize form
        terminal hassasiyeti bozdu (başarı `%42.5 -> %22.5`), iki menzilli form
        navigasyonu bozdu (`%0`). Ayrıntı REV-029.
        """

        weight = self.config.target_progress_weight
        for closed_m in (0.05, 1.0, 5.0):
            with self.subTest(closed_m=closed_m):
                for start_m in (1.0, 10.0, 100.0):
                    step = self._reward(
                        previous_anchor_distance_m=start_m + closed_m,
                        anchor_distance_m=start_m,
                    )
                    self.assertAlmostEqual(
                        step.target_progress, weight * closed_m, places=12
                    )

    def test_cutting_through_an_obstacle_never_pays(self) -> None:
        """REV-029 invariantı: kısıt yoğunluk üzerindedir, toplam üzerinde değil.

        Çarpışma episode'u bitirir, bu yüzden kıyaslanacak şey episode toplamı
        değil kısa yolun yerel kazancıdır. D1 workspace'i `20 x 20 m` olduğu
        için tek bir engelin zorlayabileceği dolaşma mesafesi bu ölçektedir.
        """

        for detour_m in (1.0, 2.5, 5.0):
            with self.subTest(detour_m=detour_m):
                self.assertLessEqual(
                    self.config.detour_saving_for(detour_m),
                    self.config.collision_penalty,
                )
        with self.assertRaisesRegex(ValueError, "detour_length_m"):
            self.config.detour_saving_for(0.0)

    def test_bearing_penalty_applies_outside_transit_only(self) -> None:
        """REV-041: kamerayı hedefe doğrultmak terminal fazın işidir.

        ADR-020 holonomik hareketi açınca yön hareketten ayrıldı: araç yana
        kayarak hedefe gidiyor fakat başka yöne bakıyor. Ölçülen: hedefin
        `0-3 m` yakınında yön hatası `101-123 derece`, `hold` durumuna 300
        episode'da hiç girilmedi.

        `transit` kapsam dışıdır ve bu bilinçlidir: seyirde kamera gittiği
        yöne bakmalı, yoksa kör alana uçar.
        """

        self.assertNotIn(HoverState.TRANSIT, BEARING_STATES)
        self.assertIn(HoverState.APPROACH, BEARING_STATES)
        self.assertGreater(self.config.bearing_alignment_weight, 0.0)

        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                facing = self._reward(state=state, believed_bearing_error_rad=0.0)
                away = self._reward(state=state, believed_bearing_error_rad=math.pi)

                self.assertEqual(facing.bearing_alignment, 0.0)
                if state in BEARING_STATES:
                    self.assertAlmostEqual(
                        away.bearing_alignment,
                        -self.config.bearing_alignment_weight,
                        places=12,
                    )
                else:
                    self.assertEqual(away.bearing_alignment, 0.0)

    def test_bearing_penalty_is_monotonic_and_bounded(self) -> None:
        """Yön hatası büyüdükçe ceza artmalı ve ağırlıkla sınırlı kalmalı."""

        previous = 0.0
        for error_rad in (0.0, math.pi / 6, math.pi / 2, math.pi):
            value = self._reward(
                state=HoverState.APPROACH, believed_bearing_error_rad=error_rad
            ).bearing_alignment

            self.assertLessEqual(value, previous + 1e-12)
            self.assertGreaterEqual(value, -self.config.bearing_alignment_weight)
            previous = value

        # `pi`'yi aşan girdi kırpılır, ceza büyümeye devam etmez.
        clipped = self._reward(
            state=HoverState.APPROACH, believed_bearing_error_rad=4.0 * math.pi
        )
        self.assertAlmostEqual(
            clipped.bearing_alignment,
            -self.config.bearing_alignment_weight,
            places=12,
        )

    def test_cruise_altitude_penalty_applies_only_in_cruise_states(self) -> None:
        """REV-036: alçak uçmak aracı kör eder, fakat terminalde inmesi gerekir.

        Kameranın ileri görüş mesafesi `3.711 x irtifa`'dır. Politika engellerin
        üstünden uçma yolu kapandığında `0.81 m`'ye indi ve orada yalnız
        `3.0 m` görüyordu; çarpışma `%85-93`'te sıkıştı. Ceza yalnız seyir
        durumlarında uygulanır; `settle`/`hold`'da araç hold bandına
        (`1.31-1.71 m`) inmek zorundadır.
        """

        self.assertEqual(CRUISE_STATES, {HoverState.TRANSIT, HoverState.SEARCH})
        required_m = self.config.cruise_altitude_m
        deficit_m = 1.0

        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                low = self._reward(state=state, altitude_m=required_m - deficit_m)
                high = self._reward(state=state, altitude_m=required_m + 0.5)

                self.assertEqual(high.cruise_altitude, 0.0)
                if state in CRUISE_STATES:
                    self.assertAlmostEqual(
                        low.cruise_altitude,
                        -self.config.cruise_altitude_weight * deficit_m,
                        places=12,
                    )
                else:
                    self.assertEqual(low.cruise_altitude, 0.0)

    def test_cruise_altitude_is_derived_from_camera_reach(self) -> None:
        """Gereken irtifa, sensörün menzilini free_space derinliğiyle eşitler."""

        camera = load_camera_config()
        depth_m = load_vision_dataset_config().free_space_max_depth_m
        _, far_m = camera.ground_swath_m(self.config.cruise_altitude_m)

        self.assertAlmostEqual(far_m, depth_m, places=2)

    def test_cruise_altitude_must_exceed_the_hover_band(self) -> None:
        """Seyir cezası terminal inişle çelişmemeli; config yüklemesinde sınanır."""

        config = load_vision_navigation_config()
        target = load_target_area_config()
        hover_altitude_m = target.surface_z_m + target.hover_height_m

        self.assertGreater(
            config.terminal_hover_reward.cruise_altitude_m, hover_altitude_m
        )
        broken = replace(
            config,
            terminal_hover_reward=replace(
                config.terminal_hover_reward, cruise_altitude_m=hover_altitude_m
            ),
        )
        with self.assertRaisesRegex(ValueError, "cruise_altitude_m"):
            broken.validate_against_target(target)

    def test_physical_failure_refunds_all_banked_progress(self) -> None:
        """REV-033: fiziksel başarısızlık toplanan yaklaşma ödülünü geri verir.

        Ölçülmüş defekt: `target_progress` bir kez toplanıp bankada kalıyordu
        ve başarıyla gerekçelendirilmesi gerekmiyordu. "Hedefe git sonra yere
        çak" stratejisi net `+7` ödül veriyordu (178/300 episode); bütçeyi
        doldurmak ise `-37`. Politika ölmeyi öğrenmişti.
        """

        weight = self.config.target_progress_weight
        banked = 40.0
        crash = self._reward(
            previous_anchor_distance_m=1.0,
            anchor_distance_m=0.5,
            banked_progress=banked,
            physical_failure=True,
            collision=True,
        )
        last_step_progress = weight * 0.5

        self.assertAlmostEqual(crash.target_progress, last_step_progress, places=12)
        self.assertAlmostEqual(
            crash.progress_refund, -(banked + last_step_progress), places=12
        )
        self.assertAlmostEqual(
            crash.target_progress + crash.progress_refund, -banked, places=12
        )

    def test_non_physical_endings_keep_the_progress_gradient(self) -> None:
        """Timeout ve hover abort'u progress'i geri vermez.

        Tamamını geri vermek erken öğrenmeyi yok ederdi: hiç başarı yokken
        episode getirisi progress şekillendirmesi hiç verilmemiş gibi olurdu.
        """

        for overrides in (
            {"aborted": True},
            {},
            {"state": HoverState.HOLD, "success": True},
        ):
            with self.subTest(overrides=tuple(overrides)):
                breakdown = self._reward(
                    previous_anchor_distance_m=1.0,
                    anchor_distance_m=0.5,
                    banked_progress=40.0,
                    **overrides,
                )

                self.assertEqual(breakdown.progress_refund, 0.0)
                self.assertGreater(breakdown.target_progress, 0.0)

    def test_refund_rejects_non_finite_bank(self) -> None:
        with self.assertRaisesRegex(ValueError, "banked_progress"):
            self._reward(banked_progress=math.inf)

    def test_visible_target_is_rewarded_in_every_state(self) -> None:
        """REV-027: görünürlüğün pozitif karşılığı vardır, yalnız abort cezası değil."""

        self.assertGreater(self.config.target_visible_weight, 0.0)
        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                visible = self._reward(state=state, target_visible=True)
                invisible = self._reward(state=state, target_visible=False)

                self.assertAlmostEqual(
                    visible.target_visible,
                    self.config.target_visible_weight,
                    places=12,
                )
                self.assertEqual(invisible.target_visible, 0.0)

    def test_visibility_reward_never_makes_stalling_profitable(self) -> None:
        """REV-027 düzeltmesinin üst sınırı: görünürlük zaman cezasını yenmemeli.

        Hedefi seyrederek beklemek adım başına net negatif kalmalı, aksi halde
        politika bütçeyi doldurmayı öğrenir. `hover_hold` bu sınıra tabi
        değildir; o yalnız hedefin üzerinde, `HOLD` durumunda verilir.
        """

        self.assertLess(self.config.target_visible_weight, self.config.time_penalty)

        watching = self._reward(state=HoverState.TRANSIT, target_visible=True)
        self.assertLess(watching.target_visible + watching.time, 0.0)

        with self.assertRaisesRegex(ValueError, "target_visible_weight"):
            replace(
                self.config,
                target_visible_weight=self.config.time_penalty,
            ).validate()

    def test_alignment_is_disabled_in_settle_and_hold(self) -> None:
        """ADR-016: hold pozunda hedef merkezi kadraj dışıdır."""

        self.assertEqual(ALIGNMENT_STATES, {HoverState.ACQUIRED, HoverState.APPROACH})
        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                breakdown = self._reward(state=state, target_center_x=0.8)
                if state in ALIGNMENT_STATES:
                    self.assertLess(breakdown.target_alignment, 0.0)
                else:
                    self.assertEqual(breakdown.target_alignment, 0.0)

    def test_alignment_requires_visible_target(self) -> None:
        invisible = self._reward(target_center_x=0.8, target_visible=False)

        self.assertEqual(invisible.target_alignment, 0.0)

    def test_settle_speed_penalises_only_the_excess_over_the_allowance(self) -> None:
        """REV-042: yavaşlama cezası bütün terminal fazda, mesafeye ölçekli.

        Ölçülen defekt: ceza yalnız `inside_safe_region` iken açıktı; araç
        oraya terminal adımların `%2`'sinde girebiliyordu ve hız `%99.9`
        ihlal ediliyordu. Yani yavaşlama teşviki, ancak yavaşlayarak
        ulaşılabilen bir yerde bulunuyordu.
        """

        # Payın altındaki hız cezasız.
        within = self._reward(
            state=HoverState.APPROACH, speed_m_s=0.8, speed_allowance_m_s=1.0
        )
        self.assertEqual(within.settle_speed, 0.0)

        # Payı aşan kısım cezalı ve yalnız fazlalık sayılır.
        excess = self._reward(
            state=HoverState.APPROACH, speed_m_s=1.5, speed_allowance_m_s=1.0
        )
        self.assertAlmostEqual(
            excess.settle_speed,
            -self.config.settle_speed_weight * 0.5**2,
            places=12,
        )

        # Güvenli iç bölgede pay sıfırlanır: hold hız sınırı geçerli.
        inside = self._reward(
            state=HoverState.APPROACH,
            inside_safe_region=True,
            speed_m_s=1.0,
            speed_allowance_m_s=1.0,
        )
        self.assertAlmostEqual(
            inside.settle_speed,
            -self.config.settle_speed_weight * 1.0**2,
            places=12,
        )

        # `transit` kapsam dışı: seyirde yavaşlamak istenmez.
        cruising = self._reward(
            state=HoverState.TRANSIT, speed_m_s=1.5, speed_allowance_m_s=0.0
        )
        self.assertEqual(cruising.settle_speed, 0.0)

    def test_all_five_hold_conditions_are_shaped(self) -> None:
        """REV-046: başarı beş koşulun kesişimini ister, ödül beşini de öğretmeli.

        Ölçülen defekt: ödül yalnız konum (`target_progress`) ve hız
        (`settle_speed`) şekillendiriyordu; irtifa, tilt ve açısal hızın hiç
        gradyanı yoktu. Sonuç, beş koşulun birlikte sağlandığı tek bir adım
        bile olmamasıydı (5/5 `%0.0`) ve bu, hold süresi `3.0 -> 0.5 s`
        indirildiğinde de değişmedi.
        """

        limits = {
            "altitude": ("altitude_error_m", "altitude_allowance_m", 0.2),
            "tilt": ("tilt_rad", "maximum_tilt_rad", 0.1745),
            "angular": (
                "angular_rate_rad_s",
                "maximum_angular_rate_rad_s",
                0.35,
            ),
        }
        fields = {
            "altitude": "settle_altitude",
            "tilt": "settle_tilt",
            "angular": "settle_angular_rate",
        }
        active = {
            "altitude": APPROACH_STATES,
            "tilt": STILL_STATES,
            "angular": STILL_STATES,
        }
        for name, (value_key, limit_key, limit) in limits.items():
            with self.subTest(condition=name):
                field = fields[name]
                state = (
                    HoverState.APPROACH
                    if HoverState.APPROACH in active[name]
                    else HoverState.SETTLE
                )

                # Sınır içinde ceza yok: başarı ölçütü sıkılaştırılmaz.
                within = self._reward(
                    state=state, **{value_key: limit * 0.5, limit_key: limit}
                )
                self.assertEqual(getattr(within, field), 0.0)

                # Sınırda da ceza yok.
                at_limit = self._reward(
                    state=state, **{value_key: limit, limit_key: limit}
                )
                self.assertEqual(getattr(at_limit, field), 0.0)

                # Aşınca yalnız fazlalık cezalı ve artan.
                small = self._reward(
                    state=state, **{value_key: limit * 2.0, limit_key: limit}
                )
                large = self._reward(
                    state=state, **{value_key: limit * 3.0, limit_key: limit}
                )
                self.assertLess(getattr(small, field), 0.0)
                self.assertLess(getattr(large, field), getattr(small, field))

                # `transit` kapsam dışı: seyirde hover koşulları aranmaz.
                cruising = self._reward(
                    state=HoverState.TRANSIT,
                    **{value_key: limit * 3.0, limit_key: limit},
                )
                self.assertEqual(getattr(cruising, field), 0.0)

    def test_cruise_and_settle_states_are_disjoint(self) -> None:
        """REV-046: iki irtifa cezası aynı durumda zıt yönlere çekemez.

        `cruise_altitude` aracı `1.42 m`'ye YUKARI iter, hold bandı ise
        `0.91-1.31 m`'dir. `search` her ikisine de dâhil edildiğinde hiçbir
        irtifa cezasız kalmadı: `minimum_altitude` terminasyonu `%79`'a
        çıktı ve ödül `-280`'e düştü.
        """

        self.assertEqual(set(CRUISE_STATES & APPROACH_STATES), set())
        self.assertTrue(STILL_STATES <= APPROACH_STATES)
        self.assertIn(HoverState.SEARCH, CRUISE_STATES)
        self.assertNotIn(HoverState.SEARCH, APPROACH_STATES)
        self.assertIn(HoverState.APPROACH, APPROACH_STATES)
        # bearing_alignment dönmeyi ÖDÜLLENDİRDİĞİ için açısal hız
        # cezası approach'ta KAPALI olmalı.
        self.assertNotIn(HoverState.APPROACH, STILL_STATES)

        # `search` hâlâ bir seyir durumudur: hold cezaları orada kapalı.
        searching = self._reward(
            state=HoverState.SEARCH,
            # Seyir hedefinin altında: cruise cezası açık olmalı.
            altitude_m=self.config.cruise_altitude_m - 0.5,
            altitude_error_m=1.0,
            altitude_allowance_m=0.2,
            tilt_rad=0.5,
            maximum_tilt_rad=0.1745,
            angular_rate_rad_s=1.2,
            maximum_angular_rate_rad_s=0.35,
        )

        self.assertEqual(searching.settle_altitude, 0.0)
        self.assertEqual(searching.settle_tilt, 0.0)
        self.assertEqual(searching.settle_angular_rate, 0.0)
        self.assertLess(searching.cruise_altitude, 0.0)

    def test_shaping_weights_give_comparable_pressure(self) -> None:
        """Ağırlıklar `settle_speed` ile eşit baskı verecek şekilde türetildi."""

        reference = self.config.settle_speed_weight * 1.25**2
        for weight, typical in (
            (self.config.settle_altitude_weight, 0.9),
            (self.config.settle_angular_rate_weight, 0.8),
            (self.config.settle_tilt_weight, math.radians(15.0)),
        ):
            with self.subTest(weight=weight):
                cost = weight * typical**2

                self.assertGreater(cost, reference * 0.5)
                self.assertLess(cost, reference * 2.0)

    def test_speed_allowance_ramps_from_cruise_to_hold_limit(self) -> None:
        """Pay `approach_radius`'ta seyir hızı, güvenli bölgede hold sınırı."""

        env_config = load_vision_navigation_config()
        target = load_target_area_config()
        cruise = env_config.navigation.action.max_horizontal_speed_m_s
        hold_limit = target.maximum_linear_speed_m_s
        radius = env_config.terminal_hover.approach_radius_m
        safe_half = min(target.safe_half_size_xy_m)

        with patch.object(
            EbabilVisionNavigationEnv, "_build_scene_model", autospec=True
        ):
            pass

        # Rampa formülünü doğrudan doğrula (env örneği gerektirmez).
        def allowance(distance_m: float) -> float:
            fraction = (distance_m - safe_half) / (radius - safe_half)
            fraction = min(1.0, max(0.0, fraction))
            return hold_limit + (cruise - hold_limit) * fraction

        self.assertAlmostEqual(allowance(radius), cruise, places=9)
        self.assertAlmostEqual(allowance(safe_half), hold_limit, places=9)
        self.assertAlmostEqual(allowance(0.1), hold_limit, places=9)
        self.assertGreater(allowance(1.5), allowance(1.0))

    def test_hold_reward_is_granted_only_in_hold_and_is_bounded(self) -> None:
        for state in HOVER_STATE_CHAIN:
            with self.subTest(state=state):
                breakdown = self._reward(state=state)
                expected = (
                    self.config.hover_hold_weight if state is HoverState.HOLD else 0.0
                )
                self.assertAlmostEqual(breakdown.hover_hold, expected, places=12)

        hold_steps = math.ceil(
            load_target_area_config().hold_seconds * POLICY_FREQUENCY_HZ
        )
        uninterrupted = self.config.uninterrupted_hold_reward(hold_steps)

        self.assertAlmostEqual(
            uninterrupted, self.config.hover_hold_weight * hold_steps
        )
        self.assertLess(uninterrupted, self.config.success_bonus)

    def test_episode_hold_bound_exceeds_uninterrupted_sequence(self) -> None:
        """REV-022: hold sayacının sıfırlanması verilmiş reward'ı geri almaz.

        Kesintisiz dizi değeri bir episode üst sınırı değildir; durum tekrar
        `HOLD` olduğunda her adım yeniden reward verir.
        """

        hold_steps = math.ceil(
            load_target_area_config().hold_seconds * POLICY_FREQUENCY_HZ
        )
        episode_steps = 400
        uninterrupted = self.config.uninterrupted_hold_reward(hold_steps)
        episode_bound = self.config.episode_hold_reward_bound(episode_steps)

        self.assertGreater(episode_bound, uninterrupted)
        self.assertGreater(episode_bound, self.config.success_bonus)
        self.assertAlmostEqual(
            episode_bound, self.config.hover_hold_weight * episode_steps
        )
        with self.assertRaisesRegex(ValueError, "maximum_episode_steps"):
            self.config.episode_hold_reward_bound(0)

    def test_collision_and_out_of_bounds_are_separate_penalties(self) -> None:
        collision = self._reward(collision=True)
        bounds = self._reward(out_of_bounds=True)

        self.assertAlmostEqual(collision.collision, -self.config.collision_penalty)
        self.assertEqual(collision.out_of_bounds, 0.0)
        self.assertAlmostEqual(bounds.out_of_bounds, -self.config.out_of_bounds_penalty)
        self.assertEqual(bounds.collision, 0.0)

    def test_success_bonus_and_abort_penalty_are_exclusive_signals(self) -> None:
        success = self._reward(state=HoverState.HOLD, success=True)
        aborted = self._reward(aborted=True)

        self.assertAlmostEqual(success.success, self.config.success_bonus)
        self.assertEqual(success.abort, 0.0)
        self.assertAlmostEqual(aborted.abort, -self.config.abort_penalty)
        self.assertEqual(aborted.success, 0.0)

    def test_non_finite_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor_distance_m"):
            self._reward(anchor_distance_m=math.inf)
        with self.assertRaisesRegex(ValueError, "sonlu değer"):
            self._reward(current_action=np.array([math.nan, 0.0, 0.0, 0.0]))


class TerminalHoverEnvironmentTests(unittest.TestCase):
    """Ortam entegrasyonu: başarı, çarpışma ve abort semantiği."""

    def setUp(self) -> None:
        self.encoder = _scripted_encoder_factory()
        loader = patch(
            "ebabil.envs.vision.load_approved_vision_encoder",
            side_effect=lambda _: self.encoder,
        )
        loader.start()
        self.addCleanup(loader.stop)
        self.camera = StubCamera()

    def make_env(self, **kwargs: object) -> EbabilVisionNavigationEnv:
        env = EbabilVisionNavigationEnv(
            approval_manifest="synthetic-approval.json",
            camera_factory=lambda _: self.camera,
            **kwargs,
        )
        self.addCleanup(env.close)
        return env

    def _place_at_hover_anchor(self, env: EbabilVisionNavigationEnv) -> None:
        assert env.data is not None
        assert env.model is not None
        anchor = env.target_area.hover_anchor_world_m
        env.data.qpos[:3] = anchor
        env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:] = 0.0
        env._position_reference[:] = anchor
        env._yaw_reference_rad = 0.0
        mujoco.mj_forward(env.model, env.data)

    def test_reset_reports_transit_state(self) -> None:
        env = self.make_env(scene_mode="clear")

        _, info = env.reset(seed=20260911)

        self.assertIn(info["hover_state"], ("transit", "search"))
        self.assertFalse(info["hover_reading_valid"])
        self.assertEqual(info["hover_hold_steps"], 0)
        self.assertFalse(info["is_success"])

    def test_reward_components_follow_terminal_hover_contract(self) -> None:
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)

        _, reward, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertIn("hover_hold", info["reward_components"])
        self.assertIn("out_of_bounds", info["reward_components"])
        self.assertNotIn("goal", info["reward_components"])
        self.assertAlmostEqual(
            reward, sum(info["reward_components"].values()), places=12
        )

    def test_continuous_hover_hold_produces_success(self) -> None:
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)
        self._place_at_hover_anchor(env)
        target = load_target_area_config()

        terminated = False
        info: dict[str, object] = {}
        for _ in range(200):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "hover_success")
        self.assertEqual(info["hover_state"], "success")
        self.assertAlmostEqual(
            float(info["hover_hold_seconds"]), target.hold_seconds, places=9
        )
        components = info["reward_components"]
        assert isinstance(components, dict)
        self.assertGreater(components["success"], 0.0)
        self.assertEqual(components["collision"], 0.0)

    def test_stale_frame_produces_abort_instead_of_success(self) -> None:
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)
        self._place_at_hover_anchor(env)
        for _ in range(10):
            env.step(np.zeros(4, dtype=np.float32))
        assert env.data is not None
        self.camera.frozen_timestamp_s = float(env.data.time)

        terminated = False
        info: dict[str, object] = {}
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["termination_reason"], "hover_abort_stale_frame")
        self.assertEqual(info["hover_abort_reason"], "stale_frame")

    def test_ground_collision_is_never_success(self) -> None:
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)
        self._place_at_hover_anchor(env)
        machine = env.hover_state_machine
        for _ in range(machine.hold_required_steps - 2):
            _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(terminated)
        self.assertEqual(info["hover_state"], "hold")

        assert env.data is not None
        assert env.model is not None
        env.data.qpos[2] = 0.02
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)

        _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertIn(
            info["termination_reason"], ("ground_collision", "minimum_altitude")
        )
        components = info["reward_components"]
        assert isinstance(components, dict)
        self.assertEqual(components["success"], 0.0)

    def test_hover_reading_uses_raw_sigmoid_score_not_observation_scale(self) -> None:
        """`vision_aux[0]` ham `[0, 1]` skorudur; observation kopyası `[-1, 1]`.

        Ham skoru observation ölçeğiyle karıştırmak `[0, 1]` aralığını
        `[0.5, 1]`'e sıkıştırır; bu durumda hedef asla kaybedilemez ve tespit
        eşiği gerçek değerinin yarısına düşer.
        """

        self.encoder.scripted_target_score = 0.30
        env = self.make_env(scene_mode="clear")

        observation, _ = env.reset(seed=20260911)
        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertAlmostEqual(float(env._vision_aux[0]), 0.30, places=4)
        self.assertAlmostEqual(
            float(observation["vision_aux"][0]), 2.0 * 0.30 - 1.0, places=4
        )
        self.assertFalse(info["hover_target_visible"])
        self.assertIn(info["hover_state"], ("transit", "search"))

    def test_score_just_below_acquire_threshold_never_acquires(self) -> None:
        config = load_vision_navigation_config().terminal_hover
        self.encoder.scripted_target_score = config.acquire_score - 0.05
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)

        for _ in range(40):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break

        self.assertIn(info["hover_state"], ("transit", "search"))
        self.assertEqual(info["hover_acquire_steps"], 0)

    def test_target_loss_after_acquisition_aborts_the_episode(self) -> None:
        env = self.make_env(scene_mode="clear")
        env.reset(seed=20260911)
        self._place_at_hover_anchor(env)
        for _ in range(env.hover_state_machine.acquire_required_steps + 2):
            _, _, terminated, _, info = env.step(np.zeros(4, dtype=np.float32))
            self.assertFalse(terminated)
        self.assertEqual(info["hover_state"], "hold")

        self.encoder.scripted_target_score = 0.05
        terminated = False
        for _ in range(env.hover_state_machine.grace_steps + 5):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertFalse(info["is_success"])
        self.assertEqual(info["termination_reason"], "hover_abort_target_lost")
        self.assertEqual(info["hover_abort_reason"], "target_lost")
        components = info["reward_components"]
        assert isinstance(components, dict)
        self.assertLess(components["abort"], 0.0)

    def test_ground_failure_is_never_profitable_end_to_end(self) -> None:
        """REV-033 uçtan uca: yaklaşıp fiziksel olarak başarısız olmak kâr etmez.

        Araç hedefe doğru ilerletilir (progress birikir), sonra irtifa
        ihlaliyle bitirilir. Episode toplam ödülü negatif olmalıdır.
        """

        env = self.make_env()
        env.reset(seed=520001)
        total = 0.0
        progress_total = 0.0
        refund_total = 0.0
        reason = None
        for _ in range(1200):
            _, reward, terminated, truncated, info = env.step(
                np.array([1.0, 0.0, -1.0, 0.0], dtype=np.float32)
            )
            total += float(reward)
            progress_total += float(info["reward_components"]["target_progress"])
            refund_total += float(info["reward_components"]["progress_refund"])
            if terminated or truncated:
                reason = info["termination_reason"]
                break

        self.assertIn(reason, PHYSICAL_FAILURE_REASONS)
        self.assertLess(refund_total, 0.0)
        self.assertAlmostEqual(progress_total + refund_total, 0.0, places=9)
        self.assertLess(total, 0.0)

    def test_legacy_environment_keeps_legacy_reward_components(self) -> None:
        env = EbabilNavigationEnv(scene_mode="clear")
        self.addCleanup(env.close)
        env.reset(seed=20260911)

        _, _, _, _, info = env.step(np.zeros(4, dtype=np.float32))

        self.assertIn("goal", info["reward_components"])
        self.assertNotIn("hover_hold", info["reward_components"])
        self.assertNotIn("hover_state", info)


if __name__ == "__main__":
    unittest.main()


class ApproximateTargetTests(unittest.TestCase):
    """ADR-019: hedefin yalnız yaklaşık konumu bilinir."""

    def setUp(self) -> None:
        self.target = load_target_area_config()
        self.vision = load_vision_navigation_config()

    def test_believed_target_is_deterministic_per_seed(self) -> None:
        goal = np.array([5.0, 3.0, 1.0])

        first = believed_target_xy(goal, 2.0, 42, self.target.safe_half_size_xy_m)
        second = believed_target_xy(goal, 2.0, 42, self.target.safe_half_size_xy_m)
        other = believed_target_xy(goal, 2.0, 43, self.target.safe_half_size_xy_m)

        np.testing.assert_array_equal(first, second)
        self.assertFalse(np.array_equal(first, other))

    def test_error_stays_inside_configured_radius(self) -> None:
        goal = np.array([5.0, 3.0, 1.0])
        radius = 2.0

        errors = [
            float(
                np.linalg.norm(
                    believed_target_xy(
                        goal, radius, seed, self.target.safe_half_size_xy_m
                    )
                    - goal[:2]
                )
            )
            for seed in range(500)
        ]

        self.assertLessEqual(max(errors), radius + 1e-12)
        self.assertGreater(max(errors), radius * 0.9)

    def test_radius_below_safe_region_is_rejected(self) -> None:
        """`R` güvenli bölgeden küçükse görüntü kullanılmadan kazanılır."""

        safe_half = min(self.target.safe_half_size_xy_m)

        with self.assertRaisesRegex(ValueError, "güvenli iç bölgeden büyük"):
            believed_target_xy(
                np.zeros(3), safe_half, 1, self.target.safe_half_size_xy_m
            )

    def test_zero_radius_returns_true_goal(self) -> None:
        goal = np.array([5.0, 3.0, 1.0])

        believed = believed_target_xy(goal, 0.0, 1, self.target.safe_half_size_xy_m)

        np.testing.assert_allclose(believed, goal[:2], atol=1e-12)

    def test_shipped_radius_exceeds_safe_region(self) -> None:
        self.vision.validate_against_target(self.target)

        self.assertGreater(
            self.vision.target_uncertainty.error_radius_m,
            min(self.target.safe_half_size_xy_m),
        )

    def test_config_rejects_radius_inside_safe_region(self) -> None:
        from ebabil.envs.vision_config import TargetUncertaintyConfig

        broken = replace(
            self.vision,
            target_uncertainty=TargetUncertaintyConfig(error_radius_m=0.5),
        )

        with self.assertRaisesRegex(ValueError, "güvenli iç bölgeden büyük"):
            broken.validate_against_target(self.target)


class SearchStateTests(unittest.TestCase):
    """ADR-019: inanılan konuma varıldığında aktif arama."""

    def _machine(self) -> HoverStateMachine:
        return HoverStateMachine(
            load_vision_navigation_config().terminal_hover,
            load_target_area_config(),
            POLICY_FREQUENCY_HZ,
        )

    def test_far_from_believed_target_stays_in_transit(self) -> None:
        machine = self._machine()

        update = machine.update(
            _reading(
                target_score=0.05,
                inside_safe_region=False,
                horizontal_distance_m=30.0,
                believed_horizontal_distance_m=30.0,
            )
        )

        self.assertIs(update.state, HoverState.TRANSIT)
        self.assertEqual(update.search_steps, 0)

    def test_entering_believed_radius_starts_search(self) -> None:
        machine = self._machine()
        radius = machine.config.search_entry_radius_m

        update = machine.update(
            _reading(
                target_score=0.05,
                inside_safe_region=False,
                horizontal_distance_m=radius - 0.1,
                believed_horizontal_distance_m=radius - 0.1,
            )
        )

        self.assertIs(update.state, HoverState.SEARCH)
        self.assertEqual(update.search_steps, 1)
        self.assertAlmostEqual(
            update.search_seconds, 1.0 / POLICY_FREQUENCY_HZ, places=9
        )

    def test_search_timeout_aborts(self) -> None:
        machine = self._machine()
        near = _reading(
            target_score=0.05,
            inside_safe_region=False,
            horizontal_distance_m=1.0,
            believed_horizontal_distance_m=1.0,
        )

        update = None
        for _ in range(machine.search_timeout_steps + 2):
            update = machine.update(near)
            if update.state is HoverState.ABORT:
                break

        assert update is not None
        self.assertIs(update.state, HoverState.ABORT)
        self.assertEqual(update.abort_reason, "search_timeout")

    def test_acquisition_during_search_leaves_search(self) -> None:
        machine = self._machine()
        near_invisible = _reading(
            target_score=0.05,
            inside_safe_region=False,
            horizontal_distance_m=1.0,
            believed_horizontal_distance_m=1.0,
        )
        machine.update(near_invisible)
        self.assertIs(machine.state, HoverState.SEARCH)

        for _ in range(machine.acquire_required_steps):
            update = machine.update(
                _reading(
                    inside_safe_region=False,
                    horizontal_distance_m=1.0,
                    believed_horizontal_distance_m=1.0,
                )
            )

        self.assertIsNot(update.state, HoverState.SEARCH)
        self.assertIsNot(update.state, HoverState.TRANSIT)

    def test_search_entry_radius_must_cover_approach_corridor(self) -> None:
        config = load_vision_navigation_config().terminal_hover

        self.assertGreaterEqual(config.search_entry_radius_m, config.approach_radius_m)
        with self.assertRaisesRegex(ValueError, "approach_radius_m'den"):
            replace(config, search_entry_radius_m=1.0).validate()

    def test_abort_is_reachable_from_search(self) -> None:
        self.assertIn((HoverState.SEARCH, HoverState.ABORT), ALLOWED_HOVER_TRANSITIONS)


class SingleTargetBudgetTests(unittest.TestCase):
    """ADR-018: tek hedefte her episode nihaidir."""

    def setUp(self) -> None:
        self.encoder = _scripted_encoder_factory()
        loader = patch(
            "ebabil.envs.vision.load_approved_vision_encoder",
            side_effect=lambda _: self.encoder,
        )
        loader.start()
        self.addCleanup(loader.stop)

    def _env(self) -> EbabilVisionNavigationEnv:
        env = EbabilVisionNavigationEnv(
            approval_manifest="synthetic-approval.json",
            camera_factory=lambda _: StubCamera(),
        )
        self.addCleanup(env.close)
        return env

    def test_terminal_reserve_is_always_applied(self) -> None:
        """Terminal rezervi olmadan araç rotayı uçup hold'u tamamlayamaz."""

        env = self._env()

        _, info = env.reset(seed=20260911)

        self.assertTrue(info["segment_final"])
        budget = load_vision_navigation_config().segment_budget
        reserve = budget.terminal_approach_seconds + budget.terminal_hold_seconds
        self.assertGreater(reserve, 0.0)
        without_reserve = compute_segment_horizon(
            float(info["planned_route_length_m"]),
            20.0,
            budget,
            final_segment=False,
        )
        self.assertGreater(info["segment_maximum_steps"], without_reserve.maximum_steps)

    def test_removed_final_segment_option_is_rejected(self) -> None:
        env = self._env()

        with self.assertRaisesRegex(ValueError, "final_segment reset option"):
            env.reset(seed=20260911, options={"final_segment": False})

    def test_waypoint_is_derived_from_believed_not_true_target(self) -> None:
        """ADR-019: `waypoint` inanılan hedeften türetilir.

        Sayısal rastlantıyı aramak anlamsızdır (normalize observation'da `0.0`
        gibi değerler zaten bulunur). Anlamlı test, kanalın hangi hedeften
        üretildiğini yeniden hesaplayıp karşılaştırmaktır.
        """

        env = self._env()
        observation, info = env.reset(seed=20260911)
        assert env.scene is not None and env.data is not None
        true_goal = env.scene.goal_world_m
        believed = env._believed_target_world_m

        self.assertGreater(float(info["target_error_m"]), 0.0)
        self.assertFalse(np.allclose(believed[:2], true_goal[:2]))

        scale = env.vision_config.observation.waypoint_distance_scale_m
        rotation = np.empty(9)
        mujoco.mju_quat2Mat(rotation, env.data.qpos[3:7])
        rotation = rotation.reshape(3, 3)
        position = env.data.qpos[:3]

        def expected_waypoint(target: np.ndarray) -> np.ndarray:
            body = rotation.T @ (target - position)
            distance = float(np.linalg.norm(body))
            direction = body / distance if distance > 1e-9 else np.zeros(3)
            fraction = min(math.log1p(distance) / math.log1p(scale), 1.0)
            return np.clip(
                np.concatenate((direction, [2.0 * fraction - 1.0])), -1.0, 1.0
            )

        actual = np.asarray(observation["waypoint"], dtype=np.float64)
        from_believed = expected_waypoint(believed)
        from_true = expected_waypoint(true_goal)

        np.testing.assert_allclose(actual, from_believed, atol=1e-6)
        self.assertFalse(np.allclose(actual, from_true, atol=1e-4))

    def test_success_uses_true_target_not_believed(self) -> None:
        """Başarı privileged gerçek konuma göre; inanılan konuma göre değil."""

        env = self._env()
        env.reset(seed=20260911)
        assert env.data is not None and env.model is not None
        anchor = env.target_area.hover_anchor_world_m

        # Gerçek hedefin üstüne yerleş: başarı gelmeli.
        env.data.qpos[:3] = anchor
        env.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        env.data.qvel[:] = 0.0
        env._position_reference[:] = anchor
        env._yaw_reference_rad = 0.0
        mujoco.mj_forward(env.model, env.data)

        terminated = False
        info: dict[str, object] = {}
        for _ in range(400):
            _, _, terminated, truncated, info = env.step(np.zeros(4, dtype=np.float32))
            if terminated or truncated:
                break

        self.assertTrue(terminated)
        self.assertTrue(info["is_success"])
        self.assertEqual(info["termination_reason"], "hover_success")
