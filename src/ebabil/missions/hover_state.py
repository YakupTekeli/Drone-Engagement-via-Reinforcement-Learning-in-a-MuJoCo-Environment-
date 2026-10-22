"""ADR-012 hedef üstü terminal hover durum makinesi."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from ebabil.scenes import TargetAreaConfig


class HoverState(StrEnum):
    """ADR-012 terminal görev durumları."""

    TRANSIT = "transit"
    SEARCH = "search"
    ACQUIRED = "acquired"
    APPROACH = "approach"
    SETTLE = "settle"
    HOLD = "hold"
    SUCCESS = "success"
    ABORT = "abort"


#: Durum zincirinin nominal ilerleyiş sırası. Koşullar iç içe geçmeli olduğu
#: için araç zincirde ileri ve geri hareket edebilir; atlama koşulların aynı
#: adımda birden fazla seviye iyileşmesi anlamına gelir.
HOVER_STATE_CHAIN: tuple[HoverState, ...] = (
    HoverState.TRANSIT,
    HoverState.SEARCH,
    HoverState.ACQUIRED,
    HoverState.APPROACH,
    HoverState.SETTLE,
    HoverState.HOLD,
)

#: Terminal durumlar; bunlardan çıkış yoktur.
TERMINAL_HOVER_STATES: frozenset[HoverState] = frozenset(
    {HoverState.SUCCESS, HoverState.ABORT}
)

#: Durum makinesinin ürettiği abort sebepleri. Çarpışma, workspace ihlali,
#: minimum irtifa ve tutum ihlali ortamın fiziksel failure yolundan gelir.
HOVER_ABORT_REASONS: tuple[str, ...] = (
    "target_lost",
    "stale_frame",
    "search_timeout",
)


def allowed_hover_transitions() -> frozenset[tuple[HoverState, HoverState]]:
    """İzinli `(önceki, sonraki)` durum geçişlerini üret.

    Zincir içinde serbest hareket vardır: durumlar koşullardan türetildiği ve
    koşullar iç içe geçmeli olduğu için araç bir adımda birden fazla seviye
    iyileşebilir veya geri düşebilir. Buna karşılık iki kısıt katıdır:

    - `SUCCESS` yalnız `HOLD` durumundan girilir; kesintisiz hold sayacı
      tamamlanmadan hiçbir durumdan başarıya geçilemez.
    - Terminal durumlar soğurucudur; `SUCCESS` ve `ABORT` dışına çıkış yoktur.

    `ABORT` her zincir durumundan mümkündür. `stale_frame` her durumda,
    `target_lost` yalnız tespit sonrasında abort üretir.
    """

    allowed: set[tuple[HoverState, HoverState]] = set()
    for state in HOVER_STATE_CHAIN:
        for other in HOVER_STATE_CHAIN:
            allowed.add((state, other))
        allowed.add((state, HoverState.ABORT))
    allowed.add((HoverState.HOLD, HoverState.SUCCESS))
    return frozenset(allowed)


ALLOWED_HOVER_TRANSITIONS = allowed_hover_transitions()


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} pozitif ve sonlu olmalı")
    return value


@dataclass(frozen=True)
class TerminalHoverConfig:
    """Görsel süreklilik ve yaklaşma koridoru parametreleri."""

    acquire_score: float
    loss_score: float
    acquire_seconds: float
    grace_seconds: float
    approach_radius_m: float
    search_entry_radius_m: float
    search_timeout_seconds: float
    #: ADR-017 Revizyon 1 (30 Eylül 2026, ADR-025 ile): hedef arama yarıçapının
    #: dışında (`believed_horizontal_distance_m > search_entry_radius_m`) iken
    #: `acquired` durumunda kaybedilirse abort yerine `transit`'e dönülür;
    #: uzaktan bir bakışla görülen hedefin kadrajdan çıkması görevi bitirmez.
    #: Varsayılan `False` v0/v1 davranışını korur.
    far_loss_returns_to_transit: bool = False

    def validate(self) -> None:
        for name, value in (
            ("acquire_score", self.acquire_score),
            ("loss_score", self.loss_score),
        ):
            if not math.isfinite(value) or not 0.0 < value < 1.0:
                raise ValueError(f"terminal_hover.{name} (0, 1) aralığında olmalı")
        if self.loss_score >= self.acquire_score:
            raise ValueError(
                "terminal_hover.loss_score acquire_score'dan küçük olmalı; "
                "histerezis olmadan skor gürültüsü durum titremesi üretir"
            )
        _positive(self.acquire_seconds, "terminal_hover.acquire_seconds")
        _positive(self.grace_seconds, "terminal_hover.grace_seconds")
        _positive(self.approach_radius_m, "terminal_hover.approach_radius_m")
        _positive(self.search_entry_radius_m, "terminal_hover.search_entry_radius_m")
        _positive(self.search_timeout_seconds, "terminal_hover.search_timeout_seconds")
        if self.search_entry_radius_m < self.approach_radius_m:
            raise ValueError(
                "terminal_hover.search_entry_radius_m approach_radius_m'den "
                "küçük olamaz; arama yaklaşma koridorunu kapsamalıdır"
            )

    def validate_against_target(self, target_config: TargetAreaConfig) -> None:
        """Yaklaşma koridorunun güvenli iç bölgeyi kapsadığını doğrula.

        Durum koşullarının iç içe geçmeli olması bu koşula bağlıdır: güvenli iç
        bölgedeki her nokta aynı zamanda yaklaşma koridorunda olmalıdır. Aksi
        halde araç `settle` koşulunu sağlarken `approach` koşulunu sağlamaz ve
        zincir tutarsız hâle gelir.
        """

        target_config.validate()
        safe_half_x, safe_half_y = target_config.safe_half_size_xy_m
        safe_diagonal_m = math.hypot(safe_half_x, safe_half_y)
        if self.approach_radius_m <= safe_diagonal_m:
            raise ValueError(
                "terminal_hover.approach_radius_m güvenli iç bölge köşegenini "
                f"aşmalı: {self.approach_radius_m:.3f} m <= {safe_diagonal_m:.3f} m"
            )


@dataclass(frozen=True)
class HoverConditionReading:
    """Tek policy adımının durum makinesi girdisi.

    `target_score` actor yolundan (encoder) gelir. Geometrik alanlar ADR-012 ve
    ADR-016 uyarınca privileged ground truth'tan hesaplanır ve actor
    observation'ına girmez.
    """

    target_score: float
    frame_stale: bool
    horizontal_distance_m: float
    believed_horizontal_distance_m: float
    inside_safe_region: bool
    altitude_error_m: float
    speed_m_s: float
    tilt_rad: float
    angular_rate_rad_s: float

    def validate(self) -> None:
        values = {
            "target_score": self.target_score,
            "horizontal_distance_m": self.horizontal_distance_m,
            "believed_horizontal_distance_m": self.believed_horizontal_distance_m,
            "altitude_error_m": self.altitude_error_m,
            "speed_m_s": self.speed_m_s,
            "tilt_rad": self.tilt_rad,
            "angular_rate_rad_s": self.angular_rate_rad_s,
        }
        for name, value in values.items():
            if not math.isfinite(value):
                raise ValueError(f"HoverConditionReading.{name} sonlu olmalı")
        if not 0.0 <= self.target_score <= 1.0:
            raise ValueError("target_score [0, 1] aralığında olmalı")
        for name in (
            "horizontal_distance_m",
            "believed_horizontal_distance_m",
            "altitude_error_m",
            "speed_m_s",
            "tilt_rad",
            "angular_rate_rad_s",
        ):
            if values[name] < 0.0:
                raise ValueError(f"HoverConditionReading.{name} negatif olamaz")


@dataclass(frozen=True)
class HoverStateUpdate:
    """Bir adım sonrası durum makinesi çıktısı."""

    state: HoverState
    previous_state: HoverState
    success: bool
    abort_reason: str | None
    hold_steps: int
    hold_seconds: float
    hold_progress: float
    acquire_steps: int
    search_steps: int
    search_seconds: float
    loss_steps: int
    loss_seconds: float
    hover_conditions_met: bool
    target_visible: bool

    def as_info(self) -> dict[str, object]:
        return {
            "hover_state": str(self.state),
            "hover_previous_state": str(self.previous_state),
            "hover_abort_reason": self.abort_reason,
            "hover_hold_steps": self.hold_steps,
            "hover_hold_seconds": self.hold_seconds,
            "hover_hold_progress": self.hold_progress,
            "hover_acquire_steps": self.acquire_steps,
            "hover_search_steps": self.search_steps,
            "hover_search_seconds": self.search_seconds,
            "hover_loss_steps": self.loss_steps,
            "hover_loss_seconds": self.loss_seconds,
            "hover_conditions_met": self.hover_conditions_met,
            "hover_target_visible": self.target_visible,
        }


class HoverStateMachine:
    """`transit -> search -> acquired -> approach -> settle -> hold -> success`.

    Görsel süreklilik yalnız `target_score` ile kapılanır. ADR-016 uyarınca
    `target_center_xy` kanalı kapı olarak kullanılmaz: hedef merkezi hover
    pozunda kamera kadrajı dışındadır ve görünmediğinde `(0, 0)` sentinel'i
    "tam merkezde" durumundan ayırt edilemez.
    """

    def __init__(
        self,
        config: TerminalHoverConfig,
        target_config: TargetAreaConfig,
        policy_frequency_hz: float,
    ) -> None:
        config.validate()
        target_config.validate()
        config.validate_against_target(target_config)
        _positive(policy_frequency_hz, "policy_frequency_hz")
        self.config = config
        self.target_config = target_config
        self.policy_frequency_hz = float(policy_frequency_hz)
        self._policy_period_s = 1.0 / self.policy_frequency_hz
        self.acquire_required_steps = math.ceil(
            config.acquire_seconds * self.policy_frequency_hz
        )
        self.grace_steps = math.ceil(config.grace_seconds * self.policy_frequency_hz)
        self.search_timeout_steps = math.ceil(
            config.search_timeout_seconds * self.policy_frequency_hz
        )
        self.hold_required_steps = math.ceil(
            target_config.hold_seconds * self.policy_frequency_hz
        )
        self._state = HoverState.TRANSIT
        self._hold_steps = 0
        self._acquire_steps = 0
        self._loss_steps = 0
        self._search_steps = 0

    @property
    def state(self) -> HoverState:
        return self._state

    def reset(self) -> None:
        self._state = HoverState.TRANSIT
        self._hold_steps = 0
        self._acquire_steps = 0
        self._loss_steps = 0
        self._search_steps = 0

    def hover_conditions_met(self, reading: HoverConditionReading) -> bool:
        """ADR-012 hover koşullarının tamamının birlikte sağlanması."""

        return bool(
            reading.inside_safe_region
            and reading.altitude_error_m <= self.target_config.altitude_tolerance_m
            and reading.speed_m_s <= self.target_config.maximum_linear_speed_m_s
            and reading.tilt_rad <= self.target_config.maximum_tilt_rad
            and reading.angular_rate_rad_s
            <= self.target_config.maximum_angular_rate_rad_s
        )

    def update(self, reading: HoverConditionReading) -> HoverStateUpdate:
        """Bir policy adımı ilerlet ve yeni durumu üret."""

        reading.validate()
        previous_state = self._state
        if previous_state in TERMINAL_HOVER_STATES:
            raise RuntimeError(
                f"Terminal durumdan ({previous_state}) sonra update çağrılamaz; "
                "reset gerekir"
            )

        visible = reading.target_score >= self.config.acquire_score
        lost = reading.target_score < self.config.loss_score
        self._loss_steps = self._loss_steps + 1 if lost else 0
        conditions_met = self.hover_conditions_met(reading)

        if reading.frame_stale:
            self._state = HoverState.ABORT
            return self._result(
                previous_state, conditions_met, visible, abort_reason="stale_frame"
            )

        if previous_state in (HoverState.TRANSIT, HoverState.SEARCH):
            # ADR-019: araç inanılan hedef konumuna yaklaştığında aktif aramaya
            # geçer. Hedefin gerçek konumu inanılan konumdan `R` kadar uzakta
            # olabileceği için tespit garantili değildir.
            if (
                reading.believed_horizontal_distance_m
                <= self.config.search_entry_radius_m
            ):
                self._search_steps += 1
            self._acquire_steps = self._acquire_steps + 1 if visible else 0
            if self._acquire_steps < self.acquire_required_steps:
                self._hold_steps = 0
                if self._search_steps > self.search_timeout_steps:
                    self._state = HoverState.ABORT
                    return self._result(
                        previous_state,
                        conditions_met,
                        visible,
                        abort_reason="search_timeout",
                    )
                if self._search_steps > 0:
                    self._state = HoverState.SEARCH
                return self._result(previous_state, conditions_met, visible)
        elif self._loss_steps > self.grace_steps:
            if (
                self.config.far_loss_returns_to_transit
                and previous_state is HoverState.ACQUIRED
                and reading.believed_horizontal_distance_m
                > self.config.search_entry_radius_m
            ):
                # Arama yarıçapının dışında erken tespit kaybı: zincir başa döner.
                self._state = HoverState.TRANSIT
                self._acquire_steps = 0
                self._loss_steps = 0
                self._hold_steps = 0
                return self._result(previous_state, conditions_met, visible)
            self._state = HoverState.ABORT
            return self._result(
                previous_state, conditions_met, visible, abort_reason="target_lost"
            )

        self._state = self._chain_state(reading, conditions_met)
        if self._state is HoverState.HOLD:
            self._hold_steps += 1
            # `SUCCESS` yalnız `HOLD` durumundan girilir: hold sayacı aynı
            # adımda girilen hold ile tamamlanamaz.
            if (
                previous_state is HoverState.HOLD
                and self._hold_steps >= self.hold_required_steps
            ):
                self._state = HoverState.SUCCESS
        else:
            # ADR-012: hold koşulu bozulduğunda sayaç sıfırlanır.
            self._hold_steps = 0
        return self._result(previous_state, conditions_met, visible)

    def _chain_state(
        self, reading: HoverConditionReading, conditions_met: bool
    ) -> HoverState:
        """Koşullardan en ileri geçerli zincir durumunu türet."""

        if conditions_met:
            return HoverState.HOLD
        if reading.inside_safe_region:
            return HoverState.SETTLE
        if reading.horizontal_distance_m <= self.config.approach_radius_m:
            return HoverState.APPROACH
        return HoverState.ACQUIRED

    def _result(
        self,
        previous_state: HoverState,
        conditions_met: bool,
        target_visible: bool,
        abort_reason: str | None = None,
    ) -> HoverStateUpdate:
        transition = (previous_state, self._state)
        if transition not in ALLOWED_HOVER_TRANSITIONS:
            raise RuntimeError(f"İzinsiz hover durum geçişi: {transition}")
        if self._state is HoverState.SUCCESS and (
            self._hold_steps < self.hold_required_steps or not conditions_met
        ):
            raise RuntimeError(
                "Başarı yalnız kesintisiz hold sayacı tamamlandığında üretilebilir"
            )
        return HoverStateUpdate(
            state=self._state,
            previous_state=previous_state,
            success=self._state is HoverState.SUCCESS,
            abort_reason=abort_reason,
            hold_steps=self._hold_steps,
            hold_seconds=self._hold_steps * self._policy_period_s,
            hold_progress=min(1.0, self._hold_steps / self.hold_required_steps),
            acquire_steps=self._acquire_steps,
            search_steps=self._search_steps,
            search_seconds=self._search_steps * self._policy_period_s,
            loss_steps=self._loss_steps,
            loss_seconds=self._loss_steps * self._policy_period_s,
            hover_conditions_met=conditions_met,
            target_visible=target_visible,
        )


__all__ = [
    "ALLOWED_HOVER_TRANSITIONS",
    "HOVER_ABORT_REASONS",
    "HOVER_STATE_CHAIN",
    "TERMINAL_HOVER_STATES",
    "HoverConditionReading",
    "HoverState",
    "HoverStateMachine",
    "HoverStateUpdate",
    "TerminalHoverConfig",
    "allowed_hover_transitions",
]
