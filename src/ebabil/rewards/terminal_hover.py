"""ADR-012 terminal hover reward bileşenleri."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from ebabil.missions.hover_state import HoverState


def _non_negative(value: float, name: str) -> float:
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} negatif olmayan ve sonlu olmalı")
    return value


def _positive(value: float, name: str) -> float:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} pozitif ve sonlu olmalı")
    return value


@dataclass(frozen=True)
class TerminalHoverRewardConfig:
    """Terminal hover reward ağırlıkları; hepsi ayrı loglanır."""

    target_progress_weight: float
    target_visible_weight: float
    cruise_altitude_weight: float
    bearing_alignment_weight: float
    cruise_altitude_m: float
    target_alignment_weight: float
    settle_speed_weight: float
    settle_altitude_weight: float
    settle_tilt_weight: float
    settle_angular_rate_weight: float
    hover_hold_weight: float
    proximity_weight: float
    proximity_threshold_m: float
    smoothness_weight: float
    success_bonus: float
    collision_penalty: float
    out_of_bounds_penalty: float
    abort_penalty: float
    time_penalty: float

    def validate(self) -> None:
        for name, value in vars(self).items():
            _non_negative(float(value), f"terminal_hover_reward.{name}")
        if self.proximity_threshold_m <= 0.0:
            raise ValueError(
                "terminal_hover_reward.proximity_threshold_m pozitif olmalı"
            )
        _positive(self.cruise_altitude_m, "terminal_hover_reward.cruise_altitude_m")
        if self.target_visible_weight >= self.time_penalty:
            raise ValueError(
                "terminal_hover_reward.target_visible_weight time_penalty'den "
                "küçük olmalı; aksi halde hedefi seyrederek beklemek adım "
                "başına net pozitif olur ve görevi bitirme baskısı kalkar: "
                f"{self.target_visible_weight:.4f} >= {self.time_penalty:.4f}"
            )

    def detour_saving_for(self, detour_length_m: float) -> float:
        """Bir engelin çevresinden dolaşmak yerine içinden geçmenin kazancı.

        REV-029: REV-026'nın "toplam progress collision_penalty'yi aşmamalı"
        invariantı **yanlış kurulmuştu.** Çarpışma episode'u bitirir, yani
        kıyaslanacak şey episode toplamı değil, kısa yolun **yerel** kazancıdır:
        engelin çevresinden dolaşmak yerine içinden geçmek `detour_length_m`
        kadar mesafe kazandırır ve bu, metre başına ağırlıkla fiyatlanır.

        Bu nedenle kısıt yoğunluk üzerindedir ve rota uzunluğundan **doğal
        olarak bağımsızdır**; normalizasyon gerekmez.
        """

        _positive(detour_length_m, "detour_length_m")
        return self.target_progress_weight * detour_length_m

    def uninterrupted_hold_reward(self, hold_required_steps: int) -> float:
        """**Kesintisiz** bir hold dizisinin toplayacağı hold reward'ı.

        Bu bir episode üst sınırı **değildir**. Hold sayacı koşul bozulduğunda
        sıfırlanır fakat durum yeniden `HOLD` olabilir ve her `HOLD` adımı
        yeniden reward verir; sayacın sıfırlanması verilmiş reward'ı geri almaz.
        Bu nedenle bir episode'da toplanan hold reward'ı bu değerin katlarına
        çıkabilir (REV-022). D0-V koşusunda gözlenen başarı episode'u `103`
        `HOLD` adımı biriktirdi ve `41.20` hold reward'ı aldı; bu, kesintisiz
        dizi değerinin `1.7` katı ve `success_bonus`'un `1.37` katıdır.

        Episode düzeyindeki gerçek üst sınır için `episode_hold_reward_bound`
        kullanılmalıdır.
        """

        if hold_required_steps <= 0:
            raise ValueError("hold_required_steps pozitif olmalı")
        return self.hover_hold_weight * float(hold_required_steps)

    def episode_hold_reward_bound(self, maximum_episode_steps: int) -> float:
        """Bir episode'da toplanabilecek hold reward'ının gerçek üst sınırı.

        Politikanın episode boyunca `HOLD` koşullarını sağlaması hâlinde
        biriken toplam. ADR-012'nin "hold reward'u sınırlı olacaktır" koşulu bu
        sınırla değerlendirilmelidir.
        """

        if maximum_episode_steps <= 0:
            raise ValueError("maximum_episode_steps pozitif olmalı")
        return self.hover_hold_weight * float(maximum_episode_steps)


@dataclass(frozen=True)
class TerminalHoverRewardBreakdown:
    """ADR-012'nin istediği ayrık bileşenler."""

    target_progress: float
    progress_refund: float
    cruise_altitude: float
    bearing_alignment: float
    target_visible: float
    target_alignment: float
    settle_speed: float
    settle_altitude: float
    settle_tilt: float
    settle_angular_rate: float
    hover_hold: float
    proximity: float
    smoothness: float
    collision: float
    out_of_bounds: float
    success: float
    abort: float
    time: float

    @property
    def total(self) -> float:
        return float(sum(vars(self).values()))

    def as_dict(self) -> dict[str, float]:
        return {name: float(value) for name, value in vars(self).items()}


#: Hizalama bileşeninin verilebileceği durumlar. ADR-016 uyarınca `settle` ve
#: `hold` durumlarında hedef merkezi kadraj dışında olduğu için bu bileşen
#: kapatılır; aksi halde sistematik kenar kayması ödüllenir.
ALIGNMENT_STATES: frozenset[HoverState] = frozenset(
    {HoverState.ACQUIRED, HoverState.APPROACH}
)

#: Seyir fazı. REV-036: kameranın ileri görüş mesafesi irtifayla doğrusaldır
#: (`3.711 x irtifa`, ADR-016 zemin izi uzak sınırı), bu yüzden alçak uçmak
#: aracı kör eder. `cruise_altitude` yalnız bu durumlarda uygulanır; terminal
#: durumlarda araç hold bandına (`1.31-1.71 m`) İNMEK zorundadır ve orada
#: ceza verilmemelidir.
CRUISE_STATES: frozenset[HoverState] = frozenset(
    {HoverState.TRANSIT, HoverState.SEARCH}
)

#: REV-041: kamerayı inanılan hedefe DOĞRULTMANIN ödüllendirildiği durumlar.
#: ADR-020 holonomik hareketi açtığı için yön hareketten ayrıldı: araç yana
#: kayarak hedefe gidebiliyor fakat başka yöne bakıyor. Ölçülen: hedefin
#: `0-3 m` yakınında yön hatası `101-123 derece` ve `hold` durumuna 300
#: episode'da hiç girilmedi.
#:
#: `transit` KAPSAM DIŞIDIR ve bu bilinçlidir: seyirde kamera gittiği yöne
#: bakmalı, yoksa kör alana uçar (ölçülen: görüş konisi dışında `2 m`'den
#: yakın engel sayısı görünür alandakinin 3 katı). Hedefe dönmek terminal
#: fazın işidir.
BEARING_STATES: frozenset[HoverState] = frozenset(
    {
        HoverState.SEARCH,
        HoverState.ACQUIRED,
        HoverState.APPROACH,
        HoverState.SETTLE,
        HoverState.HOLD,
    }
)

#: REV-046: irtifa ve hızın şekillendirildiği durumlar. Araç burada hâlâ
#: hareket hâlindedir; bu yüzden ikisi de MESAFEYE GÖRE ölçeklenen bir payla
#: uygulanır (bkz. `_speed_allowance_m_s` ve `altitude_allowance_m`).
#:
#: `SEARCH` dışarıdadır: orası hâlâ seyir durumudur ve `cruise_altitude`
#: aracı yukarı iter. İkisi aynı durumda açık olursa hiçbir irtifa cezasız
#: kalmaz; ölçüldü, `minimum_altitude` terminasyonu `%79`'a çıktı.
APPROACH_STATES: frozenset[HoverState] = frozenset(
    {
        HoverState.ACQUIRED,
        HoverState.APPROACH,
        HoverState.SETTLE,
        HoverState.HOLD,
    }
)

#: REV-046: aracın DURMASI gereken durumlar. Tilt ve açısal hız yalnız burada
#: cezalanır.
#:
#: `ACQUIRED`/`APPROACH` bilinçli olarak dışarıdadır ve bu kritik:
#: `bearing_alignment` o durumlarda kamerayı hedefe DÖNDÜRMEYİ ödüllendirir,
#: dönmek ise açısal hız sınırını (`0.35 rad/s`, action üst sınırı
#: `1.5 rad/s`) aşmayı gerektirir. İkisi aynı durumda açık olursa araç aynı
#: anda dönmeye ve dönmemeye zorlanır. Tilt de benzer: ölçülen denge tilt'i
#: tam hızda `3.4 derece` (sınır `10`) fakat hızlanma/yavaşlama geçişlerinde
#: aşılır ve yaklaşma bunu gerektirir.
STILL_STATES: frozenset[HoverState] = frozenset({HoverState.SETTLE, HoverState.HOLD})

if CRUISE_STATES & APPROACH_STATES:
    raise AssertionError(
        "CRUISE_STATES ve APPROACH_STATES ayrık olmalı; aksi halde irtifa "
        "cezaları aynı durumda zıt yönlere çeker"
    )
if not STILL_STATES <= APPROACH_STATES:
    raise AssertionError("STILL_STATES, APPROACH_STATES altkümesi olmalı")


def compute_terminal_hover_reward(
    config: TerminalHoverRewardConfig,
    *,
    state: HoverState,
    previous_anchor_distance_m: float,
    anchor_distance_m: float,
    target_center_x: float,
    target_visible: bool,
    inside_safe_region: bool,
    speed_m_s: float,
    speed_allowance_m_s: float = 0.0,
    altitude_error_m: float = 0.0,
    altitude_allowance_m: float = 0.0,
    tilt_rad: float = 0.0,
    maximum_tilt_rad: float = 0.0,
    angular_rate_rad_s: float = 0.0,
    maximum_angular_rate_rad_s: float = 0.0,
    minimum_range_m: float,
    current_action: npt.ArrayLike,
    previous_action: npt.ArrayLike,
    success: bool,
    collision: bool,
    out_of_bounds: bool,
    aborted: bool,
    altitude_m: float = 0.0,
    believed_bearing_error_rad: float = 0.0,
    physical_failure: bool = False,
    banked_progress: float = 0.0,
) -> TerminalHoverRewardBreakdown:
    """Tek policy adımı için bileşen bazlı terminal hover reward'ı üret.

    `target_progress` hover anchor'a olan 3B mesafenin adım başına azalmasını
    metre başına sabit bir ağırlıkla ödüllendirir. Ağırlık rota uzunluğuna
    **normalize edilmez** (REV-029): normalizasyon iki koşuda ölçülerek
    reddedildi. Kısıt yoğunluk üzerindedir (`detour_saving_for`), çünkü
    çarpışma episode'u bitirir ve kıyaslanacak şey episode toplamı değil kısa
    yolun yerel kazancıdır. Bu kısıt rota uzunluğundan doğal olarak bağımsızdır.

    `target_visible` hedef kamerada görünürken adım başına küçük pozitif ödül
    verir (REV-027). Bundan önce görünürlüğü sağlayan tek mekanizma
    `target_lost` abort'unun sert cezasıydı ve bu kazara bir şekillendirme
    sinyali işlevi görüyordu; grace süresi uzatıldığında bu baskı kalktı ve
    politika bütçeyi doldurmaya başladı. Ağırlık `time_penalty`'nin altında
    tutulur, yani hedefi seyrederek beklemek adım başına hâlâ zararlıdır.
    Bu üst sınır `validate()` içinde zorlanır. `HOLD` durumundaki
    `hover_hold` ödülü bu sınıra tabi değildir ve olması gereken de budur.

    `target_alignment` yalnız hedef görünürken ve `acquired`/`approach`
    durumlarında uygulanır (ADR-016). `settle_speed` yalnız güvenli iç bölgede
    uygulanır ve yerleşmeyi teşvik eder. `hover_hold` yalnız `HOLD` durumunda
    verilir; hedef dışında bekleyerek biriktirilemez.
    """

    config.validate()
    action = np.asarray(current_action, dtype=np.float64)
    prior_action = np.asarray(previous_action, dtype=np.float64)
    for name, value in (
        ("previous_anchor_distance_m", previous_anchor_distance_m),
        ("anchor_distance_m", anchor_distance_m),
        ("target_center_x", target_center_x),
        ("speed_m_s", speed_m_s),
        ("speed_allowance_m_s", speed_allowance_m_s),
        ("altitude_error_m", altitude_error_m),
        ("altitude_allowance_m", altitude_allowance_m),
        ("tilt_rad", tilt_rad),
        ("maximum_tilt_rad", maximum_tilt_rad),
        ("angular_rate_rad_s", angular_rate_rad_s),
        ("maximum_angular_rate_rad_s", maximum_angular_rate_rad_s),
        ("minimum_range_m", minimum_range_m),
        ("altitude_m", altitude_m),
        ("believed_bearing_error_rad", believed_bearing_error_rad),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} sonlu olmalı")
    if not np.all(np.isfinite(action)) or not np.all(np.isfinite(prior_action)):
        raise ValueError("Action dizileri yalnız sonlu değer içermeli")
    if not math.isfinite(banked_progress):
        raise ValueError("banked_progress sonlu olmalı")
    progress = config.target_progress_weight * (
        previous_anchor_distance_m - anchor_distance_m
    )
    # REV-033: fiziksel başarısızlıkla biten episode toplanan progress'i geri
    # verir. Ölçülmüş defekt: progress bir kez toplanıp bankada kalıyordu ve
    # başarıyla gerekçelendirilmesi gerekmiyordu, bu yüzden "hedefe git sonra
    # yere çak" net +7 ödül veriyordu (178/300 episode).
    refund = -(float(banked_progress) + progress) if physical_failure else 0.0
    visible = config.target_visible_weight if target_visible else 0.0
    # `believed_bearing_error_rad` inanılan hedeften türetilir, yani
    # ayrıcalıklı değildir: araç `waypoint` kanalından bu yönü zaten biliyor.
    bearing = (
        -config.bearing_alignment_weight
        * min(abs(float(believed_bearing_error_rad)), math.pi)
        / math.pi
        if state in BEARING_STATES
        else 0.0
    )
    cruise = (
        -config.cruise_altitude_weight
        * max(0.0, config.cruise_altitude_m - float(altitude_m))
        if state in CRUISE_STATES
        else 0.0
    )
    alignment = (
        -config.target_alignment_weight * float(np.square(target_center_x))
        if target_visible and state in ALIGNMENT_STATES
        else 0.0
    )
    # REV-042: yavaşlama cezası artık yalnız güvenli iç bölgede değil, bütün
    # terminal fazda uygulanır ve mesafeyle ölçeklenen bir hız PAYINI aşan
    # kısmı cezalandırır. Ölçülen defekt: ceza yalnız `inside_safe_region`
    # iken açıktı, araç oraya terminal adımların `%2`'sinde girebiliyordu ve
    # hız `%99.9` ihlal ediliyordu. Yani yavaşlama teşviki, ancak yavaşlayarak
    # ulaşılabilen bir yerde bulunuyordu.
    #
    # `inside_safe_region` hâlâ payı sıfırlar: orada hold hız sınırı geçerli.
    approaching = state in APPROACH_STATES
    still = state in STILL_STATES
    allowance = 0.0 if inside_safe_region else max(0.0, float(speed_allowance_m_s))
    excess = max(0.0, float(speed_m_s) - allowance)
    settle = (
        -config.settle_speed_weight * float(np.square(excess))
        if inside_safe_region or approaching
        else 0.0
    )
    # REV-046: ADR-012 başarısı BEŞ hover koşulunun kesişimini ister fakat
    # ödül yalnız ikisini (konum ve hız) şekillendiriyordu. İrtifa, tilt ve
    # açısal hızın hiç gradyanı yoktu; ölçülen sonuç, beş koşulun birlikte
    # sağlandığı tek bir adım bile olmamasıydı (5/5 = %0.0, en uzun dizi 0
    # adım) ve bu hold süresi 3.0 s'den 0.5 s'ye indirildiğinde de değişmedi.
    #
    # Üç bileşen de `settle_speed` ile aynı biçimdedir: yalnız terminal
    # durumlarda, sınırı AŞAN fazlalığın karesi. Sınır içinde ceza yoktur,
    # yani başarı ölçütü gevşetilmez veya sıkılaştırılmaz; yalnız ölçütün
    # kendisi öğretilir.
    settle_altitude = (
        -config.settle_altitude_weight
        * float(
            np.square(max(0.0, float(altitude_error_m) - float(altitude_allowance_m)))
        )
        if approaching
        else 0.0
    )
    settle_tilt = (
        -config.settle_tilt_weight
        * float(np.square(max(0.0, float(tilt_rad) - float(maximum_tilt_rad))))
        if still
        else 0.0
    )
    settle_angular = (
        -config.settle_angular_rate_weight
        * float(
            np.square(
                max(
                    0.0,
                    float(angular_rate_rad_s) - float(maximum_angular_rate_rad_s),
                )
            )
        )
        if still
        else 0.0
    )
    hold = config.hover_hold_weight if state is HoverState.HOLD else 0.0
    proximity_ratio = float(
        np.clip(
            (config.proximity_threshold_m - minimum_range_m)
            / config.proximity_threshold_m,
            0.0,
            1.0,
        )
    )
    return TerminalHoverRewardBreakdown(
        target_progress=float(progress),
        progress_refund=float(refund),
        cruise_altitude=float(cruise),
        bearing_alignment=float(bearing),
        target_visible=float(visible),
        target_alignment=float(alignment),
        settle_speed=float(settle),
        settle_altitude=float(settle_altitude),
        settle_tilt=float(settle_tilt),
        settle_angular_rate=float(settle_angular),
        hover_hold=float(hold),
        proximity=-config.proximity_weight * float(proximity_ratio**2),
        smoothness=-config.smoothness_weight
        * float(np.sum(np.square(action - prior_action))),
        collision=-config.collision_penalty if collision else 0.0,
        out_of_bounds=-config.out_of_bounds_penalty if out_of_bounds else 0.0,
        success=config.success_bonus if success else 0.0,
        abort=-config.abort_penalty if aborted else 0.0,
        time=-config.time_penalty,
    )


__all__ = [
    "ALIGNMENT_STATES",
    "TerminalHoverRewardBreakdown",
    "TerminalHoverRewardConfig",
    "compute_terminal_hover_reward",
]
