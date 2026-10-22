# ADR-015: Vision Navigation ve Observation Sözleşmesi v1

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-008 legacy environment'ı dondurur; ADR-011 actor donanım
sınırını, ADR-013 segment bütçesini ve ADR-014 encoder approval zincirini
tanımlar.

## Bağlam

`EbabilNavigation-v0` actor observation'ındaki rangefinder ve doğrudan hedef
konumu gerçek IMX477 + Pixhawk 2.4.8 + Jetson Orin NX araçta aynı anlamla
üretilemez. Yeni ortamın kamera, FC state, GPS-denied yerel waypoint ve sonlu
ufuk bilgisini tek, test edilebilir ve ayrıcalıklı kanal sızıntısına kapalı bir
sözleşmeyle birleştirmesi gerekir.

## Karar

Yeni ortam ID'si `EbabilVisionNavigation-v0`, observation sürümü `vision-v1`
olacaktır. Legacy ortam ve checkpoint sözleşmesi değiştirilmez.

### Encoder kaynağı

- Runtime encoder yalnız ADR-014 uyumlu `approval_manifest.json` üzerinden
  `load_approved_vision_encoder` ile yüklenir. Checkpoint yolu doğrudan kabul
  edilmez.
- SHA-256/dataset/quality/audit zinciri yüklemede fail-fast doğrulanır.
- Encoder `eval` modunda ve bütün parametreleri `requires_grad=False` olarak
  çalışır. PPO ilk sürümde encoder ağırlıklarını güncellemez.
- RGB yalnız `ActorCameraFrame` ve `preprocess_actor_rgb` yolundan encoder'a
  girer; observation içinde ham görüntü tutulmaz.

### Action space

```text
Box(low=-1, high=1, shape=(3,), dtype=float32)
```

> **SÜRÜM NOTU.** Aşağıdaki action tablosu `vision-action-v1`'dir ve
> yürürlükte değildir; güncel sözleşme ADR-020'dedir (dört boyut, işaretli,
> izotropik yatay hız sınırı).

| İndeks | Anlam | Dönüşüm |
|---:|---|---|
| 0 | İleri hız | `max(0, a[0]) * 1.5 m/s` |
| 1 | Dünya dikey hızı | `a[1] * 1.0 m/s` |
| 2 | Yaw-rate | `a[2] * 1.5 rad/s` |

Negatif ileri komut hover/yerinde dönüş anlamına gelir; geri ve yanal hız
komutu yoktur. Bu, tek kameranın kör yan/arka alanına doğru translasyonu
başlangıç sözleşmesinde kapatır. Action yanlış shape, NaN veya Inf ise reddedilir;
aralık dışı sonlu değerler kırpılıp `info["action_clipped"]` ile raporlanır.

### Observation space `vision-v1`

> **SÜRÜM NOTU (13 Eylül 2026).** Aşağıdaki tablo `vision-v1`'i, yani bu ADR'nin
> özgün sözleşmesini tarif eder ve **artık yürürlükte değildir.** Güncel
> sözleşme **ADR-021 `vision-v2`**'dir: `obstacle_map` `(108,)` kanalı eklendi
> ve `vision_aux` `(9,) -> (6,)` küçüldü (`collision_risk` çıkarıldı, REV-038).
> Action sözleşmesi de **ADR-020 `vision-action-v2`** ile `(3,) -> (4,)` oldu.
> Bu bölüm tarihsel kayıt olarak korunur.

Observation bir `spaces.Dict` olur. Bütün değerler `float32` ve kendi space
sınırları içindedir:

| Anahtar | Shape/aralık | İçerik |
|---|---|---|
| `vision_latent` | `(64,)`, `[-1, 1]` | Encoder latent'i, config ölçeğiyle kırpılmış |
| `vision_aux` | `(9,)`, `[-1, 1]` | target score, target center xy, free-space L/C/R ve risk L/C/R |
| `fc_state` | `(10,)`, `[-1, 1]` | body hız xyz, body açısal hız xyz, body gravity yönü xyz, estimator altitude |
| `waypoint` | `(4,)`, `[-1, 1]` | body-frame birim yön xyz ve `log1p` normalize mesafe |
| `timing` | `(2,)`, `[0, 1]` | kalan segment bütçesi oranı ve normalize kamera frame yaşı |

Target score ve üç risk skoru sigmoid sonrası `[0, 1]` değerlerinden `[-1, 1]`
aralığına taşınır. Target center zaten normalize görüntü koordinatıdır;
free-space de aynı biçimde `[-1, 1]` aralığına taşınır.

İlk simülasyonda FC/waypoint adapter'ı MuJoCo ideal state'inden bu deploy
edilebilir semantiği üretir. Bu ideal-estimator baseline'ıdır; gerçek Pixhawk/VIO
doğruluğu iddiası değildir. Actor API'si dünya konumu veya hedef dünya
koordinatı yayımlamaz. Gerçek sistemde aynı alanları Pixhawk estimator ve
GPS-denied VIO/global rota katmanı sağlamalıdır.

Depth, segmentation, raycast/rangefinder, obstacle dünya koordinatları, tam
MuJoCo `qpos/qvel`, hedef dünya koordinatı ve privileged label hiçbir observation
anahtarında bulunamaz. Rangefinder yalnız reward/termination/evaluation için
privileged minimum açıklık hesabında kullanılabilir.

### Kamera-policy zamanlaması

- Fizik ve alt kontrol: `500 Hz`
- Policy: `20 Hz`, action başına `25` fizik adımı
- IMX477 simülasyon capture/inference: `30 Hz`

Reset anında ilk frame alınır. Sonraki frame'ler fizik saati üzerinde
`n / 30 s` deadline'larını geçen ilk fizik tick'inde alınır. Encoder yalnız yeni
frame geldiğinde çalışır; aradaki policy observation'ları son özelliği
sample-and-hold kullanır. Gerçek capture timestamp'i, frame sayısı ve yaş
`info` içinde; normalize yaş observation `timing` içinde bulunur. Scheduler
wall-clock değil simülasyon zamanına bağlı ve seed'den bağımsız deterministiktir.

### Dinamik segment horizon'ı

ADR-013 formülü aynen uygulanır:

```text
T = clamp(T_fixed + k_time * L_plan / max(v_nominal, v_floor)
          + T_terminal, T_min, T_max)
N = ceil(T * policy_frequency_hz)
```

Başlangıç değerleri `v_nominal=1.5 m/s`, `v_floor=0.25 m/s`, `k_time=1.75`,
`T_fixed=5 s`, `T_min=20 s`, `T_max=60 s` olur. Nihai segmentte
`T_terminal=10 s yaklaşma + 3 s hover hold`, ara segmentte sıfırdır.
`reset(options)` içindeki `planned_route_length_m`, düz hat mesafesinden kısa
olamaz; verilmezse ilk iskelette düz hat mesafesi kullanılır ve bu fallback info
içinde belirtilir. Zaman aşımı yalnız `truncated=True` ve
`termination_reason="time_limit"` üretir.

### Seed ve hata semantiği

Aynı episode/component seed'leri, approval manifest'i, config ve action dizisi;
aynı sahne, capture deadline'ları ve observation dizisini üretmelidir. Kamera
veya encoder sonlu olmayan/yanlış şekilli veri üretirse ortam sessizce kırpmak
yerine hata verir. Fiziksel failure ve hedef başarısı `terminated`, bütçe sonu
`truncated` semantiğini korur.

## Doğrulama kapısı

1. Observation anahtar/shape/dtype/sınır ve privileged leakage testleri.
2. Encoder approval-only yükleme ve frozen parametre testi.
3. `30/20/500 Hz` scheduler deadline, frame-age ve sample-and-hold testi.
4. Kısa/uzun/final segment bütçesi, monotonluk, clamp ve ceil testleri.
5. Aynı seed/action rollout eşitliği.
6. Gymnasium `check_env`, uzun random-policy smoke, kısa PPO smoke ve ayrı
   deterministic evaluation.

## Sonuçlar ve sınırlar

- Bu sürüm terminal hover state machine'ini içermez; ADR-012 uygulaması sonraki
  iş paketidir.
- Simülasyon ideal FC/VIO adapter'ı localization drift veya Pixhawk doğruluğunu
  modellemez.
- Onaylı sentetik encoder, gerçek IMX477/Jetson parity ve gecikme kanıtı değildir.
- Privileged proximity reward güvenlik garantisi değildir ve deploy inference
  girdisi değildir.
