# Faz 6.5: Vision Navigation Environment

**Tarih:** 11 Eylül 2026  
**Durum:** Tamamlandı; environment, test ve gerçek renderer smoke kapıları geçti

## Amaç ve kapsam

Bu iş dilimi, bağımsız sentetik audit tarafından onaylanan
`vision-encoder-v9` modelini MuJoCo uçuş dinamiğine bağlayan ilk Gymnasium
ortamını kurar. Ortamın görevi yerel ego-frame waypoint takibi ve kamera tabanlı
engel kaçınma eğitimi için deploy edilebilir actor girdisi sağlamaktır.

Terminal hedef üstü hover state machine, curriculum eğitimi, uzun mission
segment zinciri ve gerçek Pixhawk/VIO adapter'ı bu dilimin kapsamı dışındadır.

## Environment sözleşmesi

- Environment ID: `EbabilVisionNavigation-v0`
- Observation sürümü: `vision-v1`
- Action: normalize `[forward_speed, vertical_speed, yaw_rate]`, shape `(3,)`
- Fizik/kontrol: `500 Hz`
- Policy: `20 Hz`
- Kamera ve frozen encoder inference: `30 Hz`
- Runtime model kaynağı: yalnız ADR-014 approval manifest'i

Actor observation bir `spaces.Dict` yapısıdır:

| Anahtar | Shape | Kaynak |
|---|---:|---|
| `vision_latent` | 64 | Dondurulmuş RGB encoder latent'i |
| `vision_aux` | 9 | Target score/center, free-space ve risk çıktıları |
| `fc_state` | 10 | Body hız/rate, gravity yönü ve estimator altitude semantiği |
| `waypoint` | 4 | Body-frame birim yön ve log-normalize mesafe |
| `timing` | 2 | Kalan bütçe oranı ve normalize frame yaşı |

Ham RGB encoder sınırında kalır. Depth, segmentation, raycast/rangefinder,
engel/hedef dünya koordinatları ve MuJoCo `qpos/qvel` actor observation'a
girmez. İdeal rangefinder yalnız reward için privileged minimum açıklık
hesabında kullanılmaya devam eder.

İlk simülasyon FC/waypoint adapter'ı ideal MuJoCo state'inden deploy edilebilir
semantiği üretir. Bu yapı gerçek Pixhawk 2.4.8 veya VIO hata karakteristiğini
doğrulamaz; gerçek adapter ve bozulma modeli sonraki iş paketidir.

## Kamera sample-and-hold

`SimulationDeadlineScheduler`, frame deadline'larını wall-clock yerine MuJoCo
zamanında `n / 30 s` olarak izler. Reset anında ilk frame alınır. `500 Hz`
fizik gridinde ilk iki sonraki capture yaklaşık `0.034 s` ve `0.068 s`
anlarındadır. Policy observation'ı capture olmayan aralıkta son encoder
özelliğini tutar.

Frame timestamp, frame sayısı, yaş ve stale durumu `info` içinde; normalize yaş
`timing` observation'ında loglanır. Aynı seed/action akışının frame deadline ve
observation dizisi otomatik testte eşittir.

## Dinamik segment horizon'ı

ADR-013 formülü `compute_segment_horizon` ile uygulanmıştır. Varsayılanlar:

| Örnek | Bütçe | Policy adımı |
|---|---:|---:|
| `4 m`, ara segment | `20 s` minimum clamp | `400` |
| `30 m`, ara segment | `40 s` | `800` |
| `30 m`, final segment | `53 s` | `1060` |

`planned_route_length_m` reset option'ı düz çizgi mesafesinden kısa ve yerel
`30 m` üst sınırından uzun olamaz. Verilmezse düz çizgi fallback'i kullanılır ve
kaynak `info` içinde belirtilir. Timeout yalnız `truncated=True` üretir.

## Oluşturulan/değiştirilen dosyalar

- `src/ebabil/envs/vision.py`
- `src/ebabil/envs/vision_config.py`
- `src/ebabil/configs/environment/vision_navigation.toml`
- `src/ebabil/envs/navigation.py` içindeki davranış değiştirmeyen alt sınıf hook'ları
- `src/ebabil/envs/__init__.py` environment kaydı ve public API
- `tests/test_vision_navigation_env.py`
- `scripts/smoke_vision_navigation_env.py`
- `docs/decisions/ADR-015-vision-navigation-contract-v1.md`

## Doğrulama

Hedefli doğrulama:

```bash
.venv/bin/python -m pytest -q \
  tests/test_navigation_env.py tests/test_camera_capture.py \
  tests/test_vision_navigation_env.py
# 31 passed, 1 skipped, 7 subtests passed
```

Gerçek approval manifest, MuJoCo RGB renderer ve approved encoder ile:

```bash
.venv/bin/python scripts/smoke_vision_navigation_env.py \
  runs/vision_audit_v2_20260911/approval_manifest.json \
  --mode random_obstacles --seed 20260911 \
  --random-steps 100 --ppo-timesteps 64 --evaluation-steps 80 \
  --output-dir runs/vision_env_smoke_v3_20260911
```

- Gymnasium checker + 100 random policy step: `2.117 s`
- 64-step `MultiInputPolicy` PPO + 80-step deterministic evaluation: `2.547 s`
- Toplam: `4.664 s`
- Random smoke: `100/100` sonlu step, bir fiziksel failure sonrası seed'li reset
- Deterministic evaluation: API/inference temiz; yalnız 64-step smoke model
  hedefe ulaşmadı ve 80-step test cap'ine geldi. Bu beklenen şekilde öğrenme
  kalitesi iddiası değildir.
- Artefact: `runs/vision_env_smoke_v3_20260911/{summary.json,smoke_model.zip}`

Tam kalite kapısı:

```text
pytest: 131 passed, 1 skipped, 12 subtests passed
ruff check: temiz
ruff format --check: 85 dosya biçimli
mypy: 46 source dosya temiz
compileall: temiz
pip check: bozuk bağımlılık yok
wheel build/install: temiz; paket içinden vision config/ID yüklendi
```

Skip, sandbox içindeki beklenen macOS offscreen capture testidir; aynı gerçek
capture production smoke'ta grafik erişimli headless süreçte geçmiştir.

## Operasyon notu ve bilinen sınırlar

Headless training/smoke `.venv/bin/python` ile çalıştırılır. Viewer replay
`scripts/run_mjpython.sh` kullanır. İlk smoke denemesinde headless script'in
`mjpython` ile başlatılması macOS'ta CPU kullanmadan kapanmayan süreç üretti;
hesap darboğazı değildir ve production smoke doğru Python girişiyle `4.664 s`
sürdü.

Kalan teknik sınırlar:

- Terminal hover state machine henüz yoktur; güncel başarı legacy nokta/hold
  semantiğidir.
- Rota uzunluğu verilmezse gerçek planner yolu yerine düz çizgi fallback'i vardır.
- Vision encoder frozen'dır; temporal history, jitter/drop ve uçtan uca
  fine-tuning henüz açılmamıştır.
- Gerçek IMX477/Orin latency-parity ve Pixhawk/VIO state adapter'ı bench kapısıdır.
- Sentetik başarı gerçek uçuş güvenliği değildir.

## Sonraki adım

ADR-012 uyarınca `transit -> acquired -> approach -> settle -> hold -> success`
terminal state machine'ini; target-loss/unsafe/stale abort semantiğini ve bütün
reward bileşenlerini ayrı loglayarak uygulamak.
