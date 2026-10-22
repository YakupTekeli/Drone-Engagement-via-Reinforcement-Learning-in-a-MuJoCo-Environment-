# Faz 6.5: Kamera ve Hedef Alan Temeli

**Tarih:** 11 Eylül 2026  
**Durum:** Kamera/target dilimi tamamlandı; dataset/encoder dilimi ayrı kayıtta

## Kapsam

Bu dilim, IMX477 ailesiyle eşleşen MuJoCo kamerasını ve dikdörtgen hedef alanı
legacy rangefinder ortamını değiştirmeden ekler. Henüz görsel encoder,
`EbabilVisionNavigation-v0`, PPO reward veya terminal state machine içermez.

## Uygulanan sözleşmeler

### Kamera

- Config: `src/ebabil/configs/environment/camera.toml`
- Tipli model: `ebabil.perception.CameraConfig`
- Simülasyon başlangıç profili:
  - sensör: `4056 x 3040` IMX477 ailesi,
  - actor render: `320 x 240 RGB`,
  - frame rate metadata: `30 Hz`,
  - yatay FOV: `75 derece`,
  - gövde montajı: `[0.10, 0.0, -0.01] m`,
  - aşağı pitch: `45 derece`.

MuJoCo kamera optik ekseni yerel `-z`, görüntü sağı `+x` ve görüntü yukarısı
`+y` kabul edilerek proje FLU gövde frame'ine dönüştürülür. Projection API'si
`data.cam_xpos/cam_xmat` üzerinden dünya noktalarını piksele taşır.

### Hedef alan

- Config: `src/ebabil/configs/environment/target_area.toml`
- Tipli model: `ebabil.scenes.TargetAreaConfig` ve `TargetArea`
- İlk simülasyon alanı: `2 x 2 m`
- Safety margin: `0.35 m`
- Güvenli iç bölge half-size: `0.65 x 0.65 m`
- Hover anchor: hedef yüzeyinden `2.0 m` yukarı

Hedef alan collision üretmeyen yeşil bir dikdörtgen, daha koyu güvenli iç bölge
ve beyaz merkez işareti olarak worldbody'ye eklenir. Bu geometri yalnız hedef
algısı ve başarı tanımı içindir; fiziksel iniş yüzeyi değildir.

## Actor ve privileged veri sınırı

`HeadlessCamera.capture_actor()` yalnız:

- `rgb: uint8[H,W,3]`
- `timestamp_s`

üretir. `capture_privileged()` ayrı bir veri tipiyle:

- metre cinsinden depth,
- raw MuJoCo segmentation,
- hedef binary maskesi,
- timestamp

üretir. Privileged veri teacher, reward ve evaluation içindir; actor payload'ına
giremez.

Legacy sahnedeki havada duran `target_marker_geom`, görsel politika için kolay
ama gerçek dışı bir hedef sızıntısıdır. `build_scene_model(...,
include_legacy_goal_marker=False)` vision modelinde zorunludur. `HeadlessCamera`
marker bulunan modeli reddederek bu sözleşmeyi fail-fast korur.

## Görünürlük sonucu

`scripts/sweep_camera_visibility.py`, düz gövde tutumunda `6, 4, 2, 0 m`
yaklaşma mesafelerinde hedef köşelerini projekte eder. İlk sweep sonucu:

- `40`, `45` ve `50 derece` pitch adayları her örnekte en az `%25` hedef köşesi
  görünürlüğünü korudu.
- Varsayılan `45 derece`, start ve doğrudan hedef üstü hover pozunda `%50`
  köşe görünürlüğü üretti.
- Hover'da hedef merkezi görüntü dışında kalabilir; hedefin öndeki yarısı hâlâ
  görünürdür. Bu durum encoder ve target-loss mantığında ayrıca ölçülecektir.

Bu sonuç yalnız pinhole geometri baseline'ıdır. Gerçek lens distorsiyonu,
montaj toleransı ve hedef boyutu ölçülmeden fiziksel montaj kararı değildir.

## Kullanım

Saf geometri sweep'i:

```bash
.venv/bin/python scripts/sweep_camera_visibility.py
```

Renderer açmadan smoke hazırlığı:

```bash
.venv/bin/python scripts/capture_camera_frame.py --pose start --dry-run
.venv/bin/python scripts/capture_camera_frame.py --pose hover --dry-run
```

Gerçek offscreen capture:

```bash
.venv/bin/python scripts/capture_camera_frame.py --pose hover
```

Viewer:

```bash
scripts/run_mjpython.sh scripts/view_target_camera.py --pose hover --camera overview
scripts/run_mjpython.sh scripts/view_target_camera.py --pose hover --camera onboard
```

## Doğrulama ve bilinen sınırlılıklar

- Config/geometri, projection, actor schema, target marker leakage ve viewer
  dry-run testleri otomatik çalışır.
- macOS sandbox içinde CoreGraphics bağlantısı kurulamadığından offscreen render
  testi `CameraRenderUnavailable` ile skip edilir; aynı test grafik erişimli
  süreçte ayrıca çalıştırılmalıdır.
- Grafik erişimli doğrulamada RGB `(240, 320, 3)`, depth `(240, 320)` üretildi;
  hedef maskesi start için `2532`, hover için `8974` piksel oldu.
- MuJoCo/CGL `ARB_clip_control unavailable` uyarısı verdi. Depth actor girdisi
  değildir; yine de privileged depth metriği kullanılmadan önce bilinen mesafe
  düzlemleriyle doğruluk kalibrasyonu yapılmalıdır.
- `StaticScene.goal_world_m` legacy nokta hedefidir. Vision hover anchor'ı
  `TargetArea` üzerinden türetilir; sonraki environment bu ikisini karıştırmamalıdır.
- Tam regresyon sonucu `96 passed, 1 skipped, 10 subtests passed`; sandbox'ta
  skip edilen offscreen test grafik erişimli süreçte ayrıca geçti. Ruff, format
  ve mypy temizdir.
- İzole wheel build/install smoke, kamera/hedef config dosyaları ile perception
  modülünün paket artefact'ına dahil olduğunu doğruladı.

## Devam kaydı

Seed'li dataset, actor/privileged ayrımı, görsel randomization ve hafif encoder
kalite kapısı `PHASE_6_5_VISION_DATASET_ENCODER.md` içinde uygulanmış ve üç
model seed'iyle doğrulanmıştır. Sıradaki kapı `EbabilVisionNavigation-v0` ve
deploy edilebilir `vision-v1` observation sözleşmesidir.
