# Sıfırdan Kurulum ve İlk Çalıştırma

*English version: [GETTING_STARTED.md](GETTING_STARTED.md)*

Bu belge, depoyu yeni klonlamış birinin projeyi kendi bilgisayarında
çalıştırana kadar izleyeceği adımları anlatır. Adımlar 22 Ekim 2026'da temiz
bir sanal ortamda baştan sona denenmiştir; her adımın beklenen çıktısı
yanında yazılıdır.

Projenin ne yaptığı, kararların gerekçesi ve güncel durum için
[README](../README.md), [ilerleme günlüğü](PROGRESS_LOG.md) ve
[ADR'ler](decisions/) esas kaynaktır. Bu belge yalnız "nasıl çalıştırırım"
sorusuna cevap verir.

## 1. Gereksinimler

| Gereksinim | Not |
|---|---|
| İşletim sistemi | Geliştirme ve bütün ölçümler macOS 15 (Apple Silicon M3) üzerinde yapıldı. Linux'ta çalışması beklenir fakat ölçülmedi; farklar §8'de. |
| Python | **3.13.x** zorunludur (`requires-python >= 3.13, < 3.14`). python.org kurulumu önerilir; `python3.13 --version` ile doğrulayın. |
| Git | Depoyu klonlamak için. Kodun kendisi Git gerektirmez. |
| ffmpeg | Yalnız demo videosu üretmek için (`brew install ffmpeg`). |
| Disk | Sanal ortam ~1.3 GB. Eğitim çıktıları (`runs/`) ayrıca: görsel dataset 1.9 GB, YOLO dataset 0.9 GB, politika koşusu başına ~20 MB. |
| İnternet | Yalnız `pip` kurulumu ve bir kerelik YOLO11n ön-eğitimli ağırlık indirmesi için. Eğitim ve değerlendirme çevrimdışı çalışır. |

GPU gerekmez. Dedektör ve politika eğitimi Apple MPS üzerinde ölçüldü; CPU
ile de çalışır, yalnız daha yavaştır.

## 2. Depoyu klonlama

```bash
git clone https://github.com/YakupTekeli/Drone-Engagement-via-Reinforcement-Learning-in-a-MuJoCo-Environment-.git ebabil-mujoco
cd ebabil-mujoco
```

Klasör yolunda boşluk olmaması işleri kolaylaştırır. Boşluk varsa da
çalışır; yalnız macOS viewer komutları `scripts/run_mjpython.sh` sarmalayıcısı
üzerinden verilir (§6).

## 3. Sanal ortam ve bağımlılıklar

Proje yalnız kendi kökündeki `.venv` ile çalışır. Sistem Python'una veya
`sudo pip` ile global paket kurulumu yapılmaz.

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install --no-build-isolation -e .
```

`requirements.lock` çekirdek bağımlılıkları (MuJoCo 3.13, Gymnasium 1.3,
Stable-Baselines3 2.9, PyTorch 2.14), dedektör zincirini (Ultralytics 8.4,
Pillow, OpenCV) ve geliştirme araçlarını (pytest, ruff, mypy) birlikte
sabitler; ayrıca bir "extras" kurulumu gerekmez. Son satır projeyi
düzenlenebilir (editable) kurar; `setuptools` lock dosyasında olduğu için
`--no-build-isolation` güvenlidir.

Kurulumu tek satırla doğrulayın:

```bash
.venv/bin/python -c "import ebabil, mujoco, torch; print('ok', mujoco.__version__, torch.__version__)"
# ok 3.13.0 2.14.0
```

Komutları her zaman `.venv/bin/python ...` biçiminde vermek en güvenlisidir;
`source .venv/bin/activate` sonrası kısa `python` da kullanılabilir.

## 4. Test ve kalite kapıları

Dört kanonik komut vardır; dördü de temiz kurulumda geçmelidir:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy
```

Beklenen sonuç: `382 passed, 117 subtests passed`, ruff'ta "All checks
passed!" ve "125 files already formatted", mypy'da "Success: no issues found
in 61 source files". Test paketi headless çalışır, grafik bağlamı veya eğitim
çıktısı gerektirmez; M3 üzerinde yaklaşık bir dakika sürer.

## 5. İlk çalıştırma (eğitim çıktısı gerektirmez)

Aşağıdaki komutlar yalnız depodaki kodla çalışır ve saniyeler sürer.

```bash
# Quadrotor modelini yazdır (kütle, atalet, rotor konumları)
.venv/bin/python scripts/inspect_model.py

# Klasik kademeli kontrolcüyle headless hover; itki ve tork özetini basar
.venv/bin/python scripts/run_hover_demo.py

# Legacy navigasyon ortamında tek episode (beklenen: termination_reason=goal)
.venv/bin/python scripts/smoke_navigation_env.py --mode clear

# PPO altyapısının kısa smoke eğitimi (~3 s, kalite iddiası taşımaz)
.venv/bin/python scripts/train_ppo.py --smoke --device cpu --run-name ppo-smoke
```

Son komut `runs/ppo-smoke/` altında `final_model.zip`, `vecnormalize.pkl`,
`checkpoints/`, `evaluation/`, `logs/` ve `metadata.json` üretir. Bu dizin
düzeni bütün eğitim koşularında aynıdır.

## 6. Görselleştirme (MuJoCo viewer)

Eğitim ve toplu değerlendirme her zaman headless'tır; viewer yalnız seçilmiş
kayıtları incelemek içindir. macOS'ta MuJoCo viewer `mjpython` ister ve proje
bunu sarmalayıcıyla başlatır:

```bash
scripts/run_mjpython.sh scripts/view_hover_demo.py --loop
scripts/run_mjpython.sh scripts/view_static_scene.py --mode single_blocker --seed 20260910
scripts/run_mjpython.sh scripts/view_target_camera.py --pose hover --camera onboard
```

Bir politika replay'i izlemek için önce `record_vision_replay.py` ile `.npz`
kaydı üretilir (§7.3), sonra:

```bash
scripts/run_mjpython.sh scripts/view_policy_replay.py runs/<kayıt>.npz --loop
```

## 7. Eğitim çıktıları: depoda olanlar ve olmayanlar

`runs/` altındaki eğitim çıktıları genel kural olarak Git dışıdır (`runs/*`,
`*.pt`, `*.zip`). İstisna olarak **onaylı YOLO11n dedektör zinciri ve en iyi
PPO politikası depoya alınmıştır**; temiz bir klon demo ve değerlendirme için
gereken her şeyi içerir. Dataset'ler, ara checkpoint'ler ve diğer seed'lerin
koşuları depoda yoktur; bunlar §7.2 ile yeniden üretilir.

### 7.1 Yol A: depodaki eğitilmiş modelleri kullanmak

Klonla gelen model dosyaları (toplam ~7.8 MB):

| Dosya | Boyut | SHA-256 (ilk 16) |
|---|---:|---|
| `runs/detector_audit_v1_20260929/approval_manifest.json` | 2.7 KB | `5b9d92e05c088253` |
| `runs/detector_audit_v1_20260929/audit_report.json` | 4.2 KB | `ba85938821a83635` |
| `runs/detector_quality_v1_20260929/quality_report.json` | 6.7 KB | `51f9bcbd89d517d0` |
| `runs/detector_quality_v1_20260929/seed_20260911/weights/last.pt` | 5.4 MB | `8a9b537d863cee9a` |
| `runs/detector_cone_d1a_seed20260912_20260929/final_model.zip` | 2.2 MB | `a9a25ebbd12a337d` |

Yanlarında küçük kayıt dosyaları da gelir: dedektör eğitim ve kalite
metrikleri (`seed_20260911/*.json`, `audit_consumed.json`), politika
koşusunun `metadata.json` dosyası (config, seed, paket sürümleri),
`vecnormalize.pkl` (devam eğitimi için ödül normalizasyonu) ve nihai protokol
ölçümü `evaluation_heldout_final_det_assist_farloss_governor.json`.

Onay manifesti diğer üç dedektör dosyasını göreli yol ve SHA-256 ile bağlar;
dosyalardan biri taşınır veya değişirse yükleyici "Onay SHA-256 eşleşmiyor"
hatasıyla durur (fail-closed). Bu tasarım gereğidir. `.gitattributes` bu
dosyalarda satır sonu dönüşümünü kapatır; böylece Windows'ta da hash'ler
değişmez. Dosyaları elle düzenlemeyin ve `runs/` altındaki yerleşimi
bozmayın.

Aşağıdaki dört komut depodaki modelle, klon ve kurulumdan hemen sonra
çalışır (hepsi 22 Ekim 2026'da bu yollarla denendi). İlk üçü headless'tır.

**a) Ortam ve dedektör smoke'u** (check_env, rastgele rollout, kısa PPO,
kısa evaluation; ~20 s). Her şeyin yüklendiğini ve dedektörün hedefi
gördüğünü doğrular:

```bash
.venv/bin/python scripts/smoke_detection_vision_env.py \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --output-dir runs/detection_vision_smoke
```

**b) Politikayı değerlendirme.** Eğitimde hiç görülmeyen görsel alanda
(`validation`), rastgele engel ve başlangıç/hedef yerleşimiyle, deterministik
politika ve terminal yardımcı açık; 40 episode ~5 dk:

```bash
.venv/bin/python scripts/evaluate_vision_policy.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/eval_seed12_heldout.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --episodes 40 --max-steps 1200 --env-version v2 \
  --visual-domain-split validation --deterministic --terminal-assist on
```

Ekrana `success_rate`, `arrival_rate`, `collision_rate` ve bitiş sebepleri
basılır; ayrıntı JSON'a yazılır. Bu komutla yazarın ölçümü: başarı `0.875`,
varış `0.875`, çarpışma `0.050`. Hızlı kontrol için `--episodes 4` verin;
22 Ekim 2026'da sonuç `4/4 hover_success`, çarpışma `0` idi. `--terminal-assist
off` politikanın yardımcı olmadan ne yaptığını ölçer (hover başarısı `0`,
ADR-025).

**c) Bir episode'u MuJoCo viewer'da izleme.** Önce `500 Hz` replay kaydı
(~20 s), sonra viewer:

```bash
.venv/bin/python scripts/record_vision_replay.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/replays/seed12_ep310004.npz \
  --seed 310004 --env-version v2 --terminal-assist on \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --visual-domain-split development_ood
scripts/run_mjpython.sh scripts/view_policy_replay.py runs/replays/seed12_ep310004.npz --loop
```

`--seed 310004` + `development_ood`, README'deki demo episode'unun kendisidir
(389 adımda `hover_success`). Başka bir episode için seed'i değiştirin;
karanlık held-out alan için `--visual-domain-split validation`.

**d) Üç panelli demo videosu** (drone kamerası + YOLO kutuları, dış takip
kamerası, üstten harita; ffmpeg gerekir, ~30 s):

```bash
.venv/bin/python scripts/render_demo_video.py \
  runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  runs/detector_audit_v1_20260929/approval_manifest.json \
  runs/demo/bright_310004.mp4 \
  --episode-seed 310004 \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --visual-domain-split development_ood
```

Depodaki politikayı devam ettirmek (ek adım eğitmek) için §7.2 adım 7'deki
`--resume` notuna bakın; `vecnormalize.pkl` bunun için depodadır.

### 7.2 Yol B: her şeyi sıfırdan üretmek

Tam zincir aşağıdaki sırayla koşar. Süreler M3 (MPS) ölçümüdür.

**1. Görsel dataset (~7 dk, 1.9 GB).** 40 engelli D1 sahnesinde 6528 kare;
train / validation / development_ood / audit bölümleri seed'le ayrılır.

```bash
.venv/bin/python scripts/generate_vision_dataset.py runs/vision_dataset_d1 \
  --config src/ebabil/configs/training/vision_dataset_d1.toml \
  --scene-config src/ebabil/configs/environment/static_scene_d1.toml
```

**2. YOLO kutu etiketleri (~1 dk, 0.9 GB)** ve rastgele karelerde görsel
kontrol:

```bash
.venv/bin/python scripts/export_yolo_dataset.py runs/vision_dataset_d1 runs/yolo_dataset_d1
.venv/bin/python scripts/review_yolo_labels.py runs/yolo_dataset_d1 --count 8 --output-dir runs/yolo_dataset_d1/review
```

**3. YOLO11n ön-eğitimli ağırlık (bir kez, internet gerekir).** Eğitim COCO
ağırlığından başlar; dosya Ultralytics'in `v8.4.0` sürüm varlıklarından
indirilir ve SHA-256'sı `detector.toml` içindeki değerle eşleşmelidir:

```bash
mkdir -p runs/weights
curl -L -o runs/weights/yolo11n.pt \
  https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
shasum -a 256 runs/weights/yolo11n.pt
# 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1
```

Sonraki bütün Ultralytics çağrıları çevrimdışıdır (kod `YOLO_OFFLINE=1`
ayarlar); hesap veya giriş gerekmez.

**4. Dedektör kalite koşusu (seed başına ~76 dk; config üç seed ister).**

```bash
.venv/bin/python scripts/run_detector_quality.py runs/yolo_dataset_d1 runs/detector_quality
```

Bir seed yeterli görülürse koşu durdurulup tamamlanan seed'lerle kapatılır;
karar metni rapora yazılır:

```bash
.venv/bin/python scripts/finalize_detector_quality.py runs/yolo_dataset_d1 runs/detector_quality \
  --seeds 20260911 --decision "Tek seed ile devam kararı: ..."
```

**5. Audit ve onay.** Audit bölümü yalnız elipsoit engeller içerir ve bir kez
tüketilir. Bu dataset config'iyle audit'in engel kapılarını geçmemesi
beklenen durumdur (REV-050); o zaman dedektör, eğitimde görülen şekillerle
sınırlı olarak kullanıcı kararıyla onaylanır:

```bash
.venv/bin/python scripts/run_detector_audit.py runs/yolo_dataset_d1 \
  runs/detector_quality/quality_report.json runs/detector_audit
# audit geçmediyse:
.venv/bin/python scripts/approve_detector_by_decision.py \
  runs/detector_quality/quality_report.json runs/detector_audit/audit_report.json \
  --finding REV-050 --geometries box cylinder capsule sphere \
  --decision "Eğitim şekilleriyle sınırlı onay: ..."
```

Manifest `runs/detector_audit/approval_manifest.json` olarak yazılır. Eğitim
config'i `src/ebabil/configs/training/ppo_detector_d1a.toml` içindeki
`approval_manifest` alanı bu yola işaret etmelidir (varsayılan yol yazarın
koşu adıdır; kendi dizin adınızla güncelleyin).

**6. Ortam smoke'u** (§7.1'deki `smoke_detection_vision_env.py` komutu).

**7. Politika eğitimi (~3 saat / seed, 4 paralel ortam).**

```bash
.venv/bin/python scripts/train_ppo.py \
  --config src/ebabil/configs/training/ppo_detector_d1a.toml \
  --seed 20260912 --run-name detector_cone_d1a_seed20260912
```

İlerleme `runs/<run-name>/logs/progress.csv` ve TensorBoard'dadır
(`.venv/bin/tensorboard --logdir runs/<run-name>/logs`). Devam eğitimi
`--resume` ve `--resume-normalization` ile yapılır; `--total-timesteps` bu
durumda **ek** adım sayısıdır, toplam değil.

**8. Held-out değerlendirme (40 episode, ~5 dk).** Eğitimde hiç görülmeyen
görsel alanda (`validation`), deterministik politika ve terminal yardımcı
açık:

```bash
.venv/bin/python scripts/evaluate_vision_policy.py \
  runs/detector_cone_d1a_seed20260912/final_model.zip \
  runs/detector_audit/approval_manifest.json \
  runs/detector_cone_d1a_seed20260912/evaluation_heldout.json \
  --scene-config src/ebabil/configs/environment/static_scene_d1a.toml \
  --scene-mode random_obstacles --endpoint-mode random_both \
  --episodes 40 --max-steps 1200 --env-version v2 \
  --visual-domain-split validation --deterministic --terminal-assist on
```

Çıktı JSON'unda `success_rate`, `arrival_rate`, `collision_rate` ve episode
başına sonuçlar bulunur. Yazarın üç seed ölçümü (30 Eylül 2026):

| Seed | Hover başarısı | Hedefe varış | Çarpışma |
|---|---:|---:|---:|
| 20260912 | %87.5 | %87.5 | %5.0 |
| 20260913 | %77.5 | %77.5 | %22.5 |
| 20260914 | %87.5 | %87.5 | %12.5 |
| Ortalama | %84.2 | %84.2 | %13.3 |

Seed'ler arası yayılım büyüktür ve MPS üzerinde eğitim bit düzeyinde
tekrarlanabilir değildir; kendi koşunuzun bu aralığa düşmesi beklenir, aynı
sayıları vermesi beklenmez.

### 7.3 Kendi politikanızla replay ve demo

§7.1'deki c ve d komutlarında checkpoint yolunu kendi koşunuzun
`final_model.zip` dosyasıyla, manifest yolunu kendi `approval_manifest.json`
dosyanızla değiştirmek yeterlidir; diğer seçenekler aynıdır.

## 8. Diğer platformlar (Linux, CUDA, yalnız CPU)

Bunlar denenmemiştir; bilinen farklar şunlardır.

- **Cihaz seçimi.** Dedektör cihazı iki yerde `mps` olarak sabittir:
  `src/ebabil/configs/training/detector.toml` (`[detector] device`) ve
  `src/ebabil/configs/environment/vision_navigation_v4.toml`
  (`[detections] device`). İzin verilen değerler `mps`, `cpu`, `cuda`.
  Apple dışı makinede ikisini de `cpu` veya `cuda` yapın. PPO için
  `train_ppo.py --device auto|cpu|cuda` yeterlidir.
- **Headless render.** Linux'ta ekran yoksa `MUJOCO_GL=egl` (veya `osmesa`)
  ortam değişkeni gerekir.
- **Viewer.** `mjpython` yalnız macOS içindir; Linux'ta viewer script'leri
  doğrudan `.venv/bin/python` ile çalıştırılır.
- **Onaylı dedektör paketi** platformdan bağımsızdır; `.pt` ağırlığı ve
  `.zip` politikası aynı sürümlerle her yerde yüklenir.

## 9. Sık karşılaşılan sorunlar

| Belirti | Sebep ve çözüm |
|---|---|
| `python3.13: command not found` | Python 3.13 kurulu değil veya PATH'te değil. python.org kurulumu `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3.13` yoluna koyar. |
| Viewer `bad interpreter` hatası | Proje yolunda boşluk var; `.venv/bin/mjpython` yerine `scripts/run_mjpython.sh` kullanın. |
| `Onay SHA-256 eşleşmiyor: ...` | `runs/` altındaki dedektör dosyaları taşınmış veya değişmiş. §7.1'deki düzeni koruyun; dosyaları elle düzenlemeyin. |
| `Onay artefact kök dışına çıkıyor` | Quality ve audit dizinleri aynı `runs/` kökünün altında olmalı. |
| `detections.device mps/cpu/cuda olmalı` veya MPS bulunamadı | §8'deki iki config'te cihazı değiştirin. |
| Ultralytics internete çıkmaya çalışıyor | Olmamalı; kod `YOLO_OFFLINE=1` ve yerel `YOLO_CONFIG_DIR` ayarlar. Yalnız §7.2 adım 3'teki indirme internet ister. |
| Eğitim başlangıçta takılı kalıyor (%0 CPU) | Bir kez gözlendi (30 Eylül 2026); süreci öldürüp aynı komutu yeniden başlatmak yeterli oldu. |
| `--resume` ile eğitim beklenenden uzun sürüyor | `--total-timesteps` devam eğitiminde ek adımdır; `600k`'dan `1.2M`'ye çıkmak için `600000` verilir. |

## 10. Proje yapısı

```text
src/ebabil/
  assets/mjcf/        quadrotor.xml (MuJoCo modeli)
  configs/            vehicle, controller, environment, training, evaluation TOML'leri
  controllers/        kademeli klasik uçuş kontrolcüsü (ADR-004, ADR-011)
  dynamics/           rotor, mixer, sürükleme
  scenes/, sensors/   sahne üretimi, kamera, rangefinder (legacy)
  envs/               Gymnasium ortamları; detection_vision.py = EbabilVisionNavigation-v2
  perception/         dataset üretimi, YOLO export, dedektör eğitimi/kalite/audit, kutu yuvası kodlayıcı
  missions/           hover durum makinesi (ADR-017), terminal yardımcı (ADR-025)
  rewards/            terminal hover ödülü (18 bileşen)
  training/, evaluation/, experiments/   PPO koşucusu, değerlendirme, replay
scripts/              komut satırı araçları (bu belgedeki bütün komutlar)
tests/                382 test; headless
docs/                 tasarım, plan, ilerleme günlüğü, decisions/ (ADR), reports/, media/
runs/                 eğitim çıktıları; yalnız onaylı dedektör + politika depoda, gerisi Git dışı
```

## 11. Okuma sırası

1. [AGENTS.md](../AGENTS.md): çalışma kuralları (venv, testler, koordinat ve
   isimlendirme sözleşmesi, güvenlik).
2. [docs/PROGRESS_LOG.md](PROGRESS_LOG.md): "Güncel durum" bölümü ve en
   üstteki kayıtlar.
3. [docs/decisions/](decisions/): ADR-011 (mimari), ADR-017 (hover durum
   makinesi), ADR-023 (dedektör + politika), ADR-024 (hareket kısıtı),
   ADR-025 (terminal yardımcı).
4. [docs/reports/2026-09-30-rl-sistemi-raporu.docx](reports/2026-09-30-rl-sistemi-raporu.docx):
   observation, action ve ödül ayrıntıları;
   [sistem mimarisi diyagramı](reports/2026-10-01-sistem-mimarisi.png).

Simülasyondaki sonuçlar gerçek uçuş güvenliği iddiası değildir (AGENTS.md
§8).
