# Faz 6: Engelsiz PPO Baseline

**Durum:** Tek-seed 100k pilot başarılı; donanım kararı sonrası legacy baseline olarak donduruldu  
**Başlangıç:** 10 Eylül 2026

## Amaç

`EbabilNavigation-v0` ortamında engelsiz uçuş için tekrarlanabilir bir PPO
eğitim, checkpoint, deterministic evaluation ve viewer replay hattı kurmak.
Bu faz engelden kaçınmayı öğretmez; Faz 7 curriculum'u için uçuş ve hedefe
ulaşma baseline'ını oluşturur.

> **11 Eylül 2026 donanım kararı:** Hedef araçta LiDAR/rangefinder yoktur.
> Observation v1 değişmeden regression baseline'ı olarak korunur. Kalan iki
> seed ve full kalite koşusu ana geliştirme yolundan çıkarılmış; kamera tabanlı
> `EbabilVisionNavigation-v0` Faz 6.5'e alınmıştır. Bkz. ADR-011.

## Sabitlenen ilk kapsam

- Eğitim varsayılanı: `scene_mode=clear`, `sensor_mode=ideal`.
- Algoritma: Stable-Baselines3 PPO ve `MlpPolicy`.
- Üretim vektörleştirmesi: `SubprocVecEnv`; smoke/test alternatifi:
  `DummyVecEnv`.
- Observation v1 zaten `[-1, 1]` aralığına ölçeklendiği için observation
  normalization varsayılan olarak kapalıdır.
- Reward normalization yapılandırılabilir ve varsayılan olarak açıktır.
- Eğitim headless; seçilmiş deterministic episode'lar sayısal replay olarak
  kaydedilip ayrı MuJoCo viewer scriptiyle oynatılır.
- Her koşu config, seed, paket sürümleri, checkpoint, best model ve logları tek
  bir koşu dizininde saklar.

## İş dilimleri ve doğrulama kapıları

| İş dilimi | Durum | Doğrulama |
|---|---|---|
| Tipli PPO TOML config'i | Tamamlandı | `4` hedefli test + Ruff + mypy |
| Vektör environment fabrikası | Tamamlandı | Shape, bağımsız seed ve kapanış testi |
| Headless eğitim ve resume | Tamamlandı | CLI 64 adım + 64 adım resume |
| Checkpoint, best model ve loglar | Tamamlandı | ZIP, VecNormalize, CSV, TensorBoard |
| Deterministic evaluation ve metrikler | Tamamlandı | Aynı seed ile byte-eşit iki koşu |
| Episode replay kaydı | Tamamlandı | NPZ/JSON, 20 Hz policy + 500 Hz physics |
| Viewer replay | Tamamlandı | Gerçek kayıtta `--dry-run` başarılı |
| Tam regresyon | Tamamlandı | 83 test, Ruff, mypy, compile, pip check |

## Kabul ölçütleri

Altyapı kabulü ile politika kalitesi birbirinden ayrılır:

1. Bu implementasyon turunun kapısı kısa smoke eğitiminin tamamlanması,
   checkpoint'in tekrar yüklenmesi, deterministic evaluation ve replay
   üretimidir.
2. Bilimsel baseline kapısı en az üç tam eğitim seed'i, held-out episode'larda
   en az `%95` başarı ve random/doğrudan-hedef baseline karşılaştırmasıdır.
3. Kısa smoke koşusunun başarı oranı model kalitesi iddiası olarak
   raporlanmayacaktır.

## Bilinen kapsam dışı konular

- Statik engel curriculum'u ve avoidance başarısı Faz 7 kapsamındadır.
- Gerçek uçuş güvenliği, yalnızca simülasyon PPO başarısıyla sağlanmış sayılmaz.
- Viewer eğitime bağlanmaz ve GUI testi otomatik kalite kapısına girmez.

## İş dilimi 1 doğrulama notu

İlk testte SB3 `Monitor` sınıfının çalışma anında `Path` yerine `str` beklediği
görüldü. Fabrika monitor yolunu açıkça string'e çevirir hale getirildi. Ardından
`4` hedefli test, Ruff lint/format ve mypy başarıyla tamamlandı.

## İş dilimi 2 — Eğitim ve artifact hattı

- `train_ppo.py` config override, smoke, checkpoint, best model ve yeni dizine
  resume desteğiyle eklendi.
- 64 adımlık CLI smoke koşusu final/best model, iki periyodik checkpoint,
  evaluation, CSV ve normalization kaydı üretti.
- Aynı final model + normalization kaydı yeni koşuda 64 adımdan 128 toplam
  adıma başarıyla devam etti.
- TensorBoard 2.21.0 `.venv` içine kuruldu; event-file üretimi otomatik testte
  doğrulandı.

## İş dilimi 3 — Evaluation ve replay

- Aynı checkpoint/config/seed ile iki evaluation koşusunun summary ve replay
  dosyaları byte düzeyinde aynı çıktı.
- Replay yükleme pickle'a izin vermez; shape, sonluluk ve zaman monotonluğu
  doğrulanır.
- İlk uygulamadaki 20 Hz state kaydı ADR-005 ile çeliştiği için düzeltilmiştir.
  Güncel kayıt 20 Hz policy verisine ek olarak 500 Hz fizik state'i, rotor
  thrust ve doygunluk izlerini taşır.
- Gerçek PPO smoke episode'u `40` policy adımı, `1000` fizik adımı ve başlangıç
  dahil `1001` viewer frame'iyle doğrulandı.
- `view_policy_replay.py --dry-run` sahneyi aynı seed ile yeniden kurdu ve state
  boyutlarını MuJoCo modeliyle doğruladı.

## Kullanım

Altyapı smoke testi:

```bash
.venv/bin/python scripts/train_ppo.py --smoke --device cpu \
  --run-name ppo-smoke
```

Tam baseline eğitimi:

```bash
.venv/bin/python scripts/train_ppo.py --run-name ppo-clear-seed-20260910
```

Deterministic evaluation:

```bash
.venv/bin/python scripts/evaluate_ppo.py \
  runs/ppo-clear-seed-20260910/final_model.zip \
  --output-dir runs/ppo-clear-seed-20260910/held-out \
  --episodes 20 --record-replays 3
```

Viewer replay:

```bash
scripts/run_mjpython.sh scripts/view_policy_replay.py \
  runs/ppo-clear-seed-20260910/held-out/replays/episode_000.npz --loop
```

TensorBoard:

```bash
.venv/bin/tensorboard --logdir runs
```

## Final doğrulama

- `83` test ve `10` subtest başarılı.
- Ruff lint ve format başarılı.
- Mypy `36` üretim dosyasında hatasız.
- Compile ve `pip check` başarılı.
- İki subprocess environment ile gerçek 64 timestep PPO smoke eğitimi
  checkpoint, best-model, best/final normalization, CSV ve TensorBoard
  artifact'lerini üretti.
- Deterministic evaluation, güvenli replay round-trip ve gerçek replay viewer
  dry-run başarılı.
- İzole wheel kurulumundan PPO config'i, MJCF ve environment reset erişildi.
- Mevcut clear scripted baseline regresyonu `118` adımda `goal` ile tamamlandı;
  final mesafe `0.072982 m` ve 10-adımlık hold sağlandı.

## İş dilimi 4 — Kalite kapısı hazırlığı

- Hedef başarısı `0.50 s / 10 policy adımı` kesintisiz yarıçap+hız koşuluna
  bağlandı; sayaç bozulunca sıfırlanır.
- `fixed`, `random_start`, `random_goal`, `random_both` endpoint modları scene
  seed'ine bağlı ve güvenli sınırlar içinde eklendi.
- `phase6_protocol.toml` üç training seed'i, ayrık yirmi held-out seed'i, pilot
  ve full timestep bütçelerini ve dört evaluation senaryosunu sabitler.
- Random ve privileged direct-goal baseline'ları aynı held-out seed'lerde ortak
  metrik sözleşmesiyle değerlendirilir.
- `run_phase6_experiment.py` baseline, PPO train, best checkpoint evaluation,
  replay ve aggregate kalite kapısını tek komutta yürütür.
- Runtime eğitim hataları hem training hem experiment metadata'sında `failed`
  olarak kayıt altına alınır.

Doğrulama sonucu: `83` test + `10` subtest, Ruff, mypy (`36` kaynak), compile
ve `pip check` başarılıdır. Uçtan uca experiment CLI smoke koşusu bir training
seed'i ve iki held-out episode ile tamamlandı; kısa politika beklendiği gibi
kalite kapısını geçmedi.

Tek-seed 100k pilot:

```bash
.venv/bin/python scripts/run_phase6_experiment.py \
  --stage pilot --training-seed-count 1 --device cpu \
  --output-dir runs/phase6-pilot-seed-20260910
```

Üç-seed full deney ancak pilot incelemesinden sonra çalıştırılmalıdır:

```bash
.venv/bin/python scripts/run_phase6_experiment.py \
  --stage full --device cpu --output-dir runs/phase6-full-v1
```

## Tek-seed pilot sonucu — 11 Eylül 2026

`20260910` training seed'i istenen `100k`, gerçek `102.4k` timestep sonunda
held-out `20/20` başarıya ulaştı. PPO ortalama return/adım/yol değerleri sırasıyla
`38.6791 / 61.0 / 4.1447 m`; direct-goal için
`38.4351 / 118.0 / 4.2850 m` oldu. Random baseline başarı sağlayamadı.

Pilot bütün performans kontrollerini geçti. Final kalite kapısı yalnız tek seed
ve pilot aşaması olduğu için bilinçli olarak geçmedi. Ayrıntılı artifact:
`runs/phase6-pilot-seed-20260910/summary.json`.

Viewer:

```bash
scripts/run_mjpython.sh scripts/view_policy_replay.py \
  runs/phase6-pilot-seed-20260910/ppo/seed_20260910/held_out/replays/episode_000.npz
```
