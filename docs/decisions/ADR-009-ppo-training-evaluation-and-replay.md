# ADR-009: PPO Eğitim, Evaluation ve Replay Sözleşmesi

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

> **Donanım güncellemesi (11 Eylül 2026):** Bu eğitim sözleşmesi rangefinder
> içeren legacy `EbabilNavigation-v0` içindir. Tamamlanan tek-seed pilot altyapı
> baseline'ı olarak saklanır; kalan kalite koşuları kamera tabanlı ortama
> taşınmıştır. Bkz. ADR-011.

## Bağlam

Faz 5, PPO ile uyumlu headless Gymnasium ortamını sabitledi. Faz 6 için farklı
agentların aynı deneyleri tekrar edebilmesi; eğitim ve değerlendirme
dağılımlarını karıştırmaması; checkpoint ile normalization state'ini birlikte
yönetmesi ve episode'ları macOS MuJoCo viewer'da inceleyebilmesi gerekir.

## Karar

### Eğitim dağılımı

- İlk baseline `clear + ideal` dağılımında eğitilir. Bu politika engelden
  kaçınma iddiası taşımaz.
- PPO `MlpPolicy`, yüksek seviyeli dört boyutlu hız/yaw-rate action'ı üretir;
  klasik uçuş kontrolcüsü korunur.
- Üretim varsayılanı dört adet `SubprocVecEnv` ortamıdır. Test ve kısa smoke
  koşuları `DummyVecEnv` kullanabilir.
- Her vektör ortamı ortak run seed'inden farklı Gymnasium episode seed'i alır;
  environment bu seed'den bağımsız scene ve sensor seed'leri türetir.

### Normalization

- Observation v1 environment içinde `[-1,1]` aralığına ölçeklendiği için
  varsayılan observation normalization kapalıdır.
- Reward normalization varsayılan olarak açıktır. `VecNormalize` state'i
  checkpoint'lerle ve final modelle birlikte kaydedilir.
- Observation normalization ileride açılırsa evaluation aynı normalization
  dosyasını yüklemek zorundadır. Evaluation başarı ve return metrikleri ham
  environment reward'u üzerinden hesaplanır.

### Artifact ve log sözleşmesi

Bir eğitim koşusu şu yapıyı üretir:

```text
runs/<run-name>/
├── metadata.json
├── final_model.zip
├── vecnormalize.pkl
├── best/
│   ├── best_model.zip
│   └── vecnormalize.pkl
├── checkpoints/
├── evaluation/evaluations.npz
└── logs/
    ├── progress.csv
    ├── events.out.*
    ├── train_monitor/
    └── eval_monitor/
```

- Dolu bir koşu dizininin üzerine yazılmaz.
- Metadata tipli config'in tamamını, seed'i, runtime/paket sürümlerini, resume
  kaynaklarını ve tamamlanma durumunu içerir.
- Periyodik checkpoint frekansları global timestep olarak yapılandırılır ve
  SB3 callback çağrı frekansına `n_envs` dikkate alınarak çevrilir.
- CSV ve TensorBoard logları birlikte üretilir. TensorBoard 2.21.0 proje
  bağımlılığı olarak sabitlenmiştir.
- Resume yeni bir koşu dizinine yapılır. Normalization açıkken checkpoint ile
  eşleşen normalization kaydı zorunludur.

### Deterministic evaluation

- Evaluation ayrı seed dizisinde, `deterministic=True` ile ve headless çalışır.
- Summary; başarı/çarpışma oranı, return ortalama/standart sapması, episode
  uzunluğu, termination dağılımı ve episode bazında final mesafe, minimum
  clearance ve path length taşır.
- Aynı checkpoint, config ve seed dizisi aynı summary ve replay'i üretmelidir.
- Faz 6 protokolü üç training seed'i ve onlarla ayrık yirmi held-out episode
  seed'ini paketlenmiş `phase6_protocol.toml` içinde sabitler.
- Evaluation CLI varsayılan olarak bu manifestteki kesin seed listesini kullanır;
  `--seed` yalnız açık ad-hoc koşular içindir.

### Baseline ve deney matrisi

- `random` baseline her episode seed'iyle ayrı ve tekrar üretilebilir action RNG
  kullanır.
- `direct_goal` baseline privileged world-state kullanan, engel bilgisi olmayan
  bir P-controller'dır; PPO actor observation sözleşmesinin parçası değildir.
- Baseline'lar ve PPO aynı held-out episode seed'lerinde değerlendirilir.
- Faz 6 runner pilot veya full timestep sınırını, seçilen training seed'leri,
  tamamlanan seed'leri ve hataları `experiment.json` içinde izler.
- Aggregate kalite kapısı her PPO seed'inde başarı/çarpışma eşiklerini ve ortalama
  PPO return'ının hem random hem direct-goal baseline'dan yüksek olmasını ister.

### Replay

- Replay iki zaman ölçeğini birbirine karıştırmaz:
  - `time_s`, `qpos`, `qvel`, rotor thrust ve kontrol/doygunluk izleri her
    `500 Hz` fizik adımında tutulur.
  - action, reward ve reward bileşenleri `20 Hz` policy adımlarında tutulur.
- Sayısal diziler sıkıştırılmış NPZ, semantik metadata JSON olarak yazılır.
  Yükleme `allow_pickle=False` kullanır.
- Viewer sahneyi `scene_mode + scene_seed` ile yeniden kurar ve kaydedilmiş
  state'i simülasyonu tekrar çözmeden oynatır.

## Gerekçe

Bu ayrım eğitim throughput'unu GUI'den bağımsız tutarken politika kararlarını,
fizik tepkisini ve actuator doygunluklarını aynı episode üzerinde incelemeyi
sağlar. Ham evaluation reward'u, eğitimde reward normalization açık olsa bile
deneyler arası karşılaştırılabilir kalır.

## Sonuçlar ve sınırlılıklar

- Kısa smoke eğitim yalnız altyapı kanıtıdır; başarı oranı bilimsel sonuç
  değildir.
- Best model yalnız en az bir periyodik evaluation tetiklenmişse oluşur;
  metadata aksi durumda `null` taşır.
- State replay görüntülenebilir bir deney kaydıdır; sıkıştırılmış MP4 üretimi
  bu ilk baseline'ın zorunlu artifact'i değildir.
- Gerçek uçuş için safety filter, SITL/HITL ve fiziksel test prosedürü ayrıca
  gereklidir.
