# Faz 6.5: Vision Dataset, Randomization ve Encoder

**Tarih:** 11 Eylül 2026  
**Durum:** Tamamlandı; üç-seed development quality ve bağımsız audit geçti

## Kapsam ve sınır

Bu dilim MuJoCo IMX477 RGB görüntülerini seed'li, sahne bazında ayrılmış bir
dataset'e dönüştürür. Privileged depth/segmentation yalnız supervision ve
değerlendirme etiketi üretir; actor depth, raycast, engel dünya koordinatı veya
tam MuJoCo state'i almaz.

Sonuç sentetik perception bileşenini vision environment'a kabul eder. Gerçek
IMX477 genellemesi, Jetson Orin NX gecikmesi, PPO başarımı veya gerçek uçuş
güvenliği kanıtı değildir.

## Dört-split dataset sözleşmesi

Kabul edilen şema `vision-dataset-v4` ve split salt'ı `ebabil-vision-v3`'tür.
Explicit sahne kotaları hash bucket ve gerçekleşen oranların tek kaynağıdır:

| Split | Sahne | Frame/sahne | Toplam | Geometri |
|---|---:|---:|---:|---|
| Train | 64 | 8 | 512 | box, cylinder, capsule, sphere |
| Validation | 8 | 8 | 64 | box, cylinder, capsule, sphere |
| Development OOD | 8 | 8 | 64 | box, cylinder, capsule, sphere |
| Audit | 12 | 8 | 96 | train-dışı ellipsoid |

Trainer ve quality runner yalnız ilk üç split'i kullanır. Audit split'i ancak
model, risk score eşiği, quality eşikleri ve candidate checkpoint dondurulduktan
sonra ayrı CLI ile açılır.

Dataset `metadata.json`, `statistics.json`, `manifest.jsonl`, `actor/` ve
`privileged/` ağaçlarından oluşur. Actor NPZ allowlist'i yalnız `sample_id`,
`rgb`, `timestamp_s` alanlarıdır. Validator RGB dtype/shape, ID/timestamp,
path traversal, yinelenen sample, scene-split sızıntısı ve actor/privileged
ayrımını fail-fast denetler.

## Semantik kısa yol kontrolleri

- Engel materyali train'de bütün RGB kanallarında `[0.05, 0.92]` aralığında
  seed'li örneklenir; validation/development/audit ayrı appearance domainleridir.
- Her sahne dört çarpışmasız ve privileged maskeye girmeyen distractor içerir.
  Bunların yarısı gerçek hedef alanıyla aynı renktedir; dolayısıyla yeşil piksel
  tek başına hedef etiketi değildir.
- Train/validation/development dört farklı primitive görür; audit yalnız
  görülmemiş ellipsoid kullanır. AABB rota kontrolü gerçek geometriyi dıştan
  güvenli biçimde sarar.
- Eğitim örneklerinin yüzde 50'sinde hue bilgisi seed'li grayscale augmentation
  ile kaldırılır.
- Nominal metriklere ek olarak grayscale free-space ve risk P/R/S aynı quality
  eşiklerini geçmek zorundadır.

Gerçek hedef görünümü kesinleşmediği için sentetik target metriği yalnız bu
simülasyon geliştirme sözleşmesinin parçasıdır.

## Supervision ve encoder

Privileged render'dan target visibility/center, target area fraction, üç bölgeli
free-space ve üç bölgeli collision-risk etiketi türetilir. Risk etiketi
`free_space < 0.25`; 8 m normalizasyon altında yaklaşık 2 m'dir. MuJoCo
`ARB_clip_control unavailable` uyarısı nedeniyle depth ayrı kalibrasyon
olmadan gerçek mesafe doğruluğu iddiası taşımaz.

`vision-encoder-v9`:

- kanonik HWC `uint8` -> CHW `float32 [0,1]` preprocessing,
- `96 x 72` RGB ve kanal-bazlı adaptive exposure,
- 12/24/32 kanallı üç convolution katmanı,
- `3 x 4` pooling ve 64 elemanlı latent,
- sabit `9 x 12` target/obstacle heatmap başları,
- soft-argmax target center, target visibility, free-space ve risk başları,
- config çözünürlüğünden bağımsız adaptif heatmap pooling,
- risk score eşiği `0.62`.

Checkpoint formatı `ebabil-vision-checkpoint-v9`, dosya adı
`vision_encoder_v9.pt`'dir. Checkpoint dataset provenance, model/training config
ve ağırlıkları içerir. Model yanlış dtype, `[0,1]` dışı veya sonlu olmayan
girdiyi reddeder.

## Quality, audit ve provenance

Her model seed'i validation ve development-OOD üzerinde hem nominal hem
grayscale engel kapılarını geçmelidir:

| Metrik | Eşik |
|---|---:|
| Target precision / recall | `>=0.78 / >=0.85` |
| Normalize center error | `<=0.30` |
| Free-space MAE | `<=0.30` |
| Trivial free-space baseline iyileşmesi | `>=0.15` |
| Risk precision / recall / specificity | `>=0.65 / >=0.70 / >=0.50` |
| Grayscale risk P/R/S | aynı eşikler |
| Grayscale free-space MAE | `<=0.30` |

Model seed'leri `20260911/12/13`; eğitim 45 epoch, batch 32 ve learning rate
`0.001` kullanır. Risk `0.62` ve P/R/S kapıları yeni salt'lı
validation/development sonuçlarından, ikinci audit açılmadan önce donduruldu.

Dataset config/metadata/manifest/statistics, candidate checkpoint, quality ve
audit raporları SHA-256 ile bağlanır. Yollar göreli yazılır. Quality yalnız
candidate üretir; `approval_manifest.json` sadece başarılı audit sonrası oluşur.
Atomik `audit_consumed.json`, aynı quality raporunun ikinci kez audit edilmesini
başarısızlık halinde de engeller. Runtime model yükleme
`load_approved_vision_encoder` ile tüm zinciri doğrular.

## Deney ve audit kaydı

- v2-v11 eski geliştirme döngüsü daha sonra `development_ood` adını alan eski
  test split'ini tekrar gözledi; bu sonuçlar bağımsız akademik test değildir.
- REV-008/009/011/012/013 ile fail-closed CLI, ortak preprocessing, adaptif
  heatmap, provenance ve dört-split sözleşmesi eklendi.
- v12 semantik renk/distractor ve box/cylinder train ile development quality
  risk/grayscale kapılarında başarısız oldu.
- v13 grayscale augmentation ekledi; nominal ve grayscale farkı azaldı fakat
  eski risk eşikleri geçmedi.
- v14 yeni development kapısını geçti. İlk bağımsız capsule/sphere audit risk
  precision `0.671/0.692` ile başarısız oldu ve approval üretmedi.
- İlk audit'e eşik uydurulmadı. Train zarfına capsule/sphere eklendi; salt
  değiştirildi ve yeni seed'li ellipsoid audit ayrıldı.
- v15 yeni development verisinde risk skoru kalibrasyonunu gerektirdi. Yalnız
  validation/development taramasıyla `0.62` ve P/R/S `0.65/0.70/0.50`
  donduruldu.
- v16 üç seed'in tüm development kapılarını geçti. Candidate seed `20260911`
  yeni ellipsoid audit'te tek koşuda onaylandı.

## Son sonuç

Development v16 ortalamaları:

| Split | Target P/R | Center error | Free-space MAE | Risk P/R/S |
|---|---:|---:|---:|---:|
| Validation | `0.845 / 0.980` | `0.184` | `0.218` | `0.851 / 0.728 / 0.810` |
| Development OOD | `0.918 / 0.929` | `0.226` | `0.257` | `0.709 / 0.813 / 0.562` |

Bağımsız audit (`96` kare):

| Koşul | Target P/R | Center error | Free-space MAE | Risk P/R/S |
|---|---:|---:|---:|---:|
| Nominal | `0.863 / 0.986` | `0.189` | `0.255` | `0.780 / 0.784 / 0.694` |
| Grayscale | `0.859 / 0.871` | `0.254` | `0.279` | `0.744 / 0.713 / 0.661` |

Onay manifesti:
`runs/vision_audit_v2_20260911/approval_manifest.json`. Approved checkpoint
SHA-256: `db5eca8ffa2a9a3bb03091bc67ff55a20fab9bbe8a5c5e9993c06ba3c55a848d`.

## Kullanım

```bash
.venv/bin/python scripts/generate_vision_dataset.py runs/vision_dataset_v4
.venv/bin/python scripts/run_vision_quality.py \
  runs/vision_dataset_v4 runs/vision_quality
.venv/bin/python scripts/run_vision_audit.py \
  runs/vision_dataset_v4 runs/vision_quality/quality_report.json runs/vision_audit
```

macOS dataset renderer'ı grafik bağlamı ister. Eğitim ve değerlendirme headless
çalışır. Çıktılar `runs/` altında Git dışındadır.

## Kalan riskler

- Lens distorsiyonu, rolling shutter, focus, gerçek motion blur ve kamera/IMU
  timing henüz tam modellenmiyor.
- Gerçek IMX477 kayıtları ayrı ve yalnız değerlendirmelik veri seti olmalıdır.
- ONNX parity ve Orin NX TensorRT capture-to-setpoint profiling donanım kapısıdır.
- Vision environment approved manifest dışında checkpoint kabul etmemeli;
  privileged kanallar actor observation'a girmemelidir.
