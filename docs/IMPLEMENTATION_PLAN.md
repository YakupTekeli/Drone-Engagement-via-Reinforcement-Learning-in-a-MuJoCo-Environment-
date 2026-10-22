# Ebabil MuJoCo: Implementasyon Planı

**Plan tarihi:** 10 Eylül 2026  
**Son güncelleme:** 22 Eylül 2026  
**Yaklaşım:** Her faz bir doğrulama kapısıyla biter. Bir kapı geçilmeden sonraki karmaşıklık seviyesi ana geliştirme hattına eklenmez.

## 1. Hedef teslimat

İlk ana teslimat aşağıdaki özelliklere sahip olmalıdır:

- MuJoCo'da parametrik ve doğrulanmış quadrotor modeli
- Kararlı klasik hover/hız/attitude kontrol zinciri
- IMX477 monoküler kamera ve deploy edilebilir FC state girdileriyle Gymnasium
  uyumlu 3B hedefe ulaşma ve statik engelden kaçınma ortamı
- Hedef bölgenin güvenli iç alanı üzerinde kararlı hover ve kesintisiz hold
- PPO eğitim, checkpoint, evaluation ve video üretim akışı
- Başarı tabanlı curriculum
- Sensör ve dinamik domain randomization
- Tekrarlanabilir deney ve ablation altyapısı
- Gerçek uçuş öncesi kullanılabilecek güvenlik katmanı arayüzü

## Güncel revize yürütme planı — 11 Eylül 2026

Bu sıra, tamamlanmış legacy baseline'dan kamera tabanlı hedef-hover görevine
geçiş için güncel ana geliştirme yoludur. Güncel durumun tek kaynağı
`docs/PROGRESS_LOG.md` olmaya devam eder.

| Sıra | İş paketi | Durum | Çıkış kapısı |
|---:|---|---|---|
| 1 | Donanım, güvenli görev ve ölçek mimarisi kararları | Tamamlandı | ADR-010–ADR-013 kabul edildi |
| 2 | Parametrik IMX477/MuJoCo kamera ve hedef alan geometrisi | Tamamlandı | Seed'li headless RGB + privileged etiket doğrulandı |
| 3 | Hedef görünürlüğü ve kamera montaj açısı doğrulaması | Tamamlandı (sim); ADR-016 ile revize | Yaklaşma kapısı geçti; hold pozunda hedef merkezi kadraj dışı (nadir eşiği `60.08 derece`), fiziksel bench bekliyor |
| 4 | Hafif hedef/serbest-alan görsel encoder'ı | Tamamlandı; bağımsız sentetik audit geçti | ADR-014 approval zinciri ve review düzeltmeleri doğrulandı |
| 5 | `EbabilVisionNavigation-v0` + `vision-v1` | Tamamlandı | ADR-015, leakage/seed/horizon testleri ve gerçek renderer PPO smoke geçti |
| 6 | Hover geometrisi ve hedef açıklığı güvencesi | Tamamlandı | ADR-016; REV-014–REV-017 kapandı, legacy sahne dizisi korundu |
| 7 | Hover terminal state machine ve reward | Tamamlandı | ADR-017; geçiş/hold/abort/reward testleri ve gerçek renderer PPO smoke geçti |
| 8 | D0-V kamera tabanlı yerel politika | Tamamlandı (11 Eylül 2026, tek seed) | Held-out hover başarısı `20/20`; engel yok |
| 9 | D1 ölçek dilimi + yaklaşık hedef + engel ekseni | Koşuldu; kapı geçilmedi (`%0`, REV-039..046) | ADR-018 D1 kapısı; `20 x 20 m`, ADR-019 `R=2 m` |
| 9b | Etiketsiz görsel temsil (VAE) + `vision-v3` + ortam içi DR | Sıradaki iş | ADR-022 kapıları; §23 |
| 10 | D2-D4 mesafe merdiveni | ADR-018 ön koşullarını bekliyor | `vision-v2`, hız yükseltmesi, yerel pencere |
| 11 | D5 kompozisyon evaluation'ı | Bekliyor | Eğitilmeden ölçülür |
| 10 | Mission segment manager ve zincir bütçesi | Bekliyor | Waypoint geçişinde fizik sürekliliği ve horizon ablation'ı |
| 11 | D3-D5 mission-level evaluation | Bekliyor | GPS-denied localization kapısı ve birleşik mission metrikleri |
| 12 | Robustluk, Jetson profiling, SITL/HITL ve safety | Bekliyor | ADR-011 donanım kapıları ve güvenlik doğrulamaları |

Kod bağımlılığı sırası `kamera -> perception -> vision env -> hover -> yerel
kaçınma -> segmentli mission` olacaktır. Fiziksel kamera veya Pixhawk'ın henüz
bench'e hazır olmaması ilk dört simülasyon dilimini bloke etmez; donanım
kalibrasyonu config üzerinden sonradan uygulanır.

## 2. Planlanan proje yapısı

```text
Ebabil Mujoco/
├── .venv/                       # Git dışında yerel sanal ortam
├── docs/
│   ├── PROJECT_DESIGN.md
│   ├── LITERATURE_REVIEW.md
│   └── IMPLEMENTATION_PLAN.md
├── src/ebabil/
│   ├── assets/mjcf/             # Wheel içine alınan MuJoCo kaynakları
│   ├── configs/                 # Wheel içine alınan varsayılan TOML'ler
│   ├── dynamics/
│   ├── controllers/
│   ├── sensors/
│   ├── perception/
│   ├── scenes/
│   ├── envs/
│   ├── rewards/
│   ├── curriculum/
│   ├── randomization/
│   ├── safety/
│   ├── training/
│   ├── deployment/
│   └── evaluation/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   └── regression/
├── runs/                        # Git dışında deney çıktıları
├── pyproject.toml
├── requirements.lock
└── README.md
```

## 3. Faz 0 — Proje ve geliştirme ortamı

### İşler

- Proje kökünde `.venv` oluşturmak
- Python 3.13.2 ile çekirdek paket uyumluluğunu smoke test etmek
- `pyproject.toml` ve paket yapısını oluşturmak
- MuJoCo, Gymnasium, Stable-Baselines3 ve PyTorch'u kurmak
- Doğrulanmış sürümleri lock dosyasına almak
- `.gitignore` hazırlamak
- `pytest`, formatlama ve statik analiz araçlarını eklemek
- Temel script giriş noktalarını tanımlamak; kararlı kullanıcı akışı oluştuğunda CLI entry-point eklemek
- Seed ve config yükleme altyapısını oluşturmak

### Teslimatlar

- Aktifleştirilebilir `.venv`
- Kurulabilir `ebabil` Python paketi
- Tek komutla dependency ve import smoke testi
- Test ve lint komutları

### Doğrulama kapısı

- `import mujoco`, `gymnasium`, `stable_baselines3`, `torch` başarılı
- Basit MuJoCo XML modeli headless yüklenebiliyor
- Basit render testi yerel makinede çalışıyor
- Test komutu temiz geçiyor
- Temiz ortamdan lock dosyasıyla yeniden kurulum yapılabiliyor

**Gerçekleşen kalite kapısı:** Test keşfi `pytest`, lint/import denetimi `ruff`,
format doğrulaması `ruff format --check` ve üretim kodu tip denetimi `mypy` ile
yürütülür. Varsayılan MJCF/TOML kaynakları wheel package-data içindedir.

## 4. Faz 1 — Quadrotor MJCF modeli

### İşler

- Gövde, kollar, rotor siteleri ve collision geometry oluşturmak
- Kütle merkezi ve atalet parametrelerini config'e taşımak
- Free joint, gravity ve zemin eklemek
- Rotor frame ve dönüş yönlerini açık biçimde tanımlamak
- Debug amaçlı coordinate frame/site görselleştirmeleri eklemek
- Model yükleme ve pasif viewer script'i yazmak

### Teslimatlar

- Parametrik `quadrotor.xml`
- Model inspector/viewer
- Kütle, atalet ve rotor yerleşim raporu

### Doğrulama kapısı

- Model uyarısız derleniyor
- Toplam kütle config ile eşleşiyor
- Center of mass beklenen yerde
- Serbest düşüş fiziksel olarak doğru yönde ve beklenen ivmede
- Collision geometry görsel modelle tutarlı

## 5. Faz 2 — Rotor ve araç dinamiği

### İşler

- Motor komutundan rotor hızına birinci dereceden gecikme
- Rotor hızından thrust üretimi
- Rotor reaction torque ve yaw momenti
- X-konfigürasyon mixer
- Motor alt/üst limitleri
- Aerodinamik gövde sürüklemesi
- Harici kuvvet ve rüzgâr arayüzü
- Physics substep yapısı

### Birim testleri

- Tek rotor thrust yönü
- Dört eşit rotorla net roll/pitch/yaw momentinin yaklaşık sıfır olması
- Karşılıklı rotor farkıyla doğru roll/pitch işareti
- CW/CCW rotor farkıyla doğru yaw işareti
- Hover thrust değerinin `m*g` ile eşleşmesi
- Motor step response zaman sabiti
- Doygunluk davranışı

### Doğrulama kapısı

- Açık döngü kuvvet ve moment testleri tolerans içinde
- Hover thrust analitik hesapla eşleşiyor
- Sabit fizik timestep'inde enerji patlaması veya solver kararsızlığı yok
- Aynı seed ve girdilerle deterministik nominal sonuç alınabiliyor

## 6. Faz 3 — Klasik uçuş kontrolü

**Durum:** Baseline tamamlandı ve doğrulama kapısı geçildi (`100/100` seed'li nominal başlangıç, 10 Eylül 2026). Ayrıntılar `docs/PROGRESS_LOG.md` ve `docs/decisions/ADR-004-cascaded-flight-controller.md` içindedir.

### İşler

- Body-rate PID
- Attitude kontrolcüsü
- Hız kontrolcüsü
- İrtifa ve yaw kontrolü
- Anti-windup ve komut limitleri
- Mixer saturation handling
- Referans komut üreticileri
- Kontrol logları ve plotting

### Test senaryoları

- Hover
- Başlangıç attitude hatasından toparlanma
- Dikey step response
- X/Y hız step response
- Yaw-rate takibi
- Dairesel ve yumuşak rastgele trajectory takibi
- Sabit rüzgâr altında hover

### Doğrulama kapısı

- Nominal başlangıçların en az yüzde 99'unda belirlenen süre boyunca çarpışmasız hover
- Kalıcı attitude ve irtifa hataları tanımlanan tolerans içinde
- Motor doygunluğu sürekli hâle gelmiyor
- Kontrol sinyallerinde açıklanamayan yüksek frekanslı salınım yok
- Tüm işaret ve frame dönüşümleri otomatik testlerle korunuyor

Baseline'da hover, birleşik konum/tutum bozucusu, yaw referansı ve sabit rüzgâr regression testidir. Dairesel/rastgele trajectory, multi-rate scheduling ve ayrıntılı frekans analizi Faz 5 ortam entegrasyonuyla genişletilecektir.

## 7. Faz 4 — Sensörler ve sahne üretimi

**Durum:** İlk baseline tamamlandı ve doğrulama kapısı geçildi (`45/45` toplam test, 10 Eylül 2026). Ayrıntılar `docs/PROGRESS_LOG.md` ile ADR-005/ADR-006 içindedir.

### İşler

- IMU benzeri gyro ve accelerometer okuması
- Durum kestirimi arayüzü: önce ground-truth üzerine kontrollü hata modeli
- Rangefinder/LiDAR ışınları
- Sensör normalize etme ve clipping
- Noise, bias, delay, sample-and-hold ve dropout modelleri
- Statik obstacle primitive'leri
- Seed tabanlı prosedürel sahne üretici
- Başlangıç/hedef/engel çakışma denetimi
- Basit geometrik yol bulunabilirlik testi

### Doğrulama kapısı

- Işın mesafeleri bilinen geometrilerde doğru
- Body/world frame dönüşümleri referans örneklerle doğru
- Gürültüsüz sensör çıktısı fizik state'iyle eşleşiyor
- Sabit seed aynı sahneyi ve aynı gürültü serisini üretiyor
- Üretilen sahnelerde başlangıç ve hedef güvenli açıklığa sahip

İlk baseline 16 ışınlı planar rangefinder, rangefinder'a bağlı
sample/noise/bias/delay/dropout modeli ve axis-aligned kutular kullanır. Hedef
donanımda LiDAR bulunmadığı için bu sensör artık yalnız legacy regression,
privileged teacher/etiket ve değerlendirme metriğidir; deploy edilecek actor
girdisi değildir. Kamera geçişi ADR-011 ve Faz 6.5 kapsamındadır.

## 8. Faz 5 — Gymnasium ortamı

**Durum:** Baseline tamamlandı ve doğrulama kapısı geçildi (`58/58` toplam test, SB3 `check_env`, 2000 adım random rollout; 10 Eylül 2026). Kesin v1 sözleşmesi ADR-008 içindedir.

### İşler

- `EbabilNavigationEnv` sınıfı
- Observation ve action space tanımları
- Normalize action dönüşümü
- Physics/control frame skipping
- `reset`, `step`, `render`, `close`
- `terminated` ve `truncated` ayrımı
- Reward bileşenleri
- Başarı, collision ve failure reason bilgileri
- `human` ve `rgb_array` render modları
- Gymnasium registration

### Testler

- Gymnasium environment checker
- Observation sınır ve dtype testi
- Action clipping/validation testi
- Time-limit semantiği
- Goal/collision termination
- Reset sonrası geçerli state
- Aynı seed ile rollout tekrarı
- Binlerce random-action adımında NaN/Inf kontrolü

### Doğrulama kapısı

- `check_env` başarılı
- Random policy uzun smoke testte simülatörü bozmuyor
- Reward bileşenleri tek tek doğrulanabiliyor
- Eğitim ve evaluation modları aynı görev semantiğini koruyor

## 9. Faz 6 — Engelsiz PPO baseline

**Uygulama durumu (11 Eylül 2026):** Eğitim/checkpoint/resume, çok süreçli
vektör ortam, normalization state'i, CSV/TensorBoard logları, deterministic
evaluation, 500 Hz viewer replay, kararlı hedef bekleme, endpoint senaryoları,
held-out manifest, baseline karşılaştırmaları ve çoklu-seed aggregate runner
tamamlandı. `20260910` seed'li 100k pilot held-out `20/20` başarıyla tüm
performans kontrollerini geçti. Gerçek araçta rangefinder bulunmadığının
kesinleşmesiyle kalan iki seed ve full kalite kapısı ana geliştirme yolundan
çıkarıldı. Bu koşu PPO altyapı baseline'ıdır; görsel kaçınma veya sim-to-real
kanıtı değildir. Ayrıntı:
`docs/PHASE_6_PPO_BASELINE.md` ve ADR-009.

### İşler

- PPO eğitim script'i
- Paralel vectorized environments
- Observation/reward normalization stratejisi
- Checkpoint, best model ve resume desteği
- TensorBoard/CSV logları
- Deterministic evaluation
- Episode video kaydı
- Config ve seed metadata kaydı

### Başlangıç deneyleri

- Sabit başlangıç → sabit hedef
- Rastgele başlangıç → sabit hedef
- Rastgele başlangıç → rastgele hedef
- İdeal sensör → düşük gürültülü sensör

### Doğrulama kapısı

- Held-out engelsiz sahnelerde en az yüzde 95 başarı
- Araç hedefte belirlenen süre kararlı kalabiliyor
- Sonuç en az 3 seed üzerinde benzer eğilim gösteriyor
- PPO, random policy'den ve basit doğrudan-hedef baseline'ından açıkça iyi
- Evaluation videosunda reward hacking belirtisi yok

## 10. Faz 6.5 — Kamera algısı ve vision environment temeli

### İşler

- Kamera modeli, intrinsics/extrinsics ve görüntü zamanlama konfigürasyonu
- Arducam B0273 olduğu doğrulanırsa 75 derece yatay FOV ve motorized-focus
  parametreleri; gerçek SKU/lens ile kalibrasyon
- Waveshare Orin NX carrier board + JetPack/L4T + IMX477 driver bench gate
- Quadrotor MJCF üzerinde tek kamera ve headless RGB/depth/segmentation capture
- Dikdörtgen/kare hedef alan primitive'i, güvenli iç bölge ve hover anchor
- Kamera extrinsic/FOV sweep'iyle approach ve hedef üstü hover görünürlük testi
- RGB actor verisi ile privileged etiketlerin kesin ayrımı
- Hafif görsel encoder ve serbest-alan/çarpışma-riski yardımcı hedefleri
- Görüntü + deploy edilebilir FC state + waypoint için `Dict` observation
- Normalize kalan segment zaman bütçesinin deploy edilebilir observation alanı
- `EbabilVisionNavigation-v0` / `vision-v1` Gymnasium sözleşmesi
- Tek kamera FOV'una uygun ileri/dikey/yaw action sözleşmesi
- Frame history veya recurrent encoder için ilk ablation
- Kamera latency, jitter, frame-drop ve görüntü randomization altyapısı
- ONNX export ve Jetson Orin NX 8GB üzerinde TensorRT benchmark planı
- Pixhawk/autopilot setpoint ve telemetry adapter arayüzü; MAVLink adayı
- Pixhawk 2.4.8 board ID/FMU hedefi, flash ve sensör envanterinin pervanesiz
  bench testi
- Privileged channel leakage testleri

### Doğrulama kapısı

- Aynı seed aynı RGB/etiket akışını üretir
- Actor observation'ında raycast, depth veya simülatör tam state'i bulunmaz
- Görüntü coordinate frame'i ve kamera montaj dönüşümü sentetik hedeflerle doğru
- Hedef alan approach ve hover sırasında beklenen görüş/takip koşulunu sağlıyor
- Kamera görünürlük alanı dışındaki hareketler action/safety katmanında kısıtlı
- Encoder, held-out texture/aydınlatma sahnelerinde belirlenen perception
  metriklerini karşılar
- Kısa PPO smoke training ve ayrı deterministic evaluation başarılı
- ONNX çıkışı ile PyTorch çıktısı tolerans içinde eşleşir
- Jetson Orin NX üzerinde uçtan uca capture-to-setpoint gecikmesi ölçülür
- Kamera 30 dakikalık capture testinde beklenmeyen frame drop/driver hatası üretmez

## 11. Faz 7 — Kamera tabanlı statik engelden kaçınma ve curriculum

### İşler

- Engel yakınlığı ve collision reward bileşenleri
- Curriculum durum makinesi
- Başarı tabanlı seviye geçişleri
- Kamera görüntüsü/latent'i üzerinden kaçınma; rangefinder actor girdisi yasak
- Privileged teacher ve camera student karşılaştırması
- Texture, aydınlatma ve kamera parametresi sahne havuzları
- Sahne havuzu ve zorluk parametreleri
- Geçit, koridor, sütun ve rastgele kutu sahneleri
- Başarısız episode sınıflandırması
- Curriculum checkpoint/resume
- Eski seviyeleri eğitim karışımında tutan forgetting kontrolü
- Global rota/waypoint üretici arayüzü
- Ego-merkezli yerel sahne penceresi ve yakın engel etkinleştirme
- Observation v2 ve yeni environment ID'si
- ADR-013 formülüyle rota uzunluğuna bağlı episode/segment zaman bütçesi
- Yerel eğitim episode'u ile kesintisiz mission evaluation ayrımı
- Hedef alan için `transit -> acquired -> approach -> settle -> hold -> success`
  state machine
- Hedef kaybında hover/abort ve yeniden arama davranışı
- Hedef dikdörtgeninin araç izdüşümü ve localization belirsizliğiyle
  küçültülmesinden elde edilen güvenli iç bölge
- Hover success için irtifa, doğrusal hız, roll/pitch, açısal hız ve kesintisiz
  hold kontrolleri

### Curriculum sırası

1. Tek engel
2. Seyrek engeller
3. Geniş geçit
4. Koridor
5. Dar geçit
6. Yoğun karma sahne

Engel sırasına ek olarak görev mesafesi bağımsız curriculum ekseni olacaktır.
**Kesin merdiven ADR-018 içindedir**; aşağıdaki özet ondan türetilmiştir.

| Seviye | Rota | Workspace | Seyir | Episode | Engel | Rejim |
|---|---:|---:|---:|---:|---:|---|
| D0-V | `4-6 m` | `6 x 5 m` | `1.5 m/s` | `~400` | `0` | ✅ tamamlandı |
| D1 | `5-15 m` | `20 x 20 m` | `1.5 m/s` | `~710` | `~40` | eğitim |
| D2 | `15-50 m` | `65 x 65 m` | `6 m/s` | `~650` | `~210` | eğitim |
| D3 | `50-150 m` | `200 x 200 m` | `6 m/s` | `~1 235` | `~800` | eğitim |
| D4 | `150-500 m` | `650 x 650 m` | `6 m/s` | `~3 277` | yerel pencere | eğitim |
| D5 | `0.5-2 km` | `2.5 x 2.5 km` | `6 m/s` | `~12 027` | yerel pencere | yalnız evaluation |

**Hedef konumu yalnız yaklaşık bilinir (ADR-019).** Araç kendi konumunu
hatasız bilir (görsel pozisyonlama varsayımı, bu projede modellenmez); hedefin
konumunu episode başına `R` yarıçaplı hatayla bilir ve gerçek konumunu hiçbir
zaman öğrenmez. `R` merdiveni: D1 `2 m`, D2 `5 m`, D3 `15 m`, D4-D5 `30 m`.
Bu, kamerayı yer bulma işine dahil eder ve durum makinesine `search` durumunu
zorunlu kılar. Arama geometrisi: kapsama `= 3.71 x irtifa`, optimum arama
irtifası `10.9 m` (kapsama `40.3 m`). Davranış: tırmanarak ara, alçalarak yerleş.

**Görev kapsamı sadeleştirildi (ADR-018).** Her episode'un tek hedef noktası
vardır; konumu episode'lar arasında değişir fakat **ara waypoint yoktur**.
Mission segment manager ve çoklu waypoint iş paketleri plandan çıkarıldı.
ADR-013 yalnız rota uzunluğuna bağlı zaman bütçesi formülüyle geçerliliğini
korur ve artık tüm episode'a uygulanır.

**Seyir hızı algı menziliyle sınırlıdır.** `8 m` free-space ufkunda `30 derece`
seyir tiltiyle durabilmek ham `~9 m/s` verir; encoder'ın free-space hatası
(`MAE 0.25` x `8 m` = `+-2 m`) düşülünce tasarım hızı `6 m/s`'dir. Hız D2'de
yükseltilir ve `fc_state` normalizasyonunu değiştirdiği için `vision-v2`
gerektirir.

**D5 eğitim rejimi değildir.** `12 027` adımlık episode'da PPO kredi ataması
pratik olmadığı için D5 aynı politikayla eğitilmeden ölçülür; kompozisyon
kanıtı olarak raporlanır.

Mevcut `10 cm` grid bütün büyük alana genişletilmeyecektir; D1'den itibaren
kaba grid ön elemesi + aday koridorda ince grid doğrulaması kullanılır. Ölçülen
maliyet: `6 m` workspace `70 ms/sahne`, `15 m` `611 ms`, `30 m` `8.1 s`.
Ego-merkezli yerel sahne penceresi D3'ten zorunludur; ölçülen MuJoCo geom
bütçesi `~800` engeldir.

### Doğrulama kapısı

- Held-out statik sahnelerde hedef başarı oranı karşılanıyor
- Çarpışma oranı tanımlanan üst sınırın altında
- Politika tek bir engel geometrisine aşırı uyum göstermiyor
- Curriculum açık/kapalı ablation'ı raporlanmış
- Minimum clearance ve path efficiency ölçülmüş
- Dinamik ve sabit zaman bütçesi ablation'ında timeout/başarı raporlanmış
- Ara waypoint geçişi mission modunda fizik state'ini sıfırlamıyor
- Nihai başarı yalnız hedef güvenli iç alanındaki `3 s` hover hold ile oluşuyor
- Hedef dışında bekleme, kısa süreli bölge geçişi veya hedef tespiti tek başına
  başarı üretmiyor

İlk önerilen hedef: kolay/orta held-out sahnelerde en az yüzde 90, zor sahnelerde en az yüzde 75 başarı. Bu eşikler görev geometrisi netleştiğinde revize edilebilir.

## 12. Faz 8 — Robustluk ve sim-to-real hazırlığı

### İşler

- Randomization parametre kayıt sistemi
- Kütle, atalet, thrust, motor lag ve drag varyasyonları
- Sensör bias/noise/delay/dropout
- Kamera pozlama, blur, lens, extrinsic ve frame-timing bozulmaları
- Sabit, gust ve yön değiştiren rüzgâr
- Action latency ve observation latency
- Selective domain randomization
- Observation/action history ablation
- Smoothness reward ablation
- Çoklu PPO seed deneyleri

### Test dağılımları

- In-distribution nominal test
- In-distribution randomized test
- Eğitim aralığı sınırlarında test
- Eğitim aralığı dışında kontrollü OOD test
- Kombine worst-case'e yakın test

### Doğrulama kapısı

- Nominal politika ile robust politika arasında kontrollü karşılaştırma mevcut
- Randomizasyon performansı iyileştirirken nominal davranışı kabul edilemez ölçüde bozmuyor
- Fizik/sensör parametre sweep sonuçları raporlanıyor
- Action smoothness, enerji ve başarı birlikte değerlendiriliyor
- Sonuçlar en az 5 seed ile raporlanıyor

## 13. Faz 9 — Güvenlik katmanı ve gerçek sistem köprüsü

### İşler

- Policy output validator
- Hız, acceleration, attitude ve thrust envelope
- Sensör timeout ve stale-data denetimi
- Acil hover/land state machine
- Basit reactive obstacle override
- Kamera FOV dışı hareket ve stale-frame kısıtları
- Safety filter arayüzü
- Composite CBF/QP fizibilite prototipi
- Flight controller adapter arayüzü
- SITL ve daha sonra HITL planı
- Log replay ve actuation playback testi
- QGC ile seçilen tek firmware'in konfigürasyon ve telemetry doğrulaması
- Pixhawk 2.4.8 için firmware modül/flash yeterliliği ve pre-arm sensör
  kontrolleri

### Doğrulama kapısı

- NaN, Inf ve aralık dışı politika eylemleri uçuş kontrolüne ulaşmıyor
- Güvenlik katmanı kapalı/açık çarpışma ve müdahale sonuçları raporlanıyor
- Policy process'i durduğunda failsafe devreye giriyor
- SITL/HITL boyunca frekans ve latency bütçeleri karşılanıyor
- Manuel override ve kill mekanizması doğrulanmadan pervaneli test yapılmıyor

## 14. Faz 10 — İleri araştırma kolları

Temel teslimattan sonra bağımsız dallar olarak:

- Hareketli engeller
- Recurrent PPO veya frame stacking karşılaştırması
- Asymmetric actor-critic
- Body-rate + thrust eylem seviyesi
- Doğrudan motor/RPM kontrolü
- Derinlik kamerası veya ikinci kamera donanım ablation'ı
- Uçtan uca RGB politika ile modüler encoder karşılaştırması
- Kısa horizon waypoint/trajectory politikası
- Gerçek loglardan residual dynamics/perception model
- Çoklu İHA ve merkezi olmayan collision avoidance

## 15. Deney yönetimi

Her eğitim koşusu şu metadata'yı saklamalıdır:

- Deney adı ve tarih
- Config'in tam kopyası
- Global, environment ve model seed'leri
- Git commit kimliği mevcutsa
- Python ve paket sürümleri
- Donanım ve device seçimi
- Toplam timestep ve duvar saati
- Checkpoint'ler
- Evaluation metrikleri
- En az bir örnek video

Önerilen koşu adlandırması:

```text
ppo_<task>_<curriculum>_<randomization>_seed<N>_<timestamp>
```

## 16. Test stratejisi

### Birim testleri

- Matematik ve frame dönüşümleri
- Mixer
- Thrust ve torque
- Reward bileşenleri
- Noise ve delay buffer'ları
- Kamera projection, preprocessing ve timestamp mantığı
- Privileged observation leakage denetimi
- Curriculum geçişleri
- Rota-temelli zaman bütçesi, clamp ve step yuvarlama mantığı
- Hover güvenli iç bölgesi ve kesintisiz hold sayacı
- Config doğrulama

### Entegrasyon testleri

- MJCF + dinamik + kontrolcü
- Sensör + scene ray casting
- Kamera RGB/depth/segmentation render ve frame senkronizasyonu
- Environment reset/step
- Yerel segmentten mission waypoint geçişine fizik state sürekliliği
- Terminal hover state machine ve target-loss abort
- PPO kısa smoke training
- Checkpoint save/load

### Regresyon testleri

- Sabit seed trajectory özeti
- Hover hata sınırları
- Bilinen sahnede klasik controller sonucu
- Observation sırası ve ölçeği
- Reward decomposition
- `terminated`/`truncated` ve time-limit semantiği
- Segment/mission bütçe ve hover başarı özeti
- Policy inference action shape/range
- PyTorch/ONNX/TensorRT inference parity ve latency bütçesi

Fizik floating-point ayrıntıları platformlara göre değişebileceğinden regresyon testleri tam dizi eşitliği yerine gerekçeli toleranslar kullanmalıdır.

## 17. Ana ablation planı

| Kod | Değişken | Kontrol | Deney |
|---|---|---|---|
| A1 | Curriculum | Kapalı | Açık |
| A2 | Sensör | İdeal | Noise + bias + delay |
| A3 | Randomization | Yok | Seçici |
| A4 | Smoothness | Yok | Action-difference penalty |
| A5 | History | Yok | Observation/action stack |
| A6 | Rotation | Quaternion | Gravity/rotation representation |
| A7 | Safety | Kapalı | Safety filter açık |
| A8 | Kontrol seviyesi | Velocity | Body-rate + thrust |
| A9 | Görsel temsil | Etiketsiz donmuş VAE (ADR-022) | (a) etiketli encoder, (b) PPO içinde ince ayar, (c) saf piksel PPO |
| A10 | Zaman bilgisi | Latent yığını `k = 4` (ADR-022) | `k = 1` / recurrent |
| A11 | Öğretmen | Yok | Privileged teacher/student |

Her ablation diğer değişkenleri mümkün olduğunca sabit tutmalı ve aynı evaluation seed setini kullanmalıdır.

## 18. Riskler ve azaltma yöntemleri

| Risk | Belirti | Azaltma |
|---|---|---|
| Hatalı frame/işaret | Kontrolcü beklenen yönün tersine gider | Analitik birim testleri ve frame görselleştirmesi |
| Fizik kararsızlığı | NaN, hız patlaması, solver uyarısı | Küçük timestep, limitler, enerji ve state kontrolleri |
| Reward hacking | Hareketsiz kalma veya engel yanında titreşim | Reward decomposition, video ve davranış testleri |
| Aşırı randomization | PPO temel görevi öğrenemez | Nominal başlangıç ve başarı tabanlı kademeli açılım |
| State leakage | Simülasyonda yüksek, gerçekçi observation'da düşük başarı | Actor observation sözleşmesi ve ayrı privileged state |
| Overfitting | Eğitim sahnesinde iyi, testte kötü | Ayrı seed/sahne havuzları ve prosedürel test |
| Gürültülü action | Motor ısınması ve enerji kaybı | Smoothness reward, filtre ve spektral metrik |
| Sim-to-real güven yanılgısı | Sim başarısının gerçek güvenlik sayılması | SITL → HITL → korumalı gerçek test sırası |
| Yavaş eğitim | Düşük simulation throughput | Headless çalışma, vectorization ve profiling |
| Görsel sim-to-real farkı | Sentetik başarı, gerçek görüntüde başarısızlık | Görsel randomization, gerçek kayıt test seti ve teacher/student yaklaşımı |
| Tek kamera kör bölgesi | Yan/geri harekette görünmeyen engele yaklaşma | FOV uyumlu action, yaw-before-translate ve temporal yerel harita |
| Jetson gecikmesi | Stale görüntüyle kontrol ve düşük policy frekansı | Cihaz üstü profiling, hafif encoder, ONNX/TensorRT ve watchdog |
| Localization drift | Uzun görevde waypoint yönünün bozulması | Görsel harita/relocalization; VIO drift ve loop-closure testi |
| GPS bulunmaması | Hedef alan görünmeden global referansın kaybolması | Önceden görsel harita, VIO + loop closure ve relocalization kalite kapısı |
| IMX477/carrier uyumsuzluğu | Kamera enumerate olmuyor veya frame drop | SKU/FFC/CSI/JetPack driver bench gate |
| Değişken odak | Kalibrasyon ve görsel temsil uçuşta değişiyor | Kalkış öncesi kalibre edilmiş sabit focus; autofocus ayrı ablation |
| Pixhawk 2.4.8 varyant belirsizliği | Yanlış firmware hedefi, eksik modül veya sensör pre-arm hatası | QGC board ID/FMU, flash ve sensör envanteri; otomatik hedef doğrulaması ve pervanesiz bench gate |
| Sabit episode süresinin yetersiz kalması | Geçerli uzun rotalarda yüksek timeout | ADR-013 rota temelli bütçe ve segmentli mission evaluation |
| Aşırı uzun episode | Zayıf kredi atama, düşük reset çeşitliliği ve pahalı failure | Yerel `5-30 m` eğitim segmentleri ve `T_max=60 s` başlangıç sınırı |
| Hedefin hover sırasında kameradan çıkması | Acquired sonrası target-loss | Kamera extrinsic/FOV sweep'i, aşağı eğimli montaj adayı ve sınırlı VIO grace period |

## 19. Tamamlanan implementasyon dilimi: vision-v1

Görsel randomization, dört-split dataset, fail-closed quality ve bağımsız audit
kapıları tamamlandı; ayrıntılar `docs/PHASE_6_5_VISION_DATASET_ENCODER.md` ve
ADR-014 içindedir. Tamamlanan sonraki dilim
`EbabilVisionNavigation-v0` ve `vision-v1` observation sözleşmesidir:

1. `runs/vision_audit_v2_20260911/approval_manifest.json` üzerinden hash zinciri
   doğrulanan `vision-encoder-v9` checkpoint'ini yüklemek ve eğitim sırasında
   varsayılan olarak dondurmak
2. Actor observation'ını encoder latent/çıktıları, deploy edilebilir Pixhawk FC
   state'i, ego-frame waypoint ve normalize kalan segment zamanı olarak kurmak
3. Depth, raycast, obstacle dünya koordinatı, tam MuJoCo state'i ve privileged
   etiketleri actor yolundan fail-fast reddetmek
4. ADR-013 rota uzunluğu formülüyle dinamik segment horizon'ı ve
   `terminated/truncated` semantiğini uygulamak
5. Kamera `30 Hz`, policy/control frekansı ve sample-and-hold zamanlamasını
   deterministik scheduler ile bağlamak
6. `check_env`, observation sınır/dtype, seed tekrarı, random-policy uzun smoke
   ve kısa PPO smoke + ayrı deterministic evaluation çalıştırmak
7. Observation/action/frekans sözleşmesini yeni ADR ile sabitlemek

Bu dilimin tamamlanma tanımı:

- Actor observation yalnız gerçek sistemde üretilebilecek kanalları içerir.
- Aynı seed aynı sahne, görüntü zamanlaması ve observation dizisini üretir.
- Dinamik horizon kısa segmenti gereksiz uzatmaz, uzun geçerli segmenti sabit
  süre yüzünden erken kesmez ve üst sınırı aşmaz.
- Encoder inference kamera frame'leri arasında sample-and-hold davranır; stale
  frame bilgisi observation/info içinde ölçülebilir.
- Gymnasium checker, hedefli testler, kısa PPO smoke ve deterministic evaluation
  temiz geçer.

Bu kapıdan sonra hedef bölge için
`acquired -> approach -> settle -> hold -> success` state machine ve reward
bileşenleri uygulanacaktır. Kilometre mission manager'ı D0-V–D2 yerel politika
kapısından önce açılmayacaktır.

## 20. Tamamlanan implementasyon dilimi: terminal hover

`EbabilVisionNavigation-v0` ve `vision-v1` kapısı ADR-015 ile; hover geometrisi
ve hedef açıklığı güvencesi ADR-016 ile tamamlandı. ADR-016'nın bu dilimi
bağlayan üç kısıtı şudur:

- Hold ve durum geçişleri `target_score` ile kapılanır; `target_center_xy`
  yalnız yaklaşma sırasında yumuşak hizalama sinyalidir. Hedef merkezi hover
  pozunda kadraj dışındadır ve görünmediğinde `(0, 0)` sentinel'i "tam merkezde"
  durumundan ayırt edilemez.
- Başarı kararı privileged ground truth ile verilir ve actor observation'a
  girmez.
- Hedefin hover hacmi sahne üretiminde engelsizdir; merkez-görünür hold bandı
  güvenli iç bölgenin arka kenarında ~`0.11 m` genişindedir.

ADR-012 terminal görevi ADR-017 ile uygulanmıştır. Uygulanan altı madde:

1. `transit -> acquired -> approach -> settle -> hold -> success` durumlarını
   ve izinli geçişlerini açık enum/state machine olarak tanımlamak
2. Target score sürekliliği, stale frame ve sınırlı target-loss grace
   semantiğini config ve testlerle sabitlemek (merkez kanalı kapı değildir)
3. Hedef güvenli iç bölgesi, hover irtifası, hız, tilt ve angular-rate
   koşullarını privileged başarı doğrulamasında birlikte uygulamak
4. `target_progress`, `target_alignment`, `settle_speed`, `hover_hold`,
   `smoothness`, `collision` ve `out_of_bounds` reward bileşenlerini ayrı loglamak
5. Hold koşulu bozulduğunda sayacı sıfırlamak; çarpışmayı hiçbir durumda başarı
   saymamak; target-loss/unsafe/stale durumda abort üretmek
6. Durum geçiş, kesintisiz hold, target-loss, abort ve reward toplamı testlerinden
   sonra kısa PPO + deterministic evaluation smoke çalıştırmak

Bu dilim mission segment zincirini veya D0-V–D2 uzun eğitimini başlatmadı.
Ayrıntılar `docs/PHASE_7_TERMINAL_HOVER.md` ve ADR-017 içindedir.

## 21. Tamamlanan implementasyon dilimi: D0-V yerel vision politikası

**Durum (22 Eylül 2026):** 11 Eylül 2026'da tamamlandı (`20/20`, tek seed,
engel yok); ayrıntı `docs/PHASE_7_D0V_VISION_POLICY.md`. Bölüm tarihsel
kayıt olarak korunur; güncel iş §23'tedir.

**Kapsam düzeltmesi.** REV-021 ilk yazımında D0-V'nin ön koşulu sayılmıştı; bu
yanlıştı. ADR-010'a göre D0-V "legacy geometriyi `vision-v1` ile yeniden
öğrenmek", yani sabit `4-6 m` geometridir; değişken endpoint **D2**'de gelir.
Onaylı `vision-encoder-v9` sabit geometride eğitildiği için D0-V'nin gerektirdiği
encoder tam olarak odur. REV-021 D2 ve sonrasını bloke eder.

D0-V iş paketi:

1. Eğitim altyapısını vision ortamına bağlamak: `EbabilVisionNavigationEnv`
   fabrikası, `MultiInputPolicy`, Dict observation normalization kararı ve
   hover metriklerinin monitor'a alınması
2. Throughput ölçmek ve gerçek bir timestep bütçesi belirlemek
3. Sabit geometride D0-V eğitimini çalıştırmak
4. Held-out hover başarı oranı, çarpışma, abort dağılımı (`target_lost`,
   `stale_frame`), timeout oranı ve yerleşme süresini ayrı raporlamak
5. ADR-017'de "D0-V'de ölçülecek" diye kaydedilen iki soruyu yanıtlamak:
   terminal durumun actor observation'ında olmamasının öğrenme maliyeti ve
   `hover_hold` üst sınırı ile `success_bonus` oranı
6. En az üç seed üzerinde tekrarlamak

## 22. D2 ön koşulu: risk head

REV-018 dataset düzeyinde kapatıldı: `vision-dataset-v4c` rastgele rota
geometrisi (`2.08-4.65 m`) ve ADR-012 hover zarfından örneklenen hold pozu
ailesi (`%23.4`) kullanıyor. Ancak bu dağılımda onaylanabilir encoder
üretilemedi; risk kanalı altı kalite kapısını birlikte geçemedi (REV-021).
Ayrıntılar `docs/REVIEW_FINDINGS.md` içindedir.

**D2 öncesinde yapılacaklar:**

1. Risk head'ine kapasite eklemek. Şu an `latent -> 3` tek doğrusal katman;
   gizli katman veya ayrı bir gövde denenmeli
2. Risk etiketini üç bölge ikili eşiğinden (`free_space < 0.25`) daha zengin bir
   temsile taşımak; yumuşak hedef veya bölge sayısı artırımı
3. Grayscale ablation'ını eğitim sırasında açıkça düzenleyici olarak kullanmak;
   bağlayıcı kısıt grayscale precision'ıdır
4. Kapı eşikleri veya karar eşiği değiştirilecekse gerekçesini ayrı ADR ile
   belgelemek. Bu dilimde hiçbiri değiştirilmedi
5. Development bütçesini dikkate almak: mevcut set üç quality koşusu, bir eşik
   taraması ve iki `pos_weight` denemesi tüketti; gerekiyorsa yeni salt ile yeni
   development seti üretilmeli

**REV-021 kapandıktan sonra D1-D2 curriculum'u:**

1. Curriculum durum makinesi ve başarı tabanlı seviye geçişlerini kurmak
3. Hover başarı oranı, çarpışma oranı, abort dağılımı (`target_lost`,
   `stale_frame`), timeout oranı ve yerleşme süresini ayrı metrik olarak
   raporlamak
4. Terminal durumun actor observation'ında olmamasının öğrenme maliyetini
   ölçmek; gerekiyorsa deploy edilebilir görsel faz kestirimini ayrı ADR ile
   açmak
5. `hover_hold` üst sınırı ile `success_bonus` oranını ablation ile
   doğrulamak
6. En az üç seed üzerinde held-out sahne kapısını çalıştırmak

Bu dilim kilometre mission manager'ını açmaz.

## 23. Tamamlanan/durdurulan implementasyon dilimi: etiketsiz görsel temsil (ADR-022)

**Not (26 Eylül 2026):** Kullanıcı kararıyla actor algısı ADR-023'e (YOLO
dedektör + RL politikası) geçti. Dilim 1 ve 2 tamamlandı ve baseline olarak
korunur; Dilim 3 eğitimi `528k / 600k` adımda iptal edildi; Dilim 3-5 bu
haliyle yürütülmez. Yürürlükteki plan §24'tür.

**Kabul:** 22 Eylül 2026, kullanıcı onayı. Ayrıntılı gerekçe ADR-022'dedir;
bu bölüm yalnız yürütme sırasını, dosyaları ve çıkış kapılarını verir.
Her dilim `docs/PROGRESS_LOG.md`'ye AGENTS.md §5 formatında yazılır.

### Bağlayıcı kurallar

- Actor'a giren görsel kanal, depth/segmentation/raycast türevi etiketle
  eğitilmiş ağdan gelemez. Etiketler yalnız probe ölçümü, privileged
  reward/termination ve tanı içindir.
- `EbabilVisionNavigation-v0` / `vision-v2`, `vision-encoder-v11` zinciri ve
  `LEGACY_ROLLOUT_SHA256` dondurulur; testleri her dilimde geçmelidir.
- Kapı eşikleri ölçüm sonucuna göre **düşürülmez**; geçilmeyen kapı yeni
  bulgu (`REV-0xx`) açar.
- Her eğitim sonucu raporlanmadan önce en az bir episode replay olarak
  izlenir (13 Eylül dersi).

### Dilim 1 — VAE kodlayıcı ve onay zinciri

**Durum (22 Eylül 2026):** kod, testler ve onay zinciri tamam; üç seed
kalite koşusu sabit eşikli hedef recall kapısında takıldı, tarama geçiremedi
(REV-049). Kullanıcı kararıyla ADR-022 Revizyon 1 (eşikten bağımsız hedef
kapısı, en iyi eşikte harita IoU, latent 128). İkinci kalite koşusu:
`runs/vae_quality_v2_20260922`: eşikten bağımsız kapıların hepsi üç seed'de geçti (AUROC `0.81-0.82`, en iyi eşikte harita IoU `0.50-0.54`); korunan sabit `0.5` eşiğindeki harita IoU iki seed'de `0.25-0.28` (kapı `0.30`), `quality_passed = False`. Kullanıcının 23 Eylül onayıyla Revizyon 1b: bu kapı kaldırıldı; üçüncü kalite koşusu (`runs/vae_quality_v3_20260923`) ve audit (`runs/vae_audit_v1_20260923`) geçti; onaylı kodlayıcı seed `20260911`. **Dilim 1 kapandı (23 Eylül 2026).**

İşler:

- `src/ebabil/perception/vae.py`: `VisualVAEConfig` (`input 96 x 72`,
  `latent_size 128` (Revizyon 1), `beta`, kod çözücü genişlikleri), `VisualVAE`
  (kodlayıcı ortalama/log-varyans, ayna kod çözücü), `vae_loss`
  (yeniden üretim + `beta * KL`), `train_visual_vae`, sürümlü checkpoint
  (`ebabil-vae-checkpoint-v1`), `load_visual_vae_checkpoint`,
  `encode_actor_rgb` (yalnız ortalama vektör).
- Eğitim verisi: `VisionNpzDataset`'in yalnız actor arşivini okuyan türevi
  (`ActorOnlyNpzDataset`); `record.privileged_file` hiç açılmaz.
- `src/ebabil/perception/probes.py`: donmuş `z` üzerinde doğrusal probe'lar
  (`free_space (3)`, hedef görünürlüğü (1), engel haritası (108)); metrikler
  mevcut `compute_metrics` ile aynı tanımlar. Yalnız ölçüm; probe
  ağırlıkları checkpoint'e yazılmaz.
- Etkin latent boyutu metriği (KL'si `> 0.01 nat` olan boyut sayısı) ve
  yeniden üretim MSE/PSNR raporu.
- `quality.py` genişletmesi veya `vae_quality.py`: `run_vae_quality`
  (3 seed), `run_vae_audit` (audit split, tek seferlik `audit_consumed.json`),
  `approval_manifest.json` (`kind = "visual-vae"`), `load_approved_visual_vae`
  (SHA-256 zinciri, probe kapıları geçilmemişse fail-closed).
- Config: `src/ebabil/configs/training/visual_vae.toml` (model, eğitim,
  `[quality]` eşikleri ADR-022 §3).
- Script'ler: `scripts/train_visual_vae.py`, `scripts/run_vae_quality.py`,
  `scripts/run_vae_audit.py`.
- Testler `tests/test_visual_vae.py`: shape/aralık/NaN reddi; `np.load`
  casusu ile privileged dosya erişiminin sıfır olması; privileged dizini
  kaldırılmış kopya dataset'te eğitimin çalışması; checkpoint round-trip;
  probe metriklerinin sentetik etiketle doğrulanması; manifest fail-closed.

Teslimatlar: onaylı `visual-vae-v1` manifest'i
(`runs/vae_audit_v1_<tarih>/approval_manifest.json`), `metrics.json`,
PROGRESS_LOG kaydı.

Doğrulama kapısı: ADR-022 §3 tablosu; 3 seed quality `passed`, audit
`passed`; `pytest`, `ruff`, `mypy` temiz; `LEGACY_ROLLOUT_SHA256` değişmedi.

Tahmini maliyet: `6528` kare, `96 x 72`, `45` epoch, CPU/MPS -> seed başına
dakikalar. Dilim toplamı bir iş günü.

### Dilim 2 — `EbabilVisionNavigation-v1` ve `vision-v3`

**Durum (23 Eylül 2026): tamamlandı.** Teslimatlar `runs/latent_vision_smoke_20260923`
(check_env temiz, 200 adım rastgele rollout, 64 adım PPO, 80 adım deterministik
evaluation) ve throughput ölçümü: `v1 / v0 = 0.76` (DR açık), `0.84` (DR kapalı),
kapı `>= 0.7` geçti; 16 yeni test, tam paket 318 test. Aşağıdaki listeden
sapmalar (gerekçeleri `docs/PROGRESS_LOG.md` Dilim 2 kaydında): (1) `v1`,
`v0`'dan türetildi; `v0` davranışı değiştirilmeden algı kancalarına ayrıştırıldı,
`v0` testleri ve legacy parmak izi değişmedi. (2) `HoverStateMachine` değişmedi:
privileged görünürlük `{0, 1}` skoru olarak mevcut skor girişinden geçer,
histerezis bu skorda inert. (3) Sahne geometrileri `box/cylinder/capsule`; küre
ve ellipsoid REV-035 kaçışını geri getireceği için dışarıda. (4) Config alanları
`[visual_randomization]` ve `[privileged_visibility]`; testler tek dosyada
(`tests/test_latent_vision_env.py`). (5) Fotometrik etkiler torch arama tablosu
ve seed'li gürültü bankasıyla `1.2 ms`/kare.

İşler:

- `src/ebabil/envs/latent_vision.py`: `EbabilLatentVisionNavigationEnv`
  (`EbabilVisionNavigationEnv`'den türetilmez; ortak parçalar
  `EbabilNavigationEnv` tabanından ve paylaşılan yardımcılardan gelir ki
  `v0` bit-bit donuk kalsın). `VISION_OBSERVATION_VERSION = "vision-v3"`,
  `VISION_ENVIRONMENT_ID = "EbabilVisionNavigation-v1"`.
- Latent yığını: `LatentStack(k = 4)`; reset'te ilk latent ile kopyalama;
  `latent_scale` kırpma ve kırpma oranının `info`'ya yazılması.
- Privileged görünürlük: `src/ebabil/perception/target_visibility.py`,
  `PrivilegedTargetObserver` (ortamın sahip olduğu segmentation kamerası;
  `target_visible`, `target_pixel_count`, `target_center_x`). Actor camera
  factory'sinden ayrı nesne; actor yolundan erişilemez.
- `HoverStateMachine`: `target_score` kapısı yerine boolean görünürlük
  girişi; zaman filtreleri (`acquire_seconds`, `grace_seconds`) aynen.
  Mevcut `v0` davranışı için skor tabanlı yol korunur (iki giriş modu).
- Görsel DR: `build_scene_model(..., visual_parameters=...)`;
  `RandomizedActorCamera` sarmalayıcısı kare başına fotometrik etkileri
  `(scene_seed, frame_index)` seed'iyle uygular. `VisionNavigationEnvConfig`'e
  `[visual_domain]` (train/held-out alan adları, açık/kapalı).
- Sahne config'i: `obstacle_geometries` alanı (`static_scene_d1a.toml`'a
  `["box", "cylinder", "capsule", "sphere"]`); `has_navigable_xy_path`
  AABB şişirmesi bütün primitive'ler için muhafazakâr olduğundan değişmez.
- `training/config.py`: `kind = "latent_vision"`, `vae_approval_manifest`,
  `visual_domain`; `env_factory` bağlantısı; `MultiInputPolicy` aynen.
- `scripts/smoke_latent_vision_env.py`, throughput ölçümü (`v0` ile
  karşılaştırma).
- Testler `tests/test_latent_vision_env.py`,
  `tests/test_visual_randomization_env.py`: ADR-022 doğrulama kapısı 2-6,
  8, 9.

Teslimatlar: `check_env` temiz, 2000 adım random rollout, `64` adım PPO
smoke, deterministik evaluation smoke, adım/s ölçümü.

Doğrulama kapısı: bütün testler; `v0` testleri ve legacy parmak izi
değişmedi; throughput `>= 0.7 x v0` (aksi halde privileged render
optimizasyonu: yalnız terminal fazda segmentation, ötesinde projection).

Tahmini maliyet: iki iş günü.

### Dilim 3 — Kademe 1 eğitimi: D1a, donmuş VAE

İşler:

- `src/ebabil/configs/training/ppo_latent_d1a.toml`: `600k`, `n_envs 4`,
  `static_scene_d1a.toml` (5 engel, geometri çeşitli), train görsel alanı.
- `scripts/evaluate_vision_policy.py`: `--env-version v0|v1`,
  `--visual-domain train|heldout`, `arrival_rate` metriği (gerçek anchor'a
  yatay mesafe `<= approach_radius_m` ve fiziksel failure yok), kapı
  geçilmemişse kırpma oranı ve hover koşul kesişimi raporu.
- Replay: `record_vision_replay.py` ve `view_policy_replay.py`'nin `v1`
  ile çalışması; en az bir episode izlenir.
- Üç seed (`20260912`, `20260913`, `20260914`).

Doğrulama kapısı (ADR-022 §10): deterministik, held-out görsel alan, 40
episode, 3 seed: `arrival >= %80`, `çarpışma <= %10`. `hover_success`
raporlanır, kapı değildir. Stokastik sonuç ayrıca raporlanır (REV-023).

Tahmini maliyet: seed başına `~1.5 saat` eğitim; toplam yarım iş günü koşu.

### Dilim 4 — Karşılaştırma ve ablation'lar

| Kod | Baseline | Varyant | Soru |
|---|---|---|---|
| A9-a | donmuş VAE | etiketli `v11` encoder, aynı `v1` ortamı ve DR | etiketin bedeli |
| A9-b | donmuş VAE | kodlayıcı PPO içinde çözülür (kademe 2) | ince ayarın kazancı |
| A9-c | donmuş VAE | saf piksel PPO, sıfırdan, aynı bütçe | uçtan uca tavanı |
| A10 | `k = 4` | `k = 1` | zamansal bilginin katkısı |
| L | latent `64` | latent `32` | kapasite |
| DR | açık | kapalı | görsel çeşitliliğin katkısı |
| P | `proximity` açık | `proximity_weight = 0` | privileged şekillendirmenin katkısı |

Her satır aynı evaluation seed setiyle, 3 seed, deterministik + stokastik.
Kademe 2 (A9-b) yalnız kademe 1 kapısı geçildikten sonra ana yola alınır.

### Dilim 5 — Baseline, merdiven ve sonrası

- DroNet tarzı reaktif baseline (`evaluation/baselines.py`): `v11`
  `obstacle_map`'ten yön seçimi, `v = v_max * (1 - P_coll)`; PPO ile aynı
  protokolde raporlanır (tasarım §12).
- Yoğunluk merdiveni `5 -> 10 -> 20 -> 40` (REV-039), warm start.
- REV-043 hover ölçütü kararı kullanıcıyla; `arrival` geçilip `hover_success`
  geçilmiyorsa ölçüt sorusu ADR-012 düzeyinde açılır.
- Ertelenenler: pitch jitter (`+-3 derece`), dinamik DR (Faz 8), optik
  akış/TTC tanı, asimetrik critic için yoğun derinlik.

### Riskler

| Risk | Belirti | Azaltma |
|---|---|---|
| VAE ince engeli kodlamıyor | probe harita IoU `< 0.30` | `beta` düşürme, latent `128`, kontrastif yardımcı kayıp; eşik düşürülmez |
| Posterior collapse | etkin boyut `< 32` | `beta` ısınma programı, KL serbest bit |
| Throughput düşüşü | `< 0.7 x v0` | privileged render'ı terminal faza sınırlama |
| Kapsama dışı poz | latent kırpma oranı yüksek, OOD davranış | dataset poz zarfını genişletme (REV-030 dersi) |
| `arrival` geçer, `hover_success` `%0` kalır | REV-043 duvarı | ADR-012 kararı kullanıcıya; bu dilim algı sorusunu izole eder |

## 24. Sıradaki implementasyon dilimi: iki modelli algı, YOLO dedektör + RL politikası (ADR-023)

**Kabul:** 26 Eylül 2026, kullanıcı kararı. Gerekçe ve sözleşme ADR-023'tedir;
bu bölüm yürütme sırasını, dosyaları ve çıkış kapılarını verir. Her dilim
`docs/PROGRESS_LOG.md`'ye AGENTS.md §5 formatında yazılır.

### Bağlayıcı kurallar

- Politikaya engel/hedef koordinatı, depth, segmentation veya bunların
  değerlerini taşıyan hiçbir kanal girmez; tek görsel girdi dedektör kutu
  listesidir (`vision-v4`). Waypoint ADR-019 gibi yaklaşık kalır.
- Dedektör yalnız onaylı manifestle yüklenir (ADR-014 deseni); `visual-vae`
  ve `vision-encoder` manifestleri reddedilir.
- `EbabilVisionNavigation-v0/v1`, `vision-v2/v3`, VAE zinciri ve
  `LEGACY_ROLLOUT_SHA256` dondurulur; testleri her dilimde geçmelidir.
- Kapı eşikleri ölçüm sonucuna göre **düşürülmez**; geçilmeyen kapı yeni
  bulgu (`REV-0xx`) açar ve karar kullanıcıya gider.
- Her eğitim sonucu raporlanmadan önce en az bir episode replay olarak
  izlenir.
- Ultralytics bağımlılığı `pyproject.toml`'da isteğe bağlı `detector`
  grubunda sabitlenir; çekirdek testler bu grup kurulu olmadan da geçer
  (dedektör testleri kurulu değilse atlanır ve bunu açıkça raporlar).

### Dilim A — Karar ve plan (belgeler)

**Durum (26 Eylül 2026): tamamlandı.** ADR-023, bu bölüm, literatür §2.17,
`PROJECT_DESIGN.md` güncellemesi, REV-048 yeniden çerçevesi, README ve
ilerleme kaydı. Kod değişikliği yok.

### Dilim B — Kutu etiketleri, YOLO eğitimi ve onay zinciri

**Durum (29 Eylül 2026): tamamlandı, kullanıcı kararlarıyla.** Zemin yansıması
kapatıldı, dataset yeniden üretildi, YOLO11n tek seed (kullanıcı kararı)
geliştirme kapılarını geçti; audit elipsoit engellerde geçmedi (REV-050) ve
dedektör kullanıcı kararıyla eğitim şekilleriyle sınırlı onaylandı
(`runs/detector_audit_v1_20260929/approval_manifest.json`). ADR-023 Revizyon 1.

İşler:

- `src/ebabil/perception/detections.py`: `Detection` (sınıf, `xyxy` px,
  güven), `boxes_from_segmentation(segmentation, target_geom_ids,
  obstacle_geom_ids, *, min_side_px=3, min_area_px=16)` (geom başına
  görünür piksel kutusu; hedef geom'ları tek nesne), YOLO metin yazıcısı,
  `dataset.yaml` üretimi, bölüm ve görsel alan eşlemesi.
- `scripts/export_yolo_dataset.py <dataset_dir> <output_dir>`: mevcut NPZ
  dataset'inden görüntü + etiket dizinleri (`train`, `validation`,
  `development_ood`, `audit`); sızıntı denetimi ve provenance (SHA-256).
- `pyproject.toml`: `[project.optional-dependencies] detector =
  ["ultralytics==8.4.163"]`; kurulum `.venv/bin/pip install -e ".[detector]"`;
  çevrimdışı ayarlar (`settings sync=False`, font/asset indirmesi kapalı).
- `src/ebabil/perception/detector.py`: `DetectorConfig` (model `yolo11n`,
  `imgsz 320`, `conf`, `iou`, `max_det`), `train_detector` (seed'li, MPS/CPU),
  `load_detector_checkpoint`, `detect_rgb(rgb) -> tuple[Detection, ...]`,
  gecikme ölçümü. Ağırlık dosyası SHA-256 ile manifestlenir.
- `src/ebabil/perception/detector_quality.py`: ADR-023 §3 kapıları
  (engel recall/precision, hedef recall `<= 5 m` yer menzili, hedef
  precision, hedefsiz karede sahte hedef oranı, gecikme; mAP bilgi),
  `run_detector_quality` (3 seed), `run_detector_audit` (tek kullanımlık
  `audit`), `approval_manifest.json` (`kind = "yolo-detector"`),
  `load_approved_detector` (fail-closed).
- Config: `src/ebabil/configs/training/detector.toml` (model, epoch `60`,
  batch, seed'ler `20260911-13`, kapı eşikleri).
- Scriptler: `scripts/train_detector.py`, `scripts/run_detector_quality.py`,
  `scripts/run_detector_audit.py`.
- Testler `tests/test_detections.py` (sentetik segmentation -> kutu,
  örtülme, küçük kutu filtresi, çeldirici etiketlenmez, YOLO formatı,
  bölüm sızıntısı), `tests/test_detector.py` (ultralytics kuruluysa: yükleme,
  `detect_rgb` sözleşmesi, manifest fail-closed; değilse atlanır).

Teslimatlar: `runs/yolo_dataset_d1_<tarih>`, `runs/detector_quality_v1_<tarih>`
(3 seed), `runs/detector_audit_v1_<tarih>/approval_manifest.json`.

Doğrulama kapısı: ADR-023 doğrulama 1-3; bütün mevcut testler geçer;
`ruff`, `mypy` temiz.

Tahmini maliyet: 1.5 iş günü; eğitim seed başına `~30-60 dk` (M3, MPS).

### Dilim C — `EbabilVisionNavigation-v2` ve `vision-v4`

**Durum (29 Eylül 2026): tamamlandı.** 20 test (sahte dedektörle; menzil
tahmini MuJoCo projeksiyonuyla çapraz doğrulandı), onaylı dedektörle gerçek
renderer smoke'u temiz, throughput `0.54 x v0` (tanı render'ı kapalı; açıkken
`0.49 x`), PPO kuru koşusu 4 ortamla `~100` adım/s. ADR-023 Revizyon 1.

İşler:

- `src/ebabil/envs/detection_vision.py`: `EbabilDetectionVisionNavigationEnv`
  (`EbabilVisionNavigationEnv`'in algı kancaları üzerinden, Dilim 2 deseni),
  `DETECTION_VISION_OBSERVATION_VERSION = "vision-v4"`,
  `EbabilVisionNavigation-v2`; `DetectionSlotEncoder` (9 yuva x 8 alan,
  sıralama, boş yuva, `range_norm`/`range_valid`), zemin düzlemi menzil
  tahmini (`camera.toml` pitch + `fc_state` attitude/irtifa),
  `DetectorTargetObserver` (hover kapısı: `conf >= 0.50`, alan `>= 16 px^2`),
  privileged gözlemci yalnız `info` (uyum metrikleri).
- `vision_config.py`: `[detections]` (`obstacle_slots 8`, `range_max_m 12`,
  `target_conf 0.50`, `k 1`); `vision_navigation_v4.toml` (reward/hover
  `v0` ile birebir).
- `training/config.py`, `env_factory.py`, `runner.py`: `kind =
  "detection_vision"`; `ppo_detector_d1a.toml` (PPO ayarları `d1a` ile
  birebir, manifest `yolo-detector`).
- `scripts/smoke_detection_vision_env.py`, `evaluate_vision_policy.py` ve
  `record_vision_replay.py` `--env-version v2`; replay görünümünde kutuların
  çizimi (tanı).
- Testler `tests/test_detection_vision_env.py`: ADR-023 doğrulama 4-8;
  throughput ölçümü rastgele politikayla (`v0` ile aynı koşul).

Doğrulama kapısı: bütün testler; `v0/v1` testleri ve legacy parmak izi
değişmedi; throughput `>= 0.5 x v0` (aksi halde ADR-023 §8 yedek yolu
Revizyon olarak açılır ve ölçülür).

Tahmini maliyet: 1.5 iş günü.

### Dilim D — D1a eğitimi: dedektör döngüde, 3 seed

**Durum (30 Eylül 2026):** seed `20260912` holonomik: arrival `%55`, çarpışma
`%35` (REV-051). ADR-024 koni kısıtı: `%70` / `%17.5`. ADR-025 terminal
yardımcısı ile aynı politika: hover başarısı `%77.5`, arrival `%77.5`, çarpışma
`%10` (projede ilk başarılı hover'lar). ADR-017 Revizyon 1 (erken tespit kaybı)
ve `1.2M` devam eğitimi ölçüldü. **Üç seed protokolü (30 Eylül):** ortalama
hover başarısı `%84.2`, arrival `%84.2` (kapı geçti), çarpışma `%13.3` (kapı
`<= %10` geçilmedi; seed'ler `%5 / %22.5 / %12.5`). Dilim E öncesi çarpışma
kapısı için karar kullanıcıda.

İşler: `ppo_detector_d1a.toml` ile `scripts/train_ppo.py`, üç seed
(`20260912-14`); `evaluate_vision_policy.py --env-version v2` held-out görsel
alanda 40 episode deterministik (+ stokastik rapor); ek raporlar (ADR-023
§10); en az bir replay izlenir.

Doğrulama kapısı (ADR-023 §10): `arrival >= %80`, `çarpışma <= %10`, 3 seed.

Tahmini maliyet: seed başına `~2 saat` eğitim (kuru koşuda `~100` adım/s ölçüldü); yarım iş günü koşu ve rapor.

### Dilim E — Ablation'lar ve yoğunluk merdiveni

ADR-023 §11 tablosu (R, K, N, V, S, DR) ve REV-039 merdiveni
(`10 -> 20 -> 40` engel) D1a kapısı geçildikten sonra; V satırı ADR-022
zincirini aynı ortam ve DR ile karşılaştırır.

Tahmini maliyet: 1 iş günü + koşu süreleri.
