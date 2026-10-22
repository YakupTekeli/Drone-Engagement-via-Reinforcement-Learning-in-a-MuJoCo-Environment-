# ADR-022: Etiketsiz Görsel Temsil (VAE) ve Observation Sözleşmesi `vision-v3`

**Durum:** Kabul edildi (kullanıcı onayı, 22 Eylül 2026); **Revizyon 1** aynı gün
kullanıcı onayıyla (kapı tanımı ve latent boyutu, aşağıda); **Revizyon 1b** 23 Eylül.
**Yerini bıraktı (26 Eylül 2026):** actor yolu ADR-023 ile dedektör + politikaya
geçti; VAE zinciri (`runs/vae_audit_v1_20260923`), `v1` ortamı ve `vision-v3`
karşılaştırma baseline'ı olarak korunur.  
**Tarih:** 22 Eylül 2026  
**İlişki:** ADR-011'in "encoder önce simülasyon etiketleriyle eğitilebilir"
maddesini **actor** için geçersiz kılar. ADR-014 approval zinciri desenini
korur, kapıları değiştirir. ADR-015/ADR-021 observation sözleşmesini
değiştirir; `vision-v2` ve `vision-encoder-v11` legacy tanı baseline'ı olur.
ADR-017 durum makinesinin görsel kapısının **kaynağını** değiştirir.
ADR-019 waypoint ve ADR-020 action sözleşmeleri değişmez. REV-048'i açar ve
kademeleriyle kapatır.

## Bağlam

### Kullanıcının itirazı ve ölçülen gerçek

Kullanıcı 22 Eylül 2026'da mevcut yapıyı "drone engellerin konumunu biliyor"
diyerek reddetti ve engel konumu bilgisi almayan, CNN ile öğrenen bir
kaçınma politikası istedi.

Kod ve testler denetlendi. Actor observation'ında (`vision-v2`) engel dünya
koordinatı, depth, raycast veya maske **yoktur**; sızıntı testleri bunu
zorlar. Fakat "bilme" üç kapıdan giriyor:

1. **Encoder simülatörün cevabını kopyalıyor.** `TinyVisionEncoder`,
   MuJoCo depth ve segmentation tamponlarından türetilen `obstacle_mask`,
   `free_space`, `target_mask` ve `target_center_xy` etiketleriyle supervised
   eğitiliyor, sonra donduruluyor. PPO onu güncellemiyor; çarpışmadan hiçbir
   şey öğrenmiyor. 12 Eylül 2026 kaydı bunu "tanınmıyor, söylendi" diye
   yazmıştı; gerçek araçta bu etiketleri üreten piksel bazlı depth ve
   segmentation yoktur.
2. **Ödül ayrıcalıklı bilgi kullanıyor.** `proximity` bileşeni 16 ışınlı
   rangefinder'ın en kısa mesafesinden, `target_progress` ve hover koşulları
   gerçek hedef konumundan hesaplanır. ADR-012 buna izin verir ve deploy'da
   ödül bulunmadığı için actor sızıntısı değildir.
3. **Araç kendi konumunu hatasız bilir** (ADR-019, bilinçli kapsam kararı).

Kullanıcının reddettiği mekanizma birincisidir.

### Etiketli algı düzeldi, kaçınma öğrenilmedi

REV-037 ve REV-044 encoder'ı D1 ortamında yeniden eğitti; rollout içi
`free_space` hatası `3.07 -> 1.02 m`'ye indi, engel haritası IoU `0.66`
ölçüldü. Buna rağmen engelli bütün koşular `%0` başarı verdi:

| Koşu | Engel | Başarı | Not |
|---|---:|---:|---|
| `d1-v11-seed20260912` | 40 | `%0` | çarpışma `%47-75` |
| `ladder4-d1a-seed20260912` | 5 | `%0` | REV-043: beş hover koşulu birlikte hiç sağlanmadı |
| `p40-d1a-seed20260912` | 5 | `%0` | 40 derece kamera zinciri |
| `rev046c-d1a-seed20260912` | 5 | `%0` | REV-046 sonrası; ortalama ödül negatif |

Algı doğruluğu bağlayıcı kısıt değildi (REV-037, REV-044). Kalan iki duvar
algı yönteminden bağımsızdır: keşif (REV-039) ve terminal hover ölçütünün
kesişimi (REV-043).

### Kullanıcının derin araştırma raporu

22 Eylül 2026'da kullanıcının sağladığı literatür raporu proje durumuyla
satır satır eşleştirildi (bkz. `docs/LITERATURE_REVIEW.md` §2.16). Raporun
projeye taşınan üç önerisi: (a) saf uçtan uca DRL yerine önce VAE ile
etiketsiz temsil öğrenimi, (b) `k = 3-4` kare yığınlama, (c) sıfırlama anında
görsel domain randomization. Rapor ayrıca hız komutu + alt seviye kontrolcü
ayrımını, gövde eksenli hızı ve sektör tabanlı serbest alanı önerir; bunlar
projede zaten vardır.

### Kısıt: örnek verimliliği

Vision ortamı `~135-140` adım/s ölçülmüştür (kamera render + encoder), yani
`600k` adım `~70 dk`. Piksellerden doğrudan PPO tipik olarak milyonlarca adım
ister; Apple M3 / 16 GB üzerinde bu gün mertebesine çıkar. Literatür (§2.9,
§2.16) de saf pixel-to-motor'u ilk tercih yapmaz.

## Karar

### 1. Actor algısı etiketsiz eğitilir

Actor observation'ına giren hiçbir görsel kanal; MuJoCo depth, segmentation,
raycast, `obstacle_mask`, `target_mask`, `free_space` veya bunlardan türetilen
bir etiketle eğitilmiş ağdan gelemez. Bu etiketler yalnız üç yerde
kullanılabilir: kalite **ölçümü** (probe), privileged reward/termination ve
tanı. `vision-encoder-v11` zinciri actor için legacy'dir; tanı, karşılaştırma
ve klasik baseline'da (madde 8) kullanılmaya devam eder.

### 2. Birinci kademe: donmuş VAE kodlayıcı

- Mimari: `96 x 72` RGB girdi (mevcut encoder bütçesi), konvolüsyonel
  kodlayıcı, `latent_size = 128` ortalama/log-varyans (Revizyon 1; özgün
  metin `64` diyordu), ayna kod çözücü.
  Kayıp `yeniden üretim + beta * KL`; `beta` config'tedir, başlangıç `1.0`.
- Ön işleme mevcut public `preprocess_actor_rgb` yoludur; adaptive exposure
  normalizasyonu etiketsiz olduğu için korunur. ADR-014'ün `%50` grayscale
  augmentation'ı korunur.
- Eğitim verisi yalnız `vision-dataset-v4` şemasındaki **actor** arşivleridir
  (`actor/<split>/*.npz`, anahtarlar `rgb`, `sample_id`, `timestamp_s`).
  İlk aday `runs/vision_dataset_d1_p40_20260914` (6528 kare, 40 derece pitch).
  Trainer privileged arşivleri **hiç açmaz**; bu bir testle zorlanır.
- Kodlayıcı PPO'da `eval` modunda ve `requires_grad=False` çalışır; kod
  çözücü runtime'a girmez.

### 3. Kalite ve onay zinciri

ADR-014 deseni aynen korunur: dataset -> quality (3 seed) -> audit ->
`approval_manifest.json`, SHA-256 zinciri, fail-closed yükleme
(`load_approved_visual_vae`). Değişen şey kapılardır:

| Kapı | Eşik | Kaynak |
|---|---:|---|
| Etkin latent boyutu (posterior collapse) | `>= 32` | yeni |
| Doğrusal probe `free_space` MAE | `<= 0.30` | ADR-014 |
| Doğrusal probe hedef görünürlüğü AUROC / AP | `>= 0.80 / 0.85` | Revizyon 1 |
| Doğrusal probe engel haritası en iyi eşikte IoU | `>= 0.45` | Revizyon 1 |
| Sabit `0.5` eşiğinde hedef P/R ve harita IoU | raporlanır, kapı değil | Revizyon 1 ve 1b |
| Yeniden üretim hatası | raporlanır, ilk ölçümden sonra eşiklenir | yeni |

Probe'lar donmuş `z` üzerinde privileged etiketlerle eğitilen tek katmanlı
modellerdir; yalnız **ölçüm** içindir, ağırlıkları hiçbir yere gönderilmez ve
kodlayıcıya gradyan vermez. Anlamı: etiketsiz kodlayıcı, etiketli zincirin
kapılarının istediği engel/hedef bilgisini doğrusal olarak çözülebilir biçimde
taşımak zorundadır. Kapı geçilmezse eşik düşürülmez (REV-021 ilkesi);
kaldıraçlar `beta`, latent boyutu ve kontrastif yardımcı kayıptır (§2.16,
CURL).

### 4. Observation sözleşmesi `vision-v3`, environment `EbabilVisionNavigation-v1`

| Anahtar | Shape/aralık | İçerik |
|---|---|---|
| `vision_latent` | `(k * 128,)`, `[-1, 1]`, `k = 4` | son `k` policy adımının VAE ortalama vektörü, `latent_scale` ile kırpılmış (Revizyon 1: `64 -> 128`) |
| `fc_state` | `(10,)` | değişmez (ADR-015) |
| `waypoint` | `(4,)` | değişmez (ADR-019, inanılan hedef) |
| `timing` | `(2,)` | değişmez |

Çıkarılanlar: `obstacle_map` ve `vision_aux` (`target_score`, `center_xy`,
`free_space`). Toplam boyut `194 -> 528` (Revizyon 1; latent 64 ile `272`
olurdu).

- Yığınlama **latent düzeyinde** yapılır: kodlayıcı kare başına bağımsız
  çalıştığı için piksel yığınlamayla eşdeğerdir ve bellek `4x` küçüktür.
- Reset'te yığın ilk latent ile **kopyalanarak** doldurulur; sıfırla
  doldurmak sahte bir "hareket" sinyali üretirdi.
- ADR-015 zamanlaması değişmez: kamera `30 Hz`, policy `20 Hz`, sample-and-hold.
- `EbabilVisionNavigation-v0` / `vision-v2` dondurulur ve testleri korunur.

### 5. Hover durum makinesinin görsel kapısı privileged görünürlüğe taşınır

ADR-017 kapıları (`acquired`, `target_lost`, `hold`) encoder'ın
`target_score`'una dayanıyordu. Etiketsiz kodlayıcının böyle bir çıktısı
yoktur ve olmamalıdır. Kapı, ortamın sahip olduğu **ayrı** bir privileged
kamera nesnesinin segmentation maskesinden hesaplanan hedef piksel sayısına
(`>= minimum_target_pixels = 8`, dataset sözleşmesiyle aynı) taşınır.
`acquire_seconds` ve `grace_seconds` zaman filtreleri aynen kalır; skor
histerezisi (`0.60/0.40`) boolean görünürlükle anlamsızlaştığı için `v1`
ortamında kullanılmaz. `target_alignment` için `target_center_x` aynı maskenin
ağırlık merkezinden türetilir (`derive_supervision` ile aynı kural).

Gerekçe: ADR-012/017 başarıyı zaten privileged gerçekle doğrular; görsel
kapı bir termination/reward mekanizmasıdır, actor girdisi değildir. Actor,
hedefin görünür olup olmadığını `z`'den kendisi çıkarmak zorundadır. Privileged
kamera nesnesi actor yolundan erişilemez; `ActorCameraFrame` yalnız
`("rgb", "timestamp_s")` taşımaya devam eder.

### 6. Görsel domain randomization ortama taşınır

- Reset'te `SceneVisualParameters` (ışık, doku tonu, engel ve hedef rengi)
  scene seed'inden türetilen seed ile örneklenir ve renderer kurulmadan
  modele uygulanır; her yakalanan kareye `FrameVisualParameters` (pozlama,
  gamma, gürültü, vinyet, bulanıklık) `(scene seed, kare sayacı)` seed'iyle
  uygulanır. Mevcut `visual_randomization.py` yeniden kullanılır.
- Eğitim alanı dataset'in `train_broad_randomized` alanıdır; evaluation
  eğitim alanı ve held-out alan (`heldout_dim_cool`) için **ayrı** raporlanır.
- RL sahnelerinde engel geometrisi kutu/silindir/capsule/sphere arasında
  örneklenir (dataset'in train kümesiyle aynı); audit'e özel ellipsoid
  eğitime girmez. Sahne config'inde açık bir alan olarak tutulur.
- Aynı seed aynı bayt dizisini üretmek zorundadır (rollout parmak izi testi).

Bugünkü ortamda engel rengi sabit `(0.55, 0.22, 0.12)` ve geometri yalnız
kutudur; kodlayıcı randomize dağılımda eğitilip politikaya tek bir dilim
gösteriliyordu (REV-048).

### 7. Action ve reward değişmez

`vision-action-v2` (ADR-020) ve ADR-017 reward bileşenleri aynen kalır.
Privileged `proximity` şekillendirmesi **açık kalır**: deploy'da ödül yoktur,
şekillendirme actor'a sızmaz ve birinci kademenin bağlayıcı problemi keşiftir
(REV-039); yoğun sinyali kaldırmak bunu ağırlaştırırdı. Raporun
`r_clearance` terimi de aynı bileşendir. `proximity_weight = 0` bir ablation
eksenidir. Not: 22 Eylül sabahında önerilen "varsayılan kapalı" seçeneği bu
gerekçeyle geri alındı.

### 8. Kademeler ve ablation'lar

| Kademe | İçerik | Statü |
|---|---|---|
| 1 | Donmuş VAE + `vision-v3` + ortam içi DR, D1a (5 engel) | ana yol |
| 2 | Kodlayıcı PPO içinde çözülür, VAE ağırlıklarından başlar | ana yol, plan A9 |
| A | Etiketli `v11` encoder aynı ortamda | birincil bilimsel karşılaştırma: etiketin bedeli |
| B | Saf piksel PPO, sıfırdan | ablation |
| C | `k = 1` vs `k = 4`; latent `32` vs `64`; DR açık/kapalı; `proximity` açık/kapalı | ablation |
| D | DroNet tarzı reaktif baseline: `v = v_max * (1 - P_coll)`, `v11` haritasından | klasik baseline (tasarım §12) |

### 9. Ertelenenler

- **Kamera pitch jitter'ı.** Hold halkası alt sınırı `36.8 derece`, mevcut
  `40 derece`; sapma `+-3 derece` ile sınırlı olmalı ve hover geometrisi
  yeniden doğrulanmadan açılmamalı. Birinci kademede yoktur.
- **Dinamik randomization** (kütle, rotor zaman sabiti, Dryden rüzgâr):
  altyapı vardır, kullanıcı kararıyla (12 Eylül 2026) Faz 8'e ertelidir.
- **Optik akış / TTC:** hover'da tanımsız, karşıdaki engelde sıfıra yakın,
  40 derece aşağı bakan kamerada zemin akışı baskın; yalnız tanı/baseline adayı.
- **FastDepth tarzı yoğun derinlik:** madde 1 ile çelişir; ancak asimetrik
  actor-critic ablation'ında **critic** girdisi olabilir.

### 10. Birinci kademe evaluation protokolü

`hover_success` (ADR-012) değişmez ve raporlanır. Ek olarak `arrival_rate`
raporlanır: episode boyunca gerçek hover anchor'a yatay mesafe
`<= approach_radius_m` olmuş **ve** terminasyon fiziksel failure değil.
Birinci kademe kabul ölçütü, D1a (5 engel), deterministik, 3 seed:
`arrival >= %80`, `çarpışma <= %10`. ADR-018 D1 kapısı (`>= %85` başarı,
`<= %5` çarpışma) değişmez ve iddia edilmez.

## Revizyon 1 (22 Eylül 2026, kullanıcı onayı)

**Ne değişti.** (a) Hedef görünürlük kapısı sabit `0.5` eşiğindeki
recall/precision (`0.85 / 0.78`) yerine eşikten bağımsız çözülebilirlik
oldu: doğrusal probe AUROC `>= 0.80` ve average precision `>= 0.85`. (b)
Engel haritası için `0.5` eşiğindeki IoU `>= 0.30` korundu, en iyi eşikte
IoU `>= 0.45` eklendi. (c) `latent_size` `64 -> 128`; `vision-v3`
`vision_latent` kanalı `512`, toplam observation `528`. Sabit eşik
precision/recall raporlanmaya devam eder. Onaylı zincir yükleyicisi yeni
kapıların zincirde bulunmasını ister; eski (sabit eşikli) zincir fail-closed
reddedilir.

**Neden.** REV-049 ölçümü: varsayılan config üç seed'de hedef recall
`0.60 / 0.69` verdi; §3'te sayılan bütün etiketsiz kaldıraçlar (`beta`,
latent, çözünürlük, fotometrik değişmezlik, kontrastif) 11 varyantta
recall'u `0.61`'in üstüne taşımadı. Eşikten bağımsız ölçüm, sabit eşiğin
probe'un **kalibrasyonunu** ölçtüğünü gösterdi: engel haritası en iyi eşikte
`0.50-0.56` IoU taşıyor (`0.5`'te `0.27-0.40`), hedef için AUROC `0.81` ve
en iyi F1'de recall `0.86-0.91`. Politika doğrusal değildir ve kendi eşiğini
öğrenir; latent'in bilgi taşıyıp taşımadığı kapının ölçmesi gereken şeydir.
Kapasite (`64 -> 128`) hedef AUROC'unu `0.78 -> 0.81`, serbest alan hatasını
`0.25 -> 0.21` iyileştiren tek kaldıraç olduğu için latent büyütüldü.

**Bedel ve sınır.** Etiketsiz latent hedefi etiketli encoder kadar keskin
ayırmaz (AUROC `0.81`, etiketli zincirde recall `0.92`/precision `0.95`).
Hedef bulma tamamen politikaya kalır; bu bedel kademe 1 kapısında (varış +
çarpışma) görünmez, hover aşamasında görünür. Hover aşamasında hedefin ayrı
bir işaret dedektörüne bırakılması (REV-049 seçenek 3) açık tutulur ve ayrı
karar gerektirir. Eşikten bağımsız kapı, sabit eşikli kapıdan **daha
zayıf** bir garantidir; bu bilinçli bir ölçüm sözleşmesi değişikliğidir,
"kapı avcılığı" değil: ölçülecek şey değişti, ölçüt gevşetilmedi.

## Doğrulama kapısı

1. VAE trainer'ın privileged arşivleri hiç açmaması (`np.load` casusu ve
   privileged dizini kaldırılmış dataset ile eğitim).
2. `vision-v3` anahtar/shape/dtype/sınır testi; `obstacle_map` ve
   `vision_aux` yok; yasaklı token listesi.
3. Latent yığını: reset'te kopyalama, kaydırma sırası, kodlayıcı çıktısıyla
   `atol 1e-6` eşitlik.
4. Privileged kamera nesnesinin actor yolundan erişilemezliği;
   `ActorCameraFrame` alanlarının değişmemesi.
5. DR determinizmi: aynı seed aynı RGB baytları, farklı seed farklı
   parametre; held-out alan anahtarı.
6. Boolean görünürlükle durum makinesi geçiş, grace ve abort testleri.
7. Onay zinciri: 3 seed quality + audit + fail-closed manifest (probe kapıları).
8. `check_env`, uzun random rollout, kısa PPO smoke, ayrı deterministik
   evaluation, throughput ölçümü.
9. `LEGACY_ROLLOUT_SHA256` ve `vision-v2` testlerinin değişmemesi.

## Ölçülen sonuçlar

**Dilim 1, 22 Eylül 2026 (REV-049), özgün kapılarla.** Varsayılan config
(`beta 1`, latent 64, 72x96, 45 epoch), üç seed, `runs/vae_quality_v1_20260922`:

```text
etkin latent boyutu        : 64 / 64                  geçti
probe free_space MAE       : 0.248 / 0.237            geçti  (<= 0.30)
probe target precision     : 0.862 / 0.832            geçti  (>= 0.78)
probe target recall        : 0.597 / 0.685            GEÇMEDİ (>= 0.85)
probe engel haritası IoU   : 0.268 / 0.333            validation GEÇMEDİ (>= 0.30)
quality_passed = False, audit koşulmadı
```

§3'te sayılan bütün kaldıraçlar ölçüldü (beta, latent, çözünürlük,
fotometrik değişmezlik, kontrastif yardımcı kayıp; 11 varyant). Hedef
recall hiçbirinde `0.61`'i geçmedi; `beta 1 + latent 128` diğer beş kapıyı
geçti ve hedef için AUROC `0.81 / 0.82` taşıyor. Eşik düşürülmedi; kapı
tanımı Revizyon 1 ile değiştirildi (kullanıcı onayı).

**Revizyon 1 kapılarıyla ikinci kalite koşusu** (`runs/vae_quality_v2_20260922`,
`beta 1`, latent 128, üç seed): üç seed'in hepsi eşikten bağımsız kapıları geçti
(validation / development_ood, min-max): AUROC `0.805-0.809 / 0.817-0.822`,
AP `0.88 / 0.90`, en iyi eşikte harita IoU `0.51-0.54 / 0.50-0.51`, serbest
alan MAE `0.21-0.23`, etkin boyut `128/128`. Tek takılan kapı, korunan sabit
`0.5` eşiğindeki harita IoU (ADR-021 mirası): seed `20260912` `0.251`, seed
`20260913` `0.275`, kapı `0.30`; seed `20260911` `0.332` ile geçti.
`quality_passed = False`, audit koşulmadı.

**Revizyon 1b (23 Eylül 2026, kullanıcı onayı).** Sabit `0.5` eşiğindeki
harita IoU kapısı, hedef için kaldırılan sabit eşikli kapıyla aynı
kalibrasyon ölçüsü olduğu için kapı olmaktan çıkarıldı; değeri raporlanmaya
devam eder. Zorunlu kapılar: etkin latent boyutu, probe `free_space` MAE,
hedef AUROC ve AP, en iyi eşikte harita IoU, grayscale `free_space` MAE.

**Revizyon 1b kapılarıyla üçüncü kalite koşusu** (`runs/vae_quality_v3_20260923`):
üç seed'in hepsi yedi kapıyı iki bölümde geçti; değerler v2 ile birebir aynı
(deterministik eğitim): AUROC `0.805-0.809 / 0.817-0.822`, AP `0.88 / 0.90`,
en iyi eşikte harita IoU `0.51-0.54 / 0.50-0.51`, serbest alan MAE
`0.21-0.23`, etkin boyut `128/128`; bilgi alanı olarak harita IoU@0.5
`0.25-0.33 / 0.34-0.40`. `quality_passed = True`, aday seed `20260911`.
Tek kullanımlık held-out audit (`runs/vae_audit_v1_20260923`, 768 kare):
AUROC `0.809`, AP `0.892`, en iyi eşikte harita IoU `0.569`, serbest alan
MAE `0.203`, grayscale serbest alan MAE `0.233`; `audit_passed = True`.
Onaylı kodlayıcı `runs/vae_quality_v3_20260923/seed_20260911/visual_vae_v1.pt`
(sha256 `44bde1a8...`), manifest `runs/vae_audit_v1_20260923/approval_manifest.json`.

**Dilim 2 (23 Eylül 2026).** `EbabilVisionNavigation-v1` / `vision-v3`
kodlandı (§4-§6) ve onaylı manifestle gerçek renderer'da smoke edildi
(`runs/latent_vision_smoke_20260923`). Rastgele politikayla `v1 / v0`
throughput oranı `0.76` (DR açık) ve `0.84` (DR kapalı); kapı `>= 0.7` geçti.
Fotometrik etkiler kare başına `1.2 ms` (torch arama tablosu + seed'li
gürültü bankası), segmentation gözlemcisi `2.6 ms`; latent kırpma oranı ilk
ölçümlerde `0.0`.

## Sonuçlar ve sınırlar

- VAE latent'i piksel alanı büyük yüzeyleri önceler; ince engeller zayıf
  temsil edilebilir. Probe kapısı bunu eğitime başlamadan ölçer.
- Donmuş kodlayıcının poz dağılımı politikanın uçuş zarfını kapsamalıdır;
  ilk dataset `1.20-2.90 m` irtifa, `2.5 m` yanal offset ve `%20` hold pozu
  içerir. Kapsama dışı davranış REV-030 sınıfı bir hata üretir.
- Privileged görünürlük kapısı yalnız simülasyon eğitimi içindir; gerçek
  araçta terminal durum makinesi bir hedef dedektörü ister (ADR-011 tezgâh
  kapısı).
- Kare başına ek segmentation render'ı ve fotometrik etkiler throughput'u
  `v0`'ın `0.76`'sına düşürür (ölçüldü, kapı `>= 0.7`); `subproc` vektör
  ortamında süreç başına iki GL bağlamı açılır.
- Observation `528` boyuttur (Revizyon 1, latent `128 x 4`); `latent_scale`
  kırpma oranı `info["latent_clip_fraction"]` ile raporlanır.
- Hiçbir maddede gerçek uçuş güvenliği iddiası yoktur.
