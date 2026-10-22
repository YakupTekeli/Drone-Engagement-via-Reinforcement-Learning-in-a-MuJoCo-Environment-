# ADR-023: İki Modelli Algı (YOLO Dedektör + RL Politikası) ve Observation Sözleşmesi `vision-v4`

**Durum:** Kabul edildi (kullanıcı kararı, 26 Eylül 2026). Dedektör
kütüphanesi ve kapı değerleri aşağıda "kullanıcı itirazına açık varsayım"
olarak işaretlenmiştir; itiraz gelmezse bağlayıcıdır.  
**Tarih:** 26 Eylül 2026  
**İlişki:** ADR-022'nin "actor'a giren görsel kanal etiketle eğitilmiş ağdan
gelemez" kuralını **actor yolu için geçersiz kılar**; ADR-022'nin VAE zinciri,
`EbabilVisionNavigation-v1` ve `vision-v3` karşılaştırma baseline'ı olarak
korunur, silinmez. ADR-014 approval zinciri desenini (dataset -> çoklu seed
kalite -> tek kullanımlık audit -> SHA-256 manifest, fail-closed yükleyici)
dedektör için yeniden kullanır. ADR-017 hover durum makinesinin görsel
kapısının kaynağını dedektörün hedef kutusuna taşır. ADR-019 waypoint
(yaklaşık hedef) ve ADR-020 action sözleşmeleri **değişmez**. ADR-011'in
"gerçek araçta terminal durum makinesi bir hedef dedektörü ister" notunu
kapatır. REV-048'in etiket bağımlılığı bulgusunu yeniden çerçeveler (§9).

## Bağlam

### Kullanıcının kararı

Kullanıcı 26 Eylül 2026'da Dilim 3 eğitimini (`latent_d1a_seed20260912`,
`528k / 600k` adım) durdurup mimariyi değiştirdi. Kararın kendi ifadesiyle
özü:

> Drone hedefi ve engelleri kamerasından aldığı görüntü ile, YOLO modeli
> kullanarak tanıyacak. Bu sırada drone hedefin tam konumunu bilmeyecek,
> yalnızca yaklaşık konumuna hakim olacak. Engellerin etrafında çıkacak olan
> bounding box'lardan kaçınarak hedefe gitmeye çalışacak. İki farklı AI
> yapımız olacak: YOLO ile görüntü tespiti, ardından bu çıktı RL modeline
> verilecek.

Bu karar, 22 Eylül'deki itirazın ("drone engellerin konumunu biliyor")
gereğini farklı yoldan sağlar: politika engel **koordinatlarını** görmez,
yalnız görüntüdeki kutuları görür. Etiket kullanımı yasak değildir; etiketler
dedektörü eğitir, politikaya sızmaz.

### ADR-022 yolunun bıraktığı ölçümler

- Etiketsiz VAE zinciri onaylandı (`runs/vae_audit_v1_20260923`), fakat
  latent'in hedefi ayırma gücü sınırlı kaldı (AUROC `0.81`; sabit eşikte
  recall `0.57-0.69`). Hedef bulma tamamen politikaya kalıyordu.
- `v1` ortamında `528k` adımlık kesilen eğitimde rollout ödülü `-70`'ten
  `-46`'ya çıktı, başarı `0`, çarpışma payı `%50 -> %43`, hedef kaybı abort'u
  `%19 -> %28`. Politika hedef bölgesine ulaşmayı öğreniyor, terminal fazda
  hedefi tutamıyordu (REV-043 duvarı).
- Kutu tabanlı algı bu iki noktayı doğrudan hedefler: hedef, dedektörün açık
  bir sınıfıdır; engeller görüntü koordinatlarıyla ve menzil ipucuyla gelir.

### Gerçek araç uyumu

Hedef araç IMX477 monoküler kamera + Jetson Orin NX'tir (ADR-011). Bir
YOLO dedektörü bu donanımın olağan iş yüküdür (TensorRT export). Politika ile
dedektör arasındaki arayüz **sembolik** (kutu listesi) olduğu için dedektör
gerçek görüntülerle yeniden eğitildiğinde politika sözleşmesi değişmez; bu,
piksel tabanlı politikalara göre sim-to-real aktarımının ana avantajıdır
(Swift, §2.8: algı sistemi düşük boyutlu temsile indirger, politika o temsil
üzerinde öğrenir).

## Karar

### 1. İki model, tek yönlü veri akışı

```text
kamera RGB (320 x 240, 30 Hz)
   -> YOLO dedektör (sınıflar: obstacle, target)        [model 1, denetimli]
   -> kutu listesi -> vision-v4 yuva vektörü (72)
   -> PPO politikası (+ fc_state 10, timing 2, waypoint 4) [model 2, RL]
   -> holonomik hız komutu (ADR-020)
```

- Politikaya engel koordinatı, hedefin gerçek konumu, depth, segmentation
  veya bunlardan türetilmiş hiçbir kanal girmez. Waypoint ADR-019'daki gibi
  **inanılan** hedeften türetilir (`2 m`'ye kadar sapma) ve değişmez.
- Dedektör, simülatörün otomatik kutu etiketleriyle eğitilir (§2). Actor
  yolunda tek öğrenilmiş görsel bileşen dedektördür; politika kutu
  vektörünü ham girdi olarak alır.
- Dedektör eğitim döngüsünün **içinde** çalışır (kullanıcı kararı): RL
  ortamı her policy adımında yakalanan kareyi dedektöre verir. Hız için
  yedek yol §8'dedir ve yalnız ölçülen throughput kapısı geçilmezse açılır.

### 2. Dedektör verisi: segmentation'dan otomatik kutu etiketi

Mevcut dataset üreteci (`perception/dataset.py`) her karede ham
`segmentation (H, W, 2)` (geom kimliği, nesne türü) kaydeder. Kutu etiketi
buradan türetilir; elle etiketleme yoktur:

- Her `obstacle_*` geom'u için görünür piksellerinin eksen-hizalı kutusu;
  sınıf `obstacle`. Örtülme doğal olarak yalnız görünür kısmı kutular.
- `target_area*` geom'ları tek nesne olarak birleştirilir; sınıf `target`.
- Görsel çeldiriciler (`_sample_visual_distractors`) etiketlenmez; negatif
  örnektir.
- Kenarı `< 3 px` veya alanı `< 16 px` olan kutular etiketten düşer ve
  metrikte yok sayılır.
- Format: Ultralytics YOLO metin formatı (`sınıf cx cy w h`, `[0, 1]`
  normalize) + `dataset.yaml`. Bölümler dataset'in mevcut bölümleridir
  (`train`, `validation`, `development_ood`, `audit`); görsel alanlar ve
  sızıntı denetimi (`validate_dataset_leakage`) aynen kullanılır.
- İlk kaynak `runs/vision_dataset_d1_p40nr_20260926` (6528 kare, pitch `40`,
  `static_scene_d1` sahneleri, zemin yansıması kapalı; §6). Sahne ve poz
  seed'leri `runs/vision_dataset_d1_p40_20260914` ile aynıdır. Kamera geometrisi değişirse dataset ve
  onay zinciri yeniden kurulur (camera.toml notu).

### 3. Dedektör modeli, kalite kapıları ve onay zinciri

**Model (kullanıcı itirazına açık varsayım):** Ultralytics `YOLO11n`,
`ultralytics==8.4.163`, giriş `imgsz 320`, `60` epoch, üç seed
(`20260911-13`). Lisans **AGPL-3.0**; akademik kullanım için uygundur, ürün
dağıtımında lisans yükümlülüğü doğurur ve `docs/PROJECT_DESIGN.md`'de not
edilir. Apache lisanslı alternatifler (YOLOX, RT-DETR) §2.17'de listelenir;
kullanıcı isterse aynı kutu arayüzüyle değiştirilir.

**Kapılar** (validation ve development_ood bölümleri, held-out görsel alan,
üç seed'in hepsi geçmeli; eşikler ölçüm sonucuna göre **düşürülmez**,
geçilmeyen kapı REV bulgusu açar):

| Kapı | Eşik | Gerekçe |
|---|---|---|
| Engel recall, alan `>= 400 px^2` | `>= 0.90` | Politikanın kaçınma kararı verdiği yakın/orta menzil kutuları |
| Engel precision, güven `>= 0.25` | `>= 0.90` | Sahte kutular gereksiz kaçınma manevrası üretir |
| Hedef recall, yer menzili `<= 5 m` | `>= 0.85` | Yaklaşma ve hold fazı bu menzilde geçer (kamera geometrisi: `h = 1.5 m`'de hedef `<= 7.5 m`'ye kadar görünür) |
| Hedef precision, güven `>= 0.50` | `>= 0.90` | Sahte hedef `acquired` kapısını yanlış tetikler |
| Hedefsiz karede sahte hedef oranı | `<= 0.05` | Arama fazı sağlamlığı |
| Gecikme (`320 x 240`, M3, tek kare) | `<= 15 ms` | Ortam throughput kapısı (§8) için bütçe |
| Bilgi (kapı değil) | mAP50 / mAP50-95 iki sınıf; grayscale kopyada aynı metrikler | Etiketli zincirle karşılaştırma |

**Onay zinciri:** `run_detector_quality` (3 seed, kapılar) ->
`run_detector_audit` (tek kullanımlık `audit` bölümü) ->
`approval_manifest.json` (`kind = "yolo-detector"`, checkpoint, dataset ve
rapor SHA-256'ları) -> `load_approved_detector` (fail-closed). RL ortamı
yalnız manifestle açılır; `visual-vae` veya `vision-encoder` manifestleri
reddedilir.

### 4. Observation sözleşmesi `vision-v4`, environment `EbabilVisionNavigation-v2`

| Anahtar | Shape/aralık | İçerik |
|---|---|---|
| `detections` | `(72,)`, `[0, 1]` | `9` yuva x `8` alan: `8` engel yuvası + `1` hedef yuvası |
| `fc_state` | `(10,)` | değişmez (ADR-015) |
| `waypoint` | `(4,)` | değişmez (ADR-019, inanılan hedef) |
| `timing` | `(2,)` | değişmez |

Toplam `88` boyut (`vision-v3`'te `528`).

Yuva alanları: `present, cx, cy, w, h, conf, range_norm, range_valid`.

- `cx, cy, w, h` görüntü boyutuna normalize kutu merkezi ve boyutu; `conf`
  dedektör güveni.
- `range_norm`: kutunun **alt kenarından** zemin düzlemi geometrisiyle
  tahmin edilen yer menzili, `R_max = 12 m` ile normalize ve kırpılmış.
  Kamera pitch'i (`40` derece, mount) ve araç attitude'u `fc_state`'ten,
  kamera yüksekliği `fc_state` irtifasından alınır; ikisi de gerçek araçta
  mevcuttur. Odak uzaklığı `f = 160 / tan(37.5 deg) = 208.5 px`; satır `v`
  için depresyon açısı `pitch + atan(v / f)`, menzil `h / tan(depresyon)`.
  Tek kamerada derinlik yoktur; bu tahmin engellerin zemine bastığı
  varsayımına dayanır ve bu varsayım sahne üreticisinde sağlanır (REV-035:
  engeller tavana kadar, zeminden başlar).
- `range_valid = 0`: kutunun alt kenarı görüntünün alt satırına değiyorsa
  (engel tabanı kadraj dışı) veya araç attitude'u geçerli tahmini
  engelliyorsa; bu durumda `range_norm = 1.0`.
- Engel yuvaları sıralıdır: geçerli menzili olanlar artan menzille, ardından
  geçersizler azalan alanla; eşitlikte `cx`. Boş yuva: bütün alanlar `0`,
  yalnız `range_norm = 1.0`.
- Hedef yuvası: en yüksek güvenli `target` kutusu.
- Kare yığını `k = 1`. Hareket bilgisi `fc_state` hızlarından gelir; yuvalar
  kareler arası eşlenmediği için yığın belirsizlik üretirdi. `k` config
  alanıdır ve ablation'dır (§11).
- ADR-015 zamanlaması değişmez: kamera `30 Hz`, policy `20 Hz`, sample-and-
  hold; dedektör policy adımı başına bir kez çalışır.
- `EbabilVisionNavigation-v0/v1`, `vision-v2/v3` dondurulur; testleri korunur.

### 5. Hover durum makinesinin görsel kapısı dedektörün hedef kutusuna taşınır

ADR-017 kapıları (`acquired`, `target_lost`, `hold`) `v1`'de privileged
segmentation'dan geliyordu (ADR-022 §5). `v2`'de kaynak dedektördür:
hedef görünür sayılır ancak ve ancak bir `target` kutusu `conf >= 0.50` ve
alan `>= 16 px^2` ise. `target_center_x` kutu merkezidir. `acquire_seconds`
ve `grace_seconds` zaman filtreleri aynen kalır.

Gerekçe: gerçek araçta bu kapının tek kaynağı dedektördür (ADR-011 tezgâh
notu). Dedektörün kaçırmaları artık terminal fazı gerçekçi biçimde etkiler;
bu bedel bilinçli kabul edilir. Privileged gözlemci (`PrivilegedTargetObserver`)
**yalnız tanı** için kalır: `info` alanları dedektör-gerçek uyumunu
(`target_visible_detector`, `target_visible_privileged`, engel kutusu sayısı
vs görünür engel sayısı) her adımda raporlar; reward ve termination bunları
okumaz.

### 6. Görsel domain randomization korunur

ADR-022 §6 aynen geçerlidir: sahne ve kare düzeyinde fotometrik
randomization hem dedektör verisinde hem RL ortamında; engel geometrisi
`box/cylinder/capsule`; held-out görsel alan evaluation'da. Dedektör
verisindeki çeldiriciler, renk kestirmesini engellemeye devam eder.

**Zemin yansıması kapalı (kullanıcı kararı, 26 Eylül 2026).** Etiket
kontrolünde zemin materyalinin `reflectance = 0.05` değerinin engellerin yarı
saydam yansımalarını çizdiği görüldü; bunlar geometri olmadığı için
etiketlenmiyor ve dedektör verisinde etiketsiz sahte engel görünümü
üretiyordu. `quadrotor.xml` zemin materyali `reflectance = 0` yapıldı; bu
ayar bütün ortamlarda (legacy dahil) render'ı etkiler, fiziği etkilemez.
Dedektör dataset'i bu ayarla yeniden üretilir
(`runs/vision_dataset_d1_p40nr_20260926`). ADR-022 VAE zinciri ve legacy
`vision-encoder-v11` zinciri yansımalı görüntülerle eğitilmişti; baseline
olarak koşulacaklarsa dataset ve onay zincirleri yeniden üretilmelidir.

### 7. Action ve reward değişmez

ADR-020 holonomik action ve ADR-017 reward bileşenleri aynen kalır.
Privileged `proximity` şekillendirmesi simülasyon eğitiminde kalır (ADR-022
§7 gerekçesi). `target_alignment` artık dedektör merkezini kullanır (§5).

### 8. Dedektör eğitim döngüsünün içinde; hız kapısı ve yedek yol

- Kabul: `v2` ortamı, dedektör dahil, rastgele politikayla `>= 0.5 x v0`
  throughput verir (ölçüm Dilim C). `v1` için bu kapı `0.7` idi; dedektör
  bütçesi (`<= 15 ms`) nedeniyle `0.5` konur ve bu, `600k` adımlık D1a
  eğitimini seed başına `~3-4 saate` çıkarır (kullanıcı bunu kabul etti).
- Kapı geçilmezse **yedek yol** Revizyon olarak açılır: eğitimde
  simülatörün gerçek kutularına dedektörün held-out'ta **ölçülmüş** hata
  modeli (menzile bağlı kaçırma oranı, kutu koordinat sapması, sahte kutu
  oranı) enjekte edilir; evaluation her zaman gerçek dedektörle yapılır ve
  ikisi arasındaki `arrival` farkı raporlanır. Bu yol ayrıca ablation olarak
  ölçülür (§11) fakat varsayılan değildir.

### 9. Etiketler ve REV-048'in yeniden çerçevelenmesi

REV-048 "actor algısı simülatör etiketlerini kopyalıyor" diyordu ve ADR-022
bunu etiketsiz temsil ile çözdü. Kullanıcının 26 Eylül kararıyla gereklilik
netleşti: yasak olan **engel/hedef koordinatlarının politikaya girmesidir**,
etiketle eğitilmiş bir dedektör değil. Dedektör etiketleri kutu düzeyinde ve
gerçek araçta da aynı biçimde (gerçek görüntü etiketleriyle) üretilebilir;
depth/segmentation gibi yalnız simülatörde var olan kanallar politikaya hâlâ
girmez. ADR-022'nin bağlayıcı kuralı bu ADR ile "actor'a giren görsel kanal
depth/segmentation/raycast **değerlerini** taşıyamaz" biçimine daralır.

### 10. Birinci kademe evaluation protokolü

ADR-022 §10 aynen: D1a (5 engel), deterministik, 3 seed, held-out görsel
alan, 40 episode, dedektör döngüde: `arrival >= %80`, `çarpışma <= %10`;
`hover_success` raporlanır, kapı değildir. Ek raporlar: dedektör-gerçek
görünürlük uyumu (kare başına), yuva doluluk dağılımı, `range_valid` oranı,
en yakın engel menzil tahmininin gerçek menzile ortalama mutlak hatası.
Her eğitim sonucu raporlanmadan önce en az bir episode replay olarak
izlenir (13 Eylül dersi).

### 11. Ablation'lar (Dilim E, kapı geçildikten sonra)

| Kod | Baseline | Varyant | Soru |
|---|---|---|---|
| R | `range_norm` açık | menzil alanları sıfır | zemin düzlemi menzil ipucunun katkısı |
| K | `k = 1` | `k = 2` (yuvalar kareler arası eşlenmeden) | zamansal bilginin katkısı |
| N | dedektör döngüde | ölçülmüş hata modeli (§8) | eğitim hızına karşı sadakat |
| V | `vision-v4` | `vision-v3` (ADR-022 VAE, aynı ortam ve DR) | sembolik kutu ile latent'in karşılaştırması |
| S | `8` engel yuvası | `4` / `16` | yuva sayısı |
| DR | açık | kapalı | görsel çeşitliliğin katkısı |
| H | terminal yardımcı açık (ADR-025) | kapalı | RL'in tek başına yerleşme yetkinliği |
| C | koni kısıtı açık (ADR-024) | holonomik | kör yöne uçuşun bedeli (REV-051'de ölçüldü) |

### 12. Ertelenenler

- Dedektörün gerçek IMX477 görüntüleriyle ince ayarı (tezgâh verisi yok).
- Dedektör çıktısını zaman içinde takip eden bir izleyici (tracker); önce
  `k = 1` ölçülür.
- Kamera geometrisi değişikliği; dataset ve onay zinciri yeniden kurulur.
- Hiçbir maddede gerçek uçuş güvenliği iddiası yoktur.

## Doğrulama kapısı

1. Kutu türetme testleri: sentetik segmentation'dan beklenen kutular,
   örtülme, küçük kutu filtresi, çeldiricilerin etiketlenmemesi.
2. YOLO formatı ve `dataset.yaml`; bölüm sızıntısı denetimi geçer.
3. Dedektör kalite kapıları (§3) üç seed'de geçer; audit tek kullanımlık;
   manifest fail-closed (`visual-vae` manifesti reddedilir).
4. `vision-v4` sözleşmesi: anahtarlar, shape, aralık, yuva sıralaması,
   boş yuva değeri, menzil tahmini geometrisi (sentetik kutu -> bilinen
   menzil), `range_valid` kuralı.
5. Hover kapısı dedektör kutusundan; privileged gözlemci yalnız `info`'da.
6. Actor yolunda privileged yakalama yok (`ActorCameraFrame` yalnız
   `rgb`, `timestamp_s`).
7. Aynı seed ve action dizisi aynı observation ve reward'ı verir.
8. `v0` ve `v1` testleri ile legacy parmak izi değişmez.
9. Throughput `>= 0.5 x v0` (aksi halde §8 yedek yolu Revizyon olarak açılır).
10. D1a eğitimi ADR §10 kapısı.

## Revizyon 1 (29 Eylül 2026, kullanıcı kararları)

1. **Tek seed.** İlk seed bütün kapıları iki geliştirme bölümünde geçtiği
   için kalan iki seed beklenmedi. Kalite raporu tamamlanmış seed'lerden
   kurulur (`finalize_detector_quality`); rapor kararı, planlanan seed
   listesini ve yarıda kalan dizini taşır.
2. **Şekil kısıtlı onay (REV-050).** Audit'in elipsoit engellerinde engel
   kapıları geçmedi. Kullanıcı yeni audit veya yeniden eğitim istemedi;
   dedektör eğitimde görülen şekillerle (kutu, silindir, kapsül, küre)
   sınırlı onaylandı. Onay manifestinin iki temeli vardır: `audit_passed`
   (kısıtsız) ve `user_decision` (karar metni, bulgu kimliği, audit raporuyla
   birebir eşleşen geçmeyen kapılar ve onaylı şekil kümesi zorunlu). `v2`
   ortamı şekil kümesi dışındaki sahne config'lerini kurulumda reddeder.
   Kalite kapıları (geliştirme bölümleri) her iki temelde de zorunludur.
3. **Tanı render'ı eğitimde kapalı.** Privileged hedef görünürlüğü yalnız
   `info` tanısıdır ve her adımda ek segmentation render'ı ister. Ölçülen
   throughput (rastgele politika, D1a, tek ortam): `v0` 88.0 adım/s;
   `v2` tanı açık 43.0 (`0.49 x`), tanı kapalı 47.7 (`0.54 x`),
   randomization da kapalı 54.3 (`0.62 x`). §8 kapısı (`>= 0.5 x`) tanı
   kapalıyken geçer; `vision_navigation_v4.toml` varsayılanı kapalıdır,
   evaluation ve replay script'leri açar.

## Ölçülen sonuçlar

**Dilim B, 29 Eylül 2026.** YOLO11n, `runs/yolo_dataset_d1nr_20260926`
(zemin yansıması kapalı), 60 epoch, COCO ön-eğitimli başlangıç, aday
`last.pt`. **Kullanıcı kararıyla tek seed** (`20260911`): ilk seed bütün
kapıları iki geliştirme bölümünde geçtiği için kalan iki seed beklenmedi
(`runs/detector_quality_v1_20260929/quality_report.json` kararı ve planlanan
seed listesini taşır). Geliştirme bölümleri (validation / development_ood):
engel recall `0.950 / 0.983`, engel precision `0.939 / 0.956`, hedef recall
`0.973 / 0.983`, hedef precision `0.924 / 0.982`, sahte hedef
`0.013 / 0.005`, gecikme `5.8 / 5.2 ms`. Tek kullanımlık audit
(`runs/detector_audit_v1_20260929`, yalnız elipsoit engeller) **geçmedi**:
engel recall `0.456`, precision `0.887`; hedef kapıları geçti. Tanı ve
seçenekler REV-050'de; kullanıcı kararıyla şekil kısıtlı onay (Revizyon 1).
Bilgi: gri tonlamada hedef recall `0.17`'ye düşer, dedektör hedefi büyük
ölçüde renginden tanır.

**Dilim C, 29 Eylül 2026.** Menzil tahmini MuJoCo projeksiyonuyla çapraz
doğrulandı (roll 6, pitch 9 derece; göreli hata `< 1e-6`). Onaylı dedektörle
gerçek renderer smoke'u temiz (`runs/detection_vision_smoke_20260929`).
40 engelli sahnede rastgele uçuşta 451 karenin `%99`'unda engel kutusu
(kare başına ortalama `2.0`); kadraj kenarında yarım görünen engel için
zaman zaman üst üste 2-3 kutu. PPO kuru koşusu (4 `subproc` ortam, her
birinde dedektör MPS'te): `~100` adım/s.

**Dilim D, 30 Eylül 2026 (ADR-024/025 ve ADR-017 R1 ile).** Held-out 40
episode deterministik, üç seed: hover başarısı `%87.5 / %77.5 / %87.5`
(ortalama `%84.2`), arrival aynı, çarpışma `%5 / %22.5 / %12.5` (ortalama
`%13.3`). §10 kapısı: arrival geçti, çarpışma geçilmedi. Ayrıntı
`docs/PROGRESS_LOG.md` "Dilim D kapanışı".

## Sonuçlar ve sınırlar

- Dedektör bir kaçırma yaptığında politika o engeli o karede görmez; kaçınma
  payı (`vehicle_clearance_m`) ve `proximity` şekillendirmesi bunu kısmen
  telafi eder. Kaçırma oranı kapıyla sınırlanır ama sıfır değildir.
- Menzil tahmini engelin zemine bastığı varsayımına dayanır; havada asılı
  engel sahne üreticisinde yoktur, gerçek dünyada olabilir. Politika
  `range_valid` ile bunu öğrenebilir; garanti değildir.
- Ultralytics bağımlılığı büyüktür (torchvision `0.29`, opencv-python `5.0`,
  matplotlib, polars, pyyaml, requests, psutil, nvidia-ml-py) ve
  telemetri/senkron ayarları kapatılmalıdır (`settings sync=False`,
  çevrimdışı çalışma). Bağımlılık `pyproject.toml`'da isteğe bağlı grup
  (`detector`) olarak sabitlenir ki çekirdek ortam etkilenmesin.
- Lisans AGPL-3.0'dır (kullanıcı itirazına açık varsayım).
- Zemin yansıması kapatıldığı için simülatör görüntüsü gerçek parlak
  zeminlerden daha az yansıma içerir; gerçek görüntü ince ayarı (§12) bu farkı
  taşımak zorundadır.
- Dedektör döngüde olduğu için eğitim süresi `~2-3x` uzar; `subproc`
  vektör ortamında süreç başına bir dedektör kopyası bellek alır.
- Hiçbir maddede gerçek uçuş güvenliği iddiası yoktur.
