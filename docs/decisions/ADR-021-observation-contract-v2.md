# ADR-021: Observation Sözleşmesi `vision-v2`

**Durum:** Kabul edildi  
**Tarih:** 13 Eylül 2026  
**İlişki:** ADR-015 observation sözleşmesini değiştirir. ADR-009 approval
zincirini yeniler. ADR-020 action sözleşmesinden bağımsızdır (o `vision-action-v2`).
REV-037 ve REV-038'i kapatır.

## Bağlam

`vision-v1` politikaya engel hakkında **altı sayı** veriyordu:

```text
vision_aux[3:6]  free_space  L / C / R
vision_aux[6:9]  collision_risk L / C / R
```

Görüntü üç dikey dilime bölünüyordu ve **dikey çözünürlük yoktu**.

D1 ortamında (`20 x 20 m`, tavana kadar engeller, `~2 m` koridorlar) bu temsil
yetersiz kaldı. Ölçülen (534 adım, ayrıcalıklı gerçekle karşılaştırma):

```text
free_space mutlak hata : 3.07 m (sol 3.63, orta 3.27, sağ 2.30)
hatanın %90 dilimi     : 6.98 m
koridor genişliği      : ~2 m
```

Sinyal gürültünün altındaydı ve politika `600k` adımda kaçınma öğrenemedi
(çarpışma `%84`'te yatay, dört farklı düzeltme denemesine rağmen).

Oysa encoder **zaten** `9 x 12` engel ısı haritası üretiyor ve
`obstacle_mask`'e karşı BCE ile eğitiliyordu. Politikaya verilmiyordu ve
kalitesi hiç ölçülmemişti.

## Karar

### 1. `obstacle_map` kanalı eklenir

```text
obstacle_map : (108,) float32, [0, 1]     9 x 12 ızgara, satır-major
```

Encoder'ın `obstacle_heatmap_logits` çıktısının sigmoid'i, düzleştirilmiş.

**Neden bu ızgara anlamlıdır.** Kamera `45 derece` aşağı bakar ve düz zeminde
bütün ışınlar zemine çarpar; bu yüzden görüntü **satırları zemin mesafesine**,
**kolonları yön açısına** karşılık gelir (ADR-016 zemin izi geometrisi). Yani
kanal bir **yerel mesafe x yön doluluk ızgarasıdır** ve koridorun *nerede*
olduğunu taşır. Üç dikey dilim bunu taşıyamıyordu.

Kanal kendi space'inde `[0, 1]` ham kalır; `vision_aux` gibi `[-1, 1]`'e
kaydırılmaz, çünkü olasılıktır.

### 2. `collision_risk` kanalı çıkarılır

`vision_aux` `9 -> 6`:

```text
[target_score, center_x, center_y, free_L, free_C, free_R]
```

`collision_risk` etiketi tanım gereği `free_space < collision_risk_threshold`'dı,
yani politikanın zaten sahip olduğu kanalın eşiklenmiş kopyası, daha düşük
bilgi içerikli. `obstacle_map` aynı bilgiyi `108` hücrede taşır.

Bu kanal approval kapısı olmaktan da çıkar (`gate_collision_risk = false`).
Varsayılan `true`'dur, yani `v9`/`v10` zincirinin anlamı korunur. Metrikler
raporlanmaya devam eder, tanısal değerleri sürer.

Gerekçe sırası kayda geçirilir: kanal **zaten gereksizdi** ve yerine daha
yüksek çözünürlüklü olanı geldi; kalite kapısından kalması bunu fark ettirdi.
Eşik düşürmek (kapı avcılığı, REV-021'de reddedilmişti) ile kanalı
sözleşmeden çıkarmak farklı şeylerdir.

### 3. Engel haritası approval kapısına alınır

```text
minimum_obstacle_map_iou       = 0.30
minimum_obstacle_map_precision = 0.45
minimum_obstacle_map_recall    = 0.55
```

Ölçülmemiş bir kanalı actor observation'ına koymak ADR-009 approval zincirinin
amacına aykırıdır. Eşikler "kullanılabilir harita" alt sınırıdır ve ilk
ölçümlerden sonra sıkılaştırılabilir.

### 4. Encoder D1 ortamında yeniden eğitilir

`vision-encoder-v11`, `vision_dataset_d1_20260913` üzerinde:

```text
kare sayısı  : 736 -> 6528
sahne config : D1 (20 x 20 m, tavana kadar engeller, 0.10/m^2, 21.13 m zemin)
irtifa bandı : 0.80-2.10 -> 1.20-2.90 m
               eski üst sınır seyir irtifasının (2.156 m, REV-036) ALTINDAYDI
yanal offset : 1.25 -> 2.50 m
split_salt   : ebabil-vision-d1-v5
```

Mimari değişmez (`TinyVisionEncoder`, `latent_size 64`,
`risk_head_hidden_size 64`); değişen şey eğitim dağılımıdır. Risk head'i
modelde **kalır**: kaybı temsil öğrenmesine katkı verir, yalnız çıktısı
observation'a girmez.

### 5. Sızıntı güvencesi isimden ölçüme taşınır

`obstacle_map` ayrıcalıklı değildir fakat isim tabanlı yasak listesi
`"obstacle"` token'ını reddediyordu. Liste `obstacle_mask` ve `segmentation`'a
daraltıldı ve yerine iki iddia kondu:

1. Kanal, encoder ısı haritasının sigmoid'iyle **birebir** aynıdır.
2. Actor kamerası `capture_privileged` sunmaz; `ActorCameraFrame` yalnız
   `("rgb", "timestamp_s")` taşır.

Yani güvence "isminde obstacle yok" değil, "kaynağı yalnız RGB olabilir"dir.

## Doğrulama kapısı

1. `observation_space` anahtarları sıralı ve `VISION_OBSERVATION_KEYS` ile birebir.
2. `obstacle_map` shape `(108,)`, aralık `[0, 1]`, sonlu.
3. Kanal encoder çıktısıyla `atol 1e-6` içinde aynı.
4. Actor kamerasının ayrıcalıklı çıktı üretmemesi.
5. `vision_aux` shape `(6,)` ve `collision_risk` içermemesi.
6. Kalite kapısında engel haritası eşiklerinin uygulanması; `gate_collision_risk`
   varsayılanının `true` olması.
7. 3 seed kalite + bağımsız audit geçişi.
8. Legacy rollout parmak izinin değişmemesi.

## Ölçülen sonuçlar

```text
free_space hatası (development_ood) : 2.06 m -> 1.05 m
risk_precision                      : 0.709 -> 0.849
target_recall                       : 0.929 -> 0.921
engel haritası IoU                  : ölçülmemiş -> 0.651 (audit 0.614)
engel haritası recall               : ölçülmemiş -> 0.796 (audit 0.860)
quality_passed                      : True  (3 seed)
audit_passed                        : True
LEGACY_ROLLOUT_SHA256               : 03f16232...4e3f5e23 (değişmedi)
```

## Sonuçlar ve sınırlar

- **Politika davranışı henüz ölçülmedi.** Bu ADR algı tarafını düzeltir;
  kaçınmanın gerçekten öğrenildiği ancak eğitim koşusuyla anlaşılır.
- Engel haritası `9 x 12`'dir ve satır-mesafe eşlemesi **düz zemin** varsayar.
  Eğimli arazide satır indeksi mesafeye karşılık gelmez.
- Harita encoder'ın tahminidir; `precision 0.782` demek beşte bir hücrenin
  yanlış pozitif olduğu anlamına gelir. Politika bunu gürültülü bir ölçüm
  olarak öğrenmek zorundadır.
- `collision_risk` modelde durmaya devam eder fakat hiçbir tüketicisi yoktur.
  İleride tamamen kaldırılması ayrı bir sadeleştirme kararıdır.
- Observation boyutu `89 -> 194` büyüdü (`fc_state 10`, `timing 2`,
  `vision_aux 6`, `obstacle_map 108`, `vision_latent 64`, `waypoint 4`).
  PPO için sorun beklenmiyor fakat örnek verimliliğini etkileyebilir.
- Encoder hâlâ simülasyon etiketleriyle eğitilmiştir; ADR-011 tezgâh kapısı ve
  gerçek görüntü doğrulaması açıktır.
