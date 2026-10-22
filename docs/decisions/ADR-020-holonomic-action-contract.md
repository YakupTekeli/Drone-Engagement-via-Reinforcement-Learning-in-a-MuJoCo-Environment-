# ADR-020: Holonomik Action Sözleşmesi (`vision-action-v2`)

**Durum:** Kabul edildi  
**Tarih:** 12 Eylül 2026  
**İlişki:** ADR-015 action uzayını değiştirir (observation sözleşmesi
**değişmez**). ADR-012/ADR-016 terminal hover görevini çözülebilir kılar.
REV-028 ve REV-032'yi kapatır. ADR-008 legacy ortamı etkilenmez.

## Bağlam

`vision-action-v1` üç boyutluydu:

```text
a[0]  ileri hız    max(0, a[0]) * 1.5 m/s      <- GERİ YOK
a[1]  dikey hız    a[1] * 1.0 m/s
a[2]  yaw-rate     a[2] * 1.5 rad/s
```

ADR-015 bunu gerekçelendirmişti: tek ileri bakan kameranın yan ve arka kör
alanı var, oraya translasyon yapmak güvensiz. Gerekçe geçerlidir fakat **bir
donanım kısıtını araç dinamiğine gömer.** Gerçek bir quadrotor yatay düzlemde
izotropiktir; yana ve geriye uçabilir. Legacy `EbabilNavigation-v0` (ADR-008)
dört boyutlu action ile bunu zaten yapıyordu; kısıt yalnız vision ortamına
konmuştu.

## Ölçülen zarar

Dört D1 koşusu boyunca (reward tasarımı üç kez değişti, zemin düzeltildi)
terminal aşamada yatay mesafe medyanı hiç `0.8 m`'nin altına inmedi:

| Koşu | progress formu | waypoint'e mesafe medyanı |
|---|---|---:|
| 1 (kanıt) | `5.0/m` | `0.591 m` (kırık zemin) / `0.869 m` (düzeltilmiş) |
| 3 | `15.0 * dd / L` | `0.991 m` |
| 4 | iki menzilli | `1.709 m` |
| 5 (düzeltilmiş zemin) | `5.0/m` | `1.144 m` |

Belirleyici ölçüm, aracın halkaya **girdiği** fakat **kalamadığı**dır:

```text
hedefin 0.65 m içine giren episode : %65
başarı                             : %15
terminal adımlarda güvenli bölge dışı ihlali : %85.6
```

Neden kalamadığının geometrisi:

- Hold koşulu: merkeze `<= 0.650 m`, hız `<= 0.25 m/s`, kesintisiz `3 s`
- Hedefin görünür kalması: merkeze `>= 0.407 m` (nadir kör bölgesi, ADR-016)
- Uygulanabilir halka: `0.243 m` genişliğinde
- Araç yalnız baktığı yöne ilerleyebiliyor, yani yanal düzeltme için **yaw
  çevirmek zorunda**
- Yatay FOV `75 derece`, yani `38 dereceden` fazla çevirince hedef kadraj
  dışına çıkıyor ve hold koşulu bozuluyor

Yani ince yanal düzeltmenin **çıkışı yoktu**: düzeltmek için dönmek gerekiyor,
dönmek hedefi kaybettiriyor. Bu bir reward veya algılama sorunu değil, action
uzayının görevle yapısal çelişkisidir.

## Karar

### 1. Action dört boyutlu ve tamamen işaretli olur

```text
a[0]  ileri/geri gövde hızı
a[1]  yanal gövde hızı
a[2]  dünya dikey hızı        a[2] * 1.0 m/s
a[3]  yaw-rate                a[3] * 1.5 rad/s
```

`Box(low=-1, high=1, shape=(4,), dtype=float32)`.

### 2. Yatay hız sınırı izotropiktir ve büyüklük üzerinden uygulanır

```python
horizontal_body = a[:2] * max_horizontal_speed_m_s
speed = ||horizontal_body||
if speed > max_horizontal_speed_m_s:
    horizontal_body *= max_horizontal_speed_m_s / speed
```

Gerçek bir quadrotorun hız zarfı yöne göre değişmez. Bileşenleri tek tek
kırpmak çapraz komutta `sqrt(2) = 1.41` kat hızlı uçmaya izin verirdi; legacy
ortam bunu yapar ve bu bir legacy kusurudur, kopyalanmadı.

`max_horizontal_speed_m_s = 1.5` korunur, yani **hız zarfı büyütülmedi**;
yalnız yön kısıtı kaldırıldı.

### 3. Kör alan kısıtı koddan kaldırılır, ödüle bırakılır

Kör alan gerçektir: ölçüldüğünde `random_obstacles` sahnelerinde kamera görüş
konisinin dışında, `2 m`'den yakın engel sayısı görünür alandakinin `3` katıdır
(606 vs 193 tespit). Fakat gerçek sistemler bunu **davranışla** çözer: gittiği
yöne döner. Bu bir donanım kısıtı değil öğrenilecek bir politikadır.

Dolayısıyla körlemesine yanal uçmanın bedeli `collision_penalty` ile
fiyatlanır ve politikanın bunu öğrenmesi beklenir. Öğrenip öğrenmediği
çarpışma oranıyla ölçülür. Bu, kısıtı action uzayına gömmekten daha dürüst bir
deney kurar.

### 4. Observation sözleşmesi değişmez

`vision-v1` observation'ı (`fc_state`, `timing`, `vision_aux`,
`vision_latent`, `waypoint`) **aynı kalır**. Encoder yalnız görüntü tüketir ve
action uzayına hiç bağlı değildir; doğrulandı. Bu nedenle onaylı zincir
(`vision_dataset_v4b/v4c` -> `vision_quality_v16/v20` -> `vision_audit_v2/v3`
-> `vision-encoder-v9/v10`) **geçerliliğini korur** ve dataset yenilemesi
gerekmez. Yalnız politika yeniden eğitilir.

Sürümleme ayrılır: `VISION_OBSERVATION_VERSION = "vision-v1"` değişmez,
`VISION_ACTION_VERSION = "vision-action-v2"` eklenir.

### 5. Legacy ortam dokunulmaz

`EbabilNavigationEnv` (ADR-008) dört boyutlu action'ını ve bileşen bazlı
kırpmasını korur. Legacy rollout parmak izi doğrulandı:
`03f16232adae103e370fe2a5cbcb2d0c62900838f90b51fc7db408e14e3f5e23`, değişmedi.

## Doğrulama kapısı

1. `action_space.shape == (4,)`.
2. Dört temel yön (ileri, geri, sağa, sola) beklenen dünya hızını üretir.
3. Çapraz komutta yatay hız büyüklüğü `max_horizontal_speed_m_s`'i aşmaz.
4. Yaw-rate kanalı `a[3]`'e taşındı ve ölçeği değişmedi.
5. Yanlış shape (3 boyut) ve NaN reddedilir.
6. Legacy rollout parmak izinin değişmemesi.
7. Observation sözleşmesinin `vision-v1` kalması ve approval zincirinin
   doğrulanabilir olması.
8. Davranışsal: terminal adımlarda yatay mesafe medyanı `0.869 m`'nin
   (düzeltilmiş zeminde en iyi ölçüm) altına inmeli.

## Sonuçlar ve sınırlar

- **Çarpışma oranı yükselebilir.** Politika artık göremediği yöne uçabiliyor.
  Bu beklenen bir sonuçtur ve deneyin sorusudur; ölçülmeden yorum yapılmaz.
- Bu ADR bir hız zarfı artışı **değildir**. `1.5 m/s` korunur; ADR-018'in
  `6 m/s` tasarım hızı ayrı bir karardır ve D2 işidir.
- Kör alanda güvenli uçuş gerçek bir güvenlik gereksinimidir. Simülasyonda
  ödülle öğretmek, gerçek uçuşta güvenlik garantisi sağlamaz; ADR-011 tezgâh
  kapısı ve SITL/HITL aşaması geçerliliğini korur.
- Engel algısının zayıflığı (`~2 m` mesafe hatası, `risk_specificity 0.56`)
  bu ADR ile çözülmez. Aracın gidebileceği yönleri artırmak, gördüğü şeyi
  iyileştirmez. İki iş ayrıdır ve algılama tarafı açıktır.
