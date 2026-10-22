# ADR-018: Tek Hedefli Ölçek Merdiveni ve Algı-Hız Kısıtı

**Durum:** Kabul edildi  
**Tarih:** 12 Eylül 2026  
**İlişki:** ADR-010 mesafe curriculum'unu ve ADR-013 episode/bütçe mimarisini
**revize eder**. ADR-012 terminal görevini ve ADR-016/ADR-017 hover
sözleşmesini değiştirmez.

## Bağlam

İki girdi bu kararı gerektirdi.

**Görev kapsamı kullanıcı tarafından sadeleştirildi.** Drone'un episode başına
**tek** hedef noktası olacaktır. Hedefin konumu episode'lar arasında
değişebilir fakat ara waypoint yoktur. ADR-010 D3-D5 tanımları ("çoklu
waypoint ve yerel pencere geçişi") ve ADR-013 mission-level segment zinciri bu
kapsamın dışındadır.

**Mevcut ortam gerçekçi değil ve bir sonraki basamağı taşımıyor.** Workspace
`6 x 5 x 2.5 m`; endpoint kutusunun köşeden köşeye mesafesi `5.79 m`. D1'in
(`5-15 m`) üst yarısı bu ortama fiziksel olarak sığmaz.

## Ölçümler

Kararlar aşağıdaki ölçümlere dayanır.

### Algı menzili seyir hızını sınırlar

Fren mesafesi `v^2 / (2 * g * tan(tilt))` artı kamera kare-arası yol (`30 Hz`)
artı bir policy adımı (`20 Hz`). Ortamın failure tilt eşiği `75 derece`
olduğundan seyirde `10 derece` hover koşulu geçerli değildir; gerçekçi seyir
tilti `30 derece` alınmıştır.

| Seyir tilti | `3 m/s` | `5 m/s` | `8 m/s` | `10 m/s` | `12 m/s` |
|---:|---:|---:|---:|---:|---:|
| `10°` | `2.9 m` | `7.6 m` | `19.2 m` | `29.7 m` | `42.6 m` |
| `20°` | `1.5 m` | `3.9 m` | `9.6 m` | `14.8 m` | `21.2 m` |
| `30°` | `1.0 m` | `2.6 m` | `6.3 m` | `9.7 m` | `13.7 m` |
| `40°` | `0.8 m` | `1.9 m` | `4.6 m` | `6.9 m` | `9.7 m` |

`free_space_max_depth_m = 8.0` olduğundan `30 derece` tiltte ham üst sınır
`~9 m/s`'dir. Ancak encoder'ın free-space hatası `MAE ~0.25` (normalize `[0, 1]`
ölçekte, yani `+-2 m` mutlak) olduğundan kullanılabilir ufuk `~6 m` kabul
edilmelidir. Bu da **tasarım seyir hızını `6 m/s`** yapar.

Bu kısıt fiziksel ve indirgenemezdir: araç algı ufkunun içinde durabileceğinden
hızlı uçamaz. Daha yüksek hız ancak algı menzilini büyütmekle (yeni dataset ve
encoder) veya seyir tiltini artırmakla mümkündür.

### Tek hedefle episode uzunluğu

ADR-013 formülü tek hedefe uygulandığında (segment değil, tüm episode):

```text
T = clamp(T_fixed + k_time * L / v + T_terminal, T_min, T_max),  N = ceil(T * 20)
```

| Mesafe | `v=1.5 m/s` | `v=6 m/s` |
|---:|---:|---:|
| `15 m` | `710` adım | `413` adım |
| `50 m` | `1 527` adım | `652` adım |
| `150 m` | `3 860` adım | `1 235` adım |
| `500 m` | `12 027` adım | `3 277` adım |
| `2 000 m` | `47 027` adım | `12 027` adım |

### MuJoCo engel bütçesi

| Engel | `ngeom` | `1000` fizik adımı | `step/s` |
|---:|---:|---:|---:|
| `8` | `29` | `16 ms` | `61 972` |
| `50` | `71` | `27 ms` | `37 038` |
| `200` | `221` | `71 ms` | `14 024` |
| `800` | `821` | `268 ms` | `3 738` |

`4` paralel ortam `500 Hz` fizikte `2 000 step/s` gerektirir; pratik geom
bütçesi **`~800` engel**tir.

### Sahne üretimi grid ile doğrusal büyür

| Workspace | Grid hücre | Engel | Süre/sahne |
|---:|---:|---:|---:|
| `6 m` | `2 988` | `8` | `70 ms` |
| `15 m` | `18 675` | `8` | `611 ms` |
| `20 m` | `33 200` | `20` | `2.5 s` |
| `30 m` | `74 700` | `40` | `8.1 s` |

Maliyet `has_navigable_xy_path` BFS'inden gelir ve `10 cm` grid hücresiyle
doğrusaldır. D2 kenarında (`65 m`) sahne başına `~38 s` olur; eğitim her
episode'da reset ettiği için bu kullanılamaz.

## Karar

### 1. Tek hedef, ara waypoint yok

Her episode'un tek bir hedef noktası vardır. Konumu episode'lar arasında
değişebilir. Aşağıdakiler kapsam dışıdır ve uygulanmayacaktır:

- Ara waypoint üretimi ve waypoint indeksi
- Mission segment manager ve segmentler arası fizik durumu sürekliliği
- ADR-013'ün "yerel eğitim segmenti / mission evaluation" ayrımı

ADR-013'ten **korunan** tek şey rota uzunluğuna bağlı zaman bütçesi
formülüdür; artık segmente değil tüm episode'a uygulanır. `vision-v1`
observation'ındaki `waypoint` anahtarı zaten tek hedefi taşıdığı için anlamı
değişmez.

### 2. Uzun mesafe eğitimle değil kompozisyonla doğrulanır

Tek hedefle ara waypoint olmadığı için uzun mesafe episode uzunluğuna doğrudan
yansır. `2 km`'de `12 027` adım PPO kredi ataması için pratik değildir.

Bu nedenle merdiven iki rejime ayrılır:

- **Eğitim rejimi: D1-D4.** Zor olan iki davranış burada öğrenilir: seyirde
  engelden kaçınma ve terminal hover.
- **Evaluation rejimi: D5.** Aynı politika eğitilmeden uzun mesafede
  değerlendirilir. Ölçülen şey, yerel olarak öğrenilen davranışın mesafe
  boyunca kompoze olup olmadığıdır.

Bu ayrım ADR-013'ün kredi atama gerekçesini korur fakat çoklu waypoint
gerektirmez. Uzun seyrin öğrenilebilir olmasının nedeni `target_progress`
bileşeninin yoğun (her adım) reward vermesidir; zorluk uçların ikisinde
toplanmıştır.

### 3. Seyir hızı algı menziliyle birlikte yükseltilir

`max_forward_speed_m_s` D2'de `1.5 -> 6.0` yükseltilir. Bu `fc_state` içindeki
gövde hızı normalizasyonunu değiştirdiği için **observation `vision-v2`**
gerektirir ve yeni environment ID'si alır. `vision-v1` ve D0-V sonucu
dondurulur.

Hız yükseltilirken birlikte doğrulanması zorunlu olanlar:

- Kademeli kontrolcünün `6 m/s` seyirde kararlılığı ve tilt zarfı
- Fren mesafesinin ölçülmüş free-space hatasıyla birlikte algı ufkuna sığması
- Kamera kare-arası yolun (`6 m/s` -> `0.20 m`) engel tespitini bozmaması

### 4. Sahne üretimi iki çözünürlüklü doğrulamaya geçer

`10 cm` grid bütün alana genişletilmeyecektir. `has_navigable_xy_path`:

1. Kaba grid (`~1 m`) ile aday koridoru bulur
2. Yalnız aday koridor çevresinde ince grid (`10 cm`) doğrular

**Güncelleme (12 Eylül 2026).** Profil, maliyetin grid çözünürlüğünden değil
hücre başına bütün engelleri tarayan Python döngüsünden geldiğini gösterdi.
Vektörleştirme ve önceden hesaplanan occupancy grid'i ile D1 ölçeği
(`20 x 20 m`, 40 engel) `2 524 ms` -> `164 ms` düştü ve davranış bit-bit
korundu. Bu nedenle iki çözünürlüklü doğrulama **D1'de gerekmez**; D2 ön koşulu
olarak kalır (`65 m`, 210 engel hâlâ `4.9 s`).

Uygulandığında kaba geçiş **iyimser** olmalıdır (aday koridoru asla elememeli)
ve ince doğrulama koridorda yapılmalıdır; aksi halde REV-003 sınıfı
false-negative geri gelir.

### 5. Ego-merkezli yerel sahne penceresi D3'ten zorunludur

Engel yoğunluğu ölçekle sabit tutulamaz. Pratik geom bütçesi `~800` engeldir;
`200 x 200 m` alanda mevcut yoğunluk (`0.27/m^2`) `10 800` engel isterdi.

İki kaldıraç birlikte kullanılır:

- **Yoğunluk azaltımı:** dış mekân için gerçekçi olan `0.02-0.10/m^2` aralığı
- **Yerel pencere:** yalnız aracın çevresindeki engeller MJCF'e alınır

Yerel pencere tek hedefli tasarımla çelişmez; waypoint geçişiyle değil
**render/çarpışma kapsamıyla** ilgilidir.

### 6. Merdiven

| Seviye | Rota | Workspace | Seyir | Episode | Yoğunluk | Engel | Rejim |
|---|---:|---:|---:|---:|---:|---:|---|
| D1 | `5-15 m` | `20 x 20 x 3 m` | `1.5 m/s` | `~710` | `0.10/m^2` | `~40` | eğitim |
| D2 | `15-50 m` | `65 x 65 x 5 m` | `6 m/s` | `~650` | `0.05/m^2` | `~210` | eğitim |
| D3 | `50-150 m` | `200 x 200 x 8 m` | `6 m/s` | `~1 235` | `0.02/m^2` | `~800` | eğitim |
| D4 | `150-500 m` | `650 x 650 x 10 m` | `6 m/s` | `~3 277` | `0.02/m^2` | yerel pencere | eğitim |
| D5 | `0.5-2 km` | `2.5 x 2.5 km` | `6 m/s` | `~12 027` | `0.02/m^2` | yerel pencere | **yalnız evaluation** |

Engel zorluğu bağımsız eksen olarak kalır (tek engel -> seyrek -> geçit ->
koridor -> dar geçit -> yoğun karma) ve her mesafe seviyesinde ayrıca
ilerletilir.

## Her seviyenin çıkış kapısı

Ortak koşullar: held-out sahnelerde ölçülür, en az `3` seed, deterministik **ve**
stokastik policy ayrı raporlanır (REV-023), çarpışma oranı üst sınırı aşmaz,
`hover_success` dışında hiçbir terminasyon başarı sayılmaz.

| Seviye | Başarı | Çarpışma | Ek kapı |
|---|---:|---:|---|
| D1 | `>= %85` | `<= %5` | Tek blocker etrafından kaçınma gözlenir; `out_of_bounds <= %5` |
| D2 | `>= %80` | `<= %5` | `vision-v2` parity; `6 m/s` seyirde kontrolcü kararlılığı |
| D3 | `>= %75` | `<= %8` | Yerel pencere geçişinde fizik sürekliliği; forgetting kontrolü |
| D4 | `>= %65` | `<= %10` | Bütçe kullanım oranı ve timeout dağılımı raporlanır |
| D5 | `>= %50` | `<= %10` | Eğitilmeden ölçülür; kompozisyon kanıtı olarak raporlanır |

Eşikler başlangıç hipotezidir ve ölçümle revize edilebilir; revizyon gerekçesi
`docs/REVIEW_FINDINGS.md` veya yeni ADR ile kayda geçmelidir.

## Ön koşullar

| Ön koşul | Gereken seviye | Durum |
|---|---|---|
| ADR-013 rota temelli bütçe | D1 | ✅ uygulanmış |
| Rota doğrulaması D1 ölçeğinde yeterli hız | D1 | ✅ vektörleştirildi (`164 ms`) |
| İki çözünürlüklü rota doğrulaması | D2 | ✗ (`65 m` hâlâ `4.9 s`) |
| Büyütülmüş workspace ve ölçekli yoğunluk | D1 | ✗ |
| REV-023 hover geometrisi kararı | D1 | ✗ açık |
| REV-021 risk head (değişken endpoint) | D2 | ✗ açık |
| `vision-v2` observation ve hız yükseltmesi | D2 | ✗ |
| Ego-merkezli yerel sahne penceresi | D3 | ✗ |
| Hıza bağlı algı menzili doğrulaması | D2 | ✗ |

## Sonuçlar ve sınırlar

- Mission segment manager ve çoklu waypoint iş paketleri plandan **çıkarıldı**.
  ADR-013 yalnız zaman bütçesi formülüyle geçerliliğini korur.
- D5 eğitim rejimi değildir. `12 027` adımlık episode'da PPO kredi ataması
  pratik değildir; bu bir kapasite sınırıdır, tasarım tercihi değil.
- `6 m/s` tasarım hızı ölçülmüş algı hatasına dayanır. Encoder'ın free-space
  doğruluğu iyileşirse hız yükseltilebilir; bu ayrı bir ADR gerektirir.
- Bu ADR hiçbir gerçek uçuş güvenliği iddiası taşımaz. Fren mesafesi hesabı
  noktasal kütle yaklaşımıdır; rüzgâr, rotor gecikmesi ve kontrolcü tepkisi
  dahil değildir.
- Mevcut `6 x 5 m` workspace ve D0-V sonucu legacy referans olarak korunur.
