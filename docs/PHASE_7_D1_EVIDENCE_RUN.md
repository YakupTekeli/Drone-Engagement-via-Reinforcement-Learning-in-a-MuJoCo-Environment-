# Faz 7: D1 Kanıt Koşusu

**Tarih:** 12 Eylül 2026  
**Durum:** Tamamlandı; ADR-018 D1 kapısı **geçilmedi**. Koşunun asıl çıktısı üç
ölçülmüş blocker'dır (REV-023, REV-025, REV-026).

## Amaç ve kapsam

Projede ilk kez **engelli** ve **büyütülmüş** bir ortamda, hedefin yalnız
yaklaşık konumu bilinerek kamera tabanlı politika eğitmek.

```text
workspace   20 x 20 x 3 m   (D0-V: 6 x 5 x 2.5 m)
rota        5.15-14.87 m    (D0-V: sabit 4.00 m)
engel       1 (single_blocker)  (D0-V: 0)
hedef       R = 2 m yaklaşık konum (D0-V: kusursuz oracle)
endpoint    random_both     (D0-V: fixed)
mod         single_blocker
```

**Düzeltme (12 Eylül 2026).** Bu tablo önce "engel 40, 0.100/m^2" diyordu ve
**yanlıştı.** `static_scene_d1.toml` içindeki `obstacle_count = 40` yalnız
`random_obstacles` modunda kullanılır; bütün D1 koşuları
`scene_mode = "single_blocker"` ile yapıldı, yani sahnede **tek** engel vardır.
Ölçülen yerleşim (300 sahne):

```text
engel sayısı          1
rotadaki konumu       %49.2 - %50.0   (her zaman orta nokta)
rotadan yanal sapma   0.000000 m      (her zaman tam çizgi üzerinde)
yatay yarı-boyut      0.15 - 0.45 m
yükseklik yarı-boyut  0.60 - 1.20 m   (yani 1.2 - 2.4 m boyunda)
hedefe mesafesi       rota 5-8 m'de medyan 3.43 m; hiç terminal bölgeye
                      (<= 2.5 m) düşmüyor
```

Yani D1, `20 x 20 m` boş bir alanda rotanın tam ortasına konmuş **tek** küçük
bir kutudur. Bu düzeltme D1 sonuçlarının yorumunu değiştirir: `%15-20`
çarpışma oranı, yoğun bir engel ormanında değil **tek ve konumu tümüyle
öngörülebilir** bir engelde ölçülmüştür.

Bu koşu tek seed'dir (`20260912`) ve ADR-009'un üç-seed gereksinimini
karşılamaz. Kapı geçilmediği için ek seed'ler koşulmadı.

## Eğitim sonucu

```bash
.venv/bin/python scripts/train_ppo.py \
  --config src/ebabil/configs/training/ppo_vision_d1.toml \
  --device cpu --run-name d1-seed20260912
```

`600 064` timestep, `1 760` episode, `71.1 dk`, `140 ts/s`. SB3 son raporu:
`ep_rew_mean 34.7`, `success_rate 0`, `ep_len_mean 404`.

| Timestep | Reward | `hold_max` | Baskın terminasyon |
|---|---:|---:|---|
| `0-79k` | `-16.95` | `0.00` | `target_lost`, `out_of_bounds` |
| `79-185k` | `-0.03` | `0.15` | `target_lost` |
| `185-279k` | `+7.26` | `0.30` | `time_limit` |
| `280-384k` | `+11.28` | `0.45` | `time_limit` |
| `384-488k` | `+17.68` | `0.85` | `time_limit` |
| `489-599k` | `+22.10` | `0.85` | `time_limit` |

Reward monoton yükseldi (`-16.95 -> +22.10`). `hold_max` son `200k` adımda
`0.85 s`'de tavan yaptı; gereken `3.0 s`.

Son `300` episode: `time_limit` 169, `obstacle_collision` 62,
`target_lost` 59, `out_of_bounds` 10. `hold_seconds` `263/300` episode'da sıfır;
`hold_progress` maksimum `0.283`.

## Evaluation

20 episode, seed `410001`, `single_blocker` + `random_both`, D1 sahnesi:

| Checkpoint | Policy | Başarı | Çarpışma | Abort | Timeout | Return |
|---|---|---:|---:|---:|---:|---:|
| `final` | deterministik | **`%25`** | `%5` | `%45` | `%15` | `31.6` |
| `final` | stokastik | `%0` | `%10` | `%45` | `%35` | `25.6` |
| `best` | deterministik | `%10` | `%5` | `%30` | `%50` | `19.1` |

**Eğitimin `success_rate 0` raporu stokastik rollout istatistiğidir.**
Deterministik politika `%25` başarı üretir. Bu, D0-V'de bulunan
deterministik/stokastik ayrışmasının (REV-023) D1'de de geçerli olduğunu, hatta
D0-V'nin `final` checkpoint'indeki yönün tersine döndüğünü gösterir: burada
gürültü politikayı `11 cm` halkadan çıkarıyor.

**ADR-018 D1 kapısı geçilmedi:** başarı `%25` (gereken `>=%85`). Çarpışma `%5`
sınırda (gereken `<=%5`). Eşikler değiştirilmedi.

`final` deterministik ayrıntı: rota ort `10.70 m`, hedef hatası ort `1.338 m`,
yerleşme adımı `157.5`, ilk hold adımı `206.9`.

## Üç ölçülmüş blocker

### REV-023 — Hold halkası `11 cm`

```text
güvenli iç bölge (hold koşulu)      : merkeze <= 0.650 m
hedef görünür (abort olmamak için)  : merkeze >= 0.542 m
kesişim                             : 0.108 m
```

Politika `R = 2 m` hatayla varılan noktadan gerçek hedefi bulup bu halkada
`3 s` durmak zorunda. D0-V'de hedef waypoint'in tam gösterdiği yerdeydi, bu
yüzden halkaya doğrudan iniliyordu.

### REV-025 — Grace period toparlanmaya yetmiyor

İnanılan konumda, yaw taraması yapılmazsa gerçek hedefin kadrajda kalma
olasılığı `%19.2` (200k örnek). `360 derece` tarama ile `%92.7`.

`max_yaw_rate = 1.5 rad/s` ile `0.50 s`'de yalnız `43 derece` dönülebilir;
rastgele yönde kaybedilen hedef ortalama `90 derece` (`1.05 s`), en kötü
`180 derece` (`2.09 s`) ister. Abort, arama davranışını öğretmek yerine
öğrenmeyi kesiyor. Evaluation'da abort oranı `%45`.

### REV-026 — Çarpışma cezası ölçeklenmiyor

`obstacle_collision` oranı eğitim boyunca **arttı**:
`%15 -> %12 -> %13 -> %13 -> %17 -> %20`.

`target_progress` rota uzunluğuyla doğrusal (`5.0 x L`), `collision_penalty`
sabit (`25.0`). D1 ortalama rotada oran `2.0x`, D3'te `30x`, D4'te `100x`.
Engele çarpıp kısa yoldan gitmek net kazançlı hâle geliyor.

## `search` durumu D1'de inert

Bütün evaluation'larda `mean_search_seconds = 0.00`. Nedeni geometrik: hedef
hover irtifasında `7.46 m`'ye kadar görünür, `search_entry_radius_m` ise
`4.0 m`. `R = 2 m` ile araç arama yarıçapına girmeden hedefi tespit ediyor.

Bu bir defekt değildir: ADR-019 arama geometrisine göre `search` ancak
`R > ~7.5 m` olduğunda gerekir, yani D3'ten itibaren. Durum makinesi uygulanmış
ve testlidir fakat D1-D2'de devreye girmez.

## Bilinen sınırlılıklar

- Tek seed; ADR-009 en az üç ister. Kapı geçilmediği için ek seed koşulmadı.
- `600k` bütçe yetersiz olabilir, fakat `hold_max`'in son `200k`'da `0.85 s`'de
  tavan yapması yalnız bütçe sorunu olmadığını gösterir.
- Deterministik ve stokastik sonuçlar `%25` vs `%0` ile ayrışıyor; tek kipte
  raporlanan başarı bu görevde yanıltıcıdır.
- Privileged reward şekillendirmesi kullanılıyor (`target_progress` ve hover
  koşulları gerçek hedefe göre); ADR-012 buna izin verir.
- Sentetik encoder ve privileged başarı doğrulaması gerçek uçuş güvenliği
  değildir.

## Artefact

```text
runs/d1-seed20260912/
  final_model.zip, best/best_model.zip, vecnormalize.pkl
  checkpoints/, evaluation/evaluations.npz
  logs/{train_monitor,eval_monitor,progress.csv,events.out.tfevents.*}
  policy_evaluation/{final,final_stoch,best}.json
```

## REV-025 düzeltmesi ve ikinci koşu

`grace_seconds` `0.50 -> 2.50 s` yapıldı (ADR-017 güncellendi, başarı koşuluna
dokunulmadı) ve aynı `600k` bütçeyle yeniden koşuldu:
`runs/d1-grace-seed20260912`.

Eğitim karşılaştırması (tam `600k`):

| Metrik | `grace 0.5 s` | `grace 2.5 s` |
|---|---:|---:|
| `hover_abort_target_lost` | `%32` | `%13` |
| `obstacle_collision` | `%15` | `%22` |
| `time_limit` | `%48` | `%57` |
| ortalama episode uzunluğu | `340` adım | `412` adım |
| reward (son çeyrek) | `20.69` | `13.24` |
| `hold_max` | `0.85 s` | `0.70 s` |

Evaluation karşılaştırması (`final`, 20 episode, seed `410001`):

| Koşu | Policy | Başarı | Çarpışma | Abort | Timeout |
|---|---|---:|---:|---:|---:|
| `grace 0.5 s` | deterministik | **`%25`** | `%5` | `%45` | `%15` |
| `grace 0.5 s` | stokastik | `%0` | `%10` | `%45` | `%35` |
| `grace 2.5 s` | deterministik | `%5` | `%15` | `%0` | `%70` |
| `grace 2.5 s` | stokastik | `%0` | `%15` | `%5` | `%80` |

**Düzeltme kendi ekseninde kusursuz çalıştı** (`target_lost` abort'u `%45 -> %0`)
fakat **net sonucu kötüleştirdi** (başarı `%25 -> %5`, timeout `%15 -> %70`).

Bu, dördüncü bulguyu ortaya çıkardı: **REV-027**, hedef görünürlüğünün reward'da
hiçbir pozitif karşılığı olmaması. Görünürlüğü sağlayan tek mekanizma
`target_lost` abort'unun sert cezasıydı; kısa grace kazara bir şekillendirme
sinyali işlevi görüyordu. Grace uzatılınca bu baskı kalktı ve politika hedeften
uzaklaşıp bütçeyi doldurmaya başladı.

`grace_seconds` `2.50 s`'de bırakıldı, geri alınmadı: `0.50 s`'nin aksiyon
limitleriyle çeliştiği ölçüm geçerliliğini koruyor. Ölçülmüş biçimde yanlış bir
değeri, ikinci bir eksiklik onu kazara telafi ettiği için geri koymak doğru
olmaz. Doğru sıra REV-027'yi kapatmak, sonra grace'i kalibre etmektir.

**Bugünün en iyi D1 politikası** kanıt koşusunun `final` checkpoint'idir
(`%25` deterministik): `runs/d1-seed20260912/final_model.zip`.

## Sonraki adım

Dört bulgu açık ve dördü de ADR sözleşmesi değiştirir:

| Bulgu | Değiştirdiği | Öneri | Durum |
|---|---|---|---|
| **REV-026** (bağlayıcı) | ADR-017 reward semantiği | `target_progress`'i `L_plan`'a normalize et, `w = 15.0` | uygulandı |
| **REV-027** | ADR-017 reward semantiği | Hedef görünürlüğü için pozitif bileşen ekle | uygulandı |
| **REV-023** | ADR-012 başarı geometrisi | `hover_height_m` `2.0 -> 1.5` (bant `0.108 -> 0.243 m`) | uygulandı |
| REV-025 | ADR-017 (uygulandı) | REV-027 kapandıktan sonra yeniden kalibre et | bekliyor |

REV-026 ve REV-027 birlikte ele alınmalıdır; ikisi de aynı reward dengesini
ilgilendirir.

**Üçü de 12 Eylül 2026'da kullanıcı onayıyla uygulandı** ve üçüncü D1 koşusu
aynı `600k` bütçe, aynı seed (`20260912`) ve aynı encoder (`v9`) ile başlatıldı:
`runs/d1-rev026-seed20260912`. Encoder bilinçli olarak değiştirilmedi ki sonuç
reward/geometri değişikliklerine atfedilebilsin. Ayrıntı `PROGRESS_LOG.md`,
12 Eylül 2026 "REV-023/026/027 düzeltmeleri" kaydındadır.

Kabul ölçütü bu koşu için iki davranışsal sinyaldir: `obstacle_collision`
trendinin azalması (REV-026'nın asıl testi) ve deterministik başarının `%25`
üzerine çıkması.

## Üçüncü koşu: REV-023/026/027 düzeltmeleriyle

`runs/d1-rev026-seed20260912`, aynı `600k` bütçe, aynı seed, aynı `v9` encoder.
Deterministik, 40 episode, seed `520001`:

| Koşu / checkpoint | Başarı | Çarpışma | Abort | Timeout |
|---|---:|---:|---:|---:|
| kanıt koşusu `final` | **`%42.5`** | `%15.0` | `%5.0` | `%35.0` |
| üçüncü koşu `final` | `%22.5` | `%15.0` | `%12.5` | `%50.0` |
| üçüncü koşu `best` | `%20.0` | `%12.5` | `%7.5` | `%60.0` |

Kapı geçilmedi ve başarı geriledi. Gerileme tek eksende: araç waypoint'e
`0.591 m` yerine `0.991 m` uzakta duruyor, yani konum tutma hassasiyeti
kaybedildi. Kazanımlar: çarpışma teşviki kapandı
(eğitim ortasında `%20 -> %7`), `target_lost` abort'u eğitimde `%30 -> %6`,
irtifa ihlali `%65 -> %17` ve tam `3 s` hold ilk kez ulaşıldı. Ayrıntı ve kök neden REV-028'dedir.

Aynı ölçümde saf waypoint takibinin tavanı da hesaplandı:
`P(hata <= 0.65 m) = (0.65/2)^2 = %10.6`. Hedef hatası `> 0.65 m` olan
episode'larda başarı kanıt koşusunda `%38`, üçüncü koşuda `%25`. İkisi de
tavanın üstünde, yani kamera yer bulmaya gerçekten katkı veriyor.

**Bu belgedeki `%25` sayısı seed `410001` / 20 episode ölçümüdür.** Aynı
politika seed `520001` / 40 episode'da `%42.5` verdi. Bu görevde başarı oranı
tek başına karar ölçütü olarak kullanılmamalıdır; adım bazlı koşul dağılımı
kullanılmalıdır.

**Bugünün en iyi D1 politikası hâlâ kanıt koşusunun `final` checkpoint'idir:**
`runs/d1-seed20260912/final_model.zip`.

## Dördüncü koşu ve REV-026'nın çürütülmesi

`runs/d1-rev028-seed20260912` (iki menzilli progress) 40 episode'da `%0` başarı
verdi; `target_progress` toplamı episode başına yalnız `+4.26` oldu (3. koşu
`+11.71`), yani araç hedefe hiç varmadı.

Bu noktada REV-026 geriye dönük test edildi ve **çürütüldü** (REV-029):

| Rota | 1. koşu | 3. koşu | 4. koşu |
|---|---:|---:|---:|
| `5-8 m` | `%23.3` | `%24.2` | `%22.5` |
| `8-11 m` | `%15.1` | `%12.3` | `%10.5` |
| `11-15 m` | `%7.6` | `%7.5` | `%8.3` |

Çarpışma **kısa** rotalarda daha sık ve desen üç ödül tasarımında da aynı. Bu
belgede "çarpışma oranı eğitim boyunca arttı (`%15 -> %20`)" diye kaydedilen
gözlem survivorship'tir: hedef bölgesine ulaşan episode oranı `%53 -> %74`
yükseldi, **ulaşanlar arasında** çarpışma sabit kaldı (`%27, %12, %15, %18,
%17, %18`).

`target_progress` bu belgedeki forma (`5.0/m`, normalize edilmez) geri alındı.
`hover_height 1.5` ve `target_visible` korundu.

**Bu belgedeki `final` checkpoint hâlâ projenin en iyi D1 politikasıdır**
(`%42.5`, waypoint'e `0.591 m`): `runs/d1-seed20260912/final_model.zip`.

## REV-030: bu belgedeki bütün sonuçlar kırık bir observation üzerinde alındı

`quadrotor.xml`'deki zemin düzlemi görsel olarak yalnız `10 x 10 m`'dir. D1
workspace'i `20 x 20 m` olduğu için bu belgedeki koşularda:

```text
HEDEF zeminin dışında                     : %65 episode
rotanın zemin dışı kısmı                  : medyan %39
kameranın hiç zemin görmediği D1 pozu     : %52
```

Onaylı encoder legacy `6 x 5 m` workspace'te eğitildi ve orada bu oran `%0`'dır.
Yani bu belgedeki `%25-42.5` başarı, encoder'ın hiç görmediği bir girdi
dağılımında ölçülmüştür ve D1 zorluğunun gerçek ölçüsü değildir. Başarının
seed'e aşırı duyarlılığı (`%25` vs `%42.5`) da bununla açıklanır: episode'un
merkeze yakın düşüp düşmemesine bağlıdır.

Düzeltme ve ölçümler REV-030'dadır. Düzeltilmiş ortamdaki ilk koşu:
`runs/d1-rev030-seed20260912`.
