# ADR-017: Terminal Hover Durum Makinesi ve Reward Sözleşmesi

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-012 terminal görev kapsamını, ADR-016 hover geometrisini ve
ADR-013 segment bütçesini uygular. ADR-015 observation sözleşmesini
(bugün ADR-021 `vision-v2`)
**değiştirmez**. ADR-008 legacy `EbabilNavigation-v0` sözleşmesi dokunulmadan
korunur.

## Bağlam

ADR-015 `EbabilVisionNavigation-v0` ortamını kurdu fakat başarı semantiği legacy
nokta/hold davranışında kaldı: hedef noktasının `0.20 m` yarıçapında `0.50 s`
bekleme. ADR-012 ise terminal görevi hedef bölgenin güvenli iç alanı üzerinde
kesintisiz `3.0 s` hover olarak tanımlar ve çarpmayı hiçbir durumda başarı
saymaz. ADR-016 ayrıca hold koşulunun hedef merkezi görünürlüğüne
dayandırılamayacağını ölçümle sabitledi.

## Karar

### 1. Durum makinesi açık bir enum ve geçiş tablosudur

`ebabil.missions.hover_state` modülü `HoverState` enum'unu ve
`ALLOWED_HOVER_TRANSITIONS` tablosunu tanımlar:

```text
transit -> acquired -> approach -> settle -> hold -> success
   \          \          \          \         \
    \-------- stale_frame / target_lost -----> abort
```

Zincir durumları (`transit`, `acquired`, `approach`, `settle`, `hold`) koşullardan
türetilir ve koşullar iç içe geçmelidir. Bu nedenle araç zincirde bir adımda
birden fazla seviye ilerleyebilir veya geri düşebilir. İki kısıt katıdır:

- `SUCCESS` yalnız `HOLD` durumundan girilir ve yalnız kesintisiz hold sayacı
  tamamlandığında üretilir. Durum makinesi bunu ayrıca runtime invariant olarak
  doğrular.
- `SUCCESS` ve `ABORT` soğurucudur; terminal durumdan sonra `update` çağrısı
  hata verir.

İç içe geçme, `terminal_hover.approach_radius_m` değerinin güvenli iç bölge
köşegenini aşması koşuluna bağlıdır ve bu koşul config doğrulamasında
uygulanır.

### 2. Görsel süreklilik yalnız `target_score` ile kapılanır

Tespit ve kayıp histerezisli iki eşik kullanır:

| Parametre | Değer | Anlam |
|---|---:|---|
| `acquire_score` | `0.60` | Tespit eşiği |
| `loss_score` | `0.40` | Kayıp eşiği |
| `acquire_seconds` | `0.30 s` | Kesintisiz tespit süresi (`6` adım) |
| `grace_seconds` | `2.50 s` | Tespit sonrası tolere edilen kesintisiz kayıp (`50` adım) |

Histerezis zorunludur: `loss_score >= acquire_score` config doğrulamasında
reddedilir, aksi halde skor gürültüsü durum titremesi üretir. İki eşik arasındaki
skor ne tespit ne kayıp sayılır.

ADR-016 uyarınca `target_center_xy` kapı olarak kullanılmaz. `transit`
durumunda `target_lost` abort'u üretilmez; hedef hiç tespit edilmemiştir.
`stale_frame` her durumdan abort üretir.

**Skor ölçeği:** encoder'ın ham sigmoid çıktısı `[0, 1]` aralığındadır ve durum
makinesi bu ham değeri kullanır. `[-1, 1]` kaydırması yalnız actor
observation'ındaki `vision_aux` kopyasına uygulanır. İki ölçeğin karıştırılması
`[0, 1]` aralığını `[0.5, 1]`'e sıkıştırır; bu durumda hedef asla kaybedilemez ve
tespit eşiği gerçek değerinin yarısına düşer. Bu ayrım testle sabitlenmiştir.

### 3. Başarı kararı privileged ground truth ile verilir

Hover koşullarının tamamı birlikte sağlanmalıdır:

| Koşul | Kaynak | Değer |
|---|---|---:|
| Araç merkezi güvenli iç bölgede | privileged | `+-0.65 m` |
| İrtifa hatası | privileged | `<= 0.20 m` |
| Doğrusal hız | privileged | `<= 0.25 m/s` |
| Roll/pitch | privileged | `<= 10 derece` |
| Açısal hız | privileged | `<= 0.35 rad/s` |
| Kesintisiz hold | privileged | `3.0 s` = `60` adım |

Bu bilgiler actor observation'ına girmez. Sözleşme o dilimde değişmemişti;
bugünkü hâli ADR-021 `vision-v2`'dir:
terminal durum yalnız `info` içinde raporlanır. Durum privileged geometriden
türetildiği için observation'a eklenmesi localization sızıntısı olurdu. Gerçek
sistemde aynı fazın görsel kestirim ve VIO'dan üretilmesi ayrı bir iş paketidir.

Hold koşulu bozulduğu anda sayaç sıfırlanır. Fiziksel failure (çarpışma,
workspace ihlali, minimum irtifa, tutum ihlali, geçersiz state) durum makinesini
ilerletmez ve hiçbir koşulda başarı üretmez.

### 4. Terminal sebepler ayrıktır

| `termination_reason` | Anlam |
|---|---|
| `hover_success` | Kesintisiz hold tamamlandı |
| `hover_abort_target_lost` | Tespit sonrası grace aşıldı |
| `hover_abort_stale_frame` | Kamera karesi yaş sınırını aştı |
| `obstacle_collision` / `ground_collision` | Fiziksel çarpışma |
| `out_of_bounds` / `minimum_altitude` / `unsafe_attitude` / `invalid_state` | Fiziksel failure |
| `time_limit` | Segment bütçesi doldu (`truncated=True`) |

### 5. Reward bileşenleri ayrı loglanır

`TerminalHoverRewardBreakdown` on üç bileşen üretir ve `total` bunların
toplamıdır:

| Bileşen | Ağırlık | Uygulama koşulu |
|---|---:|---|
| `target_progress` | `5.0` | Hover anchor'a 3B yaklaşma farkı, metre başına sabit |
| `target_visible` | `0.005` | Hedef kamerada görünürken, adım başına |
| `progress_refund` | — | Fiziksel başarısızlıkta toplanan progress'i geri verir |
| `target_alignment` | `0.20` | Yalnız hedef görünürken ve `acquired`/`approach` durumunda |
| `settle_speed` | `0.50` | Yalnız güvenli iç bölgede; hız karesi cezası |
| `hover_hold` | `0.40` | Yalnız `HOLD` durumunda, adım başına |
| `proximity` | `0.05` | Privileged minimum açıklık (`0.75 m` eşiği) |
| `smoothness` | `0.02` | Action farkı karesi |
| `collision` | `25.0` | Çarpışma cezası |
| `out_of_bounds` | `15.0` | Workspace ihlali cezası |
| `success` | `30.0` | Hover başarısı bonusu |
| `abort` | `10.0` | Abort ve diğer fiziksel failure cezası |
| `time` | `0.01` | Adım başına zaman cezası |

`hover_hold` yalnız `HOLD` durumunda verilir, yani hedef dışında bekleyerek
biriktirilemez.

**Düzeltme (REV-022).** Bu bölümün ilk yazımı "toplamı `0.40 * 60 = 24.0`
değerini aşamaz ve `success_bonus` altında kalır" diyordu; bu **yanlıştı**. Hold
sayacı koşul bozulduğunda sıfırlanır fakat durum yeniden `HOLD` olabilir ve her
`HOLD` adımı yeniden reward verir; sayacın sıfırlanması verilmiş reward'ı geri
almaz. `24.0` yalnız **kesintisiz** bir dizinin değeridir
(`uninterrupted_hold_reward`). Episode üst sınırı episode uzunluğuyla verilir
(`episode_hold_reward_bound`): `4 m` rota için `400 * 0.40 = 160.0`. D0-V
koşusunda gözlenen başarı episode'u `103` `HOLD` adımıyla `41.20` topladı.

`target_alignment` `settle` ve `hold` durumlarında kapatılır. ADR-016 hedef
merkezinin hold pozunda kadraj dışında kaldığını ölçmüştür; bileşen açık
kalırsa sistematik kenar kayması ödüllenir.

`collision` ve `out_of_bounds` ayrı bileşenlerdir; legacy reward bunları tek bir
`failure` bileşeninde topluyordu.

#### Değişiklik (12 Eylül 2026): görünürlük bileşeni; progress değişmedi

**`target_progress` bu ADR'nin özgün formunda kalmıştır** (`5.0 * delta_d`,
rotaya normalize edilmez). 12 Eylül 2026'da REV-026 bulgusuna dayanarak iki kez
normalize edildi, iki koşuda ölçüldü ve **geri alındı**. Süreç REV-029'da
kayıtlıdır; özeti:

| Koşu | progress formu | waypoint'e mesafe | Başarı |
|---|---|---:|---:|
| 1 | `5.0/m` sabit (bu ADR) | `0.591 m` | **`%42.5`** |
| 3 | `15.0 * dd / L` | `0.991 m` | `%22.5` |
| 4 | `10.0/L` + `5.0/m` (`< 2.5 m`) | `1.709 m` | `%0.0` |

REV-026 "uzun rotada engele çarpmak kârlı olur" diyordu. Ölçüm tersini gösterdi:
çarpışma **kısa** rotalarda daha sık (`5-8 m`'de `%23`, `11-15 m`'de `%8`) ve bu
desen üç ödül tasarımında da aynı kaldı. Eğitim boyunca gözlenen `%15 -> %20`
artış survivorship'ti (hedefe ulaşan episode oranı `%53 -> %74`; ulaşanlar
arasında çarpışma sabit).

**Doğru invariant yoğunluk üzerindedir, toplam üzerinde değil.** Çarpışma
episode'u bitirdiği için politikanın kıyaslaması kısa yolun yerel kazancıdır:

```text
w_metre * dolaşma_mesafesi <= collision_penalty      5.0 * 5 m = 25.0 <= 25.0
```

Bu kısıt `L`'ye hiç bakmadığı için rota uzunluğundan doğal olarak bağımsızdır;
normalizasyon gereksizdi. Uygulama: `detour_saving_for()`.

**Açık kalan gerçek ölçekleme sorunu.** Normalizasyonun çözmeye çalıştığı ayrı
bir sorun gerçektir: `5.0/m` ile D3'te (`100 m`) episode toplamı `~500`,
`success_bonus` ise `30.0`'dır ve başarı sinyali boğulur. Çözümü progress'i
bölmek **değil**, `success` ve `hold` ödüllerini mesafe seviyesine göre
ölçeklemektir. D3 öncesinde ele alınmalıdır; D1'de bağlayıcı değildir.

#### Yeni bileşen: `progress_refund` (REV-033)

`target_progress` bir kez toplanıp **bankada kalıyordu** ve başarıyla
gerekçelendirilmesi gerekmiyordu. Ölçülmüş sonuç: hedefe uçup yere çakılmak
`+7.31` ödül veriyordu (`178/300` episode), bütçeyi doldurmak `-36.81`. Yani
**ölmek kârlıydı** ve politika bunu öğrenmişti.

```text
progress_refund = -(banked_progress + o adımın progress'i)   fiziksel başarısızlıkta
                = 0                                           diğer hâllerde
```

Fiziksel başarısızlık: `obstacle_collision`, `ground_collision`,
`out_of_bounds`, `minimum_altitude`, `maximum_tilt`.

`time_limit` ve hover abort'u **listede değildir** ve bu bilinçlidir: progress'i
her başarısızlıkta geri vermek, hiç başarı yokken episode getirisini
şekillendirme hiç verilmemiş hâline eşitler ve erken öğrenmeyi yok eder.

Sonuç teşvik yapısı:

```text
başarı               : progress + 30 + hold   -> en iyi
timeout              : progress - zaman       -> güvenli fakat yetersiz
fiziksel başarısızlık: 0 - ceza                -> her zaman zararlı
```

Bu defekt ADR-020'den önce de vardı: beşinci koşu (ileri-yalnız action)
`time_limit`'e yerleşip `+21.28` topluyordu. Holonomik action yalnız daha hızlı
bir bankalama yolu açtı (`460 -> 182` adım). Kök neden ödüldedir, action
uzayında değil.

#### Yeni bileşen: `target_visible` (REV-027)

Hedefi kadrajda tutmayı teşvik eden tek mekanizma `target_lost` abort'unun
`-10.0` cezasıydı; yani görünürlük yalnız *kaybedilince* fiyatlanıyordu.
`grace_seconds` REV-025 için `0.50 -> 2.50 s` yükseltildiğinde bu örtük baskı
kalktı ve ölçülen sonuç şu oldu: `target_lost` abort `%45 -> %0` fakat timeout
`%15 -> %70` ve başarı `%25 -> %5`. Kısa grace, kazara bir görünürlük
şekillendirmesi işlevi görüyordu.

`target_visible` bunu açık hâle getirir: hedef görünür olduğu her adımda
`+0.005`, bütün durumlarda (`target_alignment`'ın aksine `transit` ve `search`
dâhil — REV-027'nin en çok ısırdığı yer orasıdır). Ölçülen etkisi eğitimde
`target_lost` abort'unun `%30`'dan `%6`'ya inmesidir.

Ağırlığın üst sınırı türetilmiştir ve `validate()` içinde zorlanır:

```text
target_visible_weight < time_penalty   (0.005 < 0.01)
```

Eşit veya üzerinde bir değer, hedefi uzaktan seyrederek beklemeyi adım başına
net pozitif yapar ve görevi bitirme baskısını kaldırır — yani REV-027'yi
çözerken timeout'u besleyen yeni bir yol açar. `0.005` ile seyrederek beklemek
hâlâ `-0.005/adım`'dır. `hover_hold` bu sınıra tabi değildir ve olmamalıdır:
o yalnız hedefin üzerinde, `HOLD` durumunda verilir.

**Değişmeyenler:** başarı koşulu, `hold_seconds`, abort sebepleri, durum
makinesi geçiş tablosu, `grace_seconds = 2.50 s` ve diğer dokuz bileşenin
ağırlıkları. REV-025'in grace kalibrasyonu açıktır.

### 6. Legacy ortam davranışı değişmez

`EbabilNavigationEnv.step` iki override noktasına ayrılmıştır:
`_terminal_decision` ve `_compute_step_reward`. Varsayılan uygulamalar ADR-008
legacy semantiğidir; yalnız `EbabilVisionNavigationEnv` bunları geçersiz kılar.
Refactor'un davranış değiştirmediği legacy rollout parmak iziyle kanıtlanır.

## Revizyon 1 (30 Eylül 2026): arama yarıçapı dışında erken tespit kaybı

ADR-025 ölçümünde (dedektör politikası + terminal yardımcı, held-out 40
episode) 4 episode hedefi `5-7 m`'den görüp (`acquired`), kadrajdan çıkınca
`2.5 s`'de `target_lost` ile bırakıldı; araç henüz arama yarıçapının
(`search_entry_radius_m = 4 m`) dışındaydı. Bu kural uzaktan bir bakışı
bağlayıcı sayıyordu. `TerminalHoverConfig.far_loss_returns_to_transit`
(varsayılan `False`, `v0`/`v1` değişmez; `vision_navigation_v4.toml`'de
`true`): `acquired` durumunda kayıp, inanılan hedef mesafesi arama yarıçapının
dışındayken abort yerine `transit`'e döner ve tespit sayaçları sıfırlanır.
Arama yarıçapı içindeki kayıp aynen abort üretir; `approach`/`settle`/`hold`
kayıpları değişmez. `SUCCESS` ve hover ölçütü değişmez.

## Doğrulama kapısı

1. Geçiş tablosu: `SUCCESS` yalnız `HOLD`'dan, terminal durumlar soğurucu,
   `ABORT` her zincir durumundan.
2. `transit` durumundan çıkış yalnız kesintisiz `acquire_seconds` sonrası.
3. Histerezis: iki eşik arasındaki skor ne tespit ne kayıp üretir.
4. Beş hover koşulunun her biri tek tek bozulduğunda hold sayacının sıfırlanması.
5. Hold kesintisi sonrası orijinal deadline'da başarı üretilmemesi.
6. Grace içinde kayıp abort üretmez; grace aşıldığında `target_lost` abort'u.
7. `stale_frame` abort'unun her durumdan çalışması.
8. Terminal durumdan sonra `update` çağrısının reddi.
9. On üç reward bileşeninin ayrı loglanması ve `total == sum`.
10. Ham `[0, 1]` skor ile observation `[-1, 1]` ölçeğinin karıştırılmaması.
11. Çarpışmanın hiçbir koşulda başarı üretmemesi.
12. Legacy rollout parmak izinin değişmemesi.
13. Kısa PPO eğitimi ve ayrı deterministic evaluation smoke'u.
14. Progress yoğunluğunun metre başına sabit ve rota uzunluğundan bağımsız
    olması (REV-029).
15. `detour_saving_for(d) <= collision_penalty` invariantının `d <= 5 m` için
    sağlanması ve `d <= 0` değerlerin reddi.
16. `target_visible` her durumda görünürlükle verilir, görünmezken sıfırdır.
17. `target_visible_weight >= time_penalty` konfigürasyonunun `validate()`
    tarafından reddi ve görünürken bekleme adımının net negatif kalması
    (REV-027).

## Sonuçlar ve sınırlar

- Bu dilim mission segment zincirini veya D0-V–D2 uzun eğitimini başlatmaz.
- Terminal durum actor observation'ında yoktur; politika fazı `vision_aux`,
  `fc_state` ve `waypoint` üzerinden çıkarmak zorundadır. Deploy edilebilir
  görsel faz kestirimi ayrı bir iş paketidir.
- Reward ağırlıkları ölçülmüş değil başlangıç hipotezidir; curriculum eğitimi
  sonrası ablation ile revize edilecektir. D0-V koşusu ağırlıkların görevi
  öğrenmeye yettiğini gösterdi fakat `hover_hold`/`success_bonus` oranı
  ablation'ı yapılmadı (REV-022).
- D0-V koşusu uygulanabilir hold bandının (o zamanki `hover_height_m = 2.0`
  ile) `0.108 m` olduğunu ve başarının
  `3.6 cm` konum farkıyla belirlendiğini ölçtü (REV-023). Deterministik ve
  stokastik evaluation bu görevde birbirinin yerine kullanılamaz.
- Privileged başarı doğrulaması ve sentetik encoder gerçek uçuş güvenliği
  değildir. Görsel kestirim hatası ayrıca ölçülmelidir.
- `grace_seconds = 0.50 s` ölçülmüş VIO belirsizliğine değil, ADR-012'nin
  "sınırlı grace period" ilkesine dayanır; gerçek VIO drift ölçümüyle
  kalibre edilmelidir.
