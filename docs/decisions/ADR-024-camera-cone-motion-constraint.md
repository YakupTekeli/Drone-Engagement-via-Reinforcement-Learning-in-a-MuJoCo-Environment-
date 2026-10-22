# ADR-024: Kamera Konisi Hareket Kısıtı ("Baktığın Yöne Uç", `vision-action-v3`)

**Durum:** Kabul edildi (kullanıcı kararı, 29 Eylül 2026)  
**Tarih:** 29 Eylül 2026  
**İlişki:** ADR-020 holonomik action'ın kör alan varsayımını
`EbabilVisionNavigation-v2` için daraltır; ADR-020 `v0`/`v1` için aynen
geçerlidir. REV-051'i (çarpışma kısmı) kapatmayı hedefler. ADR-023
observation sözleşmesi (`vision-v4`) değişmez; action uzayının şekli
değişmez, **anlamı** değişir.

## Bağlam

ADR-020 action'ı holonomik yaptı: vision-action-v1 yalnız ileri gidebildiği
için araç hedef üstünde ince yanal düzeltme yapamıyor, düzeltmek için dönmek
hedefi kadrajdan çıkarıyordu. ADR-020 kör alan riskini açıkça bıraktı:
"Politika artık göremediği yöne uçabiliyor. Bu beklenen bir sonuçtur ve
deneyin sorusudur; ölçülmeden yorum yapılmaz." ve bedelin
`collision_penalty` ile öğretilmesini bekledi.

REV-051 bu soruyu ölçtü (dedektör politikası, seed `20260912`, 600k adım,
held-out, 40 episode deterministik): çarpışma `%35`; 14 çarpışmanın 13'ünde
son 10 adımdaki hız vektörü kameranın yatay görüş açısının (`±37.5` derece)
dışındaydı (5 yana, 8 geriye çapraz veya geri). Politika kör yöne uçmanın
bedelini ödülden öğrenmedi. Kullanıcı kararı: "baktığın yöne uç" kısıtını
uygula.

## Karar

### 1. Durum bağımlı yatay hız kısıtı

Action şekli (`Box(-1, 1, (4,))`), ölçekleri ve ADR-020 büyüklük kırpması
aynen kalır. Kırpmadan sonra gövde çerçevesindeki yatay hız komutu
`v = (v_ileri, v_yanal)` şöyle işlenir:

- **Seyir ve arama** (`transit`, `search`, `acquired`): `v`, kameranın optik
  ekseni etrafında yarım açısı `30` derece olan koniye
  (`{x >= 0, |y| <= tan(30) x}`) Öklid izdüşümüyle taşınır. Koni içindeki
  komut değişmez; yana komutun koni kenarına izdüşümü kalır (daha yavaş,
  ileri-çapraz); kenar ışınından `90` dereceden fazla uzak komut (açı
  `> 120` derece, yani geriye doğru) sıfırlanır. Yön değiştirmek için yaw
  kullanılır.
- **Terminal** (`approach`, `settle`, `hold`): koni içi bileşen korunur, koni
  dışı bileşen `0.5 m/s` ile sınırlanır. ADR-020'nin gerekçesi olan hedef
  üstünde ince yanal/geri düzeltme mümkün kalır; hold hız sınırı zaten
  `0.25 m/s`'dir.

Dikey hız ve yaw-rate etkilenmez. Yarım açı yatay FOV'un yarısından
(`37.5` derece) `7.5` derece dardır: araç gövdesinin ve hız gecikmesinin payı.

### 2. Sürüm ve raporlama

- `vision_navigation_v4.toml` `[motion_constraint]` (`enabled = true`,
  `half_angle_deg = 30.0`, `terminal_blind_speed_m_s = 0.5`,
  `relaxed_states = ["approach", "settle", "hold"]`). `v0` ve `v3` config'lerinde
  bölüm yoktur, varsayılan kapalıdır (test).
- Kısıt açıkken `v2` ortamı `action_version = "vision-action-v3"` raporlar;
  `info["motion_constrained_fraction"]` komutun değiştirildiği alt adım payıdır.
- Uygulama `EbabilVisionNavigationEnv._constrain_horizontal_body` kancasıdır;
  tabanda kimlik fonksiyonudur, `v0`/`v1` davranışı değişmez.

### 3. Politika yeniden eğitilir

Action anlamı değiştiği için seed `20260912` holonomik politikası bu ortamda
kullanılmaz; aynı PPO tarifiyle yeniden eğitilir. Holonomik sonuç (REV-051)
karşılaştırma baseline'ıdır.

## Doğrulama kapısı

1. Saf fonksiyon: koni içi komut değişmez; yana komut kenara izdüşer; açı
   `> 120` derece komut sıfırlanır; izdüşüm hızı artırmaz ve idempotenttir;
   terminal modda koni dışı bileşen `0.5 m/s` ile sınırlıdır (test).
2. Ortam: `transit`'te sürekli geri komutla 20 adım boyunca gövde ileri hızı
   `> -0.05 m/s` kalır; `approach`'ta geri komut `0.5 m/s` ile sınırlanır;
   `action_version = "vision-action-v3"` (test).
3. `v0`/`v3` config'lerinde kısıt kapalı, ADR-020 testleri değişmeden geçer.
4. Davranışsal (Dilim D): çarpışma oranı holonomik politikanın `%35`'inin
   altına iner ve çarpışmalar görüş açısı içinde yoğunlaşır; ADR-023 §10
   kapısı (`arrival >= %80`, `çarpışma <= %10`) ölçülür.

## Revizyon 1 (30 Eylül 2026, kullanıcı onayı)

İlk ölçümde `approach` durumunda izin verilen kör yanal hareket 7 çarpışmanın
3'ünü üretti (`0.60-0.74 m/s`, `83-99` derece). `approach` hedeften `2.5 m`
uzakta başlar ve orada engel bulunabilir. Serbest durumlar `settle` ve `hold`
ile sınırlandı (`relaxed_states = ["settle", "hold"]`); `approach` artık
seyirle aynı koni kısıtına tabidir. Görülen engelden kaçınamama için ödül
değiştirilmedi; önce eğitim süresi sınanır: seed `20260912` politikası
`final_model.zip` ve `vecnormalize.pkl`'den devam ettirilerek `+600k` adım
(`1.2M` toplam) eğitilir (`runs/detector_cone2_d1a_seed20260912_20260930`).

## Revizyon 2 (30 Eylül 2026): menzil tabanlı hız sınırlayıcı

Seed `20260913` (koni R1 + yardımcı açık eğitildi) held-out 40 episode'da
`%25` çarpıştı; 10 çarpışmanın 9'u seyir/`acquired` durumunda, engel kutusu
görünürken, `1.3-1.9 m/s` hızla ve görüş açısı içinde/kenarında hareketle
oldu. Politika görülen engeli kaçınacak kadar erken yavaşlamıyor ve seed'ler
arası fark büyük (`%10` vs `%25`). Yeniden eğitim yerine deploy edilebilir bir
güvenlik katmanı eklendi (`SpeedGovernorConfig`, `[speed_governor]`):

- Kadrajın orta koridorunda (`|cx_norm| <= 0.4`) engel yuvası varsa yatay hız
  büyüklüğü `1.0/s x (menzil - 0.6 m)` ile sınırlanır (taban `0.15 m/s`).
  Yön korunur; politika dönmek için zaman kazanır.
- Menzili geçersiz (tabanı kadraj dışı = çok yakın) yuva `0.5 m` sayılır.
- Girdi yalnız `vision-v4` yuvalarıdır (dedektör + FC attitude/irtifa);
  ayrıcalıklı bilgi yoktur. Koni izdüşümünden sonra, yardımcı komutları dahil
  her durumda uygulanır. `info["speed_governor_limit_m_s"]` raporlanır.
- `v0`/`v3` config'lerinde alan yoktur (kapalı).

Doğrulama: birim testler (koridor dışı yuva sınır üretmez, menzil-hız
eşlemesi, geçersiz menzil tabanı, hedef yuvası sayılmaz) ve ortam testi
(öndeki sahte engelle ileri komut sınırlanır).

**Ölçüm (30 Eylül 2026, yeniden eğitim olmadan, held-out 40 episode):**
seed `20260912` çarpışma `%10 -> %5` (başarı `%87.5`), seed `20260913`
`%25 -> %22.5` (başarı `%77.5`). İki ek kural denendi ve **geri alındı**:
dönüş-hız bağlaması (`turn_penalty 0.8`) çarpışmayı artırdı (`%5 -> %12.5`,
`%22.5 -> %45`); kenardaki yakın engelleri de sayma artırdı (`%5 -> %12.5`,
`%22.5 -> %30`). Yeniden eğitilmemiş politikaya eklenen ek yavaşlama
davranışı bozuyor; `turn_penalty` varsayılan `0`, kod eğitimle birlikte
açılacak ablation için korunur.

## Ölçülen sonuçlar

**30 Eylül 2026, seed `20260912`, 600k adım, held-out görsel alan, 40 episode
deterministik** (`runs/detector_cone_d1a_seed20260912_20260929/evaluation_heldout_final_det.json`),
aynı tarifin holonomik sonucuyla (REV-051) karşılaştırmalı:

| Metrik | holonomik (ADR-020) | koni kısıtlı (ADR-024) | Kapı |
|---|---:|---:|---|
| arrival | `%55` | `%70` | `>= %80` |
| çarpışma | `%35` | `%17.5` | `<= %10` |
| hedef kaybı abort | `%55` | `%42.5` | |
| süre dolması | `%0` | `%37.5` | |
| hover başarısı | `%0` | `%0` | raporlanır |
| ortalama episode getirisi | `-25.2` | `+16.5` | |

Çarpışmaların yönü (son 10 adım ortalama hız, gövdeye göre): holonomikte
14'ün 13'ü görüş açısı dışındaydı; koni kısıtlıda 7 çarpışmanın 4'ü seyirde
görüş açısı içinde (`-10` ile `-30` derece, `1.35-1.99 m/s`, son 10 adımın
hepsinde engel kutusu vardı), 3'ü `approach` durumunda yana (`83-99` derece,
`0.60-0.74 m/s`, izin verilen terminal kör hareket). Süre dolan 15 episode'un
hepsi hedefe varmıştı (en yakın mesafe medyanı `0.49 m`, adımlarının `%61`'i
`approach`'ta): araç hedef çevresinde kalıyor fakat `hold`'a yerleşemiyor
(REV-043 ekseni).

Sonuç: kısıt kör yöne uçuş çarpışmalarını ortadan kaldırdı (seyirde görüş
dışı çarpışma yok); kalan çarpışmalar (a) görülen engelden kaçınamama ve (b)
`approach`'ta izin verilen kör yanal hareketten.

## Sonuçlar ve sınırlar

- Araç yana/geri kaçamaz; bir engelden kurtulmak için dönmek zorundadır.
  Dönüş sırasında ileri hız sıfıra yakın tutulursa bekleme süresi artar.
- Terminal durumlarda `0.5 m/s`'lik kör hız hâlâ görülmeyen bir engele
  yaklaşabilir; hedef çevresi `endpoint_clearance_m` ile açık tutulduğu için
  risk düşüktür, sıfır değildir.
- Hedef üstünde tutunamama (REV-043, REV-051 madde 2) bu kısıtla çözülmez;
  ayrı eksendir.
- Kısıt simülasyonda bir davranış sınırıdır; gerçek uçuş güvenlik garantisi
  değildir (ADR-011 tezgâh kapısı ve SITL/HITL geçerli).
