# ADR-025: Terminal Hover Yardımcısı (Dedektör Kutusuyla Klasik Yerleşme)

**Durum:** Kabul edildi (30 Eylül 2026; kullanıcının "kontrol sende, bizi
sonuca ulaştır" yetkisiyle alındı, kullanıcı onayına açık)  
**Tarih:** 30 Eylül 2026  
**İlişki:** ADR-011 mimarisini (RL politikası + klasik kademeli kontrol)
terminal faza genişletir. ADR-012 hover başarı ölçütü ve ADR-017 durum
makinesi **değişmez**. ADR-023 (dedektör + politika) ve ADR-024 (kamera
konisi kısıtı) üzerine kurulur. REV-043'ün ölçtüğü "beş hover koşulu birlikte
sağlanmıyor" duvarını ve REV-051'in tutunamama kısmını hedefler.

## Bağlam

REV-043'ten beri hiçbir politika (etiketli encoder, VAE, dedektör; holonomik
ve koni kısıtlı) tek bir başarılı hover üretemedi. Son ölçümde (ADR-024,
`1.2M` adıma devam eğitimi sırasında) eğitim episode'larının `%68`'i hedefe
varıp süre dolana kadar çevresinde dolanmakla bitiyordu: politika hedefe
ulaşmayı öğrendi, fakat `0.65 m` içinde `0.25 m/s` altında, doğru irtifada,
`10` dereceden az eğimle `3 s` durmayı öğrenemedi. Bu, RL'in örnek verimliliği
açısından zor bir hassas kontrol problemi; klasik kontrol için ise düz bir
regülasyon problemidir.

Gerçek araçta da mimari zaten kademelidir (ADR-011): politika hız komutu
verir, altında klasik kontrolcü vardır. Terminal yerleşmeyi klasik bir
görsel-servo döngüsüne vermek bu mimariyle tutarlıdır ve yalnız deploy
edilebilir girdiler kullanır: dedektörün hedef kutusu ile uçuş kontrolcüsünün
attitude ve irtifa tahmini.

## Karar

### 1. Terminal durumlarda politikanın action'ı yardımcıyla değiştirilir

`EbabilVisionNavigation-v2` ortamında `terminal_assist.enabled = true` iken
hover durumu `approach`, `settle` veya `hold` ise politikanın action'ı
kullanılmaz; yardımcının komutu uygulanır. Diğer durumlarda (transit, search,
acquired) politika sürer. Reward, termination ve hover ölçütü değişmez;
`info["terminal_assist_active"]` ve `info["terminal_assist_fraction"]`
raporlanır.

### 2. Hedef merkezi tahmini (`missions/terminal_assist.py`)

- Kutunun üst kenarı (uzak kenar) ve merkez sütunu zemin düzlemine
  izdüşürülür (ADR-023 §4 ile aynı kamera geometrisi; düzey çerçeve
  yerçekimi yönünden kurulur, yaw korunur).
- Kutunun alt kenarı kadraj içindeyse (`bottom_margin_px = 4`) yakın kenar
  da izdüşürülür ve merkez iki kenarın ortasıdır.
- Alt kenar kadraj dışındaysa (hedef üstündeyken olağan durum: kamera
  `0.40 m`'den yakını görmez) merkez, uzak kenar menzilinden
  `half_extent_guess_m = 1.2 m` (hizalı `1.0`, köşegen `1.41`) geri alınır;
  yanal konum aynı sütunda o menzile karşılık gelen satırdan okunur.
- Ölçülen hata (MuJoCo projeksiyonuyla sentetik kutu, güvenli bölge içinde
  rastgele `397` poz, her yön ve yaw): medyan `0.24 m`, p90 `0.50 m`, en kötü
  `0.77 m`; hedef önde iken (`±45` derece) p90 `0.32 m`. Yanal hata
  döndürülmüş hedefin kesik kutusundan gelir ve yaw kontrolü hedefi öne
  aldıkça küçülür.

### 3. Komut

Düzey çerçevede `v_xy = kp_horizontal * offset` (`kp 1.0`, sınır `0.5 m/s`),
`v_z = kp_vertical * (hover irtifası - z)` (`kp 1.0`, sınır `0.3 m/s`),
yaw-rate `-kyaw * cx_norm` (`0.8`, sınır `0.5 rad/s`; hedefi yatayda ortalar).
Hedef kutusu yoksa yerinde hover (sıfır yatay hız, irtifa tutma). Komut action
uzayına normalize edilir; ADR-024 kısıtı (`settle`/`hold`'da `0.5 m/s` kör
serbestlik) ondan sonra uygulanır.

### 4. Eğitim ve evaluation

Yardımcı `vision_navigation_v4.toml`'de açıktır; evaluation ve replay
script'leri `--terminal-assist on|off` ile config'ten bağımsız aç/kapatır.
Yardımcı kapalı ölçüm, RL'in tek başına ulaştığı yerleşme yetkinliğinin
ablation'ıdır (ADR-023 §11'e satır eklenir). Devam eğitimi
(`runs/detector_cone2_d1a_seed20260912_20260930`) yardımcısız başlatıldığı
için yardımcısız sürer; sonraki seed'ler yardımcı açıkken eğitilir ki
politika teslim etmeyi (hedefi görünür tutarak `approach`'a girmeyi) öğrensin.

## Doğrulama kapısı

1. Düzey çerçeve ve zemin noktası geometrisi; tahmin hatası dağılımı
   (`tests/test_terminal_assist.py`: 5 sabit poz + 60 rastgele poz,
   max `< 0.85 m`, p90 `< 0.6 m`).
2. Komut yönleri, hız sınırları, hedef yokken yerinde hover (test).
3. Ortam: yardımcı yalnız `states` içindeki durumlarda devreye girer; `info`
   alanları (test).
4. Davranışsal: ADR-023 §10 protokolüyle hover başarısı raporlanır; kapı
   `arrival >= %80`, `çarpışma <= %10` değişmez.

## Ölçülen sonuçlar

**Ön ölçüm (30 Eylül 2026):** koni kısıtlı seed `20260912` (`600k`, ADR-024)
politikası yardımcısız 12 held-out episode'da 0 başarı; yardımcı açıkken
aynı seed'lerde `10/12` hover başarısı, arrival `%83`, çarpışma `%0`,
ortalama getiri `+92.3` (yardımcısız `+16.5`). İki başarısızlık hedefe hiç
yaklaşamayan episode'lardı (en yakın `6-7 m`).

**40 episode, held-out, deterministik (30 Eylül 2026), koni kısıtlı `600k`
politikası** (`runs/detector_cone_d1a_seed20260912_20260929/evaluation_heldout_final_det_assist.json`):

| Metrik | yardımcısız | yardımcı açık | Kapı |
|---|---:|---:|---|
| hover başarısı | `%0` | **`%77.5`** (31/40) | raporlanır |
| arrival | `%70` | `%77.5` | `>= %80` |
| çarpışma | `%17.5` | `%10.0` | `<= %10` |
| süre dolması | `%37.5` | `%0` | |
| ortalama getiri | `+16.5` | `+85.7` | |
| ilk `hold`'a giriş (ortalama adım) | yok | `353` | |

Hedefe varan (`<= 2.5 m`) 31 episode'un **hepsi** başarıyla bitti. Kalan 9
başarısızlık seyir fazındadır: 4 çarpışma, hedefi `5-7 m`'den görüp
sonra kadrajdan çıkarınca `2.5 s` içinde bırakılan 4 erken abort
(`acquired` durumunda ADR-017 kuralı), 1 sınır dışı. Yardımcı REV-043 duvarını
aştı; kapılar seyir fazının iyileşmesini bekliyor (devam eğitimi, ADR-024
Revizyon 1).

**ADR-017 Revizyon 1 ile (aynı gün, aynı politika ve seed'ler):** hover
başarısı **`%87.5`** (35/40), arrival `%87.5` (kapı `>= %80` **geçti**),
çarpışma `%10.0` (kapı `<= %10` **geçti**, sınırda), abort `%0`, ortalama
getiri `+93.1`. Erken tespit kaybıyla bırakılan 4 episode'un hepsi başarıya
döndü. Kalan 5 başarısızlık: 4 çarpışma (seyrin `24-151`. adımında, hedeften
`9-16 m` uzakta) ve 1 sınır dışı. Tek seed; ADR-023 §10 üç seed protokolü
sürüyor.

## Sonuçlar ve sınırlar

- Hover artık iki bileşenin ortak işidir: politika hedefi görünür tutarak
  `2.5 m` içine getirir, yardımcı yerleşir. Politika tek başına
  değerlendirilirken yardımcı kapatılır.
- Tahmin, hedefin `2 x 2 m` kare olduğu ve zeminde durduğu bilgisini
  kullanır (görev bilgisi, ayrıcalıklı simülatör bilgisi değil).
- Yardımcı hedef kutusu olmadan yerinde hover eder; dedektör hedefi
  `2.5 s` boyunca kaybederse ADR-017 abort'u aynen çalışır.
- Simülasyon sonucudur; gerçek uçuş güvenliği iddiası yoktur.
