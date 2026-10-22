# Ebabil MuJoCo: İlerleme Günlüğü

Bu belge yaşayan proje durumudur. En yeni kayıt üstte tutulur. Her agent iş dilimi sonunda bu dosyayı güncellemelidir.

## Güncel durum

| Alan | Durum |
|---|---|
| Planlama ve literatür | IMX477/Pixhawk 2.4.8/Orin NX ve GPS-denied kararıyla güncellendi; 22 Eylül 2026'da etiketsiz temsil literatürü eklendi (§2.16) |
| Faz 0: `.venv` ve çekirdek bağımlılıklar | Tamamlandı |
| Faz 1: Quadrotor MJCF modeli | Tamamlandı |
| Faz 2: Rotor dinamiği | Tamamlandı |
| Faz 3: Klasik kontrol | Tamamlandı |
| Faz 4: Sensörler ve sahne üretimi | Baseline tamamlandı |
| Faz 5: Gymnasium ortamı | Baseline tamamlandı |
| Faz 6: Engelsiz PPO baseline | Tek-seed pilot başarılı; legacy olarak donduruldu |
| Faz 6.5: Kamera/vision environment | `vision-v2` + `vision-encoder-v11` zinciri **legacy tanı baseline'ı** oldu (ADR-022) |
| Faz 7: Terminal hover state machine ve reward | Tamamlandı (ADR-017); REV-040..046 düzeltmeleri uygulandı |
| Faz 7: D0-V kamera tabanlı yerel politika | Tarihsel; `20/20` ölçümü `vision-v1`/`action-v1` ile alındı |
| Faz 7: D1 engelli + yaklaşık hedef | Dokuz koşu (13-14 Eylül); engelli her koşu `%0` başarı; REV-039..046 kendi ekseninde çalıştı, başarı üretmedi |
| **Faz 7.5: Görsel algı (ADR-022 VAE -> ADR-023 dedektör + politika)** | **26 Eylül 2026: kullanıcı kararıyla iki modelli algıya geçildi (ADR-023):** YOLO dedektör (hedef/engel kutuları) + PPO politikası, `vision-v4` (88 boyut), `EbabilVisionNavigation-v2`; plan §24 Dilim A (belgeler) tamam; Dilim B 1/2 tamam: zemin yansıması kapatıldı (kullanıcı kararı), dataset yeniden üretildi, YOLO etiketleri `runs/yolo_dataset_d1nr_20260926` (6528 kare, 26554 engel + 3874 hedef kutusu), 8 karede görsel kontrol hatasız. ADR-022 zinciri (onaylı `visual-vae-v1`, `v1` ortamı, `vision-v3`, Dilim 1-2) karşılaştırma baseline'ı olarak korunur; Dilim 3 eğitimi `528k / 600k` adımda iptal edildi, evaluation koşulmadı. **Dilim B ve C tamam (29 Eylül):** YOLO11n tek seed (kullanıcı kararı) geliştirme kapılarını geçti (engel recall `0.95-0.98`, hedef recall `0.97-0.98`, `~5.5 ms`); audit elipsoitlerde geçmedi (REV-050), kullanıcı kararıyla eğitim şekilleriyle sınırlı onay (`runs/detector_audit_v1_20260929/approval_manifest.json`). `EbabilVisionNavigation-v2` / `vision-v4` tamam: 20 test (tam paket 357), gerçek dedektörle smoke temiz, throughput `0.54 x v0` (tanı kapalı), PPO kuru koşusu `~100` adım/s. **Dilim D seed 1 (29 Eylül):** held-out arrival `%55` (kapı `%80`), çarpışma `%35` (kapı `%10`), başarı 0; çarpışmaların 13/14'ü kameranın görmediği yöne uçuştan (REV-051), hedef kayıplarının çoğu hedef üstünde tutunamamadan. Kullanıcı kararıyla ADR-024 "baktığın yöne uç" kısıtı uygulandı (`vision-action-v3`, 367 test); koni kısıtlı seed `20260912` held-out: arrival `%70` (önce `%55`), çarpışma `%17.5` (önce `%35`), hover başarısı 0; kapılar henüz geçilmedi. **30 Eylül, Dilim D kapanışı:** nihai yapılandırma (koni R1 + koridor hız sınırlayıcı + erken kayıp kuralı + ADR-025 terminal yardımcı) üç seed held-out ortalaması: hover başarısı `%84.2`, arrival `%84.2` (kapı `>= %80` geçti), çarpışma `%13.3` (kapı `<= %10` geçilmedi; seed'ler `%5 / %22.5 / %12.5`). Projede ilk başarılı hover'lar (önceki bütün politikalar `%0`). Demo: `runs/demo_20260930/`. |
| Depo yayını hazırlığı | 22 Ekim 2026: `docs/KURULUM.md` / `docs/GETTING_STARTED.md` (temiz venv'de doğrulandı), `docs/media/` demo GIF+mp4, README Demo/Kurulum bölümleri; onaylı dedektör + seed 20260912 politikası (11 dosya, ~7.8 MB) `.gitignore` istisnasıyla depoya dahil, `.gitattributes` hash zincirini korur. GitHub'a gönderme kullanıcıda. |
| Faz 7+: D1 yoğunluk merdiveni | ADR-023 Dilim D kapısını bekliyor |
| Faz 7+: D2-D4 mesafe merdiveni | ADR-018 ön koşullarını bekliyor |
| Faz 7+: D5 kompozisyon evaluation'ı | Bekliyor; eğitim rejimi değil |

**Güncel hedef (26 Eylül 2026):** Kullanıcı kararıyla algı iki modele
ayrıldı (ADR-023). Model 1: simülatörün otomatik kutu etiketleriyle eğitilen
YOLO11n dedektörü (sınıflar `obstacle`, `target`), ADR-014 deseniyle onay
zincirinden geçer. Model 2: PPO politikası; girdisi dedektör kutularının
`9 x 8` yuva vektörü (`vision-v4`, toplam `88` boyut), `fc_state`, yaklaşık
`waypoint` ve `timing`. Politika engel/hedef koordinatı görmez; hover durum
makinesinin görsel kapısı dedektörün hedef kutusudur; dedektör eğitim
döngüsünün içinde çalışır (throughput kapısı `>= 0.5 x v0`, yedek yol ölçülmüş
hata modeli). Action, reward ve waypoint sözleşmeleri değişmez. Sıra plan
§24'tedir: Dilim B (kutu etiketleri, dedektör eğitimi ve audit), C (`v2`
ortamı), D (D1a, 3 seed, `arrival >= %80`, `çarpışma <= %10`), E (ablation
ve yoğunluk merdiveni).

**Önceki hedef (22 Eylül 2026, ADR-022; baseline olarak korunur):** Yön değişti. Kullanıcı, actor algısının
simülatör etiketleriyle eğitilmiş bir encoder'dan gelmesini reddetti; karar
ADR-022'dir. Birinci kademe: yalnız actor RGB'siyle eğitilen **donmuş VAE
kodlayıcı**, latent düzeyinde `k = 4` yığın, ortam içi görsel domain
randomization ve `vision-v3` observation sözleşmesi
(`EbabilVisionNavigation-v1`). İkinci kademe: kodlayıcının PPO içinde
çözülmesi (plan A9). Etiketli `v11` zinciri aynı ortamda karşılaştırma
baseline'ı olarak kalır. Uygulama sırası ve kapılar
`docs/IMPLEMENTATION_PLAN.md` §23'tedir. **Dilim 1 (22 Eylül 2026, akşam):**
VAE modülü, probe ölçümü ve onay zinciri kodlandı ve testlendi; üç seed
kalite koşusu hedef recall (`0.60 / 0.69`, kapı `0.85`) ve validation harita
IoU (`0.27`, kapı `0.30`) kapılarında takıldı. ADR-022'nin saydığı bütün
etiketsiz kaldıraçlar (beta, latent, çözünürlük, fotometrik değişmezlik,
kontrastif) tek seed taramasıyla ölçüldü; hedef recall hiçbirinde `0.61`'i
geçmedi, `beta 1 + latent 128` diğer beş kapıyı geçti. Eşik düşürülmedi.
Kullanıcı REV-049 seçenek 1'i onayladı: ADR-022 Revizyon 1 ile hedef kapısı
eşikten bağımsız çözülebilirlik (AUROC `>= 0.80`, AP `>= 0.85`), harita için
en iyi eşikte IoU `>= 0.45`, latent `128`. İkinci kalite koşusu:
`runs/vae_quality_v2_20260922`: eşikten bağımsız kapıların hepsi üç seed'de geçti (AUROC `0.81-0.82`, en iyi eşikte harita IoU `0.50-0.54`); korunan sabit `0.5` eşiğindeki harita IoU iki seed'de `0.25-0.28` (kapı `0.30`), `quality_passed = False`. Kullanıcının 23 Eylül "ilerlemeye devam" onayıyla Revizyon 1b: bu kapı kaldırıldı; üçüncü kalite koşusu (`runs/vae_quality_v3_20260923`, v2 ile birebir aynı deterministik değerler) ve audit (`runs/vae_audit_v1_20260923`; held-out AUROC `0.81`, AP `0.89`, en iyi eşikte harita IoU `0.57`) geçti, onay manifesti üretildi. **Dilim 2 (23 Eylül 2026):** `EbabilVisionNavigation-v1` (`vision-v3`), ortam içi görsel randomization ve privileged görünürlük kapısı kodlandı, testlendi ve onaylı manifestle gerçek renderer'da smoke edildi; throughput `0.76 x v0` (kapı `>= 0.7`). Dilim 3: `ppo_latent_d1a.toml` ile seed `20260912` eğitimi `runs/latent_d1a_seed20260912_20260923` altında başlatıldı ve kullanıcı kararıyla `528k / 600k` adımda iptal edildi. Rollout ortalama ödülü `~-70`'ten (`75-95k`) `~-46`'ya (`170-200k`) yükseldi; evaluation noktaları `-40, -24, -61, -50, -25, -25, -16, -19, -10.5, -22` (`50k..500k`, 10 episode). Eğitim izleme monitörlerinde çarpışma payı `%50`'den `%43`'e indi, hedef kaybı abort'u `%19`'dan `%28`'e çıktı; başarı `0`. Checkpoint'ler `100k-500k` ve en iyi model (`450k`) duruyor; held-out evaluation koşulmadı, Dilim 3 kapısı ölçülmedi.

**13-14 Eylül 2026 koşularının özeti (REV-039..046).** `vision-encoder-v11`
ile 40 engelli D1 `%0` verdi; kök neden zorluk atlaması ölçüldü (REV-039,
koridorda ortalama `6.1` ardışık karar) ve `5 -> 10 -> 20 -> 40` engel merdiveni
kuruldu. 5 engelli basamakta dört yapısal defekt bulunup düzeltildi: hedefin
workspace kenarına manevra payı bırakmadan örneklenmesi (REV-040), holonomik
hareketin kamerayı hedeften ayırması (REV-041, `bearing_alignment`), yavaşlama
cezasının yalnız güvenli bölge içinde açık olması (REV-042, mesafeye ölçekli
hız payı), ödülün beş hover koşulundan yalnız ikisini şekillendirmesi
(REV-046, `settle_altitude/tilt/angular_rate`). Kamera pitch'i `45 -> 40`
dereceye çekildi ve hover yüksekliği `1.5 -> 1.1 m` yapıldı; onay zinciri
yenilendi (REV-044, `vision_audit_v5_20260914`). Hold süresi kademesi kuruldu
ve yanlış eksen olduğu ölçüldü (REV-045). Her düzeltme kendi ekseninde
çalıştı; son koşu `rev046c` yine `%0` başarı ve negatif ortalama ödül verdi.
Ölçülen duvar REV-043'tür: beş hover koşulu birlikte tek bir adımda bile
sağlanmıyor. Bu duvar algı yönteminden bağımsızdır ve ADR-022 onu çözmez;
kademe 1 bu yüzden başarıya ek olarak `arrival_rate` raporlar.

**Ölçülmüş tasarım sınırı:** waypoint kasıtlı olarak `2 m`'ye kadar yanlıştır ve
başarı **gerçek** hedefe göre ölçülür; kusursuz waypoint takibinin başarı tavanı
`(0.65/2)^2 = %10.6`'dır (ADR-019).

**Görev sözleşmesi (ADR-019).** Araç kendi konumunu hatasız bilir; hedefin
yalnız `R` yarıçaplı yaklaşık konumunu bilir. Bu değişmedi.

Açık ve bu dilimde dokunulmayan işler: ADR-009 üç-seed gereksinimi, iki
çözünürlüklü rota doğrulaması (D2 ön koşulu), REV-025 grace kalibrasyonu,
REV-043 hover ölçütü sorusu (ADR-012 düzeyinde, kullanıcı kararı bekler).

**Bekleyen donanım/görev girdileri:** IMX477'nin gerçek lens/carrier/sürücü
ayrıntıları, Pixhawk 2.4.8 board ID/FMU ve sensör envanteri, seçilecek tek
autopilot firmware'i (PX4 veya ArduPilot), Orin NX JetPack/L4T sürümü,
kamera montaj açısı, hedef alanın görsel/fiziksel tanımı, GPS-denied
localization yöntemi ve kamera-IMU zaman senkronizasyonu.

### Revize ilerleme sırası

| Sıra | İş paketi | Durum |
|---:|---|---|
| 1 | Güvenli hover, kamera/FC ve büyük alan ADR'leri | Tamamlandı |
| 2 | Kamera + hedef alan config/MJCF/headless capture | Tamamlandı |
| 3 | Hedef görünürlüğü/projection doğrulaması | Tamamlandı (sim) |
| 4 | Görsel dataset ve hafif encoder | Tamamlandı; ADR-022 ile legacy tanı baseline'ı |
| 5 | `EbabilVisionNavigation-v0` / `vision-v1` -> `vision-v2` | Tamamlandı; donduruldu |
| 6 | Hover state machine ve reward | Tamamlandı (ADR-017, REV-040..046) |
| 7 | D0-V yerel vision politikası | Tamamlandı (tek seed, tarihsel) |
| 7b | D1 ölçek dilimi + engel ekseni | Koşuldu; kapı geçilmedi (`%0`) |
| **7c** | **Görsel algı: ADR-022 VAE (baseline) -> ADR-023 YOLO dedektör + RL politikası** | **Dilim A, B, C tamam (29 Eylül 2026); dedektör şekil kısıtlı onaylı (REV-050); Dilim D (D1a eğitimi) sırada** |
| 7d | Kademe 2: kodlayıcı ince ayarı ve ablation'lar | 7c kapısını bekliyor |
| 8 | D1 yoğunluk merdiveni `5 -> 10 -> 20 -> 40` | 7c kapısını bekliyor |
| 9 | D2-D5 mesafe merdiveni ve kompozisyon evaluation'ı | Bekliyor |
| 10 | Robustluk, Jetson, SITL/HITL ve safety | Bekliyor |

## 22 Ekim 2026 — Depo yayını hazırlığı: kurulum belgeleri (TR/EN) ve demo medyası

### Amaç ve kapsam

Kullanıcı projeyi GitHub'a (`YakupTekeli/Drone-Engagement-via-Reinforcement-Learning-in-a-MuJoCo-Environment-`)
gönderecek. Öncesinde istenenler: projeyi sıfırdan kuran birinin izleyeceği
adım adım Türkçe ve İngilizce kurulum belgesi; proje dizinine ve README'ye
örnek bir demo videosu. Git bağlama ve push kullanıcıya aittir (11 Eylül
kararı), bu dilimde yapılmadı.

### Oluşturulan/değiştirilen dosyalar

```text
docs/KURULUM.md                      yeni: gereksinimler, klonlama, venv, kalite kapıları, ilk çalıştırma, viewer,
                                     runs/ artefact'leri (hazır paket yolu A / sıfırdan üretim yolu B), diğer
                                     platformlar, sorun giderme, proje yapısı, okuma sırası
docs/GETTING_STARTED.md              yeni: aynı içeriğin İngilizcesi
docs/media/demo_hover_bright.mp4     yeni: runs/demo_20260930/bright_seed12_ep_seed310004.mp4 kopyası (1.9 MB)
docs/media/demo_hover_bright.gif     yeni: README için 640 px, 8 fps, 96 renk (3.7 MB)
docs/media/README.md                 yeni: medya kaynağı, episode bilgisi, yeniden üretme komutu
README.md                            "Demo" (GIF + mp4 bağlantısı) ve "Kurulum" bölümleri; belge bağlantıları;
                                     eski venv snippet'i kurulum belgesine yönlendirildi
.gitignore                           build/, MUJOCO_LOG.TXT, .mypy_cache/ eklendi; runs/ -> runs/* ve onaylı
                                     dedektör zinciri + seed 20260912 politikası için 11 dosyalık istisna listesi
.gitattributes                       yeni: runs/** -text (hash zinciri satır sonu dönüşümünden korunur), ikili türler
```

### Alınan teknik kararlar ve gerekçeleri

1. **Belge, temiz ortamda denenmiş komutlardan yazıldı.** Scratchpad'de
   `python3.13 -m venv` ile sıfır sanal ortam kuruldu; `requirements.lock`
   + `--no-build-isolation -e .` sorunsuz; `pytest` 382 passed, 117
   subtests (52 s); ruff ve mypy temiz. İlk çalıştırma komutları
   (`inspect_model`, `run_hover_demo`, `smoke_navigation_env --mode clear`,
   `train_ppo --smoke`) saniyeler içinde geçti.
2. **`runs/` depoda olmadığı için iki yol tanımlandı.** Yol A: demo ve
   evaluation için gereken beş dosya (onay manifesti, audit ve quality
   raporları, `last.pt`, `final_model.zip`; toplam ~7.7 MB) boyut ve
   SHA-256 ile listelendi; onay yükleyicisi göreli düzeni ve hash'leri
   doğruladığı için yerleşim şartı açıkça yazıldı. Yol B: dataset ->
   YOLO export -> ön-eğitimli ağırlık indirme -> kalite -> audit/karar
   onayı -> smoke -> PPO -> held-out evaluation, ölçülmüş sürelerle.
3. **Demo videosu depoya girdi.** AGENTS.md §2 videoları Git dışında
   tutar; kullanıcı README'de örnek video istedi. Tek, küçük bir örnek
   (`development_ood` parlak alan, episode seed `310004`, `hover_success`)
   `docs/media/` altına alındı; GitHub README mp4 gömmediği için GIF
   üretildi, mp4 bağlantı olarak verildi. `runs/demo_20260930/` ve diğer
   videolar Git dışında kalır.
4. **Cihaz varsayımı belgelendi.** Dedektör cihazı iki config'te `mps`
   sabit; Linux/CUDA için değiştirilecek alanlar ve `MUJOCO_GL=egl`
   gereksinimi "denenmedi" notuyla yazıldı.
5. **YOLO11n ön-eğitimli ağırlık** depoda değildir (`*.pt` ignore); indirme
   adresi ve `detector.toml` ile eşleşmesi gereken SHA-256 belgeye yazıldı.
6. **Eğitilmiş modeller depoya alındı (kullanıcı kararı, aynı gün:
   "Eğitilmiş modeli de pushlayalım arkadaşım kullanacak").** Onay
   yükleyicisi manifestten göreli yol ve SHA-256 ile çözdüğü için dosyalar
   `runs/` altındaki asıl yerleşimde bırakıldı ve `.gitignore`'da `runs/*`
   kuralına 11 dosyalık istisna açıldı: onay manifesti, audit raporu,
   quality raporu, `audit_consumed.json`, seed `20260911` `last.pt` +
   metrik/metadata JSON'ları, seed `20260912` `final_model.zip`,
   `vecnormalize.pkl`, `metadata.json` ve nihai protokol evaluation JSON'u
   (toplam ~7.8 MB). Dataset'ler, checkpoint'ler, diğer seed'ler ve
   `runs/weights/yolo11n.pt` (üçüncü taraf, AGPL) Git dışında kaldı.
   `.gitattributes` ile `runs/**` satır sonu dönüşümünden muaf tutuldu;
   aksi hâlde Windows `autocrlf` JSON hash'lerini bozup yükleyiciyi
   kapatırdı. Kurallar, proje ağacının boş dosyalı kopyasında `git init` +
   `git add -A -n` ile doğrulandı: 227 dosya, `runs/` altından tam 11,
   `.venv`/`build`/cache/`__pycache__` sızıntısı yok. Projenin kendisinde
   git komutu çalıştırılmadı.

### Çalıştırılan komutlar/testler ve sonuçları

```text
python3.13 -m venv <scratchpad>/fresh-venv; pip install -r requirements.lock; pip install --no-build-isolation -e .   ok
<fresh-venv> python -m pytest -q      382 passed, 2 warnings, 117 subtests passed in 51.89s
<fresh-venv> ruff check / format --check   All checks passed! / 125 files already formatted
<fresh-venv> mypy                     Success: no issues found in 61 source files
scripts/inspect_model.py (0.1 s), run_hover_demo.py (0.4 s), smoke_navigation_env.py --mode clear (1.5 s, termination_reason=goal),
train_ppo.py --smoke --device cpu (2.5 s, run dizini tam)   hepsi ok
ffmpeg GIF: 960 px/10 fps 10.5 MB (reddedildi), 800 px/8 fps 6.2 MB, 640 px/8 fps 3.7 MB (seçildi)
.gitignore doğrulaması (scratchpad'de boş dosyalı kopya, 73189 dosya): git add -A -n -> 227 dosya; runs/ 11, docs 46, src 99, scripts 38, tests 27
git check-attr: runs/**/approval_manifest.json text=unset; last.pt, .gif binary
```

### Bilinen sınırlılıklar ve riskler

- Depoya alınan `metadata.json` yazarın yerel mutlak yollarını içerir
  (bilgi amaçlı alan, kod okumaz).
- `runs/weights/yolo11n.pt` depoda değildir; dedektörü yeniden eğitmek
  isteyen belgeye göre indirir.
- Linux/CUDA adımları denenmedi; belge bunu açıkça söyler.
- `runs/*/metadata.json` dosyaları mutlak yerel yollar içerir; depoya
  girmedikleri için sorun değildir.

### Sonraki önerilen adım

REV-051 çarpışma kapısı kararı (dönüş-hız bağlamasıyla yeni seed veya engel
belleği); ardından yoğunluk merdiveni.

### Ek (aynı gün): GitHub'a gönderim

Kullanıcı "Repoya sen pushlayabilirsin izin veriyorum" dedi; 11 Eylül
"Git'i agent bağlamaz" kuralı bu kararla kalktı. Gönderim öncesi depoya
girecek dosyalarla doğrulama: `evaluate_vision_policy.py` (held-out
`validation`, deterministik, yardımcı açık, 4 episode) `4/4 hover_success`,
çarpışma 0, ortalama getiri `+111.5`; onay yükleyicisi hash zincirini aynı
dosyalardan doğruladı. Sonra `git init -b main`, `git add .` (227 dosya,
`runs/` 11, sızıntı 0, 15.2 MB), commit `35d9495` ve
`git push -u origin main`; uzak `refs/heads/main` aynı hash'te.
Kimlik global git config'ten, kimlik bilgisi macOS keychain'den geldi.

## 1 Ekim 2026 — Sistem mimarisi diyagramı (PNG)

### Amaç ve kapsam

Kullanıcı isteğiyle güncel sistemin tek sayfalık mimari görseli üretildi:
kamera -> YOLO11n -> kutu yuvası kodlayıcı -> PPO politikası -> hareket
kısıtları (ADR-024) -> klasik kademeli kontrolcü (ADR-011) ana hattı; hover
durum makinesi (ADR-017) ve terminal hover yardımcısı (ADR-025) yan hattı;
yalnız eğitimde var olan ödül/terminasyon yolu kesikli kırmızı; fizik kapalı
döngüsü yeşil. Üç seed sonucu ve "simülasyon sonucudur" notu görselin
içindedir.

### Oluşturulan/değiştirilen dosyalar

```text
docs/reports/2026-10-01-sistem-mimarisi.png   yeni (matplotlib, 1727x1021 px)
README.md                                     rapor listesine bağlantı
```

### Alınan teknik kararlar ve gerekçeleri

- Diyagram ADR-023 §3-4, ADR-024, ADR-025 ve `vision_navigation_v4.toml`
  değerlerinden çizildi; kod değişmedi. Üretim betiği tek seferliktir ve
  depoya alınmadı (görsel yeniden gerekiyorsa `docs/reports/` içindeki PNG
  esas alınır).

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python <matplotlib betiği>   -> docs/reports/2026-10-01-sistem-mimarisi.png; görsel kontrol: ok çakışması yok
```

### Bilinen sınırlılıklar ve riskler

- Görsel yalnız veri akışını gösterir; ortam config'i değiştiğinde (ör.
  kısıt parametreleri) elle güncellenmesi gerekir.

### Sonraki önerilen adım

REV-051 çarpışma kapısı için kullanıcı kararı (dönüş-hız bağlamasıyla yeni
seed eğitimi veya engel belleği); ardından yoğunluk merdiveni.

## 30 Eylül 2026 — Rapor: RL sistemi (observation, action, ödül, YOLO çıktısının kullanımı), Word

### Amaç ve kapsam

Kullanıcı isteğiyle güncel sistemin (ADR-023/024/025, ADR-017 R1) Türkçe,
ayrıntılı ve anlaşılır bir raporu `.docx` formatında hazırlandı: observation
space (vision-v4, 88 sayı), action space (vision-action-v3 ve kısıt katmanları),
18 bileşenli ödül fonksiyonu (formül, ağırlık, aktif durumlar), YOLO çıktısının
dört kullanım yeri, durum makinesi, eğitim düzeni, üç seed sonuçları, sözlük.

### Oluşturulan/değiştirilen dosyalar

```text
docs/reports/2026-09-30-rl-sistemi-raporu.docx   yeni (docx-js ile üretildi; 10 bölüm, 16 tablo)
README.md                                         rapor bağlantısı
```

### Alınan teknik kararlar ve gerekçeleri

- Ödül tablosu doğrudan `rewards/terminal_hover.py` ve
  `vision_navigation_v4.toml` değerlerinden yazıldı; aktif durum kümeleri
  koddaki `ALIGNMENT/CRUISE/BEARING/APPROACH/STILL_STATES` ile aynı.
- Belge, gerçek uçuş güvenliği iddiası içermediğini kapakta belirtir.

### Çalıştırılan komutlar/testler ve sonuçları

```text
node build_report.js docs/reports/2026-09-30-rl-sistemi-raporu.docx  (docx 9.x, scratchpad'e yerel kurulum)
pandoc -t markdown ile geri okuma: 10 ana bölüm, 16 tablo, 71 madde; LibreOffice olmadığı için PDF önizleme yapılmadı
```

### Bilinen sınırlılıklar ve riskler

- İçindekiler alanı Word'de açılışta güncellenir (alan güncelleme isteği).

### Sonraki önerilen adım

REV-051 çarpışma kapısı için kullanıcı kararı (dönüş-hız bağlamasıyla eğitim
veya engel belleği).

## 30 Eylül 2026 — Dilim D kapanışı: üç seed protokolü (ADR-023 §10) ve demo

### Amaç ve kapsam

Nihai yapılandırmayla (ADR-023 dedektör + ADR-024 R1 koni kısıtı + ADR-024 R2
koridor hız sınırlayıcı + ADR-017 R1 erken kayıp kuralı + ADR-025 terminal
yardımcı) üç seed'in held-out 40 episode deterministik ölçümü; demo videosu
aracı (`scripts/render_demo_video.py`).

### Oluşturulan/değiştirilen dosyalar

```text
runs/detector_cone_d1a_seed20260914_20260930/   seed 3 eğitimi (yardımcı açık, 600k) ve evaluation dosyaları
runs/detector_cone_d1a_seed*/evaluation_heldout_final_det_assist_farloss_governor.json   nihai protokol ölçümleri
scripts/render_demo_video.py                     yeni: drone kamerası + YOLO kutuları | dış takip kamerası | üstten harita
runs/demo_20260930/seed12_ep_seed310015.mp4, seed12_ep_seed310020.mp4   demo videoları (~1 MB, H.264)
docs/decisions/ADR-023-...md, IMPLEMENTATION_PLAN.md §24, README.md   durum
```

### Alınan teknik kararlar ve gerekçeleri

1. **Tek rejimde ölçüm.** Üç seed aynı ortam config'iyle (koni R1, koridor
   sınırlayıcı, erken kayıp kuralı, yardımcı) ölçüldü. Eğitim rejimleri
   farklıdır: seed 1 yardımcısız (600k), seed 2-3 yardımcı açık (600k);
   bu fark raporlanır.
2. **Kapı yorumu:** ADR-023 §10 kapısı seed ortalaması üzerinden okunur ve
   seed başına ayrıca raporlanır; eşik değişmez.

### Çalıştırılan komutlar/testler ve sonuçları

```text
seed 20260914 eğitim içi evaluation (10 ep): 100k +21.3, 150k +77.3, 200k +90.6, 250k +101.6, 300k +104.2,
  550k +98.1, 600k +99.5; eğitim 3 s 8 dk
held-out 40 episode, deterministik, validation görsel alanı, nihai config:
  seed 20260912 : başarı %87.5  arrival 0.875  çarpışma 0.050  getiri +95.3  (2 çarpışma, 1 sınır dışı, 1 min irtifa, 1 hedef kaybı)
  seed 20260913 : başarı %77.5  arrival 0.775  çarpışma 0.225  getiri +81.4  (9 çarpışma)
  seed 20260914 : başarı %87.5  arrival 0.875  çarpışma 0.125  getiri +96.9  (5 çarpışma)
  ORTALAMA      : başarı %84.2  arrival 0.842  çarpışma 0.133
  kapı: arrival >= 0.80 GEÇTİ (ortalama; 2/3 seed), çarpışma <= 0.10 GEÇMEDİ (ortalama 0.133; 1/3 seed)
  seed 3 sınırlayıcısız: başarı %87.5, arrival 0.875, çarpışma 0.125 (sınırlayıcı seed 3'te etkisiz)
demo: scripts/render_demo_video.py (seed 1 politikası, episode seed 310015 ve 310020, ikisi de hover_success; 411 ve 438 adım)
```

Kıyas (aynı protokol, held-out 40 episode): VAE politikası ve holonomik
dedektör politikası `%0` hover başarısı; bu dilim sonunda üç seed ortalaması
`%84`.

### Bilinen sınırlılıklar ve riskler

- Çarpışma kapısı ortalamada geçilmedi; seed'ler arası yayılım büyük
  (`%5-%22.5`). Kalan çarpışmalar dönüş sırasında kadraj dışına kayan
  engellerden; yeniden eğitilmemiş politikaya eklenen yavaşlama kuralları
  bunu kötüleştirdi (kayıtlı). Çözüm adayı: dönüş-hız bağlaması veya engel
  belleği ile **eğitim** (ablation).
- Eğitim rejimi seed'ler arasında aynı değil (yardımcı açık/kapalı).
- Sonuçlar simülasyon; gerçek uçuş güvenliği iddiası yoktur.

### Sonraki önerilen adım

Çarpışma kapısı için: (1) dönüş-hız bağlaması açıkken yeni seed eğitimi,
(2) yoğunluk merdiveni (10 -> 20 -> 40 engel) öncesi 3 seed'in tamamında
`<= %10`. Kullanıcı kararı.

## 30 Eylül 2026 — Dilim D seed `20260913` (yardımcı açık eğitim) ve ADR-024 Revizyon 2 hız sınırlayıcı

### Amaç ve kapsam

Üç seed protokolünün ikinci seed'i: güncel v4 config (koni R1 + erken kayıp
kuralı + yardımcı açık) ile 600k adım; held-out ölçüm ve çarpışma tanısı.
Tanıya göre menzil tabanlı hız sınırlayıcı eklendi ve seed'ler yeniden
eğitilmeden ölçüme alındı.

### Oluşturulan/değiştirilen dosyalar

```text
runs/detector_cone_d1a_seed20260913_20260930/          eğitim (2 s 35 dk), evaluation_heldout_final_det_assist_farloss.json
runs/_stuck_detector_cone_d1a_seed20260913_20260930_attempt1/   ilk denemenin kilitlenen çıktısı (kanıt)
src/ebabil/envs/motion_constraint.py                   SpeedGovernorConfig, governor_speed_limit, apply_speed_limit
src/ebabil/envs/detection_vision.py                    kanca: koni izdüşümünden sonra hız sınırı; info speed_governor_limit_m_s
src/ebabil/envs/vision_config.py, vision_navigation_v4.toml   [speed_governor] (v4'te açık)
tests/test_detection_vision_env.py                     +2 test; koni testleri sınırlayıcıyı açıkça kapatır
docs/decisions/ADR-024-...md                           Revizyon 2
```

### Alınan teknik kararlar ve gerekçeleri

1. **Sınırlayıcı, yeniden eğitim değil.** Çarpışmalar görülen engele yüksek
   hızla gitmekten; hız sınırı deploy edilebilir bir güvenlik katmanıdır ve
   üç seed'e de eğitim olmadan uygulanabilir (karşılaştırma aynı politikalarla).
2. **Koridor ve menzil parametreleri** (`0.4`, `0.6 m`, `1.0/s`, `0.15 m/s`)
   kamera geometrisinden: orta koridor hareket konisini (`±30` derece) kapsar;
   `0.6 m` araç payı (`0.25`) + fren payı.
3. **Seed 3 eğitimi sınırlayıcısız sürüyor** (config eğitim başında okundu);
   ölçüm sınırlayıcıyla yapılır. Üç seed aynı rejimde ölçülür.

### Çalıştırılan komutlar/testler ve sonuçları

```text
seed 20260913 eğitim içi evaluation (10 ep, yardımcı açık): 50k +16.5, 100k +71.7, 150k +86.3, 200k +29.0,
  250k +83.6, 300k +60.0, 350k +61.7, 400k +118.1 (sapma 8.0), 450k +85.4, 500k +105.5, 550k +105.2
held-out 40 episode, deterministik, yardımcı + erken kayıp kuralı (sınırlayıcısız):
  hover başarısı %75 (30/40), arrival 0.75, çarpışma 0.25 (10), ortalama getiri +78.0
çarpışma tanısı (yeniden oynatma): 9/10 seyir veya acquired durumunda politika kontrolünde, engel kutusu
  görünürken, 1.3-1.9 m/s; yalnız 1/10 yardımcı aktifken (approach, 1.7 m). Yön: 6/10 görüş açısı içinde,
  kalanlar koni kenarında (44-68 derece; dönüş sırasında gerçek hız komuttan geride kalıyor)
tests/test_detection_vision_env.py + test_motion_constraint.py : 32 passed; ruff/mypy temiz
hız sınırlayıcı (koridor kuralı) ile aynı politikalar, held-out 40 episode, yardımcı + erken kayıp kuralı:
  seed 20260912 (600k): başarı %87.5, arrival 0.875, çarpışma 0.05 (2/40; sınırlayıcısız 0.10), getiri +95.3
  seed 20260913:        başarı %77.5, arrival 0.775, çarpışma 0.225 (9/40; sınırlayıcısız 0.25), getiri +81.4
  seed 2 tanısı (sınırlayıcıyla): çarpışma anında kutu yok ya da kadrajın en kenarında (cx_norm -0.9, menzil geçersiz);
    araç 1.3-1.6 m/s ile ilerlerken sürekli yaw döndürüyor (a_yaw 0.4-1.0) -> dönüş sırasında kör alana sürüklenme
DENENDİ ve GERİ ALINDI (yeniden eğitim olmadan, aynı politikalar):
  (a) dönüş-hız bağlaması (tam yaw'da 0.3 m/s): seed 1 çarpışma %5 -> %12.5 (başarı %77.5, süre dolması 3),
      seed 2 %22.5 -> %45. Sürekli yaw kullanan politikanın beklediği ilerleme kırılıyor.
  (b) kenardaki yakın (menzili geçersiz) engeli de sayma: seed 1 %5 -> %12.5, seed 2 %22.5 -> %30.
  Sonuç: yeniden eğitilmemiş politikaya eklenen her ek yavaşlama davranışı bozuyor; yalnız koridor kuralı kalır
  (turn_penalty varsayılan 0, kod korunur; sonraki eğitimlerde açık ablation adayı).
```

### Bilinen sınırlılıklar ve riskler

- Seed'ler arası çarpışma farkı büyük (`%10` vs `%25`); 40 episode'luk
  belirsizlik `~±5` puan.
- Sınırlayıcı seed 1'de yardımcı oldu (`%10 -> %5`), seed 2'de az (`%25 -> %22.5`);
  seed 2'nin çarpışmaları dönüş sırasında kör alana sürüklenmeden ve
  yalnız bir yavaşlama kuralıyla çözülmüyor. Eğitim sırasında açık olan bir
  kısıt (dönüş-hız bağlaması) politikanın ona uyum sağlamasını gerektirir;
  yeniden eğitim ablation'ı olarak açık.

### Sonraki önerilen adım

Sınırlayıcıyla seed 1 ve 2 ölçümü; seed 3 bitince üç seed raporu.

## 30 Eylül 2026 — ADR-017 Revizyon 1 (erken tespit kaybı) ve D1a kapısının tek seed'de geçilmesi

### Amaç ve kapsam

ADR-025 ölçümünde kalan 9 başarısızlığın 4'ü, hedefi `5-7 m`'den görüp
kadrajdan çıkarınca `2.5 s`'de bırakılan episode'lardı. Arama yarıçapının
dışındaki erken tespit kaybının abort üretmemesi için durum makinesine
`far_loss_returns_to_transit` bayrağı eklendi (yalnız v4 config'inde açık).
Aynı politika ve seed'lerle ölçüm yinelendi.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/missions/hover_state.py            TerminalHoverConfig.far_loss_returns_to_transit; acquired +
                                              arama yarıçapı dışında kayıp -> transit (sayaçlar sıfır)
src/ebabil/envs/vision_config.py              yükleyici (varsayılan false)
src/ebabil/configs/environment/vision_navigation_v4.toml   far_loss_returns_to_transit = true
tests/test_terminal_hover.py                  +2 test (varsayılan abort; uzak kayıp transit, yakın kayıp abort)
tests/test_detection_vision_env.py            v4 hover config karşılaştırması bayrak dışında
docs/decisions/ADR-017-...md                  Revizyon 1; ADR-025 ölçülen sonuçlar; ADR-023 ablation satırları H, C
docs/IMPLEMENTATION_PLAN.md                   §24 Dilim D durumu
```

### Alınan teknik kararlar ve gerekçeleri

1. **Abort kuralı menzille sınırlandı, kaldırılmadı.** Arama yarıçapı (`4 m`)
   içindeki kayıp ve `approach`/`settle`/`hold` kayıpları aynen abort;
   yalnız `acquired` + uzak kayıp zincire başa döner. Hover ölçütü ve `SUCCESS`
   değişmez; `v0`/`v1` davranışı korunur (varsayılan `false`, test).
2. **Ölçüm aynı politikayla** (`runs/detector_cone_d1a_seed20260912_20260929`,
   600k): değişiklik yalnız termination'ı etkilediği için yeniden eğitim
   gerekmedi; fark kurala atfedilebilir.

### Çalıştırılan komutlar/testler ve sonuçları

```text
tests/test_terminal_hover.py + test_detection_vision_env.py + test_vision_navigation_env.py : 110 passed
ruff / mypy temiz
.venv/bin/python scripts/evaluate_vision_policy.py runs/detector_cone_d1a_seed20260912_20260929/final_model.zip runs/detector_audit_v1_20260929/approval_manifest.json runs/detector_cone_d1a_seed20260912_20260929/evaluation_heldout_final_det_assist_farloss.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --scene-mode random_obstacles --endpoint-mode random_both --episodes 40 --max-steps 1200 --env-version v2 --visual-domain-split validation --deterministic --terminal-assist on
hover başarısı 35/40 (%87.5), arrival 0.875 (kapı 0.80 GEÇTİ), çarpışma 0.10 (kapı 0.10 GEÇTİ, sınırda), abort 0,
  ortalama getiri +93.1; başarısızlar: 4 çarpışma (24-151. adım, hedeften 9-16 m), 1 sınır dışı
```

Aynı seed'in ölçüm zinciri (held-out, 40 episode, deterministik):

| Aşama | arrival | çarpışma | hover başarısı |
|---|---:|---:|---:|
| holonomik (ADR-020) | %55 | %35 | %0 |
| koni kısıtı (ADR-024) | %70 | %17.5 | %0 |
| + terminal yardımcı (ADR-025) | %77.5 | %10 | %77.5 |
| + erken kayıp kuralı (ADR-017 R1) | **%87.5** | **%10** | **%87.5** |

### Bilinen sınırlılıklar ve riskler

- Tek seed; çarpışma tam sınırda (4/40). Üç seed protokolü ve `1.2M` devam
  eğitimi bekleniyor.
- Kalan çarpışmalar seyrin başında, görülen engelden kaçınamama.

### Sonraki önerilen adım

`1.2M` devam eğitimini aynı protokolle ölçmek; seed `20260913` ve `20260914`
eğitimleri (600k, güncel v4 config: koni R1 + erken kayıp kuralı + yardımcı
açık) ve üç seed raporu.

## 30 Eylül 2026 — ADR-025 terminal hover yardımcısı: projede ilk başarılı hover'lar

### Amaç ve kapsam

Kullanıcının verdiği yetkiyle (afk, "bizi sonuca ulaştır"), öncelik sırasının
ikinci maddesi olan hover sorununa hazırlık: RL politikası hedefe ulaşıyor ama
yerleşemiyordu (eğitim episode'larının `%68`'i hedef çevresinde dolanıp süre
dolarak bitiyordu). Terminal durumlarda politikanın action'ını, dedektör
hedef kutusu ve uçuş kontrolcüsü attitude/irtifa tahminiyle hesaplanan
klasik bir yerleşme komutuyla değiştiren yardımcı yazıldı ve mevcut koni
kısıtlı politikayla ölçüldü.

### Oluşturulan/değiştirilen dosyalar

```text
docs/decisions/ADR-025-terminal-hover-assist.md   yeni
src/ebabil/missions/terminal_assist.py            yeni: TerminalAssistConfig, level_frame, ground_point_level,
                                                  estimate_target_offset (uzak kenar + kesik kutu için sütun-menzil
                                                  ikiye bölmesi), assist_action
src/ebabil/perception/detection_slots.py          SlotEncoding.target_box
src/ebabil/envs/detection_vision.py               step() override: terminal durumlarda yardımcı komutu;
                                                  info terminal_assist_active / terminal_assist_fraction
src/ebabil/envs/vision_config.py                  [terminal_assist] yükleyici
src/ebabil/configs/environment/vision_navigation_v4.toml   [terminal_assist] enabled = true
scripts/evaluate_vision_policy.py, record_vision_replay.py   --terminal-assist on|off
tests/test_terminal_assist.py                     yeni, 8 test (düzey çerçeve, MuJoCo projeksiyonlu sentetik
                                                  kutudan tahmin, komut yönleri)
tests/test_detection_vision_env.py                +2 test; yardımcı test yardımcılarında açıkça kapalı
```

### Alınan teknik kararlar ve gerekçeleri

1. **Klasik yerleşme, RL değil.** Beş koşullu hover kesişimi RL için zor,
   klasik regülasyon için kolaydır; mimari zaten kademelidir (ADR-011).
   Yalnız deploy edilebilir girdiler kullanılır.
2. **Kesik kutu problemi.** Hedef üstündeyken kamera `0.40 m`'den yakını
   görmez; kutunun alt kenarı kadraj dışı kalır. Merkez, uzak kenar
   menzilinden `1.2 m` geri alınır (kare `2 x 2 m`: hizalı 1.0, köşegen
   1.41). Yanal konum aynı sütunda merkez menziline karşılık gelen satırdan
   okunur. Ölçülen hata (397 rastgele poz, güvenli bölge içi): medyan 0.24,
   p90 0.50, max 0.77 m; hedef önde iken p90 0.32 m.
3. **Alt kenar payı 4 px:** dedektör kutusu kenardan birkaç piksel kısa
   kalabilir; 1 px payla yakın kenar yanlışlıkla "görünür" sayılıp hata
   1.1 m'ye çıkıyordu (testte yakalandı).
4. **Yardımcı v4 config'inde açık;** ablation için script'lerde
   kapatılabilir. Süren devam eğitimi yardımcısız başladığı için yardımcısız
   sürer; sonraki seed'ler yardımcı açıkken eğitilir.

### Çalıştırılan komutlar/testler ve sonuçları

```text
tests/test_terminal_assist.py + tests/test_detection_vision_env.py : 31 passed; ruff/mypy temiz
ön ölçüm (12 episode, held-out, deterministik, koni 600k politikası, yardımcı açık): hover başarısı 10/12,
  arrival %83, çarpışma %0, ortalama getiri +92.3 (yardımcısız +16.5)
.venv/bin/python scripts/evaluate_vision_policy.py runs/detector_cone_d1a_seed20260912_20260929/final_model.zip runs/detector_audit_v1_20260929/approval_manifest.json runs/detector_cone_d1a_seed20260912_20260929/evaluation_heldout_final_det_assist.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --scene-mode random_obstacles --endpoint-mode random_both --episodes 40 --max-steps 1200 --env-version v2 --visual-domain-split validation --deterministic --terminal-assist on
40 episode: hover başarısı 31 (%77.5), arrival 0.775 (kapı 0.80), çarpışma 0.10 (kapı 0.10), süre dolması 0,
  ortalama getiri +85.7; başarısızlar: 4 çarpışma, 4 erken hedef kaybı (5-7 m'den), 1 sınır dışı
  -> hedefe varan 31 episode'un hepsi başarılı
```

### Bilinen sınırlılıklar ve riskler

- Kapılar henüz geçilmedi (arrival `%77.5`, çarpışma tam sınırda);
  başarısızlıklar seyir fazında.
- Erken hedef kaybı abort'u: `acquired` durumunda hedef `5-7 m`'den
  görülüp kadrajdan çıkınca `2.5 s`'de bırakılıyor (ADR-017 kuralı). Politika
  yardımcı açıkken eğitilmediği için hedefi görünür tutmayı öğrenmedi.
- Tahmin, hedefin `2 x 2 m` kare ve zeminde olduğu görev bilgisine dayanır.

### Sonraki önerilen adım

Devam eğitimi (`1.2M`) bitince aynı ölçüm yardımcı açık/kapalı; ardından
yardımcı açıkken yeni seed eğitimi (politika teslimi öğrensin) ve erken
hedef kaybı için ADR-017 abort kuralının menzille sınırlanması değerlendirmesi.

## 30 Eylül 2026 — ADR-024 Revizyon 1 (serbestlik yalnız settle/hold) ve devam eğitimi (sürüyor)

### Amaç ve kapsam

Kullanıcı onayıyla iki adım: (1) terminal kör hareket serbestliği
`approach`'tan çıkarıldı (held-out'ta 7 çarpışmanın 3'ü oradan geliyordu);
(2) görülen engelden kaçınamama için ödül değiştirilmeden eğitim süresi
sınanıyor: koni kısıtlı seed `20260912` politikası `final_model.zip` +
`vecnormalize.pkl`'den devam ettirilerek `1.2M` toplam adıma eğitiliyor.
Sonuçlar bu kaydın altına eklenecek.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/configs/environment/vision_navigation_v4.toml   relaxed_states = ["settle", "hold"]
src/ebabil/envs/motion_constraint.py, vision_config.py     varsayılan relaxed_states settle/hold
tests/test_detection_vision_env.py                         approach artık kısıtlı (test), v4 config testi
docs/decisions/ADR-024-...md                               Revizyon 1
runs/detector_cone2_d1a_seed20260912_20260930/             devam eğitimi (sürüyor)
```

### Alınan teknik kararlar ve gerekçeleri

1. **Tek değişken:** ödül ve PPO ayarları aynı; yalnız serbest durum kümesi
   daraltıldı ve eğitim uzatıldı. Böylece fark eğitim süresine ve approach
   kısıtına atfedilebilir.
2. **Sıcak başlangıç:** ortam dinamiği `approach`'ta değiştiği için politika
   sıfırdan değil mevcut ağırlıklardan uyarlanır; SB3 `reset_num_timesteps=False`
   ile `total_timesteps` kümülatif hedeftir (`1.2M`).

### Çalıştırılan komutlar/testler ve sonuçları

```text
tests/test_detection_vision_env.py + tests/test_motion_constraint.py : 28 passed; mypy temiz
.venv/bin/python scripts/train_ppo.py --config src/ebabil/configs/training/ppo_detector_d1a.toml --seed 20260912 --total-timesteps 1200000 --resume runs/detector_cone_d1a_seed20260912_20260929/final_model.zip --resume-normalization runs/detector_cone_d1a_seed20260912_20260929/vecnormalize.pkl --run-name detector_cone2_d1a_seed20260912_20260930
başladı: 600k'dan devam, ilk rollout 97 adım/s
DÜZELTME (30 Eylül, 13:50): SB3 `learn(total_timesteps, reset_num_timesteps=False)` hedefi mevcut sayaca EKLER
  (base_class.py: total_timesteps += num_timesteps); koşu 1.2M'de durmayıp 1.8M'e gidiyordu. 1.27M adımda
  durduruldu; planlanan 1.2M için checkpoints/ppo_1200064_steps.zip kullanılır. Kural: devam eğitiminde
  --total-timesteps EK adım sayısıdır. final_model.zip yok (koşu kesildi).
eğitim içi evaluation (10 episode): 650k +27.7, 700k +24.9, 750k +5.7, 800k +46.3, 850k +44.5, 900k +34.2,
  950k +45.2, 1.0M +45.3, 1.05M +41.4, 1.1M +41.4; eğitim episode'larında (yardımcısız) çarpışma ~%7, süre dolması ~%68
held-out 40 episode, deterministik, yardımcı açık + erken kayıp kuralı, checkpoint 1.2M:
  hover başarısı %85 (34/40), arrival 0.85 (kapı 0.80 geçti), çarpışma 0.15 (kapı 0.10 GEÇMEDİ), ortalama getiri +94.3
  -> aynı protokolde 600k modeli: %87.5 / 0.875 / 0.10. Uzatılmış eğitim held-out çarpışmayı düşürmedi (6 vs 4);
     600k modeli seed 20260912'nin aday politikası olarak kalır.
```

### Bilinen sınırlılıklar ve riskler

- Uzatılmış eğitim (1.2M) held-out çarpışmayı iyileştirmedi (`%15` vs 600k `%10`).
  Ablation: 600k politikası eğitim görsel alanında (deterministik, 40 episode,
  yardımcı + erken kayıp kuralı) başarı `%80`, arrival `%80`, çarpışma `%17.5`
  (`runs/detector_cone_d1a_seed20260912_20260929/evaluation_traindomain_final_det_assist_farloss.json`).
  Karanlık held-out alan çarpışmanın nedeni değil; kalan çarpışmalar politikanın
  kaçınma yetkinliğinden ve 40 episode'luk örnekleme belirsizliğinden (`~±5` puan).
- 1.2M devam eğitiminin `approach` kısıtı 600k'dan sonra değiştiği için iki
  model tam karşılaştırılabilir değildir.
- Altyapı: seed `20260913` eğitimi ilk denemede başlangıçta kilitlendi (55 dk,
  6 süreç `%0` CPU, hiç adım yok; dizin `runs/_stuck_detector_cone_d1a_seed20260913_20260930_attempt1`
  olarak saklandı). Neden bulunamadı (nadir başlangıç yarışı; daha önce üç
  koşu sorunsuz başlamıştı). Zincire 5 dakikalık başlangıç bekçisi eklendi
  (`dilimD_chain3.sh`); ikinci deneme 30 s içinde ilerledi.

### Sonraki önerilen adım

Bitince held-out 40 episode ölçümü; çarpışma yönleri ve süre dolması
episode'larının analizi; kapı kararı.

## 30 Eylül 2026 — Dilim D: koni kısıtlı (ADR-024) seed `20260912` eğitimi ve held-out evaluation

### Amaç ve kapsam

ADR-024 kısıtının davranışsal doğrulaması: aynı PPO tarifi, aynı seed, kısıtlı
ortam; held-out 40 episode deterministik ölçüm ve holonomik sonuçla
karşılaştırma. Kullanıcı önceliği: önce kaçınma + varış, sonra hover.

### Oluşturulan/değiştirilen dosyalar

```text
runs/detector_cone_d1a_seed20260912_20260929/   final_model.zip, best/, logs/, evaluation_heldout_final_det.json
docs/decisions/ADR-024-...md                    Ölçülen sonuçlar
docs/REVIEW_FINDINGS.md                         REV-051 ölçüm ve durum
```

### Alınan teknik kararlar ve gerekçeleri

1. Kapı ölçümü `final_model` ile (seçim yanlılığı yok); çarpışmalar aynı
   seed ve deterministik politikayla yeniden oynatılıp yönleri ölçüldü.
2. Aynı adım sayısında eğitim episode'ları karşılaştırıldı (~150k adım):
   çarpışma `%41 -> %13`, hedefin 2.5 m yakınında biten `%47 -> %71`.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/train_ppo.py --config src/ebabil/configs/training/ppo_detector_d1a.toml --seed 20260912 --run-name detector_cone_d1a_seed20260912_20260929
600k adım, ~2.9 saat; eğitim içi evaluation (10 episode): 50k +20.6, 100k -39.0, 150k +23.0, 200k +15.4,
  250k +25.0, 300k +12.8, 350k +10.1, 400k +28.5 (en iyi), 450k +19.7, 500k +0.1, 550k -71.5 (sapma 153.8), 600k +14.3
.venv/bin/python scripts/evaluate_vision_policy.py runs/detector_cone_d1a_seed20260912_20260929/final_model.zip runs/detector_audit_v1_20260929/approval_manifest.json runs/detector_cone_d1a_seed20260912_20260929/evaluation_heldout_final_det.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --scene-mode random_obstacles --endpoint-mode random_both --episodes 40 --max-steps 1200 --env-version v2 --visual-domain-split validation --deterministic
arrival 0.70 (kapı 0.80; holonomik 0.55), çarpışma 0.175 (kapı 0.10; holonomik 0.35), başarı 0.00,
  süre dolması 15, hedef kaybı 17, çarpışma 7, sınır dışı 1; ortalama getiri +16.5 (holonomik -25.2)
çarpışma yönleri: 4 seyirde görüş açısı içinde (engel kutusu görünürken), 3 approach'ta yana kör hareket
süre dolan 15 episode'un hepsi hedefe varmış (en yakın medyan 0.49 m), adımlarının %61'i approach'ta
```

### Bilinen sınırlılıklar ve riskler

- Tek seed; varyans ölçülmedi.
- Kalan çarpışmaların iki kaynağı var: görülen engelden kaçınamama ve
  `approach`'taki kör yanal serbestlik.
- Hover yerleşmesi (REV-043) hâlâ sıfır; süre dolması yeni başarısızlık biçimi.

### Sonraki önerilen adım

Kaçınma + varış önceliğiyle: (1) terminal serbestliği yalnız `settle`/`hold`
ile sınırlamak (approach'taki kör çarpışmalar), (2) görülen engelden
kaçınmayı güçlendirmek (daha uzun eğitim veya menzil tabanlı yakınlık
şekillendirmesi). Karar kullanıcıda.

## 29 Eylül 2026 — ADR-024: "baktığın yöne uç" kamera konisi hareket kısıtı (`vision-action-v3`)

### Amaç ve kapsam

REV-051'in çarpışma tanısına (13/14 çarpışma kör yöne hareketle) kullanıcının
seçtiği çözüm: seyir ve aramada araç yalnız kameranın gördüğü koni içinde
hareket eder; hedefin yanında yavaş kör düzeltmeye izin verilir. Kısıtlı
ortamla yeniden eğitim başlatıldı.

### Oluşturulan/değiştirilen dosyalar

```text
docs/decisions/ADR-024-camera-cone-motion-constraint.md   yeni
src/ebabil/envs/motion_constraint.py        yeni: MotionConstraintConfig, project_to_camera_cone
                                            (Öklid izdüşümü), constrain_horizontal_velocity
src/ebabil/envs/vision.py                   _constrain_horizontal_body kancası (tabanda kimlik)
src/ebabil/envs/detection_vision.py         kanca: durum bağımlı kısıt; action_version v3;
                                            info["motion_constrained_fraction"]
src/ebabil/envs/vision_config.py            [motion_constraint] yükleyici (varsayılan kapalı)
src/ebabil/configs/environment/vision_navigation_v4.toml  [motion_constraint] enabled, 30 derece, 0.5 m/s
tests/test_motion_constraint.py             yeni, 7 test
tests/test_detection_vision_env.py          +3 test (durum bağımlılığı, geri komutla uçuş, v0/v3 kapalı)
docs/REVIEW_FINDINGS.md                     REV-051 kararı
```

### Alınan teknik kararlar ve gerekçeleri

1. **İzdüşüm, kesme değil.** Koni dışı komut en yakın güvenli komuta
   (koni kenarına) taşınır; yana-hafif-geri komut ileri-çapraz yavaş harekete
   dönüşür, açısı `> 120` derece olan komut sıfırlanır. Bu, politikanın niyetini
   mümkün olduğunca koruyan sürekli bir eşlemedir.
2. **Yarım açı 30 derece** (FOV yarısı 37.5): gövde ve hız gecikmesi payı.
3. **Terminal serbestliği 0.5 m/s:** ADR-020'nin gerekçesi olan hedef üstünde
   ince düzeltme korunur (hold hız sınırı 0.25 m/s).
4. **Seed 2 durduruldu:** holonomik tarifin ikinci seed'i (30720 adım) kararı
   değiştirmeyeceği için makine yeni eğitime ayrıldı.
5. `v0`/`v1` ve ADR-020 değişmez; kısıt yalnız v4 config'inde açık.

### Çalıştırılan komutlar/testler ve sonuçları

```text
tests/test_motion_constraint.py + tests/test_detection_vision_env.py : 28 passed
  (ilk yazımda bir test beklentisi yanlıştı: yana-hafif-geri komutun sıfırlanacağı varsayılmıştı;
   doğru Öklid izdüşümü onu koni kenarına taşır. Test düzeltildi, kod değişmedi)
ruff check / format --check : temiz     mypy : Success, 60 source files
pytest : 367 passed, 2 warnings, 112 subtests in 47.8 s
.venv/bin/python scripts/train_ppo.py --config src/ebabil/configs/training/ppo_detector_d1a.toml --seed 20260912 --run-name detector_cone_d1a_seed20260912_20260929
başladı (ilk rollout 98 adım/s)
```

### Bilinen sınırlılıklar ve riskler

- Araç engelden kurtulmak için dönmek zorundadır; bekleme süresi artabilir.
- Hover tutunma sorunu (REV-043) bu kısıtla çözülmez.

### Sonraki önerilen adım

Kısıtlı seed `20260912` eğitimini (~2.7 saat) held-out 40 episode ile ölçmek,
holonomik sonuçla (`%55` varış, `%35` çarpışma) karşılaştırmak; çarpışmaların
yön dağılımını yeniden ölçmek.

## 29 Eylül 2026 — Dilim D (1/3): seed `20260912` eğitimi ve held-out evaluation (kapı geçilmedi, REV-051)

### Amaç ve kapsam

ADR-023 Dilim D'nin ilk seed'i: `ppo_detector_d1a.toml` ile 600k adım,
dedektör döngüde; ardından ADR-023 §10 protokolüyle held-out görsel alanda
40 episode deterministik evaluation ve başarısızlık tanısı.

### Oluşturulan/değiştirilen dosyalar

```text
runs/detector_d1a_seed20260912_20260929/           final_model.zip, best/, checkpoints/, logs/,
                                                   evaluation_heldout_final_det.json
docs/REVIEW_FINDINGS.md                            REV-051
```

### Alınan teknik kararlar ve gerekçeleri

1. **Evaluation final modelle.** `best_model` eğitim içi 10 episode'luk
   gürültülü evaluation'la seçildiği için kapı ölçümü `final_model` ile yapıldı.
2. **Tanı yeniden oynatmayla.** Başarısız episode'lar aynı seed ve
   deterministik politikayla yeniden oynatıldı; dedektör çağrıları ve gövde
   hızı kaydedildi. Evaluation sonuçları değiştirilmedi.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/train_ppo.py --config src/ebabil/configs/training/ppo_detector_d1a.toml --seed 20260912 --run-name detector_d1a_seed20260912_20260929
600k adım, ~2.7 saat (ortalama ~61 adım/s; kuru koşudaki 100 adım/s PPO güncellemeleri ve eval turları olmadan ölçülmüştü)
eğitim içi evaluation (10 episode, eğitim görsel alanı): 50k -30.6, 100k -21.5, 150k -19.2, 200k -39.6, 250k -15.4,
  300k -27.9, 350k -27.6, 400k +9.4 (en iyi), 450k -14.1, 500k -17.1, 550k -25.2, 600k -3.2; eğitimde başarı 0
.venv/bin/python scripts/evaluate_vision_policy.py runs/detector_d1a_seed20260912_20260929/final_model.zip runs/detector_audit_v1_20260929/approval_manifest.json runs/detector_d1a_seed20260912_20260929/evaluation_heldout_final_det.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --scene-mode random_obstacles --endpoint-mode random_both --episodes 40 --max-steps 1200 --env-version v2 --visual-domain-split validation --deterministic
arrival 0.55 (kapı 0.80), çarpışma 0.35 (kapı 0.10), başarı 0.00; hedef kaybı 22, çarpışma 14, sınır dışı 3, arama zaman aşımı 1 (5 dk)
tanı: 14 çarpışmanın 13'ü kameranın görüş açısı dışına hareketle (5 yana, 8 geriye çapraz/geri);
  hedef kaybı abort'larının 14/22'sinde araç hedef merkezine < 0.5 m yaklaşmıştı (tutunamama, REV-043)
```

### Bilinen sınırlılıklar ve riskler

- Tek seed; seed `20260913` aynı tarifle koşuyor (varyans).
- Held-out görsel alan (`heldout_dim_cool`) çok karanlık; yine de dedektör
  hedefi doğru anda görüyordu, başarısızlık kontrol tarafında.

### Sonraki önerilen adım

REV-051 kararı: önerilen "baktığın yöne uç" kısıtı (ADR-020 revizyonu,
yeniden eğitim). Karar gelene kadar seed `20260913` koşusu varyans için
sürer; kısıt seçilirse durdurulabilir.

## 29 Eylül 2026 — Dilim C: `EbabilVisionNavigation-v2` (`vision-v4`), kutu yuvaları ve zemin düzlemi menzili

### Amaç ve kapsam

ADR-023 §4-§5: dedektör kutularını politikanın tek görsel girdisi yapan
`vision-v4` sözleşmesi, hover kapısının dedektörden gelmesi, eğitim ve
evaluation zincirine bağlantı. Kod ve testler dedektör eğitimiyle paralel
yazıldı (sahte dedektörle); throughput ve gerçek renderer smoke'u onaylı
dedektörle ölçülür.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/detection_slots.py   yeni: DetectionSlotConfig, CameraGeometry (config'ten
                                           f, asal nokta, montaj pitch/konum), ground_range_m
                                           (kutu alt kenarı ışını ile z = 0 kesişimi),
                                           encode_detection_slots (9 x 8 yuva, sıralama, boş yuva)
src/ebabil/envs/detection_vision.py        yeni: EbabilDetectionVisionNavigationEnv (v0 algı
                                           kancaları), FrameDetector protokolü, detector_factory
src/ebabil/envs/latent_vision.py           build_randomized_scene (v1 _build_scene_model gövdesi
                                           davranış değişmeden ortak fonksiyona taşındı)
src/ebabil/envs/vision_config.py           DetectionObservationConfig ve [detections] yükleyici
src/ebabil/configs/environment/vision_navigation_v4.toml   yeni (action/hover/reward v0 ve v3 ile aynı)
src/ebabil/configs/training/ppo_detector_d1a.toml          yeni (PPO ve run ayarları d1a ile birebir)
src/ebabil/envs/__init__.py                EbabilVisionNavigation-v2 kaydı
src/ebabil/training/config.py, env_factory.py, runner.py  kind = "detection_vision"
scripts/smoke_detection_vision_env.py      yeni (check_env, rastgele rollout, PPO, evaluation,
                                           dedektör-gerçek hedef görünürlük uyumu)
scripts/evaluate_vision_policy.py, record_vision_replay.py  --env-version v2
tests/test_detection_vision_env.py         yeni, 20 test (şekil kısıtlı onayın ortamda zorlanması dahil)
```

### Alınan teknik kararlar ve gerekçeleri

1. **Menzil, gövde çerçevesinde ışın + yerçekimi yönü.** Işın montaj
   pitch'i ve iç parametrelerle kurulur, zemin kesişimi yalnız gövdede
   ifade edilen yerçekimi yönü ve irtifayla hesaplanır; yaw gerekmez. Bu
   büyüklükler `fc_state`'in kullandıklarıyla aynıdır (uçuş kontrolcüsü
   attitude + irtifa tahmini). Kutular yarı-açık kenar koordinatında,
   asal nokta piksel-merkez koordinatında; dönüşüm `- 0.5`.
2. **Yuvalar kare yakalandığı anda kodlanır.** Attitude ve irtifa karenin
   yakalandığı fizik durumundan okunur; policy adımında sample-and-hold
   değişmez.
3. **Hover kapısı dedektörün hedef yuvası** (`conf >= 0.50`, alan
   `>= 16 px^2`). Privileged gözlemci yalnız `info` tanısıdır ve config ile
   kapatılabilir (`privileged_diagnostics`); kapı ona bakmaz (test: gözlemci
   görünür derken dedektör görmüyorsa durum makinesi `transit/search`'te
   kalır).
4. **v1 sahne randomizasyonu ortak fonksiyona taşındı**; v1 testleri
   değişmeden geçti.
5. **Uyarılar:** Gymnasium, `v2` kaydedilince `v0/v1` için "sürüm eski"
   uyarısı verir; iki ortam bilinçli olarak baseline tutulduğu için beklenen
   durumdur.

### Çalıştırılan komutlar/testler ve sonuçları

```text
geometri (1.5 m, düz uçuş): merkez piksel 1.776 m (analitik 1.776), üst satır 8.38 m, alt satır 0.545 m
MuJoCo çapraz doğrulaması: roll 6, pitch 9 derece, 35 yer noktasından kadraja girenlerin hepsinde
  project_world_points pikselinden hesaplanan menzil gerçek yatay mesafeye 1e-6 göreli hata içinde
tests/test_detection_vision_env.py : 17 passed
ruff check / format --check : temiz     mypy : Success, 59 source files
pytest : 357 passed, 2 warnings (Gymnasium sürüm uyarısı), 112 subtests in 48.0 s (son koşu)
throughput (D1a, rastgele politika, 400 adım, onaylı dedektör, makine boşken):
  v0 vision-v2 88.0 adım/s | v2 tanı açık 43.0 (0.49 x) | tanı kapalı 47.7 (0.54 x) | DR da kapalı 54.3 (0.62 x)
  -> kapı >= 0.5 x tanı kapalıyken geçer; vision_navigation_v4.toml privileged_diagnostics = false
PPO kuru koşusu: train_ppo.py --config ppo_detector_d1a.toml --total-timesteps 4096 (4 subproc ortam, her birinde
  YOLO MPS'te): fps 95-100, hata yok -> 600k adım ~1.7-2 saat/seed
.venv/bin/python scripts/smoke_detection_vision_env.py runs/detector_audit_v1_20260929/approval_manifest.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --output-dir runs/detection_vision_smoke_20260929
check_env temiz; 200 adım rastgele rollout (43.1 adım/s, tanı açık), PPO smoke, 80 adım deterministik evaluation;
  dedektör-gerçek hedef görünürlük uyumu 1.0; seyrek sahnede kare başına 0.065 engel kutusu (engeller nadiren kadrajda)
ortam içi görsel kontrol (static_scene_d1, 40 engel, 300 adım ileri uçuş, gerçek dedektör): 451 kare, %99'unda
  >= 1 engel kutusu, kare başına ortalama 2.0; üç kare çizilip incelendi: kutular engelleri sıkıca sarıyor,
  kadraj kenarındaki yarım engelde zaman zaman üst üste 2-3 kutu
```

### Bilinen sınırlılıklar ve riskler

- Menzil, engelin zemine bastığı varsayımına dayanır; kutunun alt kenarı
  örtülmüşse (başka engel önünde) menzil olduğundan kısa/uzun çıkabilir.
- `subproc` vektör ortamında her süreç kendi dedektör kopyasını MPS'te
  çalıştırır; bellek ve GPU paylaşımı throughput ölçümünde görülecek.
- Kadraj kenarında yarım görünen engel için dedektör bazen üst üste 2-3
  kutu üretir (NMS IoU 0.7); aynı engel birden fazla yuva kaplayabilir.
- Throughput kapısı ancak tanı render'ı kapalıyken geçer; evaluation
  tanıyı açtığı için daha yavaştır.
- Dedektör şekil kısıtlı onaylıdır (REV-050); ortam bunu zorlar.

### Sonraki önerilen adım

Dilim D: `ppo_detector_d1a.toml` ile üç seed (`20260912-14`) D1a eğitimi
(seed başına ~2 saat), ardından `evaluate_vision_policy.py --env-version v2
--visual-domain-split validation` ile 40 episode deterministik evaluation ve
en az bir replay. Kapı: `arrival >= %80`, `çarpışma <= %10`.

## 29 Eylül 2026 — REV-050 kararı: dedektörün eğitim şekilleriyle sınırlı onayı

### Amaç ve kapsam

Audit'in elipsoit engellerinde engel kapıları geçmedi (REV-050). Kullanıcı
kararı: "Buna gerek yok. Eğitimde kullandığımız engelleri kullanalım
environmentte." Karar sessizce değil, onay zincirinde ve ortamda açıkça
uygulandı.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/detector_quality.py  onay temeli: audit_passed | user_decision; ApprovedDetector
                                           (basis, obstacle_geometries); approve_detector_by_decision
                                           (tüketilmiş audit kaydı, geçmeyen kapılar, şekil kümesi zorunlu)
src/ebabil/envs/detection_vision.py        onaylı şekil kümesi dışındaki sahne config'ini kurulumda reddeder
scripts/approve_detector_by_decision.py    yeni
tests/test_detector_quality.py             +2 test (kısıtlı onay gidiş-dönüş, bozulma/eksik alan reddi)
tests/test_detection_vision_env.py         +1 test (elipsoit isteyen sahne reddi), yükleme testi güncellendi
runs/detector_audit_v1_20260929/approval_manifest.json   approval_basis = user_decision
docs/REVIEW_FINDINGS.md, ADR-023 (Revizyon 1)             karar ve durum
```

### Alınan teknik kararlar ve gerekçeleri

1. **Başarısız audit zincirde kalır.** Manifest audit raporunu SHA-256 ile
   bağlar ve geçmeyen kapıları (`obstacle_precision`, `obstacle_recall`)
   audit raporuyla birebir eşleşecek şekilde listeler; yükleyici uyuşmazlıkta
   reddeder. Kalite kapıları (geliştirme bölümleri) yine zorunludur.
2. **Kısıt ortamda zorlanır.** Onaylı şekiller kutu, silindir, kapsül, küre
   (dataset'in eğitim şekilleri). `static_scene_d1a` kutu/silindir/kapsül
   kullandığı için ortam değişmedi; elipsoit isteyen config kurulumda hata verir.
3. **Kapsamı açık:** simülasyon görevi için onay; gerçek dünya şekil
   genellemesi ADR-023 §12'ye ertelendi.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/approve_detector_by_decision.py runs/detector_quality_v1_20260929/quality_report.json runs/detector_audit_v1_20260929/audit_report.json --finding REV-050 --geometries box cylinder capsule sphere --decision "..."
approval_manifest=runs/detector_audit_v1_20260929/approval_manifest.json, basis=user_decision,
approved_obstacle_geometries=[box, cylinder, capsule, sphere], approved_weights=seed_20260911/weights/last.pt
tests/test_detector_quality.py + test_detection_vision_env.py : 32 passed
```

### Bilinen sınırlılıklar ve riskler

- Dedektör görülmemiş şekillerde güvenilir değildir; sahne üreticisine yeni
  şekil eklenirse yeniden eğitim ve yeni audit gerekir.

### Sonraki önerilen adım

Dilim C'nin onaylı dedektörle throughput ve smoke ölçümü (aşağıdaki kayıt).

## 29 Eylül 2026 — Dilim B (2/2): YOLO11n dedektör zinciri, tek seed kalite (kullanıcı kararı) ve audit (geçmedi, REV-050)

### Amaç ve kapsam

ADR-023 §3: ultralytics bağımlılığını çevrimdışı ve proje-yerel kurmak,
YOLO11n'i `runs/yolo_dataset_d1nr_20260926` üzerinde üç seed eğitmek,
ADR-023 kapılarını iki geliştirme bölümünde ölçmek, adayı tek kullanımlık
audit bölümünde doğrulamak ve fail-closed onay manifestini üretmek.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/detector.py          yeni: DetectorConfig (+ TOML yükleyici), configure_ultralytics_environment
                                           (YOLO_CONFIG_DIR = runs/ultralytics_config, YOLO_OFFLINE=1, sync kapalı),
                                           yolo_dataset_provenance, train_detector (COCO ağırlığından ince ayar,
                                           aday last.pt), YoloDetector (RGB -> BGR dönüşümü, Detection çıktısı,
                                           sınıf adı denetimi), measure_latency
src/ebabil/perception/detector_quality.py  yeni: metrik tanımları (box_iou, greedy_match, compute_detection_metrics),
                                           gate_checks (NaN geçmez), evaluate_detector_split (+ grayscale bilgi),
                                           ultralytics mAP bilgi alanı, run_detector_quality (3 seed),
                                           run_detector_audit (audit_consumed.json tek kullanım; smoke raporu reddi),
                                           verify_detector_approval (ultralytics'siz hash zinciri), load_approved_detector
src/ebabil/configs/training/detector.toml  yeni: [detector] ve [quality] (ADR-023 §3 eşikleri)
scripts/train_detector.py, run_detector_quality.py, run_detector_audit.py   yeni
scripts/finalize_detector_quality.py       yeni: yarıda kesilen koşuyu tamamlanmış seed'lerle raporlama (karar metni zorunlu)
tests/test_detector_quality.py             yeni, 12 test (IoU, eşleşme, metrik tanımları, kapılar, config,
                                           onay zinciri: geçerli / SHA bozulması / yanlış tür / geçilmemiş veya
                                           eksik kapı; COCO ağırlığının sınıf denetimiyle reddi)
pyproject.toml / requirements.lock         detector grubu kuruldu; lock pip freeze ile yenilendi (66 paket)
runs/weights/yolo11n.pt                    COCO ön-eğitimli ağırlık (sha256 0ebbc80d...), bir kez indirildi
```

### Alınan teknik kararlar ve gerekçeleri

1. **Aday `last.pt`, `best.pt` değil.** `best.pt` validation bölümünde
   seçilir; kapıyı aynı bölümde ölçmek seçim sızıntısı olurdu.
2. **COCO ön-eğitimli başlangıç.** Tek seferlik indirme (5.4 MB,
   SHA-256 config'te sabit); sonraki bütün koşular `YOLO_OFFLINE=1` ile.
   Sınıf başlığı iki sınıfa yeniden kurulur; COCO ağırlığı doğrudan
   dedektör olarak yüklenmeye çalışılırsa sınıf denetimi reddeder (test).
3. **Renk sözleşmesi.** Ortam ve dataset RGB üretir; ultralytics NumPy
   girdisini BGR varsayar. Dönüşüm yalnız `YoloDetector.detect` içinde.
4. **Metrik tanımları kodda ve testte sabit** (modül docstring'i):
   eşleşme IoU `>= 0.5`, güvene göre açgözlü; engel precision'da eşleşmeyen
   küçük (`< 400 px^2`) tahminler paydadan çıkar; sahte hedef oranı yalnız
   privileged olarak da hedef görünmeyen karelerde.
5. **Veri yükleme paralel** (`workers 4`, `cache ram`): olabilirlik
   ölçümünde epoch süresi yarıya indi (512 görüntü: 15.7 s -> 7.2 s).
6. **Gecikme ölçümü** tek kare, MPS, ısınmadan sonra medyan (100 kare).

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/pip install -e ".[detector]"   : ultralytics 8.4.163, torchvision 0.29.0, opencv-python 5.0.0.93 (numpy 2.5.3, torch 2.14.0 değişmedi)
olabilirlik (512 train / 128 val görüntü, 1 epoch): workers 0 15.7 s; workers 4 7.2 s (ram) / 7.9 s (önbelleksiz)
tek kare gecikme (COCO ağırlığı, 320 px): MPS 4.2 ms, CPU 10.7 ms
ruff check / format --check : temiz     mypy : Success, 57 source files
pytest : 335 passed, 1 warning, 112 subtests passed in 46.5 s
.venv/bin/python scripts/run_detector_quality.py runs/yolo_dataset_d1nr_20260926 <scratchpad>/detector_quality_smoke --epochs-override 1
uçtan uca smoke (her seed 1 epoch; audit'e alınamaz): exit 0, rapor + kapılar + mAP + grayscale üretildi; seed başına 63 s
  1 epoch sonrası (validation / development_ood, 3 seed aralığı):
  engel recall 0.76-0.79 / 0.89-0.92   engel precision 0.71-0.89 / 0.61-0.88
  hedef recall (<= 5 m) 0.88-0.90 / 0.81-0.88   hedef precision 0.82-0.99 / 0.86-0.97
  sahte hedef 0.004-0.069 / 0.000-0.055   gecikme medyan 5.3-5.8 ms   quality_passed=False (beklenen)
.venv/bin/python scripts/run_detector_quality.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929
seed 20260911: 60 epoch, 4589 s (76 dk; epoch ~61 s yalnız, paralel testlerle ~77 s); bütün kapılar iki bölümde GEÇTİ
  validation / development_ood: engel recall 0.950 / 0.983, engel precision 0.939 / 0.956,
  hedef recall (<= 5 m) 0.973 / 0.983, hedef precision 0.924 / 0.982, sahte hedef 0.013 / 0.005,
  gecikme medyanı 5.8 / 5.2 ms; bilgi: mAP50 0.926 / 0.971, mAP50-95 0.815 / 0.890
  grayscale validation (bilgi): engel recall 0.792, hedef recall 0.169 -> hedef büyük ölçüde renkle tanınıyor
Kullanıcı kararı (29 Eylül): kalan iki seed beklenmeden tek seed ile devam. Koşu durduruldu
  (seed 20260912 4. epoch'ta, dizini kanıt olarak duruyor; 20260913 başlamadı).
.venv/bin/python scripts/finalize_detector_quality.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929 --seeds 20260911 --decision "..."
quality_passed=True; rapor kararı, planlanan seed'leri ve yarıda kalan dizini taşır
.venv/bin/python scripts/run_detector_audit.py runs/yolo_dataset_d1nr_20260926 runs/detector_quality_v1_20260929/quality_report.json runs/detector_audit_v1_20260929
audit_passed=False (768 kare, yalnız elipsoit engeller; 19 s): engel recall 0.456, engel precision 0.887 GEÇMEDİ;
  hedef recall 0.966, hedef precision 0.957, sahte hedef 0.003, gecikme 5.6 ms geçti; mAP50 engel 0.722 / hedef 0.971
tanı (tüketilmiş audit): büyük elipsoitlerin %45'i doğru, %26'sı yeri doğru ama güven < 0.25, %23'ü hiç tespit yok
  (validation'da aynı ölçüm %94 / %3 / %1) -> kök neden şekil genellemesi (REV-050)
onay manifesti üretilmedi; audit bölümü tüketildi (audit_consumed.json)
tests/test_detector_quality.py : 12 passed (tek seed raporlama testleri dahil)
```

### Bilinen sınırlılıklar ve riskler

- **Audit geçmedi (REV-050):** dedektör görülmemiş şekle genellemiyor.
  Onay manifesti yok; `v2` ortamı fail-closed yükleyici yüzünden bu
  dedektörle henüz açılamaz. Karar kullanıcıda.
- Tek seed: eğitim tarifinin tekrarlanabilirliği ölçülmedi (kullanıcı kararı).
- Hedef tespiti renge dayanıyor (gri tonlamada recall 0.17).
- Sonuçlar sentetik görüntüdedir; gerçek kamera performansı bilinmiyor.

### Sonraki önerilen adım

REV-050 kararı: (1) RL görevinin şekilleriyle (kutu/silindir/kapsül) yeni
salt'lı ayrı bir audit bölümü üretip mevcut adayı bir kez ölçmek, veya
(2) elipsoidi eğitime alıp yeni bir şekil ailesiyle audit kurarak yeniden
eğitmek. Onaydan sonra Dilim C'nin throughput ve gerçek renderer smoke'u.

## 26 Eylül 2026 — Dilim B (1b): zemin yansıması kapatıldı, dataset ve YOLO etiketleri yeniden üretildi

### Amaç ve kapsam

Etiket kontrolünde zemin materyalinin `reflectance = 0.05` değeri engellerin
yarı saydam yansımalarını çiziyordu (geometri değil, etiketsiz). Kullanıcı
kararıyla yansıma kapatıldı ve dedektör verisi bu ayarla yeniden üretildi.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/assets/mjcf/quadrotor.xml         ground_material reflectance 0.05 -> 0 (yorumla gerekçe)
docs/decisions/ADR-023-...md                 §6 yansıma kararı; §2 kaynak dataset; sınırlar
runs/vision_dataset_d1_p40nr_20260926/       yeni NPZ dataset (yansımasız), 6528 kare
runs/yolo_dataset_d1nr_20260926/             yeni YOLO dataset + review/
```

### Alınan teknik kararlar ve gerekçeleri

1. **Global ayar, config değil.** Yansıma MJCF materyalinde tanımlı; bütün
   ortamlar (legacy dahil) artık yansımasız render eder. Fizik etkilenmez;
   sahne parmak izi (`LEGACY_SCENE_FINGERPRINT`) geometriye dayandığı için
   değişmez. Etiketli `vision-encoder-v11` ve ADR-022 VAE zincirleri
   yansımalı görüntülerle eğitilmişti; baseline olarak koşulacaklarsa
   dataset ve onay zincirleri yeniden üretilmelidir (ADR-023 §6 notu).
2. **Aynı seed'ler, aynı sahneler.** Üretim deterministik: yeni dataset'in
   bölüm sayıları, kutu sayıları (`26554` engel, `3874` hedef), düşen küçük
   kutu sayısı (`1686`) ve seçilen kontrol kareleri öncekiyle birebir aynı;
   yalnız piksel içerikleri değişti. Bu, etiket türetmenin geometriye bağlı
   olduğunu ve render'dan bağımsız kaldığını da doğrular.
3. **Sahne config'i `static_scene_d1` (40 engel).** Orijinal dataset'in
   üretim komutu kayıtta açık değildi; kare başına ortalama `4.5` görünür
   engel örneği 40 engelli sahneyle uyumludur ve yeni dataset aynı
   istatistiği verdi.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/generate_vision_dataset.py runs/vision_dataset_d1_p40nr_20260926 --config src/ebabil/configs/training/vision_dataset_d1.toml --scene-config src/ebabil/configs/environment/static_scene_d1.toml
records=6528, splits train 4608 / validation 576 / development_ood 576 / audit 768, leakage_check=passed (~7 dk)
.venv/bin/python scripts/export_yolo_dataset.py runs/vision_dataset_d1_p40nr_20260926 runs/yolo_dataset_d1nr_20260926
kutular: engel 26554, hedef 3874; düşen küçük 1686 (öncekiyle birebir aynı)
.venv/bin/python scripts/review_yolo_labels.py runs/yolo_dataset_d1nr_20260926 --count 8 --seed 20260926 --output-dir runs/yolo_dataset_d1nr_20260926/review
aynı 8 kare; yansıma izleri yok, kutular aynı yerde; gölgeler (key_light) duruyor
pytest (etkilenen dosyalar): 63 passed; tam paket: 325 passed, 112 subtests passed
```

### Bilinen sınırlılıklar ve riskler

- Gerçek parlak zeminlerde yansıma olur; simülatör artık bunu üretmez.
  Gerçek görüntü ince ayarı (ADR-023 §12) bu farkı taşımalıdır.
- `runs/yolo_dataset_d1_20260926` (yansımalı, 920 MB) artık kullanılmaz;
  silinmesi kullanıcı kararıdır.

### Sonraki önerilen adım

Dilim B (2/2): ultralytics kurulumu ve YOLO11n üç seed eğitimi
`runs/yolo_dataset_d1nr_20260926` üzerinde; kalite kapıları ve audit.

## 26 Eylül 2026 — Dilim B (1/2): segmentation'dan kutu etiketleri, YOLO dataset dışa aktarımı ve görsel etiket kontrolü

### Amaç ve kapsam

ADR-023 §2: dedektör verisini elle etiketlemeden, mevcut NPZ dataset'inin
privileged `segmentation` / `obstacle_mask` / `target_mask` alanlarından
YOLO formatında kutu etiketleri üretmek; kullanıcının isteğiyle rastgele
seçilmiş karelerde etiketleri çizerek doğrulamak. Dedektör eğitimi ve onay
zinciri Dilim B'nin ikinci yarısıdır.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/detections.py   yeni: Detection, boxes_from_masks (geom başına görünür
                                      piksel kutusu, hedef tek kutu, küçük kutu filtresi),
                                      to_yolo_lines / parse_yolo_lines, draw_boxes (NumPy),
                                      label_record (hedef menzili, görünür örnek sayısı),
                                      export_yolo_dataset (images/, labels/, frames.jsonl,
                                      dataset.yaml, dataset_audit.yaml, export_summary.json)
scripts/export_yolo_dataset.py        yeni
scripts/review_yolo_labels.py         yeni: diskteki etiket dosyalarından rastgele kare çizimi
tests/test_detections.py              yeni, 7 test (örnek ayrımı, örtülme, küçük filtre,
                                      YOLO satırı gidiş-dönüş, çizim, sentetik dataset dışa aktarımı)
pyproject.toml                        [project.optional-dependencies] detector = pillow, ultralytics
src/ebabil/perception/__init__.py     dışa aktarımlar
runs/yolo_dataset_d1_20260926/        dışa aktarılan dataset (920 MB, PNG) + review/
```

### Alınan teknik kararlar ve gerekçeleri

1. **Kutu = geom'un görünür piksellerinin eksen-hizalı kutusu.** Örtülen
   kısım kutuya girmez (test); eğik/uzun bir cismin kutusu arka planı da
   kapsar, bu AABB etiketlerinin bilinen sınırıdır ve dedektör eğitiminde
   standarttır.
2. **Küçük kutu filtresi `min_side 3 px`, `min_area 16 px`.** Filtreye
   takılan kutular etiketten düşer ve `dropped_small` ile sayılır; kalite
   ölçümünde de yok sayılır (ADR-023 §2).
3. **Çeldiriciler ve zemin yansımaları etiketlenmez.** Çeldiriciler
   `obstacle_*` prefix'i taşımadığı için maskede yoktur (negatif örnek).
   Zemin materyalinin `reflectance = 0.05` değeri engellerin yarı saydam
   yansımalarını çizer; bunlar geometri değildir, etiketlenmez. Dedektör
   yansımayı engelden ayırmayı öğrenmek zorundadır; bu gerçek dünyadaki
   parlak zeminler için yararlı bir sağlamlıktır, fakat simülatöre özgü
   olduğu da not edilir.
4. **PNG (kayıpsız).** Dataset'in fotometrik gürültüsü JPEG ile
   bozulmasın diye; bedeli `920 MB`.
5. **Kare başına yan bilgi (`frames.jsonl`)**: hedef görünürlüğü, hedefin
   medyan derinliği (`target_range_m`, kalite kapısındaki "`<= 5 m`" ölçütü
   için), görünür engel örneği sayısı, düşen küçük kutu sayısı.
6. **Bölümler dataset'in bölümleridir.** `dataset.yaml` = train /
   validation / development_ood; `dataset_audit.yaml` yalnız audit runner'ı
   için (tek kullanımlık). Audit bölümü ellipsoid geometrisiyle üretilmişti
   (`vision_dataset_d1.toml`), yani geometri açısından da held-out'tur.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/export_yolo_dataset.py runs/vision_dataset_d1_p40_20260914 runs/yolo_dataset_d1_20260926
kareler        : train 4608, validation 576, development_ood 576, audit 768 (toplam 6528; 43 s)
kutular        : engel 26554 (train 18639 / val 2424 / dev 2298 / audit 3193), hedef 3874
düşen küçük    : 1686; hedefli kare 3874 (%59); engelsiz kare 680 (%10)
etiket sanity  : 30428 satır, geçersiz 0 (sınıf {0,1}, koordinatlar [0,1], boyutlar > 0)
engel kutuları : genişlik p10/p50/p90 = 10/27/77 px, yükseklik 10/43/139 px; alanı < 400 px^2 olan pay %29
hedef kutuları : genişlik 32/117/320 px, yükseklik 11/60/140 px; alanı < 400 px^2 olan pay %11
.venv/bin/python scripts/review_yolo_labels.py runs/yolo_dataset_d1_20260926 --count 8 --seed 20260926 --output-dir runs/yolo_dataset_d1_20260926/review
8 kare (bölüm başına 2), etiketler diskten okunup çizildi; review_sheet.png + tek tek PNG'ler
ruff check / format --check : temiz      mypy : Success, 55 source files
pytest : 325 passed, 1 warning, 112 subtests passed in 46.4 s
```

Görsel kontrol bulguları (8 kare, `runs/yolo_dataset_d1_20260926/review/`):

- Bütün engel kutuları görünür engel piksellerini sıkıca sarıyor; örtülen
  hedef kısmı kutuya girmiyor (train `s0020261342-p008`: capsule hedefin
  üstünde, hedef kutusu yalnız görünür parça).
- Uzak engeller kadrajın üst kenarında küçük kutular olarak doğru (validation
  `s0020261480-p010`: 8 engel, ikisi `3-5 px`); bunlar `min_side 3` filtresini
  geçer, YOLO için zor örneklerdir, kalite kapısı `>= 400 px^2` alanla ölçer.
- Zemin yansımaları ve çeldiriciler etiketsiz (doğru); heldout_dim_cool
  bölümü çok karanlık, etiketler yine doğru yerde.
- Audit ellipsoid'leri ve kenardan görünen `8 px`'lik dilimler doğru
  kutulanmış.
- Hata bulunmadı.

### Bilinen sınırlılıklar ve riskler

- Kontrol 8 kare üzerindedir; sistematik hata (maske-geom eşlemesi) olsaydı
  bütün karelerde görünürdü, tek tek nadir hatalar için istatistiksel
  garanti değildir.
- Diagonal uzun cisimlerde AABB arka planı kapsar; politika kutuyu
  "işgal edilmiş alan" olarak okuyacağı için muhafazakâr hata yönündedir.
- Küçük kutu payı yüksek (`%29` engel kutusu `< 400 px^2`); dedektörün
  bunları öğrenememesi beklenir ve kapı bunları saymaz.

### Sonraki önerilen adım

Dilim B (2/2): `.venv/bin/pip install -e ".[detector]"` (ultralytics
`8.4.163`, çevrimdışı ayarlar), `perception/detector.py` ve
`detector_quality.py`, `detector.toml`, YOLO11n `imgsz 320` üç seed eğitimi,
ADR-023 §3 kapıları, tek kullanımlık audit ve `approval_manifest.json`
(`kind = "yolo-detector"`).

## 26 Eylül 2026 — Yön değişikliği: iki modelli algı (YOLO dedektör + RL politikası), ADR-023 ve plan §24

### Amaç ve kapsam

Kullanıcı, Dilim 3 eğitimini (`runs/latent_d1a_seed20260912_20260923`,
`528k / 600k` adım) teknik bir değişiklik için durdurdu ve mimariyi belirledi:
drone hedefi ve engelleri kameradan gelen görüntüde bir YOLO modeliyle
tanıyacak, hedefin yalnız yaklaşık konumunu bilecek ve engellerin bounding
box'larından kaçınarak hedefe gidecek; iki ayrı model olacak (YOLO tespiti,
ardından çıktısı RL modeline). Bu kayıt plan §24 Dilim A'dır: karar ADR-023
ile belgelendi, plan, literatür, tasarım belgesi ve bulgu çerçevesi
güncellendi. Kod değişikliği yok.

### Oluşturulan/değiştirilen dosyalar

```text
docs/decisions/ADR-023-detector-policy-two-model-perception.md   yeni: iki model, vision-v4, kapılar,
                                                                  hover kapısı, throughput ve yedek yol
docs/decisions/ADR-022-label-free-visual-representation.md        durum: actor yolu için yerini bıraktı
docs/IMPLEMENTATION_PLAN.md            §23 notu; §24 (Dilim A-E, dosyalar, kapılar, maliyet)
docs/LITERATURE_REVIEW.md              §2.17 nesne tespiti ve sembolik algı arayüzü
docs/PROJECT_DESIGN.md                 görsel algı satırı; ADR-023 güncellemesi
docs/REVIEW_FINDINGS.md                REV-048 durumu yeniden çerçevelendi
README.md, docs/PROGRESS_LOG.md        durum
```

### Alınan teknik kararlar ve gerekçeleri

1. **Dedektör eğitim döngüsünün içinde** (kullanıcı kararı). Politika
   dedektörün gerçek hatalarını görerek öğrenir; bedeli seed başına `~3-4`
   saat eğitim. Throughput kapısı `>= 0.5 x v0`; geçilmezse ölçülmüş hata
   modeli yedek yol olarak Revizyonla açılır (ADR-023 §8).
2. **`vision-v4` yuva vektörü:** `8` engel + `1` hedef yuvası x `8` alan
   (`present, cx, cy, w, h, conf, range_norm, range_valid`), toplam `88`
   boyut. Menzil ipucu kutunun alt kenarından zemin düzlemi geometrisiyle
   (`f = 208.5 px`, pitch `40`, `fc_state` irtifa/attitude) türetilir; tek
   kamerada derinlik olmadığı için bu, politikanın tek mesafe bilgisidir ve
   gerçek araçta da hesaplanabilir. Yuvalar menzile göre sıralı, `k = 1`.
3. **Hover kapısı dedektörden** (`conf >= 0.50`, alan `>= 16 px^2`);
   privileged gözlemci yalnız `info` uyum metrikleri için kalır. Gerçek
   araçta bu kapının tek kaynağı dedektördür.
4. **Ultralytics YOLO11n, `imgsz 320`, 3 seed, AGPL-3.0** kullanıcı
   itirazına açık varsayım; isteğe bağlı bağımlılık grubu (`detector`) ki
   çekirdek ortam ve testler etkilenmesin.
5. **ADR-022 zinciri baseline** olarak kalır (ablation V); silinmez.
6. **Kapı eşikleri a priori** (engel recall/precision `0.90`, hedef recall
   `0.85` `<= 5 m`, hedef precision `0.90`, sahte hedef `<= 0.05`, gecikme
   `<= 15 ms`); ADR-022'de bu yaklaşım REV-049'a yol açmıştı, kural aynı:
   geçilmezse eşik düşmez, bulgu açılır ve karar kullanıcıya gider.

### Çalıştırılan komutlar/testler ve sonuçları

```text
Kod değişikliği yok. Ortam denetimi:
.venv/bin/pip download ultralytics --no-deps : 8.4.163, py3-none-any, Requires-Python >= 3.8, AGPL-3.0
bağımlılık uyumu : numpy 2.5.3 Darwin kısıtına uyar; torchvision 0.29.0 ve opencv-python 5.0.0.93 pip'te mevcut;
                   ek paketler matplotlib, polars, pyyaml, requests, psutil, nvidia-ml-py, ultralytics-thop, ultralytics-platform
kamera geometrisi : f = 160 / tan(37.5 deg) = 208.5 px, VFOV ~60 deg, pitch 40 deg -> bütün kadraj ufkun altında;
                    yer menzili h / tan(pitch + atan(v / f)); üst satırda ~5.6 x h (h = 1.5 m -> 8.4 m; hedef <= ~7.5 m görünür)
Dilim 3 kesilen eğitim (bilgi): rollout ödülü -70 -> -46, evaluation 50k..500k: -40, -24, -61, -50, -25, -25, -16, -19, -10.5, -22;
                    çarpışma payı %50 -> %43, hedef kaybı abort'u %19 -> %28, başarı 0; checkpoint'ler 100k-500k duruyor
```

### Bilinen sınırlılıklar ve riskler

- Dedektör kaçırması o karede engeli görünmez kılar; kapı bunu sınırlar,
  sıfırlamaz. Menzil tahmini engelin zemine bastığı varsayımına dayanır.
- Ultralytics bağımlılığı büyüktür ve ağ erişimi/telemetri ayarları
  kapatılmalıdır; kurulum Dilim B'de doğrulanır.
- Eğitim süresi `v1`'e göre `2-3x`; `subproc` ortamında süreç başına bir
  dedektör kopyası.
- Kapı eşikleri ölçülmeden konuldu; ilk kalite koşusu bunları sınar.

### Sonraki önerilen adım

Dilim B: `detections.py` (segmentation -> kutu, YOLO formatı) ve
`scripts/export_yolo_dataset.py` ile `runs/vision_dataset_d1_p40_20260914`'ten
YOLO dataset'i; `detector` bağımlılık grubu ve çevrimdışı kurulum;
`detector.py`, `detector_quality.py`, üç seed eğitim, kalite ve audit;
onay manifesti. Ardından Dilim C (`EbabilVisionNavigation-v2`, `vision-v4`).

## 23 Eylül 2026 — Dilim 2: `EbabilVisionNavigation-v1` (`vision-v3`), ortam içi görsel randomization ve privileged görünürlük kapısı

### Amaç ve kapsam

ADR-022 kademe 1'in ortam yarısı: etiketsiz VAE latent yığınıyla observation
sözleşmesi `vision-v3`, reset ve kare başına görsel domain randomization,
engel geometrisi çeşitliliği, hover durum makinesi için ayrıcalıklı
segmentation gözlemcisi, eğitim ve evaluation bağlantısı. `v0` (`vision-v2`)
davranışı korunur; legacy parmak izi değişmez.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/envs/vision.py                  v0'a davranışı değiştirmeyen algı kancaları:
                                           _load_perception, _allocate_perception_buffers,
                                           _perception_spaces, _create_actor_camera,
                                           _encode_actor_frame, _perception_observation,
                                           _target_visibility_score, _target_center_x;
                                           observation_keys sınıf niteliği;
                                           info["hover_horizontal_distance_m"] (arrival metriği)
src/ebabil/envs/latent_vision.py           yeni: EbabilLatentVisionNavigationEnv (v1),
                                           RandomizedActorCamera, latent yığını, DR, gözlemci
src/ebabil/envs/vision_config.py           latent_stack_size; [visual_randomization];
                                           [privileged_visibility] (varsayılanlar v0'ı korur)
src/ebabil/envs/__init__.py                EbabilVisionNavigation-v1 kaydı ve dışa aktarımlar
src/ebabil/perception/target_visibility.py yeni: PrivilegedTargetObserver, TargetVisibility
src/ebabil/perception/visual_randomization.py  apply_frame_visual_parameters_torch (LUT + önbellekli
                                           vinyet + gürültü bankası, 4.5 -> 1.2 ms/kare), make_noise_bank
src/ebabil/scenes/generator.py             SceneGeneratorConfig.obstacle_geometries;
                                           randomize_obstacle_geometries
src/ebabil/configs/environment/vision_navigation_v3.toml   yeni (reward/hover v0 ile birebir)
src/ebabil/configs/environment/static_scene_d1a.toml       obstacle_geometries = box/cylinder/capsule
src/ebabil/configs/training/ppo_latent_d1a.toml            yeni (Dilim 3 config'i, PPO ayarları d1a ile birebir)
src/ebabil/training/config.py, env_factory.py, runner.py   kind = "latent_vision", visual_domain_split
scripts/smoke_latent_vision_env.py         yeni (check_env, random rollout, PPO smoke, adım/s)
scripts/evaluate_vision_policy.py          --env-version, --visual-domain-split, arrival_rate
scripts/record_vision_replay.py            --env-version, --visual-domain-split
tests/test_latent_vision_env.py            yeni, 16 test (sözleşme, yığın, DR determinizmi,
                                           geometri, privileged kapı, torch/NumPy paritesi,
                                           gürültü bankası, eğitim config'i)
```

### Alınan teknik kararlar ve gerekçeleri

1. **v1, v0'ın alt sınıfıdır; v0 kancalara ayrıştırıldı.** Plan §23 "v0'dan
   türetilmez" demişti; bunun gerekçesi v0'ı bit-bit dondurmaktı. Kancalar
   v0'ın davranışını değiştirmeden (aynı hesap, aynı sıra) açıldı ve v0
   testleri (85 test) ile terminal hover testleri geçti. Zamanlama, horizon,
   hover durum makinesi, reward yolu ve holonomik action ortak kaldı; kod
   tekrarı yok.
2. **Latent yığını latent düzeyinde, en yeni önce.** Reset'te ilk latent ile
   kopyalanarak dolar; her reset/step'te tam bir kez kaydırılır. Kırpma
   oranı `info["latent_clip_fraction"]` ile raporlanır.
3. **Hover kapısı boolean privileged görünürlük.** `TargetVisibility.score`
   `{0, 1}`; ADR-017'nin `0.60/0.40` histerezisi bu skorda inert olduğu için
   durum makinesi değişmedi. Gözlemci policy adımı başına bir segmentation
   render'ı yapar (`2.6 ms`), reward yolu aynı okumayı kullanır.
   `target_alignment` merkezi maskenin ağırlık merkezinden gelir.
4. **Randomization kodlayıcı dataset'inin alanlarını okur.** Sahne parametreleri
   `(scene seed, 0xDA)`, kare parametreleri `(scene seed, kare indeksi, 0xF0)`
   seed'iyle üretilir; aynı seed aynı baytları verir (test). Fotometrik
   etkiler için NumPy sürümü (`4.5 ms`/kare, `240 x 320`) yerine aynı anlamlı
   torch sürümü yazıldı: kazanç + gamma 256 girişli arama tablosu (piksel
   değerinin fonksiyonu olduğu için birebir), vinyet haritası şekle göre
   önbellekte, gürültü reset başına seed'li 16 alanlık bankadan seed'li dilim
   + dairesel kaydırma (kare başına `randn` `2.2 ms` idi). Ölçülen maliyet
   `1.2 ms`/kare; gürültüsüz koşulda NumPy ile bir gri seviye içinde eşleşir
   (test). Gürültünün marjinal dağılımı aynıdır; kareler arası bağımsızlık
   augmentation için gerekmediğinden bankadan feragat edilmedi.
5. **Engel geometrisi kutu/silindir/capsule.** Küre ve ellipsoid bilinçli
   dışarıda: yarıçapı en fazla `0.45 m` olan küre havada asılı kalır ve
   üstünden/altından geçilebilir (REV-035 kaçışı geri gelirdi). Geometri
   yalnız `geometry` alanını değiştirir; AABB rota doğrulaması muhafazakâr.
6. **v3 config v0'ın kopyasıdır.** `[action]`, `[terminal_hover]`,
   `[terminal_hover_reward]` birebir aynı (testle sabit) ki ölçülen fark
   yalnız algı katmanına atfedilebilsin.
7. **`arrival_rate` evaluation script'inde.** Gerçek anchor'a yatay mesafe
   `<= approach_radius_m` ve fiziksel failure yok; `info` alanı v0'da da
   üretilir.

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff check / format --check : temiz
mypy                        : Success, 54 source files
pytest                      : 318 passed, 1 warning, 112 subtests passed in 60.3 s (tam paket)
tests/test_vision_navigation_env.py + test_terminal_hover.py (v0 koruması): 85 passed
tests/test_latent_vision_env.py : 16 passed
```

Gerçek renderer üzerinde hız ölçümü (`static_scene_d1a`, 5 engel, rastgele
politika, 400 adım, onaylı seed-`20260911` checkpoint'i; kalite koşusu
bittikten sonra, arka plan yükü olmadan):

```text
v0 vision-v2             : 72.9 adım/s
v1 vision-v3 (DR açık)   : 55.1 adım/s   (0.76 x v0)
v1 vision-v3 (DR kapalı) : 61.1 adım/s   (0.84 x v0)
latent_clip_fraction 0.0; visual_domain train_broad_randomized; geometriler box/cylinder
Optimizasyon öncesi (NumPy fotometrik, yük altında): 0.66 x v0 (DR açık), 0.85 (DR kapalı)
Fotometrik kare maliyeti (240 x 320): NumPy 4.46 ms -> torch LUT + gürültü bankası 1.03-1.17 ms
  (parametre örnekleme 0.016 ms; banka kurulumu reset başına 33 ms, 16 x 240x320x3 float32 ~15 MB)
```

Dilim 2 kapısı `>= 0.7 x v0`: `0.76` ile geçti; privileged render
optimizasyonu (yalnız terminal fazda segmentation) gerekmedi.

Gerçek renderer smoke'u, onaylı manifest ile:

```text
.venv/bin/python scripts/smoke_latent_vision_env.py runs/vae_audit_v1_20260923/approval_manifest.json --scene-config src/ebabil/configs/environment/static_scene_d1a.toml --output-dir runs/latent_vision_smoke_20260923
check_env temiz; 200 adım rastgele rollout (56.2 adım/s, 1 episode, hover_state transit,
latent_clip_fraction 0.0); 64 adım PPO (MultiInputPolicy, smoke_model.zip); 80 adım deterministik
evaluation (hover_horizontal_distance_m 17.2, terminal yok); süre 9.2 s; summary.json yazıldı
```

### Bilinen sınırlılıklar ve riskler

- Smoke yalnız çalışırlığı gösterir; 64 adım PPO öğrenme kanıtı değildir.
  Öğrenme Dilim 3'te ölçülür.
- Fotometrik torch sürümü NumPy sürümüyle bit-bit değil; gürültü farklı RNG'den
  ve reset başına 16 alanlık bankadan gelir. Dataset ve RL aynı dağılımı görür,
  aynı örnekleri değil.
- Segmentation gözlemcisi ikinci bir GL bağlamı açar; `subproc` vektör
  ortamında süreç başına iki renderer olur.
- Latent kırpma oranı ilk ölçümde raporlanacak; yüksekse `latent_scale`
  yeniden kalibre edilmelidir (sözleşme değişikliği, ADR notu ister).

### Sonraki önerilen adım

Dilim 3: `src/ebabil/configs/training/ppo_latent_d1a.toml` ile
`scripts/train_ppo.py` üzerinden üç seed (`20260912`, `20260913`, `20260914`)
D1a eğitimi (`600k` adım, 4 subproc env, seed başına `~1.5 saat`); ardından
`scripts/evaluate_vision_policy.py --env-version v1` ile held-out görsel
alanda 40 episode deterministik evaluation (`arrival_rate`, çarpışma,
`hover_success`, latent kırpma oranı) ve `record_vision_replay.py
--env-version v1` ile en az bir episode izleme. Kapı (ADR-022 §10):
`arrival >= %80`, `çarpışma <= %10`; `hover_success` raporlanır, kapı değildir.

## 23 Eylül 2026 — Revizyon 1b: sabit eşikli harita IoU kapısının kaldırılması, üçüncü kalite koşusu ve onaylı `visual-vae-v1`

### Amaç ve kapsam

Dilim 1 kapanışında kullanıcıya bırakılan ek karar, "ilerlemeye devam"
onayı ve önerim doğrultusunda uygulandı: ADR-021 mirası sabit `0.5`
eşiğindeki engel haritası IoU kapısı, hedef için kaldırılan sabit eşikli
kapıyla aynı kalibrasyon ölçüsü olduğu için kapı olmaktan çıkarıldı; değeri
raporlanmaya devam eder. Üç seed kalite koşusu ve tek kullanımlık held-out
audit yeniden koşuldu; onay manifesti üretildi. Hiçbir eşik gevşetilmedi.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/vae_quality.py          REQUIRED_GATES'ten obstacle_map_iou çıktı;
                                              obstacle_map_iou_value bilgi alanı olarak kalır
src/ebabil/configs/training/visual_vae.toml   minimum_obstacle_map_iou kaldırıldı
tests/test_visual_vae.py                      kapı listesi: obstacle_map_iou zorunlu değil
docs/decisions/ADR-022-...md                  Revizyon 1b bölümü; ölçülen sonuçlar
docs/REVIEW_FINDINGS.md                       REV-049 karar ve durum (kapatıldı)
runs/vae_quality_v3_20260923/                 üç seed kalite raporu ve checkpoint'ler
runs/vae_audit_v1_20260923/                   audit_report.json, approval_manifest.json
```

### Alınan teknik kararlar ve gerekçeleri

1. **Kapı kaldırıldı, ölçüm kaldırılmadı.** `obstacle_map_iou_value`
   (`0.5` eşiği) her raporda yazılmaya devam eder; etiketli zincirle
   karşılaştırma kaybolmaz.
2. **Deterministik tekrar doğrulandı.** v3 koşusu v2 ile aynı seed'lerde
   birebir aynı metrikleri verdi (ör. sabit eşikte validation harita IoU
   `0.2505 / 0.2754 / 0.3324`); kod değişikliği yalnız kapı listesini etkiledi.
3. **Aday seed `20260911`** (kalite runner'ının seçtiği ilk geçen seed);
   audit aynı checkpoint'i held-out `audit` bölümünde tek kez ölçtü.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/run_vae_quality.py runs/vision_dataset_d1_p40_20260914 runs/vae_quality_v3_20260923
quality_passed=True  (3 seed x 2 bölüm, 7 kapı; ~45 dk, Dilim 2 testleri ve ölçümleri paralelde koşarken)
  validation / development_ood (min-max):
  AUROC 0.805-0.809 / 0.817-0.822     AP 0.882-0.884 / 0.901-0.903
  en iyi eşikte harita IoU 0.514-0.538 / 0.498-0.514
  free_space MAE 0.213-0.228 / 0.222-0.229     etkin boyut 128/128
  bilgi: harita IoU@0.5 0.251-0.332 / 0.342-0.402; hedef recall@0.5 0.565-0.597 / 0.692-0.695
.venv/bin/python scripts/run_vae_audit.py runs/vision_dataset_d1_p40_20260914 runs/vae_quality_v3_20260923/quality_report.json runs/vae_audit_v1_20260923
audit_passed=True  (held-out audit bölümü, 768 kare, tek kullanım; 24 s)
  AUROC 0.809   AP 0.892   en iyi eşikte harita IoU 0.569 (eşik 0.35)
  free_space MAE 0.203   grayscale free_space MAE 0.233   etkin boyut 128   PSNR 23.2 dB
  bilgi: harita IoU@0.5 0.458; hedef precision/recall@0.5 0.865/0.670; grayscale hedef AUROC 0.778
  onaylı checkpoint: runs/vae_quality_v3_20260923/seed_20260911/visual_vae_v1.pt (sha256 44bde1a8...)
  manifest: runs/vae_audit_v1_20260923/approval_manifest.json (visual-vae-approval-manifest-v1)
```

### Bilinen sınırlılıklar ve riskler

- Kapı kaldırma bir tanım değişikliğidir; sabit eşikli IoU (`0.25-0.33`)
  probe'un seyrek pozitiflerde tutucu kaldığını göstermeye devam eder.
- Grayscale audit'te hedef AUROC `0.78`'e düşer (kapı değil); renk ipucunun
  kaybı hedef ayrımını zayıflatır.
- Audit bölümü tek kullanımlıktır; ikinci bir VAE adayı yeni bir audit
  bölümü ister.

### Sonraki önerilen adım

Dilim 2 (aynı gün, üstteki kayıt) ve ardından Dilim 3: `ppo_latent_d1a.toml`
ile üç seed D1a eğitimi.

## 23 Eylül 2026 — Raporlar: ilerleme raporu ve RL/CNN sistem raporu

### Amaç ve kapsam

Kullanıcı isteğiyle iki anlaşılır Türkçe rapor hazırlandı: (1) projenin
başından bugüne ilerleme raporu, terimler sözlüğüyle; (2) RL sistemi ve CNN
yapısını anlatan rapor: observation space, action space, reward
fonksiyonunun 18 bileşeni, hover durum makinesi, iki kodlayıcı mimarisi,
eğitim düzeni. Kod değişmedi.

### Oluşturulan dosyalar

```text
docs/reports/2026-09-23-ilerleme-raporu.md              proje kaydı kopyası
docs/reports/2026-09-23-rl-sistemi-ve-cnn-yapisi.md     proje kaydı kopyası
Canlı belgeler (yorum/düzenleme): claude.ai Claude Docs, bağlantılar dosyaların başında
```

### Notlar

- Raporlardaki sayılar `docs/PROGRESS_LOG.md`, `docs/REVIEW_FINDINGS.md`,
  ADR-015/017/019/020/021/022, `vision_navigation.toml`, `target_area.toml`,
  `camera.toml`, `ppo_vision_d1a.toml` ve `visual_vae.toml`'dan alındı.
- Bekleyen karar (sabit `0.5` eşikli harita IoU kapısı) raporlarda açık
  olarak işaretlendi; karar gelince raporlar güncellenmeli.

### Sonraki önerilen adım

Bekleyen kapı kararı; ardından üç seed kalite + audit ve Dilim 2.

## 22 Eylül 2026 — Dilim 1 kapanışı: ADR-022 Revizyon 1, latent 128 ve onaylı `visual-vae-v1` zinciri

### Amaç ve kapsam

REV-049'un üç seçeneğinden kullanıcı 1 numarayı onayladı: hedef kapısını
sabit eşikli recall/precision yerine eşikten bağımsız çözülebilirlik yapmak.
Bu dilim ADR-022 Revizyon 1'i, kodun yeni kapılara geçişini, `beta 1 +
latent 128` ile üç seed kalite koşusunu ve audit'i kapsar. Ortam tarafı
(Dilim 2) başlamadı.

### Oluşturulan/değiştirilen dosyalar

```text
docs/decisions/ADR-022-...md          Revizyon 1 bölümü; §2/§3/§4 güncellemesi
                                      (latent 128, vision_latent 512, toplam 528)
src/ebabil/perception/probes.py       ThresholdFreeMetrics, binary_auroc,
                                      average_precision, threshold_free_metrics,
                                      evaluate_probe_split (sabit eşik + eşiksiz)
src/ebabil/perception/vae_quality.py  kapılar: target_auroc, target_average_precision,
                                      obstacle_map_best_iou; REQUIRED_GATES yenilendi;
                                      sabit eşik P/R yalnız bilgi alanı
src/ebabil/configs/training/visual_vae.toml   latent_size 128; yeni kapı eşikleri
src/ebabil/perception/__init__.py     dışa aktarımlar
tests/test_visual_vae.py              16 test: AUROC/AP tanımları, yeni kapılar,
                                      eski sabit eşikli zincirin fail-closed reddi
docs/REVIEW_FINDINGS.md               REV-049 karar ve sonuç
docs/PROGRESS_LOG.md, IMPLEMENTATION_PLAN.md, README.md   durum
```

### Alınan teknik kararlar ve gerekçeleri

1. **Kapı tanımı değişti, ölçüt gevşetilmedi.** Sabit `0.5` eşiği probe'un
   kalibrasyonunu ölçüyordu; AUROC/AP latent'in bilgi taşıyıp taşımadığını
   ölçer. Politika doğrusal değildir ve kendi eşiğini öğrenir. Eşikler
   ölçülen en iyi varyantın (`beta 1 + latent 128`, AUROC `0.81 / 0.82`, AP
   `0.88 / 0.90`, en iyi eşikte IoU `0.54 / 0.51`) altında değil, hemen
   üstünde konuldu ki üç seed'in tamamı geçmek zorunda olsun.
2. **Sabit eşik değerleri raporlanmaya devam eder** (`*_value` alanları),
   böylece etiketli zincirle karşılaştırma kaybolmaz.
3. **Eski zincir reddedilir.** `REQUIRED_GATES` artık `target_auroc`,
   `target_average_precision` ve `obstacle_map_best_iou` ister; ilk koşunun
   raporu bu anahtarları içermediği için `load_approved_visual_vae` onu
   fail-closed reddeder (test).
4. **AUROC bağ-duyarlı Mann-Whitney, AP sıralama tabanlı;** tek sınıflı
   dejenere durumda AUROC `0.5`, AP `0.0` (muhafazakâr). Tanımlar testle
   sabitlendi.
5. **Latent 128.** Tarama, kapasitenin tek olumlu kaldıraç olduğunu ölçtü;
   Dilim 2 observation'ı `4 x 128 = 512` latent + `16` = `528` boyut olacak.

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/python scripts/run_vae_quality.py runs/vision_dataset_d1_p40_20260914 runs/vae_quality_v2_20260922
quality_passed=False  (3 seed; takılan tek kapı: validation obstacle_map_iou@0.5,
                       seed 20260912 0.251 ve 20260913 0.275 < 0.30; diğer 8 kapı 3 seed x 2 split geçti)
(süre ~26 dk; aggregate: AUROC 0.807/0.819, AP 0.883/0.902, best-IoU 0.525/0.506,
 free_space 0.222/0.227, recall@0.5 0.577/0.693, precision@0.5 0.878/0.861)
.venv/bin/python scripts/run_vae_audit.py runs/vision_dataset_d1_p40_20260914 runs/vae_quality_v2_20260922/quality_report.json runs/vae_audit_v2_20260922
koşulmadı (kalite geçmedi)
ruff check / format --check : temiz
mypy                        : Success, 52 source files
pytest                      : 302 passed, 110 subtests passed in 54.0 s
```

### Bilinen sınırlılıklar ve riskler

- Eşikten bağımsız kapı sabit eşikli kapıdan daha zayıf bir garantidir;
  hedef bulma politikaya kalır ve bu ancak hover aşamasında görünür. REV-049
  seçenek 3 (ayrı işaret dedektörü) açık tutuldu.
- Etiketsiz latent hedefi etiketli encoder kadar keskin ayırmaz (AUROC
  `~0.81` vs etiketli recall `0.92`).
- Engel haritası `0.5` eşiğinde IoU `0.30`'a yakın geçer (`0.33 / 0.40`);
  kapı payı dardır ve seed'e duyarlı olabilir.
- Latent 128 observation'ı büyütür (`528`); PPO örnek verimliliği Dilim 3'te
  ölçülür.

### Sonraki önerilen adım

Ek karar (kullanıcı): sabit `0.5` eşiğindeki harita IoU kapısını kaldırıp
yalnız en iyi eşikte IoU (`>= 0.45`) ile devam etmek. Onay gelirse config
ve `REQUIRED_GATES` güncellenir, üç seed kalite (deterministik, aynı
seed'ler, `~26 dk`) ve audit tekrarlanır, ADR-022 Revizyon 1 notu
tamamlanır; ardından Dilim 2 (`vision-v3`, latent `4 x 128`).

## 22 Eylül 2026 — Dilim 1: etiketsiz VAE kodlayıcı, probe ölçümü ve onay zinciri

### Amaç ve kapsam

ADR-022 kademe 1'in algı yarısı: yalnız actor RGB arşivleriyle eğitilen VAE
kodlayıcı, donmuş latent üzerinde yalnız ölçüm amaçlı doğrusal probe'lar ve
ADR-014 desenini koruyan yeni bir onay zinciri (`visual-vae`). Ortam tarafı
(`vision-v3`, `EbabilVisionNavigation-v1`) Dilim 2'dir ve bu dilimde
dokunulmadı. Etiketli zincir (`vision-encoder-v11`, `quality.py`) bit-bit
korundu.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/vae.py              yeni: VisualVAEConfig/TrainingConfig,
                                          ActorOnlyNpzDataset, validate_actor_archives,
                                          VisualVAE, vae_loss, kl_per_dimension,
                                          evaluate_visual_vae (etiketsiz metrikler),
                                          train_visual_vae, checkpoint yükleme,
                                          encode_actor_rgb; REV-049 kaldıraçları:
                                          photometric_augment_batch,
                                          random_resized_crop_batch, info_nce_loss,
                                          projeksiyon başlığı (varsayılan kapalı)
src/ebabil/perception/probes.py           yeni: ProbeConfig/ProbeLabels/LinearProbes,
                                          collect_probe_data, fit/predict/evaluate,
                                          probe_visual_vae
src/ebabil/perception/vae_quality.py      yeni: VisualVAEQualityConfig,
                                          run_visual_vae_quality (3 seed),
                                          run_visual_vae_audit (tek seferlik),
                                          load_approved_visual_vae (fail-closed)
src/ebabil/perception/__init__.py         dışa aktarımlar
src/ebabil/configs/training/visual_vae.toml   yeni
scripts/train_visual_vae.py               yeni
scripts/run_vae_quality.py                yeni
scripts/run_vae_audit.py                  yeni
tests/test_visual_vae.py                  yeni, 15 test
docs/REVIEW_FINDINGS.md                   REV-049 (kapı ölçümü, tarama, karar noktası)
docs/decisions/ADR-022-...md              Ölçülen sonuçlar bölümü
docs/IMPLEMENTATION_PLAN.md, README.md    Dilim 1 durumu
```

### Alınan teknik kararlar ve gerekçeleri

1. **Trainer privileged dosyaya dokunmaz ve bu testle zorlanır.** Etiketli
   zincirin `validate_dataset_leakage`'ı privileged arşivleri açtığı için VAE
   trainer'ı onu çağırmaz; yalnız actor allowlist'ini denetleyen
   `validate_actor_archives` yazıldı. Test iki yoldan doğrular: `numpy.load`
   casusu hiçbir `privileged/` yolu görmemeli ve privileged dizini silinmiş
   dataset kopyasında eğitim çalışmalı.
2. **Probe'lar ayrı modülde ve ayrı dosyada raporlanır.** `probes.py`
   privileged etiketleri okur; seed dizininde `metrics.json` etiketsiz,
   `probe_metrics.json` etiketli ölçümdür. Probe ağırlıkları checkpoint'e
   yazılmaz. Probe'lar her zaman `train` split latent'inde eğitilir ve
   train dışı split'lerde ölçülür; audit'te de aynı kural.
3. **Probe eğitimi deterministik kapalı form/L-BFGS.** Serbest alan ve merkez
   için ridge regresyon (kapalı form, bias düzenlileştirilmez), görünürlük
   ve 9x12 engel haritası için sıfırdan başlayan L-BFGS lojistik regresyon.
   Latent standardize edilir. Rastgelelik yoktur; aynı checkpoint aynı
   probe sonucunu verir.
4. **Metrik tanımları etiketli zincirle birebir.** `compute_metrics` yeniden
   kullanıldı; `free_space_mae`, `target_precision/recall`,
   `obstacle_map_iou/precision/recall` aynı formülle hesaplanır, böylece
   ADR-014/021 eşikleri anlamını korur.
5. **Kayıp ölçeği.** Yeniden üretim hatası piksel TOPLAMI (3x72x96), KL boyut
   toplamı, `beta = 1.0`, 5 epoch KL ısınması, free bits kapalı. Etkin latent
   boyutu iki ölçütle raporlanır: ortalama KL `> 0.01 nat` (kapı) ve
   posterior ortalamasının veri kümesi varyansı `> 0.01` (bilgi).
6. **Mimari.** Dört stride-2 konvolüsyon (16/32/64/64), 72x96 -> 4x6, latent
   64, ayna kod çözücü. Kodlayıcı + ortalama başlığı `205 808` parametre,
   toplam `511 395`. Adaptive exposure normalizasyonu etiketli encoder ile
   aynı formül (etiketsiz olduğu için korundu). Kodlayıcı runtime'a, kod
   çözücü yalnız eğitime.
7. **Onay zinciri şemaları ayrı.** `visual-vae-quality-report-v1`,
   `visual-vae-audit-report-v1`, `visual-vae-approval-manifest-v1`, hepsi
   `kind = "visual-vae"`. Etiketli zincirin manifest'i etiketsiz yükleyiciden
   geçemez (test), tersi de geçemez (şema farkı). Yükleyici ADR-022 §3
   kapılarının (`active_latent_units`, `free_space_mae`, `target_recall`,
   `obstacle_map_iou`) hem quality hem audit `checks` içinde bulunmasını ve
   geçilmiş olmasını ister; eksikse fail-closed.
8. **Bellek önbelleği.** `ActorOnlyNpzDataset` decompress edilmiş uint8
   kareleri bellekte tutar; 4608 train karesi `~1.1 GB`. Bir epoch'un
   maliyeti npz açmadan çok konvolüsyona düştü.
9. **İki etiketsiz kaldıraç daha eklendi, ikisi de varsayılan kapalı**
   (REV-049 taraması için): fotometrik değişmezlik (girişe seed'li kanal
   kazancı/gamma/ofset/gürültü, kod çözücü temiz görüntüyü üretir;
   `forward(images, target_images=...)`) ve SimCLR/CURL tarzı kontrastif
   yardımcı kayıp (iki rastgele kırpılmış + bozulmuş görünüm, InfoNCE,
   yalnız eğitimde kullanılan projeksiyon başlığı). Eski checkpoint'ler
   `contrastive_projection_size` olmadan yüklenmeye devam eder.
10. **Geliştirme taraması yalnız development split'lerinde.** Üç tur, on bir
    varyant, tek seed; audit split'ine dokunulmadı, eşik değiştirilmedi.
    Tarama script'i scratch'te tutuldu (`runs/vae_sweep_v1_20260922`
    altında her varyantın checkpoint, `metrics.json` ve
    `probe_metrics.json` dosyaları var).

### Çalıştırılan komutlar/testler ve sonuçları

```text
.venv/bin/ruff check src scripts tests          : All checks passed
.venv/bin/ruff format --check src scripts tests : temiz
.venv/bin/mypy                                  : Success, 52 source files
.venv/bin/python -m pytest -q                   : 298 passed, 110 subtests passed
.venv/bin/python -m pytest -q tests/test_visual_vae.py : 12 passed, 3 subtests
```

Tek epoch zamanlaması (gerçek D1 dataset'i, 6528 kare, CPU):
`29 s` toplam (actor doğrulaması + önbellek + eğitim + iki split x nominal/grayscale
değerlendirme); kodlayıcı gecikmesi `0.64 ms`/kare.

Üç seed kalite koşusu (`runs/vae_quality_v1_20260922`, 45 epoch, `~23 dk`):

```text
.venv/bin/python scripts/run_vae_quality.py runs/vision_dataset_d1_p40_20260914 runs/vae_quality_v1_20260922
quality_passed=False   (üç seed, iki split: target_recall; validation: obstacle_map_iou)
audit                 : koşulmadı (kapı geçilmedi)
```

Geçen kapılar (3 seed ort., validation / development_ood): etkin boyut
`64/64`, free_space MAE `0.248 / 0.237`, target precision `0.862 / 0.832`.
Geçmeyenler: target recall `0.597 / 0.685` (`>= 0.85`), harita IoU
`0.268 / 0.333` (`>= 0.30`, validation).

Geliştirme taraması (`runs/vae_sweep_v1_20260922`, tek seed, 11 varyant,
`~2.5 saat`): `beta 0.1/0.01`, latent 128, 108x144, fotometrik, kontrastif
ve kombinasyonları. Hiçbiri hedef recall'u `0.61`'in (validation) üstüne
taşımadı; `beta 1 + latent 128` diğer beş kapıyı geçti. Eşikten bağımsız
ölçümde (AUROC/AP/en iyi F1) harita kalibrasyon meselesi çıktı (en iyi eşikte
IoU `0.50-0.56`), hedef görünürlüğü bilgi meselesi (en iyi AUROC `0.81`).
Tam tablo ve teşhis REV-049'dadır.

Doğrulama (kod son hâli):

```text
ruff check / format --check : temiz
mypy                        : Success, 52 source files
pytest                      : 301 passed, 110 subtests passed in 45.82s
```

### Bilinen sınırlılıklar ve riskler

- Probe kapıları etiketli zincirin eşikleridir; etiketsiz temsilin bunları
  doğrusal probe ile sağlaması güçlü bir istektir. Geçilmezse eşik
  düşürülmez; kaldıraçlar `beta`, latent boyutu ve kontrastif yardımcı
  kayıptır (ADR-022 §3).
- Yeniden üretim hatası kapısı şimdilik etkisizdir (`1.0`); ilk ölçümden
  sonra kalibre edilecek.
- Probe'lar doğrusal olduğu için latent'te doğrusal olmayan biçimde kodlanmış
  bilgi ölçümde görünmez; kapı bu anlamda muhafazakârdır (yanlış negatif
  verebilir, yanlış pozitif vermez).
- `active_latent_units` KL ölçütü zayıf bir eşiktir (`0.01 nat`); varyans
  ölçütü bilgi amaçlı raporlanır ve ikisi birlikte okunmalıdır.
- Dataset 40 derece pitch zincirinindir (`vision_dataset_d1_p40_20260914`);
  kamera geometrisi değişirse dataset ve VAE zinciri yenilenmek zorundadır
  (REV-044 dersi).
- CPU eğitimi; MPS denenmedi (determinizm için bilinçli).

### Sonraki önerilen adım

Kullanıcı kararı: REV-049'daki üç seçenekten biri. Öneri, hedef kapısını
eşikten bağımsız çözülebilirlik (AUROC `>= 0.80`, AP `>= 0.85`) ve harita
için en iyi eşikte IoU `>= 0.45` olarak yeniden tanımlayan ADR-022
revizyonu; ardından `beta 1 + latent 128` ile üç seed kalite + audit ve
Dilim 2 (`vision-v3`, latent `4 x 128 = 512` boyut). Karar gelmeden ADR ve
kapı config'i değiştirilmeyecek.

## 22 Eylül 2026 — Yön değişikliği: etiketsiz görsel temsil (ADR-022) ve implementasyon planı

### Amaç ve kapsam

Yeni oturum. Kullanıcı mevcut yapıyı "drone engellerin konumunu biliyor"
diyerek reddetti ve engel konumu bilgisi almayan, CNN ile öğrenen bir
politika istedi. Bu dilim yalnız inceleme, karar ve planlamadır; **kod
değişmedi**. Üç iş yapıldı: (1) proje ve dokümanların tam incelemesi ve
iddianın kodda doğrulanması, (2) kullanıcının sağladığı derin araştırma
raporunun proje durumuyla eşleştirilmesi, (3) kararların ADR-022 ve
implementasyon planı §23 olarak yazılması.

### Geriye dönük kayıt: 13-14 Eylül 2026 koşuları

Bu koşular ve REV-040..046 bulguları ilerleme günlüğüne yazılmamıştı (REV-047).
Aşağıdaki tablo `runs/*/metadata.json` ve `docs/REVIEW_FINDINGS.md`'den
**yeniden kurulmuştur**; koşuları yapan agent'ın notu değildir.

| Koşu | Başlangıç (UTC) | Adım | Sonuç | Bulgu |
|---|---|---:|---|---|
| `ladder-d1a-seed20260912-ABORTED-goal-at-boundary` | 13 Eyl 22:03 | iptal | hedeflerin `%56`'sı kenara `2 m`'den yakın | REV-040 |
| `ladder2-d1a-seed20260912-ABORTED-heading-decoupled` | 13 Eyl 23:18 | iptal | hedef yakınında yön hatası `101-123 derece` | REV-041 |
| `ladder3-d1a-seed20260912-ABORTED-no-deceleration` | 13 Eyl 23:55 | iptal | terminal fazda hız `%99.9` ihlal | REV-042 |
| `ladder4-d1a-seed20260912` | 14 Eyl 00:40 | `600k` | `%0` başarı; `5/5` koşul `%0.0`, en uzun dizi `0` adım | REV-043 |
| `p40-d1a-seed20260912` | 14 Eyl 10:37 | `300k` | `%0`; çarpışma `%32 -> %37` | REV-044 |
| `hold-hold05-seed20260912-ABORTED-wrong-axis` | 14 Eyl 11:42 | iptal | `0.5 s` hold ile de `5/5` `%0` | REV-045 |
| `rev046-d1a-seed20260912-ABORTED-contradictory-altitude` | 14 Eyl 13:24 | iptal | `minimum_altitude` `%79` | REV-046 |
| `rev046b-d1a-seed20260912-ABORTED-premature-descent` | 14 Eyl 13:36 | iptal | `minimum_altitude` `%70` | REV-046 |
| `rev046c-d1a-seed20260912` | 14 Eyl 13:48 | `600k` | `%0` başarı; eval `0/10`; `ep_rew_mean -14.8` | REV-046 |

Aynı dönemde kamera pitch'i `45 -> 40 derece`, hover yüksekliği `1.5 -> 1.1 m`,
`cruise_altitude_m 2.156 -> 1.422`, zemin yarı-boyutu `21.13 -> 26.88 m`
yapıldı ve onay zinciri yenilendi:
`vision_dataset_d1_p40_20260914 -> vision_quality_v23_20260914 ->
vision_audit_v5_20260914 -> vision-encoder-v11` (REV-044). Ayrıntı
REVIEW_FINDINGS REV-040..046.

### İnceleme bulgusu: iddia kodda nasıl karşılık buluyor

- Actor observation'ında (`vision-v2`, 194 boyut) engel dünya koordinatı,
  depth, raycast veya maske yok; `tests/test_vision_navigation_env.py`
  sızıntı testleri bunu zorluyor.
- "Bilme" üç kapıdan giriyor: (1) encoder MuJoCo depth/segmentation türevi
  etiketlerle supervised eğitilip dondurulmuş, (2) reward'da privileged
  rangefinder `proximity` ve gerçek hedef, (3) araç konumu hatasız (ADR-019).
  Kullanıcının reddettiği mekanizma birincisidir.
- Engelli bütün koşular `%0`; algı hatası `3.07 -> 1.02 m` düzelmesine
  rağmen. Kalan iki duvar (REV-039 keşif, REV-043 hover kesişimi) algı
  yönteminden bağımsız.

### Raporun proje ile eşleşmesi

Kullanıcının derin araştırma raporu satır satır karşılaştırıldı. Alınanlar:
VAE ile etiketsiz temsil (raporun kendi önerisi de saf uçtan uca yerine
budur), `k = 3-4` kare yığınlama, reset anında görsel DR, DroNet tarzı hız
modülasyonu (baseline/güvenlik filtresi adayı), `r_clearance` teriminin mevcut
`proximity` ile eşdeğerliği. Zaten var olanlar: hız komutu + 500 Hz kademeli
kontrolcü ayrımı, gövde eksenli hız, sol/merkez/sağ sektör serbest alanı,
MJCF kamera ve headless renderer. Çelişen veya uygulanamayanlar: FastDepth
tarzı depth-supervised ağ (etiket bağımlılığının aynısı), MJX paralellik
iddiası (darboğaz render + encoder, `0.35x` legacy hız), optik akış/TTC
(hover'da tanımsız), ImageNet normalizasyonu (ön eğitimli omurga yok),
`MuJoCo-Drones-Gym` (projenin ortamı aynı özellikleri taşıyor). Raporda
atıf yer tutucuları (`[cite: 24, 25]`) kalmıştı; DroNet için verilen
`1710.02110` arXiv kimliği yanlış çıktı (kuantum fiziği makalesi); doğru
kaynak RA-L 2018, DOI `10.1109/LRA.2018.2795643`. Diğer beş atıf doğrulandı.

### Oluşturulan/değiştirilen dosyalar

```text
docs/decisions/ADR-022-label-free-visual-representation.md   yeni
docs/REVIEW_FINDINGS.md            REV-047 (günlük boşluğu), REV-048 (etiket
                                   bağımlılığı, DR yokluğu, zamansal bilgi yokluğu)
docs/PROGRESS_LOG.md               Güncel durum yeniden yazıldı; bu kayıt;
                                   geriye dönük 13-14 Eylül tablosu
docs/IMPLEMENTATION_PLAN.md        yürütme tablosu, §17 A9/A10, §21 durumu,
                                   yeni §23 (kademeli plan ve kapılar)
docs/PROJECT_DESIGN.md             §2 görsel temsil satırı, §6.1 vision-v3
                                   notu, §14 kapsam notu
docs/LITERATURE_REVIEW.md          §2.16 etiketsiz temsil ve hafif monoküler
                                   ağlar; §3.3, §4, §5 güncellemeleri
README.md                          durum maddeleri ve ADR-022 bağlantısı
```

### Alınan teknik kararlar ve gerekçeleri

1. **Actor algısı etiketsiz eğitilir** (ADR-022 §1). Depth/segmentation
   türevi etiketlerle eğitilmiş hiçbir ağ actor'a görsel kanal veremez;
   etiketler yalnız ölçüm, privileged reward/termination ve tanı içindir.
2. **Birinci kademe donmuş VAE, ikinci kademe ince ayar.** Saf piksel PPO
   örnek verimliliği yüzünden ablation'a alındı; `~135 adım/s` ile milyonlarca
   adım gün mertebesidir.
3. **Onay zinciri korunur, kapılar probe'a taşınır.** Donmuş `z` üzerinde
   doğrusal probe'lar ADR-014/021 eşiklerini (free_space MAE `0.30`, hedef
   P/R `0.78/0.85`, harita IoU `0.30`) sağlamak zorunda; probe ağırlıkları
   hiçbir yere gitmez. Kapı geçilmezse eşik düşürülmez.
4. **`vision-v3`:** `vision_latent (256)` + `fc_state (10)` + `waypoint (4)`
   + `timing (2)` = `272`; `obstacle_map` ve `vision_aux` çıkar. Yığın latent
   düzeyinde, reset'te kopyalama ile dolar.
5. **Hover kapısı privileged görünürlüğe taşınır** (segmentation maskesi,
   `>= 8` piksel). Actor hedef görünürlüğünü `z`'den çıkarmak zorunda.
6. **Görsel DR ortama taşınır, geometri çeşitlenir.** Bugün engel rengi sabit
   ve yalnız kutu (REV-048).
7. **Action ve reward değişmez; `proximity` açık kalır.** Sabahki "varsayılan
   kapalı" önerisi geri alındı: şekillendirme actor'a sızmaz ve keşif
   (REV-039) bağlayıcı problemdir. `proximity_weight = 0` ablation'dır.
8. **Ertelenenler:** pitch jitter (`+-3 derece` sınırıyla, hover geometrisi
   doğrulanmadan açılmaz), dinamik DR (Faz 8), optik akış/TTC, FastDepth.
9. **Kademe 1 evaluation'ı** `arrival_rate` raporlar; `hover_success` ve
   ADR-018 D1 kapısı değişmez.

### Çalıştırılan komutlar/testler ve sonuçları

Kod değişmediği için baseline olarak çalıştırıldı:

```text
.venv/bin/python -m pytest -q                   : 286 passed, 107 subtests passed, 44.3 s
.venv/bin/ruff check src scripts tests          : All checks passed
.venv/bin/ruff format --check src scripts tests : 93 files already formatted
.venv/bin/mypy                                  : Success, 49 source files
```

Ortam: Apple M3, 16 GB, `torch 2.14.0` (MPS mevcut), `mujoco 3.13.0`,
`stable-baselines3 2.9.0`. `runs/` toplam `5.1 GB`.

Atıf doğrulaması (arXiv/RPG sayfaları): VAE `1312.6114`, World Models
`1803.10122`, SAC+AE `1910.01741`, CURL `2004.04136`, FastDepth `1903.03273`
doğrulandı; DroNet RA-L 2018 (`10.1109/LRA.2018.2795643`, `rpg.ifi.uzh.ch`).

### Bilinen sınırlılıklar ve riskler

- Hiçbir şey ölçülmedi; ADR-022 kapıları ilk ölçümden sonra kalibre edilecek.
- VAE latent'i ince engelleri zayıf temsil edebilir; probe kapısı bunu
  eğitimden önce yakalar, fakat kapı geçilmezse `beta`/latent/kontrastif
  yardımcı kayıp denemesi gerekir.
- Kare başına ek segmentation render'ı ve fotometrik etkiler throughput'u
  düşürebilir; ölçülmeden bütçe verilmez.
- REV-043 hover kesişimi duvarı bu kararla çözülmez; `arrival_rate` bu
  yüzden ayrı raporlanır. ADR-012 düzeyinde karar kullanıcıya aittir.
- Tek seed geleneği sürüyor; kademe 1 kabulü üç seed ister.

### Sonraki önerilen adım

Plan §23 Dilim 1: `src/ebabil/perception/vae.py` (kodlayıcı/kod çözücü,
kayıp, trainer, checkpoint), `probes.py` (ölçüm probe'ları), quality/audit
runner'ları ve `load_approved_visual_vae`; `tests/test_visual_vae.py` içinde
privileged dosyaların hiç açılmadığını zorlayan test. İlk eğitim
`runs/vision_dataset_d1_p40_20260914` üzerinde, üç seed, ardından audit.

## 14 Eylül 2026 — Algı yenilemesi: `vision-v2`, `vision-encoder-v11` ve zorluk merdiveni bulgusu

### Amaç ve kapsam

Kullanıcı onayıyla algı tarafını baştan kurmak: dataset'i D1 ortamında
yeniden üretmek, encoder'ın ürettiği fakat politikaya verilmeyen engel
haritasını ölçüp observation'a almak, onay zincirini yeniden kurmak ve
politikayı yeni gözle eğitmek. Ardından code ve doc review.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/perception/dataset.py        build_scene_model'e ground_half_size_m
                                        geçirildi (REV-030 dataset tarafı)
src/ebabil/perception/model.py          obstacle_map_iou/precision/recall
                                        metrikleri; v11 şeması kabul
src/ebabil/perception/quality.py        harita kapıları; gate_collision_risk
                                        opt-in; _require_gated_obstacle_map
                                        (fail-closed provenance)
src/ebabil/envs/vision.py               vision-v2: obstacle_map (108,),
                                        vision_aux 9 -> 6, sürüm mesajları
src/ebabil/evaluation/replay.py         action boyutu yorumu güncellendi
scripts/generate_vision_dataset.py      --scene-config
src/ebabil/configs/training/
  vision_dataset_d1.toml                yeni (6528 kare, D1 poz dağılımı)
  vision_encoder_v11.toml               yeni (harita kapıları, risk kapısı off)
  vision_encoder.toml, _v10.toml        geriye dönük harita eşikleri (0.0)
  ppo_vision_d1.toml                    yeni onaylı zincir
  ppo_vision_d0v.toml                   artık koşulamaz durumu yazıldı
docs/decisions/ADR-021-...md            yeni
docs/decisions/ADR-015-...md            sürüm notları (v1 artık yürürlükte değil)
docs/decisions/ADR-017-...md            referans güncellemesi
docs/REVIEW_FINDINGS.md                 REV-036, REV-037, REV-038, REV-039
tests/                                  10 yeni test
```

### Teknik kararlar

**1. Dataset D1 ortamında yeniden üretildi.** `736 -> 6528` kare. En kritik
düzeltme poz dağılımındaydı: eski irtifa üst sınırı `2.10 m`, seyir
irtifasının (`2.156 m`, REV-036) **altındaydı**, yani encoder politikanın
uçtuğu irtifayı hiç görmemişti. Yeni band `1.20-2.90 m`.

**2. `obstacle_map` kanalı politikaya verildi (ADR-021).** Encoder zaten
`9 x 12` engel haritası üretip `obstacle_mask`'e karşı eğitiyordu; politikaya
verilmiyor ve kalitesi hiç ölçülmüyordu. Kamera `45 derece` aşağı baktığı için
harita satırları zemin mesafesine, kolonları yön açısına karşılık gelir; yani
yerel mesafe x yön doluluk ızgarasıdır. Üç dikey dilimlik `free_space`
koridorun *nerede* olduğunu taşıyamıyordu.

**3. `collision_risk` kanalı çıkarıldı (REV-038).** Etiketi tanım gereği
`free_space < 0.25`, yani politikanın zaten sahip olduğu kanalın eşiklenmiş
kopyası. Kalite kapısında tek takılan şey buydu; eşiği düşürmek yerine kanalı
sözleşmeden çıkarmak seçildi ve gerekçe sırası kayda geçirildi. Kapı opt-in
oldu (`gate_collision_risk`, varsayılan `true`), yani `v9`/`v10` zincirinin
anlamı korunur.

**4. Fail-closed provenance kontrolü eklendi.** `vision-v2` engel haritasını
actor'a verdiği için `load_approved_vision_encoder` artık zincirde harita
kapısının **geçilmiş** olmasını şart koşar. Eski zincirler (`vision_audit_v2`)
reddedilir. Bu review sırasında bulundu: aksi halde v9 encoder ile de
`obstacle_map` açılıyordu ve o harita hiç ölçülmemişti.

**5. Sızıntı güvencesi isimden ölçüme taşındı.** İsim tabanlı yasak listesi
`"obstacle"` token'ını reddediyordu; liste `obstacle_mask`/`segmentation`'a
daraltıldı ve yerine kanalın encoder çıktısıyla birebir aynı olduğunu ve actor
kamerasının ayrıcalıklı çıktı üretmediğini kanıtlayan test kondu.

### Çalıştırılan komutlar/testler ve sonuçları

```text
dataset  : 6528 kayıt, leakage_check=passed, 1.9 GB
kalite   : vision_quality_v22_20260913, 3 seed, quality_passed=True
audit    : vision_audit_v4_20260913, audit_passed=True
encoder  : vision-encoder-v11, sha256 cf69f096...
ruff     : All checks passed
mypy     : 49 source file temiz
pytest   : 276 passed, 92 subtests passed
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23 (değişmedi)
```

Algı kalitesi, `development_ood`, 3 seed:

| Metrik | v9 (legacy) | v11 (D1) |
|---|---:|---:|
| `free_space_mae` | `2.06 m` | **`1.05 m`** |
| rollout içi `free_space` hatası | `3.07 m` | **`1.26 m`** |
| `risk_precision` | `0.709` | `0.849` |
| `target_recall` | `0.929` | `0.921` |
| engel haritası IoU | ölçülmemiş | **`0.651`** (audit `0.614`) |

### Politika sonucu: başarı %0

`runs/d1-v11-seed20260912`, 40 episode deterministik: `final` `%0` başarı /
`%47.5` çarpışma, `best` `%0` / `%75`. Eğitimde çarpışma `%67 -> %64`, hiç
başarı yok (`2067` episode).

### Kalan kök neden ölçüldü (REV-039)

Koridor genişliği yoğunlukla neredeyse değişmiyor (`2.70 -> 2.10 m`), değişen
şey **ardışık karar sayısı**:

| Engel | Koridorda (`+-1.5 m`) | `%20` hata ile hayatta kalma |
|---:|---:|---:|
| `1` | `1.0` | `%80` |
| `10` | `2.0` | `%63` |
| `40` | `6.1` | `%26` |

`2067` episode'un hiçbirinde başarı olmadığı için politika `success_bonus`'u
hiç örneklemedi. Terminal görev için pozitif sinyal yok; yalnız çarpışma
cezasından kaçınma var. Bu bir keşif problemidir.

ADR-018 engel zorluğunu zaten kademeli bir eksen olarak tanımlıyor
(tek engel -> seyrek -> geçit -> koridor -> dar geçit -> yoğun karma).
REV-034 düzeltmesiyle birinci basamaktan doğrudan yüksek bir basamağa
atlanmıştı.

### Bilinen sınırlılıklar

- Algı `1.26 m` hataya indi fakat koridor `2.10 m`; oran `1.7` ve engel
  `4 m`'den yakınken tahmin hâlâ **iyimser** tarafta (`%54`, ortalama
  `2.44` vs gerçek `1.95 m`). İyimser sapma çarpışmaya yol açar.
- Engel haritası `precision 0.782`, yani beşte bir hücre yanlış pozitif.
- Harita satır-mesafe eşlemesi düz zemin varsayar.
- `collision_risk` modelde kalıyor fakat tüketicisi yok; sadeleştirme ayrı iş.
- Observation `89 -> 194` boyuta büyüdü; örnek verimliliğini etkileyebilir.
- D0-V config'i artık koşulamaz (eski zincir fail-closed reddediliyor);
  tarihsel sonuçları `vision-v1`/`action-v1` ile alınmıştır.
- Tek seed. ADR-009 üç ister.
- Dataset `1.9 GB`; `runs/` toplam `3.1 GB`.

### Sonraki adım

REV-039 yoğunluk merdiveni: `5 -> 10 -> 20 -> 40` engel, her basamak
öncekinin politikasından warm start. Observation sözleşmesi sabit kaldığı için
bu mümkün. Kabul ölçütü basamak başına: başarı önceki basamağın altına
düşmemeli, çarpışma trendi azalan olmalı.

## 13 Eylül 2026 — D1 spesifikasyonu, kaçınmanın hiç öğrenilmediğinin bulunması

### Amaç ve kapsam

Kullanıcı iki temel soru sordu ve ikisi de yapısal hata ortaya çıkardı:

1. *"D1 eğitimi engelli senaryo ile olması gerekmiyor muydu?"* -> REV-034:
   yedi koşu da **tek** engelle yapıldı, ADR-018 `~40` istiyordu.
2. *"BU DRONE ENGELLERİ AŞMIYOR Kİ? ÜSTÜNDEN UÇUYOR."* -> REV-035: engeller
   tavandan `0.60 m` kısaydı, politika üstündeki koridordan uçuyordu.

### REV-034: yedi koşu spesifikasyon altında yapıldı

ADR-018 D1 satırı `0.10/m^2`, `~40` engel diyor. Bütün koşular
`scene_mode = "single_blocker"` (tek engel) ile yapıldı; `obstacle_count = 40`
yalnız `random_obstacles` modunda okunur. Seçim hiçbir yerde karar olarak
kaydedilmemişti.

Sıfır-shot sonda bedeli ölçtü (7. koşu `best`, 20 episode):

| | 1 engel (eğitildiği) | 40 engel (spesifikasyon) |
|---|---:|---:|
| Başarı | `%30.0` | `%25.0` |
| Çarpışma | `%0.0` | **`%30.0`** |

Bu, "D1 kapısının çarpışma yarısı geçildi" ifadesini geçersiz kıldı.

Ayrıca kullanıcı kararıyla rota bandı `5-15 m` -> **`16-22 m`** yapıldı:
`20 x 20 m` alanda `5 m`'lik rota anlamsızdı, araç alanı hiç geçmiyordu.
Örnekleme kutusu `18 x 18 m` olduğu için yeni band karşı kenardan başlamak
demektir.

### Spesifikasyona uygun ilk D1 koşusu: başarı %0

`runs/d1-spec-seed20260912`, `600k`, 40 engel, `16-22 m` rota. 40 episode,
deterministik:

| | `final` | `best` |
|---|---:|---:|
| Başarı | `%0.0` | `%0.0` |
| Çarpışma | `%20.0` | `%10.0` |
| Timeout | `%67.5` | `%80.0` |

Eğitim trendi sağlıklı görünüyordu: çarpışma `%45 -> %13`, sınır ihlali
`%21 -> %11`, hedef kaybı `%15 -> %1`, reward `-39 -> +1.3`. Araç hedef
bölgesine episode'ların `%88`'inde varıyor, en yakın anda `0.179 m`'ye kadar
yaklaşıyor.

Hold koşulları ölçüldü ve **irtifa `%100` ihlal** çıktı. Ardından irtifalar
ölçüldü:

```text
hold irtifa bandı     : 1.31 - 1.71 m
engel yükseklikleri   : 2.33 - 2.39 m
seyir irtifası         : medyan 2.51 m
terminal irtifa        : medyan 2.21 m
banda giren adım       : %0.0
bandın üstünde kalan   : %100.0
```

### REV-035: kaçınma hiç öğrenilmemişti

Kullanıcı demoyu izleyip sorunu gördü. Engellerin tepesi `2.40 m`, tavan
`3.00 m`, yani `0.60 m`'lik bir koridor vardı ve politika **tam o koridordan**
uçuyordu (`2.36-2.51 m`). Sıfır yatay kaçınma manevrası.

Yani eğitim trendindeki `%45 -> %13` çarpışma düşüşü kaçınma öğrenmesi değil
**tırmanma** öğrenmesiydi. Ve "hedefin üzerine geliyor fakat inmiyor" gözlemi
de aynı şeyin sonucuydu: araç hiç inmemişti.

**Kendi kodumuzla çelişki:** `has_navigable_xy_path` sahneyi 2 boyutlu grid
üzerinde doğrular, yani üretici her sahnede "etrafından dolaşılabilir" garanti
eder (60/60 geçiyordu). Üstünden uçmak bu doğrulamayı anlamsız kılıyordu. Bu
bir zorluk ayarı değil, fizik ile sahne doğrulaması arasındaki tutarsızlığın
giderilmesidir.

Düzeltme: engel yüksekliği tavana sabitlendi (`z` yarı-boyutu `1.50`,
yükseklik `3.00 m`, boşluk `0.00 m`). Yatay boyutlar değişmedi.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/configs/environment/
  static_scene_d1.toml     endpoint separation 5-15 -> 16-22 m (gerekçeli)
                           obstacle z yarı-boyutu 0.40-1.20 -> 1.50 (gerekçeli)
src/ebabil/configs/training/
  ppo_vision_d1.toml       scene_mode single_blocker -> random_obstacles
docs/REVIEW_FINDINGS.md    REV-034, REV-035
```

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff check : All checks passed
mypy       : 49 source file temiz
pytest     : 267 passed, 86 subtests passed
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23 (değişmedi)
```

Yeni ortam doğrulaması (60 sahne):

```text
çözülebilir 2B yol var : 60/60
hedef hover zarfı boş  : 60/60
doğrudan rota kapalı   : 60/60
hedefe en yakın engel  : medyan 2.28 m, en yakın 1.20 m
geçit açıklığı         : medyan 1.92 m, %10 dilim 0.67 m, %6.3'ü 0.50 m altı
```

### Bu dilimin dersi

İki hata da **eğitim metriklerinden görünmüyordu.** Çarpışma oranının düşmesi,
ödülün yükselmesi, hedefe varma oranının `%88` olması — hepsi sağlıklı
görünüyordu. Hata ancak (a) spesifikasyonla karşılaştırınca ve (b) davranışı
**gözle izleyince** ortaya çıktı.

Bundan sonra bir eğitim sonucu raporlanmadan önce:
1. Konfigürasyon ADR spesifikasyonuyla karşılaştırılmalı.
2. En az bir episode replay olarak izlenmeli; metrikler davranışın yerine
   geçmez.

### Bilinen sınırlılıklar

- Yükseklik çeşitliliği kaldırıldı (hepsi `3.00 m`). Karışık yükseklik daha
  gerçekçi olurdu fakat üstünden uçma kaçışını kısmen geri açar; önce tam
  kapalı hâlde taban ölçümü alınır.
- Encoder daha da dağılım dışında: `6 x 5 m` odada `0.80-2.40 m` engellerle
  eğitildi, kamera artık tavana kadar duvar görecek.
- REV-033 `progress_refund`, REV-030 zemin düzeltmesi ve ADR-020 holonomik
  action bu koşuda birlikte aktif; katkıları ayrıştırılmadı.
- Tek seed. ADR-009 üç ister.

### Sonraki adım

`runs/d1-walls-seed20260912` koşuyor. İlk kez yatay kaçınma **zorunlu**.
Beklenti: çarpışma oranı başta yüksek olacak (ilk 45 episode'da 22 çarpışma,
13 yere çakılma) ve kapı geçilmeyecek. Ölçülecek asıl şey politikanın
**etrafından dolaşmayı** öğrenip öğrenmediği; bunun için replay izlenmeli,
metrik yetmez.

## 12 Eylül 2026 — İlk geçerli D1 ölçümü ve ADR-020 holonomik action

### Amaç ve kapsam

İki iş: (1) REV-030 düzeltmesinden sonra ilk **geçerli** D1 taban ölçümünü
almak, (2) kullanıcının yönlendirmesiyle aracın uçuş mekanizmasını gerçeğe
uygun hâle getirmek.

Kullanıcı sensör kusurlarını (IMU gürültüsü, gecikme, rüzgâr) "şimdilik erken"
diyerek kapsam dışı bıraktı ve uçuş mekanizmasına odaklanılmasını istedi.

### İlk geçerli D1 ölçümü (`runs/d1-rev030-seed20260912`)

`600k`, seed `20260912`, `v9` encoder, düzeltilmiş zemin, `5.0/m` progress,
`hover_height 1.5`, `target_visible 0.005`. 40 episode, deterministik,
seed `520001`:

| Model | Başarı | Çarpışma | Timeout |
|---|---:|---:|---:|
| 1. koşu final (kırık zeminde eğitilmiş, düzeltilmişte ölçülmüş) | `%12.5` | `%42.5` | `%40.0` |
| **5. koşu `best`** | **`%15.0`** | **`%10.0`** | `%70.0` |
| 5. koşu `final` | `%2.5` | `%40.0` | `%57.5` |

Doğru karşılaştırma ilk satırdır; `%42.5` sayısı kırık ortamın eseriydi.
Buna göre `best` her eksende daha iyi: başarı `%12.5 -> %15`, çarpışma
`%42.5 -> %10`, ve **kamera gerektiren** episode'larda (hedef hatası
`> 0.65 m`) başarı `%6 -> %16`.

Eğitim trendi sağlıklıydı: çarpışma `%30 -> %20`, `target_lost` `%15 -> %1`,
reward `-14.4 -> +22.4`. Kapı (`>= %85`) geçilmedi.

### Bağlayıcı kısıt ölçüldü: action uzayı (REV-032)

Araç hedefin `0.65 m` içine **giriyor** fakat kalamıyor:

```text
hedefin 0.65 m içine giren episode           : %65
başarı                                       : %15
terminal adımlarda güvenli bölge dışı ihlali : %85.6
yatay mesafe medyanı                         : 1.038 m
```

Dört reward tasarımı ve zemin düzeltmesi boyunca bu medyan hiç `0.8 m`'nin
altına inmedi. Geometrik sebep:

```text
hold koşulu           : merkeze <= 0.650 m, hız <= 0.25 m/s, 3 s kesintisiz
hedef görünür kalması : merkeze >= 0.407 m
uygulanabilir halka   : 0.243 m
action uzayı (v1)     : yalnız ileri; yanal ve geri YOK
yatay FOV             : 75 derece -> +-37.5 derece
```

Yanal düzeltme yaw çevirmeyi gerektiriyor, `38 dereceden` fazla çevirince hedef
kadraj dışına çıkıyor. İnce düzeltmenin çıkışı yoktu.

### ADR-020: holonomik action sözleşmesi

```text
v1: a = [ileri(>=0), dikey, yaw]                3 boyut
v2: a = [ileri/geri, yanal, dikey, yaw]         4 boyut, hepsi işaretli
    yatay hız BÜYÜKLÜĞÜ 1.5 m/s ile sınırlı (yöne göre değişmez)
```

Hız zarfı **büyütülmedi**; yalnız yön kısıtı kaldırıldı. Gerekçe: gerçek bir
quadrotor yatay düzlemde izotropiktir ve legacy ortam (ADR-008) bunu dört
boyutlu action ile zaten yapıyordu. ADR-015 bir **sensör** kısıtını
(tek ileri kameranın kör alanı) **aktüatör** kısıtına çevirmişti.

Kör alan riski koddan kaldırılıp `collision_penalty`'ye bırakıldı. Ölçülen
risk: `random_obstacles` sahnelerinde görüş konisi dışında `2 m`'den yakın
engel sayısı görünür alandakinin `3` katı (606 vs 193 tespit). Gerçek
sistemler bunu gittiği yöne dönerek çözer; öğrenilecek bir davranıştır.

**Observation sözleşmesi değişmedi** ve bu önemli: encoder yalnız görüntü
tüketir, action uzayına hiç bağlı değildir (doğrulandı). Onaylı zincir
geçerliliğini korur, dataset yenilemesi gerekmez, yalnız politika yeniden
eğitilir. Sürümleme ayrıldı: `VISION_OBSERVATION_VERSION = "vision-v1"`
değişmedi, `VISION_ACTION_VERSION = "vision-action-v2"` eklendi.

### Bu dilimde ayrıca yanıtlanan soru: engel nasıl tanınıyor

Kullanıcı üç kez sordu, cevabı kayda geçiyor: **tanınmıyor, söylendi.**

1. Dataset üretilirken MuJoCo'nun segmentasyon ve derinlik tamponları
   "bu pikseller `obstacle_` nesnesi, şu kadar uzakta" bilgisini **bedava**
   veriyor. Çıkarsama yok, etiket hazır.
2. Ağ bu etiketlemeyi `512` kare üzerinde taklit etmeyi öğreniyor.
3. Uçuşta tanıma adımı yok: donmuş ağ pikselleri 6 sayıya çeviriyor
   (`free_space` L/C/R + `risk` L/C/R). Nesne tespiti, "bu bir kutu" çıkarımı
   yok.
4. Encoder RL sırasında **donmuş**; çarpışmalardan hiçbir şey öğrenmiyor.

Renk kısayolu kullanıp kullanmadığı ölçüldü (engeller kırmızımsı kahve
`0.55, 0.22, 0.12`, zemin gri):

```text
normal RGB      : risk_balanced_accuracy 0.681 | derinlik hatası 2.05 m
hue kaldırılmış : 0.641 | 2.19 m
sadece renk eşiği: 0.500 (rastgele) | 4.40 m
```

Rengi silmek yalnız `0.04` kaybettiriyor, yani model şekil/parlaklık yapısını
kullanıyor, rengi ezberlemiyor. **Fakat zayıf:** `2 m` derinlik hatası ve
`risk_specificity 0.53` ("engel yok" derken yazı-turaya yakın).

Gerçeklik açısından asıl nokta: etiketleri üreten piksel bazlı derinlik +
segmentasyon gerçek bir drone'da **yok**. Engel algılama problemi çözülmedi;
simülatörün cevabını kopyalayan bir vekil kondu.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/envs/vision.py               ACTION_SIZE 3 -> 4, holonomik
                                        _velocity_command, büyüklük kırpma,
                                        VISION_ACTION_VERSION
src/ebabil/envs/__init__.py             VISION_ACTION_VERSION ihracı
src/ebabil/configs/environment/
  vision_navigation.toml                [action] gerekçesi yazıldı
scripts/smoke_vision_navigation_env.py  action boyutu action_space'ten okunur
tests/test_vision_navigation_env.py     test_holonomic_action_... (yeniden yazıldı)
tests/test_terminal_hover.py            action boyutları 3 -> 4
docs/decisions/ADR-020-...md            yeni
docs/REVIEW_FINDINGS.md                 REV-032
```

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff check    : All checks passed
mypy          : 49 source file temiz
pytest        : 262 passed, 83 subtests passed
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23  (değişmedi)
```

Holonomik davranış doğrulandı:

```text
ileri  [1,0,0,0] -> [+1.50, 0.00, 0.00]  büyüklük 1.50 m/s
geri   [-1,0,0,0]-> [-1.50, 0.00, 0.00]  büyüklük 1.50 m/s
sağa   [0,1,0,0] -> [ 0.00,+1.50, 0.00]  büyüklük 1.50 m/s
çapraz [1,1,0,0] -> [+1.06,+1.06, 0.00]  büyüklük 1.50 m/s  (sqrt(2) sızıntısı yok)
```

### Bilinen sınırlılıklar

- **Çarpışma oranı yükselebilir.** Politika artık göremediği yöne uçabiliyor.
  Bu beklenen ve ölçülecek sonuç; ilk `4%` penceresinde çarpışma `%44`
  görünüyor, fakat erken eğitim bunu söylemeye yetmez.
- Engel algısının zayıflığı bu dilimde **çözülmedi**. Aracın gidebileceği
  yönleri artırmak gördüğü şeyi iyileştirmez; iki iş ayrıdır.
- Sensör kusurları (gürültü, `40 ms` gecikme, `%1` veri kaybı) kod olarak var
  fakat vision ortamı `sensor_mode="ideal"` kullanıyor. Kullanıcı kararıyla
  kapsam dışı bırakıldı.
- Rüzgâr modellenmiyor (simülasyonda destek var, ortam sıfır rüzgârla koşuyor).
- Engel yerleşimi rastgeleleştirmesi (REV-031) hazır ve **kapalı**; ADR-020
  ölçüldükten sonra açılacak.
- Tek seed. ADR-009 üç ister.

### Sonraki adım

`runs/d1-adr020-seed20260912` başlatıldı (`600k`, aynı seed, aynı encoder).
Kabul ölçütü:

1. Terminal adımlarda yatay mesafe medyanı `0.869 m`'nin altına inmeli
   (düzeltilmiş zeminde en iyi ölçüm).
2. Başarı `%15`'in üstüne çıkmalı.
3. Çarpışma oranı raporlanmalı; yükselirse kör alan davranışının öğrenilmediği
   anlamına gelir ve ayrı bir bulgu olur.

Ardından REV-031 engel rastgeleleştirmesi, sonra engel algısı (dataset
büyütme + uzamsal harita).

## 12 Eylül 2026 — REV-030: D1'in asıl blocker'ı bulundu (görsel zemin düzlemi)

### Amaç ve kapsam

Kullanıcı D1 ortamını MuJoCo viewer'da görmek istedi ("belki sorun engel
yerleşimindedir"). Viewer'ı D1 sahnesini gösterebilecek hâle getirmek gerekti;
ekran görüntülerinde engellerin çoğunun zeminin **dışında** durduğu görüldü.
Araştırma D1'in asıl blocker'ını ortaya çıkardı ve bu, dört koşuluk reward
çalışmasının neden hiç işe yaramadığını tek başına açıklıyor.

### Bulgu

`quadrotor.xml`: `<geom name="ground" type="plane" size="5 5 0.1" .../>`

MuJoCo'da `plane` geom'in `size`'ı **yalnız render** içindir; çarpışmada düzlem
**sonsuzdur**. İkisi de ölçüldü:

```text
fizik  : x = 0 / 4 / 9 / 15 m -> zemin teması var, fizik doğru
render : x,y > ~4 m -> hiçbir zemin pikseli yok, derinlik 532.81 m (boşluk)
```

Zemin **görsel olarak 10 x 10 m**, D1 workspace'i **20 x 20 m**. 300 sahnede:

```text
HEDEF zeminin dışında                     : %65
başlangıç zeminin dışında                 : %70
rotanın zemin dışı kısmı                  : medyan %39
rotası tamamen zemin içinde olan episode  : %7
```

Kamera kareleri (40 poz, `%50` civarı normaldir — kadrajın üst yarısı gökyüzü):

| Poz ailesi | boşluk piksel | hiç zemin görmeyen poz |
|---|---:|---:|
| encoder dataset (legacy `6 x 5 m`) | `%61.8` | **`%0`** |
| D1 (`20 x 20 m`) | `%86.7` | **`%52`** |

Onaylı encoder legacy `6 x 5 m` workspace'te eğitildi ve o workspace görsel
zeminin tamamen içindedir. Encoder "zemin yok" karesini **hiç görmedi**; D1'de
pozların yarısında tam olarak bunu görüyor.

### Bu bulgu her şeyi açıklıyor

| Daha önce ayrı ayrı açıklamaya çalıştığım gözlem | Bu bulguyla |
|---|---|
| D0-V `20/20`, D1 `%25-42` | D0-V workspace'i zeminin tamamen içinde |
| Başarının seed'e duyarlılığı (`%25` vs `%42.5`) | Episode merkeze yakın mı değil mi |
| Yüksek `target_lost` abort | Hedef pedi render edilse de çevresi boş; encoder OOD |
| **Hiçbir reward değişikliğinin işe yaramaması** | Sorun ödülde değil observation'da |
| Çarpışmanın kısa rotalarda yoğunlaşması | `5-8 m`'de hedefin `%61`'i zemin dışı |

REV-026'dan REV-029'a kadarki dört koşuluk reward çalışması bu kırık
observation'ın üzerinde yürütüldü. REV-029'un "çarpışma oranı ödül tasarımına
duyarlı değil" ölçümü de bununla tutarlı: bağlayıcı kısıt ödül değildi.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/scenes/generator.py          LEGACY_GROUND_HALF_SIZE_M,
                                        SceneGeneratorConfig.ground_half_size_m,
                                        required_ground_half_size_m(),
                                        build_scene_model(ground_half_size_m=...)
src/ebabil/envs/vision.py               scene_config.ground_half_size_m bağlandı
src/ebabil/configs/environment/
  static_scene_d1.toml                  ground_half_size_m = 21.13 (gerekçesiyle)
scripts/view_static_scene.py            --scene-config, --endpoint-mode, --seeds,
                                        engel yerleşimi raporu, zemin boyutu
scripts/record_vision_replay.py         --scene-config, --scene-mode, --endpoint-mode
tests/test_scenes.py                    GroundPlaneExtentTests (5 test)
docs/REVIEW_FINDINGS.md                 REV-030
docs/PHASE_7_D1_EVIDENCE_RUN.md         engel sayısı düzeltmesi (40 -> 1)
```

### Teknik kararlar

**1. Zemin boyutu sahne config'ine bağlandı, MJCF'e sabitlenmedi.** Her
curriculum seviyesi kendi workspace'ini kapsayan zemini ister; tek bir global
değer legacy'yi kırar.

```text
gereken = workspace köşesi + kamera zemin izinin tavan irtifasındaki uzak sınırı
legacy  : 3.0  + 3.71 * 2.5 = 12.28 m
D1      : 10.0 + 3.71 * 3.0 = 21.13 m
```

**2. Legacy bilinçli olarak `5.0 m`'de bırakıldı.** Gereken `12.28 m`, yani
legacy de teknik olarak eksik. Fakat ADR-008 legacy sahnesi ve onaylı encoder
zinciri (`vision_dataset_v4b/v4c`, `vision-encoder-v9/v10`) bu değerle
üretildi; değiştirmek approval zincirini geçersiz kılar ve dataset → kalite
(3 seed) → audit yeniden koşulmasını gerektirir. Legacy config'te alan yoktur,
varsayılan korunur. Bu, bilinen bir eksiğin bilinçli olarak dondurulmasıdır.

**3. `obstacle_count = 40` yanlış anlaşılmıştı.** D1 koşularının tamamı
`scene_mode = "single_blocker"` ile yapıldı, yani sahnede **tek** engel var.
`obstacle_count` yalnız `random_obstacles` modunda kullanılır. Ölçülen yerleşim:
engel her zaman rotanın **tam ortasında** (`%49.2-%50.0`) ve **tam çizgi
üzerinde** (yanal sapma `0.000000 m`), boyutu `0.3-0.9 m` genişlik /
`1.2-2.4 m` yükseklik. `PHASE_7_D1_EVIDENCE_RUN.md` düzeltildi.

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff check    : All checks passed
mypy          : 49 source file temiz
pytest        : 258 passed, 70 subtests passed  (öncesi 253/66)
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23        değişmedi
legacy kamera boşluk oranı   %61.8 -> %61.8        değişmedi
D1 kamera boşluk oranı       %86.7 -> %51.1
D1 hiç zemin görmeyen poz    %52   -> %0
```

### Bilinen sınırlılıklar

- **Encoder hâlâ legacy dağılımında eğitilmiş.** Düzeltme D1 karelerini
  encoder'ın eğitim dağılımına *doğru* taşır (`%86.7 -> %51.1`, legacy
  `%61.8`'in yakınına) fakat encoder `6 x 5 m` workspace'te, `5 m` zeminle,
  `+-1.25 m` yanal offset ve `0.80-2.10 m` irtifa bandında üretilmiş bir
  dataset'le eğitildi. D1'in `20 x 20 m` geometrisi bu dağılımın dışındadır.
  Yani bu düzeltme D1'i **çözülebilir** hâle getirir, "çözülmüş" hâle
  getirmez.
- Legacy zemini de eksik (`5.0` vs gereken `12.28 m`) ve bilinçli olarak
  donduruldu. D0-V sonuçları bu eksikle alındı; `6 x 5 m` workspace'te etkisi
  ölçüldü ve hiç zemin görmeyen poz `%0` olduğu için sonuçlar geçerliliğini
  korur.
- `single_blocker` yerleşimi çeşitlilik açısından fazla dar (her zaman tam orta
  nokta, tam çizgi üzerinde). Politika genel kaçınma yerine tek duruma uyum
  sağlıyor olabilir. Bu ayrı bir konudur ve henüz ele alınmadı.
- `runs/d1-rev029-seed20260912` koşusu `%43`'te durduruldu ve
  `-ABORTED-broken-ground` olarak yeniden adlandırıldı; kırık ortamda
  koştuğu için sonucu yorumlanamaz.

### Sonraki adım

`runs/d1-rev030-seed20260912` başlatıldı: düzeltilmiş zemin + REV-029 sonrası
reward (`5.0/m` sabit progress) + `hover_height 1.5` + `target_visible`.

Kabul ölçütü:

1. **waypoint'e** mesafe medyanı `0.591 m`'nin altına inmeli.
2. Hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı.
3. Deterministik başarı `%42.5`'in üstüne çıkmalı.

Sonuç hâlâ kapının altındaysa sıradaki iş dataset yenilemesidir: D1
workspace'inde, doğru zeminle, genişletilmiş poz ailesiyle; ardından approval
zinciri (kalite 3 seed + bağımsız audit) yeniden kurulur.

## 12 Eylül 2026 — REV-029: REV-026 yanlış teşhis çıktı, progress geri alındı

### Amaç ve kapsam

Dördüncü D1 koşusunun sonucunu ölçmek ve neden başarısız olduğunu bulmak.
Sonuç: **REV-026 bulgusunun kendisi yanlıştı.** Ona dayanan iki düzeltme iki
koşu harcadı ve ikisi de geriletti. Kod REV-026 öncesi forma geri alındı.

### Dördüncü koşunun sonucu

`runs/d1-rev028-seed20260912`, `600k`, aynı seed, aynı `v9` encoder.
40 episode, deterministik, seed `520001`:

| Koşu | progress formu | waypoint'e mesafe | gerçek hedefe | Başarı |
|---|---|---:|---:|---:|
| 1 (kanıt) | `5.0/m` sabit | `0.591 m` | `0.653 m` | **`%42.5`** |
| 3 | `15.0 * dd / L` (`1.40/m`) | `0.991 m` | `0.917 m` | `%22.5` |
| 4 `final` | `10.0/L` + `5.0/m` | `1.709 m` | `1.548 m` | `%0.0` |
| 4 `best` | aynı | `2.399 m` | `1.472 m` | `%0.0` |

Ödül ayrıştırması nedeni gösterdi (10 episode, deterministik):

```text
              3. koşu    4. koşu
target_progress  +11.71     +4.26     <- navigasyon hiç tamamlanmıyor
     hover_hold   +7.44     +0.00
        success   +9.00     +0.00
         return  +14.69     -8.38
```

Kruvaziyer yoğunluğu `1.40/m`'den `0.93/m`'ye indirmek (terminal ağırlığa yer
açmak için) navigasyonu tamamen bozdu. Araç hedefe hiç varmıyor.

### REV-026'nın çürütülmesi

Bu noktada asıl iddiayı geriye dönük test ettim. Rota uzunluğu
`segment_budget_seconds`'tan geri hesaplanıp üç koşunun tamamı sınıflandırıldı.

**Çürütme 1 — çarpışma uzun değil KISA rotalarda daha sık:**

| Rota | 1. koşu | 3. koşu | 4. koşu |
|---|---:|---:|---:|
| `5-8 m` | **`%23.3`** | **`%24.2`** | **`%22.5`** |
| `8-11 m` | `%15.1` | `%12.3` | `%10.5` |
| `11-15 m` | `%7.6` | `%7.5` | `%8.3` |

REV-026 tam tersini öngörüyordu. Desen üç farklı ödül tasarımında da aynı, yani
çarpışma oranı progress ağırlığına duyarlı **değil**. Nedeni geometrik:
`20 x 20 m` workspace'te `5-8 m`'lik rotada blocker manevra alanının çok daha
büyük kısmını kapatır.

**Çürütme 2 — eğitim boyunca artış survivorship'ti:**

| Pencere | çarpışma (ham) | hedefe ulaşan | **ulaşanlar arasında** | rota `5-8 m` |
|---|---:|---:|---:|---:|
| 1 | `%15` | `%53` | `%27` | `%23` |
| 2 | `%12` | `%66` | `%12` | `%21` |
| 3 | `%13` | `%60` | `%15` | `%22` |
| 4 | `%13` | `%66` | `%18` | `%23` |
| 5 | `%17` | `%69` | `%17` | `%26` |
| 6 | `%20` | `%74` | `%18` | `%25` |

Hedef bölgesine ulaşan episode oranı `%53 -> %74` yükseldi (politika uçmayı
öğrendi, `out_of_bounds` ve `minimum_altitude` düştü) ve daha çok episode engel
yoğun terminal bölgeye kadar hayatta kaldı. Koşullu oranlar sabit.

Yani politika çarpışmayı öğrenmiyordu; daha çok episode çarpışabileceği yere
kadar gidiyordu.

### Doğru invariant yoğunluk üzerinde

REV-026'nın invariantı (`toplam progress < collision_penalty`) yanlış
kurulmuştu. Çarpışma episode'u **bitirir**, bu yüzden politikanın kıyaslaması
episode toplamı değil kısa yolun **yerel** kazancıdır:

```text
w_metre * dolaşma_mesafesi <= collision_penalty     5.0 * 5 m = 25.0 <= 25.0
```

Bu kısıt `L`'ye hiç bakmadığı için rota uzunluğundan doğal olarak bağımsızdır.
Normalizasyon gereksizdi; `L`'ye bölmek yoğunluğu `L` ile ters orantılı yaptı ve
hem navigasyonu hem terminal hassasiyeti bozdu.
`TerminalHoverRewardConfig.detour_saving_for()` olarak uygulandı.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/rewards/terminal_hover.py    progress REV-026 öncesi forma döndü;
                                        detour_saving_for eklendi;
                                        cruise/terminal ayrımı, progress_reward_bound,
                                        planned_route_length_m ve
                                        terminal_progress_radius_m kaldırıldı
src/ebabil/envs/vision_config.py        iki parametre yüklemesi geri alındı,
                                        _validate_progress_cannot_outweigh_collision silindi
src/ebabil/envs/vision.py               call-site geri alındı
src/ebabil/configs/environment/
  vision_navigation.toml                target_progress_weight = 5.0 (gerekçesiyle)
tests/test_terminal_hover.py            yoğunluk ve detour testleri
docs/decisions/ADR-017-...md            §5 progress kararı geri alındı
docs/REVIEW_FINDINGS.md                 REV-029; REV-026 GEÇERSİZ; REV-028 notu
```

### Geri alınmayanlar

İkisi de kendi ekseninde ölçülerek fayda gösterdi ve korunuyor:

| Değişiklik | Ölçülen etki |
|---|---|
| REV-023 `hover_height_m = 1.5` | irtifa ihlali `%65 -> %17`, hold halkası `2.25` kat |
| REV-027 `target_visible = 0.005` | eğitimde `target_lost` abort `%30 -> %6` |

Bu üçlü (`5.0/m` progress + `1.5 m` hover + görünürlük ödülü) **hiç
koşulmamıştır.**

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff format   : temiz
ruff check    : All checks passed
mypy          : 49 source file temiz
pytest        : 253 passed, 66 subtests passed in 29.00s
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23  (değişmedi)
```

### Bu dilimin dersi

Teşhisi **koşullu** oranlarla test etmeden düzeltme uygulamak iki koşu (yaklaşık
`2.5` saat) ve iki ADR revizyonu maliyeti çıkardı. "Çarpışma oranı eğitim
boyunca arttı" gözlemi tek başına bir teşvik sorunu kanıtı değildir; episode'un
o noktaya kadar hayatta kalma olasılığı da değişiyorsa ham oran yanıltır.

Bundan sonra bir reward bulgusu şu iki kontrolden geçmeden düzeltme önerisine
çevrilmemelidir:
1. Etki, iddia edilen değişkene göre ayrıştırıldığında da görünüyor mu
   (burada: rota uzunluğu)?
2. Ham oran yerine koşullu oran da aynı yönü gösteriyor mu?

### Bilinen sınırlılıklar

- Önerilen üçlü henüz koşulmadı; `%42.5` hâlâ en iyi ölçülmüş sonuç ve o koşuda
  hover yüksekliği `2.0 m` ile görünürlük ödülü yoktu.
- REV-028'in **ölçümü** geçerli (araç waypoint'in üzerinde duramıyor) fakat o
  gerileme normalizasyondan kaynaklandığı için geri alma ile ortadan kalkmalı;
  bu doğrulanmadı.
- D3 ölçekleme sorunu açık: `5.0/m` ile D3 toplamı `~500` vs `success_bonus 30`.
  Çözümü progress'i bölmek değil `success`/`hold`'u seviyeye göre ölçeklemek.
- Çarpışmanın kısa rotalarda yoğunlaşması **açıklanmış fakat çözülmemiştir**.
  Bu bir sahne/curriculum sorunudur: `5-8 m` rotada blocker manevra alanının
  çok büyük kısmını kapatıyor. D1 kapısı `<= %5` isterken kısa rotalardaki
  `%23` bunu tek başına imkânsız kılabilir; sahne üretiminin bu rejimde
  çözülebilirliği ölçülmelidir.

### Sonraki adım

Beşinci D1 koşusu: `5.0/m` progress + `hover_height 1.5` + `target_visible`.
Kabul ölçütü:

1. **waypoint'e** mesafe medyanı `0.591 m`'nin altına inmeli.
2. Hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı.

Ardından kısa rota çözülebilirliği ölçülmelidir; D1 kapısının `<= %5` çarpışma
koşulu ona bağlı.

## 12 Eylül 2026 — REV-028: iki menzilli progress ve dördüncü D1 koşusu

### Amaç ve kapsam

Üçüncü D1 koşusunda ölçülen gerilemeyi, REV-026 kazanımını geri almadan
düzeltmek. Kullanıcı onayıyla uygulandı.

Sorun tek cümleyle: progress'i rota uzunluğuna bölmek çarpışma teşvikini doğru
biçimde kapattı, fakat son metrenin metre başına değerini `5.00`'dan `1.40`'a
indirdi ve araç **konumunu bildiği bir noktanın üzerinde bile duramaz** hâle
geldi (waypoint'e mesafe medyanı `0.591 -> 0.991 m`).

### Kullanıcının sorusu bu dilimi yönlendirdi

Kullanıcı şunu sordu: *"Drone sonuçta hedefin konumunu biliyor. Kamera ile
waypointi göremese dahi ona doğru hareket edip tam üstünde durabilmeli, buna
engel olan şey ne?"*

Bu soru bir ölçüm eksiğini açığa çıkardı. O ana kadar yalnız **gerçek** hedefe
olan mesafe ölçülmüştü. Waypoint'e olan mesafe ayrı ölçüldü
(`believed_horizontal_distance_m`, 40 episode, deterministik, seed `520001`):

| | kanıt koşusu `final` | üçüncü koşu `final` |
|---|---:|---:|
| **waypoint'e** mesafe medyanı | `0.591 m` | `0.991 m` |
| gerçek hedefe mesafe medyanı | `0.653 m` | `0.917 m` |
| waypoint'e en yakın anın medyanı | `0.417 m` | `0.661 m` |
| waypoint'in `0.65 m` içinde kalan episode | `%53` | `%25` |

**Önceki teşhis yanlıştı ve düzeltildi.** "Araç inanılan konuma park ediyor"
demiştim; dayanağı terminal mesafe medyanının (`1.316 m`) ortalama hedef
hatasına (`1.338 m`) yakın olmasıydı ve bu bir çakışmaydı, kanıt değildi. Araç
waypoint'ten de `~1 m` uzaktadır: sorun bilgi değil **konum tutma
hassasiyetidir**.

Aynı ölçümde saf waypoint takibinin tavanı da hesaplandı. Hedef hatası diskte
düzgün dağıldığı için:

```text
P(hata <= 0.65 m) = (0.65 / 2.0)^2 = %10.6
```

Hedef hatası `> 0.65 m` olan episode'larda, yani kamerasız kazanılması imkânsız
olanlarda başarı kanıt koşusunda `%38`, üçüncü koşuda `%25`. İkisi de `%10.6`
tavanının üzerinde, yani **ADR-019'un kamera iddiası çalışıyor**; gerileme o
eksende değil.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/rewards/terminal_hover.py    iki menzilli progress, _positive,
                                        progress_reward_bound (yeniden adlandırma)
src/ebabil/envs/vision_config.py        iki ağırlığın yüklenmesi,
                                        _validate_progress_cannot_outweigh_collision
src/ebabil/envs/vision.py               terminal_progress_radius_m call-site'a bağlandı
src/ebabil/configs/environment/
  vision_navigation.toml                cruise 10.0 / terminal 5.0 (gerekçeleriyle)
tests/test_terminal_hover.py            6 yeni/yeniden yazılmış progress testi
docs/decisions/ADR-017-...md            §5 nihai form, kapı 14-17
docs/REVIEW_FINDINGS.md                 REV-028 çözüldü, REV-026 notu
```

### Teknik kararlar

**1. İki menzilli progress.**

```text
d > R   :  w_cruise   * delta_d / L_plan     w_cruise   = 10.0
d <= R  :  w_terminal * delta_d              w_terminal =  5.0
R = terminal_hover.approach_radius_m = 2.5 m
```

Sınırı geçen adımda fark sınırda ikiye bölünür
(`max(d,R)` ve `min(d,R)` farkları), yani iki parçanın toplamı her zaman
`d_önce - d_sonra`'dır ve hiçbir mesafe kaybolmaz veya iki kez sayılmaz.

**Neden iki özellik birlikte sağlanır:** kruvaziyer parça normalize olduğu için
rota uzunluğuyla büyümez; terminal parça normalize edilmez fakat uzunluğu `R`
ile üst sınırlı olduğundan o da büyümez.

| | tam normalize | iki menzilli |
|---|---:|---:|
| son 1 m, D1 (`10.7 m`) | `1.40` | **`5.00`** |
| son 1 m, D3 (`100 m`) | `0.15` | **`5.00`** |
| toplam üst sınırı | `15.00` | `22.50` |
| `collision_penalty` | `25.00` | `25.00` |

**2. Yarıçap için ayrı config alanı açılmadı.** `terminal_progress_radius_m`
fonksiyon parametresidir ve tek kaynağı `terminal_hover.approach_radius_m`'dir.
Alternatif, aynı sayıyı iki TOML bölümünde tutup eşitliklerini kontrol etmekti;
tek kaynak daha az kırılgan.

**3. Invariant iki yerde zorlanıyor.** `TerminalHoverRewardConfig.validate()`
ağırlıkları tek tek sınar; yarıçapı görmediği için toplam sınırı
`VisionNavigationEnvConfig.validate()` içinde sınanır. Bir agent ağırlığı
sessizce yükseltirse config **yüklenmez**.

**4. `route_independent_progress_total` -> `progress_reward_bound`.** Eski ad
yanıltıcıydı: gerçek toplam `L_plan` ile artar ve sınıra yalnız `L_plan >> R`
iken yaklaşır. Rota bağımsız olan toplam değil **sınırdır**.

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff format   : temiz
ruff check    : All checks passed
mypy          : 49 source file temiz
pytest        : 257 passed, 63 subtests passed in 29.24s  (öncesi 254/64)
```

```text
LEGACY_ROLLOUT_SHA256 = 03f16232adae103e370fe2a5cbcb2d0c62900838f90b51fc7db408e14e3f5e23  (değişmedi)
```

Vision env smoke (`single_blocker`, seed `20260912`, 1024 PPO adımı) iki menzilli
progress'i ve görünürlük bileşenini üretti; sonlu ve beklenen işaretlerde.

### Yeni eklenen testler

| Test | Kanıtladığı |
|---|---|
| `test_progress_bound_does_not_grow_with_route_length` | `500 m` toplamı `10 m` toplamının `1.5` katından az |
| `test_terminal_progress_density_is_route_independent` | Son 1 m her `L_plan` için `w_terminal` |
| `test_progress_splits_at_the_terminal_radius` | Sınır geçişinde farkın bölünmesi, toplamın korunması |
| `test_collision_penalty_dominates_progress_at_every_route_length` | REV-026 invariantı |
| `test_shipped_config_enforces_the_collision_invariant` | Invariantın config yüklemesinde fail-closed olması |
| `test_progress_normalization_rejects_invalid_ranges` | `<= 0`, `inf`, `NaN` menzil reddi |

### Bilinen sınırlılıklar

- `R = 2.5 m` seçimi `approach_radius_m`'den miras alındı, ayrıca kalibre
  edilmedi. Terminal menzilin hedef belirsizliği `R_hedef = 2 m`'yi kapsaması
  mantıklıdır fakat bu bir ölçüm değil gerekçeli seçimdir.
- `w_terminal = 5.0` kanıt koşusunun yoğunluğunu geri getirmek için seçildi;
  daha yükseği denenmedi. Üst sınır invariant tarafından belirlenir: `R = 2.5 m`
  ile `w_terminal < (25.0 - 10.0) / 2.5 = 6.0`.
- Tek seed. ADR-009 üç ister.
- `hover_hold` hâlâ normalize edilmemiştir (REV-022) ve bu dilimde dokunulmadı.
- Bu dilim dördüncü D1 koşusudur ve reward dengesi üç kez değişti. Değişikliklerin
  tek tek katkısı ablation olmadan bilinmez.

### Sonraki adım

`runs/d1-rev028-seed20260912` başlatıldı (`600k`, seed `20260912`, `v9` encoder).
Kabul ölçütü üç davranışsal sinyaldir:

1. **waypoint'e** mesafe medyanı `0.591 m`'nin altına inmeli.
2. Hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı.
3. Çarpışma oranı trendi azalan kalmalı.

Başarı oranı tek başına ölçüt değildir: aynı politika iki seed'de `%25` ve
`%42.5` verdi.

## 12 Eylül 2026 — REV-023/026/027 düzeltmeleri ve D1 yeniden eğitimi

### Amaç ve kapsam

D1 kanıt koşusunun ortaya çıkardığı üç bulgunun kullanıcı onayıyla uygulanması
ve aynı bütçeyle (`600k`) yeniden eğitim. Üç değişiklik iki ADR sözleşmesini
değiştirir, bu yüzden ADR'ler de güncellendi.

| Bulgu | Değişiklik | Değiştirdiği |
|---|---|---|
| REV-026 | `target_progress` `L_plan`'a normalize, `w` `5.0 -> 15.0` | ADR-017 |
| REV-027 | yeni `target_visible` bileşeni, `0.005/adım` | ADR-017 |
| REV-023 | `hover_height_m` `2.0 -> 1.5` | ADR-016 (ve ADR-012 geometrisi) |

**Başarı koşuluna, `hold_seconds`'a, abort sebeplerine ve durum makinesi geçiş
tablosuna dokunulmadı.** Kapı eşikleri de değişmedi. REV-023 yalnız o kapının
geçilebileceği konum kümesini büyütür.

### Oluşturulan/değiştirilen dosyalar

```text
src/ebabil/rewards/terminal_hover.py          progress normalizasyonu, target_visible,
                                              validate() üst sınırı, iki yardımcı
src/ebabil/envs/vision.py                     planned_route_length_m call-site'a bağlandı
src/ebabil/envs/vision_config.py              target_visible_weight yüklemesi
src/ebabil/configs/environment/
  vision_navigation.toml                      w = 15.0, target_visible_weight = 0.005
  target_area.toml                            hover_height_m = 1.5
tests/test_terminal_hover.py                  4 yeni reward testi, base call güncellendi
tests/test_hover_geometry.py                  halka 0.243 m, monotonluk testi
tests/test_vision_config.py                   hover anchor 1.51 m
docs/decisions/ADR-016-...md                  §3'e hover yüksekliği değişikliği
docs/decisions/ADR-017-...md                  §5'e reward değişikliği, kapı 14-18
docs/REVIEW_FINDINGS.md                       REV-023/026/027 çözüldü, REV-025 durumu
```

### Teknik kararlar

**1. Progress normalizasyonu `L_plan` ile, `w = 15.0`.**

```text
eski: 5.0 * (d_önce - d_sonra)              toplam = 5.0 * L_plan
yeni: 15.0 * (d_önce - d_sonra) / L_plan    toplam = 15.0, L_plan'dan bağımsız
```

Oran `collision_penalty / progress_toplamı` artık her seviyede `1.67`. Önce
D1'de `2.0x`, D3'te `30x`, D4'te `100x` olacaktı. `w = 15.0`, `success_bonus`
`30.0`'ın payını korumak için seçildi (bonus toplam pozitif sinyalin `%67`'si).

**2. `target_visible` ağırlığı `time_penalty` ile üst sınırlandı.**

Buradaki asıl karar sayı değil **sınırın nereden geldiği**. Ağırlık
`time_penalty` (`0.01`) değerine eşit veya üzerinde olursa, hedefi uzaktan
seyrederek beklemek adım başına net pozitif olur; yani REV-027'yi çözerken
timeout'u besleyen yeni bir yol açılır. `0.005` seçildi ve invariant
`TerminalHoverRewardConfig.validate()` içinde zorlandı, yani gelecekte bir
agent'ın bu değeri sessizce yükseltmesi mümkün değil.

Bileşen `target_alignment`'ın aksine **bütün** durumlarda uygulanır. `transit`
ve `search` durumlarında görünürlükle ilgili başka hiçbir sinyal yoktur ve
REV-027 en çok orada ısırır.

**3. `hover_height_m = 1.5`: bedeli kayda geçirildi.**

```text
irtifa       2.01 -> 1.51 m
yakın sınır  0.542 -> 0.407 m
hold halkası 0.108 -> 0.243 m   (2.25 kat)
uzak sınır   7.46 -> 5.60 m     (bedel)
```

Uzak sınırın kısalması, hover irtifasında karşılanabilen belirsizlik yarıçapını
`R <= 5.60 m` yapar. D1 (`R = 2 m`) ve D2 (`R = 5 m`) içeride; D3'ten itibaren
ADR-019 zaten `4.0 m`'ye tırmanmayı şart koşar. Merdiven etkilenmez.

**4. Encoder bilinçli olarak değiştirilmedi.** Onaylı zincir `vision-encoder-v10`
(`vision_audit_v3`) olmasına rağmen bu koşu `v9` (`vision_audit_v2`) ile
yapıldı; `%25` baseline'ı da `v9` ile alınmıştı. Aynı koşuda hem reward hem
encoder değiştirilirse sonucun hangisinden geldiği ölçülemez. `v10` geçişi ayrı
bir dilimdir ve ADR/REV-021 uyarınca D2'den **önce** zorunludur.

### Çalıştırılan komutlar/testler ve sonuçları

```text
ruff format   : temiz
ruff check    : All checks passed
mypy          : 49 source file temiz
pytest        : 254 passed, 64 subtests passed in 28.78s  (öncesi 248/54)
```

Legacy donmuş davranış:

```text
LEGACY_ROLLOUT_SHA256 = 03f16232adae103e370fe2a5cbcb2d0c62900838f90b51fc7db408e14e3f5e23  (değişmedi)
LEGACY_SCENE_SHA256   = 3a9da4aa...ff1231 (240 sahne, test suite içinde, değişmedi)
```

Vision env smoke (`single_blocker`, seed `20260912`, 1024 PPO adımı) iki yeni
bileşeni de üretti ve tasarlanan yönde çalıştılar:

```text
target_progress  +0.009081   (normalize; araç çok az ilerledi)
target_visible   +0.165000   (33 görünür adım x 0.005)
time             -0.400000   (40 adım x 0.01)
```

Yani `target_visible + time` net negatif: seyrederek beklemek hâlâ zararlı.

### Yeni eklenen testler

| Test | Kanıtladığı |
|---|---|
| `test_progress_total_is_route_independent` | `L_plan` 2/10/65/500 m için toplam `= w` |
| `test_collision_penalty_dominates_progress_...` | `collision > progress_toplamı` invariantı |
| `test_progress_normalization_rejects_non_positive_route_length` | `<= 0` ve `inf` reddi |
| `test_visible_target_is_rewarded_in_every_state` | Görünürlük her durumda ödüllü |
| `test_visibility_reward_never_makes_stalling_profitable` | `w_v < time_penalty`, `validate()` reddi |
| `test_lowering_hover_height_widens_the_hold_band` | Halkanın irtifayla monoton genişlemesi |

### Bilinen sınırlılıklar

- Üç değişiklik **aynı** koşuda uygulandı. Kapı geçilirse hangisinin ne kadar
  katkı verdiği bilinmez; ablation gerekir. Birlikte uygulandılar çünkü REV-026
  ve REV-027 aynı reward dengesini ilgilendiriyor ve tek tek uygulanınca
  birbirini bozdukları zaten ölçülmüştü (REV-025 örneği).
- `grace_seconds = 2.50 s` doğrulanmış bir değer **değildir**; REV-025 kaydında
  gerekçesiyle yazıldı. Kalibrasyonu bu koşunun sonucundan sonra yapılacak.
- Tek seed (`20260912`). ADR-009 üç seed ister.
- `route_independent_progress_total()` bir üst sınır değil, **tam** toplamdır ve
  yalnız araç monoton yaklaşırsa toplanır; ileri-geri hareket toplamı düşürür.
  (**Sonradan geçersiz:** bu fonksiyon REV-028 ile `progress_reward_bound()`
  olarak yeniden adlandırıldı ve artık gerçekten bir üst sınırdır.)
- `hover_hold` hâlâ normalize edilmemiştir ve episode uzunluğuyla ölçeklenir
  (REV-022). Bu dilimde dokunulmadı: `HOLD` yalnız hedefin üzerinde mümkündür,
  yani progress'in aksine uzun rotada bedava büyümez.

### Koşu sonucu: kapı geçilmedi, başarı geriledi

`runs/d1-rev026-seed20260912`, `600 064` timestep, `1 231` episode, `74.9 dk`.
Kabul ölçütü ikiydi; biri karşılandı, biri karşılanmadı.

**Karşılanan:** çarpışma oranı eğitim ortasında `%20 -> %7`'ye indi (kanıt
koşusunda `%15 -> %20` **artıyordu**). Politika artık uzun rotada engele çarpıp
kısa yoldan gitmeyi kârlı bulmuyor. Son iki pencerede `%16-18`'e geri çıktı,
çünkü terminal manevrayı denemeye başladı ve hedefin çevresindeki engellere
yaklaşıyor.

**Karşılanmayan:** deterministik başarı `%25`'in üzerine çıkmadı, **altına
indi.** 40 episode, seed `520001`:

| Checkpoint | Başarı | Çarpışma | Abort | Timeout |
|---|---:|---:|---:|---:|
| kanıt koşusu `final` | **`%42.5`** | `%15.0` | `%5.0` | `%35.0` |
| yeni `final` | `%22.5` | `%15.0` | `%12.5` | `%50.0` |
| yeni `best` | `%20.0` | `%12.5` | `%7.5` | `%60.0` |

### Ölçülen kök neden (REV-028)

Timeout'ların `%51`'i `approach`, `%22`'si `settle` durumunda bitiyor. Beş
hover koşulu adım adım sınandığında bağlayıcı olanın **yatay konum** olduğu
görüldü; diğer dördü iyileşmişti:

| İhlal edilen koşul | kanıt koşusu | yeni koşu |
|---|---:|---:|
| **güvenli bölge dışı (yatay)** | `%71.3` | **`%90.9`** |
| irtifa toleransı | `%65.0` | `%17.4` |
| hız | `%26.5` | `%17.6` |
| açısal hız | `%15.4` | `%8.9` |
| tilt | `%3.8` | `%1.9` |
| yatay mesafe medyanı | `0.879 m` | `1.065 m` |
| hold halkasında geçen adım | `%16` | `%6` |

Halka REV-023 ile `2.25` kat genişletildiği hâlde halkada geçen süre düştü,
yani gerileme geometriden değil kontrolden gelir.

İki kök neden ölçüldü:

**1. Araç waypoint'in üzerinde bile duramıyor.** İlk teşhisim "araç inanılan
konuma park ediyor" idi ve **yanlıştı**; dayanağı iki sayının çakışmasıydı
(`1.316 m` terminal mesafe, `1.338 m` ortalama hedef hatası) ve bu kanıt
değildi. Waypoint'e olan mesafe ayrı ölçüldü
(`believed_horizontal_distance_m`, 40 episode, deterministik):

| | kanıt koşusu | üçüncü koşu |
|---|---:|---:|
| **waypoint'e** mesafe medyanı | `0.591 m` | `0.991 m` |
| gerçek hedefe mesafe medyanı | `0.653 m` | `0.917 m` |
| waypoint'in `0.65 m` içinde kalan episode | `%53` | `%25` |

Araç waypoint'ten de `~1 m` uzakta. Sorun bilgi eksikliği değil **konum tutma
hassasiyeti**. Ayrıca saf waypoint takibinin tavanı ölçüldü:
`P(hata <= 0.65) = (0.65/2)^2 = %10.6`. Hata `> 0.65 m` olan episode'larda
başarı kanıt koşusunda `%38`, üçüncü koşuda `%25` — ikisi de tavanın üstünde,
yani ADR-019'un kamera iddiası çalışıyor, gerileme o eksende değil.

**2. Normalizasyon son metrenin değerini düşürdü.** REV-026 toplamı korur fakat
yoğunluğu rota uzunluğuyla ters orantılı yapar:

```text
eski  5.0/m sabit               -> son 1.0 m = 5.00
yeni  15.0 / 10.70 m = 1.40/m   -> son 1.0 m = 1.40   (D1)
yeni  15.0 / 100 m   = 0.15/m   -> son 1.0 m = 0.15   (D3)
```

Yani düzeltme çarpışma teşvikini doğru biçimde kapatırken kameranın yapmak
zorunda olduğu işin ödülünü `3.6` kat azalttı. **Bu, benim önerdiğim
normalizasyonun yan etkisidir.** Normalizasyon tek başına kalıcı bir tasarım
değildir; D3'te son metre `0.15` değerinde olurdu.

### İlk kez olan şey

Eğitim sırasında **tam `3.00 s` hold** ilk kez ulaşıldı (son 300 episode'da 3
episode). Kanıt koşusunda aynı pencerede maksimum `0.55 s`'ydi ve `3 s` hiç
görülmedi. Terminal beceri artık öğrenilebilir; eksik olan ona yeterince sık
ulaşmak.

### Sonraki adım

REV-028 önerisi: progress'i iki menzile ayır, toplamın rota bağımsızlığını
koru, son metrenin yoğunluğunu geri getir.

```text
d > approach_radius_m (2.5 m) : w_cruise * delta_d / L_plan   (rotaya normalize)
d <= approach_radius_m        : w_terminal * delta_d          (sabit, m başına)
```

Terminal parça `2.5 m` ile üst sınırlı olduğu için sabit ağırlık rota
bağımsızlığını bozmaz. `w_cruise = 10.0`, `w_terminal = 5.0` ile toplam
`22.5 < 25.0` ve REV-026 invariantı (`collision > progress toplamı`) korunur.

Kabul ölçütü başarı oranı **değil** (seed'e çok duyarlı; aynı politika iki
seed'de `%25` ve `%42.5` verdi), iki adım bazlı sayıdır:

1. **waypoint'e** mesafe medyanı `0.591 m`'nin altına inmeli — bilgi sınırından
   bağımsız, doğrudan kontrol hassasiyetini ölçer.
2. Hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı —
   kameranın yer bulmaya katkısı ancak orada görünür.

Kullanıcı kararı bekler.

## 12 Eylül 2026 — REV-025 düzeltmesinin ölçülmüş sonucu ve REV-027

### Amaç ve kapsam

REV-025 (`grace_seconds` `0.50 -> 2.50 s`) düzeltmesinin katkısını, kanıt
koşusuyla **aynı `600k` bütçede** yeniden koşarak izole etmek.
`runs/d1-grace-seed20260912`.

**Sonuç:** Düzeltme kendi ekseninde kusursuz çalıştı (`target_lost` abort'u
evaluation'da `%45 -> %0`) fakat **net sonucu kötüleştirdi** (deterministik
başarı `%25 -> %5`, timeout `%15 -> %70`). Bu, dördüncü bulguyu ortaya çıkardı:
REV-027.

### Ölçümler

Eğitim (tam `600k`, 1453 vs 1760 episode):

```text
metrik                       grace 0.5s   grace 2.5s
hover_abort_target_lost            %32          %13
obstacle_collision                 %15          %22
time_limit                         %48          %57
ortalama episode uzunluğu     340 adım     412 adım
reward (son çeyrek)              20.69        13.24
hold_max                          0.85s        0.70s
```

Evaluation (`final`, 20 episode, seed `410001`):

```text
koşu          policy          başarı  çarpışma  abort  timeout  return
grace 0.5s    deterministik      %25       %5     %45      %15    31.6
grace 0.5s    stokastik           %0      %10     %45      %35    25.6
grace 2.5s    deterministik       %5      %15      %0      %70    11.1
grace 2.5s    stokastik           %0      %15      %5      %80     8.9
```

### REV-027 — hedef görünürlüğü reward'da temsil edilmiyor

Teşhis: hedefi görüşte tutmanın reward'da **pozitif karşılığı yok.** Görünürlüğü
sağlayan tek mekanizma `target_lost` abort'unun sert cezasıydı ve kısa grace
kazara bir şekillendirme sinyali işlevi görüyordu — politikayı hedefi kadrajda,
dolayısıyla hedefin yakınında tutmaya zorluyordu. Grace uzatılınca bu baskı
kalktı.

`target_alignment` bu işi yapmıyor: ağırlık `0.20`, ADR-016 uyarınca yalnız
`acquired`/`approach` durumlarında etkin, `settle`/`hold`'da kapalı; ayrıca
kaybedilen hedefte `target_center_xy` `(0, 0)` sentinel'i verdiği için sinyal
taşımıyor (REV-017).

### Alınan karar

`grace_seconds` `2.50 s`'de **bırakıldı**, `0.50 s`'ye geri alınmadı. Gerekçe:
`0.50 s`'nin aksiyon limitleriyle çeliştiği ölçüm geçerliliğini koruyor
(`0.50 s`'de yalnız `43 derece` dönülebilir, gereken `90-180 derece`). Ölçülmüş
biçimde yanlış bir değeri, ikinci bir eksiklik onu kazara telafi ettiği için geri
koymak doğru olmaz. Doğru sıra REV-027'yi kapatmak, sonra grace'i kalibre
etmektir.

Bugünün en iyi D1 politikası kanıt koşusunun `final` checkpoint'idir
(`%25` deterministik): `runs/d1-seed20260912/final_model.zip`.

### Bilinen sınırlılıklar ve riskler

- D1 kapısı her iki koşuda da geçilmedi. Kapı eşiklerine dokunulmadı.
- Dört bulgu açık: REV-026 (bağlayıcı), REV-027, REV-023 ve REV-025'in yeniden
  kalibrasyonu. Dördü de ADR sözleşmesi değiştirir ve kullanıcı kararı bekler.
- REV-026 ve REV-027 aynı reward dengesini ilgilendirdiği için birlikte
  ele alınmalıdır.
- Her iki koşu da tek seed'dir; ADR-009 üç ister.

### Sonraki önerilen adım

REV-026 ve REV-027 kararları alındıktan sonra ADR-017'yi tek seferde güncellemek
ve D1'i aynı bütçeyle yeniden koşmak. Kabul kriteri: çarpışma oranı trendi
azalan olmalı ve deterministik başarı kanıt koşusunun `%25`'ini aşmalı.

## 12 Eylül 2026 — D1 kanıt koşusu ve üç blocker

### Amaç ve kapsam

İlk kez engelli, büyütülmüş (`20 x 20 x 3 m`) ortamda ve hedefin yalnız
yaklaşık konumu (`R = 2 m`) bilinerek kamera tabanlı politika eğitmek.
Ayrıntılı kayıt: `docs/PHASE_7_D1_EVIDENCE_RUN.md`.

**Sonuç:** ADR-018 D1 kapısı **geçilmedi** (deterministik başarı `%25`, gereken
`>=%85`). Koşunun asıl çıktısı üç ölçülmüş blocker'dır. Kapı eşiklerine
dokunulmadı.

### Eğitim ve evaluation

`600 064` timestep, `1 760` episode, `71.1 dk`, `140 ts/s`. Reward monoton
yükseldi (`-16.95 -> +22.10`); `hold_max` son `200k`'da `0.85 s`'de tavan yaptı
(gereken `3.0 s`).

20 episode evaluation (`single_blocker` + `random_both`, D1 sahnesi):

```text
checkpoint    policy          başarı  çarpışma  abort  timeout  return
final         deterministik      %25       %5    %45      %15    31.6
final         stokastik           %0      %10    %45      %35    25.6
best          deterministik      %10       %5    %30      %50    19.1
```

**Önemli ayrım:** eğitimin `success_rate 0` raporu stokastik rollout
istatistiğidir. Deterministik politika `%25` başarı üretir. D0-V'de bulunan
deterministik/stokastik ayrışması (REV-023) D1'de de geçerli ve burada gürültü
politikayı `11 cm` halkadan çıkarıyor.

### Açılan üç bulgu

**REV-023 (mevcut, artık blocker).** Hold halkası `0.542-0.650 m = 11 cm`.
Bu bulguyu daha önce "blocker değil, ölçülüp raporlanacak kırılganlık" diye
kaydetmiştim; **D1 için yanlıştı.** D0-V'de hedef waypoint'in tam gösterdiği
yerdeydi, bu yüzden halkaya doğrudan iniliyordu. `R = 2 m` hatayla artık önce
hedefi bulmak, sonra `11 cm` halkaya yerleşmek gerekiyor.

**REV-025 (yeni, düzeltildi).** `grace_seconds = 0.50 s` iç tutarsızdı. İnanılan
konumda yaw taraması yapılmazsa hedefin kadrajda kalma olasılığı `%19.2`
(200k örnek); `360 derece` tarama ile `%92.7`. `max_yaw_rate = 1.5 rad/s` ile
`0.50 s`'de yalnız `43 derece` dönülebilir, oysa rastgele yönde kaybedilen hedef
ortalama `90 derece` (`1.05 s`) ister. Evaluation'da abort oranı `%45`.

**REV-026 (yeni, açık).** `obstacle_collision` oranı eğitim boyunca **arttı**:
`%15 -> %12 -> %13 -> %13 -> %17 -> %20`. `target_progress` rota uzunluğuyla
doğrusal (`5.0 x L`), `collision_penalty` sabit (`25.0`). D1 ortalama rotada
oran `2.0x`, D3'te `30x`, D4'te `100x`. Engele çarpıp kısa yoldan gitmek net
kazançlı hâle geliyor. Bu bulgu yalnız engel ekseni açıldığı için görünür oldu;
D0-V'de `proximity` tam olarak `-0.0` ve `collision` tam olarak `0.00` idi.

### `search` durumu D1'de inert

Bütün evaluation'larda `mean_search_seconds = 0.00`. Hedef hover irtifasında
`7.46 m`'ye kadar görünür, `search_entry_radius_m` ise `4.0 m`; `R = 2 m` ile
araç arama yarıçapına girmeden tespit ediyor. Defekt değildir: ADR-019 arama
geometrisine göre `search` ancak `R > ~7.5 m`'de gerekir, yani D3'ten itibaren.

### Uygulanan düzeltme ve devam eden koşu

REV-025 uygulandı: `grace_seconds` `0.50 -> 2.50 s` (`180 derece` dönüş `2.09 s`
artı pay). ADR-017 güncellendi. **Başarı koşulunun tamamı aynı** ve doğrulandı:
güvenli iç bölge `0.65 m`, irtifa toleransı `0.20 m`, hız `0.25 m/s`, tilt
`10 derece`, açısal hız `0.35 rad/s`, kesintisiz hold `3.0 s`.

Etkiyi izole etmek için **aynı `600k` bütçeyle** yeniden koşu başlatıldı:
`runs/d1-grace-seed20260912`. Karşılaştırma referansı yukarıdaki kanıt koşusudur.

### Çalıştırılan doğrulamalar

```text
pytest      : 248 passed, 54 subtests
ruff / mypy : temiz
kırık link  : yok
```

### Bilinen sınırlılıklar ve riskler

- Tek seed; ADR-009 üç ister. Kapı geçilmediği için ek seed koşulmadı.
- REV-026 ve REV-023 **açık** ve ikisi de ADR sözleşmesi değiştirir:
  REV-026 ADR-017 reward semantiğini, REV-023 ADR-012 başarı geometrisini.
  Bu ikisi kullanıcı kararı olmadan değiştirilmedi.
- Grace düzeltmesi tek başına yetmeyebilir: politika güvenli bölgenin
  merkezinde otursa hedef kalıcı olarak nadir kör bölgesinde kalır ve `2.5 s`
  sonra yine abort alır. Salınımlı bir çözüm öğrenmesi gerekir.
- REV-021 v10 kalite deneyi CPU çekişmesi nedeniyle eğitim sonrasına bırakıldı.

### Sonraki önerilen adım

Grace koşusu bitince kanıt koşusuyla birebir karşılaştırmak; REV-025 katkısını
ve REV-023/REV-026'nın artık etkisini ayırmak. Ardından REV-021 v10 kalite
deneyi. REV-026 ve REV-023 kararları kullanıcıyı bekler.

## 12 Eylül 2026 — REV-021 için risk head kapasitesi (D2 ön koşulu, deney bekliyor)

### Amaç ve kapsam

REV-021'in kendi kabul ölçütü "çözüm hiperparametre aramasıyla değil
kapasite/mimari veya risk süpervizyonu düzeyinde olmalı" diyordu. Risk head'i
`latent -> 3` tek doğrusal katmandı. Bu dilim kapasiteyi **konfigüre edilebilir**
hâle getirir; kalite deneyi D1 eğitimi bitince çalıştırılacaktır (CPU çekişmesi
olmaması için).

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/perception/model.py` — `risk_head_hidden_size`, şema kapısı,
  şemadan türetilen checkpoint dosya adı
- `src/ebabil/configs/training/vision_encoder_v10.toml` (yeni deney config'i)
- `tests/test_vision_encoder.py` — `RiskHeadCapacityTests` (6 test)

### Alınan teknik kararlar ve gerekçeleri

**Geriye dönük uyum zorunlu tutuldu.** `risk_head_hidden_size = 0` tek doğrusal
katmandır ve onaylı `vision-encoder-v9` bu değerle birebir yüklenir. Doğrulandı:
`load_approved_vision_encoder` v9'u `Linear` head'le yüklüyor. Bu kritikti
çünkü D1 eğitimi şu anda o encoder'ı kullanıyor.

**Kapasite yalnız v10 şemasıyla açılabilir.** `risk_head_hidden_size > 0` iken
`schema_version` `vision-encoder-v10` olmak zorunda; aksi halde config
reddedilir. Böylece v9 mimarisi ile v10 mimarisi karışmaz.

**Shipped `vision_encoder.toml` dokunulmadı.** Deney ayrı bir config dosyasında
(`vision_encoder_v10.toml`, `--config` ile verilir) ve `pos_weight = 2.0`
kullanır. Karar eşiği `0.62` yine değişmedi.

**Checkpoint dosya adı şemadan türetiliyor.** Öncesinde sabit
`vision_encoder_v9.pt` idi; v10 checkpoint'i v9 adıyla kaydedilirdi.

v9 -> v10 parametre: `35 635 -> 39 795` (`+4 160`).

### Development bütçesi notu

Bu, `vision-dataset-v4c` development split'lerinin **dördüncü** değerlendirmesi
olacaktır (önceki üçü: ağırlıksız, `pos_weight=2.0`, `pos_weight=3.0`). İlk üçü
sabit mimaride hiperparametre aramasıydı; bu deney REV-021'in kabul ölçütündeki
mimari değişikliktir. Buna rağmen bütçe tükenmeye yakındır: bu deney de
düşerse yeni salt ile yeni development seti üretilmelidir, daha fazla arama
yapılmamalıdır.

### Çalıştırılan doğrulamalar

```text
pytest      : 248 passed, 54 subtests (öncesi 242/52)
ruff / mypy : temiz
onaylı v9 yüklemesi: OK, Linear head, şema vision-encoder-v9
v10 config        : OK, Sequential(64->64->ReLU->3), pos_weight 2.0
```

### Bilinen sınırlılıklar ve riskler

- Kalite deneyi **henüz çalıştırılmadı**; REV-021 açık kalır.
- v10 onaylanırsa `vision-v2` ve D2 zinciri yeni approval manifest'i
  gerektirir; D1 sonucu v9 ile dondurulmuş kalır.
- Gizli katman boyutu `64` ölçülmüş değil seçilmiş bir değerdir.

### Sonraki önerilen adım

D1 eğitimi bitince `vision_encoder_v10.toml` ile üç-seed quality akışını
çalıştırmak; geçerse audit ve yeni approval manifest, geçmezse REV-021'i açık
bırakıp yeni development seti gereksinimini kaydetmek.

## 12 Eylül 2026 — D1 hazırlığı: iki defekt, ADR-019 uygulaması ve config

### Amaç ve kapsam

D1 eğitimini başlatmak için eksik olan her şeyi tamamlamak: iki defektin
kapatılması, ADR-019 yaklaşık hedef ve `search` durumunun uygulanması, D1 sahne
ve eğitim config'leri.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/envs/vision.py` — terminal rezervi koşulsuz, inanılan hedef,
  `search` okuması, yeni info alanları
- `src/ebabil/missions/hover_state.py` — `SEARCH` durumu, `search_timeout`,
  arama config'i ve sayaçları
- `src/ebabil/envs/vision_config.py` — `TargetUncertaintyConfig`, arama
  parametreleri, hedef çapraz doğrulaması
- `src/ebabil/scenes/target.py` — `believed_target_xy`
- `src/ebabil/scenes/generator.py` — `maximum_endpoint_separation_m`
- `src/ebabil/training/{config,env_factory}.py` — `scene_config_path`
- `src/ebabil/configs/environment/{vision_navigation,static_scene_d1}.toml`
- `src/ebabil/configs/training/ppo_vision_d1.toml`
- `tests/test_terminal_hover.py` (+16), `tests/test_scenes.py` (+4)
- `docs/decisions/ADR-019-*.md` — uygulama notları

### Kapatılan iki defekt

**1. Terminal rezervi eklenmiyordu.** ADR-018 ara waypoint'i kaldırdı, yani her
episode nihaidir; fakat `final_segment` varsayılanı `False` olduğu için
`13 s` terminal rezervi bütçeye girmiyordu. `15 m` rotada `450` adım veriyordu,
ADR-018 tablosu `~710` diyor. Artık koşulsuz `final_segment=True`; reset
option'ı açık hata veriyor (sessizce yanlış bütçe üretmemesi için).

**2. Rota üst sınırı yoktu.** `maximum_endpoint_separation_m` eklendi; varsayılan
`inf` legacy davranışı korur. Öncesinde D1 config'i `5.29-21.55 m` rota
üretiyordu, yani D1 aslında D1 değildi.

### ADR-019 uygulaması

`believed_target_xy` inanılan hedefi scene seed'inden deterministik üretir
(diskte düzgün dağılım). `waypoint` artık yalnız bundan türetilir; `_observation`
içinde `goal_world_m` hiç kullanılmıyor. `search` durumu
`believed_horizontal_distance_m <= 4.0 m` ile girilir, `12 s` aşılınca
`hover_abort_search_timeout` üretir.

`R = 2.0 m` config'te ve güvenli iç bölgeden (`0.65 m`) büyük olması iki yerde
doğrulanıyor. `R <= 0.65 m` olursa görüntü kullanılmadan başarıya ulaşılabilir.

### Çalıştırılan doğrulamalar

```text
pytest       : 242 passed, 52 subtests (öncesi 222/52)
ruff / mypy  : temiz, 49 source file
LEGACY_SCENE_SHA256    değişmedi
LEGACY_ROLLOUT_SHA256  değişmedi
32 md dosyası: kırık link yok, görünmez karakter yok
```

Code review — sızıntı denetimi:

```text
_observation içinde goal_world_m : YOK (doğru)
_hover_reading içinde goal_world_m: YOK (privileged anchor üzerinden)
observation sürümü/anahtarları    : vision-v1, 5 anahtar, shape'ler aynı
```

Documentation review — ADR iddiaları koda karşı:

```text
zincirde search var, search_timeout abort sebebi, R=2.0, R>güvenli bölge,
search_entry >= approach, belirsizlik kanalı yok (D2 işi)     -> hepsi OK
D1 config vs ADR-018: 20x20x3 m, 5-15 m, 40 engel, 0.100/m2   -> hepsi OK
```

D1 sahne üretimi (30 seed, `random_both`, hedef-farkında):

```text
single_blocker   : rota 5.15-14.87 m (ort 9.92), 303 ms/sahne, 30/30 bandda
random_obstacles : rota 5.15-14.87 m (ort 9.92), 291 ms/sahne, 30/30 bandda
üretilemeyen sahne: 0/30
```

Throughput: `230.7 ts/s` ham (4 subproc); PPO yüküyle 600k ~= `63 dk`/seed.

### Bilinen sınırlılıklar ve riskler

- Privileged reward şekillendirmesi: `target_progress` ve hover koşulları gerçek
  hedefe göre. ADR-012 buna izin veriyor ve deploy'da reward yok, fakat
  raporlarken belirtilmeli.
- `search_entry_radius_m = 4.0 m`, D0-V'nin `4 m` sahnesinde episode başından
  `search` üretir. D1 rotaları `5-15 m` olduğu için dejenerasyon yok.
- REV-023 açık: `hover_height_m` `2.0`'da bırakıldı (kullanıcı onayı gelmedi).
  D1'de de deterministik/stokastik terslik beklenebilir; ölçülüp raporlanacak.
- D1 sahne üretimi `~300 ms`; episode `~25 s` olduğu için `%1.2` yük.
- Arama davranışı için reward bileşeni **yok**. Politika aramayı yalnız
  `target_progress` ve `hover_hold` üzerinden dolaylı öğrenmek zorunda. Bu
  kasıtlı (yeni bileşen ADR gerektirir) fakat D1 sonucunda ölçülmeli.

### Sonraki önerilen adım

D1 eğitimi: seed `20260912` ile 600k timestep, öğreniyorsa iki seed daha, sonra
held-out evaluation (deterministik + stokastik) ve ADR-018 D1 kapısı
(başarı `>=%85`, çarpışma `<=%5`).

## 12 Eylül 2026 — Rota doğrulamasının vektörleştirilmesi (D1 ön koşulu)

### Amaç ve kapsam

ADR-018'in D1 ön koşulu olan sahne üretim maliyetini çözmek. Plan "iki
çözünürlüklü rota doğrulaması" öngörüyordu; profil, maliyetin grid
çözünürlüğünden değil **hücre başına bütün engelleri tarayan Python
döngüsünden** geldiğini gösterdi. Bu nedenle yaklaşım yapmak yerine
vektörleştirildi ve davranış bit-bit korundu.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/scenes/generator.py` — `_segment_intersects_aabb_batch`,
  `_inflated_obstacle_bounds_xy`, `has_navigable_xy_path` içinde önceden
  hesaplanan occupancy grid'i
- `tests/test_scenes.py` — `PathValidationVectorizationTests` (7 test)

### Alınan teknik kararlar ve gerekçeleri

**Yaklaşım yerine optimizasyon seçildi.** Profil (`7.5 m` yarıçap, 20 engel,
3 çağrı):

```text
53 271 segment_is_free çağrısı -> 1 095 138 generator iterasyonu
1 043 880 _segment_intersects_aabb çağrısı, 0.825 s
generator overhead'i tek başına 1.646 s tottime
```

Yani maliyet `O(hücre x engel)` idi. İki çözünürlüklü heuristik false-negative
riski taşır (REV-003'te aynı sınıf hata düzeltilmişti); vektörleştirme aynı
sonucu verdiği için tercih edildi.

**Üç değişiklik:**

1. `_segment_intersects_aabb_batch` — aynı slab yöntemi, Python döngüsü yerine
   numpy. Tek segment, `N` kutu.
2. `_inflated_obstacle_bounds_xy` — genişletilmiş sınırlar çağrı başına bir kez
   hesaplanır; önce 1M kez yeniden hesaplanıyordu.
3. Occupancy grid'i bütün hücreler için tek seferde `einsum` ile üretilir;
   `is_occupied` artık dizi okuması. Hücre doluluğu dejenere segmentin noktasal
   kapsanma testidir, bu nedenle birebir eşdeğer.

### Çalıştırılan ölçümler ve sonuçları

```text
kenar     hücre   engel   başlangıç    şimdi   kazanç
   6 m    3 600       8       70 ms    24 ms      3x
  15 m   22 500       8      611 ms   225 ms      3x
  20 m   40 000      40    2 524 ms   164 ms     15x
  30 m   90 000      40    8 067 ms   646 ms     12x
  65 m  422 500     210           -  4 934 ms      -
 200 m  4 000 000   800           -  8 633 ms      -
```

Kazanç engel sayısıyla büyüyor; beklenen davranış.

Davranış korunması:

```text
LEGACY_SCENE_SHA256   = 75274526...c390dfe6  (720 sahne, değişmedi)
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23  (değişmedi)
pytest: 222 passed, 49 subtests (öncesi 215/46)
ruff / mypy: temiz
```

Tam test suite süresi `47.4 s` -> `24.7 s`'ye düştü; sahne üretimi bütün
testlerde hızlandı.

### ADR-018 ön koşulunun kısmi karşılanması

**D1 için yeterli.** `20 x 20 m`, 40 engel: `164 ms/sahne`. D1 episode'ları
`~710` adım ve 4 env ile env başına `~21 s` sürüyor; `164 ms` reset maliyeti
`%0.8` yüktür.

**D2 ve sonrası için yetmez.** `65 m`, 210 engel: `4 934 ms`. Kalan maliyet
artık Python BFS'inin hücre başına yüküdür (`4M` hücreye kadar). ADR-018'in
iki çözünürlüklü doğrulama maddesi **D2 için açık kalır**; D1 kapsamından
çıkarılır.

### Bilinen sınırlılıklar ve riskler

- İki çözünürlüklü doğrulama D2 ön koşulu olarak açık. Uygulanırken kaba
  geçişin **iyimser** (aday koridoru asla elemez) olması, ince doğrulamanın
  koridorda yapılması gerekir; aksi halde REV-003 sınıfı false-negative geri
  gelir.
- `einsum` ile üretilen occupancy grid'i `hücre_x * hücre_y` bellek tutar;
  `200 m` kenarda `4M` boolean (`4 MB`), `650 m` kenarda `42M` (`42 MB`). D4'te
  yerel pencere zaten zorunlu olduğu için bu sınır bağlayıcı değil.
- Perf guard testi (`test_d1_scale_generation_stays_within_budget`, `< 1.0 s`)
  makine hızına duyarlıdır; yavaş CI'da gevşetilmesi gerekebilir.

### Sonraki önerilen adım

D1 diliminin kalanı: (1) REV-023 hover geometrisi kararı, (2) workspace
`20 x 20 x 3 m` ve ölçekli engel yoğunluğu config'i, (3) ADR-019 yaklaşık hedef
(`R = 2 m` sabit) ve `search` durumu, (4) `single_blocker` ile ilk engelden
kaçınma eğitimi, (5) ADR-018 D1 kapısı üç seed.

## 12 Eylül 2026 — Yaklaşık hedef konumu ve görsel arama kararı

### Amaç ve kapsam

D0-V sonrası ortaya çıkan soruyu çözmek: ajan waypoint'i kusursuz biliyor,
yani GPS-denied navigasyon aslında test edilmiyor. Kullanıcı kapsamı
netleştirdi: araç kendi konumunu görsel pozisyonlama ile bilir (bu katman
modellenmeyecek), hedefin ise yalnız **yaklaşık** konumunu bilir. Kod
değişikliği yoktur; bu bir karar, ölçüm ve plan dilimidir.

### Oluşturulan/değiştirilen dosyalar

- `docs/decisions/ADR-019-approximate-target-and-visual-search.md` (yeni)
- `docs/REVIEW_FINDINGS.md` — REV-024
- `docs/IMPLEMENTATION_PLAN.md`, `README.md`

### Alınan teknik kararlar ve gerekçeleri

**Waypoint oracle'ı ölçüldü ve kaldırılacak.** `waypoint` kanalı
`scene.goal_world_m - reading.ideal.position_world_m` ile sıfır hatayla
hesaplanıyordu. Ablation (`best`, 6 episode, deterministik) iki kanalın da
taşıyıcı olduğunu gösterdi: `waypoint = 0` -> `0/6` (`out_of_bounds`),
`vision_latent = 0` -> `0/6` (`time_limit`). Ancak waypoint kusursuz olduğu
sürece kamera **yer bulma** işine katılmıyor.

**Yeni sözleşme (ADR-019):** kendi konum hatasız (VPS varsayımı, modellenmez),
hedef konumu episode başına `R` yarıçaplı hatayla bilinir. Hata episode içinde
sabittir ve scene seed'inden deterministik üretilir. `waypoint` kanalının
shape/semantiği değişmez, yalnız türetildiği hedef konumu değişir; bu nedenle
`vision-v1` sözleşmesi kırılmaz.

**`R`'nin bandı ölçüldü.** Alt sınır `0.65 m` (güvenli iç bölge): altında
görüntü gereksiz kalır, değişiklik anlamsız olur. Üst sınır `~40 m`: tek
tırmanma + `360 derece` yaw taramasıyla bulunabilecek en büyük hata.

**Arama davranışı zorunlu hâle geldi ve geometrisi ölçüldü.** Hover irtifasında
(`2.01 m`) tek poz `5 m` yarıçaplı diskin yalnız `%46`'sını görüyor; `360 derece`
tarama `R <= 7.5 m` kapsıyor. Kapsama irtifayla doğrusal (`uzak sınır = 3.71h`)
fakat hedef piksel olarak küçülüyor; optimum arama irtifası `10.9 m`
(kapsama `40.3 m`, uzak sınırda `10 px`). Davranış: **tırmanarak ara, alçalarak
yerleş.**

**Durum makinesine `search` eklenecek.** Mevcut `transit` pasif bekliyor; aktif
arama yok. Geçişler: `transit -> search -> acquired`, ve
`search -> abort (search_timeout)`.

**Belirsizlik kanalı D2'ye bırakıldı.** D1'de `R` sabit olduğu için politika
onu örtük öğrenir; observation değişmez. D2'den itibaren `R` değişken olacağı
için normalize `waypoint_uncertainty` skaleri `vision-v2` ile eklenir.

### Çalıştırılan ölçümler ve sonuçları

Waypoint oracle ablation'ı (yukarıda). Görsel tespit menzili, `2 x 2 m` hedef,
odak uzaklığı `208.5 px`:

```text
mesafe   hedef genişliği   durum
  40 m         10.4 px     güvenilir
 100 m          4.2 px     sınırda
 200 m          2.1 px     güvenilmez
 500 m          0.8 px     alt-piksel, imkansız
```

Arama irtifası takası:

```text
irtifa   kapsama R   uzak sınırda hedef
  4 m      14.8 m          27.1 px
  8 m      29.7 m          13.6 px
 10.9 m    40.3 m          10.0 px   <- optimum
 15 m      55.7 m           7.2 px
 20 m      74.2 m           5.4 px
```

`R` merdiveni: D1 `2 m` sabit, D2 `5 m`, D3 `15 m`, D4-D5 `30 m`; gereken arama
irtifaları sırayla `2.0 / 2.0 / 4.0 / 8.1 m`.

### Bilinen sınırlılıklar ve riskler

- **REV-024:** ilk drift ablation'ı yanıltıcıydı. Rastgele yürüyüş modeli `%20`
  oranda bile yalnız `~7 cm` mutlak hata üretti; güvenli bölge `65 cm`. Yani
  localization hatası **hiç test edilmedi**. `5/6` sonucu dayanıklılık kanıtı
  değildir.
- Kendi konum hatası ADR-019 kapsam kararıyla modellenmeyecek. Gerçek sistemde
  toplam bütçe `R + drift`'tir; D4-D5 kapılarında menzil bütçesi formülü
  (`tespit_yarıçapı / drift_oranı`) ile raporlanmalıdır.
- **Encoder kapsama blocker'ı:** mevcut dataset `+-1.25 m` yanal offset ve
  `0.80-2.10 m` irtifa kullanıyor. D3'te `4 m`, D4'te `8.1 m` arama irtifası
  gerekiyor; encoder bu irtifaları hiç görmedi. D1/D2 için sorun yok
  (`R <= 5 m`, arama `2.0 m`). D3'ten itibaren dataset yenilemesi zorunlu ve
  bu REV-021 ile aynı dilimde yapılabilir.
- `R` bandının üst sınırı hedef boyutuna bağlı; `2 x 2 m` hedef için `~42 m`.
  Daha uzun menzil daha büyük hedef veya daha uzun odak uzaklığı gerektirir.
- Arama geometrisi noktasal kamera ve düz zemin varsayar; eğim, occlusion ve
  rüzgâr hesapta değil.

### Sonraki önerilen adım

D1 dilimi: (1) REV-023 hover geometrisi kararı, (2) iki çözünürlüklü rota
doğrulaması, (3) workspace `20 x 20 x 3 m` + ölçekli engel yoğunluğu,
(4) ADR-019 yaklaşık hedef (`R = 2 m` sabit) ve `search` durumu,
(5) `single_blocker` ile ilk engelden kaçınma eğitimi, (6) ADR-018 D1 kapısı.

## 12 Eylül 2026 — Tek hedefli ölçek merdiveni kararı

### Amaç ve kapsam

Kullanıcı görev kapsamını sadeleştirdi (episode başına tek hedef, ara waypoint
yok) ve daha gerçekçi/büyük bir ortamda engelli eğitim istedi. ADR-010 mesafe
curriculum'u ve ADR-013 episode mimarisi buna göre revize edildi. Kod
değişikliği yoktur; bu bir karar ve plan dilimidir.

### Oluşturulan/değiştirilen dosyalar

- `docs/decisions/ADR-018-single-target-scaling-ladder.md` (yeni)
- `docs/IMPLEMENTATION_PLAN.md` — mesafe ekseni ve yürütme planı tablosu
- `README.md`

### Alınan teknik kararlar ve gerekçeleri

**Tek hedef, ara waypoint yok.** Mission segment manager, waypoint indeksi ve
ADR-013'ün "yerel segment / mission evaluation" ayrımı plandan çıkarıldı.
ADR-013'ten korunan tek şey rota uzunluğuna bağlı zaman bütçesi formülüdür;
artık segmente değil tüm episode'a uygulanır. `vision-v1` içindeki `waypoint`
anahtarı zaten tek hedefi taşıdığı için anlamı değişmedi.

**Uzun mesafe eğitimle değil kompozisyonla doğrulanacak.** Ara waypoint
olmadığı için mesafe doğrudan episode uzunluğuna yansıyor. Ölçüldü: `2 km`'de
`12 027` adım (tasarım hızında), PPO kredi ataması için pratik değil. Bu nedenle
D1-D4 eğitim, D5 yalnız evaluation rejimi. Uzun seyrin öğrenilebilir olmasının
nedeni `target_progress` bileşeninin her adım yoğun reward vermesidir; zorluk
uçların ikisinde (kaçınma ve terminal hover) toplanmıştır.

**Algı menzili seyir hızını sınırlıyor — ölçüldü.** Fren mesafesi
`v^2/(2*g*tan(tilt))` artı kare-arası yol. Hover koşulundaki `10 derece` seyir
limiti değil (ortamın failure eşiği `75 derece`); gerçekçi seyir tilti
`30 derece` alındı. `8 m` free-space ufkunda ham üst sınır `~9 m/s`, fakat
encoder'ın free-space hatası (`MAE 0.25` x `8 m` = `+-2 m`) düşülünce tasarım
hızı `6 m/s`. Bu fiziksel ve indirgenemez bir kısıt: araç algı ufkunun içinde
durabileceğinden hızlı uçamaz.

**Hız yükseltmesi `vision-v2` gerektirir.** `max_forward_speed_m_s` `1.5 -> 6.0`
değişimi `fc_state` gövde hızı normalizasyonunu değiştirir, yani observation
sözleşmesini kırar. D2'de yeni environment ID'siyle yapılacak; `vision-v1` ve
D0-V sonucu dondurulacak.

**Sahne üreticisi iki çözünürlüğe geçecek.** Ölçülen maliyet `has_navigable_xy_path`
BFS'inden gelir ve grid hücresiyle doğrusaldır.

**Ego-merkezli yerel pencere D3'ten zorunlu.** Ölçülen MuJoCo geom bütçesi
`~800` engel (`4` env, `500 Hz` için gereken `2 000 step/s` sınırında). Sabit
yoğunlukta `200 x 200 m` alan `10 800` engel isterdi. Yerel pencere tek hedefli
tasarımla çelişmez; waypoint geçişiyle değil render/çarpışma kapsamıyla
ilgilidir.

### Çalıştırılan ölçümler ve sonuçları

Sahne üretim maliyeti:

```text
workspace   grid hücre   engel   süre/sahne
      6 m        2 988       8        70 ms
     15 m       18 675       8       611 ms
     20 m       33 200      20       2.5 s
     30 m       74 700      40       8.1 s
```

MuJoCo engel bütçesi (`1000` fizik adımı):

```text
engel   ngeom   süre     step/s
    8      29    16ms     61 972
   50      71    27ms     37 038
  200     221    71ms     14 024
  800     821   268ms      3 738   <- 4 env sınırı 2 000 step/s
```

Fren mesafesi (`8 m` free-space ufku):

```text
tilt     3 m/s   5 m/s   8 m/s  10 m/s  12 m/s
 10°      2.9     7.6    19.2    29.7    42.6
 20°      1.5     3.9     9.6    14.8    21.2
 30°      1.0     2.6     6.3     9.7    13.7
 40°      0.8     1.9     4.6     6.9     9.7
```

Episode uzunluğu (ADR-013 formülü, `20 Hz`):

```text
mesafe    v=1.5 m/s    v=6 m/s
  15 m          710        413
  50 m        1 527        652
 150 m        3 860      1 235
 500 m       12 027      3 277
2 000 m      47 027     12 027
```

Mevcut workspace `6 x 5 x 2.5 m`; endpoint kutusu köşeden köşeye `5.79 m`,
yani D1'in üst yarısı sığmıyor.

### Bilinen sınırlılıklar ve riskler

- Kapı eşikleri (D1 `%85` ... D5 `%50`) başlangıç hipotezidir, ölçülmüş değil.
- `6 m/s` tasarım hızı noktasal kütle fren yaklaşımına dayanır; rüzgâr, rotor
  gecikmesi ve kontrolcü tepkisi hesapta yok. `6 m/s` seyirde kontrolcü
  kararlılığı ayrıca doğrulanmalıdır.
- Engel yoğunluğu aralığı (`0.02-0.10/m^2`) gerçek dış mekân ölçümüne değil
  MuJoCo bütçesine ve makul dış mekân varsayımına dayanır.
- REV-023 (hover bandı `0.108 m`) D1 öncesinde çözülmelidir; ortam büyüdükçe ve
  konum kestirim hatası arttıkça daha kritik olur.
- REV-021 (risk head) D2 değişken endpoint'i için açık kalır.
- D5 başarı eşiği bir kapasite beklentisidir; eğitilmeden ölçüldüğü için düşük
  çıkması tasarım hatası değil kompozisyon sınırı olarak yorumlanmalıdır.

### Sonraki önerilen adım

D1 ölçek dilimi: (1) REV-023 hover geometrisi kararı, (2) iki çözünürlüklü rota
doğrulaması, (3) workspace `20 x 20 x 3 m` ve ölçekli engel yoğunluğu,
(4) `single_blocker` ile ilk gerçek engelden kaçınma eğitimi, (5) ADR-018 D1
kapısı üç seed üzerinde.

## 11 Eylül 2026 — D0-V kamera tabanlı yerel politika eğitimi

### Amaç ve kapsam

Projede ilk kez kamera + telemetri girdisiyle bir navigasyon politikası
eğitmek. Bu dilime kadar vision tarafındaki en uzun koşu `64` PPO adımıydı.
D0-V, ADR-010'un en alt basamağıdır: legacy `4-6 m` geometrisinin `vision-v1`
ile yeniden öğrenilmesi. **Engel yoktur** ve sahne sabittir; engelden kaçınma
curriculum'un bağımsız ekseni olup henüz başlamamıştır.

Ayrıntılı çalışma kaydı: `docs/PHASE_7_D0V_VISION_POLICY.md`.

**Sonuç özeti:** Görev öğrenildi. `best` checkpoint deterministik evaluation'da
`20/20` `hover_success`, sıfır çarpışma, sıfır abort. Koşu iki yeni bulgu
açtı: ADR-017'deki hold reward üst sınırı iddiası yanlıştı (REV-022, düzeltildi)
ve uygulanabilir hold bandı `0.108 m` olduğu için başarı `3.6 cm` konum farkıyla
belirleniyor (REV-023, açık).

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/training/config.py` — `EnvironmentKind`, vision/legacy alan
  doğrulaması, `require_legacy_environment`
- `src/ebabil/training/env_factory.py` — vision ortam fabrikası,
  `VISION_MONITOR_INFO_KEYS`, `monitor_info_keys`
- `src/ebabil/training/runner.py` — Dict observation için `MultiInputPolicy`
- `src/ebabil/evaluation/{runner,baselines}.py` — legacy guard
- `src/ebabil/configs/training/ppo_vision_d0v.toml` (yeni)
- `scripts/train_ppo.py` — vision için smoke override düzeltmesi
- `scripts/evaluate_vision_policy.py` (yeni)
- `src/ebabil/rewards/terminal_hover.py` — REV-022 düzeltmesi
- `tests/test_training.py` (+9), `tests/test_terminal_hover.py` (+1)
- `docs/decisions/ADR-017-*.md` — REV-022 düzeltmesi
- `docs/{REVIEW_FINDINGS,PHASE_7_D0V_VISION_POLICY}.md`, plan, README
- Artefact: `runs/d0v-seed20260911/`

### Alınan teknik kararlar ve gerekçeleri

**Kapsam düzeltmesi uygulandı.** Önceki kayıt REV-021'in D0-V'yi bloke ettiğini
söylüyordu; ADR-010'a göre D0-V sabit legacy geometridir ve değişken endpoint
D2'dedir. Onaylı `vision-encoder-v9` sabit geometride eğitildiği için D0-V'nin
gerektirdiği encoder tam olarak odur. Dört dokümanda düzeltildi.

**Vision ortamında `sensor_mode` ve `max_episode_steps` config'de yasaklandı.**
Episode sınırı ADR-013 dinamik horizon'ından gelir; config'de bir değer kabul
etmek sessiz çelişki üretirdi. `--smoke` yolu da buna uyarlandı.

**Observation normalization kapalı tutuldu.** `vision-v1` zaten `[-1, 1]`
aralığındadır; `VecNormalize` ile tekrar normalize etmek latent ölçeğini bozar.
Reward normalization açık kaldı.

**Evaluation iki kipli yazıldı ve etiketlendi.** `clear` + `fixed` sahnesi
seed'den bağımsız aynı olduğu için `in_distribution` bir genelleme ölçümü
değildir; `encoder_ood` (değişken başlangıç) tanısal problardır. Script bu
ayrımı çıktı JSON'unda `generalization_claim` alanıyla taşır.

### Çalıştırılan komutlar/testler ve sonuçları

Throughput ölçümü:

```text
legacy 4 subproc                 625.8 ts/s
vision 1 dummy                    93.4 ts/s
vision 4 subproc                 216.5 ts/s
vision + PPO optimize (gerçek)     135 ts/s   -> 300k = 37.1 dk
```

Eğitim: `300434` timestep, `852` episode, `37.1 dk`. SB3 son raporu
`success_rate 0.91`, `ep_rew_mean 101`, `explained_variance 0.838`.

```text
timestep      episode  reward   başarı    hold_max  baskın termination
0-28k         106      -22.07   0/106     0.05      target_lost 44, out_of_bounds 44
29-71k        106      +26.77   0/106     0.40      time_limit 106
71-113k       106      +51.25   0/106     1.10      time_limit 105
113-155k      106      +60.96   0/106     0.40      time_limit 106
156-198k      106      +83.80   0/106     0.95      time_limit 106
198-239k      106     +108.45  13/106     3.00      time_limit 93, hover_success 13
239-273k      106     +118.85  67/106     3.00      hover_success 67, time_limit 39
273-300k      110     +101.87  99/110     3.00      hover_success 99, time_limit 11
```

Evaluation, 20 episode, seed `310001`:

```text
checkpoint  kip               policy          başarı  timeout  çarpışma  abort  return
best        in_distribution   deterministik    20/20        0         0      0   84.51
best        in_distribution   stokastik        11/20        9         0      0  117.15
final       in_distribution   deterministik     0/20       20         0      0   23.84
final       in_distribution   stokastik        20/20        0         0      0   94.83
best        encoder_ood       deterministik     6/20       13         0      0   19.90
```

Bütün evaluation'larda sıfır çarpışma ve sıfır abort; başarısızlıkların tamamı
timeout. `best` deterministik koşusu tam deterministik: `20/20` episode `193`
adım, yerleşme adımı `51`, ilk hold adımı `61`, `hold_seconds` `3.00`.

Reward bileşen dağılımı (`best`, başarı episode'u, `+84.51`):
`hover_hold +41.20` (%48.8), `success +30.00` (%35.5),
`target_progress +17.36` (%20.5), kalanlar `< %3`. `proximity`, `collision`,
`out_of_bounds`, `abort` tam olarak `0.00`.

Kalite kapısı:

```text
pytest      : 215 passed, 46 subtests passed
ruff check  : All checks passed
ruff format : temiz
mypy        : 49 source file temiz
```

### Koşu öncesi endişenin doğrulanmaması

Eğitim başlarken `target_lost` abort'unun toparlanmayı öğrenmeyi engelleyebileceği
not edilmişti. Gözlem tersini gösterdi: `45` `target_lost` ve `44`
`out_of_bounds` terminasyonunun **tamamı** ilk `28k` adımda gerçekleşti,
sonrasında pratikten kalktı. Politika hedefi görüşte tutmayı öğrendi. ADR-017
abort semantiği değiştirilmedi.

### Bilinen sınırlılıklar ve riskler

- **Engel yok.** Bu koşu engelden kaçınma kanıtı değildir; `minimum_range_m`
  bütün episode boyunca `5.0 m` ve proximity bileşeni tam olarak `-0.0`. İlk
  gerçek kaçınma kanıtı `single_blocker` eğitimi olacaktır.
- **Genelleme yok.** `clear` + `fixed` tek deterministik sahnedir.
- **Tek seed.** ADR-009 en az üç seed ister; `20260911` ile koşuldu.
- **REV-023 açık:** uygulanabilir hold bandı `0.108 m`; `best` `0.646 m`,
  `final` `0.682 m` yatay offsette yerleşiyor ve aradaki `3.6 cm` başarıyı
  belirliyor. Deterministik ve stokastik evaluation birbirinin yerine
  kullanılamaz; tek kipte raporlanan başarı oranı yanıltıcıdır.
- **REV-022 kısmen açık:** sınır iddiası düzeltildi fakat
  `hover_hold`/`success_bonus` oranı ablation'ı yapılmadı.
- `encoder_ood` `%30` onaylı encoder'ın sabit geometri kısıtını gösterir;
  rastgele endpoint için REV-021 açıktır.
- Sentetik encoder ve privileged başarı doğrulaması gerçek uçuş güvenliği
  değildir.

### Sonraki önerilen adım

İkisi bağımsız: (1) REV-023 kararı — `hover_height_m` düşürülüp D0-V yeniden
koşulur ve deterministik/stokastik açıklığın kapanıp kapanmadığı ölçülür;
(2) engel ekseni — `single_blocker` ile ilk gerçek engelden kaçınma eğitimi,
proximity ve collision bileşenleri ilk kez canlı olur. Ayrıca D0-V'nin ADR-009
uyarınca en az üç seed ile tekrarlanması gerekir.

## 11 Eylül 2026 — D0-V ön koşulu: dataset yenilemesi ve risk kanalı bulgusu

### Amaç ve kapsam

REV-018'i (encoder dataset'i tek rota geometrisinde) kapatmak; dataset'i rastgele rota geometrisiyle yeniden
üretip ADR-014 approval zincirini baştan çalıştırmak. Curriculum eğitimi kapsam
dışıdır.

**Sonuç özeti:** Dataset düzeyindeki iki bulgu kapandı (REV-018, REV-019) ve bir
gerçek kod defekti bulunup düzeltildi (REV-020). Ancak yeni dağılımda
**onaylanabilir encoder üretilemedi**; risk kanalı altı kapıyı birlikte geçemedi
(REV-021, açık). Onaylı zincir değişmedi ve `audit` split'i tüketilmedi.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/configs/training/vision_dataset.toml` — `endpoint_mode`,
  `split_salt`, `hold_pose_fraction`
- `src/ebabil/perception/dataset.py` — `hold_pose_fraction` config alanı ve
  `make_capture_spec` hold pozu ailesi
- `src/ebabil/perception/model.py` — `collision_risk_positive_weight` ve risk
  BCE'sinde `pos_weight`
- `src/ebabil/configs/training/vision_encoder.toml` —
  `collision_risk_positive_weight`
- `tests/test_vision_dataset.py` — `HoldPoseSamplingTests` (5 test)
- `tests/test_vision_encoder.py` — `CollisionRiskPositiveWeightTests` (4 test)
- `docs/REVIEW_FINDINGS.md` — REV-018 kapanışı, REV-019/REV-020/REV-021
- Artefact: `runs/vision_dataset_v4c_20260911`,
  `runs/vision_quality_v1{7,8,9}_20260911`

### Alınan teknik kararlar ve gerekçeleri

**REV-018 (tek rota geometrisi) kapatıldı.** `endpoint_mode` `"fixed"` yerine
`"random_both"`; `split_salt` yenilendi ki audit split'i yeni scene seed'leriyle
üretilsin (REV-007 disiplini). Schema `vision-dataset-v4` korundu; değişiklik
config düzeyinde ve `config_sha256` ile ayırt ediliyor (v4 -> v4b -> v4c).
Rota uzunluğu artık sabit `4.00 m` değil, `2.08-4.65 m`.

**REV-019 (hold pozu dağılımda yok) bulundu ve kapatıldı.** ADR-017 terminal
hover'ın abort mantığı `target_score`'un hold boyunca eşik üstünde kalmasına
dayanıyor, fakat rota poz ailesi hover zarfını yalnız `%0.6-1.1` temsil
ediyordu (`512` train örneğinde `3-6` kare). `hold_pose_fraction = 0.20`
eklendi; hover zarfı temsili `%23.4`'e çıktı.

Ölçülen sonuç terminal hover tasarımı için olumlu: hold pozlarında hedef
`%88-93` oranında görünür etiketli (rota pozları `%82-86`), fakat kapladığı alan
yarısı kadar (`~0.11` vs `~0.22`). Bu ADR-016'nın hover anchor'da
`visible_fraction = 0.50` ölçümüyle tutarlıdır ve `target_score` kapısının hold
sırasında çalışabileceğini gösterir.

**REV-020 (risk kaybında `pos_weight` yok) bulundu ve düzeltildi.**
`collision_risk_loss` çıplak BCE kullanıyordu; aynı fonksiyondaki iki mask
kaybı `pos_weight=3.0` uygularken risk head'inde hiçbir telafi yoktu. Defekt
`v4b` üzerinde gizli kalmıştı çünkü orada risk-pozitif oranı `%60.8` ile
çoğunluktu. `v4c`'de oran `%49.0`'a düşünce risk head'i prior'ı takip edip
"risk yok" demeye kaydı: recall `0.578`'e düşerken specificity `0.856`'ya
çıktı. Kaçırılan çarpışma riski yanlış alarmdan pahalı olduğu için pozitif
ağırlık zaten doğru tasarımdır.

**Gönderilen `collision_risk_positive_weight` değeri `1.0` seçildi.** Bu
ağırlıksız BCE ile matematiksel olarak aynıdır; böylece onaylı
`vision-encoder-v9` gönderilen config'den tekrar üretilebilir kalır. `2.0` ve
`3.0` değerleri ölçüldü ve REV-021'de raporlandı; kalibrasyon D0-V risk head
çalışmasına bırakıldı.

**REV-021: kapı avcılığı yapılmadı.** Üç quality koşusu da düştü. Kapı eşikleri
değiştirilmedi, karar eşiği `0.62` değiştirilmedi, `pos_weight` araması iki
denemede durduruldu. Ayrıntılı gerekçe REV-021 kaydındadır.

### Çalıştırılan komutlar/testler ve sonuçları

Dataset üretimi:

```bash
.venv/bin/python scripts/generate_vision_dataset.py runs/vision_dataset_v4c_20260911
# records=736, splits={train:512, validation:64, development_ood:64, audit:96}
# leakage_check=passed
```

Hold pozu ölçümü (üretilen dataset üzerinde, privileged etiketlerden):

```text
split             aile    n  görünür   oran   ort hedef alanı
train             hold  126      117  92.9%   0.110
train             route 386      330  85.5%   0.220
validation        hold   10        9  90.0%   0.140
validation        route  54       46  85.2%   0.207
development_ood   hold   19       17  89.5%   0.109
development_ood   route  45       37  82.2%   0.234
audit             hold   17       15  88.2%   0.125
audit             route  79       68  86.1%   0.175
hold ailesi toplam: 172/736 (%23.4)
```

Üç quality koşusu (hepsi üç seed):

```text
koşu   pos_weight   kapı     düşen
v17    yok          DÜŞTÜ    risk_recall, 3/3 seed, iki split (min 0.567)
v18    2.0          DÜŞTÜ    risk_recall, seed 20260912 dev-OOD (0.678)
v19    3.0          DÜŞTÜ    grayscale_risk_precision, seed 20260911 ve 20260913
```

Onaylı encoder'ın tekrar üretilebilirliği. Gönderilen config (`pos_weight=1.0`)
ile `v4b` üzerinde seed `20260911` yeniden eğitildi ve `v16` aday metrikleriyle
karşılaştırıldı:

```text
validation      : yedi metrik de fark +0.000000
development_ood : yedi metrik de fark +0.000000
```

Checkpoint dosya SHA'sı farklıdır (`55075eff...` vs onaylı `db5eca8f...`) çünkü
checkpoint payload'u eğitim config'ini gömüyor ve config'e yeni alan eklendi.
Ağırlıklar özdeştir. Onaylı artefact diskte değişmedi ve
`load_approved_vision_encoder` doğrulamayı geçiyor.

Audit split'i durumu:

```text
runs/vision_quality_v16_20260911  audit_consumed: VAR   (onaylı zincir)
runs/vision_quality_v17_20260911  audit_consumed: yok
runs/vision_quality_v18_20260911  audit_consumed: yok
runs/vision_quality_v19_20260911  audit_consumed: yok
üretilen approval manifest: yalnız runs/vision_audit_v2_20260911
```

Kalite kapısı:

```text
pytest        : 201 passed, 46 subtests passed
ruff check    : All checks passed
ruff format   : temiz
mypy          : 49 source file temiz
```

### Düzeltme sonrası bağımsız denetleme

| Denetim | Sonuç |
|---|---|
| Tam test suite | `205 passed, 46 subtests` |
| Legacy rollout parmak izi | `03f16232...4e3f5e23`, değişmedi |
| Legacy sahne parmak izi, 720 sahne | `75274526...c390dfe6`, değişmedi |
| `load_approved_vision_encoder` (v2 manifest) | doğrulamayı geçiyor, `latent=64` |
| Onaylı encoder ağırlık tekrar üretimi (`v4b`, seed `20260911`) | 14 metrikte fark `+0.000000` |
| `audit_consumed.json` (v17/v18/v19) | üçünde de yok; audit split'i dokunulmadı |
| Üretilen approval manifest sayısı | `1` (yalnız `vision_audit_v2_20260911`) |
| Gerçek renderer vision env smoke | `3.439 s`, hover durumları raporlandı |
| Ağsız wheel build + kurulu paketten config | `endpoint_mode=random_both`, `hold_pose_fraction=0.2`, `pos_weight=1.0`, eşik `0.62` |
| `29` Markdown yerel link + görünmez unicode | kırık link yok, görünmez karakter yok |
| ruff check / format / mypy | temiz, `49` source file |

Onaylı checkpoint dosyasının SHA'sı yeniden eğitimde farklı çıkar
(`55075eff...` vs `db5eca8f...`) çünkü checkpoint payload'u eğitim config'ini
gömüyor ve config'e yeni alan eklendi. Ağırlıklar özdeştir; diskteki onaylı
artefact değişmedi ve approval manifest saklanan dosyanın SHA'sını doğrular.

### Bilinen sınırlılıklar ve riskler

- **REV-021 açık.** `vision-dataset-v4c` dağılımında onaylanabilir encoder yok.
  Çözüm hiperparametre aramasıyla değil risk head kapasitesi/mimarisi veya risk
  süpervizyonu düzeyinde olmalı.
  **Düzeltme:** Bu kayıt ilk yazımında REV-021'in D0-V'yi bloke ettiğini
  söylüyordu; yanlıştı. ADR-010'da D0-V sabit legacy geometridir, değişken
  endpoint D2'de gelir. REV-021 D2 ve sonrasını bloke eder.
- Onaylı zincir hâlâ `vision-encoder-v9` / `v4b` / `v16` / `audit_v2`, yani
  **sabit rota geometrisi**. Bu D0-V için doğru encoder'dır (ADR-010: D0-V
  legacy geometri). Rastgele endpoint'e geçildiğinde encoder kalitesi
  doğrulanmamıştır; REV-018 dataset düzeyinde kapandı fakat onaylı modelin tek
  geometride eğitilmiş olması D2 öncesinde REV-021 ile giderilmek zorundadır.
- Onaylı checkpoint dosyası artık byte düzeyinde yeniden türetilemez; yalnız
  ağırlık düzeyinde tekrar üretilebilir. Approval manifest saklanan dosyanın
  SHA'sını doğruladığı için zincir geçerliliğini koruyor.
- `hold_pose_fraction = 0.20` ölçülmüş değil seçilmiş bir orandır; hold
  pozlarının `%7-12`'si görünmez etiketli ve bunlar güvenli iç bölgenin hedeften
  uzağa bakan kenarındaki pozlardır. Bu oran `grace_seconds` payının varlığını
  gerekçelendirir fakat ADR-017'deki `0.50 s` değerini kalibre etmez.
- `v17`/`v18`/`v19` koşuları diskte tutuldu; hiçbiri onaylı değildir ve
  eğitim/deploy için kullanılmamalıdır.
- Development split'i bu dilimde üç quality koşusu + bir eşik taraması + iki
  `pos_weight` denemesi tüketti. Sonraki risk head çalışması bu bütçeyi dikkate
  almalı; gerekiyorsa yeni salt ile yeni development seti üretilmeli.

### Sonraki önerilen adım

D0-V yerel vision politikasını eğitmek. REV-021 bunu bloke etmez: D0-V sabit
legacy geometridir ve onaylı `vision-encoder-v9` bu geometride eğitilmiştir.
REV-021, D2 (değişken endpoint) öncesinde kapatılmalıdır: risk head'ine kapasite
eklemek (şu an `latent -> 3` tek doğrusal katman), risk etiketini üç bölge ikili
eşiğinden daha zengin bir temsile taşımak veya grayscale ablation'ını eğitim
sırasında düzenleyici olarak kullanmak. Kapı eşikleri değiştirilecekse ayrı ADR
gerekir.

## 11 Eylül 2026 — Terminal hover durum makinesi ve reward

### Amaç ve kapsam

ADR-012 terminal görevini `EbabilVisionNavigation-v0` ortamına uygulamak:
`transit -> acquired -> approach -> settle -> hold -> success` durum makinesi,
`target_score` ile kapılanan görsel süreklilik, abort semantiği ve ayrı
loglanan reward bileşenleri. Curriculum eğitimi ve mission segment zinciri
kapsam dışıdır.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/missions/{__init__,hover_state}.py` (yeni paket)
- `src/ebabil/rewards/terminal_hover.py` (yeni), `src/ebabil/rewards/__init__.py`
- `src/ebabil/envs/navigation.py` — `TerminalDecision`, `StepReward` ve iki
  override noktası
- `src/ebabil/envs/vision.py` — durum makinesi, hover reading, terminal reward
- `src/ebabil/envs/vision_config.py`,
  `src/ebabil/configs/environment/vision_navigation.toml`
- `tests/test_terminal_hover.py` (yeni, 40 test + 28 subtest)
- `scripts/smoke_vision_navigation_env.py` — hover durumu ve bileşen raporlaması
- `docs/decisions/ADR-017-terminal-hover-state-machine-and-reward.md` (yeni)
- `docs/PHASE_7_TERMINAL_HOVER.md` (yeni), README, plan

### Alınan teknik kararlar ve gerekçeleri

**Durum makinesi nötr bir `missions` paketine kondu.** İlk yerleşim
`envs/hover_state.py` idi; `rewards/terminal_hover.py` bu modülü import ettiği
için `rewards -> envs/__init__ -> navigation -> rewards` import döngüsü oluştu.
Durum makinesi görev mantığıdır ve yalnız `scenes`'e bağımlıdır, bu yüzden
`ebabil.missions` altına taşındı.

**Zincir durumları koşullardan türetiliyor, geçiş tablosu iki kısıtla katı.**
Koşullar iç içe geçmeli olduğu için (hold ⊂ settle ⊂ approach) araç zincirde
serbest hareket eder. Katı olan iki şey: `SUCCESS` yalnız `HOLD`'dan girilir ve
yalnız kesintisiz hold sayacı tamamlandığında üretilir; terminal durumlar
soğurucudur. İç içe geçme `approach_radius_m > güvenli iç bölge köşegeni`
koşuluna bağlı ve bu config doğrulamasında uygulanıyor.

**Histerezis zorunlu kılındı.** `loss_score >= acquire_score` config
doğrulamasında reddediliyor; tek eşikle skor gürültüsü durum titremesi üretir.

**Terminal durum actor observation'ına eklenmedi.** Durum privileged
geometriden türetildiği için observation'a eklenmesi localization sızıntısı
olurdu. Bu sayede `vision-v1` sözleşmesi değişmedi ve onaylı encoder zinciri
yeniden üretilmek zorunda kalmadı. Politika fazı `vision_aux`, `fc_state` ve
`waypoint` üzerinden çıkarmak zorunda; maliyeti D0-V'de ölçülecek.

**`target_alignment` `settle`/`hold` durumlarında kapatıldı.** ADR-016 hedef
merkezinin hold pozunda kadraj dışında kaldığını ölçmüştü; bileşen açık kalırsa
sistematik kenar kayması ödüllenir.

**Legacy ortam iki override noktasıyla genişletildi.** `step()` içine
`_terminal_decision` ve `_compute_step_reward` hook'ları açıldı; varsayılanlar
ADR-008 legacy semantiğidir. `step()`'i vision ortamında tamamen kopyalamak
fizik sabitlerini iki yerde tekrarlayacağı için (AGENTS.md §4) reddedildi.

### Uygulama sırasında yakalanan ölçek hatası

İlk uygulama `_vision_aux[0]` ham sigmoid skorunu observation ölçeğiyle
karıştırıp `target_score = (raw + 1) / 2` hesapladı. Ham skor `[0, 1]`, ölçek
kaydırması ise yalnız `_observation` içindeki kopyaya uygulanıyor. Sonuç
`[0, 1]` aralığının `[0.5, 1]`'e sıkışmasıydı:

- `loss_score = 0.40` hiçbir zaman tetiklenemiyordu; hedef kaybı algılanamıyordu,
- tespit eşiği efektif olarak ham `0.20`'ye düşüyordu; tespit neredeyse
  otomatikti.

Hata gerçek renderer smoke'unda görüldü: random policy `100` adımda `91` adım
`acquired` durumundaydı. Düzeltmeden sonra aynı koşu
`transit 36 / acquired 62 / approach 2` verdi.

İlk yazılan 37 test hatayı yakalamıyordu: eğitilmemiş test encoder'ı her görüntü
için `~0.505` skor üretiyor ve bu değer iki eşik arasında kaldığı için görsel
kapılar hiç sınanmıyordu. `ScriptedVisionEncoder` eklendi (skoru test sürüyor,
shape sözleşmesi üst sınıftan korunuyor) ve üç yeni regresyon testi yazıldı.
Hatanın gerçekten yakalandığı, düzeltme geçici olarak geri alınarak kanıtlandı:

```text
hatalı kod : 3 failed, 37 passed
düzeltilmiş: 40 passed, 28 subtests passed
```

### Çalıştırılan komutlar/testler ve sonuçları

Legacy davranışın korunduğunun kanıtı. Refactor'dan **önce** alınan uçtan uca
rollout parmak izi (3 sahne modu x 2 sensör modu x 2 seed x 60 adım;
observation, reward, terminated/truncated, termination_reason, reward
bileşenleri ve hold sayaçları dahil):

```text
LEGACY_ROLLOUT_SHA256 = 03f16232adae103e370fe2a5cbcb2d0c62900838f90b51fc7db408e14e3f5e23
refactor öncesi: aynı    refactor sonrası: aynı    dilim sonu: aynı
LEGACY_SCENE_SHA256   = 75274526eae51f52475ea4bf7ae0d0f44f12d7c2f4cc3f6f84418a93c390dfe6 (değişmedi)
```

Kalite kapısı:

```text
pytest        : 196 passed, 46 subtests passed in 39.30s  (öncesi 156/18)
ruff check    : All checks passed
ruff format   : 90 files already formatted
mypy          : 49 source file temiz
```

Gerçek approval manifest, MuJoCo RGB renderer ve approved encoder ile smoke:

```bash
.venv/bin/python scripts/smoke_vision_navigation_env.py \
  runs/vision_audit_v2_20260911/approval_manifest.json \
  --mode random_obstacles --seed 20260911 \
  --random-steps 100 --ppo-timesteps 64 --evaluation-steps 80 \
  --output-dir runs/vision_hover_smoke_v1_20260911
```

`check_env` + `100` random step `1.870 s`, `64` PPO step + `80` deterministic
evaluation `2.332 s`, toplam `4.202 s`. Deterministic evaluation `80` adım
cap'ine geldi; bileşen toplamları ayrı loglandı
(`target_alignment -0.100`, `target_progress -0.037`, `time -0.800`,
`hover_hold 0.0`). `64`-step smoke model kalite iddiası taşımaz.

Kesintisiz hold'un başarıya gittiği ortam entegrasyon testiyle doğrulandı: araç
hover anchor'a yerleştirildiğinde `transit` `5` adım, `hold` `60` adım, sonra
`hover_success` ve `hover_hold_seconds = 3.0`.

### Düzeltme sonrası bağımsız denetleme

| Denetim | Sonuç |
|---|---|
| Legacy rollout parmak izi (dilim sonu) | `03f16232...4e3f5e23`, değişmedi |
| Legacy sahne parmak izi, 720 sahne | `75274526...c390dfe6`, değişmedi |
| `smoke_navigation_env.py --mode clear` | `termination_reason=goal` |
| `smoke_navigation_env.py --mode random_obstacles` | `obstacle_collision` |
| `ebabil.{scenes,perception,envs,missions,rewards}` `__all__` | `22/68/22/10/6`, eksik sembol yok |
| Vision ortamı aynı seed tekrarı (reward + durum + sayaçlar) | deterministik, `40/40` adım |
| `compileall` + `pip check` | temiz, bozuk bağımlılık yok |
| Ağsız `--no-build-isolation` wheel build | `missions/`, `rewards/terminal_hover.py` ve yeni TOML bölümleri paket içinde |
| Kurulu wheel içinden tam `hold -> success` | `65` adımda `success`, `hold_s=3.0`, `11` reward bileşeni |
| `29` Markdown dosyası yerel link + görünmez unicode | kırık link yok, görünmez karakter yok |

Wheel denetiminin ilk denemesi `RuntimeError: Terminal durumdan (success) sonra
update çağrılamaz` verdi; bu denetim script'imin döngü hatasıydı, durum
makinesinin doğru davranışıdır ve ayrıca testle sabitlenmiştir.

### Bilinen sınırlılıklar ve riskler

- Kesintisiz hold başarısı yalnız araç hover anchor'a **elle yerleştirilerek**
  gösterildi. Öğrenilmiş politikayla başarı D0-V curriculum kapısıdır; bu dilim
  öğrenme kalitesi iddiası taşımaz.
- Terminal durum actor observation'ında yok; politika fazı dolaylı çıkarmak
  zorunda. Öğrenmeyi zorlaştırabilir, D0-V'de ölçülecek.
- Reward ağırlıkları ölçülmüş değil başlangıç hipotezidir. `hover_hold` üst
  sınırı `24.0` ile `success_bonus` `30.0` oranı ablation gerektirir.
- `grace_seconds = 0.50 s` gerçek VIO drift ölçümüne değil ADR-012 ilkesine
  dayanır.
- Hover koşulları policy adımı çözünürlüğünde (`50 ms`) ölçülür; bu pencere
  içindeki geçici ihlaller görünmez.
- `vision_navigation.toml` içindeki `[termination]` ve `[reward]` bölümleri
  terminal hover görevinde kullanılmıyor; `NavigationEnvConfig` kurulumu için
  duruyorlar ve TOML içinde bu durum yorumla işaretlendi.
- REV-018 açık; encoder dataset'i hâlâ tek rota geometrisinde.
- Privileged başarı doğrulaması ve sentetik encoder gerçek uçuş güvenliği
  değildir.

### Sonraki önerilen adım

D0-V yerel vision curriculum iş paketi: aynı `4-6 m` görev geometrisini kamera
girdisiyle yeniden eğitmek, hover başarı oranı ile abort/timeout dağılımını
ölçmek. REV-018 dataset yenilemesi bu paketin ön koşuludur.

## 11 Eylül 2026 — Terminal hover öncesi inceleme ve geometri düzeltmeleri

### Amaç ve kapsam

ADR-012 terminal hover iş dilimine geçmeden önce hedef alan geometrisi, sahne
üretimi ve perception sözleşmesini denetlemek; bulunan gerçek defektleri
kapatmak. Terminal state machine ve reward bileşenleri bu dilimin kapsamı
dışındadır; bu dilim onların üzerinde çalışacağı geometriyi güvence altına alır.

### Oluşturulan/değiştirilen dosyalar

- `docs/decisions/ADR-016-hover-geometry-and-target-clearance.md` (yeni)
- `docs/REVIEW_FINDINGS.md` — REV-014–REV-018 ve durum özeti
- `tests/test_hover_geometry.py` (yeni, 24 test + 6 subtest)
- `src/ebabil/perception/config.py` — nadir/zemin izi türetilmiş geometri
- `src/ebabil/scenes/target.py` — `HoverEnvelope`, `target_area_from_goal`,
  `target_area_from_scene`, `validate_hover_envelope`,
  `reachable_safe_region_xy`
- `src/ebabil/scenes/generator.py` — opt-in hedef-farkında açıklık, blocker
  kaydırma, hedef-farkında hata mesajı
- `src/ebabil/scenes/__init__.py` — yeni public API
- `src/ebabil/envs/navigation.py` — `_scene_target_config` hook'u (legacy `None`)
- `src/ebabil/envs/vision.py` — `target_config`, `target_area`, hover
  doğrulamaları
- `src/ebabil/perception/dataset.py` — `derive_supervision` sözleşme docstring'i
- `scripts/sweep_camera_visibility.py` — hold pozu kapısı ve geometri raporu
- `scripts/capture_camera_frame.py`, `scripts/view_target_camera.py` — tek
  kaynaklı factory

### Alınan teknik kararlar ve gerekçeleri

**REV-014 (kamera nadir kör bölgesi).** Sweep script'i `center_visible`
değerini hesaplayıp kapıya dahil etmiyordu; "40/45/50 derece projection kapısını
geçti" kaydı hold pozunu hiç denetlememişti. Ölçüm, nadir noktasının kadrajda
olması için `pitch_down >= 90 - vfov/2 = 60.08 derece` gerektiğini, mevcut
`45 derece` montajda hedef merkezinin hover anchor üzerinde kadraj dışı
kaldığını gösterdi. Zemin izi yakın sınırı `irtifa * 0.27` olarak ölçeklendiği
için hover yüksekliğini artırmak çözüm değil; pitch'i eşiğin üstüne çıkarmak ise
`4 m` yaklaşmayı bozuyor (`65 derece`de `visible_fraction 0.00`). Karar: hold
koşulu `target_score` ile kapılanır, merkez görünürlüğüne dayandırılmaz; montaj
açısı ve hover yüksekliği bench ölçümüne kadar değişmez. Geometri artık
config'den türetiliyor ve testle sabit.

**REV-015 (örtük türetme ve doğrulanmayan hover bandı).** Hedef merkezi dört
yerde ayrı türetiliyordu. Daha önemlisi, konum referansı
`workspace +- vehicle_clearance_m` içine kırpıldığı için hover toleransının üst
kenarı (`2.21 m`) ile erişilebilir üst sınır (`2.25 m`) arasında yalnız `0.04 m`
pay vardı ve bu hiçbir yerde doğrulanmıyordu; `hover_height_m = 2.1` gibi makul
bir değer hold koşulunu sessizce imkânsız hâle getirirdi. Tek kaynaklı factory
ve iki fail-fast doğrulama eklendi.

**REV-016 (sahne üreticisi hedefi korumuyor).** Engeller yalnız hedef
*noktasına* karşı denetleniyordu; `2 x 2 m` ayak izi ve üzerindeki hover hacmi
korunmuyordu. Kısıt opt-in eklendi: `target_config=None` legacy davranıştır.
Uygulama sırasında iki yan etki çıktı ve kapatıldı — blocker'ın hover zarfına
tam teğet yerleşmesi kesişim kararını floating-point yuvarlamaya bırakıyordu
(epsilon payı eklendi), ve blocker'ı başlangıca doğru kaydırmak başlangıcın
engel içinde kalmasını mümkün kılıyordu (açıklık kontrolü eklendi).

**REV-017 (sentinel belirsizliği).** `target_center_xy` görünmeyen hedefte
`(0.0, 0.0)` yazıyor; bu "tam merkezde" durumundan ayırt edilemez. Ayrıca
görünürken değer hedefin *görünen* piksellerinin ağırlık merkezidir. Dataset
schema'sı ve approval zinciri değiştirilmedi; düzeltme sözleşme ve test
düzeyinde kaldı.

**REV-018 (dataset tek rota geometrisi).** `endpoint_mode = "fixed"` nedeniyle
92 sahnenin tamamı aynı start/goal geometrisini kullanıyor; `vision-encoder-v9`
kalite sayıları bu tek geometri üzerinde ölçülmüş. Bilinçli olarak açık
bırakıldı: çözümü onaylı artefact zincirini baştan üretmeyi gerektirir ve D0-V
curriculum iş paketinin kapsamındadır. Hover dilimi frozen encoder ile
ilerleyebilir.

### Çalıştırılan komutlar/testler ve sonuçları

Legacy sahne dizisinin korunduğunun kanıtı (`3 mod x 4 endpoint modu x 60 seed
= 720 sahne`), düzeltmelerden önce ve sonra:

```text
LEGACY_SCENE_SHA256 = 75274526eae51f52475ea4bf7ae0d0f44f12d7c2f4cc3f6f84418a93c390dfe6
öncesi: UNCHANGED   sonrası: UNCHANGED
```

Hedef-farkında üretim, `2 mod x 4 endpoint modu x 300 seed = 2400 sahne`:

```text
hover hacmi ihlali          : 0/2400   (legacy: fixed modda 131/300, random_both 193/300)
başlangıcı yutan blocker    : 0/2400
doğrudan rotası açık kalan  : 0/2400
üretilemeyen sahne          : 4/2400   (%0.17, nedeni adlandıran RuntimeError)
```

Kamera geometrisi:

```text
$ .venv/bin/python scripts/sweep_camera_visibility.py --pitch-deg 45 60 65 \
    --approach-distance-m 4 0
45.0,4.00,1.00,True,False,0.542,7.460
45.0,0.00,0.50,False,False,0.542,7.460
65.0,4.00,0.00,False,True,0.000,2.862
65.0,0.00,0.50,True,True,0.000,2.862
minimum_pitch_for_nadir_view_deg=60.08
passing_pitch_deg_with_hold_center=[]
```

Kalite kapısı:

```text
pytest        : 156 passed, 18 subtests passed in 37.22s  (öncesi 132/12, 18s)
ruff check    : All checks passed
ruff format   : 86 files already formatted
mypy          : 46 source file temiz
```

Gerçek approval manifest, MuJoCo RGB renderer ve approved encoder ile smoke:

```bash
.venv/bin/python scripts/smoke_vision_navigation_env.py \
  runs/vision_audit_v2_20260911/approval_manifest.json \
  --mode random_obstacles --seed 20260911 \
  --random-steps 100 --ppo-timesteps 64 --evaluation-steps 80 \
  --output-dir runs/vision_env_smoke_v4_20260911
```

`check_env` + `100` random step (`2.140 s`), `64` PPO step + `80` deterministic
evaluation (`2.401 s`), toplam `4.541 s`. Önceki v3 koşusu `4.664 s` idi.
Deterministic evaluation yine 80-step cap'ine geldi; 64-step smoke model kalite
iddiası taşımaz.

### Düzeltme sonrası bağımsız denetleme

Düzeltmelerin başka bir şeyi bozmadığını göstermek için ayrı bir denetleme
turu çalıştırıldı.

**Legacy yolun uçtan uca korunduğu:**

| Denetim | Sonuç |
|---|---|
| Sahne parmak izi, 720 sahne | `75274526...c390dfe6`, değişmedi |
| MJCF hedef gövde konumu: eski örtük ifade vs yeni factory, 200 sahne | maksimum sapma `0.0` |
| Derlenmiş legacy model parmak izi (10 sahne, body/geom poz-boyut-tip-grup-rgba) | `5df392dd...c458ff85` |
| `smoke_navigation_env.py --mode clear` | `118` adım, `termination_reason=goal` |
| `smoke_navigation_env.py --mode random_obstacles` | `28` adım, `obstacle_collision` |
| `train_ppo.py --smoke` | checkpoint/normalization artefact'ları üretildi |
| `run_phase6_experiment.py --smoke` | tamamlandı; `quality_gate_passed=False` |

Faz 6 smoke'unun kalite kapısını geçmemesi REV-008 fail-closed tasarımıdır ve
`tests/test_phase6_experiment.py:59` bunu açıkça bekler; regresyon değildir.

**Paketleme, API ve provenance:**

| Denetim | Sonuç |
|---|---|
| `ebabil.scenes/perception/envs` `__all__` tutarlılığı | `22/68/22`, eksik sembol yok |
| Aynı seed ile hedef-farkında sahne tekrarı | deterministik |
| `compileall` + `pip check` | temiz, bozuk bağımlılık yok |
| Ağsız `--no-build-isolation` wheel build | `ebabil_mujoco-0.1.0-py3-none-any.whl` |
| Kurulu wheel içinden yeni API | `nadir eşiği 60.08 derece`, zemin izi `(0.542, 7.460)`, hedef-farkında sahnede ihlal yok |
| `load_approved_vision_encoder` | yüklendi, `latent=64`, `eval` modunda |
| `vision_dataset_provenance("runs/vision_dataset_v4b_20260911")` | `config_sha256` değişmedi |
| `generate_vision_dataset.py` smoke (4 sahne, 8 kayıt) | `leakage_check=passed` |
| `27` Markdown dosyası yerel link + görünmez unicode | kırık link yok, görünmez karakter yok |

Encoder'ın `requires_grad` bayrağı loader çıktısında `True` kalır; dondurma
ADR-015 uyarınca ortamda yapılır ve
`test_config_and_encoder_are_versioned_and_frozen` bunu doğrular.

**Uygulama sırasında yakalanıp kapatılan iki kendi kendine yol açılan defekt**
REV-016 kaydında ayrıntılıdır: hover zarfına teğet blocker yerleşiminin
floating-point'e bağlı kalması ve kaydırılan blocker'ın başlangıcı yutabilmesi.
İkisi de ölçümle bulundu (`38/300` ve potansiyel), düzeltildi ve `0/2400`
doğrulandı.

### Bilinen sınırlılıklar ve riskler

- `runs/vision_env_smoke_v3_20260911` artefact'ları artık farklı sahneleri
  anlatıyor: vision ortamı hedef-farkında üretim kullandığı için aynı seed
  farklı engel yerleşimi verir. Referans olarak v4 koşusu kullanılmalıdır.
- `vision-dataset-v4b` hash zinciri ve approval manifest etkilenmedi; dataset
  `generate_static_scene` çağrısını `target_config` olmadan yapar. Ortam
  sahneleri artık dataset sahnelerinin hover hacmi boşaltılmış alt kümesidir.
- Hedef merkezi hold pozunda hâlâ kadraj dışıdır. Bu kapatılmadı, sözleşmeyle
  kabul edildi; gimbal, ikinci kamera veya `hover_height_m = 1.5` seçenekleri
  ADR-016'da ölçülmüş olarak kayıtlı ve donanım/görev kararı bekliyor.
- Merkez-görünür hold bandı güvenli iç bölgenin arka kenarında yalnız ~`0.11 m`
  genişindedir; hold pozu tasarımı bu payı hesaba katmalıdır.
- Hover irtifa bandında erişilebilir workspace'e `0.04 m` pay kalmıştır. Hover
  yüksekliği veya workspace değişirse `validate_hover_envelope` fail-fast verir.
- `2400` sahnede `4` üretilemeyen vaka kısa endpoint ayrımlarındaki gerçek
  geometrik çözümsüzlüktür; curriculum random endpoint kullanacaksa
  `minimum_endpoint_separation_m` gözden geçirilmelidir.
- REV-018 açık. Rastgele endpoint'e geçildiğinde encoder kalitesi
  doğrulanmamıştır.
- Hedef-farkında test sweep'i tam regresyona `~19 s` ekledi.

### Sonraki önerilen adım

ADR-012 + ADR-016 uyarınca terminal hover state machine'ini test-first
uygulamak: `transit -> acquired -> approach -> settle -> hold -> success`
durumları, `target_score` ile kapılanan görsel süreklilik ve grace period,
privileged başarı doğrulaması (güvenli iç bölge + irtifa + hız + tilt + açısal
hız + kesintisiz hold) ve yedi ayrı reward bileşeni. Ardından D0-V curriculum
iş paketinde REV-018 dataset yenilemesi.

## 11 Eylül 2026 — Vision navigation environment kapanışı

### Amaç ve kapsam

ADR-015 sözleşmesini production Gymnasium ortamına geçirmek; approved encoder,
deploy edilebilir Dict observation, kamera-policy scheduler ve dinamik segment
horizon'ını test edip gerçek renderer üzerinde kısa PPO zincirini doğrulamak.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/envs/{vision,vision_config}.py`
- `src/ebabil/configs/environment/vision_navigation.toml`
- `src/ebabil/envs/{navigation,__init__}.py`
- `tests/test_vision_navigation_env.py`
- `scripts/smoke_vision_navigation_env.py`
- `docs/PHASE_6_5_VISION_NAVIGATION_ENV.md`
- ADR-015, README, proje tasarımı ve implementasyon planı

### Teknik kararlar ve uygulama sonucu

- `EbabilVisionNavigation-v0`, yalnız approval manifest hash zincirinden yüklenen
  frozen `vision-encoder-v9` kullanır; doğrudan checkpoint kabul etmez.
- Actor observation beş açık anahtarlı `vision-v1` Dict'tir. Depth, raycast,
  obstacle/hedef dünya koordinatı ve tam MuJoCo state actor yolunda yoktur.
- Action shape `(3,)`; geri/yanal translasyon kapalıdır. Kamera `30 Hz`, policy
  `20 Hz`, fizik/kontrol `500 Hz` deadline tabanlı sample-and-hold kullanır.
- ADR-013 local horizon uygulandı: `4 m -> 20 s/400`, `30 m -> 40 s/800`,
  `30 m final -> 53 s/1060`.
- Legacy ortam yalnız model/reset/tick/close extension hook'larıyla genişletildi;
  legacy action/observation davranışı korunmuştur.

### Çalıştırılan doğrulamalar

- Hedefli legacy + camera + vision testleri:
  `31 passed, 1 skipped, 7 subtests passed`.
- Tam regresyon: `131 passed, 1 skipped, 12 subtests passed`.
- Ruff lint temiz; format `85 files already formatted`; mypy `46` source file
  temiz; compileall ve `pip check` geçti.
- Ağsız `--no-build-isolation` wheel build/install smoke geçti; kurulu wheel
  içinden vision environment ID, 64 latent ve 60 s üst bütçe config'i yüklendi.
- Gerçek approval + RGB renderer smoke:
  `100` random step, `64` PPO step, `80` deterministic evaluation; toplam
  `4.664 s`. Artefact'lar `runs/vision_env_smoke_v3_20260911` içindedir.
- Deterministic smoke evaluation 80-step dış test cap'inde hedefe ulaşmadı;
  64-step model kalite iddiası taşımadığından bu beklenen sonuçtur.

### Bilinen sınırlılıklar ve riskler

- İlk `mjpython` headless smoke denemesi macOS'ta CPU kullanmadan kapanmayan
  süreç üretti. Headless smoke `.venv/bin/python`; viewer yalnız
  `scripts/run_mjpython.sh` ile çalıştırılmalıdır. Doğru production koşu hızlı
  tamamlandı.
- FC/waypoint verisi ideal sim adapter'ından gelir; Pixhawk/VIO hatası değildir.
- Terminal hover state machine, gerçek planner route length, frame jitter/drop,
  temporal history ve gerçek IMX477/Orin profiling henüz yoktur.
- Privileged proximity reward ve sentetik encoder gerçek uçuş güvenliği değildir.

### Sonraki önerilen adım

ADR-012 terminal hover state machine ve reward bileşenlerini test-first
uygulamak; sonra D0-V–D2 curriculum iş paketine geçmek.

## 11 Eylül 2026 — Vision-v1 sözleşme iş dilimi

### Amaç ve kapsam

Onaylı encoder'ı PPO ortamına bağlamadan önce action, actor observation,
privileged sınır, kamera-policy scheduler ve rota-temelli horizon semantiğini
değişmez bir teknik sözleşme olarak sabitlemek.

### Alınan kararlar ve değişen dosyalar

- `docs/decisions/ADR-015-vision-navigation-contract-v1.md` oluşturuldu.
- Ortam `EbabilVisionNavigation-v0`, observation `vision-v1` ve action
  `[forward_speed, vertical_speed, yaw_rate]` olarak sürümlendi.
- Observation; dondurulmuş encoder latent/yardımcı çıktıları, deploy edilebilir
  FC state, ego-frame waypoint ve zaman bilgisinden oluşan açık `Dict`
  sözleşmesine ayrıldı.
- Approval manifest zorunlu, encoder frozen, kamera `30 Hz`, policy `20 Hz`,
  fizik `500 Hz` ve deadline tabanlı sample-and-hold olarak kararlaştırıldı.
- ADR-013 dinamik horizon formülü ve final segmentte `10 + 3 s` terminal rezervi
  kesinleştirildi.
- `docs/IMPLEMENTATION_PLAN.md` içindeki tamamlanmış perception review durumu ve
  aktif iş paketi düzeltildi.

### Doğrulama ve riskler

- ADR-008, ADR-010–ADR-014, proje tasarımı ve güncel kod arayüzleri karşılıklı
  kontrol edildi; bu dilim yalnız belge değişikliğidir.
- Gerçek Pixhawk/VIO adapter'ı, IMX477/Jetson latency ve terminal hover state
  machine bu sözleşme diliminin dışında kalır.

### Sonraki önerilen adım

ADR-015'i config, deterministic scheduler, approved encoder adapter ve
`EbabilVisionNavigationEnv` koduna test-first geçirmek.

## 11 Eylül 2026 — Perception review düzeltmelerinin kapanışı

### Amaç ve kapsam

`REV-007`–`REV-013` bulgularının tümünü kod, test, artefact ve belgelerde
kapatmak; düzeltmelerin birbirini bozmadığını hedefli ve tam regresyonla
doğrulamak. Bu dilim vision PPO environment'ını içermez.

### Uygulanan düzeltmeler ve kararlar

- `vision-dataset-v4` explicit `64/8/8/12` sahne kotalı
  train/validation/development-OOD/audit sözleşmesine geçirildi. Trainer ve
  quality audit'i yüklemez.
- Quality başarısızlığında CLI non-zero çıkar ve yalnız candidate yayımlar.
  Approval yalnız ayrı audit başarısından sonra üretilebilir.
- Atomik `audit_consumed.json`, başarısız veya başarılı bir audit sonrasında
  aynı quality raporunun farklı çıktı yolu üzerinden yeniden denenmesini engeller.
- Dataset config/metadata/manifest/statistics, checkpoint, quality ve audit
  artefact'ları SHA-256 ve göreli yollarla bağlandı. Runtime yükleyici bütün
  zinciri doğrular.
- Tek `preprocess_actor_rgb` offline/online sözleşmesi ve encoder dtype/aralık/
  sonluluk kontrolleri eklendi. Heatmap başları config çözünürlüklerinden
  bağımsız `9x12` adaptif pooling kullanır.
- Engel RGB kanalları geniş randomize edildi; hedef renginde çarpışmasız ve
  maskesiz distractor'lar eklendi. Train/validation/development box, cylinder,
  capsule ve sphere; audit görülmemiş ellipsoid kullanır.
- Yüzde 50 seed'li grayscale augmentation ile nominal + grayscale free-space/
  risk kalite kapıları eklendi. Risk score eşiği `0.62`, risk P/R/S kapıları
  `0.65/0.70/0.50` olarak yalnız yeni validation/development verisinden
  ikinci audit açılmadan önce donduruldu.
- Encoder/checkpoint sözleşmesi `vision-encoder-v9` /
  `ebabil-vision-checkpoint-v9` oldu. Teknik sözleşme ADR-014'te sabitlendi.

### Audit disiplini ve sonuç

- İlk bağımsız capsule/sphere audit nominal/grayscale risk precision
  `0.671/0.692` ile başarısız oldu; CLI non-zero döndü ve approval üretmedi.
- Bu audit'e eşik uydurulmadı. Eğitim geometri zarfı genişletildi, split salt'ı
  değiştirildi ve yeni seed'li ellipsoid audit ayrıldı.
- v16 üç model seed'inde development kapısını geçti. Dondurulmuş seed
  `20260911`, yeni 96-kare audit'i tek koşuda geçti.
- Audit nominal risk P/R/S `0.780/0.784/0.694`, grayscale
  `0.744/0.713/0.661`; approval checkpoint SHA-256
  `db5eca8ffa2a9a3bb03091bc67ff55a20fab9bbe8a5c5e9993c06ba3c55a848d`.

### Oluşturulan/değiştirilen başlıca dosyalar

- `src/ebabil/perception/{dataset,model,quality,visual_randomization}.py`
- `src/ebabil/scenes/generator.py`
- `src/ebabil/configs/training/{vision_dataset,vision_encoder}.toml`
- `scripts/run_vision_audit.py`, quality/dataset CLI'ları
- `tests/test_{vision_dataset,vision_encoder,scenes}.py`
- ADR-014, Faz 6.5 kaydı, review bulguları, implementasyon planı ve README
- `runs/vision_dataset_v4b_20260911`, `runs/vision_quality_v16_20260911`,
  `runs/vision_audit_v2_20260911`

### Doğrulama ve bilinen sınırlar

- Targeted perception/scene testleri temiz geçti. Tam regresyon:
  `121 passed, 1 skipped, 12 subtests passed`; skip yalnız sandbox içindeki
  beklenen macOS offscreen capture testidir.
- Ruff lint geçti; format kontrolü `81 files already formatted`; mypy
  `44` source file için temiz; compileall ve `pip check` geçti.
- Grafik erişimli 8-kare smoke ve 736-kare tam dataset üretimi leakage
  kontrolüyle geçti. MuJoCo `ARB_clip_control unavailable` uyarısı sürüyor.
- Hash-tamper regression testi değiştirilmiş checkpoint özetini reddediyor;
  gerçek approval manifest'i üzerinden model yükleme geçti.
- İzole wheel build/install smoke'ta paket içinden `vision-dataset-v4`,
  `vision-encoder-v9`, ellipsoid model derlemesi, audit API'ları ve gerçek
  approval yüklemesi geçti.
- `24` Markdown dosyasında yerel link, zero-width karakter ve HTML boşluk
  entity kontrolü temiz geçti.
- Bu approval yalnız sentetik perception içindir. Gerçek IMX477 kayıtları,
  lens/rolling-shutter/timing, Orin NX parity/latency ve uçuş güvenliği henüz
  doğrulanmadı.

### Sonraki önerilen adım

ADR-014 approval manifest'ini zorunlu giriş yapan `EbabilVisionNavigation-v0`
iskeleti, `vision-v1` observation/leakage sözleşmesi, kamera-policy scheduler ve
dinamik segment horizon testleri.

## 11 Eylül 2026 — Perception review düzeltmeleri

### REV-008 — Fail-closed quality kapısı (tamamlandı)

- `VisionQualityArtifacts` yalnız candidate checkpoint'i açıkça adlandırır.
- Başarısız kapı report/candidate tanılamasını korur, approval üretmez ve CLI
  non-zero çıkar; approval yalnız ayrı audit sürecine aittir.
- Gerçek subprocess regression testi eklendi.
- Doğrulama: encoder hedefli `7 passed`; Ruff ve mypy temiz.

### REV-009 — Ortak RGB preprocessing (tamamlandı)

- Public `preprocess_actor_rgb`, HWC `uint8` actor karesini CHW
  `float32 [0,1]` tensöre dönüştüren tek kaynaktır; offline dataset bu yolu
  kullanır ve online environment da bunu kullanacaktır.
- Encoder yanlış dtype, aralık ve sonlu olmayan girdiyi fail-fast reddeder.
- Doğrulama: perception hedefli `19 passed`; Ruff, format ve mypy temiz.

### REV-011 — Çözünürlükten bağımsız heatmap (tamamlandı)

- Convolution feature map'i heatmap başlarından önce adaptif `9x12` pooling
  kullanır; sabit soft-argmax grid'i artık config çözünürlüğüyle çatışmaz.
- Doğrulama: encoder `10 passed, 2 subtests passed`; `64x64` ve `80x128`
  regression'ları, Ruff, format ve mypy temiz.

### REV-013 — Tek kaynaklı dört-split sözleşmesi (tamamlandı)

- `vision-dataset-v4` yanıltıcı fraction alanlarını kaldırır; explicit
  train/validation/development-OOD/audit sahne sayıları hash bucket ve çıktı
  oranlarının tek kaynağıdır.
- Eski test split'i geliştirme OOD olarak adlandırıldı. Trainer/quality audit
  split'ini açmaz.
- Doğrulama: perception hedefli `20 passed, 2 subtests passed`; oran/seed
  regression'ları, Ruff, format ve mypy temiz.

## 11 Eylül 2026 — Vision environment öncesi code/document review

### Amaç ve kapsam

Faz 6.5 perception kodunu, testlerini, config'lerini, quality artefact'ını ve
proje belgelerini `EbabilVisionNavigation-v0` implementasyonundan önce
doğruluk, fail-closed çalışma, bilimsel değerlendirme ayrımı, deploy
edilebilirlik ve belge-kod tutarlılığı açısından incelemek. Bu dilimde production
kodu düzeltilmedi; bulgular kalıcılaştırıldı.

### Bulgular

- `REV-007` (yüksek): v2-v11 geliştirme kararlarında tekrar gözlenen test split'i
  artık bağımsız akademik test değildir; buna rağmen belgede öyle ima ediliyordu.
- `REV-008` (yüksek): quality CLI başarısız kapıda süreç kodu 0 döndürüyor ve
  seçilmiş checkpoint yayımlıyor.
- `REV-009` (yüksek): offline dataset ile online kamera inference arasında ortak
  RGB preprocessing API/sözleşmesi yok; uint8 çökerken float `[0,255]` sessizce
  kabul ediliyor.
- `REV-010` (yüksek): bütün domainlerde engel kırmızı/turuncu, hedef yeşil;
  distractor ve görülmemiş geometri bulunmadığından semantik renk kısa yolu
  dışlanamıyor.
- `REV-011` (orta): `>=16 px` doğrulamasından geçen `64x64` ve `80x128`
  encoder config'leri sabit `9x12` heatmap grid'inde runtime boyut hatası veriyor.
- `REV-012` (orta): quality report dataset/manifest/checkpoint hash'i taşımıyor,
  mutlak checkpoint yolu yazıyor ve dataset schema uyumluluğu zorlanmıyor.
- `REV-013` (orta): config fraction'ları `70/15/15`, explicit sahne kotaları ve
  gerçekleşen split ise `80/10/10`.

Tam kanıt, etki ve kabul ölçütleri `docs/REVIEW_FINDINGS.md` içindedir.

### Değiştirilen belgeler

- `docs/REVIEW_FINDINGS.md`
- `docs/PROGRESS_LOG.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/PHASE_6_5_VISION_DATASET_ENCODER.md`

### Salt-okunur doğrulamalar

- Encoder config reproducer: varsayılan `72x96` geçti; `64x64` ve `80x128`
  doğrulamadan sonra sırasıyla `64/108` ve `160/108` heatmap uyuşmazlığıyla
  çöktü.
- Online girdi reproducer: HWC uint8 açık schema hatası, BCHW uint8 PyTorch
  dtype hatası üretti; float `[0,255]` ise yanlış ölçeğe rağmen kabul edildi.
- Split kontrolü: bildirilen `0.70/0.15/0.15`, gerçekleşen
  `0.80/0.10/0.10`, sahne sayıları `64/8/8`.
- Quality artefact kontrolü: report'ta dataset kimliği ve SHA-256 alanları yok;
  selected checkpoint mutlak yerel yol.
- `.venv/bin/python -m pip check`: bozuk bağımlılık yok.
- `.venv/bin/python -m compileall -q src scripts tests`: geçti.
- `.venv/bin/python -m pytest -q`: `112 passed, 1 skipped, 10 subtests passed`;
  skip beklenen macOS sandbox offscreen testidir.
- Ruff lint geçti; format kontrolünde `80 files already formatted`.
- Mypy `44` source file için temiz geçti.
- `23` Markdown dosyasında yerel link ve görünmez Unicode kontrolü geçti.

### Sonraki önerilen adım

Önce `REV-008`, `REV-009` ve `REV-011` kod sözleşmelerini test-first kapatmak;
ardından `REV-013` split kaynağını tekilleştirip `REV-012` provenance manifest'ini
eklemek. Son olarak `REV-010` için semantik renk/geometri çeşitliliğiyle yeni
dataset üretmek, mevcut test'i development OOD olarak yeniden adlandırmak ve
önceden kilitlenmiş yeni audit split'ini yalnız dondurulmuş model üzerinde bir
kez çalıştırmak.

## 11 Eylül 2026 — Görsel randomization ve encoder kalite kapısı

### Amaç ve kapsam

Gerçek araçta bulunmayacak depth/raycast bilgisini actor'a vermeden, MuJoCo
RGB görüntülerinden hedef görünürlüğü/merkezi ve üç bölgeli yakın engel riskini
öğrenen hafif encoder'ı ayrı görsel domainler ve üç model seed'iyle doğrulamak.
Bu dilim perception supervision'ıdır; PPO veya gerçek uçuş güvenliği kanıtı
değildir.

### Uygulanan yapı ve teknik kararlar

- `vision-dataset-v3`, 64/8/8 ayrık sahne ve sahne başına 8 kare ile
  `512/64/64` train/validation/test kaydı üretir. Scene seed split'leri
  kesişmez; actor allowlist'i yalnız RGB, sample ID ve timestamp'tir.
- Train görsel randomization zarfı ışık, kanal tint'i, engel/hedef materyali,
  pozlama, gamma, noise, vignette ve blur bakımından genişletildi. Saklı
  validation `heldout_dim_cool`, test `heldout_bright_warm` profilidir.
- v2 ile v3 dataset'lerinin validation/test actor ve privileged dosyaları
  birleşik SHA-256 karşılaştırmasında birebir aynı kaldı; yalnız training
  dağılımı ve büyüklüğü değişti.
- `vision-encoder-v8`, `96 x 72` RGB, üç convolution katmanı, `3 x 4` uzamsal
  pooling ve 64 boyutlu latent kullanır. Hedef/engel heatmap yardımcı başları,
  soft-argmax hedef merkezi, üç bölgeli continuous free-space ve doğrudan
  yakın-engel risk başı üretir.
- Kanal bazlı actor-RGB adaptive exposure, dim/cool ve bright/warm domainleri
  arasındaki risk skoru kaymasını giderdi. Hedef görünürlük başı latent ile
  hedef heatmap maksimum/ortalama kanıtını birlikte kullanır.
- Risk etiketi privileged free-space değerinin `< 0.25` olmasıdır; 8 m
  normalizasyonla yaklaşık 2 m eşiğine karşılık gelir. MuJoCo depth uyarısı
  nedeniyle bu fiziksel eşleşme kalibre edilmeden gerçek mesafe iddiası değildir.
- Kalite kapısı her seed'in hem validation hem test split'inde ayrı ayrı
  geçmesini ister: target precision/recall `>=0.85`, center error `<=0.30`,
  free-space MAE `<=0.30`, trivial baseline iyileşmesi `>=0.15`, risk
  precision/recall/specificity sırasıyla `>=0.80/0.90/0.60`.

### Deneyler ve fix bazlı doğrulama

- v2 global/uzamsal başlangıçlarda hedef merkezi ve free-space genellemesi
  yetersizdi; `3 x 4` latent ve heatmap/soft-argmax ile center error düzeldi.
- v4 adaptive exposure hedef metriklerini geçirdi, ancak bright/warm test
  free-space MAE `0.232` ile eski `0.20` eşiğini aşmıştı. Süreyi 60 epoch'a
  çıkarmak anlamlı çözüm üretmedi.
- v5 bölgesel free-space head test MAE'yi `0.250` düzeyine kötüleştirdi ve geri
  alındı. Continuous depth başından eşiklenen riskin test specificity'si düşük
  kaldığı için ayrı risk head eklendi.
- v6 dar train domaininde validation risk recall ve test specificity birlikte
  geçmedi. Train zarfı genişletildi ve kayıt sayısı 192'den 512'ye çıkarıldı.
- v7 geniş train dağılımında ilerledi fakat sabit risk skorları domainler
  arasında kayıyordu. Kanal bazlı normalize edilen v8 hattında risk kapıları
  geçti. Görünürlük başının heatmap kanıtıyla birleştirilmesi precision/recall
  sapmasını giderdi; center-loss ağırlığının 2'den 3'e çıkarılması son merkez
  regresyonunu kapattı.
- Son koşu `runs/vision_quality_v11_20260911`: `passed=true`; seçilen seed
  `20260912`, checkpoint `vision_encoder_v8.pt`.

### Son kalite sonucu

| Split | Target P/R ort. | Center error ort./maks. | Free-space MAE ort./maks. | Risk P/R/S ort. |
|---|---:|---:|---:|---:|
| Validation | `0.963 / 0.956` | `0.278 / 0.287` | `0.149 / 0.156` | `0.959 / 0.928 / 0.882` |
| Test | `0.992 / 0.910` | `0.148 / 0.160` | `0.175 / 0.183` | `0.887 / 0.975 / 0.718` |

Üç masaüstü CPU tek-frame ölçümü `0.310-0.373 ms` aralığındadır; Jetson Orin
NX gecikmesi olarak yorumlanmaz.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/configs/training/vision_dataset.toml`
- `src/ebabil/configs/training/vision_encoder.toml`
- `src/ebabil/perception/visual_randomization.py`
- `src/ebabil/perception/dataset.py`
- `src/ebabil/perception/model.py`
- `src/ebabil/perception/quality.py`, `src/ebabil/perception/__init__.py`
- `scripts/generate_vision_dataset.py`, `scripts/run_vision_quality.py`
- `tests/test_vision_dataset.py`, `tests/test_vision_encoder.py`
- `docs/PHASE_6_5_VISION_DATASET_ENCODER.md`, implementasyon planı, README ve
  bu günlük

### Doğrulama ve bilinen sınırlar

- Hedefli perception testleri: `16 passed`.
- Tam regresyon: `112 passed, 1 skipped, 10 subtests passed`. Skip yalnız
  sandbox içindeki beklenen macOS offscreen grafik testidir.
- `.venv/bin/ruff check src scripts tests`: geçti.
- `.venv/bin/ruff format --check src scripts tests`: `80 files already formatted`.
- `.venv/bin/mypy`: `Success: no issues found in 44 source files`.
- Grafik erişimli dataset üretimi: 640 kayıt, leakage check geçti. Sandbox
  içindeki beklenen CoreGraphics hatası sonrası komut grafik erişimli süreçte
  çalıştırıldı.
- İzole wheel build/install smoke geçti; kurulu wheel içinden
  `vision-dataset-v3`, `vision-encoder-v8`, 30 epoch config'i ve üç quality
  seed'i yüklendi. Yeni perception modülleri package metadata'sına dahil edildi.
- Seçilen gerçek checkpoint yeniden yüklendi; bir test karesinde latent `(1,64)`
  ve risk `(1,3)` çıktıları sonlu üretildi.
- `23` Markdown dosyasında yerel link ve görünmez Unicode kontrolü geçti.
- MuJoCo `ARB_clip_control unavailable` uyarısı sürüyor; privileged depth
  bilinen düzlemlerle ayrıca kalibre edilmelidir.
- Sentetik hedef ve kutu engeller gerçek IMX477 dağılımını temsil etmez. Gerçek
  kamera kayıtları yalnız değerlendirme seti olarak eklenmeden sim-to-real
  iddiası kurulamaz.
- Sonraki adım encoder checkpoint'ini dondurup observation/leakage/dinamik
  horizon testleriyle `EbabilVisionNavigation-v0` iskeletini kurmaktır.

## 11 Eylül 2026 — Faz 6.5 vision dataset ve encoder altyapısı

### Amaç ve kapsam

MuJoCo IMX477 RGB görüntülerini actor'a privileged bilgi sızdırmadan seed'li
dataset'e dönüştürmek; sahne bazında ayrık train/validation/test split'i,
hedef/engel supervision'ı, küçük CNN, classical baseline, metrik ve sürümlü
checkpoint akışını kurmak. Bu dilim PPO veya gerçek perception kalite eğitimi
değildir.

### Alt dilim 1 — Dataset sözleşmesi ve leakage koruması

- `vision-dataset-v1` config'i varsayılan `24/8/8` sahne ve sahne başına 8
  frame ile `192/64/64` split planını tanımlar.
- Scene seed, salt'lı BLAKE2 hash ile bütün olarak tek split'e atanır; scene
  seed'leri split'ler arasında kesişmez.
- Pose örnekleme workspace/engel açıklığını korur. Yüzde 30 örnek hedef-yok
  sınıfı için `70-140 derece`, kalanlar `+-35 derece` yaw sapması kullanır.
- Actor NPZ allowlist'i yalnız `sample_id`, `rgb`, `timestamp_s` alanlarıdır.
  Depth, segmentation, hedef/engel maskeleri, dünya pozu ve türetilen hedefler
  ayrı privileged NPZ'dedir.
- Leakage validator anahtar, RGB dtype/shape, ID/timestamp eşleşmesi, path
  traversal, yinelenen ID ve scene-seed split sızıntısını denetler.
- Dataset ve encoder çıktıları mevcut dosyaların üzerine sessizce yazılmaz.
- `capture_camera_frame.py` birleşik actor/privileged NPZ yazmayacak şekilde
  düzeltildi; `--save-prefix` iki ayrı dosya üretir, eski `--save-npz` yalnız
  geriye uyumlu alias'tır.

### Alt dilim 2 — Supervision, encoder ve karşılaştırma hattı

- Privileged segmentation'dan `obstacle_mask` eklendi. Serbest alan etiketi
  sol/orta/sağ görüntü dilimlerinde yalnız engel maskesindeki robust depth
  quantile'ından türetilir; actor depth almaz.
- `vision-encoder-v1`, `96 x 72` RGB, üç convolution katmanı ve 32 boyutlu
  latent kullanır; target-visible, target-center ve üç bölmeli free-space
  başlıkları üretir.
- `ebabil-vision-checkpoint-v1`, dataset/model/training sürümleri, ağırlıklar ve
  metriklerle kaydedilip yeniden yüklenir.
- Precision/recall, normalize center error, free-space MAE ve masaüstü CPU
  latency raporlanır. Açıklanabilir yeşil renk hedef baseline'ı ve sabit açık
  free-space baseline'ı eklendi.
- Validation yanında bağımsız test split'i de CNN ve baseline için raporlanır.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/configs/training/vision_dataset.toml`
- `src/ebabil/configs/training/vision_encoder.toml`
- `src/ebabil/perception/dataset.py`
- `src/ebabil/perception/model.py`
- `src/ebabil/perception/camera.py`, `src/ebabil/perception/__init__.py`
- `src/ebabil/scenes/generator.py`
- `scripts/generate_vision_dataset.py`
- `scripts/train_vision_encoder.py`
- `scripts/capture_camera_frame.py`
- `tests/test_vision_dataset.py`
- `tests/test_vision_encoder.py`
- `tests/test_camera_capture.py`
- `docs/PHASE_6_5_VISION_DATASET_ENCODER.md`
- README, implementasyon planı, kamera handoff belgesi ve bu günlük

### Fix bazlı doğrulama

- Dataset/camera ilk hedefli testleri: `15 passed, 1 skipped`; mypy 42 kaynak
  dosyasında temiz.
- Viewer regresyonunda yeni package importlarının oluşturduğu circular import
  yakalandı. Sahne üreticisindeki kamera config importu runtime'a ertelendi;
  hedefli viewer + perception seti tekrarında `19 passed, 1 skipped` ve mypy
  temiz geçti.
- Capture birleşik NPZ fix'i gerçek render ile doğrulandı: actor dosyasında
  yalnız `rgb/sample_id/timestamp_s`; privileged dosyada
  `depth/segmentation/target_mask/obstacle_mask/sample_id/timestamp_s` kaldı.

### Gerçek render ve eğitim smoke sonucu

- Grafik erişimli MuJoCo sürecinde üç farklı scene seed'den, her split'te iki
  frame olacak şekilde 6 örnek üretildi; leakage denetimi geçti.
- Hedef görünürlük dağılımında hem pozitif hem negatif örnek oluştu.
- Bir epoch eğitim checkpoint'i yazıldı ve unit testte yeniden yüklenebildi.
- Küçük validation smoke'unda renk/trivial baseline target precision/recall
  `1.0/1.0`, center error `0.005`, free-space MAE `0.823`; CNN sırasıyla
  `0.5/1.0`, `0.600`, `0.343` verdi. Bunlar kalite sonucu değildir.
- Masaüstü CPU tek-frame ölçümü yaklaşık `0.205 ms` oldu; Jetson gecikmesi
  olarak yorumlanamaz.
- MuJoCo `ARB_clip_control unavailable` depth uyarısı devam etti. Free-space
  supervision fiziksel mesafe etiketi olarak kullanılmadan önce depth
  kalibrasyonu zorunludur.

### Bilinen sınırlılıklar ve sonraki adım

- Gerçek hedef görünümü tanımlı olmadığı için sabit yeşil hedef renk baseline'ı
  tarafından yapay biçimde kolay çözülür.
- Görsel texture/ışık/pozlama/blur/lens/timing randomization henüz üreticiye
  bağlı değildir; varsayılan 320-frame dataset ve çoklu-seed kalite koşusu
  çalıştırılmadı.
- Sonraki adım held-out görsel domainleri uygulamak, tam dataset/kalite raporunu
  üretmek ve yalnız kalite kapısı geçerse encoder'ı `vision-v1` observation'a
  almaktır.

### Tam regresyon ve paket doğrulaması

- `.venv/bin/python -m pytest -q`: `106 passed, 1 skipped, 10 subtests passed`.
- Skip yalnız sandbox içindeki macOS offscreen grafik testidir; gerçek dataset
  ve ayrı actor/privileged capture grafik erişimli süreçte başarıyla çalıştı.
- `.venv/bin/ruff check src scripts tests`: geçti.
- `.venv/bin/ruff format --check src scripts tests`: `77 files already formatted`.
- `.venv/bin/mypy`: `Success: no issues found in 42 source files`.
- Wheel build ve ayrı target dizinine `--no-deps` install smoke'u geçti;
  kurulu wheel içinden `vision-dataset-v1` ve `vision-encoder-v1` config'leri
  yüklendi.
- `23` Markdown dosyasında yerel link ve görünmez Unicode kontrolü geçti.

## 11 Eylül 2026 — Faz 6.5 kamera/target ilk implementasyon dilimi

### Amaç ve kapsam

Parametrik IMX477 simülasyon kamerasını, dikdörtgen hedef alanı, güvenli hover
iç bölgesini, headless actor/privileged capture sınırını ve approach/hover
görünürlük araçlarını legacy ortamı bozmadan uygulamak.

### Alt dilim 1 — Config ve saf hedef geometrisi (tamamlandı)

- `src/ebabil/configs/environment/camera.toml`: IMX477 sensör/çıktı çözünürlüğü,
  `75 derece` yatay FOV, `30 Hz`, gövde montaj konumu ve `45 derece` aşağı pitch.
- `src/ebabil/perception/config.py`: doğrulanan `CameraConfig`, türetilmiş dikey
  FOV, focal length, principal point ve FLU -> MuJoCo camera axes.
- `src/ebabil/configs/environment/target_area.toml`: `2 x 2 m` hedef, `0.35 m`
  safety margin ve ADR-012 hover eşikleri.
- `src/ebabil/scenes/target.py`: `TargetAreaConfig`, hover anchor, hedef köşeleri
  ve güvenli iç bölge membership hesabı.
- `tests/test_vision_config.py`: config, eksen, köşe, anchor ve geçersiz margin
  testleri.

Doğrulama:

- `.venv/bin/python -m pytest -q tests/test_vision_config.py`: `7 passed`.
- `.venv/bin/ruff check src/ebabil/perception src/ebabil/scenes/target.py tests/test_vision_config.py`: geçti.
- `.venv/bin/mypy`: `Success: no issues found in 39 source files`.

### Alt dilim 2 — MuJoCo model ve capture (tamamlandı)

- `build_scene_model` quadrotor body'ye config tabanlı `imx477` kamera ekliyor.
- Worldbody'ye collision üretmeyen hedef alan, güvenli iç bölge ve merkez işareti
  eklendi.
- `ActorCameraFrame` yalnız RGB ve timestamp taşır;
  `PrivilegedCameraLabels` depth, segmentation ve target mask taşır.
- `project_world_points` MuJoCo camera pose'u ile pinhole projection yapar.
- Legacy yeşil hedef küresi vision modelinde kapatıldı; `HeadlessCamera` marker
  sızıntısını fail-fast reddeder.
- `scripts/capture_camera_frame.py` actor/privileged capture smoke aracı olarak
  eklendi.

Doğrulama:

- Normal/sandbox hedefli test seti: `27 passed, 1 skipped, 3 subtests passed`.
- Skip edilen gerçek offscreen test grafik erişimli süreçte ayrıca `1 passed`.
- Start capture: RGB `(240, 320, 3)`, depth `(240, 320)`, target `2532` piksel.
- Hover capture: RGB `(240, 320, 3)`, depth `(240, 320)`, target `8974` piksel.
- MuJoCo/CGL depth render sırasında `ARB_clip_control unavailable` uyarısı
  verdi; privileged depth sonraki kalibrasyon işi olarak kaydedildi.

### Alt dilim 3 — Projection, görünürlük ve viewer (tamamlandı)

- Optical-axis/principal-point projection, target corner görünürlüğü ve
  actor-schema leakage testleri eklendi.
- `scripts/sweep_camera_visibility.py` `30-60 derece` pitch ve `6-0 m`
  yaklaşma sweep'i olarak eklendi.
- `40/45/50 derece` adayları minimum `%25` görünürlük kapısını geçti; varsayılan
  `45 derece` start ve hover'da `%50` köşe görünürlüğü verdi.
- `scripts/view_target_camera.py` overview ve onboard sabit kamera modlarıyla
  eklendi; dry-run testleri geçti.
- Teknik handoff `docs/PHASE_6_5_CAMERA_TARGET.md` içinde tutuluyor.

### Oluşturulan/değiştirilen dosyalar

- `src/ebabil/configs/environment/camera.toml`
- `src/ebabil/configs/environment/target_area.toml`
- `src/ebabil/perception/__init__.py`
- `src/ebabil/perception/config.py`
- `src/ebabil/perception/camera.py`
- `src/ebabil/scenes/target.py`
- `src/ebabil/scenes/generator.py`
- `src/ebabil/scenes/__init__.py`
- `scripts/capture_camera_frame.py`
- `scripts/sweep_camera_visibility.py`
- `scripts/view_target_camera.py`
- `tests/test_camera_capture.py`
- `tests/test_vision_config.py`
- `tests/test_viewer_scripts.py`
- `docs/PHASE_6_5_CAMERA_TARGET.md`
- `README.md`, `docs/IMPLEMENTATION_PLAN.md`, ADR-011 ve bu günlük

### Tam doğrulama

- `.venv/bin/python -m pytest -q`: `96 passed, 1 skipped, 10 subtests passed`.
- Skip, sandbox içindeki macOS CoreGraphics kısıtlı offscreen testidir; aynı
  test sandbox dışında ayrıca `1 passed`.
- `.venv/bin/ruff check src scripts tests`: geçti.
- `.venv/bin/ruff format --check src scripts tests`: `71 files already formatted`.
- `.venv/bin/mypy`: `Success: no issues found in 40 source files`.
- Kamera sweep, start/hover capture dry-run ve overview/onboard viewer dry-run
  komutları geçti.
- `21` Markdown dosyasında yerel bağlantı ve görünmez Unicode kontrolü geçti.
- İzole wheel build/install smoke geçti; `ebabil.perception`, `camera.toml` ve
  `target_area.toml` kurulu wheel içinden başarıyla yüklendi.
- Wheel üretimi `src/ebabil_mujoco.egg-info/SOURCES.txt` paket metadata'sını yeni
  kaynaklarla güncelledi.

### Sonraki adım

Seed'li görsel dataset üreticisi ve hafif encoder baseline'ını uygulamak; sonra
`EbabilVisionNavigation-v0` observation ve dinamik horizon iskeletine geçmek.

## 11 Eylül 2026 — Hedef bölge hover ve ölçeklenebilir episode planı

### Amaç ve kapsam

Terminal görevi hedef bölgenin üzerinde kararlı hover olarak kesinleştirmek;
harita mesafesi büyüdüğünde sabit `20 s` episode sınırının yetersiz kalmasını
PPO kredi ufkunu kontrolsüz büyütmeden çözmek; güncel ilerleme ve implementasyon
sırasını yeniden hazırlamak.

### Kabul edilen hover sözleşmesi

- Aktif terminal görev yalnız hedef bölgenin güvenli iç alanı üzerinde hover'dır;
  iniş ve yanal inceleme ana implementasyon yolundan çıkarılıp geleceğe ertelendi.
- Durum makinesi `transit -> acquired -> approach -> settle -> hold -> success`
  olacaktır; target-loss, stale state veya güvensiz durumda abort uygulanır.
- İlk simülasyon varsayımları `2.0 m` hover yüksekliği, `0.20 m` irtifa
  toleransı, `0.25 m/s` maksimum doğrusal hız, `10 derece` maksimum roll/pitch,
  `0.35 rad/s` maksimum açısal hız ve `3 s` kesintisiz hold'dur.
- Yatay başarı alanı, hedef dikdörtgeninin araç izdüşümü ve konum belirsizliği
  için safety margin ile küçültülmesinden elde edilir.
- Başarı ground-truth ile ölçülebilir fakat privileged konum actor'a verilmez.
- Kamera montaj/FOV'u hedefi approach ve hover sırasında görmeli veya kısa ve
  ölçülmüş VIO grace period ile takip edebilmelidir.

### Episode süresi çözümü

- Legacy `EbabilNavigation-v0` için `20 s / 400 adım` değişmedi.
- Yeni vision eğitimi D0-V'deki kısa görevlerden başlayıp `5-30 m` yerel
  segment episode'larına genişleyecek.
- Yerel zaman bütçesi planlanan rota uzunluğu, nominal hız, `k_time`, sabit
  toparlanma payı ve terminal rezervinden hesaplanıp `T_min/T_max` ile
  sınırlandırılacak.
- İlk değerler `v_nominal=1.5 m/s`, `k_time=1.75`, `T_fixed=5 s`, yerel
  `T_min=20 s` ve `T_max=60 s` olarak kabul edildi.
- D3-D5 uzun görevlerinde ara waypoint episode'u bitirmeyecek ve fizik state'i
  sıfırlanmayacak. Mission manager segmentleri zincirleyecek; nihai hedefte
  hover hold tamamlanınca başarı oluşacak.
- Zaman dolumu `truncated=True`; success/collision gibi son durumlar
  `terminated=True` olacak.
- Normalize kalan bütçe deploy edilebilir actor girdisi ve ayrıntılı `info`
  metriği olacaktır. PPO rollout uzunluğu harita mesafesiyle doğrusal
  artırılmayacaktır.

### Plan ve teknik kararlar

- ADR-010 `Önerildi` durumundan `Kabul edildi` durumuna geçirildi.
- ADR-012 aktif terminal görev ve ölçülebilir hover başarı koşullarıyla
  kesinleştirildi.
- ADR-013 segmentli episode, rota temelli zaman bütçesi ve Gymnasium
  `terminated/truncated` semantiği için oluşturuldu.
- `docs/IMPLEMENTATION_PLAN.md` tamamlanmış legacy fazlar sonrasında kamera,
  perception, vision environment, hover, D0-V–D2 yerel eğitim ve D3-D5 mission
  evaluation sırasıyla yeniden düzenlendi.

### Değiştirilen dosyalar

- `docs/decisions/ADR-010-hierarchical-distance-curriculum.md`
- `docs/decisions/ADR-012-safe-terminal-mission-scope.md`
- `docs/decisions/ADR-013-segmented-episodes-and-time-budget.md`
- `docs/PROJECT_DESIGN.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `README.md`
- `docs/PROGRESS_LOG.md`

### Doğrulama

- Belge içi hover görevi, D0-V–D5 curriculum'u, episode semantiği ve ADR yerel
  bağlantı kontrolleri geçti.
- `.venv/bin/python -m pytest -q`: `83 passed, 10 subtests passed`.
- `.venv/bin/ruff check src scripts tests`: geçti.
- `.venv/bin/ruff format --check src scripts tests`: `62 files already formatted`.
- `.venv/bin/mypy`: `Success: no issues found in 36 source files`.
- Runtime kodu bu iş diliminde değiştirilmedi; tam regresyon temiz geçti.

### Sonraki adım

`docs/IMPLEMENTATION_PLAN.md` bölüm 19'daki kamera/target ilk kod dilimini
uygulamak; dinamik horizon kodunu vision environment iskeletiyle birlikte
eklemek.

## 11 Eylül 2026 — Pixhawk 2.4.8 kararı ve donanım yeterlilik kapısı

### Amaç ve kapsam

IMX477 sensör ailesini ve Pixhawk 2.4.8 FC modelini proje girdisi olarak
kesinleştirmek; eski/klon Pixhawk varyantlarının firmware ve sensör riskini
resmî kaynaklarla inceleyip implementasyon planına yansıtmak.

### Teknik kararlar

- Kamera sensörü IMX477 olarak kabul edildi. Lens, FOV, shutter, CSI bağlantısı
  ve sürücü sürümü gerçek donanım kalibrasyon/bench girdisi olarak parametrik
  kalacak; vision environment implementasyonunu bloke etmeyecek.
- FC modeli Pixhawk 2.4.8 olarak sabitlendi.
- `2.4.8` etiketi üzerinden FMUv2/FMUv3, flash veya sensör bileşimi varsayılmayacak.
  QGC/USB üzerinden gerçek board ID, firmware hedefi, flash ve sensör envanteri
  kaydedilecek.
- QGC bir autopilot değildir; kartta PX4 veya ArduPilot'tan yalnız biri çalışır.
  Firmware seçimi bench bulgularından sonra ayrı karar olarak kaydedilecek.
- Hedefe çarpmayı başarı yapan veya zarar vermeye yönelik terminal güdüm,
  akademik amaç belirtilmiş olsa da ADR-012 kapsamı dışında kalır. Güvenli alan
  tespiti, hover, kontrollü iniş, temassız inceleme ve abort çalışmaları devam
  edebilir.

### Resmî kaynak bulguları

- PX4, klasik Pixhawk 1/FMUv2 hedefinde 1 MB flash sınırı nedeniyle bazı
  modüllerin kapalı olabileceğini ve Pixhawk 1'in discontinued olduğunu belirtir.
- ArduPilot güncel Copter ağacında `Pixhawk1` ve `Pixhawk1-1M` hedefleri yayımlar.
- ArduPilot ayrıca bazı `Pixhawk 2.4.x` etiketli kartlarda barometre/IMU
  ikamelerinin pre-arm sorunlarına neden olabileceği konusunda uyarır.

Kaynak bağlantıları ADR-011 ve `docs/LITERATURE_REVIEW.md` içinde tutuldu.

### Değiştirilen dosyalar

- `docs/decisions/ADR-011-camera-only-pixhawk-jetson-architecture.md`
- `docs/PROJECT_DESIGN.md`
- `docs/IMPLEMENTATION_PLAN.md`
- `docs/LITERATURE_REVIEW.md`
- `README.md`
- `docs/PROGRESS_LOG.md`

ADR-011 içindeki yarım kalmış GPS-denied hedef paragrafı da düzeltilmiştir.

### Doğrulama

- Donanım destek iddiaları PX4, ArduPilot ve QGC'nin resmî belgeleriyle kontrol
  edildi.
- Pixhawk modelinin artık “bekleyen girdi” olarak geçmediği; firmware, gerçek
  board ID ve sensör envanterinin ise açık risk olarak korunduğu belgeler arası
  aramayla doğrulandı.
- Değiştirilen belgelerde görünmez Unicode karakteri ve yarım kalan cümle için
  otomatik tutarlılık kontrolleri geçti.
- Yerel Markdown bağlantı kontrolü geçti.
- `.venv/bin/python -m pytest -q`: `83 passed, 10 subtests passed`.
- `.venv/bin/ruff check src scripts tests`: geçti.
- `.venv/bin/ruff format --check src scripts tests`: `62 files already formatted`.
- `.venv/bin/mypy`: `Success: no issues found in 36 source files`.
- Runtime kodu değişmedi; tam regresyon yine de çalıştırıldı ve temiz geçti.

### Sonraki adım

Güvenli vision environment için parametrik IMX477 camera config ve headless
RGB/privileged-label capture dilimini implemente etmek. Fiziksel kart erişilebilir
olduğunda Pixhawk 2.4.8 pervanesiz donanım envanteri bench gate'ini çalıştırmak.

## 11 Eylül 2026 — Kesin donanım, GPS-denied görev ve güvenli kapsam

### Kullanıcıdan alınan girdiler

- Kamera: `IMX477 12MP Motorized Focus Camera for Jetson`.
- FC: Pixhawk 2.4.8; PX4 ve ArduPilot seçenekleri değerlendiriliyor, QGC
  bilgisayarda kurulu.
- MC: Waveshare Jetson Orin NX AI Kit 8GB.
- GPS/GNSS kesinlikle kullanılmayacak.
- Hedef ilk olarak dikdörtgen/kare bir alan; kamera alanı gördüğünde terminal
  moda geçmesi isteniyor.

### Güvenlik kapsamı

Otonom çarpma, saldırı veya zarar vermeye yönelik terminal güdüm proje kapsamı
dışına alındı. Desteklenen terminal modlar güvenli alan tespiti, düşük hızda
hover, kontrollü iniş, temassız inceleme ve hedef kaybında abort ile sınırlıdır.
Hedefe veya zemine kontrolsüz çarpma başarı değil failure'dır. Bu sınır ADR-012
ile sabitlendi.

### Donanım araştırması

- Verilen kamera ürün adı Arducam `B0273` ile eşleşmektedir; kesin üretici/SKU
  fiziksel etiketten doğrulanmalıdır.
- Arducam sayfası `4056 x 3040` sensör, motorized focus, `3.9 mm` lens ve
  `75 derece` yatay FOV bildirir.
- Aynı üretici sürücünün JetPack/kernel sürümüne bağlı olduğunu ve üçüncü taraf
  carrier board desteğinin garanti edilmediğini belirtir.
- Waveshare kit üçüncü taraf `JETSON-ORIN-IO-BASE` carrier board kullandığı için
  SKU/FFC/CSI/JetPack driver uyumluluğu ayrı bench gate oldu.
- QGC hem PX4 hem ArduPilot firmware'ini Pixhawk ailesine yükleyebilir; ancak
  kartta aynı anda tek autopilot çalışır. Firmware seçimi hâlâ gereklidir.

### GPS-denied sonuçları

- Simülatör ground-truth hedef koordinatı deploy actor'a verilemez.
- Hedef alan başlangıçta görünmüyorsa takeoff-relative VIO tek başına uzun
  mesafede drift biriktirir; görsel loop closure/relocalization veya önceden
  tanımlı görsel harita gerekir.
- Kilometre curriculum'u localization kalite kapısı geçilmeden açılmayacaktır.
- Hedef görüldüğünde güvenli terminal state machine'e geçiş yapılabilir; görsel
  tespit tek başına episode başarısı sayılmaz.

### Teknik kararlar

- Orin NX için model ve görüntü çözünürlüğü cihaz üzerinde capture-to-setpoint
  latency ölçülerek seçilecek.
- Motorized focus uçuş sırasında varsayılan olarak değiştirilmeyecek; kalkış
  öncesi kalibre edilip kilitlenecek. Autofocus ayrı ablation olacaktır.
- Carrier board, soğutma, NVMe, kamera ve güç dönüştürücünün toplam kütle,
  ortalama/tepe güç ve termal throttling etkisi ölçülecek.

### Değiştirilen dosyalar

- ADR-011 Jetson Nano varsayımından IMX477 + Waveshare Jetson Orin NX 8GB
  donanımına geçirildi ve camera/carrier bench gate eklendi.
- `docs/decisions/ADR-012-safe-terminal-mission-scope.md` eklendi.
- `README.md`, `docs/PROJECT_DESIGN.md`, `docs/IMPLEMENTATION_PLAN.md`,
  `docs/LITERATURE_REVIEW.md` ve bu günlük donanım/görev kararıyla güncellendi.

### Doğrulama

- Arducam kamera, Waveshare Orin NX kit ve QGC firmware davranışı resmi üretici
  belgelerinden kontrol edildi; bağlantılar ADR-011 ve literatür incelemesinde.
- Yeni plan GPS/GNSS kullanmayacak şekilde dokümanlar arasında karşılaştırıldı.
- Runtime kodu değişmedi; bu iş diliminde test çalıştırılmadı.

### Sonraki adım

Kamera üzerindeki lens/carrier/sürücü ayrıntılarını, Pixhawk 2.4.8 gerçek board
ID/FMU hedefini ve PX4/ArduPilot seçimini doğrulamak; güvenli terminal görev
olarak hover, iniş veya temassız incelemeden birini seçmek. Ardından camera
config/render ilk kod dilimine geçmek.

## 11 Eylül 2026 — LiDAR'sız hedef donanım ve kamera mimarisi

### Amaç ve kapsam

Hedef araçta LiDAR bulunmadığı; tek kamera, Pixhawk FC ve Jetson Nano MC
kullanılacağı bilgisini mevcut RL, perception, curriculum ve büyük alan
tasarımıyla uzlaştırmak.

### Ana sonuç

Mevcut `EbabilNavigation-v0` içindeki 16 planar rangefinder actor girdisi gerçek
araca taşınamaz. Bu ortam ve tamamlanmış 100k PPO koşusu regression/altyapı
baseline'ı olarak donduruldu. Kalan iki rangefinder seed ve 1M full kalite koşusu
ana geliştirme yolundan çıkarıldı; hesaplama bütçesi kamera tabanlı ortama
aktarılacak.

### Kabul edilen mimari

- Tek kamera dış algının tek fiziksel kaynağıdır; model öğrenilene kadar
  monoküler RGB geçici varsayımdır.
- Jetson Nano görüntü capture/preprocessing, hafif visual encoder, PPO inference,
  waypoint ve üst seviye safety kontrollerini yürütür.
- Pixhawk üzerindeki seçilecek autopilot state estimation, attitude/body-rate
  kontrolü, mixer ve temel failsafe'i yürütür; PX4 henüz yalnız entegrasyon
  adayıdır.
- Jetson, Pixhawk'a doğrudan motor komutu yerine MAVLink üzerinden
  sınırlandırılmış yüksek seviye setpoint gönderir.
- Deploy actor yalnız kamera türevi bilgi, gerçek araçta üretilebilen Pixhawk/VIO
  state'i ve waypoint girdisini kullanabilir.
- MuJoCo depth, segmentation, raycast ve tam state yalnız perception etiketi,
  privileged teacher/critic, reward, termination veya evaluation metriğidir.
- Tek kameranın yan/arka kör bölgeleri nedeniyle vision action sözleşmesi mevcut
  dört yönlü hız komutunu aynen kullanmayacak; ilk aday ileri hız, dikey hız ve
  yaw-rate'tir.

### Konum kestirimi bulgusu

Pixhawk IMU/attitude verisi tek başına kilometre ölçekli hedefin göreli konumunu
sağlamaz. PX4 companion computer kaynaklı external vision/VIO pose ve velocity
verisini kabul edebilir. Bununla birlikte monoküler VIO, mutlak düzeltme olmadan
uzun görevde biriken drift'i ortadan kaldırmaz. GPS/GNSS kullanımı veya hedefin
görsel olarak tanımlanması observation v2 kesinleşmeden kararlaştırılmalıdır.

### Jetson Nano kararı

- Yüksek çözünürlüklü doğrudan RGB-to-PPO ilk yaklaşım olmayacaktır.
- Hafif görsel encoder + düşük boyutlu PPO politikası kullanılacaktır;
  uçtan uca fine-tuning ablation olarak kalır.
- Model ve çözünürlük Jetson üzerinde capture-to-setpoint latency, bellek ve
  doğruluk birlikte ölçülmeden sabitlenmeyecektir.
- ONNX/TensorRT optimizasyonu PyTorch parity ve gerçek cihaz benchmark'ından
  sonra kabul edilecektir.

### Kaynaklar

- CAD2RL ve Learning High-Speed Flight in the Wild kamera sim-to-real,
  privileged teacher ve kısa-horizon planlama kararlarını destekledi.
- PX4 VIO ve companion computer resmi belgeleri Jetson/Pixhawk sorumluluk
  ayrımını doğruladı.
- NVIDIA Jetson Nano ve TensorRT resmi belgeleri cihaz üstü profiling ve
  optimize inference gereksinimini destekledi.
- Ayrıntılı bağlantılar `docs/LITERATURE_REVIEW.md` ve ADR-011 içindedir.

### Değiştirilen dosyalar

- `docs/decisions/ADR-011-camera-only-pixhawk-jetson-architecture.md` eklendi.
- ADR-006, ADR-008 ve ADR-009 legacy rangefinder kapsamını belirten notlarla
  güncellendi; ADR-010 kamera algısıyla ilişkilendirildi.
- `docs/PROJECT_DESIGN.md`, `docs/IMPLEMENTATION_PLAN.md`,
  `docs/LITERATURE_REVIEW.md`, `docs/PHASE_6_PPO_BASELINE.md`, `README.md` ve bu
  ilerleme günlüğü güncellendi.

### Doğrulama ve sınırlılıklar

- Eski observation/action checkpoint sözleşmeleri değiştirilmedi.
- Yeni environment ID'si, actor/privileged veri sınırı ve Faz 6.5 kalite kapısı
  dokümanlar arasında karşılaştırıldı.
- Yalnız dokümantasyon değişti; runtime test çalıştırılmadı.
- Kamera ve hedef/localization ayrıntıları gelmeden action sınırları, görüntü
  çözünürlüğü ve VIO mimarisi kesinleştirilemez.

### Sonraki adım

Donanım girdilerini kesinleştirdikten sonra kamera MJCF/config, deterministik
headless RGB/depth/segmentation capture ve privileged leakage testlerini içeren
Faz 6.5'in ilk kod dilimini uygulamak.

## 11 Eylül 2026 — Büyük alan ve mesafe curriculum'u tasarım incelemesi

### Amaç ve kapsam

Mevcut `6 x 5 x 2.5 m` ortamın yüzlerce metre/kilometre ölçekli gerçek görev
senaryolarına nasıl genişletileceğini ve curriculum learning'in bu genişlemede
hangi rolü üstleneceğini değerlendirmek.

### Bulgular

- Observation v1 hedef bileşenlerini `6 m` ölçeğinde kırptığı için uzak hedefler
  ayırt edilemiyor.
- Güncel `1.5 m/s` hız ve `20 s` episode, kilometre göreviyle uyumlu değil;
  `1 km` için mevcut hızda teorik uçuş alt sınırı yaklaşık `667 s`.
- `0.10 m` çözünürlüklü mevcut rota grid'ini `1 km x 1 km` alana yaymak yaklaşık
  `100 milyon` hücre üretir ve episode başına sahne doğrulaması için uygun değil.
- Sabit `5 m` rangefinder menzili daha yüksek seyir hızında frenleme/gecikme
  bütçesi doğrulanmadan güvenli kabul edilemez.

### Teknik öneri

- Kilometre görevini global rota/waypoint üretici, yerel PPO kaçınma politikası
  ve klasik uçuş kontrolcüsü şeklinde üç katmana ayırmak.
- Görev mesafesini engel yoğunluğundan bağımsız, başarı tabanlı curriculum ekseni
  yapmak.
- Tüm büyük dünyayı MuJoCo'ya yüklemek yerine ego-merkezli yakın çevreyi fiziksel
  sahnede tutmak; global dünyayı seyrek harita olarak yönetmek.
- Faz 6 karşılaştırmalarını bozmamak için `EbabilNavigation-v0`/observation v1'i
  değiştirmemek; ölçeklenebilir waypoint gözlemini yeni sürümde tasarlamak.
- Eski curriculum seviyelerini örnekleme karışımında tutarak catastrophic
  forgetting'i ayrıca ölçmek.

### Değiştirilen dosyalar

- `docs/decisions/ADR-010-hierarchical-distance-curriculum.md` eklendi ve karar
  kabul öncesi doğrulama gerektiren **öneri** olarak kaydedildi.
- `docs/PROJECT_DESIGN.md` büyük alan için hiyerarşik mimariyle genişletildi.
- `docs/IMPLEMENTATION_PLAN.md` mesafe curriculum seviyeleri ve gerekli altyapı
  işleriyle genişletildi.
- `docs/PROGRESS_LOG.md` bu iş dilimiyle güncellendi.

### Doğrulama

- Mevcut scene, navigation ve sensor TOML dosyalarıyla gerçek çalışma sınırları
  karşılaştırıldı.
- Environment observation kırpması, timeout, hız limitleri, workspace sonlandırma
  ve grid rota araması kod üzerinden kontrol edildi.
- Yalnızca dokümantasyon değişti; çalışma zamanı sözleşmesi ve checkpoint
  uyumluluğu değiştirilmedi. Bu nedenle eğitim/test koşusu gerekmedi.

### Riskler ve sonraki adım

- Global planlayıcı türü, yerel pencere boyutu ve hedef seyir hızı henüz
  seçilmedi; ADR-010 bu nedenle `Önerildi` durumundadır.
- Güncel Faz 6 hedefi değişmedi: kalan iki 100k pilot seed koşusu tamamlanmalı.
- Ardından Faz 7'nin ilk dilimi olarak D1 (`5-15 m`) waypoint/observation v2
  prototipi ve hız-algı bütçesi uygulanmalıdır.

## 11 Eylül 2026 — Faz 6 tek-seed 100k pilot

### Amaç

Sabit başlangıç/hedef, engelsiz sahne ve ideal sensör dağılımında PPO'nun öğrenme
sinyalini, held-out başarısını ve baseline üstünlüğünü tam üç-seed deneyinden
önce tek seed ile doğrulamak.

### Koşu

```bash
.venv/bin/python scripts/run_phase6_experiment.py \
  --stage pilot --training-seed-count 1 --device cpu \
  --output-dir runs/phase6-pilot-seed-20260910
```

- Training seed: `20260910`
- İstenen timestep: `100,000`
- PPO rollout sınırı nedeniyle gerçek timestep: `102,400`
- Held-out episode: manifestteki ayrık `20` seed
- Eğitim süresi: yaklaşık `5 dakika 25 saniye`
- Dört `SubprocVecEnv`, CPU

### Öğrenme eğrisi özeti

- `32,768` timestep: stochastic rollout başarı `%1.22`
- `49,152` timestep: stochastic rollout başarı `%10`
- `50,000` deterministic evaluation: `%100`, return `38.31`, `109` adım
- `77,824` timestep: stochastic rollout başarı `%63`
- `90,112` timestep: stochastic rollout başarı `%97`
- `98,304` timestep: stochastic rollout başarı `%100`
- `100,000` deterministic evaluation: `%100`, return `38.68`, `61` adım

### Held-out ve baseline sonuçları

| Politika | Başarı | Çarpışma | Ortalama return | Ortalama adım | Ortalama yol |
|---|---:|---:|---:|---:|---:|
| PPO best | `%100` (`20/20`) | `%0` | `38.6791` | `61.0` | `4.1447 m` |
| Direct-goal | `%100` (`20/20`) | `%0` | `38.4351` | `118.0` | `4.2850 m` |
| Random | `%0` | `%0` | `-25.2454` | `359.5` | `7.5612 m` |

- PPO termination dağılımı: `goal=20`.
- PPO ortalama final hedef mesafesi: `0.13071 m`.
- Daha yüksek PPO return'ı daha kısa episode/yol ile açıklanıyor; sayısal
  metriklerde reward hacking belirtisi görülmedi.
- Seçilmiş replay başlangıç dahil `1526` fizik frame'iyle viewer dry-run'da
  doğrulandı.

### Rapor semantiği düzeltmesi

İlk aggregate rapor tek-seed pilot performans kontrollerini final kalite kapısı
gibi `True` göstermişti. Sözleşme ikiye ayrıldı:

- `performance_checks_passed=True`: pilot başarı/çarpışma ve iki baseline return
  kontrolünü geçti.
- `quality_gate_passed=False`: final kapı için `stage=full`, üç training seed ve
  held-out manifestin tamamı zorunlu.

Mevcut pilot `summary.json` yeni semantiğe geçirildi ve PPO rollout yuvarlamasını
görünür kılmak için gerçek `102400` timestep training/aggregate metadata'sına
eklendi.

### Final doğrulama

- `83` test ve `10` subtest başarılı.
- Ruff lint/format başarılı.
- Mypy `36` üretim dosyasında hatasız.
- Compile, `pip check` ve gerçek pilot replay dry-run başarılı.

### Bilinen sınırlılıklar

- Sonuç yalnız bir training seed ve `fixed + clear + ideal` dağılımına aittir.
- GUI replay henüz insan tarafından görsel reward-hacking incelemesinden
  geçirilmedi; otomatik dry-run ve sayısal rota metrikleri başarılıdır.
- Engelden kaçınma bu politikanın veya Faz 6 kalite iddiasının parçası değildir.

### Sonraki adım

- Kalan `20260911` ve `20260912` seed'lerinde aynı 100k pilotu çalıştırıp
  başarı, return ve sample-efficiency varyansını karşılaştırmak.
- Üç 100k koşu benzer sonuç verirse, doğrudan 1M harcamak yerine Faz 6 bütçesini
  kanıta dayalı olarak yeniden değerlendirmek.

## 10 Eylül 2026 — Faz 6 kalite kapısı hazırlığı

### Amaç

Uzun PPO koşularından önce kalite kapısında eksik olan hedefte kararlı bekleme,
endpoint dağılımları, held-out seed'ler, karşılaştırma baseline'ları ve çoklu
seed deney özetini tamamlamak.

### İş dilimi 1 — Hedefte kararlı bekleme

- `termination.goal_hold_seconds = 0.50` eklendi; `20 Hz` policy hızında başarı
  için 10 kesintisiz uygun adım gerekir.
- Hedef yarıçapı veya düşük hız koşulu bozulursa sayaç sıfırlanır.
- Goal bonus yalnız bekleme tamamlandığında verilir.
- Hold adımı, gerekli adım ve süre `info` telemetrisi içinde raporlanır.
- ADR-008 yeni termination semantiğiyle güncellendi.

Doğrulama:

- `15` environment testi ve `7` subtest başarılı.
- İlk dokuz uygun adımda termination/goal bonus olmadığı, onuncu adımda
  `goal` başarısı geldiği doğrulandı.
- Ruff ve mypy başarılı.

### Sıradaki iş dilimi

- Tek training seed, yirmi held-out episode ve sabit endpoint senaryosuyla 100k
  timestep pilot çalıştırmak.

### İş dilimi 2 — Endpoint dağılımları ve held-out protokol

- `fixed`, `random_start`, `random_goal`, `random_both` endpoint modları eklendi.
- Rastgele endpoint'ler scene seed'ine bağlı, güvenli 3B sınırlar içinde ve en
  az `2.0 m` ayrımla üretilir.
- Üç training seed'i, onlarla ayrık yirmi held-out seed, pilot/full bütçeleri ve
  dört senaryo `phase6_protocol.toml` içinde sabitlendi.
- Replay metadata ve viewer endpoint modunu taşıyıp aynı sahneyi yeniden kurar.

Doğrulama: `36` hedefli test + `10` subtest; protokol için ek `2` test; Ruff ve
mypy başarılı.

### İş dilimi 3 — Baseline ve deney orkestrasyonu

- Deterministik random ve privileged direct-goal baseline eklendi.
- İki held-out CLI kontrolünde direct-goal `%100`, random `%0` başarı verdi.
- PPO evaluation varsayılanı ad-hoc ardışık seed yerine held-out manifest oldu.
- Pilot/full, seed seçimi, baseline, train, best-checkpoint evaluation, replay
  ve aggregate kalite raporu `run_phase6_experiment.py` içinde birleştirildi.
- Kalite kontrolleri başarı/çarpışma eşikleri ile PPO return'ının random ve
  direct-goal baseline'ları aşmasını ayrı ayrı raporlar.
- Eğitim/deney runtime hataları metadata'da `failed` olarak saklanır.

Uçtan uca experiment smoke sonucu: bir training seed, iki held-out episode,
baseline'lar, PPO checkpoint, evaluation ve summary başarıyla üretildi;
8-timestep politika beklendiği gibi kalite kapısını geçmedi.

Final kod doğrulaması:

- `83` test ve `10` subtest başarılı.
- Ruff lint/format başarılı; mypy `36` üretim dosyasında hatasız.
- Compile ve `pip check` başarılı.

## 10 Eylül 2026 — Faz 6: PPO eğitim, evaluation ve replay altyapısı

### Amaç

Engelsiz/ideal dağılımda headless PPO eğitimi, çoklu environment, checkpoint,
resume, loglama, deterministic evaluation ve sonradan viewer incelemesi için
tekrarlanabilir bir uçtan uca hat kurmak.

### Değişiklikler

- `src/ebabil/configs/training/ppo_baseline.toml` tipli başlangıç
  hiperparametreleriyle eklendi.
- `src/ebabil/training/` config, `DummyVecEnv`/`SubprocVecEnv` fabrikası ve PPO
  runner katmanıyla eklendi.
- `scripts/train_ppo.py` smoke, checkpoint, best-model ve resume CLI'sı olarak
  eklendi.
- `src/ebabil/evaluation/` deterministic metrik ve güvenli NPZ/JSON replay
  sözleşmesiyle eklendi.
- `scripts/evaluate_ppo.py` ve `scripts/view_policy_replay.py` eklendi.
- Environment'a varsayılan kapalı, evaluation sırasında açılan 500 Hz fizik
  izi eklendi.
- TensorBoard 2.21.0 `.venv`, `pyproject.toml` ve `requirements.lock` içine
  sabitlendi.
- Tasarım ve devir teslim ayrıntıları `docs/PHASE_6_PPO_BASELINE.md` ve
  ADR-009 içinde kaydedildi.

### Kararlar

- Eğitim varsayılanı `clear + ideal`; avoidance Faz 7'de curriculum ile gelir.
- Üretim default'u dört `SubprocVecEnv`, test/smoke default'u `DummyVecEnv`.
- Observation normalization kapalı, reward normalization açık.
- Resume eski koşuyu değiştirmez; yeni boş dizine yazılır ve normalization açık
  ise eşleşen state dosyasını zorunlu tutar.
- Replay action/reward verisini 20 Hz, MuJoCo state/rotor/control verisini 500 Hz
  saklar.

### İş sırasında bulunan ve düzeltilen sorunlar

- SB3 `Monitor` runtime'da `Path` kabul etmiyordu; dosya sınırında `str`
  dönüşümü eklendi.
- Reward-only `VecNormalize` durumunda `obs_rms` bulunmuyordu; istatistik
  paylaşımı yalnız observation normalization açıkken yapılır hale getirildi.
- `VecNormalize.load/save` yol tipleri açıkça string'e çevrildi.
- İlk replay state kaydı 20 Hz idi ve ADR-005 ile çelişiyordu; environment
  düzeyinde 500 Hz fizik iziyle düzeltildi.
- Best model yolu, evaluation hiç tetiklenmezse artık var olmayan bir artifact
  gibi raporlanmaz; `null` döner.
- Best checkpoint ile eşleşen normalization state'i ilk uygulamada ayrıca
  saklanmıyordu; new-best callback'iyle `best/vecnormalize.pkl` eklendi.

### Aşamalı doğrulama

- Config/vector factory: `4/4` hedefli test + Ruff + mypy.
- Eğitim runner: `6/6` hedefli test; 64 adım CLI smoke ve yeni dizinde 64 adım
  resume başarılı.
- Evaluation/replay/viewer: deterministic byte-eşit iki koşu; NPZ/JSON
  round-trip; gerçek replay viewer dry-run başarılı.
- Physics trace düzeltmesi: `22` hedefli test ve mypy başarılı.
- Subprocess vektörleştirme: iki environment ile reset/observation testi
  başarılı.
- TensorBoard event üretimi ve `pip check` başarılı.

### Smoke sonucu nasıl yorumlanmalı

64 adımlık politika evaluation'da `%0` başarı verdi. Bu beklenen bir altyapı
smoke sonucudur ve PPO kalitesi hakkında iddia değildir. Kalite kapısı, tam
uzunluktaki en az üç bağımsız seed tamamlandıktan sonra değerlendirilecektir.

### Final regresyon

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
.venv/bin/python scripts/smoke_navigation_env.py --mode clear
```

Sonuçlar:

- `73` test ve `7` subtest başarılı.
- Ruff lint başarılı; `53` dosya formatlı.
- Mypy `32` üretim dosyasında hatasız.
- Compile ve dependency bütünlüğü başarılı.
- Bu kayıt dönemindeki eski clear baseline sonucu hold ekinden önceydi. Güncel
  regresyon: `118` adımda `goal`, final mesafe `0.072982 m`, 10 adım hold.
- İzole wheel kurulumunda PPO TOML, MJCF ve `EbabilNavigationEnv.reset`
  başarılı.
- İki süreçli 64 timestep PPO CLI smoke eğitimi başarılı.

### Bilinen sınırlılıklar

- Uzun eğitim ve `%95` held-out başarı kapısı henüz çalıştırılmadı.
- Random ve doğrudan-hedef baseline karşılaştırma runner'ı henüz eklenmedi.
- MP4 yerine tekrar üretilebilir state replay saklanır.
- GUI viewer otomatik testte açılmaz; `--dry-run` doğrulanır.

### Sonraki adım

- Önce tek uzun seed ile öğrenme eğrisini ve reward tasarımını kontrol etmek;
  ardından üç seed baseline matrisi çalıştırmak.

## 10 Eylül 2026 — Kod ve dokümantasyon review düzeltmeleri

### Amaç

PPO implementasyonundan önce paketlenebilirlik, seed bağımsızlığı, sahne rota
garantileri, Gym sınır doğrulamaları ve dokümantasyon sözleşmelerini incelemek;
bulunan her problemi hedefli testten sonra kapatmak ve bütün sistemi yeniden
regresyon testinden geçirmek.

### Review kaydı

- Bulgular, kabul ölçütleri, çözüm notları ve her fix'in doğrulaması
  `docs/REVIEW_FINDINGS.md` içinde tutulur.
- Git repository olmaması kullanıcı kararıyla problem kapsamından çıkarıldı.
- Toplam altı review bulgusunun tamamı kapatıldı.

### Uygulanan düzeltmeler

- MJCF ve varsayılan TOML dosyaları wheel'e dahil olmak üzere
  `src/ebabil/assets` ve `src/ebabil/configs` altına taşındı.
- Episode RNG'den bağımsız scene/sensor seed'leri türetildi; açık
  `scene_seed`/`sensor_seed` ve `info` telemetrisi eklendi.
- Doğrudan rota, araç açıklığıyla genişletilmiş 3B AABB üzerinde kesin slab
  kesişimine geçirildi; grid kenarları sürekli 2B geometriyle doğrulanıyor.
- Yetersiz blocker yüksekliği, geçersiz episode sınırı ve uyumsuz policy
  frekansı erken ve açık hata üretir hale getirildi.
- Observation v1'in ideal state/IMU + seçilebilir imperfect rangefinder
  sözleşmesi ADR ve tasarım belgelerinde kesinleştirildi.
- `pytest 9.1.1`, `ruff 0.16.7` ve `mypy 2.3.1` proje-yerel, sabitlenmiş
  geliştirme araçları olarak eklendi; README ve agent protokolü güncellendi.
- SB3 teknik dokümantasyon bağlantıları kurulu `2.9.0` sürümüne sabitlendi.

### Eklenen regression kapsamı

- Sabit sahnede farklı episode seed ile farklı sensör bozulması
- Açık component seed'leriyle birebir tekrar üretim
- Seed tipi/aralık doğrulaması
- Kesin 3B segment–AABB rota kontrolü
- Yetersiz blocker yüksekliği reddi
- Grid düğümleri arasında kalan engelin tespiti
- Episode step sınır tipi/aralığı ve constructor-time policy frekansı kontrolü

### Final doğrulama

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check src scripts tests
.venv/bin/ruff format --check src scripts tests
.venv/bin/mypy
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
.venv/bin/python scripts/run_hover_demo.py
.venv/bin/python scripts/smoke_navigation_env.py --mode clear
.venv/bin/python scripts/smoke_navigation_env.py --mode single_blocker
.venv/bin/python scripts/validate_navigation_env.py --steps 2000 \
  --mode random_obstacles --seed 20260910
```

Sonuçlar:

- `66` test ve `7` subtest başarılı.
- Ruff lint başarılı; `42` dosya formatlı.
- Mypy `25` üretim dosyasında hatasız.
- Compile ve bağımlılık kontrolü başarılı.
- İzole wheel kurulumundan model, tüm varsayılan config'ler ve Gym reset çalıştı.
- Hover final position error: `0.008606 m`.
- Clear yönlendirici: `109` adımda `goal`.
- Single blocker yönlendirici: `28` adımda beklenen `obstacle_collision`.
- Random rollout: `2000` finite adım, yaklaşık `160.34 policy step/s`.
- Statik viewer ve hover replay hazırlığı dry-run olarak başarılı.

### Bilinen sınırlılıklar

- Observation v1 state/IMU kanalları hâlâ ideal MuJoCo sensörleridir.
- Alternatif rota kabul testi 2B'dir; engel üzerinden uçuşu çözüm saymaz.
- MuJoCo sahnesi her reset'te yeniden derlenir.
- GUI viewer bu otomatik review turunda açılmadı.

### Sonraki adım

- Faz 6 engelsiz PPO eğitim/checkpoint/evaluation/replay hattı.

## 10 Eylül 2026 — Faz 5: Gymnasium navigasyon ortamı v1

### Amaç

Sahne, sensör, klasik kontrolcü, propulsion ve MuJoCo fiziğini Stable-Baselines3 ile uyumlu tek bir headless Gymnasium ortamında birleştirmek; action/observation, reward ve episode sonu sözleşmelerini PPO öncesinde sabitlemek.

### Değişiklikler

- `src/ebabil/configs/environment/navigation.toml` policy frekansı, action limitleri, observation ölçeği, termination eşikleri ve reward katsayılarıyla eklendi.
- `src/ebabil/envs/config.py` tipli ortam konfigürasyonu ve doğrulamasıyla eklendi.
- `src/ebabil/envs/navigation.py` içinde `EbabilNavigationEnv` uygulandı.
- `src/ebabil/envs/__init__.py` ile `EbabilNavigation-v0` Gymnasium kaydı eklendi.
- `src/ebabil/rewards/navigation.py` sekiz ayrı reward bileşeniyle eklendi.
- `scripts/smoke_navigation_env.py` scripted-policy smoke koşusu olarak eklendi.
- `scripts/validate_navigation_env.py` uzun random-action sayısal kararlılık kontrolü olarak eklendi.
- `tests/test_navigation_env.py` içinde dokuz environment testi eklendi.
- `docs/decisions/ADR-008-gymnasium-navigation-contract-v1.md` ile sözleşme sabitlendi.

### Action ve zaman sözleşmesi

```text
action = [vx_heading, vy_heading, vz_world, yaw_rate]
Box(-1, 1, shape=(4,), float32)
```

- Yatay hız limiti: `±1.5 m/s`
- Dikey hız limiti: `±1.0 m/s`
- Yaw-rate limiti: `±1.5 rad/s`
- Policy: `20 Hz`
- MuJoCo/kontrol: `500 Hz`
- Her Gym adımı: `25` fizik/kontrol alt adımı
- Maksimum episode: `20 s / 400 adım`

### Observation v1

`float32`, shape `(30,)`, aralık `[-1,1]`:

```text
[goal_body_xyz, goal_distance,
 velocity_body_xyz, angular_velocity_body_xyz,
 gravity_direction_body_xyz, altitude,
 rangefinder_16]
```

### Reward ve episode sonu

- Reward bileşenleri: progress, goal, collision, failure, proximity, smoothness, attitude ve time.
- Bileşenlerin tamamı her adımda `info["reward_components"]` altında ayrı loglanır.
- Goal, collision, workspace dışı, minimum irtifa, unsafe attitude ve sayısal hata `terminated` üretir.
- Yalnızca episode zaman sınırı `truncated` üretir.
- `scene_seed`, `scene_mode`, `sensor_mode`, mesafe, minimum range ve termination nedeni `info` içinde raporlanır.

### Fonksiyonel sonuçlar

Headless scripted-policy, `clear`, seed `20260910`:

- Hedefe ulaşma: başarılı
- Policy adımı: `109`
- Final hedef mesafesi: `0.188445 m`
- Episode return: `37.946889`
- Sonuç: `terminated=True`, neden `goal`

Aynı doğrudan hedef yönlendiricisi, `single_blocker`:

- Policy adımı: `28`
- Final hedef mesafesi: `2.388200 m`
- Episode return: `-12.292293`
- Sonuç: `terminated=True`, neden `obstacle_collision`

Bu karşılaştırma blocker'ın kaçınmayı zorunlu kıldığını; henüz kaçınma politikası bulunmadığını gösterir.

### Uzun random-action doğrulaması

```bash
.venv/bin/python scripts/validate_navigation_env.py \
  --steps 2000 --mode random_obstacles --seed 20260910
```

- `2000` policy step
- `10` tamamlanan episode
- Bitişler: `5 obstacle_collision`, `3 out_of_bounds`, `2 time_limit`
- Tüm observation ve reward değerleri sonlu
- Ölçülen hız: yaklaşık `165.12 policy step/s` tek süreç

### Otomatik doğrulama

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

Sonuç: `58/58` test başarılı; SB3 `check_env`, Gym registration, seed determinizmi, action clipping/validation, reward toplamı, terminated/truncated ayrımı, goal/collision ve random rollout kontrolleri geçti.

### Bilinen sınırlılıklar

- Ortam yalnızca headless'tır; eğitim episode replay kaydı Faz 6 evaluation hattında eklenecektir.
- Baseline varsayılanı `scene_mode="clear"`, `sensor_mode="ideal"` değerleridir.
- State/IMU ideal, rüzgâr sıfır ve fizik parametreleri nominaldir.
- MuJoCo sahnesi her reset'te yeniden derlenir; eğitim throughput'u gerekirse optimize edilecektir.
- Reward katsayıları başlangıç değerleridir; PPO davranışı ve ablation olmadan nihai kabul edilmez.

### Sonraki adım

- Faz 6: PPO eğitim script'i, vectorized environment, checkpoint, deterministic evaluation ve viewer replay kaydı.

## 10 Eylül 2026 — Kaçınmayı zorunlu kılan curriculum sahneleri

### Amaç

Kullanıcının viewer incelemesinde tespit ettiği açık doğrudan rota sorununu gidermek; engelsiz hedef takibi ile gerçek engelden kaçınma görevlerini ayırmak ve engelli sahnelerin çözülebilir kalmasını otomatik doğrulamak.

### Değişiklikler

- `src/ebabil/scenes/generator.py` artık `clear`, `single_blocker` ve `random_obstacles` modlarını üretir.
- `single_blocker` doğrudan başlangıç-hedef segmentini bir engelle zorunlu olarak kapatır.
- `random_obstacles`, aynı zorunlu blocker ile toplam sekiz engel üretir.
- Araç açıklığıyla genişletilmiş occupancy grid üzerinde alternatif rota kontrolü eklendi; dolu köşelerden diyagonal geçiş yasaktır.
- `src/ebabil/configs/environment/static_scene.toml` içine `vehicle_clearance_m=0.25`, `path_grid_resolution_m=0.10` ve sahne yeniden örnekleme sınırı eklendi.
- `scripts/view_static_scene.py` için `--mode` argümanı eklendi ve varsayılan `single_blocker` yapıldı.
- Sahne ve viewer regression testleri yeni sözleşmeye geçirildi.
- `docs/decisions/ADR-007-curriculum-scene-modes-and-path-validation.md` oluşturuldu.

### Doğrulama

```bash
.venv/bin/python scripts/view_static_scene.py --seed 20260910 --mode single_blocker --dry-run
.venv/bin/python scripts/view_static_scene.py --seed 20260910 --mode random_obstacles --dry-run
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

Sonuç:

- `single_blocker`: `1` engel, öndeki ilk hit `1.855 m`, doğrudan rota kapalı, alternatif rota açık.
- `random_obstacles`: `8` engel, doğrudan rota kapalı, alternatif rota açık.
- Seed `0–24` arasındaki 25 rastgele sahnenin tamamı iki rota koşulunu sağladı.
- Toplam `49/49` otomatik test başarılı; derleme ve bağımlılık kontrolü başarılı.

### Bilinen sınırlılıklar

- Çözülebilirlik kontrolü 2B'dir ve engel üstünden uçuşu hesaba katmaz.
- Grid rotası yalnızca sahne kabul testidir; PPO politikasına ayrıcalıklı yol bilgisi verilmez.
- Başarı oranına göre otomatik curriculum seviye geçişi Faz 7'de uygulanacaktır.

### Viewer kullanımı

```bash
# Varsayılan: kaçınmayı zorunlu kılan tek engel
scripts/run_mjpython.sh scripts/view_static_scene.py --seed 20260910

# Yoğun rastgele sahne
scripts/run_mjpython.sh scripts/view_static_scene.py --mode random_obstacles --seed 20260910

# Engelsiz başlangıç seviyesi
scripts/run_mjpython.sh scripts/view_static_scene.py --mode clear
```

### Sonraki adım

- Faz 5 Gymnasium ortamında bu modları reset seçeneklerine bağlamak ve ilk reward/termination sözleşmesini uygulamak.

## 10 Eylül 2026 — Viewer `UnboundLocalError` düzeltmesi

### Sorun

Hem `view_static_scene.py` hem `view_hover_demo.py`, `mujoco.MjData` oluştururken aşağıdaki hatayı veriyordu:

```text
UnboundLocalError: cannot access local variable 'mujoco'
```

### Kök neden ve düzeltme

Fonksiyon içindeki `import mujoco.viewer` ifadesi Python scope kuralları nedeniyle `mujoco` adını bütün fonksiyon boyunca yerel değişken kabul ettiriyordu. Import satırından önceki `mujoco.MjData(...)` bu nedenle erişilemeyen yerel değişkene başvuruyordu.

- İki scriptte import, yalnızca `mj_viewer` adını bağlayan `from mujoco import viewer as mj_viewer` biçimine geçirildi.
- Her iki viewer scriptine GUI açmadan model/replay hazırlığını çalıştıran `--dry-run` seçeneği eklendi.
- `tests/test_viewer_scripts.py` içinde iki subprocess regression testi eklendi; bunlar gerçek script giriş noktalarını çalıştırarak aynı scope hatasının geri gelmesini önler.

### Doğrulama

```bash
.venv/bin/python scripts/view_static_scene.py --seed 20260910 --dry-run
.venv/bin/python scripts/view_hover_demo.py --duration 0.01 --dry-run
sh -n scripts/run_mjpython.sh
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

Sonuç: iki dry-run başarılı; toplam `47/47` test başarılı; wrapper sözdizimi, Python derlemesi ve bağımlılık kontrolü başarılı.

### Kullanıcı doğrulama komutları

```bash
scripts/run_mjpython.sh scripts/view_static_scene.py --seed 20260910
scripts/run_mjpython.sh scripts/view_hover_demo.py --loop
```

GUI penceresinin açılması yerel grafik oturumunda kullanıcı tarafından yeniden doğrulanacaktır.

## 10 Eylül 2026 — Faz 4: Headless/replay ayrımı, sensörler ve statik sahneler

### Amaç

Eğitimi rendering maliyetinden ayırmak; navigasyon ortamı için deterministik rangefinder ve sensör bozulma katmanını kurmak; seed ile tekrar üretilebilen, başlangıç-hedef güvenliği ve geçerli rota garantisi olan ilk statik engel sahnelerini oluşturmak.

### Değişiklikler

- `src/ebabil/simulation.py` replay için fizik adımı bazında `qpos/qvel` izi kaydedecek şekilde genişletildi.
- `scripts/view_hover_demo.py` headless hesaplanan manevrayı MuJoCo viewer'da oynatmak için eklendi.
- `scripts/view_static_scene.py` drone, engeller ve collision üretmeyen hedef marker'ını görüntülemek için eklendi.
- `scripts/run_mjpython.sh` boşluk içeren proje yolunda macOS `mjpython` çalıştırma problemi için eklendi.
- `src/ebabil/configs/environment/sensors.toml` 16 ışın, menzil ve sensör hata parametreleriyle oluşturuldu.
- `src/ebabil/sensors/config.py`, `rangefinder.py`, `noise.py` ve `suite.py` eklendi.
- `src/ebabil/configs/environment/static_scene.toml` workspace, başlangıç/hedef, engel boyutları ve açıklık sınırlarıyla oluşturuldu.
- `src/ebabil/scenes/generator.py` seed tabanlı kutu engel üretimi ve MuJoCo `MjSpec` model derlemesiyle eklendi.
- `tests/test_sensors.py` ve `tests/test_scenes.py` ile 12 yeni test eklendi.
- ADR-005 ve ADR-006 ile execution, replay, rangefinder ve sahne sözleşmeleri kaydedildi.

### Teknik kararlar

- Eğitim, smoke test ve toplu evaluation headless; görsel inceleme ayrı viewer replay sürecidir.
- Rangefinder ışın sırası body `+x` ile başlar ve body `xy` düzleminde saat yönünün tersine ilerler.
- Sadece `geom group=1` engeller ölçülür; no-hit maksimum menzil olarak kodlanır.
- Rangefinder bozulma sırası bias/noise, clipping, sample-and-hold/dropout ve delay'dir.
- Seed hem sahne hem sensör rastgeleliğinin tekrar üretilebilir kaynağıdır.
- İlk kolay sahnelerde hedefe düz yatay koridor geometrik olarak açık bırakılır.

### Nominal smoke sonucu

Seed `20260910`:

- Engel sayısı: `8`
- Derlenen toplam geom: `25`
- Mocap hedef sayısı: `1`
- Rangefinder ışını: `16`
- Başlangıç ölçüm aralığı: `1.554446–5.000000 m`

### Doğrulama

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
.venv/bin/python scripts/view_hover_demo.py --help
.venv/bin/python scripts/view_static_scene.py --help
```

Sonuç: toplam `45/45` test başarılı; derleme ve bağımlılık kontrolü başarılı; iki viewer CLI sözleşmesi headless olarak doğrulandı.

Yeni doğrulanan davranışlar:

- Analitik kutu yüzeyi mesafesi ve no-hit maksimum menzil kodlaması
- Body yaw ile ışınların doğru dünya yönüne dönüşmesi
- Sample-and-hold ve delay zamanlaması
- `%100` dropout altında son ölçümün korunması
- Aynı seed ile noise/bias/dropout dizisinin birebir tekrarı
- MJCF ideal state/IMU ve rangefinder'ın ortak reading sözleşmesi
- Aynı seed ile aynı, farklı seed ile farklı sahne
- Engellerin collision geometry olarak derlenmesi
- Başlangıç/hedef marker konumları
- Garantili doğrudan koridorda engel ray hit'i olmaması

### Viewer kullanımı

```bash
scripts/run_mjpython.sh scripts/view_hover_demo.py --loop
scripts/run_mjpython.sh scripts/view_static_scene.py --seed 20260910
```

GUI viewer bu agent koşusunda açılmadı; grafik oturumu gerektirmeyen replay verisi ve CLI yolları doğrulandı. Viewer, eğitim veya otomatik test bağımlılığı değildir.

### Bilinen sınırlılıklar

- İlk rangefinder yalnızca yatay planardır.
- Noise/bias/delay/dropout şu anda yalnızca rangefinder çıkışına bağlanmıştır; ideal state/IMU ayrı tutulur.
- Sahne primitive'i yalnızca axis-aligned kutudur ve ilk seviye düz rota garantilidir.
- Ray görselleştirmesi viewer'a henüz çizilmez.
- Viewer replay henüz PPO reward/termination metadata'sı taşımaz.

### Sonraki adım

- Faz 5: Gymnasium ortamı, yüksek seviye velocity/yaw-rate action adapter'ı, observation dizisi, reward ve termination sözleşmeleri.

## 10 Eylül 2026 — Faz 3: Kademeli klasik uçuş kontrolü

### Amaç

RL politikasından bağımsız olarak nominal quadrotoru konum, hız, tutum ve yaw hedeflerinde kararlı tutan; propulsion katmanına collective thrust ve body torque veren klasik kontrol zincirini kurmak.

### Değişiklikler

- `src/ebabil/configs/controller/cascaded.toml` ile position/velocity, attitude ve body-rate kazançları ile fiziksel limitler tek yerde tanımlandı.
- `src/ebabil/controllers/config.py` tipli TOML yükleyici ve değer doğrulamasıyla eklendi.
- `src/ebabil/controllers/state.py` simülatörden bağımsız `FlightState`/`FlightTarget` sözleşmelerini ve MuJoCo state adaptörünü ekledi.
- `src/ebabil/controllers/cascaded.py` position/velocity PD, SO(3) attitude P ve anti-windup içeren body-rate PI katmanlarını birleştirdi.
- `src/ebabil/simulation.py` deterministik headless kapalı çevrim koşusu, RPY/quaternion dönüşümü, state/control/rotor izleri ve özet metrikleriyle eklendi.
- `scripts/run_hover_demo.py` tek senaryolu hızlı kontrol komutu olarak eklendi.
- `scripts/validate_controller.py` seed'li çoklu başlangıç kabul testi olarak eklendi.
- `tests/test_controllers.py` ve `tests/test_closed_loop.py` ile dokuz yeni test eklendi.
- `docs/decisions/ADR-004-cascaded-flight-controller.md` ile kontrol mimarisi ve sınırlılıkları kaydedildi.

### Teknik kararlar

- Dış döngü dünya-frame position/velocity PD ile desired force ve desired rotation üretir.
- Tutum hatası Euler açıları yerine SO(3) rotation matrix hatasıyla hesaplanır.
- Body-rate iç döngüsü atalet ölçekli PI ve gyroscopic compensation kullanır; torque saturation olan eksende integral dondurulur.
- Yatay/dikey ivme, tilt (`35°`), body-rate, torque ve collective thrust sınırları açıkça uygulanır.
- İlk baseline'da tüm kontrol döngüleri MuJoCo fizik adımıyla senkron `500 Hz` çalışır.
- Controller `data.ctrl` alanına doğrudan yazmaz; ADR-003 uyarınca `QuadrotorPropulsion` üzerinden mixer ve rotor gecikmesine bağlanır.

### Nominal sonuçlar

Birleşik `x=0.5 m`, roll `12°` bozucusu, `6 s`:

- Final konum hatası: `0.002439 m`
- Final hız: `0.005363 m/s`
- Final tutum hatası: `0.077757°`
- Mixer/tork doygunluğu: `%0 / %0`

Sabit `+x` yönünde `1 m/s` rüzgâr, `8 s`:

- Final konum hatası: `0.039987 m`
- Final hız: `0.000005 m/s`
- Final tutum hatası: `0.467209°`
- Mixer/tork doygunluğu: `%0 / %0`

### Monte Carlo kabul koşusu

Komut:

```bash
.venv/bin/python scripts/validate_controller.py --samples 100 --duration 6 --seed 20260910
```

Dağılım: `x,y ∈ [-0.5,0.5] m`, `z ∈ [0.8,1.2] m`, roll/pitch `[-15°,15°]`, yaw `[-30°,30°]`.

Sonuç:

- Başarı: `100/100`
- En kötü final konum hatası: `0.003535 m`
- En kötü final hız: `0.008660 m/s`
- En kötü final tutum hatası: `0.118509°`
- Minimum irtifa: `0.808897 m`
- En kötü mixer/tork doygunluk oranı: `0 / 0`

### Otomatik doğrulama

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src scripts tests
.venv/bin/python -m pip check
```

Sonuç: toplam `33/33` test başarılı; derleme başarılı; bozuk bağımlılık yok.

Yeni doğrulanan davranışlar:

- Hover thrust ve identity desired attitude
- Pozitif dünya `x` hedefi için doğru tilt yönü ve tilt limiti
- Roll hatası ve body-rate torque işaretleri
- Torque saturation altında integrator anti-windup
- Kusursuz başlangıçta durağan hover
- Birleşik konum ve roll/pitch bozucusundan toparlanma
- `45°` yaw referansı takibi
- `1 m/s` sabit rüzgâr altında sınırlı hover

### Bilinen sınırlılıklar

- Kontrolcü ideal ground-truth state kullanır; sensör bozulmaları ve estimator Faz 4 kapsamındadır.
- Position integral terimi yoktur; sabit rüzgârda küçük kalıcı konum hatası beklenir.
- Bütün döngüler şimdilik aynı `500 Hz` frekanstadır; gerçekçi multi-rate scheduler RL ortamıyla eklenecektir.
- Mixer saturation raporlanır ancak mixer kırpması body-rate integrator'a geri beslenmez.
- Dairesel/rastgele trajectory ve doygunluk sınırı dışındaki recovery senaryoları henüz regression kapısı değildir.
- Kazançlar yalnızca nominal model içindir ve gerçek uçuşa hazır kabul edilemez.

### Sonraki adım

- Faz 4: Sensör hata modeli, rangefinder/LiDAR ışınları ve deterministik statik sahne üretimi.

## 10 Eylül 2026 — Faz 2: Mixer, rotor dinamiği ve aerodinamik sürükleme

### Amaç

İstenen toplam thrust ve gövde momentini fiziksel rotor thrust'larına dağıtmak; thrust komutlarına motor gecikmesi eklemek ve hava-relative gövde sürüklemesini MuJoCo'ya uygulamak.

### Değişiklikler

- `src/ebabil/configs/vehicle/quadrotor.toml` nominal araç, rotor ve aerodinamik parametrelerinin kaynağı olarak oluşturuldu.
- `src/ebabil/config.py` tipli TOML yükleme ve fiziksel tutarlılık doğrulamasıyla eklendi.
- `src/ebabil/dynamics/mixer.py` X-konfigürasyon allocation matrix ve saturation raporlamasıyla eklendi.
- `src/ebabil/dynamics/rotor.py` exact-discretized birinci dereceden rotor hız modeliyle eklendi.
- `src/ebabil/dynamics/propulsion.py` mixer, rotor modeli ve MuJoCo actuator girdisini birleştirdi.
- `src/ebabil/dynamics/aerodynamics.py` quadratic body drag ve dünya çerçevesinde rüzgâr arayüzüyle eklendi.
- `scripts/inspect_propulsion.py` hover ve step-response inceleme komutu olarak eklendi.
- `tests/test_propulsion.py` ve `tests/test_aerodynamics.py` eklendi.
- `docs/decisions/ADR-003-propulsion-and-drag.md` oluşturuldu.

### Model denklemleri

Rotor thrust'ı:

```text
T_i = k_f * omega_i²
```

Birinci dereceden rotor hızı exact discretization ile:

```text
omega_next = omega + (1 - exp(-dt/tau)) * (omega_target - omega)
```

Gövde sürüklemesi:

```text
F_drag_body = -c * abs(v_body - wind_body) * (v_body - wind_body)
```

Mixer wrench sırası:

```text
[collective_thrust, tau_x, tau_y, tau_z]
```

### Nominal propulsion sonuçları

- Hover thrust: `2.4516625 N/rotor`
- Hover rotor hızı: `≈607.265 rad/s`
- Maksimum rotor hızı: `950 rad/s`
- Thrust-to-weight oranı: `≈2.4473`
- Yükselme zaman sabiti: `0.03 s`
- Düşme zaman sabiti: `0.05 s`
- 0.1 saniyelik sıfırdan hover komutu sonunda rotor thrust'ı: `≈2.2799 N`
- 0.5 saniye sonunda nominal hover thrust'ına yakınsama

### Doğrulama

```bash
.venv/bin/python scripts/inspect_propulsion.py
.venv/bin/python -m unittest discover -s tests -v
```

Sonuç: toplam `24/24` test başarılı.

Yeni doğrulanan davranışlar:

- TOML parametrelerinin fiziksel iç tutarlılığı
- TOML ile MJCF kütle, atalet, actuator limiti ve yaw işareti eşleşmesi
- Mixer'ın saturation yokken istenen wrench'i geri üretmesi
- Hover'da eşit rotor dağılımı
- FLU roll/pitch/yaw mixer işaretleri
- Rotor thrust limitleri
- Monoton ve kararlı rotor step response
- Exact discretization'ın sabit hedefte zaman adımı tutarlılığı
- Rotor steady-state ile MuJoCo hover entegrasyonu
- Sürüklemenin relative hava hızına karşı koyması
- Dünya rüzgârının doğru gövde/world kuvvetine dönüşmesi
- Sürüklemenin MuJoCo'da yatay hızı azaltması

### Bilinen sınırlılıklar

- Mixer saturation şu anda rotorları bağımsız kırpar; wrench önceliklendirmeli constrained allocation henüz yoktur.
- Rotor thrust modeli ideal quadratic eğridir; voltage sag ve üretim farkı yoktur.
- Drag katsayıları nominaldir ve sistem tanımlamasından gelmemektedir.
- Ground effect, blade flapping, rotor wake ve gövdeye uygulanan aerodinamik momentler yoktur.
- `xfrc_applied` her fizik adımı başında çağıran katman tarafından temizlenmelidir.

### Sonraki adım

- Faz 3: Kademeli klasik uçuş kontrolü ve bozucudan hover'a dönüş testleri.

## 10 Eylül 2026 — Faz 1: Quadrotor MJCF modeli

### Amaç

Kontrol ve RL katmanlarından bağımsız olarak derlenebilen, görsel olarak incelenebilen ve temel kuvvet/moment davranışı otomatik test edilen ilk quadrotor modelini oluşturmak.

### Değişiklikler

- `src/ebabil/assets/mjcf/quadrotor.xml` oluşturuldu.
- `src/ebabil/models.py` model yükleme API'si eklendi.
- `src/ebabil/__init__.py` paket dışa aktarımları eklendi.
- `scripts/inspect_model.py` model inceleme komutu eklendi.
- `tests/test_quadrotor_model.py` içinde dokuz otomatik test oluşturuldu.
- `docs/decisions/ADR-002-initial-quadrotor-model.md` ile nominal model parametreleri ve sınırlılıkları kaydedildi.

### Nominal model

- Kütle: `1.0 kg`
- Diyagonal atalet: `[0.005, 0.005, 0.009] kg·m²`
- Rotor merkezlerinin gövde merkezine uzaklığı: `0.23 m`
- Maksimum rotor thrust'ı: `6 N`
- Toplam thrust-to-weight oranı: yaklaşık `2.45`
- Fizik timestep: `0.002 s` (`500 Hz`)
- Başlangıç yüksekliği: `1 m`
- İlk aktüatör anlamı: rotor başına Newton cinsinden doğrudan thrust

### Koordinat ve rotor sözleşmesi

- Gövde FLU: `+x` ileri, `+y` sol, `+z` yukarı.
- Rotor sırası: front-left, front-right, rear-right, rear-left.
- Yaw reaction moment işaretleri: `[+, -, +, -]`.
- Görsel propeller geometrilerine ek olarak araç footprint'ini temsil eden collision diskleri eklendi.

### Sensörler

- Dünya konumu
- Quaternion orientation
- Gövde doğrusal hızı
- Gövde açısal hızı/gyro
- Gövde specific force/accelerometer

Bu sensörler henüz ideal ve gecikmesizdir; noise, bias, sampling ve delay Faz 4 kapsamındadır.

### Doğrulama

Komutlar:

```bash
.venv/bin/python scripts/inspect_model.py
.venv/bin/python -m unittest discover -s tests -v
```

Sonuç: `9/9` test başarılı.

Doğrulanan davranışlar:

- MJCF derleme ve model boyutları
- Kütle, atalet ve yerçekimi
- FLU/X rotor yerleşimi ve `0.23 m` kol uzunluğu
- Serbest düşüş
- Zemine nüfuz etmeden kararlı temas; 3 saniye sonunda `z≈0.03496 m`, hız normu `≈6.3e-13`
- Eşit thrust altında analitik hover dengesi
- Diferansiyel thrust için roll, pitch ve yaw işaretleri
- Yaw momentlerinin eşit thrust altında iptali
- Sensör adları ve boyutları

### Test sırasında bulunan ve düzeltilen sorunlar

- Test kodundaki hatalı `mj_step_step` çağrısı `mj_step` olarak düzeltildi.
- XML'deki altı basamaklı rotor koordinatı nedeniyle kol uzunluğu kontrolü fiziksel anlamlı `1e-6 m` toleransa geçirildi.

### Bilinen sınırlılıklar

- Aktüatörler şu anda doğrudan thrust alır; motor RPM, thrust eğrisi ve gecikme henüz yoktur.
- Nominal parametreler genel amaçlı örnek quadrotordur; gerçek platform ölçümü değildir.
- Gövde sürüklemesi, propeller aerodinamiği, ground effect ve batarya modeli yoktur.
- Sensörler noise ve delay içermez.

### Sonraki adım

- Faz 2: Araç parametre konfigürasyonu, motor/rotor dinamiği ve mixer uygulaması.

## 10 Eylül 2026 — Ortak agent çalışma düzeni

### Amaç

Birden fazla agentın proje durumunu, teknik sözleşmeleri ve bir sonraki işi ek açıklama gerekmeden anlayabilmesi.

### Değişiklikler

- `AGENTS.md` ile çalışma, test, koordinat ve dokümantasyon protokolü tanımlandı.
- `README.md` ile proje giriş noktası, kurulum ve belge indeksi oluşturuldu.
- `docs/PROGRESS_LOG.md` yaşayan durum ve devir teslim kaydı olarak oluşturuldu.
- `docs/decisions/ADR-001-control-and-frames.md` ile kontrol seviyesi ve frame sözleşmeleri kayda alındı.

### Kararlar

- Her iş diliminde ilerleme günlüğü güncellenecek.
- Mimari sözleşme değişiklikleri ADR ile izlenecek.
- Gövde ekseni FLU ve rotor sırası sabit olacak.

### Doğrulama

- Belge bağlantıları ve dosya yolları proje yapısına göre kontrol edilecek.

### Sonraki adım

- Faz 1 quadrotor MJCF modelinin oluşturulması ve otomatik test edilmesi.

## 10 Eylül 2026 — Faz 0: Python ortamı ve MuJoCo kurulumu

### Amaç

Proje bağımlılıklarını global Python'dan izole etmek ve MuJoCo/RL çekirdek zincirini doğrulamak.

### Değişiklikler

- Proje kökünde Python 3.13.2 tabanlı `.venv` oluşturuldu.
- `pyproject.toml`, `requirements.lock` ve `.gitignore` oluşturuldu.
- Proje editable paket metadata'sı olarak kuruldu.

### Kurulan çekirdek sürümler

- MuJoCo 3.13.0
- Gymnasium 1.3.0
- Stable-Baselines3 2.9.0
- PyTorch 2.14.0
- NumPy 2.5.3

### Doğrulama

- Tüm çekirdek importlar başarılı.
- `pip check`: bozuk bağımlılık yok.
- 100 adımlık MuJoCo serbest düşüş smoke testi başarılı: `z=1.0` konumundan `z≈0.801838`.
- 64×64 offscreen render başarılı; çıktı `uint8`, değer aralığı `0–220`.
- macOS CoreGraphics erişimi sandbox içinde mümkün olmadığından render testi izinli grafik bağlamında yürütüldü.

### Bilinen sınırlılıklar

- Henüz test/lint geliştirme bağımlılıkları eklenmedi.
- Grafik viewer otomatik test kapsamına alınmadı.

### Sonraki adım

- Quadrotor MJCF modeli ve analitik model testleri.

## 10 Eylül 2026 — Planlama ve literatür

### Amaç

RL tabanlı quadrotor navigasyonunun mimarisini ve bilimsel dayanaklarını belirlemek.

### Değişiklikler

- `docs/PROJECT_DESIGN.md` oluşturuldu.
- `docs/LITERATURE_REVIEW.md` oluşturuldu.
- `docs/IMPLEMENTATION_PLAN.md` oluşturuldu.

### Ana kararlar

- Ana algoritma PPO.
- Politika ilk aşamada gövde hızları ve yaw-rate hedefi üretecek.
- Alt seviye klasik uçuş kontrolcüsü kullanılacak.
- İlk dış algı LiDAR/rangefinder ışınları olacak.
- Curriculum ve seçici domain randomization kullanılacak.
- Gerçek uçuş öncesi bağımsız safety filter tasarlanacak.
