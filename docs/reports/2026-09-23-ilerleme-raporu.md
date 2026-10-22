# Ebabil MuJoCo İlerleme Raporu

**Tarih:** 23 Eylül 2026  
**Canlı kopya:** https://claude.ai/code/artifact/5f19f871-267d-45b2-a194-aba467579b2a  
Bu dosya proje kaydı için Markdown kopyasıdır; canlı belge yorum ve düzenleme içindir.

## Projenin amacı ve kapsamı

Ebabil, dört rotorlu bir drone'un GPS olmadan, yalnız tek bir kamera ve uçuş kontrolcüsü verisiyle engellerden kaçınıp hedef alanın üzerinde kararlı hover yapmayı öğrenmesini hedefleyen bir simülasyon araştırmasıdır. Öğrenme yöntemi pekiştirmeli öğrenmedir; drone'a engellerin yerleri hiçbir zaman verilmez, onları kameradan çıkarmak zorundadır.

**Görev.** Rastgele bir başlangıç noktasından 2 x 2 metrelik yeşil hedef alanına uçmak, alanın güvenli iç bölgesinin üzerinde 1.1 metre yükseklikte 3 saniye kesintisiz durmak. Hedefin yalnız 2 metreye kadar hatalı, yaklaşık konumu bilinir; gerçek konumu görüntüden bulunmalıdır. Çarpışma her koşulda başarısızlıktır.

**Hedef donanım.** IMX477 monoküler kamera, Pixhawk 2.4.8 uçuş kontrolcüsü ve Jetson Orin NX görev bilgisayarı. LiDAR, mesafe sensörü ve GPS bulunmaz; bu yüzden politikaya yalnız gerçek araçta üretilebilecek girdiler verilir.

**Kapsam sınırı.** Bütün çalışma MuJoCo fizik simülatöründedir. Gerçek uçuş, güvenlik onayı ve donanım entegrasyonu bu aşamanın dışındadır. Aracın kendi konumunu hatasız bildiği varsayılır; görsel konumlama sistemi modellenmez. Rüzgâr ve sensör kusurları şimdilik kapalıdır.

| Bilgi | Değer |
| --- | --- |
| Çalışma alanı | 20 x 20 x 3 m |
| Rota uzunluğu | 16 ile 21 m |
| Engel sayısı (merdiven) | 5, 10, 20, 40 |
| Hedef alan | 2 x 2 m, güvenli iç yarıçap 0.65 m |
| Hover yüksekliği | 1.1 m, tolerans 0.20 m |
| Hold süresi | 3 s |
| Azami yatay hız | 1.5 m/s |

## Terimler sözlüğü

| Terim | Anlamı |
| --- | --- |
| MuJoCo | Robotik için fizik simülatörü; drone'un kütlesi, motorları, çarpışmaları ve kamerası burada modellenir |
| Pekiştirmeli öğrenme (RL) | Ajanın deneme yanılmayla, ödül alarak davranış öğrendiği yöntem |
| Politika (policy) | Gözlemi eyleme çeviren sinir ağı; drone'un "beyni" |
| Episode | Bir başlangıçtan bitişe kadar süren tek deneme uçuşu |
| Observation (gözlem) | Her adımda politikaya verilen sayılar; kamera özeti, hız, hedef yönü |
| Action (eylem) | Politikanın ürettiği komut; burada dört hız değeri |
| Reward (ödül) | Her adımda davranışı puanlayan sayı; toplamı büyütmek öğrenmenin amacı |
| PPO | Kullandığımız RL algoritması; politikayı küçük ve güvenli adımlarla günceller |
| Gymnasium | RL ortamları için standart arayüz: reset ve step |
| Seed | Rastgeleliği sabitleyen sayı; aynı seed aynı sahneyi ve sonucu üretir |
| Checkpoint | Eğitimin belirli anında kaydedilmiş ağ ağırlıkları |
| CNN | Görüntü işleyen evrişimli sinir ağı |
| Encoder / kodlayıcı | Görüntüyü küçük bir sayı vektörüne sıkıştıran CNN |
| Latent (gizli vektör) | Kodlayıcının çıktısı; politikaya verilen görüntü özeti |
| VAE | Görüntüyü yeniden üretmeyi öğrenerek etiketsiz eğitilen kodlayıcı türü |
| Etiket / supervised | Simülatörün verdiği doğru cevapla eğitim; örneğin "bu piksel engel" |
| Privileged (ayrıcalıklı) bilgi | Simülatörün bildiği ama gerçek araçta olmayan bilgi; yalnız ödül ve ölçümde kullanılır |
| Domain randomization | Renk, ışık, doku gibi özellikleri her episode'da rastgele değiştirmek; genelleme için |
| Curriculum / merdiven | Görevi kolaydan zora kademeli sunmak |
| Held-out / OOD | Eğitimde görülmeyen sahne veya görsel koşul; genelleme ölçümü |
| Probe | Dondurulmuş latent üzerinde eğitilen basit okuyucu; latent'in bilgi taşıyıp taşımadığını ölçer |
| AUROC | Sıralama başarısı; 1.0 mükemmel ayrım, 0.5 rastgele |
| IoU | Tahmin edilen ve gerçek engel hücrelerinin örtüşme oranı |
| ADR | Mimari karar kaydı; sözleşme değiştiren her karar için yazılır |
| REV | Numaralı bulgu kaydı; ölçülmüş kusur ve düzeltmesi |
| Onay zinciri | Dataset, kalite ve audit raporlarının SHA-256 ile bağlandığı, yalnız geçen modelin yüklenebildiği düzen |
| Hover / hold | Hedef üzerinde yerinde, kararlı durma ve bunu kesintisiz sürdürme |

## Zaman çizelgesi

Proje 10 Eylül 2026'da başladı; on üç günde fizik, kontrol, algı ve öğrenme katmanlarının tamamı kuruldu, engelsiz görev öğrenildi, engelli görev henüz öğrenilmedi.

| Tarih | Kilometre taşı |
| --- | --- |
| 22-23 Eylül | Etiketsiz görsel temsil kararı (ADR-022); VAE kodlayıcı, probe ölçümü ve yeni onay zinciri kodlandı; kalite kapıları ölçüldü, Revizyon 1 uygulandı |
| 14 Eylül | Algı yenilendi: engel haritası kanalı (vision-v2), encoder v11, kamera açısı 45'ten 40 dereceye; beş engelli merdiven basamağında dört yapısal kusur bulunup düzeltildi |
| 13 Eylül | D1 spesifikasyonu düzeltildi: 40 engel, tavana kadar yükseklik; drone'un engellerin üstünden uçtuğu fark edildi |
| 12 Eylül | İlk geçerli engelli ölçüm; holonomik dört boyutlu eylem uzayı (ADR-020); görsel zemin düzlemi kusuru (REV-030); yaklaşık hedef konumu (ADR-019); ölçek merdiveni D1-D5 (ADR-018) |
| 11 Eylül | Kamera tabanlı ilk politika D0-V engelsiz sahnede 20/20 başarı; terminal hover durum makinesi (ADR-017); vision ortamı ve encoder onay zinciri |
| 10-11 Eylül | Faz 0-6: sanal ortam, quadrotor modeli, rotor dinamiği, kademeli kontrolcü, sensörler, Gymnasium ortamı ve engelsiz PPO pilotu (20/20) |

Kod tabanı bugün 302 otomatik test, 22 karar kaydı ve 49 numaralı bulgu kaydı içeriyor.

## Engelli ortam koşuları ve öğrenilen dersler

12 ile 14 Eylül arasında engelli ortamda on üç eğitim koşusu yapıldı ve hiçbiri D1 kapısını geçmedi. Tek engelli erken koşular yüzde 15 ile 42 arası başarı verdi, fakat bu sayılar kusurlu bir görsel zeminde ölçüldü. Beş ve daha fazla engelli bütün koşularda başarı sıfır kaldı.

| Bulgu | Ne bulundu | Düzeltme |
| --- | --- | --- |
| REV-030 | Görsel zemin yalnız 10 x 10 m'ydi; 20 m'lik alanda hedeflerin yüzde 65'i boşlukta görünüyordu | Zemin sahne boyutuna bağlandı |
| REV-035 | Engeller tavandan 0.6 m kısaydı; drone kaçınmak yerine üstünden uçtu | Engeller tavana kadar uzatıldı |
| REV-032 | Eylem uzayı yalnız ileri uçuşa izin veriyordu; hedef halkasında ince düzeltme imkânsızdı | Dört boyutlu holonomik eylem (ADR-020) |
| REV-036 | Alçak uçuşta kamera yalnız 3 m ileriyi görüyordu | Seyir irtifası ödül bileşeni |
| REV-037 | Encoder engel mesafesinde 3.07 m hata yapıyordu, koridorlar 2 m | D1 dataset'i ve engel haritası kanalı; hata 1.02 m'ye indi |
| REV-039 | Bir engelden doğrudan 40 engele geçildi; hiç başarı örneklenmedi | 5, 10, 20, 40 engel merdiveni |
| REV-040 | Hedef alan çalışma alanı kenarına çok yakın örekleniyordu | Kenar payı 2.5 m |
| REV-041 | Yanal uçuş açılınca kamera hedefe bakmayı bıraktı | Bakış hizalama ödülü |
| REV-042 | Yavaşlama cezası yalnız ulaşılamayan bölgede açıktı | Mesafeye göre azalan hız payı |
| REV-043 | Beş hover koşulu tek bir adımda bile birlikte sağlanmıyor | REV-046: irtifa, eğim ve dönüş hızı için üç yeni ödül bileşeni |
| REV-044 | Kamera 45 derece aşağı bakınca ileri görüş kısıtlıydı | Kamera 40 derece, hover yüksekliği 1.1 m; onay zinciri yenilendi |

**Dersler.** Eğitim metrikleri iki büyük kusuru gizledi: çarpışma oranı düşerken drone aslında engellerin üstünden uçuyordu. Bu ancak koşular izlenerek görüldü; artık her sonuç raporlanmadan en az bir episode tekrar oynatılıyor. İkinci ders, kapı eşiklerini sonuç kötü diye gevşetmemek; her kusur ölçülüp ayrı kayıt olarak kapatıldı.

## 22-23 Eylül çalışması: etiketsiz görsel temsil

Mevcut yapı reddedildi, çünkü görüntüyü özetleyen kodlayıcı simülatörün derinlik ve segmentation etiketleriyle eğitiliyordu; drone engelleri görmeyi değil, simülatörün cevabını kopyalamayı öğreniyordu. Yeni karar (ADR-022): politikaya giden görsel kanal etiket görmeden eğitilir; etiketler yalnız ölçüm ve eğitim ödülü için kullanılır.

**Kurulan yapı.** Yalnız kamera karelerini okuyan bir VAE kodlayıcı; trainer'ın etiket dosyalarını hiç açmadığı testle zorlanıyor. Kodlayıcının bilgi taşıyıp taşımadığı, dondurulmuş latent üzerinde eğitilen basit doğrusal okuyucularla ölçülüyor. Eski onay zinciri deseni korundu: dataset, üç seed kalite, tek seferlik audit, SHA-256 bağı, yalnız geçen modelin yüklenmesi.

**İlk ölçüm.** Üç seed'in hepsi engel ve serbest alan kapılarını geçti, fakat hedef alanının görünürlüğü için sabit eşikli recall kapısında takıldı. On bir farklı ayar tarandı: KL ağırlığı, latent boyutu, giriş çözünürlüğü, ışık ve renk değişimine dayanıklılık, kontrastif kayıp. Hiçbiri recall'u 0.61'in üstüne taşımadı. Teşhis: yeniden üretim kaybı kadrajın küçük bir kısmını kaplayan hedefi önemsemiyor; etiketli ağ bunu etiket sayesinde öğreniyordu.

**Revizyon 1.** Kullanıcı onayıyla hedef kapısı sabit eşik yerine eşikten bağımsız çözülebilirlik oldu; sabit eşik probe'un kalibrasyonunu ölçüyordu, latent'in bilgisini değil. Latent boyutu 64'ten 128'e çıkarıldı, çünkü kapasite bilgi taşımayı iyileştiren tek kaldıraçtı.

| Kapı | İlk koşu (latent 64) | Revizyon 1 koşusu (latent 128) | Eşik |
| --- | --- | --- | --- |
| Etkin latent boyutu | 64 / 64 | 128 / 128 | en az 32 |
| Serbest alan hatası (normalize) | 0.25 | 0.22 | en çok 0.30 |
| Hedef AUROC | 0.78 | 0.81 ile 0.82 | en az 0.80 |
| Hedef AP | 0.87 | 0.88 ile 0.90 | en az 0.85 |
| Engel haritası, en iyi eşikte IoU | 0.52 | 0.50 ile 0.54 | en az 0.45 |
| Engel haritası, sabit 0.5 eşiğinde IoU | 0.27 / 0.33 | 0.25 ile 0.40 | en az 0.30, iki seed geçmedi |

İkinci koşuda tek takılan kapı, korunan sabit 0.5 eşiğindeki harita IoU'sudur. Bu kapı da kaldırılan hedef kapısıyla aynı türden bir kalibrasyon ölçüsüdür; kaldırılıp kaldırılmayacağı bekleyen karardır.

## Mevcut durum

| Bileşen | Durum | Ölçüm |
| --- | --- | --- |
| Quadrotor modeli, rotor dinamiği, kademeli kontrolcü | Tamam | 100/100 seed'li hover doğrulaması |
| Sahne üretimi ve rota doğrulaması | Tamam | 20 x 20 m sahne 164 ms |
| Engelsiz PPO (rangefinder, legacy) | Tamam, donduruldu | Held-out 20/20 |
| Kamera tabanlı engelsiz politika D0-V | Tamam, tarihsel | 20/20, sıfır çarpışma |
| Etiketli encoder v11 | Tamam, artık yalnız karşılaştırma için | Serbest alan hatası 1.02 m, engel haritası IoU 0.66 |
| Engelli D1 politikası | Öğrenilmedi | Beş engelde başarı yüzde 0, çarpışma yüzde 12 ile 37 |
| Etiketsiz VAE kodlayıcı (Dilim 1) | Kod tamam; bir kapı kararı bekliyor | Hedef AUROC 0.81, en iyi eşikte harita IoU 0.52 |
| Yeni ortam vision-v3 (Dilim 2) | Başlamadı | - |
| Otomatik testler | Temiz | 302 test, lint ve tip denetimi temiz |
| Eğitim hızı | Ölçüldü | Kamera ortamında 135 adım/s; 600 bin adım yaklaşık 70 dk |

Donanım tarafında henüz hiçbir bileşen tezgâh testinden geçmedi; kamera lensi, uçuş kontrolcüsü varyantı ve görev bilgisayarı sürümü bekleyen girdilerdir.

## Açık sorunlar, riskler ve bekleyen kararlar

- **Bekleyen karar:** engel haritası için sabit 0.5 eşiğindeki IoU kapısının kaldırılması. Kaldırılırsa etiketsiz zincir üç seed'de geçer ve audit koşulur.
- **Keşif sorunu (REV-039):** 40 engelde politika hiç başarı örneklemiyor; merdiven 5 engelden başlıyor ve orada da hover başarısı sıfır.
- **Hover duvarı (REV-043):** beş hover koşulu tek bir adımda bile birlikte sağlanmıyor. Bu, algı yönteminden bağımsız bir sorundur; hover ölçütünün kendisi ADR-012 düzeyinde yeniden değerlendirilebilir, karar kullanıcıya aittir.
- **Etiketsiz algının sınırı:** VAE latent'i hedefi etiketli ağ kadar keskin ayırmaz (AUROC 0.81'e karşı etiketli recall 0.92). Hedef bulma politikaya kalıyor; gerekirse hedef için ayrı bir işaret dedektörü seçeneği açık.
- **Tek seed geleneği:** çoğu koşu tek seed'le yapıldı; kabul ölçütleri üç seed ister.
- **Sim-to-real boşluğu:** sensör gürültüsü, gecikme, rüzgâr ve dinamik randomization kapalı; gerçek kamera görüntüsüyle hiçbir doğrulama yapılmadı.
- **Donanım girdileri:** kamera lens ve sürücü uyumu, Pixhawk varyantı, Jetson yazılım sürümü, autopilot firmware seçimi bekliyor.
- **Hesaplama bütçesi:** tek Apple M3 üzerinde eğitim; 600 bin adım yaklaşık bir saat, üç seed'lik ablation'lar gün mertebesinde.

## Sonraki adımlar ve belge dizini

| Dilim | İçerik | Çıkış kapısı |
| --- | --- | --- |
| 1 (bitiyor) | Etiketsiz VAE, probe ölçümü, onay zinciri | Üç seed kalite ve audit geçer |
| 2 | Yeni ortam vision-v3: latent yığını, ortam içi görsel randomization, privileged görünürlük kapısı | Testler, 2000 adım rastgele rollout, kısa PPO smoke, hız en az eskinin 0.7 katı |
| 3 | Beş engelli sahnede eğitim, üç seed | Varış en az yüzde 80, çarpışma en çok yüzde 10 |
| 4 | Karşılaştırmalar: etiketli encoder, kodlayıcı ince ayarı, saf piksel, yığın derinliği, randomization, yakınlık ödülü | Aynı seed setinde üç seed |
| 5 | Klasik reaktif baseline ve engel yoğunluğu merdiveni | Basamak başına düşmeyen başarı, azalan çarpışma |

| Ne | Nerede |
| --- | --- |
| Yaşayan durum ve iş günlüğü | docs/PROGRESS_LOG.md |
| Kararlar | docs/decisions/ADR-001 ile ADR-022 |
| Bulgular | docs/REVIEW_FINDINGS.md, REV-001 ile REV-049 |
| Plan | docs/IMPLEMENTATION_PLAN.md, bölüm 23 |
| Tasarım ve literatür | docs/PROJECT_DESIGN.md, docs/LITERATURE_REVIEW.md |
| Etiketsiz kodlayıcı kodu | src/ebabil/perception/vae.py, probes.py, vae_quality.py |
| Ortam kodu | src/ebabil/envs/vision.py, navigation.py |
| Ödül ve durum makinesi | src/ebabil/rewards/terminal_hover.py, src/ebabil/missions/hover_state.py |
| Eğitim koşuları | runs/ (d1-*, ladder*, rev046*, vae_quality_v1 ve v2, vae_sweep_v1) |
| Çalışma protokolü | AGENTS.md |
