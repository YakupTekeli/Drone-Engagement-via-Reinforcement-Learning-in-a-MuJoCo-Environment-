# Ebabil RL Sistemi ve CNN Yapısı

**Tarih:** 23 Eylül 2026  
**Canlı kopya:** https://claude.ai/code/artifact/c97b1ae4-750d-45d9-887b-5587a1661e99  
Bu dosya proje kaydı için Markdown kopyasıdır; canlı belge yorum ve düzenleme içindir.

## Genel mimari

Sistem üç hızda çalışan üç katmandan oluşur: kamera ve kodlayıcı 30 Hz, öğrenilen politika 20 Hz, klasik uçuş kontrolcüsü ve fizik 500 Hz. Politika motorlara değil, kontrolcüye hız komutu verir; bu ayrım gerçek araçta Pixhawk'ın hızlı döngüyü üstlenmesine karşılık gelir.

```text
Kamera (320x240, 30 Hz) -> Kodlayıcı CNN (donmuş) -> Latent vektör (son 4 kare)
Telemetri (hız, eğim, irtifa) + Hedef yönü (yaklaşık, 2 m hata) + Latent -> PPO politikası (20 Hz)
-> Hız komutu (4 sayı) -> Kademeli kontrolcü (500 Hz) -> 4 motor itkisi -> MuJoCo fizik (2 ms adım) -> Kamera
```

Bugün iki sözleşme vardır. Yürürlükteki `vision-v2` etiketli encoder'ı kullanır ve engelli görevi öğrenememiştir. Kurulmakta olan `vision-v3` etiketsiz VAE kodlayıcısını kullanır; bu rapor ikisini de anlatır ve farkı işaretler.

## Simülasyon ortamı

Ortam MuJoCo 3.13 üzerinde 500 Hz fizikle çalışan, seed ile birebir tekrar üretilebilen bir sahnedir. Her episode yeni bir sahne üretir; başlangıç, hedef ve engeller rastgeledir fakat aynı seed aynı sahneyi verir.

**Drone.** X yerleşimli dört rotorlu, 1.0 kg kütleli bir quadrotor. Her rotor 0 ile 6 N arası itki üretir ve komuta birinci dereceden gecikmeyle uyar (yükselirken 30 ms, düşerken 50 ms). Gövde sürüklemesi hızın karesiyle modellenir. Klasik kademeli kontrolcü konum ve hız hatasından istenen eğimi, eğimden gövde dönüş hızını, oradan motor itkilerini hesaplar.

**Sahne.** Çalışma alanı 20 x 20 x 3 metredir. Başlangıç ve hedef, kenarlara 2.5 m pay bırakan bir kutuda örneklenir ve araları 16 ile 21 m'dir. Engeller 0.3 ile 0.9 m genişliğinde ve tavana kadar yüksektir; üstünden uçmak imkânsızdır. Üretici her sahnede 10 cm'lik ızgara üzerinde etrafından dolaşılabilir bir yol olduğunu doğrular. Hedef, zemindeki 2 x 2 m yeşil bir levha ve üzerindeki beyaz çarpı işaretidir.

**Kamera.** IMX477 modeli: 320 x 240 piksel, 75 derece yatay görüş açısı, gövdeye göre 40 derece aşağı bakış. Kamera aşağı baktığı için ileri görüş mesafesi irtifayla büyür; 1.42 m irtifada 8 m ileriyi görür. Görüntüler ekransız (headless) render edilir.

| Parametre | Değer |
| --- | --- |
| Fizik adımı | 2 ms (500 Hz), RK4 |
| Politika adımı | 50 ms (20 Hz), adım başına 25 fizik adımı |
| Kamera | 30 Hz, sample-and-hold |
| Episode bütçesi | Rota uzunluğuna göre 20 ile 60 s; 21 m rota için 43.7 s, yani 874 adım |
| Engel yoğunluğu merdiveni | 5, 10, 20, 40 engel |
| Zemin | 54 m genişlikte görsel düzlem; kamera hiçbir pozda boşluk görmez |

## Pekiştirmeli öğrenme ve PPO

Pekiştirmeli öğrenmede politika her adımda bir gözlem alır, bir eylem seçer, ortamdan bir ödül ve yeni gözlem döner; amaç, bir episode boyunca toplanan ödülü büyütmektir. Politika başta rastgeledir; binlerce episode'da hangi eylemlerin ödül getirdiğini istatistiksel olarak öğrenir.

**PPO (Proximal Policy Optimization).** İki ağ eğitilir. Aktör, gözlemden eylem dağılımı üretir; eğitimde bu dağılımdan örnekleme yapılır, değerlendirmede ortalaması alınır. Eleştirmen, bir gözlemden itibaren beklenen toplam ödülü tahmin eder; bu tahmin, bir eylemin beklenenden iyi mi kötü mü olduğunu ("avantaj") ölçmeye yarar. PPO'nun ayırt edici özelliği güncelleme adımını kırpmasıdır: yeni politika eskisinden en fazla yüzde 20 sapabilir.

**Bizim kurulum.** Dört simülasyon paralel koşar ve her biri 512 adım toplar; 2048 adımlık bu paket 256'lık parçalarla on kez geçilerek ağ güncellenir. Ödüller 0.99 iskonto ile toplanır. Politika ağı, gözlem sözlüğünü tek vektöre birleştirip iki gizli katmanlı (256 ve 256 nöron) bir MLP'den geçirir.

```
L_PPO = E_t[ min( r_t A_t, clip(r_t, 0.8, 1.2) A_t ) ],   r_t = pi_yeni(a_t|s_t) / pi_eski(a_t|s_t)
```

Toplam kayıp buna eleştirmenin değer hatasını (ağırlık 0.5) ekler; entropi bonusu kullanılmaz. Öğrenme yalnız politika içindir; görüntü kodlayıcısı PPO sırasında dondurulur.

## Observation Space

Politika ham görüntü görmez; her adımda altı anahtarlı bir sözlük alır ve her anahtar yalnız gerçek araçta üretilebilecek bir bilgiyi taşır. Simülatörün derinlik, segmentation, engel koordinatı veya gerçek hedef konumu hiçbir anahtarda bulunmaz; bu, otomatik testlerle zorlanır.

**Yürürlükteki sözleşme `vision-v2`, toplam 194 sayı.**

| Anahtar | Boyut | Aralık | İçerik | Gerçek araçtaki kaynağı |
| --- | --- | --- | --- | --- |
| fc_state | 10 | -1..1 | Gövde ekseninde hız (3, azami 1.5/1.5/1.0 m/s'e bölünmüş), açısal hız (3), yerçekimi yönü (3, eğimi anlatır), irtifa (0-3 m aralığına göre) | Pixhawk durum kestirimi |
| waypoint | 4 | -1..1 | İnanılan hedefe gövde ekseninde birim yön (3) ve logaritmik mesafe (30 m'ye göre) | Görev brifingi + görsel konumlama |
| timing | 2 | 0..1 | Kalan zaman bütçesi oranı; son kamera karesinin yaşı (0.10 s'ye göre) | Araç saati |
| vision_latent | 64 | -1..1 | Kodlayıcının görüntü özeti, 5'e bölünüp kırpılmış | Jetson üzerindeki CNN |
| vision_aux | 6 | -1..1 | Hedef görünürlük skoru, hedef merkezi x/y, sol/orta/sağ serbest alan (8 m'ye göre) | Aynı CNN'in başlıkları |
| obstacle_map | 108 | 0..1 | 9 x 12 engel ısı haritası; satırlar mesafeye, sütunlar yöne karşılık gelir | Aynı CNN'in ısı haritası |

**Kurulmakta olan sözleşme `vision-v3`, toplam 528 sayı.** fc_state, waypoint ve timing aynen kalır. vision_aux ve obstacle_map kalkar, çünkü etiketle eğitilmiş başlıklardır. vision_latent, etiketsiz VAE kodlayıcısının 128 boyutlu ortalama vektörünün son dört politika adımındaki değerlerini üst üste koyar (512 sayı).

**Hedef neden yaklaşık biliniyor.** GPS olmadığı için hedefin gerçek koordinatı verilemez. Her episode'da hedefin inanılan konumu, gerçek konuma en çok 2 m'lik sabit bir hata eklenerek üretilir; waypoint bu inanılan konumdan hesaplanır. Aracın kendi konumu hatasız kabul edilir.

**Zamanlama.** Kamera 30 Hz, politika 20 Hz çalışır; kodlayıcı yalnız yeni kare geldiğinde çalışır. Kare yaşı 0.10 s'yi aşarsa episode bayat kare nedeniyle iptal edilir.

## Action Space

Politika her adımda -1 ile 1 arasında dört sayı üretir; ortam bunları hız komutuna çevirir ve kademeli kontrolcü 500 Hz'de takip eder.

| İndeks | Anlam | Ölçek | Not |
| --- | --- | --- | --- |
| 0 | İleri / geri hız, gövde ekseni | 1.5 m/s | Negatif değer geri uçuş |
| 1 | Yanal hız, gövde ekseni | 1.5 m/s | Pozitif değer sola |
| 2 | Dikey hız, dünya ekseni | 1.0 m/s | Pozitif değer yukarı |
| 3 | Dönüş (yaw) hızı | 1.5 rad/s | Yaklaşık 86 derece/s |

Yatay hız büyüklük olarak sınırlanır: ilk iki bileşenin bileşkesi 1.5 m/s'yi aşarsa vektör küçültülür. Gövde eksenindeki yatay hız aracın dönüş açısıyla dünya eksenine çevrilir, bir konum referansına entegre edilir ve çalışma alanının içinde tutulur. Politika geçersiz sayı üretirse adım reddedilir; aralık dışı sonlu değerler kırpılır ve raporlanır.

İlk kamera sözleşmesi yalnız ileri, dikey ve dönüş içeriyordu; ölçüm, hedef üzerindeki 24 cm'lik halkada dönmeden ince düzeltmenin imkânsız olduğunu gösterdi ve ADR-020 ile eylem uzayı dört boyuta çıkarıldı.

## Reward fonksiyonu

Her adımın ödülü on sekiz ayrı bileşenin toplamıdır; her bileşen ayrı kaydedilir. Bileşenlerin bir kısmı simülatörün ayrıcalıklı bilgisini kullanır; bu serbesttir, çünkü ödül yalnız eğitimde vardır.

| Bileşen | Formül | Ağırlık | Aktif olduğu durumlar | Ayrıcalıklı mı |
| --- | --- | --- | --- | --- |
| target_progress | ağırlık x (önceki mesafe - şimdiki mesafe), gerçek hover noktasına 3B mesafe | 5.0 / m | hepsi | evet |
| progress_refund | fiziksel başarısızlıkla biten episode'da biriken ilerleme ödülü geri alınır | - | bitişte | evet |
| cruise_altitude | -ağırlık x max(0, 1.42 m - irtifa) | 0.05 / m | transit, search | hayır |
| bearing_alignment | -ağırlık x (kamera ile inanılan hedef arası açı) / 180 derece | 0.10 | search'ten hold'a | hayır |
| target_visible | hedef kadrajdayken sabit artı | +0.005 / adım | hepsi | v2 hayır, v3 evet |
| target_alignment | -ağırlık x (hedef merkezinin yatay kayması)², hedef görünürken | 0.20 | acquired, approach | v2 hayır, v3 evet |
| settle_speed | -ağırlık x max(0, hız - izin verilen hız)² | 0.50 | acquired'dan hold'a | evet |
| settle_altitude | -ağırlık x max(0, irtifa hatası - izin verilen pay)² | 0.96 | acquired'dan hold'a | evet |
| settle_tilt | -ağırlık x max(0, eğim - 10 derece)² | 11.4 | settle, hold | hayır |
| settle_angular_rate | -ağırlık x max(0, dönüş hızı - 0.35 rad/s)² | 1.22 | settle, hold | hayır |
| hover_hold | hold durumunda her adım sabit artı | +0.40 / adım | hold | evet |
| proximity | -ağırlık x ((0.75 m - en yakın engel) / 0.75)², 0.75 m içinde | 0.05 | hepsi | evet |
| smoothness | -ağırlık x ||eylem - önceki eylem||² | 0.02 | hepsi | hayır |
| collision | engel veya zemine çarpma, episode biter | -25 | bitişte | evet |
| out_of_bounds | çalışma alanından çıkma, episode biter | -15 | bitişte | evet |
| abort | hedef kaybı, bayat kare veya arama zaman aşımı | -10 | bitişte | kısmen |
| success | 3 saniyelik hold tamamlanır, episode biter | +30 | bitişte | evet |
| time | her adım sabit eksi | -0.01 / adım | hepsi | hayır |

İlerleme ödülü yoğun ana sinyaldir; 5 m'lik bir kısa yol 25 puan kazandırır ve bu tek çarpışmanın bedeline eşittir, yani engelin içinden geçmek hiçbir zaman kârlı değildir. Yerleşme cezaları yalnız sınırı aşan fazlalığın karesidir. İzin verilen hız payı hedefe yaklaştıkça doğrusal daralır: 2.5 m'de 1.5 m/s, 0.65 m'de 0.25 m/s.

## Terminal görev: hover durum makinesi

Başarı, hedefin güvenli iç bölgesinin üzerinde beş koşulu aynı anda 3 saniye boyunca kesintisiz sağlamaktır.

```text
transit -> search (inanılan hedefe 4 m) -> acquired (hedef 0.3 s görünür) -> approach (2.5 m)
        -> settle (güvenli bölge) -> hold (beş koşul) -> success (3 s)
abort: arama 12 s, hedef 2.5 s kayıp, bayat kare
```

| Koşul | Eşik |
| --- | --- |
| Yatay konum | Hedef merkezine en çok 0.65 m |
| İrtifa | 1.11 m, tolerans 0.20 m |
| Doğrusal hız | En çok 0.25 m/s |
| Eğim | En çok 10 derece |
| Dönüş hızı | En çok 0.35 rad/s |
| Süre | 60 ardışık politika adımı (3 s) |

Görsel kapı `vision-v2`'de encoder skoruyla (0.60 / 0.40 histerezis), `vision-v3`'te ortamın ayrıcalıklı kamerasından hesaplanan hedef piksel sayısıyla (en az 8) verilir. Kamera 40 derece aşağı baktığı için 1.11 m irtifada hold halkası 0.41 ile 0.65 m arasındaki 24 cm'dir.

## CNN yapısı

İki kodlayıcı vardır: yürürlükteki etiketli encoder ve onun yerine kurulan etiketsiz VAE. İkisi de 96 x 72 piksele küçültülmüş RGB kareyi alır; uçuşta yalnız kodlayıcı çalışır, CPU'da kare başına yaklaşık 0.5 ms.

**Ortak ön işleme.** Piksel değerleri 0-1 aralığına çekilir ve her renk kanalı ortalaması 0.25 olacak şekilde ölçeklenir (kazanç 0.5 ile 4 arası).

**Etiketli encoder (TinyVisionEncoder, v11).** Üç evrişim katmanı (12, 24, 32 kanal), 3 x 4 havuzlama, 64 boyutlu latent; başlıklar: hedef görünürlük skoru, hedef merkezi, sol/orta/sağ serbest alan, 9 x 12 hedef ve engel ısı haritaları, çarpışma riski. Simülatör etiketleriyle 6528 karede 45 epoch eğitildi. Ölçülen: serbest alan hatası 1.02 m, engel haritası IoU 0.66, hedef recall 0.92. Sorun: gerçek araçta bu etiketler yoktur.

**Etiketsiz VAE (VisualVAE, v1).** Dört evrişim katmanı (16, 32, 64, 64 kanal, 4 x 4 çekirdek, adım 2): 72 x 96 -> 36 x 48 -> 18 x 24 -> 9 x 12 -> 4 x 6. 1536 sayıdan iki doğrusal katman 128 boyutlu ortalama ve log-varyans üretir; kod çözücü aynı yolu tersine yürür.

```
L = sum_piksel (x_hat - x)^2 + beta * sum_j 0.5 (mu_j^2 + sigma_j^2 - 1 - log sigma_j^2),   beta = 1
```

KL ağırlığı ilk beş epoch'ta sıfırdan bire ısınır; karelerin yüzde 50'si gri tona çevrilir. Varsayılan kapalı iki kaldıraç: fotometrik bozulmadan temiz görüntü üretme ve kontrastif kayıp.

**Kalite ölçümü.** Dondurulmuş latent üzerinde yalnız ölçüm için doğrusal probe'lar eğitilir; ağırlıkları hiçbir yere gitmez.

| Kapı | Eşik | Ölçülen (3 seed) |
| --- | --- | --- |
| Etkin latent boyutu | en az 32 | 128 |
| Serbest alan hatası (0-1 ölçek, 8 m) | en çok 0.30 | 0.22 |
| Hedef AUROC | en az 0.80 | 0.81 ile 0.82 |
| Hedef average precision | en az 0.85 | 0.88 ile 0.90 |
| Engel haritası, en iyi eşikte IoU | en az 0.45 | 0.50 ile 0.54 |
| Engel haritası, sabit 0.5 eşiğinde IoU | en az 0.30 | 0.25 ile 0.40, karar bekliyor |

Uçuşta yalnız ortalama vektör alınır, -1..1 aralığına kırpılır ve son dört adımın vektörleri üst üste konur. Kodlayıcı PPO sırasında dondurulur.

## Eğitim ve değerlendirme düzeni

| PPO ayarı | Değer |
| --- | --- |
| Öğrenme hızı | 0.0003 |
| Paralel ortam | 4 |
| Rollout uzunluğu | ortam başına 512 adım |
| Mini-batch | 256 |
| Epoch / güncelleme | 10 |
| İskonto (gamma) | 0.99 |
| GAE lambda | 0.95 |
| Kırpma | 0.20 |
| Entropi katsayısı | 0 |
| Değer kaybı katsayısı | 0.5 |
| Politika ağı | 256 x 256 MLP |
| Normalizasyon | ödül açık, gözlem kapalı |
| Checkpoint | her 100 bin adım; her 50 bin adımda 10 episode iç değerlendirme |

Kabul ölçümü ayrı script'le, eğitimde görülmeyen seed'lerle, 40 episode ve üç eğitim seed'iyle yapılır; deterministik ve stokastik politika ayrı raporlanır. Engel sayısı 5, 10, 20, 40 basamaklarıyla artırılır. Kademe 1 kabulü: beş engelde varış en az yüzde 80, çarpışma en çok yüzde 10; hover başarısı raporlanır, kapı değildir.

## Kısa sözlük

| Terim | Anlamı |
| --- | --- |
| Aktör / eleştirmen | Eylemi seçen ağ / beklenen toplam ödülü tahmin eden ağ |
| Avantaj | Bir eylemin o durumdaki ortalama beklentiden ne kadar iyi olduğu |
| GAE | Avantajın gürültüsünü azaltan tahmin yöntemi; lambda 0.95 |
| İskonto (gamma) | Gelecekteki ödülün bugünkü değerini azaltan çarpan; 0.99 |
| Kırpma (clip) | PPO'nun politika değişimini yüzde 20 ile sınırlaması |
| Deterministik / stokastik politika | Dağılımın ortalamasını kullanmak / dağılımdan örneklemek |
| Latent | Kodlayıcının ürettiği küçük sayı vektörü |
| KL terimi | Latent dağılımını standart normale yakın tutan düzenleyici |
| Probe | Latent'in bilgi taşıyıp taşımadığını ölçen basit okuyucu |
| AUROC | Pozitif örneği negatiften üstün sıralama olasılığı |
| Average precision | Precision-recall eğrisinin altındaki alan |
| IoU | Tahmin ile gerçeğin kesişiminin birleşime oranı |
| Ayrıcalıklı (privileged) | Simülatörün bildiği, gerçek araçta olmayan bilgi |
| Sample-and-hold | Yeni kare gelene kadar son değeri tutmak |
| Warm start | Yeni eğitime önceki ağırlıklarla başlamak |
| Held-out | Eğitimde hiç kullanılmayan sahne veya seed kümesi |
