# ADR-011: Tek Kamera, Pixhawk ve Jetson Orin NX Mimarisi

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-006 ve ADR-008 içindeki rangefinder actor gözlemini gerçek araç hedefi için geçersiz kılar; bu sözleşmeler yalnız legacy simülasyon baseline'ı olarak korunur.

## Bağlam

Hedef fiziksel platformda LiDAR bulunmayacaktır. Donanım üç ana bileşenden
oluşacaktır:

- `IMX477 12MP Motorized Focus Camera for Jetson`; sensör ailesi kesin kabul,
  gerçek lens, carrier bağlantısı ve sürücü varyantı bench testinde kaydedilecek,
- Pixhawk 2.4.8 uçuş kontrolcüsü (FC),
- Waveshare Jetson Orin NX AI Kit 8GB görev/companion computer'ı (MC).

GPS/GNSS kullanılmayacağı kesin karardır. QGroundControl (QGC) geliştirme
bilgisayarında mevcuttur; ancak QGC bir autopilot değildir. Pixhawk kartına PX4
veya ArduPilot firmware'lerinden biri kurulur ve aynı anda yalnız biri çalışır.

Mevcut `EbabilNavigation-v0` actor observation'ı 16 planar rangefinder değeri
içerir. Bu girdiler gerçek donanımda üretilemeyeceği için bu ortamda öğrenilen
engelden kaçınma davranışı doğrudan gerçek araca taşınamaz. Ayrıca tek ileri
bakan kamera aracın arka ve yan çevresini aynı anda göremez; mevcut bağımsız
ileri/yan/geri hız serbestliği görüş alanı dışında harekete izin verir.

## Karar

### Donanım sorumlulukları

```text
IMX477 tek kamera
      |
      v
Jetson Orin NX: görüntü alma -> görsel temsil -> PPO/yerel planlama
      |                                      ^
      | MAVLink setpoint                     | Pixhawk state/health
      v                                      |
Pixhawk/autopilot: state estimation -> attitude/rate control -> mixer -> motorlar
```

- Jetson Orin NX görüntü işleme, yerel navigasyon, PPO inference, waypoint yönetimi
  ve politika güvenlik kontrollerini yürütür.
- Pixhawk üzerinde seçilecek autopilot hızlı attitude/body-rate kontrolü, mixer,
  temel failsafe ve sensör füzyonundan sorumludur. PX4 entegrasyon adayıdır;
  firmware henüz kesinleşmemiştir.
- Jetson, Pixhawk'a doğrudan motor komutu değil yüksek seviyeli ve sınırlandırılmış
  setpoint gönderir.
- Actor yalnız fiziksel araçta aynı anlamla üretilebilen kamera türevi veri,
  doğrulanmış Pixhawk state estimate'i ve görev/waypoint bilgisini kullanabilir.

### Kamera tabanlı actor

İlk üretim adayı doğrudan yüksek çözünürlüklü `RGB -> PPO` değildir. Modüler
hat kullanılacaktır:

```text
düşük çözünürlüklü/normalize görüntü + kısa zaman geçmişi
                      |
                      v
  hafif görsel encoder: serbest alan / çarpışma riski / latent temsil
                      |
                      v
   actor-deployable FC state + waypoint bilgisiyle PPO
```

- IMX477 renkli monoküler görüntüsüyle eşleşen MuJoCo RGB görüntüsü actor girdisi
  olabilir. Kamera depth üretmez.
- Simülasyon depth buffer'ı, segmentation, raycast ve tam state yalnız supervised
  etiket, privileged teacher/critic, reward, termination ve değerlendirme metriği
  olarak kullanılabilir.
- Bu ayrıcalıklı kanallar deploy edilecek actor tensörüne giremez. Observation
  sınırı otomatik leakage testiyle korunacaktır.
- Encoder önce simülasyon etiketleri veya privileged teacher ile eğitilebilir;
  PPO ile uçtan uca fine-tuning ayrı ablation olacaktır.
- Tek karede hız, örtülme ve derinlik belirsiz olduğu için frame history,
  recurrent encoder veya optical-flow benzeri zamansal bilgi ayrı deney olarak
  değerlendirilir.

### Görüş alanına uygun eylem

Yeni vision environment mevcut dört boyutlu action sözleşmesini otomatik olarak
devralmayacaktır. Kamera görüşü dışına kör yanal/geri hareketi engellemek için
ilk sürümde hareket kamera optik eksenine bağlanacaktır. Aday sözleşme:

```text
[forward_speed, vertical_speed, yaw_rate]
```

Yanal veya geri hareket ancak temporal yerel harita ve görünürlük temelli
güvenlik kısıtı doğrulandıktan sonra açılır. Kesin sınırlar kamera görüş alanı,
frenleme ivmesi, toplam algı-kontrol gecikmesi ve hedef seyir hızından
hesaplanacaktır.

### Ortam ve checkpoint sürümleme

- `EbabilNavigation-v0` ile observation v1 değiştirilmez; araştırma ve regression
  baseline'ı olarak dondurulur.
- Yeni görev geçici adıyla `EbabilVisionNavigation-v0`, observation sözleşmesi
  `vision-v1` olarak geliştirilir.
- Rangefinder tabanlı Faz 6'nın kalan çoklu-seed kalite koşuları ana geliştirme
  yolundan çıkarılır. Tamamlanmış tek-seed koşu PPO altyapı smoke/baseline kanıtı
  olarak saklanır; gerçek donanım veya görsel kaçınma başarısı sayılmaz.

### Görsel sim-to-real

Kamera eğitiminde en az aşağıdakiler seed'li ve ayrı loglanan randomization
eksenleri olacaktır:

- texture ve malzeme,
- aydınlatma, gölge ve pozlama,
- kamera intrinsics/extrinsics ve montaj toleransı,
- lens bozulması,
- motion blur ve rolling-shutter yaklaşımı,
- frame rate, capture-to-policy gecikmesi, jitter ve frame drop,
- görüntü gürültüsü, sıkıştırma ve renk değişimi.

MuJoCo görsel eğitimi başlangıç platformudur; yalnız sentetik başarı sim-to-real
kanıtı değildir. Gerçek kamera kayıtlarıyla perception değerlendirmesi ve daha
sonra kontrollü SITL/HITL gerekir.

İlk simülasyon implementasyonu actor render'ını `320 x 240`, nominal frame
rate'i `30 Hz`, yatay FOV'u `75 derece` ve kamera pitch'ini `45 derece` olarak
config'e almıştır. Projection sweep'inde `40/45/50 derece` adayları `6, 4, 2, 0
m` yaklaşma örneklerinde en az `%25` hedef köşe görünürlüğü sağladı. Bu yalnız
simülasyon başlangıç profilidir; fiziksel IMX477 lens ve montaj kalibrasyonunun
yerini tutmaz. Ayrıntılar `docs/PHASE_6_5_CAMERA_TARGET.md` içindedir.

### Kamera ve carrier board uyumluluğu

Verilen ürün adı Arducam `B0273` ise resmi ürün sayfası IMX477 sensör,
`4056 x 3040` sensor çözünürlüğü, motorized focus, `3.9 mm` lens ve `75 derece`
yatay FOV bildirir. Aynı sayfa sürücünün JetPack/kernel sürümüne bağlı olduğunu,
ürünün resmi NVIDIA developer kit için tasarlandığını ve üçüncü taraf carrier
board desteğinin garanti edilmediğini belirtir.

Waveshare Orin NX AI Kit üçüncü taraf `JETSON-ORIN-IO-BASE` carrier board
kullanır. Bu nedenle uygulamadan önce aşağıdaki bench gate zorunludur:

1. Kamera etiketi/SKU ve FFC pin/cable uyumluluğunu doğrulamak.
2. Waveshare kart üzerindeki CSI portunu ve lane düzenini doğrulamak.
3. Kurulu JetPack/L4T kernel için IMX477 sürücüsünü doğrulamak.
4. Kamera enumeration, seçilen çözünürlük/FPS ve en az 30 dakikalık frame-drop
   testini çalıştırmak.
5. Motorized focus I2C kontrolünü ve sabit-focus çalışma modunu test etmek.

VIO ve geometrik algı sırasında odak değişimi effective intrinsics'i ve görüntü
keskinliğini değiştirebilir. İlk uçuş profilinde focus kalkış öncesi kalibre
edilip kilitlenecek; uçuş sırasında autofocus ayrı ve ölçülmüş bir varyant
olmadan açılmayacaktır.

### Pixhawk 2.4.8 donanım yeterlilik kapısı

`Pixhawk 2.4.8` hedef FC olarak kesinleşmiştir. Bununla birlikte bu etiket tek
başına kartın üreticisini, gerçek FMU hedefini, kullanılabilir flash miktarını
ve sensör bileşimini kanıtlamaz. PX4, klasik Pixhawk 1/FMUv2 hedefinin 1 MB
flash sınırı nedeniyle bazı modülleri devre dışı bıraktığını ve Pixhawk 1'in
artık üretilmeyen bir tasarım olduğunu belirtir. ArduPilot ise `Pixhawk1` ve
`Pixhawk1-1M` imajlarını yayımlamakla birlikte bazı `2.4.x` etiketli kartlarda
barometre/IMU ikameleri görülebildiğini açıkça uyarır.

Bu nedenle firmware seçimi veya saha kullanımı öncesinde aşağıdaki bench gate
zorunludur:

1. Kartın üretici/PCB işaretleri ve bootloader/board ID çıktısını kaydetmek.
2. QGC'nin otomatik seçtiği PX4 hedefini ve bildirilen flash kapasitesini
   kaydetmek; elle `FMUv2`/`FMUv3` varsaymamak.
3. IMU, barometre, manyetometre ve SD kart durumunu envanterleyip pre-arm
   kontrollerini çalıştırmak.
4. Kullanılacak tek firmware için güncel kararlı sürüm, gerekli modüller,
   parameter seti ve kaynak kullanımını doğrulamak.
5. Jetson bağlantısında TELEM portu, voltage level, baud rate, MAVLink heartbeat,
   telemetry kaybı ve failsafe davranışını pervanesiz bench testinde ölçmek.

QGC geliştirme ve konfigürasyon aracı olarak kullanılabilir; PX4 ve ArduPilot
aynı kartta eşzamanlı çalışma mimarisi değildir. Firmware kararı ayrı bir ADR
ile sabitlenecektir.

### Jetson Orin NX yürütme bütçesi

Model boyutu ve görüntü çözünürlüğü masaüstü throughput'una göre sabitlenmez.
Jetson Orin NX 8GB üzerinde kamera capture, preprocessing, encoder, policy ve MAVLink
iletimini kapsayan uçtan uca gecikme benchmark'ı yapılır. Eğitim modeli ONNX'e
aktarılıp cihazın JetPack sürümüyle uyumlu TensorRT üzerinden optimize edilebilir;
FP16/INT8 seçimi doğruluk ve gecikme ölçümüyle belirlenir.

Carrier board, soğutma, NVMe, kamera ve güç dönüştürücü birlikte tartılacak;
ortalama/tepe güç tüketimi uçuş bataryası ve termal throttling bütçesine
katılacaktır.

## Konum kestirimi ve hedef tanımı

Pixhawk attitude/rate/IMU verisi tek başına kilometre ölçekli hedef alanın göreli
konumunu sağlamaz. Hedef koordinat tabanlıysa aşağıdakilerden biri gereklidir:

1. kamera + zaman senkronlu IMU ile VIO ve görsel loop closure,
2. önceden bilinen görsel harita üzerinde relocalization,
3. hedef alanın sürekli görülen ve tanımlanabilir bir görsel işaret olması.

PX4 seçilirse companion computer üzerinde çalışan VIO sisteminden external
pose/velocity alabilir. Ancak tek monoküler kamera, zaman senkronizasyonu ve
mutlak düzeltme olmadan kilometre boyunca sınırsız drift garantisi vermez.
Hedef dikdörtgen/kare bir alandır. GPS/GNSS bulunmadığı için nihai hedef
koordinatı actor'a ground-truth olarak verilemez. Alan başlangıçtan görünmüyorsa
önceden tanımlı görsel harita, takeoff-relative VIO rotası veya başka bir
deploy-edilebilir görsel referans gerekir. Hedef görülünce terminal davranış
ADR-012 kapsamındaki hedef bölge üzerinde güvenli hover'a geçer.

## Literatür ve teknik dayanak

- CAD2RL, sentetik monoküler görüntü ve yoğun domain randomization ile gerçek
  araçta collision avoidance aktarımının mümkün olduğunu gösterir:
  [RSS 2017](https://www.roboticsproceedings.org/rss13/p34.html).
- Learning High-Speed Flight in the Wild, doğrudan pixel-to-motor yerine
  privileged teacher ve kısa-horizon planlama yaklaşımını destekler:
  [Science Robotics 2021](https://doi.org/10.1126/scirobotics.abg5810).
- PX4 external vision/VIO akışı companion computer pose bilgisinin uçuş
  kontrolcüsüne verilmesini açıklar:
  [PX4 VIO dokümantasyonu](https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry).
- Jetson Orin NX 8GB kitinin GPU/memory ve güç seçenekleri Waveshare ürün
  sayfasında, cihaz
  inference optimizasyonu TensorRT dokümantasyonunda verilir:
  [Waveshare Jetson Orin NX AI Kit](https://www.waveshare.com/product/ai/boards-kits/jetson-orin/jetson-orin-nx-16g-dev-kit.htm),
  [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/).
- Bildirilen motorized-focus kamera adı Arducam `B0273` ile eşleşmektedir:
  [Arducam IMX477 B0273](https://www.arducam.com/arducam-12mp-imx477-motorized-focus-high-quality-camera-for-jetson.html).
- QGC'nin Pixhawk ailesine PX4 veya ArduPilot firmware yükleyebildiği resmi
  kılavuzda açıklanır:
  [QGC firmware yükleme](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/setup_view/firmware.html).
- PX4, FMUv2 hedeflerinin 1 MB flash sınırını ve Pixhawk 1'in discontinued
  durumunu açıklar:
  [PX4 Pixhawk Series](https://docs.px4.io/main/en/flight_controller/pixhawk_series).
- ArduPilot, `Pixhawk` etiketli kartlarda genellikle `Pixhawk1` hedefini önerir;
  `2.4.x` kartlardaki sensör ikamelerine karşı ayrıca uyarır:
  [ArduPilot firmware yükleme](https://ardupilot.org/copter/docs/common-loading-firmware-onto-pixhawk.html).
- Güncel Copter firmware ağacı `Pixhawk1` ve `Pixhawk1-1M` hedeflerini ayrı
  yayımlar:
  [ArduPilot Copter latest](https://firmware.ardupilot.org/Copter/latest/).

## Sonuçlar ve açık girdiler

- LiDAR kaçınma observation'ı üretim hattından çıkarılmıştır.
- Raycast/depth kullanımı yasaklanmamış, fakat yalnız privileged eğitim ve metrik
  tarafına sınırlandırılmıştır.
- Vision pipeline tamamlanmadan engelli uzun PPO eğitimi başlatılmayacaktır.
- IMX477 sensör ailesi kabul edilmiştir; gerçek lens/FOV, shutter, montaj açısı
  ve Waveshare carrier board sürücü uyumluluğu fiziksel donanımla doğrulanmalıdır.
- FC modeli Pixhawk 2.4.8'dir; gerçek board ID/FMU hedefi, flash ve sensör
  envanteri bench testinde kaydedilmelidir.
- Seçilecek tek autopilot firmware'i hâlâ gereklidir.
- Jetson Orin NX kit carrier board revizyonu, JetPack/L4T sürümü ve güç modu
  gereklidir.
- GPS/GNSS kullanılmayacaktır; uzun menzilli GPS-denied localization yöntemi
  henüz seçilmemiştir.
