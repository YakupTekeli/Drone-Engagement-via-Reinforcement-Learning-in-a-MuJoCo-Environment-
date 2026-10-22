# Ebabil MuJoCo: Literatür İncelemesi

**Tarama tarihi:** 11 Eylül 2026  
**Kapsam:** Quadrotor kontrolü, engelden kaçınma, PPO, curriculum learning, sim-to-real, politika yumuşaklığı, güvenlik filtreleri ve RL simülasyon altyapısı.

## 1. Yönetici özeti

İncelenen çalışmaların ortak sonucu, başarılı gerçek dünya RL uçuşunun yalnızca doğru algoritmayı seçmekten ibaret olmadığıdır. En belirleyici unsurlar şunlardır:

1. RL politikasının hangi kontrol seviyesinde çalıştığı
2. Politikaya gerçek araçta bulunabilir ve iyi temsil edilmiş gözlemler verilmesi
3. Çözülebilir görevlerden başlayan curriculum
4. Komut yumuşaklığının doğrudan optimize edilmesi
5. Sensör ve aktüatör gecikmelerinin modellenmesi
6. Körlemesine geniş randomizasyon yerine sistem tanımlama ve seçici randomizasyon
7. Eğitimden tamamen ayrılmış sahne ve bozulma dağılımlarında değerlendirme
8. Gerçek uçuşta RL'den bağımsız bir güvenlik katmanı

Bu bulgular, projenin ilk sürümünde PPO'nun yüksek seviyeli hız/yaw-rate komutları üretmesi ve alt seviyede klasik kontrolcü bulunması kararını desteklemektedir.

## 2. Çalışmaların ayrıntılı incelemesi

### 2.1 Proximal Policy Optimization Algorithms

**Kaynak:** Schulman, Wolski, Dhariwal, Radford ve Klimov, 2017.  
**Bağlantı:** [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)  
**Tür:** Temel algoritma makalesi.

PPO, politikanın tek bir güncellemede aşırı değişmesini clipped surrogate objective ile sınırlar. Uygulama kolaylığı, sürekli action space desteği ve paralel rollout toplamaya uygunluğu nedeniyle robotik kontrol için güçlü bir baseline olmuştur.

**Projeye çıkarım:**

- İlk ana algoritma PPO olacaktır.
- Sürekli eylemler `[-1, 1]` aralığında normalize edilecektir.
- PPO varsayılan hiperparametreleri nihai kabul edilmeyecek; rollout uzunluğu, batch size, entropy ve learning rate görev üzerinde doğrulanacaktır.
- Sonuçlar birden fazla seed ile raporlanacaktır.

Stable-Baselines3 2.9.0 rehberi de sürekli ve paralelleştirilebilir görevlerde PPO'yu önerir; observation/action normalization ile ayrı evaluation ortamını vurgular: [SB3 RL Tips and Tricks](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/rl_tips.html).

### 2.2 Sim-to-Real Deep Reinforcement Learning Based Obstacle Avoidance for UAVs Under Measurement Uncertainty

**Kaynak:** Joshi, Kapur ve Kandath, ilk preprint 2023; ICARA 2024.  
**Bağlantı:** [arXiv:2303.07243](https://arxiv.org/abs/2303.07243)  
**Tür:** Projemizin görevine en yakın PPO ve gerçek araç çalışması.

Çalışma, PPO ile sürekli durum/eylem uzayında waypoint navigasyonu ve engelden kaçınmayı ele alır. Her bölümde engel sayıları ve konumları değiştirilir; konum ve engel ölçümlerine farklı bias/variance seviyelerinde gürültü eklenir. Politika gerçek Crazyflie 2.1 üzerinde denenmiştir.

Önemli sonuçları:

- Gürültüsüz eğitim, ölçüm belirsizliği yükseldiğinde hızlı performans kaybı yaşayabilir.
- Eğitimde gürültü gören politika daha dayanıklı olabilir.
- Düşük geçiren veya Kalman filtresi unbiased gürültüye yardım eder; sabit bias için tek başına yeterli değildir.
- Yüksek seviyeli hız eylemleri politika aktarımını kolaylaştırır.

**Sınırlılık:** Kullanılan araç modeli 2B konum ve hız girişli birinci dereceden modeldir. Bu nedenle çalışma, tam 6-DoF uçuş dinamiği veya agresif manevralar için doğrulama sayılmaz.

**Projeye çıkarım:**

- Rastgele engel ve ölçüm belirsizliği deneyleri baseline olarak yeniden üretilecektir.
- Gürültü; variance, bias, dropout ve gecikme olarak ayrı test edilecektir.
- PPO yüksek seviyeli hız eylemleriyle başlatılacaktır.
- MuJoCo modelimiz bu çalışmadaki fizik modelinden daha ayrıntılı olacaktır.

### 2.3 What Matters in Learning a Zero-Shot Sim-to-Real RL Policy for Quadrotor Control? — SimpleFlight

**Kaynak:** Chen ve diğerleri, IEEE Robotics and Automation Letters, 2025.  
**Bağlantılar:** [arXiv:2412.11764](https://arxiv.org/abs/2412.11764), [IEEE Xplore](https://ieeexplore.ieee.org/document/11017653)  
**Tür:** PPO tabanlı kapsamlı ablation ve gerçek quadrotor doğrulaması.

Çalışma beş faktörü özellikle önemli bulur:

1. Actor girdisinde doğrusal hız ve rotation matrix
2. Critic girdisinde zaman bilgisi
3. Ardışık eylem farkının regularize edilmesi
4. Sistem tanımlama ile seçici domain randomization
5. Büyük batch size

Çalışmada SimpleFlight, Crazyflie üzerinde çeşitli trajectory tracking görevlerinde diğer RL baseline'larına kıyasla takip hatasını yarıdan fazla azaltmıştır.

**Projeye çıkarım:**

- Gövde doğrusal hızı gözlemde zorunlu olacaktır.
- Quaternion yerine veya yanında sürekli rotation representation kullanılacaktır.
- `||a_t-a_{t-1}||²` hem reward bileşeni hem metrik olacaktır.
- Randomizasyon aralıkları ölçüm ve sistem tanımlamayla gerekçelendirilecektir.
- PPO batch size küçük tutulmayacak; sim ve gerçekçilik sonuçları ayrı değerlendirilecektir.
- Önceki action'ın observation'a katılması bu çalışma ve başka çalışmalar arasında tutarlı sonuç vermediğinden ablation olacaktır.

### 2.4 Flying Through a Narrow Gap Using End-to-End Deep Reinforcement Learning Augmented With Curriculum Learning and Sim2Real

**Kaynak:** Xiao, Lu ve He, IEEE Transactions on Neural Networks and Learning Systems, 2021.  
**DOI:** [10.1109/TNNLS.2021.3107742](https://doi.org/10.1109/TNNLS.2021.3107742)  
**Açık sürüm:** [arXiv:2108.12869](https://arxiv.org/abs/2108.12869)

Çalışma eğimli dar bir açıklıktan agresif geçişi inceler. Curriculum boyunca açıklık boyutu kademeli küçültülür. Curriculum kaldırıldığında ajanın seyrek hedef ödülünü bulamadığı gösterilir. Gerçek sistemde dış döngü 50 Hz, Pixhawk attitude döngüsü 250 Hz'dir. Gerçek deney başarı oranı simülasyondaki sonuçla aynı mertebededir ancak yaklaşık yüzde 40 seviyesindedir; bu, zor görevlerde sim-to-real başarısının dahi mutlak güvenlik anlamına gelmediğini gösterir.

**Projeye çıkarım:**

- Curriculum zorunlu tasarım öğesidir.
- Dış navigasyon ile hızlı attitude kontrolü ayrılmalıdır.
- Başarı oranı yüksek görünse bile yüzlerce held-out episode ile ölçülmelidir.
- Dar geçit senaryosu temel görevden sonra ayrı curriculum seviyesi olmalıdır.

### 2.5 How to Train Your Quadrotor: RE+AL

**Kaynak:** Mysore, Mabsout, Saenko ve Mancuso, 2020/2021.  
**Bağlantı:** [arXiv:2012.06656](https://arxiv.org/abs/2012.06656)  
**Tür:** Düşük seviyeli quadrotor RL kontrolü ve gerçek donanım aktarımı.

RE+AL, takip hatası düşük olsa bile gürültülü ve titreşimli motor komutlarının fiziksel donanımda kararsızlık, ısınma ve enerji tüketimi oluşturabileceğini gösterir. Reward bileşenlerinin ölçek ve işaret çatışmaları da eğitim tekrarlanabilirliğini etkiler. Çalışma, gerçekçi ve sürekli değişen setpoint üretimini de önemli görür.

**Projeye çıkarım:**

- Episode reward yanında action spektrumu ve action-difference raporlanacaktır.
- Hareketsiz hedeflerin yanı sıra yumuşak değişen hedef komutlarıyla kontrolcü test edilecektir.
- Reward bileşenleri tek tek loglanacak ve ölçekleri incelenecektir.
- En iyi checkpoint'i elle seçmek yerine eğitim tekrarlanabilirliği ölçülecektir.

### 2.6 Regularizing Action Policies for Smooth Control with Reinforcement Learning — CAPS

**Kaynak:** Mysore ve diğerleri, 2020/2021.  
**Bağlantı:** [arXiv:2012.06644](https://arxiv.org/abs/2012.06644)

CAPS, politikanın yakın durumlarda yakın eylemler üretmesini teşvik eden regularization yaklaşımıdır. Gerçek quadrotor deneylerinde daha yumuşak kontrolün güç tüketimini önemli ölçüde düşürdüğü raporlanmıştır.

**Projeye çıkarım:**

- İlk baseline basit action-difference reward kullanacaktır.
- İleriki bir deneyde temporal ve spatial policy smoothness regularization uygulanacaktır.
- Enerji ve frekans-domain ölçümleri başarı oranından ayrı raporlanacaktır.

### 2.7 Learning to Fly in Seconds

**Kaynak:** Eschmann, Albani ve Loianno, IEEE Robotics and Automation Letters, 2024.  
**DOI:** [10.1109/LRA.2024.3396025](https://doi.org/10.1109/LRA.2024.3396025)  
**Açık sürüm:** [arXiv:2311.13081](https://arxiv.org/abs/2311.13081)  
**Kod:** [arplaboratory/learning-to-fly](https://github.com/arplaboratory/learning-to-fly)

Bu çalışma asymmetric actor-critic, action history, curriculum ve yüksek hızlı simülasyonla doğrudan motor RPM üreten bir politikanın gerçek Crazyflie'a aktarılabildiğini gösterir. Actor yalnızca gerçek sistemde bulunabilen bilgileri kullanırken critic eğitim sırasında ayrıcalıklı simülasyon bilgisi kullanabilir.

**Projeye çıkarım:**

- Asymmetric actor-critic, temel navigasyon tamamlandıktan sonra değerlendirilecektir.
- Action/observation history, gecikme ve kısmi gözlemlenebilirlik için deney değişkenidir.
- Doğrudan RPM kontrolünün mümkün olması, ilk sürüm için en doğru tercih olduğu anlamına gelmez; farklı kontrol seviyesi ve daha sıkı gerçek-zaman gereksinimi vardır.

### 2.8 Champion-Level Drone Racing Using Deep Reinforcement Learning — Swift

**Kaynak:** Kaufmann ve diğerleri, Nature, 2023.  
**DOI ve makale:** [10.1038/s41586-023-06419-4](https://www.nature.com/articles/s41586-023-06419-4)

Swift, görsel-inertial algıyı düşük boyutlu bir temsile dönüştüren algı sistemi ile simülasyonda eğitilen on-policy kontrol politikasını birleştirir. Simülasyon ve gerçek dünya arasındaki farkı azaltmak için az miktarda gerçek uçuş verisinden algı ve dinamik residual modelleri çıkarılmıştır.

**Projeye çıkarım:**

- Gerçek uçuş verisi elde edildiğinde gürültü modeli Gaussian varsayımla sınırlı kalmamalıdır.
- Perception failure ve dinamik residual ayrı modellenmelidir.
- Kamera fazında algı ile kontrol modüler tutulmalıdır.
- Saf simülasyon başarısı, gerçek uçuş iddiası için yeterli değildir.

### 2.9 Learning High-Speed Flight in the Wild

**Kaynak:** Loquercio ve diğerleri, Science Robotics, 2021.  
**DOI:** [10.1126/scirobotics.abg5810](https://doi.org/10.1126/scirobotics.abg5810)  
**Açık sürüm:** [arXiv:2110.05113](https://arxiv.org/abs/2110.05113)

Çalışma, karmaşık ortamlarda onboard sensör ve hesaplama kullanarak gürültülü gözlemleri collision-free kısa horizon trajectory'lere dönüştürür. Politika, simülasyonda ayrıcalıklı bilgiye sahip bir uzmandan öğrenir ve gerçekçi sensör bozulmalarıyla eğitilir.

**Projeye çıkarım:**

- Yüksek hızda yalnızca reaktif anlık komut yerine kısa horizon waypoint/trajectory çıktısı ileride değerlendirilebilir.
- Kamera girdisi kullanılacaksa doğrudan pixel-to-motor PPO ilk tercih olmamalıdır.
- Privileged teacher/student yaklaşımı, karmaşık algı fazı için güçlü bir alternatiftir.

### 2.10 CAD2RL: Real Single-Image Flight Without a Single Real Image

**Kaynak:** Sadeghi ve Levine, Robotics: Science and Systems, 2017.  
**DOI:** [10.15607/RSS.2017.XIII.034](https://doi.org/10.15607/RSS.2017.XIII.034)  
**Makale sayfası:** [RSS Proceedings](https://www.roboticsproceedings.org/rss13/p34.html)

CAD2RL, simüle monoküler görüntülerden velocity komutu üreten collision-avoidance politikasını yoğun görsel domain randomization ile gerçek iç mekâna aktarır.

**Projeye çıkarım:**

- Görsel gerçekçiliğin tek başına artırılması yerine texture, aydınlatma ve görünüş çeşitliliği önemlidir.
- Velocity-level çıktı sim-to-real açısından tekrar desteklenmektedir.
- Bu çalışma görsel algı transferi için önemlidir; 6-DoF motor dinamiği doğrulaması değildir.

### 2.11 Safe Quadrotor Navigation Using Composite Control Barrier Functions

**Kaynak:** Harms, Jacquet ve Alexis, IEEE ICRA, 2025.  
**DOI:** [10.1109/ICRA55743.2025.11127368](https://doi.org/10.1109/ICRA55743.2025.11127368)  
**Açık sürüm:** [arXiv:2502.04101](https://arxiv.org/abs/2502.04101)

Çalışma, çok sayıda LiDAR kaynaklı konum kısıtını tek Composite Control Barrier Function altında birleştiren güvenlik filtresi sunar. Yöntem iç ve dış mekânda gerçek quadrotor üzerinde, naive ve adversarial nominal komutlara karşı gösterilmiştir.

**Projeye çıkarım:**

- RL reward'una güvenlik garantisi gibi davranılmayacaktır.
- PPO eylemi gerçek uçuşta doğrudan uygulanmadan önce bağımsız safety filter'dan geçmelidir.
- Safety filter müdahale oranı ayrı bir değerlendirme metriği olmalıdır.
- İlk prototipte basit kinematik limit/override, ileri fazda CBF-QP uygulanabilir.

### 2.12 Safe and Efficient Reinforcement Learning Using Disturbance-Observer-Based Control Barrier Functions

**Kaynak:** Cheng, Zhao ve Hovakimyan, Learning for Dynamics and Control Conference, 2023.  
**Makale:** [PMLR 211](https://proceedings.mlr.press/v211/cheng23a.html)

Çalışma, model-free RL eylemini gerektiğinde minimum değişiklikle güvenli kümeye taşıyan disturbance-observer destekli CBF yaklaşımını 2B quadrotor dahil sistemlerde değerlendirir.

**Projeye çıkarım:**

- Politika ve güvenlik katmanının ayrılması pratik ve bilimsel olarak güçlü bir tasarımdır.
- Bozucu kuvvet gözlemcisi, rüzgâr ve model hataları arttığında ileri fazda değerlendirilebilir.
- 2B sonuçlar tam 3B quadrotora doğrudan genellenmiş kabul edilmemelidir.

### 2.13 Flightmare: A Flexible Quadrotor Simulator

**Kaynak:** Song ve diğerleri, PMLR/CoRL, 2021.  
**Makale:** [PMLR PDF](https://proceedings.mlr.press/v155/song21a/song21a.pdf)  
**Açık sürüm:** [arXiv:2009.00563](https://arxiv.org/abs/2009.00563)

Flightmare fiziği ve rendering'i ayırarak yüksek örnekleme hızı, paralel RL ortamları ve çoklu sensör desteği sağlar. Bu proje Flightmare kullanmayacaktır; çalışma simülatör tasarımı ve performans hedefleri için referanstır.

**Projeye çıkarım:**

- Headless fizik eğitimi ile görselleştirme ayrılmalıdır.
- Render sadece gerektiğinde açılmalıdır.
- Paralel ortam performansı en baştan benchmark edilmelidir.
- Kamera fazının eğitim maliyeti fizik-only fazdan ayrı ele alınmalıdır.

### 2.14 Curriculum Reinforcement Learning for Quadrotor Racing with Random Obstacles

**Kaynak:** Sun ve diğerleri, 2026.  
**Bağlantı:** [arXiv:2602.24030](https://arxiv.org/abs/2602.24030)  
**Kod:** [CRL-Drone-Racing](https://github.com/SJTU-ViSYS-team/CRL-Drone-Racing)  
**Durum:** Güncel çalışma; sonuçları temel ve daha uzun süredir doğrulanmış yayınlardan ayrı değerlendirilmelidir.

Çalışma random obstacle drone racing için multi-stage curriculum, domain randomization ve multi-scene update yaklaşımını birleştirir; HITL ve gerçek uçuş deneyleri raporlar.

**Projeye çıkarım:**

- Engel yoğunluğu, geçit zorluğu ve sahne çeşitliliği aynı anda değil aşamalı artırılmalıdır.
- Tek bir sahne ailesine ardışık uzun süre aşırı uyum yerine sahne havuzu dönüşümlü güncellenebilir.
- Açık kod, ileride curriculum tasarımını karşılaştırmak için incelenebilir.

### 2.15 Kamera, Pixhawk/QGC ve Jetson Orin NX teknik entegrasyonu

**Kaynaklar:** [PX4 Visual-Inertial Odometry](https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry),
[PX4 Companion Computers](https://docs.px4.io/main/en/companion_computer/),
[Arducam IMX477 B0273](https://www.arducam.com/arducam-12mp-imx477-motorized-focus-high-quality-camera-for-jetson.html),
[Waveshare Jetson Orin NX AI Kit](https://www.waveshare.com/product/ai/boards-kits/jetson-orin/jetson-orin-nx-16g-dev-kit.htm),
[QGC firmware yükleme](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/setup_view/firmware.html),
[PX4 Pixhawk Series](https://docs.px4.io/main/en/flight_controller/pixhawk_series),
[ArduPilot Pixhawk firmware yükleme](https://ardupilot.org/copter/docs/common-loading-firmware-onto-pixhawk.html),
[ArduPilot Copter latest firmware](https://firmware.ardupilot.org/Copter/latest/),
[NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/).

PX4, companion computer üzerinde üretilen external vision pose/velocity
bilgisinin uçuş kontrolcüsüne aktarılmasını destekler. Bu, kamera/VIO ve yüksek
seviyeli navigasyonun Jetson üzerinde; hızlı attitude/rate kontrolü ve temel
failsafe'in Pixhawk üzerinde tutulduğu mimariyle uyumludur.

Waveshare Jetson Orin NX AI Kit 8GB, Jetson Nano'dan farklı ve daha güçlü bir
Ampere tabanlı platformdur. Yine de görsel model masaüstü eğitim throughput'una
göre değil cihaz üzerinde capture-to-setpoint gecikmesiyle seçilmelidir.
TensorRT, ONNX tabanlı inference modelini
cihaz GPU'su için optimize edebilir; ancak gerçek hızlanma ve doğruluk kaybı
model, precision ve JetPack/TensorRT uyumluluğuyla ölçülmelidir.

Pixhawk 2.4.8 ürün etiketi, farklı üretici ve bileşen varyantlarında kullanılan
bir pazar adı olduğundan firmware hedefini tek başına belirlemek için yeterli
değildir. PX4 dokümantasyonu klasik FMUv2 hedefinde 1 MB flash sınırı nedeniyle
bazı modüllerin kapalı olabileceğini ve Pixhawk 1 tasarımının discontinued
olduğunu belirtir. ArduPilot güncel firmware ağacında `Pixhawk1` ile
`Pixhawk1-1M` hedeflerini ayrı yayımlar ve bazı `2.4.x` etiketli kartlardaki
barometre/IMU ikamelerine karşı uyarır. Sonuç olarak kart adı üzerinden FMU
varsaymak yerine bağlanan cihazın board ID, flash ve sensör envanteri ölçülür.

**Projeye çıkarım:**

- Kamera işleme ve PPO inference Jetson Orin NX'te, uçuş kontrolü Pixhawk'ta kalır;
  autopilot firmware'i ayrıca seçilir.
- FC modeli Pixhawk 2.4.8 olarak sabittir; gerçek firmware hedefi ve sensör
  yeterliliği pervanesiz bench testinde belirlenir.
- QGC hem PX4 hem ArduPilot firmware'ini Pixhawk ailesine yükleyebilir; QGC'nin
  kurulu olması iki firmware'in aynı anda kullanıldığı anlamına gelmez.
- Verilen motorized-focus kamera adı Arducam B0273 ile eşleşirse ürün sayfasındaki
  üçüncü taraf carrier board uyarısı nedeniyle Waveshare kartla sürücü testi
  zorunludur.
- VIO kullanılacaksa kamera-IMU timestamp ve frame dönüşümleri açıkça test edilir.
- GPS/GNSS olmayacağı için monoküler VIO drift'i görsel loop closure,
  relocalization veya önceden tanımlı görsel haritayla ayrıca sınanmalıdır.
- ONNX/TensorRT export parity ve gerçek cihaz latency testi kalite kapısıdır.

### 2.16 Etiketsiz görsel temsil öğrenimi ve hafif monoküler ağlar

**Eklenme:** 22 Eylül 2026; tetikleyici, kullanıcının sağladığı derin
araştırma raporu (GPS-denied monoküler seyrüsefer için CNN mimarileri).
Rapor projede yayımlanmadı; buradaki kaynaklar arXiv/RPG sayfalarından
doğrulandı. Raporun DroNet için verdiği `1710.02110` kimliği yanlıştı.

| Çalışma | Kaynak | Projeye çıkarım |
|---|---|---|
| Auto-Encoding Variational Bayes (Kingma ve Welling, 2013) | [arXiv 1312.6114](https://arxiv.org/abs/1312.6114) | Etiketsiz kodlayıcının temeli; yeniden üretim + KL kaybı (ADR-022 §2) |
| World Models (Ha ve Schmidhuber, 2018) | [arXiv 1803.10122](https://arxiv.org/abs/1803.10122) | Dondurulmuş VAE latent'i üzerinde küçük politika; iki aşamalı ayrım |
| Improving Sample Efficiency in Model-Free RL from Images (Yarats ve diğerleri, 2019) | [arXiv 1910.01741](https://arxiv.org/abs/1910.01741) | Yeniden üretim kaybının politika ile **birlikte** eğitildiğinde kararsızlaşabildiğini gösterir; kademe 1'de kodlayıcıyı dondurma gerekçesi |
| CURL (Srinivas ve diğerleri, 2020) | [arXiv 2004.04136](https://arxiv.org/abs/2004.04136) | Probe kapısı geçilmezse kontrastif yardımcı kayıp alternatifi |
| DroNet: Learning to Fly by Driving (Loquercio ve diğerleri, RA-L 2018) | [DOI 10.1109/LRA.2018.2795643](https://doi.org/10.1109/LRA.2018.2795643), [RPG sayfası](https://rpg.ifi.uzh.ch/dronet.html) | Çarpışma olasılığı başlığı ve `v = v_max (1 - P_coll)` hız modülasyonu; klasik baseline ve güvenlik filtresi adayı |
| FastDepth (Wofk ve diğerleri, 2019) | [arXiv 1903.03273](https://arxiv.org/abs/1903.03273) | Jetson TX2'de `178 FPS` monoküler derinlik; depth etiketiyle eğitildiği için actor'a giremez, asimetrik critic girdisi olabilir |

Raporun projeye taşınan diğer iki önerisi `k = 3-4` kare yığınlama ve
sıfırlama anında görsel domain randomization'dır (CAD2RL §2.10 ile uyumlu).
Raporun MJX paralellik iddiası kamera tabanlı eğitime uymaz: ölçülen darboğaz
fizik değil render ve encoder'dır. Optik akış/TTC yaklaşımı hover'da tanımsız
olduğu için yalnız tanı adayıdır.

### 2.17 Nesne tespiti ve sembolik algı arayüzü (dedektör + politika)

**Eklenme:** 26 Eylül 2026; tetikleyici, kullanıcının ADR-023 kararı (YOLO
dedektör çıktısının RL politikasına verilmesi). Kaynaklar arXiv/DOI ve
resmî belgelerden doğrulandı.

| Çalışma | Kaynak | Projeye çıkarım |
|---|---|---|
| You Only Look Once (Redmon ve diğerleri, CVPR 2016) | [arXiv 1506.02640](https://arxiv.org/abs/1506.02640) | Tek geçişli gerçek zamanlı tespit ailesinin temeli; kutu + sınıf + güven arayüzü |
| Ultralytics YOLO11 (belgeler) | [docs.ultralytics.com/models/yolo11](https://docs.ultralytics.com/models/yolo11/) | `n` ölçeği ve `imgsz 320` ile Jetson/TensorRT export; lisans AGPL-3.0 (ADR-023 §3 notu) |
| YOLOX (Ge ve diğerleri, 2021) | [arXiv 2107.08430](https://arxiv.org/abs/2107.08430) | Apache-2.0 lisanslı anchor-free alternatif; lisans itirazında aynı kutu arayüzüyle değiştirilebilir |
| DETRs Beat YOLOs on Real-time Object Detection, RT-DETR (Lv ve diğerleri, 2023) | [arXiv 2304.08069](https://arxiv.org/abs/2304.08069) | NMS'siz gerçek zamanlı alternatif; küçük modelde gecikme bütçesi ölçülmeli |
| Domain Randomization for Transferring Deep Neural Networks (Tobin ve diğerleri, IROS 2017) | [arXiv 1703.06907](https://arxiv.org/abs/1703.06907) | Sentetik görüntüyle eğitilen nesne dedektörünün gerçek kameraya aktarımı; ADR-022 §6 randomization'ının dedektör verisinde de kullanılma gerekçesi |
| Swift (Kaufmann ve diğerleri, Nature 2023; §2.8) | [DOI 10.1038/s41586-023-06419-4](https://doi.org/10.1038/s41586-023-06419-4) | Algı sistemi (kapı dedektörü) düşük boyutlu temsil üretir, politika o temsil üzerinde simülasyonda eğitilir; gerçek veriden ölçülen algı residual'ı eğitime enjekte edilir (ADR-023 §8 yedek yolunun öncülü) |

Projeye taşınan üç sonuç: (1) politika ile dedektör arasındaki arayüz
sembolik olduğunda dedektör gerçek görüntüyle yeniden eğitilse bile politika
sözleşmesi değişmez; (2) sentetik dedektör verisi görsel randomization ister;
(3) dedektör hataları politikaya ya döngü içinde (varsayılan, ADR-023 §1) ya
da ölçülmüş hata modeliyle (yedek, §8) gösterilmelidir, aksi halde politika
kusursuz algıya aşırı uyar.

## 3. Tematik sentez

### 3.1 PPO seçimi

PPO hem doğrudan quadrotor kontrolünde hem yüksek seviyeli navigasyonda yaygın ve doğrulanmış bir baseline'dır. Ancak literatür, algoritmayı değiştirmenin kötü tasarlanmış observation, reward veya fizik modelini telafi etmediğini gösterir. Bu nedenle ilk hedef PPO'yu “optimize etmek” değil, güvenilir bir deney platformunda doğru problemi tanımlamaktır.

### 3.2 Kontrol soyutlama seviyesi

Üç ana seçenek vardır:

| Seviye | Politika çıktısı | Avantaj | Risk |
|---|---|---|---|
| Yüksek | Velocity/yaw-rate | Kolay öğrenme, platformlar arası aktarım | Çok agresif manevra kapasitesi sınırlı |
| Orta | Total thrust + body rates | Daha çevik ve hâlâ yönetilebilir | Dinamik modele daha bağımlı |
| Düşük | Motor thrust/RPM | Tam uçtan uca kontrol | Yüksek frekans, zor sim-to-real, güvenlik riski |

İlk sürüm yüksek seviyede olacaktır. Orta ve düşük seviye, aynı benchmark üzerinde sonraki deneylerdir.

### 3.3 Gözlem uzayı

Tam simülatör durumu kullanmak hızlı bir prototip sağlar ancak sim-to-real için yanıltıcı olabilir. Gözlem tasarımında üç katman ayrılmalıdır:

- `privileged_state`: yalnızca eğitim/teşhis
- `estimated_state`: gerçek araçta durum kestiriminden gelebilecek veriler
- `exteroception`: hedef donanım için tek monoküler kamera

Actor yalnızca son iki katmana erişmelidir. Critic/teacher, supervised perception
etiketi ve evaluation için privileged depth/raycast kullanılabilir. Observation
v1 içindeki rangefinder ve ideal state kanalları legacy geçiş baseline'ıdır;
sim-to-real observation doğrulaması sayılmaz ve camera actor'a taşınamaz.

22 Eylül 2026 (ADR-022): `exteroception` katmanındaki kodlayıcı da
`privileged_state` etiketleriyle eğitilemez. Etiketler yalnız probe ölçümü,
privileged reward/termination ve tanıdır; actor kodlayıcısı etiketsiz (VAE)
öğrenir.

### 3.4 Reward ve yumuşaklık

Literatür, yalnızca hedefe ulaşma ve çarpışma cezasının fiziksel olarak iyi bir politika üretmediğini göstermektedir. Eylem değişimi, kontrol eforu ve takip kalitesi ayrı ölçülmelidir. Bununla birlikte çok sayıda çatışan ve farklı ölçekli reward bileşeni reward hacking ile kararsızlığa yol açabilir. Her bileşen normalize edilmeli ve ablation ile gerekçelendirilmelidir.

### 3.5 Curriculum

Curriculum şu boyutları bağımsız kontrol etmelidir:

- Başlangıç durumu dağılımı
- Hedef uzaklığı
- Engel sayısı ve yoğunluğu
- Minimum geçit genişliği
- Sensör bozulması
- Rüzgâr ve fizik belirsizliği
- Hız limiti

Bir seviyede yüksek reward yerine başarı, çarpışma ve minimum clearance birlikte geçiş ölçütü olmalıdır.

### 3.6 Sim-to-real

İki tamamlayıcı yöntem öne çıkmaktadır:

1. **Domain randomization:** Gerçek değerin eğitim dağılımı içinde kalmasını hedefler.
2. **Data-driven residual model:** Gerçek loglarda gözlenen sistematik dinamik ve algı hatalarını simülasyona geri ekler.

Proje ilk olarak sistem tanımlama ve seçici randomization kullanacak; gerçek uçuş logları elde edilirse residual model aşamasına geçecektir.

### 3.7 Güvenlik

Collision penalty öğrenmeyi yönlendirir fakat hard constraint sağlamaz. Gerçek sistemde politika çıktısı şu sıralamadan geçmelidir:

```text
PPO action
   → sayı/aralık doğrulama
   → hız ve attitude limitleri
   → obstacle safety filter
   → alt seviye kontrolcü
   → flight controller failsafe
```

## 4. Proje için önerilen ablation matrisi

| Deney | Baseline | Varyant | Ölçülecek etki |
|---|---|---|---|
| Curriculum | Kapalı | Açık | Öğrenme hızı, başarı, seed varyansı |
| Sensor realism | İdeal | Noise/bias/delay | Robustluk |
| Domain randomization | Yok | Seçici | Held-out fizik performansı |
| Smoothness | Yok | Action-difference | Enerji, jerk, başarı |
| Observation history | Yok | Frame/action history | Gecikmeye dayanıklılık |
| Rotation input | Quaternion | Gravity vector/rotation representation | Öğrenme ve genelleme |
| Safety filter | Kapalı | Açık | Çarpışma ve müdahale oranı |
| Control level | Velocity | Body-rate/thrust | Çeviklik ve aktarılabilirlik |
| Visual representation | Etiketsiz donmuş VAE | Etiketli encoder / PPO içinde ince ayar / saf piksel | Etiketin bedeli, sample-efficiency, latency |
| Temporal vision | Latent yığını `k = 4` | `k = 1` / recurrent | Kısmi gözlemlenebilirlik ve gecikme |
| Privileged teacher | Yok | Teacher/student | Görsel öğrenme hızı ve aktarım |

## 5. Kaynakların projede kullanım önceliği

### Birinci öncelik: doğrudan tasarımı belirleyenler

- [PPO](https://arxiv.org/abs/1707.06347)
- [SimpleFlight](https://arxiv.org/abs/2412.11764)
- [Sim-to-Real Obstacle Avoidance Under Measurement Uncertainty](https://arxiv.org/abs/2303.07243)
- [Flying Through a Narrow Gap](https://arxiv.org/abs/2108.12869)
- [RE+AL](https://arxiv.org/abs/2012.06656)
- [Composite CBF](https://arxiv.org/abs/2502.04101)
- [Learning High-Speed Flight in the Wild](https://arxiv.org/abs/2110.05113)
- [CAD2RL](https://www.roboticsproceedings.org/rss13/p34.html)
- [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114)
- [SAC+AE: Sample Efficiency from Images](https://arxiv.org/abs/1910.01741)

### İkinci öncelik: ileri fazlar

- [Learning to Fly in Seconds](https://arxiv.org/abs/2311.13081)
- [Swift](https://www.nature.com/articles/s41586-023-06419-4)
- [Flightmare](https://arxiv.org/abs/2009.00563)
- [CRL Drone Racing](https://arxiv.org/abs/2602.24030)
- [World Models](https://arxiv.org/abs/1803.10122)
- [CURL](https://arxiv.org/abs/2004.04136)
- [DroNet](https://doi.org/10.1109/LRA.2018.2795643)
- [FastDepth](https://arxiv.org/abs/1903.03273)

## 6. Resmî teknik dokümantasyon

- [MuJoCo dokümantasyonu](https://mujoco.readthedocs.io/en/stable/)
- [MuJoCo Python API](https://mujoco.readthedocs.io/en/stable/python.html)
- [MuJoCo XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html)
- [Gymnasium özel ortam geliştirme](https://gymnasium.farama.org/main/introduction/create_custom_env/)
- [Stable-Baselines3 2.9.0 PPO](https://stable-baselines3.readthedocs.io/en/v2.9.0/modules/ppo.html)
- [Stable-Baselines3 2.9.0 RL önerileri](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/rl_tips.html)
- [Stable-Baselines3 2.9.0 evaluation](https://stable-baselines3.readthedocs.io/en/v2.9.0/common/evaluation.html)
- [PX4 Visual-Inertial Odometry](https://docs.px4.io/main/en/computer_vision/visual_inertial_odometry)
- [PX4 Companion Computers](https://docs.px4.io/main/en/companion_computer/)
- [Arducam IMX477 B0273](https://www.arducam.com/arducam-12mp-imx477-motorized-focus-high-quality-camera-for-jetson.html)
- [Waveshare Jetson Orin NX AI Kit](https://www.waveshare.com/product/ai/boards-kits/jetson-orin/jetson-orin-nx-16g-dev-kit.htm)
- [QGC firmware yükleme](https://docs.qgroundcontrol.com/master/en/qgc-user-guide/setup_view/firmware.html)
- [PX4 Pixhawk Series](https://docs.px4.io/main/en/flight_controller/pixhawk_series)
- [ArduPilot Pixhawk firmware yükleme](https://ardupilot.org/copter/docs/common-loading-firmware-onto-pixhawk.html)
- [ArduPilot Copter latest firmware](https://firmware.ardupilot.org/Copter/latest/)
- [NVIDIA TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/)

## 7. Literatür incelemesinin sınırları

- Preprint sonuçları, peer-reviewed ve bağımsız olarak tekrar edilmiş sonuçlarla aynı kanıt seviyesinde değerlendirilmemiştir.
- Farklı çalışmaların başarı oranları; görev, araç, sensör, simülatör ve kontrol seviyeleri farklı olduğu için doğrudan karşılaştırılmamalıdır.
- Gerçek dünya denemesi yapılmış olması, yöntemin genel güvenliğini veya her platforma aktarılabileceğini kanıtlamaz.
- MuJoCo seçimi bu çalışmalardaki Flightmare, PyBullet, Gazebo veya özel simülatör sonuçlarını otomatik olarak yeniden üretmez; dinamik model bağımsız doğrulanmalıdır.
