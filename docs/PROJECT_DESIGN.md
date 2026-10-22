# Ebabil MuJoCo: Proje Tasarımı

**Belge durumu:** İlk teknik tasarım  
**Son güncelleme:** 22 Eylül 2026  
**Amaç:** Döner kanatlı bir insansız hava aracının MuJoCo ortamında, pekiştirmeli öğrenme kullanarak engellerden kaçıp belirlenen hedefe ulaşması ve geliştirilen yaklaşımın ileride gerçek araca aktarılabilecek şekilde tasarlanması.

## 1. Projenin hedefi

Projenin ilk hedefi, tam 3B hareket eden dört rotorlu bir hava aracının aşağıdaki görevi güvenilir şekilde gerçekleştirmesidir:

1. Rastgele fakat geçerli bir başlangıç durumundan kalkmak veya uçuşa başlamak.
2. Yalnızca gerçek bir araçta elde edilebilecek sensör bilgilerini kullanmak.
3. Statik engellerle çarpışmadan hedef bölgesine ulaşmak.
4. Hedef alan görüldüğünde bölgenin güvenli iç alanı üzerinde kararlı hover'a
   geçmek ve tanımlı hold süresini tamamlamak.
5. Uçuş boyunca kararlı, yumuşak ve fiziksel sınırlar içinde komutlar üretmek.
6. Eğitimde görülmeyen sahnelerde ve makul fizik/sensör belirsizliklerinde davranışı korumak.

İlk sürüm bir araştırma ve simülasyon platformudur. Gerçek araç uçuşu, sistem tanımlama, software-in-the-loop (SITL), hardware-in-the-loop (HITL), güvenlik filtresi ve kontrollü test alanı doğrulanmadan kapsam içinde kabul edilmez.

## 2. Temel teknik kararlar

| Konu | Karar | Gerekçe |
|---|---|---|
| Fizik motoru | MuJoCo | Hızlı rijit gövde dinamiği, temas modeli, sensörler ve Python API desteği |
| RL arayüzü | Gymnasium | Standart `reset`/`step` sözleşmesi, test edilebilir observation ve action space |
| Ana RL algoritması | PPO | Sürekli eylemleri ve paralel örneklemeyi destekleyen kararlı bir başlangıç algoritması |
| Politika seviyesi | Yüksek seviyeli hız ve yaw-rate komutları | Öğrenmeyi kolaylaştırır ve politikayı belirli motor/araç ayrıntılarına daha az bağımlı yapar |
| Alt seviye kontrol | Klasik kademeli kontrolcü | Hover, attitude ve motor kontrolünü RL probleminden ayırır; emniyet ve hata ayıklamayı kolaylaştırır |
| Fiziksel dış algı | IMX477 12MP motorized-focus monoküler kamera | Hedef araçta LiDAR/rangefinder bulunmayacak |
| Görsel algı | İki model: YOLO11n dedektör (hedef/engel kutuları) + PPO politikası (`vision-v4` kutu yuvaları) | Politikaya engel/hedef koordinatı girmez, yalnız görüntü kutuları; sembolik arayüz gerçek kamerayla yeniden eğitilen dedektörde değişmez (ADR-023). Etiketsiz VAE zinciri baseline (ADR-022) |
| Uçuş kontrolcüsü | Pixhawk 2.4.8; autopilot firmware'i bekleniyor | Hızlı state estimation, attitude/rate kontrolü, mixer ve failsafe; eski/klon kart varyantı bench testinde tanımlanacak |
| Görev bilgisayarı | Waveshare Jetson Orin NX AI Kit 8GB | Kamera işleme, yerel planlama ve PPO inference |
| Global konum | GPS/GNSS yok | Localization yalnız deploy edilebilir görsel-inertial kaynaklara dayanmalı |
| Terminal görev | Hedef bölgenin güvenli iç alanı üzerinde hover | Başarı, düşük hız/tutum hatasıyla kesintisiz hold; çarpışma her zaman failure |
| Episode ufku | Yerel segment eğitimi + rota temelli mission bütçesi | Harita büyürken PPO kredi ufkunu kontrol altında tutar |
| Eğitim yöntemi | Başarı tabanlı curriculum learning | Başlangıçta çözülebilir görev sunar ve zorluğu kontrollü artırır |
| Sim-to-real | Sistem tanımlama + seçici domain randomization | Ölçülebilen parametreleri sabitleyip gerçekten belirsiz parametreleri çeşitlendirir |
| Bağımlılık izolasyonu | Proje kökünde `.venv` | Yerel ve tekrarlanabilir Python çalışma ortamı |

## 3. Sistem mimarisi

```text
IMX477 kamera ──> Jetson Orin NX görsel encoder
                              │
Waypoint + Pixhawk state ─────┤
                              ▼
             PPO yerel navigasyon politikası
                      yaklaşık 20 Hz
                              │
                 sınırlandırılmış setpoint
                              ▼
         Pixhawk/autopilot state estimation + failsafe
                              │
               attitude → body-rate → mixer
                         100–500 Hz
                              │
                              ▼
             Motorlar / MuJoCo araç dinamiği
```

Politika doğrudan motor RPM üretmeyecektir. Jetson Orin NX, Pixhawk'a yüksek
seviyeli setpoint verir; MAVLink ilk entegrasyon adayıdır. Hızlı kontrol ve temel
failsafe Pixhawk üzerindeki autopilot'ta kalır. Doğrudan RPM/motor kontrolü, temel sistem
tamamlandıktan sonra ayrı bir araştırma kolu ve ablation olarak
değerlendirilecektir.

Eğitim ve otomatik evaluation headless çalışır. Görsel inceleme, seçilmiş episode'ların `qpos/qvel` ve metadata kayıtlarını ayrı MuJoCo viewer sürecinde oynatır; ayrıntılar [ADR-005](decisions/ADR-005-headless-training-and-viewer-replay.md) içindedir.

## 4. Araç modeli

Başlangıç modeli X konfigürasyonlu bir quadrotor olacaktır. Araç özellikleri kod içine dağınık sabitler olarak yazılmayacak, tek bir araç konfigürasyonundan okunacaktır.

Modellenecek ana bileşenler:

- Altı serbestlik dereceli rijit gövde ve free joint
- Araç kütlesi ve 3B atalet tensörü
- Kol uzunluğu ve rotor konumları
- Her rotor için thrust katsayısı
- Reaction torque/yaw moment katsayısı
- Motor komut ve thrust limitleri
- Birinci dereceden motor/rotor tepki gecikmesi
- Gövde eksenlerinde aerodinamik sürükleme
- Yerçekimi, zemin ve çarpışma geometrileri
- Harici rüzgâr kuvveti ve bozucu moment arayüzü

MuJoCo aktüatör ve sensör altyapısı kullanılacak; ihtiyaç duyulan rotor aerodinamiği veya rüzgâr kuvveti Python tarafında açık ve test edilebilir bir dinamik katmanla uygulanacaktır. MuJoCo'nun Python binding'leri, aktüatör modeli ve sensörleri resmi dokümantasyonda açıklanmaktadır: [MuJoCo Overview](https://mujoco.readthedocs.io/en/stable/), [Python API](https://mujoco.readthedocs.io/en/stable/python.html), [XML Reference](https://mujoco.readthedocs.io/en/stable/XMLreference.html).

## 5. Kontrol katmanı

RL eklenmeden önce aşağıdaki klasik kontrol zinciri çalışır durumda olmalıdır:

1. Motor mixer
2. Body-rate kontrolcüsü
3. Attitude kontrolcüsü
4. Gövde veya dünya ekseninde hız kontrolcüsü
5. İrtifa ve yaw kontrolü

Her kontrolcü doygunluk, integrator windup ve geçersiz sayısal değerler için korunmalıdır. Kontrol katmanının doğrulanacağı temel deneyler:

- Serbest düşüş ve yerçekimi işareti
- Dört motorda eşit thrust ile dikey hareket
- Mixer işaretlerinin roll, pitch ve yaw için doğruluğu
- Hover thrust hesabı
- Küçük attitude adım komutları
- Hız ve konum hedefi takibi
- Motor doygunluğu altında kararlılık

İlk baseline 10 Eylül 2026'da tamamlanmıştır: dünya-frame position/velocity PD, SO(3) attitude P ve body-rate PI zinciri fizik adımıyla senkron `500 Hz` çalışır. Güncel uygulama ayrıntıları ve nominal doğrulama sınırları [ADR-004](decisions/ADR-004-cascaded-flight-controller.md) içinde tutulur.

## 6. RL ortamı sözleşmesi

### 6.1 Observation space

`EbabilNavigation-v0` observation v1, ilk PPO ve fizik entegrasyonu için aşağıdaki
düşük boyutlu girdileri kullanır:

- hedefin body-frame göreli konumu ve normalize mesafesi,
- doğrusal/açısal hız, gövde yerçekimi yönü ve irtifa,
- 16 planar rangefinder ışını.

Hedef fiziksel araçta rangefinder bulunmadığı için bu sözleşme yalnız legacy
simülasyon baseline'ıdır. Deploy edilebilir `EbabilVisionNavigation-v0` için
kabul edilen `vision-v1` girdileri şunlardır:

- IMX477 kameradan düşük çözünürlüklü RGB görüntü veya hafif encoder latent'i,
- gerekli görülürse kısa görüntü/action geçmişi,
- Pixhawk veya doğrulanmış VIO'dan elde edilebilen attitude, angular rate,
  velocity ve altitude alt kümesi,
- global katmanın ürettiği yerel waypoint yönü/mesafesi,
- gerçek araç saatinden de üretilebilen normalize kalan segment zaman bütçesi.

Gözlem, simülatörün ayrıcalıklı tam durumunu doğrudan politikaya vermeyecektir.
MuJoCo depth, segmentation, raycast ve ground-truth state yalnızca perception
etiketi, privileged critic/teacher, curriculum, reward veya teşhis metriği için
kullanılabilir. Actor tensörüne sızmadığı otomatik test edilmelidir.

Legacy rangefinder sözleşmesi [ADR-006](decisions/ADR-006-rangefinder-and-static-scenes.md),
kamera/Pixhawk/Jetson sınırı
[ADR-011](decisions/ADR-011-camera-only-pixhawk-jetson-architecture.md), kesin
observation/action/frekans sözleşmesi ise
[ADR-015](decisions/ADR-015-vision-navigation-contract-v1.md) içindedir.

**Güncelleme (22 Eylül 2026, ADR-022).** Deploy edilebilir sözleşme artık
`vision-v3`'tür: yalnız actor RGB'siyle eğitilmiş donmuş VAE kodlayıcının son
dört ortalama vektörü, `fc_state`, inanılan hedeften türetilen `waypoint` ve
`timing`. Actor'a giren hiçbir görsel kanal depth/segmentation türevi etiketle
eğitilmiş ağdan gelemez; bu etiketler yalnız kalite probe'u, privileged
reward/termination ve tanı içindir. `vision-v1/v2` ve etiketli encoder zinciri
legacy tanı baseline'ıdır. Ayrıntı
[ADR-022](decisions/ADR-022-label-free-visual-representation.md).

**Güncelleme (26 Eylül 2026, ADR-023).** Kullanıcı kararıyla deploy
edilebilir sözleşme `vision-v4` oldu: bir YOLO dedektörünün (sınıflar
`obstacle`, `target`) kutu listesi `9` yuva x `8` alan (`present, cx, cy, w,
h, conf, range_norm, range_valid`) olarak politikaya girer; `fc_state`,
`waypoint` (yaklaşık hedef) ve `timing` değişmez, toplam `88` boyut.
Politika engel/hedef koordinatı görmez. Dedektör simülatörün otomatik kutu
etiketleriyle eğitilir ve onay zincirinden geçer; hover durum makinesinin
görsel kapısı dedektörün hedef kutusudur. Dedektör kütüphanesi Ultralytics
(AGPL-3.0; kullanıcı itirazına açık). ADR-022 VAE zinciri ve `vision-v3`
karşılaştırma baseline'ıdır. Ayrıntı
[ADR-023](decisions/ADR-023-detector-policy-two-model-perception.md).

Rotation representation ve doğrusal hızın gözlemde bulunması, gerçek araç aktarımını inceleyen SimpleFlight çalışmasının temel bulgularındandır: [Chen ve diğerleri, 2025](https://arxiv.org/abs/2412.11764).

### 6.2 Action space

Legacy ortam `Box(-1, 1, shape=(4,))` içinde
`[vx_ref, vy_ref, vz_ref, yaw_rate_ref]` üretir. Tek kamera yan ve arka çevreyi
aynı anda görmediğinden bu serbestlik vision ortamına aynen taşınmayacaktır.

İlk vision action sözleşmesi `Box(-1, 1, shape=(3,))` üzerinde
`[forward_speed, vertical_speed, yaw_rate]` olarak kabul edilmiştir. İleri hız
`max(0, action) * 1.5 m/s` dönüşümünü kullanır; negatif değer translasyon
üretmez. Yanal/geri hareket ancak temporal yerel harita ve görüş temelli
güvenlik doğrulanırsa açılır.

Ortam normalize eylemleri güvenli fiziksel sınırlara dönüştürecektir.
Stable-Baselines3, sürekli eylemlerin simetrik `[-1, 1]` aralığında normalize
edilmesini önermektedir: [SB3 2.9.0 RL Tips](https://stable-baselines3.readthedocs.io/en/v2.9.0/guide/rl_tips.html).

### 6.3 Zaman ölçekleri

Önerilen başlangıç değerleri deneysel olarak doğrulanacaktır:

- MuJoCo physics timestep: 1–2 ms
- Alt seviye kontrol: 200–500 Hz
- PPO politika frekansı: 20–50 Hz
- Sensör frekansları: gerçek sensör adaylarına göre ayrı ayrı
- Maksimum bölüm süresi: görev boyutuna bağlı yapılandırma

Tek bir Gymnasium adımı içinde birden fazla fizik ve kontrol alt adımı çalıştırılacaktır.

İlk ortam sözleşmesi `20 Hz` policy, `500 Hz` fizik/kontrol ve adım başına `25`
alt adım olarak ADR-008 ile sabitlenmiştir. Legacy ortamın `20 s / 400 adım`
sınırı korunur. Yeni vision ortamında yerel segment bütçesi planlanan rota
uzunluğu ve nominal hızdan hesaplanır; uzun görev evaluation'ı segmentleri fizik
durumunu sıfırlamadan zincirler. Ayrıntılar
[ADR-013](decisions/ADR-013-segmented-episodes-and-time-budget.md) ve uygulama
sözleşmesi [ADR-015](decisions/ADR-015-vision-navigation-contract-v1.md) içindedir.

## 7. Ödül tasarımı

İlk shaped reward taslağı:

```text
r_t =
    w_progress * (distance_previous - distance_current)
  + goal_bonus
  - collision_penalty
  - w_proximity * proximity_cost
  - w_smooth * ||action_t - action_(t-1)||²
  - w_effort * control_effort
  - w_attitude * unsafe_attitude_cost
  - time_penalty
```

Tasarım ilkeleri:

- Hedefe net ilerleme temel yoğun ödül olmalıdır.
- Hedef bonusu görevin asıl amacını baskın tutmalıdır.
- Başarı, hedef yarıçapı ve düşük hız koşulunun `0.50 s` kesintisiz korunmasını
  gerektirir; anlık hedef geçişi başarı sayılmaz.
- Yakınlık cezası engellerden uzak durmayı öğretmeli, fakat hareketsiz kalmayı avantajlı yapmamalıdır.
- Action-difference cezası gerçek donanım için komut yumuşaklığını teşvik etmelidir.
- Enerji ve kontrol eforu başarıyı engelleyecek büyüklükte olmamalıdır.
- Her ödül bileşeni `info` içinde ayrı raporlanmalıdır.
- Katsayılar testlerle ve ablation çalışmalarıyla doğrulanmalıdır.

Kontrol yumuşaklığının fiziksel araçta enerji tüketimi ve motor sağlığına etkisi [RE+AL](https://arxiv.org/abs/2012.06656) ve [CAPS](https://arxiv.org/abs/2012.06644) çalışmalarında gösterilmiştir.

## 8. Bölüm sonları

`terminated=True` koşulları:

- Hedef bölgesine ulaşma
- Engel, zemin veya sınırla ciddi çarpışma
- İzin verilen attitude/irtifa sınırlarının aşılması
- Kurtarılamaz kontrol kaybı veya geçersiz fizik durumu

`truncated=True` koşulu:

- Maksimum bölüm süresine ulaşılması

Gerçek görev sonu ile zaman aşımının ayrımı Gymnasium sözleşmesine uygun tutulacaktır: [Gymnasium Custom Environment](https://gymnasium.farama.org/main/introduction/create_custom_env/).

## 9. Sahne ve curriculum tasarımı

Sahneler seed ile yeniden üretilebilir, prosedürel ve çözülebilir olmalıdır. Başlangıç ve hedef arasında en az bir fiziksel olarak geçerli rota bulunduğu sahne üretim aşamasında kontrol edilmelidir.

İlk üç görev dağılımı kodda `clear`, `single_blocker` ve `random_obstacles` olarak ayrılmıştır. Engelli modlar doğrudan rotayı zorunlu olarak kapatır; alternatif rota araç açıklığıyla genişletilmiş 2B occupancy grid üzerinde doğrulanır. Bu sözleşmenin ayrıntıları [ADR-007](decisions/ADR-007-curriculum-scene-modes-and-path-validation.md) içindedir.

Başlangıç/hedef dağılımı engel seviyesinden ayrı seçilir: `fixed`,
`random_start`, `random_goal` ve `random_both`. Rastgele endpoint'ler scene
seed'iyle güvenli workspace sınırlarında ve minimum ayrımla örneklenir.

Curriculum seviyeleri:

1. Hover
2. Engelsiz hedef takibi
3. Tek büyük engel
4. Seyrek ve geniş aralıklı engeller
5. Koridor ve geniş geçitler
6. Dar geçitler
7. Yoğun, rastgele statik engeller
8. Sensör gürültüsü ve gecikme
9. Rüzgâr ve dinamik parametre belirsizliği
10. Hareketli engeller

Bir üst seviyeye geçiş yalnızca timestep sayısına değil, kayan penceredeki başarı ve çarpışma oranına bağlanacaktır. Curriculum yaklaşımının zor quadrotor görevlerinde keşif ve eğitim kararlılığına etkisi [Xiao, Lu ve He, 2021](https://arxiv.org/abs/2108.12869) tarafından gösterilmiştir.

### 9.1 Büyük alan ve görev mesafesi

Metre ölçekli yerel engelden kaçınma ile yüzlerce metre/kilometre ölçekli görev
planlama aynı problem olarak ele alınmayacaktır. Kabul edilen mimaride global
planlayıcı nihai görevi yerel waypoint veya güvenli koridorlara böler; PPO
politikası yalnızca yakın çevrede waypoint takibi ve engelden kaçınmadan
sorumludur. Klasik kontrolcü yine hızlı uçuş dinamiğini yönetir.

Görev mesafesi; engel yoğunluğu, geçit genişliği, hız, sensör bozulması ve
rüzgârdan ayrı bir curriculum ekseni olacaktır. MuJoCo sahnesinde tüm kilometre
karesini yüksek çözünürlükte temsil etmek yerine yakın çevre fiziksel olarak
aktif tutulacak, uzak dünya seyrek global harita olarak yönetilecektir. Mevcut
`EbabilNavigation-v0` ve observation v1 baseline olarak dondurulacak; waypoint
yönü ve ölçeklenebilir mesafe kodlaması yeni bir observation/environment
sürümünde tanımlanacaktır.

Mimarinin ayrıntıları ve uygulama doğrulamaları
[ADR-010](decisions/ADR-010-hierarchical-distance-curriculum.md) içindedir.

Yerel PPO eğitim episode'ları `5-30 m` plan uzunluklu segmentlerle sınırlı
tutulur. Zaman bütçesi rota uzunluğu, nominal hız, kaçınma/toparlanma çarpanı ve
terminal rezervinden türetilir. D3-D5 uzun görevlerinde ara waypoint episode'u
bitirmez; mission manager aynı fizik durumuyla sonraki segmente geçer. Yalnız
nihai hedef üzerindeki hover hold görevi başarıyla tamamlar. Bu sözleşme
[ADR-013](decisions/ADR-013-segmented-episodes-and-time-budget.md) içindedir.

GPS/GNSS bulunmadığı için global katman simülatör ground-truth koordinatını
actor'a veremez. Hedef alan başlangıçta görünmüyorsa takeoff-relative VIO,
önceden tanımlı görsel harita/relocalization veya başka bir kamera tabanlı
referans gerekir. Hedef görülünce görev
`acquired -> approach -> settle -> hold` terminal state machine'ine geçer;
hover boyunca kamera görüşü veya sınırlandırılmış VIO takibi doğrulanır.
Ayrıntılar [ADR-012](decisions/ADR-012-safe-terminal-mission-scope.md) içindedir.

## 10. Sim-to-real stratejisi

Gerçekçilik tek seferde maksimuma çıkarılmayacaktır. Önce nominal model doğrulanacak, ardından aşağıdaki sıra izlenecektir:

1. Sensör beyaz gürültüsü
2. Sabit veya yavaş değişen sensör bias'ı
3. Örnekleme gecikmesi ve paket kaybı
4. Aktüatör gecikmesi ve thrust belirsizliği
5. Kütle, atalet ve ağırlık merkezi değişimleri
6. Aerodinamik sürükleme ve rüzgâr
7. Gerçek uçuş loglarından öğrenilen hata/residual dağılımları

Randomizasyon seçici olmalıdır. Ölçülebilen kütle, atalet, rotor thrust eğrisi ve gecikme gibi parametreler önce sistem tanımlamayla belirlenmeli; belirsiz kalanlar fiziksel olarak makul aralıklarda randomize edilmelidir. Bu yaklaşım [SimpleFlight](https://arxiv.org/abs/2412.11764) ile uyumludur. Gerçek loglardan algı ve dinamik residual modelleri çıkarma yaklaşımına örnek olarak [Swift](https://www.nature.com/articles/s41586-023-06419-4) kullanılacaktır.

## 11. Güvenlik mimarisi

Ödül cezası güvenlik garantisi değildir. Gerçek araca geçmeden önce politika ve uçuş kontrolcüsü arasına bağımsız bir güvenlik katmanı eklenmelidir:

- Hız, acceleration, attitude ve thrust limitleri
- Minimum irtifa ve çalışma hacmi sınırı
- Acil hover/land davranışı
- Sensör timeout denetimi
- Politika çıktısında NaN/Inf ve aralık kontrolü
- Engel yakınlığında eylem kısıtlama veya safety filter
- Manuel pilot override ve fiziksel kill switch
- Kontrolsüz hedef/zemin çarpışmasını her koşulda failure sayma

LiDAR temelli Composite Control Barrier Function yaklaşımı, politika ile
bağımsız güvenlik katmanını ayırma ilkesini destekler; ancak hedef araçta LiDAR
olmadığı için yöntem mevcut haliyle deploy edilemez. Kamera tabanlı risk
kestirimi hard güvenlik garantisi sayılmayacak; görüş dışı hareket kısıtları,
failsafe ve kontrollü SITL/HITL doğrulaması zorunlu olacaktır.

Otonom hedefe çarpma veya zarar verme davranışı proje kapsamı dışındadır;
aktif terminal görev hedef bölgenin güvenli iç alanı üzerinde kararlı hover ve
gerektiğinde abort'tur.

## 12. Değerlendirme protokolü

Ana metrikler:

- Başarı oranı
- Çarpışma oranı
- Hedefe ulaşma süresi
- Segment başarı ve timeout oranı
- Mission completion ve toplam bütçe kullanım oranı
- Hedef üzerinde hover'a yerleşme süresi, hold tamamlama ve hold ihlal sayısı
- Yol uzunluğu ve mümkün olan en kısa yola oranı
- Minimum engel açıklığı
- Kontrol eforu ve yaklaşık enerji
- Eylem değişimi, jerk ve yüksek frekanslı komut içeriği
- Kontrol kaybı ve çalışma alanından çıkma oranı
- Farklı rüzgâr, gürültü ve gecikme seviyelerinde başarı

Deney kuralları:

- Eğitim, validation ve test seed'leri ayrılacaktır.
- Her ana deney en az 3, tercihen 5 bağımsız seed ile çalıştırılacaktır.
- Sonuçlar ortalama, standart sapma ve mümkünse güven aralığıyla verilecektir.
- Eğitim reward'u tek başına başarı göstergesi sayılmayacaktır.
- PPO, klasik kontrol/planner baseline ile karşılaştırılacaktır.
- Curriculum, randomization, observation history ve smoothness bileşenleri için ablation yapılacaktır.

## 13. Yazılım ve yeniden üretilebilirlik

Yerel makine durumu:

- macOS 15.5
- Apple Silicon `arm64`
- Python 3.13.2

Planlanan kurallar:

- Sanal ortam proje kökünde `.venv` olacaktır.
- `.venv/`, eğitim çıktıları ve büyük checkpoint'ler `.gitignore` içine alınacaktır.
- Paket tanımı `pyproject.toml` içinde tutulacaktır.
- Uyumluluk smoke testinden sonra kesin bağımlılık sürümleri lock dosyasına yazılacaktır.
- Tüm deneyler config, scene/sensor/training seed'leri, mevcutsa commit kimliği ve bağımlılık sürümlerini kaydedecektir.
- Eğitim, değerlendirme ve görselleştirme giriş noktaları birbirinden ayrılacaktır.
- Render gerektirmeyen eğitim headless çalışabilecektir.

Çekirdek paket ailesi:

- `mujoco`
- `gymnasium`
- `stable-baselines3`
- `torch`
- `numpy`
- Konfigürasyon, test, loglama ve grafik paketleri

10 Eylül 2026 itibarıyla çekirdek paketlerin Python 3.13 desteği paket sayfalarında bulunmaktadır: [MuJoCo PyPI](https://pypi.org/project/mujoco/), [Gymnasium PyPI](https://pypi.org/project/gymnasium/), [Stable-Baselines3 PyPI](https://pypi.org/project/stable-baselines3/), [PyTorch PyPI](https://pypi.org/project/torch/).

## 14. Kapsam dışı veya sonraki çalışmalar

- İlk aşamada yüksek çözünürlüklü doğrudan pixel-to-motor PPO (ADR-022:
  kodlayıcının PPO içinde ince ayarı ikinci kademedir, saf piksel PPO
  ablation'dır)
- İlk aşamada doğrudan motor RPM kontrolü
- Çoklu İHA/sürü koordinasyonu
- Sertifikalı gerçek uçuş güvenliği iddiası
- Açık alanda izinsiz veya gözetimsiz test

Kamera tabanlı algı artık ana teslimatın parçasıdır. GPS-denied VIO/SLAM ve
görsel relocalization, alan başlangıçta görünmüyorsa ana araştırma gereksinimidir;
kilometre ölçeğine çıkmadan önce ayrı doğrulanacaktır.

## 15. Henüz netleştirilecek girdiler

Implementasyon bu kararları bloke etmeyecek şekilde parametrik başlayacaktır; ancak gerçek araca yaklaşmadan önce aşağıdakiler kesinleştirilmelidir:

- Hedef fiziksel quadrotor platformu
- Araç kütlesi, atalet ve thrust ölçümleri
- Pixhawk 2.4.8'in gerçek board ID/FMU hedefi, flash kapasitesi ve sensör
  envanteri
- Seçilecek tek autopilot firmware'i: PX4 veya ArduPilot
- Kamera üzerindeki kesin üretici/SKU, gerçek lens/FOV, shutter ve montaj açısı
- IMX477 ile Waveshare carrier board'un JetPack/L4T sürücü uyumluluğu
- Hedef alanın görsel görünümü, fiziksel boyutu ve güvenli inişe uygunluğu
- GPS-denied uzun menzil localization: VIO + visual map/loop-closure seçimi
- Kamera ile Pixhawk IMU arasındaki timestamp/senkronizasyon imkânı
- İç/dış mekân hedef senaryosu
- Maksimum uçuş hızı ve güvenlik hacmi
- Jetson Orin NX JetPack/L4T sürümü, güç modu ve ölçülecek uçtan uca
  inference/gecikme bütçesi
