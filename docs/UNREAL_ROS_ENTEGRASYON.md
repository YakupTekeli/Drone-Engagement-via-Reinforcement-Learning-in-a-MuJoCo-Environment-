# Unreal Engine + ROS Entegrasyonu: Eğitilmiş Politikayı Çalıştırma

Bu belge, eğitilmiş Ebabil RL politikasını Unreal Engine'de çalışan bir
simülasyona ROS 2 üzerinden bağlamayı anlatır. Senaryo: drone ağaçlardan
(engel) kaçınarak tanka (hedef) yönelir ve güvenli bir **temassız** yakın
mesafede durur. Hedefe fiziksel temas/çarpma kapsam dışıdır ve köprüde pozitif
bir standoff tabanıyla engellenir.

## 1. Mimari

```
Unreal Engine (oyun + fizik)
   │  kamera karesi (Image)         │  uçuş durumu (Odometry)
   ▼                                ▼
ROS 2  ────────────  ebabil_policy_bridge (scripts/ros_policy_bridge.py)
   ▲                        │  1. YOLO (ağaç, tank) -> kutular
   │  hız komutu (Twist)    │  2. vision-v4 gözlemi (ebabil.deploy)
   │                        │  3. PPO.predict -> action
   └────────────────────────┘  4. koni + hız sınırı + standoff -> Twist
```

Köprünün beyni `src/ebabil/deploy/policy_runner.py` içindeki `PolicyRunner`'dır.
Gözlem kurulumu ve hareket kısıtları, politikanın MuJoCo'da eğitildiği **aynı**
`ebabil` fonksiyonlarını çağırır; böylece politika eğitimdekiyle birebir aynı
88 sayılık girdiyi görür. ROS düğümü yalnız ince bir sarmalayıcıdır.

## 2. Çerçeve ve birim sözleşmesi

Bu en kritik konudur; yanlışı "drone ters yöne uçuyor" demektir.

| | Ebabil / politika | ROS (REP-103) | Unreal Engine |
|---|---|---|---|
| Ellilik | sağ-elli | sağ-elli | **sol-elli** |
| Gövde | FLU (+x ileri, +y sol, +z yukarı) | FLU (base_link) | +x ileri, +y sağ, +z yukarı |
| Dünya | +z yukarı | ENU | +z yukarı |
| Birim | metre, rad, m/s | metre, rad, m/s | **santimetre** |

Politika ve ROS aynı FLU/sağ-elli kuralı kullandığı için **köprünün ROS tarafında
işaret çevirisi yapmaya gerek yoktur**. Sol-elli, santimetre Unreal çerçevesine
dönüşüm, zaten kullandığınız Pixhawk-Unreal köprüsünün (ROS <-> UE) işidir. Yani
bu düğüm ROS tarafında SI + FLU üretir; UE'ye çeviriyi mevcut köprünüz yapar.

Doğrulama ipucu: UE'de drone'u eliyle sabit tutup köprüye yalnız `+forward`
komutu verdirin; araç kameranın baktığı yöne gitmeli. `+left` komutunda sola
gitmeli. Yaw `+` komutunda saat yönünün tersine (sola) dönmeli. Biri ters ise
dönüşüm Pixhawk-UE köprüsündedir, bu düğümde değil.

## 3. Unreal kamerasını eğitimle eşleyin

Politikanın menzil tahmini kameranın iç parametrelerine ve montaj açısına
bağlıdır. Eğitim değerleri:

| Parametre | Değer |
|---|---|
| Yatay FOV | `75°` |
| Aşağı pitch | `40°` |
| Kamera konumu (gövde) | `(0.10, 0.0, -0.01) m` |
| Çözünürlük | `320 x 240` (eğitim); köprü başka çözünürlüğü de kabul eder |

FOV ve pitch'i bu değerlere ayarlayın. Çözünürlük farklı olabilir: köprü kamera
geometrisini gerçek kare boyutundan kurar, kutular da o piksel uzayındadır. Ama
FOV veya pitch farklı olursa menzil tahmini eğitim dağılımının dışına çıkar ve
kaçınma bozulur.

## 4. Gereken ROS topic'leri

| Yön | Varsayılan topic | Tip | İçerik |
|---|---|---|---|
| Giriş | `/drone/camera/image_raw` | `sensor_msgs/Image` | Ön kamera RGB |
| Giriş | `/drone/odom` | `nav_msgs/Odometry` | Konum (dünya), attitude (kuaterniyon), hız (gövde) |
| Çıkış | `/drone/cmd_vel` | `geometry_msgs/Twist` | Gövde-FLU hız komutu: `linear.x`=ileri, `linear.y`=sol, `linear.z`=yukarı, `angular.z`=yaw-rate |

Topic adları düğüm parametreleriyle değişir (bkz. §6).

Notlar:
- **Odometry hız çerçevesi.** Köprü `twist.twist`'i gövde (base_link)
  çerçevesinde varsayar (REP-103). UE/MAVROS dünya çerçevesinde hız veriyorsa
  köprüde `linear_velocity_body_m_s` kurulurken `R.T @ v_world` ile çevirin.
- **İrtifa.** Politika irtifayı yerden yükseklik olarak kullanır ve `0-3 m`
  bandında eğitildi. UE sahnesi daha büyükse `DeploymentConfig.workspace_max_z_m`
  değerini sahnenize göre ayarlayın.
- **Hız komutu tüketimi.** Çıktı gövde-FLU hız setpoint'idir. Mevcut
  Pixhawk/MAVROS kurulumunuz bunu `setpoint_velocity/cmd_vel` benzeri bir
  arayüzle alıp altındaki klasik kontrolcüye vermelidir (politika yalnız hız
  komutu verir, ADR-011 kademeli mimarisi gibi).

## 5. Kurulum (ROS 2 tarafı)

Köprü ROS 2, `cv_bridge`, `ultralytics` ve bu depodaki `ebabil` paketini ister.
ROS 2 çalışma alanınızdaki Python ortamında:

```bash
pip install -r requirements.lock        # ebabil bağımlılıkları (ultralytics dahil)
pip install --no-build-isolation -e .   # ebabil paketini kur
# cv_bridge ROS 2 dağıtımınızla gelir (ör. ros-humble-cv-bridge).
```

`scripts/ros_policy_bridge.py` dosyasını ROS 2 paketinizin `scripts/` veya
`nodes/` klasörüne kopyalayın (ya da bu depoyu PYTHONPATH'e ekleyin).

## 6. Çalıştırma

Depodaki eğitilmiş politika ve kendi YOLO ağırlığınızla:

```bash
ros2 run <paketiniz> ros_policy_bridge \
  --ros-args \
  -p policy_path:=runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  -p yolo_weights:=/yol/agac_tank_yolo.pt \
  -p image_topic:=/drone/camera/image_raw \
  -p odometry_topic:=/drone/odom \
  -p command_topic:=/drone/cmd_vel \
  -p obstacle_class_names:="['tree']" \
  -p target_class_names:="['tank']" \
  -p believed_target_world_m:="[12.0, 0.0, 0.0]" \
  -p target_radius_m:=3.5 \
  -p surface_clearance_m:=0.5
```

- `obstacle_class_names` / `target_class_names`: kendi YOLO modelinizdeki sınıf
  adları. Köprü bunları politikanın `obstacle`/`target` sınıflarına eşler.
- `believed_target_world_m`: tankın **yaklaşık** dünya konumu (ADR-019 waypoint
  girdisi, `±2 m` belirsizlik varsayılır). Oyun tankın kabaca nerede olduğunu
  biliyorsa buraya verin; politika oraya yönelir ve tankı kamerayla arar.
- `target_radius_m`: tankın yaklaşık yarı-boyutu (metre). Durma mesafesi tankın
  **merkezine** değil **yüzeyine** göre ölçülür; bu yüzden tank boyutu gerekir.
  Varsayılan `3.5` (yaklaşık bir ana muharebe tankı yarı-uzunluğu); kendi
  modelinize göre ayarlayın.
- `surface_clearance_m`: tankın kenarından bırakılacak pay (metre). "Dur ve
  gözle" mesafesi budur; merkeze durma mesafesi = `target_radius_m + clearance`.
  Köprü `0.3 m`'lik pozitif bir pay tabanı uygular; temassızlık bununla
  garanti edilir (`surface_clearance_m` 0 verilse bile).

> Neden merkeze değil yüzeye? Yaklaşık hedef konumu `±2 m` belirsiz ve tank
> birkaç metre yer kaplar. Merkeze küçük bir mesafe (ör. `0.5 m`) tankın gövdesi
> içine düşer, yani temas olur. Yüzeye `0.5 m`, gerçekten yakın ve temassız bir
> gözlem mesafesidir.

## 7. Davranış ve beklenen sonuç

- **Seyir.** Politika action verir; köprü kamera konisi kısıtını (yalnız
  görülen yöne hareket) ve koridor hız sınırlayıcısını (öndeki yakın ağaçta
  yavaşla) uygular. Drone ağaçlardan kaçınarak tanka yönelir.
- **Yaklaşma ve durma.** Yaklaşık tanka yatay mesafe standoff değerinin altına
  inince köprü otomatik olarak "dur ve hedefi ortala" durumuna geçer: yatay hız
  sıfırlanır, irtifa tutulur, yaw ile tank kadraj ortasına alınır.

## 8. Önemli sınırlılık: aktarım kalitesi

Politika, hedefin düz bir `2 x 2 m` zemin karesi, engellerin de yere oturan
basit cisimler (kutu, silindir, küre) olduğu MuJoCo sahnesinde eğitildi. Unreal'de:

- **Ağaç** uzun ve ince, tepesi geniştir. Menzil, YOLO kutusunun alt kenarının
  zemine izdüşümünden hesaplanır; ağaç gövdesinin tabanı kadrajdaysa makuldür,
  tepe kutusu havada kalırsa menzil yanlış olur. Kaçınma "önde ~X m'de bir engel
  var" düzeyinde çalışır; ince ayar gerekebilir.
- **Tank** yükseltilmiş 3B bir cisimdir, düz `2 x 2 m` kare değildir. Bu yüzden
  standoff tetikleyicisi dedektörün tank menzili yerine **yaklaşık waypoint**
  mesafesini kullanır (köprüde zaten böyle).

Gerçekçi beklenti: ilk denemede kaçınma ve yönelme çalışır, ince ayar
gerekebilir. Davranış zayıfsa en sağlam yol, ağaç/tank geometrisine benzeyen bir
MuJoCo sahnesi (uzun ince engeller + yükseltilmiş hedef) kurup politikayı kısa
bir süre yeniden eğitmektir (ADR-023 §12'deki gerçek-görüntü ince ayar adımının
geometri karşılığı). Bu, ayrı bir iş dilimi olarak planlanabilir.

## 9. Windows + ROS 2 Jazzy (WSL2) kurulumu — adım adım

Hedef kurulum: Unreal Engine Windows'ta, ROS 2 Jazzy aynı makinede WSL2
(Ubuntu 24.04, Python 3.12) içinde, köprü de WSL2 içinde. ROS'unuz zaten
Unreal ile konuşuyorsa (Pixhawk-Unreal ROS köprüsü), biz yalnız aynı ROS
grafiğine bir düğüm ekliyoruz; Unreal <-> ROS ağ bağlantısını yeniden kurmaya
gerek yok.

Önemli: `ebabil` Python `3.13` ister, Jazzy'nin Python'u `3.12`'dir. `rclpy`
`3.12`'ye bağlı olduğu için köprüyü `3.12`'de çalıştırırız ve `ebabil`'i
**kurmadan**, PYTHONPATH ile içe aktarırız (kurulum sürüm pinine takılmaz).
Kod `3.12`'de çalışır; tek risk budur ve adım 4'te doğrulanır.

### Adım 1 — Depoyu ve YOLO ağırlığını WSL'e taşı

WSL2 Ubuntu 24.04 terminalinde:

```bash
sudo apt update
sudo apt install -y git python3-venv python3-pip
git clone https://github.com/YakupTekeli/Drone-Engagement-via-Reinforcement-Learning-in-a-MuJoCo-Environment-.git ~/ebabil
```

Klon, eğitilmiş politikayı (`runs/detector_cone_d1a_seed20260912_20260929/final_model.zip`)
da getirir. Kendi ağaç/tank YOLO ağırlığınızı (`.pt`) WSL dosya sistemine
kopyalayın, ör. `~/agac_tank_yolo.pt` (Windows'taki dosyaya `/mnt/c/...`
üzerinden de erişebilirsiniz).

### Adım 2 — ROS 2 cv_bridge ve derleme paketleri

```bash
sudo apt install -y ros-jazzy-cv-bridge
```

(ROS 2 Jazzy'nin kendisi zaten kuruludur; değilse resmi Jazzy Ubuntu 24.04
kurulumunu yapın.)

### Adım 3 — Python bağımlılıkları (3.12)

Sistem paketlerini (rclpy, cv_bridge) görecek bir sanal ortam:

```bash
cd ~/ebabil
python3 -m venv --system-site-packages .venv-deploy
source .venv-deploy/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "stable-baselines3==2.9.0" "gymnasium==1.3.0" mujoco ultralytics numpy pillow
```

`ebabil`'i KURMUYORUZ (3.13 pini engeller); bunun yerine PYTHONPATH kullanırız
(adım 5). `torch` CPU sürümüdür; politika CPU'da gerçek zamanlı koşar.

### Adım 4 — İçe aktarma doğrulaması

ROS ortamını ve venv'i birlikte yükleyip hem `rclpy`'nin hem `ebabil`'in
göründüğünü doğrulayın:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ebabil/.venv-deploy/bin/activate
cd ~/ebabil
PYTHONPATH=$PWD/src:$PYTHONPATH python3 -c "import rclpy, cv_bridge; import ebabil.deploy.policy_runner; print('import OK')"
```

`import OK` görmelisiniz. Hata alırsanız (çoğunlukla bir Python paketi `3.12`
tekerleği bulamadıysa) bana çıktısını iletin; alternatif olarak §9'un sonundaki
soket-ayırma yolu politikayı Windows'ta tam `3.13` ortamında tutar.

### Adım 5 — Mevcut topic adlarını bul

Unreal'li ROS grafiğiniz çalışırken, köprünün bağlanacağı üç topic'i tespit edin:

```bash
ros2 topic list
ros2 topic info -v <aday-topic>     # tip ve yayıncı/aboneyi görmek için
```

- Kamera görüntüsü: `sensor_msgs/Image` (Unreal'in drone kamerası).
- Uçuş durumu: `nav_msgs/Odometry` (konum + attitude + hız).
- Hız komutu: Unreal/Pixhawk tarafının hız setpoint'i olarak TÜKETTİĞİ topic
  (`geometry_msgs/Twist`). Köprü buraya yayınlar.

### Adım 6 — Köprüyü çalıştır

Aynı iki kaynak (`setup.bash` + venv) yüklüyken:

```bash
cd ~/ebabil
PYTHONPATH=$PWD/src:$PYTHONPATH python3 scripts/ros_policy_bridge.py --ros-args \
  -p policy_path:=runs/detector_cone_d1a_seed20260912_20260929/final_model.zip \
  -p yolo_weights:=/home/<kullanici>/agac_tank_yolo.pt \
  -p image_topic:=<sizin-kamera-topic> \
  -p odometry_topic:=<sizin-odometry-topic> \
  -p command_topic:=<sizin-hiz-komutu-topic> \
  -p obstacle_class_names:="['tree']" \
  -p target_class_names:="['tank']" \
  -p believed_target_world_m:="[12.0, 0.0, 0.0]" \
  -p target_radius_m:=3.5 \
  -p surface_clearance_m:=0.5
```

`obstacle_class_names` / `target_class_names` kendi YOLO modelinizin sınıf
adlarıdır. `believed_target_world_m` tankın yaklaşık dünya konumudur.

### Adım 7 — İlk uçuş kontrolü

- `ros2 topic echo <hiz-komutu-topic>` ile komutların aktığını görün.
- §2'deki doğrulama: drone yalnız `+forward` komutunda kameranın baktığı yöne,
  `+left`'te sola gitmeli; biri ters ise dönüşüm Pixhawk-Unreal köprüsündedir.
- Beklenen davranış: ağaçlardan kaçınarak tanka yönelir; yaklaşık tanka yatay
  mesafe `target_radius + 0.5 m`'nin altına inince durur ve yaw ile tankı
  kadraj ortasına alır (temassız gözlem).

### Alternatif: soket-ayırma (3.12 sorun çıkarırsa)

Adım 4 içe aktarma veya model yükleme `3.12`'de sorun çıkarırsa: politikayı
Windows'ta tam `3.13` venv'inde (depodaki `requirements.lock`) ayrı bir süreç
olarak çalıştırıp, WSL'deki ince bir ROS düğümünün kamera+odometri'yi yerel
soketle bu sürece iletmesi en sağlam yoldur. Bu yolu isterseniz soket
sunucu/istemci kodunu yazarım.

## 10. Güvenlik notu

Bu simülasyon davranışıdır; gerçek uçuş güvenliği iddiası değildir. Pixhawk
HITL/SITL hattına bağlıysanız AGENTS.md §8 geçerlidir: pervaneli gerçek test,
failsafe, manuel override ve fiziksel test prosedürü ayrı ve zorunludur.
