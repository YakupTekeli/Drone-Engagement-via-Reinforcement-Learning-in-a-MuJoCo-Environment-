# ADR-016: Hover Geometrisi, Nadir Kör Bölgesi ve Hedef Açıklığı

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026  
**İlişki:** ADR-011 kamera donanımını, ADR-012 terminal hover görevini ve
ADR-008 legacy sahne sözleşmesini kısıtlar. REV-014–REV-017 bulgularını kapatır.

## Bağlam

ADR-012 terminal görevi hedef bölgenin güvenli iç alanı üzerinde kesintisiz
hover olarak tanımladı ve kamera montaj açısının hedefi yaklaşma ile hover
boyunca görüşte tutmasını kabul kapısı saydı. REV-014 incelemesi bu kapının
hover pozu için hiç ölçülmediğini gösterdi.

Ölçülen geometri şudur. Kamera dikey görüş alanı `59.84 derece`, montaj açısı
`45 derece` aşağıdır. Görüntünün alt kenarı bu nedenle yatayın `74.92 derece`
altına bakar. Aracın tam altındaki noktanın kadrajda olması için

```text
pitch_down >= 90 derece - vfov / 2 = 60.08 derece
```

koşulu gerekir. Zemin izinin nadir'den yakın sınırı `irtifa / tan(pitch + vfov/2)`
olduğu için irtifayla doğrusal ölçeklenir:

| Hover irtifası | Zemin izi yakın sınırı |
|---:|---:|
| `1.00 m` | `0.270 m` |
| `1.50 m` | `0.404 m` |
| `2.01 m` | `0.542 m` |
| `3.00 m` | `0.808 m` |

Yani hover yüksekliğini artırmak nadir kör bölgesini kapatmaz. Montaj açısını
eşiğin üstüne çıkarmak ise yaklaşmayı bozar: `65 derece`de hedef `4 m` yaklaşma
mesafesinde tamamen kadraj dışına çıkar. Sabit montajlı tek kamerayla hem uzak
yaklaşma görüşü hem merkez-görünür hold aynı anda elde edilemez.

Bu bir konfigürasyon hatası değil, sabit montajın geometrik sonucudur.

## Karar

### 1. Hold koşulu merkez görünürlüğüne dayandırılmaz

`acquired -> approach -> settle -> hold -> success` geçişleri ve hover hold
sayacı `vision_aux` içindeki `target_center_xy` kanalına dayandırılmayacaktır.
Görsel süreklilik kapısı yalnız encoder'ın `target_logit` sigmoid skoru
(`target_score`) ve ADR-012'nin sınırlı target-loss grace period'u ile
kurulacaktır.

Gerekçe iki katmanlıdır:

- Hedef merkezi hold pozunda kaçınılmaz biçimde kadraj dışındadır.
- `target_center_xy` etiketi hedefin **görünen** piksellerinin ağırlık
  merkezidir; hedefin geometrik merkezi değildir. Hedef kadrajı taştığında
  etiket görünen kenara kayar. Görünmeyen hedefte ise değer `(0.0, 0.0)`
  sentinel'ine düşer ve "tam merkezde" durumundan ayırt edilemez (REV-017).

Bu nedenle `target_center_xy` yalnız yaklaşma sırasında yumuşak bir hizalama
sinyali olarak kullanılabilir; terminal başarı veya durum geçişi için yeterli
kanıt sayılmaz.

### 2. Başarı kararı privileged ground truth ile verilir

Hover başarısı simülasyonda privileged ground truth ile hesaplanır: aracın
merkezi hedefin güvenli iç bölgesinde, irtifa toleransı içinde, ADR-012
hız/tilt/açısal hız sınırları altında ve kesintisiz `hold_seconds` boyunca.
Bu bilgi actor observation'ına girmez. Görsel kestirim hatası ayrı metrik olarak
raporlanır.

### 3. Montaj açısı ve hover yüksekliği bench ölçümüne kadar değişmez

`pitch_down_deg = 45` korunur. `hover_height_m` için özgün karar `2.0`'dır ve
aşağıdaki 12 Eylül 2026 değişikliğiyle `1.5`'e çekilmiştir. ADR-012 bu değerleri
"fiziksel hedef/araç ölçümü sonrası yeniden kalibre edilecek ilk simülasyon
varsayılanları" olarak tanımlamıştır; ölçüm yapılmadan değiştirilmezler.

Kalibrasyon kaldıraçları ölçülmüş olarak kayda geçirilir. Hedef merkezi,
hedefin arkasında en az `0.542 m` offset'li bir hold pozunda kadrajda kalır.
Güvenli iç bölge yarı-boyutu `0.65 m` olduğundan merkez-görünür bant güvenli
bölgenin arka kenarında yalnız ~`0.11 m` genişindedir. `hover_height_m = 1.5`
seçilirse yakın sınır `0.404 m`'ye düşer ve bant ~`0.25 m`'ye genişler. Bu
değişiklik ADR-012 başarı geometrisini etkilediği için yalnız bench ölçümü veya
ayrı bir ablation kararıyla uygulanır.

#### Değişiklik (12 Eylül 2026): `hover_height_m = 1.5`

Yukarıdaki paragrafın öngördüğü "ayrı karar" verilmiştir. Kaldıraç bir
kalibrasyon tercihi olmaktan çıkıp **ölçülmüş bir blocker** hâline geldiği için
uygulandı (REV-023): D1 kanıt koşusunda deterministik politika `%25`, aynı
politikanın stokastik kipi `%0` başarı üretti. Ayrışmanın nedeni `0.108 m`
genişliğindeki halkanın action gürültüsünü soğuramamasıdır.

```text
hover_height_m = 2.0 -> irtifa 2.01 m, yakın sınır 0.542 m, halka 0.108 m
hover_height_m = 1.5 -> irtifa 1.51 m, yakın sınır 0.407 m, halka 0.243 m
```

Halka `2.25` kat genişler. Değişmeyenler: `pitch_down_deg = 45`, güvenli iç
bölge yarı-boyutu `0.65 m`, `hold_seconds`, ADR-012 hız/tilt/açısal hız
sınırları ve başarı koşulunun kendisi. Yani **başarı kapısı gevşetilmemiştir**;
yalnız o kapının geçilebileceği fiziksel konum kümesi büyütülmüştür.

Bedeli, zemin izinin uzak sınırının `7.46 m`'den `5.60 m`'ye inmesidir. Bu, hover
irtifasından tespit edilebilen en uzak hedef mesafesini kısaltır. ADR-019 arama
geometrisine göre `h_search >= R / 3.71` olduğundan, hover irtifasında
karşılanabilen belirsizlik yarıçapı `R <= 5.60 m` olur (önceki `7.46 m`). D1
(`R = 2 m`) ve D2 (`R = 5 m`) hâlâ hover irtifasında karşılanır; D3'ten itibaren
gereken tırmanma irtifası zaten `4.0 m`'dir, dolayısıyla merdiven etkilenmez.

### 4. Nadir geometrisi config'den türetilir ve test edilir

`CameraConfig` aşağıdaki türetilmiş değerleri yayımlar:

| Üye | Anlam |
|---|---|
| `vertical_half_fov_rad` | Dikey yarım görüş alanı |
| `minimum_pitch_for_nadir_view_rad` | `pi/2 - vfov/2`; nadir eşiği |
| `nadir_in_view` | Mevcut montaj aracın tam altını görüyor mu |
| `ground_swath_m(altitude)` | Nadir'den yakın ve uzak zemin izi sınırı |

`scripts/sweep_camera_visibility.py` hold pozu sonucunu raporlar
(`passing_pitch_deg_with_hold_center`) ve `--require-hold-center` bayrağıyla bunu
kapıya dahil eder. Fiziksel montaj kararı değiştiğinde kapı bu bayrakla
çalıştırılır.

### 5. Hedef hover hacmi sahne üretiminde engelsizdir

Hedefin güvenli iç bölgesi ve irtifa toleransı `HoverEnvelope` olarak
tanımlanır; `vehicle_clearance_m` payıyla genişletilip sahne üretiminde yasak
hacim olur. Bu kısıt opt-in'dir:

- `generate_static_scene(..., target_config=None)` legacy davranıştır ve
  `EbabilNavigation-v0` sahne dizisini bit-bit korur (ADR-008).
- `EbabilVisionNavigationEnv` kısıtı zorunlu kılar.

Zorunlu blocker hedeften uzaklaşacak şekilde başlangıç-hedef hattı üzerinde
`0.5`-`0.85` kesir aralığında kaydırılır. Kaydırma doğrudan rotayı kapatma
görevini bozmaz; koşul yine `direct_path_is_blocked` ile doğrulanır. Kaydırılan
blocker başlangıç noktasını `vehicle_clearance_m` içine alamaz.

### 6. Erişilemez hover geometrisi fail-fast reddedilir

Ortam konum referansını `workspace +- vehicle_clearance_m` içine kırptığı için
hover irtifa bandının bu aralığın dışına çıkması hold koşulunu sessizce
imkânsız hâle getirir. `validate_hover_envelope` bunu kurulumda reddeder.
Hedef merkezi workspace kenarına yakınsa `reachable_safe_region_xy` güvenli iç
bölgenin erişilebilir kısmını hesaplar ve kesişim boşsa hata verir.

Mevcut konfigürasyonda pay incedir ve kayda geçirilir: hover toleransının üst
kenarı `2.21 m`, erişilebilir üst sınır `2.25 m`, kalan pay `0.04 m`.

### 7. Hedef merkezi türetmesi tek kaynaklıdır

Hedef alanın dünya merkezinin hedef noktasının XY'sinden geldiği ve hedef
noktasının irtifasının hedef alan geometrisine girmediği kuralı yalnız
`target_area_from_goal` içinde tanımlıdır. MJCF üretimi, sahne açıklığı, reward,
başarı doğrulaması ve viewer/capture script'leri bu factory'yi kullanır.

## Doğrulama kapısı

1. Nadir eşiğinin `pi/2 - vfov/2` olması ve mevcut montajda `nadir_in_view`
   değerinin `False` olması.
2. Zemin izinin irtifayla doğrusal ölçeklenmesi; irtifa artışının kör bölgeyi
   kapatmaması.
3. Hold pozunda hedef merkezinin kadraj dışı kalması ve merkez-görünür bandın
   güvenli iç bölge içinde ölçülmesi.
4. Hover irtifa bandının erişilebilirliği ve erişilemez config'in reddi.
5. Güvenli iç bölgenin workspace kenarında kırpılması, workspace dışında reddi.
6. Legacy sahne dizisinin parmak izi ile değişmediğinin gösterilmesi.
7. Hedef-farkında sahnelerde hover hacminin engelsiz, doğrudan rotanın kapalı ve
   başlangıcın engelsiz kalması.
8. Görünmeyen hedef merkezinin sentinel belirsizliğinin testle sabitlenmesi.

## Sonuçlar ve sınırlar

- Bu ADR terminal hover state machine'ini ve reward bileşenlerini tanımlamaz;
  onların üzerinde çalışacağı geometriyi ve kapılama kuralını sabitler.
- Sayılar simülasyon geometrisinden türetilmiştir. Gerçek IMX477 lens
  distorsiyonu, FFC/carrier montaj toleransı ve titreşim altındaki efektif FOV
  ölçülmemiştir; ADR-011 bench kapısı geçerliliğini korur.
- Hedef merkezini hold sırasında görüşte tutmak isteniyorsa seçenekler gimbal,
  ikinci kamera veya `hover_height_m` düşürülmesidir. Üçüncüsü 12 Eylül 2026'da
  uygulandı (`2.0 -> 1.5`); bu hold halkasını genişletir fakat merkezi kadraja
  **sokmaz**, çünkü yakın sınır (`0.407 m`) hâlâ hedef yarı-boyutunun
  (`1.0 m`) içindedir. Gimbal ve ikinci kamera donanım kararıdır ve bu ADR
  kapsamında uygulanmamıştır.
- Hover hacminin engelsiz olması bir güvenlik garantisi değildir; yalnız eğitim
  görevinin çözülebilir olmasını sağlar.
