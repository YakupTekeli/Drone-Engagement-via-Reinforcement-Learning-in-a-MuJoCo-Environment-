# Kod ve Dokümantasyon İncelemesi Bulguları

**İlk inceleme tarihi:** 10 Eylül 2026  
**Son inceleme tarihi:** 12 Eylül 2026 (waypoint oracle ablation'ı)  
**Kapsam:** Paketleme, Gymnasium ortam sözleşmesi, seed akışı, sahne geometrisi,
test altyapısı, Faz 6.5 perception hattı, hedef alan/hover geometrisi ve proje
belgeleri.  
**Not:** Projenin Git repository olmaması kullanıcı kararıyla problem kapsamından çıkarılmıştır.

## Durum özeti

| Kimlik | Öncelik | Bulgu | Durum |
|---|---|---|---|
| REV-001 | Yüksek | Wheel paketi MJCF ve TOML kaynaklarını içermiyor | Çözüldü |
| REV-002 | Yüksek | Sensör rastgeleliği episode seed yerine scene seed'e bağlı | Çözüldü |
| REV-003 | Orta | Doğrudan rota kontrolü yalnızca XY ve konservatif yaklaşık geometri kullanıyor | Çözüldü |
| REV-004 | Orta | ADR-008 ground-truth kullanımıyla gerçek observation davranışı çelişiyor | Çözüldü |
| REV-005 | Düşük | Episode uzunluğu ve policy frekansı sınırları geç veya yanlış doğrulanıyor | Çözüldü |
| REV-006 | Düşük | Faz 0 araç hedefleri ile mevcut test/lint dokümantasyonu uyuşmuyor | Çözüldü |
| REV-007 | Yüksek | Tekrar gözlenen test split'i bağımsız akademik test olarak sunuluyor | Çözüldü |
| REV-008 | Yüksek | Başarısız perception kalite kapısı süreç kodu 0 ve seçilmiş checkpoint üretiyor | Çözüldü |
| REV-009 | Yüksek | Offline ve online RGB preprocessing için tek bir sözleşme/API yok | Çözüldü |
| REV-010 | Yüksek | Engel/hedef renkleri semantik kısa yol oluşturuyor; kalite kapısı geometri genellemesini kanıtlamıyor | Çözüldü |
| REV-011 | Orta | Geçerli sayılan encoder çözünürlükleri heatmap boyutunda runtime hatası üretiyor | Çözüldü |
| REV-012 | Orta | Quality/checkpoint artefact'larında dataset kimliği ve taşınabilir bütünlük bilgisi eksik | Çözüldü |
| REV-013 | Orta | Bildirilen split oranları ile üretilen sahne oranları uyuşmuyor | Çözüldü |
| REV-014 | Yüksek | Hold pozunda hedef merkezi kadraj dışında; sweep kapısı bunu denetlemiyor | Çözüldü |
| REV-015 | Yüksek | Hedef alan merkezi örtük türetiliyor; hover bandının erişilebilirliği doğrulanmıyor | Çözüldü |
| REV-016 | Yüksek | Sahne üreticisi hedefin hover hacmini engelsiz bırakmıyor | Çözüldü |
| REV-017 | Orta | Hedef merkezi sentinel değeri görünürlükten ayırt edilemiyor | Çözüldü |
| REV-018 | Yüksek | Encoder dataset'i tek rota geometrisinde üretilmiş | Çözüldü |
| REV-019 | Yüksek | Terminal hover'ın dayandığı hold pozu eğitim dağılımında yok | Çözüldü |
| REV-020 | Yüksek | Collision-risk kaybında pozitif sınıf ağırlığı yok | Çözüldü (kalibrasyon REV-021'e bağlı) |
| REV-021 | Yüksek | Risk head'i zorlaştırılmış dağılımda altı kapıyı birlikte geçemiyor | Çözüldü (kapasite; v10 onaylandı) |
| REV-022 | Orta | ADR-017 hold reward üst sınırı iddiası yanlış | Çözüldü (oran ablation'ı açık) |
| REV-023 | Yüksek | Uygulanabilir hold bandı `0.108 m`; başarı `3.6 cm` ile belirleniyor | Açık (hover geometrisi kararı) |
| REV-024 | Yüksek | Localization hatası hiç test edilmedi; ilk drift ablation'ı yanıltıcı | Kısmen kapatıldı (ADR-019 kapsamı) |
| REV-025 | Yüksek | `grace_seconds` yaw ile toparlanma süresinden kısa; `target_lost` abort'u öğrenmeyi kesiyor | Düzeltildi; etkisi ölçülüyor |
| REV-026 | Yüksek | `collision_penalty` sabit, `target_progress` rota ile ölçekleniyor; çarpışma oranı eğitimde arttı | Açık (bağlayıcı) |
| REV-027 | Yüksek | Hedef görünürlüğü reward'da temsil edilmiyor; kısa grace kazara şekillendirme yapıyordu | Açık |

## 11 Eylül 2026 — Terminal hover öncesi inceleme

Bu inceleme, ADR-012 terminal hover iş dilimine geçmeden önce hedef alan
geometrisi, sahne üretimi ve perception sözleşmesini denetler. REV-014–REV-017
düzeltmeleri aynı iş diliminde uygulandı; REV-018 kapsam gereği D2 curriculum
iş paketine bırakıldı. Aşağıdaki sayılar `.venv/bin/python` ile ölçülmüştür.

## REV-014 — Hold pozunda hedef merkezinin kadraj dışında kalması

`scripts/sweep_camera_visibility.py` kamera montaj açısı kapısını yalnız
`visible_fraction >= 0.25` ölçütüyle değerlendiriyor. Script `center_visible`
değerini hesaplayıp yazdırıyor fakat kapıya dahil etmiyor. Bu nedenle
"`40/45/50 derece` adayları projection kapısını geçti" kaydı hover anchor
üzerindeki hold pozunu hiç denetlememiş oluyor.

Ölçüm, sabit montajlı kameranın hedef merkezini hold pozunda göremediğini
gösteriyor:

```text
$ .venv/bin/python scripts/sweep_camera_visibility.py \
    --pitch-deg 45 60 65 --approach-distance-m 4 0
pitch_deg,distance_m,visible_fraction,center_visible,nadir_in_view,swath_near_m,swath_far_m
45.0,4.00,1.00,True,False,0.542,7.460
45.0,0.00,0.50,False,False,0.542,7.460
60.0,4.00,0.50,False,False,0.003,3.470
65.0,4.00,0.00,False,True,0.000,2.862
65.0,0.00,0.50,True,True,0.000,2.862
minimum_pitch_for_nadir_view_deg=60.08
```

Geometrik neden analitiktir: nadir noktasının kadrajda olması için
`pitch_down >= 90 - vfov/2 = 60.08 derece` gerekir. Güncel değer
`45 + 29.92 = 74.92 derece` olduğundan aracın tam altı hiçbir irtifada görünmez.
Zemin izinin yakın sınırı `irtifa * 0.27` olarak ölçeklendiği için hover
yüksekliğini artırmak kör bölgeyi kapatmaz (`2.01 m` için `0.542 m`, `3.0 m`
için `0.808 m`). Pitch'i eşiğin üstüne çıkarmak ise yaklaşmayı bozar: `65
derece`de hedef `4 m` mesafede tamamen kadraj dışına çıkar
(`visible_fraction 0.00`).

**Etki:** Terminal hover state machine'inin `settle`/`hold` koşulu
`target_center_xy` üzerine kurulursa hedef üstünde kaçınılmaz biçimde
kaybedilir. Sabit pitch'li tek kamerayla hem `4 m` yaklaşma görüşü hem
merkez-görünür hold aynı anda sağlanamaz.

**Kabul ölçütleri:**

- Nadir eşiği ve zemin izi konfigürasyondan türetilebilir ve test edilebilir
  olmalı.
- Sweep kapısı hold pozu sonucunu raporlamalı ve talep edildiğinde kapıya
  dahil etmeli.
- Hold koşulunun merkez görünürlüğüne dayanamayacağı ADR ile sabitlenmeli.

**Çözüm:** `CameraConfig` üzerine `vertical_half_fov_rad`,
`minimum_pitch_for_nadir_view_rad`, `nadir_in_view` ve `ground_swath_m(altitude)`
eklendi. `sweep_camera_visibility.py` artık `nadir_in_view`, zemin izi sınırları
ve `passing_pitch_deg_with_hold_center` listesini yazıyor; `--require-hold-center`
bayrağıyla hold pozu kapıya dahil ediliyor. Karar ADR-016 ile sabitlendi: hold
koşulu `target_score` ve privileged ground truth ile kurulur, merkez
görünürlüğüne dayandırılmaz. Montaj açısı ve hover yüksekliği fiziksel bench
ölçümüne kadar değiştirilmedi.

**Doğrulama:** `tests/test_hover_geometry.py::CameraNadirGeometryTests` yedi
test: eşiğin `60.08 derece` olması, mevcut montajda `nadir_in_view=False`, eşik
altı/üstü davranış, zemin izinin irtifayla doğrusal ölçeklenmesi, hold pozunda
`swath_near = 0.542 m` ile merkezin kadraj dışı kalması, merkez-görünür bandın
güvenli iç bölge içinde `0.15 m`'den ince olması ve geçersiz irtifanın
reddedilmesi.

## REV-015 — Hedef alan geometrisinin örtük türetilmesi ve doğrulanmaması

Hedef alanın dünya merkezi yalnız `build_scene_model` içinde
`scene.goal_world_m[:2]` ifadesinden örtük olarak türetiliyordu; `StaticScene`
hedef alanı taşımıyor ve hiçbir ortak factory yok. Üç viewer/capture script'i ise
`TargetArea(tuple(scene.goal_world_m[:2]), target_config)` ifadesini
`# type: ignore` ile tekrarlıyordu. Aynı kural dört yerde kopyalanmış durumda.

Ayrıca hedef nokta ile hover anchor arasındaki dikey uyumsuzluk hiçbir yerde
doğrulanmıyor. `scene.goal_world_m[2]` fixed modda `1.0 m`, random endpoint
modda `0.75-1.75 m`; hover anchor ise `surface_z + hover_height = 2.01 m`.
`target_area.toml` içindeki `hover_height_m`, `altitude_tolerance_m`,
`maximum_linear_speed_m_s`, `maximum_angular_rate_rad_s` ve `hold_seconds`
alanlarını hiçbir ortam tüketmiyor; yalnız kamera script/testleri okuyor.

En kritik nokta: ortam konum referansını
`workspace +- vehicle_clearance_m` içine kırpıyor, yani erişilebilir üst sınır
`2.5 - 0.25 = 2.25 m`. Hover toleransının üst kenarı `2.01 + 0.20 = 2.21 m`
olduğundan yalnız `0.04 m` pay kalıyor. `hover_height_m = 2.1` gibi makul
görünen bir değer hold koşulunu sessizce imkânsız hâle getirirdi.

**Etki:** Hedef merkezi türetmesi dört yerde ayrı yaşadığı için hover slice'ında
sessiz tutarsızlık riski var. Erişilemez hover bandı hiçbir hata üretmeden
sürekli başarısız eğitim koşusu üretebilir.

**Kabul ölçütleri:**

- Hedef merkezi türetmesi tek bir fonksiyonda tanımlanmalı ve tüm tüketiciler
  bunu kullanmalı.
- Hover irtifa bandının erişilebilirliği fail-fast doğrulanmalı.
- Hedef merkezi workspace kenarına yakınken güvenli iç bölgenin erişilebilir
  kısmı hesaplanabilmeli, boşsa hata vermeli.

**Çözüm:** `scenes/target.py` içine `target_area_from_goal` tek kaynaklı
factory'si ve ona delege eden `target_area_from_scene` eklendi;
`build_scene_model` ve üç script bu factory'ye bağlandı ve `# type: ignore`
ifadeleri kalktı. `HoverEnvelope` veri tipi (`inflated`, `intersects_box`) ve
`TargetArea.hover_envelope` eklendi. `validate_hover_envelope` hover toleransının
erişilebilir workspace dışına çıkmasını; `reachable_safe_region_xy` güvenli iç
bölgenin workspace ile kesişimini hesaplayıp boş kesişimi reddediyor.
`EbabilVisionNavigationEnv` kurulumda `validate_hover_envelope`, her reset'te
`reachable_safe_region_xy` çağırıyor ve `target_area` özelliğini yayımlıyor.

**Doğrulama:** `tests/test_hover_geometry.py::HoverEnvelopeTests` dokuz test:
merkezin yalnız hedef XY'sinden gelmesi ve hedef irtifasını yok sayması, iki
factory'nin eşitliği, zarf sınırlarının güvenli bölge ve irtifa toleransıyla
eşleşmesi, kesişim testinin katı eşitsizlik kullanması, gönderilen config'in
geçmesi, `hover_height_m = 2.1` değerinin açık hata vermesi, workspace kenarında
kırpma ve workspace dışında reddetme.

## REV-016 — Sahne üreticisinin hedef hover hacmini korumaması

`generate_static_scene` engelleri yalnız başlangıç ve hedef **noktalarına** karşı
`endpoint_clearance_m = 0.40` ile denetliyor. Hedefin `2 x 2 m` ayak izi ve
üzerindeki hover hacmi korunmuyor. Engeller zemine oturuyor ve yarı-yükseklik üst
sınırı `1.20 m` olduğu için tepeleri `2.40 m`'ye kadar çıkabiliyor, yani
`1.81-2.21 m` hover bandını kesebiliyorlar.

`random_obstacles` + `fixed` modunda 300 seed ölçümü:

```text
hover anchor kolonu (merkez üstü) bloke   :   0/300
güvenli iç bölge kolonuna engel girmiş    : 131/300  (%44)
engel XY'si 2x2 m hedef ayak izine taşmış : 245/300  (%82)
```

Anchor'ın kendisi `endpoint_clearance_m` 3B mesafe kullandığı için dolaylı olarak
korunuyor. Ancak ADR-012 başarısı anchor noktası değil güvenli iç bölgedir; bu
bölge sahnelerin %44'ünde kısmen fiziksel olarak tıkalı.

**Etki:** Hover eğitimi, başarı geometrisinin bir kısmı engelle dolu sahnelerde
yürütülür. Politika çözülemeyen episode'lardan ceza öğrenir ve başarı oranı
sahne şansına bağlı olur.

**Kabul ölçütleri:**

- Hedef hover hacmi araç açıklığı payıyla engelsiz olmalı.
- Legacy `EbabilNavigation-v0` sahne dizisi ADR-008 gereği bit-bit korunmalı.
- Kısıt zorunlu blocker'ın doğrudan rotayı kapatma görevini bozmamalı.
- Geometrik olarak çözülemeyen durum sessiz ihlal değil açık hata üretmeli.

**Çözüm:** `generate_static_scene` opt-in `target_config` parametresi aldı;
`None` varsayılanı legacy davranışı değiştirmez. Verildiğinde hedefin
`HoverEnvelope`'ı `vehicle_clearance_m` ile genişletilip hem rastgele engeller
hem zorunlu blocker için yasak hacim olarak uygulanıyor. Zorunlu blocker orta
noktada sabit olduğu için ilk uygulama rastgele endpoint'te sahnelerin %17'sini
üretilemez hâle getirdi; bu nedenle blocker artık hedeften uzaklaşacak şekilde
başlangıç-hedef hattı üzerinde `0.5`-`0.85` kesir aralığında kaydırılıyor.
`EbabilNavigationEnv._scene_target_config` hook'u legacy için `None`,
`EbabilVisionNavigationEnv` için hedef config'i döndürüyor.

İki yan etki uygulama sırasında tespit edilip kapatıldı:

1. Blocker hover zarfına tam teğet yerleştiğinde kesişim kararı floating-point
   yuvarlamaya kalıyordu ve 300 seed'de 38 ihlal üretiyordu. Ayrım mesafesine
   `HOVER_ENVELOPE_SEPARATION_EPSILON_M = 1e-6` payı eklendi.
2. Blocker'ı başlangıca doğru kaydırmak, başlangıç noktasının engelin içinde
   kalmasını mümkün kıldı (legacy orta nokta yerleşiminde bu risk yoktu).
   Hedef-farkında yolda blocker `_point_to_box_distance(start) <
   vehicle_clearance_m` koşulunda reddediliyor.

**Doğrulama:** Legacy parmak izi `3 mod x 4 endpoint modu x 60 seed = 720 sahne`
üzerinde `75274526eae51f52475ea4bf7ae0d0f44f12d7c2f4cc3f6f84418a93c390dfe6`
değerinde kaldı; test içinde 240 sahnelik kısa varyantı kullanılıyor.
Hedef-farkında üretim 2400 sahne (2 mod x 4 endpoint modu x 300 seed) üzerinde:

```text
hover hacmi ihlali          : 0/2400
başlangıcı yutan blocker    : 0/2400
doğrudan rotası açık kalan  : 0/2400
üretilemeyen sahne          : 4/2400  (%0.17, açık RuntimeError)
```

Kalan 4 vaka `2.16 m` gibi kısa endpoint ayrımlarında blocker'ın hem rotayı
kapatıp hem hover hacmini boşaltamadığı gerçek geometrik çözümsüzlüktür ve
nedeni adlandıran bir hata mesajı üretir.

## REV-017 — Hedef merkezi sentinel değerinin görünürlükten ayırt edilememesi

`derive_supervision`, hedef görünmediğinde `target_center_xy` değerini
`(0.0, 0.0)` yazıyor. Bu değer "hedef tam görüntü merkezinde" durumundan
bit-bit ayırt edilemez. Encoder de aynı semantiği öğreniyor, yani runtime
`vision_aux[1:3]` kanalı tek başına görünürlük kanıtı değildir.

REV-014 ile birlikte ikinci bir semantik sınır ortaya çıkıyor: hedef görünürken
merkez, hedefin **görünen** piksellerinin ağırlık merkezidir. Hold pozunda
hedefin yakın yarısı kadraj dışında kaldığı için etiket görünen kenara kayar ve
hedefin gerçek geometrik merkezini göstermez. Development-OOD üzerinde ölçülen
`target_center_mean_error = 0.226` (normalize `[-1, 1]`, ~`36 px`/`320 px`) bu
kanalın çözünürlük sınırını da gösteriyor.

**Etki:** Terminal state machine geçişleri veya `target_alignment` reward'u
merkez kanalına dayandırılırsa hem görünmeyen hedef "merkezde" sayılır hem de
hold pozundaki sistematik kayma ödüllenir.

**Kabul ölçütleri:**

- Sentinel belirsizliği ve görünen-piksel semantiği sözleşmede açıkça yazılı
  olmalı.
- Görünürlük kapısının yalnız `target_visible`/`target_score` olduğu testle
  sabitlenmeli.

**Çözüm:** Dataset schema'sı ve onaylı encoder zinciri değiştirilmedi; değişiklik
sözleşme ve test düzeyinde. `derive_supervision` docstring'i sentinel
belirsizliğini, görünen-piksel semantiğini ve nadir kör bölgesi etkisini açıkça
tanımlıyor. ADR-016 hold/geçiş koşullarının `target_score` ile kapılanmasını
zorunlu kılıyor.

**Doğrulama:** `tests/test_hover_geometry.py::SupervisionSentinelTests` iki test:
görünmeyen hedefin merkezinin tam kadraj merkezindeki görünür hedefle aynı değeri
üretmesi (`(0.0, 0.0)`), ve hedefin yalnız alt kenarı görünürken merkezin
`> 0.7` değerine kayması.

## REV-018 — Encoder eğitim dağılımının tek rota geometrisinde olması

`src/ebabil/configs/training/vision_dataset.toml` içinde
`endpoint_mode = "fixed"`. Bu nedenle `vision-dataset-v4b`'nin 92 sahnesinin
tamamında başlangıç `(-2, 0, 1)`, hedef `(2, 0, 1)`; hedef her zaman aynı dünya
XY'sinde ve yaklaşma her zaman `+x` yönünde. Poz örneklemesi progress `0-1`,
lateral `+-1.25 m`, irtifa `0.80-2.10 m` ve yaw jitter `+-35 derece` uygularken
rota geometrisi tektir.

`vision-encoder-v9`'un bütün kalite sayıları bu tek geometri üzerinde ölçüldü:

| Metrik | validation | development-OOD | audit |
|---|---:|---:|---:|
| target precision | 0.845 | 0.918 | 0.863 |
| target recall | 0.980 | 0.929 | 0.986 |
| target center error | 0.184 | 0.226 | 0.189 |
| free-space MAE | 0.218 | 0.257 | 0.255 |
| risk balanced accuracy | 0.769 | 0.688 | 0.739 |

**Etki:** D2 seviyesinde `endpoint_mode` rastgeleye açıldığı anda encoder,
held-out kümesi görev dağılımını kapsamayan bir modele dönüşür. Hover
politikasının perception tabanı doğrulanmamış bölgede çalışır. D0-V ve D1 sabit
legacy geometri kullandığı için bu etki onlara ulaşmaz (ADR-010).

**Kabul ölçütleri:**

- Dataset rastgele endpoint'le yeniden üretilmeli veya rastgele endpoint'li ayrı
  bir audit split'i eklenmeli.
- Yeniden üretim ADR-014 approval zincirini baştan çalıştırmalı: yeni dataset
  hash'leri, üç-seed quality ve dokunulmamış audit.
- Karşılaştırma için mevcut `vision-encoder-v9` sonuçları referans olarak
  korunmalı.

**Çözüm:** `vision_dataset.toml` içinde `endpoint_mode` `"fixed"` yerine
`"random_both"` yapıldı ve `split_salt` `"ebabil-vision-v4c"` olarak yenilendi.
Salt yenilemesi REV-007 disiplininin gereğidir: audit split'i yeni scene
seed'leriyle üretilir, böylece önceki audit koşusunun sahneleri tekrar
kullanılmaz. Schema `vision-dataset-v4` korundu; değişiklik config düzeyindedir
ve `config_sha256` üzerinden ayırt edilir (v4 -> v4b -> v4c deseni).

Üretilen dataset `runs/vision_dataset_v4c_20260911`: `736` kayıt,
`512/64/64/96` train/validation/development-OOD/audit, leakage denetimi temiz.
Rota geometrisi artık tek değil:

```text
endpoint_mode=fixed        : rota uzunluğu hep 4.00 m
endpoint_mode=random_both  : rota uzunluğu 2.08-4.65 m (20 seed, ort 3.06)
```

Poz fallback oranı her iki modda `0/160` ölçüldü, yani rastgele endpoint poz
örneklemesini bozmuyor.

**Doğrulama:** `tests/test_vision_dataset.py::HoldPoseSamplingTests`
`endpoint_mode` değerinin `fixed` olmadığını ve 30 seed üzerinde 20'den fazla
ayrık rota uzunluğu üretildiğini sabitliyor. Yeni dağılımda üç-seed quality ve
bağımsız audit yeniden çalıştırıldı; sonuçlar REV-020 ve ilerleme günlüğünde
ayrıntılıdır.

Dataset yenilemesi iki yeni bulgu ortaya çıkardı ve ikisi de aynı iş diliminde
kapatıldı: hold pozunun eğitim dağılımında bulunmaması (REV-019) ve
collision-risk kaybının sınıf dengesi telafisi olmaması (REV-020). İkincisi
yalnız prior değiştiği için görünür hâle geldi; `vision-dataset-v4b` üzerinde
sessizce gizli kalıyordu.

**Not:** REV-016 düzeltmesi dataset üretimini etkilemez. Dataset
`generate_static_scene` çağrısını `target_config` olmadan yapar, bu nedenle
`vision-dataset-v4b` hash zinciri ve approval manifest geçerliliğini korur.
Ortam sahneleri artık dataset sahnelerinin hover hacmi boşaltılmış bir alt
kümesidir.

## REV-019 — Hold pozunun eğitim dağılımında neredeyse hiç bulunmaması

ADR-017 terminal hover durum makinesi, hedef üstü hold sırasında görsel
sürekliliği `target_score >= loss_score` koşuluyla kapılar ve `grace_seconds`
aşıldığında episode'u abort eder. Yani `3.0 s` hold başarısı, encoder'ın hover
zarfı içindeki pozlarda kararlı yüksek skor üretmesine bağlıdır.

Ölçüm, `vision-dataset-v4b` poz dağılımının bu bölgeyi neredeyse hiç
kapsamadığını gösterdi. Rota ailesi start-goal hattı boyunca `progress`, yanal
offset ve irtifa örnekler; hover anchor'ın yakın komşuluğuna denk gelme olasılığı
düşüktür:

```text
100 sahne x 8 poz, hover anchor'a +-0.35 m XY / +-0.30 m Z yakınlık:
endpoint_mode=fixed        : 5/800  (%0.62)
endpoint_mode=random_both  : 9/800  (%1.12)
```

`512` train örneğinde bu yaklaşık `3-6` kare demektir.

**Etki:** ADR-017'nin abort mantığı, encoder'ın istatistiksel olarak
doğrulanmamış bir girdi bölgesine dayanır. Hold sırasında skor `loss_score`
altına düşerse politika hatasız uçsa bile `target_lost` abort'u alır ve terminal
görev öğrenilemez. Bu, terminal hover dilimiyle birlikte ortaya çıkan yeni bir
bağımlılıktır; REV-018 dataset yenilemesi yapılırken kapatılmazsa dataset iki
kez üretilmek zorunda kalır.

**Kabul ölçütleri:**

- Hover zarfı içindeki pozlar açık bir poz ailesi olarak örneklenmeli ve oranı
  konfigüre edilebilir olmalı.
- Hold pozlarında hedef görünürlüğü zorlanmamalı; etiket segmentation
  maskesinden dürüstçe türetilmeli. ADR-016 nadir kör bölgesi nedeniyle bir
  kısmının negatif etiketlenmesi beklenen davranıştır.
- Hold ailesi negatif yaw dalına dahil edilmemeli; aksi halde hold pozunun
  gerçek görünürlük dağılımı ölçülemez.
- Poz örneklemesi deterministik kalmalı.

**Çözüm:** `VisionDatasetConfig` içine `hold_pose_fraction` eklendi.
`make_capture_spec` iki poz ailesi üretiyor: rota ailesi (mevcut davranış) ve
bu olasılıkla seçilen hold ailesi. Hold ailesi ADR-012 hover zarfından örnekler
(güvenli iç bölge XY'si, hover anchor irtifası `+-` irtifa toleransı), sahne
açıklığını aynı `_point_is_clear` denetiminden geçirir, yaw'ı rota yönünde tutar
ve negatif yaw dalını uygulamaz. `hold_pose_fraction = 0.20` seçildi;
`hold_pose_fraction + target_negative_fraction < 1.0` koşulu config
doğrulamasında uygulanıyor.

**Doğrulama:** `vision-dataset-v4c` üretiminde hover zarfı içindeki poz oranı
`%1.1`'den `%23.4`'e (`172/736`) çıktı. Üretilen dataset üzerinde ölçülen
gerçek görünürlük dağılımı:

| Split | Aile | n | Görünür | Oran | Ort. hedef alanı |
|---|---|---:|---:|---:|---:|
| train | hold | 126 | 117 | 92.9% | 0.110 |
| train | route | 386 | 330 | 85.5% | 0.220 |
| validation | hold | 10 | 9 | 90.0% | 0.140 |
| validation | route | 54 | 46 | 85.2% | 0.207 |
| development_ood | hold | 19 | 17 | 89.5% | 0.109 |
| development_ood | route | 45 | 37 | 82.2% | 0.234 |
| audit | hold | 17 | 15 | 88.2% | 0.125 |
| audit | route | 79 | 68 | 86.1% | 0.175 |

Sonuç terminal hover tasarımı için olumlu: hold pozlarında hedef rota
pozlarından **daha sık** görünür (`%88-93` vs `%82-86`), fakat kapladığı alan
yarısı kadardır (`~0.11` vs `~0.22`). Bu, ADR-016'nın hover anchor'da
`visible_fraction = 0.50` ölçümüyle tutarlıdır: hedefin yakın yarısı nadir kör
bölgesinde kalır, uzak yarısı kadrajda kalır. Dolayısıyla `target_score` kapısı
hold sırasında çalışabilir; kalan `%7-12` görünmez hold pozu güvenli iç bölgenin
hedeften uzağa bakan kenarındaki pozlardır ve `grace_seconds` payının varlığını
gerekçelendirir.

`tests/test_vision_dataset.py::HoldPoseSamplingTests` beş test: gönderilen
config'in hover zarfını örneklemesi, `hold_pose_fraction = 0` ile ailenin
kapanması, determinizm, `endpoint_mode`'un rastgele olması ve oran toplamı
doğrulaması.
## REV-020 — Collision-risk kaybının sınıf dengesi telafisi olmaması

`perception_loss` içindeki `collision_risk_loss` çıplak
`binary_cross_entropy_with_logits` kullanıyordu. Aynı fonksiyondaki iki mask
kaybı `pos_weight=mask_positive_weight` (`3.0`) uygularken risk head'inde
hiçbir pozitif sınıf ağırlığı yoktu. Bu nedenle risk head'i doğrudan eğitim
prior'ını takip ediyordu.

Hata `vision-dataset-v4b` üzerinde gizli kaldı: o dataset'te risk-pozitif oranı
`%60.8` (train) olduğu için çoğunluk sınıfı zaten pozitifti ve recall yüksek
görünüyordu. REV-018/REV-019 dataset yenilemesi bu oranı düşürdüğünde kapı
düştü:

```text
risk-pozitif oranı        v4b (sabit endpoint)   v4c (rastgele + hold pozu)
train                              60.81%                     49.02%
development_ood                    56.77%                     46.88%
```

`vision_quality_v17` (v4c, düzeltme öncesi) üç seed'in tamamında ve her iki
development split'inde `risk_recall` kapısını düşürdü:

| Metrik | v16 (v4b) | v17 (v4c, düzeltme öncesi) | Eşik |
|---|---:|---:|---|
| `risk_recall` validation | 0.728 | **0.590** | `>= 0.70` |
| `risk_recall` dev-OOD | 0.813 | **0.578** | `>= 0.70` |
| `risk_precision` validation | 0.851 | 0.844 | `>= 0.65` |
| `risk_specificity` dev-OOD | 0.562 | 0.856 | `>= 0.50` |

İmza nettir: recall çökerken precision ve specificity yükselmiş, yani model
"risk yok" demeye kaymıştır.

**Etki:** Kaçırılan çarpışma riski (false negative), yanlış alarmdan (false
positive) daha pahalıdır. Ağırlıksız BCE bu asimetriyi modellemediği için risk
kanalının davranışı dataset prior'ına bağlı hâle geliyor; dataset dağılımı her
değiştiğinde risk kanalı sessizce yeniden kalibre olmak zorunda kalıyor.

**Kabul ölçütleri:**

- Risk kaybı pozitif sınıf ağırlığı almalı ve bu ağırlık konfigüre edilebilir
  olmalı.
- Karar eşiği (`collision_risk_score_threshold`) kapıyı geçirmek için
  değiştirilmemeli; aksi halde düzeltme eşik ayarına dönüşür.
- Kapı, üç seed ve iki development split'inin tamamında marjla geçmeli.

**Çözüm:** `VisionTrainingConfig` içine `collision_risk_positive_weight`
eklendi ve risk BCE'sinde `pos_weight` olarak uygulanıyor;
`collision_risk_positive_weight = 2.0` seçildi. Mevcut mask kayıplarının
`pos_weight` deseni izlendi.

Karar eşiği `collision_risk_score_threshold = 0.62` **değiştirilmedi**. Önce
eşik taraması denendi ve bilinçli olarak reddedildi: düzeltme olmadan `0.45`
eşiği bir seed'de `risk_recall = 0.689` ile düşüyor, `0.40` eşiği ise bir
seed/split kombinasyonunda `risk_precision` marjını `+0.007`'ye indiriyordu. Bu,
kapıyı kök nedeni çözmeden geçirmek olurdu.

**Doğrulama:** `pos_weight = 2.0` ve değişmeyen `0.62` eşiğiyle tek seed
taraması, mevcut operasyon noktasında sağlıklı marj gösterdi:

| Split | recall | precision | specificity | marj (rec/prec/spec) |
|---|---:|---:|---:|---|
| validation | 0.752 | 0.810 | 0.747 | `+0.052 / +0.160 / +0.247` |
| development_ood | 0.733 | 0.702 | 0.725 | `+0.033 / +0.052 / +0.225` |

Eşik seçimi ve pozitif ağırlık yalnız `validation` ve `development_ood`
üzerinde yapıldı; `audit` split'i dokunulmadı ve bağımsız kapı olarak
çalıştırıldı.
## REV-021 — Risk head'inin zorlaştırılmış dağılımda altı kapıyı birlikte geçemmesi

REV-018/REV-019 dataset yenilemesi ve REV-020 `pos_weight` düzeltmesi sonrasında
`vision-dataset-v4c` üzerinde üç ayrı üç-seed quality koşusu çalıştırıldı.
Hiçbiri altı risk kapısını (`risk_precision`, `risk_recall`,
`risk_specificity` ve bunların grayscale ablation eşdeğerleri) üç seed ve iki
development split'inin tamamında geçemedi:

| Koşu | `pos_weight` | Sonuç | Düşen kapı |
|---|---:|---|---|
| `vision_quality_v17` | yok (ağırlıksız) | Düştü | `risk_recall` — 3/3 seed, iki split |
| `vision_quality_v18` | `2.0` | Düştü | `risk_recall` — seed `20260912` dev-OOD (`0.678`) |
| `vision_quality_v19` | `3.0` | Düştü | `grayscale_risk_precision` — seed `20260911` ve `20260913` dev-OOD |

Ölçümler, ağırlığın precision-recall eğrisi üzerinde bir operasyon noktasını
kaydırdığını, modelin ayırt etme gücünü artırmadığını gösteriyor:

```text
pos_weight   dev-OOD risk_recall (min)   dev-OOD risk_precision (min)
yok                      0.567                       0.754
2.0                      0.678                       0.670
3.0                      0.744                       0.652  (grayscale: <0.65)
```

Bağlayıcı kısıt grayscale ablation precision'ıdır. Bu ablation REV-010 gereği
modelin rengi kısa yol olarak kullanmadığını kanıtlamak için vardır; renkten
arındırılmış girdide risk kanalının precision'ı, recall'ı yukarı çeken her
ağırlıkta eşiğin altına düşüyor.

**Etki ve kapsam düzeltmesi:** `vision-dataset-v4c` dağılımında onaylanabilir
bir encoder üretilemedi.

Bu bulgu ilk yazımında D0-V'nin ön koşulu olarak kaydedilmişti; bu **yanlıştı**.
ADR-010 curriculum merdiveninde D0-V "legacy geometriyi `vision-v1` ile yeniden
öğrenmek" olarak tanımlıdır, yani sabit `4-6 m` geometridir. "Değişken
endpoint'ler" **D2** (`15-50 m`) seviyesinde gelir. Onaylı `vision-encoder-v9`
sabit geometride eğitildiği için D0-V'nin gerektirdiği encoder tam olarak odur.

Dolayısıyla REV-021 **D0-V'yi bloke etmez; D2'yi bloke eder.** Rastgele rota
geometrisine geçildiğinde onaylanabilir bir perception tabanı bulunmadığı için
D1 sonrası seviyeler bu bulgu kapanmadan açılamaz.

**Güncelleme (12 Eylül 2026):** Bulgu kapatıldı; `vision-encoder-v10`
rastgele rota geometrisinde onaylandı. Bu blocker artık yoktur.

**Bu iş diliminde bilinçli olarak yapılmayanlar:**

- Kapı eşikleri (`minimum_risk_recall`, `minimum_risk_precision`,
  `minimum_risk_specificity`) **değiştirilmedi**. Kapıyı geçirmek için eşik
  düşürmek REV-007/REV-008'in engellemek için var olduğu davranıştır.
- Karar eşiği `collision_risk_score_threshold = 0.62` **değiştirilmedi**. Önce
  eşik taraması denendi; düzeltme olmadan `0.45` bir seed'de düşüyor, `0.40` ise
  bir kombinasyonda precision marjını `+0.007`'ye indiriyordu.
- `pos_weight` araması `2.0` ve `3.0` ile **iki denemede durduruldu**.
  `2.0`-`3.0` arasında bir değer aramak, üç seed x altı kontrol x iki split
  üzerinde şansa bağlı bir geçiş bulmak olurdu; uygulanabilir bölgenin razor-thin
  olması bulgunun kendisidir.
- `audit` split'i **hiç tüketilmedi**. `run_vision_audit` başarısız quality
  raporunu reddeder; `runs/vision_quality_v1{7,8,9}_20260911` dizinlerinde
  `audit_consumed.json` yoktur ve yeni approval manifest üretilmemiştir.

**Kabul ölçütleri:**

- Risk kanalı, rastgele rota geometrisi ve hold pozu içeren dağılımda altı
  kapıyı üç seed boyunca marjla geçmeli.
- Çözüm hiperparametre aramasıyla değil, kapasite/mimari veya risk süpervizyonu
  düzeyinde olmalı. Aday yönler: risk head'ine gizli katman eklemek (şu an
  `latent -> 3` tek doğrusal katman), risk etiketini üç bölge yerine daha zengin
  bir temsilden türetmek, `free_space` regresyonundan türetilen ikili eşiği
  yumuşak hedefle değiştirmek, veya grayscale ablation'ını eğitim sırasında
  açıkça düzenleyici olarak kullanmak.
- Eşik veya kapı değerleri değiştirilecekse gerekçesi ayrı bir ADR ile
  belgelenmeli.

**Çözüm (12 Eylül 2026).** Risk head'e konfigüre edilebilir gizli katman
eklendi (`risk_head_hidden_size`, `vision-encoder-v10` şeması,
`latent -> 64 -> ReLU -> 3`, `+4 160` parametre) ve `pos_weight = 2.0`
kullanıldı. **Karar eşiği `0.62` ve hiçbir kapı eşiği değiştirilmedi.** Bu,
REV-021'in kendi kabul ölçütündeki "hiperparametre aramasıyla değil
kapasite/mimari düzeyinde" çözümdür.

Geriye dönük uyum korundu: `risk_head_hidden_size = 0` tek doğrusal katmandır ve
onaylı `vision-encoder-v9` bu değerle birebir yüklenir. Shipped
`vision_encoder.toml` dokunulmadı; deney ayrı `vision_encoder_v10.toml`
dosyasındadır.

**Doğrulama.** Üç-seed quality kapısı (`runs/vision_quality_v20_20260911`) düşen
kontrol olmadan geçti:

| Seed | Split | `risk_recall` | `risk_precision` | `risk_specificity` |
|---|---|---:|---:|---:|
| `20260911` | validation | `0.770` | `0.813` | `0.747` |
| `20260911` | dev-OOD | `0.722` | `0.747` | `0.784` |
| `20260912` | validation | `0.770` | `0.798` | `0.722` |
| `20260912` | dev-OOD | `0.756` | `0.723` | `0.745` |
| `20260913` | validation | `0.770` | `0.791` | `0.709` |
| `20260913` | dev-OOD | `0.767` | `0.711` | `0.725` |

Bağımsız audit (`runs/vision_audit_v3_20260911`, dokunulmamış split, train-dışı
ellipsoid geometri) tek koşuda geçti:

| Metrik | v10 (`v4c`) | v9 (`v4b`) |
|---|---:|---:|
| `risk_recall` | `0.832` | `0.784` |
| `risk_precision` | `0.764` | `0.780` |
| `risk_balanced_accuracy` | `0.738` | `0.739` |
| `target_precision` | `0.886` | `0.863` |
| `target_recall` | `0.940` | `0.986` |
| `free_space_mae` | `0.243` | `0.255` |

Renk kısa yolu baseline'ı (REV-010 kontrolü) `risk_balanced_accuracy 0.500` ve
`free_space_mae 0.549` verdi; model sırasıyla `0.738` ve `0.243`. Yani model
rengi kısa yol olarak kullanmıyor.

**Durum:** Çözüldü. Yeni onaylı zincir:
`vision_dataset_v4c` -> `vision_quality_v20` -> `vision_audit_v3` ->
`approval_manifest`. Onaylı model `vision-encoder-v10`'dur ve **rastgele rota
geometrisiyle** eğitilmiştir, yani REV-018'in özgün etkisi (onaylı modelin tek
geometride eğitilmiş olması) da giderilmiştir.

D0-V ve D1 sonuçları `vision-encoder-v9` ile alınmıştır ve o zincir
(`vision_audit_v2`) referans olarak geçerliliğini korur.

**Not:** Onaylı zincir bu dilimde değişmedi ve doğrulanabilir durumda:
`vision-encoder-v9` / `vision_dataset_v4b` / `vision_quality_v16` /
`vision_audit_v2`. Terminal hover (ADR-017) ve vision environment (ADR-015)
bu encoder ile çalışmaya devam eder; kısıtı sabit rota geometrisidir ve bu kısıt
yalnız D2 öncesinde kapanmak zorundadır.

## REV-022 — Hold reward üst sınırı iddiasının yanlış olması

ADR-017 ve `TerminalHoverRewardConfig.maximum_hold_reward` docstring'i şunu
iddia ediyordu: "`hover_hold` yalnız `HOLD` durumunda verildiği ve sayaç
tamamlandığında episode bittiği için toplamı `0.40 * 60 = 24.0` değerini aşamaz
ve `success_bonus = 30.0` altında kalır."

Bu iddia **yanlıştır.** Hold sayacı koşul bozulduğunda sıfırlanır fakat durum
yeniden `HOLD` olabilir ve her `HOLD` adımı yeniden reward verir; sayacın
sıfırlanması daha önce verilmiş reward'ı geri almaz. Üst sınır kesintisiz dizi
uzunluğu değil **episode uzunluğudur**.

D0-V koşusunda ölçülen başarı episode'u:

```text
193 adım, sonuç hover_success, toplam reward +84.51
  hover_hold        +41.20   %48.8   <- 103 HOLD adımı
  success           +30.00   %35.5
  target_progress   +17.36   %20.5
  time               -1.93   -%2.3
  settle_speed       -1.83   -%2.2
  target_alignment   -0.15   -%0.2
  smoothness         -0.14   -%0.2
durum dağılımı: transit 23, acquired 3, approach 53, settle 10, hold 103, success 1
```

Politika `103` `HOLD` adımı biriktirmiş, yani hold bandına girip çıkmış ve
yalnız son `60` adımı kesintisiz olmuş. Hold reward'ı `41.20`, iddia edilen
sınırın `1.7` katı ve `success_bonus`'un `1.37` katıdır.

Gerçek episode üst sınırları:

| Episode bütçesi | Hold reward üst sınırı |
|---|---:|
| `400` adım (`4 m` rota, `20 s` min clamp) | `160.0` |
| `1060` adım (`30 m` final segment, `53 s`) | `424.0` |

**Etki:** ADR-012 "hold reward'u sınırlı olacaktır; hedef dışında bekleyerek
reward biriktirmek başarı sayılmaz" diyor. Bileşen hâlâ yalnız `HOLD`
durumunda verildiği için "hedef dışında" birikim yoktur; koşul gerçekten
sağlanıyordur. Fakat hold bandında salınarak biriken reward, hold'u tamamlama
teşvikini görece zayıflatır: bir politika `60` kesintisiz adım yerine `150` kesik
adım biriktirmeyi tercih edebilir. D0-V koşusunda bu davranış başarıyı
engellemedi, fakat reward tasarımı belgelendiği gibi değildir.

**Kabul ölçütleri:**

- Kesintisiz dizi değeri ile episode üst sınırı ayrı isimlendirilmeli ve
  ikisinin farkı belgelenmeli.
- ADR-017'deki yanlış sınır iddiası düzeltilmeli.
- Episode üst sınırının `success_bonus`'u aştığı testle sabitlenmeli.

**Çözüm:** `maximum_hold_reward` adı `uninterrupted_hold_reward` olarak
değiştirildi ve docstring'i bunun bir episode sınırı olmadığını açıkça söylüyor.
`episode_hold_reward_bound(maximum_episode_steps)` eklendi. ADR-017 düzeltildi.

**Doğrulama:** `test_episode_hold_reward_bound_exceeds_uninterrupted_sequence`
episode sınırının kesintisiz dizi değerini ve `success_bonus`'u aştığını
sabitliyor.

**Açık kalan tasarım sorusu:** `hover_hold_weight` ile `success_bonus` oranının
doğru değeri ölçülmedi. ADR-017 bunu "D0-V'de ölçülecek" diye kaydetmişti; D0-V
sonucu oranın çalıştığını (başarı öğrenildi) fakat belgelenen gibi olmadığını
gösterdi. Ablation D1 curriculum diliminde yapılmalıdır.
## REV-023 — Uygulanabilir hold bandının politika hassasiyetine göre çok dar olması

ADR-016, sabit `45 derece` montajda hedef merkezinin hover anchor üzerinde
kadraj dışında kaldığını ve merkez-görünür hold pozunun ancak hedefin
`>= 0.542 m` gerisinde mümkün olduğunu ölçmüştü. Güvenli iç bölge yarı-boyutu
`0.65 m` olduğundan uygulanabilir hold bandı bu ikisinin kesişimidir:

```text
uygulanabilir hold bandı = [merkez-görünür sınırı, güvenli bölge sınırı]

hover irtifası   merkez-görünür   güvenli sınır     bant
      2.00 m          0.542 m         0.650 m     0.108 m   <- mevcut
      1.75 m          0.474 m         0.650 m     0.176 m
      1.50 m          0.407 m         0.650 m     0.243 m
      1.25 m          0.339 m         0.650 m     0.311 m
```

ADR-016 bu bandı `~0.11 m` olarak kaydetmiş ve `hover_height_m`'i düşürmeyi
"ölçülmüş bir kalibrasyon kaldıracı" olarak bırakmıştı. D0-V koşusu bu
kaldıracın **gerekçesini** üretti.

Eğitilmiş politikanın iki checkpoint'i deterministik kipte hedefin arkasında
neredeyse aynı yere yerleşiyor, fakat başarı `3.6 cm` ile belirleniyor:

| Checkpoint | Yatay offset | Güvenli bölgede | İrtifa hatası |
|---|---:|---|---:|
| `best` | `0.646 m` | evet | `0.100 m` |
| `final` | `0.682 m` | **hayır** | `0.163 m` |

Sonuç, deterministik/stokastik kipte ters yönde çıkıyor:

| Checkpoint | Kip | Başarı |
|---|---|---:|
| `best` | deterministik | `20/20` |
| `best` | stokastik | `11/20` |
| `final` | deterministik | `0/20` |
| `final` | stokastik | `20/20` |

Mekanizma nettir: her politikanın ortalama action'ı bandın bir tarafına oturur.
`best`'in ortalaması bandın **içinde** (`0.646 < 0.650`), bu yüzden deterministik
her zaman başarılı; action gürültüsü onu `%45` oranında bandın dışına atıyor.
`final`'in ortalaması bandın **dışında** (`0.682 > 0.650`), bu yüzden
deterministik hiç başarılı değil; gürültü onu yeterince sık bandın içine
sokuyor. İki checkpoint de `hold` durumuna giriyor (`final` deterministik kipte
`400` adımın `32`'sinde) fakat `60` kesintisiz adımı tamamlayamıyor.

**Etki:** Görev başarısı, politikanın konum hassasiyetinden küçük bir banda
bağlıdır. Bu bir öğrenme başarısızlığı değil, görev geometrisinin kırılganlığıdır
ve üç sonucu vardır: checkpoint seçimi başarıyı belirler (`best` vs `final`),
deterministik ile stokastik evaluation birbirinin yerine kullanılamaz, ve
gerçek araçta konum kestirim hatası bu bandı doğrudan tüketir.

**Kabul ölçütleri:**

- Uygulanabilir hover bandı, hedeflenen konum kestirim hatasına göre
  gerekçelendirilmeli.
- `hover_height_m` değişecekse ADR-012 başarı geometrisi etkilendiği için
  ADR-016 güncellenmeli veya yeni ADR yazılmalı.
- Deterministik ve stokastik evaluation ayrı raporlanmalı; biri diğerinin yerine
  geçmemeli.

**Durum:** Çözüldü (12 Eylül 2026). `hover_height_m` `2.0` -> `1.5` uygulandı;
ADR-016 §3'e dayanağı ve bedeli yazıldı.

```text
irtifa        2.01 m -> 1.51 m
yakın sınır   0.542 m -> 0.407 m
hold halkası  0.108 m -> 0.243 m   (2.25 kat)
uzak sınır    7.46 m -> 5.60 m     (bedel: tespit menzili kısalır)
```

Başarı koşulu, güvenli iç bölge (`0.65 m`) ve `hold_seconds` **değişmedi**;
yalnız o koşulun sağlanabileceği konum kümesi büyütüldü. Uzak sınırın kısalması
ADR-019 merdivenini etkilemez: hover irtifasında karşılanabilen belirsizlik
yarıçapı `R <= 5.60 m` olur, D1 (`R = 2 m`) ve D2 (`R = 5 m`) içeride kalır,
D3'ten itibaren zaten `4.0 m`'ye tırmanılır.

Testler: `tests/test_hover_geometry.py` halkayı `0.243 m` olarak sabitler,
`> 0.20 m` alt sınırını zorlar ve irtifa düştükçe halkanın genişlediğini
monotonluk testiyle doğrular; `tests/test_vision_config.py` hover anchor'ı
`1.51 m` bekler. Deterministik/stokastik ayrışmasının gerçekten kapandığı ancak
yeni koşu evaluation'ıyla ölçülebilir.

**Not:** Evaluation script'i bu bulgudan sonra deterministik ve stokastik kipleri
ayrı raporlar. Tek bir kipte raporlanan başarı oranı bu görevde yanıltıcıdır.

## REV-024 — Localization hatasının hiç test edilmemiş olması ve ilk drift ablation'ının yanıltıcılığı

D0-V sonrası, politikanın waypoint oracle'ına bağımlılığını ölçmek için bir
ablation çalıştırıldı. İki sonucu geçerlidir, üçüncüsü **yanıltıcıdır ve bu
kayıt onu düzeltmek için vardır.**

Geçerli olan iki sonuç (`best` checkpoint, 6 episode, deterministik):

| Ablation | Başarı | Terminasyon |
|---|---:|---|
| baseline (oracle) | `6/6` | `hover_success` |
| `waypoint = 0` | `0/6` | `out_of_bounds` |
| `vision_latent = 0` | `0/6` | `time_limit` |

Bunlar iki kanalın da taşıyıcı olduğunu gösterir: waypoint kaba navigasyonu,
görüntü terminal hover'ı yapar.

**Yanıltıcı olan sonuç:** aynı ablation'da enjekte edilen "VIO drift" `%20`
oranında bile `5/6` başarı verdi. Bu, drift dayanıklılığı kanıtı **değildir.**
Drift modeli her policy adımında **rastgele yönde** hata ekliyordu, yani bir
rastgele yürüyüştü ve hatalar birbirini götürüyordu:

```text
yol ~5 m, 193 adım, adım başına 2.6 cm

oran    modelin ürettiği (rastgele yürüyüş)    gerçekçi (bias-baskın, doğrusal)
%1                        0.4 cm                                  5.0 cm
%2                        0.7 cm                                 10.0 cm
%5                        1.8 cm                                 25.0 cm
%10                       3.6 cm                                 50.0 cm
%20                       7.2 cm                                100.0 cm
```

Güvenli iç bölge `65 cm`'dir. Yani `%20` oranında bile yalnız `7 cm` mutlak
hata üretildi; **anlamlı bir localization hatası hiç test edilmedi.** Gerçek
VIO hatası ölçek hatası ve gyro bias'ından gelir ve mesafeyle yaklaşık doğrusal
büyür, rastgele yürüyüş gibi sönmez.

**Etki:** `5/6` sonucunu okuyan bir sonraki agent, mimarinin localization
hatasına dayanıklı olduğu sonucuna varabilir. Böyle bir kanıt **yoktur.**
Ölçülen tek şey `~7 cm` hatanın tolere edildiğidir ve bu zaten güvenli bölgenin
onda biridir.

**Ayrıca ölçülen fiziksel sınırlar (bunlar geçerlidir):**

Görsel tespit menzili, `2 x 2 m` hedef, odak uzaklığı `208.5 px`:

| Mesafe | Hedef genişliği | Durum |
|---:|---:|---|
| `40 m` | `10.4 px` | güvenilir |
| `100 m` | `4.2 px` | sınırda |
| `200 m` | `2.1 px` | güvenilmez |
| `500 m` | `0.8 px` | alt-piksel, imkansız |

Menzil bütçesi: varışta localization hatası tespit yarıçapını aşarsa araç
sandığı yere gider, hedefi göremez ve düzeltme kaynağı kalmaz.

```text
maksimum görev mesafesi = tespit_yarıçapı / drift_oranı

tespit yarıçapı 42 m:   %0.5 -> 8400 m    %1 -> 4200 m    %2 -> 2100 m    %5 -> 840 m
```

ADR-018'in D5 hedefi `2 km`'dir, yani `%2` drift tam sınırdadır.

Terminal hover localization ile kapatılamaz: `%1` drift `100 m` sonra `1.0 m`
hata verir, güvenli bölge ise `0.65 m`'dir. Terminal fazda görsel servoing
zorunludur.

**Kabul ölçütleri:**

- Bu kayıt `5/6` sonucunun geçersizliğini kalıcı olarak belgelemeli.
- Localization hatası modellenecekse bias-baskın bileşen içermeli; saf rastgele
  yürüyüş kullanılmamalı.
- Menzil bütçesi formülü (`tespit_yarıçapı / drift_oranı`) D4-D5 kapılarında
  açıkça değerlendirilmeli.

**Durum:** Kısmen kapatıldı. ADR-019 kapsam kararı gereği **kendi konum hatası
modellenmeyecektir**; görsel pozyonlamanın çözdüğü varsayılır. Bu nedenle drift
modeli bu projede uygulanmayacaktır ve bulgunun "drift ablation'ını düzelt"
kısmı kapsam dışıdır.

Açık kalan kısım: ADR-019 hedef konum hatası (`R`) yolu, localization hatasının
**yerine** değil **yanına** gelir. Gerçek sistemde toplam bütçe `R + drift`'tir.
Bu toplam, D4-D5 kapılarında menzil bütçesi formülüyle raporlanmalıdır; aksi
halde `R` tek başına gerçekçi bir hata bütçesi değildir.

**Not:** Ablation script'i `runs/` altında saklanmadı; tek kullanımlık teşhis
aracıydı. Tekrarlanacaksa drift modeli yeniden yazılmalıdır.

## REV-025 — Grace period'un yaw ile toparlanma süresinden kısa olması

ADR-017 `grace_seconds = 0.50 s` değerini "ADR-012'nin sınırlı grace period
ilkesine dayanır" diye kaydetmiş ve ölçülmediğini belirtmişti. ADR-019 yaklaşık
hedef konumunu devreye alınca bu değer ölçülebilir hâle geldi ve **iç tutarsız**
olduğu görüldü.

**Ölçüm 1 — hedef kadrajda kalmıyor.** Araç inanılan hedef konumunda hover
irtifasında (`2.01 m`) duruyorsa, zemin izi nadir'den `0.54-7.46 m` halkasıdır ve
yatay FOV `75 derece`dir. `R = 2 m` hata diskinden 200k örnekle:

```text
gerçek hedef nadir kör bölgesinde (d < 0.54 m)  : %7.3
halka içinde                                     : %92.7
halka İÇİNDE ve kadrajda (yaw taraması yok)      : %19.2
```

Yani yaw taraması yapılmazsa hedefin kadrajda kalma olasılığı `%19`. `360 derece`
tarama yapılırsa `%92.7`'ye çıkar; kaybedilen yalnız nadir kör bölgesidir.

**Ölçüm 2 — grace period toparlanmaya yetmiyor.** `max_yaw_rate_rad_s = 1.5`
ile `0.50 s` içinde yalnız `43 derece` dönülebilir:

| Gereken dönüş | Süre | `0.50 s` grace |
|---:|---:|---|
| `37.5 deg` (FOV yarısı, yeni çıkmış hedef) | `0.44 s` | yeter |
| `90 deg` (rastgele yön, ortalama) | `1.05 s` | **yetmez** |
| `180 deg` (arkada) | `2.09 s` | **yetmez** |

**Etki:** Politika hedefi kaybettiğinde yaw ile geri bulmaya çalışsa bile grace
dolmadan yetişemez ve `hover_abort_target_lost` alır. D1 eğitiminin ilk `100k`
adımında terminasyonların `%43`'ü (`148/347`) bu sebeple oluştu ve bu dağılım
eğitim boyunca baskın kaldı. Yani abort, arama davranışını öğretmek yerine
öğrenmeyi kesiyor.

Bu, ADR-019 ile birlikte ortaya çıkan yeni bir tutarsızlıktır: D0-V'de hedef
waypoint'in gösterdiği yerde olduğu için kayıp nadirdi ve `0.50 s` sorun
yaratmıyordu.

**Kabul ölçütleri:**

- `grace_seconds`, aksiyon limitleriyle tutarlı olmalı: hedefi yeniden bulmak
  için gereken yaw süresinden kısa olmamalı.
- Değer ölçümle gerekçelendirilmeli, kapıyı geçirmek için seçilmemeli.
- Başarı koşulu (güvenli iç bölge, irtifa, hız, tilt, açısal hız, kesintisiz
  hold) **değişmemeli**; yalnız görsel süreklilik toleransı düzeltilmeli.
- Arama davranışının reward'da karşılığı olup olmadığı ayrıca değerlendirilmeli.

**Çözüm:** `grace_seconds` `0.50 -> 2.50 s` yapıldı (`180 derece` dönüş
`2.09 s` artı pay) ve ADR-017 güncellendi. **Başarı koşuluna dokunulmadı**:
güvenli iç bölge `0.65 m`, irtifa toleransı `0.20 m`, hız `0.25 m/s`, tilt
`10 derece`, açısal hız `0.35 rad/s` ve kesintisiz `3.0 s` hold aynıdır.

**Doğrulama:** Etkisi, kanıt koşusuyla aynı bütçede yeniden koşularak izole
edildi. Aynı timestep penceresinde (`0-133k`) karşılaştırma:

| Terminasyon | `grace = 0.50 s` | `grace = 2.50 s` |
|---|---:|---:|
| `hover_abort_target_lost` | `%40` | **`%11`** |
| `obstacle_collision` | `%14` | **`%33`** |
| `time_limit` | `%33` | `%39` |
| `out_of_bounds` | `%10` | `%10` |
| `minimum_altitude` | `%4` | `%6` |
| ortalama episode uzunluğu | `302` adım | `340` adım |

Düzeltme amaçlandığı gibi çalıştı: `target_lost` abort'u `%40`'tan `%11`'e
düştü ve episode'lar uzadı. Ancak serbest kalan episode süresi çarpışmaya gitti
(`%14 -> %33`), yani **REV-026 artık bağlayıcı kısıttır.** Reward bu pencerede
`-2.60 -> -5.91` geriledi çünkü her çarpışma `-25` maliyetlidir ve daha sık
oluyor.

Bu, bir blocker'ın kaldırılmasının sıradakini açığa çıkardığının temiz bir
gösterimidir. REV-025 kendi ekseninde kapatılmıştır (`target_lost` abort'u
evaluation'da `%45 -> %0`).

**Ancak düzeltme net sonucu kötüleştirdi:** deterministik başarı `%25 -> %5`,
timeout `%15 -> %70`. Nedeni REV-027'dir — hedefi görüşte tutmanın reward'da
pozitif karşılığı yok, kısa grace kazara bir şekillendirme sinyali işlevi
görüyordu. Ayrıntı REV-027 kaydındadır.

**Not:** Arama davranışı için ayrı bir reward bileşeni **yoktur**. Politika
aramayı yalnız `target_progress` ve `hover_hold` üzerinden dolaylı öğrenmek
zorundadır. Grace düzeltmesi tek başına yetmezse bu ikinci eksiklik
değerlendirilmelidir.

**Durum:** Kendi ekseninde çözüldü, kalibrasyonu açık. `grace_seconds = 2.50 s`
uygulandı ve `target_lost` abort'u evaluation'da `%45 -> %0` oldu; bu kısım
kapandı. Fakat düzeltme net sonucu kötüleştirdiği için değer **doğrulanmış
sayılmaz**: `2.50 s` yaw toparlanmasının ölçülmüş gereksinimini karşılar
(`180 derece` için `2.09 s`) ama tek başına doğru denge değildir.

Yukarıdaki "ikinci eksiklik" REV-027 olarak yazıldı ve 12 Eylül 2026'da
kapatıldı (`target_visible` bileşeni). Grace'in yeniden kalibrasyonu REV-026 ve
REV-027 düzeltmelerinin birlikte ölçülmesinden **sonra** yapılacaktır; sıranın
tersi, birbirini telafi eden iki hatalı değere yol açar. Ölçüm hazır olduğunda
sorulacak soru şudur: `target_visible` görünürlüğü zaten sağlıyorsa grace
kısaltılabilir mi, yoksa `2.50 s` arama davranışı için gerekli mi?

## REV-026 — Collision cezasının rota uzunluğuyla ölçeklenmemesi (GEÇERSİZ, bkz. REV-029)

D1 eğitiminde `obstacle_collision` oranı eğitim boyunca **arttı**, azalmadı:

```text
600k adım, 1760 episode, altı eşit dilim:
  %15 -> %12 -> %13 -> %13 -> %17 -> %20
```

Yani politika hedefe yaklaşmayı öğrenirken engellere **daha çok** çarptı. Bu
reward dengesinden kaynaklanıyor: `target_progress` rota uzunluğuyla doğrusal
ölçeklenirken `collision_penalty` sabittir.

| Seviye | Rota | Toplam `target_progress` | `collision` | Oran |
|---|---:|---:|---:|---:|
| D0-V | `4 m` | `+20.0` | `-25.0` | `0.8x` |
| D1 (ort) | `10 m` | `+50.0` | `-25.0` | `2.0x` |
| D1 (max) | `15 m` | `+75.0` | `-25.0` | `3.0x` |
| D2 | `50 m` | `+250.0` | `-25.0` | `10x` |
| D3 | `150 m` | `+750.0` | `-25.0` | `30x` |
| D4 | `500 m` | `+2 500.0` | `-25.0` | `100x` |

D0-V'de oran `0.8x` olduğu için ceza baskındı — fakat o koşuda hiç engel yoktu,
yani denge hiç sınanmadı. D1'de oran `2-3x`'e çıktı ve engele çarparak kısa
yoldan gitmek net kazançlı hâle geldi.

`proximity` bileşeni bu dengeyi kurtarmıyor: `0.05/adım` ve yalnız `0.75 m`
altındaki açıklıkta etkin; `400` adımlık episode'da teorik üst sınırı `20.0`, ve
bu ancak açıklık sürekli tam sıfırken oluşur.

**Etki:** ADR-018 merdiveni mesafeyi büyüttükçe engelden kaçınma teşviki
sistematik olarak zayıflar. D3'te oran `30x`, D4'te `100x`. Bu, "engellere
çarpmadan hedefe gitme" olan proje ana hedefiyle doğrudan çelişir.

**Kabul ölçütleri:**

- Çarpışma cezası rota uzunluğundan bağımsız biçimde baskın kalmalı; ya ceza
  rota ile ölçeklenmeli ya progress ödülü normalize edilmeli.
- Tercih edilen yön: `target_progress`'i rota uzunluğuna **normalize** etmek
  (kapatılan mesafenin oranı), böylece toplam progress ödülü her seviyede aynı
  mertebede kalır ve sabit ceza anlamlı olur.
- Değişiklik ADR-017 reward sözleşmesini değiştirir ve ADR güncellemesi
  gerektirir. Başarı koşuluna dokunulmamalıdır.
- Düzeltme sonrası çarpışma oranı trendi **azalan** olmalı; D1 kapısı
  `<= %5` ister.

**Durum:** **GEÇERSİZ — yanlış teşhis.** REV-029 bu bulgunun iki dayanağını da
ölçümle çürüttü: çarpışma uzun değil kısa rotalarda daha sık (`5-8 m`'de `%23`,
`11-15 m`'de `%8`) ve eğitim boyunca gözlenen artış survivorship'ti (hedefe
ulaşan episode oranı `%53 -> %74`, ulaşanlar arasında çarpışma sabit). Aşağıdaki
düzeltme uygulandı, iki koşu boyunca ölçüldü, geriletti ve **geri alındı**.
Kayıt tarihsel bütünlük için bırakılmıştır; uygulanmış hâli REV-029'dadır.

Özgün kayıt (artık geçersiz): `target_progress` planlanan rota uzunluğuna
normalize edildi ve ağırlık `5.0` -> `15.0` yapıldı.

```text
eski: 5.0 * (d_önce - d_sonra)              toplam = 5.0 * L_plan
yeni: 15.0 * (d_önce - d_sonra) / L_plan    toplam = 15.0  (L_plan'dan bağımsız)
```

Sabit `collision_penalty = 25.0` artık progress toplamının `1.67` katıdır ve bu
oran D1'den D5'e kadar sabit kalır (önce D1'de `2.0x`, D4'te `100x` olacaktı).
`w = 15.0` seçimi `success_bonus = 30.0`'ın payını korur: bonus toplam pozitif
sinyalin `%67`'sini taşır.

Testler: `L_plan` 2 / 10 / 65 / 500 m için toplamın `progress_reward_bound`
altında kalması ve rota uzunluğuyla doğrusal büyümemesi,
`collision_penalty > progress_reward_bound(approach_radius_m)` invariantının
config yüklemesinde fail-closed olması, geçersiz menzil değerlerinin reddi.

**Not (REV-028):** tam normalizasyon bu bulguyu çözerken terminal aşamada yeni
bir gerileme üretti ve nihai form iki menzillidir: kruvaziyer menzilde rotaya
normalize (`w_cruise = 10.0`), `approach_radius_m` içinde sabit
(`w_terminal = 5.0`). Bu bulgunun invariantı korunur; ayrıntı REV-028'dedir.

Düzeltmenin asıl kabul ölçütü bir invariant değil **davranıştır**: çarpışma oranı
trendi azalmalı. Bu ancak yeni koşuyla ölçülebilir.

**Not:** Bu bulgu yalnız engel ekseni açıldığı için görünür oldu. D0-V'nin
engelsiz koşusunda `proximity` bileşeni tam olarak `-0.0` ve `collision` tam
olarak `0.00` idi; denge test edilmemişti.

### Somut düzeltme önerisi (karar bekliyor)

**Kök neden:** `target_progress = w * (d_prev - d_now)` toplandığında `w * L`
verir, yani rota uzunluğuyla doğrusal büyür. `collision_penalty` sabittir.

**Öneri:** progress'i planlanan rota uzunluğuna normalize etmek.

```text
mevcut : progress = w * (d_prev - d_now)          -> toplam = w * L
öneri  : progress = w * (d_prev - d_now) / L_plan -> toplam = w  (rota bağımsız)
```

`L_plan` zaten ADR-013 bütçe formülünde kullanılıyor ve `info` içinde
raporlanıyor, yani yeni bir bilgi kaynağı gerekmez. Politika açısından sinyalin
işareti ve şekli aynı kalır; yalnız ölçeği rotadan bağımsız hâle gelir.

Normalize edilirse `w` yükseltilmelidir, aksi halde hedefe gitme teşviki
zayıflar. Ölçülen denge tablosu:

| `w` | Toplam progress | `collision` oranı | `success_bonus` payı |
|---:|---:|---:|---:|
| `5.0` (mevcut) | `+5.0` | `0.20x` | `%86` |
| `15.0` | `+15.0` | `0.60x` | `%67` |
| `25.0` | `+25.0` | `1.00x` | `%55` |
| `40.0` | `+40.0` | `1.60x` | `%43` |

**Önerilen değer `w = 15.0`:** çarpışma cezası progress'in `1.7` katı kalır
(oran `0.60x`), yani engele çarpmak her mesafe seviyesinde net kayıptır; buna
karşın `success_bonus` payı `%67` ile baskın kalır, yani asıl ödül hâlâ görevi
tamamlamaktır.

Bu değişiklik **ADR-017 reward semantiğini** değiştirir ve ADR güncellemesi
gerektirir. Başarı koşuluna dokunmaz. Uygulandıktan sonra D1 aynı `600k`
bütçeyle yeniden koşulup çarpışma oranı trendinin **azalan** olduğu
doğrulanmalıdır.

## REV-027 — Hedef görünürlüğünün reward'da temsil edilmemesi

REV-025 düzeltmesi (`grace_seconds` `0.50 -> 2.50 s`) uygulandıktan sonra D1
aynı `600k` bütçeyle yeniden koşuldu. Düzeltme hedeflediği ekseni tamamen
düzeltti fakat **sonucu kötüleştirdi.**

Evaluation (20 episode, seed `410001`, `final` checkpoint):

| Koşu | Policy | Başarı | Çarpışma | Abort | Timeout | Return |
|---|---|---:|---:|---:|---:|---:|
| kanıt (`grace 0.5 s`) | deterministik | **`%25`** | `%5` | `%45` | `%15` | `31.6` |
| kanıt (`grace 0.5 s`) | stokastik | `%0` | `%10` | `%45` | `%35` | `25.6` |
| grace `2.5 s` | deterministik | **`%5`** | `%15` | `%0` | `%70` | `11.1` |
| grace `2.5 s` | stokastik | `%0` | `%15` | `%5` | `%80` | `8.9` |

Eğitim tarafı (tam `600k`):

| Metrik | `grace 0.5 s` | `grace 2.5 s` |
|---|---:|---:|
| `hover_abort_target_lost` | `%32` | `%13` |
| `obstacle_collision` | `%15` | `%22` |
| `time_limit` | `%48` | `%57` |
| ortalama episode uzunluğu | `340` adım | `412` adım |
| reward (son çeyrek) | `20.69` | `13.24` |
| `hold_max` | `0.85 s` | `0.70 s` |

**Teşhis.** `target_lost` abort'u `%45`'ten `%0`'a indi, yani düzeltme kendi
ekseninde kusursuz çalıştı. Ancak başarı `%25`'ten `%5`'e düştü ve timeout
`%15`'ten `%70`'e çıktı.

Nedeni şudur: hedefi görüşte tutmanın reward'da **hiçbir pozitif karşılığı
yoktur**. Görünürlüğü sağlayan tek mekanizma `target_lost` abort'unun sert
cezasıydı. Kısa grace, kazara bir şekillendirme sinyali gibi davranıyordu:
politikayı hedefi kadrajda tutmaya, dolayısıyla hedefin yakınında kalmaya
zorluyordu. Grace uzatılınca bu baskı kalktı ve politika hedeften uzaklaşıp
bütçeyi doldurmaya başladı.

`target_alignment` bileşeni bu işi yapmıyor: ağırlığı `0.20` ve ADR-016 uyarınca
yalnız `acquired`/`approach` durumlarında etkin; `settle`/`hold` sırasında
kapalı. Ayrıca kaybedilen hedef için `target_center_xy` `(0, 0)` sentinel'i
verdiği için sinyal taşımıyor (REV-017).

**Etki:** Görsel süreklilik, reward ile şekillendirilmek yerine terminasyonla
zorlanıyor. Bu iki sorun üretir: (a) grace değeri serbestçe kalibre edilemez
çünkü aynı zamanda gizli bir şekillendirme parametresidir; (b) politika hedefi
görüşte tutmayı bir amaç olarak öğrenmez, yalnız cezadan kaçar.

**Kabul ölçütleri:**

- Hedef görünürlüğü açık bir reward bileşeniyle temsil edilmeli; örneğin
  `target_visible` durumunda küçük pozitif bir terim veya `target_score`
  üzerinden sürekli bir terim.
- Bileşen eklendikten sonra `grace_seconds` ölçümle kalibre edilebilir hâle
  gelmeli; abort artık gizli şekillendirme yapmamalı.
- `hover_hold` ile toplamı `success_bonus`'u aşmamalı; hedefi izleyip görevi
  tamamlamamak kazançlı olmamalı.
- Değişiklik ADR-017 reward sözleşmesini değiştirir ve ADR güncellemesi
  gerektirir. Başarı koşuluna dokunmamalıdır.

**Durum:** Çözüldü (12 Eylül 2026). REV-026 ile aynı dilimde, aynı reward
dengesini ilgilendirdiği için birlikte uygulandı. `target_visible` bileşeni
eklendi: hedef kamerada görünür olduğu her adımda `+0.005`, **bütün**
durumlarda. `target_alignment`'ın aksine `transit` ve `search` durumlarında da
uygulanır; REV-027'nin en çok ısırdığı yer orasıdır, çünkü orada görünürlükle
ilgili başka hiçbir sinyal yoktur.

Ağırlığın üst sınırı türetildi ve `TerminalHoverRewardConfig.validate()`
içinde zorlandı:

```text
target_visible_weight < time_penalty   (0.005 < 0.01)
```

Eşit veya üzerinde bir değer, hedefi uzaktan seyrederek beklemeyi adım başına
net pozitif yapardı; yani görünürlük sorununu çözerken timeout'u besleyen yeni
bir yol açardı. `0.005` ile seyrederek beklemek hâlâ `-0.005/adım`'dır.
`hover_hold` bu sınıra tabi değildir ve olmamalıdır: o yalnız hedefin üzerinde,
`HOLD` durumunda verilir.

Testler: her durumda görünürlükle verilmesi ve görünmezken sıfır olması,
`target_visible_weight >= time_penalty` konfigürasyonunun reddi, görünürken
bekleme adımının net negatif kalması.

`grace_seconds` `2.50 s`'de bırakıldı. REV-025'in kalibrasyonu bu düzeltmenin
etkisi ölçüldükten sonra yeniden ele alınacaktır.

**`grace_seconds` hakkında karar notu.** Değer `2.50 s`'de bırakıldı,
`0.50 s`'ye geri alınmadı. Gerekçe: `0.50 s`'nin aksiyon limitleriyle çeliştiği
ölçüm geçerliliğini koruyor (`0.50 s`'de yalnız `43 derece` dönülebilir, gereken
`90-180 derece`). Ölçülmüş biçimde yanlış bir değeri, ikinci bir eksiklik onu
kazara telafi ettiği için geri koymak doğru olmaz. Doğru sıra REV-027'yi
kapatmak, sonra grace'i kalibre etmektir.

**Bugünün en iyi D1 politikası** yine de kanıt koşusunun `final` checkpoint'idir
(`%25` deterministik): `runs/d1-seed20260912/final_model.zip`. Bu bir kalite
iddiası değil, mevcut en iyi artefaktın kaydıdır.

## Kapanış doğrulaması — REV-001–REV-013

Tüm düzeltmeler birlikte uygulandıktan sonra tam regresyon yeniden çalıştırıldı:

- pytest: `121 passed, 1 skipped, 12 subtests passed`; tek skip sandbox içindeki
  beklenen macOS offscreen capture testidir,
- Ruff lint temiz; format `81 files already formatted`,
- mypy `44` source file için temiz; compileall ve dependency kontrolü geçti,
- grafik erişimli MuJoCo smoke dataset ve 736-kare tam dataset leakage kontrolü
  geçti,
- izole wheel build/install içinde v4/v9 config, ellipsoid model, audit API'sı
  ve gerçek approval manifest yüklemesi geçti,
- `24` Markdown dosyasının yerel linkleri ve görünmez Unicode kontrolleri temiz.

Bu sonuçlar düzeltmelerin mevcut test kapsamındaki eski davranışları bozmadığını
gösterir; sentetik audit gerçek uçuş güvenliği veya gerçek IMX477 genellemesi
olarak yorumlanmaz.

## 11 Eylül 2026 — Vision environment öncesi inceleme

Bu inceleme yalnız bulgu ve doğrulama kapsamındadır; production koduna düzeltme
uygulanmamıştır. Faz 6.5'in mevcut sayıları sentetik geliştirme kapısı olarak
geçerlidir, fakat aşağıdaki yüksek öncelikli bulgular kapanmadan bağımsız
akademik test veya gerçekçi görsel kaçınma kanıtı sayılmamalıdır.

## REV-007 — Test split'inin geliştirme döngüsünde tüketilmesi

`train_vision_encoder` her eğitimde test split'ini değerlendiriyor ve quality
runner her seed için test metriklerini geçiş kararına katıyor. Daha önemlisi,
v2-v11 ablation kararları test free-space/risk/target sonuçlarına bakılarak
alındı. Buna karşın Faz 6.5 belgesi test split'inin model/eşik seçimine
katılmadığını söylüyor. Seed seçici doğrudan yalnız validation kullansa da
mimari ve training kararları test sonucundan etkilendiği için mevcut test seti
artık bağımsız değildir.

**Etki:** Raporlanan test başarımı iyimser olabilir; akademik nihai test iddiası
desteklenmez.

**Kabul ölçütleri:**

- Mevcut test split'i `development_ood` olarak yeniden sınıflandırılmalı.
- Eşikler ve model dondurulduktan sonra yeni salt/scene seed ve en az bir yeni
  appearance/geometri profiliyle dokunulmamış `audit` split'i oluşturulmalı.
- Audit manifest/config hash'i koşudan önce kilitlenmeli; ara geliştirme
  kararlarında audit metrikleri açılmamalı.
- Faz belgesindeki “test model seçimine katılmaz” ifadesi bu geçmişi doğru
  anlatacak şekilde düzeltilmeli.

**Çözüm:** Eski test split'i kalıcı olarak `development_ood` adını aldı ve
trainer/quality runner `audit` split'ini yüklemiyor. `vision-dataset-v4` dört
ayrık kota kullanıyor. Aday ve eşikler yalnız validation/development-OOD ile
donduruluyor; audit ayrı CLI ile çalışıyor ve yalnız başarı halinde approval
manifest üretiyor. Quality dizinindeki atomik consumption kaydı aynı adayın
ikinci kez audit edilmesini engelliyor.

**Doğrulama:** İlk bağımsız audit başarısız oldu ve approval üretmedi. Bu sete
göre eşik ayarı yapılmadan geometri zarfı genişletildi, split salt'ı değiştirildi
ve görülmemiş ellipsoid geometrili yeni 96-kare audit oluşturuldu. Dondurulmuş
v16 adayının bu audit'i tek koşuda geçtiği
`runs/vision_audit_v2_20260911/audit_report.json` ile kayıtlıdır.

## REV-008 — Kalite kapısının otomasyonu durdurmaması

`scripts/run_vision_quality.py`, `quality_passed=False` durumunda da normal
biçimde tamamlanıp süreç kodu `0` döndürüyor. `run_vision_quality` ayrıca kapı
başarısızken de `selected_checkpoint` seçiyor ve CLI bunu kullanıcıya seçilmiş
model olarak yazıyor. Önceki başarısız v6-v10 koşularının süreç kodu 0 ile
bitmesi davranışı doğruluyor.

**Etki:** CI veya sonraki agent başarısız kalite koşusunu başarılı kabul edip
geçersiz checkpoint'i vision environment'a bağlayabilir.

**Kabul ölçütleri:**

- CLI başarısız kalite kapısında raporu yazdıktan sonra non-zero dönmeli.
- Kapı geçmedikçe `approved_checkpoint` üretilmemeli; tanılama modeli gerekiyorsa
  açıkça `candidate_checkpoint` adı kullanılmalı.
- Başarılı/başarısız CLI çıkış kodu ve artefact semantiği regression testine
  dönüşmeli.

**Çözüm:** Quality artefact'ı yalnız `candidate_checkpoint` yayımlar. Kapı
başarısızsa CLI süreç kodu non-zero olur. `approved_checkpoint` yalnız ayrı ve
başarılı audit'in `approval_manifest.json` dosyasında bulunabilir.

**Doğrulama:** Gerçek `run_vision_quality.py` subprocess testi bir epoch'luk
başarısız koşunun non-zero döndüğünü, `quality_passed=False` yazdığını ve
`approved_checkpoint` yayımlamadığını doğruladı. Encoder hedefli testleri
`7 passed`; Ruff ve mypy temiz geçti.

## REV-009 — Online RGB preprocessing sözleşmesinin eksikliği

Offline `VisionNpzDataset`, actor RGB'sini HWC `uint8`'den BCHW `[0,1]` float
tensöre çeviriyor. Bu dönüşüm ayrı bir public inference API'si değildir ve
`TinyVisionEncoder.forward` yalnız rank/kanal kontrolü yapar. Doğrudan kamera
`uint8 BCHW` girdisi `mean()` aşamasında çöker; `float [0,255]` ise hata vermeden
kabul edilip clipping nedeniyle farklı sonuç üretir.

**Etki:** Yeni online environment veya Jetson adapter'ı offline eğitimden farklı
preprocessing uygulayabilir; sessiz ölçek hatası politika girdisini bozabilir.

**Kabul ölçütleri:**

- `ActorCameraFrame -> float32 BCHW [0,1]` dönüşümü tek public fonksiyonda
  tanımlanmalı ve dataset/online inference aynı fonksiyonu kullanmalı.
- Model sınırında dtype, sonluluk ve değer aralığı fail-fast doğrulanmalı veya
  model uint8 dönüşümünü kendi içinde açıkça sahiplenmeli.
- Offline sample ile online frame'in encoder çıktısının birebir eşleştiği test
  eklenmeli.

**Çözüm:** Public `preprocess_actor_rgb` fonksiyonu HWC `uint8` girdiyi tek
kanonik yoldan CHW `float32 [0,1]` tensöre çeviriyor; `VisionNpzDataset` de aynı
fonksiyonu kullanıyor. Encoder BCHW yanında floating dtype, sonluluk ve `[0,1]`
aralığını fail-fast doğruluyor.

**Doğrulama:** Dönüşümün sayısal eşliği, yanlış NumPy dtype, uint8 BCHW,
float `[0,255]` ve NaN girdileri regression testine dönüştürüldü. Perception
hedefli testleri `19 passed`; Ruff, format ve mypy temiz geçti.

## REV-010 — Semantik renk kısa yolu ve sınırlı görsel çeşitlilik

Train/validation/test profillerinin tamamında engeller kırmızı/turuncu ağırlıklı,
hedef ise yeşil ağırlıklıdır. Sahne ailesi axis-aligned kutular ve ayırt edici
distractor içermeyen zeminle sınırlıdır. Aydınlatma ve ton değişse de sınıf-renk
ilişkisi korunur. Target için renk baseline'ının güçlü olması belgelenmiştir;
engel tarafında benzer bir renk-kısa-yol baseline'ı yoktur.

**Etki:** Encoder gerçek engel geometrisi/derinliği yerine simülatör materyal
rengini öğrenmiş olabilir. Mevcut quality pass, kamera tabanlı gerçekçi engelden
kaçınma için yeterli kanıt değildir.

**Kabul ölçütleri:**

- Engel rengi/texture'ı semantik sınıftan bağımsızlaştırılmalı; engelle aynı
  renkte engelsiz distractor ve farklı renk/texture/geometri engeller eklenmeli.
- Validation/audit tarafında görülmemiş primitive/geometri ve arka planlar
  bulunmalı.
- Renk tabanlı engel baseline'ı veya color-ablation ile kısa yol ölçülmeli.
- Gerçek hedef görünümü kesinleşene kadar sentetik yeşil target sonucu yalnız
  geliştirme metriği olarak etiketlenmeli.

**Çözüm:** Engel RGB aralıkları tüm kanallarda genişletildi. Hedefle aynı renge
sahip fakat hedef/engel maskesine girmeyen çarpışmasız distractor'lar eklendi.
Train/validation/development kutu, silindir, capsule ve sphere; bağımsız audit
ise train-dışı ellipsoid engeller kullanıyor. Eğitim karelerinin yarısına seed'li
grayscale augmentation uygulanıyor. Nominal metriklerin yanında hue bilgisini
kaldıran grayscale ablation free-space ve risk kapısına dahil edildi.

**Doğrulama:** Çoklu primitive/distractor collision ve renk eşliği regression
testleri geçti. Grafik erişimli 8-kare smoke dataset üretildi. İlk audit'in
precision başarısızlığı fail-closed yakalandı; yeni salt'lı audit nominal risk
P/R/S `0.780/0.784/0.694`, grayscale risk P/R/S
`0.744/0.713/0.661` ile geçti. Bu hâlâ sentetik geometri kanıtıdır; gerçek
IMX477 genellemesi değildir.

## REV-011 — Config doğrulamasından geçen çözünürlüklerin çökmesi

`VisionEncoderConfig.validate` yalnız iki boyutun en az 16 olmasını istiyor.
Heatmap koordinat buffer'ları ise sabit `9 x 12`; convolution çıktısı doğrudan
bu buffer'larla çarpılıyor. `64 x 64` ve `80 x 128` config'leri doğrulamadan
geçip forward sırasında sırasıyla 64/108 ve 160/108 boyut uyuşmazlığıyla
çöküyor; varsayılan `72 x 96` çalışıyor.

**Kabul ölçütleri:**

- Feature map heatmap başından önce açıkça `9 x 12` boyutuna adaptif
  örneklenmeli veya config yalnız desteklenen boyutları fail-fast kabul etmeli.
- En az iki varsayılan dışı geçerli çözünürlük ve geçersiz sınır regression
  testine eklenmeli.

**Çözüm:** Convolution feature map'i target/obstacle heatmap başlarından önce
adaptif olarak sabit `9x12` boyutuna örnekleniyor. Latent pooling bağımsız
`3x4` sözleşmesini koruyor.

**Doğrulama:** Varsayılanın yanında `64x64` ve `80x128` config'leri target ve
obstacle için `(B,9,12)`, center için `(B,2)` çıktı verdi. Encoder testleri
`10 passed, 2 subtests passed`; Ruff, format ve mypy temiz geçti.

## REV-012 — Artefact provenance ve taşınabilirlik eksikliği

`quality_report.json` dataset path/schema, manifest/config hash'i veya checkpoint
hash'i taşımıyor. `selected_checkpoint` mutlak yerel dosya yolu; proje taşınırsa
geçersiz olur. Checkpoint yalnız dataset şema adını içeriyor ve trainer kabul
edilen `vision-dataset-v3` sözleşmesini zorunlu kılmıyor; mevcut synthetic test
bilerek `vision-dataset-v1` metadata ile v8 checkpoint üretiyor.

**Kabul ölçütleri:**

- Report dataset schema, manifest/config/statistics SHA-256 ve checkpoint
  SHA-256 içermeli.
- Checkpoint yolu report köküne göre göreli yazılmalı.
- Trainer desteklenen dataset şemalarını açıkça doğrulamalı.
- Sonraki environment yalnız `passed=true` quality manifest'i ve eşleşen
  checkpoint hash'i üzerinden model yüklemeli.

**Çözüm:** Dataset provenance şema/config/metadata/manifest/statistics SHA-256
özetlerini taşır. Quality raporu ve checkpoint aynı provenance'ı içerir; bütün
yollar artefact köküne göre göreli yazılır. Audit quality/checkpoint/dataset hash
eşliğini doğrular. `load_approved_vision_encoder` yalnız başarılı quality + audit
zincirini ve tüm dosya hash'lerini doğrulayarak model yükler. Encoder ve
checkpoint sözleşmesi v9'a çıkarıldı.

**Doğrulama:** Synthetic audit success ve değiştirilmiş checkpoint hash'inin
reddi regression testidir. Gerçek approval manifest üzerinden yüklenen model
`vision-encoder-v9` olarak doğrulandı; dataset ve checkpoint SHA-256 değerleri
manifestte kilitlidir.

## REV-013 — Split oranı ve sahne sayısı tutarsızlığı

Config hash ataması için `0.70/0.15/0.15` fraction bildirirken explicit sahne
sayıları `64/8/8` olduğundan üretilen dataset fiilen `0.80/0.10/0.10`'dur.
Kod her hash kovasını kendi kotası dolana kadar taradığı için fraction alanları
çıktı oranını belirlemez.

**Kabul ölçütleri:**

- Tek kaynak seçilmeli: ya toplam sahne sayısı + fraction'dan kotalar
  türetilmeli ya da explicit kotalar kullanılıp yanıltıcı fraction alanları
  kaldırılmalı.
- Metadata bildirilen ve gerçekleşen oranları ayrı kaydetmeli; config testi
  eşleşmeyi korumalı.

**Çözüm:** Yanıltıcı fraction alanları kaldırıldı. `vision-dataset-v4` için
train/validation/development-OOD/audit explicit sahne kotaları tek kaynaktır;
hash bucket sınırları aynı kotalardan türetilir ve seçici tam bu sayıları üretir.
Eski test split'i `development_ood` olarak yeniden adlandırıldı; trainer ve
quality runner audit split'ini yüklemez.

**Doğrulama:** Test, gerçekleşen her split oranının explicit sahne sayısı/toplam
ile eşit olduğunu ve bütün seed'lerin ayrık kaldığını doğruluyor. Perception
hedefli sonuç `20 passed, 2 subtests passed`; Ruff, format ve mypy temiz.

## REV-001 — Paket kaynaklarının wheel dışında kalması

`pyproject.toml` yalnızca `src/` altındaki Python paketlerini topluyor. Varsayılan yükleyiciler ise proje kökündeki `assets/` ve `configs/` yollarına bağlı. Geçici dizinde üretilen wheel incelendiğinde XML/TOML dosyalarının bulunmadığı ve wheel üzerinden `load_vehicle_config()` çağrısının `FileNotFoundError` ürettiği doğrulandı.

**Kabul ölçütleri:**

- MJCF ve bütün varsayılan TOML dosyaları wheel içinde bulunmalı.
- Kaynak checkout ve kurulu wheel aynı yükleyici API'sini kullanmalı.
- İzole wheel smoke testinde model ve bütün config'ler yüklenebilmeli.

**Çözüm:** MJCF ve varsayılan TOML dosyaları `src/ebabil/assets` ile
`src/ebabil/configs` altına taşındı. Bütün varsayılan yollar paket köküne göre
hesaplanıyor ve dosyalar `pyproject.toml` package-data listesine eklendi.

**Doğrulama:** Geçici dizinde wheel oluşturuldu ve ayrı bir hedef dizine
kuruldu. Kurulu wheel içinden MJCF derlendi; vehicle, controller, sensor,
navigation ve scene config yükleyicileri başarıyla çalıştı. İlgili model,
propulsion ve closed-loop testleri `24/24` geçti.

## REV-002 — Seed akışlarının birbirine bağlanması

Ortam sensör seed'ini `scene_seed + 1` olarak üretiyor. Açık `scene_seed` kullanıldığında farklı Gymnasium episode seed'leri aynı sensör bias/noise serisini oluşturuyor. Ayrıca `dict.get` varsayılanındaki RNG çağrısı açık scene seed verilse bile episode RNG'yi ilerletiyor.

**Kabul ölçütleri:**

- Episode RNG'den ayrı scene ve sensor alt seed'leri türetilmeli.
- Açık scene seed sahneyi sabitlerken farklı episode seed sensör serisini değiştirmeli.
- Aynı episode seed, scene seed ve action dizisi tam tekrarlanabilir kalmalı.
- Kullanılan scene ve sensor seed'leri `info` içinde raporlanmalı.

**Çözüm:** Scene ve sensor seed'leri episode RNG'den bağımsız çekilişlerle
türetiliyor. `options["scene_seed"]` ve `options["sensor_seed"]` ile bileşenler
ayrı ayrı sabitlenebiliyor; değerler negatif olmayan tamsayı olarak doğrulanıyor
ve `info` içine yazılıyor. Açık seed verildiğinde gereksiz RNG tüketimi kaldırıldı.

**Doğrulama:** Aynı scene seed ile `seed=1` ve `seed=999` farklı sensor seed ve
farklı imperfect observation üretti. İki bileşen seed'i de açık verildiğinde
farklı episode seed'lerine rağmen observation birebir tekrarlandı. Environment ve
sensor hedefli testleri `20/20` geçti.

## REV-003 — Sahne engelleme ve rota geometrisi

`direct_path_is_blocked` Z eksenini yok sayıyor ve kutu yerine kutuyu çevreleyen daireye dayalı alt sınır kullanıyor. Bu nedenle gerçekte açık bir rota kapalı bildirilebiliyor. Zorunlu blocker yüksekliği de gerekli yüksekliği mevcut maksimuma sessizce kırpabildiği için bazı geçerli özel config'lerde drone engelin üzerinden düz uçabiliyor.

**Kabul ölçütleri:**

- Doğrudan rota, araç açıklığıyla genişletilmiş 3B AABB üzerinde kesin segment kesişimiyle değerlendirilmeli.
- Konfigürasyon blocker'ın gerekli yüksekliğini sağlayamıyorsa açık hata vermeli.
- Engelli modlar döndürülmeden önce doğrudan rota koşulu açıkça doğrulanmalı.
- Grid rotasındaki her kenar sürekli geometriye karşı kontrol edilmeli.
- Yanlış pozitif XY örneği ve yüksek irtifa örneği regression testine dönüşmeli.

**Çözüm:** Yaklaşık merkez-yarıçap hesabı kaldırıldı ve sonlu segment–AABB
kesişimi için slab yöntemi eklendi. Doğrudan rota 3B, grid kenarları 2B
genişletilmiş kutularla kesin olarak kontrol ediliyor. Başlangıç/hedef ile en
yakın grid düğümü arasındaki bağlantılar da doğrulamaya dahil edildi. Zorunlu
blocker gerekli yüksekliğe ulaşamıyorsa config açık `ValueError` üretiyor ve
engelli sahne döndürülmeden önce doğrudan rota koşulu ayrıca kontrol ediliyor.

**Doğrulama:** Önceden yanlış biçimde kapalı sayılan `0.4 m` açıklıklı offset
kutu artık açık; uçuş çizgisi `z=2.0 m`, engel tepesi `0.8 m` olan örnek açık;
yetersiz blocker config'i reddediliyor. Düğümler arasında kalan ince engeli
geçen sahte grid yolu da reddedildi. Scene, environment ve sensor hedefli
testleri `29/29` geçti.

## REV-004 — Observation ve ground-truth belgesi

Observation v1, ideal MuJoCo konum/orientation/hız değerlerini kullanıyor; yalnızca rangefinder `imperfect` modda bozuluyor. ADR-008 ise ground-truth bilgisinin actor'a verilmediğini ima ediyor. İlerleme günlüğü gerçek davranışı doğru biçimde “state/IMU ideal” olarak kaydediyor.

**Kabul ölçütleri:**

- ADR-008 ve proje tasarımı mevcut ideal-state baseline'ını açıkça belirtmeli.
- Bunun sim-to-real observation olmadığı ve estimator/noise çalışmasının sonraki faz olduğu yazılmalı.

**Çözüm:** ADR-008 ve literatür sentezi observation v1'in ideal MuJoCo
konum/orientation/hız sensörlerini kullandığını, imperfect modun yalnızca
rangefinder'ı bozduğunu ve bu baseline'ın sim-to-real estimator doğrulaması
olmadığını açıkça belirtecek şekilde güncellendi.

**Doğrulama:** ADR-004, ADR-006, ADR-008, proje tasarımı, literatür sentezi ve
ilerleme günlüğündeki state/IMU ifadeleri çapraz karşılaştırıldı; mevcut kod
davranışıyla çelişen “ground-truth yalnızca reward” ifadesi kalmadı.

## REV-005 — Ortam sınır doğrulamaları

`max_episode_steps=0`, truthy/falsy seçimi nedeniyle reddedilmek yerine configured değere dönüşüyor. Policy frekansının MuJoCo timestep ile tam bölünebilirliği constructor/config doğrulamasında değil, ilk reset sırasında anlaşılıyor.

**Kabul ölçütleri:**

- Episode adım sınırı yalnızca `None` olduğunda varsayılmalı; bool, sıfır, negatif ve tamsayı olmayan değerler reddedilmeli.
- Policy/fizik oranı ortam oluşturulurken doğrulanmalı.
- Geçersiz sınırlar için regression testleri bulunmalı.

**Çözüm:** `max_episode_steps` yalnızca `None` olduğunda config değerine
dönüyor; diğer değerler pozitif tamsayı olarak doğrulanıyor. Temel MJCF timestep
değeri model derlenmeden `MjSpec` üzerinden okunuyor ve policy oranı constructor
aşamasında hesaplanıyor. Reset sırasında derlenen sahne modelinin aynı timestep'i
koruduğu ayrıca denetleniyor.

**Doğrulama:** `0`, `-1`, `1.5` ve `True` episode sınırları reddedildi. `30 Hz`
policy frekansı ilk reset yerine constructor'da reddedildi. Navigation ve model
hedefli testleri `23/23` geçti.

## REV-006 — Araç ve plan dokümantasyonu

Implementasyon planı Faz 0'da `pytest`, formatlama, statik analiz ve CLI giriş noktaları öngörüyor; mevcut proje standart kütüphane `unittest` komutunu kullanıyor ve lint/type araçları tanımlı değil. Bu fark ilerleme günlüğünde kısmen sınırlılık olarak kayıtlı olsa da Faz 0 durumu tamamlandı görünüyor. Ayrıca hareketli `master/main/stable` dokümantasyon bağlantıları kurulu bağımlılık sürümlerinden zamanla ayrılabilir.

**Kabul ölçütleri:**

- Projenin kanonik test/lint/type-check komutları tek yerde tanımlanmalı.
- Plan ve ilerleme günlüğü gerçek araç seçimiyle uyumlu olmalı.
- Sürüm davranışına dayanan teknik bağlantılar mümkün olduğunda sürümlenmeli.

**Çözüm:** `pytest==9.1.1`, `ruff==0.16.7` ve `mypy==2.3.1` geliştirme
bağımlılıkları olarak eklendi ve transitive bağımlılıkları lock dosyasına
sabitlendi. Ruff lint/format ile mypy üretim-kodu ayarları `pyproject.toml`
içinde tanımlandı. README ve agent protokolü kanonik komutlarla güncellendi;
SB3 bağlantıları kurulu `2.9.0` sürümüne sabitlendi. Script giriş noktalarının
mevcut faz için kanonik olduğu, paket CLI entry-point'lerinin kararlı kullanıcı
akışından sonra ekleneceği plana işlendi.

**Doğrulama:** Ruff lint başarılı, `42` dosya formatlı, mypy `25` üretim
dosyasında hatasız ve pytest `66` test + `7` subtest ile başarılıdır. Editable
`.[dev]` kurulumu ve `pip check` de başarılıdır.

## İnceleme başlangıç doğrulaması

```text
unittest: 58/58 başarılı
compileall: başarılı
pip check: bozuk bağımlılık yok
hover smoke: başarılı
clear navigation smoke: hedefe ulaştı
single_blocker smoke: beklenen obstacle_collision
wheel smoke: başarısız; paket kaynakları eksik
```

Her düzeltmenin uygulama notu ve hedefli doğrulama sonucu bu belgede ilgili bulgunun altına eklenecektir.

## REV-028 — Konum tutma hassasiyetinin kaybı ve normalizasyonun terminal gradyanı zayıflatması

**Tarih:** 12 Eylül 2026  
**Kaynak:** REV-023/026/027 düzeltmeleri sonrası üçüncü D1 koşusu
(`runs/d1-rev026-seed20260912`)  
**Etkilediği:** ADR-017 reward semantiği, ADR-019 görsel yer bulma iddiası

### Bulgu

Üçüncü koşu D1 kapısını geçmedi **ve başarıyı kanıt koşusunun altına düşürdü.**
40 episode, deterministik, seed `520001`, aynı D1 sahnesi:

| Checkpoint | Başarı | Çarpışma | Abort | Timeout |
|---|---:|---:|---:|---:|
| kanıt koşusu `final` (eski reward) | **`%42.5`** | `%15.0` | `%5.0` | `%35.0` |
| yeni `final` | `%22.5` | `%15.0` | `%12.5` | `%50.0` |
| yeni `best` | `%20.0` | `%12.5` | `%7.5` | `%60.0` |

20 episode / seed `410001` ölçümü de aynı yönü verdi (eski `%25`, yeni `final`
`%10`, yeni `best` `%20`). Başarı oranı seed'e çok duyarlıdır; ölçümün sağlam
kısmı aşağıdaki adım bazlı dağılımdır.

### Bağlayıcı koşul ölçüldü: yatay konum

`approach`/`settle`/`hold` durumlarındaki her adımda beş hover koşulu tek tek
sınandı (`final` checkpoint, 20 episode, ~5-6 bin adım):

| İhlal edilen koşul | kanıt koşusu | yeni koşu |
|---|---:|---:|
| **güvenli bölge dışı (yatay)** | `%71.3` | **`%90.9`** |
| irtifa toleransı | `%65.0` | `%17.4` |
| hız | `%26.5` | `%17.6` |
| açısal hız | `%15.4` | `%8.9` |
| tilt | `%3.8` | `%1.9` |
| | | |
| yatay mesafe medyanı | `0.879 m` | `1.065 m` |
| `<= 0.650 m` içinde geçen adım | `%25` | `%8` |
| hold halkasında geçen adım | `%16` | `%6` |

Dört koşulun dördü de iyileşti; yalnız **yatay konum** kötüleşti ve bağlayıcı
kısıt odur. Halka REV-023 ile `0.108 -> 0.243 m` genişletildiği hâlde halkada
geçen süre `%16 -> %6` düştü, yani gerileme geometriden değil kontrolden gelir.

### Kök neden 1: araç waypoint'in üzerinde bile duramıyor

**Bu bölümün ilk yazımı yanlıştı ve düzeltilmiştir.** İlk teşhis "araç inanılan
konuma park ediyor ve son `1.3 m`'yi görüntüyle kapatmıyor" idi; dayanağı
terminal yatay mesafe medyanının (`1.316 m`) ortalama hedef hatasına (`1.338 m`)
çok yakın olmasıydı. Bu bir çakışmaydı, kanıt değildi. Ayrı ölçüm yapıldı:
`HoverConditionReading.believed_horizontal_distance_m`, yani aracın
**waypoint'e** olan mesafesi, gerçek hedefe olan mesafeden ayrı kaydedildi
(40 episode, deterministik, seed `520001`).

| | kanıt koşusu `final` | üçüncü koşu `final` |
|---|---:|---:|
| **waypoint'e** mesafe medyanı | `0.591 m` | `0.991 m` |
| gerçek hedefe mesafe medyanı | `0.653 m` | `0.917 m` |
| waypoint'e en yakın anın medyanı | `0.417 m` | `0.661 m` |
| waypoint'in `0.65 m` içinde kalan episode | `%53` | `%25` |

Araç waypoint'ten de `~1 m` uzaktadır. Yani sorun **bilgi eksikliği değil,
konum tutma hassasiyetidir**: politika waypoint'i biliyor ve yine de üzerine
oturamıyor. Kanıt koşusundaki politika bunu `0.591 m` ile yapıyordu.

### Saf waypoint takibinin tavanı ölçüldü

Hedef hatası diskte düzgün dağıldığı için, waypoint'in **tam** üzerinde duran
kusursuz bir politikanın başarı tavanı:

```text
P(hata <= 0.65 m) = (0.65 / 2.0)^2 = %10.6
```

Ölçülen değerler bunun üzerindedir, yani iki politika da kamerayı bir ölçüde
kullanıyor:

| | hata `<= 0.65 m` (kolay) | hata `> 0.65 m` (kamera gerekli) | toplam |
|---|---:|---:|---:|
| kanıt koşusu `final` | `5/8` = `%62` | `12/32` = **`%38`** | `%42.5` |
| üçüncü koşu `final` | `1/8` = `%12` | `8/32` = `%25` | `%22.5` |

`%38` ve `%25`, `%10.6` tavanının üzerindedir. Yani ADR-019'un asıl iddiası
(kameranın yer bulmaya katkı vermesi) **çalışıyor**; gerileme o eksende değil,
konum tutma hassasiyetinde.

İki sınır ayrı ayrı akılda tutulmalıdır:

1. **Şimdiki kısıt:** konum tutma. Araç waypoint'e `0.99 m` uzakta duruyor,
   `0.65 m` gerekiyor. Bu bir kontrol/teşvik sorunudur ve düzeltilebilir.
2. **Tasarım sınırı:** waypoint kasıtlı olarak `2 m`'ye kadar yanlıştır ve
   başarı **gerçek** hedefe göre ölçülür. Kusursuz waypoint takibi bile `%10.6`
   tavanındadır. Kalanı kameranın işidir ve bu projenin amacıdır, bir defekt
   değildir.

### Kök neden 2: normalizasyon son metrenin değerini düşürdü

REV-026 düzeltmesi progress'i rota uzunluğuna böler. Toplam korunur fakat
**yoğunluk** rota uzunluğuyla ters orantılı hâle gelir:

```text
eski  : 5.0/m sabit                          -> son 1.0 m = 5.00
yeni  : 15.0 / 10.70 m = 1.40/m  (D1)        -> son 1.0 m = 1.40
yeni  : 15.0 / 100 m   = 0.15/m  (D3)        -> son 1.0 m = 0.15
```

Yani düzeltme, çarpışma teşvikini doğru biçimde kapatırken **kameranın yapmak
zorunda olduğu işin ödülünü `3.6` kat azalttı.** D3'te aynı iş `0.15` değerinde
olurdu, yani pratikte hiç. Normalizasyon tek başına kalıcı bir tasarım değildir.

`target_visible` (`+0.005/adım`) uzakta beklemeyi kârlı yapmaz — `time_penalty`
ile net `-0.005/adım`'dır — fakat `1 m` uzakta beklemek ile `0.5 m`'de beklemek
arasında **hiçbir ayrım yapmaz**. Yaklaşmanın tek karşılığı `target_progress`'tir
ve onun yoğunluğu `1.40/m`'ye inmiştir.

### Önerilen düzeltme

Progress'i iki menzile ayır. Toplamın rota bağımsızlığı korunur (REV-026), son
metrenin yoğunluğu geri gelir:

```text
d > approach_radius_m (2.5 m) : w_cruise * delta_d / L_plan     (rotaya normalize)
d <= approach_radius_m        : w_terminal * delta_d            (sabit, m başına)
```

Terminal parçanın uzunluğu `approach_radius_m = 2.5 m` ile üst sınırlıdır, bu
yüzden sabit ağırlık rota bağımsızlığını bozmaz: terminal toplam en fazla
`w_terminal * 2.5`. REV-026 invariantı korunur:

```text
collision_penalty (25.0) > w_cruise + w_terminal * 2.5
```

`w_cruise = 10.0` ve `w_terminal = 5.0` ile toplam `22.5 < 25.0` olur ve son
metre eski koşudaki `5.0` değerine döner. Bu sayılar öneridir; kabul ölçütü
yine davranıştır.

### Doğrulama kapısı

- `L_plan` 2 / 10 / 65 / 500 m için toplam progress rota bağımsız kalmalı.
- `collision_penalty > toplam progress üst sınırı` invariantı testlenmeli.
- Terminal menzildeki metre başına ödül `L_plan`'dan **bağımsız** olmalı.
- Ölçülecek davranış, **waypoint'e** olan mesafe medyanının `0.591 m`'nin
  (kanıt koşusu) altına inmesidir. Bu, bilgi sınırından bağımsız ve doğrudan
  kontrol hassasiyetini ölçen bir sayıdır.
- İkinci ölçüt, hata `> 0.65 m` olan episode'larda başarının `%38`'in üstüne
  çıkması: kameranın yer bulmaya katkısı ancak orada görünür.
- Başarı oranı tek başına yeterli ölçüt değildir; aynı politika iki seed'de
  `%25` ve `%42.5` verdi.

### Not: hangi değişiklik neyi yaptı

Üç değişiklik aynı koşuda uygulandığı için katkılar ayrıştırılamaz, fakat
ölçümler yönleri ayırt etmeye yetiyor:

| Değişiklik | Ölçülen etki |
|---|---|
| REV-023 (`hover_height_m 1.5`) | irtifa ihlali `%65 -> %17`; halka `2.25` kat |
| REV-027 (`target_visible`) | eğitimde `target_lost` abort `%30 -> %6` (son 300 ep) |
| REV-026 (normalizasyon) | çarpışma eğitim ortasında `%20 -> %7`; **konum tutma gerilemesi** (waypoint'e `0.591 -> 0.991 m`) |

REV-026 ve REV-027 ikisi de kendi eksenlerinde çalıştı. Gerileme
normalizasyonun yan etkisidir ve yukarıdaki iki menzilli form onu geri almadan
düzeltir.

### Bir şey ilk kez oldu

Eğitim sırasında **tam `3.00 s` hold** ilk kez ulaşıldı (son 300 episode'da 3
episode). Kanıt koşusunda aynı pencerede maksimum `0.55 s`'ydi ve `3 s` hiç
görülmedi. Terminal beceri artık öğrenilebilir durumdadır; eksik olan ona
yeterince sık ulaşmak.

**Durum:** Gerekçesi ortadan kalktı; REV-029'a bakınız. Bu bulgunun **ölçümü**
(araç waypoint'in üzerinde duramıyor, `0.591 -> 0.991 m`) geçerliliğini korur ve
değerlidir. Fakat önerdiği düzeltme (iki menzilli progress) koşuldu ve
navigasyonu tamamen bozdu (başarı `%0`, `target_progress` toplamı `+4.26`).
Asıl neden REV-026'nın kendisinin yanlış olmasıydı: normalizasyon hiç
yapılmamalıydı. Kod REV-026 öncesi forma geri alındı ve bu bulgunun ölçtüğü
gerileme de böylece ortadan kalkar.

Özgün öneri (uygulandı, geri alındı):

Progress iki menzile ayrıldı. Tek ağırlık (`target_progress_weight = 15.0`)
yerine iki ağırlık geldi:

```text
d > R   :  cruise_progress_weight   * delta_d / L_plan     w_cruise   = 10.0
d <= R  :  terminal_progress_weight * delta_d              w_terminal =  5.0
R = terminal_hover.approach_radius_m = 2.5 m  (tek kaynak, ayrı config alanı yok)
```

Sınırı geçen adımda fark sınırda ikiye bölünür, yani iki parçanın toplamı her
zaman `d_önce - d_sonra` değerine eşittir ve hiçbir mesafe "kaybolmaz".

| | tam normalize | iki menzilli |
|---|---:|---:|
| son 1 m, D1 (`10.7 m`) | `1.40` | **`5.00`** |
| son 1 m, D3 (`100 m`) | `0.15` | **`5.00`** |
| toplam üst sınırı | `15.00` | `22.50` |
| `collision_penalty` | `25.00` | `25.00` |

REV-026 invariantı korunur (`22.5 < 25.0`) ve artık **iki yerde** zorlanır:
`TerminalHoverRewardConfig.validate()` ağırlıkları, `VisionNavigationEnvConfig`
ise yarıçapla birlikte toplam sınırı sınar. Yani bir agent ağırlığı sessizce
yükseltirse config yüklenmez.

`progress_reward_bound()` adı bilinçli seçildi: eski ad
(`route_independent_progress_total`) yanıltıcıydı, çünkü gerçek toplam `L_plan`
ile artar ve sınıra yalnız `L_plan >> R` iken yaklaşır. Rota bağımsız olan şey
toplam değil **sınırdır**.

Testler (6 yeni): toplamın rota uzunluğuyla doğrusal büyümemesi (`500 m` toplamı
`10 m` toplamının `1.5` katından az), terminal yoğunluğun `L_plan`'dan
bağımsızlığı, sınır geçişinde farkın bölünmesi, invariantın config
yüklemesinde fail-closed olması, geçersiz menzil değerlerinin reddi.

**Davranışsal kabul ölçütü** (`runs/d1-rev028-seed20260912`, `600k`, aynı seed,
aynı `v9` encoder):

1. **waypoint'e** mesafe medyanı `0.591 m`'nin (kanıt koşusu) altına inmeli.
   Bilgi sınırından bağımsız, doğrudan konum tutma hassasiyetini ölçer.
2. Hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı.
   Kameranın yer bulmaya katkısı ancak orada görünür.
3. Çarpışma oranı trendi azalan kalmalı (REV-026 gerilememiş olmalı).

## REV-029 — REV-026'nın yanlış teşhis olduğunun kanıtlanması

**Tarih:** 12 Eylül 2026  
**Kaynak:** Dördüncü D1 koşusu (`runs/d1-rev028-seed20260912`) ve üç koşunun
monitor verisinin geriye dönük analizi  
**Etkilediği:** REV-026 (geçersiz kılar), REV-028 (gerekçesini ortadan
kaldırır), ADR-017 reward semantiği

### Bulgu

REV-026 şunu iddia ediyordu: `target_progress` rota uzunluğuyla doğrusal
büyüdüğü (`5.0 * L`) ve `collision_penalty` sabit (`25.0`) olduğu için, uzun
rotalarda engele çarpıp kısa yoldan gitmek net kazançlı hâle gelir. Kanıt olarak
çarpışma oranının eğitim boyunca artması gösteriliyordu (`%15 -> %20`).

**İddianın iki ayağı da ölçümle çürütüldü.**

### Çürütme 1: çarpışma uzun değil KISA rotalarda daha sık

Rota uzunluğu `segment_budget_seconds`'tan geri hesaplanarak üç koşunun tamamı
episode bazında sınıflandırıldı:

| Rota | 1. koşu (`5.0/m` sabit) | 3. koşu (`1.40/m` normalize) | 4. koşu (iki menzilli) |
|---|---:|---:|---:|
| `5-8 m` | **`%23.3`** | **`%24.2`** | **`%22.5`** |
| `8-11 m` | `%15.1` | `%12.3` | `%10.5` |
| `11-15 m` | `%7.6` | `%7.5` | `%8.3` |

REV-026 bunun tam tersini öngörüyordu. Dahası desen **üç farklı ödül
tasarımında da neredeyse aynıdır**, yani çarpışma oranı progress ağırlığına ve
normalizasyona duyarlı değildir. Çarpışmanın nedeni ödül ölçeği değil
geometridir: `20 x 20 m` workspace'te `5-8 m`'lik bir rotada, yola yerleştirilen
blocker manevra alanının çok daha büyük bir kısmını kapatır.

### Çürütme 2: eğitim boyunca artış survivorship'ti

`%15 -> %20` artışı, koşullu oranlar hesaplanınca kayboluyor:

| Pencere | çarpışma (ham) | hedef bölgesine ulaşan | **ulaşanlar arasında çarpışma** | rota `5-8 m` çarpışma |
|---|---:|---:|---:|---:|
| 1 | `%15` | `%53` | `%27` | `%23` |
| 2 | `%12` | `%66` | `%12` | `%21` |
| 3 | `%13` | `%60` | `%15` | `%22` |
| 4 | `%13` | `%66` | `%18` | `%23` |
| 5 | `%17` | `%69` | `%17` | `%26` |
| 6 | `%20` | `%74` | `%18` | `%25` |

Hedef bölgesine **ulaşan** episode oranı `%53 -> %74` yükseldi, çünkü politika
uçmayı öğrendi ve `out_of_bounds` / `minimum_altitude` başarısızlıkları düştü.
Daha fazla episode engel yoğun terminal bölgeye kadar hayatta kaldı. Ulaşanlar
arasında çarpışma oranı **sabit** kaldı (`%27, %12, %15, %18, %17, %18`) ve
sabit rota aralığında da sabit kaldı (`%23, %21, %22, %23, %26, %25`).

Yani politika çarpışmayı öğrenmiyordu; daha çok episode çarpışabileceği yere
kadar gidiyordu.

### Maliyeti

REV-026 teşhisine dayanan iki düzeltme iki koşu harcadı ve ikisi de geriletti:

| Koşu | progress formu | waypoint'e mesafe | Başarı |
|---|---|---:|---:|
| 1 (kanıt) | `5.0/m` sabit | `0.591 m` | **`%42.5`** |
| 3 | `15.0 * dd / L` (`1.40/m`) | `0.991 m` | `%22.5` |
| 4 | `10.0/L` + `5.0/m` (`< 2.5 m`) | `1.709 m` | `%0.0` |

4. koşuda `target_progress` toplamı episode başına yalnız `+4.26` oldu (3. koşu
`+11.71`): kruvaziyer yoğunluğu `1.40/m`'den `0.93/m`'ye indirmek navigasyonu
tamamen bozdu. Araç hedefe hiç varmadı.

### Doğru invariant yoğunluk üzerindedir

REV-026'nın invariantı (`toplam progress < collision_penalty`) yanlış kurulmuştu.
**Çarpışma episode'u bitirir**, yani politikanın kıyaslaması "episode toplamı"
değildir; kısa yolun **yerel** kazancıdır:

```text
kısa yol kazancı = w_metre * dolaşma_mesafesi
kısıt            : w_metre * dolaşma_mesafesi <= collision_penalty
```

`5.0/m` ve `5 m` dolaşma ile `25.0 <= 25.0`. Bu kısıt rota uzunluğundan
**doğal olarak bağımsızdır**, çünkü yoğunluk `L`'ye hiç bakmaz. Normalizasyon
gereksizdi ve zararlıydı: `L`'ye bölmek, yoğunluğu `L` ile ters orantılı yapıp
hem navigasyonu hem terminal hassasiyeti bozdu.

Uygulama: `TerminalHoverRewardConfig.detour_saving_for(detour_length_m)`.

### Yapılan değişiklik

`target_progress` REV-026 öncesi formuna geri alındı:

```text
target_progress = 5.0 * (d_önce - d_sonra)     rotaya normalize EDİLMEZ
```

`cruise_progress_weight` / `terminal_progress_weight` ayrımı,
`planned_route_length_m` ve `terminal_progress_radius_m` parametreleri,
`progress_reward_bound()` ve env-config invariant kontrolü kaldırıldı.

**Geri alınmayanlar** (ikisi de kendi ekseninde ölçülerek fayda gösterdi):

| Değişiklik | Ölçülen etki | Durum |
|---|---|---|
| REV-023 `hover_height_m = 1.5` | irtifa ihlali `%65 -> %17`, halka `2.25` kat | korunuyor |
| REV-027 `target_visible = 0.005` | `target_lost` abort `%30 -> %6` | korunuyor |

Bu üçlü (`5.0/m` progress + `1.5 m` hover + görünürlük ödülü) **hiç
koşulmamıştır** ve doğrulanması gerekir.

### Açık kalan gerçek kısıt (REV-026'nın yerine)

Normalizasyonun çözmeye çalıştığı **başka** bir sorun gerçektir fakat çarpışmayla
ilgili değildir: `5.0/m` yoğunlukla D3'te (`100 m`) episode toplamı `~500`,
`success_bonus` ise `30.0`'dır. Başarı sinyali progress içinde boğulur. Bu bir
**merdiven ölçekleme** sorunudur ve çözümü progress'i bölmek değil, `success` ile
`hold` ödüllerini mesafe seviyesine göre ölçeklemektir. D3 öncesinde ele alınmalı;
D1'de bağlayıcı değildir (`10.7 m` rotada toplam `~53` vs `30`).

### Doğrulama kapısı

- Progress yoğunluğunun sabit ve `L`'den bağımsız olması.
- `detour_saving_for(d) <= collision_penalty` invariantının `d <= 5 m` için
  sağlanması.
- Kabul ölçütü: **waypoint'e** mesafe medyanı `0.591 m`'nin altına inmeli ve
  hedef hatası `> 0.65 m` olan episode'larda başarı `%38`'in üstüne çıkmalı.

**Durum:** Teşhis çürütüldü ve kod geri alındı. Doğrulama koşusu bekliyor.

## REV-030 — Görsel zemin düzleminin workspace'i kapsamaması (D1'in asıl blocker'ı)

**Tarih:** 12 Eylül 2026  
**Kaynak:** Kullanıcının D1 ortamını MuJoCo viewer'da gözlemlemesi ve
"belki sorun engel yerleşimindedir" sorusu  
**Etkilediği:** ADR-011 kamera, ADR-015 vision observation, ADR-018 ölçek
merdiveni, ADR-019 görsel arama, bütün D1 sonuçları

### Bulgu

`quadrotor.xml` içindeki zemin geom'i şudur:

```xml
<geom name="ground" type="plane" size="5 5 0.1" material="ground_material"/>
```

MuJoCo'da bir `plane` geom'in `size` parametreleri **yalnız render** içindir;
çarpışmada düzlem **sonsuzdur**. İkisi de ölçülerek doğrulandı:

```text
fizik  : x = 0 / 4 / 9 / 15 m'de zemin teması var  -> düzlem sonsuz, fizik doğru
render : x,y > ~4 m'de kamera hiçbir zemin pikseli görmüyor, derinlik 532.81 m
         (uzak kırpma düzlemi), yani tam boşluk
```

Yani zemin **görsel olarak `10 x 10 m`**, D1 workspace'i ise `20 x 20 m`
(`[-10, 10]^2`). Kapsama alanın `%25`'i.

### Ölçülen etki

300 D1 sahnesi (`single_blocker`, `random_both`):

```text
başlangıç zeminin dışında                  : %70
HEDEF zeminin dışında                      : %65
rotanın zemin dışı kısmı                   : medyan %39, ortalama %46
rotası tamamen zemin içinde olan episode   : %7
rotası tamamen zemin dışında olan episode  : %16
```

Kamera kareleri (40 rastgele poz, irtifa `0.80-2.10 m`, boşluk piksel oranı;
`%50` civarı normaldir çünkü kadrajın üst yarısı gökyüzüdür):

| Poz ailesi | boşluk piksel | **hiç zemin görmeyen poz** |
|---|---:|---:|
| encoder dataset (legacy `6 x 5 m`) | `%61.8` | **`%0`** |
| D1 (`20 x 20 m`) | `%86.7` | **`%52`** |

**Onaylı encoder legacy `6 x 5 m` workspace'te eğitildi** ve o workspace
görsel zeminin tamamen içindedir. Yani encoder "zemin yok" karesini **hiç
görmedi**; D1'de pozların yarısında tam olarak bunu görüyor.

### Neden bu D1'in asıl blocker'ı

Bu bulgu, daha önce ayrı ayrı açıklamaya çalıştığım gözlemlerin tamamını tek
başına açıklıyor:

| Gözlem | Bu bulguyla açıklaması |
|---|---|
| D0-V `20/20` başarı, D1 `%25-42` | D0-V workspace'i (`6 x 5 m`) zeminin **tamamen** içinde |
| Başarının seed'e aşırı duyarlılığı (`%25` vs `%42.5`) | Episode'un merkeze yakın düşüp düşmemesine bağlı |
| `target_lost` abort oranının yüksekliği | Hedef pedi render edilse de çevresinde zemin yok; encoder OOD |
| Hiçbir reward değişikliğinin işe yaramaması | Sorun ödülde değil **observation**'da |
| Çarpışmanın kısa rotalarda yoğunlaşması | Kısa rotalar daha sık köşelerde: `5-8 m`'de hedefin `%61`'i zemin dışı |

Reward üzerinde yapılan dört koşuluk çalışma (REV-026 → REV-029) bu kırık
observation'ın üzerinde yürütüldü. REV-029'un "çarpışma oranı ödül tasarımına
duyarlı değil" ölçümü de bununla tutarlıdır: bağlayıcı kısıt ödül değildi.

### Düzeltme

Zemin yarı-boyutu sahne config'ine bağlandı: `SceneGeneratorConfig.ground_half_size_m`.
`build_scene_model(..., ground_half_size_m=None)` MJCF değerini korur.

```text
gereken yarı-boyut = workspace köşesi + kamera zemin izinin tavan irtifasındaki
                     uzak sınırı (3.71 * irtifa, ADR-016)
legacy : 3.0  + 3.71 * 2.5 = 12.28 m   -> UYGULANMADI (aşağıya bakın)
D1     : 10.0 + 3.71 * 3.0 = 21.13 m   -> uygulandı
```

**Legacy bilinçli olarak `5.0 m`'de bırakıldı.** ADR-008 legacy sahnesi ve
onaylı encoder zinciri (`vision_dataset_v4b/v4c`, `vision-encoder-v9/v10`) bu
değerle üretildi; değiştirmek approval zincirini geçersiz kılar ve dataset →
kalite (3 seed) → audit yeniden koşulmasını gerektirir. Legacy config'te alan
yoktur, varsayılan korunur.

Doğrulama:

```text
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23        değişmedi
legacy kamera boşluk oranı   %61.8 -> %61.8        değişmedi (bit-bit)
D1 kamera boşluk oranı       %86.7 -> %51.1
D1 hiç zemin görmeyen poz    %52   -> %0
pytest 253 passed, ruff temiz, mypy temiz
```

### Doğrulama: eski politika düzeltilmiş zeminde çöküyor

Kanıt koşusunun `final` checkpoint'i (kırık zeminde en iyi politika) düzeltilmiş
zeminde yeniden değerlendirildi. Aynı checkpoint, aynı 40 episode, aynı seed:

| Ortam | Başarı | Çarpışma | Timeout |
|---|---:|---:|---:|
| kırık zemin (eğitildiği ortam) | `%42.5` | `%15.0` | `%35.0` |
| düzeltilmiş zemin | `%12.5` | **`%42.5`** | `%40.0` |

Politika **boşluğa uyum sağlamıştı.** Zemin görünür hâle gelince davranışı
dağılıyor ve çarpışma neredeyse üç katına çıkıyor. Muhtemel mekanizma:
encoder'ın `free_space` ve `collision_risk` çıktıları boşluk karesinde ile
dokulu zemin karesinde tamamen farklı; politika birincisine göre kalibre
olmuştu.

Bu, bulgunun ciddiyetinin bağımsız doğrulamasıdır ve şu sonucu getirir:
**REV-023'ten REV-029'a kadar ölçülen bütün D1 sayıları geçersizdir.**
`runs/d1-rev030-seed20260912` projenin ilk geçerli D1 ölçümüdür.

### Yan bulgu: engel yerleşimi ezberlenebilir

Kullanıcı "waypoint hep aynı yerde mi kalıyor, ajan ezberlemez mi?" diye sordu.
Ölçüm, hedef tarafında ezberleme riski **olmadığını** gösterdi:

```text
episode başına farklı scene_seed        : 1760/1760 (hiç tekrar yok)
hedef dağılımı                         : x -8.85..+8.99, y -8.93..+8.92 m
                                         std 5.14 / 5.15 m (düzgün ~5.77)
20x20 m'yi 4x4 hücreye böl -> dolu     : 16/16
hedef hatası dağılımı                  : düzgün disk, izotropik
                                         (ortalama yarıçap 1.327, teori 1.333;
                                          8 yön dilimi 612-641 arası)
eğitim / evaluation seed kesişimi      : 0
```

Fakat **engel** tarafında risk gerçektir. `_blocking_obstacle` engeli her zaman
rotanın **tam ortasına** ve **tam çizgi üzerine** koyar; yalnız boyutu
rastgeledir (`0.3-0.9 m` genişlik, `1.2-2.4 m` yükseklik) ve hedefin hover
zarfını boşaltmak gerekirse kesir `0.5`'ten `0.85`'e kaydırılır. Yanal sapma
300 sahnede `0.000000 m` ölçüldü.

Yani politika "yolun yarısında yana kaç" kuralını kamerayı kullanmadan
öğrenebilir. Engelden kaçınma bu kurulumda amaçlandığı kadar sınanmıyor.

Engeli kaydırarak yapılan ilk sonda ezberlemeyi **kanıtlamadı** (20 episode,
düzeltilmiş zemin): kesir `%30`'da çarpışma `%20`, `%50`'de `%40`, `%70`'te
`%55`. Desen "yarıyı ezberledi"den çok **tepki mesafesi**ne uyuyor — engel
başlangıca yaklaştıkça kaçmak için yol azalıyor. Ayrıca sonda, politikanın
eğitildiği zeminden farklı bir zeminde koştuğu için karışıktır.

Önerilen düzeltme: blocker'ın rota üzerindeki kesrini ve yanal offset'ini
rastgeleleştirmek. `BLOCKER_MAXIMUM_PATH_FRACTION = 0.85` üst sınırı ve hover
zarfı kısıtı korunur. Bu ayrı bir bulgu olarak ele alınmalıdır.

### Açık kalan: encoder hâlâ legacy dağılımında eğitilmiş

Düzeltme D1 karelerini encoder'ın **eğitim dağılımına doğru** taşır
(`%86.7 -> %51.1`, legacy `%61.8`'in yakınına) fakat encoder hâlâ `6 x 5 m`
workspace'te, `5 m` zeminle, `+-1.25 m` yanal offset ve `0.80-2.10 m` irtifa
bandında üretilmiş bir dataset'le eğitildi. D1'in `20 x 20 m` geometrisi ve
uzak mesafeleri bu dağılımın dışındadır.

Yani bu düzeltme D1'i **çözülebilir** hâle getirir, "çözülmüş" hâle getirmez.
Dataset yenilemesi REV-019 ve ADR-019 §6'da zaten D3 öncesi zorunlu
sayılmıştı; bu bulgu onu D1 için de gündeme getirir. Ölçüm sırası:

1. Düzeltilmiş ortamda mevcut encoder ile D1 koşusu (başlatıldı:
   `runs/d1-rev030-seed20260912`).
2. Sonuç hâlâ kapının altındaysa, dataset'i D1 workspace'inde ve doğru zeminle
   yenileyip approval zincirini tekrar kurmak.

### Doğrulama kapısı

- `ground_half_size_m` verilmediğinde MJCF değerinin korunması (legacy donuk).
- `required_ground_half_size_m()` fonksiyonunun workspace ve tavan irtifasından
  türetilmesi.
- D1 sahnesinde workspace'in her köşesinde kameranın zemin görmesi.
- Legacy rollout parmak izinin ve legacy kamera karelerinin değişmemesi.
- Düzlemin çarpışmada sonsuz kaldığının testi (fizik regresyonu olmadığı).

**Durum:** Kök neden bulundu, düzeltme uygulandı ve doğrulandı. Davranışsal
sonuç koşuyor.

## REV-031 — `single_blocker` engelinin konumunun sabit olması

**Tarih:** 12 Eylül 2026  
**Kaynak:** Kullanıcının sorusu: *"D1'de waypoint hep aynı yerde mi kalıyor?
Böyle olunca ajan öğrenmek yerine ezberlemez mi?"*  
**Etkilediği:** ADR-018 engel ekseni, ADR-008 sahne üretimi (opt-in korunur)

### Ölçüm 1: hedef tarafında ezberleme riski yok

Soru önce hedef için test edildi ve orada risk **bulunmadı**:

```text
episode başına farklı scene_seed        : 1760/1760 (hiç tekrar yok)
hedef x aralığı                        : -8.85 .. +8.99 m
hedef y aralığı                        : -8.93 .. +8.92 m
hedef std                              : x 5.14 m, y 5.15 m (düzgün ~5.77)
20x20 m'yi 4x4 hücreye böl -> dolu     : 16/16
hedef hatası dağılımı                  : düzgün disk, izotropik
                                         ortalama yarıçap 1.327 m (teori 1.333)
                                         8 yön dilimi 612-641 arası
eğitim / evaluation seed kesişimi      : 0
```

Hedef konumu ve hedef belirsizliği her episode'da yeniden çekilir, evaluation
seed'leri eğitimde hiç görülmemiştir.

### Ölçüm 2: engel tarafında risk gerçek

`_blocking_obstacle` engeli **her zaman** rotanın tam ortasına ve tam çizgi
üzerine koyar; yalnız boyutu rastgeledir. 300 sahnede ölçüldü:

```text
rotadaki konumu     : %49.2 - %50.0
rotadan yanal sapma : 0.000000 m
yatay yarı-boyut    : 0.15-0.45 m
yükseklik yarı-boyut: 0.60-1.20 m
```

Politika "yolun yarısında yana kaç" kuralını kamerayı kullanmadan
öğrenebilir. Engelden kaçınma bu kurulumda amaçlandığı kadar sınanmıyor.

### Ezberleme kanıtlanamadı

Engel kaydırılarak sonda yapıldı (kanıt koşusunun `final` checkpoint'i,
20 episode, düzeltilmiş zemin). Kesir başlangıçtan ölçülür:

| Engelin konumu | Çarpışma |
|---|---:|
| `%30` | `%20` |
| `%50` (eğitimdeki yer) | `%40` |
| `%70` | `%55` |

Desen "yarıyı ezberledi"den çok **tepki mesafesi**ne uyuyor: engel başlangıca
yaklaştıkça kaçmak için yol azalıyor. Ayrıca sonda, politikanın eğitildiği
zeminden farklı bir zeminde koştuğu için karışıktır (REV-030). Yani risk
yapısaldır ve gerçektir, fakat **ölçülmüş bir zarar değildir**.

### Yapılan: opt-in rastgeleleştirme, varsayılan kapalı

`SceneGeneratorConfig` iki alan kazandı:

```text
blocker_path_fraction_min = 0.5    (varsayılan; LEGACY_BLOCKER_PATH_FRACTION)
blocker_path_fraction_max = 0.5    (varsayılan)
```

Varsayılan, bugüne kadarki bütün sahnelerin bit-bit aynısını üretir. Kritik
ayrıntı: **rastgeleleştirme kapalıyken `rng`'den hiç çekilmez**, yoksa akış
ilerler ve legacy sahneler değişirdi. Doğrulandı:

```text
LEGACY_ROLLOUT_SHA256 = 03f16232...4e3f5e23        değişmedi
kapalı: engel kesri 0.500-0.500, std 0.000
açık  : engel kesri 0.151-0.493, std 0.096  (başlangıçtan ölçülerek)
```

### Kaydırma tek yönlüdür ve bu geometrik bir zorunluluk

Kod kesri **hedeften** ölçülür (`0.5` orta nokta, `1.0` başlangıç). Alt sınır
`0.5`'in altına inemez: hedefe daha yakın bir engel ADR-016 hover zarfını
ihlal eder ve terminal görevi çözülemez kılar. Üst sınır
`BLOCKER_MAXIMUM_PATH_FRACTION = 0.85`'tir.

Yani engel yalnız **orta noktadan başlangıca doğru** kaydırılabilir;
başlangıçtan ölçüldüğünde rotanın `%15` ile `%50`'si arası. Bu, sabit `%50`'ye
göre gerçek bir çeşitlilik sağlar fakat tam bir rastgeleleştirme değildir.
Hedefin çevresine engel koymanın doğru yolu `random_obstacles` modudur ve o
ADR-018 merdiveninin sonraki adımıdır.

Yanal offset eklenmedi: `single_blocker` sözleşmesi doğrudan rotanın kapanmasını
gerektirir (`direct_path_is_blocked`), bu yüzden hâlâ kapatan bir yanal offset
zorunlu olarak engelin yarı-boyutu mertebesinde kalır ve kayda değer çeşitlilik
üretmez.

### Sıralama kararı

Rastgeleleştirme **açılmadı.** REV-030 düzeltmesiyle birlikte açılırsa iki
değişken aynı koşuda karışır; bu hata REV-026'da yapıldı ve iki koşu harcadı.
Sıra: düzeltilmiş zeminde temiz D1 taban ölçümü
(`runs/d1-rev030-seed20260912`), sonra rastgeleleştirmeyi ayrı koşuda ölçmek.
Kullanıcı bu sıralamayı onayladı.

### Doğrulama kapısı

- Varsayılanın engeli tam orta noktada tutması (`atol 1e-9`).
- Legacy rollout parmak izinin değişmemesi (rng akışı korunuyor).
- Açıkken kesir std'sinin `> 0.05` olması ve `[0.15, 0.50]` bandında kalması.
- Açıkken doğrudan rotanın hâlâ kapanması.
- Aralık dışı ve NaN değerlerin `validate()` tarafından reddi.

### Ölçülen bedel: `%0` çarpışma tek engelin eseri (13 Eylül 2026)

Yedinci koşunun `best` politikası (`single_blocker` ile eğitildi, `%0` çarpışma)
aynı workspace'te `random_obstacles` (40 engel) ile sınandı. 20 episode,
deterministik, seed `520001`:

| | 1 engel (eğitildiği) | 40 engel (görülmemiş) |
|---|---:|---:|
| Başarı | `%30.0` | `%25.0` |
| **Çarpışma** | **`%0.0`** | **`%30.0`** |
| Timeout | `%67.5` | `%45.0` |

**`%0` çarpışma neredeyse tamamen tek engelin eseridir.** D1 kapısının çarpışma
koşulu (`<= %5`) bu kurulumda geçildi fakat bu, engelden kaçınmanın
öğrenildiği anlamına gelmez.

Başarının korunması (`%30 -> %25`) ayrıca bilgilendirici: hedefe kamerayla
inme davranışı genelleşiyor, engelden kaçınma genelleşmiyor. Bu, engel
algısının ölçülmüş zayıflığıyla tutarlıdır (`~2 m` derinlik hatası,
`risk_specificity 0.56`, dikey çözünürlük yok).

**Durum:** Ölçüldü ve uygulandı (varsayılan kapalı). Açılması D1 taban
ölçümünden sonra. Yukarıdaki sonda, sıradaki işin engel yoğunluğu değil
**engel algısı** olduğunu gösteriyor: 40 engelli ortamda kaçınma çöküyor ve
politikaya giden engel bilgisi 6 sayıdan ibarettir.

## REV-032 — Action uzayının terminal görevle yapısal çelişkisi

**Tarih:** 12 Eylül 2026  
**Kaynak:** Beşinci D1 koşusunun (düzeltilmiş zemin) ölçümleri ve kullanıcının
"gerçeğe uygun olsun" yönlendirmesi  
**Etkilediği:** ADR-015 action uzayı -> ADR-020 ile çözüldü

### Bulgu

Araç hedefin `0.65 m` içine **giriyor** fakat orada **kalamıyor**:

```text
hedefin 0.65 m içine giren episode          : %65
başarı                                      : %15
terminal adımlarda güvenli bölge dışı ihlali: %85.6
yatay mesafe medyanı                        : 1.038 m
```

Dört farklı reward tasarımı ve zemin düzeltmesi boyunca bu medyan hiç
`0.8 m`'nin altına inmedi. Yani bağlayıcı kısıt reward veya algılama değil.

### Geometrik sebep

```text
hold koşulu           : merkeze <= 0.650 m, hız <= 0.25 m/s, kesintisiz 3 s
hedef görünür kalması : merkeze >= 0.407 m (nadir kör bölgesi)
uygulanabilir halka   : 0.243 m
action uzayı (v1)     : yalnız ileri; yanal ve geri YOK
yatay FOV             : 75 derece -> +-37.5 derece
```

Araç yalnız baktığı yöne ilerleyebildiği için yanal düzeltme **yaw çevirmeyi**
gerektiriyor. `38 dereceden` fazla çevirince hedef kadraj dışına çıkıyor ve
hold koşulu bozuluyor. İnce düzeltmenin çıkışı yoktu.

### Kök neden: donanım kısıtının araç dinamiğine gömülmesi

ADR-015 tek ileri kameranın kör alanını gerekçe göstererek yanal ve geri hızı
kapatmıştı. Gerekçe geçerlidir fakat bir **sensör** kısıtını **aktüatör**
kısıtına çevirir. Gerçek bir quadrotor yatay düzlemde izotropiktir; legacy
ortam (ADR-008) bunu dört boyutlu action ile zaten yapıyordu.

### Çözüm: ADR-020

Action `vision-action-v2` oldu: `[ileri/geri, yanal, dikey, yaw-rate]`, hepsi
işaretli, yatay hız **büyüklüğü** `1.5 m/s` ile sınırlı (yöne göre değişmez).
Hız zarfı büyütülmedi, yalnız yön kısıtı kaldırıldı.

Kör alan kısıtı koddan kaldırılıp `collision_penalty`'ye bırakıldı. Ölçülen
risk: `random_obstacles` sahnelerinde görüş konisi dışında `2 m`'den yakın
engel sayısı, görünür alandakinin `3` katı (606 vs 193). Gerçek sistemler bunu
gittiği yöne dönerek çözer; bu bir donanım kısıtı değil öğrenilecek
davranıştır ve çarpışma oranıyla ölçülür.

**Observation sözleşmesi değişmedi.** Encoder action uzayına bağlı değildir
(doğrulandı), bu yüzden onaylı zincir geçerliliğini korur ve dataset
yenilemesi gerekmez. Yalnız politika yeniden eğitilir.

```text
action_space.shape                 : (3,) -> (4,)
VISION_OBSERVATION_VERSION         : vision-v1 (değişmedi)
VISION_ACTION_VERSION              : vision-action-v2 (yeni)
LEGACY_ROLLOUT_SHA256              : 03f16232...4e3f5e23 (değişmedi)
pytest                             : 262 passed, 83 subtests
```

### Doğrulama kapısı

- Dört temel yönün beklenen dünya hızını üretmesi.
- Çapraz komutta yatay hız büyüklüğünün sınırı aşmaması (bileşen bazlı kırpma
  `sqrt(2)` kat hızlı uçmaya izin verirdi; legacy bunu yapar, kopyalanmadı).
- Yanlış shape ve NaN reddi.
- Legacy parmak izinin değişmemesi.
- Davranışsal: terminal adımlarda yatay mesafe medyanı `0.869 m`'nin altına
  inmeli. Çarpışma oranı yükselebilir; bu deneyin sorusudur.

**Durum:** Çözüldü (ADR-020). Davranışsal doğrulama koşuyor:
`runs/d1-adr020-seed20260912`.

## REV-033 — Yaklaşma ödülünün bankaya yazılması: "git ve öl" stratejisi

**Tarih:** 12 Eylül 2026  
**Kaynak:** ADR-020 koşusunun ilk çeyreğinde `minimum_altitude` oranının
`%70`'e çıkması  
**Etkilediği:** ADR-017 reward sözleşmesi (13. bileşen)

### Bulgu

ADR-020 koşusunda irtifa ihlali baskın terminasyon oldu (`178/300`) ve
episode'lar kısaldı (`460 -> 182` adım) fakat **ödül yükseliyordu**. Terminasyon
başına ortalama episode ödülü ölçüldü:

| Terminasyon | n | Ortalama ödül |
|---|---:|---:|
| `minimum_altitude` (yere çakılma) | `178` | **`+7.31`** |
| `obstacle_collision` | `74` | `-9.99` |
| `time_limit` (bütçeyi doldurmak) | `11` | **`-36.81`** |

**Yere çakılmak `+7`, hayatta kalmak `-37`.** Politika ölmeyi öğrenmişti ve bu
rasyoneldi.

Ödül ayrıştırması nedeni gösterdi (checkpoint `100k`, 8 episode):

```text
target_progress   +43.59      <- hedefe yaklaşma bir kez toplanıyor
settle_speed       -9.53
abort              -7.50
time               -1.82
hover_hold          0.00
success             0.00
return            +19.07  /  182 adım
```

### Kök neden

`target_progress` bir kez toplanıp **bankada kalıyor** ve başarıyla
gerekçelendirilmesi gerekmiyordu. Hedefe uçmak `~+43` kazandırıyor; ardından
yere çakılmak `-10` ödüyor. Net `+33`. Hold denemesini sürdürmek ise adım
başına `settle_speed` ve `time` ödemek demek, yani **hayatta kalmak pahalı,
ölmek ucuzdu.**

### Bu defekt ADR-020 tarafından yaratılmadı

Beşinci koşu (ileri-yalnız action) aynı ölçümle kontrol edildi:

| Koşu | Baskın strateji | n | Ödül |
|---|---|---:|---:|
| 5 (ileri-yalnız) | hedefe git, bütçeyi doldur (`time_limit`) | `217/300` | `+21.28` |
| 6 (holonomik) | hedefe git, yere çak (`minimum_altitude`) | `178/300` | `+7.31` |

İkisinde de politika **başarısız** bir stratejiye yerleşip bankadaki
progress'i koruyor. Holonomik action yalnız daha hızlı bir bankalama yolu
açtı (`460 -> 182` adım). Kök neden ikisinde de aynı ve ADR-020'den
önce de vardı; ADR-020 korunur, çünkü fiziksel olarak doğru olan odur.

### Düzeltme: `progress_refund` bileşeni

Fiziksel başarısızlıkla biten episode, toplanan bütün `target_progress`'i geri
verir:

```text
progress_refund = -(banked_progress + o adımın progress'i)   fiziksel başarısızlıkta
                = 0                                           diğer hâllerde
```

Fiziksel başarısızlık sebepleri (`PHYSICAL_FAILURE_REASONS`):
`obstacle_collision`, `ground_collision`, `out_of_bounds`, `minimum_altitude`,
`maximum_tilt`.

**`time_limit` ve hover abort'u listede DEĞİLDİR.** Tamamını geri vermek erken
öğrenmeyi yok ederdi: hiç başarı yokken episode getirisi, progress
şekillendirmesi hiç verilmemiş gibi olur ve hedefe yönelme gradyanı kaybolur.
Düzeltme yalnız ölçülmüş patolojiyi hedefler.

Yeni teşvik yapısı:

```text
başarı            : progress + 30 + hold          -> en iyi
timeout           : progress - zaman              -> güvenli fakat yetersiz
fiziksel başarısızlık: 0 - ceza                   -> her zaman zararlı
```

### Doğrulama

Koşu başında ölçülen ilk 28 episode:

```text
obstacle_collision : -43.54
out_of_bounds      : -31.34
time_limit         : -38.41   (eğitilmemiş politika, progress toplayamıyor)
```

Fiziksel başarısızlıklar artık net negatif.

### Doğrulama kapısı

- Fiziksel başarısızlıkta `target_progress + progress_refund == -banked`.
- `time_limit`, hover abort ve başarıda `progress_refund == 0` ve
  `target_progress > 0`.
- Uçtan uca: aracı hedefe doğru sürüp irtifa ihlaliyle bitiren episode'un
  **toplam** ödülü negatif olmalı ve `progress + refund` sıfırlanmalı.
- `banked_progress` için sonsuz değer reddi.
- 13 bileşenin ayrı loglanması ve `total == sum`.

**Durum:** Çözüldü. Davranışsal doğrulama koşuyor:
`runs/d1-rev033-seed20260912`.

## REV-034 — D1 koşularının ADR-018 spesifikasyonunu karşılamaması

**Tarih:** 13 Eylül 2026  
**Kaynak:** Kullanıcının sorusu: *"D1 eğitimi engelli senaryo ile olması
gerekmiyor muydu?"*  
**Etkilediği:** ADR-018 D1 satırı, REV-023'ten REV-033'e kadar bütün D1 sonuçları

### Bulgu

ADR-018 merdiveni D1'i şöyle tanımlar:

```text
| D1 | 5-15 m | 20 x 20 x 3 m | 1.5 m/s | ~710 adım | 0.10/m^2 | ~40 engel | eğitim |
```

**Yedi D1 koşusunun tamamı `scene_mode = "single_blocker"` ile, yani TEK
engelle yapıldı.** `static_scene_d1.toml` içindeki `obstacle_count = 40`
doğrudur fakat bu alan yalnız `random_obstacles` modunda okunur.

Bu seçim hiçbir ADR'de, hiçbir günlük kaydında karar olarak geçmiyor. ADR-018
"engel zorluğu bağımsız eksen kalır (tek engel -> seyrek -> ...)" der ve config
muhtemelen bu cümleye dayanarak ilk basamağı almış, fakat merdiven tablosu
`~40` engel diyor. İki ifade çelişiyor ve çelişki çözülmemiş.

### Ölçülen bedel

Yedinci koşunun `best` politikası (tek engelle eğitildi) aynı workspace'te
`random_obstacles` ile sıfır-shot sınandı (20 episode, deterministik):

| | 1 engel (eğitildiği) | 40 engel (D1 spesifikasyonu) |
|---|---:|---:|
| Başarı | `%30.0` | `%25.0` |
| **Çarpışma** | **`%0.0`** | **`%30.0`** |
| Timeout | `%67.5` | `%45.0` |

**Bu, daha önce raporlanan "D1 kapısının çarpışma yarısı geçildi" ifadesini
geçersiz kılar.** `%0` çarpışma tek engelin eseriydi. Spesifikasyondaki
zorlukta ne başarı (`>= %85`) ne çarpışma (`<= %5`) koşulu karşılanıyor.

Başarının korunması (`%30 -> %25`) ayrıca bilgilendirici: hedefe kamerayla inme
davranışı genelleşiyor, engelden kaçınma genelleşmiyor.

### İkinci sapma: rota bandı (kullanıcı kararı)

Kullanıcı D1 ortamını görsel olarak inceledi ve `20 x 20 m` alanda `5-15 m`
rotanın anlamsız olduğunu belirtti: araç hedefin yanında başlıyor ve alanı hiç
geçmiyordu. Band **`16-22 m`**'ye çıkarıldı.

Örnekleme kutusu `18 x 18 m` olduğu için bu, aracın karşı kenardan başlaması
demektir. Episode bütçesi kırpılmıyor (`22 m -> 43.7 s = 874 adım`, sınır
`60 s`); maliyet `600k` adımda episode sayısının `~1350`'den `~690`'a
düşmesidir.

Bu, ADR-018 D1 satırından bilinçli bir sapmadır ve kullanıcı kararıdır.

### Yapılan

```text
ppo_vision_d1.toml   : scene_mode "single_blocker" -> "random_obstacles"
static_scene_d1.toml : endpoint separation 5-15 m -> 16-22 m
```

Yeni ortam 60 sahnede doğrulandı:

```text
engel sayısı            : 40 (0.10/m^2)
rota uzunluğu           : 16.06 - 21.43 m (ort 17.72)
hedefe en yakın engel   : medyan 2.22 m, en yakın 0.69 m
terminal bölgede engel  : ortalama 1.0, en fazla 5
doğrudan rota kapalı    : 60/60
çözülebilir yol var     : 60/60
hedef hover zarfı boş   : 60/60   (REV-016 kısıtı çalışıyor)
```

Başlangıç, hedef ve 40 engelin tamamı her episode yeniden örneklenir; 80 sahne
üzerinde en yakın iki başlangıç arası `0.076 m`, en yakın iki hedef arası
`0.040 m`, yani hiçbiri tekrar etmiyor.

### Sonuç

REV-023'ten REV-033'e kadar ölçülen bütün D1 sayıları **spesifikasyonun
altındaki bir zorlukta** alınmıştır. O ölçümler kendi içinde geçerlidir
(reward defektleri, action uzayı çelişkisi, zemin düzlemi hatası hepsi
gerçekti ve düzeltildi) fakat **D1 kapısına dair hiçbir iddia taşımazlar.**

`runs/d1-spec-seed20260912` projenin ilk spesifikasyona uygun D1 koşusudur.

**Durum:** Sapma kayda geçirildi ve kapatıldı. Taban ölçümü koşuyor.

## REV-035 — Engellerin üstünden uçulabilmesi: kaçınma hiç öğrenilmemişti

**Tarih:** 13 Eylül 2026  
**Kaynak:** Kullanıcının spec-D1 demosunu izlemesi: *"BU DRONE ENGELLERİ
AŞMIYOR Kİ? ENGELLERİN ÜSTÜNDEN UÇUYOR. 0 KAÇINMA MANEVRASI."*  
**Etkilediği:** ADR-018 engel ekseni, `static_scene_d1.toml`, bütün engelli
koşuların yorumu

### Bulgu

```text
engel yükseklikleri : 0.80 - 2.40 m
workspace tavanı    : 3.00 m
üstteki boşluk      : 0.60 m
ölçülen seyir irtifası: 2.36 - 2.51 m   <- tam o boşluk
```

Politika engellerin tepesiyle tavan arasındaki `0.60 m`'lik koridordan uçmayı
öğrendi. **Hiçbir yatay kaçınma manevrası yapmıyordu.**

Bu, daha önce "kaçınma öğrenildi" diye raporlanan eğitim trendini
(`çarpışma %45 -> %13`) yeniden yorumlar: öğrenilen şey kaçınma değil
**tırmanmaydı**.

Aynı bulgu, "hedefin üzerine geliyor fakat hold bandına inmiyor" gözlemini de
açıklar (REV-034 sonrası ölçüm: terminal irtifa medyanı `2.21 m`, hold bandı
`1.31-1.71 m`, banda giren adım `%0`). Araç inmiyordu çünkü **hiç inmemişti**;
bütün episode'u engellerin üstündeki koridorda geçiriyordu.

### Kendi kodumuzla çelişki

`has_navigable_xy_path` sahneyi **2 boyutlu** occupancy grid üzerinde doğrular,
yani üretici "bu sahnenin etrafından dolaşılabilir bir çözümü var" garanti
eder ve her sahnede bunu sınar (60/60 geçiyordu). Üstünden uçmak bu
doğrulamayı tamamen anlamsız kılıyordu: fizik, üreticinin garanti ettiği
problemi çözmeyi hiç gerektirmiyordu.

Yani bu bir zorluk ayarı değil, **fizik ile sahne doğrulaması arasındaki
tutarsızlığın** giderilmesidir.

### Düzeltme

```text
obstacle_half_size_min_m = [0.15, 0.15, 1.50]    (önce z: 0.40)
obstacle_half_size_max_m = [0.45, 0.45, 1.50]    (önce z: 1.20)
-> engel yüksekliği 3.00 m = workspace tavanı, üstteki boşluk 0.00 m
```

Yatay boyutlar değişmedi (`0.30-0.90 m` genişlik). Üstünden geçmek fiziksel
olarak imkânsız; `out_of_bounds` tavanı `3.00 m`'dir.

### Doğrulama (60 sahne)

```text
çözülebilir 2B yol var  : 60/60
hedef hover zarfı boş   : 60/60
doğrudan rota kapalı    : 60/60
hedefe en yakın engel   : medyan 2.28 m, en yakın 1.20 m  (önce 2.22 / 0.69)
terminal bölgede engel  : ortalama 0.9  (önce 1.0)
```

Son iki satır beklenmeyen bir kazanç: tavana kadar uzanan engeller hedefin
hover zarfını boşaltmak zorunda olduğu için hedeften **daha uzağa** itiliyorlar
ve iniş için daha fazla yer kalıyor.

Geçit genişlikleri (30 sahne, `3 m` altındaki komşu engel çiftleri, n=2523):

```text
açıklık medyanı          : 1.92 m
%10 dilim                : 0.67 m
vehicle_clearance        : 0.25 m
0.50 m'den dar çift oranı: %6.3
```

Çoğu geçit rahat, `%6`'sı dar. Dar geçit ADR-018 merdiveninin sonraki
basamağıdır; bu kurulumda kendiliğinden az miktarda mevcut.

### Sonuçlar

- **Engelli koşuların hiçbirinde yatay kaçınma öğrenilmemiştir.** REV-034'ün
  "navigasyon öğrenildi" değerlendirmesi bu ışıkta geçersizdir.
- Bu değişiklik encoder'ı daha da dağılım dışına atar: encoder `6 x 5 m`
  odada, `0.80-2.40 m` engellerle eğitildi; kamera artık tavana kadar uzanan
  duvarlar görecek. Kaçınma için doğru ortam budur fakat engel algısının
  zayıflığı (`~2 m` derinlik hatası, dikey çözünürlük yok) daha çok ısıracak.
- Yükseklik çeşitliliği kaldırıldı (hepsi `3.00 m`). Karışık yükseklik
  (bazıları drone'dan kısa) daha gerçekçi olurdu fakat o, üstünden uçma
  kaçışını kısmen geri açar; önce tam kapalı hâlde taban ölçümü alınır.

**Durum:** Düzeltildi. Taban ölçümü koşuyor:
`runs/d1-walls-seed20260912`.

## REV-036 — Kameranın ileri görüş mesafesinin irtifaya bağlı olması

**Tarih:** 13 Eylül 2026  
**Kaynak:** REV-035 düzeltmesinden sonra politikanın `0.81 m` irtifaya inip
çarpışma oranının `%85-93`'te sıkışması  
**Etkilediği:** ADR-016 kamera geometrisi, ADR-017 reward (14. bileşen)

### Bulgu

```text
kamera pitch aşağı : 45.0 derece
dikey yarım FOV    : 29.9 derece
en üst ışın        : yatayın 15.1 derece altında
-> ileri görüş mesafesi = 3.711 x irtifa
```

Düz zeminde **bütün** kamera ışınları zemine çarptığı için irtifa ileri görüşü
doğrudan sınırlar:

| İrtifa | Görebildiği en uzak | `1.5 m/s` ile tepki süresi |
|---:|---:|---:|
| `0.81 m` | `3.0 m` | `2.0 s` |
| `1.50 m` | `5.6 m` | `3.7 s` |
| `2.16 m` | `8.0 m` | `5.3 s` |
| `2.50 m` | `9.3 m` | `6.2 s` |

REV-035'ten önce politika `2.36-2.51 m`'de uçuyordu, yani `9 m` görüyordu **ve**
engellerin üstünden geçebiliyordu. Üstünden geçme kapatıldığında alçalmasını
engelleyen hiçbir şey kalmadı (`target_progress` hover anchor'a **3B** mesafeye
bakar, yani alçalmak ödüllüdür) ve `0.81 m`'ye indi. Orada `free_space`
hatası (`~2 m`) görüş mesafesiyle (`3.0 m`) aynı mertebeye geldi.

Ölçüm hatası da vardı: `free_space` etiketi `8.0 m` ile normalize edilir fakat
kamera `0.81 m`'de `3.0 m` görür. "8 m boş" ile "sadece 3 m görebiliyorum"
ayırt edilemez; ikisi de `1.0`'dır.

### Düzeltme: `cruise_altitude` bileşeni

```text
cruise_altitude = -w * max(0, cruise_altitude_m - irtifa)   seyir durumlarında
                = 0                                          terminal durumlarda
CRUISE_STATES = {transit, search}
```

Gereken irtifa türetildi, tahmin edilmedi:

```text
cruise_altitude_m = free_space_max_depth_m / 3.711 = 8.0 / 3.711 = 2.156 m
```

Bu irtifada zemin izi `0.58 - 8.00 m`, yani **sensörün gerçek menzili etiketin
normalize edildiği derinlikle örtüşür**.

Ağırlık `0.05`: `1 m` eksik irtifa, `300` adımlık bir seyirde `-15.0` maliyet;
tek çarpışmanın (`-25.0`) altında fakat anlamlı.

`cruise_altitude_m > hover irtifası` invariantı config yüklemesinde zorlanır;
aksi halde seyir cezası terminal inişle çelişirdi.

### Ölçülen sonuç: kendi ekseninde çalıştı, problemi çözmedi

| | Önce | Sonra |
|---|---:|---:|
| Uçuş irtifası medyanı | `0.81 m` | **`1.76 m`** |
| Kameranın gördüğü mesafe | `3.0 m` | **`6.5 m`** |
| Çarpışma oranı | `%85-93` | `%80-87` |

Görüş mesafesi ikiye katlandı, çarpışma düşmedi. **Yani görüş mesafesi bağlayıcı
kısıt değildi.** Bulgu geçerlidir ve düzeltme korunur (alçak uçmak gerçekten
kör eder), fakat asıl kısıt REV-037'de bulundu.

**Durum:** Çözüldü. Bağlayıcı kısıt değildi.

## REV-037 — Encoder'ın D1 ortamında engelleri görememesi

**Tarih:** 13 Eylül 2026  
**Kaynak:** REV-036 düzeltmesine rağmen çarpışmanın `%84`'te yatay kalması  
**Etkilediği:** ADR-015 observation sözleşmesi (`vision-v1` -> `vision-v2`),
ADR-009 approval zinciri

### Ortam ve ödül suçlu değil

Önce ortamın çözülebilirliği ölçüldü — "yol var mı" değil, "ne kadar geniş":

```text
en geniş koridorun darboğaz yarı-genişliği : medyan 1.00 m (min 0.45)
başlangıçtaki açıklık                      : medyan 1.31 m
hedefteki açıklık                          : medyan 1.30 m
araç açıklık payı                          : 0.25 m
darboğazı 0.50 m'den dar olan sahne        : %4
```

Koridorlar `~2 m` genişliğinde, araç `0.25 m` paya ihtiyaç duyuyor. Ortam
rahatça geçilebilir.

### Ölçülen kök neden

Encoder'ın `free_space` tahmini D1 ortamında doğrudan ayrıcalıklı gerçekle
karşılaştırıldı (534 adım, 6 episode, aynı etiket kuralı):

```text
                sol    orta   sağ
gerçek  medyan  4.13   3.31   2.28  m
tahmin  medyan  0.74   1.07   1.66  m
mutlak hata     3.63   3.27   2.30  m   -> genel 3.07 m
hatanın %90 dilimi                6.98  m
```

**`3.07 m` ortalama hata ve sistematik olarak "daha dolu" yönünde.** Koridor
genişliği `~2 m`, ölçüm hatası `~3 m`: sinyal gürültünün altında. Politika
`600k` adımda öğrenemez çünkü öğrenilecek sinyal yok.

Nedeni: onaylı encoder legacy `6 x 5 m` odada, `0.80-2.40 m` engellerle ve
**512 kare** üzerinde eğitilmişti. D1 `20 x 20 m`, engeller tavana kadar.

### İkinci kök neden: temsil çözünürlüğü

Politikaya engel hakkında giden bilgi **6 sayıydı**: `free_space` L/C/R ve
`collision_risk` L/C/R. Görüntü üç dikey dilime bölünüyordu, dikey çözünürlük
yoktu.

Oysa encoder **zaten** `9 x 12 = 108` hücrelik bir engel ısı haritası üretiyor
ve bu, `obstacle_mask`'e karşı BCE ile eğitiliyordu. Politikaya verilmiyordu.

Kamera `45 derece` aşağı baktığı için bu haritanın **satırları zemin
mesafesine**, **kolonları yön açısına** karşılık gelir; yani yerel bir
mesafe x yön doluluk ızgarasıdır. Koridorun *nerede* olduğunu üç dilimle
anlatmak mümkün değildi, bununla mümkündür.

Bu çıktı bugüne kadar **hiç ölçülmemişti**: kalite raporunda heatmap metriği
yoktu.

### Yapılan

**1. Dataset D1 ortamından yeniden üretildi.**

```text
kare sayısı  : 736 -> 6528        (384/48/48/64 sahne x 12 poz)
sahne config : D1 (20 x 20 m, tavana kadar engeller, 0.10/m^2)
zemin        : 21.13 m yarı-boyut (REV-030 dataset tarafında da geçiriliyor)
irtifa bandı : 0.80-2.10 -> 1.20-2.90 m
               eski üst sınır seyir irtifasının (2.156 m) ALTINDAYDI, yani
               encoder politikanın uçtuğu irtifayı hiç görmemişti
yanal offset : 1.25 -> 2.50 m
split_salt   : ebabil-vision-d1-v5
sızıntı kontrolü: geçti
```

**2. Engel haritası kalite kapısına alındı.** `obstacle_map_iou`,
`obstacle_map_precision`, `obstacle_map_recall` metrikleri eklendi; eşikler
`0.30 / 0.45 / 0.55`. Ölçülmemiş bir kanalı actor observation'ına koymak
approval zincirinin amacına aykırı olurdu.

**3. `obstacle_map` kanalı politikaya verildi** (`vision-v2`, `(108,)`,
`[0, 1]`).

### Ölçülen kazanç

| | Onaylı v9 (legacy) | **v11 (D1 dataset)** |
|---|---:|---:|
| `free_space` hatası (`development_ood`) | `2.06 m` | **`1.05 m`** |
| Aynı ölçüm D1 ortamında (rollout) | `3.07 m` | ölçülecek |
| `risk_precision` | `0.709` | `0.849` |
| `target_recall` | `0.929` | `0.921` |
| **engel haritası IoU** | **hiç ölçülmemiş** | **`0.651`** |

Engel haritası, 3 seed ortalaması: `development_ood` IoU `0.651`
(en kötü seed `0.638`), precision `0.782`, recall `0.796`. Audit split'inde
(hiç görülmemiş `ellipsoid` geometrisi) IoU `0.614`, recall `0.860`.

### Sızıntı güvencesi değiştirildi

`obstacle_map` ayrıcalıklı **değildir** (encoder'ın RGB'den tahmini, `free_space`
ile aynı statüde), fakat isim tabanlı yasak listesi `"obstacle"` token'ını
reddediyordu. Liste `obstacle_mask` ve `segmentation`'a daraltıldı ve yerine
daha güçlü bir test kondu: kanalın encoder ısı haritasıyla **birebir aynı**
olduğu ve actor kamerasının ayrıcalıklı çıktı üretmediği doğrulanır.

### Yeni onaylı zincir

```text
vision_dataset_d1_20260913 -> vision_quality_v22_20260913 (3 seed, passed)
                           -> vision_audit_v4_20260913 (passed)
                           -> vision-encoder-v11
```

### Ölçülen sonuç: algı düzeldi, politika yine öğrenmedi

`runs/d1-v11-seed20260912`, `600k`, yeni onaylı zincir. 40 episode,
deterministik:

| | `final` | `best` |
|---|---:|---:|
| Başarı | `%0.0` | `%0.0` |
| Çarpışma | `%47.5` | `%75.0` |
| Abort | `%15.0` | `%20.0` |

Algı gerçekten düzeldi (aynı rollout ölçümü, `free_space`):

```text
v9  : ortalama hata 3.07 m, %90 dilim 6.98 m, sistematik "daha dolu"
v11 : ortalama hata 1.26 m, %90 dilim 3.49 m
```

Fakat çarpışma düşmedi. **Yani algı da tek başına bağlayıcı kısıt değildi.**
Kalan iki ölçüm nedeni gösteriyor (REV-039).

**Durum:** Algı tarafı çözüldü ve kalıcıdır (`3.07 -> 1.26 m`). Politika
başarısızlığının nedeni REV-039'dur.

## REV-038 — `collision_risk` kanalının gereksizliği

**Tarih:** 13 Eylül 2026  
**Kaynak:** v11 kalite koşusunda tek takılan kapının risk head'i olması  
**Etkilediği:** ADR-015 observation sözleşmesi, ADR-009 approval kapısı

### Bulgu

İlk v11 kalite koşusu `quality_passed = False` verdi ve **tek** sebep
`collision_risk` head'iydi, yalnız `validation` split'inde:

```text
risk_recall    : validation 0.681 (kapı 0.70) | development_ood 0.722 GEÇTİ
risk_precision : validation 0.694 ort, bir seed 0.553 (kapı 0.65)
```

Diğer bütün kapılar geçti, engel haritası dahil.

### Neden takıldı

`collision_risk` etiketi **tanım gereği** `free_space < collision_risk_threshold`
(`0.25`, yani `2.0 m`). Yani politikanın zaten sahip olduğu kanalın eşiklenmiş
kopyası, daha düşük bilgi içerikli.

D1 dataset'inde `free_space` dağılımının standart sapması `0.42`'den `0.355`'e
indi (statistics.json), yani daha fazla örnek eşiğin **yakınına** düştü ve
evet/hayır kararı kıl payı yanlış çıkmaya başladı. Regresyonun kendisi
**iyileşti** (`2.06 -> 1.05 m`).

### Karar

Kanal actor observation'ından çıkarıldı (`vision_aux` `9 -> 6`) ve approval
kapısı olmaktan çıktı (`gate_collision_risk = false`, varsayılan `true` olduğu
için `v9`/`v10` zincirinin anlamı korunur). Metrikler raporlanmaya devam eder.

Gerekçe sırası önemlidir: kanal **zaten gereksizdi** ve yerine `obstacle_map`
(108 hücre) geldi; kapıdan kalması bunu fark ettirdi. Eşiği düşürmek
(kapı avcılığı, REV-021'de reddedilen yol) ile kanalı sözleşmeden çıkarmak
farklı şeylerdir: birincisi ölçütü esnetir, ikincisi ölçülecek şeyi değiştirir.
Karar kullanıcıya sunuldu ve onaylandı.

### Sonuç

```text
vision_quality_v22_20260913 : quality_passed = True
vision_audit_v4_20260913    : audit_passed = True
vision_aux                  : 9 -> 6  [target_score, center_x, center_y, free_L/C/R]
```

**Durum:** Çözüldü.

## REV-039 — Engel zorluk merdiveninin atlanması

**Tarih:** 14 Eylül 2026  
**Kaynak:** `vision-encoder-v11` ile yapılan D1 koşusunun da `%0` başarı
vermesi  
**Etkilediği:** ADR-018 engel ekseni, D1 eğitim protokolü

### Bulgu

Dört ayrı kök neden bulunup düzeltildikten sonra (REV-030 zemin, REV-033 ödül
bankalaması, ADR-020 action uzayı, REV-035 üstünden uçma, REV-036 seyir
irtifası, REV-037 algı) D1 hâlâ `%0` başarı veriyor ve çarpışma `%47-75`.

Kalan neden ölçüldü ve ortam ya da algı değil: **zorluk atlaması.**

ADR-018 engel zorluğunu açıkça kademeli bir eksen olarak tanımlar:

> tek engel -> seyrek -> geçit -> koridor -> dar geçit -> yoğun karma

Bütün D1 koşuları **birinci** basamakta (tek engel) yapıldı; REV-034
düzeltmesiyle doğrudan `40` engele, tavana kadar yükseklikle geçildi. Aradaki
basamaklar atlandı.

### Neden zorluk bu kadar arttı: ardışık karar sayısı

Koridor genişliği yoğunlukla neredeyse değişmiyor:

| Engel | Yoğunluk | Darboğaz yarı-gen. | Koridor | Koridor / algı hatası |
|---:|---:|---:|---:|---:|
| `5` | `0.013/m^2` | `1.35 m` | `2.70 m` | `2.1` |
| `10` | `0.025/m^2` | `1.25 m` | `2.50 m` | `2.0` |
| `20` | `0.050/m^2` | `1.20 m` | `2.40 m` | `1.9` |
| `40` | `0.100/m^2` | `1.05 m` | `2.10 m` | `1.7` |

Değişen şey **rota koridorundaki engel sayısı**, yani ardışık doğru karar
sayısı:

| Engel | Koridorda (`+-1.5 m`) | Karar başına `%20` hata ile hayatta kalma |
|---:|---:|---:|
| `1` | `1.0` | `%80` |
| `5` | `1.6` | `%71` |
| `10` | `2.0` | `%63` |
| `20` | `3.3` | `%48` |
| `40` | **`6.1`** | **`%26`** |

`40` engelde araç `16-22 m` boyunca ortalama `6.1` engelden kaçmak zorunda ve
bir hata episode'u bitiriyor.

### Asıl mekanizma: hiç başarı örneklenmiyor

`2067` episode'un **hiçbirinde** başarı yok. Politika `success_bonus`'u hiç
deneyimlemedi, yani terminal görev için öğreneceği pozitif sinyal **yok**;
yalnız `-25` çarpışma cezasından kaçınmayı öğrenmeye çalışıyor.

Tek engelli kurulumda `%30` başarı alınmıştı çünkü araç hedefe varabiliyordu.
`40` engelde `%65-75` oranında yolda ölüyor ve terminal davranışı hiç
öğrenemiyor.

Bu bir keşif (exploration) problemidir ve reward ya da algı ayarıyla
çözülmez.

### Önerilen düzeltme: yoğunluk merdiveni

ADR-018'in kendi eksenini takip etmek:

```text
D1a : 5 engel  (0.013/m^2)  koridorda ~1.6 -> hedefe varabilir, hold öğrenir
D1b : 10 engel (0.025/m^2)  koridorda ~2.0
D1c : 20 engel (0.050/m^2)  koridorda ~3.3
D1  : 40 engel (0.100/m^2)  koridorda ~6.1  ADR-018 spesifikasyonu
```

Her basamak öncekinin politikasından devam eder (warm start). Bu, observation
sözleşmesi sabit kaldığı için mümkündür.

Kabul ölçütü basamak başına: başarı bir önceki basamağın altına düşmemeli ve
çarpışma trendi azalan olmalı.

**Durum:** Ölçüldü, düzeltme önerildi, kullanıcı kararı bekler.

## REV-040 — Hedefin workspace kenarına terminal manevraya yer bırakmadan örneklenmesi

**Tarih:** 14 Eylül 2026  
**Kaynak:** REV-039 yoğunluk merdiveninin ilk basamağında (5 engel) çarpışmanın
`%38 -> %12`'ye inmesine rağmen başarının `0` kalması  
**Etkilediği:** `static_scene_d1*.toml` endpoint örneklemesi

### Bulgu

Merdivenin ilk basamağında kaçınma gerçekten öğrenildi (çarpışma `%38 -> %12`,
medyan çarpışma mesafesi hedefe `9.40 m`, yani yol ortasında) fakat başarı
`1083` episode'da `0` kaldı. Baskın terminasyon `hover_abort_target_lost`
(`%39`) oldu.

İlk hipotez yanlış çıktı. "Uzun rotada hedef `8 m`'den öteye görünmediği için
kaybediliyor" varsayımı ölçümle çürütüldü: abort'ların yalnız `%14`'ü `8 m`'den
uzakta oluyor. Terminasyon anındaki hedefe mesafe medyanları:

```text
hover_abort_target_lost : 2.33 m
out_of_bounds           : 2.29 m
time_limit              : 1.62 m
obstacle_collision      : 9.40 m
```

Yani araç hedef bölgesine **varıyor** ve orada başarısız oluyor.

### Kök neden

```text
endpoint örnekleme kutusu : +-9.0 m
workspace                 : +-10.0 m
hedefin kenara en kötü payı: 1.0 m
approach_radius_m         : 2.5 m   <- terminal manevranın ihtiyacı
```

100 sahnede ölçülen: hedefin kenara mesafesi medyan `1.91 m`, **`%56`'sı
`2 m`'den yakın**, `%34`'ü `1.5 m`'den yakın.

Yani episode'ların yarısından fazlasında terminal manevra workspace dışına
taşmak zorundaydı. `out_of_bounds` terminasyonlarının hedefe `2.29 m` mesafede
olması bunun doğrudan imzasıdır.

**Bu episode'lar kazanılamazdı.** Politika ne kadar iyi olsa da hedefin
üzerinde `3 s` duramazdı.

### Düzeltme

```text
random_endpoint_min/max : +-9.0 -> +-7.5 m    (kenar payı 2.5 m = approach_radius_m)
maximum_endpoint_separation_m : 22.0 -> 21.0 m
```

Rota bandının üst sınırı düşürülmek zorundaydı: `+-7.5 m` kutusunun köşegeni
`21.21 m`'dir, `22 m` isteyen üretici uzun rotaları bulamazdı.

Doğrulama (50 sahne, `d1a` ve `d1`):

```text
hedefin kenara mesafesi : medyan 2.97 m, min 2.50 m
2.5 m'den yakın hedef   : %0
rota uzunluğu           : 16.0 - 20.2 m
```

### Neden daha önce görünmedi

Bu defekt `single_blocker` dönemindeki bütün koşularda da vardı fakat maskeliydi:
o kurulumda başarı `%25-42` alınıyordu ve kenardaki hedefler basitçe
başarısız episode'lar olarak sayılıyordu. Rota bandı `5-15 m`'den `16-22 m`'ye
çıkarıldığında (REV-034, kullanıcı kararı) endpoint'ler kutunun **kenarlarına**
itildi ve oran `%56`'ya çıktı.

Yani iki doğru değişiklik (uzun rota + spesifikasyon yoğunluğu) birlikte üçüncü
bir defekti açığa çıkardı.

### Doğrulama kapısı

- Her sahnede hedefin workspace kenarına mesafesi `>= approach_radius_m`.
- Rota bandının üst sınırının endpoint kutusu köşegeninden küçük olması.
- `out_of_bounds` terminasyonlarının hedefe yakın kümelenmemesi (davranışsal).

**Durum:** Düzeltildi. Merdivenin ilk basamağı yeniden koşuyor:
`runs/ladder2-d1a-seed20260912`.

## REV-041 — Holonomik hareketin yönü hareketten ayırması

**Tarih:** 14 Eylül 2026  
**Kaynak:** REV-040 düzeltmesinden sonra da `hold` durumuna hiç girilmemesi  
**Etkilediği:** ADR-017 reward (yeni bileşen), ADR-020 ile ortaya çıkan yan etki

### Bulgu

`hover_abort_target_lost` baskın terminasyon (`%39`) ve `hold_seconds` **300
episode'da sıfır**. Yön hatası ölçüldü (kamera bakış yönü ile gerçek hedef
yönü arasındaki açı):

| Hedefe mesafe | Yön hatası | Hedef gerçekten kadrajda |
|---:|---:|---:|
| `0-1 m` | `101.9 derece` | `%92` |
| `1-2 m` | `123.5 derece` | `%26` |
| `2-3 m` | `118.7 derece` | `%14` |
| `5-10 m` | `78.1 derece` | `%8` |

Araç hedefin yakınında **hedefe sırtı dönük** uçuyor.

### Kök neden

ADR-020 holonomik hareketi açtı, yani yön hareketten **ayrıldı**: araç yana
kayarak hedefe gidebiliyor fakat başka yöne bakabiliyor. Önceki ileri-yalnız
action uzayında hedefe gitmek hedefe dönmek demekti; bu kısıt kalkınca
karşılığı konmadı.

`target_alignment` bu boşluğu kapatamaz: yalnız hedef **görünürken** merkez
kaymasını cezalandırır, kadraj dışındayken hangi yöne dönüleceğine dair
gradyan vermez.

### Düzeltme: `bearing_alignment`

```text
bearing_alignment = -w * min(|yön hatası|, pi) / pi     BEARING_STATES içinde
                  = 0                                    transit'te
w = 0.10  ->  120 derece hata = -0.067/adım
BEARING_STATES = {search, acquired, approach, settle, hold}
```

Hata **inanılan** hedeften türetilir (waypoint kanalı bu yönü zaten taşır),
yani ayrıcalıklı değildir.

`transit` bilinçli olarak kapsam dışıdır: seyirde kamera gittiği yöne bakmalı,
yoksa kör alana uçar (ölçülen: görüş konisi dışında `2 m`'den yakın engel
sayısı görünür alandakinin `3` katı).

### Ölçülen sonuç

| Mesafe | Yön hatası önce | sonra | Kadrajda önce | sonra |
|---:|---:|---:|---:|---:|
| `5-10 m` | `78.1d` | **`11.8d`** | `%8` | `%42` |
| `3-5 m` | `87.2d` | `68.7d` | `%0` | `%45` |
| `1-2 m` | `123.5d` | `106.4d` | `%26` | **`%56`** |

Uzak mesafede sorun çözüldü, yakın mesafede iyileşti fakat kalmadı.

**Durum:** Çözüldü, kalıcı. Tek başına yeterli değildi.

## REV-042 — Yavaşlama teşvikinin ulaşılamayan yerde olması

**Tarih:** 14 Eylül 2026  
**Kaynak:** REV-041'den sonra da `hold` durumuna girilmemesi  
**Etkilediği:** ADR-017 `settle_speed` bileşeni

### Bulgu

Hover koşulları adım adım sınandı (terminal fazda, 679 adım):

```text
hız                : %99.9 ihlal      <- baskın
açısal hız         : %99.3 ihlal
güvenli bölge dışı : %94.4 ihlal
irtifa toleransı   : %89.0 ihlal
tilt               : %74.4 ihlal
```

Araç terminal bölgeden **seyir hızıyla** geçiyor. Hold hız sınırı `0.25 m/s`,
seyir hızı `1.5 m/s`, yani `6` kat.

### Kök neden

`settle_speed` cezası yalnız `inside_safe_region` (merkeze `<= 0.65 m`) iken
açıktı. Araç oraya terminal adımların **`%2`**'sinde girebiliyordu.

Yani **yavaşlama teşviki, ancak yavaşlayarak ulaşılabilen bir yerde
bulunuyordu.** Bölgeye hızlı gelen araç içeri giremiyor, giremediği için
yavaşlama sinyalini hiç almıyordu.

### Düzeltme: mesafeye ölçekli hız payı

Ceza bütün terminal fazda (`BEARING_STATES`) uygulanır ve mesafeyle ölçeklenen
bir **payı aşan** kısmı cezalandırır:

```text
pay(d) = hold_limit + (seyir - hold_limit) * clip((d - safe_half) / (R - safe_half), 0, 1)
ceza   = -w * max(0, hız - pay)^2
```

| Mesafe | İzin verilen hız |
|---:|---:|
| `2.50 m` (`approach_radius`) | `1.50 m/s` |
| `1.50 m` | `0.82 m/s` |
| `1.00 m` | `0.49 m/s` |
| `0.65 m` (güvenli bölge) | `0.25 m/s` |

Güvenli iç bölgede pay sıfırlanır; orada hold hız sınırı geçerlidir.
`transit` kapsam dışıdır.

### Ölçülen sonuç (eğitimin `%20`'sinde)

```text
güvenli bölgeye (<= 0.65 m) giren terminal adım : %2  -> %14
yatay mesafe medyanı                            : 1.571 m -> 1.307 m
```

Yön doğru fakat yeterli değil.

**Durum:** Çözüldü, kalıcı. Tek başına yeterli değildi.

## REV-043 — Beş hover koşulunun birlikte hiç sağlanmaması

**Tarih:** 14 Eylül 2026  
**Kaynak:** REV-040/041/042 düzeltmelerinden sonra da başarı `0` kalması  
**Etkilediği:** ADR-012 başarı tanımı, ADR-017 reward yapısı

### Bulgu

Koşullar tek tek değil **kesişim** olarak ölçüldü (1565 terminal adım):

```text
0/5 koşul birlikte : %51.6
1/5                : %34.5
2/5                : %12.7
3/5                : % 1.3
4/5                : % 0.0
5/5                : % 0.0
```

**Hepsi birlikte sağlanan en uzun kesintisiz dizi: `0` adım.** Gereken `3.0 s`
= `60` adım.

Yani sorun "3 saniye tutamıyor" değil, **tek bir adımda bile beş koşulu
birlikte sağlayamıyor**.

### Yorum

Bu, tek tek ölçümlerin yanıltıcı olabileceğinin temiz bir gösterimidir. Her
koşulun ihlal oranı `%65-99` arasındayken kesişimin sıfır olması şaşırtıcı
değildir; fakat "hız `%99` ihlal" gibi tek eksenli bir okuma, düzeltmenin tek
bir bileşeni hedeflemesi gerektiğini ima eder ve bu yanlıştır.

Ölçülen üç düzeltme (REV-041 yön, REV-042 yavaşlama, REV-040 kenar payı)
kendi eksenlerinde çalıştı ve kalıcıdır. Fakat başarı, beş koşulun **aynı
anda** ve `60` adım boyunca sağlanmasını ister ve bu, her eksende ayrı ayrı
iyileşmekten çok daha zor bir hedeftir.

### Açık soru: görev fazla mı kısıtlı

Hold koşulları:

```text
merkeze <= 0.650 m  ve  hedef görünür olmak için >= 0.407 m   -> 0.243 m halka
irtifa 1.51 +- 0.20 m
hız <= 0.25 m/s
tilt <= maximum_tilt_rad
açısal hız <= 0.35 rad/s
kesintisiz 3.0 s
```

Bunların hiçbiri bu oturumda değiştirilmedi ve kasten değiştirilmedi: başarı
ölçütünü gevşetmek ölçümü anlamsız kılar. Fakat ADR-012'nin bu kombinasyonunun
`600k` adımda sıfırdan öğrenilebilir olup olmadığı **ölçülmemiş bir
varsayımdır** ve şu ana kadarki bütün kanıt aksini gösteriyor.

Seçenekler (kullanıcı kararı):

1. **Merdiveni sürdürmek** ve daha uzun bütçe vermek (`600k -> 2M`). Kanıt:
   her düzeltme sonrası metrikler doğru yönde ilerliyor fakat çok yavaş.
2. **Terminal görevi ayrı kademelendirmek**: `hold_seconds`'ı merdiven ekseni
   yapmak (`0.5 s -> 1 s -> 2 s -> 3 s`). Başarı ölçütü D1 kapısında `3 s`
   kalır; kademeler yalnız eğitim protokolüdür.
3. **Hold koşullarını ADR-012 düzeyinde yeniden gözden geçirmek.** En riskli
   seçenek: ölçütü değiştirmek geçmiş bütün ölçümleri karşılaştırılamaz kılar.

**Durum:** Ölçüldü, kullanıcı kararı bekler.

**Güncelleme (30 Eylül 2026):** ADR-025 terminal hover yardımcısı (dedektör hedef kutusu + FC attitude/irtifa ile klasik yerleşme) beş koşullu kesişimi sağladı: held-out 40 episode'da `%77.5` hover başarısı, hedefe varan her episode başarılı. Duvar RL'in tek başına değil, hibrit mimarinin işiyle aşıldı.

## REV-044 — Kamera pitch'i ile hover yüksekliğinin tek bir geometrik karar olması

**Tarih:** 14 Eylül 2026  
**Kaynak:** Kullanıcının gözlemi: *"Kamera yere çok fazla bakıyor, 45 derece çok
fazla. Bunu düzeltmeme sebebimiz ne?"*  
**Etkilediği:** ADR-011 kamera montajı, ADR-016 hover geometrisi, ADR-009
approval zinciri

### Gözlem doğruydu ve ölçülebilir bir bedeli vardı

Kamera aşağı baktığı için ileri görüş mesafesi irtifayla ölçeklenir:
`ileri görüş = irtifa / tan(pitch - vfov/2)`.

```text
45 derece -> 3.711 x irtifa
40 derece -> 5.626 x irtifa      +%52
```

REV-036'da ölçülen defekt buna bağlıydı: politika `0.81 m` irtifaya indiğinde
yalnız `3.0 m` ileri görüyordu.

### Alt sınır kesindir ve hold koşulundan gelir

Araç hedefin güvenli iç bölgesinde (`merkeze <= 0.65 m`) dururken hedefi
**görmek** zorundadır. Yakın sınır `irtifa / tan(pitch + vfov/2)` olduğu için:

| Pitch | Yakın sınır (h=1.51 m) | Hold mümkün mü |
|---:|---:|---|
| `30 derece` | `0.87 m` | **HAYIR** |
| `35 derece` | `0.71 m` | **HAYIR** |
| `37 derece` | `0.64 m` | evet (kıl payı) |
| `40 derece` | `0.55 m` | evet |
| `45 derece` | `0.41 m` | evet |

**Alt sınır `36.8 derece`.**

### Ölçülen yan etki: pitch tek başına değiştirilemez

`45 -> 40` derece yapıldığında hold halkası `0.243 m`'den **`0.098 m`**'ye
düştü — REV-023'te blocker sayılan `0.108 m`'den bile dar. Sebep: yakın sınır
`0.407 -> 0.552 m` taşındı, güvenli bölge ise `0.65 m`'de sabit.

Halkayı korumak için hover yüksekliği düşürülmek **zorunda**:

```text
pitch 40 derece, halka 0.243 m icin gereken hover yüksekliği = 1.103 m
```

Yani **pitch ve hover yüksekliği tek bir geometrik karardır**, ayrı
seçilemezler. Bu, ADR-016'nın "pitch bench ölçümüne kadar değişmez"
kararının neden bu kadar bağlayıcı olduğunu da açıklar: pitch'e dokunmak
hover geometrisinin tamamını yeniden hesaplamayı gerektirir.

### Uygulanan paket

```text
pitch_down_deg    45.0  -> 40.0
hover_height_m    1.50  -> 1.10     (halkayı korumak için, zorunlu)
cruise_altitude_m 2.156 -> 1.422    (8 m görüş için gereken irtifa düştü)
ground_half_size  21.13 -> 26.88 m  (daha yatay kamera daha uzağı görür)
```

Yeni geometri:

```text
hover irtifası 1.11 m | zemin izi 0.406 - 6.24 m | HOLD HALKASI 0.244 m
seyir irtifası 1.422 m -> o irtifada ileri görüş 8.00 m
```

Bonus: `cruise_altitude` gereksinimi `2.156 -> 1.422 m` düştü, yani REV-036'nın
tırmanma cezası hafifledi.

### Onay zinciri yenilendi (zorunlu)

Kamera geometrisi değişince bütün görüntüler değişir, yani encoder dataset'i
geçersiz olur:

```text
vision_dataset_d1_p40_20260914 (6528 kare, leakage_check=passed)
  -> vision_quality_v23_20260914 (3 seed, quality_passed=True)
  -> vision_audit_v5_20260914 (audit_passed=True)
  -> vision-encoder-v11
```

Algı kalitesi korundu (`development_ood`, 3 seed):

| Metrik | 45 derece (v22) | 40 derece (v23) |
|---|---:|---:|
| engel haritası IoU | `0.651` | `0.658` |
| `free_space_mae` | `1.05 m` | `1.07 m` |
| `target_recall` | `0.921` | `0.921` |

Rollout içi `free_space` hatası ise **iyileşti**: `1.26 m -> 1.02 m`
(`%90` dilimi `3.49 -> 2.54 m`).

### Davranışsal doğrulama: kaçınma İYİLEŞMEDİ

`runs/p40-d1a-seed20260912`, `300k`, 5 engel, hold `3.0 s` (kademe yok).
Eğitim trendi:

```text
çarpışma  : %32 -> %37   (45 derece koşusunda aynı noktada ~%30, sonda %8)
hedef kaybı: %35
başarı    : 0/916
```

Terminal koşullar (10 episode, 1654 adım):

```text
hız                : %100.0 ihlal
irtifa toleransı   : % 96.4
açısal hız         : % 94.9
tilt               : % 83.7
güvenli bölge dışı : % 75.3   (45 derecede %80.4)
yatay mesafe medyanı: 1.173 m (45 derecede 1.307 m)
güvenli bölgeye giren terminal adım: %21 (45 derecede %14)
```

Terminal yakınlaşma bir miktar iyileşti (`1.307 -> 1.173 m`, bölgeye giriş
`%14 -> %21`) fakat **çarpışma iyileşmedi.** `300k` bütçe `600k`'lık koşuyla
doğrudan karşılaştırılamaz; 45 derece koşusu da çarpışmayı ancak son çeyrekte
`%8`'e indirmişti.

### Sonuç

Geometri kararı doğru ve kalıcıdır: ileri görüş `+%52`, algı hatası
`1.26 -> 1.02 m`, hold halkası korundu, alt sınıra `3.2 derece` pay var.
Fakat **bağlayıcı kısıt kamera açısı değildi**; REV-043'te ölçülen "beş koşul
birlikte hiç sağlanmıyor" duvarı yerinde duruyor.

**Durum:** Uygulandı ve doğrulandı. Bağlayıcı kısıt değildi.

## REV-045 — Hold süresi kademelendirmesi (kuruldu, koşulmadı)

**Tarih:** 14 Eylül 2026  
**Kaynak:** REV-043 (beş koşul birlikte hiç sağlanmıyor)  
**Etkilediği:** eğitim protokolü; ADR-012 başarı ölçütü **değişmez**

### Gerekçe

REV-043 ölçümü: `5/5` koşul `%0.0`, en uzun kesintisiz dizi `0` adım.
Politika `success_bonus`'u hiç örneklemiyor, yani terminal görev hakkında
öğreneceği pozitif sinyal yok. Bu bir keşif problemidir.

### Kurulan yapı

`EnvironmentConfig.target_config_path` eklendi ve `env_factory` üzerinden
ortama geçiriliyor. Kademeler yalnız `hold_seconds` alanında farklıdır:

```text
target_area_hold05.toml : 0.5 s
target_area_hold10.toml : 1.0 s
target_area_hold20.toml : 2.0 s
target_area.toml        : 3.0 s   <- D1 KAPISI, değişmedi
```

Eğitim config'leri: `ppo_vision_d1a_hold{05,10,20,30}.toml`
(`400k / 300k / 300k / 400k`), her biri öncekinden warm start.

### Korumalar

- `target_config_path` yalnız `vision` ortamında kabul edilir.
- Kademe dosyaları `hold_seconds` dışında varsayılanla **birebir aynıdır**;
  test bunu `replace(ladder, hold_seconds=default) == default` ile doğrular.
- Varsayılan config `3.0 s`'de kalır ve evaluation onu kullanır; test bunu
  ayrıca sabitler.

### Ölçüldü ve YANLIŞ EKSEN çıktı

`hold05` kademesi (`0.5 s`) koşuldu ve `%42`'de ölçüldü:

```text
5/5 koşul birlikte : %0.0
en uzun kesintisiz dizi: 0 adım   (gereken 10 adım)
```

**Sıfırı 10'a bölmek de sıfırdır.** Süreyi `60` adımdan `10` adıma indirmek
hiçbir şey değiştirmedi çünkü beş koşulun birlikte sağlandığı **tek bir adım
bile yoktu**.

REV-043 bunu zaten ölçmüştü ("en uzun dizi 0 adım") fakat çıkarım yanlış
yapıldı: "3 s çok uzun" diye okundu, oysa veri "hiç sağlanmıyor" diyordu.
Bir sayıyı doğru ölçüp yanlış yorumlamanın örneğidir.

**Durum:** Kuruldu, testlendi, koşuldu ve **yanlış eksen olduğu ölçüldü**.
Altyapı zararsızdır ve durur; asıl eksik REV-046'dadır.

## REV-046 — Ödülün başarı ölçütünün beş koşulundan yalnız ikisini öğretmesi

**Tarih:** 14 Eylül 2026  
**Kaynak:** Hold kademelendirmesinin (REV-045) etkisiz çıkması  
**Etkilediği:** ADR-017 reward sözleşmesi (18 bileşen)

### Bulgu

ADR-012 başarısı beş hover koşulunun **kesişimini** ister. Ödül bileşenleri
bununla karşılaştırıldı:

| Hold koşulu | Şekillendiren bileşen | Durum |
|---|---|---|
| konum (güvenli iç bölge) | `target_progress` | var |
| hız | `settle_speed` | var |
| **irtifa** | — | **YOK** |
| **tilt** | — | **YOK** |
| **açısal hız** | — | **YOK** |

Üç koşulun hiç gradyanı yoktu. Politika onları ancak şans eseri tutturabilir,
beşini birlikte tutturma olasılığı pratikte sıfırdır. Ölçülen terminal ihlal
oranları bunu doğruluyor: hız `%100`, irtifa `%96`, açısal `%95`, tilt `%84`,
konum `%75` — şekillendirilmeyen üçü hiç iyileşmiyordu.

`cruise_altitude` bileşeni vardı fakat işe yaramıyordu: yalnız
`transit`/`search`'te aktif ve **farklı** bir irtifayı (seyir `1.42 m`)
hedefliyor, hold bandını (`0.91-1.31 m`) değil.

### Düzeltme

Üç bileşen eklendi, hepsi `settle_speed` ile aynı biçimde (sınırı **aşan**
fazlalığın karesi, sınır içinde ceza yok — yani başarı ölçütü ne gevşetilir
ne sıkılaştırılır):

```text
settle_altitude      weight 0.96
settle_tilt          weight 11.4
settle_angular_rate  weight 1.22
```

Ağırlıklar `settle_speed` ile **eşit baskı** verecek şekilde türetildi
(tipik fazlalık² × ağırlık ≈ `0.78/adım`) ve bu bir testle sabitlendi.

Eşikler hedef alan config'inden okunur, yani ödül ile başarı ölçütü tek
kaynaktan beslenir ve ayrışamaz.

### İki çelişki ölçülerek bulundu ve düzeltildi

**Çelişki 1: irtifa cezaları zıt yönlere çekiyordu.** İlk uygulamada üç
bileşen `BEARING_STATES`'te (search dâhil) aktifti. `search`'te
`cruise_altitude` aracı **yukarı** (`1.42 m`), `settle_altitude` **aşağı**
(`0.91-1.31 m`) çekiyordu. Hiçbir irtifa cezasız değildi.

```text
minimum_altitude terminasyonu: %79   ödül: -280
```

Düzeltme: `CRUISE_STATES` (`transit`, `search`) ile `APPROACH_STATES`
(`acquired`, `approach`, `settle`, `hold`) **ayrık** tanımlandı ve ayrıklık
modül yüklenirken `AssertionError` ile zorlanıyor.

**Çelişki 2: dönme aynı anda ödüllü ve cezalıydı.** `bearing_alignment`
`acquired`/`approach`'ta kamerayı hedefe döndürmeyi **ödüllendirir**; dönmek
açısal hız sınırını (`0.35 rad/s`, action üst sınırı `1.5`) aşmayı gerektirir.
`settle_angular_rate` aynı durumlarda bunu **cezalandırıyordu**.

Aynı mantık tilt için de geçerli: ölçülen denge tilt'i tam hızda `3.4 derece`
(sınır `10`) fakat hızlanma/yavaşlama geçişlerinde aşılır ve yaklaşma bunu
gerektirir.

Düzeltme: tilt ve açısal hız yalnız `STILL_STATES` (`settle`, `hold`) —
aracın gerçekten durması gereken yer. `STILL_STATES <= APPROACH_STATES`
invariantı da zorlanıyor.

**Çelişki 3: irtifa cezası mesafeden bağımsızdı.** Sabit tolerans, aracı
hedefi görür görmez hold bandına çekiyordu, hedef `10 m` uzakta olsa bile
(`2.5 m` irtifada ceza `1.36/adım`, `settle_speed` tipiğinin iki katı).

```text
minimum_altitude terminasyonu: %70
```

Düzeltme: irtifa payı da mesafeyle rampalanır (`settle_speed` payının aynı
deseni):

| Hedefe mesafe | İrtifa payı |
|---:|---:|
| `2.50 m` | `0.312 m` |
| `1.50 m` | `0.251 m` |
| `0.65 m` | `0.200 m` (hold toleransı) |

Düzeltmeden sonra `minimum_altitude` `%70 -> %6`.

### Bu dilimin dersi

Üç çelişkinin üçü de **aynı hata sınıfındandır**: bir ödül bileşeni eklerken
hangi durumlarda aktif olduğunu ve o durumlarda başka bir bileşenle zıtlaşıp
zıtlaşmadığını kontrol etmemek. Üçü de ancak eğitim çöktükten sonra fark
edildi.

Bunun için iki yapısal koruma kondu (`CRUISE_STATES ∩ APPROACH_STATES = ∅` ve
`STILL_STATES ⊆ APPROACH_STATES`), modül yüklenirken zorlanıyor. Bundan sonra
yeni bir terminal bileşeni eklenirken hangi kümede olduğu açıkça seçilmek
zorunda.

**Durum:** Uygulandı, üç çelişki düzeltildi, davranışsal doğrulama koşuyor
(`runs/rev046c-d1a-seed20260912`).

## REV-047 — İlerleme günlüğünün REV-040..046 için güncellenmemesi

**Tarih:** 22 Eylül 2026  
**Kaynak:** Yeni oturumda proje incelemesi  
**Etkilediği:** `docs/PROGRESS_LOG.md`, AGENTS.md §5 ve §9

### Bulgu

`PROGRESS_LOG.md` "Güncel durum" bölümü ve en yeni kayıt (14 Eylül 2026,
"Algı yenilemesi") REV-030..039 ile `d1-v11-seed20260912` koşusunda
duruyordu. Oysa 13-14 Eylül 2026 arasında dokuz koşu daha yapıldı ve yedi
bulgu (REV-040..046) yalnız `REVIEW_FINDINGS.md`'ye yazıldı:

```text
ladder-d1a   ABORTED  goal-at-boundary        REV-040
ladder2-d1a  ABORTED  heading-decoupled       REV-041
ladder3-d1a  ABORTED  no-deceleration         REV-042
ladder4-d1a  600k     %0 başarı               REV-043 ölçümü
p40-d1a      300k     %0 başarı               REV-044
hold-hold05  ABORTED  wrong-axis              REV-045
rev046       ABORTED  contradictory-altitude  REV-046
rev046b      ABORTED  premature-descent       REV-046
rev046c      600k     %0 başarı, eval 0/10    REV-046 doğrulaması
```

AGENTS.md §9 "güncel faz için yalnız Güncel durum bölümünü esas alın" der;
bölüm en az bir gün geride kaldığı için sonraki agent yanlış başlangıç
noktası alırdı.

### Düzeltme

22 Eylül 2026 kaydına "Geriye dönük kayıt" alt bölümü eklendi; koşu
metadata'sı (`runs/*/metadata.json`) ve REVIEW_FINDINGS'ten yeniden
kurulduğu açıkça belirtildi. "Güncel durum" bölümü yeniden yazıldı.

**Durum:** Kapatıldı.

## REV-048 — Actor algısının simülatör etiketlerini kopyalaması ve RL ortamında görsel çeşitliliğin yokluğu

**Tarih:** 22 Eylül 2026  
**Kaynak:** Kullanıcının itirazı: *"drone engellerin konumunu biliyor"*  
**Etkilediği:** ADR-011, ADR-014, ADR-015/021 observation sözleşmesi, ADR-017
görsel kapı, `build_scene_model`, `EbabilVisionNavigationEnv`

### Bulgu

Üç ayrı gözlem tek bulgu altında toplanır; hepsi ADR-022 ile ele alınır.

**(a) Etiket bağımlılığı.** Actor observation'ında engel konumu yoktur
(testli). Fakat actor'ın gördüğü bütün görsel kanallar (`vision_latent`,
`vision_aux`, `obstacle_map`), MuJoCo depth/segmentation tamponlarından
türetilen etiketlerle supervised eğitilmiş ve dondurulmuş bir ağdan gelir.
12 Eylül 2026 kaydı bunu "tanınmıyor, söylendi" diye yazmış fakat bulgu
olarak açmamıştı. Gerçek araçta bu etiketler yoktur; simülasyon başarısı
etiketleyicinin kalitesine bağlı kalır.

**(b) RL ortamında görsel randomization yok.** `apply_scene_visual_parameters`
ve `apply_frame_visual_parameters` yalnız dataset üreticisinde çağrılır.
`EbabilVisionNavigationEnv` sahneyi sabit engel rengi `(0.55, 0.22, 0.12)`,
sabit ışık ve yalnız kutu geometrisiyle kurar (`Obstacle.geometry`
varsayılanı `"box"`). Kodlayıcı geniş randomize dağılımda eğitilmiş, politika
o dağılımın tek bir dilimini görmüştür. Bunun politika başarısına etkisi
ölçülmemiştir; risk olarak kaydedilir.

**(c) Zamansal bilgi yok.** Observation tek karedir (sample-and-hold);
frame stack veya recurrent yapı kodda yoktur. Tek kareden bağıl hız ve
yaklaşma kestirilemez (plan ablation A10, ADR-011 "ayrı deney").

### Düzeltme

ADR-022: etiketsiz VAE kodlayıcı, latent düzeyinde `k = 4` yığın, reset ve
kare başına DR, geometri çeşitliliği. Uygulama planı
`docs/IMPLEMENTATION_PLAN.md` §23.

**Durum:** Yeniden çerçevelendi (26 Eylül 2026, ADR-023 §9). Kullanıcının kararıyla yasak olanın koordinat sızıntısı olduğu, etiketle eğitilmiş bir dedektör olmadığı netleşti: etiketler dedektörde kalır, politika yalnız görüntü kutularını görür. Görsel çeşitlilik kısmı ADR-022 §6 / Dilim 2 ile kapandı (ortam içi DR). Bulgu ADR-023 Dilim C-D ile kapanır.

## REV-049 — Etiketsiz VAE'nin ilk denemede hedef recall ve harita IoU kapılarını geçememesi

**Tarih:** 22 Eylül 2026  
**Kaynak:** `runs/vae_quality_v1_20260922` (ADR-022 §3 kapıları, ilk ölçüm)  
**Etkilediği:** ADR-022 kademe 1 zaman planı; ADR-022 §3 eşikleri **değişmez**

### Bulgu

Varsayılan config (`beta = 1.0`, latent `64`, `72 x 96`, 45 epoch) ile üç
seed eğitildi. Temsil çökmedi: `64/64` latent boyutu her iki ölçütle etkin,
validation yeniden üretim MSE `~0.008` (PSNR `~21 dB`). Serbest alan ve
hedef precision kapıları geçti; **hedef recall** ve **validation engel
haritası IoU** geçmedi:

| Metrik (3 seed ortalaması) | validation | development_ood | Kapı |
|---|---:|---:|---|
| etkin latent boyutu | `64 / 64` | `64 / 64` | `>= 32` geçti |
| yeniden üretim MSE | `0.008` | `0.003` | raporlanır |
| probe `free_space_mae` | `0.248` | `0.237` | `<= 0.30` geçti |
| probe `target_precision` | `0.862` | `0.832` | `>= 0.78` geçti |
| probe `target_recall` | **`0.597`** | **`0.685`** | `>= 0.85` **geçmedi** |
| probe `obstacle_map_iou` | **`0.268`** | `0.333` | `>= 0.30` validation **geçmedi** |
| probe `obstacle_map_precision` | `0.942` | `0.882` | raporlanır |
| probe `obstacle_map_recall` | `0.273` | `0.349` | raporlanır |

Seed'ler arası yayılım küçük (recall `0.583-0.606`, IoU `0.245-0.299`); üç
seed de aynı iki kapıda takıldı. `quality_passed = False`, audit koşulmadı.

### Teşhis: kayıp küçük nesneleri önemsemiyor

Birinci seed probe'unun hedef görünürlük hataları hedefin kadraj alanına
göre ayrıştırıldı (validation, 576 kare, `%60` görünür):

```text
hedef alanı (kadraj oranı)   n    recall
< 0.05 %                    11    0.27
0.05 - 0.2 %                28    0.18
0.2 - 1 %                   63    0.16
1 - 5 %                     77    0.39
> 5 %                      166    0.97
```

Kadrajın `%5`'inden büyük hedefler neredeyse eksiksiz çözülüyor; küçükler
çözülmüyor. `72 x 96` girişte `%1`'lik hedef `~70` piksel, `%0.2`'lik hedef
`~14` pikseldir; piksel toplamı üzerinden hesaplanan yeniden üretim kaybı bu
bölgeleri yok sayabiliyor. Etiketli encoder aynı çözünürlükte `0.92` recall
almıştı çünkü etiket ona tam bu küçük ipuçlarını öğretiyordu; etiketsiz
temsil bunu kendiliğinden öğrenmek zorunda.

Engel haritasında aynı desen mesafe ekseninde: satır bazlı recall üstten
alta `0.49 -> 0.36 -> 0.24 -> 0.15 -> 0.07 -> 0.03 -> 0.01 -> 0 -> 0`,
precision her satırda `>= 0.83`. Yani latent engel varlığını kaba biçimde
(üst satırlar, uzak/geniş bölgeler) taşıyor, yakın satırlardaki seyrek
pozitifleri (`%6-10`) taşımıyor.

Skor eşiği (`0.5`) sorunun kaynağı değildir: en iyi F1 eşiği `0.3-0.4`'te
recall `0.82-0.90` verir fakat precision `0.66-0.73`'e düşer; ölçüm
sözleşmesi (`compute_metrics`, `0.5`) değiştirilmedi.

### Karar

ADR-022 §3 uyarınca eşik düşürülmedi. Etiketsiz kaldıraçlar tek seed'lik
geliştirme taramasıyla ölçüldü (`runs/vae_sweep_v1_20260922`, yalnız
development split'leri, audit'e dokunulmadı):

Tek seed (`20260911`), 45 epoch, validation / development_ood:

| Varyant | free_space MAE | target P | target R | harita IoU | süre |
|---|---:|---:|---:|---:|---:|
| temel (`beta 1`, latent 64, 72x96) | `0.249 / 0.239` | `0.85 / 0.81` | `0.61 / 0.68` | `0.26 / 0.34` | `6.7 dk` |
| `beta 0.1` | `0.252 / 0.242` | `0.88 / 0.71` | `0.44 / 0.81` | **`0.34 / 0.36`** | `6.3 dk` |
| `beta 0.01` | `0.257 / 0.243` | `0.86 / 0.68` | `0.42 / 0.83` | `0.37 / 0.35` | `6.9 dk` |
| `beta 0.1` + latent 128 | **`0.217 / 0.229`** | `0.89 / 0.84` | `0.57 / 0.70` | **`0.38 / 0.37`** | `7.9 dk` |
| `beta 0.1` + 108x144 | `0.244 / 0.244` | `0.90 / 0.75` | `0.31 / 0.80` | `0.36 / 0.36` | `16.3 dk` |
| `beta 0.1` + fotometrik değişmezlik | `0.242 / 0.237` | `0.86 / 0.71` | `0.51 / 0.78` | `0.34 / 0.37` | `17.5 dk` |
| `beta 0.01` + fotometrik | `0.238 / 0.243` | `0.84 / 0.69` | `0.53 / 0.80` | `0.36 / 0.37` | `17.2 dk` |
| `beta 0.1` + 108x144 + fotometrik | `0.243 / 0.245` | `0.80 / 0.73` | `0.55 / 0.76` | `0.35 / 0.38` | `23.1 dk` |
| `beta 1` + latent 128 | **`0.213 / 0.222`** | `0.88 / 0.86` | `0.57 / 0.69` | `0.33 / 0.40` | `7.8 dk` |
| `beta 1` + latent 128 + kontrastif (InfoNCE, `w = 1`) | `0.211 / 0.219` | `0.90 / 0.85` | `0.54 / 0.69` | `0.35 / 0.38` | `34.4 dk` |

Okuma: `beta` düşürmek ve latent 128 engel haritası kapısını her iki
split'te geçiriyor; latent 128 ayrıca serbest alan hatasını `0.25 -> 0.22`'ye
indiriyor. Hedef recall ise hiçbir çözünürlük/kapasite varyantında `0.85`'e
yaklaşmıyor ve **iki split'te zıt yönde** hareket ediyor: validation
(`heldout_dim_cool`) düşük recall / yüksek precision, development_ood
(`development_bright_warm`) yüksek recall / düşük precision. Aynı probe
farklı görsel alanlarda farklı yerde kesiyor; yani latent'teki hedef yönü
küresel ton ve parlaklıkla karışık. Çözünürlük artışı bunu düzeltmedi
(`0.31 / 0.80`), yani sorun küçük nesne ayrıntısı değil, görünüş
değişmezliğidir.



### Eşikten bağımsız ölçüm: bilgi mi, kalibrasyon mu

Kapı `0.5` eşiğinde recall/precision ister. Aynı probe'lar için eşikten
bağımsız ölçümler çıkarıldı (validation / development_ood):

| Varyant | hedef AUROC | hedef AP | en iyi F1 (recall, precision) | harita IoU `@0.5` -> en iyi eşik |
|---|---:|---:|---|---|
| temel (`beta 1`, 64) | `0.78 / 0.79` | `0.87 / 0.89` | `0.76 (0.94, 0.64) / 0.78 (0.80, 0.75)` | `0.26 -> 0.52` / `0.34 -> 0.49` |
| `beta 0.1` | `0.73 / 0.76` | `0.83 / 0.86` | `0.76 / 0.77` | `0.34 -> 0.53` / `0.36 -> 0.50` |
| `beta 0.01` | `0.73 / 0.74` | `0.83 / 0.85` | `0.75 / 0.77` | `0.37 -> 0.53` / `0.35 -> 0.50` |
| `beta 0.1` + latent 128 | **`0.81 / 0.80`** | **`0.89 / 0.89`** | `0.78 (0.86, 0.72) / 0.78 (0.76, 0.80)` | **`0.38 -> 0.56`** / `0.37 -> 0.51` |
| `beta 0.1` + 108x144 | `0.69 / 0.78` | `0.79 / 0.87` | `0.72 / 0.78` | `0.36 -> 0.54` / `0.36 -> 0.49` |
| `beta 0.1` + fotometrik | `0.76 / 0.75` | `0.84 / 0.86` | `0.77 / 0.77` | `0.34 -> 0.51` / `0.37 -> 0.50` |
| `beta 0.01` + fotometrik | `0.75 / 0.73` | `0.84 / 0.85` | `0.76 / 0.77` | `0.36 -> 0.52` / `0.37 -> 0.50` |
| `beta 0.1` + 108x144 + fotometrik | `0.72 / 0.76` | `0.82 / 0.86` | `0.75 / 0.78` | `0.35 -> 0.52` / `0.38 -> 0.50` |
| `beta 1` + latent 128 | **`0.81 / 0.82`** | **`0.88 / 0.90`** | `0.78 (0.91, 0.68) / 0.80 (0.79, 0.80)` | `0.33 -> 0.54` / `0.40 -> 0.51` |
| `beta 1` + latent 128 + kontrastif | `0.80 / 0.81` | `0.88 / 0.90` | `0.78 / 0.79` | `0.35 -> 0.54` / `0.38 -> 0.51` |

İki sonuç: (1) Engel haritası büyük ölçüde **kalibrasyon** meselesi; en iyi
eşikte IoU `0.50-0.56`, yani latent haritayı taşıyor, probe `0.5`'te
tutucu kalıyor (seyrek pozitifler). (2) Hedef görünürlüğü **bilgi**
meselesi; en iyi varyantta bile AUROC `0.81`, oysa `0.85 recall + 0.78
precision`'ı aynı anda sağlamak için `~0.93+` AUROC gerekir. `beta`
düşürmek, çözünürlük ve fotometrik değişmezlik hedef AUROC'unu
**düşürdü** veya değiştirmedi; tek olumlu kaldıraç latent kapasitesidir.
Etiketli encoder aynı girişte `0.92 recall / 0.95 precision` almıştı;
yeniden üretim kaybı küçük ve seyrek bir nesneyi bu düzeyde
ayrıştırmaya zorlamıyor.

### Karar noktası (kullanıcı kararı bekler)

ADR-022 §3'ün saydığı bütün etiketsiz kaldıraçlar ölçüldü: `beta` (`1 / 0.1 /
0.01`), latent (`64 / 128`), giriş çözünürlüğü (`72x96 / 108x144`),
fotometrik değişmezlik (bozulmuş giriş, temiz hedef) ve kontrastif yardımcı
kayıp (InfoNCE, rastgele kırpma + fotometrik görünümler). Hiçbiri hedef
recall kapısını (`>= 0.85 @ 0.5`, aynı anda precision `>= 0.78`)
geçirmedi. En iyi etiketsiz aday `beta 1 + latent 128`: diğer beş kapıyı iki
split'te de geçiyor, hedef için AUROC `0.81 / 0.82` taşıyor. Eşik
düşürülmedi.

Seçenekler:

1. **Kapı tanımını değiştirmek (ADR-022 revizyonu).** Hedef kapısını sabit
   eşikli recall/precision yerine eşikten bağımsız çözülebilirlik yapmak:
   probe AUROC `>= 0.80` ve AP `>= 0.85`; engel haritası için en iyi eşikte
   IoU `>= 0.45` eklemek (`0.5`'teki `0.30` korunur). Gerekçe: sabit eşik
   probe'un kalibrasyonunu ölçüyor, latent'in bilgisini değil; politika
   doğrusal değildir ve kendi eşiğini öğrenir. Bedel: hedef bulma tamamen
   politikaya kalır; kademe 1 kapısı varış + çarpışma olduğu için bu bedel
   kademe 1'de değil hover aşamasında görünür.
2. **Etiketsiz aramaya devam.** Latent 256, `90+` epoch, daha geniş kod
   çözücü, kenar ağırlıklı yeniden üretim. Kapasite `64 -> 128` AUROC'u
   `0.78 -> 0.81` taşıdı; `0.93`'e ulaşacağına dair kanıt yok, her deneme
   `10-35 dk`.
3. **Hedefi ayrı, mühendislik ürünü bir dedektöre bırakmak.** ADR-022'nin
   engel için koyduğu etiketsizlik kuralı korunur; hedef işareti bilinen bir
   yeşil alan olduğu için deploy edilebilir bir işaret dedektörü (renk/şekil
   tabanlı veya yalnız hedef etiketiyle eğitilmiş küçük bir baş) kullanılır.
   Kullanıcının itirazı engel bilgisiydi; işaret dedektörü gerçek sistemde
   de mühendislik ürünüdür. Bedel: ADR-022 §1'e hedef için açık bir istisna.

Öneri: 1 numaralı seçenekle kademe 1'e devam etmek ve 3 numarayı hover
aşaması için açık tutmak.

### Karar (22 Eylül 2026, kullanıcı onayı): seçenek 1

ADR-022 Revizyon 1 yazıldı: hedef kapısı probe AUROC `>= 0.80` ve AP
`>= 0.85`; engel haritası için `0.5` eşiğinde IoU `>= 0.30` korunup en iyi
eşikte IoU `>= 0.45` eklendi; `latent_size 64 -> 128`. Sabit eşik
precision/recall raporlanır, kapı değildir. Kod tarafında `probes.py`
eşikten bağımsız ölçümleri üretir, `vae_quality.py` kapıları ve yükleyicinin
zorunlu kapı listesi yeni tanıma geçti; eski sabit eşikli zincir yükleyici
tarafından fail-closed reddedilir (test). Seçenek 3 hover aşaması için açık.

İkinci kalite koşusu (`runs/vae_quality_v2_20260922`, `beta 1`, latent 128,
üç seed): üç seed'in hepsi eşikten bağımsız kapıları geçti
(validation / development_ood, min-max): AUROC `0.805-0.809 / 0.817-0.822`,
AP `0.88 / 0.90`, en iyi eşikte harita IoU `0.51-0.54 / 0.50-0.51`, serbest
alan MAE `0.21-0.23`, etkin boyut `128/128`. Tek takılan kapı, korunan sabit
`0.5` eşiğindeki harita IoU (ADR-021 mirası): seed `20260912` `0.251`, seed
`20260913` `0.275`, kapı `0.30`; seed `20260911` `0.332` ile geçti.
`quality_passed = False`, audit koşulmadı.

Bu kapı, hedef için kaldırılan sabit eşikli kapıyla aynı türdendir: en iyi
eşikte IoU `0.52` iken `0.5`'te `0.25-0.33` olması latent'in haritayı
taşıdığını ve probe'un seyrek pozitiflerde tutucu kaldığını gösterir.
Onaylanan seçenek metni bu kapıyı açıkça koruduğu için kaldırılması ayrı
karar gerektirir.

**Karar (23 Eylül 2026, kullanıcı "ilerlemeye devam" onayı, öneri doğrultusunda): Revizyon 1b.** Sabit `0.5` eşiğindeki harita IoU kapısı kaldırıldı, değeri raporlanır. Üçüncü kalite koşusu (`runs/vae_quality_v3_20260923`): üç seed'in hepsi yedi kapıyı geçti (değerler v2 ile birebir aynı; deterministik eğitim), `quality_passed = True`; tek kullanımlık held-out audit (`runs/vae_audit_v1_20260923`): AUROC `0.809`, AP `0.892`, en iyi eşikte harita IoU `0.569`, serbest alan MAE `0.203`, `audit_passed = True`. Onaylı kodlayıcı seed `20260911`, manifest `runs/vae_audit_v1_20260923/approval_manifest.json`. Sabit eşikli değerler (`obstacle_map_iou_value` `0.25-0.33`, hedef recall `0.57-0.69`) raporlanmaya devam eder.

**Durum:** Kapatıldı (23 Eylül 2026). Seçenek 3 (ayrı işaret dedektörü) Dilim 3 hover sonuçlarına göre açık tutulur.

## REV-050 — YOLO dedektörünün audit'teki görülmemiş engel şeklinde (elipsoit) engel kapılarını geçememesi

**Tarih:** 29 Eylül 2026  
**Kaynak:** `runs/detector_audit_v1_20260929/audit_report.json` (ADR-023 §3 kapıları, tek kullanımlık audit)  
**Etkilediği:** ADR-023 Dilim B onayı; ADR-023 §3 eşikleri **değişmez**

### Bulgu

YOLO11n (seed `20260911`, 60 epoch, COCO ön-eğitimli) iki geliştirme
bölümünde bütün kapıları geçti; kullanıcı kararıyla tek seed ile audit'e
geçildi. Audit bölümü `vision_dataset_d1.toml` gereği yalnız **elipsoit**
engellerle üretilmiştir (eğitim/validation/development: kutu, silindir,
kapsül, küre). Engel kapıları audit'te geçmedi; hedef kapıları geçti:

| Kapı | Eşik | validation | development_ood | audit (elipsoit) |
|---|---|---:|---:|---:|
| engel recall (`>= 400 px^2`) | `>= 0.90` | `0.950` | `0.983` | **`0.456`** |
| engel precision | `>= 0.90` | `0.939` | `0.956` | **`0.887`** |
| hedef recall (`<= 5 m`) | `>= 0.85` | `0.973` | `0.983` | `0.966` |
| hedef precision | `>= 0.90` | `0.924` | `0.982` | `0.957` |
| sahte hedef oranı | `<= 0.05` | `0.013` | `0.005` | `0.003` |
| gecikme medyanı | `<= 15 ms` | `5.8` | `5.2` | `5.6` |
| bilgi: mAP50 engel / hedef | | `0.902 / 0.950` | `0.963 / 0.980` | `0.722 / 0.971` |

### Tanı (tüketilmiş audit bölümünde, karar kurallarını değiştirmeden)

Büyük gerçek elipsoitlerin (`2235`) en iyi eşleşen tahminine göre dağılımı;
karşılaştırma için validation (`1727` görülmüş şekil):

| Durum | audit | validation |
|---|---:|---:|
| doğru (IoU `>= 0.5`, güven `>= 0.25`) | `%45` | `%94` |
| yeri doğru, güven `< 0.25` | `%26` | `%3` |
| hiç örtüşen tahmin yok (IoU `< 0.1`) | `%23` | `%1` |
| kısmi kutu (`0.1 <= IoU < 0.5`) | `%6` | `%1` |

Güven eşiği `0.01`'e indirilse bile audit recall `0.71`'de kalır. Görsel
alan tek başına neden değildir: aynı audit karelerinde hedef recall `0.966`.
Kök neden **şekil genellemesi**: dedektör "yerden yükselen herhangi bir cisim"
kavramını değil, eğitimde gördüğü dört şekil ailesini öğrenmiştir.

### Etki

- Simülasyondaki RL görevi (`static_scene_d1a`: kutu/silindir/kapsül) bu
  şekillerle çalıştığı için dedektör o görevde geçerlidir (validation ve
  development_ood `>= 0.94`).
- Gerçek dünyada engel şekilleri keyfidir; şekle özgü öğrenme sim-to-real
  riskidir. ADR-023 §12 zaten gerçek görüntüyle yeniden eğitimi öngörür.

### Seçenekler (karar kullanıcıda)

1. **Yeni audit, RL görevinin şekilleriyle:** yeni salt ile yalnız audit
   bölümü üretilir (kutu/silindir/kapsül, held-out görsel alan, eğitim sahne
   seed'leriyle çakışmaz), mevcut aday yeniden eğitilmeden bir kez ölçülür.
   Elipsoit sonucu bilinen sınır olarak kalır. Audit tanımı değişir.
2. **Şekil çeşitliliğini büyütüp yeniden eğitmek:** elipsoit eğitime alınır,
   audit için yeni bir şekil ailesi (ör. rastgele dışbükey mesh) eklenir;
   dataset, eğitim ve audit yeniden. Daha sağlam, yaklaşık yarım gün.
3. Eşik düşürmek **seçenek değildir** (proje kuralı).

**Karar (29 Eylül 2026, kullanıcı):** "Buna gerek yok. Eğitimde kullandığımız
engelleri kullanalım environmentte." Yeni audit veya yeniden eğitim yapılmadı.
Dedektör, eğitimde görülen engel şekilleriyle (kutu, silindir, kapsül, küre)
**sınırlı** olarak onaylandı: `runs/detector_audit_v1_20260929/approval_manifest.json`
(`approval_basis = "user_decision"`, `failed_audit_gates = [obstacle_precision,
obstacle_recall]`, başarısız audit raporu hash zincirinde). `EbabilVisionNavigation-v2`
bu küme dışındaki şekil isteyen sahne config'ini kurulumda reddeder (test).
Gerçek dünya şekil genellemesi ADR-023 §12'ye (gerçek görüntüyle yeniden eğitim)
ertelendi.

**Durum:** Kullanıcı kararıyla kapatıldı (kısıtlı onay); şekil genellemesi
bilinen sınır olarak açık.

## REV-051 — Dedektör politikasının kameranın görmediği yöne uçarak çarpışması ve hedef üstünde tutunamaması

**Tarih:** 29 Eylül 2026  
**Kaynak:** `runs/detector_d1a_seed20260912_20260929/evaluation_heldout_final_det.json` (ADR-023 §10, seed `20260912`, 600k adım, deterministik, held-out görsel alan, 40 episode)  
**Etkilediği:** ADR-023 Dilim D kapısı; ADR-020 holonomik action sözleşmesi

### Bulgu

| Metrik | Değer | Kapı |
|---|---:|---|
| arrival (gerçek hedefe `<= 2.5 m`, fiziksel failure yok) | `%55` | `>= %80` **geçmedi** |
| çarpışma | `%35` | `<= %10` **geçmedi** |
| hover başarısı | `%0` | raporlanır |
| sonlanma | hedef kaybı abort 22, çarpışma 14, sınır dışı 3, arama zaman aşımı 1 | |

Eğitim sırasında evaluation ödülü `-30.6`'dan (50k) `+9.4`'e (400k) çıktı, son
nokta `-3.2` (600k); VAE politikası (`v1`) hiç pozitife çıkmamıştı.

### Tanı

1. **Çarpışmalar kör yöne uçuştan.** 14 çarpışma episode'u deterministik
   olarak yeniden oynatıldı; son 10 adımdaki ortalama hız vektörünün gövdeye
   göre yönü ölçüldü. Yalnız 1'i kameranın yatay görüş açısı içinde
   (`±37.5` derece) hareketle; 5'i yana (`75-106` derece), 8'i geriye çapraz
   veya tam geri (`130-180` derece). 6 episode'da son 10 adımda hiç engel
   kutusu yoktu. ADR-020 holonomik action yönü hareketten ayırır; REV-041'in
   ölçtüğü yapısal risk ("görüş konisi dışında yakın engel") burada çarpışma
   nedeni olarak doğrulandı. Algı hatası değildir: çarpışılan engeller
   kadraja hiç girmemiştir.
2. **Hedef üstünde tutunamama.** Hedef kaybı abort'larının 14/22'sinde araç
   hedef merkezine `0.5 m`'den fazla yaklaşmıştı. Yeniden oynatılan bir
   episode'da dedektör hedefi gördü, durum makinesi `settle`'a geçti
   (`0.51 m`), ardından araç kaydı, hedef kadrajdan çıktı ve `2.5 s` sonra
   görev bırakıldı. Bu REV-043 duvarıdır (beş hover koşulu birlikte); algı
   hedefi doğru anda görüyordu.

### Seçenekler (karar kullanıcıda)

1. **"Baktığın yöne uç" kısıtı (önerilen):** transit ve aramada hareket
   yönü kameranın görüş konisiyle sınırlanır (ileri hız `>= 0`, yanal/ileri
   oranı koniyle sınırlı; dönüş yaw ile). Terminal fazlarda (approach,
   settle, hold) düşük hızlı holonomik düzeltmeye izin verilir. Action
   sözleşmesi değişir: ADR-020 revizyonu ve yeniden eğitim gerekir.
2. **Ödülle caydırma:** holonomik action korunur, hız vektörünün görüş
   konisi dışındaki bileşenine ceza eklenir. Daha yumuşak; çarpışmayı
   garanti etmez.
3. **Hafıza:** geçmiş kutuları odometriyle dünya çerçevesinde tutan bir
   engel belleği; kadrajdan çıkan engelleri politika görmeye devam eder.
   En karmaşık seçenek.

Hover tutunma sorunu (2) ayrı bir eksendir; seçenek 1 yanal kaymayı da
azaltabilir, fakat REV-043'ün ölçtüğü beş koşullu kesişimi tek başına çözmez.

**Karar (29 Eylül 2026, kullanıcı):** "baktığın yöne uç kısıtını uygulayalım"
(seçenek 1). ADR-024: seyir/aramada yatay hız kamera konisine (yarım açı 30
derece) izdüşürülür, terminal durumlarda koni dışı hız 0.5 m/s ile sınırlıdır
(`vision-action-v3`). Aynı tarifle koşan seed `20260913` eğitimi 30720 adımda
durduruldu (sonuç kararı değiştirmezdi). Kısıtlı ortamla seed `20260912`
yeniden eğitiliyor: `runs/detector_cone_d1a_seed20260912_20260929`.

**Ölçüm (30 Eylül 2026):** koni kısıtlı seed `20260912` held-out: arrival
`%70` (önce `%55`), çarpışma `%17.5` (önce `%35`); kör yöne uçuş çarpışması
seyirde kalmadı. Kalan 7 çarpışmanın 4'ü görülen engelden kaçınamama
(seyir, görüş açısı içinde), 3'ü `approach`'ta izin verilen kör yanal hareket.
Ayrıntı ADR-024 "Ölçülen sonuçlar".

**Durum:** Çarpışma kısmı büyük ölçüde giderildi, kapı (`<= %10`) henüz
geçilmedi; hover tutunma kısmı ADR-025 terminal yardımcısıyla aşıldı
(30 Eylül: held-out 40 episode'da `%77.5` hover başarısı).

## Son regresyon sonucu

Altı bulgunun tamamı kapatıldı. Bütün düzeltmeler birlikteyken aşağıdaki kapılar
yeniden çalıştırıldı:

```text
pytest: 66 test + 7 subtest başarılı
ruff lint: başarılı
ruff format --check: 42 dosya formatlı
mypy: 25 üretim dosyasında sorun yok
compileall: başarılı
pip check: bozuk bağımlılık yok
wheel install smoke: model, 5 config ailesi ve Gym reset başarılı
hover smoke: final position error 0.008606 m
clear smoke: 109 adımda goal
single_blocker smoke: 28 adımda beklenen obstacle_collision
random rollout: 2000 adım, finite observation/reward, 160.34 step/s
viewer dry-run: statik sahne ve 3001-frame hover replay başarılı
```

Final turda önceki model, hover, action/observation, reward, termination ve viewer
davranışlarında regresyon gözlenmedi. GUI penceresi bu otomatik turda açılmadı;
viewer giriş noktaları dry-run ve otomatik test ile doğrulandı.
