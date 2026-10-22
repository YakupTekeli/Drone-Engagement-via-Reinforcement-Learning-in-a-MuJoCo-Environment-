# ADR-006: Rangefinder, Sensör Bozulmaları ve Statik Sahne Üretimi

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

> **Güncelleme:** Açık düz koridor yalnızca `clear` seviyesi için korunmuştur. Kaçınma seviyeleri ve alternatif rota doğrulaması ADR-007 ile tanımlanır.
>
> **Donanım güncellemesi (11 Eylül 2026):** Hedef araçta LiDAR/rangefinder
> bulunmayacaktır. Bu ADR yalnız `EbabilNavigation-v0` araştırma baseline'ı,
> privileged eğitim etiketi ve değerlendirme metriği için geçerlidir. Deploy
> edilecek actor kamera tabanlı olacaktır; bkz. ADR-011.

## Bağlam

Navigasyon ortamı, politika için sabit sıralı mesafe gözlemleri ve aynı seed ile yeniden üretilebilen engel sahneleri gerektirir. İdeal MuJoCo sensörleri tek başına gerçek sensörlerin örnekleme, gecikme, bias, noise ve dropout davranışını temsil etmez.

## Karar

### Rangefinder

- İlk algılayıcı gövde `xy` düzleminde 16 ışınlı 360° planar rangefinder'dır.
- Işın `0` gövde `+x` yönündedir; kalan ışınlar pozitif body `z` çevresinde eşit aralıklıdır.
- Işınlar MuJoCo `mj_ray` ile atılır ve quadrotor gövdesi sorgudan çıkarılır.
- Yalnızca `geom group=1` engeller algılanır; zemin, drone görselleri ve hedef marker'ı menzil ölçümüne girmez.
- İsabet yoksa değer maksimum menzil `5 m`; observation için normalize değer `distance / max_range` olur.

### Sensör zaman ve hata modeli

Genel `SampledSensor` katmanı şu sırayı uygular:

```text
true value → episode bias + white noise → clipping
           → sample-and-hold / dropout → delay buffer → observation
```

Rastgelelik episode seed'ine bağlıdır. İlk entegrasyonda bu bozulma rangefinder'a uygulanır; MJCF position/orientation/velocity/gyro/accelerometer değerleri `IdealSensorReading` altında ayrıca korunur. Diğer kanallar için birimlerine uygun noise parametreleri Gymnasium ortamı kurulurken ayrıştırılacaktır.

### Statik sahneler

- Engeller zemin üzerinde duran axis-aligned kutulardır.
- Sahne üretimi NumPy RNG ve açık seed kullanır.
- Başlangıç/hedef çevresinde açıklık ve engeller arasında minimum XY açıklığı uygulanır.
- İlk baseline, başlangıç ile hedef arasında belirlenen yarıçapta doğrudan yatay koridoru engellerden uzak tutarak çözülebilirliği garanti eder.
- Engeller çalışma zamanında MuJoCo `MjSpec` ile temel quadrotor modeline eklenir; ayrı XML kopyaları üretilmez.
- Hedef, collision üretmeyen mocap sphere ile viewer'da görünür kılınır.

## Gerekçe

- Planar ışınlar ilk PPO baseline'ı için düşük boyutlu, sıralı ve yorumlanabilir observation sağlar.
- Geom group filtresi sensörün kendi aracını veya görsel yardımcıları yanlış engel saymasını önler.
- Seed'li üretim başarısız episode'ların tam tekrarını mümkün kılar.
- Garantili düz koridor, ilk curriculum seviyesinde çözülemez sahne üretme riskini kaldırır.
- `MjSpec` yaklaşımı araç modelini tek kaynakta tutar.

## Sonuçlar ve sınırlılıklar

- Planar sensör aracın üstündeki/altındaki engelleri göremez; ileride dikey/eğimli ışın katmanları veya 3B LiDAR ablation gerekir.
- Düz koridor garantisi geometrik olarak konservatiftir ve yalnızca ilk kolay seviye içindir.
- İlk engeller yalnızca kutudur; silindir, duvar, geçit ve koridor şablonları curriculum aşamasında eklenecektir.
- İdeal state/IMU henüz estimator değildir; gerçek uçuş iddiası oluşturmaz.
