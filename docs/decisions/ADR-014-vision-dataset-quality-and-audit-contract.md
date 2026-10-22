# ADR-014: Vision Dataset, Quality ve Bağımsız Audit Sözleşmesi

**Durum:** Kabul edildi  
**Tarih:** 11 Eylül 2026

## Bağlam

İlk perception geliştirme döngüsünde aynı test split'i mimari ve eşik
kararlarında tekrar gözlendi. Engel/hedef materyal renkleri semantik kısa yol
oluşturuyor, bütün engeller kutu olarak çiziliyor ve quality artefact'ı dataset
ile checkpoint bütünlüğünü taşınabilir biçimde kanıtlamıyordu.

## Karar

- Dataset şeması `vision-dataset-v4`; splitler `train`, `validation`,
  `development_ood` ve `audit` olarak ayrılır. Explicit sahne kotaları oranların
  tek kaynağıdır.
- Trainer ve development quality runner audit split'ini hiçbir zaman yüklemez.
  Audit yalnız model, eşikler ve candidate checkpoint dondurulduktan sonra ayrı
  `run_vision_audit.py` süreciyle çalışır.
- Başarısız quality veya audit süreç kodu non-zero döndürür. Quality yalnız
  candidate üretir; approved checkpoint yalnız başarılı audit'in
  `approval_manifest.json` dosyasında yayınlanır.
- Audit başlamadan quality dizininde atomik `audit_consumed.json` oluşturulur.
  Başarısızlık veya süreç kesintisi dahil aynı quality raporu ikinci kez audit
  edilemez; yeni deneme yeni dataset salt ve quality koşusu gerektirir.
- Dataset config/metadata/manifest/statistics, quality report, audit report ve
  checkpoint SHA-256 zinciriyle bağlanır. Artefact yolları göreli olmalıdır.
  Runtime yükleme yalnız `load_approved_vision_encoder` üzerinden yapılır.
- Actor preprocessing tek public `preprocess_actor_rgb` yoludur: HWC `uint8`
  RGB, CHW `float32 [0,1]` olur. Model yanlış dtype/aralık/NaN girdisini reddeder.
- Train/validation/development engelleri kutu, silindir, capsule ve sphere
  primitive'lerini; audit train-dışı ellipsoid primitive'ini kullanır. Engel
  renkleri bütün kanallarda geniş randomize edilir ve hedef renginde
  çarpışmasız/maskesiz distractor'lar bulunur.
- Eğitimde yüzde 50 seed'li grayscale augmentation vardır. Nominal metriklere
  ek olarak grayscale free-space ve risk metrikleri de quality/audit kapısını
  geçmelidir.
- `vision-encoder-v9` risk score eşiği `0.62`'dir. Geliştirme kapıları target
  P/R `0.78/0.85`, center error `0.30`, free-space MAE `0.30`, trivial baseline
  iyileşmesi `0.15`, risk P/R/S `0.65/0.70/0.50` değerleridir. Aynı risk
  eşikleri grayscale ablation'a da uygulanır.

## Gerekçe

Dört-split ve tek-seferlik audit, geliştirme kararlarıyla nihai sentetik
değerlendirmeyi ayırır. Renk distractor'ları, geniş materyal dağılımı, farklı
primitive'ler ve grayscale kapısı salt renk ezberini ölçülebilir biçimde
zorlaştırır. Hash zinciri yanlış dataset/checkpoint eşleşmesini ve kazara
artefact değişimini fail-fast yakalar.

Risk eşikleri yeni salt'lı validation/development-OOD verisindeki üç model seed'i
ve nominal/grayscale koşullar birlikte değerlendirilerek, ikinci audit açılmadan
önce dondurulmuştur. İlk audit başarısızlığı eşik düşürmek için kullanılmamış;
geometri zarfı genişletilmiş ve yeni salt + yeni ellipsoid audit oluşturulmuştur.

## Sonuçlar ve sınırlar

- v16 development quality ve yeni 96-kare ellipsoid audit geçti.
- İlk başarısız audit tarihsel kayıt olarak korunur ve approval içermez.
- Onay sentetik perception bileşenini vision environment'a kabul eder; gerçek
  IMX477, Jetson gecikmesi veya gerçek uçuş güvenliği kanıtı değildir.
- Gerçek kamera görünümü kesinleşince ayrı, yalnız değerlendirmelik gerçek kayıt
  seti ve donanım üstü latency/parity kapısı gerekir.
