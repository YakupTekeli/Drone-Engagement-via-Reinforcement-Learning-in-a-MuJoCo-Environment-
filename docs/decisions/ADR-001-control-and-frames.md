# ADR-001: Kontrol Soyutlaması ve Koordinat Çerçeveleri

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

## Bağlam

Quadrotor politikası doğrudan motor komutu, thrust/body-rate komutu veya yüksek seviyeli hız komutu üretebilir. Ayrıca frame ve rotor sırası açık tanımlanmazsa mixer, sensör ve reward kodunda sessiz işaret hataları oluşabilir.

## Karar

İlk navigasyon politikası şu normalize edilmiş komutları üretecektir:

```text
[vx_ref_body, vy_ref_body, vz_ref_world_or_body, yaw_rate_ref]
```

`vz_ref` çerçevesi Gymnasium ortamı uygulanırken deney sözleşmesinde kesinleştirilecek ve ayrı ADR veya bu ADR güncellemesiyle sabitlenecektir. İlk tercih, pilotaj sezgisini kolaylaştırdığı için dünya `z` eksenidir.

> **Güncelleme:** `vz_ref`, ADR-008 ile dünya `z` ekseni olarak kesinleştirilmiştir.

Alt seviyede klasik hız, attitude ve body-rate kontrol döngüleri ile motor mixer bulunacaktır.

Koordinat sözleşmesi:

- Dünya: sağ elli, `+z` yukarı.
- Başlangıç gövde çerçevesi: FLU (`+x` ileri, `+y` sol, `+z` yukarı).
- Pozitif açılar: ilgili pozitif eksen çevresinde sağ el kuralı.
- Rotor sırası:
  1. `front_left`: `(+x, +y)`
  2. `front_right`: `(+x, -y)`
  3. `rear_right`: `(-x, -y)`
  4. `rear_left`: `(-x, +y)`

İlk MJCF aktüatör kontrolü Newton cinsinden rotor thrust'ıdır. Motor komutu, rotor hızı ve birinci dereceden motor gecikmesi Faz 2'de ayrı dinamik katman olarak eklenecektir.

## Gerekçe

- Yüksek seviyeli eylem, navigasyon ve temel uçuş kararlılığını ayırır.
- Aynı politika farklı araçlara daha kolay taşınabilir.
- Klasik alt seviye kontrolcü fizik ve reward hatalarının teşhisini kolaylaştırır.
- Açık frame sözleşmesi mixer ve sensör işaretlerinin otomatik test edilmesini sağlar.

## Sonuçlar

- PPO ilk aşamada motor arızası veya agresif akrobasi öğrenmeyecektir.
- Alt seviye kontrolcü ayrıca uygulanmalı ve doğrulanmalıdır.
- Daha düşük seviye action space'ler sonraki ablation olarak aynı benchmark üzerinde karşılaştırılabilir.
