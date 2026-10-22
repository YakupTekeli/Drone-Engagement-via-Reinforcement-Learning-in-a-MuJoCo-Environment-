# ADR-003: Propulsion, Mixer ve Sadeleştirilmiş Aerodinamik Model

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

## Bağlam

MJCF actuator'ları doğrudan Newton thrust kabul eder. Gerçek motor ve propeller ise komuta anında cevap vermez; rotor hızı üzerinden yaklaşık quadratic thrust üretir. Ayrıca hava-relative sürükleme olmadan yatay hareket ve rüzgâr etkisi gerçekçi değildir.

## Karar

### Parametre kaynağı

Nominal araç parametreleri `src/ebabil/configs/vehicle/quadrotor.toml` içinde tutulur. MJCF ile tekrar edilen kritik değerler otomatik eşleşme testine tabidir.

### Mixer

Rotor thrust vektörü aşağıdaki lineer ilişkiyle çözülür:

```text
[Fz, tau_x, tau_y, tau_z] = A * [f_FL, f_FR, f_RR, f_RL]
```

Allocation matrix rotorların FLU gövde konumlarından ve reaction torque işaretlerinden oluşturulur. İlk uygulamada negatif veya maksimum üzeri çözümler rotor bazında `[0, max_thrust]` aralığına kırpılır ve `saturated` olarak raporlanır.

### Rotor dinamiği

Controller/mixer istenen thrust'ı üretir. Rotor modeli bunu hedef rotor hızına dönüştürür:

```text
omega_target = sqrt(thrust_target / k_f)
```

Rotor hızı farklı yükselme ve düşme zaman sabitlerine sahip birinci dereceden modelle ilerletilir. Sayısal zaman adımı bağımlılığını azaltmak için exact exponential discretization kullanılır.

### Sürükleme ve rüzgâr

Sürükleme gövde eksenlerinde relative hava hızına göre quadratic uygulanır:

```text
v_air_body = v_body - wind_body
F_drag_body = -c * |v_air_body| * v_air_body
```

Kuvvet dünya çerçevesine dönüştürülerek MuJoCo `xfrc_applied` alanına eklenir. Uygulama mevcut harici wrench'i silmez; çağıran simülasyon döngüsü her adım başında alanı temizlemelidir.

## Gerekçe

- Thrust seviyesindeki mixer kontrol tasarımını motor donanımından ayırır.
- Exact rotor modeli 1–2 ms timestep değişikliklerinde daha tutarlı cevap verir.
- Ayrı yükselme/düşme sabiti gerçek motorların asimetrik cevabına yaklaşır.
- Body-axis drag yönelime bağlı sürüklemeyi düşük maliyetle temsil eder.
- Tüm bileşenlerin saf hesap kısımları MuJoCo'dan bağımsız test edilebilir.

## Sonuçlar

- Controller doğrudan `data.ctrl` yazmamalı; `QuadrotorPropulsion` üzerinden geçmelidir.
- Hover başlatma testlerinde rotor state'i hover thrust'ına initialize edilebilir; kalkış senaryosunda sıfır RPM kullanılmalıdır.
- Mixer saturation, controller anti-windup ve RL gözlemi açısından raporlanmalıdır.
- Daha sonra gerçek thrust stand verisi gelirse quadratic eğri lookup table veya fit edilmiş modelle değiştirilebilir.

## Bilinen sınırlılıklar

- Constrained control allocation ve eksen öncelikleri yoktur.
- Aerodinamik moment, rotor inflow, ground effect ve battery voltage yoktur.
- Katsayılar ölçülmüş değildir; sim-to-real iddiasında kullanılamaz.
