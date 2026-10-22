# ADR-004: Kademeli Klasik Uçuş Kontrolcüsü

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

## Bağlam

PPO navigasyon politikasından önce aracın temel uçuş kararlılığını sağlayan, MuJoCo'dan bağımsız bir arayüzle test edilebilen alt seviye kontrol zinciri gereklidir. Bu katman hem RL problemini sadeleştirmeli hem de frame, mixer veya fizik hatalarının politikaya yüklenmeden teşhis edilmesini sağlamalıdır.

## Karar

Kontrol zinciri aşağıdaki sırada uygulanacaktır:

```text
FlightTarget(position, velocity, acceleration, yaw, yaw-rate)
                              │
                              ▼
               dünya-frame position/velocity PD
                              │
                    desired force + attitude
                              ▼
                    SO(3) attitude P
                              │
                    desired body rates
                              ▼
          body-rate PI + gyroscopic compensation
                              │
                 collective thrust + torque
                              ▼
           mixer → rotor gecikmesi → MuJoCo
```

İlk baseline'da bütün döngüler fizik zaman adımında, `500 Hz` hızında senkron çalışır. PPO eylemi ileride daha düşük frekansta tutulacak ve bu hedef yapısına dönüştürülecektir.

### Konum ve hız döngüsü

Dünya çerçevesindeki komut ivmesi:

```text
a_cmd = a_ff + Kp_pos * (p_ref - p) + Kd_pos * (v_ref - v)
F_des = m * (a_cmd + g * e_z)
```

Yatay/dikey ivme, maksimum tilt ve collective thrust sınırlanır. İstenen thrust yönü ile yaw referansı ortonormal bir desired rotation matrisi üretir.

### Tutum döngüsü

Tutum hatası rotation matrix'ler üzerinden SO(3) üzerinde hesaplanır. Euler açı farkı kontrol hatası olarak kullanılmaz; bu sayede açı sarma ve eksen sırası sorunları azaltılır. Çıkış body-rate hedefidir ve eksen bazında sınırlanır.

### Body-rate döngüsü

Body-rate kontrolü PI'dır. Ölçülen açısal hız zaten iç döngü geri beslemesi olduğundan ayrı bir derivative terimi eklenmemiştir. Nominal diyagonal ataletle ivme komutu torka çevrilir ve `omega × I*omega` gyroscopic compensation eklenir. Tork limiti devreye girdiğinde ilgili eksendeki integrator güncellenmez.

### Konfigürasyon ve arayüz

- Kazanç ve limitlerin tek kaynağı: `src/ebabil/configs/controller/cascaded.toml`
- Simülatörden bağımsız veri tipleri: `FlightState` ve `FlightTarget`
- MuJoCo adaptörü: `read_flight_state`
- Headless kapalı çevrim doğrulama: `run_hover_simulation`
- Kontrol ve rotor izleri sonuç nesnesinde tutulur; doygunluk oranları ayrıca raporlanır.

## Gerekçe

- Position/velocity hedefi, ilerideki yüksek seviyeli hız eylemiyle uyumludur.
- SO(3) tutum hatası quaternion işaret belirsizliğinden ve Euler tekilliklerinden kaçınır.
- Atalet ölçekli rate kontrolü eksenlerin fiziksel tepkisini hesaba katar.
- Açık limitler ve anti-windup, hatalı veya agresif hedeflerin kontrol zincirini sayısal olarak bozmasını engeller.
- Kontrolcünün propulsion katmanından ayrılması saf birim testlerine ve farklı araç parametreleriyle yeniden kullanıma izin verir.

## Doğrulama

Sabit seed `20260910` ile 100 başlangıç örneği kullanılmıştır:

- Konum: `x,y ∈ [-0.5, 0.5] m`, `z ∈ [0.8, 1.2] m`
- Roll/pitch: `[-15°, 15°]`
- Yaw: `[-30°, 30°]`
- Süre: `6 s`
- Başarı: final konum hatası `<0.05 m`, hız `<0.08 m/s`, tutum hatası `<2°`, minimum irtifa `>0.10 m`

Sonuç `100/100` başarılıdır. En kötü final konum hatası `0.003535 m`, hız `0.008660 m/s`, tutum hatası `0.118509°`; gözlenen mixer ve tork doygunluğu `0` olmuştur.

## Sonuçlar ve sınırlılıklar

- Baseline ground-truth konum, hız ve tutum kullanır; sensör noise, bias, delay ve state estimator henüz yoktur.
- Sabit `1 m/s` rüzgâr altında PD konum döngüsü yaklaşık `0.04 m` kalıcı yatay hata bırakır. Konum integrali bilinçli olarak eklenmemiştir.
- Mixer kırpmasının integrator'a geri beslenmesi ve wrench önceliklendirmeli constrained allocation henüz yoktur.
- Kazançlar nominal simülasyon modeli için ayarlanmıştır; gerçek araçta kullanılmadan önce sistem tanımlama, tekrar ayar ve bağımsız güvenlik doğrulaması gerekir.
- Multi-rate scheduler ve PPO action hold, Gymnasium ortamı kurulurken eklenecektir.
