# ADR-002: İlk Quadrotor Modeli ve Aktüatör Anlamı

**Durum:** Kabul edildi  
**Tarih:** 10 Eylül 2026

## Bağlam

Kontrolcü geliştirmeden önce modelin kütle, atalet, rotor geometrisi ve actuator işaretleri üzerinde ortak ve test edilebilir bir başlangıç noktası gereklidir. Henüz hedef gerçek araç seçilmediği için bu değerler sistem tanımlamasına dayanmamaktadır.

## Karar

İlk nominal model aşağıdaki parametreleri kullanır:

| Parametre | Değer |
|---|---:|
| Toplam kütle | `1.0 kg` |
| Atalet `Ixx` | `0.005 kg·m²` |
| Atalet `Iyy` | `0.005 kg·m²` |
| Atalet `Izz` | `0.009 kg·m²` |
| Rotor merkez kolu | `0.23 m` |
| Tek rotor maksimum thrust | `6 N` |
| Yaw moment/thrust oranı | `0.02 m` |
| Physics timestep | `0.002 s` |

Dört MJCF actuator kontrolü şimdilik doğrudan rotor thrust'ını Newton cinsinden alır. Actuator sırası:

```text
[front_left, front_right, rear_right, rear_left]
```

Yaw moment oranı işaretleri:

```text
[+0.02, -0.02, +0.02, -0.02]
```

Bu işaretler fiziksel rotor dönme yönünün görsel animasyonu değildir; rotor thrust'ına eşlik eden reaction torque yönünü tanımlar.

## Gerekçe

- `1 kg` nominal araç ve yaklaşık `2.45` thrust-to-weight oranı kontrol geliştirmek için makul bir başlangıç aralığı sağlar.
- Doğrudan Newton thrust girdisi, mixer ve frame işaretlerini motor eğrisi belirsizliğinden bağımsız doğrulamayı sağlar.
- Model parametreleri gerçek araç seçildiğinde sistem tanımlama sonuçlarıyla değiştirilebilir.
- Görsel ve collision geometrilerinin ayrılması, rendering ayrıntılarının fizik kütlesini değiştirmesini önler.

## Doğrulama

Otomatik testler şunları doğrular:

- Kütle, atalet ve yerçekimi
- Rotor yerleşimi ve isim sırası
- Hover thrust dengesi
- Diferansiyel thrust moment işaretleri
- Yaw moment iptali
- Serbest düşüş ve zemin teması
- Sensör isim ve boyutları

## Sonuçlar ve sınırlılıklar

- Faz 2 motor komutunu doğrudan MJCF `ctrl` alanına yazmayacaktır; önce rotor dinamik katmanından geçirecektir.
- Maksimum thrust, motor zaman sabiti ve thrust katsayısı config'e taşınacaktır.
- Gerçek platform belirlendiğinde bu ADR “superseded” yapılabilir ve ölçülmüş parametrelerle yeni ADR oluşturulabilir.
- Ground effect, blade flapping ve rotor-rotor etkileşimi ilk model kapsamında değildir.

